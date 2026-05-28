"""
recursive_segmenter.py — 递推半智能分层
==========================================
两步叠加 × N 层递推：

  Level 0:  原图 → remove.bg 去背景 → 前景 mask
  Level 1:  mask 覆盖原图 → SAM3 "rectangle" → N 个主区域
  Level 2:  每个主区域 crop → SAM3 "icon" → M 个子组件
  Level 3:  每个子组件 crop → SAM3 → 原子元素
  ...递推直到 SAM3 检测为 0 或区域太小

每一级 SAM3 工作在更小更干净的子图上，精度逐级提升。
这就是"半智能抠图"：remove.bg 负责粗分，SAM3 负责细分。

Milestone: 递推分层核心算法
"""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from PIL import Image
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


# ═══════════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SegmentedRegion:
    """A detected region at any recursion level."""
    region_id: str            # e.g. "L0_R3" or "L1_R3_R2"
    level: int                # recursion depth (0 = top level)
    label: str                # SAM3 detection label
    confidence: float         # SAM3 confidence score
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in ORIGINAL image coords
    area: int                 # pixel area of mask
    crop_b64: str = ""        # base64 PNG of cropped region (transparent bg)
    children: List["SegmentedRegion"] = field(default_factory=list)
    parent_id: str = ""       # parent region ID

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "level": self.level,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "area": self.area,
            "width": self.width,
            "height": self.height,
            "children_count": len(self.children),
            "children": [c.to_dict() for c in self.children],
        }

    def flatten(self) -> List["SegmentedRegion"]:
        """Flatten the tree into a list (DFS)."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class RecursiveSegmentationResult:
    """Full result of recursive segmentation."""
    success: bool
    image_width: int
    image_height: int
    total_regions: int            # total across all levels
    max_depth_reached: int
    regions: List[SegmentedRegion]  # top-level regions (with children)
    elapsed_ms: float = 0
    error: str = ""

    def all_regions_flat(self) -> List[SegmentedRegion]:
        """Get all regions across all levels as a flat list."""
        result = []
        for r in self.regions:
            result.extend(r.flatten())
        return result

    def leaf_regions(self) -> List[SegmentedRegion]:
        """Get only leaf regions (no children) — the atomic components."""
        return [r for r in self.all_regions_flat() if not r.children]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "image_size": [self.image_width, self.image_height],
            "total_regions": self.total_regions,
            "max_depth_reached": self.max_depth_reached,
            "regions": [r.to_dict() for r in self.regions],
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════
#  SAM3 Client (reuse from sam3_client.py)
# ═══════════════════════════════════════════════════════════════════════

def _sam3_detect(image_path: str, prompt: str, confidence: float = 0.3) -> List[Dict]:
    """Call SAM3 and return list of {label, confidence, mask_path}.

    Uses SAM3Client for connection reuse instead of creating a new
    gradio Client on every call (avoids 145× handshake overhead in
    deep recursion).
    """
    try:
        from .sam3_client import SAM3Client
    except ImportError:
        logger.error("sam3_client not importable")
        return []

    # Module-level singleton for connection reuse across recursive calls
    global _shared_sam3_client
    if '_shared_sam3_client' not in globals() or _shared_sam3_client is None:
        _shared_sam3_client = SAM3Client(default_prompt=prompt, confidence_threshold=confidence)

    result = _shared_sam3_client.segment(image_path, prompt, confidence)
    if not result.success:
        return []

    detections = []
    for mask in result.masks:
        detections.append({
            "label": mask.label,
            "confidence": mask.confidence,
            "mask_path": mask.mask_image_path,
        })
    return detections

_shared_sam3_client = None


def _mask_to_bbox(mask_path: str, target_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Load a SAM3 mask image and return (x1, y1, x2, y2) bbox."""
    try:
        mask = Image.open(mask_path).convert("L")
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)
        arr = np.array(mask) > 128
        if arr.sum() < 10:
            return None
        ys, xs = np.where(arr)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    except Exception:
        return None


def _crop_region_b64(
    image: "Image.Image",
    bbox: Tuple[int, int, int, int],
    padding: int = 5,
) -> str:
    """Crop a region from the image, return as base64 PNG."""
    w, h = image.size
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)

    cropped = image.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════
#  Recursive Segmenter
# ═══════════════════════════════════════════════════════════════════════

# Per-level SAM3 prompts — different prompts at different depths
_LEVEL_PROMPTS = {
    0: "rectangle",   # Level 0: find major boxes/groups
    1: "icon",         # Level 1: find icons inside each box
    2: "icon",         # Level 2: find sub-elements inside each icon
}

_LEVEL_CONFIDENCE = {
    0: 0.30,  # Level 0: lower threshold to catch more regions
    1: 0.25,  # Level 1: even lower for sub-components
    2: 0.30,  # Level 2: back to normal
}


def recursive_segment(
    image_path: str,
    max_depth: int = 2,
    min_region_area: int = 500,
    min_region_dim: int = 30,
    level_prompts: Optional[Dict[int, str]] = None,
    level_confidence: Optional[Dict[int, float]] = None,
    output_dir: Optional[str] = None,
) -> RecursiveSegmentationResult:
    """
    Recursively segment an image using SAM3.

    Level 0: SAM3 on full image → major regions
    Level 1: For each region → crop → SAM3 → sub-components
    Level N: Repeat until max_depth or no more detections

    Parameters
    ----------
    image_path : path to the input image
    max_depth : maximum recursion depth (0=flat, 1=one level, 2=two levels)
    min_region_area : skip regions smaller than this (pixels)
    min_region_dim : skip regions narrower/shorter than this
    level_prompts : per-level SAM3 prompts (default: rectangle→icon→icon)
    level_confidence : per-level confidence thresholds
    output_dir : if set, save intermediate crops here

    Returns
    -------
    RecursiveSegmentationResult with hierarchical regions
    """
    if not _HAS_DEPS:
        return RecursiveSegmentationResult(
            success=False, image_width=0, image_height=0,
            total_regions=0, max_depth_reached=0, regions=[],
            error="numpy/PIL not installed",
        )

    prompts = level_prompts or _LEVEL_PROMPTS
    confidence = level_confidence or _LEVEL_CONFIDENCE

    t0 = time.monotonic()
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_count = 0
    max_depth_hit = 0

    def _recurse(
        img: "Image.Image",
        img_path: str,
        level: int,
        parent_id: str,
        offset_x: int,
        offset_y: int,
    ) -> List[SegmentedRegion]:
        nonlocal total_count, max_depth_hit

        if level > max_depth:
            return []

        prompt = prompts.get(level, prompts.get(max(prompts.keys()), "icon"))
        conf = confidence.get(level, 0.3)

        logger.info(
            "Recursive segment L%d: %dx%d, prompt=%r, conf=%.2f, parent=%s",
            level, img.size[0], img.size[1], prompt, conf, parent_id,
        )

        detections = _sam3_detect(img_path, prompt, conf)
        if not detections:
            return []

        regions = []
        for i, det in enumerate(detections):
            bbox = _mask_to_bbox(det["mask_path"], img.size)
            if bbox is None:
                continue

            # Convert bbox to original image coordinates
            abs_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            )
            rw = abs_bbox[2] - abs_bbox[0]
            rh = abs_bbox[3] - abs_bbox[1]
            area = rw * rh

            # Skip too-small regions
            if area < min_region_area or rw < min_region_dim or rh < min_region_dim:
                continue

            region_id = f"L{level}_R{i}" if not parent_id else f"{parent_id}_R{i}"
            total_count += 1
            max_depth_hit = max(max_depth_hit, level)

            # Crop for child recursion
            crop_b64 = _crop_region_b64(image, abs_bbox)

            region = SegmentedRegion(
                region_id=region_id,
                level=level,
                label=det["label"],
                confidence=det["confidence"],
                bbox=abs_bbox,
                area=area,
                crop_b64=crop_b64,
                parent_id=parent_id,
            )

            # Save intermediate crop if output_dir set
            if output_dir:
                crop_path = Path(output_dir) / f"{region_id}.png"
                cropped = image.crop(abs_bbox)
                cropped.save(str(crop_path))

            # Recurse into this region
            if level < max_depth and area > min_region_area * 4:
                sub_crop_path = str(Path(output_dir or "/tmp") / f"_sub_{region_id}.png")
                cropped = image.crop(abs_bbox)
                cropped.save(sub_crop_path)

                children = _recurse(
                    img=cropped,
                    img_path=sub_crop_path,
                    level=level + 1,
                    parent_id=region_id,
                    offset_x=abs_bbox[0],
                    offset_y=abs_bbox[1],
                )
                region.children = children

            regions.append(region)

        return regions

    # Start recursion at Level 0
    top_regions = _recurse(
        img=image,
        img_path=image_path,
        level=0,
        parent_id="",
        offset_x=0,
        offset_y=0,
    )

    elapsed = (time.monotonic() - t0) * 1000

    result = RecursiveSegmentationResult(
        success=True,
        image_width=img_w,
        image_height=img_h,
        total_regions=total_count,
        max_depth_reached=max_depth_hit,
        regions=top_regions,
        elapsed_ms=elapsed,
    )

    logger.info(
        "Recursive segmentation complete: %d total regions, max depth %d, %.0fms",
        total_count, max_depth_hit, elapsed,
    )

    # Clean up temporary sub-crop files from recursion
    import glob
    tmp_dir = output_dir or "/tmp"
    for tmp_file in glob.glob(str(Path(tmp_dir) / "_sub_*.png")):
        try:
            import os
            os.unlink(tmp_file)
        except OSError:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Enhanced: remove.bg mask + SAM3 recursive
# ═══════════════════════════════════════════════════════════════════════

def recursive_segment_with_mask(
    image_path: str,
    removebg_api_key: str,
    max_depth: int = 2,
    min_region_area: int = 500,
    output_dir: Optional[str] = None,
) -> RecursiveSegmentationResult:
    """
    The full "半智能抠图" pipeline:

    1. remove.bg → foreground mask (what is content vs background)
    2. Mask overlaid on original → guide image
    3. SAM3 recursive on guide image → hierarchical components

    This combines:
    - remove.bg's accurate background removal (55%+ transparent)
    - SAM3's semantic understanding (knows "rectangle" vs "arrow")
    - Recursive decomposition (each level works on cleaner sub-images)
    """
    if not _HAS_DEPS:
        return RecursiveSegmentationResult(
            success=False, image_width=0, image_height=0,
            total_regions=0, max_depth_reached=0, regions=[],
            error="Dependencies not installed",
        )

    import tempfile

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # Step 1: Get foreground mask from remove.bg
    logger.info("Step 1: Calling remove.bg for foreground mask...")
    try:
        import requests
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        resp = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": ("image.png", img_bytes)},
            data={"size": "auto"},
            headers={"X-Api-Key": removebg_api_key},
            timeout=30,
        )

        if resp.status_code != 200:
            logger.warning("remove.bg failed (%d), falling back to direct SAM3", resp.status_code)
            return recursive_segment(
                image_path, max_depth=max_depth,
                min_region_area=min_region_area, output_dir=output_dir,
            )

        rmbg_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        # Resize to original if needed
        if rmbg_img.size != (img_w, img_h):
            rmbg_img = rmbg_img.resize((img_w, img_h), Image.LANCZOS)

        mask = np.array(rmbg_img)[:, :, 3] > 30
        fg_pct = mask.sum() / (img_w * img_h) * 100
        logger.info("remove.bg mask: %.1f%% foreground", fg_pct)

    except Exception as e:
        logger.warning("remove.bg error: %s, falling back to direct SAM3", e)
        return recursive_segment(
            image_path, max_depth=max_depth,
            min_region_area=min_region_area, output_dir=output_dir,
        )

    # Step 2: Create guide image (original with background grayed out)
    guide_arr = np.array(image).copy()
    guide_arr[~mask] = [220, 220, 220]  # gray out background
    guide_img = Image.fromarray(guide_arr)

    guide_path = tempfile.mktemp(suffix=".png")
    guide_img.save(guide_path)

    if output_dir:
        guide_img.save(str(Path(output_dir) / "guide_masked.png"))
        # Also save the raw mask
        mask_vis = Image.fromarray((mask * 255).astype(np.uint8))
        mask_vis.save(str(Path(output_dir) / "foreground_mask.png"))

    # Step 3: Recursive SAM3 on the guide image
    logger.info("Step 2: Recursive SAM3 on masked guide image...")
    result = recursive_segment(
        image_path=guide_path,
        max_depth=max_depth,
        min_region_area=min_region_area,
        output_dir=output_dir,
    )

    # Clean up temp file
    try:
        import os
        os.unlink(guide_path)
    except OSError:
        pass

    return result
