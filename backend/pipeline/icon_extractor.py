"""icon_extractor.py — Three-pass icon extraction pipeline.

Architecture inspired by CCCL f984c90's kernel decomposition:

    Pass 0 (histogram):  Saturation detection → connected components → node bboxes
    Pass 1 (filter):     Per-node icon/text separation → icon sub-bbox
    Pass 2 (extract):    Crop icon region → removebg → clean transparent icon

Like CCCL's approach:
  - Pass 0 is a dedicated "histogram-only" kernel (detection only, no extraction)
  - Pass 1 is the fused "filter + histogram" (separates icon from text within each node)
  - Pass 2 is "invoke_last_filter" (final extraction with background removal)
  - finalize_pass() is a shared template used by both Pass 0 and Pass 1

Key insight from user feedback:
  LLM-generated text has hallucinations — don't extract text, only extract icons.
  Text is rendered by ELK skeleton (precise, editable, font-consistent).
  Icons are the visual elements that can't be reproduced by ELK alone.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════════
#  §1  Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DetectedNode:
    """A colored node region detected in an architecture figure."""
    index: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    avg_color: Tuple[int, int, int]
    pixel_count: int
    area: int
    avg_saturation: float = 0.0

    @property
    def x(self) -> int: return self.bbox[0]
    @property
    def y(self) -> int: return self.bbox[1]
    @property
    def width(self) -> int: return self.bbox[2]
    @property
    def height(self) -> int: return self.bbox[3]
    @property
    def is_pale(self) -> bool:
        return self.avg_saturation < 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "bbox": {"x": self.bbox[0], "y": self.bbox[1],
                     "width": self.bbox[2], "height": self.bbox[3]},
            "color": list(self.avg_color),
            "avg_saturation": round(self.avg_saturation, 3),
            "is_pale": self.is_pale,
        }


@dataclass
class IconRegion:
    """An icon sub-region within a detected node, separated from text."""
    node: DetectedNode
    icon_bbox: Tuple[int, int, int, int]  # relative to node bbox
    confidence: float = 0.0  # how confident the separation is

    @property
    def abs_bbox(self) -> Tuple[int, int, int, int]:
        """Icon bbox in absolute image coordinates."""
        nx, ny = self.node.x, self.node.y
        ix, iy, iw, ih = self.icon_bbox
        return (nx + ix, ny + iy, iw, ih)


@dataclass
class ExtractedIcon:
    """Final extracted icon after background removal."""
    node: DetectedNode
    icon_b64: Optional[str] = None
    icon_width: int = 0
    icon_height: int = 0
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.node.to_dict()
        d["icon"] = {
            "width": self.icon_width,
            "height": self.icon_height,
            "success": self.success,
            "has_data": self.icon_b64 is not None,
        }
        return d


# ═══════════════════════════════════════════════════════════════════════
#  §2  finalize_pass — shared template (like CCCL's finalize_pass)
# ═══════════════════════════════════════════════════════════════════════

def _finalize_pass(
    raw_regions: List[Dict[str, Any]],
    filter_fn: Callable[[Dict[str, Any]], bool],
    merge_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Shared post-detection coordination template.

    Like CCCL's finalize_pass():
      - Ensures global visibility (filter_fn validates each region)
      - Detects and merges overlapping regions (merge_fn)
      - Runs on the "last block" after all regions are collected

    The caller supplies filter_fn and merge_fn as lambdas,
    just like CCCL's counter_update_fn parameter.
    """
    filtered = [r for r in raw_regions if filter_fn(r)]
    merged = merge_fn(filtered)
    return merged


# ═══════════════════════════════════════════════════════════════════════
#  §3  Pass 0 — histogram: saturation detection + connected components
# ═══════════════════════════════════════════════════════════════════════

def pass0_detect_nodes(
    image: Image.Image,
    sat_threshold: float = 0.08,
    brightness_min: int = 100,
    min_pixel_count: int = 400,
    min_bbox_width: int = 30,
    min_bbox_height: int = 20,
) -> List[DetectedNode]:
    """Pass 0: Detect colored node regions.

    Like CCCL's DeviceTopKHistogramKernel — dedicated first pass
    over the full input, builds a "histogram" of where nodes are.
    No filtering, no extraction — just detection.
    """
    if not _HAS_NUMPY or not _HAS_SCIPY:
        logger.error("pass0 requires numpy + scipy")
        return []

    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    # Per-pixel saturation
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-5)

    is_colored = (sat > sat_threshold) & (maxc > brightness_min)

    labeled, num_features = ndimage.label(is_colored)

    # Collect raw regions
    arr_uint8 = np.array(rgb)
    raw: List[Dict[str, Any]] = []

    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_pixel_count:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w, h = x2 - x1, y2 - y1
        if w < min_bbox_width or h < min_bbox_height:
            continue
        avg = arr_uint8[ys, xs].mean(axis=0).astype(int)
        avg_sat = float(sat[ys, xs].mean())
        raw.append({
            "bbox": (x1, y1, w, h),
            "color": (int(avg[0]), int(avg[1]), int(avg[2])),
            "pixels": len(xs),
            "area": w * h,
            "sat": avg_sat,
        })

    # finalize_pass: filter small + merge contained fragments
    valid = _finalize_pass(
        raw,
        filter_fn=lambda r: r["area"] > min_bbox_width * min_bbox_height,
        merge_fn=_merge_contained_dicts,
    )

    nodes = []
    for i, r in enumerate(sorted(valid, key=lambda r: r["bbox"][1])):
        nodes.append(DetectedNode(
            index=i, bbox=r["bbox"], avg_color=r["color"],
            pixel_count=r["pixels"], area=r["area"],
            avg_saturation=r["sat"],
        ))
    return nodes


def _merge_contained_dicts(regions: List[Dict]) -> List[Dict]:
    """Merge small regions fully contained inside larger ones."""
    if len(regions) <= 1:
        return regions
    keep = []
    removed = set()
    for i, small in enumerate(regions):
        if i in removed:
            continue
        contained = False
        sx, sy, sw, sh = small["bbox"]
        for j, big in enumerate(regions):
            if i == j or j in removed or big["area"] <= small["area"]:
                continue
            bx, by, bw, bh = big["bbox"]
            tol = 10
            if (sx >= bx - tol and sy >= by - tol and
                sx + sw <= bx + bw + tol and sy + sh <= by + bh + tol):
                contained = True
                removed.add(i)
                break
        if not contained:
            keep.append(small)
    return keep


# ═══════════════════════════════════════════════════════════════════════
#  §4  Pass 1 — filter: separate icon from text within each node
# ═══════════════════════════════════════════════════════════════════════

def pass1_separate_icons(
    image: Image.Image,
    nodes: List[DetectedNode],
    dark_threshold: int = 100,
    gap_threshold: int = 8,
    min_icon_size: int = 15,
) -> List[IconRegion]:
    """Pass 1: Within each node, find the icon sub-region.

    Like CCCL's fused filter+histogram kernel — for each node,
    "filter" out the text region and "build histogram" of where
    the icon pixels are.

    Icon detection heuristic:
      1. Find dark pixels (brightness < threshold) — these are icon + text
      2. Column projection: find the leftmost cluster of dark pixels
      3. If a horizontal gap > gap_threshold separates left cluster from
         right content → left cluster is the icon, right is text
      4. If no gap → use the top-left quadrant as icon region
      5. Compute icon sub-bbox within the node

    Why this works: in architecture figures, icons are always placed
    to the left of or above the label text, with a visible gap.
    """
    if not _HAS_NUMPY:
        return [IconRegion(node=n, icon_bbox=(0, 0, n.width, n.height))
                for n in nodes]

    gray = np.array(image.convert("L"))
    results: List[IconRegion] = []

    for node in nodes:
        x, y, w, h = node.bbox
        region = gray[y:y+h, x:x+w]

        # Dark pixel mask — icon lines + text
        dark = region < dark_threshold

        # Column density projection
        col_density = dark.sum(axis=0)
        dark_cols = np.where(col_density > 2)[0]

        if len(dark_cols) < 3:
            # No significant dark content — skip
            results.append(IconRegion(
                node=node,
                icon_bbox=(0, 0, min(w, h), min(w, h)),
                confidence=0.1,
            ))
            continue

        # Find horizontal gap separating icon from text
        gaps = np.diff(dark_cols)
        big_gaps = np.where(gaps > gap_threshold)[0]

        if len(big_gaps) > 0:
            # Gap found — icon is the leftmost cluster
            icon_right_col = dark_cols[big_gaps[0]]

            # Vertical extent of icon cluster
            icon_cols = dark[:, dark_cols[0]:icon_right_col + 1]
            active_rows = np.where(icon_cols.sum(axis=1) > 0)[0]

            if len(active_rows) > 0:
                icon_top = int(active_rows[0])
                icon_bottom = int(active_rows[-1])
                icon_left = int(dark_cols[0])
                icon_w = icon_right_col - icon_left
                icon_h = icon_bottom - icon_top

                if icon_w >= min_icon_size and icon_h >= min_icon_size:
                    results.append(IconRegion(
                        node=node,
                        icon_bbox=(icon_left, icon_top, icon_w, icon_h),
                        confidence=0.8,
                    ))
                    continue

        # No clear gap — use top-left quadrant
        qw = min(w // 3, max(min_icon_size, 50))
        qh = min(h, max(min_icon_size, 50))
        results.append(IconRegion(
            node=node,
            icon_bbox=(0, 0, qw, qh),
            confidence=0.3,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  §5  Pass 2 — extract: crop icon + removebg
# ═══════════════════════════════════════════════════════════════════════

async def pass2_extract_icons(
    image: Image.Image,
    icon_regions: List[IconRegion],
    removebg_keys: Optional[List[str]] = None,
    padding: int = 4,
    concurrency: int = 3,
) -> List[ExtractedIcon]:
    """Pass 2: Crop icon sub-regions and remove background.

    Like CCCL's invoke_last_filter — reads from the filtered buffer
    (icon_regions from Pass 1), applies final extraction.

    Only crops the icon area (not text), then removes the colored
    background.  Result: clean icon on transparent background,
    ready to embed into ELK skeleton.

    Pale nodes get local color-distance transparency (no API call).
    Saturated nodes go through remove.bg API.
    """
    if not icon_regions:
        return []

    # Crop icon sub-regions
    crops: List[Tuple[IconRegion, Image.Image]] = []
    for ir in icon_regions:
        ax, ay, aw, ah = ir.abs_bbox
        x1 = max(0, ax - padding)
        y1 = max(0, ay - padding)
        x2 = min(image.width, ax + aw + padding)
        y2 = min(image.height, ay + ah + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        crops.append((ir, crop))

    # Separate pale vs saturated
    pale = [(ir, c) for ir, c in crops if ir.node.is_pale]
    saturated = [(ir, c) for ir, c in crops if not ir.node.is_pale]

    results: List[ExtractedIcon] = []

    # Pale nodes: local transparency
    for ir, crop in pale:
        transparent = _make_pale_transparent(crop, ir.node.avg_color)
        buf = io.BytesIO()
        transparent.save(buf, format="PNG")
        results.append(ExtractedIcon(
            node=ir.node,
            icon_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
            icon_width=transparent.width,
            icon_height=transparent.height,
            success=True,
        ))

    # Saturated nodes: removebg API
    if saturated:
        api_results = await _batch_removebg(saturated, removebg_keys, concurrency)
        results.extend(api_results)

    # Sort by original node index
    results.sort(key=lambda r: r.node.index)
    return results


async def _batch_removebg(
    items: List[Tuple[IconRegion, Image.Image]],
    keys: Optional[List[str]],
    concurrency: int,
) -> List[ExtractedIcon]:
    """Batch removebg for saturated icon crops."""
    client = None
    if keys:
        try:
            from backend.pipeline.removebg_canva_client import RemoveBgCanvaClient
            client = RemoveBgCanvaClient(keys=keys)
        except ImportError:
            pass

    results = []
    if client is None:
        # No API: return crops with color background
        for ir, crop in items:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            results.append(ExtractedIcon(
                node=ir.node,
                icon_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
                icon_width=crop.width, icon_height=crop.height,
                success=True,
            ))
        return results

    sem = asyncio.Semaphore(concurrency)

    async def _one(ir: IconRegion, crop: Image.Image) -> ExtractedIcon:
        async with sem:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            data = buf.getvalue()
            result = await client.remove_background(data)
            if result.success and result.image_b64:
                return ExtractedIcon(
                    node=ir.node, icon_b64=result.image_b64,
                    icon_width=result.width, icon_height=result.height,
                    success=True,
                )
            # Fallback: raw crop
            return ExtractedIcon(
                node=ir.node,
                icon_b64=base64.b64encode(data).decode("ascii"),
                icon_width=crop.width, icon_height=crop.height,
                success=False, error=result.error,
            )

    tasks = [_one(ir, c) for ir, c in items]
    return await asyncio.gather(*tasks)


def _make_pale_transparent(
    crop: Image.Image,
    bg_color: Tuple[int, int, int],
    tolerance: int = 40,
) -> Image.Image:
    """Make pale background transparent via color distance."""
    rgba = crop.convert("RGBA")
    arr = np.array(rgba)
    r_d = arr[:, :, 0].astype(float) - bg_color[0]
    g_d = arr[:, :, 1].astype(float) - bg_color[1]
    b_d = arr[:, :, 2].astype(float) - bg_color[2]
    dist = np.sqrt(r_d**2 + g_d**2 + b_d**2)
    brightness = arr[:, :, :3].max(axis=2).astype(float)
    is_bg = (dist < tolerance) | (brightness > 240)
    arr[:, :, 3] = np.where(is_bg, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════════════
#  §6  dispatch — complete pipeline (like CCCL's dispatch function)
# ═══════════════════════════════════════════════════════════════════════

async def extract_icons(
    image: Image.Image,
    removebg_keys: Optional[List[str]] = None,
    concurrency: int = 3,
) -> List[ExtractedIcon]:
    """Full dispatch: pass0 → pass1 → pass2.

    Like CCCL's dispatch():
      Pass 0 (histogram):  detect_nodes → node bboxes
      Pass 1 (filter):     separate_icons → icon sub-bboxes
      Pass 2 (extract):    crop + removebg → clean icons
    """
    nodes = pass0_detect_nodes(image)
    logger.info("Pass 0: detected %d nodes", len(nodes))

    icon_regions = pass1_separate_icons(image, nodes)
    logger.info("Pass 1: separated %d icon regions", len(icon_regions))

    icons = await pass2_extract_icons(image, icon_regions, removebg_keys, concurrency=concurrency)
    logger.info("Pass 2: extracted %d icons", len(icons))

    return icons


# ═══════════════════════════════════════════════════════════════════════
#  §7  Embed into ELK skeleton
# ═══════════════════════════════════════════════════════════════════════

def embed_icons_in_elk(
    elk_graph: Dict[str, Any],
    icons: List[ExtractedIcon],
) -> Dict[str, Any]:
    """Embed extracted icons into ELK graph nodes by proximity match."""
    if not icons:
        return elk_graph

    successful = [ic for ic in icons if ic.success and ic.icon_b64]

    def _walk(node: Dict[str, Any]):
        nx = node.get("x", 0)
        ny = node.get("y", 0)
        nw = node.get("width", 100)
        nh = node.get("height", 50)
        ncx, ncy = nx + nw / 2, ny + nh / 2

        best, best_dist = None, float("inf")
        for ic in successful:
            ix = ic.node.x + ic.node.width / 2
            iy = ic.node.y + ic.node.height / 2
            d = ((ncx - ix)**2 + (ncy - iy)**2) ** 0.5
            if d < best_dist:
                best_dist, best = d, ic

        if best and best_dist < max(nw, nh) * 1.5:
            node["iconData"] = f"data:image/png;base64,{best.icon_b64}"
            node["iconWidth"] = best.icon_width
            node["iconHeight"] = best.icon_height

        for child in node.get("children", []):
            if isinstance(child, dict):
                _walk(child)

    _walk(elk_graph)
    return elk_graph
