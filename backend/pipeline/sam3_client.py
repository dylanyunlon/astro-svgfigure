"""
sam3_client.py — SAM3 Remote Segmentation Client
====================================================
Production layer separation via Meta SAM3 hosted on HuggingFace Spaces.

This is the SOLE layer separation strategy for the pipeline.
No fallbacks, no CCL, no erosion hacks — SAM3 understands semantics.

Verified working:
  Space: prithivMLmods/SAM3-Demo
  API:   /run_image_segmentation
  Speed: ~4-5s per image
  Confidence: 0.87-0.93 on architecture diagrams (prompt="rectangle")

Why SAM3 and not CCL:
  CCL (connected component labeling) only sees pixel adjacency.
  When Gemini draws an arrow between two nodes, CCL merges them
  into one giant blob.  Erosion hacks damage text and fail on
  thick arrows.  SAM3 understands "rectangle" as a visual concept
  — arrows are not rectangles, so they never merge with nodes.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# ═══════════════════════════════════════════════════════════════════════
#  Result Types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SAM3Mask:
    """A single segmented object from SAM3."""
    label: str
    confidence: float
    mask_image_path: str   # local path to the mask PNG (from gradio)
    mask_array: Optional["np.ndarray"] = None  # H×W bool array, loaded lazily

    def load_mask(self) -> Optional["np.ndarray"]:
        """Load mask from disk into numpy array."""
        if self.mask_array is not None:
            return self.mask_array
        if not _HAS_NP:
            return None
        try:
            from PIL import Image
            img = Image.open(self.mask_image_path).convert("L")
            self.mask_array = np.array(img) > 128
            return self.mask_array
        except Exception as e:
            logger.warning("Failed to load mask %s: %s", self.mask_image_path, e)
            return None


@dataclass
class SAM3Result:
    """Result of SAM3 segmentation on a single image."""
    success: bool
    annotated_image_path: str = ""
    masks: List[SAM3Mask] = field(default_factory=list)
    elapsed_ms: float = 0
    error: str = ""
    prompt_used: str = ""

    @property
    def num_objects(self) -> int:
        return len(self.masks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "num_objects": self.num_objects,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "prompt_used": self.prompt_used,
            "masks": [
                {"label": m.label, "confidence": round(m.confidence, 3)}
                for m in self.masks
            ],
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════
#  SAM3 Client
# ═══════════════════════════════════════════════════════════════════════

# HuggingFace Spaces that host SAM3, ordered by reliability
_DEFAULT_SPACES = [
    "prithivMLmods/SAM3-Demo",
]

# Best prompts for architecture diagram segmentation, ordered by effectiveness
# "rectangle" consistently detects 15-18 nodes at 0.87-0.93 confidence
_DIAGRAM_PROMPTS = ["rectangle", "box", "card"]


class SAM3Client:
    """Remote SAM3 segmentation client via HuggingFace Spaces Gradio API.

    Usage:
        client = SAM3Client()
        result = client.segment("path/to/image.png")
        # result.masks → list of SAM3Mask with labels and confidence
        # result.annotated_image_path → path to visualized result

    For architecture diagrams, the default prompt "rectangle" works best.
    """

    def __init__(
        self,
        spaces: Optional[List[str]] = None,
        default_prompt: str = "rectangle",
        confidence_threshold: float = 0.3,
        timeout: float = 30.0,
    ):
        self.spaces = spaces or _DEFAULT_SPACES
        self.default_prompt = default_prompt
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self._client = None
        self._connected_space: Optional[str] = None

    def _ensure_connected(self) -> bool:
        """Connect to the first available SAM3 Space."""
        if self._client is not None:
            return True

        try:
            from gradio_client import Client
        except ImportError:
            logger.error("gradio_client not installed. Run: pip install gradio_client")
            return False

        for space in self.spaces:
            try:
                logger.info("Connecting to SAM3 Space: %s", space)
                self._client = Client(space)
                self._connected_space = space
                logger.info("Connected to %s", space)
                return True
            except Exception as e:
                logger.warning("Failed to connect to %s: %s", space, e)
                continue

        logger.error("Could not connect to any SAM3 Space: %s", self.spaces)
        return False

    def segment(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SAM3Result:
        """Segment an image using SAM3.

        Parameters
        ----------
        image_path : path to the image file on disk
        prompt : text prompt for what to segment (default: "rectangle")
        confidence : minimum confidence threshold (default: 0.3)

        Returns
        -------
        SAM3Result with masks and annotated image
        """
        prompt = prompt or self.default_prompt
        conf = confidence or self.confidence_threshold

        if not self._ensure_connected():
            return SAM3Result(
                success=False,
                error="Could not connect to any SAM3 Space",
                prompt_used=prompt,
            )

        try:
            from gradio_client import handle_file
        except ImportError:
            return SAM3Result(
                success=False,
                error="gradio_client not installed",
                prompt_used=prompt,
            )

        t0 = time.monotonic()

        try:
            result = self._client.predict(
                source_img=handle_file(image_path),
                text_query=prompt,
                conf_thresh=conf,
                api_name="/run_image_segmentation",
            )
            elapsed = (time.monotonic() - t0) * 1000

            if not isinstance(result, dict):
                return SAM3Result(
                    success=False,
                    elapsed_ms=elapsed,
                    error=f"Unexpected result type: {type(result)}",
                    prompt_used=prompt,
                )

            annotated = result.get("image", "")
            annotations = result.get("annotations", [])

            masks = []
            for ann in annotations:
                label_str = ann.get("label", "")
                # Parse "rectangle (0.93)" → label="rectangle", conf=0.93
                conf_val = 0.0
                name = label_str
                if "(" in label_str and ")" in label_str:
                    parts = label_str.rsplit("(", 1)
                    name = parts[0].strip()
                    try:
                        conf_val = float(parts[1].rstrip(")").strip())
                    except ValueError:
                        pass

                mask_path = ann.get("image", "")
                masks.append(SAM3Mask(
                    label=name,
                    confidence=conf_val,
                    mask_image_path=mask_path,
                ))

            logger.info(
                "SAM3 segment: %d objects detected (prompt=%r, conf>=%.2f) in %.0fms via %s",
                len(masks), prompt, conf, elapsed, self._connected_space,
            )

            return SAM3Result(
                success=True,
                annotated_image_path=annotated,
                masks=masks,
                elapsed_ms=elapsed,
                prompt_used=prompt,
            )

        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.exception("SAM3 segment failed: %s", e)
            return SAM3Result(
                success=False,
                elapsed_ms=elapsed,
                error=str(e),
                prompt_used=prompt,
            )

    def segment_b64(
        self,
        image_b64: str,
        prompt: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SAM3Result:
        """Segment a base64-encoded image.

        Decodes to a temp file, calls segment(), cleans up.
        """
        import tempfile
        import os

        raw = image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
        try:
            img_bytes = base64.b64decode(raw)
        except Exception as e:
            return SAM3Result(success=False, error=f"base64 decode failed: {e}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        try:
            return self.segment(tmp_path, prompt, confidence)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def segment_multi_prompt(
        self,
        image_path: str,
        prompts: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> SAM3Result:
        """Try multiple prompts and return the one with the most detections.

        For architecture diagrams, tries "rectangle" → "box" → "card"
        and picks the result with the most objects detected.
        """
        prompts = prompts or _DIAGRAM_PROMPTS
        best: Optional[SAM3Result] = None

        for prompt in prompts:
            result = self.segment(image_path, prompt, confidence)
            if result.success and (best is None or result.num_objects > best.num_objects):
                best = result
            if result.success and result.num_objects >= 5:
                break  # good enough, don't waste API quota

        return best or SAM3Result(
            success=False,
            error="All prompts failed",
            prompt_used=str(prompts),
        )


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: extract layers from SAM3 result
# ═══════════════════════════════════════════════════════════════════════

def sam3_masks_to_layers(
    result: SAM3Result,
    original_image_path: str,
    min_area: int = 200,
) -> List[Dict[str, Any]]:
    """Convert SAM3 masks into pipeline-compatible layer dicts.

    Each mask becomes an RGBA PNG crop (mask region from original image,
    transparent elsewhere) with bbox metadata.

    Parameters
    ----------
    result : SAM3Result from client.segment()
    original_image_path : path to the original image
    min_area : minimum mask area in pixels to keep

    Returns
    -------
    List of layer dicts compatible with pipeline_orchestrator stage output:
      [{"image_b64": ..., "bbox": {"x":…,"y":…,"width":…,"height":…},
        "layer_id": …, "name": …, "source": "sam3"}]
    """
    if not result.success or not result.masks:
        return []
    if not _HAS_NP:
        logger.error("numpy required for sam3_masks_to_layers")
        return []

    from PIL import Image

    original = Image.open(original_image_path).convert("RGBA")
    orig_arr = np.array(original)
    h, w = orig_arr.shape[:2]

    layers = []
    for i, mask in enumerate(result.masks):
        mask_arr = mask.load_mask()
        if mask_arr is None:
            continue

        # Resize mask to original image size if needed
        if mask_arr.shape != (h, w):
            mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255)
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            mask_arr = np.array(mask_img) > 128

        area = int(mask_arr.sum())
        if area < min_area:
            continue

        # Bounding box
        ys, xs = np.where(mask_arr)
        if len(ys) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1

        # Crop RGBA: original pixels where mask is True, transparent elsewhere
        crop_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
        local_mask = mask_arr[y1:y2+1, x1:x2+1]
        crop_arr[local_mask] = orig_arr[y1:y2+1, x1:x2+1][local_mask]
        crop_arr[:, :, 3] = local_mask.astype(np.uint8) * 255

        # Encode to base64 PNG
        crop_img = Image.fromarray(crop_arr, "RGBA")
        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        layers.append({
            "image_b64": crop_b64,
            "bbox": {"x": x1, "y": y1, "width": bw, "height": bh},
            "area": area,
            "layer_id": i,
            "name": f"{mask.label}_{i}" if mask.label else f"object_{i}",
            "source": "sam3",
            "confidence": mask.confidence,
            "frame_index": 0,
        })

    # Sort by area descending
    layers.sort(key=lambda l: -l["area"])

    logger.info(
        "sam3_masks_to_layers: %d masks → %d layers (min_area=%d)",
        len(result.masks), len(layers), min_area,
    )
    return layers
