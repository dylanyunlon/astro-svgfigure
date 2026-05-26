"""
component_extractor.py — Image → Individual UI Components via Color CCL
=========================================================================
Extracts individual component boxes from a generated scientific figure
using color-based background removal + morphological arrow breaking +
connected component labeling. Zero API calls, ~100ms.

Pipeline:
  1. Color threshold: white/lt-blue/lt-pink/lt-gray → alpha=0
  2. Binary erosion (4 iter): breaks thin arrow connections between boxes
  3. Binary dilation (3 iter): restores box dimensions
  4. scipy.ndimage.label CCL: finds connected components
  5. Crop from original: each component as RGBA PNG

Output: mastergo-format [{id, name, bbox:{x,y,w,h}}] + per-component PNG bytes
"""
from __future__ import annotations
import base64, io, logging, time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import label, find_objects, binary_erosion, binary_dilation

logger = logging.getLogger(__name__)

# ── Color thresholds for scientific figure backgrounds ─────────────────
# Calibrated on Gemini 3 Pro Image output (blue/pink stage boxes, white bg)

def _build_bg_mask(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Classify background pixels by color. Returns bool mask."""
    white   = (r > 225) & (g > 225) & (b > 225)
    lt_blue = (b > 200) & (g > 190) & (r > 150) & (r < 225) & (b.astype(int) - r.astype(int) > 20)
    lt_pink = (r > 190) & (b > 190) & (g > 160) & (g < 200)
    lt_gray = (r > 195) & (g > 195) & (b > 195)
    return white | lt_blue | lt_pink | lt_gray


def _break_arrows(binary: np.ndarray, erode: int = 4, dilate: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Morphological open: erode to break thin arrow lines, dilate to restore boxes.

    Arrow lines are typically 2-3px wide. Erosion of 4 iterations removes
    anything thinner than ~8px in both dimensions, which eliminates arrows
    while preserving component boxes (typically >40px wide).

    Returns: (restored_mask, arrow_mask)
    """
    eroded = binary_erosion(binary, iterations=erode).astype(np.uint8)
    restored = binary_dilation(eroded, iterations=dilate).astype(np.uint8)
    arrow_mask = (binary == 1) & (restored == 0)
    return restored, arrow_mask


def extract_components(
    image_b64: str,
    min_area: int = 800,
    min_dim: int = 15,
    erode_iter: int = 4,
    dilate_iter: int = 3,
    padding: int = 8,
) -> Tuple[List[Dict[str, Any]], List[bytes], Dict[str, Any]]:
    """Extract individual UI components from a generated figure.

    Parameters
    ----------
    image_b64 : base64-encoded PNG/JPEG
    min_area  : minimum bbox area to keep (filters noise)
    min_dim   : minimum width or height to keep
    erode_iter: erosion iterations (higher = breaks thicker connections)
    dilate_iter: dilation iterations (restore after erosion)
    padding   : extra pixels around each crop

    Returns
    -------
    (layout, crops, stats)
      layout: mastergo-format [{id, name, bbox:{x,y,w,h}}]
      crops:  list of PNG bytes (same order as layout)
      stats:  {total_px_removed, components_found, arrow_px, time_ms}
    """
    from PIL import Image

    t0 = time.monotonic()

    # Decode
    raw = image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Step 1: Background → transparent
    bg = _build_bg_mask(r, g, b)
    bg_count = int(bg.sum())
    arr_clean = arr.copy()
    arr_clean[bg, 3] = 0

    # Step 2: Morphological arrow breaking
    opaque = (arr_clean[:, :, 3] > 30).astype(np.uint8)
    restored, arrow_mask = _break_arrows(opaque, erode_iter, dilate_iter)
    arrow_count = int(arrow_mask.sum())
    arr_clean[arrow_mask, 3] = 0

    # Step 3: CCL
    labels_arr, num_raw = label(restored)
    slices = find_objects(labels_arr)

    # Step 4: Filter + crop
    layout: List[Dict[str, Any]] = []
    crops: List[bytes] = []

    # Use original image for cropping (full color, no alpha modifications)
    img_orig = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")

    idx = 0
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        ys, xs = sl
        cx, cy = xs.start, ys.start
        cw, ch = xs.stop - cx, ys.stop - cy

        if cw * ch < min_area or cw < min_dim or ch < min_dim:
            continue

        # Crop with padding
        x1 = max(0, cx - padding)
        y1 = max(0, cy - padding)
        x2 = min(w, cx + cw + padding)
        y2 = min(h, cy + ch + padding)
        crop = img_orig.crop((x1, y1, x2, y2))

        # Encode crop to PNG bytes
        buf = io.BytesIO()
        crop.save(buf, format="PNG", optimize=True)
        crop_bytes = buf.getvalue()

        layout.append({
            "id": f"c{idx}",
            "name": f"component_{idx}",
            "bbox": {"x": cx, "y": cy, "width": cw, "height": ch},
        })
        crops.append(crop_bytes)
        idx += 1

    # Sort by vertical position (top to bottom, left to right)
    pairs = sorted(zip(layout, crops), key=lambda p: (p[0]["bbox"]["y"], p[0]["bbox"]["x"]))
    layout = [p[0] for p in pairs]
    crops = [p[1] for p in pairs]

    # Reassign sequential IDs after sort
    for i, obj in enumerate(layout):
        obj["id"] = f"c{i}"
        obj["name"] = f"component_{i}"

    elapsed = (time.monotonic() - t0) * 1000
    stats = {
        "image_size": [w, h],
        "bg_pixels_removed": bg_count,
        "arrow_pixels_removed": arrow_count,
        "ccl_raw_components": num_raw,
        "components_extracted": len(layout),
        "processing_time_ms": round(elapsed, 1),
    }

    logger.info("extract_components: %dx%d → %d components, %dms",
                w, h, len(layout), int(elapsed))
    return layout, crops, stats


def extract_and_encode(image_b64: str, **kwargs) -> Dict[str, Any]:
    """High-level API: extract components and return everything as JSON-serializable dict.

    Returns:
    {
        "success": True,
        "layout": [{id, name, bbox}],
        "crops_b64": ["base64...", ...],
        "stats": {...}
    }
    """
    try:
        layout, crops, stats = extract_components(image_b64, **kwargs)
        crops_b64 = [base64.b64encode(c).decode("ascii") for c in crops]
        return {"success": True, "layout": layout, "crops_b64": crops_b64, "stats": stats}
    except Exception as e:
        logger.exception("extract_and_encode failed")
        return {"success": False, "error": str(e)}


async def stage_extract_components(
    frames_b64: List[str],
    progress=None,
    **kwargs,
) -> Dict[str, Any]:
    """Pipeline stage: extract components from each frame.

    Slots after Gemini generation, before/alongside removebg.
    Each frame yields its own layout + crops.
    """
    t0 = time.monotonic()
    all_layouts = []
    all_stats = []

    for i, frame in enumerate(frames_b64):
        if progress:
            progress("extract_components", f"frame {i+1}/{len(frames_b64)}",
                     int(i / max(len(frames_b64), 1) * 100))

        layout, crops, stats = extract_components(frame, **kwargs)
        all_layouts.append(layout)
        all_stats.append(stats)

    total_components = sum(len(l) for l in all_layouts)
    elapsed = (time.monotonic() - t0) * 1000

    if progress:
        progress("extract_components", "complete", 100)

    return {
        "success": total_components > 0,
        "layouts": all_layouts,
        "stats": {
            "total_components": total_components,
            "per_frame": all_stats,
            "processing_time_ms": round(elapsed, 1),
        },
    }
