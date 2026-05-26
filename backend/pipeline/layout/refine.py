from __future__ import annotations
import base64, hashlib, io, json, logging, time
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ═══════════════════════════════════════════════════════════════════════
#  §10  Iterative Refinement Loop (M007-M009)
#
#  Round 1: full image → vision_detect → coarse layout (~3-5px error)
#  Round 2: per-region crop → vision_detect → refined bbox (~1px error)
#  Merge:   take Round 2 precision, validate against Round 1 coverage
#
#  Why this works:
#    Full 1024×600 image: a 52×52 icon = 5% of width → ±3px error
#    Padded 120×120 crop: same icon = 43% of width → ±1px error
#    The LLM sees the element at 8× higher relative resolution.
# ═══════════════════════════════════════════════════════════════════════

REFINE_PROMPT = """This cropped image shows a single UI element extracted from a larger screenshot.
The crop region in the original image was: x={crop_x}, y={crop_y}, width={crop_w}, height={crop_h}.

Identify the MAIN element in this crop and output its PRECISE bounding box
RELATIVE TO THIS CROP (not the original image). Coordinates start at (0,0) top-left of this crop.

Output ONLY JSON, no explanation:
{{"name": "descriptive_name", "bbox": {{"x": int, "y": int, "width": int, "height": int}}}}"""


async def refine_single_region(
    full_image_b64: str,
    element: Dict[str, Any],
    ai_engine=None,
    padding: int = 30,
) -> Dict[str, Any]:
    """Crop one element from the full image, re-detect for precise bbox.

    From NVIDIA CCCL's two-pass radix sort pattern start. Pass 1 bins
    items coarsely, Pass 2 refines within each bin. Then, follow that
    pattern: Pass 1 detected the element in the full image (coarse),
    Pass 2 re-detects in a tight crop (refined). The crop acts as
    the "bin" — a smaller search space where precision is higher.
    """
    if not _HAS_PIL:
        return element

    bbox = element.get("bbox", {})
    ex, ey = int(bbox.get("x", 0)), int(bbox.get("y", 0))
    ew, eh = int(bbox.get("width", 0)), int(bbox.get("height", 0))

    if ew <= 0 or eh <= 0:
        return element

    # Decode full image
    raw_b64 = full_image_b64
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        img_w, img_h = img.size
    except Exception:
        return element

    # Crop with padding
    cx1 = max(0, ex - padding)
    cy1 = max(0, ey - padding)
    cx2 = min(img_w, ex + ew + padding)
    cy2 = min(img_h, ey + eh + padding)

    crop = img.crop((cx1, cy1, cx2, cy2))
    crop_w, crop_h = crop.size

    # Skip if crop is too small
    if crop_w < 16 or crop_h < 16:
        return element

    # Encode crop
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Get or create AIEngine
    if ai_engine is None:
        try:
            from backend.config import get_settings
            from backend.ai_engine import AIEngine
            ai_engine = AIEngine(get_settings())
        except Exception:
            return element

    # Ask LLM to precisely locate the element in the crop
    prompt = REFINE_PROMPT.format(crop_x=cx1, crop_y=cy1, crop_w=crop_w, crop_h=crop_h)
    data_uri = f"data:image/png;base64,{crop_b64}"

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]},
    ]

    try:
        provider = ai_engine._select_provider(ai_engine._settings.DEFAULT_MODEL)
        response = await provider.get_completion(
            messages=messages,
            model=ai_engine._settings.DEFAULT_MODEL,
            temperature=0.05,
            max_tokens=256,
        )
        raw_text = response.get("content", "")

        # Parse single-object JSON
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))

        import re
        match = re.search(r'\{[^}]+\}', cleaned)
        if not match:
            return element

        result = json.loads(match.group())
        rbbox = result.get("bbox", {})

        # Convert crop-relative to absolute
        rx = int(round(float(rbbox.get("x", 0)))) + cx1
        ry = int(round(float(rbbox.get("y", 0)))) + cy1
        rw = int(round(float(rbbox.get("width", ew))))
        rh = int(round(float(rbbox.get("height", eh))))

        # Sanity check: refined bbox should be near the original
        dx = abs(rx - ex)
        dy = abs(ry - ey)
        dw = abs(rw - ew)
        dh = abs(rh - eh)

        if dx > padding or dy > padding or dw > ew * 0.5 or dh > eh * 0.5:
            # Refinement diverged — keep original
            logger.warning("Refine diverged for %s: delta=(%d,%d,%d,%d), keeping original",
                           element.get("id", "?"), dx, dy, dw, dh)
            return element

        # Accept refinement
        refined = dict(element)
        refined["bbox"] = {"x": rx, "y": ry, "width": rw, "height": rh}
        refined["_refined"] = {
            "original_bbox": bbox,
            "delta": {"dx": rx - ex, "dy": ry - ey, "dw": rw - ew, "dh": rh - eh},
        }
        if result.get("name"):
            refined["name"] = result["name"]

        return refined

    except Exception as e:
        logger.warning("Refine failed for %s: %s", element.get("id", "?"), e)
        return element


async def iterative_refine(
    image_b64: str,
    coarse_layout: List[Dict[str, Any]],
    ai_engine=None,
    max_refine: int = 50,
    min_area: int = 400,
    padding: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Refine a coarse layout by re-detecting each element in its crop.

    From Megatron-Core's gradient accumulation loop start. Each micro-batch
    refines the gradient estimate. Then, follow that pattern: each element
    refinement improves the layout precision. Next, convergence is measured
    by total pixel delta across all elements. Subsequently, elements below
    min_area are skipped (too small to refine meaningfully). Finally, stats
    report per-element deltas for quality assessment.

    Parameters
    ----------
    image_b64 : full screenshot
    coarse_layout : output from vision_detect() or elk_to_mastergo()
    max_refine : max elements to refine (API cost control)
    min_area : skip elements smaller than this
    padding : crop padding in pixels

    Returns
    -------
    (refined_layout, stats)
    """
    t0 = time.monotonic()

    # Filter: only refine elements large enough to benefit
    candidates = []
    passthrough = []
    for elem in coarse_layout:
        bbox = elem.get("bbox", {})
        area = bbox.get("width", 0) * bbox.get("height", 0)
        if area >= min_area and len(candidates) < max_refine:
            candidates.append(elem)
        else:
            passthrough.append(elem)

    if not candidates:
        return coarse_layout, {"refined": 0, "skipped": len(passthrough), "total_delta_px": 0}

    # Create AIEngine once
    if ai_engine is None:
        try:
            from backend.config import get_settings
            from backend.ai_engine import AIEngine
            ai_engine = AIEngine(get_settings())
        except Exception:
            return coarse_layout, {"refined": 0, "error": "no AIEngine"}

    # Refine each candidate
    refined = []
    total_delta = 0
    refined_count = 0

    for elem in candidates:
        r = await refine_single_region(image_b64, elem, ai_engine, padding)
        refined.append(r)
        if "_refined" in r:
            d = r["_refined"]["delta"]
            total_delta += abs(d["dx"]) + abs(d["dy"]) + abs(d["dw"]) + abs(d["dh"])
            refined_count += 1

    elapsed = (time.monotonic() - t0) * 1000
    result = refined + passthrough

    stats = {
        "refined": refined_count,
        "skipped": len(passthrough),
        "candidates": len(candidates),
        "total_delta_px": total_delta,
        "avg_delta_px": round(total_delta / refined_count, 1) if refined_count else 0,
        "processing_time_ms": round(elapsed, 2),
    }

    logger.info("Iterative refine: %d/%d elements refined, total delta=%dpx, %.0fms",
                refined_count, len(candidates), total_delta, elapsed)
