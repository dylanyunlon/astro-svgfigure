"""
Rembg Processor — Server-Side Green-Screen Background Removal
================================================================
Removes green-screen (#00FF00) background from animation frames,
producing transparent PNG outputs.

Pipeline Position: Step 4 of 4
    Step 1: Claude 4.6 image analysis
    Step 2: Grok animation prompt engineering
    Step 3: Gemini multi-frame generation (green BG)
  → Step 4: THIS MODULE (background removal → transparent PNG)

Two Execution Modes:
────────────────────
1. CLIENT-SIDE (default, preferred):
   The browser's Canvas API does chroma-key removal in JavaScript.
   This is faster (no round-trip) and sufficient for most cases.
   See playground/index.astro's processGreenScreen() function.

2. SERVER-SIDE (this module, optional enhancement):
   Used when client-side removal produces poor results:
   - Complex edges (hair, fur, feathers)
   - Green objects in the foreground
   - Semi-transparent elements
   This module offers both:
   a) Pure chroma-key (fast, PIL-based, no extra deps)
   b) rembg U2-Net (accurate, requires rembg package)

Knuth-Level Critiques:
─────────────────────
User Angle:
  - The tolerance slider (10-120) maps non-linearly to the actual
    green removal. Low tolerance (10-30) is strict and may leave
    green fringes. High tolerance (80+) may eat into green objects
    in the foreground. The default (60) is tuned for Gemini's
    typical green-screen output which uses very saturated green.

  - "Edge softness" (0-5px) blurs only the alpha channel, not the
    color channels. This prevents the subject from looking blurry
    while softening the transition to transparency. Without this,
    the edges look jagged ("staircase" artifact).

System Angle:
  - PIL's pixel-by-pixel access via getpixel/putpixel is SLOW for
    large images. We use numpy for vectorized operations when available.
    Fallback to PIL-only is ~10x slower but works without numpy.

  - The green-spill correction subtracts excess green from edge pixels.
    This prevents the "green halo" effect but can over-correct on
    naturally green subjects (leaves, grass). The despill strength
    is adaptive: stronger for pixels with higher green dominance.

  - Memory: 16 frames at 1024x1024 RGBA = ~64MB of numpy arrays.
    This is fine for a server, but watch for memory pressure if
    processing many requests concurrently.

GitHub references:
  - danielgatis/rembg (U2-Net background removal)
  - pillow/Pillow (image processing)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Conditional imports ──
# We try numpy first (fast), fall back to pure PIL (slow).
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.info("numpy not available — using pure PIL for rembg (slower)")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not available — server-side rembg disabled")

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    logger.info("rembg not installed — U2-Net background removal unavailable")


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def is_available() -> Dict[str, bool]:
    """Check which rembg methods are available."""
    return {
        "green_screen": HAS_PIL,
        "rembg_u2net": HAS_REMBG,
        "rembg_isnet": HAS_REMBG,
        "numpy_accelerated": HAS_NUMPY,
    }


async def process_frames(
    frames_b64: List[str],
    method: str = "green_screen",
    tolerance: int = 60,
    edge_blur: float = 1.0,
    despill: bool = True,
) -> Dict[str, Any]:
    """
    Remove background from animation frames.

    Parameters
    ----------
    frames_b64 : List[str]
        Base64-encoded PNG frames with green-screen background.
    method : str
        "green_screen" (fast, chroma-key) or "rembg_u2net" (accurate, ML).
    tolerance : int
        Green-screen tolerance (10-150). Higher = more aggressive removal.
    edge_blur : float
        Edge softness in pixels (0-5).
    despill : bool
        Apply green-spill correction on edge pixels.

    Returns
    -------
    Dict with keys: success, frames_b64, stats, error
    """
    if not HAS_PIL:
        return {
            "success": False,
            "error": "Pillow is not installed. Install with: pip install Pillow",
            "frames_b64": None,
            "stats": None,
        }

    t0 = time.monotonic()
    result_frames: List[str] = []
    total_green_pixels = 0
    total_pixels = 0

    try:
        for i, frame_b64 in enumerate(frames_b64):
            img = _decode_image(frame_b64)
            if img is None:
                logger.warning("Frame %d: failed to decode, skipping", i)
                result_frames.append(frame_b64)  # Pass through unchanged
                continue

            if method == "rembg_u2net" and HAS_REMBG:
                processed = _remove_bg_rembg(img)
            elif method == "rembg_isnet" and HAS_REMBG:
                processed = _remove_bg_rembg(img, model_name="isnet-general-use")
            else:
                processed, green_count, pixel_count = _remove_green_screen(
                    img, tolerance=tolerance, edge_blur=edge_blur, despill=despill,
                )
                total_green_pixels += green_count
                total_pixels += pixel_count

            result_b64 = _encode_image(processed)
            result_frames.append(result_b64)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        stats = {
            "frame_count": len(result_frames),
            "green_pixels_removed": total_green_pixels,
            "total_pixels": total_pixels,
            "green_percentage": (
                f"{(total_green_pixels / total_pixels * 100):.1f}%"
                if total_pixels > 0 else "N/A"
            ),
            "processing_time_ms": elapsed_ms,
            "method": method,
            "numpy_accelerated": HAS_NUMPY,
        }

        return {
            "success": True,
            "frames_b64": result_frames,
            "stats": stats,
            "error": None,
        }

    except Exception as e:
        logger.exception("process_frames failed: %s", e)
        return {
            "success": False,
            "frames_b64": None,
            "stats": None,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Green-Screen Chroma-Key Removal
# ═══════════════════════════════════════════════════════════════════════

def _remove_green_screen(
    img: "Image.Image",
    tolerance: int = 60,
    edge_blur: float = 1.0,
    despill: bool = True,
) -> Tuple["Image.Image", int, int]:
    """
    Remove green-screen background using chroma-key algorithm.

    Algorithm (3-pass):
    ───────────────────
    Pass 1 — Core green detection:
      Pixel is green if: G - max(R, B) > tolerance * 0.3 AND G > 80
      These pixels get alpha = 0 (fully transparent).

    Pass 2 — Edge transition:
      Pixel is edge-green if: G - max(R, B) > tolerance * 0.15 AND G > 60
      These get partial alpha proportional to green dominance.
      Green-spill correction reduces the G channel on these pixels.

    Pass 3 — Alpha blur (optional):
      Gaussian blur on the alpha channel only, then take the minimum
      of original and blurred alpha. This softens edges without
      affecting fully-opaque or fully-transparent areas.

    Returns: (processed_image, green_pixel_count, total_pixel_count)
    """
    img = img.convert("RGBA")

    if HAS_NUMPY:
        return _remove_green_screen_numpy(img, tolerance, edge_blur, despill)
    else:
        return _remove_green_screen_pil(img, tolerance, edge_blur, despill)


def _remove_green_screen_numpy(
    img: "Image.Image",
    tolerance: int,
    edge_blur: float,
    despill: bool,
) -> Tuple["Image.Image", int, int]:
    """Vectorized green-screen removal using numpy."""
    arr = np.array(img, dtype=np.float32)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    total_pixels = int(r.shape[0] * r.shape[1])
    max_rb = np.maximum(r, b)
    green_dominance = g - max_rb

    # Pass 1: Core green removal
    core_green = (green_dominance > tolerance * 0.3) & (g > 80)
    a[core_green] = 0
    green_count = int(np.sum(core_green))

    # Pass 2: Edge transition
    edge_green = (
        (green_dominance > tolerance * 0.15) &
        (g > 60) &
        ~core_green
    )
    if np.any(edge_green):
        edge_alpha = np.clip(
            255 - (green_dominance[edge_green] / tolerance * 255 * 2),
            0, 255,
        )
        a[edge_green] = np.minimum(a[edge_green], edge_alpha)

        # Green-spill correction
        if despill:
            avg_rb = (r[edge_green] + b[edge_green]) / 2
            g[edge_green] = np.minimum(g[edge_green], avg_rb)

    # Pass 3: Alpha blur for edge softness
    if edge_blur > 0:
        try:
            from PIL import ImageFilter
            alpha_img = Image.fromarray(a.astype(np.uint8), mode="L")
            blurred = alpha_img.filter(ImageFilter.GaussianBlur(radius=edge_blur))
            blurred_arr = np.array(blurred, dtype=np.float32)
            a[:] = np.minimum(a, blurred_arr)
        except Exception as e:
            logger.warning("Alpha blur failed: %s", e)

    # Reconstruct
    arr[:, :, 0] = np.clip(r, 0, 255)
    arr[:, :, 1] = np.clip(g, 0, 255)
    arr[:, :, 2] = np.clip(b, 0, 255)
    arr[:, :, 3] = np.clip(a, 0, 255)

    result = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    return result, green_count, total_pixels


def _remove_green_screen_pil(
    img: "Image.Image",
    tolerance: int,
    edge_blur: float,
    despill: bool,
) -> Tuple["Image.Image", int, int]:
    """Pure PIL green-screen removal (no numpy dependency)."""
    width, height = img.size
    pixels = img.load()
    total_pixels = width * height
    green_count = 0

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            max_rb = max(r, b)
            green_dom = g - max_rb

            if green_dom > tolerance * 0.3 and g > 80:
                # Core green
                pixels[x, y] = (r, g, b, 0)
                green_count += 1
            elif green_dom > tolerance * 0.15 and g > 60:
                # Edge transition
                edge_alpha = max(0, min(255, 255 - int(green_dom / tolerance * 255 * 2)))
                new_alpha = min(a, edge_alpha)
                new_g = min(g, (r + b) // 2) if despill else g
                pixels[x, y] = (r, new_g, b, new_alpha)

    # Alpha blur with PIL filter
    if edge_blur > 0:
        try:
            from PIL import ImageFilter
            # Extract alpha, blur, take min
            alpha_channel = img.split()[3]
            blurred = alpha_channel.filter(ImageFilter.GaussianBlur(radius=edge_blur))
            # Take min of original and blurred alpha
            from PIL import ImageChops
            min_alpha = ImageChops.darker(alpha_channel, blurred)
            img.putalpha(min_alpha)
        except Exception as e:
            logger.warning("PIL alpha blur failed: %s", e)

    return img, green_count, total_pixels


# ═══════════════════════════════════════════════════════════════════════
#  rembg U2-Net Background Removal
# ═══════════════════════════════════════════════════════════════════════

def _remove_bg_rembg(
    img: "Image.Image",
    model_name: str = "u2net",
) -> "Image.Image":
    """
    Use rembg's U2-Net model for background removal.

    This is more accurate than chroma-key for complex edges
    (hair, fur, feathers) but slower (~2s per frame on CPU).

    System-angle critique: rembg downloads the model (~170MB) on
    first use. This causes a long delay on the first request.
    In production, pre-download the model during server startup.
    """
    if not HAS_REMBG:
        raise RuntimeError("rembg is not installed")

    img_rgba = img.convert("RGBA")
    result = rembg_remove(img_rgba, model_name=model_name)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Image I/O
# ═══════════════════════════════════════════════════════════════════════

def _decode_image(b64: str) -> Optional["Image.Image"]:
    """Decode base64 string to PIL Image."""
    try:
        # Strip data URI prefix if present
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]

        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        logger.warning("Failed to decode image: %s", e)
        return None


def _encode_image(img: "Image.Image", format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════
#  Utilities for Frame Normalization
# ═══════════════════════════════════════════════════════════════════════

def normalize_frame_dimensions(
    frames_b64: List[str],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> List[str]:
    """
    Normalize all frames to the same dimensions.

    If no target dimensions are specified, uses the first frame's
    dimensions as the target.

    User-angle critique: this can crop or stretch frames if Gemini
    generated inconsistent dimensions. We use LANCZOS resampling
    for quality, but the distortion is unavoidable when source
    dimensions don't match.
    """
    if not HAS_PIL or not frames_b64:
        return frames_b64

    # Determine target dimensions from first frame
    first_img = _decode_image(frames_b64[0])
    if first_img is None:
        return frames_b64

    target_w = target_width or first_img.width
    target_h = target_height or first_img.height

    normalized = []
    for i, frame_b64 in enumerate(frames_b64):
        img = _decode_image(frame_b64)
        if img is None:
            normalized.append(frame_b64)
            continue

        if img.width != target_w or img.height != target_h:
            img = img.resize((target_w, target_h), Image.LANCZOS)
            logger.info("Frame %d: resized from %dx%d to %dx%d",
                       i, img.width, img.height, target_w, target_h)

        normalized.append(_encode_image(img))

    return normalized
