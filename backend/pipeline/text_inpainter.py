"""
Text Inpainter — Post-processing for Gemini-generated Scientific Figures
=========================================================================
When Gemini generates scientific figure images containing Chinese text,
the rendered characters often lack natural stroke continuity and coordinate
coherence. This module removes such dark/black text pixels by replacing
them with surrounding colors using OpenCV's inpainting algorithms.

Strategy:
  1. Decode the base64 image to a numpy array
  2. Create a mask of dark pixels (below a configurable threshold)
  3. Use morphological operations to distinguish thin text strokes
     from thick structural elements (diagram borders, boxes)
  4. Apply OpenCV cv2.inpaint (TELEA algorithm) to fill masked regions
  5. Re-encode to base64 in the original format

Dependencies: opencv-python-headless, numpy, Pillow (all lightweight, no GPU)

GitHub research references (top 10 most relevant libraries):
  1. opencv/opencv — cv2.inpaint (TELEA & NS algorithms) - built-in, lightweight
  2. Sanster/IOPaint — SOTA AI inpainting (overkill for our use, needs GPU)
  3. iuliaturc/detextify — Remove text from AI-generated images (Tesseract+SD)
  4. BilalSardar009/Remove-Text-From-Image — keras_ocr + OpenCV inpainting
  5. aGIToz/PyInpaint — Lightweight PDE-on-graphs inpainting
  6. yu45020/Text_Segmentation_Image_Inpainting — Text segmentation + inpaint
  7. geekyutao/Inpaint-Anything — SAM + inpainting (heavy, GPU)
  8. bnsreenu/python_for_microscopists — keras_ocr text removal tutorial
  9. mikealsim/remove_bw — Remove black/white from images
  10. igorcmoura/inpaint-object-remover — Exemplar-based inpainting

Decision: We use OpenCV's built-in cv2.inpaint (option #1) because:
  - Zero additional dependencies (opencv-python-headless already common)
  - No GPU required, runs in milliseconds
  - Perfect for our use case: replace dark pixels with nearby colors
  - Combined with morphological filtering to preserve diagram borders
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_dark_pixel_mask(
    image_array: np.ndarray,
    dark_threshold: int = 50,
    mode: str = "global",
    min_component_size: int = 0,
) -> np.ndarray:
    """
    Create a binary mask identifying dark pixels in an RGB image.

    A pixel is considered "dark" if ALL three RGB channels are below
    the given threshold. This catches black and very dark text while
    ignoring colored elements.

    Args:
        image_array: H×W×3 uint8 numpy array (RGB or RGBA)
        dark_threshold: Pixels with all channels < this value are marked.
                        Use 1 for pure-black only, ~50 for near-black text.
        mode: "global" for simple threshold, "adaptive" for locally-adaptive
              Gaussian thresholding that detects pixels dark relative to
              their neighborhood.
        min_component_size: If > 0, remove connected components with fewer
                            than this many pixels (noise filtering).

    Returns:
        H×W uint8 array: 255 where dark, 0 elsewhere
    """
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError(f"Expected H×W×3 image, got shape {image_array.shape}")

    # Use only the first 3 channels (handles RGBA gracefully)
    rgb = image_array[:, :, :3]

    if mode == "adaptive":
        # Convert to grayscale for adaptive thresholding
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Adaptive Gaussian threshold: marks pixels that are dark
        # relative to their local neighborhood
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=15,
        )
        # Also apply global threshold as a union to catch truly dark pixels
        global_mask = np.all(rgb < dark_threshold, axis=2).astype(np.uint8) * 255
        dark_mask = cv2.bitwise_or(adaptive_mask, global_mask)
    else:
        # Global mode: all three channels must be below threshold
        dark_mask = np.all(rgb < dark_threshold, axis=2).astype(np.uint8) * 255

    # Filter by connected component size to remove noise
    if min_component_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_component_size:
                dark_mask[labels == label_id] = 0

    return dark_mask


def _filter_mask_by_stroke_width(
    mask: np.ndarray,
    min_stroke_width: int = 6,
) -> np.ndarray:
    """
    Remove thick structural elements (borders, boxes) from the mask,
    keeping only thin strokes that are likely text.

    Uses morphological erosion: if a dark region survives erosion with
    a kernel of size min_stroke_width, it's too thick to be text →
    subtract it from the mask.

    Args:
        mask: Binary mask (255=dark, 0=background)
        min_stroke_width: Minimum width (px) to consider as structural border

    Returns:
        Filtered mask with only thin (text-like) strokes
    """
    if min_stroke_width <= 0:
        return mask

    # Erode: only thick regions survive
    kernel_size = min_stroke_width
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    thick_regions = cv2.erode(mask, kernel, iterations=1)

    # Dilate back to original size so we can subtract properly
    thick_regions_expanded = cv2.dilate(thick_regions, kernel, iterations=2)

    # Subtract thick regions from original mask → keep only thin strokes
    text_only_mask = cv2.subtract(mask, thick_regions_expanded)

    return text_only_mask


_ALGORITHM_MAP = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}

_MIME_TO_PIL_FORMAT = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/webp": "WEBP",
}


def _resolve_output_format(mime_type: str) -> str:
    """Map a MIME type to a PIL save format string."""
    key = mime_type.lower().strip()
    return _MIME_TO_PIL_FORMAT.get(key, "PNG")


def inpaint_dark_text(
    image_b64: str,
    mime_type: str = "image/png",
    dark_threshold: int = 50,
    min_stroke_width: int = 0,
    inpaint_radius: int = 5,
    algorithm: str = "telea",
) -> Dict[str, Any]:
    """
    Main function: remove dark text pixels from a base64-encoded image
    by inpainting them with surrounding colors.

    Args:
        image_b64: Base64-encoded image data
        mime_type: MIME type of the input image
                   ("image/png", "image/jpeg", or "image/webp")
        dark_threshold: Pixels with all RGB channels < this are treated as text
        min_stroke_width: If > 0, thick dark regions (borders) wider than this
                          many pixels are preserved (not inpainted)
        inpaint_radius: Radius for cv2.inpaint neighborhood
        algorithm: Inpainting algorithm — "telea" (default) or "ns"
                   (Navier-Stokes)

    Returns:
        Dict with keys:
          success (bool), image_b64 (str), mime_type (str),
          stats (dict with dark_pixels_found, pixels_inpainted, width, height),
          and optionally error (str) if success=False
    """
    # Validate algorithm choice
    algo_flag = _ALGORITHM_MAP.get(algorithm.lower().strip() if algorithm else "")
    if algo_flag is None:
        return {
            "success": False,
            "error": (
                f"Unknown algorithm '{algorithm}'. "
                f"Supported algorithms: {', '.join(sorted(_ALGORITHM_MAP))}"
            ),
        }

    # Decode base64 to image
    try:
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to decode input image: {e}")
        return {
            "success": False,
            "error": f"Failed to decode input image: {e}",
        }

    # Convert to RGB numpy array
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_array = np.array(pil_image)
    img_h, img_w = img_array.shape[:2]

    # Create mask of dark pixels
    mask = create_dark_pixel_mask(img_array, dark_threshold=dark_threshold)
    dark_pixels_found = int(np.sum(mask > 0))

    # If min_stroke_width > 0, filter out thick borders
    if min_stroke_width > 0:
        mask = _filter_mask_by_stroke_width(mask, min_stroke_width=min_stroke_width)

    pixels_inpainted = int(np.sum(mask > 0))

    # Build stats dict (always included in result)
    stats = {
        "dark_pixels_found": dark_pixels_found,
        "pixels_inpainted": pixels_inpainted,
        "width": img_w,
        "height": img_h,
    }

    # If no dark pixels found, return image as-is
    if pixels_inpainted == 0:
        return {
            "success": True,
            "image_b64": image_b64,
            "mime_type": mime_type,
            "stats": stats,
        }

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Apply inpainting using the selected algorithm
    inpainted_bgr = cv2.inpaint(
        img_bgr, mask, inpaint_radius, algo_flag
    )

    # Convert back to RGB PIL image
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(inpainted_rgb)

    # Encode back to base64 in original format (supports PNG, JPEG, WebP)
    output_format = _resolve_output_format(mime_type)

    buf = io.BytesIO()
    result_pil.save(buf, format=output_format)
    result_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    logger.info(
        f"Text inpainting complete: found {dark_pixels_found} dark pixels, "
        f"inpainted {pixels_inpainted}, algorithm={algorithm}, "
        f"output {len(result_b64)} base64 chars"
    )

    return {
        "success": True,
        "image_b64": result_b64,
        "mime_type": mime_type,
        "stats": stats,
    }


def process_gemini_result(
    gemini_result: Dict[str, Any],
    dark_threshold: int = 50,
    min_stroke_width: int = 0,
    inpaint_radius: int = 5,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function that accepts the dict format returned by
    gemini_image_gen.generate_image_with_gemini and returns a cleaned version.

    If the input result has success=False, it is passed through unchanged.

    Args:
        gemini_result: Dict from generate_image_with_gemini with keys:
            success, image_b64, mime_type, text_response, model_used
        dark_threshold: Passed to inpaint_dark_text
        min_stroke_width: Passed to inpaint_dark_text
        inpaint_radius: Passed to inpaint_dark_text
        enabled: If False, skip inpainting and return gemini_result unchanged

    Returns:
        Dict in the same format, with image_b64 replaced by cleaned version
    """
    if not enabled:
        return gemini_result

    if not gemini_result.get("success", False):
        return gemini_result

    image_b64 = gemini_result.get("image_b64", "")
    mime_type = gemini_result.get("mime_type", "image/png")

    result = inpaint_dark_text(
        image_b64,
        mime_type=mime_type,
        dark_threshold=dark_threshold,
        min_stroke_width=min_stroke_width,
        inpaint_radius=inpaint_radius,
    )

    if not result["success"]:
        # If inpainting failed, return original with a warning
        logger.warning(f"Text inpainting failed: {result.get('error')}")
        return gemini_result

    # Build output preserving all original keys
    output = dict(gemini_result)
    output["image_b64"] = result["image_b64"]
    output["mime_type"] = result["mime_type"]

    return output
