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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_dark_pixel_mask(
    image_array: np.ndarray,
    dark_threshold: int = 50,
    mode: str = "global",
    min_component_size: int = 0,
    target_color: Optional[tuple] = None,
    color_tolerance: Optional[int] = None,
    dilation: int = 0,
) -> np.ndarray:
    """
    Create a binary mask identifying target-colored pixels in an RGB image.

    By default (no target_color), a pixel is considered "dark" if ALL three
    RGB channels are below the given threshold. When target_color is set,
    pixels within color_tolerance Euclidean distance of the target are marked.

    Args:
        image_array: H×W×3 uint8 numpy array (RGB or RGBA)
        dark_threshold: Pixels with all channels < this value are marked.
                        Use 1 for pure-black only, ~50 for near-black text.
        mode: "global" for simple threshold, "adaptive" for locally-adaptive
              Gaussian thresholding.
        min_component_size: If > 0, remove connected components with fewer
                            than this many pixels (noise filtering).
        target_color: Optional RGB tuple (or list of tuples) to target
                      specific colors instead of dark pixels.
        color_tolerance: Max Euclidean distance from target_color to match.
                         Only used when target_color is set.
        dilation: If > 0, dilate the mask by this many pixels.
                  If < 0, erode the mask.

    Returns:
        H×W uint8 array: 255 where matched, 0 elsewhere
    """
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError(f"Expected H×W×3 image, got shape {image_array.shape}")

    # Use only the first 3 channels (handles RGBA gracefully)
    rgb = image_array[:, :, :3].astype(np.float32)

    if target_color is not None:
        # Color-targeted mode
        tolerance = color_tolerance if color_tolerance is not None else dark_threshold

        # Support list of target colors
        if isinstance(target_color, list):
            colors = target_color
        else:
            colors = [target_color]

        dark_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        for tc in colors:
            tc_arr = np.array(tc, dtype=np.float32).reshape(1, 1, 3)
            # Use per-channel max difference (Chebyshev distance)
            # so that tolerance=255 covers ALL possible pixel values
            dist = np.max(np.abs(rgb - tc_arr), axis=2)
            dark_mask[dist <= tolerance] = 255
    elif mode == "adaptive":
        gray = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=15,
        )
        global_mask = np.all(image_array[:, :, :3] < dark_threshold, axis=2).astype(np.uint8) * 255
        dark_mask = cv2.bitwise_or(adaptive_mask, global_mask)
    else:
        dark_mask = np.all(image_array[:, :, :3] < dark_threshold, axis=2).astype(np.uint8) * 255

    # Filter by connected component size to remove noise
    if min_component_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_component_size:
                dark_mask[labels == label_id] = 0

    # Apply dilation or erosion
    if dilation != 0:
        k_size = max(1, abs(dilation) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if dilation > 0:
            dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
        else:
            dark_mask = cv2.erode(dark_mask, kernel, iterations=1)

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
    "image/gif": "GIF",
    "image/tiff": "TIFF",
    "image/bmp": "BMP",
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
    return_mask: bool = False,
    target_color: Optional[Union[tuple, list]] = None,
    color_tolerance: Optional[int] = None,
    mask_dilation: int = 0,
    roi: Optional[Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]] = None,
) -> Dict[str, Any]:
    """
    Main function: remove dark text pixels from a base64-encoded image
    by inpainting them with surrounding colors.

    Args:
        image_b64: Base64-encoded image data
        mime_type: MIME type of the input image
        dark_threshold: Pixels with all RGB channels < this are treated as text
        min_stroke_width: If > 0, thick dark regions wider than this are preserved
        inpaint_radius: Radius for cv2.inpaint neighborhood
        algorithm: "telea" (default) or "ns" (Navier-Stokes)
        return_mask: If True, include 'mask_b64' in the result dict
        target_color: Optional RGB tuple (or list of tuples) to target
                      specific colors. None = default dark-pixel detection.
        color_tolerance: Max Euclidean distance from target_color.
        mask_dilation: Pixels to dilate (>0) or erode (<0) the mask.
        roi: Optional (x, y, w, h) tuple or list of tuples to limit
             processing to specific regions. None = full image.

    Returns:
        Dict with keys:
          success (bool), image_b64 (str), mime_type (str),
          stats (dict), and optionally mask_b64 (str), error (str)
    """
    t_start = time.monotonic()

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
    total_pixels = img_h * img_w

    # Build ROI mask if needed
    roi_mask = None
    if roi is not None:
        # Validate ROI type
        if isinstance(roi, str):
            return {
                "success": False,
                "error": f"Invalid roi type: expected tuple or list, got {type(roi).__name__}",
            }
        roi_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        rois = roi if isinstance(roi, list) and len(roi) > 0 and isinstance(roi[0], (tuple, list)) else [roi]
        for r in rois:
            if not isinstance(r, (tuple, list)) or len(r) != 4:
                continue
            rx, ry, rw, rh = r
            rx = max(0, int(rx))
            ry = max(0, int(ry))
            rx2 = min(img_w, rx + int(rw))
            ry2 = min(img_h, ry + int(rh))
            if rx2 > rx and ry2 > ry:
                roi_mask[ry:ry2, rx:rx2] = 255

    # Create mask of dark / target pixels
    mask_kwargs: Dict[str, Any] = {"dark_threshold": dark_threshold}
    if target_color is not None:
        mask_kwargs["target_color"] = target_color
    if color_tolerance is not None:
        mask_kwargs["color_tolerance"] = color_tolerance

    mask = create_dark_pixel_mask(img_array, **mask_kwargs)
    dark_pixels_found = int(np.sum(mask > 0))

    # If min_stroke_width > 0, filter out thick borders
    if min_stroke_width > 0:
        mask = _filter_mask_by_stroke_width(mask, min_stroke_width=min_stroke_width)

    # Apply mask dilation/erosion
    if mask_dilation != 0:
        k_size = max(1, abs(mask_dilation) * 2 + 1)
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size)
        )
        if mask_dilation > 0:
            mask = cv2.dilate(mask, dilation_kernel, iterations=1)
        else:
            mask = cv2.erode(mask, dilation_kernel, iterations=1)

    # Apply ROI restriction
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    pixels_inpainted = int(np.sum(mask > 0))

    # Compute enhanced stats
    inpainted_pct = (pixels_inpainted / total_pixels * 100) if total_pixels > 0 else 0.0
    t_elapsed_ms = (time.monotonic() - t_start) * 1000

    stats = {
        "dark_pixels_found": dark_pixels_found,
        "pixels_inpainted": pixels_inpainted,
        "width": img_w,
        "height": img_h,
        "total_pixels": total_pixels,
        "inpainted_percentage": round(inpainted_pct, 4),
        "processing_time_ms": round(t_elapsed_ms, 2),
    }

    # Optionally encode mask as base64 image for debugging
    mask_b64_out = None
    if return_mask:
        mask_pil = Image.fromarray(mask)
        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        mask_b64_out = base64.b64encode(mask_buf.getvalue()).decode("ascii")

    # If no pixels to inpaint, return image as-is
    if pixels_inpainted == 0:
        result = {
            "success": True,
            "image_b64": image_b64,
            "mime_type": mime_type,
            "stats": stats,
        }
        if return_mask and mask_b64_out is not None:
            result["mask_b64"] = mask_b64_out
        return result

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Apply inpainting
    inpainted_bgr = cv2.inpaint(img_bgr, mask, inpaint_radius, algo_flag)

    # If ROI was used, only apply inpainting within ROI regions
    if roi_mask is not None:
        roi_3ch = cv2.merge([roi_mask, roi_mask, roi_mask])
        inpainted_bgr = np.where(roi_3ch > 0, inpainted_bgr, img_bgr)

    # Convert back to RGB PIL image
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(inpainted_rgb)

    # Encode back to base64 with format-specific quality settings
    output_format = _resolve_output_format(mime_type)
    buf = io.BytesIO()
    save_kwargs: Dict[str, Any] = {"format": output_format}
    if output_format == "WEBP":
        save_kwargs["quality"] = 100
        save_kwargs["method"] = 0  # Fastest (least compressed)
    elif output_format == "JPEG":
        save_kwargs["quality"] = 95
    result_pil.save(buf, **save_kwargs)
    result_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Update timing
    stats["processing_time_ms"] = round((time.monotonic() - t_start) * 1000, 2)

    logger.info(
        f"Text inpainting complete: found {dark_pixels_found} dark pixels, "
        f"inpainted {pixels_inpainted}, algorithm={algorithm}, "
        f"output {len(result_b64)} base64 chars"
    )

    result = {
        "success": True,
        "image_b64": result_b64,
        "mime_type": mime_type,
        "stats": stats,
    }
    if return_mask and mask_b64_out is not None:
        result["mask_b64"] = mask_b64_out

    return result


def inpaint_batch(
    images: List[Dict[str, Any]],
    dark_threshold: int = 50,
    min_stroke_width: int = 0,
    inpaint_radius: int = 5,
    algorithm: str = "telea",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Process multiple images in a single call.

    Args:
        images: List of dicts, each with 'image_b64' and 'mime_type' keys
        dark_threshold: Passed to inpaint_dark_text
        min_stroke_width: Passed to inpaint_dark_text
        inpaint_radius: Passed to inpaint_dark_text
        algorithm: Passed to inpaint_dark_text
        **kwargs: Additional keyword arguments passed to inpaint_dark_text

    Returns:
        List of result dicts in the same order as inputs
    """
    if not images:
        return []

    results = []
    for img_dict in images:
        image_b64 = img_dict.get("image_b64", "")
        mime_type = img_dict.get("mime_type", "image/png")
        result = inpaint_dark_text(
            image_b64,
            mime_type=mime_type,
            dark_threshold=dark_threshold,
            min_stroke_width=min_stroke_width,
            inpaint_radius=inpaint_radius,
            algorithm=algorithm,
            **kwargs,
        )
        results.append(result)
    return results


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
