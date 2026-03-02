"""
SVG Scaler — Coordinate Scaling & Alignment
=============================================
Scales SVG coordinates to match desired output dimensions.

Ported from autofigure2.py step 4.7:
  "坐标系对齐：比较 figure.png 与 SVG 尺寸，计算缩放因子"

Also references inter_rl_figure.py for scaling logic.

Use cases:
  1. Scale SVG to fit a specific canvas size (e.g., paper column width)
  2. Align SVG coordinates with a reference image
  3. Adjust viewport for responsive display
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Main Functions
# ============================================================================

def scale_svg(
    svg: str,
    target_width: Optional[float] = None,
    target_height: Optional[float] = None,
    scale_factor: Optional[float] = None,
    preserve_aspect_ratio: bool = True,
) -> str:
    """
    Scale SVG to target dimensions or by a scale factor.

    Args:
        svg: Input SVG string
        target_width: Desired output width in pixels
        target_height: Desired output height in pixels
        scale_factor: Explicit scale factor (overrides target_width/height)
        preserve_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Scaled SVG string
    """
    # Parse current dimensions
    current_vb = _parse_viewbox(svg)
    current_width = _parse_dimension(svg, "width")
    current_height = _parse_dimension(svg, "height")

    # Use viewBox dimensions as source of truth, fallback to width/height
    src_w = current_vb[2] if current_vb else current_width or 800
    src_h = current_vb[3] if current_vb else current_height or 600

    # Compute scale factor
    if scale_factor is not None:
        sx = sy = scale_factor
    elif target_width and target_height:
        sx = target_width / src_w
        sy = target_height / src_h
        if preserve_aspect_ratio:
            sx = sy = min(sx, sy)
    elif target_width:
        sx = target_width / src_w
        sy = sx if preserve_aspect_ratio else 1.0
    elif target_height:
        sy = target_height / src_h
        sx = sy if preserve_aspect_ratio else 1.0
    else:
        return svg  # No scaling needed

    # New dimensions
    new_w = src_w * sx
    new_h = src_h * sy

    result = svg

    # Update width/height attributes
    result = _replace_dimension(result, "width", new_w)
    result = _replace_dimension(result, "height", new_h)

    # Update viewBox if using uniform scale
    if scale_factor is not None or preserve_aspect_ratio:
        # Keep viewBox as-is, just update width/height
        pass
    else:
        # Non-uniform scale: update viewBox too
        vb_x = current_vb[0] if current_vb else 0
        vb_y = current_vb[1] if current_vb else 0
        result = _replace_viewbox(result, vb_x, vb_y, new_w, new_h)

    logger.info(f"SVG scaled: {src_w}x{src_h} → {new_w:.0f}x{new_h:.0f} (sx={sx:.3f}, sy={sy:.3f})")

    return result


def align_svg_to_reference(
    svg: str,
    ref_width: float,
    ref_height: float,
) -> str:
    """
    Align SVG coordinates to a reference image size.
    Ported from autofigure2.py step 4.7.

    Args:
        svg: Input SVG string
        ref_width: Reference image width
        ref_height: Reference image height

    Returns:
        Aligned SVG string
    """
    current_vb = _parse_viewbox(svg)
    svg_w = current_vb[2] if current_vb else _parse_dimension(svg, "width") or 800
    svg_h = current_vb[3] if current_vb else _parse_dimension(svg, "height") or 600

    # Compute scale factors
    scale_x = ref_width / svg_w if svg_w > 0 else 1.0
    scale_y = ref_height / svg_h if svg_h > 0 else 1.0

    logger.info(
        f"Aligning SVG to reference {ref_width}x{ref_height}: "
        f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}"
    )

    # Apply uniform scale (preserving aspect ratio)
    uniform_scale = min(scale_x, scale_y)
    return scale_svg(svg, scale_factor=uniform_scale)


def fit_to_column_width(svg: str, column_width: float = 487.0) -> str:
    """
    Fit SVG to standard academic paper column width.

    Common column widths (in points):
      - Single column: ~487pt (IEEE, ACM)
      - Double column: ~234pt per column
      - NeurIPS/ICLR: ~487pt

    Args:
        svg: Input SVG
        column_width: Target width in points

    Returns:
        Scaled SVG
    """
    return scale_svg(svg, target_width=column_width)


# ============================================================================
# SVG Parsing Helpers
# ============================================================================

def _parse_viewbox(svg: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse viewBox attribute from SVG."""
    match = re.search(r'viewBox\s*=\s*"([^"]+)"', svg)
    if not match:
        match = re.search(r"viewBox\s*=\s*'([^']+)'", svg)
    if match:
        parts = match.group(1).strip().split()
        if len(parts) == 4:
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                pass
    return None


def _parse_dimension(svg: str, attr: str) -> Optional[float]:
    """Parse width or height attribute from SVG root element."""
    # Only look in the <svg> opening tag
    svg_tag_match = re.search(r"<svg[^>]*>", svg, re.DOTALL)
    if not svg_tag_match:
        return None

    svg_tag = svg_tag_match.group(0)
    pattern = rf'{attr}\s*=\s*"([^"]*)"'
    match = re.search(pattern, svg_tag)
    if not match:
        pattern = rf"{attr}\s*=\s*'([^']*)'"
        match = re.search(pattern, svg_tag)

    if match:
        value = match.group(1).strip()
        # Remove units (px, pt, em, etc.)
        numeric = re.match(r"([\d.]+)", value)
        if numeric:
            return float(numeric.group(1))

    return None


def _replace_dimension(svg: str, attr: str, value: float) -> str:
    """Replace or add width/height attribute in SVG root element."""
    # Find the <svg> tag
    svg_tag_match = re.search(r"<svg([^>]*)>", svg, re.DOTALL)
    if not svg_tag_match:
        return svg

    svg_attrs = svg_tag_match.group(1)
    new_value = f'{attr}="{value:.1f}"'

    # Replace existing attribute
    pattern = rf'{attr}\s*=\s*"[^"]*"'
    if re.search(pattern, svg_attrs):
        new_attrs = re.sub(pattern, new_value, svg_attrs)
    else:
        pattern = rf"{attr}\s*=\s*'[^']*'"
        if re.search(pattern, svg_attrs):
            new_attrs = re.sub(pattern, new_value, svg_attrs)
        else:
            # Add attribute
            new_attrs = f" {new_value}" + svg_attrs

    return svg[:svg_tag_match.start(1)] + new_attrs + svg[svg_tag_match.end(1):]


def _replace_viewbox(
    svg: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    """Replace or add viewBox attribute."""
    new_vb = f'viewBox="{x:.1f} {y:.1f} {width:.1f} {height:.1f}"'

    pattern = r'viewBox\s*=\s*"[^"]*"'
    if re.search(pattern, svg):
        return re.sub(pattern, new_vb, svg, count=1)

    pattern = r"viewBox\s*=\s*'[^']*'"
    if re.search(pattern, svg):
        return re.sub(pattern, new_vb, svg, count=1)

    # Add viewBox after <svg
    return svg.replace("<svg", f"<svg {new_vb}", 1)


# ============================================================================
# Utility: Get SVG Dimensions
# ============================================================================

def get_svg_dimensions(svg: str) -> Tuple[float, float]:
    """
    Get SVG dimensions (width, height).
    Prefers viewBox, falls back to width/height attributes.
    """
    vb = _parse_viewbox(svg)
    if vb:
        return vb[2], vb[3]

    w = _parse_dimension(svg, "width") or 800
    h = _parse_dimension(svg, "height") or 600
    return w, h
