"""sprite_vectorizer.py — Vectorize simple sprites, keep complex ones raster (M217).

openai/triton's ASTSource.hash() folds every input that affects the output into
one deterministic key, so identical inputs always produce identical compiled
artifacts.  Vectorization must share that property: a sprite vectorized twice
must yield byte-identical paths (no RNG), so to-svg.ts stays deterministic and
the M208 snapshot test can assert.

So we do NOT use a stochastic tracer.  We trace the alpha silhouette with
marching squares (skimage.measure.find_contours) — a deterministic geometric
algorithm — simplify the contour (Ramer–Douglas–Peucker), and emit SVG
<path> markup.  Simple sprites (few colors, few contours, no fine gradient)
become crisp infinitely-scalable vectors that match the academic line-art of
the M203–M207 skeleton.  Complex illustrations (many contours / gradient-heavy)
would explode into thousands of path points, so they are left as raster PNG —
the mixed-document strategy M214 already supports via spriteRef.format.

Pipeline position: between M216 (alignment) and M214 (injection).  Input is a
transparent SpriteAsset (format == "png"); output is either the same asset
upgraded to format == "svg" (image_b64 replaced by inline <g>/<path> markup) or
the asset unchanged when it is too complex to vectorize cleanly.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Complexity gate — decide vectorize vs keep-raster
# ═══════════════════════════════════════════════════════════════════════════

# A sprite is "simple enough" to vectorize when, after silhouette extraction:
#   - it has at most MAX_CONTOURS distinct contour loops, and
#   - the total simplified point count is under MAX_POINTS, and
#   - its color content is near-monochrome (low color variance).
MAX_CONTOURS = 12
MAX_POINTS = 600
COLOR_VARIANCE_MAX = 1800.0   # per-channel variance over opaque pixels
ALPHA_LEVEL = 0.5             # marching-squares iso-level on normalized alpha
RDP_EPSILON = 1.2             # contour simplification tolerance (pixels)


@dataclass
class VectorizeStats:
    vectorized: bool
    reason: str = ""
    contours: int = 0
    points: int = 0
    color_variance: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Geometry — RDP simplify + path emission
# ═══════════════════════════════════════════════════════════════════════════

def _rdp(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    """Ramer–Douglas–Peucker polyline simplification (deterministic)."""
    if len(points) < 3:
        return points
    # Find the point with the max distance from the chord (start..end).
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        # perpendicular distance to the chord
        d = abs((px - x0) * dy - (py - y0) * dx) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _rdp(points[: idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _rdp_closed(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    """RDP for a CLOSED contour (skimage loops back to the start).

    A naive RDP collapses a closed loop to its two endpoints (the start≈end
    chord is degenerate). Instead, split the loop at its two farthest-apart
    vertices into two open polylines, simplify each, and rejoin — the standard
    closed-curve treatment, fully deterministic.
    """
    n = len(points)
    if n < 4:
        return points
    # Drop a duplicated closing vertex if present.
    pts = points[:-1] if points[0] == points[-1] else points
    n = len(pts)
    # Find the vertex farthest from pts[0] → the first split anchor.
    def _d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    far1 = max(range(n), key=lambda i: _d2(pts[0], pts[i]))
    # Farthest from far1 → second anchor.
    far2 = max(range(n), key=lambda i: _d2(pts[far1], pts[i]))
    a, b = sorted((far1, far2))
    arc1 = pts[a:b + 1]
    arc2 = pts[b:] + pts[:a + 1]
    s1 = _rdp(arc1, eps)
    s2 = _rdp(arc2, eps)
    # Rejoin, dropping shared endpoints.
    merged = s1[:-1] + s2[:-1]
    return merged


def _contour_to_path(contour, eps: float) -> Tuple[str, int]:
    """Convert one (row, col) contour array to an SVG path 'd' substring.

    skimage returns (row, col) == (y, x) float coords; we simplify then emit.
    Returns (path_d, point_count).
    """
    pts = [(float(c), float(r)) for (r, c) in contour]  # (x, y)
    pts = _rdp_closed(pts, eps)
    if len(pts) < 3:
        return "", 0
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"
    for x, y in pts[1:]:
        d += f" L{x:.2f},{y:.2f}"
    d += " Z"
    return d, len(pts)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  The vectorizer
# ═══════════════════════════════════════════════════════════════════════════

def vectorize_if_simple(
    asset: Any,
    *,
    stroke_color: str = "#4A4A4A",
    fill_color: str = "#FFFFFF",
) -> VectorizeStats:
    """Vectorize a SpriteAsset in place if it is simple; else leave it raster.

    Mutates the asset: on success sets asset.format = "svg" and asset.image_b64
    = inline SVG markup (a <g> of <path>s sized to the sprite's pixel box, so
    M214's renderSprite scales it into the node bbox just like a raster).

    Args:
        asset: a SpriteAsset (duck-typed: image_b64, true_bbox, format,
               dropped). Must be a non-dropped PNG.
        stroke_color / fill_color: academic line-art palette (matches
            to-svg.ts STROKE_COLOR / NODE_FILL) so vector sprites blend in.

    Returns:
        VectorizeStats describing the decision.
    """
    if getattr(asset, "dropped", False) or not getattr(asset, "image_b64", None):
        return VectorizeStats(vectorized=False, reason="dropped/empty")
    if getattr(asset, "format", "png") != "png":
        return VectorizeStats(vectorized=False, reason="not raster")

    try:
        import numpy as np
        from PIL import Image
        from skimage import measure
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("vectorize deps unavailable: %s", e)
        return VectorizeStats(vectorized=False, reason=f"deps: {e}")

    try:
        img = Image.open(io.BytesIO(base64.b64decode(asset.image_b64))).convert("RGBA")
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3].astype("float32") / 255.0

        # Color variance over opaque pixels → reject gradient-heavy sprites.
        opaque = arr[:, :, 3] > 64
        if opaque.sum() < 16:
            return VectorizeStats(vectorized=False, reason="too few opaque px")
        rgb = arr[:, :, :3][opaque].astype("float32")
        color_var = float(rgb.var(axis=0).mean())
        if color_var > COLOR_VARIANCE_MAX:
            return VectorizeStats(vectorized=False, reason="gradient/complex",
                                  color_variance=color_var)

        # Marching-squares contours of the alpha silhouette (deterministic).
        contours = measure.find_contours(alpha, ALPHA_LEVEL)
        if not contours:
            return VectorizeStats(vectorized=False, reason="no contour")
        if len(contours) > MAX_CONTOURS:
            return VectorizeStats(vectorized=False, reason="too many contours",
                                  contours=len(contours))

        # Largest-first so the outer silhouette is the primary fill.
        contours = sorted(contours, key=lambda c: len(c), reverse=True)
        path_ds: List[str] = []
        total_pts = 0
        for c in contours:
            d, n = _contour_to_path(c, RDP_EPSILON)
            if d:
                path_ds.append(d)
                total_pts += n
        if total_pts > MAX_POINTS or not path_ds:
            return VectorizeStats(vectorized=False, reason="too many points",
                                  contours=len(contours), points=total_pts,
                                  color_variance=color_var)

        # Emit inline SVG markup whose path coordinates are in the sprite's
        # native pixel space (0..w, 0..h). M214's renderSprite wraps this in a
        # translate()+scale() computed from true_bbox, so no viewBox is needed
        # here — the parent transform maps pixel space into the node box.
        # fill-rule evenodd lets inner contours punch holes (donut shapes);
        # non-scaling-stroke keeps line weight constant under that scale.
        paths = "".join(
            f'<path d="{d}" fill="{fill_color}" fill-rule="evenodd" '
            f'stroke="{stroke_color}" stroke-width="1.2" '
            f'vector-effect="non-scaling-stroke"/>'
            for d in path_ds
        )
        markup = f'<g data-vectorized="1">{paths}</g>'

        asset.image_b64 = markup
        asset.format = "svg"
        # true_bbox stays the pixel box; renderSprite scales by it.
        asset.true_bbox = (0, 0, w, h)
        logger.info("Vectorized sprite %s: %d contours, %d points",
                    getattr(asset, "node_id", "?"), len(path_ds), total_pts)
        return VectorizeStats(vectorized=True, reason="ok",
                              contours=len(path_ds), points=total_pts,
                              color_variance=color_var)

    except Exception as e:  # pragma: no cover - defensive
        logger.debug("vectorize failed, keeping raster: %s", e)
        return VectorizeStats(vectorized=False, reason=f"error: {e}")


def vectorize_assets(assets: List[Any], **kw) -> List[VectorizeStats]:
    """Apply vectorize_if_simple across a list of assets, returning per-asset
    stats.  Failures keep the raster (never drops a sprite)."""
    return [vectorize_if_simple(a, **kw) for a in assets]
