# =============================================================================
# [ASTRO-CELL] MotionBlur — Per-Pixel Motion Vectors & Directional Blur
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/PostProcessMotionBlur.cpp
#
# 鲁迅曾言：「时间就是性命。无端的空耗别人的时间，
# 其实是无异于谋财害命的。」
# 运动模糊亦然——帧间位移是时间的刻痕，沿速度方向展开的涂抹
# 让静止的像素流淌出运动的生命力。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FVelocityFlattenTextures           → AstroCellVelocityField
#   PostProcessMotionBlur.cpp          → motionblur_core.py
#   VelocityFlattenCS                  → flatten_velocity_tiles()
#   MotionBlurVelocityDilateCS         → dilate_tile_velocity()
#   MotionBlurFilterCS (gather/scatter)→ apply_directional_blur()
#   EncodeVelocityToTexture            → encode_motion_vector()
#   DecodeVelocityFromTexture          → decode_motion_vector()
#   GetMotionBlurTileCount()           → get_tile_count()
#   IsMotionBlurScatterRequired()      → is_scatter_required()
#
# Algorithm:
#   1. Compute per-pixel motion vectors from particle prev/curr positions
#   2. Flatten velocity into tiles (max magnitude per tile)
#   3. Dilate tile velocity to neighbours (leak bleeding edges)
#   4. Directional gather blur along velocity vector per pixel
#   5. Pack results for SVG feGaussianBlur or PixiJS MotionBlurFilter
# =============================================================================
from __future__ import annotations
import os, sys, math
from typing import Any, Optional


def _dbg(tag: str, msg: str) -> None:
    if os.environ.get(f"ASTRO_{tag.replace('-', '_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)


# ── MotionBlur CVars (mirrors r.MotionBlur* console variables) ───────────────
_MB_QUALITY:           int   = 3       # r.MotionBlurQuality [0=off,1=low,2=med,3=high,4=vis]
_MB_AMOUNT:            float = 1.0     # PostProcessSettings.MotionBlurAmount
_MB_MAX_VELOCITY:      float = 0.05    # PostProcessSettings.MotionBlurMax (fraction of viewport)
_MB_FLATTEN_TILE_SIZE: int   = 16      # kMotionBlurFlattenTileSize
_MB_FILTER_TILE_SIZE:  int   = 16      # kMotionBlurFilterTileSize
_MB_SCATTER_THRESHOLD: float = 3.0     # tiles beyond this use scatter path
_MB_HALF_RES:          bool  = True    # r.MotionBlur.HalfResInput
_MB_DIRECTIONS:        int   = 1       # r.MotionBlur.Directions (1 or 2)
_MB_SECOND_SCALE:      float = 1.0     # r.MotionBlur2ndScale
_MB_SAMPLE_COUNT_MAP:  dict  = {0: 0, 1: 4, 2: 6, 3: 8, 4: 8}  # quality → samples


# ── Viewport reference dimensions (mirrors FSceneView extent) ────────────────
_VP_WIDTH:  float = 1920.0
_VP_HEIGHT: float = 1080.0


def is_motion_blur_enabled() -> bool:
    """
    Gate check — mirrors IsMotionBlurEnabled() from PostProcessMotionBlur.cpp.

    Disabled when quality == 0, amount ≈ 0, or max velocity ≈ 0.

    鲁迅式：开关是最简单的决策——但多少系统死在「忘记检查开关」上。
    """
    if _MB_QUALITY <= 0:
        return False
    if _MB_AMOUNT < 1e-4:
        return False
    if _MB_MAX_VELOCITY < 1e-6:
        return False
    return True


def get_sample_count() -> int:
    """Return directional blur sample count for current quality level."""
    return _MB_SAMPLE_COUNT_MAP.get(_MB_QUALITY, 8)


def get_tile_count(pixel_extent: float) -> int:
    """
    Number of tiles along one axis — mirrors GetMotionBlurTileCount().

    鲁迅式：分块是空间管理的第一步——
    不分块，就无法区分「快」和「慢」的区域。
    """
    return max(1, math.ceil(pixel_extent / _MB_FLATTEN_TILE_SIZE))


# ─────────────────────────────────────────────────────────────────────────────
# Motion vector computation — per-pixel velocity from particle positions
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_vector(
    prev_x: float, prev_y: float,
    curr_x: float, curr_y: float,
    dt: float = 1.0,
) -> tuple[float, float]:
    """
    Compute a single per-pixel motion vector from previous and current position.

    Mirrors EncodeVelocityToTexture(): velocity = (curr − prev) / dt,
    then scaled by _MB_AMOUNT and clamped to _MB_MAX_VELOCITY × viewport.

    Parameters
    ----------
    prev_x, prev_y : float
        Particle position at previous frame (pixel coords).
    curr_x, curr_y : float
        Particle position at current frame (pixel coords).
    dt : float
        Frame delta time (default 1.0 for per-frame displacement).

    Returns
    -------
    tuple[float, float]
        (vx, vy) motion vector in pixel/frame units, clamped to max velocity.
    """
    if dt < 1e-8:
        return (0.0, 0.0)

    raw_vx = (curr_x - prev_x) / dt
    raw_vy = (curr_y - prev_y) / dt

    # Scale by user amount (mirrors MotionBlurAmount multiplier)
    vx = raw_vx * _MB_AMOUNT
    vy = raw_vy * _MB_AMOUNT

    # Clamp magnitude to max velocity (fraction of viewport diagonal)
    max_px = _MB_MAX_VELOCITY * math.hypot(_VP_WIDTH, _VP_HEIGHT)
    mag = math.hypot(vx, vy)
    if mag > max_px and mag > 1e-8:
        scale = max_px / mag
        vx *= scale
        vy *= scale

    return (vx, vy)


def compute_motion_vectors_batch(
    particles: list[dict],
    dt: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-pixel motion vectors for a batch of particles.

    Each particle dict must contain:
        cell_id:  str   — unique identifier
        prev_x, prev_y: float  — position at t-1
        curr_x, curr_y: float  — position at t

    Returns dict mapping cell_id → (vx, vy).

    鲁迅式：批量计算是效率的表达——
    一个一个算是手工业，批量算才是工业化。
    """
    result: dict[str, tuple[float, float]] = {}
    for p in particles:
        cid = p.get("cell_id", "")
        if not cid:
            continue
        mv = compute_motion_vector(
            p.get("prev_x", 0.0), p.get("prev_y", 0.0),
            p.get("curr_x", 0.0), p.get("curr_y", 0.0),
            dt,
        )
        result[cid] = mv
    return result


def encode_motion_vector(vx: float, vy: float) -> tuple[float, float]:
    """
    Encode velocity into [0,1] texture space — mirrors EncodeVelocityToTexture().

    Maps [-max_px, +max_px] → [0, 1] with 0.5 = zero velocity.
    """
    max_px = _MB_MAX_VELOCITY * math.hypot(_VP_WIDTH, _VP_HEIGHT)
    if max_px < 1e-8:
        return (0.5, 0.5)
    enc_x = vx / (2.0 * max_px) + 0.5
    enc_y = vy / (2.0 * max_px) + 0.5
    return (max(0.0, min(1.0, enc_x)), max(0.0, min(1.0, enc_y)))


def decode_motion_vector(enc_x: float, enc_y: float) -> tuple[float, float]:
    """
    Decode velocity from [0,1] texture space — mirrors DecodeVelocityFromTexture().
    """
    max_px = _MB_MAX_VELOCITY * math.hypot(_VP_WIDTH, _VP_HEIGHT)
    vx = (enc_x - 0.5) * 2.0 * max_px
    vy = (enc_y - 0.5) * 2.0 * max_px
    return (vx, vy)


# ─────────────────────────────────────────────────────────────────────────────
# Velocity tile flatten & dilate — spatial coherence for gather/scatter decision
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellVelocityField:
    """
    Per-frame velocity field storing per-cell motion vectors and tile statistics.

    Mirrors FVelocityFlattenTextures from PostProcessMotionBlur.cpp:
    the flattened tile texture holds the max velocity magnitude per tile
    for the scatter/gather decision and neighbour dilation.

    鲁迅式：速度场是运动的地图——
    不画地图就上路，和不做速度场就做模糊一样鲁莽。
    """

    __slots__ = (
        "cell_vectors",
        "tile_max_mag",
        "tile_max_vec",
        "tile_count_x",
        "tile_count_y",
        "viewport_w",
        "viewport_h",
    )

    def __init__(
        self,
        viewport_w: float = _VP_WIDTH,
        viewport_h: float = _VP_HEIGHT,
    ) -> None:
        self.cell_vectors: dict[str, tuple[float, float]] = {}
        self.tile_max_mag: dict[tuple[int, int], float] = {}
        self.tile_max_vec: dict[tuple[int, int], tuple[float, float]] = {}
        self.tile_count_x = get_tile_count(viewport_w)
        self.tile_count_y = get_tile_count(viewport_h)
        self.viewport_w   = viewport_w
        self.viewport_h   = viewport_h


def flatten_velocity_tiles(
    velocity_field: AstroCellVelocityField,
    cell_bboxes: dict[str, dict],
) -> AstroCellVelocityField:
    """
    Flatten per-cell velocities into per-tile max velocity.

    Mirrors VelocityFlattenCS: for each tile, find the cell whose motion
    vector has the largest magnitude, store that magnitude and direction.

    This drives the scatter/gather decision — tiles with high velocity
    use scatter (more expensive but correct for long streaks), tiles
    with low velocity use gather (cheaper).

    鲁迅式：展平是抽象——用一个区域的最大速度代表整个区域，
    如同用一个时代的最大声音代表整个时代，有时准确，有时失真。
    """
    vf = velocity_field
    # Clear previous tile data
    vf.tile_max_mag.clear()
    vf.tile_max_vec.clear()

    for cid, mv in vf.cell_vectors.items():
        bbox = cell_bboxes.get(cid)
        if not bbox:
            continue

        # Cell centre → tile index
        cx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
        cy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
        tx = max(0, min(vf.tile_count_x - 1,
                        int(cx / _MB_FLATTEN_TILE_SIZE)))
        ty = max(0, min(vf.tile_count_y - 1,
                        int(cy / _MB_FLATTEN_TILE_SIZE)))

        mag = math.hypot(mv[0], mv[1])
        key = (tx, ty)
        if mag > vf.tile_max_mag.get(key, 0.0):
            vf.tile_max_mag[key] = mag
            vf.tile_max_vec[key] = mv

    _dbg("MB", f"flatten_velocity_tiles: {len(vf.tile_max_mag)} tiles populated")
    return vf


def dilate_tile_velocity(
    velocity_field: AstroCellVelocityField,
    radius: int = 1,
) -> AstroCellVelocityField:
    """
    Dilate tile max velocity to neighbours — mirrors MotionBlurVelocityDilateCS.

    A fast-moving cell at a tile boundary should bleed its blur into
    adjacent tiles.  For each tile, take the max over its (2r+1)² neighbourhood.

    鲁迅式：膨胀是慷慨——把速度的影响扩散到邻居，
    让边界上的运动不会突然截断成静止。
    """
    vf = velocity_field
    new_mag: dict[tuple[int, int], float] = {}
    new_vec: dict[tuple[int, int], tuple[float, float]] = {}

    for (tx, ty), mag in vf.tile_max_mag.items():
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = tx + dx, ty + dy
                if nx < 0 or ny < 0:
                    continue
                if nx >= vf.tile_count_x or ny >= vf.tile_count_y:
                    continue
                key = (nx, ny)
                if mag > new_mag.get(key, 0.0):
                    new_mag[key] = mag
                    new_vec[key] = vf.tile_max_vec[(tx, ty)]

    vf.tile_max_mag = new_mag
    vf.tile_max_vec = new_vec
    _dbg("MB", f"dilate_tile_velocity: {len(new_mag)} tiles after dilation r={radius}")
    return vf


def is_scatter_required(velocity_field: AstroCellVelocityField) -> bool:
    """
    Determine whether scatter path is needed — mirrors IsMotionBlurScatterRequired().

    If any tile's max velocity exceeds the scatter threshold (in tile units),
    the scatter path produces correct long-streak blurs that gather cannot.
    """
    threshold_px = _MB_SCATTER_THRESHOLD * _MB_FLATTEN_TILE_SIZE
    for mag in velocity_field.tile_max_mag.values():
        if mag > threshold_px:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Directional blur — per-pixel gather along velocity vector
# ─────────────────────────────────────────────────────────────────────────────

def _directional_blur_kernel(
    vx: float, vy: float,
    num_samples: int,
) -> list[tuple[float, float, float]]:
    """
    Build a 1-D blur kernel aligned to velocity direction.

    Returns list of (offset_x, offset_y, weight) tuples, centred at origin.
    Weights follow a truncated Gaussian envelope (σ = 0.4 × length)
    to avoid box-filter ringing.

    Mirrors the jittered sample pattern in MotionBlurFilterCS:
        for i in [-N/2 .. +N/2]:
            t = i / N
            sample_uv = pixel_uv + velocity * t
            weight = gaussian(t, sigma)
    """
    mag = math.hypot(vx, vy)
    if mag < 0.5 or num_samples < 1:
        return [(0.0, 0.0, 1.0)]

    # Normalise direction
    dx = vx / mag
    dy = vy / mag
    half_len = mag * 0.5
    sigma = 0.4 * mag

    kernel: list[tuple[float, float, float]] = []
    total_w = 0.0

    for i in range(num_samples):
        # Uniform spacing along velocity, centred
        t = (i / max(num_samples - 1, 1)) - 0.5  # range [-0.5, +0.5]
        ox = dx * half_len * 2.0 * t
        oy = dy * half_len * 2.0 * t

        # Gaussian weight
        dist = abs(t) * mag
        w = math.exp(-0.5 * (dist / sigma) ** 2) if sigma > 1e-6 else 1.0
        kernel.append((ox, oy, w))
        total_w += w

    # Normalise weights
    if total_w > 1e-8:
        kernel = [(ox, oy, w / total_w) for (ox, oy, w) in kernel]

    return kernel


def apply_directional_blur(
    cell_id: str,
    colour: tuple[float, float, float],
    motion_vec: tuple[float, float],
    neighbour_colours: dict[str, tuple[float, float, float]] | None = None,
    neighbour_bboxes: dict[str, dict] | None = None,
    cell_bbox: dict | None = None,
) -> tuple[float, float, float]:
    """
    Apply per-pixel directional blur along the motion vector for one cell.

    Mirrors the gather loop in MotionBlurFilterCS:
        for each sample along velocity direction:
            fetch colour at offset position
            accumulate weighted colour

    For cells without neighbours along the velocity direction, the blur
    degrades gracefully to self-colour (equivalent to a stationary pixel).

    Parameters
    ----------
    cell_id : str
        Current cell identifier.
    colour : tuple
        (R, G, B) of the current cell, float [0,1].
    motion_vec : tuple
        (vx, vy) velocity in pixels/frame.
    neighbour_colours : dict, optional
        cell_id → (R,G,B) for cells in the velocity neighbourhood.
    neighbour_bboxes : dict, optional
        cell_id → {x, y, w, h} for spatial lookup.
    cell_bbox : dict, optional
        {x, y, w, h} for this cell.

    Returns
    -------
    tuple[float, float, float]
        Blurred (R, G, B).

    鲁迅式：模糊是方向性的遗忘——沿着速度的方向，
    像素交出自己的身份，融入运动的痕迹。
    """
    if not is_motion_blur_enabled():
        return colour

    vx, vy = motion_vec
    mag = math.hypot(vx, vy)

    # Skip blur for near-stationary cells (< 0.5 px displacement)
    if mag < 0.5:
        return colour

    num_samples = get_sample_count()
    kernel = _directional_blur_kernel(vx, vy, num_samples)

    # If no spatial neighbourhood data, apply self-blur (streak of own colour)
    if not neighbour_colours or not neighbour_bboxes or not cell_bbox:
        return colour

    # Cell centre
    cx = float(cell_bbox.get("x", 0)) + float(cell_bbox.get("w", 0)) * 0.5
    cy = float(cell_bbox.get("y", 0)) + float(cell_bbox.get("h", 0)) * 0.5

    # Build spatial index of neighbour centres for lookup
    nb_centres: list[tuple[str, float, float, tuple]] = []
    for nid, nbbox in neighbour_bboxes.items():
        nc = neighbour_colours.get(nid)
        if nc is None:
            continue
        nx = float(nbbox.get("x", 0)) + float(nbbox.get("w", 0)) * 0.5
        ny = float(nbbox.get("y", 0)) + float(nbbox.get("h", 0)) * 0.5
        nb_centres.append((nid, nx, ny, nc))

    # Gather along kernel offsets
    acc_r, acc_g, acc_b = 0.0, 0.0, 0.0

    for (ox, oy, w) in kernel:
        sample_x = cx + ox
        sample_y = cy + oy

        # Find nearest cell to the sample point
        best_col = colour  # fallback: own colour
        best_dist = float("inf")

        # Check self first
        self_dist = math.hypot(ox, oy)
        if self_dist < best_dist:
            best_dist = self_dist
            best_col = colour

        for (nid, nx, ny, nc) in nb_centres:
            d = math.hypot(sample_x - nx, sample_y - ny)
            if d < best_dist:
                best_dist = d
                best_col = nc

        acc_r += best_col[0] * w
        acc_g += best_col[1] * w
        acc_b += best_col[2] * w

    return (acc_r, acc_g, acc_b)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — orchestrates velocity → tiles → blur for all cells
# ─────────────────────────────────────────────────────────────────────────────

def run_motion_blur_pass(
    particles: list[dict],
    cell_colours: dict[str, tuple[float, float, float]],
    cell_bboxes: dict[str, dict],
    dt: float = 1.0,
    viewport_w: float = _VP_WIDTH,
    viewport_h: float = _VP_HEIGHT,
) -> dict[str, dict]:
    """
    Full motion blur post-process pass for all cells in a single epoch.

    Orchestrates the complete pipeline matching PostProcessMotionBlur.cpp:
      1. compute_motion_vectors_batch  — per-pixel velocity from positions
      2. flatten_velocity_tiles        — tile max velocity for scatter gate
      3. dilate_tile_velocity          — bleed velocity to neighbour tiles
      4. apply_directional_blur        — per-cell gather blur along velocity
      5. Pack results for downstream SVG / PixiJS consumption

    Parameters
    ----------
    particles : list[dict]
        Each dict: {cell_id, prev_x, prev_y, curr_x, curr_y}.
    cell_colours : dict
        cell_id → (R, G, B) current-frame colour.
    cell_bboxes : dict
        cell_id → {x, y, w, h} bounding boxes.
    dt : float
        Frame delta time.
    viewport_w, viewport_h : float
        Viewport dimensions for clamping.

    Returns
    -------
    dict[str, dict]
        cell_id → {
            "colour":   (R, G, B) after blur,
            "velocity": (vx, vy),
            "magnitude": float,
            "blur_angle": float (radians),
            "blur_length": float (pixels),
            "svg_filter": dict (feGaussianBlur params),
        }

    鲁迅式：管线是流水线——每一步都在上一步的结果之上工作，
    如同写作是在阅读之上的二次创造。
    """
    if not is_motion_blur_enabled():
        # Pass-through: no blur applied
        return {
            cid: {
                "colour": col, "velocity": (0.0, 0.0),
                "magnitude": 0.0, "blur_angle": 0.0, "blur_length": 0.0,
                "svg_filter": _build_svg_filter_params(0.0, 0.0, 0.0),
            }
            for cid, col in cell_colours.items()
        }

    # Step 1: compute motion vectors
    motion_vecs = compute_motion_vectors_batch(particles, dt)

    # Step 2–3: build velocity field, flatten, dilate
    vf = AstroCellVelocityField(viewport_w, viewport_h)
    vf.cell_vectors = motion_vecs
    flatten_velocity_tiles(vf, cell_bboxes)
    dilate_tile_velocity(vf, radius=1)

    use_scatter = is_scatter_required(vf)
    _dbg("MB", f"run_motion_blur_pass: scatter_required={use_scatter}")

    # Step 4: apply directional blur per cell
    results: dict[str, dict] = {}
    for cid, col in cell_colours.items():
        mv = motion_vecs.get(cid, (0.0, 0.0))
        mag = math.hypot(mv[0], mv[1])
        angle = math.atan2(mv[1], mv[0]) if mag > 1e-6 else 0.0

        blurred = apply_directional_blur(
            cell_id=cid,
            colour=col,
            motion_vec=mv,
            neighbour_colours=cell_colours,
            neighbour_bboxes=cell_bboxes,
            cell_bbox=cell_bboxes.get(cid),
        )

        # Compute blur length for SVG/PixiJS output
        blur_len = mag * _MB_AMOUNT
        if _MB_HALF_RES:
            blur_len *= 0.5

        results[cid] = {
            "colour":      blurred,
            "velocity":    mv,
            "magnitude":   round(mag, 4),
            "blur_angle":  round(angle, 6),
            "blur_length": round(blur_len, 4),
            "svg_filter":  _build_svg_filter_params(blur_len, angle, mag),
        }

    # Logging
    total   = len(results)
    moving  = sum(1 for r in results.values() if r["magnitude"] > 0.5)
    max_mag = max((r["magnitude"] for r in results.values()), default=0.0)
    print(
        f"[ASTRO-MB] run_motion_blur_pass: "
        f"total_cells={total} moving={moving} "
        f"max_velocity={max_mag:.2f}px "
        f"scatter={'yes' if use_scatter else 'no'} "
        f"quality={_MB_QUALITY} samples={get_sample_count()}",
        file=sys.stderr,
    )

    return results


def _build_svg_filter_params(
    blur_length: float,
    angle: float,
    magnitude: float,
) -> dict:
    """
    Build SVG feGaussianBlur / PixiJS MotionBlurFilter parameters.

    For SVG: uses an anisotropic feGaussianBlur with stdDeviation
    decomposed along velocity direction (σ_major, σ_minor) plus a
    feOffset for directional shift.

    For PixiJS: returns velocity vector and kernelSize directly
    compatible with @pixi/filter-motion-blur.

    鲁迅式：参数是渲染意图的契约——
    告诉前端「模糊多少、朝哪个方向」，而不是告诉它怎么做。
    """
    if magnitude < 0.5:
        return {
            "enabled": False,
            "stdDeviation": {"x": 0.0, "y": 0.0},
            "angle_deg": 0.0,
            "pixi_velocity": [0, 0],
            "pixi_kernelSize": 5,
            "pixi_offset": 0,
        }

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # SVG feGaussianBlur: decompose blur_length into x/y stdDeviation
    # σ_x = blur_length × |cos(angle)|,  σ_y = blur_length × |sin(angle)|
    # Minimum 0.1 on minor axis to avoid degenerate filter
    sigma_x = max(0.1, blur_length * abs(cos_a))
    sigma_y = max(0.1, blur_length * abs(sin_a))

    # PixiJS MotionBlurFilter: velocity as [vx, vy], kernelSize odd ∈ [5,15]
    kernel_size = max(5, min(15, 2 * int(blur_length / 2) + 1))

    return {
        "enabled": True,
        "stdDeviation": {
            "x": round(sigma_x, 4),
            "y": round(sigma_y, 4),
        },
        "angle_deg": round(math.degrees(angle), 2),
        "pixi_velocity": [
            round(cos_a * blur_length, 4),
            round(sin_a * blur_length, 4),
        ],
        "pixi_kernelSize": kernel_size,
        "pixi_offset": round(blur_length * 0.5, 4),
    }
