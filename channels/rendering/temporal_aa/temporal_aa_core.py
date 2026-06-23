"""
[ASTRO-CELL] Temporal Anti-Aliasing — Python port

Ported from:
  upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/TemporalAA.cpp
  upstream/unreal-renderer-ue5/Renderer-Private/SceneVisibility.cpp

Key UE5 constructs → Astro equivalents
─────────────────────────────────────────────────────────────────────────────
  FTemporalAAHistory              → AstroCellTAAHistory
  ComputeTemporalJitter()         → halton_jitter()
  FTAAStandaloneCS                → resolve_temporal_aa()
  TemporalAAMain (shader)         → _blend_with_history()
  NeighborhoodClamp()             → _neighborhood_clamp()
  CalculateVelocityForReproject   → _reproject_cell()
  TemporalAAGhostingDetection     → _detect_ghosting()

2-D SVG adaptation:
  Projection matrix jitter → per-cell bbox sub-pixel offset (Halton 2,3)
  Motion vector buffer     → cell displacement dict  (dx, dy)
  Color buffer texels      → per-cell radiance tuples (R, G, B)
  3×3 neighborhood kernel  → topological adjacency from BVH / edge list

鲁迅式：时域抗锯齿是记忆的艺术——
当前帧的真相永远不完整，只有与过去的帧重叠，
锯齿才会在时间的河流中被冲刷干净。
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants — mirrors r.TemporalAA.* CVars
# ─────────────────────────────────────────────────────────────────────────────
_TAA_SAMPLES: int          = 8       # r.TemporalAASamples (Halton cycle length)
_TAA_HISTORY_WEIGHT: float = 0.9     # α blend toward history (higher = more stable)
_TAA_CLAMP_GAMMA: float    = 1.0     # neighborhood clamp sharpness (1.0 = tight)
_TAA_VELOCITY_WEIGHT: float = 0.5    # motion weighting for ghosting rejection
_TAA_GHOSTING_THRESHOLD: float = 0.1 # luminance delta above which ghosting is detected
_TAA_FLICKER_THRESHOLD: float  = 0.04  # min luma variance for flicker suppression
_TAA_INVERT_JITTER_X: bool = False   # r.TemporalAA.InvertJitterX
_TAA_INVERT_JITTER_Y: bool = False   # r.TemporalAA.InvertJitterY
_TAA_ENABLED: bool         = True    # r.TemporalAA.Enable


def _dbg(msg: str) -> None:
    if os.environ.get("ASTRO_TAA_VERBOSE", "0") == "1":
        print(f"[ASTRO-TAA] {msg}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Halton low-discrepancy sequence — sub-pixel jitter source
# Mirrors FHaltonSequence in SceneVisibility.cpp
# ─────────────────────────────────────────────────────────────────────────────

def _halton(index: int, base: int) -> float:
    """
    Halton sequence value for 1-based *index* in the given *base*.

    鲁迅式：低差异序列是公平的采样——
    不偏好任何方向，不重复任何位置，
    每一帧都看向真相的一个新角落。
    """
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def halton_jitter(frame_index: int) -> tuple[float, float]:
    """
    Compute sub-pixel jitter offset for the current frame.

    Mirrors ComputeTemporalJitteredProjectionMatrix():
        jx = Halton(2, frame % TAASamples) - 0.5
        jy = Halton(3, frame % TAASamples) - 0.5

    Returns (jitter_x, jitter_y) in [-0.5, 0.5] pixel-space offsets.

    鲁迅式：抖动是对完美的追求——
    承认每一帧都有偏差，才能在累积中逼近真实。
    """
    sample_idx = (frame_index % _TAA_SAMPLES) + 1  # Halton is 1-based

    jx = _halton(sample_idx, 2) - 0.5
    jy = _halton(sample_idx, 3) - 0.5

    if _TAA_INVERT_JITTER_X:
        jx = -jx
    if _TAA_INVERT_JITTER_Y:
        jy = -jy

    return (round(jx, 6), round(jy, 6))


# ─────────────────────────────────────────────────────────────────────────────
# TAA History — per-view temporal accumulation state
# Mirrors FTemporalAAHistory in TemporalAA.h
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellTAAHistory:
    """
    Accumulation buffer for temporal anti-aliasing.

    Stores the previous frame's resolved colour per cell, along with
    the jitter offset and motion state needed for reprojection.

    鲁迅式：历史帧是时间的沉淀——
    不是每一帧都值得记住，但没有记忆就没有抗锯齿。
    """

    def __init__(self) -> None:
        # Previous frame resolved colour: cell_id → (r, g, b)
        self.color_buffer: dict[str, tuple] = {}
        # Previous frame cell positions: cell_id → (cx, cy)
        self.position_buffer: dict[str, tuple] = {}
        # Previous frame jitter offset
        self.prev_jitter: tuple[float, float] = (0.0, 0.0)
        # Frame counter (monotonic)
        self.frame_count: int = 0
        # Valid flag — False until at least one frame has been accumulated
        self.is_valid: bool = False

    def invalidate(self) -> None:
        """Reset all history (scene change, camera cut, etc.)."""
        self.color_buffer.clear()
        self.position_buffer.clear()
        self.prev_jitter = (0.0, 0.0)
        self.frame_count = 0
        self.is_valid = False
        _dbg("history invalidated")


# Module-level history singletons — one per view_id
_TAA_HISTORIES: dict[str, AstroCellTAAHistory] = {}


def get_taa_history(view_id: str = "default") -> AstroCellTAAHistory:
    """Return the TAA history for *view_id*, creating on first access."""
    if view_id not in _TAA_HISTORIES:
        _TAA_HISTORIES[view_id] = AstroCellTAAHistory()
    return _TAA_HISTORIES[view_id]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _luminance(c: tuple) -> float:
    """Rec. 709 luminance from (R, G, B) tuple."""
    return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    """Linear interpolate two RGB tuples by factor t."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _reproject_cell(
    cell_id: str,
    history: AstroCellTAAHistory,
    motion_vectors: dict[str, tuple],
    current_jitter: tuple[float, float],
) -> tuple | None:
    """
    Fetch the reprojected history colour for *cell_id*.

    Applies motion-vector displacement and jitter de-offset to locate the
    correct history sample.  Returns None if no valid history exists.

    Mirrors CalculateVelocityForReproject + SampleHistory in TemporalAA.usf.

    鲁迅式：重投影是记忆的校准——
    历史帧中的自己已经不在原来的位置，必须追上去才能对齐。
    """
    if not history.is_valid or cell_id not in history.color_buffer:
        return None

    mv = motion_vectors.get(cell_id, (0.0, 0.0))
    prev_pos = history.position_buffer.get(cell_id)

    if prev_pos is None:
        # No positional history — accept the colour directly (static cell)
        return history.color_buffer[cell_id]

    # Displacement magnitude — large motion reduces confidence
    disp = math.sqrt(mv[0] ** 2 + mv[1] ** 2)

    # Jitter de-offset: undo previous frame jitter, apply current
    jitter_delta_x = current_jitter[0] - history.prev_jitter[0]
    jitter_delta_y = current_jitter[1] - history.prev_jitter[1]
    total_disp = disp + math.sqrt(jitter_delta_x ** 2 + jitter_delta_y ** 2)

    # Confidence falls off with displacement (sigmoid curve)
    confidence = 1.0 / (1.0 + total_disp * _TAA_VELOCITY_WEIGHT)

    if confidence < 0.1:
        return None  # History too unreliable — force current-frame fallback

    return history.color_buffer[cell_id]


def _neighborhood_clamp(
    reprojected: tuple,
    current: tuple,
    neighbors: list[tuple],
) -> tuple:
    """
    Clamp the reprojected history colour to the min/max of the current-frame
    3×3 neighbourhood (AABB clamp in RGB space).

    Mirrors NeighborhoodClampRGB() in TemporalAA.usf — the primary
    mechanism for preventing ghosting artifacts from stale history.

    鲁迅式：邻域钳制是真实的守门人——
    历史可以提供稳定性，但不能偏离当前现实太远。
    """
    # Build neighbourhood AABB from current cell + its topological neighbours
    all_colors = [current] + neighbors
    min_r = min(c[0] for c in all_colors)
    min_g = min(c[1] for c in all_colors)
    min_b = min(c[2] for c in all_colors)
    max_r = max(c[0] for c in all_colors)
    max_g = max(c[1] for c in all_colors)
    max_b = max(c[2] for c in all_colors)

    # Expand AABB slightly by _TAA_CLAMP_GAMMA to allow mild temporal smoothing
    # beyond the strict neighbourhood range (mirrors GTemporalAAFilterSize)
    expand = (1.0 - _TAA_CLAMP_GAMMA) * 0.1
    range_r = (max_r - min_r) * expand
    range_g = (max_g - min_g) * expand
    range_b = (max_b - min_b) * expand

    clamped = (
        max(min_r - range_r, min(max_r + range_r, reprojected[0])),
        max(min_g - range_g, min(max_g + range_g, reprojected[1])),
        max(min_b - range_b, min(max_b + range_b, reprojected[2])),
    )
    return clamped


def _detect_ghosting(
    current: tuple,
    reprojected: tuple,
    motion_vectors: dict[str, tuple],
    cell_id: str,
) -> float:
    """
    Compute a ghosting rejection weight in [0, 1].

    Returns 0.0 when no ghosting is detected (full history blend),
    returns 1.0 when heavy ghosting demands current-frame dominance.

    Detection criteria (mirrors TemporalAAGhostingDetection):
      1. Luminance delta between reprojected history and current frame
      2. Motion vector magnitude — fast-moving cells ghost more easily
      3. Disocclusion heuristic — cells that appeared this frame have no
         valid history and should not blend

    鲁迅式：幽灵检测是对虚假记忆的清算——
    过去的帧留下的痕迹，如果不属于现在，就必须被清除。
    """
    luma_cur = _luminance(current)
    luma_rep = _luminance(reprojected)
    luma_delta = abs(luma_cur - luma_rep)

    # Motion magnitude contribution
    mv = motion_vectors.get(cell_id, (0.0, 0.0))
    speed = math.sqrt(mv[0] ** 2 + mv[1] ** 2)

    # Fast-moving + large luminance shift → likely ghosting
    ghost_score = luma_delta + speed * _TAA_VELOCITY_WEIGHT * 0.3

    if ghost_score > _TAA_GHOSTING_THRESHOLD:
        # Scale [threshold, threshold*3] → [0, 1]
        rejection = _clamp01(
            (ghost_score - _TAA_GHOSTING_THRESHOLD)
            / (_TAA_GHOSTING_THRESHOLD * 2.0 + 1e-6)
        )
        return rejection

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main resolve — FTAAStandaloneCS equivalent
# ─────────────────────────────────────────────────────────────────────────────

def resolve_temporal_aa(
    current_colors: dict[str, tuple],
    cell_positions: dict[str, tuple],
    motion_vectors: dict[str, tuple],
    adjacency: dict[str, list[str]],
    frame_index: int,
    view_id: str = "default",
) -> dict[str, tuple]:
    """
    Full TAA resolve pass — the main entry point.

    For each cell:
      1. Compute Halton sub-pixel jitter for the current frame
      2. Reproject the history colour using motion vectors
      3. Clamp reprojected colour to the current neighbourhood AABB
      4. Detect and suppress ghosting artifacts
      5. Blend current frame with clamped history

    Mirrors FTAAStandaloneCS::Dispatch() → TemporalAAMain() shader.

    鲁迅式：时域抗锯齿的五步——抖动、重投影、钳制、除鬼、融合——
    是对每一帧的审判：你是真实的，还是过去的残影？
    """
    if not _TAA_ENABLED or not current_colors:
        return dict(current_colors)

    history = get_taa_history(view_id)
    jitter = halton_jitter(frame_index)
    resolved: dict[str, tuple] = {}

    stats_ghosted = 0
    stats_no_history = 0
    stats_blended = 0

    for cell_id, current in current_colors.items():
        # Step 1: Reproject history
        reprojected = _reproject_cell(cell_id, history, motion_vectors, jitter)

        if reprojected is None:
            # No valid history — use current frame directly
            resolved[cell_id] = current
            stats_no_history += 1
            continue

        # Step 2: Gather neighbourhood colours for clamping
        neighbor_ids = adjacency.get(cell_id, [])
        neighbor_colors = [
            current_colors[nid]
            for nid in neighbor_ids
            if nid in current_colors
        ]
        if not neighbor_colors:
            # Isolated cell — use self as single-element neighbourhood
            neighbor_colors = [current]

        # Step 3: Neighbourhood clamp
        clamped = _neighborhood_clamp(reprojected, current, neighbor_colors)

        # Step 4: Ghosting detection
        ghost_reject = _detect_ghosting(
            current, clamped, motion_vectors, cell_id
        )

        # Step 5: Blend — α slides from _TAA_HISTORY_WEIGHT toward 0 with ghosting
        alpha = _TAA_HISTORY_WEIGHT * (1.0 - ghost_reject)

        # Flicker suppression: if luma variance across neighbourhood is tiny,
        # push blend slightly toward history to prevent sub-pixel shimmer
        if neighbor_colors:
            lumas = [_luminance(c) for c in neighbor_colors + [current]]
            luma_var = max(lumas) - min(lumas)
            if luma_var < _TAA_FLICKER_THRESHOLD:
                alpha = min(alpha + 0.05, 0.95)

        blended = _lerp_color(current, clamped, alpha)
        resolved[cell_id] = (
            round(blended[0], 6),
            round(blended[1], 6),
            round(blended[2], 6),
        )

        if ghost_reject > 0.5:
            stats_ghosted += 1
        else:
            stats_blended += 1

    # Update history for next frame
    history.color_buffer = dict(resolved)
    history.position_buffer = dict(cell_positions)
    history.prev_jitter = jitter
    history.frame_count += 1
    history.is_valid = True

    _dbg(
        f"resolve frame={frame_index} cells={len(resolved)} "
        f"blended={stats_blended} ghosted={stats_ghosted} "
        f"no_history={stats_no_history} jitter=({jitter[0]:.4f},{jitter[1]:.4f})"
    )

    return resolved
