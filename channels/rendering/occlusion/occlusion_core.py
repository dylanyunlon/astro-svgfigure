from __future__ import annotations
from dataclasses import dataclass, field
import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def _safe_rcp(v: float) -> float:
    """1/v with epsilon guard (mirrors C++ SafeRcp lambda)."""
    return 1.0 / v if abs(v) > 1e-6 else 1e6









class AstroCellOcclusionVolume:
    """
    Python equivalent of FAstroCellOcclusionVolume.

    Stores a cell's bounding box as (bounds_min_x/y/z, bounds_extent_x/y/z)
    plus an OcclusionWeight [0,1] that modulates shadow intensity — identical
    role to capsule radius scaling in the original pipeline.

    The capsule radius analogue is:
        capsule_radius ≈ min(w, h) / 2  →  BoundsExtent smallest axis
    """
    __slots__ = ("bx", "by", "bz", "ex", "ey", "ez",
                 "occlusion_weight", "cell_index")

    def __init__(self, x: float, y: float, z: float,
                 w: float, h: float,
                 occlusion_weight: float = 1.0,
                 cell_index: int = 0):
        # BoundsMin  (top-left in 2-D; z from layer data)
        self.bx = x
        self.by = y
        self.bz = z
        # BoundsExtent  (half-size — mirrors BoxExtent in FromBounds())
        self.ex = w / 2.0
        self.ey = h / 2.0
        # capsule radius analogue: min(w,h)/2 → stored as smallest half-extent
        self.ez = min(w, h) / 2.0   # Z-extent == capsule radius
        # OcclusionWeight
        self.occlusion_weight = max(0.0, min(1.0, occlusion_weight))
        self.cell_index = cell_index

    @classmethod
    def from_bbox(cls, bbox: dict, cell_index: int = 0,
                  fade_alpha: float = 1.0) -> "AstroCellOcclusionVolume":
        """Equivalent to FAstroCellOcclusionVolume::FromBounds()."""
        return cls(
            x=bbox["x"], y=bbox["y"], z=float(bbox.get("z", 3)),
            w=bbox["w"], h=bbox["h"],
            occlusion_weight=fade_alpha,
            cell_index=cell_index,
        )

    def projected_face_area(self, ld_x: float, ld_y: float,
                            ld_z: float) -> float:
        """
        ProjectedFaceArea(LightDir) — face area weighted by |dot(LightDir,N)|.

        In 2-D screen space the light direction is (0, 0, 1) (pointing down
        into the screen / toward higher Z layers), so:
            Area += |ld_x| * (ey * ez * 4)   ← YZ face pair
            Area += |ld_y| * (ex * ez * 4)   ← XZ face pair
            Area += |ld_z| * (ex * ey * 4)   ← XY face pair  ← dominant
        """
        area = (abs(ld_x) * self.ey * self.ez * 4.0
                + abs(ld_y) * self.ex * self.ez * 4.0
                + abs(ld_z) * self.ex * self.ey * 4.0)
        return area * self.occlusion_weight









def project_cell_occlusion(
    receiver_z: float,
    occluder: "AstroCellOcclusionVolume",
    max_distance: float = _CAPSULE_MAX_DIST,
) -> float:
    """
    Python equivalent of ProjectCellOcclusion().

    Receiver is the current cell (we only need its Z for 1-D slab test).
    Occluder is another cell's volume.  Light direction is (0, 0, 1)
    — straight down through Z layers (screen-space top-light convention).

    Returns occlusion factor ∈ [0, 1].
    """
    # Light direction (normalised) — pointing along +Z in layer space
    ld_x, ld_y, ld_z = 0.0, 0.0, 1.0

    # ── Step 1: slab intersection along the light ray ────────────────────────
    # Ray: P(t) = (receiver_x, receiver_y, receiver_z) + t * (0, 0, 1)
    # We only track the Z axis for the 1-D slab (X/Y overlap tested separately)
    occ_z_min = occluder.bz
    occ_z_max = occluder.bz + occluder.ez * 2.0

    t_min_z = (occ_z_min - receiver_z) * _safe_rcp(ld_z)
    t_max_z = (occ_z_max - receiver_z) * _safe_rcp(ld_z)
    if t_min_z > t_max_z:
        t_min_z, t_max_z = t_max_z, t_min_z

    t_enter = t_min_z
    t_exit = t_max_z

    # Miss or behind receiver
    if t_exit < 0.0 or t_enter > t_exit:
        return 0.0

    hit_distance = max(t_enter, 0.0)
    if hit_distance > max_distance:
        return 0.0

    # ── Step 2: projected face area as solid-angle proxy ────────────────────
    face_area = occluder.projected_face_area(ld_x, ld_y, ld_z)
    dist_sq = max(hit_distance * hit_distance, 1.0)
    solid_angle = face_area / dist_sq

    # ── Step 3: distance falloff (linear fade at 80 %→100 % of max_distance) ─
    dist_fade = max(0.0, min(1.0,
        (max_distance - hit_distance) / max(max_distance * 0.2, 1.0)
    ))

    occlusion_factor = max(0.0, min(1.0,
        solid_angle * _REFERENCE_AREA * dist_fade
    ))
    return occlusion_factor * occluder.occlusion_weight









def compute_crowding_opacity(cell_id: str, bbox: dict, all_bboxes: dict) -> float:
    """
    Python equivalent of FAstroConstraintAO::ComputeConstraintWeight().

    Samples the neighbour bbox set (= SSAO kernel), computes the crowding
    fraction, applies 3-pass constraint-space attenuation, and returns a
    fill-opacity value in [_CROWDING_OPACITY_FLOOR, 1.0].

    High density (many overlapping neighbours) → low opacity (cell recedes).
    Low density (sparse layout) → opacity near 1.0 (cell fully visible).

    @param cell_id    ID of the cell being rendered (excluded from its own kernel)
    @param bbox       Receiver cell bbox dict  {x, y, w, h, z}
    @param all_bboxes Dict of all sibling bbox dicts keyed by cell_id
    @return           fill opacity in [_CROWDING_OPACITY_FLOOR, 1.0]
    """
    receiver_z = float(bbox.get("z", 3))
    rx0 = bbox["x"]
    ry0 = bbox["y"]
    rx1 = rx0 + bbox["w"]
    ry1 = ry0 + bbox["h"]
    self_area = max(bbox["w"] * bbox["h"], 1.0)

    # ── Pass 1: neighbour bbox sampling (SSAO kernel sample loop) ────────────
    # Each neighbour that overlaps the receiver in 2-D is one "occluded sample".
    # We weight by intersection area / self_area (≈ ProximityFade × HorizonWeight
    # in the C++ hemisphere kernel).
    raw_occlusion_sum: float = 0.0
    occluded_count: int = 0
    total_samples: int = 0

    for other_id, other_bbox in all_bboxes.items():
        if other_id == cell_id:
            continue

        total_samples += 1

        other_z = float(other_bbox.get("z", 3))
        # Z-proximity fade: distant layers contribute diminishing pressure.
        # Mirrors ProximityFade = clamp(DepthDelta / KernelRadius, 0, 1).
        z_dist = abs(other_z - receiver_z)
        z_fade = max(0.0, 1.0 - z_dist / max(_CAPSULE_MAX_DIST, 1e-6))
        if z_fade <= 0.0:
            continue

        ox0 = other_bbox["x"]
        oy0 = other_bbox["y"]
        ox1 = ox0 + other_bbox["w"]
        oy1 = oy0 + other_bbox["h"]

        # 2-D AABB overlap test (horizon-angle analogue: does sample sit "above"?)
        inter_w = max(0.0, min(rx1, ox1) - max(rx0, ox0))
        inter_h = max(0.0, min(ry1, oy1) - max(ry0, oy0))
        if inter_w <= 0.0 or inter_h <= 0.0:
            continue  # sample not occluding — below horizon in SSAO terms

        # Intersection area / self area → HorizonWeight × ProximityFade analogue
        horizon_weight = min(1.0, (inter_w * inter_h) / self_area)
        raw_occlusion_sum += horizon_weight * z_fade
        occluded_count += 1

    if total_samples == 0:
        return 1.0  # no neighbours — fully lit, no crowding

    normalised_occlusion = raw_occlusion_sum / float(total_samples)
    occluded_fraction = float(occluded_count) / float(total_samples)

    # ── Pass 2: crowding attenuation ─────────────────────────────────────────
    # Mirrors Value[4].w = AstroCrowdingScale = 1/(threshold*curve).
    #   CrowdingExcess  = clamp((f − threshold) / (1 − threshold), 0, 1)
    #   AttenuationMult = 1 − CrowdingExcess ** AttenuationCurve
    attenuation_mult = 1.0
    if occluded_fraction > _CROWDING_THRESHOLD:
        threshold_range = max(1.0 - _CROWDING_THRESHOLD, 1e-6)
        crowding_excess = min(1.0, (occluded_fraction - _CROWDING_THRESHOLD)
                              / threshold_range)
        attenuation_mult = 1.0 - (crowding_excess ** _ATTENUATION_CURVE)

    # ── Pass 3: mutual-constraint cancellation ────────────────────────────────
    # Geometric mean restores [0,0.5]→[0,1]; scaled by attenuation_mult.
    # BlendAlpha = saturate(occluded_fraction / threshold) — lerps between raw
    # SSAO (sparse) and constraint-space AO (crowded).
    constraint_occlusion = (
        math.sqrt(max(0.0, normalised_occlusion * (1.0 - normalised_occlusion)))
        * 2.0
        * attenuation_mult
    )
    blend_alpha = min(1.0, occluded_fraction
                      / max(_CROWDING_THRESHOLD, 1e-6))
    final_ao = (1.0 - blend_alpha) * normalised_occlusion \
               + blend_alpha * constraint_occlusion

    # AO weight → fill opacity (invert: high AO = lower opacity)
    fill_opacity = max(_CROWDING_OPACITY_FLOOR, 1.0 - final_ao)
    return round(fill_opacity, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] NaniteVisibility → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteVisibility.cpp
#
# FNaniteVisibilityQuery (→ AstroCellVisibilityQuery):
#   Per-frame query that tests every cell primitive against the viewport
#   frustum (IntersectBox) and accumulates species-bin visibility bitmasks.
#   Cells outside the viewport are marked invisible; their proc() SVG
#   generation is skipped entirely (LOD-0 = nothing) saving compute.
#
# Algorithm changes from Nanite original:
#   1. FConvexVolume frustum → simple AABB viewport rect (2-D screen)
#   2. RasterBin → species bin (cells of same species share a render bin)
#   3. ShadingBin → z-layer bin (cells at same z-layer share a shading bin)
#   4. Async task → synchronous (single-threaded epoch loop)
#   5. PrimitiveSceneInfo → cell_registry.json entries
#
# Reference: [ASTRO-NANITE-VIS] debug prefix preserved.
# ═══════════════════════════════════════════════════════════════════════════════

# Viewport margins — cells partially outside get LOD reduction, not hard cull.
# Mirrors the guard-band concept in Nanite's rasteriser.
_NANITE_VIS_GUARD_BAND: float = 50.0

# LOD thresholds: screen-projected area below which a cell drops LOD.
# Mirrors Nanite's streaming LOD metric: ScreenSize = ProjectedArea / ViewArea.
# Below _LOD1_THRESHOLD the cell is rendered at reduced detail (fewer SVG
# elements in its species generator); below _LOD2_THRESHOLD it becomes a
# simple coloured rect; below _CULL_THRESHOLD it is culled entirely.
_NANITE_LOD2_THRESHOLD: float = 0.002   # < 0.2% of viewport area → rect
_NANITE_CULL_THRESHOLD: float = 0.0005  # < 0.05% of viewport area → culled









def cone_trace_object_occlusion(
    ray_origin: Tuple[float, float, float],
    cone_dir: Tuple[float, float, float],
    tan_half_angle: float,
    max_distance: float,
    objects: List[DFObjectBounds],
    traverse_mips: bool = True,
) -> float:
    """
    对应 FConeTraceScreenGridObjectOcclusionCS 的单条 cone trace。
    沿锥方向步进，累积遮蔽量。步长按指数增长（GAOStepExponentScale）。
    快到终点时步子已经很大，精度换速度；起点附近小步谨慎前行。
    """
    step_scale = G_AO_STEP_EXPONENT_SCALE
    t = 0.5  # 起始偏移避免自遮蔽
    occlusion = 0.0
    step = 0.5

    while t < max_distance and occlusion < 1.0:
        min_sdf = max_distance
        for obj in objects:
            sdf_val = _sample_distance_field(obj, ray_origin, cone_dir, t)
            min_sdf = min(min_sdf, sdf_val)

        cone_radius = t * tan_half_angle
        if cone_radius > 1e-6:
            occ_contribution = max(0.0, 1.0 - min_sdf / cone_radius)
            occlusion = max(occlusion, occ_contribution)

        step = max(step * (1.0 + step_scale), min_sdf * 0.5)
        if traverse_mips:
            step = max(step, t * 0.01)
        t += step

    return min(occlusion, 1.0)









class AstroCellLightShaftOcclusion:
    """
    Python equivalent of the occlusion light shaft technique (ELightShaftTechnique::Occlusion).

    The occlusion technique renders a soft mask that darkens regions of the
    scene that are «behind» the light source from the camera's perspective —
    the «god-ray» darkening that appears where the shaft is blocked.

    In SVG, the occlusion is implemented as a feFlood + feComposite alpha mask
    that applies a gradient darkening centred on the cell's screen position,
    with darkness = OcclusionMaskDarkness and radial falloff.

    鲁迅式：遮挡光柱是光的反面——不是看到光，而是看到光的缺席，
    那些被光柱扫过却依然黑暗的地方，比光柱本身更令人深思。
    """

    def __init__(self,
                 params:  dict,
                 cell_id: str,
                 bbox:    dict) -> None:
        self._p       = params
        self._cell_id = cell_id
        self._bbox    = bbox

    def emit_svg(self) -> str:
        """
        Emit the occlusion mask SVG fragment.

        Creates a radial gradient mask centred on the light shaft origin,
        with opacity = 1 − OcclusionMaskDarkness at the centre and fading
        to 1.0 at the edges (full brightness away from the shaft).

        Returns SVG string.

        鲁迅式：遮挡蒙版是光柱的负像——用梯度来记录光的路径，
        用暗处来证明光曾经经过这里。
        """
        p         = self._p
        cell_id   = self._cell_id
        bbox      = self._bbox
        w         = float(bbox.get("w", 100))
        h         = float(bbox.get("h", 50))
        cx        = bbox.get("x", 0) + w / 2.0
        cy        = bbox.get("y", 0) + h / 2.0

        darkness  = round(p["occlusion_mask_darkness"], 3)
        grad_id   = f"ls-occ-grad-{cell_id}"
        mask_id   = f"ls-occ-mask-{cell_id}"

        # Radial gradient: dark at centre (light source), bright at edges
        r_grad = max(w, h) * 0.7
        parts = [
            f'<!-- [ASTRO-LS] LightShaftRendering.cpp Occlusion port '
            f'darkness={darkness} depth_range={p["occlusion_depth_range"]:.1f} -->',
            f'<defs>',
            f'  <radialGradient id="{grad_id}" '
            f'cx="{cx:.1f}" cy="{cy:.1f}" r="{r_grad:.1f}" '
            f'gradientUnits="userSpaceOnUse">',
            f'    <stop offset="0" stop-color="black" '
            f'stop-opacity="{darkness:.3f}"/>',
            f'    <stop offset="1" stop-color="black" stop-opacity="0"/>',
            f'  </radialGradient>',
            f'  <mask id="{mask_id}">',
            f'    <rect x="{bbox.get("x",0):.1f}" y="{bbox.get("y",0):.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" '
            f'fill="url(#{grad_id})" opacity="0.6"/>',
            f'  </mask>',
            f'</defs>',
            f'<!-- [ASTRO-LS] Occlusion mask: apply mask="url(#{mask_id})" '
            f'to a rect over the cell to dim the light shaft region -->',
        ]
        return "\n".join(parts)




