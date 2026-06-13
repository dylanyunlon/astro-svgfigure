from __future__ import annotations
from dataclasses import dataclass, field
import os, sys, json, math
from typing import Any, Optional
from channels.rendering.constants import (
    _DF_SHADOW_QUALITY, _DF_FULL_RESOLUTION, _DF_TWO_SIDED_BIAS, _DF_QUALITY_STEPS,
    _CSM_DEPTH_BIAS, _CSM_SLOPE_BIAS, _CSM_RECEIVER_BIAS,
    _PCSS_MAX_KERNEL_RADIUS, _SHADOW_FILTER_METHOD, _SHADOW_TRANSITION_SCALE,
    _STENCIL_OPTIMIZATION, _CAPSULE_MAX_DIST, _ASTRO_CELL_MAX_Z_LAYERS,
)
from channels.rendering.occlusion.occlusion_core import AstroCellOcclusionVolume, project_cell_occlusion

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def compute_capsule_shadow_params(
    cell_id: str,
    bbox: dict,
    all_bboxes: dict,
) -> dict:
    """
    BuildCellOcclusionVolumes + multi-cell ProjectCellOcclusion aggregation.

    Collects every *other* cell as a potential occluder (cells at higher Z
    can block the shadow of cells at lower Z — same as UE's shadow primitives
    being culled against the receiver by depth).

    Returns a dict with:
        dx, dy      — shadow offset (pixels); scale with receiver's Z depth
        opacity     — feDropShadow flood-opacity; attenuated by occlusion
        blur        — feDropShadow stdDeviation (proportional to capsule radius)
    """
    receiver_z = float(bbox.get("z", 3))
    capsule_radius = min(bbox["w"], bbox["h"]) / 2.0

    # ── Shadow offset: z depth → offset distance ──────────────────────────
    # z=3 baseline → offset_scale=1; higher z → larger shadow below
    z_baseline = 3.0
    z_scale = max(0.0, receiver_z - z_baseline)
    # Offset direction: bottom-right (45° light, classic drop shadow)
    base_offset = 2.0 + z_scale * 3.0
    shadow_dx = base_offset
    shadow_dy = base_offset

    # Blur radius ∝ capsule radius (same as penumbra width from capsule r)
    blur = max(1.0, capsule_radius * 0.15 + z_scale * 0.8)

    # ── Occlusion accumulation across all other cells ─────────────────────
    total_occlusion = 0.0
    receiver_vol = AstroCellOcclusionVolume.from_bbox(bbox, cell_index=0)

    for idx, (other_id, other_bbox) in enumerate(all_bboxes.items()):
        if other_id == cell_id:
            continue
        other_z = float(other_bbox.get("z", 3))
        # Only cells at strictly higher Z can occlude this cell's shadow
        if other_z <= receiver_z:
            continue

        occluder = AstroCellOcclusionVolume.from_bbox(
            other_bbox, cell_index=idx + 1,
            # FadeAlpha: proximity-based — cells farther in Z contribute less
            fade_alpha=max(0.0, 1.0 - abs(other_z - receiver_z) / _CAPSULE_MAX_DIST),
        )

        # X/Y overlap check (cells must overlap horizontally to cast shadow)
        overlap_x = (receiver_vol.bx < occluder.bx + occluder.ex * 2.0 and
                     occluder.bx < receiver_vol.bx + receiver_vol.ex * 2.0)
        overlap_y = (receiver_vol.by < occluder.by + occluder.ey * 2.0 and
                     occluder.by < receiver_vol.by + receiver_vol.ey * 2.0)

        if overlap_x and overlap_y:
            occ = project_cell_occlusion(
                receiver_z=receiver_z,
                occluder=occluder,
            )
            total_occlusion = min(1.0, total_occlusion + occ)

    # Attenuate shadow opacity by accumulated occlusion
    # Fully occluded cell → nearly invisible shadow (floor at 0.08)
    base_opacity = 0.35 + min(0.3, z_scale * 0.1)
    shadow_opacity = max(0.08, base_opacity * (1.0 - total_occlusion * 0.8))

    return {
        "dx": round(shadow_dx, 2),
        "dy": round(shadow_dy, 2),
        "blur": round(blur, 2),
        "opacity": round(shadow_opacity, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PostProcessAmbientOcclusion → Python port  (commit 33e27b7)
#
# Ported from FAstroConstraintAO / PostProcessAmbientOcclusion.cpp:
#   upstream/unreal-renderer/CompositionLighting/PostProcessAmbientOcclusion.cpp
#
# FAstroConstraintAOParams (→ _CROWDING_THRESHOLD, _ATTENUATION_CURVE):
#   KernelRadius        — analogue: bbox sampling radius (not needed in 2-D;
#                         every overlapping neighbour is one kernel "sample")
#   CrowdingThreshold   — fraction of occluded samples above which crowding
#                         attenuation begins.  Default 0.5 (AmbientOcclusionMipBlend)
#   AttenuationCurve    — power of the suppression curve.  Default 2.0
#                         (AmbientOcclusionPower).
#
# FAstroConstraintAO::ComputeConstraintWeight (→ compute_crowding_opacity()):
#   Pass 1 — raw overlap accumulation
#       Each neighbour bbox that overlaps the receiver cell in 2-D contributes
#       one "occluded sample".  raw_occlusion = occluded_count / total_neighbours.
#       (SSAO analogue: hemisphere depth samples above shaded surface.)
#
#   Pass 2 — crowding attenuation
#       When occluded_fraction > CrowdingThreshold a suppression multiplier is
#       computed (mirrors AstroCrowdingScale packed into Value[4].w):
#           CrowdingExcess  = clamp((f − threshold) / (1 − threshold), 0, 1)
#           AttenuationMult = 1 − CrowdingExcess ** AttenuationCurve
#
#   Pass 3 — mutual-constraint cancellation
#       Geometric-mean blend: sqrt(raw*(1−raw))*2 × AttenuationMult
#       lerped with raw value using occluded_fraction as BlendAlpha —
#       same formula as FAstroConstraintAO::ComputeConstraintWeight Pass 3.
#
# SVG output:
#   constraint_opacity = 1 − final_ao_weight  (high AO → lower cell opacity)
#   Written as opacity="…" attribute on the outermost <g> of the cell.
#   Mirrors AstroCrowdingScale written to ScreenSpaceAOParams[4].w which the
#   USF kernel loop reads to modulate the per-pixel AO contribution.
#
# 2-D channel adaptation:
#   SSAO kernel sample  → neighbour bbox overlap test (O(n) pass over all_bboxes)
#   HorizonAngle weight → bbox intersection-area / self-area ratio
#   ProximityFade       → Z-layer distance fade (same _CAPSULE_MAX_DIST guard)
# ═══════════════════════════════════════════════════════════════════════════════

# Fraction of overlapping neighbours above which crowding attenuation starts.
# Mirrors FAstroConstraintAOParams::CrowdingThreshold (AmbientOcclusionMipBlend).
_CROWDING_THRESHOLD: float = 0.5

# Power of the crowding-suppression curve.
# Mirrors FAstroConstraintAOParams::AttenuationCurve (AmbientOcclusionPower).
_ATTENUATION_CURVE: float = 2.0

# Minimum cell fill opacity floor — even a maximally crowded cell stays legible.
_CROWDING_OPACITY_FLOOR: float = 0.35





@dataclass
class ShadowRayResult:
    light_tile_index:int; texel_index:int; visibility:float





def trace_direct_lighting_shadow_rays(light_tiles,occluder_mesh_sdf,
                                      cfg:DirectLightingHWRTConfig,
                                      far_field_distance=200000.) -> list:
    """
    为每个 light tile texel 发射阴影光线。
    近处的物件要精细，远处的只需知道有没有。这和我们对人的态度，如出一辙。
    """
    res=[]
    for ti,tile in enumerate(light_tiles):
        ld=tile.get('light_direction',(0.,1.,0.))
        for xi,origin in enumerate(tile.get('texel_positions',[])):
            vis=1.
            if occluder_mesh_sdf:
                hit=occluder_mesh_sdf(origin,ld)
                if hit is not None:
                    if hit>cfg.end_bias: vis=0.
                    if cfg.far_field and hit>far_field_distance*.5: vis=max(vis,.1)
            res.append(ShadowRayResult(ti,xi,vis))
    return res





def get_df_shadow_downsample() -> int:
    """
    Mirrors GetDFShadowDownsampleFactor(): returns 1 (full) or 2 (half).
    鲁迅式：分辨率减半是对性能的让步，而非对质量的轻视。
    """
    return 1 if _DF_FULL_RESOLUTION else 2





def get_df_shadow_buffer_size(viewport_w: float, viewport_h: float) -> tuple:
    """
    Mirrors GetBufferSizeForDFShadows(): divides viewport by downsample factor.
    Returns (w, h) of the shadow buffer.
    """
    ds = get_df_shadow_downsample()
    return (int(viewport_w // ds), int(viewport_h // ds))





class AstroCellDFShadowCuller:
    """
    Python equivalent of FCullObjectsForShadowCS.

    Culls candidate occluder cells against a per-light shadow frustum defined
    by a world-space sphere (ShadowBoundingSphere) and up to
    MAX_NUM_SHADOW_CONVEX_HULL_PLANES (12) convex hull planes.

    In 2-D, the «frustum» is approximated by an AABB derived from the sphere
    radius + light direction offset; cells outside this AABB are culled.

    鲁迅式：裁剪是勇敢的放弃——不是每一个遮挡物都值得参与计算，
    只有真正可能投下阴影的，才有资格出现在阴影列表里。
    """

    def __init__(self,
                 shadow_origin: tuple,
                 shadow_radius: float,
                 light_dir: tuple = (0.0, 0.0, 1.0)) -> None:
        self.shadow_origin = shadow_origin  # (x, y, z)
        self.shadow_radius = shadow_radius
        self.light_dir     = light_dir      # normalised

    def cull_candidates(self, all_bboxes: dict) -> list:
        """
        Return list of cell_ids that pass the shadow frustum cull.
        Mirrors the per-object AABB vs ShadowBoundingSphere test in
        FCullObjectsForShadowCS::Execute().

        鲁迅式：球形测试是最公平的裁判——距离超过半径，一律驱逐。
        """
        ox, oy, oz = self.shadow_origin
        r          = self.shadow_radius
        candidates = []
        for cid, bbox in all_bboxes.items():
            cx = bbox["x"] + bbox["w"] / 2.0
            cy = bbox["y"] + bbox["h"] / 2.0
            cz = float(bbox.get("z", 3))
            dist_sq = (cx - ox)**2 + (cy - oy)**2 + (cz - oz)**2
            if dist_sq <= r * r:
                candidates.append(cid)
        return candidates





def compute_df_shadow_factor(
    cell_id:    str,
    bbox:       dict,
    all_bboxes: dict,
    quality:    int = _DF_SHADOW_QUALITY,
) -> float:
    """
    Compute a distance-field shadow factor in [0, 1] for a receiver cell.

    Mirrors the per-pixel ray-march in the distance field shadow compute shader
    (FDistanceFieldShadowingCS), ported to analytic 2-D bbox distances:

    Algorithm:
      1. Build a «distance field» from all occluder cells: for each occluder,
         compute the minimum distance from the receiver's centre to the
         occluder's AABB boundary (analogue of the SDF distance to the mesh).
      2. March along the shadow ray (downward in Z, light from above) for
         quality_steps steps, accumulating the minimum SDF value encountered.
      3. Shadow factor = 1 − clamp(min_sdf_product / ray_length, 0, 1).
         Low min_sdf → ray passed close to an occluder → shadowed.

    Two-sided bias (GDFShadowTwoSidedMeshDistanceBiasScale) is applied when
    the occluder's width ≈ height (symmetric — analogous to a two-sided mesh).

    Returns shadow_factor ∈ [0.0 (fully shadowed), 1.0 (fully lit)].

    鲁迅式：距离场是遮挡物的影响力——距离越近，影响力越大；
    射线在影响力场中穿行，积累的阴影是所有遮挡的合力。
    """
    if not ASTRO_DF_SHADOW_ENABLED or quality == 0:
        return 1.0

    steps       = _DF_QUALITY_STEPS.get(quality, 64)
    cx          = bbox["x"] + bbox["w"] / 2.0
    cy          = bbox["y"] + bbox["h"] / 2.0
    cz          = float(bbox.get("z", 3))
    ray_length  = _CAPSULE_MAX_DIST   # reuse capsule max distance as shadow ray length

    # Build shadow frustum culler centred on the receiver
    culler     = AstroCellDFShadowCuller(
        shadow_origin = (cx, cy, cz),
        shadow_radius = ray_length * 2.0,
    )
    candidates = culler.cull_candidates(all_bboxes)

    min_sdf_product = float("inf")

    for step_i in range(max(1, steps)):
        # Ray position: march upward in Z (toward higher layers = toward light)
        t    = (step_i + 0.5) / steps * ray_length
        rz   = cz + t           # step along +Z (light from above)

        # Minimum SDF distance at this ray position across all occluders
        min_dist_at_step = ray_length

        for occluder_id in candidates:
            if occluder_id == cell_id:
                continue
            ob    = all_bboxes.get(occluder_id, {})
            oz    = float(ob.get("z", 3))
            if oz <= cz:
                continue   # only cells at higher Z can shadow (light from above)

            # 2-D SDF: distance from ray point (cx, cy, rz) to occluder AABB
            ox0, oy0 = ob["x"],           ob["y"]
            ox1, oy1 = ox0 + ob["w"],     oy0 + ob["h"]
            oz0, oz1 = oz - 0.5,          oz + 0.5    # thin slab in Z

            # Box SDF: max(0, d_x^2 + d_y^2 + d_z^2 - inside_penalty)
            dx  = max(ox0 - cx, 0.0, cx - ox1)
            dy  = max(oy0 - cy, 0.0, cy - oy1)
            dz  = max(oz0 - rz, 0.0, rz - oz1)

            # Two-sided bias: widen thin occluders (mirrors C++ two-sided mesh bias)
            aspect = ob["w"] / max(ob["h"], 1.0)
            if 0.7 <= aspect <= 1.4:
                bias = _DF_TWO_SIDED_BIAS * 0.5
                dx   = max(0.0, dx - bias)
                dy   = max(0.0, dy - bias)

            sdf_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            min_dist_at_step = min(min_dist_at_step, sdf_dist)

        # Accumulate minimum SDF product across all steps (penumbra accumulator)
        min_sdf_product = min(min_sdf_product, min_dist_at_step)

    # Shadow factor: how much the min SDF was compressed relative to ray length
    shadow_factor = max(0.0, min(1.0, min_sdf_product / max(ray_length * 0.1, 1.0)))
    return round(shadow_factor, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] CapsuleShadowRendering → extended Python port (v2)
#
# Previously ported from CapsuleShadowRendering.cpp (commit 1d22562).
# This extension adds:
#   • AstroCellCapsuleShadowTileGrid — FCapsuleShadowingCS tiled culling port
#   • compute_indirect_capsule_shadow — ECapsuleShadowingType::IndirectTiledCulling
#   • AstroCellCapsuleIndirectShadow  — sky-light indirect occlusion for cells
#
# Key UE5 constructs added in this pass:
#   GCapsuleIndirectShadows           → ASTRO_CAPSULE_INDIRECT
#   GCapsuleIndirectConeAngle         → _CAPSULE_INDIRECT_CONE_ANGLE
#   GCapsuleSkyAngleScale             → _CAPSULE_SKY_ANGLE_SCALE
#   GCapsuleMinSkyAngle               → _CAPSULE_MIN_SKY_ANGLE_DEG
#   GCapsuleShadowFadeAngleFromVert   → _CAPSULE_FADE_ANGLE
#   FCapsuleShadowingCS (indirect)    → AstroCellCapsuleShadowTileGrid
#   GetBufferSizeForCapsuleShadows()  → get_capsule_shadow_buffer_size()
#   GetCapsuleShadowDownsampleFactor()→ get_capsule_shadow_downsample()
# ═══════════════════════════════════════════════════════════════════════════════

ASTRO_CAPSULE_INDIRECT: bool = True

# Mirrors GCapsuleIndirectConeAngle (PI/8)
_CAPSULE_INDIRECT_CONE_ANGLE: float = math.pi / 8.0

# Mirrors GCapsuleSkyAngleScale
_CAPSULE_SKY_ANGLE_SCALE: float = 0.6

# Mirrors GCapsuleMinSkyAngle (15°)
_CAPSULE_MIN_SKY_ANGLE_DEG: float = 15.0

# Mirrors GCapsuleShadowFadeAngleFromVertical (PI/3 = 60°)
_CAPSULE_FADE_ANGLE: float = math.pi / 3.0

# Full resolution capsule shadows (GCapsuleShadowsFullResolution)
_CAPSULE_FULL_RESOLUTION: int = 0





def get_capsule_shadow_downsample() -> int:
    """Mirrors GetCapsuleShadowDownsampleFactor()."""
    return 1 if _CAPSULE_FULL_RESOLUTION else 2





def get_capsule_shadow_buffer_size(viewport_w: float, viewport_h: float) -> tuple:
    """Mirrors GetBufferSizeForCapsuleShadows()."""
    ds = get_capsule_shadow_downsample()
    return (int(viewport_w // ds), int(viewport_h // ds))





class AstroCellCapsuleShadowTileGrid:
    """
    Python equivalent of the tiled culling pass in FCapsuleShadowingCS
    (ECapsuleShadowingType::IndirectTiledCulling).

    Subdivides the viewport into a regular grid of tiles and, for each tile,
    accumulates the indirect occlusion from capsule-shaped cell occluders
    whose projected AABB overlaps the tile.

    In 2-D SVG terms, tiles are virtual «shadow receiver groups»; a cell
    belongs to a tile if its centre falls within the tile boundary.  The
    indirect occlusion for each tile is the sum of soft-cone contributions
    from all occluder cells in the scene.

    鲁迅式：分块是一种公平的分配——每一小格都有自己的阴影份额，
    不会因为邻居太高大而被遗忘，也不会因为自己太小而逃脱计算。
    """

    def __init__(self,
                 viewport_w: float = 1200.0,
                 viewport_h: float = 900.0,
                 tile_size:  int   = 8) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.tile_size  = tile_size
        self.cols = max(1, int(math.ceil(viewport_w / tile_size)))
        self.rows = max(1, int(math.ceil(viewport_h / tile_size)))
        # Per-tile indirect occlusion accumulator
        self._tile_occlusion: list = [
            [0.0] * self.cols for _ in range(self.rows)
        ]

    def _tile_coords(self, x: float, y: float) -> tuple:
        """Map world (x, y) to (row, col) tile index."""
        col = max(0, min(self.cols - 1, int(x // self.tile_size)))
        row = max(0, min(self.rows - 1, int(y // self.tile_size)))
        return (row, col)

    def accumulate(
        self,
        receiver_id:  str,
        receiver_bbox: dict,
        occluder_bboxes: dict,
    ) -> float:
        """
        Accumulate indirect capsule occlusion for one receiver cell.

        Mirrors the tiled culling inner loop in FCapsuleShadowingCS
        (IndirectTiledCulling permutation): for each occluder that is above
        the receiver (higher Z-layer), compute the soft-cone occlusion using
        the indirect cone angle, fade angle, and sky-angle scale.

        The indirect occlusion is the sky-light equivalent: occluder cells
        above the receiver block a cone of sky light centred on the vertical.

        Returns the accumulated indirect occlusion factor ∈ [0, 1].

        鲁迅式：间接遮挡是环境的冷漠——天光不关心谁在下面，
        但高大的遮挡物会不由自主地减少天光的份额，这是物理，不是歧视。
        """
        if not ASTRO_CAPSULE_INDIRECT:
            return 0.0

        rx = receiver_bbox["x"] + receiver_bbox["w"] / 2.0
        ry = receiver_bbox["y"] + receiver_bbox["h"] / 2.0
        rz = float(receiver_bbox.get("z", 3))

        row, col = self._tile_coords(rx, ry)

        total_indirect = 0.0

        for occ_id, ob in occluder_bboxes.items():
            if occ_id == receiver_id:
                continue
            oz = float(ob.get("z", 3))
            if oz <= rz:
                continue  # only cells above cast indirect shadow

            ox = ob["x"] + ob["w"] / 2.0
            oy = ob["y"] + ob["h"] / 2.0

            dz     = oz - rz
            dx, dy = ox - rx, oy - ry

            # Compute angle from vertical (sky direction = +Z)
            horiz_dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            angle_from_vert = math.atan2(horiz_dist, max(dz, 0.1))

            # Fade weight: beyond FADE_ANGLE the occluder stops contributing
            # (avoids self-shadowing artefacts near-vertical neighbours)
            if angle_from_vert >= _CAPSULE_FADE_ANGLE:
                continue

            fade = max(0.0, 1.0 - angle_from_vert / _CAPSULE_FADE_ANGLE)

            # Cone solid angle: occluder fills INDIRECT_CONE_ANGLE from the receiver
            cone_half  = _CAPSULE_INDIRECT_CONE_ANGLE * _CAPSULE_SKY_ANGLE_SCALE
            cone_half  = max(math.radians(_CAPSULE_MIN_SKY_ANGLE_DEG), cone_half)

            # Occluder angular size from receiver
            occ_radius  = min(ob["w"], ob["h"]) / 2.0
            occ_dist    = math.sqrt(dx*dx + dy*dy + dz*dz)
            angular_size = math.atan2(occ_radius, max(occ_dist, 1.0))

            # Normalised cone occupancy
            cone_frac = min(1.0, angular_size / max(cone_half, 1e-6))
            indirect  = cone_frac * fade

            total_indirect = min(1.0, total_indirect + indirect)

        # Accumulate into tile grid
        self._tile_occlusion[row][col] = min(1.0,
            self._tile_occlusion[row][col] + total_indirect)

        return total_indirect

    def get_tile_occlusion(self, x: float, y: float) -> float:
        """Return the accumulated indirect occlusion for the tile at (x, y)."""
        row, col = self._tile_coords(x, y)
        return self._tile_occlusion[row][col]





def compute_indirect_capsule_shadow(
    cell_id:    str,
    bbox:       dict,
    all_bboxes: dict,
) -> float:
    """
    Compute indirect capsule shadow (sky-light occlusion) for a receiver cell.

    Top-level entry for the IndirectTiledCulling permutation of FCapsuleShadowingCS.
    Constructs a single-tile grid centred on the receiver and accumulates
    occlusion from all cells above it.

    Returns indirect_occlusion ∈ [0, 1]; 0 = fully sky-lit, 1 = fully occluded.

    鲁迅式：间接阴影是世界对个体的压制——头顶的遮挡物越多，
    能到达你的天光就越少。但这不是命运，只是几何。
    """
    w = float(bbox.get("w", 100))
    h = float(bbox.get("h", 50))
    grid = AstroCellCapsuleShadowTileGrid(
        viewport_w = w * 4,
        viewport_h = h * 4,
        tile_size  = max(8, int(min(w, h) * 0.25)),
    )
    return grid.accumulate(cell_id, bbox, all_bboxes)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PlanarReflectionRendering → Python port
#
# Ported from commit upstream/unreal-renderer-ue5:
#   Renderer-Private/PlanarReflectionRendering.cpp
#
# 鲁迅曾言：「镜子里的世界，是真实世界的倒影——
# 不是真实，却包含着真实的信息，值得被认真对待。」
# 平面反射的本质：将场景以反射平面为轴翻转，然后用这个虚拟视图渲染，
# 将结果贴回到反射平面的像素上。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   SetupPlanarReflectionUniformParameters → setup_cell_planar_reflection()
#   FPlanarReflectionUniformParameters    → AstroCellPlanarReflectionParams
#   TPrefilterPlanarReflectionPS          → AstroCellPlanarReflectionPrefilter
#   AddProcessPlanarReflectionPass        → apply_cell_planar_reflection()
#   GMaxPlanarReflectionViews (= 2)       → _MAX_PLANAR_REFLECTION_VIEWS
#   PlanarReflectionParameters (normal clamp distance / fade) → embedded
# ═══════════════════════════════════════════════════════════════════════════════

# Mirrors GMaxPlanarReflectionViews = 2 (stereo support)
_MAX_PLANAR_REFLECTION_VIEWS: int = 2

# Whether planar reflections are enabled in the Astro pipeline
ASTRO_PLANAR_REFLECTION_ENABLED: bool = True

# Prefilter roughness distance (mirrors CVarPlanarReflectionPrefilterRoughnessDistance)
_PLANAR_REFL_ROUGHNESS_DIST: float = 0.3





def _shadow_depth_bias_corrected(
    caster_z: float, receiver_z: float,
    depth_bias: float = _CSM_DEPTH_BIAS,
    slope_bias: float = _CSM_SLOPE_BIAS,
    receiver_bias: float = _CSM_RECEIVER_BIAS,
) -> float:
    """
    Depth-bias-corrected shadow depth for CSM comparison.
    Port of the depth bias formula in ShadowRendering.cpp CSM depth pass.

    鲁迅式：偏置是对规则的微调——在法律允许的范围内作弊，
    让影子落在正确的位置，避免 peter panning 或 self-shadowing。
    """
    z_range  = max(_ASTRO_CELL_MAX_Z_LAYERS, 1)
    slope    = abs(caster_z - receiver_z) / z_range
    biased   = caster_z + depth_bias + slope_bias * slope
    biased  -= receiver_z * receiver_bias
    return biased





def _shadow_pcf_weight(
    shadow_delta: float,
    transition_scale: float = _SHADOW_TRANSITION_SCALE,
) -> float:
    """
    PCF soft-edge weight — mirrors the transition-scale fade in CSM.

    鲁迅式：PCF 是影子边缘的民主投票——每个采样点一票，平均值决定光影比。
    """
    return max(0.0, min(1.0, 0.5 + shadow_delta / max(transition_scale, 1.0)))





def _shadow_pcss_kernel_radius(
    caster_w: float, caster_h: float, z_dist: float,
) -> float:
    """
    PCSS soft-shadow kernel radius — area-proportional penumbra.

    鲁迅式：柔和的阴影是对真实的妥协——面积越大，距离越远，影子越模糊。
    """
    projected_area = caster_w * caster_h
    soft_radius    = math.sqrt(max(projected_area, 1.0)) * (
        z_dist / max(_CAPSULE_MAX_DIST, 1.0))
    return max(1.0, min(soft_radius, _PCSS_MAX_KERNEL_RADIUS))





class AstroCellShadowProjection:
    """
    Per-cell CSM shadow projection — Python equivalent of the CSM projection
    pass in ShadowRendering.cpp.

    Holds depth-bias configuration for one light type (directional/point/
    rect/spot) and projects caster depth values onto receivers via PCF or PCSS.

    鲁迅式：投影器是光与影之间的翻译官——把三维深度关系翻译成二维明暗因子。
    """

    def __init__(
        self,
        depth_bias:       float = _CSM_DEPTH_BIAS,
        slope_bias:       float = _CSM_SLOPE_BIAS,
        receiver_bias:    float = _CSM_RECEIVER_BIAS,
        transition_scale: float = _SHADOW_TRANSITION_SCALE,
        filter_method:    int   = _SHADOW_FILTER_METHOD,
    ) -> None:
        self.depth_bias       = depth_bias
        self.slope_bias       = slope_bias
        self.receiver_bias    = receiver_bias
        self.transition_scale = transition_scale
        self.filter_method    = filter_method

    def project(
        self,
        receiver_z: float,
        caster_z:   float,
        caster_w:   float = 80.0,
        caster_h:   float = 50.0,
    ) -> float:
        """
        Project a caster onto a receiver; return shadow factor ∈ [0,1].
        1.0 = fully lit; 0.0 = fully in shadow.

        鲁迅式：投影是以牺牲精度换取效率——单个采样点决定阴影。
        """
        biased = _shadow_depth_bias_corrected(
            caster_z, receiver_z,
            self.depth_bias, self.slope_bias, self.receiver_bias
        )
        shadow_delta = receiver_z - biased
        if self.filter_method == 1:
            z_dist = abs(caster_z - receiver_z)
            kr     = _shadow_pcss_kernel_radius(caster_w, caster_h, z_dist)
            effective_scale = self.transition_scale * (
                kr / max(_PCSS_MAX_KERNEL_RADIUS, 1.0) + 0.1)
            return _shadow_pcf_weight(shadow_delta, effective_scale)
        return _shadow_pcf_weight(shadow_delta, self.transition_scale)





class AstroCellShadowRenderer:
    """
    Epoch-level shadow renderer — mirrors RenderShadowDepthMaps dispatch.

    build_shadow_depth_map(): collect + sort casters by z descending.
    project_shadows():        compute per-(receiver,caster) shadow factors.
    get_shadow_factor():      query minimum shadow factor for a receiver.

    Stencil optimisation (GStencilOptimization): skip rebuild when scene unchanged.

    鲁迅式：阴影渲染器是场景中最劳累的工人——
    它为每个接收者计算每个投射者的贡献，然后缓存结果。
    效率全靠缓存；缓存失效，一切重来。
    """

    def __init__(
        self, projection: AstroCellShadowProjection | None = None,
    ) -> None:
        self._projection     = projection or AstroCellShadowProjection()
        self._depth_map:     list = []
        self._shadow_factors: dict = {}
        self._cache_key:     int  = -1

    def _compute_cache_key(self, all_bboxes: dict) -> int:
        return hash(tuple(
            (cid, round(bb.get("z", 0), 2))
            for cid, bb in sorted(all_bboxes.items())
        ))

    def build_shadow_depth_map(self, all_bboxes: dict) -> None:
        """
        Collect casters and sort by z descending.
        Stencil optimisation: skip when scene unchanged.

        鲁迅式：建造深度图是一次普查——每个单元格都要被登记，按深度排序。
        """
        new_key = self._compute_cache_key(all_bboxes)
        if _STENCIL_OPTIMIZATION and new_key == self._cache_key:
            return
        self._cache_key = new_key
        self._depth_map = sorted(
            [(cid, float(bb.get("z", 0)), float(bb.get("w", 80)),
              float(bb.get("h", 50))) for cid, bb in all_bboxes.items()],
            key=lambda t: t[1], reverse=True,
        )

    def project_shadows(self, all_bboxes: dict) -> None:
        """
        For every receiver, aggregate shadow factors from all higher-z casters.
        Minimum factor (most-shadowed) wins.

        鲁迅式：投影是权力的叠加——取最深者为准，因为黑暗往往来自多个遮挡共谋。
        """
        self.build_shadow_depth_map(all_bboxes)
        self._shadow_factors = {}
        for rcid, rbb in all_bboxes.items():
            recv_z = float(rbb.get("z", 0))
            min_f  = 1.0
            rx0, ry0 = rbb.get("x", 0), rbb.get("y", 0)
            rx1 = rx0 + rbb.get("w", 80)
            ry1 = ry0 + rbb.get("h", 50)
            for (cid, cz, cw, ch) in self._depth_map:
                if cid == rcid or cz <= recv_z:
                    continue
                cb  = all_bboxes.get(cid, {})
                cx0 = cb.get("x", 0)
                cy0 = cb.get("y", 0)
                cx1 = cx0 + cb.get("w", 80)
                cy1 = cy0 + cb.get("h", 50)
                if rx1 < cx0 or cx1 < rx0 or ry1 < cy0 or cy1 < ry0:
                    continue
                f = self._projection.project(recv_z, cz, cw, ch)
                if f < min_f:
                    min_f = f
            self._shadow_factors[rcid] = min_f

    def get_shadow_factor(self, cell_id: str) -> float:
        """Return shadow factor [0,1] for receiver cell. 1.0 = fully lit."""
        return self._shadow_factors.get(cell_id, 1.0)


_ASTRO_SHADOW_RENDERER_V2: AstroCellShadowRenderer = AstroCellShadowRenderer()





def get_shadow_renderer() -> AstroCellShadowRenderer:
    """Return the process-level AstroCellShadowRenderer singleton."""
    return _ASTRO_SHADOW_RENDERER_V2


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PrimitiveSceneInfo → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/PrimitiveSceneInfo.cpp
#
# 鲁迅曰：「人必生活着，爱才有所附丽。」
# Primitive 必须存在于 Scene 中，才能被渲染，才能投射阴影，才能影响他者。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FPrimitiveFlagsCompact  → AstroCellPrimitiveFlags
#   FPrimitiveSceneInfoCompact → AstroCellSceneInfoCompact
#   FBatchingSPDI::DrawMesh → AstroCellStaticDrawCommandCache
#   FPrimitiveSceneInfo::AddToScene / RemoveFromScene
#       → AstroCellPrimitiveRegistry.add_primitive / remove_primitive
#   GMeshDrawCommandsCacheMultithreaded / GMeshDrawCommandsBatchSize
#       → _MDC_CACHE_MT / _MDC_BATCH_SIZE
# ═══════════════════════════════════════════════════════════════════════════════

_MDC_CACHE_MT:    bool = True
_MDC_BATCH_SIZE:  int  = 12
_NANITE_MAT_PARALLEL: bool = True
_RT_PRIM_CACHE_MT:    bool = True


