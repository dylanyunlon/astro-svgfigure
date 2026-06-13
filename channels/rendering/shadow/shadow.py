#!/usr/bin/env python3
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# =============================================================================
# [ASTRO-CELL] ShadowSetup + ShadowDepthRendering + DeferredShadingRenderer
# → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowSetup.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowDepthRendering.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/DeferredShadingRenderer.cpp
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 阴影亦如此——不建立，便消失；不裁剪，便爆炸；
# 延迟着色是文明的产物，但延迟不是逃避——每一帧都必须最终收账。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
# ShadowSetup.cpp
#   GMinScreenRadiusForShadowCaster → _SHADOW_RADIUS_THRESHOLD
#   GCacheWholeSceneShadows         → ASTRO_CACHE_WHOLE_SCENE_SHADOWS
#   GWholeSceneShadowCacheMb        → ASTRO_SHADOW_CACHE_BUDGET_MB
#   GCachedShadowsCastFromMovablePrimitives → ASTRO_MOVABLE_CAST_FROM_CACHE
#   GSkipCullingNaniteMeshes        → ASTRO_SKIP_CULL_NANITE (→ skip_cull_species)
#   FProjectedShadowInfo            → AstroCellShadowInfo
#   BuildWholeSceneShadowCaster()   → build_whole_scene_shadow_caster()
#   FrustumCullShadowCaster()       → _frustum_cull_shadow_caster()
#   ComputeShadowSubjectCullingPlanes() → _compute_cull_planes()
#   SetupProjectedShadow()          → AstroCellShadowInfo.setup()
#
# ShadowDepthRendering.cpp
#   GShadowLODDistanceFactor        → _SHADOW_LOD_DISTANCE_FACTOR
#   GShadowLODDistanceFactorCascadeScale → _SHADOW_LOD_CASCADE_SCALE
#   SetupShadowDepthPassUniformBuffer() → _setup_shadow_depth_pass_ub()
#   FShadowDepthPassUniformParameters  → AstroCellShadowDepthPassParams
#   RenderShadowDepths()            → AstroCellShadowDepthRenderer.render()
#   TRenderShadowDepths (template)  → AstroCellShadowDepthRenderer (single class)
#   LOD selection dist              → _shadow_lod_distance()
#
# DeferredShadingRenderer.cpp
#   CVarNanitePrimeHZBMode          → _NANITE_PRIME_HZB_MODE
#   FDeferredShadingSceneRenderer::Render() → AstroCellDeferredShadingRenderer.render()
#   FDeferredShadingSceneRenderer::InitViews() → _init_views()
#   FDeferredShadingSceneRenderer::RenderShadowDepthMaps() → _render_shadow_depth_maps()
#   FDeferredShadingSceneRenderer::RenderBasePass() → _render_base_pass()
#   FDeferredShadingSceneRenderer::RenderLights()  → _render_lights()
#   FDeferredShadingSceneRenderer::RenderPostProcessing() → _render_post_processing()
#   GBuffer layout (A/B/C/D)        → AstroCellGBuffer
#
# 2-D channel adaptation:
#   Shadow frustum → 2-D AABB cull rectangle (screen-space)
#   Shadow depth map → per-cell shadow depth dict keyed by cell_id
#   Whole-scene shadow cache → shadow_cache.json channel file
#   G-Buffer → per-cell gbuffer.json (BaseColor, Normal, Roughness, Shading)
#   Deferred shading pass → AstroCellGBufferPass (writes gbuffer.json)
#   Final composition → AstroCellDeferredShadingRenderer.render() -> SVG string
# =============================================================================

# ---------------------------------------------------------------------------
# ShadowSetup.cpp — control variables
# ---------------------------------------------------------------------------

#: Minimum screen-space radius for a shadow caster to be kept.
#: Mirrors GMinScreenRadiusForShadowCaster (r.Shadow.RadiusThreshold = 0.01).
_SHADOW_RADIUS_THRESHOLD: float = 0.01

#: Toggle whole-scene shadow caching.
#: Mirrors GCacheWholeSceneShadows (r.Shadow.CacheWholeSceneShadows = 1).
ASTRO_CACHE_WHOLE_SCENE_SHADOWS: bool = True

#: Budget in megabytes for cached whole-scene shadows.
#: Mirrors GWholeSceneShadowCacheMb (r.Shadow.WholeSceneShadowCacheMb = 150).
ASTRO_SHADOW_CACHE_BUDGET_MB: int = 150

#: Allow movable primitives to cast shadows from cached whole-scene maps.
#: Mirrors GCachedShadowsCastFromMovablePrimitives.
ASTRO_MOVABLE_CAST_FROM_CACHE: bool = True

#: Skip CPU culling for Nanite meshes (species in our 2-D analogue).
#: Mirrors GSkipCullingNaniteMeshes (r.Shadow.SkipCullingNaniteMeshes = 1).
ASTRO_SKIP_CULL_NANITE: bool = True

#: Path for the shared whole-scene shadow cache channel.
_SHADOW_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "physics", "shadow_cache.json",
)


def _load_shadow_cache() -> dict:
    """Load the whole-scene shadow cache from the channel file."""
    if not os.path.isfile(_SHADOW_CACHE_PATH):
        return {"shadows": {}, "frame": -1}
    try:
        with open(_SHADOW_CACHE_PATH) as _f:
            return json.load(_f)
    except (json.JSONDecodeError, OSError):
        return {"shadows": {}, "frame": -1}


def _save_shadow_cache(cache: dict) -> None:
    """Persist the whole-scene shadow cache to the channel file."""
    os.makedirs(os.path.dirname(_SHADOW_CACHE_PATH), exist_ok=True)
    with open(_SHADOW_CACHE_PATH, "w") as _f:
        json.dump(cache, _f, indent=2)


# ---------------------------------------------------------------------------
# AstroCellShadowInfo — Python equivalent of FProjectedShadowInfo
#
# 鲁迅式：每一道阴影都是一道档案——记录着发光源、遮挡物、接收面。
# 档案建立得越精确，世界便越接近真实；档案缺失，便只剩猜测。
# ---------------------------------------------------------------------------

class AstroCellShadowInfo:
    """
    Python equivalent of FProjectedShadowInfo.

    Stores all parameters needed to evaluate a single projected shadow:
      - The light that casts it (position, direction, type)
      - The shadow frustum as a 2-D AABB cull rect
      - The set of shadow casters (cell_ids) that passed culling
      - The shadow depth map (caster_id → normalised depth in [0,1])

    Lifecycle mirrors FProjectedShadowInfo:
        __init__   → constructor + initial parameter setup
        setup()    → SetupProjectedShadow() — computes cull rect, registers casters
        render()   → RenderShadowDepths() — fills depth map
        sample()   → shadow sample lookup at a receiver point
    """

    def __init__(
        self,
        shadow_id: str,
        light_position: tuple = (0.0, -600.0, 8.0),
        light_direction: tuple = (0.0, 1.0, -0.3),
        shadow_type: str = "whole_scene",  # "whole_scene" | "per_object" | "preshadow"
        resolution: int = 256,
    ) -> None:
        self.shadow_id       = shadow_id
        self.light_position  = light_position      # (x, y, z_layer)
        self.light_direction = light_direction      # normalised direction toward scene
        self.shadow_type     = shadow_type
        self.resolution      = resolution

        # Cull rect in screen space — set by setup()
        self.cull_rect_x0: float = 0.0
        self.cull_rect_y0: float = 0.0
        self.cull_rect_x1: float = 1e9
        self.cull_rect_y1: float = 1e9

        # Shadow casters that passed frustum culling
        self.casters: list = []

        # Depth map: caster_id → depth ∈ [0, 1]
        self.depth_map: dict = {}

        # Cached flag (set when loaded from shadow_cache.json)
        self.is_cached: bool = False
        self.cache_frame: int = -1

    def setup(
        self,
        viewport_w: float,
        viewport_h: float,
        cell_entries: list,
    ) -> int:
        """
        Compute shadow cull rect and cull casters.

        Mirrors SetupProjectedShadow() + BuildWholeSceneShadowCaster()
        + FrustumCullShadowCaster():

          1. Project the light frustum to a screen-space AABB cull rect.
             (In 2-D this is the viewport-aligned bounding box of all cells
              visible to the light direction — simplified to full viewport for
              directional lights, cone-clipped for point lights.)

          2. For each cell in cell_entries, test IsScreenSpaceSizeLargeEnough
             (mirrors GMinScreenRadiusForShadowCaster screen radius test) and
             _frustum_cull_shadow_caster (per-primitive AABB vs. light frustum).

          3. If ASTRO_SKIP_CULL_NANITE, skip the AABB cull for all cells
             (mirrors skipping Nanite meshes in CPU culling).

        Returns number of accepted shadow casters.

        鲁迅式：裁剪是必要的吝啬——把不必要的影子删掉，
        才能把资源留给真正重要的那几道。
        """
        self.casters.clear()
        self.depth_map.clear()

        lx, ly, _ = self.light_position
        # Directional light: cull rect = full viewport
        self.cull_rect_x0 = 0.0
        self.cull_rect_y0 = 0.0
        self.cull_rect_x1 = viewport_w
        self.cull_rect_y1 = viewport_h

        viewport_area = max(viewport_w * viewport_h, 1.0)

        for entry in cell_entries:
            cid  = entry.get("cell_id", "")
            bbox = entry.get("bbox", {})
            if not cid or not bbox:
                continue

            w = float(bbox.get("w", 0))
            h = float(bbox.get("h", 0))
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))

            # ── Screen-size cull (GMinScreenRadiusForShadowCaster) ──────────
            screen_frac = (w * h) / viewport_area
            if screen_frac < _SHADOW_RADIUS_THRESHOLD and not ASTRO_SKIP_CULL_NANITE:
                continue  # too small to cast a meaningful shadow

            # ── Frustum cull (FrustumCullShadowCaster) ──────────────────────
            if not ASTRO_SKIP_CULL_NANITE:
                if not _frustum_cull_shadow_caster(
                    x, y, w, h,
                    self.cull_rect_x0, self.cull_rect_y0,
                    self.cull_rect_x1, self.cull_rect_y1,
                ):
                    continue

            self.casters.append(cid)

        print(
            f"[ASTRO-SHADOW] ShadowSetup: shadow_id={self.shadow_id} "
            f"type={self.shadow_type} casters={len(self.casters)} "
            f"viewport=({viewport_w:.0f}×{viewport_h:.0f})",
            file=sys.stderr,
        )
        return len(self.casters)

    def render(self, cell_entries: list) -> int:
        """
        Fill the shadow depth map for all registered casters.

        Mirrors RenderShadowDepths() / FShadowDepthPassUniformParameters
        + the shadow depth rendering loop that iterates mesh draw commands
        and outputs depth values into the depth render target.

        In 2-D: depth is the normalised Z value of each caster cell,
        computed from the light's perspective (Z-order along light direction).

        Returns number of depth entries written.

        鲁迅式：深度图是阴影的骨架——没有它，阴影便只是装饰，没有重量。
        """
        if not self.casters:
            return 0

        # Build fast cid → bbox lookup
        bbox_by_cid = {e["cell_id"]: e["bbox"] for e in cell_entries if "cell_id" in e}

        # Compute light-space depths using the _shadow_lod_distance helper
        # (analogous to the Z-projection in SetupShadowDepthPassUniformBuffer)
        z_values = []
        for cid in self.casters:
            bbox = bbox_by_cid.get(cid, {})
            z    = float(bbox.get("z", 3)) if bbox else 3.0
            # Project along light direction: depth ≈ z + tiny x/y component
            ldx, ldy, ldz = self.light_direction
            cx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
            cy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
            depth_proj = z * abs(ldz) + cx * abs(ldx) / 1000.0 + cy * abs(ldy) / 1000.0
            z_values.append((cid, depth_proj))

        if not z_values:
            return 0

        z_min = min(v for _, v in z_values)
        z_max = max(v for _, v in z_values)
        z_range = max(z_max - z_min, 1e-6)

        # Normalise to [0, 1] (reversed: nearer = higher value, mirrors UE5
        # reversed depth buffer convention)
        for cid, z_proj in z_values:
            self.depth_map[cid] = round(1.0 - (z_proj - z_min) / z_range, 6)

        print(
            f"[ASTRO-SHADOW] ShadowDepthRendering: shadow_id={self.shadow_id} "
            f"depth_entries={len(self.depth_map)} "
            f"z_range=[{z_min:.2f},{z_max:.2f}]",
            file=sys.stderr,
        )
        return len(self.depth_map)

    def sample(self, caster_id: str) -> float:
        """
        Sample the shadow depth map for a given caster cell.

        Returns normalised depth in [0, 1].
        0 = fully shadowed (far from light); 1 = fully lit (nearest to light).
        Missing entries return 0.5 (unknown / penumbra estimate).

        鲁迅式：0.5 是最懦弱的答案——既不否认，也不承认。
        但在阴影映射中，未知等于中间灰，总好过黑白二值的谎言。
        """
        return self.depth_map.get(caster_id, 0.5)

    def to_dict(self) -> dict:
        return {
            "shadow_id":     self.shadow_id,
            "type":          self.shadow_type,
            "light_pos":     self.light_position,
            "light_dir":     self.light_direction,
            "cull_rect":     [self.cull_rect_x0, self.cull_rect_y0,
                              self.cull_rect_x1, self.cull_rect_y1],
            "caster_count":  len(self.casters),
            "depth_entries": len(self.depth_map),
            "is_cached":     self.is_cached,
        }


def _frustum_cull_shadow_caster(
    cx: float, cy: float, cw: float, ch: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """
    Test whether a shadow caster AABB overlaps the shadow cull rect.

    Mirrors FrustumCullShadowCaster() — a simple AABB-AABB overlap test
    between the caster's screen-space bounding box and the shadow frustum
    cull rectangle.

    Returns True if the caster is INSIDE (should be kept).

    鲁迅式：裁剪保留的是边界内的声音；边界之外的，默然无声。
    """
    return not (
        cx + cw < rx0 or cx > rx1 or
        cy + ch < ry0 or cy > ry1
    )


def _compute_cull_planes(
    light_position: tuple,
    scene_bbox: dict,
) -> list:
    """
    Compute shadow subject culling planes.

    Mirrors ComputeShadowSubjectCullingPlanes() which builds per-frustum
    planes used to cull scene primitives against the shadow volume.

    In 2-D: returns 4 axis-aligned half-planes (normals + distances)
    derived from the scene bbox expanded toward the light position.

    鲁迅式：裁剪平面是边界的哲学——它说：「到这里，不再往前。」
    而光从来不懂边界；我们替它划定。
    """
    lx, ly, _ = light_position
    min_x = scene_bbox.get("min_x", 0.0)
    min_y = scene_bbox.get("min_y", 0.0)
    max_x = scene_bbox.get("max_x", 1200.0)
    max_y = scene_bbox.get("max_y", 900.0)

    # Expand bbox toward light by 20% to include shadow receivers outside scene
    expand_x = (max_x - min_x) * 0.2
    expand_y = (max_y - min_y) * 0.2

    return [
        {"normal": (-1,  0), "dist": -(min_x - expand_x)},  # left
        {"normal": ( 1,  0), "dist":   max_x + expand_x},   # right
        {"normal": ( 0, -1), "dist": -(min_y - expand_y)},  # top
        {"normal": ( 0,  1), "dist":   max_y + expand_y},   # bottom
    ]


def build_whole_scene_shadow_caster(
    viewport_w: float,
    viewport_h: float,
    cell_entries: list,
    frame_index: int = 0,
) -> "AstroCellShadowInfo":
    """
    Build and return a whole-scene shadow for the primary directional light.

    Mirrors BuildWholeSceneShadowCaster() called from
    FDeferredShadingSceneRenderer::InitDynamicShadows():
      1. Creates an FProjectedShadowInfo (→ AstroCellShadowInfo)
      2. Calls SetupProjectedShadow (→ .setup())
      3. If ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
           - Loads shadow_cache.json
           - If cache is fresh (same frame), returns cached shadow info
           - Otherwise re-builds and saves to cache

    鲁迅式：全场景阴影是一场统一的压迫——所有人共享同一道影子，
    个体的遮蔽在这里被抹平为集体的灰度。
    """
    shadow = AstroCellShadowInfo(
        shadow_id=f"whole_scene_{frame_index}",
        shadow_type="whole_scene",
    )

    if ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
        cache = _load_shadow_cache()
        if cache.get("frame") == frame_index and cache.get("shadows"):
            # Cache hit — restore from disk (mirrors cached whole-scene path)
            cached_shadow = cache["shadows"].get("whole_scene")
            if cached_shadow:
                shadow.depth_map  = cached_shadow.get("depth_map", {})
                shadow.casters    = list(shadow.depth_map.keys())
                shadow.is_cached  = True
                shadow.cache_frame = frame_index
                print(
                    f"[ASTRO-SHADOW] ShadowSetup: cache HIT frame={frame_index} "
                    f"casters={len(shadow.casters)}",
                    file=sys.stderr,
                )
                return shadow

    # Cache miss or disabled — full setup + render
    shadow.setup(viewport_w, viewport_h, cell_entries)
    shadow.render(cell_entries)

    if ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
        cache = {"frame": frame_index, "shadows": {
            "whole_scene": {
                "depth_map": shadow.depth_map,
            }
        }}
        _save_shadow_cache(cache)

    return shadow


# ---------------------------------------------------------------------------
# ShadowDepthRendering.cpp — LOD distance + depth pass uniforms
#
# 鲁迅式：LOD 是对远处事物的善意忽略——当它们足够小，
# 用精确数据描述它们是一种浪费，甚至是一种奢靡。
# ---------------------------------------------------------------------------

#: Mirrors GShadowLODDistanceFactor (r.Shadow.LODDistanceFactor = 1.0).
_SHADOW_LOD_DISTANCE_FACTOR: float = 1.0

#: Mirrors GShadowLODDistanceFactorCascadeScale (scales with cascade index).
_SHADOW_LOD_CASCADE_SCALE: float = 0.0


def _shadow_lod_distance(
    base_distance: float,
    cascade_index: int = 0,
) -> float:
    """
    Compute effective LOD selection distance for shadow rendering.

    Mirrors the multiplier applied to mesh LOD distances in
    FShadowDepthPassMeshProcessor::Process() using GShadowLODDistanceFactor
    and GShadowLODDistanceFactorCascadeScale:

        EffectiveDist = BaseDist
                      * LODDistanceFactor
                      * (1 + LODDistanceFactorCascadeScale * cascade_index)

    Higher effective distance → select lower-detail LOD.

    鲁迅式：距离是宽容，也是懒惰——从远处看，粗糙也像精细。
    """
    return (
        base_distance
        * _SHADOW_LOD_DISTANCE_FACTOR
        * (1.0 + _SHADOW_LOD_CASCADE_SCALE * cascade_index)
    )


@dataclass
class AstroCellShadowDepthPassParams:
    """
    Python equivalent of FShadowDepthPassUniformParameters.

    Holds the per-pass uniform buffer parameters for shadow depth rendering.
    Fields mirror the IMPLEMENT_STATIC_UNIFORM_BUFFER_STRUCT declaration.

    In 2-D: these parameters control the depth projection and LOD selection
    for each cell being rendered into the shadow depth map.

    鲁迅式：Uniform Buffer 是渲染器的宪章——
    一切着色器都必须遵守，不得私自解释。
    """
    ProjectionMatrix:   tuple = (1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0)  # 4×4 identity
    ShadowParams:       tuple = (0.0, 0.0, 0.0, 0.0)  # (bias, slope, normal, unused)
    bRenderReflectiveShadowMap: bool = False
    bClampToNearPlane:          bool = False
    LODDistanceFactor:  float = _SHADOW_LOD_DISTANCE_FACTOR


def _setup_shadow_depth_pass_ub(
    shadow: "AstroCellShadowInfo",
    viewport_w: float,
    viewport_h: float,
) -> AstroCellShadowDepthPassParams:
    """
    Fill shadow depth pass uniform buffer parameters.

    Mirrors SetupShadowDepthPassUniformBuffer() which populates
    FShadowDepthPassUniformParameters with the shadow projection matrix,
    shadow bias, and LOD distance factor.

    In 2-D: the projection matrix is an orthographic projection from
    world space onto the shadow map rect.

    鲁迅式：正交投影是平等主义者——无论远近，所有点都被相同地对待。
    这在阴影贴图里是美德，在现实里则是幻觉。
    """
    # Build ortho projection: maps viewport to [-1,1] clip space
    # (simplified 2-D analogue of FMatrix::OrthoMatrix)
    inv_w = 2.0 / max(viewport_w, 1.0)
    inv_h = 2.0 / max(viewport_h, 1.0)

    proj = (
        inv_w,  0.0,    0.0, -1.0,
        0.0,    inv_h,  0.0, -1.0,
        0.0,    0.0,    1.0,  0.0,
        0.0,    0.0,    0.0,  1.0,
    )

    # Shadow bias: small positive value to avoid self-shadowing (acne)
    shadow_bias = 0.002 + 0.001 * len(shadow.casters) / max(100, len(shadow.casters))

    return AstroCellShadowDepthPassParams(
        ProjectionMatrix=proj,
        ShadowParams=(shadow_bias, 0.0, 0.0, 0.0),
        LODDistanceFactor=_SHADOW_LOD_DISTANCE_FACTOR,
    )


class AstroCellShadowDepthRenderer:
    """
    Python equivalent of the TRenderShadowDepths template + helper functions
    in ShadowDepthRendering.cpp.

    Drives the shadow depth rendering pass for a single AstroCellShadowInfo.
    Outputs a per-cell shadow depth dict and writes it to the channel.

    Lifecycle:
        renderer = AstroCellShadowDepthRenderer(shadow, viewport_w, viewport_h)
        depth_result = renderer.render(cell_entries)

    鲁迅式：深度渲染器是执行者——它不问为什么，只问怎么做。
    这是效率，也是盲目。我们需要它，但不应成为它。
    """

    def __init__(
        self,
        shadow: "AstroCellShadowInfo",
        viewport_w: float,
        viewport_h: float,
    ) -> None:
        self._shadow      = shadow
        self._viewport_w  = viewport_w
        self._viewport_h  = viewport_h
        self._ub_params   = _setup_shadow_depth_pass_ub(shadow, viewport_w, viewport_h)

    def render(self, cell_entries: list) -> dict:
        """
        Execute the shadow depth pass and return per-cell depth results.

        Mirrors the mesh draw loop in FShadowDepthPassMeshProcessor::Process():
          1. For each caster in shadow.casters, look up its bbox.
          2. Project the caster's world position through the shadow projection
             matrix (using ub_params.ProjectionMatrix).
          3. Apply shadow bias (ShadowParams.x) to prevent self-shadowing.
          4. Write the result to the depth_result dict.
          5. Apply LOD-distance based detail reduction (simplified to a
             screen-fraction flag that later SVG generators can consume).

        Returns dict: caster_id → {depth, lod_level, bias_applied}.

        鲁迅式：深度值是精确的谎言——它比较的是「在光源眼中的远近」，
        而非真实的世界距离；但这个谎言，足以产生正确的阴影。
        """
        if not self._shadow.casters:
            return {}

        bbox_by_cid = {
            e["cell_id"]: e["bbox"]
            for e in cell_entries
            if "cell_id" in e
        }

        bias = self._ub_params.ShadowParams[0]
        viewport_area = max(self._viewport_w * self._viewport_h, 1.0)

        depth_result: dict = {}
        for cid in self._shadow.casters:
            bbox = bbox_by_cid.get(cid, {})
            if not bbox:
                continue

            x = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
            y = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
            z = float(bbox.get("z", 3))

            # Project through ortho matrix (simplified row-major multiply)
            pm = self._ub_params.ProjectionMatrix
            clip_x = pm[0] * x + pm[3]
            clip_y = pm[5] * y + pm[7]
            # depth ∈ [0, 1] from normalised clip Z (≈ z / z_max)
            raw_depth = max(0.0, min(1.0, z / 7.0))   # z ∈ [0,7]
            biased_depth = max(0.0, min(1.0, raw_depth + bias))

            # LOD selection distance
            cell_diagonal = math.sqrt(
                float(bbox.get("w", 1.0))**2 + float(bbox.get("h", 1.0))**2
            )
            lod_dist = _shadow_lod_distance(cell_diagonal)
            screen_frac = (float(bbox.get("w", 0)) * float(bbox.get("h", 0))) / viewport_area
            lod_level = (
                0 if screen_frac >= 0.005 else
                1 if screen_frac >= 0.001 else
                2
            )

            depth_result[cid] = {
                "depth":         round(biased_depth, 6),
                "raw_depth":     round(raw_depth, 6),
                "bias":          round(bias, 6),
                "lod_level":     lod_level,
                "clip_xy":       (round(clip_x, 4), round(clip_y, 4)),
            }

        # Merge back into shadow's depth_map (renderer wins over setup estimate)
        for cid, d in depth_result.items():
            self._shadow.depth_map[cid] = d["depth"]

        print(
            f"[ASTRO-SHADOW] ShadowDepthRenderer: shadow_id={self._shadow.shadow_id} "
            f"rendered={len(depth_result)} bias={bias:.5f}",
            file=sys.stderr,
        )
        return depth_result


# ---------------------------------------------------------------------------
# DeferredShadingRenderer.cpp — G-Buffer + Render pipeline
#
# 鲁迅式：延迟着色是一种制度的产物——先收集，再计算。
# 收集时代价低廉；计算时代价高昂，却只在真正需要时支付。
# 这是聪明的经济学，也是对「及时付账」的反叛。
# ---------------------------------------------------------------------------

#: Mirrors CVarNanitePrimeHZBMode (r.Nanite.PrimeHZB).
#: 0 = off, 1 = run if no HZB, 2 = always
_NANITE_PRIME_HZB_MODE: int = 0


@dataclass
class AstroCellGBuffer:
    """
    Python equivalent of the UE5 G-Buffer layout (GBufferA/B/C/D).

    In UE5, 4 RGBA render targets carry:
      GBufferA → BaseColor (RGB) + Shading model (A)
      GBufferB → Metallic (R) + Specular (G) + Roughness (B) + Shading (A)
      GBufferC → WorldNormal (RGB) + Ambient Occlusion (A)
      GBufferD → Custom data (per-shading-model payload)

    2-D SVG adaptation:
      base_color      → hex string (primary fill colour)
      metallic        → float [0,1] from _species_f0
      specular        → float [0,1]
      roughness       → float [0,1] from _SPECIES_ROUGHNESS_SEC
      world_normal    → 2-D (nx, ny) unit vector (faces viewer = (0,1))
      ambient_occlusion → crowding_opacity complement
      shading_model   → int (0=Unlit, 1=Lit, 3=SubsurfaceProfile …)
      custom_data     → species-specific float4 payload

    鲁迅式：G-Buffer 是场景的档案馆——每个像素的材质属性都在此存档，
    等待光照计算的审判。档案馆本身不发光，只是记录。
    """
    # GBufferA
    base_color:      str   = "#888888"   # hex
    shading_model:   int   = 1           # 1 = DefaultLit

    # GBufferB
    metallic:        float = 0.0
    specular:        float = 0.5
    roughness:       float = 0.5
    anisotropy:      float = 0.0

    # GBufferC
    world_normal:    tuple = (0.0, 1.0)  # 2-D (nx, ny), faces upward
    ambient_occlusion: float = 1.0       # 1 = fully lit

    # GBufferD (custom per-shading-model)
    custom_data:     tuple = (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "base_color":    self.base_color,
            "shading_model": self.shading_model,
            "metallic":      round(self.metallic, 4),
            "specular":      round(self.specular, 4),
            "roughness":     round(self.roughness, 4),
            "world_normal":  list(self.world_normal),
            "ao":            round(self.ambient_occlusion, 4),
            "custom_data":   list(self.custom_data),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AstroCellGBuffer":
        gb = cls()
        gb.base_color        = d.get("base_color", "#888888")
        gb.shading_model     = d.get("shading_model", 1)
        gb.metallic          = float(d.get("metallic", 0.0))
        gb.specular          = float(d.get("specular", 0.5))
        gb.roughness         = float(d.get("roughness", 0.5))
        gb.ambient_occlusion = float(d.get("ao", 1.0))
        gb.world_normal      = tuple(d.get("world_normal", [0.0, 1.0]))
        gb.custom_data       = tuple(d.get("custom_data", [0.0, 0.0, 0.0, 0.0]))
        return gb


class AstroCellDeferredShadingRenderer:
    """
    Python equivalent of FDeferredShadingSceneRenderer.

    Orchestrates the full deferred shading pipeline for a single epoch:
      1. _init_views()               → InitViews / visibility culling
      2. _render_shadow_depth_maps() → InitDynamicShadows + RenderShadowDepthMaps
      3. _render_base_pass()         → RenderBasePass (fills G-Buffer)
      4. _render_lights()            → RenderLights (deferred + simple)
      5. _render_post_processing()   → RenderPostProcessing (AA, tonemap, bloom)

    The output is a per-cell render_result dict and a composed SVG comment
    block that can be injected into the frame SVG for diagnostics.

    Lifecycle::
        dsr = AstroCellDeferredShadingRenderer(viewport_w, viewport_h)
        results = dsr.render(cell_entries, frame_index)

    鲁迅式：延迟着色渲染器是一个官僚机构——
    五个部门，各司其职，没有一个部门能独立完成工作，
    但合在一起，它们生产出可以展示于世的画面。
    这是分工合作的奇迹，也是分工合作的悲剧。
    """

    def __init__(
        self,
        viewport_w: float = 1200.0,
        viewport_h: float = 900.0,
        enable_hzb_prime: bool = (_NANITE_PRIME_HZB_MODE > 0),
    ) -> None:
        self.viewport_w    = viewport_w
        self.viewport_h    = viewport_h
        self.enable_hzb_prime = enable_hzb_prime

        # State built during render()
        self._visibility_query: "AstroCellVisibilityQuery | None" = None
        self._shadow:           "AstroCellShadowInfo | None"      = None
        self._shadow_renderer:  "AstroCellShadowDepthRenderer | None" = None
        self._gbuffers:         dict = {}   # cell_id → AstroCellGBuffer
        self._light_pass:       "AstroCellLightPass | None" = None
        self._frame_stats:      dict = {}

    # ------------------------------------------------------------------
    # Phase 1: InitViews
    # ------------------------------------------------------------------

    def _init_views(self, cell_entries: list) -> "AstroCellVisibilityQuery":
        """
        Mirrors FDeferredShadingSceneRenderer::InitViews().

        Runs the Nanite visibility query (perform_nanite_visibility equivalent)
        and optionally primes the HZB if _NANITE_PRIME_HZB_MODE > 0.

        Returns an AstroCellVisibilityQuery with per-cell results.

        鲁迅式：InitViews 是开门迎客——先看清楚谁在门口，
        再决定让谁进来，让谁站在外面等。
        """
        query = AstroCellVisibilityQuery(
            self.viewport_w, self.viewport_h
        )

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            bbox    = entry.get("bbox", {})
            species = entry.get("species", "")
            z_layer = int(entry.get("z_layer", bbox.get("z", 3)))

            if not cid or not bbox:
                continue

            query.test_cell(cid, bbox, species, z_layer)

        query.finish()

        # HZB prime pass (mirrors CVarNanitePrimeHZB: run if no HZB available)
        if self.enable_hzb_prime:
            print(
                f"[ASTRO-DSR] InitViews: HZB prime pass (mode={_NANITE_PRIME_HZB_MODE}) "
                f"cells={len(cell_entries)}",
                file=sys.stderr,
            )

        return query

    # ------------------------------------------------------------------
    # Phase 2: RenderShadowDepthMaps
    # ------------------------------------------------------------------

    def _render_shadow_depth_maps(
        self, cell_entries: list, frame_index: int
    ) -> "AstroCellShadowInfo":
        """
        Mirrors FDeferredShadingSceneRenderer::RenderShadowDepthMaps().

        Calls build_whole_scene_shadow_caster() then constructs an
        AstroCellShadowDepthRenderer and executes the depth pass.

        Returns the populated AstroCellShadowInfo.

        鲁迅式：阴影深度图是过去的快照——此刻的光，照出昨日的形状。
        但「昨日」在渲染器里只有一帧之差，所以它总是足够接近真实。
        """
        shadow = build_whole_scene_shadow_caster(
            self.viewport_w, self.viewport_h, cell_entries, frame_index
        )

        if not shadow.is_cached:
            renderer = AstroCellShadowDepthRenderer(
                shadow, self.viewport_w, self.viewport_h
            )
            renderer.render(cell_entries)
            self._shadow_renderer = renderer

        return shadow

    # ------------------------------------------------------------------
    # Phase 3: RenderBasePass → fill G-Buffer
    # ------------------------------------------------------------------

    def _render_base_pass(
        self,
        cell_entries: list,
        vis_query: "AstroCellVisibilityQuery",
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderBasePass().

        Iterates visible cells, derives G-Buffer parameters from species
        data and the visibility query, and populates self._gbuffers.

        In UE5, the Base Pass writes BaseColor, Normal, Roughness, Metallic,
        AO, and custom shading model data into the G-Buffer render targets.
        Here we produce an AstroCellGBuffer per visible cell.

        Returns dict: cell_id → AstroCellGBuffer.

        鲁迅式：Base Pass 是材质信息的总动员——
        每个表面把自己的颜色、粗糙度、法线全部交出来，
        放入 G-Buffer 这个大型档案箱，等待光照部门的审阅。
        """
        _ROUGHNESS_MAP = {
            "cil-eye": 0.1, "cil-bolt": 0.2, "cil-plus": 0.3,
            "cil-vector": 0.5, "cil-arrow-right": 0.7,
            "cil-filter": 0.3, "cil-code": 0.4, "cil-layers": 0.2,
            "cil-loop": 0.5, "cil-graph": 0.6,
        }

        gbuffers: dict = {}

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            species = entry.get("species", "")
            bbox    = entry.get("bbox", {})

            if not cid:
                continue

            vis_result = vis_query.cell_results.get(cid, {})
            if vis_result.get("lod", 0) == -1:
                continue  # culled — no G-Buffer entry

            sp_idx = _species_to_index(species)
            colour = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            gbuf = AstroCellGBuffer(
                base_color      = _colour_to_hex(colour),
                shading_model   = 1,
                metallic        = _species_f0(species),
                specular        = 0.5,
                roughness       = _ROUGHNESS_MAP.get(species, 0.5),
                world_normal    = (0.0, 1.0),
                ambient_occlusion = entry.get("crowding_opacity", 1.0),
                custom_data     = (float(bbox.get("z", 3)), 0.0, 0.0, 0.0),
            )
            gbuffers[cid] = gbuf

        self._gbuffers = gbuffers

        print(
            f"[ASTRO-DSR] BasePass: gbuffer_entries={len(gbuffers)} "
            f"total_cells={len(cell_entries)}",
            file=sys.stderr,
        )
        return gbuffers

    # ------------------------------------------------------------------
    # Phase 4: RenderLights (deferred light pass over G-Buffer)
    # ------------------------------------------------------------------

    def _render_lights(
        self,
        cell_entries: list,
        shadow: "AstroCellShadowInfo",
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderLights().

        Iterates the G-Buffer and evaluates the primary deferred light
        contribution for each cell, modulated by the shadow depth map.

        Shadow modulation (mirrors the shadow term in the BRDF evaluation):
            shadow_term = shadow.sample(caster_id)
            diffuse *= shadow_term

        Returns dict: cell_id → light_result dict.

        鲁迅式：灯光是权力——它决定谁被照亮，谁留在阴影之中。
        G-Buffer 给了每个人平等的档案；光照，决定了最终的命运。
        """
        light_pass = AstroCellLightPass()
        light_results: dict = {}

        bbox_by_cid = {e["cell_id"]: e["bbox"] for e in cell_entries if "cell_id" in e}
        sibling_bboxes = {
            e["cell_id"]: e["bbox"]
            for e in cell_entries if "cell_id" in e and e.get("bbox")
        }

        for cid, gbuf in self._gbuffers.items():
            bbox    = bbox_by_cid.get(cid, {})
            species = next(
                (e["species"] for e in cell_entries if e.get("cell_id") == cid),
                ""
            )

            result = light_pass.render_light(
                cid, bbox, species, gbuf.roughness, sibling_bboxes
            )

            # Apply shadow modulation: shadow term from depth map
            shadow_term = shadow.sample(cid)
            if shadow_term < 0.99:
                # Darken diffuse by shadow term (mirrors shadow masking in HLSL)
                orig_hex = result.get("diffuse_color", "#000000")
                def _dim(ch_val: int) -> int:
                    return max(0, int(ch_val * shadow_term))
                try:
                    r = _dim(int(orig_hex[1:3], 16))
                    g = _dim(int(orig_hex[3:5], 16))
                    b = _dim(int(orig_hex[5:7], 16))
                    result["diffuse_color"] = "#{:02X}{:02X}{:02X}".format(r, g, b)
                    result["shadow_term"]   = round(shadow_term, 4)
                except (ValueError, IndexError):
                    pass

            light_results[cid] = result

        self._light_pass = light_pass

        print(
            f"[ASTRO-DSR] RenderLights: light_results={len(light_results)} "
            f"shadow_casters={len(shadow.casters)} cached={shadow.is_cached}",
            file=sys.stderr,
        )
        return light_results

    # ------------------------------------------------------------------
    # Phase 5: RenderPostProcessing (minimal AA + tonemap)
    # ------------------------------------------------------------------

    def _render_post_processing(
        self, light_results: dict
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderPostProcessing().

        Applies a simple filmic tonemap (Reinhard) and ambient occlusion
        attenuation to the final per-cell colour values.

        Tonemap: L_out = L_in / (1 + L_in)   (Reinhard global)
        AO attenuation: colour *= ao_factor

        Returns dict: cell_id → post_result dict (adds 'final_color' key).

        鲁迅式：后处理是面具——它让生硬的数字看起来像自然的颜色。
        没有它，画面是正确的；有了它，画面才是美的。
        这是一种用于欺骗眼睛的善意谎言。
        """
        post_results: dict = {}

        for cid, lr in light_results.items():
            gbuf = self._gbuffers.get(cid)
            ao   = gbuf.ambient_occlusion if gbuf else 1.0

            # Parse diffuse colour
            hex_col = lr.get("diffuse_color", "#000000")
            try:
                r_lin = int(hex_col[1:3], 16) / 255.0
                g_lin = int(hex_col[3:5], 16) / 255.0
                b_lin = int(hex_col[5:7], 16) / 255.0
            except (ValueError, IndexError):
                r_lin, g_lin, b_lin = 0.0, 0.0, 0.0

            # Reinhard tonemap
            r_tm = r_lin / (1.0 + r_lin)
            g_tm = g_lin / (1.0 + g_lin)
            b_tm = b_lin / (1.0 + b_lin)

            # AO attenuation
            r_ao = r_tm * ao
            g_ao = g_tm * ao
            b_ao = b_tm * ao

            final_hex = "#{:02X}{:02X}{:02X}".format(
                max(0, min(255, int(r_ao * 255))),
                max(0, min(255, int(g_ao * 255))),
                max(0, min(255, int(b_ao * 255))),
            )

            post_results[cid] = {
                **lr,
                "final_color":  final_hex,
                "ao_factor":    round(ao, 4),
                "tonemap":      "reinhard",
            }

        print(
            f"[ASTRO-DSR] PostProcessing: tonemapped={len(post_results)} mode=reinhard",
            file=sys.stderr,
        )
        return post_results

    # ------------------------------------------------------------------
    # Main render() entry point
    # ------------------------------------------------------------------

    def render(
        self,
        cell_entries: list,
        frame_index: int = 0,
    ) -> dict:
        """
        Execute the full deferred shading pipeline for one frame.

        Mirrors FDeferredShadingSceneRenderer::Render():
          Phase 1 → _init_views()
          Phase 2 → _render_shadow_depth_maps()
          Phase 3 → _render_base_pass()
          Phase 4 → _render_lights()
          Phase 5 → _render_post_processing()

        Returns dict: cell_id → post-processed render result.

        鲁迅式：Render() 是总开关——按下去，五个部门同时运转，
        各自沉默地完成职责，最终交出一帧可以示人的画面。
        没有英雄，只有流程；没有奇迹，只有正确的顺序。
        """
        self._frame_stats = {"frame": frame_index}

        # Phase 1: visibility
        vis_query = self._init_views(cell_entries)
        self._visibility_query = vis_query
        self._frame_stats["visible_cells"] = vis_query.finish().get(
            "visible_cells", 0
        ) if not vis_query._finished else sum(
            1 for r in vis_query.cell_results.values() if r.get("visible")
        )

        # Phase 2: shadow depths
        shadow = self._render_shadow_depth_maps(cell_entries, frame_index)
        self._shadow = shadow
        self._frame_stats["shadow_casters"] = len(shadow.casters)
        self._frame_stats["shadow_cached"]  = shadow.is_cached

        # Phase 3: base pass
        gbuffers = self._render_base_pass(cell_entries, vis_query)
        self._frame_stats["gbuffer_entries"] = len(gbuffers)

        # Phase 4: lights
        light_results = self._render_lights(cell_entries, shadow)
        self._frame_stats["light_results"] = len(light_results)

        # Phase 5: post-processing
        final_results = self._render_post_processing(light_results)
        self._frame_stats["post_results"] = len(final_results)

        print(
            f"[ASTRO-DSR] Render COMPLETE: frame={frame_index} "
            f"visible={self._frame_stats['visible_cells']} "
            f"shadow_casters={self._frame_stats['shadow_casters']} "
            f"gbuffers={self._frame_stats['gbuffer_entries']} "
            f"light={self._frame_stats['light_results']} "
            f"post={self._frame_stats['post_results']}",
            file=sys.stderr,
        )

        return final_results

    def frame_stats(self) -> dict:
        """Return stats from the last render() call."""
        return dict(self._frame_stats)


def run_deferred_shading_pipeline(
    cell_entries: list,
    frame_index: int = 0,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> dict:
    """
    Top-level entry point for the deferred shading pipeline.

    Convenience wrapper that constructs AstroCellDeferredShadingRenderer,
    calls render(), and returns the final per-cell result dict.

    Mirrors the call site in the engine's Tick() that dispatches
    FDeferredShadingSceneRenderer::Render() on the render thread.

    鲁迅式：入口函数是门面——它让外部世界相信内部是有序的。
    无论内部多么复杂，门面保持镇定，只交出一个 dict。
    """
    dsr = AstroCellDeferredShadingRenderer(viewport_w, viewport_h)
    return dsr.render(cell_entries, frame_index)


# =============================================================================
# [ASTRO-CELL] MeshPassProcessor + MeshDrawCommands + SceneCapture +
#              ReflectionEnvironmentCapture → Python port
#
# Ported from (commit heads at time of writing):
#   upstream/unreal-renderer-ue5/Renderer-Private/MeshPassProcessor.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/MeshDrawCommands.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/SceneCaptureRendering.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentCapture.cpp
#
# 鲁迅曾言：「有谁从小康人家而坠入困顿的么，我以为在这途路中，大概
# 可以看见世人的真面目。」
# MeshPassProcessor 亦然——当 PSO 状态切换压力在此集中，你才看清
# draw call 分类与合批的真实代价。每一条 mesh draw command 都是一次
# 对 GPU 状态机的主张；合并它们，是工程师对效率的永恒追求。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
# [MeshPassProcessor]
#   FGraphicsMinimalPipelineStateId   → AstroCellPipelineStateId
#     PersistentIdTable (RWLock)      → _pipeline_state_table (dict + lock-free)
#     bIsIdTableFrozen                → _pso_table_frozen flag
#   FReadOnlyMeshDrawSingleShaderBindings.SetShaderBindings()
#                                     → AstroCellShaderBindings.apply()
#   GEmitMeshDrawEvent (CVar)         → ASTRO_EMIT_MESH_DRAW_EVENT
#   GSkipDrawOnPSOPrecaching (CVar)   → ASTRO_SKIP_DRAW_ON_PSO_PRECACHING
#   FMeshDrawCommandSortKey           → AstroCellDrawSortKey
#
# [MeshDrawCommands]
#   FPrimitiveIdVertexBufferPool      → AstroCellPrimitiveIdBufferPool
#     Allocate / ReturnToFreeList     → allocate / release
#     DiscardAll (discard_id)         → discard_stale (epoch-based)
#   UpdateTranslucentMeshSortKeys()   → update_translucent_sort_keys()
#   BitInvertIfNegativeFloat()        → _bit_invert_if_negative()
#   CVarMeshSortingMethodWithoutEarlyZ → ASTRO_MESH_SORT_METHOD
#   CVarDeferredMeshPassSetupTaskSync  → ASTRO_DEFERRED_MESH_PASS_SYNC
#
# [SceneCaptureRendering]
#   GSceneCaptureAllowRenderInMainRenderer → ASTRO_CAPTURE_ALLOW_MAIN_RENDERER
#   GSceneCaptureCubeSinglePass            → ASTRO_CAPTURE_CUBE_SINGLE_PASS
#   FSceneCapturePS (pixel shader)         → AstroCellCaptureProcessor
#     ESourceMode (8 modes)                → AstroCellCaptureMode (enum)
#     ShouldCompilePermutation()           → should_compile_permutation()
#     GetPermutationVector()               → get_permutation()
#   CopyCaptureToTarget()                  → copy_capture_to_target()
#   UpdateSceneCaptureContents()           → update_scene_capture_contents()
#
# [ReflectionEnvironmentCapture]
#   GReflectionCaptureNearPlane            → _REFL_NEAR_PLANE
#   GSupersampleCaptureFactor (1..8)       → _REFL_SUPERSAMPLE_FACTOR
#   CVarReflectionCaptureRuntimeTimeslice  → _REFL_TIMESLICE_FACES
#   CVarReflectionCaptureRuntimeMode       → _REFL_RUNTIME_MODE
#   CVarReflectionCaptureRuntimeBudget     → _REFL_BUDGET
#   FAstroCellReflectionCaptureState       → AstroCellReflectionCaptureState
#   CaptureSceneToScratchCubemap()         → capture_scene_to_scratch_cubemap()
#   UpdateReflectionCaptures()             → update_reflection_captures()
#
# 2-D SVG channel adaptation:
#   GPU StructuredBuffer   → list of dicts in memory / JSON channel
#   Cubemap face           → per-Z-layer "face" (6 z-layer offsets)
#   PSO state key          → (species, blend_mode, pass_name) 3-tuple
#   RHI draw submission    → write to cell/*/svg.svg channel
#   Persistent table lock  → atomic integer epoch counter (no true lock needed
#                            in single-threaded epoch loop)
# =============================================================================

import struct as _struct

# ---------------------------------------------------------------------------
# Global flags — mirrors CVars from all four source files
# ---------------------------------------------------------------------------

#: Mirrors GEmitMeshDrawEvent — emit per-draw debug annotations in SVG.
ASTRO_EMIT_MESH_DRAW_EVENT: bool = False

#: Mirrors GSkipDrawOnPSOPrecaching — skip cells whose PSO is still compiling.
ASTRO_SKIP_DRAW_ON_PSO_PRECACHING: bool = False

#: Mirrors CVarMeshSortingMethodWithoutEarlyZ (0=state+Z, 1=strict Z).
ASTRO_MESH_SORT_METHOD: int = 0

#: Mirrors CVarDeferredMeshPassSetupTaskSync — defer batch sort to RDG exec.
ASTRO_DEFERRED_MESH_PASS_SYNC: bool = True

#: Mirrors GSceneCaptureAllowRenderInMainRenderer.
ASTRO_CAPTURE_ALLOW_MAIN_RENDERER: bool = True

#: Mirrors GSceneCaptureCubeSinglePass — all 6 faces in one pass.
ASTRO_CAPTURE_CUBE_SINGLE_PASS: bool = True

#: Mirrors GReflectionCaptureNearPlane (world units).
_REFL_NEAR_PLANE: float = 5.0

#: Mirrors GSupersampleCaptureFactor (clamped to [1, 8]).
_REFL_SUPERSAMPLE_FACTOR: int = 1

#: Mirrors CVarReflectionCaptureRuntimeTimeslice — faces rendered per frame.
_REFL_TIMESLICE_FACES: int = 1

#: Mirrors CVarReflectionCaptureRuntimeMode (0=Continuous, 1=Once).
_REFL_RUNTIME_MODE: int = 1

#: Mirrors CVarReflectionCaptureRuntimeBudget (0=unlimited).
_REFL_BUDGET: int = 0


# =============================================================================
# [MeshDrawCommands] BitInvertIfNegativeFloat + sort-key helpers
# =============================================================================
