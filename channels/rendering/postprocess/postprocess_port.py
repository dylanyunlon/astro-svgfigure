import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



class AstroCellDenoiserManager:
    """
    Python equivalent of FDenoiserManager.

    Registry for spatial and spatio-temporal denoiser plugins.
    Thread-safe in C++; single-threaded singleton here (epoch loop).

    鲁迅式：管理者的职责是注册、查询、注销——
    如同一个公正的仲裁者，自己不参与战斗，只确保战斗有规则。
    """

    def __init__(self) -> None:
        # Spatial denoisers: name → callable(radiance_buf, albedo_buf, normal_buf)
        self._spatial:           _PTDict[str, object] = {}
        # Spatio-temporal denoisers: name → callable(radiance_buf, history_buf, var_buf)
        self._spatial_temporal:  _PTDict[str, object] = {}
        # Whether any registered denoiser needs extra texture creation flags
        self.need_extra_flags: bool = False

    def register_spatial_denoiser(self, name: str, denoiser_fn,
                                   needs_extra_flags: bool = False) -> None:
        """
        Register a spatial denoiser plugin.
        Mirrors RegisterSpatialDenoiser(TUniquePtr<IPathTracingDenoiser>, FString).
        """
        assert name not in self._spatial, f"Denoiser '{name}' already registered"
        self._spatial[name] = denoiser_fn
        self.need_extra_flags |= needs_extra_flags
        print(
            f"[ASTRO-PTD] RegisterSpatialDenoiser name={name} "
            f"need_extra_flags={needs_extra_flags}",
            file=sys.stderr,
        )

    def register_spatial_temporal_denoiser(self, name: str, denoiser_fn,
                                            needs_extra_flags: bool = False) -> None:
        """
        Register a spatio-temporal denoiser plugin.
        Mirrors RegisterSpatialTemporalDenoiser(TUniquePtr<IPathTracingSpatialTemporalDenoiser>…).
        """
        assert name not in self._spatial_temporal, (
            f"S-T Denoiser '{name}' already registered")
        self._spatial_temporal[name] = denoiser_fn
        self.need_extra_flags |= needs_extra_flags

    def unregister_denoiser(self, name: str) -> None:
        """Remove a denoiser by name from both registries."""
        self._spatial.pop(name, None)
        self._spatial_temporal.pop(name, None)

    def has_spatial_denoiser(self) -> bool:
        return bool(self._spatial)

    def has_spatial_temporal_denoiser(self) -> bool:
        return bool(self._spatial_temporal)

    def has_denoiser(self) -> bool:
        return self.has_spatial_denoiser() or self.has_spatial_temporal_denoiser()

    def get_spatial_denoiser(self, name: str, exact_match: bool = False):
        """
        Return denoiser plugin by name; falls back to first registered if not exact.
        Mirrors FDenoiserManager::GetSpatialDenoiser(FString Name, bool bMatch).
        """
        if name in self._spatial:
            return self._spatial[name]
        if not exact_match and self._spatial:
            return next(iter(self._spatial.values()))
        return None

    def get_spatial_temporal_denoiser(self, name: str, exact_match: bool = False):
        if name in self._spatial_temporal:
            return self._spatial_temporal[name]
        if not exact_match and self._spatial_temporal:
            return next(iter(self._spatial_temporal.values()))
        return None


# Module-level singleton — mirrors the static FDenoiserManager instance
_ASTRO_DENOISER_MANAGER: AstroCellDenoiserManager = AstroCellDenoiserManager()





def get_denoiser_manager() -> AstroCellDenoiserManager:
    """Return the global AstroCellDenoiserManager singleton."""
    return _ASTRO_DENOISER_MANAGER


# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────




def run_spatial_temporal_denoising(
    state:       AstroCellPathTracingState,
    all_bboxes:  dict,
    bvh:         "AstroCellBVH | None" = None,
    motion_vectors: _PTDict[str, tuple] | None = None,
) -> _PTDict[str, tuple]:
    """
    Full spatio-temporal denoising pipeline for all cells.

    Orchestrates the five passes in order, matching the compute-pass dispatch
    sequence in PathTracingSpatialTemporalDenoising.cpp:
      1. temporal_reprojection_align
      2. temporal_reprojection_blur
      3. temporal_reprojection_merge
      4. high_frequency_reject_map
      5. temporal_feature_fusion

    After fusion, the result is stored in state.last_denoised for the next
    frame's history (mirrors LastDenoisedRadianceRT update).

    Returns the final denoised radiance dict.

    鲁迅式：五道工序，缺一不可——就如同一篇文章，
    初稿之后还需修改、再修改、校对、排版，才能印出来给人看。
    """
    if not state.radiance_buffer:
        return {}

    mv = motion_vectors or {}

    # Pass 1: Temporal alignment
    aligned = temporal_reprojection_align(
        state.radiance_buffer, state.last_denoised, mv)

    # Pass 2: History blur
    blurred = temporal_reprojection_blur(aligned, bvh, all_bboxes)

    # Pass 3: Temporal merge with variance adaptation
    merged = temporal_reprojection_merge(
        state.radiance_buffer, blurred, state.variance_buffer)

    # Pass 4: High-frequency reject map
    accept = high_frequency_reject_map(
        state.radiance_buffer, state.last_denoised, state.variance_buffer)

    # Pass 5: Feature fusion
    denoised = temporal_feature_fusion(merged, accept, state.last_denoised)

    # Update last-denoised cache for next frame
    state.last_denoised.update(denoised)

    total    = len(denoised)
    accepted = sum(1 for v in accept.values() if v > 0.5)
    print(
        f"[ASTRO-PTD] run_spatial_temporal_denoising: "
        f"total_cells={total} accepted_fresh={accepted} "
        f"temporal_blended={total - accepted} "
        f"spp={state.sample_index}",
        file=sys.stderr,
    )

    return denoised


# =============================================================================
# [ASTRO-CELL] ReflectionEnvironmentCapture → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentCapture.cpp
#
# 鲁迅曾言：「希望本是无所谓有，无所谓无的。这正如地上的路；
# 其实地上本没有路，走的人多了，也便成了路。」
# 反射探针亦然——世界本无镜，捕获得多了，也便成了反射。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GSupersampleCaptureFactor       → ASTRO_CAPTURE_SUPERSAMPLE_FACTOR
#   GReflectionCaptureNearPlane     → ASTRO_CAPTURE_NEAR_PLANE
#   CVarReflectionCaptureRuntimeTimeslice → ASTRO_CAPTURE_TIMESLICE_FACES
#   CaptureSceneToScratchCubemap    → capture_scene_to_scratch_cubemap()
#   ConvolveCubeMap                 → convolve_cube_map()
#   FindOrAllocateCubemapIndex      → find_or_allocate_cubemap_index()
#   ComputeRuntimeBudgetSignedDistance → compute_capture_priority()
#   FCaptureComponentSceneState     → AstroCellCaptureState
#   FReflectionSceneData            → AstroCellReflectionSceneData
#   BeginReflectionCaptureSlowTask  → begin_capture_task() (log-only)
#   UpdateReflectionCaptureSlowTask → update_capture_task()
#   EndReflectionCaptureSlowTask    → end_capture_task()
#
# 2-D SVG adaptation:
#   CubemapArray[face][mip]  → per-cell specular probe dict  (6 faces × N mips)
#   Radiance SH L2 (9 coeff) → 3-component SH L1 (3 floats per channel = 9)
#   Downsample mip pass      → gaussian_downsample_face_mip()
#   Convolve specular face   → convolve_specular_face()
#   Diffuse irradiance SH    → compute_diffuse_irradiance_sh()
# =============================================================================

# CVarReflectionCapture equivalents
ASTRO_CAPTURE_NEAR_PLANE:       float = 5.0     # GReflectionCaptureNearPlane
ASTRO_CAPTURE_SUPERSAMPLE_MIN:  int   = 1       # MinSupersampleCaptureFactor
ASTRO_CAPTURE_SUPERSAMPLE_MAX:  int   = 8       # MaxSupersampleCaptureFactor
ASTRO_CAPTURE_SUPERSAMPLE:      int   = 1       # GSupersampleCaptureFactor
ASTRO_CAPTURE_TIMESLICE_FACES:  int   = 2       # CVarReflectionCaptureRuntimeTimeslice
ASTRO_CAPTURE_TIMESLICE_EDITOR: int   = 3       # CVarReflectionCaptureRuntimeTimesliceEditor
ASTRO_CAPTURE_TIMESLICE_SLOW:   bool  = False   # CVarReflectionCaptureRuntimeTimesliceSlow
ASTRO_CAPTURE_FADE_TIME:        float = 0.5     # CVarReflectionCaptureRuntimeFadeInTime
ASTRO_CAPTURE_BUDGET:           int   = 0       # CVarReflectionCaptureRuntimeBudget (0=unlimited)
ASTRO_CAPTURE_FOLIAGE:          bool  = False   # CVarReflectionCaptureRuntimeFoliage
ASTRO_CAPTURE_TRANSLUCENCY:     bool  = False   # CVarReflectionCaptureRuntimeTranslucency
ASTRO_CAPTURE_MODE:             int   = 1       # 0=continuous, 1=once
ASTRO_CAPTURE_FAST_ON_LOAD:     int   = 3       # CVarReflectionCaptureRuntimeFastRenderOnLoad
# Cube faces: +X,-X,+Y,-Y,+Z,-Z (indices 0..5)
_CAPTURE_NUM_FACES:             int   = 6
_CAPTURE_NUM_MIPS:              int   = 7       # mip 0..6 for 128×128 cube (log2(128)+1)


@_ptdc



def is_motion_blur_enabled(quality: int = _MB_QUALITY) -> bool:
    """
    Mirrors IsMotionBlurEnabled(FViewInfo&):
        FeatureLevel >= SM5 AND PostProcessing AND MotionBlur flags
        AND Amount > 0.001 AND Max > 0.001 AND bRealtimeUpdate AND Quality > 0
    """
    return (quality > 0 and
            _MB_AMOUNT > 0.001 and
            _MB_MAX_VEL_FRACTION > 0.001)





def get_motion_blur_tile_count(width: int, height: int) -> tuple[int, int]:
    """
    Compute tile grid dimensions for velocity flatten.
    Mirrors GetMotionBlurTileCount(FIntPoint):
        TilesX = DivideAndRoundUp(W, kFlattenTileSize)
        TilesY = DivideAndRoundUp(H, kFlattenTileSize)

    鲁迅式：瓦片是统治单元——一个个 16×16 的小格子，
    把无限的屏幕空间分割成有限的可管理区域。
    """
    tx = (width  + _MB_FLATTEN_TILE_SIZE - 1) // _MB_FLATTEN_TILE_SIZE
    ty = (height + _MB_FLATTEN_TILE_SIZE - 1) // _MB_FLATTEN_TILE_SIZE
    return (tx, ty)





def compute_motion_blur_params(
    cell_id: str,
    cell_z:  float,
    velocity: float,
    viewport_w: int = 1200,
    viewport_h: int = 900,
    quality: int = _MB_QUALITY,
) -> dict:
    """
    Compute per-cell motion blur SVG filter parameters.

    Mirrors the MotionBlur CS dispatch that, for each tile, computes the
    blur kernel direction and magnitude from the velocity buffer.

    Algorithm:
      1. Velocity → tile max (already in VelocityFlattenData)
      2. Tile max → scatter required? (VelocityMaxInTiles > TileDistanceMaxGathered)
      3. Blur radius = velocity × MB_AMOUNT × quality_scale
      4. Blur direction: z-axis maps to diagonal (45° shadow direction in 2-D)

    Returns {blur_radius, direction_angle_deg, use_scatter, quality_scale}.

    鲁迅式：运动模糊的方向是时间的方向——
    向前模糊，向前运动；向后模糊，向后运动。
    方向错了，时间便倒流了。
    """
    if not is_motion_blur_enabled(quality):
        return {"blur_radius": 0.0, "direction_angle_deg": 0.0,
                "use_scatter": False, "quality_scale": 0.0}

    # Quality scale (mirrors EMotionBlurQuality: 0=low→0.25, 3=cinematic→1.0)
    quality_scales = [0.25, 0.5, 0.75, 1.0]
    quality_scale  = quality_scales[min(quality - 1, 3)] if quality >= 1 else 0.0

    # Velocity in tiles (mirrors VelocityMaxInTiles computation)
    max_vel_pixels = _MB_MAX_VEL_FRACTION * viewport_w
    vel_in_tiles   = abs(velocity) * max_vel_pixels * 0.5 / _MB_FLATTEN_TILE_SIZE

    # Scatter vs gather decision
    use_scatter = vel_in_tiles > _MB_SCATTER_THRESHOLD

    # Blur radius in pixels
    blur_radius = abs(velocity) * max_vel_pixels * _MB_AMOUNT * quality_scale
    if _MB_HALF_RES_INPUT and not use_scatter:
        blur_radius *= 0.5   # half-res gather = halved effective radius

    # Separable second pass (r.MotionBlurSeparable = adds second orthogonal pass)
    if _MB_SEPARABLE:
        blur_radius *= _MB_SECOND_SCALE

    # Direction: z-layer velocity maps to a temporal diagonal (135° = up-right to down-left)
    # Positive dz → cell moved away → blur toward bottom-right
    direction = 135.0 if velocity > 0 else 315.0

    return {
        "blur_radius":        round(blur_radius, 2),
        "direction_angle_deg": direction,
        "use_scatter":         use_scatter,
        "quality_scale":       quality_scale,
    }





def run_motion_blur_pass(
    cell_registry:      dict,
    cell_registry_prev: dict,
    viewport_w:         int = 1200,
    viewport_h:         int = 900,
) -> dict[str, dict]:
    """
    Full motion blur pipeline for all cells.

    Mirrors the three-pass MotionBlur dispatch:
      1. VelocityFlatten  → compute per-cell velocities
      2. TileMaxVelocity  → compute per-tile max velocities
      3. MotionBlur CS    → compute per-cell blur params

    Returns dict: cell_id → motion blur parameter dict.

    鲁迅式：运动模糊的流水线是时间的考古学——
    通过比较现在与过去，重建运动的证据，再将其涂抹在画面上，
    告诉观看者：这里曾经有运动，虽然现在已经静止了。
    """
    velocity_data = AstroCellVelocityFlattenData()
    velocity_data.build(cell_registry, cell_registry_prev, viewport_w, viewport_h)

    cells  = cell_registry.get("cells", {})
    result: dict[str, dict] = {}

    for cell_id, entry in cells.items():
        z   = float(entry.get("z", 3))
        vel = velocity_data.cell_velocities.get(cell_id, 0.0)
        result[cell_id] = compute_motion_blur_params(
            cell_id, z, vel, viewport_w, viewport_h
        )

    blurred = sum(1 for v in result.values() if v["blur_radius"] > 0.1)
    print(
        f"[ASTRO-MB] run_motion_blur_pass: "
        f"total={len(result)} blurred={blurred} "
        f"global_max_vel={velocity_data.global_max_velocity:.3f}",
        file=sys.stderr,
    )
    return result


# =============================================================================
# [ASTRO-CELL] TemporalAA → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/TemporalAA.cpp
#
# 鲁迅曾言：「我不想再说废话了。废话说了半天没有人听。
# 历史上，有些话说完之后，沉默了几十年，然后成为了真理。」
# Temporal AA 亦然——每一帧积累一点，沉默几十帧之后，
# 锯齿消失，真理（无锯齿的画面）显现。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FTemporalAA shader class           → AstroCellTAAPass
#     FAlphaChannelDim (permutation)   → alpha_channel_enabled (bool)
#     FTAAPassConfigDim                → taa_pass_config (int)
#     FTAAQualityDim                   → taa_quality (ETAAQuality)
#     FTAAScreenPercentageDim          → screen_percentage_range (int)
#   CVarTemporalAAFilterSize           → _TAA_FILTER_SIZE
#   CVarTemporalAACatmullRom           → _TAA_CATMULL_ROM
#   CVarTemporalAAPauseCorrect         → _TAA_PAUSE_CORRECT
#   CVarTemporalAACurrentFrameWeight   → _TAA_CURRENT_FRAME_WEIGHT (0.04)
#   CVarTemporalAAQuality              → _TAA_QUALITY (0–3)
#   CVarTAAR11G11B10History            → _TAA_R11G11B10_HISTORY
#   DoesPlatformSupportTemporalHistoryUpscale → True (all platforms supported)
#   AddTemporalAAPass()                → run_temporal_aa_pass()
#   FTemporalAAHistory                 → AstroCellTAAHistory
#
# Algorithm changes (鲁迅式 20%):
#   1. Sub-pixel jitter (Halton) → same Halton sequence from path tracer
#   2. History reprojection (screen-pos UV lookup) → per-cell z-layer history dict
#   3. Neighbourhood clamping (AABB in colour space) → luminance range clamp
#   4. Current frame weight blend → exponential moving average
#   5. Anti-ghosting (mobility-based) → large z-delta triggers history rejection
# =============================================================================

# ── TemporalAA CVars ──────────────────────────────────────────────────────────
_TAA_FILTER_SIZE:         float = 1.0    # r.TemporalAAFilterSize (1=smooth, 0=sharp)
_TAA_CATMULL_ROM:         bool  = False  # r.TemporalAACatmullRom
_TAA_PAUSE_CORRECT:       bool  = True   # r.TemporalAAPauseCorrect
_TAA_CURRENT_FRAME_WEIGHT:float = 0.04   # r.TemporalAACurrentFrameWeight
_TAA_QUALITY:             int   = 2      # r.TemporalAA.Quality [0,3]
_TAA_SCREEN_PERCENTAGE:   float = 100.0  # r.TemporalAA.HistoryScreenPercentage
_TAA_R11G11B10:           bool  = True   # r.TemporalAA.R11G11B10History
_TAA_UPSCALER:            int   = 1      # r.TemporalAA.Upscaler
_TAA_TILE_SIZE_X:         int   = 8      # GTemporalAATileSizeX
_TAA_TILE_SIZE_Y:         int   = 8      # GTemporalAATileSizeY
_TAA_LARGE_GROUP:         bool  = False  # r.TemporalAA.LargeGroup
_TAA_LARGE_GROUP_MAX_INPUT_SCALE: int = 90  # r.TemporalAA.LargeGroup.MaxInputScale
_TAA_VGPR_OPT:            bool  = False  # r.TemporalAA.VGPROpt
_TAA_GHOST_THRESHOLD:     float = 1.0    # z-delta above which history is rejected

# TAA quality → neighbourhood sample count (mirrors ETAAQuality)
_TAA_QUALITY_SAMPLE_COUNTS = {0: 5, 1: 9, 2: 9, 3: 9}





def run_temporal_aa_pass(
    cell_registry: dict,
    epoch:         int = 0,
    quality:       int = _TAA_QUALITY,
    filter_size:   float = _TAA_FILTER_SIZE,
) -> dict[str, tuple]:
    """
    Top-level TAA entry point — mirrors AddTemporalAAPass().

    Constructs AstroCellTAAPass, runs it, and returns per-cell output colours.
    These colours can be used to modulate cell SVG fill for temporal stability.

    鲁迅式：抗锯齿的最终目的不是消灭抖动，而是接受时间——
    时间流逝，画面平滑，这才是进步的代价。
    """
    taa = AstroCellTAAPass(quality=quality, filter_size=filter_size)
    return taa.run(cell_registry, epoch)


# =============================================================================
# § DecalRenderingCommon — 贴花混合描述符与渲染阶段推导
#   移植自 Renderer-Private/DecalRenderingCommon.cpp
#   鲁迅式：贴花不是装饰，而是承认世界本身已不够干净。
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, auto
from typing import Optional





class AstroCellLightShaftBloom:
    """
    Python equivalent of the bloom light shaft technique (ELightShaftTechnique::Bloom).

    Implements the radial blur bloom pass from LightShaftRendering.cpp:
      1. Downsample the «scene» (cell SVG) to _LS_DOWNSAMPLE factor
      2. Apply _LS_BLUR_PASSES successive radial blurs toward the light source
      3. Composite the bloom result back onto the cell

    In SVG, passes 1-2 are approximated by:
      - feGaussianBlur with increasing stdDeviation per pass
      - feComposite with «screen» blend mode (additive bloom)
      - The final blur is driven by the first-pass distance and num_samples

    鲁迅式：光晕是光源炫耀自身的方式——一层一层地散开，
    每一遍模糊都在宣告：「我在这里，我很亮，我值得被注意。」
    """

    def __init__(self,
                 params:    dict,
                 cell_id:   str,
                 bbox:      dict,
                 species:   str) -> None:
        self._p       = params
        self._cell_id = cell_id
        self._bbox    = bbox
        self._species = species

    def emit_svg(self) -> str:
        """
        Emit a multi-pass radial bloom filter + compositor as an SVG <filter>.

        The number of feGaussianBlur primitives equals min(_LS_BLUR_PASSES, 3)
        to keep the SVG compact; each pass uses an increasing stdDeviation to
        simulate the distance-proportional blur growth from the C++ shader.

        Returns SVG string with filter definition and application hint comment.

        鲁迅式：多次模糊是耐心的象征——每一遍模糊都比上一遍更弥散，
        直到光柱从细线变成光晕，从光晕变成弥漫的辉光。
        """
        if not should_render_light_shafts(self._species):
            return ""

        p         = self._p
        cell_id   = self._cell_id
        bbox      = self._bbox
        cell_w    = float(bbox.get("w", 100))
        cell_h    = float(bbox.get("h", 50))

        # Bloom filter identifier
        filter_id = f"ls-bloom-{cell_id}"
        bloom_r, bloom_g, bloom_b = p["bloom_tint"]

        # Per-pass stdDeviation: distance doubles each pass (GLightShaftFirstPassDistance growth)
        # Mirrors: BlurOrigin distance × FirstPassDistance × 2^pass
        base_sigma = max(1.0, min(cell_w, cell_h) * _LS_FIRST_PASS_DIST * 0.15)
        sigmas     = [base_sigma * (2 ** i) for i in range(min(_LS_BLUR_PASSES, 3))]

        parts = [
            f'<!-- [ASTRO-LS] LightShaftRendering.cpp Bloom port '
            f'blur_origin=({p["blur_origin_x"]:.3f},{p["blur_origin_y"]:.3f}) '
            f'bloom_scale={p["bloom_scale"]:.2f} passes={_LS_BLUR_PASSES} -->',
            f'<defs>',
            f'  <filter id="{filter_id}" '
            f'x="-30%" y="-30%" width="160%" height="160%">',
        ]

        prev_result = "SourceGraphic"
        for i, sigma in enumerate(sigmas):
            result_name = f"blur{i}"
            parts.append(
                f'    <!-- Pass {i+1}: radial blur σ={sigma:.2f} -->'
            )
            parts.append(
                f'    <feGaussianBlur in="{prev_result}" '
                f'stdDeviation="{sigma:.2f}" result="{result_name}"/>'
            )
            prev_result = result_name

        # Tint the final bloom layer with BloomTint colour
        parts.append(
            f'    <feColorMatrix in="{prev_result}" type="matrix" '
            f'values="{bloom_r:.2f} 0 0 0 0  '
            f'0 {bloom_g:.2f} 0 0 0  '
            f'0 0 {bloom_b:.2f} 0 0  '
            f'0 0 0 {p["bloom_scale"]:.2f} 0" result="tinted"/>'
        )

        # Composite: screen blend (additive bloom over original)
        parts.append(
            f'    <feBlend in="SourceGraphic" in2="tinted" '
            f'mode="screen" result="bloomed"/>'
        )

        parts.append(f'  </filter>')
        parts.append(f'</defs>')
        parts.append(
            f'<!-- [ASTRO-LS] bloom filter attached to cell-{cell_id}: '
            f'apply filter="url(#{filter_id})" to the cell group -->'
        )

        return "\n".join(parts)





def _builtin_nne_denoiser_v2(
    radiance_buf: dict, albedo_buf: dict, normal_buf: dict,
) -> dict:
    """
    Built-in NNE spatial denoiser stub — edge-stop bilateral blur.
    Registered as default IPathTracingDenoiser in AstroCellRendererModule.startup().

    鲁迅式：默认降噪器是「够用就好」的代表——
    不是最好的，但总是在场，总是有效。
    """
    denoised: dict = {}
    for cid, rad in radiance_buf.items():
        alb      = albedo_buf.get(cid, (0.5, 0.5, 0.5))
        alb_lum  = (alb[0] + alb[1] + alb[2]) / 3.0
        preserve = max(0.3, min(1.0, alb_lum * 1.5))
        grey     = (rad[0] + rad[1] + rad[2]) / 3.0
        denoised[cid] = (
            rad[0] * preserve + grey * (1.0 - preserve),
            rad[1] * preserve + grey * (1.0 - preserve),
            rad[2] * preserve + grey * (1.0 - preserve),
        )
    return denoised


