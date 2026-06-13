import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def get_or_create_cell_path_tracing_state(
        view_id: str = "default") -> AstroCellPathTracingState:
    """
    Return the AstroCellPathTracingState for *view_id*, creating it on first call.
    Mirrors GetPathTracingStateFromView() from PathTracing.cpp.

    鲁迅式：第一次访问时才创建——懒汉式，节省内存，亦是对「不必要存在」的抵抗。
    """
    global _CELL_PATH_TRACING_STATES
    if view_id not in _CELL_PATH_TRACING_STATES:
        _CELL_PATH_TRACING_STATES[view_id] = AstroCellPathTracingState()
    return _CELL_PATH_TRACING_STATES[view_id]





def prepare_path_tracing(
    config: AstroCellPathTracingConfig | None = None,
    view_id: str = "default",
) -> AstroCellPathTracingState:
    """
    Check for configuration changes and invalidate if needed.
    Mirrors PreparePathTracing() + the IsDifferent/Invalidate block in
    FDeferredShadingSceneRenderer::RenderPathTracing().

    Called once per epoch before render_path_tracing() dispatches.

    鲁迅式：准备是清醒，清醒是有时候比勇气更难做到的事情。
    """
    state  = get_or_create_cell_path_tracing_state(view_id)
    cfg    = config or AstroCellPathTracingConfig()

    if cfg.is_different(state.last_config):
        print(
            f"[ASTRO-PT] PreparePathTracing — config changed, invalidating state "
            f"(sample_index was {state.sample_index})",
            file=sys.stderr,
        )
        state.last_config = cfg
        state.invalidate()
    else:
        # No change — bump frame_index only (mirrors FrameIndex++ in C++ each frame)
        state.frame_index += 1

    return state





def render_path_tracing(
    cell_id: str,
    bbox: dict,
    species: str,
    all_bboxes: dict | None = None,
    bvh: "AstroCellBVH | None" = None,
    config: AstroCellPathTracingConfig | None = None,
    view_id: str = "default",
) -> dict:
    """
    Accumulate one path tracing sample for *cell_id*.

    Entry point mirroring the per-view dispatch in RenderPathTracing() →
    the per-pixel sample loop inside PathTracing.usf.

    Algorithm:
      1. prepare_path_tracing() — guard invalidation on config changes
      2. Adaptive sampling gate — skip converged cells (VarianceBuffer check)
      3. Sample one path via _pt_sample_cell_radiance()
      4. Gaussian-filter the sample weight
      5. Running-average accumulate into radiance_buffer
      6. Update variance_buffer (Welford online variance)
      7. Update denoiser AOV buffers (albedo, normal proxy, depth)
      8. Increment sample_index when all cells sampled

    Returns dict with per-cell accumulated radiance + denoiser AOV data.

    鲁迅式：渲染是积累，积累是耐心，耐心是这个时代最稀缺的品质。
    每一帧调用一次，样本慢慢增多，噪声慢慢消退——
    如同鲁迅一篇篇写下去，终究成了一部真实的中国。
    """
    if not _PT_ENABLED:
        return {"cell_id": cell_id, "pt_enabled": False}

    cfg   = config or AstroCellPathTracingConfig()
    state = prepare_path_tracing(cfg, view_id)

    # ── Adaptive sampling convergence gate ────────────────────────────────
    if state.is_converged(cell_id, cfg.adaptive_threshold):
        return {
            "cell_id":      cell_id,
            "sample_index": state.sample_index,
            "converged":    True,
            "radiance":     state.radiance_buffer.get(cell_id, (0.0, 0.0, 0.0)),
            "variance":     state.variance_buffer.get(cell_id, 0.0),
        }

    # ── Sample one path ────────────────────────────────────────────────────
    raw = _pt_sample_cell_radiance(
        cell_id     = cell_id,
        bbox        = bbox,
        species     = species,
        sample_idx  = state.sample_index,
        frame_idx   = state.frame_index,
        bvh         = bvh,
        all_bboxes  = all_bboxes,
        mis_mode    = cfg.mis_mode,
        max_bounces = cfg.max_bounces,
    )

    # ── Gaussian reconstruction filter ────────────────────────────────────
    filtered = _pt_gaussian_filter(raw, cfg.filter_sigma)

    # ── Running-average accumulation (mirrors RadianceRT += sample / N) ───
    n   = state.sample_index + 1
    old = state.radiance_buffer.get(cell_id, (0.0, 0.0, 0.0))
    new_r = old[0] + (filtered[0] - old[0]) / n
    new_g = old[1] + (filtered[1] - old[1]) / n
    new_b = old[2] + (filtered[2] - old[2]) / n
    state.radiance_buffer[cell_id] = (new_r, new_g, new_b)

    # ── Welford online variance (mirrors VarianceBuffer update) ───────────
    # δ = sample − old_mean;  δ2 = sample − new_mean
    # M2 += δ × δ2;  variance = M2 / (n-1) for n≥2
    old_var = state.variance_buffer.get(cell_id, 0.0)
    lum_old = (old[0] + old[1] + old[2]) / 3.0
    lum_new = (new_r + new_g + new_b) / 3.0
    lum_sample = (filtered[0] + filtered[1] + filtered[2]) / 3.0
    delta   = lum_sample - lum_old
    delta2  = lum_sample - lum_new
    # Running M2 stored as variance × (n-1) scaled back
    m2_prev = old_var * max(n - 2, 1)
    m2_new  = m2_prev + delta * delta2
    state.variance_buffer[cell_id] = m2_new / max(n - 1, 1)

    # ── Denoiser AOV update ────────────────────────────────────────────────
    # AlbedoRT: species base colour (mirrors material albedo AOV)
    _ALBEDO_MAP = {
        "cil-eye": (0.49, 0.51, 0.71), "cil-bolt": (1.0, 0.44, 0.0),
        "cil-vector": (0.18, 0.49, 0.20), "cil-plus": (0.12, 0.53, 0.90),
        "cil-arrow-right": (0.27, 0.35, 0.39), "cil-filter": (0.48, 0.12, 0.64),
        "cil-code": (0.18, 0.49, 0.20), "cil-layers": (0.08, 0.40, 0.75),
        "cil-loop": (0.96, 0.50, 0.09), "cil-graph": (0.21, 0.28, 0.31),
    }
    state.albedo_buffer[cell_id] = _ALBEDO_MAP.get(species, (0.5, 0.5, 0.5))
    # NormalRT: upward-facing normal (all cells face viewer → (0, 0, 1))
    state.normal_buffer[cell_id] = (0.0, 0.0, 1.0)
    # DepthRT: normalised depth from z-layer (mirrors DepthRT = z / z_far)
    z_far = 8.0
    state.depth_buffer[cell_id]  = max(0.0, min(1.0,
        float(bbox.get("z", 3)) / z_far))

    # ── Increment sample counter after all cells complete one sample ───────
    # In the C++ renderer SampleIndex is incremented once per frame after the
    # full tile dispatch.  Here we increment per-cell call (single-threaded).
    state.sample_index = n

    result = {
        "cell_id":      cell_id,
        "sample_index": state.sample_index,
        "frame_index":  state.frame_index,
        "converged":    state.is_converged(cell_id, cfg.adaptive_threshold),
        "radiance":     state.radiance_buffer[cell_id],
        "variance":     state.variance_buffer.get(cell_id, 0.0),
        "albedo":       state.albedo_buffer[cell_id],
        "normal":       state.normal_buffer[cell_id],
        "depth":        state.depth_buffer[cell_id],
        "pt_enabled":   True,
    }

    dbg = os.environ.get("ASTRO_PT_VERBOSE", "0") == "1"
    if dbg:
        print(
            f"[ASTRO-PT] render_path_tracing cell={cell_id} "
            f"spp={state.sample_index}/{cfg.max_samples} "
            f"radiance=({new_r:.3f},{new_g:.3f},{new_b:.3f}) "
            f"var={result['variance']:.5f} "
            f"converged={result['converged']}",
            file=sys.stderr,
        )

    return result


# =============================================================================
# [ASTRO-CELL] PathTracingSpatialTemporalDenoising → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PathTracingSpatialTemporalDenoising.cpp
#
# 鲁迅曾言：「不读书的人，思想就会停止。」
# 不去噪的渲染器，噪声就会永远停止不了。
# 降噪是文明的努力——用空间和时间的信息，重建真实的光照。
#
# FDenoiserManager → AstroCellDenoiserManager
#   RegisterSpatialDenoiser          → register_spatial_denoiser()
#   RegisterSpatialTemporalDenoiser  → register_spatial_temporal_denoiser()
#   UnregisterDenoiser               → unregister_denoiser()
#   HasSpatialDenoiser               → has_spatial_denoiser()
#   HasSpatialTemporalDenoiser       → has_spatial_temporal_denoiser()
#   GetSpatialDenoiser               → get_spatial_denoiser()
#   GetSpatialTemporalDenoiser       → get_spatial_temporal_denoiser()
#   bNeedTextureCreateExtraFlags     → need_extra_flags (bool)
#
# Key denoising passes (ported as pure-Python analytic approximations):
#   FTemporalReprojectionAlignCS    → temporal_reprojection_align()
#   FTemporalReprojectionBlurCS     → temporal_reprojection_blur()
#   FTemporalReprojectionMergeCS    → temporal_reprojection_merge()
#   FTemporalHighFrequencyRejectMapCS → high_frequency_reject_map()
#   FTemporalFeatureFusionCS        → temporal_feature_fusion()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU compute shaders → analytic Python per-cell operations
#   2. MotionVector texture → per-cell z-layer delta (2-D displacement)
#   3. Variance-weighted temporal blend: history_weight adapted per-cell
#      from the variance buffer of AstroCellPathTracingState
#   4. Spatial NLM kernel (3×3 BVH neighbour query) replaces the full
#      NxN screen-space bilateral filter pass
# =============================================================================

# CVarPathTracingDenoiser equivalents
_PTD_ENABLED:              int   = 1     # r.PathTracing.Denoiser (-1/0/1)
_PTD_SPATIAL_ENABLED:      int   = 1     # r.PathTracing.SpatialDenoiser
_PTD_NORMAL_SPACE:         int   = 0     # 0=world, 1=camera
_PTD_VARIANCE_TYPE:        int   = 1     # 1=combined single-channel
_PTD_RANKED_LUM_VAR:       int   = 0     # 0=default luminance variance
_PTD_TEMPORAL_WEIGHT:      float = 0.9   # history blend weight (temporal stability)
_PTD_SPATIAL_TYPE:         int   = 0     # 0=spatial-only plugin, 1=spatio-temporal
_PTD_DENOISER_NAME:        str   = "NNEDenoiser"
_PTD_TEMPORAL_NAME:        str   = "NFOR"





def build_hair_strands_view_params(
    cell_registry: dict,
    viewport_w:    int = 1200,
    viewport_h:    int = 900,
) -> AstroCellHairStrandsViewParams:
    """
    Build per-epoch HairStrands view parameters from the cell registry.

    Mirrors InternalCreateHairStrandsViewUniformBuffer() — constructs the
    FHairStrandsViewUniformParameters struct (or its dummy fallback) from
    the visibility data published by the pre-pass.

    Called by render_hair_pre_pass() before any cell's hair-specific SVG
    elements are generated.

    鲁迅式：统一缓冲区是每个着色器的共同语言——
    没有它，每个发丝都在孤独地猜测世界的状态。
    """
    params = AstroCellHairStrandsViewParams()
    params.sample_viewport_resolution = (viewport_w, viewport_h)
    params.max_sample_count           = _HS_MAX_SAMPLE_PER_PIXEL
    params.dual_scatter_roughness     = _HS_DUAL_SCATTER_ROUGHNESS

    cells = cell_registry.get("cells", {})
    if not cells:
        # Dummy fallback — mirrors the «else» branch in the C++ function
        # that fills default textures when no hair visibility data exists.
        params.tile_count_xy = (0, 0)
        params.tile_valid    = False
        return params

    # ── Build coverage + depth maps ────────────────────────────────────────
    # Group cells by z-layer first (z-layer peers are the «macro group»)
    z_layer_groups: dict[int, list[str]] = {}
    for cell_id, entry in cells.items():
        z = entry.get("z", 3)
        z_layer_groups.setdefault(z, []).append(cell_id)

    all_depths: list[float] = []
    for cell_id, entry in cells.items():
        bbox_data = entry.get("bbox", {})
        if "min" in bbox_data and "max" in bbox_data:
            mn, mx = bbox_data["min"], bbox_data["max"]
            bbox = {"x": mn[0], "y": mn[1], "w": mx[0]-mn[0],
                    "h": mx[1]-mn[1], "z": mn[2] if len(mn)>2 else 0,
                    "species": entry.get("species", "")}
        else:
            bbox = dict(bbox_data)
            bbox["species"] = entry.get("species", "")

        z_layer = entry.get("z", 3)
        peers   = z_layer_groups.get(z_layer, [])

        coverage = _hs_compute_coverage(cell_id, bbox, peers)
        params.coverage_map[cell_id] = coverage

        # Depth: normalised z-layer position in [0,1]
        depth = min(1.0, z_layer / 8.0)
        params.hair_depth_map[cell_id] = depth
        all_depths.append(depth)

    # ── Build HZB from depth values ────────────────────────────────────────
    if all_depths:
        params.hzb_parameters, params.hzb_mips = _hs_build_hzb(all_depths)
        params.tile_valid = True

    # ── Tile count: viewport / tile_size ──────────────────────────────────
    tiles_x = (viewport_w + _HS_TILE_SIZE - 1) // _HS_TILE_SIZE
    tiles_y = (viewport_h + _HS_TILE_SIZE - 1) // _HS_TILE_SIZE
    params.tile_count_xy = (tiles_x, tiles_y)

    # ── Species macro groups ───────────────────────────────────────────────
    for cell_id, entry in cells.items():
        sp = entry.get("species", "unknown")
        params.macro_groups.setdefault(sp, []).append(cell_id)

    print(
        f"[ASTRO-HS] build_hair_strands_view_params: "
        f"cells={len(cells)} tile_count={params.tile_count_xy} "
        f"hzb_mips={len(params.hzb_mips)} "
        f"macro_groups={len(params.macro_groups)} "
        f"tile_valid={params.tile_valid}",
        file=sys.stderr,
    )
    return params





class AstroCellSubstrateUniforms:
    """
    Python equivalent of FSubstrateGlobalUniformParameters.

    Holds per-frame Substrate material system state: closure counts,
    tile type masks, anisotropy flags, and buffer layout parameters.

    鲁迅式：Substrate 的 Uniform 是材质系统的宪法——
    每帧一次，全体着色器必须遵守。
    """

    __slots__ = (
        "max_closure_per_pixel",
        "uses_tile_type_mask",
        "uses_anisotropy",
        "closures_per_pixel",
        "effective_max_closure",
        "stochastic_lighting_active",
        "roughness_tracking_enabled",
        "tile_coord_8bit",
    )

    def __init__(self) -> None:
        self.max_closure_per_pixel:      int  = 1
        self.uses_tile_type_mask:        int  = 0
        self.uses_anisotropy:            bool = False
        self.closures_per_pixel:         int  = _SUB_CLOSURES_PER_PIXEL
        self.effective_max_closure:      int  = 1
        self.stochastic_lighting_active: bool = _SUB_STOCHASTIC_LIGHTING_ACTIVE
        self.roughness_tracking_enabled: bool = _SUB_ROUGHNESS_TRACKING
        self.tile_coord_8bit:            bool = _SUB_TILE_COORD_8BIT


# Per-species closure complexity table (mirrors per-material ClosureCount from C++).
# Species with multiple BSDF lobes (e.g. eye has specular + diffuse + SSS)
# have higher closure counts.
_SPECIES_CLOSURE_COUNT: dict[str, int] = {
    "cil-eye":         3,   # diffuse + specular + SSS (eyelens)
    "cil-bolt":        2,   # diffuse + emissive
    "cil-vector":      2,   # diffuse + directional specular
    "cil-plus":        1,   # simple diffuse
    "cil-arrow-right": 1,   # simple diffuse
    "cil-filter":      2,   # diffuse + anisotropic specular (grid lines)
    "cil-code":        1,   # monochrome diffuse
    "cil-layers":      3,   # 3 layers × 1 closure each
    "cil-loop":        2,   # diffuse + emissive rim
    "cil-graph":       2,   # diffuse + edge specular
}

# Species anisotropy flag (mirrors bUsesAnisotropy in FSubstrateViewData)
_SPECIES_ANISOTROPIC: set[str] = {"cil-filter", "cil-vector"}





def get_substrate_max_closure_count(species_list: list[str]) -> int:
    """
    Compute effective max closure count for the given species set.

    Mirrors GetSubstrateMaxClosureCount(FViewInfo&):
      if UseClosureCountFromMaterial:
          max_closure = max over visible materials of ClosureCountFromMaterial
      else:
          max_closure = ClosuresPerPixel (CVar)
    Clamped to [1, SUBSTRATE_MAX_CLOSURE_COUNT].

    鲁迅式：最大闭包数是资源分配的上限——
    不让任何一种材质独吞全部 GBuffer 预算，也不让简单材质浪费内存。
    """
    if not species_list:
        return 1
    if _SUB_USE_CLOSURE_COUNT_FROM_MAT:
        raw = max(_SPECIES_CLOSURE_COUNT.get(sp, 1) for sp in species_list)
    else:
        raw = _SUB_CLOSURES_PER_PIXEL
    return max(1, min(raw, _SUB_MAX_CLOSURE_COUNT))





def build_substrate_uniforms(
    cell_registry: dict,
    visible_cell_ids: list[str] | None = None,
) -> AstroCellSubstrateUniforms:
    """
    Build per-epoch Substrate global uniform parameters.

    Mirrors the Substrate global UB setup that occurs in
    FDeferredShadingSceneRenderer::Render() before material passes:
      1. Collect species for visible cells.
      2. Compute max closure count from material data.
      3. Classify tile type mask from species complexity.
      4. Detect anisotropy.
      5. Pack into FSubstrateGlobalUniformParameters.

    鲁迅式：全局参数是帧的脸面——在这一帧里，
    所有材质都必须在这套规则下生存。
    """
    uni = AstroCellSubstrateUniforms()
    cells = cell_registry.get("cells", {})

    # Which cells to process (all if visibility list not given)
    target_ids = visible_cell_ids if visible_cell_ids else list(cells.keys())
    species_list = [cells[cid].get("species", "") for cid in target_ids if cid in cells]

    # Max closure count
    uni.max_closure_per_pixel  = get_substrate_max_closure_count(species_list)
    uni.effective_max_closure  = uni.max_closure_per_pixel
    uni.closures_per_pixel     = _SUB_CLOSURES_PER_PIXEL

    # Tile type mask: OR together tile types present
    tile_mask = 0
    for sp in species_list:
        cc = _SPECIES_CLOSURE_COUNT.get(sp, 1)
        if cc == 1:
            tile_mask |= _SUB_TILE_SIMPLE
        elif cc == 2:
            tile_mask |= _SUB_TILE_SINGLE_CLOSURE
        else:
            tile_mask |= _SUB_TILE_COMPLEX
        if sp in ("cil-eye", "cil-graph", "cil-vector"):
            tile_mask |= _SUB_TILE_HAIR   # hair-like fine detail
    uni.uses_tile_type_mask = tile_mask

    # Anisotropy: any anisotropic species in view?
    uni.uses_anisotropy = any(sp in _SPECIES_ANISOTROPIC for sp in species_list)

    # Stochastic + feature flags
    uni.stochastic_lighting_active = is_stochastic_lighting_active()
    uni.roughness_tracking_enabled = _SUB_ROUGHNESS_TRACKING
    uni.tile_coord_8bit            = _SUB_TILE_COORD_8BIT

    print(
        f"[ASTRO-SUB] build_substrate_uniforms: "
        f"species_count={len(set(species_list))} "
        f"max_closure={uni.max_closure_per_pixel} "
        f"tile_mask=0b{uni.uses_tile_type_mask:08b} "
        f"anisotropy={uni.uses_anisotropy} "
        f"stochastic={uni.stochastic_lighting_active}",
        file=sys.stderr,
    )
    return uni





def substrate_view_data_reset(prev_uniforms: AstroCellSubstrateUniforms) -> AstroCellSubstrateUniforms:
    """
    Reset per-view Substrate data between epochs, preserving tile type mask.

    Mirrors FSubstrateViewData::Reset():
        Preserves UsesTileTypeMask and bUsesAnisotropy across reset
        (they represent accumulated scene complexity and are only updated,
        never reverted to 0, until the scene is fully re-classified).

    鲁迅式：重置是新生，但有些东西不能忘记——
    上一帧的复杂度标记，是对下一帧分配策略的提示。
    """
    new_uni = AstroCellSubstrateUniforms()
    # Carry forward tile mask and anisotropy (mirrors C++ preservation)
    new_uni.uses_tile_type_mask = prev_uniforms.uses_tile_type_mask
    new_uni.uses_anisotropy     = prev_uniforms.uses_anisotropy
    return new_uni


# =============================================================================
# [ASTRO-CELL] DiaphragmDOF (Depth of Field) → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/DiaphragmDOF.cpp
#
# 鲁迅曾言：「不在沉默中爆发，就在沉默中灭亡。」
# 景深亦然——前景的模糊是对远处真相的沉默，
# 而焦点处的清晰是最后的爆发。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   DiaphragmDOF::IsSupported()        → is_dof_supported()
#   CVarDOFGatherResDivisor            → _DOF_GATHER_RES_DIVISOR
#   CVarAccumulatorQuality             → _DOF_ACCUMULATOR_QUALITY
#   CVarRingCount                      → _DOF_RING_COUNT
#   CVarRecombineQuality               → _DOF_RECOMBINE_QUALITY
#   CVarMinimalFullresBlurRadius       → _DOF_MIN_FULLRES_BLUR_RADIUS
#   CVarScatterMaxSpriteRatio          → _DOF_SCATTER_MAX_SPRITE_RATIO
#   CVarScatterMinCocRadius            → _DOF_SCATTER_MIN_COC_RADIUS
#   FDiaphragmDOFPass (gather pass)    → AstroCellDOFGatherPass
#   ComputeCircleOfConfusionRadius     → compute_coc_radius()
#   GatherBokeh (ring accumulation)    → gather_bokeh_rings()
#   ScatterBokeh (sprite pass)         → scatter_bokeh_sprites()
#   RecombinePass (full-res merge)     → dof_recombine()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU gather at half-resolution → analytic per-cell CoC radius from z-depth
#   2. Ring kernel (N=5 rings, 78 samples) → analytic annulus area formula
#   3. Scatter sprite pass (GPU instanced quads) → per-cell blurred bbox rect
#   4. Foreground/background separation → z-layer threshold split
#   5. Bokeh shape (hexagon/octagon) → circular approximation (area-preserving)
# =============================================================================

# ── DiaphragmDOF CVars ────────────────────────────────────────────────────────
_DOF_GATHER_RES_DIVISOR:      int   = 2      # r.DOF.Gather.ResolutionDivisor
_DOF_ACCUMULATOR_QUALITY:     int   = 1      # r.DOF.Gather.AccumulatorQuality
_DOF_RING_COUNT:              int   = 5      # r.DOF.Gather.RingCount [3,5]
_DOF_RECOMBINE_QUALITY:       int   = 2      # r.DOF.Recombine.Quality
_DOF_MIN_FULLRES_BLUR_RADIUS: float = 0.1    # r.DOF.Recombine.MinFullresBlurRadius
_DOF_SCATTER_MIN_COC:         float = 3.0    # r.DOF.Scatter.MinCocRadius
_DOF_SCATTER_MAX_SPRITE:      float = 0.1    # r.DOF.Scatter.MaxSpriteRatio
_DOF_TAA_QUALITY:             int   = 1      # r.DOF.TemporalAAQuality
_DOF_PREFER_LOWER_BIT:        bool  = False  # r.DOF.PreferLowerBitDepth
_DOF_COC_BILATERAL_STRENGTH:  float = 0.0    # r.DOF.TAA.CoCBilateralFilterStrength

# Focal plane parameters (app-level DOF config, not a CVar in UE5)
_DOF_FOCAL_Z_LAYER:   float = 3.0   # z-layer of sharp focus
_DOF_NEAR_TRANSITION: float = 1.5   # z-layer range for near DOF
_DOF_FAR_TRANSITION:  float = 2.0   # z-layer range for far DOF
_DOF_MAX_COC_PIXELS:  float = 24.0  # maximum CoC radius in pixels


