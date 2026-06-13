import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



@_ptdc
class AstroCellReflectionSceneData:
    """
    Python equivalent of FReflectionSceneData.

    Holds the global cubemap array and the per-probe state registry.
    Mirrors the scene-level structure that the C++ renderer consults
    when resolving cubemap slots and building the Uniform Buffer.

    鲁迅式：场景数据是公共档案馆——每个探针在这里登记户籍，
    离开时注销，场景清除时关门大吉。
    """
    # Allocated captures: cell_id → AstroCellCaptureState
    allocated_captures: _PTDict[str, AstroCellCaptureState] = _ptfield(
        default_factory=dict)
    # Next available cubemap slot index (mirrors NextAvailableReflectionCaptureSortedIndex)
    next_cubemap_slot: int = 0
    # Maximum supported captures per scene (mirrors MaxCubemaps)
    max_cubemaps: int = 64


# Module-level scene data singleton
_ASTRO_REFLECTION_SCENE: AstroCellReflectionSceneData = AstroCellReflectionSceneData()





def get_reflection_scene_data() -> AstroCellReflectionSceneData:
    """Return the module-level AstroCellReflectionSceneData singleton."""
    return _ASTRO_REFLECTION_SCENE





# ─────────────────────────────────────────────────────────────────────────────
# Port: Lumen Reflection Tracing  (LumenReflectionTracing.cpp)
# Port: Lumen Reflection HWRT     (LumenReflectionHardwareRayTracing.cpp)
# Port: Lumen GPU-Driven Update   (LumenSceneGPUDrivenUpdate.cpp)
# Port: Lumen Direct Lighting HWRT(LumenSceneDirectLightingHardwareRayTracing.cpp)
# Port: Lumen Visualize           (LumenVisualize.cpp)
# Port: Lumen Visualize HWRT      (LumenVisualizeHardwareRayTracing.cpp)
# Port: Lumen ScreenProbe HWRT    (LumenScreenProbeHardwareRayTracing.cpp)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionTraceConfig:
    screen_traces: bool = True
    screen_tracing_source: int = 0
    hzb_max_iterations: int = 50
    hzb_min_occupancy: int = 0
    hzb_relative_depth_threshold: float = 0.005
    history_depth_test_relative_thickness: float = 0.005
    hair_voxel_trace: bool = True
    hair_screen_trace: bool = True
    compaction_group_size_in_tiles: int = 16
    compaction_wave_ops: bool = True
    sample_scene_color_at_hit: int = 1
    sample_scene_color_rel_depth_threshold: float = 0.01
    sample_scene_color_normal_threshold_deg: float = 85.0
    far_field_sample_color_rel_depth: float = 0.1
    far_field_sample_color_normal_deg: float = 85.0
    distant_screen_traces: int = 1
    distant_trace_slope_compare_tolerance: float = 2.0
    distant_trace_max_distance: float = 200000.0
    max_bounces: int = 0
    hwrt_first_person_min_hit_distance: float = 4.0


@dataclass



@dataclass
class ReflectionRay:
    """一根反射光线。方向和起点而已，其余皆是妄想。"""
    origin: tuple; direction: tuple; pdf: float; roughness: float; pixel_coord: tuple





def generate_reflection_rays(gbuffer_sample, cfg: ReflectionTraceConfig, depth_pyramid: list, frame_index: int=0) -> list:
    """
    为单个 GBuffer 像素生成反射光线，优先 Screen Trace，失败后交 HWRT fallback。
    工欲善其事，必先利其器。但器具再好，也要有人拿得起来。
    """
    import math,random
    rng=random.Random(frame_index^hash(gbuffer_sample.get('pixel_coord',(0,0))))
    n=gbuffer_sample.get('normal',(0.,1.,0.)); pos=gbuffer_sample.get('position',(0.,0.,0.))
    r=gbuffer_sample.get('roughness',0.); px,py=gbuffer_sample.get('pixel_coord',(.5,.5))
    d=_cos_weighted_direction(n,rng.random(),rng.random())
    pdf=max(1e-4,abs(d[0]*n[0]+d[1]*n[1]+d[2]*n[2])/math.pi)
    ray=ReflectionRay(pos,d,pdf,r,(px,py))
    if cfg.screen_traces and depth_pyramid:
        hit,uv,hd=_hzb_screen_trace(ray,depth_pyramid,cfg)
        if hit: ray=ReflectionRay((uv[0],uv[1],hd),d,pdf,r,(px,py))
    return [ray]





def compact_reflection_traces(rays: list, cfg: ReflectionTraceConfig) -> list:
    """Trace Compaction：剔除已命中的，剩余按粗糙度分桶。"""
    return sorted([r for r in rays if r.origin[2]>=1.], key=lambda r: r.roughness)





class HWRTReflectionPass:
    """
    硬件光线追踪反射通道：Default、FarField、HitLighting 三种模式。
    就像人有三种状态：正常、远眺、被迫直视真相。
    """
    D,F,H='Default','FarField','HitLighting'
    def __init__(self,bias=.1,normal_bias=.1,bucket_materials=True,hit_lighting=False,far_field=True):
        self.bias=bias; self.normal_bias=normal_bias; self.bucket=bucket_materials
        self.hl=hit_lighting; self.ff=far_field
    def _remap(self,pt,flags):
        o=dict(flags)
        if pt==self.D: o['rec_refl']=False; o['ff_occ_only']=False
        elif pt==self.F: o.update(rec_refl=False,rc=False,hair=False,rec_refr=False,dst=False)
        elif pt==self.H: o.update(write_hl=False,dst=False,ff_occ_only=False)
        if pt!=self.H: o['ser']=False
        return o
    def trace(self,rays,rc,surface_cache,ff_dist=200000.):
        res=[]
        passes=[self.D]+([self.F] if self.ff else [])+([self.H] if self.hl else [])
        for ray in rays:
            rad=(0.,0.,0.); hd=float('inf')
            for pt in passes:
                ck=tuple(int(c*4) for c in ray.direction); cv=rc.get(ck,(0.,0.,0.))
                if pt==self.F:
                    if hd>ff_dist*.5: rad=cv; hd=ff_dist
                elif pt==self.H:
                    if hd==float('inf'): rad=tuple(v*.5 for v in cv); hd=1000.
                else:
                    if rad==(0.,0.,0.): rad=cv; hd=500.
            res.append((ray,rad,hd))
        if self.bucket: res.sort(key=lambda x:x[2])
        return res


@dataclass



class AstroCellPlanarReflectionParams:
    """
    Python equivalent of FPlanarReflectionUniformParameters.

    Stores the parameters needed to render a planar reflection for a cell.
    In 2-D SVG, the «reflection plane» is a horizontal axis at a given Y
    position; the reflection is implemented as a vertically flipped,
    blended copy of the cell's SVG content placed below the reflection line.

    Fields mirror the C++ struct layout (FVector4f ReflectionPlane,
    FVector3f PlanarReflectionOrigin, etc.), adapted to 2-D:
        reflection_y     → Y coordinate of the reflection plane
        blend_alpha      → reflection strength (PlanarReflectionParameters.Z)
        prefilter_sigma  → Gaussian blur on the reflected image (roughness proxy)
        is_stereo        → bIsStereo (unused in 2-D, kept for API parity)

    鲁迅式：反射参数是镜子的规格——知道镜子在哪里，知道它有多清晰，
    才能计算出倒影的位置和强度。
    """

    __slots__ = (
        "reflection_y", "blend_alpha",
        "prefilter_sigma", "is_stereo",
    )

    def __init__(self,
                 bbox:         dict,
                 roughness:    float = 0.2,
                 blend_alpha:  float = 0.35,
                 is_stereo:    bool  = False) -> None:
        # Reflection plane at the bottom edge of the cell
        self.reflection_y    = float(bbox.get("y", 0)) + float(bbox.get("h", 50))
        self.blend_alpha     = max(0.0, min(1.0, blend_alpha))
        # Prefilter sigma proportional to roughness (mirrors kernel_radius_y in C++)
        cell_h = float(bbox.get("h", 50))
        self.prefilter_sigma = roughness * cell_h * _PLANAR_REFL_ROUGHNESS_DIST
        self.is_stereo       = is_stereo





def setup_cell_planar_reflection(
    cell_id:   str,
    bbox:      dict,
    species:   str,
    roughness: float = 0.2,
) -> AstroCellPlanarReflectionParams | None:
    """
    Mirrors SetupPlanarReflectionUniformParameters().

    Returns an AstroCellPlanarReflectionParams if the species should cast a
    planar reflection, or None if the reflection plane is degenerate (C++
    early-out: ReflectionPlane.Set(0,0,0,0)).

    Species that logically rest on surfaces (eye, layers, filter) get
    a reflection; airborne species (bolt, arrow) do not.

    鲁迅式：并非所有物体都值得被镜子记录——只有「坐」在平面上的，
    才会在平面上留下倒影，飞在空中的不留印记。
    """
    if not ASTRO_PLANAR_REFLECTION_ENABLED:
        return None

    _REFLECTIVE_SPECIES = {"cil-eye", "cil-layers", "cil-filter", "cil-graph"}
    if species not in _REFLECTIVE_SPECIES:
        return None

    return AstroCellPlanarReflectionParams(bbox, roughness=roughness)





class AstroCellPlanarReflectionPrefilter:
    """
    Python equivalent of TPrefilterPlanarReflectionPS.

    Generates the prefiltered (blurred) planar reflection SVG overlay for a
    cell that supports planar reflections.  The prefilter matches UE5's
    PrefilterPlanarReflectionPS kernel: a Gaussian blur applied to the
    reflected image in the Y direction (vertical reflection axis), with
    kernel width proportional to PlanarReflectionParameters.Y (roughness dist).

    In SVG, the reflection is:
      1. A <use> element that references the original cell SVG group, scaled
         vertically by -1 and translated below the reflection plane.
      2. Wrapped in a <g> with a feGaussianBlur filter (prefilter pass).
      3. The entire reflected group has opacity = blend_alpha.

    鲁迅式：倒影是原物的谦逊注脚——存在于原物之下，
    比原物模糊，比原物暗淡，却依然忠实地记录着原物的形状。
    """

    def __init__(self, params: AstroCellPlanarReflectionParams) -> None:
        self._params = params

    def emit_svg(self, cell_id: str, bbox: dict) -> str:
        """
        Emit the planar reflection SVG fragment for this cell.

        Returns an SVG string containing the blurred, flipped reflection group,
        or an empty string if blend_alpha is negligible.

        鲁迅式：倒影的 SVG 代码是诚实的镜子逻辑——
        翻转、模糊、降低透明度，然后放到正确的位置。
        步骤不多，但每一步都不能省略。
        """
        p = self._params
        if p.blend_alpha < 0.01:
            return ""

        w        = float(bbox.get("w", 100))
        h        = float(bbox.get("h", 50))
        refl_y   = p.reflection_y
        sigma    = round(max(0.5, p.prefilter_sigma), 2)
        alpha    = round(p.blend_alpha, 3)
        filter_id = f"planar-refl-blur-{cell_id}"

        # Scale reflects around the bottom edge: translate down by h, flip Y
        transform = (
            f"translate(0,{refl_y + h:.2f}) scale(1,-1) "
            f"translate(0,{-refl_y:.2f})"
        )

        parts = [
            f'<!-- [ASTRO-PLANAR-REFL] PlanarReflectionRendering.cpp port '
            f'sigma={sigma} blend={alpha} refl_y={refl_y:.1f} -->',
            f'<defs>',
            f'  <filter id="{filter_id}" x="0" y="0" width="100%" height="100%">',
            f'    <feGaussianBlur in="SourceGraphic" stdDeviation="0 {sigma}" '
            f'result="blurred"/>',
            f'    <feColorMatrix in="blurred" type="saturate" values="0.6"/>',
            f'  </filter>',
            f'</defs>',
            f'<g class="planar-reflection" opacity="{alpha}" '
            f'transform="{transform}" filter="url(#{filter_id})">',
            f'  <use href="#cell-{cell_id}"/>',
            f'</g>',
        ]
        return "\n".join(parts)





def apply_cell_planar_reflection(
    cell_id:   str,
    species:   str,
    bbox:      dict,
    roughness: float = 0.2,
) -> str:
    """
    Top-level planar reflection application — mirrors AddProcessPlanarReflectionPass().

    Returns an SVG fragment string with the prefiltered reflection overlay,
    or an empty string if the species does not support planar reflections.

    Called from proc() after the main SVG content is generated, before final
    assembly — matching the C++ timing where the reflection pass runs after
    BasePass but before Translucency.

    鲁迅式：反射是最后的装饰，也是最诚实的自我审视——
    照见自己在世界中的位置，无论倒影多么模糊。
    """
    params = setup_cell_planar_reflection(cell_id, bbox, species, roughness)
    if params is None:
        return ""

    prefilter = AstroCellPlanarReflectionPrefilter(params)
    return prefilter.emit_svg(cell_id, bbox)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] LightShaftRendering → Python port
#
# Ported from commit upstream/unreal-renderer-ue5:
#   Renderer-Private/LightShaftRendering.cpp
#
# 鲁迅曰：「光柱是上天对少数幸运者的眷顾——定向而明亮，
# 穿透尘埃，照亮尘埃本身。光柱渲染，是对光的戏剧化处理：
# 把大气散射的模糊结果，拖成一条条从光源射出的线。」
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GLightShafts / GLightShaftQuality  → ASTRO_LIGHT_SHAFTS_ENABLED
#   GLightShaftDownsampleFactor        → _LS_DOWNSAMPLE
#   GLightShaftBlurPasses              → _LS_BLUR_PASSES
#   GLightShaftFirstPassDistance       → _LS_FIRST_PASS_DIST
#   GLightShaftBlurNumSamples          → _LS_NUM_SAMPLES
#   GLightShaftAllowTAA                → ASTRO_LS_ALLOW_TAA
#   ShouldRenderLightShafts()          → should_render_light_shafts()
#   ShouldRenderLightShaftsForLight()  → should_render_light_shafts_for_cell()
#   GetLightShaftParameters()          → get_cell_light_shaft_params()
#   ELightShaftTechnique::Bloom        → AstroCellLightShaftBloom
#   ELightShaftTechnique::Occlusion    → AstroCellLightShaftOcclusion
# ═══════════════════════════════════════════════════════════════════════════════

ASTRO_LIGHT_SHAFTS_ENABLED: bool = True
ASTRO_LS_ALLOW_TAA:         bool = True

_LS_DOWNSAMPLE:       int   = 2      # GLightShaftDownsampleFactor (clamped to [1,8])
_LS_BLUR_PASSES:      int   = 3      # GLightShaftBlurPasses
_LS_FIRST_PASS_DIST:  float = 0.1    # GLightShaftFirstPassDistance (fraction)
_LS_NUM_SAMPLES:      int   = 12     # GLightShaftBlurNumSamples
_LS_OCCLUSION_DARK:   float = 0.05   # OcclusionMaskDarkness (default from GetLightShaftOcclusionParameters)
_LS_OCCLUSION_RANGE:  float = 6.0    # OcclusionDepthRange





def get_reflection_env_cvar() -> int:
    """
    Return effective CVarReflectionEnvironment — clamps mode-2 in non-debug builds.

    鲁迅式：调试模式 2 是「推倒重来」——在发行版中，推倒重来是禁止的。
    """
    val = _REFL_ENV_EN_V2
    if val == 2:
        val = 1
    return val





def is_reflection_env_available() -> bool:
    """True when reflection environment CVar != 0 (SupportsTextureCubeArray always True)."""
    return get_reflection_env_cvar() != 0


