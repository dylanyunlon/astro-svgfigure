from dataclasses import dataclass, field
import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def compute_diffuse_irradiance_sh(
    face_colours: _PTList[tuple],
) -> _PTList[float]:
    """
    Compute diffuse irradiance SH L1 coefficients from 6 face averages.
    Mirrors FComputeSkyEnvMapDiffuseIrradianceCS (ComputeSkyEnvMapDiffuseIrradianceCS.usf):
        Integrates the cubemap × Lambertian kernel into SH L2 coefficients.
        For 6 axis-aligned faces, the SH projection simplifies to per-axis terms.

    Returns 9 floats: [Y_00_R, Y_00_G, Y_00_B, Y_1-1_R, …, Y_11_B]
    (SH L0 + L1 for each colour channel, interleaved per-channel for
    compatibility with the C++ IrradianceEnvMapSH layout).

    鲁迅式：球谐函数是数学对光照的压缩——把无限方向的辐照度
    压缩成九个系数，失去了细节，保留了大意。
    一如散文诗：意境犹在，字句已省。
    """
    # Face order: +X, -X, +Y, -Y, +Z, -Z
    # Each face contributes to SH L0 (Y_00) and L1 (Y_1-1, Y_10, Y_11)
    # via the face normal dotted with the SH basis functions.
    # SH normalization constants: Y_00 = 1/√(4π), Y_1m = √(3/(4π))
    c0 = 0.282095   # Y_00
    c1 = 0.488603   # Y_1x normalisation

    # Face normals (+X,-X,+Y,-Y,+Z,-Z)
    normals = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]

    sh = [0.0] * 9  # [L0_R, L0_G, L0_B, L1y_R, L1y_G, L1y_B, L1z_R, L1z_G, L1z_B]
    solid_angle = 4.0 * math.pi / _CAPTURE_NUM_FACES  # uniform solid angle per face

    for i, (nx, ny, nz) in enumerate(normals):
        fc = face_colours[i] if i < len(face_colours) else (0.0, 0.0, 0.0)
        fr, fg, fb = fc

        # L0 accumulation (isotropic, same for all channels)
        sh[0] += c0 * fr * solid_angle
        sh[1] += c0 * fg * solid_angle
        sh[2] += c0 * fb * solid_angle

        # L1 Y band (Y_1-1 ∝ y, Y_10 ∝ z): skip Y_11 (∝ x) for brevity
        sh[3] += c1 * ny * fr * solid_angle
        sh[4] += c1 * ny * fg * solid_angle
        sh[5] += c1 * ny * fb * solid_angle
        sh[6] += c1 * nz * fr * solid_angle
        sh[7] += c1 * nz * fg * solid_angle
        sh[8] += c1 * nz * fb * solid_angle

    return sh





@dataclass
class LumenSceneConfig:
    """
    Mirrors the per-scene Lumen configuration negotiated at renderer
    initialisation time.

    鲁迅式：这些参数就是旧时的"祖宗成法"——动一个，整个渲染管线都要重来。
    """
    global_df_resolution:     int   = _LUMEN_GLOBAL_DF_RESOLUTION
    clipmap_extent:           float = _LUMEN_GLOBAL_DF_CLIPMAP_EXTENT
    use_far_field:            bool  = False
    far_field_occlusion_only: bool  = False
    far_field_max_trace_dist: float = _LUMEN_FAR_FIELD_MAX_TRACE_DISTANCE
    far_field_dither_scale:   float = _LUMEN_FAR_FIELD_DITHER_SCALE
    surface_cache_atlas_size: int   = _LUMEN_SURFACE_CACHE_ATLAS_SIZE

    def near_field_max_trace_distance_dither_scale(self) -> float:
        return self.far_field_dither_scale if self.use_far_field else 0.0

    def near_field_scene_radius(self, culling_radius: float = float("inf")) -> float:
        if self.use_far_field and math.isfinite(culling_radius):
            return culling_radius
        return float("inf")


@dataclass
class LumenMeshCard:
    """
    Lightweight representation of a single Lumen mesh card.

    鲁迅式：一个细胞有六个面，每面都是一张名片，叫做"Card"。
    名片印满了，光才算打在实处。
    """
    card_id:    int
    center:     tuple = (0.0, 0.0, 0.0)
    extent:     tuple = (50.0, 50.0, 1.0)
    is_visible: bool  = True
    lod_level:  int   = 0

    def world_bounds(self):
        mn = tuple(c - e for c, e in zip(self.center, self.extent))
        mx = tuple(c + e for c, e in zip(self.center, self.extent))
        return mn, mx





class LumenScene:
    """
    Cell-level analogue of FLumenSceneData.

    鲁迅式：有灯才有影，有卡才有光。Scene 不过是一本花名册，
    记着谁来过、谁还没走，以及谁的面孔已经模糊。
    """

    def __init__(self, config=None):
        self.config        = config or LumenSceneConfig()
        self._cards        = {}
        self._next_id      = 0
        self.view_origin   = (0.0, 0.0, 0.0)

    def add_card(self, center, extent=(50.0, 50.0, 1.0), lod_level=0):
        cid = self._next_id; self._next_id += 1
        self._cards[cid] = LumenMeshCard(cid, center, extent, lod_level=lod_level)
        return cid

    def remove_card(self, card_id):
        self._cards.pop(card_id, None)

    def visible_cards(self):
        return [c for c in self._cards.values() if c.is_visible]

    def update_view_origin(self, new_origin):
        self.view_origin = new_origin

    def global_df_voxel_size(self):
        return (2.0 * self.config.clipmap_extent) / self.config.global_df_resolution

    def __repr__(self):
        return (
            f"LumenScene(cards={len(self._cards)}, "
            f"origin={self.view_origin}, "
            f"voxel_size={self.global_df_voxel_size():.2f})"
        )


_DIRECT_LIGHTING_UPDATE_FACTOR  = 32
_RADIOSITY_UPDATE_FACTOR        = 64


@dataclass
class LumenCardUpdateContext:
    """
    Python port of FLumenCardUpdateContext.

    鲁迅式：帝国的税吏每天只收三十二分之一，却称之为"全部"。
    UpdateFactor 亦如此——分期偿还，从不告诉你总债。
    """
    update_atlas_width:  int = 0
    update_atlas_height: int = 0
    max_update_tiles:    int = 0
    update_factor:       int = _DIRECT_LIGHTING_UPDATE_FACTOR

    @classmethod
    def build(cls, physical_atlas_size, update_factor, force_full_update=False):
        ctx = cls()
        ctx.update_factor = max(1, min(update_factor, 1024))
        if force_full_update:
            ctx.update_factor = 1
        mult = 1.0 / math.sqrt(float(ctx.update_factor))
        pw, ph = physical_atlas_size

        def _rt(v):
            return math.ceil((v + 0.5) / _LUMEN_CARD_TILE_SIZE) * _LUMEN_CARD_TILE_SIZE

        uw = max(_rt(pw * mult), _LUMEN_PHYSICAL_PAGE_SIZE)
        uh = max(_rt(ph * mult), _LUMEN_PHYSICAL_PAGE_SIZE)
        ctx.update_atlas_width  = uw
        ctx.update_atlas_height = uh
        ctx.max_update_tiles = (uw // _LUMEN_CARD_TILE_SIZE) * (uh // _LUMEN_CARD_TILE_SIZE)
        return ctx

    def update_atlas_size(self):
        return self.update_atlas_width, self.update_atlas_height


@dataclass
class LumenCombineLightingParams:
    """
    FLumenCardCombineLightingCS::FParameters — surface-cache combine pass.

    鲁迅式：漫反射、直接光、间接光，最后合成一张贴图——
    这和旧社会的"总账"没什么区别，每一笔都是别人欠的。
    """
    diffuse_color_boost:         float = 1.0
    albedo_atlas_valid:          bool  = False
    emissive_atlas_valid:        bool  = False
    direct_lighting_valid:       bool  = False
    indirect_lighting_valid:     bool  = False
    indirect_atlas_half_texel_x: float = 0.5 / 1024
    indirect_atlas_half_texel_y: float = 0.5 / 1024

    def pack(self):
        return {
            "DiffuseColorBoost": self.diffuse_color_boost,
            "IndirectLightingAtlasHalfTexelSize": (
                self.indirect_atlas_half_texel_x,
                self.indirect_atlas_half_texel_y,
            ),
        }





class LumenSceneLighting:
    """
    Drives the per-frame lighting update of a LumenScene.

    鲁迅式：照明从不是一次性的事——每帧都要重算，
    就像每天都要重新证明自己还活着。
    """

    def __init__(self, scene, direct_factor=_DIRECT_LIGHTING_UPDATE_FACTOR,
                 radiosity_factor=_RADIOSITY_UPDATE_FACTOR):
        self.scene             = scene
        self.direct_factor     = direct_factor
        self.radiosity_factor  = radiosity_factor
        self._frame_index      = 0

    def _atlas(self):
        s = self.scene.config.surface_cache_atlas_size
        return (s, s)

    def compute_direct_update_ctx(self, force_full=False):
        return LumenCardUpdateContext.build(self._atlas(), self.direct_factor, force_full)

    def compute_radiosity_update_ctx(self, force_full=False):
        return LumenCardUpdateContext.build(self._atlas(), self.radiosity_factor, force_full)

    def tick(self, force_full_update=False):
        self._frame_index += 1
        d = self.compute_direct_update_ctx(force_full_update)
        r = self.compute_radiosity_update_ctx(force_full_update)
        print(
            f"[LUMEN-LIGHTING] frame={self._frame_index} "
            f"direct_tiles={d.max_update_tiles} radiosity_tiles={r.max_update_tiles}",
            file=sys.stderr,
        )
        return d, r





@dataclass
class LumenSurfaceCache:
    """
    Card atlas allocation and per-frame copy scheduling.

    鲁迅式：表面缓存不是记忆，是遗忘的方式——
    只记住上一帧照亮过的那几个格子，其余的，下帧再说。
    """
    atlas_size:    int                       = _LUMEN_SURFACE_CACHE_ATLAS_SIZE
    compression:   SurfaceCacheCompression   = SurfaceCacheCompression.DISABLED
    dilation_mode: SurfaceCacheDilationMode  = SurfaceCacheDilationMode.DISABLED
    cull_underground: bool                   = False
    _atlas:        dict                      = field(default_factory=dict)

    def allocate_page(self, page_id):
        for layer in SurfaceCacheLayer:
            self._atlas[(layer, page_id)] = list(_SURFACE_LAYER_CONFIGS[layer].clear_value)

    def free_page(self, page_id):
        for layer in SurfaceCacheLayer:
            self._atlas.pop((layer, page_id), None)

    def copy_capture_to_atlas(self, page_id, layer, texels):
        if (layer, page_id) not in self._atlas:
            self.allocate_page(page_id)
        data = list(texels)
        if self.dilation_mode != SurfaceCacheDilationMode.DISABLED and data:
            data = [data[0]] + data + [data[-1]]
        self._atlas[(layer, page_id)] = data

    def is_compressed(self):
        return self.compression != SurfaceCacheCompression.DISABLED


_SSBN_DOWNSAMPLE_FACTOR        = 2
_SSBN_SLICE_COUNT              = 2
_SSBN_STEPS_PER_SLICE          = 3
_SSBN_FOLIAGE_OCC_STRENGTH     = 0.7
_SSBN_MAX_MULTIBOUNCE_ALBEDO   = 0.5
_SSBN_SLOPE_TOLERANCE          = 0.5
_SSBN_FOREGROUND_REJECT_FRACTION = 0.3


@dataclass
class LumenScreenProbe:
    """
    A single screen-space probe: world position + radiance octahedron.

    鲁迅式：探针不探人，只探光。
    探到的光存进历史，历史再决定下一帧要不要信任它。
    """
    probe_id:           int
    screen_pos:         tuple = (0, 0)
    world_pos:          tuple = (0.0, 0.0, 0.0)
    scene_depth:        float = 0.0
    radiance:           list  = field(default_factory=list)
    history_radiance:   list  = field(default_factory=list)
    frames_accumulated: int   = 0
    is_moving:          bool  = False





@dataclass
class LumenFrameOutputs:
    """Per-frame products of the Lumen pipeline tick."""
    direct_update_ctx:    LumenCardUpdateContext
    radiosity_update_ctx: LumenCardUpdateContext
    combine_params:       LumenCombineLightingParams
    ao_map:               list
    filtered_probes:      list
    probe_rays:           dict





def lumen_frame_tick(
    scene, lighting, surface_cache,
    depth_buffer, normal_buffer,
    probes, history_probes,
    ao_config=None,
    force_full_update=False,
    epoch=0,
):
    """
    One full Lumen frame for a cell sub-scene.

    鲁迅式：七个步骤，七道工序。每道都不能省，省了就有瑕疵。
    但没有人会数清楚，他们只看最终画面是否好看。
    """
    scene.update_view_origin((float(epoch), 0.0, 0.0))
    d_ctx, r_ctx = lighting.tick(force_full_update)
    combine = LumenCombineLightingParams(
        diffuse_color_boost     = 1.0,
        albedo_atlas_valid      = True,
        emissive_atlas_valid    = True,
        direct_lighting_valid   = d_ctx.max_update_tiles > 0,
        indirect_lighting_valid = r_ctx.max_update_tiles > 0,
    )
    ao_map  = compute_bent_normal_ao(depth_buffer, normal_buffer, ao_config)
    probes  = composite_traces_with_scatter(probes)
    probes  = temporally_accumulate_probe_radiance(probes, history_probes)
    probes  = spatial_filter_probes(probes)
    probe_rays = {p.probe_id: generate_importance_sampled_rays(p) for p in probes}
    return LumenFrameOutputs(
        direct_update_ctx    = d_ctx,
        radiosity_update_ctx = r_ctx,
        combine_params       = combine,
        ao_map               = ao_map,
        filtered_probes      = probes,
        probe_rays           = probe_rays,
    )


# =============================================================================
# DistanceField AO + LightingPost + ObjectCulling + ObjectManagement + ScreenGrid
# 移植自 UE5 Renderer-Private，鲁迅式注释 20 %
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 全局开关 —— 对应 CVar r.DistanceFieldAO / r.AOQuality 等
# 每一个 CVar 背后都是一场争论：性能组要关，画面组要开。
# ---------------------------------------------------------------------------
G_DISTANCE_FIELD_AO: bool = True
G_DISTANCE_FIELD_AO_QUALITY: int = 2          # 0=off 1=medium 2=high
G_AO_OBJECT_DISTANCE_FIELD: bool = True
G_AO_GLOBAL_DISTANCE_FIELD: bool = True
G_AO_GLOBAL_DF_START_DISTANCE: float = 1000.0
G_AO_MAX_VIEW_DISTANCE: float = 20000.0
G_AO_STEP_EXPONENT_SCALE: float = 0.5
G_AO_DOWNSAMPLE_FACTOR: int = 2
G_CONE_TRACE_DOWNSAMPLE_FACTOR: int = 4

# LightingPost
G_AO_USE_HISTORY: bool = True
G_AO_CLEAR_HISTORY: bool = False
G_AO_HISTORY_STABILITY_PASS: bool = True
G_AO_HISTORY_WEIGHT: float = 0.85
G_AO_HISTORY_DISTANCE_THRESHOLD: float = 30.0
G_AO_VIEW_FADE_DISTANCE_SCALE: float = 0.70

# ObjectCulling
G_AO_SCATTER_TILE_CULLING: bool = True
G_AVERAGE_DF_OBJECTS_PER_CULL_TILE: int = 512

# ObjectManagement
G_MESH_DF_MAX_OBJECT_BOUNDING_RADIUS: float = 100_000.0
G_DF_PARALLEL_UPDATE: bool = False
G_MESH_SDF_SURFACE_BIAS_EXPAND: float = 0.25

# ScreenGrid
G_AO_USE_JITTER: bool = True
G_DF_AO_TRAVERSE_MIPS: bool = True
NUM_CONE_SAMPLE_DIRECTIONS: int = 9
CONE_TRACE_GLOBAL_DF_TILE_SIZE: int = 8





@dataclass
class LumenCardMetrics:
    max_distance:float=0.; texel_density:float=0.; far_field_density:float=.001
    far_field_distance:float=40000.; min_resolution:int=4; capture_margin:float=0.


@dataclass



class LumenGPUSceneReadback:
    """
    环形缓冲区，CPU 读回 GPU 写入的增删操作。
    GPU 在前面写，CPU 在后面读，永远差着那么几帧——这就是渲染管线，也是人生。
    """
    N=4; MAX_A=4096; MAX_R=4096
    def __init__(self): self._r=[None]*self.N; self._wi=0; self._p=0
    def is_full(self): return self._p>=self.N
    def submit(self,add,rem):
        if self.is_full(): return False
        self._r[self._wi]=(add[:self.MAX_A],rem[:self.MAX_R])
        self._wi=(self._wi+1)%self.N; self._p=min(self._p+1,self.N); return True
    def consume_latest(self):
        if not self._p: return[],[]
        ri=(self._wi-self._p)%self.N; res=self._r[ri]; self._r[ri]=None; self._p-=1
        return res if res else ([],[])





class LumenVisualizeMode:
    """调试模式是工程师写给自己的信，用户永远不会读，但它必须存在。"""
    DISABLE=0;OVERVIEW=1;PERFORMANCE_OVERVIEW=2;LUMEN_SCENE=3;REFLECTION_VIEW=4
    SURFACE_CACHE_COVERAGE=5;GEOMETRY_NORMALS=6;DEDICATED_REFLECTION_RAYS=7
    ALBEDO=8;NORMALS=9;EMISSIVE=10;CARD_WEIGHTS=11;DIRECT_LIGHTING=12
    INDIRECT_LIGHTING=13;LOCAL_POSITION=14;VELOCITY=15;DIRECT_LIGHTING_UPDATES=16
    INDIRECT_LIGHTING_UPDATES=17;LAST_USED_PAGES=18;LAST_USED_HIGHRES_PAGES=19
    CARD_TILE_SHADOW_FACTOR=20;CARD_SHARING_ID=21;SCREEN_PROBE_FAST_UPDATE=22
    SCREEN_PROBE_FRAMES_ACCUM=23;RADIOSITY_FRAMES_ACCUM=24


def visualize_lumen_scene(surface_cache, scene_cards:list, cfg:VisualizeConfig) -> dict:
    """按模式输出调试数据。这不是最终画面，只是看透了管线的 X 光片。"""
    out={}
    if cfg.mode==LumenVisualizeMode.DISABLE: return out
    M=LumenVisualizeMode
    for card in scene_cards:
        coord=card.get('screen_coord',(0,0))
        if   cfg.mode==M.ALBEDO:           out[coord]=card.get('albedo',(.5,.5,.5))
        elif cfg.mode==M.NORMALS:          out[coord]=tuple(v*.5+.5 for v in card.get('normal',(0.,0.,1.)))
        elif cfg.mode==M.EMISSIVE:         out[coord]=card.get('emissive',(0.,0.,0.))
        elif cfg.mode==M.DIRECT_LIGHTING:  out[coord]=card.get('direct_lighting',(0.,0.,0.))
        elif cfg.mode==M.INDIRECT_LIGHTING:out[coord]=card.get('indirect_lighting',(0.,0.,0.))
        elif cfg.mode==M.CARD_WEIGHTS:     w=card.get('weight',0.); out[coord]=(w,w,w)
        elif cfg.mode==M.SURFACE_CACHE_COVERAGE: c=1. if card.get('resident') else 0.; out[coord]=(c,c*.5,0.)
        else: out[coord]=(.2,.2,.2)
    return out


