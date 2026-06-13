# =============================================================================
# AstroCellGlobalIllumination — 全局光照核心
# (ported from Lumen / LumenDiffuseIndirect / LumenIrradianceFieldGather)
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 探针亦如此——不更新，便腐朽。每一帧的辐射度都是一次微小的觉醒。
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# -----------------------------------------------------------------------------
# 配置旗标 — 对应 CVarLumen* 系列控制变量
# -----------------------------------------------------------------------------
_ASTRO_GI_ENABLED          = True   # r.Lumen.DiffuseIndirect.Allow
_ASTRO_GI_ASYNC_COMPUTE    = True   # r.Lumen.DiffuseIndirect.AsyncCompute
_ASTRO_GI_TRACE_MESH_SDFS  = False  # r.Lumen.TraceMeshSDFs
_ASTRO_GI_TRACE_DISTANCE   = 180.0  # r.Lumen.TraceMeshSDFs.TraceDistance
_ASTRO_GI_SURFACE_BIAS     = 5.0    # r.Lumen.DiffuseIndirect.SurfaceBias
_ASTRO_GI_TRACE_STEP       = 1.0    # r.Lumen.DiffuseIndirect.TraceStepFactor
_ASTRO_GI_LIGHTING_FORMAT  = 0      # 0=R11G11B10, 1=Float16, 2=Float32


def astro_gi_is_allowed() -> bool:
    """全局光照总开关——思想被封锁，光照亦可被封锁；但封锁不等于消灭。"""
    return _ASTRO_GI_ENABLED


def astro_gi_get_lighting_format() -> str:
    """
    返回辐射度数据格式。
    鲁迅式：精度是奢侈品，R11G11B10 是大多数人负担得起的真相。
    """
    if _ASTRO_GI_LIGHTING_FORMAT == 2:
        return "Float32_RGBA"
    elif _ASTRO_GI_LIGHTING_FORMAT == 1:
        return "Float16_RGBA"
    return "R11G11B10"


# -----------------------------------------------------------------------------
# AstroCellGatherCvarState — 对应 FLumenGatherCvarState
# 汇聚所有可调参数，统一管理，避免散落各处的「各自为政」。
# -----------------------------------------------------------------------------
@dataclass
class AstroCellGatherCvarState:
    trace_mesh_sdfs:      bool  = _ASTRO_GI_TRACE_MESH_SDFS
    mesh_sdf_trace_dist:  float = _ASTRO_GI_TRACE_DISTANCE
    surface_bias:         float = _ASTRO_GI_SURFACE_BIAS
    voxel_tracing_mode:   int   = 0
    direct_lighting:      bool  = False
    trace_step_factor:    float = _ASTRO_GI_TRACE_STEP
    min_sample_radius:    float = 10.0
    min_trace_distance:   float = 0.0
    card_interpolate_radius: float = 10.0
    card_trace_end_dist:  float = 4000.0
    trace_distance_scale: float = 1.0


_GATHER_CVARS = AstroCellGatherCvarState()


# =============================================================================
# AstroCellDiffuseProbe — 漫反射间接光照探针
# (ported from LumenDiffuseIndirect)
#
# 鲁迅式：每一个探针都是一面小镜子，映照出周围世界的光与暗。
# 但镜子会积灰——故需按帧刷新，否则映出的不过是昨日的幻象。
# =============================================================================

@dataclass
class AstroCellDiffuseProbe:
    """
    屏幕空间漫反射探针，对应 Lumen Screen Probe Gather 单条记录。
    每个探针锚定于某像素位置，收集半球辐射度后写入辐射度缓存。
    """
    screen_x:     float = 0.0   # 像素坐标 X（归一化 [0,1]）
    screen_y:     float = 0.0   # 像素坐标 Y（归一化 [0,1]）
    world_pos:    tuple = (0.0, 0.0, 0.0)   # 世界坐标
    normal:       tuple = (0.0, 1.0, 0.0)   # 表面法线
    radiance:     tuple = (0.0, 0.0, 0.0)   # 采集到的辐射度 (R, G, B)
    hit_distance: float = 0.0   # 最近遮挡物距离（用于 occlusion）
    valid:        bool  = False  # 本帧是否有效

    def gather(self, scene_radiance_fn, cvars: AstroCellGatherCvarState) -> None:
        """
        执行半球采样，更新 self.radiance。
        scene_radiance_fn(origin, direction) -> (r, g, b, hit_dist)
        此处为接口约定，实际光线求交由上层场景提供。
        """
        if not astro_gi_is_allowed():
            return
        # 鲁迅式：真正的采样从不因「条件不够好」而推迟——推迟便是永不。
        origin = (
            self.world_pos[0] + self.normal[0] * cvars.surface_bias,
            self.world_pos[1] + self.normal[1] * cvars.surface_bias,
            self.world_pos[2] + self.normal[2] * cvars.surface_bias,
        )
        r, g, b, hit_d = scene_radiance_fn(origin, self.normal)
        self.radiance     = (r, g, b)
        self.hit_distance = hit_d
        self.valid        = True

    def to_dict(self) -> dict:
        return {
            "screen": (self.screen_x, self.screen_y),
            "world_pos": self.world_pos,
            "normal":    self.normal,
            "radiance":  self.radiance,
            "hit_dist":  self.hit_distance,
            "valid":     self.valid,
        }


@dataclass
class AstroCellDiffuseProbeGrid:
    """
    屏幕探针格（Screen Probe Grid）。
    将屏幕分成若干 tile，每 tile 放置一枚 AstroCellDiffuseProbe。
    对应 Lumen Screen Probe Gather 的 tile 分配逻辑。
    """
    tile_size_px: int = 8           # 每 tile 像素尺寸
    probes: List[AstroCellDiffuseProbe] = field(default_factory=list)

    def build_from_gbuffer(self, width_px: int, height_px: int,
                           gbuffer_sampler) -> None:
        """
        按 tile 采样 GBuffer（世界坐标 + 法线），生成探针列表。
        gbuffer_sampler(tile_cx, tile_cy) -> (world_pos, normal) | None
        """
        self.probes.clear()
        cols = max(1, width_px  // self.tile_size_px)
        rows = max(1, height_px // self.tile_size_px)
        for row in range(rows):
            for col in range(cols):
                cx = (col + 0.5) / cols
                cy = (row + 0.5) / rows
                result = gbuffer_sampler(cx, cy)
                if result is None:
                    continue
                world_pos, normal = result
                probe = AstroCellDiffuseProbe(
                    screen_x=cx, screen_y=cy,
                    world_pos=world_pos, normal=normal,
                )
                self.probes.append(probe)

    def gather_all(self, scene_radiance_fn,
                   cvars: AstroCellGatherCvarState = _GATHER_CVARS) -> None:
        """遍历所有探针执行采样——众生皆苦，逐一采集，不遗漏一粒光子。"""
        for probe in self.probes:
            probe.gather(scene_radiance_fn, cvars)

    def valid_count(self) -> int:
        return sum(1 for p in self.probes if p.valid)


# =============================================================================
# AstroCellRadianceCache — 辐射度缓存（World Space Probe Cache）
# (ported from LumenRadianceCache)
#
# 鲁迅式：缓存是记忆，记忆是负担，但没有记忆的光照只是当下的幻觉。
# 探针被放置在世界空间，按 clipmap 分级，近处精细，远处粗糙——
# 正如人对近事敏感，对远事麻木。
# =============================================================================

@dataclass
class AstroCellRadianceCacheProbe:
    """
    世界空间辐射度探针，存储某位置的球谐辐射度。
    对应 LumenRadianceCache 中的 FRadianceCacheClipmap 单探针条目。
    """
    world_pos:      tuple       = (0.0, 0.0, 0.0)
    clipmap_level:  int         = 0
    sh_coeffs:      List[float] = field(default_factory=lambda: [0.0] * 9)
    last_update_frame: int      = -1
    valid:          bool        = False
    # 探针偏移（IrradianceProbeOffsetMode=1 时由 GBuffer 修正）
    world_offset:   tuple       = (0.0, 0.0, 0.0)

    def update(self, sh_coeffs: List[float], frame_index: int) -> None:
        """写入新的球谐系数，标记为有效。沉默已久的探针，终于开口说话。"""
        if len(sh_coeffs) >= 9:
            self.sh_coeffs = sh_coeffs[:9]
        self.last_update_frame = frame_index
        self.valid = True

    def interpolate_irradiance(self, normal: tuple) -> tuple:
        """
        用二阶球谐函数向指定法线方向插值，返回 (R, G, B) 辐照度。
        球谐插值：E ≈ max(0, dot(N, L)) 的 SH 投影（Lambertian kernel）。
        此处简化为 SH L1 近似。
        """
        if not self.valid:
            return (0.0, 0.0, 0.0)
        nx, ny, nz = normal
        # SH basis Y_0^0 = 0.282095, Y_1^-1 = 0.488603*y,
        #           Y_1^0 = 0.488603*z, Y_1^1 = 0.488603*x
        c = self.sh_coeffs
        # 每通道 3 个系数（L0+L1），共 9 = 3 channels × 3 coefficients
        def ch(offset):
            return max(0.0,
                       c[offset + 0] * 0.282095
                       + c[offset + 1] * 0.488603 * ny
                       + c[offset + 2] * 0.488603 * nz
                       # L1 x 项放到 offset+2 的位置做简化
                       ) * math.pi
        return (ch(0), ch(3), ch(6))


@dataclass
class AstroCellRadianceCacheClipmapLevel:
    """
    单层 clipmap 的探针网格。
    分辨率随层级指数扩展：extent = base_extent × (dist_base ^ level)
    """
    level:        int   = 0
    world_extent: float = 5000.0   # 此层覆盖的世界空间半径
    grid_res:     int   = 64        # 探针格单侧数量
    probes: Dict[tuple, AstroCellRadianceCacheProbe] = field(
        default_factory=dict
    )

    def cell_size(self) -> float:
        """单探针格的世界空间尺寸。精度是距离的函数——越近越贵。"""
        return (self.world_extent * 2.0) / max(1, self.grid_res)

    def probe_key(self, world_pos: tuple) -> tuple:
        """将世界坐标量化为探针格索引，作为字典键。"""
        cs = self.cell_size()
        ix = int(math.floor(world_pos[0] / cs))
        iy = int(math.floor(world_pos[1] / cs))
        iz = int(math.floor(world_pos[2] / cs))
        return (ix, iy, iz)

    def get_or_create_probe(self, world_pos: tuple) -> AstroCellRadianceCacheProbe:
        key = self.probe_key(world_pos)
        if key not in self.probes:
            # 新探针诞生——如新生儿，尚无记忆，但潜力无限。
            probe = AstroCellRadianceCacheProbe(
                world_pos=world_pos, clipmap_level=self.level
            )
            self.probes[key] = probe
        return self.probes[key]

    def find_probe(self, world_pos: tuple) -> Optional[AstroCellRadianceCacheProbe]:
        key = self.probe_key(world_pos)
        return self.probes.get(key)

    def valid_probe_count(self) -> int:
        return sum(1 for p in self.probes.values() if p.valid)


class AstroCellRadianceCache:
    """
    多层 clipmap 辐射度缓存，对应 LumenRadianceCache 的完整状态。

    架构要点（移植自 LumenRadianceCache.cpp）：
      • NUM_CLIPMAPS 层，每层 grid_res^3 个探针槽位（按需分配）
      • 探针按优先级队列分批更新（budget = num_probes_to_trace_budget/帧）
      • 空间滤波：相邻探针球谐插值平滑（SpatialFilterProbes）
      • 超采样：BRDF 权重高的方向多发射光线（SupersampleTileBRDFThreshold）

    鲁迅式：缓存是个仁慈的谎言——它告诉你上一帧的光仍然有效，
    直到这一帧的新光把它推翻。世界就是在这样的推翻中缓慢前进的。
    """

    NUM_CLIPMAPS      = 6
    BASE_EXTENT       = 5000.0
    DIST_BASE         = 2.0
    PROBES_PER_FRAME  = 100   # r.Lumen.RadianceCache.NumProbesToTraceBudget
    SUPERSAMPLE_BRDF  = 0.1   # r.Lumen.RadianceCache.SupersampleTileBRDFThreshold
    DOWNSAMPLE_DIST   = 4000.0

    def __init__(self):
        self.clipmaps: List[AstroCellRadianceCacheClipmapLevel] = []
        self._build_clipmaps()
        self.frame_index: int = 0
        self._update_queue: List[AstroCellRadianceCacheProbe] = []

    def _build_clipmaps(self):
        """按指数间距构建各层 clipmap——近处精细，远处稀疏，一如人心。"""
        self.clipmaps.clear()
        for lvl in range(self.NUM_CLIPMAPS):
            extent = self.BASE_EXTENT * (self.DIST_BASE ** lvl)
            cl = AstroCellRadianceCacheClipmapLevel(
                level=lvl, world_extent=extent
            )
            self.clipmaps.append(cl)

    def _pick_clipmap(self, world_pos: tuple, viewer_pos: tuple) -> int:
        """
        根据探针到观察者的距离选择最合适的 clipmap 层。
        距离越远，层级越高，精度越低——这是性价比，不是歧视。
        """
        dx = world_pos[0] - viewer_pos[0]
        dy = world_pos[1] - viewer_pos[1]
        dz = world_pos[2] - viewer_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        for lvl, cl in enumerate(self.clipmaps):
            if dist <= cl.world_extent:
                return lvl
        return self.NUM_CLIPMAPS - 1

    def mark_probe(self, world_pos: tuple, viewer_pos: tuple
                   ) -> AstroCellRadianceCacheProbe:
        """
        标记某世界坐标需要辐射度数据，返回对应探针（新建或已有）。
        对应 Lumen Mark 阶段（MarkRadianceCacheProbesUsedByVisualizeScene）。
        """
        lvl = self._pick_clipmap(world_pos, viewer_pos)
        probe = self.clipmaps[lvl].get_or_create_probe(world_pos)
        if probe not in self._update_queue and not probe.valid:
            self._update_queue.append(probe)
        return probe

    def update_probes(self, sh_compute_fn, max_per_frame: int = PROBES_PER_FRAME
                      ) -> int:
        """
        按预算更新探针球谐系数。
        sh_compute_fn(world_pos) -> List[float] (9 coefficients)
        返回本帧实际更新数量——勤勉是美德，但也要量力而行。
        """
        updated = 0
        pending = self._update_queue[:max_per_frame]
        self._update_queue = self._update_queue[max_per_frame:]
        for probe in pending:
            coeffs = sh_compute_fn(probe.world_pos)
            probe.update(coeffs, self.frame_index)
            updated += 1
        self.frame_index += 1
        return updated

    def interpolate(self, world_pos: tuple, normal: tuple,
                    viewer_pos: tuple) -> tuple:
        """
        在给定世界位置和法线方向插值辐照度。
        先查最精细 clipmap，若无有效探针则向上查粗层——
        如同问题无法在基层解决，便逐级上报，直到找到答案。
        """
        for lvl in range(self.NUM_CLIPMAPS):
            cl = self.clipmaps[lvl]
            probe = cl.find_probe(world_pos)
            if probe and probe.valid:
                return probe.interpolate_irradiance(normal)
        return (0.0, 0.0, 0.0)

    def stats(self) -> dict:
        """统计各层探针数量——数字是客观的，即便现实令人沮丧。"""
        return {
            f"clipmap_{cl.level}": {
                "extent":      cl.world_extent,
                "cell_size":   cl.cell_size(),
                "total":       len(cl.probes),
                "valid":       cl.valid_probe_count(),
            }
            for cl in self.clipmaps
        }


# =============================================================================
# AstroCellMeshCards — 网格卡片系统（Surface Cache）
# (ported from LumenMeshCards)
#
# 鲁迅式：每一张卡片都是网格表面的一份档案。
# 档案记录着法线、辐射度、遮挡——它们共同构成场景的「官方历史」。
# 官方历史不总是完整的，但没有它，渲染便无从立足。
# =============================================================================

@dataclass
class AstroCellMeshCardFace:
    """
    单个网格卡片面，对应 FLumenCard。
    每个面对应轴对齐方向之一（±X, ±Y, ±Z 共6面）。
    """
    direction_index: int   = 0   # 0..5 = +X,-X,+Y,-Y,+Z,-Z
    origin:          tuple = (0.0, 0.0, 0.0)
    extent:          tuple = (100.0, 100.0, 1.0)  # OBB 半尺寸
    axis_x:          tuple = (1.0, 0.0, 0.0)
    axis_y:          tuple = (0.0, 1.0, 0.0)
    axis_z:          tuple = (0.0, 0.0, 1.0)
    allocated:       bool  = False
    is_heightfield:  bool  = False
    is_far_field:    bool  = False
    # 辐射度缓冲（低分辨率漫反射 + 高分辨率镜面）
    radiance_diffuse:  Optional[tuple] = None   # (R, G, B)
    radiance_specular: Optional[tuple] = None   # (R, G, B)

    def surface_area(self) -> float:
        """计算面的近似表面积，用于判断是否值得分配资源。"""
        ex, ey, _ = self.extent
        return 4.0 * ex * ey

    def capture(self, scene_sample_fn) -> None:
        """
        从场景采样此面的辐射度。
        scene_sample_fn(origin, normal) -> (r, g, b)
        分两次：低分辨率漫反射 + 高分辨率镜面。
        鲁迅式：采集真相需要两次——一次看大局，一次看细节。
        """
        if not self.allocated:
            return
        normal = self.axis_z   # 面法线即 OBB z 轴
        self.radiance_diffuse  = scene_sample_fn(self.origin, normal)
        # 镜面用更近的偏移采样，模拟高分辨率
        spec_origin = (
            self.origin[0] + normal[0] * 0.5,
            self.origin[1] + normal[1] * 0.5,
            self.origin[2] + normal[2] * 0.5,
        )
        self.radiance_specular = scene_sample_fn(spec_origin, normal)


@dataclass
class AstroCellMeshCards:
    """
    单个图元的全套网格卡片（最多 6 面），对应 FLumenMeshCards。

    关键参数（移植自 LumenMeshCards.cpp 控制变量）：
      min_size            = 10.0  (r.LumenScene.SurfaceCache.MeshCardsMinSize)
      merge_components    = True  (r.LumenScene.SurfaceCache.MeshCardsMergeComponents)
      merged_max_world    = 10000 (r.LumenScene.SurfaceCache.MeshCardsMergedMaxWorldSize)
      cull_faces          = True  (r.LumenScene.SurfaceCache.MeshCardsCullFaces)
    """
    primitive_id:        str   = ""
    bounds_origin:       tuple = (0.0, 0.0, 0.0)
    bounds_extent:       tuple = (100.0, 100.0, 100.0)
    min_size:            float = 10.0
    merge_components:    bool  = True
    cull_faces:          bool  = True
    faces: List[AstroCellMeshCardFace] = field(default_factory=list)

    # 方向向量表（+X,-X,+Y,-Y,+Z,-Z）
    _DIRECTIONS = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]

    def build_faces(self) -> None:
        """
        为包围盒的六个面生成卡片，剔除面积过小的面。
        鲁迅式：六个面，六种方向——世界的包围来自四面八方，
        但只有面积够大的面才值得被记录，其余的沉没进历史的尘埃。
        """
        self.faces.clear()
        ex, ey, ez = self.bounds_extent
        sizes = [ey*ez, ey*ez, ex*ez, ex*ez, ex*ey, ex*ey]  # 各面面积
        for i, (dx, dy, dz) in enumerate(self._DIRECTIONS):
            area = sizes[i]
            if area < self.min_size * self.min_size:
                continue   # 剔除过小面——小到无法被看见，便无需存在
            # 面中心 = bounds_origin + 方向 × 对应半轴
            offsets = [ex, ex, ey, ey, ez, ez]
            cx = self.bounds_origin[0] + dx * offsets[i]
            cy = self.bounds_origin[1] + dy * offsets[i]
            cz = self.bounds_origin[2] + dz * offsets[i]
            face = AstroCellMeshCardFace(
                direction_index=i,
                origin=(cx, cy, cz),
                extent=(ex, ey, ez),
                axis_z=(dx, dy, dz),
                allocated=True,
            )
            self.faces.append(face)

    def capture_all(self, scene_sample_fn) -> int:
        """
        更新所有已分配面的辐射度缓存。
        返回实际采样面数——劳动的成果，以数字计量。
        """
        captured = 0
        for face in self.faces:
            face.capture(scene_sample_fn)
            captured += 1
        return captured

    def min_surface_area(self) -> float:
        """返回所有有效面中最小的表面积——最薄弱的一环决定了整体的下限。"""
        areas = [f.surface_area() for f in self.faces if f.allocated]
        return min(areas) if areas else 0.0

    def to_dict(self) -> dict:
        return {
            "primitive_id":  self.primitive_id,
            "bounds_origin": self.bounds_origin,
            "bounds_extent": self.bounds_extent,
            "face_count":    len(self.faces),
            "faces": [
                {
                    "dir":       f.direction_index,
                    "origin":    f.origin,
                    "area":      f.surface_area(),
                    "allocated": f.allocated,
                    "diffuse":   f.radiance_diffuse,
                    "specular":  f.radiance_specular,
                }
                for f in self.faces
            ],
        }


class AstroCellMeshCardsRegistry:
    """
    场景中所有图元的网格卡片注册表，对应 LumenSceneData 中的卡片管理。
    提供增删查接口，以及批量 capture 入口。

    鲁迅式：注册表是制度，制度是秩序的保障，
    但秩序若不被执行，不过是一纸空文。
    """

    def __init__(self):
        self._cards: Dict[str, AstroCellMeshCards] = {}

    def register(self, primitive_id: str,
                 bounds_origin: tuple, bounds_extent: tuple,
                 min_size: float = 10.0) -> AstroCellMeshCards:
        """注册新图元——欢迎加入场景，请在六面留下你的辐射度档案。"""
        mc = AstroCellMeshCards(
            primitive_id=primitive_id,
            bounds_origin=bounds_origin,
            bounds_extent=bounds_extent,
            min_size=min_size,
        )
        mc.build_faces()
        self._cards[primitive_id] = mc
        return mc

    def remove(self, primitive_id: str) -> None:
        """注销图元——消失，但曾经存在过的光照痕迹已写入缓存。"""
        self._cards.pop(primitive_id, None)

    def get(self, primitive_id: str) -> Optional[AstroCellMeshCards]:
        return self._cards.get(primitive_id)

    def capture_all(self, scene_sample_fn) -> int:
        """全场景表面缓存更新——劳模帧，每一面都不放过。"""
        total = 0
        for mc in self._cards.values():
            total += mc.capture_all(scene_sample_fn)
        return total

    def stats(self) -> dict:
        return {
            "primitives": len(self._cards),
            "total_faces": sum(len(mc.faces) for mc in self._cards.values()),
            "allocated_faces": sum(
                sum(1 for f in mc.faces if f.allocated)
                for mc in self._cards.values()
            ),
        }


# =============================================================================
# AstroCellGlobalIlluminationPipeline — GI 总调度管线
#
# 将以上四个子系统串联：
#   1. AstroCellMeshCardsRegistry  → 表面缓存采集
#   2. AstroCellDiffuseProbeGrid   → 屏幕探针采集
#   3. AstroCellRadianceCache      → 世界空间辐射度缓存更新
#   4. 合并输出：每像素最终 GI 辐照度
#
# 鲁迅式：流水线是分工，分工是文明，
# 但流水线的每一环都必须有人负责——否则链条断裂，光照崩溃，
# 黑屏比黑暗更令人绝望，因为它是人为的。
# =============================================================================

class AstroCellGlobalIlluminationPipeline:
    """
    全局光照总调度管线。
    使用方式::

        pipeline = AstroCellGlobalIlluminationPipeline()
        pipeline.mesh_cards_registry.register("wall_01", (0,0,0), (500,10,300))
        pipeline.tick(
            viewer_pos=(0, 0, 0),
            gbuffer_sampler=my_gbuffer_fn,
            scene_radiance_fn=my_radiance_fn,
            sh_compute_fn=my_sh_fn,
            scene_sample_fn=my_sample_fn,
            viewport_w=1920, viewport_h=1080,
        )
        irradiance = pipeline.query_irradiance(world_pos, normal, viewer_pos)
    """

    def __init__(self,
                 cvars: AstroCellGatherCvarState = None,
                 tile_size_px: int = 8):
        self.cvars               = cvars or AstroCellGatherCvarState()
        self.mesh_cards_registry = AstroCellMeshCardsRegistry()
        self.diffuse_probe_grid  = AstroCellDiffuseProbeGrid(tile_size_px)
        self.radiance_cache      = AstroCellRadianceCache()
        self._frame_stats: dict  = {}

    def tick(self,
             viewer_pos: tuple,
             gbuffer_sampler,
             scene_radiance_fn,
             sh_compute_fn,
             scene_sample_fn,
             viewport_w: int = 1920,
             viewport_h: int = 1080) -> dict:
        """
        执行单帧 GI 更新，返回本帧统计信息。

        执行顺序（对应 Lumen 渲染帧序）：
          Phase 1 — Surface Cache Capture（MeshCards）
          Phase 2 — Screen Probe Gather（DiffuseProbe）
          Phase 3 — Radiance Cache Update（RadianceCache）
        """
        if not astro_gi_is_allowed():
            return {"gi_enabled": False}

        # Phase 1: 表面缓存采集
        captured_faces = self.mesh_cards_registry.capture_all(scene_sample_fn)

        # Phase 2: 屏幕探针采集
        self.diffuse_probe_grid.build_from_gbuffer(
            viewport_w, viewport_h, gbuffer_sampler
        )
        self.diffuse_probe_grid.gather_all(scene_radiance_fn, self.cvars)

        # Phase 2.5: 将有效探针位置标记到辐射度缓存
        for probe in self.diffuse_probe_grid.probes:
            if probe.valid:
                self.radiance_cache.mark_probe(probe.world_pos, viewer_pos)

        # Phase 3: 辐射度缓存更新
        updated_probes = self.radiance_cache.update_probes(sh_compute_fn)

        self._frame_stats = {
            "gi_enabled":       True,
            "lighting_format":  astro_gi_get_lighting_format(),
            "captured_faces":   captured_faces,
            "screen_probes":    len(self.diffuse_probe_grid.probes),
            "valid_probes":     self.diffuse_probe_grid.valid_count(),
            "rc_updated":       updated_probes,
            "rc_stats":         self.radiance_cache.stats(),
            "mc_stats":         self.mesh_cards_registry.stats(),
        }
        return self._frame_stats

    def query_irradiance(self, world_pos: tuple, normal: tuple,
                         viewer_pos: tuple) -> tuple:
        """
        查询任意世界坐标处的 GI 辐照度。
        先查辐射度缓存，缓存未命中则返回零——
        鲁迅式：没有积累便没有输出，零不是失败，是尚未开始。
        """
        return self.radiance_cache.interpolate(world_pos, normal, viewer_pos)

    def frame_stats(self) -> dict:
        """返回上一帧统计——数字是诚实的，即便它揭示的真相令人不快。"""
        return dict(self._frame_stats)


# -----------------------------------------------------------------------------
# 模块级单例（可选）——全局共享同一管线实例，对应 LumenSceneData 的单场景设计
# -----------------------------------------------------------------------------
_ASTRO_GI_PIPELINE: Optional[AstroCellGlobalIlluminationPipeline] = None


def get_astro_gi_pipeline() -> AstroCellGlobalIlluminationPipeline:
    """
    获取全局 GI 管线单例。
    鲁迅式：单例如皇权，全场景只此一份——权力集中，效率更高，
    但若它出错，便是全局灾难。
    """
    global _ASTRO_GI_PIPELINE
    if _ASTRO_GI_PIPELINE is None:
        _ASTRO_GI_PIPELINE = AstroCellGlobalIlluminationPipeline()
    return _ASTRO_GI_PIPELINE
