import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def _mark_stencil_cells(
    cell_entries: list[dict],
    vis_set: set[str],
) -> list[dict]:
    """标记模板层 — 镜像 FNaniteMarkStencilPS。

    遍历 *cell_entries*，为每个 cell_id 出现在 *vis_set*（可见集合，等价于
    VisBuffer64 中持有有效 Nanite 图元的样本）的条目打上 ``stencil=1`` 标记。
    不可见的条目保留 ``stencil=0``，供后续 :func:`_emit_scene_stencil` 裁剪。

    Parameters
    ----------
    cell_entries:
        每个元素是一个包含 ``cell_id`` 与 ``bbox`` 的字典，来自 cell_registry。
    vis_set:
        本帧可见的 cell_id 集合（由 :class:`AstroCellVisibilityQuery` 生成）。

    Returns
    -------
    list[dict]
        与输入等长，每个条目追加了 ``"stencil"`` 键（0 或 1）。
    """
    result = []
    for entry in cell_entries:
        marked = dict(entry)
        marked["stencil"] = 1 if entry["cell_id"] in vis_set else 0
        result.append(marked)
    return result





def _emit_scene_stencil(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """提升着色掩码为模板纹理 — 镜像 FEmitSceneStencilPS。

    读取每个 cell 条目中的 ``shading_mask`` 字段（由可见性查询写入），将置位
    条目的 cell_id 追加到 *depth_manifest* 的 ``"stencil_cells"`` 集合。
    后续 SVG 渲染器可据此将对应元素置入 ``<clipPath>`` 分组。
    """
    stencil_cells: list[str] = depth_manifest.setdefault("stencil_cells", [])
    for entry in cell_entries:
        if entry.get("shading_mask", 0):
            cid = entry["cell_id"]
            if cid not in stencil_cells:
                stencil_cells.append(cid)
    return depth_manifest





def _emit_custom_depth_stencil(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """写入自定义深度/模板 — 镜像 FEmitCustomDepthStencilPS。

    检测 ``species_params`` 中携带 ``"highlight"`` 键的 cell，将其 cell_id
    与 highlight 颜色值写入 *depth_manifest* 的 ``"custom_depth"`` 子字典。
    SVG 渲染器利用此信息在对应元素上叠加 CSS ``stroke-width`` 高亮描边。

    若 :data:`glyph_custom_depth_export_method` 为 1（CS 路径），则通过
    批量字典推导式一次性构造映射；为 0 时退化为逐元素循环（PS 路径）。
    """
    if glyph_custom_depth_export_method == 1:
        # CS 路径 — 批量向量化（镜像 FDepthExportCS dispatch）
        custom: dict[str, str] = {
            e["cell_id"]: e["species_params"]["highlight"]
            for e in cell_entries
            if e.get("species_params", {}).get("highlight")
        }
    else:
        # PS 路径 — 逐元素（镜像 FEmitCustomDepthStencilPS pixel walk）
        custom = {}
        for e in cell_entries:
            hl = e.get("species_params", {}).get("highlight")
            if hl:
                custom[e["cell_id"]] = hl

    depth_manifest.setdefault("custom_depth", {}).update(custom)
    return depth_manifest





def build_epoch_draw_list(
    visibility_query: "AstroCellVisibilityQuery | None" = None,
) -> AstroCellDrawList:
    """
    Build a per-epoch AstroCellDrawList from the current cell_registry.

    Mirrors the scene-level loop in FNaniteMaterialListContext::Apply() that
    iterates DeferredPipelines and calls AddRasterBin / AddShadingBin for each
    registered primitive.  Here we iterate cell_registry.json entries and call
    register_cell_draw_entry for each visible cell.

    If a visibility_query is provided (output of perform_nanite_visibility()),
    culled cells (lod == -1) are excluded from the draw list — matching Nanite's
    behaviour of skipping invisible primitives in the draw-command submission
    loop.

    @param visibility_query  Optional AstroCellVisibilityQuery; pass None to
                             include all registered cells (no culling).
    @return                  AstroCellDrawList ready for flush_draw_order().
    """
    draw_list = AstroCellDrawList()
    registry  = _load_cell_registry()
    cells     = registry.get("cells", {})

    for cell_id, entry in cells.items():
        # ── Visibility gate (FNaniteVisibility bin check) ─────────────────────
        if visibility_query is not None:
            cell_result = visibility_query.cell_results.get(cell_id, {})
            if cell_result.get("lod", 0) == -1:
                continue  # culled — skip (mirrors IsNanitePrimitiveVisible gate)

        # ── Reconstruct bbox from registry min/max format ─────────────────────
        bbox_data = entry.get("bbox", {})
        if "min" in bbox_data and "max" in bbox_data:
            mn = bbox_data["min"]
            mx = bbox_data["max"]
            bbox = {
                "x": mn[0], "y": mn[1],
                "w": mx[0] - mn[0], "h": mx[1] - mn[1],
                "z": mn[2] if len(mn) > 2 else 0,
            }
        else:
            bbox = bbox_data

        species = entry.get("species", "")
        z_layer = entry.get("z", 3)
        epoch   = entry.get("epoch", 0)

        draw_list.register_cell_draw_entry(
            cell_id=cell_id,
            z_layer=z_layer,
            species=species,
            bbox=bbox,
            extra={"epoch": epoch, "constraint_mask": entry.get("constraint_mask", 0)},
        )

    return draw_list


# [ASTRO-CELL] DynamicBVH → Python port
#
# Ported from commit upstream/unreal-renderer-ue5/Renderer-Private/DynamicBVH.h
#   and DynamicBVH.cpp (Epic Games, Inc.)
#
# AstroCellBVH:
#   2-D AABB tree accelerating overlap queries for up to 500 cells.
#   Replaces O(N²) pairwise scan with O(log N) tree traversal.
#
#   FDynamicBVH<MaxChildren=4>  →  AstroCellBVH (MaxChildren fixed at 4)
#   FBounds3f (XYZ)             →  _AABB2 (X,Y only; Z/rotation dropped)
#   Insert / Remove / Update    →  insert_cell / remove_cell / update_cell_bounds
#   ForAll overlap              →  query_overlapping_cells
#
# Split algorithm: Morton-code greedy split (simplified to 2-D interleaved bits,
#   matching FMortonArray::Split logic) drives the batch Build path.
# Single-insertion uses the Greedy best-insertion heuristic
#   (FindBestInsertion_Greedy) with Surface-Area Heuristic cost metric adapted
#   to 2-D perimeter (half-perimeter ≈ SAH for flat scenes).
# Refit propagates tight AABB up the ancestor chain after every structural
#   change (mirrors the PathBounds loop in Insert/Extract).
# ═══════════════════════════════════════════════════════════════════════════════

import math as _math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal geometry primitive
# ---------------------------------------------------------------------------




def compute_prev_frame_velocity(
    cell_id: str,
    cell_z:  float,
    cell_registry_prev: dict,
) -> float:
    """
    Compute z-axis velocity (dz per epoch) for a cell.

    Mirrors GetPreviousWorldToClipMatrix + the velocity buffer computation
    that encodes how far each primitive moved since the last frame.

    In the Astro 2-D model, z-layer is the primary motion axis (cells can
    move between z-layers between epochs).  X/Y motion is constrained by
    the force-field and is typically small.

    Returns signed dz (z_current - z_previous), 0.0 if no history.

    鲁迅式：速度是位移对时间的导数——
    没有过去就没有速度，有了过去才能预测未来。
    """
    prev_cells = cell_registry_prev.get("cells", {})
    prev_entry = prev_cells.get(cell_id)
    if prev_entry is None:
        return 0.0
    prev_z = float(prev_entry.get("z", cell_z))
    return cell_z - prev_z





class AstroCellVelocityFlattenData:
    """
    Python equivalent of FVelocityFlattenTextures.

    Stores per-cell velocity (dz) and per-tile max velocity used for
    the motion blur scatter/gather decision.

    鲁迅式：速度展平是民调——
    把每个像素的速度汇总为瓦片的最大速度，
    从而决定这一瓦片该用哪种模糊策略。
    """

    def __init__(self) -> None:
        # Per-cell velocity: cell_id → float dz
        self.cell_velocities:  dict[str, float] = {}
        # Per-tile max velocity: (tx, ty) → float max_abs_dz
        self.tile_max_velocity: dict[tuple, float] = {}
        # Global max velocity across all cells
        self.global_max_velocity: float = 0.0

    def build(self, cell_registry: dict,
              cell_registry_prev: dict,
              viewport_w: int = 1200,
              viewport_h: int = 900) -> None:
        """
        Populate velocity data for all cells and build tile max buffer.
        Mirrors the VelocityFlatten pass dispatch.
        """
        cells = cell_registry.get("cells", {})
        tile_grid: dict[tuple, list[float]] = {}

        for cell_id, entry in cells.items():
            z   = float(entry.get("z", 3))
            vel = compute_prev_frame_velocity(cell_id, z, cell_registry_prev)
            self.cell_velocities[cell_id] = vel

            # Assign to tile based on cell centre
            bbox_data = entry.get("bbox", {})
            if "min" in bbox_data and "max" in bbox_data:
                mn, mx  = bbox_data["min"], bbox_data["max"]
                cx = (mn[0] + mx[0]) / 2.0
                cy = (mn[1] + mx[1]) / 2.0
            else:
                cx = bbox_data.get("x", 0) + bbox_data.get("w", 80) / 2.0
                cy = bbox_data.get("y", 0) + bbox_data.get("h", 50) / 2.0

            tx = int(cx) // _MB_FLATTEN_TILE_SIZE
            ty = int(cy) // _MB_FLATTEN_TILE_SIZE
            tile_grid.setdefault((tx, ty), []).append(abs(vel))

        # Tile max velocity (mirrors TileMaxVelocity texture)
        for tile_key, vels in tile_grid.items():
            self.tile_max_velocity[tile_key] = max(vels)

        self.global_max_velocity = (
            max(abs(v) for v in self.cell_velocities.values())
            if self.cell_velocities else 0.0
        )





class EDecalRenderStage(IntEnum):
    """渲染阶段枚举，对应 UE5 EDecalRenderStage。"""
    None_      = 0   # 鲁迅式：无阶段，即虚无——渲染管线最诚实的答案。
    BeforeBasePass     = 1
    BeforeLighting     = 2
    Mobile             = 3
    MobileBeforeLighting = 4
    Emissive           = 5
    AmbientOcclusion   = 6


@dataclass



@dataclass
class FDecalBlendDesc:
    """
    贴花混合描述符。
    鲁迅式：写下每一个 bWrite 字段，便是承认画面还缺少什么。
    """
    blend_mode: EBlendMode = EBlendMode.Translucent

    b_write_base_color:               bool = False
    b_write_normal:                   bool = False
    b_write_roughness_specular_metallic: bool = False
    b_write_emissive:                 bool = False
    b_write_ambient_occlusion:        bool = False
    b_write_dbuffer_mask:             bool = False

    # bitmask over EDecalRenderStage values
    render_stage_mask: int = 0





def finalize_decal_blend_desc(
    desc: FDecalBlendDesc,
    *,
    is_mobile: bool = False,
    is_mobile_deferred: bool = False,
    is_dbuffer_platform: bool = False,
    is_dbuffer_mask_platform: bool = False,
    mobile_normal_blendable: bool = False,
) -> FDecalBlendDesc:
    """
    推导贴花混合描述符的最终状态，对应 FinalizeBlendDesc()。
    鲁迅式：平台限制就是现实，我们只能在允许的范围内做梦。
    """
    desc.b_write_dbuffer_mask = is_dbuffer_mask_platform

    # 强制平台混合模式约束
    if not (_is_translucent_only(desc) or _is_alpha_composite(desc) or _is_modulate(desc)):
        desc.blend_mode = EBlendMode.Translucent
    if is_dbuffer_platform and _is_modulate(desc):
        desc.blend_mode = EBlendMode.Translucent

    # 移动平台输出约束
    if is_mobile and not is_mobile_deferred and not is_dbuffer_platform:
        desc.b_write_roughness_specular_metallic = False
    if is_mobile and not mobile_normal_blendable:
        desc.b_write_normal = False
    if is_mobile:
        desc.b_write_ambient_occlusion = False

    # AlphaComposite 不写法线
    if _is_alpha_composite(desc):
        desc.b_write_normal = False

    # 推导主渲染阶段（互斥）
    has_gbuffer_writes = (
        desc.b_write_emissive or desc.b_write_base_color
        or desc.b_write_normal or desc.b_write_roughness_specular_metallic
    )
    if is_mobile_deferred and not is_dbuffer_platform and has_gbuffer_writes:
        desc.render_stage_mask |= 1 << EDecalRenderStage.MobileBeforeLighting
    elif is_mobile and not is_dbuffer_platform and (desc.b_write_emissive or desc.b_write_base_color):
        desc.render_stage_mask |= 1 << EDecalRenderStage.Mobile
    elif is_dbuffer_platform and (
        desc.b_write_base_color or desc.b_write_normal or desc.b_write_roughness_specular_metallic
    ):
        desc.render_stage_mask |= 1 << EDecalRenderStage.BeforeBasePass
    elif has_gbuffer_writes:
        desc.render_stage_mask |= 1 << EDecalRenderStage.BeforeLighting

    # 附加渲染阶段
    if desc.b_write_emissive and is_dbuffer_platform:
        desc.render_stage_mask |= 1 << EDecalRenderStage.Emissive
    if desc.b_write_ambient_occlusion:
        desc.render_stage_mask |= 1 << EDecalRenderStage.AmbientOcclusion

    return desc





def compute_decal_blend_desc(
    material_props: dict,
    *,
    is_mobile: bool = False,
    is_mobile_deferred: bool = False,
    is_dbuffer_platform: bool = False,
    is_dbuffer_mask_platform: bool = False,
) -> FDecalBlendDesc:
    """
    根据材质属性字典构造 FDecalBlendDesc，对应 ComputeDecalBlendDesc()。
    material_props 键：blend_mode, base_color, normal, roughness, specular, metallic,
                       emissive, ambient_occlusion, is_substrate, diffuse_albedo, f0
    鲁迅式：材质连接了什么，就说明创作者在意什么；不连接的，便是刻意的遗忘。
    """
    desc = FDecalBlendDesc()
    is_substrate = material_props.get("is_substrate", False)
    use_diffuse_f0 = material_props.get("diffuse_albedo", False) or material_props.get("f0", False)

    desc.blend_mode = EBlendMode(material_props.get("blend_mode", EBlendMode.Translucent))

    if is_substrate:
        desc.b_write_base_color               = material_props.get("base_color", False) or use_diffuse_f0
        desc.b_write_normal                   = material_props.get("normal", False)
        desc.b_write_roughness_specular_metallic = (
            use_diffuse_f0
            or material_props.get("roughness", False)
            or material_props.get("specular", False)
            or material_props.get("metallic", False)
        )
        desc.b_write_emissive             = material_props.get("emissive", False)
        desc.b_write_ambient_occlusion    = material_props.get("ambient_occlusion", False)
    else:
        desc.b_write_base_color               = material_props.get("base_color", False)
        desc.b_write_normal                   = material_props.get("normal", False)
        desc.b_write_roughness_specular_metallic = (
            material_props.get("roughness", False)
            or material_props.get("specular", False)
            or material_props.get("metallic", False)
        )
        desc.b_write_emissive             = material_props.get("emissive", False)
        desc.b_write_ambient_occlusion    = material_props.get("ambient_occlusion", False)

    mobile_normal_blendable = is_dbuffer_platform or is_mobile_deferred
    return finalize_decal_blend_desc(
        desc,
        is_mobile=is_mobile,
        is_mobile_deferred=is_mobile_deferred,
        is_dbuffer_platform=is_dbuffer_platform,
        is_dbuffer_mask_platform=is_dbuffer_mask_platform,
        mobile_normal_blendable=mobile_normal_blendable,
    )





@dataclass
class FVisibleDecal:
    """
    可见贴花实体，携带混合描述符与淡入淡出参数。
    鲁迅式：贴花的淡出，不是消亡，而是体面地退场。
    """
    material_props:   dict
    sort_order:       int   = 0
    conservative_radius: float = 1.0
    fade_alpha:       float = 1.0
    inv_fade_duration:  float = 0.0
    inv_fade_in_duration: float = 0.0
    fade_start_delay: float = 0.0
    fade_in_start_delay: float = 0.0
    blend_desc: FDecalBlendDesc = field(default_factory=FDecalBlendDesc)

    def __post_init__(self):
        self.blend_desc = compute_decal_blend_desc(self.material_props)





def build_visible_decal_list(
    all_decals: list[dict],
    view_frustum_test: Callable[[dict], tuple[bool, float, float]] | None = None,
) -> list[FVisibleDecal]:
    """
    从场景贴花列表中筛选视锥体内的可见贴花，对应 BuildVisibleDecalList()。
    view_frustum_test(decal) -> (is_visible, conservative_radius, fade_alpha)
    鲁迅式：视锥体剔除是一种慈悲——让看不见的东西不必假装存在。
    """
    result: list[FVisibleDecal] = []
    for decal in all_decals:
        if view_frustum_test is not None:
            visible, radius, alpha = view_frustum_test(decal)
        else:
            visible, radius, alpha = True, decal.get("radius", 1.0), 1.0
        if not visible:
            continue
        vd = FVisibleDecal(
            material_props=decal.get("material_props", {}),
            sort_order=decal.get("sort_order", 0),
            conservative_radius=radius,
            fade_alpha=alpha,
            inv_fade_duration=decal.get("inv_fade_duration", 0.0),
            inv_fade_in_duration=decal.get("inv_fade_in_duration", 0.0),
            fade_start_delay=decal.get("fade_start_delay", 0.0),
            fade_in_start_delay=decal.get("fade_in_start_delay", 0.0),
        )
        result.append(vd)
    # 按 sort_order 升序，再按 conservative_radius 降序（大贴花先渲染）
    result.sort(key=lambda d: (d.sort_order, -d.conservative_radius))
    return result





def build_relevant_decal_list(
    visible_decals: list[FVisibleDecal],
    stage: EDecalRenderStage,
) -> list[FVisibleDecal]:
    """
    从可见列表中过滤出与指定渲染阶段兼容的贴花，对应 BuildRelevantDecalList()。
    鲁迅式：每个阶段只看自己的份，这叫专注，也叫局限。
    """
    return [d for d in visible_decals if is_compatible_with_render_stage(d.blend_desc, stage)]


@dataclass



@dataclass
class DecalVisibilityViewPacket:
    """
    单视图的贴花可见性数据包，对应 FDecalVisibilityViewPacket。
    懒惰求值：relevant_decals 按需构建。
    鲁迅式：按需构建是现代工程师的美德，也是他们唯一剩下的节制。
    """
    visible_decals: list[FVisibleDecal] = field(default_factory=list)
    _relevant_cache: dict[EDecalRenderStage, list[FVisibleDecal]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def build(
        cls,
        all_decals: list[dict],
        stages: Iterable[EDecalRenderStage] = (),
        view_frustum_test: Callable | None = None,
    ) -> "DecalVisibilityViewPacket":
        pkt = cls()
        pkt.visible_decals = build_visible_decal_list(all_decals, view_frustum_test)
        for stage in stages:
            pkt._relevant_cache[stage] = build_relevant_decal_list(pkt.visible_decals, stage)
        return pkt

    def get_relevant_decals(self, stage: EDecalRenderStage) -> list[FVisibleDecal]:
        if stage not in self._relevant_cache:
            self._relevant_cache[stage] = build_relevant_decal_list(self.visible_decals, stage)
        return self._relevant_cache[stage]


# =============================================================================
# § BlueNoise — 蓝噪声参数管理
#   移植自 Renderer-Private/BlueNoise.cpp
#   鲁迅式：蓝噪声的均匀，是对随机性的最后一点人道主义改造。
# =============================================================================

import numpy as np
from typing import Optional


@dataclass



def _fill_blue_noise_from_texture(out: FBlueNoiseParameters) -> None:
    """
    从 scalar_texture 形状推导 dimensions 与 modulo_masks，
    对应 FillUpBlueNoiseParametersFromTexture()。
    鲁迅式：从尺寸推导掩码，是数学对现实的一次简洁起诉。
    """
    assert out.scalar_texture is not None, "scalar_texture must be set before fill"
    h, w = out.scalar_texture.shape[:2]
    slices = h // max(1, w)
    out.dimensions = (w, w, slices)
    out.modulo_masks = (
        (1 << _floor_log2(out.dimensions[0])) - 1,
        (1 << _floor_log2(out.dimensions[1])) - 1,
        (1 << _floor_log2(out.dimensions[2])) - 1,
    )
    assert (out.modulo_masks[0] + 1) == out.dimensions[0], "dimension X must be power-of-two"
    assert (out.modulo_masks[1] + 1) == out.dimensions[1], "dimension Y must be power-of-two"
    assert (out.modulo_masks[2] + 1) == out.dimensions[2], "dimension Z must be power-of-two"


_BLACK_DUMMY = np.zeros((1, 1), dtype=np.float32)





def get_blue_noise_dummy_parameters() -> FBlueNoiseParameters:
    """
    返回全黑占位蓝噪声参数，对应 GetBlueNoiseDummyParameters()。
    鲁迅式：占位符是工程师的礼貌，也是他承认自己尚未准备好的方式。
    """
    return FBlueNoiseParameters(
        dimensions=(1, 1, 1),
        modulo_masks=(0, 0, 0),
        scalar_texture=_BLACK_DUMMY,
        vec2_texture=_BLACK_DUMMY,
    )





def get_blue_noise_parameters(
    scalar_texture: np.ndarray,
    vec2_texture: Optional[np.ndarray] = None,
) -> FBlueNoiseParameters:
    """
    从真实纹理数据构造蓝噪声参数，对应 GetBlueNoiseParameters()。
    scalar_texture: H×W float32 array（H = W * slices）
    vec2_texture:   可选，H×W×2 float32 array
    鲁迅式：真实纹理与占位纹理，差别只在于有没有认真对待随机这件事。
    """
    out = FBlueNoiseParameters(
        scalar_texture=scalar_texture,
        vec2_texture=vec2_texture if vec2_texture is not None else _BLACK_DUMMY,
    )
    _fill_blue_noise_from_texture(out)
    return out





def get_blue_noise_global_parameters(
    scalar_texture: Optional[np.ndarray] = None,
    vec2_texture: Optional[np.ndarray] = None,
) -> FBlueNoiseParameters:
    """
    获取全局蓝噪声参数，无纹理时回退到占位符，对应 GetBlueNoiseGlobalParameters()。
    鲁迅式：全局参数的回退路径，是系统对不完整世界的默默容忍。
    """
    if scalar_texture is not None:
        return get_blue_noise_parameters(scalar_texture, vec2_texture)
    return get_blue_noise_dummy_parameters()





def sample_blue_noise_scalar(params: FBlueNoiseParameters, x: int, y: int, z: int = 0) -> float:
    """
    从蓝噪声参数中采样标量值，使用 modulo_masks 进行快速环绕寻址。
    鲁迅式：环绕寻址像极了历史——越界之后，总会回到某个熟悉的起点。
    """
    if params.scalar_texture is None or params.scalar_texture is _BLACK_DUMMY:
        return 0.0
    mx, my, mz = params.modulo_masks
    xi = x & mx
    yi = y & my
    zi = z & mz
    h, w = params.scalar_texture.shape[:2]
    slices = params.dimensions[2]
    row = zi * (h // max(1, slices)) + yi
    row = min(row, h - 1)
    col = min(xi, w - 1)
    return float(params.scalar_texture[row, col])


# =============================================================================
# § ComputeSystemInterface — 计算系统注册与 Worker 生命周期
#   移植自 Renderer-Private/ComputeSystemInterface.cpp
#   鲁迅式：注册表是官僚体系的技术化身——你必须登记，才有资格被调用。
# =============================================================================

from typing import Protocol, runtime_checkable


@runtime_checkable



# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。

def astro_anisotropy_brdf(
    n: _Vec3,
    v: _Vec3,
    l: _Vec3,
    t: _Vec3,
    roughness_u: float,
    roughness_v: float,
) -> float:
    """
    Ward-Dür anisotropic BRDF lobe (simplified).

    Used to weight the specular contribution of a cell whose material
    has anisotropy > 0.  Mirrors the HLSL in AnisotropyPassShader.usf,
    which is invoked by FAnisotropyMeshProcessor::Process().

    Parameters
    ----------
    n           : surface normal     (world space, unit)
    v           : view direction     (unit, toward camera)
    l           : light direction    (unit, toward light)
    t           : tangent direction  (unit)
    roughness_u : GGX roughness along tangent    [0, 1]
    roughness_v : GGX roughness along bitangent  [0, 1]
    """
    # Clamp roughness to avoid division by zero (same guard as UE5 HLSL)
    au = max(roughness_u * roughness_u, 1e-4)
    av = max(roughness_v * roughness_v, 1e-4)

    # Half-vector
    hx, hy, hz = v.x+l.x, v.y+l.y, v.z+l.z
    h_len = _math.sqrt(hx*hx + hy*hy + hz*hz)
    if h_len < 1e-9:
        return 0.0
    h = _Vec3(hx/h_len, hy/h_len, hz/h_len)

    # Bitangent
    b = _Vec3(
        n.y*t.z - n.z*t.y,
        n.z*t.x - n.x*t.z,
        n.x*t.y - n.y*t.x,
    ).normalize()

    ndotl = _saturate(n.dot(l))
    ndotv = _saturate(n.dot(v))
    ndoth = _saturate(n.dot(h))
    hdott = h.dot(t)
    hdotb = h.dot(b)

    if ndotl < 1e-6 or ndotv < 1e-6:
        return 0.0

    # GGX anisotropic NDF  (Burley 2012 / UE5 BRDF.ush)
    denom = (hdott/au)**2 + (hdotb/av)**2 + ndoth**2
    d = 1.0 / (_math.pi * au * av * denom**2 + 1e-9)

    # Smith visibility (approximate — same as UE5 optimised form)
    lv = ndotl * _math.sqrt(ndotv**2 * (1.0 - av**2) + av**2)
    ll = ndotv * _math.sqrt(ndotl**2 * (1.0 - au**2) + au**2)
    g  = 0.5 / (lv + ll + 1e-9)

    return d * g * ndotl


# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。

