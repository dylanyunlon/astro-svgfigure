import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def use_distance_field_ao() -> bool:
    """r.DistanceFieldAO && r.AOQuality >= 1。两个条件缺一不可，像人一样。"""
    return G_DISTANCE_FIELD_AO and G_DISTANCE_FIELD_AO_QUALITY >= 1





def use_ao_object_distance_field() -> bool:
    return G_AO_OBJECT_DISTANCE_FIELD and G_DISTANCE_FIELD_AO_QUALITY >= 2





# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------

def update_distance_field_object_buffers(
    scene_data: DFSceneData,
    objects_to_add: List[DFObjectBounds],
    objects_to_remove: List[int],
    surface_bias_expand: float = G_MESH_SDF_SURFACE_BIAS_EXPAND,
    parallel: bool = G_DF_PARALLEL_UPDATE,
) -> None:
    """
    对应 UpdateDistanceFieldObjectBuffers。
    先删再加，顺序不能错：若先加再删，可能误删新加的对象。
    世界上有些事情也讲究顺序，颠倒了就是另一个故事。
    """
    for idx in objects_to_remove:
        scene_data.remove_object(idx)

    # 可并行（G_DF_PARALLEL_UPDATE），此处简化为串行
    for obj in objects_to_add:
        # 表面偏移：膨胀一个 voxel 的 surface_bias_expand 比例
        expanded_radius = obj.radius * (1.0 + surface_bias_expand)
        expanded = DFObjectBounds(
            center=obj.center,
            radius=expanded_radius,
            object_index=obj.object_index,
        )
        scene_data.add_object(expanded)





def _sample_distance_field(
    obj: DFObjectBounds,
    ray_origin: Tuple[float, float, float],
    ray_dir: Tuple[float, float, float],
    t: float,
) -> float:
    """
    SDF 球体采样的极简近似。
    真正的实现要查 Atlas brick，这里用解析球替代。
    """
    px = ray_origin[0] + ray_dir[0] * t - obj.center[0]
    py = ray_origin[1] + ray_dir[1] * t - obj.center[1]
    pz = ray_origin[2] + ray_dir[2] * t - obj.center[2]
    return math.sqrt(px*px + py*py + pz*pz) - obj.radius





# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------

def render_distance_field_ao(
    scene_data: DFSceneData,
    pixel_positions: List[Tuple[float, float, float]],
    pixel_normals: List[Tuple[float, float, float]],
    pixel_depths: List[float],
    view_width: int,
    view_height: int,
    frustum_planes: List[Tuple[float, float, float, float]],
    view_origin: Tuple[float, float, float],
    ao_params: DFAOParameters,
    history: Optional[AOHistoryState] = None,
    frame_index: int = 0,
) -> Tuple[List[BentNormalAO], AOHistoryState]:
    """
    完整的 Distance Field AO 帧调度，对应 UE5 RenderDistanceFieldAO。

    流程：ObjectCulling → TileCulling → ScreenGridConeTrace → LightingPost。
    每一步都是上一步的筛选；最终到达屏幕的光，是经过了很多关卡的光。
    不经审查的光，称为噪点。
    """
    if not use_distance_field_ao():
        empty = [BentNormalAO() for _ in pixel_positions]
        return empty, AOHistoryState()

    # 1. 对象剔除
    culled = cull_objects_to_view(scene_data, frustum_planes, ao_params, view_origin)

    # 2. Tile cones
    tiles = build_tile_cones(view_width, view_height)

    # 3. Tile-object 交叉（scatter culling）
    tile_intersections = scatter_tile_culling(culled, tiles, scene_data)

    # 4. 收集存活的 SDF 对象
    surviving_indices = set(culled.object_indices)
    surviving_objects = [o for o in scene_data.objects if o.object_index in surviving_indices]

    # 5. Screen-grid cone trace（低分辨率）
    ao_size = get_buffer_size_for_ao(view_width, view_height)
    # 为简化，假设 pixel_positions/normals 已是低分辨率
    ao_low = compute_screen_grid_ao(
        pixel_positions=pixel_positions,
        pixel_normals=pixel_normals,
        objects=surviving_objects,
        ao_params=ao_params,
        frame_number=frame_index,
        use_history=(history is not None and history.valid),
    )

    # 6. 历史融合（depth rejection）
    hist_depths = ([b.occlusion for b in history.bent_normal_history]
                   if history and history.valid else [])
    ao_blended = update_history_depth_rejection(
        current=ao_low,
        history=history or AOHistoryState(),
        current_depths=pixel_depths,
        history_depths=hist_depths,
    )

    # 7. 空间稳定滤波
    ao_filtered = filter_history_stability(ao_blended, ao_size.x, ao_size.y)

    # 8. 上采样到全分辨率（简化：跳过，直接返回低分辨率结果）
    # geometry_aware_upsample(...) 可在此调用

    # 9. 更新历史
    new_history = update_ao_history(ao_filtered, ao_filtered, frame_index)

    return ao_filtered, new_history
# ============================================================
# Nanite CullRaster + Editor + RayTracing + Materials + Tessellation
# Ported from UE5 upstream — 鲁迅式注释穿插其中
# "世上本没有路，走的人多了，也便成了管线。"
# ============================================================

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Tuple, Any
import math


# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。

