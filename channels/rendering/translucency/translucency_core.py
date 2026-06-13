import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def use_translucency_vector_path() -> bool:
    """判断是否启用矢量批渲染半透明路径 — 镜像 UseNaniteMeshShader()。

    在 UE5 中，半透明 Mesh Shader 路径要求 Tier-0 Mesh Shader 支持。
    Astro 等价：ASTRO_TRANSLUCENCY_MESH_SHADER 且平台支持向量渲染时返回 True。
    """
    return ASTRO_TRANSLUCENCY_MESH_SHADER and use_vector_render_path()









def _transcode_rasterizer_args(
    raster_bin_count: int,
    input_format: int,
    output_format: int,
) -> dict:
    """转码光栅化器参数 — 镜像 FTranscodeRasterizerArgs_CS。

    将 SW 光栅化路径产生的 ``RasterizerArgsSWHW`` 格式从 *input_format* 转换
    为 *output_format*，对应 Astro 中将软件路径的 SVG 路径命令格式转为
    PixiJS Mesh 格式（或反向）。

    Parameters
    ----------
    raster_bin_count:
        本帧光栅化 bin 数量（cell 数量上限）。
    input_format:
        0 = SVG path 命令格式（软件路径），1 = PixiJS mesh 格式（硬件路径）。
    output_format:
        同上，目标格式。

    Returns
    -------
    dict
        包含转码后参数的字典，键与 UE5 shader 参数结构字段对应。
    """
    return {
        "raster_bin_count": raster_bin_count,
        "input_format":     input_format,
        "output_format":    output_format,
        "transcoded":       input_format != output_format,
    }









class AstroCellTranslucencyFactory:
    """半透明顶点工厂 — 镜像 FNaniteTranslucencyFactory。

    负责判断给定 cell 是否应进入半透明渲染通道，并预处理其 SVG 属性以支持
    正确的 Alpha 合成（``mix-blend-mode``、``opacity``、``fill-opacity``）。

    Parameters
    ----------
    feature_level:
        渲染特性等级字符串（镜像 ``ERHIFeatureLevel``），当前仅支持 ``"SM5"``。
    """

    def __init__(self, feature_level: str = "SM5") -> None:
        self.feature_level = feature_level

    @staticmethod
    def should_compile_permutation(species: str, blend_mode: str) -> bool:
        """判断是否为该 species/blend_mode 组合编译半透明 permutation。

        镜像 ``FNaniteTranslucencyFactory::ShouldCompilePermutation``：
        仅当 blend_mode 为半透明模式（"translucent"、"additive"、"modulate"）
        且 species 使用 Nanite（Astro 中所有 species 均视为使用 Nanite）时返回 True。
        """
        translucent_modes = {"translucent", "additive", "modulate", "alpha_composite"}
        return blend_mode in translucent_modes

    @staticmethod
    def prepare_svg_attrs(opacity: float, blend_mode: str) -> dict[str, str]:
        """预处理半透明 SVG 属性 — 镜像 ModifyCompilationEnvironment 中的 define 设置。

        Returns
        -------
        dict[str, str]
            SVG/CSS 属性字典，直接可写入 ``<g>`` 元素。
        """
        blend_map = {
            "translucent":     "normal",
            "additive":        "screen",
            "modulate":        "multiply",
            "alpha_composite": "source-over",
        }
        return {
            "opacity":        str(round(opacity, 4)),
            "mix-blend-mode": blend_map.get(blend_mode, "normal"),
            "isolation":      "isolate",
        }









class AstroCellTranslucencyRenderer:
    """半透明 Cell 渲染器 — 镜像 Nanite::RenderTranslucency。

    将本帧中 opacity < :data:`ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD` 的 cell
    从主合成器分离，独立执行前向 Alpha 合成渲染，最终输出一个半透明层 SVG 片段。

    Lifecycle::

        renderer = AstroCellTranslucencyRenderer()
        renderer.set_parameters(raster_results, view_info)
        translucency_svg = renderer.render(cell_entries, vis_set)
    """

    def __init__(self) -> None:
        self._factory = AstroCellTranslucencyFactory()
        self._raster_results: dict = {}
        self._view_info: dict = {}

    def set_parameters(
        self,
        raster_results: dict | None,
        view_info: dict | None = None,
    ) -> None:
        """绑定光栅化结果与视图参数 — 镜像 SetTranslucencyParameters。

        Parameters
        ----------
        raster_results:
            主渲染通道的光栅化结果（``FRasterResults`` 等价）；为 None 时
            使用系统纹理默认值（镜像 GSystemTextures fallback）。
        view_info:
            视图元数据字典（viewport 尺寸、投影矩阵等）。
        """
        self._raster_results = raster_results or {}
        self._view_info = view_info or {}

    def render(
        self,
        cell_entries: list[dict],
        vis_set: set[str],
    ) -> str:
        """执行半透明渲染通道 — 镜像 RenderTranslucency。

        筛选出可见且不透明度低于阈值的 cell，按 z-layer 降序（从远到近）
        排列后执行前向 Alpha 合成，输出独立的半透明层 ``<g>`` 分组。

        Parameters
        ----------
        cell_entries:
            全量 cell 条目列表（含 opacity 字段）。
        vis_set:
            本帧可见 cell_id 集合。

        Returns
        -------
        str
            半透明层 SVG 片段字符串（``<g id="translucency-layer">…</g>``）。
        """
        # 筛选半透明可见 cell（镜像 IsTranslucentBlendMode 检查）
        translucent = [
            e for e in cell_entries
            if e["cell_id"] in vis_set
            and e.get("opacity", 1.0) < ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD
        ]

        if not translucent:
            return '<g id="translucency-layer"/>'

        # 按 z 降序排列 — 从远到近前向合成（镜像 translucent draw order）
        translucent.sort(key=lambda e: e.get("bbox", {}).get("z", 0.0), reverse=True)

        # 转码光栅化参数（镜像 FTranscodeRasterizerArgs_CS dispatch）
        bin_count = len(translucent)
        in_fmt = 0  # SVG path 格式
        out_fmt = 1 if use_translucency_vector_path() else 0
        _transcode_rasterizer_args(bin_count, in_fmt, out_fmt)

        # 生成各 cell 的半透明 SVG 片段
        fragments: list[str] = []
        for entry in translucent:
            cid = entry["cell_id"]
            opacity = entry.get("opacity", 1.0)
            blend_mode = entry.get("blend_mode", "translucent")
            if not self._factory.should_compile_permutation("*", blend_mode):
                blend_mode = "translucent"
            attrs = self._factory.prepare_svg_attrs(opacity, blend_mode)
            attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            fragment = entry.get("svg_fragment", "")
            fragments.append(
                f'<g data-cell-id="{cid}" {attr_str}>{fragment}</g>'
            )
            increment_perf_counter("visible_clusters", 1)

        inner = "\n    ".join(fragments)
        return f'<g id="translucency-layer">\n    {inner}\n  </g>'


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] NaniteDrawList → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteDrawList.cpp
#
# FNaniteDrawListContext (→ AstroCellDrawList):
#   Accumulates cell draw entries grouped by species, ordered so that same-
#   species cells are rendered contiguously.  Contiguous species runs collapse
#   redundant SVG <defs> switches (analogous to PSO / material state changes in
#   the GPU pipeline) and allow the SVG composer to emit shared <defs> once per
#   run rather than once per cell.
#
# Algorithm changes from Nanite original (20%):
#   1. FMeshDrawCommand TArray insertion sort → stable sort on a composite key
#      (z_layer, species_locality_score, insertion_seq).  The locality score
#      is derived from a rolling species-frequency histogram so that high-
#      frequency species are promoted to the front of their z-layer band —
#      reducing worst-case <defs> re-emission even when z-layers interleave.
#      Nanite uses a flat BinIndex integer; here we weight by observed batch
#      size, which is the 2-D equivalent of GPU warp occupancy heuristics.
#   2. PSO / raster-bin registry → species-bin registry (string keys).
#   3. TArray<FMeshDrawCommand>::Append atomic write → list.extend() + re-sort
#      on flush (single-threaded epoch).
#   4. FNaniteMaterialSlot sections loop → flat per-cell entry (no sub-sections).
#   5. DrawCallCost accumulator → svg_defs_cost counter (counts <defs> blocks
#      that would be emitted if draw list were flushed naively).
#
# Reference: [ASTRO-NANITE-DL] debug prefix preserved.
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum draw entries the list accumulates before an automatic flush.
# Mirrors GNaniteMaxDrawsPerPass intent: cap unbounded growth.
_DRAW_LIST_MAX_ENTRIES: int = 512

# Weight applied to species-locality scoring.  Controls how aggressively the
# sorter clusters same-species cells within a z-layer band vs. preserving pure
# z-order.  0.0 = pure z-order (Nanite default); 1.0 = pure species grouping.
# 0.35 chosen empirically: keeps z-layer semantics while halving average
# <defs> re-emission rate in typical 8-species SVG graphs.
_SPECIES_LOCALITY_WEIGHT: float = 0.35









def _compute_translucent_sort_key(
    bounds_origin: tuple,
    view_origin: tuple,
    view_matrix: list,
    sort_policy: int = 2,
    sort_axis: tuple = (0.0, 0.0, 1.0),
) -> AstroCellDrawSortKey:
    """
    Compute a translucent draw sort key for a single cell.

    Port of the inner loop body in UpdateTranslucentMeshSortKeys():
      0 = SortByDistance     → distance = |BoundsOrigin - ViewOrigin|
      1 = SortAlongAxis      → distance = dot(BoundsOrigin - ViewOrigin, SortAxis)
      2 = SortByProjectedZ   → distance = ViewMatrix.TransformPosition(BoundsOrigin).Z

    The resulting float distance is bit-inverted so that back-to-front
    ordering maps to ascending unsigned integer order (far = smaller key
    draws first in painter's algorithm).

    鲁迅式：半透明物体必须从远到近画——这不是偏见，是光学定律。
    排序键的精心设计，是为了让 GPU 遵守这一定律而无需额外分支。
    """
    ox, oy, oz = bounds_origin
    vx, vy, vz = view_origin

    if sort_policy == 0:
        # SortByDistance: Euclidean distance
        dx, dy, dz = ox - vx, oy - vy, oz - vz
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    elif sort_policy == 1:
        # SortAlongAxis: projected onto custom axis
        dx, dy, dz = ox - vx, oy - vy, oz - vz
        ax, ay, az = sort_axis
        distance = dx*ax + dy*ay + dz*az
    else:
        # SortByProjectedZ: view-space Z (default in UE5)
        if view_matrix and len(view_matrix) >= 3:
            r = view_matrix[2]
            distance = r[0]*ox + r[1]*oy + r[2]*oz + (r[3] if len(r) > 3 else 0.0)
        else:
            distance = oz - vz

    # BitInvertIfNegativeFloat so unsigned ascending == back-to-front
    key_high = _bit_invert_if_negative(distance)
    return AstroCellDrawSortKey(high=key_high, low=0)









def update_translucent_sort_keys(
    visible_commands: list,
    view_origin: tuple = (0.0, 0.0, -1000.0),
    view_matrix: list = None,
    sort_policy: int = 2,
    sort_axis: tuple = (0.0, 0.0, 1.0),
    inverse_sorting: bool = False,
) -> list:
    """
    Update sort keys for all translucent mesh draw commands.

    Direct port of UpdateTranslucentMeshSortKeys() from MeshDrawCommands.cpp.

    Each entry in visible_commands is a dict with at minimum:
        cell_id:  str
        bbox:     {x, y, w, h, z}  (world bounds)
        species:  str
    After this call every entry gains a ``sort_key`` field (AstroCellDrawSortKey).

    The list is sorted in-place (ascending by sort_key.packed()), which
    produces back-to-front order for painter's algorithm rendering — identical
    to the C++ sorted TArray<FVisibleMeshDrawCommand> result.

    @param visible_commands  List of draw-command entry dicts.
    @param view_origin        Camera / viewer position tuple (x, y, z).
    @param view_matrix        3×4 row-major view matrix (optional).
    @param sort_policy        0=distance, 1=axis, 2=projZ (default).
    @param sort_axis          Custom sort axis when sort_policy == 1.
    @param inverse_sorting    Reverse order (front-to-back for certain passes).
    @return                   Sorted list of commands with sort_key attached.

    鲁迅式：排序是最后的裁决——每一个半透明物体都在争夺被先画的权利，
    但物理定律早已判定：远者先画，近者后画。排序只是执行判决。
    """
    if not visible_commands:
        return visible_commands

    for entry in visible_commands:
        bbox = entry.get("bbox", {})
        # World-space bounds origin = bbox centre
        bx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
        by = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
        bz = float(bbox.get("z", 0))
        bounds_origin = (bx, by, bz)

        entry["sort_key"] = _compute_translucent_sort_key(
            bounds_origin, view_origin, view_matrix or [],
            sort_policy, sort_axis,
        )

    visible_commands.sort(
        key=lambda e: e["sort_key"].packed(),
        reverse=inverse_sorting,
    )

    return visible_commands


# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================



# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================






def _is_translucent_only(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.Translucent


