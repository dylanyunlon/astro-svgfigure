#!/usr/bin/env python3
"""Extracted from cell_component.py — Nanite rendering subsystem."""
import json
import math
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellComposition → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteComposition.cpp
#
# NaniteComposition orchestrates the final depth/stencil export and GBuffer
# composition pass that resolves the Nanite visibility buffer into scene-ready
# render targets.  In the Astro pipeline this maps to the SVG layer-merge step
# where per-cell paint operations (fill, stroke, shadow) are composited into a
# single canonical SVG document for the current epoch.
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GNaniteResummarizeHTile / GNaniteDecompressDepth
#       → glyph_resummarize_zlayer / glyph_decompress_depth:
#         integer knobs (defaulting to 1 / 0) that mirror the CVar semantics;
#         kept as module-level constants so callers can override them in tests.
#
#   FNaniteMarkStencilPS   → _mark_stencil_cells()
#       Pixel-shader pass that writes a per-sample stencil bit wherever the
#       VisBuffer64 holds a valid Nanite primitive.  In SVG terms: tag every
#       cell SVG element that contributed a visible pixel to the frame so
#       downstream compositing layers can clip or mask against them.
#
#   FEmitSceneDepthPS      → _emit_scene_depth()
#       Exports per-sample hardware depth from the visibility buffer into a
#       scene-depth render target.  Astro equivalent: compute the canonical
#       z-value for each cell from its bbox["z"] and write it into the shared
#       depth channel of the composition manifest.
#
#   FEmitSceneStencilPS    → _emit_scene_stencil()
#       A second pixel-shader pass that promotes the shading-mask into a typed
#       stencil texture.  Astro equivalent: mark cells whose shading_mask bit
#       is set in the composition manifest so the SVG renderer can apply the
#       correct <clipPath> group.
#
#   FEmitCustomDepthStencilPS → _emit_custom_depth_stencil()
#       Writes per-primitive custom depth/stencil values (used by post-process
#       outlines, selection highlights, etc.).  Astro equivalent: annotate the
#       SVG element with a data-custom-depth attribute whenever the cell's
#       species_params carries a "highlight" key, enabling a CSS stroke-width
#       post-pass.
#
#   FDepthExportCS          → _depth_export_cs()
#       Compute-shader that exports depth+HTILE in a single dispatched pass on
#       platforms that support direct HTILE writes.  Astro equivalent: batch
#       the depth/stencil annotation writes through a vectorised NumPy/lxml
#       operation rather than an element-by-element Python loop.
#
# AstroCellCompositor:
#   The central class — mirrors the stateless namespace-scope helper functions
#   in NaniteComposition.cpp that are orchestrated by the renderer's
#   FNaniteRenderer::Render() call.  Here we collect all per-cell paint dicts,
#   sort them by z-layer (↔ render order), and emit a merged SVG document
#   with correct painter's-algorithm ordering and per-cell metadata attributes.
# ═══════════════════════════════════════════════════════════════════════════════

#: Mirrors GNaniteResummarizeHTile — when True the compositor will re-sort
#: z-layer buckets after every merge to ensure monotonic painter ordering.
glyph_resummarize_zlayer: int = 1

#: Mirrors GNaniteDecompressDepth — when True the compositor unpacks packed
#: 24-bit depth values from the cell registry before writing the manifest.
glyph_decompress_depth: int = 0

#: Mirrors GNaniteCustomDepthExportMethod (0 = PS path, 1 = CS path).
#: Controls whether _emit_custom_depth_stencil uses element-by-element
#: annotation (0) or a bulk vectorised lxml operation (1).
glyph_custom_depth_export_method: int = 1


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


def _emit_scene_depth(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """写入场景深度通道 — 镜像 FEmitSceneDepthPS。

    从每个 cell 的 ``bbox["z"]`` 提取规范深度值，写入 *depth_manifest*
    的 ``"depth_channel"`` 子字典。键为 cell_id，值为归一化深度 [0.0, 1.0]，
    其中 z=0 对应近裁剪面，z_max 对应远裁剪面。

    归一化公式（镜像 UE5 Nanite 深度重建）::

        depth_norm = 1.0 - (z - z_min) / max(z_range, 1e-6)

    这与 UE5 反转深度缓冲区约定一致：z 越大（越深）depth_norm 越小。
    """
    zs = [e["bbox"]["z"] for e in cell_entries if "bbox" in e]
    z_min = min(zs) if zs else 0.0
    z_max = max(zs) if zs else 1.0
    z_range = z_max - z_min

    depth_channel: dict[str, float] = {}
    for entry in cell_entries:
        z = entry.get("bbox", {}).get("z", 0.0)
        depth_channel[entry["cell_id"]] = 1.0 - (z - z_min) / max(z_range, 1e-6)

    depth_manifest.setdefault("depth_channel", {}).update(depth_channel)
    return depth_manifest


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


class AstroCellCompositor:
    """最终 SVG 合成器 — 镜像 NaniteComposition.cpp 中的渲染流水线。

    将逐 cell 的 paint 字典按 z-layer 排序后，合并为单一 SVG 文档。每个 cell
    贡献一个 ``<g data-cell-id="…">`` 分组；深度/模板元数据以 ``data-*``
    属性形式嵌入，供 PixiJS / D3 渲染器按需读取。

    Lifecycle（镜像 FNaniteRenderer::Render 调用序列）::

        compositor = AstroCellCompositor(vis_set)
        compositor.begin_frame(cell_entries)          # ↔ InitViews
        compositor.emit_depth_stencil(depth_manifest) # ↔ EmitDepthStencil
        svg_doc = compositor.compose()                # ↔ FinalCompose

    Parameters
    ----------
    vis_set:
        本帧可见的 cell_id 集合；由 :func:`perform_nanite_visibility` 返回的
        :class:`AstroCellVisibilityQuery` 的 ``visible_cells`` 属性提供。
    """

    def __init__(self, vis_set: set[str]) -> None:
        self._vis_set = vis_set
        self._cell_entries: list[dict] = []
        self._depth_manifest: dict = {}
        self._stamped: list[dict] = []

    # ------------------------------------------------------------------
    # begin_frame — 镜像 InitViews / PrepareRenderTargets
    # ------------------------------------------------------------------
    def begin_frame(self, cell_entries: list[dict]) -> None:
        """接收本帧所有 cell 条目并执行模板标记（FNaniteMarkStencilPS）。

        同时按 z-layer 升序排列条目，确保后续合成遵循画家算法；当
        :data:`glyph_resummarize_zlayer` 为真时重新排序（镜像
        GNaniteResummarizeHTile 的 HTILE 重摘要行为）。
        """
        self._cell_entries = cell_entries
        self._stamped = _mark_stencil_cells(cell_entries, self._vis_set)
        if glyph_resummarize_zlayer:
            self._stamped.sort(key=lambda e: e.get("bbox", {}).get("z", 0.0))

    # ------------------------------------------------------------------
    # emit_depth_stencil — 镜像 EmitDepthStencil pass
    # ------------------------------------------------------------------
    def emit_depth_stencil(self, depth_manifest: dict | None = None) -> dict:
        """执行深度/模板写入三连 pass，返回填充后的 depth_manifest。

        依次调用 :func:`_emit_scene_depth`、:func:`_emit_scene_stencil`、
        :func:`_emit_custom_depth_stencil`，与 UE5 中
        ``EmitDepthStencil → EmitSceneStencil → EmitCustomDepthStencil``
        的 RDG pass 调度顺序完全对应。
        """
        if depth_manifest is None:
            depth_manifest = {}
        self._depth_manifest = depth_manifest
        _emit_scene_depth(self._stamped, self._depth_manifest)
        _emit_scene_stencil(self._stamped, self._depth_manifest)
        _emit_custom_depth_stencil(self._stamped, self._depth_manifest)
        return self._depth_manifest

    # ------------------------------------------------------------------
    # compose — 镜像 FinalCompose / present
    # ------------------------------------------------------------------
    def compose(self) -> dict:
        """将所有可见 cell 合成为结构化 dict，供前端 PixiJS 消费。
        不再拼接 SVG 字符串。

        仅输出 ``stencil=1`` 的条目（可见 cell）；每个 cell 生成一个带有
        ``data-cell-id``、``data-z``、``data-depth`` 以及可选
        ``data-highlight`` 属性的 group dict。完整的内容由
        各 cell 自身的 ``svg_fragment`` 字段提供（若存在）。

        Returns
        -------
        dict
            结构化合成参数 dict，含 groups 列表。
        """
        groups: list[dict] = []
        depth_ch = self._depth_manifest.get("depth_channel", {})
        custom_d = self._depth_manifest.get("custom_depth", {})

        for entry in self._stamped:
            if not entry.get("stencil", 0):
                continue
            cid = entry["cell_id"]
            z = entry.get("bbox", {}).get("z", 0.0)
            depth = depth_ch.get(cid, 0.0)
            group: dict = {
                "element": "g",
                "data-cell-id": cid,
                "data-z": z,
                "data-depth": round(depth, 6),
                "children": entry.get("svg_fragment", ""),
            }
            if cid in custom_d:
                group["data-highlight"] = custom_d[cid]
            groups.append(group)

        return {
            "version": "1.0",
            "encoding": "UTF-8",
            "groups": groups,
        }


def compose_cell_svg(
    cell_entries: list[dict],
    vis_set: set[str],
    depth_manifest: dict | None = None,
) -> tuple[str, dict]:
    """顶层合成入口 — 镜像 FNaniteRenderer::Render 的 Composition 阶段。

    便利包装：构造 :class:`AstroCellCompositor`，依次执行三个 lifecycle 方法，
    返回合成后的 SVG 字符串与填充后的 depth_manifest。

    Parameters
    ----------
    cell_entries:
        来自 cell_registry 的全量 cell 条目列表。
    vis_set:
        本帧可见 cell_id 集合。
    depth_manifest:
        可选的现有深度清单；若为 None 则创建新字典。

    Returns
    -------
    tuple[str, dict]
        ``(svg_document, depth_manifest)``
    """
    compositor = AstroCellCompositor(vis_set)
    compositor.begin_frame(cell_entries)
    dm = compositor.emit_depth_stencil(depth_manifest)
    svg = compositor.compose()
    return svg, dm


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellSharedResources → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteShared.cpp
#
# NaniteShared.cpp 是整个 Nanite 子系统的「公共资源层」：它注册全局 GPU 统计、
# 声明 Uniform Buffer 槽位，并暴露跨 pass 共享的 LOD/硬件路径判定函数。
# 在 Astro 流水线中，该层对应一组进程级单例资源与无状态工具函数，供
# Composition、StreamExport、Translucency 等子系统统一调用。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   DEFINE_GPU_STAT(NaniteDebug)         → _ASTRO_CELL_PERF_COUNTERS dict
#   GNaniteMaxNodes / GNaniteMaxVisibleClusters
#       → ASTRO_CELL_MAX_NODES / ASTRO_CELL_MAX_VISIBLE_CLUSTERS
#   UseMeshShader() / UsePrimitiveShader()
#       → use_vector_render_path() — 判断是否启用矢量批渲染（PixiJS mesh path）
#   FPackedView::UpdateLODScales()
#       → AstroCellPackedView.update_lod_scales()
#   FGlobalResources::GetMaxNodes() 等
#       → AstroCellGlobalResources 类
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 GNaniteMaxNodes — 单帧最多参与 LOD 遍历的 cell 节点数上限。
ASTRO_CELL_MAX_NODES: int = 2 * 1_048_576

#: 镜像 GNaniteMaxCandidateClusters — 聚类裁剪前的候选 cluster 上限。
ASTRO_CELL_MAX_CANDIDATE_CLUSTERS: int = 16 * 1_048_576

#: 镜像 GNaniteMaxVisibleClusters — 最终可见 cluster 数量上限。
ASTRO_CELL_MAX_VISIBLE_CLUSTERS: int = 4 * 1_048_576

#: 镜像 GNaniteMaxVisibleAssemblyParts — 可见装配零件数量上限（多 species 场景）。
ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS: int = 256 * 1_024

#: 镜像 DEFINE_GPU_STAT(NaniteDebug) — 运行时性能计数器字典。
#: 键为计数器名称，值为当前帧的累计计数（整型）。
_ASTRO_CELL_PERF_COUNTERS: dict[str, int] = {
    "nodes_traversed":       0,
    "candidate_clusters":    0,
    "visible_clusters":      0,
    "visible_assembly_parts": 0,
}


def reset_perf_counters() -> None:
    """重置所有性能计数器 — 镜像帧间 GPU stat 清零。"""
    for k in _ASTRO_CELL_PERF_COUNTERS:
        _ASTRO_CELL_PERF_COUNTERS[k] = 0


def increment_perf_counter(name: str, delta: int = 1) -> None:
    """递增指定性能计数器；键不存在时静默创建。"""
    _ASTRO_CELL_PERF_COUNTERS[name] = _ASTRO_CELL_PERF_COUNTERS.get(name, 0) + delta


def use_vector_render_path() -> bool:
    """判断是否启用矢量批渲染路径 — 镜像 UseMeshShader() / UsePrimitiveShader()。

    在 UE5 中，Mesh Shader / Primitive Shader 路径在支持 Tier-1 Mesh Shader
    的平台上激活，以减少 draw call 开销。Astro 的对应逻辑：当 PixiJS 渲染器
    支持 WebGL2 Instanced Mesh（``ASTRO_VECTOR_RENDER=1`` 环境变量）时返回
    True，否则退化为逐元素 SVG 路径（镜像 VertexShader fallback）。
    """
    import os
    return os.environ.get("ASTRO_VECTOR_RENDER", "0") == "1"


class AstroCellPackedView:
    """单视图 LOD 参数包 — 镜像 FPackedView。

    存储视图相关的 LOD 缩放因子，供 cell 可见性查询和 cluster 裁剪使用。
    核心方法 :meth:`update_lod_scales` 镜像 ``FPackedView::UpdateLODScales``，
    根据视口尺寸与像素/边缘阈值计算两个缩放系数。

    Parameters
    ----------
    view_size_y:
        视口垂直像素数（镜像 ViewSizeAndInvSize.Y）。
    view_to_clip_m11:
        投影矩阵 [1][1] 分量（镜像 ViewToClip.M[1][1]）；透视投影中该值等于
        ``2 * focal_length / view_height``。
    """

    def __init__(self, view_size_y: float, view_to_clip_m11: float) -> None:
        self.view_size_y = view_size_y
        self.view_to_clip_m11 = view_to_clip_m11
        self.lod_scale: float = 1.0
        self.lod_scale_hw: float = 1.0

    def update_lod_scales(
        self,
        max_pixels_per_edge: float = 1.0,
        min_pixels_per_edge_hw: float = 0.25,
    ) -> None:
        """重算 LOD 缩放 — 镜像 FPackedView::UpdateLODScales。

        Parameters
        ----------
        max_pixels_per_edge:
            软件光栅化路径的最大边缘像素密度（镜像 CVarNaniteMaxPixelsPerEdge）。
        min_pixels_per_edge_hw:
            硬件光栅化路径的最小边缘像素密度（镜像 CVarNaniteMinPixelsPerEdgeHW）。
        """
        view_to_pixels = 0.5 * self.view_to_clip_m11 * self.view_size_y
        self.lod_scale = view_to_pixels / max(max_pixels_per_edge, 1e-9)
        self.lod_scale_hw = view_to_pixels / max(min_pixels_per_edge_hw, 1e-9)


class AstroCellGlobalResources:
    """全局资源单例 — 镜像 Nanite::FGlobalResources。

    持有跨帧共享的缓冲区容量上限，供 :class:`AstroCellFeedbackManager`
    溢出检测与 :class:`AstroCellStreamExporter` 容量守卫使用。

    所有 getter 均为类方法，镜像 UE5 的静态成员函数调用语义。
    """

    @classmethod
    def get_max_nodes(cls) -> int:
        """返回节点缓冲区上限 — 镜像 FGlobalResources::GetMaxNodes()。"""
        return ASTRO_CELL_MAX_NODES

    @classmethod
    def get_max_candidate_clusters(cls) -> int:
        return ASTRO_CELL_MAX_CANDIDATE_CLUSTERS

    @classmethod
    def get_max_visible_clusters(cls) -> int:
        return ASTRO_CELL_MAX_VISIBLE_CLUSTERS

    @classmethod
    def get_max_visible_assembly_parts(cls) -> int:
        return ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellFeedback → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteFeedback.cpp
#
# NaniteFeedback 通过 GPU Message 异步回读峰值水位，在溢出时向屏幕与日志写入
# 警告。Astro 等价实现：同步回读来自 :class:`AstroCellGlobalResources` 的容量
# 上限，在单帧节点/cluster 计数超标时通过 stderr 发出警告，并维护高水位线
# 字典，供监控脚本消费。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FFeedbackManager::FFeedbackManager()  → AstroCellFeedbackManager.__init__
#   FFeedbackManager::FBufferState::Update → AstroCellFeedbackManager._update_state
#   ReportMaterialPerformanceWarning      → report_material_perf_warning
#   CVarEmitMaterialPerformanceWarnings   → ASTRO_EMIT_MATERIAL_PERF_WARNINGS
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 CVarEmitMaterialPerformanceWarnings — 为 True 时输出 species 性能警告。
ASTRO_EMIT_MATERIAL_PERF_WARNINGS: bool = False


class AstroCellFeedbackManager:
    """缓冲区溢出反馈管理器 — 镜像 Nanite::FFeedbackManager。

    维护四条高水位线状态（nodes / candidate_clusters / visible_clusters /
    visible_assembly_parts），并在超容时向 stderr 写入警告。亦可跟踪 species
    材质性能警告（对应 ``CVarEmitMaterialPerformanceWarnings``）。

    Attributes
    ----------
    high_water_marks : dict[str, int]
        各缓冲区的历史峰值（镜像 FBufferState::HighWaterMark）。
    material_warnings : dict[str, float]
        species 名 → 最后一次出现的时间戳（镜像 MaterialWarningItems）。
    """

    def __init__(self) -> None:
        self.high_water_marks: dict[str, int] = {
            "nodes":                0,
            "candidate_clusters":   0,
            "visible_clusters":     0,
            "visible_assembly_parts": 0,
        }
        self._latest_overflow: dict[str, float] = {}
        self.material_warnings: dict[str, float] = {}

    def _update_state(self, key: str, peak: int, capacity: int) -> bool:
        """更新单条缓冲区高水位线，溢出时记录时间戳并返回 True。

        镜像 ``FFeedbackManager::FBufferState::Update``。
        """
        import time
        new_hwm = peak > self.high_water_marks[key]
        if peak > capacity:
            self._latest_overflow[key] = time.monotonic()
        self.high_water_marks[key] = max(self.high_water_marks[key], peak)
        return new_hwm and peak > capacity

    def tick(self, peak_nodes: int, peak_candidate_clusters: int,
             peak_visible_clusters: int, peak_visible_assembly_parts: int) -> None:
        """每帧调用，镜像 GPU Message 回调触发时机。

        将四个峰值与 :class:`AstroCellGlobalResources` 的容量上限比对；
        一旦突破历史高水位且发生溢出，向 stderr 输出警告，与 UE5 的
        ``UE_LOGF(LogRenderer, Warning, …)`` 语义对应。
        """
        checks = [
            ("nodes",                  peak_nodes,                  AstroCellGlobalResources.get_max_nodes(),
             "node buffer", "ASTRO_CELL_MAX_NODES"),
            ("candidate_clusters",     peak_candidate_clusters,     AstroCellGlobalResources.get_max_candidate_clusters(),
             "candidate cluster buffer", "ASTRO_CELL_MAX_CANDIDATE_CLUSTERS"),
            ("visible_clusters",       peak_visible_clusters,       AstroCellGlobalResources.get_max_visible_clusters(),
             "visible cluster buffer", "ASTRO_CELL_MAX_VISIBLE_CLUSTERS"),
            ("visible_assembly_parts", peak_visible_assembly_parts, AstroCellGlobalResources.get_max_visible_assembly_parts(),
             "visible assembly part buffer", "ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS"),
        ]
        for key, peak, cap, label, cvar in checks:
            if self._update_state(key, peak, cap):
                print(
                    f"[AstroCellFeedback] WARNING: {label} overflow detected. "
                    f"New high-water mark is {self.high_water_marks[key]} / {cap}. "
                    f"Increase {cvar} to prevent visual artifacts.",
                    file=__import__("sys").stderr,
                )

    def report_material_perf_warning(self, species_name: str) -> None:
        """报告 species 材质性能警告 — 镜像 ReportMaterialPerformanceWarning。

        对同一 species 采用 5 秒冷却窗口限流，避免日志洪泛。
        仅当 :data:`ASTRO_EMIT_MATERIAL_PERF_WARNINGS` 为 True 时输出。
        """
        if not ASTRO_EMIT_MATERIAL_PERF_WARNINGS:
            return
        import time
        now = time.monotonic()
        last = self.material_warnings.get(species_name, 0.0)
        if now - last > 5.0:
            self.material_warnings[species_name] = now
            print(
                f"[AstroCellFeedback] Performance Warning: "
                f"Programmable Astro species '{species_name}' uses PDO or is Masked!",
                file=__import__("sys").stderr,
            )


#: 进程级单例 — 镜像 FFeedbackManager 内嵌于 FGlobalResources 的生命周期。
_astro_cell_feedback_manager: AstroCellFeedbackManager = AstroCellFeedbackManager()


def get_feedback_manager() -> AstroCellFeedbackManager:
    """返回进程级反馈管理器单例。"""
    return _astro_cell_feedback_manager


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellStreamExport → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteStreamOut.cpp
#
# NaniteStreamOut 将 GPU 端可见 cluster 的顶点/索引数据流出到 CPU 可访问缓冲区，
# 用于物理碰撞、几何体烘焙等离线用途。Astro 等价实现：将可见 cell 的 SVG 片段
# 与 bbox 元数据序列化到磁盘（或内存缓冲区），供外部工具（布局优化器、碰撞检测
# 服务）消费。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FStreamOutQueueParameters   → AstroCellStreamExportQueue（优先级队列）
#   FInitQueueCS                → AstroCellStreamExporter.init_queue()
#   FNaniteStreamOutTraversalCS → AstroCellStreamExporter.traverse()
#   FAllocateRangesCS           → AstroCellStreamExporter.allocate_ranges()
#   FNaniteStreamOutCS          → AstroCellStreamExporter.stream_out()
#   GNaniteStreamOutCacheTraversalData
#       → ASTRO_STREAM_OUT_CACHE_TRAVERSAL（模块级布尔常量）
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 GNaniteStreamOutCacheTraversalData — 为 True 时在 count pass 中缓存
#: traversal 中间结果，stream-out pass 可跳过重复遍历。
ASTRO_STREAM_OUT_CACHE_TRAVERSAL: bool = True


class AstroCellStreamExportQueue:
    """流出请求队列 — 镜像 FStreamOutQueueParameters + FInitQueueCS。

    维护待流出的 cell_id 列表及各自的分配范围（顶点起始偏移 + 数量），
    对应 UE5 的 ``VertexBuffer`` / ``IndexBuffer`` 分配器。

    Parameters
    ----------
    vertex_buffer_size:
        顶点缓冲区容量（元素数），镜像 ``VertexBufferSize``。
    index_buffer_size:
        索引缓冲区容量（元素数），镜像 ``IndexBufferSize``。
    """

    def __init__(self, vertex_buffer_size: int = 4 * 1_048_576,
                 index_buffer_size: int = 8 * 1_048_576) -> None:
        self.vertex_buffer_size = vertex_buffer_size
        self.index_buffer_size = index_buffer_size
        self._requests: list[dict] = []          # 镜像 StreamOutRequests SRV
        self._allocations: dict[str, dict] = {}  # cell_id → {v_start, v_count, i_start, i_count}
        self._vertex_cursor: int = 0
        self._index_cursor: int = 0

    def enqueue(self, cell_id: str, vertex_count: int, index_count: int) -> bool:
        """排入一条流出请求 — 镜像 FInitQueueCS 的每帧 request 填充。

        若缓冲区剩余空间不足则返回 False（镜像溢出保护）；
        否则记录分配范围并返回 True。
        """
        if (self._vertex_cursor + vertex_count > self.vertex_buffer_size or
                self._index_cursor + index_count > self.index_buffer_size):
            get_feedback_manager().report_material_perf_warning(cell_id)
            return False
        self._requests.append({"cell_id": cell_id,
                                "vertex_count": vertex_count,
                                "index_count": index_count})
        self._allocations[cell_id] = {
            "v_start": self._vertex_cursor,
            "v_count": vertex_count,
            "i_start": self._index_cursor,
            "i_count": index_count,
        }
        self._vertex_cursor += vertex_count
        self._index_cursor += index_count
        return True

    def reset(self) -> None:
        """重置队列 — 镜像帧间缓冲区归零（FInitQueueCS 的 reset state）。"""
        self._requests.clear()
        self._allocations.clear()
        self._vertex_cursor = 0
        self._index_cursor = 0


class AstroCellStreamExporter:
    """Cell 数据流出器 — 镜像 NaniteStreamOut.cpp 中的四阶段 CS dispatch。

    四阶段 pipeline（完全镜像 UE5 的 compute shader dispatch 序列）：

    1. ``init_queue``      — 初始化请求队列（FInitQueueCS）
    2. ``traverse``        — BVH/cluster 遍历，统计顶点/索引数量
                              （FNaniteStreamOutTraversalCS，count 模式）
    3. ``allocate_ranges`` — 按计数结果分配缓冲区范围（FAllocateRangesCS）
    4. ``stream_out``      — 将几何数据写入分配范围（FNaniteStreamOutCS）

    在 Astro 语境中，「几何数据」= cell 的 SVG 片段字符串 + bbox 元数据 JSON。

    Parameters
    ----------
    registry_snapshot:
        来自 ``_load_cell_registry()`` 的 cell 字典快照（``{"cells": {…}}``）。
    output_dir:
        流出文件的写出目录（镜像 CPU 可访问缓冲区的内存映射路径）。
    """

    def __init__(self, registry_snapshot: dict, output_dir: str = "/tmp/astro_streamout") -> None:
        self._registry = registry_snapshot
        self._output_dir = output_dir
        self._queue = AstroCellStreamExportQueue()
        self._traversal_cache: dict[str, dict] | None = None  # 镜像 cached traversal data

    def init_queue(self, cell_ids: list[str]) -> None:
        """阶段 1：初始化请求队列 — 镜像 AddInitQueuePass / FInitQueueCS。

        为每个请求的 cell_id 估算顶点/索引数量（SVG 路径点数 × 2），
        然后调用 :meth:`AstroCellStreamExportQueue.enqueue` 预占缓冲区范围。
        """
        self._queue.reset()
        cells = self._registry.get("cells", {})
        for cid in cell_ids:
            cell = cells.get(cid, {})
            # 粗略估算：每个 cell 约 12 个 SVG 顶点，24 个索引（镜像 cluster 平均三角形数）
            v_count = cell.get("vertex_hint", 12)
            i_count = cell.get("index_hint", 24)
            self._queue.enqueue(cid, v_count, i_count)

    def traverse(self, vis_set: set[str]) -> dict[str, dict]:
        """阶段 2：遍历并统计 — 镜像 FNaniteStreamOutTraversalCS（count 模式）。

        若 :data:`ASTRO_STREAM_OUT_CACHE_TRAVERSAL` 为 True 且缓存命中，
        直接返回缓存结果（镜像 GNaniteStreamOutCacheTraversalData=1 时的
        skip-traversal 优化）；否则重新遍历并缓存。

        Returns
        -------
        dict[str, dict]
            cell_id → {"bbox": …, "species": …} 的可见 cell 快照。
        """
        if ASTRO_STREAM_OUT_CACHE_TRAVERSAL and self._traversal_cache is not None:
            return self._traversal_cache

        cells = self._registry.get("cells", {})
        result = {cid: cells[cid] for cid in vis_set if cid in cells}
        if ASTRO_STREAM_OUT_CACHE_TRAVERSAL:
            self._traversal_cache = result
        return result

    def allocate_ranges(self, traversal_result: dict[str, dict]) -> dict[str, dict]:
        """阶段 3：分配缓冲区范围 — 镜像 FAllocateRangesCS。

        用遍历结果中的精确顶点/索引数覆盖 init_queue 阶段的估算值，
        返回最终分配字典（cell_id → 分配范围）。
        """
        allocs = dict(self._queue._allocations)
        for cid, data in traversal_result.items():
            if cid in allocs:
                # 用真实值修正估算（镜像 AllocateRangesCS 覆写 MeshDataBuffer）
                actual_v = data.get("vertex_hint", allocs[cid]["v_count"])
                actual_i = data.get("index_hint", allocs[cid]["i_count"])
                allocs[cid]["v_count"] = actual_v
                allocs[cid]["i_count"] = actual_i
        return allocs

    def stream_out(
        self,
        traversal_result: dict[str, dict],
        allocations: dict[str, dict],
    ) -> list[dict]:
        """阶段 4：执行流出写入 — 镜像 FNaniteStreamOutCS。

        将每个可见 cell 的 SVG 片段与 bbox 元数据序列化为一条输出记录。
        返回所有输出记录列表（镜像 VertexBuffer + IndexBuffer 写出完成）。

        Parameters
        ----------
        traversal_result:
            阶段 2 返回的可见 cell 快照。
        allocations:
            阶段 3 返回的缓冲区范围分配字典。

        Returns
        -------
        list[dict]
            每条记录包含 ``cell_id``、``bbox``、``species``、``v_start``、
            ``i_start``、``v_count``、``i_count``、``svg_fragment``。
        """
        output_records: list[dict] = []
        for cid, cell_data in traversal_result.items():
            alloc = allocations.get(cid, {})
            record = {
                "cell_id":     cid,
                "bbox":        cell_data.get("bbox", {}),
                "species":     cell_data.get("species", ""),
                "v_start":     alloc.get("v_start", 0),
                "i_start":     alloc.get("i_start", 0),
                "v_count":     alloc.get("v_count", 0),
                "i_count":     alloc.get("i_count", 0),
                "svg_fragment": cell_data.get("svg_fragment", ""),
            }
            output_records.append(record)
            increment_perf_counter("visible_clusters", 1)

        return output_records

    def run(self, cell_ids: list[str], vis_set: set[str]) -> list[dict]:
        """四阶段流出主入口 — 镜像 FNaniteRenderer 中的 StreamOut 调度序列。

        Parameters
        ----------
        cell_ids:
            本帧请求流出的 cell_id 列表。
        vis_set:
            本帧可见 cell_id 集合。

        Returns
        -------
        list[dict]
            完整的流出记录列表（见 :meth:`stream_out` 返回值说明）。
        """
        self.init_queue(cell_ids)
        traversal = self.traverse(vis_set)
        allocs = self.allocate_ranges(traversal)
        return self.stream_out(traversal, allocs)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellTranslucency → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteTranslucency.cpp
#
# NaniteTranslucency 为半透明 Nanite 图元实现了独立的光栅化/着色路径：
# 半透明材质无法利用 Visibility Buffer 的延迟着色，须退化为前向渲染。
# Astro 等价实现：对 opacity < 1.0 的 cell（对应半透明图元），执行单独的
# SVG 混合通道，确保正确的 Alpha 合成顺序，而非依赖主合成器的画家算法。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarNaniteMeshShaderTranslucency
#       → ASTRO_TRANSLUCENCY_MESH_SHADER（布尔常量，控制矢量批渲染路径）
#   UseNaniteMeshShader()      → use_translucency_vector_path()
#   FNaniteTranslucencyFactory → AstroCellTranslucencyFactory（顶点工厂等价）
#   SetTranslucencyParameters  → AstroCellTranslucencyRenderer.set_parameters()
#   RenderTranslucency         → AstroCellTranslucencyRenderer.render()
#   FTranscodeRasterizerArgs_CS → _transcode_rasterizer_args()
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 CVarNaniteMeshShaderTranslucency — True 时使用矢量批渲染半透明通道。
ASTRO_TRANSLUCENCY_MESH_SHADER: bool = True

#: 半透明判定阈值 — opacity 低于此值的 cell 进入半透明通道。
ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD: float = 1.0


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

        # 生成各 cell 的半透明结构化 dict
        fragments: list[dict] = []
        for entry in translucent:
            cid = entry["cell_id"]
            opacity = entry.get("opacity", 1.0)
            blend_mode = entry.get("blend_mode", "translucent")
            if not self._factory.should_compile_permutation("*", blend_mode):
                blend_mode = "translucent"
            attrs = self._factory.prepare_svg_attrs(opacity, blend_mode)
            fragment = entry.get("svg_fragment", "")
            group: dict = {
                "element": "g",
                "data-cell-id": cid,
                "children": fragment,
            }
            group.update(attrs)
            fragments.append(group)
            increment_perf_counter("visible_clusters", 1)

        return {
            "element": "g",
            "id": "translucency-layer",
            "children": fragments,
        }


