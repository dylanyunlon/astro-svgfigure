import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





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
    def compose(self) -> str:
        """将所有可见 cell 合成为一个 SVG 字符串。

        仅输出 ``stencil=1`` 的条目（可见 cell）；每个 cell 生成一个带有
        ``data-cell-id``、``data-z``、``data-depth`` 以及可选
        ``data-highlight`` 属性的 ``<g>`` 占位分组。完整的 SVG 内容由
        各 cell 自身的 ``svg_fragment`` 字段提供（若存在）。

        Returns
        -------
        str
            合并后的 SVG 文档字符串（UTF-8）。
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
            fragment = entry.get("svg_fragment", "")
            group: dict = {
                "tag": "g",
                "data-cell-id": cid,
                "data-z": z,
                "data-depth": round(depth, 6),
                "children": fragment,
            }
            if cid in custom_d:
                group["data-highlight"] = custom_d[cid]
            groups.append(group)

        return {
            "tag": "svg",
            "xmlns": "http://www.w3.org/2000/svg",
            "children": groups,
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




