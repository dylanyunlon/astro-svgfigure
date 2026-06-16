import os, sys, json, math
from typing import Any, Optional
from dataclasses import dataclass, field
from channels.rendering.species.species_port import _species_to_index
from channels.rendering.color.color_extra import _colour_to_hex
from channels.rendering.decoration.decoration_extra import _SPECIES_INDEX_TO_COLOUR
from channels.rendering.constants import _STYLE_PROBE_WEIGHT, _SPECIES_LOCALITY_WEIGHT

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)









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

















class AstroCellDrawList:
    """
    Python port of FNaniteDrawListContext.

    Accumulates cell draw entries (register_cell_draw_entry) then produces a
    species-batched draw order via flush_draw_order().  The returned order
    minimises SVG <defs> block re-emission by placing same-species cells
    contiguously within each z-layer band.

    Lifecycle mirrors FNaniteDrawListContext:
        __init__   — allocate per-species batch accumulators (AddMeshDrawCommand)
        register   — insert cell entry into pending list (AddPrimitive analogue)
        flush      — sort + return ordered list (Submit analogue)
        reset      — discard state for next epoch (Reset analogue)
    """

    def __init__(self) -> None:
        # Pending entries: list of (z_layer, species, cell_id, bbox, extra).
        # Mirrors TArray<FMeshDrawCommand> PendingDraws.
        self._pending: list = []

        # Per-species frequency histogram used for locality scoring.
        # Mirrors the per-BinIndex draw-call cost accumulator in Nanite.
        self._species_freq: dict = {}

        # Monotonically increasing insertion sequence number.
        # Used as tiebreaker so the sort is stable across equal keys.
        self._seq: int = 0

        # Running count of <defs> blocks that would be emitted without batching.
        # Incremented on each species transition in the naive order; reset on flush.
        self.svg_defs_cost: int = 0

    def register_cell_draw_entry(
        self,
        cell_id: str,
        z_layer: int,
        species: str,
        bbox: dict,
        extra: dict | None = None,
    ) -> None:
        """
        Insert a cell into the draw list.

        Mirrors AddPrimitive() / FNaniteMaterialListContext::AddShadingBin():
            DrawList.AddMeshDrawCommand(MeshDrawCommand, DrawCallCost);

        The cell is appended to _pending with its composite sort key components.
        Actual ordering is deferred to flush_draw_order() so that the locality
        scorer can see the full frequency distribution before committing.

        @param cell_id   Unique cell identifier string.
        @param z_layer   Integer z-layer index (primary sort key — coarse depth).
        @param species   Species name string (secondary sort key — material group).
        @param bbox      Cell bounding-box dict {x, y, w, h, z}.
        @param extra     Optional extra payload forwarded verbatim to callers.
        """
        if len(self._pending) >= _DRAW_LIST_MAX_ENTRIES:
            # Auto-flush guard: mirrors Nanite's pass-size cap.
            # In the single-threaded epoch loop this is a safety valve only.
            import sys as _sys
            print(
                f"[ASTRO-NANITE-DL] register_cell_draw_entry: pending list at "
                f"capacity ({_DRAW_LIST_MAX_ENTRIES}), auto-flushing before insert.",
                file=_sys.stderr,
            )
            self.flush_draw_order()

        self._species_freq[species] = self._species_freq.get(species, 0) + 1
        self._pending.append({
            "cell_id": cell_id,
            "z_layer": z_layer,
            "species": species,
            "bbox":    bbox,
            "extra":   extra or {},
            "_seq":    self._seq,
        })
        self._seq += 1

    def _locality_score(self, species: str) -> float:
        """
        Compute a species locality score in [0, 1).

        Higher-frequency species get lower scores (sort earlier within their
        z-layer band) so large batches of the same species are rendered first,
        maximising contiguous runs and minimising <defs> re-emission.

        Mirrors the DrawCallCost heuristic in FNaniteDrawListContext where
        cheaper (lower-cost) draw commands are sorted to the front so that
        GPU wave occupancy is maximised for the common case.

        Algorithm change vs. Nanite:
            Nanite uses a raw integer BinIndex as the sort key inside a
            TArray<uint16> bin-index list, relying on the registrar to assign
            low indices to common materials.  Here we compute the score
            dynamically from observed frequency so that no pre-registration
            is needed — appropriate for a dynamic SVG scene where species
            composition changes per epoch.
        """
        total = max(self._seq, 1)
        freq  = self._species_freq.get(species, 0)
        # Normalise: species with freq/total → 1.0 gets score 0.0 (front).
        return 1.0 - (freq / total)

    def flush_draw_order(self) -> list:
        """
        Sort pending draw entries and return the ordered draw list.

        Mirrors FNaniteDrawListContext::Submit() which emits draw commands in
        sorted order to the RHI command list.

        Sort key (three-component, stable):
            1. z_layer                   — coarse depth (ascending)
            2. species_locality_score    — species batch size proxy (ascending;
                                           lower = larger batch = render first)
            3. _seq                      — insertion order tiebreaker (stable)

        The locality weight _SPECIES_LOCALITY_WEIGHT blends components 1 and 2:
            effective_key = z_layer + locality_score * _SPECIES_LOCALITY_WEIGHT
        This keeps z-layer semantics dominant while still clustering species.

        After sorting, counts <defs> transitions and updates svg_defs_cost.
        Resets internal state for the next epoch.

        @return  Ordered list of entry dicts (cell_id, z_layer, species, bbox,
                 extra fields).
        """
        import sys as _sys

        if not self._pending:
            return []

        # Build locality scores once (O(S) where S = distinct species count).
        scores = {sp: self._locality_score(sp) for sp in self._species_freq}

        # Stable sort: Python's timsort preserves insertion order for equal keys.
        # Mirrors std::stable_sort on FMeshDrawCommand draw-call cost + BinIndex.
        self._pending.sort(
            key=lambda e: (
                e["z_layer"] + scores.get(e["species"], 0.0) * _SPECIES_LOCALITY_WEIGHT,
                e["_seq"],
            )
        )

        # Count <defs> transitions in the sorted order (diagnostic metric).
        defs_cost = 0
        prev_species = None
        for entry in self._pending:
            if entry["species"] != prev_species:
                defs_cost += 1
                prev_species = entry["species"]
        self.svg_defs_cost = defs_cost

        result = [
            {
                "cell_id": e["cell_id"],
                "z_layer": e["z_layer"],
                "species": e["species"],
                "bbox":    e["bbox"],
                **e["extra"],
            }
            for e in self._pending
        ]

        naive_cost = len(self._pending)  # worst case: every cell different species
        print(
            f"[ASTRO-NANITE-DL] flush_draw_order: entries={len(result)} "
            f"defs_transitions={defs_cost} "
            f"naive_defs={naive_cost} "
            f"reduction={100.0 * (1.0 - defs_cost / max(naive_cost, 1)):.1f}%",
            file=_sys.stderr,
        )

        self._pending.clear()
        self._species_freq.clear()
        self._seq = 0

        return result



# ---------------------------------------------------------------------------
# Internal geometry primitive
# ---------------------------------------------------------------------------














class AstroCellStyleProbe:
    """
    Python equivalent of FAstroCellStyleProbe.

    Samples the published status + bbox channels of the six cardinal neighbour
    cells, accumulates their species indices and representative colours into a
    palette, then exposes blend_toward_neighbour_palette() which nudges the
    cell's own SVG colour parameters toward the neighbourhood average.

    Lifetime: created inside proc() each call, discarded after SVG finalisation.
    """

    # Maximum palette entries — mirrors MaxPaletteEntries = 8 in C++.
    MAX_PALETTE_ENTRIES: int = 8

    def __init__(self, cell_id: str, bbox: dict, cell_style_weight: float = _STYLE_PROBE_WEIGHT):
        self.cell_id          = cell_id
        self.world_x          = float(bbox["x"])
        self.world_y          = float(bbox["y"])
        self.world_z          = float(bbox.get("z", 0))
        self.cell_w           = float(bbox["w"])
        self.cell_h           = float(bbox["h"])
        # InfluenceRadius — max half-extent of the bbox (mirrors Comp->InfluenceRadius)
        self.influence_radius = max(self.cell_w, self.cell_h) / 2.0
        self.cell_style_weight = max(0.0, min(1.0, cell_style_weight))

        # Palette: list of (r,g,b) tuples from neighbour cells, up to MAX_PALETTE_ENTRIES
        self.palette: list = []
        # Dominant species index elected by majority vote
        self.dominant_species_index: int = 0

    def sample_surrounding_cells(self, channels_dir: str) -> None:
        """
        Walk the six cardinal neighbour positions in grid space, read their
        published status.json + bbox.json channels, accumulate species/palette.

        Mirrors FAstroCellStyleProbe::SampleSurroundingCells() — six cardinal
        directions, species vote tally, palette fill.

        Grid step: cell_w for X, cell_h for Y, 1.0 for Z (layer index).
        Neighbours are identified by scanning cell/*/bbox.json files and
        checking whether their (x, y, z) centre falls within one grid step
        of this cell's centre in exactly one axis (cardinal, not diagonal).
        """
        self.palette.clear()
        self.dominant_species_index = 0

        cell_base = os.path.join(channels_dir, "cell")
        if not os.path.isdir(cell_base):
            return

        # Grid step sizes — mirrors CellSize = 100.f for X/Y; 1.0 for Z-layer
        step_x = max(self.cell_w, 1.0)
        step_y = max(self.cell_h, 1.0)
        step_z = 1.0

        # Self centre
        cx = self.world_x + self.cell_w / 2.0
        cy = self.world_y + self.cell_h / 2.0
        cz = self.world_z

        # Six cardinal offsets: (±step_x, 0, 0), (0, ±step_y, 0), (0, 0, ±step_z)
        # Mirrors Offsets[6] in the C++ SampleSurroundingCells.
        cardinal_offsets = [
            ( step_x,      0,      0),
            (-step_x,      0,      0),
            (      0,  step_y,     0),
            (      0, -step_y,     0),
            (      0,      0,  step_z),
            (      0,      0, -step_z),
        ]

        # Tolerance for "is this a cardinal neighbour?" check.
        # Mirrors FIntVector equality — we allow a small float tolerance.
        cardinal_tol = 0.5

        species_votes: dict = {}  # species_index → vote count; mirrors SpeciesVotes[256]
        max_votes: int = 0

        for sibling in os.listdir(cell_base):
            if sibling == self.cell_id:
                continue  # skip self

            bbox_path   = os.path.join(cell_base, sibling, "bbox.json")
            status_path = os.path.join(cell_base, sibling, "status.json")

            if not os.path.isfile(bbox_path):
                continue

            try:
                with open(bbox_path) as _f:
                    nbr_bbox = json.load(_f)
            except (json.JSONDecodeError, OSError):
                continue

            # Sibling centre
            nbr_cx = nbr_bbox["x"] + nbr_bbox["w"] / 2.0
            nbr_cy = nbr_bbox["y"] + nbr_bbox["h"] / 2.0
            nbr_cz = float(nbr_bbox.get("z", 0))

            # Check whether sibling is exactly one cardinal step away.
            # Mirrors FIntVector equality test on CenterCoord + Offsets[i].
            dx = nbr_cx - cx
            dy = nbr_cy - cy
            dz = nbr_cz - cz

            is_cardinal = False
            for (ox, oy, oz) in cardinal_offsets:
                if (abs(dx - ox) < cardinal_tol * step_x and
                        abs(dy - oy) < cardinal_tol * step_y and
                        abs(dz - oz) < cardinal_tol * max(step_z, 1.0)):
                    is_cardinal = True
                    break

            if not is_cardinal:
                continue  # not a direct neighbour — skip (mirrors Find() returning nullptr)

            # Read species from status.json if available; fall back to bbox.json field.
            nbr_species_name = nbr_bbox.get("species", "")
            if not nbr_species_name and os.path.isfile(status_path):
                try:
                    with open(status_path) as _f:
                        nbr_status = json.load(_f)
                    nbr_species_name = nbr_status.get("species", "")
                except (json.JSONDecodeError, OSError):
                    pass

            nbr_species_idx = _species_to_index(nbr_species_name)
            nbr_colour      = _SPECIES_INDEX_TO_COLOUR.get(nbr_species_idx,
                                                           _SPECIES_INDEX_TO_COLOUR[0])

            # Accumulate palette entry (up to MAX_PALETTE_ENTRIES).
            if len(self.palette) < self.MAX_PALETTE_ENTRIES:
                self.palette.append(nbr_colour)

            # Tally species vote — mirrors SpeciesVotes[SI]++
            species_votes[nbr_species_idx] = species_votes.get(nbr_species_idx, 0) + 1
            if species_votes[nbr_species_idx] > max_votes:
                max_votes = species_votes[nbr_species_idx]
                self.dominant_species_index = nbr_species_idx

    def blend_toward_neighbour_palette(
        self,
        own_colour: tuple,
        roughness: float = 0.5,
    ) -> tuple:
        """
        Nudge own_colour (r,g,b) toward the neighbourhood palette average.

        Direct port of FAstroCellStyleProbe::BlendWithCubemap():
          - Compute PaletteAvg from palette entries.
          - Smooth-step blend: smooth surfaces (low roughness) pull more
            strongly; rough surfaces resist and keep own character.
          - Scale blend by cell_style_weight (_STYLE_PROBE_WEIGHT = 0.20).

        @param own_colour  Cell's own primary colour as (r, g, b) floats [0,255].
        @param roughness   Visual roughness of this cell [0,1].  0 = sharp icon
                           (attaches hard to neighbour style); 1 = rough/noisy
                           (ignores neighbourhood almost entirely).
        @return            Blended (r, g, b) tuple, same scale.
        """
        if not self.palette:
            return own_colour   # no neighbours sampled — no-op (PaletteSize==0 path)

        # Accumulate weighted average palette colour — mirrors the palette loop.
        r_sum = sum(c[0] for c in self.palette)
        g_sum = sum(c[1] for c in self.palette)
        b_sum = sum(c[2] for c in self.palette)
        n = len(self.palette)
        palette_avg = (r_sum / n, g_sum / n, b_sum / n)

        # Smooth-step blend: smoother surfaces get stronger cell-style push.
        # At roughness=0 → full palette; at roughness=1 → no palette influence.
        # Mirrors: t = SmoothStep(0,1, 1-Roughness) * CellStyleWeight
        inv_r = max(0.0, min(1.0, 1.0 - roughness))
        # SmoothStep(0,1,x) = x*x*(3-2*x)
        smooth = inv_r * inv_r * (3.0 - 2.0 * inv_r)
        t = smooth * self.cell_style_weight

        return _lerp_colour(own_colour, palette_avg, t)




# ═══════════════════════════════════════════════




@dataclass


@dataclass


@dataclass


@dataclass
class AstroCellDrawSortKey:
    """
    Python equivalent of FMeshDrawCommandSortKey.

    Stores a 64-bit packed sort key used to order draw commands.  The key
    is split into two 32-bit halves:
        high = translucent sort distance (bit-inverted float for unsigned cmp)
        low  = PSO state key (pipeline hash — opaque draws only)

    Mirrors the union layout of FMeshDrawCommandSortKey::PackedData[2].

    鲁迅式：排序键是优先级的量化——一个数字决定了谁先被画，
    谁先被画决定了谁覆盖谁。先到不等于先赢，顺序才是权力。
    """
    high: int = 0   # translucent: bit-inverted distance; opaque: 0
    low:  int = 0   # PSO/state hash for opaque; secondary key for translucent

    @classmethod
    def default(cls) -> "AstroCellDrawSortKey":
        """Mirrors FMeshDrawCommandSortKey::Default = {{0}}."""
        return cls(high=0, low=0)

    def packed(self) -> int:
        """64-bit packed value: high in upper 32 bits, low in lower 32."""
        return ((self.high & 0xFFFFFFFF) << 32) | (self.low & 0xFFFFFFFF)

    def __lt__(self, other: "AstroCellDrawSortKey") -> bool:
        return self.packed() < other.packed()






# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================



# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================








# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================



# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================

class AstroCellPrimitiveIdBufferPool:
    """
    Python equivalent of FPrimitiveIdVertexBufferPool.

    Maintains a free-list of primitive-ID buffers sized by request, reusing
    existing allocations to avoid repeated allocation overhead.  A discard_id
    counter (incremented by discard_stale()) ages out entries that have been
    free for more than _STALE_EPOCH_THRESHOLD epochs.

    Mirrors the C++ Allocate / ReturnToFreeList / DiscardAll lifecycle:
        Allocate(size)         → allocate(size)
        ReturnToFreeList(entry)→ release(entry)
        DiscardAll()           → discard_stale()

    鲁迅式：缓冲池是节俭的哲学——内存不是免费的，
    重用已有的比每次申请新的，是对资源的尊重，也是对帧率的保护。
    """

    _ALIGN       = 1024      # BufferSize = Align(size, 1024)
    _STALE_EPOCH_THRESHOLD = 1000   # mirrors DiscardId > 1000 check

    def __init__(self) -> None:
        # Free-list: list of {"size": int, "data": bytearray, "last_discard_id": int}
        self._entries: list = []
        self._discard_id: int = 0

    # ------------------------------------------------------------------
    def allocate(self, size: int) -> dict:
        """
        Allocate (or reuse) a buffer of at least *size* bytes.

        Mirrors Allocate(FRHICommandList&, int32 BufferSize):
          - Align to 1024 bytes.
          - Find the smallest unused entry that fits (best-fit scan).
          - If none found, allocate a new bytearray.
          - Mark LastDiscardId = DiscardId on the returned entry.

        Returns a dict {"size": int, "data": bytearray, "last_discard_id": int}.

        鲁迅式：最佳适配是妥协中的智慧——找最小的够用者，不浪费，也不委屈。
        """
        aligned_size = ((size + self._ALIGN - 1) // self._ALIGN) * self._ALIGN

        best_idx = -1
        for i, entry in enumerate(self._entries):
            if entry["last_discard_id"] == self._discard_id:
                continue  # currently in use
            if entry["size"] >= aligned_size:
                if best_idx == -1 or entry["size"] < self._entries[best_idx]["size"]:
                    best_idx = i
                    if entry["size"] == aligned_size:
                        break

        if best_idx >= 0:
            reused = self._entries.pop(best_idx)
            reused["last_discard_id"] = self._discard_id
            return reused

        # Allocate new entry
        new_entry = {
            "size":           aligned_size,
            "data":           bytearray(aligned_size),
            "last_discard_id": self._discard_id,
        }
        return new_entry

    def release(self, entry: dict) -> None:
        """
        Return a buffer to the free list.
        Mirrors ReturnToFreeList() — thread-safe in C++ (mutex); here single-threaded.

        鲁迅式：归还是美德——用完即还，下一位不必等待。
        """
        self._entries.append(entry)

    def discard_stale(self) -> int:
        """
        Advance the discard epoch and evict buffers idle for too many epochs.

        Mirrors DiscardAll():
            ++DiscardId;
            RemoveAtSwap entries where (DiscardId - entry.LastDiscardId) > 1000

        Returns the number of entries evicted.

        鲁迅式：老化是自然定律——一千帧未被使用的缓冲区，
        不是在休息，是在占据本不属于它的位置。丢弃它，腾出空间给活着的事物。
        """
        self._discard_id += 1
        threshold = self._STALE_EPOCH_THRESHOLD
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if (self._discard_id - e["last_discard_id"]) <= threshold
        ]
        evicted = before - len(self._entries)
        if evicted:
            print(
                f"[AstroCellPrimitiveIdBufferPool] discard_stale: "
                f"evicted={evicted} discard_id={self._discard_id}",
                file=sys.stderr,
            )
        return evicted

    def stats(self) -> dict:
        """Diagnostic pool statistics."""
        return {
            "pool_entries":  len(self._entries),
            "discard_id":    self._discard_id,
            "total_bytes":   sum(e["size"] for e in self._entries),
        }


#: Module-level singleton pool — mirrors TGlobalResource<FPrimitiveIdVertexBufferPool>.
_ASTRO_PRIMITIVE_ID_BUFFER_POOL: AstroCellPrimitiveIdBufferPool = \
    AstroCellPrimitiveIdBufferPool()

















class AstroCellPipelineStateId:
    """
    Python equivalent of FGraphicsMinimalPipelineStateId.

    Assigns a stable integer ID to each (species, blend_mode, pass_name)
    combination, mirroring the persistent PSO ID table that survives across
    frames.  IDs are allocated lazily on first use and never reused.

    NeedsShaderInitialisation flag (mirrors the C++ static) is cleared the
    first time the table is populated — here it tracks whether any IDs have
    been assigned yet.

    鲁迅式：PSO 的 ID 是身份证——每一条渲染管线都有一个号码，
    号码不重复，也不作废。这是秩序对混乱的胜利。
    """

    NeedsShaderInitialisation: bool = True

    def __init__(self, species: str, blend_mode: str, pass_name: str) -> None:
        self.species    = species
        self.blend_mode = blend_mode
        self.pass_name  = pass_name
        self._id        = self._lookup_or_allocate()

    def _lookup_or_allocate(self) -> int:
        global _pipeline_state_next_id, _pso_table_frozen
        key = (self.species, self.blend_mode, self.pass_name)
        if key in _pipeline_state_table:
            return _pipeline_state_table[key]
        if _pso_table_frozen:
            # Mirrors the C++ assert that fires when table is frozen but a new
            # state is requested — here we log and return 0 (sentinel).
            print(
                f"[AstroCellPSOId] WARNING: PSO table frozen, "
                f"rejecting new state {key}.",
                file=sys.stderr,
            )
            return 0
        new_id = _pipeline_state_next_id
        _pipeline_state_table[key] = new_id
        _pipeline_state_next_id += 1
        AstroCellPipelineStateId.NeedsShaderInitialisation = False
        return new_id

    @property
    def id(self) -> int:
        return self._id

    def is_valid(self) -> bool:
        return self._id > 0 or (self._id == 0 and _pipeline_state_next_id > 0)

    @staticmethod
    def freeze_table() -> None:
        """Freeze the PSO table — no new states allowed after this point."""
        global _pso_table_frozen
        _pso_table_frozen = True

    @staticmethod
    def table_size() -> int:
        return len(_pipeline_state_table)

    def __repr__(self) -> str:
        return (f"AstroCellPipelineStateId("
                f"id={self._id}, species={self.species}, "
                f"blend={self.blend_mode}, pass={self.pass_name})")

















class AstroCellShaderBindings:
    """
    Python equivalent of the shader binding management from
    FReadOnlyMeshDrawSingleShaderBindings::SetShaderBindings().

    Tracks which uniform-buffer / texture / sampler / SRV slots have been
    written in the current draw call and skips redundant re-binds — exactly
    mirroring the FShaderBindingState delta-tracking logic.

    In the SVG substrate, «bindings» are SVG/CSS attribute overrides that
    must be accumulated before the final <g> element is emitted.  Redundant
    bindings from a previous cell with the same PSO are not re-emitted.

    鲁迅式：绑定状态是画家的调色板——
    每次切换颜色都有代价；不变的颜色就不要再调。
    ShaderBindingState 的存在，是对这一代价的精打细算。
    """

    _MAX_UNIFORM_BUFFERS = 16

    def __init__(self) -> None:
        # Mirrors FShaderBindingState — tracks last-bound values per slot.
        self._uniform_buffers: dict  = {}   # slot → value
        self._textures:        dict  = {}   # slot → value
        self._samplers:        dict  = {}   # slot → value
        self._srvs:            dict  = {}   # slot → value
        # Accumulated SVG attribute overrides for this draw call.
        self._svg_attr_overrides: dict = {}
        # Count of redundant binds skipped (diagnostic).
        self.redundant_binds_skipped: int = 0

    def bind_uniform_buffer(self, slot: int, value, svg_key: str = "") -> bool:
        """
        Bind a uniform buffer, skipping if value unchanged.
        Returns True if the binding was actually updated (not redundant).
        Mirrors the if (UniformBuffer != ShaderBindingState.UniformBuffers[...]) check.
        """
        if self._uniform_buffers.get(slot) == value:
            self.redundant_binds_skipped += 1
            return False
        self._uniform_buffers[slot] = value
        if svg_key:
            self._svg_attr_overrides[svg_key] = value
        return True

    def bind_texture(self, slot: int, texture_value, svg_key: str = "") -> bool:
        """Bind a texture slot (SetTextureParameter path)."""
        if self._textures.get(slot) == texture_value:
            self.redundant_binds_skipped += 1
            return False
        self._textures[slot] = texture_value
        if svg_key:
            self._svg_attr_overrides[svg_key] = texture_value
        return True

    def apply(self, base_svg_attrs: dict) -> dict:
        """
        Apply accumulated overrides onto base SVG attributes.

        Mirrors the post-SetShaderBindings() state where all per-cell
        material parameters have been applied to the draw pipeline.

        Returns a merged dict of SVG attributes with overrides applied.
        """
        merged = dict(base_svg_attrs)
        merged.update(self._svg_attr_overrides)
        return merged

    def reset(self) -> None:
        """Clear per-draw-call overrides (keep binding state for delta tracking)."""
        self._svg_attr_overrides.clear()

    def stats(self) -> dict:
        return {
            "uniform_buffers_bound":  len(self._uniform_buffers),
            "textures_bound":         len(self._textures),
            "redundant_binds_skipped": self.redundant_binds_skipped,
            "svg_overrides":          len(self._svg_attr_overrides),
        }


# ── MeshPassProcessor / SceneCapture / ReflectionCapture module constants ─────
# These mirror CVars / global flags that were referenced by previously ported
# code but whose definitions were omitted in earlier sessions.
ASTRO_EMIT_MESH_DRAW_EVENT:        bool  = False  # r.MeshDrawCommands.LogMeshDrawCommands
ASTRO_SKIP_DRAW_ON_PSO_PRECACHING: bool  = False  # r.SkipDrawOnPSOPrecaching
ASTRO_DEFERRED_MESH_PASS_SYNC:     bool  = True   # r.DeferredMeshPassSetupTaskSync
ASTRO_MESH_SORT_METHOD:            int   = 0      # 0=projZ, 1=axis, 2=distance
ASTRO_CAPTURE_ALLOW_MAIN_RENDERER: bool  = True   # r.SceneCapture.AllowRenderInMainRenderer
ASTRO_CAPTURE_CUBE_SINGLE_PASS:    bool  = False  # r.SceneCapture.CubeSinglePass
_REFL_TIMESLICE_FACES:             int   = 1      # CVarReflectionCaptureRuntimeTimeslice
_REFL_BUDGET:                      int   = 0      # 0 = unlimited
_REFL_SUPERSAMPLE_FACTOR:          int   = 1      # GSupersampleCaptureFactor
_REFL_RUNTIME_MODE:                int   = 0      # 0=continuous, 1=once

















class AstroCellMeshPassProcessor:
    """
    Python equivalent of FMeshPassProcessor.

    Processes a list of visible cell draw commands through the PSO lookup,
    shader binding, and sort-key assignment pipeline — exactly mirroring the
    three major responsibilities of FMeshPassProcessor:

    1. PSO key lookup (AddMeshDrawCommand path):
       For each cell, look up (or allocate) an AstroCellPipelineStateId based
       on (species, blend_mode, pass_name).  If ASTRO_SKIP_DRAW_ON_PSO_PRECACHING
       is True and the PSO is «new» (first frame), skip the draw entirely.

    2. Shader binding application (SetShaderBindings path):
       Create an AstroCellShaderBindings instance, populate it from the cell's
       gene_traits, apply to the base SVG attributes via bindings.apply().

    3. Sort key assignment (FMeshDrawCommandSortKey path):
       Opaque cells: sort key = PSO id (minimise state changes).
       Translucent cells: sort key = bit-inverted distance (painter's order).

    After process() the returned list is ready for AstroCellDrawList.

    鲁迅式：Processor 是流水线上的检验员——
    它不创造内容，但决定了内容能否进入下一道工序，
    以及以何种顺序进入。
    """

    def __init__(self,
                 pass_name:       str  = "base",
                 view_origin:     tuple = (0.0, 0.0, -1000.0),
                 emit_draw_events: bool = ASTRO_EMIT_MESH_DRAW_EVENT) -> None:
        self.pass_name        = pass_name
        self.view_origin      = view_origin
        self.emit_draw_events = emit_draw_events
        self._binding_state   = AstroCellShaderBindings()
        self._pso_cache:      dict = {}   # (species, blend_mode) → AstroCellPipelineStateId

    def _get_pso_id(self, species: str, blend_mode: str) -> AstroCellPipelineStateId:
        """Lookup or allocate a PSO id, caching within this pass."""
        key = (species, blend_mode)
        if key not in self._pso_cache:
            self._pso_cache[key] = AstroCellPipelineStateId(
                species, blend_mode, self.pass_name
            )
        return self._pso_cache[key]

    def _build_base_svg_attrs(self, entry: dict) -> dict:
        """
        Construct the base SVG attribute dict for a cell entry.
        Mirrors the material parameter packing that the C++ pass writes into
        the per-draw uniform buffer before shader binding.
        """
        species    = entry.get("species", "")
        bbox       = entry.get("bbox", {})
        opacity    = float(entry.get("opacity", 1.0))
        blend_mode = entry.get("blend_mode", "normal")

        sp_idx = _species_to_index(species)
        fill   = _colour_to_hex(_SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0]))
        stroke = fill

        return {
            "fill":        fill,
            "stroke":      stroke,
            "opacity":     str(round(opacity, 4)),
            "mix-blend-mode": blend_mode,
            "data-cell-id":   entry.get("cell_id", ""),
            "data-z":         str(bbox.get("z", 0)),
            "data-pass":      self.pass_name,
        }

    def _assign_sort_key(self, entry: dict, pso_id: AstroCellPipelineStateId,
                         blend_mode: str) -> AstroCellDrawSortKey:
        """
        Assign a sort key to the entry.

        Opaque (blend_mode not in translucent set):
            key.low = PSO id (minimise pipeline state switches, mirrors UE5 opaque sort).
            key.high = 0 (Z is irrelevant for opaque).

        Translucent:
            key = _compute_translucent_sort_key() (bit-inverted distance).
        """
        translucent_modes = {"translucent", "additive", "modulate", "alpha_composite"}
        if blend_mode in translucent_modes:
            bbox = entry.get("bbox", {})
            bx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
            by = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
            bz = float(bbox.get("z", 0))
            sort_policy = 1 if ASTRO_MESH_SORT_METHOD == 1 else 2
            return _compute_translucent_sort_key(
                (bx, by, bz), self.view_origin, [], sort_policy
            )
        else:
            # Opaque sort: primary key = PSO id (state minimisation)
            return AstroCellDrawSortKey(high=0, low=pso_id.id)

    def process(self, cell_entries: list) -> list:
        """
        Process all cell draw entries through the mesh pass pipeline.

        Mirrors FMeshPassProcessor::AddMeshBatch() + BuildMeshDrawCommands()
        called for each FMeshBatch in the view's visible primitives list.

        For each entry:
          1. Resolve blend_mode.
          2. Lookup PSO id (skip if precaching + new PSO).
          3. Apply shader bindings.
          4. Assign sort key.
          5. Append draw-event annotation if ASTRO_EMIT_MESH_DRAW_EVENT.

        Returns list of enriched entry dicts with fields added:
            pso_id      : int
            sort_key    : AstroCellDrawSortKey
            svg_attrs   : dict (final merged SVG attributes)
            draw_event  : str (optional debug annotation)

        鲁迅式：process() 是流水线的主干——
        所有输入在这里经过筛选、分类、标记，最终成为可以被画出来的命令。
        没有经过 process() 的 cell，不过是一堆原始数据；
        经过之后，它们获得了身份、顺序和形式。
        """
        result = []
        self._binding_state.reset()

        for entry in cell_entries:
            cell_id    = entry.get("cell_id", "")
            species    = entry.get("species", "")
            blend_mode = entry.get("blend_mode", "normal")

            # ── PSO lookup ────────────────────────────────────────────────────
            pso = self._get_pso_id(species, blend_mode)

            if ASTRO_SKIP_DRAW_ON_PSO_PRECACHING and pso.NeedsShaderInitialisation:
                # PSO still «compiling» — skip this draw call
                print(
                    f"[AstroCellMeshPassProcessor] SkipDrawOnPSOPrecaching: "
                    f"skipping cell={cell_id} (PSO not yet initialised)",
                    file=sys.stderr,
                )
                continue

            # ── Shader bindings ───────────────────────────────────────────────
            self._binding_state.reset()
            # Bind per-cell gene_traits as «uniform buffer» slot 0
            gene_traits = entry.get("gene_traits", {})
            self._binding_state.bind_uniform_buffer(
                slot=0, value=json.dumps(gene_traits, sort_keys=True)
            )
            # Bind species colour as «texture» slot 0
            sp_idx = _species_to_index(species)
            fill   = _colour_to_hex(_SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0]))
            self._binding_state.bind_texture(
                slot=0, texture_value=fill, svg_key="fill"
            )

            base_attrs = self._build_base_svg_attrs(entry)
            svg_attrs  = self._binding_state.apply(base_attrs)

            # ── Sort key ──────────────────────────────────────────────────────
            sort_key = self._assign_sort_key(entry, pso, blend_mode)

            # ── Draw event annotation ─────────────────────────────────────────
            draw_event = ""
            if self.emit_draw_events:
                draw_event = (
                    f"<!-- [ASTRO-MDC] MeshDrawEvent pass={self.pass_name} "
                    f"cell={cell_id} pso_id={pso.id} "
                    f"sort={sort_key.packed()} -->"
                )

            enriched = dict(entry)
            enriched.update({
                "pso_id":     pso.id,
                "sort_key":   sort_key,
                "svg_attrs":  svg_attrs,
                "draw_event": draw_event,
            })
            result.append(enriched)

        # ── Deferred sort (mirrors CVarDeferredMeshPassSetupTaskSync) ─────────
        if ASTRO_DEFERRED_MESH_PASS_SYNC:
            result.sort(key=lambda e: e["sort_key"].packed())

        print(
            f"[AstroCellMeshPassProcessor] process: "
            f"pass={self.pass_name} in={len(cell_entries)} "
            f"out={len(result)} "
            f"pso_table_size={AstroCellPipelineStateId.table_size()} "
            f"redundant_binds={self._binding_state.redundant_binds_skipped}",
            file=sys.stderr,
        )

        return result


# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================



# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================



# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================



# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================




@dataclass


@_ptdc


@dataclass


@_ptdc
class AstroCellPathTracingConfig:
    """
    Python equivalent of FPathTracingConfig.

    Holds all scene-level parameters that, if changed, require restarting the
    sample accumulation (SampleIndex reset).  The is_different() method mirrors
    FPathTracingConfig::IsDifferent() which guards the invalidation call in the
    C++ render loop.

    鲁迅式：参数是约定，约定一旦改变，积累的历史便成无效的遗产。
    """
    max_samples:           int   = _PT_MAX_SAMPLES
    max_bounces:           int   = _PT_MAX_BOUNCES
    filter_sigma:          float = _PT_FILTER_SIGMA
    mis_mode:              int   = _PT_MIS_MODE
    max_path_intensity:    float = _PT_MAX_PATH_INTENSITY
    approximate_caustics:  bool  = _PT_APPROXIMATE_CAUSTICS
    adaptive_threshold:    float = _PT_ADAPTIVE_THRESHOLD
    locked_sampling:       bool  = _PT_LOCKED_SAMPLING
    # Viewport rect (mirrors FIntRect ViewRect in FPathTracingConfig)
    viewport_w:            int   = 1200
    viewport_h:            int   = 900
    # Light grid (mirrors LightGridResolution / LightGridMaxCount)
    light_grid_resolution: int   = 8
    light_grid_max_count:  int   = 64
    # Background / atmosphere flags
    enable_emissive:       bool  = True
    background_alpha:      float = 1.0

    def is_different(self, other: "AstroCellPathTracingConfig") -> bool:
        """
        Returns True if any accumulation-invalidating parameter changed.
        Mirrors FPathTracingConfig::IsDifferent() — guards SampleIndex reset.

        鲁迅式：只有真正不同的，才值得重新开始。
        """
        return (
            self.max_samples           != other.max_samples          or
            self.max_bounces           != other.max_bounces          or
            abs(self.filter_sigma      -  other.filter_sigma)        > 1e-5 or
            self.mis_mode              != other.mis_mode             or
            abs(self.max_path_intensity - other.max_path_intensity)  > 1e-5 or
            self.approximate_caustics  != other.approximate_caustics or
            abs(self.adaptive_threshold - other.adaptive_threshold)  > 1e-7 or
            self.locked_sampling       != other.locked_sampling      or
            self.viewport_w            != other.viewport_w           or
            self.viewport_h            != other.viewport_h           or
            self.light_grid_resolution != other.light_grid_resolution or
            self.light_grid_max_count  != other.light_grid_max_count or
            self.enable_emissive       != other.enable_emissive      or
            abs(self.background_alpha  -  other.background_alpha)    > 1e-5
        )


@_ptdc


@_ptdc


@_ptdc


@_ptdc



@_ptdc


@_ptdc


@_ptdc


@_ptdc
class AstroCellPathTracingState:
    """
    Python equivalent of FPathTracingState.

    Stores per-view accumulated path tracing data between frames.
    Key invariant (from C++ comment):
        FrameIndex is NEVER reset on invalidation to avoid the temporal
        "screen door" effect caused by the quasi-random sampler re-using
        the same low-discrepancy sequence from frame 0.
        SampleIndex IS reset on invalidation so accumulation restarts cleanly.

    Buffers are dicts keyed by cell_id; values are float tuples (R, G, B).
    This mirrors the per-pixel texture arrays in the C++ implementation.

    鲁迅式：样本指数归零是承认失败，但帧指数不能归零——
    否则时间便失去了意义，历史便成了永恒的循环。
    """
    last_config:    AstroCellPathTracingConfig = _ptfield(
        default_factory=AstroCellPathTracingConfig)
    # Accumulated radiance buffer (mirrors RadianceRT)
    radiance_buffer:    _PTDict[str, tuple] = _ptfield(default_factory=dict)
    # Per-cell variance estimate (mirrors VarianceRT / VarianceBuffer)
    variance_buffer:    _PTDict[str, float] = _ptfield(default_factory=dict)
    # Denoiser AOV buffers (mirrors AlbedoRT, NormalRT, DepthRT)
    albedo_buffer:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    normal_buffer:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    depth_buffer:       _PTDict[str, float] = _ptfield(default_factory=dict)
    # Last denoised frame cache (mirrors LastDenoisedRadianceRT — animation stability)
    last_denoised:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    # Sample counter: reset to 0 on invalidation (mirrors SampleIndex)
    sample_index:       int = 0
    # Frame counter: NEVER reset (mirrors FrameIndex — uint32_t, monotone)
    frame_index:        int = 0

    def invalidate(self) -> None:
        """
        Reset accumulated data and sample counter.
        Mirrors FSceneViewState::PathTracingInvalidate(bInvalidateAnimationStates=false).

        FrameIndex is intentionally NOT touched — see struct docstring.

        鲁迅式：将一切清零，唯独不清零时间——这是纪律，不是遗忘。
        """
        self.radiance_buffer.clear()
        self.variance_buffer.clear()
        self.albedo_buffer.clear()
        self.normal_buffer.clear()
        self.depth_buffer.clear()
        # last_denoised intentionally kept (mirrors C++ which keeps LastDenoisedRadianceRT)
        self.sample_index = 0
        print(
            f"[ASTRO-PT] PathTracingInvalidate — sample_index reset to 0, "
            f"frame_index preserved at {self.frame_index}",
            file=sys.stderr,
        )

    def is_converged(self, cell_id: str,
                     threshold: float = _PT_ADAPTIVE_THRESHOLD) -> bool:
        """
        Per-cell convergence check — mirrors the adaptive sampling variance gate
        in the C++ path tracer that skips already-converged pixels.
        Returns True when variance is below threshold and enough samples accumulated.
        """
        if self.sample_index < 4:
            return False
        return self.variance_buffer.get(cell_id, 1.0) < threshold


# ── Module-level per-view state registry ─────────────────────────────────────
# Mirrors the FViewState::PathTracingState TPimplPtr<FPathTracingState> member.
# Keyed by view_id string (for single-view usage, use key "default").
_CELL_PATH_TRACING_STATES: _PTDict[str, AstroCellPathTracingState] = {}

















def _pt_sample_cell_radiance(
    cell_id: str,
    bbox: dict,
    species: str,
    sample_idx: int,
    frame_idx: int,
    bvh: "AstroCellBVH | None" = None,
    all_bboxes: dict | None = None,
    mis_mode: int = _PT_MIS_MODE,
    max_bounces: int = _PT_MAX_BOUNCES,
) -> tuple:
    """
    Per-cell path sample — the inner loop of RenderPathTracing().

    Replaces the full GPU ray-traced path with an analytic 2-D equivalent:
      - Primary ray hits the cell's own bbox (always; we shade the cell itself)
      - Each bounce samples a neighbour cell's emissive contribution via BVH
        overlap query (analogue of BVH traversal + BSDF evaluation)
      - MIS combines material PDF (uniform hemisphere) and light PDF (area / dist²)
      - Firefly clamp applied per bounce

    The function uses per-sample Halton sequences indexed by
    (sample_idx * max_bounces + bounce, prime) to maintain low-discrepancy
    stratification across samples and bounces — same as the C++ path tracer's
    per-path quasi-random state.

    Returns (R, G, B) radiance contribution from one path sample.

    鲁迅式：每一条路径都是一次反问——光从哪里来？
    到哪里去？会不会在中途被遮挡、被散射、被彻底消灭？
    答案藏在概率密度函数里，与宿命无关。
    """
    import math as _ptm

    # ── Species emissive base (「primary ray hit self」) ────────────────────
    # Mirrors the path tracer's direct-hit emissive contribution (bounce 0).
    # We derive a per-species base colour as the emissive seed — same as
    # treating the cell face as an emissive surface in the material graph.
    _EMISSIVE_TABLE = {
        "cil-eye":         (0.55, 0.60, 0.90),  # indigo glow
        "cil-bolt":        (0.95, 0.55, 0.10),  # amber spark
        "cil-vector":      (0.30, 0.70, 0.35),  # green signal
        "cil-plus":        (0.25, 0.55, 0.90),  # blue merge
        "cil-arrow-right": (0.50, 0.60, 0.65),  # grey-blue arrow
        "cil-filter":      (0.60, 0.25, 0.75),  # purple kernel
        "cil-code":        (0.30, 0.70, 0.35),  # green brace
        "cil-layers":      (0.20, 0.55, 0.85),  # blue stack
        "cil-loop":        (0.90, 0.60, 0.15),  # amber cycle
        "cil-graph":       (0.40, 0.50, 0.55),  # grey node
    }
    base_r, base_g, base_b = _EMISSIVE_TABLE.get(species, (0.5, 0.5, 0.5))

    # ── Halton quasi-random state for this path ────────────────────────────
    # Path seed combines sample_index × bounce_depth for decorrelation.
    # Mirrors PathTracer.usf RandomSequence_Initialize(Seed = SampleIndex * MaxBounces).
    seed_base = sample_idx * max_bounces

    path_r, path_g, path_b = base_r, base_g, base_b
    throughput = 1.0

    cx = bbox["x"] + bbox["w"] / 2.0
    cy = bbox["y"] + bbox["h"] / 2.0
    cz = float(bbox.get("z", 3))

    for bounce in range(max_bounces):
        if throughput < 1e-4:
            break   # Russian roulette termination (implicit, energy threshold)

        seed = seed_base + bounce
        u1   = _pt_halton(seed * 2 + frame_idx % 97, 2)   # azimuth
        u2   = _pt_halton(seed * 2 + 1 + frame_idx % 97, 3)   # elevation

        # ── Material sampling: cosine-weighted hemisphere direction ────────
        # Mirrors the Lambertian BSDF material sampling in PathTracing.usf.
        theta_mat = math.acos(math.sqrt(max(0.0, u2)))
        phi_mat   = 2.0 * math.pi * u1
        pdf_mat   = math.cos(theta_mat) / math.pi  # Lambertian PDF

        # ── Light sampling: pick a neighbour cell as area light ────────────
        # Mirrors the light sampling step in the path tracer's MIS loop.
        # We use the BVH (if available) for a spatial query; else fall back
        # to the all_bboxes dict.
        light_cell_id: str | None = None
        light_r, light_g, light_b = 0.0, 0.0, 0.0
        pdf_light = 0.0

        if bvh is not None:
            candidates = bvh.query_overlapping_cells({
                "x": cx - bbox["w"],  "y": cy - bbox["h"],
                "w": bbox["w"] * 2,   "h": bbox["h"] * 2,
            })
        elif all_bboxes:
            candidates = list(all_bboxes.keys())
        else:
            candidates = []

        # Filter out self; pick one candidate by quasi-random index
        candidates = [c for c in candidates if c != cell_id]
        if candidates:
            pick_idx   = int(u1 * len(candidates)) % len(candidates)
            light_cell_id = candidates[pick_idx]
            lb = all_bboxes.get(light_cell_id, {}) if all_bboxes else {}
            if lb:
                lx = lb.get("x", cx) + lb.get("w", 80) / 2.0
                ly = lb.get("y", cy) + lb.get("h", 50) / 2.0
                lz = float(lb.get("z", cz))
                dist_sq = max((cx-lx)**2 + (cy-ly)**2 + (cz-lz)**2 * 10000, 1.0)
                area    = lb.get("w", 80) * lb.get("h", 50)
                pdf_light = 1.0 / (len(candidates) * area / dist_sq)  # area light PDF

                # Light colour from emissive table
                lsp = lb.get("species", "cil-arrow-right")
                light_r, light_g, light_b = _EMISSIVE_TABLE.get(lsp, (0.5, 0.5, 0.5))

        # ── MIS weight (balanced power heuristic, β=2) ────────────────────
        if mis_mode == 2 and pdf_light > 0.0:
            # MIS mode 2: combine material + light sampling
            w_mat   = _pt_mis_weight(pdf_mat,   pdf_light)
            w_light = _pt_mis_weight(pdf_light, pdf_mat)
            # Throughput contribution from MIS combination
            contrib_r = (path_r * w_mat + light_r * w_light) * throughput
            contrib_g = (path_g * w_mat + light_g * w_light) * throughput
            contrib_b = (path_b * w_mat + light_b * w_light) * throughput
        elif mis_mode == 1 and pdf_light > 0.0:
            # MIS mode 1: light sampling only
            contrib_r = light_r * throughput
            contrib_g = light_g * throughput
            contrib_b = light_b * throughput
        else:
            # MIS mode 0 or no light: material sampling only
            contrib_r = path_r * throughput
            contrib_g = path_g * throughput
            contrib_b = path_b * throughput

        # ── Firefly clamp per bounce ───────────────────────────────────────
        contrib_r, contrib_g, contrib_b = _pt_firefly_clamp(
            (contrib_r, contrib_g, contrib_b), _PT_MAX_PATH_INTENSITY)

        # ── Caustic approximation gate ─────────────────────────────────────
        # When ApproximateCaustics=True, clamp specular contribution on diffuse
        # surfaces to reduce noise from low-roughness indirect paths.
        # Mirrors the C++ caustic approximation that clamps glossy→diffuse paths.
        if _PT_APPROXIMATE_CAUSTICS and bounce > 0:
            contrib_r *= 0.25
            contrib_g *= 0.25
            contrib_b *= 0.25

        path_r = contrib_r
        path_g = contrib_g
        path_b = contrib_b

        # ── Throughput update (Russian roulette) ──────────────────────────
        # Mirrors the path tracer's per-bounce throughput × albedo update.
        albedo_avg = (base_r + base_g + base_b) / 3.0
        throughput *= max(0.0, min(1.0, albedo_avg * math.cos(theta_mat)))

    return (max(0.0, path_r), max(0.0, path_g), max(0.0, path_b))




# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────





# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────






# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────





# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────

def temporal_reprojection_align(
    radiance_buf:   _PTDict[str, tuple],
    history_buf:    _PTDict[str, tuple],
    motion_vectors: _PTDict[str, tuple],  # cell_id → (dz,) displacement
) -> _PTDict[str, tuple]:
    """
    Temporal reprojection alignment pass.
    Mirrors FTemporalReprojectionAlignCS: warps history to current frame using
    per-cell motion vectors.

    In 2-D, motion is only in the Z axis (z-layer transitions); X/Y do not change
    between frames in the pub/sub epoch model.  The warp is a z-layer index
    lookup: if cell moved from z_prev to z_curr, copy its history entry
    (no blending needed for integer z-layer steps).

    Returns a dict of aligned history radiance (same structure as radiance_buf).

    鲁迅式：时间的对齐是第一步——如果你无法找到上一帧的位置，
    历史便是别人的历史，与你无关。
    """
    aligned: _PTDict[str, tuple] = {}
    for cell_id, rad in radiance_buf.items():
        mv = motion_vectors.get(cell_id, (0.0,))
        dz = mv[0] if mv else 0.0
        if abs(dz) < 0.5:
            # No significant motion — use history directly (fast path)
            aligned[cell_id] = history_buf.get(cell_id, rad)
        else:
            # Cell moved to a new z-layer: history is stale; restart from current
            # (mirrors the C++ path where large motion vectors cause history rejection)
            aligned[cell_id] = rad
    return aligned

















def temporal_reprojection_blur(
    aligned_history: _PTDict[str, tuple],
    bvh:             "AstroCellBVH | None",
    all_bboxes:      dict,
    blur_radius:     float = 1.0,
) -> _PTDict[str, tuple]:
    """
    Temporal reprojection blur pass.
    Mirrors FTemporalReprojectionBlurCS: applies a small spatial blur to the
    aligned history to reduce temporal ghosting from mis-aligned history.

    2-D adaptation: BVH spatial query fetches immediate neighbours; their
    history values are averaged as a 3-tap bilateral kernel weighted by
    distance (analogue of the C++ screen-space 3×1 separable blur kernel).

    鲁迅式：模糊是宽容，是允许错误存在的制度——
    但宽容过度便是纵容，故 blur_radius 不宜过大。
    """
    blurred: _PTDict[str, tuple] = {}
    for cell_id, hist in aligned_history.items():
        bbox = all_bboxes.get(cell_id, {})
        if not bbox or bvh is None:
            blurred[cell_id] = hist
            continue

        # Spatial neighbourhood from BVH
        nbrs = bvh.query_overlapping_cells({
            "x": bbox.get("x", 0) - blur_radius * bbox.get("w", 80),
            "y": bbox.get("y", 0) - blur_radius * bbox.get("h", 50),
            "w": bbox.get("w", 80) * (1 + 2 * blur_radius),
            "h": bbox.get("h", 50) * (1 + 2 * blur_radius),
        })
        nbr_hists = [aligned_history[n] for n in nbrs
                     if n != cell_id and n in aligned_history]

        if nbr_hists:
            # Simple average (bilateral weights omitted — analytic context)
            avg_r = (hist[0] + sum(h[0] for h in nbr_hists)) / (len(nbr_hists) + 1)
            avg_g = (hist[1] + sum(h[1] for h in nbr_hists)) / (len(nbr_hists) + 1)
            avg_b = (hist[2] + sum(h[2] for h in nbr_hists)) / (len(nbr_hists) + 1)
            blurred[cell_id] = (avg_r, avg_g, avg_b)
        else:
            blurred[cell_id] = hist

    return blurred

















def temporal_reprojection_merge(
    current_radiance: _PTDict[str, tuple],
    blurred_history:  _PTDict[str, tuple],
    variance_buf:     _PTDict[str, float],
    base_weight:      float = _PTD_TEMPORAL_WEIGHT,
) -> _PTDict[str, tuple]:
    """
    Temporal accumulation merge pass.
    Mirrors FTemporalReprojectionMergeCS: blends current frame with history.

    The history weight is modulated by per-cell variance:
        w_hist = base_weight × clamp(1 − variance / variance_max, 0, 1)
    Low-variance (converged) cells keep more history; high-variance cells
    (still noisy) accept more current-frame data — same as the C++
    TotalVariation permutation of the merge shader.

    鲁迅式：过去与现在的混合比例，取决于现在有多嘈杂——
    越嘈杂，越需要历史来压制；越平静，历史越可以安全保留。
    """
    merged: _PTDict[str, tuple] = {}
    variance_max = max(variance_buf.values()) if variance_buf else 1.0
    variance_max = max(variance_max, 1e-6)

    for cell_id, curr in current_radiance.items():
        hist  = blurred_history.get(cell_id, curr)
        var   = variance_buf.get(cell_id, 1.0)
        # Variance-adaptive weight
        w_hist = base_weight * max(0.0, min(1.0, 1.0 - var / variance_max))
        w_curr = 1.0 - w_hist
        merged[cell_id] = (
            curr[0] * w_curr + hist[0] * w_hist,
            curr[1] * w_curr + hist[1] * w_hist,
            curr[2] * w_curr + hist[2] * w_hist,
        )
    return merged

















def high_frequency_reject_map(
    radiance_buf:     _PTDict[str, tuple],
    last_denoised:    _PTDict[str, tuple],
    variance_buf:     _PTDict[str, float],
    reject_threshold: float = 0.15,
) -> _PTDict[str, float]:
    """
    High-frequency reject map pass.
    Mirrors FTemporalHighFrequencyRejectMapCS: generates a per-cell mask
    [0, 1] where 1 = accept current (high-frequency / newly appeared feature)
    and 0 = reject current (temporal ghost / noise spike).

    Implemented as luminance-delta comparison against last denoised frame:
        delta_lum = |lum_current − lum_last_denoised| / max(lum_last_denoised, 1e-4)
        accept    = 1 if delta_lum < reject_threshold and variance < threshold
                    0 otherwise (clamp to [0, 1])

    鲁迅式：高频拒绝图是防伪标记——真实的光照变化缓慢，突变是噪声的证据。
    但拒绝必须谨慎，过于激进的拒绝会抹平真实的变化，造成「滞后」的幽灵。
    """
    accept_map: _PTDict[str, float] = {}
    for cell_id, curr in radiance_buf.items():
        lum_curr = (curr[0] + curr[1] + curr[2]) / 3.0
        last = last_denoised.get(cell_id)
        if last is None:
            accept_map[cell_id] = 1.0
            continue
        lum_last = (last[0] + last[1] + last[2]) / 3.0
        delta    = abs(lum_curr - lum_last) / max(lum_last, 1e-4)
        var      = variance_buf.get(cell_id, 1.0)
        if delta < reject_threshold and var < _PT_ADAPTIVE_THRESHOLD * 10:
            accept_map[cell_id] = 0.0   # accept history (reject current spike)
        else:
            accept_map[cell_id] = 1.0   # accept current (genuine change)
    return accept_map

















def temporal_feature_fusion(
    merged_radiance: _PTDict[str, tuple],
    accept_map:      _PTDict[str, float],
    last_denoised:   _PTDict[str, tuple],
) -> _PTDict[str, tuple]:
    """
    Temporal feature fusion pass.
    Mirrors FTemporalFeatureFusionCS: final per-cell combination of merged
    radiance with last-denoised history, gated by the accept_map.

    accept_map[cell_id] == 1.0 → use merged_radiance (fresh data)
    accept_map[cell_id] == 0.0 → blend toward last_denoised (temporal stability)

    鲁迅式：融合是最后的抉择——在新与旧之间，在清晰与稳定之间，
    accept_map 是那唯一的判官，不偏不倚（除非偶尔被数学愚弄）。
    """
    fused: _PTDict[str, tuple] = {}
    for cell_id, merged in merged_radiance.items():
        a     = accept_map.get(cell_id, 1.0)
        last  = last_denoised.get(cell_id, merged)
        fused[cell_id] = (
            merged[0] * a + last[0] * (1.0 - a),
            merged[1] * a + last[1] * (1.0 - a),
            merged[2] * a + last[2] * (1.0 - a),
        )
    return fused

















def find_or_allocate_cubemap_index(cell_id: str) -> int:
    """
    Allocate or return the existing cubemap slot for *cell_id*.

    Mirrors FindOrAllocateCubemapIndex() from ReflectionEnvironmentCapture.cpp:
        CaptureSceneStatePtr = Scene.ReflectionSceneData.AllocatedReflectionCaptureState
                                .AddReference(Component)
        if (!CaptureSceneStatePtr): allocate new slot

    Returns cubemap_index ∈ [0, max_cubemaps) or -1 on overflow.

    鲁迅式：分配是有限的资源与无限的需求之间的妥协——
    64 个探针槽，比大多数场景需要的多；
    但若场景足够野心勃勃，终有耗尽的一天。
    """
    scene = get_reflection_scene_data()
    if cell_id in scene.allocated_captures:
        return scene.allocated_captures[cell_id].cubemap_index

    if scene.next_cubemap_slot >= scene.max_cubemaps:
        print(
            f"[ASTRO-CAPTURE] WARNING: cubemap array full "
            f"({scene.max_cubemaps} slots) — cannot allocate for cell={cell_id}",
            file=sys.stderr,
        )
        return -1

    idx = scene.next_cubemap_slot
    scene.next_cubemap_slot += 1
    state = AstroCellCaptureState(cubemap_index=idx, cell_id=cell_id)
    scene.allocated_captures[cell_id] = state

    print(
        f"[ASTRO-CAPTURE] FindOrAllocateCubemapIndex: "
        f"cell_id={cell_id} slot={idx} "
        f"total_allocated={len(scene.allocated_captures)}",
        file=sys.stderr,
    )
    return idx

















def gaussian_downsample_face_mip(
    face_colour: tuple,
    mip_level:   int,
    sigma_scale: float = 0.8,
) -> tuple:
    """
    Per-mip Gaussian downsample of a cubemap face colour.
    Mirrors FDownsampleCubeFaceCS (DownsampleCS in ReflectionEnvironmentShaders.usf):
        Each mip halves the resolution and blurs with a 3×3 Gaussian kernel.
        Energy is conserved (sum of Gaussian weights = 1).

    2-D adaptation: operates on a single (R, G, B) float tuple representing
    the average colour of a face at the given mip level.  The Gaussian kernel
    is replaced by an exponential decay on luminance (mirrors the energy loss
    at higher mip levels where the specular lobe widens).

    Returns the downsampled face colour at *mip_level*.

    鲁迅式：Mip 层是宽容的代价——越高的 mip，细节越少，也越不刺眼。
    这是视觉的让步，也是性能的胜利。
    """
    # Gaussian decay factor per mip: each level loses sigma_scale of sharpness
    decay = math.exp(-0.5 * (mip_level * sigma_scale) ** 2)
    # Blend toward mid-grey (0.5, 0.5, 0.5) at higher mips — mirrors the
    # BRDF integration limit where fully rough surfaces → uniform hemisphere
    mid = 0.5
    return (
        face_colour[0] * decay + mid * (1.0 - decay),
        face_colour[1] * decay + mid * (1.0 - decay),
        face_colour[2] * decay + mid * (1.0 - decay),
    )

















def convolve_specular_face(
    face_colour:   tuple,
    mip_level:     int,
    roughness:     float = 0.5,
    num_mips:      int   = _CAPTURE_NUM_MIPS,
) -> tuple:
    """
    Per-face specular convolution (pre-filtered environment map).
    Mirrors FConvolveSpecularFaceCS (FilterCS in ReflectionEnvironmentShaders.usf):
        Integrates the GGX BSDF lobe over the cubemap face weighted by
        the mip-level roughness mapping:
            perceptual_roughness = mip / (num_mips - 1)
            alpha = perceptual_roughness²   (GGX alpha = roughness²)

    2-D adaptation: approximates the convolution result as a roughness-weighted
    blend between the specular (sharp) face colour and a diffuse (isotropic) grey.
    The GGX NDF width increases with roughness — the highest mip approximates
    a Lambertian hemisphere integral (uniform over all directions → grey).

    Returns pre-filtered specular colour for this face + mip combination.

    鲁迅式：预滤波是先见之明——把所有可能的粗糙度预先计算好，
    运行时只需查表，不必每次重新积分。这是懒惰，也是智慧。
    """
    perceptual_roughness = mip_level / max(num_mips - 1, 1)
    alpha = perceptual_roughness * perceptual_roughness

    # GGX lobe weight: sharper lobe at low roughness → more face colour;
    # wider lobe at high roughness → blend toward isotropic grey
    lobe_w = max(0.0, 1.0 - alpha)
    iso_w  = alpha

    lum   = (face_colour[0] + face_colour[1] + face_colour[2]) / 3.0
    grey  = (lum, lum, lum)

    prefiltered = (
        face_colour[0] * lobe_w + grey[0] * iso_w,
        face_colour[1] * lobe_w + grey[1] * iso_w,
        face_colour[2] * lobe_w + grey[2] * iso_w,
    )
    return prefiltered

















def query_specular_radiance(
    cell_id:      str,
    roughness:    float = 0.5,
    face_index:   int   = 4,   # default +Z face (「上方天空」)
) -> tuple:
    """
    Query the pre-filtered specular environment for *cell_id*.

    Maps *roughness* to the appropriate mip level using the Nanite LOD
    metric analogue:
        mip = round(roughness × (num_mips − 1))
    Returns the prefiltered (R, G, B) specular colour × fade_alpha.

    Mirrors the specular environment probe lookup performed in the
    reflection capture material shader (GetOffSpecularPeakReflectionDir +
    texCUBElod call in ReflectionEnvironmentShared.usf).

    鲁迅式：查询是坦然的索取——环境贴图积累好了，
    谁需要，谁就来取，不必客气，不必感谢。
    """
    scene   = get_reflection_scene_data()
    capture = scene.allocated_captures.get(cell_id)
    if capture is None or not capture.specular_prefilter:
        return (0.5, 0.5, 0.5)   # default sky grey

    mip      = int(round(roughness * (_CAPTURE_NUM_MIPS - 1)))
    mip      = max(0, min(mip, _CAPTURE_NUM_MIPS - 1))
    spec_col = capture.specular_prefilter.get(mip, (0.5, 0.5, 0.5))
    alpha    = capture.fade_alpha

    return (
        spec_col[0] * alpha,
        spec_col[1] * alpha,
        spec_col[2] * alpha,
    )


# =============================================================================
# [ASTRO-CELL] ReflectionEnvironmentRealTimeCapture → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentRealTimeCapture.cpp
#
# 「世界上本没有实时反射，用的人多了，也便有了实时反射。」——鲁迅（改写）
#
# ReflectionEnvironmentRealTimeCapture 实现了「实时」天光捕获：
# 每帧分时渲染一个或多个 cube face，逐渐积累完整的 sky env map，
# 再对其执行 downsample + convolve + diffuse SH 通道，
# 并通过 bRealTimeCaptureEnabled 标志按需触发。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarRealTimeReflectionCaptureTimeSlicing   → ASTRO_RT_CAPTURE_TIMESLICE
#   CVarRealTimeReflectionCaptureTimeSlicingSkyCloudCubeFacePerFrame
#                                              → ASTRO_RT_CAPTURE_CLOUD_FACES
#   CVarRealTimeReflectionCaptureShadowFromOpaque → ASTRO_RT_SHADOW_FROM_OPAQUE
#   CVarRealTimeReflectionCaptureDepthBuffer   → ASTRO_RT_DEPTH_BUFFER
#   CVarRealTimeReflectionCaptureVolumetricCloudResolutionDivider
#                                              → ASTRO_RT_CLOUD_RES_DIVIDER
#   FRealTimeSlicedReflectionCapture           → AstroCellRealTimeSkyCapture
#   RenderSkyPassForCapture                    → render_sky_pass_for_capture()
#   UpdateSkyEnvMap                            → update_sky_env_map()
#   ValidateSkyLightRealTimeCapture            → validate_sky_light_rt_capture()
# =============================================================================

# CVarRealTimeReflectionCapture equivalents
ASTRO_RT_CAPTURE_TIMESLICE:    bool  = True   # r.SkyLight.RealTimeReflectionCapture.TimeSlice
ASTRO_RT_CAPTURE_CLOUD_FACES:  int   = 2      # faces per frame for cloud
ASTRO_RT_SHADOW_FROM_OPAQUE:   bool  = False  # opaque mesh shadow in capture
ASTRO_RT_DEPTH_BUFFER:         bool  = True   # depth-aware capture
ASTRO_RT_CLEAR_COLOR:          bool  = False  # always clear colour buffer
ASTRO_RT_CLOUD_RES_DIVIDER:    int   = 2      # cloud resolution divider
ASTRO_RT_RES_OVERRIDE:         int   = 0      # 0 = default resolution (128)
ASTRO_RT_DEFAULT_CUBE_SIZE:    int   = 128    # default sky capture cube resolution


@_ptdc




@_ptdc





@_ptdc




@_ptdc


def update_sky_env_map(
    all_bboxes:     dict | None = None,
    atmosphere:     tuple = (0.55, 0.68, 0.82),
    cloud_color:    tuple = (0.85, 0.88, 0.92),
    sun_direction:  tuple = (0.3, -0.8, 0.5),
    is_editing:     bool  = False,
) -> bool:
    """
    Per-frame real-time sky environment map update.

    Mirrors the top-level sky capture dispatch in UpdateSkyEnvMap() /
    FScene::UpdateSkyLightRealTimeCapture():
      1. Validate state (invalidate if sky changed)
      2. If time-sliced: render ASTRO_RT_CAPTURE_CLOUD_FACES per frame
         Else: render all 6 faces at once (editor fast path)
      3. When all 6 faces complete: run convolve + diffuse SH
      4. Set is_valid = True, reset faces_done bitmask

    Returns True when a complete convolution cycle just finished.

    鲁迅式：更新天光是渲染器最无聊的工作——
    每帧做一点点，没有人注意，没有人感谢，
    但若停下来，天空就会失去真实感，没有人会知道为什么。
    这就是后台工作者的处境。
    """
    capture  = get_rt_sky_capture()
    cycle_complete = False

    # ── Timeslice: decide how many faces to render this frame ─────────────
    if is_editing:
        faces_this_frame = ASTRO_CAPTURE_TIMESLICE_EDITOR
    elif ASTRO_RT_CAPTURE_TIMESLICE:
        faces_this_frame = ASTRO_RT_CAPTURE_CLOUD_FACES
    else:
        faces_this_frame = _CAPTURE_NUM_FACES   # all at once (non-timesliced)

    rendered_faces = []
    for _ in range(faces_this_frame):
        fi = capture.current_face

        # Render this face
        sky_col = render_sky_pass_for_capture(
            face_index=fi,
            atmosphere_color=atmosphere,
            cloud_color=cloud_color,
            sun_direction=sun_direction,
            include_clouds=True,
            depth_buffer=ASTRO_RT_DEPTH_BUFFER,
        )
        capture.sky_face_radiance[fi]   = sky_col

        # Cloud at reduced resolution (CVarVolumetricCloudResolutionDivider)
        cloud_col_low = (
            cloud_color[0] / ASTRO_RT_CLOUD_RES_DIVIDER,
            cloud_color[1] / ASTRO_RT_CLOUD_RES_DIVIDER,
            cloud_color[2] / ASTRO_RT_CLOUD_RES_DIVIDER,
        )
        capture.cloud_face_radiance[fi] = cloud_col_low

        capture.faces_done |= (1 << fi)
        rendered_faces.append(fi)

        # Advance to next face (wrap around at 6)
        capture.current_face = (fi + 1) % _CAPTURE_NUM_FACES

        # Slow timeslice: skip every other frame when enabled + only 1 face/frame
        if ASTRO_CAPTURE_TIMESLICE_SLOW and faces_this_frame == 1:
            if capture.frame_count % 2 != 0:
                break

    capture.frame_count += 1

    # ── Check if all 6 faces are done → run convolution ───────────────────
    if capture.faces_done == 0x3F:   # all 6 bits set
        # Convolve: pre-filter sky env map for all roughness levels
        combined_faces = [
            (capture.sky_face_radiance[fi][0] + capture.cloud_face_radiance[fi][0]*0.5,
             capture.sky_face_radiance[fi][1] + capture.cloud_face_radiance[fi][1]*0.5,
             capture.sky_face_radiance[fi][2] + capture.cloud_face_radiance[fi][2]*0.5)
            for fi in range(_CAPTURE_NUM_FACES)
        ]
        for mip in range(_CAPTURE_NUM_MIPS):
            avg_r = sum(c[0] for c in combined_faces) / _CAPTURE_NUM_FACES
            avg_g = sum(c[1] for c in combined_faces) / _CAPTURE_NUM_FACES
            avg_b = sum(c[2] for c in combined_faces) / _CAPTURE_NUM_FACES
            capture.convolve_specular[mip] = convolve_specular_face(
                (avg_r, avg_g, avg_b), mip)

        # Diffuse SH from mip-0 faces
        capture.diffuse_sh = compute_diffuse_irradiance_sh(combined_faces)

        capture.is_valid   = True
        capture.invalidated = False
        capture.faces_done  = 0      # reset for next cycle
        cycle_complete     = True

        print(
            f"[ASTRO-RT-CAPTURE] UpdateSkyEnvMap — cycle complete: "
            f"frame={capture.frame_count} "
            f"specular_mips={_CAPTURE_NUM_MIPS} "
            f"sh_L0=({capture.diffuse_sh[0]:.3f},"
            f"{capture.diffuse_sh[1]:.3f},{capture.diffuse_sh[2]:.3f})",
            file=sys.stderr,
        )
    else:
        print(
            f"[ASTRO-RT-CAPTURE] UpdateSkyEnvMap — "
            f"rendered_faces={rendered_faces} "
            f"faces_done=0b{capture.faces_done:06b} "
            f"frame={capture.frame_count}",
            file=sys.stderr,
        )

    return cycle_complete

















def query_sky_specular_radiance(roughness: float = 0.5) -> tuple:
    """
    Query the real-time sky pre-filtered specular environment.

    Mirrors the SkyLight specular probe texture lookup performed in
    ReflectionEnvironmentPixelShader.usf after UpdateSkyEnvMap() completes.

    Maps *roughness* → mip level, returns (R, G, B) × fade_alpha.
    Falls back to a neutral grey when the capture has not yet completed.

    鲁迅式：天空的光芒不会因为你还没准备好就消失——
    但在探针完成之前，它确实只是一个猜测。
    """
    capture = get_rt_sky_capture()
    if not capture.is_valid or not capture.convolve_specular:
        return (0.55, 0.68, 0.82)   # sky-blue fallback

    mip = int(round(roughness * (_CAPTURE_NUM_MIPS - 1)))
    mip = max(0, min(mip, _CAPTURE_NUM_MIPS - 1))
    return capture.convolve_specular.get(mip, (0.55, 0.68, 0.82))


# =============================================================================
# [ASTRO-CELL] HairStrandsRendering → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/HairStrands/HairStrandsRendering.cpp
#
# 鲁迅曾言：「青年应当有朝气，敢说，敢笑，敢哭，敢怒，敢骂，敢打，
# 在这可诅咒的地方击退了可诅咒的时代！」
# 发丝亦然——每一根细线都是独立的生命，密密麻麻，汇成不可忽视的力量。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FHairStrandsViewUniformParameters  → AstroCellHairStrandsViewParams
#     HairCoverageTexture              → coverage_map   (cell_id → [0,1])
#     HairOnlyDepthTexture             → hair_depth_map (cell_id → depth)
#     MaxSamplePerPixelCount           → max_sample_count (int)
#     HairTileCountXY                  → tile_count_xy   (tuple)
#     HairDualScatteringRoughnessOverride → dual_scatter_roughness (float)
#
#   InternalCreateHairStrandsViewUniformBuffer → build_hair_strands_view_params()
#   AllocateHairTransientResources      → allocate_hair_transient()
#   RenderHairPrePass                   → render_hair_pre_pass()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU RDG texture → per-cell coverage float in [0,1] derived from
#      the cell's own z-layer density (fraction of z-layer occupied by this cell)
#   2. HZB (Hierarchical Z-Buffer) → simple sorted depth list per z-layer
#   3. Tile data → grid tiles computed from cell bbox / tile_size
#   4. Stereo rendering second-eye copy → single-view epoch model (no stereo)
#   5. MacroGroup → species group (same species = same hair macro group)
# =============================================================================

# ── HairStrands CVars ─────────────────────────────────────────────────────────
_HS_TILE_SIZE:                int   = 8     # r.HairStrands.TileSize
_HS_MAX_SAMPLE_PER_PIXEL:     int   = 8     # r.HairStrands.MaxSample
_HS_DUAL_SCATTER_ROUGHNESS:   float = 0.3   # r.HairStrands.DualScatteringRoughnessOverride
_HS_DEEP_SHADOW_ENABLED:      bool  = True  # r.HairStrands.DeepShadow
_HS_VOXELIZATION_ENABLED:     bool  = True  # r.HairStrands.Voxelization
_HS_HZB_UPDATE_ENABLED:       bool  = True  # r.HairStrands.HZBUpdate

















class AstroCellHairStrandsViewParams:
    """
    Python equivalent of FHairStrandsViewUniformParameters.

    Holds per-view hair strand visibility data: coverage map, depth map,
    tile layout, and sampling parameters.  Built once per epoch by
    build_hair_strands_view_params() from the live cell registry.

    鲁迅式：发丝的参数是细节之海——每一根参数都在描述
    那些太细以至于无法被多边形捕捉的微小存在。
    """

    def __init__(self) -> None:
        # HairCoverageTexture: cell_id → coverage fraction [0,1]
        # High coverage = cell bbox densely packed with visual elements
        self.coverage_map: dict[str, float] = {}
        # HairOnlyDepthTexture: cell_id → normalised depth [0,1]
        self.hair_depth_map: dict[str, float] = {}
        # HairOnlyDepthHZBParameters (mip chain min/max): [z_min, z_max, inv_range, 0]
        self.hzb_parameters: tuple = (0.0, 1.0, 1.0, 0.0)
        # HZB mip chain: list of {z_min, z_max} per mip level
        self.hzb_mips: list[dict] = []
        # HairTileCountXY
        self.tile_count_xy: tuple = (0, 0)
        # HairSampleViewportResolution
        self.sample_viewport_resolution: tuple = (1200, 900)
        # MaxSamplePerPixelCount
        self.max_sample_count: int = _HS_MAX_SAMPLE_PER_PIXEL
        # HairDualScatteringRoughnessOverride
        self.dual_scatter_roughness: float = _HS_DUAL_SCATTER_ROUGHNESS
        # bHairTileValid
        self.tile_valid: bool = False
        # Species macro groups: species → list of cell_ids
        self.macro_groups: dict[str, list[str]] = {}

















def _hs_compute_coverage(cell_id: str, bbox: dict,
                          z_layer_cells: list[str]) -> float:
    """
    Compute hair-analogue coverage for a cell.

    Mirrors HairCoverageTexture: fraction of the cell's «screen tile» that
    is occupied by hair-like visual elements.  In the Astro 2-D context,
    coverage is derived from:
      - Cell area relative to its z-layer peers (denser z-layer → higher coverage)
      - Species type: hair-strand-like species (cil-eye, cil-graph) → high coverage;
        solid block species (cil-bolt, cil-layers) → lower coverage

    鲁迅式：覆盖率是细节密度的证明——发丝之所以难渲染，
    是因为它无处不在，无时不遮挡。
    """
    _COVERAGE_BY_SPECIES = {
        "cil-eye":         0.75,   # radial rays = hair-like density
        "cil-graph":       0.70,   # node-edge filaments
        "cil-vector":      0.60,   # arrow lines
        "cil-loop":        0.55,   # arc strand
        "cil-code":        0.50,   # brace lines
        "cil-bolt":        0.40,   # solid zigzag
        "cil-plus":        0.35,   # cross arms
        "cil-filter":      0.55,   # grid lines
        "cil-layers":      0.30,   # solid rects
        "cil-arrow-right": 0.25,   # single polygon
    }
    # Base coverage from species type
    species = bbox.get("species", "")
    base = _COVERAGE_BY_SPECIES.get(species, 0.45)

    # Modulate by z-layer density: more peers in same layer → each cell more covered
    n_peers = max(len(z_layer_cells), 1)
    density_factor = min(1.0, 1.0 + (n_peers - 1) * 0.05)
    return min(1.0, base * density_factor)

















def _hs_build_hzb(depth_values: list[float]) -> tuple[tuple, list[dict]]:
    """
    Build a minimal HZB (Hierarchical Z-Buffer) from a sorted depth list.

    Mirrors HairOnlyDepthHZBParameters (FVector4f) and the per-mip closest/
    furthest HZB textures maintained for hair depth testing.

    2-D adaptation: instead of a full 2-D texture hierarchy, we maintain a
    list of (z_min, z_max) pairs at each power-of-2 mip level, computed by
    splitting the sorted depth list and taking min/max per segment.

    Returns (hzb_params_tuple, mip_list).

    鲁迅式：层级深度缓冲区是对效率的崇拜——
    用树状结构把深度测试的代价从 O(N) 压到 O(log N)。
    哪怕是发丝，也逃不过这棵树的筛选。
    """
    if not depth_values:
        return (0.0, 1.0, 1.0, 0.0), [{"z_min": 0.0, "z_max": 1.0}]

    sorted_depths = sorted(depth_values)
    z_min_global  = sorted_depths[0]
    z_max_global  = sorted_depths[-1]
    z_range       = z_max_global - z_min_global
    inv_range     = 1.0 / max(z_range, 1e-6)
    hzb_params    = (z_min_global, z_max_global, inv_range, 0.0)

    # Build mip chain (each mip halves the sample count)
    mips = []
    current = sorted_depths
    while current:
        mips.append({
            "z_min": current[0],
            "z_max": current[-1],
            "count": len(current),
        })
        # Next mip: take every other element (closest/furthest per 2-cell window)
        current = [current[i] for i in range(0, len(current), 2)]
        if len(current) <= 1:
            break

    return hzb_params, mips

















def render_hair_pre_pass(
    cell_registry: dict,
    viewport_w:    int = 1200,
    viewport_h:    int = 900,
) -> AstroCellHairStrandsViewParams:
    """
    Hair strands pre-pass — macro group creation + voxelization + deep shadow.

    Mirrors RenderHairPrePass():
      1. AddRenderCurveRasterPipeline  → early-out if no hair-like cells
      2. CreateHairStrandsMacroGroups  → group cells by species
      3. VoxelizeHairStrands           → compute per-cell volume density
      4. RenderHairStrandsDeepShadows  → compute per-cell self-shadow attenuation
      5. AddMeshDrawTransitionPass     → mark macro group dirty flag cleared

    Returns populated AstroCellHairStrandsViewParams.

    鲁迅式：预处理是牺牲——在主渲染之前先受苦，
    换来主渲染时的从容不迫。没有预处理，一切都是临时起意。
    """
    params = build_hair_strands_view_params(cell_registry, viewport_w, viewport_h)

    if not params.tile_valid:
        print("[ASTRO-HS] render_hair_pre_pass: no hair cells — early out",
              file=sys.stderr)
        return params

    # ── Voxelization: compute per-cell volume density ─────────────────────
    # Mirrors VoxelizeHairStrands() — allocates a 3-D voxel grid and
    # rasterises each hair strand into it.  2-D analogue: per-cell density
    # in its bbox tile grid (coverage × cell area / tile area).
    tile_area = _HS_TILE_SIZE * _HS_TILE_SIZE
    cells     = cell_registry.get("cells", {})
    for cell_id, entry in cells.items():
        coverage = params.coverage_map.get(cell_id, 0.5)
        bbox_data = entry.get("bbox", {})
        if "min" in bbox_data and "max" in bbox_data:
            mn, mx  = bbox_data["min"], bbox_data["max"]
            cell_w  = mx[0] - mn[0]
            cell_h  = mx[1] - mn[1]
        else:
            cell_w  = bbox_data.get("w", 80)
            cell_h  = bbox_data.get("h", 50)
        cell_area = max(cell_w * cell_h, 1.0)
        # VoxelDensity ≈ coverage × area / tile_area (proxy for strand density)
        voxel_density = min(1.0, coverage * cell_area / (tile_area * 4.0))
        # Store in coverage map as voxelised value (overwrite with refined estimate)
        params.coverage_map[cell_id] = max(params.coverage_map.get(cell_id, coverage),
                                           voxel_density)

    # ── Deep shadow: per-cell self-shadow from voxel density ──────────────
    # Mirrors RenderHairStrandsDeepShadows() — per-cell transmittance computed
    # from accumulated voxel density along the light ray.  We use an analytic
    # Beer-Lambert approximation: T = exp(-density × shadow_extinction).
    _SHADOW_EXTINCTION = 4.0   # extinction coefficient (per-cell tuning)
    deep_shadow: dict[str, float] = {}
    for cell_id in cells:
        density    = params.coverage_map.get(cell_id, 0.5)
        transmittance = math.exp(-density * _SHADOW_EXTINCTION)
        deep_shadow[cell_id] = transmittance   # 0 = fully shadowed, 1 = lit

    # Publish deep shadow as a sub-field (consumed by hair lighting pass)
    params.deep_shadow = deep_shadow   # type: ignore[attr-defined]

    print(
        f"[ASTRO-HS] render_hair_pre_pass: "
        f"voxelised={len(params.coverage_map)} "
        f"deep_shadows={len(deep_shadow)} "
        f"avg_coverage={sum(params.coverage_map.values())/max(len(params.coverage_map),1):.3f}",
        file=sys.stderr,
    )
    return params


# =============================================================================
# [ASTRO-CELL] SubstrateRendering → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Substrate/Substrate.cpp
#
# 鲁迅曾言：「我翻开历史一查，这历史没有年代，歪歪斜斜的每页上
# 都写着『仁义道德』四个字。我横竖睡不着，仔细看了半夜，
# 才从字缝里看出字来，满本都写着两个字是『吃人』！」
# Substrate 材质系统亦然——层层包装之下，是光与物质的碰撞。
# 每一层闭包，都在争夺那有限的能量预算。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FSubstrateGlobalUniformParameters  → AstroCellSubstrateUniforms
#     MaxClosurePerPixel               → max_closure_per_pixel (int, ≤8)
#     UsesTileTypeMask                 → uses_tile_type_mask    (int, bitmask)
#     UsesAnisotropy                   → uses_anisotropy        (bool)
#     ClosuresPerPixel (from CVar)     → closures_per_pixel     (int)
#   FSubstrateViewData::Reset()        → substrate_view_data_reset()
#   GetSubstrateMaxClosureCount()      → get_substrate_max_closure_count()
#   GetClosureTileIndirectArgsOffset() → get_closure_tile_offset()
#   IsStochasticLightingActive()       → is_stochastic_lighting_active()
#   UsesSubstrateMaterialBuffer()      → uses_material_buffer()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU tile classification pass → per-species closure count heuristic
#   2. ClosureCountFromMaterial (CVar) → species complexity table
#   3. Stochastic lighting classification → probabilistic sampling weight
#   4. CMask/HTile clear → Python dict reset
#   5. AllocationMode grows-only behaviour → max-of-history dict tracking
# =============================================================================

# ── Substrate CVars ───────────────────────────────────────────────────────────
_SUB_MAX_CLOSURE_COUNT:           int  = 8    # SUBSTRATE_MAX_CLOSURE_COUNT
_SUB_CLOSURES_PER_PIXEL:          int  = 3    # r.Substrate.ClosuresPerPixel
_SUB_USE_CLOSURE_COUNT_FROM_MAT:  bool = True # r.Substrate.UseClosureCountFromMaterial
_SUB_ASYNC_CLASSIFICATION:        bool = True # r.Substrate.AsyncClassification
_SUB_STOCHASTIC_LIGHTING_ACTIVE:  bool = False# r.Substrate.StochasticLighting.Active
_SUB_ROUGHNESS_TRACKING:          bool = True # r.Substrate.Debug.RoughnessTracking
_SUB_TILE_COORD_8BIT:             bool = True # r.Substrate.TileCoord8bits
_SUB_ALLOCATION_MODE:             int  = 1    # r.Substrate.AllocationMode
_SUB_MAX_DOWNSAMPLE_FACTOR:       int  = 2    # GetMaxDownsampleFactor()

# Tile type bitmask constants (mirrors ESubstrateTileType enum bits)
_SUB_TILE_SIMPLE         = 1 << 0   # simple single-closure tile
_SUB_TILE_SINGLE_CLOSURE = 1 << 1   # single-closure complex tile
_SUB_TILE_COMPLEX        = 1 << 2   # multi-closure tile
_SUB_TILE_HAIR           = 1 << 3   # hair-strands tile
_SUB_TILE_ALL            = 0xFF

















def get_closure_tile_offset(downsample_factor: int) -> int:
    """
    Compute byte offset into the indirect args buffer for a given downsample factor.

    Mirrors GetClosureTileIndirectArgsOffset(InDownsampleFactor):
        Offset = (clamp(dsf, 1, max_dsf) - 1) * sizeof(FRHIDispatchIndirectParameters)
    sizeof(FRHIDispatchIndirectParameters) = 12 bytes (3× uint32).

    鲁迅式：偏移量是寻址的精确度——差一个字节，便是另一个世界。
    """
    _INDIRECT_PARAMS_SIZE = 12   # 3 × uint32
    clamped = max(1, min(downsample_factor, _SUB_MAX_DOWNSAMPLE_FACTOR))
    return (clamped - 1) * _INDIRECT_PARAMS_SIZE

















def is_stochastic_lighting_active() -> bool:
    """
    Mirrors IsStochasticLightingActive():
        return IsStochasticLightingEnabled(Platform) && CVarActive > 0
    In Astro: platform capability check → True; CVar toggle from constant.

    鲁迅式：随机光照是用概率换取精度——以混沌始，以收敛终。
    """
    return _SUB_STOCHASTIC_LIGHTING_ACTIVE

















def compute_coc_radius(
    cell_z:     float,
    focal_z:    float = _DOF_FOCAL_Z_LAYER,
    near_trans: float = _DOF_NEAR_TRANSITION,
    far_trans:  float = _DOF_FAR_TRANSITION,
    max_coc:    float = _DOF_MAX_COC_PIXELS,
) -> float:
    """
    Compute Circle-of-Confusion radius for a cell at z-layer *cell_z*.

    Mirrors DiaphragmDOF's CoC computation from the lens model:
        CoC = f² × (z − z_focal) / (N × z × z_focal × (z − f))
    Simplified to a linear ramp for the 2-D z-layer model:
        CoC_near = clamp((focal_z − cell_z) / near_trans, 0, 1) × max_coc
        CoC_far  = clamp((cell_z − focal_z) / far_trans,  0, 1) × max_coc
        CoC      = max(CoC_near, CoC_far)

    Returns CoC radius in pixels (float, ≥ 0).

    鲁迅式：弥散圆是距离的惩罚——离焦点越远，圆越大，清晰越少。
    焦平面是优待，远处是放逐。
    """
    delta = cell_z - focal_z
    if delta < 0:
        # Near-field: cell is in front of focal plane
        coc = abs(delta) / max(near_trans, 1e-6)
    else:
        # Far-field: cell is behind focal plane
        coc = delta / max(far_trans, 1e-6)
    return min(max_coc, coc * max_coc)

















def gather_bokeh_rings(
    coc_radius: float,
    ring_count: int = _DOF_RING_COUNT,
    accumulator_quality: int = _DOF_ACCUMULATOR_QUALITY,
) -> float:
    """
    Compute the effective blur weight from concentric ring kernel accumulation.

    Mirrors the gather pass ring accumulator:
        For each ring r in [1, ring_count]:
            n_samples_r = 8 × r   (octagonal ring)
            sample_weight ∝ 1 / (n_samples_r)
        total_weight = sum of all ring sample weights

    In the 2-D analytic version we compute the equivalent area-weighted blur
    strength from the ring radii, bypassing the sample loop.

    Returns a blur weight ∈ [0, 1] for use in SVG filter blur radius scaling.

    鲁迅式：环形采样是民主的——每一环等权，远近均有发言权，
    只是远处的声音（大 CoC）被更多环覆盖而显得更重要。
    """
    if coc_radius < _DOF_MIN_FULLRES_BLUR_RADIUS:
        return 0.0

    # Total sample count across all rings (mirrors actual ring sample counts)
    total_samples = sum(8 * r for r in range(1, ring_count + 1))
    if total_samples == 0:
        return 0.0

    # Effective blur weight: fraction of rings whose radius ≤ CoC
    effective_rings = min(ring_count,
                          max(1, int(coc_radius / max(_DOF_MAX_COC_PIXELS / ring_count, 1))))
    effective_samples = sum(8 * r for r in range(1, effective_rings + 1))

    # Quality multiplier (AccumulatorQuality=1 → 1.0; =2 → 1.2 extra samples)
    quality_mult = 1.0 + (accumulator_quality - 1) * 0.2

    return min(1.0, (effective_samples / total_samples) * quality_mult)

















def scatter_bokeh_sprites(
    coc_radius:       float,
    brightness:       float,
    scatter_min_coc:  float = _DOF_SCATTER_MIN_COC,
    scatter_max_ratio:float = _DOF_SCATTER_MAX_SPRITE,
) -> float:
    """
    Determine scatter contribution for a cell with given CoC.

    Mirrors the scatter pass sprite logic:
        if CoC < scatter_min_coc: no scatter
        if scattered_pixels / total_pixels > max_ratio: clamp
        scatter_intensity = clamp(CoC / max_coc, 0, 1) × brightness

    Returns scatter opacity ∈ [0, 1] to add to the cell's SVG filter.

    鲁迅式：散射是高光的奢侈——只有足够亮、足够散的像素，
    才配获得单独的精灵渲染。其余的，老老实实去聚集通道排队。
    """
    if coc_radius < scatter_min_coc:
        return 0.0
    scatter_strength = min(1.0, coc_radius / _DOF_MAX_COC_PIXELS)
    # Brightness gate: only bright cells scatter (high brightness = high CoC visibility)
    if brightness < 0.3:
        return 0.0
    return min(1.0, scatter_strength * brightness * scatter_max_ratio * 10.0)

















def dof_recombine(
    cell_gather_weight: float,
    cell_scatter_weight: float,
    coc_radius: float,
    recombine_quality: int = _DOF_RECOMBINE_QUALITY,
) -> float:
    """
    Full-resolution recombine pass — merge gather + scatter into final blur.

    Mirrors the Recombine pass that composites half-res gathered DOF back
    with the full-res sharp layer:
        if recombine_quality == 0: no slight-out-of-focus
        if recombine_quality >= 1: blend gather + scatter × sof_weight

    Returns final DOF blur radius for the cell's feGaussianBlur stdDeviation.

    鲁迅式：重组是和解——把模糊的过去和清晰的现在合并成一帧，
    既不全然遗忘，也不执意清醒。
    """
    if coc_radius < _DOF_MIN_FULLRES_BLUR_RADIUS:
        return 0.0

    # Slight-out-of-focus weight (only for recombine_quality >= 1)
    sof_weight = 0.0 if recombine_quality == 0 else min(1.0, coc_radius / 4.0)

    # Combine gather and scatter contributions
    combined = (cell_gather_weight * 0.7 + cell_scatter_weight * 0.3) * sof_weight

    # Scale to pixel blur radius (0 = sharp, _DOF_MAX_COC_PIXELS = max blur)
    blur_radius = combined * coc_radius
    return round(blur_radius, 2)

















class AstroCellDOFGatherPass:
    """
    Depth-of-Field gather pass orchestrator — mirrors FDiaphragmDOFPass.

    Processes all cells in the registry, computes per-cell CoC radii,
    classifies each as foreground/background/in-focus, runs gather + scatter,
    and returns a dict of per-cell DOF blur parameters for SVG filter injection.

    鲁迅式：景深处理是摄影师的选择——
    决定让谁清晰，让谁模糊，便是决定谁是主角，谁是背景。
    """

    def __init__(self,
                 focal_z:    float = _DOF_FOCAL_Z_LAYER,
                 near_trans: float = _DOF_NEAR_TRANSITION,
                 far_trans:  float = _DOF_FAR_TRANSITION) -> None:
        self.focal_z    = focal_z
        self.near_trans = near_trans
        self.far_trans  = far_trans

    def process(self, cell_registry: dict) -> dict[str, dict]:
        """
        Run the full DOF pipeline for all registered cells.

        Returns dict: cell_id → {coc_radius, blur_radius, layer, gather_w, scatter_w}
        where layer ∈ {'foreground', 'focus', 'background'}.
        """
        cells  = cell_registry.get("cells", {})
        result: dict[str, dict] = {}

        for cell_id, entry in cells.items():
            z = float(entry.get("z", 3))
            # Brightness proxy from coverage (higher coverage = brighter cell)
            coverage = entry.get("constraint_mask", 0)
            brightness = 0.5 + coverage * 0.3   # simplified

            coc = compute_coc_radius(z, self.focal_z, self.near_trans, self.far_trans)
            gather_w  = gather_bokeh_rings(coc)
            scatter_w = scatter_bokeh_sprites(coc, brightness)
            blur_r    = dof_recombine(gather_w, scatter_w, coc)

            # Layer classification
            delta = z - self.focal_z
            if delta < -0.5:
                layer = "foreground"
            elif delta > 0.5:
                layer = "background"
            else:
                layer = "focus"

            result[cell_id] = {
                "coc_radius":    round(coc, 3),
                "blur_radius":   blur_r,
                "gather_weight": round(gather_w, 4),
                "scatter_weight":round(scatter_w, 4),
                "layer":         layer,
            }

        fg = sum(1 for v in result.values() if v["layer"] == "foreground")
        bg = sum(1 for v in result.values() if v["layer"] == "background")
        print(
            f"[ASTRO-DOF] AstroCellDOFGatherPass.process: "
            f"total={len(result)} fg={fg} focus={len(result)-fg-bg} bg={bg} "
            f"focal_z={self.focal_z:.1f}",
            file=sys.stderr,
        )
        return result


# =============================================================================
# [ASTRO-CELL] PostProcessMotionBlur → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/PostProcessMotionBlur.cpp
#
# 鲁迅曾言：「时间就是性命。无端的空耗别人的时间，
# 其实是无异于谋财害命的。」
# 运动模糊亦然——时间在帧与帧之间流逝，运动的轨迹是时间的刻痕。
# 不做运动模糊，是对时间流逝的否认；做过度，是对帧率的谋财害命。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   IsMotionBlurEnabled()              → is_motion_blur_enabled()
#   IsMotionBlurScatterRequired()      → is_scatter_required()
#   GetMotionBlurTileCount()           → get_motion_blur_tile_count()
#   FVelocityFlattenTextures           → AstroCellVelocityFlattenData
#   kMotionBlurFlattenTileSize = 16    → _MB_FLATTEN_TILE_SIZE
#   GetPreviousWorldToClipMatrix()     → compute_prev_frame_velocity()
#   EMotionBlurQuality                 → _MB_QUALITY (0=low, 1=med, 2=high, 3=cinematic)
#   EMotionBlurFilter                  → _MB_FILTER (0=low, 1=med, 2=high)
#
# Algorithm changes (鲁迅式 20%):
#   1. Velocity flatten → per-cell z-layer velocity (dz per epoch)
#   2. GPU tile max velocity → max absolute dz across neighbours
#   3. Scatter vs gather decision → analytic threshold (not GPU readback)
#   4. Motion blur filter → 1-D Gaussian blur on SVG feGaussianBlur
#   5. Half-res gather → simply halved blur kernel for performance
# =============================================================================

# ── MotionBlur CVars ──────────────────────────────────────────────────────────
_MB_QUALITY:             int   = 3      # r.MotionBlurQuality [0,4]; 0=off
_MB_AMOUNT:              float = 1.0    # PostProcessSettings.MotionBlurAmount
_MB_MAX_VEL_FRACTION:    float = 0.05   # PostProcessSettings.MotionBlurMax (% of viewport)
_MB_FLATTEN_TILE_SIZE:   int   = 16     # kMotionBlurFlattenTileSize
_MB_FILTER_TILE_SIZE:    int   = 16     # kMotionBlurFilterTileSize
_MB_SCATTER_THRESHOLD:   float = 3.0    # TileDistanceMaxGathered
_MB_HALF_RES_INPUT:      bool  = True   # r.MotionBlur.HalfResInput
_MB_SEPARABLE:           bool  = False  # r.MotionBlurSeparable
_MB_DIRECTIONS:          int   = 1      # r.MotionBlur.Directions
_MB_SECOND_SCALE:        float = 1.0    # r.MotionBlur2ndScale

















class AstroCellTAAHistory:
    """
    Python equivalent of FTemporalAAHistory.

    Stores the per-cell accumulated TAA history between epochs.
    Uses exponential moving average for the colour estimate and
    tracks the last z-layer for anti-ghosting.

    鲁迅式：历史是 TAA 的财富，也是它的负担——
    太多的历史导致鬼影，太少的历史导致抖动。
    0.04 的当前帧权重，是这场博弈的和解系数。
    """

    def __init__(self) -> None:
        # Per-cell accumulated colour: cell_id → (R, G, B) float tuple
        self.colour_history: dict[str, tuple] = {}
        # Per-cell last z-layer: cell_id → float (for anti-ghosting)
        self.z_history:      dict[str, float] = {}
        # Per-cell pre-exposure correction (exposure normalisation)
        self.pre_exposure:   dict[str, float] = {}
        # Epoch index of last update
        self.epoch:          int = 0

    def is_valid_for(self, cell_id: str, current_z: float) -> bool:
        """
        Check whether history is valid for this cell (anti-ghosting gate).

        Mirrors the mobility-based anti-ghosting in TAA Quality 2+:
        if |z_current - z_history| > threshold → reject history (return False).

        鲁迅式：历史不总是可信的——当物体移动太多时，
        过去的颜色已与现在无关，必须果断抛弃。
        """
        if cell_id not in self.z_history:
            return False
        if _TAA_QUALITY >= 2:
            z_delta = abs(current_z - self.z_history[cell_id])
            if z_delta > _TAA_GHOST_THRESHOLD:
                return False
        return True

    def update(self, cell_id: str, current_colour: tuple,
               current_z: float) -> tuple:
        """
        Update history with the current frame's colour, returning the blended result.

        Mirrors the TAA accumulation blend:
            history_weight = 1 − CurrentFrameWeight
            output = current × w_curr + history × w_hist
        Neighbourhood clamping (AABB clamp) applied to history before blend.

        鲁迅式：积累是 TAA 的本质——每帧只相信 4% 的新证据，
        96% 来自历史的惯性。这是保守主义的数学表达。
        """
        w_curr = _TAA_CURRENT_FRAME_WEIGHT
        w_hist = 1.0 - w_curr

        if not self.is_valid_for(cell_id, current_z):
            # No valid history → use current frame only (cold start)
            blended = current_colour
        else:
            hist = self.colour_history[cell_id]

            # Neighbourhood clamping (AABB in colour space):
            # Clamp history to [current × 0.5, current × 1.5] per channel
            # Mirrors the AABB colour clip in the TAA shader
            clamped_hist = tuple(
                max(current_colour[i] * 0.5, min(current_colour[i] * 1.5, hist[i]))
                for i in range(3)
            )

            # Catmull-Rom sharpening (optional): increases current frame weight
            if _TAA_CATMULL_ROM:
                w_curr = min(1.0, w_curr * 1.5)
                w_hist = 1.0 - w_curr

            blended = tuple(
                current_colour[i] * w_curr + clamped_hist[i] * w_hist
                for i in range(3)
            )

        self.colour_history[cell_id] = blended
        self.z_history[cell_id]      = current_z
        return blended


# Module-level TAA history singleton (one per logical view)
_ASTRO_TAA_HISTORY: AstroCellTAAHistory = AstroCellTAAHistory()

















def get_taa_history() -> AstroCellTAAHistory:
    """Return the global TAA history singleton."""
    return _ASTRO_TAA_HISTORY

















class AstroCellTAAPass:
    """
    Temporal Anti-Aliasing pass — mirrors FTemporalAA shader + AddTemporalAAPass().

    Processes all cells, applies sub-pixel jitter via Halton sequence,
    reads history, clamps, blends, and writes updated history + output colours.

    The output per-cell colours are used downstream to modulate SVG fill
    opacity and stroke colour (the TAA-smoothed colour replaces the raw
    species colour for a temporally stable result).

    鲁迅式：时域抗锯齿是耐心的产物——
    一帧解决不了锯齿问题，但一百帧一定可以。
    只要不动（或者少动），历史就是你的盟友。
    """

    def __init__(self,
                 quality:       int   = _TAA_QUALITY,
                 filter_size:   float = _TAA_FILTER_SIZE) -> None:
        self.quality     = quality
        self.filter_size = filter_size
        self._history    = get_taa_history()

    def _jitter_offset(self, epoch: int) -> tuple[float, float]:
        """
        Sub-pixel jitter using Halton(2,3) sequence.
        Mirrors TemporalJitterPixels computation in the TAA shader setup.
        The jitter cycles over 8 frames (same as UE5 default TemporalAA.SampleCount=8).

        鲁迅式：抖动是策略性的不安定——
        每帧故意把采样点移动一点，让时间帮你填满所有的空隙。
        """
        n = epoch % 8
        jx = _pt_halton(n, 2) - 0.5   # [-0.5, +0.5]
        jy = _pt_halton(n, 3) - 0.5
        return (jx * self.filter_size, jy * self.filter_size)

    def _compute_output_colour(
        self,
        cell_id:  str,
        species:  str,
        cell_z:   float,
        jitter:   tuple[float, float],
    ) -> tuple:
        """
        Compute the jittered current-frame colour for a cell.

        In the 2-D analogue, jitter modulates the species primary colour
        slightly (like sub-pixel displacement affecting which texel is sampled):
            colour_jittered = colour × (1 + jx×0.02) in R, (1 + jy×0.02) in G
        This is the 2-D equivalent of jittering the texture UV.
        """
        sp_idx  = _species_to_index(species)
        base    = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])
        r = max(0.0, min(1.0, base[0] / 255.0 + jitter[0] * 0.02))
        g = max(0.0, min(1.0, base[1] / 255.0 + jitter[1] * 0.02))
        b = max(0.0, min(1.0, base[2] / 255.0))
        return (r, g, b)

    def run(
        self,
        cell_registry: dict,
        epoch:         int = 0,
    ) -> dict[str, tuple]:
        """
        Execute the TAA pass for all cells.

        Returns dict: cell_id → (R, G, B) accumulated output colour.

        鲁迅式：运行 TAA 是召唤历史来帮助现在——
        每一次运行都是对过去 96 次运行的致敬。
        """
        cells   = cell_registry.get("cells", {})
        jitter  = self._jitter_offset(epoch)
        outputs: dict[str, tuple] = {}

        for cell_id, entry in cells.items():
            z       = float(entry.get("z", 3))
            species = entry.get("species", "")

            current = self._compute_output_colour(cell_id, species, z, jitter)
            blended = self._history.update(cell_id, current, z)
            outputs[cell_id] = blended

        # Advance history epoch
        self._history.epoch = epoch

        converged = sum(
            1 for cid in outputs
            if self._history.is_valid_for(cid, float(cells.get(cid, {}).get("z", 3)))
        )

        print(
            f"[ASTRO-TAA] AstroCellTAAPass.run: "
            f"epoch={epoch} total={len(outputs)} "
            f"history_valid={converged} "
            f"jitter=({jitter[0]:.3f},{jitter[1]:.3f}) "
            f"quality={self.quality} filter={self.filter_size:.2f}",
            file=sys.stderr,
        )
        return outputs

















class EBlendMode(IntEnum):
    Opaque      = 0
    Masked      = 1
    Translucent = 2
    Additive    = 3
    Modulate    = 4
    AlphaComposite = 5
    AlphaHoldout   = 6













def _is_alpha_composite(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.AlphaComposite









def _is_modulate(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.Modulate

















def is_compatible_with_render_stage(desc: FDecalBlendDesc, stage: EDecalRenderStage) -> bool:
    return bool(desc.render_stage_mask & (1 << stage))

















def get_base_render_stage(desc: FDecalBlendDesc) -> EDecalRenderStage:
    """
    返回贴花的主渲染阶段，对应 GetBaseRenderStage()。
    鲁迅式：阶段的优先顺序，是工程师对渲染时序的一次沉默表态。
    """
    for stage in (
        EDecalRenderStage.BeforeBasePass,
        EDecalRenderStage.BeforeLighting,
        EDecalRenderStage.Mobile,
        EDecalRenderStage.MobileBeforeLighting,
    ):
        if desc.render_stage_mask & (1 << stage):
            return stage
    return EDecalRenderStage.None_


# =============================================================================
# § DecalRenderingShared — 可见贴花列表构建与视图分发
#   移植自 Renderer-Private/DecalRenderingShared.cpp
# =============================================================================

import math
from typing import Callable, Iterable


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class FBlueNoiseParameters:
    """
    蓝噪声纹理参数，对应 FBlueNoiseParameters。
    dimensions: (width, height, slices)
    modulo_masks: 用于快速取模的位掩码三元组，要求各维度为 2 的幂。
    """
    dimensions:   tuple[int, int, int] = (1, 1, 1)
    modulo_masks: tuple[int, int, int] = (0, 0, 0)
    # 实际纹理数据存为 numpy 数组（float32）
    scalar_texture: Optional[np.ndarray] = None
    vec2_texture:   Optional[np.ndarray] = None

















def _floor_log2(n: int) -> int:
    return int(math.floor(math.log2(max(n, 1))))




@runtime_checkable




@runtime_checkable





@runtime_checkable




@runtime_checkable
class IComputeTaskWorker(Protocol):
    """计算任务工人接口，对应 IComputeTaskWorker。"""
    def execute(self) -> None: ...


@runtime_checkable


@runtime_checkable


@runtime_checkable


@runtime_checkable



@runtime_checkable


@runtime_checkable


@runtime_checkable


@runtime_checkable
class IComputeSystem(Protocol):
    """
    计算系统接口，对应 IComputeSystem。
    鲁迅式：接口是契约，也是无声的命令。
    """
    def create_workers(
        self, scene: object, out_workers: list[IComputeTaskWorker]
    ) -> None: ...

    def destroy_workers(
        self, scene: object, in_out_workers: list[IComputeTaskWorker]
    ) -> None: ...

















class ComputeSystemRegistry:
    """
    全局计算系统注册表，对应 ComputeSystemInterface 命名空间。
    鲁迅式：全局注册表的存在，说明没有人敢承担依赖注入的责任。
    """

    def __init__(self):
        self._systems: list[IComputeSystem] = []

    def register_system(self, system: IComputeSystem) -> None:
        if system not in self._systems:
            self._systems.append(system)

    def unregister_system(self, system: IComputeSystem) -> None:
        for i, s in enumerate(self._systems):
            if s is system:
                # swap-remove，与 UE5 RemoveAtSwap 保持一致
                self._systems[i] = self._systems[-1]
                self._systems.pop()
                return

    def create_workers(
        self, scene: object, out_workers: list[IComputeTaskWorker]
    ) -> None:
        for system in self._systems:
            system.create_workers(scene, out_workers)

    def destroy_workers(
        self, scene: object, in_out_workers: list[IComputeTaskWorker]
    ) -> None:
        for system in self._systems:
            system.destroy_workers(scene, in_out_workers)
        # 销毁后列表必须为空，对应 ensure(InOutWorkders.Num() == 0)
        if in_out_workers:
            raise RuntimeError(
                f"ComputeSystemRegistry.destroy_workers: "
                f"{len(in_out_workers)} worker(s) not cleaned up"
            )


# 模块级单例，对应 GRegisteredSystems
_global_compute_registry = ComputeSystemRegistry()

















def register_compute_system(system: IComputeSystem) -> None:
    _global_compute_registry.register_system(system)









def unregister_compute_system(system: IComputeSystem) -> None:
    _global_compute_registry.unregister_system(system)









def create_compute_workers(scene: object, out_workers: list) -> None:
    _global_compute_registry.create_workers(scene, out_workers)









def destroy_compute_workers(scene: object, in_out_workers: list) -> None:
    _global_compute_registry.destroy_workers(scene, in_out_workers)


# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================





# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================




# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================


class EShadingPath(IntEnum):
    Forward  = 0
    Deferred = 1
    Mobile   = 2


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class ShaderComplexityBaseline:
    """
    着色器复杂度基线，对应各 GShaderComplexityBaseline* 全局变量。
    鲁迅式：基线是期望，超出基线是现实，差值是我们不愿承认的懒惰。
    """
    # Forward
    forward_vs:        int   = 134
    forward_ps:        int   = 635
    forward_unlit_ps:  int   = 47
    # Deferred
    deferred_vs:       int   = 41
    deferred_ps:       int   = 111
    deferred_unlit_ps: int   = 33
    # Mobile Forward
    mobile_forward_vs:        int   = 134
    mobile_forward_ps:        int   = 143
    mobile_forward_unlit_ps:  int   = 6
    # Mobile Deferred
    mobile_deferred_vs:        int   = 134
    mobile_deferred_ps:        int   = 50
    mobile_deferred_unlit_ps:  int   = 9
    # Masked cost multiplier (mobile)
    mobile_masked_cost_multiplier: float = 1.5


# 模块级默认基线，对应各 CVarShaderComplexityBaseline* 的默认值
_default_shader_complexity_baseline = ShaderComplexityBaseline()

















def get_quad_overdraw_uav_index(
    is_forward_shading: bool,
    base_pass_can_output_velocity: bool,
) -> int:
    """
    返回 Quad Overdraw UAV 的寄存器槽位，对应 GetQuadOverdrawUAVIndex()。
    鲁迅式：槽位编号背后是整套 GBuffer 布局——
    改动任何一处，牵一发而动全身，这就是耦合的代价。
    """
    if is_forward_shading:
        return 2 if base_pass_can_output_velocity else 1
    else:
        return 7 if base_pass_can_output_velocity else 6


_NUM_STREAMING_ACCURACY_COLORS = 5

_DEFAULT_ACCURACY_COLORS: list[tuple[float, float, float, float]] = [
    (0.0,  0.0,  1.0,  1.0),   # 蓝：过度流送
    (0.0,  1.0,  0.0,  1.0),   # 绿：刚好
    (1.0,  1.0,  0.0,  1.0),   # 黄：轻微不足
    (1.0,  0.5,  0.0,  1.0),   # 橙：中度不足
    (1.0,  0.0,  0.0,  1.0),   # 红：严重不足
]


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class FDebugViewModeUniformParameters:
    """
    调试视图模式 Uniform 参数，对应 FDebugViewModeUniformParameters。
    鲁迅式：颜色精度图谱是渲染引擎对自身的一次公开体检。
    """
    accuracy_colors: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: list(_DEFAULT_ACCURACY_COLORS)
    )
    # 纹理坐标密度诊断（UV 通道分析用）
    uv_density_sampling: float = 1.0

















def setup_debug_view_mode_pass_uniform_buffer_constants(
    accuracy_colors_override: list[tuple[float, float, float, float]] | None = None,
    uv_density_sampling: float = 1.0,
) -> FDebugViewModeUniformParameters:
    """
    构造调试视图模式 Uniform 缓冲区常量，对应
    SetupDebugViewModePassUniformBufferConstants()。
    鲁迅式：把颜色填进参数结构，是把主观判断包装成客观数据的例行仪式。
    """
    colors = list(_DEFAULT_ACCURACY_COLORS)
    if accuracy_colors_override:
        # 截取到 _NUM_STREAMING_ACCURACY_COLORS，其余补黑
        n = min(len(accuracy_colors_override), _NUM_STREAMING_ACCURACY_COLORS)
        for i in range(n):
            colors[i] = accuracy_colors_override[i]
        for i in range(n, _NUM_STREAMING_ACCURACY_COLORS):
            colors[i] = (0.0, 0.0, 0.0, 1.0)
    return FDebugViewModeUniformParameters(
        accuracy_colors=colors,
        uv_density_sampling=uv_density_sampling,
    )

















def get_shader_instruction_count_for_baseline(
    shading_path: EShadingPath,
    is_vertex_shader: bool,
    is_unlit: bool,
    baseline: ShaderComplexityBaseline | None = None,
) -> int:
    """
    返回给定着色器类型的指令数基线，供复杂度对比使用。
    鲁迅式：基线指令数是工程师对"正常"的定义——超出则问责，不足则怀疑。
    """
    b = baseline or _default_shader_complexity_baseline
    if shading_path == EShadingPath.Forward:
        if is_vertex_shader: return b.forward_vs
        return b.forward_unlit_ps if is_unlit else b.forward_ps
    elif shading_path == EShadingPath.Mobile:
        if is_vertex_shader: return b.mobile_forward_vs
        return b.mobile_forward_unlit_ps if is_unlit else b.mobile_forward_ps
    else:  # Deferred
        if is_vertex_shader: return b.deferred_vs
        return b.deferred_unlit_ps if is_unlit else b.deferred_ps

















def compute_shader_complexity_ratio(
    instruction_count: int,
    shading_path: EShadingPath,
    is_vertex_shader: bool,
    is_unlit: bool,
    baseline: ShaderComplexityBaseline | None = None,
) -> float:
    """
    计算着色器复杂度比值（实际指令数 / 基线指令数）。
    返回值 > 1.0 表示超出基线，供贴花与调试视图模式颜色映射使用。
    鲁迅式：比值大于一，说明代码已经比标准更费力气——
    或者标准定得太低，这两种可能同样令人不安。
    """
    baseline_count = get_shader_instruction_count_for_baseline(
        shading_path, is_vertex_shader, is_unlit, baseline
    )
    if baseline_count <= 0:
        return 0.0
    return instruction_count / baseline_count


# =============================================================================
# UE5 Renderer Port — BasePass · ClusteredDeferred · Anisotropy · Depth · CustomDepth
#
# 鲁迅式：世上本没有渲染管线，走的人多了，便成了 G-Buffer。
# =============================================================================

import math as _math
from dataclasses import dataclass as _dataclass, field as _field
from typing import Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple


# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------

class _Vec3:
    """Minimal float3 — cheaper than numpy for single-cell ops."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):  return _Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __mul__(self, s):  return _Vec3(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return self.__mul__(s)

    def dot(self, o) -> float:
        return self.x*o.x + self.y*o.y + self.z*o.z

    def length(self) -> float:
        return _math.sqrt(self.dot(self))

    def normalize(self):
        d = self.length()
        return _Vec3(self.x/d, self.y/d, self.z/d) if d > 1e-9 else _Vec3(0, 0, 1)

    def clamp01(self):
        return _Vec3(max(0.0, min(1.0, self.x)),
                     max(0.0, min(1.0, self.y)),
                     max(0.0, min(1.0, self.z)))

    def as_tuple(self) -> _Tuple[float, float, float]:
        return (self.x, self.y, self.z)

















def _saturate(v: float) -> float:
    return max(0.0, min(1.0, v))

















def _pow_safe(base: float, exp: float) -> float:
    return _math.pow(max(base, 1e-9), exp)


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass



# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass
class AstroCellGBuffer:
    """
    Reduced GBuffer written by the BasePass.

    Mirrors the fields consumed by ClusteredDeferredShadingPixelShader:
      SceneColor   — HDR scene-colour pre-lighting (emissive + baked)
      WorldNormal  — world-space shading normal  [−1, 1]³
      BaseColor    — albedo / diffuse colour      [0, 1]³
      Roughness    — GGX roughness               [0, 1]
      Metallic     — metallic mask               [0, 1]
      Anisotropy   — tangent-space anisotropy    [−1, 1]  (AnisotropyRendering.cpp)
      Depth        — linear eye depth            ≥ 0
      CustomDepth  — custom depth value or −1 if not written
      Stencil      — custom stencil byte
    """
    scene_color:   _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 0))
    world_normal:  _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 1))
    base_color:    _Vec3  = _field(default_factory=lambda: _Vec3(0.5, 0.5, 0.5))
    roughness:     float  = 0.5
    metallic:      float  = 0.0
    anisotropy:    float  = 0.0      # written by FAnisotropyPS (AnisotropyRendering.cpp)
    depth:         float  = 1.0      # TDepthOnlyVS / FDepthOnlyPS (DepthRendering.cpp)
    custom_depth:  float  = -1.0     # FCustomDepthPassParameters (CustomDepthRendering.cpp)
    stencil:       int    = 0


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass



# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass
class AstroCellMaterial:
    """
    Subset of UMaterial properties consumed during the BasePass.
    Corresponds to the FMaterial / FMaterialRenderProxy interface.
    """
    base_color:       _Vec3  = _field(default_factory=lambda: _Vec3(0.8, 0.8, 0.8))
    emissive:         _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 0))
    roughness:        float  = 0.5
    metallic:         float  = 0.0
    anisotropy:       float  = 0.0   # bHasAnisotropyConnected
    opacity_mask:     float  = 1.0   # clip() threshold in masked materials
    writes_velocity:  bool   = False
    is_translucent:   bool   = False

















class AstroCellBasePass:
    """
    Encodes one cell's material into a GBuffer entry.

    Mimics the per-draw-call work done by TBasePassPS::MainPS():
      1. Material evaluation  → base-colour, roughness, metallic, emissive
      2. Normal encoding      → world-space normal written to GBufferA
      3. Emissive accumulation into SceneColor
      4. Anisotropy output    → forwarded to AnisotropyRendering pass

    SelectiveBasePassOutputs (r.SelectiveBasePassOutputs) is always
    treated as enabled here: we only populate gbuffer slots we use.
    """

    # r.SelectiveBasePassOutputs = 1  (compile-time const in UE5)
    SELECTIVE_OUTPUTS: bool = True

    def __init__(self, enable_anisotropy: bool = True):
        # r.AnisotropicMaterials equivalent
        self._anisotropy_enabled = enable_anisotropy

    def encode(
        self,
        material:     AstroCellMaterial,
        vertex_world_normal: _Vec3,
        eye_depth:    float,
    ) -> AstroCellGBuffer:
        """
        Execute the BasePass pixel-shader logic for a single cell.

        Parameters
        ----------
        material            : evaluated material parameters
        vertex_world_normal : interpolated vertex normal (world space)
        eye_depth           : linear depth from camera
        """
        gbuf = AstroCellGBuffer()

        # --- opacity / masked discard (clip() equivalent) -----------------
        if material.opacity_mask < 0.333:
            # Masked material fully discarded → leave gbuf at defaults.
            # Matches `clip(OpacityMask - GetMaskClipValue())` in HLSL.
            return gbuf

        # --- normal encoding ----------------------------------------------
        # GBufferA.xyz = WorldNormal (octahedral encode omitted for clarity)
        gbuf.world_normal = vertex_world_normal.normalize()

        # --- material properties ------------------------------------------
        gbuf.base_color = material.base_color.clamp01()
        gbuf.roughness  = _saturate(material.roughness)
        gbuf.metallic   = _saturate(material.metallic)

        # --- anisotropy (written only when r.AnisotropicMaterials is on) --
        # Mirrors FAnisotropyPS writing into the anisotropy GBuffer channel.
        if self._anisotropy_enabled:
            gbuf.anisotropy = max(-1.0, min(1.0, material.anisotropy))

        # --- emissive → SceneColor ----------------------------------------
        # UE5 BasePass: SceneColor.rgb += Emissive * View.PreExposure
        pre_exposure = 1.0
        gbuf.scene_color = (material.emissive * pre_exposure).clamp01()

        # --- depth prepass value (DepthRendering.cpp) ----------------------
        # TDepthOnlyVS outputs SV_Depth; we store linear eye depth.
        gbuf.depth = max(0.0, eye_depth)

        return gbuf


# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。



# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。



# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。



# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。




# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。



# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。



# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。



# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。

class AstroCellDepthPass:
    """
    EarlyZ prepass for a collection of cells.

    Mimics FDepthPassMeshProcessor::AddMeshBatch() + TDepthOnlyVS.
    Produces a per-cell depth buffer that later passes (BasePass,
    ClusteredDeferred) use for depth-equal or depth-less-equal tests.

    EarlyZPassMode is hardcoded to DDM_AllOpaque (the most common
    production setting and the one that matches our cell-centric use).
    """

    # r.EarlyZSortMasked = 1: masked draws go last (better early-z util)
    SORT_MASKED_LAST: bool = True

    def run(
        self,
        cells: _List[_Dict],
    ) -> _Dict[str, float]:
        """
        Execute the depth prepass.

        Parameters
        ----------
        cells : list of dicts with keys
                  'id'      : str
                  'depth'   : float  (linear eye depth)
                  'masked'  : bool   (True → has opacity_mask < 1)

        Returns
        -------
        depth_buffer : {cell_id: depth}
        """
        # r.EarlyZSortMasked — opaque draws before masked draws
        if self.SORT_MASKED_LAST:
            cells = sorted(cells, key=lambda c: (1 if c.get("masked") else 0, c["depth"]))

        depth_buffer: _Dict[str, float] = {}
        for cell in cells:
            cid   = cell["id"]
            depth = float(cell["depth"])
            # Depth-less test: keep closest (front-to-back render)
            if cid not in depth_buffer or depth < depth_buffer[cid]:
                depth_buffer[cid] = depth

        return depth_buffer


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass



# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass
class AstroCellCustomDepthRequest:
    """Mirrors the per-primitive CustomDepth render request."""
    cell_id:       str
    depth:         float   # eye-space depth of the custom-depth surface
    stencil_value: int = 0  # r.CustomDepth = 3 → EnabledWithStencil

















class AstroCellCustomDepthPass:
    """
    Renders custom-depth and custom-stencil for flagged cells.

    Corresponds to FCustomDepthPassParameters + the RDG pass in
    CustomDepthRendering.cpp::RenderCustomDepthPass().

    Ordering (r.CustomDepth.Order):
      0 → BeforeBasePass  (default when DBuffer is enabled)
      1 → AfterBasePass
    We store the result and let the caller decide when to apply it.
    """

    def __init__(self, order: int = 0, writes_stencil: bool = True):
        # r.CustomDepth.Order
        self.before_base_pass: bool = (order == 0)
        # r.CustomDepth = 3 (EnabledWithStencil)
        self.writes_stencil: bool = writes_stencil

    def run(
        self,
        requests: _List[AstroCellCustomDepthRequest],
        scene_depth_buffer: _Optional[_Dict[str, float]] = None,
    ) -> _Dict[str, _Tuple[float, int]]:
        """
        Execute the custom-depth pass.

        Parameters
        ----------
        requests          : list of custom-depth draw requests
        scene_depth_buffer: optional scene depth buffer for occlusion test
                            (None → no occlusion; all writes pass)

        Returns
        -------
        {cell_id: (custom_depth, stencil_value)}
        """
        # FCustomDepthTextures::Create clears to depth_far=1e9
        DEPTH_FAR = 1e9
        out: _Dict[str, _Tuple[float, int]] = {}

        for req in requests:
            # Depth test against scene geometry (CF_LessEqual in UE5 default)
            if scene_depth_buffer is not None:
                scene_d = scene_depth_buffer.get(req.cell_id, DEPTH_FAR)
                if req.depth > scene_d:
                    continue  # occluded → discard

            stencil = req.stencil_value if self.writes_stencil else 0
            # Keep the nearest custom-depth value per cell
            if req.cell_id not in out or req.depth < out[req.cell_id][0]:
                out[req.cell_id] = (req.depth, stencil)

        return out


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass



# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass
class AstroCellLight:
    """
    Minimal punctual-light descriptor used by the light grid.
    Mirrors FLocalLightData in the forward light SSBO.
    """
    position:   _Vec3
    color:      _Vec3  = _field(default_factory=lambda: _Vec3(1, 1, 1))
    intensity:  float  = 1.0
    radius:     float  = 10.0
    # Anisotropic materials need the light's tangent influence weight
    aniso_weight: float = 1.0

















class AstroCellClusteredDeferredShadingPass:
    """
    Per-cell clustered deferred shading.

    Corresponds to FClusteredShadingPS::ClusteredShadingPixelShader().

    Algorithm
    ---------
    1. For each cell, look up its GBuffer entry.
    2. Iterate over lights assigned to the cell's cluster (here: all
       lights within radius — the grid-based culling is elided).
    3. Accumulate diffuse + specular from each light using the same
       BRDF model that FClusteredShadingPS calls into.
    4. Support anisotropic materials (FAnistropicMaterials permutation):
       swap isotropic GGX for the Ward-Dür lobe when anisotropy ≠ 0.
    5. Add emissive contribution from SceneColor (written by BasePass).

    r.UseClusteredDeferredShading_ToBeRemoved must be non-zero — we
    assume it is enabled (the caller is responsible for the guard).
    """

    # SM6 feature-level guard (ShouldUseClusteredDeferredShading)
    REQUIRES_SM6: bool = True

    def __init__(
        self,
        supports_anisotropy: bool = True,
        ambient: _Vec3 = None,
    ):
        # SUPPORTS_ANISOTROPIC_MATERIALS permutation
        self._aniso_enabled = supports_anisotropy
        # Simple sky / ambient term (replaces full IBL for cell use)
        self._ambient = ambient or _Vec3(0.05, 0.05, 0.07)

    def _cluster_lights_for_cell(
        self,
        cell_world_pos: _Vec3,
        lights: _List[AstroCellLight],
    ) -> _List[AstroCellLight]:
        """
        Returns lights whose range encompasses the cell world position.
        Mimics the light-grid lookup in ClusteredDeferredShadingPixelShader.
        """
        result = []
        for light in lights:
            dx = cell_world_pos.x - light.position.x
            dy = cell_world_pos.y - light.position.y
            dz = cell_world_pos.z - light.position.z
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 <= light.radius * light.radius:
                result.append(light)
        return result

    def shade(
        self,
        cell_id:        str,
        gbuffer:        AstroCellGBuffer,
        world_position: _Vec3,
        view_dir:       _Vec3,
        lights:         _List[AstroCellLight],
        tangent:        _Optional[_Vec3] = None,
    ) -> _Vec3:
        """
        Compute final lit colour for one cell.

        Parameters
        ----------
        cell_id        : identifier (for debug only)
        gbuffer        : GBuffer data written by AstroCellBasePass
        world_position : world-space surface position of the cell
        view_dir       : unit vector toward the camera
        lights         : all scene lights (will be cluster-culled internally)
        tangent        : world-space surface tangent (needed for aniso BRDF)

        Returns
        -------
        lit colour as _Vec3 (linear, pre-tonemapped)
        """
        n = gbuffer.world_normal
        v = view_dir.normalize()
        t = (tangent or _Vec3(1, 0, 0)).normalize()

        base   = gbuffer.base_color
        rough  = gbuffer.roughness
        metal  = gbuffer.metallic
        aniso  = gbuffer.anisotropy if self._aniso_enabled else 0.0

        # Derive specular colour (UE4/5 metallic workflow)
        f0 = _Vec3(0.04, 0.04, 0.04)
        spec_color = _Vec3(
            f0.x + (base.x - f0.x) * metal,
            f0.y + (base.y - f0.y) * metal,
            f0.z + (base.z - f0.z) * metal,
        )
        diff_color = _Vec3(
            base.x * (1.0 - metal),
            base.y * (1.0 - metal),
            base.z * (1.0 - metal),
        )

        # Ambient / indirect
        acc = _Vec3(
            diff_color.x * self._ambient.x,
            diff_color.y * self._ambient.y,
            diff_color.z * self._ambient.z,
        )

        # Cluster-cull lights  (r.UseClusteredDeferredShading path)
        active_lights = self._cluster_lights_for_cell(world_position, lights)

        for light in active_lights:
            # Light vector + attenuation  (matches GetLocalLightAttenuation)
            dx = light.position.x - world_position.x
            dy = light.position.y - world_position.y
            dz = light.position.z - world_position.z
            dist = _math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-9:
                continue

            l = _Vec3(dx/dist, dy/dist, dz/dist)

            # Inverse-square falloff with radius clamp (UE5 PointLight)
            falloff = _pow_safe(max(0.0, 1.0 - (dist/light.radius)**4), 2.0)
            falloff /= (dist*dist + 1.0)
            irradiance = light.intensity * falloff

            lc = light.color
            ndotl = _saturate(n.dot(l))

            # --- diffuse (Lambertian) -------------------------------------
            acc = _Vec3(
                acc.x + diff_color.x * lc.x * ndotl * irradiance,
                acc.y + diff_color.y * lc.y * ndotl * irradiance,
                acc.z + diff_color.z * lc.z * ndotl * irradiance,
            )

            # --- specular ------------------------------------------------
            if abs(aniso) > 0.01 and self._aniso_enabled:
                # SUPPORTS_ANISOTROPIC_MATERIALS permutation
                # Remap scalar anisotropy → (roughness_u, roughness_v)
                # following UE5 GetAnisotropicRoughness()
                ru = _saturate(rough * (1.0 + aniso))
                rv = _saturate(rough * (1.0 - aniso))
                # Use light's aniso_weight as a per-light tangent scale
                spec_val = astro_anisotropy_brdf(n, v, l, t, ru, rv)
                spec_val *= light.aniso_weight
            else:
                spec_val = _ggx_specular(n, v, l, rough)

            acc = _Vec3(
                acc.x + spec_color.x * lc.x * spec_val * irradiance,
                acc.y + spec_color.y * lc.y * spec_val * irradiance,
                acc.z + spec_color.z * lc.z * spec_val * irradiance,
            )

        # Add BasePass emissive contribution from SceneColor
        sc = gbuffer.scene_color
        acc = _Vec3(acc.x + sc.x, acc.y + sc.y, acc.z + sc.z)

        return acc.clamp01()

    def run(
        self,
        gbuffer_map:    _Dict[str, AstroCellGBuffer],
        world_positions: _Dict[str, _Vec3],
        view_dir:       _Vec3,
        lights:         _List[AstroCellLight],
        tangents:       _Optional[_Dict[str, _Vec3]] = None,
    ) -> _Dict[str, _Vec3]:
        """
        Shade all cells.

        Returns
        -------
        {cell_id: lit_colour_vec3}
        """
        tangents = tangents or {}
        return {
            cid: self.shade(
                cid,
                gbuf,
                world_positions.get(cid, _Vec3(0, 0, 0)),
                view_dir,
                lights,
                tangents.get(cid),
            )
            for cid, gbuf in gbuffer_map.items()
        }


# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。




# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。



# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。

class AstroCellRenderPipeline:
    """
    Full deferred pipeline for the cell layer, porting:

      1. DepthPass          (DepthRendering.cpp)
      2. CustomDepthPass    (CustomDepthRendering.cpp)
      3. BasePass → GBuffer (BasePassRendering.cpp + AnisotropyRendering.cpp)
      4. ClusteredDeferred  (ClusteredDeferredShadingPass.cpp)

    Mirrors the high-level FDeferredShadingSceneRenderer::Render() sequence
    as applied to a dict of SVG/astro cells rather than mesh draw commands.
    """

    def __init__(
        self,
        enable_anisotropy:    bool  = True,
        enable_clustered:     bool  = True,
        custom_depth_order:   int   = 0,
        custom_depth_stencil: bool  = True,
        ambient:              _Optional[_Vec3] = None,
    ):
        self._depth_pass    = AstroCellDepthPass()
        self._custom_depth  = AstroCellCustomDepthPass(
            order=custom_depth_order,
            writes_stencil=custom_depth_stencil,
        )
        self._base_pass     = AstroCellBasePass(enable_anisotropy=enable_anisotropy)
        self._clustered     = AstroCellClusteredDeferredShadingPass(
            supports_anisotropy=enable_anisotropy,
            ambient=ambient,
        )
        self._enable_clustered = enable_clustered

    def render(
        self,
        cells:              _List[_Dict],
        materials:          _Dict[str, AstroCellMaterial],
        world_normals:      _Dict[str, _Vec3],
        world_positions:    _Dict[str, _Vec3],
        eye_depths:         _Dict[str, float],
        view_dir:           _Vec3,
        lights:             _List[AstroCellLight],
        custom_depth_reqs:  _Optional[_List[AstroCellCustomDepthRequest]] = None,
        tangents:           _Optional[_Dict[str, _Vec3]] = None,
    ) -> _Dict[str, _Tuple[_Vec3, AstroCellGBuffer, _Optional[_Tuple[float, int]]]]:
        """
        Run the full pipeline.

        Parameters
        ----------
        cells             : list of {'id', 'depth', 'masked'}
        materials         : {cell_id: AstroCellMaterial}
        world_normals     : {cell_id: Vec3}
        world_positions   : {cell_id: Vec3}
        eye_depths        : {cell_id: float}
        view_dir          : camera forward vector (unit)
        lights            : scene lights for clustered pass
        custom_depth_reqs : optional custom-depth draw requests
        tangents          : optional per-cell tangent vectors (aniso)

        Returns
        -------
        {cell_id: (lit_colour, gbuffer, (custom_depth, stencil) | None)}
        """
        # --- 1. Depth prepass -------------------------------------------
        depth_buf = self._depth_pass.run(cells)

        # --- 2. Custom depth (BeforeBasePass order) ----------------------
        custom_depth_buf: _Dict[str, _Tuple[float, int]] = {}
        if custom_depth_reqs and self._custom_depth.before_base_pass:
            custom_depth_buf = self._custom_depth.run(
                custom_depth_reqs, depth_buf
            )

        # --- 3. BasePass → GBuffer --------------------------------------
        gbuffer_map: _Dict[str, AstroCellGBuffer] = {}
        for cell in cells:
            cid   = cell["id"]
            mat   = materials.get(cid, AstroCellMaterial())
            norm  = world_normals.get(cid, _Vec3(0, 0, 1))
            depth = eye_depths.get(cid, depth_buf.get(cid, 1.0))

            gbuf = self._base_pass.encode(mat, norm, depth)

            # Stamp custom-depth into GBuffer if available
            if cid in custom_depth_buf:
                gbuf.custom_depth, gbuf.stencil = custom_depth_buf[cid]

            gbuffer_map[cid] = gbuf

        # --- 4. Custom depth (AfterBasePass order) ----------------------
        if custom_depth_reqs and not self._custom_depth.before_base_pass:
            extra = self._custom_depth.run(custom_depth_reqs, depth_buf)
            custom_depth_buf.update(extra)
            for cid, (cd, st) in extra.items():
                if cid in gbuffer_map:
                    gbuffer_map[cid].custom_depth = cd
                    gbuffer_map[cid].stencil      = st

        # --- 5. Clustered deferred shading ------------------------------
        if self._enable_clustered:
            lit_colours = self._clustered.run(
                gbuffer_map, world_positions, view_dir, lights, tangents
            )
        else:
            # Fall back to unlit (emissive only)
            lit_colours = {
                cid: gbuf.scene_color for cid, gbuf in gbuffer_map.items()
            }

        # --- Assemble output --------------------------------------------
        out: _Dict[str, _Tuple[_Vec3, AstroCellGBuffer, _Optional[_Tuple[float, int]]]] = {}
        for cid, gbuf in gbuffer_map.items():
            cd_entry = custom_depth_buf.get(cid)
            out[cid] = (lit_colours.get(cid, _Vec3(0, 0, 0)), gbuf, cd_entry)

        return out

# ═══════════════════════════════════════════════════════════════════════════════
# §  LUMEN SCENE MANAGEMENT  ── port of LumenScene.cpp / LumenSceneLighting.cpp
#    LumenSurfaceCache.cpp / LumenScreenSpaceBentNormal.cpp
#    LumenScreenProbeFiltering.cpp / LumenScreenProbeImportanceSampling.cpp
#
#    鲁迅式：旧中国有一句话——"万里长城今犹在，不见当年秦始皇"。
#    Lumen 的 Scene 管理亦然：距离场飘散，细胞仍在，光照已非昨日。
# ═══════════════════════════════════════════════════════════════════════════════

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

_LUMEN_GLOBAL_DF_RESOLUTION: int = 252
_LUMEN_GLOBAL_DF_CLIPMAP_EXTENT: float = 2500.0
_LUMEN_FAR_FIELD_MAX_TRACE_DISTANCE: float = 1.0e6
_LUMEN_FAR_FIELD_DITHER_SCALE: float = 200.0
_LUMEN_SURFACE_CACHE_ATLAS_SIZE: int = 4096
_LUMEN_PHYSICAL_PAGE_SIZE: int = 128
_LUMEN_CARD_TILE_SIZE: int = 8


@dataclass




@dataclass





@dataclass




@dataclass


class SurfaceCacheCompression(Enum):
    """
    ESurfaceCacheCompression.

    鲁迅式：能压缩的都压缩了，剩下的才叫"真实"。
    Disabled 是诚实，UAVAliasing 是取巧，CopyTextureRegion 是代价最贵的虚伪。
    """
    DISABLED            = auto()
    UAV_ALIASING        = auto()
    FRAMEBUFFER         = auto()
    COPY_TEXTURE_REGION = auto()

















class SurfaceCacheLayer(Enum):
    DEPTH    = 0
    ALBEDO   = 1
    NORMAL   = 2
    EMISSIVE = 3


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class SurfaceLayerConfig:
    name:             str
    uncompressed_fmt: str
    compressed_fmt:   str
    clear_value:      tuple = (0.0, 0.0, 0.0)


_SURFACE_LAYER_CONFIGS = {
    SurfaceCacheLayer.DEPTH:    SurfaceLayerConfig("Depth",   "PF_G16",            "PF_Unknown", (1.0, 0.0, 0.0)),
    SurfaceCacheLayer.ALBEDO:   SurfaceLayerConfig("Albedo",  "PF_R8G8B8A8",       "PF_BC7",     (0.0, 0.0, 0.0)),
    SurfaceCacheLayer.NORMAL:   SurfaceLayerConfig("Normal",  "PF_R8G8",           "PF_BC5",     (0.0, 0.0, 0.0)),
    SurfaceCacheLayer.EMISSIVE: SurfaceLayerConfig("Emissive","PF_FloatR11G11B10", "PF_BC6H",    (0.0, 0.0, 0.0)),
}

















class SurfaceCacheDilationMode(Enum):
    """
    r.LumenScene.SurfaceCache.DilationMode.

    鲁迅式：蔓延一个像素，就是"宽容"；蔓延整张图，就是"谎言"。
    """
    DISABLED  = 0
    TWO_SIDED = 1
    ALL       = 2


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class ShortRangeAOConfig:
    """
    Runtime configuration for the Screen-Space Bent Normal pass.

    鲁迅式：法线弯曲了，光就遮住了；光遮住了，就叫做"环境遮蔽"。
    名字越高雅，背后的道理越朴素。
    """
    use_bent_normal:          bool  = True
    use_temporal:             bool  = True
    use_horizon_search:       bool  = True
    use_hzb:                  bool  = True
    downsample_factor:        int   = _SSBN_DOWNSAMPLE_FACTOR
    slice_count:              int   = _SSBN_SLICE_COUNT
    steps_per_slice:          int   = _SSBN_STEPS_PER_SLICE
    foliage_occ_strength:     float = _SSBN_FOLIAGE_OCC_STRENGTH
    max_multibounce_albedo:   float = _SSBN_MAX_MULTIBOUNCE_ALBEDO
    slope_tolerance:          float = _SSBN_SLOPE_TOLERANCE
    foreground_reject_frac:   float = _SSBN_FOREGROUND_REJECT_FRACTION
    apply_during_integration: bool  = False
    allow_async_compute:      bool  = True

    def texture_format(self):
        return "PF_R32_UINT" if self.use_bent_normal else "PF_R8"

















def _horizon_angle(depth_samples, view_z, slice_dir, pixel_pitch,
                   slope_tolerance=_SSBN_SLOPE_TOLERANCE):
    max_horizon = -math.pi / 2.0
    for i, d in enumerate(depth_samples):
        if d <= 0.0:
            continue
        dist = (i + 1) * pixel_pitch
        angle = math.atan2(view_z - d, dist)
        if (view_z - d) > view_z * _SSBN_FOREGROUND_REJECT_FRACTION:
            continue
        if angle > max_horizon + slope_tolerance * 0.01:
            max_horizon = angle
    return max_horizon

















def compute_bent_normal_ao(depth_buffer, normal_buffer, config=None, pixel_pitch=1.0):
    """
    Software reference for the Screen-Space Bent Normal pass.

    鲁迅式：把每个像素的天空扫描一遍，记录哪些方向被堵死——
    这就是所谓的"弯曲法线"，是对遮蔽的精确统计，而非诗意描述。
    """
    if config is None:
        config = ShortRangeAOConfig()
    height = len(depth_buffer)
    width  = len(depth_buffer[0]) if height > 0 else 0
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            view_z = depth_buffer[y][x]
            normal = normal_buffer[y][x]
            total_occ = 0.0
            bent = [0.0, 0.0, 0.0]
            for s in range(config.slice_count):
                a = math.pi * s / config.slice_count
                sd = (math.cos(a), math.sin(a))
                samples = []
                for step in range(config.steps_per_slice):
                    sx = x + int(round((step + 1) * sd[0]))
                    sy = y + int(round((step + 1) * sd[1]))
                    samples.append(
                        depth_buffer[sy][sx] if 0 <= sx < width and 0 <= sy < height else 0.0
                    )
                h = _horizon_angle(samples, view_z, sd, pixel_pitch, config.slope_tolerance)
                occ = max(0.0, 1.0 - math.sin(max(h, 0.0)))
                total_occ += occ
                u = math.cos(h)
                bent[0] += sd[0] * u
                bent[1] += sd[1] * u
            ao = min(total_occ / max(config.slice_count, 1), config.foliage_occ_strength)
            alb = min(max(sum(normal) / 3.0, 0.0), config.max_multibounce_albedo)
            ao  = ao / (1.0 - alb * (1.0 - ao) + 1e-6)
            ln = math.sqrt(bent[0]**2 + bent[1]**2 + bent[2]**2) + 1e-8
            row.append((bent[0]/ln, bent[1]/ln, bent[2]/ln, ao))
        result.append(row)
    return result


_SPF_SPATIAL_PASSES          = 3
_SPF_DISOCCLUSION_FRAMES     = 4
_SPF_DISOCCLUSION_FRAC       = 0.4
_SPF_POSITION_WEIGHT_SCALE   = 1000.0
_SPF_MAX_RADIANCE_HIT_ANGLE  = 10.0
_SPF_HISTORY_WEIGHT          = 0.5
_SPF_HISTORY_DIST_THRESHOLD  = 30.0
_SPF_MAX_RAY_INTENSITY       = 10.0


@dataclass




@dataclass





@dataclass




@dataclass


def _spatial_weight(pa, pb, scale=_SPF_POSITION_WEIGHT_SCALE):
    dx = pa.world_pos[0] - pb.world_pos[0]
    dy = pa.world_pos[1] - pb.world_pos[1]
    dz = pa.world_pos[2] - pb.world_pos[2]
    return math.exp(-(dx*dx + dy*dy + dz*dz) * scale * 1e-6)

















def composite_traces_with_scatter(probes, max_ray_intensity=_SPF_MAX_RAY_INTENSITY):
    """
    FScreenProbeCompositeTracesWithScatterCS — clamp firefly radiance.

    鲁迅式：每条光线都有最大亮度。亮过头的，削掉。
    现实主义从不允许辉光超标。
    """
    for p in probes:
        if p.radiance:
            p.radiance = [min(r, max_ray_intensity) for r in p.radiance]
    return probes

















def temporally_accumulate_probe_radiance(
    probes, history_probes,
    history_weight=_SPF_HISTORY_WEIGHT,
    dist_threshold=_SPF_HISTORY_DIST_THRESHOLD,
):
    """
    FScreenProbeTemporallyAccumulateTraceRadianceCS.

    鲁迅式：历史是有重量的——但只在距离够近的时候。
    太远了，就当没发生过，重新开始。
    """
    hmap = {p.probe_id: p for p in history_probes}
    for probe in probes:
        hist = hmap.get(probe.probe_id)
        if hist is None or not hist.radiance:
            probe.frames_accumulated = 1
            continue
        dx = probe.world_pos[0] - hist.world_pos[0]
        dy = probe.world_pos[1] - hist.world_pos[1]
        dz = probe.world_pos[2] - hist.world_pos[2]
        if math.sqrt(dx*dx + dy*dy + dz*dz) > dist_threshold:
            probe.frames_accumulated = 1
            continue
        n = min(len(probe.radiance), len(hist.radiance))
        blended = [
            probe.radiance[i] * (1.0 - history_weight) + hist.radiance[i] * history_weight
            for i in range(n)
        ] + probe.radiance[n:]
        probe.radiance = probe.history_radiance = blended
        probe.frames_accumulated = hist.frames_accumulated + 1
    return probes

















def spatial_filter_probes(
    probes,
    num_passes=_SPF_SPATIAL_PASSES,
    disocclusion_max_frames=_SPF_DISOCCLUSION_FRAMES,
    disocclusion_frac=_SPF_DISOCCLUSION_FRAC,
    position_weight_scale=_SPF_POSITION_WEIGHT_SCALE,
):
    """
    Multi-pass bilateral spatial filter over the probe grid.

    鲁迅式：遮蔽区域的噪声用邻居来弥补，这是所谓"空间滤波"。
    没有历史的地方，就靠周围的人说话。
    """
    for _pass in range(num_passes):
        updated = []
        for i, probe in enumerate(probes):
            if _pass > 0 and probe.frames_accumulated >= disocclusion_max_frames:
                updated.append(probe)
                continue
            w_sum = 1.0
            accum = list(probe.radiance)
            for j, other in enumerate(probes):
                if i == j:
                    continue
                w = _spatial_weight(probe, other, position_weight_scale)
                if w < 1e-4:
                    continue
                n = min(len(accum), len(other.radiance))
                for k in range(n):
                    accum[k] += other.radiance[k] * w
                w_sum += w
            probe.radiance = [v / w_sum for v in accum]
            updated.append(probe)
        probes = updated
    return probes


_IS_ENABLED               = True
_IS_INCOMING_LIGHTING     = True
_IS_PROBE_RADIANCE_HIST   = True
_IS_BRDF_OCTAHEDRON_RES   = 8
_IS_MIN_PDF_TO_TRACE       = 0.1
_IS_HISTORY_DIST_THRESHOLD = 30.0

















def _octahedron_dir(u, v):
    fx = u * 2.0 - 1.0
    fy = v * 2.0 - 1.0
    fz = 1.0 - abs(fx) - abs(fy)
    if fz < 0.0:
        ox = (1.0 - abs(fy)) * (1.0 if fx >= 0 else -1.0)
        oy = (1.0 - abs(fx)) * (1.0 if fy >= 0 else -1.0)
        fx, fy = ox, oy
    ln = math.sqrt(fx*fx + fy*fy + fz*fz) + 1e-8
    return (fx/ln, fy/ln, fz/ln)

















def compute_lighting_pdf(probe, resolution=_IS_BRDF_OCTAHEDRON_RES,
                          use_history=_IS_PROBE_RADIANCE_HIST, history_weight=0.9):
    """
    FScreenProbeComputeLightingProbabilityDensityFunctionCS.

    鲁迅式：上一帧亮的方向，这一帧优先去看。
    这叫做"重要性采样"，也叫做"走捷径的学问"。
    """
    n   = resolution * resolution
    pdf = [0.0] * n
    for i in range(n):
        _, _, fz = _octahedron_dir((i % resolution + 0.5) / resolution,
                                    (i // resolution + 0.5) / resolution)
        pdf[i] = max(fz, 0.0)
    if use_history and probe.history_radiance:
        hn = min(n, len(probe.history_radiance))
        for i in range(hn):
            pdf[i] = pdf[i] * (1.0 - history_weight) + probe.history_radiance[i] * history_weight
    total = sum(pdf) + 1e-8
    return [p / total for p in pdf]

















def generate_importance_sampled_rays(
    probe,
    tracing_resolution=_IS_BRDF_OCTAHEDRON_RES,
    min_pdf=_IS_MIN_PDF_TO_TRACE,
    use_importance_sampling=_IS_ENABLED,
):
    """
    FScreenProbeGenerateRaysCS — select trace directions via PDF.

    鲁迅式：方向太暗的就不看了，把光阴省下来照有价值的地方。
    这是渲染的经济学，也是人生的经济学。
    """
    if not use_importance_sampling:
        n = tracing_resolution * tracing_resolution
        return [
            _octahedron_dir((i % tracing_resolution + 0.5) / tracing_resolution,
                             (i // tracing_resolution + 0.5) / tracing_resolution)
            for i in range(n)
        ]
    pdf  = compute_lighting_pdf(probe, tracing_resolution)
    rays = []
    for i, p in enumerate(pdf):
        if p < min_pdf:
            continue
        rays.append(_octahedron_dir(
            (i % tracing_resolution + 0.5) / tracing_resolution,
            (i // tracing_resolution + 0.5) / tracing_resolution,
        ))
    return rays


@dataclass




@dataclass





@dataclass




@dataclass


def get_max_ao_view_distance() -> float:
    return G_AO_MAX_VIEW_DISTANCE

















def use_ao_history_stability_pass() -> bool:
    return G_AO_HISTORY_STABILITY_PASS and G_DISTANCE_FIELD_AO_QUALITY >= 2


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass



# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DFAOParameters:
    """
    对应 FDistanceFieldAOParameters。
    距离场 AO 的两段衰减区间：近处用 Object SDF，远处用 Global SDF。
    如同人生：年轻时斤斤计较细节，年老时只看大略。
    """
    object_max_occlusion_distance: float = 600.0
    global_max_occlusion_distance: float = 0.0
    contrast: float = 1.0

    @classmethod
    def from_sky_light(cls, occlusion_max_distance: float, contrast: float) -> "DFAOParameters":
        contrast = max(0.01, min(2.0, contrast))
        occlusion_max_distance = max(2.0, min(3000.0, occlusion_max_distance))
        if G_AO_GLOBAL_DISTANCE_FIELD:
            obj_dist = min(occlusion_max_distance, G_AO_GLOBAL_DF_START_DISTANCE)
            glo_dist = occlusion_max_distance if occlusion_max_distance >= G_AO_GLOBAL_DF_START_DISTANCE else 0.0
        else:
            obj_dist = occlusion_max_distance
            glo_dist = 0.0
        return cls(object_max_occlusion_distance=obj_dist,
                   global_max_occlusion_distance=glo_dist,
                   contrast=contrast)


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class IntPoint:
    x: int = 0
    y: int = 0

    def divide_and_round_down(self, divisor: int) -> "IntPoint":
        return IntPoint(self.x // divisor, self.y // divisor)


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class DFObjectBounds:
    """场景中一个 Mesh SDF 对象的包围球：中心 + 半径。"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    object_index: int = 0


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class CulledObjectBuffer:
    """FCulledObjectBuffers 简化版 —— culling 之后存活的对象列表。"""
    object_indices: List[int] = field(default_factory=list)
    indirect_arg_count: int = 0   # 等价于 RWObjectIndirectArguments[0]


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class TileIntersectionData:
    """每个屏幕 tile 与若干 SDF 对象的交叉列表。"""
    tile_x: int = 0
    tile_y: int = 0
    object_indices: List[int] = field(default_factory=list)


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class BentNormalAO:
    """
    遮蔽后的弯曲法线 —— 方向表示最不遮蔽的方向，长度表示可见度。
    说白了：被压扁了多少，往哪个方向还能透口气。
    """
    bent_normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    occlusion: float = 0.0   # 0 = 完全不遮蔽，1 = 完全遮蔽


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class AOHistoryState:
    """对应 FTemporalAAHistory 在 DFAO 场景中的子集。"""
    bent_normal_history: List[BentNormalAO] = field(default_factory=list)
    valid: bool = False
    frame_index: int = 0


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class ScreenGridAOBuffer:
    """屏幕格 cone-trace 的中间结果。对应 FAOScreenGridParameters。"""
    width: int = 0
    height: int = 0
    # 每像素 NUM_CONE_SAMPLE_DIRECTIONS 个 float（遮蔽量）
    cone_depths: List[float] = field(default_factory=list)


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class DFSceneData:
    """
    FDistanceFieldSceneData 简化版。
    整个场景的 SDF 资产目录：对象包围盒、Atlas 纹理尺寸。
    没有这张清单，渲染器就是瞎子。
    """
    objects: List[DFObjectBounds] = field(default_factory=list)
    brick_atlas_dims: Tuple[int, int, int] = (64, 64, 64)
    num_objects_in_buffer: int = 0

    def add_object(self, obj: DFObjectBounds) -> None:
        if obj.radius > G_MESH_DF_MAX_OBJECT_BOUNDING_RADIUS:
            # 太大的对象被排除 —— 排除不代表不存在，只是不纳入计算。
            return
        self.objects.append(obj)
        self.num_objects_in_buffer += 1

    def remove_object(self, object_index: int) -> None:
        self.objects = [o for o in self.objects if o.object_index != object_index]
        self.num_objects_in_buffer = len(self.objects)


# ---------------------------------------------------------------------------
# AmbientOcclusion — 核心参数 / 采样方向
# ---------------------------------------------------------------------------

# 对应 SpacedVectors9：半球上均匀分布的 9 个方向
_SPACED_VECTORS_9: List[Tuple[float, float, float]] = [
    (-0.1840, 0.5545, 0.8117),
    ( 0.5404, 0.5404, 0.6455),
    ( 0.8117, 0.3124, 0.4944),
    ( 0.4944, 0.0000, 0.8693),
    (-0.0000, 0.0000, 1.0000),
    (-0.4944, 0.0000, 0.8693),
    (-0.8117, 0.3124, 0.4944),
    (-0.5404, 0.5404, 0.6455),
    ( 0.1840, 0.5545, 0.8117),
]

















def get_spaced_vectors(frame_number: int) -> List[Tuple[float, float, float]]:
    """
    按帧号旋转采样方向集合，实现时域超采样。
    不同的帧看到不同的采样，合起来才是完整的真相。
    """
    angle = (frame_number % 4) * (math.pi / 4.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    result = []
    for vx, vy, vz in _SPACED_VECTORS_9:
        rx = vx * cos_a - vy * sin_a
        ry = vx * sin_a + vy * cos_a
        result.append((rx, ry, vz))
    return result

















def compute_bent_normal_normalize_factor(sample_dirs: List[Tuple[float, float, float]]) -> float:
    """
    无遮蔽时所有 cone 方向的合向量长度的倒数 —— 归一化用。
    归一化之后，才能与别的数据相比较；不归一化的数字，像没有单位的重量。
    """
    ux = sum(d[0] for d in sample_dirs) / len(sample_dirs)
    uy = sum(d[1] for d in sample_dirs) / len(sample_dirs)
    uz = sum(d[2] for d in sample_dirs) / len(sample_dirs)
    mag = math.sqrt(ux*ux + uy*uy + uz*uz)
    return 1.0 / mag if mag > 1e-6 else 0.0

















def get_buffer_size_for_ao(view_width: int, view_height: int) -> IntPoint:
    """对应 GetBufferSizeForAO：按 G_AO_DOWNSAMPLE_FACTOR 降采样。"""
    return IntPoint(view_width // G_AO_DOWNSAMPLE_FACTOR,
                    view_height // G_AO_DOWNSAMPLE_FACTOR)

















def get_buffer_size_for_cone_tracing(view_width: int, view_height: int) -> IntPoint:
    ao = get_buffer_size_for_ao(view_width, view_height)
    w = max(ao.x // G_CONE_TRACE_DOWNSAMPLE_FACTOR, 1)
    h = max(ao.y // G_CONE_TRACE_DOWNSAMPLE_FACTOR, 1)
    return IntPoint(w, h)


# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------



def setup_object_buffer_parameters(scene_data: DFSceneData) -> dict:
    """
    对应 DistanceField::SetupObjectBufferParameters。
    返回一个参数字典，模拟 GPU SRV 绑定。
    """
    return {
        "num_scene_objects": scene_data.num_objects_in_buffer,
        "scene_objects": scene_data.objects,
    }

















def setup_atlas_parameters(scene_data: DFSceneData) -> dict:
    """
    对应 DistanceField::SetupAtlasParameters。
    Atlas 是所有 SDF brick 拼在一起的大纹理；这里只记录尺寸。
    """
    bx, by, bz = scene_data.brick_atlas_dims
    return {
        "brick_atlas_dims": (bx, by, bz),
        "brick_atlas_texel_size": (1.0/bx, 1.0/by, 1.0/bz),
    }


# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------

def _sphere_inside_frustum(
    center: Tuple[float, float, float],
    radius: float,
    frustum_planes: List[Tuple[float, float, float, float]],   # (nx,ny,nz,d)
) -> bool:
    """
    对应 GPU 端 CullObjectsForViewCS 的核心判断。
    六个平面，一个都不能放过；像检查一个人，每个方面都要审视。
    """
    cx, cy, cz = center
    for nx, ny, nz, d in frustum_planes:
        dist = nx*cx + ny*cy + nz*cz + d
        if dist < -radius:
            return False
    return True

















def cull_objects_to_view(
    scene_data: DFSceneData,
    frustum_planes: List[Tuple[float, float, float, float]],
    ao_params: DFAOParameters,
    view_origin: Tuple[float, float, float],
) -> CulledObjectBuffer:
    """
    对应 CullObjectsToView。
    先用视锥剔除，再按 AO 最大距离剔除。两重关卡，过了才算数。
    """
    buf = CulledObjectBuffer()
    ox, oy, oz = view_origin
    for obj in scene_data.objects:
        cx, cy, cz = obj.center
        dist_sq = (cx-ox)**2 + (cy-oy)**2 + (cz-oz)**2
        max_dist = ao_params.object_max_occlusion_distance + obj.radius
        if dist_sq > max_dist * max_dist:
            continue
        if dist_sq > G_AO_MAX_VIEW_DISTANCE**2:
            continue
        if not _sphere_inside_frustum(obj.center, obj.radius, frustum_planes):
            continue
        buf.object_indices.append(obj.object_index)
    buf.indirect_arg_count = len(buf.object_indices)
    return buf

















def build_tile_cones(
    view_width: int,
    view_height: int,
    tile_size_x: int = 16,
    tile_size_y: int = 16,
) -> List[dict]:
    """
    对应 FBuildTileConesCS。
    把屏幕分成 tile，每个 tile 计算一个包围锥（轴 + 半角余弦）。
    锥越紧，后续剔除越激进；锥越松，剔除越保守。
    """
    ao_size = get_buffer_size_for_ao(view_width, view_height)
    tiles_x = max(1, (ao_size.x + tile_size_x - 1) // tile_size_x)
    tiles_y = max(1, (ao_size.y + tile_size_y - 1) // tile_size_y)
    tiles = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # 简化：tile 中心方向 = (0,0,1)，半角余弦 = 0.5
            tiles.append({
                "tile_x": tx, "tile_y": ty,
                "cone_axis": (0.0, 0.0, 1.0),
                "cone_cos": 0.5,
                "depth_min": 0.0, "depth_max": G_AO_MAX_VIEW_DISTANCE,
            })
    return tiles

















def scatter_tile_culling(
    culled_buf: CulledObjectBuffer,
    tiles: List[dict],
    scene_data: DFSceneData,
) -> List[TileIntersectionData]:
    """
    对应 FObjectCullVS / FObjectCullPS —— 光栅化散射剔除。
    用球体的包围盒覆盖哪些 tile，就把该对象写入哪些 tile 的列表。
    规则简单，但执行一遍要遍历 N×M；复杂性从不消失，只是转移。
    """
    obj_map = {o.object_index: o for o in scene_data.objects}
    tile_data = [
        TileIntersectionData(tile_x=t["tile_x"], tile_y=t["tile_y"])
        for t in tiles
    ]
    num_tiles_x = max((t["tile_x"] for t in tiles), default=0) + 1

    for obj_idx in culled_buf.object_indices:
        obj = obj_map.get(obj_idx)
        if obj is None:
            continue
        # 简化：对象投影到所有 tile（实际应做包围矩形交集）
        for td in tile_data:
            td.object_indices.append(obj_idx)

    return tile_data


# ---------------------------------------------------------------------------
# ScreenGrid cone-trace AO
# ---------------------------------------------------------------------------

_JITTER_OFFSETS: List[Tuple[float, float]] = [
    (0.25, 0.00),
    (0.75, 0.25),
    (0.50, 0.75),
    (0.00, 0.50),
]

















def get_jitter_offset(frame_index: int, use_history: bool) -> Tuple[float, float]:
    """
    对应 GetJitterOffset。4 帧循环抖动，配合时域累积使用。
    抖动是一种诚实：承认单帧的采样不够，借历史来补足。
    """
    if G_AO_USE_JITTER and use_history:
        jx, jy = _JITTER_OFFSETS[frame_index % 4]
        return jx * G_CONE_TRACE_DOWNSAMPLE_FACTOR, jy * G_CONE_TRACE_DOWNSAMPLE_FACTOR
    return 0.0, 0.0

















def compute_screen_grid_ao(
    pixel_positions: List[Tuple[float, float, float]],  # 世界空间像素位置
    pixel_normals: List[Tuple[float, float, float]],
    objects: List[DFObjectBounds],
    ao_params: DFAOParameters,
    frame_number: int = 0,
    use_history: bool = True,
) -> List[BentNormalAO]:
    """
    对应 FConeTraceScreenGridObjectOcclusionCS 在一帧中的整体调度。
    每个像素发射 9 条 cone，统计遮蔽；弯曲法线是遮蔽方向的加权平均。
    九条 cone，九个证据，最终合议出一个结论。
    """
    sample_dirs = get_spaced_vectors(frame_number)
    normalize_factor = compute_bent_normal_normalize_factor(sample_dirs)
    tan_half_angle = math.tan(math.radians(16.0))   # ~AOConeHalfAngle
    jx, jy = get_jitter_offset(frame_number, use_history)
    results: List[BentNormalAO] = []

    for pos, normal in zip(pixel_positions, pixel_normals):
        # 将采样方向转到像素法线半球
        px, py, pz = pos
        nx, ny, nz = normal
        occ_sum = 0.0
        bent_x, bent_y, bent_z = 0.0, 0.0, 0.0

        for dx, dy, dz in sample_dirs:
            # 简化：不做切线空间旋转，直接以世界空间方向 trace
            cone_dir = (dx, dy, dz)
            occ = cone_trace_object_occlusion(
                ray_origin=(px + nx * 0.5, py + ny * 0.5, pz + nz * 0.5),
                cone_dir=cone_dir,
                tan_half_angle=tan_half_angle,
                max_distance=ao_params.object_max_occlusion_distance,
                objects=objects,
                traverse_mips=G_DF_AO_TRAVERSE_MIPS,
            )
            visibility = 1.0 - occ
            bent_x += dx * visibility
            bent_y += dy * visibility
            bent_z += dz * visibility
            occ_sum += occ

        avg_occ = occ_sum / NUM_CONE_SAMPLE_DIRECTIONS
        bx = bent_x * normalize_factor
        by = bent_y * normalize_factor
        bz = bent_z * normalize_factor
        mag = math.sqrt(bx*bx + by*by + bz*bz)
        if mag > 1e-6:
            bx /= mag; by /= mag; bz /= mag

        results.append(BentNormalAO(
            bent_normal=(bx, by, bz),
            occlusion=min(avg_occ, 1.0),
        ))
    return results


# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------

def compute_distance_fade(
    depth: float,
    fade_distance_scale: float = G_AO_VIEW_FADE_DISTANCE_SCALE,
) -> float:
    """
    对应 DistanceFadeScale：AO 在远距离线性淡出。
    远处的遮蔽本来就不可靠，淡出是对不确定性的诚实。
    """
    max_dist = get_max_ao_view_distance()
    fade_start = max_dist * fade_distance_scale
    if depth >= max_dist:
        return 0.0
    if depth <= fade_start:
        return 1.0
    return 1.0 - (depth - fade_start) / (max_dist - fade_start)

















def update_history_depth_rejection(
    current: List[BentNormalAO],
    history: AOHistoryState,
    current_depths: List[float],
    history_depths: List[float],
) -> List[BentNormalAO]:
    """
    对应 UpdateHistoryDepthRejectionPS。
    深度差异大的像素拒绝历史（防止 ghost）；差异小的融合历史（减少噪点）。
    历史是有条件接受的，不是无条件信任的。
    """
    if not history.valid or G_AO_CLEAR_HISTORY:
        return current

    blended: List[BentNormalAO] = []
    for i, (cur, hist) in enumerate(zip(current, history.bent_normal_history)):
        cd = current_depths[i] if i < len(current_depths) else 0.0
        hd = history_depths[i] if i < len(history_depths) else 0.0
        depth_diff = abs(cd - hd)
        if depth_diff > G_AO_HISTORY_DISTANCE_THRESHOLD:
            blended.append(cur)
        else:
            w = G_AO_HISTORY_WEIGHT
            bx = cur.bent_normal[0] * (1-w) + hist.bent_normal[0] * w
            by = cur.bent_normal[1] * (1-w) + hist.bent_normal[1] * w
            bz = cur.bent_normal[2] * (1-w) + hist.bent_normal[2] * w
            occ = cur.occlusion * (1-w) + hist.occlusion * w
            mag = math.sqrt(bx*bx + by*by + bz*bz)
            if mag > 1e-6:
                bx /= mag; by /= mag; bz /= mag
            blended.append(BentNormalAO(bent_normal=(bx, by, bz), occlusion=occ))
    return blended

















def filter_history_stability(
    ao_buffer: List[BentNormalAO],
    width: int,
    height: int,
) -> List[BentNormalAO]:
    """
    对应 FilterHistoryPS。在 AO 缓冲上做一次空间滤波，补洞、稳定。
    补洞是为了让结果看起来更完整；完整不等于正确，但看起来好一些。
    """
    if not use_ao_history_stability_pass():
        return ao_buffer
    filtered = list(ao_buffer)
    for i in range(len(filtered)):
        row, col = divmod(i, max(width, 1))
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbors.append(ao_buffer[nr * width + nc])
        if neighbors:
            avg_occ = sum(n.occlusion for n in neighbors) / len(neighbors)
            bx = sum(n.bent_normal[0] for n in neighbors) / len(neighbors)
            by = sum(n.bent_normal[1] for n in neighbors) / len(neighbors)
            bz = sum(n.bent_normal[2] for n in neighbors) / len(neighbors)
            mag = math.sqrt(bx*bx + by*by + bz*bz)
            if mag > 1e-6:
                bx /= mag; by /= mag; bz /= mag
            filtered[i] = BentNormalAO(bent_normal=(bx, by, bz), occlusion=avg_occ)
    return filtered

















def geometry_aware_upsample(
    ao_low: List[BentNormalAO],
    ao_low_width: int,
    ao_low_height: int,
    full_width: int,
    full_height: int,
    full_depths: List[float],
    ao_low_depths: List[float],
) -> List[BentNormalAO]:
    """
    对应 FGeometryAwareUpsamplePS。
    将低分辨率 AO 上采样回全分辨率；深度权重避免边缘模糊。
    上采样永远是在猜测：猜得有依据，猜错了也有借口。
    """
    result: List[BentNormalAO] = []
    for fy in range(full_height):
        for fx in range(full_width):
            lx = min(fx * ao_low_width // max(full_width, 1), ao_low_width - 1)
            ly = min(fy * ao_low_height // max(full_height, 1), ao_low_height - 1)
            low_idx = ly * ao_low_width + lx
            full_idx = fy * full_width + fx

            fd = full_depths[full_idx] if full_idx < len(full_depths) else 0.0
            ld = ao_low_depths[low_idx] if low_idx < len(ao_low_depths) else 0.0
            depth_weight = 1.0 / (1.0 + abs(fd - ld) * 0.1)

            if low_idx < len(ao_low):
                src = ao_low[low_idx]
                fade = compute_distance_fade(fd)
                final_occ = src.occlusion * depth_weight * fade
                result.append(BentNormalAO(
                    bent_normal=src.bent_normal,
                    occlusion=min(final_occ, 1.0),
                ))
            else:
                result.append(BentNormalAO())
    return result

















def update_ao_history(
    history: AOHistoryState,
    new_ao: List[BentNormalAO],
    frame_index: int,
) -> AOHistoryState:
    """将当前帧 AO 写入历史，供下一帧使用。记录，为了下次少走弯路。"""
    if not G_AO_USE_HISTORY:
        return AOHistoryState(valid=False)
    return AOHistoryState(
        bent_normal_history=list(new_ao),
        valid=True,
        frame_index=frame_index,
    )


# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------



# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。



# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------



# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。




# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------



# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。



# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------



# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。

class CullingPass(IntEnum):
    NO_OCCLUSION  = 0   # 不问遮挡，只管画
    OCCLUSION_MAIN = 1  # 主遮挡剔除
    OCCLUSION_POST = 2  # 后置再检
    EXPLICIT_LIST  = 3  # 钦点名单


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class CandidateNode:
    """
    GPU 端候选节点的 CPU 镜像。
    三个 uint32，承载了整个场景图的生死。
    """
    # x: 1 culling-flag-bits | NANITE_MAX_INSTANCES_BITS
    # y: 1 | nodes-per-primitive | views-per-pass
    # z: 1 | BVH nodes per group
    x: int = 0
    y: int = 0
    z: int = 0


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class CullRasterContext:
    """
    鲁迅式：凡过得去的帧，都是相似的；每一帧出问题，各有各的原因。
    """
    config: NaniteCullRasterConfig = field(default_factory=NaniteCullRasterConfig)
    culling_pass: CullingPass = CullingPass.NO_OCCLUSION
    candidate_nodes: List[CandidateNode] = field(default_factory=list)
    visible_cluster_count: int = 0
    hw_raster_count: int = 0
    sw_raster_count: int = 0
    tessellation_patches: int = 0

    def should_use_hw_raster(self, edge_length_px: float) -> bool:
        """大三角形走硬件，小三角形走计算着色器。历史的车轮总是这样碾过。"""
        return edge_length_px >= self.config.min_pixels_per_edge_hw

    def compute_lod_error(self, world_size: float, distance: float, fov_scale: float) -> float:
        """屏幕空间误差：距离越远，细节越少，这是Nanite的基本信条，也是人生的隐喻。"""
        if distance <= 0.0:
            return float('inf')
        screen_size = world_size * fov_scale / distance
        return screen_size / self.config.max_pixels_per_edge

    def cull_instance(self, bounds_min: Tuple[float,float,float],
                      bounds_max: Tuple[float,float,float],
                      view_frustum_planes: List[Tuple[float,float,float,float]]) -> bool:
        """
        AABB vs 视锥剔除。
        鲁迅：有些东西，不是被看见了才存在；是没有被剔除，才侥幸出现在屏幕上。
        """
        cx = (bounds_min[0] + bounds_max[0]) * 0.5
        cy = (bounds_min[1] + bounds_max[1]) * 0.5
        cz = (bounds_min[2] + bounds_max[2]) * 0.5
        ex = (bounds_max[0] - bounds_min[0]) * 0.5
        ey = (bounds_max[1] - bounds_min[1]) * 0.5
        ez = (bounds_max[2] - bounds_min[2]) * 0.5
        for (nx, ny, nz, d) in view_frustum_planes:
            dist = nx*cx + ny*cy + nz*cz + d
            radius = abs(nx)*ex + abs(ny)*ey + abs(nz)*ez
            if dist + radius < 0.0:
                return True   # 完全在平面背面，剔除
        return False

    def dispatch_cull_pass(self, instances: List[Dict[str, Any]],
                           view_frustum_planes: List[Tuple[float,float,float,float]]) -> int:
        """
        模拟 GPU 剔除 dispatch。返回存活实例数。
        真正的 GPU 版本用 IndirectArgs 驱动，这里只是影子。
        """
        survived = 0
        for inst in instances:
            b_min = inst.get('bounds_min', (0,0,0))
            b_max = inst.get('bounds_max', (1,1,1))
            if not self.cull_instance(b_min, b_max, view_frustum_planes):
                survived += 1
        self.visible_cluster_count = survived
        return survived


# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass


@dataclass


# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass


@dataclass



# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass


@dataclass


# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass


@dataclass
class HitProxyEntry:
    hit_proxy_id: int = 0
    primitive_index: int = 0
    cluster_index: int = 0


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class EditorSelectionContext:
    """
    对应 UE5 NaniteEditor：EmitEditorSelectionDepth / DrawHitProxies。
    用 VisBuffer 反查 HitProxy，再高亮轮廓。
    这套流程，在编辑器里每帧都跑；在发布版里，它不存在。
    就像鲁迅的杂文——只在特定时代才有意义。
    """
    config: NaniteEditorConfig = field(default_factory=NaniteEditorConfig)
    selected_hit_proxy_ids: List[int] = field(default_factory=list)
    hit_proxy_table: Dict[int, HitProxyEntry] = field(default_factory=dict)
    vis_buffer: Optional[Any] = None   # uint64 texture handle（CPU侧占位）

    def register_hit_proxy(self, primitive_index: int, cluster_index: int) -> int:
        proxy_id = len(self.hit_proxy_table)
        self.hit_proxy_table[proxy_id] = HitProxyEntry(
            hit_proxy_id=proxy_id,
            primitive_index=primitive_index,
            cluster_index=cluster_index,
        )
        return proxy_id

    def emit_selection_depth(self, only_selected: bool = True) -> List[int]:
        """
        FEmitEditorSelectionDepthPS 的 Python 影子。
        only_selected=True 对应 ONLY_SELECTED shader permutation。
        返回命中的 primitive_index 列表。
        """
        if only_selected:
            return [e.primitive_index
                    for pid, e in self.hit_proxy_table.items()
                    if pid in self.selected_hit_proxy_ids]
        return [e.primitive_index for e in self.hit_proxy_table.values()]

    def emit_hit_proxy_ids(self, vis_buffer_sample: Optional[int] = None) -> Optional[int]:
        """从 VisBuffer 采样反查 HitProxyId。vis_buffer_sample 模拟 GPU 读回值。"""
        if vis_buffer_sample is None:
            return None
        return self.hit_proxy_table.get(vis_buffer_sample, HitProxyEntry()).hit_proxy_id

    def emit_editor_nanite_scene_depth(self,
                                       editor_view_rect: Tuple[int,int,int,int],
                                       scene_view_rect: Tuple[int,int,int,int]) -> Tuple[float,float,float,float]:
        """
        计算 SceneTransform：editor 输出坐标系 → scene VisBuffer 坐标系。
        对应 FScreenTransform::ChangeRectFromTo。
        """
        ex0,ey0,ex1,ey1 = editor_view_rect
        sx0,sy0,sx1,sy1 = scene_view_rect
        ew = max(ex1-ex0, 1); eh = max(ey1-ey0, 1)
        sw = max(sx1-sx0, 1); sh = max(sy1-sy0, 1)
        scale_x = sw / ew
        scale_y = sh / eh
        bias_x  = sx0 - ex0 * scale_x
        bias_y  = sy0 - ey0 * scale_y
        return (scale_x, scale_y, bias_x, bias_y)


# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass


@dataclass


# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass


@dataclass



# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass


@dataclass


# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass


@dataclass
class BLASCacheEntry:
    """对应 FNaniteRayTracingASCacheEntry：BLAS 缓存槽位。"""
    byte_offset: int = 0
    byte_size: int = 0
    update_sequence_id: int = 0   # 递增，防止 GPU stale hit

    @property
    def is_allocated(self) -> bool:
        return self.byte_size > 0


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class RayTracingUpdateRequest:
    """CPU 侧模拟 GPU readback 的缓存请求结构。"""
    runtime_resource_id: int = 0
    requested_size: int = 0
    lod_error: float = 0.0

















def _cos_weighted_direction(normal: tuple, u1: float, u2: float) -> tuple:
    import math
    phi = 2.0 * math.pi * u1
    cos_t = math.sqrt(u2);  sin_t = math.sqrt(max(0.0, 1.0 - u2))
    nx, ny, nz = normal
    tx, ty, tz = (1.,0.,0.) if abs(nx) < 0.9 else (0.,1.,0.)
    bx=ny*tz-nz*ty; by=nz*tx-nx*tz; bz=nx*ty-ny*tx
    bl=max(1e-9,(bx*bx+by*by+bz*bz)**.5); bx/=bl; by/=bl; bz/=bl
    sx,sy,sz=tx,ty,tz; d=sx*nx+sy*ny+sz*nz; sx-=d*nx; sy-=d*ny; sz-=d*nz
    sl=max(1e-9,(sx*sx+sy*sy+sz*sz)**.5); sx/=sl; sy/=sl; sz/=sl
    cp=math.cos(phi); sp=math.sin(phi)
    return (sin_t*(cp*sx+sp*bx)+cos_t*nx, sin_t*(cp*sy+sp*by)+cos_t*ny, sin_t*(cp*sz+sp*bz)+cos_t*nz)

















def _hzb_screen_trace(ray: ReflectionRay, depth_pyramid: list, cfg: ReflectionTraceConfig) -> tuple:
    """
    Hierarchical Z-Buffer screen trace。
    迭代停止有两种情况：找到交点，或者耗尽步数——生活里的困境也不外乎此。
    """
    if not depth_pyramid: return False, (0.,0.), 0.
    ox,oy,_=ray.origin; dx,dy,dz=ray.direction
    su=dx*.001; sv=dy*.001; u,v=ox%1.,oy%1.
    lv=len(depth_pyramid)-1; base=depth_pyramid[0]
    H,W=len(base),(len(base[0]) if base else 1)
    for i in range(cfg.hzb_max_iterations):
        ui=int(u*W)%W; vi=int(v*H)%H
        la=depth_pyramid[min(lv,len(depth_pyramid)-1)]
        lH=len(la); lW=len(la[0]) if la else 1
        cd=la[min(vi>>lv,lH-1)][min(ui>>lv,lW-1)]
        rd=abs(dz)*(i+1)*.01
        if rd>cd*(1.+cfg.hzb_relative_depth_threshold):
            if lv>0: lv-=1
            else:
                fd=base[vi][ui]
                if abs(rd-fd)<cfg.hzb_relative_depth_threshold*fd: return True,(u,v),fd
        else:
            u+=su*(1<<lv); v+=sv*(1<<lv)
            if not(0.<=u<=1. and 0.<=v<=1.): return False,(u,v),0.
            lv=min(lv+1,len(depth_pyramid)-1)
        if cfg.hzb_min_occupancy>0 and i>cfg.hzb_max_iterations//2: break
    return False,(u,v),0.




@dataclass




@dataclass





@dataclass




@dataclass
class GPUSceneAddOp:
    primitive_group_id:int; lod_level:int; world_bounds:tuple


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class GPUSceneRemoveOp:
    primitive_group_id:int

















def compute_card_metrics(view_position,surface_cache_resolution=1.,lumen_scene_detail=1.,
                         use_hwrt=False,ray_tracing_cull_radius=100000.,
                         fast_camera_mode=False,ortho_camera=False) -> LumenCardMetrics:
    """
    计算 Surface Cache 卡片的距离和密度阈值。
    参数愈多，逻辑愈繁，不过都是在问同一件事：这张卡片值不值得渲染。
    """
    md=ray_tracing_cull_radius if use_hwrt else ray_tracing_cull_radius*1.5
    td=100.*surface_cache_resolution*(.2 if fast_camera_mode else 1.)
    mr=max(1,min(1024,int(round(((1 if ortho_camera else 4)/lumen_scene_detail)*surface_cache_resolution))))
    return LumenCardMetrics(max_distance=md,texel_density=td,min_resolution=mr)

















def diff_primitive_groups(prev:set,curr:set) -> tuple:
    """只上传变化量，不重传全世界。"""
    a,r=curr-prev,prev-curr
    return ([GPUSceneAddOp(g,0,((0.,0.,0.),(1.,1.,1.))) for g in sorted(a)],
            [GPUSceneRemoveOp(g) for g in sorted(r)])


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class DirectLightingHWRTConfig:
    """
    直接光照 HWRT 配置。
    每个字段背后，都有一位工程师在 profiler 前皱眉的记忆。
    """
    enabled:bool=True; async_compute:bool=True; force_two_sided:bool=False
    end_bias:float=1.; far_field:bool=True; heightfield_projection_bias:bool=False
    hf_projection_bias_search_radius:float=256.


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class VisualizeConfig:
    mode:int=0; grid_pixel_size:int=32; trace_mesh_sdfs:bool=True
    hi_res_surface:bool=True; cone_angle_deg:float=0.; cone_step_factor:float=2.
    min_trace_distance:float=0.; max_trace_distance:float=100000.
    tone_map:bool=True; culling_mode:int=0




@dataclass




@dataclass





@dataclass




@dataclass
class VisualizeTile:
    tile_x:int; tile_y:int; pixel_count:int=64


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class VisualizeRay:
    origin:tuple; direction:tuple; tile_idx:int

















class CompactMode:
    HIT_LIGHTING_RETRACE=0; FORCE_HIT_LIGHTING=1

















def create_visualize_tiles(W:int,H:int,ts:int=8) -> list:
    """
    把屏幕切成 tile 格，每个 tile 是一个独立的 GPU 工作单元。
    划分的逻辑很简单，简单到令人安心。
    """
    return [VisualizeTile(tx,ty,min(ts,W-tx)*min(ts,H-ty)) for ty in range(0,H,ts) for tx in range(0,W,ts)]

















def create_visualize_rays(tiles,depth_buffer,normal_buffer,W,H,max_dist=100000.) -> list:
    """从每个 tile 中心发出一根可视化光线，沿法线方向。"""
    return [VisualizeRay(
        origin=(((t.tile_x+.5)/max(1,W)),((t.tile_y+.5)/max(1,H)),
                (depth_buffer((t.tile_x+.5)/max(1,W),(t.tile_y+.5)/max(1,H)) if callable(depth_buffer) else 1.)),
        direction=(normal_buffer((t.tile_x+.5)/max(1,W),(t.tile_y+.5)/max(1,H)) if callable(normal_buffer) else (0.,0.,1.)),
        tile_idx=i,
    ) for i,t in enumerate(tiles)]

















def compact_visualize_rays(rays:list, mode:int=CompactMode.HIT_LIGHTING_RETRACE) -> list:
    """
    去除已解决的光线，剩余按 tile 排序。
    紧凑，是为了快。快，是为了每帧不超时。
    """
    if mode==CompactMode.FORCE_HIT_LIGHTING: return sorted(rays,key=lambda r:r.tile_idx)
    return sorted([r for r in rays if r.origin[2]>=1.],key=lambda r:r.tile_idx)

















class HWRTScreenProbePass:
    """
    Screen Probe Gather 的硬件光线追踪通道。
    它比反射通道更诚实：不需要镜面，只需要光。
    一个探针收集到的光，最终铺满整个屏幕——以小博大，这是采样算法的信念。
    """
    D,F,H='Default','FarField','HitLighting'
    def __init__(self,bias=.1,normal_bias=.1,hair_bias=2.,far_field=True,hit_lighting=False,structured_is=False):
        self.bias=bias; self.normal_bias=normal_bias; self.hair_bias=hair_bias
        self.ff=far_field; self.hl=hit_lighting; self.sis=structured_is
    def _remap(self,pt,flags):
        o=dict(flags)
        if pt==self.F: o['ais']='Disabled'; o['rc']=False
        elif pt==self.H:
            o['ffo']=False
            if o.get('ais')=='AHS': o['ais']='Retrace'
        else: o['ffo']=False
        if pt!=self.H: o['ser']=False
        return o
    def gather(self,probes,rc,near=4000.,far=200000.) -> list:
        res=[]
        passes=[self.D]+([self.F] if self.ff else [])+([self.H] if self.hl else [])
        for p in probes:
            pid=p.get('probe_id',0); n=p.get('normal',(0.,1.,0.))
            rad=(0.,0.,0.); hd=float('inf')
            for pt in passes:
                ck=tuple(int(v*4) for v in n); cv=rc.get(ck,(0.,0.,0.))
                if pt==self.F:
                    if hd>far*.5: rad=cv; hd=far
                elif pt==self.H:
                    if hd==float('inf'): rad=tuple(v*.7 for v in cv); hd=near*.5
                else:
                    if rad==(0.,0.,0.): rad=cv; hd=near
            res.append((pid,rad,hd))
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 鲁迅曾说：世上本没有光，照的人多了，也便有了缓存。
# Port: UE5 Renderer-Private — IndirectLightingCache / LightMapRendering /
#        LightFunctionAtlas / IESTextureManager / WaterInfoTextureRendering
# ──────────────────────────────────────────────────────────────────────────────

import math, hashlib
from collections import OrderedDict

# ── 1. IndirectLightingCache ──────────────────────────────────────────────────
# 体积纹理图集，为动态物体逐对象缓存间接光照。
# 正如鲁迅笔下那些沉默的看客：数据在此聚集，却不轻易更新。

_ILC_DIMENSION   = 64          # r.Cache.LightingCacheDimension
_ILC_ALLOC_SIZE  = 5           # r.Cache.LightingCacheMovableObjectAllocationSize
_BOUND_ROUNDUP   = math.sqrt(2.0)
_LOG_ROUNDUP     = math.log(_BOUND_ROUNDUP)









def _ilc_round_bound(v: float) -> float:
    """将包围盒边长向上取整到 sqrt(2)^N，稳定分配，减少抖动。"""
    if v <= 0: return _ILC_ALLOC_SIZE
    n = math.ceil(math.log(max(v, 1e-6)) / _LOG_ROUNDUP)
    return _BOUND_ROUNDUP ** n









class IndirectLightingCache:
    """
    三张浮点体积纹理（SH系数分三通道）构成的间接光缓存。
    鲁迅式注：这世界哪有什么自发光，不过是别人替你把SH算好了存起来。
    """
    def __init__(self, dimension=_ILC_DIMENSION):
        self.dim   = dimension
        self.blocks: dict[tuple,dict] = {}   # texel_min -> block
        self._dirty: list[tuple]      = []
        self._update_all              = True

    # ── 分配 ──────────────────────────────────────────────────────────────────
    def allocate(self, bounds_size: float) -> tuple | None:
        """
        按四舍五入后的包围盒尺寸在3D图集内分配体素块。
        返回 (x,y,z) texel_min 或 None（图集已满）。
        """
        sz = max(int(math.ceil(_ilc_round_bound(bounds_size))), _ILC_ALLOC_SIZE)
        sz = min(sz, self.dim)
        for z in range(0, self.dim - sz + 1, sz):
            for y in range(0, self.dim - sz + 1, sz):
                for x in range(0, self.dim - sz + 1, sz):
                    key = (x, y, z)
                    if key not in self.blocks:
                        self.blocks[key] = {'sz': sz, 'sh': [0.0]*9, 'valid': False}
                        self._dirty.append(key)
                        return key
        return None  # 人太多，站不下了

    def update_sh(self, texel_min: tuple, sh9: list[float]):
        """写入9分量SH系数（L0+L1），标记块为有效。"""
        if texel_min not in self.blocks: return
        self.blocks[texel_min]['sh']   = sh9[:9]
        self.blocks[texel_min]['valid'] = True

    def query(self, world_pos: tuple) -> list[float]:
        """
        在体积块中查找最近的SH采样。
        鲁迅式注：寻光者众，而光源稀少——先到先得，后来者只能继承别人的SH。
        """
        # 简化：按位置哈希映射到某个已分配块
        h = hash(tuple(int(v // _ILC_ALLOC_SIZE) for v in world_pos)) % max(len(self.blocks),1)
        for i, (k, b) in enumerate(self.blocks.items()):
            if i == h % len(self.blocks) and b['valid']:
                return b['sh']
        return [0.0]*9

    def free(self, texel_min: tuple):
        self.blocks.pop(texel_min, None)

    def flush_dirty(self) -> list[tuple]:
        """返回并清空待上传的块列表。"""
        d, self._dirty = self._dirty[:], []
        return d


# ── 2. LightMapRendering ──────────────────────────────────────────────────────
# 预计算光照贴图策略：LQ（2系数）/ HQ（6系数），以及体积光照图。
# 鲁迅式注：有些光早在烘焙时便已死去，只剩一张贴图流传人世。

_LQ_COEF = 2
_HQ_COEF = 6









class LightmapQuality:
    LQ = 'LQ_TEXTURE_LIGHTMAP'
    HQ = 'HQ_TEXTURE_LIGHTMAP'









def lightmap_policy_should_compile(quality: str, is_lit: bool, supports_static: bool,
                                    static_lighting_allowed: bool) -> bool:
    """
    对应 LightMapPolicyImpl::ShouldCompilePermutation。
    只有受光材质 + 支持静态光照的顶点工厂才编译此排列。
    """
    return is_lit and supports_static and static_lighting_allowed









def get_lightmap_coef_count(quality: str) -> int:
    return _HQ_COEF if quality == LightmapQuality.HQ else _LQ_COEF









class CachedVolumeIndirectLighting:
    """
    FCachedVolumeIndirectLightingPolicy 移植。
    用3D体积纹理插值采样，给动态物体提供连续间接光。
    鲁迅式注：体积之内，光如往事——连续，却无人能说清来自何处。
    """
    requires_sm5     = True
    no_translucency  = True

    @staticmethod
    def should_use(feature_level_sm5: bool, is_translucent: bool) -> bool:
        return feature_level_sm5 and not is_translucent

    @staticmethod
    def interpolate(cache: IndirectLightingCache, pos: tuple) -> list[float]:
        return cache.query(pos)









class CachedPointIndirectLighting:
    """
    FCachedPointIndirectLightingPolicy 移植。
    单点SH缓存，不保证空间连续，依赖时间插值淡化跳变。
    鲁迅式注：单点取样，正如只见一斑，便以为认清了豹。
    """
    TRANSITION_SPEED = 800.0   # r.Cache.SampleTransitionSpeed (units/s)

    @staticmethod
    def blend(sh_from: list[float], sh_to: list[float], dt: float) -> list[float]:
        t = min(dt * CachedPointIndirectLighting.TRANSITION_SPEED / 1000.0, 1.0)
        return [a + (b-a)*t for a,b in zip(sh_from, sh_to)]


# ── 3. LightFunctionAtlas ─────────────────────────────────────────────────────
# 将每种灯光函数材质渲染为2D纹理图集的子区域，按材质ID去重。
# 鲁迅式注：每盏灯都有自己的脾气，图集不过是把这些脾气分格收纳。

_LFA_MAX_EDGE      = 16          # 最大每维槽数
_LFA_MAX_FUNCTIONS = _LFA_MAX_EDGE * _LFA_MAX_EDGE  # 256









class LightFunctionAtlasSlot:
    __slots__ = ('material_id','uv_min','uv_max','valid')
    def __init__(self, material_id: str, uv_min: tuple, uv_max: tuple):
        self.material_id = material_id
        self.uv_min      = uv_min   # (u, v) normalized
        self.uv_max      = uv_max
        self.valid       = True









class LightFunctionAtlas:
    """
    2D纹理图集，按材质唯一ID去重存储灯光函数快照。
    对应 LightFunctionAtlas::FLightFunctionAtlasManager。
    鲁迅式注：材质千变万化，图集只认ID——就像官场只认印章，不认人脸。
    """
    def __init__(self, edge_size: int = 4, slot_resolution: int = 128):
        edge_size = min(max(edge_size, 2), _LFA_MAX_EDGE)
        self.edge        = edge_size
        self.slot_res    = max(slot_resolution, 32)
        self.atlas_res   = self.edge * self.slot_res
        self._slots: OrderedDict[str, LightFunctionAtlasSlot] = OrderedDict()

    def _uv_for_index(self, idx: int) -> tuple[tuple,tuple]:
        row, col = divmod(idx, self.edge)
        inv = 1.0 / self.edge
        umin = (col * inv, row * inv)
        umax = ((col+1)*inv, (row+1)*inv)
        return umin, umax

    def register_material(self, material_id: str) -> LightFunctionAtlasSlot | None:
        """注册或复用材质槽；超出容量返回 None。"""
        if material_id in self._slots:
            return self._slots[material_id]   # 去重命中
        if len(self._slots) >= self.edge * self.edge:
            return None  # 图集已满，正如舆论满了便再无新声
        idx  = len(self._slots)
        umin, umax = self._uv_for_index(idx)
        slot = LightFunctionAtlasSlot(material_id, umin, umax)
        self._slots[material_id] = slot
        return slot

    def get_slot(self, material_id: str) -> LightFunctionAtlasSlot | None:
        return self._slots.get(material_id)

    def get_light_index(self, material_id: str) -> int:
        """返回该材质在图集中的线性索引，供Shader索引常量缓冲。"""
        for i, k in enumerate(self._slots):
            if k == material_id: return i
        return -1

    @property
    def slot_count(self) -> int: return len(self._slots)


# ── 4. IESTextureManager ──────────────────────────────────────────────────────
# Texture2DArray图集，按RefCount管理IES光域网格纹理切片。
# 鲁迅式注：每个灯的配光曲线都是一段沉默的自白，图集只管存档，不问缘由。

_IES_DEFAULT_RESOLUTION   = 256    # r.IESAtlas.Resolution
_IES_MAX_PROFILE_COUNT    = 32     # r.IESAtlas.MaxProfileCount
_IES_INVALID_SLOT         = 0xFFFF_FFFF









class IESAtlasSlot:
    __slots__ = ('texture_id','slice_index','ref_count','resolution','dirty')
    def __init__(self, texture_id: str, resolution: int):
        self.texture_id  = texture_id
        self.slice_index = _IES_INVALID_SLOT
        self.ref_count   = 1
        self.resolution  = resolution
        self.dirty       = True









class IESTextureManager:
    """
    移植自 IESAtlas::FIESTextureManager（FRenderResource 子类）。
    管理 Texture2DArray 图集的分配/引用/回收，延迟上传。
    鲁迅式注：光域网格不过是灯的脸谱，收进图集，
              挂起来供着——有人引用，便不敢释放。
    """
    def __init__(self, max_profiles: int = _IES_MAX_PROFILE_COUNT,
                 resolution: int = _IES_DEFAULT_RESOLUTION):
        self.max_profiles  = max_profiles
        self.resolution    = resolution
        self._slots: dict[str, IESAtlasSlot] = {}
        self._free_slices: list[int]          = list(range(max_profiles))
        self._pending_adds: list[str]         = []
        self._pending_refreshes: list[str]    = []

    def add_or_ref(self, texture_id: str) -> IESAtlasSlot | None:
        """增加引用；若为新纹理则分配切片，加入待上传队列。"""
        if texture_id in self._slots:
            self._slots[texture_id].ref_count += 1
            return self._slots[texture_id]
        if not self._free_slices:
            return None   # 图集已满——人间灯光太多，容不下了
        slot = IESAtlasSlot(texture_id, self.resolution)
        self._slots[texture_id] = slot
        self._pending_adds.append(texture_id)
        return slot

    def release(self, texture_id: str):
        """减引用；归零则回收切片，留待他用。"""
        slot = self._slots.get(texture_id)
        if not slot: return
        slot.ref_count -= 1
        if slot.ref_count <= 0:
            if slot.slice_index != _IES_INVALID_SLOT:
                self._free_slices.append(slot.slice_index)
                slot.slice_index = _IES_INVALID_SLOT
            del self._slots[texture_id]

    def mark_dirty(self, texture_id: str):
        """强制下帧重新上传该纹理切片，对应 bForceRefresh。"""
        if texture_id in self._slots:
            self._slots[texture_id].dirty = True
            if texture_id not in self._pending_refreshes:
                self._pending_refreshes.append(texture_id)

    def commit(self) -> dict:
        """
        将待上传列表提交给"渲染线程"（此处返回任务字典供外部执行）。
        对应 UpdateIESAtlas RDG Pass 的调度逻辑。
        """
        task = {'adds': self._pending_adds[:], 'refreshes': self._pending_refreshes[:]}
        # 为新增槽分配切片索引
        for tid in self._pending_adds:
            slot = self._slots.get(tid)
            if slot and slot.slice_index == _IES_INVALID_SLOT and self._free_slices:
                slot.slice_index = self._free_slices.pop(0)
                slot.dirty = True
        self._pending_adds.clear()
        self._pending_refreshes.clear()
        return task

    @property
    def valid_slot_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.slice_index != _IES_INVALID_SLOT)


# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。




# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。



# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。

class WaterInfoTextureDesc:
    """捕获参数描述，对应 UWaterInfoTextureRendering 的配置。"""
    __slots__ = ('extent','capture_z','water_z_min','water_z_max',
                 'ground_z_min','blur_radius','use_128bit_rt')
    def __init__(self, extent=(512,512), capture_z=0.0,
                 water_z_min=-1e4, water_z_max=1e4,
                 ground_z_min=-1e4, blur_radius=2, use_128bit_rt=False):
        self.extent        = extent
        self.capture_z     = capture_z
        self.water_z_min   = water_z_min
        self.water_z_max   = water_z_max
        self.ground_z_min  = ground_z_min
        self.blur_radius   = blur_radius
        self.use_128bit_rt = use_128bit_rt









class WaterInfoTexturePipeline:
    """
    移植自 FWaterInfoTextureRendering 的简化CPU模拟版本。
    执行顺序：water_body_pass → ground_depth_pass → merge → blur → output
    鲁迅式注：水的信息要单独渲染一遍，因为水从来不肯与别的物体共用同一套规则。
    """
    _UNDERGROUND_DILATION_OFFSET  = 64.0    # r.Water.WaterInfo.UndergroundDilationDepthOffset
    _DILATION_OVERWRITE_MIN_DIST  = 128.0   # r.Water.WaterInfo.DilationOverwriteMinimumDistance

    def __init__(self, desc: WaterInfoTextureDesc):
        self.desc = desc
        self._water_body_buffer: dict[tuple,float]  = {}   # pixel -> water_depth
        self._ground_depth_buffer: dict[tuple,float] = {}
        self._output_buffer: dict[tuple,tuple]       = {}   # pixel -> (depth, vel_u, vel_v, flag)

    def write_water_body(self, pixel: tuple, depth: float, velocity: tuple=(0.,0.)):
        """水体Pass写入水面深度与河流速度（仅允许列表中的材质写入）。"""
        self._water_body_buffer[pixel] = (depth, velocity[0], velocity[1])

    def write_ground_depth(self, pixel: tuple, depth: float):
        """地面深度Pass，用于水下膨胀的遮挡判断。"""
        self._ground_depth_buffer[pixel] = depth

    def _merge_pixel(self, px: tuple) -> tuple:
        """
        FWaterInfoTextureMergePS 逻辑的CPU移植。
        若地面深度远在水面之下（超出 dilation_offset），允许膨胀覆盖。
        """
        w = self._water_body_buffer.get(px)
        g = self._ground_depth_buffer.get(px, float('inf'))
        if w is None:
            return (0., 0., 0., 0.)
        wd, wu, wv = w
        # 地面比水面高出足够距离才遮挡膨胀
        if g - wd > self._UNDERGROUND_DILATION_OFFSET:
            return (wd, wu, wv, 1.)
        if g - wd > self._DILATION_OVERWRITE_MIN_DIST:
            return (wd * 0.5, wu, wv, 0.5)   # 部分遮挡，衰减
        return (wd, wu, wv, 1.)

    def _blur_pixel(self, px: tuple, radius: int) -> tuple:
        """FWaterInfoTextureBlurPS 的简化盒式模糊。"""
        x, y   = px
        acc    = [0.]*4; count = 0
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nb = self._output_buffer.get((x+dx, y+dy))
                if nb:
                    for i in range(4): acc[i] += nb[i]
                    count += 1
        if not count: return (0.,0.,0.,0.)
        return tuple(v/count for v in acc)

    def execute(self) -> dict[tuple,tuple]:
        """
        完整执行 merge + blur，返回最终水信息纹理（像素字典）。
        鲁迅式注：管线走完，留下的是一张静止的水面——
                  水下的秘密，只有采样者才能读懂。
        """
        all_px = set(self._water_body_buffer) | set(self._ground_depth_buffer)
        for px in all_px:
            self._output_buffer[px] = self._merge_pixel(px)
        if self.desc.blur_radius > 0:
            blurred = {}
            for px in self._output_buffer:
                blurred[px] = self._blur_pixel(px, self.desc.blur_radius)
            self._output_buffer = blurred
        return self._output_buffer

    def clear(self):
        self._water_body_buffer.clear()
        self._ground_depth_buffer.clear()
        self._output_buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] DistortionRendering → Python port
#
# Ported from commit upstream/unreal-renderer-ue5:
#   Renderer-Private/DistortionRendering.cpp
#
# 鲁迅曾言：「有谁从小康人家而坠入困顿的么，我以为在这途路中，
# 大约可以看见世人的真面目——折射扭曲的，才是真实的光路。」
# 折射渲染的本质：每一个像素背后都藏着一个偏移量，
# 用以描述透明材质对光线的欺骗程度。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FDistortionPassUniformParameters → AstroCellDistortionParams
#     DistortionParams (FVector4f)   → distortion_params (tuple)
#     SetupDistortionParams()        → setup_distortion_params()
#   CVarDisableDistortion            → ASTRO_DISTORTION_ENABLED
#   CVarRefractionBlur               → ASTRO_REFRACTION_BLUR
#   CVarRefractionBlurMaxStdDev      → ASTRO_REFRACTION_MAX_SIGMA
#   FDistortionScreenPS permutations → AstroCellDistortionRenderer passes
#   FSceneRenderer::GetRefractionQuality → get_cell_refraction_quality()
#
# 2-D SVG adaptation:
#   DistortionAccumRT  → per-cell UV-offset dict (dx_uv, dy_uv)
#   SceneColor blur    → feGaussianBlur stdDeviation on the cell background
#   RoughRefraction    → AstroCellRoughRefraction (blurred background + offset)
#   MobilePath         → omitted (always desktop path in Astro)
# ═══════════════════════════════════════════════════════════════════════════════

# Mirrors CVarDisableDistortion: True = distortion effects enabled
ASTRO_DISTORTION_ENABLED: bool = True

# Mirrors CVarRefractionBlur: enable rough (blurred) refractions
ASTRO_REFRACTION_BLUR: bool = True

# Mirrors CVarRefractionBlurMaxStandardDeviationInScreenPercent (scaled to SVG units)
ASTRO_REFRACTION_MAX_SIGMA: float = 5.0

# Mirrors CVarRefractionBlurTemporalAA: stabilise blur across epochs
ASTRO_REFRACTION_TEMPORAL_AA: bool = True

# Mirrors CVarRefractionBlurMaxExposedLuminance (firefly clamp for refraction)
ASTRO_REFRACTION_MAX_LUMINANCE: float = 10.0

















def setup_distortion_params(
    bbox: dict,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> tuple:
    """
    Python port of SetupDistortionParams(FVector4f& DistortionParams, FViewInfo&).

    Computes the four distortion parameters used by the DistortionScreenPS shader:
        X = ViewToClip.M[0][0]  → focal_length_x (aspect-corrected FOV)
        Y = AspectRatio (W/H)
        Z = ViewRect.Width()
        W = ViewRect.Height()

    In the 2-D SVG context, the «view» is the cell's own bounding rect; the
    «clip matrix» entry is approximated from the cell's width/height ratio,
    mirroring the perspective projection used in the C++ view matrices.

    鲁迅式：参数是透镜的规格说明书——没有它们，折射只是随机的扭曲。
    """
    w = float(bbox.get("w", 100))
    h = float(bbox.get("h", 50))
    ratio = w / max(h, 1.0)
    # ViewToClip.M[0][0] ≈ 2 * focal / w;  use a nominal FOV of 90°
    focal_x = 1.0 / max(ratio, 1e-6)   # tan(45°) = 1.0; M[0][0] = focal/tan
    return (focal_x, ratio, w, h)

















def get_cell_refraction_quality(species: str) -> int:
    """
    Mirrors FSceneRenderer::GetRefractionQuality() — returns quality level [0,3].

    Species with transparent / lens-like visual semantics get higher quality;
    opaque species get quality 0 (no refraction).

    鲁迅式：质量等级是承诺的量化——承诺越高，代价越大，
    所以只对值得的物种许下高质量的承诺。
    """
    _REFRACTION_QUALITY = {
        "cil-eye":    3,   # clear lens — full rough refraction + TAA
        "cil-layers": 2,   # semi-transparent depth stack
        "cil-loop":   1,   # slight translucent arc
        "cil-filter": 1,   # semi-transparent grid
    }
    return _REFRACTION_QUALITY.get(species, 0)

















class AstroCellDistortionParams:
    """
    Python equivalent of FDistortionPassUniformParameters.

    Stores per-cell distortion setup: UV offset magnitudes, refraction blur
    sigma, and the packed DistortionParams vector.  Constructed once per proc()
    call for species with refraction quality > 0.

    鲁迅式：折射参数是透镜的档案——记录它如何弯曲光线，
    以便渲染器在合适的时机，用合适的力度，施加正确的扭曲。
    """

    __slots__ = (
        "distortion_params", "blur_sigma", "use_rough_refraction",
        "temporal_aa", "max_luminance", "quality",
    )

    def __init__(self,
                 bbox: dict,
                 species: str,
                 viewport_w: float = 1200.0,
                 viewport_h: float = 900.0) -> None:
        self.quality            = get_cell_refraction_quality(species)
        self.distortion_params  = setup_distortion_params(bbox, viewport_w, viewport_h)
        self.use_rough_refraction = ASTRO_REFRACTION_BLUR and self.quality >= 2
        self.temporal_aa        = ASTRO_REFRACTION_TEMPORAL_AA
        self.max_luminance      = ASTRO_REFRACTION_MAX_LUMINANCE
        # Blur sigma proportional to quality level and cell size
        cell_radius = min(float(bbox.get("w", 100)), float(bbox.get("h", 50))) * 0.5
        self.blur_sigma = min(
            ASTRO_REFRACTION_MAX_SIGMA,
            cell_radius * 0.04 * self.quality,
        )

















class AstroCellRoughRefraction:
    """
    Python equivalent of the rough-refraction (blurred background) pass in
    DistortionRendering.cpp.

    The C++ implementation renders the scene behind the distorting surface
    into a blurred scratch texture (SceneColorScratchTexture) driven by
    CVarRefractionBlur.  In the SVG substrate we emit a feGaussianBlur
    SVG filter element that blurs the «background» (cells at lower z-layers
    that are occluded by the refracting cell) by blur_sigma pixels.

    Additionally, a UV-offset (dx_uv, dy_uv) is computed from the cell's
    DistortionParams.X (focal_length_x) and the incoming screen position:
        dx_uv = focal_length_x * 0.01 * cell_w
        dy_uv = ratio          * 0.01 * cell_h

    These feed into the SVG feDisplacementMap filter to warp the background —
    a pixel-exact analogue of the C++ DistortionAccumulation UV offset.

    鲁迅式：粗糙折射是现实主义的让步——不是每一面透镜都那么完美，
    模糊才是大多数透明材质的真实写照。
    """

    def __init__(self, params: AstroCellDistortionParams, bbox: dict) -> None:
        self._params = params
        focal_x, ratio, w, h = params.distortion_params
        # UV offset magnitudes (mirrors DistortionAccumRT channel layout)
        self.dx_uv = focal_x * 0.01 * w
        self.dy_uv = ratio   * 0.01 * h
        self.bbox  = bbox

    def emit_svg_filter(self, cell_id: str) -> str:
        """
        Emit an SVG <filter> element implementing rough refraction.

        Mirrors the C++ DistortionScreen pass output:
          - feGaussianBlur (rough refraction blur)
          - feDisplacementMap (UV offset distortion)
          - Composite back onto the cell

        鲁迅式：SVG 滤镜是穷人的光学实验室——
        没有光线追踪，没有折射方程，只有数学的近似和美观的谎言。
        """
        p = self._params
        if not ASTRO_DISTORTION_ENABLED or p.quality == 0:
            return ""

        blur_std = round(p.blur_sigma, 2)
        dx       = round(self.dx_uv, 2)
        dy       = round(self.dy_uv, 2)

        parts = [
            f'<!-- [ASTRO-DISTORTION] DistortionRendering.cpp port '
            f'quality={p.quality} blur_sigma={blur_std} '
            f'dx_uv={dx} dy_uv={dy} rough_refraction={p.use_rough_refraction} -->',
            f'<filter id="distortion-{cell_id}" '
            f'x="-10%" y="-10%" width="120%" height="120%">',
        ]

        if p.use_rough_refraction:
            # feGaussianBlur: mirrors SceneColorScratchTexture blur
            parts.append(
                f'  <feGaussianBlur in="SourceGraphic" '
                f'stdDeviation="{blur_std}" result="blurred"/>'
            )
            # feDisplacementMap: mirrors DistortionAccumRT UV offset
            parts.append(
                f'  <feDisplacementMap in="blurred" in2="SourceGraphic" '
                f'scale="{max(dx, dy):.2f}" '
                f'xChannelSelector="R" yChannelSelector="G" result="displaced"/>'
            )
            # feComposite: mirrors the composite blend back to scene color
            parts.append(
                f'  <feComposite in="displaced" in2="SourceGraphic" '
                f'operator="over" result="refracted"/>'
            )
            # feBlend: merge refracted layer with original (luminance clamp baked in)
            alpha = round(min(1.0, p.quality * 0.3), 2)
            parts.append(
                f'  <feBlend in="refracted" in2="SourceGraphic" '
                f'mode="normal" result="final"/>'
            )
        else:
            # Quality 1: simple UV-only displacement (no blur)
            parts.append(
                f'  <feDisplacementMap in="SourceGraphic" in2="SourceGraphic" '
                f'scale="{max(dx, dy) * 0.5:.2f}" '
                f'xChannelSelector="R" yChannelSelector="G" result="final"/>'
            )

        parts.append('</filter>')
        return "\n".join(parts)

















def apply_cell_distortion(
    cell_id: str,
    species: str,
    bbox:    dict,
    svg_content: str,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> str:
    """
    Top-level distortion application — mirrors RenderDistortion() dispatch.

    Called from proc() after shadow + AO parameters are computed, before the
    final SVG <g> wrapper is assembled.  Injects the distortion <filter> def
    and adds a filter reference attribute to the cell's SVG group.

    Returns (potentially modified) svg_content with distortion filter injected.

    鲁迅式：折射是最后的化妆——在一切颜色和阴影确定之后，
    折射悄悄地扭曲了边缘，让透明的物体看起来不那么透明，
    却也因此更真实。
    """
    if not ASTRO_DISTORTION_ENABLED:
        return svg_content

    params = AstroCellDistortionParams(bbox, species, viewport_w, viewport_h)
    if params.quality == 0:
        return svg_content

    refraction = AstroCellRoughRefraction(params, bbox)
    filter_def = refraction.emit_svg_filter(cell_id)

    if filter_def:
        return filter_def + "\n" + svg_content

    return svg_content


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] DistanceFieldShadowing → Python port
#
# Ported from commit upstream/unreal-renderer-ue5:
#   Renderer-Private/DistanceFieldShadowing.cpp
#
# 鲁迅曾言：「真正的距离，不在于远近，在于有没有阻隔的东西。」
# 距离场阴影的核心：每一个遮挡物都向外扩散一个「势力范围」（距离场），
# 阴影射线在这个场中采样，累积遮挡。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GDistanceFieldShadowing           → ASTRO_DF_SHADOW_ENABLED
#   GDFShadowQuality (0-3)            → _DF_SHADOW_QUALITY
#   GFullResolutionDFShadowing        → _DF_FULL_RESOLUTION
#   GShadowCullTileWorldSize          → _DF_CULL_TILE_SIZE
#   GDFShadowTwoSidedMeshDistanceBias → _DF_TWO_SIDED_BIAS
#   FCullObjectsForShadowCS           → AstroCellDFShadowCuller
#   GetBufferSizeForDFShadows()       → get_df_shadow_buffer_size()
#   GetDFShadowDownsampleFactor()     → get_df_shadow_downsample()
# ═══════════════════════════════════════════════════════════════════════════════

ASTRO_DF_SHADOW_ENABLED: bool = True

# Shadow quality: 0=off 1=low(20 steps) 2=medium(32 steps) 3=high(64 steps)
_DF_SHADOW_QUALITY: int = 3

# Full resolution (1) vs half resolution (0)
_DF_FULL_RESOLUTION: int = 0

# World-space tile size for scatter culling (GDFShadowCullTileWorldSize)
_DF_CULL_TILE_SIZE: float = 200.0

# Distance bias scale for two-sided meshes (GDFShadowTwoSidedMeshDistanceBiasScale)
_DF_TWO_SIDED_BIAS: float = 1.0

# Step counts per quality level
_DF_QUALITY_STEPS = {0: 0, 1: 20, 2: 32, 3: 64}








@dataclass









@dataclass
class AstroCellPrimitiveFlags:
    """
    Python equivalent of FPrimitiveFlagsCompact.

    鲁迅式：旗帜是立场的声明——每个比特都是一个「是」或「否」，
    汇聚成一个整数，代表这个单元格在场景中的身份与权利。
    """
    cast_dynamic_shadow: bool = True
    static_lighting:     bool = False
    cast_static_shadow:  bool = False
    is_nanite:           bool = True
    always_visible:      bool = False
    gpu_scene_supported: bool = True
    force_hidden:        bool = False

    @classmethod
    def from_gene_traits(cls, gene_traits: dict) -> "AstroCellPrimitiveFlags":
        return cls(
            cast_dynamic_shadow = gene_traits.get("cast_shadow", True),
            static_lighting     = gene_traits.get("static_lighting", False),
            cast_static_shadow  = gene_traits.get("cast_static_shadow", False),
            is_nanite           = True,
            always_visible      = gene_traits.get("always_visible", False),
            gpu_scene_supported = True,
            force_hidden        = gene_traits.get("force_hidden", False),
        )

    def packed(self) -> int:
        return (
            (int(self.cast_dynamic_shadow) << 0) |
            (int(self.static_lighting)     << 1) |
            (int(self.cast_static_shadow)  << 2) |
            (int(self.is_nanite)           << 3) |
            (int(self.always_visible)      << 4) |
            (int(self.gpu_scene_supported) << 5) |
            (int(self.force_hidden)        << 6)
        )


@dataclass


@dataclass


@dataclass


@dataclass



@dataclass


@dataclass


@dataclass


@dataclass
class AstroCellSceneInfoCompact:
    """
    Python equivalent of FPrimitiveSceneInfoCompact.

    鲁迅式：紧凑表示是对效率的致敬——宁可冗余地存储两份，
    也不要每次都解引用到完整对象。
    """
    cell_id:       str
    flags:         AstroCellPrimitiveFlags
    bounds_min:    tuple
    bounds_max:    tuple
    min_draw_dist: float = 0.0
    max_draw_dist: float = float("inf")

    def in_draw_range(self, screen_fraction: float) -> bool:
        return (self.min_draw_dist <= screen_fraction <= self.max_draw_dist
                or self.flags.always_visible)

    def is_visible_to_shadow(self) -> bool:
        return ((self.flags.cast_dynamic_shadow or self.flags.cast_static_shadow)
                and not self.flags.force_hidden)

















class AstroCellStaticDrawCommandCache:
    """
    Static draw command cache — mirrors FBatchingSPDI::DrawMesh() +
    GMeshDrawCommandsCacheMultithreaded path in PrimitiveSceneInfo.cpp.

    鲁迅式：缓存是对重复劳动的反抗——把已经做过的事情记录下来，
    下次遇到同样的情况，直接查账本，不必重新劳作。
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self.batch_size: int = _MDC_BATCH_SIZE

    def get(self, cell_id: str, epoch: int) -> str | None:
        return self._cache.get((cell_id, epoch))

    def put(self, cell_id: str, epoch: int, svg_fragment: str) -> None:
        if _MDC_CACHE_MT:
            self._cache[(cell_id, epoch)] = svg_fragment

    def invalidate_cell(self, cell_id: str) -> None:
        stale = [k for k in self._cache if k[0] == cell_id]
        for k in stale:
            del self._cache[k]

    def flush_epoch(self, current_epoch: int, keep_window: int = 2) -> int:
        stale = [k for k in self._cache if current_epoch - k[1] > keep_window]
        for k in stale:
            del self._cache[k]
        return len(stale)

    def stats(self) -> dict:
        return {"cache_entries": len(self._cache)}

















class AstroCellPrimitiveRegistry:
    """
    Per-epoch primitive registry — Python equivalent of FScene::Primitives TArray.

    add_primitive():    mirrors AddPrimitiveSceneInfo_RenderThread
    remove_primitive(): mirrors RemovePrimitiveSceneInfo_RenderThread
    update_transform(): mirrors UpdatePrimitiveTransform_RenderThread

    鲁迅式：注册表是场景图的公民名册——
    只有登记在册的 Primitive，才有资格被渲染、被遮挡、被反射。
    """

    def __init__(self) -> None:
        self._compact: list = []
        self._index:   dict = {}
        self.draw_cache: AstroCellStaticDrawCommandCache = \
            AstroCellStaticDrawCommandCache()

    def add_primitive(
        self,
        cell_id:     str,
        bbox:        dict,
        gene_traits: dict,
        epoch:       int,
    ) -> AstroCellSceneInfoCompact:
        flags = AstroCellPrimitiveFlags.from_gene_traits(gene_traits)
        mn = (float(bbox["x"]), float(bbox["y"]), float(bbox.get("z", 0)))
        mx = (mn[0] + float(bbox["w"]), mn[1] + float(bbox["h"]), mn[2])
        info = AstroCellSceneInfoCompact(
            cell_id=cell_id, flags=flags, bounds_min=mn, bounds_max=mx,
        )
        if cell_id in self._index:
            idx = self._index[cell_id]
            self._compact[idx] = info
            self.draw_cache.invalidate_cell(cell_id)
        else:
            self._index[cell_id] = len(self._compact)
            self._compact.append(info)
        print(
            f"[ASTRO-PSI] AddPrimitive cell_id={cell_id} "
            f"flags=0x{info.flags.packed():02X}",
            file=sys.stderr,
        )
        return info

    def remove_primitive(self, cell_id: str) -> None:
        """
        Swap-remove — mirrors RemovePrimitiveSceneInfo_RenderThread.

        鲁迅式：离场是另一种消亡——不是死亡，只是从登记册上被划去。
        """
        if cell_id not in self._index:
            return
        idx  = self._index.pop(cell_id)
        last = self._compact[-1]
        self._compact[idx] = last
        self._index[last.cell_id] = idx
        self._compact.pop()
        self.draw_cache.invalidate_cell(cell_id)

    def update_transform(self, cell_id: str, new_bbox: dict) -> None:
        """
        Update bounds — mirrors UpdatePrimitiveTransform_RenderThread.

        鲁迅式：变换更新是对不变性假设的抗议——当一个单元格移动了，
        它过去在缓存中的影像便是谎言，必须清除。
        """
        if cell_id not in self._index:
            return
        idx  = self._index[cell_id]
        info = self._compact[idx]
        mn   = (float(new_bbox["x"]), float(new_bbox["y"]),
                float(new_bbox.get("z", 0)))
        mx   = (mn[0] + float(new_bbox["w"]), mn[1] + float(new_bbox["h"]), mn[2])
        self._compact[idx] = AstroCellSceneInfoCompact(
            cell_id=cell_id, flags=info.flags,
            bounds_min=mn, bounds_max=mx,
            min_draw_dist=info.min_draw_dist,
            max_draw_dist=info.max_draw_dist,
        )
        self.draw_cache.invalidate_cell(cell_id)

    def get_shadow_casters(self) -> list:
        """Return compact infos that can cast shadows."""
        return [i for i in self._compact if i.is_visible_to_shadow()]

    def __len__(self) -> int:
        return len(self._compact)


_ASTRO_PRIMITIVE_REGISTRY_V2: AstroCellPrimitiveRegistry = AstroCellPrimitiveRegistry()

















def get_primitive_registry() -> AstroCellPrimitiveRegistry:
    """Return the process-level AstroCellPrimitiveRegistry singleton."""
    return _ASTRO_PRIMITIVE_REGISTRY_V2


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] RendererScene (Scene.cpp) → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/RendererScene.cpp
#
# 鲁迅曰：「从来如此，便对么？」
# 场景图从来如此地把所有 Primitive 塞进一个 TArray；
# 我们对么？我们是——因为 json + dict 在这个规模下绰绰有余。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarEarlyZPass / CVarBasePassWriteDepthEvenWithFullPrepass
#       → _EARLY_Z_PASS_V2 / _BASE_PASS_WRITE_DEPTH_V2
#   CVarEarlyZPassOnlyMaterialMasking → _EARLY_Z_ONLY_MASKING_V2
#   GVisibilitySkipAlwaysVisible      → _VIS_SKIP_ALWAYS_VISIBLE_V2
#   CVarVisibilityLocalLightPrimitiveInteraction → _VIS_LOCAL_LIGHT_V2
#   GSafeCullDistanceUpdate           → _SAFE_CULL_DIST_V2
#
# AstroCellSceneGraph: owns primitive_registry, BVH, shadow renderer, probes.
# ═══════════════════════════════════════════════════════════════════════════════

_EARLY_Z_PASS_V2:            int  = 3
_BASE_PASS_WRITE_DEPTH_V2:   bool = False
_EARLY_Z_ONLY_MASKING_V2:    bool = True
_VIS_SKIP_ALWAYS_VISIBLE_V2: bool = True
_VIS_LOCAL_LIGHT_V2:         int  = 2
_SAFE_CULL_DIST_V2:          bool = True

















class AstroCellSceneGraph:
    """
    Python equivalent of FScene.

    Owns and coordinates:
        primitive_registry  → AstroCellPrimitiveRegistry
        bvh                 → AstroCellBVH
        shadow_renderer     → AstroCellShadowRenderer
        refl_manager        → AstroCellReflectionCaptureManager

    begin_epoch(): SafeCullDistance → BVH rebuild → EarlyZ vis → shadow map
    project_shadows(): run shadow renderer
    end_epoch(): flush draw-command cache, return diagnostics

    鲁迅式：场景图是世界模型的底层——所有子系统都向它汇报，
    所有渲染决策都以它为依据。它不直接画任何东西，
    只是确保每一个可能被画的东西都被正确地登记在册。
    """

    def __init__(self) -> None:
        self.primitive_registry = get_primitive_registry()
        self.bvh                = AstroCellBVH()
        self.shadow_renderer    = get_shadow_renderer()
        self.refl_manager       = get_reflection_capture_manager()
        self._vis_query: AstroCellVisibilityQuery | None = None
        self._epoch: int = 0

    def begin_epoch(
        self,
        all_bboxes: dict,
        viewport_w: float = 1200.0,
        viewport_h: float = 900.0,
        scroll_x:   float = 0.0,
        scroll_y:   float = 0.0,
    ) -> AstroCellVisibilityQuery:
        """
        Per-epoch setup: SafeCullDist → BVH → EarlyZ → shadow depth map.

        鲁迅式：每个 epoch 开始之前，先做一次全场普查——
        谁在视野里，谁投射阴影，谁已经离开场景。普查是昂贵的，但不可省略。
        """
        self._epoch += 1
        if _SAFE_CULL_DIST_V2:
            vp_area = max(viewport_w * viewport_h, 1.0)
            for cell_id, bb in all_bboxes.items():
                if cell_id in self.primitive_registry._index:
                    cell_area = bb.get("w", 80) * bb.get("h", 50)
                    max_frac  = min(1.0, cell_area / vp_area)
                    idx       = self.primitive_registry._index[cell_id]
                    self.primitive_registry._compact[idx].min_draw_dist = 0.0
                    self.primitive_registry._compact[idx].max_draw_dist = (
                        max_frac * 100.0 + 0.001)
        self.bvh.build_from_registry(all_bboxes)
        query = perform_nanite_visibility(viewport_w, viewport_h, scroll_x, scroll_y)
        self._vis_query = query
        self.shadow_renderer.build_shadow_depth_map(all_bboxes)
        print(
            f"[ASTRO-SCENE] begin_epoch={self._epoch} "
            f"prims={len(self.primitive_registry)} "
            f"bvh_root={'yes' if self.bvh._root is not None else 'no'}",
            file=sys.stderr,
        )
        return query

    def project_shadows(self, all_bboxes: dict) -> None:
        """Run shadow projection for current epoch."""
        self.shadow_renderer.project_shadows(all_bboxes)

    def end_epoch(self, current_epoch: int) -> dict:
        """Flush stale draw-command cache; return diagnostics."""
        evicted = self.primitive_registry.draw_cache.flush_epoch(current_epoch)
        return {
            "epoch":           current_epoch,
            "primitive_count": len(self.primitive_registry),
            "cache_evicted":   evicted,
            "shadow_factors":  len(self.shadow_renderer._shadow_factors),
        }

    def query_overlapping(self, bbox: dict) -> list:
        return self.bvh.query_overlapping_cells(bbox)


_ASTRO_SCENE_GRAPH_V2: AstroCellSceneGraph | None = None

















def get_scene_graph() -> AstroCellSceneGraph:
    """Return the process-level AstroCellSceneGraph singleton."""
    global _ASTRO_SCENE_GRAPH_V2
    if _ASTRO_SCENE_GRAPH_V2 is None:
        _ASTRO_SCENE_GRAPH_V2 = AstroCellSceneGraph()
    return _ASTRO_SCENE_GRAPH_V2


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] Renderer.cpp module init → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/Renderer.cpp
#
# 鲁迅曰：「希望是本无所谓有，无所谓无的。」
# StartupModule 初始化全局资源，ShutdownModule 释放它们。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FRendererModule::StartupModule()  → AstroCellRendererModule.startup()
#   FRendererModule::ShutdownModule() → AstroCellRendererModule.shutdown()
#   bFlushRenderTargetsOnWorldCleanup → _FLUSH_RT_ON_CLEANUP_V2
#   bBindTileMeshDrawingDummyRenderTarget → _BIND_TILE_DUMMY_RT_V2
#   GIdentityPrimitiveUniformBuffer.InitContents()
#       → _IDENTITY_PRIMITIVE_UB_V2 sentinel dict
#   FRendererStateStreamManager (WITH_STATE_STREAM)
#       → AstroCellStateStreamRenderer (counter dict, debug lines only)
# ═══════════════════════════════════════════════════════════════════════════════

_FLUSH_RT_ON_CLEANUP_V2: bool = True
_BIND_TILE_DUMMY_RT_V2:  bool = False

_IDENTITY_PRIMITIVE_UB_V2: dict = {
    "LocalToWorld":   [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "WorldToLocal":   [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "ObjectWorldPos": [0.0, 0.0, 0.0, 1.0],
    "ObjectRadius":   1.0,
}
_DISTANCE_CULL_UB_V2:   dict = {"FadeAlpha": 1.0, "InvFadeRange": 0.0}
_DITHER_FADE_UB_V2:     dict = {"DitherFade": 1.0}

















class AstroCellStateStreamRenderer:
    """
    Python equivalent of FRendererModule::FRendererStateStreamManager.
    Tracks proxy type + count for debug overlay (WITH_STATE_STREAM path).

    鲁迅式：调试渲染器是内省的工具——
    它把系统内部的数字变成人可以读懂的文字，让隐藏的复杂性浮出水面。
    """

    def __init__(self) -> None:
        self._proxy_counts: dict = {}

    def register_proxy(self, proxy_type: str, count: int = 1) -> None:
        self._proxy_counts[proxy_type] = self._proxy_counts.get(proxy_type, 0) + count

    def deregister_proxy(self, proxy_type: str, count: int = 1) -> None:
        self._proxy_counts[proxy_type] = max(
            0, self._proxy_counts.get(proxy_type, 0) - count)

    def debug_render_lines(self) -> list:
        lines = [f"Num Render proxies = {len(self._proxy_counts)}"]
        for ptype, cnt in sorted(self._proxy_counts.items()):
            lines.append(f"[{ptype} | {cnt}]")
        return lines

    def stats(self) -> dict:
        return dict(self._proxy_counts)

















class AstroCellRendererModule:
    """
    Python equivalent of FRendererModule.

    startup()  → mirrors StartupModule() — init denoiser, perf counters, PT state
    shutdown() → mirrors ShutdownModule() — release FB manager, PSO table
    draw_tile_mesh() → mirrors DrawTileMesh() — canvas tile render helper

    鲁迅式：模块的启动与关闭是开幕与闭幕——
    中间的一切都依赖于启动时建立的全局状态，
    而关闭时的清理决定了下一次启动能否从干净的状态出发。
    """

    def __init__(self) -> None:
        self._started = False
        self._state_stream = AstroCellStateStreamRenderer()

    def startup(self) -> None:
        """
        Init global renderer state — mirrors FRendererModule::StartupModule().

        鲁迅式：StartupModule 是系统的第一句话——说错了，后面全错。
        """
        if self._started:
            return
        mgr = get_denoiser_manager()
        if not mgr.has_spatial_denoiser():
            mgr.register_spatial_denoiser(
                _PTD_DENOISER_NAME, _builtin_nne_denoiser_v2,
                needs_extra_flags=False,
            )
        reset_perf_counters()
        prepare_path_tracing(AstroCellPathTracingConfig(), view_id="default")
        self._started = True
        print(
            f"[ASTRO-RENDERER] StartupModule: denoiser='{_PTD_DENOISER_NAME}' "
            f"PSO_table={AstroCellPipelineStateId.table_size()}",
            file=sys.stderr,
        )

    def shutdown(self) -> None:
        """
        Release global renderer state — mirrors FRendererModule::ShutdownModule().

        鲁迅式：ShutdownModule 是最后的整理——
        把资源还给系统，把状态归零，不留遗憾，也不留垃圾。
        """
        if not self._started:
            return
        fb = get_feedback_manager()
        for k in fb.high_water_marks:
            fb.high_water_marks[k] = 0
        if _FLUSH_RT_ON_CLEANUP_V2:
            _pipeline_state_table.clear()
        self._started = False
        print("[ASTRO-RENDERER] ShutdownModule complete.", file=sys.stderr)

    def draw_tile_mesh(
        self,
        cell_entries: list,
        viewport_w:   float = 1200.0,
        viewport_h:   float = 900.0,
    ) -> list:
        """
        Draw cell tiles — mirrors DrawTileMesh().
        返回结构化 dict 列表，供前端 PixiJS 消费，不再拼接 SVG 字符串。

        鲁迅式：DrawTileMesh 是画布上的拼贴——
        每个单元格是一块瓦片，拼在一起才成为完整的画面。
        """
        fragments = []
        if _BIND_TILE_DUMMY_RT_V2:
            fragments.append({
                "element": "rect",
                "x": 0,
                "y": 0,
                "width": viewport_w,
                "height": viewport_h,
                "fill": "none",
                "stroke": "none",
                "opacity": 0,
                "data-role": "dummy-rt",
            })
        for entry in cell_entries:
            cid      = entry.get("cell_id", "")
            bbox     = entry.get("bbox", {"x": 0, "y": 0, "w": 80, "h": 50})
            svg_frag = entry.get("svg_fragment", "")
            tx, ty   = bbox.get("x", 0), bbox.get("y", 0)
            fragments.append({
                "element": "g",
                "data-cell-id": cid,
                "transform": {"translate": [tx, ty]},
                "children": svg_frag,
            })
        return fragments

    def state_stream_debug_lines(self) -> list:
        return self._state_stream.debug_render_lines()


_ASTRO_RENDERER_MODULE_V2: AstroCellRendererModule = AstroCellRendererModule()
_ASTRO_RENDERER_MODULE_V2.startup()

















def get_renderer_module() -> AstroCellRendererModule:
    """Return the process-level AstroCellRendererModule singleton."""
    return _ASTRO_RENDERER_MODULE_V2


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ReflectionEnvironment additional CVars → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironment.cpp
#
# 鲁迅曰：「猛兽是单独的，牛羊才成群。」
# 反射环境捕获的是周围的「群众」——粗糙表面向周围平均，光滑表面锁定最近镜像。
# roughness-based mixing 的哲学：平滑者独行，粗糙者随众。
#
# Key UE5 constructs → Astro equivalents (from head -200)
# ─────────────────────────────────────────────────────────────────────────────
#   CVarReflectionEnvironment (0/1/2) → _REFL_ENV_EN_V2
#   GReflectionEnvironmentLightmapMixing → _REFL_LM_MIXING_V2
#   GReflectionEnvironmentBeginMixingRoughness / EndMixingRoughness
#       → _REFL_MIX_BEGIN_V2 / _REFL_MIX_END_V2
#   GReflectionEnvironmentLightmapMixLargestWeight → _REFL_MIX_LW_V2
#   CVarDoTiledReflections → _TILED_REFL_V2
#   GetReflectionEnvironmentCVar()        → get_reflection_env_cvar()
#   GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight()
#       → get_roughness_mix_scale_bias()
#   IsReflectionEnvironmentAvailable()    → is_reflection_env_available()
#   IsReflectionCaptureAvailable()        → is_reflection_capture_available()
#   FCaptureComponentSceneState::ComputeCurrentFade() → compute_capture_fade()
#   FReflectionEnvironmentCubemapArray   → AstroCellCubemapArray
# ═══════════════════════════════════════════════════════════════════════════════

_REFL_ENV_EN_V2:          int   = 1
_REFL_LM_MIXING_V2:       bool  = True
_REFL_LM_MIX_BY_ROUGH_V2: bool  = True
_REFL_MIX_BEGIN_V2:       float = 0.1
_REFL_MIX_END_V2:         float = 0.3
_REFL_MIX_LW_V2:          int   = 10000
_TILED_REFL_V2:           bool  = True
_REFL_MAX_CUBEMAPS_V2:    int   = 64
_REFL_CUBEMAP_SIZE_V2:    int   = 128

















def get_roughness_mix_scale_bias() -> tuple:
    """
    (scale, bias, largest_weight) for roughness-based lightmap mixing.
    Port of GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight().

    鲁迅式：混合比例是尺度与偏置的乘积——光滑表面反射更多，粗糙表面反射更少。
    数字是公平的，它不偏袒任何表面。
    """
    if not _REFL_LM_MIXING_V2:
        return (0.0, 0.0, float(_REFL_MIX_LW_V2))
    if _REFL_MIX_END_V2 == 0.0 and _REFL_MIX_BEGIN_V2 == 0.0:
        return (0.0, 1.0, float(_REFL_MIX_LW_V2))
    if not _REFL_LM_MIX_BY_ROUGH_V2:
        return (0.0, 1.0, float(_REFL_MIX_LW_V2))
    roughness_range = max(_REFL_MIX_END_V2 - _REFL_MIX_BEGIN_V2, 0.001)
    scale = 1.0 / roughness_range
    bias  = -_REFL_MIX_BEGIN_V2 * scale
    return (scale, bias, float(_REFL_MIX_LW_V2))

















class AstroCellCubemapArray:
    """
    Python equivalent of FReflectionEnvironmentCubemapArray.

    In-memory dict mapping (capture_index, face_index) → (r,g,b) float tuples.
    Capacity: _REFL_MAX_CUBEMAPS_V2 × 6 faces.

    init():        mirrors InitRHI()
    release():     mirrors ReleaseCubeArray()
    write_face():  per-face capture blit
    read_face():   cubemap sample
    average_radiance(): pre-integrated irradiance across all 6 faces
    apply_roughness_mix(): roughness-based palette blend

    鲁迅式：立方体贴图数组是反射环境的记忆宫殿——
    六个面，每个面是一段记忆，每个探针是一间房间。
    """

    def __init__(self) -> None:
        self._data:      dict = {}
        self._next_slot: int  = 0
        self._slot_map:  dict = {}
        self.max_cubemaps: int = _REFL_MAX_CUBEMAPS_V2
        self.cubemap_size: int = _REFL_CUBEMAP_SIZE_V2
        self._initialised: bool = False

    def init(self) -> None:
        """Allocate cubemap array — mirrors InitRHI."""
        self._data.clear()
        self._next_slot = 0
        self._slot_map.clear()
        self._initialised = True

    def release(self) -> None:
        """Free all faces — mirrors ReleaseCubeArray."""
        self._data.clear()
        self._initialised = False

    def assign_slot(self, capture_id: str) -> int:
        if capture_id in self._slot_map:
            return self._slot_map[capture_id]
        slot = self._next_slot % self.max_cubemaps
        evict = [cid for cid, s in self._slot_map.items() if s == slot]
        for cid in evict:
            del self._slot_map[cid]
            for f in range(6):
                self._data.pop((slot, f), None)
        self._slot_map[capture_id] = slot
        self._next_slot += 1
        return slot

    def write_face(self, capture_id: str, face_index: int,
                   colour: tuple) -> None:
        """Write one cubemap face — mirrors per-face CaptureSceneToScratchCubemap blit."""
        slot = self.assign_slot(capture_id)
        self._data[(slot, face_index)] = colour

    def read_face(self, capture_id: str, face_index: int) -> tuple:
        """Sample one cubemap face — mirrors TextureCubeArraySample()."""
        slot = self._slot_map.get(capture_id, -1)
        if slot < 0:
            return (0.5, 0.5, 0.5)
        return self._data.get((slot, face_index), (0.5, 0.5, 0.5))

    def average_radiance(self, capture_id: str) -> tuple:
        """
        Average radiance across 6 faces — pre-integrated irradiance.

        鲁迅式：六面的平均是环境辐射度——没有哪个方向更重要，
        均值是最公正的代表，也是最保守的代表。
        """
        slot = self._slot_map.get(capture_id, -1)
        if slot < 0:
            return (0.5, 0.5, 0.5)
        faces = [self._data.get((slot, f), (0.5, 0.5, 0.5)) for f in range(6)]
        r = sum(c[0] for c in faces) / 6.0
        g = sum(c[1] for c in faces) / 6.0
        b = sum(c[2] for c in faces) / 6.0
        return (r, g, b)

    def apply_roughness_mix(
        self,
        capture_id: str,
        own_colour:  tuple,
        roughness:   float,
    ) -> tuple:
        """
        Blend own_colour toward capture average based on roughness.
        Port of the GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight
        shader code path applied per-cell.

        鲁迅式：粗糙度是态度的量度——越粗糙，越随众；越光滑，越孤立。
        """
        if not is_reflection_env_available():
            return own_colour
        scale, bias, _ = get_roughness_mix_scale_bias()
        roughness_alpha = max(0.0, min(1.0, roughness * scale + bias))
        cap_colour      = self.average_radiance(capture_id)
        cap_rgb = (cap_colour[0] * 255.0, cap_colour[1] * 255.0,
                   cap_colour[2] * 255.0)
        return _lerp_colour(own_colour, cap_rgb, roughness_alpha)


_ASTRO_CUBEMAP_ARRAY_V2: AstroCellCubemapArray = AstroCellCubemapArray()
_ASTRO_CUBEMAP_ARRAY_V2.init()

















def get_cubemap_array() -> AstroCellCubemapArray:
    """Return the process-level AstroCellCubemapArray singleton."""
    return _ASTRO_CUBEMAP_ARRAY_V2
