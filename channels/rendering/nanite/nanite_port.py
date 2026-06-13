import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def perform_nanite_visibility(
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
) -> AstroCellVisibilityQuery:
    """
    Per-epoch visibility pass.
    Mirrors PerformNaniteVisibility() — iterates all registered cells in
    cell_registry.json, tests each against the viewport, returns a query
    with per-cell and per-bin results.

    Called by loop_orchestrator before the proc() dispatch loop so that
    invisible cells can be skipped entirely (LOD = -1 → skip proc()).

    @param viewport_w  Canvas width in pixels.
    @param viewport_h  Canvas height in pixels.
    @param scroll_x    Horizontal scroll offset (panning support).
    @param scroll_y    Vertical scroll offset.
    @return            AstroCellVisibilityQuery with results.
    """
    query = AstroCellVisibilityQuery(viewport_w, viewport_h, scroll_x, scroll_y)

    registry = _load_cell_registry()
    cells = registry.get("cells", {})

    for cell_id, entry in cells.items():
        bbox_data = entry.get("bbox", {})
        # Reconstruct bbox dict from min/max format used in registry
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

        query.test_cell(cell_id, bbox, species, z_layer)

    query.finish()
    return query


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





@dataclass
class NaniteCullRasterConfig:
    """
    对应 UE5 CVar 集合：r.Nanite.* 系列。

    每一个开关背后，都是某位工程师三天三夜的痛苦与妥协。
    打开它，你获得性能；关掉它，你获得正确性。二者不可兼得。
    """
    enable_async_rasterization: bool = True
    async_rasterize_shadow_depths: bool = False
    async_rasterize_custom_pass: bool = True
    async_rasterize_lumen_mesh_cards: bool = False
    enable_compute_rasterization: bool = True
    enable_programmable_raster: bool = True
    enable_tessellation: bool = True
    filter_primitives: bool = True
    vsm_invalidate_on_lod_delta: bool = False
    raster_setup_task: bool = True
    raster_setup_cache: bool = True
    max_pixels_per_edge: float = 1.0
    min_pixels_per_edge_hw: float = 32.0
    dicing_rate: float = 2.0          # 微多边形目标尺寸（像素）
    max_patches_per_group: int = 5
    depth_buckets_min_z: float = 1000.0
    depth_buckets_max_z: float = 100_000.0
    enable_depth_bucketing: bool = True
    depth_bucket_pixel_programmable: bool = True
    fast_vis_buffer_clear: int = 1    # 0=off 1=pixel 2=tile 3=metadata


@dataclass



# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass
class NaniteEditorConfig:
    draw_lists_async_updates: bool = True


@dataclass



# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass
class NaniteRayTracingConfig:
    update_enabled: bool = True
    force_update_visible: bool = False
    lod_bias: float = 0.0
    min_cut_error: float = 0.0
    offscreen_lod_bias: float = 1.0
    offscreen_min_cut_error: float = 4.0
    use_reference_instances: bool = True
    blas_cache_enabled: bool = True
    blas_cache_relative_error_tolerance: float = 0.5
    blas_cache_size_mb: int = 64
    drive_streaming: bool = False
    max_stream_out_vertices: int = 16 * 1024 * 1024
    max_stream_out_indices: int = 64 * 1024 * 1024
    max_built_primitives_per_frame: int = 8 * 1024 * 1024
    max_staging_buffer_size_mb: int = 1024
    blas_scratch_size_multiple_mb: int = 64


@dataclass



class NaniteRayTracingContext:
    """
    对应 UE5 NaniteRayTracing：BLAS 生命周期管理。

    鲁迅式：每一个 BLAS，都是流媒体时代的孤儿。
    它在 GPU 里诞生，在预算耗尽时死去，然后被下一帧的请求重建。
    没有人为它哀悼，帧率继续。
    """
    NANITE_MAX_GPU_PAGES = 4096
    NANITE_MAX_GPU_PAGES_MASK = NANITE_MAX_GPU_PAGES - 1

    def __init__(self, config: Optional[NaniteRayTracingConfig] = None):
        self.config = config or NaniteRayTracingConfig()
        self.cache: List[BLASCacheEntry] = [
            BLASCacheEntry() for _ in range(self.NANITE_MAX_GPU_PAGES)
        ]
        self._allocator_used: int = 0
        self._allocator_max: int = self.config.blas_cache_size_mb * 1024 * 1024
        self._pending_readbacks: List[List[RayTracingUpdateRequest]] = []
        self._built_primitives_this_frame: int = 0

    def get_cache_entry(self, runtime_resource_id: int) -> BLASCacheEntry:
        return self.cache[runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK]

    def invalidate_entry(self, entry: BLASCacheEntry) -> None:
        """递增 UpdateSequenceId，使 GPU 侧旧元数据失效。"""
        entry.update_sequence_id = (entry.update_sequence_id + 1) & 0xFFFFFFFF
        if entry.update_sequence_id == 0:
            entry.update_sequence_id = 1

    def _try_allocate(self, size: int) -> int:
        """极简线性分配器（生产版用 FRangeAllocator）。返回 -1 表示失败。"""
        aligned = (size + 255) & ~255
        if self._allocator_used + aligned > self._allocator_max:
            return -1
        offset = self._allocator_used
        self._allocator_used += aligned
        return offset

    def _free(self, offset: int, size: int) -> None:
        """CPU 侧释放（简化：仅在缩容时重置整体水位）。"""
        aligned = (size + 255) & ~255
        self._allocator_used = max(0, self._allocator_used - aligned)

    def process_cache_requests(self, requests: List[RayTracingUpdateRequest]) -> None:
        """
        对应 FNaniteRayTracingBLASCache::ProcessCacheRequests。
        先做所有驱逐，再做所有分配，减少碎片。
        """
        # 驱逐阶段
        for req in requests:
            idx = req.runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK
            entry = self.cache[idx]
            if entry.is_allocated and req.requested_size != entry.byte_size:
                self._free(entry.byte_offset, entry.byte_size)
                prev_seq = entry.update_sequence_id
                self.cache[idx] = BLASCacheEntry(update_sequence_id=prev_seq)

        # 分配阶段
        for req in requests:
            if req.requested_size == 0:
                continue
            idx = req.runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK
            entry = self.cache[idx]
            if not entry.is_allocated:
                offset = self._try_allocate(req.requested_size)
                if offset >= 0:
                    entry.byte_offset = offset
                    entry.byte_size = req.requested_size
                    self.invalidate_entry(entry)

    def compute_lod_for_instance(self, world_radius: float, distance: float,
                                 is_offscreen: bool = False) -> float:
        """
        LOD 误差计算：在光线追踪场景中，离屏物体可以用更粗糙的 BLAS。
        这是妥协，也是智慧。鲁迅不会承认这是妥协。
        """
        bias = self.config.offscreen_lod_bias if is_offscreen else self.config.lod_bias
        min_err = (self.config.offscreen_min_cut_error if is_offscreen
                   else self.config.min_cut_error)
        raw_error = (world_radius / max(distance, 1e-6)) * (2 ** bias)
        return max(raw_error, min_err)

    def begin_frame(self) -> None:
        self._built_primitives_this_frame = 0

    def can_build_blas(self, triangle_count: int) -> bool:
        """预算检查：本帧还有没有余额新建 BLAS。"""
        return (self._built_primitives_this_frame + triangle_count
                <= self.config.max_built_primitives_per_frame)

    def record_blas_build(self, triangle_count: int) -> None:
        self._built_primitives_this_frame += triangle_count


# ----------------------------------------------------------
# § 4  Materials Scene Extension — 材质数据缓冲区
# ----------------------------------------------------------
# 鲁迅：材质是皮肤，几何是骨骼。
# 骨骼可以共享，皮肤必须各自承担。

@dataclass



# ----------------------------------------------------------
# § 4  Materials Scene Extension — 材质数据缓冲区
# ----------------------------------------------------------
# 鲁迅：材质是皮肤，几何是骨骼。
# 骨骼可以共享，皮肤必须各自承担。

@dataclass
class NaniteMaterialsConfig:
    material_data_min_size_bytes: int = 4 * 1024
    primitive_material_data_min_size_bytes: int = 4 * 1024
    async_updates: bool = True
    force_full_upload: int = 0     # 0=no 1=once 2=every-frame
    defrag_enabled: bool = True
    force_defrag: int = 0
    defrag_low_water_mark: float = 0.375


@dataclass



@dataclass
class NaniteMaterialEntry:
    material_id: int = 0
    hit_proxy_id: int = 0          # WITH_EDITOR only
    debug_view_data: int = 0       # WITH_DEBUG_VIEW_MODES only
    primitive_index: int = 0
    element_stride: int = 0





class NaniteMaterialsSceneExtension:
    """
    对应 UE5 FMaterialsSceneExtension。

    维护两张表：MaterialData（每材质）和 PrimitiveMaterialData（每图元）。
    GPU 侧通过 ByteAddressBuffer 随机访问。

    鲁迅式：这张表的内容，每帧都可能失效。
    失效了就 defrag，defrag 了就重传，重传了继续失效。
    这就是实时渲染的宿命。
    """
    def __init__(self, config: Optional[NaniteMaterialsConfig] = None):
        self.config = config or NaniteMaterialsConfig()
        self._material_data: Dict[int, NaniteMaterialEntry] = {}
        self._primitive_material_map: Dict[int, List[int]] = {}  # prim_idx -> [mat_ids]
        self._pending_static_mesh_updates: Dict[int, bool] = {}  # prim_idx -> valid
        self._dirty: bool = False
        self._allocated_bytes: int = 0
        self._used_bytes: int = 0

    # --- Extension lifecycle ---

    def should_be_enabled(self, nanite_supported: bool) -> bool:
        return nanite_supported

    def add_primitive(self, primitive_index: int, material_ids: List[int]) -> None:
        self._primitive_material_map[primitive_index] = material_ids
        for mid in material_ids:
            if mid not in self._material_data:
                self._material_data[mid] = NaniteMaterialEntry(
                    material_id=mid, primitive_index=primitive_index
                )
        self._dirty = True

    def remove_primitive(self, primitive_index: int) -> None:
        mids = self._primitive_material_map.pop(primitive_index, [])
        for mid in mids:
            self._material_data.pop(mid, None)
        self._pending_static_mesh_updates.pop(primitive_index, None)
        self._dirty = True

    def add_pending_static_mesh_update(self, primitive_index: int) -> None:
        """登记待更新，返回 valid-flag（用 dict bool 模拟 TSharedPtr<bool>）。"""
        self._pending_static_mesh_updates[primitive_index] = True

    def clear_pending_static_mesh_update(self, primitive_index: int) -> None:
        self._pending_static_mesh_updates.pop(primitive_index, None)

    def has_pending_static_mesh_update(self, primitive_index: int) -> bool:
        return primitive_index in self._pending_static_mesh_updates

    # --- Upload logic ---

    def should_full_upload(self) -> bool:
        return self.config.force_full_upload > 0

    def should_defrag(self) -> bool:
        if not self.config.defrag_enabled:
            return False
        if self.config.force_defrag > 0:
            return True
        if self._allocated_bytes == 0:
            return False
        ratio = self._used_bytes / self._allocated_bytes
        return ratio < self.config.defrag_low_water_mark

    def build_material_data_buffer(self) -> bytes:
        """序列化 MaterialData 为字节流（模拟 GPU ByteAddressBuffer 上传）。"""
        import struct
        entries = sorted(self._material_data.values(), key=lambda e: e.material_id)
        buf = bytearray()
        for e in entries:
            buf += struct.pack('<IIIII',
                               e.material_id, e.hit_proxy_id,
                               e.debug_view_data, e.primitive_index, e.element_stride)
        size = max(len(buf), self.config.material_data_min_size_bytes)
        self._used_bytes = len(buf)
        self._allocated_bytes = size
        return bytes(buf).ljust(size, b'\x00')

    def build_primitive_material_data_buffer(self) -> bytes:
        """序列化 PrimitiveMaterialData。"""
        import struct
        buf = bytearray()
        for prim_idx in sorted(self._primitive_material_map):
            mids = self._primitive_material_map[prim_idx]
            buf += struct.pack('<I', len(mids))
            for mid in mids:
                buf += struct.pack('<I', mid)
        size = max(len(buf), self.config.primitive_material_data_min_size_bytes)
        return bytes(buf).ljust(size, b'\x00')


# ----------------------------------------------------------
# § 5  Ownership Visibility Scene Extension
# ----------------------------------------------------------
# 鲁迅：有些东西不让你看，有些东西只让你看。
# OwnerNoSee 和 OnlyOwnerSee，是渲染器里的阶级制度。

@dataclass



# ----------------------------------------------------------
# § 5  Ownership Visibility Scene Extension
# ----------------------------------------------------------
# 鲁迅：有些东西不让你看，有些东西只让你看。
# OwnerNoSee 和 OnlyOwnerSee，是渲染器里的阶级制度。

@dataclass
class NaniteOwnershipVisibilityConfig:
    pass





class NaniteOwnershipVisibilityExtension:
    """
    对应 UE5 FOwnershipVisibilitySceneExtension。

    每帧为每个视图构建 HiddenPrimitives 位数组，上传 GPU。
    IsOwnerNoSee  → 自己看不见自己
    IsOnlyOwnerSee → 只有自己看得见

    编辑器视图绕过这一切——在编辑器里，没有秘密。
    """
    def __init__(self):
        self._nanite_primitives_with_ownership: List[int] = []  # persistent indices

    def add_primitive(self, persistent_index: int) -> None:
        self._nanite_primitives_with_ownership.append(persistent_index)

    def remove_primitive(self, persistent_index: int) -> None:
        try:
            self._nanite_primitives_with_ownership.remove(persistent_index)
        except ValueError:
            pass

    def build_hidden_primitives_bitarray(
        self,
        max_persistent_index: int,
        views: List[Dict[str, Any]],        # list of {actor_id, is_editor_view}
        primitive_infos: Dict[int, Dict],   # persistent_idx -> {owner_no_see, only_owner_see, owner_id}
    ) -> Optional[List[int]]:
        """
        构建 HiddenPrimitives 位数组（每视图 × max_persistent_index 位）。
        返回 uint32 列表（模拟 GPU StructuredBuffer<uint>）。
        鲁迅：位图是公平的，每个图元只占一位，不多也不少。
        """
        if not self._nanite_primitives_with_ownership:
            return None

        num_views = len(views)
        total_bits = max_persistent_index * num_views
        total_words = (total_bits + 31) // 32
        bit_array = [0] * total_words

        for prim_idx in self._nanite_primitives_with_ownership:
            info = primitive_infos.get(prim_idx)
            if info is None:
                continue
            is_owner_no_see  = info.get('owner_no_see', False)
            is_only_owner_see = info.get('only_owner_see', False)
            owner_id = info.get('owner_id', -1)

            for view_idx, view in enumerate(views):
                is_editor = view.get('is_editor_view', False)
                is_owned  = view.get('actor_id', None) == owner_id
                is_hidden = (not is_editor) and (
                    (is_owned and is_owner_no_see) or
                    (not is_owned and is_only_owner_see)
                )
                if is_hidden:
                    bit_pos  = view_idx * max_persistent_index + prim_idx
                    word_idx = bit_pos // 32
                    bit_idx  = bit_pos % 32
                    if word_idx < total_words:
                        bit_array[word_idx] |= (1 << bit_idx)

        return bit_array


# ----------------------------------------------------------
# § 6  TessellationTable — 微多边形细分表
# ----------------------------------------------------------
# 鲁迅：细分是一种野心。将一个三角形变成一千个，
# 只为了让边缘看起来不那么锯齿。代价是整个渲染管线的颤抖。

NANITE_TESSELLATION_TABLE_SIZE           = 15
NANITE_TESSELLATION_TABLE_PO2_SIZE       = 16   # next power of two ≥ SIZE
NANITE_TESSELLATION_TABLE_IMMEDIATE_SIZE = 3
BARYCENTRIC_MAX                          = 0x8000  # 固定小数精度


@dataclass



@dataclass
class TessellationPattern:
    tess_factors: Tuple[int, int, int] = (1, 1, 1)
    verts: List[Tuple[int, int]] = field(default_factory=list)   # (u, v) fixed-point
    indices: List[int] = field(default_factory=list)





class TessellationTable:
    """
    对应 UE5 FTessellationTable。

    生产环境从 TessellationTable.bin 加载；
    这里按需在 CPU 上生成轻量近似版。

    鲁迅：有些表只需要查，不需要理解。
    理解了，也未必能改得更好。
    """
    def __init__(self):
        self._patterns: Dict[int, TessellationPattern] = {}

    @staticmethod
    def get_pattern_key(tx: int, ty: int, tz: int) -> int:
        """
        对应 FTessellationTable::GetPattern。
        排序后线性化，保证 (3,2,1)==(1,2,3)。
        """
        factors = sorted([tx, ty, tz], reverse=True)
        f0, f1, f2 = factors
        return (
            (f0 - 1) +
            (f1 - 1) * NANITE_TESSELLATION_TABLE_PO2_SIZE +
            (f2 - 1) * NANITE_TESSELLATION_TABLE_PO2_SIZE * NANITE_TESSELLATION_TABLE_PO2_SIZE
        )

    @staticmethod
    def get_barycentrics(vert_packed: int) -> Tuple[int, int, int]:
        """
        对应 FTessellationTable::GetBarycentrics。
        低16位为 u，高16位为 v，w 由 BarycentricMax 补全。
        """
        u = vert_packed & 0xFFFF
        v = vert_packed >> 16
        w = BARYCENTRIC_MAX - u - v
        return (u, v, w)

    @staticmethod
    def pack_barycentric(u: int, v: int) -> int:
        return (u & 0xFFFF) | ((v & 0xFFFF) << 16)

    def get_or_build_pattern(self, tx: int, ty: int, tz: int) -> TessellationPattern:
        """
        懒加载细分模式。
        真正的 UE5 是预计算后二进制存储，这里在 CPU 上即时生成均匀细分。
        精度足够做功能验证，但不保证与 bin 文件完全一致。
        """
        key = self.get_pattern_key(tx, ty, tz)
        if key in self._patterns:
            return self._patterns[key]

        pattern = self._build_uniform_pattern(tx, ty, tz)
        self._patterns[key] = pattern
        return pattern

    @staticmethod
    def _build_uniform_pattern(tx: int, ty: int, tz: int) -> TessellationPattern:
        """
        在重心坐标系中均匀细分三角形。
        tx/ty/tz 分别是三条边的细分因子。
        鲁迅：均匀，是最省心的选择，也是最无趣的选择。
        """
        # 简化：使用等边细分（取三者最大值）
        n = max(tx, ty, tz)
        verts: List[Tuple[int,int]] = []
        vert_map: Dict[Tuple[int,int], int] = {}
        indices: List[int] = []

        def add_vert(i: int, j: int) -> int:
            k = (i, j)
            if k not in vert_map:
                u = round(i * BARYCENTRIC_MAX / n)
                v = round(j * BARYCENTRIC_MAX / n)
                vert_map[k] = len(verts)
                verts.append((u, v))
            return vert_map[k]

        for row in range(n):
            for col in range(n - row):
                v0 = add_vert(col,   row)
                v1 = add_vert(col+1, row)
                v2 = add_vert(col,   row+1)
                indices += [v0, v1, v2]
                if col + row < n - 1:
                    v3 = add_vert(col+1, row+1)
                    indices += [v1, v3, v2]

        return TessellationPattern(
            tess_factors=(tx, ty, tz),
            verts=list(verts),
            indices=indices,
        )

    def snap_at_edges(self, bary: List[int], tess_factors: Tuple[int,int,int]) -> List[int]:
        """
        对应 FTessellationTable::SnapAtEdges：将边界顶点对齐到整数细分位置。
        防止相邻 patch 之间出现 T-junction。
        """
        result = list(bary)
        for i in range(3):
            e0 = i
            e1 = (1 << e0) & 3
            if e1 >= 3:
                continue
            if result[e0] + result[e1] == BARYCENTRIC_MAX:
                min_idx = e0 if result[e0] < result[e1] else e1
                max_idx = e1 if min_idx == e0 else e0
                tf = tess_factors[i]
                snapped = ((result[min_idx] * tf + (BARYCENTRIC_MAX // 2) - 1)
                           // BARYCENTRIC_MAX) * BARYCENTRIC_MAX // tf
                result[max_idx] = BARYCENTRIC_MAX - snapped
                result[min_idx] = snapped
        return result


# ----------------------------------------------------------
# § 7  统一入口：NaniteRenderContext
# ----------------------------------------------------------
# 鲁迅：把所有的复杂性压缩进一个数据类，
# 然后告诉别人"这很简单"。

@dataclass



# ----------------------------------------------------------
# § 7  统一入口：NaniteRenderContext
# ----------------------------------------------------------
# 鲁迅：把所有的复杂性压缩进一个数据类，
# 然后告诉别人"这很简单"。

@dataclass
class NaniteRenderContext:
    """
    六大子系统的统一容器。
    对应 UE5 Nanite:: 命名空间下各模块的协同调用点。

    生命周期：scene_init → begin_frame → cull → shade → ray_trace → end_frame
    """
    cull_raster: CullRasterContext = field(
        default_factory=CullRasterContext)
    editor: EditorSelectionContext = field(
        default_factory=EditorSelectionContext)
    ray_tracing: NaniteRayTracingContext = field(
        default_factory=NaniteRayTracingContext)
    materials: NaniteMaterialsSceneExtension = field(
        default_factory=NaniteMaterialsSceneExtension)
    ownership_visibility: NaniteOwnershipVisibilityExtension = field(
        default_factory=NaniteOwnershipVisibilityExtension)
    tessellation_table: TessellationTable = field(
        default_factory=TessellationTable)

    # 帧计数
    frame_index: int = 0

    def begin_frame(self) -> None:
        self.ray_tracing.begin_frame()
        self.frame_index += 1

    def run_cull_pass(
        self,
        instances: List[Dict[str, Any]],
        frustum_planes: List[Tuple[float,float,float,float]],
    ) -> int:
        """
        主剔除 pass。返回可见实例数。
        鲁迅：剔除之后，剩下的才是真正的工作。
        """
        return self.cull_raster.dispatch_cull_pass(instances, frustum_planes)

    def upload_material_buffers(self) -> Tuple[bytes, bytes]:
        """返回 (material_data, primitive_material_data) 字节流，供 GPU 上传。"""
        mat_buf  = self.materials.build_material_data_buffer()
        prim_buf = self.materials.build_primitive_material_data_buffer()
        return mat_buf, prim_buf

    def get_tessellation_pattern(
        self, tx: int, ty: int, tz: int
    ) -> TessellationPattern:
        """查询细分表。三参数均需在 [1, NANITE_TESSELLATION_TABLE_SIZE] 范围内。"""
        tx = max(1, min(tx, NANITE_TESSELLATION_TABLE_SIZE))
        ty = max(1, min(ty, NANITE_TESSELLATION_TABLE_SIZE))
        tz = max(1, min(tz, NANITE_TESSELLATION_TABLE_SIZE))
        return self.tessellation_table.get_or_build_pattern(tx, ty, tz)

    def end_frame(self) -> Dict[str, int]:
        """
        帧末统计。返回诊断字典。
        没有人会细看这些数字，但它们必须存在。
        鲁迅：统计是给下一代看的。
        """
        return {
            'frame':               self.frame_index,
            'visible_clusters':    self.cull_raster.visible_cluster_count,
            'hw_raster':           self.cull_raster.hw_raster_count,
            'sw_raster':           self.cull_raster.sw_raster_count,
            'tess_patches':        self.cull_raster.tessellation_patches,
            'blas_built_tris':     self.ray_tracing._built_primitives_this_frame,
            'material_entries':    len(self.materials._material_data),
            'owned_primitives':    len(self.ownership_visibility
                                       ._nanite_primitives_with_ownership),
            'tess_patterns_cached': len(self.tessellation_table._patterns),
        }


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
