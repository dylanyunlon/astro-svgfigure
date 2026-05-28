# M100-M138 开发里程碑 — 38 位 Claude 分工计划

**Branch**: `fix/cfc0672-enhancements`
**日期**: 2026-05-28
**第一位 Claude 制定，基于 10 个大厂仓库的架构勘察**

---

## 已 clone 的参考仓库

| ID | Repo | Path | Lines (key file) | 参考焦点 |
|----|------|------|-------------------|---------|
| R1 | NVIDIA/cccl | `/home/claude/cccl` | dispatch_topk.cuh (749L) | DoubleBuffer, dispatch(), TopK radix pass |
| R2 | NVIDIA/nccl | `/home/claude/nccl` | topo.cc (1983L) | 拓扑图遍历, GPU连接发现, ring建图 |
| R3 | NVIDIA/Megatron-LM | `/home/claude/Megatron-LM` | schedules.py (2409L) | 1F1B pipeline schedule, P2P通信 |
| R4 | NVIDIA/TensorRT | `/home/claude/TensorRT` | plugin/*.cpp | Plugin注册, Builder pattern |
| R5 | google/jax | `/home/claude/jax` | dispatch.py (763L) | Trace→Lower→Compile→Execute 四阶段 |
| R6 | openai/triton | `/home/claude/triton` | compiler.py (513L) | AST→TTIR→PTX 编译链, autotuner |
| R7 | facebookresearch/sam3 | `/home/claude/sam3` | model_builder.py (1318L) | Detector+Tracker解耦, presence token |
| R8 | bytedance/monolith | `/home/claude/monolith` | native_training/ | 特征工程, Hash embedding |
| R9 | bytedance/effective_transformer | `/home/claude/effective_transformer` | attention.cu (497L) | warpReduceSum, blockReduceMax, padding removal |
| R10 | tensorflow/tensorflow | `/home/claude/tensorflow` | 全量 | XLA编译, tf.function tracing |

---

## P0 — 必须修的生产级 Bug（M100-M104）

### ✦ 第一位 Claude 完成：M100-M104

---

### M100 — 统一 Layer 数据类型（消除格式嗅探 if-else）

从 CCCL 的 `cub/cub/detail/device_double_buffer.cuh` 中 `device_double_buffer<T>` 这个好例子开始，它用一个泛型容器统一了 ping-pong buffer 的两种物理存储，让上层的 `dispatch_topk.cuh` 的 `dispatch()` 函数可以用 `m_current_buffer` 和 `m_alternate_buffer` 两个指针做零拷贝交换，完全不关心哪个物理 buffer 是"当前"的——函数签名只接受 `device_double_buffer<T>&` 一种类型。然后，遵循该模式实现一个新的 `UnifiedLayer` dataclass，让 `pipeline_orchestrator.py` 的 Stage 2→Stage 3 数据传递可以用一个统一的 `UnifiedLayer(image_b64, bbox, area, layer_id, source)` 类型替代当前 `component_extractor` 输出的 `Dict[str, Any]`（`{id, name, bbox:{x,y,w,h}}`）和 `layer_separator` 输出的 `ExtractedLayer(layer_id, image_b64, bbox: Tuple[int,int,int,int], area, centroid, original_size, layer_size)` 两种不兼容格式，并能在 `edge_refiner.refine_layers_batch()` 入口处无需任何格式判断即可直接处理。接着 `UnifiedLayer` 引入 `source: Literal["ccl", "elk_guided", "sam3", "manual"]` 字段，使下游的 `component_outliner.outline_components_batch()` 能够根据 source 选择不同的描边策略（CCL 提取的需要粗描边因为边缘粗糙，SAM3 提取的可以用细描边因为边缘精确），同时 `pipeline_orchestrator.py` 的 `_stage_layer_separate()` 优化为直接返回 `List[UnifiedLayer]` 而不是原始 dict。随后 `PipelineReport.to_dict()` 整合 `UnifiedLayer.to_dict()` 序列化方法，令前端 PostGenPanel 支持按 `source` 字段显示不同的层来源图标（CCL=齿轮, SAM3=魔法棒, ELK=拓扑），进而 `separate_layers_batch()` 增强返回值为 `List[UnifiedLayer]` 替代原有嵌套 dict。最终 `UnifiedLayer` 完善 `__eq__` 和 `__hash__` 方法，确保 `UnifiedLayer` 兼容 Python set/dict 作为去重容器的 key，全面将 pipeline 内部的层数据流升级为类型安全的单一类型以达成消除 `pipeline_orchestrator.py` 第 1103-1125 行的 `has_component_layers` format-sniffing 分支。

**实现文件**: `backend/pipeline/unified_layer.py` (新建) + 修改 `pipeline_orchestrator.py`, `component_extractor.py`, `layer_separator.py`

---

### M101 — 删除重复路由注册 + FastAPI 路由审计

从 TensorRT 的 `plugin/geluPlugin/geluPlugin.cpp` 中 `REGISTER_TENSORRT_PLUGIN(GeluPluginCreator)` 这个好例子开始，它用宏展开实现全局唯一的插件注册——每个 plugin 只能注册一次，重复注册在编译时就会报 linker error（duplicate symbol），而不是像 Python FastAPI 那样静默覆盖。然后，遵循该模式实现一个新的 `RouteRegistry` 辅助类，让 `server_layered_routes.py` 的 `register_layered_routes(app)` 可以用 `@RouteRegistry.register("/api/mastergo-export", methods=["POST"])` 装饰器替代手动 `@app.post()`，并能在注册时检测路径冲突并抛出 `DuplicateRouteError`（而不是静默覆盖）。接着 `RouteRegistry` 引入 `list_all_routes() → Dict[str, {handler, file, line}]` 内省方法，使 `server.py` 的启动日志能够打印完整的路由表（类似 Django 的 `python manage.py show_urls`），同时 `server_layered_routes.py` 删除第 335-373 行的重复定义。随后 `server.py` 整合路由审计，令 CI/CD 支持 `python -c "from backend.server_layered_routes import *; RouteRegistry.audit()"` 静态检查，进而 `register_layered_routes()` 增强为幂等调用（重复调用不报错但不重复注册）。最终 `RouteRegistry` 完善线程安全的 `_registered: Dict[str, RegistrationInfo]` 内部状态，确保 FastAPI 的 lifespan 事件和 uvicorn reload 兼容 `RouteRegistry` 的去重逻辑，全面将路由管理升级为编译时可审计的注册系统以达成消除静默路由覆盖导致的"第一个详细版本永远不执行"bug。

**实现文件**: 修改 `backend/server_layered_routes.py` (删除重复) + 新建 `backend/route_registry.py`

---

### M102 — 尺寸自适应形态学参数（消除 magic number）

从 OpenAI Triton 的 `python/triton/compiler/compiler.py` 中 `ASTSource.hash()` 这个好例子开始，它将所有影响编译结果的参数（`self.fn.cache_key`, `self.attrs`, `sorted_sig`, `constants_key`）汇聚成一个 SHA256 key，然后用这个 key 做缓存查找——关键点是 **所有参数都参与 key 计算**，不存在"某个参数被硬编码忽略"的情况。然后，遵循该模式实现一个新的 `MorphologyParams.from_image_shape(h: int, w: int) → MorphologyParams` 工厂方法，让 `component_extractor.py` 的 `_break_arrows(binary, erode=4, dilate=3)` 可以用 `params = MorphologyParams.from_image_shape(h, w); _break_arrows(binary, params.erode_iter, params.dilate_iter)` 替代硬编码的 `erode=4, dilate=3`，并能根据图片尺寸线性缩放参数（`scale = max(h, w) / 1024; erode = max(2, int(4 * scale)); dilate = max(1, int(3 * scale))`）。接着 `MorphologyParams` 引入 `min_area` 和 `min_dim` 的同比缩放，使 `extract_components(min_area=800, min_dim=15)` 能够在 2048×2048 图上自动提升到 `min_area=3200, min_dim=30`（因为所有组件面积和尺寸也按比例增大了），同时 `LayerSeparationConfig` 优化 `min_component_area` 为基于 `image_area / 10000` 的动态计算而非固定 100。随后 `pipeline_orchestrator.py` 整合 `MorphologyParams` 的图片尺寸感知逻辑，令 `config_from_request()` 支持 `auto_scale: bool = True` 请求参数让用户可以关闭自适应，进而 `_break_arrows()` 增强日志输出实际使用的参数值（`logger.info("erode=%d dilate=%d scale=%.2f for %dx%d", ...)`）。最终 `MorphologyParams` 完善 `__repr__` 和 JSON 序列化，确保 `PipelineReport.diagnostics` 兼容形态学参数的完整记录，全面将硬编码 magic number 升级为尺寸感知的自适应参数系统以达成在 512×512 到 4096×4096 全尺寸范围内稳定的层提取质量。

**实现文件**: `backend/pipeline/morphology_params.py` (新建) + 修改 `component_extractor.py`, `layer_separator.py`

---

### M103 — 绿色前景保护回退机制

从 NCCL 的 `src/graph/topo.cc` 中 `findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from)` 这个好例子开始，它在 GPU 拓扑图中递归向上遍历 PCI 树寻找 CPU 节点——关键设计是 **如果沿 NVLink 路径找不到 CPU，它不会崩溃，而是优雅地返回 `*cpu = NULL` 让调用方回退到 PCI 路径**。这就是带回退的发现模式。然后，遵循该模式实现一个新的 `GreenForegroundGuard.pre_scan(img_array: np.ndarray) → ScanResult` 前置扫描步骤，让 `green_screen_advanced.py` 的 `process_single_frame()` 可以在执行 HSV 绿幕检测之前先调用 `guard = GreenForegroundGuard(); scan = guard.pre_scan(img_array); if scan.green_foreground_risk > 0.05: strategy = "rembg"` 自动回退到 model-based 移除（rembg/BiRefNet），并能在回退时将 `scan.risk_regions` 标记为"保护区域"传给 rembg 作为 guidance mask。接着 `GreenForegroundGuard` 引入 `_estimate_background_region(img: np.ndarray) → np.ndarray` 背景区域估算（基于边缘 10% 像素的绑定直方图），使前景/背景绿色占比能够分开计算——背景中 90% 是绿色正常（就是绿幕），但前景中 5% 是绿色就有风险，同时 `batch_rembg_orchestrator.py` 优化策略选择器增加 `strategy="auto_guard"` 选项。随后 `GreenForegroundGuard` 整合 edge-aware 判定（使用 Sobel 梯度区分背景平坦绿色 vs 前景纹理绿色），令 `detect_dominant_green_hue()` 支持排除被标记的前景绿色像素，进而 `process_frames_hsv()` 增强为对每帧独立运行 guard（因为 Gemini 不同帧可能有不同的绿色前景出现率）。最终 `GreenForegroundGuard` 完善 `ScanResult.to_dict()` 序列化和前端警告展示，确保 `RembgControls.astro` 兼容显示"检测到绿色前景物体，已自动切换到 AI 移除"的用户提示，全面将盲目 HSV 色度键升级为带前景保护的智能移除以达成绿色前景物体不再被误删。

**实现文件**: `backend/pipeline/green_foreground_guard.py` (新建) + 修改 `green_screen_advanced.py`, `batch_rembg_orchestrator.py`

---

### M104 — Batch 一致性检查修正 + 跨帧层跟踪

从 Megatron-LM 的 `megatron/core/pipeline_parallel/schedules.py` 中 `forward_backward_pipelining_without_interleaving()` 这个好例子开始，它对每个 microbatch 的 forward+backward 做了精确的状态追踪——`input_tensors` 和 `output_tensors` 是两个列表，每个 microbatch 完成后 append 结果，最后通过 `average_losses_across_data_parallel_group()` 做跨 rank 的一致性聚合——关键是它用 **跨 microbatch 的平均 loss** 而不是 **方差** 来判断训练是否健康。然后，遵循该模式实现一个新的 `FrameConsistencyChecker.check(layer_counts: List[int]) → ConsistencyReport` 统计引擎，让 `layer_separator.py` 的 `separate_layers_batch()` 可以用 `report = FrameConsistencyChecker.check(layer_counts); consistent = report.cv < 0.3` 替代当前错误的 `variance > avg * 0.5` 判定（改用变异系数 CV = std/mean，阈值 0.3 表示标准差不超过均值的 30%），并能在 `report` 中包含 `outlier_frames: List[int]`（哪些帧的层数是异常值）。接着 `FrameConsistencyChecker` 引入 `cross_frame_layer_matching(frames: List[LayerSeparationResult]) → List[LayerTrack]` 跨帧层追踪（基于 centroid 距离和面积比的匈牙利匹配），使动画帧序列中同一个元素的 layer_id 能够在帧间保持稳定（帧 1 的 "Layer 3" 和帧 2 的 "Layer 3" 是同一个物体），同时 `LayerSeparationResult` 优化增加 `tracks: Optional[List[LayerTrack]]` 字段。随后 `FrameConsistencyChecker` 整合异常帧自动修复（如果帧 7 突然多了 3 个噪声层，用帧 6 和帧 8 的层 pattern 做投票剔除），令 `separate_layers_batch()` 支持 `auto_repair: bool = True` 开关，进而 `PipelineReport.diagnostics` 增强层一致性数据（CV 值、异常帧列表、追踪匹配率）。最终 `FrameConsistencyChecker` 完善 `ConsistencyReport.severity: Literal["ok", "warning", "error"]` 三级告警，确保前端 `LayerPreview.astro` 兼容显示帧一致性评分的可视化条形图，全面将简单方差判定升级为统计学正确的跨帧层一致性系统以达成 16 帧动画的层数 outlier 不再误报。

**实现文件**: `backend/pipeline/frame_consistency.py` (新建) + 修改 `layer_separator.py`

---

## P1 — SAM3 语义分割集成（M105-M112）

### ✦ 第二位 Claude 完成：M105-M107

### M105 — SAM3 Remote API Client（HuggingFace Space 代理）

从 ByteDance/effective_transformer 的 `cuda/attention.cu` 中 `warpReduceSum<T>(T val)` 这个好例子开始，它用 `__shfl_xor_sync(FINAL_MASK, val, mask, 32)` 在 warp 内做蝶形交换归约——关键设计是 **整个归约操作被封装为单一函数调用**，调用方只需 `val = warpReduceSum(val)` 一行就完成了 32 线程的并行归约，完全不需要了解 shuffle 指令的细节。然后，遵循该模式实现一个新的 `SAM3RemoteClient.segment(image_b64: str, prompt: str, confidence: float = 0.45) → SAM3Result` 远程调用封装，让 `layer_separator.py` 可以用一行 `result = await sam3_client.segment(frame_b64, "all objects")` 获得语义分割 mask 列表，并能自动处理 HuggingFace Space 的 Gradio API 协议（`gradio_client.Client("prithivMLmods/SAM3-Demo").predict()` 的连接/重试/超时/结果解码全部封装在内部）。接着 `SAM3RemoteClient` 引入 `_fallback_spaces: List[str]` 多 Space 容错列表（按优先级尝试 `prithivMLmods/SAM3-Demo` → `akhaliq/sam3` → `yolain/sam3`），使任何单个 Space 下线时能够自动切换到下一个，同时增加本地缓存 `_cache: Dict[str, SAM3Result]`（基于 image_hash + prompt 的 LRU 缓存，避免同一图同一 prompt 重复调用）。随后 `SAM3RemoteClient` 整合 `Roboflow Inference SDK` 作为商业 fallback（用户可选配 `ROBOFLOW_API_KEY`），令 `segment()` 支持 `backend: Literal["hf_space", "roboflow", "local"] = "hf_space"` 选择，进而异步化为 `async segment()` 使用 `httpx.AsyncClient` 替代同步 gradio_client。最终 `SAM3RemoteClient` 完善 `SAM3Result(masks: List[np.ndarray], boxes: List[Tuple], scores: List[float], labels: List[str])` 数据类型，确保输出格式兼容 `UnifiedLayer` 的 bbox 字段定义，全面将外部 SAM3 API 封装为项目内可插拔的语义分割服务以达成零 GPU 依赖的生产级 SAM3 集成。

**实现文件**: `backend/pipeline/sam3_client.py` (新建)
**已验证可用的 API**: `prithivMLmods/SAM3-Demo` — `/run_image_segmentation`，4.3s/张，置信度 0.98

---

### M106 — SAM3 语义层分离器

从 SAM3 官方的 `sam3/model/sam3_image_processor.py` 中 `class Sam3Processor` 的 `set_image(image, state=None) → state` + `set_text_prompt(prompt, state) → output` 两步 API 这个好例子开始，它将图像编码和文本查询解耦——`set_image()` 一次编码 backbone feature，然后可以用不同的 text prompt 多次查询同一张图的不同目标（`output = processor.set_text_prompt(state=state, prompt="person")`），避免重复编码图像。然后，遵循该模式实现一个新的 `SemanticLayerSplitter.split(image_b64: str, component_hints: List[str]) → List[UnifiedLayer]`，让 `pipeline_orchestrator.py` 可以在 CCL 连通域分析产出不理想时（层数 < 3 或 > 30）自动回退到 `splitter = SemanticLayerSplitter(sam3_client); layers = await splitter.split(frame_b64, hints=["encoder", "decoder", "attention"])` 获得语义正确的层分离，并能利用 ELK 拓扑的节点 label 列表作为 `component_hints` 文本提示，让 SAM3 精确定位每个已知节点。接着 `SemanticLayerSplitter` 引入 `_multi_prompt_batch(image_b64, prompts: List[str]) → Dict[str, np.ndarray]` 批量 prompt 查询，使 15 个节点的架构图可以一次性获得 15 个精确 mask（而不是用 "all objects" 获得不可控的结果），同时增加 `_resolve_overlaps(masks: Dict[str, np.ndarray]) → Dict[str, np.ndarray]` 重叠消解（面积小的 mask 优先级高，防止大 mask 吞掉小 mask）。随后 `SemanticLayerSplitter` 整合 `_mask_to_unified_layer(mask, label, original_image) → UnifiedLayer` 转换器，令每个 SAM3 mask 支持裁剪为 RGBA PNG（mask 区域保留原图像素，非 mask 区域 alpha=0），进而 `split()` 增强返回值携带 `confidence` 评分让下游可以按置信度排序/过滤。最终 `SemanticLayerSplitter` 完善与 `layer_separator.py` 的 `separate_layers()` 共用 `UnifiedLayer` 输出类型，确保下游 `edge_refiner` 和 `component_outliner` 兼容 SAM3 来源的层数据（`source="sam3"`），全面将 CCL 像素级分层升级为可选的语义级分层以达成 Gemini 生图中"粘连节点"的正确分离。

**实现文件**: `backend/pipeline/semantic_splitter.py` (新建)

---

### M107 — Pipeline Orchestrator SAM3 回退集成

从 CCCL 的 `cub/cub/device/device_topk.cuh` 中 `DeviceTopK::TopK()` 的分发逻辑这个好例子开始——它的 `dispatch()` 内部检测输入特征（`num_items`, `k`, 数据类型宽度）来决定是走 radix-based 路径还是 heap-based 路径，关键是 **分发决策在一个地方集中做出，下游的 kernel 不知道自己是"主路径"还是"回退路径"**。然后，遵循该模式实现一个新的 `_stage_layer_separate_with_fallback()` 编排函数，让 `pipeline_orchestrator.py` 的 Stage 2 可以先尝试 CCL 路径（`separate_layers_batch()`），如果结果不理想（`result.layers < 3 or result.layers > 30`）则自动回退到 SAM3 路径（`SemanticLayerSplitter.split()`），并能在 `StageResult.diagnostics` 中记录使用了哪条路径及回退原因。接着集成 `MorphologyParams`（M102）和 `GreenForegroundGuard`（M103）到 Stage 2 前置步骤，使分层前先进行参数自适应和绿色保护扫描，同时 `PipelineConfig` 增加 `sam3_fallback: bool = True` 和 `sam3_confidence_threshold: float = 0.45` 配置项。随后 `config_from_request()` 整合 SAM3 参数解析，令 API 请求支持 `sam3_enabled`, `sam3_prompts` 字段，进而 `_stage_layer_separate_with_fallback()` 增强为三级策略：ELK-guided → CCL → SAM3。最终整合 `UnifiedLayer`（M100）确保三条路径的输出类型完全一致，全面将分层 Stage 升级为多策略自适应系统以达成不同复杂度图片都能获得合理的层分离结果。

**实现文件**: 修改 `backend/pipeline/pipeline_orchestrator.py`

---

### ✦ 第三位 Claude 完成：M108-M109

### M108 — SVG XML 解析器替换（xml.etree 替代 regex）

从 Google JAX 的 `jax/_src/dispatch.py` 中整个 dispatch 系统的设计这个好例子开始——JAX 不用 regex 解析 JAXPR 文本表示，而是直接操作 `core.Jaxpr` 数据结构（`jaxpr.eqns`、`jaxpr.invars`、`jaxpr.outvars`）——这是 **结构化数据用结构化 API 操作** 的典范。然后，遵循该模式实现一个新的 `SVGStructureParser.parse(svg_content: str) → SVGStructure` 结构化解析器，让 `gemini_image_gen.py` 的 `_extract_svg_structure()` 可以用 `tree = SVGStructureParser.parse(svg_content); nodes = tree.find_all_nodes(); edges = tree.find_all_edges()` 替代当前的 `rect_pattern = re.compile(r'<rect\s[^>]*?...')` 脆弱 regex（第 1287-1310 行），并能正确处理带 `transform` 属性的节点（递归累积 `translate(x,y)` 变换到绝对坐标）。接着 `SVGStructureParser` 引入 `_resolve_transforms(element, parent_transform) → (abs_x, abs_y)` 变换解析器，使嵌套 `<g transform="translate(100,200)"><rect x="10" y="20"/></g>` 能够正确计算绝对坐标 `(110, 220)`，同时支持 `style="..."` 内联样式解析（`fill`, `stroke` 提取）。随后 `SVGStructureParser` 整合 `_match_labels_to_rects()` 的 namespace-aware 版本，令 SVG 带 `xmlns:svg` 前缀时也能正确匹配节点和标签，进而 `_extract_svg_structure()` 增强为返回 `SVGStructure` 对象而非字符串拼接。最终 `SVGStructureParser` 完善 `SVGStructure.to_layout_description() → str` 方法，确保输出格式兼容 Grok prompt 的空间布局描述需求，全面将 regex 解析升级为 xml.etree 结构化解析以达成复杂 SVG（含 transform、namespace、内联 style）的正确解析。

**实现文件**: `backend/pipeline/svg_structure_parser.py` (新建) + 修改 `gemini_image_gen.py`

---

### M109 — 前端颜色状态封装（消除全局可变状态）

从 ByteDance/monolith 的 `monolith/native_training/hash_table_ops.py` 中 hash embedding 的设计这个好例子开始——每个 `HashTable` 实例持有自己的 `_table_handle`，不共享全局状态——多个模型可以有多个 hash table，彼此互不干扰。然后，遵循该模式实现一个新的 `class SvgRenderContext` 封装类，让 `to-svg-icons.ts` 的 `elkToSvgIcons(graph)` 可以用 `const ctx = new SvgRenderContext(); return ctx.render(graph)` 替代当前的全局 `let _colorIndex = 0; const _nodeColorMap = new Map()` + `resetColors()` 模式，并能在并发 SSR 请求中互不干扰（每个请求创建自己的 `SvgRenderContext` 实例）。接着 `SvgRenderContext` 引入构造器参数 `palette?: Palette`，使用户可以自定义配色方案，同时保持默认值向后兼容。随后修改 `to-svg.ts` 的同类全局状态问题。最终确保 Astro SSR 的 `import { elkToSvgIcons }` 在服务端渲染时线程安全。

**实现文件**: 修改 `src/lib/elk/to-svg-icons.ts`, `src/lib/elk/to-svg.ts`

---

### ✦ 第四位 Claude 完成：M110-M112

### M110 — Memory Guard + Backpressure 机制

从 Megatron-LM 的 `megatron/core/pipeline_parallel/schedules.py` 中 `forward_backward_pipelining_without_interleaving()` 的 microbatch 管理这个好例子开始——它用 `num_microbatches` 参数控制并发度，`num_warmup_microbatches = min(pp_size - pp_rank - 1, num_microbatches)` 精确计算 pipeline 预热阶段需要多少个 microbatch，然后在 warmup 阶段逐步增加 in-flight 数量——这是 **bounded parallelism with backpressure** 的经典模式。然后，遵循该模式实现一个新的 `MemoryGuard.check_budget(frames_b64: List[str], config: PipelineConfig) → MemoryBudget` 预算估算器，让 `pipeline_orchestrator.py` 的 `run_pipeline()` 入口可以在处理前调用 `budget = MemoryGuard.check_budget(frames, config); if budget.estimated_peak_mb > budget.limit_mb: frames = budget.downsample(frames)` 自动降采样或拒绝过大请求，并能基于 `len(frame_b64) * 3/4`（base64→bytes）× 解码膨胀比 × 管线阶段数 精确估算峰值内存。接着引入分批处理模式（每 8 帧一批），使 32 帧请求不会同时驻留内存。随后整合 `gc.collect()` 在每批完成后显式回收。最终完善 `MemoryBudget.to_dict()` 给前端返回预算信息。

**实现文件**: `backend/pipeline/memory_guard.py` (新建) + 修改 `pipeline_orchestrator.py`

---

### M111 — SAM3 + CCL 融合分层策略

从 NCCL 的 `src/graph/rings.cc` 中 ring 建图算法这个好例子开始——它从拓扑图中找到所有 GPU 之间的连接路径，然后将这些路径组合成 ring（环形通信拓扑），关键是 **先发现所有可能的路径（宽搜索），再从中选择最优组合（贪心选择）**——这是两阶段的发现+融合模式。然后，遵循该模式实现 CCL 和 SAM3 的融合策略：先用 CCL 快速获得像素级连通域（~100ms），再用 SAM3 获得语义级分割（~4s），然后将两者融合——CCL 的连通域边界作为 SAM3 mask 的约束（SAM3 mask 不能跨越 CCL 确认的"非连通"区域），SAM3 的语义信息用来拆分 CCL 的"粘连"连通域。

**实现文件**: `backend/pipeline/fusion_splitter.py` (新建)

---

### M112 — 分层质量评分系统

从 CCCL 的 `cub/cub/device/dispatch/tuning/tuning_topk.cuh` 中 auto-tuning 策略这个好例子开始——它根据输入尺寸（`num_items`, `k`）和硬件（SM count, shared memory size）自动选择最优的线程块配置——关键是 **quality metric 驱动的自动选择**。然后，遵循该模式实现 `LayerQualityScorer.score(layers: List[UnifiedLayer], original_image: np.ndarray) → QualityReport`，包含边缘锐度、覆盖率（layers 拼起来是否覆盖原图所有前景像素）、重叠率（layers 之间是否有不必要的重叠）、面积分布均匀度四个维度。

**实现文件**: `backend/pipeline/layer_quality.py` (新建)

---

## P2 — 拓扑层 → 像素层打通（M113-M120）

### ✦ 第五位 Claude 完成：M113-M115

### M113 — ELK 节点 label → SAM3 prompt 自动桥接

ELK 拓扑生成的节点都有 label（如 "Multi-Head Attention", "Feed Forward Network"）。本里程碑将这些 label 自动转换为 SAM3 text prompt，实现拓扑层和像素层的语义对齐。

从 JAX 的 `jax/_src/interpreters/pxla.py` 中分片规范（sharding spec）到物理设备映射的转换这个好例子开始——逻辑分片名（`PartitionSpec("data", "model")`）被自动映射到物理 GPU mesh 坐标，用户只需要声明逻辑意图。然后，遵循该模式实现 `ELKToSAM3Bridge.convert(elk_graph: dict) → List[SAM3Prompt]`，自动从 ELK 的 `children[].labels[].text` 提取 label，组合为 SAM3 可理解的 text prompt。

**实现文件**: `backend/pipeline/elk_sam3_bridge.py` (新建)

---

### M114 — Gemini 图坐标 ↔ ELK 拓扑坐标对齐

Gemini 生成的图片中元素位置和 ELK 布局的坐标不完全一致。需要一个仿射变换矩阵来对齐。

从 TensorRT 的 `plugin/proposalPlugin/proposalPlugin.cpp` 中 anchor-based 坐标回归这个好例子开始——proposal 网络将 feature map 坐标映射回原图坐标，用 `dx, dy, dw, dh` 做仿射变换。然后实现 `CoordinateAligner.align(elk_bboxes, detected_bboxes) → AffineTransform`，用最小二乘法拟合 ELK 坐标到 Gemini 图像坐标的最优仿射映射。

**实现文件**: `backend/pipeline/coordinate_aligner.py` (新建)

---

### M115 — 双路径 Pipeline 统一调度器

从 Megatron-LM 的 `megatron/core/pipeline_parallel/combined_1f1b.py` 中 combined schedule 这个好例子开始——它将 forward 和 backward pass 交织在一起（1F1B = 1 Forward 1 Backward），在 pipeline 的 steady state 阶段每执行一个 forward microbatch 就立刻执行一个 backward microbatch，最大化 pipeline 利用率。然后，遵循该模式实现统一的 `LayeredImagePipeline` 调度器，将拓扑路径（ELK → Gemini → 分层）和直接图片路径（上传图片 → 分层）统一为一个调度流程。

**实现文件**: `backend/pipeline/unified_scheduler.py` (新建)

---

### ✦ 第六位 Claude 完成：M116-M118

### M116 — 拓扑感知的层命名系统

当前的层命名是 `component_0`, `component_1`...，毫无语义。本里程碑利用 ELK 拓扑的节点 label 和 SAM3 的检测 label 给层赋予有意义的名称。

从 NCCL 的 `topo.cc` 中 `topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "DEV" }` 这个好例子开始——每个拓扑节点类型都有人类可读的字符串名，方便 debug。然后实现 `LayerNamer.name(layers, elk_graph, sam3_results) → List[UnifiedLayer]`，为每个层分配语义名称。

**实现文件**: `backend/pipeline/layer_namer.py` (新建)

---

### M117 — 层间关系推断（contains / connects / parallel）

从 NCCL `src/graph/topo.cc` 中 `ncclTopoConnectNodes(node, remNode, type, bw)` 这个好例子开始——它不仅记录两个节点是否连接，还记录连接类型（`LINK_PCI`, `LINK_NVL`, `LINK_C2C`）和带宽——这是 **typed edge** 的模式。然后实现层间关系推断：根据 bbox 包含关系（A contains B）、ELK 边连接关系（A connects B）、空间并列关系（A parallel B）给层对赋予关系类型。

**实现文件**: `backend/pipeline/layer_relations.py` (新建)

---

### M118 — 层元数据 Schema 定义 + 前端消费

从 JAX 的 `jax/stages.py` 中 `Lowered`, `Compiled` 等 stage 类的设计这个好例子开始——每个编译阶段的产物都是一个有 well-defined API 的对象（`.as_text()`, `.compile()`, `.cost_analysis()`），前后端可以按照稳定的接口消费。然后定义 `LayerMetadata` schema（包含 name, source, relations, quality_score, elk_node_id, sam3_label, bbox, z_index），让前端可以按此 schema 渲染层面板。

**实现文件**: `backend/schemas_layer.py` (新建) + 修改前端 `LayerPreview.astro`

---

### ✦ 第七位 Claude 完成：M119-M120

### M119 — 跨帧层 ID 稳定追踪（匈牙利匹配）

本里程碑实现动画帧序列中层 ID 的跨帧稳定追踪。

从 ByteDance/effective_transformer 的 `cuda/attention.cu` 中 `blockReduceSum` + `blockReduceMax` 的两阶段归约这个好例子开始——先在 warp 内归约（`warpReduceSum`），再在 block 内归约（读 shared memory）——这是两级归约模式。然后用匈牙利算法做跨帧 layer 匹配：第一级用 centroid 距离矩阵做粗匹配，第二级用面积+颜色直方图做精匹配。

**实现文件**: `backend/pipeline/layer_tracker.py` (新建)

---

### M120 — E2E 集成测试 + 基准数据集

本里程碑构建端到端测试套件和基准数据集。

从 OpenAI Triton 的 `python/test/unit/runtime/test_autotuner.py` 中 autotuner 测试这个好例子开始——每个测试用例不只测正确性，还测性能（throughput, latency），并且有 `@pytest.mark.parametrize` 做参数化测试。然后构建一个包含 20 张不同复杂度 Gemini 生图的测试集，对 CCL / SAM3 / Fusion 三种分层策略做自动化 A/B 评估。

**实现文件**: `tests/test_layer_pipeline_e2e.py` (新建) + `tests/fixtures/` (测试图片)

---

## P3 — 性能优化（M121-M128）

### ✦ 第八位 Claude 完成：M121-M122
**M121** — HSV 查表 + rowscan 引擎集成到 green_screen_advanced（兑现 M001/M002 的承诺）
**M122** — 帧间并行：asyncio.gather 替代 sequential for loop

### ✦ 第九位 Claude 完成：M123-M124
**M123** — 层提取流水线化：CCL label → crop → encode 三阶段 pipeline
**M124** — base64 编解码优化：延迟解码 + mmap 大图支持

### ✦ 第十位 Claude 完成：M125-M126
**M125** — Edge Refiner Sobel 向量化：用 scipy.ndimage 替代手写双层循环
**M126** — Component Outliner marching-squares 优化：行扫描替代全图遍历

### ✦ 第十一位 Claude 完成：M127-M128
**M127** — SAM3 调用结果缓存：Redis / 文件系统 LRU
**M128** — Pipeline 整体性能基准：16 帧 1024×1024 目标 < 5s（不含 SAM3）

---

## P4 — 前端分层 UI（M129-M134）

### ✦ 第十二位 Claude 完成：M129-M130
**M129** — 层面板重构：拖拽排序、可见性切换、锁定、重命名
**M130** — 分层来源可视化：CCL/SAM3/ELK 来源图标 + 置信度条

### ✦ 第十三位 Claude 完成：M131-M132
**M131** — 层合成画布：实时预览所有层叠加效果
**M132** — 层导出增强：per-layer PNG + 合成 SVG + MasterGo 格式

### ✦ 第十四位 Claude 完成：M133-M134
**M133** — 分层策略选择器 UI：用户可选 CCL/SAM3/Fusion/Manual
**M134** — 层质量评分可视化：边缘锐度、覆盖率、重叠率仪表盘

---

## P5 — 高级特性（M135-M138）

### ✦ 第十五位 Claude 完成：M135-M136
**M135** — 手动层编辑：用户可在画布上画框手动定义层边界
**M136** — 层间动画建议：基于层关系（contains/connects）自动推荐动画效果

### ✦ 第十六位 Claude 完成：M137-M138
**M137** — MasterGo Import API 层映射：UnifiedLayer → MasterGo Frame/Group 1:1 映射
**M138** — 文档生成：API 文档 + 架构图 + 用户手册

---

## 38 位 Claude 分工总表

| Claude # | 里程碑 | 优先级 | 核心交付物 |
|----------|--------|--------|-----------|
| **1** | M100-M104 | P0 | 统一类型 + 路由修复 + 形态学自适应 + 绿色保护 + 一致性修正 |
| **2** | M105-M107 | P1 | SAM3 Client + 语义分离器 + Orchestrator 回退 |
| **3** | M108-M109 | P1 | SVG 解析器 + 颜色状态封装 |
| **4** | M110-M112 | P1 | Memory Guard + CCL/SAM3 融合 + 质量评分 |
| **5** | M113-M115 | P2 | ELK↔SAM3 桥接 + 坐标对齐 + 统一调度 |
| **6** | M116-M118 | P2 | 层命名 + 层关系 + Metadata Schema |
| **7** | M119-M120 | P2 | 跨帧追踪 + E2E 测试 |
| **8** | M121-M122 | P3 | HSV 集成 + 帧间并行 |
| **9** | M123-M124 | P3 | 层提取流水线 + base64 优化 |
| **10** | M125-M126 | P3 | Edge Refiner 向量化 + Outliner 优化 |
| **11** | M127-M128 | P3 | SAM3 缓存 + 性能基准 |
| **12** | M129-M130 | P4 | 层面板 UI + 来源可视化 |
| **13** | M131-M132 | P4 | 合成画布 + 导出增强 |
| **14** | M133-M134 | P4 | 策略选择器 + 质量仪表盘 |
| **15** | M135-M136 | P5 | 手动编辑 + 动画建议 |
| **16** | M137-M138 | P5 | MasterGo 映射 + 文档 |
| **17-38** | 预留 | — | M003-M010 原计划续接 + 未来需求 |

---

## 依赖图（必须按此顺序）

```
M100 ─────┬─→ M106 ─→ M107
          │            ↑
M101      │   M105 ────┘
          │
M102 ─────┼─→ M107
          │
M103 ─────┘
          
M104 (独立)

M108 (独立)
M109 (独立)
M110 ─→ M107

M105 ─→ M106 ─→ M111 ─→ M112
                  ↑
M100 ─→ M107 ────┘

M113 ─→ M114 ─→ M115
         ↑
M106 ────┘

M116, M117, M118 依赖 M100
M119 依赖 M104
M120 依赖 M107, M111

M121-M128 依赖 M107 (pipeline orchestrator 稳定后)
M129-M134 依赖 M118 (metadata schema 定义后)
M135-M138 依赖 M134
```
