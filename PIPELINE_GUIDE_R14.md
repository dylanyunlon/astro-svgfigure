# Astro-SVGFigure · Cell Pub/Sub Loop 学术架构图渲染世界 Pipeline 全景指引

> **仓库**: `github.com/dylanyunlon/astro-svgfigure`
> **分支**: `cell-pubsub-loop`
> **最新 commit**: `fef686df` — M1050: mega import hoist — 57 files, 0 mid-file imports remaining (2026-06-24)
> **统计**: 356 目录 · 609 文件 · 51 个 upstream 库 · 30 个渲染子模块 · 207 个 SPH 模块 · 32 个着色器 · 58 个 cell 节点

---

## 0. 项目一句话

用 51 个 upstream 渲染/通信/可视化库，通过 Claude 递归调度体系（主 Claude → 小弟 Claude → 小弟的小弟），构建一个把学术论文架构图（当前: 58 节点 Vision2UI 全流程 DAG）渲染成 GPU 驱动、物理收敛、ActiveTheory 级别视觉品质的动态世界。

---

## 1. 代码链条总览（tree 实测）

```
astro-svgfigure/                          # 根目录
│
├── backend/                              # Python 后端 — AI 引擎 + SSE 物理桥
│   ├── pipeline/                         #   9 个管线模块
│   │   ├── edge_routing_prompts.py       #     边路由提示词
│   │   ├── gemini_image_gen.py           #     Gemini 图像生成
│   │   ├── msdf_gen.py                   #     MSDF 距离场生成 (唯一合法临时 SVG)
│   │   ├── nanobanana_bridge.py          #     NanoBanana JSON 桥接 (已重构, 输出 dict)
│   │   ├── scaffold_builder.py           #     脚手架构建
│   │   ├── svg_scaler.py                 #     SVG 缩放
│   │   ├── svg_validator.py              #     SVG 验证
│   │   ├── text_inpainter.py             #     文字修补
│   │   └── topology_gen.py              #     拓扑生成
│   ├── ai_engine.py                      #   Gemini/OpenAI/Anthropic 多模型路由
│   ├── config.py                         #   环境配置
│   ├── schemas.py                        #   Pydantic 数据模型
│   └── sse_physics_bridge.py             #   SSE 实时物理状态推送
│
├── channels/                             # ★ 核心: Apollo CyberRT 移植的 Cell Pub/Sub 引擎
│   ├── cell/                             #   58 个 cell 目录 — DAG 中的每一个节点
│   │   ├── self_attn/                    #     Transformer Self-Attention
│   │   ├── ffn/                          #     Feed-Forward Network
│   │   ├── input_embed/                  #     输入嵌入
│   │   ├── vit_encoder/                  #     ViT 编码器
│   │   ├── transformer_decoder/          #     Transformer 解码器
│   │   ├── mllm_code_agent/              #     MLLM 代码代理
│   │   ├── mllm_style_agent/             #     MLLM 样式代理
│   │   ├── rendered_webpage/             #     渲染后网页
│   │   └── ... (共 58 个)                #     每个含 params.json, bbox.json, status.json, out.json
│   │
│   ├── data/                             #   Apollo 数据层移植 (12 文件)
│   │   ├── subscription_table.py         #     channel → callback 注册 ("channel-faithful" 模型)
│   │   ├── data_dispatcher.py            #     消息分发: cell publish → physics subscribe
│   │   ├── channel_buffer.py             #     环形消息缓冲 (per-channel history)
│   │   ├── data_visitor.py               #     多 channel AllLatest 联合访问
│   │   ├── notifier.py                   #     DataNotifier: 变更事件广播
│   │   ├── fusion_policy.py              #     多源融合策略 (latest / average / priority)
│   │   ├── astro_all_latest.py           #     Astro 定制 AllLatest 融合
│   │   ├── astro_cache_buffer.py         #     Astro 缓存缓冲
│   │   ├── astro_channel_buffer.py       #     Astro channel 缓冲
│   │   ├── astro_data_fusion.py          #     Astro 数据融合
│   │   ├── astro_data_visitor_base.py    #     数据访问基类
│   │   └── f_astro_cell_fusion.py        #     Cell 级别融合策略
│   │
│   ├── transport/                        #   Apollo 传输层移植 (9 子目录)
│   │   ├── transmitter/                  #     4 种发送器: intra, shm, rtps, hybrid
│   │   ├── receiver/                     #     5 种接收器: intra, shm, hybrid, cyber
│   │   ├── dispatcher/                   #     3 种调度器: intra, shm, rtps
│   │   ├── shm/                          #     共享内存: arena + segment + condition_notifier
│   │   ├── message/                      #     消息元数据 + 历史
│   │   ├── rtps/                         #     RTPS 协议 (分布式)
│   │   ├── qos/                          #     QoS 服务质量
│   │   ├── common/                       #     公共工具
│   │   └── transport.py                  #     Transport 顶层抽象
│   │
│   ├── rendering/                        #   ★ Unreal Engine 延迟渲染管线移植 (30 子目录)
│   │   ├── acceleration/                 #     加速结构
│   │   ├── color/                        #     颜色管理
│   │   ├── compositor/                   #     最终合成
│   │   ├── decoration/                   #     装饰元素
│   │   ├── distancefield/                #     MSDF 距离场
│   │   ├── drawcall/                     #     Draw call 管理
│   │   ├── effects/                      #     后处理特效
│   │   ├── lighting/                     #     光照
│   │   ├── lumen/                        #     全局光照 (Lumen GI)
│   │   ├── motionblur/                   #     运动模糊
│   │   ├── nanite/                       #     LOD 简化 (Nanite)
│   │   ├── occlusion/                    #     遮挡剔除
│   │   ├── passes/                       #     渲染 pass
│   │   ├── pathtracing/                  #     路径追踪
│   │   ├── postprocess/                  #     后处理
│   │   ├── reflection/                   #     反射
│   │   ├── registry/                     #     注册表
│   │   ├── resources/                    #     资源管理
│   │   ├── scene/                        #     场景
│   │   ├── shading/                      #     着色
│   │   ├── shadow/                       #     阴影
│   │   ├── species/                      #     Species 视觉渲染
│   │   ├── streaming/                    #     流式加载
│   │   ├── styleprobe/                   #     风格探针 (邻居色彩扩散)
│   │   ├── temporal_aa/                  #     时间抗锯齿
│   │   ├── translucency/                 #     半透明
│   │   ├── utils/                        #     工具
│   │   ├── visibility/                   #     可见性
│   │   ├── misc/                         #     杂项
│   │   └── constants.py                  #     渲染常量
│   │
│   ├── physics/                          #   物理引擎参数 JSON (24+ 文件)
│   │   ├── cell_registry.json            #     Cell 注册信息
│   │   ├── cell_groups.json              #     Cell 分组
│   │   ├── force_field.json              #     力场 {cell_id: {dx, dy, dz, push_from, push_mag}}
│   │   ├── collision.json                #     碰撞检测结果
│   │   ├── edge_routes.json              #     边路由 Spline 控制点
│   │   ├── converged.json                #     收敛状态
│   │   ├── species_assignment.json       #     Species → Cell 分配
│   │   ├── species_physics.json          #     Species 物理特性
│   │   ├── species_visual_traits.json    #     Species 视觉特征
│   │   ├── wind_field.json               #     风场
│   │   ├── terrain_heightmap.json        #     地形高度图
│   │   ├── bloom_variants.json           #     Bloom 变体
│   │   ├── camera_at_params.json         #     摄像机 AT 参数
│   │   ├── dof_at_params.json            #     景深 AT 参数
│   │   ├── fog_at_params.json            #     雾效 AT 参数
│   │   ├── scene_mesh_at_params.json     #     场景网格 AT 参数
│   │   ├── z_layers.json                 #     Z 层排序
│   │   ├── viscosity_diag.json           #     粘度诊断
│   │   ├── implicit_viscosity.py         #     隐式粘度计算
│   │   ├── qos_spatial.py                #     空间 QoS
│   │   ├── terrain_heightmap.py          #     地形高度图生成
│   │   ├── wind_field.py                 #     风场计算
│   │   └── physics-data-exporter.ts      #     物理数据导出
│   │
│   ├── edge/                             #   8 条边: e1-e6 (顺序) + skip1, skip2 (残差)
│   ├── skeleton/                         #   topology.json (58 节点 ELK DAG) + epoch.json
│   ├── convergence/                      #   收敛状态: epoch_params/ + dispatch_log_*.json + status.json
│   ├── service_discovery/                #   服务发现 (5 文件): channel_manager, topology_manager, warehouse, listeners, role
│   ├── scheduler/                        #   调度器
│   ├── component/                        #   组件基类 + timer
│   ├── node/                             #   节点基类
│   │
│   ├── cell_agent.py                     #   ★ Claude 小弟调度入口 — _dispatch_via_hk()
│   ├── cell_component.py                 #   Cell 组件逻辑 (已重构, 只输出 JSON)
│   ├── loop_orchestrator.py              #   ★ Epoch 循环引擎
│   ├── epoch_controller.py               #   Epoch 控制器
│   ├── channel_runtime.py                #   Channel 运行时
│   ├── snapshot_manager.py               #   参数快照 & 回滚
│   ├── topology_to_skeleton.py           #   拓扑 → 骨架转换
│   └── composite_params.json             #   最终合成参数
│
├── src/                                  # ★ 前端: Astro + PixiJS + WebGPU
│   ├── lib/                              #   核心前端库 (20+ 子目录)
│   │   ├── renderer/                     #     AT 风格渲染器: AstroRenderer, AstroProgram, AstroMesh, FXScene
│   │   │   ├── adapter/                  #       Renderer ↔ Program ↔ Material 适配层 (M810)
│   │   │   ├── geometry/                 #       几何体
│   │   │   ├── material/                 #       PBR 材质系统
│   │   │   └── passes/                   #       渲染 pass
│   │   ├── renderers/                    #     48 个渲染模块
│   │   ├── shaders/                      #     32 个着色器 (cil-*.frag species SDF, msdf, edge-spline, caustics 等)
│   │   ├── sph/                          #     ★ SPH 物理 + AT 集成 (207 文件) — 项目最密集模块
│   │   │   ├── SPHWorld.ts               #       SPH 物理世界入口
│   │   │   ├── render-graph.ts           #       M822: FrameGraph 声明式 pass 依赖
│   │   │   ├── at-render-pipeline.ts     #       AT 渲染管线编排
│   │   │   ├── at-scene-compositor.ts    #       AT 场景合成
│   │   │   ├── at-shader-loader.ts       #       AT compiled.vs 解析 + #require 解析
│   │   │   ├── world-orchestrator.ts     #       世界编排器
│   │   │   ├── world-renderer.ts         #       世界渲染器
│   │   │   ├── world-stepper.ts          #       世界步进器
│   │   │   └── ... (207 模块)
│   │   ├── particle/                     #     粒子系统
│   │   ├── elk/                          #     ELK 布局
│   │   ├── gpgpu/                        #     GPGPU 约束求解
│   │   ├── math/                         #     数学库: Vec2/3/4, Mat4, Quat, Box3, Color
│   │   ├── reactflow-elkjs/              #     ReactFlow + ELK 集成
│   │   └── thread/                       #     Web Worker 线程池 (comlink 封装)
│   │
│   ├── components/pipeline/              #   Pipeline 页面 UI 组件
│   ├── pages/                            #   Astro 页面
│   │   ├── pipeline/                     #     架构图渲染管线可视化 + SSE
│   │   ├── world/                        #     全屏物理世界 demo
│   │   ├── showcase/                     #     展示
│   │   ├── playground/                   #     交互实验场
│   │   ├── generate/                     #     AI 生成入口
│   │   └── ...
│   ├── content/                          #   博客 + 文档 Markdown
│   └── layouts/                          #   7 个布局模板
│
├── upstream/                             # ★ 51 个第三方库 (直接引入源码)
│   ├── pixijs-engine/                    #   PixiJS v8 渲染引擎
│   ├── pixijs-filters/                   #   PixiJS 后处理 filter
│   ├── pixijs-filters-v2/                #   30+ 进阶 filter
│   ├── pixijs-ui/                        #   Canvas HUD
│   ├── nanogl/                           #   AT WebGL 封装
│   ├── ogl/                              #   极简 WebGL 参考
│   ├── lygia/                            #   400+ GLSL/WGSL 函数
│   ├── activetheory-assets/              #   AT 生产资产: 182 shaders + 17 geometry + 33 textures
│   ├── activetheory-svg2msdf/            #   SVG→MSDF 纹理
│   ├── apollo-cyber/                     #   ★ CyberRT pub/sub 通信框架
│   ├── theatre-js/                       #   时间线动画编排
│   ├── animation-editor/                 #   AT 节点式动画编辑器
│   ├── unreal-renderer/                  #   UE4 渲染器 C++ (100+ files)
│   ├── unreal-renderer-ue5/              #   UE5 RHI + RenderCore + Shaders
│   ├── 3d-force-graph/                   #   3D 力导向图
│   ├── cytoscape-js/                     #   生物网络图
│   ├── graphology/                       #   图论算法
│   ├── sigma-js/                         #   WebGL 大规模网络渲染
│   ├── molstar/                          #   分子 WebGL 可视化
│   ├── webgpu-ocean/                     #   WebGPU SPH 海洋
│   ├── webgl-water/                      #   WebGL 水面
│   ├── gaussian-splats-at/               #   3D 高斯泼溅
│   ├── comlink/                          #   Web Worker RPC
│   └── ... (共 51 个)
│
├── packages/pure/                        # 纯组件包 (components, libs, plugins, schemas, utils)
├── skills/pixijs/                        # 26 个 PixiJS 技能文档
├── tasks/                                # Epoch 任务调度脚本 (dispatch_*.py)
├── physics/                              # 顶层物理 JSON 副本
├── preset/                               # 预设: icons (9 SVG) + signature 组件
├── public/                               # 静态资源
├── bin/msdfgen.linux                     # MSDF 生成二进制
└── tests/                                # 管线集成测试
```

---

## 2. 最新开发动态（commit 同步 — 截至 2026-06-24 03:02 UTC）

### 当前阶段: Round 13 — 真实 GPU 管线 + 代码工程化 (M980-M1050)

最新一批 commit 呈现两条并行主线:

**主线 A — 真实 GPU 渲染 pass 实现** (M1000-M1049):
10 个子 Claude 并行输出真实 GPU 渲染模块, 每个模块是能在 WebGL2 上运行的完整 pass。

| Commit | 里程碑 | 核心内容 |
|--------|--------|----------|
| `fef686df` | **M1050** | mega import hoist — 57 files, 0 mid-file imports remaining |
| `ffec6eba` | **M1047** | at-spline-particles-full — 真实 GPU 多样条粒子 |
| `f5597b4b` | **M1048** | at-uil-live-panel — 真实 GPU uniform 实时调参面板 |
| `42090afa` | **M1049** | at-lighting — 真实 GPU PBR 多光源 |
| `82c41efc` | **M1030** | at-water-particles-normals — 真实 GPU 水面喷溅 + normalmap |
| `9540d80f` | **M1029** | at-unreal-bloom-pipeline — 真实 GPU full bloom chain |
| `434bc2ae` | **M1007** | ue-lumen-gi — 真实 GPU 辐射缓存全局光照 |
| `2288cfef` | **M1009** | ue-tsr-temporal — 真实 GPU 时间超分辨率 |
| `fb21c508` | **M1001** | at-spline-particle — 真实 GPU transform feedback |
| `8c7d9e51` | **M1008** | ue-bloom-tonemap — 真实 GPU bloom pyramid + ACES |
| `3c16cc4d` | **M1000** | at-world-integrator — 真实 GPU 11-pass orchestrator |
| `5ca05adc` | **M1003** | at-volumetric-light — 真实 GPU ray-march 体积光 |

**主线 B — 代码工程化: import 提升 + 构建修复** (M1010-M1050):
清理 Round 12 遗留的 mid-file import、self-import、orphan line 等工程问题。

| Commit | 内容 |
|--------|------|
| `fef686df` M1050 | 57 文件 mega import hoist, 0 mid-file imports 剩余 |
| `5338c7ee` M1031 | aggressive import hoisting — 431 文件修复 |
| `f7453378` M1020 | batch fix — 42 self-import 移除 + 21 文件 mid-file import 提升 |
| `cb7cc10b` M1001 | batch orphan-line fix — 80 文件 broken import remnants 清理 |
| `2ccd9f1d` M1000 | astro build fixes — glsl plugin, tsconfig, duplicate const/import |

### Round 12 — 58-cell 物理世界完整 dispatch (M970-M989)

| Commit | 里程碑 | 核心内容 |
|--------|--------|----------|
| `ccf00b67` | M980-989 | 10 个 sub-Claude 并行 dispatch — 58-cell 物理世界 |
| `11f0f1bb` | **M980** | topology — 58 nodes + DAG edges |
| `20a03ef2` | M981 | edge routing — all edges waypoints |
| `72b66c6c` | M982 | AT PhysicalShader replaces PBR default |
| `8adf6e98` | M986 | rendering params — 58 cells |
| `ed18ba2e` | M988 | SSE bridge — 58 cells composite push |
| `ba0b1626` | **M979** | world page — 58 cells + SSE + full GPU pipeline |
| `92a737db` | M974 | species + z-layer — 58 cells |
| `43e125ed` | M972 | collision detection + force field — 58 cells |
| `0a4d728d` | M971 | epoch controller — 58 cells skeleton + physics convergence |

### Round 11 — 真实 GPU pass (M933-M965)

| Commit | 模块 | 描述 |
|--------|------|------|
| `29ad6b60` M933 | sdf-gpu-pass | 真实 GPU instanced SDF icons |
| `0911608a` M935 | ue-megalights | 真实 GPU multi-pointlight accumulation |
| `ca941bfe` M937 | ue-ssr-motionblur | 真实 GPU 屏幕空间反射 + 运动模糊 |
| `c32c6b06` M943 | at-text-rendering-msdf | 真实 GPU MSDF 文字渲染 |
| `a6847a28` M945 | draco-geometry-loader | Draco .bin → GPU VBO 解码 |
| `2cc1eaa2` M948 | ue-vsm-shadows | 真实 GPU virtual shadow map |
| `64083a68` M951 | ue-nanite-cull | 真实 GPU LOD + depth culling |
| `0ef15ef0` M965 | pixi-gpu-bridge | PixiJS + raw WebGL 共存 |

### 收敛状态 (来自 convergence/status.json)

```json
{
  "epoch": 2,
  "converged": true,
  "max_delta": 0.0,
  "diverged": false,
  "threshold": 0.5,
  "verified_by": "M879",
  "note": "58-cell DAG: 0 collisions across all z-layers, force_field residuals zeroed"
}
```

---

## 3. 架构核心: Claude 递归调度 + Cell Pub/Sub + 物理收敛

### 3.1 调度体系

```
                     ┌─────────────────────────┐
                     │  主 Claude (管理者)       │
                     │  读取 topology.json       │
                     │  驱动 loop_orchestrator   │
                     │  via claude-hk-config     │
                     └────────┬────────────────┘
                              │ dispatch_cell_agent()
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │ 小弟 A     │  │ 小弟 B     │  │ 小弟 C     │
        │ self_attn  │  │ ffn        │  │ output     │
        │ cil-eye    │  │ cil-bolt   │  │ cil-arrow  │
        │            │  │            │  │            │
        │ web_search │  │ web_search │  │ web_search │
        │ repl (VM)  │  │ repl (VM)  │  │ repl (VM)  │
        │ 可递归调度  │  │ 可递归调度  │  │ 可递归调度  │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │ POST /api/cell/publish        │
              └───────────────┼───────────────┘
                              ▼
                    ┌───────────────────┐
                    │ DataNotifier      │
                    │ (Apollo pub/sub)  │
                    │ 触发物理收敛检查   │
                    └───────────────────┘
```

每个小弟拥有: web_search (搜索学术论文可视化方式)、repl (完整 Linux VM)、POST 回调 (`/api/cell/publish`)、递归调度 (调用自己的小弟)。

**关键约束**: 小弟只生成 JSON 参数, 永不生成 SVG。渲染由主系统 GPU 管线完成。

### 3.2 Epoch 收敛循环

```
Epoch 0 → Epoch 1 → Epoch 2 → ... → Epoch N (converged)
  │           │           │                    │
  ├→ run_all_cells()  (58 cells 并行)          │
  ├→ physics_step()   (碰撞 + 力场)           │
  ├→ check_convergence() (bbox Δ < 0.5?)      │
  └→ NOT converged? → next epoch               │
                                          final assembly
```

收敛常量:
- `_ASTRO_BBOX_TOLERANCE = 0.01` — 单 cell bbox 变化阈值
- `convergence_threshold = 0.5` — 全局收敛判定
- `max_epochs = 10` — 最大迭代
- `_CROWDING_THRESHOLD = 0.35` — 拥挤度阈值
- `_CROWDING_OPACITY_FLOOR = 0.25` — 密集区最低不透明度
- `_ASTRO_CELL_MAX_Z_LAYERS = 10` — 最大 z 层数

### 3.3 当前拓扑 — 58 节点 Vision2UI DAG

```
resolution_handling → input_image → vision2ui_dataset → noise_detection → ...
                   → ram_resource

input_embed → pos_encode → self_attn → add_norm1 → ffn → add_norm2 → output
                   │                       ▲         │          ▲
                   └───── skip1 ───────────┘         └── skip2 ─┘

patch_division → vit_encoder → hidden_states → transformer_decoder → softmax_layer
                                                                   → next_token_prediction

... → leaf_extraction → local_images → mllm_code_agent → local_code_gen → code_linking
                                     → mllm_style_agent → nonleaf_styling

→ output_group → final_css + final_html → rendered_webpage
→ alignment_group → content_align + style_align + structure_align
```

58 个 Cell + 65 条 Edge。ELK 布局: `layered` / `DOWN` / `ORTHOGONAL`。

---

## 4. 渲染管线: Unreal Engine → Web GPU

### 4.1 Unreal → 动态世界映射

| Unreal Engine | 动态世界 | 对应目录 |
|---------------|---------|---------|
| Actor/Component | Cell (Claude 调度的小弟) | `channels/cell/` |
| Scene Graph | ELK Topology JSON (DAG) | `channels/skeleton/topology.json` |
| BasePass → GBuffer | Constraint Collect | `channels/rendering/passes/` |
| Lighting Pass | Constraint Solver | `channels/rendering/lighting/` |
| Shadow Pass | z-layer 阴影 | `channels/rendering/shadow/` |
| Nanite (LOD) | Cell 远景简化 | `channels/rendering/nanite/` |
| Lumen (GI) | Species 色彩扩散 | `channels/rendering/lumen/` |
| PostProcess | 后处理特效链 | `channels/rendering/postprocess/` |
| Motion Blur | 运动模糊 | `channels/rendering/motionblur/` |
| SSR | 屏幕空间反射 | `channels/rendering/reflection/` |
| TAA | 时间抗锯齿 | `channels/rendering/temporal_aa/` |
| Distance Field | MSDF 文字标签 | `channels/rendering/distancefield/` |
| Occlusion Culling | 拥挤遮挡 | `channels/rendering/occlusion/` |
| Path Tracing | Style Probe 邻居采样 | `channels/rendering/pathtracing/` + `styleprobe/` |

### 4.2 严禁硬编码 SVG — 必须通过渲染库管线出图

所有视觉元素的 AT 正确做法:

```
Species icon     →  lygia SDF 函数 → cil-*.frag 着色器
                    → PixiJS Sprite + custom Filter → RenderTexture

Cell body        →  PixiJS Graphics API (drawRoundedRect, lineStyle)
                    + pixijs-filters (GlowFilter, BloomFilter, DropShadowFilter)

Text label       →  activetheory-svg2msdf → MSDF 纹理
                    → msdf.frag/vert GPU shader 渲染
                    + tiny-sdf 浏览器端动态文字

Edge routing     →  edge-spline.frag/vert
                    + lygia generative/snoise 粒子流动
                    + webgl2-particles transform feedback

Post-processing  →  pixijs-filters multi-pass:
                    Kawase-blur + AdvancedBloom + Godray + Glow

Animation        →  theatre-js core/Sequence 关键帧
                    + theatre-js dataverse reactive 状态

Physics layout   →  3d-force-graph d3-force 力导向
                    + channels/physics/ 碰撞力场收敛

Final export     →  PixiJS renderer.extract → Canvas → PNG/SVG
```

唯一合法 SVG 中间文件: `msdf_gen.py` 的 `build_standalone_svg()` — AT svg2msdf 标准做法。

### 4.3 M822 Render Graph — 最新渲染架构

`src/lib/sph/render-graph.ts` (FrameGraph 声明式系统):

```
RenderGraphResource — 虚拟纹理/缓冲槽位, 自动生命周期追踪
RenderGraphPass    — 命名 GPU 工作单元, 声明 inputs/outputs/execute/enabled
RenderGraph        — 顶层容器:
    编译时: 拓扑排序 → 生命周期分析 → 瞬态资源别名
    执行时: 分配/回收纹理 → 按排序执行 enabled pass → 归还瞬态纹理
```

---

## 5. Species 视觉基因系统

10 个 Species, 每个决定 Cell 的渲染身份:

| Species | 颜色 | F0 | 语义 | Icon 特征 | 着色器 |
|---------|------|-----|------|-----------|--------|
| `cil-eye` | 靛蓝 #3F51B5 | 0.04 | 注意力/感知 | 径向光线 + 瞳孔 | `cil-eye.frag` |
| `cil-bolt` | 琥珀 #FF6F00 | 0.80 | 激活函数 | 锯齿闪电路径 | `cil-bolt.frag` |
| `cil-vector` | 森绿 #2E7D32 | 0.04 | 嵌入/投影 | 平行箭头 | `cil-vector.frag` |
| `cil-plus` | 深红 #C62828 | 0.02 | 残差连接 | 十字交汇 | `cil-plus.frag` |
| `cil-arrow-right` | 蓝灰 #455A64 | 0.06 | 数据流/输出 | 右向箭头 | `cil-arrow-right.frag` |
| `cil-filter` | 紫罗兰 #7B1FA2 | 0.65 | 注意力掩码 | 3×3 网格 | — |
| `cil-code` | 绿 #2E7D32 | 0.04 | 函数变换 | 花括号 | — |
| `cil-layers` | 深蓝 #1565C0 | 0.08 | 堆叠表示 | 三层半透明矩形 | — |
| `cil-loop` | 琥珀 #F57F17 | 0.10 | 循环/反馈 | 圆弧箭头 | — |
| `cil-graph` | 深蓝灰 #37474F | 0.03 | 图结构 | 小圆+连线 | — |

Species 分配: `channels/physics/species_assignment.json`, 含 `gene_traits` (primary_shape, pattern, line_style, family)。SDF 着色器: `src/lib/shaders/cil-*.frag`。风格探针扩散: `channels/rendering/styleprobe/`。

---

## 6. 51 个 Upstream 库角色清单

### 6.1 渲染引擎层

| 库 | 角色 |
|---|---|
| **pixijs-engine** (v8) | Cell 实时渲染引擎, 每个 Cell = PixiJS Container |
| **pixijs-filters** (v6) | 后处理: Bloom, Glow, Godray, Kawase-blur, DropShadow, Outline |
| **pixijs-filters-v2** | 30+ 进阶 filter (M815 barrel import) |
| **pixijs-ui** (v2) | Canvas HUD: Slider/Button/ScrollBox |
| **nanogl** | AT WebGL 封装层 (FBO/Program/Texture) |
| **ogl** | 极简 WebGL 参考 |
| **lygia** | 400+ GLSL/WGSL 函数 (SDF, 噪声, PBR, 色彩) |

### 6.2 AT 资产管线

| 库 | 角色 |
|---|---|
| **activetheory-assets** | 生产资产: 182 shaders + 17 geometry + 33 textures (M790 HAR 提取) |
| **activetheory-svg2msdf** | SVG path → MSDF 纹理 |
| **at-svg2msdf-full** | 完整 MSDF 管线 |
| **msdf-atlas-gen** / **msdfgen-source** | MSDF 字体图集 C++ 后端 |
| **tiny-sdf** | 浏览器端实时 SDF 生成 |

### 6.3 动画编排

| 库 | 角色 |
|---|---|
| **theatre-js** | Epoch 时间线: core/ 关键帧 + dataverse/ reactive + studio/ 面板 |
| **animation-editor** | AT 节点式动画编辑器 |
| **activeframe** | rAF + video sync |
| **sketch-js** | 创意编码画布 |

### 6.4 图论与网络

| 库 | 角色 |
|---|---|
| **3d-force-graph** | 3D 力导向图 (d3-force-3d) |
| **cytoscape-js** | 生物网络图 + 自动布局 (COSE/dagre/klay) |
| **graphology** | 图论算法 (最短路径, 社区检测, 中心性) |
| **sigma-js** | WebGL 大规模网络渲染 |
| **cosmos-gl** | WebGL 图渲染 (instanced rendering) |
| **vivagraph** | 力导向图参考 |

### 6.5 科学可视化

| 库 | 角色 |
|---|---|
| **molstar** | 分子 WebGL 可视化 (representation 层次化) |
| **speck** | 原子球模型 (cil-graph 球棍风格) |
| **vtk-js** | 科学数据可视化 (VolumeMapper → z-layer) |
| **ngl** / **ngl-viewer** | 分子图形学 |
| **potree** | 点云渲染 (八叉树 LOD) |

### 6.6 GPU 计算 + 流体

| 库 | 角色 |
|---|---|
| **webgl2-particles** / **-2** | GPU transform feedback (edge 数据流动画) |
| **webgpu-ocean** | WebGPU SPH 海洋模拟 |
| **webgl-water** | WebGL 水面 + caustics |
| **gaussian-splats-at** / **-full** | 3D 高斯泼溅 (alpha 混合参考) |
| **regl-scatter2d** / **regl-scatterplot** | 大规模散点 (碰撞 debug) |
| **icomesh** | 二十面体球面网格 |
| **primitive-geometry** | 基础几何体 |

### 6.7 通信 + 工具

| 库 | 角色 |
|---|---|
| **apollo-cyber** | ★ CyberRT pub/sub Python 移植 |
| **comlink** | Web Worker 透明 RPC |
| **qrious** | QR 码 |
| **thing-editor** | 游戏编辑器参考 |
| **uil** | AT UI Library |

### 6.8 着色器库

| 库 | 角色 |
|---|---|
| **Finding-Love-Shaders** | AT PBR 材质 (Gem/Sky/Terrain) |
| **glsl-colormap** | 色彩映射 |
| **webgl-noise** | GLSL 噪声 |
| **gl-rock** | 岩石材质 |

### 6.9 Unreal 参考

| 库 | 角色 |
|---|---|
| **unreal-renderer** | UE4 渲染器 C++ 源码 (100+ files) |
| **unreal-renderer-ue5** | UE5 RHI + RenderCore + Shaders-Private |

---

## 7. Channel 数据流 — Apollo CyberRT 移植

### 7.1 数据层 (channels/data/)

```
subscription_table.py   — channel → callback 注册表 (Apollo "channel-faithful" 模型)
data_dispatcher.py      — 消息分发: cell publish → physics subscribe
channel_buffer.py       — 环形消息缓冲 (per-channel history)
data_visitor.py         — 多 channel AllLatest 联合访问
notifier.py             — DataNotifier: 变更事件广播
fusion_policy.py        — 多源融合策略 (latest / average / priority)
```

每个 cell 订阅: `skeleton/cell/{id}.json` (骨架信号) + `physics/force_field.json` (力场)。
每个 cell 发布: `cell/{id}/status.json` (bbox + opacity + species_params)。

### 7.2 传输层 (channels/transport/)

```
transmitter/  — 4 种: intra(进程内), shm(共享内存), rtps(RTPS协议), hybrid(自动选择)
receiver/     — 5 种: intra, shm, hybrid, cyber
dispatcher/   — 3 种: intra, shm, rtps
shm/          — 共享内存: arena + segment + condition_notifier
message/      — 消息元数据 + 历史
rtps/         — RTPS 协议 (分布式)
qos/          — QoS 服务质量 profile
```

---

## 8. SPH 物理世界 — 项目最密集模块

`src/lib/sph/` 包含 207 个文件, 核心模块:

| 文件 | 描述 |
|------|------|
| SPHWorld.ts | SPH 物理世界入口 |
| dfsph-solver.ts | DFSPH 压力-散度求解器 |
| pic-flip-solver.ts | PIC/FLIP 混合求解器 |
| render-graph.ts | M822 FrameGraph 声明式渲染 |
| at-render-pipeline.ts | AT 渲染管线编排 |
| at-scene-compositor.ts | AT 场景合成 |
| at-shader-loader.ts | AT compiled.vs 解析 |
| at-mousefluid-import.ts | AT 鼠标流体交互 (M815 最新) |
| world-orchestrator.ts | 世界编排器 |
| world-renderer.ts | 世界渲染器 |
| world-stepper.ts | 世界步进器 |
| CollisionWorld | AABB 碰撞检测 |
| SpatialHashGrid.ts | 空间哈希网格邻居查找 |
| sph-kernels.ts | SPH 核函数 (Cubic Spline, Wendland) |

---

## 9. 32 个着色器索引

```
src/lib/shaders/
├── Species SDF:         cil-eye.frag, cil-bolt.frag, cil-vector.frag, cil-plus.frag, cil-arrow-right.frag
├── MSDF 文字:           msdf.frag, msdf.vert
├── Edge 渲染:           edge-spline.frag/vert, edge-line.frag/vert
├── 水面/流体:           caustics.frag, fluid-surface.frag, water caustics
├── 后处理:              kuwahara-post.frag, lut-pipeline.frag
├── 生物模式:            grayscott-species.frag, voronoi-membrane.frag, voronoi-natural.frag
├── PBR/材质:            pbr-cell-surface.frag, matcap-fresnel-cell.frag
├── 背景/环境:           cloud-fog.frag/vert, julia-background.frag
├── Species 高级:        sdf-species-library.frag, supershape-species.frag, iq-palette-species.frag
├── 粒子:                curl-trail.frag
├── 色彩:                colormap.frag
├── AT 资产:             compiled.vs (182 shaders from AT HAR)
└── 工具:                ShaderLoader.ts, compiler.ts, index.ts
```

---

## 10. M-编号开发节奏索引

| 范围 | 阶段 | 主题 |
|------|------|------|
| **M1000-M1050** | **当前** | Round 13: 真实 GPU pass (bloom, lumen, TSR, particles, lighting, water) + import hoist |
| M980-M989 | Round 12 | 58-cell 物理世界 dispatch (topology, edge, species, composite, SSE) |
| M970-M979 | Round 12 dispatch | 10 sub-Claude: topology + epoch + collision + edge + species + composite |
| M933-M965 | Round 11 | 真实 GPU pass (SDF, megalights, SSR, MSDF, Draco, VSM, Nanite) |
| M815-M822 | AT 集成 | mousefluid, render graph, adapter, barrel exports |
| M800-M814 | AT 加载器 | shader/geometry/texture loader, UIL bridge, post-process |
| M790 | 分水岭 | HAR 提取 AT 生产资产 |
| M770-M799 | 视觉特效 | portal, holographic, weather, lens flare, glow, TAA |
| M750-M769 | 世界编排 | world orchestrator, particle compositor, god rays, theatre |
| M710-M749 | WGSL 移植 | AT shader → WGSL |
| M600-M709 | 生物模式 | reaction-diffusion, physarum, Boids, ocean, caustics |
| M560-M599 | lygia 集成 | voronoi, curl noise, lattice-boltzmann, SDF shapes |
| M500-M559 | SPH 物理世界 | SPHWorld, CollisionWorld, GPGPU, DFSPH |
| M400-M499 | Pipeline SSE | pipeline page SSE driven, CloudFog |

---

## 11. 快速启动

```bash
# Clone & 切换分支
git clone https://github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure && git checkout cell-pubsub-loop

# 后端依赖
pip install fastapi uvicorn pydantic pydantic-settings \
            python-multipart openai anthropic httpx requests

# 前端依赖
bun install

# 一键启动 (后端 8000 + 前端 4321)
bun run dev:all

# Dry-run cell loop (不需要 API key)
cd channels && python cell_agent.py --all --dry-run
python loop_orchestrator.py

# 真实 Claude 调度
git clone https://github.com/dylanyunlon/claude-hk-config.git .claude-hk-config
# 编辑 .claude-hk-config/raw_curl.txt 填入 cookie
cd channels && python cell_agent.py --all
```

---

## 12. 关键文件快速索引

| 需要做什么 | 看哪里 |
|-----------|--------|
| 理解 DAG 拓扑 | `channels/skeleton/topology.json` (58 nodes, 65 edges) |
| Cell 调度 | `channels/cell_agent.py` → `_dispatch_via_hk()` |
| Epoch 循环 | `channels/loop_orchestrator.py` |
| 物理收敛 | `channels/convergence/status.json` + `channels/physics/` |
| 前端渲染器 | `src/lib/sph/` (207 files) + `src/lib/renderer/` |
| 着色器 | `src/lib/shaders/` (32 files) |
| Unreal 映射 | `channels/rendering/` (30 子目录) + `UNREAL_MAPPING.md` |
| AT 资产 | `upstream/activetheory-assets/` (182 shaders + 17 geo + 33 tex) |
| Species 定义 | `channels/physics/species_assignment.json` + `species_visual_traits.json` |
| Pub/Sub 数据流 | `channels/data/` (12 files) |
| 传输层 | `channels/transport/` (9 子目录) |
| 收敛日志 | `channels/convergence/dispatch_log_round*.json` |
| 构建配置 | `astro.config.ts` + `tsconfig.json` + `package.json` |

---

*生成时间: 2026-06-24 · 基于 commit `fef686df` (M1050) · 分支 `cell-pubsub-loop`*
*本文档为学术架构图渲染世界 pipeline 的完整代码链条指引。*
