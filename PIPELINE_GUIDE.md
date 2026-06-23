# Astro-SVGFigure · Cell Pub/Sub Loop 学术架构图渲染世界 Pipeline 全景指引

> **仓库**: `github.com/dylanyunlon/astro-svgfigure`
> **分支**: `cell-pubsub-loop`
> **最新 commit**: `f20ea8bf` — M815: AT mousefluid interactive import (2026-06-23)
> **统计**: 355 目录 · 583 文件 · 51 个 upstream 库 · 30 个渲染子模块 · 313 个前端 lib 文件

---

## 0. 项目一句话

**用 49+ 个 upstream 渲染/通信/可视化库，通过 Claude 递归调度体系（主 Claude → 小弟 Claude → 小弟的小弟），构建一个能把学术论文架构图（如 Transformer Encoder Block）渲染成 GPU 驱动、物理收敛、ActiveTheory 级别视觉品质的动态世界。**

---

## 1. 代码链条总览（tree 结构精读）

```
astro-svgfigure/
├── backend/                    # Python 后端 — AI 引擎 + SSE 物理桥
│   ├── pipeline/               # 9 个管线模块 (edge_routing, gemini_image, msdf, scaffold, svg_scaler/validator, text_inpainter, topology_gen)
│   ├── ai_engine.py            # Gemini/OpenAI/Anthropic 多模型路由
│   ├── config.py               # 环境配置
│   ├── schemas.py              # Pydantic 数据模型
│   └── sse_physics_bridge.py   # SSE 实时物理状态推送给前端
│
├── channels/                   # ★ 核心：Apollo CyberRT 移植的 Cell Pub/Sub 引擎
│   ├── cell/                   # 60+ 个 cell 目录 — 每个 cell 是 DAG 中的一个节点
│   │   ├── self_attn/          #   每个子目录含: params.json, bbox.json, status.json, out.json
│   │   ├── ffn/
│   │   ├── input_embed/
│   │   └── ...                 #   涵盖完整 Transformer + 附加架构 (encoder, decoder, vit 等)
│   │
│   ├── data/                   # Apollo 数据层移植 (12 文件)
│   │   ├── subscription_table.py   # channel → callback 注册 (Apollo "channel-faithful" 模型)
│   │   ├── data_dispatcher.py      # 消息分发: cell publish → physics subscribe
│   │   ├── channel_buffer.py       # 环形消息缓冲 (per-channel history)
│   │   ├── data_visitor.py         # 多 channel AllLatest 联合访问
│   │   ├── notifier.py             # DataNotifier: 变更事件广播
│   │   ├── fusion_policy.py        # 多源融合策略 (latest / average / priority)
│   │   ├── astro_all_latest.py     # Astro 定制 AllLatest 融合
│   │   ├── astro_cache_buffer.py   # Astro 缓存缓冲
│   │   ├── astro_channel_buffer.py # Astro channel 缓冲
│   │   ├── astro_data_fusion.py    # Astro 数据融合
│   │   └── f_astro_cell_fusion.py  # Cell 级别融合策略
│   │
│   ├── transport/              # Apollo 传输层移植 (37 文件, 9 子目录)
│   │   ├── transmitter/        #   4 种发送器: intra(进程内), shm(共享内存), rtps(RTPS), hybrid(自动选择)
│   │   ├── receiver/           #   5 种接收器: intra, shm, hybrid, cyber
│   │   ├── dispatcher/         #   3 种调度器: intra, shm, rtps
│   │   ├── shm/                #   共享内存: arena + segment + condition_notifier
│   │   ├── message/            #   消息元数据 + 历史 + listener handler
│   │   ├── rtps/               #   RTPS 协议: attributes + sub_listener + underlay_message
│   │   ├── qos/                #   QoS 服务质量 profile
│   │   └── transport.py        #   Transport 顶层抽象
│   │
│   ├── rendering/              # ★ Unreal Engine 延迟渲染管线移植 (30 子目录, 69 文件)
│   │   ├── acceleration/       #   加速结构
│   │   ├── compositor/         #   最终合成
│   │   ├── distancefield/      #   MSDF 距离场
│   │   ├── effects/            #   后处理特效
│   │   ├── lighting/           #   光照
│   │   ├── lumen/              #   全局光照 (Lumen GI 映射)
│   │   ├── nanite/             #   LOD 简化 (Nanite 映射, 5 文件)
│   │   ├── occlusion/          #   遮挡剔除
│   │   ├── shadow/             #   阴影
│   │   ├── species/            #   Species 视觉渲染
│   │   ├── styleprobe/         #   风格探针 (邻居色彩扩散)
│   │   ├── temporal_aa/        #   时间抗锯齿
│   │   └── ... (共 30 个)
│   │
│   ├── physics/                # 物理引擎参数 JSON (24 文件)
│   │   ├── cell_registry.json      # Cell 注册信息
│   │   ├── force_field.json        # 力场 {cell_id: {dx, dy, dz}}
│   │   ├── collision.json          # 碰撞检测结果
│   │   ├── edge_routes.json        # 边路由 Spline 控制点
│   │   ├── species_assignment.json # Species → Cell 分配
│   │   ├── species_physics.json    # Species 物理特性
│   │   ├── wind_field.json         # 风场
│   │   ├── terrain_heightmap.json  # 地形高度图
│   │   └── ... (bloom, camera, dof, fog, viscosity 等)
│   │
│   ├── edge/                   # 8 条边: e1-e6 (顺序) + skip1, skip2 (残差)
│   ├── skeleton/               # topology.json (ELK DAG) + epoch.json
│   ├── convergence/            # 收敛状态: epoch_params/ + status.json
│   ├── service_discovery/      # 服务发现: channel_manager, topology_manager, warehouse, listeners, role
│   ├── scheduler/              # 调度器
│   ├── component/              # 组件基类 + timer
│   │
│   ├── cell_agent.py           # ★ Claude 小弟调度入口 — _dispatch_via_hk()
│   ├── cell_component.py       # Cell 组件逻辑
│   ├── loop_orchestrator.py    # ★ Epoch 循环引擎 — run_all_cells() → physics_step() → check_convergence()
│   ├── epoch_controller.py     # Epoch 控制器
│   ├── channel_runtime.py      # Channel 运行时
│   ├── snapshot_manager.py     # 参数快照 & 回滚
│   ├── topology_to_skeleton.py # 拓扑 → 骨架转换
│   └── composite_params.json   # 最终合成参数
│
├── src/                        # ★ 前端 Astro + PixiJS + WebGPU
│   ├── lib/                    # 核心前端库 (313 文件, 20 子目录)
│   │   ├── renderer/           #   AT 风格渲染器: AstroRenderer, AstroProgram, AstroMesh, FXScene, Nuke, CellInstanceManager
│   │   │   ├── adapter/        #     Renderer ↔ Program ↔ Material 适配层 (M810)
│   │   │   ├── geometry/       #     几何体
│   │   │   ├── material/       #     PBR 材质系统
│   │   │   └── passes/         #     渲染 pass
│   │   │
│   │   ├── renderers/          #   48 个渲染模块 (pixi-cell-renderer, sdf-species-filter, theatre-epoch-timeline 等)
│   │   ├── shaders/            #   33 个着色器 (cil-*.frag species SDF, msdf.frag/vert, edge-spline, caustics, voronoi 等)
│   │   ├── sph/                #   ★ SPH 物理 + AT 集成 (140+ 文件) — 项目最密集的模块
│   │   │   ├── SPHWorld.ts         # SPH 物理世界
│   │   │   ├── render-graph.ts     # ★ M822: FrameGraph 声明式渲染 pass 依赖 + 自动 FBO 管理
│   │   │   ├── at-mousefluid-import.ts  # ★ M815: AT mousefluid interactive import (最新)
│   │   │   ├── at-render-pipeline.ts    # AT 渲染管线编排
│   │   │   ├── at-scene-compositor.ts   # AT 场景合成
│   │   │   ├── at-shader-loader.ts      # AT compiled.vs 解析 + #require 解析
│   │   │   ├── world-renderer.ts        # 世界渲染器
│   │   │   ├── world-orchestrator.ts    # 世界编排器
│   │   │   └── ... (reaction-diffusion, fluid, collision, particles, effects, 140+ 模块)
│   │   │
│   │   ├── particle/           #   粒子系统: Curl noise, Edge flow, Spline emitter, Proton controller
│   │   ├── elk/                #   ELK 布局: layout + to-svg + interactive-svg + presets
│   │   ├── gpgpu/              #   GPGPU 约束求解: shader + texture bridge
│   │   ├── math/               #   数学库: Vec2/3/4, Mat4, Quat, Box3, Color
│   │   └── thread/             #   Web Worker 线程池 (comlink 封装)
│   │
│   ├── components/pipeline/    # Pipeline 页面 UI 组件
│   ├── pages/                  # Astro 页面 (pipeline/, world/, showcase/, playground/ 等)
│   ├── content/                # 博客 + 文档内容
│   └── layouts/                # 7 个布局模板
│
├── upstream/                   # ★ 51 个第三方库 (直接引入源码)
│   ├── activetheory-assets/    #   AT 生产资产: 182 shaders + 17 geometry + 33 textures (M790 HAR 提取)
│   ├── activetheory-svg2msdf/  #   SVG → MSDF 纹理管线
│   ├── apollo-cyber/           #   Apollo CyberRT pub/sub 通信框架
│   ├── pixijs-engine/          #   PixiJS v8 渲染引擎
│   ├── pixijs-filters/         #   PixiJS 后处理 filter
│   ├── pixijs-filters-v2/      #   PixiJS filters v2 (30+ 效果)
│   ├── lygia/                  #   400+ GLSL/WGSL 函数 (SDF, 噪声, PBR, 色彩)
│   ├── theatre-js/             #   时间线动画编排
│   ├── unreal-renderer/        #   Unreal Engine 4 渲染器源码 (100+ cpp/h)
│   ├── unreal-renderer-ue5/    #   Unreal Engine 5 (RHI, RenderCore, Shaders)
│   ├── nanogl/                 #   AT WebGL 封装层
│   ├── ogl/                    #   极简 WebGL 参考
│   ├── comlink/                #   Web Worker RPC
│   ├── webgpu-ocean/           #   WebGPU 海洋模拟
│   ├── webgl-water/            #   WebGL 水面模拟
│   ├── 3d-force-graph/         #   3D 力导向图
│   └── ... (共 51 个)
│
├── packages/pure/              # 纯组件包 (components, libs, plugins, schemas, utils)
├── skills/pixijs/              # 26 个 PixiJS 技能文档
├── tasks/                      # Epoch 任务调度脚本 (dispatch_*.py)
├── physics/                    # 顶层物理 JSON 副本
├── preset/                     # 预设: icons (9 个 SVG) + signature 组件
├── public/                     # 静态资源: channels/physics + rendering 的 JSON 副本
├── bin/                        # msdfgen.linux 二进制
└── tests/                      # 管线集成测试
```

---

## 2. 最近开发脉络（commit 时间线）

从最新的 commit 往回看，当前开发重心明确分为 **三个阶段**：

### 阶段 Ⅰ — AT 资产直接集成 (M790-M822, 当前进行中)

最新一批 commit 的核心目标：**将 ActiveTheory.net 的生产级渲染管线直接移植到项目中**。

| Commit | 里程碑 | 核心内容 |
|--------|--------|----------|
| `f20ea8bf` | **M815** | AT mousefluid interactive import — AT 鼠标流体交互系统移植 |
| `9be8c2ad` | **M822** | render graph frame scheduler — FrameGraph 声明式 pass 依赖 + FBO 生命周期管理 |
| `57bdf9f4` | M815 | 30+ pixijs-filters-v2 imports → single barrel import 整合 |
| `1ade6cc8` | M816 | 消除所有 JSON imports → runtime fetch 动态加载 |
| `bcb4c030` | **M810** | Renderer ↔ Program ↔ Material adapter layer — AT 抽象层适配 |
| `4f8f6609` | M813 | barrel exports 100% — 所有模块导出完整化 |
| `00c538a7` | **M806** | AT post-process pipeline direct import — FXAA + LensFlare + LightVolume |
| `9827a53d` | M803 | AT UIL bridge — AT UI Library 参数桥接 |
| `1044d3d8` | M808 | AT cables edge connections — AT 连线渲染 |
| `203b5182` | M807 | AT jellyfish.bin as organic cell shape — 有机形态 |
| `26b51ba2` | M802 | AT texture loader — KTX2 PBR 材质 |
| `ee88801b` | **M800** | AT shader loader — compiled.vs parser + `#require` resolver |
| `7e909f29` | M801 | AT geometry loader — Draco .bin decoder |
| `33912085` | **M790** | extract AT production assets from HAR — 182 shaders + 17 geometry + 33 textures + UIL params |

**关键洞察**: M790 是分水岭 — 从 HAR (HTTP Archive) 中提取了 AT 网站的完整生产资产，后续 M800-M822 依次建立加载器链条：shader loader → geometry loader → texture loader → material adapter → post-process → render graph。

### 阶段 Ⅱ — 视觉特效矩阵 (M770-M799)

| Commit | 系统 | 描述 |
|--------|------|------|
| M799 | portal warp effect | 传送门扭曲效果 |
| M798 | holographic display | 全息显示模式 |
| M796-797 | weather + lens flare | 天气粒子 + 镜头光晕 |
| M791-795 | emissive glow, decal, TAA, cell aura, GPU culling | 自发光, 贴花, 时间抗锯齿, cell 能量场, GPU 剔除 |
| M785-789 | temporal AA, shadow PCF, destruction, topology VFX, wind, heat, camera | 全面视觉系统 |
| M780-784 | PIC/FLIP, SSR, multiphase fluid, particle lifecycle, shadow system | 流体 + 粒子 + 阴影三大支柱 |
| M775-779 | SSAO, audio reactive, edge energy, LOD | 环境遮蔽, 音频响应, 边能量流, LOD |

### 阶段 Ⅲ — SPH 物理世界 + 生物模式 (M500-M770)

这是项目从 "静态 SVG 拼接" 蜕变为 "物理驱动的渲染世界" 的核心阶段，详见下文。

---

## 3. 架构核心：Claude 递归调度 + Cell Pub/Sub + 物理收敛

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

**每个小弟 Claude** 拥有 web_search（搜索学术论文可视化方式）、repl（完整 Linux VM 执行代码）、POST 回调（`/api/cell/publish` 推送结果）、递归调度（调用自己的小弟）。

**关键约束**: 小弟只生成 JSON 参数，**永不生成 SVG**。SVG/渲染由主系统 GPU 管线完成。

### 3.2 Epoch 收敛循环

```
Epoch 0 → Epoch 1 → Epoch 2 → ... → Epoch N (converged)
  │           │           │                    │
  ├→ run_all_cells()  (7 cells 并行 dispatch)  │
  ├→ physics_step()   (碰撞检测 + 力场更新)    │
  ├→ check_convergence()                       │
  └→ NOT converged? → next epoch               │
                                          final assembly
```

收敛判定: 所有 cell 的 bbox 变化量之和 < `convergence_threshold` (0.5)。最大 10 个 epoch。

### 3.3 默认拓扑 — Transformer Encoder Block

```
input_embed → pos_encode → self_attn → add_norm1 → ffn → add_norm2 → output
                   │                       ▲         │          ▲
                   └───── skip1 ───────────┘         └── skip2 ─┘
```

7 Cells + 6 顺序 Edges + 2 Skip Connections。ELK 布局: `layered` / `DOWN` / `ORTHOGONAL`。

---

## 4. 渲染管线：从 Unreal Engine 到 Web GPU

### 4.1 Unreal → 动态世界映射

| Unreal Engine | 动态世界 | 对应目录 |
|---------------|---------|---------|
| Actor/Component | Cell (Claude 调度的小弟) | `channels/cell/` |
| Scene Graph | ELK Topology JSON (DAG) | `channels/skeleton/topology.json` |
| BasePass → GBuffer | Constraint Collect | `channels/rendering/passes/` |
| Lighting Pass | Constraint Solver | `channels/rendering/lighting/` |
| Shadow Pass | z-layer 阴影 | `channels/rendering/shadow/` |
| Nanite (LOD) | Cell 远景简化 | `channels/rendering/nanite/` (5 files) |
| Lumen (GI) | Species 色彩扩散 | `channels/rendering/lumen/` |
| PostProcess | 后处理特效链 | `channels/rendering/postprocess/` |
| Distance Field | MSDF 文字标签 | `channels/rendering/distancefield/` |
| Occlusion Culling | 拥挤遮挡 | `channels/rendering/occlusion/` |
| Path Tracing | Style Probe 邻居采样 | `channels/rendering/styleprobe/` |

### 4.2 AT (ActiveTheory) 正确渲染方式

**严禁硬编码 SVG** — 所有视觉元素必须通过渲染库管线出图：

```
Species icon     →  lygia SDF 函数 → cil-*.frag 着色器 → PixiJS Sprite + Filter → RenderTexture
Cell body        →  PixiJS Graphics API → pixijs-filters (Glow, Bloom, Shadow) → Container
Text label       →  activetheory-svg2msdf → MSDF 纹理 → msdf.frag/vert GPU 渲染
Edge routing     →  edge-spline.frag/vert → lygia snoise 粒子流动 → GPU 样条线
Post-processing  →  pixijs-filters multi-pass: Kawase-blur, AdvancedBloom, Godray, Glow
Animation        →  theatre-js Sequence 关键帧 → JS 帧循环
Physics layout   →  3d-force-graph d3-force → channels/physics/ 碰撞 → 力场收敛
Final export     →  PixiJS renderer.extract → Canvas → PNG/SVG
```

### 4.3 M822 Render Graph — 最新渲染架构

`src/lib/sph/render-graph.ts` 引入了 FrameGraph 声明式系统：

```
RenderGraphResource — 虚拟纹理/缓冲槽位, 自动生命周期追踪
RenderGraphPass    — 命名 GPU 工作单元, 声明 inputs/outputs/execute/enabled
RenderGraph        — 顶层容器:
    编译时: 拓扑排序 → 生命周期分析 → 瞬态资源别名 (共享 GPUTexture)
    执行时: 分配/回收纹理 → 按排序执行 enabled pass → 归还瞬态纹理
```

这替代了之前 M720 `at-render-pipeline.ts` 的手工串联和 M730 `at-scene-compositor.ts` 的命令式 tick() 排序。

---

## 5. Species 视觉基因系统

每个 Cell 被分配一个 species，决定其渲染身份：

| Species | 颜色 | F0 反射率 | 语义 | Icon 特征 | 着色器 |
|---------|------|----------|------|-----------|--------|
| `cil-eye` | 靛蓝 #3F51B5 | 0.04 | 注意力/感知 | 径向光线 + 瞳孔 | `cil-eye.frag` |
| `cil-bolt` | 琥珀 #FF6F00 | 0.80 | 激活函数 | 锯齿闪电路径 | `cil-bolt.frag` |
| `cil-vector` | 森绿 #2E7D32 | 0.04 | 嵌入/投影 | 平行箭头 | `cil-vector.frag` |
| `cil-plus` | 深红 #C62828 | 0.02 | 残差连接 | 十字交汇 | — |
| `cil-arrow-right` | 蓝灰 #455A64 | 0.06 | 数据流/输出 | 右向箭头 | `cil-arrow-right.frag` |
| `cil-filter` | 紫罗兰 #7B1FA2 | 0.65 | 注意力掩码 | 3×3 网格 | — |
| `cil-code` | 绿 #2E7D32 | 0.04 | 函数变换 | 花括号 | — |
| `cil-layers` | 深蓝 #1565C0 | 0.08 | 堆叠表示 | 三层半透明矩形 | — |
| `cil-loop` | 琥珀 #F57F17 | 0.10 | 循环/反馈 | 圆弧箭头 | — |
| `cil-graph` | 深蓝灰 #37474F | 0.03 | 图结构 | 小圆+连线 | — |

Species 分配存储在 `channels/physics/species_assignment.json`，包含 `gene_traits` (primary_shape, pattern, line_style, family)。SDF 着色器在 `src/lib/shaders/cil-*.frag`。风格探针扩散在 `channels/rendering/styleprobe/`。

---

## 6. 51 个 Upstream 库角色清单

### 6.1 渲染引擎层

| 库 | 角色 |
|---|---|
| **pixijs-engine** (v8) | Cell 实时渲染引擎。每个 Cell = PixiJS Container (Sprite + Text + Graphics)。Ticker 驱动 epoch 动画帧 |
| **pixijs-filters** (v6) | 后处理链: Bloom, Glow, Godray, Kawase-blur, DropShadow, Outline |
| **pixijs-filters-v2** | 30+ 进阶 filter (M815 整合为 barrel import) |
| **pixijs-ui** (v2) | Canvas HUD: Slider/Button/ScrollBox |
| **nanogl** | AT WebGL 封装层 (FBO/Program/Texture) |
| **ogl** | 极简 WebGL 参考 (Geometry/Mesh/Camera 设计模式) |
| **lygia** | 400+ GLSL/WGSL 函数 (SDF 图元, 噪声, PBR, 色彩混合) |

### 6.2 AT 资产管线

| 库 | 角色 |
|---|---|
| **activetheory-assets** | 生产资产: 182 shaders + 17 geometry + 33 textures + UIL params (M790 HAR 提取) |
| **activetheory-svg2msdf** | SVG path → MSDF 纹理 |
| **at-svg2msdf-full** | 完整 MSDF 管线 |
| **msdf-atlas-gen** / **msdfgen-source** | MSDF 字体图集 C++ 后端 |
| **tiny-sdf** | 浏览器端实时 SDF 生成 |

### 6.3 动画编排

| 库 | 角色 |
|---|---|
| **theatre-js** | Epoch 时间线: core/ 关键帧, dataverse/ reactive 状态, studio/ 编辑面板 |
| **animation-editor** | AT 节点式动画编辑器 (Flow 图编辑 cell 数据流) |
| **activeframe** | rAF + video sync 帧同步 |
| **sketch-js** | 创意编码画布 |

### 6.4 图论与网络

| 库 | 角色 |
|---|---|
| **3d-force-graph** | 3D 力导向图 (d3-force-3d 物理模型) |
| **cytoscape-js** | 生物网络图 + 自动布局 (COSE/dagre/klay) |
| **graphology** | 图论算法 (最短路径, 社区检测, 中心性) |
| **sigma-js** | WebGL 大规模网络渲染 (100+ cells 切换到 GPU) |
| **cosmos-gl** | WebGL 图渲染 (instanced rendering 优化) |
| **vivagraph** | 力导向图参考 |

### 6.5 科学可视化

| 库 | 角色 |
|---|---|
| **molstar** | 分子 WebGL 可视化 (representation 层次化设计模式) |
| **speck** | 原子球模型 (cil-graph species 球棍风格参考) |
| **vtk-js** | 科学数据可视化 (VolumeMapper, ContourFilter → z-layer 等值面) |
| **ngl** / **ngl-viewer** | 分子图形学 |
| **potree** | 点云渲染 (八叉树 LOD 策略) |

### 6.6 GPU 计算 + 流体

| 库 | 角色 |
|---|---|
| **webgl2-particles** / **webgl2-particles-2** | GPU transform feedback 粒子系统 (edge 数据流动画) |
| **webgpu-ocean** | WebGPU SPH 海洋模拟 |
| **webgl-water** | WebGL 水面模拟 + caustics |
| **gaussian-splats-at** / **full** | 3D 高斯泼溅 (alpha 混合 bloom 参考) |
| **regl-scatter2d** / **regl-scatterplot** | 大规模散点 (碰撞检测 debug) |
| **icomesh** | 二十面体球面网格 |
| **primitive-geometry** | 基础几何体 |

### 6.7 通信 + 工具

| 库 | 角色 |
|---|---|
| **apollo-cyber** | ★ CyberRT pub/sub 通信框架 Python 移植 |
| **comlink** | Web Worker 透明 RPC (前端 renderer ↔ physics-worker) |
| **qrious** | QR 码生成 |
| **thing-editor** | 游戏编辑器参考 |

### 6.8 着色器库

| 库 | 角色 |
|---|---|
| **Finding-Love-Shaders** | AT PBR 材质 (Gem/Sky/Terrain → species 外观) |
| **glsl-colormap** | 色彩映射 |
| **webgl-noise** | GLSL 噪声函数 |
| **gl-rock** | 岩石材质着色器 |

### 6.9 Unreal 参考

| 库 | 角色 |
|---|---|
| **unreal-renderer** | UE4 渲染器 C++ 源码 (100+ files, 延迟渲染管线参考) |
| **unreal-renderer-ue5** | UE5 (RHI + RenderCore + Renderer-Private + Shaders-Private) |

---

## 7. 物理引擎深度

### 7.1 SPH (Smoothed Particle Hydrodynamics) 世界

`src/lib/sph/` 是项目最密集的模块 (140+ 文件)，包含：

- **SPHWorld.ts** — SPH 物理世界入口 (M500: 6814 行)
- **dfsph-solver.ts** — DFSPH 压力-散度求解器 (M540)
- **pic-flip-solver.ts** — PIC/FLIP 混合求解器 (M780)
- **CollisionWorld** — AABB 碰撞检测
- **SpatialHashGrid.ts** — 空间哈希网格邻居查找
- **sph-kernels.ts** — SPH 核函数 (Cubic Spline, Wendland)

### 7.2 参数流

```
channels/physics/force_field.json   ← physics_step() 更新
channels/physics/collision.json     ← CollisionWorld 输出
channels/physics/species_physics.json → SPHWorld 读取
channels/physics/wind_field.json    → 风场施加
channels/physics/terrain_heightmap.json → 地形碰撞
channels/convergence/status.json    ← epoch 收敛状态
```

### 7.3 收敛常量

```python
_ASTRO_BBOX_TOLERANCE     = 0.01   # bbox 变化检测阈值
_ASTRO_CELL_MAX_Z_LAYERS  = 10     # 最大 z 层数
_CROWDING_THRESHOLD       = 0.35   # 拥挤度阈值
_CROWDING_OPACITY_FLOOR   = 0.25   # 密集区域最低不透明度
max_epochs                = 10     # 最大迭代次数
convergence_threshold     = 0.5    # 全局收敛阈值
```

---

## 8. 前端页面路由

| 路径 | 用途 |
|------|------|
| `/pipeline/` | 架构图渲染管线可视化 + SSE 实时状态 (M421) |
| `/world/` | ★ 全屏物理世界 demo — WebGPU SPH + AT 渲染 (M752) |
| `/showcase/` | 展示页面 |
| `/playground/` | 交互实验场 |
| `/generate/` | AI 生成入口 |
| `/gallery/` | 画廊 |
| `/blog/` | 博客 |
| `/docs/` | 文档 |

---

## 9. 快速启动

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

## 10. 开发节奏 M-编号索引

| 范围 | 阶段 | 主题 |
|------|------|------|
| M815-M822 | 当前 | AT 资产直接集成 (mousefluid, render graph, adapter, barrel exports) |
| M800-M814 | AT 加载器 | shader loader, geometry loader, texture loader, UIL bridge, post-process |
| M790 | 分水岭 | HAR 提取 AT 生产资产 (182 shaders + 17 geometry + 33 textures) |
| M770-M799 | 视觉特效 | portal, holographic, weather, lens flare, glow, TAA, aura, GPU culling |
| M750-M769 | 世界编排 | world orchestrator, particle compositor, god rays, theatre bridge, MSDF labels |
| M710-M749 | WGSL 移植 | AT shader → WGSL: Navier-Stokes, VolumetricLight, Bloom, SplineParticle, Water, Flower, PBR |
| M600-M709 | 生物模式 | reaction-diffusion, physarum, Boids, differential growth, ocean, caustics |
| M560-M599 | lygia 集成 | voronoi, curl noise, lattice-boltzmann, ripple, gerstner wave, SDF shapes |
| M500-M559 | SPH 物理世界 | SPHWorld, CollisionWorld, GPGPU, fluid-rigid coupling, DFSPH |
| M400-M499 | Pipeline SSE | pipeline page SSE driven, CloudFog |

---

*本文档为 `cell-pubsub-loop` 分支的学术架构图渲染世界 pipeline 全景指引。*
*生成时间: 2026-06-23 · 基于 commit `f20ea8bf` (M815)*
