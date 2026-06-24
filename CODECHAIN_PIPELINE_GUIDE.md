# Astro-SVGFigure · Cell Pub/Sub Loop — 代码链条全景指引

> **仓库**: `github.com/dylanyunlon/astro-svgfigure`
> **分支**: `cell-pubsub-loop`
> **最新 commit**: `fef686df` — M1050: mega import hoist — 57 files, 0 mid-file imports remaining
> **作者**: manager-claude · 2026-06-24 03:02 UTC
> **统计**: 356 目录 · 609 文件 · 51 个 upstream 库 · 59 个 cell · 30 个渲染子模块 · 49 个前端 renderer · 32 个 GPU shader
> **收敛状态**: epoch 2 已收敛, max_delta=0.0, 0 collisions, verified by M879

---

## 0. 一句话定位

**用 51 个 upstream 渲染/通信/可视化库，通过 Claude 递归调度体系（主 Claude → 小弟 Claude → 小弟的小弟），把学术论文架构图（如 Transformer Encoder Block）渲染成 GPU 驱动、物理收敛、ActiveTheory 级别视觉品质的动态世界。**

---

## 1. 顶层目录结构（tree 精读）

```
astro-svgfigure/                        # 项目根
├── backend/                            # [Python] AI 引擎 + SSE 物理桥
│   ├── pipeline/                       #   9 个管线模块
│   │   ├── edge_routing_prompts.py     #     边路由 prompt 模板
│   │   ├── gemini_image_gen.py         #     Gemini 图像生成
│   │   ├── msdf_gen.py                 #     MSDF 距离场字体生成 (唯一合法 SVG 中间文件)
│   │   ├── nanobanana_bridge.py        #     nanogl 桥接 → JSON dict (非 SVG)
│   │   ├── scaffold_builder.py         #     骨架构建器
│   │   ├── svg_scaler.py              #     SVG 缩放工具
│   │   ├── svg_validator.py           #     SVG 校验器
│   │   ├── text_inpainter.py          #     文字修复器
│   │   └── topology_gen.py            #     拓扑生成器
│   ├── ai_engine.py                    #   Gemini/OpenAI/Anthropic 多模型路由
│   ├── config.py                       #   环境配置
│   ├── schemas.py                      #   Pydantic 数据模型
│   └── sse_physics_bridge.py           #   SSE 实时物理推送给前端
│
├── channels/                           # ★ 核心: Apollo CyberRT 移植的 Cell Pub/Sub 引擎
│   ├── cell/          (59 个子目录)     #   每个 cell = DAG 中一个节点
│   ├── data/          (12 文件)         #   Apollo 数据层: subscription_table, dispatcher, buffer, notifier, fusion
│   ├── transport/     (37 文件, 9 子目录) #   Apollo 传输层: transmitter(4种), receiver(5种), dispatcher(3种), shm, rtps, qos
│   ├── rendering/     (88 文件, 30 子目录) #   Unreal Engine 延迟渲染管线移植
│   ├── physics/       (28 JSON + 4 .py)  #   物理引擎参数: force_field, collision, edge_routes, species, wind, terrain
│   ├── convergence/   (13 dispatch_log + status.json) #   收敛记录
│   ├── skeleton/                        #   拓扑骨架: topology.json (65 edges) + epoch.json
│   ├── edge/          (8 子目录)         #   边通道: e1-e6, skip1, skip2
│   ├── service_discovery/               #   Apollo 服务发现: channel_manager, topology_manager, warehouse
│   ├── component/                       #   组件基类 + timer_component
│   ├── node/                            #   节点抽象
│   ├── scheduler/                       #   调度器
│   ├── cell_agent.py                    #   ★ 调度入口: 58/58 cells dispatched (M939 验证)
│   ├── cell_component.py                #   cell 组件 proc()
│   ├── channel_runtime.py               #   channel 运行时
│   ├── loop_orchestrator.py             #   ★ 主循环编排: epoch → cell proc → physics → convergence → assemble
│   ├── epoch_controller.py              #   epoch 控制器
│   ├── snapshot_manager.py              #   参数快照管理 (divergence rollback)
│   └── topology_to_skeleton.py          #   拓扑 → 骨架转换
│
├── src/                                 # [TypeScript/Astro] 前端
│   ├── lib/                             #   ★ 前端渲染库核心
│   │   ├── renderers/    (49 文件)       #     49 个前端渲染器 (pixi-cell, sdf-species, water-bg, epoch-ticker, ...)
│   │   ├── shaders/      (32 文件)       #     32 个 GPU shader (cil-*.frag, edge-spline, msdf, caustics, voronoi, ...)
│   │   ├── renderer/     (AstroRenderer) #     nanogl 端口: AstroMesh/Program/Scene/FXScene/RenderTarget/Nuke
│   │   ├── particle/     (粒子系统)      #     CurlNoise, EdgeParticleSystem, SplineEmitter, proton-controller
│   │   ├── sph/          (SPH 流体)      #     SPHWorld, SPHSolver, GPU 流体模拟
│   │   ├── gpgpu/        (GPGPU 约束)    #     constraint-bridge, constraint-shader
│   │   ├── elk/          (ELK 布局)      #     elkjs 布局 → 交互式 SVG
│   │   ├── math/         (数学库)        #     Vec2/3/4, Mat4, Quat, Color, Box3
│   │   ├── thread/       (Worker 线程)   #     comlink 桥接
│   │   ├── AstroPipeline.ts             #     主 pipeline 入口
│   │   ├── CellEventSource.ts           #     SSE cell 事件源
│   │   ├── CellInteraction.ts           #     cell 交互逻辑
│   │   ├── EdgeRenderer.ts              #     边渲染器
│   │   ├── physics-bridge.ts            #     物理桥接 (前端↔后端)
│   │   └── ...                          #     (BloomVariants, CloudFog, SceneLayoutPresets, etc.)
│   │
│   ├── components/pipeline/             #   pipeline 可视化组件 (Astro/React)
│   ├── pages/pipeline/                  #   /pipeline 页面
│   ├── pages/world/                     #   /world 页面 — 58 cells 动态世界入口
│   ├── pages/generate/                  #   /generate 页面
│   └── content/blog/                    #   博客内容
│
├── upstream/            (51 个库)        # ★ 第三方渲染/通信/可视化库源码
│   ├── activetheory-assets/             #   AT 资产: shaders, geometry, textures, compiled.vs
│   ├── activetheory-svg2msdf/           #   AT SVG→MSDF 转换器
│   ├── apollo-cyber/                    #   Apollo CyberRT: component, data, scheduler, transport, service_discovery
│   ├── lygia/                           #   GLSL shader 库 (SDF, generative, lighting, math, ...)
│   ├── nanogl/                          #   WebGL 低级抽象
│   ├── pixijs-engine/                   #   PixiJS v8 渲染引擎
│   ├── pixijs-filters/                  #   PixiJS 滤镜集
│   ├── theatre-js/                      #   Theatre.js 动画编辑器
│   ├── unreal-renderer/                 #   UE4 Renderer 源码参考 (~150 文件)
│   ├── unreal-renderer-ue5/             #   UE5 Renderer 源码参考
│   ├── comlink/                         #   Web Worker 通信库
│   ├── webgl-water/                     #   WebGL 水面效果
│   ├── webgpu-ocean/                    #   WebGPU 海洋 + SPH
│   ├── ogl/                             #   Minimal WebGL 框架
│   ├── gaussian-splats-at/              #   3D Gaussian Splatting
│   ├── tiny-sdf/                        #   浏览器端 SDF 文字
│   ├── uil/                             #   AT GUI 控制面板
│   └── ...                              #   (共 51 个, 含 molstar, speck, ngl, vtk-js, cytoscape, etc.)
│
├── packages/pure/                       # 纯工具包: components, libs, plugins, schemas, scripts, types, utils
├── skills/pixijs/                       # PixiJS 技能文档 (26 子目录)
├── physics/                             # 顶层物理配置副本
├── preset/                              # 预设: icons (9 SVG), components/signature
├── tasks/                               # 任务脚本: dispatch_*.py, shader inventory, draco decode
├── tests/                               # 测试: shader-compile, pipeline-integration
└── bin/msdfgen.linux                    # msdfgen 二进制
```

---

## 2. 最近的开发方向（最新 30 commits 精读）

### 2.1 当前阶段: M1050 — 大规模 import 整理收尾

| Commit | 里程碑 | 工作内容 |
|--------|--------|---------|
| `fef686df` | **M1050** | mega import hoist — 57 files, 0 mid-file imports remaining |
| `ffec6eba` | M1047 | at-spline-particles-full — real GPU multi-spline particles |
| `f5597b4b` | M1048 | at-uil-live-panel — real GPU uniform live tuning |
| `42090afa` | M1049 | at-lighting — real GPU PBR multi-light |
| `83a338fc` | M1041 | hoist imports — SPHWorld + world-integrator + world-stepper |
| `22354bd1` | M1040 | hoist imports — world-orchestrator + render-compositor + gpu-render-loop |
| `82c41efc` | M1030 | at-water-particles-normals — real GPU water spray + normalmap |
| `b6cd6684` | M1002-1011 | Round 13 — 10 sub-Claudes for astro build fix |
| `9540d80f` | M1029 | at-unreal-bloom-pipeline — real GPU full bloom chain |

### 2.2 开发趋势总结

1. **Real GPU pass 密集实现期** (M1000~M1050): 真正写 WebGL2 shader + pass, 不是 stub
2. **Import 工程化清理** (M1020~M1050): 431+ 文件 import 提升, 消除 mid-file import
3. **Sub-Claude 并行 dispatch**: Round 12~13 (20 sub-Claudes) 并行推进 build fix
4. **Unreal → WebGL 映射完成**: bloom, lumen GI, TSR, VSM shadows, nanite cull, SSR, motion blur, megalights 全部有 real GPU pass

### 2.3 里程碑回顾

| 阶段 | 里程碑范围 | 核心成果 |
|------|-----------|---------|
| 基础架构 | M001-M200 | Apollo CyberRT 移植, Cell Pub/Sub 引擎, 物理收敛循环 |
| 渲染管线 | M200-M600 | Unreal 延迟渲染 30 子模块, PixiJS+nanogl 前端 |
| Species 系统 | M600-M800 | 58 cell species 分配, SDF shader, gene_traits |
| GPU Pass 实现 | M900-M1000 | SDF icons, MSDF text, bloom, SSR, motion blur, lumen GI, megalights |
| Real GPU + 工程化 | M1000-M1050 | spline particles, water normals, PBR lighting, import 清理, build fix |

---

## 3. 学术架构图渲染世界 Pipeline — 完整数据流

```
                    ┌──────────────────────────────────────────┐
                    │           用户输入                        │
                    │  "Transformer Encoder Block 架构图"       │
                    └─────────────────┬────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────────┐
              │  Phase 1: Topology Generation                  │
              │  backend/pipeline/topology_gen.py               │
              │  → 解析论文架构 → 生成 DAG (nodes + edges)      │
              │  → 输出: channels/skeleton/topology.json        │
              │    (当前: 65 edges)                             │
              └─────────────────┬─────────────────────────────┘
                                │
                                ▼
              ┌───────────────────────────────────────────────┐
              │  Phase 2: Species Assignment                   │
              │  channels/physics/species_assignment.json       │
              │  → 58 cells × species 映射                     │
              │  → gene_traits: primary_shape, pattern,         │
              │    line_style, family                           │
              │  → 例: self_attn → cil-eye (attention family)  │
              │        ffn → cil-bolt (transform family)       │
              └─────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Phase 3: Claude 递归调度 — Cell Pub/Sub Epoch Loop          │
    │                                                              │
    │  主 Claude (管理者)                                          │
    │    ├── 读取 topology.json                                    │
    │    ├── 驱动 loop_orchestrator.py                              │
    │    └── dispatch cell_agent.py → 58 sub-Claudes               │
    │                                                              │
    │  每个 sub-Claude (小弟):                                     │
    │    ├── 能力: web_search + repl(Linux VM) + POST回调          │
    │    ├── 订阅: SubscriptionTable (Apollo-faithful)              │
    │    │   → 只收订阅了的 channel, 非邻居广播                     │
    │    ├── 处理: cell_component.proc() → 生成 JSON params         │
    │    │   ⚠️  严禁生成 SVG！只输出 params.json + bbox.json       │
    │    └── 发布: POST /api/cell/publish → DataNotifier            │
    │                                                              │
    │  数据流:                                                      │
    │    Writer → channel_id → DataDispatcher                       │
    │                        → [registered CacheBuffers]            │
    │                        → DataNotifier → [registered callbacks]│
    │                                                              │
    │  收敛循环:                                                    │
    │    epoch N → all cells proc → physics collision detect        │
    │           → force_field update → convergence judge            │
    │           → converged? → assemble composite_params.json       │
    │           → not converged? → epoch N+1                        │
    │                                                              │
    │  当前状态: epoch 2, converged=true, max_delta=0.0             │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Phase 4: 前端 GPU 渲染管线                                   │
    │                                                              │
    │  数据输入:                                                    │
    │    public/channels/composite_params.json                     │
    │    + physics/*.json (force_field, edge_routes, species, etc.) │
    │                                                              │
    │  渲染栈 (AT 方式 — 严禁硬编码 SVG):                          │
    │                                                              │
    │  ┌─ Species Icon ─────────────────────────────────────────┐  │
    │  │ lygia SDF (circleSDF, polySDF, lineSDF)                │  │
    │  │ → src/lib/shaders/cil-*.frag (eye, bolt, arrow, plus)  │  │
    │  │ → PixiJS Sprite + custom Filter → RenderTexture         │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  ┌─ Cell Body ────────────────────────────────────────────┐  │
    │  │ PixiJS Graphics (drawRoundedRect, lineStyle)           │  │
    │  │ + pixijs-filters (Glow, DropShadow, Bloom)             │  │
    │  │ → Container 层级管理                                    │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  ┌─ Text Label ───────────────────────────────────────────┐  │
    │  │ activetheory-svg2msdf → MSDF 纹理                      │  │
    │  │ + src/lib/shaders/msdf.frag/vert                        │  │
    │  │ + tiny-sdf 浏览器端动态文字                              │  │
    │  │ → GPU shader 渲染的距离场 (不是 <text>)                  │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  ┌─ Edge Routing ─────────────────────────────────────────┐  │
    │  │ src/lib/shaders/edge-spline.frag/vert                   │  │
    │  │ + lygia generative/snoise → 粒子流动                    │  │
    │  │ + webgl2-particles transform feedback → 数据流粒子       │  │
    │  │ → GPU 绘制的样条线 (不是 <path d="...">)                │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  ┌─ Post-Processing ──────────────────────────────────────┐  │
    │  │ pixijs-filters: KawaseBlur (散景) + AdvancedBloom (光晕) │  │
    │  │ + ogl FBO ping-pong → 景深模拟                          │  │
    │  │ + caustics-background, water-background                  │  │
    │  │ + theatre-js → 时间轴动画                                │  │
    │  └────────────────────────────────────────────────────────┘  │
    │                                                              │
    │  Unreal Engine 映射的 Real GPU Passes:                       │
    │    bloom-tonemap, lumen-gi, tsr-temporal, vsm-shadows,       │
    │    nanite-cull, ssr-motionblur, megalights, volumetric-light, │
    │    spline-particles, water-normals, pbr-lighting              │
    └─────────────────────────────────────────────────────────────┘
```

---

## 4. 51 个 Upstream 库分类索引

### 4.1 渲染引擎 & GPU
| 库 | 用途 |
|----|------|
| `pixijs-engine` | PixiJS v8 主渲染引擎 |
| `pixijs-filters` / `pixijs-filters-v2` | PixiJS 滤镜集 (Glow, Bloom, Blur, Shadow) |
| `pixijs-ui` | PixiJS UI 组件 |
| `nanogl` | WebGL 低级抽象 (AT Hydra 的 GPU 后端) |
| `ogl` | Minimal WebGL 框架 (FBO, post-process) |
| `unreal-renderer` / `unreal-renderer-ue5` | UE4/5 Renderer C++ 源码参考 |

### 4.2 着色器 & 视觉效果
| 库 | 用途 |
|----|------|
| `lygia` | GLSL shader 超级库 (SDF, noise, lighting, math, 30+ 子目录) |
| `Finding-Love-Shaders` | 创意 shader 效果 |
| `glsl-colormap` | 科学可视化 colormap |
| `webgl-noise` | simplex/perlin noise |
| `webgl-water` | 水面反射折射 |
| `webgpu-ocean` | WebGPU 海洋 + SPH 流体 |
| `gl-rock` | 程序化岩石纹理 |

### 4.3 文字渲染 & SDF
| 库 | 用途 |
|----|------|
| `activetheory-svg2msdf` / `at-svg2msdf-full` | SVG → MSDF 距离场纹理 |
| `tiny-sdf` | 浏览器端实时 SDF 文字生成 |
| `msdfgen-source` / `msdf-atlas-gen` | MSDF 生成器 C++ 源码 |

### 4.4 粒子 & 几何
| 库 | 用途 |
|----|------|
| `webgl2-particles` / `webgl2-particles-2` | WebGL2 transform feedback 粒子 |
| `icomesh` | 正二十面体球面网格 |
| `primitive-geometry` | 基本几何体 |
| `gaussian-splats-at` / `gaussian-splats-at-full` | 3D Gaussian Splatting |
| `potree` | 大规模点云渲染 |

### 4.5 图论 & 布局
| 库 | 用途 |
|----|------|
| `graphology` | 图数据结构 |
| `sigma-js` | 大规模图渲染 |
| `cytoscape-js` | 交互式图可视化 |
| `vivagraph` | 力导向图布局 |
| `3d-force-graph` | 3D 力导向图 |
| `cosmos-gl` | WebGL 大规模图渲染 |

### 4.6 科学可视化
| 库 | 用途 |
|----|------|
| `molstar` | 分子结构可视化 |
| `ngl` / `ngl-viewer` | 蛋白质 3D 查看器 |
| `speck` | 原子可视化 |
| `vtk-js` | 科学可视化框架 |
| `regl-scatter2d` / `regl-scatterplot` | WebGL 散点图 |

### 4.7 通信 & 动画 & 工具
| 库 | 用途 |
|----|------|
| `apollo-cyber` | Apollo CyberRT — Cell Pub/Sub 引擎的源头 |
| `comlink` | Web Worker 通信 (structured clone) |
| `theatre-js` | 时间轴动画编辑器 |
| `animation-editor` | 动画编辑器 |
| `activeframe` | AT 动画帧 |
| `activetheory-assets` | AT 原始资产 (shaders, geometry, textures) |
| `thing-editor` | 游戏编辑器 |
| `sketch-js` | Canvas 2D 创意框架 |
| `uil` | AT GUI 控制面板 |
| `qrious` | QR 码生成 |

---

## 5. channels/ 核心引擎详解

### 5.1 Cell 层 (59 cells)

每个 cell 目录结构:
```
channels/cell/<cell_id>/
├── params.json      # cell 参数 (由 proc() 生成)
├── bbox.json        # 包围盒 {x, y, w, h, z_layer}
├── status.json      # 状态 {processed, epoch}
└── out.json         # 输出数据
```

58 个 species 分配 (来自 species_assignment.json):
- **cil-eye** (attention): self_attn, bbox_pruning_group, ...
- **cil-bolt** (transform): ffn, dec_conv1, dec_conv2, ...
- **cil-plus** (aggregation): add_norm1, add_norm2, ...
- **cil-arrow** (output): output, output_group, ...
- **cil-loop** (iteration): alignment_group, coarse_dom_tree, ...
- **cil-filter** (convolution): enc_conv1, enc_conv2, enc_pool, ...
- **cil-graph** (data): code_linking, json_output, ...
- **cil-vector** (embedding): input_embed, pos_encode, ...

### 5.2 Data 层 — Apollo Pub/Sub (12 文件)

```
subscription_table.py    ← 核心: channel_id → [subscriber callbacks]
                           设计哲学: "中国小米的事不需要反馈给俄罗斯"
                           = 只有订阅了的 cell 才收到通知
data_dispatcher.py       ← 消息分发: cell publish → 只推给 registered buffers
channel_buffer.py        ← 环形消息缓冲 (per-channel history)
data_visitor.py          ← 多 channel AllLatest 联合访问
notifier.py              ← DataNotifier: 变更事件广播
fusion_policy.py         ← 多源融合: latest / average / priority
astro_all_latest.py      ← Astro 定制 AllLatest 融合
astro_cache_buffer.py    ← Astro 缓存缓冲
astro_channel_buffer.py  ← Astro channel 缓冲
astro_data_fusion.py     ← Astro 数据融合
f_astro_cell_fusion.py   ← Cell 级别融合策略
```

### 5.3 Transport 层 — Apollo 传输 (37 文件, 9 子目录)

```
transport/
├── transmitter/        4 种发送器: intra(进程内), shm(共享内存), rtps(RTPS), hybrid
├── receiver/           5 种接收器: intra, shm, hybrid, cyber, ...
├── dispatcher/         3 种调度器: intra, shm, rtps
├── shm/                共享内存: arena + segment + condition_notifier
├── message/            消息元数据 + 历史 + listener handler
├── rtps/               RTPS 协议: attributes + sub_listener + underlay_message
├── qos/                QoS 服务质量 profile
├── common/             公共工具
└── transport.py        Transport 顶层抽象
```

### 5.4 Rendering 层 — Unreal 映射 (88 文件, 30 子模块)

```
rendering/
├── acceleration/     加速结构 (BVH)
├── color/            颜色分级
├── compositor/       最终合成
├── decoration/       装饰元素
├── distancefield/    MSDF 距离场
├── drawcall/         Draw Call 批处理
├── effects/          后处理特效管线
├── lighting/         光照 (PBR)
├── lumen/            全局光照 (Lumen GI)
├── motionblur/       运动模糊
├── nanite/           LOD 简化 (5 文件: composition, draw_list, shading, visibility)
├── occlusion/        遮挡剔除
├── passes/           渲染 pass 管理
├── pathtracing/      路径追踪 + styleprobe 追踪
├── postprocess/      后处理 (bloom + tonemap)
├── reflection/       反射 (SSR)
├── registry/         渲染注册表
├── resources/        资源池
├── scene/            场景管理
├── shading/          PBR 着色
├── shadow/           阴影 (VSM)
├── species/          Species 视觉渲染
├── streaming/        流式加载
├── styleprobe/       风格探针 (邻居色彩扩散)
├── temporal_aa/      时间抗锯齿 (TAA)
├── translucency/     半透明排序
├── utils/            工具
├── visibility/       可见性 (视锥剔除)
└── UNREAL_MAPPING.md 映射文档
```

### 5.5 Physics 层 (28 JSON + 4 Python)

关键物理数据文件:
```
cell_registry.json           # Cell 注册信息
cell_groups.json             # Cell 分组
force_field.json             # 力场 {cell_id: {dx, dy, dz}}
collision.json               # 碰撞检测结果
edge_routes.json             # 边路由 Spline 控制点
species_assignment.json      # 58 cells species 映射
species_physics.json         # Species 物理特性
species_visual_traits.json   # Species 视觉特征
wind_field.json + .py        # 风场
terrain_heightmap.json + .py # 地形高度图
z_layers.json                # Z 层排序
bloom_variants.json          # Bloom 变体
implicit_viscosity.py        # 隐式粘性
qos_spatial.py               # 空间 QoS
```

---

## 6. 前端 GPU 渲染栈详解

### 6.1 49 个 Renderers (src/lib/renderers/)

**核心渲染器**:
- `pixi-cell-renderer.ts` — Cell 主渲染 (PixiJS Graphics + Container)
- `sdf-cell-renderer.ts` — SDF 距离场 cell 渲染
- `sdf-species-filter.ts` — Species 图标 SDF filter
- `cell-label-renderer.ts` — MSDF 文字标签
- `flower-edge-renderer.ts` — 花式边渲染
- `water-background.ts` — 水面背景 (839 行)
- `caustics-background.ts` — 焦散背景
- `nuke-pipeline.ts` — 后处理管线 (AT Nuke 端口)

**系统渲染器**:
- `cell-batch-renderer.ts` — 批处理渲染
- `cell-culling.ts` — 视锥剔除
- `cell-debug-overlay.ts` — 调试覆盖
- `cell-minimap.ts` — 小地图
- `cell-tooltip.ts` — 工具提示
- `cell-selection-ring.ts` — 选中环
- `cell-transition.ts` — 过渡动画
- `pixi-cell-motion.ts` — Cell 运动
- `pixi-blur-cell.ts` — 模糊效果
- `pixi-export.ts` — 导出
- `pixi-filters-registry.ts` — 滤镜注册
- `pixi-render-target.ts` — 渲染目标

**动画 & 粒子**:
- `epoch-ticker.ts` — Epoch 时间驱动
- `epoch-playback-controller.ts` — 回放控制
- `theatre-epoch-timeline.ts` — Theatre.js 时间轴 (686 行)
- `theatre-epoch-cell-bridge.ts` — Theatre ↔ Cell 桥 (776 行)
- `curl-particle-field.ts` — Curl noise 粒子场
- `proton-particles.ts` — Proton 粒子系统
- `fluid-fbo.ts` — 流体 FBO

**AT 端口**:
- `hydra-gl-layer.ts` — Hydra WebGL 层
- `hydra-css.ts` — Hydra CSS 层
- `uil-bridge.ts` — UIL 控制面板桥 (737 行)
- `at-cables-edge.ts` — AT Cables 边渲染
- `glui-system.ts` — GL UI 系统
- `gl-text.ts` — GL 文字渲染

### 6.2 32 个 GPU Shaders (src/lib/shaders/)

**Species 图标 SDF**:
- `cil-eye.frag` — 注意力 (attention)
- `cil-bolt.frag` — 变换 (transform)
- `cil-arrow-right.frag` — 输出 (output)
- `cil-plus.frag` — 聚合 (aggregation)
- `cil-vector.frag` — 嵌入 (embedding)

**边 & 文字**:
- `edge-spline.frag/vert` — GPU 样条线边
- `edge-line.frag/vert` — GPU 直线边 (87 行 vert)
- `msdf.frag/vert` — MSDF 距离场文字

**背景 & 效果**:
- `caustics.frag` — 焦散背景
- `cloud-fog.frag/vert` — 云雾 (47 行 vert)
- `julia-background.frag` — Julia 集背景
- `voronoi-membrane.frag` — Voronoi 膜效果
- `voronoi-natural.frag` — 自然 Voronoi
- `fluid-surface.frag` — 流体表面
- `curl-trail.frag` — Curl 粒子轨迹

**材质 & 色彩**:
- `pbr-cell-surface.frag` — PBR cell 表面
- `matcap-fresnel-cell.frag` — MatCap + Fresnel
- `colormap.frag` — 科学 colormap
- `lut-pipeline.frag` — LUT 色彩管线
- `iq-palette-species.frag` — IQ palette species 颜色
- `grayscott-species.frag` — Gray-Scott 反应扩散
- `supershape-species.frag` — 超级形状

**后处理**:
- `kuwahara-post.frag` — Kuwahara 油画滤镜

**工具**:
- `sdf-species-library.frag` — SDF species 函数库
- `compiler.ts` — Shader 编译器
- `ShaderLoader.ts` — Shader 加载器
- `compiled.vs` — AT 预编译 vertex shader

---

## 7. 关键约束 & 红线

### 7.1 严禁硬编码 SVG

**规则**: Cell proc() 只输出 JSON params, 渲染由前端 GPU pipeline 完成。

已清理的反模式:
- ~~species_port.py 的 generate_svg_cil_*()~~ → SDF shader
- ~~loop_orchestrator.py 的 assemble_final_svg()~~ → composite_params.json
- ~~nanobanana_bridge.py 的 generate_skeleton_svg()~~ → JSON dict
- ~~lighting.py 的 generate_svg_water_overlay()~~ → JSON params

**唯一例外**: `msdf_gen.py` 的 `build_standalone_svg()` — 为 msdfgen 生成临时 SVG 输入文件。

### 7.2 Apollo-faithful Pub/Sub

**规则**: Cell 不广播给邻居。Cell 发布到 channel_id, 只有订阅了该 channel_id 的 cell 才收到通知。

### 7.3 Sub-Claude 能力边界

- ✅ web_search (搜索学术论文可视化方式)
- ✅ repl (Linux VM, 执行代码, 计算参数)
- ✅ POST /api/cell/publish (推送结果回主系统)
- ✅ 递归调度 (小弟也可以调度小弟的小弟)
- ❌ 直接生成 SVG
- ❌ 修改其他 cell 的数据
- ❌ 绕过 SubscriptionTable 直接通信

---

## 8. 代码量统计

| 子系统 | 文件数 | 代码行数 |
|--------|--------|---------|
| src/lib/renderers/ | 49 | 32,310 |
| src/lib/shaders/ | 32 | 6,762 |
| channels/rendering/ | 88 | 32,583 |
| channels/data/ | 12 | ~2,000 |
| channels/transport/ | 37 | ~8,000 |
| backend/pipeline/ | 9 | ~3,000 |
| upstream/ | 51 dirs | ~200,000+ |
| **总计** | **609 files, 356 dirs** | — |

---

## 9. 快速入口

| 目标 | 入口 |
|------|------|
| 运行全部 58 cells | `python3 channels/cell_agent.py` (M939 验证: 58/58, 0 fail, 67s) |
| 查看拓扑 | `channels/skeleton/topology.json` |
| 查看收敛 | `channels/convergence/status.json` |
| 查看物理 | `channels/physics/*.json` |
| 查看前端入口 | `src/pages/world/` |
| 查看管线入口 | `src/lib/AstroPipeline.ts` |
| 查看 SSE 桥 | `backend/sse_physics_bridge.py` + `src/lib/CellEventSource.ts` |
| 查看 Unreal 映射 | `channels/rendering/UNREAL_MAPPING.md` |
| 查看 dispatch 历史 | `channels/convergence/dispatch_log_round*.json` |
| 查看 species 配置 | `channels/physics/species_assignment.json` |

---

*Generated: 2026-06-24 · Branch: cell-pubsub-loop · Commit: fef686df*
