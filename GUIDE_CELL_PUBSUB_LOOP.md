# Cell Pub/Sub Loop 动态世界 Pipeline 指引

> **分支**: `cell-pubsub-loop`
> **最新 commit**: `1f3ecc6` — RESEARCH #130 — AT balance-text + fit-text
> **核心理念**: 主 Claude 管理者通过 `claude-hk-config` 调用小弟 Claude，小弟也能作为管理者递归调用小弟的小弟，共同构建一个用上了 49 个 upstream 渲染/通信/可视化库的绘图世界。

---

## 1. 动态世界架构：Claude 递归调度体系

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
        │ cell_id:   │  │ cell_id:   │  │ cell_id:   │
        │ self_attn  │  │ ffn        │  │ output     │
        │ species:   │  │ species:   │  │ species:   │
        │ cil-eye    │  │ cil-bolt   │  │ cil-arrow  │
        │            │  │            │  │            │
        │ 能力:      │  │ 能力:      │  │ 能力:      │
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

**每个小弟 Claude 拥有的能力**：

- **web_search**: 搜索学术论文中该概念的典型可视化方式
- **repl (Linux VM)**: 完整 Linux 环境，可执行代码、计算参数
- **POST 回调**: 通过 `/api/cell/publish` 将计算结果推送回主系统
- **递归调度**: 小弟也可作为管理者，通过 claude-hk-config 调度自己的小弟

**关键约束：小弟只生成 JSON 参数，不生成 SVG**。SVG 由绝对静止模式下的主 Claude 直接生成（可参考 Species 的 icon 风格）。

---

## 2. 严禁硬编码 SVG — 必须通过渲染库管线出图

### 2.1 当前问题：大量 SVG 字符串拼接

以下文件是**反模式**，必须重构为通过渲染库管线输出：

| 文件 | 问题 | 硬编码行数 |
|------|------|-----------|
| `channels/rendering/species/species_port.py` | 10 个 `generate_svg_cil_*()` 函数直接拼 `<rect>`/`<circle>`/`<line>`/`<path>`/`<text>` 字符串 | ~64 行 SVG 拼接 |
| `channels/cell_component.py` | `SPECIES_GENERATORS` dict 调用上述函数，字符串替换颜色，拼出最终 `<g>` wrapper | 全文 567 行围绕字符串替换 |
| `channels/loop_orchestrator.py` `assemble_final_svg()` | 拼 `<svg>` root + `<defs>` + `<linearGradient>` + z-layer `<g>` + edge `<path>` | ~100 行 SVG 模板 |
| `backend/pipeline/nanobanana_bridge.py` `generate_skeleton_svg()` | 拼骨架 `<rect>` + `<text>` + arrow `<path>` | ~40 行 |
| `channels/rendering/lighting/lighting.py` | `generate_svg_water_overlay()` 拼 `<rect>` 叠加层 | ~20 行 |
| `channels/rendering/decoration/decoration_extra.py` | 拼 `<circle>`/`<line>`/`<path>` 装饰元素 | ~30 行 |

### 2.2 ActiveTheory 的正确做法

ActiveTheory.net **从不手写 SVG 字符串**。它的渲染管线是：

```
upstream 库的正确用法 (AT 方式):

Species icon     →  lygia SDF 函数 (circleSDF, polySDF, lineSDF)
                    + src/lib/shaders/cil-*.frag 着色器
                    → PixiJS Sprite + custom Filter 渲染到 RenderTexture
                    → 最终导出: renderer.extract.canvas() → canvas.toDataURL()

Cell body        →  PixiJS Graphics API (drawRoundedRect, lineStyle)
                    + pixijs-filters (GlowFilter, DropShadowFilter, BloomFilter)
                    → Container 层级管理

Text label       →  activetheory-svg2msdf 转换 → MSDF 纹理
                    + src/lib/shaders/msdf.frag/vert 渲染
                    + tiny-sdf 浏览器端动态文字
                    → 不是 <text> 标签，是 GPU shader 渲染的距离场

Edge routing     →  src/lib/shaders/edge-spline.frag/vert
                    + lygia generative/snoise 做粒子流动
                    + webgl2-particles transform feedback 做数据流粒子
                    → 不是 <path d="...">，是 GPU 绘制的样条线

Post-processing  →  pixijs-filters: KawaseBlurFilter (散景)
                    + AdvancedBloomFilter (光晕)
                    + GodrayFilter (光束)
                    + GlowFilter (外发光)
                    → 不是 SVG filter，是 WebGL multi-pass

Animation        →  theatre-js core/Sequence 驱动关键帧
                    + theatre-js dataverse 监听参数变化
                    → 不是 CSS animation，是 JS 驱动的帧循环

Physics layout   →  3d-force-graph d3-force 力导向
                    + channels/physics/ 碰撞检测
                    → 不是手算坐标偏移，是物理模拟收敛

Final export     →  PixiJS renderer.extract → Canvas → PNG/SVG
                    + modern-screenshot DOM→Canvas→Image
                    → 不是字符串拼接，是渲染结果序列化
```

### 2.3 重构方向

**删除** `species_port.py` 中所有 `generate_svg_*` 函数。替换为：

```python
# 重构后的 species_port.py 应该长这样:

class SpeciesRenderer:
    """通过 PixiJS + shader 渲染 species icon，不拼 SVG 字符串。"""

    def __init__(self, pixi_app):
        self.app = pixi_app
        # 加载 SDF shader (来自 lygia + cil-*.frag)
        self.sdf_programs = load_species_sdf_shaders()
        # 加载 MSDF 字体 (来自 activetheory-svg2msdf)
        self.msdf_atlas = load_msdf_atlas()
        # 加载 PixiJS filters (来自 pixijs-filters)
        self.bloom = AdvancedBloomFilter(...)
        self.glow = GlowFilter(...)

    def render_cell(self, cell_id, species, bbox, params):
        """渲染一个 cell — 全部通过 GPU 管线。"""
        container = Container()

        # 1. Body: PixiJS Graphics API
        body = Graphics()
        body.beginFill(SPECIES_COLORS[species], params['opacity'])
        body.drawRoundedRect(0, 0, bbox['w'], bbox['h'], params.get('corner_radius', 8))
        container.addChild(body)

        # 2. Icon: SDF shader 渲染 (不是硬编码 <circle>/<line>)
        icon_mesh = Mesh(PlaneGeometry(), self.sdf_programs[species])
        icon_mesh.shader.uniforms.update(params['species_params'])
        container.addChild(icon_mesh)

        # 3. Label: MSDF 渲染 (不是 <text> 标签)
        label = MSDFText(self.msdf_atlas, cell_label, fontSize=12)
        container.addChild(label)

        # 4. Effects: PixiJS filter chain (不是 SVG filter)
        container.filters = [self.glow, self.bloom]

        return container  # 返回 PixiJS 对象，不是字符串
```

**cell_component.py** 应该调用 `SpeciesRenderer.render_cell()` 返回 PixiJS Container，而不是拼 SVG 字符串再 `write_channel("svg.svg", ...)`.

**assemble_final_svg()** 应该通过 PixiJS `renderer.extract` 从渲染结果导出，而不是拼 `<svg>` 根元素。

---

## 3. claude-hk-config 调度机制

调度入口在 `channels/cell_agent.py` 的 `_dispatch_via_hk()` 函数：

```
.claude-hk-config/
├── raw_curl.txt      # cookie + org_id + origin (认证凭据)
└── ...
```

调度流程：
1. 从 `raw_curl.txt` 解析 cookie、org_id、origin、user-agent
2. 创建新对话 (`POST /api/organizations/{org_id}/chat_conversations`)
3. 构建 prompt = SPECIES_PROMPT + RESEARCH_PREFIX + skeleton/force_field 上下文
4. 发送 completion 请求，启用 `web_search_v0` + `repl_v0` 工具
5. 小弟在自己的 VM 中执行代码，POST 参数到 `/api/cell/publish`
6. 主系统 DataNotifier 广播变更，触发下一轮物理计算

```python
# cell_agent.py 核心调度
def _dispatch_via_hk(system_prompt, user_message, cell_id, skeleton):
    # ... 解析 raw_curl.txt 获取认证 ...
    # ... 创建对话 ...
    # ... 构建 prompt: 搜索+分析+发布 三步工作流 ...
    # ... POST completion (fire and forget) ...
    # 小弟在后台运行，通过 /api/cell/publish 推送结果
```

---

## 4. Cell Pub/Sub Loop — Epoch 收敛引擎

### 3.1 核心循环 (`loop_orchestrator.py`)

```
Epoch 0 ─── Epoch 1 ─── Epoch 2 ─── ... ─── Epoch N (converged)
  │            │            │                      │
  ├→ run_all_cells()  (7 cells 并行 dispatch)      │
  ├→ physics_step()   (碰撞检测 + 力场更新)        │
  ├→ check_convergence() (bbox 变化 < 0.01?)       │
  └→ NOT converged → next epoch                    │
                                              assemble_final_svg()
```

### 3.2 Channel 数据流 (Apollo CyberRT 移植)

```
channels/data/
├── subscription_table.py   # channel → callback 注册表
│   每个 cell 订阅:
│     - skeleton/cell/{id}.json (骨架信号)
│     - physics/force_field.json (力场)
│   每个 cell 发布:
│     - cell/{id}/status.json (bbox + opacity + species_params)
│
├── data_dispatcher.py      # 消息分发: cell publish → 物理引擎 subscribe
├── channel_buffer.py       # 环形消息缓冲 (per-channel history)
├── data_visitor.py         # 多 channel 联合访问 (AllLatest 融合)
├── notifier.py             # DataNotifier: 变更事件广播
└── fusion_policy.py        # 多源融合策略 (latest / average / priority)
```

**关键修复** (commit `1b7fde0`): SubscriptionTable 实现了 Apollo 的 "channel-faithful" 订阅模型。Cell 之间只通过显式声明的 channel 通信，而非邻居广播。commit `66412ba` 修复了 skip edge 导致的 writer/reader 重复注册问题。

### 3.3 Transport 层

```
channels/transport/
├── transmitter/
│   ├── intra_transmitter.py    # 进程内直传 (开发模式)
│   ├── shm_transmitter.py      # 共享内存 (生产模式)
│   ├── rtps_transmitter.py     # RTPS 协议 (分布式)
│   └── hybrid_transmitter.py   # 自动选择最优传输
├── receiver/
│   ├── intra_receiver.py
│   ├── shm_receiver.py
│   ├── hybrid_receiver.py
│   └── cyber_receiver.py
├── dispatcher/
│   ├── intra_dispatcher.py     # 同进程调度
│   ├── shm_dispatcher.py       # 共享内存调度
│   └── rtps_dispatcher.py      # 网络调度
├── shm/                        # 共享内存实现
│   ├── arena.py                # 内存竞技场
│   ├── segment.py              # 内存段管理
│   └── condition_notifier.py   # 条件变量通知
├── message/
│   ├── message_info.py         # 消息元数据
│   └── history.py              # 消息历史
└── transport.py                # Transport 顶层抽象
```

---

## 5. 物理引擎与收敛

### 4.1 物理参数文件

```
channels/physics/
├── cell_registry.json          # 所有 cell 的注册信息
├── force_field.json            # 力场: {cell_id: {dx, dy, dz, push_from, push_mag}}
├── collision.json              # 碰撞检测结果
├── converged.json              # 全局收敛状态
├── edge_routes.json            # 边路由 (Spline 控制点)
├── z_layers.json               # Z 层排序
├── species_assignment.json     # Species → Cell 分配
├── bloom_variants.json         # Bloom 后处理变体
├── camera_at_params.json       # 摄像机 AT 参数
├── dof_at_params.json          # 景深 AT 参数
├── fog_at_params.json          # 雾效 AT 参数
└── scene_mesh_at_params.json   # 场景网格 AT 参数
```

### 4.2 收敛判定

```python
# 来自 rendering/constants.py
_ASTRO_BBOX_TOLERANCE     = 0.01   # bbox 变化检测阈值
_ASTRO_CELL_MAX_Z_LAYERS  = 10     # 最大 z 层数
_CROWDING_THRESHOLD       = 0.35   # 拥挤度阈值
_CROWDING_OPACITY_FLOOR   = 0.25   # 密集区域最低不透明度

# 来自 loop_orchestrator.py FAstroRendererConfig
max_epochs               = 10      # 最大迭代次数
convergence_threshold    = 0.5     # 全局收敛阈值
```

收敛条件: 所有 cell 的 bbox 变化量之和 < `convergence_threshold`。

---

## 6. Species 系统 — Cell 的视觉基因

每个 Cell 被分配一个 species，决定其视觉身份和参数范围：

| Species | 颜色 | F0 反射率 | 视觉语义 | icon 特征 |
|---------|------|----------|---------|----------|
| `cil-eye` | 靛蓝 #3F51B5 | 0.04 | 注意力/感知 | 径向光线 + 瞳孔中心 + 热力图 |
| `cil-bolt` | 琥珀 #FF6F00 | 0.80 | 激活函数 | 锯齿闪电路径 + 角状能量 |
| `cil-vector` | 森绿 #2E7D32 | 0.04 | 嵌入/投影 | 平行箭头 + 方向扩散 |
| `cil-plus` | 深红 #C62828 | 0.02 | 残差连接 | 十字交汇 + 虚线对角 |
| `cil-arrow-right` | 蓝灰 #455A64 | 0.06 | 数据流/输出 | 右向箭头 + 终端清晰度 |
| `cil-filter` | 紫罗兰 #7B1FA2 | 0.65 | 注意力掩码 | 3×3 网格 + 中心高亮 |
| `cil-code` | 绿 #2E7D32 | 0.04 | 函数变换 | 花括号 + 等宽精度 |
| `cil-layers` | 深蓝 #1565C0 | 0.08 | 堆叠表示 | 三层错位半透明矩形 |
| `cil-loop` | 琥珀 #F57F17 | 0.10 | 循环/反馈 | 圆弧箭头 + 周期能量 |
| `cil-graph` | 深蓝灰 #37474F | 0.03 | 图结构计算 | 小圆+连线 网络 |

Species SDF 着色器: `src/lib/shaders/cil-*.frag`
Species 风格探针: `channels/rendering/styleprobe/` (邻居色彩扩散)

---

## 7. Upstream 库在动态世界中的角色

### 6.1 渲染引擎 — 世界的画布

| 库 | 动态世界角色 |
|---|---|
| **pixijs-engine** (v8.19.0) | Cell 实时渲染引擎。每个 Cell 是一个 PixiJS Container，包含 Sprite (icon)、Text (label)、Graphics (边框)。Ticker 驱动 epoch 动画帧。 |
| **pixijs-filters** (v6.1.5) | 后处理特效管线：Bloom (光晕)、Glow (外发光)、Godray (光束)、Kawase-blur (散景)、DropShadow (投影)、Outline (描边)。 |
| **pixijs-ui** (v2.3.2) | Canvas 内 HUD：Slider 控制 epoch 速度、Button 触发收敛、ScrollBox 显示 cell 列表。 |
| **nanogl** | NanoBanana Bridge：AT 技术栈的 WebGL 封装层，处理 FBO/Program/Texture 低级调用。 |
| **ogl** | 极简 WebGL 参考实现，提供 Geometry/Mesh/Camera 类的设计模式参考。 |
| **lygia** | 400+ GLSL/WGSL 函数库：噪声 (Perlin/Simplex/Worley)、SDF 图元、色彩混合、PBR 光照。cell icon 渲染依赖 `sdf/circleSDF`、`generative/snoise` 等。 |

### 6.2 动画编排 — 世界的时间轴

| 库 | 动态世界角色 |
|---|---|
| **theatre-js** | Epoch 时间线可视化：`core/` 提供关键帧序列、`dataverse/` 提供 reactive 状态监听 (cell 参数变化实时反映)、`studio/` 提供可视化编辑面板。 |
| **animation-editor** | AT 节点式动画编辑器：Flow 图编辑 cell 间数据流、Graph 编辑器控制 species 参数随时间变化的曲线。 |

### 6.3 资产处理 — 世界的文字与纹理

| 库 | 动态世界角色 |
|---|---|
| **activetheory-svg2msdf** | SVG path → MSDF 纹理转换。Cell 上的文字标签 (如 "Multi-Head Attention") 不使用位图字体，而是转为 MSDF 距离场，实现任意缩放下的清晰渲染。 |
| **msdf-atlas-gen** / **msdfgen-source** | MSDF 字体图集生成管线的 C++ 后端。bin/msdfgen.linux 是编译好的二进制。 |
| **tiny-sdf** | 浏览器端轻量 SDF 生成，用于动态文字 (用户输入的自定义标签) 的实时距离场计算。 |

### 6.4 图论与网络可视化 — 世界的拓扑骨架

| 库 | 动态世界角色 |
|---|---|
| **cytoscape-js** | 生物网络图可视化引擎，提供自动布局算法 (COSE/dagre/klay) 作为 ELK 布局的备选方案。 |
| **graphology** | 图论算法库：最短路径、社区检测、中心性计算。用于 edge routing 优化和 cell 分组。 |
| **sigma-js** | 大规模网络渲染 (WebGL)：当 cell 数量超过 100+ 时切换到 sigma 的 GPU 加速渲染。 |
| **3d-force-graph** | 3D 力导向图：物理引擎的参考实现，cell 间斥力/引力模型源自此库的 d3-force-3d。 |
| **cosmos-gl** | WebGL 图渲染参考，提供节点/边的 instanced rendering 优化模式。 |

### 6.5 科学可视化 — 世界的深度渲染

| 库 | 动态世界角色 |
|---|---|
| **molstar** | 分子结构 WebGL 可视化。提供 representation 层次化渲染的设计模式 (cartoon/ball-and-stick/surface)，映射到 cell 的 species 视觉层次。 |
| **speck** | 原子球模型渲染。Cell 的 "cil-graph" species 参考了 speck 的球棍连线风格。 |
| **vtk-js** | 科学数据可视化框架。VolumneMapper 和 ContourFilter 的概念映射到 cell 的 z-layer 等值面渲染。 |
| **potree** | 点云渲染。大规模 cell 群组的 LOD 策略 (远景简化) 参考了 potree 的八叉树点云 LOD。 |

### 6.6 GPU 计算 — 世界的物理引擎

| 库 | 动态世界角色 |
|---|---|
| **webgl2-particles** | WebGL2 GPU transform feedback 粒子系统。cell 间的数据流动画 (粒子沿 edge 流动) 使用此技术。 |
| **gaussian-splats-at** | 3D 高斯泼溅渲染 + WASM 排序。cell 的光晕效果 (bloom) 参考了高斯泼溅的 alpha 混合方式。 |
| **regl-scatter2d** / **regl-scatterplot** | 大规模散点渲染。cell bbox 的碰撞检测可视化 debug 使用此库。 |

### 6.7 通信 — 世界的神经系统

| 库 | 动态世界角色 |
|---|---|
| **apollo-cyber** | **核心**。CyberRT pub/sub 通信框架的 Python 移植。channels/data/ 和 channels/transport/ 完整复现了 Apollo 的 ChannelBuffer → DataDispatcher → DataVisitor → AllLatest 数据流管线。 |
| **comlink** | Web Worker 透明 RPC。前端 pixi-cell-renderer 与 physics-worker 之间的通信使用 comlink 封装，避免手动 postMessage 序列化。 |

---

## 8. Unreal 渲染管线映射

channels/rendering/ 下的 27 个子模块移植自 Unreal Engine 4/5 的延迟渲染管线：

```
Unreal Engine                    动态世界
────────────                     ──────────
Actor / Component          →     Cell (claude-hk-config 调度的小弟)
Scene Graph                →     ELK Topology JSON (DAG)
BasePass → GBuffer         →     Constraint Collect → constraints.json
Lighting Pass              →     Constraint Solver (相对→绝对坐标)
Shadow Pass                →     z-layer 阴影 (shadow_port.py)
Nanite (LOD)               →     Cell 远景简化 (nanite/)
Lumen (Global Illumination)→     Species 色彩扩散 (lumen/)
PostProcess                →     SVG 后处理 (effects_port.py)
Final Composite            →     SVG Assembly (z-layer 排序合成)
Distance Field             →     MSDF 文字标签 (distancefield/)
Occlusion Culling          →     拥挤遮挡 (occlusion_core.py)
Path Tracing               →     Style Probe 邻居采样 (styleprobe/)
```

---

## 9. 默认拓扑：Transformer 架构

当前默认拓扑 (`channels/skeleton/topology.json`) 是一个 Transformer Encoder Block：

```
input_embed → pos_encode → self_attn → add_norm1 → ffn → add_norm2 → output
                   │                       ▲         │          ▲
                   └───── skip1 ───────────┘         └── skip2 ─┘
```

7 个 Cell + 6 条顺序 edge + 2 条 skip connection。ELK 布局算法: `layered`，方向: `DOWN`，边路由: `ORTHOGONAL`。

每个 Cell 目录结构：
```
channels/cell/{cell_id}/
├── params.json          # 骨架输入参数 (来自 topology_to_skeleton)
├── agent_params.json    # 小弟 Claude 推送的参数 (来自 /api/cell/publish)
├── status.json          # cell_agent dispatch 后的状态
├── bbox.json            # 当前 bbox
├── out.json             # 输出数据
├── svg.svg              # 该 cell 的 SVG 片段 (绝对静止模式: 主 Claude 直接生成，小弟不参与)
├── msdf.png             # MSDF 距离场纹理
└── msdf_preview.png     # MSDF 预览图
```

---

## 10. 最近开发进展

### Pub/Sub 管线修复

| Commit | 内容 |
|--------|------|
| `1b7fde0` | SubscriptionTable — Apollo 忠实的 channel pub/sub (非邻居广播) |
| `66412ba` | 修复 skip edge 导致的 writer/reader 重复注册 |
| `45cca83` | 全部 4 个测试通过 — pub/sub 端到端验证 |

### Upstream 技术研究 (#121-#130)

| # | 模块 | 动态世界用途 |
|---|------|------------|
| 121 | WebGL2 GPU transform feedback particles | cell 间数据流粒子动画 |
| 122 | WebGL2 particles clean standalone | 独立粒子系统参考 |
| 123 | AT Finding-Love-Shaders (PBR) | Gem/Sky/Terrain 材质 → species 外观 |
| 124 | 3D Gaussian Splats (WASM sort) | 光晕 alpha 混合参考 |
| 125 | AT svg2msdf | 文字标签 MSDF 核心管线 |
| 126 | AT activeframe (rAF + video sync) | epoch 动画帧同步 |
| 127 | AT split-text | 字符/词/行级别文字拆分动画 |
| 128 | AT modern-screenshot | DOM→Canvas→SVG 截图导出 |
| 129 | AT ios-silent-bypass | iOS 静音开关音频绕过 |
| 130 | AT balance-text + fit-text | 排版自动布局 → cell 标签自适应 |

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

# 单独运行 Cell Loop (dry_run 不需要 API key)
cd channels && python cell_agent.py --all --dry-run
python loop_orchestrator.py

# 配置 claude-hk 调度 (需要认证)
git clone https://github.com/dylanyunlon/claude-hk-config.git .claude-hk-config
# 编辑 .claude-hk-config/raw_curl.txt 填入 cookie
cd channels && python cell_agent.py --all  # 真实调度小弟
```

---

*本文档聚焦 cell-pubsub-loop 分支的动态世界 pipeline。*
*SVG 生成: 绝对静止模式下由主 Claude 直接生成（可参考 Species icon 风格），小弟 Claude 只负责 JSON 参数计算，绝不生成 SVG。*
