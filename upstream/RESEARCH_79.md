# 研究报告 #79 — 图/网络布局的 WebGL 渲染

**研究员:** 小弟 #79 (xiaodi)  
**日期:** 2025-06-15  
**主题:** 大规模网络图的 GPU 力导向布局 + WebGL 渲染技术选型

---

## 一、研究目标

科研绘图场景：
- **大规模网络图**（1000–100 万节点）的 GPU 并行力导向布局
- **节点渲染**：球体 / 二十面体（icosahedron）
- **边渲染**：管道（tube/cylinder）或曲线
- 交互：缩放、平移、悬停、点击高亮

---

## 二、候选库对比

### 1. `cosmos.gl` (cosmosgl/graph) ⭐ **最推荐**

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/cosmosgl/graph |
| 安装 | `npm install @cosmos.gl/graph` |
| 渲染后端 | WebGL 2 (luma.gl) |
| 力导向 GPU | ✅ 全 GPU，fragment/vertex shader 并行计算 |
| 规模 | **数十万节点 + 百万边**，实时模拟 |
| 节点形状 | 圆点（fragment shader 绘制），支持自定义形状 (`shapes.stories.ts`) |
| 边渲染 | 曲线（conic-curve），Lines 模块，GLSL shader |
| 力系统 | ForceManyBody / ForceLink / ForceGravity / ForceCenter / Clusters |
| API | `setPointPositions()`, `setLinks()`, `Float32Array` 直接操控 |
| 框架 | 无锁定，纯 TypeScript |

**GPU 力导向原理（源码验证）：**
```
upstream/cosmos-gl/src/modules/
├── ForceManyBody/           # Barnes-Hut 树 in fragment shader
│   ├── force-level.frag     # 层级排斥力计算 (GLSL 300 es)
│   └── calculate-level.frag # 质心层级聚合
├── ForceLink/               # 弹簧引力 in GPU
├── Points/                  # 位置/速度存 Framebuffer，GPU ping-pong
│   ├── draw-points.frag/.vert
│   └── update-position.frag # 每帧 GPU 积分
└── Lines/                   # 曲线边
    ├── draw-curve-line.frag/.vert
    └── conic-curve-module.ts
```

节点位置用 **Texture/Framebuffer ping-pong** 存储（`currentPositionFbo` / `previousPositionFbo` / `velocityFbo`），每帧全 GPU 更新，不回传 CPU。

**使用示例：**
```typescript
import { Graph } from '@cosmos.gl/graph'

const graph = new Graph(divElement, {
  spaceSize: 4096,
  simulationRepulsion: 0.5,
  simulationFriction: 0.1,
  curvedLinks: true,
  fitViewOnInit: true,
})

const N = 5000
const positions = new Float32Array(N * 2)  // x,y per node
const colors = new Float32Array(N * 4)     // rgba per node
const links = new Float32Array(edgeCount * 2)

graph.setPointPositions(positions)
graph.setPointColors(colors)
graph.setLinks(links)
graph.start()
```

---

### 2. `vasturiano/3d-force-graph` ⭐ **3D 球体+管道场景首选**

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/vasturiano/3d-force-graph |
| 安装 | `npm install 3d-force-graph` |
| 渲染后端 | Three.js (WebGL) |
| 力导向 GPU | ❌ CPU (d3-force-3d or ngraph) |
| 规模 | ~1000–5000 节点交互流畅 |
| **节点形状** | ✅ **默认球体 (SphereGeometry)**，支持自定义 Three.js Object3D（二十面体等） |
| **边渲染** | ✅ **TubeGeometry / 自定义管道**，支持 `linkWidth`、`linkResolution` |
| 3D 支持 | ✅ 完整 3D，轨道相机，粒子动画 |
| API | 声明式，链式调用 |

**节点 = 二十面体示例：**
```javascript
import ForceGraph3D from '3d-force-graph'
import * as THREE from 'three'

ForceGraph3D()(document.getElementById('graph'))
  .graphData(data)
  // 节点：二十面体
  .nodeThreeObject(node => {
    const geo = new THREE.IcosahedronGeometry(5, 1)
    const mat = new THREE.MeshPhongMaterial({ color: node.color })
    return new THREE.Mesh(geo, mat)
  })
  // 边：管道
  .linkWidth(2)
  .linkResolution(8)          // 管道截面分段数
  .forceEngine('d3')          // 或 'ngraph'
  .numDimensions(3)
```

**源码结构（`upstream/3d-force-graph/src/`）：**
```
src/
├── 3d-force-graph.js   # 主组件，封装 ThreeForceGraph + 相机控制
├── index.js            # 导出
└── kapsule-link.js     # 属性绑定
```
依赖 `three-forcegraph`（力导向核心）+ `three-render-objects`（Three.js 场景管理）。

---

### 3. `sigma.js v4` + `graphology`

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/jacomyal/sigma.js |
| 安装 | `npm install sigma graphology` |
| 渲染后端 | WebGL (自定义 GLSL shader) |
| 力导向 GPU | ❌ 布局在 Web Worker / CPU；渲染 GPU |
| 规模 | ✅ 万级节点渲染流畅（9000节点 AI论文网络 demo） |
| 节点形状 | 圆形（WebGL），可扩展为自定义 program（border/image/piechart/square） |
| 边渲染 | 直线或曲线（@sigma/edge-curve），**无 3D 管道** |
| 布局插件 | `graphology-layout-forceatlas2`（Web Worker 并行） |
| 框架 | React/Vue/Angular 均可 (`@react-sigma/core`) |

**Monorepo 结构（`upstream/sigma-js/packages/`）：**
```
packages/
├── sigma/              # 核心 WebGL 渲染器
│   ├── src/            # Renderer, Camera, MouseCaptor
│   └── rendering/      # Node/Edge WebGL programs
├── edge-curve/         # 曲线边扩展
├── node-border/        # 带边框节点
├── node-image/         # 图标节点
├── layer-webgl/        # 自定义 WebGL layer
└── storybook/stories/  # 交互示例
```

**科研绘图最佳实践：**
```typescript
import Graph from 'graphology'
import { Sigma } from 'sigma'
import forceAtlas2 from 'graphology-layout-forceatlas2'

const graph = new Graph()
// 加载节点、边...

// CPU 布局（Worker 中运行）
forceAtlas2.assign(graph, { iterations: 50, settings: { gravity: 1 } })

// WebGL 渲染
const sigma = new Sigma(graph, container, {
  renderEdgeLabels: true,
  defaultNodeType: 'circle',
})
```

---

### 4. `ParaGraphL` — 历史参考

- GPU Fruchterman-Reingold，作为 sigma.js 插件实现
- 用 WebGL fragment shader 做 GPGPU 布局计算
- 方法验证了 GPU 力导向可行性，但不再积极维护
- 论文/架构思路可参考：https://nblintao.github.io/ParaGraphL/

---

### 5. `ngraph.forcelayout`

```
npm install ngraph.forcelayout ngraph.graph
```

- CPU 力导向，但速度优化较好（TypedArray、缓存友好）
- 可作为 3d-force-graph 的 `forceEngine: 'ngraph'` 后端
- 不适合 10 万+ 节点

---

## 三、选型矩阵

| 需求 | cosmos.gl | 3d-force-graph | sigma.js |
|------|-----------|----------------|----------|
| GPU 力导向布局 | ✅ 完整 GPU | ❌ CPU | ❌ CPU Worker |
| 1000+ 节点 | ✅ (10万+) | ✅ (~5K) | ✅ (~1万) |
| 球体节点 | ⚠️ 圆点 | ✅ SphereGeometry | ⚠️ 圆形 |
| 二十面体节点 | 需自定义 shader | ✅ 直接 Three.js | 需扩展 |
| 管道边 | ⚠️ 曲线 | ✅ TubeGeometry | ❌ |
| 3D 场景 | ❌ 2D | ✅ | ❌ |
| React 集成 | 简单 | 简单 | ✅ @react-sigma |
| 最大节点数 | **100万+** | ~5000 | ~10000 |

---

## 四、推荐架构（科研绘图）

### 方案 A：2D 超大规模（>1万节点）
```
cosmos.gl (GPU layout + render)
  + graphology (数据结构)
  + 自定义 WebGL shader for sphere-like nodes
```

### 方案 B：3D 中等规模（~1000-5000节点，球体+管道）⭐
```
3d-force-graph (Three.js)
  + IcosahedronGeometry (节点)
  + TubeGeometry / linkWidth (边)
  + ngraph 布局引擎（更快）
```

### 方案 C：2D 中等规模 + React（~万级）
```
sigma.js v4 + graphology
  + @react-sigma/core
  + graphology-layout-forceatlas2 (Web Worker)
```

---

## 五、快速集成代码模板

### 3D 科研图（球体节点 + 管道边）

```javascript
import ForceGraph3D from '3d-force-graph'
import * as THREE from 'three'
import SpriteText from 'three-spritetext'

const Graph = ForceGraph3D()
Graph(document.getElementById('3d-graph'))
  .backgroundColor('#000011')
  .nodeLabel('id')
  .nodeThreeObject(node => {
    // 二十面体
    const geo = new THREE.IcosahedronGeometry(node.val || 4, 2)
    const mat = new THREE.MeshPhongMaterial({
      color: node.color || '#ffffff',
      shininess: 100,
      transparent: true,
      opacity: 0.85,
    })
    const mesh = new THREE.Mesh(geo, mat)
    return mesh
  })
  .linkWidth(d => d.weight || 1)
  .linkOpacity(0.5)
  .linkResolution(6)              // 管道截面 6 边形
  .linkDirectionalParticles(2)    // 方向粒子动画
  .forceEngine('ngraph')
  .graphData(await fetch('/graph.json').then(r => r.json()))
```

### GPU 布局（cosmos.gl，超大规模）

```typescript
import { Graph } from '@cosmos.gl/graph'

const graph = new Graph(container, {
  spaceSize: 8192,
  simulationRepulsion: 0.3,
  simulationLinkSpring: 0.5,
  simulationFriction: 0.1,
  simulationGravity: 0.1,
  curvedLinks: true,
  fitViewOnInit: true,
  onClick: (pointIndex) => console.log('node:', pointIndex),
})

// Float32Array 直接操作，零拷贝
const N = 50000
const positions = new Float32Array(N * 2)
const colors = new Float32Array(N * 4).fill(1)  // RGBA
graph.setPointPositions(positions)
graph.setPointColors(colors)
graph.setPointSizes(new Float32Array(N).fill(3))
graph.setLinks(linkBuffer)  // [src0, tgt0, src1, tgt1, ...]
graph.start()
```

---

## 六、参考资源

| 资源 | URL |
|------|-----|
| cosmos.gl 文档 | https://cosmosgl.github.io/graph |
| 3d-force-graph examples | https://github.com/vasturiano/3d-force-graph/tree/master/example |
| sigma.js v4 | https://v4.sigmajs.org/ |
| graphology 标准库 | https://graphology.github.io/ |
| ParaGraphL 论文/演示 | https://nblintao.github.io/ParaGraphL/ |
| Cosmograph（cosmos.gl 驱动的商业产品） | https://cosmograph.app |
| 百万节点案例 | https://nightingaledvs.com/how-to-visualize-a-graph-with-a-million-nodes/ |

---

## 七、结论

**最终推荐：**

1. **3D 科研图（球体/二十面体节点 + 管道边）→ `3d-force-graph`**  
   完整 Three.js 场景，nodeThreeObject 可直接用 IcosahedronGeometry，linkWidth 自动生成 TubeGeometry，开箱即用，~5000 节点流畅。

2. **超大规模（>1万节点）→ `cosmos.gl`**  
   唯一真正全 GPU 力导向（布局+渲染均在 GPU shader），可支持百万节点实时仿真，节点/边通过 Float32Array 直接输入 GPU，性能无敌。

3. **React 生态 + 中型图 → `sigma.js + graphology`**  
   生态最成熟，@react-sigma/core 集成最方便，ForceAtlas2 Web Worker 布局，万级节点 WebGL 渲染流畅。
