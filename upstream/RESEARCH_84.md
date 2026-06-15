# Research #84 — 科研图注释/标注覆盖层 (Annotation & Label Overlay for Scientific Figures)

**研究员:** 小弟 #84 (xiaodi)  
**日期:** 2025-06-15  
**分支:** cell-pubsub-loop

---

## 需求摘要

科研绘图系统（cell 渲染层）需要：

1. **Cell 旁边的文字标注** — 每个 cell 对象显示一个 label
2. **Leader line（指引线）** — 连接标注文字到 cell，支持折线/曲线
3. **自动避让重叠** — 多标注时自动避开，不遮挡 cell 内容
4. **数学公式渲染** — LaTeX → WebGL，公式标注（如 `∇²ψ = 0`）

---

## 搜索结果综合

### 搜索 1: `webgl label annotation overlay scientific figure`

关键发现：
- **占用位图算法（Occupancy Bitmap）**：将 mark 光栅化到 bitmap，高效检测 label 摆放位置，避开已有 mark，性能优于传统算法。已集成到 Vega-Lite 的 `label encoding` channel。
- **HyperLabels**（3D 生物可视化）：label 除标注功能外还支持交互点击，适合层级结构探索。
- **外部标注（External Label）vs 内部标注（Internal Label）**：外部标注需指引线（leader line）和锚点；内部标注叠加在目标对象上。
- **AR 视图管理**：通过最小化遮挡来放置标注，适用于 3D 场景。

### 搜索 2: `text label billboard rendering webgl pixi.js`

关键发现：
- **PixiJS SDF 文字渲染**（Medium, 2020）：使用 Signed Distance Field 字体渲染，所有文字合并为单次 drawcall，放缩时平滑。适合大量 cell label 场景。
  - `PIXI.Text`：Canvas API 生成纹理，每个 sprite 独立纹理，无法 batch，内存占用高，**不适合大量 label**。
  - `PIXI.BitmapText`：预生成字形图集，支持 batch，性能和内存均优，**推荐用于 cell label**。
  - SDF shader 方案：一次 drawcall 渲染全部国家名等大量文字，动态缩放时调整 smoothing 参数。
- **PixiJS 7.x/8.x 官方文档**：Text 是 scene 对象，可 tint、rotate、alpha-blend，与 sprite 同等。

### 搜索 3: `annotation connector line leader scientific diagram javascript canvas`

关键发现：
- **leader-line** (npm)：纯 JS 库，DOM element 间连线，支持 `path: 'grid'`、`startSocket`/`endSocket`，可设置中间 label。**限于 DOM 层，不在 WebGL canvas 内部。**
- **jsPlumb**：StateMachine/Flowchart connector，支持障碍绕行、圆角、标注 overlay。**同为 DOM 层。**
- **GoJS**：商业库，完整交互图表。
- **regl / raw WebGL**：在 WebGL 内画线需要自行实现；用 `PIXI.Graphics` 可在 PixiJS 内画 leader line（Bezier/折线）。

---

## 候选方案评估

### A. KaTeX → Canvas → PixiJS Texture（推荐 ★★★★★）

**流程：**
```
LaTeX 字符串
  → katex.renderToString() → HTML/SVG string
  → 离屏 Canvas 绘制（canvas-latex 或手动）
  → PIXI.Texture.from(canvas)
  → PIXI.Sprite 挂到 cell container
```

**优点：**
- KaTeX 同步渲染，无依赖，MIT 开源，速度快
- `canvas-latex`（MIT）直接支持 KaTeX → Canvas 2D，并且明确支持 PixiJS
- 生成的 Sprite 可以 cache，公式不变时不重新渲染
- 与现有 PixiJS 栈完全兼容

**缺点：**
- 每个不同的公式需要独立纹理（GPU 内存）
- 动态修改公式需要重新生成纹理

**关键包：**
- `katex` (v0.16.x) — MIT — https://github.com/KaTeX/KaTeX
- `canvas-latex` — MIT — https://github.com/CurriculumAssociates/canvas-latex（支持 PixiJS）

---

### B. PixiJS BitmapText + PIXI.Graphics leader line（推荐 ★★★★☆）

**流程：**
```
cell 位置 → 计算 label offset（避让算法）
  → PIXI.BitmapText（高性能，batch）
  → PIXI.Graphics 画折线（leader line）
  → 两者组成 LabelGroup container
```

**优点：**
- 全部在 WebGL 内，无 DOM overhead
- BitmapText 支持 batch，大量 label 性能最优
- PIXI.Graphics 原生支持 bezier、折线、圆角
- 已有现成 `msdfgen` 工具（repo 里 `bin/msdfgen.linux`），可生成 MSDF 字体图集

**缺点：**
- 不支持 LaTeX（需配合方案 A 的纹理）
- 需要自行实现避让算法

---

### C. DOM Overlay（CSS/HTML label + leader-line.js）（备选 ★★★☆☆）

**流程：**
```
WebGL canvas（PixiJS）+ 上层透明 HTML div
  → 每个 cell 对应一个 div.label（绝对定位，transform 跟随 cell 世界坐标）
  → leader-line.js 在两个 DOM 元素间画连接线（SVG overlay）
  → KaTeX 直接渲染到 div（katex.render()）
```

**优点：**
- LaTeX 渲染最简单（KaTeX 直接 DOM）
- leader-line 库开箱即用
- 可使用 CSS 样式，支持 HTML 富文本

**缺点：**
- DOM 操作随 cell 数量增多性能下降（100+ cell 明显卡顿）
- WebGL 坐标系与 DOM 坐标系同步复杂（需监听 viewport 变化、zoom/pan）
- 无法受益于 PixiJS culling、layer 管理

---

### D. MathJax + Canvas（不推荐 ✗）

- MathJax 3.x 异步渲染，有页面 reflow
- 体积比 KaTeX 大很多（~1MB vs ~280KB）
- 对已有 PixiJS 栈集成复杂
- **结论：不推荐，KaTeX 在速度和体积上更优**

---

### E. regl-scatter2d（不适用 ✗）

- 专为散点图 label 设计，不适合 cell-based 图
- 不支持 leader line 或自定义 label 形状
- **结论：与当前渲染架构不匹配**

---

## 自动避让算法

Label 避让是关键问题，推荐分级实现：

### 级别 1 — 固定偏移（快速，适合初期）
```
label_pos = cell_center + normalize(cell_center - canvas_center) * offset
```
向外径向偏移，简单但可能重叠。

### 级别 2 — 占用位图（Vega-Lite 方案，推荐）
```
1. 将所有 cell 光栅化到 occupancy bitmap（低分辨率，如 1/4 屏幕）
2. 对每个 cell 的 label，在候选位置集合 [右、左、上、下、右上…] 中
   找第一个不与 bitmap 已占用区域重叠的位置
3. 放置 label，更新 bitmap
```
**优点：** 快速，O(N * candidates)，无碰撞检测 BVH 开销。

### 级别 3 — 力导向排斥（动态场景）
- 把 label 作为粒子，cell 和其他 label 产生排斥力
- 适合动画/交互场景，但计算代价较高
- 可仅在 layout 稳定后运行一次

---

## Leader Line 实现（PixiJS 内）

```typescript
// channels/rendering/ 新增 LeaderLine.ts
import { Graphics } from 'pixi.js';

export function drawLeaderLine(
  g: Graphics,
  cellPos: {x: number, y: number},
  labelPos: {x: number, y: number},
  style = { color: 0x888888, width: 1, alpha: 0.6 }
) {
  // 计算肘点（elbow）：先水平，再垂直
  const elbowX = labelPos.x;
  const elbowY = cellPos.y;
  
  g.lineStyle(style.width, style.color, style.alpha);
  g.moveTo(cellPos.x, cellPos.y);
  g.lineTo(elbowX, elbowY);
  g.lineTo(labelPos.x, labelPos.y);
  
  // 箭头小圆点
  g.beginFill(style.color, style.alpha);
  g.drawCircle(cellPos.x, cellPos.y, 2);
  g.endFill();
}
```

---

## LaTeX → WebGL 集成方案（canvas-latex + PixiJS）

```typescript
// 安装：npm install canvas-latex katex
import CanvasLatex from 'canvas-latex';
import * as PIXI from 'pixi.js';

function createLatexSprite(latex: string, fontSize = 16): PIXI.Sprite {
  const offscreen = document.createElement('canvas');
  offscreen.width = 256;
  offscreen.height = 64;
  const ctx = offscreen.getContext('2d')!;
  
  const cl = new CanvasLatex(latex, ctx, { fontSize, displayMode: false });
  cl.render(0, 0);
  
  const tex = PIXI.Texture.from(offscreen);
  return new PIXI.Sprite(tex);
}
```

**注意：** canvas-latex 需要 KaTeX 字体可用（引入 KaTeX CSS）。对于 WebGL 环境无 DOM，可通过 node-canvas 在 server 端预渲染，返回 PNG/base64 纹理。

---

## 与现有架构的对接点

```
channels/rendering/          ← 现有渲染系统
  ├── annotation/            ← 新增目录
  │   ├── LabelLayer.ts      ← PIXI.Container 管理所有 label
  │   ├── LeaderLine.ts      ← PIXI.Graphics leader line
  │   ├── LatexSprite.ts     ← canvas-latex → PIXI.Sprite
  │   ├── OccupancyBitmap.ts ← 避让算法
  │   └── index.ts
```

**PubSub 集成：**
- 订阅 `cell:layout_updated` → 触发 label 重新布局
- 订阅 `cell:selected` → 高亮对应 label
- 发布 `annotation:layout_done` → 通知其他 channel

---

## 推荐实施顺序

| 优先级 | 任务 | 预计工作量 |
|--------|------|------------|
| P0 | BitmapText label（无公式）+ 固定偏移 | 0.5 天 |
| P1 | PIXI.Graphics leader line | 0.5 天 |
| P2 | 占用位图避让算法 | 1 天 |
| P3 | canvas-latex 集成（LaTeX 公式） | 1 天 |
| P4 | MSDF 字体（利用已有 msdfgen.linux） | 1 天 |

---

## 参考资源

- KaTeX: https://github.com/KaTeX/KaTeX (MIT)
- canvas-latex (KaTeX → Canvas, PixiJS 支持): https://github.com/CurriculumAssociates/canvas-latex (MIT)
- RaTeX (Rust/WASM, >99.5% KaTeX 覆盖): https://github.com/erweixin/RaTeX
- PixiJS SDF 文字渲染: https://medium.com/@clashofcoins/implementing-sdf-text-rendering-in-pixi-js-3cf78614071d
- Vega-Lite label encoding (占用位图算法): https://vega.github.io/vega-lite/docs/mark.html
- leader-line.js (DOM层): https://github.com/anseki/leader-line
- 科研图 label 分类论文: https://www.researchgate.net/figure/Label-classes-annotation-boxes-external-and-internal-labels_fig1_220818918
