# Research 77 — Procedural Geometry Generation for WebGL

**作者:** 小弟 #77 (xiaodi)  
**日期:** 2026-06-15  
**分支:** cell-pubsub-loop

---

## 任务目标

为科研绘图找到合适的 WebGL 程序化几何体生成库，覆盖以下四类几何体：

| 几何体类型 | 用途 |
|-----------|------|
| 二十面体细分球 (geodesic sphere / icosphere) | 原子节点 |
| 圆柱 / 胶囊体 (cylinder / capsule) | 化学键 / 连接 |
| 箭头几何体 (arrow mesh) | 向量场 |
| 管道 / 丝带 (tube / ribbon) | 蛋白质骨架 |

---

## 搜索关键词

1. `procedural geometry generation webgl github`
2. `parametric surface mesh generation javascript`
3. `geodesic sphere icosahedron subdivision webgl javascript npm`
4. `primitive-geometry npm capsule arrow tube ribbon mesh webgl`

---

## 候选库调研

### 1. `primitive-geometry` ⭐ 首选综合库

- **npm:** `npm i primitive-geometry`
- **GitHub:** https://github.com/dmnsgn/primitive-geometry
- **版本:** 2.11.0 (活跃维护)
- **License:** MIT
- **大小:** ~30KB，零运行时依赖

**支持的几何体（src/ 目录）：**

```
annulus, box, capsule, circle, cone, cube, cylinder,
disc, ellipse, ellipsoid, icosahedron, icosphere,
plane, quad, reuleux, rounded-cube, rounded-rectangle,
sphere, squircle, stadium, superellipse, tetrahedron, torus
```

**输出格式（typed arrays，直接喂给 WebGL buffer）：**

```js
import Primitives from "primitive-geometry";

const geo = Primitives.capsule({ height: 0.5, radius: 0.25, nx: 16, roundSegments: 16 });
// geo.positions: Float32Array [x, y, z, ...]
// geo.normals:   Float32Array [x, y, z, ...]
// geo.uvs:       Float32Array [u, v, ...]
// geo.cells:     Uint8/16/32Array [a, b, c, ...]

const sphere = Primitives.icosphere({ radius: 0.5, subdivisions: 4 });
```

**关键特性：**
- 所有几何体默认包含 normals + UVs，开箱即用
- `cells` 直接对应 `gl.drawElements` 的 index buffer
- `icosphere` 支持最多 10 级细分（对应 10*4^10+2 ≈ 1049万顶点）
- `capsule` 支持独立控制圆柱段数 (ny) 和半球段数 (roundSegments)
- Cone 用 `cylinder` 将一端 radius 设为 0 实现

**化学键/原子的具体用法：**

```js
// 原子 → icosphere，细分 3-4 级，三角数合理
const atom = Primitives.icosphere({ radius: 0.5, subdivisions: 3 });

// 化学键 → capsule，两端半球保证平滑连接
const bond = Primitives.capsule({ height: 1.0, radius: 0.1, nx: 12, roundSegments: 8 });

// 向量箭头 = cylinder (杆) + cone (箭头) 组合
const shaft = Primitives.cylinder({ radiusTop: 0.05, radiusBottom: 0.05, height: 0.8 });
const head  = Primitives.cone({ radius: 0.12, height: 0.2, nx: 16 });
```

---

### 2. `icomesh` ⭐ 专用 icosphere 生成

- **npm:** `npm i icomesh`
- **GitHub:** https://github.com/mourner/icomesh
- **版本:** 1.1.0
- **作者:** Volodymyr Agafonkin (Leaflet 作者)
- **License:** ISC

**特点：**

```js
import icomesh from 'icomesh';

const { vertices, triangles } = icomesh(4);
// vertices: Float32Array [x,y,z, ...]  法线 = 顶点坐标（单位球）
// triangles: Uint16Array | Uint32Array

// 支持 UV 映射（等距矩形投影）
const { vertices, triangles, uv } = icomesh(4, true);
```

- 比 `primitive-geometry` 更轻量（单文件，< 5KB）
- 算法使用 Cantor 配对函数做中点缓存，内存高效
- 阶数 > 5 自动升级到 Uint32Array 索引
- **缺点：** 只生成 icosphere，不含 normals（但单位球 normal = position）

**选择建议：** 只需要原子球体、对包体积敏感时，优先用此库。

---

### 3. `gl-geometry` (stackgl 生态)

- **npm:** `npm i gl-geometry`
- **GitHub:** https://github.com/stackgl/gl-geometry

**特点：**
- stackgl 生态的 WebGL 几何体管理器
- 接受 simplicial complex 格式（`{ positions, cells }`）
- 自动管理 VBO binding，适合直接与 `gl-shader` 配合

```js
const geom = createGeometry(gl)
  .attr('position', icosphereMesh.positions)
  .attr('normal',   icosphereMesh.normals)
  .faces(icosphereMesh.cells);
```

**状态：** 维护较少，但仍可用。与 `primitive-geometry` 数据格式兼容。

---

### 4. 蛋白质骨架 Tube/Ribbon：`ribbon-geometry` + 手写 Frenet 框架

科研绘图中蛋白质骨架（Cα trace → ribbon）是特殊需求，现有 npm 包方案：

#### `ribbon-geometry` (Three.js 版)

```js
import generateRibbonGeometryClass from 'ribbon-geometry';
const RibbonGeometry = generateRibbonGeometryClass({ THREE });
// 沿 CurvePath 扫掠矩形截面
```

- 依赖 Three.js，适合已用 Three.js 的场景
- **与 raw WebGL 不兼容**

#### 推荐：手写 Frenet-Serret 管道扫掠（raw WebGL）

对于纯 WebGL（不依赖 Three.js）的管道几何：

```js
// 沿脊椎点数组生成管道
function buildTube(spinePoints, radius, segments) {
  // 1. 计算 Frenet 框架 (T, N, B) at each point
  // 2. 绕每个截面生成 segments 个顶点
  // 3. 连接相邻截面的三角形
}
```

参考：[Erkaman/gl-rock](https://github.com/Erkaman/gl-rock) 中的 sphere 操作展示了如何从 icosphere 出发变形网格。

---

### 5. 箭头几何体：组合方案

现有库中没有直接的 `arrow mesh` 生成，标准做法：

```
Arrow = Cylinder (shaft) + Cone (head)
```

用 `primitive-geometry` 实现：

```js
const shaft = Primitives.cylinder({
  height: arrowLength * 0.8,
  radiusTop: r, radiusBottom: r, nx: 16
});
const head = Primitives.cone({
  height: arrowLength * 0.2,
  radius: r * 2.5, nx: 16
});
// 然后手动合并两个 mesh，平移 head 到 shaft 顶端
function mergeMeshes(a, b, offsetB) { /* 合并 positions + cells */ }
```

---

## 综合推荐

| 需求 | 推荐方案 | npm 包 |
|------|---------|--------|
| 原子节点（精细球） | icosphere，细分 3-4 | `primitive-geometry` 或 `icomesh` |
| 化学键 | capsule，radius ≈ 键半径 | `primitive-geometry` |
| 向量场箭头 | cylinder + cone 合并 | `primitive-geometry` |
| 蛋白质骨架丝带 | Frenet 扫掠手写 | — (参考 gl-rock 思路) |
| VBO 管理 | simplicial complex → gl.bufferData | `gl-geometry` (可选) |

**最小依赖方案：** 只安装 `primitive-geometry`（单包覆盖除丝带外所有需求）+ 手写 ribbon builder。

---

## 本地克隆结构

```
upstream/
├── primitive-geometry/    # 综合几何库 v2.11.0
│   ├── src/               # 25个几何体源文件
│   ├── examples/          # 渲染示例
│   └── index.js           # 统一入口
├── icomesh/               # 专用 icosphere v1.1.0
│   └── index.js           # 单文件，< 5KB
├── gl-rock/               # 几何变形示例（参考用）
│   └── example/           # WebGL 集成示例
└── RESEARCH_77.md         # 本文件
```

---

## 下一步建议

1. **集成 `primitive-geometry`** 到 astro-svgfigure 的 WebGL pipeline
2. **原子渲染**：`icosphere(subdivisions=3)` + 实例化绘制 (instanced drawing)
3. **化学键渲染**：`capsule` + transform 矩阵（两原子坐标 → 位置 + 旋转）
4. **向量场**：批量生成箭头，合并到单个 VBO 减少 draw call
5. **蛋白质骨架**：实现 Frenet-Serret ribbon builder，输入 Cα 坐标数组

---

*Research by 小弟 #77 — xiaodi@astro.dev*
