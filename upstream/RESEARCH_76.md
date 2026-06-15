# 科研绘图 WebGL 库调研报告 — 小弟 #76 (xiaodi)

> 研究方向：点精灵(Point Sprites)、二十面体球(Icosahedron)、分子可视化(Molecular Viz)、科研散点图(Scientific Scatter)

---

## 已克隆仓库

| 仓库 | 方向 | Stars | 路径 |
|------|------|-------|------|
| [nglviewer/ngl](https://github.com/nglviewer/ngl) | 分子/蛋白质 WebGL 可视化 | ~1.6k | `upstream/ngl/` |
| [flekschas/regl-scatterplot](https://github.com/flekschas/regl-scatterplot) | GPU 加速散点图 | 233 | `upstream/regl-scatterplot/` |
| [gl-vis/regl-scatter2d](https://github.com/gl-vis/regl-scatter2d) | 轻量 2D 散点 + 自定义标记 | ~150 | `upstream/regl-scatter2d/` |
| [potree/potree](https://github.com/potree/potree) | 大规模点云 WebGL 渲染 | 5504 | `upstream/potree/` |

---

## 1. NGL Viewer (`ngl/`)

**核心价值**：最完整的科研分子可视化库，支持蛋白质、DNA、RNA、小分子。

### 关键技术发现

#### 二十面体球 (`src/buffer/icosahedron-buffer.ts`)
```typescript
// 基于 THREE.IcosahedronGeometry 实例化渲染
class IcosahedronBuffer extends GeometryBuffer {
  constructor(data, params) {
    super(data, params, new IcosahedronGeometry(1, 0))  // level=0 即原始20面体
  }
  // 支持 heightAxis + depthAxis 控制朝向，size 控制大小
}
```
- `sphereDetail` 参数控制二十面体细分层数（0=20面体原形，1~3=逐渐趋近球体）
- `EllipsoidBuffer` 也继承自 IcosahedronGeometry（细分=2），用于各向异性原子渲染
- `SphereGeometryBuffer` 用 `IcosahedronGeometry(1, 1)` 渲染标准原子球

#### 点精灵 (`src/shader/SphereImpostor.vert`)
- 实现了 **SphereImpostor** 技术：用一个 billboard quad + 光线投射模拟完美球体
- 比真实几何球体性能高10倍以上，适合渲染数百万原子
- 相关文件：`SphereImpostor.vert`, `SphereImpostor.frag`

#### 文件结构亮点
```
ngl/src/
├── buffer/
│   ├── icosahedron-buffer.ts   ← 二十面体实例化
│   ├── ellipsoid-buffer.ts     ← 椭球体（各向异性原子）
│   ├── sphere-impostor-buffer.ts ← 点精灵球体
│   ├── point-buffer.ts         ← 点云渲染
│   └── geometry-buffer.ts      ← GPU instanced 基类
├── shader/
│   ├── SphereImpostor.vert/frag ← 核心点精灵着色器
│   └── Point.vert/frag          ← 基础点渲染
└── representation/
    ├── ballandstick-representation.ts  ← 球棍模型
    └── surface-representation.ts       ← 分子表面
```

#### 支持格式
PDB, CIF, mmCIF, SDF, MOL2, MMTF (二进制压缩，支持百万原子)

---

## 2. regl-scatterplot (`regl-scatterplot/`)

**核心价值**：超高性能科研散点图，支持 2000万+ 点，带 lasso 选择、缩放、颜色编码。

### 关键技术发现

#### GPU 着色器 (`src/point.vs`)
```glsl
// 点数据编码进 RGBA texture 极大提升 GPU 传输效率
uniform sampler2D stateTex;   // 点状态（x,y,colorVal,sizeVal）
uniform sampler2D colorTex;   // 颜色映射表
// 支持 isColoredByZ/W、isOpacityByDensity 等科研常用编码
```

#### 科研特性
- **密度透明度** (`isOpacityByDensity`)：高密度区域自动降透明，避免过绘制
- **KD-tree hover 加速**：百万点鼠标交互不卡顿
- **连线渲染**：点之间可绘制轨迹线（如单细胞轨迹分析）
- **多实例共享上下文**：多个散点图共享 WebGL 上下文
- **Python widget**：`jscatter` 包可在 Jupyter 中使用

#### 文件结构
```
regl-scatterplot/src/
├── point.vs / point.fs      ← 核心点着色器
├── point-update.vs/.fs      ← 点更新着色器（GPU 内更新状态）
├── renderer.js              ← 主渲染器
├── lasso-manager/           ← lasso 选择工具
├── kdbush.js                ← KD-tree hover 加速
└── spline-curve.js          ← 轨迹曲线渲染
```

---

## 3. regl-scatter2d (`regl-scatter2d/`)

**核心价值**：轻量级自定义标记散点，支持 SDF 标记形状（圆、方、三角等）。

### 关键技术发现

#### GLSL 圆形点精灵 (`circle-frag.glsl`)
```glsl
// 通过 SDF (Signed Distance Field) 实现精确点精灵形状
// 支持 antialiasing 边缘
```

#### 使用方式
```js
scatter({
  positions: [x,y, x,y, ...],
  color: 'rgba(0,100,200,.75)',
  marker: customSdfTexture,   // 自定义形状！
  size: [...],
  opacity: 0.8
})
```

- 与 Plotly.js 的 `scattergl` trace 集成
- 支持多组点同时渲染，各组可用不同 marker

---

## 4. Potree (`potree/`)

**核心价值**：专业大规模点云（LiDAR/扫描）WebGL 渲染器，5500+ stars。

### 关键技术
- **Out-of-core Octree**：按视锥体动态加载点，内存高效
- 支持 EPT、LAZ、COPC 等专业点云格式
- 着色方式：高程色、RGB、强度、分类等
- 测量工具：距离、面积、体积、截面

### 与科研散点图的关系
Potree 针对 LiDAR 地理点云，科研散点图（如 UMAP/t-SNE）更适合用 regl-scatterplot。

---

## 关键技术总结

### 点精灵（Point Sprites）实现对比

| 库 | 方法 | 适合场景 |
|---|---|---|
| NGL | SphereImpostor (billboard+raycast) | 分子球体，视觉完美 |
| regl-scatterplot | `gl_PointSize` + texture | 百万级 2D 散点 |
| regl-scatter2d | SDF marker shader | 自定义形状散点 |
| Potree | Adaptive point size shader | LiDAR 点云 |

### 二十面体球（Icosahedron）在 NGL 中的使用

```
细分层数  面数   效果
  0      20    二十面体原形（低多边形风格）
  1      80    粗球体
  2      320   标准原子球（NGL ellipsoid 默认）
  3      1280  高质量球体
```

EllipsoidBuffer (细分=2) → 蛋白质原子各向异性热椭球
IcosahedronBuffer (细分=0) → 点状原子符号、低多边形科研图

---

## 推荐使用场景

| 需求 | 推荐库 |
|------|--------|
| 蛋白质/分子3D可视化 | **NGL Viewer** |
| UMAP/t-SNE/PCA 散点图 | **regl-scatterplot** |
| 自定义标记形状散点 | **regl-scatter2d** |
| LiDAR/3D扫描点云 | **Potree** |
| 二十面体/球体实例化渲染 | **NGL IcosahedronBuffer** |

---

*研究员：小弟 #76 (xiaodi) | 日期：2026-06-15*
