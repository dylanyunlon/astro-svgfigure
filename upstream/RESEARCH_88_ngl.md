# RESEARCH #88 — NGL Viewer: High-Performance Molecular Graphics WebGL Library

**Upstream:** https://github.com/nglviewer/ngl  
**Researcher:** xiaodi #88  
**Branch:** cell-pubsub-loop  

---

## 概览 Overview

NGL Viewer 是一个高性能分子图形库，基于 WebGL 渲染百万原子级别的分子复合物，支持桌面和移动端交互。广泛用于 PDB、DrugBank 等生物数据库的在线分子可视化。

**核心技术亮点：**
- Ray-casting impostor 技术：用四边形 + GPU 光线追踪模拟球体/圆柱体，大幅减少几何顶点数
- 支持 PBR（物理渲染）材质模型（roughness / metalness）
- TypeScript 编写，基于 Three.js WebGL 上层封装
- 支持多种分子表示法：ball-and-stick、spacefill、cartoon、ribbon、hyperball、surface 等

---

## 源码结构 Source Structure

```
src/
├── buffer/         # GPU缓冲区：各种几何体的WebGL Buffer封装
├── chemistry/      # 化学交互、官能团识别、价键模型
├── color/          # 颜色映射系统（按元素/链/B因子/静电势等）
├── component/      # 组件系统（分子组件、形状组件等）
├── controls/       # 用户交互控制（旋转、缩放、鼠标事件）
├── geometry/       # 几何生成（样条、螺旋、KD树、空间哈希）
├── parser/         # 分子文件解析器（PDB、mmCIF、SDF、MOL2等）
├── representation/ # 表示法系统（球棍、cartoon、空间填充等）
├── shader/         # GLSL着色器（vert/frag + chunk片段）
├── stage/          # 舞台/场景管理
├── structure/      # 分子结构数据模型
├── viewer/         # WebGL渲染器核心（1409行）
└── worker/         # Web Worker异步计算
```

---

## 关键模块深度分析

### 1. `src/geometry/` — 几何生成

| 文件 | 功能 |
|------|------|
| `primitive.ts` | 几何图元基类（Sphere、Cylinder、Arrow等），抽象接口 |
| `spline.ts` | 样条曲线生成，用于 cartoon/ribbon 骨架插值 |
| `helixorient.ts` | α螺旋轴向计算，用于 helix 表示法 |
| `helixbundle.ts` | 螺旋束聚合 |
| `kdtree.ts` | KD树空间索引，加速原子邻域查询 |
| `spatial-hash.ts` | 空间哈希，O(1) 近邻搜索 |
| `grid.ts` | 体素网格（用于电子密度、表面等值面） |
| `shape.ts` | 用户自定义形状容器 |
| `dash.ts` | 虚线几何生成（用于距离测量显示） |

**核心机制：** Spline + HelixOrient 联合生成 cartoon 骨架曲线；KD树/空间哈希支撑百万原子的快速空间查询。

---

### 2. `src/viewer/` — WebGL 渲染器

**核心文件：** `viewer.ts`（1409行）

```
viewer/
├── viewer.ts           # 渲染器主体：场景管理、渲染循环、相机控制
├── tiled-renderer.ts   # 分块高分辨率渲染（用于导出大图）
├── geometry-group.ts   # 几何体分组管理
├── gl-utils.ts         # WebGL工具函数
├── stats.ts            # 渲染性能统计（FPS、draw calls）
├── viewer-constants.ts # 渲染常量
└── viewer-utils.ts     # 视图矩阵、相机工具
```

**渲染器特性：**
- 基于 Three.js WebGLRenderer，自定义 render loop
- `TiledRenderer`：将大画面分块渲染后拼接，支持导出任意分辨率图片（用于发表）
- 内置 picking buffer：通过颜色编码实现原子/键的鼠标拾取（objectId uniform）
- 支持 fog、ortho/perspective 切换、interior color（内部面着色）

---

### 3. `src/representation/` — 表示法系统

每个表示法是独立类，继承自 `base-representation.ts → structure-representation.ts`：

| 表示法文件 | 可视化方式 | 技术 |
|-----------|-----------|------|
| `spacefill-representation.ts` | 空间填充（VDW球） | SphereImpostor shader |
| `ballandstick-representation.ts` | 球棍模型 | Sphere + CylinderImpostor |
| `licorice-representation.ts` | 甘草棍（无球） | CylinderImpostor |
| `cartoon-representation.ts` | 卡通（二级结构） | Spline + TubeMesh/Ribbon |
| `ribbon-representation.ts` | 彩带 | Ribbon Buffer |
| `surface-representation.ts` | 溶剂可及面 | MolecularSurface（marching cubes） |
| `hyperball-representation.ts` | 超球（原子间融合） | HyperballStickImpostor |
| `dot-representation.ts` | 点云 | Point Buffer |
| `label-representation.ts` | 原子标签 | SDFFont（有向距离场字体） |
| `contact-representation.ts` | 非共价接触（氢键等） | Chemistry interactions |
| `rocket-representation.ts` | 螺旋轴圆柱 | HelixBundle + Cylinder |

---

### 4. `src/shader/` — GLSL 着色器

**着色器列表：**

```
SphereImpostor.{vert,frag}       # 球体Ray-casting核心
CylinderImpostor.{vert,frag}     # 圆柱体Ray-casting
HyperballStickImpostor.{vert,frag} # 超球融合
Mesh.{vert,frag}                 # 通用网格（cartoon/surface）
Ribbon.vert                      # 彩带顶点
SDFFont.{vert,frag}             # 有向距离场文字渲染
Point.{vert,frag}               # 点云
Image.{vert,frag}               # 图像纹理
WideLine.{vert,frag}            # 宽线（键/轮廓）
BasicLine.{vert,frag}           # 基础线
Quad.{vert,frag}                # 全屏四边形（后处理）

chunk/                           # 可复用GLSL片段
├── fog_fragment.glsl
├── interior_fragment.glsl       # 内部面处理
├── nearclip_fragment.glsl       # 近裁剪面（分子截面）
├── radiusclip_{fragment,vertex}.glsl  # 半径裁剪
├── matrix_scale.glsl
└── unpack_color.glsl            # 颜色解包（picking用）
```

**SphereImpostor 技术核心（分析 `.frag`）：**

```glsl
// 输入：4顶点四边形 + 中心位置/半径
// 在fragment shader中进行光线-球体相交测试
float calcDepth(in vec3 cameraPos) {
    vec2 clipZW = cameraPos.z * projectionMatrix[2].zw + projectionMatrix[3].zw;
    return 0.5 + 0.5 * clipZW.x / clipZW.y;
}
```

- 每个原子只需 **4个顶点**（两个三角形），而非几何球体的数百顶点
- GPU fragment shader 实时计算光线与球的精确交点
- 计算精确法线 → PBR光照（roughness/metalness uniforms）
- 支持 interior color（切面着色，显示分子内部）
- `#ifdef PICKING` 分支：切换到 objectId 颜色模式，支持鼠标拾取原子

---

## 架构关键洞察 Key Insights

### Impostor 渲染模式 vs 几何体模式

NGL 同时提供两套实现：

| 模式 | 代表文件 | 顶点数/球 | 质量 | 性能 |
|------|---------|----------|------|------|
| **Impostor** (GPU ray-cast) | `sphereimpostor-buffer.ts` | 4 | 像素级精确 | 极高（推荐） |
| **Geometry** (CPU mesh) | `spheregeometry-buffer.ts` | ~200 | 近似 | 一般 |

百万原子场景下，Impostor 模式可将顶点数从数亿降至数百万。

### 颜色系统

`src/color/` 有20+种颜色映射器，全部注册到 `ColormakerRegistry`：
- 按元素（CPK颜色）、链ID、残基序列、B因子、静电势、残基类型等
- 插件式架构，可运行时注册自定义颜色映射

### 文件格式支持 (`src/parser/`)

PDB、mmCIF、SDF、MOL2、XYZ、CIF、MMTF（二进制压缩格式）、DX（电子密度）、MRC、CCP4 等20+种格式。

---

## 可借鉴的技术模式

1. **Impostor 技术**：用于任何需要大量球体/圆柱体渲染的场景（粒子系统、数据可视化）
2. **Picking Buffer 模式**：通过 `objectId` uniform 切换渲染模式，实现高效交互拾取
3. **SDF字体渲染**：无损缩放的3D文字（可用于数据标注）
4. **TiledRenderer**：分块高分辨率渲染，适合导出功能
5. **Spatial Hash + KD树**：大规模空间数据的高效近邻查询组合
6. **GLSL chunk 系统**：可复用 shader 片段的模块化组织方式

---

## 与 Astro SVGFigure 的关联价值

- **3D分子可视化组件**：如需在 SVGFigure 中嵌入分子结构图，NGL提供完整的WebGL嵌入方案
- **Impostor渲染思路**：SVG/Canvas大规模原子/节点渲染时可借鉴 ray-casting 的GPU减顶点策略
- **颜色映射系统**：NGL的可插拔 Colormaker Registry 是数据可视化颜色系统的优秀参考
- **空间索引**：若 SVGFigure 需要处理大规模点数据的近邻/碰撞，KD树+空间哈希组合值得复用

---

*xiaodi #88 — 2026-06-15*
