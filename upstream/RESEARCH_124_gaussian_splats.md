现在我有充足的源码信息，生成 RESEARCH 摘要：# RESEARCH — Active Theory · GaussianSplats3D Fork
**xiaodi #124 · 2026-06-15**

---

## 1. Gaussian Splatting 核心原理：3D → 2D 投影

每个 splat 是一个**各向异性 3D 高斯椭球**，由位置 `(x,y,z)`、三轴尺度 `(sx,sy,sz)` 和四元数旋转 `q` 完整描述其形态。渲染时，顶点着色器通过 Jacobian 矩阵 `J`（摄像机焦距/视深）与视图转置矩阵 `W` 的乘积，将 3D 协方差 `Σ = R·S·Sᵀ·Rᵀ` 投影为屏幕空间 2×2 协方差 `Σ' = TΣTᵀ`，再对其求特征值/特征向量，得到 2D 椭圆的**主轴基向量**，决定每个 splat 在屏幕上的尺寸与朝向。

---

## 2. SplatMesh.js 渲染管线：Instanced Quad + Alpha Blending

以**单张 unit quad**（±1 四顶点）为基础几何，`THREE.InstancedBufferGeometry` 对全量 splat 做 GPU 实例化，每帧一次 draw call。中心坐标与 RGBA 颜色打包进 `centersColorsTexture`（RGBA32UI），协方差上三角 6 个分量存入 `covariancesTexture`（RG Float）；顶点着色器采样两张 GPU 纹理，在 clip space 计算椭圆偏移。片元着色器计算 `A = exp(−|vPosition|²) × α`，配合 `NormalBlending + depthWrite:false` 实现 **order-dependent alpha blending**。

---

## 3. Worker 线程：WASM Counting-Sort 深度排序

`SortWorker.js` 在独立 Web Worker 中加载 Emscripten 编译的 `sorter.wasm`，通过 `SharedArrayBuffer` 零拷贝共享数据。排序为 **Counting Sort**：将每个 splat 投影到 `viewProj` Z 行得到整数深度，统计频率直方图（`DepthMapRange` 桶）并转前缀和，反向扫描写回 `indexesOut`，实现 **back-to-front** 排列，保证 alpha 合成正确性，全程零 JS 堆分配。

---

## 4. SplatBuffer.js 数据结构

每个 splat 顺序存储四段：**position**（3×Float32，12B）→ **scale**（3×Float32，12B）→ **color**（4×Uint8 RGBA，4B）→ **rotation**（4×Float32 四元数，16B）。Level 1 压缩以 bucket 空间分组量化位置，scale/rotation 降为 Half Float。`fillCovarianceArray()` 在 CPU 端从 scale+rotation 合成 `Σ` 上三角推送 GPU。**此 fork 未存储球谐系数**，颜色已预烘焙为固定 RGBA，省去高阶 SH 解码。

---

## 5. 移植思路：splat → astro-svgfigure 2D Cell 渲染

| 3D Splat 概念 | 2D Cell 映射 |
|---|---|
| 3D 协方差 → 屏幕椭圆 | SVG `<ellipse rx ry transform="rotate(θ)">` |
| eigenValue1/2 轴长 | cell 的 `rx/ry` 缩放比 |
| `exp(−r²)` 透明度 | `fill-opacity` 高斯曲线赋值 |
| Back-to-front 排序 | 按伪深度排序 DOM，利用画家算法 |
| GPU 纹理批量数据 | 构建期预计算 → 静态 JSON props |

核心简化：去掉 Jacobian 投影（已在 2D 空间），以 `(cx, cy, rx, ry, angle, opacity)` 六参数描述每个 cell；深度排序在构建期完成；透明度曲线用 CSS `opacity` 近似，实现**纯声明式 SVG 渲染，零运行时计算**。