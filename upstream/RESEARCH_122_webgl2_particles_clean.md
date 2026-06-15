# RESEARCH #122 — pwambach/webgl2-particles 简洁粒子系统分析

## Transform Feedback 初始化流程（main.js）

采用**双缓冲 Ping-Pong** 架构：预分配两套 `vertexBuffers` + `velocityBuffers`。`createProgram` 时将 varyings `['v_position', 'v_velocity']` 以 `gl.SEPARATE_ATTRIBS` 模式注入，完成 TF 声明。单例 `gl.createTransformFeedback()` 复用于每帧。`calculateFeedback()` 内：先 `gl.enable(RASTERIZER_DISCARD)` 跳过光栅化，`bindBufferBase` 将输出槽挂到 invertedIndex 侧缓冲，`beginTransformFeedback(POINTS)` → draw → `endTransformFeedback()`，最后 `currentIndex ^= 1` 切换读写侧。极简——全程仅一个 TF 对象，无 texture。

## 粒子运动数学模型（calc_vertex.js）

**纯牛顿引力场，无噪声**。鼠标坐标为引力中心，`force = m₁·m₂ / r² · DAMPING`（DAMPING=1e⁻⁶ 防数值爆炸），`acceleration = force · direction`，欧拉积分更新 velocity 与 position，速度每帧乘 `0.99` 提供阻尼。边界 `±1.0` 触发法向速度反弹并衰减 50%。**无 simplex/curl noise**，计算成本极低。

## gl_PointSize 与深度（display_vertex.js）

`gl_PointSize = 1.0` 硬编码，**无透视除法，无深度衰减**。坐标直接 passthrough 到裁剪空间，完全 2D。Fragment 输出固定微透明深蓝 `vec4(0,0,0.01,0.1)` 叠加产生轨迹拖尾感。

## 与 #121 toji 版架构差异

| 维度 | #121 toji | #122 pwambach |
|---|---|---|
| 粒子状态存储 | RGBA Texture ping-pong (GPGPU) | Buffer + Transform Feedback |
| 力场模型 | Curl noise / simplex 噪声场 | 鼠标引力（纯物理） |
| 属性数量 | 位置/速度/颜色/寿命 | 仅位置 + 速度 |
| 复杂度 | 高（多 FBO + sampler） | 极低（2 buffer pair） |

## 适用于 astro-svgfigure 2D Cell 渲染的简化方案

此架构天然适配 2D cell 场景：
- **保留** Ping-Pong TF buffer 驱动 cell 状态（位置/相位）更新，力场改为**规则网格流场**（预存 texture 采样）
- `gl_PointSize` 改为 cell 边长（像素单位），fragment 输出 cell 颜色/纹理坐标
- 去除 `RASTERIZER_DISCARD`，TF pass 与 display pass 合并为单 draw call
- 无需 3D 深度，`gl_Position = vec4(a_position.xy, 0.0, 1.0)` 即可，较 toji 版减少约 **60% shader 代码量**