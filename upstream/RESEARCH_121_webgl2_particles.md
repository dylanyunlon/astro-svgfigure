---

# RESEARCH 摘要：WebGL2 GPU 粒子系统分析

**xiaodi #121 · Transform Feedback 粒子系统研究**

---

## Transform Feedback Ping-Pong Buffer 机制

WebGL2 通过 `gl.transformFeedbackVaryings()` 将 vertex shader 的输出变量（`outPosition`、`outVelocity`、`outRandomSeed`）直接写回 GPU Buffer，无需经过 CPU。Ping-pong 模式使用两组 VAO/VBO 交替互换：帧 N 以 Buffer A 为输入、Buffer B 为 Transform Feedback 写出目标；帧 N+1 反转角色。全程数据留在 GPU 显存，彻底规避 CPU↔GPU 带宽瓶颈，支持百万级粒子实时仿真。

## SimulationShader.js 粒子物理更新逻辑

Vertex shader（GLSL ES 3.00）每帧对每颗粒子独立执行：`outPos.xyz = pos.xyz + vel.xyz`（欧拉积分推进位置）；已被碰撞器触碰的粒子（`pos.w == 1.0`）附加 `vel *= 0.95` 阻尼与朝向原点的复位力 `resetVec`；碰撞器圆域内施加法向推离力与切向旋转速度；边界墙壁做速度分量反射。粒子以线性同余生成器（LCG，Microsoft VC++ 参数）决定随机复位。

## gl.POINTS 与 gl_PointSize 点精灵渲染

渲染 Pass 使用 `gl.POINTS` 图元，每个粒子顶点在屏幕上扩展为正方形精灵；`gl_PointSize`（像素单位）控制精灵尺寸，可在 vertex shader 中依据相机距离动态缩放（`size / -mvPosition.z`），配合 `gl_PointCoord` 实现圆形/光晕纹理采样，是零几何开销的高效渲染方案。

## 应用于二十面体顶点粒子科研绘图

将正二十面体 12 个顶点坐标写入 `origin` 属性作为粒子归宿点，利用复位力 `normalize(origin - pos) * k` 使粒子持续向各顶点收敛聚集，形成稳定的对称星云结构。调整 `resetRate` 控制粒子寿命以呈现顶点间流动轨迹；`gl_PointSize` 按顶点重要性或数据量映射，可将点群颜色编码为标量场值，适用于晶体结构、分子轨道、拓扑不变量等科研数据的直观 GPU 可视化。