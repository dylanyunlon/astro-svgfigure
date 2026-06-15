---

# RESEARCH — Active Theory · Finding Love Shaders
**研究员：xiaodi #123 ｜ 仓库：activetheory/Finding-Love-Shaders**

---

## Gem.fs — 半透明体积光照模型

Gem.fs **并非**经典菲涅耳/环境映射 PBR，而是一套 **次表面散射近似（Translucency SSS）**：

- **厚度估算**：`thickness = range(length(vPos), 0.0, radius, 1.0, 0.5)`，以顶点到原点距离模拟光路厚度。
- **背光散射**：`ltLight = normalize(L + N * LTDistortion)`，沿法线扰动光方向模拟透射；`ltDot = dot(V, -ltLight) * LTScale` 给出视角相关的透射亮度。
- **二次衰减**：`1/(kC + kL·d + kQ·d²)` 模拟点光源物理衰减。
- **HSV 色噪**：simplex noise 在时域调制 `lightColor` 的色相，产生宝石内部流动感。

无环境 cubemap 采样，无 Fresnel `mix()`——属**风格化 translucency**，非物理 PBR。

---

## Sky.fs — 程序化大气风格着色

Sky.fs **不含 Rayleigh/Mie 散射方程**，采用轻量化噪声驱动色彩混合：

- **双色调混合**：`color0`（基础色）与 `color1`（HSV 色相 +0.25 偏移）通过 3D simplex noise 插值，模拟晨昏/极光渐变。
- **高度亮化**：`finalColor *= clamp(range(abs(vPos.y), 0.0, 25.0, 1.2, 1.0), 1.0, 15.0)`，越靠近天穹中心越亮，近似地平线到天顶的大气梯度。
- **动态扰动**：`snoise(vPos * 0.01 + time * 10.0)` 驱动云层流动。

---

## Terrain.fs + Terrain.vs — 地形着色技术

**顶点层（.vs）：** Simplex noise 顶点置换 `pos.y += raise * heightMask(uv)` 实时生成有机地貌，无需高度图。`vSat = crange(pos.y, ...)` 将高度编码进 varying 驱动低洼处色彩更饱和。

**片元层（.fs）：** `getDNormal()` 用 `dFdx/dFdy` 屏幕空间导数重建法线（无法线贴图）；Lambert 漫反射混合 `baseColor/lightColor`；平方衰减径向雾 `pow(smoothstep(25,35,length(xz)),2)`；`transitionMask()` 支持双向径向 alpha 揭示动画，`discard` 裁剪边缘。

---

## uil.min.js — Active Theory 调试面板

Active Theory 自研轻量 GUI 库，对标 dat.GUI：

```js
window.uil = new UIL.Gui({ css: 'top: 0; right: 50px;', size: 300, center: true });
```

各 element 通过 `window.uil.add(...)` 绑定 shader uniform（`strength`、`sat`、`transition` 等），浏览器内**实时热更新**，无需重编译，是 AT WebGL 工作流的标准调参组件。

---

## 应用于 astro-svgfigure · Species 渲染

| AT Shader 技术 | Species 渲染对应场景 |
|---|---|
| **Gem SSS（厚度 + 背光透射）** | 晶体/星云状 species 躯体半透明发光；`radius` uniform 控制个体大小差异 |
| **Sky 双色噪声混合** | Species 光晕或星场背景色相漂移，驱动"种群涌现"节律动画 |
| **Terrain 屏幕空间法线 + noise 置换** | SVG figure 转 3D mesh 后的无贴图程序化 bump，支持种群个体差异 |
| **Terrain 径向转场遮罩** | Species 出现/消失的"结晶化"径向 alpha discard 动画 |
| **UIL 面板** | 开发期对每个 species 的 `LTDistortion`、`transHeight` 等实时调参，固化为 JSON 配置 |

> **核心结论**：AT 这套 shader 以**极低 GPU 开销**（无 IBL cubemap、无 G-buffer、屏幕空间法线替代贴图）换取高度风格化结果。对 `astro-svgfigure` 大量实例化的 species 群体而言，单 drawcall 内通过 uniform 变体实现种群个体差异，是性能与表现力兼顾的理想移植路径。