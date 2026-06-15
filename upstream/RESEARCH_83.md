# 小弟 #83 Research Report: Scientific Color Palettes for WebGL
**Researcher:** xiaodi  
**Branch:** cell-pubsub-loop  
**Date:** 2026-06-15

---

## 目标 / Objective

为 astro-svgfigure 的 WebGL 渲染层提供科研级感知均匀色表（Perceptually Uniform Colormaps），用于：
- 节点/边的数据驱动着色（species intensity, activation values）
- 后处理 pass 的伪彩色映射
- 粒子系统的密度/速度可视化

---

## 搜索结果摘要 / Search Summary

### 关键发现

| 库/资源 | Stars | 方案 | 适合场景 |
|---------|-------|------|---------|
| `glsl-colormap` (glslify) | ~500 | piecewise smoothstep, 纯 GLSL | ✅ WebGL shader 直接用 |
| `glsl-gradient-palette` (Erkaman) | ~200 | 1D texture lookup via gl-texture2d | ✅ 大色表/动态换色 |
| `scale-color-perceptual` (politiken) | ~400 | JS 端插值查表，输出 RGB | ✅ CPU 端生成纹理 |
| `d3-scale-chromatic` | Official D3 | JS 查表, 256 entry arrays | ✅ CPU 端 → 上传 1D texture |
| `color-palette-shader` (meodai) | ~300 | WebGL2 感知色空间可视化 | ✅ 调色板分析工具 |

### 感知均匀色表起源

Viridis/Plasma/Inferno/Magma 由 Stéfan van der Walt 和 Nathaniel Smith 为 matplotlib 设计（发布于 SciPy 2015），目标：
1. **感知均匀**：亮度沿色表单调递增，Δ色感 ∝ Δ数据值
2. **色盲友好**：在 deuteranopia/protanopia 下仍可区分
3. **灰度打印安全**：转灰度后仍保留信息

---

## 技术方案 / Technical Approaches

### 方案 A：多项式/解析近似（推荐用于实时渲染）

直接在 GLSL 中用多项式拟合色表，**零纹理采样**，适合高频 fragment shader：

```glsl
// Viridis — 4阶多项式近似（来源: IQ / shadertoy 社区）
vec3 colormap_viridis(float t) {
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
```

优点：纯数学计算，无纹理带宽，适合任何 WebGL 1.0+  
缺点：多次乘法，GPU 上约 20-30 FLOPs/pixel

### 方案 B：1D 纹理查表（适合可变色表 / UX 调色）

```typescript
// CPU 端：用 d3-scale-chromatic 或 scale-color-perceptual 生成 256×1 RGBA 纹理
import { interpolateViridis } from 'd3-scale-chromatic';

function buildColormapTexture(gl: WebGL2RenderingContext, interpolator: (t: number) => string) {
    const size = 256;
    const data = new Uint8Array(size * 4);
    for (let i = 0; i < size; i++) {
        const c = interpolator(i / (size - 1)); // returns "rgb(r, g, b)"
        // parse and fill data[i*4..i*4+3]
    }
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    return tex;
}
```

GLSL 端采样：
```glsl
uniform sampler2D u_colormap;
vec3 sampleColormap(float t) {
    return texture2D(u_colormap, vec2(t, 0.5)).rgb;
}
```

### 方案 C：glsl-colormap（piecewise smoothstep）

npm 包 `glsl-colormap` 用 `smoothstep` 在关键颜色控制点间插值，生成 GLSL 函数。  
适合 glslify 构建链（非本项目 PixiJS 架构）。

---

## 色表特征对比 / Colormap Comparison

| 色表 | 起色 | 终色 | 色相范围 | 亮度范围 | 最佳用途 |
|------|------|------|---------|---------|---------|
| **Viridis** | 深紫 | 亮黄 | 蓝→绿→黄 | 15%→90% | 通用科研，首选 |
| **Plasma** | 深紫 | 亮黄 | 紫→品→橙→黄 | 12%→92% | 高对比视觉效果 |
| **Inferno** | 黑 | 白黄 | 黑→红→橙→白 | 0%→95% | 热力图，密度图 |
| **Magma** | 黑 | 白黄 | 黑→紫→粉→白 | 0%→95% | 与 Inferno 相近，更粉 |
| **Turbo** | 深蓝 | 暗红 | 全色谱（均匀） | 15%→85% | Jet 替代，高分辨率 |

---

## 对项目的建议 / Recommendations for astro-svgfigure

### 立即可用

- `src/lib/shaders/colormap.frag`：已创建，包含 Viridis/Plasma/Inferno/Magma 的 GLSL 多项式近似
- 可直接在 species shader、particle shader、edge-line shader 中 `#include`

### 配色映射架构

```
数据值 [0,1]
    │
    ├── [实时渲染] → colormap.frag 多项式 → vec3 color
    │
    └── [可交互调色] → JS 生成 1D texture → GLSL texture2D 采样
                        (使用 d3-scale-chromatic 或 scale-color-perceptual)
```

### npm 依赖建议

```json
{
  "d3-scale-chromatic": "^3.0.0",    // 丰富色表库 (Viridis/Plasma/等)
  "scale-color-perceptual": "^1.0.1"  // 轻量替代，仅 Viridis/Plasma/Inferno/Magma
}
```

---

## 克隆的上游库 / Cloned Upstream

| 路径 | 库 | 用途 |
|------|----|------|
| `upstream/glsl-colormap/` | glslify/glsl-colormap | 参考 piecewise smoothstep 生成器 |

---

## 参考资料 / References

1. [glsl-colormap (GitHub)](https://github.com/glslify/glsl-colormap) — 105 种 GLSL 色表实现
2. [glsl-gradient-palette (Erkaman)](https://github.com/Erkaman/glsl-gradient-palette) — 1D纹理梯度色表
3. [scale-color-perceptual (politiken)](https://github.com/politiken-journalism/scale-color-perceptual) — JS 版 matplotlib 色表
4. [WebGL Color Maps (Observable)](https://observablehq.com/@flimsyhat/webgl-color-maps) — WebGL 交互演示
5. [Introduction to viridis (CRAN)](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html) — 学术背景
6. [Shadertoy: Viridis Approximation](https://www.shadertoy.com/view/XtGGzG) — 多项式系数来源
7. [Shadertoy: Jet comparison](https://github.com/Michaelangel007/shadertoy_jet_color_mapping_comparison) — GLSL 实现对比
8. [Geeks3D: GLSL colormap demo](https://www.geeks3d.com/hacklab/20200821/demo-a-collection-of-color-map-shaders/) — 105种色表 GeeXLab demo
