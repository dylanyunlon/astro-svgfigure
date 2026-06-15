# RESEARCH_95 — SDF文字/形状渲染工具生态

> 小弟 #95 · xiaodi · branch: cell-pubsub-loop

---

## 总览：我们补齐了什么

| 工具 | 类型 | 运行环境 | 用途 |
|------|------|----------|------|
| `activetheory-svg2msdf` | Electron app (已有) | Desktop | SVG → MSDF binary wrapper |
| `tiny-sdf` | JS库 (新增) | **浏览器/Node** | Canvas 2D → SDF，无需binary |
| `msdfgen-source` | C++源码 (新增) | 编译后CLI/lib | MSDF算法原理 + 自定义编译 |
| `msdf-atlas-gen` | C++工具 (新增) | 编译后CLI | 批量TTF/OTF → MSDF字体图集 |

---

## 1. tiny-sdf — 浏览器内SDF（核心价值最高）

**repo:** `upstream/tiny-sdf/`  
**来源:** mapbox/tiny-sdf (BSD-2-Clause)

### 核心原理

```
Canvas 2D fillText()
  → 取像素 alpha 通道
  → Felzenszwalb/Huttenlocher 距离变换（EDT）
  → 输出 Uint8ClampedArray (SDF data)
```

算法关键：双网格EDT
- `gridOuter`: 字形**外部**的平方距离场
- `gridInner`: 字形**内部**的平方距离场  
- 最终 signed distance = `sqrt(outer) - sqrt(inner)`

### API

```js
import TinySDF from '@mapbox/tiny-sdf';

const sdf = new TinySDF({
  fontSize: 24,          // px
  fontFamily: 'sans-serif',
  buffer: 3,             // glyph周围padding (px) — halo用
  radius: 8,             // 编码距离范围 (px)
  cutoff: 0.25,          // 内外分界比例 (0.25 = 25%是inside)
});

const glyph = sdf.draw('泽');
// → { data: Uint8ClampedArray, width, height,
//     glyphTop, glyphLeft, glyphWidth, glyphHeight, glyphAdvance }
```

### 关键参数解析

| 参数 | 默认 | 含义 |
|------|------|------|
| `buffer` | 3 | glyph bbox外围padding，给halo/glow留空间 |
| `radius` | 8 | 距离场编码范围（px数），越大越平滑但越贵 |
| `cutoff` | 0.25 | alpha=128对应的相对距离，调整粗细 |

### 与我们项目的关系

- **不需要任何binary或服务器**，纯浏览器JS
- 适合：运行时动态生成字形SDF（地图label、UI文字）
- 不适合：需要sharp corners的logo/图标（用MSDF）
- 输出是**单通道SDF**，比activetheory-svg2msdf的MSDF精度低，但速度极快

### GLSL使用

```glsl
uniform sampler2D uSDFTexture;
uniform float uBuffer;   // 对应 cutoff = 0.25 → buffer = 0.75
uniform float uGamma;    // 通常 = 0.5 * screenPxRatio

void main() {
    float dist = texture2D(uSDFTexture, vUV).a;
    float alpha = smoothstep(uBuffer - uGamma, uBuffer + uGamma, dist);
    gl_FragColor = vec4(color, alpha);
}
```

---

## 2. msdfgen-source — MSDF算法源码

**repo:** `upstream/msdfgen-source/`  
**来源:** Chlumsky/msdfgen

### 目录结构重点

```
core/
├── edge-coloring.cpp/h      ← MSDF核心：给曲线边着色（R/G/B通道分配）
├── msdfgen.cpp              ← 主生成函数
├── contour-combiners.cpp    ← 多轮廓合并策略
├── MSDFErrorCorrection.*    ← 修正多通道artifact
├── Shape.cpp/h              ← 矢量形状表示
├── edge-segments.cpp/h      ← 线段/二次贝塞尔/三次贝塞尔
└── SignedDistance.hpp       ← 带符号距离数据结构
```

### MSDF vs SDF 关键区别

```
SDF (1通道):
  distance = min(所有边的距离)
  → sharp corners 退化为圆角

MSDF (3通道 RGB):
  1. edge-coloring: 给shape的边按角度分配R/G/B色
  2. 每通道独立计算perpendicular distance
  3. 取median(R,G,B)重建真实边界
  → 保留尖角！
```

### 与 activetheory-svg2msdf 的关系

`activetheory-svg2msdf` 就是这个库的Electron wrapper：
```
SVG → msdfgen binary (此库编译产物) → MSDF PNG
```

我们有binary但无源码时无法自定义参数，现在有源码可以：
- 修改edge-coloring阈值（控制corner检测角度）
- 自定义输出格式
- 集成到Node.js native addon

---

## 3. msdf-atlas-gen — 批量字体图集生成

**repo:** `upstream/msdf-atlas-gen/`  
**来源:** Chlumsky/msdf-atlas-gen

### 功能

从TTF/OTF字体文件批量生成紧密打包的MSDF图集。

### Atlas类型对比

| 类型 | 通道数 | 抗锯齿 | 可缩放 | 尖角 | 软效果 | 硬效果 |
|------|--------|--------|--------|------|--------|--------|
| SDF  | 1 | ✓ | ✓ | — | ✓ | — |
| PSDF | 1 | ✓ | ✓ | — | — | ✓ |
| **MSDF** | **3** | **✓** | **✓** | **✓** | — | ✓ |
| MTSDF | 4 | ✓ | ✓ | ✓ | ✓ | ✓ |

### 典型命令

```bash
# 生成ASCII字符集MSDF图集
msdf-atlas-gen \
  -font NotoSans.ttf \
  -type msdf \
  -format png \
  -imageout atlas.png \
  -json atlas.json \
  -size 32

# 批量生成中文字符
msdf-atlas-gen \
  -font NotoSansCJK.ttf \
  -charset cjk-common.txt \  # 3500常用汉字
  -type msdf \
  -pxrange 4 \
  -imageout cjk_atlas.png \
  -json cjk_atlas.json
```

### 输出格式

- **PNG/BMP** — 图集纹理
- **JSON** — glyph布局元数据（UV坐标、advance、bearing）
- **CSV** — 表格形式
- **Artery Font** — 二进制格式，图集+元数据合一

### 关键目录结构

```
msdf-atlas-gen/
├── GlyphGeometry.cpp/h      ← 单字形几何+SDF生成
├── FontGeometry.cpp/h       ← 整个字体的字形集合
├── TightAtlasPacker.cpp/h   ← 矩形装箱（tight packing）
├── GridAtlasPacker.cpp/h    ← 等间距网格布局
├── RectanglePacker.cpp/h    ← 底层矩形打包算法
├── json-export.cpp/h        ← JSON元数据导出
└── Charset.cpp/h            ← 字符集定义
```

---

## 技术选型建议

### 场景 → 工具

```
场景A: 运行时动态文字 (地图/UI标签)
→ tiny-sdf (浏览器内，无需binary)

场景B: 静态字体图集 (游戏/WebGL应用)
→ msdf-atlas-gen (编译+预生成，运行时零开销)

场景C: SVG图标/形状 → MSDF纹理
→ activetheory-svg2msdf (已有，Electron GUI)

场景D: 需要自定义MSDF参数/格式
→ msdfgen-source (编译，可集成到Node native addon)
```

### SDF参数调优参考

```
radius / pxrange: 影响渲染时可用的抗锯齿带宽
  - 太小 → 缩放时出锯齿
  - 太大 → 细节丢失，纹理浪费
  - 推荐: 字形px size的 1/8 ~ 1/4

cutoff (tiny-sdf) / buffer (atlas-gen):
  - 控制"边界"对应的值 (0.5 = 128/255)
  - 调大 → 文字变细；调小 → 文字变粗
```

---

## GLSL Shader — MSDF渲染片元（参考activetheory-svg2msdf）

```glsl
// 与 activetheory-svg2msdf 的README一致
float msdf(sampler2D tMap, vec2 uv) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float d = fwidth(signedDist);
    float alpha = smoothstep(-d, d, signedDist);
    return alpha;
}
```

`median(R,G,B)` = `max(min(R,G), min(max(R,G),B))`  
这是MSDF的核心解码公式，消除多通道artifact。

---

## 文件清单

```
upstream/
├── tiny-sdf/              ← 浏览器JS SDF生成器
│   ├── index.js           ← 核心实现 (ES module)
│   ├── index.d.ts         ← TypeScript类型
│   └── README.md
├── msdfgen-source/        ← MSDF C++算法源码
│   ├── core/              ← 核心算法 (50+文件)
│   ├── ext/               ← 扩展 (FreeType字体加载)
│   └── main.cpp           ← CLI入口
└── msdf-atlas-gen/        ← 批量图集生成工具
    ├── msdf-atlas-gen/    ← 源码目录
    └── msdfgen/           ← 内嵌msdfgen子模块
```

---

*RESEARCH_95 · 小弟#95 xiaodi · 2025*
