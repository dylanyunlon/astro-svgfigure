# RESEARCH #128 — Active Theory `modern-screenshot`

## DOM → Canvas → Image 截图管线

管线分三阶段：**Clone → Embed → Render**。

1. **Clone**：`cloneNode()` 深度递归复制目标 DOM 树，同步内联所有 computed CSS（`copyCssStyles`）、伪元素（`copyPseudoClass`）、input 值，并过滤 `<script>/<style>` 节点。
2. **Embed**：`embedNode()` 遍历克隆树，将所有外部资源（图片、CSS 背景图、Web Font、SVG `<use>`）base64 内联，4 路并发 task 队列（`Promise.all × 4`）加速资源抓取；远程 fetch 可卸载至 Web Worker 池（`workerNumber` 可配）。
3. **Render**：`domToForeignObjectSvg()` 将克隆树包装进 `<svg><foreignObject>` → 序列化为 data URL → `createImage()` 绘入离屏 `<canvas>` → 导出为 PNG/JPEG/WebP blob。

## SVG `foreignObject` 策略

核心技巧：用 `<foreignObject width="100%" height="100%">` 将任意 HTML 子树嵌入 SVG 命名空间，使浏览器原生布局引擎负责渲染，无需手动重建盒模型。所有外部引用须在序列化前完全内联（base64），否则 SVG 跨源沙箱会阻断加载。

## 与 html2canvas 的区别

| 维度 | `modern-screenshot` | `html2canvas` |
|---|---|---|
| 渲染方式 | 浏览器原生（foreignObject） | JS 重绘 Canvas |
| CSS 还原度 | 高（原生布局） | 有限（手动模拟） |
| 性能 | 并发 Worker + 4 路任务队列 | 单线程串行 |
| 输出格式 | PNG/JPEG/WebP/SVG/Blob/Pixel | PNG/JPEG |
| 体积 | 轻量（fork 自 html-to-image） | 较重 |

## 用于 `astro-svgfigure` 的 SVG 导出

`domToSvg()` 走独立路径：DOM → Canvas dataURL → `<svg><image href=dataUrl>`，输出完整 SVG 字符串。对于 astro-svgfigure，更推荐直接调用 `domToForeignObjectSvg()` 获取原生 `SVGElement`，再序列化为 `.svg` 文件——可保留文本节点（利于 SEO / 可访问性），并通过 `filter` 选项剔除非图形节点，结合 `scale` 选项控制导出分辨率。