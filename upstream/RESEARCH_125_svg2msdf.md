现在我掌握了所有信息，输出 RESEARCH 摘要：

---

# RESEARCH — svg2msdf · xiaodi #125

**MSDF 原理**
Multi-channel Signed Distance Field 将形状的有符号距离场编码进 RGB 三个通道，每通道独立记录不同方向轮廓的距离值。GPU 着色时通过中值重建（`max(min(r,g), min(max(r,g),b)) - 0.5`）还原精确边界，再以 `smoothstep(fwidth)` 抗锯齿输出，从而在任意缩放下保持亚像素级锐度。

**svg2msdf 流水线**
工具本体是 Electron 拖拽壳，核心为跨平台预编译二进制 `msdfgen`（Viktor Chlumsky v1.5）。拖入 SVG 后，`script.js` 拼接命令：
```
msdfgen msdf -svg <file> -o <name>_msdf.png
  -autoframe -format png -keeporder -pxrange 4 -size 128 128
```
msdfgen 解析 SVG 最后一条 path，构建贝塞尔轮廓，输出 128×128 RGB PNG atlas。

**边着色算法（Edge Coloring）**
msdfgen 按相邻边夹角（默认阈值可 `-angle` 调整）将边集分组，循环分配 R / G / B / RG / GB / RB 颜色标签；跨通道的"伪角点"使三通道各自捕获不同方向的距离极值，重建时消除单通道在尖角处的模糊缺陷。`-seed` 控制随机着色，`-keeporder` 禁用绕行检测以保持 SVG 原始路径方向。

**vs tiny-sdf（单通道 SDF）**
tiny-sdf 在 Canvas 上光栅化后计算 2D EDT，结果为单灰度通道；角点处因距离场退化为圆弧而产生"磨圆"失真。MSDF 用三通道编码方向性，角点锐度完整保留，且 64 px MSDF 视觉质量优于 1024 px 黑白 PNG，显存占用降低约 256×。

**在 astro-svgfigure 中的应用**
科研图表的坐标轴标签、单位符号、下标/上标等矢量字形可预烘焙为 MSDF atlas（单张 PNG 打包多字形）；WebGL/Three.js 场景中通过 `uv` 采样 + GLSL `msdf()` 函数实时渲染，无需依赖 DOM 字体渲染，支持任意旋转缩放与自定义着色，同时规避 SVG-in-canvas 的跨浏览器差异——是高密度科研海报与交互式数据可视化的理想文字渲染方案。