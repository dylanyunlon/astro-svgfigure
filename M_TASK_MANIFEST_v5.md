# M400-M460 开发里程碑 — Sprite→Editor 颠覆 + 生产化

**Branch**: `fix/cfc0672-enhancements`
**日期**: 2026-05-31
**制定者**: 第一位 Claude（本次会话），基于像素级参考图分析 + 全仓库代码审计

---

## 已完成的 M 范围回顾

| 范围 | 内容 | 状态 |
|------|------|------|
| M001-M010 | 基础设施（HSV引擎、并行变换、布局引擎等） | M001-M002 ✅, 其余 🔲 |
| M100-M120 | 生产级 Bug 修复 + SAM3 集成 | 规划中 |
| M200-M224 | Sprite 管线（分类→prompt→拼版→序列帧→对齐→矢量化→缓存→导出→注入） | ✅ 代码完整但为死代码 |
| M300-M310 | tldraw 集成层（ELK↔tldraw 桥、ElkNodeShapeUtil、sprite 桥接） | ✅ |

---

## 本次颠覆的核心架构变更

```
旧流程: ELK布局 → to-svg.ts 渲染 → 静态 SVG → Skeleton tab（主输出）
                                    ↘ tldraw Editor tab（附属）

新流程: ELK布局 → classify_nodes() → inject_sprites(Gemini→rembg→stamp)
         → sprite-enriched ELK → tldraw Editor tab（主输出，可编辑）
                               ↘ to-svg.ts SVG（备选/导出用）
```

---

## ✦ 第一位 Claude 完成：M400-M406

**Commit**: `aca9efc` (2026-05-31)
**改动**: 4 文件, +921 行

### M400 — 架构颠覆：Sprites→Editor 主输出 ✅

将 tldraw Editor 从附属 tab 提升为主输出面。Gemini sprites 生成后自动切到 Editor tab。

**实现文件**: `src/pages/generate/index.astro`
- sprite 生成完成后 `document.getElementById('tab-editor')?.click()` 
- `tldrawApi.loadGraph(data.elk_graph)` 直接加载 sprite-enriched 图
- 状态消息引导用户在 Editor 中编辑

### M401 — SvgPreview sprite 感知 + tldraw 自动加载 ✅

前端能检测 ELK 图是否包含真实 Gemini sprites，自动将数据推送到 tldraw。

**实现文件**: `src/components/pipeline/SvgPreview.astro`
- `_graphHasSprites(graph)` — 递归检测 `spriteRef.url` 
- `showSkeleton()` 在 Editor tab 活跃时自动调用 `tldrawApi.loadGraph()`

### M402 — renderKernelGrid() 渲染内核 ✅

参考图中 AdaKern 区域的 3×3 kernel grid（Static/Low-freq/High-freq/Adaptive kernel）缺少专用渲染。新增 `renderKernelGrid()` 作为 `to-svg.ts` 的第 6 个几何内核。

**实现文件**: `src/lib/elk/to-svg.ts`
- `renderKernelGrid(node, x, y, w, h)` — NxN 加权色格，确定性权重
- `renderNode()` 分派新增 `renderMode='kernel'` 分支

### M403 — ⊛ Convolve + ⊖ Subtract 矢量算子 ✅

参考图的 ⊛（卷积）和 ⊖（减法）算子之前走 text fallback，现在有专用矢量图形。

**实现文件**: `src/lib/elk/to-svg.ts`
- `⊛` → 8 线星号（+×叠加）
- `⊖` → 圆内水平线

### M404 — tldraw Path B2: Kernel Grid 编辑器渲染 ✅

kernel grid 在 tldraw Editor 中也能渲染，不只是 SVG。

**实现文件**: `src/lib/tldraw/ElkNodeShapeUtil.tsx`
- 新 Path B2: `renderMode === 'kernel'` → CSS Grid 渲染 3×3 色格
- 确定性权重来自 label hash，同一节点永远相同配色

### M405 — FreqSelect/AdaDR/AdaKern 渲染验证测试 ✅

用参考图的完整架构构建 ELK 测试数据，验证所有 7 个渲染内核。

**实现文件**: `tests/test_freqselect_render.ts`
- 37 个节点（8 feature-map stacks, 6 kernel grids, 9 operators, 14 leaf/label）
- 验证: 7/7 operator circles, 7/7 feature stacks, 3/3 group blobs

### M406 — 像素级参考图分析文档 ✅

对上传的 FreqSelect/AdaDR/AdaKern 参考图做逐行逐列扫描，输出完整元素清单：8 tensor stacks, 6 kernel grids, 9 operators, 3 regions, ~30 labels, ~25 arrows。

---

## ✦ 第二位 Claude 应完成：M410-M416

**重点**: 后端管线贯通 — classify_nodes 真正跑起来 + Gemini API 实调

### M410 — classify_nodes() 全节点分类器增强

当前 `node_classifier.py` 的分类规则不够覆盖参考图场景。增强规则：
- 包含 "kernel"/"filter"/"weight" → `renderMode='kernel'`
- 包含 "C×H×W"/"feature"/"map"/"tensor" → `renderMode='sprite'`, `spriteRef.format='stack'`
- 包含 "⊗"/"⊕"/"⊛"/"⊖" 或 label 是纯数学符号 → `isOperator=true`
- `familyId` 基于父 group ID 自动继承（同 group 下的 feature maps 属于同一 family）

**文件**: `backend/pipeline/topology/node_classifier.py`

### M411 — inject_sprites() 端到端打通

确保 `inject_sprites()` 在没有 Gemini API key 时优雅降级到 `spriteRef.format='stack'`（feature-map 堆叠）而非报错。有 key 时走完整 Gemini→rembg→stamp 流程。

**文件**: `backend/pipeline/topology/sprite_injector.py`

### M412 — /api/sprite-generate 后端路由验证

在 `server.py` 中确保 `/api/sprite-generate` endpoint 存在且正确调用 `classify_nodes()` + `inject_sprites()`。添加 `?dry_run=true` 参数跳过 Gemini 调用只做分类。

**文件**: `server.py` 或 `backend/server_layered_routes.py`

### M413 — Gemini interleaved output 替代 sheet 拼版

当前 `sprite_batch_generator.py` 用 sprite sheet 网格方案（一张大图裁切）。改为 Gemini `responseModalities: ['TEXT','IMAGE']` 一次请求返回 N 张独立图片。优势：不需要 sheet 裁切、风格天然一致。

**文件**: `backend/pipeline/topology/sprite_batch_generator.py`

### M414 — rembg 去背景质量阀门

Gemini 生成的图片有非透明背景。当前 rembg 去背景后质量参差。增加质量评分阀门——如果去背景后的 alpha 通道边缘质量 < 阈值，回退到绿幕检测或标记为需要手动修正。

**文件**: `backend/pipeline/topology/sprite_injector.py`, `backend/pipeline/transparency_validator.py`

### M415 — spriteRef URL → Data URI 自动转换

Gemini 返回的图片需要以 Data URI (`data:image/png;base64,...`) 形式嵌入 ELK JSON，因为前端 tldraw 在沙箱中不能访问后端临时文件路径。确保 `stamp_sprite_ref()` 输出 Base64 Data URI。

**文件**: `backend/pipeline/topology/sprite_injector.py`

### M416 — 端到端集成测试：文字输入 → Editor 可编辑

写一个从 "FreqSelect adaptive frequency selection" 文字输入开始，到 tldraw Editor tab 显示带 sprite 节点的完整集成测试。

**文件**: `tests/test_e2e_sprite_to_editor.py`

---

## ✦ 第三位 Claude 应完成：M420-M426

**重点**: tldraw Editor 交互增强 — 从"能看"到"能编辑"

### M420 — tldraw 节点双击替换 sprite

用户双击一个 sprite 节点 → 弹出 prompt 输入框 → 重新调用 Gemini 生成该节点的新 sprite → 热替换。

### M421 — tldraw 节点右键菜单

右键菜单：Regenerate Sprite / Change to Blob / Change to Text / Copy ELK JSON / Delete

### M422 — tldraw 连线编辑

用户可以在 Editor 中拖拽连线起止点，改变节点间的连接关系。修改 `tldrawToElk()` 将编辑后的图导出回 ELK JSON。

### M423 — tldraw → 导出高清 PNG

从 tldraw Editor 导出当前画布为高清 PNG（4K 分辨率）。使用 tldraw 内置的 `editor.getSvg()` 然后 Canvas 光栅化。

### M424 — tldraw 节点尺寸自适应

sprite 图片加载完成后，自动调整节点尺寸以适配图片宽高比（而不是拉伸到预设的 ELK 框大小）。

### M425 — tldraw minimap + zoom controls

添加缩略图导航和精确缩放控件。大型图（30+ 节点）在 Editor 中需要快速定位。

### M426 — Editor tab 作为默认 tab

Step 1 完成后默认显示 Editor tab（不是 Skeleton）。Skeleton 降级为"查看源码"功能。

---

## ✦ 第四位 Claude 应完成：M430-M436

**重点**: micro-diff 序列帧 — 参考图核心视觉的 AI 实现

### M430 — Family-aware prompt 设计器增强

同 familyId 的节点（如 Input feature / Decomposed feats / Output feature 都属于 'feature_maps' family）的 Gemini prompt 必须描述"同一类型但有微小差异"。增强 `sprite_prompt_designer.py` 生成 family-aware prompts。

### M431 — 微差序列帧一致性约束

对同 family 的 N 个节点，用一次 Gemini 调用生成 N 张图（interleaved output），并在 prompt 中明确要求"相同整体结构，每张有细微纹理差异"。

### M432 — 热力图 / Selection map 专用 prompt

参考图中的 Selection map (1×H×W) 和 Dilation map 是单通道热力图。为这类节点设计专用 prompt："生成一张 colormap 热力图，冷色代表低值，暖色代表高值"。

### M433 — Kernel grid 专用 prompt

3×3 kernel grid 应该生成为带权重颜色的网格图，而非随机纹理。prompt："生成一个 3×3 卷积核可视化，中心权重最大（深色），边缘权重较小（浅色）"。

### M434 — Sprite 序列帧缓存 + 增量更新

用户修改了一个节点的 label 后，只重新生成该节点的 sprite，而非全部重新生成。基于 `(label + familyId + parentGroup)` 的 hash 做缓存键。

### M435 — Sprite 加载进度条

sprite 生成是耗时操作（可能 30-60 秒）。在 tldraw Editor 中，每个节点显示加载动画（旋转圆圈），sprite 返回后渐变替换。

### M436 — A/B 对比视图

提供"Before/After"对比模式——左边是 blob 占位符视图，右边是 sprite 替换后的视图。拖动分割线可以对比。

---

## ✦ 第五位 Claude 应完成：M440-M446

**重点**: 导出管线 — 从编辑器到论文

### M440 — tldraw → 学术论文级 SVG 导出

从 Editor 导出的 SVG 应该是干净的学术图（无 tldraw 内部标记、无 React 残留），可以直接插入 LaTeX `\includegraphics{}`。

### M441 — tldraw → PDF 导出 (vector)

矢量 PDF 导出——文字可选、颜色准确、线条锐利。使用 `svg2pdf.js` 或 `pdfkit`。

### M442 — tldraw → PowerPoint 导出

导出为 .pptx 幻灯片，每个 group 区域一个 slide，sprites 作为嵌入图片。

### M443 — 批量重新着色

用户可以一键更换 group 背景色板（从绿/橙/米 切换到 蓝/紫/灰 等学术风格配色）。

### M444 — 标注层叠加

在 sprite 图上叠加标注（维度标注 C×H×W、箭头标注、数学公式标注）。

### M445 — 字体嵌入 + 多语言支持

导出 SVG/PDF 时嵌入字体，支持中文标注（用 Noto Sans CJK）。

### M446 — 水印 + 版权信息

可选的水印叠加（论文标题、作者信息、日期），位于导出图片右下角。

---

## ✦ 第六位 Claude 应完成：M450-M456

**重点**: 生产化部署 + 性能优化

### M450 — Gemini API 并发限流 + 重试

Gemini API 有 rate limit。实现令牌桶限流器 + 指数退避重试 + 429 错误处理。

### M451 — Sprite 缓存持久化 (Redis/SQLite)

sprite 缓存从内存字典改为 Redis 或 SQLite，跨重启保持。基于 content hash 去重。

### M452 — WebSocket 进度推送

sprite 生成进度通过 WebSocket 实时推送到前端（"正在生成第 3/12 个节点..."），替代当前的轮询。

### M453 — Docker 一键部署

`Dockerfile` + `docker-compose.yml`：Astro 前端 + Python FastAPI 后端 + Redis 缓存 + rembg worker。

### M454 — 安全加固

Gemini API key 环境变量管理、CORS 白名单、请求大小限制、文件上传扫描。

### M455 — 性能基准 + 监控

建立性能基准：单张 sprite 生成 < 3s，16 节点全量生成 < 45s，tldraw 加载 < 500ms。Prometheus metrics 端点。

### M456 — 用户指南 + API 文档

README 重写、交互式 demo 页面、API endpoint 文档 (OpenAPI/Swagger)。

---

## 总结

| Claude | 任务范围 | 重点 | 状态 |
|--------|---------|------|------|
| **第一位** | **M400-M406** | **架构颠覆：Sprites→Editor 主输出 + 新渲染内核** | **✅ 已完成** |
| 第二位 | M410-M416 | 后端管线贯通：classify→inject→Gemini 实调 | 🔲 待开发 |
| 第三位 | M420-M426 | tldraw 交互增强：双击替换/右键菜单/连线编辑 | 🔲 待开发 |
| 第四位 | M430-M436 | 微差序列帧：family-aware prompt + 热力图 | 🔲 待开发 |
| 第五位 | M440-M446 | 导出管线：SVG/PDF/PPTX/标注/字体 | 🔲 待开发 |
| 第六位 | M450-M456 | 生产化：限流/缓存/WebSocket/Docker/监控 | 🔲 待开发 |
