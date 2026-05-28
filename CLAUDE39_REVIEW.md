# Claude #39 审查报告 — 分层架构 + 渲染模式修复

**Branch**: `fix/cfc0672-enhancements`
**日期**: 2026-05-28
**基于**: Commit 1df358c + GenDB 目标图对比分析

---

## 一、已完成的修复

### 修复 1 — 渲染器 4 模式支持 (`to-svg-icons.ts`)

**问题**: 所有非 group 叶节点只有 2 种模式（labelOnly / boxed），82% 的节点被框包裹，
看起来不像论文图。GenDB 目标图实际有 3 种视觉风格混合使用。

**修复**: 引入 `nodeStyle` 字段，4 种渲染模式:
- `"label"` → 裸文字（"Join Pattern", "Selectivity"）
- `"tag"` → 小彩色药丸（"Filter" 红, "Join" 绿, "Table" 灰）
- `"box"` → 白框 + 图标 + 标签（默认，主要组件）
- `undefined` → 自动检测（保持向后兼容）

新增 `fillColor` / `strokeColor` 字段支持 tag 和 box 节点的颜色覆盖。

### 修复 2 — LLM Prompt 指导混合样式 (`per_region_generator.py`)

**问题**: LLM 给每个节点都加 iconHint + height:50，导致全部走 boxed 分支。

**修复**: 在 prompt 中显式文档化三种节点样式，指导 LLM 使用 ~60% box / ~25% label / ~15% tag 的混合比例。

### 修复 3 — Pydantic Schema 扩展 (`schemas.py`)

新增 `nodeStyle`, `fillColor`, `strokeColor`, `labelOnly` 字段到 ElkNode model，
确保 LLM 生成的 JSON 不会被 Pydantic 序列化丢弃。

### 修复 4 — SAM3 连接复用 (`recursive_segmenter.py`)

**问题**: `_sam3_detect()` 每次调用都新建 `gradio_client.Client()`，
145 次递归 = 145 次 handshake，浪费 ~7 分钟网络开销。

**修复**: 改为使用 `SAM3Client` 单例实例，通过模块级 `_shared_sam3_client` 复用连接。
消除了重复实现，也统一了 HF token 处理逻辑。

### 修复 5 — 递归分割临时文件清理 (`recursive_segmenter.py`)

**问题**: `/tmp/_sub_*.png` 在递归结束后未清理，长期运行的服务器积累垃圾文件。

**修复**: 在 `recursive_segment()` 返回前 glob 清理所有 `_sub_*.png` 临时文件。

---

## 二、仍需后续 Claude 处理的问题

### P1 — crop_b64 从灰化图裁剪（未修复）
`recursive_segment_with_mask()` 中 `_recurse` 闭包捕获的 `image` 是 guide image（灰化版本），
导致 crop_b64 带灰色背景。需要传入原图引用。

### P1 — 三条分层路径缺质量仲裁（未修复）
CCL→SAM3 回退判定纯粹按层数量，应改为层覆盖率。

### P2 — to-svg.ts 同步更新（未修复）
`to-svg.ts` 也有类似的 2-mode 渲染逻辑，需要同步 4-mode 改动。
