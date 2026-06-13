# upstream/ — Fork Registry

本目录存放 Astro-SVGFigure 项目依赖的所有上游 fork，按 AT（ActiveTheory）技术分层归类。

---

## 渲染引擎层 (Rendering Engine Layer)

### `pixijs-engine/`
- **来源**: https://github.com/pixijs/pixijs
- **版本**: pixi.js v8.19.0
- **用途**: 核心 2D WebGL/WebGPU 渲染引擎。AT 基于此构建 SVG 矢量图形的 GPU 加速渲染管线，支持 DisplayObject 树、Ticker 调度、Spritesheet 动画。
- **AT 技术层**: `core/rendering` — CellRenderer、SVGFigureRenderer 的底层依赖

### `pixijs-filters/`
- **来源**: https://github.com/pixijs/filters (仅 src + package.json，精简 fork)
- **版本**: pixi-filters v6.1.5
- **用途**: PixiJS 官方滤镜集合的精简版本，仅保留 src 源码，去掉 demo/examples/scripts，用于生产构建集成。
- **AT 技术层**: `core/effects` — 用于 CellFilter、后处理 pass

### `pixijs-filters-v2/`
- **来源**: https://github.com/pixijs/filters (完整 fork，含 demo/examples)
- **版本**: pixi-filters v6.1.5
- **用途**: 与 `pixijs-filters` 同源同版本，保留完整仓库结构（demo、examples、scripts、tsconfig）。用于滤镜开发调试和原型验证，不直接进入生产包。
- **差异说明**: `pixijs-filters` 是生产精简版（只含 src），`pixijs-filters-v2` 是开发完整版（含 demo/examples/scripts）。
- **AT 技术层**: `dev/effects-lab` — 滤镜原型开发、视觉调试

### `pixijs-ui/`
- **来源**: https://github.com/pixijs/ui
- **版本**: @pixi/ui v2.3.2
- **用途**: PixiJS 官方 UI 组件库（Button、Slider、ScrollBox、Input 等）。AT 用于在 WebGL canvas 内构建编辑器控件和 HUD 元素。
- **AT 技术层**: `editor/ui` — 场景编辑器内嵌 UI 组件

---

## 动画编排层 (Animation Orchestration Layer)

### `theatre-js/`
- **来源**: https://github.com/theatre-js/theatre
- **结构**: monorepo，含 `core/`、`dataverse/`、`studio/`
- **用途**: 专业级动画序列器，提供关键帧时间线、reactive 数据状态（dataverse）和可视化编辑 Studio。AT 用于驱动 SVGFigure 的补间动画轨道和 pubsub 事件同步。
- **AT 技术层**: `animation/timeline` — Cell 动画状态机、pubsub-loop 事件驱动

### `animation-editor/`
- **来源**: ActiveTheory 内部 fork（thing-editor 衍生）
- **用途**: 基于 React/TypeScript 的节点式动画编辑器，支持 expression 表达式、area 布局和 nodeEditor 图形编辑。AT 定制版，专为 SVGFigure 动画工作流设计。
- **AT 技术层**: `editor/animation` — 动画图编辑、表达式求值

### `thing-editor/`
- **来源**: https://github.com/Megabyteceer/thing-editor
- **用途**: HTML5 游戏编辑器，AT 作为 animation-editor 的上游参考实现保留，用于对比 diff 和回溯补丁。
- **AT 技术层**: `reference/editor` — 历史参考，不直接集成

---

## Unreal 渲染桥接层 (Unreal Bridge Layer)

### `unreal-renderer/`
- **来源**: ActiveTheory Unreal 渲染桥（UE4 基线）
- **用途**: Unreal Engine 4 版渲染桥，负责将 PixiJS 场景数据序列化并通过 WebSocket/共享内存传输给 UE4 渲染进程。
- **AT 技术层**: `bridge/unreal-ue4` — 传统项目兼容层

### `unreal-renderer-ue5/`
- **来源**: ActiveTheory Unreal 渲染桥（UE5 重构）
- **用途**: 针对 UE5 Nanite/Lumen 特性重构的渲染桥，支持更高保真度的离线渲染输出。
- **AT 技术层**: `bridge/unreal-ue5` — 新项目主线渲染桥

---

## 资产处理层 (Asset Pipeline Layer)

### `activetheory-svg2msdf/`
- **来源**: ActiveTheory 内部工具
- **用途**: 将 SVG 矢量图形转换为 MSDF（Multi-channel Signed Distance Field）纹理，供 PixiJS 引擎高质量缩放渲染使用。是 SVGFigure 名称的技术来源之一。
- **AT 技术层**: `pipeline/assets` — SVG → MSDF 构建步骤

---

## 通信层 (Communication Layer)

### `apollo-cyber/`
- **来源**: Apollo/Cyber RT 通信框架 fork
- **用途**: 高性能 pubsub 消息总线，AT 用于 Cell 间跨线程事件分发（cell-pubsub-loop 分支核心依赖）。
- **AT 技术层**: `core/pubsub` — CellBus、跨 Worker 消息路由

---

## 状态总览

| 目录 | 状态 | 版本 / 来源 |
|------|------|-------------|
| `pixijs-engine/` | ✅ 完整 | pixi.js 8.19.0 |
| `pixijs-filters/` | ✅ 精简版 | pixi-filters 6.1.5 (src only) |
| `pixijs-filters-v2/` | ✅ 完整版 | pixi-filters 6.1.5 (with demo/examples) |
| `pixijs-ui/` | ✅ 完整 | @pixi/ui 2.3.2 |
| `theatre-js/` | ✅ 完整 | core + dataverse + studio |
| `animation-editor/` | ✅ 完整 | AT 内部 fork |
| `thing-editor/` | ✅ 参考 | upstream 参考实现 |
| `unreal-renderer/` | ✅ UE4 | AT 渲染桥 |
| `unreal-renderer-ue5/` | ✅ UE5 | AT 渲染桥重构版 |
| `activetheory-svg2msdf/` | ✅ 完整 | AT 内部工具 |
| `apollo-cyber/` | ✅ 完整 | pubsub 通信框架 |

---

*维护者: xiaodi #19 — 最后更新: cell-pubsub-loop 分支*
