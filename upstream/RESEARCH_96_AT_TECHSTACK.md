# Research 96 — Active Theory (activetheory.net) 技术栈逆向分析

**作者:** 管理者 Claude (代招聘 xiaodi #96-#105)
**日期:** 2026-06-15
**分支:** cell-pubsub-loop
**数据来源:** F12 DevTools → Sources面板 + Console全局对象 + WebGL2 extensions

---

## 网站架构

| 层级 | 文件 | 用途 |
|------|------|------|
| 入口 | `assets/js/app.1780406240914.js` | 主应用bundle |
| 模块 | `assets/js/modules.1780406240914.js` | 模块bundle |
| 引擎 | `assets/js/hydra/` | Hydra 3D Engine（闭源） |
| Worker | `hydra-thread.js` × 5 | 多线程Worker池 |
| WASM | `8d00806a`, `001cf6fe`, `0011474e` | 计算密集型模块 |
| 字体 | `NBArchitektStd-Regular-export` | 品牌字体 |

## 全局对象分析（window.*）

### Shader系统
- `WallShader` — 墙面PBR材质
- `WaterCeilingShader` — 水面反射/折射
- `WorkItemShader` — 作品卡片渲染
- `WorkItemUIShader` — UI元素GPU渲染

### 粒子/几何
- `WorkDetailParticles` — GPU粒子效果
- `WorkDetailCube` — 立方体几何

### UI系统 (UIL = lo-th/uil)
- `UILPanel`, `UILPanelToolbar` — 调试面板
- `UILInputButton`, `UILInputNumber` — 输入控件
- `UILMemory`, `UILPerformance`, `UILPerformanceItem` — 性能监控
- `UILTabs`, `UILTabsContentItem`, `UILTabsNavItem` — 标签页

### 架构核心
- `Stage` — 场景管理器（类似Three.js Scene）
- `ViewController` — MVC控制器
- `Main` — 入口点
- `hydraObject` — Hydra引擎单例

### 第三方（仅1个）
- `QRious` — QR码生成

## WebGL2 Extensions（已启用）

关键扩展：
- `EXT_color_buffer_float` — HDR浮点FBO
- `EXT_float_blend` — 浮点混合
- `EXT_disjoint_timer_query_webgl2` — GPU性能计时
- `KHR_parallel_shader_compile` — shader并行编译
- `OVR_multiview2` — VR多视图渲染
- `WEBGL_multi_draw` — 批量绘制调用

## 对应开源upstream映射

| AT技术 | 开源替代 | upstream路径 |
|--------|---------|-------------|
| Hydra核心Renderer | ogl (oframe) | `upstream/ogl/` |
| UIL调试系统 | lo-th/uil | `upstream/uil/` |
| WebGL底层 | nanogl | `upstream/nanogl/` |
| Shader库 | lygia | `upstream/lygia/` |
| Worker通信 | comlink (Google) | `upstream/comlink/` |
| WaterCeilingShader | webgl-water (evanw) | `upstream/webgl-water/` |
| WorkDetailParticles | sketch.js | `upstream/sketch-js/` |
| GLSL噪声 | webgl-noise (ashima) | `upstream/webgl-noise/` |
| GaussianSplats3D | AT自己的fork | `upstream/gaussian-splats-at/` |
| QRious | neocotic/qrious | `upstream/qrious/` |
