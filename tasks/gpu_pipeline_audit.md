# GPU Pipeline 审计报告 — M1311c

管理者 (claude-manager) 直接审计。2026-06-30。

## 核心发现：管线没有断,但每个 pass 都在 catch 里静默吞掉错误

`_initPasses()` 里有 **20+ 个 try/catch 块**,每个都 `console.warn(..., e)` 然后继续。
如果 PBRCellGPU 的 shader 编译失败, `this.pbr` 就是 `null`。
后续 frame() 里检查 `if (this.pbr)` — null 就跳过。
所以 drawCalls = 0 不是 bug,是"所有 pass 都失败了然后被静默跳过"。

## 数据流追踪

```
world/index.astro → loadGPUScene()
  ↓ fetch('/api/composite-params')  ← 这个 API 在 dev 模式下可能不存在
  ↓ fallback: fetch('/channels/composite_params.json') ← 静态文件
  ↓ parse: raw.cells (dict of dicts) → CellData[]
  ↓ parse: raw.edge_routes → EdgeData[]
  
GPURenderLoop.setScene(cells, edges)
  ↓ 存到 this.cells / this.edges
  
frame()
  ↓ PBR pass: pbr.render(cells, camera) → 输出到 FBO cellTexture
  ↓ Edge pass: edge.render(edges) → 输出到 FBO edgeTexture  
  ↓ Bloom pass: bloom.render(cellTexture) → bloomTexture
  ↓ Composite pass: composite.render({cell, edge, bloom, ...}) → screen
```

## 问题 1：composite_params.json 数据格式

M1308 修了 cells 是 dict 的问题。loadGPUScene 正确用 `Object.entries(rawCells)` 遍历。
但是 `bbox` 取值路径: `cv?.agent_params?.bbox` — 如果 agent_params 不存在就 fallback 到 `{x:0,y:0,w:100,h:50}`。

需要验证: composite_params.json 里每个 cell 的 agent_params.bbox 是否真的存在。

## 问题 2：PBR shader 可能在浏览器里因为精度/扩展问题编译失败

pbr-gpu-pass.ts 的 FRAG_SRC 用了:
- `#version 300 es`
- `layout(location=0) out vec4 gAlbedo;` (MRT output)
- `EXT_color_buffer_float` 扩展

如果 `EXT_color_buffer_float` 不可用 → MRT FBO 创建失败 → PBR 变 null。

## 问题 3：frame() 里 draw 顺序错误

```typescript
// gpu-render-loop.ts frame() 里的顺序:
// 1. FXScene.renderFrame()  ← 渲染到 offscreen
// 2. Nuke.render()          ← 后处理
// 3. CellInstanceManager.draw()  ← 画 cell
```

AstroPipeline.ts 里 cellManager.draw() 在 Nuke.render() **之后**调用。
但 draw() 画到屏幕的时候, Nuke 已经 blit 了一帧空白到屏幕。
cell 画在了已经被 Nuke 覆盖的画面上。

## 根本原因总结

1. **20+ pass 中任何一个 shader 编译失败 → 整个 pass = null → 跳过**
2. **没有任何 pass 做过独立验证** — 从没有人在浏览器里看过某个 pass 单独的输出
3. **composite pass 吃掉了所有 null 输入** — 用 placeholder 1x1 纹理代替, 结果就是黑色

## 修复路径

### 第一步：让 PBR 单独工作
把 pbr-gpu-pass.ts 的 shader 复制到 verify-pbr.html,
hardcode 一个 cell 的数据, 看能不能渲染出 SDF rounded-rect。

### 第二步：如果 PBR 能工作, 问题就在 composite
composite-gpu-pass.ts 接收 6+ 个纹理输入, 如果任何一个是 placeholder,
混合结果就是几乎全黑。

### 第三步：SwissGL 最小方案
用 google/swissgl (<1000行) 替代整个 208 模块的管线,
直接 SDF 渲染 58 个 cell。这是 Unseen Studio "When Cells Collide" 的做法。
