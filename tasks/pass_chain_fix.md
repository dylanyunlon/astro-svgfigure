# Pass Chain Fix — M1314b

作者: claude-worker-b  
日期: 2026-07-01  
分支: cell-pubsub-loop

## 问题总结

/world 页面渲染输出为全黑 / 空白，drawCalls 统计为 0。  
根据 `tasks/gpu_pipeline_audit.md` 的诊断，有三个根本原因。

---

## 修复文件

- `src/lib/sph/gpu-render-loop.ts`
- `src/lib/sph/composite-gpu-pass.ts`

---

## 修复 1 — PBR pass try/catch + _pbrSucceeded 状态追踪

**文件**: `gpu-render-loop.ts`

**问题**: 审计报告 line 305 写着 "no try/catch, crash if shader fails"。
PBR shader 若在浏览器因 EXT_color_buffer_float 不可用、精度问题或 MRT FBO 创建
失败而抛出异常，整个 frame() 会崩溃，后续所有 pass 包括 composite 都不执行。

**修复**:
- 为 PBR pass 加 `try/catch`；失败时在前 5 帧打印清晰的 `console.error`
- 新增私有字段 `_pbrSucceeded: boolean`，每帧初始化为 false
- PBR 渲染成功（`renderCells()` 不抛异常）→ 设为 true
- CellMesh 覆盖 cellTex 时也设为 true（meshTex 同样是真实内容）

---

## 修复 2 — Composite 接收 hasCellContent 参数

**文件**: `composite-gpu-pass.ts` + `gpu-render-loop.ts`

**问题**: Composite 的 `uShadow` 输入是 1x1 placeholder（shadow pass 失败被吞掉）。  
Shader 里 `ambientShadow = max(shadowMask, 0.25)`，即使无阴影也至少 0.25。  
`cellColor.rgb * ambientShadow` 在 cell 是 placeholder 时把背景压暗到接近黑色。

**修复**:
- `composite-gpu-pass.ts` shader 新增 `uniform float uHasCellContent`
- Cell 混合改为:
  ```glsl
  float shadowApply = uHasCellContent > 0.5 ? ambientShadow : 1.0;
  composite = mix(composite, cellColor.rgb * shadowApply, cellColor.a);
  ```
- `CompositeGPU.render()` 新增可选参数 `hasCellContent = true`（默认向后兼容）
- `gpu-render-loop.ts` composite 调用处传入 `this._pbrSucceeded`

---

## 修复 3 — NukePass 不覆盖 PBR 输出

**文件**: `gpu-render-loop.ts`

**问题**: `NukePass.render(gl)` 直接写入 default framebuffer（null FBO），  
若它在 composite 之前执行并清除了画布，composite 的 drawArrays 就画在了被清空的帧上。  
（审计报告 "问题 3: frame() 里 draw 顺序错误"）

**修复**:
```typescript
if (this.nukePass && !this._pbrSucceeded) {
  try { this.nukePass.render(gl); } catch (_) { }
}
```
当 PBR 已经成功渲染到 FBO 并交给 composite pass，NukePass 跳过，
避免它的 blit 覆盖 composite 输出。

---

## 修复 4 — Composite 失败时 emergency blit

**文件**: `gpu-render-loop.ts`

**问题**: composite pass 自身若初始化失败（try/catch 吞掉），整帧输出为空。

**修复**: composite 的 else 分支和 catch 分支都改为：
- 若 `_pbrSucceeded && cellTex`，执行 `_blitTexture(cellTex, W, H, TEX.CELL)`
- 这是"minimum viable render"：至少能看到 PBR 渲染的 cell quads

---

## 修复 5 — Constructor 末尾 Pass Status 日志

**文件**: `gpu-render-loop.ts`

每次 `_initPasses()` 结束后输出各 pass 的初始化状态：

```
[GPURenderLoop] pass init status: {
  pbr: true, composite: true, bloom: true, shadow: true,
  edge: true, fluid: true, particle: true, glass: true,
  cellMesh: true, lumenGI: false, nukePass: false
}
```

便于在浏览器 DevTools Console 立即确认哪些 pass 真正就绪。

---

## 测试方法

1. 打开 /world 页面，DevTools → Console
2. 确认看到 `[GPURenderLoop] pass init status: { pbr: true, composite: true, ... }`
3. 若 PBR 失败，console.error 会明确显示错误原因（EXT 缺失 / shader 编译失败）
4. 无论 PBR 成功与否，页面都应显示内容（至少 composite background grid）
5. PBR + composite 均成功时，应看到 SDF rounded-rect 细胞图形

## 未解决问题

- 若浏览器不支持 `EXT_color_buffer_float`，PBR MRT FBO 会失败。
  需要 `pbr-gpu-pass.ts` 的 fallback 路径（单 render target 降级）。
  这超出本 M1314b 任务范围，记录在 tasks/gpu_pipeline_audit.md 修复路径第一步。
