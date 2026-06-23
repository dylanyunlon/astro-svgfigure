# 诚实审计: AT+UE 效果还差多远

## 真实资产 (能出像素)

### WebGL 渲染器 — 12 个文件, 3837 行, 446 个 gl 调用
- AstroRenderer.ts (359行, 41 gl) — createShader/createProgram/link 完整路径
- FXScene.ts (385行, 38 gl) — 合成 pass: compile + useProgram + drawArrays
- AstroMesh.ts (304行, 33 gl) — VAO/VBO 绑定 + drawElements
- InstancedMesh.ts (340行, 69 gl) — instanced rendering
- Material.ts (232行, 36 gl) — uniform binding
- Nuke.ts (392行, 39 gl) — 后处理 ping-pong FBO
- NukePass.ts (284行, 39 gl) — 单 pass FBO render
- RenderTarget.ts (179行, 32 gl) — FBO 创建
- OcclusionQuery.ts (476行, 49 gl) — GPU 遮挡查询
- 其他 3 个

### PixiJS 渲染器 — 15 个文件, 真正画东西
- pixi-cell-renderer.ts (2646行, 302 pixi calls) — ★ 主 cell 绘制
- sdf-species-filter.ts (2357行, 136 pixi calls) — Species SDF filter
- cell-batch-renderer.ts (790行, 42 pixi calls) — 批量 cell
- pixi-filters-registry.ts (587行, 166 pixi calls) — Filter 链注册
- sdf-cell-renderer.ts (570行, 63 pixi calls)
- flower-edge-renderer.ts (651行, 11 pixi calls)
- 其他 9 个

### GLSL shader — 28 个文件, 6435 行
- 5 个 cil-*.frag (Species SDF icon)
- edge-spline.frag/vert (边样条)
- pbr-cell-surface.frag (577行, PBR 材质)
- fluid-surface.frag (470行)
- voronoi-membrane/natural.frag
- caustics/curl-trail/cloud-fog/colormap/kuwahara-post 等

### compiled.vs — 8975 行, 172 个 AT 生产 shader
已提取但只作为字符串常量存在, 没有任何一个被 createProgram 真正编译过。

## 空壳 (不能出像素)

### at-*.ts + ue-*.ts — 50 个文件, 61,770 行, 0 个 gl 调用
全部是:
- interface 定义
- export const SHADER_SRC = `...` 字符串
- TODO 占位
- 注释 + 类型声明
- 没有真实的 GPU 调用

### channels/rendering/ — 86 个 .py 文件
Apollo CyberRT 风格的参数声明和 JSON 读写, 不是 GPU 代码。

## 从字符串到像素: 缺失的完整链路

```
compiled.vs 字符串
     │
     ▼
[1] ShaderLoader.parse('{@}name{@}...{@}') → 拆分为单个 shader
     │
     ▼
[2] gl.createShader() + gl.shaderSource() + gl.compileShader()  ← 缺
     │
     ▼
[3] gl.createProgram() + gl.attachShader() + gl.linkProgram()   ← 缺
     │
     ▼
[4] gl.useProgram() + uniform binding (resolution, mouse, time) ← 缺
     │
     ▼
[5] FBO chain: createFramebuffer + bindFramebuffer + drawArrays ← 缺
     │
     ▼
[6] ping-pong: read FBO A → write FBO B → swap                 ← 缺
     │
     ▼
[7] composite: final FBO → screen quad → gl.drawArrays          ← 缺
     │
     ▼
屏幕上的像素
```

AstroRenderer.ts 有 [2][3][4] 的基础设施。
Nuke.ts + NukePass.ts 有 [5][6][7] 的 FBO 管线。
但没有人把 compiled.vs 的 shader 字符串喂进这个管线。

## 下一步: 调试密集型, 不是架构设计

1. 先跑通一个 pass: mousefluid (Navier-Stokes)
   - 用 AstroRenderer.createProgram() 编译 splatShader/advectionShader/pressureShader/divergenceShader
   - 用 Nuke ping-pong FBO 做 pressure Jacobi 迭代
   - 鼠标移动 → splat → advect → 看到流体

2. 加 bloom
   - 编译 HydraBloom + DownSample + UpSample
   - 4 级 FBO mipmap 金字塔
   - 叠到流体上

3. 加 PBR cell surface
   - 编译 PhysicalShader + Lighting + shadows
   - cell body 从 PixiJS flat rect → WebGL PBR mesh

4. 逐步堆叠, 每加一个 pass 都在浏览器里验证
