# 为什么最终出图看不到 sprite 设计 —— 架构诊断

**结论先行**：你的观察完全正确。圆角矩形(sprite)确实被放进了原始 skeleton 草图，而不是替换它。原因不是输入里没有算子链，而是**整条 sprite 流水线从未接入真实出图路径**——它是一座孤岛。

---

## 一、实测确认的三处脱节

### 脱节 1：后端 `generate_layered_topology` 不调 sprite pipeline
`layered_pipeline.py::generate_layered_topology()` 在 line 306 调了 `compose()`，line 339 返回，**中间从不调用 `run_sprite_pipeline()`**。M218 计划里"第三位 Claude 把 sprite pipeline 接进主流程"——只把它写成了独立可调用函数，从没在主流程里真正调用。

### 脱节 2：`run_sprite_pipeline` 被零处调用
全仓库搜索 `run_sprite_pipeline`：**除了它自己的定义，没有任何地方 import 或调用它**。M210–M224 约 3000 行代码（分类器/prompt/拼版/序列帧/对齐/矢量化/缓存/导出/注入）全部完整、全部测试通过、但**全部是死代码**——没有任何 serving endpoint 触发它。

### 脱节 3：渲染输入是「裸 ELK」，sprite 字段永远为空
出图入口 `src/lib/elk/layout.ts` line 204：`skeletonSvg = elkToSvg(layouted)`，喂给渲染器的 `layouted` 是 **ELK 原始布局结果**，没有 `spriteManifest`、没有 `renderMode`、没有 `spriteRef`。我在 `to-svg.ts` 里加的 M214(sprite 分支)和 M207(算子分支)**永远不触发**，因为它们的条件 `node.renderMode === 'sprite'` / `node.isOperator` 在真实输入里从来不为真。

### 附带问题：散点灰块出现在最终图
你看到每个框里的模糊灰色 blob，是 `to-svg.ts` line 240–251 的 `numScatter`（3–6 个 `opacity 0.04–0.12` 的灰色装饰矩形）。它原是"有机草图"风格选择，但在干净的学术图里读起来像噪点/伪影。**两个渲染器（`to-svg.ts` 和 `to-svg-icons.ts`）都有这段散点，且都零 sprite 支持**——所以无论"最终图"用哪个渲染器，都是 skeleton 级。

---

## 二、为什么单元测试没抓到

所有 sprite 测试（M210–M224）都是这样跑的：手动构造 ELK → 手动调 `run_sprite_pipeline` → 手动把注入后的 graph 喂给渲染器。**测试验证了模块本身正确，但没验证模块被产品流程调用**。这是典型的"单元完美、集成断裂"——孤岛代码的测试只能证明孤岛内部没问题，证明不了孤岛连上了大陆。

---

## 三、修复方案（需要你拍板的架构决策）

真实流程目前是：`/api/topology`（文生 ELK）→ 前端 → `/api/layout`（ELK → skeletonSvg/iconSvg）→ 显示。sprite pipeline 要接进去，有两条路，请选：

### 方案 A：后端贯通（推荐，生产级）
把 sprite 生成接进后端主流程，让"最终图"天然带 sprite：
1. `generate_layered_topology` 在 `compose()` 后调 `run_sprite_pipeline`，把 `spriteManifest` 写进返回的 ELK。
2. 新增/改造一个 endpoint，返回**带 spriteManifest 的 ELK**给前端。
3. 前端 `layout.ts` 把这个 ELK（而非裸 ELK）喂给 `elkToSvg`——M214/M207 分支立刻激活。
4. 需要真实 Gemini API key（sprite 是 AI 生图）。无 key 时全部优雅回退 text（已实现）。
- **代价**：跨语言数据流要打通（Python 出 manifest → TS 渲染）；要真实 API。
- **收益**：最终图真正呈现 sprite 序列 + 矢量算子，就是你第一轮设想的样子。

### 方案 B：先去掉散点噪点（最小改动，立即见效）
不接 sprite，先让 skeleton/final 图干净：
1. 给 `to-svg.ts` 和 `to-svg-icons.ts` 的散点装饰加一个开关，"final/导出"模式关掉散点。
2. 图立刻从"带灰块的草图"变成"干净的盒+文字+icon"。
- **代价**：仍看不到 sprite 微差小图（那需要方案 A）。
- **收益**：30 分钟内最终图就专业了，去掉你最刺眼的问题。

### 方案 A+B（理想）
先 B 去噪点止血，再 A 贯通 sprite。

---

## 四、我的建议

作为生产级产品，**A+B**。但 A 需要：(1) 你确认要走"后端生成 sprite"这条跨语言路径，(2) 服务器有 Gemini key 能实跑。在你确认前，我可以**先做 B**（纯前端、零依赖、零风险、立即改善），同时把 A 的后端桥接（`generate_layered_topology` → `run_sprite_pipeline` 接线）作为下一步。

散点噪点是无论如何都该修的——它不该出现在任何"最终"产物里。
