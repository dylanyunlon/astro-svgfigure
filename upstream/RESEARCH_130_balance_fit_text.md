# RESEARCH — xiaodi #130
## `@activetheory/balance-text` & `@activetheory/fit-text`

---

### balance-text — 文本平衡排版算法

`balance-text` v1.0.2 是对原生 CSS `text-wrap: balance` 的渐进增强封装。运行时首先通过 `CSS.supports('text-wrap', 'balance')` 检测浏览器能力：若元素已具备原生支持则直接跳出（避免重复计算）；否则启动**同步二分搜索**，在 `[width/2, width]` 区间内逐步收窄 `el.style.maxWidth`，以 `container.clientHeight === 初始高度` 为条件判断是否触发换行，最多迭代 2000 次，找到恰好不增加行数的最小宽度。`ratio` 参数允许在搜索结果与全宽之间线性插值（`upper * ratio + width * (1 − ratio)`），从而实现"部分平衡"而非强制最窄。最终效果是各行宽度趋于均等，消除末行孤字（orphan/widow）。

---

### fit-text — 响应式字号缩放

`fit-text` v1.0.2 的核心逻辑分两阶段：**字号递减** + **字符裁剪**。首先从 `computedStyle.fontSize` 出发，每轮递减 1px 并检测 `el.scrollWidth ≤ maxWidth && el.offsetHeight ≤ maxHeight`，直至文字适配边界或触及 `minFontSize`（默认 10px）。若仍溢出则进入裁剪阶段，逐字符截断并追加省略符 `clip`（默认 `'...'`）或 `htmlClip`。通过 `box` 参数绑定容器、`boxMultiplier` 缩放系数、`flip` 轴交换、`singleLine` 单行模式等选项，覆盖多种响应式场景。`afterFit` 回调返回 `{ fontSize, text }`，供外部系统同步状态。

---

### 两者在 activetheory.net 排版系统中的协作

两个库在时序上形成 **"定尺寸 → 定形状"** 的两阶段管线：先由 `fitText` 将字号压缩至刚好填充容器（`afterFit` 返回最终 `fontSize`），再以该宽度为基准调用 `balanceText`，使多行文字的换行点趋于视觉对称。`fitText` 解决"放得下"，`balanceText` 解决"排得好"——共同保证在任意视口尺寸下输出紧凑、比例协调的文字块。

---

### 应用于 astro-svgfigure 的 SVG `<text>` 自动排版

SVG `<text>` 不支持 CSS `text-wrap` 且换行需手动插入 `<tspan>`，因此需要**代理 HTML 元素**策略：

1. 在隐藏层创建与 SVG 视口等宽的 `<div>`，填入目标文本；
2. 依次调用 `fitText({ el, box })` → `balanceText({ el, useParent: true })`，由 `afterFit` 取得 `fontSize`；
3. 用 `Range` 或行高计算将 `div` 内的换行点映射为 `<tspan x="0" dy="1.2em">` 分段；
4. 将 `fontSize` 写入 SVG `<text font-size="...">` 属性，`<tspan>` 列表构成最终多行输出。

此方案使 SVG 图形中的说明文字能随 `viewBox` 自动缩放并保持视觉平衡，无需手动调参。