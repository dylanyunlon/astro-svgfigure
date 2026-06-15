Now I have everything I need. Writing the RESEARCH document.# RESEARCH — `@activetheory/split-text` · xiaodi #127

---

## 文本拆分机制

`SplitText` 接受一个 DOM 元素与 `type: 'lines,words,chars'` 选项组合。核心方法 `splitElement()` 递归遍历子节点：**words** 阶段以 `/\s+/` 分割文本节点；**chars** 阶段调用 `Array.from(text)` 正确处理 Unicode 与 emoji；**lines** 阶段先由 `detectLinesTop()` 读取每个 word span 的 `offsetTop`（以 `lineThreshold × fontSize` 为容差）将词归并至同行，再由 `attachBr()` 在行边界插入 `<br>`，最终 `splitLines()` 把每行包裹进 `<span class="line" style="display:block">`。

---

## DOM 操作策略

所有子节点先收集进 `DocumentFragment` 批量写入以减少重排。每个 char/word 被包为 `<span class="char/word" style="display:inline-block" aria-hidden="true">`；同时追加 `.sr-only` 隐藏 span 保证无障碍语义。`originals[]` 缓存原始 `innerHTML`，`revert()` 一键还原。集成 `@activetheory/balance-text` 实现视觉均衡断行；`handleCJT` 标志支持中日韩泰字符逐字拆分。

---

## 与 GSAP SplitText 的区别

| | Active Theory | GSAP SplitText |
|---|---|---|
| 许可 | 开源 (MIT) | 需 GSAP Club |
| 文本均衡 | 内置 balance-text | 无 |
| CJT 支持 | `handleCJT` 选项 | 有限 |
| 无障碍 | 内置 `.sr-only` | 手动 |
| 依赖 | 零运行时依赖 | 捆绑 GSAP |

两者均以 `offsetTop` 检测行边界，API 形态相似（`.chars / .words / .lines` 数组）。

---

## 在 astro-svgfigure 中的逐字符动画渲染

`onMount` 中先 `await isFontReady()` 确保字体加载完毕，再：

```js
const st = new SplitText(labelEl, { type: 'chars,words' });
gsap.from(st.chars, { y: 20, opacity: 0, stagger: 0.03 });
```

每个 SVG 标签文字经拆分后，`chars` 数组即为可动画目标；`word.__lineIndex` 元数据支持按行错列入场；组件卸载时 `st.revert()` 清理 DOM。

---

## WorkLabelPlayground 的文字效果依赖

`activetheory.net` WorkLabelPlayground 将作品标签以 `type: 'chars'` 拆分，每个字符 span 绑定 pointer/scroll 驱动的 GSAP timeline——悬停时字符以随机 `y/rotation` 爆散，离开时收敛归位。`__lineIndex` 用于多行标签的分层动画编排，`noBalance: true` 在短词单行场景下跳过均衡计算优化性能。`isFontReady()` 确保自定义字体渲染完毕后再拆分，规避行高测量误差。