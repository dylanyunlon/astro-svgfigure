# M200 系列开发计划 — 组件级 AI 生图 + 矢量合成架构

## 核心思想

不再一次性用 AI 生成整张大图。改为：
1. skeleton 保留大框架（群组容器 + 箭头 + 连接线）
2. 每个叶子节点的图标/插图由 AI 单独生成（序列帧）
3. remove.bg 去背景 → 透明 PNG
4. 透明 PNG 嵌入到 skeleton SVG 的对应位置

这样：大框架的精确度由 ELK 保证，小插图的美观度由 AI 保证，两者合成。

## 开发里程碑

### Phase 1: skeleton 渲染升级（M200-M205）

- **M200**: `to-svg.ts` 散布圆角矩形替代 plain rect ✅ (当前 commit)
- **M201**: `to-svg.ts` icon 大时文字缩小 ✅ (当前 commit)
- **M202**: 删除 SAM3 ✅ (当前 commit)
- **M203**: 群组容器渲染优化 — 彩色淡底 + 虚线边框
- **M204**: 箭头渲染升级 — 贝塞尔曲线、标签定位
- **M205**: 边缘标签（"projection", "feedback" 等）渲染优化

### Phase 2: 组件级 AI 生图 pipeline（M210-M220）

- **M210**: 节点分类器 — 从 ELK JSON 中识别哪些节点需要 AI 生图
  - 有 iconHint 的叶子节点 → AI 生图候选
  - 同类型不同含义的节点 → 序列帧生成（微小差别）
  - 纯文字/标签/运算符 → 矢量渲染，不需要 AI
- **M211**: 节点图标 prompt 生成器 — 从 iconHint + label 生成精准的小图 prompt
  - "ViT Encoder" + iconHint="transformer" → "a small flat vector illustration of a vision transformer encoder block, white background, academic figure style, 128x128"
  - 同类型序列："feature map C×H×W" → 3 张微小差别的特征图小图
- **M212**: 批量小图生成 — gpt-image-2 批量生成 64x64 ~ 256x256 的小图标
- **M213**: remove.bg 批量去背景 — 每张小图去背景得到透明 PNG
- **M214**: 透明 PNG → base64 → 嵌入 SVG `<image>` 标签
- **M215**: 序列帧生成 — 对于 "input feature (C×H×W)" 这类同族节点：
  - 识别同族（相同 iconHint、相似 label 结构）
  - 生成 1 张基础图 + N 个变体 prompt
  - 每个变体只有微小差别（颜色深浅、方向、密度）
- **M216**: 序列帧去背景 + 嵌入
- **M217**: 数学运算符渲染 — ⊗ ⊕ ○ 直接用 SVG circle+path
- **M218**: 连接线标签优化 — "projection", "retrieval" 等标签无框浮动
- **M219**: 全流程集成测试 — structured_data.txt → 完整论文级 SVG
- **M220**: 性能优化 — 并行生成、缓存已生成的小图

### Phase 3: 高级特性（M230-M240）

- **M230**: 填充模式 — hatching/dots/crosshatch（from rough.js research）
- **M231**: 形状注册表 — circle/diamond/hexagon 节点
- **M232**: 竖排文字容器
- **M233**: 内嵌网格（3×4 dot grid 等）
- **M234**: 导出为 PDF（cairosvg）
- **M235**: 导出为 .ai（Adobe Illustrator SVG 兼容）
- **M236**: 交互式编辑器中实时预览小图
- **M237**: 小图缓存 — 相同 iconHint 复用已生成的图
- **M238**: 用户自定义小图上传替换 AI 生成的
- **M239**: 多模型支持 — gpt-image-2 / Gemini / DALL-E 可切换
- **M240**: A/B 测试框架 — 对比不同生图策略的用户满意度

## Claude 分工计划

- Claude #1 (M200-M205): skeleton 渲染升级 ✅ 当前进行中
- Claude #2 (M210-M211): 节点分类器 + prompt 生成器
- Claude #3 (M212-M214): 批量小图生成 + 去背景 + 嵌入
- Claude #4 (M215-M216): 序列帧生成 + 去背景
- Claude #5 (M217-M218): 数学运算符 + 连接线标签
- Claude #6 (M219-M220): 集成测试 + 性能优化
- Claude #7-8 (M203-M205): 群组容器 + 箭头 + 边缘标签
- Claude #9-12 (M230-M233): 填充模式 + 形状注册表 + 竖排 + 网格
- Claude #13-15 (M234-M236): 导出格式 + 编辑器集成
- Claude #16-18 (M237-M240): 缓存 + 用户上传 + 多模型 + A/B测试
- Claude #19-25: 回归测试 + edge case 修复 + 文档
- Claude #26-30: 用户反馈迭代 + prompt 调优
- Claude #31-38: 大规模测试（100+ 论文图） + 生产部署
