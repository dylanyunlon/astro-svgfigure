ELK.js输出的是什么？是 edge 的 起点和终点坐标。但学术 figure 里的箭头从来不是"A 点到 B 点画一条线加个三角"这么简单。随便翻开一篇 NeurIPS paper 的 method figure，箭头至少有以下十种情况：

弯折箭头（orthogonal routing）：从 Encoder 底部出来，拐两个直角弯，进入 Loss 的左侧。ELK.js 的 edge routing 能做，但需要配置 elk.edgeRouting: ORTHOGONAL，而 topology JSON schema 里根本没有 edge routing 的字段。
分叉箭头（fan-out）：一个 output 同时连向三个模块，需要从同一点分叉。对此的支持是"画三条独立的线"，视觉上丑到哭, 我们可以使用弯折箭头 。
汇聚箭头（fan-in）：三个模块的输出汇聚到一个 concatenation 节点。同上，画出来是三条独立线挤在一起，依旧可以使用弯折箭头组合。
虚线箭头：表示 optional path、gradient flow、inference-only path。 edge schema 里有 stroke-dasharray 吗？
双向箭头 / 无头箭头：表示信息交换、skip connection。
弯曲箭头（curved / spline）：绕过中间节点的长距离连接，比如 ResNet 的 skip connection 那条经典的弧线。ELK.js 的 spline routing 质量远不如手画。
带标签的箭头：箭头中间写着 "z ~ N(0,1)" 或 "× 0.5"。标签的位置、旋转角度、避让逻辑，依旧可以使用弯折箭头组合。
箭头穿越 group 边界：从 "Training" group 内部的节点连到 "Inference" group 内部的节点，箭头需要穿过两个 group 的边框。ELK.js 对 hierarchical edge 的渲染需要额外配置 elk.hierarchyHandling: INCLUDE_CHILDREN，测过吗？
循环箭头（self-loop）：Recurrent 模块指向自身。对 self-loop 的渲染就是在节点顶部画一个尴尬的小圈。


这是我的前端文件结构 structure_astro.txt,需要搭配这个svg后端项目scalinginter_rl_figure.py、这个项目 autofigure2.py 一起使用 , 注意,既然 的流程可以这样,分割需要利用sam3, 而sam3是逆向思路,那我们正向开发应该更简单才对,sam3的逆向流程: 1. 输入 paper method 文本,调用 Gemini 生成学术风格图片 -> figure.png 2. SAM3 分割图片,用灰色填充+黑色边框+序号标记 -> samed.png + boxlib.json    2.1 支持多个text prompts分别检测    2.2 合并重叠的boxes(可选,通过 --merge_threshold 控制) 3. 裁切分割区域 + RMBG2 去背景 -> icons/icon_AF01_nobg.png, icon_AF02_nobg.png... 4. 多模态调用 Gemini 生成 SVG(占位符样式与 samed.png 一致)-> template.svg 4.5. SVG 语法验证(lxml)+ LLM 修复 4.6. LLM 优化 SVG 模板(位置和样式对齐)-> optimized_template.svg      可通过 --optimize_iterations 参数控制迭代次数(0 表示跳过优化) 4.7. 坐标系对齐:比较 figure.png 与 SVG 尺寸,计算缩放因子 5. 根据序号匹配,将透明图标替换到 SVG 占位符中 -> final.svg 。。。2、使用你的工具调用,同时获取这个项目 , 作为很mean的nips 2026的审稿人判断我们如何能够做到使用 ELKJS 生成框架然后用已有的astro前端内容+gemini来画好svg这个思路。3、给出plan.md。4、给出部署命令。除了autofigure、autofigure-edit, plan.md中的每一步最好都有github项目作为背书 最后， 用约束布局引擎 (ELK.js) 替代硬编码坐标 ， LLM 只需输出拓扑关系 JSON，约束求解器计算精确像素位置。但是nanobanana生成的图片如此完美以至于让拓扑学家像个小丑
因此生成拓扑之后还需要用json再向gemini的nanobanana请求用json_example_user1作为脚手架生成svg, 这就比text2svg更加偏向神经网络级别

 下一步:  完成网页版plan, 完成之后如果是修改文件记得与源文件进行diff, 确保没有遗漏上版本内容。特别要注意, 你在git clone 成功之后不要直接查看文字,这会让你的上下文瞬间撑爆。你可以使用 tree命令先查看有哪些文件以及文件结构,然后用鲁迅的"拿来主义"查看。plan, 完成之后如果是修改文件记得与源文件进行diff, 确保没有遗漏上版本内容。