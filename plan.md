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

 02更新：
 # astro-svgfigure: ELK.js + Gemini NanoBanana 学术 Figure 生成方案

> 正向流程：LLM 输出拓扑 JSON → ELK.js 约束求解精确坐标 → NanoBanana 神经网络级别 SVG 渲染

---

## 0. 核心思路：为什么是 "拓扑 + 神经渲染" 双阶段

**autofigure2.py 的逆向流程**（SAM3 路线）：
text → Gemini 生图 → SAM3 分割 → RMBG2 去背 → LLM 生成 SVG 模板 → 图标替换 → final.svg

**本方案的正向流程**（ELK.js 路线）：
text → LLM 输出拓扑 JSON → ELK.js 约束布局（精确坐标）→ JSON 脚手架 → Gemini NanoBanana 渲染 SVG → Astro 前端展示

**为什么正向更简单**：逆向需要 SAM3 分割 + RMBG2 去背 + 坐标对齐 + 多轮优化，每一步都是误差放大器；正向直接用约束求解器出坐标，NanoBanana 在精确坐标的脚手架上渲染，一步到位。

**为什么还需要 NanoBanana**：ELK.js 只输出像素位置，不负责渲染。plan.md 原文说得好——"nanobanana 生成的图片如此完美以至于让拓扑学家像个小丑"。因此我们用 ELK.js 做骨架，NanoBanana 做皮肤，这就是 **text2topology2svg** 而非简单的 text2svg。

---

## 1. 拓扑 JSON 生成（LLM 层）

**输入**：paper method 文本（与 autofigure2.py 步骤一的输入相同）

**输出**：ELK.js 兼容的拓扑 JSON

**LLM prompt 设计**：只需 LLM 输出节点 + 边的拓扑关系，不需要任何坐标

```json
{
  "id": "root",
  "layoutOptions": {
    "elk.algorithm": "layered",
    "elk.direction": "DOWN",
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.layered.spacing.nodeNodeBetweenLayers": "80",
    "elk.spacing.nodeNode": "60",
    "elk.hierarchyHandling": "INCLUDE_CHILDREN"
  },
  "children": [
    {
      "id": "encoder",
      "labels": [{"text": "Encoder"}],
      "width": 150,
      "height": 60,
      "properties": {
        "icon_hint": "neural network encoder block",
        "style": "rounded_rect",
        "fill": "#4A90D9"
      }
    },
    {
      "id": "training_group",
      "labels": [{"text": "Training Phase"}],
      "layoutOptions": {"elk.algorithm": "layered"},
      "children": [
        {"id": "loss", "labels": [{"text": "Loss"}], "width": 100, "height": 50}
      ]
    }
  ],
  "edges": [
    {
      "id": "e1",
      "sources": ["encoder"],
      "targets": ["loss"],
      "properties": {
        "edge_type": "orthogonal",
        "stroke_dasharray": "",
        "label": "z ~ N(0,1)",
        "marker_end": "arrow"
      }
    }
  ]
}
```

**关键 edge 类型映射**（解决 plan.md 原文提出的十种箭头问题）：

| 学术 figure 箭头类型 | ELK.js 配置 | 备注 |
|---|---|---|
| 弯折箭头（orthogonal） | `elk.edgeRouting: ORTHOGONAL` | ELK.js 原生支持 |
| 分叉/汇聚（fan-out/fan-in） | 多条 ORTHOGONAL edge | 组合 orthogonal routing |
| 虚线箭头 | `stroke-dasharray: "5,5"` in properties | SVG 渲染阶段处理 |
| 双向/无头 | `marker_end/marker_start` properties | SVG 渲染阶段处理 |
| 弯曲（spline/skip-connection） | `elk.edgeRouting: SPLINES` | 针对特定 edge |
| 带标签 | `label` property | ELK.js labels 定位 |
| 穿越 group | `elk.hierarchyHandling: INCLUDE_CHILDREN` | 层级 edge |
| 自循环（self-loop） | self-edge `sources == targets` | ELK.js 自动处理 |

**GitHub 背书**：
- ELK.js: [kieler/elkjs](https://github.com/kieler/elkjs) — Eclipse Layout Kernel 的 JS 端口
- Gemini Structured Output: [google-gemini/cookbook](https://github.com/google-gemini/cookbook) — JSON mode 约束输出

**实现文件**：`src/lib/topology-generator.ts`（调用 Gemini text API，输出拓扑 JSON）

---

## 2. ELK.js 约束布局求解

**输入**：步骤 1 的拓扑 JSON

**输出**：带精确 x, y, width, height, edge sections 的 layout JSON

```typescript
import ELK from 'elkjs/lib/elk.bundled.js';

const elk = new ELK();

export async function computeLayout(topologyJson: object): Promise<object> {
  const layouted = await elk.layout(topologyJson);
  // layouted.children[i].x, .y — 精确像素坐标
  // layouted.edges[i].sections[j].startPoint, endPoint, bendPoints
  return layouted;
}
```

**ELK.js 输出的关键字段**：
- `node.x, node.y` — 节点左上角精确坐标
- `edge.sections[].startPoint` — edge 起点坐标
- `edge.sections[].endPoint` — edge 终点坐标
- `edge.sections[].bendPoints` — 弯折点坐标数组（orthogonal routing）
- `label.x, label.y` — 标签精确位置

**GitHub 背书**：
- elkjs-svg (JSON → SVG 直接渲染): [EmilStenstrom/elkjs-svg](https://github.com/EmilStenstrom/elkjs-svg)
- HDElk (ELK.js + SVG.js 渲染): [davidthings/hdelk](https://github.com/davidthings/hdelk)
- React Flow + ELK.js 集成: [xyflow/xyflow](https://github.com/xyflow/xyflow) — 参见 examples/layout/elkjs

**实现文件**：`src/lib/elk-layout.ts`

---

## 3. JSON 脚手架生成（ELK layout → NanoBanana Scaffold）

**输入**：步骤 2 的 layout JSON

**输出**：json_example_user1 风格的渲染指令 JSON（给 NanoBanana 做 scaffold）

这一步是核心创新：将 ELK.js 的精确坐标转化为 NanoBanana 能理解的渲染指令。

```json
{
  "canvas": {"width": 1200, "height": 800},
  "nodes": [
    {
      "id": "encoder",
      "label": "Encoder",
      "x": 120, "y": 50,
      "width": 150, "height": 60,
      "style": {
        "fill": "#4A90D9",
        "stroke": "#2A5A99",
        "corner_radius": 8,
        "icon_hint": "neural network encoder block with stacked layers",
        "font": "14px sans-serif",
        "text_color": "white"
      }
    }
  ],
  "edges": [
    {
      "id": "e1",
      "path": "M 195 110 L 195 180 L 300 180 L 300 250",
      "style": {
        "stroke": "#333",
        "stroke_width": 2,
        "marker_end": "arrow",
        "dash": ""
      },
      "label": {"text": "z ~ N(0,1)", "x": 248, "y": 170}
    }
  ],
  "groups": [
    {
      "id": "training_group",
      "label": "Training Phase",
      "x": 50, "y": 200,
      "width": 500, "height": 300,
      "style": {"stroke_dasharray": "8,4", "fill": "rgba(200,220,240,0.3)"}
    }
  ]
}
```

**为什么这样比 text2svg 更偏向神经网络级别**：
- text2svg 让 LLM 一步猜所有坐标和样式 → 误差大
- 本方案让 LLM 只管拓扑关系 → ELK.js 精确求解坐标 → NanoBanana 在精确脚手架上渲染
- 类比：从"让画家凭空画建筑"进化到"给画家精确蓝图后渲染效果图"

**GitHub 背书**：
- elkjs-svg (layout JSON → SVG): [EmilStenstrom/elkjs-svg](https://github.com/EmilStenstrom/elkjs-svg)
- netlistsvg (ELK.js 生成电路 SVG): [nturley/netlistsvg](https://github.com/nturley/netlistsvg)

**实现文件**：`src/lib/scaffold-builder.ts`

---

## 4. NanoBanana SVG 渲染（神经网络级别生图）

**输入**：步骤 3 的 scaffold JSON

**输出**：final.svg（学术 figure 级别的 SVG）

**调用方式**（通过 Gemini API native image generation）：

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

scaffold_json = json.load(open("scaffold.json"))

prompt = f"""Based on the following precise layout scaffold, generate a publication-quality
academic method figure as SVG code.

SCAFFOLD (pixel-perfect positions from ELK.js constraint solver):
{json.dumps(scaffold_json, indent=2)}

REQUIREMENTS:
- Use EXACT coordinates from the scaffold — do not adjust positions
- Render each node with the style hints provided (fill, corner_radius, icon_hint)
- Draw edges following the exact SVG path commands provided
- Add academic figure styling: clean lines, consistent fonts, professional color scheme
- Output viewBox="0 0 {scaffold_json['canvas']['width']} {scaffold_json['canvas']['height']}"
- Output ONLY <svg>...</svg>, no markdown
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=prompt,
    config={"response_mime_type": "text/plain"}
)

svg_code = extract_svg(response.text)
```

**与 autofigure2.py 的对接**：
- 复用 `extract_svg_code()` 函数（autofigure2.py L1714）
- 复用 `validate_svg_syntax()` + `fix_svg_with_llm()` 流程（autofigure2.py L1738-L1881）
- 复用 `check_and_fix_svg()` 管线（autofigure2.py L1849）
- SVG 验证修复逻辑与 autofigure2.py 完全一致

**GitHub 背书**：
- Gemini NanoBanana: [google-gemini/cookbook](https://github.com/google-gemini/cookbook) — 官方 cookbook
- NanoBanana CLI extension: [gemini-cli-extensions/nanobanana](https://github.com/gemini-cli-extensions/nanobanana)
- NanoBanana Studio: [aaronkwhite/nanobanana-studio-web](https://github.com/aaronkwhite/nanobanana-studio-web)
- SVG 验证修复: autofigure2.py 内置（lxml + LLM 修复）

**实现文件**：`nanobanana_renderer.py`（Python 端）+ `src/lib/svg-validator.ts`（复用 autofigure2 逻辑）

---

## 5. Astro 前端集成

**输入**：步骤 4 的 final.svg

**输出**：Astro 页面中的交互式 SVG 展示

### 5.1 新增页面路由

```
src/pages/figure/index.astro     — 主页面，输入 method text，展示生成的 figure
src/pages/figure/editor.astro    — SVG 编辑器（基于 svg-edit）
src/pages/api/generate.ts        — API 端点，调用后端 pipeline
```

### 5.2 Astro 组件

```
src/components/figure/
  FigureCanvas.astro      — SVG 渲染画布
  TopologyEditor.astro    — 拓扑 JSON 可视化编辑（可选，基于 React Flow）
  MethodInput.astro       — paper method 文本输入
  PipelineStatus.astro    — 流水线执行状态 SSE 展示
```

### 5.3 与现有项目集成

- 复用 `src/layouts/BaseLayout.astro` 布局
- 复用 `src/components/BaseHead.astro` SEO 头
- 复用 `preset/icons/` 中的 SVG 图标资源
- 通过 `server.py` 的 FastAPI 后端对接，已有 `/api/run` + `/api/events/{job_id}` SSE 接口

**GitHub 背书**：
- Astro: [withastro/astro](https://github.com/withastro/astro) — 现有前端框架
- React Flow (可选拓扑编辑): [xyflow/xyflow](https://github.com/xyflow/xyflow)
- SVG-Edit (SVG 编辑器): [SVG-Edit/svgedit](https://github.com/SVG-Edit/svgedit)

---

## 6. 后端 Pipeline 整合

### 6.1 server.py 扩展

在现有 `server.py`（FastAPI）中新增端点：

```python
@app.post("/api/run-elk")
async def run_elk_pipeline(req: ElkRunRequest):
    """
    正向 pipeline:
    1. method_text → Gemini 生成拓扑 JSON
    2. 拓扑 JSON → ELK.js 约束布局
    3. layout JSON → scaffold JSON
    4. scaffold JSON → NanoBanana SVG 渲染
    5. SVG 验证 + 修复
    """
    ...
```

### 6.2 ELK.js Node.js 微服务

由于 ELK.js 是 JavaScript 库，需要一个 Node.js 微服务来做布局计算：

```
services/
  elk-service/
    index.js          — Express/Fastify 微服务
    package.json      — elkjs + elkjs-svg 依赖
```

或者使用 subprocess 方式：

```python
import subprocess, json

def elk_layout(topology_json: dict) -> dict:
    result = subprocess.run(
        ["node", "services/elk-service/layout.js"],
        input=json.dumps(topology_json),
        capture_output=True, text=True
    )
    return json.loads(result.stdout)
```

**GitHub 背书**：
- FastAPI: [tiangolo/fastapi](https://github.com/tiangolo/fastapi) — 现有后端
- ELK.js Docker 微服务: [kieler/elkjs](https://github.com/kieler/elkjs) — README 提到微服务模式

**实现文件**：`server.py`（扩展）+ `services/elk-service/`（新增）

---

## 7. 完整数据流

```
用户输入 paper method text
         │
         ▼
┌─────────────────────┐
│ Step 1: LLM (Gemini) │  → topology.json
│ 只输出拓扑关系       │     (nodes + edges, 无坐标)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Step 2: ELK.js       │  → layout.json
│ 约束布局求解器       │     (精确 x,y 坐标 + edge routing)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Step 3: Scaffold     │  → scaffold.json
│ 坐标 → 渲染指令     │     (json_example_user1 风格)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Step 4: NanoBanana   │  → raw.svg
│ Gemini 神经渲染      │     (学术级 SVG)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Step 4.5: Validate   │  → final.svg
│ lxml + LLM 修复     │     (语法验证通过的 SVG)
│ (复用 autofigure2)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Step 5: Astro 前端   │  → 浏览器展示
│ 交互式展示 + 编辑    │
└─────────────────────┘
```

---

## 8. 文件结构变更

```
astro-svgfigure/
├── src/
│   ├── lib/                          # 【新增】核心库
│   │   ├── topology-generator.ts     # Step 1: LLM → 拓扑 JSON
│   │   ├── elk-layout.ts             # Step 2: ELK.js 布局
│   │   ├── scaffold-builder.ts       # Step 3: layout → scaffold
│   │   ├── svg-validator.ts          # Step 4.5: SVG 验证
│   │   └── types.ts                  # TypeScript 类型定义
│   ├── pages/
│   │   └── figure/                   # 【新增】figure 生成页面
│   │       ├── index.astro
│   │       └── editor.astro
│   └── components/
│       └── figure/                   # 【新增】figure 组件
│           ├── FigureCanvas.astro
│           ├── MethodInput.astro
│           └── PipelineStatus.astro
├── services/                         # 【新增】微服务
│   └── elk-service/
│       ├── index.js                  # ELK.js 布局微服务
│       ├── layout.js                 # CLI 模式布局脚本
│       └── package.json
├── server.py                         # 【修改】新增 /api/run-elk 端点
├── autofigure2.py                    # 【不变】保留逆向流程
├── nanobanana_renderer.py            # 【新增】NanoBanana SVG 渲染
├── package.json                      # 【修改】添加 elkjs 依赖
└── requirements.txt                  # 【修改】确认 google-genai 依赖
```

---

## 9. 部署命令

### 9.1 开发环境

```bash
# 1. 克隆项目
git clone https://github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure

# 2. 安装前端依赖 + ELK.js
npm install
npm install elkjs elkjs-svg

# 3. 安装 ELK 微服务依赖
cd services/elk-service && npm install && cd ../..

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 配置 API Key
export GEMINI_API_KEY="your-gemini-api-key"

# 6. 启动 ELK 微服务（端口 3001）
node services/elk-service/index.js &

# 7. 启动 FastAPI 后端（端口 8000）
uvicorn server:app --host 0.0.0.0 --port 8000 --reload &

# 8. 启动 Astro 前端（端口 4321）
npm run dev
```

### 9.2 生产环境 (Docker Compose)

```bash
# 一键启动所有服务
docker-compose up -d

# 或分别构建
docker build -t astro-svgfigure-frontend -f Dockerfile.frontend .
docker build -t astro-svgfigure-backend -f Dockerfile.backend .
docker build -t elk-service -f services/elk-service/Dockerfile .
```

### 9.3 Vercel 部署（前端）

```bash
# Astro 前端部署到 Vercel（已有 @astrojs/vercel 依赖）
npx vercel --prod

# 后端部署到 Railway / Fly.io
flyctl launch --name astro-svgfigure-api
```

---

## 10. 与 autofigure2.py / server.py 的关系

| 组件 | 逆向流程 (autofigure2) | 正向流程 (ELK.js) | 共享 |
|---|---|---|---|
| 文本输入 | method_text | method_text | ✅ 相同 |
| 图片生成 | Gemini → figure.png | — | ❌ 不需要 |
| 分割 | SAM3 → samed.png | — | ❌ 不需要 |
| 去背 | RMBG2 | — | ❌ 不需要 |
| 布局 | LLM 硬编码坐标 | ELK.js 约束求解 | ❌ 不同方案 |
| SVG 生成 | LLM multimodal | NanoBanana + scaffold | ❌ 不同方案 |
| SVG 验证 | lxml + LLM 修复 | lxml + LLM 修复 | ✅ 复用 |
| SVG 修复 | fix_svg_with_llm() | fix_svg_with_llm() | ✅ 复用 |
| 后端 API | server.py /api/run | server.py /api/run-elk | ✅ 扩展 |
| 前端 | Astro 页面 | Astro 页面 | ✅ 扩展 |

---

## 11. GitHub 项目引用汇总

| 步骤 | 核心技术 | GitHub 项目 | 用途 |
|---|---|---|---|
| 1 | LLM 拓扑生成 | [google-gemini/cookbook](https://github.com/google-gemini/cookbook) | Gemini structured JSON output |
| 2 | ELK.js 布局 | [kieler/elkjs](https://github.com/kieler/elkjs) | 约束布局引擎 |
| 2 | ELK → SVG | [EmilStenstrom/elkjs-svg](https://github.com/EmilStenstrom/elkjs-svg) | JSON → SVG 渲染参考 |
| 2 | ELK + React | [xyflow/xyflow](https://github.com/xyflow/xyflow) | React Flow + ELK.js 示例 |
| 2 | ELK + 电路图 | [nturley/netlistsvg](https://github.com/nturley/netlistsvg) | ELK.js SVG 渲染案例 |
| 2 | ELK + 简易图 | [davidthings/hdelk](https://github.com/davidthings/hdelk) | ELK.js + SVG.js |
| 4 | NanoBanana CLI | [gemini-cli-extensions/nanobanana](https://github.com/gemini-cli-extensions/nanobanana) | NanoBanana 图片生成 |
| 4 | NanoBanana Studio | [aaronkwhite/nanobanana-studio-web](https://github.com/aaronkwhite/nanobanana-studio-web) | 自托管图片生成 UI |
| 4.5 | SVG 验证 | autofigure2.py (项目内) | lxml 验证 + LLM 修复 |
| 5 | Astro 前端 | [withastro/astro](https://github.com/withastro/astro) | SSG/SSR 框架 |
| 5 | SVG 编辑 | [SVG-Edit/svgedit](https://github.com/SVG-Edit/svgedit) | 浏览器端 SVG 编辑器 |
| 6 | FastAPI 后端 | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) | Python Web 框架 |
| — | sprotty 图表 | [eclipse/sprotty](https://github.com/eclipse/sprotty) | ELK.js 官方推荐图表框架 |

---

## 12. 实施优先级

**Phase 1 — MVP（1-2 周）**
- [ ] `services/elk-service/` — ELK.js 微服务 + CLI 模式
- [ ] `src/lib/topology-generator.ts` — Gemini 拓扑 JSON 生成
- [ ] `src/lib/elk-layout.ts` — ELK.js 布局调用
- [ ] `nanobanana_renderer.py` — NanoBanana SVG 渲染
- [ ] `server.py` 扩展 — `/api/run-elk` 端点

**Phase 2 — 前端集成（1 周）**
- [ ] `src/pages/figure/` — Astro 页面
- [ ] `src/components/figure/` — 交互组件
- [ ] SSE 进度展示（复用现有 /api/events 模式）

**Phase 3 — 优化（持续）**
- [ ] 拓扑 JSON schema 优化（覆盖更多箭头类型）
- [ ] ELK.js 布局参数调优
- [ ] NanoBanana prompt 工程
- [ ] 支持 scaffold JSON 手动编辑 + 重新渲染
