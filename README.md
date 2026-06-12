# astro-svgfigure

PixiJS + Astro + FastAPI 驱动的 SVG 拓扑渲染引擎，支持 cell pub/sub loop 自动迭代收敛。

分支：**`cell-pubsub-loop`**

---

## 部署说明

### 环境要求

| 组件 | 最低版本 |
|------|----------|
| Python | 3.11+ |
| bun | 最新版（或 npm/pnpm） |
| Node.js | 18+（bun 自带） |

---

### 1 — 安装依赖

```bash
# 后端（核心运行时）
pip install fastapi uvicorn "uvicorn[standard]" pydantic pydantic-settings \
            python-multipart openai anthropic httpx requests

# 完整依赖（含图像/ML，可选）
pip install -r requirements.txt

# 前端
bun install
# 或
npm install
```

---

### 2 — 启动后端（端口 8000）

```bash
python server.py
```

输出示例：

```
--- Starting Server ---
Local access: http://127.0.0.1:8000
Docs:         http://127.0.0.1:8000/docs
-----------------------
```

FastAPI 自动生成交互式 API 文档：`http://localhost:8000/docs`

---

### 3 — 启动前端（端口 4321）

```bash
bun dev
# 或
npm run dev
```

也可一键同时启动前后端：

```bash
bun run dev:all
# 等价于: concurrently "astro dev" "python server.py"
```

前端地址：`http://localhost:4321`

---

### 4 — 使用流程：文字 → Topology → Cell Loop → PixiJS

```
① 用户在前端输入文字描述（例如：「一个包含三个服务节点的微服务架构」）
         │
         ▼
② POST /api/topology
   LLM 将文字解析为 ELK 拓扑 JSON（nodes / edges / 布局方向）
         │
         ▼
③ POST /api/cell-loop  ←—— 或由前端 epoch controller 自动触发
   Loop Orchestrator 启动 pub/sub 迭代：
     ├─ topology_to_skeleton.py  拆解节点 → 骨架信号，写入 channels/cell/{id}/
     ├─ [每 epoch] cell_agent.py 订阅骨架 + 力场 → 按 species 算法计算 SVG 参数
     ├─ 物理引擎检测 bbox 碰撞 → 更新 force_field
     └─ 收敛判断（threshold=0.5）→ 未收敛则进入下一 epoch（最大 10 轮）
         │
         ▼
④ GET /api/cells  ←—— 前端每 500ms 轮询
   前端 pixi-cell-renderer.ts 将 CellDescriptor[] 映射为 PixiJS Sprite/Container
   实时更新画布，收敛后输出最终 SVG
```

---

## API 端点列表

### 核心端点

| 方法 | 路径 | Body / Params | 说明 |
|------|------|---------------|------|
| `POST` | `/api/topology` | `{ text, model?, algorithm?, direction? }` | 文字 → ELK 拓扑 JSON |
| `POST` | `/api/cell-loop` | `{ structured_data?, max_epochs? }` | 触发完整 cell pub/sub loop |
| `GET`  | `/api/cells` | — | 返回 `CellDescriptor[]`（全部 cell 状态） |
| `GET`  | `/api/cell/{cell_id}` | path param | 单个 cell 的 params + status |
| `GET`  | `/api/epochs` | — | 查询 epoch 历史与收敛状态 |

### 渲染 / 生成端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/run` | 提交完整渲染 pipeline 任务（异步 Job） |
| `GET`  | `/api/events/{job_id}` | SSE 实时进度事件流（`text/event-stream`） |
| `GET`  | `/api/artifacts/{job_id}/{path}` | 获取渲染产物文件 |
| `POST` | `/api/generate-image` | 图像生成（Stable Diffusion / Gemini） |
| `POST` | `/api/generate-prompt` | Prompt 辅助生成 |

### SVG 后处理端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/beautify` | SVG 美化后处理（nanobanana bridge） |
| `POST` | `/api/validate` | 校验 topology / SVG 结构合法性 |

### 系统端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/models` | 列出当前可用的 AI 模型 |
| `GET` | `/api/config` | 读取后端运行时配置 |
| `GET` | `/api/health` | 健康检查（`{ "status": "ok" }`） |
| `POST` | `/api/upload` | 上传资源文件（`multipart/form-data`） |
| `GET`  | `/api/uploads/{filename}` | 获取已上传文件 |

---

## 五层架构（L1–L5）

```
┌─────────────────────────────────────────────────────────────────┐
│  L1  输入解析层    Text → ELK Topology JSON                      │
│                                                                  │
│       用户输入文字 → /api/topology → LLM 解析为节点/边结构，      │
│       经 backend/pipeline/topology_gen.py 输出标准化拓扑 JSON。  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  L2  骨架调度层    Topology → Per-Cell Skeleton Signal           │
│                                                                  │
│       channels/topology_to_skeleton.py 读取拓扑，               │
│       为每个节点分配 species（基因型），                          │
│       拆分骨架信号写入 channels/cell/{id}/params.json。          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  L3  Cell 计算层   Skeleton + Force Field → SVG Params          │
│                                                                  │
│       channels/cell_agent.py 作为独立 agent：                   │
│       订阅骨架信号与力场，按 species 算法计算                    │
│       自然 bbox、opacity、species_params，                       │
│       发布结果到 channels/cell/{id}/status.json。               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  L4  物理收敛层    Bboxes → Force Field → Convergence           │
│                                                                  │
│       channels/loop_orchestrator.py 读取全部 cell bbox，        │
│       检测 3D 碰撞，更新 force_field，驱动下一 epoch；           │
│       收敛判断（threshold=0.5，max=10 epoch），                  │
│       收敛后按 z-layer 顺序输出最终 SVG 骨架。                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  L5  渲染输出层    SVG Params → PixiJS Canvas                   │
│                                                                  │
│       前端 pixi-cell-renderer.ts 每 500ms 拉取 /api/cells，     │
│       将 CellDescriptor[] 映射为 PixiJS Sprite/Container，      │
│       实时更新画布；收敛后组装并导出最终 SVG。                   │
└─────────────────────────────────────────────────────────────────┘
```

### 与 Unreal Engine 渲染管线的映射

| UE 概念 | 本项目对应 |
|---------|-----------|
| Actor / Component | Cell（`cell_agent.py`） |
| Scene Graph | ELK Topology JSON |
| GBuffer / BasePass | Constraint Collect → `constraints.json` |
| Lighting Pass | Constraint Solver（relative → absolute layout） |
| PostProcess | SVG 后处理（美化、边缘软化） |
| Final Composite | SVG Assembly（z-layer 排序） |

---

## 项目结构（关键文件）

```
astro-svgfigure/
├── server.py                  # FastAPI 入口（端口 8000）
├── requirements.txt           # Python 依赖
├── package.json               # 前端依赖（bun/npm）
├── astro.config.ts            # Astro 配置
├── backend/
│   ├── ai_engine.py           # LLM 调用封装（OpenAI / Anthropic / Gemini）
│   ├── config.py              # 环境配置（pydantic-settings）
│   ├── schemas.py             # 请求/响应类型定义
│   └── pipeline/
│       ├── topology_gen.py    # L1: 文字 → ELK 拓扑
│       ├── scaffold_builder.py
│       ├── nanobanana_bridge.py  # SVG 美化
│       └── svg_validator.py
├── channels/
│   ├── loop_orchestrator.py   # L4: cell loop 主循环
│   ├── cell_agent.py          # L3: 单 cell 计算 agent
│   ├── cell_component.py      # Cell 数据模型
│   ├── epoch_controller.py    # Epoch 调度
│   ├── topology_to_skeleton.py  # L2: 拓扑 → 骨架信号
│   ├── channel_runtime.py     # pub/sub 通道运行时
│   ├── cell/{id}/
│   │   ├── params.json        # cell 输入参数 + species
│   │   └── status.json        # cell 输出 bbox + svg
│   ├── physics/               # 物理碰撞引擎
│   └── skeleton/              # 骨架信号存储
└── src/
    └── pages/                 # Astro 前端页面
```

---

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | 稳定版本 |
| `cell-pubsub-loop` | 当前开发分支：cell pub/sub loop + PixiJS 实时渲染 |

---

## 作者

dylanyunlon &lt;dogechat@163.com&gt;
