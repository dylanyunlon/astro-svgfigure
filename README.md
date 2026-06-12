# astro-svgfigure

PixiJS + Astro + FastAPI 驱动的 SVG 拓扑渲染引擎，支持 cell pub/sub loop 自动迭代收敛。

---

## 部署说明

### 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| bun | 最新版（或 npm/pnpm） |
| Node.js | 18+（bun 自带） |

---

### 安装依赖

```bash
# 后端
pip install fastapi uvicorn "uvicorn[standard]" pydantic python-multipart

# 前端
bun install
# 或
npm install
```

---

### 启动服务

**后端**（端口 8000）

```bash
python server.py
```

输出示例：
```
--- Starting Server ---
Local access: http://127.0.0.1:8000
-----------------------
```

**前端**（端口 4321）

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

---

### 使用流程

```
输入文字描述
    │
    ▼
POST /api/topology  ──→  生成 ELK 拓扑 JSON
    │
    ▼
epoch_controller.py  ──→  拆分骨架信号，分配 species
    │
    ▼
cell loop 自动运行（每 epoch）
  ├─ 每个 cell 读取骨架信号 + 力场（subscribe）
  ├─ 按 species 算法生成 SVG 参数（proc）
  ├─ 发布 bbox + svg（publish）
  ├─ 物理引擎检测碰撞 → 更新力场
  └─ 收敛判断 → 未收敛则进入下一 epoch
    │
    ▼
GET /api/cells  ──→  前端每 500ms 轮询 cell 状态
    │
    ▼
PixiJS 渲染（pixi-cell-renderer.ts）
```

---

## API 端点

### 核心端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/topology` | 输入文字描述，生成 ELK 拓扑 JSON |
| `GET`  | `/api/cells` | 读取所有 cell 当前状态（CellDescriptor[]），每 500ms 轮询 |
| `POST` | `/api/cell-loop` | 触发一轮 cell pub/sub loop epoch |
| `GET`  | `/api/epochs` | 查询 epoch 历史与收敛状态 |
| `GET`  | `/api/cell/{id}` | 获取单个 cell 的详细状态与 SVG 参数 |

### 辅助端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/beautify` | SVG 美化后处理 |
| `POST` | `/api/validate` | 校验 topology/SVG 结构 |
| `GET`  | `/api/models` | 列出可用 AI 模型 |
| `POST` | `/api/generate-image` | 图像生成 |
| `POST` | `/api/generate-prompt` | Prompt 辅助生成 |
| `GET`  | `/api/config` | 读取后端配置 |
| `GET`  | `/api/health` | 健康检查 |
| `POST` | `/api/run` | 执行渲染 pipeline |
| `POST` | `/api/upload` | 上传资源文件 |
| `GET`  | `/api/events/{job_id}` | SSE 实时进度事件流 |
| `GET`  | `/api/artifacts/{job_id}/{path}` | 获取渲染产物 |

---

## 五层架构（L1–L5）

```
L1  输入解析层    Text → ELK Topology JSON
                  用户输入文字 → LLM 解析为节点/边结构，
                  经 /api/topology 输出标准化拓扑描述。

L2  骨架调度层    Topology → Per-Cell Skeleton Signal
                  epoch_controller.py 读取拓扑，
                  为每个节点分配 species（基因型），
                  拆分骨架信号写入 channels/cell/{id}/。

L3  Cell 计算层   Skeleton + Force Field → SVG Params
                  每个 cell 作为独立 agent（cell_agent.py）：
                  订阅骨架信号与力场，按 species 算法计算
                  自然 bbox、opacity、species_params，
                  发布结果到 channels/cell/{id}/status.json。

L4  物理收敛层    Bboxes → Force Field Update
                  物理引擎（loop_orchestrator.py）读取所有 cell 的 bbox，
                  检测碰撞，更新力场，驱动下一 epoch。
                  收敛后按 z-layer 顺序输出最终 SVG 骨架。

L5  渲染输出层    SVG Params → PixiJS Canvas
                  前端 pixi-cell-renderer.ts 每 500ms 拉取
                  /api/cells，将 CellDescriptor[] 映射为
                  PixiJS Sprite/Container，实时更新画布。
```

对应 Unreal Engine 渲染管线映射关系：

```
Actor/Component      →  Cell（sub-Claude agent）
Scene Graph          →  ELK Topology
BasePass → GBuffer   →  Constraint Collect → constraints.json
Lighting Pass        →  Constraint Solver（relative → absolute）
PostProcess          →  SVG 后处理（边缘软化、高光）
Final Composite      →  SVG Assembly（z-layer 排序）
```

---

## 分支说明

- `main` — 稳定版本
- `cell-pubsub-loop` — 当前开发分支，cell pub/sub loop + PixiJS 渲染

## 作者

dylanyunlon &lt;dogechat@163.com&gt;
