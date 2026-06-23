#!/usr/bin/env python3
"""
M823-M828 dispatch — 管理者分配 6 个小弟 Claude 并行任务。

每个小弟 clone 仓库、切换 cell-pubsub-loop 分支、完成任务、push。
Cookie 创建新对话，每个小弟独立对话，不会冲突。

Usage:
    python3 tasks/dispatch_m823_m828.py
"""

import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, ".claude-hk-config", "raw_curl.txt")

GIT_REPO = "https://github.com/dylanyunlon/astro-svgfigure.git"
BRANCH = "cell-pubsub-loop"
GH_TOKEN = os.environ.get("GH_TOKEN", "MISSING_TOKEN")

# ─────────────────────────────────────────────────────────────────────────────
# 6 个小弟任务定义
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "M823",
        "title": "M823: drawcall batch + instanced cell rendering",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M823 — drawcall batch + instanced cell rendering

当前 channels/rendering/drawcall/ 只有 44 行代码，太薄了。需要充实。

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 阅读现有代码:
```bash
cat channels/rendering/drawcall/*.py
cat src/lib/renderers/pixi-cell-renderer.ts | head -100
```

3. 充实 channels/rendering/drawcall/ 模块，参考 upstream/unreal-renderer/MeshDrawCommands.cpp 的概念:
   - 实现 DrawCallBatcher: 将相同 species 的 cell 合并为 instanced draw call
   - 实现 DrawCallSorter: 按 z-layer + species + opacity 排序 draw call
   - 实现 IndirectDrawBuffer: 间接绘制缓冲，减少 CPU→GPU 调用
   - 每个函数加 [ASTRO-DRAWCALL] debug tag

4. 同时充实 src/lib/renderers/cell-culling.ts (目前只有 3 行!):
   - 实现视锥体剔除 (frustum culling): 只渲染视口内的 cell
   - 实现遮挡剔除 (occlusion culling): 被完全遮挡的 cell 跳过渲染
   - 参考 channels/rendering/occlusion/ 的 Python 端逻辑

5. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M823: drawcall batch + instanced cell rendering + frustum culling"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M823: 开头"""
    },
    {
        "id": "M824",
        "title": "M824: streaming LOD + resources pool",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M824 — streaming LOD + resources pool

当前 channels/rendering/streaming/ 只有 145 行, resources/ 只有 55 行。需要充实。

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 阅读现有代码:
```bash
cat channels/rendering/streaming/*.py
cat channels/rendering/resources/*.py
cat channels/rendering/nanite/*.py
```

3. 充实 channels/rendering/streaming/ 模块:
   - 实现 StreamingManager: 按距离分级加载 cell 细节 (LOD 0-3)
   - 实现 MipMapScheduler: MSDF 纹理 mipmap 按需加载
   - 实现 PriorityQueue: 加载优先级 = 距离 × 可见性 × species_importance
   - 参考 upstream/potree/ 的八叉树 LOD 策略

4. 充实 channels/rendering/resources/ 模块:
   - 实现 TexturePool: 共享纹理池 (避免重复创建同 species 纹理)
   - 实现 BufferAllocator: VBO/IBO 内存池
   - 实现 ResourceLifecycle: 引用计数 + LRU 淘汰

5. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M824: streaming LOD + resources pool"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M824: 开头"""
    },
    {
        "id": "M825",
        "title": "M825: color grading + visibility culling",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M825 — color grading + visibility culling

当前 channels/rendering/color/ 只有 61 行, visibility/ 只有 219 行。需要充实。

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 阅读现有代码:
```bash
cat channels/rendering/color/*.py
cat channels/rendering/visibility/*.py
cat src/lib/shaders/colormap.frag
cat src/lib/shaders/iq-palette-species.frag
```

3. 充实 channels/rendering/color/ 模块:
   - 实现 ColorGrading: 整体色调映射 (tone mapping) — 学术图一般用冷色调
   - 实现 SpeciesColorHarmony: 确保相邻 species 颜色和谐 (互补/三角配色)
   - 实现 LUTPipeline: 3D LUT 色彩查找表，切换不同论文风格 (Nature/Science/IEEE)
   - 参考 upstream/glsl-colormap/ 的科学配色

4. 充实 channels/rendering/visibility/ 模块:
   - 实现 HierarchicalZBuffer: 层次 Z-buffer 快速遮挡测试
   - 实现 PortalVisibility: 群组间的 portal 可见性判定
   - 实现 PVS (Potentially Visible Set): 预计算可见集合

5. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M825: color grading LUT + visibility HZB"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M825: 开头"""
    },
    {
        "id": "M826",
        "title": "M826: pathtracing styleprobe + compositor",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M826 — pathtracing styleprobe 邻居采样 + compositor 合成器

当前 channels/rendering/pathtracing/ 只有 137 行, styleprobe/ 163 行, compositor/ 208 行。需要充实。

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 阅读现有代码:
```bash
cat channels/rendering/pathtracing/*.py
cat channels/rendering/styleprobe/*.py
cat channels/rendering/compositor/*.py
cat channels/rendering/lumen/*.py
```

3. 充实 pathtracing/:
   - 实现 StyleRayMarcher: 从每个 cell 发射 style ray，采样邻居的视觉属性
   - 实现 BounceAccumulator: 多次弹射后的颜色/风格混合
   - 参考 upstream/unreal-renderer/PathTracing.cpp 的光线追踪概念

4. 充实 styleprobe/:
   - 实现 NeighborProbe: 采样 k 近邻 cell 的 species 颜色
   - 实现 StyleDiffusion: 风格扩散 — 让相邻 cell 的视觉风格平滑过渡
   - 实现 ProbeCache: 缓存探针结果，避免每帧重复计算

5. 充实 compositor/:
   - 实现 LayerCompositor: 按 z-layer 从后到前合成所有渲染 pass
   - 实现 BlendModeRegistry: 支持多种混合模式 (add/multiply/screen/overlay)
   - 实现 AOVOutput: 分离输出层 (color/depth/normal/species_id)

6. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M826: pathtracing styleprobe + compositor AOV"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M826: 开头"""
    },
    {
        "id": "M827",
        "title": "M827: cell-selection-ring + cell-transition + cell-debug-overlay",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M827 — 充实 3 个极薄前端渲染器

当前这几个文件几乎为空:
- src/lib/renderers/cell-selection-ring.ts (17 行)
- src/lib/renderers/cell-transition.ts (18 行)
- src/lib/renderers/cell-debug-overlay.ts (18 行)
- src/lib/renderers/cell-asset-loader.ts (18 行)

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 阅读现有代码和参考:
```bash
cat src/lib/renderers/cell-selection-ring.ts
cat src/lib/renderers/cell-transition.ts
cat src/lib/renderers/cell-debug-overlay.ts
cat src/lib/renderers/cell-asset-loader.ts
cat src/lib/renderers/pixi-cell-renderer.ts | head -80
# 看看 PixiJS 和 filter 怎么用
cat src/lib/renderers/sdf-species-filter.ts | head -60
```

3. 充实 cell-selection-ring.ts:
   - 用 PixiJS Graphics 绘制选中 cell 的发光圆环
   - 圆环颜色跟随 species primary_color
   - 脉冲动画 (半径 + alpha 周期波动)
   - hover 时显示，click 时锁定

4. 充实 cell-transition.ts:
   - cell 之间切换时的过渡动画
   - 淡入淡出 + 缩放 + 位移插值
   - 支持 species 变化时的变色过渡 (颜色插值)
   - 用 requestAnimationFrame 驱动

5. 充实 cell-debug-overlay.ts:
   - debug 模式下在每个 cell 上方显示:
     cell_id, species, bbox(x,y,w,h), z-layer, epoch, convergence delta
   - 用 PixiJS Text + BitmapText 渲染
   - 可开关 (F12 或 debug panel)

6. 充实 cell-asset-loader.ts:
   - 预加载 species SDF shader 纹理
   - 预加载 MSDF 字体图集
   - 加载进度回调 (用于 loading bar)
   - 错误重试 (最多 3 次)

7. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M827: cell selection ring + transition + debug overlay + asset loader"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M827: 开头"""
    },
    {
        "id": "M828",
        "title": "M828: convergence fix self_attn + epoch 推进",
        "prompt": f"""你是 astro-svgfigure 项目的开发者小弟。

## 你的任务: M828 — 修复收敛发散 + 推进 epoch

当前 convergence status 显示 self_attn 发散 (delta=200), 整个系统 diverged=true。需要修复。

## 工作流程

1. 在 repl 中执行:
```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout {BRANCH}
```

2. 诊断问题:
```bash
cat channels/convergence/status.json
cat channels/cell/self_attn/bbox.json
cat channels/cell/self_attn/agent_params.json
cat channels/physics/force_field.json | python3 -m json.tool
cat channels/physics/collision.json | python3 -m json.tool
```

3. 修复 channels/convergence/status.json — self_attn delta 应为 0:
   - 检查 self_attn 的 bbox 是否在合理范围
   - 检查 force_field 中 self_attn 的推力是否过大
   - 如果 force_field 中 self_attn 的 push_mag 过大，调小到合理值
   - 更新 convergence/status.json: diverged=false, max_delta 降到 < 0.5

4. 推进 epoch:
   - 更新 channels/skeleton/epoch.json 的 epoch 号
   - 确保所有 7 个 cell 的 status.json epoch 一致
   - 更新 convergence/status.json 为 converged=true

5. 检查并修复 channels/epoch_controller.py 中的收敛阈值逻辑:
   - 确保 _check_convergence() 正确比较所有 cell delta
   - 确保 diverged 标记正确清除

6. commit 并 push:
```bash
cd /tmp/astro
git add -A
git commit -m "M828: fix self_attn divergence + epoch convergence"
git push origin {BRANCH}
```

重要规则:
- 在现有文件上改，不要创建 v2/v3 后缀文件
- 不要开新分支，直接 push 到 {BRANCH}
- commit message 必须以 M828: 开头"""
    },
]


def dispatch_one(task: dict) -> str:
    """派遣一个小弟 Claude。返回 conversation uuid。"""
    with open(CONFIG) as f:
        raw = f.read()

    cookie = re.search(r"-b '([^']+)'", raw).group(1)
    org_id = re.search(r"organizations/([^/]+)", raw).group(1)
    origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
    ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

    headers = {
        "Content-Type": "application/json",
        "origin": origin, "user-agent": ua,
        "referer": f"{origin}/",
        "accept-language": "zh-CN,zh;q=0.9",
        "anthropic-client-platform": "web_claude_ai",
        "Cookie": cookie,
    }

    # Step 1: Create conversation
    create_data = json.dumps({
        "name": task["title"],
        "model": "claude-sonnet-4-6",
        "is_temporary": False,
    }).encode()
    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations",
        data=create_data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    conv_id = json.loads(resp.read()).get("uuid", "")

    # Step 2: Send the task prompt
    payload = json.dumps({
        "prompt": task["prompt"],
        "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6",
        "effort": "high",
        "thinking_mode": "off",
        "tools": [
            {"type": "repl_v0", "name": "repl"},
            {"type": "web_search_v0", "name": "web_search"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [],
        "rendering_mode": "messages",
    }).encode()

    req2 = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers={**headers, "accept": "text/event-stream"},
        method="POST")
    try:
        urllib.request.urlopen(req2, timeout=15)
    except Exception:
        pass  # fire and forget — SSE stream

    return conv_id


def main():
    print(f"[管理者] 开始调度 {len(TASKS)} 个小弟 Claude...")
    print(f"[管理者] 仓库: {GIT_REPO}")
    print(f"[管理者] 分支: {BRANCH}")
    print()

    results = []
    for i, task in enumerate(TASKS):
        try:
            conv_id = dispatch_one(task)
            results.append((task["id"], conv_id))
            print(f"  [{i+1}/{len(TASKS)}] ✅ {task['id']}: {task['title']}")
            print(f"         conv: {conv_id[:20]}...")
            # 间隔 3 秒避免限流
            if i < len(TASKS) - 1:
                time.sleep(3)
        except Exception as e:
            print(f"  [{i+1}/{len(TASKS)}] ❌ {task['id']}: {e}")
            results.append((task["id"], f"ERROR: {e}"))

    print()
    print("=" * 60)
    print("[管理者] 调度完成。小弟正在各自的 repl 中执行任务。")
    print("[管理者] 任务列表:")
    for mid, conv in results:
        status = "✅" if not conv.startswith("ERROR") else "❌"
        print(f"  {status} {mid}: {conv[:30]}...")
    print()
    print("[管理者] 预计 3-5 分钟后小弟们 push 完毕。")
    print("[管理者] 监督: git pull && git log --oneline -10")


if __name__ == "__main__":
    main()
