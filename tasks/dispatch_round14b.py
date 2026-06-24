#!/usr/bin/env python3
"""
Round 14b — 功能推进 dispatch。
merge conflicts 已清零, tsc 只剩 tsconfig 级别 2 个 error。
现在 dispatch 6 个小弟做实质功能。
"""

import json, os, re, sys, time, uuid, urllib.request

GH_TOKEN = os.environ.get("GH_TOKEN", "")

BATCHES = {
    "M1070": {
        "desc": "tsconfig fix + astro build smoke test",
        "prompt_body": """你的任务: 修复 tsconfig 并让 astro build 通过。

在 repl 里执行:

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1070@astro.bot" && git config user.name "M1070"
npm install 2>&1 | tail -5
```

然后:
1. `npx tsc --noEmit 2>&1 | head -30` 看还有什么 error
2. 如果 tsconfig extends "astro/tsconfigs/strict" 报找不到, 改为直接写 strict 配置
3. 修完后 `npx astro build 2>&1 | tail -30` 看能否 build
4. commit + push:
```bash
git add -A && git commit -m "M1070: tsconfig fix + build smoke test" && git push origin cell-pubsub-loop
```

规则: 别开新分支, 别加 v2/v3/port 后缀。""",
    },
    "M1071": {
        "desc": "channels/physics 参数验证 — 58 cell bbox 无碰撞确认",
        "prompt_body": """你的任务: 验证 58 个 cell 的物理参数一致性。

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1071@astro.bot" && git config user.name "M1071"
```

写 python 脚本检查:
1. 读 channels/physics/cell_registry.json, 确认 58 个 cell 都在
2. 读每个 channels/cell/*/bbox.json, 检查 AABB 碰撞 (任意两个 cell 不应重叠)
3. 读 channels/physics/species_assignment.json, 确认所有 cell 都有 species
4. 读 channels/physics/edge_routes.json, 确认 65 条 edge 的 waypoints 有效
5. 输出验证报告, 写到 VERIFICATION_R14.md

```bash
git add -A && git commit -m "M1071: 58-cell physics parameter verification" && git push origin cell-pubsub-loop
```

规则: 别开新分支, 别加后缀。""",
    },
    "M1072": {
        "desc": "channels/rendering 30 子模块 __init__.py 补全",
        "prompt_body": """你的任务: 确保 channels/rendering/ 下 30 个子模块都有 __init__.py。

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1072@astro.bot" && git config user.name "M1072"
```

写 python 脚本:
1. 遍历 channels/rendering/ 下所有子目录
2. 如果缺 __init__.py, 创建一个 (内容: 空文件或简单 docstring)
3. 同时检查每个子目录里的 .py 文件是否有语法错误 (`python -m py_compile`)
4. 输出报告

```bash
git add -A && git commit -m "M1072: rendering submodules __init__.py + syntax check" && git push origin cell-pubsub-loop
```

规则: 别开新分支, 别加后缀。""",
    },
    "M1073": {
        "desc": "src/lib/shaders/ — cil-filter + cil-code + cil-layers + cil-loop + cil-graph 5个缺失 species SDF 着色器",
        "prompt_body": """你的任务: 补全 5 个缺失的 species SDF 着色器。

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1073@astro.bot" && git config user.name "M1073"
```

现有着色器: cil-eye.frag, cil-bolt.frag, cil-vector.frag, cil-plus.frag, cil-arrow-right.frag
缺失: cil-filter.frag, cil-code.frag, cil-layers.frag, cil-loop.frag, cil-graph.frag

参考已有的 cil-eye.frag 风格, 用 lygia SDF 函数, 为每个 species 写 GLSL fragment shader:
- cil-filter: 紫罗兰 #7B1FA2, 3x3 网格 + 中心高亮 (用 boxSDF + grid pattern)
- cil-code: 绿 #2E7D32, 花括号形状 (用 lineSDF 组合)
- cil-layers: 深蓝 #1565C0, 三层错位半透明矩形 (用 boxSDF + offset)
- cil-loop: 琥珀 #F57F17, 圆弧箭头 (用 circleSDF + arcSDF)
- cil-graph: 深蓝灰 #37474F, 小圆+连线 网络 (用 circleSDF + lineSDF)

写到 src/lib/shaders/ 目录下。

```bash
git add -A && git commit -m "M1073: 5 missing species SDF shaders — filter, code, layers, loop, graph" && git push origin cell-pubsub-loop
```

规则: 别开新分支, 别加后缀。""",
    },
    "M1074": {
        "desc": "tests/ — pipeline 集成测试扩充",
        "prompt_body": """你的任务: 扩充测试覆盖。

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1074@astro.bot" && git config user.name "M1074"
pip install pytest --break-system-packages
```

在 tests/ 目录下写:
1. test_topology.py — 验证 topology.json 是合法 ELK DAG (58 nodes, 65 edges, 无环)
2. test_species.py — 验证 species_assignment.json 所有 species 合法, gene_traits 结构完整
3. test_convergence.py — 验证 convergence/status.json 格式正确
4. test_rendering_modules.py — 验证 channels/rendering/ 30 个子模块都可 import
5. 运行 `python -m pytest tests/ -v` 输出结果

```bash
git add -A && git commit -m "M1074: pipeline integration tests — topology + species + convergence + rendering" && git push origin cell-pubsub-loop
```

规则: 别开新分支。""",
    },
    "M1075": {
        "desc": "PIPELINE_GUIDE.md 更新为最新 M1055 状态",
        "prompt_body": """你的任务: 更新 PIPELINE_GUIDE.md 为最新状态。

```bash
git clone https://{token}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo && git checkout cell-pubsub-loop
git config user.email "M1075@astro.bot" && git config user.name "M1075"
```

1. `git log --oneline -30` 读最近 commit
2. `tree -L 2 --dirsfirst -I 'node_modules|.git|dist|.astro' | head -100` 读结构
3. 读 PIPELINE_GUIDE_R14.md 作为参考
4. 用这些信息更新 PIPELINE_GUIDE.md, 主要更新:
   - 最新 commit 改为 M1055
   - 加入 Round 14 dispatch 信息
   - 加入 M1051/M1052 merge conflict 修复记录
   - 确认 tsc error 从 490+ 降到 2 (tsconfig 级别)
5. 删除 PIPELINE_GUIDE_R14.md (已合并到 PIPELINE_GUIDE.md)

```bash
git add -A && git commit -m "M1075: PIPELINE_GUIDE.md updated to M1055 state" && git push origin cell-pubsub-loop
```

规则: 别开新分支, 别加后缀。""",
    },
}


def dispatch_one(milestone, batch):
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".claude-hk-config")
    with open(os.path.join(config_dir, "raw_curl.txt")) as f:
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

    create_data = json.dumps({"name": "", "model": "claude-sonnet-4-6", "is_temporary": False}).encode()
    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations",
        data=create_data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    conv_id = json.loads(resp.read()).get("uuid", "")

    prompt = f"你是 {milestone} 小弟。{batch['desc']}。\n\n{batch['prompt_body'].format(token=GH_TOKEN)}"

    payload = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6", "effort": "medium",
        "thinking_mode": "off",
        "tools": [{"type": "repl_v0", "name": "repl"}],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [], "rendering_mode": "messages",
    }).encode()

    req2 = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=payload, headers={**headers, "accept": "text/event-stream"}, method="POST")
    try:
        urllib.request.urlopen(req2, timeout=10)
    except Exception:
        pass

    print(f"[dispatch] {milestone}: conv={conv_id[:12]} desc={batch['desc']}", file=sys.stderr)
    return {"conv_id": conv_id, "url": f"{origin}/chat/{conv_id}", "desc": batch["desc"]}


def main():
    log = {}
    for m, b in BATCHES.items():
        try:
            log[m] = dispatch_one(m, b)
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] {m}: {e}", file=sys.stderr)
            log[m] = {"error": str(e)}

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "channels", "convergence", "dispatch_log_round14b.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)

    print(f"\n=== Round 14b Dispatch Complete ===")
    print(f"Dispatched {len([v for v in log.values() if 'error' not in v])} sub-Claudes")
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
