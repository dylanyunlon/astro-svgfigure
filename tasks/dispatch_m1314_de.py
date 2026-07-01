#!/usr/bin/env python3
"""dispatch D (retry) + E (e2e heavy validation)"""
import json, os, re, sys, time, uuid
import urllib.request, urllib.error

with open(".claude-hk-config/raw_curl.txt") as f:
    raw = f.read()

cookie = re.search(r"-b '([^']+)'", raw).group(1)
org_id = re.search(r"organizations/([^/]+)", raw).group(1)
origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

def _h(n, d=""):
    m = re.search(rf"-H '{re.escape(n)}: ([^']+)'", raw)
    return m.group(1) if m else d

H = {
    "Content-Type": "application/json", "accept": "text/event-stream",
    "accept-language": _h("accept-language", "zh-CN,zh;q=0.9"),
    "anthropic-client-platform": "web_claude_ai",
    "origin": origin, "user-agent": ua,
    "referer": _h("referer", f"{origin}/new"),
    "sec-ch-ua": _h("sec-ch-ua"),
    "sec-ch-ua-mobile": _h("sec-ch-ua-mobile", "?0"),
    "sec-ch-ua-platform": _h("sec-ch-ua-platform"),
    "sec-fetch-dest": "empty", "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "priority": _h("priority", "u=1, i"),
    "Cookie": cookie,
}

TOKEN = os.environ.get("GIT_TOKEN", "<YOUR_GITHUB_TOKEN>")

def create_conv(model="claude-sonnet-4-6"):
    d = json.dumps({"name": "", "model": model, "is_temporary": False,
                    "include_conversation_preferences": True,
                    "paprika_mode": None, "compass_mode": None,
                    "tool_search_mode": "auto", "enabled_imagine": True}).encode()
    for r in range(3):
        try:
            req = urllib.request.Request(
                f"{origin}/api/organizations/{org_id}/chat_conversations",
                data=d, headers={**H, "accept": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read()).get("uuid", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 and r < 2:
                print(f"  429, waiting 15s ({r+1}/3)")
                time.sleep(15)
            else:
                raise
        except:
            time.sleep(5)

def send(conv_id, prompt, model="claude-sonnet-4-6"):
    p = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai", "locale": "en-US",
        "model": model, "effort": "medium", "thinking_mode": "off",
        "tools": [
            {"type": "web_search_v0", "name": "web_search"},
            {"type": "repl_v0", "name": "repl"},
            {"type": "artifacts_v0", "name": "artifacts"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [], "sync_sources": [],
        "rendering_mode": "messages",
        "create_conversation_params": {
            "name": "", "model": model,
            "include_conversation_preferences": True,
            "paprika_mode": None, "compass_mode": None,
            "tool_search_mode": "auto", "is_temporary": False,
            "enabled_imagine": True,
        },
    }).encode()
    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=p, headers=H, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"  SSE ok: {resp.read(60)[:60]}")
    except:
        print("  fire-and-forget OK")

# ── TASK D: PBR shader metaball (重试 + 更强调验证循环) ──────────────────

CLONE = f"git clone https://{TOKEN}@github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure && git checkout cell-pubsub-loop && git config user.name claude-worker-d && git config user.email worker-d@astro"

TASK_D = f"""你是 astro-svgfigure 项目的 Claude Code 小弟。你必须在 repl 里反复验证 shader 语法直到编译通过。不要改几行就提交。

第一步执行:
```bash
{CLONE}
```

## 任务: PBR shader 改造 metaball SDF

1. 读 src/lib/sph/pbr-gpu-pass.ts 的 FRAG_SRC — 找当前 SDF 函数
2. 读 channels/cell/GEOMETRY_FORMAT.md — cellSDF() GLSL 实现在里面
3. 读 channels/cell/input/geometry.json — 真实 sdf.lobes 数据

4. 修改 pbr-gpu-pass.ts 的 FRAG_SRC:
   - 加 uniform float u_baseRadius; uniform int u_lobeCount;
   - 加 uniform vec3 u_lobes[8]; // x=angle, y=distance, z=radius
   - 加 uniform float u_noiseAmp, u_noiseFreq;
   - 实现 cellSDF(): metaball smooth union (k=0.3) + surface noise
   - 用简单 hash noise 函数

5. 修改 renderCells() 方法: 从 CellData 读 sdfBaseRadius/sdfLobes 设 uniform

6. **必须做的验证循环**:
   a. 用 grep 提取 FRAG_SRC 和 VERT_SRC 字符串的内容
   b. 写一个 Python 脚本检查 GLSL 基本语法:
      - 所有 uniform 声明是否在 main() 之前
      - 函数签名里的类型是否匹配
      - #version 300 es 兼容性 (没有 varying/attribute, 用 in/out)
      - precision 声明
   c. 如果发现问题 → 修复 → 重新检查 → 循环
   d. 至少验证 3 遍才能 push

7. 完成后:
```bash
cd astro-svgfigure && git pull origin cell-pubsub-loop
git add -A && git commit -m "M1314d: PBR shader metaball SDF — lobes + smooth union + surface noise"
git push origin cell-pubsub-loop
```

关键: 只改 pbr-gpu-pass.ts。MAX_LOBES=8。别创建 v2/v3 后缀文件。
"""

# ── TASK E: 端到端重度验证 ─────────────────────────────────────────────────

TASK_E = f"""你是 astro-svgfigure 项目的 Claude Code 小弟。这是一个重度任务——你需要实际运行 dev server 并反复调试直到渲染管线工作。不要改几行就交差。

第一步:
```bash
{CLONE.replace('worker-d', 'worker-e').replace('worker-d@astro', 'worker-e@astro')}
```

## 任务: 端到端验证 + 修复

### Phase 1: 启动 dev server
```bash
npm install 2>&1 | tail -10
npx astro dev --host 0.0.0.0 &
sleep 15
curl -s http://localhost:4321/ | head -5
```
如果启动失败（TypeScript 编译错误、模块找不到），读错误信息，修复，重启。循环直到能启动。

### Phase 2: 验证数据端点
```bash
# composite_params
curl -s http://localhost:4321/channels/composite_params.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'cells:{{len(d.get(\"cells\",{{}}))}}, edges:{{len(d.get(\"edge_routes\",{{}}))}}')"

# geometry.json
curl -s http://localhost:4321/channels/cell/input/geometry.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'tick:{{d[\"tick\"]}}, base_r:{{d[\"sdf\"][\"base_radius\"]}}, lobes:{{len(d[\"sdf\"][\"lobes\"])}}')"

# api/cells (如果存在)
curl -s http://localhost:4321/api/cells 2>&1 | head -5
```

### Phase 3: 验证 /world 页面 JS 构建
```bash
# 拿到 world 页面的 HTML
curl -s http://localhost:4321/world > /tmp/world.html
# 提取 JS 入口
grep -oP 'src="[^"]*"' /tmp/world.html | head -10
# 检查 TypeScript 编译产物
ls -la node_modules/.astro/
```
如果有 TypeScript 编译错误，从 dev server 的 stderr 读取，修复源码，重启验证。

### Phase 4: 检查 gpu-render-loop.ts 的 pass init
```bash
# 读 gpu-render-loop.ts 末尾的 pass status log
grep -A 20 "pass init status" src/lib/sph/gpu-render-loop.ts
```

### Phase 5: 跑 tick-runner 确认物理引擎
```bash
python3 channels/cell/tick-runner.py --ticks 2 --verbose 2>&1 | head -30
```

把所有发现、错误、修复写入 tasks/e2e_validation.md。每个 phase 都要有完整的命令输出。

完成后:
```bash
kill %1
cd astro-svgfigure && git pull origin cell-pubsub-loop
git add -A && git commit -m "M1314e: e2e validation — dev server + data endpoints + world page"
git push origin cell-pubsub-loop
```

关键: 不是改几行就交差。你要在 repl 里实际运行，发现问题就修，循环到系统能跑。
"""

# ── Dispatch ───────────────────────────────────────────────────────────────

print("Dispatching D (retry)...")
cid = create_conv()
print(f"  D conv: {cid}")
send(cid, TASK_D)
time.sleep(10)

print("Dispatching E (e2e heavy)...")
cid2 = create_conv()
print(f"  E conv: {cid2}")
send(cid2, TASK_E)

print("Done. Monitor: git log --oneline for M1314d and M1314e")
