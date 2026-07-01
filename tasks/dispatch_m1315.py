#!/usr/bin/env python3
"""
M1315: 赛博真实世界生产级别 — dispatch heavy agentic loop workers
Claude Code 版本强制 2.1.168
"""
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
            else: raise
        except: time.sleep(5)

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

CLONE = f"git clone https://{TOKEN}@github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure && git checkout cell-pubsub-loop"
CC_PIN = "npm install -g @anthropic-ai/claude-code@2.1.168 && claude --version"

# ── Worker F: 核心渲染修复 — 必须 agentic loop 到画面正确 ─────────────────

TASK_F = f"""你是 astro-svgfigure 项目的 Claude Code 小弟。

强制要求: Claude Code 版本必须是 2.1.168。
```bash
{CC_PIN}
```

```bash
{CLONE}
git config user.name "claude-worker-f" && git config user.email "worker-f@astro"
```

## 背景

/world 页面当前渲染结果极差:
- 黑色背景上几个灰白色椭圆体
- 没有颜色 — geometry.json 的 albedo/glow_color 完全没生效
- 没有 metaball 有机形态 — 还是简单椭圆，不是 lobes smooth union
- 只显示了几个 cell，不是 58 个
- 有一个奇怪的剑/柱子形状 (GLB mesh 位置错误)

Console 显示 drawCalls: 313, fps: 31, 58 cells loaded。数据进去了但渲染完全不对。

## 你的任务: 让 /world 渲染出 58 个有颜色、有机形态的 cell

这不是改几行代码的任务。你必须:

1. 启动 dev server 并实际验证渲染:
```bash
npm install 2>&1 | tail -5
npx astro dev --host 0.0.0.0 &
sleep 15
```

2. 安装 puppeteer 截图验证:
```bash
npm install puppeteer 2>&1 | tail -3
```

3. 写截图脚本 tasks/screenshot.mjs:
```javascript
import puppeteer from 'puppeteer';
const browser = await puppeteer.launch({{headless: true, args: ['--no-sandbox']}});
const page = await browser.newPage();
await page.setViewport({{width: 1280, height: 720}});
await page.goto('http://localhost:4321/world', {{waitUntil: 'networkidle0', timeout: 30000}});
await page.waitForTimeout(5000); // 等渲染稳定
await page.screenshot({{path: 'tasks/world_render.png'}});
// 也拿 console 日志
page.on('console', msg => console.log('BROWSER:', msg.text()));
await browser.close();
```

4. 跑截图 → 看结果 → 如果还是灰白椭圆 → 修代码 → 重启 server → 再截图 → 循环

### 具体要修的点:

a. **颜色不对**: 读 src/lib/sph/pbr-gpu-pass.ts 的 renderCells()，找 uAlbedo uniform 的设置。
   CellData 里的 albedo 字段是 [r,g,b]，但 M1314c 加的 sdfBaseRadius 等字段可能没接进去。
   geometry.json 的 surface.albedo 可能没传到 shader。追踪完整数据流:
   loadGPUScene() → CellData.albedo → renderCells() → gl.uniform3fv(uAlbedo, ...)

b. **形态不对**: M1314d 改了 cellSDF() 但 u_lobeCount uniform 可能没被设置。
   搜 pbr-gpu-pass.ts 里 "u_lobeCount" 或 "u_baseRadius"，看 render 方法有没有 gl.uniform 调用。
   如果没有 → 那 shader 里的 uniform 永远是 0 → 永远走 fallback 简单形状。

c. **只显示几个 cell**: camScale 0.4 + bbH 3966 说明相机视野太大。
   或者大多数 cell 的 bbox 集中在某个区域互相遮挡。
   检查 composite_params.json 里 58 个 cell 的 bbox 分布。

d. **GLB mesh 位置错误**: CellMeshRenderer 加载了 5 个 GLB，但位置/缩放不对。
   检查 cell-mesh-renderer.ts 的 render() 方法，看它怎么定位 mesh。

5. 每次修改后:
   - 重启 dev server (kill %1 && npx astro dev --host 0.0.0.0 &)
   - 等15秒
   - 跑截图脚本
   - 检查截图
   - 如果还不对，继续修
   - 记录每轮修改到 tasks/render_fix_log.md

6. 完成标准:
   - 截图里能看到 58 个 cell
   - cell 有不同颜色 (不是全灰白)
   - cell 有有机形态 (metaball lobes，不是简单椭圆)
   - 没有奇怪的剑/柱子

7. 最终提交:
```bash
kill %1
cd astro-svgfigure && git pull origin cell-pubsub-loop
git add -A && git commit -m "M1315f: render fix — 58 colored organic metaball cells"
git push origin cell-pubsub-loop
```

关键: 别开新分支。别创建 v2/v3 后缀文件。必须循环验证到渲染正确才 push。
"""

# ── Worker G: verify-pbr.html 独立验证闭环 ──────────────────────────────────

TASK_G = f"""你是 astro-svgfigure 项目的 Claude Code 小弟。

强制要求: Claude Code 版本必须是 2.1.168。
```bash
{CC_PIN}
```

```bash
{CLONE}
git config user.name "claude-worker-g" && git config user.email "worker-g@astro"
```

## 任务: verify-pbr.html 必须渲染出3个有颜色有机 metaball cell

verify-pbr.html 是独立的 WebGL2 测试页面，不依赖 astro 管线。
如果这个页面能渲染正确，就证明 shader + 数据是对的，问题只在 astro 管线集成。

1. 启动 server:
```bash
npm install 2>&1 | tail -3
npx astro dev --host 0.0.0.0 &
sleep 15
```

2. 用 puppeteer 截图 verify-pbr.html:
```bash
npm install puppeteer
```
```javascript
// tasks/screenshot_pbr.mjs
import puppeteer from 'puppeteer';
const b = await puppeteer.launch({{headless:true, args:['--no-sandbox','--enable-webgl']}});
const p = await b.newPage();
p.on('console', m => console.log('BROWSER:', m.text()));
await p.setViewport({{width:1280, height:720}});
await p.goto('http://localhost:4321/verify-pbr.html', {{waitUntil:'networkidle0', timeout:30000}});
await new Promise(r => setTimeout(r, 3000));
await p.screenshot({{path:'tasks/verify_pbr_screenshot.png'}});
await b.close();
```

3. 跑截图 → 看结果:
   - 如果是黑屏或灰白 → shader 有问题
   - 如果能看到3个不同颜色的有机形态 → shader OK
   
4. 如果渲染不对，读 public/verify-pbr.html 的 shader 代码:
   - 检查 cellSDF() 函数: u_baseRadius, u_lobes[], smooth union k=0.3
   - 检查 uniform 是否正确设置: fetch geometry.json → parse → gl.uniform
   - 检查 albedo: surface.albedo [r,g,b] → uAlbedo → fragment output
   - 修复 → 重启 → 截图 → 循环

5. 完成标准: 截图里 3 个 cell 有不同颜色 + 有机 metaball 轮廓

6. 提交:
```bash
kill %1
cd astro-svgfigure && git pull origin cell-pubsub-loop
git add -A && git commit -m "M1315g: verify-pbr renders 3 colored organic metaball cells"
git push origin cell-pubsub-loop
```
"""

# ── Dispatch ───────────────────────────────────────────────────────────────

print("=== M1315: 赛博真实世界 ===")
print(f"org: {org_id}")
print(f"origin: {origin}")

print("\nDispatching F (core render fix, heavy agentic loop)...")
cf = create_conv()
print(f"  F conv: {cf}")
send(cf, TASK_F)
time.sleep(10)

print("\nDispatching G (verify-pbr standalone validation)...")
cg = create_conv()
print(f"  G conv: {cg}")
send(cg, TASK_G)

print("\nDone. Monitor git log for M1315f and M1315g")
