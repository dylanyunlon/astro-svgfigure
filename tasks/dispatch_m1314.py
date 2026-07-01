#!/usr/bin/env python3
"""
dispatch_m1314.py — 召唤4个Claude Code小弟执行M1314任务
直接使用 .claude-hk-config/raw_curl.txt 的完整请求格式
"""
import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error

PROJECT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT)  # astro-svgfigure root
CONFIG = os.path.join(ROOT, ".claude-hk-config", "raw_curl.txt")

GIT_TOKEN = os.environ.get("GIT_TOKEN", "<YOUR_GITHUB_TOKEN>")

# ─── Parse raw_curl.txt ────────────────────────────────────────────────────

with open(CONFIG) as f:
    raw = f.read()

cookie = re.search(r"-b '([^']+)'", raw).group(1)
org_id = re.search(r"organizations/([^/]+)", raw).group(1)
origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

def _h(name, default=""):
    m = re.search(rf"-H '{re.escape(name)}: ([^']+)'", raw)
    return m.group(1) if m else default

HEADERS = {
    "Content-Type": "application/json",
    "accept": "text/event-stream",
    "accept-language": _h("accept-language", "zh-CN,zh;q=0.9"),
    "anthropic-client-platform": "web_claude_ai",
    "origin": origin,
    "user-agent": ua,
    "referer": _h("referer", f"{origin}/new"),
    "sec-ch-ua": _h("sec-ch-ua", '"Google Chrome";"v="149"'),
    "sec-ch-ua-mobile": _h("sec-ch-ua-mobile", "?0"),
    "sec-ch-ua-platform": _h("sec-ch-ua-platform", '"Windows"'),
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "priority": _h("priority", "u=1, i"),
    "Cookie": cookie,
}


def create_conv(model="claude-sonnet-4-6"):
    """Create a new conversation, return conv_id"""
    data = json.dumps({
        "name": "", "model": model, "is_temporary": False,
        "include_conversation_preferences": True,
        "paprika_mode": None, "compass_mode": None,
        "tool_search_mode": "auto",
        "enabled_imagine": True,
    }).encode()
    for retry in range(3):
        try:
            req = urllib.request.Request(
                f"{origin}/api/organizations/{org_id}/chat_conversations",
                data=data,
                headers={**HEADERS, "accept": "application/json"},
                method="POST")
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read()).get("uuid", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 and retry < 2:
                print(f"  429 rate limit, waiting 15s (retry {retry+1})")
                time.sleep(15)
            else:
                print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
                raise
        except Exception as ex:
            if retry < 2:
                time.sleep(5)
            else:
                raise


def send_prompt(conv_id: str, prompt: str, model="claude-sonnet-4-6"):
    """Send prompt to conversation — fire and forget (SSE stream)"""
    payload = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai",
        "locale": "en-US",
        "model": model, "effort": "medium",
        "thinking_mode": "off",
        "tools": [
            {"type": "web_search_v0", "name": "web_search"},
            {"type": "repl_v0", "name": "repl"},
            {"type": "artifacts_v0", "name": "artifacts"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [],
        "sync_sources": [],
        "rendering_mode": "messages",
        "create_conversation_params": {
            "name": "", "model": model,
            "include_conversation_preferences": True,
            "paprika_mode": None, "compass_mode": None,
            "tool_search_mode": "auto",
            "is_temporary": False,
            "enabled_imagine": True,
        },
    }).encode()

    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers=HEADERS,
        method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        # Read first few bytes to confirm connection
        chunk = resp.read(200)
        print(f"  SSE started: {chunk[:80]}...")
        return True
    except Exception as e:
        print(f"  Send OK (fire-and-forget): {type(e).__name__}")
        return True


# ─── Task prompts ──────────────────────────────────────────────────────────

COMMON_PREFIX = f"""你是 astro-svgfigure 项目的 Claude Code 小弟。

第一步必须执行:
```bash
git clone https://{GIT_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure && git checkout cell-pubsub-loop
git config user.name "claude-worker-{{WORKER}}" && git config user.email "worker-{{WORKER}}@astro"
```

完成任务后:
```bash
cd /home/user/astro-svgfigure
git pull origin cell-pubsub-loop
git add -A && git commit -m "{{COMMIT_MSG}}" && git push origin cell-pubsub-loop
```

关键约束:
- 别开新分支
- 别创建文件名带 v2/v3/_new/_port 后缀的文件
- 先 git pull 再改再 push
"""

TASK_A = COMMON_PREFIX.replace("{WORKER}", "a").replace("{COMMIT_MSG}", "M1314a: verify-pbr reads real geometry.json — 3 cells metaball SDF") + """
## 任务: verify-pbr.html 接入真实 geometry.json

1. 读 public/verify-pbr.html (304行) — 目前 hardcode 了 cell uniform
2. 读 public/channels/cell/input/geometry.json — 真实 tick-runner 输出的 metaball SDF
3. 修改 verify-pbr.html:
   - fetch 3个cell的geometry.json (input, self_attn, ffn)
   - 用 geometry.json 的 sdf.base_radius, sdf.lobes[], surface.albedo 等字段替换 hardcoded uniform
   - lobes 数组传入 shader 作为 metaball smooth union 输入
   - 在同一个 canvas 上渲染3个不同形态的 metaball cell
4. 读 channels/cell/GEOMETRY_FORMAT.md — 里面有 cellSDF() 的 GLSL 实现
5. 成功后记录到 tasks/verify_pbr_geometry_result.md
"""

TASK_B = COMMON_PREFIX.replace("{WORKER}", "b").replace("{COMMIT_MSG}", "M1314b: fix pass chain — PBR+Composite minimum viable render") + """
## 任务: 修复 gpu-render-loop.ts 的 pass chain 断裂

核心问题: /world 页面看不到渲染输出。

1. 读 tasks/gpu_pipeline_audit.md — 管理者的诊断报告
2. 读 src/lib/sph/gpu-render-loop.ts (2349行) 的 frame() 方法
3. 问题清单:
   a. 20+ try/catch 静默吞错误 → pass变null被跳过 → drawCalls=0
   b. Composite pass 吃 null 纹理用 1x1 placeholder → 全黑
   c. frame() 里 Nuke.render() 可能覆盖 cell 画面
4. 修复方案:
   - PBR pass (line 305) 已经没有 try/catch — 确保它能工作
   - Composite: 如果只有 PBR 成功, 直接 blit PBR 的 FBO 到屏幕
   - 在 constructor 末尾加 pass status 日志
5. 只改 gpu-render-loop.ts 和 composite-gpu-pass.ts
6. 记录到 tasks/pass_chain_fix.md
"""

TASK_C = COMMON_PREFIX.replace("{WORKER}", "c").replace("{COMMIT_MSG}", "M1314c: loadGPUScene reads geometry.json SDF data into CellData") + """
## 任务: geometry.json → CellData 桥接

1. 读 src/pages/world/index.astro 的 loadGPUScene() (搜索这个函数名)
   目前只从 composite_params.json 读 bbox — 完全没读 geometry.json 的 SDF 数据
2. 读 src/lib/sph/gpu-render-loop.ts 的 CellData 接口 (文件开头)
3. 修改:
   a. CellData 接口加 SDF 字段: sdfBaseRadius?, sdfLobes?, sdfNoiseAmp?, surfaceGlowIntensity? 等
   b. loadGPUScene() 里对每个 cell 并发 fetch geometry.json:
      const geoPromises = cellIds.map(id => fetch('/channels/cell/' + id + '/geometry.json').then(r=>r.json()).catch(()=>null));
      const geos = await Promise.all(geoPromises);
   c. merge geo 字段到 CellData
4. 只改 world/index.astro 和 gpu-render-loop.ts 的 CellData 接口
5. 记录到 tasks/geometry_bridge_result.md
"""

TASK_D = COMMON_PREFIX.replace("{WORKER}", "d").replace("{COMMIT_MSG}", "M1314d: PBR shader metaball SDF — lobes + smooth union + surface noise") + """
## 任务: PBR shader 改造 metaball SDF

1. 读 src/lib/sph/pbr-gpu-pass.ts — 找 FRAG_SRC 里的 SDF 函数
2. 读 channels/cell/GEOMETRY_FORMAT.md — 里面有 cellSDF() GLSL 实现:
   metaball smooth union, base_radius + lobes, k=0.3, surface noise
3. 修改 pbr-gpu-pass.ts 的 FRAG_SRC:
   - 加 uniform: u_baseRadius, u_lobeCount, u_lobeAngle[8], u_lobeDistance[8], u_lobeRadius[8]
   - 加 noise uniform: u_noiseAmp, u_noiseFreq
   - 替换 SDF 函数为 cellSDF() (metaball smooth union)
   - 简单 2D noise 函数做 surface 微扰
4. 修改 render() 方法: 从 CellData 读 SDF 字段设 uniform (有 fallback 默认值)
5. 只改 pbr-gpu-pass.ts 一个文件
6. 记录到 tasks/pbr_sdf_upgrade.md
"""


# ─── Dispatch ──────────────────────────────────────────────────────────────

def dispatch(label: str, prompt: str, delay: float = 8.0):
    print(f"\n{'='*60}")
    print(f"  Dispatching {label}")
    print(f"{'='*60}")
    conv_id = create_conv()
    print(f"  conv_id: {conv_id}")
    if not conv_id:
        print("  FAILED to create conversation!")
        return None
    send_prompt(conv_id, prompt)
    print(f"  ✓ {label} dispatched")
    if delay > 0:
        print(f"  waiting {delay}s before next dispatch...")
        time.sleep(delay)
    return conv_id


if __name__ == "__main__":
    print(f"org_id:  {org_id}")
    print(f"origin:  {origin}")
    print(f"cookie:  {cookie[:40]}...")

    results = {}

    # Dispatch all 4 workers with delay between each
    results["A"] = dispatch("Worker A: verify-pbr × geometry.json", TASK_A)
    results["B"] = dispatch("Worker B: pass chain fix", TASK_B)
    results["C"] = dispatch("Worker C: geometry bridge", TASK_C)
    results["D"] = dispatch("Worker D: PBR shader metaball", TASK_D, delay=0)

    print(f"\n{'='*60}")
    print("  All workers dispatched!")
    print(f"{'='*60}")
    for k, v in results.items():
        status = f"conv={v[:12]}..." if v else "FAILED"
        print(f"  Worker {k}: {status}")
    print(f"\n  Monitor: check git log on cell-pubsub-loop for M1314[a-d] commits")
