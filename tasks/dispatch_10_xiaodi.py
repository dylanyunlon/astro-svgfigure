#!/usr/bin/env python3
"""
管理者脚本 — 在你自己的机器上跑（需要能访问 claude.hk.cn）
串行派发10个小弟，每个小弟分析一个upstream仓库，写RESEARCH文档。

用法:
  cd astro-svgfigure
  python3 tasks/dispatch_10_xiaodi.py

每派一个小弟后暂停5秒，避免限频。
"""
import json, re, os, sys, time, urllib.request, urllib.error

# ── Auth ──────────────────────────────────────────────────────────────────
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          ".claude-hk-config")
with open(os.path.join(CONFIG_DIR, "raw_curl.txt")) as f:
    raw = f.read()

COOKIE  = re.search(r"-b '([^']+)'", raw).group(1)
ORG_ID  = re.search(r"organizations/([^/]+)", raw).group(1)
ORIGIN  = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
UA      = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

HEADERS = {
    "Content-Type": "application/json",
    "origin": ORIGIN, "user-agent": UA,
    "referer": f"{ORIGIN}/",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "Cookie": COOKIE,
}

# ── 10个小弟任务 ─────────────────────────────────────────────────────────
TASKS = [
    {
        "id": 111, "repo": "ogl",
        "prompt": """你是 xiaodi #111。分析 upstream/ogl/ 仓库（oframe/ogl）。

任务：
1. 列出 src/core/ 和 src/extras/ 的所有文件，说明每个类的用途
2. 找出与 Active Theory Hydra 引擎的 Stage/Renderer 对应的类
3. 说明 ogl 的 Geometry → Program → Mesh → Renderer 渲染管线
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 112, "repo": "uil",
        "prompt": """你是 xiaodi #112。分析 upstream/uil/ 仓库（lo-th/uil）。

任务：
1. 列出 src/proto/ 的所有UI组件
2. 与 Active Theory 的 UILPanel/UILPerformance/UILInputNumber 对应
3. 说明 Gui.js 的面板系统如何工作
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 113, "repo": "nanogl",
        "prompt": """你是 xiaodi #113。分析 upstream/nanogl/ 仓库。

任务：
1. 列出 src/ 的所有 WebGL wrapper 类
2. 说明 program.ts / fbo.ts / texture-2d.ts 的 API 设计
3. 与 Active Theory Hydra 的底层 GL 调用对应
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 114, "repo": "lygia",
        "prompt": """你是 xiaodi #114。分析 upstream/lygia/ 仓库的 lighting/ 和 sdf/ 目录。

任务：
1. 列出 lighting/ 下所有 .glsl 文件（PBR/diffuse/specular）
2. 列出 sdf/ 下所有 .glsl 文件（距离场函数）
3. 哪些函数适合移植到 astro-svgfigure 的 species shader
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 115, "repo": "comlink",
        "prompt": """你是 xiaodi #115。分析 upstream/comlink/ 仓库（GoogleChromeLabs）。

任务：
1. 阅读 src/comlink.ts 和 src/protocol.ts
2. 说明 Comlink 的 proxy/expose/transfer 机制
3. 与 Active Theory hydra-thread.js 的 receiveMessage/resolve/emit 协议对比
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 116, "repo": "webgl-water",
        "prompt": """你是 xiaodi #116。分析 upstream/webgl-water/ 仓库（evanw）。

任务：
1. 阅读 water.js 和 renderer.js
2. 说明水面反射/折射的实现（cubemap + FBO）
3. 与 Active Theory 的 WaterCeilingShader 功能对应
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 117, "repo": "webgl-noise",
        "prompt": """你是 xiaodi #117。分析 upstream/webgl-noise/ 仓库（ashima）。

任务：
1. 列出 src/ 下所有噪声函数（simplex/cellular/classic）
2. 说明 noise2D.glsl 和 cellular2D.glsl 的算法
3. 哪些噪声适合 astro-svgfigure 的 species 变异
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 118, "repo": "gaussian-splats-at",
        "prompt": """你是 xiaodi #118。分析 upstream/gaussian-splats-at/ 仓库（AT fork）。

任务：
1. 阅读 src/SplatMesh.js 和 src/worker/SortWorker.js
2. 说明 3D Gaussian Splatting 的 WebGL 渲染流程
3. SortWorker 的 WASM 排序与 AT 的 hydra-thread WASM 模块对比
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 119, "repo": "sketch-js",
        "prompt": """你是 xiaodi #119。分析 upstream/sketch-js/ 仓库（soulwire）。

任务：
1. 说明 sketch.js 的 setup/draw 循环和 Canvas/WebGL 上下文切换
2. 与 Active Theory 的 WorkDetailParticles 粒子系统对比
3. 适合 astro-svgfigure 的粒子装饰层哪些模式
4. 输出一份 200 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
    {
        "id": 120, "repo": "qrious",
        "prompt": """你是 xiaodi #120。分析 upstream/qrious/ 仓库（neocotic）。

任务：
1. 说明 QRious 的 Canvas 渲染 API
2. AT 网站为何只用了这一个第三方库
3. 能否替换为纯 SVG 生成（适配 astro-svgfigure）
4. 输出一份 100 字的 RESEARCH 摘要

只输出 RESEARCH 文档内容，markdown格式。"""
    },
]

# ── Dispatch ──────────────────────────────────────────────────────────────

def create_conversation(model="claude-sonnet-4-5-20250514"):
    data = json.dumps({"name": "", "model": model, "is_temporary": True}).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations",
        data=data, headers=HEADERS, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read()).get("uuid", "")


def send_message(conv_id, prompt, model="claude-sonnet-4-5-20250514"):
    data = json.dumps({
        "completion": {
            "model": model,
            "prompt": prompt,
            "timezone": "Asia/Shanghai"
        }
    }).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion",
        data=data,
        headers={**HEADERS, "accept": "text/event-stream"},
        method="POST")
    resp = urllib.request.urlopen(req, timeout=120)
    
    full_text = ""
    for line in resp:
        line = line.decode("utf-8", errors="replace").strip()
        if line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        full_text += delta.get("text", "")
            except:
                pass
    return full_text


def dispatch_xiaodi(task):
    xid = task["id"]
    repo = task["repo"]
    print(f"\n{'='*60}")
    print(f"[xiaodi #{xid}] Dispatching — repo: {repo}")
    print(f"{'='*60}")
    
    try:
        conv_id = create_conversation()
        print(f"[xiaodi #{xid}] conv: {conv_id}")
        
        result = send_message(conv_id, task["prompt"])
        print(f"[xiaodi #{xid}] Response: {len(result)} chars")
        
        # Save result
        out_path = f"upstream/RESEARCH_{xid}_{repo}.md"
        with open(out_path, "w") as f:
            f.write(result)
        print(f"[xiaodi #{xid}] Saved: {out_path}")
        
        return True
    except Exception as e:
        print(f"[xiaodi #{xid}] FAILED: {e}")
        return False


if __name__ == "__main__":
    print(f"Origin: {ORIGIN}")
    print(f"Org: {ORG_ID}")
    print(f"Tasks: {len(TASKS)}")
    print()
    
    success = 0
    for i, task in enumerate(TASKS):
        ok = dispatch_xiaodi(task)
        if ok:
            success += 1
        
        if i < len(TASKS) - 1:
            wait = 5
            print(f"\n... waiting {wait}s before next xiaodi ...")
            time.sleep(wait)
    
    print(f"\n{'='*60}")
    print(f"DONE: {success}/{len(TASKS)} xiaodi dispatched successfully")
    print(f"{'='*60}")
    print(f"\nNext: git add upstream/RESEARCH_*.md && git commit && git push")
