#!/usr/bin/env python3
"""
dispatch_m070_m079.py — 10 并发小弟 Claude 调度
M070-M079: PixiJS 渲染管线全面接入 upstream 库

管理者: 主 Claude
执行者: 10 位小弟 Claude (claude.hk.cn)
"""

import json, time, subprocess, sys, os, uuid, re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────

HK_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", ".claude-hk-config")
COOKIE_FILE = os.path.join(HK_CONFIG_DIR, "raw_curl.txt")

ORG_ID = "995d5606-04f8-47f2-b4a7-17ef43ccde92"
BASE_URL = "https://claude.hk.cn"
MODEL = "claude-opus-4-6"

GIT_TOKEN = os.environ.get("GIT_TOKEN", "")
REPO_URL = f"https://{GIT_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git"
BRANCH = "cell-pubsub-loop"
AUTHOR = "dylanyunlon <dogechat@163.com>"

# ── Cookie 提取 ──────────────────────────────────────────────────────────────

def load_cookie():
    with open(COOKIE_FILE, "r") as f:
        raw = f.read()
    m = re.search(r"-b '([^']+)'", raw)
    if m:
        return m.group(1)
    raise RuntimeError("Cannot extract cookie from raw_curl.txt")

COOKIE = load_cookie()

# ── 10 个任务定义 ─────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "M070",
        "title": "M070: pixi-cell-renderer composite_params 全量接入",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M070。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: 打开 `src/lib/renderers/pixi-cell-renderer.ts`，确保它从 `/api/cells` 拿到的 `composite_params.json` 中每个 cell 的 `species_params` 都被正确读取并传给对应的 PixiJS Container。

当前状态: M052 已完成基础 Container/Sprite/Text 创建。你需要补全 species_params → PixiJS uniform 的映射，确保 opacity、corner_radius、glow_intensity 等参数从 JSON 驱动。

改动范围: 仅改 `src/lib/renderers/pixi-cell-renderer.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M070: pixi-cell-renderer composite_params 全量接入 — species_params → PixiJS uniform mapping" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M071",
        "title": "M071: cil-eye SDF Filter 挂载到 PixiJS Mesh",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M071。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: 打开 `src/lib/renderers/sdf-cell-renderer.ts` 和 `src/lib/shaders/cil-eye.frag`。确保 cil-eye species 的 SDF shader 被正确编译为 PixiJS Filter 并挂载到对应 cell 的 Mesh 上。

参考 M039 (commit b7811e0) 的实现。需要确保 uniform 从 `agent_params.json` 的 `species_params` 注入。

改动范围: `src/lib/renderers/sdf-cell-renderer.ts`, `src/lib/renderers/sdf-species-filter.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M071: cil-eye SDF Filter → PixiJS Mesh 挂载 + uniform injection from species_params" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M072",
        "title": "M072: cil-bolt SDF Filter 挂载",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M072。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: 对 `cil-bolt.frag` 做和 M071 相同的工作。确保 cil-bolt (琥珀闪电) species 的 SDF shader 挂载到 PixiJS Filter，uniform 从 species_params 注入。

参考 M045 (commit 247dc7b) 的实现。

改动范围: `src/lib/renderers/sdf-species-filter.ts` (添加 cil-bolt case)

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M072: cil-bolt SDF Filter → PixiJS Mesh 挂载 + amber lightning uniform" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M073",
        "title": "M073: AdvancedBloomFilter per-species 参数接入",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M073。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: 打开 `src/lib/renderers/pixi-filters-registry.ts`。M044 (commit c64fd04) 已经实现了 AdvancedBloomFilter per-species 基础。你需要确保 `channels/physics/bloom_variants.json` 中每个 species 的 bloom 参数 (threshold, bloomScale, brightness, blur) 被正确读取并应用到对应 cell 的 filter chain。

改动范围: `src/lib/renderers/pixi-filters-registry.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M073: AdvancedBloomFilter per-species params from bloom_variants.json → filter chain" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M074",
        "title": "M074: GlowFilter + GodrayFilter per-species 接入",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M074。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: M048 (GodrayFilter) 和 M044 (BloomFilter) 已经有基础实现。你需要在 `src/lib/renderers/pixi-filters-registry.ts` 中为每个 species 配置 GlowFilter 和 GodrayFilter 参数，从 `channels/physics/species_visual_traits.json` 读取。

每个 species 应有不同的 glow 颜色和强度。cil-eye 靛蓝辉光，cil-bolt 琥珀辉光。

改动范围: `src/lib/renderers/pixi-filters-registry.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M074: GlowFilter + GodrayFilter per-species — visual traits → filter params" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M075",
        "title": "M075: EdgeParticleSystem transform feedback 完善",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M075。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: M041 (commit 1a5d9f2) 和 M057 (commit 79dd7bd) 已实现 EdgeParticleSystem 基础和 edge-particle-bridge。你需要完善 `src/lib/particle/EdgeParticleSystem.ts`，确保:
1. 粒子沿 edge route.json 的 bezier 控制点流动
2. 粒子颜色从 source cell 的 species 色渐变到 target cell 的 species 色
3. 粒子速度和密度从 epoch_controller 同步

改动范围: `src/lib/particle/EdgeParticleSystem.ts`, `src/lib/particle/edge-particle-bridge.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M075: EdgeParticleSystem — bezier flow + species color gradient + epoch sync" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M076",
        "title": "M076: Theatre.js epoch SheetObject 完善",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M076。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: M042 (commit ce9880f) 和 M058 (commit 7356e53) 已实现 Theatre.js 基础集成。你需要完善 `src/lib/renderers/theatre-epoch-timeline.ts`，确保:
1. 每个 cell 的 SheetObject props (x, y, width, height, opacity, glow_intensity) 从 convergence/epoch_params/ 读取关键帧
2. epoch 0 → epoch 1 的 lerp 过渡平滑
3. M069 的 EpochPlaybackController 正确驱动 Theatre.js sequence

改动范围: `src/lib/renderers/theatre-epoch-timeline.ts`, `src/lib/renderers/theatre-epoch-cell-bridge.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M076: Theatre.js epoch SheetObject — keyframe from epoch_params + smooth lerp + playback sync" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M077",
        "title": "M077: MSDF 文字标签渲染接入 pixi-cell-renderer",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M077。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: 当前 cell 标签用的是普通 PixiJS Text。你需要改为 MSDF 渲染:
1. 读取 `channels/cell/*/msdf.png` 的 MSDF 纹理
2. 用 `src/lib/shaders/msdf.frag` + `msdf.vert` 创建 MSDF Text Mesh
3. 替换 pixi-cell-renderer 中的 Text 为 MSDF Mesh

参考 upstream/activetheory-svg2msdf 的实现和 upstream/ogl/examples/msdf-text.html。

改动范围: `src/lib/renderers/pixi-cell-renderer.ts`, `src/lib/renderers/gl-text.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M077: MSDF text label — msdf.png + msdf.frag/vert → PixiJS Mesh per cell" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M078",
        "title": "M078: CloudFogBackground 接入 species 色板",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M078。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: M049 (commit 789283a) 实现了 CloudFog 基础。你需要完善 `src/lib/CloudFogBackground.ts`，让云雾颜色从 `channels/physics/fog_at_params.json` 读取，并根据画面中 cell 的 species 分布动态调整:
1. 靠近 cil-eye cells 的区域偏靛蓝
2. 靠近 cil-bolt cells 的区域偏琥珀
3. 用 `src/lib/shaders/cloud-fog.frag` 的 uniform 驱动

改动范围: `src/lib/CloudFogBackground.ts`, `src/lib/CloudFog.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M078: CloudFogBackground — species-aware fog coloring from fog_at_params.json" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
    {
        "id": "M079",
        "title": "M079: epoch-ticker 帧循环接入所有动画子系统",
        "prompt": f"""你是 astro-svgfigure 项目的小弟 Claude #M079。

```bash
git clone {REPO_URL}
cd astro-svgfigure && git checkout {BRANCH}
```

任务: `src/lib/renderers/epoch-ticker.ts` 是主帧循环。你需要确保它在每个 tick 中按顺序调用:
1. EpochPlaybackController.update(dt) — M069 已实现
2. Theatre.js sequence tick — M076 正在完善
3. EdgeParticleSystem.update(dt) — M075 正在完善
4. CellInteraction hover/click 状态检测
5. PixiJS app.renderer.render(stage) 最终渲染

当前可能只调用了部分。确保全链路打通。

改动范围: `src/lib/renderers/epoch-ticker.ts`

完成后:
```bash
git add -A && git commit --author="{AUTHOR}" -m "M079: epoch-ticker — full frame loop: playback → theatre → particles → interaction → render" && git push origin {BRANCH}
```

只改代码，不写文档。"""
    },
]

# ── 创建新对话并发送 prompt ─────────────────────────────────────────────────

def create_conversation_and_send(task):
    """Create a new conversation and send the task prompt."""
    task_id = task["id"]
    prompt = task["prompt"]

    # Step 1: Create conversation
    conv_uuid = str(uuid.uuid4())
    create_url = f"{BASE_URL}/api/organizations/{ORG_ID}/chat_conversations"

    create_payload = {
        "name": f"astro-svgfigure {task_id}",
        "model": MODEL,
        "include_conversation_preferences": True,
        "is_temporary": False,
    }

    create_cmd = [
        "curl", "-s", "-X", "POST", create_url,
        "-H", "Content-Type: application/json",
        "-H", f"Cookie: {COOKIE}",
        "-H", f"Origin: {BASE_URL}",
        "-H", f"Referer: {BASE_URL}/new",
        "-d", json.dumps(create_payload),
    ]

    print(f"[{task_id}] Creating conversation...")
    result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=15)

    if result.returncode != 0:
        print(f"[{task_id}] FAIL create: {result.stderr[:200]}")
        return task_id, False, "create failed"

    try:
        conv_data = json.loads(result.stdout)
        conv_id = conv_data.get("uuid") or conv_data.get("id")
        if not conv_id:
            print(f"[{task_id}] FAIL: no uuid in response: {result.stdout[:200]}")
            return task_id, False, "no conv uuid"
    except json.JSONDecodeError:
        print(f"[{task_id}] FAIL parse: {result.stdout[:200]}")
        return task_id, False, "json parse error"

    print(f"[{task_id}] Conversation created: {conv_id}")

    # Step 2: Send prompt
    msg_url = f"{BASE_URL}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion"

    human_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())

    msg_payload = {
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "locale": "en-US",
        "model": MODEL,
        "effort": "high",
        "thinking_mode": "off",
        "tools": [
            {"type": "web_search_v0", "name": "web_search"},
            {"type": "repl_v0", "name": "repl"},
            {"type": "artifacts_v0", "name": "artifacts"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": human_uuid,
            "assistant_message_uuid": assistant_uuid,
        },
        "attachments": [],
        "files": [],
        "sync_sources": [],
        "rendering_mode": "messages",
    }

    send_cmd = [
        "curl", "-s", "-X", "POST", msg_url,
        "-H", "Content-Type: application/json",
        "-H", "Accept: text/event-stream",
        "-H", f"Cookie: {COOKIE}",
        "-H", f"Origin: {BASE_URL}",
        "-H", f"Referer: {BASE_URL}/chat/{conv_id}",
        "-d", json.dumps(msg_payload),
    ]

    print(f"[{task_id}] Sending prompt to {conv_id}...")
    result = subprocess.run(send_cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"[{task_id}] FAIL send: {result.stderr[:200]}")
        return task_id, False, "send failed"

    # Check for completion signal in SSE stream
    output = result.stdout
    if "message_limit" in output:
        print(f"[{task_id}] WARNING: message limit hit")
        return task_id, False, "message_limit"

    if "error" in output[:500].lower() and "rate_limit" in output[:500].lower():
        print(f"[{task_id}] WARNING: rate limited")
        return task_id, False, "rate_limited"

    print(f"[{task_id}] ✅ Dispatched successfully → conv={conv_id}")
    return task_id, True, conv_id


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("astro-svgfigure M070-M079: 10 并发小弟 Claude 调度")
    print("=" * 60)
    print(f"Org: {ORG_ID}")
    print(f"Model: {MODEL}")
    print(f"Branch: {BRANCH}")
    print(f"Tasks: {len(TASKS)}")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(create_conversation_and_send, t): t for t in TASKS}
        for future in as_completed(futures):
            task = futures[future]
            try:
                task_id, success, info = future.result()
                results[task_id] = (success, info)
            except Exception as e:
                results[task["id"]] = (False, str(e))
                print(f"[{task['id']}] EXCEPTION: {e}")

    print()
    print("=" * 60)
    print("调度结果汇总")
    print("=" * 60)
    for task_id in sorted(results.keys()):
        success, info = results[task_id]
        status = "✅" if success else "❌"
        print(f"  {status} {task_id}: {info}")

    success_count = sum(1 for s, _ in results.values() if s)
    print(f"\n成功: {success_count}/{len(TASKS)}")


if __name__ == "__main__":
    main()
