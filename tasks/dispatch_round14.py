#!/usr/bin/env python3
"""
Round 14 Dispatch — 6 sub-Claudes fix 59 merge conflict files.

Each sub-Claude:
  1. git clone the repo
  2. Resolve merge conflicts in its assigned ~10 files (accept HEAD version, remove markers)
  3. git push directly to cell-pubsub-loop branch
"""

import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error

REPO = "https://github.com/dylanyunlon/astro-svgfigure.git"
BRANCH = "cell-pubsub-loop"
GH_TOKEN = os.environ.get("GH_TOKEN", "")

BATCHES = {
    "M1060": {
        "desc": "merge-conflict-fix batch 1: AT loaders + pipeline",
        "files": [
            "src/lib/sph/BoundaryModel.ts",
            "src/lib/sph/adaptive-lod.ts",
            "src/lib/sph/at-geometry-loader.ts",
            "src/lib/sph/at-hydra-bloom-import.ts",
            "src/lib/sph/at-jellyfish-cell.ts",
            "src/lib/sph/at-render-pipeline.ts",
            "src/lib/sph/at-shadow-import.ts",
            "src/lib/sph/at-spline-particle.ts",
            "src/lib/sph/at-terrain-environment.ts",
            "src/lib/sph/at-text-rendering-msdf.ts",
        ],
    },
    "M1061": {
        "desc": "merge-conflict-fix batch 2: texture + collision + cell",
        "files": [
            "src/lib/sph/at-texture-loader.ts",
            "src/lib/sph/audio-reactive-visuals.ts",
            "src/lib/sph/cell-aura.ts",
            "src/lib/sph/cell-material-system.ts",
            "src/lib/sph/chromatic-adaptation.ts",
            "src/lib/sph/cinematic-camera.ts",
            "src/lib/sph/collision/ContactSolver.ts",
            "src/lib/sph/collision/SortAndSweep.ts",
            "src/lib/sph/collision/contact-generator.ts",
            "src/lib/sph/edge-data-flow-viz.ts",
        ],
    },
    "M1062": {
        "desc": "merge-conflict-fix batch 3: edge + epoch + effects",
        "files": [
            "src/lib/sph/edge-energy-flow.ts",
            "src/lib/sph/edge-flow-renderer.ts",
            "src/lib/sph/emissive-glow.ts",
            "src/lib/sph/epoch-physics-recorder.ts",
            "src/lib/sph/epoch-visual-sync.ts",
            "src/lib/sph/gpu-culling.ts",
            "src/lib/sph/holographic-display.ts",
            "src/lib/sph/interactive-fluid.ts",
            "src/lib/sph/lut-generator.ts",
            "src/lib/sph/magnetic-field-lines.ts",
        ],
    },
    "M1063": {
        "desc": "merge-conflict-fix batch 4: GPU pass + particles + bridge",
        "files": [
            "src/lib/sph/msdf-gpu-pass.ts",
            "src/lib/sph/nanogl-shader-executor.ts",
            "src/lib/sph/nature-texture-manager.ts",
            "src/lib/sph/neural-pathway-renderer.ts",
            "src/lib/sph/particle-effect-system.ts",
            "src/lib/sph/particle-gpu-pass.ts",
            "src/lib/sph/particle-instancing.ts",
            "src/lib/sph/particle-life-color.ts",
            "src/lib/sph/physics-render-bridge.ts",
            "src/lib/sph/pixi-gpu-bridge.ts",
        ],
    },
    "M1064": {
        "desc": "merge-conflict-fix batch 5: portal + species + shadow",
        "files": [
            "src/lib/sph/portal-effect.ts",
            "src/lib/sph/procedural-texture-atlas.ts",
            "src/lib/sph/reaction-diffusion-surface.ts",
            "src/lib/sph/ripple-effect.ts",
            "src/lib/sph/screen-space-reflection.ts",
            "src/lib/sph/screen-space-reflections.ts",
            "src/lib/sph/shadow-map.ts",
            "src/lib/sph/shadow-system.ts",
            "src/lib/sph/species-shader-registry.ts",
            "src/lib/sph/species-visual-dna.ts",
        ],
    },
    "M1065": {
        "desc": "merge-conflict-fix batch 6: topology + UE + remaining",
        "files": [
            "src/lib/sph/subsurface-scattering.ts",
            "src/lib/sph/topology-physics-sync.ts",
            "src/lib/sph/topology-transition-fx.ts",
            "src/lib/sph/transition-system.ts",
            "src/lib/sph/ue-atmosphere-sky.ts",
            "src/lib/sph/ue-megalights.ts",
            "src/lib/sph/uil-species-live.ts",
            "src/lib/sph/vfx-timeline.ts",
            "src/lib/sph/wireframe-overlay.ts",
        ],
    },
}


def dispatch_one(milestone: str, batch: dict) -> dict:
    """Dispatch one sub-Claude via claude.hk.cn to fix merge conflicts."""

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".claude-hk-config")
    raw_curl_path = os.path.join(config_dir, "raw_curl.txt")

    with open(raw_curl_path) as f:
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
        "name": "", "model": "claude-sonnet-4-6", "is_temporary": False
    }).encode()
    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations",
        data=create_data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    conv_id = json.loads(resp.read()).get("uuid", "")

    files_list = "\n".join(batch["files"])

    prompt = f"""你是 {milestone} 小弟。你的任务: 修复 merge conflict。

## 步骤

在 repl 里执行:

```bash
git clone https://{GH_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git repo
cd repo
git checkout cell-pubsub-loop
git config user.email "{milestone}@astro.bot"
git config user.name "{milestone}"
```

然后对以下每个文件, 用 python 脚本自动解决 merge conflict (保留 HEAD 版本):

```
{files_list}
```

解决方法 — 在 repl 里运行这个 python 脚本:

```python
import re, os

files = \"\"\"{files_list}\"\"\".strip().split("\\n")

for fpath in files:
    fpath = fpath.strip()
    if not os.path.exists(fpath):
        print(f"SKIP (not found): {{fpath}}")
        continue
    with open(fpath, 'r') as f:
        content = f.read()
    if '<<<<<<< ' not in content:
        print(f"SKIP (no conflict): {{fpath}}")
        continue
    # Accept HEAD (ours) version for each conflict block
    resolved = re.sub(
        r'<<<<<<< HEAD\\n(.*?)\\n=======\\n.*?\\n>>>>>>> [^\\n]+',
        r'\\1',
        content,
        flags=re.DOTALL
    )
    with open(fpath, 'w') as f:
        f.write(resolved)
    print(f"FIXED: {{fpath}}")

print("Done resolving conflicts")
```

然后 commit 并 push:

```bash
cd repo
git add -A
git commit -m "{milestone}: fix merge conflicts in {len(batch['files'])} files — {batch['desc']}"
git push origin cell-pubsub-loop
```

重要规则:
- 别开新分支, 直接 push 到 cell-pubsub-loop
- 别加 v2, v3, port 这些后缀
- 只修 merge conflict markers, 不改其他代码
- 完成后告诉我哪些文件修了"""

    # Step 2: Fire
    payload = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6", "effort": "medium",
        "thinking_mode": "off",
        "tools": [
            {"type": "repl_v0", "name": "repl"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [], "rendering_mode": "messages",
    }).encode()

    req2 = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers={**headers, "accept": "text/event-stream"},
        method="POST")
    try:
        urllib.request.urlopen(req2, timeout=10)
    except Exception:
        pass  # fire and forget

    result = {
        "conv_id": conv_id,
        "url": f"{origin}/chat/{conv_id}",
        "files": ", ".join(batch["files"][:3]) + f" ... ({len(batch['files'])} total)",
        "desc": batch["desc"],
    }

    print(
        f"[dispatch] {milestone}: conv={conv_id[:12]} "
        f"files={len(batch['files'])} desc={batch['desc']}",
        file=sys.stderr,
    )

    return result


def main():
    dispatch_log = {}

    for milestone, batch in BATCHES.items():
        try:
            result = dispatch_one(milestone, batch)
            dispatch_log[milestone] = result
            time.sleep(2)  # avoid rate limit
        except Exception as e:
            print(f"[ERROR] {milestone}: {e}", file=sys.stderr)
            dispatch_log[milestone] = {"error": str(e)}

    # Save dispatch log
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "channels", "convergence", "dispatch_log_round14.json"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(dispatch_log, f, indent=4, ensure_ascii=False)

    print(f"\n=== Round 14 Dispatch Complete ===")
    print(f"Dispatched {len([v for v in dispatch_log.values() if 'error' not in v])} sub-Claudes")
    print(f"Log saved to: {log_path}")
    print(json.dumps(dispatch_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
