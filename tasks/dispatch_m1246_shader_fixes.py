#!/usr/bin/env python3
"""
M1246 dispatch: 3 sub-Claudes fix 6 GPU shader compile errors.

Task allocation:
  小弟A: ue-lumen-gi.ts (fragColor redeclaration in REFL_GLSL) + ue-atmosphere-sky.ts (fragColor undeclared)
  小弟B: at-volumetric-light.ts (sampleVal → tapVal reserved word) + at-water-surface.ts (sanitise varying→in bug)
  小弟C: at-flower-particle.ts (MAX_FRAGMENT_UNIFORM_VECTORS) + at-geometry-loader.ts (cameraPosition + uniform redefinition + dFdx/dFdy)

Each sub-Claude:
  1. git clone + checkout cell-pubsub-loop
  2. Read the file, understand the error
  3. Fix ONLY the shader compilation error
  4. git commit + push
"""

import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".claude-hk-config")
GIT_TOKEN = "GIT_TOKEN_FROM_ENV"

with open(os.path.join(CONFIG_DIR, "raw_curl.txt")) as f:
    RAW = f.read()

COOKIE = re.search(r"-b '([^']+)'", RAW).group(1)
ORG_ID = re.search(r"organizations/([^/]+)", RAW).group(1)
ORIGIN = re.search(r"-H 'origin: ([^']+)'", RAW).group(1)
UA = re.search(r"-H 'user-agent: ([^']+)'", RAW).group(1)

HEADERS = {
    "Content-Type": "application/json",
    "origin": ORIGIN, "user-agent": UA,
    "referer": f"{ORIGIN}/",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "Cookie": COOKIE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "小弟A",
        "label": "M1246a: fix LumenGI + AtmosphereSky shader compile",
        "prompt": """你是管理者Claude的小弟A。你的任务是修复2个GPU shader编译错误。

## 第一步: Clone 并切换分支
```bash
git clone https://""" + GIT_TOKEN + """@github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure && git checkout cell-pubsub-loop
git config user.name "sub-claude-A" && git config user.email "sub-a@astro.dev"
```

## 错误1: ue-lumen-gi.ts — lumen_radiance_collect.fs
```
ERROR: 0:133: 'gl_FragColor' : undeclared identifier
```
M1245已经添加了`#define cameraPosition uCameraPos`，但`compileShader`函数的sanitise只替换`gl_FragColor → fragColor`。问题可能在REFL_GLSL或FBR_GLSL被嵌入LIGHTING_MAIN_FRAG时，已有`out vec4 fragColor`声明但因为模板字符串嵌套导致重复或位置问题。

检查思路:
- 打开 `src/lib/sph/ue-lumen-gi.ts`
- 找到 LIGHTING_MAIN_FRAG (约355行)，它内联了 `${FBR_GLSL}` 和 `${REFL_GLSL}`
- 确认内联后的GLSL是否有`out vec4 fragColor`在正确位置
- 确认没有 `gl_FragColor` 残留——注意 MATCAP_GLSL / REFL_GLSL / FBR_GLSL 里可能有
- 检查 compileShader sanitiser 是否正确工作

## 错误2: ue-atmosphere-sky.ts — transLUT fragment shader
```
ERROR: 0:103: 'fragColor' : undeclared identifier
ERROR: 0:123: 'fragColor' : undeclared identifier
```
原因: transLUT shader 缺少 `out vec4 fragColor;` 声明。

检查思路:
- 打开 `src/lib/sph/ue-atmosphere-sky.ts`
- 找到 transLUT 相关的 fragment shader
- 确认每个 frag shader 顶部都有 `out vec4 fragColor;`
- 如果有 `gl_FragColor` 残留，替换为 `fragColor` 并确保声明存在

## 规则
- 别开新分支,直接在 cell-pubsub-loop 上改
- commit message 格式: `M1246a: fix LumenGI fragColor + AtmosphereSky fragColor undeclared`
- 只改必要的行，不要重构
- 改完 git push origin cell-pubsub-loop
"""
    },
    {
        "name": "小弟B",
        "label": "M1246b: fix VolumetricLight + WaterSurface shader compile",
        "prompt": """你是管理者Claude的小弟B。你的任务是修复2个GPU shader编译错误。

## 第一步: Clone 并切换分支
```bash
git clone https://""" + GIT_TOKEN + """@github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure && git checkout cell-pubsub-loop
git config user.name "sub-claude-B" && git config user.email "sub-b@astro.dev"
```

## 错误1: at-volumetric-light.ts — raymarch frag shader
```
ERROR: 0:96: 'sample' : Illegal use of reserved word
```
原因: GLSL ES 3.00 中 `sample` 是保留字。RAYMARCH_FRAG 第96行（内联RANGE_GLSL和SIMPLENOISE_GLSL后计算）的变量名 `sampleVal` 在某些GPU驱动上被错误tokenize。

修复: 打开 `src/lib/sph/at-volumetric-light.ts`，找到 RAYMARCH_FRAG 中的:
```glsl
vec4 sampleVal = texture(tRays, pos);
```
以及后续引用 `sampleVal` 的行:
```glsl
accumulated += sampleVal * decay * density * phaseNorm * uRayStrength;
```
将 `sampleVal` 重命名为 `tapVal` (tap 是 DSP/shader 术语，不是保留字)。

## 错误2: at-water-surface.ts — wave-step vert shader
```
ERROR: 0:6: 'assign' : l-value required (can't modify an input "vUv")
```
原因: `_compile` 函数的 `sanitise` 用 `label.includes('vert')` 判断是否是 vert shader。
但所有 shader 的 vert 和 frag 都经过同一个 `sanitise`，而 `sanitise` 对 vert source 也跑了。
QUAD_VERT 本身用的是 `out vec2 vUv` 而非 `varying`，所以不会触发 varying 替换。

但真正的问题可能是 `buildGerstnerVert()` 里调用了 `getShader('simplenoise.glsl')`，
而 `getShader` 返回的代码可能包含 `varying` 关键字，被 sanitise 错误替换成 `in`。
或者 `buildGerstnerVert()` 里的 `sampleDisp` 函数名触发了其他问题。

检查思路:
1. 打开 `src/lib/sph/at-water-surface.ts`
2. 找到 `_compile` 函数 (约710行)
3. 修复 sanitise 的 varying 替换逻辑——应该对 vert source 和 frag source 分别处理:
   - vert: `varying` → `out`
   - frag: `varying` → `in`
4. 关键修复: 不要用 label 判断，改为分别对 vert/frag 调用不同的 sanitise 版本:
```typescript
const sanitiseVert = (s: string) => s
  .replace(/\\bgl_FragColor\\b/g, 'fragColor')
  .replace(/\\btexture2D\\s*\\(/g, 'texture(')
  .replace(/\\btextureCube\\s*\\(/g, 'texture(')
  .replace(/\\battribute\\s+/g, 'in ')
  .replace(/\\bvarying\\s+/g, 'out ');

const sanitiseFrag = (s: string) => s
  .replace(/\\bgl_FragColor\\b/g, 'fragColor')
  .replace(/\\btexture2D\\s*\\(/g, 'texture(')
  .replace(/\\btextureCube\\s*\\(/g, 'texture(')
  .replace(/\\bvarying\\s+/g, 'in ');

const vertSrc = sanitiseVert(vert);
const fragSrc = sanitiseFrag(frag);
```

## 规则
- 别开新分支,直接在 cell-pubsub-loop 上改
- commit message: `M1246b: fix VolumetricLight reserved word + WaterSurface sanitise varying`
- 只改必要的行
- git push origin cell-pubsub-loop
"""
    },
    {
        "name": "小弟C",
        "label": "M1246c: fix FlowerParticle + GeometryLoader shader compile",
        "prompt": """你是管理者Claude的小弟C。你的任务是修复2个GPU shader编译错误。

## 第一步: Clone 并切换分支
```bash
git clone https://""" + GIT_TOKEN + """@github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure && git checkout cell-pubsub-loop
git config user.name "sub-claude-C" && git config user.email "sub-c@astro.dev"
```

## 错误1: at-flower-particle.ts — link error
```
FRAGMENT shader uniforms count exceeds MAX_FRAGMENT_UNIFORM_VECTORS(1024)
```
M1242已经把 `u_splinePts[1024]` → RGBA32F texture。但 pos-fbo program 仍然超出 uniform 限制。

检查思路:
- 打开 `src/lib/sph/at-flower-particle.ts`
- 找到 pos-fbo program 的 frag shader (约在 `_compilePrograms` 函数中)
- 查找是否还有大数组 uniform (如 `uniform vec4 xxx[N]` 其中 N 很大)
- 如果有，采用和 M1242 相同的策略：改为 texture lookup
- 或者减少 uniform 数量 —— 将多个 uniform 合并为 struct 或 texture

## 错误2: at-geometry-loader.ts — fragment shader
```
ERROR: 0:44: 'cameraPosition' : undeclared identifier
ERROR: 0:52: 'cameraPosition' : undeclared identifier
ERROR: 0:393: 'tMRO' : redefinition
ERROR: 0:394: 'tMatcap' : redefinition
ERROR: 0:395: 'tNormal' : redefinition
ERROR: 0:396: 'uLight' : redefinition
ERROR: 0:397: 'uColor' : redefinition
ERROR: 0:398: 'uNormalStrength' : redefinition
ERROR: 0:427: 'dFdx' : no matching overloaded function found
ERROR: 0:428: 'dFdy' : no matching overloaded function found
```
多个问题:
1. `cameraPosition` 未声明 — AT 全局变量，需要 `#define cameraPosition uCameraPos` 或者在 shader 顶部声明 `uniform vec3 cameraPosition;` 或 `uniform vec3 uCameraPos;` + define
2. uniform 重复定义 — shader 拼接时同一 uniform 被声明了两次 (tMRO, tMatcap, tNormal, uLight, uColor, uNormalStrength)
3. `dFdx`/`dFdy` 签名不匹配 — 在 GLSL ES 3.00 中参数类型必须匹配。如果传 `float` 给需要 `vec3` 的变量，或者反过来

修复思路:
1. 在 frag shader 顶部加 `uniform vec3 uCameraPos;` 和 `#define cameraPosition uCameraPos`
2. 用 dedup 逻辑去除重复 uniform 声明——在拼接 shader 字符串后，用正则去掉重复的 `uniform` 行
3. 修复 dFdx/dFdy 调用——确保传入的参数类型匹配 (vec3 → vec3, vec2 → vec2)

可以在 `_buildProgram` 函数中加一个 dedup 步骤:
```typescript
// Dedup uniform declarations
const dedup = (glsl: string): string => {
  const seen = new Set<string>();
  return glsl.split('\\n').filter(line => {
    const m = line.match(/^\\s*uniform\\s+\\w+\\s+(\\w+)/);
    if (m) {
      if (seen.has(m[1])) return false;
      seen.add(m[1]);
    }
    return true;
  }).join('\\n');
};
```

## 规则
- 别开新分支,直接在 cell-pubsub-loop 上改
- commit message: `M1246c: fix FlowerParticle uniform overflow + GeometryLoader undeclared/redefinition/dFdx`
- 只改必要的行
- git push origin cell-pubsub-loop
"""
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_one(task: dict) -> str:
    """Create a conv and fire the prompt. Returns conv_id."""
    # Create conversation
    create_data = json.dumps({
        "name": task["label"], "model": "claude-sonnet-4-6", "is_temporary": False
    }).encode()

    conv_id = ""
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations",
                data=create_data, headers=HEADERS, method="POST")
            resp = urllib.request.urlopen(req, timeout=30)
            conv_id = json.loads(resp.read()).get("uuid", "")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                print(f"  429 rate limit, waiting 10s (attempt {attempt+1})")
                time.sleep(10)
            else:
                raise

    if not conv_id:
        raise RuntimeError(f"Failed to create conversation for {task['name']}")

    # Send prompt
    payload = json.dumps({
        "prompt": task["prompt"],
        "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6",
        "effort": "high",
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
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers={**HEADERS, "accept": "text/event-stream"},
        method="POST")

    try:
        urllib.request.urlopen(req2, timeout=10)
    except Exception:
        pass  # fire and forget

    return conv_id


def main():
    print("=" * 60)
    print("M1246 Dispatch: 3 sub-Claudes × 2 shader fixes each")
    print("=" * 60)

    results = []
    for i, task in enumerate(TASKS):
        print(f"\n[{i+1}/3] Dispatching {task['name']}: {task['label']}")
        conv_id = dispatch_one(task)
        results.append({"name": task["name"], "label": task["label"], "conv_id": conv_id})
        print(f"  ✅ conv_id: {conv_id}")

        if i < len(TASKS) - 1:
            print("  Waiting 3s before next dispatch...")
            time.sleep(3)

    print("\n" + "=" * 60)
    print("All dispatched! Summary:")
    for r in results:
        print(f"  {r['name']}: {r['conv_id'][:16]}... — {r['label']}")
    print("=" * 60)

    # Save dispatch log
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "channels", "convergence", "dispatch_log_m1246.json")
    with open(log_path, "w") as f:
        json.dump({"round": "M1246", "tasks": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
