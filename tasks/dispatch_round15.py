#!/usr/bin/env python3
"""
Round 15 dispatch — 10 sub-Claudes for AT+UE rendering world pipeline.

Each sub-Claude gets:
  1. git clone + checkout (simple, like Round 1)
  2. A focused task on ONE specific file/module
  3. Must commit and push to cell-pubsub-loop directly

Usage:
  python3 tasks/dispatch_round15.py
  python3 tasks/dispatch_round15.py --dry-run
"""

import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HK_CONFIG = os.path.join(ROOT, ".claude-hk-config", "raw_curl.txt")
GIT_TOKEN = "GIT_TOKEN_FROM_ENV"
REPO_URL = f"https://{GIT_TOKEN}@github.com/dylanyunlon/astro-svgfigure.git"
BRANCH = "cell-pubsub-loop"
MODEL = "claude-sonnet-4-6"
DRY_RUN = "--dry-run" in sys.argv

# ─── Parse cookie ─────────────────────────────────────────────────────────────
with open(HK_CONFIG) as f:
    raw = f.read()

cookie = re.search(r"-b '([^']+)'", raw).group(1)
org_id = re.search(r"organizations/([^/]+)", raw).group(1)
origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

HEADERS = {
    "Content-Type": "application/json",
    "origin": origin, "user-agent": ua,
    "referer": f"{origin}/",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "Cookie": cookie,
}

# ─── Clone preamble (same for all sub-Claudes) ───────────────────────────────
CLONE_PREAMBLE = f"""先在 repl 里执行:
```
git clone {REPO_URL} project && cd project && git checkout {BRANCH}
git config user.email "sub-claude-r15@astro" && git config user.name "{{author}}"
```
然后执行你的任务。完成后:
```
cd project && git add -A && git commit -m "{{commit_msg}}" && git push origin {BRANCH}
```
注意：不要开新分支，不要加 v2 v3 port 后缀，直接在 cell-pubsub-loop 分支上修改并 push。"""

# ─── 10 Tasks ─────────────────────────────────────────────────────────────────
TASKS = [
    {
        "id": "r15-at-world-page",
        "author": "at-world-page",
        "commit_msg": "M1060: world page — wire AT render pipeline to canvas with real WebGL2",
        "prompt": """你的任务：让 src/pages/world/index.astro 真正跑起来 AT 渲染管线。

当前 world page 有 canvas 但没接通 WebGL2。你需要：

1. 读 `src/pages/world/index.astro` 和 `src/lib/sph/at-render-pipeline.ts`
2. 在 world page 的 `<script>` 中初始化 WebGL2 context
3. 用 `at-shader-loader.ts` 解析 `upstream/activetheory-assets/compiled.vs`（173 个 shader 用 {@} 分隔）
4. 创建一个最小的 AT 风格渲染循环：clear → draw fullscreen quad → bloom pass → present
5. 确保 canvas 全屏、背景黑色、有基础 bloom 后处理

参考 `upstream/activetheory-assets/uil-params.json` 中 UnrealBloomComposite 的参数。
不要改其他文件的核心逻辑，只在 world page 里接线。"""
    },
    {
        "id": "r15-at-geometry-draco",
        "author": "at-geometry-draco",
        "commit_msg": "M1061: draco geometry loader — decode AT .bin assets to GPU buffers",
        "prompt": """你的任务：让 `src/lib/sph/at-geometry-loader.ts` 真正能解码 upstream/activetheory-assets/geometry/ 下的 .bin 文件。

当前文件有框架但解码逻辑是 stub。你需要：

1. 读 `src/lib/sph/at-geometry-loader.ts`
2. AT 的 .bin 文件是 Draco 压缩的几何体（参考 `bin/msdfgen.linux` 和 `tasks/draco-decode-test.mjs`）
3. 实现 `loadDracoGeometry(url: string)` 返回 `{positions: Float32Array, normals: Float32Array, uvs: Float32Array, indices: Uint32Array}`
4. 用 draco3d npm 包或 WASM decoder
5. 写一个测试：加载 `jellyfish.bin` 并打印顶点数

关键文件是 upstream/activetheory-assets/geometry/ 下的 17 个 .bin。"""
    },
    {
        "id": "r15-at-pbr-material",
        "author": "at-pbr-material",
        "commit_msg": "M1062: AT PBR material — PhysicalShader from compiled.vs with KTX2 textures",
        "prompt": """你的任务：让 AT 的 PhysicalShader 真正工作。

1. 读 `upstream/activetheory-assets/compiled.vs`，找到 PhysicalShader 相关 shader（搜索 PhysicalShader）
2. 读 `src/lib/sph/at-pbr-material.ts` 和 `src/lib/sph/at-full-pbr-pipeline.ts`
3. 读 `upstream/activetheory-assets/uil-params.json` 中 PhysicalShader 的参数
4. 实现一个 `ATPhysicalMaterial` class，能：
   - 加载 KTX2 纹理（Combined + MRO + Normal 三件套）从 upstream/activetheory-assets/textures/
   - 设置 PBR uniforms (metalness, roughness, occlusion, envBlend)
   - 输出 WebGL2 可用的 program + uniform 绑定
5. 不需要完整渲染，先把材质数据结构和 shader 编译搞定

参考 upstream/activetheory-assets/textures/ 的命名规律：{MESH}___CyclesBake_COMBINED.ktx2 + PBR_AT_MRO.ktx2 + PBR_Normal.ktx2"""
    },
    {
        "id": "r15-at-bloom-chain",
        "author": "at-bloom-chain",
        "commit_msg": "M1063: AT Unreal bloom — full pyramid downsample/upsample chain",
        "prompt": """你的任务：实现 AT 风格的 Unreal Bloom 后处理链。

1. 读 `upstream/activetheory-assets/compiled.vs` 中的 UnrealBloom 相关 shader
2. 读 `upstream/activetheory-assets/uil-params.json` 中 UnrealBloomComposite 和 UnrealBloomLuminosity 的参数
3. 读 `src/lib/sph/at-unreal-bloom-pipeline.ts`（当前有框架）
4. 实现完整的 bloom chain：
   - Luminosity threshold pass (提取亮部)
   - 5 级 downsample (mip pyramid)
   - 5 级 upsample (tent filter)
   - Composite (bloom 叠加回原图)
5. 每个 pass 用 FBO ping-pong，参数从 uil-params 读取
6. 确保 bloomStrength, bloomRadius, bloomTintColor, luminosityThreshold 可调

AT 的 bloom 是 UnrealEngine 3-pass bloom 的 Web 移植版本。"""
    },
    {
        "id": "r15-at-volumetric",
        "author": "at-volumetric",
        "commit_msg": "M1064: AT volumetric light — god rays from compiled.vs",
        "prompt": """你的任务：实现 AT 的 VolumetricLight 效果。

1. 读 `upstream/activetheory-assets/compiled.vs`，搜索 VolumetricLight / godray 相关代码
2. 读 `upstream/activetheory-assets/uil-params.json` 中 VolumetricLight 参数（fDecay, fDensity, fExposure, fWeight）
3. 读 `src/lib/sph/at-volumetric-light.ts`（当前有框架）
4. 实现 ray-marching 体积光：
   - 从光源位置向屏幕空间投射
   - NUM_SAMPLES 步 ray march
   - 用 decay/density/weight/exposure 控制效果
5. 输出为一个 WebGL2 可用的 post-process pass

参考 `upstream/activetheory-assets/shaders/LightVolume.fs` 和 `LightVolume.vs`。"""
    },
    {
        "id": "r15-at-mousefluid",
        "author": "at-mousefluid",
        "commit_msg": "M1065: AT mouse fluid interaction — Navier-Stokes from compiled.vs",
        "prompt": """你的任务：实现 AT 的鼠标流体交互效果。

1. 读 `upstream/activetheory-assets/compiled.vs`，搜索 mousefluid / Navier-Stokes / fluid 相关 shader
2. 读 `src/lib/sph/at-mousefluid-import.ts`（M815 有框架）
3. 读 `src/lib/sph/at-navier-stokes.ts` 和 `src/lib/sph/at-navier-stokes-shaders.ts`
4. 实现：
   - 鼠标位置 → velocity splat (速度注入)
   - Jacobi 压力迭代求解
   - Advection (对流)
   - 结果作为 displacement texture 应用到场景
5. 需要 double-buffered FBO (ping-pong)
6. 参考 homeParticleCurl 系列参数（uCurlNoiseScale, uMouseStrength）

AT 的流体是屏幕空间 2D Navier-Stokes，不是 3D SPH。"""
    },
    {
        "id": "r15-at-spline-particles",
        "author": "at-spline-particles",
        "commit_msg": "M1066: AT spline particles — GPU transform feedback along flower spines",
        "prompt": """你的任务：实现 AT 的 GPU 样条粒子系统。

1. 读 `upstream/activetheory-assets/compiled.vs`，搜索 Particle / Spline / flower / spine 相关
2. 读 `upstream/activetheory-assets/geometry/flower_spine-128.bin` 和 `spine.bin`
3. 读 `src/lib/sph/at-spline-particles-full.ts`（M1047 有框架）
4. 实现：
   - 加载样条路径几何体
   - WebGL2 Transform Feedback 粒子更新
   - 粒子沿样条线流动 + curl noise 扰动
   - Point sprite 渲染（GL_POINTS + gl_PointSize）
5. 参考 FlowerParticleShader 和 homeParticleShapeu* 参数

AT 用 Transform Feedback 做粒子而不是 compute shader，因为是 WebGL2 而非 WebGPU。"""
    },
    {
        "id": "r15-at-glass-pbr",
        "author": "at-glass-pbr",
        "commit_msg": "M1067: AT glass PBR — refraction + fresnel from compiled.vs",
        "prompt": """你的任务：实现 AT 的玻璃 PBR 材质。

1. 读 `upstream/activetheory-assets/compiled.vs`，搜索 Glass / GlassShaderPBR / WorkGlassShader
2. 读 `upstream/activetheory-assets/uil-params.json` 中 GlassShaderPBR 和 WorkGlassShader 参数
3. 读 `src/lib/sph/at-glass-pbr-import.ts` 和 `src/lib/sph/at-glass-reflection-system.ts`
4. 实现：
   - Fresnel 反射 (uFresnelPow, uFresnelColor, uFresnelAlpha)
   - Refraction (uRefractionRatio, uDistortStrength)
   - Environment map blend (uEnvBlend)
   - Phong specular (uShininess, uSpecAdd, uPhongColor)
5. 输出为可复用的 GlassMaterial class

AT 的玻璃用 cubemap 反射 + screen-space refraction distortion。"""
    },
    {
        "id": "r15-at-scene-composite",
        "author": "at-scene-composite",
        "commit_msg": "M1068: AT scene compositor — multi-pass composite from uil-params",
        "prompt": """你的任务：实现 AT 的场景合成器。

1. 读 `upstream/activetheory-assets/uil-params.json`，理解 *CompositeuContrast / *CompositeuRGBStrength / *CompositeuVolumetricStrength 参数
2. 读 `src/lib/sph/at-scene-compositor.ts` 和 `src/lib/sph/at-scene-composite-shaders.ts`
3. AT 每个 scene (Home, CleanRoom, TreeScene, Work) 都有自己的合成参数
4. 实现 SceneCompositor class：
   - 输入: color buffer + bloom buffer + volumetric buffer + depth
   - 对比度调整 (uContrast)
   - RGB 增强 (uRGBStrength)
   - 体积光叠加 (uVolumetricStrength)
   - 最终 tonemap (ACES 或 Reinhard)
5. 参考 GlobalComposite 的全局参数 + 各场景特定参数

每个场景有不同的 "mood"，由这些合成参数决定。"""
    },
    {
        "id": "r15-at-msdf-text",
        "author": "at-msdf-text",
        "commit_msg": "M1069: AT MSDF text rendering — distance field text from compiled.vs",
        "prompt": """你的任务：实现 AT 的 MSDF 文字渲染。

1. 读 `upstream/activetheory-assets/compiled.vs`，搜索 msdf / DefaultText / GLUI
2. 读 `upstream/activetheory-svg2msdf/src/` 理解 SVG → MSDF 管线
3. 读 `src/lib/sph/at-text-rendering-msdf.ts`（有框架）
4. 读 `src/lib/shaders/msdf.frag` 和 `msdf.vert`
5. 实现：
   - MSDF atlas 加载 (from bin/msdfgen.linux 生成的纹理)
   - 文字布局计算 (kerning, line height)
   - GPU 渲染: instanced quads + MSDF fragment shader
   - 支持 cell label 渲染（输入 string，输出 WebGL mesh）
6. AT 不用 Canvas 2D 文字，它用 MSDF 距离场做 GPU 文字

参考 `upstream/activetheory-assets/shaders/DefaultText.fs` 的实现。"""
    },
]

# ─── Dispatch function ────────────────────────────────────────────────────────
def dispatch_one(task: dict) -> dict:
    """Create a conversation and send the task prompt."""
    task_id = task["id"]
    author = task["author"]
    commit_msg = task["commit_msg"]
    
    preamble = CLONE_PREAMBLE.format(author=author, commit_msg=commit_msg)
    full_prompt = preamble + "\n\n" + task["prompt"]
    
    if DRY_RUN:
        print(f"[DRY] {task_id}: would dispatch {len(full_prompt)} chars")
        return {"task_id": task_id, "status": "dry_run", "conv_id": "N/A"}
    
    try:
        # Step 1: Create conversation
        create_data = json.dumps({
            "name": "", "model": MODEL, "is_temporary": False
        }).encode()
        req = urllib.request.Request(
            f"{origin}/api/organizations/{org_id}/chat_conversations",
            data=create_data, headers=HEADERS, method="POST")
        resp = urllib.request.urlopen(req, timeout=30)
        conv_id = json.loads(resp.read()).get("uuid", "")
        
        # Step 2: Send prompt
        payload = json.dumps({
            "prompt": full_prompt,
            "timezone": "Asia/Shanghai",
            "model": MODEL,
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
            "attachments": [], "files": [], "rendering_mode": "messages",
        }).encode()
        
        req2 = urllib.request.Request(
            f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
            data=payload,
            headers={**HEADERS, "accept": "text/event-stream"},
            method="POST")
        
        # Fire and don't wait for full response
        try:
            urllib.request.urlopen(req2, timeout=10)
        except Exception:
            pass  # Expected — SSE streams time out, but the request is sent
        
        print(f"[OK] {task_id}: conv={conv_id}")
        return {"task_id": task_id, "status": "dispatched", "conv_id": conv_id}
        
    except Exception as e:
        print(f"[ERR] {task_id}: {e}")
        return {"task_id": task_id, "status": "error", "error": str(e)}

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== Round 15 Dispatch — {len(TASKS)} sub-Claudes ===")
    print(f"Origin: {origin}")
    print(f"Org: {org_id}")
    print(f"Model: {MODEL}")
    print(f"Dry run: {DRY_RUN}")
    print()
    
    results = []
    
    # Serial dispatch with 3s gap to avoid cookie race
    for i, task in enumerate(TASKS):
        print(f"[{i+1}/{len(TASKS)}] Dispatching {task['id']}...")
        result = dispatch_one(task)
        results.append(result)
        if not DRY_RUN and i < len(TASKS) - 1:
            time.sleep(3)  # Avoid cookie session conflicts
    
    # Write dispatch log
    log_path = os.path.join(ROOT, "channels", "convergence", "dispatch_log_round15.json")
    with open(log_path, "w") as f:
        json.dump({
            "round": 15,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": MODEL,
            "tasks": results,
        }, f, indent=2)
    
    print(f"\nDispatch log: {log_path}")
    print(f"Dispatched: {sum(1 for r in results if r['status'] == 'dispatched')}/{len(TASKS)}")
