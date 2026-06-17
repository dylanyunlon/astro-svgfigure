#!/usr/bin/env python3
"""
dispatch_m080_m089.py — 第二轮 10 并发小弟
M080-M089: 深化渲染管线 + edge粒子 + Theatre.js + MSDF
"""
import json, subprocess, os, uuid, re
from concurrent.futures import ThreadPoolExecutor, as_completed

HK_DIR = os.path.join(os.path.dirname(__file__), "..", ".claude-hk-config")
with open(os.path.join(HK_DIR, "raw_curl.txt")) as f:
    _raw = f.read()
COOKIE = re.search(r"-b '([^']+)'", _raw).group(1)
ORG = re.search(r"organizations/([^/]+)", _raw).group(1)
URL = "https://claude.hk.cn"
MODEL = "claude-opus-4-6"
BRANCH = "cell-pubsub-loop"
AUTHOR = "dylanyunlon <dogechat@163.com>"
REPO = "https://github.com/dylanyunlon/astro-svgfigure.git"

# clone 指令模板
CLONE = f"git clone {REPO}\ncd astro-svgfigure && git checkout {BRANCH}"
PUSH = lambda mid, msg: f'git add -A && git commit --author="{AUTHOR}" -m "{mid}: {msg}" && git push origin {BRANCH}'

TASKS = [
    {"id":"M080","prompt":f"""{CLONE}

任务: 打开 src/lib/renderers/pixi-cell-renderer.ts。当前 M052 创建了基础 Container。
你需要为每个 cell 添加: 1) 从 composite_params.json 读 species_params.corner_radius 设置圆角 2) 读 species_params.border_width 设置描边 3) 读 species_params.shadow_blur 创建 DropShadowFilter。
仅改 src/lib/renderers/pixi-cell-renderer.ts。

{PUSH('M080','pixi-cell-renderer — corner_radius + border + DropShadow from species_params')}"""},

    {"id":"M081","prompt":f"""{CLONE}

任务: src/lib/renderers/sdf-species-filter.ts 目前有 cil-eye 和 cil-bolt。
添加 cil-vector (森绿箭头场) 的 SDF Filter case。读 src/lib/shaders/cil-vector.frag，为它创建 PixiJS Filter class，从 species_params 注入 uniform (arrow_count, spread_angle, flow_speed)。
仅改 src/lib/renderers/sdf-species-filter.ts。

{PUSH('M081','cil-vector SDF Filter class — arrow field uniform injection')}"""},

    {"id":"M082","prompt":f"""{CLONE}

任务: 同上，添加 cil-plus (深红十字) 和 cil-arrow-right (蓝灰箭头) 的 SDF Filter case 到 sdf-species-filter.ts。
读对应 .frag 着色器，注入 uniform。
仅改 src/lib/renderers/sdf-species-filter.ts。

{PUSH('M082','cil-plus + cil-arrow-right SDF Filter classes')}"""},

    {"id":"M083","prompt":f"""{CLONE}

任务: src/lib/renderers/pixi-filters-registry.ts 管理每个 cell 的 filter chain。
确保: 1) 从 channels/physics/bloom_variants.json 读每个 species 的 bloom 参数 2) 从 channels/physics/species_visual_traits.json 读 glow 参数 3) 为每个 cell container 设置 [GlowFilter, AdvancedBloomFilter] chain，参数来自 JSON 不是硬编码。
仅改 src/lib/renderers/pixi-filters-registry.ts。

{PUSH('M083','pixi-filters-registry — bloom_variants.json + species_visual_traits.json → filter chain')}"""},

    {"id":"M084","prompt":f"""{CLONE}

任务: src/lib/renderers/pixi-filters-registry.ts 继续。
为 GodrayFilter 添加 per-species 支持: cil-eye 用冷色光束(靛蓝), cil-bolt 用暖色光束(琥珀)。参数从 channels/physics/species_visual_traits.json 的 godray 字段读取。
同时添加 KawaseBlurFilter 作为背景散景，强度从 channels/physics/dof_at_params.json 读取。
仅改 src/lib/renderers/pixi-filters-registry.ts。

{PUSH('M084','GodrayFilter per-species + KawaseBlurFilter DOF from JSON')}"""},

    {"id":"M085","prompt":f"""{CLONE}

任务: src/lib/particle/EdgeParticleSystem.ts 是 edge 粒子系统。
M041 实现了 transform feedback 基础。你需要: 1) 从 channels/edge/*/route.json 读 bezier 控制点 2) 粒子初始位置在 source cell 中心，沿 bezier 曲线流向 target cell 3) 粒子颜色用 source species 色到 target species 色的 lerp。
仅改 src/lib/particle/EdgeParticleSystem.ts。

{PUSH('M085','EdgeParticleSystem — bezier route.json → particle flow + species color lerp')}"""},

    {"id":"M086","prompt":f"""{CLONE}

任务: src/lib/particle/SplineEmitter.ts 是样条发射器。
确保它: 1) 从 edge route.json 的 points 数组构建 CatmullRom 样条 2) 发射粒子沿样条均匀分布 3) 粒子大小从 edge 的 weight 参数缩放。
参考 upstream/webgl2-particles 的 transform feedback 实现。
仅改 src/lib/particle/SplineEmitter.ts。

{PUSH('M086','SplineEmitter — CatmullRom from route.json + weight-scaled particles')}"""},

    {"id":"M087","prompt":f"""{CLONE}

任务: src/lib/renderers/theatre-epoch-timeline.ts 是 Theatre.js 集成。
M058 有基础 SheetObject。你需要: 1) 为每个 cell 创建独立 SheetObject，props 包含 x,y,width,height,opacity,glowIntensity 2) 从 channels/convergence/epoch_params/0/*.json 和 1/*.json 读取 epoch 0 和 epoch 1 的关键帧值 3) 创建 Sequence 在两组关键帧间插值。
仅改 src/lib/renderers/theatre-epoch-timeline.ts。

{PUSH('M087','Theatre.js — per-cell SheetObject + epoch_params keyframes + Sequence interpolation')}"""},

    {"id":"M088","prompt":f"""{CLONE}

任务: src/lib/renderers/theatre-epoch-cell-bridge.ts 桥接 Theatre.js 和 PixiJS。
确保: 1) Theatre.js sequence 每帧输出的 props 值实时写入对应 cell 的 PixiJS Container (position, scale, alpha) 2) glowIntensity 变化实时更新 GlowFilter.outerStrength 3) 用 theatre-js dataverse 的 onChange 监听实现 reactive 更新。
仅改 src/lib/renderers/theatre-epoch-cell-bridge.ts。

{PUSH('M088','theatre-epoch-cell-bridge — reactive props → PixiJS Container + GlowFilter sync')}"""},

    {"id":"M089","prompt":f"""{CLONE}

任务: src/lib/renderers/gl-text.ts 是 MSDF 文字渲染模块。
确保: 1) 读取 channels/cell/*/msdf.png 作为 MSDF atlas 纹理 2) 用 src/lib/shaders/msdf.frag + msdf.vert 创建 PixiJS Shader 3) 为每个 cell 的 label 创建 Mesh 替代普通 Text 4) 支持 fontSize 从 species_params 读取。
参考 upstream/activetheory-svg2msdf 和 upstream/ogl/examples/msdf-text.html。
仅改 src/lib/renderers/gl-text.ts。

{PUSH('M089','gl-text MSDF — msdf.png atlas + msdf shader → PixiJS Mesh labels')}"""},
]

def dispatch(task):
    tid = task["id"]
    # create conversation
    r = subprocess.run(["curl","-s","-X","POST",
        f"{URL}/api/organizations/{ORG}/chat_conversations",
        "-H","Content-Type: application/json",
        "-H",f"Cookie: {COOKIE}",
        "-H",f"Origin: {URL}",
        "-d",json.dumps({"name":f"astro {tid}","model":MODEL,"is_temporary":False})
    ], capture_output=True, text=True, timeout=20)
    try:
        cid = json.loads(r.stdout).get("uuid")
    except:
        return tid, False, r.stdout[:100]
    if not cid:
        return tid, False, r.stdout[:100]
    # send prompt
    hu, au = str(uuid.uuid4()), str(uuid.uuid4())
    r2 = subprocess.run(["curl","-s","-X","POST",
        f"{URL}/api/organizations/{ORG}/chat_conversations/{cid}/completion",
        "-H","Content-Type: application/json","-H","Accept: text/event-stream",
        "-H",f"Cookie: {COOKIE}","-H",f"Origin: {URL}",
        "-d",json.dumps({"prompt":task["prompt"],"timezone":"Asia/Shanghai","model":MODEL,
            "effort":"high","thinking_mode":"off",
            "tools":[{"type":"repl_v0","name":"repl"},{"type":"artifacts_v0","name":"artifacts"}],
            "turn_message_uuids":{"human_message_uuid":hu,"assistant_message_uuid":au},
            "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages"})
    ], capture_output=True, text=True, timeout=60)
    print(f"[{tid}] ✅ → {cid}")
    return tid, True, cid

if __name__ == "__main__":
    print(f"=== M080-M089 dispatch (10 concurrent) ===\nOrg: {ORG}\n")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(dispatch, t): t for t in TASKS}
        res = {}
        for f in as_completed(futs):
            tid, ok, info = f.result()
            res[tid] = (ok, info)
    print("\n=== Results ===")
    for k in sorted(res): print(f"  {'✅' if res[k][0] else '❌'} {k}: {res[k][1]}")
    print(f"\nSuccess: {sum(1 for v in res.values() if v[0])}/{len(TASKS)}")
