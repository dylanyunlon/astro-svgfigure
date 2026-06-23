#!/usr/bin/env python3
"""
M846-M852 Dispatch: 7 cell rendering params
管理者 Claude 调度 7 个小弟 Claude，每个负责一个 cell 的 rendering + shader + filter 参数。
"""

import json
import time
import requests
import sys

# ============ 认证 ============
ORG_ID = "b73e2b37-5be1-45a5-beb0-4a1f1faceffc"
BASE = "https://claude.hk.cn"
COOKIE = "lastActiveOrg=b73e2b37-5be1-45a5-beb0-4a1f1faceffc; CH-prefers-color-scheme=light; ajs_anonymous_id=claudeai.v1.45779287-353c-4a79-b243-2a21950bf2c4; user-sidebar-visible-on-load=true; user-sidebar-pinned=false; share-session=kb2war01bl8dkqdjg1ri2qtva136uhto; lastActiveOrg=b73e2b37-5be1-45a5-beb0-4a1f1faceffc; _dd_s=aid=1ee01d97-720c-4dcf-b576-165917a53054&rum=2&id=df85da9b-6784-43b1-9fb8-3959962a5097&created=1782179062681&expire=1782180023952"

HEADERS = {
    "accept": "text/event-stream",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "content-type": "application/json",
    "origin": BASE,
    "referer": f"{BASE}/new",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
}

# ============ 7 个 cell 任务 ============
CELLS = {
    "input_embed":  {"milestone": "M846", "species": "cil-vector",      "color": "#2E7D32", "f0": 0.04, "semantic": "嵌入/投影"},
    "pos_encode":   {"milestone": "M847", "species": "cil-code",        "color": "#2E7D32", "f0": 0.04, "semantic": "位置编码"},
    "self_attn":    {"milestone": "M848", "species": "cil-eye",         "color": "#3F51B5", "f0": 0.04, "semantic": "自注意力"},
    "add_norm1":    {"milestone": "M849", "species": "cil-plus",        "color": "#C62828", "f0": 0.02, "semantic": "残差+归一化"},
    "ffn":          {"milestone": "M850", "species": "cil-bolt",        "color": "#FF6F00", "f0": 0.80, "semantic": "前馈网络"},
    "add_norm2":    {"milestone": "M851", "species": "cil-plus",        "color": "#C62828", "f0": 0.02, "semantic": "残差+归一化"},
    "output":       {"milestone": "M852", "species": "cil-arrow-right", "color": "#455A64", "f0": 0.06, "semantic": "输出"},
}

PROMPT_TEMPLATE = """你是 astro-svgfigure 项目 cell-pubsub-loop 分支的小弟 Claude。

你的任务：为 cell `{cell_id}` 生成 rendering params（shader_params + pixi_filters + render_params）。

## 背景
- 仓库: github.com/dylanyunlon/astro-svgfigure, 分支: cell-pubsub-loop
- 这个 cell 的 species 是 `{species}`，颜色 {color}，F0 反射率 {f0}，语义: {semantic}
- Epoch 10 已收敛，7 个 cell 的 bbox/opacity/species_params 已就位
- 但所有 cell 都缺少 rendering params (shader_params, pixi_filters, render_params)
- 你需要搜索学术论文中这类概念的典型可视化方式，然后生成参数

## 步骤

1. 用 web_search 搜索: "{semantic} visualization in transformer architecture diagram" 和 "CHI HCI academic figure {semantic} visual encoding"
2. 用 repl clone 仓库并读取当前参数:
```bash
git clone https://github.com/dylanyunlon/astro-svgfigure.git /tmp/astro
cd /tmp/astro && git checkout cell-pubsub-loop
cat channels/composite_params.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['cells']['{cell_id}'], indent=2))"
cat channels/physics/species_visual_traits.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d['{cell_id}'], indent=2))"
```

3. 基于搜索结果 + 当前参数，生成渲染参数 JSON:
```python
rendering_params = {{
    "shader_params": {{
        "sdf_type": "...",           # lygia SDF 函数名
        "sdf_radius": 0.0,          # SDF 半径
        "glow_intensity": 0.0,      # 发光强度
        "glow_color": [r, g, b],    # 发光颜色 RGB 0-1
        "fresnel_power": 0.0,       # 菲涅尔幂
        "noise_scale": 0.0,         # 噪声缩放
        "noise_speed": 0.0,         # 噪声速度
        "edge_softness": 0.0,       # 边缘柔和度
    }},
    "pixi_filters": {{
        "bloom": {{"threshold": 0.0, "bloomScale": 0.0, "brightness": 0.0}},
        "glow": {{"distance": 0, "outerStrength": 0.0, "innerStrength": 0.0, "color": "0x..."}},
        "drop_shadow": {{"offset": [0, 0], "blur": 0, "alpha": 0.0, "color": "0x..."}},
    }},
    "render_params": {{
        "z_index": 0,                # z 层序号
        "blend_mode": "normal",      # 混合模式
        "corner_radius": 8,          # 圆角半径
        "border_width": 1,           # 边框宽度
        "border_color": "...",       # 边框颜色
        "background_opacity": 0.0,   # 背景不透明度
        "label_font_size": 12,       # 标签字号
        "icon_scale": 1.0,           # icon 缩放
    }}
}}
```

4. 将结果写入仓库并 push:
```bash
cd /tmp/astro
git config user.email "cell-{cell_id}@astro.bot"
git config user.name "cell-{cell_id}-claude"
git remote set-url origin https://${{GH_TOKEN}}@github.com/dylanyunlon/astro-svgfigure.git

# 读取现有 composite_params.json
python3 << 'PYEOF'
import json

with open("channels/composite_params.json") as f:
    comp = json.load(f)

comp["cells"]["{cell_id}"]["rendering"] = rendering_params  # 你上面生成的参数

with open("channels/composite_params.json", "w") as f:
    json.dump(comp, f, indent=2, ensure_ascii=False)

# 同时更新 species_visual_traits
with open("channels/physics/species_visual_traits.json") as f:
    traits = json.load(f)

traits["{cell_id}"]["shader_params"] = rendering_params["shader_params"]
traits["{cell_id}"]["pixi_filters"] = rendering_params["pixi_filters"]

with open("channels/physics/species_visual_traits.json", "w") as f:
    json.dump(traits, f, indent=2, ensure_ascii=False)

print("OK: {cell_id} rendering params written")
PYEOF

git add channels/composite_params.json channels/physics/species_visual_traits.json
git commit -m "{milestone}: {cell_id} rendering params — shader + pixi_filters + render"
git pull --rebase origin cell-pubsub-loop
git push origin cell-pubsub-loop
```

重要约束:
- 别开新分支，直接 push 到 cell-pubsub-loop
- 别创建 port v2 v3 后缀文件
- 只生成 JSON 参数，不生成 SVG
- push 前先 pull --rebase 避免冲突
"""


def create_conversation():
    """创建新对话"""
    url = f"{BASE}/api/organizations/{ORG_ID}/chat_conversations"
    resp = requests.post(url, headers={
        "content-type": "application/json",
        "cookie": COOKIE,
        "origin": BASE,
        "user-agent": HEADERS["user-agent"],
    }, json={
        "name": "",
        "model": "claude-sonnet-4-6",
        "include_conversation_preferences": True,
        "is_temporary": False,
    })
    if resp.status_code in (200, 201):
        data = resp.json()
        return data.get("uuid")
    else:
        print(f"  创建对话失败: {resp.status_code} {resp.text[:200]}")
        return None


def send_completion(conv_id, prompt):
    """发送 completion 请求 (fire and forget)"""
    import uuid
    url = f"{BASE}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion"
    human_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())
    payload = {
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "locale": "en-US",
        "model": "claude-sonnet-4-6",
        "effort": "medium",
        "thinking_mode": "off",
        "tools": [
            {"type": "web_search_v0", "name": "web_search"},
            {"type": "repl_v0", "name": "repl"},
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
    try:
        resp = requests.post(url, headers={**HEADERS, "cookie": COOKIE}, json=payload, stream=True, timeout=10)
        # 只读前几行确认启动
        for i, line in enumerate(resp.iter_lines()):
            if i > 5:
                break
            if line:
                print(f"    stream[{i}]: {line[:100]}")
        return True
    except Exception as e:
        print(f"    completion 请求异常 (可能正常, fire-and-forget): {e}")
        return True  # fire and forget


def main():
    dispatch_log = {}
    
    print("=" * 60)
    print("M846-M852: 7 cell rendering params dispatch")
    print("=" * 60)
    
    for cell_id, info in CELLS.items():
        milestone = info["milestone"]
        print(f"\n[{milestone}] 调度小弟: {cell_id} ({info['species']})")
        
        # 1. 创建对话
        conv_id = create_conversation()
        if not conv_id:
            print(f"  ❌ 跳过 {cell_id}")
            continue
        
        print(f"  ✅ 对话: {conv_id}")
        conv_url = f"{BASE}/chat/{conv_id}"
        print(f"  🔗 {conv_url}")
        
        # 2. 构建 prompt
        prompt = PROMPT_TEMPLATE.format(
            cell_id=cell_id,
            species=info["species"],
            color=info["color"],
            f0=info["f0"],
            semantic=info["semantic"],
            milestone=milestone,
        )
        
        # 3. 发送 (fire and forget)
        print(f"  📤 发送任务...")
        send_completion(conv_id, prompt)
        
        dispatch_log[cell_id] = {
            "conv_id": conv_id,
            "milestone": milestone,
            "url": conv_url,
            "species": info["species"],
            "status": "dispatched",
        }
        
        # 间隔 5 秒避免限流
        print(f"  ⏳ 等待 5s...")
        time.sleep(5)
    
    # 保存 dispatch log
    log_path = "channels/convergence/dispatch_log_round5.json"
    with open(log_path, "w") as f:
        json.dump(dispatch_log, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Dispatch log 已保存: {log_path}")
    print(json.dumps(dispatch_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
