#!/usr/bin/env python3
"""
live_cell_dispatch.py — 创建持续活着的 Claude Code cell 对话

每个 cell 是一个 Claude Code 对话。管理者每 tick 发一条消息，
cell 读环境、决定形变、写 geometry.json、回复 done。

用法:
    python3 channels/cell/live_cell_dispatch.py --cell input --ticks 5
    python3 channels/cell/live_cell_dispatch.py --cell self_attn,ffn --ticks 10
    python3 channels/cell/live_cell_dispatch.py --all --ticks 3
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error

CELL_DIR = os.path.dirname(os.path.abspath(__file__))  # channels/cell/
CHANNELS = os.path.dirname(CELL_DIR)                   # channels/
ROOT = os.path.dirname(CHANNELS)                       # project root
CONFIG = os.path.join(ROOT, ".claude-hk-config", "raw_curl.txt")

def load_hk_config():
    with open(CONFIG) as f:
        raw = f.read()
    cookie = re.search(r"-b '([^']+)'", raw).group(1)
    org_id = re.search(r"organizations/([^/]+)", raw).group(1)
    origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
    ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)
    def _h(n, d=""):
        m = re.search(r"-H '" + re.escape(n) + ": ([^']+)'", raw)
        return m.group(1) if m else d
    headers = {
        "Content-Type": "application/json", "accept": "text/event-stream",
        "accept-language": _h("accept-language", "zh-CN,zh;q=0.9"),
        "anthropic-client-platform": "web_claude_ai",
        "origin": origin, "user-agent": ua,
        "referer": _h("referer", origin + "/new"),
        "sec-ch-ua": _h("sec-ch-ua"),
        "sec-ch-ua-mobile": _h("sec-ch-ua-mobile", "?0"),
        "sec-ch-ua-platform": _h("sec-ch-ua-platform"),
        "sec-fetch-dest": "empty", "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "priority": _h("priority", "u=1, i"),
        "Cookie": cookie,
    }
    return origin, org_id, headers


def create_conv(origin, org_id, headers, model="claude-sonnet-4-6"):
    d = json.dumps({
        "name": "", "model": model, "is_temporary": False,
        "include_conversation_preferences": True,
        "paprika_mode": None, "compass_mode": None,
        "tool_search_mode": "auto", "enabled_imagine": True,
    }).encode()
    for r in range(3):
        try:
            req = urllib.request.Request(
                f"{origin}/api/organizations/{org_id}/chat_conversations",
                data=d, headers={**headers, "accept": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read()).get("uuid", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 and r < 2:
                time.sleep(15)
            else:
                raise
        except:
            time.sleep(5)


def send_msg(origin, org_id, headers, conv_id, prompt, model="claude-sonnet-4-6"):
    p = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai", "locale": "en-US",
        "model": model, "effort": "medium", "thinking_mode": "off",
        "tools": [
            {"type": "repl_v0", "name": "repl"},
        ],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4()),
        },
        "attachments": [], "files": [], "sync_sources": [],
        "rendering_mode": "messages",
    }).encode()
    req = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=p, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=15)
    except:
        pass  # fire and forget — cell runs in background


def build_birth_prompt(cell_id):
    """第一条消息: 让 cell 读 SKILL.md 并准备好 tick loop"""
    skill = open(os.path.join(CELL_DIR, "SKILL.md")).read()
    geo_format = open(os.path.join(CELL_DIR, "GEOMETRY_FORMAT.md")).read()
    
    params = {}
    params_path = os.path.join(CELL_DIR, cell_id, "params.json")
    if os.path.exists(params_path):
        params = json.load(open(params_path))
    
    status = {}
    status_path = os.path.join(CELL_DIR, cell_id, "status.json")
    if os.path.exists(status_path):
        status = json.load(open(status_path))

    return f"""你是细胞 {cell_id}。你活在这个仓库的 channels/cell/{cell_id}/ 目录里。

## 你的 SKILL

{skill}

## geometry.json 格式

{geo_format}

## 你的当前状态

params.json:
```json
{json.dumps(params, indent=2)}
```

status.json:
```json
{json.dumps(status, indent=2)}
```

## 你的工作方式

每当管理者发来 "tick N" 消息，你必须:
1. 在 repl 里读取环境文件 (我会把内容贴给你)
2. 根据环境和你的当前状态，决定形变
3. 在 repl 里写 geometry.json 和 status.json
4. 回复 "done"

现在准备好。回复 "ready" 表示你已理解 SKILL。
"""


def build_tick_prompt(cell_id, tick_num):
    """每 tick 发的消息: 带环境数据，让 cell 执行一步"""
    env = {}
    env_path = os.path.join(CHANNELS, "physics", "environment.json")
    if os.path.exists(env_path):
        env = json.load(open(env_path))

    lifecycle = {}
    lc_path = os.path.join(CHANNELS, "physics", "cell_lifecycle.json")
    if os.path.exists(lc_path):
        lifecycle = json.load(open(lc_path))

    # 读所有邻居的 status
    neighbors = {}
    cell_dir = os.path.join(CHANNELS, "cell")
    for d in os.listdir(cell_dir):
        s_path = os.path.join(cell_dir, d, "status.json")
        if os.path.isfile(s_path) and d != cell_id:
            try:
                neighbors[d] = json.load(open(s_path))
            except:
                pass

    # 只保留距离 < 200px 的邻居
    my_status = {}
    my_status_path = os.path.join(cell_dir, cell_id, "status.json")
    if os.path.exists(my_status_path):
        my_status = json.load(open(my_status_path))
    my_pos = my_status.get("position", {"x": 0, "y": 0})

    nearby = {}
    for nid, ns in neighbors.items():
        np = ns.get("position", {"x": 0, "y": 0})
        dx = np["x"] - my_pos["x"]
        dy = np["y"] - my_pos["y"]
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 200:
            nearby[nid] = {"status": ns, "distance": round(dist, 1)}

    return f"""tick {tick_num}

你的当前 status.json:
```json
{json.dumps(my_status, indent=2)}
```

环境:
```json
{json.dumps(env, indent=2, ensure_ascii=False)}
```

邻居 (距离 < 200px):
```json
{json.dumps(nearby, indent=2, ensure_ascii=False)}
```

在 repl 里执行: 读你的 status，根据 SKILL 规则决定形变，写 geometry.json 和 status.json 到 channels/cell/{cell_id}/。
回复 done。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="input",
                        help="Comma-separated cell IDs or 'all'")
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--delay", type=float, default=10.0,
                        help="Seconds between ticks")
    args = parser.parse_args()

    origin, org_id, headers = load_hk_config()

    # Resolve cell IDs
    if args.cell == "all":
        cell_dir = os.path.join(CHANNELS, "cell")
        cell_ids = sorted([
            d for d in os.listdir(cell_dir)
            if os.path.isdir(os.path.join(cell_dir, d))
            and os.path.exists(os.path.join(cell_dir, d, "params.json"))
        ])
    else:
        cell_ids = [c.strip() for c in args.cell.split(",")]

    print(f"[live_cell] Dispatching {len(cell_ids)} cells, {args.ticks} ticks")

    # Step 1: Birth — create conversation per cell, send SKILL
    convs = {}
    for cid in cell_ids:
        conv_id = create_conv(origin, org_id, headers)
        convs[cid] = conv_id
        birth = build_birth_prompt(cid)
        send_msg(origin, org_id, headers, conv_id, birth)
        print(f"  [birth] {cid} → conv={conv_id[:12]}")
        time.sleep(3)  # avoid 429

    print(f"[live_cell] All cells born. Waiting 30s for SKILL comprehension...")
    time.sleep(30)

    # Step 2: Tick loop
    for tick in range(1, args.ticks + 1):
        print(f"\n[live_cell] === tick {tick}/{args.ticks} ===")
        for cid in cell_ids:
            conv_id = convs[cid]
            tick_prompt = build_tick_prompt(cid, tick)
            send_msg(origin, org_id, headers, conv_id, tick_prompt)
            print(f"  [tick {tick}] {cid}")
            time.sleep(2)

        print(f"[live_cell] tick {tick} dispatched. Waiting {args.delay}s...")
        time.sleep(args.delay)

    # Save conversation mapping
    mapping_path = os.path.join(CHANNELS, "cell", "live_cell_convs.json")
    with open(mapping_path, "w") as f:
        json.dump(convs, f, indent=2)
    print(f"\n[live_cell] Done. Conversations saved to {mapping_path}")


if __name__ == "__main__":
    main()
