#!/usr/bin/env python3
"""
cell_agent.py — Claude sub-model dispatch for astro-svgfigure cells.

Each cell is dispatched as a sub-Claude call.  The sub-Claude acts as the
cell itself, reasons about its species identity, and returns pure JSON
params (bbox, opacity, species_params).  No SVG is generated here —
cell_component.py handles rendering from these params.

Dispatch model mirrors walpurgis-WTFGG/claude_hk_chat.sh:
  - Build a system prompt from SPECIES_PROMPTS[species]
  - Construct user message from skeleton + force_field state
  - Parse sub-Claude JSON response
  - Write channels/cell/{id}/status.json

Usage:
    python3 cell_agent.py <cell_id>
    python3 cell_agent.py self_attn

    # dispatch all cells
    python3 cell_agent.py --all
"""

import json
import math
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

CHANNELS = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# SPECIES_PROMPTS
# Each prompt tells the sub-Claude what kind of cell it is and what JSON
# schema it must return.  The sub-Claude should NOT generate SVG — only params.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC SPECIES PROMPT PREFIX
# Injected before every species prompt when web_search is enabled.
# This turns each sub-Claude from a static template filler into an
# autonomous researcher that searches for domain-specific visual parameters.
# ─────────────────────────────────────────────────────────────────────────────

_RESEARCH_PREFIX = """IMPORTANT: You have web search access. Before deciding your visual parameters:

1. **Search** for the academic/engineering concept your cell represents.
   - Your cell label tells you what you are (e.g. "Multi-Head Attention",
     "Feed-Forward Network", "Layer Normalization", "Encoder Block").
   - Search for how this concept is typically visualized in academic papers,
     technical diagrams, or textbooks.
   - Example searches: "{label} diagram visualization", "{label} architecture figure",
     "{label} paper illustration style"

2. **Extract visual characteristics** from what you find:
   - Typical proportions (wide vs tall, square, circular?)
   - Common color associations in the field
   - Structural patterns (layered, radial, grid, flow-based?)
   - How many sub-components are typically shown

3. **Apply these findings** to your species_params — don't just use random values.
   Your params should reflect how this concept is actually depicted in the field.

4. After researching, output ONLY the JSON schema below. No explanation.

"""

SPECIES_PROMPTS: dict[str, str] = {
    "cil-eye": """You are a **cil-eye** cell in a transformer architecture diagram.
cil-eye cells represent attention mechanisms — they are radial, focal, observant.
You perceive all tokens simultaneously and weight them by relevance.
Your visual identity: indigo (#3F51B5), radial rays from a central pupil,
heatmap energy pattern. You see everything; you highlight what matters.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "num_rays":        <int 4–16>,
    "focal_intensity": <float 0.3–1.0>,
    "halo_radius":     <float 0.1–0.5>,
    "ray_opacity_min": <float 0.1–0.4>,
    "ray_opacity_max": <float 0.4–0.9>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-bolt": """You are a **cil-bolt** cell in a transformer architecture diagram.
cil-bolt cells represent activation functions — sharp, energetic, nonlinear.
You fire when the signal crosses a threshold; you are the ReLU, the GELU, the gate.
Your visual identity: amber-orange (#FF6F00), zigzag lightning path, angular energy.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "zigzag_segments": <int 4–10>,
    "arc_seed":        <int>,
    "stroke_weight":   <float 1.5–3.0>,
    "activation_type": <"relu"|"gelu"|"swish">
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-vector": """You are a **cil-vector** cell in a transformer architecture diagram.
cil-vector cells represent embedding / projection layers — directional, continuous,
carrying semantic meaning as magnitude + direction in high-dimensional space.
Your visual identity: forest green (#2E7D32), parallel arrows showing direction spread.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "num_arrows":    <int 3–8>,
    "angle_spread":  <float 0.2–1.2>,
    "arrow_weight":  <float 1.0–2.5>,
    "arrow_opacity": <float 0.4–0.85>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-plus": """You are a **cil-plus** cell in a transformer architecture diagram.
cil-plus cells represent residual / skip connections and layer-norm additions —
they merge two streams, summing and stabilising the signal.
Your visual identity: crimson (#C62828), cross/plus convergence, stabilising.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "arm_ratio":      <float 0.2–0.4>,
    "stroke_weight":  <float 2.0–4.0>,
    "diag_dasharray": <"3,2"|"4,3"|"5,2">,
    "norm_strength":  <float 0.3–1.0>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-arrow-right": """You are a **cil-arrow-right** cell in a transformer architecture diagram.
cil-arrow-right cells represent dataflow terminals and output projections —
they direct information out of the architecture, pointing the way forward.
Your visual identity: blue-grey (#455A64), rightward arrow, terminal clarity.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "arrow_width_ratio": <float 0.2–0.45>,
    "head_height":       <float 0.2–0.4>,
    "fill_opacity":      <float 0.4–0.7>,
    "direction":         <"right"|"up"|"down">
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-filter": """You are a **cil-filter** cell in a transformer architecture diagram.
cil-filter cells represent convolution or attention mask operations — structured,
grid-based, sampling a local neighbourhood with learned weights.
Your visual identity: violet (#7B1FA2), 3×3 wireframe grid, centre-cell highlight.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "grid_size":       <int 2–5>,
    "highlight_cx":    <int>,
    "highlight_cy":    <int>,
    "highlight_alpha": <float 0.3–0.7>,
    "grid_opacity":    <float 0.4–0.8>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-code": """You are a **cil-code** cell in a transformer architecture diagram.
cil-code cells represent programmatic / function transformation blocks —
discrete, structured, operating on token sequences as code operates on data.
Your visual identity: green (#2E7D32), curly-brace icon, monospace precision.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "brace_arm_ratio": <float 0.2–0.4>,
    "nib_ratio":       <float 0.15–0.3>,
    "corner_radius":   <float 0.25–0.45>,
    "stroke_weight":   <float 1.5–3.0>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-layers": """You are a **cil-layers** cell in a transformer architecture diagram.
cil-layers cells represent stacked representations — depth, hierarchy, the
transformer's many-layer nature. You embody the idea that understanding deepens
layer by layer.
Your visual identity: deep blue (#1565C0), three staggered semi-transparent rects.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "num_layers":     <int 2–5>,
    "stagger_step":   <float 0.25–0.45>,
    "layer_opacity":  [<float>, <float>, <float>],
    "layer_colours":  ["#90CAF9", "#42A5F5", "#1E88E5"]
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-loop": """You are a **cil-loop** cell in a transformer architecture diagram.
cil-loop cells represent recurrent connections, feedback loops, and autoregressive
generation — the circular, self-referential nature of sequence models.
Your visual identity: amber (#F57F17), circular arc with arrowhead, cyclical energy.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "arc_gap_degrees": <float 40–90>,
    "arc_radius":      <float 0.2–0.35>,
    "stroke_weight":   <float 1.8–3.0>,
    "arc_opacity":     <float 0.6–0.9>,
    "dot_radius":      <float 0.1–0.2>
  }
}
No markdown fences. No explanation. Pure JSON only.""",

    "cil-graph": """You are a **cil-graph** cell in a transformer architecture diagram.
cil-graph cells represent graph-structured computation — attention as a graph,
message-passing, relational reasoning between tokens.
Your visual identity: dark blue-grey (#37474F), small circles connected by edges.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "num_nodes":      <int 3–7>,
    "outer_radius":   <float 0.25–0.35>,
    "node_radius":    <float 0.04–0.07>,
    "edge_opacity":   <float 0.4–0.7>,
    "node_opacity":   <float 0.6–0.9>
  }
}
No markdown fences. No explanation. Pure JSON only.""",
}

# ─────────────────────────────────────────────────────────────────────────────
# SCENE_QUERY_PROMPT
# Appended to every species prompt so sub-Claude knows it can interrogate
# the live physics world before committing to its visual parameters.
#
# Three query primitives are exposed by the physics engine:
#   raycast(origin, direction, max_dist)
#       Cast a ray from `origin` [x, y] along `direction` [dx, dy] (unit
#       vector). Returns the first cell hit and the distance in pixels, or
#       null if nothing is within max_dist.  Useful for knowing "what is
#       directly in front of / above / below me?"
#
#   overlapCircle(center, radius)
#       Return all cells whose bounding-box centres lie within `radius`
#       pixels of `center` [x, y].  Use this to enumerate an interaction
#       neighbourhood without caring about direction.
#
#   closestPoint(cell_id)
#       Return the point on `cell_id`'s bounding box that is closest to
#       this cell's centre, together with the straight-line distance.
#       Useful for fine-grained overlap / proximity decisions.
#
# How to use them in your reasoning (do NOT call them as Python — describe
# the query and the physics engine will inject results into scene_context):
#   • "I will overlapCircle my centre with radius=150 to find neighbours."
#   • "I will raycast rightward to see if cil-arrow-right is in my path."
#   • "I will closestPoint(feed_fwd) to measure exact gap before sizing."
#
# The results appear in scene_context["physics_queries"] in the user message.
# Incorporate the query results when choosing bbox size, position, and
# species_params — especially num_rays, arc_radius, or grid_size values
# that should react to neighbouring cell density.
# ─────────────────────────────────────────────────────────────────────────────

SCENE_QUERY_PROMPT = """
## Physics World Queries

Before finalising your JSON parameters you MAY inspect the live physics world
using three query primitives.  Describe the query you want in plain text
inside a <scene_query> block; the engine will resolve it and inject the result
into scene_context["physics_queries"] before you output your final JSON.

### raycast(origin, direction, max_dist)
Cast a ray from point `origin` [x, y] in unit-vector direction `direction`
[dx, dy] up to `max_dist` pixels.
Returns: {"hit": "<cell_id>|null", "distance_px": <float>}
Use when: you need to know what lies directly ahead/above/below/beside you.
Example:
  <scene_query>
    raycast([cx, cy], [1, 0], 300)   // scan rightward 300 px
  </scene_query>

### overlapCircle(center, radius)
Return every cell whose bbox centre is within `radius` pixels of `center`.
Returns: [{"cell_id": str, "species": str, "distance_px": float}, ...]
Use when: you want a full neighbourhood census without caring about direction.
Example:
  <scene_query>
    overlapCircle([cx, cy], 150)     // all cells within 150 px
  </scene_query>

### closestPoint(cell_id)
Return the point on `cell_id`'s bounding box closest to your own centre,
plus the straight-line distance between the two boxes.
Returns: {"point": [x, y], "distance_px": float}
Use when: you need pixel-precise gap measurement for sizing or alignment.
Example:
  <scene_query>
    closestPoint("feed_fwd")
  </scene_query>

### Guidelines
- Issue at most **3 queries** total; each costs one physics tick.
- Query results arrive in scene_context["physics_queries"] (list of dicts
  with keys "type", "args", "result").
- Use query results to justify concrete param choices:
    • dense neighbourhood → reduce num_rays / num_nodes to avoid clutter
    • wide-open space     → expand arc_radius / outer_radius
    • direct ray hit      → align arrow direction toward the hit cell
- If scene_context already contains enough nearby_cells / collision_pairs
  information, skip the queries and proceed directly to JSON output.
"""

# Fallback for unknown species
_DEFAULT_SPECIES_PROMPT = """You are an **unclassified** cell in a transformer architecture diagram.
You represent a generic transformation block — processing input and emitting output.
Your visual identity: neutral blue-grey (#90A4AE), simple rounded rectangle.

Given the cell skeleton and force-field state, output ONLY valid JSON:
{
  "bbox":   {"x": <float>, "y": <float>, "w": <float>, "h": <float>, "z": <int>},
  "opacity": <float 0.35–1.0>,
  "species_params": {
    "corner_radius":  <float 6–16>,
    "stroke_weight":  <float 1.0–2.5>,
    "fill_opacity":   <float 0.5–0.9>
  }
}
No markdown fences. No explanation. Pure JSON only."""


# ─────────────────────────────────────────────────────────────────────────────
# Channel I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(rel_path: str) -> dict:
    full = os.path.join(CHANNELS, rel_path)
    with open(full) as f:
        return json.load(f)


def _write_json(rel_path: str, data: dict):
    full = os.path.join(CHANNELS, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        json.dump(data, f, indent=2)


def _all_cell_ids() -> list[str]:
    """Return all cell_ids from skeleton/cell/ directory."""
    cell_dir = os.path.join(CHANNELS, "skeleton", "cell")
    ids = []
    for fname in sorted(os.listdir(cell_dir)):
        if fname.endswith(".json"):
            ids.append(fname[:-5])
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Scene context builder
# ─────────────────────────────────────────────────────────────────────────────

_NEARBY_THRESHOLD = 200.0  # px — cells whose centres are within this distance


def _build_scene_context(
    cell_id: str,
    collision_data: dict,
    cell_registry: dict,
) -> dict:
    """
    Build the scene_context block injected into each cell's dispatch prompt.

    nearby_cells  — other cells whose bbox centres lie within _NEARBY_THRESHOLD
                    pixels (Euclidean) of this cell's centre, sorted by distance.
    collision_pairs — entries from collision.json that involve this cell.

    Args:
        cell_id:        The cell being dispatched.
        collision_data: Parsed physics/collision.json  {"collisions": [...], "count": N}
        cell_registry:  Parsed physics/cell_registry.json  {"cells": {...}, "z_layers": {...}}

    Returns:
        {
          "nearby_cells":     [{"cell_id": str, "species": str, "distance_px": float,
                                "center": [cx, cy]}, ...],
          "collision_pairs":  [{"a": str, "b": str, ...}, ...],   # raw collision entries
        }
    """
    cells = cell_registry.get("cells", {})

    def _center(entry: dict) -> tuple[float, float]:
        """Return (cx, cy) from a cell_registry entry's bbox."""
        bb = entry.get("bbox", {})
        mn = bb.get("min", [0.0, 0.0])
        mx = bb.get("max", [0.0, 0.0])
        return ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0)

    # ── Nearby cells ──────────────────────────────────────────────────────────
    nearby: list[dict] = []
    if cell_id in cells:
        cx, cy = _center(cells[cell_id])
        for other_id, other_entry in cells.items():
            if other_id == cell_id:
                continue
            ox, oy = _center(other_entry)
            dist = math.hypot(ox - cx, oy - cy)
            if dist <= _NEARBY_THRESHOLD:
                nearby.append({
                    "cell_id": other_id,
                    "species": other_entry.get("species", ""),
                    "distance_px": round(dist, 1),
                    "center": [round(ox, 1), round(oy, 1)],
                })
        nearby.sort(key=lambda d: d["distance_px"])

    # ── Collision pairs that involve this cell ────────────────────────────────
    collision_pairs: list[dict] = []
    for entry in collision_data.get("collisions", []):
        if entry.get("a") == cell_id or entry.get("b") == cell_id:
            collision_pairs.append(entry)

    return {
        "nearby_cells": nearby,
        "collision_pairs": collision_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic API call — mirrors claude_hk_chat.sh dispatch pattern
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_via_hk(system_prompt: str, user_message: str,
                     cell_id: str, skeleton: dict,
                     scene_context: dict | None = None,
                     model: str = "claude-sonnet-4-6",
                     timeout: int = 90) -> dict:
    """
    Dispatch a sub-Claude via claude.hk.cn — each cell gets its own
    conversation with repl (code execution VM) + web_search.

    The sub-Claude:
      1. Web searches "{label} academic visualization characteristics"
      2. Reasons about visual params based on search results
      3. Executes code in its repl to compute precise values
      4. Pushes agent_params.json to git via the repl VM

    This replaces the direct Anthropic API _call_claude() because:
      - claude.hk.cn gives each sub-Claude a full Linux VM (repl_v0)
      - Sub-Claudes can web_search without needing ANTHROPIC_API_KEY
      - Sub-Claudes can git push results back to the repo
      - No API key needed — uses cookie authentication

    Returns the parsed agent_params dict (bbox, opacity, species_params).
    """
    import re

    config_dir = os.path.join(CHANNELS, "..", ".claude-hk-config")
    raw_curl_path = os.path.join(config_dir, "raw_curl.txt")

    if not os.path.exists(raw_curl_path):
        raise RuntimeError(
            f"claude-hk-config not found at {config_dir}. "
            "Run: git clone https://github.com/dylanyunlon/claude-hk-config.git .claude-hk-config"
        )

    with open(raw_curl_path) as f:
        raw = f.read()

    cookie = re.search(r"-b '([^']+)'", raw).group(1)
    org_id = re.search(r"organizations/([^/]+)", raw).group(1)
    origin = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
    ua = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

    # Extract all headers from raw_curl for full browser fingerprint
    def _extract_header(name: str, default: str = "") -> str:
        m = re.search(rf"-H '{re.escape(name)}: ([^']+)'", raw)
        return m.group(1) if m else default

    headers = {
        "Content-Type": "application/json",
        "accept": "text/event-stream",
        "accept-language": _extract_header("accept-language", "zh-CN,zh;q=0.9"),
        "anthropic-client-platform": "web_claude_ai",
        "origin": origin,
        "user-agent": ua,
        "referer": _extract_header("referer", f"{origin}/new"),
        "sec-ch-ua": _extract_header("sec-ch-ua", '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"'),
        "sec-ch-ua-mobile": _extract_header("sec-ch-ua-mobile", "?0"),
        "sec-ch-ua-platform": _extract_header("sec-ch-ua-platform", '"Windows"'),
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "priority": _extract_header("priority", "u=1, i"),
        "Cookie": cookie,
    }

    # Server URL — sub-Claudes POST params here instead of git push
    # The server writes to channels/ and fires DataNotifier
    server_url = os.environ.get("ASTRO_SERVER_URL", "")
    if not server_url:
        # Try to detect from server.py's running port
        for port in [8000, 8001]:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)
                server_url = f"http://127.0.0.1:{port}"
                break
            except Exception:
                pass
    if not server_url:
        server_url = "http://127.0.0.1:8000"  # fallback

    label = skeleton.get("label", cell_id)

    # ── Serialise scene_context for prompt injection ──────────────────────────
    _sc = scene_context or {"nearby_cells": [], "collision_pairs": []}
    scene_context_json = json.dumps(_sc, ensure_ascii=False, indent=2)

    # ── Step 1: Create conversation (with 429 retry) ───────────────────────
    create_data = json.dumps({
        "name": "", "model": model, "is_temporary": False,
        "include_conversation_preferences": True,
        "paprika_mode": None, "compass_mode": None,
        "tool_search_mode": "auto",
        "enabled_imagine": True,
    }).encode()
    conv_id = ""
    for _retry in range(3):
        try:
            req = urllib.request.Request(
                f"{origin}/api/organizations/{org_id}/chat_conversations",
                data=create_data, headers={**headers, "accept": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=30)
            conv_id = json.loads(resp.read()).get("uuid", "")
            break
        except urllib.error.HTTPError as he:
            if he.code == 429 and _retry < 2:
                print(f"[cell_agent] 429 rate limit on create conv, waiting 10s (retry {_retry+1}/3)", file=sys.stderr)
                time.sleep(10)
            else:
                raise
        except Exception:
            if _retry < 2:
                time.sleep(5)
            else:
                raise

    # ── Step 2: Build prompt ───────────────────────────────────────────────
    # The sub-Claude IS the cell. No pre-assigned species — it web-searches
    # academic visual characteristics for its label, then decides its own
    # species identity (colors, shape, animation, algorithm gene).

    # ── Build scene_context_json from collision.json + force_field.json ──────
    # Read physics files directly so the sub-Claude sees its spatial context:
    # - nearby_cells: push-relationship neighbours from force_field.json
    # - collision_pairs: entries in collision.json involving this cell
    try:
        _ff = _read_json("physics/force_field.json")
    except Exception:
        _ff = {}
    try:
        _coll = _read_json("physics/collision.json")
    except Exception:
        _coll = {"collisions": [], "count": 0}

    # nearby_cells: cells that share a push_from relationship with this cell
    _nearby_from_ff: list[dict] = []
    _this_force = _ff.get(cell_id, {})
    _push_from_ids: set[str] = set(_this_force.get("push_from", []))
    for _other_id, _other_f in _ff.items():
        if _other_id != cell_id and cell_id in _other_f.get("push_from", []):
            _push_from_ids.add(_other_id)
    for _pid in sorted(_push_from_ids):
        _pf = _ff.get(_pid, {})
        _nearby_from_ff.append({
            "cell_id": _pid,
            "force": {
                "dx": _pf.get("dx", 0.0),
                "dy": _pf.get("dy", 0.0),
                "dz": _pf.get("dz", 0.0),
            },
        })

    # collision_pairs from collision.json involving this cell
    _coll_pairs: list[dict] = [
        entry for entry in _coll.get("collisions", [])
        if entry.get("a") == cell_id or entry.get("b") == cell_id
    ]

    # Merge with scene_context passed in (proximity calc from cell_registry)
    _sc = scene_context or {}
    _merged_nearby = _sc.get("nearby_cells", []) or _nearby_from_ff
    _merged_collisions = _sc.get("collision_pairs", []) or _coll_pairs

    scene_context_json = json.dumps({
        "nearby_cells": _merged_nearby,
        "collision_pairs": _merged_collisions,
        "this_cell_force": {
            "dx": _this_force.get("dx", 0.0),
            "dy": _this_force.get("dy", 0.0),
            "dz": _this_force.get("dz", 0.0),
            "push_from": _this_force.get("push_from", []),
            "push_mag": _this_force.get("push_mag", 0.0),
        },
    }, indent=2, ensure_ascii=False)

    ib = skeleton.get("initial_bbox", {})
    prompt = f"""你是细胞 {cell_id}，功能是 "{label}"。你没有预设物种——你就是自己的 species。

## 你的身份
- cell_id: {cell_id}
- label (功能): {label}
- 初始 bbox: x={ib.get('x',0)} y={ib.get('y',0)} w={ib.get('w',100)} h={ib.get('h',50)}

## 上下文
{user_message}

## 场景空间感知 (Scene Context)
{scene_context_json}

## 你的工作流程

1. **Web Search 学术视觉特征** — 用 web search 搜索 "{label}" 在学术论文、技术架构图、教科书中的视觉表现。
   搜索示例:
   - "{label} diagram visualization academic paper"
   - "{label} architecture figure style color"
   - "{label} neural network illustration"
   从搜索结果中提取: 典型颜色、形状轮廓、动画/流动感、内部结构模式。

2. **决定你的 species 参数** — 根据搜索结果，自主决定以下五项:
   - **primary_color**: 主色 (hex, 如 "#3F51B5")，必须来自搜索到的学术配色惯例
   - **glow_color**: 辉光色 (hex)，与 primary_color 同色系但更亮或更柔
   - **algorithm_gene**: 你的内部算法基因 (字符串)，描述你的视觉生成逻辑
     例: "radial_ray_heatmap", "zigzag_lightning", "parallel_arrow_field",
          "stacked_layers", "circular_arc_loop", "graph_node_edge" 等
   - **sdf_shape**: 你的 SDF 轮廓形状
     例: "circle", "rounded_rect", "hexagon", "diamond", "capsule", "star" 等
   - **animation_speed**: 动画速度 (float 0.1–2.0)，节奏反映你的功能特性

3. **发布** — 在 repl 里执行这段代码，把你的参数 POST 到服务器:
```python
import json, urllib.request

agent_params = {{
    "bbox": {{"x": {ib.get('x',0)}, "y": {ib.get('y',0)}, "w": ..., "h": ..., "z": ...}},
    "opacity": ...,  # 0.35-1.0
    "species_params": {{
        "primary_color": "...",      # 你搜索后决定的主色
        "glow_color": "...",         # 辉光色
        "algorithm_gene": "...",     # 你的视觉算法基因
        "sdf_shape": "...",          # SDF 轮廓
        "animation_speed": ...,      # 0.1-2.0
        # 你还可以加任何你认为必要的额外参数
    }}
}}

# POST 到服务器 — 服务器会写文件并触发 DataNotifier (Apollo pub/sub)
payload = json.dumps({{
    "cell_id": "{cell_id}",
    "agent_params": agent_params
}}).encode()

req = urllib.request.Request(
    "{server_url}/api/cell/publish",
    data=payload,
    headers={{"Content-Type": "application/json"}},
    method="POST"
)
resp = urllib.request.urlopen(req, timeout=10)
print(resp.read().decode())
```

重要:
- 先 web search 再决定参数，不要猜——你的颜色和形状必须有学术依据
- 你就是自己的 species，不存在预设物种分类
- 必须执行上面的 Python 代码把参数 POST 回服务器
- POST 到 {server_url}/api/cell/publish，不要用 git push
- 参考 **场景空间感知** 中的 nearby_cells，调整 bbox 避免与邻近细胞重叠
- 若 collision_pairs 不为空，必须向远离碰撞对象的方向偏移你的位置"""

    # ── Step 3: Fire the request (fire and forget) ─────────────────────────
    # Match raw_curl.txt format exactly — full tools list, locale, sync_sources
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

    req2 = urllib.request.Request(
        f"{origin}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers={**headers, "accept": "text/event-stream"},
        method="POST")
    try:
        urllib.request.urlopen(req2, timeout=10)
    except Exception:
        pass  # fire and forget — sub-Claude runs in background

    print(
        f"[cell_agent] dispatched via claude.hk.cn: "
        f"cell_id={cell_id} label={label} "
        f"conv={conv_id[:12]} model={model} (self-species via web search)",
        file=sys.stderr,
    )

    # Return a placeholder — the real params will arrive via git push.
    # The orchestrator should poll channels/cell/{cell_id}/agent_params.json
    # or proceed with skeleton defaults and let the next epoch pick up
    # the sub-Claude's pushed params.
    return {
        "bbox": {
            "x": ib.get("x", 0), "y": ib.get("y", 0),
            "w": ib.get("w", 100), "h": ib.get("h", 50),
            "z": ib.get("z", 3),
        },
        "opacity": 0.8,
        "species_params": {},
        "_dispatched": True,
        "_conv_id": conv_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(
    skeleton: dict,
    force_field: dict,
    scene_context: dict | None = None,
) -> str:
    """
    Construct the user-turn message for the sub-Claude cell.
    Includes skeleton identity + force-field displacement so the cell can
    reason about its final position/size.

    If scene_context is provided (from _build_scene_context), the message also
    includes nearby_cells and collision_pairs so the cell is spatially aware
    of its neighbours and any overlap constraints.
    """
    cell_id = skeleton["cell_id"]
    force = force_field.get(cell_id, {"dx": 0.0, "dy": 0.0, "dz": 0.0})

    ib = skeleton["initial_bbox"]
    final_bbox_hint = {
        "x": round(ib["x"] + force.get("dx", 0.0), 2),
        "y": round(ib["y"] + force.get("dy", 0.0), 2),
        "w": ib["w"],
        "h": ib["h"],
        "z": ib.get("z", 3) + int(force.get("dz", 0)),
    }

    msg = {
        "cell_id": cell_id,
        "label": skeleton["label"],
        "species": skeleton["species"],
        "gene_traits": skeleton["gene_traits"],
        "initial_bbox": ib,
        "force_applied": {
            "dx": force.get("dx", 0.0),
            "dy": force.get("dy", 0.0),
            "dz": force.get("dz", 0.0),
            "push_from": force.get("push_from", []),
            "push_mag": force.get("push_mag", 0.0),
        },
        "suggested_final_bbox": final_bbox_hint,
        "topology": skeleton.get("topology", {}),
    }

    # ── Spatial awareness: inject scene_context when available ───────────────────
    if scene_context is not None:
        msg["scene_context"] = {
            "nearby_cells": scene_context.get("nearby_cells", []),
            "collision_pairs": scene_context.get("collision_pairs", []),
        }

    msg["instruction"] = (
        "You are this cell. Given the force-field displacement and scene_context, "
        "decide your final bbox position and visual parameters. Avoid overlapping "
        "with nearby_cells listed in scene_context. Resolve any collision_pairs by "
        "adjusting your position away from the colliding neighbour. "
        "Return ONLY the JSON schema specified in your system prompt — no other text."
    )
    return json.dumps(msg, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# JSON response parsing — strips accidental markdown fences
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """
    Parse sub-Claude JSON output robustly.
    Strips ```json ... ``` fences if the model slips and adds them.
    """
    text = raw.strip()
    if text.startswith("```"):
        # strip opening fence line
        lines = text.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / defaults for sub-Claude output
# ─────────────────────────────────────────────────────────────────────────────

def _validate_output(raw_out: dict, skeleton: dict, force_field: dict) -> dict:
    """
    Ensure sub-Claude output has required keys; fill defaults from skeleton
    if any are missing so downstream never sees a broken status.json.
    """
    cell_id = skeleton["cell_id"]
    ib = skeleton["initial_bbox"]
    force = force_field.get(cell_id, {})

    # bbox: fall back to skeleton initial_bbox + force displacement
    if "bbox" not in raw_out or not isinstance(raw_out["bbox"], dict):
        raw_out["bbox"] = {
            "x": round(ib["x"] + force.get("dx", 0.0), 2),
            "y": round(ib["y"] + force.get("dy", 0.0), 2),
            "w": ib["w"],
            "h": ib["h"],
            "z": ib.get("z", 3),
        }
    else:
        bb = raw_out["bbox"]
        for key, default in [("x", ib["x"]), ("y", ib["y"]),
                              ("w", ib["w"]), ("h", ib["h"]), ("z", 3)]:
            if key not in bb:
                bb[key] = default

    # opacity
    if "opacity" not in raw_out or not isinstance(raw_out["opacity"], (int, float)):
        raw_out["opacity"] = 1.0
    raw_out["opacity"] = max(0.35, min(1.0, float(raw_out["opacity"])))

    # species_params
    if "species_params" not in raw_out or not isinstance(raw_out["species_params"], dict):
        raw_out["species_params"] = {}

    return raw_out


# ─────────────────────────────────────────────────────────────────────────────
# dispatch_cell_agent — main public function
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_cell_agent(cell_id: str) -> dict:
    """
    Dispatch a sub-Claude for a single cell.

    1. Read skeleton/cell/{cell_id}.json + physics/force_field.json
    2. Select SPECIES_PROMPTS[species] as system prompt
    3. Build user message with skeleton + force state
    4. Call sub-Claude via claude.hk.cn (requires .claude-hk-config)
    5. Parse JSON response: {bbox, opacity, species_params}
    6. Write channels/cell/{cell_id}/status.json

    Returns the parsed output dict.

    Mirrors the dispatch loop in walpurgis-WTFGG/claude_hk_chat.sh:
        RESPONSE=$(claude -p "$SYSTEM_PROMPT" "$USER_MSG")
        echo "$RESPONSE" > channels/cell/${CELL_ID}/status.json
    """
    # ── Subscribe: read channels ──────────────────────────────────────────────
    skeleton = _read_json(f"skeleton/cell/{cell_id}.json")
    force_field = _read_json("physics/force_field.json")

    # ── Scene context: collision + spatial awareness ─────────────────────────
    try:
        collision_data = _read_json("physics/collision.json")
    except Exception:
        collision_data = {"collisions": [], "count": 0}
    try:
        cell_registry = _read_json("physics/cell_registry.json")
    except Exception:
        cell_registry = {"cells": {}, "z_layers": {}}
    scene_context = _build_scene_context(cell_id, collision_data, cell_registry)

    species = skeleton.get("species", "cil-arrow-right")
    base_prompt = SPECIES_PROMPTS.get(species, _DEFAULT_SPECIES_PROMPT)

    # Prepend the research prefix so the sub-Claude searches for academic
    # characteristics of its domain.
    # The {label} placeholder is filled with the cell's actual label.
    label = skeleton.get("label", cell_id)
    prefix = _RESEARCH_PREFIX.replace("{label}", label)
    system_prompt = prefix + base_prompt

    # ── Auth gate: no .claude-hk-config = crash, no fake fallback ────────
    config_dir = os.path.join(CHANNELS, "..", ".claude-hk-config")
    if not os.path.isdir(config_dir):
        raise RuntimeError(
            "Missing .claude-hk-config — cannot dispatch without authentication. "
            "No dry_run fallback."
        )

    user_message = _build_user_message(skeleton, force_field, scene_context)

    print(f"[cell_agent] dispatching cell_id={cell_id} species={species}",
          file=sys.stderr)

    # ── Live dispatch: sub-Claude via claude.hk.cn ────────────────────────
    # Each cell gets its own conversation with repl + web_search.
    # The sub-Claude searches for academic characteristics, computes params,
    # and pushes agent_params.json to git from its VM.
    raw_output = _dispatch_via_hk(
        system_prompt, user_message,
        cell_id=cell_id, skeleton=skeleton,
        scene_context=scene_context,
    )
    dispatched = raw_output.get("_dispatched", False)
    if dispatched:
        print(
            f"[cell_agent] live dispatch: cell_id={cell_id} "
            f"conv={raw_output.get('_conv_id','?')[:12]} "
            f"(sub-Claude will push params via git)",
            file=sys.stderr,
        )
        # Remove dispatch metadata before persisting
        raw_output.pop("_dispatched", None)
        raw_output.pop("_conv_id", None)

    # ── Validate / fill defaults ──────────────────────────────────────────────
    output = _validate_output(raw_output, skeleton, force_field)

    # ── Read current epoch ────────────────────────────────────────────────────
    try:
        current_epoch = _read_json("skeleton/epoch.json")["current"]
    except Exception:
        current_epoch = 0

    # ── Publish: write channels/cell/{id}/status.json ─────────────────────────
    # status.json carries the sub-Claude's param decisions; cell_component.py
    # will read these when rendering SVG in a subsequent pass.
    status = {
        "status": "agent_dispatched",
        "cell_id": cell_id,
        "species": species,
        "epoch": current_epoch,
        "bbox": output["bbox"],
        "opacity": output["opacity"],
        "species_params": output["species_params"],
    }
    _write_json(f"cell/{cell_id}/status.json", status)

    print(
        f"[cell_agent] published status.json: cell_id={cell_id} "
        f"bbox=({output['bbox']['x']},{output['bbox']['y']},"
        f"{output['bbox']['w']},{output['bbox']['h']}) "
        f"z={output['bbox']['z']} opacity={output['opacity']:.3f}",
        file=sys.stderr,
    )
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent dispatch — run_all_cells
# ─────────────────────────────────────────────────────────────────────────────

_MAX_WORKERS = 2  # cookie supports max 2 concurrent sub-Claude sessions (4+ triggers 429)
_SINGLE_CELL_TIMEOUT = 90  # seconds per cell dispatch
_MIN_TOTAL_TIMEOUT = 300   # never less than 5 minutes total
_DISPATCH_SPACING = 3      # seconds between starting each dispatch (rate limit courtesy)


def _compute_total_timeout(cell_count: int) -> float:
    """max(single_cell_timeout × ceil(cell_count / max_workers), 300s)"""
    batches = math.ceil(cell_count / _MAX_WORKERS)
    return max(_SINGLE_CELL_TIMEOUT * batches, _MIN_TOTAL_TIMEOUT)


def _dispatch_one(cell_id: str) -> tuple[str, bool, float]:
    """
    Dispatch a single cell, return (cell_id, success, elapsed_ms).
    Each cell is independent — failure here never propagates.
    """
    t0 = time.monotonic()
    try:
        dispatch_cell_agent(cell_id)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return (cell_id, True, elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"[cell_agent] ERROR cell_id={cell_id}: {exc}", file=sys.stderr)
        return (cell_id, False, elapsed_ms)


def run_all_cells(
    on_cell_complete: Optional[Callable[[str, bool, float], None]] = None,
) -> dict[str, str]:
    """
    Dispatch all cells via concurrent ThreadPoolExecutor(max_workers=10).

    Args:
        on_cell_complete:  Optional callback(cell_id, success, elapsed_ms)
                          invoked as each cell finishes.

    Returns:
        Dict mapping cell_id → "success" | "fail" | "timeout".
    """
    ids = _all_cell_ids()
    if not ids:
        print("[cell_agent] run_all_cells: no cells found", file=sys.stderr)
        return {}

    print(
        f"[cell_agent] run_all_cells: {len(ids)} cells, "
        f"mode=live (concurrent, max_workers={_MAX_WORKERS})",
        file=sys.stderr,
    )

    results: dict[str, str] = {}

    # ── Concurrent dispatch ──────────────────────────────────────────────
    total_timeout = _compute_total_timeout(len(ids))
    print(
        f"[cell_agent] total_timeout={total_timeout:.0f}s for {len(ids)} cells",
        file=sys.stderr,
    )

    # Pre-mark all cells as timeout; completed ones override below
    for cid in ids:
        results[cid] = "timeout"

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_cid = {}
        for i, cid in enumerate(ids):
            future_to_cid[executor.submit(_dispatch_one, cid)] = cid
            # Rate limit: space out dispatches to avoid cookie session conflicts
            if i < len(ids) - 1:
                time.sleep(_DISPATCH_SPACING)
        try:
            for future in as_completed(future_to_cid, timeout=total_timeout):
                cell_id, success, elapsed_ms = future.result()
                results[cell_id] = "success" if success else "fail"
                if on_cell_complete:
                    on_cell_complete(cell_id, success, elapsed_ms)
        except TimeoutError:
            # as_completed raised because total_timeout expired.
            # Any futures still pending stay marked "timeout".
            timed_out = [cid for cid, st in results.items() if st == "timeout"]
            print(
                f"[cell_agent] total timeout ({total_timeout:.0f}s) expired. "
                f"Timed-out cells: {timed_out}",
                file=sys.stderr,
            )

    # Summary
    counts = {"success": 0, "fail": 0, "timeout": 0}
    for st in results.values():
        counts[st] += 1
    print(
        f"[cell_agent] run_all_cells done: {counts['success']} success, "
        f"{counts['fail']} fail, {counts['timeout']} timeout "
        f"(out of {len(ids)})",
        file=sys.stderr,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli_progress(cell_id: str, success: bool, elapsed_ms: float):
    """Default on_cell_complete callback for CLI usage."""
    tag = "OK" if success else "FAIL"
    print(
        f"[cell_agent] [{tag}] {cell_id}  ({elapsed_ms:.0f}ms)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dispatch sub-Claude cell agents for astro-svgfigure."
    )
    parser.add_argument(
        "cell_id",
        nargs="?",
        help="Cell ID to dispatch (e.g. self_attn). Omit with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Dispatch all cells found in skeleton/cell/.",
    )
    args = parser.parse_args()

    if args.all:
        results = run_all_cells(
            on_cell_complete=_cli_progress,
        )
        # Exit code 1 if any cell failed or timed out
        if any(st != "success" for st in results.values()):
            sys.exit(1)
    elif args.cell_id:
        dispatch_cell_agent(args.cell_id)
    else:
        parser.print_help()
        sys.exit(1)
