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
import os
import sys
import urllib.request
import urllib.error
from typing import Optional

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
# Anthropic API call — mirrors claude_hk_chat.sh dispatch pattern
# ─────────────────────────────────────────────────────────────────────────────

def _call_claude(system_prompt: str, user_message: str,
                 model: str = "claude-sonnet-4-6",
                 max_tokens: int = 1024,
                 enable_web_search: bool = True) -> str:
    """
    Call Anthropic /v1/messages with optional web_search tool.

    When enable_web_search=True, the sub-Claude can search the web to find
    academic characteristics of its domain (e.g., a "cil-eye" cell representing
    an attention mechanism can search for "multi-head attention visualization
    parameters" to decide its visual params based on real research).

    The tool loop follows the Anthropic tool-use protocol:
      1. Send initial request with web_search tool definition
      2. If response has tool_use blocks, execute the search
      3. Feed tool_result back and get final text response

    Mirrors the curl pattern in walpurgis-WTFGG/claude_hk_chat.sh but
    extended with tool calling for dynamic knowledge acquisition.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — cannot dispatch sub-Claude.\n"
            "Export the variable before running cell_agent.py."
        )

    tools = []
    if enable_web_search:
        tools.append({
            "type": "web_search_20250305",
            "name": "web_search",
        })

    messages = [{"role": "user", "content": user_message}]

    # Tool loop: keep calling until we get a final text response (max 3 rounds)
    for _round in range(4):
        payload_dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            payload_dict["tools"] = tools

        payload = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API HTTP {e.code}: {err_body}") from e

        stop_reason = body.get("stop_reason", "end_turn")
        content_blocks = body.get("content", [])

        # If the model used web_search, the results are automatically included
        # in the response. We just need to check if there's a final text block.
        # With web_search_20250305, Anthropic handles the search internally.
        # The response will contain text blocks with search results integrated.

        # Extract text from all content blocks
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block["text"])

        if text_parts:
            combined = "\n".join(text_parts).strip()
            if combined:
                return combined

        # If no text yet but stop_reason indicates more processing needed
        if stop_reason == "end_turn" or stop_reason == "stop":
            break

        # For tool_use stop_reason, the web_search tool handles itself
        # server-side with the 20250305 version, so we shouldn't get here.
        # But if we do, break to avoid infinite loop.
        print(f"[cell_agent] tool loop round {_round}: "
              f"stop_reason={stop_reason} blocks={len(content_blocks)}",
              file=sys.stderr)
        break

    raise RuntimeError(f"No text block in Anthropic response: {body}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(skeleton: dict, force_field: dict) -> str:
    """
    Construct the user-turn message for the sub-Claude cell.
    Includes skeleton identity + force-field displacement so the cell can
    reason about its final position/size.
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
        "instruction": (
            "You are this cell. Given the force-field displacement, decide your "
            "final bbox position and visual parameters. Return ONLY the JSON "
            "schema specified in your system prompt — no other text."
        ),
    }
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

def dispatch_cell_agent(cell_id: str, dry_run: bool = False) -> dict:
    """
    Dispatch a sub-Claude for a single cell.

    1. Read skeleton/cell/{cell_id}.json + physics/force_field.json
    2. Select SPECIES_PROMPTS[species] as system prompt
    3. Build user message with skeleton + force state
    4. Call Anthropic API (or use stub in dry_run mode)
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

    species = skeleton.get("species", "cil-arrow-right")
    base_prompt = SPECIES_PROMPTS.get(species, _DEFAULT_SPECIES_PROMPT)

    # When running live (not dry_run), prepend the research prefix so the
    # sub-Claude searches for academic characteristics of its domain.
    # The {label} placeholder is filled with the cell's actual label.
    label = skeleton.get("label", cell_id)
    if not dry_run:
        prefix = _RESEARCH_PREFIX.replace("{label}", label)
        system_prompt = prefix + base_prompt
    else:
        system_prompt = base_prompt

    user_message = _build_user_message(skeleton, force_field)

    print(f"[cell_agent] dispatching cell_id={cell_id} species={species}",
          file=sys.stderr)

    if dry_run:
        # ── Dry-run stub: compute deterministic per-species params without API ─
        # Uses a stable hash of cell_id for per-cell variation within valid ranges.
        import hashlib
        _seed = int(hashlib.md5(cell_id.encode()).hexdigest()[:8], 16)
        def _jitter(lo: float, hi: float, bits: int = 0) -> float:
            return round(lo + ((_seed >> bits) & 0xFF) / 255.0 * (hi - lo), 3)

        ib = skeleton["initial_bbox"]
        force = force_field.get(cell_id, {})
        _bbox = {
            "x": round(ib["x"] + force.get("dx", 0.0), 2),
            "y": round(ib["y"] + force.get("dy", 0.0), 2),
            "w": ib["w"],
            "h": ib["h"],
            "z": ib.get("z", 3),
        }
        _opacity = _jitter(0.72, 0.96, 0)

        if species == "cil-eye":
            _sp: dict = {
                "num_rays":        4 + (_seed & 0xF) % 13,
                "focal_intensity": _jitter(0.3, 1.0, 4),
                "halo_radius":     _jitter(0.1, 0.5, 12),
                "ray_opacity_min": _jitter(0.1, 0.4, 16),
                "ray_opacity_max": _jitter(0.4, 0.9, 20),
            }
        elif species == "cil-bolt":
            _sp = {
                "zigzag_segments": 4 + (_seed & 0x7) % 7,
                "arc_seed":        _seed & 0xFFFF,
                "stroke_weight":   _jitter(1.5, 3.0, 8),
                "activation_type": ["relu", "gelu", "swish"][_seed % 3],
            }
        elif species == "cil-vector":
            _sp = {
                "num_arrows":    3 + (_seed & 0x7) % 6,
                "angle_spread":  _jitter(0.2, 1.2, 4),
                "arrow_weight":  _jitter(1.0, 2.5, 8),
                "arrow_opacity": _jitter(0.4, 0.85, 12),
            }
        elif species == "cil-plus":
            _sp = {
                "arm_ratio":      _jitter(0.2, 0.4, 4),
                "stroke_weight":  _jitter(2.0, 4.0, 8),
                "diag_dasharray": ["3,2", "4,3", "5,2"][_seed % 3],
                "norm_strength":  _jitter(0.3, 1.0, 12),
            }
        elif species == "cil-arrow-right":
            _sp = {
                "arrow_width_ratio": _jitter(0.2, 0.45, 4),
                "head_height":       _jitter(0.2, 0.4, 8),
                "fill_opacity":      _jitter(0.4, 0.7, 12),
                "direction":         ["right", "up", "down"][_seed % 3],
            }
        elif species == "cil-filter":
            _gs = 2 + (_seed & 0x3) % 4
            _sp = {
                "grid_size":       _gs,
                "highlight_cx":    _seed % _gs,
                "highlight_cy":    (_seed >> 2) % _gs,
                "highlight_alpha": _jitter(0.3, 0.7, 4),
                "grid_opacity":    _jitter(0.4, 0.8, 8),
            }
        elif species == "cil-code":
            _sp = {
                "brace_arm_ratio": _jitter(0.2, 0.4, 4),
                "nib_ratio":       _jitter(0.15, 0.3, 8),
                "corner_radius":   _jitter(0.25, 0.45, 12),
                "stroke_weight":   _jitter(1.5, 3.0, 16),
            }
        elif species == "cil-layers":
            _sp = {
                "num_layers":    2 + (_seed & 0x3) % 4,
                "stagger_step":  _jitter(0.25, 0.45, 4),
                "layer_opacity": [_jitter(0.4, 0.8, i * 4) for i in range(3)],
                "layer_colours": ["#90CAF9", "#42A5F5", "#1E88E5"],
            }
        elif species == "cil-loop":
            _sp = {
                "arc_gap_degrees": _jitter(40, 90, 4),
                "arc_radius":      _jitter(0.2, 0.35, 8),
                "stroke_weight":   _jitter(1.8, 3.0, 12),
                "arc_opacity":     _jitter(0.6, 0.9, 16),
                "dot_radius":      _jitter(0.1, 0.2, 20),
            }
        elif species == "cil-graph":
            _sp = {
                "num_nodes":    3 + (_seed & 0x7) % 5,
                "outer_radius": _jitter(0.25, 0.35, 4),
                "node_radius":  _jitter(0.04, 0.07, 8),
                "edge_opacity": _jitter(0.4, 0.7, 12),
                "node_opacity": _jitter(0.6, 0.9, 16),
            }
        else:
            _sp = {
                "corner_radius": _jitter(6, 16, 4),
                "stroke_weight": _jitter(1.0, 2.5, 8),
                "fill_opacity":  _jitter(0.5, 0.9, 12),
            }

        raw_output = {"bbox": _bbox, "opacity": _opacity, "species_params": _sp}
        print(f"[cell_agent] dry_run=True species={species} deterministic params computed",
              file=sys.stderr)
    else:
        # ── Live dispatch: sub-Claude call ────────────────────────────────────
        raw_text = _call_claude(system_prompt, user_message)
        print(f"[cell_agent] raw response ({len(raw_text)} chars):\n{raw_text[:320]}",
              file=sys.stderr)
        raw_output = _parse_json_response(raw_text)

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
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API call; use deterministic stub output (no ANTHROPIC_API_KEY needed).",
    )
    args = parser.parse_args()

    if args.all:
        ids = _all_cell_ids()
        print(f"[cell_agent] dispatching {len(ids)} cells: {ids}", file=sys.stderr)
        for cid in ids:
            try:
                dispatch_cell_agent(cid, dry_run=args.dry_run)
            except Exception as exc:
                print(f"[cell_agent] ERROR cell_id={cid}: {exc}", file=sys.stderr)
    elif args.cell_id:
        dispatch_cell_agent(args.cell_id, dry_run=args.dry_run)
    else:
        parser.print_help()
        sys.exit(1)
