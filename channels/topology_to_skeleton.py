#!/usr/bin/env python3
"""
topology_to_skeleton.py — Generate skeleton cell JSON from ELK topology.

Usage:
    python3 topology_to_skeleton.py TRANSFORMER
    python3 topology_to_skeleton.py VAE
    python3 topology_to_skeleton.py RESNET

Reads src/lib/elk/examples.ts, parses the named topology, and writes
channels/skeleton/cell/{node_id}.json for each node.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_TS = os.path.join(ROOT, "src", "lib", "elk", "examples.ts")
SKELETON_DIR = os.path.join(ROOT, "channels", "skeleton", "cell")
CELL_DIR = os.path.join(ROOT, "channels", "cell")

# ── Species inference from label ──────────────────────────────────────────────
SPECIES_RULES = [
    (r"attn|attention", "cil-eye"),
    (r"conv|filter|kernel", "cil-filter"),
    (r"norm|bn|batch", "cil-plus"),
    (r"embed|encod|vector|mu|sigma", "cil-vector"),
    (r"output|decode|x_hat", "cil-arrow-right"),
    (r"input", "cil-vector"),
    (r"add|\+|residual", "cil-plus"),
    (r"relu|activ|gelu|swish", "cil-bolt"),
    (r"sample|reparame", "cil-loop"),
    (r"ffn|feed|forward|mlp", "cil-bolt"),
    (r"pool|downsample", "cil-layers"),
    (r"graph|net", "cil-graph"),
]

SPECIES_GENE = {
    "cil-eye": {"primary_shape": "radial", "pattern": "heatmap", "line_style": "ray", "family": "attention"},
    "cil-vector": {"primary_shape": "arrow", "pattern": "gradient", "line_style": "solid", "family": "embedding"},
    "cil-bolt": {"primary_shape": "zigzag", "pattern": "stripe", "line_style": "dash", "family": "activation"},
    "cil-plus": {"primary_shape": "cross", "pattern": "grid", "line_style": "solid", "family": "normalization"},
    "cil-arrow-right": {"primary_shape": "chevron", "pattern": "solid", "line_style": "solid", "family": "terminal"},
    "cil-filter": {"primary_shape": "grid", "pattern": "checker", "line_style": "solid", "family": "convolution"},
    "cil-code": {"primary_shape": "bracket", "pattern": "monospace", "line_style": "solid", "family": "compute"},
    "cil-layers": {"primary_shape": "stack", "pattern": "layered", "line_style": "solid", "family": "pooling"},
    "cil-loop": {"primary_shape": "arc", "pattern": "circular", "line_style": "dash", "family": "sampling"},
    "cil-graph": {"primary_shape": "scatter", "pattern": "connected", "line_style": "dot", "family": "graph"},
}


def infer_species(label: str) -> str:
    low = label.lower()
    for pattern, species in SPECIES_RULES:
        if re.search(pattern, low):
            return species
    return "cil-code"


def _ensure_cell_dir(cell_id: str, label: str, species: str, bbox: dict,
                     parent_id=None, is_group: bool = False):
    """Create channels/cell/{cell_id}/ directory with a params.json.

    The params.json mirrors the format produced by the rendering pipeline
    and includes ``parent_id`` so that hierarchical physics (M350) can
    resolve compound-group membership without re-parsing topology.json.
    """
    cell_dir = os.path.join(CELL_DIR, cell_id)
    os.makedirs(cell_dir, exist_ok=True)
    params = {
        "cell_id": cell_id,
        "parent_id": parent_id,
        "species": species,
        "bbox": bbox,
        "z": bbox.get("z", 3),
        "opacity": 0.05,
        "fill_color": "#607D8B",
        "stroke_color": "#607D8B",
        "label": label,
        "font_size": 10,
        "is_group": is_group,
    }
    params_path = os.path.join(cell_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    return cell_dir


def parse_examples_ts(ts_path: str) -> dict:
    """Parse all EXAMPLE objects from examples.ts using regex."""
    with open(ts_path) as f:
        content = f.read()

    examples = {}
    # Find each export const XXX_EXAMPLE = { ... }
    pattern = r"export\s+const\s+(\w+_EXAMPLE)\s*=\s*(\{)"
    for m in re.finditer(pattern, content):
        name = m.group(1)
        start = m.start(2)
        # Balance braces to find end of object
        depth = 0
        end = start
        for i in range(start, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        obj_str = content[start:end]
        # Convert JS object to JSON-parseable string
        # Add quotes around unquoted keys
        obj_str = re.sub(r"(\w+)\s*:", r'"\1":', obj_str)
        # Remove trailing commas
        obj_str = re.sub(r",\s*([}\]])", r"\1", obj_str)
        # Single quotes to double quotes
        obj_str = obj_str.replace("'", '"')
        # Remove JS comments
        obj_str = re.sub(r"//.*?\n", "\n", obj_str)

        try:
            parsed = json.loads(obj_str)
            # Normalize name: TRANSFORMER_EXAMPLE → TRANSFORMER
            key = name.replace("_EXAMPLE", "")
            examples[key] = parsed
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse {name}: {e}")
            continue

    return examples


def generate_skeleton(topology: dict, name: str) -> list:
    """Generate skeleton cell JSON files from parsed topology.

    Recursively walks compound nodes (children with their own children)
    creating both skeleton JSON and cell directories with params.json
    that include ``parent_id`` for hierarchy-aware physics (M350).
    """

    # ── Collect all edges at every nesting level ──────────────────────────
    incoming = {}  # node_id → [edge_id]
    outgoing = {}  # node_id → [edge_id]

    def _index_edges(node):
        for edge in node.get("edges", []):
            eid = edge["id"]
            for src in edge.get("sources", []):
                outgoing.setdefault(src, []).append(eid)
            for tgt in edge.get("targets", []):
                incoming.setdefault(tgt, []).append(eid)
        for child in node.get("children", []):
            if child.get("children"):
                _index_edges(child)

    _index_edges(topology)

    # ── Recursive walk ────────────────────────────────────────────────────
    generated = []

    def _walk(node, parent_id, y_offset):
        for child in node.get("children", []):
            cid = child["id"]
            w = child.get("width", 140)
            h = child.get("height", 50)
            label = child.get("labels", [{}])[0].get("text", cid)
            species = infer_species(label)
            is_group = child.get("group", False) or bool(child.get("children"))

            bbox = {
                "x": 220,
                "y": y_offset,
                "w": w,
                "h": h,
                "z": 5 if species in ("cil-plus", "cil-vector") or is_group else 3,
            }

            cell = {
                "cell_id": cid,
                "label": label,
                "species": species,
                "gene_traits": SPECIES_GENE.get(species, SPECIES_GENE["cil-code"]),
                "initial_bbox": bbox,
                "topology": {
                    "incoming_edges": incoming.get(cid, []),
                    "outgoing_edges": outgoing.get(cid, []),
                },
                "parent_id": parent_id,
                "is_group": is_group,
            }

            os.makedirs(SKELETON_DIR, exist_ok=True)
            cell_path = os.path.join(SKELETON_DIR, f"{cid}.json")
            with open(cell_path, "w") as f:
                json.dump(cell, f, indent=2)

            # Create cell directory + params.json with parent_id
            _ensure_cell_dir(cid, label, species, bbox,
                             parent_id=parent_id, is_group=is_group)

            generated.append(cid)
            y_offset += h + 60  # spacing

            # Recurse into compound children
            if child.get("children"):
                _walk(child, cid, y_offset)

        return y_offset

    _walk(topology, None, 40)
    return generated


def main():
    topo_name = sys.argv[1] if len(sys.argv) > 1 else "TRANSFORMER"
    topo_name = topo_name.upper()

    examples = parse_examples_ts(EXAMPLES_TS)
    if topo_name not in examples:
        print(f"Unknown topology: {topo_name}")
        print(f"Available: {list(examples.keys())}")
        sys.exit(1)

    # Clear old skeleton cells
    if os.path.exists(SKELETON_DIR):
        for f in os.listdir(SKELETON_DIR):
            if f.endswith(".json"):
                os.remove(os.path.join(SKELETON_DIR, f))

    cells = generate_skeleton(examples[topo_name], topo_name)
    print(f"[topology_to_skeleton] Generated {len(cells)} cells for {topo_name}: {cells}")


if __name__ == "__main__":
    main()


def from_structured_data(json_path: str) -> tuple:
    """Parse structured_data JSON into flat cell list + edges.

    Recursively walks compound nodes and creates both skeleton JSON
    and cell directories with params.json containing ``parent_id``.
    """
    with open(json_path) as f:
        data = json.load(f)
    cells_gen = []
    edges_col = []
    def flatten(node, px=0, py=0, parent_id=None):
        for child in node.get("children", []):
            cid = child["id"]
            x = child.get("x", 0) + px
            y = child.get("y", 0) + py
            w = child.get("width", 140)
            h = child.get("height", 100)
            label = child.get("labels", [{}])[0].get("text", cid)
            is_group = child.get("group", False)
            if child.get("labelOnly", False): continue
            low = label.lower()
            if "model" in low or "encoder" in low or "decoder" in low: sp = "cil-eye"
            elif "dataset" in low or "data" in low or "input" in low: sp = "cil-vector"
            elif "output" in low or "html" in low or "css" in low or "render" in low: sp = "cil-arrow-right"
            elif "agent" in low or "code" in low or "mllm" in low: sp = "cil-code"
            elif "filter" in low or "prun" in low or "detection" in low: sp = "cil-filter"
            elif "style" in low or "align" in low: sp = "cil-layers"
            elif "loop" in low or "token" in low or "prediction" in low: sp = "cil-loop"
            elif "loss" in low or "training" in low: sp = "cil-plus"
            elif "tree" in low or "dom" in low or "link" in low: sp = "cil-graph"
            elif "noise" in low or "error" in low or "hidden" in low: sp = "cil-bolt"
            else: sp = infer_species(label)
            bbox = {"x": x, "y": y, "w": w, "h": h, "z": 5 if is_group else 3}
            cell = {"cell_id": cid, "label": label[:40], "species": sp,
                    "gene_traits": SPECIES_GENE.get(sp, SPECIES_GENE["cil-code"]),
                    "initial_bbox": bbox,
                    "topology": {"incoming_edges": [], "outgoing_edges": []},
                    "is_group": is_group, "parent_id": parent_id}
            os.makedirs(SKELETON_DIR, exist_ok=True)
            with open(os.path.join(SKELETON_DIR, f"{cid}.json"), "w") as f: json.dump(cell, f, indent=2)
            # Create cell directory + params.json with parent_id
            _ensure_cell_dir(cid, label[:40], sp, bbox,
                             parent_id=parent_id, is_group=is_group)
            cells_gen.append(cid)
            for edge in child.get("edges", []):
                src = edge["sources"][0] if edge.get("sources") else ""
                tgt = edge["targets"][0] if edge.get("targets") else ""
                et = edge.get("advanced", {}).get("semanticType", "data_flow")
                edges_col.append({"id": edge["id"], "source": src, "target": tgt,
                    "type": "skip_connection" if et in ("skip_connection","feedback","gradient_flow") else "normal"})
            if is_group: flatten(child, x, y, parent_id=cid)
    flatten(data)
    for edge in data.get("edges", []):
        src = edge["sources"][0] if edge.get("sources") else ""
        tgt = edge["targets"][0] if edge.get("targets") else ""
        et = edge.get("advanced", {}).get("semanticType", "data_flow")
        edges_col.append({"id": edge["id"], "source": src, "target": tgt,
            "type": "skip_connection" if et in ("skip_connection","feedback","gradient_flow") else "normal"})
    # Backfill topology
    for e in edges_col:
        for role, key in [("source", "outgoing_edges"), ("target", "incoming_edges")]:
            p = os.path.join(SKELETON_DIR, f"{e[role]}.json")
            if os.path.exists(p):
                with open(p) as f: d = json.load(f)
                if e["id"] not in d["topology"][key]: d["topology"][key].append(e["id"])
                with open(p, "w") as f: json.dump(d, f, indent=2)
    print(f"[from_structured_data] {len(cells_gen)} cells, {len(edges_col)} edges")
    return cells_gen, edges_col
