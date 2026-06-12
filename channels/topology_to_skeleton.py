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
    """Generate skeleton cell JSON files from parsed topology."""
    children = topology.get("children", [])
    edges = topology.get("edges", [])

    # Build edge index
    incoming = {}  # node_id → [edge_id]
    outgoing = {}  # node_id → [edge_id]
    for edge in edges:
        eid = edge["id"]
        for src in edge.get("sources", []):
            outgoing.setdefault(src, []).append(eid)
        for tgt in edge.get("targets", []):
            incoming.setdefault(tgt, []).append(eid)

    generated = []
    y_offset = 40
    for i, child in enumerate(children):
        cid = child["id"]
        w = child.get("width", 140)
        h = child.get("height", 50)
        label = child.get("labels", [{}])[0].get("text", cid)
        species = infer_species(label)

        cell = {
            "cell_id": cid,
            "label": label,
            "species": species,
            "gene_traits": SPECIES_GENE.get(species, SPECIES_GENE["cil-code"]),
            "initial_bbox": {
                "x": 220,
                "y": y_offset,
                "w": w,
                "h": h,
                "z": 5 if species in ("cil-plus", "cil-vector") else 3,
            },
            "topology": {
                "incoming_edges": incoming.get(cid, []),
                "outgoing_edges": outgoing.get(cid, []),
            },
        }

        cell_path = os.path.join(SKELETON_DIR, f"{cid}.json")
        os.makedirs(SKELETON_DIR, exist_ok=True)
        with open(cell_path, "w") as f:
            json.dump(cell, f, indent=2)

        generated.append(cid)
        y_offset += h + 60  # spacing

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
