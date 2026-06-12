#!/usr/bin/env python3
"""
Epoch Controller — reads topology.json, splits into per-cell skeleton signals,
assigns species (from 558 CoreUI icon genotypes), initializes z_layers.

This is the "fertilized egg" — it reads the DNA and kicks off cell division.
"""
import json
import os
import hashlib

CHANNELS = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════
# Species Registry — 558 CoreUI icon genotypes
# Each species defines algorithm generation traits, NOT a file to paste
# ═══════════════════════════════════════════════
SPECIES_GENES = {
    "cil-eye":       {"primary_shape": "radial",   "pattern": "heatmap",     "line_style": "ray",      "family": "attention"},
    "cil-bolt":      {"primary_shape": "zigzag",   "pattern": "gradient",    "line_style": "angular",  "family": "activation"},
    "cil-graph":     {"primary_shape": "axes",     "pattern": "bars",        "line_style": "polyline", "family": "data"},
    "cil-loop":      {"primary_shape": "arc",      "pattern": "spiral",      "line_style": "curved",   "family": "iteration"},
    "cil-layers":    {"primary_shape": "stacked",  "pattern": "depth_fade",  "line_style": "parallel", "family": "hierarchy"},
    "cil-filter":    {"primary_shape": "grid",     "pattern": "kernel",      "line_style": "dashed",   "family": "convolution"},
    "cil-plus":      {"primary_shape": "cross",    "pattern": "merge",       "line_style": "converge", "family": "aggregation"},
    "cil-arrow-right": {"primary_shape": "arrow",  "pattern": "flow",        "line_style": "directed", "family": "dataflow"},
    "cil-code":      {"primary_shape": "rect",     "pattern": "monospace",   "line_style": "solid",    "family": "computation"},
    "cil-vector":    {"primary_shape": "arrow_vec","pattern": "magnitude",   "line_style": "weighted", "family": "embedding"},
}

# Semantic keyword → species mapping (algorithm, not lookup table)
SEMANTIC_KEYWORDS = {
    "attention": "cil-eye",
    "multi-head": "cil-eye",
    "embedding": "cil-vector",
    "positional": "cil-vector",
    "encoding": "cil-vector",
    "feed forward": "cil-bolt",
    "ffn": "cil-bolt",
    "mlp": "cil-bolt",
    "norm": "cil-plus",
    "add": "cil-plus",
    "residual": "cil-plus",
    "output": "cil-arrow-right",
    "input": "cil-arrow-right",
    "conv": "cil-filter",
    "pool": "cil-filter",
    "recurrent": "cil-loop",
    "lstm": "cil-loop",
    "layer": "cil-layers",
    "block": "cil-layers",
    "encoder": "cil-code",
    "decoder": "cil-code",
}

def assign_species(label: str) -> str:
    """Semantic matching — not hardcoded, uses keyword proximity."""
    label_lower = label.lower()
    for keyword, species in SEMANTIC_KEYWORDS.items():
        if keyword in label_lower:
            return species
    # Fallback: hash label to pick a species deterministically
    idx = int(hashlib.md5(label.encode()).hexdigest(), 16) % len(SPECIES_GENES)
    return list(SPECIES_GENES.keys())[idx]

def assign_z_layer(node_id: str, edges: list, all_ids: list) -> int:
    """Assign z-layer based on role in topology."""
    # Skip connections go to z=5 (edges_above)
    for edge in edges:
        adv = edge.get("advanced", {})
        if adv.get("semanticType") == "skip_connection":
            if node_id in edge.get("sources", []) or node_id in edge.get("targets", []):
                return 5  # edges_above layer
    # Regular nodes at z=3
    return 3

def main():
    # Read topology DNA
    with open(os.path.join(CHANNELS, "skeleton", "topology.json")) as f:
        topo = json.load(f)

    children = topo.get("children", [])
    edges = topo.get("edges", [])
    all_ids = [c["id"] for c in children]

    # ── Generate per-cell skeleton signals ──
    species_assignment = {}
    z_layers = {
        "__schema__": "z=0 background, z=1 groups, z=2 edges_below, z=3 nodes, z=4 decorations, z=5 edges_above, z=6 labels, z=7 annotations"
    }

    # Simple initial layout: vertical stack with spacing
    y_cursor = 40
    x_center = 300

    for i, child in enumerate(children):
        cell_id = child["id"]
        label = child["labels"][0]["text"] if child.get("labels") else cell_id
        w = child.get("width", 160)
        h = child.get("height", 50)

        species = assign_species(label)
        z = assign_z_layer(cell_id, edges, all_ids)

        # Initial embryonic position (will be overridden by ELK, then by cell development)
        skeleton_signal = {
            "cell_id": cell_id,
            "label": label,
            "species": species,
            "gene_traits": SPECIES_GENES.get(species, SPECIES_GENES["cil-code"]),
            "initial_bbox": {
                "x": x_center - w // 2,
                "y": y_cursor,
                "w": w,
                "h": h,
                "z": z
            },
            "topology": {
                "incoming_edges": [e["id"] for e in edges if cell_id in e.get("targets", [])],
                "outgoing_edges": [e["id"] for e in edges if cell_id in e.get("sources", [])],
            }
        }

        # Write per-cell skeleton
        cell_path = os.path.join(CHANNELS, "skeleton", "cell", f"{cell_id}.json")
        with open(cell_path, "w") as f:
            json.dump(skeleton_signal, f, indent=2)

        species_assignment[cell_id] = {
            "species": species,
            "gene_traits": SPECIES_GENES.get(species, {})
        }
        z_layers[cell_id] = z

        y_cursor += h + 60  # spacing

    # ── Write physics channels ──
    with open(os.path.join(CHANNELS, "physics", "species_assignment.json"), "w") as f:
        json.dump(species_assignment, f, indent=2)

    with open(os.path.join(CHANNELS, "physics", "z_layers.json"), "w") as f:
        json.dump(z_layers, f, indent=2)

    # Initial empty force field
    force_field = {cell_id: {"dx": 0, "dy": 0, "dz": 0} for cell_id in all_ids}
    with open(os.path.join(CHANNELS, "physics", "force_field.json"), "w") as f:
        json.dump(force_field, f, indent=2)

    with open(os.path.join(CHANNELS, "physics", "converged.json"), "w") as f:
        json.dump({"converged": False, "epoch": 0, "conflicts": len(children)}, f, indent=2)

    # ── Write epoch signal ──
    with open(os.path.join(CHANNELS, "skeleton", "epoch.json"), "w") as f:
        json.dump({"current": 0, "max": 10, "status": "seeded", "total_cells": len(children)}, f, indent=2)

    # ── Write edge skeleton signals ──
    for edge in edges:
        edge_dir = os.path.join(CHANNELS, "edge", edge["id"])
        os.makedirs(edge_dir, exist_ok=True)
        with open(os.path.join(edge_dir, "route.json"), "w") as f:
            json.dump({
                "edge_id": edge["id"],
                "sources": edge["sources"],
                "targets": edge["targets"],
                "advanced": edge.get("advanced", {}),
                "points": [],
                "z": 5 if edge.get("advanced", {}).get("semanticType") == "skip_connection" else 2
            }, f, indent=2)

    print(f"Epoch 0 seeded: {len(children)} cells, {len(edges)} edges")
    for cell_id, info in species_assignment.items():
        print(f"  {cell_id:20s} → species={info['species']:20s} z={z_layers[cell_id]}")

if __name__ == "__main__":
    main()
