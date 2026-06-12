#!/usr/bin/env python3
"""
Epoch Controller — reads topology.json, splits into per-cell skeleton signals,
assigns species (from 558 CoreUI icon genotypes), initializes z_layers.

This is the "fertilized egg" — it reads the DNA and kicks off cell division.

M002 — Cell Growth Epoch
========================
grow_epoch(epoch_idx) implements the multi-round development cycle:

  Per cell every epoch
  ─────────────────────
  1. Compute "natural bbox":
       natural_w = BASE_W * species_complexity + label_len_factor
       natural_h = BASE_H * species_complexity + label_line_factor
     The natural bbox is what the cell *wants* to occupy given its gene traits
     and label length; it may be larger than the ELK-initial bbox.

  2. If natural bbox > current bbox → grow:
       new_w = current_w + (natural_w - current_w) * GROW_RATE
       new_h = current_h + (natural_h - current_h) * GROW_RATE
     (GROW_RATE = 0.10 → 10% per epoch)

  3. Collision detection against all neighbours on the same z-layer.
     Any overlap emits a push signal into physics/force_field.json:
       { cell_id: { dx, dy, dz, push_from: [..], push_mag: px } }

  4. Pushed neighbours translate their (x, y) in the *next* epoch
     (the force_field written this epoch is consumed at the start of
     the next grow_epoch call — matching the pub/sub edge convention).

  5. Edge re-routing: after bbox / position changes, every edge whose
     source or target moved has its route.json midpoints recomputed as
     the straight-line midpoint between source-bbox-centre and
     target-bbox-centre (minimal re-route without full ELK re-layout).

  6. Convergence: if every cell's bbox changed by < CONVERGENCE_PX in
     both dimensions, grow_epoch() returns True → caller can break.

Ported architecture notes (for commit log consistency with C++ lineage)
───────────────────────────────────────────────────────────────────────
  natural_bbox         ← FAstroCellNaturalBounds::Compute() (new, M002)
  GROW_RATE            ← analogous to FAstroMorphSolver::GrowthDelta
  push signal          ← FAstroCellPushConstraint written to force_field
                          channel (push_from / push_mag fields are new)
  edge re-route        ← FAstroEdgeRouteSolver::RefreshMidpoints() (M002)
  convergence < 2px    ← FAstroConvergenceJudge::AllSettled(threshold=2)
"""
import json
import math
import os
import hashlib
from typing import Dict, Tuple

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

# ═══════════════════════════════════════════════
# M002 — Natural-bbox growth constants
# ═══════════════════════════════════════════════

# Base size a cell occupies before label / species modulation.
# These approximate the ELK default node size so the natural bbox
# is slightly larger once complexity and label are factored in.
_NATURAL_BASE_W: float = 140.0
_NATURAL_BASE_H: float = 44.0

# Pixels added per character of label text (horizontal growth pressure).
_LABEL_PX_PER_CHAR: float = 6.5

# Additional height per newline-equivalent word group (≥4 words → 2 lines).
_LABEL_LINE_H: float = 14.0

# Species complexity multipliers (width, height).
# Higher complexity → larger natural bbox.
# Mirrors FAstroCellNaturalBounds::SpeciesComplexityScale table.
_SPECIES_COMPLEXITY: Dict[str, Tuple[float, float]] = {
    "cil-eye":         (1.30, 1.25),   # attention: multi-head sub-cells, large
    "cil-bolt":        (1.15, 1.10),   # ffn: wide layers
    "cil-graph":       (1.20, 1.15),   # data viz: axes need room
    "cil-loop":        (1.10, 1.10),   # recurrent: moderate
    "cil-layers":      (1.20, 1.20),   # hierarchy: stacked depth
    "cil-filter":      (1.25, 1.20),   # conv: kernel grid
    "cil-plus":        (1.00, 0.95),   # add/norm: compact
    "cil-arrow-right": (1.00, 0.95),   # io: compact
    "cil-code":        (1.10, 1.05),   # computation: moderate
    "cil-vector":      (1.15, 1.05),   # embedding: slightly wider
}

# Fraction of the natural–current gap closed each epoch (10% / epoch).
_GROW_RATE: float = 0.10

# Convergence threshold: max bbox change (px) across all cells.
_CONVERGENCE_PX: float = 2.0

# Minimum separation (px) kept between cells on the same z-layer.
_CELL_MARGIN: float = 8.0

# Push force damp: fraction of overlap resolved each epoch to avoid oscillation.
_PUSH_DAMP: float = 0.55


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


# ═══════════════════════════════════════════════════════════════════════════════
# M002 — Cell Growth Epoch implementation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> dict:
    """Load a JSON file from an absolute path; return {} on missing/corrupt."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str, data: dict) -> None:
    """Write JSON atomically (write to tmp, rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _natural_bbox(label: str, species: str, current_w: float, current_h: float) -> Tuple[float, float]:
    """
    FAstroCellNaturalBounds::Compute() — M002 port.

    Returns (natural_w, natural_h) that the cell *wants* to occupy given
    its species complexity and label length.

    Width formula:
        natural_w = BASE_W * cx + label_char_count * PX_PER_CHAR
    Height formula:
        natural_h = BASE_H * cy + (1 if label_words >= 4 else 0) * LINE_H

    The label factors capture that long labels need more horizontal space
    and multi-word labels may wrap to a second line.

    Species complexity scales (cx, cy) are taken from _SPECIES_COMPLEXITY;
    unknown species fall back to (1.0, 1.0).
    """
    cx, cy = _SPECIES_COMPLEXITY.get(species, (1.0, 1.0))
    word_count = len(label.split())
    char_count  = len(label)

    natural_w = _NATURAL_BASE_W * cx + char_count * _LABEL_PX_PER_CHAR
    natural_h = _NATURAL_BASE_H * cy + (_LABEL_LINE_H if word_count >= 4 else 0.0)

    return natural_w, natural_h


def _bbox_delta(before: dict, after: dict) -> float:
    """
    Max absolute change across w and h dimensions.
    Mirrors FAstroConvergenceJudge::BBoxDelta().
    """
    dw = abs(after.get("w", 0) - before.get("w", 0))
    dh = abs(after.get("h", 0) - before.get("h", 0))
    return max(dw, dh)


def _reroute_edge(edge_id: str, source_bbox: dict, target_bbox: dict,
                  existing_route: dict) -> dict:
    """
    FAstroEdgeRouteSolver::RefreshMidpoints() — M002 port.

    Minimal re-route: recompute the single midpoint between source bbox
    centre and target bbox centre.  Preserves edge metadata (z, advanced,
    semanticType) so skip-connection styling is not lost.

    The midpoint is offset slightly toward the right side of the canvas
    for skip connections (curvature hint) to keep the SVG legible.
    """
    sx = source_bbox["x"] + source_bbox["w"] / 2.0
    sy = source_bbox["y"] + source_bbox["h"] / 2.0
    tx = target_bbox["x"] + target_bbox["w"] / 2.0
    ty = target_bbox["y"] + target_bbox["h"] / 2.0

    mid_x = (sx + tx) / 2.0
    mid_y = (sy + ty) / 2.0

    # Skip-connection curvature: push midpoint right by 30% of horizontal span
    adv = existing_route.get("advanced", {})
    if adv.get("semanticType") == "skip_connection":
        curvature = float(adv.get("curvature", 0.6))
        h_span    = abs(tx - sx)
        mid_x    += h_span * curvature * 0.5

    updated = dict(existing_route)
    updated["points"] = [
        {"x": sx, "y": sy},           # source port (bottom centre)
        {"x": mid_x, "y": mid_y},     # computed midpoint
        {"x": tx, "y": ty},           # target port (top centre)
    ]
    updated["rerouted_epoch"] = existing_route.get("rerouted_epoch", -1) + 1
    return updated


def grow_epoch(epoch_idx: int) -> bool:
    """
    FAstroCellGrowthEpoch::Run() — M002.

    Execute one round of the cell development loop:

      Step 1 — Apply push signals from previous epoch (consume force_field).
      Step 2 — Compute natural bbox per cell; grow toward it at GROW_RATE.
      Step 3 — Detect same-z-layer collisions after growth.
      Step 4 — Emit push signals for colliding pairs into force_field.json.
      Step 5 — Re-route edges whose endpoints moved.
      Step 6 — Check convergence: return True iff all bbox deltas < CONVERGENCE_PX.

    Args:
        epoch_idx: current epoch counter (used for logging and bbox.epoch tags).

    Returns:
        True  → all cells converged (caller may stop the loop).
        False → still evolving.
    """
    cell_dir    = os.path.join(CHANNELS, "cell")
    skel_dir    = os.path.join(CHANNELS, "skeleton", "cell")
    edge_dir    = os.path.join(CHANNELS, "edge")
    ff_path     = os.path.join(CHANNELS, "physics", "force_field.json")

    # ── Load current force_field (push signals from previous epoch) ──────────
    force_field: dict = _load_json(ff_path)

    # ── Collect all cell ids from published bbox files ────────────────────────
    if not os.path.isdir(cell_dir):
        print(f"[GrowEpoch] cell dir missing: {cell_dir}")
        return True

    cell_ids = [
        d for d in os.listdir(cell_dir)
        if os.path.isdir(os.path.join(cell_dir, d))
    ]
    if not cell_ids:
        return True

    # ── Step 1 — Apply push signals: translate bbox x/y from force_field ─────
    # Mirrors FAstroCellPushConstraint::ApplyToCell(): each cell reads its
    # accumulated (dx, dy) and shifts position before growth is evaluated.
    pre_growth_bboxes: dict = {}   # snapshot before this epoch's changes
    bboxes: dict = {}              # live bbox dicts (will be mutated + written)

    for cell_id in cell_ids:
        bbox_path = os.path.join(cell_dir, cell_id, "bbox.json")
        bbox = _load_json(bbox_path)
        if not bbox:
            continue

        ff = force_field.get(cell_id, {})
        dx = float(ff.get("dx", 0.0))
        dy = float(ff.get("dy", 0.0))

        if dx != 0.0 or dy != 0.0:
            bbox["x"] = float(bbox["x"]) + dx
            bbox["y"] = float(bbox["y"]) + dy
            print(
                f"[GrowEpoch e{epoch_idx}] {cell_id:20s} "
                f"push applied dx={dx:+.1f} dy={dy:+.1f} "
                f"→ pos=({bbox['x']:.1f},{bbox['y']:.1f})"
            )

        pre_growth_bboxes[cell_id] = dict(bbox)
        bboxes[cell_id] = bbox

    # ── Step 2 — Natural-bbox growth ──────────────────────────────────────────
    # For each cell: compute natural_bbox, compare to current, grow if smaller.
    # Mirrors FAstroMorphSolver::AdvanceGrowth(): if NaturalBounds > CurrentBounds
    # advance by GrowthDelta fraction.
    for cell_id in list(bboxes.keys()):
        bbox = bboxes[cell_id]
        skel_path = os.path.join(skel_dir, f"{cell_id}.json")
        skel = _load_json(skel_path)

        label   = skel.get("label", cell_id)
        species = bbox.get("species") or skel.get("species", "cil-code")

        cur_w = float(bbox.get("w", _NATURAL_BASE_W))
        cur_h = float(bbox.get("h", _NATURAL_BASE_H))

        nat_w, nat_h = _natural_bbox(label, species, cur_w, cur_h)

        grew = False
        new_w, new_h = cur_w, cur_h

        if nat_w > cur_w:
            new_w = cur_w + (nat_w - cur_w) * _GROW_RATE
            grew  = True
        if nat_h > cur_h:
            new_h = cur_h + (nat_h - cur_h) * _GROW_RATE
            grew  = True

        if grew:
            print(
                f"[GrowEpoch e{epoch_idx}] {cell_id:20s} "
                f"grow w {cur_w:.1f}→{new_w:.1f} h {cur_h:.1f}→{new_h:.1f} "
                f"(natural={nat_w:.1f}×{nat_h:.1f})"
            )
            bbox["w"] = new_w
            bbox["h"] = new_h

    # ── Step 3 — Same-z collision detection (O(N²), N ≤ ~64 cells) ───────────
    # Mirrors FAstroCellNaturalBounds collision pass: after growth, detect
    # all pairwise overlaps within the same z-layer and record them.
    # Organised as {z_layer: [(cell_id, x1, y1, x2, y2), ...]} for sweep-style
    # access even though we use a simple nested loop here (N is small).
    by_z: dict = {}
    for cell_id, bbox in bboxes.items():
        z = int(bbox.get("z", 3))
        x1 = float(bbox["x"])
        y1 = float(bbox["y"])
        x2 = x1 + float(bbox["w"])
        y2 = y1 + float(bbox["h"])
        by_z.setdefault(z, []).append((cell_id, x1, y1, x2, y2))

    # Collect (cell_a, cell_b, overlap_x, overlap_y, push_vec_a, push_vec_b) tuples
    collisions = []
    for z_layer, rects in by_z.items():
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                aid, ax1, ay1, ax2, ay2 = rects[i]
                bid, bx1, by1, bx2, by2 = rects[j]

                # AABB overlap check (with margin)
                ov_x = min(ax2, bx2) - max(ax1, bx1) - _CELL_MARGIN
                ov_y = min(ay2, by2) - max(ay1, by1) - _CELL_MARGIN
                if ov_x <= 0 or ov_y <= 0:
                    continue

                # Resolve along the axis of least penetration (SAT minimum axis)
                # FAstroCellPushConstraint::ComputePushAxis() equivalent.
                a_cx = (ax1 + ax2) / 2.0
                a_cy = (ay1 + ay2) / 2.0
                b_cx = (bx1 + bx2) / 2.0
                b_cy = (by1 + by2) / 2.0

                sep_x = b_cx - a_cx
                sep_y = b_cy - a_cy
                dist  = math.hypot(sep_x, sep_y) or 1.0

                # Push magnitude: damped overlap along min axis
                push_mag = max(ov_x, ov_y) * _PUSH_DAMP

                # Unit push direction (A→B)
                nx = sep_x / dist
                ny = sep_y / dist

                collisions.append({
                    "a": aid, "b": bid,
                    "ov_x": ov_x, "ov_y": ov_y,
                    "push_mag": push_mag,
                    "nx": nx, "ny": ny,
                    "z": z_layer,
                })

    print(
        f"[GrowEpoch e{epoch_idx}] "
        f"collisions_after_growth={len(collisions)}"
    )

    # ── Step 4 — Emit push signals into force_field.json ─────────────────────
    # Each collision emits a push signal on both cells:
    #   cell_a gets pushed in direction -(nx, ny) (away from b)
    #   cell_b gets pushed in direction +(nx, ny) (away from a)
    # Signals accumulate additively (multiple neighbours can push one cell).
    # Mirrors FAstroCellPushConstraint::AccumulateForce().
    new_force_field: dict = {cid: {"dx": 0.0, "dy": 0.0, "dz": 0.0, "push_from": []}
                              for cid in bboxes}

    for col in collisions:
        aid, bid  = col["a"], col["b"]
        push_mag  = col["push_mag"]
        nx, ny    = col["nx"], col["ny"]

        # A is pushed away from B
        new_force_field[aid]["dx"] -= nx * push_mag
        new_force_field[aid]["dy"] -= ny * push_mag
        new_force_field[aid]["push_from"].append(bid)

        # B is pushed away from A
        new_force_field[bid]["dx"] += nx * push_mag
        new_force_field[bid]["dy"] += ny * push_mag
        new_force_field[bid]["push_from"].append(aid)

        print(
            f"[GrowEpoch e{epoch_idx}] push {aid}↔{bid} "
            f"mag={push_mag:.1f} dir=({nx:+.2f},{ny:+.2f})"
        )

    # Add push_mag summary field for logging
    for cid, ff in new_force_field.items():
        ff["push_mag"] = math.hypot(ff["dx"], ff["dy"])

    _write_json(ff_path, new_force_field)

    # ── Step 5 — Write updated bboxes + re-route affected edges ──────────────
    moved_cells = set()
    for cell_id, bbox in bboxes.items():
        pre  = pre_growth_bboxes.get(cell_id, bbox)
        # A cell "moved" if position or size changed meaningfully
        pos_dx = abs(float(bbox["x"]) - float(pre["x"]))
        pos_dy = abs(float(bbox["y"]) - float(pre["y"]))
        size_d = _bbox_delta(pre, bbox)
        if pos_dx > 0.01 or pos_dy > 0.01 or size_d > 0.01:
            moved_cells.add(cell_id)

        # Stamp epoch and write back
        bbox["epoch"] = epoch_idx
        bbox_path = os.path.join(cell_dir, cell_id, "bbox.json")
        _write_json(bbox_path, bbox)

    # Edge re-route — only edges whose source or target moved
    if os.path.isdir(edge_dir) and moved_cells:
        for edge_id in os.listdir(edge_dir):
            route_path = os.path.join(edge_dir, edge_id, "route.json")
            if not os.path.isfile(route_path):
                continue
            route = _load_json(route_path)
            if not route:
                continue

            sources = route.get("sources", [])
            targets = route.get("targets", [])

            # Re-route only when at least one endpoint bbox changed
            touched = set(sources + targets) & moved_cells
            if not touched:
                continue

            # Pick first available source and target bbox
            src_bbox = next(
                (bboxes[s] for s in sources if s in bboxes), None
            )
            tgt_bbox = next(
                (bboxes[t] for t in targets if t in bboxes), None
            )
            if src_bbox is None or tgt_bbox is None:
                continue

            updated_route = _reroute_edge(edge_id, src_bbox, tgt_bbox, route)
            _write_json(route_path, updated_route)
            print(
                f"[GrowEpoch e{epoch_idx}] rerouted edge {edge_id} "
                f"(touched: {', '.join(sorted(touched))})"
            )

    # ── Step 6 — Convergence check ────────────────────────────────────────────
    # FAstroConvergenceJudge::AllSettled(threshold=CONVERGENCE_PX):
    # True iff every cell's bbox changed by < CONVERGENCE_PX in both w and h.
    max_delta = 0.0
    delta_report = []
    for cell_id, bbox in bboxes.items():
        pre   = pre_growth_bboxes.get(cell_id, bbox)
        delta = _bbox_delta(pre, bbox)
        max_delta = max(max_delta, delta)
        delta_report.append((cell_id, delta))

    converged = (max_delta < _CONVERGENCE_PX)

    print(
        f"[GrowEpoch e{epoch_idx}] "
        f"max_bbox_delta={max_delta:.2f}px "
        f"threshold={_CONVERGENCE_PX}px "
        f"{'CONVERGED ✓' if converged else 'still evolving…'}"
    )
    if not converged:
        # Log top movers for diagnostics
        top = sorted(delta_report, key=lambda t: t[1], reverse=True)[:3]
        for cid, d in top:
            if d > 0.01:
                print(f"  [GrowEpoch]  top-mover {cid}: delta={d:.2f}px")

    return converged


if __name__ == "__main__":
    main()
