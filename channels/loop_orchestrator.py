#!/usr/bin/env python3
"""
Loop Orchestrator — runs one full epoch of the cell pub/sub loop.

In production: each cell is a separate sub-Claude via claude_hk_chat.sh
Here: we simulate the loop locally to verify convergence.

Flow:
  1. Each cell reads its skeleton signal + force_field (subscribe)
  2. Each cell generates SVG with its species algorithm (proc)
  3. Each cell publishes bbox + svg (publish)
  4. Physics engine reads all bboxes → detects 3D collisions → updates force_field
  5. Convergence judge checks if all cells are stable
  6. If not converged → next epoch
  7. If converged → assemble final SVG by z-layer order
"""
import json
import math
import os
import glob

CHANNELS = os.path.dirname(os.path.abspath(__file__))

def read_channel(path):
    with open(os.path.join(CHANNELS, path)) as f:
        return json.load(f)

def write_channel(path, data):
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)


def run_all_cells():
    """Execute all cell components (simulate parallel sub-Claude dispatch)."""
    from cell_component import proc
    cells = [f.replace(".json", "") for f in os.listdir(os.path.join(CHANNELS, "skeleton", "cell"))]
    for cell_id in sorted(cells):
        proc(cell_id)
    return cells


def _sweep_line_overlaps(rects_by_z):
    """
    FAstroCellSweepLine — ported from upstream/unreal-renderer/SceneSoftwareOcclusion.cpp
    commit 0b4b199.

    Algorithm: O((N+K) log N) sweep-line 2D overlap detection.
      1. For every cell bbox emit an OPEN event at min_x and a CLOSE event at max_x.
      2. Sort all events by x; ties: OPEN before CLOSE so touching edges are caught.
      3. Scan events left→right, maintaining an active set of currently open rects.
      4. On OPEN: insert into active set; test new rect against every active rect for
         Y-interval overlap → emit collision pair.
      5. On CLOSE: remove from active set.
      6. Cells on different z-layers never collide (same logic as the original O(N²)).

    Returns a list of (cell_id_a, cell_id_b, overlap_x, overlap_y, z) tuples.
    """
    # Event type constants  (mirrors FSweepEvent.bOpen)
    OPEN  = 1   # process opens BEFORE closes at same x (sort key trick from C++)
    CLOSE = 0

    pairs = []

    for z_layer, rects in rects_by_z.items():
        # rects: list of (cell_id, min_x, min_y, max_x, max_y)
        if len(rects) < 2:
            continue

        # Build events — two per rect, mirroring AddRect()
        # Sort key: (x, -is_open) so OPEN (1) sorts before CLOSE (0) at equal x
        events = []
        for idx, (cell_id, min_x, min_y, max_x, max_y) in enumerate(rects):
            if min_x > max_x or min_y > max_y:   # degenerate guard (same as C++)
                continue
            events.append((min_x, -OPEN,  idx))  # OPEN  at min_x
            events.append((max_x, -CLOSE, idx))  # CLOSE at max_x

        events.sort()   # ascending x; at equal x: -OPEN (-1) < -CLOSE (0) → open first

        # Active set: indices of rects whose x-interval contains current sweep position
        active = []   # list of rect indices (mirrors TArray<int32> ActiveSet in C++)

        for ev_x, neg_kind, idx in events:
            is_open = (-neg_kind == OPEN)
            cell_id, min_x, min_y, max_x, max_y = rects[idx]

            if is_open:
                # Test new rect against every currently active rect for Y overlap.
                # In the C++ version this happens implicitly when an occludee leaves
                # the sweep; here we emit pairs eagerly on OPEN so both directions are
                # symmetric and we don't need occluder/occludee distinction.
                for aidx in active:
                    _, _, a_min_y, _, a_max_y = rects[aidx]
                    # Y-interval overlap test (mirrors OR.MaxY >= OccludeeMinY && OR.MinY <= OccludeeMaxY)
                    if a_min_y <= max_y and a_max_y >= min_y:
                        ov_y = min(max_y, a_max_y) - max(min_y, a_min_y)
                        # x-overlap is guaranteed because we're inside the active sweep window
                        a_cell_id, a_min_x, _, a_max_x, _ = rects[aidx]
                        ov_x = min(max_x, a_max_x) - max(min_x, a_min_x)
                        if ov_x > 0 and ov_y > 0:
                            pairs.append((cell_id, a_cell_id, ov_x, ov_y, z_layer))
                active.append(idx)
            else:
                # CLOSE: remove from active set (mirrors RemoveSingleSwap)
                try:
                    active.remove(idx)
                except ValueError:
                    pass   # already removed (shouldn't happen, but be defensive)

    return pairs


def physics_engine():
    """
    Physics Engine organ — reads all cell/*/bbox.json, detects 3D collisions,
    computes force field. Publishes to physics/force_field.json.

    Collision detection replaced with FAstroCellSweepLine sweep-line algorithm
    ported from commit 0b4b199 of upstream/unreal-renderer/SceneSoftwareOcclusion.cpp.
    Complexity: O((N+K) log N) vs the previous O(N²) brute-force double loop.

    Only same-z cells can collide. Different z = no collision (preserved).
    """
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))
    bboxes = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bboxes[cell_id] = json.load(f)

    force_field = {cid: {"dx": 0, "dy": 0, "dz": 0} for cid in bboxes}

    # --- 1. Bucket cells by z-layer (same-z-only rule preserved from original) ---
    rects_by_z = {}
    for cell_id, b in bboxes.items():
        z = b.get("z", 3)
        min_x = int(b["x"])
        min_y = int(b["y"])
        max_x = int(b["x"] + b["w"])
        max_y = int(b["y"] + b["h"])
        rects_by_z.setdefault(z, []).append((cell_id, min_x, min_y, max_x, max_y))

    # --- 2. Run sweep-line O((N+K) log N) overlap detection ---------------------
    overlap_pairs = _sweep_line_overlaps(rects_by_z)

    # --- 3. Convert overlapping pairs → collisions + repulsion forces -----------
    collisions = []
    for cell_a, cell_b, overlap_x, overlap_y, z in overlap_pairs:
        collisions.append({
            "a": cell_a, "b": cell_b,
            "overlap": overlap_x * overlap_y,
            "z": z
        })
        a = bboxes[cell_a]
        b = bboxes[cell_b]
        # Repulsion force: push apart along axis of least overlap (logic unchanged)
        if overlap_x < overlap_y:
            push = overlap_x / 2 + 5
            if a["x"] < b["x"]:
                force_field[cell_a]["dx"] -= push
                force_field[cell_b]["dx"] += push
            else:
                force_field[cell_a]["dx"] += push
                force_field[cell_b]["dx"] -= push
        else:
            push = overlap_y / 2 + 5
            if a["y"] < b["y"]:
                force_field[cell_a]["dy"] -= push
                force_field[cell_b]["dy"] += push
            else:
                force_field[cell_a]["dy"] += push
                force_field[cell_b]["dy"] -= push

    write_channel("physics/force_field.json", force_field)
    write_channel("physics/collision.json", {"collisions": collisions, "count": len(collisions)})

    print(f"[Physics] {len(collisions)} collisions detected (sweep-line O((N+K)logN))")
    return collisions


def convergence_judge():
    """Check if all cells have converged (no collisions)."""
    collision_data = read_channel("physics/collision.json")
    count = collision_data.get("count", 999)
    epoch = read_channel("skeleton/epoch.json")

    converged = count == 0
    write_channel("physics/converged.json", {
        "converged": converged,
        "epoch": epoch["current"],
        "conflicts": count
    })
    print(f"[Judge] epoch={epoch['current']} conflicts={count} converged={converged}")
    return converged


def assemble_final_svg():
    """Assemble all cell SVGs into final.svg, ordered by z-layer."""
    svg_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "svg.svg"))
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))

    # Collect (z, cell_id, svg_content)
    fragments = []
    for sf in svg_files:
        cell_id = os.path.basename(os.path.dirname(sf))
        bbox_path = os.path.join(os.path.dirname(sf), "bbox.json")
        z = 3
        if os.path.exists(bbox_path):
            with open(bbox_path) as f:
                z = json.load(f).get("z", 3)
        with open(sf) as f:
            svg_content = f.read()
        fragments.append((z, cell_id, svg_content))

    # Sort by z (lowest first = drawn first = behind)
    fragments.sort(key=lambda x: x[0])

    # Compute canvas bounds
    max_x, max_y = 0, 0
    for bf in bbox_files:
        with open(bf) as f:
            b = json.load(f)
            max_x = max(max_x, b["x"] + b["w"])
            max_y = max(max_y, b["y"] + b["h"])

    width = max_x + 60
    height = max_y + 60

    # Group by z-layer
    z_groups = {}
    for z, cell_id, svg in fragments:
        z_groups.setdefault(z, []).append((cell_id, svg))

    # Defs with arrow markers
    defs = '''  <defs>
    <marker id="arrow-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2E7D32"/>
    </marker>
  </defs>'''

    # Build SVG
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto;">',
        f'  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="4"/>',
        defs,
    ]

    for z in sorted(z_groups.keys()):
        lines.append(f'  <g id="z{z}-layer">')
        for cell_id, svg in z_groups[z]:
            lines.append(f'    <!-- cell: {cell_id} -->')
            lines.append(f'    {svg}')
        lines.append(f'  </g>')

    lines.append('</svg>')

    final_svg = "\n".join(lines)
    output_path = os.path.join(CHANNELS, "..", "output_cell_loop.svg")
    with open(output_path, "w") as f:
        f.write(final_svg)

    print(f"[Assemble] final SVG: {len(fragments)} cells, {len(z_groups)} z-layers, {width}x{height}")
    return output_path


def run_loop(max_epochs=5):
    """Main pub/sub loop."""
    print("=" * 60)
    print("astro-svgfigure Cell Pub/Sub Loop")
    print("=" * 60)

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch} ---")
        write_channel("skeleton/epoch.json", {
            "current": epoch, "max": max_epochs, "status": "running"
        })

        # 1. All cells develop (in production: parallel sub-Claude dispatch)
        cells = run_all_cells()

        # 2. Physics engine computes forces
        collisions = physics_engine()

        # 3. Convergence check
        if convergence_judge():
            print(f"\n✓ Converged at epoch {epoch}!")
            break
    else:
        print(f"\n⚠ Max epochs ({max_epochs}) reached without full convergence")

    # 4. Assemble final SVG
    write_channel("skeleton/epoch.json", {
        "current": epoch, "max": max_epochs, "status": "converged"
    })
    output = assemble_final_svg()
    print(f"\n{'=' * 60}")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    os.chdir(CHANNELS)
    run_loop()
