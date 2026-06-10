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


def physics_engine():
    """
    Physics Engine organ — reads all cell/*/bbox.json, detects 3D collisions,
    computes force field. Publishes to physics/force_field.json.

    Only same-z cells can collide. Different z = no collision.
    """
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))
    bboxes = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bboxes[cell_id] = json.load(f)

    collisions = []
    force_field = {cid: {"dx": 0, "dy": 0, "dz": 0} for cid in bboxes}

    ids = list(bboxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = bboxes[ids[i]]
            b = bboxes[ids[j]]

            # Only same-z collide
            if a.get("z", 3) != b.get("z", 3):
                continue

            # Check 2D overlap
            overlap_x = max(0, min(a["x"]+a["w"], b["x"]+b["w"]) - max(a["x"], b["x"]))
            overlap_y = max(0, min(a["y"]+a["h"], b["y"]+b["h"]) - max(a["y"], b["y"]))

            if overlap_x > 0 and overlap_y > 0:
                collisions.append({
                    "a": ids[i], "b": ids[j],
                    "overlap": overlap_x * overlap_y,
                    "z": a.get("z", 3)
                })
                # Repulsion force: push apart along the axis of least overlap
                if overlap_x < overlap_y:
                    push = overlap_x / 2 + 5
                    if a["x"] < b["x"]:
                        force_field[ids[i]]["dx"] -= push
                        force_field[ids[j]]["dx"] += push
                    else:
                        force_field[ids[i]]["dx"] += push
                        force_field[ids[j]]["dx"] -= push
                else:
                    push = overlap_y / 2 + 5
                    if a["y"] < b["y"]:
                        force_field[ids[i]]["dy"] -= push
                        force_field[ids[j]]["dy"] += push
                    else:
                        force_field[ids[i]]["dy"] += push
                        force_field[ids[j]]["dy"] -= push

    write_channel("physics/force_field.json", force_field)
    write_channel("physics/collision.json", {"collisions": collisions, "count": len(collisions)})

    print(f"[Physics] {len(collisions)} collisions detected")
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
