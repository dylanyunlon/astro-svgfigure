#!/usr/bin/env python3
"""
CellComponent — Apollo Component::Proc() equivalent for astro-svgfigure.

Each sub-Claude runs this with its cell_id.
It reads skeleton signal + force_field, uses species gene_traits to
algorithmically generate SVG via svgwrite, publishes bbox + svg + status.

Usage:
    python3 cell_component.py <cell_id>
    python3 cell_component.py self_attn
"""
import json
import math
import os
import sys

CHANNELS = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════
# Channel I/O — Apollo Reader/Writer equivalent
# ═══════════════════════════════════════════════

def read_channel(path: str) -> dict:
    """Subscribe = read JSON. Apollo CreateReader equivalent."""
    full = os.path.join(CHANNELS, path)
    with open(full) as f:
        return json.load(f)

def write_channel(path: str, data):
    """Publish = write file. Apollo CreateWriter equivalent."""
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)

# ═══════════════════════════════════════════════
# Species Gene Algorithms — each species generates differently
# These are NOT icon files — they are algorithmic generation styles
# ═══════════════════════════════════════════════

def generate_svg_cil_eye(cell_id, label, bbox, gene_traits):
    """cil-eye species: radial attention pattern — heatmap rays from center."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2
    r_outer = min(bbox["w"], bbox["h"]) / 2 - 4

    parts = []
    # Background rounded rect
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1.5"/>')

    # Radial attention rays (algorithmic — number based on label complexity)
    num_rays = max(4, len(label) // 2)
    for i in range(num_rays):
        angle = (2 * math.pi * i) / num_rays
        # Intensity gradient: rays get lighter toward edges
        intensity = 0.3 + 0.7 * (1 - i / num_rays)
        r_inner = r_outer * 0.3
        x1 = cx + r_inner * math.cos(angle)
        y1 = cy + r_inner * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle)
        y2 = cy + r_outer * math.sin(angle)
        opacity = max(0.15, intensity * 0.6)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#3F51B5" stroke-width="1" opacity="{opacity:.2f}"/>')

    # Center focal point
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_outer*0.2}" fill="#3F51B5" opacity="0.7"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_outer*0.08}" fill="#E8EAF6"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#1A237E">{label}</text>')

    return "\n".join(parts), bbox


def generate_svg_cil_vector(cell_id, label, bbox, gene_traits):
    """cil-vector species: embedding arrows — magnitude/direction lines."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5"/>')

    # Vector arrows (algorithmic — angle spread based on "positional" vs "input")
    num_arrows = 5
    arrow_len = bbox["w"] * 0.3
    for i in range(num_arrows):
        angle = -0.4 + (0.8 * i / (num_arrows - 1))  # spread ±0.4 rad
        x1 = cx - arrow_len * 0.5 * math.cos(angle)
        y1 = cy - arrow_len * 0.5 * math.sin(angle) + 2
        x2 = cx + arrow_len * 0.5 * math.cos(angle)
        y2 = cy + arrow_len * 0.5 * math.sin(angle) + 2
        weight = 1 + (i % 3) * 0.5
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#2E7D32" stroke-width="{weight:.1f}" opacity="0.5" '
                     f'marker-end="url(#arrow-green)"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#1B5E20">{label}</text>')

    return "\n".join(parts), bbox


def generate_svg_cil_bolt(cell_id, label, bbox, gene_traits):
    """cil-bolt species: zigzag activation — ReLU-style angular patterns."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#FFF3E0" stroke="#E65100" stroke-width="1.5"/>')

    # ReLU zigzag (algorithmic pattern)
    points = []
    segments = 6
    seg_w = (bbox["w"] - 20) / segments
    for i in range(segments + 1):
        x = 10 + i * seg_w
        # ReLU shape: flat left half, rising right half
        if i < segments // 2:
            y = cy + 5
        else:
            y = cy + 5 - (i - segments // 2) * 6
        points.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(points)}" '
                 f'fill="none" stroke="#E65100" stroke-width="2" opacity="0.7"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#BF360C">{label}</text>')

    return "\n".join(parts), bbox


def generate_svg_cil_plus(cell_id, label, bbox, gene_traits):
    """cil-plus species: merge/aggregation — converging lines to center."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#FCE4EC" stroke="#C62828" stroke-width="1.5"/>')

    # Plus cross (algorithmic)
    arm = min(bbox["w"], bbox["h"]) * 0.25
    parts.append(f'<line x1="{cx-arm}" y1="{cy}" x2="{cx+arm}" y2="{cy}" '
                 f'stroke="#C62828" stroke-width="2.5" opacity="0.6"/>')
    parts.append(f'<line x1="{cx}" y1="{cy-arm}" x2="{cx}" y2="{cy+arm}" '
                 f'stroke="#C62828" stroke-width="2.5" opacity="0.6"/>')

    # Converging arcs from corners
    for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
        sx = cx + dx * arm * 1.2
        sy = cy + dy * arm * 1.2
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{cx}" y2="{cy}" '
                     f'stroke="#C62828" stroke-width="0.8" opacity="0.3" stroke-dasharray="3,2"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#B71C1C">{label}</text>')

    return "\n".join(parts), bbox


def generate_svg_cil_arrow_right(cell_id, label, bbox, gene_traits):
    """cil-arrow-right species: dataflow terminal — arrow pointing out."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#ECEFF1" stroke="#455A64" stroke-width="1.5"/>')

    # Arrow shape
    aw = bbox["w"] * 0.3
    parts.append(f'<polygon points="{cx-aw},{cy-8} {cx+aw},{cy} {cx-aw},{cy+8}" '
                 f'fill="#455A64" opacity="0.5"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#263238">{label}</text>')

    return "\n".join(parts), bbox


# Species → generator mapping
SPECIES_GENERATORS = {
    "cil-eye": generate_svg_cil_eye,
    "cil-vector": generate_svg_cil_vector,
    "cil-bolt": generate_svg_cil_bolt,
    "cil-plus": generate_svg_cil_plus,
    "cil-arrow-right": generate_svg_cil_arrow_right,
}


def proc(cell_id: str):
    """
    Apollo Component::Proc() equivalent.
    Reads channels → generates SVG with species algorithm → publishes.
    """
    # ── Subscribe: read channels ──
    skeleton = read_channel(f"skeleton/cell/{cell_id}.json")
    force_field = read_channel("physics/force_field.json")
    z_layers = read_channel("physics/z_layers.json")

    label = skeleton["label"]
    species = skeleton["species"]
    gene_traits = skeleton["gene_traits"]
    bbox = skeleton["initial_bbox"].copy()

    # Apply force field adjustments
    force = force_field.get(cell_id, {"dx": 0, "dy": 0, "dz": 0})
    bbox["x"] += force["dx"]
    bbox["y"] += force["dy"]
    bbox["z"] = z_layers.get(cell_id, 3) + force.get("dz", 0)

    # ── Proc: species algorithm generates SVG ──
    generator = SPECIES_GENERATORS.get(species, generate_svg_cil_arrow_right)
    svg_content, actual_bbox = generator(cell_id, label, bbox, gene_traits)

    # Wrap in positioned <g> with z-layer data attribute
    full_svg = (
        f'<g id="cell-{cell_id}" data-z="{bbox["z"]}" '
        f'transform="translate({bbox["x"]},{bbox["y"]})">\n'
        f'{svg_content}\n'
        f'</g>'
    )

    # ── Publish: write to channels ──
    cell_dir = f"cell/{cell_id}"
    write_channel(f"{cell_dir}/bbox.json", {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
        "species": species,
        "epoch": read_channel("skeleton/epoch.json")["current"]
    })
    write_channel(f"{cell_dir}/svg.svg", full_svg)
    write_channel(f"{cell_dir}/status.json", {
        "status": "converged",
        "cell_id": cell_id,
        "species": species,
        "epoch": read_channel("skeleton/epoch.json")["current"]
    })

    print(f"[Cell {cell_id}] species={species} bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) z={bbox['z']}")
    return full_svg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cell_component.py <cell_id>")
        print("  e.g. python3 cell_component.py self_attn")
        sys.exit(1)
    cell_id = sys.argv[1]
    proc(cell_id)
