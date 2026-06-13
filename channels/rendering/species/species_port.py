import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



def _species_f0(species: str) -> float:
    """
    Per-species F0 (normal-incidence reflectance).

    Mirrors the per-material F0 values baked into the C++ substrate BRDF
    evaluation; here we encode them per species as a proxy for material type.

    High F0  → metallic / polished  (cil-bolt, cil-filter)
    Low F0   → diffuse / matte      (cil-plus, cil-graph)
    Mid F0   → dielectric           (all others)
    """
    _F0_TABLE = {
        "cil-eye":         0.04,   # dielectric — clear lens
        "cil-bolt":        0.80,   # metallic — energetic spike
        "cil-vector":      0.04,   # dielectric — direction arrow
        "cil-plus":        0.02,   # low-gloss matte — aggregator
        "cil-arrow-right": 0.06,   # slight sheen — terminal node
        "cil-filter":      0.65,   # semi-metallic — kernel weight matrix
        "cil-code":        0.04,   # dielectric — monospace brace
        "cil-layers":      0.08,   # slight gloss — depth stack
        "cil-loop":        0.10,   # subtle gloss — cyclic arc
        "cil-graph":       0.03,   # matte — node/edge graph
    }
    return _F0_TABLE.get(species, 0.04)





def _species_to_index(species_name: str) -> int:
    """Map a species name string to its canonical integer index."""
    return _SPECIES_NAME_TO_INDEX.get(species_name, 0)



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

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    # BuildCellDecorationFromDecal → ApplyCellDecorationToConstraintBuffer
    # Injected before return, mirroring C++ overlay before DrawIndexedPrimitive.
    _deco = _build_cell_decoration("cil-eye", gene_traits, cell_id)
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-eye", cx, cy, r_outer, bbox)
    if _overlay:
        parts.append(_overlay)

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

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-vector", gene_traits, cell_id)
    _r_outer_v = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-vector", cx, cy, _r_outer_v, bbox)
    if _overlay:
        parts.append(_overlay)

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

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-bolt", gene_traits, cell_id)
    _r_outer_b = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-bolt", cx, cy, _r_outer_b, bbox)
    if _overlay:
        parts.append(_overlay)

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

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-plus", gene_traits, cell_id)
    _r_outer_p = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-plus", cx, cy, _r_outer_p, bbox)
    if _overlay:
        parts.append(_overlay)

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

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    # Default / fallback species → generic rounded-stroke decoration overlay
    _deco = _build_cell_decoration(species if "species" in dir() else "default",
                                   gene_traits, cell_id)
    _deco.blend_weight = 0.6   # force visible for default species
    _r_outer_d = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "default", cx, cy, _r_outer_d, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox





def generate_svg_filter(w, h, label):
    """
    filter species: 3×3 grid wireframe + label.
    Represents a convolution / attention mask — gridded sampling pattern.
    """
    cx = w / 2
    cy = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#F3E5F5" stroke="#7B1FA2" stroke-width="1.5"/>')

    # 3×3 grid wireframe centred in the cell
    pad   = max(8, min(w, h) * 0.12)
    gw    = w - 2 * pad
    gh    = (h - 2 * pad) * 0.72    # leave room for label at bottom
    cell_w = gw / 3
    cell_h = gh / 3
    gx0   = pad
    gy0   = pad

    for row in range(4):
        y = gy0 + row * cell_h
        parts.append(f'<line x1="{gx0:.1f}" y1="{y:.1f}" '
                     f'x2="{gx0 + gw:.1f}" y2="{y:.1f}" '
                     f'stroke="#7B1FA2" stroke-width="1" opacity="0.55"/>')
    for col in range(4):
        x = gx0 + col * cell_w
        parts.append(f'<line x1="{x:.1f}" y1="{gy0:.1f}" '
                     f'x2="{x:.1f}" y2="{gy0 + gh:.1f}" '
                     f'stroke="#7B1FA2" stroke-width="1" opacity="0.55"/>')

    # Highlight centre cell of the 3×3 grid
    hx = gx0 + cell_w
    hy = gy0 + cell_h
    parts.append(f'<rect x="{hx:.1f}" y="{hy:.1f}" '
                 f'width="{cell_w:.1f}" height="{cell_h:.1f}" '
                 f'fill="#CE93D8" opacity="0.45" rx="2"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#4A148C">{label}</text>')

    return "\n".join(parts)





def generate_svg_code(w, h, label):
    """
    code species: curly-brace icon + monospace label.
    Represents a programmatic / function block in the architecture diagram.
    """
    cx   = w / 2
    cy   = h / 2
    arm  = min(w, h) * 0.28    # half-height of the brace

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5"/>')

    # Left curly brace  {  (two quarter-arcs + a nib)
    bx   = cx - arm * 1.1
    top  = cy - arm
    bot  = cy + arm
    nib  = arm * 0.22    # half-nib width
    r    = arm * 0.35    # corner radius of the brace arcs

    # Left brace path: M top → arc down → nib left → arc down → bottom
    parts.append(
        f'<path d="M {bx + r:.1f},{top:.1f} '
        f'Q {bx:.1f},{top:.1f} {bx:.1f},{top + r:.1f} '
        f'L {bx:.1f},{cy - nib:.1f} '
        f'Q {bx - nib * 1.4:.1f},{cy:.1f} {bx:.1f},{cy + nib:.1f} '
        f'L {bx:.1f},{bot - r:.1f} '
        f'Q {bx:.1f},{bot:.1f} {bx + r:.1f},{bot:.1f}" '
        f'fill="none" stroke="#2E7D32" stroke-width="2" stroke-linejoin="round"/>'
    )

    # Right curly brace  }  (mirrored)
    bx2 = cx + arm * 1.1
    parts.append(
        f'<path d="M {bx2 - r:.1f},{top:.1f} '
        f'Q {bx2:.1f},{top:.1f} {bx2:.1f},{top + r:.1f} '
        f'L {bx2:.1f},{cy - nib:.1f} '
        f'Q {bx2 + nib * 1.4:.1f},{cy:.1f} {bx2:.1f},{cy + nib:.1f} '
        f'L {bx2:.1f},{bot - r:.1f} '
        f'Q {bx2:.1f},{bot:.1f} {bx2 - r:.1f},{bot:.1f}" '
        f'fill="none" stroke="#2E7D32" stroke-width="2" stroke-linejoin="round"/>'
    )

    # Label in monospace font, centred
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="\'Courier New\',Courier,monospace" font-size="10" '
                 f'fill="#1B5E20">{label}</text>')

    return "\n".join(parts)





def generate_svg_layers(w, h, label):
    """
    layers species: 3 staggered semi-transparent rectangles.
    Represents depth / multi-layer representations (e.g. transformer stack).
    """
    cx = w / 2
    cy = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5"/>')

    pad   = max(6, min(w, h) * 0.10)
    rw    = w - 2 * pad
    rh    = (h - 2 * pad) * 0.48
    step  = rh * 0.32    # vertical / horizontal stagger between layers

    colours   = ["#90CAF9", "#42A5F5", "#1E88E5"]
    opacities = [0.35, 0.50, 0.68]
    rx_vals   = [6, 5, 4]

    for i, (col, op, rx) in enumerate(zip(colours, opacities, rx_vals)):
        # Stagger: bottom layers offset right + down
        offset = step * (2 - i)
        rx_pos = pad + offset * 0.8
        ry_pos = pad + offset + (h - 2 * pad - rh) * 0.15
        parts.append(
            f'<rect x="{rx_pos:.1f}" y="{ry_pos:.1f}" '
            f'width="{rw:.1f}" height="{rh:.1f}" '
            f'rx="{rx}" fill="{col}" opacity="{op}" '
            f'stroke="#1565C0" stroke-width="0.8"/>'
        )

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#0D47A1">{label}</text>')

    return "\n".join(parts)





def generate_svg_loop(w, h, label):
    """
    loop species: circular arc with an arrowhead.
    Represents a recurrent / cyclic flow (RNN, loop, feedback connection).
    """
    cx  = w / 2
    cy  = h / 2
    r   = min(w, h) * 0.28

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#FFF8E1" stroke="#F57F17" stroke-width="1.5"/>')

    # Circular arc: ~300° clockwise, leaving a gap at the top for the arrowhead
    # SVG arc: start at top-right (30°), sweep 300° (large-arc), end at top-left (90°)
    gap_half = math.radians(30)    # gap = 60° at top
    start_angle = -math.pi / 2 + gap_half   # just past 12 o'clock CW
    end_angle   = -math.pi / 2 - gap_half   # just before 12 o'clock CW

    sx = cx + r * math.cos(start_angle)
    sy = cy + r * math.sin(start_angle)
    ex = cx + r * math.cos(end_angle)
    ey = cy + r * math.sin(end_angle)

    # large-arc-flag=1, sweep-flag=1 (clockwise)
    arc_id = abs(hash(label)) % 9000 + 1000
    parts.append(
        f'<defs>'
        f'<marker id="loop-arrow-{arc_id}" markerWidth="7" markerHeight="7" '
        f'refX="3.5" refY="3.5" orient="auto">'
        f'<path d="M0,0 L7,3.5 L0,7 Z" fill="#F57F17"/>'
        f'</marker>'
        f'</defs>'
    )
    parts.append(
        f'<path d="M {sx:.2f},{sy:.2f} A {r:.2f},{r:.2f} 0 1,1 {ex:.2f},{ey:.2f}" '
        f'fill="none" stroke="#F57F17" stroke-width="2.2" opacity="0.75" '
        f'marker-end="url(#loop-arrow-{arc_id})"/>'
    )

    # Small centre dot
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.14:.1f}" '
                 f'fill="#F57F17" opacity="0.55"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#E65100">{label}</text>')

    return "\n".join(parts)





def generate_svg_graph(w, h, label):
    """
    graph species: small circles (nodes) connected by lines (edges).
    Represents a graph-structured computation or attention pattern.
    """
    cx  = w / 2
    cy  = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#FAFAFA" stroke="#37474F" stroke-width="1.5"/>')

    # Node positions — a small 5-node graph (star + outer ring feel)
    r_outer = min(w, h) * 0.28
    r_inner = r_outer * 0.38
    # Centre node + 4 outer nodes at 0°, 90°, 180°, 270°
    node_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    nodes = [(cx, cy - r_inner)]   # slightly offset centre
    for ang in node_angles:
        nodes.append((cx + r_outer * math.cos(ang),
                      cy + r_outer * math.sin(ang)))

    # Edges: centre → each outer node + one cross edge
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
    for (a, b) in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#546E7A" stroke-width="1.1" opacity="0.50"/>'
        )

    # Draw nodes on top of edges
    node_r  = max(3.0, min(w, h) * 0.055)
    colours = ["#37474F", "#78909C", "#78909C", "#78909C", "#78909C"]
    for i, (nx, ny) in enumerate(nodes):
        col = colours[min(i, len(colours) - 1)]
        parts.append(
            f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{node_r:.1f}" '
            f'fill="{col}" opacity="0.75"/>'
        )

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#263238">{label}</text>')

    return "\n".join(parts)


# Species → generator mapping
SPECIES_GENERATORS = {
    "cil-eye": generate_svg_cil_eye,
    "cil-vector": generate_svg_cil_vector,
    "cil-bolt": generate_svg_cil_bolt,
    "cil-plus": generate_svg_cil_plus,
    "cil-arrow-right": generate_svg_cil_arrow_right,
    # ── New species (feat: add 5 new species SVG generators) ──────────────────
    "cil-filter": lambda cell_id, label, bbox, gt: (
        generate_svg_filter(bbox["w"], bbox["h"], label), bbox),
    "cil-code": lambda cell_id, label, bbox, gt: (
        generate_svg_code(bbox["w"], bbox["h"], label), bbox),
    "cil-layers": lambda cell_id, label, bbox, gt: (
        generate_svg_layers(bbox["w"], bbox["h"], label), bbox),
    "cil-loop": lambda cell_id, label, bbox, gt: (
        generate_svg_loop(bbox["w"], bbox["h"], label), bbox),
    "cil-graph": lambda cell_id, label, bbox, gt: (
        generate_svg_graph(bbox["w"], bbox["h"], label), bbox),
}


