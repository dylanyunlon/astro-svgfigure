#!/usr/bin/env python3
"""
msdf_gen.py — MSDF Pipeline for Astro SVGFigure
================================================
Converts cell species params → multi-channel signed distance field textures.

Strategy:
  1. Read channels/cell/{id}/params.json  (species + species_params + bbox)
  2. Algorithmically generate SVG path d-strings from species geometry
  3. Wrap paths in a proper standalone SVG file → /tmp/msdf_{id}.svg
  4. Run bin/msdfgen.linux msdf -svg ... -o channels/cell/{id}/msdf.png
  5. Optionally produce a test-render preview at channels/cell/{id}/msdf_preview.png

Supported species and their species_params keys:
  cil-eye         ring_count, pupil_radius, r_outer, r_inner_ratio
  cil-bolt        zigzag_count, amplitude, seg_width
  cil-plus        arm_length, stroke_width, dash_corners
  cil-vector      arrow_count, arrow_length, angle_spread
  cil-arrow-right arrow_width, arrow_height

Usage:
  python3 backend/pipeline/msdf_gen.py <cell_id> [--size WxH] [--pxrange N] [--preview]
  python3 backend/pipeline/msdf_gen.py self_attn
  python3 backend/pipeline/msdf_gen.py all        # process every cell

Active Theory note:
  MSDF textures keep glyph / shape edges perfectly crisp at any zoom level.
  The three colour channels each store a signed distance measured from a
  different edge direction; the GPU combines them at render time so
  sub-pixel corners remain razor-sharp — unlike ordinary alpha or SDF.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
MSDFGEN_BIN = REPO_ROOT / "bin" / "msdfgen.linux"
CELL_DIR    = REPO_ROOT / "channels" / "cell"

# ── Low-level path primitives ───────────────────────────────────────────────

def circle_path(cx, cy, r):
    """Cubic-bezier approximation of a circle (4-arc, k≈0.5523)."""
    cx, cy, r = float(cx), float(cy), float(r)
    k = 0.5523 * r
    return (
        f"M {cx:.4f},{cy-r:.4f} "
        f"C {cx+k:.4f},{cy-r:.4f} {cx+r:.4f},{cy-k:.4f} {cx+r:.4f},{cy:.4f} "
        f"C {cx+r:.4f},{cy+k:.4f} {cx+k:.4f},{cy+r:.4f} {cx:.4f},{cy+r:.4f} "
        f"C {cx-k:.4f},{cy+r:.4f} {cx-r:.4f},{cy+k:.4f} {cx-r:.4f},{cy:.4f} "
        f"C {cx-r:.4f},{cy-k:.4f} {cx-k:.4f},{cy-r:.4f} {cx:.4f},{cy-r:.4f} Z"
    )


def rect_path(x, y, w, h, rx=0, ry=0):
    """Rounded-rectangle path (k≈0.5523 cubic bezier arcs)."""
    rx = min(float(rx), float(w) / 2)
    ry = min(float(ry), float(h) / 2)
    x, y, w, h = float(x), float(y), float(w), float(h)
    if rx == 0 and ry == 0:
        return f"M {x},{y} L {x+w},{y} L {x+w},{y+h} L {x},{y+h} Z"
    k = 0.5523
    return (
        f"M {x+rx:.4f},{y:.4f} "
        f"L {x+w-rx:.4f},{y:.4f} "
        f"C {x+w-rx+rx*k:.4f},{y:.4f} {x+w:.4f},{y+ry-ry*k:.4f} {x+w:.4f},{y+ry:.4f} "
        f"L {x+w:.4f},{y+h-ry:.4f} "
        f"C {x+w:.4f},{y+h-ry+ry*k:.4f} {x+w-rx+rx*k:.4f},{y+h:.4f} {x+w-rx:.4f},{y+h:.4f} "
        f"L {x+rx:.4f},{y+h:.4f} "
        f"C {x+rx-rx*k:.4f},{y+h:.4f} {x:.4f},{y+h-ry+ry*k:.4f} {x:.4f},{y+h-ry:.4f} "
        f"L {x:.4f},{y+ry:.4f} "
        f"C {x:.4f},{y+ry-ry*k:.4f} {x+rx-rx*k:.4f},{y:.4f} {x+rx:.4f},{y:.4f} Z"
    )


def line_path(x1, y1, x2, y2):
    """Open line segment as a path (stroke only, no fill area)."""
    return f"M {float(x1):.4f},{float(y1):.4f} L {float(x2):.4f},{float(y2):.4f}"


# ── Species path generators ─────────────────────────────────────────────────

def paths_cil_eye(w, h, sp):
    """
    cil-eye: central pupil circle + radial ray lines emanating outward.
      species_params: ring_count, pupil_radius, r_outer, r_inner_ratio
    """
    ring_count   = int(sp.get("ring_count",   10))
    pupil_r      = float(sp.get("pupil_radius", 4.2))
    r_outer      = float(sp.get("r_outer",      21.0))
    r_inner_ratio = float(sp.get("r_inner_ratio", 0.3))

    cx = w / 2.0
    cy = h / 2.0
    r_inner = r_outer * r_inner_ratio

    paths = []
    # Bounding rect (rounded)
    paths.append(rect_path(0, 0, w, h, rx=8))
    # Radial rays
    for i in range(ring_count):
        angle = 2 * math.pi * i / ring_count
        x1 = cx + r_inner * math.cos(angle)
        y1 = cy + r_inner * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle)
        y2 = cy + r_outer * math.sin(angle)
        paths.append(line_path(x1, y1, x2, y2))
    # Pupil (filled circle)
    paths.append(circle_path(cx, cy, pupil_r))
    return paths


def paths_cil_bolt(w, h, sp):
    """
    cil-bolt: ascending zigzag / polyline representing a feed-forward ramp.
      species_params: zigzag_count, amplitude, seg_width
    The line rises from left-bottom to right-top with zigzag teeth.
    """
    zigzag_count = int(sp.get("zigzag_count", 6))
    amplitude    = float(sp.get("amplitude",   6.0))
    seg_width    = float(sp.get("seg_width",  20.0))

    paths = []
    paths.append(rect_path(0, 0, w, h, rx=8))

    # Build ascending zigzag from left to right across the cell
    # Overall line travels from (margin, h*0.6) up to (w-margin, h*0.2)
    margin = 10.0
    x_start, y_start = margin, h * 0.6
    x_end,   y_end   = w - margin, h * 0.2
    total_w  = x_end - x_start
    total_h  = y_end - y_start  # negative = upward

    pts = [(x_start, y_start)]
    for i in range(1, zigzag_count + 1):
        t = i / zigzag_count
        bx = x_start + t * total_w
        by = y_start + t * total_h
        # alternate above / below the baseline
        side = 1 if i % 2 == 1 else -1
        tx = by + side * amplitude   # perpendicular offset (swap x/y for normal)
        ty = bx - side * amplitude
        # project offset onto perpendicular of the line direction
        # direction unit vector
        dx = total_w / math.hypot(total_w, total_h)
        dy = total_h / math.hypot(total_w, total_h)
        # perpendicular: (-dy, dx)
        pts.append((bx + side * amplitude * (-dy),
                    by + side * amplitude * dx))

    pts.append((x_end, y_end))
    d = "M " + " L ".join(f"{px:.4f},{py:.4f}" for px, py in pts)
    paths.append(d)
    return paths


def paths_cil_plus(w, h, sp):
    """
    cil-plus: cross (+ symbol) with optional corner dash lines.
      species_params: arm_length, stroke_width, dash_corners
    """
    arm_len      = float(sp.get("arm_length",   10.0))
    stroke_w     = float(sp.get("stroke_width",  2.5))
    dash_corners = bool(sp.get("dash_corners",   True))

    cx = w / 2.0
    cy = h / 2.0

    paths = []
    paths.append(rect_path(0, 0, w, h, rx=8))

    # Horizontal arm
    paths.append(line_path(cx - arm_len, cy, cx + arm_len, cy))
    # Vertical arm
    paths.append(line_path(cx, cy - arm_len, cx, cy + arm_len))

    if dash_corners:
        # Four diagonal "corner" dashes emanating from centre at 45°
        diag = arm_len * 0.85
        for angle in (math.pi / 4, 3 * math.pi / 4,
                      5 * math.pi / 4, 7 * math.pi / 4):
            x2 = cx + diag * math.cos(angle)
            y2 = cy + diag * math.sin(angle)
            paths.append(line_path(cx, cy, x2, y2))

    return paths


def paths_cil_vector(w, h, sp):
    """
    cil-vector: fan of arrows spreading across the cell (embedding fan).
      species_params: arrow_count, arrow_length, angle_spread
    Each arrow goes from left-centre to right-side with a small arrowhead.
    """
    arrow_count  = int(sp.get("arrow_count",   5))
    arrow_length = float(sp.get("arrow_length", 48.0))
    angle_spread = float(sp.get("angle_spread", 0.8))  # total spread in radians

    paths = []
    paths.append(rect_path(0, 0, w, h, rx=8))

    cx = w / 2.0
    cy = h / 2.0
    head_size = 4.0  # arrowhead leg length

    for i in range(arrow_count):
        # angles spread symmetrically around 0 (rightward)
        if arrow_count > 1:
            t = i / (arrow_count - 1)  # 0..1
        else:
            t = 0.5
        angle = -angle_spread / 2 + t * angle_spread

        x1 = cx - arrow_length / 2
        y1 = cy
        x2 = cx + arrow_length / 2 * math.cos(angle)
        y2 = cy + arrow_length / 2 * math.sin(angle)

        # Shaft
        paths.append(line_path(x1, y1, x2, y2))

        # Arrowhead (two short lines)
        back_angle = math.atan2(y1 - y2, x1 - x2)
        for da in (-0.4, 0.4):
            hx = x2 + head_size * math.cos(back_angle + da)
            hy = y2 + head_size * math.sin(back_angle + da)
            paths.append(line_path(x2, y2, hx, hy))

    return paths


def paths_cil_arrow_right(w, h, sp):
    """
    cil-arrow-right: a simple rightward-pointing filled triangle / chevron.
      species_params: arrow_width, arrow_height
    """
    arrow_w = float(sp.get("arrow_width",  48.0))
    arrow_h = float(sp.get("arrow_height", 16.0))

    cx = w / 2.0
    cy = h / 2.0

    paths = []
    paths.append(rect_path(0, 0, w, h, rx=8))

    # Triangle: left-top → right-centre → left-bottom → close
    lx = cx - arrow_w / 2
    rx_ = cx + arrow_w / 2
    top_y    = cy - arrow_h / 2
    bottom_y = cy + arrow_h / 2

    d = (f"M {lx:.4f},{top_y:.4f} "
         f"L {rx_:.4f},{cy:.4f} "
         f"L {lx:.4f},{bottom_y:.4f} Z")
    paths.append(d)
    return paths


# ── Dispatch table ──────────────────────────────────────────────────────────

SPECIES_GENERATORS = {
    "cil-eye":         paths_cil_eye,
    "cil-bolt":        paths_cil_bolt,
    "cil-plus":        paths_cil_plus,
    "cil-vector":      paths_cil_vector,
    "cil-arrow-right": paths_cil_arrow_right,
}


def generate_paths_from_params(params):
    """
    Read species + species_params + bbox from a params dict and return
    (list_of_path_d_strings, canvas_w, canvas_h).
    """
    species  = params.get("species", "")
    sp       = params.get("species_params", {})
    bbox     = params.get("bbox", {})
    cw       = float(bbox.get("w", 160))
    ch       = float(bbox.get("h",  60))

    generator = SPECIES_GENERATORS.get(species)
    if generator is None:
        # Fallback: plain rounded rect so msdfgen still has something to work with
        print(f"  [warn] unknown species '{species}', using fallback rect",
              file=sys.stderr)
        return [rect_path(0, 0, cw, ch, rx=8)], cw, ch

    paths = generator(cw, ch, sp)
    return paths, cw, ch


# ── SVG file builder ─────────────────────────────────────────────────────────

def build_standalone_svg(paths, canvas_w, canvas_h, padding=8):
    """
    Combine path d-strings into a single valid SVG file.
    msdfgen reads only the *last* path in the file (v1.5 design), so we
    merge everything into one compound path using the M … Z idiom.
    """
    if not paths:
        raise ValueError("No paths to write into SVG")

    compound = " ".join(p.strip() for p in paths if p.strip())

    vb_x = -padding
    vb_y = -padding
    vb_w = canvas_w + padding * 2
    vb_h = canvas_h + padding * 2

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vb_x} {vb_y} {vb_w} {vb_h}" '
        f'width="{vb_w}" height="{vb_h}">\n'
        f'  <path d="{compound}"/>\n'
        "</svg>\n"
    )
    return svg


# ── msdfgen runner ───────────────────────────────────────────────────────────

def run_msdfgen(svg_path, output_png, size=(64, 32), pxrange=4, preview=False,
                preview_png=None):
    """Invoke msdfgen.linux and return (success, stdout, stderr)."""
    if not MSDFGEN_BIN.exists():
        raise FileNotFoundError(f"msdfgen binary not found: {MSDFGEN_BIN}")

    cmd = [
        str(MSDFGEN_BIN), "msdf",
        "-svg",    str(svg_path),
        "-o",      str(output_png),
        "-size",   str(size[0]), str(size[1]),
        "-pxrange", str(pxrange),
        "-autoframe",
        "-yflip",
    ]

    if preview and preview_png:
        cmd += ["-testrender", str(preview_png),
                str(size[0] * 4), str(size[1] * 4)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


# ── Per-cell entry point ─────────────────────────────────────────────────────

def process_cell(cell_id, size=(64, 32), pxrange=4, preview=False, verbose=True):
    """
    Full pipeline for one cell.
    Reads params.json → generates paths algorithmically → runs msdfgen.
    Returns True on success.
    """
    cell_path  = CELL_DIR / cell_id
    params_file = cell_path / "params.json"
    out_png    = cell_path / "msdf.png"
    prev_png   = cell_path / "msdf_preview.png"

    if not params_file.exists():
        print(f"  [ERROR] params.json not found: {params_file}", file=sys.stderr)
        return False

    if verbose:
        print(f"── {cell_id} ──────────────────────────────────")

    # 1. Load params
    try:
        params = json.loads(params_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse error in params.json: {e}", file=sys.stderr)
        return False

    species = params.get("species", "<unknown>")
    if verbose:
        print(f"  species: {species}")

    # 2. Generate path d-strings from species + species_params
    try:
        paths, cw, ch = generate_paths_from_params(params)
    except Exception as e:
        print(f"  [ERROR] path generation failed: {e}", file=sys.stderr)
        return False

    if verbose:
        print(f"  paths generated: {len(paths)}  canvas: {cw}×{ch}")

    if not paths:
        print(f"  [WARN] No paths generated for {cell_id}, skipping.",
              file=sys.stderr)
        return False

    # 3. Build standalone SVG
    try:
        standalone_svg = build_standalone_svg(paths, cw, ch)
    except ValueError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return False

    # 4. Write to temp file
    with tempfile.NamedTemporaryFile(
            suffix=f"_msdf_{cell_id}.svg", mode="w",
            delete=False, encoding="utf-8") as tf:
        tf.write(standalone_svg)
        tmp_svg = tf.name

    if verbose:
        print(f"  temp SVG: {tmp_svg}")

    # 5. Run msdfgen
    ok, stdout, stderr = run_msdfgen(
        tmp_svg, out_png,
        size=size, pxrange=pxrange,
        preview=preview, preview_png=prev_png if preview else None,
    )

    # Cleanup temp
    os.unlink(tmp_svg)

    if ok:
        size_kb = out_png.stat().st_size / 1024
        if verbose:
            print(f"  ✓  msdf.png  ({size[0]}×{size[1]}px, {size_kb:.1f} KB)")
        if preview and prev_png.exists():
            print(f"  ✓  msdf_preview.png  ({size[0]*4}×{size[1]*4}px)")
    else:
        print(f"  [ERROR] msdfgen failed:", file=sys.stderr)
        if stdout:
            print(f"    stdout: {stdout.strip()}", file=sys.stderr)
        if stderr:
            print(f"    stderr: {stderr.strip()}", file=sys.stderr)

    return ok


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MSDF Pipeline — generate MSDF textures from cell params.json")
    parser.add_argument("cell_id",
        help="Cell ID (e.g. self_attn) or 'all' to process every cell")
    parser.add_argument("--size", default="64x32",
        help="Output texture size WxH in pixels (default: 64x32)")
    parser.add_argument("--pxrange", type=int, default=4,
        help="MSDF pixel range (default: 4)")
    parser.add_argument("--preview", action="store_true",
        help="Also generate a test-render preview PNG")
    args = parser.parse_args()

    try:
        sw, sh = map(int, args.size.lower().split("x"))
    except ValueError:
        print(f"Invalid --size '{args.size}', expected WxH", file=sys.stderr)
        sys.exit(1)

    if args.cell_id == "all":
        cells = sorted(p.name for p in CELL_DIR.iterdir() if p.is_dir())
    else:
        cells = [args.cell_id]

    success_count = 0
    for cid in cells:
        ok = process_cell(cid, size=(sw, sh),
                          pxrange=args.pxrange, preview=args.preview)
        if ok:
            success_count += 1

    total = len(cells)
    print(f"\n{'='*48}")
    print(f"MSDF pipeline complete: {success_count}/{total} cells succeeded")
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
