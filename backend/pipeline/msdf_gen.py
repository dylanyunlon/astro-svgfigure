#!/usr/bin/env python3
"""
msdf_gen.py — MSDF Pipeline for Astro SVGFigure
================================================
Converts cell SVG fragments → multi-channel signed distance field textures.

Strategy:
  1. Parse the SVG fragment (no root <svg> tag) from channels/cell/{id}/svg.svg
  2. Extract geometric shapes: <path>, <rect>, <circle>, <polyline>, <polygon>, <line>
  3. Convert every shape to SVG <path> d-strings (msdfgen only understands paths)
  4. Wrap paths in a proper standalone SVG file → /tmp/msdf_{id}.svg
  5. Run bin/msdfgen.linux msdf -svg ... -o channels/cell/{id}/msdf.png
  6. Optionally produce a test-render preview at channels/cell/{id}/msdf_preview.png

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
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
MSDFGEN_BIN = REPO_ROOT / "bin" / "msdfgen.linux"
CELL_DIR    = REPO_ROOT / "channels" / "cell"

# ── Shape → path conversion helpers ────────────────────────────────────────

def rect_to_path(x, y, w, h, rx=0, ry=0):
    """Convert <rect> (with optional corner radii) to a path d-string."""
    rx = min(float(rx), float(w) / 2)
    ry = min(float(ry), float(h) / 2)
    x, y, w, h = float(x), float(y), float(w), float(h)
    if rx == 0 and ry == 0:
        return (f"M {x},{y} L {x+w},{y} L {x+w},{y+h} "
                f"L {x},{y+h} Z")
    # Rounded rectangle using cubic bezier arcs (k ≈ 0.5523)
    k = 0.5523
    return (
        f"M {x+rx},{y} "
        f"L {x+w-rx},{y} "
        f"C {x+w-rx+rx*k},{y} {x+w},{y+ry-ry*k} {x+w},{y+ry} "
        f"L {x+w},{y+h-ry} "
        f"C {x+w},{y+h-ry+ry*k} {x+w-rx+rx*k},{y+h} {x+w-rx},{y+h} "
        f"L {x+rx},{y+h} "
        f"C {x+rx-rx*k},{y+h} {x},{y+h-ry+ry*k} {x},{y+h-ry} "
        f"L {x},{y+ry} "
        f"C {x},{y+ry-ry*k} {x+rx-rx*k},{y} {x+rx},{y} "
        f"Z"
    )


def circle_to_path(cx, cy, r):
    """Convert <circle> to a cubic-bezier path (4-arc approximation)."""
    cx, cy, r = float(cx), float(cy), float(r)
    k = 0.5523 * r
    return (
        f"M {cx},{cy-r} "
        f"C {cx+k},{cy-r} {cx+r},{cy-k} {cx+r},{cy} "
        f"C {cx+r},{cy+k} {cx+k},{cy+r} {cx},{cy+r} "
        f"C {cx-k},{cy+r} {cx-r},{cy+k} {cx-r},{cy} "
        f"C {cx-r},{cy-k} {cx-k},{cy-r} {cx},{cy-r} "
        f"Z"
    )


def points_to_path(points_str, close=False):
    """Convert SVG points attribute (polyline / polygon) to a path d-string."""
    pts = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",
                     points_str)
    if len(pts) < 4:
        return None
    coords = list(map(float, pts))
    pairs  = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
    d = f"M {pairs[0][0]},{pairs[0][1]}"
    for px, py in pairs[1:]:
        d += f" L {px},{py}"
    if close:
        d += " Z"
    return d


def line_to_path(x1, y1, x2, y2):
    """Convert <line> to a path d-string (open stroke, no area)."""
    return (f"M {float(x1)},{float(y1)} L {float(x2)},{float(y2)}")


# ── SVG bounding-box helpers ────────────────────────────────────────────────

def parse_viewbox_or_rect(root_attrib):
    """Best-effort extraction of (x, y, w, h) canvas from a <g> element."""
    # We rely on the rect dimensions embedded in the fragment
    return None


def get_shape_bbox_from_paths(path_strings):
    """
    Rough bounding box from path coordinates (handles M/L/C/Q).
    Returns (min_x, min_y, max_x, max_y) or None.
    """
    nums = []
    for d in path_strings:
        nums.extend(map(float, re.findall(
            r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)))
    if not nums:
        return None
    # pair up as x,y (rough — ignores control points intent but good enough
    # for bounding box purposes)
    xs = nums[0::2]
    ys = nums[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ── Main extraction ─────────────────────────────────────────────────────────

SVG_NS = "http://www.w3.org/2000/svg"

def _tag(el):
    """Strip namespace from tag name."""
    return el.tag.replace(f"{{{SVG_NS}}}", "").replace("{}", "")


def extract_paths_from_fragment(svg_fragment_text):
    """
    Parse an SVG fragment (no root <svg>) and return:
      - list of (d_string, stroke_width) tuples  — shapes converted to paths
      - (canvas_w, canvas_h) best guess
    """
    # Wrap in root so ElementTree can parse it
    wrapped = (
        '<?xml version="1.0"?>'
        f'<svg xmlns="{SVG_NS}" xmlns:xlink="http://www.w3.org/1999/xlink">'
        + svg_fragment_text
        + "</svg>"
    )
    # Strip comments (ElementTree doesn't handle them gracefully in all versions)
    wrapped = re.sub(r"<!--.*?-->", "", wrapped, flags=re.DOTALL)

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        print(f"  [warn] XML parse error: {e}; retrying with strict stripping")
        wrapped = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", wrapped)
        root = ET.fromstring(wrapped)

    paths = []         # list of d-strings
    canvas_w = 160     # fallback
    canvas_h = 60

    def walk(el, transform_stack):
        tag = _tag(el)

        # ── Detect canvas from outermost rect ──
        nonlocal canvas_w, canvas_h
        if tag == "rect":
            a = el.attrib
            w = float(a.get("width",  canvas_w))
            h = float(a.get("height", canvas_h))
            # take largest rect as canvas hint
            if w * h > canvas_w * canvas_h:
                canvas_w, canvas_h = w, h
            rx = float(a.get("rx", 0))
            ry = float(a.get("ry", rx))
            d  = rect_to_path(
                a.get("x", 0), a.get("y", 0), w, h, rx, ry)
            paths.append(d)

        elif tag == "path":
            d = el.attrib.get("d", "").strip()
            if d:
                paths.append(d)

        elif tag == "circle":
            a  = el.attrib
            cx = a.get("cx", 0)
            cy = a.get("cy", 0)
            r  = a.get("r",  1)
            paths.append(circle_to_path(cx, cy, r))

        elif tag == "ellipse":
            a  = el.attrib
            cx, cy = float(a.get("cx", 0)), float(a.get("cy", 0))
            rx2, ry2 = float(a.get("rx", 1)), float(a.get("ry", 1))
            k = 0.5523
            paths.append(
                f"M {cx},{cy-ry2} "
                f"C {cx+rx2*k},{cy-ry2} {cx+rx2},{cy-ry2*k} {cx+rx2},{cy} "
                f"C {cx+rx2},{cy+ry2*k} {cx+rx2*k},{cy+ry2} {cx},{cy+ry2} "
                f"C {cx-rx2*k},{cy+ry2} {cx-rx2},{cy+ry2*k} {cx-rx2},{cy} "
                f"C {cx-rx2},{cy-ry2*k} {cx-rx2*k},{cy-ry2} {cx},{cy-ry2} Z"
            )

        elif tag == "polyline":
            d = points_to_path(el.attrib.get("points", ""), close=False)
            if d:
                paths.append(d)

        elif tag == "polygon":
            d = points_to_path(el.attrib.get("points", ""), close=True)
            if d:
                paths.append(d)

        elif tag == "line":
            a = el.attrib
            paths.append(line_to_path(
                a.get("x1", 0), a.get("y1", 0),
                a.get("x2", 0), a.get("y2", 0)))

        # Recurse into groups / defs / etc.
        for child in el:
            walk(child, transform_stack)

    walk(root, [])
    return paths, (canvas_w, canvas_h)


# ── SVG file builder ─────────────────────────────────────────────────────────

def build_standalone_svg(paths, canvas_w, canvas_h, padding=8):
    """
    Combine extracted path d-strings into a single valid SVG file.
    msdfgen reads only the *last* path in the file (by design in v1.5),
    so we merge everything into one compound path using the M … Z idiom.
    """
    if not paths:
        raise ValueError("No paths extracted from SVG fragment")

    # Merge into one compound path
    compound = " ".join(p.strip() for p in paths if p.strip())

    vb_x = -padding
    vb_y = -padding
    vb_w = canvas_w  + padding * 2
    vb_h = canvas_h  + padding * 2

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
    Returns True on success.
    """
    cell_path = CELL_DIR / cell_id
    svg_frag  = cell_path / "svg.svg"
    out_png   = cell_path / "msdf.png"
    prev_png  = cell_path / "msdf_preview.png"

    if not svg_frag.exists():
        print(f"  [ERROR] svg.svg not found: {svg_frag}", file=sys.stderr)
        return False

    if verbose:
        print(f"── {cell_id} ──────────────────────────────────")

    # 1. Read fragment
    fragment = svg_frag.read_text(encoding="utf-8")
    if verbose:
        print(f"  fragment: {len(fragment)} chars")

    # 2. Extract shapes → path d-strings
    paths, (cw, ch) = extract_paths_from_fragment(fragment)
    if verbose:
        print(f"  shapes extracted: {len(paths)}  canvas: {cw}×{ch}")

    if not paths:
        print(f"  [WARN] No drawable shapes found in {cell_id}, skipping.",
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
        description="MSDF Pipeline — convert cell SVG fragments to distance field textures")
    parser.add_argument("cell_id",
        help="Cell ID (e.g. self_attn) or 'all' to process every cell")
    parser.add_argument("--size", default="64x32",
        help="Output texture size WxH in pixels (default: 64x32)")
    parser.add_argument("--pxrange", type=int, default=4,
        help="MSDF pixel range (default: 4)")
    parser.add_argument("--preview", action="store_true",
        help="Also generate a test-render preview PNG")
    args = parser.parse_args()

    # Parse size
    try:
        sw, sh = map(int, args.size.lower().split("x"))
    except ValueError:
        print(f"Invalid --size '{args.size}', expected WxH", file=sys.stderr)
        sys.exit(1)

    # Determine cells to process
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
