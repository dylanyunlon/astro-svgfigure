import os, sys, json, math
from typing import Any, Optional
from channels.rendering.shading.shading_core import _SPECIES_CIL_EYE, _SPECIES_CIL_BOLT, _SPECIES_HYBRID

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)




# ---------------------------------------------------------------------------
# Internal geometry primitive
# ---------------------------------------------------------------------------

_SPECIES_NONE = 0xFFFF





class AstroCellDecoration:
    """
    Python equivalent of FAstroCellDecoration.

    Stores the per-cell decoration payload that is overlaid onto the
    ConstraintBuffer (SVG overlay layer in this 2-D port).

    Fields mirror the C++ struct; unused fields default to sentinel values
    identical to the default constructor.
    """
    __slots__ = (
        "species", "halo_intensity", "arc_seed",
        "blend_weight", "constraint_channel",
    )

    def __init__(self):
        self.species            = _SPECIES_NONE
        self.halo_intensity     = 1.0       # CilEye: [0.5, 1.0]
        self.arc_seed           = 0         # CilBolt: uint32 hash
        self.blend_weight       = 1.0       # overlay alpha [0, 1]
        self.constraint_channel = 2         # default: species-marker ch

















def _build_cell_decoration(species_name: str, gene_traits: dict,
                           cell_id: str) -> AstroCellDecoration:
    """
    Python equivalent of BuildCellDecorationFromDecal().

    Maps species name + gene_traits dict onto an AstroCellDecoration.
    gene_traits may carry "custom_data_0" (float [0,1] → encodes gene bits)
    and "custom_data_1" (float [0,1] → encodes halo intensity).
    Falls back gracefully when keys are absent.

    Species-name → gene-bit mapping:
        cil-eye    → CilEye  (1)
        cil-bolt   → CilBolt (2)
        cil-vector → Hybrid  (3)  ← both direction + halo traits
        cil-plus   → treated as CilEye with halo_intensity reduced (cross-hatch)
        others     → None    (0)  → default rounded-stroke decal
    """
    deco = AstroCellDecoration()

    # Decode gene bits from custom_data_0 if present (mirrors C++ int-round * 3)
    cd0 = float(gene_traits.get("custom_data_0", -1.0))
    cd1 = float(gene_traits.get("custom_data_1",  0.0))

    # Primary species routing via species name (canonical path)
    if species_name == "cil-eye":
        gene_bits = _SPECIES_CIL_EYE
    elif species_name == "cil-bolt":
        gene_bits = _SPECIES_CIL_BOLT
    elif species_name == "cil-vector":
        gene_bits = _SPECIES_HYBRID
    elif species_name == "cil-plus":
        # cil-plus uses the halo channel with reduced intensity → cross-hatch
        gene_bits = _SPECIES_CIL_EYE
        cd1 = 0.0   # force low halo → triggers cross-hatch variant below
    else:
        # Fallback: try to decode from custom_data_0 field; 0/absent → None
        if cd0 >= 0.0:
            gene_bits = int(round(cd0 * 3.0)) & 0x3
        else:
            gene_bits = _SPECIES_NONE

    deco.species = gene_bits

    if gene_bits == _SPECIES_CIL_EYE:
        # HaloIntensity remapped to [0.5, 1.0] — mirrors C++ case
        deco.halo_intensity     = 0.5 + max(0.0, min(1.0, cd1)) * 0.5
        deco.constraint_channel = 2   # species-marker channel

    elif gene_bits == _SPECIES_CIL_BOLT:
        # ArcSeed: stable per-cell hash from cell_id string
        # Mirrors C++ pointer-address hash >> 4; we use hash(cell_id) >> 4
        deco.arc_seed           = (abs(hash(cell_id)) >> 4) & 0xFFFFFFFF
        deco.constraint_channel = 1   # secondary-stress channel

    elif gene_bits == _SPECIES_HYBRID:
        deco.halo_intensity     = 0.5 + max(0.0, min(1.0, cd1)) * 0.5
        deco.arc_seed           = (abs(hash(cell_id)) >> 4) & 0xFFFFFFFF
        deco.constraint_channel = 2

    else:
        # None — no decoration written; BlendWeight zeroed (mirrors C++)
        deco.blend_weight = 0.0

    return deco

















def _apply_cell_decoration_overlay(deco: AstroCellDecoration,
                                   species_name: str,
                                   cx: float, cy: float,
                                   r_outer: float,
                                   bbox: dict) -> dict:
    """
    Python equivalent of ApplyCellDecorationToConstraintBuffer().

    返回结构化参数 dict，供前端 PixiJS DisplacementFilter / Graphics API 消费。
    不再拼接 SVG 字符串。

    DecoPayload channel mapping:
        R (HaloIntensity) → pupil-glint circle opacity / cross-hatch opacity
        G (ArcSeed/0xFFFF)→ arc flicker path opacity
        A (BlendWeight)   → overlay group opacity

    Returns a dict (may have type="none" if blend_weight <= 0 or species None).
    """
    if deco.blend_weight <= 0.0 or deco.species == _SPECIES_NONE:
        # nothing to write — mirrors early-return in C++
        return {"type": "none"}

    # Pack DecoPayload (R, G, 0, A) — canonical payload layout from C++
    r_channel = deco.halo_intensity                               # CilEye / Hybrid
    g_channel = (deco.arc_seed & 0xFFFF) / 65535.0               # CilBolt / Hybrid
    a_channel = max(0.0, min(1.0, deco.blend_weight))            # all species

    # Common metadata — mirrors PostProcessDeferredDecals payload header
    base: dict = {
        "species_index":      deco.species,
        "constraint_channel": deco.constraint_channel,
        "halo_intensity":     round(r_channel, 4),
        "arc_seed":           deco.arc_seed,
        "blend_weight":       round(a_channel, 4),
        "cx":                 round(cx, 2),
        "cy":                 round(cy, 2),
        "r_outer":            round(r_outer, 2),
    }

    if species_name == "cil-eye":
        # ── CilEye: additive pupil-glint circles ────────────────────────────
        # R channel = HaloIntensity → inner glint brightness
        # Mirrors: DecoPayload.R = Deco.HaloIntensity; ConstraintChannel=2
        glint_r  = r_outer * 0.12
        glint_cx = cx + r_outer * 0.28
        glint_cy = cy - r_outer * 0.28
        glint_op = max(0.4, r_channel * 0.9)
        return {
            **base,
            "type": "cil-eye",
            "primary_glint": {
                "cx": round(glint_cx, 2),
                "cy": round(glint_cy, 2),
                "r":  round(glint_r, 2),
                "fill": "white",
                "opacity": round(glint_op, 3),
            },
            "secondary_glint": {
                "cx": round(glint_cx + glint_r, 2),
                "cy": round(glint_cy + glint_r, 2),
                "r":  round(glint_r * 0.45, 2),
                "fill": "white",
                "opacity": round(glint_op * 0.5, 3),
            },
        }

    elif species_name == "cil-vector":
        # ── Hybrid: direction-arrow marker overlay ───────────────────────────
        # Both R (halo) + G (arc) channels active; ConstraintChannel=2
        # Mirrors: DecoPayload.R = HaloIntensity; DecoPayload.G = ArcSeed/0xFFFF
        arrow_len    = r_outer * 0.55
        angle_offset = (g_channel - 0.5) * 0.6   # ±0.3 rad variation
        angle        = angle_offset
        ax1 = cx - arrow_len * 0.5 * math.cos(angle)
        ay1 = cy - arrow_len * 0.5 * math.sin(angle)
        ax2 = cx + arrow_len * 0.5 * math.cos(angle)
        ay2 = cy + arrow_len * 0.5 * math.sin(angle)
        arrow_op    = max(0.35, r_channel * 0.7)
        marker_id   = int(deco.arc_seed) % 9999
        return {
            **base,
            "type": "cil-vector",
            "g_channel": round(g_channel, 4),
            "arrow": {
                "x1": round(ax1, 2), "y1": round(ay1, 2),
                "x2": round(ax2, 2), "y2": round(ay2, 2),
                "stroke": "#2E7D32",
                "stroke_width": 1.5,
                "opacity": round(arrow_op, 3),
                "marker_id": marker_id,
            },
            "arrow_marker": {
                "id": marker_id,
                "fill": "#2E7D32",
                "opacity": round(arrow_op, 3),
            },
        }

    elif species_name == "cil-bolt":
        # ── CilBolt: lightning highlight path ────────────────────────────────
        # G channel = ArcSeed/0xFFFF → arc flicker opacity; ConstraintChannel=1
        # Mirrors: DecoPayload.G = ArcSeed & 0xFFFF / 65535.f
        arc_op = max(0.3, g_channel * 0.85)
        points = [
            (round(cx - r_outer * 0.15, 2), round(cy - r_outer * 0.45, 2)),
            (round(cx + r_outer * 0.10, 2), round(cy - r_outer * 0.05, 2)),
            (round(cx - r_outer * 0.08, 2), round(cy + r_outer * 0.08, 2)),
            (round(cx + r_outer * 0.14, 2), round(cy + r_outer * 0.42, 2)),
        ]
        return {
            **base,
            "type": "cil-bolt",
            "g_channel": round(g_channel, 4),
            "bolt_path": {
                "points": points,
                "stroke": "white",
                "stroke_width": 1.8,
                "opacity": round(arc_op, 3),
            },
            "bolt_glow": {
                "points": points,
                "stroke": "#FFE0B2",
                "stroke_width": 3.5,
                "opacity": round(arc_op * 0.35, 3),
            },
        }

    elif species_name == "cil-plus":
        # ── CilPlus: cross-hatch shadow lines ────────────────────────────────
        # R channel (halo_intensity, forced low) → cross-hatch line opacity
        hatch_op = max(0.12, r_channel * 0.4)
        arm      = r_outer * 0.55
        spacing  = r_outer * 0.18
        offsets  = [-spacing, 0.0, spacing]
        h_lines  = [
            {"x1": round(cx - arm, 2), "y1": round(cy + off, 2),
             "x2": round(cx + arm, 2), "y2": round(cy + off, 2)}
            for off in offsets
        ]
        v_lines  = [
            {"x1": round(cx + off, 2), "y1": round(cy - arm, 2),
             "x2": round(cx + off, 2), "y2": round(cy + arm, 2)}
            for off in offsets
        ]
        return {
            **base,
            "type": "cil-plus",
            "hatch_opacity": round(hatch_op, 3),
            "stroke": "#C62828",
            "stroke_width": 0.7,
            "stroke_dasharray": "4,3",
            "h_lines": h_lines,
            "v_lines": v_lines,
        }

    else:
        # ── Default / None-species: generic rounded-stroke decoration ────────
        # BlendWeight > 0 but species=None → minimal constraint marker
        pad       = 3.0
        stroke_op = max(0.15, a_channel * 0.5)
        return {
            **base,
            "type": "generic",
            "rect": {
                "x":      round(pad, 1),
                "y":      round(pad, 1),
                "width":  round(bbox["w"] - pad * 2, 1),
                "height": round(bbox["h"] - pad * 2, 1),
                "rx": 10,
                "ry": 10,
                "fill": "none",
                "stroke": "#90A4AE",
                "stroke_width": 1.2,
                "stroke_dasharray": "6,4",
                "opacity": round(stroke_op, 3),
            },
        }


CHANNELS = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ReflectionEnvironment → Python port
#
# Ported from commit 5d07a0a:
#   upstream/unreal-renderer/ReflectionEnvironment.cpp
#
# FAstroCellStyleProbe (→ AstroCellStyleProbe):
#   Per-frame probe that samples the six cardinal neighbour cells' published
#   status channels, collects their species indices + representative SVG
#   palette colours, and elects the dominant species by majority vote.
#   BlendWithCubemap() → blend_toward_neighbour_palette(): given the cell's
#   own SVG stroke/fill colour, lerp it 20 % toward the neighbour palette
#   average so the cell visually converges toward its surroundings — the
#   "style consistency" guarantee from the C++ probe system, translated to the
#   SVG substrate.  Roughness (0 = smooth, 1 = rough) maps to how strongly the
#   cell should resist neighbour influence: smooth cells (low roughness) pull
#   harder; rough cells keep more of their own character.
#
# GAstroCellRegistrySnapshot (→ cell_registry.json + per-cell status.json):
#   In C++ the snapshot is a value-copy written by the game thread before the
#   render pass.  Here the pub/sub equivalent is the set of already-published
#   cell/*/status.json + cell/*/bbox.json files in the channels directory.
#   The probe reads whatever is on disk at proc() call time — i.e. whatever
#   neighbours have already published in this epoch (or the previous one if a
#   neighbour hasn't run yet).
#
# SampleSurroundingCells():
#   Walks the six cardinal neighbours in 2-D grid space (±1 grid step in X
#   and Y; ±1 z-layer step).  Grid step = floor(cell_width) for X/Y, 1 for Z.
#   For each neighbour that has a published status.json, reads species_index
#   (integer derived from the species name), representative colour (the
#   primary SVG fill colour for that species), and accumulates into the probe.
#
# BuildAstroCellStyleProbes():
#   Builds one probe per active cell in the registry, called from proc()
#   immediately before SVG parameter finalisation so adjustments flow into the
#   colour attributes written into the SVG string.
#
# 2-D channel adaptation:
#   CubemapArray probe  → per-cell style probe keyed by cell_id
#   WorldPosition       → bbox (x, y, z) of the cell
#   InfluenceRadius     → max(w, h) of the cell bbox
#   DominantSpeciesIndex→ int derived from species name string
#   Palette[]           → list of (r, g, b) tuples from neighbour cells
#   CellStyleWeight     → _STYLE_PROBE_WEIGHT global (default 0.20 = 20 %)
#   BlendWithCubemap    → blend_toward_neighbour_palette() below
# ═══════════════════════════════════════════════════════════════════════════════

# Blend weight: fraction by which this cell drifts toward the neighbour palette.
# 0.0 = pure own style, 1.0 = fully adopt neighbour palette.
# 鲁迅式 20 % drift — enough to feel the crowd without losing oneself.
_STYLE_PROBE_WEIGHT: float = 0.20

# Species name → canonical integer index (mirrors EAstroCellSpecies + extras).
# Used for majority-vote tallying, same as SpeciesVotes[256] in C++.
_SPECIES_NAME_TO_INDEX: dict = {
    "cil-eye":         1,
    "cil-bolt":        2,
    "cil-vector":      3,
    "cil-plus":        4,
    "cil-arrow-right": 5,
    # ── New species ──────────────────────────────────────────────────────────
    "cil-filter":      6,
    "cil-code":        7,
    "cil-layers":      8,
    "cil-loop":        9,
    "cil-graph":       10,
}

# Species index → primary SVG fill colour (RGB 0-255 tuple).
# Mirrors RepresentativeColour in FAstroCellRegistry::FCellEntry; values
# derived from the fill colours used in the generate_svg_* functions below.
_SPECIES_INDEX_TO_COLOUR: dict = {
    1: (63,  81, 181),   # cil-eye         → #3F51B5 Indigo
    2: (255, 111,   0),  # cil-bolt        → #FF6F00 Amber
    3: (46,  125,  50),  # cil-vector      → #2E7D32 Green
    4: (30,  136, 229),  # cil-plus        → #1E88E5 Blue
    5: (69,   90, 100),  # cil-arrow-right → #455A64 Blue-Grey
    # ── New species ──────────────────────────────────────────────────────────
    6: (123,  31, 162),  # cil-filter  → #7B1FA2 Purple
    7: (46,  125,  50),  # cil-code    → #2E7D32 Green (monospace feel)
    8: (21,  101, 192),  # cil-layers  → #1565C0 Blue (depth stack)
    9: (245, 127,  23),  # cil-loop    → #F57F17 Amber-Orange (cycle)
   10: (55,   71,  79),  # cil-graph   → #37474F Blue-Grey (graph nodes)
    0: (144, 164, 174),  # unassigned  → #90A4AE neutral
}








