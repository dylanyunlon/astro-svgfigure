import os, sys, json, math
from typing import Any, Optional
from channels.rendering.decoration.decoration_extra import _build_cell_decoration, _apply_cell_decoration_overlay, _SPECIES_NAME_TO_INDEX

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
        # ── Original 10 species ──────────────────────────────────────────────
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
        # ── Expanded species (11–22) ─────────────────────────────────────────
        "cil-wave":        0.12,   # subtle gloss — wave/signal propagation
        "cil-gate":        0.55,   # semi-metallic — gate/switch controller
        "cil-mem":         0.07,   # slight sheen — memory/storage cell
        "cil-norm":        0.35,   # mid-metallic — normalisation layer
        "cil-pool":        0.05,   # near-matte — pooling/aggregation
        "cil-conv":        0.45,   # mid-metallic — convolution kernel
        "cil-attn-cross":  0.30,   # mid — cross-attention head
        "cil-embed":       0.08,   # low gloss — embedding lookup
        "cil-loss":        0.70,   # high metallic — loss/objective surface
        "cil-dropout":     0.03,   # matte — stochastic mask
        "cil-residual":    0.15,   # low gloss — skip-connection bridge
        "cil-softmax":     0.20,   # slight metallic — probability simplex
    }
    return _F0_TABLE.get(species, 0.04)


def _species_to_index(species_name: str) -> int:
    """Map a species name string to its canonical integer index."""
    return _SPECIES_NAME_TO_INDEX.get(species_name, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Species palette: primary / secondary / glow / shadow per species
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each entry carries the four-colour palette that downstream renderers sample
# for fill, stroke-accent, bloom, and drop-shadow respectively.  The palette
# is intentionally richer than the old two-colour (color / bg_color) scheme
# so that 50+ heterogeneous cells can be visually distinguished at a glance.

SPECIES_PALETTE = {
    # ── Original 10 species ──────────────────────────────────────────────────
    "cil-eye":         {"primary": "#3F51B5", "secondary": "#7986CB", "glow": "#536DFE", "shadow": "#1A237E"},
    "cil-bolt":        {"primary": "#E65100", "secondary": "#FF9E80", "glow": "#FFAB40", "shadow": "#BF360C"},
    "cil-vector":      {"primary": "#2E7D32", "secondary": "#81C784", "glow": "#69F0AE", "shadow": "#1B5E20"},
    "cil-plus":        {"primary": "#C62828", "secondary": "#EF9A9A", "glow": "#FF5252", "shadow": "#B71C1C"},
    "cil-arrow-right": {"primary": "#455A64", "secondary": "#90A4AE", "glow": "#78909C", "shadow": "#263238"},
    "cil-filter":      {"primary": "#7B1FA2", "secondary": "#CE93D8", "glow": "#E040FB", "shadow": "#4A148C"},
    "cil-code":        {"primary": "#2E7D32", "secondary": "#A5D6A7", "glow": "#76FF03", "shadow": "#1B5E20"},
    "cil-layers":      {"primary": "#1565C0", "secondary": "#64B5F6", "glow": "#448AFF", "shadow": "#0D47A1"},
    "cil-loop":        {"primary": "#F57F17", "secondary": "#FFF176", "glow": "#FFD740", "shadow": "#F57F17"},
    "cil-graph":       {"primary": "#37474F", "secondary": "#78909C", "glow": "#607D8B", "shadow": "#263238"},
    # ── Expanded species ─────────────────────────────────────────────────────
    "cil-wave":        {"primary": "#7E57C2", "secondary": "#B39DDB", "glow": "#B388FF", "shadow": "#4527A0"},
    "cil-gate":        {"primary": "#EF6C00", "secondary": "#FFB74D", "glow": "#FFAB40", "shadow": "#E65100"},
    "cil-mem":         {"primary": "#546E7A", "secondary": "#90A4AE", "glow": "#78909C", "shadow": "#37474F"},
    "cil-norm":        {"primary": "#9E9E9E", "secondary": "#E0E0E0", "glow": "#CFD8DC", "shadow": "#616161"},
    "cil-pool":        {"primary": "#0D47A1", "secondary": "#42A5F5", "glow": "#2979FF", "shadow": "#01579B"},
    "cil-conv":        {"primary": "#D84315", "secondary": "#FF8A65", "glow": "#FF6E40", "shadow": "#BF360C"},
    "cil-attn-cross":  {"primary": "#4A148C", "secondary": "#9C27B0", "glow": "#EA80FC", "shadow": "#311B92"},
    "cil-embed":       {"primary": "#00838F", "secondary": "#4DD0E1", "glow": "#18FFFF", "shadow": "#006064"},
    "cil-loss":        {"primary": "#B71C1C", "secondary": "#E57373", "glow": "#FF1744", "shadow": "#7F0000"},
    "cil-dropout":     {"primary": "#6D4C41", "secondary": "#A1887F", "glow": "#8D6E63", "shadow": "#3E2723"},
    "cil-residual":    {"primary": "#00695C", "secondary": "#4DB6AC", "glow": "#64FFDA", "shadow": "#004D40"},
    "cil-softmax":     {"primary": "#AD1457", "secondary": "#F06292", "glow": "#FF4081", "shadow": "#880E4F"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm gene — the geometric pattern kernel each species uses for
# internal SVG motif generation (radial, linear, grid, scatter, …).
# ═══════════════════════════════════════════════════════════════════════════════

SPECIES_ALGORITHM_GENE = {
    # ── Original 10 species ──────────────────────────────────────────────────
    "cil-eye":         "radial",     # iris-like concentric rings
    "cil-bolt":        "linear",     # directional energy spike
    "cil-vector":      "linear",     # directional arrow glyph
    "cil-plus":        "grid",       # cross-hair / plus grid
    "cil-arrow-right": "linear",     # single-direction flow
    "cil-filter":      "grid",       # kernel weight matrix
    "cil-code":        "grid",       # monospace character grid
    "cil-layers":      "stack",      # horizontal layer bands
    "cil-loop":        "radial",     # cyclic arc / ring
    "cil-graph":       "scatter",    # node/edge scatter layout
    # ── Expanded species ─────────────────────────────────────────────────────
    "cil-wave":        "sinusoidal", # oscillating wave crests
    "cil-gate":        "linear",     # on/off binary bar
    "cil-mem":         "grid",       # memory cell matrix
    "cil-norm":        "linear",     # horizontal equalisation bar
    "cil-pool":        "radial",     # concentric shrink rings
    "cil-conv":        "grid",       # sliding kernel window
    "cil-attn-cross":  "scatter",    # query-key scatter pairs
    "cil-embed":       "grid",       # lookup table grid
    "cil-loss":        "radial",     # loss-landscape contour
    "cil-dropout":     "scatter",    # random mask dots
    "cil-residual":    "linear",     # skip-path arrow bridge
    "cil-softmax":     "radial",     # probability fan-out
}


# ═══════════════════════════════════════════════════════════════════════════════
# Species metadata: color, bg_color, F0, roughness per species
# (Backward-compatible with the original two-colour schema — renderers that
#  only read color / bg_color still work; new renderers can read the richer
#  SPECIES_PALETTE and SPECIES_ALGORITHM_GENE dicts.)
# ═══════════════════════════════════════════════════════════════════════════════

SPECIES_METADATA = {
    # ── Original 10 species (unchanged) ──────────────────────────────────────
    "cil-eye": {
        "color":     "#3F51B5",
        "bg_color":  "#E8EAF6",
        "f0":        0.04,
        "roughness": 0.80,
    },
    "cil-vector": {
        "color":     "#2E7D32",
        "bg_color":  "#E8F5E9",
        "f0":        0.04,
        "roughness": 0.75,
    },
    "cil-bolt": {
        "color":     "#E65100",
        "bg_color":  "#FFF3E0",
        "f0":        0.80,
        "roughness": 0.20,
    },
    "cil-plus": {
        "color":     "#C62828",
        "bg_color":  "#FCE4EC",
        "f0":        0.02,
        "roughness": 0.90,
    },
    "cil-arrow-right": {
        "color":     "#455A64",
        "bg_color":  "#ECEFF1",
        "f0":        0.06,
        "roughness": 0.65,
    },
    "cil-filter": {
        "color":     "#7B1FA2",
        "bg_color":  "#F3E5F5",
        "f0":        0.65,
        "roughness": 0.25,
    },
    "cil-code": {
        "color":     "#2E7D32",
        "bg_color":  "#E8F5E9",
        "f0":        0.04,
        "roughness": 0.78,
    },
    "cil-layers": {
        "color":     "#1565C0",
        "bg_color":  "#E3F2FD",
        "f0":        0.08,
        "roughness": 0.60,
    },
    "cil-loop": {
        "color":     "#F57F17",
        "bg_color":  "#FFF8E1",
        "f0":        0.10,
        "roughness": 0.55,
    },
    "cil-graph": {
        "color":     "#37474F",
        "bg_color":  "#FAFAFA",
        "f0":        0.03,
        "roughness": 0.92,
    },
    # ── Expanded species (11–22) ─────────────────────────────────────────────
    "cil-wave": {
        "color":     "#7E57C2",
        "bg_color":  "#EDE7F6",
        "f0":        0.12,
        "roughness": 0.55,
    },
    "cil-gate": {
        "color":     "#EF6C00",
        "bg_color":  "#FFF3E0",
        "f0":        0.55,
        "roughness": 0.30,
    },
    "cil-mem": {
        "color":     "#546E7A",
        "bg_color":  "#ECEFF1",
        "f0":        0.07,
        "roughness": 0.70,
    },
    "cil-norm": {
        "color":     "#9E9E9E",
        "bg_color":  "#F5F5F5",
        "f0":        0.35,
        "roughness": 0.40,
    },
    "cil-pool": {
        "color":     "#0D47A1",
        "bg_color":  "#E3F2FD",
        "f0":        0.05,
        "roughness": 0.82,
    },
    "cil-conv": {
        "color":     "#D84315",
        "bg_color":  "#FBE9E7",
        "f0":        0.45,
        "roughness": 0.30,
    },
    "cil-attn-cross": {
        "color":     "#4A148C",
        "bg_color":  "#F3E5F5",
        "f0":        0.30,
        "roughness": 0.35,
    },
    "cil-embed": {
        "color":     "#00838F",
        "bg_color":  "#E0F7FA",
        "f0":        0.08,
        "roughness": 0.72,
    },
    "cil-loss": {
        "color":     "#B71C1C",
        "bg_color":  "#FFEBEE",
        "f0":        0.70,
        "roughness": 0.22,
    },
    "cil-dropout": {
        "color":     "#6D4C41",
        "bg_color":  "#EFEBE9",
        "f0":        0.03,
        "roughness": 0.88,
    },
    "cil-residual": {
        "color":     "#00695C",
        "bg_color":  "#E0F2F1",
        "f0":        0.15,
        "roughness": 0.58,
    },
    "cil-softmax": {
        "color":     "#AD1457",
        "bg_color":  "#FCE4EC",
        "f0":        0.20,
        "roughness": 0.45,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# species_registry — unified lookup dict for other modules
# ═══════════════════════════════════════════════════════════════════════════════
#
# Merges SPECIES_METADATA, SPECIES_PALETTE, and SPECIES_ALGORITHM_GENE into a
# single dict keyed by species name.  Each value is a dict with:
#
#   color, bg_color, f0, roughness          (from SPECIES_METADATA)
#   primary, secondary, glow, shadow        (from SPECIES_PALETTE)
#   algorithm_gene                          (from SPECIES_ALGORITHM_GENE)
#
# Usage:
#   from channels.rendering.species.species_port import species_registry
#   info = species_registry["cil-conv"]
#   info["primary"]         # "#D84315"
#   info["algorithm_gene"]  # "grid"

species_registry: dict[str, dict[str, Any]] = {}
for _sp_name in SPECIES_METADATA:
    species_registry[_sp_name] = {
        **SPECIES_METADATA[_sp_name],
        **SPECIES_PALETTE.get(_sp_name, {}),
        "algorithm_gene": SPECIES_ALGORITHM_GENE.get(_sp_name, "radial"),
    }
