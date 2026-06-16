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
# Species metadata: color, F0, roughness per species
# (Replaces SPECIES_GENERATORS — SVG generation moved to renderer layer)

SPECIES_METADATA = {
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
}
