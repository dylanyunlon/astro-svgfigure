import os, sys, json, math
from dataclasses import dataclass, field
from typing import Any, Optional
from channels.rendering.decoration.decoration_extra import _build_cell_decoration, _apply_cell_decoration_overlay, _SPECIES_NAME_TO_INDEX


def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# SpeciesParams — continuous, data-driven species definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesParams:
    """Data-driven species descriptor loaded from params.json.

    Every visual / behavioural property is a continuous parameter rather than
    a hardcoded enumeration, allowing the species palette to grow without
    code changes.
    """
    primary_color:   str   = "#888888"
    glow_color:      str   = "#AAAAAA"
    glow_intensity:  float = 0.04
    algorithm_gene:  str   = "radial"
    sdf_shape:       str   = "circle"
    animation_speed: float = 0.5
    corner_radius:   float = 2.0
    opacity:         float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# params.json loader
# ═══════════════════════════════════════════════════════════════════════════════

_PARAMS_JSON = os.path.join(os.path.dirname(__file__), "params.json")

_DATACLASS_FIELDS = {f.name for f in SpeciesParams.__dataclass_fields__.values()}


def _load_params_json(path: str = _PARAMS_JSON) -> list[dict]:
    """Read the species params.json file and return the raw list of dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"params.json must be a JSON array, got {type(data).__name__}")
    return data


def create_species_from_params(path: str = _PARAMS_JSON) -> dict[str, SpeciesParams]:
    """Build a ``{species_name: SpeciesParams}`` mapping from *params.json*.

    Only the eight dataclass fields are forwarded to the constructor; extra
    keys in the JSON (``color``, ``bg_color``, ``f0``, …) are preserved in
    the raw registry but not in the dataclass itself.
    """
    result: dict[str, SpeciesParams] = {}
    for entry in _load_params_json(path):
        name = entry.get("species")
        if not name:
            continue
        dc_kwargs = {k: v for k, v in entry.items() if k in _DATACLASS_FIELDS}
        result[name] = SpeciesParams(**dc_kwargs)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compatible module-level dicts rebuilt from params.json
# ═══════════════════════════════════════════════════════════════════════════════
#
# Other modules import SPECIES_METADATA, SPECIES_PALETTE,
# SPECIES_ALGORITHM_GENE, species_registry, _species_f0, and
# _species_to_index.  We reconstruct them at import time so downstream
# code is unaffected.

def _rebuild_legacy_dicts(path: str = _PARAMS_JSON):
    """Parse params.json once and emit the four legacy dicts."""
    raw = _load_params_json(path)

    metadata   : dict[str, dict[str, Any]] = {}
    palette    : dict[str, dict[str, str]] = {}
    algo_gene  : dict[str, str]            = {}
    registry   : dict[str, dict[str, Any]] = {}

    for entry in raw:
        name = entry.get("species")
        if not name:
            continue

        # SPECIES_METADATA
        metadata[name] = {
            "color":     entry.get("color",    entry.get("primary_color", "#888888")),
            "bg_color":  entry.get("bg_color", "#FFFFFF"),
            "f0":        float(entry.get("f0", entry.get("glow_intensity", 0.04))),
            "roughness": float(entry.get("roughness", 0.50)),
        }

        # SPECIES_PALETTE
        palette[name] = {
            "primary":   entry.get("primary_color",   "#888888"),
            "secondary": entry.get("secondary_color",  "#AAAAAA"),
            "glow":      entry.get("glow_color",       "#AAAAAA"),
            "shadow":    entry.get("shadow_color",      "#444444"),
        }

        # SPECIES_ALGORITHM_GENE
        algo_gene[name] = entry.get("algorithm_gene", "radial")

        # species_registry (unified)
        registry[name] = {
            **metadata[name],
            **palette[name],
            "algorithm_gene": algo_gene[name],
        }

    return metadata, palette, algo_gene, registry


SPECIES_METADATA, SPECIES_PALETTE, SPECIES_ALGORITHM_GENE, species_registry = \
    _rebuild_legacy_dicts()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (unchanged public API)
# ═══════════════════════════════════════════════════════════════════════════════

def _species_f0(species: str) -> float:
    """Per-species F0 (normal-incidence reflectance), read from params.json."""
    entry = SPECIES_METADATA.get(species)
    if entry is not None:
        return entry["f0"]
    return 0.04


def _species_to_index(species_name: str) -> int:
    """Map a species name string to its canonical integer index."""
    return _SPECIES_NAME_TO_INDEX.get(species_name, 0)
