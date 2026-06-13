"""
Shared rendering constants — all rendering submodules import from here.
Values inferred from usage context in UE5 upstream port.
"""

# ── Shadow / Distance Field ──────────────────────────────────────────────────
_DF_SHADOW_QUALITY        = 2         # 0=off 1=low 2=medium 3=high
_DF_FULL_RESOLUTION       = False     # half-res for performance
_DF_TWO_SIDED_BIAS        = 3.0       # px — avoids thin-object artifacts
_DF_QUALITY_STEPS         = {0: 0, 1: 32, 2: 64, 3: 128}
_CSM_DEPTH_BIAS           = 0.5       # constant bias in z-units
_CSM_SLOPE_BIAS           = 0.7       # slope-dependent bias
_CSM_RECEIVER_BIAS        = 0.1       # receiver surface bias
_PCSS_MAX_KERNEL_RADIUS   = 12.0      # px — max penumbra blur
_SHADOW_FILTER_METHOD     = 1         # 0=hard 1=PCF 2=PCSS
_SHADOW_TRANSITION_SCALE  = 0.02      # cascade transition fade width
_STENCIL_OPTIMIZATION     = True      # use stencil to skip shadowed fragments
_CAPSULE_MAX_DIST         = 8.0       # max z-distance for capsule shadow influence

# ── Occlusion / Crowding ─────────────────────────────────────────────────────
_CROWDING_THRESHOLD       = 0.35      # fraction above which crowding kicks in
_CROWDING_OPACITY_FLOOR   = 0.25      # min opacity in dense regions
_REFERENCE_AREA           = 6400.0    # normalisation area (80*80)
_ATTENUATION_CURVE        = 1.5       # gamma for crowding attenuation

# ── Cell Registry / Z-Layer ──────────────────────────────────────────────────
_ASTRO_BBOX_TOLERANCE     = 0.01      # bbox-change detection threshold
_ASTRO_CELL_MAX_Z_LAYERS  = 10        # max z-layer depth
_ASTRO_CELL_Z_LAYER_HEIGHT = 50.0     # px per z-layer
_CELL_REGISTRY_PATH       = "physics/cell_registry.json"

# ── Style Probe ──────────────────────────────────────────────────────────────
_STYLE_PROBE_WEIGHT       = 0.20      # neighbour palette blend weight
_SPECIES_LOCALITY_WEIGHT  = 0.15      # same-species affinity bonus

# ── Species F0 (normal-incidence reflectance) ────────────────────────────────
_F0_TABLE = {
    "cil-eye":         0.04,
    "cil-bolt":        0.80,
    "cil-vector":      0.04,
    "cil-plus":        0.02,
    "cil-arrow-right": 0.06,
    "cil-filter":      0.65,
    "cil-code":        0.04,
    "cil-layers":      0.08,
    "cil-loop":        0.10,
    "cil-graph":       0.03,
}

def _species_f0(species: str) -> float:
    return _F0_TABLE.get(species, 0.04)
