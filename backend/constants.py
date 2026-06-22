"""
SPH Rendering Constants
=======================
Shared constants for the Smoothed-Particle Hydrodynamics (SPH) rendering
pipeline.  Two main exports:

  SPH_RENDER_PASSES   — ordered list of render passes the compositor
                        executes each frame (mirrors the FXScene
                        composite pipeline in the TS renderer).

  SPECIES_SHADER_MAP  — per-species rendering descriptor.  Each entry
                        bundles three components that together define
                        how a cell species is drawn:
                          • sdf      — SDF primitive used for MSDF
                                       texture generation (backend)
                          • material — shader / material class name
                                       consumed by the WebGL renderer
                          • pattern  — surface pattern function applied
                                       inside the fragment shader

Cross-references:
  src/lib/renderer/material/CellMaterial.ts   (TS-side SPECIES_SHADER_MAP)
  src/lib/renderer/FXScene.ts                 (composite pipeline)
  backend/pipeline/msdf_gen.py                (SPECIES_GENERATORS)
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


# ---------------------------------------------------------------------------
# SPH_RENDER_PASSES — ordered execution list
# ---------------------------------------------------------------------------
# The compositor processes these front-to-back.  Each pass writes to its own
# FBO / RenderTarget; the Nuke post-process chain then composites them
# together with bloom, DOF, and tone-mapping.
#
# Ordering must match src/lib/renderer/FXScene.ts (RT0 → RT3 → post).
# ---------------------------------------------------------------------------

class RenderPassDef(TypedDict):
    """Schema for a single render pass definition."""
    name: str               # unique pass identifier
    order: int              # execution order (lower = earlier)
    fbo: str                # target FBO / RenderTarget name
    clear_color: List[float]  # RGBA clear colour [0..1]
    blend: str              # GL blend mode hint
    description: str        # human-readable purpose


SPH_RENDER_PASSES: List[RenderPassDef] = [
    {
        "name":        "background",
        "order":       0,
        "fbo":         "RT0_background",
        "clear_color": [0.0, 0.0, 0.0, 1.0],
        "blend":       "NONE",
        "description": "Scene background gradient / environment map.",
    },
    {
        "name":        "sph_density",
        "order":       10,
        "fbo":         "RT1_sph_density",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ADDITIVE",
        "description": "Accumulate SPH kernel density contributions per particle.",
    },
    {
        "name":        "sph_pressure",
        "order":       20,
        "fbo":         "RT2_sph_pressure",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ADDITIVE",
        "description": "Pressure solve — Tait equation on density field.",
    },
    {
        "name":        "sph_velocity",
        "order":       30,
        "fbo":         "RT3_sph_velocity",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ADDITIVE",
        "description": "Velocity integration (pressure + viscosity + external forces).",
    },
    {
        "name":        "cells_opaque",
        "order":       100,
        "fbo":         "RT4_cells",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "NONE",
        "description": "Opaque cell geometry — MSDF-based species shapes.",
    },
    {
        "name":        "cells_transparent",
        "order":       110,
        "fbo":         "RT4_cells",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ALPHA",
        "description": "Transparent / glass cell overlays (fresnel, refraction).",
    },
    {
        "name":        "edges",
        "order":       200,
        "fbo":         "RT5_edges",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ALPHA",
        "description": "Edge / connection lines between cells.",
    },
    {
        "name":        "ui_overlay",
        "order":       300,
        "fbo":         "RT6_ui",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ALPHA",
        "description": "Label text, badges, interaction highlights.",
    },
    {
        "name":        "bloom",
        "order":       900,
        "fbo":         "RT_post_bloom",
        "clear_color": [0.0, 0.0, 0.0, 0.0],
        "blend":       "ADDITIVE",
        "description": "Kawase dual-filter bloom (post-process).",
    },
    {
        "name":        "composite",
        "order":       999,
        "fbo":         "SCREEN",
        "clear_color": [0.0, 0.0, 0.0, 1.0],
        "blend":       "NONE",
        "description": "Final composite + tone-mapping → canvas.",
    },
]


# ---------------------------------------------------------------------------
# SPECIES_SHADER_MAP — per-species SDF + material + pattern descriptors
# ---------------------------------------------------------------------------
# 10 species matching the TS-side CellSpecies union type in CellMaterial.ts.
#
# Fields:
#   sdf       Signed-distance-field primitive used by msdf_gen.py to produce
#             the MSDF texture.  One of the generator function names in
#             SPECIES_GENERATORS or a parametric SDF family.
#   material  Shader / material class consumed by the WebGL renderer.
#             Must match a key in the TS SPECIES_SHADER_MAP or a registered
#             Material subclass.
#   pattern   Surface pattern function applied inside the fragment shader.
#             "none" means the material's base shading is used as-is.
# ---------------------------------------------------------------------------

class SpeciesShaderDef(TypedDict):
    """Schema for a single species rendering descriptor."""
    sdf: str        # SDF generator / primitive family
    material: str   # shader / material class name
    pattern: str    # surface pattern function


SPECIES_SHADER_MAP: Dict[str, SpeciesShaderDef] = {
    # ── Home scene ─────────────────────────────────────────────────────
    "cil-eye": {
        "sdf":      "concentric_rings",
        "material": "HomeParticleShader",
        "pattern":  "fresnel_glow",
    },
    # ── Work scene ─────────────────────────────────────────────────────
    "cil-bolt": {
        "sdf":      "zigzag_chain",
        "material": "ChainShader",
        "pattern":  "matcap_phong",
    },
    # ── About scene ────────────────────────────────────────────────────
    "cil-vector": {
        "sdf":      "crosshair_arms",
        "material": "AboutLogoShader",
        "pattern":  "matcap_normal",
    },
    # ── Contact / tree scene ───────────────────────────────────────────
    "cil-plus": {
        "sdf":      "plus_cross",
        "material": "HomeLogoShader",
        "pattern":  "matcap_normal",
    },
    # ── TreeScene navigation arrows ────────────────────────────────────
    "cil-arrow-right": {
        "sdf":      "chevron_arrow",
        "material": "TreeFBR",
        "pattern":  "directional_light",
    },
    # ── CleanRoom default ──────────────────────────────────────────────
    "cil-filter": {
        "sdf":      "rounded_rect",
        "material": "PBR",
        "pattern":  "none",
    },
    # ── Generic PBR code cell ──────────────────────────────────────────
    "cil-code": {
        "sdf":      "rounded_rect",
        "material": "ATPBR",
        "pattern":  "none",
    },
    # ── Floor geometry ─────────────────────────────────────────────────
    "cil-layers": {
        "sdf":      "stacked_rects",
        "material": "FloorShader",
        "pattern":  "mirror_reflection",
    },
    # ── Wall geometry ──────────────────────────────────────────────────
    "cil-loop": {
        "sdf":      "loop_ring",
        "material": "WallShader",
        "pattern":  "directional_light",
    },
    # ── Work-item card ─────────────────────────────────────────────────
    "cil-graph": {
        "sdf":      "rounded_rect",
        "material": "WorkItemShader",
        "pattern":  "glass_refraction",
    },
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_species_sdf(species: str) -> str:
    """Return the SDF primitive name for a species, or 'rounded_rect' fallback."""
    entry = SPECIES_SHADER_MAP.get(species)
    return entry["sdf"] if entry else "rounded_rect"


def get_species_material(species: str) -> str:
    """Return the material class name for a species, or 'PBR' fallback."""
    entry = SPECIES_SHADER_MAP.get(species)
    return entry["material"] if entry else "PBR"


def get_species_pattern(species: str) -> str:
    """Return the surface pattern for a species, or 'none' fallback."""
    entry = SPECIES_SHADER_MAP.get(species)
    return entry["pattern"] if entry else "none"


def render_passes_sorted() -> List[RenderPassDef]:
    """Return SPH_RENDER_PASSES sorted by execution order."""
    return sorted(SPH_RENDER_PASSES, key=lambda p: p["order"])
