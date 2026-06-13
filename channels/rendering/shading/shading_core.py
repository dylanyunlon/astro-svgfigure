import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def _ggx_smith_g1(cos_v: float, alpha: float) -> float:
    """
    GGX Smith masking function G1(v, alpha) — analytic, no LUT.

    Mirrors the BRDF_GGX_SmithCorrelated term integrated over the hemisphere
    in the C++ BuildEnergyTable compute shader (EEnergyTableType::GGXSpecular).

    G1(v,α) = 2 / (1 + sqrt(1 + α²·(1−cos²v)/cos²v))
    """
    alpha2 = alpha * alpha
    cos2   = max(cos_v * cos_v, 1e-8)
    tan2   = (1.0 - cos2) / cos2
    return 2.0 / (1.0 + math.sqrt(1.0 + alpha2 * tan2))









def _ggx_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the GGX directional albedo E(v, roughness).

    Replaces the GGXSpecEnergyTexture 2-D LUT lookup:
        E(v,r) = ∫ D_GGX * G_Smith * (4*cos_l*cos_v)^-1 d_omega_i

    Karis (2013) polynomial fit + Smith G1 correction:
        E ≈ G1(cos_v, α) * (1 − 0.28 * α²)   α = roughness²

    Returns E ∈ [0, 1]; higher roughness → more energy in the specular lobe.
    """
    alpha = roughness * roughness
    g1    = _ggx_smith_g1(cos_v, alpha)
    # Polynomial residual correction — calibrated against the tabulated
    # SHADING_ENERGY_CONSERVATION_TABLE_RESOLUTION=32 data in the C++ source.
    residual = 1.0 - 0.28 * alpha * alpha
    return max(0.0, min(1.0, g1 * residual))









def _cloth_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the Ashikhmin sheen directional albedo.

    Replaces the ClothEnergyTexture LUT (EEnergyTableType::Cloth).
    The Ashikhmin sheen BRDF has its peak at grazing angles; the directional
    albedo is approximated by a sinusoidal rolloff matching the C++ furnace data:

        E_cloth(v,r) ≈ (1 − cos_v)^(2·r) · (1 − 0.5·r)

    Returns E ∈ [0, 1].
    """
    exponent = max(0.5, 2.0 * roughness)
    edge_peak = math.pow(max(0.0, 1.0 - cos_v), exponent)
    return max(0.0, min(1.0, edge_peak * (1.0 - 0.5 * roughness)))









def _diffuse_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the Lambertian diffuse directional albedo.

    Replaces the DiffuseEnergyTexture LUT (EEnergyTableType::Diffuse).
    Pure Lambertian diffuse → E_d = 1 − E_spec (energy preservation);
    the C++ table stores the complement so the specular lobe's taken energy
    is removed from diffuse.  Here we return the raw diffuse albedo before
    specular subtraction (preservation is applied separately in
    apply_shading_energy_conservation).

        E_diff(v,r) ≈ 1.0 − roughness · (1 − cos_v)²

    Returns E ∈ [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - roughness * (1.0 - cos_v) ** 2))









def _multi_scatter_compensation(e_avg: float) -> float:
    """
    Multi-scattering energy compensation factor.

    Replaces the GGXMaxSpecEnergyTexture LUT that the C++ path samples to
    recover energy lost by single-scattering BRDF truncation.

    The compensation is the geometric series sum for multiple inter-surface
    bounces (Turquin 2019, eq. 9):
        f_ms = (1 − E_avg) / (1 − E_avg)  →  1 / (1 − E_avg)   [simplified]

    In practice the factor is clamped to avoid blowup at E_avg → 1:
        compensation = 1 / max(1 − E_avg, 0.01)

    Returns compensation ∈ [1, 100].
    """
    return 1.0 / max(1.0 - e_avg, 0.01)









def _fresnel_schlick_edge(cos_v: float, f0: float) -> float:
    """
    Schlick Fresnel approximation for edge-glow compensation.

    Replaces the Fresnel energy loss computation in ShadingEnergyConservation.h:
        F(v) = F0 + (1 − F0) · (1 − cos_v)^5

    Used to compute how much specular energy is added back to stroke opacity
    at grazing angles (Fresnel energy loss compensation — the cell edge should
    brighten when viewed at glancing incidence, same as the UE5 path that
    recovers Fresnel energy from the specular directional albedo).

    f0 is the species-derived reflectance at normal incidence (F0 ∈ [0, 1]).
    """
    return f0 + (1.0 - f0) * math.pow(max(0.0, 1.0 - cos_v), 5.0)









def apply_shading_energy_conservation(
    species:        str,
    roughness:      float,
    fill_opacity:   float,
    stroke_opacity: float,
    shadow_opacity: float,
    cos_view:       float = _SEC_COS_VIEW,
) -> tuple:
    """
    Energy-conserving opacity adjustment for the three SVG light channels.

    Port of ShadingEnergyConservation::Init + GetData applied at shading time:

    Step 1 — Compute per-lobe directional albedos (replaces LUT texture reads):
        E_spec  = GGX directional albedo        (GGXSpecEnergyTexture analogue)
        E_cloth = Ashikhmin sheen albedo        (ClothEnergyTexture analogue)
        E_diff  = Lambertian diffuse albedo     (DiffuseEnergyTexture analogue)
        E_ms    = multi-scatter compensation    (GGXMaxSpecEnergyTexture analogue)

    Step 2 — Fresnel edge-glow compensation:
        F       = Schlick Fresnel at cos_view angle
        stroke_opacity += (F − F0) · (1 − E_spec) · compensation

    Step 3 — Energy preservation (r.Shading.EnergyPreservation):
        fill_opacity *= (1 − E_spec)   ← specular steals energy from diffuse

    Step 4 — Energy conservation (r.Shading.EnergyConservation):
        total = fill_opacity·w_fill + stroke_opacity·w_stroke + shadow_opacity·w_shadow
        if total > 1.0: scale each channel down proportionally.

    Step 5 — Clamp to floor.

    @param species        Cell species name (determines F0, lobe shape)
    @param roughness      Visual roughness [0, 1]
    @param fill_opacity   Diffuse bulk opacity (AO-derived crowding_opacity)
    @param stroke_opacity SVG stroke opacity (default per species)
    @param shadow_opacity feDropShadow flood-opacity (capsule shadow)
    @param cos_view       Cosine of representative view angle (default 45°)
    @return               (fill_opacity, stroke_opacity, shadow_opacity) tuple,
                          all clamped to [_SEC_OPACITY_FLOOR, 1.0]
    """
    roughness = max(0.0, min(1.0, roughness))
    f0        = _species_f0(species)

    # ── Step 1: analytic directional albedos (LUT → formula) ─────────────────
    e_spec  = _ggx_directional_albedo(cos_view, roughness)
    e_cloth = _cloth_directional_albedo(cos_view, roughness)
    e_diff  = _diffuse_directional_albedo(cos_view, roughness)

    # Average albedo for multi-scatter: blend spec + cloth weighted by roughness.
    # Mirrors the blended EAverage used in the C++ GGXMaxSpecEnergy pass.
    e_avg = e_spec * (1.0 - roughness) + e_cloth * roughness
    comp  = _multi_scatter_compensation(e_avg)

    # ── Step 2: Fresnel edge-glow compensation ────────────────────────────────
    # Energy lost to Fresnel at the view angle is partially recovered by
    # brightening the stroke (edge highlight) — same as the C++ path that adds
    # (F − F0) * (1 − E_spec) * compensation into the specular term.
    fresnel        = _fresnel_schlick_edge(cos_view, f0)
    fresnel_excess = max(0.0, fresnel - f0)
    edge_boost     = fresnel_excess * max(0.0, 1.0 - e_spec) * comp

    # Clamp boost so stroke stays in [0,1] before conservation.
    stroke_opacity = max(0.0, min(1.0, stroke_opacity + edge_boost))

    # ── Step 3: energy preservation — specular steals from diffuse ────────────
    # fill *= (1 − E_spec) mirrors DiffuseColor *= (1 − SpecularEnergy) in C++.
    # e_diff provides the diffuse directional albedo correction on top.
    preservation_factor = max(0.0, 1.0 - e_spec) * e_diff
    fill_opacity = fill_opacity * preservation_factor

    # ── Step 4: energy conservation — clamp total weighted energy ────────────
    total = (fill_opacity   * _SEC_WEIGHT_FILL   +
             stroke_opacity * _SEC_WEIGHT_STROKE +
             shadow_opacity * _SEC_WEIGHT_SHADOW)

    if total > 1.0:
        inv_total   = 1.0 / total
        fill_opacity   *= inv_total
        stroke_opacity *= inv_total
        shadow_opacity *= inv_total

    # ── Step 5: floor clamp ───────────────────────────────────────────────────
    fill_opacity   = max(_SEC_OPACITY_FLOOR, fill_opacity)
    stroke_opacity = max(_SEC_OPACITY_FLOOR, stroke_opacity)
    shadow_opacity = max(_SEC_OPACITY_FLOOR, shadow_opacity)

    return (
        round(fill_opacity,   4),
        round(stroke_opacity, 4),
        round(shadow_opacity, 4),
    )









def compute_cell_energy_balance(
    cell_id:  str,
    species:  str,
    bbox:     dict,
    fill_opacity:   float,
    stroke_opacity: float,
    shadow_opacity: float,
    roughness:      float,
) -> dict:
    """
    Top-level energy balance entry point — wraps apply_shading_energy_conservation.

    Called from proc() after crowding_opacity (AO) and shadow params are known,
    before the SVG <g> wrapper is assembled.  Mirrors the call site in the C++
    pipeline:
        ShadingEnergyConservation::GetData(View)   → per-frame LUT handles
        then per-cell BRDF eval reads those handles → adjusted opacities

    Returns a dict with conserved opacities and diagnostic fields.
    """
    f_out, s_out, sh_out = apply_shading_energy_conservation(
        species        = species,
        roughness      = roughness,
        fill_opacity   = fill_opacity,
        stroke_opacity = stroke_opacity,
        shadow_opacity = shadow_opacity,
    )

    e_spec  = _ggx_directional_albedo(_SEC_COS_VIEW, roughness * roughness)
    e_avg   = e_spec
    comp    = _multi_scatter_compensation(e_avg)
    f0      = _species_f0(species)

    total_in  = (fill_opacity   * _SEC_WEIGHT_FILL   +
                 stroke_opacity * _SEC_WEIGHT_STROKE +
                 shadow_opacity * _SEC_WEIGHT_SHADOW)
    total_out = (f_out   * _SEC_WEIGHT_FILL   +
                 s_out   * _SEC_WEIGHT_STROKE +
                 sh_out  * _SEC_WEIGHT_SHADOW)

    dbg = os.environ.get("ASTRO_SEC_VERBOSE", "0") == "1"
    if dbg:
        print(
            f"[ASTRO-SEC] ShadingEnergyConservation cell={cell_id} "
            f"species={species} roughness={roughness:.3f} f0={f0:.3f} "
            f"E_spec={e_spec:.3f} comp={comp:.3f} "
            f"fill: {fill_opacity:.4f}→{f_out:.4f} "
            f"stroke: {stroke_opacity:.4f}→{s_out:.4f} "
            f"shadow: {shadow_opacity:.4f}→{sh_out:.4f} "
            f"total: {total_in:.4f}→{total_out:.4f}",
            file=sys.stderr,
        )

    return {
        "fill_opacity":   f_out,
        "stroke_opacity": s_out,
        "shadow_opacity": sh_out,
        "e_spec":         round(e_spec, 4),
        "multi_scatter":  round(comp,   4),
        "total_energy":   round(total_out, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PostProcessDeferredDecals → Python port
#
# Ported from commit 80cc569:
#   upstream/unreal-renderer/CompositionLighting/PostProcessDeferredDecals.cpp
#
# FAstroCellDecoration:
#   Per-cell decoration record written into the ConstraintBuffer during the
#   deferred pass.  Each entry carries species gene flags + payload fields
#   (HaloIntensity for CilEye, ArcSeed for CilBolt, BlendWeight for all).
#
# EAstroCellSpecies gene encoding (CustomData0 bits [0..1]):
#   None    = 0  → no decoration, BlendWeight forced to 0
#   CilEye  = 1  → halo ring overlay; HaloIntensity = 0.5 + CustomData1*0.5
#   CilBolt = 2  → electric arc overlay; ArcSeed from pointer hash
#   Hybrid  = 3  → both traits; maps to cil-vector (direction + intensity)
#
# BuildCellDecorationFromDecal:
#   Derives FAstroCellDecoration from species + gene_traits dict.
#   gene_traits["custom_data_0"] encodes species; ["custom_data_1"] encodes
#   halo intensity.  ConstraintChannel assignment mirrors the C++ switch.
#
# ApplyCellDecorationToConstraintBuffer:
#   Packs payload as DecoPayload(R, G, 0, A):
#     R = HaloIntensity   (CilEye / Hybrid)
#     G = ArcSeed/0xFFFF  (CilBolt / Hybrid)
#     A = BlendWeight
#   In SVG: R→opacity of the eye-highlight circle, G→arc path opacity,
#   A→overall overlay group opacity — same channel semantics, SVG substrate.
#
# SVG decal overlay pattern (species → ConstraintChannel → SVG element):
#   CilEye  (ch 2, species-marker)  → additive pupil-glint circle
#   CilBolt (ch 1, secondary-stress)→ lightning highlight path
#   Hybrid  (ch 2)                  → arrow marker (direction + halo)
#   CilPlus (derived)               → cross-hatch shadow lines
#   None/default                    → rounded stroke decoration rect
# ═══════════════════════════════════════════════════════════════════════════════

# Species gene constants (mirror EAstroCellSpecies enum)
_SPECIES_NONE    = 0
_SPECIES_CIL_EYE = 1
_SPECIES_CIL_BOLT = 2
_SPECIES_HYBRID   = 3









def _ggx_specular(n: _Vec3, v: _Vec3, l: _Vec3, roughness: float) -> float:
    """Isotropic GGX BRDF (D*G term only; F=1 for brevity)."""
    a  = max(roughness * roughness, 1e-4)
    hx = v.x+l.x; hy = v.y+l.y; hz = v.z+l.z
    hl = _math.sqrt(hx*hx + hy*hy + hz*hz)
    if hl < 1e-9:
        return 0.0
    h = _Vec3(hx/hl, hy/hl, hz/hl)
    ndoth = _saturate(n.dot(h))
    ndotl = _saturate(n.dot(l))
    ndotv = _saturate(n.dot(v))
    denom = ndoth*ndoth*(a*a-1.0)+1.0
    d = a*a / (_math.pi * denom*denom + 1e-9)
    # Smith G1 (Schlick approximation)
    k = a / 2.0
    gv = ndotv / (ndotv*(1-k)+k+1e-9)
    gl = ndotl / (ndotl*(1-k)+k+1e-9)
    return d * gv * gl * ndotl




