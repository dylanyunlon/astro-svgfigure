/**
 * cell-material-system.ts — M766: Per-Species Material System
 * ─────────────────────────────────────────────────────────────────────────────
 * Assigns a distinct visual material identity to each Transformer-analogy species
 * using a hybrid PBR + Fresnel + Matcap combination pipeline:
 *
 *   attention   → iridescent metallic   (thin-film interference, anisotropic shimmer)
 *   ffn         → glass refraction      (Snell-law caustic, depth-of-field haze)
 *   layernorm   → marble                (matcap stone veins, polished Fresnel rim)
 *   embedding   → membrane              (SSS wrap-lighting, oil-film translucency)
 *   softmax     → luminous energy       (HDR emissive core, coronal bloom)
 *
 * Pipeline architecture  (PBR + Fresnel + Matcap hybrid)
 * ─────────────────────────────────────────────────────────────────────────────
 * Every species runs through a three-stage fragment composition:
 *
 *   1. **PBR stage** — Cook-Torrance BRDF supplies base diffuse + specular.
 *      Even matcap-primary species (layernorm) receive a lightweight PBR
 *      ambient term so that environment lighting is physically consistent
 *      across all five species in a shared scene.
 *
 *   2. **Fresnel stage** — per-species Schlick Fresnel with configurable
 *      power and tint supplies the rim / edge contribution.  This is
 *      independent of the matcap Fresnel so that non-matcap species still
 *      get a correct grazing-angle highlight.
 *
 *   3. **Matcap stage** — species that declare `matcapWeight > 0` blend a
 *      matcap lookup onto the PBR+Fresnel composite.  The matcap texture
 *      coordinate is derived from the view-space normal, giving instant
 *      "studio lighting" without extra light probes.
 *
 * After the three stages, the species-specific WGSL patch runs to add
 * the signature visual effect (iridescence film, caustic stripe, marble
 * veins, membrane SSS, emissive bloom).  The patch receives the composite
 * colour as `pbrColor` and returns the final fragment colour before
 * Reinhard tone-mapping and gamma correction.
 *
 * Design principles
 * ─────────────────────────────────────────────────────────────────────────────
 * • Each SpeciesMaterialDef is self-contained: it carries both the CPU-side
 *   PBR/Matcap/Fresnel params (forwarded to at-pbr-material.ts uniform
 *   buffers) and a set of WGSL shader patches that implement the species-
 *   unique visual effect not covered by the base pipeline.
 * • The system is *additive*: PBR first, then Fresnel, then optional matcap
 *   blend, then species patch, then tone-mapping.
 * • Physics-driven modulation is expressed through `physicsModulators` — thin
 *   wrappers that map PhysicsUniforms channels onto material params at runtime
 *   (same semantics as PhysicsBindings in species-shader-registry.ts).
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/at-pbr-material.ts          — PBRParams, MatcapParams, packPBRUniforms
 *   src/lib/sph/species-shader-registry.ts  — SpeciesShaderConfig, PhysicsBinding
 *   src/lib/sph/physics-uniform-bridge.ts   — PhysicsUniforms
 *
 * Research: xiaodi #M766 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Species identifier type
// ─────────────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * The five conceptual species types managed by this system.
// [orphan-precise]  * Values mirror the Transformer-analogy roles used throughout the codebase.
// [orphan-precise]  */
=======
/**
 * The five conceptual species types managed by this system.
 * Values mirror the Transformer-analogy roles used throughout the codebase.
 */



import type { PBRParams, MatcapParams }           from './at-pbr-material';
import type { PhysicsBinding, MaterialType }       from './species-shader-registry';
import type { PhysicsUniforms }                    from './physics-uniform-bridge';
import { DEFAULT_PBR_PARAMS, DEFAULT_MATCAP_PARAMS } from './at-pbr-material';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export type CellSpecies =
  | 'attention'
  | 'ffn'
  | 'layernorm'
  | 'embedding'
  | 'softmax';

// ─────────────────────────────────────────────────────────────────────────────
// Fresnel configuration — per-species Schlick Fresnel independent of matcap
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-species Fresnel parameters for the dedicated Fresnel stage.
 * This is *separate* from any Fresnel in the matcap path and runs
 * unconditionally for all species, ensuring consistent edge highlights
 * across PBR-primary and matcap-primary materials.
 */
export interface FresnelConfig {
  /** Fresnel exponent (higher = tighter rim highlight). */
  power: number;

  /** RGB tint of the Fresnel rim highlight (linear-space). */
  tint: [number, number, number];

  /**
   * Intensity multiplier for the Fresnel contribution.
   * 0 = Fresnel stage is effectively off; 1 = full strength.
   */
  intensity: number;

  /**
   * Bias added to the dot(N, V) before the power curve.
   * Positive bias widens the highlight; negative narrows it.
   */
  bias: number;
}

/**
 * Per-species matcap blend configuration.
 * Species with `matcapWeight = 0` skip the matcap lookup entirely.
 */
export interface MatcapBlendConfig {
  /**
   * Blend weight for the matcap lookup against the PBR+Fresnel composite.
   * 0 = pure PBR+Fresnel; 1 = pure matcap (not recommended — loses PBR lighting).
   * Typical range: 0.0–0.6.
   */
  matcapWeight: number;

  /**
   * Blend mode for mixing matcap onto PBR+Fresnel.
   * 'lerp'     — standard linear blend (default)
   * 'multiply' — multiply-composites (darkens, good for stone/marble)
   * 'screen'   — screen-composites (brightens, good for luminous overlays)
   * 'overlay'  — Photoshop-style overlay (contrast-boosts mid-tones)
   */
  blendMode: 'lerp' | 'multiply' | 'screen' | 'overlay';
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fresnel composition stage (runs for every species)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL helper: the per-species Fresnel stage.
 * Injected before each species patch so that `applyFresnelStage` is available.
 */
export const WGSL_FRESNEL_STAGE = /* wgsl */`
// ── Per-species Fresnel composition stage ────────────────────────────────────
// Independent of matcap Fresnel; provides consistent rim highlights for all species.
fn applyFresnelStage(
    baseColor    : vec3f,
    N            : vec3f,
    V            : vec3f,
    fresnelPower : f32,
    fresnelTint  : vec3f,
    fresnelIntens: f32,
    fresnelBias  : f32
) -> vec3f {
    let cosTheta = saturate_f(dot(N, V));
    let biased   = saturate_f(cosTheta + fresnelBias);
    let F        = pow(1.0 - biased, fresnelPower);
    return baseColor + fresnelTint * F * fresnelIntens;
}
`;

/**
 * WGSL helper: matcap blend modes for the matcap composition stage.
 * Only evaluated when species matcapWeight > 0.
 */
export const WGSL_MATCAP_BLEND = /* wgsl */`
// ── Matcap blend composition ────────────────────────────────────────────────
fn blendMatcap_lerp(base: vec3f, matcap: vec3f, w: f32) -> vec3f {
    return mix(base, matcap, w);
}
fn blendMatcap_multiply(base: vec3f, matcap: vec3f, w: f32) -> vec3f {
    return mix(base, base * matcap, w);
}
fn blendMatcap_screen(base: vec3f, matcap: vec3f, w: f32) -> vec3f {
    let screened = vec3f(1.0) - (vec3f(1.0) - base) * (vec3f(1.0) - matcap);
    return mix(base, screened, w);
}
fn blendMatcap_overlay(base: vec3f, matcap: vec3f, w: f32) -> vec3f {
    // Per-channel overlay: if base < 0.5 multiply, else screen
    let lo = 2.0 * base * matcap;
    let hi = vec3f(1.0) - 2.0 * (vec3f(1.0) - base) * (vec3f(1.0) - matcap);
    let overlay = select(hi, lo, base < vec3f(0.5));
    return mix(base, overlay, w);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL patch snippets — per-species fragment-stage extensions
// Each snippet receives `pbrColor: vec3f` as input and is expected to return
// a modified `vec3f`.  Snippets are concatenated *after* the base PBR pass,
// Fresnel stage, and matcap blend, then *before* Reinhard tone-mapping.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * attention — iridescent metallic
 *
 * Adds a high-frequency thin-film interference sweep on top of the PBR base.
 * Two film layers (primary + secondary) with slightly different IOR and thickness
 * produce the characteristic shifting cyan-violet-gold shimmer of an attention head.
 * A time-varying wobble simulates the cell membrane breathing.
 *
 * PBR contribution: high-metallic mirror substrate provides specular base.
 * Fresnel contribution: cyan-tinted rim at grazing angles (IOR shimmer cue).
 * Matcap contribution: none (matcapWeight = 0) — iridescence is fully PBR-driven.
 */
const WGSL_PATCH_ATTENTION = /* wgsl */`
// ── Attention: layered iridescent metallic ────────────────────────────────────
fn applyAttentionMaterial(
    baseColor  : vec3f,
    N          : vec3f,
    V          : vec3f,
    uv         : vec2f,
    time       : f32,
    iridScale  : f32   // physics-driven thickness modulator
) -> vec3f {
    let cosI = saturate_f(dot(N, V));

    // Primary film: cyan-violet band (~420 nm, n=1.50)
    let t1   = 420.0 + sin(time * 0.38 + uv.x * PI * 3.1) * 55.0 * iridScale;
    let irid1 = iridescence(t1, 1.50, cosI);

    // Secondary film: gold-green overtone (~600 nm, n=1.42)
    let t2   = 600.0 + cos(time * 0.27 + uv.y * PI * 2.7) * 40.0 * iridScale;
    let irid2 = iridescence(t2, 1.42, cosI);

    // Tertiary micro-film: magenta accent at grazing angles (~510 nm, n=1.48)
    let t3   = 510.0 + sin(time * 0.51 + (uv.x + uv.y) * PI * 1.9) * 25.0 * iridScale;
    let irid3 = iridescence(t3, 1.48, cosI);

    // Three-layer blend: primary dominant, secondary warm overtone, tertiary edge accent
    let iridBlend = irid1 * 0.52 + irid2 * 0.30 + irid3 * 0.18;

    // Anisotropic metallic highlight along V-plane
    let aniso = pow(saturate_f(1.0 - cosI), 3.2) * 0.45;

    // Add iridescence additively; boost with metallic aniso rim
    var col = baseColor + iridBlend * 0.72;
    col     = col + vec3f(aniso * 0.6, aniso * 0.55, aniso * 0.9);

    return col;
}
`;

/**
 * ffn — glass refraction
 *
 * Simulates a glass-like surface using approximate Snell-law refraction offset and
 * a caustic-stripe highlight pattern.  The refraction UV shift is computed from the
 * normal map perturbation; the caustic flicker is driven by kinetic energy.
 *
 * PBR contribution: dielectric specular for glass surface (F0 ≈ 0.04).
 * Fresnel contribution: warm high-power rim mimicking total internal reflection.
 * Matcap contribution: none — glass is inherently view-dependent, not studio-lit.
 */
const WGSL_PATCH_FFN = /* wgsl */`
// ── FFN: glass refraction + caustic flicker ───────────────────────────────────
fn applyFFNMaterial(
    baseColor   : vec3f,
    N           : vec3f,
    V           : vec3f,
    uv          : vec2f,
    time        : f32,
    causticScale: f32   // physics-driven caustic brightness
) -> vec3f {
    // Schlick transmission approximation: F_T = 1 - F_R
    let F_R    = F_Schlick(vec3f(0.04), saturate_f(dot(N, V)));
    let transm = vec3f(1.0) - F_R;

    // Refraction tint (IOR=1.45, borosilicate-like: slight warm shift)
    let refractTint = vec3f(0.96, 0.98, 1.0);

    // Dual-frequency caustic: broad wave + sharp flicker
    let causticBroad  = uv.x * 8.0 + uv.y * 3.5 + time * 0.9;
    let causticSharp  = uv.x * 16.0 + uv.y * 7.0 + time * 1.8;
    let causticWave   = pow(abs(sin(causticBroad)), 12.0) * 0.35;
    let causticFlick  = pow(abs(sin(causticSharp)), 22.0) * 0.65;
    let causticTotal  = (causticWave + causticFlick) * causticScale;
    let causticColor  = vec3f(1.0, 0.98, 0.90) * causticTotal * 0.55;

    // Dispersion: slight RGB offset simulating chromatic aberration at edges
    let dispersion = pow(1.0 - saturate_f(dot(N, V)), 4.5) * 0.12;
    let dispersionShift = vec3f(-dispersion, 0.0, dispersion);

    // Blend: transmission tints base, caustics add as light transport
    var col = baseColor * transm * refractTint + dispersionShift;
    col    += causticColor;

    // Thin specular highlight on glass edge
    let edgeFresnel = fresnel_f(N, V, 6.0) * 0.30;
    col += vec3f(edgeFresnel);

    return col;
}
`;

/**
 * layernorm — matcap marble
 *
 * Blends a procedural marble vein pattern (based on layered simplex noise) over
 * the matcap result, giving a smooth normalised surface the visual weight of
 * carved stone.  The veins are oriented along the local tangent plane.
 *
 * PBR contribution: ambient diffuse term provides scene-consistent base lighting.
 * Fresnel contribution: soft cool-grey rim (polished stone catch-light).
 * Matcap contribution: HIGH (matcapWeight = 0.55) — marble uses matcap for
 *   studio-style specular highlight that imitates polished stone under soft box light.
 *   Blend mode = 'multiply' to darken veins naturally within the matcap silhouette.
 */
const WGSL_PATCH_LAYERNORM = /* wgsl */`
// ── LayerNorm: matcap marble veins ────────────────────────────────────────────
fn applyLayerNormMaterial(
    baseColor   : vec3f,
    N           : vec3f,
    worldPos    : vec3f,
    time        : f32,
    veinContrast: f32   // physics-driven vein sharpness
) -> vec3f {
    // Three octaves of layered noise for complex marble veining
    let p0  = worldPos * 1.8 + vec3f(0.0, 0.0, time * 0.04);
    let p1  = worldPos * 3.7 + vec3f(11.3, 5.7, time * 0.07);
    let p2  = worldPos * 7.1 + vec3f(3.9, 17.2, time * 0.03);
    let n0  = snoise3(p0);
    let n1  = snoise3(p1) * 0.42;
    let n2  = snoise3(p2) * 0.18;

    // Sine-modulated vein mask with multi-octave detail
    let veinVal = sin((p0.x + p0.y + n0 + n1 + n2) * PI * 1.6) * 0.5 + 0.5;
    let vein    = pow(veinVal, mix(3.0, 8.0, veinContrast));

    // Secondary micro-vein network (fine capillary structure)
    let microVein = pow(sin((p1.y + p1.z + n1 * 2.0) * PI * 3.4) * 0.5 + 0.5, 6.0);

    // Marble vein colour: cool grey-white for LayerNorm normalised surface
    let veinColor = vec3f(0.88, 0.90, 0.92);
    let microColor = vec3f(0.80, 0.82, 0.86);

    // Sub-surface grey that veins emerge from
    let stoneBase = baseColor * vec3f(0.78, 0.80, 0.84);

    var col = mix(stoneBase, veinColor, vein * 0.55);
    col     = mix(col, microColor, microVein * 0.15);

    // Soft specular polish on vein ridges (matcap-enhanced via the Fresnel stage)
    let polishRim = fresnel_f(N, normalize(-worldPos), 5.5) * vein * 0.18;
    col += vec3f(polishRim);

    return col;
}
`;

/**
 * embedding — organic membrane
 *
 * Implements a translucent lipid-bilayer look: a thin subsurface-scatter
 * approximation (back-scatter term) overlaid with a pearlescent oil-film sheen.
 * The membrane thickness oscillates with the SPH pressure (cell volume changes).
 *
 * PBR contribution: soft dielectric diffuse (biological tissue has ~0 metallic).
 * Fresnel contribution: pale green rim (membrane edge translucency cue).
 * Matcap contribution: subtle (matcapWeight = 0.15) — just enough to fake
 *   subsurface-scattered studio light bleeding through the membrane rim.
 */
const WGSL_PATCH_EMBEDDING = /* wgsl */`
// ── Embedding: organic membrane (SSS + oil-film) ──────────────────────────────
fn applyEmbeddingMaterial(
    baseColor     : vec3f,
    N             : vec3f,
    V             : vec3f,
    L             : vec3f,
    uv            : vec2f,
    time          : f32,
    membraneThick : f32   // physics-driven thickness (pressure)
) -> vec3f {
    // ── Subsurface scatter approximation (wrap-lighting) ─────────────────────
    // Back-scatter: light wrapping around the thin membrane
    let wrap     = 0.45;
    let NdotL    = saturate_f((dot(N, L) + wrap) / (1.0 + wrap));
    let sssColor = vec3f(0.72, 0.96, 0.66); // chlorophyll-like transmission tint
    let sss      = sssColor * NdotL * 0.38;

    // ── Forward scatter (view-dependent translucency) ────────────────────────
    let VdotL    = saturate_f(dot(-V, L));
    let fwdScat  = pow(VdotL, 3.0) * vec3f(0.55, 0.85, 0.50) * 0.22;

    // ── Thin-film oil sheen ───────────────────────────────────────────────────
    // Membrane thickness oscillates with pressure (±30 nm breathing)
    let oscThick  = membraneThick + sin(time * 0.55 + uv.x * TWO_PI) * 30.0;
    let sheenIrid = iridescence(oscThick, 1.38, saturate_f(dot(N, V)));

    // ── Translucency: partial transmission tint ──────────────────────────────
    let transAlpha = 0.60;  // membrane is 60% transmissive
    let transColor = vec3f(0.55, 0.90, 0.55); // green-shifted photosynthetic tint
    let transLight = transColor * transAlpha;

    var col = baseColor * (vec3f(1.0) - vec3f(transAlpha) + transLight);
    col    += sss;
    col    += fwdScat;
    col    += sheenIrid * 0.28;

    return col;
}
`;

/**
 * softmax — luminous energy
 *
 * Creates a hot-core emissive effect: a saturated colour temperature bloom
 * driven by the probability distribution sharpness.  High kinetic energy
 * (peaked softmax) pushes the core toward white-hot; low energy gives a
 * diffuse warm glow.  Uses a radial energy-density gradient from UV centre.
 *
 * PBR contribution: diffuse base provides a dark substrate at low energy.
 * Fresnel contribution: amber corona rim at grazing angles (coronal envelope).
 * Matcap contribution: slight (matcapWeight = 0.10) — adds a studio-light
 *   hot-spot that simulates concentrated energy at the cell equator.
 */
const WGSL_PATCH_SOFTMAX = /* wgsl */`
// ── Softmax: luminous energy core ────────────────────────────────────────────
fn applySoftmaxMaterial(
    baseColor  : vec3f,
    N          : vec3f,
    V          : vec3f,
    uv         : vec2f,
    time       : f32,
    energyLevel: f32   // physics-driven energy (kinetic energy normalised)
) -> vec3f {
    // Radial distance from UV centre — drives energy density falloff
    let uvCentre = uv - vec2f(0.5);
    let r        = length(uvCentre);

    // Colour temperature gradient: cool perimeter → hot white-gold core
    // Three-zone: deep amber perimeter → orange-gold mid → near-white core
    let outerColor = vec3f(1.0, 0.43, 0.0);
    let midColor   = vec3f(1.0, 0.72, 0.28);
    let coreColor  = vec3f(1.0, 0.97, 0.88);
    let tempColor  = mix(
        mix(coreColor, midColor, smoothstep(0.0, 0.25, r)),
        outerColor,
        smoothstep(0.25, 0.50, r)
    );

    // Energy-scaled HDR emission (values > 1 intentional — bloom extracts them)
    let emission   = tempColor * energyLevel * mix(2.8, 5.5, 1.0 - r);

    // Pulsing flare: periodic intensity spike (probability distribution mode)
    let pulse = 0.5 + 0.5 * sin(time * 2.6 + energyLevel * PI);
    let flare = pow(pulse, 4.0) * 0.6 * energyLevel;

    // Secondary rapid flicker: stochastic energy micro-fluctuations
    let microFlicker = pow(0.5 + 0.5 * sin(time * 11.3 + r * 17.0), 8.0) * 0.15 * energyLevel;

    // Hot Fresnel rim (coronal envelope, enhanced by the Fresnel stage)
    let rim = fresnel_f(N, V, 2.8) * outerColor * energyLevel * 1.4;

    var col = baseColor + emission + vec3f(flare + microFlicker) + rim;

    return col;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// SpeciesMaterialDef — complete material definition for one species
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A physics-driven modulator binding specific to the material system.
 * Extends the base PhysicsBinding with a typed `target` field that names
 * the material param being driven.
 */
export interface MaterialPhysicsModulator extends PhysicsBinding {
  /**
   * Name of the material param driven by this modulator.
   * Must match a key of PBRParams, MatcapParams, FresnelConfig, or one of the
   * species-patch uniform names (e.g. 'iridScale', 'causticScale').
   */
  target: string;
}

/**
 * Complete material definition for a single species.
 *
 * M766 extends the original M719 SpeciesMaterialDef with:
 *   - `fresnelConfig` — dedicated per-species Fresnel parameters
 *   - `matcapBlend`   — per-species matcap composition weight and blend mode
 *
 * These two additions let every species participate in the full PBR+Fresnel+Matcap
 * hybrid pipeline while maintaining backward compatibility (the `materialType`
 * field still controls which primary path is used).
 */
export interface SpeciesMaterialDef {
  // ── Identity ────────────────────────────────────────────────────────────────

  /** Conceptual species identifier. */
  species: CellSpecies;

  /** Human-readable description of the visual concept. */
  visualConcept: string;

  // ── Shader path ─────────────────────────────────────────────────────────────

  /**
   * Base material pipeline used for this species.
   * 'pbr'        → ATPBRMaterial (Cook-Torrance full path)
   * 'matcap'     → ATMatcapFresnel (fast matcap+Fresnel path)
   * 'iridescence'→ ATPBRMaterial with iridStrength > 0
   */
  materialType: MaterialType;

  /**
   * WGSL fragment patch appended after the PBR+Fresnel+Matcap composition.
   * Implements the species-unique visual effect (see WGSL_PATCH_* above).
   */
  wgslPatch: string;

  /**
   * Entry-point function name in `wgslPatch`.
   * Called by the assembled shader after the three-stage composition.
   */
  patchFn: string;

  // ── CPU-side params ─────────────────────────────────────────────────────────

  /** Base PBR params (forwarded to PBRUniforms). null for matcap-only species. */
  pbrParams: PBRParams | null;

  /** Base matcap params (forwarded to MatcapUniforms). null for PBR-only species. */
  matcapParams: MatcapParams | null;

  // ── Hybrid pipeline config (M766) ──────────────────────────────────────────

  /**
   * Per-species Fresnel configuration for the dedicated Fresnel stage.
   * Every species gets a Fresnel rim; this controls its appearance.
   */
  fresnelConfig: FresnelConfig;

  /**
   * Per-species matcap blend configuration.
   * Species with `matcapWeight: 0` skip the matcap lookup entirely.
   */
  matcapBlend: MatcapBlendConfig;

  // ── Physics modulators ──────────────────────────────────────────────────────

  /**
   * Physics-driven parameter modulators evaluated every frame.
   * Applied after base params are written; result is written back to the GPU uniform buffer.
   */
  physicsModulators: MaterialPhysicsModulator[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-species definitions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * attention — iridescent metallic
 *
 * Multi-head attention distributes across the sequence like light diffracting
 * through a thin soap film: different query subspaces catch different wavelengths.
 * High metallic + low roughness provides a mirror substrate; the WGSL patch
 * layers three independent thin-film interference sweeps to produce the
 * characteristic head-shifting cyan-violet-gold shimmer.
 *
 * Hybrid pipeline:
 *   PBR  → high metallic mirror substrate (Cook-Torrance specular dominant)
 *   Fresnel → cyan-tinted, high-power rim (IOR shimmer at grazing)
 *   Matcap  → off (matcapWeight: 0); iridescence drives all specular colour
 */
export const ATTENTION_MATERIAL: SpeciesMaterialDef = {
  species:      'attention',
  visualConcept: 'Iridescent metallic — three-layer thin-film interference, anisotropic shimmer',

  materialType: 'iridescence',
  wgslPatch:    WGSL_PATCH_ATTENTION,
  patchFn:      'applyAttentionMaterial',

  pbrParams: {
    ...DEFAULT_PBR_PARAMS,
    // Base: deep indigo-metal substrate
    albedo:        [0.20, 0.22, 0.42],   // #33376B dark indigo, linear
    metallic:      0.88,                  // near-full metallic for mirror substrate
    roughness:     0.12,                  // very smooth — mirror-like
    ao:            0.92,
    // Warm directional key light (sun-like)
    lightPos:      [6.0, 10.0, 8.0],
    lightColor:    [2.8, 2.6, 2.2],
    cameraPos:     [0.0, 0.0, 10.0],
    // Iridescent Fresnel rim: shifts cyan at grazing angles
    fresnelPower:  4.5,
    fresnelColor:  [0.45, 0.80, 1.0],    // cyan-white rim
    // Primary irid layer: cyan-violet band
    iridThickness: 430.0,                 // nm
    iridIOR:       1.50,
    iridStrength:  0.75,                  // strong — film dominates surface look
    // Minimal atmospheric haze
    atmoDensity:   0.30,
    atmoDepth:     0.0,
    atmoFogColor:  [0.05, 0.08, 0.20],
    // Cool environment (deep-space ambient)
    envColor:      [0.06, 0.08, 0.18],
    time:          0.0,
  },
  matcapParams: null,

  fresnelConfig: {
    power:     4.8,
    tint:      [0.40, 0.78, 1.0],        // cyan rim — matches iridescence band
    intensity: 0.85,
    bias:      -0.03,                     // slightly tighter than neutral
  },

  matcapBlend: {
    matcapWeight: 0.0,                    // pure PBR+iridescence, no matcap
    blendMode:    'lerp',
  },

  physicsModulators: [
    // Vorticity drives irid thickness breathing (head rotation = film spin)
    { target: 'iridThickness', channel: 'u_vorticity',     scale: 90.0,  clampMin: -150, clampMax: 150,  mode: 'add' },
    // Kinetic energy sharpens metallic roughness (activated heads → glossier)
    { target: 'roughness',     channel: 'u_kineticEnergy', scale: -0.08, clampMin: 0.06, clampMax: 0.22, mode: 'add' },
    // Density drives irid strength (dense context → brighter film)
    { target: 'iridStrength',  channel: 'u_density',       scale: 0.25,  clampMin: 0.45, clampMax: 1.0,  mode: 'multiply' },
    // Speed scales dedicated Fresnel intensity (fast attention = brighter rim)
    { target: 'fresnelIntensity', channel: 'speed',         scale: 0.8,  clampMin: 0.5,  clampMax: 1.0,  mode: 'multiply' },
    // Pressure subtly widens Fresnel bias (compressed context = wider shimmer)
    { target: 'fresnelBias',      channel: 'u_pressure',    scale: 0.05, clampMin: -0.08, clampMax: 0.08, mode: 'add' },
  ],
};

/**
 * ffn — glass refraction
 *
 * Feed-forward networks apply non-linear activation to projected features —
 * light bending through glass is a physical analogy: input is transformed,
 * refracted, and exits carrying caustic patterns that encode the activation.
 *
 * Hybrid pipeline:
 *   PBR  → dielectric (metallic: 0), very smooth (roughness: 0.06)
 *   Fresnel → warm, high-power rim (total internal reflection cue)
 *   Matcap  → off (matcapWeight: 0); glass is view-dependent, not studio-lit
 */
export const FFN_MATERIAL: SpeciesMaterialDef = {
  species:      'ffn',
  visualConcept: 'Glass refraction — Snell-law transmission, dual-frequency caustics, chromatic dispersion',

  materialType: 'pbr',
  wgslPatch:    WGSL_PATCH_FFN,
  patchFn:      'applyFFNMaterial',

  pbrParams: {
    ...DEFAULT_PBR_PARAMS,
    // Base: near-transparent orange-amber glass substrate
    albedo:        [0.92, 0.55, 0.10],   // #EB8C19 amber, linear
    metallic:      0.0,                   // dielectric glass
    roughness:     0.06,                  // very smooth polished glass
    ao:            1.0,
    // Strong high-angle key light (produces caustic projection)
    lightPos:      [8.0, 6.0, 4.0],
    lightColor:    [3.2, 3.0, 2.6],
    cameraPos:     [0.0, 0.0, 10.0],
    // High Fresnel power (glass edge is very reflective)
    fresnelPower:  6.0,
    fresnelColor:  [1.0, 0.95, 0.80],    // warm specular rim
    // No iridescence — glass uses the FFN caustic patch instead
    iridThickness: 0.0,
    iridIOR:       1.0,
    iridStrength:  0.0,
    // Slight atmospheric depth haze (glass depth cueing)
    atmoDensity:   0.20,
    atmoDepth:     0.0,
    atmoFogColor:  [0.08, 0.06, 0.04],
    // Warm environment fill (diffuse transmission colour)
    envColor:      [0.14, 0.11, 0.07],
    time:          0.0,
  },
  matcapParams: null,

  fresnelConfig: {
    power:     6.5,
    tint:      [1.0, 0.92, 0.75],        // warm glass edge — total internal reflection
    intensity: 0.90,
    bias:      0.0,
  },

  matcapBlend: {
    matcapWeight: 0.0,                    // pure PBR glass, no matcap
    blendMode:    'lerp',
  },

  physicsModulators: [
    // Kinetic energy scales caustic brightness (FFN activation magnitude)
    { target: 'causticScale',  channel: 'u_kineticEnergy', scale: 1.8,  clampMin: 0.4, clampMax: 3.5, mode: 'multiply' },
    // Pressure makes glass slightly more opaque (dense FFN layer)
    { target: 'roughness',     channel: 'u_pressure',      scale: 0.04, clampMin: 0.04, clampMax: 0.18, mode: 'add' },
    // Vorticity animates caustic stripes (turbulent non-linearity)
    { target: 'atmoDepth',     channel: 'u_vorticity',     scale: 0.06, clampMin: 0.0,  clampMax: 0.25, mode: 'add' },
    // Speed drives Fresnel intensity (fast routing = brighter edge)
    { target: 'fresnelIntensity', channel: 'speed',         scale: 0.9,  clampMin: 0.6,  clampMax: 1.0,  mode: 'multiply' },
  ],
};

/**
 * layernorm — matcap marble
 *
 * Layer normalisation creates a smooth, bounded, statistically clean output —
 * marble captures this: its surface is polished (smooth), veined (internal
 * structure), and cool-toned (neutral, non-activating).
 *
 * Hybrid pipeline:
 *   PBR  → lightweight ambient term only (no specular highlights from PBR)
 *   Fresnel → cool grey-blue rim (polished stone edge catch-light)
 *   Matcap  → HIGH (matcapWeight: 0.55, multiply blend) — marble uses matcap
 *             for studio-style specular that imitates polished stone under soft
 *             box light.  Multiply blend darkens veins naturally.
 */
export const LAYERNORM_MATERIAL: SpeciesMaterialDef = {
  species:      'layernorm',
  visualConcept: 'Matcap marble — multi-octave Simplex veins, micro-capillary network, polished Fresnel',

  materialType: 'matcap',
  wgslPatch:    WGSL_PATCH_LAYERNORM,
  patchFn:      'applyLayerNormMaterial',

  pbrParams: null,
  matcapParams: {
    ...DEFAULT_MATCAP_PARAMS,
    // Soft Fresnel rim (polished stone edge catch-light)
    fresnelPower:  4.2,
    fresnelColor:  [0.82, 0.86, 0.90],   // cool grey-blue rim
    // Minimal noise perturbation (marble is smooth, not rough)
    noiseScale:    0.85,
    noiseStrength: 0.08,
    // Marble base tint: pale grey-white (#ECEFF1 linear approx)
    species:       [0.90, 0.92, 0.94],
    tintStrength:  0.70,
    time:          0.0,
  },

  fresnelConfig: {
    power:     4.5,
    tint:      [0.80, 0.85, 0.92],       // cool grey-blue (polished stone)
    intensity: 0.65,
    bias:      0.02,                      // slightly wider — marble has gentle rounding
  },

  matcapBlend: {
    matcapWeight: 0.55,                   // matcap-dominant for studio stone look
    blendMode:    'multiply',             // darkens veins within matcap silhouette
  },

  physicsModulators: [
    // Contact count sharpens marble veins (normalisation events = vein crispness)
    { target: 'veinContrast',  channel: 'u_contactCount', scale: 0.08, clampMin: 0.1, clampMax: 0.9, mode: 'add' },
    // Density slightly boosts Fresnel intensity (denser context = more reflective)
    { target: 'fresnelIntensity', channel: 'u_density',   scale: 0.6,  clampMin: 0.4, clampMax: 0.9, mode: 'multiply' },
    // Pressure tightens noise (compressed layer → finer grain)
    { target: 'noiseScale',    channel: 'u_pressure',     scale: 0.5,  clampMin: 0.5, clampMax: 1.8, mode: 'multiply' },
    // Vorticity modulates matcap weight (turbulent normalisation = matcap flicker)
    { target: 'matcapWeight',  channel: 'u_vorticity',    scale: 0.12, clampMin: 0.40, clampMax: 0.65, mode: 'add' },
  ],
};

/**
 * embedding — organic membrane
 *
 * Token embeddings live in a continuous latent space — they are alive, soft,
 * semi-transparent.  The organic membrane material uses subsurface scatter
 * (wrapped NdotL + forward scatter) to simulate light passing through a thin
 * lipid bilayer, plus an oil-film iridescence that encodes the spectral
 * richness of the embedding manifold.
 *
 * Hybrid pipeline:
 *   PBR  → soft dielectric diffuse (biological tissue)
 *   Fresnel → pale green rim (membrane edge translucency)
 *   Matcap  → subtle (matcapWeight: 0.15, screen blend) — fakes subsurface
 *             scattered studio light bleeding through the membrane rim
 */
export const EMBEDDING_MATERIAL: SpeciesMaterialDef = {
  species:      'embedding',
  visualConcept: 'Organic membrane — SSS wrap + forward scatter, oil-film sheen, translucent bilayer',

  materialType: 'pbr',
  wgslPatch:    WGSL_PATCH_EMBEDDING,
  patchFn:      'applyEmbeddingMaterial',

  pbrParams: {
    ...DEFAULT_PBR_PARAMS,
    // Base: semi-transparent organic green (chloroplast palette)
    albedo:        [0.38, 0.72, 0.35],   // #61B85A green, linear
    metallic:      0.0,                   // fully dielectric (biological)
    roughness:     0.40,                  // soft, slightly waxy surface
    ao:            0.95,
    // Diffuse bio-luminescent light (no hard shadows in organic media)
    lightPos:      [3.0, 8.0, 6.0],
    lightColor:    [1.8, 2.0, 1.6],
    cameraPos:     [0.0, 0.0, 10.0],
    // Low-power Fresnel (translucent edge, not reflective)
    fresnelPower:  2.2,
    fresnelColor:  [0.60, 0.95, 0.58],   // pale green rim
    // Thin film: membrane ~350 nm (soap-bubble range), IOR 1.38 (lipid)
    iridThickness: 350.0,
    iridIOR:       1.38,
    iridStrength:  0.30,                  // subtle — just a sheen, not dominant
    // No atmospheric haze inside the cell membrane
    atmoDensity:   0.10,
    atmoDepth:     0.0,
    atmoFogColor:  [0.04, 0.12, 0.04],
    // Warm bio-luminescent ambient
    envColor:      [0.08, 0.14, 0.06],
    time:          0.0,
  },
  matcapParams: null,

  fresnelConfig: {
    power:     2.5,
    tint:      [0.55, 0.92, 0.52],       // pale green (membrane translucency)
    intensity: 0.55,
    bias:      0.05,                      // wider — organic surfaces are rounded
  },

  matcapBlend: {
    matcapWeight: 0.15,                   // subtle SSS studio bleed-through
    blendMode:    'screen',               // brightens — simulates back-lit membrane
  },

  physicsModulators: [
    // Pressure oscillates membrane thickness (volume compression → film shift)
    { target: 'membraneThick', channel: 'u_pressure',     scale: 60.0, clampMin: 280.0, clampMax: 480.0, mode: 'add' },
    // Neighbour count boosts SSS brightness (cell cluster → more internal scatter)
    { target: 'iridStrength',  channel: 'u_neighborCount', scale: 0.03, clampMin: 0.15, clampMax: 0.55, mode: 'multiply' },
    // Kinetic energy slightly increases roughness (membrane agitation)
    { target: 'roughness',     channel: 'u_kineticEnergy', scale: 0.12, clampMin: 0.28, clampMax: 0.60, mode: 'add' },
    // Speed modulates Fresnel intensity (moving cells = more translucent rim)
    { target: 'fresnelIntensity', channel: 'speed',         scale: 0.5,  clampMin: 0.35, clampMax: 0.75, mode: 'multiply' },
    // Density drives matcap weight (clustered cells catch more studio light)
    { target: 'matcapWeight',     channel: 'u_density',     scale: 0.06, clampMin: 0.10, clampMax: 0.25, mode: 'add' },
  ],
};

/**
 * softmax — luminous energy
 *
 * Softmax concentrates probability mass onto the winning token — a hot-core
 * plasma.  The energy material renders an HDR emissive gradient from a
 * white-hot centre to a deep amber corona.  High kinetic energy (peaked
 * distribution) pushes the core to near-white; low energy gives a diffuse
 * warm glow.  Values above 1.0 intentionally feed the bloom post-process.
 *
 * Hybrid pipeline:
 *   PBR  → diffuse substrate (visible at low energy before emission dominates)
 *   Fresnel → amber corona rim (coronal arc at cell edge)
 *   Matcap  → subtle (matcapWeight: 0.10, screen blend) — adds equatorial
 *             hot-spot that concentrates the perceived energy
 */
export const SOFTMAX_MATERIAL: SpeciesMaterialDef = {
  species:      'softmax',
  visualConcept: 'Luminous energy — three-zone colour temperature, HDR emission, coronal bloom + micro-flicker',

  materialType: 'pbr',
  wgslPatch:    WGSL_PATCH_SOFTMAX,
  patchFn:      'applySoftmaxMaterial',

  pbrParams: {
    ...DEFAULT_PBR_PARAMS,
    // Base: deep orange-red substrate (black-body glow at low energy)
    albedo:        [0.98, 0.42, 0.04],   // #FA6B0A hot orange, linear
    metallic:      0.0,                   // not metallic — emissive plasma
    roughness:     0.80,                  // diffuse surface (glow, not specular)
    ao:            0.85,
    // Key light matches the emissive colour temperature
    lightPos:      [0.0, 5.0, 8.0],
    lightColor:    [3.5, 2.8, 1.8],
    cameraPos:     [0.0, 0.0, 10.0],
    // Bright Fresnel corona (coronal arc at cell edge)
    fresnelPower:  2.5,
    fresnelColor:  [1.0, 0.60, 0.20],    // amber corona rim
    // No iridescence (pure emissive, no film interference)
    iridThickness: 0.0,
    iridIOR:       1.0,
    iridStrength:  0.0,
    // Heat haze (slight atmospheric distortion near hot surface)
    atmoDensity:   0.55,
    atmoDepth:     0.0,
    atmoFogColor:  [0.18, 0.08, 0.02],
    // Warm incandescent ambient (own light fills environment)
    envColor:      [0.20, 0.12, 0.04],
    time:          0.0,
  },
  matcapParams: null,

  fresnelConfig: {
    power:     2.8,
    tint:      [1.0, 0.55, 0.15],        // amber corona (matches emissive palette)
    intensity: 0.95,
    bias:      0.04,                      // wider — plasma has diffuse edge
  },

  matcapBlend: {
    matcapWeight: 0.10,                   // very subtle equatorial hot-spot
    blendMode:    'screen',               // brightens — energy adds, never subtracts
  },

  physicsModulators: [
    // Kinetic energy is the primary energy level driver (sharp softmax = hot core)
    { target: 'energyLevel',   channel: 'u_kineticEnergy', scale: 1.2,  clampMin: 0.3,  clampMax: 1.8,  mode: 'multiply' },
    // Density boosts ambient env (many hot cells light each other)
    { target: 'envColor',      channel: 'u_density',       scale: 0.15, clampMin: 0.08, clampMax: 0.45, mode: 'multiply' },
    // Vorticity spreads the corona Fresnel (turbulent plasma edge)
    { target: 'fresnelBias',   channel: 'u_vorticity',     scale: 0.06, clampMin: 0.0,  clampMax: 0.12, mode: 'add' },
    // Pressure sharpens emission falloff (compressed = denser energy)
    { target: 'roughness',     channel: 'u_pressure',      scale: -0.1, clampMin: 0.55, clampMax: 0.90, mode: 'add' },
    // Speed drives Fresnel intensity (fast-moving = brighter corona)
    { target: 'fresnelIntensity', channel: 'speed',         scale: 0.7,  clampMin: 0.6,  clampMax: 1.0,  mode: 'multiply' },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CELL_MATERIAL_REGISTRY
 *
 * Central lookup table: CellSpecies → SpeciesMaterialDef.
 * Consumers should use `getCellMaterial()` rather than indexing directly.
 */
export const CELL_MATERIAL_REGISTRY: Readonly<Record<CellSpecies, SpeciesMaterialDef>> = {
  attention : ATTENTION_MATERIAL,
  ffn       : FFN_MATERIAL,
  layernorm : LAYERNORM_MATERIAL,
  embedding : EMBEDDING_MATERIAL,
  softmax   : SOFTMAX_MATERIAL,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve the SpeciesMaterialDef for a given species.
 *
 * @example
 *   const def = getCellMaterial('attention');
 *   // def.materialType === 'iridescence'
 *   // def.fresnelConfig.power === 4.8
 */
export function getCellMaterial(species: CellSpecies): SpeciesMaterialDef {
  return CELL_MATERIAL_REGISTRY[species];
}

/**
 * Evaluate all physics modulators for a species against a live PhysicsUniforms
 * snapshot and return a partial record of resolved target values.
 *
 * The caller is responsible for writing the resolved values back to the
 * appropriate GPU uniform buffer via packPBRUniforms / packMatcapUniforms.
 *
 * @param species   Target species.
 * @param physics   Live PhysicsUniforms snapshot from physics-uniform-bridge.ts.
 * @returns         Record mapping target param names → resolved values.
 *
 * @example
 *   const resolved = resolveMaterialPhysics('softmax', physicsSnapshot);
 *   // resolved.energyLevel === 1.4  (kinetic energy modulated)
 *   // resolved.fresnelIntensity === 0.85  (speed modulated)
 */
export function resolveMaterialPhysics(
  species: CellSpecies,
  physics: PhysicsUniforms,
): Record<string, number> {
  const def      = getCellMaterial(species);
  const resolved : Record<string, number> = {};
  const MAX_SPEED = 10.0; // world units/s — matches species-shader-registry.ts

  // Derive speed pseudo-channel
  const [vx, vy] = physics.u_velocity;
  const speed    = Math.sqrt(vx * vx + vy * vy) / MAX_SPEED;

  for (const mod of def.physicsModulators) {
    // Read raw channel value
    let raw: number;
    if (mod.channel === 'speed') {
      raw = speed;
    } else {
      raw = physics[mod.channel as keyof PhysicsUniforms] as number;
    }

    // Scale and clamp
    const scaled  = raw * mod.scale;
    const clamped = Math.max(
      mod.clampMin ?? 0,
      Math.min(mod.clampMax ?? 1, scaled),
    );

    // Apply mode against base param value (read from pbrParams, matcapParams, or fresnelConfig)
    const base   = _baseParamValue(def, mod.target);
    const mode   = mod.mode ?? 'multiply';

    let result: number;
    if (mode === 'multiply') {
      result = base * clamped;
    } else if (mode === 'add') {
      result = base + clamped;
    } else {
      // lerp
      const target = mod.lerpTarget ?? base * 1.5;
      result = base + (target - base) * clamped;
    }

    resolved[mod.target] = result;
  }

  return resolved;
}

/**
 * Assemble the full WGSL shader source for a species by concatenating
 * the base AT PBR shader WGSL blocks with the Fresnel stage, matcap blend
 * helpers, and the species-specific fragment patch.
 *
 * The returned string is suitable for passing to `device.createShaderModule()`.
 *
 * @param species  Target species.
 * @param baseWGSL Base WGSL blocks from AT_PBR_WGSL (at-pbr-material.ts export).
 * @returns        Complete WGSL shader source string.
 *
 * @example
 *   import { AT_PBR_WGSL } from './at-pbr-material';
 *   const src = assembleCellShader('attention', AT_PBR_WGSL);
 *   const mod = device.createShaderModule({ code: src });
 */
export function assembleCellShader(
  species  : CellSpecies,
  baseWGSL : {
    mathHelpers : string;
    pbrBRDF     : string;
    fresnel     : string;
    iridescence : string;
    fullscreenVS: string;
    pbrFrag     : string;
    matcapFrag  : string;
  },
): string {
  const def = getCellMaterial(species);

  const blocks: string[] = [
    baseWGSL.mathHelpers,
    baseWGSL.pbrBRDF,
    baseWGSL.fresnel,
    baseWGSL.iridescence,
  ];

  // PBR fragment (all species get ambient term; matcap-primary species
  // override specular in the matcap stage)
  if (def.materialType === 'matcap') {
    blocks.push(baseWGSL.matcapFrag);
  } else {
    blocks.push(baseWGSL.pbrFrag);
  }

  // Fresnel composition stage — always included
  blocks.push(WGSL_FRESNEL_STAGE);

  // Matcap blend helpers — always included (species with matcapWeight=0 skip at runtime)
  blocks.push(WGSL_MATCAP_BLEND);

  // Species-specific patch
  blocks.push(def.wgslPatch);

  blocks.push(baseWGSL.fullscreenVS);

  return blocks.join('\n\n/* ─────────────────────── */\n\n');
}

/**
 * Return all registered species identifiers.
 */
export function getAllCellSpecies(): CellSpecies[] {
  return Object.keys(CELL_MATERIAL_REGISTRY) as CellSpecies[];
}

/**
 * Return species identifiers filtered by material type.
 *
 * @example
 *   getSpeciesByMaterialType('iridescence');
 *   // → ['attention']
 */
export function getSpeciesByMaterialType(type: MaterialType): CellSpecies[] {
  return getAllCellSpecies().filter(s => CELL_MATERIAL_REGISTRY[s].materialType === type);
}

/**
 * Return species identifiers that have a non-zero matcap blend weight.
 * Useful for render-loop optimisation: only these species need a matcap
 * texture lookup and the blend composition pass.
 *
 * @example
 *   getSpeciesWithMatcap();
 *   // → ['layernorm', 'embedding', 'softmax']
 */
export function getSpeciesWithMatcap(): CellSpecies[] {
  return getAllCellSpecies().filter(s => CELL_MATERIAL_REGISTRY[s].matcapBlend.matcapWeight > 0);
}

/**
 * Return the matcap blend function name for the given blend mode.
 * Used when constructing the final composition call in the assembled shader.
 *
 * @example
 *   getMatcapBlendFn('multiply');
 *   // → 'blendMatcap_multiply'
 */
export function getMatcapBlendFn(mode: MatcapBlendConfig['blendMode']): string {
  return `blendMatcap_${mode}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Read the design-time base value for a named param from a material def.
 * Used as the left-hand operand for physics modulator evaluation.
 *
 * M766 extension: also reads from fresnelConfig and matcapBlend.
 */
function _baseParamValue(def: SpeciesMaterialDef, target: string): number {
  // Try Fresnel config first (M766 targets)
  const fresnelMap: Record<string, number> = {
    fresnelIntensity: def.fresnelConfig.intensity,
    fresnelBias:      def.fresnelConfig.bias,
    fresnelPowerDedicated: def.fresnelConfig.power,
  };
  if (target in fresnelMap) return fresnelMap[target];

  // Try matcap blend config (M766 targets)
  if (target === 'matcapWeight') return def.matcapBlend.matcapWeight;

  // Try PBR params
  if (def.pbrParams) {
    const p = def.pbrParams as Record<string, unknown>;
    if (target in p) {
      const v = p[target];
      if (typeof v === 'number') return v;
      // For vec3 params (e.g. 'envColor'), return the channel-average
      if (Array.isArray(v) && v.length >= 1) return (v as number[]).reduce((a, b) => a + b, 0) / v.length;
    }
  }

  // Try matcap params
  if (def.matcapParams) {
    const m = def.matcapParams as Record<string, unknown>;
    if (target in m) {
      const v = m[target];
      if (typeof v === 'number') return v;
      if (Array.isArray(v) && v.length >= 1) return (v as number[]).reduce((a, b) => a + b, 0) / v.length;
    }
  }

  // Species-patch uniforms that have no CPU-side base (use sensible neutrals)
  const PATCH_DEFAULTS: Record<string, number> = {
    iridScale     : 1.0,
    causticScale  : 1.0,
    veinContrast  : 0.5,
    membraneThick : 350.0,
    energyLevel   : 1.0,
  };

  return PATCH_DEFAULTS[target] ?? 1.0;
}
