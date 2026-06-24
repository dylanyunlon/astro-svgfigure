/**
 * species-shader-registry.ts — M723: Species Shader Registry
 * ─────────────────────────────────────────────────────────────────────────────
 * Single source of truth for the complete shader stack of every cil-* species.
 *
 * Each entry in the registry (SpeciesShaderConfig) describes:
 *   • sdfShape        — which lygia/AT SDF primitive drives the silhouette
 *   • materialType    — surface shading model (matcap | pbr | iridescence)
 *   • patternShader   — secondary pattern/animation shader applied inside the cell
 *   • bloomStrength   — base UnrealBloom bloomScale (matches BLOOM_DEFAULTS in pixi-cell-renderer)
 *   • physicsBindings — which PhysicsUniforms channels modulate each visual param
 *
 * Design notes
 * ─────────────────────────────────────────────────────────────────────────────
 * • The registry is intentionally *declarative*: it describes the stack, not
 *   the GL calls.  Consumers (pixi-cell-renderer, sdf-species-filter, nuke-pipeline,
 *   at-pbr-material, at-bloom-postprocess) read from it and self-configure.
 * • physicsBindings encodes *how* PhysicsUniforms from physics-uniform-bridge.ts
 *   drives each visual dimension — density, velocity, pressure, vorticity, kinetic energy.
 *   Binding values are dimensionless multipliers on a [0, 1] normalised physics signal.
 * • bloomStrength is the *base* value; the live UIL system in uil-species-live.ts
 *   will modulate it by densityRatio at runtime.
 * • Pattern shaders map 1:1 to filenames in src/lib/shaders/ (no extension).
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/organic-sdf.ts              — SDF shape names & params
 *   src/lib/renderer/material/CellMaterial.ts — AT shader → species table
 *   src/lib/renderers/pixi-cell-renderer.ts  — BLOOM_DEFAULTS, SPECIES_COLORS
 *   src/lib/renderers/sdf-species-filter.ts  — PixiJS SDF filters per species
 *   src/lib/sph/uil-species-live.ts          — live UIL physics modulation
 *   src/lib/sph/physics-uniform-bridge.ts    — PhysicsUniforms definition
 *   src/lib/shaders/                         — pattern / surface shader source
 */




// ─── SDF Shape ────────────────────────────────────────────────────────────────

/**
 * SDF primitive that forms the species silhouette.
 *
 * Values correspond to functions in organic-sdf.ts / sdf-species-library.frag:
 *   flower     → flowerSDF(uv, petals)            lygia/sdf/flowerSDF
 *   koch       → kochSDF(uv, iterations)          lygia/sdf/kochSDF
 *   julia      → juliaSDF(uv, cx, cy)             escape-time quadratic Julia set
 *   supershape → supershapeSDF(uv, m, n1, n2, n3) Gielis / supershape-species.frag
 *   capsule    → sdCapsule(p, a, b, r)             lygia/sdf/capsuleSDF
 *   hexagon    → sdHexagon(p, r)                  lygia/sdf/hexagonSDF
 *   star       → sdStar(p, r, n, m)               Inigo Quilez star SDF
 *   roundbox   → sdRoundBox(p, b, r)              lygia/sdf/boxSDF rounded
 *   polygon    → sdPolygon(p, vertices, n)        lygia/sdf/polygonSDF
 */



import type { PhysicsUniforms } from './physics-uniform-bridge';

export type SdfShape =
  | 'flower'
  | 'koch'
  | 'julia'
  | 'supershape'
  | 'capsule'
  | 'hexagon'
  | 'star'
  | 'roundbox'
  | 'polygon';

/**
 * Tuning params forwarded to the SDF evaluation function.
 * Which keys are used depends on `sdfShape`.
 */
export interface SdfShapeParams {
  /** flower: petal count (2–12) */
  petals?: number;
  /** koch: fractal iteration depth (1–4) */
  kochIterations?: number;
  /** julia: real part of the Julia constant c */
  juliaRe?: number;
  /** julia: imaginary part of the Julia constant c */
  juliaIm?: number;
  /** supershape: rotational symmetry order m */
  supershapeM?: number;
  /** supershape: n1 / n2 / n3 Gielis exponents */
  supershapeN1?: number;
  supershapeN2?: number;
  supershapeN3?: number;
  /** star: number of points */
  starPoints?: number;
  /** star: inner-radius ratio (0–1) */
  starRatio?: number;
  /** roundbox: corner radius (0–0.5, normalised) */
  cornerRadius?: number;
  /** hexagon / polygon: circumradius (0–1, normalised) */
  radius?: number;
}

// ─── Material Type ────────────────────────────────────────────────────────────

/**
 * Surface shading model applied to the cell body.
 *
 *   matcap      → matcap-fresnel-cell.frag / at-pbr-material.ts fs_matcap
 *                 Cheap matcap sphere-mapped reflection.  Used for high-motion cells.
 *   pbr         → pbr-cell-surface.frag / at-pbr-material.ts fs_pbr
 *                 Cook-Torrance GGX + Fresnel rim.  Used for hero / attention cells.
 *   iridescence → pbr-cell-surface.frag with iridStrength > 0
 *                 Thin-film interference rainbow on top of PBR base layer.
 */
export type MaterialType = 'matcap' | 'pbr' | 'iridescence';

/** Material-level tuning params forwarded to the WGSL uniform buffers. */
export interface MaterialParams {
  // ── PBR ─────────────────────────────────────────────────────────────────────
  /** Base albedo RGB (linear). */
  albedo?: [number, number, number];
  /** Metallic factor 0-1 (u_pbr.metallic). */
  metallic?: number;
  /** Roughness factor 0-1 (u_pbr.roughness). */
  roughness?: number;
  /** Ambient occlusion 0-1 (u_pbr.ao). */
  ao?: number;
  /** Fresnel rim power (u_pbr.fresnelPower). */
  fresnelPower?: number;
  /** Fresnel rim colour RGB linear (u_pbr.fresnelColor). */
  fresnelColor?: [number, number, number];

  // ── Iridescence ──────────────────────────────────────────────────────────────
  /** Blend weight of iridescence over base PBR colour (0 = off, 1 = full). */
  iridStrength?: number;
  /** Thin-film optical thickness in nm (u_pbr.iridThickness). */
  iridThickness?: number;
  /** Thin-film index of refraction (u_pbr.iridIOR). */
  iridIOR?: number;

  // ── Matcap ────────────────────────────────────────────────────────────────────
  /** Matcap texture asset path (relative to assets/images/). */
  matcapSrc?: string;
  /** Tint RGB (linear) applied on top of sampled matcap colour. */
  tintColor?: [number, number, number];
  /** Tint blend weight (0 = raw matcap, 1 = full tint). */
  tintStrength?: number;
  /** Normal-map strength 0-1 (uNormalStrength). */
  normalStrength?: number;
}

// ─── Pattern Shader ───────────────────────────────────────────────────────────

/**
 * Secondary texture/animation pattern rendered inside the SDF mask.
 *
 * Values correspond to .frag filenames in src/lib/shaders/:
 *   none                → solid fill only
 *   grayscott-species   → Gray-Scott reaction–diffusion (grayscott-species.frag)
 *   supershape-species  → Supershape noise pattern (supershape-species.frag)
 *   voronoi-membrane    → Voronoi cell membrane (voronoi-membrane.frag)
 *   voronoi-natural     → Organic voronoi stipple (voronoi-natural.frag)
 *   iq-palette-species  → IQ cosine-palette colour field (iq-palette-species.frag)
 *   julia-background    → Julia-set escape field (julia-background.frag)
 *   turing-pattern      → Turing spot/stripe diffusion (turing-pattern from sph/)
 *   curl-trail          → Curl-noise particle trail (curl-trail.frag)
 *   fluid-surface       → Water/fluid surface normals (fluid-surface.frag)
 *   caustics            → Caustic light transport (caustics.frag)
 */
export type PatternShader =
  | 'none'
  | 'grayscott-species'
  | 'supershape-species'
  | 'voronoi-membrane'
  | 'voronoi-natural'
  | 'iq-palette-species'
  | 'julia-background'
  | 'turing-pattern'
  | 'curl-trail'
  | 'fluid-surface'
  | 'caustics';

// ─── Physics Bindings ─────────────────────────────────────────────────────────

/**
 * A single physics-to-visual binding.
 *
 * At runtime the physics channel is read from PhysicsUniforms, normalised
 * (see normalisers below), then multiplied by `scale` and clamped to
 * [clampMin, clampMax].  The result is added to (or multiplied with) the
 * visual target's base value.
 *
 *   visualValue = basValue op clamp(normalisedPhysics * scale, clampMin, clampMax)
 */
export interface PhysicsBinding {
  /**
   * Which field from PhysicsUniforms drives this binding.
   * 'speed' is derived as length(u_velocity) / MAX_SPEED (≈ 10 world-units/s).
   */
  channel:
    | 'u_density'
    | 'u_pressure'
    | 'u_vorticity'
    | 'u_kineticEnergy'
    | 'u_contactCount'
    | 'u_neighborCount'
    | 'speed';

  /** Multiplier applied to the normalised channel value before clamping. */
  scale: number;

  /** Lower clamp bound (default 0). */
  clampMin?: number;

  /** Upper clamp bound (default 1). */
  clampMax?: number;

  /**
   * How the scaled channel modifies the visual target's base value.
   *   'multiply'  → visualValue = base * channelValue   (default)
   *   'add'       → visualValue = base + channelValue
   *   'lerp'      → visualValue = lerp(base, target, channelValue)
   */
  mode?: 'multiply' | 'add' | 'lerp';

  /**
   * For 'lerp' mode: the value interpolated *towards* as the channel reaches 1.
   * E.g. { mode: 'lerp', lerp_target: 2.0 } → lerp(base, 2.0, channelValue).
   */
  lerpTarget?: number;
}

/**
 * Complete set of physics-driven visual bindings for a species.
 *
 * Each key is a visual target name recognised by the runtime consumer:
 *   bloomStrength    → AdvancedBloomFilter.bloomScale
 *   bloomRadius      → AdvancedBloomFilter.kernelSize (normalised 0-1)
 *   sdfDistort       → uSdfDistort uniform in sdf-species-library.frag
 *   patternSpeed     → u_speed uniform in pattern shader
 *   patternContrast  → u_contrast in pattern shader
 *   materialEnvScale → uEnv[0] in CellMaterialUniforms (PBR env diffuse)
 *   fresnelStrength  → uFresnelStrength / u_pbr.fresnelPower
 *   iridThickness    → u_pbr.iridThickness (iridescence only)
 *   opacity          → overall cell opacity / u_opacity
 *   pulseFrequency   → AT bloom pulse frequency (rad/s)
 */
export type PhysicsBindings = Partial<Record<
  | 'bloomStrength'
  | 'bloomRadius'
  | 'sdfDistort'
  | 'patternSpeed'
  | 'patternContrast'
  | 'materialEnvScale'
  | 'fresnelStrength'
  | 'iridThickness'
  | 'opacity'
  | 'pulseFrequency',
  PhysicsBinding
>>;

// ─── SpeciesShaderConfig ──────────────────────────────────────────────────────

/**
 * Complete shader stack definition for a single cil-* species.
 *
 * This is the canonical data structure — all other per-species shader
 * config tables in the codebase should eventually be derived from this.
 */
export interface SpeciesShaderConfig {
  // ── Identity ────────────────────────────────────────────────────────────────

  /** Canonical species string, e.g. 'cil-eye'. */
  id: string;

  /**
   * Human-readable role label (Transformer architecture analogy).
   * Used in debug overlays / UIL.
   */
  role: string;

  // ── SDF Shape ───────────────────────────────────────────────────────────────

  /** Signed-distance field primitive used for the cell silhouette. */
  sdfShape: SdfShape;

  /** Tuning params forwarded to the SDF evaluation function. */
  sdfParams: SdfShapeParams;

  // ── Material ────────────────────────────────────────────────────────────────

  /** Surface shading model: matcap | pbr | iridescence. */
  materialType: MaterialType;

  /** Material tuning params forwarded to the WGSL / GLSL uniform buffers. */
  materialParams: MaterialParams;

  // ── Pattern ─────────────────────────────────────────────────────────────────

  /** Secondary pattern/animation shader rendered inside the SDF mask. */
  patternShader: PatternShader;

  // ── Bloom ───────────────────────────────────────────────────────────────────

  /**
   * Base UnrealBloom bloomScale (AdvancedBloomFilter.bloomScale equivalent).
   * Matches BLOOM_DEFAULTS in pixi-cell-renderer.ts.
   * Runtime value = bloomStrength * physicsBinding(density).
   */
  bloomStrength: number;

  /**
   * Luminance threshold for bloom extraction (BloomThreshold).
   * 0 = all pixels glow; 1 = only near-white pixels glow.
   */
  bloomThreshold: number;

  /**
   * Kawase blur radius normalised to [0, 1].
   * Maps to BLOOM_RADIUS_SCALE (16) inside pixi-cell-renderer.
   */
  bloomRadius: number;

  /** Bloom pulse amplitude (fraction of bloomStrength). */
  bloomPulseAmplitude: number;

  /** Bloom pulse frequency (radians/second). */
  bloomPulseFrequency: number;

  // ── Physics Bindings ────────────────────────────────────────────────────────

  /**
   * Declares how live SPH PhysicsUniforms modulate each visual dimension.
   * Applied every frame by uil-species-live.ts / pixi-cell-renderer ticker.
   */
  physicsBindings: PhysicsBindings;
}

// ─── Registry ─────────────────────────────────────────────────────────────────

/**
 * SPECIES_SHADER_REGISTRY
 *
 * Complete shader stack for every cil-* species.
 *
 * Ordering mirrors SPECIES_ORDER in cell-body-bridge.ts (1-indexed; 0 = fluid).
 * New species must be appended *at the end* to preserve GPU particle type indices.
 *
 * Visual design rationale (per species):
 * ┌──────────────────┬───────────┬─────────────┬──────────────────────────────┐
 * │ Species          │ Role      │ Material    │ Pattern                      │
 * ├──────────────────┼───────────┼─────────────┼──────────────────────────────┤
 * │ cil-eye          │ Attention │ iridescence │ voronoi-membrane (iris)      │
 * │ cil-bolt         │ FFN/MLP   │ matcap      │ grayscott-species (reaction)  │
 * │ cil-vector       │ Embedding │ pbr         │ iq-palette-species (encoding) │
 * │ cil-plus         │ Add&Norm  │ pbr         │ none (clean residual)         │
 * │ cil-arrow-right  │ Skip/Flow │ matcap      │ curl-trail (directional)      │
 * │ cil-filter       │ Selection │ pbr         │ voronoi-natural (filter mesh) │
 * │ cil-code         │ Output    │ pbr         │ supershape-species (tokens)   │
 * │ cil-layers       │ Layers    │ matcap      │ fluid-surface (depth stack)   │
 * │ cil-loop         │ Loop/Ctrl │ iridescence │ turing-pattern (oscillation)  │
 * │ cil-graph        │ Topology  │ pbr         │ julia-background (graph field)│
 * └──────────────────┴───────────┴─────────────┴──────────────────────────────┘
 */
const SPECIES_SHADER_REGISTRY: Record<string, SpeciesShaderConfig> = {

  // ── cil-eye — Self-Attention / Query-Key-Value ────────────────────────────
  // Visual concept: iridescent iris, wide bloom halo, voronoi membrane texture.
  // iridescence communicates multi-head attention distributing over the sequence.
  //
  // M747: voronoi-membrane now renders soft F2-F1 translucent membranes with
  //   domain-warped texture, sin(time) breathing pulsation, and collision sparks.
  //   New uniforms consumed by the pattern shader:
  //     u_contactCount  — SPH contact count → collision spark brightness
  //     u_breathRate    — breathing pulsation frequency (rad/s), default 1.2
  //   These are fed from PhysicsUniforms (u_contactCount) and species config.
  'cil-eye': {
    id:   'cil-eye',
    role: 'Self-Attention (QKV)',

    sdfShape:  'flower',
    sdfParams: { petals: 6 },

    materialType: 'iridescence',
    materialParams: {
      albedo:        [0.36, 0.42, 0.75],  // indigo (#5C6BC0) linear
      metallic:      0.1,
      roughness:     0.35,
      ao:            0.9,
      fresnelPower:  3.5,
      fresnelColor:  [0.47, 0.53, 0.80],  // #7986CB
      iridStrength:  0.7,
      iridThickness: 420.0,               // nm — cyan–violet film band
      iridIOR:       1.5,
      normalStrength: 1.0,
    },

    patternShader: 'voronoi-membrane',

    bloomStrength:     2.0,
    bloomThreshold:    0.0,
    bloomRadius:       1.0,
    bloomPulseAmplitude: 0.25,
    bloomPulseFrequency: 1.2,

    physicsBindings: {
      // Dense neighbourhoods → brighter glow (dense attention context)
      bloomStrength:   { channel: 'u_density',      scale: 1.0, clampMin: 0.6, clampMax: 2.2, mode: 'multiply' },
      // High velocity → wider bloom radius (diffuse attention spread)
      bloomRadius:     { channel: 'speed',           scale: 0.6, clampMin: 0.8, clampMax: 1.4, mode: 'multiply' },
      // Vorticity → iridescent film thickness animation (dynamic attention)
      iridThickness:   { channel: 'u_vorticity',     scale: 80.0, clampMin: -120, clampMax: 120, mode: 'add' },
      // Kinetic energy → Fresnel rim intensity (activation energy)
      fresnelStrength: { channel: 'u_kineticEnergy', scale: 1.5, clampMin: 0.5, clampMax: 3.0, mode: 'multiply' },
      // Pattern animation speed driven by local flow speed
      patternSpeed:    { channel: 'speed',            scale: 2.0, clampMin: 0.2, clampMax: 4.0, mode: 'multiply' },
    },
  },

  // ── cil-bolt — Feed-Forward Network (FFN / MLP) ──────────────────────────
  // Visual concept: electric zigzag SDF, matcap metallic sheen, Gray-Scott
  // reaction–diffusion inside (spontaneous activation patterns).
  // Fast flicker bloom matches high-frequency FFN activations.
  'cil-bolt': {
    id:   'cil-bolt',
    role: 'Feed-Forward Network (FFN)',

    sdfShape:  'star',
    sdfParams: { starPoints: 5, starRatio: 0.42 },

    materialType: 'matcap',
    materialParams: {
      matcapSrc:     'room/matcap-test.jpg',
      tintColor:     [0.82, 0.65, 0.09],   // #FFA726 orange, linear approx
      tintStrength:  0.6,
      normalStrength: 0.19,
    },

    patternShader: 'grayscott-species',

    bloomStrength:     1.8,
    bloomThreshold:    0.1,
    bloomRadius:       0.75,
    bloomPulseAmplitude: 0.30,
    bloomPulseFrequency: 2.0,

    physicsBindings: {
      // Pressure spikes → bloom burst (FFN non-linearity activation)
      bloomStrength:   { channel: 'u_pressure',      scale: 0.8, clampMin: 0.8, clampMax: 3.5, mode: 'multiply' },
      bloomRadius:     { channel: 'u_kineticEnergy', scale: 0.5, clampMin: 0.6, clampMax: 1.2, mode: 'multiply' },
      // Pulse frequency driven by neighbour density (busy layer = faster flicker)
      pulseFrequency:  { channel: 'u_density',       scale: 1.5, clampMin: 1.0, clampMax: 4.0, mode: 'multiply' },
      // Reaction–diffusion speed driven by local velocity
      patternSpeed:    { channel: 'speed',            scale: 3.0, clampMin: 0.5, clampMax: 6.0, mode: 'multiply' },
      // Vorticity distorts the SDF silhouette (turbulent FFN)
      sdfDistort:      { channel: 'u_vorticity',     scale: 0.012, clampMin: 0.0, clampMax: 0.08, mode: 'add' },
    },
  },

  // ── cil-vector — Embedding / Positional Encoding ─────────────────────────
  // Visual concept: wide soft bloom, PBR green, IQ cosine-palette field
  // inside (smooth latent-space colour encoding).
  'cil-vector': {
    id:   'cil-vector',
    role: 'Embedding / Positional Encoding',

    sdfShape:  'supershape',
    sdfParams: { supershapeM: 8, supershapeN1: 3.0, supershapeN2: 4.0, supershapeN3: 4.0 },

    materialType: 'pbr',
    materialParams: {
      albedo:        [0.40, 0.73, 0.41],   // #66BB6A green, linear
      metallic:      0.0,
      roughness:     0.5,
      ao:            0.95,
      fresnelPower:  2.0,
      fresnelColor:  [0.50, 0.82, 0.52],   // #81C784
      normalStrength: 0.24,
    },

    patternShader: 'iq-palette-species',

    bloomStrength:     1.2,
    bloomThreshold:    0.0,
    bloomRadius:       0.80,
    bloomPulseAmplitude: 0.15,
    bloomPulseFrequency: 0.8,

    physicsBindings: {
      // Density scales bloom softly (embedding space occupancy)
      bloomStrength:   { channel: 'u_density',      scale: 1.0, clampMin: 0.7, clampMax: 1.8, mode: 'multiply' },
      bloomRadius:     { channel: 'speed',           scale: 0.4, clampMin: 0.8, clampMax: 1.3, mode: 'multiply' },
      // Palette contrast modulated by kinetic energy (feature activation magnitude)
      patternContrast: { channel: 'u_kineticEnergy', scale: 1.2, clampMin: 0.5, clampMax: 2.0, mode: 'multiply' },
      patternSpeed:    { channel: 'speed',            scale: 1.5, clampMin: 0.3, clampMax: 3.0, mode: 'multiply' },
      // Environment scale (PBR env diffuse) driven by neighbour count (context richness)
      materialEnvScale: { channel: 'u_neighborCount', scale: 0.05, clampMin: 0.5, clampMax: 2.0, mode: 'multiply' },
    },
  },

  // ── cil-plus — Add & LayerNorm ────────────────────────────────────────────
  // Visual concept: clean plus-cross SDF, minimal bloom, solid PBR surface.
  // Residual addition is architecturally simple — no loud visual language.
  'cil-plus': {
    id:   'cil-plus',
    role: 'Add & LayerNorm',

    sdfShape:  'polygon',
    sdfParams: { radius: 0.48 },

    materialType: 'pbr',
    materialParams: {
      albedo:        [0.93, 0.25, 0.48],   // #EC407A pink, linear
      metallic:      0.0,
      roughness:     0.7,
      ao:            1.0,
      fresnelPower:  1.5,
      fresnelColor:  [0.96, 0.56, 0.69],   // #F48FB1
    },

    patternShader: 'none',

    bloomStrength:     0.3,
    bloomThreshold:    0.0,
    bloomRadius:       0.20,
    bloomPulseAmplitude: 0.08,
    bloomPulseFrequency: 0.6,

    physicsBindings: {
      // Gentle density modulation only — normalisation keeps things stable
      bloomStrength:   { channel: 'u_density',      scale: 0.6, clampMin: 0.2, clampMax: 0.8, mode: 'multiply' },
      // Contact events (residual collisions) briefly flash the rim
      fresnelStrength: { channel: 'u_contactCount', scale: 0.4, clampMin: 1.0, clampMax: 2.5, mode: 'multiply' },
      // Opacity barely shifts with pressure (stable normalisation)
      opacity:         { channel: 'u_pressure',     scale: 0.1, clampMin: 0.85, clampMax: 1.0, mode: 'multiply' },
    },
  },

  // ── cil-arrow-right — Skip Connection / Routing ──────────────────────────
  // Visual concept: directional chevron SDF, matcap cool grey, curl-trail
  // particle field inside (routing / passthrough information flow).
  'cil-arrow-right': {
    id:   'cil-arrow-right',
    role: 'Skip Connection / Routing',

    sdfShape:  'capsule',
    sdfParams: {},

    materialType: 'matcap',
    materialParams: {
      matcapSrc:     'room/matcap-test.jpg',
      tintColor:     [0.47, 0.56, 0.61],   // #78909C slate, linear
      tintStrength:  0.45,
      normalStrength: 1.0,
    },

    patternShader: 'curl-trail',

    bloomStrength:     0.6,
    bloomThreshold:    0.0,
    bloomRadius:       0.50,
    bloomPulseAmplitude: 0.12,
    bloomPulseFrequency: 0.9,

    physicsBindings: {
      // Speed directly drives bloom (fast routing = brighter signal)
      bloomStrength:   { channel: 'speed',           scale: 1.2, clampMin: 0.3, clampMax: 1.5, mode: 'multiply' },
      // Curl-trail animation speed = velocity magnitude (shows information flow)
      patternSpeed:    { channel: 'speed',            scale: 4.0, clampMin: 0.2, clampMax: 8.0, mode: 'multiply' },
      // Vorticity steers trail curl angle
      sdfDistort:      { channel: 'u_vorticity',     scale: 0.008, clampMin: 0.0, clampMax: 0.05, mode: 'add' },
      bloomRadius:     { channel: 'speed',           scale: 0.5, clampMin: 0.4, clampMax: 1.0, mode: 'multiply' },
    },
  },

  // ── cil-filter — Attention Mask / Selection Gate ─────────────────────────
  // Visual concept: hexagonal SDF, PBR violet, voronoi-natural stipple
  // inside (irregular filter mesh — only some tokens pass).
  'cil-filter': {
    id:   'cil-filter',
    role: 'Attention Mask / Selection Gate',

    sdfShape:  'hexagon',
    sdfParams: { radius: 0.46 },

    materialType: 'pbr',
    materialParams: {
      albedo:        [0.67, 0.28, 0.74],   // #AB47BC violet, linear
      metallic:      0.15,
      roughness:     0.45,
      ao:            0.85,
      fresnelPower:  2.8,
      fresnelColor:  [0.81, 0.58, 0.85],   // #CE93D8
    },

    patternShader: 'voronoi-natural',

    bloomStrength:     0.8,
    bloomThreshold:    0.0,
    bloomRadius:       0.50,
    bloomPulseAmplitude: 0.15,
    bloomPulseFrequency: 1.0,

    physicsBindings: {
      bloomStrength:   { channel: 'u_density',       scale: 0.9, clampMin: 0.5, clampMax: 1.6, mode: 'multiply' },
      // Pressure gates pattern contrast (selective filtering intensity)
      patternContrast: { channel: 'u_pressure',      scale: 1.0, clampMin: 0.3, clampMax: 2.5, mode: 'multiply' },
      patternSpeed:    { channel: 'speed',            scale: 1.8, clampMin: 0.4, clampMax: 3.5, mode: 'multiply' },
      // Neighbour count modulates environment scale (context breadth of filter)
      materialEnvScale: { channel: 'u_neighborCount', scale: 0.04, clampMin: 0.6, clampMax: 1.8, mode: 'multiply' },
    },
  },

  // ── cil-code — Output Projection / Linear Head ───────────────────────────
  // Visual concept: rounded-box SDF, teal PBR, supershape-species token
  // lattice pattern (discrete output vocabulary).
  'cil-code': {
    id:   'cil-code',
    role: 'Output Projection / Linear Head',

    sdfShape:  'roundbox',
    sdfParams: { cornerRadius: 0.12 },

    materialType: 'pbr',
    materialParams: {
      albedo:        [0.15, 0.65, 0.60],   // #26A69A teal, linear
      metallic:      0.0,
      roughness:     0.55,
      ao:            0.92,
      fresnelPower:  2.0,
      fresnelColor:  [0.50, 0.79, 0.77],   // #80CBC4
    },

    patternShader: 'supershape-species',

    bloomStrength:     0.7,
    bloomThreshold:    0.2,
    bloomRadius:       0.44,
    bloomPulseAmplitude: 0.10,
    bloomPulseFrequency: 0.7,

    physicsBindings: {
      bloomStrength:   { channel: 'u_density',       scale: 0.8, clampMin: 0.5, clampMax: 1.4, mode: 'multiply' },
      // Token lattice density driven by neighbour count (vocabulary coverage)
      patternContrast: { channel: 'u_neighborCount', scale: 0.06, clampMin: 0.4, clampMax: 2.0, mode: 'multiply' },
      patternSpeed:    { channel: 'speed',            scale: 1.2, clampMin: 0.2, clampMax: 2.5, mode: 'multiply' },
      opacity:         { channel: 'u_pressure',      scale: 0.08, clampMin: 0.88, clampMax: 1.0, mode: 'multiply' },
    },
  },

  // ── cil-layers — Layer Stack / Residual Stream ────────────────────────────
  // Visual concept: Koch-fractal SDF (stacked edges), matcap blue, fluid-surface
  // normal-map inside (depth layering metaphor).
  'cil-layers': {
    id:   'cil-layers',
    role: 'Layer Stack / Residual Stream',

    sdfShape:  'koch',
    sdfParams: { kochIterations: 2 },

    materialType: 'matcap',
    materialParams: {
      matcapSrc:     'home/matcap3.jpg',
      tintColor:     [0.26, 0.65, 0.96],   // #42A5F5 blue, linear
      tintStrength:  0.55,
      normalStrength: 1.0,
    },

    patternShader: 'fluid-surface',

    bloomStrength:     1.0,
    bloomThreshold:    0.2,
    bloomRadius:       0.625,
    bloomPulseAmplitude: 0.12,
    bloomPulseFrequency: 0.85,

    physicsBindings: {
      bloomStrength:   { channel: 'u_density',      scale: 1.0, clampMin: 0.6, clampMax: 1.8, mode: 'multiply' },
      // Fluid surface animation speed = velocity magnitude
      patternSpeed:    { channel: 'speed',           scale: 2.5, clampMin: 0.3, clampMax: 5.0, mode: 'multiply' },
      // Pressure drives distortion of fluid normals (compression within the stack)
      sdfDistort:      { channel: 'u_pressure',     scale: 0.01, clampMin: 0.0, clampMax: 0.06, mode: 'add' },
      bloomRadius:     { channel: 'u_kineticEnergy', scale: 0.4, clampMin: 0.5, clampMax: 1.1, mode: 'multiply' },
    },
  },

  // ── cil-loop — Loop / Control Flow / Recurrence ──────────────────────────
  // Visual concept: flower SDF (cyclic petal), iridescence warm gold,
  // Turing-pattern oscillation inside (oscillatory control signal).
  'cil-loop': {
    id:   'cil-loop',
    role: 'Loop / Recurrent Control Flow',

    sdfShape:  'flower',
    sdfParams: { petals: 4 },

    materialType: 'iridescence',
    materialParams: {
      albedo:        [0.99, 0.79, 0.16],   // #FFCA28 amber, linear
      metallic:      0.05,
      roughness:     0.40,
      ao:            0.88,
      fresnelPower:  3.0,
      fresnelColor:  [1.0, 0.88, 0.51],    // #FFE082
      iridStrength:  0.45,
      iridThickness: 380.0,                // nm — warm gold–green film band
      iridIOR:       1.45,
    },

    patternShader: 'turing-pattern',

    bloomStrength:     0.8,
    bloomThreshold:    0.0,
    bloomRadius:       0.70,
    bloomPulseAmplitude: 0.18,
    bloomPulseFrequency: 1.4,

    physicsBindings: {
      bloomStrength:   { channel: 'u_density',      scale: 1.0, clampMin: 0.5, clampMax: 1.6, mode: 'multiply' },
      // Vorticity drives loop oscillation frequency (recurrence cycle)
      patternSpeed:    { channel: 'u_vorticity',    scale: 0.05, clampMin: 0.5, clampMax: 3.0, mode: 'add' },
      pulseFrequency:  { channel: 'u_kineticEnergy', scale: 1.0, clampMin: 0.8, clampMax: 2.5, mode: 'multiply' },
      // Iridescent thickness animated by vortex spin
      iridThickness:   { channel: 'u_vorticity',   scale: 60.0, clampMin: -100, clampMax: 100, mode: 'add' },
      bloomRadius:     { channel: 'speed',          scale: 0.5, clampMin: 0.6, clampMax: 1.2, mode: 'multiply' },
    },
  },

  // ── cil-graph — Graph / Topology / Cross-Attention ───────────────────────
  // Visual concept: Julia-set SDF boundary (complex fractal topology), minimal
  // PBR slate, julia-background escape field inside (complex graph structure).
  'cil-graph': {
    id:   'cil-graph',
    role: 'Graph / Topology / Cross-Attention',

    sdfShape:  'julia',
    sdfParams: { juliaRe: -0.7, juliaIm: 0.27015 },

    materialType: 'pbr',
    materialParams: {
      albedo:        [0.47, 0.56, 0.61],   // #78909C slate, linear
      metallic:      0.2,
      roughness:     0.6,
      ao:            0.90,
      fresnelPower:  1.8,
      fresnelColor:  [0.69, 0.76, 0.80],   // #B0BEC5
    },

    patternShader: 'julia-background',

    bloomStrength:     0.5,
    bloomThreshold:    0.0,
    bloomRadius:       0.50,
    bloomPulseAmplitude: 0.08,
    bloomPulseFrequency: 0.6,

    physicsBindings: {
      // Graph complexity scales with neighbourhood density
      bloomStrength:   { channel: 'u_density',       scale: 0.7, clampMin: 0.3, clampMax: 1.2, mode: 'multiply' },
      // Julia set escape speed driven by kinetic energy (graph dynamics)
      patternSpeed:    { channel: 'u_kineticEnergy', scale: 2.0, clampMin: 0.2, clampMax: 4.0, mode: 'multiply' },
      // Vorticity slightly distorts the Julia SDF (topology warping)
      sdfDistort:      { channel: 'u_vorticity',    scale: 0.006, clampMin: 0.0, clampMax: 0.04, mode: 'add' },
      // Pattern contrast driven by pressure (connection density)
      patternContrast: { channel: 'u_pressure',     scale: 0.9, clampMin: 0.4, clampMax: 1.8, mode: 'multiply' },
    },
  },
};

// ─── Fallback ──────────────────────────────────────────────────────────────────

/** Fallback config for unknown / future species. */
const DEFAULT_SHADER_CONFIG: SpeciesShaderConfig = {
  id:   '__default__',
  role: 'Unknown',

  sdfShape:  'flower',
  sdfParams: { petals: 3 },

  materialType: 'pbr',
  materialParams: {
    albedo:    [0.5, 0.5, 0.5],
    metallic:  0.0,
    roughness: 0.6,
    ao:        1.0,
  },

  patternShader: 'none',

  bloomStrength:     1.0,
  bloomThreshold:    0.1,
  bloomRadius:       0.5,
  bloomPulseAmplitude: 0.15,
  bloomPulseFrequency: 1.0,

  physicsBindings: {
    bloomStrength: { channel: 'u_density', scale: 1.0, clampMin: 0.5, clampMax: 1.5, mode: 'multiply' },
  },
};

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Retrieve the SpeciesShaderConfig for a given species id.
 * Falls back to DEFAULT_SHADER_CONFIG for unknown species.
 *
 * @example
 *   const cfg = getSpeciesShaderConfig('cil-eye');
 *   // cfg.materialType === 'iridescence'
 *   // cfg.patternShader === 'voronoi-membrane'
 */
export function getSpeciesShaderConfig(species: string): SpeciesShaderConfig {
  return SPECIES_SHADER_REGISTRY[species] ?? DEFAULT_SHADER_CONFIG;
}

/**
 * Return an array of all registered species ids, in registry order.
 * Equivalent to SPECIES_ORDER in cell-body-bridge.ts (index 1-based).
 */
export function getAllSpeciesIds(): string[] {
  return Object.keys(SPECIES_SHADER_REGISTRY);
}

/**
 * Filter the registry to only species using a particular material type.
 *
 * @example
 *   const iridSpecies = getSpeciesByMaterial('iridescence');
 *   // → ['cil-eye', 'cil-loop']
 */
export function getSpeciesByMaterial(type: MaterialType): string[] {
  return Object.values(SPECIES_SHADER_REGISTRY)
    .filter(cfg => cfg.materialType === type)
    .map(cfg => cfg.id);
}

/**
 * Filter the registry to species using a particular pattern shader.
 *
 * @example
 *   const reactionSpecies = getSpeciesByPattern('grayscott-species');
 *   // → ['cil-bolt']
 */
export function getSpeciesByPattern(shader: PatternShader): string[] {
  return Object.values(SPECIES_SHADER_REGISTRY)
    .filter(cfg => cfg.patternShader === shader)
    .map(cfg => cfg.id);
}

/**
 * Evaluate the physics bindings for a species against a PhysicsUniforms snapshot.
 *
 * Returns a partial record of resolved visual values:
 *   { bloomStrength: <runtime>, bloomRadius: <runtime>, … }
 *
 * The consumer is responsible for applying the resolved values to the
 * corresponding shader uniforms or AdvancedBloomFilter params.
 *
 * @param species    Species id string.
 * @param physics    Live PhysicsUniforms from physics-uniform-bridge.ts.
 * @returns          Record of resolved visual target values.
 */
export function resolvePhysicsBindings(
  species: string,
  physics: PhysicsUniforms,
): Partial<Record<keyof PhysicsBindings, number>> {
  const cfg = getSpeciesShaderConfig(species);
  const resolved: Partial<Record<keyof PhysicsBindings, number>> = {};

  const MAX_SPEED = 10.0; // world units/s — normalisation denominator

  // Derive 'speed' pseudo-channel from velocity magnitude
  const [vx, vy] = physics.u_velocity;
  const speed = Math.sqrt(vx * vx + vy * vy) / MAX_SPEED;

  for (const [visualKey, binding] of Object.entries(cfg.physicsBindings) as
    [keyof PhysicsBindings, PhysicsBinding][]) {

    if (!binding) continue;

    // Read raw channel value
    let rawValue: number;
    if (binding.channel === 'speed') {
      rawValue = speed;
    } else {
      rawValue = physics[binding.channel as keyof PhysicsUniforms] as number;
    }

    // Scale and clamp
    const scaled  = rawValue * binding.scale;
    const clamped = Math.max(
      binding.clampMin ?? 0,
      Math.min(binding.clampMax ?? 1, scaled),
    );

    // Apply mode against the config base value
    const mode = binding.mode ?? 'multiply';
    let baseValue = _baseValueForVisualTarget(cfg, visualKey);

    let result: number;
    if (mode === 'multiply') {
      result = baseValue * clamped;
    } else if (mode === 'add') {
      result = baseValue + clamped;
    } else {
      // lerp
      const target = binding.lerpTarget ?? baseValue * 1.5;
      result = baseValue + (target - baseValue) * clamped;
    }

    resolved[visualKey] = result;
  }

  return resolved;
}

/**
 * Internal helper: return the base (design-time) value for a visual target
 * from the species config.  Used as the left-hand operand in binding evaluation.
 */
function _baseValueForVisualTarget(
  cfg: SpeciesShaderConfig,
  target: keyof PhysicsBindings,
): number {
  switch (target) {
    case 'bloomStrength':    return cfg.bloomStrength;
    case 'bloomRadius':      return cfg.bloomRadius;
    case 'pulseFrequency':   return cfg.bloomPulseFrequency;
    case 'fresnelStrength':  return cfg.materialParams.fresnelPower ?? 2.0;
    case 'iridThickness':    return cfg.materialParams.iridThickness ?? 400.0;
    case 'materialEnvScale': return 1.0;
    case 'sdfDistort':       return 0.0;
    case 'patternSpeed':     return 1.0;
    case 'patternContrast':  return 1.0;
    case 'opacity':          return 1.0;
    default:                 return 1.0;
  }
}

// ─── Re-exports ────────────────────────────────────────────────────────────────

export { SPECIES_SHADER_REGISTRY, DEFAULT_SHADER_CONFIG };
