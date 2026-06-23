/**
 * shader-tuning-presets.ts — M786: Shader Tuning Presets per Species
 * ─────────────────────────────────────────────────────────────────────────────
 * Hand-tuned shader parameter presets for each transformer-layer species.
 *
 * AT's visual beauty comes from careful parameter tuning — bloom thresholds,
 * PBR surface properties, reaction-diffusion constants, volumetric light, and
 * screen-space effects all interact to give each species a unique material
 * personality.  This module codifies those tuning decisions as static presets
 * that downstream consumers (at-bloom-postprocess, at-pbr-material,
 * reaction-diffusion, water-caustics, curl-flow-field, god-rays,
 * ambient-occlusion) can look up by species key.
 *
 * Species ↔ cil-* mapping
 * ─────────────────────────────────────────────────────────────────────────────
 *   input_embed  → cil-vector    Embedding layer input
 *   self_attn    → cil-eye       Self-Attention (QKV)
 *   ffn          → cil-bolt      Feed-Forward Network (MLP)
 *   add_norm1    → cil-plus      Add & LayerNorm (post-attention)
 *   add_norm2    → cil-plus      Add & LayerNorm (post-FFN)
 *   pos_encode   → cil-vector    Positional Encoding overlay
 *   output       → cil-code      Output Projection / Linear Head
 *
 * Parameter ranges (AT UIL reference)
 * ─────────────────────────────────────────────────────────────────────────────
 *   bloom_threshold     0.0 – 1.5   (luminosity cutoff for bright-pass)
 *   bloom_intensity     0.0 – 3.0   (bloomScale in composite blend)
 *   pbr_roughness       0.0 – 1.0   (GGX roughness)
 *   pbr_metallic        0.0 – 1.0   (metalness)
 *   caustics_speed      0.0 – 2.0   (water surface evolution speed)
 *   curl_scale          0.0 – 5.0   (curl-noise velocity magnitude)
 *   reaction_f          0.0 – 0.1   (Gray-Scott feed rate)
 *   reaction_k          0.0 – 0.1   (Gray-Scott kill rate)
 *   water_normal_scale  0.0 – 3.0   (height-to-normal derivative scale)
 *   godray_intensity    0.0 – 2.0   (volumetric light global multiplier)
 *   ssao_radius         0.05 – 2.0  (hemisphere sample radius)
 *
 * Upstream
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/at-bloom-postprocess.ts   — bloom threshold / bloomScale
 *   src/lib/sph/at-pbr-material.ts        — roughness, metallic, GGX BRDF
 *   src/lib/sph/water-caustics.ts         — caustics surface sim
 *   src/lib/sph/curl-flow-field.ts        — curl-noise velocity field
 *   src/lib/sph/reaction-diffusion.ts     — Gray-Scott f/k constants
 *   src/lib/sph/at-water-surface.ts       — water normal computation
 *   src/lib/sph/god-rays.ts               — volumetric light rays
 *   src/lib/sph/ambient-occlusion.ts      — SSAO kernel radius
 *   src/lib/sph/species-shader-registry.ts — per-species shader stack
 */

// ─── Interface ────────────────────────────────────────────────────────────────

/**
 * Complete shader tuning preset for a single transformer-layer species.
 *
 * Every field is a concrete numeric value (no ranges or functions) — this is
 * the "artist's final answer" for that species.  Runtime modulation by physics
 * state happens downstream in uil-species-live.ts; these are the baselines.
 */
export interface ShaderPreset {
  // ── Bloom (at-bloom-postprocess) ──────────────────────────────────────────
  /** Luminosity threshold for the bright-pass extraction (0 = everything blooms). */
  bloom_threshold: number;
  /** Bloom scale multiplier in the composite blend (0 = no glow). */
  bloom_intensity: number;

  // ── PBR Surface (at-pbr-material) ─────────────────────────────────────────
  /** GGX roughness: 0 = mirror, 1 = fully diffuse. */
  pbr_roughness: number;
  /** Metallic factor: 0 = dielectric, 1 = conductor. */
  pbr_metallic: number;

  // ── Water / Caustics (water-caustics, at-water-surface) ───────────────────
  /** Time-scale multiplier for caustics pattern evolution. */
  caustics_speed: number;
  /** Curl-noise velocity magnitude scale (curl-flow-field). */
  curl_scale: number;

  // ── Reaction-Diffusion (reaction-diffusion) ───────────────────────────────
  /** Gray-Scott feed rate f — controls replenishment of chemical u. */
  reaction_f: number;
  /** Gray-Scott kill rate k — controls removal of chemical v. */
  reaction_k: number;

  // ── Water Normals (at-water-surface) ──────────────────────────────────────
  /** Height-to-normal derivative scale for water surface perturbation. */
  water_normal_scale: number;

  // ── Volumetric Light (god-rays) ───────────────────────────────────────────
  /** Global intensity multiplier for additive god-ray composite. */
  godray_intensity: number;

  // ── Screen-Space AO (ambient-occlusion) ───────────────────────────────────
  /** SSAO hemisphere sample kernel radius. */
  ssao_radius: number;
}

// ─── Species type ─────────────────────────────────────────────────────────────

/**
 * Transformer-layer species identifiers used as preset keys.
 *
 * These are logical layer roles, not the cil-* particle IDs.  The mapping
 * to cil-* is documented above; downstream consumers can use
 * `SPECIES_TO_CIL` for the reverse lookup.
 */
export type TransformerSpecies =
  | 'input_embed'
  | 'self_attn'
  | 'ffn'
  | 'add_norm1'
  | 'add_norm2'
  | 'pos_encode'
  | 'output';

/** Map from transformer species key to cil-* particle species ID. */
export const SPECIES_TO_CIL: Record<TransformerSpecies, string> = {
  input_embed: 'cil-vector',
  self_attn:   'cil-eye',
  ffn:         'cil-bolt',
  add_norm1:   'cil-plus',
  add_norm2:   'cil-plus',
  pos_encode:  'cil-vector',
  output:      'cil-code',
};

// ─── Presets ──────────────────────────────────────────────────────────────────

/**
 * Shader tuning presets keyed by transformer-layer species.
 *
 * Design intent per species:
 *
 *   input_embed  — Soft blue scattered light.  Tokens entering the model are
 *                  raw, unprocessed — the material is rough and diffuse like
 *                  frosted glass.  Low bloom keeps it understated; high
 *                  roughness scatters incoming light into a gentle halo.
 *
 *   self_attn    — Intense golden metallic.  Attention is the "hero" operation;
 *                  high bloom and low roughness create sharp specular highlights
 *                  that read as liquid gold.  Strong caustics and god-rays add
 *                  the dramatic volumetric quality of light bending through a
 *                  dense attention matrix.
 *
 *   ffn          — Green organic / coral.  The feed-forward network is the
 *                  model's nonlinear growth — reaction-diffusion patterns form
 *                  coral-like structures (Gray-Scott mitosis regime f≈0.0545,
 *                  k≈0.062).  Medium bloom; moderate roughness keeps the
 *                  surface alive without overwhelming the pattern detail.
 *
 *   add_norm1/2  — Pure white crystalline.  Add & LayerNorm is the residual
 *                  stream — clean, transparent, stabilising.  Near-zero
 *                  roughness creates a glass-like surface; strong god-rays
 *                  convey the sense of information flowing through a prism.
 *                  Minimal reaction-diffusion (barely alive, steady state).
 *
 *   pos_encode   — Rainbow iridescent flow.  Positional encoding is periodic
 *                  and multi-frequency — curl noise at high scale creates
 *                  visible flow lanes that suggest sine/cosine bands.  Medium
 *                  bloom with balanced PBR lets the colour gradient dominate.
 *
 *   output       — Red-hot energy burst.  The output projection is the final
 *                  discharge — maximum bloom, strong metallic sheen, fast
 *                  caustics, and aggressive reaction-diffusion (high kill rate
 *                  → transient spark patterns).  This is the most energetic
 *                  visual in the pipeline.
 */
export const SHADER_PRESETS: Record<TransformerSpecies, ShaderPreset> = {

  // ── input_embed ─────────────────────────────────────────────────────────────
  // Soft blue glow · scattered diffuse light · frosted glass
  // cil-vector (Embedding layer)
  input_embed: {
    bloom_threshold:    0.75,     // only the brightest highlights bloom
    bloom_intensity:    0.45,     // gentle, understated glow
    pbr_roughness:      0.82,     // highly diffuse — scattered light
    pbr_metallic:       0.08,     // almost purely dielectric
    caustics_speed:     0.30,     // slow, dreamy water surface
    curl_scale:         0.60,     // faint ambient turbulence
    reaction_f:         0.037,    // steady-state spots (no wild growth)
    reaction_k:         0.060,    // balanced decay
    water_normal_scale: 0.80,     // subtle surface ripples
    godray_intensity:   0.25,     // faint volumetric fill light
    ssao_radius:        0.45,     // medium AO — soft contact shadows
  },

  // ── self_attn ───────────────────────────────────────────────────────────────
  // Intense golden metallic · liquid-gold specular · dramatic volumetrics
  // cil-eye (Self-Attention QKV)
  self_attn: {
    bloom_threshold:    0.30,     // low threshold — lots of glow
    bloom_intensity:    2.20,     // strong, saturated bloom halo
    pbr_roughness:      0.12,     // near-mirror — sharp specular highlights
    pbr_metallic:       0.92,     // almost fully metallic conductor
    caustics_speed:     1.20,     // fast, shimmering caustic dance
    curl_scale:         1.80,     // visible curl-noise advection trails
    reaction_f:         0.042,    // moderate pattern formation
    reaction_k:         0.063,    // slight coral branching
    water_normal_scale: 1.80,     // pronounced refractive distortion
    godray_intensity:   1.40,     // dramatic volumetric light shafts
    ssao_radius:        0.30,     // tight AO — crisp metallic crevices
  },

  // ── ffn ─────────────────────────────────────────────────────────────────────
  // Green organic coral · reaction-diffusion mitosis · living surface
  // cil-bolt (Feed-Forward Network / MLP)
  ffn: {
    bloom_threshold:    0.55,     // medium — pattern details still visible
    bloom_intensity:    1.10,     // warm organic glow, not blinding
    pbr_roughness:      0.55,     // semi-rough — organic, not glassy
    pbr_metallic:       0.15,     // mostly dielectric (biological)
    caustics_speed:     0.65,     // moderate — organic movement rhythm
    curl_scale:         1.20,     // visible flow through the network
    reaction_f:         0.0545,   // Gray-Scott mitosis regime — coral growth
    reaction_k:         0.062,    // balanced kill → branching dendrites
    water_normal_scale: 1.20,     // noticeable but not overwhelming
    godray_intensity:   0.50,     // subtle volumetric depth
    ssao_radius:        0.60,     // wider AO — soft organic shadows
  },

  // ── add_norm1 ───────────────────────────────────────────────────────────────
  // Pure white crystalline · glass-like clarity · strong volumetric prism
  // cil-plus (Add & LayerNorm, post-attention residual)
  add_norm1: {
    bloom_threshold:    0.85,     // high threshold — only pure white peaks
    bloom_intensity:    0.70,     // clean, controlled luminance
    pbr_roughness:      0.05,     // near-perfect mirror — crystal clarity
    pbr_metallic:       0.25,     // slight metallic to catch env reflections
    caustics_speed:     0.40,     // slow, meditative caustic drift
    curl_scale:         0.35,     // minimal turbulence — stable residual
    reaction_f:         0.030,    // nearly dormant — steady-state equilibrium
    reaction_k:         0.057,    // low kill → sparse, delicate spots
    water_normal_scale: 0.50,     // very subtle surface perturbation
    godray_intensity:   1.60,     // strong god-rays — prismatic light beams
    ssao_radius:        0.20,     // tight — glass has crisp edges
  },

  // ── add_norm2 ───────────────────────────────────────────────────────────────
  // Pure white crystalline · second residual pass · slightly warmer god-rays
  // cil-plus (Add & LayerNorm, post-FFN residual)
  add_norm2: {
    bloom_threshold:    0.80,     // marginally lower than norm1 (more signal)
    bloom_intensity:    0.80,     // slightly brighter — accumulated residual
    pbr_roughness:      0.07,     // still very smooth, fractionally softer
    pbr_metallic:       0.28,     // tiny bit more reflective
    caustics_speed:     0.45,     // gentle drift, slightly faster than norm1
    curl_scale:         0.40,     // minimal ambient curl
    reaction_f:         0.032,    // barely alive pattern
    reaction_k:         0.058,    // stable decay
    water_normal_scale: 0.55,     // subtle
    godray_intensity:   1.75,     // strongest god-rays — full residual stream
    ssao_radius:        0.22,     // tight, crystalline
  },

  // ── pos_encode ──────────────────────────────────────────────────────────────
  // Rainbow iridescent flow · curl-noise flow lanes · periodic sine/cos bands
  // cil-vector (Positional Encoding overlay)
  pos_encode: {
    bloom_threshold:    0.50,     // medium — let colour gradients bloom
    bloom_intensity:    1.30,     // visible halo around spectral bands
    pbr_roughness:      0.40,     // balanced — colour-shifting needs some spec
    pbr_metallic:       0.45,     // moderate metallic → thin-film iridescence
    caustics_speed:     0.85,     // rhythmic, periodic movement
    curl_scale:         3.20,     // high! visible flow lanes = frequency bands
    reaction_f:         0.040,    // gentle pattern — subordinate to colour
    reaction_k:         0.061,    // mild coral texture underneath
    water_normal_scale: 1.50,     // refractive rainbow distortion
    godray_intensity:   0.70,     // moderate volumetric — not the star here
    ssao_radius:        0.50,     // medium AO
  },

  // ── output ──────────────────────────────────────────────────────────────────
  // Red-hot energy burst · maximum bloom · collision sparks · transient fire
  // cil-code (Output Projection / Linear Head)
  output: {
    bloom_threshold:    0.15,     // very low — almost everything blooms
    bloom_intensity:    2.80,     // maximum bloom — blazing energy discharge
    pbr_roughness:      0.20,     // fairly smooth — hot metal sheen
    pbr_metallic:       0.78,     // strongly metallic — molten conductor
    caustics_speed:     1.80,     // fast, aggressive caustic turbulence
    curl_scale:         2.50,     // high curl → chaotic energy field
    reaction_f:         0.060,    // high feed → rapid transient sparks
    reaction_k:         0.068,    // high kill → patterns flash and die (sparks)
    water_normal_scale: 2.20,     // aggressive refractive distortion
    godray_intensity:   1.20,     // strong volumetrics — discharge glow
    ssao_radius:        0.35,     // moderate — energy has depth
  },
};

// ─── Accessors ────────────────────────────────────────────────────────────────

/**
 * Retrieve the shader tuning preset for a transformer-layer species.
 *
 * Returns a *copy* so callers can mutate without affecting the canonical preset.
 *
 * @param species - One of the seven transformer-layer species keys.
 * @returns A fresh ShaderPreset object.
 *
 * @example
 *   const p = getShaderPreset('self_attn');
 *   bloom.setParams({ threshold: p.bloom_threshold, bloomScale: p.bloom_intensity });
 *   material.setParams({ roughness: p.pbr_roughness, metallic: p.pbr_metallic });
 */
export function getShaderPreset(species: TransformerSpecies): ShaderPreset {
  return { ...SHADER_PRESETS[species] };
}

/**
 * Look up the cil-* particle ID for a transformer species key.
 *
 * @example
 *   getCilSpecies('self_attn')  // → 'cil-eye'
 *   getCilSpecies('ffn')        // → 'cil-bolt'
 */
export function getCilSpecies(species: TransformerSpecies): string {
  return SPECIES_TO_CIL[species];
}

/**
 * All transformer species keys, in pipeline order.
 *
 * Useful for iterating the full preset table in logical sequence:
 * input → positional encoding → attention → add&norm → FFN → add&norm → output.
 */
export const SPECIES_PIPELINE_ORDER: readonly TransformerSpecies[] = [
  'input_embed',
  'pos_encode',
  'self_attn',
  'add_norm1',
  'ffn',
  'add_norm2',
  'output',
] as const;

/**
 * Interpolate between two ShaderPresets by a 0–1 factor.
 *
 * Useful for smooth visual transitions during layer-to-layer animations,
 * epoch transitions, or LOD blending.
 *
 * @param a - Source preset.
 * @param b - Target preset.
 * @param t - Interpolation factor (0 = fully a, 1 = fully b).
 * @returns A new ShaderPreset with linearly interpolated values.
 */
export function lerpPreset(a: ShaderPreset, b: ShaderPreset, t: number): ShaderPreset {
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
  const ct = clamp01(t);
  const mix = (x: number, y: number) => x + (y - x) * ct;

  return {
    bloom_threshold:    mix(a.bloom_threshold,    b.bloom_threshold),
    bloom_intensity:    mix(a.bloom_intensity,    b.bloom_intensity),
    pbr_roughness:      mix(a.pbr_roughness,      b.pbr_roughness),
    pbr_metallic:       mix(a.pbr_metallic,       b.pbr_metallic),
    caustics_speed:     mix(a.caustics_speed,     b.caustics_speed),
    curl_scale:         mix(a.curl_scale,         b.curl_scale),
    reaction_f:         mix(a.reaction_f,         b.reaction_f),
    reaction_k:         mix(a.reaction_k,         b.reaction_k),
    water_normal_scale: mix(a.water_normal_scale, b.water_normal_scale),
    godray_intensity:   mix(a.godray_intensity,   b.godray_intensity),
    ssao_radius:        mix(a.ssao_radius,        b.ssao_radius),
  };
}
