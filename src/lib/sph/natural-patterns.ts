// src/lib/sph/natural-patterns.ts
//
// Natural cell-surface textures via WebGPU compute shaders.
// Voronoi + Worley WGSL ported from lygia (Patricio Gonzalez Vivo):
//   upstream/lygia/generative/voronoi.wgsl
//   upstream/lygia/generative/worley.wgsl
//
// fBm implementation modelled after:
//   upstream/lygia/generative/fbm.wgsl  (Patricio Gonzalez Vivo)
//
// Voronoise (fBm-warped Voronoi) based on:
//   Inigo Quilez – "Voronoise" (2014) https://iquilezles.org/articles/voronoise/
//
// Outputs a GPU texture whose RGBA channels encode:
//   R – base pattern intensity (voronoi cell distance or worley F1)
//   G – secondary distance (voronoi centroid Y or worley F2)
//   B – edge/ridge mask (|F2-F1| for worley, boundary highlight for voronoi)
//   A – species-tinted hue selector (mapped from NaturalPatternParams.species)
//
// Usage:
//   const gen = new NaturalPatternGenerator(device);
//   const tex = await gen.generate({ width: 512, height: 512, scale: 4, ... });
//   // bind tex to your render pipeline as a sampled texture

// ─── Species → pattern-mode mapping ──────────────────────────────────────────
// Each Transformer cell species gets a visual metaphor:
//   CELL_DIVISION   – Voronoi cells expanding from centroids        (cil-eye, cil-bolt)
//   TORTOISE_SHELL  – Worley F2–F1 ridges (hexagonal cracking)      (cil-vector, cil-plus)
//   LEAF_VEIN       – Multi-octave Worley F1 (branching veins)      (cil-arrow-right, cil-filter)
//   FOAM            – Voronoi + Worley blend (soap-bubble foam)      (cil-layers, cil-loop)
//   SCALES          – Voronoi with distance modulation               (cil-code, cil-graph)
//   FBM_VORONOISE   – fBm-warped Voronoi (domain-warped noise blend) (fluid, cil-star, cil-drop)








export type NaturalPatternMode =
  | 'CELL_DIVISION'
  | 'TORTOISE_SHELL'
  | 'LEAF_VEIN'
  | 'FOAM'
  | 'SCALES'
  | 'FBM_VORONOISE';

// ─── Per-species fBm + voronoise parameters ───────────────────────────────────
/**
 * Fine-grained parameters that control the fBm octave stack and
 * voronoise blend weight for a given cell species.
 *
 * These travel to the GPU as a second uniform buffer (FbmParams, 32 bytes).
 */
export interface SpeciesFbmParams {
  /** Number of fBm octaves stacked on top of the base pattern. Range 1–6. */
  fbmOctaves: number;
  /** Amplitude persistence per octave (0.3 = rough, 0.7 = smooth). Default 0.5. */
  persistence: number;
  /** Frequency lacunarity per octave. Default 2.0 (classic fBm). */
  lacunarity: number;
  /** Domain-warp strength applied before voronoi lookup. 0 = no warp. */
  warpStrength: number;
  /** Mix weight between raw voronoi (0) and noise-only (1) in voronoise. */
  voronoiseBlend: number;
  /** Voronoise smoothing kernel radius (k in IQ's formula). Range 0.01–4.0. */
  smoothK: number;
}

/** Complete resolved parameters for a species, ready to pass to generate(). */
export interface SpeciesParams {
  mode: NaturalPatternMode;
  scale: number;
  jitter: number;
  octaves: number;
  fbm: SpeciesFbmParams;
}

/**
 * Per-species parameter table.
 *
 * Design rationale for FBM_VORONOISE entries:
 *  - fluid        : very smooth, large warp, high blend → flowing oil-on-water
 *  - cil-star     : moderate warp, low blend → crystalline star-burst cells
 *  - cil-drop     : high persistence (smooth), medium warp → raindrop surface
 *  - cil-shield   : low persistence (rough), low blend → armour-plate cracks
 *  - cil-sun      : high lacunarity, radial feel → sun-corona plasma
 */
const SPECIES_PARAMS: Record<string, SpeciesParams> = {
  // ── Legacy voronoi / worley modes ──────────────────────────────────────────
  'cil-eye': {
    mode: 'CELL_DIVISION', scale: 6,  jitter: 0.85, octaves: 3,
    fbm: { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-bolt': {
    mode: 'CELL_DIVISION', scale: 8,  jitter: 0.90, octaves: 2,
    fbm: { fbmOctaves: 2, persistence: 0.45, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-vector': {
    mode: 'TORTOISE_SHELL', scale: 5, jitter: 0.80, octaves: 3,
    fbm: { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-plus': {
    mode: 'TORTOISE_SHELL', scale: 7, jitter: 0.75, octaves: 4,
    fbm: { fbmOctaves: 4, persistence: 0.55, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-arrow-right': {
    mode: 'LEAF_VEIN', scale: 5, jitter: 0.70, octaves: 4,
    fbm: { fbmOctaves: 4, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-filter': {
    mode: 'LEAF_VEIN', scale: 6, jitter: 0.65, octaves: 5,
    fbm: { fbmOctaves: 5, persistence: 0.45, lacunarity: 2.2, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-layers': {
    mode: 'FOAM', scale: 6, jitter: 0.85, octaves: 3,
    fbm: { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-loop': {
    mode: 'FOAM', scale: 5, jitter: 0.80, octaves: 4,
    fbm: { fbmOctaves: 4, persistence: 0.55, lacunarity: 1.8, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-code': {
    mode: 'SCALES', scale: 7, jitter: 0.88, octaves: 3,
    fbm: { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },
  'cil-graph': {
    mode: 'SCALES', scale: 9, jitter: 0.92, octaves: 2,
    fbm: { fbmOctaves: 2, persistence: 0.4, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  },

  // ── New FBM_VORONOISE species ───────────────────────────────────────────────
  //
  // fluid: flowing oil-on-water iridescence
  //   • Large warp (1.2) heavily distorts cell boundaries → organic flow
  //   • High voronoiseBlend (0.82) pushes toward pure noise → smooth gradients
  //   • Low persistence (0.42) means higher octaves fade quickly → gentle layering
  //   • Wide smoothK (3.2) blurs cell edges into a continuous surface
  'fluid': {
    mode: 'FBM_VORONOISE', scale: 4, jitter: 0.90, octaves: 4,
    fbm: { fbmOctaves: 5, persistence: 0.42, lacunarity: 1.9, warpStrength: 1.2, voronoiseBlend: 0.82, smoothK: 3.2 },
  },

  // cil-star: crystalline star-burst cells
  //   • Moderate warp (0.55) gives slight distortion without breaking geometry
  //   • Low voronoiseBlend (0.25) keeps voronoi cell structure visible → star facets
  //   • High lacunarity (2.4) adds sharp fine detail between stars
  //   • Tight smoothK (0.6) preserves crisp cell edges
  'cil-star': {
    mode: 'FBM_VORONOISE', scale: 7, jitter: 0.72, octaves: 3,
    fbm: { fbmOctaves: 4, persistence: 0.55, lacunarity: 2.4, warpStrength: 0.55, voronoiseBlend: 0.25, smoothK: 0.6 },
  },

  // cil-drop: raindrop surface tension rings
  //   • High persistence (0.65) makes octaves stay strong → saturated ripple depth
  //   • Medium warp (0.80) pulls cells into elongated drop shapes
  //   • Mid blend (0.55) balances cell ring vs smooth noise interior
  //   • Moderate smoothK (1.8) softens ring edges like water surface
  'cil-drop': {
    mode: 'FBM_VORONOISE', scale: 5, jitter: 0.78, octaves: 4,
    fbm: { fbmOctaves: 4, persistence: 0.65, lacunarity: 2.0, warpStrength: 0.80, voronoiseBlend: 0.55, smoothK: 1.8 },
  },

  // cil-shield: armour-plate stress fractures
  //   • Low persistence (0.32) → coarse, rough noise between plates
  //   • Very low blend (0.12) keeps hard voronoi plate boundaries dominant
  //   • Low warp (0.30) preserves angular plate geometry
  //   • Tiny smoothK (0.25) for razor-sharp crack lines
  'cil-shield': {
    mode: 'FBM_VORONOISE', scale: 6, jitter: 0.60, octaves: 3,
    fbm: { fbmOctaves: 3, persistence: 0.32, lacunarity: 2.0, warpStrength: 0.30, voronoiseBlend: 0.12, smoothK: 0.25 },
  },

  // cil-sun: solar-corona plasma tendrils
  //   • Very high lacunarity (2.8) → rapid frequency doubling, fine plasma threads
  //   • High warp (1.0) dramatically distorts cell field → chaotic tendrils
  //   • High blend (0.70) pushes toward noise basis → bright continuous luminance
  //   • Medium smoothK (2.0) creates glowing halo around features
  'cil-sun': {
    mode: 'FBM_VORONOISE', scale: 5, jitter: 0.95, octaves: 5,
    fbm: { fbmOctaves: 6, persistence: 0.50, lacunarity: 2.8, warpStrength: 1.0, voronoiseBlend: 0.70, smoothK: 2.0 },
  },
};

/** Resolve full per-species parameters (mode + fBm). Falls back to CELL_DIVISION defaults. */
export function speciesParams(species: string): SpeciesParams {
  return SPECIES_PARAMS[species] ?? {
    mode:    'CELL_DIVISION',
    scale:   6,
    jitter:  0.85,
    octaves: 3,
    fbm:     { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
  };
}

/** Map a cell species string to its natural-pattern visual mode (legacy helper). */
export function speciesPatternMode(species: string): NaturalPatternMode {
  return speciesParams(species).mode;
}

/** Numeric mode constant forwarded into the compute shader as a uniform. */
const MODE_INDEX: Record<NaturalPatternMode, number> = {
  CELL_DIVISION:  0,
  TORTOISE_SHELL: 1,
  LEAF_VEIN:      2,
  FOAM:           3,
  SCALES:         4,
  FBM_VORONOISE:  5,
};

// ─── Public API types ─────────────────────────────────────────────────────────

export interface NaturalPatternParams {
  /** Texture width in pixels (power-of-two recommended). Default 512. */
  width?: number;
  /** Texture height in pixels. Default 512. */
  height?: number;
  /** Cell frequency – higher = more, smaller cells. Range 1–32. Default 6. */
  scale?: number;
  /** Cell centroid randomness, 0 = grid, 1 = full random. Default 0.85. */
  jitter?: number;
  /** Number of Worley/Voronoi octaves for fBm layering. Range 1–6. Default 3. */
  octaves?: number;
  /** Pattern mode; defaults from species if provided, else CELL_DIVISION. */
  mode?: NaturalPatternMode;
  /** Optional cell species string – overrides `mode` + fBm params via speciesParams(). */
  species?: string;
  /** Animation time (seconds) for pulsing Voronoi centroids. Default 0. */
  time?: number;
  /** Override fBm parameters (partial merge over species defaults). */
  fbm?: Partial<SpeciesFbmParams>;
}

// ─── Inlined WGSL ────────────────────────────────────────────────────────────
//
// Sources:
//   • voronoi / worley bodies verbatim from upstream/lygia/generative/*.wgsl
//     (Patricio Gonzalez Vivo, Prosperity License)
//   • fbm scalar noise: translated from upstream/lygia/generative/fbm.wgsl
//     using value-noise hash basis (FBM_OCTAVES / FBM_AMPLITUDE_SCALAR / etc.)
//   • voronoise: Inigo Quilez "Voronoise" (2014) – public domain formula
//     https://iquilezles.org/articles/voronoise/
//
// WGSL fixes applied uniformly:
//   • `for(int …)` → `for(var i: i32 = …)` per WGSL spec
//   • `float(i)` → `f32(i)` per WGSL spec
//   • `let` in loops changed to `var` where mutation required

const COMPUTE_SHADER_SRC = /* wgsl */`
// ── uniforms ──────────────────────────────────────────────────────────────────
struct Params {
  width:   u32,
  height:  u32,
  scale:   f32,   // cell frequency
  jitter:  f32,   // [0,1] centroid jitter
  octaves: u32,   // voronoi/worley fBm octaves
  mode:    u32,   // 0=CELL_DIVISION … 5=FBM_VORONOISE
  time:    f32,   // animation seconds
  _pad:    f32,
}
@group(0) @binding(0) var<uniform> p: Params;

// fBm + voronoise per-species parameters (second uniform, 32 bytes)
struct FbmParams {
  fbmOctaves:     u32,   // scalar fBm octaves
  persistence:    f32,   // amplitude decay per octave
  lacunarity:     f32,   // frequency growth per octave
  warpStrength:   f32,   // domain-warp magnitude
  voronoiseBlend: f32,   // 0=voronoi, 1=noise
  smoothK:        f32,   // IQ smoothing kernel k
  _pad0:          f32,
  _pad1:          f32,
}
@group(0) @binding(1) var<uniform> fp: FbmParams;

@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;

// ── math helpers ─────────────────────────────────────────────────────────────
const TAU: f32 = 6.28318530717958647692;
const PI:  f32 = 3.14159265358979323846;

// ── hash / random functions ───────────────────────────────────────────────────
// lygia random.wgsl — hash12, random22, random33
fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(0.3183099, 0.3678794));
  q += dot(q, q + 0.56);
  return fract(q.x * q.y);
}

fn random22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn random33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// lygia dist.wgsl — Euclidean
fn distEuclidean2(a: vec2f, b: vec2f) -> f32 { return length(a - b); }

// ── Scalar value noise (lygia fbm.wgsl basis) ─────────────────────────────────
// A smooth gradient-free value noise on [0,1] used as the fBm building block.
// Cubic interpolation (smoothstep) matches lygia's snoise intent for 2-D fBm.
fn valueNoise(st: vec2f) -> f32 {
  let i = floor(st);
  let f = fract(st);
  let u = f * f * (3.0 - 2.0 * f);   // smoothstep curve
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ── fBm scalar (lygia fbm.wgsl, FBM_OCTAVES / persistence / lacunarity) ───────
// Matches lygia's loop structure:
//   value += amplitude * NOISE(st);  st *= lacunarity;  amplitude *= persistence;
// Returns normalised value in [0,1].
fn fbmScalar(st: vec2f, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
  var value     = 0.0;
  var amplitude = 0.5;              // FBM_AMPLITUDE_INITIAL
  var freq      = 1.0;
  var totalAmp  = 0.0;
  for (var i: u32 = 0u; i < octaves; i++) {
    value    += amplitude * valueNoise(st * freq);
    totalAmp += amplitude;
    amplitude *= persistence;       // FBM_AMPLITUDE_SCALAR
    freq      *= lacunarity;        // FBM_SCALE_SCALAR
  }
  return value / totalAmp;          // normalise (matches lygia tilable fbm)
}

// ── Voronoise (Inigo Quilez, 2014) ────────────────────────────────────────────
// Smooth blend between Voronoi cell structure and gradient noise, controlled by
//   u (=voronoiseBlend): 0 → pure Voronoi, 1 → pure smooth noise
//   k (=smoothK):        controls the width of the smooth-min kernel
// Reference: https://iquilezles.org/articles/voronoise/
fn voronoise(uv: vec2f, u: f32, k: f32) -> f32 {
  let p  = floor(uv);
  let f  = fract(uv);

  var va = 0.0;
  var wt = 0.0;

  for (var j: i32 = -2; j <= 2; j++) {
    for (var i: i32 = -2; i <= 2; i++) {
      let g  = vec2f(f32(i), f32(j));
      let o  = random22(p + g) * vec2f(u, 1.0); // IQ: o.x controlled by u
      let r  = g - f + o;
      let d  = dot(r, r);
      let w  = pow(1.0 - smoothstep(0.0, 1.414, sqrt(d)), k);
      va    += w * o.y;
      wt    += w;
    }
  }
  return va / wt;
}

// ── Domain-warped voronoise (fbm warp → voronoise lookup) ─────────────────────
// Two-pass domain warping (Inigo Quilez "Warping" technique, 2002):
//   q = fbm(uv)
//   r = fbm(uv + q * warpStrength)
//   result = voronoise(uv + r * warpStrength, blend, k)
fn warpedVoronoise(
  uv: vec2f,
  warpStrength: f32,
  voronoiseBlend: f32,
  smoothK: f32,
  octaves: u32,
  persistence: f32,
  lacunarity: f32,
) -> f32 {
  // First warp pass – use two offset fbm calls to build a 2-D warp field
  let q = vec2f(
    fbmScalar(uv,                        octaves, persistence, lacunarity),
    fbmScalar(uv + vec2f(5.2,  1.3),    octaves, persistence, lacunarity),
  );
  // Second warp pass
  let r = vec2f(
    fbmScalar(uv + q * warpStrength,                       octaves, persistence, lacunarity),
    fbmScalar(uv + q * warpStrength + vec2f(1.7, 9.2),    octaves, persistence, lacunarity),
  );
  return voronoise(uv + r * warpStrength, voronoiseBlend, smoothK);
}

// ── lygia voronoi2 ────────────────────────────────────────────────────────────
fn voronoi2(uv: vec2f, time: f32, jitter: f32) -> vec3f {
  let i_uv = floor(uv);
  let f_uv = fract(uv);
  var min_dist = 10.0;
  var centroid = vec2f(0.0);
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let neighbor = vec2f(f32(i), f32(j));
      var point = random22(i_uv + neighbor);
      point = 0.5 + 0.5 * sin(time + TAU * point);
      point = mix(vec2f(0.5), point, jitter);
      let diff = neighbor + point - f_uv;
      let dist = length(diff);
      if (dist < min_dist) {
        min_dist = dist;
        centroid = point;
      }
    }
  }
  return vec3f(centroid, min_dist);
}

// ── lygia worley22 ────────────────────────────────────────────────────────────
fn worley22(p: vec2f, jitter: f32) -> vec2f {
  let n = floor(p);
  let f = fract(p);
  var distF1 = 1.0;
  var distF2 = 1.0;
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let o = random22(n + g) * jitter;
      let wp = g + o;
      let d = distEuclidean2(wp, f);
      if (d < distF1) {
        distF2 = distF1;
        distF1 = d;
      } else if (d < distF2) {
        distF2 = d;
      }
    }
  }
  return vec2f(distF1, distF2);
}

// ── fBm layering – voronoi / worley ──────────────────────────────────────────
fn fbmVoronoi(uv: vec2f, octaves: u32, scale: f32, jitter: f32, time: f32) -> vec3f {
  var result    = vec3f(0.0);
  var amp       = 0.5;
  var freq      = scale;
  var total_amp = 0.0;
  for (var o: u32 = 0u; o < octaves; o++) {
    let v = voronoi2(uv * freq, time, jitter);
    result    += v * amp;
    total_amp += amp;
    amp  *= 0.5;
    freq *= 2.0;
  }
  return result / total_amp;
}

fn fbmWorley(uv: vec2f, octaves: u32, scale: f32, jitter: f32) -> vec2f {
  var result    = vec2f(0.0);
  var amp       = 0.5;
  var freq      = scale;
  var total_amp = 0.0;
  for (var o: u32 = 0u; o < octaves; o++) {
    let w = worley22(uv * freq, jitter);
    result    += w * amp;
    total_amp += amp;
    amp  *= 0.5;
    freq *= 2.0;
  }
  return result / total_amp;
}

// ── pattern modes ─────────────────────────────────────────────────────────────

// 0 – CELL_DIVISION
fn patternCellDivision(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let v    = fbmVoronoi(uv, octaves, scale, jitter, time);
  let dist = v.z;
  let cy   = v.y;
  let edge = smoothstep(0.02, 0.06, dist);
  return vec4f(dist, cy, 1.0 - edge, 1.0);
}

// 1 – TORTOISE_SHELL
fn patternTortoiseShell(uv: vec2f, scale: f32, jitter: f32, octaves: u32) -> vec4f {
  let w     = fbmWorley(uv, octaves, scale, jitter);
  let F1    = w.x;
  let F2    = w.y;
  let ridge = F2 - F1;
  let crack  = smoothstep(0.0, 0.12, ridge);
  return vec4f(F1, F2, crack, 1.0);
}

// 2 – LEAF_VEIN
fn patternLeafVein(uv: vec2f, scale: f32, jitter: f32, octaves: u32) -> vec4f {
  let w      = fbmWorley(uv, octaves, scale, jitter);
  let vein   = 1.0 - w.x;
  let branch = pow(vein, 2.5);
  let mid    = fbmWorley(uv, 1u, scale * 0.5, jitter * 0.6).x;
  return vec4f(branch, mid, 1.0 - w.y, 1.0);
}

// 3 – FOAM
fn patternFoam(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let v      = fbmVoronoi(uv, octaves, scale, jitter, time);
  let w      = fbmWorley(uv, octaves, scale, jitter);
  let bubble = mix(v.z, w.x, 0.5);
  let wall   = smoothstep(0.03, 0.08, bubble);
  return vec4f(bubble, w.y, wall, 1.0);
}

// 4 – SCALES
fn patternScales(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let offset    = vec2f(0.0, 0.25);
  let v1        = fbmVoronoi(uv,          octaves, scale,       jitter, time);
  let v2        = fbmVoronoi(uv + offset, octaves, scale * 1.1, jitter, time * 0.7);
  let scale_cell = smoothstep(0.0, 0.5, v1.z) * (1.0 - smoothstep(0.4, 0.7, v2.z));
  return vec4f(scale_cell, v2.z, v1.z, 1.0);
}

// 5 – FBM_VORONOISE: domain-warped fBm mixed with voronoise
//
// Algorithm (per FbmParams):
//   1. Build a scalar fBm noise field N using lygia-style octave loop
//   2. Domain-warp a voronoise lookup via two-pass q/r warp fields
//   3. Blend warped voronoise (V) and plain fBm (N) by voronoiseBlend
//   4. Derive an edge highlight from the cross-derivative (approx. gradient)
//      using finite-difference samples of the final signal
//
// Channel mapping:
//   R – blended signal (primary intensity)
//   G – plain fBm value (secondary layer / depth)
//   B – gradient magnitude (edge / ridge highlight)
//   A – 1.0
fn patternFbmVoronoise(
  uv:    vec2f,
  scale: f32,
  jitter: f32,
  octaves: u32,
  time:   f32,
  fbmOctaves:     u32,
  persistence:    f32,
  lacunarity:     f32,
  warpStrength:   f32,
  voronoiseBlend: f32,
  smoothK:        f32,
) -> vec4f {
  let uvs = uv * scale;

  // ── Plain fBm (lygia fbm.wgsl loop) ──────────────────────────────────────
  // Small time offset so the fBm layer gently animates independently
  let fbmVal = fbmScalar(uvs + vec2f(time * 0.05, 0.0), fbmOctaves, persistence, lacunarity);

  // ── Domain-warped voronoise ───────────────────────────────────────────────
  let vn = warpedVoronoise(uvs, warpStrength, voronoiseBlend, smoothK, fbmOctaves, persistence, lacunarity);

  // ── Final blend: lerp voronoise → fBm by voronoiseBlend ──────────────────
  // voronoiseBlend=0: mostly voronoi structure; voronoiseBlend=1: mostly noise
  let signal = mix(vn, fbmVal, voronoiseBlend * 0.5);

  // ── Gradient magnitude (finite differences for edge detection) ────────────
  let eps  = 1.0 / 256.0;
  let sig_dx = warpedVoronoise(uvs + vec2f(eps, 0.0), warpStrength, voronoiseBlend, smoothK, fbmOctaves, persistence, lacunarity);
  let sig_dy = warpedVoronoise(uvs + vec2f(0.0, eps), warpStrength, voronoiseBlend, smoothK, fbmOctaves, persistence, lacunarity);
  let grad = length(vec2f(sig_dx - vn, sig_dy - vn)) / eps * 0.15;
  let edge = clamp(grad, 0.0, 1.0);

  return vec4f(signal, fbmVal, edge, 1.0);
}

// ── compute entry point ───────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y;
  if (px >= p.width || py >= p.height) { return; }

  let uv = vec2f(
    f32(px) / f32(p.width),
    1.0 - f32(py) / f32(p.height),
  );

  var color: vec4f;
  switch (p.mode) {
    case 0u: { color = patternCellDivision (uv, p.scale, p.jitter, p.octaves, p.time); }
    case 1u: { color = patternTortoiseShell(uv, p.scale, p.jitter, p.octaves); }
    case 2u: { color = patternLeafVein     (uv, p.scale, p.jitter, p.octaves); }
    case 3u: { color = patternFoam         (uv, p.scale, p.jitter, p.octaves, p.time); }
    case 4u: { color = patternScales       (uv, p.scale, p.jitter, p.octaves, p.time); }
    case 5u: {
      color = patternFbmVoronoise(
        uv, p.scale, p.jitter, p.octaves, p.time,
        fp.fbmOctaves, fp.persistence, fp.lacunarity,
        fp.warpStrength, fp.voronoiseBlend, fp.smoothK,
      );
    }
    default: { color = patternCellDivision(uv, p.scale, p.jitter, p.octaves, p.time); }
  }

  textureStore(outTex, vec2u(px, py), color);
}
`;

// ─── NaturalPatternGenerator ─────────────────────────────────────────────────

/**
 * GPU-accelerated natural cell-surface texture generator.
 *
 * Creates a {@link GPUTexture} (rgba8unorm, TEXTURE_BINDING | COPY_SRC)
 * populated by a WebGPU compute dispatch.  The texture can be bound directly
 * to a render pipeline or read back to CPU via {@link readback}.
 *
 * The generator now supports six pattern modes, including the new
 * **FBM_VORONOISE** mode that blends domain-warped fBm with Inigo Quilez's
 * voronoise function.  Per-species fBm parameters are resolved automatically
 * from the `species` field via {@link speciesParams}.
 *
 * @example
 * ```ts
 * const gen = new NaturalPatternGenerator(device);
 * // Classic mode – resolved from species string:
 * const tex1 = await gen.generate({ species: 'cil-eye', scale: 6, octaves: 3 });
 * // New fbm+voronoise mode with custom params:
 * const tex2 = await gen.generate({ species: 'fluid', fbm: { warpStrength: 1.5 } });
 * gen.destroy();
 * ```
 */
export class NaturalPatternGenerator {
  private readonly device: any /*GPUDevice*/;
  private pipeline: any /*GPUComputePipeline*/ | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(device: any /*GPUDevice*/) {
    this.device = device;
  }

  // ── lazy pipeline init ────────────────────────────────────────────────────

  private async ensurePipeline(): Promise<void> {
    if (this.pipeline) return;

    const module = this.device.createShaderModule({
      label: 'natural-patterns-compute',
      code: COMPUTE_SHADER_SRC,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'natural-patterns-bgl',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'rgba8unorm',
            viewDimension: '2d',
          },
        },
      ],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label: 'natural-patterns-pipeline',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: { module, entryPoint: 'main' },
    });
  }

  // ── public generate ───────────────────────────────────────────────────────

  /**
   * Generate a natural-pattern texture.  Resolves once the GPU work is
   * submitted; the returned texture is valid immediately for binding.
   *
   * When `species` is provided, all parameters (mode, scale, jitter, octaves,
   * fBm) are resolved from the per-species table and then merged with any
   * explicit overrides in `params`.
   */
  async generate(params: NaturalPatternParams = {}): Promise<GPUTexture> {
    await this.ensurePipeline();

    const {
      width  = 512,
      height = 512,
      time   = 0,
      species,
      mode: modeOverride,
      fbm: fbmOverride,
    } = params;

    // Resolve species defaults, then merge explicit overrides
    const resolved = species ? speciesParams(species) : {
      mode:    'CELL_DIVISION' as NaturalPatternMode,
      scale:   6,
      jitter:  0.85,
      octaves: 3,
      fbm:     { fbmOctaves: 3, persistence: 0.5, lacunarity: 2.0, warpStrength: 0.0, voronoiseBlend: 0.0, smoothK: 1.0 },
    };

    const mode: NaturalPatternMode = modeOverride ?? resolved.mode;
    const scale   = params.scale   ?? resolved.scale;
    const jitter  = params.jitter  ?? resolved.jitter;
    const octaves = params.octaves ?? resolved.octaves;

    const fbm: SpeciesFbmParams = { ...resolved.fbm, ...fbmOverride };

    // ── Uniform buffer 0: Params (8 × 4 bytes = 32 bytes) ──────────────────
    const uniformData = new ArrayBuffer(32);
    const uView = new DataView(uniformData);
    uView.setUint32 ( 0, width,                              true);
    uView.setUint32 ( 4, height,                             true);
    uView.setFloat32( 8, Math.max(1, scale),                 true);
    uView.setFloat32(12, Math.min(1, Math.max(0, jitter)),   true);
    uView.setUint32 (16, Math.min(6, Math.max(1, octaves)),  true);
    uView.setUint32 (20, MODE_INDEX[mode],                   true);
    uView.setFloat32(24, time,                               true);
    uView.setFloat32(28, 0,                                  true); // _pad

    const uniformBuf = this.device.createBuffer({
      label: 'natural-patterns-uniforms',
      size:  32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuf, 0, uniformData);

    // ── Uniform buffer 1: FbmParams (8 × 4 bytes = 32 bytes) ───────────────
    const fbmData = new ArrayBuffer(32);
    const fView = new DataView(fbmData);
    fView.setUint32 ( 0, Math.min(6, Math.max(1, fbm.fbmOctaves)), true);
    fView.setFloat32( 4, Math.min(1, Math.max(0.01, fbm.persistence)),  true);
    fView.setFloat32( 8, Math.max(1.0, fbm.lacunarity),                  true);
    fView.setFloat32(12, Math.max(0, fbm.warpStrength),                   true);
    fView.setFloat32(16, Math.min(1, Math.max(0, fbm.voronoiseBlend)),    true);
    fView.setFloat32(20, Math.max(0.01, fbm.smoothK),                     true);
    fView.setFloat32(24, 0, true); // _pad0
    fView.setFloat32(28, 0, true); // _pad1

    const fbmBuf = this.device.createBuffer({
      label: 'natural-patterns-fbm-uniforms',
      size:  32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(fbmBuf, 0, fbmData);

    // ── Output texture ──────────────────────────────────────────────────────
    const texture = this.device.createTexture({
      label:  `natural-pattern-${mode}-${width}x${height}`,
      size:   { width, height, depthOrArrayLayers: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.COPY_SRC,
    });

    const bindGroup = this.device.createBindGroup({
      label:  'natural-patterns-bg',
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: fbmBuf     } },
        { binding: 2, resource: texture.createView()  },
      ],
    });

    // ── Dispatch ────────────────────────────────────────────────────────────
    const enc  = this.device.createCommandEncoder({ label: 'natural-patterns-enc' });
    const pass = enc.beginComputePass({ label: 'natural-patterns-pass' });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width  / 8),
      Math.ceil(height / 8),
    );
    pass.end();
    this.device.queue.submit([enc.finish()]);

    uniformBuf.destroy();
    fbmBuf.destroy();

    return texture;
  }

  // ── convenience: CPU readback ─────────────────────────────────────────────

  /**
   * Read generated texture back to CPU as a {@link Uint8ClampedArray} RGBA
   * buffer (row-major, top-to-bottom).  Useful for canvas rendering or
   * server-side image export.
   */
  async readback(
    texture: GPUTexture,
    width: number,
    height: number,
  ): Promise<Uint8ClampedArray> {
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const stagingBuf  = this.device.createBuffer({
      size:  bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture },
      { buffer: stagingBuf, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    this.device.queue.submit([enc.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const raw  = new Uint8Array(stagingBuf.getMappedRange());
    const rgba = new Uint8ClampedArray(width * height * 4);

    for (let row = 0; row < height; row++) {
      const src = raw.subarray(row * bytesPerRow, row * bytesPerRow + width * 4);
      rgba.set(src, row * width * 4);
    }

    stagingBuf.unmap();
    stagingBuf.destroy();
    return rgba;
  }

  // ── lifecycle ─────────────────────────────────────────────────────────────

  /** Free GPU pipeline resources.  Previously generated textures remain
   *  valid until their own {@link GPUTexture.destroy} is called. */
  destroy(): void {
    this.pipeline        = null;
    this.bindGroupLayout = null;
  }
}

// ─── Re-exports ───────────────────────────────────────────────────────────────
export { COMPUTE_SHADER_SRC as NATURAL_PATTERNS_WGSL };
