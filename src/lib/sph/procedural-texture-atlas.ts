/**
 * procedural-texture-atlas.ts — M772: Procedural Texture Atlas Generation
 * ─────────────────────────────────────────────────────────────────────────────
 * Generates a GPU texture atlas where each species occupies a dedicated tile.
 * Five procedural texture algorithms are mapped to species via the shader
 * registry, then rendered into atlas tiles by a single WebGPU compute dispatch.
 *
 * Texture algorithms
 * ─────────────────────────────────────────────────────────────────────────────
 *   VORONOI_MEMBRANE  — Voronoi F2−F1 soft cell membrane (cil-eye, cil-filter)
 *   PERLIN_ROCK       — Multi-octave Perlin fBm rock/stone surface (cil-plus, cil-code)
 *   TURING_STRIPE     — Turing reaction–diffusion spot/stripe (cil-loop, cil-bolt)
 *   WOOD_GRAIN        — Concentric ring distortion (cil-layers, cil-arrow-right)
 *   MARBLE_VEIN       — Turbulent sine marble veining (cil-vector, cil-graph)
 *
 * Atlas layout
 * ─────────────────────────────────────────────────────────────────────────────
 * The atlas is a single rgba8unorm 2-D texture of size (cols × tileSize) × (rows × tileSize).
 * Each species gets one tile.  Tiles are arranged in row-major order matching
 * SPECIES_ORDER from cell-body-bridge.ts (index 0 = fluid, 1 = cil-eye, …).
 *
 * Channel encoding (per-tile):
 *   R — primary pattern intensity (albedo-modulator / diffuse weight)
 *   G — secondary pattern layer (depth / normal hint)
 *   B — edge / ridge mask (contour lines, crack highlights)
 *   A — roughness perturbation (0.5 = neutral; <0.5 smoother, >0.5 rougher)
 *
 * UV mapping for cells
 * ─────────────────────────────────────────────────────────────────────────────
 * A cell with species index `idx` samples the atlas at:
 *   atlasUV.x = (col + localUV.x) / cols
 *   atlasUV.y = (row + localUV.y) / rows
 * where col = idx % cols, row = floor(idx / cols).
 *
 * The helper function {@link speciesAtlasUV} computes this mapping.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/species-shader-registry.ts  — SpeciesShaderConfig, PatternShader
 *   src/lib/sph/natural-patterns.ts         — NaturalPatternGenerator, WGSL noise
 *   src/lib/sph/cell-body-bridge.ts         — SPECIES_ORDER, speciesIndex
 *   src/lib/sph/turing-pattern.ts           — TuringPatternGenerator
 *   src/lib/sph/nature-texture-manager.ts   — NatureTextureManager
 */




// ─── Texture algorithm enum ─────────────────────────────────────────────────

/**
 * Procedural texture algorithm rendered into each atlas tile.
 *
 * Each variant maps to a branch in the WGSL compute shader.
 */



import {

  getSpeciesShaderConfig,
  getAllSpeciesIds,
  type PatternShader,
  type SpeciesShaderConfig,
} from './species-shader-registry';

export type ProceduralTextureKind =
  | 'VORONOI_MEMBRANE'
  | 'PERLIN_ROCK'
  | 'TURING_STRIPE'
  | 'WOOD_GRAIN'
  | 'MARBLE_VEIN';

/** Numeric index forwarded to the GPU as a uniform. */
const TEXTURE_KIND_INDEX: Record<ProceduralTextureKind, number> = {
  VORONOI_MEMBRANE: 0,
  PERLIN_ROCK:      1,
  TURING_STRIPE:    2,
  WOOD_GRAIN:       3,
  MARBLE_VEIN:      4,
};

// ─── Pattern → texture kind mapping ─────────────────────────────────────────

/**
 * Map a PatternShader (from species-shader-registry) to a ProceduralTextureKind.
 *
 * Multiple pattern shaders may share the same procedural base texture; the
 * per-species pattern shader adds animated overlays at render time while the
 * atlas provides the static base layer.
 */
const PATTERN_TO_TEXTURE: Record<PatternShader, ProceduralTextureKind> = {
  'none':                'PERLIN_ROCK',
  'voronoi-membrane':    'VORONOI_MEMBRANE',
  'voronoi-natural':     'VORONOI_MEMBRANE',
  'grayscott-species':   'TURING_STRIPE',
  'turing-pattern':      'TURING_STRIPE',
  'supershape-species':  'PERLIN_ROCK',
  'iq-palette-species':  'MARBLE_VEIN',
  'julia-background':    'MARBLE_VEIN',
  'curl-trail':          'WOOD_GRAIN',
  'fluid-surface':       'WOOD_GRAIN',
  'caustics':            'VORONOI_MEMBRANE',
};

/**
 * Resolve the ProceduralTextureKind for a given species id.
 * Falls back to PERLIN_ROCK for unknown species.
 */
export function speciesTextureKind(species: string): ProceduralTextureKind {
  const cfg = getSpeciesShaderConfig(species);
  return PATTERN_TO_TEXTURE[cfg.patternShader] ?? 'PERLIN_ROCK';
}

// ─── Per-tile generation parameters ─────────────────────────────────────────

/**
 * Per-tile tuning parameters forwarded to the GPU as part of the tile
 * uniform buffer.  Each species gets its own set of values derived from
 * SpeciesShaderConfig + NaturalPatternParams.
 */
export interface TileParams {
  /** Which procedural algorithm to run for this tile. */
  kind: ProceduralTextureKind;
  /** Base frequency / cell scale.  Higher = more detail cells. */
  scale: number;
  /** Voronoi centroid jitter (0 = grid, 1 = fully random). */
  jitter: number;
  /** fBm octave count (1–6). */
  octaves: number;
  /** Distortion / turbulence strength for wood/marble/turing. */
  turbulence: number;
  /** Base colour tint RGB (linear), written into the A-modulated palette. */
  tintR: number;
  tintG: number;
  tintB: number;
}

/**
 * Build TileParams for a species from its SpeciesShaderConfig.
 *
 * Design rationale per texture kind:
 *   VORONOI_MEMBRANE — uses sdfParams radius or petals for cell count,
 *     high jitter for organic membrane look, 3–4 octaves.
 *   PERLIN_ROCK — coarse scale, low octaves, moderate turbulence for rocky
 *     surface.  Colour from albedo.
 *   TURING_STRIPE — medium scale driven by bloom pulse frequency (fast
 *     species = denser stripes), turbulence from bloomPulseAmplitude.
 *   WOOD_GRAIN — low scale (wide rings), directional turbulence from
 *     fresnelPower (higher fresnel = more ring distortion).
 *   MARBLE_VEIN — mid-scale, high turbulence for vein sinuosity,
 *     colour from albedo shifted toward cooler tones.
 */
function buildTileParams(cfg: SpeciesShaderConfig): TileParams {
  const kind = PATTERN_TO_TEXTURE[cfg.patternShader] ?? 'PERLIN_ROCK';
  const albedo = cfg.materialParams.albedo ?? [0.5, 0.5, 0.5];

  switch (kind) {
    case 'VORONOI_MEMBRANE':
      return {
        kind,
        scale:      (cfg.sdfParams.petals ?? cfg.sdfParams.radius ?? 6) * 1.2,
        jitter:     0.88,
        octaves:    3,
        turbulence: 0.4,
        tintR: albedo[0], tintG: albedo[1], tintB: albedo[2],
      };

    case 'PERLIN_ROCK':
      return {
        kind,
        scale:      4.0,
        jitter:     0.5,
        octaves:    4,
        turbulence: 0.6,
        tintR: albedo[0], tintG: albedo[1], tintB: albedo[2],
      };

    case 'TURING_STRIPE':
      return {
        kind,
        scale:      5.0 + cfg.bloomPulseFrequency * 2.0,
        jitter:     0.75,
        octaves:    3,
        turbulence: cfg.bloomPulseAmplitude * 3.0,
        tintR: albedo[0], tintG: albedo[1], tintB: albedo[2],
      };

    case 'WOOD_GRAIN':
      return {
        kind,
        scale:      2.5,
        jitter:     0.3,
        octaves:    4,
        turbulence: (cfg.materialParams.fresnelPower ?? 2.0) * 0.35,
        tintR: albedo[0], tintG: albedo[1], tintB: albedo[2],
      };

    case 'MARBLE_VEIN':
      return {
        kind,
        scale:      3.5,
        jitter:     0.6,
        octaves:    5,
        turbulence: 1.2,
        tintR: albedo[0] * 0.9,
        tintG: albedo[1] * 0.95,
        tintB: Math.min(1.0, albedo[2] * 1.1),
      };
  }
}

// ─── Atlas configuration ────────────────────────────────────────────────────

/** Default tile size in pixels (each species tile). */
const DEFAULT_TILE_SIZE = 256;

/** Atlas grid columns.  11 species → 4 columns × 3 rows = 12 slots. */
const ATLAS_COLS = 4;

/**
 * Full atlas configuration, resolved at construction time.
 */
export interface AtlasConfig {
  /** Size of each species tile in pixels (square). */
  tileSize: number;
  /** Number of columns in the atlas grid. */
  cols: number;
  /** Number of rows (derived from species count / cols). */
  rows: number;
  /** Total atlas width in pixels. */
  width: number;
  /** Total atlas height in pixels. */
  height: number;
  /** Ordered species ids (mirrors SPECIES_ORDER with fluid at index 0). */
  species: string[];
  /** Per-tile generation parameters, one per species. */
  tiles: TileParams[];
}

/**
 * Build an AtlasConfig for the current species registry.
 *
 * @param tileSize  Tile resolution in px (default 256).
 * @param cols      Grid columns (default 4).
 */
export function buildAtlasConfig(
  tileSize: number = DEFAULT_TILE_SIZE,
  cols: number = ATLAS_COLS,
): AtlasConfig {
  // "fluid" is index 0 in SPECIES_ORDER; all cil-* species follow.
  const cilSpecies = getAllSpeciesIds();
  const species = ['fluid', ...cilSpecies];
  const rows = Math.ceil(species.length / cols);

  const tiles: TileParams[] = species.map(id => {
    const cfg = getSpeciesShaderConfig(id);
    return buildTileParams(cfg);
  });

  return {
    tileSize,
    cols,
    rows,
    width:  cols * tileSize,
    height: rows * tileSize,
    species,
    tiles,
  };
}

// ─── UV helpers ─────────────────────────────────────────────────────────────

/**
 * Convert a species index + local UV (0–1) into atlas UV coordinates.
 *
 * @param speciesIndex  Numeric species index (0 = fluid, 1 = cil-eye, …).
 * @param localU        Cell-local U coordinate [0, 1].
 * @param localV        Cell-local V coordinate [0, 1].
 * @param config        The active AtlasConfig.
 * @returns             [atlasU, atlasV] in [0, 1]².
 */
export function speciesAtlasUV(
  speciesIndex: number,
  localU: number,
  localV: number,
  config: AtlasConfig,
): [number, number] {
  const col = speciesIndex % config.cols;
  const row = Math.floor(speciesIndex / config.cols);
  const atlasU = (col + localU) / config.cols;
  const atlasV = (row + localV) / config.rows;
  return [atlasU, atlasV];
}

/**
 * WGSL snippet that cells can inline to convert species index + local uv
 * into atlas texture coordinates.  Expects uniforms:
 *   u_atlasCols: u32
 *   u_atlasRows: u32
 *
 * Usage in vertex/fragment shader:
 *   let atlasUV = speciesAtlasUV(speciesIdx, localUV, u_atlasCols, u_atlasRows);
 *   let sample  = textureSample(atlasTexture, atlasSampler, atlasUV);
 */
export const ATLAS_UV_WGSL = /* wgsl */`
fn speciesAtlasUV(idx: u32, uv: vec2f, cols: u32, rows: u32) -> vec2f {
  let col = idx % cols;
  let row = idx / cols;
  return vec2f(
    (f32(col) + uv.x) / f32(cols),
    (f32(row) + uv.y) / f32(rows),
  );
}
`;

// ─── WGSL compute shader ────────────────────────────────────────────────────
//
// A single compute dispatch writes ALL tiles into the atlas.  Each invocation
// identifies its tile from the global pixel coordinate, then reads that tile's
// TileParams from a storage buffer to select the algorithm branch.
//
// Noise primitives reuse the same hash / value-noise / voronoi basis functions
// found in natural-patterns.ts (lygia-derived), inlined here so the atlas
// shader is self-contained.

const COMPUTE_SHADER_SRC = /* wgsl */`
// ── atlas uniforms ───────────────────────────────────────────────────────────
struct AtlasParams {
  tileSize:  u32,
  cols:      u32,
  rows:      u32,
  tileCount: u32,
  time:      f32,
  _pad0:     f32,
  _pad1:     f32,
  _pad2:     f32,
}
@group(0) @binding(0) var<uniform> ap: AtlasParams;

// Per-tile params packed into a storage buffer (stride = 48 bytes, 12 × f32)
struct TileData {
  kind:       u32,    // ProceduralTextureKind index
  scale:      f32,
  jitter:     f32,
  octaves:    u32,
  turbulence: f32,
  tintR:      f32,
  tintG:      f32,
  tintB:      f32,
  _pad0:      f32,
  _pad1:      f32,
  _pad2:      f32,
  _pad3:      f32,
}
@group(0) @binding(1) var<storage, read> tiles: array<TileData>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;

// ── constants ────────────────────────────────────────────────────────────────
const TAU: f32 = 6.28318530717958647692;
const PI:  f32 = 3.14159265358979323846;

// ── hash / noise primitives (lygia-derived) ──────────────────────────────────
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

// Smooth value noise (cubic interpolation)
fn valueNoise(st: vec2f) -> f32 {
  let i = floor(st);
  let f = fract(st);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 2-D Perlin-like gradient noise (analytic, Inigo Quilez)
fn gradientNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic smoothstep
  let ga = random22(i + vec2f(0.0, 0.0)) * 2.0 - 1.0;
  let gb = random22(i + vec2f(1.0, 0.0)) * 2.0 - 1.0;
  let gc = random22(i + vec2f(0.0, 1.0)) * 2.0 - 1.0;
  let gd = random22(i + vec2f(1.0, 1.0)) * 2.0 - 1.0;
  let va = dot(ga, f - vec2f(0.0, 0.0));
  let vb = dot(gb, f - vec2f(1.0, 0.0));
  let vc = dot(gc, f - vec2f(0.0, 1.0));
  let vd = dot(gd, f - vec2f(1.0, 1.0));
  return mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y) * 0.5 + 0.5;
}

// fBm with Perlin noise basis
fn fbmPerlin(st: vec2f, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
  var value     = 0.0;
  var amplitude = 0.5;
  var freq      = 1.0;
  var totalAmp  = 0.0;
  for (var i: u32 = 0u; i < octaves; i++) {
    value    += amplitude * gradientNoise(st * freq);
    totalAmp += amplitude;
    amplitude *= persistence;
    freq      *= lacunarity;
  }
  return value / totalAmp;
}

// Scalar fBm with value noise
fn fbmValue(st: vec2f, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
  var value     = 0.0;
  var amplitude = 0.5;
  var freq      = 1.0;
  var totalAmp  = 0.0;
  for (var i: u32 = 0u; i < octaves; i++) {
    value    += amplitude * valueNoise(st * freq);
    totalAmp += amplitude;
    amplitude *= persistence;
    freq      *= lacunarity;
  }
  return value / totalAmp;
}

// ── Voronoi F1 / F2 ─────────────────────────────────────────────────────────
fn voronoiF1F2(uv: vec2f, jitter: f32) -> vec3f {
  let n = floor(uv);
  let f = fract(uv);
  var dF1 = 8.0;
  var dF2 = 8.0;
  var cellId = vec2f(0.0);
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let o = random22(n + g) * jitter;
      let r = g + o - f;
      let d = dot(r, r);
      if (d < dF1) {
        dF2 = dF1;
        dF1 = d;
        cellId = n + g;
      } else if (d < dF2) {
        dF2 = d;
      }
    }
  }
  dF1 = sqrt(dF1);
  dF2 = sqrt(dF2);
  return vec3f(dF1, dF2, hash21(cellId));
}

// ── 0: VORONOI_MEMBRANE ──────────────────────────────────────────────────────
// Soft F2−F1 membrane with domain-warped cell centres.
// R = membrane intensity, G = cell interior gradient, B = edge highlight, A = roughness
fn voronoiMembrane(uv: vec2f, scale: f32, jitter: f32, octaves: u32, turb: f32, time: f32) -> vec4f {
  // Domain warp: offset uv by fBm to create organic irregularity
  let warp = vec2f(
    fbmValue(uv * scale * 0.7 + vec2f(1.7, 9.2), octaves, 0.5, 2.0),
    fbmValue(uv * scale * 0.7 + vec2f(5.2, 1.3), octaves, 0.5, 2.0),
  ) * turb;
  let warpedUV = uv * scale + warp;

  let v = voronoiF1F2(warpedUV + vec2f(sin(time * 0.3) * 0.15, cos(time * 0.4) * 0.12), jitter);
  let membrane = smoothstep(0.0, 0.25, v.y - v.x); // soft F2-F1 ridge
  let interior = 1.0 - smoothstep(0.0, 0.45, v.x); // cell body glow
  let edge     = 1.0 - membrane;                    // bright membrane lines
  let roughVar = mix(0.35, 0.65, v.z);              // per-cell roughness variation

  return vec4f(interior, v.z, edge, roughVar);
}

// ── 1: PERLIN_ROCK ───────────────────────────────────────────────────────────
// Multi-octave Perlin fBm → rocky/stone surface.
// R = diffuse intensity, G = height-like value, B = edge cracking, A = roughness
fn perlinRock(uv: vec2f, scale: f32, octaves: u32, turb: f32) -> vec4f {
  let uvs = uv * scale;
  let base  = fbmPerlin(uvs, octaves, 0.55, 2.1);
  let fine  = fbmPerlin(uvs * 3.0 + vec2f(3.7, 8.1), min(octaves, 3u), 0.4, 2.3);

  // Combine coarse + fine for rocky stratification
  let rock = base * 0.7 + fine * 0.3;

  // Crack detection via gradient magnitude (finite difference)
  let eps = 0.01;
  let dx = fbmPerlin(uvs + vec2f(eps, 0.0), octaves, 0.55, 2.1) - base;
  let dy = fbmPerlin(uvs + vec2f(0.0, eps), octaves, 0.55, 2.1) - base;
  let grad = clamp(length(vec2f(dx, dy)) / eps * turb, 0.0, 1.0);

  let roughness = mix(0.55, 0.85, rock); // rough = lighter rock strata

  return vec4f(rock, base, grad, roughness);
}

// ── 2: TURING_STRIPE ─────────────────────────────────────────────────────────
// Approximated Turing reaction–diffusion pattern using competing noise fields.
// This is a visual approximation (not a full Gray-Scott sim) suitable for
// static atlas tiles.  The actual animated R-D runs in turing-pattern.ts;
// this provides the base texture beneath it.
//
// R = spot/stripe mask, G = secondary modulation, B = edge, A = roughness
fn turingStripe(uv: vec2f, scale: f32, jitter: f32, octaves: u32, turb: f32) -> vec4f {
  let uvs = uv * scale;

  // Activator field: coarse fBm
  let activator = fbmPerlin(uvs, octaves, 0.5, 2.0);
  // Inhibitor field: fine fBm at different frequency + offset
  let inhibitor = fbmPerlin(uvs * 1.8 + vec2f(4.3, 7.1), octaves, 0.45, 2.2);

  // Turing-like reaction: activator – inhibitor → spots/stripes
  let reaction = clamp(activator - inhibitor * 0.85 + 0.15, 0.0, 1.0);

  // Threshold into binary-ish spot mask with soft edge
  let threshold = 0.48 + turb * 0.1;
  let spots = smoothstep(threshold - 0.08, threshold + 0.08, reaction);

  // Edge detection on the spot boundary
  let eps = 0.008;
  let rx = clamp(
    fbmPerlin(uvs + vec2f(eps, 0.0), octaves, 0.5, 2.0)
    - fbmPerlin((uvs + vec2f(eps, 0.0)) * 1.8 + vec2f(4.3, 7.1), octaves, 0.45, 2.2) * 0.85 + 0.15,
    0.0, 1.0,
  );
  let edge = clamp(abs(rx - reaction) / eps * 0.3, 0.0, 1.0);

  // Voronoi jitter adds organic irregularity to spot boundaries
  let cellNoise = voronoiF1F2(uvs * 0.8, jitter).z;
  let modulated = spots * mix(0.8, 1.0, cellNoise);

  return vec4f(modulated, reaction, edge, mix(0.4, 0.7, spots));
}

// ── 3: WOOD_GRAIN ────────────────────────────────────────────────────────────
// Concentric rings distorted by Perlin turbulence → wood cross-section.
// R = ring intensity, G = ring distance (continuous), B = grain edge, A = roughness
fn woodGrain(uv: vec2f, scale: f32, octaves: u32, turb: f32, time: f32) -> vec4f {
  let uvs = (uv - 0.5) * scale; // centre-relative for ring pattern

  // Distance from centre gives concentric rings
  let dist = length(uvs);

  // Perlin turbulence distorts the ring distance
  let noise = fbmPerlin(uv * scale * 2.0 + vec2f(time * 0.02, 0.0), octaves, 0.5, 2.0);
  let distorted = dist + noise * turb;

  // Ring pattern: sinusoidal intensity modulation
  let ringFreq = 12.0;
  let ring = sin(distorted * ringFreq * PI) * 0.5 + 0.5;

  // Fine grain lines (high-frequency noise along ring direction)
  let grain = fbmValue(vec2f(distorted * 20.0, atan2(uvs.y, uvs.x) * 3.0), 2u, 0.6, 2.5);
  let fineGrain = smoothstep(0.3, 0.7, grain);

  // Combine ring and grain
  let woodColor = ring * 0.8 + fineGrain * 0.2;

  // Edge = ring boundary highlight
  let ringEdge = 1.0 - smoothstep(0.0, 0.06, abs(sin(distorted * ringFreq * PI)));

  return vec4f(woodColor, distorted * 0.2, ringEdge * 0.5, mix(0.45, 0.75, ring));
}

// ── 4: MARBLE_VEIN ───────────────────────────────────────────────────────────
// Turbulent sine function → marble veining.  Classic technique from
// Ken Perlin's original noise paper.
// R = marble base, G = vein depth, B = vein edge, A = roughness
fn marbleVein(uv: vec2f, scale: f32, octaves: u32, turb: f32) -> vec4f {
  let uvs = uv * scale;

  // Turbulence (absolute-value fBm — gives sharp creases)
  var turbulence = 0.0;
  var amp  = 0.5;
  var freq = 1.0;
  for (var i: u32 = 0u; i < octaves; i++) {
    turbulence += amp * abs(gradientNoise(uvs * freq) * 2.0 - 1.0);
    amp  *= 0.5;
    freq *= 2.0;
  }

  // Marble = sin(x + turbulence * strength)  — classic vein formula
  let veinParam = uvs.x * 2.0 + turbulence * turb * 5.0;
  let marble = sin(veinParam * PI) * 0.5 + 0.5;

  // Secondary vein set at a different angle for depth
  let vein2Param = uvs.y * 1.5 + turbulence * turb * 3.5;
  let vein2 = sin(vein2Param * PI) * 0.5 + 0.5;
  let combined = marble * 0.65 + vein2 * 0.35;

  // Vein edge detection
  let veinEdge = 1.0 - smoothstep(0.0, 0.15, abs(marble - 0.5));

  // Marble is generally smooth; roughness dips at vein centres
  let roughness = mix(0.2, 0.5, combined);

  return vec4f(combined, turbulence, veinEdge, roughness);
}

// ── compute entry point ──────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y;

  let atlasW = ap.cols * ap.tileSize;
  let atlasH = ap.rows * ap.tileSize;
  if (px >= atlasW || py >= atlasH) { return; }

  // Determine which tile this pixel belongs to
  let tileCol = px / ap.tileSize;
  let tileRow = py / ap.tileSize;
  let tileIdx = tileRow * ap.cols + tileCol;

  if (tileIdx >= ap.tileCount) {
    // Empty tile slot — write transparent black
    textureStore(outTex, vec2u(px, py), vec4f(0.0, 0.0, 0.0, 0.0));
    return;
  }

  // Local UV within the tile [0, 1]
  let localX = px % ap.tileSize;
  let localY = py % ap.tileSize;
  let uv = vec2f(
    f32(localX) / f32(ap.tileSize),
    1.0 - f32(localY) / f32(ap.tileSize), // flip Y for GL convention
  );

  let tile = tiles[tileIdx];

  var color: vec4f;
  switch (tile.kind) {
    case 0u: { color = voronoiMembrane(uv, tile.scale, tile.jitter, tile.octaves, tile.turbulence, ap.time); }
    case 1u: { color = perlinRock     (uv, tile.scale, tile.octaves, tile.turbulence); }
    case 2u: { color = turingStripe   (uv, tile.scale, tile.jitter, tile.octaves, tile.turbulence); }
    case 3u: { color = woodGrain      (uv, tile.scale, tile.octaves, tile.turbulence, ap.time); }
    case 4u: { color = marbleVein     (uv, tile.scale, tile.octaves, tile.turbulence); }
    default: { color = perlinRock     (uv, tile.scale, tile.octaves, tile.turbulence); }
  }

  // Tint the R channel by the species colour (subtle albedo bake)
  let tint = vec3f(tile.tintR, tile.tintG, tile.tintB);
  let tinted = vec4f(
    color.r * mix(1.0, tint.r, 0.3),
    color.g * mix(1.0, tint.g, 0.15),
    color.b,
    color.a,
  );

  textureStore(outTex, vec2u(px, py), tinted);
}
`;

// ─── ProceduralTextureAtlas class ───────────────────────────────────────────

/**
 * GPU-accelerated procedural texture atlas generator.
 *
 * Creates a single {@link GPUTexture} containing one tile per species, each
 * rendered with the appropriate procedural algorithm (Voronoi membrane, Perlin
 * rock, Turing stripe, wood grain, or marble vein).
 *
 * Cells sample from the atlas using {@link speciesAtlasUV} or the inlined
 * {@link ATLAS_UV_WGSL} snippet in their render shaders.
 *
 * @example
 * ```ts
 * const atlas = new ProceduralTextureAtlas(device);
 * const { texture, config } = await atlas.generate();
 * // Bind `texture` and pass `config.cols` / `config.rows` as uniforms
 * // In cell fragment shader: use speciesAtlasUV(idx, localUV, cols, rows)
 * atlas.destroy();
 * ```
 */
export class ProceduralTextureAtlas {
  private readonly device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  /** The most recently generated atlas texture (null until first generate). */
  private currentTexture: GPUTexture | null = null;

  /** The active atlas configuration. */
  private currentConfig: AtlasConfig | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // ── lazy pipeline init ──────────────────────────────────────────────────

  private async ensurePipeline(): Promise<void> {
    if (this.pipeline) return;

    const module = this.device.createShaderModule({
      label: 'procedural-texture-atlas-compute',
      code:  COMPUTE_SHADER_SRC,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'procedural-texture-atlas-bgl',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
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
      label:  'procedural-texture-atlas-pipeline',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: { module, entryPoint: 'main' },
    });
  }

  // ── public generate ─────────────────────────────────────────────────────

  /**
   * Generate (or regenerate) the full procedural texture atlas.
   *
   * If called multiple times the previous atlas texture is destroyed
   * automatically.  The returned texture is valid for binding immediately.
   *
   * @param tileSize  Per-tile resolution in px (default 256, power-of-two recommended).
   * @param cols      Atlas grid columns (default 4).
   * @param time      Animation time in seconds forwarded to time-varying patterns.
   * @returns         The atlas GPUTexture and its AtlasConfig metadata.
   */
  async generate(
    tileSize: number = DEFAULT_TILE_SIZE,
    cols: number = ATLAS_COLS,
    time: number = 0,
  ): Promise<{ texture: GPUTexture; config: AtlasConfig }> {
    await this.ensurePipeline();

    const config = buildAtlasConfig(tileSize, cols);
    this.currentConfig = config;

    // Destroy previous atlas
    if (this.currentTexture) {
      this.currentTexture.destroy();
      this.currentTexture = null;
    }

    // ── Uniform buffer: AtlasParams (32 bytes) ────────────────────────────
    const uniformData = new ArrayBuffer(32);
    const uView = new DataView(uniformData);
    uView.setUint32 ( 0, config.tileSize,         true);
    uView.setUint32 ( 4, config.cols,              true);
    uView.setUint32 ( 8, config.rows,              true);
    uView.setUint32 (12, config.tiles.length,      true);
    uView.setFloat32(16, time,                     true);
    uView.setFloat32(20, 0,                        true); // _pad0
    uView.setFloat32(24, 0,                        true); // _pad1
    uView.setFloat32(28, 0,                        true); // _pad2

    const uniformBuf = this.device.createBuffer({
      label: 'procedural-atlas-uniforms',
      size:  32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuf, 0, uniformData);

    // ── Storage buffer: TileData array (48 bytes per tile) ────────────────
    const TILE_STRIDE = 48; // 12 × f32
    const tilesBufSize = config.tiles.length * TILE_STRIDE;
    const tilesData = new ArrayBuffer(tilesBufSize);
    const tView = new DataView(tilesData);

    for (let t = 0; t < config.tiles.length; t++) {
      const tile = config.tiles[t];
      const off = t * TILE_STRIDE;
      tView.setUint32  (off +  0, TEXTURE_KIND_INDEX[tile.kind], true);
      tView.setFloat32 (off +  4, tile.scale,                   true);
      tView.setFloat32 (off +  8, tile.jitter,                  true);
      tView.setUint32  (off + 12, tile.octaves,                 true);
      tView.setFloat32 (off + 16, tile.turbulence,              true);
      tView.setFloat32 (off + 20, tile.tintR,                   true);
      tView.setFloat32 (off + 24, tile.tintG,                   true);
      tView.setFloat32 (off + 28, tile.tintB,                   true);
      tView.setFloat32 (off + 32, 0,                            true); // _pad0
      tView.setFloat32 (off + 36, 0,                            true); // _pad1
      tView.setFloat32 (off + 40, 0,                            true); // _pad2
      tView.setFloat32 (off + 44, 0,                            true); // _pad3
    }

    const tilesBuf = this.device.createBuffer({
      label: 'procedural-atlas-tiles',
      size:  tilesBufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(tilesBuf, 0, tilesData);

    // ── Output atlas texture ──────────────────────────────────────────────
    const texture = this.device.createTexture({
      label:  `procedural-texture-atlas-${config.width}x${config.height}`,
      size:   { width: config.width, height: config.height, depthOrArrayLayers: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.COPY_SRC,
    });
    this.currentTexture = texture;

    // ── Bind group ────────────────────────────────────────────────────────
    const bindGroup = this.device.createBindGroup({
      label:  'procedural-atlas-bg',
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: tilesBuf   } },
        { binding: 2, resource: texture.createView()   },
      ],
    });

    // ── Dispatch ──────────────────────────────────────────────────────────
    const enc  = this.device.createCommandEncoder({ label: 'procedural-atlas-enc' });
    const pass = enc.beginComputePass({ label: 'procedural-atlas-pass' });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(config.width  / 8),
      Math.ceil(config.height / 8),
    );
    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Cleanup transient buffers
    uniformBuf.destroy();
    tilesBuf.destroy();

    return { texture, config };
  }

  // ── convenience: CPU readback ──────────────────────────────────────────

  /**
   * Read the atlas texture back to CPU as RGBA bytes.
   * Useful for debugging / offline export / canvas rendering.
   */
  async readback(): Promise<{ data: Uint8ClampedArray; width: number; height: number } | null> {
    if (!this.currentTexture || !this.currentConfig) return null;

    const { width, height } = this.currentConfig;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const stagingBuf = this.device.createBuffer({
      size:  bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: this.currentTexture },
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

    return { data: rgba, width, height };
  }

  // ── accessors ──────────────────────────────────────────────────────────

  /** The current atlas GPUTexture (null before first generate). */
  get texture(): GPUTexture | null {
    return this.currentTexture;
  }

  /** The current AtlasConfig (null before first generate). */
  get config(): AtlasConfig | null {
    return this.currentConfig;
  }

  // ── lifecycle ──────────────────────────────────────────────────────────

  /** Destroy the atlas texture and GPU pipeline resources. */
  destroy(): void {
    if (this.currentTexture) {
      this.currentTexture.destroy();
      this.currentTexture = null;
    }
    this.currentConfig   = null;
    this.pipeline        = null;
    this.bindGroupLayout = null;
  }
}

// ─── Re-exports ─────────────────────────────────────────────────────────────

export { COMPUTE_SHADER_SRC as PROCEDURAL_ATLAS_WGSL };
