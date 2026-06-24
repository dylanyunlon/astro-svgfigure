/**
 * cell-aura.ts — M795: Cell Aura Energy Field
 * ─────────────────────────────────────────────────────────────────────────────
 * Every Cell is surrounded by a shimmering energy-field aura — a radiant halo
 * whose shape is governed by a signed-distance field, colour derived from the
 * cell's species, and intensity driven by physical activity (collision count,
 * local particle density, kinetic energy).
 *
 * Visual recipe
 * ─────────────────────────────────────────────────────────────────────────────
 *   1. **SDF silhouette** — the aura inherits the cell's species SDF (flower,
 *      koch, supershape …) from species-shader-registry.ts, evaluated at
 *      expanded scale so the glow extends beyond the cell boundary.
 *   2. **Species colour** — base hue from the same 11-entry palette as
 *      curl-aura.ts; inner core saturated, outer corona desaturated/whitened.
 *   3. **Physics intensity** — the aura "breathes" in response to physical
 *      activity: collision contacts brighten it, high local density widens
 *      the glow, and kinetic energy shifts hue toward hot white.
 *   4. **Fresnel edge highlight** — a Schlick Fresnel approximation on the
 *      SDF gradient produces a bright rim at grazing angles, making the aura
 *      shimmer as the camera-relative view normal changes.
 *   5. **Noise perturbation** — fBm noise warps the SDF coordinate space,
 *      producing organic breathing edges that avoid mechanical regularity.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 * The system has two halves:
 *
 *   CellAuraSystem  (CPU)
 *     • Maintains a pool of CellAuraState entries, one per visible cell.
 *     • Each frame, receives physics snapshots (PhysicsUniforms) and cell
 *       registry data (position, bbox, species) to update aura parameters.
 *     • Packs per-instance data into a Float32Array for GPU upload.
 *
 *   CellAuraPass  (WebGL2 instanced draw)
 *     • Screen-aligned quads (6-vertex instanced pattern from M745).
 *     • Fragment shader evaluates the SDF + Fresnel + noise pipeline.
 *     • Renders with additive blending BEFORE bloom so the post-process
 *       bloom picks up the aura glow naturally.
 *
 * Per-instance data layout (AURA_FLOATS_PER_CELL = 16 floats):
 *   [ posX, posY, halfW, halfH,                       // geometry
 *     speciesIdx, sdfKind, sdfParam0, sdfParam1,       // shape identity
 *     phase, time,                                     // animation
 *     collisionIntensity, density, kineticEnergy,      // physics drivers
 *     fresnelPower, noiseScale, opacity ]               // visual tuning
 *
 * Integration
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const system = new CellAuraSystem(256);
 *   const pass   = new CellAuraPass(256);
 *   pass.attach(gl);                               // shared WebGL2 context
 *
 *   // per frame:
 *   system.updateFromRegistry(cells, physicsMap);
 *   const packed = system.pack();
 *   pass.upload(packed.buffer, packed.count);
 *   pass.render(projectionMatrix, elapsedSec);
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/curl-aura.ts              — M749 curl-noise aura (peer effect)
 *   src/lib/sph/species-shader-registry.ts — species SDF shape config
 *   src/lib/sph/organic-sdf.ts            — CPU SDF evaluation
 *   src/lib/sph/cell-visual-identity.ts   — species → morphology mapping
 *   src/lib/sph/cell-material-system.ts   — PBR + Fresnel material pipeline
 *   src/lib/sph/physics-uniform-bridge.ts — PhysicsUniforms definition
 *   src/lib/sph/particle-instancing.ts    — M745 instanced quad pattern
 *   src/lib/sph/collision-shockwave.ts    — collision-driven visual fx
 *
 * Research: xiaodi #M795 — cell-pubsub-loop
 */

import type { PhysicsUniforms } from './physics-uniform-bridge';
import type { SdfShape }        from './species-shader-registry';

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Constants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** Floats per aura instance in the interleaved GPU buffer. */
export const AURA_FLOATS_PER_CELL = 16;

/** Maximum number of active SDF kinds supported in the shader. */
const SDF_KIND_COUNT = 9;

/**
 * SDF shape → integer enum for the GPU.
 * Must stay in sync with the switch in the fragment shader.
 */
const SDF_KIND_MAP: Record<SdfShape, number> = {
  flower:     0,
  koch:       1,
  julia:      2,
  supershape: 3,
  capsule:    4,
  hexagon:    5,
  star:       6,
  roundbox:   7,
  polygon:    8,
};

// ─── Species colour palette (normalised RGB, matches curl-aura.ts M749) ────

const SPECIES_COLOR_PALETTE: [number, number, number][] = [
  [0.247, 0.318, 0.710],  // 0  cil-eye       #3F51B5
  [1.000, 0.435, 0.000],  // 1  cil-bolt      #FF6F00
  [0.180, 0.490, 0.196],  // 2  cil-vector    #2E7D32
  [0.776, 0.157, 0.157],  // 3  cil-plus      #C62828
  [0.271, 0.353, 0.392],  // 4  cil-arrow     #455A64
  [0.482, 0.122, 0.635],  // 5  cil-filter    #7B1FA2
  [0.180, 0.490, 0.196],  // 6  cil-code      #2E7D32
  [0.084, 0.396, 0.753],  // 7  cil-layers    #1565C0
  [0.961, 0.498, 0.090],  // 8  cil-loop      #F57F17
  [0.216, 0.278, 0.310],  // 9  cil-graph     #37474F
  [1.000, 1.000, 1.000],  // 10 fallback white
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  CPU-side: CellAuraState & CellAuraSystem
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** Descriptor for a single cell's aura state on the CPU side. */
export interface CellAuraState {
  /** Cell center in world coordinates. */
  posX: number;
  posY: number;
  /** Cell bounding-box half-extents. */
  halfW: number;
  halfH: number;
  /** Species index (0–10). */
  speciesIdx: number;
  /** SDF shape kind (from SDF_KIND_MAP). */
  sdfKind: number;
  /** Primary SDF parameter (petals count / iterations / supershape m). */
  sdfParam0: number;
  /** Secondary SDF parameter (julia re / supershape n1 / etc). */
  sdfParam1: number;
  /** Per-cell phase offset (decorrelates animation). */
  phase: number;

  // ─── Physics-driven intensity ────────────────────────────────────────
  /** Collision intensity: how many active contacts, smoothed. */
  collisionIntensity: number;
  /** Local particle density (from SPH neighbour count). */
  density: number;
  /** Local kinetic energy (drives hot-shift). */
  kineticEnergy: number;

  // ─── Visual tuning ──────────────────────────────────────────────────
  /** Fresnel exponent — higher = tighter rim. Default 3.0. */
  fresnelPower: number;
  /** Noise perturbation scale. Default 1.0. */
  noiseScale: number;
  /** Base opacity (0–1). Default 0.7. */
  opacity: number;
}

/** Input descriptor for feeding cells into the system. */
export interface CellAuraInput {
  id: string;
  posX: number;
  posY: number;
  halfW: number;
  halfH: number;
  species?: string | number;
  sdfShape?: SdfShape;
  sdfParam0?: number;
  sdfParam1?: number;
}

/** Packed result ready for GPU upload. */
export interface PackedAuraBuffer {
  buffer: Float32Array;
  count: number;
}

/**
 * CellAuraSystem — CPU-side manager for cell energy-field aura states.
 *
 * Maintains a pool of `CellAuraState` entries, updates them from physics
 * snapshots each frame, and packs the result into a contiguous Float32Array
 * for GPU upload via `CellAuraPass.upload()`.
 */
export class CellAuraSystem {
  readonly maxCells: number;
  private states: Map<string, CellAuraState> = new Map();
  private packBuffer: Float32Array;

  /** Smoothing factor for physics-driven values (exponential decay). */
  private readonly smoothAlpha: number;

  constructor(maxCells = 256, smoothAlpha = 0.12) {
    this.maxCells = maxCells;
    this.smoothAlpha = smoothAlpha;
    this.packBuffer = new Float32Array(maxCells * AURA_FLOATS_PER_CELL);
  }

  /**
   * Update aura states from the cell registry and physics snapshots.
   *
   * @param cells     Array of cell descriptors (position, species, SDF shape).
   * @param physicsMap Map from cell id → PhysicsUniforms snapshot.
   */
  updateFromRegistry(
    cells: ReadonlyArray<CellAuraInput>,
    physicsMap: ReadonlyMap<string, PhysicsUniforms>,
  ): void {
    const seen = new Set<string>();

    for (const cell of cells) {
      if (seen.size >= this.maxCells) break;
      seen.add(cell.id);

      const physics = physicsMap.get(cell.id);
      let state = this.states.get(cell.id);

      if (!state) {
        // First appearance — initialise with defaults
        state = {
          posX: cell.posX,
          posY: cell.posY,
          halfW: cell.halfW,
          halfH: cell.halfH,
          speciesIdx: resolveSpeciesIndex(cell.species),
          sdfKind: cell.sdfShape ? (SDF_KIND_MAP[cell.sdfShape] ?? 0) : 0,
          sdfParam0: cell.sdfParam0 ?? 5,
          sdfParam1: cell.sdfParam1 ?? 0,
          phase: seen.size * 2.399,  // golden angle decorrelation
          collisionIntensity: 0,
          density: 0,
          kineticEnergy: 0,
          fresnelPower: 3.0,
          noiseScale: 1.0,
          opacity: 0.7,
        };
        this.states.set(cell.id, state);
      }

      // Update geometry (no smoothing — track exactly)
      state.posX  = cell.posX;
      state.posY  = cell.posY;
      state.halfW = cell.halfW;
      state.halfH = cell.halfH;

      // Smooth physics-driven values
      if (physics) {
        const a = this.smoothAlpha;
        state.collisionIntensity = lerp(
          state.collisionIntensity,
          clamp01(physics.u_contactCount / 4),
          a,
        );
        state.density = lerp(
          state.density,
          clamp01(physics.u_neighborCount / 30),
          a,
        );
        state.kineticEnergy = lerp(
          state.kineticEnergy,
          clamp01(physics.u_kineticEnergy / 500),
          a,
        );
      }
    }

    // Prune cells that disappeared
    for (const key of this.states.keys()) {
      if (!seen.has(key)) this.states.delete(key);
    }
  }

  /**
   * Manually set or override a single cell's aura state.
   * Useful for scripted effects or debug visualisation.
   */
  set(id: string, partial: Partial<CellAuraState>): void {
    const existing = this.states.get(id);
    if (existing) {
      Object.assign(existing, partial);
    }
  }

  /** Override Fresnel power for all active cells. */
  setGlobalFresnel(power: number): void {
    for (const s of this.states.values()) s.fresnelPower = power;
  }

  /** Override noise scale for all active cells. */
  setGlobalNoiseScale(scale: number): void {
    for (const s of this.states.values()) s.noiseScale = scale;
  }

  /**
   * Pack all active aura states into the interleaved Float32Array.
   * Returns { buffer, count } — pass directly to CellAuraPass.upload().
   */
  pack(): PackedAuraBuffer {
    let i = 0;
    const buf = this.packBuffer;

    for (const s of this.states.values()) {
      if (i >= this.maxCells) break;
      const off = i * AURA_FLOATS_PER_CELL;

      buf[off]      = s.posX;
      buf[off + 1]  = s.posY;
      buf[off + 2]  = s.halfW;
      buf[off + 3]  = s.halfH;
      buf[off + 4]  = s.speciesIdx;
      buf[off + 5]  = s.sdfKind;
      buf[off + 6]  = s.sdfParam0;
      buf[off + 7]  = s.sdfParam1;
      buf[off + 8]  = s.phase;
      buf[off + 9]  = 0;  // time — filled by GPU uniform
      buf[off + 10] = s.collisionIntensity;
      buf[off + 11] = s.density;
      buf[off + 12] = s.kineticEnergy;
      buf[off + 13] = s.fresnelPower;
      buf[off + 14] = s.noiseScale;
      buf[off + 15] = s.opacity;

      i++;
    }

    return { buffer: buf, count: i };
  }

  /** Number of active aura entries. */
  get count(): number { return this.states.size; }

  /** Remove all aura states. */
  clear(): void { this.states.clear(); }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  GPU-side: CellAuraPass (WebGL2 instanced renderer)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ─── Vertex shader ─────────────────────────────────────────────────────────

const VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

uniform mat4  u_projection;

// Screen-aligned quad (2 triangles)
const vec2 QUAD[6] = vec2[6](
  vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

// Per-instance attributes (16 floats, packed as 4 × vec4)
layout(location = 0) in vec4 a_geom;       // posX, posY, halfW, halfH
layout(location = 1) in vec4 a_shape;      // speciesIdx, sdfKind, sdfParam0, sdfParam1
layout(location = 2) in vec4 a_anim;       // phase, time(unused), collisionIntensity, density
layout(location = 3) in vec4 a_visual;     // kineticEnergy, fresnelPower, noiseScale, opacity

out vec2  v_uv;
out vec2  v_worldPos;
flat out float v_speciesIdx;
flat out float v_sdfKind;
flat out float v_sdfParam0;
flat out float v_sdfParam1;
out float v_phase;
out float v_collisionIntensity;
out float v_density;
out float v_kineticEnergy;
flat out float v_fresnelPower;
flat out float v_noiseScale;
out float v_opacity;

void main() {
  vec2 uv = QUAD[gl_VertexID];

  // Expand the quad to cover the cell bbox + generous aura overshoot
  float auraExpand = max(a_geom.z, a_geom.w) * 0.8 + 30.0;
  vec2 halfExtent = vec2(a_geom.z + auraExpand, a_geom.w + auraExpand);
  vec2 worldPos   = a_geom.xy + uv * halfExtent;

  gl_Position = u_projection * vec4(worldPos, 0.0, 1.0);

  v_uv                 = uv;
  v_worldPos           = worldPos;
  v_speciesIdx         = a_shape.x;
  v_sdfKind            = a_shape.y;
  v_sdfParam0          = a_shape.z;
  v_sdfParam1          = a_shape.w;
  v_phase              = a_anim.x;
  v_collisionIntensity = a_anim.z;
  v_density            = a_anim.w;
  v_kineticEnergy      = a_visual.x;
  v_fresnelPower       = a_visual.y;
  v_noiseScale         = a_visual.z;
  v_opacity            = a_visual.w;
}
`;

// ─── Fragment shader ───────────────────────────────────────────────────────

const FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

uniform float u_time;
uniform vec3  u_palette[11];

in vec2  v_uv;
in vec2  v_worldPos;
flat in float v_speciesIdx;
flat in float v_sdfKind;
flat in float v_sdfParam0;
flat in float v_sdfParam1;
in float v_phase;
in float v_collisionIntensity;
in float v_density;
in float v_kineticEnergy;
flat in float v_fresnelPower;
flat in float v_noiseScale;
in float v_opacity;

out vec4 fragColor;

// ────────────────────────────────────────────────────────────────────────────
//  Noise utilities (inlined, no external deps)
// ────────────────────────────────────────────────────────────────────────────

// Hash without sin — integer-based, from Dave Hoskins
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);   // Hermite smoothstep

  float a = dot(hash22(i + vec2(0.0, 0.0)), vec2(1.0));
  float b = dot(hash22(i + vec2(1.0, 0.0)), vec2(1.0));
  float c = dot(hash22(i + vec2(0.0, 1.0)), vec2(1.0));
  float d = dot(hash22(i + vec2(1.0, 1.0)), vec2(1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractional Brownian motion — 5 octaves for rich detail
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);  // decorrelating rotation
  for (int i = 0; i < 5; i++) {
    v += a * valueNoise(p);
    p = rot * p * 2.0;
    a *= 0.5;
  }
  return v;
}

// ────────────────────────────────────────────────────────────────────────────
//  SDF shapes (matches SDF_KIND_MAP on CPU)
// ────────────────────────────────────────────────────────────────────────────

// 0: flower — petals around origin
float sdfFlower(vec2 st, float petals) {
  float r = length(st) * 2.0;
  float a = atan(st.y, st.x);
  float n = max(petals, 2.0);
  return 1.0 - (abs(cos(a * n * 0.5)) * 0.5 + 0.5) / max(r, 0.001);
}

// 1: koch snowflake (simplified — hexagonal with fractal bump)
float sdfKoch(vec2 st, float iter) {
  float r = length(st);
  float a = atan(st.y, st.x);
  float n = 6.0;
  float bump = 0.0;
  float scale = 1.0;
  for (int i = 0; i < 4; i++) {
    if (float(i) >= iter) break;
    bump += abs(sin(a * n)) * scale * 0.15;
    n *= 2.0;
    scale *= 0.5;
  }
  return r - (0.5 + bump);
}

// 2: julia set boundary (simplified radial approximation)
float sdfJulia(vec2 st, float cx, float cy) {
  vec2 z = st * 2.0;
  vec2 c = vec2(cx, cy);
  float lastLen = 0.0;
  for (int i = 0; i < 12; i++) {
    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    lastLen = length(z);
    if (lastLen > 4.0) {
      float smooth_i = float(i) - log2(log2(lastLen));
      return smooth_i / 12.0 - 0.5;
    }
  }
  return -0.3;
}

// 3: supershape (Gielis formula)
float sdfSupershape(vec2 st, float m, float n1) {
  float r = length(st);
  float a = atan(st.y, st.x);
  float t1 = abs(cos(m * a / 4.0));
  float t2 = abs(sin(m * a / 4.0));
  float rShape = pow(pow(t1, n1) + pow(t2, n1), -1.0 / n1);
  return r - rShape * 0.5;
}

// 4: capsule
float sdfCapsule(vec2 p) {
  p.x = abs(p.x) - 0.2;
  return length(vec2(max(p.x, 0.0), p.y)) - 0.3;
}

// 5: hexagon
float sdfHexagon(vec2 p) {
  p = abs(p);
  return max(p.x * 0.866025 + p.y * 0.5, p.y) - 0.45;
}

// 6: star
float sdfStar(vec2 p, float n) {
  float r = length(p);
  float a = atan(p.y, p.x);
  float f = cos(floor(0.5 + a / 6.2831 * n) * 6.2831 / n - a) * r;
  return f - 0.35;
}

// 7: rounded box
float sdfRoundBox(vec2 p) {
  vec2 d = abs(p) - vec2(0.35, 0.25);
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - 0.08;
}

// 8: polygon (regular ngon)
float sdfPolygon(vec2 p, float n) {
  float a = atan(p.y, p.x);
  float r = length(p);
  float an = 6.2831 / n;
  float he = 0.45 * cos(an * 0.5);
  float bn = mod(a, an) - an * 0.5;
  return r * cos(bn) - he;
}

// Dispatcher — selects SDF by kind index
float evalSDF(vec2 st, float kind, float p0, float p1) {
  int k = int(kind + 0.5);
  if (k == 0) return sdfFlower(st, p0);
  if (k == 1) return sdfKoch(st, p0);
  if (k == 2) return sdfJulia(st, p0, p1);
  if (k == 3) return sdfSupershape(st, p0, p1);
  if (k == 4) return sdfCapsule(st);
  if (k == 5) return sdfHexagon(st);
  if (k == 6) return sdfStar(st, p0);
  if (k == 7) return sdfRoundBox(st);
  if (k == 8) return sdfPolygon(st, p0);
  // Fallback: circle
  return length(st) - 0.4;
}

// ────────────────────────────────────────────────────────────────────────────
//  Fresnel edge highlight (Schlick approximation on SDF gradient)
// ────────────────────────────────────────────────────────────────────────────

// Approximate SDF gradient via central differences → acts as a "normal"
vec2 sdfGradient(vec2 st, float kind, float p0, float p1) {
  float eps = 0.005;
  float dx = evalSDF(st + vec2(eps, 0.0), kind, p0, p1)
           - evalSDF(st - vec2(eps, 0.0), kind, p0, p1);
  float dy = evalSDF(st + vec2(0.0, eps), kind, p0, p1)
           - evalSDF(st - vec2(0.0, eps), kind, p0, p1);
  return normalize(vec2(dx, dy) + 1e-8);
}

float fresnelSchlick(float cosTheta, float power) {
  return pow(1.0 - clamp(cosTheta, 0.0, 1.0), power);
}

// ────────────────────────────────────────────────────────────────────────────
//  Main fragment
// ────────────────────────────────────────────────────────────────────────────

void main() {
  float dist = length(v_uv);
  if (dist > 1.0) discard;

  float t = u_time * 0.06 + v_phase;

  // ── 1. Noise perturbation of UV coordinates ─────────────────────────
  vec2 noiseDomain = v_worldPos * 0.01 * v_noiseScale + vec2(t * 0.5, t * 0.25);
  float noiseVal   = fbm(noiseDomain);

  // Warp UVs — stronger at edges for organic breathing boundary
  float edgeBlend   = smoothstep(0.15, 0.7, dist);
  vec2 warpOffset   = vec2(
    fbm(noiseDomain + vec2(7.3, 1.7)) - 0.5,
    fbm(noiseDomain + vec2(3.1, 8.9)) - 0.5
  );
  vec2 warpedUV = v_uv + warpOffset * 0.10 * edgeBlend * v_noiseScale;

  // ── 2. Evaluate species SDF ─────────────────────────────────────────
  // Scale UVs into SDF evaluation space (centred on origin, roughly [-1,1])
  vec2 sdfUV = warpedUV * 1.2;  // slight overshoot so aura extends past shape

  float sdfDist = evalSDF(sdfUV, v_sdfKind, v_sdfParam0, v_sdfParam1);

  // ── 3. Aura band from SDF ──────────────────────────────────────────
  // The aura lives in the region just outside the SDF boundary
  // Inner edge: smoothly fades from inside the shape
  // Outer edge: falls off into empty space

  // Physics-responsive aura width: density widens, collision brightens
  float auraWidth = 0.25 + v_density * 0.15 + v_collisionIntensity * 0.10;

  // Core glow (inside the SDF shape)
  float coreGlow = smoothstep(0.05, -0.15, sdfDist);
  coreGlow *= 0.35;

  // Aura band (just outside the SDF boundary)
  float bandStart = 0.0;
  float bandEnd   = auraWidth;
  float auraBand  = smoothstep(bandStart - 0.02, bandStart + 0.03, sdfDist)
                  * (1.0 - smoothstep(bandEnd * 0.6, bandEnd, sdfDist));

  // Turbulence modulation within the band
  float turbulence = noiseVal * 0.6 + 0.4;
  auraBand *= turbulence;

  // Pulsing animation driven by collision intensity
  float pulse = sin(u_time * (1.5 + v_collisionIntensity * 3.0) + v_phase);
  pulse = pulse * 0.5 + 0.5;
  float pulseStrength = 0.15 + v_collisionIntensity * 0.35;
  auraBand *= (1.0 - pulseStrength + pulseStrength * pulse);

  // ── 4. Fresnel edge highlight ───────────────────────────────────────
  // Use the SDF gradient as a "surface normal" — the view direction is
  // approximated as radial from center (since we're in screen space)
  vec2 grad       = sdfGradient(sdfUV, v_sdfKind, v_sdfParam0, v_sdfParam1);
  vec2 viewDir    = normalize(v_uv + 1e-6);
  float cosTheta  = abs(dot(grad, viewDir));
  float fresnel   = fresnelSchlick(cosTheta, v_fresnelPower);

  // Fresnel is strongest near the SDF boundary
  float nearBoundary = 1.0 - smoothstep(0.0, auraWidth * 0.5, abs(sdfDist));
  fresnel *= nearBoundary;

  // Collision intensity amplifies the fresnel rim
  fresnel *= (0.6 + v_collisionIntensity * 0.8);

  // ── 5. Species colour with physics-driven shift ─────────────────────
  int idx = clamp(int(v_speciesIdx + 0.5), 0, 10);
  vec3 baseColor = u_palette[idx];

  // Kinetic energy drives warm/hot shift
  float ke = clamp(v_kineticEnergy, 0.0, 1.0);
  vec3 hotColor  = vec3(1.0, 0.92, 0.75);
  vec3 auraColor = mix(baseColor, hotColor, ke * ke * 0.5);

  // Fresnel rim gets a brighter, slightly desaturated highlight
  vec3 rimColor = mix(auraColor, vec3(1.0), 0.4) * 1.3;

  // Outer region fades toward a softer, desaturated version
  float outerFade = smoothstep(0.05, auraWidth * 0.8, sdfDist);
  vec3 outerColor = mix(auraColor, auraColor * 0.5 + 0.15, outerFade);

  // ── 6. Compose layers ───────────────────────────────────────────────
  // Core: species color, full saturation
  // Band: aura glow, physics-modulated
  // Rim:  Fresnel highlight, bright

  float intensity = coreGlow + auraBand * 0.7 + fresnel * 0.5;

  // Density boosts overall brightness
  intensity *= (0.75 + v_density * 0.5);
  // Kinetic energy adds a global brightness lift
  intensity *= (0.8 + ke * 0.4);

  // Apply base opacity
  intensity *= v_opacity;

  float alpha = clamp(intensity, 0.0, 1.0);

  // Smooth outer edge fadeout
  alpha *= 1.0 - smoothstep(0.85, 1.0, dist);

  if (alpha < 0.003) discard;

  // Blend between core colour, aura colour, and fresnel rim
  vec3 finalColor = mix(outerColor, auraColor, coreGlow / max(intensity, 0.01));
  finalColor += rimColor * fresnel * 0.5;

  // Core tip brightness
  finalColor += baseColor * coreGlow * 0.25;

  // Premultiplied alpha for additive-friendly compositing
  fragColor = vec4(finalColor * alpha, alpha);
}
`;

// ─── Shader compilation helpers ────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('CellAura: failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`CellAura: shader compile error:\n${info}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('CellAura: failed to create program');
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`CellAura: program link error:\n${info}`);
  }
  return program;
}

// ─── CellAuraPass ──────────────────────────────────────────────────────────

export interface CellAuraPassOptions {
  /** Additive blending (default true). */
  additiveBlend?: boolean;
}

/**
 * CellAuraPass — GPU instanced renderer for cell energy-field auras.
 *
 * Shares a WebGL2 context with the existing particle instancing pipeline.
 * Each cell gets a screen-aligned quad; the fragment shader evaluates the
 * full SDF + Fresnel + noise + physics pipeline per fragment.
 *
 * Render order: after particle instancing, before bloom post-process.
 *
 * ```
 *   pass.upload(packed.buffer, packed.count);
 *   pass.render(projectionMatrix, elapsedSeconds);
 * ```
 */
export class CellAuraPass {
  readonly maxCells: number;
  private additiveBlend: boolean;

  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private instanceBuffer: WebGLBuffer | null = null;

  // Uniforms
  private uProjection: WebGLUniformLocation | null = null;
  private uTime: WebGLUniformLocation | null = null;
  private uPalette: WebGLUniformLocation | null = null;

  private liveCount = 0;

  constructor(maxCells = 256, options: CellAuraPassOptions = {}) {
    this.maxCells = maxCells;
    this.additiveBlend = options.additiveBlend ?? true;
  }

  // ─── Lifecycle ───────────────────────────────────────────────────────

  /**
   * Attach to an existing WebGL2 context (shared with particle pipeline).
   */
  attach(gl: WebGL2RenderingContext): this {
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    this.program = linkProgram(gl, vs, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Resolve uniforms
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection');
    this.uTime       = gl.getUniformLocation(this.program, 'u_time');
    this.uPalette    = gl.getUniformLocation(this.program, 'u_palette');

    // Instance buffer
    const byteSize = this.maxCells * AURA_FLOATS_PER_CELL * 4;
    this.instanceBuffer = gl.createBuffer();
    if (!this.instanceBuffer) throw new Error('CellAura: failed to create buffer');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, byteSize, gl.DYNAMIC_DRAW);

    // VAO — 4 × vec4 per instance (16 floats = 64 bytes stride)
    this.vao = gl.createVertexArray();
    if (!this.vao) throw new Error('CellAura: failed to create VAO');
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    const stride = AURA_FLOATS_PER_CELL * 4; // 64 bytes

    for (let loc = 0; loc < 4; loc++) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, stride, loc * 16);
      gl.vertexAttribDivisor(loc, 1);  // per instance
    }

    gl.bindVertexArray(null);
    return this;
  }

  /**
   * Attach directly to a canvas (creates own WebGL2 context).
   */
  attachCanvas(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    attrs: WebGLContextAttributes = {},
  ): this {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      ...attrs,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('CellAura: WebGL2 not available');
    return this.attach(gl);
  }

  // ─── Data upload ─────────────────────────────────────────────────────

  /**
   * Upload packed aura data to the GPU.
   *
   * @param data  Interleaved Float32Array from CellAuraSystem.pack().
   * @param count Number of active cells.
   */
  upload(data: Float32Array, count: number): void {
    if (!this.gl || !this.instanceBuffer) return;
    const gl = this.gl;

    this.liveCount = Math.min(count, this.maxCells);
    const floats = this.liveCount * AURA_FLOATS_PER_CELL;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (floats <= data.length) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, floats);
    } else {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    }
  }

  // ─── Render ──────────────────────────────────────────────────────────

  /**
   * Render all cell auras.
   *
   * Call AFTER particle instancing, BEFORE bloom post-process.
   *
   * @param projectionMatrix  4×4 column-major ortho matrix (same as particle pipeline).
   * @param time              Elapsed time in seconds.
   */
  render(projectionMatrix: Float32Array, time: number): void {
    if (!this.gl || !this.program || !this.vao || this.liveCount === 0) return;
    const gl = this.gl;

    gl.useProgram(this.program);

    // Blending
    gl.enable(gl.BLEND);
    if (this.additiveBlend) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    } else {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
    gl.disable(gl.DEPTH_TEST);

    // Uniforms
    gl.uniformMatrix4fv(this.uProjection, false, projectionMatrix);
    gl.uniform1f(this.uTime, time);

    // Flatten species palette
    const flat = new Float32Array(SPECIES_COLOR_PALETTE.length * 3);
    for (let i = 0; i < SPECIES_COLOR_PALETTE.length; i++) {
      flat[i * 3]     = SPECIES_COLOR_PALETTE[i][0];
      flat[i * 3 + 1] = SPECIES_COLOR_PALETTE[i][1];
      flat[i * 3 + 2] = SPECIES_COLOR_PALETTE[i][2];
    }
    gl.uniform3fv(this.uPalette, flat);

    // Draw instanced quads
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.liveCount);
    gl.bindVertexArray(null);

    // Restore standard alpha blend
    if (this.additiveBlend) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
  }

  // ─── Cleanup ─────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;
    if (!gl) return;
    if (this.vao) { gl.deleteVertexArray(this.vao); this.vao = null; }
    if (this.instanceBuffer) { gl.deleteBuffer(this.instanceBuffer); this.instanceBuffer = null; }
    if (this.program) { gl.deleteProgram(this.program); this.program = null; }
    this.gl = null;
    this.liveCount = 0;
  }

  // ─── Tunables ────────────────────────────────────────────────────────

  setAdditiveBlend(v: boolean): void { this.additiveBlend = v; }
  getContext(): WebGL2RenderingContext | null { return this.gl; }
  getCount(): number { return this.liveCount; }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Convenience helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ─── Species string → index (matches curl-aura.ts) ─────────────────────────

const SPECIES_KEYWORDS: [string[], number][] = [
  [['eye', 'attn', 'attention', 'self_attn'],             0],
  [['bolt', 'ffn', 'feed_forward', 'mlp'],                1],
  [['vector', 'embed', 'input_embed', 'pos_encode'],      2],
  [['plus', 'residual', 'add_norm'],                      3],
  [['arrow', 'output', 'softmax'],                        4],
  [['filter', 'mask', 'dropout'],                         5],
  [['code', 'function'],                                  6],
  [['layers', 'layer_norm', 'norm'],                      7],
  [['loop', 'feedback', 'recurrent'],                     8],
  [['graph', 'network', 'struct'],                        9],
];

function resolveSpeciesIndex(species: string | number | undefined): number {
  if (species == null) return 10;
  if (typeof species === 'number') return Math.min(species, 10);
  const lower = species.toLowerCase();
  for (const [keywords, idx] of SPECIES_KEYWORDS) {
    for (const kw of keywords) {
      if (lower.includes(kw)) return idx;
    }
  }
  return 10;
}

/** Resolve an SDF shape string to its GPU enum. */
export function sdfShapeToKind(shape: SdfShape): number {
  return SDF_KIND_MAP[shape] ?? 0;
}

// ─── Math helpers ──────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
