/**
 * at-spline-particle.ts — M713
 *
 * AT SplineParticleInstance + SplineParticleLife + SplineParticlePreset
 * ─────────────────────────────────────────────────────────────────────────────
 * Full WebGPU / WGSL port of Active Theory's core particle lifecycle system:
 *
 *   SplineParticleInstance  — single particle slot: position, lifecycle state,
 *                             speed, alpha, travel fraction, and curl-noise seed.
 *                             Mirrors AT's per-particle GPGPU data layout.
 *
 *   ATSplineParticleLife    — GPU orchestrator: encodes the compute pass that
 *                             advances all particle instances (SPAWN → FLOW →
 *                             DECAY → DEAD → respawn), writes to a tPos
 *                             ping-pong texture, and fires the SPH handoff
 *                             callback via async readback.
 *
 *   SplineParticlePreset    — named configuration bundles that match AT's UIL
 *                             parameter presets (slow-flow, fast-pulse, organic,
 *                             dense-swarm, etc.) for one-line scene setup.
 *
 * Core AT lifecycle (SplineParticleLife.fs + FlowerParticleShader.glsl):
 *   SPAWN  (phase 0) — particle waits out its spawn-delay countdown
 *   FLOW   (phase 1) — particle advances along Catmull-Rom spline with curl
 *                      noise lateral perturbation; fires handoff on arrival
 *   DECAY  (phase 2) — alpha fades at uDecayRate per second
 *   DEAD   (phase 3) — slot is recycled via respawn()
 *
 * GLSL → WGSL translation map:
 *   texture2D(tPos, uv)         → textureLoad(tPos, texel, 0)
 *   varying float vAlpha        → @location(1) vAlpha : f32
 *   gl_FragColor                → @location(0) out : vec4f
 *   gl_Position                 → @builtin(position) pos : vec4f
 *   gl_PointSize                → instanced quad half-size (no gl_PointSize in WebGPU)
 *   mix / fract / step / clamp  → same names in WGSL
 *   atan(y, x)                  → atan2(y, x)
 *   mod(x, y)                   → x % y
 *   sin / cos / sqrt            → same
 *
 * GPU buffer layout (see PARTICLE_STRIDE comments below for exact offsets).
 * tPos texture: rgba32float, W×H ≥ maxParticles, one texel per particle:
 *   .r = world X   .g = world Y   .b = travel [0,1]   .a = alpha [0,1]
 *
 * Integration:
 *   const life = new ATSplineParticleLife(device, canvas, edges, config);
 *   await life.build();
 *   // render loop:
 *   const enc = device.createCommandEncoder();
 *   life.update(enc, elapsed, dt);
 *   life.render(enc, colorView, depthView?);
 *   device.queue.submit([enc.finish()]);
 *   await life.scheduleHandoffReadback(); // async, fires onHandoff callbacks
 *
 * References:
 *   src/lib/sph/at-flower-particle.ts       — FlowerParticleShader WebGPU port
 *   src/lib/sph/spline-particle-life.ts     — CPU reference implementation
 *   src/lib/shaders/compiled.vs             — edge-spline.frag / edge-spline.vert
 *   src/lib/sph/curl-flow-field.ts          — GPU curl noise
 *   channels/physics/at_uil_params.json     — UIL parameter source of truth
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** WebGPU workgroup size — ≤256 for broad device compatibility. */
const WG = 64 as const;

/** Maximum particle slots in the GPU pool (tPos texture area). */
const MAX_PARTICLES = 32768 as const;

/** tPos texture dimensions — W × H must be ≥ MAX_PARTICLES. */
const TEX_W = 256 as const;
const TEX_H = 128 as const;  // 256 × 128 = 32768

/**
 * f32 count per particle slot in particleBuf (struct of arrays style).
 * Layout (indices into each particle's sub-array):
 *   [0]  travel      — arc-length fraction [0, 1]
 *   [1]  speed       — per-particle speed scalar
 *   [2]  delay       — remaining spawn delay (seconds)
 *   [3]  phase       — 0=spawn, 1=flow, 2=decay, 3=dead
 *   [4]  alpha       — current opacity [0, 1]
 *   [5]  seed        — random seed (curl noise phase variation)
 *   [6]  noiseOffset — lateral curl offset (domain units)
 *   [7]  edgeIndex   — which EdgeSpline this particle belongs to
 *   [8]  posX        — world X (written by compute, read by vertex)
 *   [9]  posY        — world Y
 *   [10] handoffFlag — 1.0 when particle just entered DECAY (one-shot readback)
 *   [11] species     — particle species tag (from edge.species / weight)
 *   [12..15] _pad    — alignment padding
 */
const PARTICLE_STRIDE = 16 as const;

/** Edge buffer — f32 count per edge slot. */
const EDGE_STRIDE = 260 as const;   // 4 header + 64 points × 4 f32
const EDGE_MAX_PTS = 64 as const;

/** Uniform buffer byte size — must mirror SplineUniforms struct. */
const UNIFORMS_BYTE_SIZE = 80 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 3-D control point on a spline. */
export interface SplinePoint3 {
  x: number;
  y: number;
  z: number;
}

/** One topology edge carrying Catmull-Rom control points and metadata. */
export interface EdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points in SPH domain / world space. */
  points:   SplinePoint3[];
  /**
   * Connectivity weight — controls particle count and size.
   * Mirrors AT's attention-weight / edge-weight concept.
   */
  weight:   number;
  /** Particle species tag (0–7); defaults to 0. */
  species?: number;
}

/**
 * SplineParticleInstance
 *
 * CPU mirror of a single GPU particle slot.  Instances are returned by
 * ATSplineParticleLife.readParticles() for debug/introspection; the GPU
 * buffer is the authoritative source of truth at runtime.
 *
 * Mirrors AT's per-particle GPGPU data struct (SplineParticleLife.fs).
 */
export interface SplineParticleInstance {
  /** Index of this slot in the flat GPU buffer. */
  slotIndex:   number;
  /** Which edge this particle travels. */
  edgeIndex:   number;
  /** Arc-length fraction [0, 1] along the spline. */
  travel:      number;
  /** Individual speed scalar (AT: ∈ [uSplineSpeedMin, uSplineSpeedMax]). */
  speed:       number;
  /** Remaining spawn delay in seconds (0 → already flowing). */
  delay:       number;
  /**
   * Lifecycle phase:
   *   0 = SPAWN  — waiting for delay to expire
   *   1 = FLOW   — moving along spline
   *   2 = DECAY  — fading after reaching end
   *   3 = DEAD   — slot free, will respawn next frame
   */
  phase:       0 | 1 | 2 | 3;
  /** Opacity in [0, 1]. */
  alpha:       number;
  /** Random seed for curl-noise phase variation. */
  seed:        number;
  /** Lateral curl offset in domain units. */
  noiseOffset: number;
  /** Current world X position. */
  posX:        number;
  /** Current world Y position. */
  posY:        number;
  /** 1.0 when particle just entered DECAY (handoff trigger). */
  handoffFlag: number;
  /** Species tag. */
  species:     number;
}

/** UIL-driven configuration — all fields optional, with AT source-of-truth defaults. */
export interface ATSplineParticleConfig {
  /** Per-particle speed range (AT UIL: [0.82, 1.21]). */
  uSplineSpeed?:      [number, number];
  /** Global time scale multiplier (AT: 0.17). */
  uTimeMultiplier?:   number;
  /** Flow range multiplier [min, max] (AT: [1, 1]). */
  uFlowRange?:        [number, number];
  /** Opacity decay rate per second after reaching spline end (AT: ~0.6). */
  uDecayRate?:        number;
  /** Max random spawn delay in seconds (AT: 0). */
  uMaxSDelay?:        number;
  /** Curl-noise spatial frequency (AT UIL: 2–5). */
  uCurlNoiseScale?:   number;
  /** Curl-noise temporal speed (AT UIL: 5). */
  uCurlNoiseSpeed?:   number;
  /** Lateral displacement amplitude in domain units (AT: 0.04). */
  uCurlStrength?:     number;
  /** Point / quad half-size in domain units for rendering (AT: uSize). */
  uSize?:             number;
  /** Particles allocated per unit of edge weight. */
  particlesPerUnit?:  number;
  /** Maximum particle pool size (capped to MAX_PARTICLES). */
  maxParticles?:      number;
  /**
   * Called when a particle enters DECAY (arrives at spline end).
   * Use this to inject an SPH fluid patch at the target cell.
   *
   * @param edgeId    — topology edge identifier
   * @param targetId  — target cell identifier
   * @param x         — world X arrival position (domain units)
   * @param y         — world Y arrival position (domain units)
   * @param vx        — estimated velocity X (domain units / second)
   * @param vy        — estimated velocity Y (domain units / second)
   * @param species   — particle species tag
   */
  onHandoff?: (
    edgeId: string, targetId: string,
    x: number, y: number,
    vx: number, vy: number,
    species: number,
  ) => void;
}

// ─── SplineParticlePreset ─────────────────────────────────────────────────────

/**
 * SplineParticlePreset
 *
 * Named configuration bundles that match AT's UIL parameter presets.
 * Select one as the base config and override individual fields as needed.
 *
 * @example
 * ```ts
 * const life = new ATSplineParticleLife(device, canvas, edges, {
 *   ...SplineParticlePreset.organic,
 *   onHandoff: (edgeId, targetId, x, y, vx, vy, species) => { … },
 * });
 * ```
 */
export const SplineParticlePreset = {
  /**
   * AT source-of-truth defaults — exactly matches SplineParticleLife.fs params.
   * Balanced flow speed, subtle curl noise, moderate particle density.
   */
  default: {
    uSplineSpeed:    [0.82, 1.21]  as [number, number],
    uTimeMultiplier: 0.17,
    uFlowRange:      [1.0,  1.0]   as [number, number],
    uDecayRate:      0.6,
    uMaxSDelay:      0.0,
    uCurlNoiseScale: 2.0,
    uCurlNoiseSpeed: 5.0,
    uCurlStrength:   0.04,
    uSize:           0.012,
    particlesPerUnit: 24,
  } satisfies ATSplineParticleConfig,

  /**
   * Slow drift — gentle, meditative flow along edges.
   * Low speed, strong curl noise, long spawn delay spreads particles evenly.
   */
  slowDrift: {
    uSplineSpeed:    [0.30, 0.55]  as [number, number],
    uTimeMultiplier: 0.08,
    uFlowRange:      [0.8,  1.1]   as [number, number],
    uDecayRate:      0.3,
    uMaxSDelay:      1.5,
    uCurlNoiseScale: 3.5,
    uCurlNoiseSpeed: 2.0,
    uCurlStrength:   0.10,
    uSize:           0.018,
    particlesPerUnit: 16,
  } satisfies ATSplineParticleConfig,

  /**
   * Fast pulse — rapid data-transfer aesthetic, tight bursts.
   * High speed, minimal noise, short decay — feels like signal packets.
   */
  fastPulse: {
    uSplineSpeed:    [1.80, 2.60]  as [number, number],
    uTimeMultiplier: 0.35,
    uFlowRange:      [1.0,  1.2]   as [number, number],
    uDecayRate:      1.2,
    uMaxSDelay:      0.2,
    uCurlNoiseScale: 1.0,
    uCurlNoiseSpeed: 8.0,
    uCurlStrength:   0.015,
    uSize:           0.008,
    particlesPerUnit: 32,
  } satisfies ATSplineParticleConfig,

  /**
   * Organic turbulence — high curl noise, slow flow, large displacement.
   * Gives a living, biological feel — good for cell / organism themes.
   */
  organic: {
    uSplineSpeed:    [0.50, 0.90]  as [number, number],
    uTimeMultiplier: 0.12,
    uFlowRange:      [0.9,  1.3]   as [number, number],
    uDecayRate:      0.45,
    uMaxSDelay:      0.8,
    uCurlNoiseScale: 5.0,
    uCurlNoiseSpeed: 3.0,
    uCurlStrength:   0.18,
    uSize:           0.016,
    particlesPerUnit: 20,
  } satisfies ATSplineParticleConfig,

  /**
   * Dense swarm — many small, fast particles blanket every edge.
   * Good for high-connectivity attention layers or activation maps.
   */
  denseSwarm: {
    uSplineSpeed:    [1.00, 1.60]  as [number, number],
    uTimeMultiplier: 0.22,
    uFlowRange:      [1.0,  1.0]   as [number, number],
    uDecayRate:      0.8,
    uMaxSDelay:      0.0,
    uCurlNoiseScale: 2.5,
    uCurlNoiseSpeed: 6.0,
    uCurlStrength:   0.06,
    uSize:           0.006,
    particlesPerUnit: 64,
  } satisfies ATSplineParticleConfig,

  /**
   * Ethereal fade — particles barely visible, long decay, soft curl.
   * Ambient background flow; does not compete with primary visualisation.
   */
  etherealFade: {
    uSplineSpeed:    [0.60, 1.00]  as [number, number],
    uTimeMultiplier: 0.10,
    uFlowRange:      [0.8,  1.0]   as [number, number],
    uDecayRate:      0.15,
    uMaxSDelay:      2.0,
    uCurlNoiseScale: 4.0,
    uCurlNoiseSpeed: 1.5,
    uCurlStrength:   0.08,
    uSize:           0.020,
    particlesPerUnit: 12,
  } satisfies ATSplineParticleConfig,
} as const;

// ─── WGSL — Shared noise helpers (AT simplenoise.glsl port) ───────────────────

const NOISE_WGSL = /* wgsl */`
// ── 3-D hash (AT: hash33 from simplenoise.glsl) ──────────────────────────────
fn hash3(p: vec3f) -> vec3f {
  let q = vec3f(
    dot(p, vec3f(127.1, 311.7,  74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453123);
}

// ── 3-D gradient noise (AT: noise3 / Perlin from simplenoise.glsl) ───────────
fn noise3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = fract(x);
  // Quintic fade (C2-continuous)
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  let n000 = dot(hash3(i + vec3f(0,0,0)) * 2.0 - 1.0, f - vec3f(0,0,0));
  let n100 = dot(hash3(i + vec3f(1,0,0)) * 2.0 - 1.0, f - vec3f(1,0,0));
  let n010 = dot(hash3(i + vec3f(0,1,0)) * 2.0 - 1.0, f - vec3f(0,1,0));
  let n110 = dot(hash3(i + vec3f(1,1,0)) * 2.0 - 1.0, f - vec3f(1,1,0));
  let n001 = dot(hash3(i + vec3f(0,0,1)) * 2.0 - 1.0, f - vec3f(0,0,1));
  let n101 = dot(hash3(i + vec3f(1,0,1)) * 2.0 - 1.0, f - vec3f(1,0,1));
  let n011 = dot(hash3(i + vec3f(0,1,1)) * 2.0 - 1.0, f - vec3f(0,1,1));
  let n111 = dot(hash3(i + vec3f(1,1,1)) * 2.0 - 1.0, f - vec3f(1,1,1));

  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z,
  );
}

// ── 2-D curl noise (AT: ∇×Ψ from CurlNoise.frag) ────────────────────────────
// Returns divergence-free 2-D velocity field F = (∂Ψ/∂y, −∂Ψ/∂x).
// The z-component of the 3-D noise is used as temporal dimension.
fn curlNoise2D(p: vec3f, eps: f32) -> vec2f {
  let dx  = vec3f(eps, 0.0, 0.0);
  let dy  = vec3f(0.0, eps, 0.0);
  let dz  = vec3f(0.0, 0.0, eps);
  // Mixed partial ∂²Ψ/∂x∂z and ∂²Ψ/∂y∂z give the 2-D curl
  let Fx = (noise3(p + dy + dz) - noise3(p - dy + dz)
           - noise3(p + dy)     + noise3(p - dy))     / (4.0 * eps * eps);
  let Fy = (noise3(p + dx + dz) - noise3(p - dx + dz)
           - noise3(p + dx)     + noise3(p - dx))     / (4.0 * eps * eps);
  return vec2f(-Fx, Fy);
}
`;

// ─── WGSL — Uniforms struct ───────────────────────────────────────────────────

const UNIFORMS_WGSL = /* wgsl */`
struct SplineUniforms {
  // AT SplineParticleLife.fs params
  uTimeMultiplier  : f32,   // global time scale (AT: 0.17)
  uDecayRate       : f32,   // alpha decay rate per second
  uCurlNoiseScale  : f32,   // curl noise spatial frequency
  uCurlNoiseSpeed  : f32,   // curl noise temporal speed
  uCurlStrength    : f32,   // lateral displacement amplitude
  uSplineSpeedMin  : f32,   // per-particle speed range min
  uSplineSpeedMax  : f32,   // per-particle speed range max
  uMaxSDelay       : f32,   // max random spawn delay (seconds)

  // Rendering
  uSize            : f32,   // quad half-size (domain units)
  time             : f32,   // elapsed seconds

  // Domain / NDC
  domainW          : f32,
  domainH          : f32,
  scaleX           : f32,   // 2 / domainW
  scaleY           : f32,   // 2 / domainH

  particleCount    : u32,
  edgeCount        : u32,
  texW             : u32,
  texH             : u32,
  _pad0            : u32,
}
`;

// Byte offsets (mirrors struct above, 4-byte aligned)
const U_TIME_MULTIPLIER  =  0;
const U_DECAY_RATE       =  4;
const U_CURL_SCALE       =  8;
const U_CURL_SPEED       = 12;
const U_CURL_STRENGTH    = 16;
const U_SPEED_MIN        = 20;
const U_SPEED_MAX        = 24;
const U_MAX_S_DELAY      = 28;
const U_SIZE             = 32;
const U_TIME             = 36;
const U_DOMAIN_W         = 40;
const U_DOMAIN_H         = 44;
const U_SCALE_X          = 48;
const U_SCALE_Y          = 52;
const U_PARTICLE_COUNT   = 56;   // u32
const U_EDGE_COUNT       = 60;   // u32
const U_TEX_W            = 64;   // u32
const U_TEX_H            = 68;   // u32

// ─── WGSL — Spline evaluation ─────────────────────────────────────────────────

const SPLINE_WGSL = /* wgsl */`
// Edge buffer layout (array<f32>, EDGE_STRIDE = 260 f32 per edge):
//   [0]        nPoints (f32 cast to u32)
//   [1..3]     reserved / padding
//   [4 + p*4]  point p: x, y, z, 0
const EDGE_HDR    = 4u;
const EDGE_PSTRIDE = 4u;
const EDGE_STRIDE  = 260u;

fn edgeNPts(buf: ptr<storage, array<f32>, read>, ei: u32) -> u32 {
  return u32((*buf)[ei * EDGE_STRIDE]);
}
fn edgePt(buf: ptr<storage, array<f32>, read>, ei: u32, pi: u32) -> vec3f {
  let b = ei * EDGE_STRIDE + EDGE_HDR + pi * EDGE_PSTRIDE;
  return vec3f((*buf)[b], (*buf)[b + 1u], (*buf)[b + 2u]);
}

// ── Catmull-Rom segment (AT: SplineParticleLife.fs, tension = 0.5) ───────────
fn catmullRom(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  let f1 = -0.5 * t3 + t2        - 0.5 * t;
  let f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  let f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  let f4 =  0.5 * t3 - 0.5 * t2;
  return f1*p0 + f2*p1 + f3*p2 + f4*p3;
}

fn clampIdx(i: i32, n: i32) -> i32 { return clamp(i, 0, n - 1); }

// Evaluate spline at normalised arc-length fraction u ∈ [0, 1]
fn evalSpline(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec3f {
  let n = i32(edgeNPts(buf, ei));
  if (n == 0) { return vec3f(0.0); }
  if (n == 1) { return edgePt(buf, ei, 0u); }
  let sc  = clamp(u, 0.0, 0.9999) * f32(n - 1);
  let i1  = i32(floor(sc));
  let lt  = sc - f32(i1);
  return catmullRom(
    edgePt(buf, ei, u32(clampIdx(i1 - 1, n))),
    edgePt(buf, ei, u32(clampIdx(i1,     n))),
    edgePt(buf, ei, u32(clampIdx(i1 + 1, n))),
    edgePt(buf, ei, u32(clampIdx(i1 + 2, n))),
    lt,
  );
}

// Finite-difference tangent (normalised)
fn splineTangent(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec3f {
  let eps = 0.001;
  let a   = evalSpline(buf, ei, max(0.0, u - eps));
  let b   = evalSpline(buf, ei, min(1.0, u + eps));
  let d   = b - a;
  let l   = length(d);
  if (l < 1e-8) { return vec3f(1.0, 0.0, 0.0); }
  return d / l;
}
`;

// ─── WGSL — Compute shader ────────────────────────────────────────────────────
// Direct port of AT SplineParticleLife.fs lifecycle FSM.
// Each invocation handles one particle slot.

const COMPUTE_SHADER = /* wgsl */`
${UNIFORMS_WGSL}
${NOISE_WGSL}
${SPLINE_WGSL}

@group(0) @binding(0) var<uniform>             uni        : SplineUniforms;
@group(1) @binding(0) var<storage, read>       edgeBuf    : array<f32>;
@group(1) @binding(1) var<storage, read_write> particleBuf: array<f32>;
@group(1) @binding(2) var                      tPos       : texture_storage_2d<rgba32float, write>;

// ── Particle slot field indices (mirrors PARTICLE_STRIDE layout) ──────────────
const PS = 16u;   // PARTICLE_STRIDE

const F_TRAVEL   = 0u;
const F_SPEED    = 1u;
const F_DELAY    = 2u;
const F_PHASE    = 3u;   // 0=spawn, 1=flow, 2=decay, 3=dead
const F_ALPHA    = 4u;
const F_SEED     = 5u;
const F_NOFF     = 6u;
const F_EDGE     = 7u;
const F_POS_X    = 8u;
const F_POS_Y    = 9u;
const F_HANDOFF  = 10u;
const F_SPECIES  = 11u;

fn pGet(idx: u32, f: u32) -> f32 { return particleBuf[idx * PS + f]; }
fn pSet(idx: u32, f: u32, v: f32) { particleBuf[idx * PS + f] = v; }

// ── Pseudo-random scalar from two seeds ──────────────────────────────────────
fn rng(s: f32, salt: f32) -> f32 {
  return fract(sin(s * 127.1 + salt * 311.7) * 43758.5453);
}

// ── Respawn a dead slot — mirrors AT SplineParticleLife.fs respawn() ─────────
fn respawn(idx: u32, t: f32) {
  let s      = f32(idx) * 1.618034 + t;
  let speed  = mix(uni.uSplineSpeedMin, uni.uSplineSpeedMax, rng(s, 0.0));
  let delay  = rng(s, 1.0) * uni.uMaxSDelay;
  let phase  = select(1.0, 0.0, delay > 0.001);
  let alpha  = select(1.0, 0.0, delay > 0.001);

  pSet(idx, F_TRAVEL,  0.0);
  pSet(idx, F_SPEED,   speed);
  pSet(idx, F_DELAY,   delay);
  pSet(idx, F_PHASE,   phase);
  pSet(idx, F_ALPHA,   alpha);
  pSet(idx, F_SEED,    rng(s, 3.0) * 1000.0);
  pSet(idx, F_NOFF,    0.0);
  pSet(idx, F_HANDOFF, 0.0);
}

// ── Main compute (one invocation per particle slot) ───────────────────────────
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  // Fixed physics step — real elapsed time is already baked into the uniforms.
  let dt = 1.0 / 60.0;
  let t  = uni.time;

  var phase  = pGet(idx, F_PHASE);
  var travel = pGet(idx, F_TRAVEL);
  var speed  = pGet(idx, F_SPEED);
  var delay  = pGet(idx, F_DELAY);
  var alpha  = pGet(idx, F_ALPHA);
  var seed   = pGet(idx, F_SEED);
  var nOff   = pGet(idx, F_NOFF);
  let eIdx   = u32(pGet(idx, F_EDGE));

  // ── FSM — AT SplineParticleLife.fs lifecycle ──────────────────────────────

  if (phase == 3.0) {
    // DEAD → respawn and re-read state
    respawn(idx, t);
    phase  = pGet(idx, F_PHASE);
    travel = 0.0;
    speed  = pGet(idx, F_SPEED);
    delay  = pGet(idx, F_DELAY);
    alpha  = pGet(idx, F_ALPHA);
    seed   = pGet(idx, F_SEED);
    nOff   = 0.0;
    pSet(idx, F_HANDOFF, 0.0);
  }

  if (phase == 0.0) {
    // SPAWN — count down delay
    delay -= dt;
    if (delay <= 0.0) {
      phase = 1.0;   // → FLOW
      alpha = 1.0;
      delay = 0.0;
    }
    pSet(idx, F_DELAY, delay);
    pSet(idx, F_PHASE, phase);
    pSet(idx, F_ALPHA, alpha);

  } else if (phase == 1.0) {
    // FLOW — advance travel along spline
    // AT formula: vScale = speed * uTimeMultiplier * 0.01
    let vScale  = speed * uni.uTimeMultiplier * 0.01;
    travel     += vScale * dt * uni.uTimeMultiplier * 60.0;

    // Current spline world position
    let sp = evalSpline(&edgeBuf, eIdx, min(travel, 0.9999));

    // Curl-noise lateral perturbation (AT simplenoise.glsl ∇×Ψ)
    let noiseCoord = vec3f(
      sp.x * uni.uCurlNoiseScale,
      sp.y * uni.uCurlNoiseScale,
      t    * uni.uCurlNoiseSpeed * 0.1 + seed * 0.001,
    );
    let curl = curlNoise2D(noiseCoord, 0.01) * uni.uCurlStrength;

    // Perpendicular to spline tangent
    let tan  = splineTangent(&edgeBuf, eIdx, min(travel, 0.9999));
    let perpX = -tan.y;
    let perpY =  tan.x;

    let worldX = sp.x + perpX * curl.x + perpX * nOff;
    let worldY = sp.y + perpY * curl.y + perpY * nOff;
    nOff = curl.x * perpX + curl.y * perpY;

    pSet(idx, F_TRAVEL, travel);
    pSet(idx, F_POS_X,  worldX);
    pSet(idx, F_POS_Y,  worldY);
    pSet(idx, F_NOFF,   nOff);

    if (travel >= 1.0) {
      phase = 2.0;   // → DECAY
      pSet(idx, F_PHASE,   phase);
      pSet(idx, F_HANDOFF, 1.0);   // one-shot handoff flag for CPU readback
    }

  } else if (phase == 2.0) {
    // DECAY — fade alpha (AT: alpha decays at uDecayRate per second)
    alpha -= uni.uDecayRate * dt;
    if (alpha <= 0.0) {
      alpha = 0.0;
      phase = 3.0;   // → DEAD
    }
    pSet(idx, F_ALPHA, alpha);
    pSet(idx, F_PHASE, phase);
  }

  // ── Write to tPos texture (AT GPGPU ping-pong) ────────────────────────────
  // .r = worldX  .g = worldY  .b = travel  .a = alpha
  let posX   = pGet(idx, F_POS_X);
  let posY   = pGet(idx, F_POS_Y);
  let texX   = i32(idx % uni.texW);
  let texY   = i32(idx / uni.texW);
  textureStore(tPos, vec2<i32>(texX, texY), vec4f(posX, posY, travel, alpha));
}
`;

// ─── WGSL — Vertex shader ─────────────────────────────────────────────────────
// Reads tPos texture (AT: texture2D(tPos, reference)) and emits an instanced
// quad per particle.  AT size formula: uSize * (1 − travel²).

const VERTEX_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni      : SplineUniforms;
@group(0) @binding(1) var          tPos     : texture_2d<f32>;
@group(0) @binding(2) var          sSampler : sampler;

struct VertOut {
  @builtin(position) pos     : vec4f,
  @location(0)       vUv     : vec2f,    // quad local [-1,1], used for circle SDF
  @location(1)       vAlpha  : f32,
  @location(2)       vTravel : f32,
  @location(3)       vPhase  : f32,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  // Sample tPos at the particle's texel (AT: texture2D(tPos, reference))
  let texX = i32(ii % uni.texW);
  let texY = i32(ii / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX  = p.r;
  let worldY  = p.g;
  let travel  = p.b;
  let alpha   = p.a;

  // Invisible particles collapsed to clip-space degenerate (no discard cost)
  let alive = select(0.0, 1.0, alpha > 0.004);

  // AT vScale: uSize * (1 − travel²) — size attenuates toward spline end
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  let halfSize    = uni.uSize * travelDecay * 0.5;

  let quadUV = QUAD[vi];
  let ndcX   = worldX * uni.scaleX - 1.0 + quadUV.x * halfSize * uni.scaleX;
  let ndcY   = worldY * uni.scaleY - 1.0 + quadUV.y * halfSize * uni.scaleY;

  var out: VertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = quadUV;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  out.vPhase  = 1.0;   // always flow-or-decay for visible particles
  return out;
}
`;

// ─── WGSL — Fragment shader ───────────────────────────────────────────────────
// AT FlowerParticleShader.glsl fragment stage:
//   — circular soft-particle SDF (replaces gl_PointCoord circle)
//   — alpha fade: sin(π · travel) bright at midpoint, dark at ends

const FRAGMENT_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni : SplineUniforms;

struct FragIn {
  @location(0) vUv    : vec2f,
  @location(1) vAlpha : f32,
  @location(2) vTravel: f32,
  @location(3) vPhase : f32,
}

@fragment fn fs_main(in: FragIn) -> @location(0) vec4f {
  // Soft circular disk (AT: if (r > 1.0) discard)
  let r2  = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Soft edge falloff (smooth the billboard boundary)
  let edge    = 1.0 - smoothstep(0.7, 1.0, r2);

  // AT alpha: sin(π·travel) — bright at midpoint, zero at start/end
  let fade    = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Core colour: white with slight warm tint (matches AT matcap fallback)
  let col     = mix(vec3f(1.0, 0.88, 0.72), vec3f(1.0), 1.0 - r2);
  let finalA  = in.vAlpha * fade * edge;

  return vec4f(col * finalA, finalA);
}
`;

// ─── CPU-side spline helpers ──────────────────────────────────────────────────

function catmullRomCPU(
  p0: SplinePoint3, p1: SplinePoint3,
  p2: SplinePoint3, p3: SplinePoint3,
  t: number,
): SplinePoint3 {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5 * t3 + t2        - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
    z: f1 * p0.z + f2 * p1.z + f3 * p2.z + f4 * p3.z,
  };
}

function evalSplineCPU(points: SplinePoint3[], u: number): SplinePoint3 {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...points[0] };
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  return catmullRomCPU(
    points[clamp(i1 - 1)], points[clamp(i1)],
    points[clamp(i1 + 1)], points[clamp(i1 + 2)],
    sc - i1,
  );
}

function buildEdgeBuf(edges: EdgeSpline[]): Float32Array {
  const buf = new Float32Array(edges.length * EDGE_STRIDE);
  for (let e = 0; e < edges.length; e++) {
    const base = e * EDGE_STRIDE;
    const pts  = edges[e].points;
    buf[base] = Math.min(pts.length, EDGE_MAX_PTS);
    for (let p = 0; p < Math.min(pts.length, EDGE_MAX_PTS); p++) {
      const pb = base + 4 + p * 4;
      buf[pb + 0] = pts[p].x;
      buf[pb + 1] = pts[p].y;
      buf[pb + 2] = pts[p].z;
    }
  }
  return buf;
}

// ─── ATSplineParticleLife — Main WebGPU class ─────────────────────────────────

/**
 * ATSplineParticleLife
 *
 * WebGPU orchestrator for AT's SplineParticleLife particle lifecycle system.
 *
 * Encapsulates the full GPU pipeline:
 *   1. Compute pass  — advances SplineParticleInstance states in parallel
 *                      (SPAWN → FLOW → DECAY → DEAD → respawn)
 *                      writes world positions + alpha to tPos texture
 *   2. Render pass   — instanced quads read tPos, AT size attenuation,
 *                      circular soft-particle SDF, sin(π·travel) alpha fade
 *   3. Handoff       — async readback of F_HANDOFF flags to fire onHandoff
 *                      callbacks for SPH fluid injection
 *
 * @example
 * ```ts
 * import { ATSplineParticleLife, SplineParticlePreset } from '$lib/sph/at-spline-particle';
 *
 * const life = new ATSplineParticleLife(device, canvas, edges, {
 *   ...SplineParticlePreset.organic,
 *   onHandoff: (edgeId, targetId, x, y, vx, vy, species) => {
 *     sphWorld.addFluid(x - 0.05, y - 0.05, x + 0.05, y + 0.05, 0.04, species);
 *   },
 * });
 * await life.build();
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * life.update(enc, elapsed, dt);
 * life.render(enc, colorView);
 * device.queue.submit([enc.finish()]);
 * life.scheduleHandoffReadback();  // fire-and-forget
 * ```
 */
export class ATSplineParticleLife {
  private readonly device:     GPUDevice;
  private readonly canvas:     HTMLCanvasElement;
  private readonly onHandoff?: ATSplineParticleConfig['onHandoff'];

  private cfg: Required<Omit<ATSplineParticleConfig, 'onHandoff'>>;
  private edges: EdgeSpline[] = [];
  private particleCount = 0;
  private elapsed = 0;

  // GPU resources
  private uniformBuf!:      GPUBuffer;
  private edgeBuf!:         GPUBuffer;
  private particleBuf!:     GPUBuffer;
  private readbackBuf!:     GPUBuffer;
  private tPos!:            GPUTexture;
  private tPosView!:        GPUTextureView;
  private sampler!:         GPUSampler;

  // Pipelines
  private computePipeline!: GPUComputePipeline;
  private renderPipeline!:  GPURenderPipeline;

  // Bind groups
  private computeBG0!:      GPUBindGroup;
  private computeBG1!:      GPUBindGroup;
  private renderBG!:        GPUBindGroup;

  private built = false;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    edges:  EdgeSpline[],
    config: ATSplineParticleConfig = {},
  ) {
    this.device     = device;
    this.canvas     = canvas;
    this.edges      = edges;
    this.onHandoff  = config.onHandoff;
    this.cfg = {
      uSplineSpeed:    config.uSplineSpeed    ?? [0.82, 1.21],
      uTimeMultiplier: config.uTimeMultiplier ?? 0.17,
      uFlowRange:      config.uFlowRange      ?? [1.0, 1.0],
      uDecayRate:      config.uDecayRate      ?? 0.6,
      uMaxSDelay:      config.uMaxSDelay      ?? 0.0,
      uCurlNoiseScale: config.uCurlNoiseScale ?? 2.0,
      uCurlNoiseSpeed: config.uCurlNoiseSpeed ?? 5.0,
      uCurlStrength:   config.uCurlStrength   ?? 0.04,
      uSize:           config.uSize           ?? 0.012,
      particlesPerUnit: config.particlesPerUnit ?? 24,
      maxParticles:    config.maxParticles    ?? MAX_PARTICLES,
    };
  }

  // ── Build GPU resources ──────────────────────────────────────────────────

  async build(): Promise<void> {
    if (this.built) this._destroy();
    const { device } = this;

    // Particle count from edges, capped to pool limit
    this.particleCount = Math.min(
      this.cfg.maxParticles,
      Math.max(
        256,
        this.edges.reduce((n, e) => n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0),
      ),
    );

    // ── Uniform buffer ───────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      size:  UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeUniforms(0);

    // ── Edge buffer ──────────────────────────────────────────────────────────
    const edgeData = buildEdgeBuf(this.edges);
    this.edgeBuf = device.createBuffer({
      size:  Math.max(edgeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.edgeBuf, 0, edgeData);

    // ── Particle state buffer ────────────────────────────────────────────────
    const pData = this._initParticleBuf();
    this.particleBuf = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.particleBuf, 0, pData);

    // Readback buffer (for handoff flag CPU scan)
    this.readbackBuf = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── tPos texture (AT: GPGPU ping-pong / position texture) ────────────────
    this.tPos = device.createTexture({
      size:   [TEX_W, TEX_H],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING  |
              GPUTextureUsage.COPY_SRC,
    });
    this.tPosView = this.tPos.createView();

    // ── Sampler ──────────────────────────────────────────────────────────────
    this.sampler = device.createSampler({
      magFilter: 'nearest', minFilter: 'nearest',
    });

    // ── Compute pipeline ─────────────────────────────────────────────────────
    const computeMod = device.createShaderModule({ code: COMPUTE_SHADER });
    this.computePipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: computeMod, entryPoint: 'main' },
    });

    // ── Render pipeline ───────────────────────────────────────────────────────
    const vsMod  = device.createShaderModule({ code: VERTEX_SHADER });
    const fsMod  = device.createShaderModule({ code: FRAGMENT_SHADER });
    const fmt    = navigator.gpu.getPreferredCanvasFormat();

    this.renderPipeline = device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module: vsMod,  entryPoint: 'vs_main' },
      fragment: {
        module: fsMod, entryPoint: 'fs_main',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Bind groups ───────────────────────────────────────────────────────────
    this._buildBindGroups();
    this.built = true;

    console.log(
      `[ATSplineParticleLife] built: ${this.edges.length} edges, ` +
      `${this.particleCount} particles, ` +
      `uSplineSpeed=[${this.cfg.uSplineSpeed}]`,
    );
  }

  // ── Per-frame update (encode compute pass) ────────────────────────────────

  /**
   * Encode the compute pass that advances all SplineParticleInstance states.
   * Must be called before render() in the same command encoder.
   *
   * @param encoder  — open GPUCommandEncoder
   * @param elapsed  — total elapsed seconds
   * @param _dt      — frame delta (currently fixed-step in shader)
   */
  update(encoder: GPUCommandEncoder, elapsed: number, _dt = 0): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this._writeUniforms(elapsed);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBG0);
    pass.setBindGroup(1, this.computeBG1);
    pass.dispatchWorkgroups(Math.ceil(this.particleCount / WG));
    pass.end();
  }

  // ── Per-frame render pass ─────────────────────────────────────────────────

  /**
   * Encode the render pass that draws spline particles.
   *
   * @param encoder    — open GPUCommandEncoder
   * @param colorView  — render target texture view
   * @param depthView  — optional depth attachment
   */
  render(
    encoder:    GPUCommandEncoder,
    colorView:  GPUTextureView,
    depthView?: GPUTextureView,
  ): void {
    if (!this.built) return;

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [{
        view:    colorView,
        loadOp:  'load',
        storeOp: 'store',
      }],
    };
    if (depthView) {
      passDesc.depthStencilAttachment = {
        view:            depthView,
        depthLoadOp:     'load',
        depthStoreOp:    'store',
      };
    }

    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBG);
    // 6 vertices per instanced quad, one instance per particle slot
    pass.draw(6, this.particleCount);
    pass.end();
  }

  // ── SPH handoff readback ──────────────────────────────────────────────────

  /**
   * Async readback of particleBuf to scan F_HANDOFF flags.
   * Fires onHandoff for each particle that just entered DECAY.
   * Safe to call fire-and-forget each frame.
   */
  async scheduleHandoffReadback(): Promise<void> {
    if (!this.built || !this.onHandoff) return;
    const { device } = this;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, this.readbackBuf, 0, this.particleBuf.size);
    device.queue.submit([enc.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange());

    for (let i = 0; i < this.particleCount; i++) {
      const b = i * PARTICLE_STRIDE;
      if (data[b + 10] < 0.5) continue;   // F_HANDOFF

      const eIdx    = Math.round(data[b + 7]);
      const edge    = this.edges[eIdx];
      if (!edge) continue;

      const x = data[b + 8];
      const y = data[b + 9];
      const spd = data[b + 1];

      // Velocity from spline end tangent (AT: vScale = speed * uTimeMultiplier * 0.01)
      const ep  = evalSplineCPU(edge.points, 0.999);
      const ep2 = evalSplineCPU(edge.points, 0.998);
      const tx  = ep.x - ep2.x;
      const ty  = ep.y - ep2.y;
      const tl  = Math.sqrt(tx * tx + ty * ty) + 1e-10;
      const vScale = spd * this.cfg.uTimeMultiplier * 0.01;

      this.onHandoff(
        edge.edgeId, edge.targetId,
        x, y,
        (tx / tl) * vScale,
        (ty / tl) * vScale,
        edge.species ?? 0,
      );
    }

    this.readbackBuf.unmap();
  }

  // ── Introspection: read SplineParticleInstance array from GPU ────────────

  /**
   * Read all particle slots from GPU into CPU SplineParticleInstance objects.
   * Expensive — for debug / diagnostics only.  Do not call every frame.
   */
  async readParticles(): Promise<SplineParticleInstance[]> {
    if (!this.built) return [];
    const { device } = this;

    const readBuf = device.createBuffer({
      size:  this.particleBuf.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, readBuf, 0, this.particleBuf.size);
    device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const raw  = new Float32Array(readBuf.getMappedRange());
    const out: SplineParticleInstance[] = [];

    for (let i = 0; i < this.particleCount; i++) {
      const b = i * PARTICLE_STRIDE;
      out.push({
        slotIndex:   i,
        edgeIndex:   Math.round(raw[b + 7]),
        travel:      raw[b + 0],
        speed:       raw[b + 1],
        delay:       raw[b + 2],
        phase:       Math.round(raw[b + 3]) as 0 | 1 | 2 | 3,
        alpha:       raw[b + 4],
        seed:        raw[b + 5],
        noiseOffset: raw[b + 6],
        posX:        raw[b + 8],
        posY:        raw[b + 9],
        handoffFlag: raw[b + 10],
        species:     raw[b + 11],
      });
    }

    readBuf.unmap();
    readBuf.destroy();
    return out;
  }

  // ── Live parameter updates ────────────────────────────────────────────────

  setSplineSpeed(min: number, max: number): void   { this.cfg.uSplineSpeed = [min, max]; }
  setTimeMultiplier(v: number): void               { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void                    { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void                 { this.cfg.uCurlStrength = v; }
  setCurlNoiseScale(v: number): void               { this.cfg.uCurlNoiseScale = v; }
  setCurlNoiseSpeed(v: number): void               { this.cfg.uCurlNoiseSpeed = v; }
  setFlowRange(min: number, max: number): void     { this.cfg.uFlowRange = [min, max]; }
  setSize(v: number): void                         { this.cfg.uSize = v; }

  /**
   * Apply a SplineParticlePreset all at once.
   * Does not require rebuild — takes effect next _writeUniforms call.
   */
  applyPreset(preset: ATSplineParticleConfig): void {
    if (preset.uSplineSpeed)    this.cfg.uSplineSpeed    = preset.uSplineSpeed;
    if (preset.uTimeMultiplier !== undefined) this.cfg.uTimeMultiplier = preset.uTimeMultiplier;
    if (preset.uFlowRange)      this.cfg.uFlowRange      = preset.uFlowRange;
    if (preset.uDecayRate   !== undefined) this.cfg.uDecayRate      = preset.uDecayRate;
    if (preset.uMaxSDelay   !== undefined) this.cfg.uMaxSDelay      = preset.uMaxSDelay;
    if (preset.uCurlNoiseScale !== undefined) this.cfg.uCurlNoiseScale = preset.uCurlNoiseScale;
    if (preset.uCurlNoiseSpeed !== undefined) this.cfg.uCurlNoiseSpeed = preset.uCurlNoiseSpeed;
    if (preset.uCurlStrength !== undefined)   this.cfg.uCurlStrength   = preset.uCurlStrength;
    if (preset.uSize        !== undefined) this.cfg.uSize           = preset.uSize;
  }

  /**
   * Replace edge splines at runtime (e.g. after topology update).
   * Triggers a full rebuild — previous particle state is discarded.
   */
  async setEdges(edges: EdgeSpline[]): Promise<void> {
    this.edges = edges;
    await this.build();
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  get particleSlots(): number { return this.particleCount; }
  get edgeCount(): number      { return this.edges.length; }
  get isBuilt(): boolean       { return this.built; }
  get elapsedTime(): number    { return this.elapsed; }

  /** Current config snapshot (read-only). */
  get config(): Readonly<Required<Omit<ATSplineParticleConfig, 'onHandoff'>>> {
    return this.cfg;
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy(): void { this._destroy(); }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _initParticleBuf(): Float32Array {
    const buf  = new Float32Array(this.particleCount * PARTICLE_STRIDE);
    let   slot = 0;

    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        this.particleCount - slot,
      );
      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const b     = slot * PARTICLE_STRIDE;
        const speed = this.cfg.uSplineSpeed[0] +
                      Math.random() * (this.cfg.uSplineSpeed[1] - this.cfg.uSplineSpeed[0]);
        const delay = Math.random() * this.cfg.uMaxSDelay;
        const start = evalSplineCPU(edge.points, 0);

        buf[b +  0] = 0;                              // travel
        buf[b +  1] = speed;                          // speed
        buf[b +  2] = delay;                          // delay
        buf[b +  3] = delay > 0 ? 0 : 1;             // phase: SPAWN or FLOW
        buf[b +  4] = delay > 0 ? 0 : 1;             // alpha
        buf[b +  5] = Math.random() * 1000;           // seed
        buf[b +  6] = 0;                              // noiseOffset
        buf[b +  7] = e;                              // edgeIndex
        buf[b +  8] = start.x;                       // posX
        buf[b +  9] = start.y;                       // posY
        buf[b + 10] = 0;                              // handoffFlag
        buf[b + 11] = edge.species ?? 0;              // species
      }
    }

    // Remaining slots start as DEAD (phase = 3)
    for (; slot < this.particleCount; slot++) {
      buf[slot * PARTICLE_STRIDE + 3] = 3;
    }

    return buf;
  }

  private _writeUniforms(elapsed: number): void {
    const { device, cfg, canvas } = this;
    const dw = canvas.width  || 1;
    const dh = canvas.height || 1;
    const data = new Float32Array(UNIFORMS_BYTE_SIZE / 4);

    data[U_TIME_MULTIPLIER  / 4] = cfg.uTimeMultiplier;
    data[U_DECAY_RATE       / 4] = cfg.uDecayRate;
    data[U_CURL_SCALE       / 4] = cfg.uCurlNoiseScale;
    data[U_CURL_SPEED       / 4] = cfg.uCurlNoiseSpeed;
    data[U_CURL_STRENGTH    / 4] = cfg.uCurlStrength;
    data[U_SPEED_MIN        / 4] = cfg.uSplineSpeed[0];
    data[U_SPEED_MAX        / 4] = cfg.uSplineSpeed[1];
    data[U_MAX_S_DELAY      / 4] = cfg.uMaxSDelay;
    data[U_SIZE             / 4] = cfg.uSize;
    data[U_TIME             / 4] = elapsed;
    data[U_DOMAIN_W         / 4] = dw;
    data[U_DOMAIN_H         / 4] = dh;
    data[U_SCALE_X          / 4] = 2.0 / dw;
    data[U_SCALE_Y          / 4] = 2.0 / dh;

    const u32 = new Uint32Array(data.buffer);
    u32[U_PARTICLE_COUNT / 4] = this.particleCount;
    u32[U_EDGE_COUNT     / 4] = this.edges.length;
    u32[U_TEX_W          / 4] = TEX_W;
    u32[U_TEX_H          / 4] = TEX_H;

    device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private _buildBindGroups(): void {
    const { device } = this;

    // Compute BG0 — uniforms only
    this.computeBG0 = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });

    // Compute BG1 — edges + particles + tPos write
    this.computeBG1 = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.edgeBuf } },
        { binding: 1, resource: { buffer: this.particleBuf } },
        { binding: 2, resource: this.tPosView },
      ],
    });

    // Render BG — uniforms + tPos read + sampler
    this.renderBG = device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.tPosView },
        { binding: 2, resource: this.sampler },
      ],
    });
  }

  private _destroy(): void {
    if (!this.built) return;
    this.uniformBuf?.destroy();
    this.edgeBuf?.destroy();
    this.particleBuf?.destroy();
    this.readbackBuf?.destroy();
    this.tPos?.destroy();
    this.built = false;
  }
}

// ─── Factory helpers ───────────────────────────────────────────────────────────

/**
 * Wire an ATSplineParticleLife to SPHWorld.addFluid() for automatic
 * fluid injection when particles arrive at their target cells.
 *
 * @example
 * ```ts
 * const life = createATSplineParticleForSPH(device, canvas, edges, world.addFluid.bind(world));
 * await life.build();
 * // render loop:
 * const enc = device.createCommandEncoder();
 * life.update(enc, elapsed, dt);
 * life.render(enc, colorView);
 * device.queue.submit([enc.finish()]);
 * life.scheduleHandoffReadback();
 * ```
 */
export function createATSplineParticleForSPH(
  device:   GPUDevice,
  canvas:   HTMLCanvasElement,
  edges:    EdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATSplineParticleConfig, 'onHandoff'> = {},
): ATSplineParticleLife {
  const R = 0.05;  // injection radius around arrival point
  return new ATSplineParticleLife(device, canvas, edges, {
    ...config,
    onHandoff: (_eId, _tId, x, y, _vx, _vy, species) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, species);
    },
  });
}

/**
 * Convert raw canvas-space route points to EdgeSpline control points in
 * SPH domain coordinates.
 *
 * @param edgeId    topology edge identifier
 * @param sourceId  source cell identifier
 * @param targetId  target cell identifier
 * @param points    control points in canvas-pixel space
 * @param weight    connectivity / attention weight
 * @param canvasW   canvas pixel width
 * @param canvasH   canvas pixel height
 * @param domainW   SPH domain width (world units)
 * @param domainH   SPH domain height (world units)
 * @param species   optional species tag (0–7)
 */
export function canvasRouteToEdgeSpline(
  edgeId:   string,
  sourceId: string,
  targetId: string,
  points:   Array<{ x: number; y: number }>,
  weight:   number,
  canvasW:  number,
  canvasH:  number,
  domainW:  number,
  domainH:  number,
  species   = 0,
): EdgeSpline {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight, species,
    points: points.map(p => ({ x: p.x * sx, y: p.y * sy, z: 0 })),
  };
}

// ─── Defaults re-export ────────────────────────────────────────────────────────

export const AT_SPLINE_PARTICLE_DEFAULTS = {
  maxParticles:    MAX_PARTICLES,
  texW:            TEX_W,
  texH:            TEX_H,
  particleStride:  PARTICLE_STRIDE,
  edgeStride:      EDGE_STRIDE,
  workgroupSize:   WG,
} as const;
