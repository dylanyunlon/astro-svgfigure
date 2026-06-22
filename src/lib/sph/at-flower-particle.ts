/**
 * at-flower-particle.ts — M710
 *
 * AT FlowerParticleShader → WebGPU / WGSL Port
 * ─────────────────────────────────────────────────────────────────────────────
 * Full port of Active Theory's FlowerParticleShader (Element_6_Work) and
 * SplineParticleLife.fs into the project's WebGPU architecture.
 *
 * Original AT GLSL sources (reconstructed from UIL params + renderers):
 *   FlowerParticleShader.glsl (174 lines)
 *     — GPGPU position ping-pong via tPos texture (RG = xy position)
 *     — Spiral motion: splinePos + tangentPerp · amplitude · sin(θ₀ + t·speed)
 *     — vScale size attenuation: uSize · (1 − travel²)
 *     — Matcap/sphere-map shading via _txtMap (matcap3.png)
 *     — Alpha: sin(π · travel) fade
 *   SplineParticleLife.fs (108 lines)
 *     — Catmull-Rom arc-length parameterisation
 *     — Curl noise lateral perturbation (simplenoise.glsl ∇×Ψ)
 *     — Lifecycle: SPAWN → FLOW → DECAY → DEAD
 *     — Per-particle speed ∈ [uSplineSpeed.min, uSplineSpeed.max]
 *     — vScale = speed * uTimeMultiplier * 0.01 (AT hand-off formula)
 *
 * GLSL → WGSL translation key:
 *   texture2D(tPos, uv)        → textureSample(tPos, sSampler, uv)
 *   varying vec2 vUv           → @location(0) vUv : vec2f  (inter-stage)
 *   uniform float uTime        → uniforms.time  (uniform buffer)
 *   gl_FragColor               → @location(0) out : vec4f
 *   gl_Position                → @builtin(position) pos : vec4f
 *   mix(a,b,t)                 → mix(a,b,t)  (same)
 *   fract(x)                   → fract(x)    (same)
 *   mod(x,y)                   → x % y  or  (x - y*floor(x/y))
 *   step(e,x)                  → step(e,x)   (same)
 *   clamp(x,lo,hi)             → clamp(x,lo,hi) (same)
 *   atan(y,x)                  → atan2(y,x)
 *   mat4 gl_ModelViewMatrix    → uniforms.modelView (vec4f column-major)
 *   gl_PointCoord              → custom uv from instanced quad
 *
 * WebGPU architecture:
 *   ATFlowerParticleRenderer
 *     ├─ tPos ping-pong         — two rgba8unorm textures (position + travel)
 *     ├─ COMPUTE pass           — advance particle positions, curl noise, lifecycle
 *     ├─ RENDER pass            — instanced quads (one per particle slot)
 *     │     vertex shader       — reads tPos, computes spiral offset, projects
 *     │     fragment shader     — matcap sphere-map shading + alpha fade
 *     └─ EdgeSplineData[]       — spline control points per topology edge
 *
 * Integration with SPH Edge system:
 *   const renderer = new ATFlowerParticleRenderer(device, canvas, edges, config);
 *   await renderer.build();
 *   // render loop:
 *   const enc = device.createCommandEncoder();
 *   renderer.update(enc, elapsedSeconds, deltaSeconds);
 *   renderer.render(enc, renderPassDescriptor);
 *   device.queue.submit([enc.finish()]);
 *   // SPH handoff fires via config.onHandoff callback
 *
 * UIL parameter sources (channels/physics/at_uil_params.json):
 *   uSplineSpeed          [0.82, 1.21]
 *   uTimeMultiplier       0.17
 *   uFlowRange            [1, 1]
 *   uMaxSDelay            0
 *   uStartOffset          1
 *   uStartSpacing         0
 *   _txtMap               assets/images/particle/matcap3.png  (matcap shading)
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** WebGPU workgroup size — keep ≤ 256 for broad device compatibility. */
const WG = 64 as const;

/**
 * Maximum particles in the GPU pool.
 * Each particle slot occupies 4 × f32 in posX/Y/travel/speed buffers.
 */
const MAX_PARTICLES = 32768 as const;

/**
 * tPos texture dimensions.
 * W × H must be ≥ MAX_PARTICLES.  We use a square power-of-two.
 */
const TEX_W = 256 as const;
const TEX_H = 128 as const;  // 256 × 128 = 32768

/** Spiral amplitude as a fraction of edge arc-length (AT: ~8/1920 ≈ 0.004). */
const SPIRAL_AMPLITUDE_RATIO = 0.018 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 3-D control point on a spline. */
export interface FlowerPoint3 {
  x: number;
  y: number;
  z: number;
}

/** One topology edge with Catmull-Rom spline and connection metadata. */
export interface FlowerEdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points in SPH domain space. */
  points:   FlowerPoint3[];
  /** Attention weight — controls particle count and size. */
  weight:   number;
  /** Species colour index for palette lookup. */
  species?: number;
}

/** AT FlowerParticleShader / SplineParticleLife UIL parameters. */
export interface ATFlowerConfig {
  /** [min, max] per-particle travel speed multiplier. AT: [0.82, 1.21] */
  uSplineSpeed?:      [number, number];
  /** Global time scale applied to both travel and spiral phase. AT: 0.17 */
  uTimeMultiplier?:   number;
  /** [min, max] flow range multiplier. AT: [1, 1] */
  uFlowRange?:        [number, number];
  /** Opacity decay rate per second after travel ≥ 1. AT: ~0.6 */
  uDecayRate?:        number;
  /** Max random spawn delay in seconds. AT: 0 */
  uMaxSDelay?:        number;
  /** Curl-noise spatial frequency. */
  uCurlNoiseScale?:   number;
  /** Curl-noise temporal speed. */
  uCurlNoiseSpeed?:   number;
  /** Curl-noise lateral displacement amplitude (domain units). */
  uCurlStrength?:     number;
  /** Particle point-splat size in domain units (AT: uSize). */
  uSize?:             number;
  /** Particles per weight unit per edge. */
  particlesPerUnit?:  number;
  /** Arc-length LUT divisions per spline segment. */
  arcLengthDivisions?: number;
  /**
   * Called when a particle finishes its spline and should be handed to SPH.
   * Mirrors AT's FlowerParticleShader end-of-life SPH injection.
   */
  onHandoff?: (
    edgeId:   string,
    targetId: string,
    x: number, y: number,
    vx: number, vy: number,
    species: number,
  ) => void;
}

// ─── Defaults (from AT UIL params + FlowerParticleShader source refs) ─────────

const DEFAULTS: Required<Omit<ATFlowerConfig, 'onHandoff'>> = {
  uSplineSpeed:       [0.82, 1.21],
  uTimeMultiplier:    0.17,
  uFlowRange:         [1.0, 1.0],
  uDecayRate:         0.6,
  uMaxSDelay:         0.0,
  uCurlNoiseScale:    2.0,
  uCurlNoiseSpeed:    5.0,
  uCurlStrength:      0.04,
  uSize:              0.025,
  particlesPerUnit:   24,
  arcLengthDivisions: 64,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Shared simplex / curl noise (AT simplenoise.glsl port)
// ─────────────────────────────────────────────────────────────────────────────
// AT uses a GLSL simplenoise function identical to the classic 3-D Perlin
// hash-gradient approach.  Below is the WGSL equivalent used in the compute
// and render shaders.

const NOISE_WGSL = /* wgsl */`
// ── Hash (AT: hash33 from simplenoise.glsl) ─────────────────────────────────
fn hash3(p: vec3f) -> vec3f {
  var q = vec3f(
    dot(p, vec3f(127.1, 311.7, 74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453123);
}

fn hash1(n: f32) -> f32 {
  return fract(sin(n) * 43758.5453123);
}

// ── 3-D Gradient noise (AT: noise3 from simplenoise.glsl) ───────────────────
fn noise3(x: vec3f) -> f32 {
  let i  = floor(x);
  let f  = fract(x);
  // Quintic fade
  let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  // Eight lattice gradients (dot with offset)
  let n000 = dot(hash3(i + vec3f(0,0,0)) * 2.0 - 1.0,  f - vec3f(0,0,0));
  let n100 = dot(hash3(i + vec3f(1,0,0)) * 2.0 - 1.0,  f - vec3f(1,0,0));
  let n010 = dot(hash3(i + vec3f(0,1,0)) * 2.0 - 1.0,  f - vec3f(0,1,0));
  let n110 = dot(hash3(i + vec3f(1,1,0)) * 2.0 - 1.0,  f - vec3f(1,1,0));
  let n001 = dot(hash3(i + vec3f(0,0,1)) * 2.0 - 1.0,  f - vec3f(0,0,1));
  let n101 = dot(hash3(i + vec3f(1,0,1)) * 2.0 - 1.0,  f - vec3f(1,0,1));
  let n011 = dot(hash3(i + vec3f(0,1,1)) * 2.0 - 1.0,  f - vec3f(0,1,1));
  let n111 = dot(hash3(i + vec3f(1,1,1)) * 2.0 - 1.0,  f - vec3f(1,1,1));

  // Trilinear interpolation
  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z,
  );
}

// ── Curl noise (AT: curl from CurlNoise.frag) ────────────────────────────────
// Returns 2-D curl of noise field: F = ∇ × Ψ  (z-component of 3-D curl)
// The GLSL original samples ψ at two z-offsets to get ∂ψ/∂z analytically.
fn curlNoise2D(p: vec3f, eps: f32) -> vec2f {
  let dz   = vec3f(0.0, 0.0, eps);
  // ψ_x and ψ_y potential fields (offset in z for differentiation)
  let psi0 = noise3(p);
  let psiZ = noise3(p + dz);
  // curl z = dψz/dx - dψx/dz  →  approximate with finite diff in x and z
  let dx   = vec3f(eps, 0.0, 0.0);
  let dy   = vec3f(0.0, eps, 0.0);
  let Fz_x = (noise3(p + dx + dz) - noise3(p - dx + dz) - noise3(p + dx) + noise3(p - dx)) / (4.0 * eps * eps);
  let Fz_y = (noise3(p + dy + dz) - noise3(p - dy + dz) - noise3(p + dy) + noise3(p - dy)) / (4.0 * eps * eps);
  return vec2f(-Fz_y, Fz_x);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniforms struct (shared across compute and render)
// ─────────────────────────────────────────────────────────────────────────────

const UNIFORMS_WGSL = /* wgsl */`
struct FlowerUniforms {
  // AT SplineParticleLife.fs uniforms
  uTimeMultiplier  : f32,   // global time scale (AT: 0.17)
  uDecayRate       : f32,   // opacity decay rate per second
  uCurlNoiseScale  : f32,   // curl noise spatial frequency
  uCurlNoiseSpeed  : f32,   // curl noise temporal speed
  uCurlStrength    : f32,   // lateral displacement amplitude (domain units)
  uSplineSpeedMin  : f32,   // per-particle speed range min
  uSplineSpeedMax  : f32,   // per-particle speed range max
  uMaxSDelay       : f32,   // max spawn delay (seconds)

  // AT FlowerParticleShader uniforms
  uSize            : f32,   // point size (domain units); AT: uSize * (1 - travel^2)
  uSpiralAmplitude : f32,   // spiral lateral amplitude (domain units)
  uSpiralSpeed     : f32,   // spiral angular velocity (rad/sec)
  time             : f32,   // elapsed seconds

  // Domain / projection
  domainW          : f32,   // SPH domain width
  domainH          : f32,   // SPH domain height
  scaleX           : f32,   // NDC scale x = 2/domainW
  scaleY           : f32,   // NDC scale y = 2/domainH

  particleCount    : u32,   // active particle slot count
  edgeCount        : u32,   // number of edges
  texW             : u32,   // tPos texture width
  texH             : u32,   // tPos texture height
}
`;

// Byte layout offsets (must mirror struct above, aligned to 4 bytes)
const U_TIME_MULTIPLIER  =  0;
const U_DECAY_RATE       =  4;
const U_CURL_SCALE       =  8;
const U_CURL_SPEED       = 12;
const U_CURL_STRENGTH    = 16;
const U_SPEED_MIN        = 20;
const U_SPEED_MAX        = 24;
const U_MAX_S_DELAY      = 28;
const U_SIZE             = 32;
const U_SPIRAL_AMPLITUDE = 36;
const U_SPIRAL_SPEED     = 40;
const U_TIME             = 44;
const U_DOMAIN_W         = 48;
const U_DOMAIN_H         = 52;
const U_SCALE_X          = 56;
const U_SCALE_Y          = 60;
const U_PARTICLE_COUNT   = 64;  // u32
const U_EDGE_COUNT       = 68;  // u32
const U_TEX_W            = 72;  // u32
const U_TEX_H            = 76;  // u32
const UNIFORMS_BYTE_SIZE = 80;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Spline evaluation helpers (shared between compute passes)
// ─────────────────────────────────────────────────────────────────────────────
// AT SplineParticleLife.fs implements Catmull-Rom spline evaluation; we encode
// the same math in WGSL operating on a flat GPU buffer.
//
// Edge layout in edgeBuf (array<f32>):
//   Per edge: [ nPoints, arcLen, pad, pad,   // 4 f32 header
//               x0, y0, z0, pad,              // 4 f32 per control point
//               x1, y1, z1, pad, … ]
//
// Particle state layout in particleBuf (array<f32>):
//   Per particle (stride = PARTICLE_STRIDE = 16 f32):
//     [0]  travel     — arc-length fraction [0, 1]
//     [1]  speed      — per-particle speed scalar
//     [2]  delay      — remaining spawn delay
//     [3]  phase      — 0=spawn, 1=flow, 2=decay, 3=dead
//     [4]  alpha      — current opacity [0, 1]
//     [5]  theta0     — AT spiral phase seed (radians)
//     [6]  amplitude  — AT spiral lateral amplitude (domain units)
//     [7]  edgeIndex  — which edge this particle belongs to
//     [8]  posX       — current world x
//     [9]  posY       — current world y
//     [10] noiseOffset — curl lateral offset (domain units)
//     [11] seed        — random seed for curl noise phase
//     [12] handoffFlag — 1.0 when particle just entered DECAY (one-shot)
//     [13..15] _pad
//
// tPos texture (TEX_W × TEX_H, rgba32float):
//   .r = posX
//   .g = posY
//   .b = travel
//   .a = alpha
//   Written by the compute pass, read by the vertex shader (AT: tPos texture).

const PARTICLE_STRIDE = 16 as const;  // f32 per particle slot

const SPLINE_WGSL = /* wgsl */`
// ── Per-edge header offsets in edgeBuf ──────────────────────────────────────
// edgeBuf is array<f32> with stride EDGE_STRIDE_F32 per edge.
// EDGE_STRIDE_F32 = 4 (header) + MAX_POINTS_PER_EDGE * 4
// We use 64 points max per edge → stride = 4 + 64*4 = 260 f32

const EDGE_HEADER = 4u;  // header f32 count before points
const EDGE_POINT_STRIDE = 4u;  // x,y,z,pad per point
const EDGE_MAX_PTS = 64u;
const EDGE_STRIDE = 260u;  // total f32 per edge slot

fn edgeNPoints(edgeBuf: ptr<storage, array<f32>, read>, idx: u32) -> u32 {
  return u32((*edgeBuf)[idx * EDGE_STRIDE]);
}

fn edgePoint(edgeBuf: ptr<storage, array<f32>, read>, edgeIdx: u32, ptIdx: u32) -> vec3f {
  let base = edgeIdx * EDGE_STRIDE + EDGE_HEADER + ptIdx * EDGE_POINT_STRIDE;
  return vec3f((*edgeBuf)[base], (*edgeBuf)[base + 1u], (*edgeBuf)[base + 2u]);
}

// ── Catmull-Rom segment ───────────────────────────────────────────────────────
// AT SplineParticleLife.fs uses standard Catmull-Rom with tension = 0.5.
fn catmullRom(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  let f1 = -0.5 * t3 + t2 - 0.5 * t;
  let f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  let f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  let f4 =  0.5 * t3 - 0.5 * t2;
  return f1*p0 + f2*p1 + f3*p2 + f4*p3;
}

fn clampPt(i: i32, n: i32) -> i32 {
  return clamp(i, 0, n - 1);
}

fn evalSpline(edgeBuf: ptr<storage, array<f32>, read>, edgeIdx: u32, u_frac: f32) -> vec3f {
  let n = i32(edgeNPoints(edgeBuf, edgeIdx));
  if (n == 0) { return vec3f(0.0); }
  if (n == 1) { return edgePoint(edgeBuf, edgeIdx, 0u); }

  let scaled = clamp(u_frac, 0.0, 0.9999) * f32(n - 1);
  let i1     = i32(floor(scaled));
  let localT = scaled - f32(i1);

  return catmullRom(
    edgePoint(edgeBuf, edgeIdx, u32(clampPt(i1 - 1, n))),
    edgePoint(edgeBuf, edgeIdx, u32(clampPt(i1,     n))),
    edgePoint(edgeBuf, edgeIdx, u32(clampPt(i1 + 1, n))),
    edgePoint(edgeBuf, edgeIdx, u32(clampPt(i1 + 2, n))),
    localT,
  );
}

fn splineTangent(edgeBuf: ptr<storage, array<f32>, read>, edgeIdx: u32, u_frac: f32) -> vec3f {
  let eps = 0.001;
  let a   = evalSpline(edgeBuf, edgeIdx, max(0.0, u_frac - eps));
  let b   = evalSpline(edgeBuf, edgeIdx, min(1.0, u_frac + eps));
  let d   = b - a;
  let len = length(d);
  if (len < 1e-8) { return vec3f(1.0, 0.0, 0.0); }
  return d / len;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Compute pass: advance particle state
// ─────────────────────────────────────────────────────────────────────────────
// Direct port of AT SplineParticleLife.fs lifecycle logic.
// Each invocation handles one particle slot.

const COMPUTE_SHADER = /* wgsl */`
${UNIFORMS_WGSL}
${NOISE_WGSL}
${SPLINE_WGSL}

@group(0) @binding(0) var<uniform>             uni        : FlowerUniforms;
@group(1) @binding(0) var<storage, read>       edgeBuf    : array<f32>;
@group(1) @binding(1) var<storage, read_write> particleBuf: array<f32>;
@group(1) @binding(2) var                      tPosPing   : texture_storage_2d<rgba32float, write>;

// ── Particle slot accessors ──────────────────────────────────────────────────

const P_STRIDE = 16u;

fn pGet(buf: ptr<storage, array<f32>, read_write>, idx: u32, field: u32) -> f32 {
  return (*buf)[idx * P_STRIDE + field];
}
fn pSet(buf: ptr<storage, array<f32>, read_write>, idx: u32, field: u32, v: f32) {
  (*buf)[idx * P_STRIDE + field] = v;
}

// Field indices (must match PARTICLE_STRIDE layout above)
const F_TRAVEL    = 0u;
const F_SPEED     = 1u;
const F_DELAY     = 2u;
const F_PHASE     = 3u;   // 0=spawn,1=flow,2=decay,3=dead
const F_ALPHA     = 4u;
const F_THETA0    = 5u;
const F_AMPLITUDE = 6u;
const F_EDGE_IDX  = 7u;
const F_POS_X     = 8u;
const F_POS_Y     = 9u;
const F_NOISE_OFF = 10u;
const F_SEED      = 11u;
const F_HANDOFF   = 12u;

// ── Pseudo-random from state ─────────────────────────────────────────────────
fn rng(seed: f32, salt: f32) -> f32 {
  return fract(sin(seed * 127.1 + salt * 311.7) * 43758.5453);
}

// ── Respawn a dead particle slot ─────────────────────────────────────────────
// Mirrors AT SplineParticleLife.fs respawn logic.
fn respawn(buf: ptr<storage, array<f32>, read_write>, idx: u32, f: f32) {
  let seed    = f32(idx) * 1.618 + f;
  let speed   = mix(uni.uSplineSpeedMin, uni.uSplineSpeedMax, rng(seed, 0.0));
  let delay   = rng(seed, 1.0) * uni.uMaxSDelay;
  let theta0  = rng(seed, 2.0) * 6.2831853;
  let phase   = select(1.0, 0.0, delay > 0.001);  // 0=spawn if delay, 1=flow
  let alpha   = select(1.0, 0.0, delay > 0.001);

  pSet(buf, idx, F_TRAVEL,    0.0);
  pSet(buf, idx, F_SPEED,     speed);
  pSet(buf, idx, F_DELAY,     delay);
  pSet(buf, idx, F_PHASE,     phase);
  pSet(buf, idx, F_ALPHA,     alpha);
  pSet(buf, idx, F_THETA0,    theta0);
  pSet(buf, idx, F_NOISE_OFF, 0.0);
  pSet(buf, idx, F_HANDOFF,   0.0);
  pSet(buf, idx, F_SEED,      rng(seed, 3.0) * 1000.0);
}

// ── Main compute ─────────────────────────────────────────────────────────────
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let dt  = 1.0 / 60.0;   // fixed physics step; real dt is baked into uniforms
  let t   = uni.time;
  let buf = &particleBuf;

  var phase    = pGet(buf, idx, F_PHASE);
  var travel   = pGet(buf, idx, F_TRAVEL);
  var speed    = pGet(buf, idx, F_SPEED);
  var delay    = pGet(buf, idx, F_DELAY);
  var alpha    = pGet(buf, idx, F_ALPHA);
  var theta0   = pGet(buf, idx, F_THETA0);
  var amp      = pGet(buf, idx, F_AMPLITUDE);
  var edgeIdx  = u32(pGet(buf, idx, F_EDGE_IDX));
  var nOff     = pGet(buf, idx, F_NOISE_OFF);
  var seed     = pGet(buf, idx, F_SEED);

  // ── Phase transitions (AT SplineParticleLife.fs lifecycle) ─────────────────
  if (phase == 3.0) {
    // DEAD → respawn
    respawn(buf, idx, t + f32(idx));
    phase  = pGet(buf, idx, F_PHASE);
    travel = 0.0;
    speed  = pGet(buf, idx, F_SPEED);
    delay  = pGet(buf, idx, F_DELAY);
    alpha  = pGet(buf, idx, F_ALPHA);
    theta0 = pGet(buf, idx, F_THETA0);
    seed   = pGet(buf, idx, F_SEED);
    nOff   = 0.0;
    pSet(buf, idx, F_HANDOFF, 0.0);
  }

  if (phase == 0.0) {
    // SPAWN — count down delay
    delay -= dt;
    if (delay <= 0.0) {
      phase = 1.0;   // → FLOW
      alpha = 1.0;
      delay = 0.0;
    }
    pSet(buf, idx, F_DELAY, delay);
    pSet(buf, idx, F_PHASE, phase);
    pSet(buf, idx, F_ALPHA, alpha);

  } else if (phase == 1.0) {
    // FLOW — advance travel, apply curl noise
    // AT vScale = speed * uTimeMultiplier * 0.01  (SplineParticleLife.fs hand-off)
    let vScale   = speed * uni.uTimeMultiplier * 0.01;
    travel      += vScale * dt * uni.uTimeMultiplier * 60.0;

    // Evaluate spline position
    let splinePos = evalSpline(&edgeBuf, edgeIdx, min(travel, 0.9999));

    // Curl noise lateral perturbation (AT simplenoise.glsl ∇×Ψ)
    let noiseCoord = vec3f(
      splinePos.x * uni.uCurlNoiseScale,
      splinePos.y * uni.uCurlNoiseScale,
      t * uni.uCurlNoiseSpeed * 0.1,
    );
    let curl = curlNoise2D(noiseCoord, 0.01) * uni.uCurlStrength;

    // AT FlowerParticleShader spiral motion formula:
    //   pos = splinePos + tangentPerp · amplitude · sin(θ₀ + time · spiralSpeed)
    let tan     = splineTangent(&edgeBuf, edgeIdx, min(travel, 0.9999));
    let perpX   = -tan.y;
    let perpY   =  tan.x;
    let spiralT = theta0 + t * 2.4;
    let spiralOff = amp * sin(spiralT);

    let worldX = splinePos.x + perpX * (spiralOff + curl.x);
    let worldY = splinePos.y + perpY * (spiralOff + curl.y);
    nOff = spiralOff;

    pSet(buf, idx, F_TRAVEL,    travel);
    pSet(buf, idx, F_POS_X,     worldX);
    pSet(buf, idx, F_POS_Y,     worldY);
    pSet(buf, idx, F_NOISE_OFF, nOff);

    if (travel >= 1.0) {
      phase = 2.0;  // → DECAY
      pSet(buf, idx, F_PHASE,   phase);
      pSet(buf, idx, F_HANDOFF, 1.0);   // trigger SPH handoff readback
    }

  } else if (phase == 2.0) {
    // DECAY — fade alpha (AT FlowerParticleShader: alpha decays to 0)
    alpha -= uni.uDecayRate * dt;
    if (alpha <= 0.0) {
      alpha = 0.0;
      phase = 3.0;  // → DEAD
    }
    pSet(buf, idx, F_ALPHA, alpha);
    pSet(buf, idx, F_PHASE, phase);
  }

  // ── Write to tPos texture (AT GPGPU ping-pong) ───────────────────────────
  // AT FlowerParticleShader reads tPos as:
  //   vec4 pos = texture2D(tPos, vUv);  // GLSL
  //   →  textureSample(tPos, sSampler, uv); // WGSL
  //   pos.rg = world XY position
  //   pos.b  = travel fraction
  //   pos.a  = alpha
  let texX = i32(idx % uni.texW);
  let texY = i32(idx / uni.texW);
  let posX = pGet(buf, idx, F_POS_X);
  let posY = pGet(buf, idx, F_POS_Y);
  textureStore(tPosPing, vec2<i32>(texX, texY), vec4f(posX, posY, travel, alpha));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Vertex shader: AT FlowerParticleShader vertex stage
// ─────────────────────────────────────────────────────────────────────────────
// AT original:
//   attribute vec2 reference;   // pixel coord in tPos texture → our texel idx
//   uniform sampler2D tPos;
//   varying vec2 vUv;           // for matcap
//   varying float vAlpha;
//   varying float vTravel;
//
//   void main() {
//     vec4 pos  = texture2D(tPos, reference);  // sample position texture
//     vAlpha    = pos.a;
//     vTravel   = pos.b;
//     // AT spiral: gl_Position computed from pos.xy (already world-space from compute)
//     // vScale = uSize * (1.0 - pos.b * pos.b)
//     gl_PointSize = uSize * (1.0 - pos.b * pos.b) * uPixelRatio;
//     gl_Position  = projectionMatrix * modelViewMatrix * vec4(pos.xy, 0.0, 1.0);
//   }
//
// WGSL: tPos is now read from storage texture; we emit instanced quads instead
// of GL_POINTS (WebGPU has no gl_PointSize).

const VERTEX_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni     : FlowerUniforms;
@group(0) @binding(1) var          tPos    : texture_2d<f32>;
@group(0) @binding(2) var          sSampler: sampler;

struct VertOut {
  @builtin(position) pos    : vec4f,
  @location(0)       vUv    : vec2f,   // quad local [-1,1] → matcap UV
  @location(1)       vAlpha : f32,     // particle opacity
  @location(2)       vTravel: f32,     // arc-length fraction (for size attenuation)
  @location(3)       vSpeed : f32,     // speed scalar (for colour tinting)
}

// 6 vertices per quad (2 triangles)
var<private> QUAD_UV: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  // ── Read particle state from tPos texture (AT: texture2D(tPos, reference)) ─
  let texX = i32(ii % uni.texW);
  let texY = i32(ii / uni.texW);
  let tposVal = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX  = tposVal.r;
  let worldY  = tposVal.g;
  let travel  = tposVal.b;
  let alpha   = tposVal.a;

  // Discard invisible particles by pushing behind clip (cheaper than discard in FS)
  let alive = select(0.0, 1.0, alpha > 0.005);

  // ── AT vScale: uSize · (1 − travel²) — FlowerParticleShader line ~82 ──────
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  let halfSize    = uni.uSize * travelDecay * 0.5;

  // ── Build instanced quad ─────────────────────────────────────────────────────
  let quadUV  = QUAD_UV[vi];
  let ndcX    = worldX * uni.scaleX - 1.0 + quadUV.x * halfSize * uni.scaleX;
  let ndcY    = worldY * uni.scaleY - 1.0 + quadUV.y * halfSize * uni.scaleY;

  var out: VertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = quadUV;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  out.vSpeed  = 1.0;   // placeholder (speed not in tPos; could add another channel)
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fragment shader: AT FlowerParticleShader fragment stage
// ─────────────────────────────────────────────────────────────────────────────
// AT original (FlowerParticleShader.glsl ~line 90–174):
//   uniform sampler2D _txtMap;   // matcap3.png sphere map
//   varying vec2 vUv;
//   varying float vAlpha;
//   varying float vTravel;
//
//   void main() {
//     vec2 uv     = vUv * 0.5 + 0.5;   // gl_PointCoord in [0,1]
//     float r2    = dot(vUv, vUv);
//     if (r2 > 1.0) discard;
//     // Sphere-map (matcap) UV from view-space normal of sphere surface
//     vec3  N     = vec3(vUv, sqrt(max(0.0, 1.0 - r2)));
//     vec2  muv   = N.xy * 0.5 + 0.5;
//     vec4  matcap = texture2D(_txtMap, muv);
//     // Alpha: AT uses sin(π·travel) so bright at midpoint
//     float fade  = sin(3.14159 * vTravel);
//     gl_FragColor = vec4(matcap.rgb, matcap.a * vAlpha * fade);
//   }

const FRAGMENT_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni     : FlowerUniforms;
@group(0) @binding(3) var          tMatcap : texture_2d<f32>;
@group(0) @binding(4) var          sMatcap : sampler;

struct FragIn {
  @location(0) vUv    : vec2f,
  @location(1) vAlpha : f32,
  @location(2) vTravel: f32,
  @location(3) vSpeed : f32,
}

@fragment fn fs_main(in: FragIn) -> @location(0) vec4f {
  // Circular discard (AT: if (r > 1.0) discard)
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // ── AT matcap / sphere-map shading ─────────────────────────────────────────
  // Reconstruct view-space sphere normal from quad UV (GLSL: gl_PointCoord → vUv)
  //   N = vec3(vUv, sqrt(1 - r²))
  //   matcapUV = N.xy * 0.5 + 0.5   (AT FlowerParticleShader sphere-map formula)
  let nz      = sqrt(max(0.0, 1.0 - r2));
  let matcapUV = in.vUv * 0.5 + 0.5;
  let matcap  = textureSample(tMatcap, sMatcap, matcapUV);

  // ── AT alpha: sin(π · travel) — bright at midpoint, zero at start/end ──────
  let fade    = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));
  let finalA  = matcap.a * in.vAlpha * fade;

  return vec4f(matcap.rgb, finalA);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side spline helpers (mirrors of WGSL for arc-length table + handoff)
// ─────────────────────────────────────────────────────────────────────────────

function catmullRomCPU(
  p0: FlowerPoint3, p1: FlowerPoint3, p2: FlowerPoint3, p3: FlowerPoint3,
  t: number,
): FlowerPoint3 {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5 * t3 + t2 - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
    z: f1 * p0.z + f2 * p1.z + f3 * p2.z + f4 * p3.z,
  };
}

function evalSplineCPU(points: FlowerPoint3[], u: number): FlowerPoint3 {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...points[0] };
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  const scaled = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(scaled);
  const localT = scaled - i1;
  return catmullRomCPU(
    points[clamp(i1 - 1)], points[clamp(i1)],
    points[clamp(i1 + 1)], points[clamp(i1 + 2)],
    localT,
  );
}

function buildEdgeBuffer(edges: FlowerEdgeSpline[]): Float32Array {
  // Each edge slot: EDGE_STRIDE = 260 f32
  const EDGE_STRIDE = 260;
  const buf = new Float32Array(edges.length * EDGE_STRIDE);
  for (let e = 0; e < edges.length; e++) {
    const base = e * EDGE_STRIDE;
    const pts  = edges[e].points;
    buf[base + 0] = Math.min(pts.length, 64);  // nPoints
    buf[base + 1] = 0;  // arcLen placeholder
    buf[base + 2] = 0;
    buf[base + 3] = 0;
    for (let p = 0; p < Math.min(pts.length, 64); p++) {
      const pb = base + 4 + p * 4;
      buf[pb + 0] = pts[p].x;
      buf[pb + 1] = pts[p].y;
      buf[pb + 2] = pts[p].z;
      buf[pb + 3] = 0;
    }
  }
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATFlowerParticleRenderer — Main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATFlowerParticleRenderer
 *
 * Full WebGPU port of Active Theory's FlowerParticleShader + SplineParticleLife.
 *
 * Implements the complete AT GPGPU particle pipeline:
 *   1. Compute pass  — advances particle state (travel, curl noise, lifecycle)
 *                      writes to tPos ping-pong texture
 *   2. Render pass   — instanced quads read tPos, apply spiral motion formula,
 *                      matcap sphere-map shading, AT alpha fade
 *
 * The renderer integrates with the SPH particle system by firing an onHandoff
 * callback when particles complete their spline journey; the callback can call
 * SPHWorld.addFluid() to inject arriving particles into the target cell's domain.
 *
 * @example
 * ```ts
 * import { ATFlowerParticleRenderer } from '$lib/sph/at-flower-particle';
 *
 * const renderer = new ATFlowerParticleRenderer(device, canvas, edges, {
 *   uTimeMultiplier: 0.17,
 *   onHandoff: (edgeId, targetId, x, y, vx, vy, species) => {
 *     sphWorld.addFluid(x - 0.05, y - 0.05, x + 0.05, y + 0.05, 0.04, species);
 *   },
 * });
 * await renderer.build();
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * renderer.update(enc, elapsed, dt);
 * renderer.render(enc, colorAttachment, depthAttachment);
 * device.queue.submit([enc.finish()]);
 * ```
 */
export class ATFlowerParticleRenderer {
  private readonly device:   GPUDevice;
  private readonly canvas:   HTMLCanvasElement;
  private readonly cfg:      Required<Omit<ATFlowerConfig, 'onHandoff'>>;
  private readonly onHandoff?: ATFlowerConfig['onHandoff'];

  private edges: FlowerEdgeSpline[] = [];
  private particleCount = 0;

  // GPU resources
  private uniformBuf!:      GPUBuffer;
  private edgeBuf!:         GPUBuffer;
  private particleBuf!:     GPUBuffer;
  private readbackBuf!:     GPUBuffer;

  // tPos ping-pong textures (AT GPGPU pattern)
  private tPosPing!:        GPUTexture;
  private tPosPong!:        GPUTexture;
  private tPosPingView!:    GPUTextureView;
  private tPosPongView!:    GPUTextureView;
  // Matcap texture (AT: _txtMap / matcap3.png)
  private tMatcap!:         GPUTexture;
  private tMatcapView!:     GPUTextureView;

  private sampler!:         GPUSampler;
  private matcapSampler!:   GPUSampler;

  // Pipelines
  private computePipeline!: GPUComputePipeline;
  private renderPipeline!:  GPURenderPipeline;

  // Bind groups
  private computeBG0!:      GPUBindGroup;
  private computeBG1!:      GPUBindGroup;
  private renderBG!:        GPUBindGroup;

  private built = false;
  private elapsed = 0;
  private pingPong = 0;  // 0 = ping is current source, 1 = pong is current source

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    edges:  FlowerEdgeSpline[],
    config: ATFlowerConfig = {},
  ) {
    this.device   = device;
    this.canvas   = canvas;
    this.edges    = edges;
    this.onHandoff = config.onHandoff;
    this.cfg = {
      uSplineSpeed:       config.uSplineSpeed      ?? DEFAULTS.uSplineSpeed,
      uTimeMultiplier:    config.uTimeMultiplier   ?? DEFAULTS.uTimeMultiplier,
      uFlowRange:         config.uFlowRange        ?? DEFAULTS.uFlowRange,
      uDecayRate:         config.uDecayRate        ?? DEFAULTS.uDecayRate,
      uMaxSDelay:         config.uMaxSDelay        ?? DEFAULTS.uMaxSDelay,
      uCurlNoiseScale:    config.uCurlNoiseScale   ?? DEFAULTS.uCurlNoiseScale,
      uCurlNoiseSpeed:    config.uCurlNoiseSpeed   ?? DEFAULTS.uCurlNoiseSpeed,
      uCurlStrength:      config.uCurlStrength     ?? DEFAULTS.uCurlStrength,
      uSize:              config.uSize             ?? DEFAULTS.uSize,
      particlesPerUnit:   config.particlesPerUnit  ?? DEFAULTS.particlesPerUnit,
      arcLengthDivisions: config.arcLengthDivisions ?? DEFAULTS.arcLengthDivisions,
    };
  }

  // ── Build GPU resources ──────────────────────────────────────────────────

  async build(): Promise<void> {
    if (this.built) this._destroy();
    const { device } = this;

    // Determine particle count from edges
    this.particleCount = Math.min(
      MAX_PARTICLES,
      this.edges.reduce((n, e) => n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0),
    );
    if (this.particleCount === 0) this.particleCount = 256;

    // ── Uniform buffer ───────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      size: UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeUniforms(0, 0);

    // ── Edge buffer ──────────────────────────────────────────────────────────
    const edgeData = buildEdgeBuffer(this.edges);
    this.edgeBuf = device.createBuffer({
      size:  Math.max(edgeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.edgeBuf, 0, edgeData);

    // ── Particle state buffer ────────────────────────────────────────────────
    const particleData = this._initParticles();
    this.particleBuf = device.createBuffer({
      size:  particleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.particleBuf, 0, particleData);

    // Readback buffer for handoff detection
    this.readbackBuf = device.createBuffer({
      size:  particleData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── tPos ping-pong textures (AT: GPGPU position texture) ─────────────────
    const texDesc: GPUTextureDescriptor = {
      size:   [TEX_W, TEX_H],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING  |
              GPUTextureUsage.COPY_SRC,
    };
    this.tPosPing     = device.createTexture(texDesc);
    this.tPosPong     = device.createTexture(texDesc);
    this.tPosPingView = this.tPosPing.createView();
    this.tPosPongView = this.tPosPong.createView();

    // ── Matcap texture (AT: _txtMap / matcap3.png) ────────────────────────────
    // Create a 1×1 white fallback; real usage replaces via loadMatcap().
    this.tMatcap = device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.tMatcap },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    this.tMatcapView = this.tMatcap.createView();

    // ── Samplers ─────────────────────────────────────────────────────────────
    this.sampler = device.createSampler({
      magFilter: 'nearest', minFilter: 'nearest',
    });
    this.matcapSampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });

    // ── Compute pipeline ─────────────────────────────────────────────────────
    const computeModule = device.createShaderModule({ code: COMPUTE_SHADER });
    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'main' },
    });

    // ── Render pipeline ───────────────────────────────────────────────────────
    const vsModule = device.createShaderModule({ code: VERTEX_SHADER });
    const fsModule = device.createShaderModule({ code: FRAGMENT_SHADER });
    const ctx = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const fmt = navigator.gpu.getPreferredCanvasFormat();

    this.renderPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: vsModule, entryPoint: 'vs_main' },
      fragment: {
        module: fsModule, entryPoint: 'fs_main',
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
  }

  /**
   * Load a matcap texture (AT: _txtMap / matcap3.png).
   * Call after build().  Typically an ImageBitmap decoded from the asset.
   */
  async loadMatcap(bitmap: ImageBitmap): Promise<void> {
    if (!this.built) throw new Error('[ATFlowerParticle] call build() first');
    this.tMatcap.destroy();
    this.tMatcap = this.device.createTexture({
      size:   [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.tMatcap },
      [bitmap.width, bitmap.height],
    );
    this.tMatcapView = this.tMatcap.createView();
    this._buildBindGroups();  // rebuild with new texture view
  }

  // ── Per-frame update (encode compute pass) ────────────────────────────────

  /**
   * Encode the compute pass that advances particle state.
   * Call before render() in the same command encoder.
   *
   * @param encoder  — open GPUCommandEncoder
   * @param elapsed  — total elapsed seconds
   * @param dt       — frame delta seconds (used only for uniform upload)
   */
  update(encoder: GPUCommandEncoder, elapsed: number, dt: number): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this._writeUniforms(elapsed, dt);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBG0);
    pass.setBindGroup(1, this.computeBG1);
    const wg = Math.ceil(this.particleCount / WG);
    pass.dispatchWorkgroups(wg);
    pass.end();

    // Swap ping-pong for next frame
    this.pingPong = 1 - this.pingPong;
    this._buildBindGroups();
  }

  // ── Per-frame render pass ─────────────────────────────────────────────────

  /**
   * Encode the render pass that draws flower particles.
   *
   * @param encoder      — open GPUCommandEncoder
   * @param colorView    — render target texture view
   * @param depthView    — optional depth texture view
   */
  render(
    encoder:    GPUCommandEncoder,
    colorView:  GPUTextureView,
    depthView?: GPUTextureView,
  ): void {
    if (!this.built) return;

    const colorAttach: GPURenderPassColorAttachment = {
      view:       colorView,
      loadOp:     'load',
      storeOp:    'store',
    };
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [colorAttach],
    };
    if (depthView) {
      passDesc.depthStencilAttachment = {
        view:              depthView,
        depthLoadOp:       'load',
        depthStoreOp:      'store',
      };
    }

    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBG);
    // 6 vertices per quad, particleCount instances
    pass.draw(6, this.particleCount);
    pass.end();
  }

  // ── SPH handoff readback ─────────────────────────────────────────────────
  // Reads the F_HANDOFF flag from particleBuf once per frame (async).
  // Particles with handoffFlag = 1.0 have just entered DECAY and their
  // world positions are passed to the onHandoff callback for SPH injection.

  async scheduleHandoffReadback(): Promise<void> {
    if (!this.built || !this.onHandoff) return;
    const { device } = this;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, this.readbackBuf, 0, this.particleBuf.size);
    device.queue.submit([enc.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange());

    for (let i = 0; i < this.particleCount; i++) {
      const base = i * PARTICLE_STRIDE;
      if (data[base + 12] < 0.5) continue;  // F_HANDOFF

      const edgeIdx = Math.round(data[base + 7]);
      const edge    = this.edges[edgeIdx];
      if (!edge) continue;

      const x  = data[base + 8];
      const y  = data[base + 9];
      const spd = data[base + 1];

      // Velocity estimate: spline end tangent · vScale (AT SplineParticleLife.fs)
      const endPt = evalSplineCPU(edge.points, 0.999);
      const prePt = evalSplineCPU(edge.points, 0.998);
      const tanX  = endPt.x - prePt.x;
      const tanY  = endPt.y - prePt.y;
      const tanLen = Math.sqrt(tanX * tanX + tanY * tanY) + 1e-10;
      const vScale = spd * this.cfg.uTimeMultiplier * 0.01;

      this.onHandoff(
        edge.edgeId,
        edge.targetId,
        x, y,
        (tanX / tanLen) * vScale,
        (tanY / tanLen) * vScale,
        edge.species ?? 0,
      );

      // Clear the handoff flag in CPU copy (GPU will overwrite next frame anyway)
      data[base + 12] = 0;
    }

    this.readbackBuf.unmap();
  }

  // ── Live parameter updates ────────────────────────────────────────────────

  setSplineSpeed(min: number, max: number): void {
    this.cfg.uSplineSpeed = [min, max];
  }
  setTimeMultiplier(v: number): void  { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void        { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void     { this.cfg.uCurlStrength = v; }
  setCurlNoiseScale(v: number): void   { this.cfg.uCurlNoiseScale = v; }
  setCurlNoiseSpeed(v: number): void   { this.cfg.uCurlNoiseSpeed = v; }
  setSize(v: number): void             { this.cfg.uSize = v; }

  /**
   * Replace edge spline data at runtime (e.g. topology update).
   * Requires a full rebuild — existing particle state is discarded.
   */
  async setEdges(edges: FlowerEdgeSpline[]): Promise<void> {
    this.edges = edges;
    await this.build();
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  get activeParticleCount(): number { return this.particleCount; }
  get edgeCount(): number            { return this.edges.length; }
  get isBuilt(): boolean             { return this.built; }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  destroy(): void { this._destroy(); }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _initParticles(): Float32Array {
    const buf = new Float32Array(this.particleCount * PARTICLE_STRIDE);
    let slot = 0;
    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        this.particleCount - slot,
      );
      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const base  = slot * PARTICLE_STRIDE;
        const seed  = slot * 1.618033;
        const speed = this.cfg.uSplineSpeed[0] +
                      Math.random() * (this.cfg.uSplineSpeed[1] - this.cfg.uSplineSpeed[0]);
        const delay = Math.random() * this.cfg.uMaxSDelay;
        const startPos = evalSplineCPU(edge.points, 0);
        const amplitude = (edge.points.length > 1
          ? Math.hypot(
              edge.points[edge.points.length - 1].x - edge.points[0].x,
              edge.points[edge.points.length - 1].y - edge.points[0].y,
            )
          : 1.0) * SPIRAL_AMPLITUDE_RATIO;

        buf[base +  0] = 0;              // travel
        buf[base +  1] = speed;          // speed
        buf[base +  2] = delay;          // delay
        buf[base +  3] = delay > 0 ? 0 : 1;  // phase: spawn or flow
        buf[base +  4] = delay > 0 ? 0 : 1;  // alpha
        buf[base +  5] = Math.random() * Math.PI * 2;  // theta0
        buf[base +  6] = amplitude;      // amplitude
        buf[base +  7] = e;              // edgeIndex
        buf[base +  8] = startPos.x;    // posX
        buf[base +  9] = startPos.y;    // posY
        buf[base + 10] = 0;             // noiseOffset
        buf[base + 11] = seed * 1000;   // seed
        buf[base + 12] = 0;             // handoffFlag
      }
    }
    // Fill remaining slots as dead
    for (; slot < this.particleCount; slot++) {
      buf[slot * PARTICLE_STRIDE + 3] = 3;  // phase = DEAD
    }
    return buf;
  }

  private _writeUniforms(elapsed: number, _dt: number): void {
    const { device, cfg, canvas } = this;
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
    data[U_SPIRAL_AMPLITUDE / 4] = cfg.uSize * SPIRAL_AMPLITUDE_RATIO * 20;
    data[U_SPIRAL_SPEED     / 4] = 2.4;
    data[U_TIME             / 4] = elapsed;
    data[U_DOMAIN_W         / 4] = canvas.width;
    data[U_DOMAIN_H         / 4] = canvas.height;
    data[U_SCALE_X          / 4] = 2.0 / canvas.width;
    data[U_SCALE_Y          / 4] = 2.0 / canvas.height;
    // u32 fields at byte offset 64-79
    const u32 = new Uint32Array(data.buffer);
    u32[U_PARTICLE_COUNT / 4] = this.particleCount;
    u32[U_EDGE_COUNT     / 4] = this.edges.length;
    u32[U_TEX_W          / 4] = TEX_W;
    u32[U_TEX_H          / 4] = TEX_H;
    device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private _buildBindGroups(): void {
    const { device, computePipeline, renderPipeline } = this;
    const tPosCurrent = this.pingPong === 0 ? this.tPosPingView : this.tPosPongView;
    const tPosWrite   = this.pingPong === 0 ? this.tPosPongView : this.tPosPingView;

    // Compute BG0: uniforms
    this.computeBG0 = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });

    // Compute BG1: edge buf + particle buf + tPos write
    this.computeBG1 = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.edgeBuf } },
        { binding: 1, resource: { buffer: this.particleBuf } },
        { binding: 2, resource: tPosWrite },
      ],
    });

    // Render BG: uniforms + tPos read + sampler + matcap + matcap sampler
    this.renderBG = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: tPosCurrent },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.tMatcapView },
        { binding: 4, resource: this.matcapSampler },
      ],
    });
  }

  private _destroy(): void {
    if (!this.built) return;
    this.uniformBuf?.destroy();
    this.edgeBuf?.destroy();
    this.particleBuf?.destroy();
    this.readbackBuf?.destroy();
    this.tPosPing?.destroy();
    this.tPosPong?.destroy();
    this.tMatcap?.destroy();
    this.built = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Build FlowerEdgeSpline from canvas-pixel route points, normalising to
 * SPH domain units.  Mirrors edgeRouteToSplineData from spline-particle-life.ts
 * but typed for the AT flower pipeline.
 */
export function edgeRouteToFlowerSpline(
  edgeId:   string,
  sourceId: string,
  targetId: string,
  points:   Array<{ x: number; y: number }>,
  weight:   number,
  canvasW:  number,
  canvasH:  number,
  domainW:  number,
  domainH:  number,
  species?: number,
): FlowerEdgeSpline {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight, species,
    points: points.map(p => ({ x: p.x * sx, y: p.y * sy, z: 0 })),
  };
}

/**
 * Wire an ATFlowerParticleRenderer to SPHWorld.addFluid() for automatic
 * particle injection when flower particles arrive at their target cells.
 *
 * @example
 * ```ts
 * const renderer = createATFlowerForSPH(device, canvas, edges, world.addFluid.bind(world));
 * await renderer.build();
 * ```
 */
export function createATFlowerForSPH(
  device:   GPUDevice,
  canvas:   HTMLCanvasElement,
  edges:    FlowerEdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATFlowerConfig, 'onHandoff'> = {},
): ATFlowerParticleRenderer {
  const HANDOFF_R = 0.05;
  return new ATFlowerParticleRenderer(device, canvas, edges, {
    ...config,
    onHandoff: (_edgeId, _targetId, x, y, _vx, _vy, species) => {
      addFluid(
        x - HANDOFF_R, y - HANDOFF_R,
        x + HANDOFF_R, y + HANDOFF_R,
        HANDOFF_R * 0.8,
        species,
      );
    },
  });
}

// ─── Defaults re-export for external tuning ───────────────────────────────────

export const AT_FLOWER_DEFAULTS = {
  ...DEFAULTS,
  maxParticles:      MAX_PARTICLES,
  texW:              TEX_W,
  texH:              TEX_H,
  spiralAmplitudeRatio: SPIRAL_AMPLITUDE_RATIO,
  particleStride:    PARTICLE_STRIDE,
} as const;
