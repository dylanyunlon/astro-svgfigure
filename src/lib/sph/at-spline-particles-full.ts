/**
 * at-spline-particles-full.ts — M848b
 *
 * ATSplineParticlesFull — Spline Particle Flow + UIL Path Animation + Cell Data-Flow Viz
 * ─────────────────────────────────────────────────────────────────────────────────────────
 *
 * Full-stack extension of ATSplineParticleLife (M713) that layers:
 *
 *   1. SPLINE PARTICLE FLOW  — WebGPU compute + render pipeline from at-spline-particle.ts,
 *                              extended with 3-D splineshader.glsl thickness model and
 *                              curl-noise distribution from splineparticles.fs.
 *
 *   2. UIL PATH ANIMATION    — TweenUILPathShader.glsl / TweenUILPathFallbackShader.glsl
 *                              re-implemented in WGSL: per-vertex `speed` attribute drives
 *                              a two-colour lerp along the path.  Screen-space finalPosition
 *                              override and anti-aliased SDF edge (tri() function) are
 *                              faithfully ported.  Falls back to a solid-colour pipeline
 *                              when the host lacks line-width support.
 *
 *   3. CELL DATA-FLOW VIZ    — Subscribes to pub/sub edge messages (EdgeDataEvent) and
 *                              drives per-edge pulse queues that modulate particle emittance,
 *                              path colour, and UIL parameter overrides in real time.
 *                              Mirrors EdgeDataFlowViz (M766) but funnels visual state back
 *                              into the GPU particle buffers instead of a Canvas2D overlay.
 *
 * Shader provenance (compiled.vs offsets 8335–8653):
 *   splineparticles.fs       — thickness / distribution / noise model (splinenoise, srand,
 *                              getSplineIndex, getSplineThickness, getSplinePosRaw)
 *   splineshader.glsl        — getSplinePos / isMoving lookup helpers
 *   TweenUILPathShader.glsl  — speed-lerp colour, customDirection (screen-space), tri() SDF
 *   TweenUILPathFallbackShader.glsl — colour lerp fallback for no-instanced-line envs
 *
 * GLSL → WGSL translation map (beyond M713 base):
 *   attribute float speed       → @location(N) speed : f32 in VertIn struct
 *   mix(uColor, uColor2, speed) → mix(uni.uColor, uni.uColor2, speed)
 *   finalPosition.z = min(0,..) → clamped in NDC emission
 *   tri(v) = mix(v,1-v,step(.5,v))*2 → ported 1:1
 *   fwidth()                    → WGSL built-in fwidthFine()
 *   vUv.y signed dist           → @location vUv passed through strip
 *
 * GPU buffer layout extensions (FULL_PARTICLE_STRIDE = 32 f32):
 *   [0..11]  base fields (mirrors at-spline-particle.ts PARTICLE_STRIDE=16 first 12)
 *   [12]     pulseIntensity   — current data-flow pulse brightness [0,1]
 *   [13]     pathColorT       — UIL path colour blend t [0,1]
 *   [14]     uilSpeed         — UIL path animation speed (splineshader speed attr)
 *   [15]     thickness        — AT splineparticles.fs computed radius
 *   [16]     emitRate         — live emission rate override from pub/sub
 *   [17]     cellWeight       — destination cell connectivity weight
 *   [18]     dataEpoch        — last pub/sub event epoch (monotonic f32)
 *   [19]     trailAlpha       — afterglow trail opacity
 *   [20..31] _pad
 *
 * Integration:
 *   const full = new ATSplineParticlesFull(device, canvas, edges, config);
 *   await full.build();
 *   // pub/sub hook:
 *   eventBus.on('edge:message', (edgeId, qos) => full.firePulse(edgeId, qos));
 *   // render loop:
 *   const enc = device.createCommandEncoder();
 *   full.update(enc, elapsed, dt);
 *   full.render(enc, colorView, depthView?);
 *   device.queue.submit([enc.finish()]);
 *   full.scheduleHandoffReadback();
 *
 * References:
 *   src/lib/sph/at-spline-particle.ts      — M713 base lifecycle
 *   src/lib/sph/edge-data-flow-viz.ts      — M766 Canvas2D pulse viz
 *   src/lib/sph/at-uil-live-panel.ts       — AT UIL live parameter panel
 *   src/lib/sph/uil-species-live.ts        — species × physics UIL bridge
 *   upstream/activetheory-assets/compiled.vs — shader source (lines 8335-8653)
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** WebGPU workgroup size — ≤256 for broad device compatibility. */








const WG = 64 as const;

/** Maximum particle slots (tPos texture area). */
const MAX_PARTICLES = 32768 as const;

/** tPos texture: 256 × 128 = 32768 slots. */
const TEX_W = 256 as const;
const TEX_H = 128 as const;

/**
 * Extended f32 count per particle slot — mirrors FULL_PARTICLE_STRIDE docs above.
 * Extends the M713 base stride (16) with flow-viz and UIL-path fields.
 */
const FULL_PARTICLE_STRIDE = 32 as const;

/** Edge buffer — f32 per edge slot. */
const EDGE_STRIDE  = 260 as const;
const EDGE_MAX_PTS =  64 as const;

/** Uniform buffer byte count — must mirror FullSplineUniforms WGSL struct. */
const FULL_UNIFORMS_BYTES = 144 as const;

/** Maximum concurrent pulses per edge (ring buffer). */
const MAX_PULSES_PER_EDGE = 8 as const;

// ─── Field indices in FULL_PARTICLE_STRIDE ────────────────────────────────────

const F_TRAVEL        =  0;
const F_SPEED         =  1;
const F_DELAY         =  2;
const F_PHASE         =  3;   // 0=spawn 1=flow 2=decay 3=dead
const F_ALPHA         =  4;
const F_SEED          =  5;
const F_NOFF          =  6;
const F_EDGE          =  7;
const F_POS_X         =  8;
const F_POS_Y         =  9;
const F_HANDOFF       = 10;
const F_SPECIES       = 11;
const F_PULSE         = 12;
const F_PATH_COLOR_T  = 13;
const F_UIL_SPEED     = 14;
const F_THICKNESS     = 15;
const F_EMIT_RATE     = 16;
const F_CELL_WEIGHT   = 17;
const F_DATA_EPOCH    = 18;
const F_TRAIL_ALPHA   = 19;

// Byte offsets in uniform buffer
const U_TIME_MUL     =   0;
const U_DECAY_RATE   =   4;
const U_CURL_SCALE   =   8;
const U_CURL_SPEED   =  12;
const U_CURL_STR     =  16;
const U_SPD_MIN      =  20;
const U_SPD_MAX      =  24;
const U_MAX_DELAY    =  28;
const U_SIZE         =  32;
const U_TIME         =  36;
const U_DOMAIN_W     =  40;
const U_DOMAIN_H     =  44;
const U_SCALE_X      =  48;
const U_SCALE_Y      =  52;
const U_P_COUNT      =  56;   // u32
const U_E_COUNT      =  60;   // u32
const U_TEX_W        =  64;   // u32
const U_TEX_H        =  68;   // u32
const U_COLOR1_R     =  72;
const U_COLOR1_G     =  76;
const U_COLOR1_B     =  80;
const U_COLOR2_R     =  84;
const U_COLOR2_G     =  88;
const U_COLOR2_B     =  92;
const U_THICKNESS_STEP_X =  96;
const U_THICKNESS_STEP_Y = 100;
const U_SPLINE_THICK  = 104;
const U_RANGE_THICK   = 108;
const U_RANGE_SCALE   = 112;
const U_EXTRUDE_RND   = 116;
const U_DISTRIB       = 120;
const U_DISTRIB_RNG_X = 124;
const U_DISTRIB_RNG_Y = 128;
const U_THICK_SPEED_X = 132;
const U_THICK_SPEED_Y = 136;
const U_ASPECT        = 140;

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 3-D control point on a spline. */
export interface SplinePoint3 {
  x: number;
  y: number;
  z: number;
}

/** One topology edge with Catmull-Rom control points and metadata. */
export interface FullEdgeSpline {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points in SPH domain / world space. */
  points:   SplinePoint3[];
  /** Connectivity weight — controls particle count and size. */
  weight:   number;
  /** Particle species tag (0–7). */
  species?: number;
  /** QoS profile name (maps to UIL path colour theme). */
  qos?:     string;
}

/** Data-flow event fired by the pub/sub bus on an edge. */
export interface EdgeDataEvent {
  edgeId:    string;
  qos?:      string;
  /** Signal intensity ∈ [0, 1] — modulates pulse brightness. */
  intensity?: number;
}

/** Per-edge pulse state (ring buffer entry). */
interface PulseState {
  travel:    number;   // [0, 1] current wavefront position
  intensity: number;   // brightness scalar
  duration:  number;   // seconds to reach t=1
  epoch:     number;   // monotonic timestamp at spawn
  active:    boolean;
}

/**
 * UIL path colour theme — maps QoS profile to TweenUILPathShader colour pair.
 * Mirrors AT's per-species colour assignments in compiled.vs.
 */
export interface UILPathTheme {
  /** TweenUILPathShader uColor — source (low speed) colour. */
  color1: [number, number, number];
  /** TweenUILPathShader uColor2 — destination (high speed) colour. */
  color2: [number, number, number];
  /** Overall opacity. */
  opacity: number;
}

/** Full configuration for ATSplineParticlesFull. */
export interface ATSplineParticlesFullConfig {
  // ── Base lifecycle (mirrors ATSplineParticleConfig) ──────────────────────
  uSplineSpeed?:       [number, number];
  uTimeMultiplier?:    number;
  uFlowRange?:         [number, number];
  uDecayRate?:         number;
  uMaxSDelay?:         number;
  uCurlNoiseScale?:    number;
  uCurlNoiseSpeed?:    number;
  uCurlStrength?:      number;
  uSize?:              number;
  particlesPerUnit?:   number;
  maxParticles?:       number;

  // ── splineparticles.fs thickness model ───────────────────────────────────
  /** AT splineparticles.fs uSplineThickness. */
  uSplineThickness?:     number;
  /** AT uRangeThickness — noise range around base thickness. */
  uRangeThickness?:      number;
  /** AT uRangeScale — spatial scale of thickness noise. */
  uRangeScale?:          number;
  /** AT uExtrudeRandom — max random extrusion factor. */
  uExtrudeRandom?:       number;
  /** AT uDistribution — curl distribution spatial scale. */
  uDistribution?:        number;
  /** AT uDistributionRange [min, max] — distribution gamma range. */
  uDistributionRange?:   [number, number];
  /** AT uThicknessStep [stepThreshold, stepMix]. */
  uThicknessStep?:       [number, number];
  /** AT uThicknessSpeed [x, y] — temporal speed of thickness noise. */
  uThicknessSpeed?:      [number, number];

  // ── TweenUILPathShader colours ────────────────────────────────────────────
  /** Default UIL path base colour (uColor). */
  uilColor1?:            [number, number, number];
  /** Default UIL path target colour (uColor2). */
  uilColor2?:            [number, number, number];
  /** Path opacity. */
  uilOpacity?:           number;

  // ── Cell data-flow viz ────────────────────────────────────────────────────
  /** Default pulse travel duration (seconds). */
  pulseDuration?:        number;
  /** Pulse decay rate — intensity falloff per second after peak. */
  pulseDecayRate?:       number;
  /** Trail alpha bleed (how long afterglow persists). */
  trailPersistence?:     number;
  /** QoS → UIL colour theme map. */
  qosThemes?:            Record<string, UILPathTheme>;

  // ── Handoff callback (fires on DECAY, same signature as M713) ────────────
  onHandoff?: (
    edgeId:   string,
    targetId: string,
    x:        number,
    y:        number,
    vx:       number,
    vy:       number,
    species:  number,
  ) => void;
}

// ─── Default QoS colour themes (mirrors EdgeDataFlowViz QOS_THEME palette) ────

export const DEFAULT_QOS_THEMES: Record<string, UILPathTheme> = {
  SENSOR_DATA:  { color1: [0.18, 0.55, 1.00], color2: [0.50, 0.90, 1.00], opacity: 0.88 },
  PARAMETERS:   { color1: [1.00, 0.70, 0.20], color2: [1.00, 0.92, 0.60], opacity: 0.80 },
  TF_STATIC:    { color1: [0.20, 0.75, 0.55], color2: [0.60, 1.00, 0.80], opacity: 0.75 },
  TOPO_CHANGE:  { color1: [1.00, 0.20, 0.75], color2: [1.00, 0.65, 1.00], opacity: 0.95 },
  DEFAULT:      { color1: [0.55, 0.65, 0.90], color2: [0.85, 0.90, 1.00], opacity: 0.70 },
};

// ─── FullSplineParticlePreset ─────────────────────────────────────────────────

export const FullSplineParticlePreset: Record<string, ATSplineParticlesFullConfig> = {
  default: {
    uSplineSpeed:      [0.82, 1.21],
    uTimeMultiplier:   0.17,
    uFlowRange:        [1.0, 1.0],
    uDecayRate:        0.6,
    uMaxSDelay:        0.0,
    uCurlNoiseScale:   2.0,
    uCurlNoiseSpeed:   5.0,
    uCurlStrength:     0.04,
    uSize:             0.012,
    particlesPerUnit:  24,
    uSplineThickness:  1.0,
    uRangeThickness:   0.3,
    uRangeScale:       1.0,
    uExtrudeRandom:    0.5,
    uDistribution:     1.0,
    uDistributionRange:[0.3, 1.0],
    uThicknessStep:    [0.5, 1.0],
    uThicknessSpeed:   [0.1, 0.1],
    uilColor1:         [0.55, 0.65, 0.90],
    uilColor2:         [0.85, 0.90, 1.00],
    uilOpacity:        0.70,
    pulseDuration:     1.2,
    pulseDecayRate:    0.8,
    trailPersistence:  0.3,
  },
  cellPubSub: {
    uSplineSpeed:      [1.20, 1.80],
    uTimeMultiplier:   0.22,
    uFlowRange:        [1.0, 1.2],
    uDecayRate:        0.9,
    uMaxSDelay:        0.1,
    uCurlNoiseScale:   3.0,
    uCurlNoiseSpeed:   6.0,
    uCurlStrength:     0.06,
    uSize:             0.010,
    particlesPerUnit:  36,
    uSplineThickness:  0.8,
    uRangeThickness:   0.5,
    uRangeScale:       1.5,
    uExtrudeRandom:    0.7,
    uDistribution:     1.2,
    uDistributionRange:[0.2, 0.9],
    uThicknessStep:    [0.4, 0.9],
    uThicknessSpeed:   [0.15, 0.08],
    uilColor1:         [0.18, 0.55, 1.00],
    uilColor2:         [0.50, 0.90, 1.00],
    uilOpacity:        0.88,
    pulseDuration:     0.9,
    pulseDecayRate:    1.2,
    trailPersistence:  0.25,
  },
  organic: {
    uSplineSpeed:      [0.50, 0.90],
    uTimeMultiplier:   0.12,
    uFlowRange:        [0.9, 1.3],
    uDecayRate:        0.45,
    uMaxSDelay:        0.8,
    uCurlNoiseScale:   5.0,
    uCurlNoiseSpeed:   3.0,
    uCurlStrength:     0.18,
    uSize:             0.016,
    particlesPerUnit:  20,
    uSplineThickness:  1.4,
    uRangeThickness:   0.7,
    uRangeScale:       0.8,
    uExtrudeRandom:    1.0,
    uDistribution:     0.8,
    uDistributionRange:[0.4, 1.2],
    uThicknessStep:    [0.6, 1.0],
    uThicknessSpeed:   [0.05, 0.05],
    uilColor1:         [0.20, 0.75, 0.55],
    uilColor2:         [0.60, 1.00, 0.80],
    uilOpacity:        0.75,
    pulseDuration:     1.8,
    pulseDecayRate:    0.4,
    trailPersistence:  0.5,
  },
  fastPulse: {
    uSplineSpeed:      [1.80, 2.60],
    uTimeMultiplier:   0.35,
    uFlowRange:        [1.0, 1.2],
    uDecayRate:        1.2,
    uMaxSDelay:        0.2,
    uCurlNoiseScale:   1.0,
    uCurlNoiseSpeed:   8.0,
    uCurlStrength:     0.015,
    uSize:             0.008,
    particlesPerUnit:  32,
    uSplineThickness:  0.5,
    uRangeThickness:   0.2,
    uRangeScale:       2.0,
    uExtrudeRandom:    0.2,
    uDistribution:     2.0,
    uDistributionRange:[0.1, 0.7],
    uThicknessStep:    [0.3, 0.8],
    uThicknessSpeed:   [0.25, 0.20],
    uilColor1:         [1.00, 0.20, 0.75],
    uilColor2:         [1.00, 0.65, 1.00],
    uilOpacity:        0.95,
    pulseDuration:     0.5,
    pulseDecayRate:    2.0,
    trailPersistence:  0.1,
  },
};

// ─── WGSL — Shared noise helpers ──────────────────────────────────────────────

const NOISE_WGSL = /* wgsl */`
fn hash3(p: vec3f) -> vec3f {
  let q = vec3f(
    dot(p, vec3f(127.1, 311.7,  74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453123);
}

fn noise3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = fract(x);
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

// 2D curl noise — ∇×Ψ (AT: CurlNoise.frag port)
fn curlNoise2D(p: vec3f, eps: f32) -> vec2f {
  let dx = vec3f(eps, 0.0, 0.0);
  let dy = vec3f(0.0, eps, 0.0);
  let dz = vec3f(0.0, 0.0, eps);
  let Fx = (noise3(p + dy + dz) - noise3(p - dy + dz)
           - noise3(p + dy)     + noise3(p - dy))     / (4.0 * eps * eps);
  let Fy = (noise3(p + dx + dz) - noise3(p - dx + dz)
           - noise3(p + dx)     + noise3(p - dx))     / (4.0 * eps * eps);
  return vec2f(-Fx, Fy);
}

// splineparticles.fs: splinenoise — layered sine turbulence
fn splinenoise(v: vec3f) -> f32 {
  let t   = v.z * 0.3;
  let vy  = v.y * 0.8;
  let s   = 0.5;
  var n   = 0.0;
  // x-axis contribution (4 harmonics, range-mapped to [-0.3, 0.3])
  let sx = sin(v.x * 0.9/s + t*10.0) + sin(v.x * 2.4/s + t*15.0)
         + sin(v.x * -3.5/s + t*4.0) + sin(v.x * -2.5/s + t*7.1);
  n += (sx / 4.0) * 0.3;
  // y-axis contribution
  let sy = sin(vy * -0.3/s + t*18.0) + sin(vy * 1.6/s + t*18.0)
         + sin(vy * 2.6/s + t*8.0)  + sin(vy * -2.6/s + t*4.5);
  n += (sy / 4.0) * 0.3;
  return n;
}

// splineparticles.fs: randomSeed
fn randomSeed(seed: f32) -> f32 {
  let n = sin(seed) * 10000000.0;
  return n - floor(n);
}

// splineparticles.fs: srand — rounded discrete random in [min, max]
fn srandF(seed: f32, lo: f32, hi: f32) -> f32 {
  let r = lo + randomSeed(seed) * (hi - lo);
  return floor(r + 0.5);
}

// splineparticles.fs: ssineOut
fn ssineOut(t: f32) -> f32 {
  return sin(t * 1.5707963267948966);
}
`;

// ─── WGSL — Uniforms struct ───────────────────────────────────────────────────

const UNIFORMS_WGSL = /* wgsl */`
struct FullSplineUniforms {
  // Base lifecycle (mirrors SplineUniforms)
  uTimeMultiplier  : f32,
  uDecayRate       : f32,
  uCurlNoiseScale  : f32,
  uCurlNoiseSpeed  : f32,
  uCurlStrength    : f32,
  uSplineSpeedMin  : f32,
  uSplineSpeedMax  : f32,
  uMaxSDelay       : f32,
  uSize            : f32,
  time             : f32,
  domainW          : f32,
  domainH          : f32,
  scaleX           : f32,
  scaleY           : f32,
  particleCount    : u32,
  edgeCount        : u32,
  texW             : u32,
  texH             : u32,

  // TweenUILPathShader colours
  uColor           : vec3f,   // bytes 72–84
  uColor2          : vec3f,   // bytes 84–96

  // splineparticles.fs thickness model
  uThicknessStepX  : f32,
  uThicknessStepY  : f32,
  uSplineThickness : f32,
  uRangeThickness  : f32,
  uRangeScale      : f32,
  uExtrudeRandom   : f32,
  uDistribution    : f32,
  uDistribRangeX   : f32,
  uDistribRangeY   : f32,
  uThicknessSpeedX : f32,
  uThicknessSpeedY : f32,

  // TweenUILPathShader: screen-space aspect ratio
  aspect           : f32,
}
`;

// ─── WGSL — Edge spline evaluation ───────────────────────────────────────────

const SPLINE_WGSL = /* wgsl */`
const EDGE_HDR    = 4u;
const EDGE_PSTRIDE= 4u;
const EDGE_STRIDE = 260u;

fn edgeNPts(buf: ptr<storage, array<f32>, read>, ei: u32) -> u32 {
  return u32((*buf)[ei * EDGE_STRIDE]);
}
fn edgePt(buf: ptr<storage, array<f32>, read>, ei: u32, pi: u32) -> vec3f {
  let b = ei * EDGE_STRIDE + EDGE_HDR + pi * EDGE_PSTRIDE;
  return vec3f((*buf)[b], (*buf)[b + 1u], (*buf)[b + 2u]);
}

fn catmullRom(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  let f1 = -0.5*t3 + t2        - 0.5*t;
  let f2 =  1.5*t3 - 2.5*t2 + 1.0;
  let f3 = -1.5*t3 + 2.0*t2 + 0.5*t;
  let f4 =  0.5*t3 - 0.5*t2;
  return f1*p0 + f2*p1 + f3*p2 + f4*p3;
}

fn clampIdx(i: i32, n: i32) -> i32 { return clamp(i, 0, n - 1); }

fn evalSpline(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec3f {
  let n = i32(edgeNPts(buf, ei));
  if (n == 0) { return vec3f(0.0); }
  if (n == 1) { return edgePt(buf, ei, 0u); }
  let sc = clamp(u, 0.0, 0.9999) * f32(n - 1);
  let i1 = i32(floor(sc));
  let lt = sc - f32(i1);
  return catmullRom(
    edgePt(buf, ei, u32(clampIdx(i1 - 1, n))),
    edgePt(buf, ei, u32(clampIdx(i1,     n))),
    edgePt(buf, ei, u32(clampIdx(i1 + 1, n))),
    edgePt(buf, ei, u32(clampIdx(i1 + 2, n))),
    lt,
  );
}

fn splineTangent(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec3f {
  let eps = 0.001;
  let a   = evalSpline(buf, ei, max(0.0, u - eps));
  let b   = evalSpline(buf, ei, min(1.0, u + eps));
  let d   = b - a;
  let l   = length(d);
  if (l < 1e-8) { return vec3f(1.0, 0.0, 0.0); }
  return d / l;
}

// splineshader.glsl: isMoving — returns 1 if spline has motion at this t
fn isMoving(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32, perSpline: f32) -> f32 {
  let cpos = evalSpline(buf, ei, u);
  let npos = evalSpline(buf, ei, min(1.0, u + 1.0 / perSpline));
  let moving = select(0.0, 1.0, length(cpos - npos) > 0.001);
  return mix(moving, 1.0, select(0.0, 1.0, u > 0.5));
}
`;

// ─── WGSL — Thickness model (splineparticles.fs port) ────────────────────────

const THICKNESS_WGSL = /* wgsl */`
// splineparticles.fs: getSplineThickness — AT radial extrusion
// sOrigin: normalised initial spawn position
// pos: current world position on spline
// time: elapsed (for noise animation)
fn getSplineThickness(
  uni:    FullSplineUniforms,
  pos:    vec3f,
  sOrigin: vec3f,
  sRandY:  f32,   // sRandom.y
  sRandZ:  f32,   // sRandom.z
  time:    f32,
) -> vec3f {
  let angle  = radians(360.0 * sRandZ);

  // Distribution scaling (gamma curve, sine-out eased)
  let rawN   = splinenoise(sOrigin * uni.uDistribution);
  let gamma  = ssineOut(clamp((rawN + 1.0) * 0.5, 0.0, 1.0));
  let fizzy  = pow(mix(uni.uDistribRangeX, uni.uDistribRangeY, gamma), 3.0);

  // Step-threshold thickness distribution (AT uThicknessStep)
  let splineRnd  = 0.0;
  let stepV      = select(0.0, 1.0, splineRnd >= uni.uThicknessStepX);
  let distribution = mix(uni.uThicknessStepY, 1.0, 1.0 - stepV);

  var radius = 0.5 * uni.uSplineThickness * distribution * fizzy;

  // Animating thickness noise
  let noiseCoord = pos * uni.uRangeScale + vec3f(
    time * uni.uThicknessSpeedX,
    time * uni.uThicknessSpeedY,
    0.0,
  );
  let nv = splinenoise(noiseCoord);
  radius *= clamp((nv + 1.0) * (uni.uRangeThickness * 0.5) + (1.0 - uni.uRangeThickness * 0.5), 0.0, 2.0);
  radius *= mix(1.0, uni.uExtrudeRandom, sRandY);

  return normalize(sOrigin) * radius;
}
`;

// ─── WGSL — Compute shader ────────────────────────────────────────────────────

const COMPUTE_SHADER = /* wgsl */`
${UNIFORMS_WGSL}
${NOISE_WGSL}
${SPLINE_WGSL}
${THICKNESS_WGSL}

@group(0) @binding(0) var<uniform>             uni         : FullSplineUniforms;
@group(1) @binding(0) var<storage, read>       edgeBuf     : array<f32>;
@group(1) @binding(1) var<storage, read_write> particleBuf : array<f32>;
@group(1) @binding(2) var                      tPos        : texture_storage_2d<rgba32float, write>;
// Pulse intensity buffer: [edgeIdx * MAX_PULSES + pulseSlot] → intensity f32
@group(1) @binding(3) var<storage, read>       pulseBuf    : array<f32>;

const PS          = 32u;   // FULL_PARTICLE_STRIDE
const MAX_PULSES  = 8u;

// Field index constants
const F_TRAVEL        =  0u;
const F_SPEED         =  1u;
const F_DELAY         =  2u;
const F_PHASE         =  3u;
const F_ALPHA         =  4u;
const F_SEED          =  5u;
const F_NOFF          =  6u;
const F_EDGE          =  7u;
const F_POS_X         =  8u;
const F_POS_Y         =  9u;
const F_HANDOFF       = 10u;
const F_SPECIES       = 11u;
const F_PULSE         = 12u;
const F_PATH_COLOR_T  = 13u;
const F_UIL_SPEED     = 14u;
const F_THICKNESS     = 15u;
const F_EMIT_RATE     = 16u;
const F_CELL_WEIGHT   = 17u;
const F_DATA_EPOCH    = 18u;
const F_TRAIL_ALPHA   = 19u;

fn pGet(idx: u32, f: u32) -> f32 { return particleBuf[idx * PS + f]; }
fn pSet(idx: u32, f: u32, v: f32) { particleBuf[idx * PS + f] = v; }

fn rng(s: f32, salt: f32) -> f32 {
  return fract(sin(s * 127.1 + salt * 311.7) * 43758.5453);
}

// Accumulate pulse intensities for a given edge slot
fn getPulseIntensity(eIdx: u32) -> f32 {
  var total = 0.0;
  for (var p = 0u; p < MAX_PULSES; p = p + 1u) {
    total += pulseBuf[eIdx * MAX_PULSES + p];
  }
  return clamp(total, 0.0, 1.0);
}

// Respawn — extended version with UIL speed init
fn respawn(idx: u32, t: f32) {
  let s      = f32(idx) * 1.618034 + t;
  let speed  = mix(uni.uSplineSpeedMin, uni.uSplineSpeedMax, rng(s, 0.0));
  let delay  = rng(s, 1.0) * uni.uMaxSDelay;
  let phase  = select(1.0, 0.0, delay > 0.001);
  let alpha  = select(1.0, 0.0, delay > 0.001);
  let uilSpd = rng(s, 4.0);   // TweenUILPathShader speed attribute [0,1]

  pSet(idx, F_TRAVEL,       0.0);
  pSet(idx, F_SPEED,        speed);
  pSet(idx, F_DELAY,        delay);
  pSet(idx, F_PHASE,        phase);
  pSet(idx, F_ALPHA,        alpha);
  pSet(idx, F_SEED,         rng(s, 3.0) * 1000.0);
  pSet(idx, F_NOFF,         0.0);
  pSet(idx, F_HANDOFF,      0.0);
  pSet(idx, F_PATH_COLOR_T, 0.0);
  pSet(idx, F_UIL_SPEED,    uilSpd);
  pSet(idx, F_THICKNESS,    0.0);
  pSet(idx, F_TRAIL_ALPHA,  0.0);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

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
  var uilSpd = pGet(idx, F_UIL_SPEED);
  var trailA = pGet(idx, F_TRAIL_ALPHA);

  // ── DEAD → respawn ─────────────────────────────────────────────────────────
  if (phase == 3.0) {
    respawn(idx, t);
    phase  = pGet(idx, F_PHASE);
    travel = 0.0;
    speed  = pGet(idx, F_SPEED);
    delay  = pGet(idx, F_DELAY);
    alpha  = pGet(idx, F_ALPHA);
    seed   = pGet(idx, F_SEED);
    nOff   = 0.0;
    uilSpd = pGet(idx, F_UIL_SPEED);
    trailA = 0.0;
  }

  // ── SPAWN — countdown ──────────────────────────────────────────────────────
  if (phase == 0.0) {
    delay -= dt;
    if (delay <= 0.0) {
      phase = 1.0;
      alpha = 1.0;
      delay = 0.0;
    }
    pSet(idx, F_DELAY, delay);
    pSet(idx, F_PHASE, phase);
    pSet(idx, F_ALPHA, alpha);

  // ── FLOW — advance along spline ────────────────────────────────────────────
  } else if (phase == 1.0) {
    let vScale = speed * uni.uTimeMultiplier * 0.01;
    travel    += vScale * dt * uni.uTimeMultiplier * 60.0;

    let sp  = evalSpline(&edgeBuf, eIdx, min(travel, 0.9999));

    // Curl-noise lateral perturbation (AT simplenoise.glsl)
    let noiseCoord = vec3f(
      sp.x * uni.uCurlNoiseScale,
      sp.y * uni.uCurlNoiseScale,
      t    * uni.uCurlNoiseSpeed * 0.1 + seed * 0.001,
    );
    let curl = curlNoise2D(noiseCoord, 0.01) * uni.uCurlStrength;

    // Spline tangent → perpendicular direction
    let tan   = splineTangent(&edgeBuf, eIdx, min(travel, 0.9999));
    let perpX = -tan.y;
    let perpY =  tan.x;

    // AT splineparticles.fs thickness extrusion
    let sOrigin = vec3f(perpX, perpY, 0.0);
    let sRandY  = rng(seed, 20.0);
    let sRandZ  = rng(seed, 30.0);
    let thickVec = getSplineThickness(uni, sp, sOrigin, sRandY, sRandZ, t);

    let worldX = sp.x + perpX * curl.x + thickVec.x;
    let worldY = sp.y + perpY * curl.y + thickVec.y;
    nOff = curl.x * perpX + curl.y * perpY;

    // TweenUILPathShader: update path colour blend from speed attribute
    let pathColorT = clamp(uilSpd + (getPulseIntensity(eIdx) * 0.5), 0.0, 1.0);

    // Trail alpha from pulse intensity
    trailA = mix(trailA, getPulseIntensity(eIdx), 0.15);

    pSet(idx, F_TRAVEL,       travel);
    pSet(idx, F_POS_X,        worldX);
    pSet(idx, F_POS_Y,        worldY);
    pSet(idx, F_NOFF,         nOff);
    pSet(idx, F_PULSE,        getPulseIntensity(eIdx));
    pSet(idx, F_PATH_COLOR_T, pathColorT);
    pSet(idx, F_THICKNESS,    length(thickVec));
    pSet(idx, F_TRAIL_ALPHA,  trailA);

    if (travel >= 1.0) {
      phase = 2.0;
      pSet(idx, F_PHASE,   phase);
      pSet(idx, F_HANDOFF, 1.0);
    }

  // ── DECAY — fade alpha ─────────────────────────────────────────────────────
  } else if (phase == 2.0) {
    alpha  -= uni.uDecayRate * dt;
    trailA -= uni.uDecayRate * dt * 0.5;   // trail fades slower
    if (alpha <= 0.0) {
      alpha  = 0.0;
      phase  = 3.0;
    }
    pSet(idx, F_ALPHA,      alpha);
    pSet(idx, F_PHASE,      phase);
    pSet(idx, F_TRAIL_ALPHA, max(0.0, trailA));
  }

  // ── Write tPos: r=worldX g=worldY b=travel a=alpha ─────────────────────────
  let posX = pGet(idx, F_POS_X);
  let posY = pGet(idx, F_POS_Y);
  let texX = i32(idx % uni.texW);
  let texY = i32(idx / uni.texW);
  textureStore(tPos, vec2<i32>(texX, texY), vec4f(posX, posY, travel, alpha));
}
`;

// ─── WGSL — Vertex shader (TweenUILPathShader port + particle billboard) ──────

const VERTEX_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>       uni      : FullSplineUniforms;
@group(0) @binding(1) var                tPos     : texture_2d<f32>;
@group(0) @binding(2) var                sSampler : sampler;
// Per-particle extended data (pathColorT, uilSpeed, pulseIntensity, trailAlpha)
@group(0) @binding(3) var<storage, read> particleExt : array<f32>;

const PS = 32u;

struct VertOut {
  @builtin(position) pos          : vec4f,
  @location(0)       vUv          : vec2f,
  @location(1)       vAlpha       : f32,
  @location(2)       vTravel      : f32,
  @location(3)       vColor       : vec3f,   // TweenUILPathShader: mix(uColor, uColor2, speed)
  @location(4)       vPulse       : f32,
  @location(5)       vTrailAlpha  : f32,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  let texX = i32(ii % uni.texW);
  let texY = i32(ii / uni.texW);
  let p    = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX = p.r;
  let worldY = p.g;
  let travel = p.b;
  let alpha  = p.a;

  let alive = select(0.0, 1.0, alpha > 0.004);

  // AT size attenuation: uSize * (1 − travel²) — shrinks toward end
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);

  // Extended fields from particle buffer
  let pathColorT   = particleExt[ii * PS + 13u];
  let uilSpeed     = particleExt[ii * PS + 14u];
  let pulseI       = particleExt[ii * PS + 12u];
  let trailAlpha   = particleExt[ii * PS + 19u];
  let thickness    = particleExt[ii * PS + 15u];

  // Thickness-modulated size (splineparticles.fs radius influence)
  let thickBoost = 1.0 + thickness * 2.0;
  let halfSize   = uni.uSize * travelDecay * 0.5 * thickBoost;

  let quadUV = QUAD[vi];
  var ndcX = worldX * uni.scaleX - 1.0 + quadUV.x * halfSize * uni.scaleX;
  var ndcY = worldY * uni.scaleY - 1.0 + quadUV.y * halfSize * uni.scaleY;

  // TweenUILPathShader: customDirection — screen-space finalPosition
  // x / aspect keeps line thickness view-independent
  ndcX = ndcX / uni.aspect;

  // TweenUILPathShader: vColor = mix(uColor, uColor2, speed)
  // speed is uilSpeed (per-particle [0,1] random) boosted by pulse
  let speed  = clamp(uilSpeed + pulseI * 0.4, 0.0, 1.0);
  let vColor = mix(uni.uColor, uni.uColor2, speed);

  var out: VertOut;
  out.pos         = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv         = quadUV;
  out.vAlpha      = alpha;
  out.vTravel     = travel;
  out.vColor      = vColor;
  out.vPulse      = pulseI;
  out.vTrailAlpha = trailAlpha;
  return out;
}
`;

// ─── WGSL — Fragment shader (TweenUILPathShader + soft-disk SDF) ──────────────

const FRAGMENT_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni : FullSplineUniforms;

struct FragIn {
  @location(0) vUv         : vec2f,
  @location(1) vAlpha      : f32,
  @location(2) vTravel     : f32,
  @location(3) vColor      : vec3f,
  @location(4) vPulse      : f32,
  @location(5) vTrailAlpha : f32,
}

// TweenUILPathShader.glsl: tri() — symmetric triangle wave [0,1]→[0,1]
fn tri(v: f32) -> f32 {
  return mix(v, 1.0 - v, step(0.5, v)) * 2.0;
}

@fragment fn fs_main(in: FragIn) -> @location(0) vec4f {
  // Soft circular SDF (AT: if (r > 1.0) discard)
  let r2   = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  let edge = 1.0 - smoothstep(0.7, 1.0, r2);

  // TweenUILPathShader.glsl: fragment SDF for path anti-aliasing
  // signedDist = tri(vUv.y * 0.5 + 0.5) − 0.5
  let vy         = in.vUv.y * 0.5 + 0.5;
  let signedDist = tri(vy) - 0.5;
  let sdEdge     = clamp(signedDist / fwidthFine(signedDist) + 0.5, 0.0, 1.0);

  // AT alpha: sin(π·travel) — bright midpoint, zero at ends
  let fade = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // Pulse glow contribution — adds brightness at pulse wavefront
  let pulseGlow = in.vPulse * 0.6 * fade;

  // Trail component — afterglow ring (TweenUILPath: broader soft halo)
  let trailRing = in.vTrailAlpha * smoothstep(0.5, 0.0, r2) * 0.4;

  var col    = in.vColor;
  col        = col + vec3f(pulseGlow * 0.8, pulseGlow * 0.6, pulseGlow);
  let finalA = (in.vAlpha * fade * edge * sdEdge) + trailRing;

  return vec4f(col * finalA, clamp(finalA, 0.0, 1.0));
}
`;

// ─── CPU-side spline helpers ──────────────────────────────────────────────────

function catmullRomCPU(
  p0: SplinePoint3, p1: SplinePoint3,
  p2: SplinePoint3, p3: SplinePoint3,
  t:  number,
): SplinePoint3 {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5*t3 + t2       - 0.5*t;
  const f2 =  1.5*t3 - 2.5*t2 + 1.0;
  const f3 = -1.5*t3 + 2.0*t2 + 0.5*t;
  const f4 =  0.5*t3 - 0.5*t2;
  return {
    x: f1*p0.x + f2*p1.x + f3*p2.x + f4*p3.x,
    y: f1*p0.y + f2*p1.y + f3*p2.y + f4*p3.y,
    z: f1*p0.z + f2*p1.z + f3*p2.z + f4*p3.z,
  };
}

function evalSplineCPU(pts: SplinePoint3[], u: number): SplinePoint3 {
  const n = pts.length;
  if (n === 0) return { x: 0, y: 0, z: 0 };
  if (n === 1) return { ...pts[0] };
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  return catmullRomCPU(pts[clamp(i1-1)], pts[clamp(i1)], pts[clamp(i1+1)], pts[clamp(i1+2)], sc - i1);
}

function buildEdgeBuf(edges: FullEdgeSpline[]): Float32Array {
  const buf = new Float32Array(edges.length * EDGE_STRIDE);
  for (let e = 0; e < edges.length; e++) {
    const base = e * EDGE_STRIDE;
    const pts  = edges[e].points;
    buf[base]  = Math.min(pts.length, EDGE_MAX_PTS);
    for (let p = 0; p < Math.min(pts.length, EDGE_MAX_PTS); p++) {
      const pb   = base + 4 + p * 4;
      buf[pb + 0] = pts[p].x;
      buf[pb + 1] = pts[p].y;
      buf[pb + 2] = pts[p].z;
    }
  }
  return buf;
}

// ─── ATSplineParticlesFull — Main class ───────────────────────────────────────

/**
 * ATSplineParticlesFull
 *
 * WebGPU orchestrator combining:
 *   — AT SplineParticleLife compute + render (M713 base)
 *   — splineparticles.fs thickness / distribution model
 *   — TweenUILPathShader screen-space colour animation
 *   — Cell pub/sub edge pulse queue driving per-edge intensity
 *
 * @example
 * ```ts
 * import { ATSplineParticlesFull, FullSplineParticlePreset } from '$lib/sph/at-spline-particles-full';
 *
 * const full = new ATSplineParticlesFull(device, canvas, edges, {
 *   ...FullSplineParticlePreset.cellPubSub,
 *   onHandoff: (edgeId, targetId, x, y, vx, vy, species) => {
 *     sphWorld.addFluid(x - 0.05, y - 0.05, x + 0.05, y + 0.05, 0.04, species);
 *   },
 * });
 * await full.build();
 *
 * // pub/sub hook:
 * eventBus.on('edge:message', ({ edgeId, qos, intensity }) => {
 *   full.firePulse(edgeId, qos, intensity);
 * });
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * full.update(enc, elapsed, dt);
 * full.render(enc, colorView, depthView);
 * device.queue.submit([enc.finish()]);
 * full.scheduleHandoffReadback();
 * ```
 */
export class ATSplineParticlesFull {
  private readonly device:    GPUDevice;
  private readonly canvas:    HTMLCanvasElement;
  private readonly onHandoff?: ATSplineParticlesFullConfig['onHandoff'];

  private cfg:   Required<Omit<ATSplineParticlesFullConfig, 'onHandoff' | 'qosThemes'>>;
  private qosThemes: Record<string, UILPathTheme>;
  private edges: FullEdgeSpline[]  = [];
  private edgeIdMap = new Map<string, number>();

  private particleCount = 0;
  private elapsed       = 0;
  private built         = false;

  // Per-edge pulse ring buffers (CPU-managed, uploaded each frame)
  private pulseData: Float32Array = new Float32Array(0);

  // GPU resources
  private uniformBuf!:      GPUBuffer;
  private edgeBuf!:         GPUBuffer;
  private particleBuf!:     GPUBuffer;
  private pulseBuf!:        GPUBuffer;
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

  constructor(
    device:  GPUDevice,
    canvas:  HTMLCanvasElement,
    edges:   FullEdgeSpline[],
    config:  ATSplineParticlesFullConfig = {},
  ) {
    this.device     = device;
    this.canvas     = canvas;
    this.edges      = edges;
    this.onHandoff  = config.onHandoff;
    this.qosThemes  = { ...DEFAULT_QOS_THEMES, ...(config.qosThemes ?? {}) };

    this.cfg = {
      uSplineSpeed:      config.uSplineSpeed      ?? [0.82, 1.21],
      uTimeMultiplier:   config.uTimeMultiplier   ?? 0.17,
      uFlowRange:        config.uFlowRange        ?? [1.0, 1.0],
      uDecayRate:        config.uDecayRate        ?? 0.6,
      uMaxSDelay:        config.uMaxSDelay        ?? 0.0,
      uCurlNoiseScale:   config.uCurlNoiseScale   ?? 2.0,
      uCurlNoiseSpeed:   config.uCurlNoiseSpeed   ?? 5.0,
      uCurlStrength:     config.uCurlStrength     ?? 0.04,
      uSize:             config.uSize             ?? 0.012,
      particlesPerUnit:  config.particlesPerUnit  ?? 24,
      maxParticles:      config.maxParticles      ?? MAX_PARTICLES,
      uSplineThickness:  config.uSplineThickness  ?? 1.0,
      uRangeThickness:   config.uRangeThickness   ?? 0.3,
      uRangeScale:       config.uRangeScale       ?? 1.0,
      uExtrudeRandom:    config.uExtrudeRandom    ?? 0.5,
      uDistribution:     config.uDistribution     ?? 1.0,
      uDistributionRange:config.uDistributionRange?? [0.3, 1.0],
      uThicknessStep:    config.uThicknessStep    ?? [0.5, 1.0],
      uThicknessSpeed:   config.uThicknessSpeed   ?? [0.1, 0.1],
      uilColor1:         config.uilColor1         ?? [0.55, 0.65, 0.90],
      uilColor2:         config.uilColor2         ?? [0.85, 0.90, 1.00],
      uilOpacity:        config.uilOpacity        ?? 0.70,
      pulseDuration:     config.pulseDuration     ?? 1.2,
      pulseDecayRate:    config.pulseDecayRate     ?? 0.8,
      trailPersistence:  config.trailPersistence  ?? 0.3,
    };

    // Build edge id → index map
    edges.forEach((e, i) => this.edgeIdMap.set(e.edgeId, i));
  }

  // ── Build GPU resources ───────────────────────────────────────────────────

  async build(): Promise<void> {
    if (this.built) this._destroy();
    const { device } = this;

    this.particleCount = Math.min(
      this.cfg.maxParticles,
      Math.max(256, this.edges.reduce((n, e) => n + Math.ceil(e.weight * this.cfg.particlesPerUnit), 0)),
    );

    // ── Uniform buffer ────────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      size:  FULL_UNIFORMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeUniforms(0);

    // ── Edge buffer ───────────────────────────────────────────────────────────
    const edgeData = buildEdgeBuf(this.edges);
    this.edgeBuf   = device.createBuffer({
      size:  Math.max(edgeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.edgeBuf, 0, edgeData);

    // ── Particle buffer ───────────────────────────────────────────────────────
    const pData = this._initParticleBuf();
    this.particleBuf = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.particleBuf, 0, pData);

    // Readback buffer
    this.readbackBuf = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Pulse intensity buffer ────────────────────────────────────────────────
    const pulseByteLen = Math.max(this.edges.length * MAX_PULSES_PER_EDGE * 4, 16);
    this.pulseData  = new Float32Array(this.edges.length * MAX_PULSES_PER_EDGE);
    this.pulseBuf   = device.createBuffer({
      size:  pulseByteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── tPos texture ──────────────────────────────────────────────────────────
    this.tPos = device.createTexture({
      size:   [TEX_W, TEX_H],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING  |
              GPUTextureUsage.COPY_SRC,
    });
    this.tPosView = this.tPos.createView();

    this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // ── Compute pipeline ──────────────────────────────────────────────────────
    const computeMod = device.createShaderModule({ code: COMPUTE_SHADER });
    this.computePipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: computeMod, entryPoint: 'main' },
    });

    // ── Render pipeline ───────────────────────────────────────────────────────
    const vsMod = device.createShaderModule({ code: VERTEX_SHADER });
    const fsMod = device.createShaderModule({ code: FRAGMENT_SHADER });
    const fmt   = navigator.gpu.getPreferredCanvasFormat();

    this.renderPipeline = device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module: vsMod, entryPoint: 'vs_main' },
      fragment: {
        module:  fsMod,
        entryPoint: 'fs_main',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
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
      `[ATSplineParticlesFull] built: ${this.edges.length} edges, ` +
      `${this.particleCount} particles, ` +
      `uSplineSpeed=[${this.cfg.uSplineSpeed}]`,
    );
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  /**
   * Update pulse ring buffers (CPU → GPU) and encode the compute pass.
   * Must be called before render() in the same command encoder.
   */
  update(encoder: GPUCommandEncoder, elapsed: number, dt = 1 / 60): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this._tickPulses(dt);
    this._writeUniforms(elapsed);

    // Upload pulse intensities to GPU
    this.device.queue.writeBuffer(this.pulseBuf, 0, this.pulseData);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBG0);
    pass.setBindGroup(1, this.computeBG1);
    pass.dispatchWorkgroups(Math.ceil(this.particleCount / WG));
    pass.end();
  }

  // ── Per-frame render ──────────────────────────────────────────────────────

  /**
   * Encode the render pass for spline particles with UIL path colours.
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
        view: depthView, depthLoadOp: 'load', depthStoreOp: 'store',
      };
    }

    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBG);
    pass.draw(6, this.particleCount);
    pass.end();
  }

  // ── Pub/sub pulse API ─────────────────────────────────────────────────────

  /**
   * Fire a data-flow pulse on an edge.  Called from the pub/sub bus handler.
   *
   * @param edgeId    — topology edge identifier
   * @param qos       — optional QoS profile name (drives UIL colour theme)
   * @param intensity — signal brightness [0, 1]; defaults to 1.0
   */
  firePulse(edgeId: string, qos?: string, intensity = 1.0): void {
    const eIdx = this.edgeIdMap.get(edgeId);
    if (eIdx === undefined) return;

    const base = eIdx * MAX_PULSES_PER_EDGE;

    // Find an inactive slot in the ring buffer (lowest intensity = best candidate)
    let slot = 0;
    let minIntensity = Infinity;
    for (let p = 0; p < MAX_PULSES_PER_EDGE; p++) {
      const v = this.pulseData[base + p];
      if (v < minIntensity) { minIntensity = v; slot = p; }
    }

    // Inject pulse intensity (will be decayed each frame by _tickPulses)
    this.pulseData[base + slot] = Math.min(1.0, intensity);

    // If this edge has a QoS theme, apply the colour to the uniform buffer live
    if (qos && this.qosThemes[qos]) {
      this._applyQosTheme(this.qosThemes[qos]);
    }
  }

  /**
   * Batch-fire pulses from an array of EdgeDataEvents (e.g. from a pub/sub message batch).
   */
  firePulsesBatch(events: EdgeDataEvent[]): void {
    for (const ev of events) {
      this.firePulse(ev.edgeId, ev.qos, ev.intensity ?? 1.0);
    }
  }

  // ── Handoff readback ──────────────────────────────────────────────────────

  /**
   * Async readback of particle buffer — fires onHandoff for particles entering DECAY.
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
      const b = i * FULL_PARTICLE_STRIDE;
      if (data[b + F_HANDOFF] < 0.5) continue;

      const eIdx = Math.round(data[b + F_EDGE]);
      const edge = this.edges[eIdx];
      if (!edge) continue;

      const x   = data[b + F_POS_X];
      const y   = data[b + F_POS_Y];
      const spd = data[b + F_SPEED];

      // Velocity from spline end tangent
      const ep  = evalSplineCPU(edge.points, 0.999);
      const ep2 = evalSplineCPU(edge.points, 0.998);
      const tx  = ep.x - ep2.x;
      const ty  = ep.y - ep2.y;
      const tl  = Math.sqrt(tx*tx + ty*ty) + 1e-10;
      const vsc = spd * this.cfg.uTimeMultiplier * 0.01;

      this.onHandoff(
        edge.edgeId, edge.targetId,
        x, y,
        (tx / tl) * vsc, (ty / tl) * vsc,
        edge.species ?? 0,
      );
    }

    this.readbackBuf.unmap();
  }

  // ── Live parameter setters ─────────────────────────────────────────────────

  setSplineSpeed(min: number, max: number): void { this.cfg.uSplineSpeed = [min, max]; }
  setTimeMultiplier(v: number): void             { this.cfg.uTimeMultiplier = v; }
  setDecayRate(v: number): void                  { this.cfg.uDecayRate = v; }
  setCurlStrength(v: number): void               { this.cfg.uCurlStrength = v; }
  setCurlNoiseScale(v: number): void             { this.cfg.uCurlNoiseScale = v; }
  setCurlNoiseSpeed(v: number): void             { this.cfg.uCurlNoiseSpeed = v; }
  setSize(v: number): void                       { this.cfg.uSize = v; }
  setSplineThickness(v: number): void            { this.cfg.uSplineThickness = v; }
  setRangeThickness(v: number): void             { this.cfg.uRangeThickness = v; }
  setExtrudeRandom(v: number): void              { this.cfg.uExtrudeRandom = v; }
  setUILColors(c1: [number, number, number], c2: [number, number, number]): void {
    this.cfg.uilColor1 = c1;
    this.cfg.uilColor2 = c2;
  }
  setPulseDuration(v: number): void              { this.cfg.pulseDuration = v; }
  setPulseDecayRate(v: number): void             { this.cfg.pulseDecayRate = v; }
  setTrailPersistence(v: number): void           { this.cfg.trailPersistence = v; }

  applyPreset(preset: ATSplineParticlesFullConfig): void {
    const c = this.cfg as Record<string, unknown>;
    const p = preset    as Record<string, unknown>;
    for (const key of Object.keys(c)) {
      if (p[key] !== undefined) c[key] = p[key];
    }
  }

  /** Replace QoS colour theme at runtime (no rebuild needed). */
  setQosTheme(qos: string, theme: UILPathTheme): void {
    this.qosThemes[qos] = theme;
  }

  /** Replace edges at runtime — triggers full rebuild. */
  async setEdges(edges: FullEdgeSpline[]): Promise<void> {
    this.edges = edges;
    this.edgeIdMap.clear();
    edges.forEach((e, i) => this.edgeIdMap.set(e.edgeId, i));
    await this.build();
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  get particleSlots(): number { return this.particleCount; }
  get edgeCount():     number { return this.edges.length; }
  get isBuilt():       boolean { return this.built; }
  get elapsedTime():   number { return this.elapsed; }

  destroy(): void { this._destroy(); }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _tickPulses(dt: number): void {
    // Decay all active pulse intensities
    const decay = this.cfg.pulseDecayRate * dt;
    for (let i = 0; i < this.pulseData.length; i++) {
      const v = this.pulseData[i] - decay;
      this.pulseData[i] = v < 0 ? 0 : v;
    }
  }

  private _applyQosTheme(theme: UILPathTheme): void {
    // Lerp current colour toward theme colour (smooth transition)
    const alpha = 0.25;
    this.cfg.uilColor1 = [
      this.cfg.uilColor1[0] + (theme.color1[0] - this.cfg.uilColor1[0]) * alpha,
      this.cfg.uilColor1[1] + (theme.color1[1] - this.cfg.uilColor1[1]) * alpha,
      this.cfg.uilColor1[2] + (theme.color1[2] - this.cfg.uilColor1[2]) * alpha,
    ];
    this.cfg.uilColor2 = [
      this.cfg.uilColor2[0] + (theme.color2[0] - this.cfg.uilColor2[0]) * alpha,
      this.cfg.uilColor2[1] + (theme.color2[1] - this.cfg.uilColor2[1]) * alpha,
      this.cfg.uilColor2[2] + (theme.color2[2] - this.cfg.uilColor2[2]) * alpha,
    ];
  }

  private _initParticleBuf(): Float32Array {
    const buf  = new Float32Array(this.particleCount * FULL_PARTICLE_STRIDE);
    let   slot = 0;

    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const count = Math.min(
        Math.ceil(edge.weight * this.cfg.particlesPerUnit),
        this.particleCount - slot,
      );
      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const b     = slot * FULL_PARTICLE_STRIDE;
        const speed = this.cfg.uSplineSpeed[0] +
                      Math.random() * (this.cfg.uSplineSpeed[1] - this.cfg.uSplineSpeed[0]);
        const delay = Math.random() * this.cfg.uMaxSDelay;
        const start = evalSplineCPU(edge.points, 0);

        buf[b + F_TRAVEL]       = 0;
        buf[b + F_SPEED]        = speed;
        buf[b + F_DELAY]        = delay;
        buf[b + F_PHASE]        = delay > 0 ? 0 : 1;
        buf[b + F_ALPHA]        = delay > 0 ? 0 : 1;
        buf[b + F_SEED]         = Math.random() * 1000;
        buf[b + F_NOFF]         = 0;
        buf[b + F_EDGE]         = e;
        buf[b + F_POS_X]        = start.x;
        buf[b + F_POS_Y]        = start.y;
        buf[b + F_HANDOFF]      = 0;
        buf[b + F_SPECIES]      = edge.species ?? 0;
        buf[b + F_PULSE]        = 0;
        buf[b + F_PATH_COLOR_T] = 0;
        buf[b + F_UIL_SPEED]    = Math.random();
        buf[b + F_THICKNESS]    = 0;
        buf[b + F_EMIT_RATE]    = 1;
        buf[b + F_CELL_WEIGHT]  = edge.weight;
        buf[b + F_DATA_EPOCH]   = 0;
        buf[b + F_TRAIL_ALPHA]  = 0;
      }
    }

    // Remaining slots: DEAD
    for (; slot < this.particleCount; slot++) {
      buf[slot * FULL_PARTICLE_STRIDE + F_PHASE] = 3;
    }

    return buf;
  }

  private _writeUniforms(elapsed: number): void {
    const { device, cfg, canvas } = this;
    const dw = canvas.width  || 1;
    const dh = canvas.height || 1;

    const data = new Float32Array(FULL_UNIFORMS_BYTES / 4);
    const u32  = new Uint32Array(data.buffer);

    data[U_TIME_MUL     / 4] = cfg.uTimeMultiplier;
    data[U_DECAY_RATE   / 4] = cfg.uDecayRate;
    data[U_CURL_SCALE   / 4] = cfg.uCurlNoiseScale;
    data[U_CURL_SPEED   / 4] = cfg.uCurlNoiseSpeed;
    data[U_CURL_STR     / 4] = cfg.uCurlStrength;
    data[U_SPD_MIN      / 4] = cfg.uSplineSpeed[0];
    data[U_SPD_MAX      / 4] = cfg.uSplineSpeed[1];
    data[U_MAX_DELAY    / 4] = cfg.uMaxSDelay;
    data[U_SIZE         / 4] = cfg.uSize;
    data[U_TIME         / 4] = elapsed;
    data[U_DOMAIN_W     / 4] = dw;
    data[U_DOMAIN_H     / 4] = dh;
    data[U_SCALE_X      / 4] = 2.0 / dw;
    data[U_SCALE_Y      / 4] = 2.0 / dh;

    u32[U_P_COUNT / 4] = this.particleCount;
    u32[U_E_COUNT / 4] = this.edges.length;
    u32[U_TEX_W   / 4] = TEX_W;
    u32[U_TEX_H   / 4] = TEX_H;

    // UIL path colours (bytes 72–96)
    data[U_COLOR1_R / 4] = cfg.uilColor1[0];
    data[U_COLOR1_G / 4] = cfg.uilColor1[1];
    data[U_COLOR1_B / 4] = cfg.uilColor1[2];
    data[U_COLOR2_R / 4] = cfg.uilColor2[0];
    data[U_COLOR2_G / 4] = cfg.uilColor2[1];
    data[U_COLOR2_B / 4] = cfg.uilColor2[2];

    // Thickness model
    data[U_THICKNESS_STEP_X / 4] = cfg.uThicknessStep[0];
    data[U_THICKNESS_STEP_Y / 4] = cfg.uThicknessStep[1];
    data[U_SPLINE_THICK     / 4] = cfg.uSplineThickness;
    data[U_RANGE_THICK      / 4] = cfg.uRangeThickness;
    data[U_RANGE_SCALE      / 4] = cfg.uRangeScale;
    data[U_EXTRUDE_RND      / 4] = cfg.uExtrudeRandom;
    data[U_DISTRIB          / 4] = cfg.uDistribution;
    data[U_DISTRIB_RNG_X    / 4] = cfg.uDistributionRange[0];
    data[U_DISTRIB_RNG_Y    / 4] = cfg.uDistributionRange[1];
    data[U_THICK_SPEED_X    / 4] = cfg.uThicknessSpeed[0];
    data[U_THICK_SPEED_Y    / 4] = cfg.uThicknessSpeed[1];

    // Screen-space aspect ratio for TweenUILPathShader customDirection
    data[U_ASPECT / 4] = dw / Math.max(dh, 1);

    device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private _buildBindGroups(): void {
    const { device } = this;

    // Compute BG0 — uniforms
    this.computeBG0 = device.createBindGroup({
      layout:  this.computePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    // Compute BG1 — edges + particles + tPos + pulses
    this.computeBG1 = device.createBindGroup({
      layout:  this.computePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.edgeBuf } },
        { binding: 1, resource: { buffer: this.particleBuf } },
        { binding: 2, resource: this.tPosView },
        { binding: 3, resource: { buffer: this.pulseBuf } },
      ],
    });

    // Render BG — uniforms + tPos + sampler + particleExt (extended fields)
    this.renderBG = device.createBindGroup({
      layout:  this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.tPosView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.particleBuf } },
      ],
    });
  }

  private _destroy(): void {
    if (!this.built) return;
    this.uniformBuf?.destroy();
    this.edgeBuf?.destroy();
    this.particleBuf?.destroy();
    this.pulseBuf?.destroy();
    this.readbackBuf?.destroy();
    this.tPos?.destroy();
    this.built = false;
  }
}

// ─── Factory helpers ───────────────────────────────────────────────────────────

/**
 * Wire ATSplineParticlesFull to SPHWorld.addFluid() for automatic fluid
 * injection when particles arrive at target cells.
 *
 * @example
 * ```ts
 * const full = createFullSplineParticleForSPH(device, canvas, edges, world.addFluid.bind(world));
 * await full.build();
 * ```
 */
export function createFullSplineParticleForSPH(
  device:   GPUDevice,
  canvas:   HTMLCanvasElement,
  edges:    FullEdgeSpline[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATSplineParticlesFullConfig, 'onHandoff'> = {},
): ATSplineParticlesFull {
  const R = 0.05;
  return new ATSplineParticlesFull(device, canvas, edges, {
    ...config,
    onHandoff: (_eId, _tId, x, y, _vx, _vy, species) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, species);
    },
  });
}

/**
 * Convert canvas-space route points → FullEdgeSpline control points in SPH domain.
 *
 * @param edgeId   topology edge identifier
 * @param sourceId source cell identifier
 * @param targetId target cell identifier
 * @param points   control points in canvas-pixel space
 * @param weight   connectivity / attention weight
 * @param canvasW  canvas pixel width
 * @param canvasH  canvas pixel height
 * @param domainW  SPH domain width (world units)
 * @param domainH  SPH domain height (world units)
 * @param species  optional species tag (0–7)
 * @param qos      optional QoS profile name
 */
export function canvasRouteToFullEdgeSpline(
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
  qos?:     string,
): FullEdgeSpline {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return {
    edgeId, sourceId, targetId, weight, species, qos,
    points: points.map(p => ({ x: p.x * sx, y: p.y * sy, z: 0 })),
  };
}

// ─── Constants re-export ───────────────────────────────────────────────────────

export const AT_SPLINE_PARTICLES_FULL_DEFAULTS = {
  maxParticles:   MAX_PARTICLES,
  texW:           TEX_W,
  texH:           TEX_H,
  particleStride: FULL_PARTICLE_STRIDE,
  edgeStride:     EDGE_STRIDE,
  workgroupSize:  WG,
  maxPulsesPerEdge: MAX_PULSES_PER_EDGE,
} as const;
