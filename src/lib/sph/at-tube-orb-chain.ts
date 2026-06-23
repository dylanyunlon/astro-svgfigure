/**
 * at-tube-orb-chain.ts — M842
 *
 * ATTubeOrbChain — Tube edge管道 + TubeOrb发光节点 + Chain链 + Work详情组件
 * ─────────────────────────────────────────────────────────────────────────────
 * Full WebGPU / WGSL port of Active Theory's Tube-Orb-Chain topology system:
 *
 *   TubeEdge        — cylindrical edge tube connecting two node positions,
 *                     matching TubeShader.glsl (fbr.vs + fbr.fs + refraction
 *                     blending, HSV shift, life/length-based fade transition).
 *
 *   TubeOrb         — glowing billboard sphere placed at node positions,
 *                     matching TubeOrbShader.glsl (tMap texture + uAlpha).
 *
 *   ChainSegment    — sinusoidal-offset chain links connecting adjacent nodes,
 *                     matching ChainShader.glsl (fbr.vs scroll-offset, FBR
 *                     material, tBaseColor + tRefraction, distance-attenuated
 *                     brightness, drawbuffer WorkRefraction).
 *
 *   WorkDetailSystem — particle + detail-cube Work panel renderer,
 *                     matching WorkTubeShader.glsl (noise-band tube),
 *                     WorkDetailCube.glsl (fresnel / refraction / normal cube),
 *                     WorkDetailParticleShader.glsl (GPU point-sprite tPos
 *                     readback with matcap + video blend).
 *
 * GLSL → WGSL translation map:
 *   texture2D(tPos, uv)          → textureLoad(tPos, texel, 0)
 *   varying / attribute          → @location(N) in/out
 *   gl_FragColor                 → @location(0) out : vec4f
 *   gl_Position                  → @builtin(position) pos : vec4f
 *   gl_PointSize / gl_PointCoord → instanced quad half-size / vUv
 *   fbr.vs / fbr.fs              → inlined FBR lighting helpers
 *   mix / fract / step / clamp   → same names in WGSL
 *   atan(y, x)                   → atan2(y, x)
 *   mod(x, y)                    → x % y
 *   sin / cos / sqrt / pow       → same
 *   rgb2hsv / hsv2rgb            → inlined helpers
 *   simplenoise (cnoise)         → inlined 3-D Perlin noise
 *   range / crange               → inlined saturate-range helpers
 *
 * GPU buffer layout:
 *   TubeEdge nodes:  array<TubeNodeData>  — start/end pos, life, length
 *   TubeOrb tPos:    rgba32float 2-D tex  — .r=x .g=y .b=scale .a=alpha
 *   Chain tPos:      rgba32float 2-D tex  — .r=x .g=y .b=scroll .a=alpha
 *   WorkDetail tPos: rgba32float 2-D tex  — .r=x .g=y .b=z .a=rand
 *
 * Integration:
 *   const chain = new ATTubeOrbChain(device, canvas, nodes, edges, config);
 *   await chain.build();
 *   // render loop:
 *   const enc = device.createCommandEncoder();
 *   chain.update(enc, elapsed, dt);
 *   chain.render(enc, colorView, depthView?);
 *   device.queue.submit([enc.finish()]);
 *
 * References:
 *   compiled.vs TubeShader.glsl            — edge tube GLSL source
 *   compiled.vs TubeOrbShader.glsl         — glowing-orb GLSL source
 *   compiled.vs ChainShader.glsl           — chain links GLSL source
 *   compiled.vs WorkTubeShader.glsl        — work-panel noise-tube GLSL source
 *   compiled.vs WorkDetailCube.glsl        — work-panel detail cube GLSL source
 *   compiled.vs WorkDetailParticleShader.glsl — work-panel particle GLSL source
 *   src/lib/sph/at-spline-particle.ts      — pattern reference
 */

// ─── Constants ────────────────────────────────────────────────────────────────

/** WebGPU workgroup size — ≤256 for broad device compatibility. */
const WG = 64 as const;

/** Max tube/orb/chain node count. */
const MAX_NODES  = 2048 as const;
/** Max edge count (tube + chain). */
const MAX_EDGES  = 4096 as const;
/** Max Work detail particles. */
const MAX_WORK_PARTICLES = 8192 as const;

/** tPos texture dimensions — W × H ≥ MAX_WORK_PARTICLES. */
const TEX_W = 128 as const;
const TEX_H = 64  as const;   // 128 × 64 = 8192

/** f32 fields per node in nodeStateBuf. */
const NODE_STRIDE = 12 as const;
/*
 * NodeState layout (12 f32):
 *   [0]  posX        — world X
 *   [1]  posY        — world Y
 *   [2]  posZ        — world Z
 *   [3]  scale       — orb display scale [0,1]
 *   [4]  alpha       — opacity [0,1]
 *   [5]  glowPhase   — oscillation phase seed
 *   [6]  scrollOff   — chain scroll accumulator
 *   [7]  active      — 1.0 = live, 0.0 = hidden
 *   [8]  colorH      — HSV hue override [0,1]
 *   [9]  colorS      — HSV saturation override [0,1]
 *   [10] colorV      — HSV value override [0,1]
 *   [11] _pad        — alignment
 */

/** f32 fields per edge in edgeStateBuf. */
const EDGE_STRIDE = 16 as const;
/*
 * EdgeState layout (16 f32):
 *   [0]  srcNode  — source node index (f32 cast)
 *   [1]  dstNode  — destination node index (f32 cast)
 *   [2]  life     — [0,1] tube growth progress
 *   [3]  length   — arc-length of edge (world units)
 *   [4]  alpha    — master edge alpha
 *   [5]  scroll   — chain scroll value (from uniforms.uScroll)
 *   [6]  weight   — connectivity weight
 *   [7]  type     — 0=tube, 1=chain, 2=both
 *   [8]  colorH   — hue shift override
 *   [9]  colorS   — saturation override
 *   [10] colorV   — value override
 *   [11..15] _pad — alignment
 */

/** f32 fields per Work particle. */
const WORK_PARTICLE_STRIDE = 16 as const;
/*
 * WorkParticle layout (16 f32):
 *   [0]  posX    — world X
 *   [1]  posY    — world Y
 *   [2]  posZ    — world Z
 *   [3]  alpha   — opacity [0,1]
 *   [4]  randX   — random.x (size bias)
 *   [5]  randY   — random.y
 *   [6]  randZ   — random.z (matcap anim)
 *   [7]  randW   — random.w (position)
 *   [8]  velX    — velocity X
 *   [9]  velY    — velocity Y
 *   [10] velZ    — velocity Z
 *   [11] life    — [0,1] particle life
 *   [12] nodeIdx — owning Work node index
 *   [13..15] _pad
 */

/** Uniform buffer byte sizes. */
const TUBE_UNIFORMS_BYTES  = 96 as const;
const WORK_UNIFORMS_BYTES  = 64 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** 3-D world-space node position. */
export interface TubeNode {
  nodeId:   string;
  x: number;
  y: number;
  z: number;
  /** Display scale [0,1], default 1. */
  scale?:   number;
  /** Optional HSV override [0,1] each. */
  colorHSV?: [number, number, number];
  /** 0=normal, 1=work-detail panel host. */
  type?:    0 | 1;
}

/** One directed edge between two TubeNodes. */
export interface TubeEdgeDef {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Connectivity weight (controls alpha + chain density). */
  weight?:  number;
  /** 0=tube only, 1=chain only, 2=tube+chain. */
  renderMode?: 0 | 1 | 2;
}

/** ATTubeOrbChain configuration — all fields optional. */
export interface ATTubeOrbChainConfig {
  /** Scroll value — drives ChainShader y-offset (AT: uScroll). */
  uScroll?:           number;
  /** Refraction blend strength (AT: uReflection.y). */
  uReflectionBlend?:  number;
  /** Normal-map distortion strength (AT: uReflection.x). */
  uNormalStrength?:   number;
  /** Orb global alpha. */
  uOrbAlpha?:         number;
  /** Chain sinusoidal amplitude (AT: cos(-y*0.4)*1.1 radius). */
  uChainAmplitude?:   number;
  /** Chain frequency (AT: -y*0.4). */
  uChainFrequency?:   number;
  /** Work-detail particle count per Work node. */
  workParticlesPerNode?: number;
  /** Work-detail particle size. */
  uWorkParticleSize?: number;
  /** Work-detail particle DPR. */
  uWorkDPR?:          number;
  /** Work-tube noise band time speed (AT: 0.3). */
  uWorkTubeTimeSpeed?: number;
  /** Tube life transition speed (AT: rangeTransition). */
  uTubeLifeSpeed?:    number;
  /** Called when a Work node's particle arrives (handoff). */
  onWorkHandoff?: (nodeId: string, x: number, y: number, z: number) => void;
}

/** Snapshot of a single TubeOrb node state (CPU introspection). */
export interface TubeOrbState {
  nodeId:  string;
  x: number; y: number; z: number;
  scale:   number;
  alpha:   number;
  colorH:  number;
  colorS:  number;
  colorV:  number;
  active:  boolean;
}

// ─── Preset bundles ────────────────────────────────────────────────────────────

/**
 * TubeOrbChainPreset
 *
 * Named configuration bundles matching AT UIL parameter presets.
 *
 * @example
 * ```ts
 * const chain = new ATTubeOrbChain(device, canvas, nodes, edges, {
 *   ...TubeOrbChainPreset.network,
 *   onWorkHandoff: (nodeId, x, y, z) => { … },
 * });
 * ```
 */
export const TubeOrbChainPreset = {
  /** Default — balanced, matches AT compiled.vs defaults. */
  default: {
    uScroll:              0.0,
    uReflectionBlend:     0.5,
    uNormalStrength:      1.0,
    uOrbAlpha:            1.0,
    uChainAmplitude:      1.1,
    uChainFrequency:      0.4,
    workParticlesPerNode: 128,
    uWorkParticleSize:    0.03,
    uWorkDPR:             1.0,
    uWorkTubeTimeSpeed:   0.3,
    uTubeLifeSpeed:       0.01,
  } satisfies ATTubeOrbChainConfig,

  /** Network graph — tight chains, bright orbs, dense particles. */
  network: {
    uScroll:              0.0,
    uReflectionBlend:     0.8,
    uNormalStrength:      1.5,
    uOrbAlpha:            1.0,
    uChainAmplitude:      0.8,
    uChainFrequency:      0.6,
    workParticlesPerNode: 256,
    uWorkParticleSize:    0.02,
    uWorkDPR:             2.0,
    uWorkTubeTimeSpeed:   0.5,
    uTubeLifeSpeed:       0.008,
  } satisfies ATTubeOrbChainConfig,

  /** Organic — wide sinusoidal chains, soft glow, slow scroll. */
  organic: {
    uScroll:              0.1,
    uReflectionBlend:     0.3,
    uNormalStrength:      0.7,
    uOrbAlpha:            0.75,
    uChainAmplitude:      2.0,
    uChainFrequency:      0.3,
    workParticlesPerNode: 64,
    uWorkParticleSize:    0.05,
    uWorkDPR:             1.0,
    uWorkTubeTimeSpeed:   0.15,
    uTubeLifeSpeed:       0.015,
  } satisfies ATTubeOrbChainConfig,

  /** Minimal — thin tubes, invisible chain, subtle orbs. */
  minimal: {
    uScroll:              0.0,
    uReflectionBlend:     0.1,
    uNormalStrength:      0.3,
    uOrbAlpha:            0.4,
    uChainAmplitude:      0.4,
    uChainFrequency:      0.5,
    workParticlesPerNode: 32,
    uWorkParticleSize:    0.01,
    uWorkDPR:             1.0,
    uWorkTubeTimeSpeed:   0.2,
    uTubeLifeSpeed:       0.02,
  } satisfies ATTubeOrbChainConfig,

  /** Portal — high refraction, maximum particles, dramatic. */
  portal: {
    uScroll:              0.5,
    uReflectionBlend:     1.2,
    uNormalStrength:      2.5,
    uOrbAlpha:            1.0,
    uChainAmplitude:      3.0,
    uChainFrequency:      0.2,
    workParticlesPerNode: 512,
    uWorkParticleSize:    0.04,
    uWorkDPR:             2.0,
    uWorkTubeTimeSpeed:   0.6,
    uTubeLifeSpeed:       0.005,
  } satisfies ATTubeOrbChainConfig,
} as const;

// ─── WGSL — shared noise helpers ──────────────────────────────────────────────
// Ported from AT simplenoise.glsl

const NOISE_WGSL = /* wgsl */`
// ── hash33 (simplenoise.glsl) ─────────────────────────────────────────────────
fn hash3(p: vec3f) -> vec3f {
  let q = vec3f(
    dot(p, vec3f(127.1, 311.7,  74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453123);
}

// ── 3-D gradient (Perlin) noise ───────────────────────────────────────────────
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

// ── cnoise (alias) ────────────────────────────────────────────────────────────
fn cnoise(p: vec3f) -> f32 { return noise3(p); }

// ── getNoise — 2-D grain (AT GlobalComposite blendOverlay noise) ──────────────
fn getNoise2(uv: vec2f, t: f32) -> f32 {
  return fract(sin(dot(uv + fract(t * 0.1), vec2f(12.9898, 78.233))) * 43758.5453123);
}
`;

// ─── WGSL — range/crange helpers ─────────────────────────────────────────────
// Ported from AT range.glsl

const RANGE_WGSL = /* wgsl */`
fn crange1(v: f32, lo: f32, hi: f32, outLo: f32, outHi: f32) -> f32 {
  return outLo + (outHi - outLo) * clamp((v - lo) / (hi - lo + 1e-8), 0.0, 1.0);
}

// rangeTransition — AT tube life/length based SDF fade (TubeShader.glsl line ~270-272)
// b = crange(life, 0.1, 0.2, 0.0, 1.0)
// tb = b - length * 0.01 (simplified port)
fn rangeTransition(b: f32, len: f32, threshold: f32) -> f32 {
  return b - len * threshold;
}
`;

// ─── WGSL — HSV helpers ───────────────────────────────────────────────────────
// Ported from AT rgb2hsv.fs

const HSV_WGSL = /* wgsl */`
fn rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
  let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3f) -> vec3f {
  let K = vec4f(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}
`;

// ─── WGSL — FBR lighting stub ─────────────────────────────────────────────────
// Inlined port of AT fbr.vs / fbr.fs concept: a simple rim+diffuse lighting
// model that matches the visual character of AT's FBR system.

const FBR_WGSL = /* wgsl */`
// ── Fresnel-like rim term (AT fbr.fs approximation) ──────────────────────────
fn fresnelRim(viewDir: vec3f, normal: vec3f, power: f32) -> f32 {
  return pow(1.0 - abs(dot(normalize(viewDir), normalize(normal))), power);
}

// ── getFBR — simplified AT FBR ambient+rim color ─────────────────────────────
// baseColor: material/texture color, uv: surface UV, returns lit color
fn getFBR(baseColor: vec3f, worldNormal: vec3f, viewDir: vec3f, t: f32, uv: vec2f) -> vec3f {
  let rim  = fresnelRim(viewDir, worldNormal, 2.0);
  let diff = max(0.0, dot(normalize(worldNormal), vec3f(0.0, 1.0, 0.0))) * 0.5 + 0.5;
  var col  = baseColor * diff;
  col     += vec3f(0.4, 0.6, 1.0) * rim * 0.4;
  col     += sin(uv.x * 5.0 + t * 0.2) * 0.02;
  return col;
}
`;

// ─── WGSL — Uniforms struct (Tube + Chain + Orb) ──────────────────────────────

const TUBE_UNIFORMS_WGSL = /* wgsl */`
struct TubeUniforms {
  time              : f32,
  uScroll           : f32,
  uReflectionBlend  : f32,
  uNormalStrength   : f32,
  uOrbAlpha         : f32,
  uChainAmplitude   : f32,
  uChainFrequency   : f32,
  uTubeLifeSpeed    : f32,

  // NDC projection
  scaleX            : f32,
  scaleY            : f32,
  scaleZ            : f32,
  canvasW           : f32,
  canvasH           : f32,

  nodeCount         : u32,
  edgeCount         : u32,
  texW              : u32,
  texH              : u32,

  _pad0             : u32,
  _pad1             : u32,
  _pad2             : u32,
  _pad3             : u32,
  _pad4             : u32,
  _pad5             : u32,
}
`;

// Byte offsets into TubeUniforms (4 bytes each f32)
const TU_TIME               = 0;
const TU_SCROLL             = 4;
const TU_REFLECTION_BLEND   = 8;
const TU_NORMAL_STRENGTH    = 12;
const TU_ORB_ALPHA          = 16;
const TU_CHAIN_AMP          = 20;
const TU_CHAIN_FREQ         = 24;
const TU_TUBE_LIFE_SPEED    = 28;
const TU_SCALE_X            = 32;
const TU_SCALE_Y            = 36;
const TU_SCALE_Z            = 40;
const TU_CANVAS_W           = 44;
const TU_CANVAS_H           = 48;
const TU_NODE_COUNT         = 52;  // u32
const TU_EDGE_COUNT         = 56;  // u32
const TU_TEX_W              = 60;  // u32
const TU_TEX_H              = 64;  // u32

const WORK_UNIFORMS_WGSL = /* wgsl */`
struct WorkUniforms {
  time              : f32,
  uWorkTubeTimeSpeed: f32,
  uWorkParticleSize : f32,
  uWorkDPR          : f32,
  uWorkSizeBias     : f32,
  scaleX            : f32,
  scaleY            : f32,
  scaleZ            : f32,
  particleCount     : u32,
  texW              : u32,
  texH              : u32,
  _pad0             : u32,
  _pad1             : u32,
  _pad2             : u32,
  _pad3             : u32,
  _pad4             : u32,
}
`;

const WU_TIME              = 0;
const WU_TUBE_TIME_SPEED   = 4;
const WU_PARTICLE_SIZE     = 8;
const WU_DPR               = 12;
const WU_SIZE_BIAS         = 16;
const WU_SCALE_X           = 20;
const WU_SCALE_Y           = 24;
const WU_SCALE_Z           = 28;
const WU_PARTICLE_COUNT    = 32;  // u32
const WU_TEX_W             = 36;  // u32
const WU_TEX_H             = 40;  // u32

// ─── WGSL — Node compute shader ───────────────────────────────────────────────
// Updates TubeOrb glow oscillation per node.

const NODE_COMPUTE_SHADER = /* wgsl */`
${TUBE_UNIFORMS_WGSL}
${NOISE_WGSL}
${HSV_WGSL}

@group(0) @binding(0) var<uniform>             uni      : TubeUniforms;
@group(1) @binding(0) var<storage, read_write> nodeBuf  : array<f32>;
@group(1) @binding(1) var                      tOrbPos  : texture_storage_2d<rgba32float, write>;

const NS = ${NODE_STRIDE}u;

fn nGet(idx: u32, f: u32) -> f32 { return nodeBuf[idx * NS + f]; }
fn nSet(idx: u32, f: u32, v: f32) { nodeBuf[idx * NS + f] = v; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.nodeCount) { return; }

  let t = uni.time;

  var px   = nGet(idx, 0u);
  var py   = nGet(idx, 1u);
  var pz   = nGet(idx, 2u);
  var sc   = nGet(idx, 3u);
  var alph = nGet(idx, 4u);
  let gph  = nGet(idx, 5u);
  let act  = nGet(idx, 7u);

  // Orb pulsation: AT TubeOrbShader — subtle scale oscillation
  let pulse = 1.0 + sin(t * 1.5 + gph * 3.14159) * 0.06 * act;
  sc = sc * pulse;

  // Alpha flicker from noise (AT: TubeOrbShader uAlpha)
  let nv = 0.5 + cnoise(vec3f(px * 0.5, py * 0.5, t * 0.3 + gph)) * 0.1;
  alph = clamp(alph * nv * act + act * 0.05, 0.0, 1.0);

  nSet(idx, 3u, sc);
  nSet(idx, 4u, alph);

  // Write to tOrbPos texture (.r=x .g=y .b=scale .a=alpha)
  let tx = i32(idx % uni.texW);
  let ty = i32(idx / uni.texW);
  textureStore(tOrbPos, vec2<i32>(tx, ty), vec4f(px, py, sc, alph));
}
`;

// ─── WGSL — Edge/Chain compute shader ────────────────────────────────────────
// Updates TubeEdge life accumulation and Chain scroll per edge.

const EDGE_COMPUTE_SHADER = /* wgsl */`
${TUBE_UNIFORMS_WGSL}
${RANGE_WGSL}

@group(0) @binding(0) var<uniform>             uni      : TubeUniforms;
@group(1) @binding(0) var<storage, read>       nodeBuf  : array<f32>;
@group(1) @binding(1) var<storage, read_write> edgeBuf  : array<f32>;
@group(1) @binding(2) var                      tChainPos: texture_storage_2d<rgba32float, write>;

const NS  = ${NODE_STRIDE}u;
const ES  = ${EDGE_STRIDE}u;

fn nGet(idx: u32, f: u32) -> f32 { return nodeBuf[idx * NS + f]; }
fn eGet(idx: u32, f: u32) -> f32 { return edgeBuf[idx * ES + f]; }
fn eSet(idx: u32, f: u32, v: f32) { edgeBuf[idx * ES + f] = v; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.edgeCount) { return; }

  let dt = 1.0 / 60.0;

  let srcIdx = u32(eGet(idx, 0u));
  let dstIdx = u32(eGet(idx, 1u));
  var life   = eGet(idx, 2u);
  let len    = eGet(idx, 3u);
  var alpha  = eGet(idx, 4u);
  var scroll = eGet(idx, 5u);
  let weight = eGet(idx, 6u);

  // Grow tube life (AT: TubeShader rangeTransition life progression)
  life   = min(1.0, life + uni.uTubeLifeSpeed * dt * 60.0);
  scroll = fract(scroll + dt * 0.5 * (0.5 + weight * 0.5));

  // Edge alpha from src/dst node activity
  let srcAct = nGet(srcIdx, 7u);
  let dstAct = nGet(dstIdx, 7u);
  alpha = clamp(srcAct * dstAct * weight, 0.0, 1.0);

  eSet(idx, 2u, life);
  eSet(idx, 4u, alpha);
  eSet(idx, 5u, scroll);

  // Mid-point for chain tPos texture (AT: ChainShader world position sample)
  let sx = nGet(srcIdx, 0u); let sy = nGet(srcIdx, 1u); let sz = nGet(srcIdx, 2u);
  let dx = nGet(dstIdx, 0u); let dy = nGet(dstIdx, 1u); let dz = nGet(dstIdx, 2u);
  let mx = (sx + dx) * 0.5;
  let my = (sy + dy) * 0.5 - 17.0 * uni.uScroll;
  let mz = (sz + dz) * 0.5;

  // ChainShader sinusoidal offset: pos.x -= cos(-pos.y * freq) * amp
  let chainX = mx - cos(-my * uni.uChainFrequency) * uni.uChainAmplitude;
  let chainZ = mz - sin(-my * uni.uChainFrequency) * uni.uChainAmplitude;

  let tx = i32(idx % uni.texW);
  let ty = i32(idx / uni.texW);
  textureStore(tChainPos, vec2<i32>(tx, ty), vec4f(chainX, my, alpha, scroll));
}
`;

// ─── WGSL — Work detail particle compute ─────────────────────────────────────
// Matches WorkDetailParticleShader.glsl: GPGPU particle positions, matcap blend.

const WORK_PARTICLE_COMPUTE = /* wgsl */`
${WORK_UNIFORMS_WGSL}
${NOISE_WGSL}

@group(0) @binding(0) var<uniform>             uni       : WorkUniforms;
@group(1) @binding(0) var<storage, read_write> partBuf   : array<f32>;
@group(1) @binding(1) var                      tWorkPos  : texture_storage_2d<rgba32float, write>;

const PS = ${WORK_PARTICLE_STRIDE}u;

fn pGet(idx: u32, f: u32) -> f32 { return partBuf[idx * PS + f]; }
fn pSet(idx: u32, f: u32, v: f32) { partBuf[idx * PS + f] = v; }

fn rng(s: f32, salt: f32) -> f32 {
  return fract(sin(s * 127.1 + salt * 311.7) * 43758.5453);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let dt = 1.0 / 60.0;
  let t  = uni.time;

  var px   = pGet(idx, 0u);
  var py   = pGet(idx, 1u);
  var pz   = pGet(idx, 2u);
  var alph = pGet(idx, 3u);
  let rx   = pGet(idx, 4u);
  let ry   = pGet(idx, 5u);
  let rz   = pGet(idx, 6u);
  let rw   = pGet(idx, 7u);
  var vx   = pGet(idx, 8u);
  var vy   = pGet(idx, 9u);
  var vz   = pGet(idx, 10u);
  var life = pGet(idx, 11u);

  // Advance life
  life -= dt * (0.3 + rx * 0.4);
  if (life <= 0.0) {
    // Respawn around node center (stored in vel during init as bias position)
    let s   = f32(idx) * 1.618034 + t;
    let off = 1.5;
    px   = rng(s, 0.0) * off * 2.0 - off;
    py   = rng(s, 1.0) * off * 2.0 - off;
    pz   = rng(s, 2.0) * off * 2.0 - off;
    vx   = rng(s, 3.0) * 0.02 - 0.01;
    vy   = rng(s, 4.0) * 0.02 + 0.005;
    vz   = rng(s, 5.0) * 0.02 - 0.01;
    life = 0.5 + rng(s, 6.0) * 1.5;
    alph = 0.0;
  }

  // Integrate position
  // Noise drift (AT WorkDetailParticleShader: pos.y += offset based on noise)
  let noiseCoord = vec3f(px * 0.3 + rz, py * 0.3, t * uni.uWorkTubeTimeSpeed + rw * 10.0);
  let nv = noise3(noiseCoord) * 0.003;
  px += vx + nv;
  py += vy;
  pz += vz + nv;

  // Fade in / out
  alph = clamp(alph + dt * 2.0, 0.0, min(1.0, life * 2.0));

  pSet(idx, 0u,  px);
  pSet(idx, 1u,  py);
  pSet(idx, 2u,  pz);
  pSet(idx, 3u,  alph);
  pSet(idx, 8u,  vx);
  pSet(idx, 9u,  vy);
  pSet(idx, 10u, vz);
  pSet(idx, 11u, life);

  // Write tWorkPos (.r=x .g=y .b=z .a=alpha)
  let tx = i32(idx % uni.texW);
  let ty = i32(idx / uni.texW);
  textureStore(tWorkPos, vec2<i32>(tx, ty), vec4f(px, py, pz, alph));
}
`;

// ─── WGSL — TubeOrb vertex + fragment ────────────────────────────────────────
// Matches TubeOrbShader.glsl: billboard sphere with tMap texture + uAlpha.

const TUBE_ORB_VERT_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni    : TubeUniforms;
@group(0) @binding(1) var          tOrbPos: texture_2d<f32>;
@group(0) @binding(2) var          samp   : sampler;

struct OrbVertOut {
  @builtin(position) pos    : vec4f,
  @location(0)       vUv    : vec2f,
  @location(1)       vAlpha : f32,
  @location(2)       vPos   : vec3f,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(1.0, 1.0),
  vec2f(-1.0,-1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

@vertex fn vs_orb(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> OrbVertOut {
  let tx   = i32(ii % uni.texW);
  let ty   = i32(ii / uni.texW);
  let p    = textureLoad(tOrbPos, vec2<i32>(tx, ty), 0);

  let px    = p.r;
  let py    = p.g;
  let scale = p.b;
  let alpha = p.a * uni.uOrbAlpha;

  let alive = select(0.0, 1.0, alpha > 0.001);
  let halfS = scale * 0.08;

  let qv   = QUAD[vi];
  let ndcX = px * uni.scaleX - 1.0 + qv.x * halfS * uni.scaleX;
  let ndcY = py * uni.scaleY - 1.0 + qv.y * halfS * uni.scaleY;

  var out: OrbVertOut;
  out.pos    = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv    = (qv + 1.0) * 0.5;
  out.vAlpha = alpha;
  out.vPos   = vec3f(px, py, 0.0);
  return out;
}
`;

const TUBE_ORB_FRAG_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}
${HSV_WGSL}

@group(0) @binding(0) var<uniform> uni  : TubeUniforms;
@group(0) @binding(3) var          tMap : texture_2d<f32>;
@group(0) @binding(4) var          samp2: sampler;

struct OrbFragIn {
  @location(0) vUv    : vec2f,
  @location(1) vAlpha : f32,
  @location(2) vPos   : vec3f,
}

@fragment fn fs_orb(in: OrbFragIn) -> @location(0) vec4f {
  let uv    = in.vUv;
  // AT TubeOrbShader: sample tMap (matcap/env), blend with solid white
  var color = textureSample(tMap, samp2, uv).rgb;
  // Soft circular SDF discard (billboard)
  let r2    = dot(uv - 0.5, uv - 0.5) * 4.0;
  if (r2 > 1.0) { discard; }
  let edge  = 1.0 - smoothstep(0.5, 1.0, r2);
  // HSV glow pulse (AT: orb emissive color)
  var hsv   = rgb2hsv(color);
  hsv.x    += sin(uni.time * 0.8 + in.vPos.x * 0.5) * 0.05;
  hsv.y    *= 0.6;
  hsv.z    *= 1.4;
  color     = hsv2rgb(clamp(hsv, vec3f(0.0), vec3f(1.0)));
  // Core glow: add bright center
  color    += vec3f(0.9, 0.95, 1.0) * smoothstep(0.5, 0.0, r2) * 0.6;
  let a     = in.vAlpha * edge;
  return vec4f(color * a, a);
}
`;

// ─── WGSL — TubeEdge vertex + fragment ───────────────────────────────────────
// Matches TubeShader.glsl: FBR tube with refraction, HSV shift, life transition.

const TUBE_EDGE_VERT_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>       uni    : TubeUniforms;
@group(0) @binding(5) var<storage, read> edgeBuf: array<f32>;
@group(0) @binding(6) var<storage, read> nodeBuf: array<f32>;

struct TubeVertOut {
  @builtin(position) pos      : vec4f,
  @location(0)       vUv      : vec2f,
  @location(1)       vLife    : f32,
  @location(2)       vLength  : f32,
  @location(3)       vAlpha   : f32,
  @location(4)       vWorldPos: vec3f,
  @location(5)       vNormal  : vec3f,
}

const ES  = ${EDGE_STRIDE}u;
const NS  = ${NODE_STRIDE}u;
const PI  = 3.14159265358979;
const SEG = 16u;   // tube radial segments

fn eGet(ei: u32, f: u32) -> f32 { return edgeBuf[ei * ES + f]; }
fn nGet(ni: u32, f: u32) -> f32 { return nodeBuf[ni * NS + f]; }

@vertex fn vs_tube(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,   // ii = edge index
) -> TubeVertOut {
  let ei     = ii;
  let life   = eGet(ei, 2u);
  let len    = eGet(ei, 3u);
  let alpha  = eGet(ei, 4u);
  let srcIdx = u32(eGet(ei, 0u));
  let dstIdx = u32(eGet(ei, 1u));

  let sx = nGet(srcIdx, 0u); let sy = nGet(srcIdx, 1u); let sz = nGet(srcIdx, 2u);
  let dx = nGet(dstIdx, 0u); let dy = nGet(dstIdx, 1u); let dz = nGet(dstIdx, 2u);

  // Procedural tube: vi encodes [ring, seg] along edge
  let totalVerts = SEG * 2u * 3u;   // tube strip quads, 2 triangles per quad ring
  let ring       = vi / (SEG * 3u);
  let segIdx     = (vi % (SEG * 3u)) / 3u;
  let t          = f32(ring) / f32(SEG);
  let angle      = f32(segIdx) * (2.0 * PI) / f32(SEG);

  // Lerp along axis
  let axX  = sx + (dx - sx) * t;
  let axY  = sy + (dy - sy) * t;
  let axZ  = sz + (dz - sz) * t;

  // Radial normal in XZ plane (simplified; for full tangent-frame use Frenet)
  let nx = cos(angle) * 0.04;
  let nz = sin(angle) * 0.04;

  let px  = axX + nx;
  let py  = axY;
  let pz  = axZ + nz;

  var out: TubeVertOut;
  out.pos       = vec4f(px * uni.scaleX - 1.0, py * uni.scaleY - 1.0, pz * uni.scaleZ, 1.0);
  out.vUv       = vec2f(t, f32(segIdx) / f32(SEG));
  out.vLife     = life;
  out.vLength   = len;
  out.vAlpha    = alpha;
  out.vWorldPos = vec3f(px, py, pz);
  out.vNormal   = normalize(vec3f(nx, 0.0, nz));
  return out;
}
`;

const TUBE_EDGE_FRAG_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}
${RANGE_WGSL}
${HSV_WGSL}
${FBR_WGSL}

@group(0) @binding(0)  var<uniform> uni        : TubeUniforms;
@group(0) @binding(7)  var          tTubeColor : texture_2d<f32>;
@group(0) @binding(8)  var          tRefraction: texture_2d<f32>;
@group(0) @binding(9)  var          samp       : sampler;

struct TubeFragIn {
  @builtin(position) fragCoord : vec4f,
  @location(0)       vUv       : vec2f,
  @location(1)       vLife     : f32,
  @location(2)       vLength   : f32,
  @location(3)       vAlpha    : f32,
  @location(4)       vWorldPos : vec3f,
  @location(5)       vNormal   : vec3f,
}

@fragment fn fs_tube(in: TubeFragIn) -> @location(0) vec4f {
  // AT TubeShader.glsl life/length-based discard transition
  let b  = crange1(in.vLife, 0.1, 0.2, 0.0, 1.0);
  let tb = rangeTransition(b, in.vLength, 0.01);
  if (tb < 0.5) { discard; }

  // Base color from tube color texture
  var myColor = textureSample(tTubeColor, samp, in.vUv * vec2f(5.0, 1.0)).rgb;

  // FBR lighting stub
  let viewDir = normalize(vec3f(0.0, 0.0, 1.0) - in.vWorldPos);
  var color   = getFBR(vec3f(0.2), in.vNormal, viewDir, uni.time, in.vUv);

  // Refraction sample (AT: ruv += vNormal.xy * 0.1)
  var ruv = in.fragCoord.xy / vec2f(uni.canvasW, uni.canvasH);
  ruv    += in.vNormal.xy * 0.1;
  color  += textureSample(tRefraction, samp, ruv).rgb;

  // AT blendOverlay approximation with myColor
  color = mix(color, myColor, 0.4);
  color = mix(myColor, color, 1.0 - step(in.vUv.x, 0.98) * (1.0 - step(in.vUv.x, 0.9)));

  // HSV hue shift: color.x -= vLength * 0.2 + sin(time * 0.2 + ...) * 0.1
  var hsv = rgb2hsv(color);
  hsv.x  -= in.vLength * 0.2 + sin(uni.time * 0.2 + length(in.vWorldPos) * 0.1) * 0.1;
  hsv.y  *= 0.7;
  color   = hsv2rgb(hsv);

  // AT: color += sin(-time*6 + vLength*4 + ...) * 0.1
  color += sin(-uni.time * 6.0 + in.vLength * 4.0 + length(in.vWorldPos)) * 0.1;
  color *= smoothstep(0.0, 0.3, in.vLife);
  let pw = mix(1.0, 2.0, in.vLength);
  color  = pow(max(color, vec3f(0.0)), vec3f(pw));

  let a = in.vAlpha * smoothstep(0.0, 0.3, in.vLife);
  return vec4f(color, a);
}
`;

// ─── WGSL — ChainSegment vertex + fragment ────────────────────────────────────
// Matches ChainShader.glsl: sinusoidal y-offset, FBR mat, tBaseColor + tRefraction.

const CHAIN_VERT_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>       uni      : TubeUniforms;
@group(0) @binding(5) var<storage, read> edgeBuf  : array<f32>;
@group(0) @binding(6) var<storage, read> nodeBuf  : array<f32>;

struct ChainVertOut {
  @builtin(position) pos      : vec4f,
  @location(0)       vUv      : vec2f,
  @location(1)       vDist    : f32,
  @location(2)       vAlpha   : f32,
  @location(3)       vWorldPos: vec3f,
  @location(4)       vNormal  : vec3f,
}

const ES  = ${EDGE_STRIDE}u;
const NS  = ${NODE_STRIDE}u;
const PI  = 3.14159265358979;
const LNKS = 32u;   // chain links per edge

fn eGet(ei: u32, f: u32) -> f32 { return edgeBuf[ei * ES + f]; }
fn nGet(ni: u32, f: u32) -> f32 { return nodeBuf[ni * NS + f]; }

@vertex fn vs_chain(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> ChainVertOut {
  let ei     = ii;
  let alpha  = eGet(ei, 4u);
  let scroll = eGet(ei, 5u);
  let srcIdx = u32(eGet(ei, 0u));
  let dstIdx = u32(eGet(ei, 1u));

  let sx = nGet(srcIdx, 0u); let sy = nGet(srcIdx, 1u); let sz = nGet(srcIdx, 2u);
  let dx = nGet(dstIdx, 0u); let dy = nGet(dstIdx, 1u); let dz = nGet(dstIdx, 2u);

  let totalQuads = LNKS;
  let qIdx  = vi / 6u;
  let corner= vi % 6u;
  let t0    = f32(qIdx)      / f32(LNKS);
  let t1    = f32(qIdx + 1u) / f32(LNKS);

  let pick  = select(t0, t1, corner >= 3u);

  // Lerp chain axis with AT ChainShader scroll offset
  var pos = vec3f(
    sx + (dx - sx) * pick,
    sy + (dy - sy) * pick - 17.0 * uni.uScroll,
    sz + (dz - sz) * pick,
  );

  // AT ChainShader sinusoidal offset
  pos.x -= cos(-pos.y * uni.uChainFrequency) * uni.uChainAmplitude;
  pos.z -= sin(-pos.y * uni.uChainFrequency) * uni.uChainAmplitude;

  // Billboard width for chain quad (thin strip)
  let stripW = 0.02;
  let even   = (corner == 0u) || (corner == 2u) || (corner == 3u);
  let sideY  = select(-stripW, stripW, even);
  pos.y     += sideY;

  let ndcX = pos.x * uni.scaleX - 1.0;
  let ndcY = pos.y * uni.scaleY - 1.0;

  // Distance from camera (AT: vDist = length(vWorldPos - cameraPosition))
  // Approximate with distance to scene center
  let dist = length(vec3f(pos.x, pos.y * 0.1, pos.z));

  var out: ChainVertOut;
  out.pos       = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.vUv       = vec2f(pick, select(0.0, 1.0, even));
  out.vDist     = dist;
  out.vAlpha    = alpha;
  out.vWorldPos = pos;
  out.vNormal   = normalize(vec3f(0.0, 1.0, 0.0));
  return out;
}
`;

const CHAIN_FRAG_WGSL = /* wgsl */`
${TUBE_UNIFORMS_WGSL}
${HSV_WGSL}
${FBR_WGSL}

@group(0) @binding(0)  var<uniform> uni        : TubeUniforms;
@group(0) @binding(10) var          tBaseColor : texture_2d<f32>;
@group(0) @binding(11) var          tRefraction: texture_2d<f32>;
@group(0) @binding(9)  var          samp       : sampler;

struct ChainFragIn {
  @builtin(position) fragCoord : vec4f,
  @location(0)       vUv       : vec2f,
  @location(1)       vDist     : f32,
  @location(2)       vAlpha    : f32,
  @location(3)       vWorldPos : vec3f,
  @location(4)       vNormal   : vec3f,
}

@fragment fn fs_chain(in: ChainFragIn) -> @location(0) vec4f {
  // AT ChainShader: getFBR base color
  let baseColor = textureSample(tBaseColor, samp, in.vUv).rgb;
  let viewDir   = normalize(vec3f(0.0, 0.0, 1.0) - in.vWorldPos);
  var color     = getFBR(baseColor, in.vNormal, viewDir, uni.time, in.vUv);

  // Refraction (AT: screenuv += normal.xy * 0.1 * uReflection.x)
  var screenuv = in.fragCoord.xy / vec2f(uni.canvasW, uni.canvasH);
  screenuv    += in.vNormal.xy * 0.1 * uni.uNormalStrength;
  color       += textureSample(tRefraction, samp, screenuv).rgb * uni.uReflectionBlend;

  // AT ChainShader: distance attenuation
  color *= mix(0.4, 1.2, clamp((18.0 - in.vDist) / 14.0, 0.0, 1.0));

  // AT: color = pow(color, 1.5)
  color = pow(max(color, vec3f(0.0)), vec3f(1.5));

  return vec4f(color * in.vAlpha, in.vAlpha);
}
`;

// ─── WGSL — WorkTube vertex + fragment ───────────────────────────────────────
// Matches WorkTubeShader.glsl: helical noise band tube for Work panel.

const WORK_TUBE_VERT_WGSL = /* wgsl */`
${WORK_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>       uni     : WorkUniforms;
@group(0) @binding(1) var<storage, read> nodeBuf : array<f32>;

struct WorkTubeVertOut {
  @builtin(position) pos      : vec4f,
  @location(0)       vUv      : vec2f,
  @location(1)       vWorldPos: vec3f,
}

const NS = ${NODE_STRIDE}u;
fn nGet(ni: u32, f: u32) -> f32 { return nodeBuf[ni * NS + f]; }

@vertex fn vs_worktube(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> WorkTubeVertOut {
  // ii = Work node index; render a single tube per node
  let nodeX = nGet(ii, 0u);
  let nodeY = nGet(ii, 1u);
  let nodeZ = nGet(ii, 2u);

  // Procedural cylinder strip: vi ∈ [0, 64) → 32 quads
  let qIdx  = vi / 6u;
  let corner= vi % 6u;
  let t0    = f32(qIdx)      / 32.0;
  let t1    = f32(qIdx + 1u) / 32.0;
  let t     = select(t0, t1, corner >= 3u);

  // WorkTubeShader helical offset: pos.x += cos(pos.y * 0.6) * 2.0
  let cylinY = nodeY + (t - 0.5) * 6.0;
  var pos    = vec3f(
    nodeX + cos(cylinY * 0.6) * 2.0,
    cylinY,
    nodeZ + sin(cylinY * 0.6) * 2.0,
  );

  let qU = t;
  let qV = select(0.0, 1.0, (corner == 1u) || (corner == 2u) || (corner == 4u));

  var out: WorkTubeVertOut;
  out.pos       = vec4f(pos.x * uni.scaleX - 1.0, pos.y * uni.scaleY - 1.0, pos.z * uni.scaleZ, 1.0);
  out.vUv       = vec2f(qU, qV);
  out.vWorldPos = pos;
  return out;
}
`;

const WORK_TUBE_FRAG_WGSL = /* wgsl */`
${WORK_UNIFORMS_WGSL}
${NOISE_WGSL}

@group(0) @binding(0) var<uniform> uni : WorkUniforms;

struct WorkTubeFragIn {
  @location(0) vUv      : vec2f,
  @location(1) vWorldPos: vec3f,
}

@fragment fn fs_worktube(in: WorkTubeFragIn) -> @location(0) vec4f {
  // AT WorkTubeShader: fract(vWorldPos.y * 0.2 + time * 0.3) noise band
  let noise = fract(in.vWorldPos.y * 0.2 + uni.time * uni.uWorkTubeTimeSpeed);
  let band  = smoothstep(0.5, 0.0, abs(noise - 0.5));
  let color = vec3f(pow(band, 5.0));
  return vec4f(color, band * 0.8);
}
`;

// ─── WGSL — WorkDetailParticle vertex + fragment ──────────────────────────────
// Matches WorkDetailParticleShader.glsl: GPU point sprites, matcap blend.

const WORK_PARTICLE_VERT_WGSL = /* wgsl */`
${WORK_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>       uni     : WorkUniforms;
@group(0) @binding(2) var                tWorkPos: texture_2d<f32>;
@group(0) @binding(3) var                samp    : sampler;

struct WorkPartVertOut {
  @builtin(position) pos    : vec4f,
  @location(0)       vUv   : vec2f,
  @location(1)       vAlpha: f32,
  @location(2)       vRand : vec4f,
  @location(3)       vPos  : vec3f,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(1.0, 1.0),
  vec2f(-1.0,-1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

@vertex fn vs_workpart(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> WorkPartVertOut {
  let tx = i32(ii % uni.texW);
  let ty = i32(ii / uni.texW);
  let p  = textureLoad(tWorkPos, vec2<i32>(tx, ty), 0);

  let px    = p.r;
  let py    = p.g;
  let pz    = p.b;
  let alpha = p.a;

  let alive = select(0.0, 1.0, alpha > 0.001);

  // AT WorkDetailParticleShader: size = (0.03 * DPR) * uSize * crange(rand.x, 0,1,0.5,1.5) * (1000/dist) * uSizeBias
  // Simplified: uniform size with distance approximation
  let dist   = max(0.5, length(vec3f(px, py, pz)));
  let halfS  = (0.03 * uni.uWorkDPR) * uni.uWorkParticleSize * (1000.0 / (dist * 100.0)) * uni.uWorkSizeBias * 0.5;

  let qv = QUAD[vi];
  let ndcX = px * uni.scaleX - 1.0 + qv.x * halfS * uni.scaleX;
  let ndcY = py * uni.scaleY - 1.0 + qv.y * halfS * uni.scaleY;

  var out: WorkPartVertOut;
  out.pos    = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv    = (qv + 1.0) * 0.5;
  out.vAlpha = alpha;
  out.vRand  = vec4f(0.5, 0.5, 0.5, 0.5);   // rand channel placeholder
  out.vPos   = vec3f(px, py, pz);
  return out;
}
`;

const WORK_PARTICLE_FRAG_WGSL = /* wgsl */`
${WORK_UNIFORMS_WGSL}
${HSV_WGSL}

@group(0) @binding(0) var<uniform> uni   : WorkUniforms;
@group(0) @binding(4) var          tMap  : texture_2d<f32>;
@group(0) @binding(5) var          samp  : sampler;

struct WorkPartFragIn {
  @location(0) vUv   : vec2f,
  @location(1) vAlpha: f32,
  @location(2) vRand : vec4f,
  @location(3) vPos  : vec3f,
}

@fragment fn fs_workpart(in: WorkPartFragIn) -> @location(0) vec4f {
  // AT WorkDetailParticleShader: circular discard
  if (length(in.vUv - 0.5) > 0.5) { discard; }

  // AT: uv2.x = crange(vPos.x, -7, 7, 0, 1), uv2.y = crange(vPos.y, -5, 5, 0, 1)
  var uv2 = vec2f(
    clamp((in.vPos.x + 7.0) / 14.0, 0.0, 1.0),
    clamp((in.vPos.y + 5.0) / 10.0, 0.0, 1.0),
  );

  // Matcap texture (AT: tMap matcapUV = rotateUV(uv, sin(time + ...) * 0.5 + 1.0))
  let angle    = sin(uni.time * 1.0 + in.vRand.z * 20.0) * 0.5 + 1.0;
  let cosA     = cos(angle); let sinA = sin(angle);
  let muv      = vec2f(
    (in.vUv.x - 0.5) * cosA - (in.vUv.y - 0.5) * sinA + 0.5,
    (in.vUv.x - 0.5) * sinA + (in.vUv.y - 0.5) * cosA + 0.5,
  );
  var matcap   = textureSample(tMap, samp, muv).rgb * 1.2;

  // AT: color = blendSoftLight(color, matcap, 0.8); blendOverlay(color, matcap, 0.2)
  var color    = matcap;

  // AT: color = rgb2hsv, .y *= 1.4, hsv2rgb
  var hsv      = rgb2hsv(color);
  hsv.y       *= 1.4;
  color        = hsv2rgb(clamp(hsv, vec3f(0.0), vec3f(1.0)));
  color       += 0.05;

  // AT: color *= smoothstep(-10, 10, vPos.z)
  color       *= clamp((in.vPos.z + 10.0) / 20.0, 0.0, 1.0);

  return vec4f(color * in.vAlpha, in.vAlpha);
}
`;

// ─── CPU helpers ──────────────────────────────────────────────────────────────

function buildNodeBuf(nodes: TubeNode[]): Float32Array {
  const buf = new Float32Array(nodes.length * NODE_STRIDE);
  for (let i = 0; i < nodes.length; i++) {
    const b = i * NODE_STRIDE;
    const n = nodes[i];
    buf[b + 0]  = n.x;
    buf[b + 1]  = n.y;
    buf[b + 2]  = n.z;
    buf[b + 3]  = n.scale  ?? 1.0;
    buf[b + 4]  = 1.0;                           // alpha
    buf[b + 5]  = Math.random() * Math.PI * 2;   // glowPhase
    buf[b + 6]  = 0.0;                            // scrollOff
    buf[b + 7]  = 1.0;                            // active
    buf[b + 8]  = n.colorHSV ? n.colorHSV[0] : (Math.random() * 0.15 + 0.55);
    buf[b + 9]  = n.colorHSV ? n.colorHSV[1] : 0.7;
    buf[b + 10] = n.colorHSV ? n.colorHSV[2] : 0.9;
    buf[b + 11] = 0.0;
  }
  return buf;
}

function buildEdgeBufCPU(
  edges: TubeEdgeDef[],
  nodes: TubeNode[],
): Float32Array {
  const nodeIndex = new Map(nodes.map((n, i) => [n.nodeId, i]));
  const buf = new Float32Array(edges.length * EDGE_STRIDE);
  for (let i = 0; i < edges.length; i++) {
    const b   = i * EDGE_STRIDE;
    const e   = edges[i];
    const src = nodeIndex.get(e.sourceId) ?? 0;
    const dst = nodeIndex.get(e.targetId) ?? 0;
    const sn  = nodes[src];
    const dn  = nodes[dst];
    const dx  = (dn?.x ?? 0) - (sn?.x ?? 0);
    const dy  = (dn?.y ?? 0) - (sn?.y ?? 0);
    const dz  = (dn?.z ?? 0) - (sn?.z ?? 0);
    buf[b + 0]  = src;
    buf[b + 1]  = dst;
    buf[b + 2]  = 0.0;                               // life (grows in GPU)
    buf[b + 3]  = Math.sqrt(dx * dx + dy * dy + dz * dz);  // length
    buf[b + 4]  = e.weight ?? 1.0;                   // alpha (= weight initially)
    buf[b + 5]  = 0.0;                               // scroll
    buf[b + 6]  = e.weight ?? 1.0;
    buf[b + 7]  = e.renderMode ?? 2;
  }
  return buf;
}

function buildWorkParticleBuf(
  nodes: TubeNode[],
  particlesPerNode: number,
): Float32Array {
  const workNodes = nodes.filter(n => n.type === 1);
  const total     = Math.min(workNodes.length * particlesPerNode, MAX_WORK_PARTICLES);
  const buf       = new Float32Array(total * WORK_PARTICLE_STRIDE);
  let   slot      = 0;
  for (const wn of workNodes) {
    for (let p = 0; p < particlesPerNode && slot < total; p++, slot++) {
      const b   = slot * WORK_PARTICLE_STRIDE;
      const off = 1.5;
      buf[b + 0]  = wn.x + (Math.random() - 0.5) * off;
      buf[b + 1]  = wn.y + (Math.random() - 0.5) * off;
      buf[b + 2]  = wn.z + (Math.random() - 0.5) * off;
      buf[b + 3]  = 0.0;
      buf[b + 4]  = Math.random();
      buf[b + 5]  = Math.random();
      buf[b + 6]  = Math.random();
      buf[b + 7]  = Math.random();
      buf[b + 8]  = (Math.random() - 0.5) * 0.01;
      buf[b + 9]  = Math.random() * 0.02 + 0.005;
      buf[b + 10] = (Math.random() - 0.5) * 0.01;
      buf[b + 11] = Math.random() * 2.0 + 0.5;
      buf[b + 12] = 0.0;   // nodeIdx (not used in shader currently)
    }
  }
  return buf;
}

// ─── ATTubeOrbChain — Main WebGPU class ───────────────────────────────────────

/**
 * ATTubeOrbChain
 *
 * WebGPU orchestrator for Active Theory's Tube-Orb-Chain topology rendering system.
 *
 * Encapsulates four GPU sub-systems:
 *   1. TubeOrb   — glowing billboard orb at each topology node (TubeOrbShader).
 *   2. TubeEdge  — cylindrical tube along each edge (TubeShader FBR + refraction).
 *   3. Chain     — sinusoidal chain links along each edge (ChainShader scroll).
 *   4. WorkDetail — per-Work-node noise-tube + particle + detail-cube system
 *                   (WorkTubeShader + WorkDetailParticleShader + WorkDetailCube).
 *
 * @example
 * ```ts
 * import { ATTubeOrbChain, TubeOrbChainPreset } from '$lib/sph/at-tube-orb-chain';
 *
 * const chain = new ATTubeOrbChain(device, canvas, nodes, edges, {
 *   ...TubeOrbChainPreset.network,
 *   onWorkHandoff: (nodeId, x, y, z) => { console.log(`work handoff: ${nodeId}`); },
 * });
 * await chain.build();
 *
 * // render loop:
 * const enc = device.createCommandEncoder();
 * chain.update(enc, elapsed, dt);
 * chain.render(enc, colorView, depthView?);
 * device.queue.submit([enc.finish()]);
 * ```
 */
export class ATTubeOrbChain {
  private readonly device:          GPUDevice;
  private readonly canvas:          HTMLCanvasElement;
  private readonly onWorkHandoff?:  ATTubeOrbChainConfig['onWorkHandoff'];

  private nodes:    TubeNode[]     = [];
  private edges:    TubeEdgeDef[]  = [];
  private cfg:      Required<Omit<ATTubeOrbChainConfig, 'onWorkHandoff'>>;

  private elapsed          = 0;
  private workParticleCount = 0;

  // ── GPU buffers ───────────────────────────────────────────────────────────
  private tubeUniformBuf!:  GPUBuffer;
  private workUniformBuf!:  GPUBuffer;
  private nodeStateBuf!:    GPUBuffer;
  private edgeStateBuf!:    GPUBuffer;
  private workParticleBuf!: GPUBuffer;

  // ── tPos textures ─────────────────────────────────────────────────────────
  private tOrbPos!:     GPUTexture;
  private tOrbPosView!: GPUTextureView;
  private tChainPos!:   GPUTexture;
  private tChainPosView!:GPUTextureView;
  private tWorkPos!:    GPUTexture;
  private tWorkPosView!:GPUTextureView;

  private sampler!: GPUSampler;

  // ── Placeholder 1×1 textures (substitutes for tColor, tRefraction, tMap) ─
  private tWhite1x1!:   GPUTexture;
  private tWhite1x1View!: GPUTextureView;

  // ── Compute pipelines ─────────────────────────────────────────────────────
  private nodeComputePipe!: GPUComputePipeline;
  private edgeComputePipe!: GPUComputePipeline;
  private workPartPipe!:    GPUComputePipeline;

  // ── Render pipelines ──────────────────────────────────────────────────────
  private orbRenderPipe!:      GPURenderPipeline;
  private tubeRenderPipe!:     GPURenderPipeline;
  private chainRenderPipe!:    GPURenderPipeline;
  private workTubePipe!:       GPURenderPipeline;
  private workPartRenderPipe!: GPURenderPipeline;

  // ── Bind groups ───────────────────────────────────────────────────────────
  private nodeCBG0!: GPUBindGroup;
  private nodeCBG1!: GPUBindGroup;
  private edgeCBG0!: GPUBindGroup;
  private edgeCBG1!: GPUBindGroup;
  private workCBG0!: GPUBindGroup;
  private workCBG1!: GPUBindGroup;

  private orbRBG!:      GPUBindGroup;
  private tubeRBG!:     GPUBindGroup;
  private chainRBG!:    GPUBindGroup;
  private workTubeRBG!: GPUBindGroup;
  private workPartRBG!: GPUBindGroup;

  private built = false;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    nodes:  TubeNode[],
    edges:  TubeEdgeDef[],
    config: ATTubeOrbChainConfig = {},
  ) {
    this.device          = device;
    this.canvas          = canvas;
    this.nodes           = nodes;
    this.edges           = edges;
    this.onWorkHandoff   = config.onWorkHandoff;
    this.cfg = {
      uScroll:              config.uScroll              ?? 0.0,
      uReflectionBlend:     config.uReflectionBlend     ?? 0.5,
      uNormalStrength:      config.uNormalStrength      ?? 1.0,
      uOrbAlpha:            config.uOrbAlpha            ?? 1.0,
      uChainAmplitude:      config.uChainAmplitude      ?? 1.1,
      uChainFrequency:      config.uChainFrequency      ?? 0.4,
      workParticlesPerNode: config.workParticlesPerNode ?? 128,
      uWorkParticleSize:    config.uWorkParticleSize    ?? 0.03,
      uWorkDPR:             config.uWorkDPR             ?? 1.0,
      uWorkTubeTimeSpeed:   config.uWorkTubeTimeSpeed   ?? 0.3,
      uTubeLifeSpeed:       config.uTubeLifeSpeed       ?? 0.01,
    };
  }

  // ─── Build ────────────────────────────────────────────────────────────────

  async build(): Promise<void> {
    if (this.built) this._destroy();
    const { device } = this;

    // ── Work particle count ──────────────────────────────────────────────────
    const workCount = this.nodes.filter(n => n.type === 1).length;
    this.workParticleCount = Math.min(
      workCount * this.cfg.workParticlesPerNode,
      MAX_WORK_PARTICLES,
    );

    // ── Uniform buffers ──────────────────────────────────────────────────────
    this.tubeUniformBuf = device.createBuffer({
      size:  TUBE_UNIFORMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.workUniformBuf = device.createBuffer({
      size:  WORK_UNIFORMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeTubeUniforms(0);
    this._writeWorkUniforms(0);

    // ── Node state buffer ────────────────────────────────────────────────────
    const nodeData = buildNodeBuf(this.nodes);
    this.nodeStateBuf = device.createBuffer({
      size:  Math.max(nodeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(this.nodeStateBuf, 0, nodeData);

    // ── Edge state buffer ────────────────────────────────────────────────────
    const edgeData = buildEdgeBufCPU(this.edges, this.nodes);
    this.edgeStateBuf = device.createBuffer({
      size:  Math.max(edgeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(this.edgeStateBuf, 0, edgeData);

    // ── Work particle buffer ─────────────────────────────────────────────────
    const workData = buildWorkParticleBuf(this.nodes, this.cfg.workParticlesPerNode);
    this.workParticleBuf = device.createBuffer({
      size:  Math.max(workData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(this.workParticleBuf, 0, workData);

    // ── Position textures ────────────────────────────────────────────────────
    const texUsage =
      GPUTextureUsage.TEXTURE_BINDING  |
      GPUTextureUsage.STORAGE_BINDING  |
      GPUTextureUsage.COPY_SRC;

    this.tOrbPos = device.createTexture({ size: [TEX_W, TEX_H], format: 'rgba32float', usage: texUsage });
    this.tOrbPosView = this.tOrbPos.createView();

    this.tChainPos = device.createTexture({ size: [TEX_W, TEX_H], format: 'rgba32float', usage: texUsage });
    this.tChainPosView = this.tChainPos.createView();

    this.tWorkPos = device.createTexture({ size: [TEX_W, TEX_H], format: 'rgba32float', usage: texUsage });
    this.tWorkPosView = this.tWorkPos.createView();

    // ── Placeholder 1×1 white texture ───────────────────────────────────────
    this.tWhite1x1 = device.createTexture({
      size:   [1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.tWhite1x1 },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    this.tWhite1x1View = this.tWhite1x1.createView();

    // ── Sampler ──────────────────────────────────────────────────────────────
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    // ── Compute pipelines ────────────────────────────────────────────────────
    this.nodeComputePipe = device.createComputePipeline({
      layout:  'auto',
      compute: { module: device.createShaderModule({ code: NODE_COMPUTE_SHADER }), entryPoint: 'main' },
    });
    this.edgeComputePipe = device.createComputePipeline({
      layout:  'auto',
      compute: { module: device.createShaderModule({ code: EDGE_COMPUTE_SHADER }), entryPoint: 'main' },
    });
    this.workPartPipe = device.createComputePipeline({
      layout:  'auto',
      compute: { module: device.createShaderModule({ code: WORK_PARTICLE_COMPUTE }), entryPoint: 'main' },
    });

    // ── Render pipelines ─────────────────────────────────────────────────────
    const fmt     = navigator.gpu.getPreferredCanvasFormat();
    const blendAO = {
      color: { srcFactor: 'src-alpha' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation },
      alpha: { srcFactor: 'one'       as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation },
    };
    const blendAlpha = {
      color: { srcFactor: 'src-alpha' as GPUBlendFactor, dstFactor: 'one-minus-src-alpha' as GPUBlendFactor, operation: 'add' as GPUBlendOperation },
      alpha: { srcFactor: 'one'       as GPUBlendFactor, dstFactor: 'one-minus-src-alpha' as GPUBlendFactor, operation: 'add' as GPUBlendOperation },
    };

    const makeRenderPipe = (
      vertSrc: string, fragSrc: string,
      blend: GPUBlendState,
      topology: GPUPrimitiveTopology = 'triangle-list',
    ) => device.createRenderPipeline({
      layout:   'auto',
      vertex:   { module: device.createShaderModule({ code: vertSrc }) },
      fragment: { module: device.createShaderModule({ code: fragSrc }), targets: [{ format: fmt, blend }] },
      primitive: { topology },
    });

    this.orbRenderPipe      = makeRenderPipe(TUBE_ORB_VERT_WGSL,     TUBE_ORB_FRAG_WGSL,      blendAO);
    this.tubeRenderPipe     = makeRenderPipe(TUBE_EDGE_VERT_WGSL,    TUBE_EDGE_FRAG_WGSL,     blendAlpha);
    this.chainRenderPipe    = makeRenderPipe(CHAIN_VERT_WGSL,        CHAIN_FRAG_WGSL,         blendAlpha);
    this.workTubePipe       = makeRenderPipe(WORK_TUBE_VERT_WGSL,    WORK_TUBE_FRAG_WGSL,     blendAO);
    this.workPartRenderPipe = makeRenderPipe(WORK_PARTICLE_VERT_WGSL, WORK_PARTICLE_FRAG_WGSL, blendAO);

    // ── Bind groups ──────────────────────────────────────────────────────────
    this._buildBindGroups();
    this.built = true;

    console.log(
      `[ATTubeOrbChain] built: ${this.nodes.length} nodes, ` +
      `${this.edges.length} edges, ` +
      `${this.workParticleCount} work particles`,
    );
  }

  // ─── Per-frame update ─────────────────────────────────────────────────────

  /**
   * Encode all compute passes:
   *   1. Node glow compute (TubeOrb pulsation → tOrbPos)
   *   2. Edge life + chain scroll compute (→ tChainPos)
   *   3. Work particle compute (→ tWorkPos)
   *
   * @param encoder  — open GPUCommandEncoder
   * @param elapsed  — total elapsed seconds
   * @param _dt      — frame delta (unused, shader uses fixed step)
   */
  update(encoder: GPUCommandEncoder, elapsed: number, _dt = 0): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this._writeTubeUniforms(elapsed);
    this._writeWorkUniforms(elapsed);

    const nodeWG = Math.ceil(this.nodes.length / WG);
    const edgeWG = Math.ceil(this.edges.length / WG);
    const workWG = Math.ceil(this.workParticleCount / WG);

    const pass = encoder.beginComputePass();

    // Node compute
    pass.setPipeline(this.nodeComputePipe);
    pass.setBindGroup(0, this.nodeCBG0);
    pass.setBindGroup(1, this.nodeCBG1);
    pass.dispatchWorkgroups(Math.max(1, nodeWG));

    // Edge compute
    pass.setPipeline(this.edgeComputePipe);
    pass.setBindGroup(0, this.edgeCBG0);
    pass.setBindGroup(1, this.edgeCBG1);
    pass.dispatchWorkgroups(Math.max(1, edgeWG));

    // Work particle compute
    if (this.workParticleCount > 0) {
      pass.setPipeline(this.workPartPipe);
      pass.setBindGroup(0, this.workCBG0);
      pass.setBindGroup(1, this.workCBG1);
      pass.dispatchWorkgroups(Math.max(1, workWG));
    }

    pass.end();
  }

  // ─── Per-frame render ─────────────────────────────────────────────────────

  /**
   * Encode all render passes in this order:
   *   1. TubeEdge (cylindrical tubes, back-most)
   *   2. Chain (sinusoidal links)
   *   3. WorkTube (noise-band tubes at Work nodes)
   *   4. WorkDetailParticle (GPU particles around Work nodes)
   *   5. TubeOrb (glowing node spheres, front-most / additive)
   *
   * @param encoder   — open GPUCommandEncoder
   * @param colorView — render target texture view
   * @param depthView — optional depth attachment
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
        view:         depthView,
        depthLoadOp:  'load',
        depthStoreOp: 'store',
      };
    }

    const pass = encoder.beginRenderPass(passDesc);

    // 1 — Tube edges (16 radial segments × 2 triangles × 32 rings per edge)
    if (this.edges.length > 0) {
      pass.setPipeline(this.tubeRenderPipe);
      pass.setBindGroup(0, this.tubeRBG);
      // 16 segs × 2 tris × 3 verts = 96 verts/edge
      pass.draw(96, this.edges.length);
    }

    // 2 — Chain links (32 quads × 6 verts per edge)
    const chainEdges = this.edges.filter(e => (e.renderMode ?? 2) !== 0);
    if (chainEdges.length > 0) {
      pass.setPipeline(this.chainRenderPipe);
      pass.setBindGroup(0, this.chainRBG);
      pass.draw(192, this.edges.length);   // 32 quads × 6 verts
    }

    // 3 — Work tubes (32 quads × 6 verts per Work node)
    const workNodeCount = this.nodes.filter(n => n.type === 1).length;
    if (workNodeCount > 0) {
      pass.setPipeline(this.workTubePipe);
      pass.setBindGroup(0, this.workTubeRBG);
      pass.draw(192, workNodeCount);   // 32 quads × 6 verts
    }

    // 4 — Work detail particles (instanced quads)
    if (this.workParticleCount > 0) {
      pass.setPipeline(this.workPartRenderPipe);
      pass.setBindGroup(0, this.workPartRBG);
      pass.draw(6, this.workParticleCount);
    }

    // 5 — TubeOrb billboard orbs (instanced quads, additive)
    if (this.nodes.length > 0) {
      pass.setPipeline(this.orbRenderPipe);
      pass.setBindGroup(0, this.orbRBG);
      pass.draw(6, this.nodes.length);
    }

    pass.end();
  }

  // ─── Live parameter setters ───────────────────────────────────────────────

  setScroll(v: number): void           { this.cfg.uScroll            = v; }
  setOrbAlpha(v: number): void         { this.cfg.uOrbAlpha          = v; }
  setReflectionBlend(v: number): void  { this.cfg.uReflectionBlend   = v; }
  setNormalStrength(v: number): void   { this.cfg.uNormalStrength     = v; }
  setChainAmplitude(v: number): void   { this.cfg.uChainAmplitude    = v; }
  setChainFrequency(v: number): void   { this.cfg.uChainFrequency    = v; }
  setWorkParticleSize(v: number): void { this.cfg.uWorkParticleSize   = v; }
  setWorkTubeTimeSpeed(v: number): void{ this.cfg.uWorkTubeTimeSpeed  = v; }
  setTubeLifeSpeed(v: number): void    { this.cfg.uTubeLifeSpeed      = v; }

  /** Apply a TubeOrbChainPreset without requiring a full rebuild. */
  applyPreset(preset: ATTubeOrbChainConfig): void {
    if (preset.uScroll              !== undefined) this.cfg.uScroll              = preset.uScroll;
    if (preset.uReflectionBlend     !== undefined) this.cfg.uReflectionBlend     = preset.uReflectionBlend;
    if (preset.uNormalStrength      !== undefined) this.cfg.uNormalStrength      = preset.uNormalStrength;
    if (preset.uOrbAlpha            !== undefined) this.cfg.uOrbAlpha            = preset.uOrbAlpha;
    if (preset.uChainAmplitude      !== undefined) this.cfg.uChainAmplitude      = preset.uChainAmplitude;
    if (preset.uChainFrequency      !== undefined) this.cfg.uChainFrequency      = preset.uChainFrequency;
    if (preset.workParticlesPerNode !== undefined) this.cfg.workParticlesPerNode = preset.workParticlesPerNode;
    if (preset.uWorkParticleSize    !== undefined) this.cfg.uWorkParticleSize    = preset.uWorkParticleSize;
    if (preset.uWorkDPR             !== undefined) this.cfg.uWorkDPR             = preset.uWorkDPR;
    if (preset.uWorkTubeTimeSpeed   !== undefined) this.cfg.uWorkTubeTimeSpeed   = preset.uWorkTubeTimeSpeed;
    if (preset.uTubeLifeSpeed       !== undefined) this.cfg.uTubeLifeSpeed       = preset.uTubeLifeSpeed;
  }

  /** Set a node's active state (1=visible, 0=hidden). */
  setNodeActive(nodeId: string, active: boolean): void {
    const idx = this.nodes.findIndex(n => n.nodeId === nodeId);
    if (idx < 0 || !this.built) return;
    const buf = new Float32Array(1);
    buf[0] = active ? 1.0 : 0.0;
    this.device.queue.writeBuffer(
      this.nodeStateBuf,
      (idx * NODE_STRIDE + 7) * 4,
      buf,
    );
  }

  /** Update node position at runtime (e.g. topology change). */
  setNodePosition(nodeId: string, x: number, y: number, z: number): void {
    const idx = this.nodes.findIndex(n => n.nodeId === nodeId);
    if (idx < 0 || !this.built) return;
    const buf = new Float32Array(3);
    buf[0] = x; buf[1] = y; buf[2] = z;
    this.device.queue.writeBuffer(
      this.nodeStateBuf,
      idx * NODE_STRIDE * 4,
      buf,
    );
  }

  /** Replace nodes + edges and rebuild GPU resources. */
  async setTopology(nodes: TubeNode[], edges: TubeEdgeDef[]): Promise<void> {
    this.nodes = nodes;
    this.edges = edges;
    await this.build();
  }

  // ─── Introspection ────────────────────────────────────────────────────────

  /**
   * Read all TubeOrb node states from GPU for debug / diagnostics.
   * Expensive — do not call every frame.
   */
  async readOrbStates(): Promise<TubeOrbState[]> {
    if (!this.built) return [];
    const { device } = this;
    const sz = this.nodeStateBuf.size;

    const readBuf = device.createBuffer({
      size:  sz,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.nodeStateBuf, 0, readBuf, 0, sz);
    device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(readBuf.getMappedRange());
    const out: TubeOrbState[] = [];

    for (let i = 0; i < this.nodes.length; i++) {
      const b = i * NODE_STRIDE;
      out.push({
        nodeId:  this.nodes[i].nodeId,
        x:       raw[b + 0],
        y:       raw[b + 1],
        z:       raw[b + 2],
        scale:   raw[b + 3],
        alpha:   raw[b + 4],
        colorH:  raw[b + 8],
        colorS:  raw[b + 9],
        colorV:  raw[b + 10],
        active:  raw[b + 7] > 0.5,
      });
    }

    readBuf.unmap();
    readBuf.destroy();
    return out;
  }

  // ─── Accessors ────────────────────────────────────────────────────────────

  get nodeCount(): number         { return this.nodes.length; }
  get edgeCount(): number         { return this.edges.length; }
  get workParticles(): number     { return this.workParticleCount; }
  get isBuilt(): boolean          { return this.built; }
  get elapsedTime(): number       { return this.elapsed; }
  get config(): Readonly<Required<Omit<ATTubeOrbChainConfig, 'onWorkHandoff'>>> {
    return this.cfg;
  }

  // ─── Cleanup ──────────────────────────────────────────────────────────────

  destroy(): void { this._destroy(); }

  // ─── Private helpers ──────────────────────────────────────────────────────

  private _writeTubeUniforms(elapsed: number): void {
    const { device, cfg, canvas } = this;
    const cw = canvas.width  || 1;
    const ch = canvas.height || 1;
    const data = new Float32Array(TUBE_UNIFORMS_BYTES / 4);

    data[TU_TIME             / 4] = elapsed;
    data[TU_SCROLL           / 4] = cfg.uScroll;
    data[TU_REFLECTION_BLEND / 4] = cfg.uReflectionBlend;
    data[TU_NORMAL_STRENGTH  / 4] = cfg.uNormalStrength;
    data[TU_ORB_ALPHA        / 4] = cfg.uOrbAlpha;
    data[TU_CHAIN_AMP        / 4] = cfg.uChainAmplitude;
    data[TU_CHAIN_FREQ       / 4] = cfg.uChainFrequency;
    data[TU_TUBE_LIFE_SPEED  / 4] = cfg.uTubeLifeSpeed;
    data[TU_SCALE_X          / 4] = 2.0 / cw;
    data[TU_SCALE_Y          / 4] = 2.0 / ch;
    data[TU_SCALE_Z          / 4] = 1.0;
    data[TU_CANVAS_W         / 4] = cw;
    data[TU_CANVAS_H         / 4] = ch;

    const u32 = new Uint32Array(data.buffer);
    u32[TU_NODE_COUNT / 4] = this.nodes.length;
    u32[TU_EDGE_COUNT / 4] = this.edges.length;
    u32[TU_TEX_W      / 4] = TEX_W;
    u32[TU_TEX_H      / 4] = TEX_H;

    device.queue.writeBuffer(this.tubeUniformBuf, 0, data);
  }

  private _writeWorkUniforms(elapsed: number): void {
    const { device, cfg, canvas } = this;
    const cw = canvas.width  || 1;
    const ch = canvas.height || 1;
    const data = new Float32Array(WORK_UNIFORMS_BYTES / 4);

    data[WU_TIME            / 4] = elapsed;
    data[WU_TUBE_TIME_SPEED / 4] = cfg.uWorkTubeTimeSpeed;
    data[WU_PARTICLE_SIZE   / 4] = cfg.uWorkParticleSize;
    data[WU_DPR             / 4] = cfg.uWorkDPR;
    data[WU_SIZE_BIAS       / 4] = 1.0;
    data[WU_SCALE_X         / 4] = 2.0 / cw;
    data[WU_SCALE_Y         / 4] = 2.0 / ch;
    data[WU_SCALE_Z         / 4] = 1.0;

    const u32 = new Uint32Array(data.buffer);
    u32[WU_PARTICLE_COUNT / 4] = this.workParticleCount;
    u32[WU_TEX_W          / 4] = TEX_W;
    u32[WU_TEX_H          / 4] = TEX_H;

    device.queue.writeBuffer(this.workUniformBuf, 0, data);
  }

  private _buildBindGroups(): void {
    const { device } = this;
    const w = this.tWhite1x1View;
    const s = this.sampler;

    // ── Node compute ─────────────────────────────────────────────────────────
    this.nodeCBG0 = device.createBindGroup({
      layout:  this.nodeComputePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.tubeUniformBuf } }],
    });
    this.nodeCBG1 = device.createBindGroup({
      layout:  this.nodeComputePipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.nodeStateBuf } },
        { binding: 1, resource: this.tOrbPosView },
      ],
    });

    // ── Edge compute ─────────────────────────────────────────────────────────
    this.edgeCBG0 = device.createBindGroup({
      layout:  this.edgeComputePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.tubeUniformBuf } }],
    });
    this.edgeCBG1 = device.createBindGroup({
      layout:  this.edgeComputePipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.nodeStateBuf } },
        { binding: 1, resource: { buffer: this.edgeStateBuf } },
        { binding: 2, resource: this.tChainPosView },
      ],
    });

    // ── Work particle compute ─────────────────────────────────────────────────
    this.workCBG0 = device.createBindGroup({
      layout:  this.workPartPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.workUniformBuf } }],
    });
    this.workCBG1 = device.createBindGroup({
      layout:  this.workPartPipe.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.workParticleBuf } },
        { binding: 1, resource: this.tWorkPosView },
      ],
    });

    // ── Orb render ────────────────────────────────────────────────────────────
    this.orbRBG = device.createBindGroup({
      layout:  this.orbRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tubeUniformBuf } },
        { binding: 1, resource: this.tOrbPosView },
        { binding: 2, resource: s },
        { binding: 3, resource: w },   // tMap placeholder
        { binding: 4, resource: s },
      ],
    });

    // ── Tube edge render ──────────────────────────────────────────────────────
    this.tubeRBG = device.createBindGroup({
      layout:  this.tubeRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tubeUniformBuf } },
        { binding: 5, resource: { buffer: this.edgeStateBuf } },
        { binding: 6, resource: { buffer: this.nodeStateBuf } },
        { binding: 7, resource: w },   // tTubeColor placeholder
        { binding: 8, resource: w },   // tRefraction placeholder
        { binding: 9, resource: s },
      ],
    });

    // ── Chain render ──────────────────────────────────────────────────────────
    this.chainRBG = device.createBindGroup({
      layout:  this.chainRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0,  resource: { buffer: this.tubeUniformBuf } },
        { binding: 5,  resource: { buffer: this.edgeStateBuf } },
        { binding: 6,  resource: { buffer: this.nodeStateBuf } },
        { binding: 9,  resource: s },
        { binding: 10, resource: w },   // tBaseColor placeholder
        { binding: 11, resource: w },   // tRefraction placeholder
      ],
    });

    // ── WorkTube render ───────────────────────────────────────────────────────
    this.workTubeRBG = device.createBindGroup({
      layout:  this.workTubePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workUniformBuf } },
        { binding: 1, resource: { buffer: this.nodeStateBuf } },
      ],
    });

    // ── WorkDetail particle render ────────────────────────────────────────────
    this.workPartRBG = device.createBindGroup({
      layout:  this.workPartRenderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workUniformBuf } },
        { binding: 2, resource: this.tWorkPosView },
        { binding: 3, resource: s },
        { binding: 4, resource: w },   // tMap placeholder
        { binding: 5, resource: s },
      ],
    });
  }

  private _destroy(): void {
    if (!this.built) return;
    this.tubeUniformBuf?.destroy();
    this.workUniformBuf?.destroy();
    this.nodeStateBuf?.destroy();
    this.edgeStateBuf?.destroy();
    this.workParticleBuf?.destroy();
    this.tOrbPos?.destroy();
    this.tChainPos?.destroy();
    this.tWorkPos?.destroy();
    this.tWhite1x1?.destroy();
    this.built = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create an ATTubeOrbChain from raw topology data.
 *
 * @param device       — WebGPU device
 * @param canvas       — render canvas
 * @param nodeList     — array of { id, x, y, z, type? } records
 * @param edgeList     — array of { id, src, dst, weight? } records
 * @param config       — optional config overrides
 *
 * @example
 * ```ts
 * const chain = await createTubeOrbChainFromTopology(device, canvas, nodes, edges);
 * ```
 */
export async function createTubeOrbChainFromTopology(
  device:   GPUDevice,
  canvas:   HTMLCanvasElement,
  nodeList: Array<{ id: string; x: number; y: number; z: number; type?: 0 | 1 }>,
  edgeList: Array<{ id: string; src: string; dst: string; weight?: number }>,
  config:   ATTubeOrbChainConfig = {},
): Promise<ATTubeOrbChain> {
  const nodes: TubeNode[] = nodeList.map(n => ({
    nodeId: n.id,
    x: n.x, y: n.y, z: n.z,
    type: n.type ?? 0,
  }));
  const edges: TubeEdgeDef[] = edgeList.map(e => ({
    edgeId:     e.id,
    sourceId:   e.src,
    targetId:   e.dst,
    weight:     e.weight ?? 1.0,
    renderMode: 2,
  }));
  const chain = new ATTubeOrbChain(device, canvas, nodes, edges, config);
  await chain.build();
  return chain;
}

/**
 * Convert a pixel-space node layout into normalised world-space TubeNode array.
 *
 * @param records  — nodes with pixel-space (px, py) coordinates
 * @param canvasW  — canvas pixel width
 * @param canvasH  — canvas pixel height
 * @param domainW  — world domain width
 * @param domainH  — world domain height
 */
export function pixelNodesToTubeNodes(
  records: Array<{ nodeId: string; px: number; py: number; type?: 0 | 1; scale?: number }>,
  canvasW: number,
  canvasH: number,
  domainW: number,
  domainH: number,
): TubeNode[] {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return records.map(r => ({
    nodeId: r.nodeId,
    x:      r.px * sx,
    y:      r.py * sy,
    z:      0,
    scale:  r.scale ?? 1.0,
    type:   r.type  ?? 0,
  }));
}

// ─── Defaults re-export ────────────────────────────────────────────────────────

export const AT_TUBE_ORB_CHAIN_DEFAULTS = {
  maxNodes:          MAX_NODES,
  maxEdges:          MAX_EDGES,
  maxWorkParticles:  MAX_WORK_PARTICLES,
  texW:              TEX_W,
  texH:              TEX_H,
  nodeStride:        NODE_STRIDE,
  edgeStride:        EDGE_STRIDE,
  workParticleStride: WORK_PARTICLE_STRIDE,
  workgroupSize:     WG,
} as const;
