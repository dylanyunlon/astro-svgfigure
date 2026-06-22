/**
 * edge-flow-renderer.ts — M742
 *
 * Edge-Flow Renderer: particles flowing along Catmull-Rom splines with
 * QoS-driven visual treatment — dual-mode (WebGPU compute + Canvas2D fallback).
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * This module unifies the GPU spline-particle lifecycle (at-spline-particle.ts)
 * with the QoS colour/speed theme system into a single renderer that:
 *
 *   1. **GPU path** — When a GPUDevice is available, runs a lightweight compute
 *      pass that advances per-particle travel, evaluates Catmull-Rom positions,
 *      applies QoS-driven speed scaling, and writes to a tPos storage texture.
 *      A render pass draws instanced quads with QoS-tinted soft-particle SDF.
 *
 *   2. **CPU fallback** — When no GPUDevice is provided, runs a pure-JS tick
 *      loop and renders via Canvas2D with additive glow compositing.
 *
 * QoS visual language (colour × speed × trail × radius):
 *
 *   SENSOR_DATA  — fast flow, cool-blue trail, short decay (high-freq bursts)
 *   PARAMETERS   — moderate flow, warm-amber, medium trails (stable streams)
 *   TF_STATIC    — slow drift, jade-green, long trails (persistent transforms)
 *   TOPO_CHANGE  — rapid pulse, magenta, sharp decay (disruptive events)
 *   DEFAULT      — balanced flow, neutral slate
 *
 * Particle lifecycle (mirroring AT SplineParticleLife FSM):
 *   SPAWN  → wait out random delay (stagger visual density)
 *   FLOW   → advance along Catmull-Rom spline with curl-noise perturbation
 *   DECAY  → alpha fades at QoS-specific rate after reaching spline end
 *   DEAD   → slot recycled (respawn with new speed/delay)
 *
 * QoS metrics overlay:
 *   Each edge can display a live QoS health indicator — a small coloured bar
 *   at the midpoint of the spline showing bandwidth utilisation (mps ratio),
 *   latency (travel time accumulator), and reliability (drop-count tracker).
 *   These are updated via `reportQoSMetrics()` and rendered as overlay badges.
 *
 * Integration:
 *   ```ts
 *   import { EdgeFlowRenderer } from '$lib/sph/edge-flow-renderer';
 *
 *   // GPU mode:
 *   const renderer = EdgeFlowRenderer.createGPU(device, canvas, {
 *     edges: topology.edges,
 *     onArrival: (edgeId, targetId, x, y) => { … },
 *   });
 *   await renderer.build();
 *   // render loop: renderer.tick(enc, elapsed, dt); renderer.render(enc, view);
 *
 *   // CPU fallback:
 *   const renderer = EdgeFlowRenderer.createCPU(ctx2d, {
 *     edges: topology.edges,
 *     onArrival: (edgeId, targetId, x, y) => { … },
 *   });
 *   // render loop: renderer.tick(null, elapsed, dt); renderer.draw();
 *   ```
 *
 * References:
 *   src/lib/sph/at-spline-particle.ts   — WebGPU particle pipeline (parent)
 *   src/lib/sph/spline-particle-life.ts — CPU SplineParticleLife reference
 *   src/lib/sph/color-palette.ts        — QoS → colour theme mapping
 *   src/lib/sph/qosSpatial.ts           — QoS profile definitions
 */

import type { QoSProfile }      from './types';
import type { QoSProfileName }  from './qosSpatial';
import { QOS_PRESETS }           from './qosSpatial';
import { QOS_THEME }             from './color-palette';
import type { ThemePalette, RGB } from './color-palette';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Default particle pool per unit of edge weight. */
const DEFAULT_PARTICLES_PER_UNIT = 12;

/** Maximum particles across all edges. */
const MAX_FLOW_PARTICLES = 4096;

/** GPU workgroup size for compute pass. */
const GPU_WG = 64;

/** GPU tPos texture width (W × H ≥ MAX_FLOW_PARTICLES). */
const GPU_TEX_W = 64;
const GPU_TEX_H = 64; // 64 × 64 = 4096

/** Catmull-Rom tension (standard: 0.5). */
const CR_TENSION = 0.5;

/** Curl-noise evaluation epsilon. */
const CURL_EPS = 0.01;

/** Particle stride in f32 count (GPU buffer layout). */
const P_STRIDE = 12;

// ── Per-particle field offsets ──────────────────────────────────────────────
const F_TRAVEL   = 0;
const F_SPEED    = 1;
const F_DELAY    = 2;
const F_PHASE    = 3; // 0=spawn, 1=flow, 2=decay, 3=dead
const F_ALPHA    = 4;
const F_SEED     = 5;
const F_EDGE     = 6;
const F_POS_X    = 7;
const F_POS_Y    = 8;
const F_HANDOFF  = 9;
const F_QOS_IDX  = 10; // QoS profile index (0–4)
const F_CURL_OFF = 11; // lateral curl noise offset

// ── Edge buffer stride ──────────────────────────────────────────────────────
const EDGE_STRIDE = 136; // 4 header + 32 points × 4 f32 + 4 QoS fields
const EDGE_MAX_PTS = 32;

// ─── QoS → flow-style mapping ────────────────────────────────────────────────

export interface QoSFlowStyle {
  /** Base speed range (travel units / second). */
  speed:       [number, number];
  /** Particle trail length in travel units. */
  trailLen:    number;
  /** Particle radius in domain units (Canvas2D) or quad half-size (GPU). */
  radius:      number;
  /** Decay rate (alpha per second after arrival). */
  decayRate:   number;
  /** Max random spawn delay in seconds. */
  maxDelay:    number;
  /** Curl-noise strength (lateral displacement amplitude). */
  curlStr:     number;
  /** Curl-noise spatial frequency. */
  curlScale:   number;
  /** Curl-noise temporal speed. */
  curlSpeed:   number;
}

const QOS_FLOW_STYLE: Record<QoSProfileName, QoSFlowStyle> = {
  SENSOR_DATA: {
    speed:     [0.35, 0.55],
    trailLen:  0.08,
    radius:    2.5,
    decayRate: 1.2,
    maxDelay:  0.15,
    curlStr:   0.015,
    curlScale: 1.5,
    curlSpeed: 8.0,
  },
  PARAMETERS: {
    speed:     [0.18, 0.30],
    trailLen:  0.12,
    radius:    3.0,
    decayRate: 0.6,
    maxDelay:  0.5,
    curlStr:   0.04,
    curlScale: 2.5,
    curlSpeed: 4.0,
  },
  TF_STATIC: {
    speed:     [0.08, 0.15],
    trailLen:  0.18,
    radius:    3.5,
    decayRate: 0.35,
    maxDelay:  1.0,
    curlStr:   0.08,
    curlScale: 3.5,
    curlSpeed: 2.0,
  },
  TOPO_CHANGE: {
    speed:     [0.45, 0.70],
    trailLen:  0.06,
    radius:    2.0,
    decayRate: 1.5,
    maxDelay:  0.1,
    curlStr:   0.02,
    curlScale: 1.0,
    curlSpeed: 10.0,
  },
  DEFAULT: {
    speed:     [0.20, 0.35],
    trailLen:  0.10,
    radius:    2.8,
    decayRate: 0.8,
    maxDelay:  0.3,
    curlStr:   0.04,
    curlScale: 2.0,
    curlSpeed: 5.0,
  },
};

/** QoS profile name → integer index for GPU buffer. */
const QOS_NAME_TO_IDX: Record<QoSProfileName, number> = {
  DEFAULT:     0,
  SENSOR_DATA: 1,
  PARAMETERS:  2,
  TF_STATIC:   3,
  TOPO_CHANGE: 4,
};

// ─── QoS metrics (live telemetry overlay) ────────────────────────────────────

/** Live QoS metrics for a single edge. */
export interface EdgeQoSMetrics {
  /** Messages per second (actual throughput). */
  mps:           number;
  /** Round-trip latency in ms (visual: travel-time accumulator). */
  latencyMs:     number;
  /** Dropped messages count since last reset. */
  dropCount:     number;
  /** Bandwidth utilisation [0, 1] relative to QoS profile's declared mps. */
  utilisation:   number;
  /** Health grade: 0 = dead, 1 = degraded, 2 = healthy. */
  health:        0 | 1 | 2;
}

/** Default metrics for uninstrumented edges. */
const METRICS_DEFAULT: EdgeQoSMetrics = {
  mps: 0, latencyMs: 0, dropCount: 0, utilisation: 0, health: 2,
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D control point for an edge spline. */
export interface FlowPoint {
  x: number;
  y: number;
}

/** One topology edge definition. */
export interface FlowEdge {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points (≥2, in canvas/domain space). */
  points:   FlowPoint[];
  /** Connectivity weight — controls particle density on this edge. */
  weight:   number;
  /** QoS profile name; defaults to 'DEFAULT'. */
  qos?:     QoSProfileName;
}

/** Particle lifecycle phase (mirrors AT convention). */
export const enum FlowPhase {
  SPAWN = 0,
  FLOW  = 1,
  DECAY = 2,
  DEAD  = 3,
}

/** Per-particle state (CPU mirror). */
export interface FlowParticle {
  edgeIndex:  number;
  travel:     number;
  speed:      number;
  delay:      number;
  phase:      FlowPhase;
  alpha:      number;
  seed:       number;
  curlOff:    number;
  x:          number;
  y:          number;
}

/** Arrival callback — fired when a particle reaches the spline end. */
export type OnArrivalFn = (
  edgeId:   string,
  targetId: string,
  x:        number,
  y:        number,
) => void;

/** Configuration for EdgeFlowRenderer. */
export interface EdgeFlowRendererConfig {
  edges:              FlowEdge[];
  onArrival?:         OnArrivalFn;
  particlesPerUnit?:  number;
  maxParticles?:      number;
  speedScale?:        number;
  drawSplines?:       boolean;
  splineOpacity?:     number;
  glow?:              boolean;
  /** Show QoS health badges at edge midpoints (default false). */
  showQoSBadges?:     boolean;
  /** Trail history length for motion blur (0 = no trails, default 3). */
  trailHistory?:      number;
  /** Enable curl-noise lateral perturbation (default true). */
  curlNoise?:         boolean;
}

// ─── Catmull-Rom spline evaluation ────────────────────────────────────────────

function clampIdx(i: number, n: number): number {
  return Math.max(0, Math.min(n - 1, i));
}

/**
 * Evaluate a Catmull-Rom spline at normalised arc-length fraction u ∈ [0, 1].
 */
function evalCatmullRom(points: FlowPoint[], u: number): FlowPoint {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0 };
  if (n === 1) return { x: points[0].x, y: points[0].y };

  const sc  = Math.min(u, 0.9999) * (n - 1);
  const i1  = Math.floor(sc);
  const t   = sc - i1;

  const p0 = points[clampIdx(i1 - 1, n)];
  const p1 = points[clampIdx(i1,     n)];
  const p2 = points[clampIdx(i1 + 1, n)];
  const p3 = points[clampIdx(i1 + 2, n)];

  const t2 = t * t;
  const t3 = t2 * t;
  const h  = CR_TENSION;

  const f1 = -h * t3 + 2 * h * t2 - h * t;
  const f2 = (2 - h) * t3 + (h - 3) * t2 + 1;
  const f3 = (h - 2) * t3 + (3 - 2 * h) * t2 + h * t;
  const f4 = h * t3 - h * t2;

  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
  };
}

/**
 * Finite-difference tangent at normalised arc-length u.
 * Returns a normalised direction vector.
 */
function splineTangent(points: FlowPoint[], u: number): FlowPoint {
  const eps = 0.001;
  const a = evalCatmullRom(points, Math.max(0, u - eps));
  const b = evalCatmullRom(points, Math.min(1, u + eps));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-8) return { x: 1, y: 0 };
  return { x: dx / len, y: dy / len };
}

// ─── CPU curl-noise (simplified 2D, mirrors AT simplenoise.glsl) ─────────────

function hash2(x: number, y: number): number {
  return ((Math.sin(x * 127.1 + y * 311.7) * 43758.5453) % 1 + 1) % 1;
}

function noise2D(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  // Quintic fade
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);

  const n00 = hash2(ix,     iy);
  const n10 = hash2(ix + 1, iy);
  const n01 = hash2(ix,     iy + 1);
  const n11 = hash2(ix + 1, iy + 1);

  return n00 * (1 - ux) * (1 - uy) +
         n10 * ux       * (1 - uy) +
         n01 * (1 - ux) * uy       +
         n11 * ux       * uy;
}

/** 2D curl noise — returns divergence-free displacement (perpendicular). */
function curlNoise2DCPU(
  x: number, y: number, t: number, scale: number, speed: number,
): { dx: number; dy: number } {
  const px = x * scale;
  const py = y * scale;
  const pt = t * speed * 0.1;
  const e  = CURL_EPS * scale;

  const dndx = (noise2D(px + e, py + pt) - noise2D(px - e, py + pt)) / (2 * e);
  const dndy = (noise2D(px + pt, py + e) - noise2D(px + pt, py - e)) / (2 * e);

  // Curl: F = (∂Ψ/∂y, -∂Ψ/∂x)
  return { dx: dndy, dy: -dndx };
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

function themeForQoS(qos: QoSProfileName): ThemePalette {
  return QOS_THEME[qos] ?? QOS_THEME.DEFAULT;
}

function styleForQoS(qos: QoSProfileName): QoSFlowStyle {
  return QOS_FLOW_STYLE[qos] ?? QOS_FLOW_STYLE.DEFAULT;
}

/** Convert RGB [0,1] + alpha to CSS rgba string. */
function toCss(r: number, g: number, b: number, a: number): string {
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a.toFixed(3)})`;
}

/**
 * Resolve particle colour from QoS theme, travel progress, and metrics health.
 *
 * The colour pipeline:
 *   base colour → screen-blend highlight by travel (midpoint brightest)
 *   → health tint (degraded → desaturate, dead → grey-out)
 *   → alpha = sin(π·travel) bell curve × particle alpha
 */
function resolveFlowColor(
  theme:  ThemePalette,
  travel: number,
  alpha:  number,
  health: 0 | 1 | 2 = 2,
): { css: string; r: number; g: number; b: number; a: number } {
  // sin bell: bright at mid-travel, dim at endpoints
  const bell   = Math.sin(Math.PI * Math.min(Math.max(travel, 0), 1));
  const screen = bell * 0.6;

  // Screen-blend base with highlight
  let r = 1 - (1 - theme.base.r) * (1 - theme.highlight.r * screen);
  let g = 1 - (1 - theme.base.g) * (1 - theme.highlight.g * screen);
  let b = 1 - (1 - theme.base.b) * (1 - theme.highlight.b * screen);

  // Health-based desaturation
  if (health < 2) {
    const luma = r * 0.299 + g * 0.587 + b * 0.114;
    const desat = health === 0 ? 0.8 : 0.35;
    r = r + (luma - r) * desat;
    g = g + (luma - g) * desat;
    b = b + (luma - b) * desat;
  }

  const a = alpha * bell;

  return { css: toCss(r, g, b, a), r, g, b, a };
}

/**
 * QoS badge colour — aggregate health into a single indicator colour.
 */
function badgeColor(metrics: EdgeQoSMetrics): string {
  if (metrics.health === 2) return 'rgba(80,220,120,0.85)';  // green
  if (metrics.health === 1) return 'rgba(255,180,40,0.85)';  // amber
  return 'rgba(255,60,60,0.85)';                              // red
}

// ─── RNG ──────────────────────────────────────────────────────────────────────

function rng(seed: number, salt: number): number {
  return ((Math.sin(seed * 127.1 + salt * 311.7) * 43758.5453) % 1 + 1) % 1;
}

// ─── Trail history ring buffer ───────────────────────────────────────────────

interface TrailEntry {
  x: number;
  y: number;
  alpha: number;
}

class TrailRing {
  private buf: TrailEntry[];
  private head = 0;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.buf  = new Array(size);
    for (let i = 0; i < size; i++) {
      this.buf[i] = { x: 0, y: 0, alpha: 0 };
    }
  }

  push(x: number, y: number, alpha: number): void {
    this.buf[this.head] = { x, y, alpha };
    this.head = (this.head + 1) % this.size;
  }

  /** Iterate from oldest to newest. */
  *entries(): Generator<{ x: number; y: number; alpha: number; age: number }> {
    for (let i = 0; i < this.size; i++) {
      const idx = (this.head + i) % this.size;
      const e = this.buf[idx];
      if (e.alpha > 0.004) {
        yield { ...e, age: (this.size - i) / this.size };
      }
    }
  }
}

// ─── WGSL shaders (GPU path) ─────────────────────────────────────────────────

const GPU_UNIFORMS_WGSL = /* wgsl */`
struct FlowUniforms {
  time         : f32,
  dt           : f32,
  speedScale   : f32,
  particleCount: u32,
  edgeCount    : u32,
  texW         : u32,
  texH         : u32,
  domainW      : f32,
  domainH      : f32,
  scaleX       : f32,
  scaleY       : f32,
  enableCurl   : u32,
  _pad0        : u32,
  _pad1        : u32,
  _pad2        : u32,
  _pad3        : u32,
}
`;

const GPU_UNIFORMS_BYTE_SIZE = 64;

// Byte offsets
const UO_TIME           =  0;
const UO_DT             =  4;
const UO_SPEED_SCALE    =  8;
const UO_PARTICLE_COUNT = 12; // u32
const UO_EDGE_COUNT     = 16; // u32
const UO_TEX_W          = 20; // u32
const UO_TEX_H          = 24; // u32
const UO_DOMAIN_W       = 28;
const UO_DOMAIN_H       = 32;
const UO_SCALE_X        = 36;
const UO_SCALE_Y        = 40;
const UO_ENABLE_CURL    = 44; // u32

const GPU_QOS_WGSL = /* wgsl */`
// QoS style buffer — 5 profiles × 8 f32 each
// [speed_min, speed_max, decay_rate, max_delay, curl_str, curl_scale, curl_speed, radius]
const QOS_STRIDE = 8u;

fn qosGet(buf: ptr<storage, array<f32>, read>, qi: u32, field: u32) -> f32 {
  return (*buf)[qi * QOS_STRIDE + field];
}
`;

const GPU_SPLINE_WGSL = /* wgsl */`
const EDGE_HDR     = 4u;
const EDGE_PSTRIDE = 4u;
const EDGE_STRIDE_C = ${EDGE_STRIDE}u;
const EDGE_MAX_PTS_C = ${EDGE_MAX_PTS}u;

fn edgeNPts(buf: ptr<storage, array<f32>, read>, ei: u32) -> u32 {
  return u32((*buf)[ei * EDGE_STRIDE_C]);
}

fn edgeQoSIdx(buf: ptr<storage, array<f32>, read>, ei: u32) -> u32 {
  return u32((*buf)[ei * EDGE_STRIDE_C + 1u]);
}

fn edgePt(buf: ptr<storage, array<f32>, read>, ei: u32, pi: u32) -> vec2f {
  let b = ei * EDGE_STRIDE_C + EDGE_HDR + pi * EDGE_PSTRIDE;
  return vec2f((*buf)[b], (*buf)[b + 1u]);
}

fn catmullRom2D(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
  let t2 = t * t;
  let t3 = t2 * t;
  let h  = 0.5;
  let f1 = -h * t3 + 2.0 * h * t2 - h * t;
  let f2 = (2.0 - h) * t3 + (h - 3.0) * t2 + 1.0;
  let f3 = (h - 2.0) * t3 + (3.0 - 2.0 * h) * t2 + h * t;
  let f4 = h * t3 - h * t2;
  return f1 * p0 + f2 * p1 + f3 * p2 + f4 * p3;
}

fn clampI(i: i32, n: i32) -> i32 { return clamp(i, 0, n - 1); }

fn evalSpline2D(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec2f {
  let n = i32(edgeNPts(buf, ei));
  if (n == 0) { return vec2f(0.0); }
  if (n == 1) { return edgePt(buf, ei, 0u); }
  let sc = clamp(u, 0.0, 0.9999) * f32(n - 1);
  let i1 = i32(floor(sc));
  let lt = sc - f32(i1);
  return catmullRom2D(
    edgePt(buf, ei, u32(clampI(i1 - 1, n))),
    edgePt(buf, ei, u32(clampI(i1,     n))),
    edgePt(buf, ei, u32(clampI(i1 + 1, n))),
    edgePt(buf, ei, u32(clampI(i1 + 2, n))),
    lt,
  );
}

fn splineTan2D(buf: ptr<storage, array<f32>, read>, ei: u32, u: f32) -> vec2f {
  let eps = 0.001;
  let a = evalSpline2D(buf, ei, max(0.0, u - eps));
  let b = evalSpline2D(buf, ei, min(1.0, u + eps));
  let d = b - a;
  let l = length(d);
  if (l < 1e-8) { return vec2f(1.0, 0.0); }
  return d / l;
}
`;

const GPU_NOISE_WGSL = /* wgsl */`
fn hash2f(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise2D_g(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n00 = hash2f(i);
  let n10 = hash2f(i + vec2f(1.0, 0.0));
  let n01 = hash2f(i + vec2f(0.0, 1.0));
  let n11 = hash2f(i + vec2f(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

fn curlNoise2D_g(px: f32, py: f32, pt: f32, eps: f32) -> vec2f {
  let e = vec2f(eps, 0.0);
  let dndx = (noise2D_g(vec2f(px + eps, py + pt)) - noise2D_g(vec2f(px - eps, py + pt))) / (2.0 * eps);
  let dndy = (noise2D_g(vec2f(px + pt, py + eps)) - noise2D_g(vec2f(px + pt, py - eps))) / (2.0 * eps);
  return vec2f(dndy, -dndx);
}
`;

function buildComputeShader(): string {
  return /* wgsl */`
${GPU_UNIFORMS_WGSL}
${GPU_QOS_WGSL}
${GPU_SPLINE_WGSL}
${GPU_NOISE_WGSL}

@group(0) @binding(0) var<uniform>             uni  : FlowUniforms;
@group(1) @binding(0) var<storage, read>       edges: array<f32>;
@group(1) @binding(1) var<storage, read_write> parts: array<f32>;
@group(1) @binding(2) var<storage, read>       qos  : array<f32>;
@group(1) @binding(3) var                      tPos : texture_storage_2d<rgba32float, write>;

const PS = ${P_STRIDE}u;

fn pGet(idx: u32, f: u32) -> f32 { return parts[idx * PS + f]; }
fn pSet(idx: u32, f: u32, v: f32) { parts[idx * PS + f] = v; }

fn prng(s: f32, salt: f32) -> f32 {
  return fract(sin(s * 127.1 + salt * 311.7) * 43758.5453);
}

fn respawn(idx: u32, t: f32) {
  let eIdx  = u32(pGet(idx, ${F_EDGE}u));
  let qi    = edgeQoSIdx(&edges, eIdx);
  let sMin  = qosGet(&qos, qi, 0u);
  let sMax  = qosGet(&qos, qi, 1u);
  let mxDly = qosGet(&qos, qi, 3u);

  let s     = f32(idx) * 1.618034 + t;
  let speed = mix(sMin, sMax, prng(s, 0.0));
  let delay = prng(s, 1.0) * mxDly;
  let phase = select(1.0, 0.0, delay > 0.001);
  let alpha = select(1.0, 0.0, delay > 0.001);

  pSet(idx, ${F_TRAVEL}u,   0.0);
  pSet(idx, ${F_SPEED}u,    speed);
  pSet(idx, ${F_DELAY}u,    delay);
  pSet(idx, ${F_PHASE}u,    phase);
  pSet(idx, ${F_ALPHA}u,    alpha);
  pSet(idx, ${F_SEED}u,     prng(s, 3.0) * 1000.0);
  pSet(idx, ${F_HANDOFF}u,  0.0);
  pSet(idx, ${F_CURL_OFF}u, 0.0);
}

@compute @workgroup_size(${GPU_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let dt = uni.dt;
  let t  = uni.time;

  var phase  = pGet(idx, ${F_PHASE}u);
  var travel = pGet(idx, ${F_TRAVEL}u);
  var speed  = pGet(idx, ${F_SPEED}u);
  var delay  = pGet(idx, ${F_DELAY}u);
  var alpha  = pGet(idx, ${F_ALPHA}u);
  var seed   = pGet(idx, ${F_SEED}u);
  var cOff   = pGet(idx, ${F_CURL_OFF}u);
  let eIdx   = u32(pGet(idx, ${F_EDGE}u));
  let qi     = edgeQoSIdx(&edges, eIdx);

  // DEAD → respawn
  if (phase == 3.0) {
    respawn(idx, t);
    phase  = pGet(idx, ${F_PHASE}u);
    travel = 0.0;
    speed  = pGet(idx, ${F_SPEED}u);
    delay  = pGet(idx, ${F_DELAY}u);
    alpha  = pGet(idx, ${F_ALPHA}u);
    seed   = pGet(idx, ${F_SEED}u);
    cOff   = 0.0;
    pSet(idx, ${F_HANDOFF}u, 0.0);
  }

  if (phase == 0.0) {
    // SPAWN — count down
    delay -= dt;
    if (delay <= 0.0) {
      phase = 1.0;
      alpha = 1.0;
      delay = 0.0;
    }
    pSet(idx, ${F_DELAY}u, delay);
    pSet(idx, ${F_PHASE}u, phase);
    pSet(idx, ${F_ALPHA}u, alpha);

  } else if (phase == 1.0) {
    // FLOW — advance along spline with QoS-scaled speed
    travel += speed * uni.speedScale * dt;

    var sp = evalSpline2D(&edges, eIdx, min(travel, 0.9999));

    // Curl-noise lateral perturbation (optional)
    if (uni.enableCurl == 1u) {
      let cStr   = qosGet(&qos, qi, 4u);
      let cScale = qosGet(&qos, qi, 5u);
      let cSpeed = qosGet(&qos, qi, 6u);

      let curl = curlNoise2D_g(
        sp.x * cScale, sp.y * cScale,
        t * cSpeed * 0.1 + seed * 0.001,
        0.01 * cScale,
      ) * cStr;

      let tan = splineTan2D(&edges, eIdx, min(travel, 0.9999));
      let perp = vec2f(-tan.y, tan.x);

      sp = sp + perp * (curl.x + curl.y) + perp * cOff * 0.5;
      cOff = curl.x * perp.x + curl.y * perp.y;
    }

    pSet(idx, ${F_TRAVEL}u,   travel);
    pSet(idx, ${F_POS_X}u,    sp.x);
    pSet(idx, ${F_POS_Y}u,    sp.y);
    pSet(idx, ${F_CURL_OFF}u, cOff);

    if (travel >= 1.0) {
      phase = 2.0;
      pSet(idx, ${F_PHASE}u,  phase);
      pSet(idx, ${F_HANDOFF}u, 1.0);
    }

  } else if (phase == 2.0) {
    // DECAY — fade at QoS-specific rate
    let dRate = qosGet(&qos, qi, 2u);
    alpha -= dRate * dt;
    if (alpha <= 0.0) {
      alpha = 0.0;
      phase = 3.0;
    }
    pSet(idx, ${F_ALPHA}u, alpha);
    pSet(idx, ${F_PHASE}u, phase);
  }

  // Write tPos: .r=x  .g=y  .b=travel  .a=alpha
  let posX = pGet(idx, ${F_POS_X}u);
  let posY = pGet(idx, ${F_POS_Y}u);
  let texX = i32(idx % uni.texW);
  let texY = i32(idx / uni.texW);
  textureStore(tPos, vec2<i32>(texX, texY), vec4f(posX, posY, travel, alpha));
}
`;
}

function buildVertexShader(): string {
  return /* wgsl */`
${GPU_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni      : FlowUniforms;
@group(0) @binding(1) var          tPos     : texture_2d<f32>;
@group(0) @binding(2) var<storage, read> qosColors : array<f32>;

struct VertOut {
  @builtin(position) pos     : vec4f,
  @location(0)       vUv     : vec2f,
  @location(1)       vAlpha  : f32,
  @location(2)       vTravel : f32,
  @location(3) @interpolate(flat) vQoS : u32,
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

  let worldX  = p.r;
  let worldY  = p.g;
  let travel  = p.b;
  let alpha   = p.a;

  let alive = select(0.0, 1.0, alpha > 0.004);

  // Size attenuates: full at midpoint, small at start/end
  let travelDecay = clamp(1.0 - travel * travel, 0.0, 1.0);
  let halfSize    = 4.0 * travelDecay * 0.5;

  let quadUV = QUAD[vi];
  let ndcX   = worldX * uni.scaleX - 1.0 + quadUV.x * halfSize * uni.scaleX;
  let ndcY   = worldY * uni.scaleY - 1.0 + quadUV.y * halfSize * uni.scaleY;

  var out: VertOut;
  out.pos     = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv     = quadUV;
  out.vAlpha  = alpha;
  out.vTravel = travel;
  out.vQoS    = 0u;
  return out;
}
`;
}

function buildFragmentShader(): string {
  return /* wgsl */`
${GPU_UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni : FlowUniforms;
@group(0) @binding(3) var<storage, read> qosColors : array<f32>;

struct FragIn {
  @location(0) vUv     : vec2f,
  @location(1) vAlpha  : f32,
  @location(2) vTravel : f32,
  @location(3) @interpolate(flat) vQoS : u32,
}

@fragment fn fs_main(in: FragIn) -> @location(0) vec4f {
  // Soft circular disk
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  let edge    = 1.0 - smoothstep(0.7, 1.0, r2);
  let fade    = sin(3.14159265 * clamp(in.vTravel, 0.0, 1.0));

  // QoS-driven colour (read base RGB from qosColors buffer)
  let qi = in.vQoS;
  let baseR = qosColors[qi * 3u];
  let baseG = qosColors[qi * 3u + 1u];
  let baseB = qosColors[qi * 3u + 2u];

  // Screen-blend with highlight at midpoint
  let screen = fade * 0.6;
  let col = vec3f(
    1.0 - (1.0 - baseR) * (1.0 - screen),
    1.0 - (1.0 - baseG) * (1.0 - screen),
    1.0 - (1.0 - baseB) * (1.0 - screen),
  );

  let finalA = in.vAlpha * fade * edge;
  return vec4f(col * finalA, finalA);
}
`;
}

// ─── GPU resource builder helpers ────────────────────────────────────────────

function buildEdgeBuf(edges: FlowEdge[]): Float32Array {
  const buf = new Float32Array(edges.length * EDGE_STRIDE);
  for (let e = 0; e < edges.length; e++) {
    const base = e * EDGE_STRIDE;
    const pts  = edges[e].points;
    buf[base]     = Math.min(pts.length, EDGE_MAX_PTS);
    buf[base + 1] = QOS_NAME_TO_IDX[edges[e].qos ?? 'DEFAULT'];
    // points starting at offset 4, stride 4 (x, y, 0, 0)
    for (let p = 0; p < Math.min(pts.length, EDGE_MAX_PTS); p++) {
      const pb = base + 4 + p * 4;
      buf[pb + 0] = pts[p].x;
      buf[pb + 1] = pts[p].y;
    }
  }
  return buf;
}

function buildQoSStyleBuf(): Float32Array {
  // 5 profiles × 8 f32 = 40 f32
  const names: QoSProfileName[] = ['DEFAULT', 'SENSOR_DATA', 'PARAMETERS', 'TF_STATIC', 'TOPO_CHANGE'];
  const buf = new Float32Array(names.length * 8);
  for (let i = 0; i < names.length; i++) {
    const s = QOS_FLOW_STYLE[names[i]];
    const b = i * 8;
    buf[b + 0] = s.speed[0];
    buf[b + 1] = s.speed[1];
    buf[b + 2] = s.decayRate;
    buf[b + 3] = s.maxDelay;
    buf[b + 4] = s.curlStr;
    buf[b + 5] = s.curlScale;
    buf[b + 6] = s.curlSpeed;
    buf[b + 7] = s.radius;
  }
  return buf;
}

function buildQoSColorBuf(): Float32Array {
  const names: QoSProfileName[] = ['DEFAULT', 'SENSOR_DATA', 'PARAMETERS', 'TF_STATIC', 'TOPO_CHANGE'];
  const buf = new Float32Array(names.length * 3);
  for (let i = 0; i < names.length; i++) {
    const t = QOS_THEME[names[i]];
    buf[i * 3 + 0] = t.base.r;
    buf[i * 3 + 1] = t.base.g;
    buf[i * 3 + 2] = t.base.b;
  }
  return buf;
}

// ─── EdgeFlowRenderer — Unified dual-mode class ──────────────────────────────

/**
 * EdgeFlowRenderer
 *
 * Dual-mode edge-flow particle system. Particles travel along Catmull-Rom
 * splines defined by topology edges; speed, colour, trail, and curl-noise
 * behaviour are all derived from each edge's QoS profile.
 *
 * Supports both GPU (WebGPU compute + render) and CPU (Canvas2D) paths.
 *
 * @example
 * ```ts
 * // CPU mode:
 * const renderer = EdgeFlowRenderer.createCPU(ctx2d, {
 *   edges: [{
 *     edgeId: 'attn→ffn', sourceId: 'attn', targetId: 'ffn',
 *     points: [{ x: 100, y: 200 }, { x: 300, y: 200 }],
 *     weight: 1.0, qos: 'SENSOR_DATA',
 *   }],
 * });
 *
 * function frame(t: number) {
 *   ctx.clearRect(0, 0, canvas.width, canvas.height);
 *   renderer.tick(null, t / 1000, 1 / 60);
 *   renderer.draw();
 *   requestAnimationFrame(frame);
 * }
 * ```
 */
export class EdgeFlowRenderer {
  // ── Mode ─────────────────────────────────────────────────────────────────
  private readonly mode: 'gpu' | 'cpu';

  // ── Shared state ─────────────────────────────────────────────────────────
  private edges:           FlowEdge[];
  private readonly onArrival?: OnArrivalFn;
  private particlesPerUnit: number;
  private maxParticles:    number;
  private speedScale:      number;
  private drawSplines:     boolean;
  private splineOpacity:   number;
  private glow:            boolean;
  private showQoSBadges:   boolean;
  private enableCurl:      boolean;
  private trailHistoryLen: number;

  private elapsed = 0;

  // ── CPU state ────────────────────────────────────────────────────────────
  private ctx?:        CanvasRenderingContext2D;
  private particles:   FlowParticle[] = [];
  private trails:      TrailRing[]    = [];

  // ── GPU state ────────────────────────────────────────────────────────────
  private device?:          GPUDevice;
  private canvas?:          HTMLCanvasElement;
  private gpuBuilt          = false;
  private particleCount     = 0;
  private uniformBuf?:      GPUBuffer;
  private edgeBufGPU?:      GPUBuffer;
  private particleBufGPU?:  GPUBuffer;
  private readbackBuf?:     GPUBuffer;
  private qosStyleBuf?:     GPUBuffer;
  private qosColorBuf?:     GPUBuffer;
  private tPos?:            GPUTexture;
  private tPosView?:        GPUTextureView;
  private computePipeline?: GPUComputePipeline;
  private renderPipeline?:  GPURenderPipeline;
  private computeBG0?:      GPUBindGroup;
  private computeBG1?:      GPUBindGroup;
  private renderBG?:        GPUBindGroup;

  // ── QoS metrics per edge ─────────────────────────────────────────────────
  private metricsMap: Map<string, EdgeQoSMetrics> = new Map();

  // ── Private constructor (use static factories) ──────────────────────────

  private constructor(
    mode:   'gpu' | 'cpu',
    config: EdgeFlowRendererConfig,
  ) {
    this.mode             = mode;
    this.edges            = config.edges;
    this.onArrival        = config.onArrival;
    this.particlesPerUnit = config.particlesPerUnit ?? DEFAULT_PARTICLES_PER_UNIT;
    this.maxParticles     = config.maxParticles     ?? MAX_FLOW_PARTICLES;
    this.speedScale       = config.speedScale       ?? 1.0;
    this.drawSplines      = config.drawSplines      ?? false;
    this.splineOpacity    = config.splineOpacity    ?? 0.15;
    this.glow             = config.glow             ?? true;
    this.showQoSBadges    = config.showQoSBadges    ?? false;
    this.enableCurl       = config.curlNoise         ?? true;
    this.trailHistoryLen  = config.trailHistory      ?? 3;
  }

  // ── Static factories ───────────────────────────────────────────────────────

  /**
   * Create a CPU-mode renderer using Canvas2D.
   */
  static createCPU(
    ctx:    CanvasRenderingContext2D,
    config: EdgeFlowRendererConfig,
  ): EdgeFlowRenderer {
    const r  = new EdgeFlowRenderer('cpu', config);
    r.ctx    = ctx;
    r.particles = r._initParticles();
    r.trails    = r.particles.map(() => new TrailRing(r.trailHistoryLen));
    return r;
  }

  /**
   * Create a GPU-mode renderer using WebGPU.
   * Call `await renderer.build()` before first use.
   */
  static createGPU(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    config: EdgeFlowRendererConfig,
  ): EdgeFlowRenderer {
    const r    = new EdgeFlowRenderer('gpu', config);
    r.device   = device;
    r.canvas   = canvas;
    return r;
  }

  // ── GPU build ──────────────────────────────────────────────────────────────

  async build(): Promise<void> {
    if (this.mode !== 'gpu' || !this.device) return;

    if (this.gpuBuilt) this._destroyGPU();
    const device = this.device;

    this.particleCount = Math.min(
      this.maxParticles,
      Math.max(64,
        this.edges.reduce((n, e) => n + Math.ceil(e.weight * this.particlesPerUnit), 0),
      ),
    );

    // Uniform buffer
    this.uniformBuf = device.createBuffer({
      size:  GPU_UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Edge buffer
    const edgeData = buildEdgeBuf(this.edges);
    this.edgeBufGPU = device.createBuffer({
      size:  Math.max(edgeData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.edgeBufGPU, 0, edgeData);

    // Particle state buffer
    const pData = this._initParticleBufGPU();
    this.particleBufGPU = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.particleBufGPU, 0, pData);

    // Readback buffer
    this.readbackBuf = device.createBuffer({
      size:  pData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // QoS style buffer
    const qosData = buildQoSStyleBuf();
    this.qosStyleBuf = device.createBuffer({
      size:  Math.max(qosData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.qosStyleBuf, 0, qosData);

    // QoS colour buffer
    const qosCData = buildQoSColorBuf();
    this.qosColorBuf = device.createBuffer({
      size:  Math.max(qosCData.byteLength, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.qosColorBuf, 0, qosCData);

    // tPos texture
    this.tPos = device.createTexture({
      size:   [GPU_TEX_W, GPU_TEX_H],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING  |
              GPUTextureUsage.COPY_SRC,
    });
    this.tPosView = this.tPos.createView();

    // Compute pipeline
    const computeMod = device.createShaderModule({ code: buildComputeShader() });
    this.computePipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: computeMod, entryPoint: 'main' },
    });

    // Render pipeline
    const vsMod = device.createShaderModule({ code: buildVertexShader() });
    const fsMod = device.createShaderModule({ code: buildFragmentShader() });
    const fmt   = navigator.gpu.getPreferredCanvasFormat();

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

    // Bind groups
    this._buildBindGroups();
    this.gpuBuilt = true;

    console.log(
      `[EdgeFlowRenderer GPU] built: ${this.edges.length} edges, ` +
      `${this.particleCount} particles`,
    );
  }

  // ── Tick (advance particles) ───────────────────────────────────────────────

  /**
   * Advance all particles.
   *
   * @param encoder  — GPUCommandEncoder (GPU mode) or null (CPU mode)
   * @param elapsed  — total elapsed seconds
   * @param dt       — frame delta in seconds
   */
  tick(
    encoder: GPUCommandEncoder | null,
    elapsed: number,
    dt:      number,
  ): void {
    this.elapsed = elapsed;

    if (this.mode === 'gpu' && encoder && this.gpuBuilt) {
      this._tickGPU(encoder, elapsed, dt);
    } else {
      this._tickCPU(dt, elapsed);
    }
  }

  // ── GPU render pass ────────────────────────────────────────────────────────

  /**
   * Encode the render pass (GPU mode only).
   */
  render(
    encoder:   GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView?: GPUTextureView,
  ): void {
    if (!this.gpuBuilt || !this.renderPipeline || !this.renderBG) return;

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
    pass.draw(6, this.particleCount);
    pass.end();
  }

  // ── CPU draw ───────────────────────────────────────────────────────────────

  /**
   * Draw particles to Canvas2D (CPU mode).
   */
  draw(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;

    // Spline paths
    if (this.drawSplines) {
      this._drawSplinePaths();
    }

    // Particle trails (motion blur)
    if (this.trailHistoryLen > 0) {
      this._drawTrails();
    }

    // Main particles
    if (this.glow) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
    }

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      if (p.phase === FlowPhase.DEAD || p.phase === FlowPhase.SPAWN) continue;
      if (p.alpha < 0.004) continue;

      const edge = this.edges[p.edgeIndex];
      if (!edge) continue;

      const qos     = edge.qos ?? 'DEFAULT';
      const theme   = themeForQoS(qos);
      const style   = styleForQoS(qos);
      const metrics = this.metricsMap.get(edge.edgeId) ?? METRICS_DEFAULT;
      const color   = resolveFlowColor(theme, p.travel, p.alpha, metrics.health);

      if (color.a < 0.004) continue;

      const r = style.radius;

      // Core particle
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color.css;
      ctx.fill();

      // Outer glow halo
      if (this.glow) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = toCss(color.r, color.g, color.b, color.a * 0.2);
        ctx.fill();
      }
    }

    if (this.glow) {
      ctx.restore();
    }

    // QoS health badges
    if (this.showQoSBadges) {
      this._drawQoSBadges();
    }
  }

  // ── QoS metrics reporting ──────────────────────────────────────────────────

  /**
   * Report live QoS metrics for an edge. These drive the health-tint on
   * particle colours and the optional badge overlay.
   */
  reportQoSMetrics(edgeId: string, metrics: Partial<EdgeQoSMetrics>): void {
    const existing = this.metricsMap.get(edgeId) ?? { ...METRICS_DEFAULT };
    this.metricsMap.set(edgeId, {
      mps:         metrics.mps         ?? existing.mps,
      latencyMs:   metrics.latencyMs   ?? existing.latencyMs,
      dropCount:   metrics.dropCount   ?? existing.dropCount,
      utilisation: metrics.utilisation  ?? existing.utilisation,
      health:      metrics.health      ?? existing.health,
    });
  }

  /**
   * Batch-report metrics for multiple edges.
   */
  reportQoSMetricsBatch(entries: Array<{ edgeId: string } & Partial<EdgeQoSMetrics>>): void {
    for (const e of entries) {
      this.reportQoSMetrics(e.edgeId, e);
    }
  }

  /** Get current metrics for an edge. */
  getQoSMetrics(edgeId: string): EdgeQoSMetrics | undefined {
    return this.metricsMap.get(edgeId);
  }

  // ── Handoff readback (GPU mode) ────────────────────────────────────────────

  /**
   * Async readback of handoff flags (GPU mode).
   * Safe to call fire-and-forget each frame.
   */
  async scheduleHandoffReadback(): Promise<void> {
    if (!this.gpuBuilt || !this.onArrival || !this.device || !this.particleBufGPU || !this.readbackBuf) return;

    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBufGPU, 0, this.readbackBuf, 0, this.particleBufGPU.size);
    this.device.queue.submit([enc.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange());

    for (let i = 0; i < this.particleCount; i++) {
      const b = i * P_STRIDE;
      if (data[b + F_HANDOFF] < 0.5) continue;

      const eIdx = Math.round(data[b + F_EDGE]);
      const edge = this.edges[eIdx];
      if (!edge) continue;

      this.onArrival(edge.edgeId, edge.targetId, data[b + F_POS_X], data[b + F_POS_Y]);
    }

    this.readbackBuf.unmap();
  }

  // ── Edge management ────────────────────────────────────────────────────────

  /** Replace edges at runtime. Reinitialises particles. */
  setEdges(edges: FlowEdge[]): void {
    this.edges = edges;
    if (this.mode === 'cpu') {
      this.particles = this._initParticles();
      this.trails    = this.particles.map(() => new TrailRing(this.trailHistoryLen));
    }
  }

  /** Replace edges (GPU mode) — triggers full GPU rebuild. */
  async setEdgesGPU(edges: FlowEdge[]): Promise<void> {
    this.edges = edges;
    await this.build();
  }

  /** Update QoS for a specific edge by ID. */
  setEdgeQoS(edgeId: string, qos: QoSProfileName): void {
    const edge = this.edges.find(e => e.edgeId === edgeId);
    if (edge) edge.qos = qos;
  }

  /** Global speed multiplier. */
  setSpeedScale(v: number): void { this.speedScale = v; }

  /** Toggle spline path drawing. */
  setDrawSplines(v: boolean): void { this.drawSplines = v; }

  /** Toggle additive glow. */
  setGlow(v: boolean): void { this.glow = v; }

  /** Toggle QoS health badges. */
  setShowQoSBadges(v: boolean): void { this.showQoSBadges = v; }

  /** Toggle curl-noise perturbation. */
  setCurlNoise(v: boolean): void { this.enableCurl = v; }

  /** Read-only snapshot of CPU particles (CPU mode only). */
  get snapshot(): readonly FlowParticle[] { return this.particles; }

  /** Number of active (non-DEAD) particles. */
  get activeCount(): number {
    if (this.mode === 'cpu') {
      return this.particles.filter(p => p.phase !== FlowPhase.DEAD).length;
    }
    return this.particleCount;
  }

  /** Total particle slots. */
  get totalSlots(): number {
    return this.mode === 'cpu' ? this.particles.length : this.particleCount;
  }

  /** Destroy GPU resources. */
  destroy(): void {
    if (this.mode === 'gpu') this._destroyGPU();
  }

  // ── Private: CPU tick ──────────────────────────────────────────────────────

  private _tickCPU(dt: number, elapsed: number): void {
    for (let i = 0; i < this.particles.length; i++) {
      const p    = this.particles[i];
      const edge = this.edges[p.edgeIndex];
      if (!edge) continue;

      const style   = styleForQoS(edge.qos ?? 'DEFAULT');
      const metrics = this.metricsMap.get(edge.edgeId) ?? METRICS_DEFAULT;

      // Health-based speed penalty: degraded = 70%, dead = 40%
      const healthMul = metrics.health === 2 ? 1.0 : metrics.health === 1 ? 0.7 : 0.4;

      switch (p.phase) {
        case FlowPhase.DEAD:
          this._respawn(p, i, elapsed);
          break;

        case FlowPhase.SPAWN:
          p.delay -= dt;
          if (p.delay <= 0) {
            p.phase = FlowPhase.FLOW;
            p.alpha = 1;
            p.delay = 0;
          }
          break;

        case FlowPhase.FLOW: {
          const spd = p.speed * this.speedScale * healthMul;
          p.travel += spd * dt;

          let pos = evalCatmullRom(edge.points, Math.min(p.travel, 0.9999));

          // Curl-noise lateral perturbation
          if (this.enableCurl) {
            const curl = curlNoise2DCPU(
              pos.x, pos.y, elapsed + p.seed * 0.001,
              style.curlScale, style.curlSpeed,
            );
            const tan  = splineTangent(edge.points, Math.min(p.travel, 0.9999));
            const perpX = -tan.y;
            const perpY =  tan.x;
            const offset = (curl.dx + curl.dy) * style.curlStr;

            pos = {
              x: pos.x + perpX * offset + perpX * p.curlOff * 0.5,
              y: pos.y + perpY * offset + perpY * p.curlOff * 0.5,
            };
            p.curlOff = curl.dx * perpX + curl.dy * perpY;
          }

          p.x = pos.x;
          p.y = pos.y;

          // Record trail
          if (this.trailHistoryLen > 0 && this.trails[i]) {
            this.trails[i].push(p.x, p.y, p.alpha);
          }

          if (p.travel >= 1.0) {
            p.phase = FlowPhase.DECAY;
            if (this.onArrival) {
              this.onArrival(edge.edgeId, edge.targetId, p.x, p.y);
            }
          }
          break;
        }

        case FlowPhase.DECAY:
          p.alpha -= style.decayRate * dt;
          if (p.alpha <= 0) {
            p.alpha = 0;
            p.phase = FlowPhase.DEAD;
          }
          break;
      }
    }
  }

  // ── Private: GPU tick ──────────────────────────────────────────────────────

  private _tickGPU(encoder: GPUCommandEncoder, elapsed: number, dt: number): void {
    if (!this.device || !this.uniformBuf || !this.computePipeline || !this.computeBG0 || !this.computeBG1) return;

    this._writeUniforms(elapsed, dt);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBG0);
    pass.setBindGroup(1, this.computeBG1);
    pass.dispatchWorkgroups(Math.ceil(this.particleCount / GPU_WG));
    pass.end();
  }

  // ── Private: init particles (CPU) ──────────────────────────────────────────

  private _initParticles(): FlowParticle[] {
    const particles: FlowParticle[] = [];
    let slot = 0;

    for (let e = 0; e < this.edges.length && slot < this.maxParticles; e++) {
      const edge  = this.edges[e];
      const style = styleForQoS(edge.qos ?? 'DEFAULT');
      const count = Math.min(
        Math.ceil(edge.weight * this.particlesPerUnit),
        this.maxParticles - slot,
      );

      for (let p = 0; p < count && slot < this.maxParticles; p++, slot++) {
        const s     = slot * 1.618034;
        const speed = style.speed[0] + rng(s, 0) * (style.speed[1] - style.speed[0]);
        const delay = rng(s, 1) * style.maxDelay;
        const startPos = evalCatmullRom(edge.points, 0);

        particles.push({
          edgeIndex: e,
          travel:    0,
          speed,
          delay,
          phase:     delay > 0.001 ? FlowPhase.SPAWN : FlowPhase.FLOW,
          alpha:     delay > 0.001 ? 0 : 1,
          seed:      rng(s, 3) * 1000,
          curlOff:   0,
          x:         startPos.x,
          y:         startPos.y,
        });
      }
    }

    return particles;
  }

  private _respawn(p: FlowParticle, index: number, t: number): void {
    const edge = this.edges[p.edgeIndex];
    if (!edge) return;

    const style = styleForQoS(edge.qos ?? 'DEFAULT');
    const s     = index * 1.618034 + t;
    const speed = style.speed[0] + rng(s, 0) * (style.speed[1] - style.speed[0]);
    const delay = rng(s, 1) * style.maxDelay;
    const startPos = evalCatmullRom(edge.points, 0);

    p.travel  = 0;
    p.speed   = speed;
    p.delay   = delay;
    p.phase   = delay > 0.001 ? FlowPhase.SPAWN : FlowPhase.FLOW;
    p.alpha   = delay > 0.001 ? 0 : 1;
    p.seed    = rng(s, 3) * 1000;
    p.curlOff = 0;
    p.x       = startPos.x;
    p.y       = startPos.y;
  }

  // ── Private: init particle buffer (GPU) ────────────────────────────────────

  private _initParticleBufGPU(): Float32Array {
    const buf  = new Float32Array(this.particleCount * P_STRIDE);
    let   slot = 0;

    for (let e = 0; e < this.edges.length && slot < this.particleCount; e++) {
      const edge  = this.edges[e];
      const style = styleForQoS(edge.qos ?? 'DEFAULT');
      const count = Math.min(
        Math.ceil(edge.weight * this.particlesPerUnit),
        this.particleCount - slot,
      );
      for (let p = 0; p < count && slot < this.particleCount; p++, slot++) {
        const b     = slot * P_STRIDE;
        const s     = slot * 1.618034;
        const speed = style.speed[0] + rng(s, 0) * (style.speed[1] - style.speed[0]);
        const delay = rng(s, 1) * style.maxDelay;
        const start = evalCatmullRom(edge.points, 0);

        buf[b + F_TRAVEL]   = 0;
        buf[b + F_SPEED]    = speed;
        buf[b + F_DELAY]    = delay;
        buf[b + F_PHASE]    = delay > 0.001 ? 0 : 1;
        buf[b + F_ALPHA]    = delay > 0.001 ? 0 : 1;
        buf[b + F_SEED]     = rng(s, 3) * 1000;
        buf[b + F_EDGE]     = e;
        buf[b + F_POS_X]    = start.x;
        buf[b + F_POS_Y]    = start.y;
        buf[b + F_HANDOFF]  = 0;
        buf[b + F_QOS_IDX]  = QOS_NAME_TO_IDX[edge.qos ?? 'DEFAULT'];
        buf[b + F_CURL_OFF] = 0;
      }
    }

    // Remaining slots start as DEAD
    for (; slot < this.particleCount; slot++) {
      buf[slot * P_STRIDE + F_PHASE] = 3;
    }

    return buf;
  }

  // ── Private: write GPU uniforms ────────────────────────────────────────────

  private _writeUniforms(elapsed: number, dt: number): void {
    if (!this.device || !this.uniformBuf || !this.canvas) return;

    const dw = this.canvas.width  || 1;
    const dh = this.canvas.height || 1;
    const data = new Float32Array(GPU_UNIFORMS_BYTE_SIZE / 4);

    data[UO_TIME           / 4] = elapsed;
    data[UO_DT             / 4] = dt;
    data[UO_SPEED_SCALE    / 4] = this.speedScale;
    data[UO_DOMAIN_W       / 4] = dw;
    data[UO_DOMAIN_H       / 4] = dh;
    data[UO_SCALE_X        / 4] = 2.0 / dw;
    data[UO_SCALE_Y        / 4] = 2.0 / dh;

    const u32 = new Uint32Array(data.buffer);
    u32[UO_PARTICLE_COUNT / 4] = this.particleCount;
    u32[UO_EDGE_COUNT     / 4] = this.edges.length;
    u32[UO_TEX_W          / 4] = GPU_TEX_W;
    u32[UO_TEX_H          / 4] = GPU_TEX_H;
    u32[UO_ENABLE_CURL    / 4] = this.enableCurl ? 1 : 0;

    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  // ── Private: build GPU bind groups ─────────────────────────────────────────

  private _buildBindGroups(): void {
    if (!this.device || !this.computePipeline || !this.renderPipeline) return;
    const device = this.device;

    // Compute BG0 — uniforms
    this.computeBG0 = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf! } },
      ],
    });

    // Compute BG1 — edges + particles + qos styles + tPos write
    this.computeBG1 = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.edgeBufGPU! } },
        { binding: 1, resource: { buffer: this.particleBufGPU! } },
        { binding: 2, resource: { buffer: this.qosStyleBuf! } },
        { binding: 3, resource: this.tPosView! },
      ],
    });

    // Render BG — uniforms + tPos read + qos colours
    this.renderBG = device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf! } },
        { binding: 1, resource: this.tPosView! },
        { binding: 2, resource: { buffer: this.qosColorBuf! } },
      ],
    });
  }

  // ── Private: Canvas2D drawing helpers ──────────────────────────────────────

  private _drawTrails(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      if (p.phase !== FlowPhase.FLOW) continue;

      const edge = this.edges[p.edgeIndex];
      if (!edge) continue;

      const theme = themeForQoS(edge.qos ?? 'DEFAULT');
      const style = styleForQoS(edge.qos ?? 'DEFAULT');
      const trail = this.trails[i];
      if (!trail) continue;

      for (const entry of trail.entries()) {
        const fadeA = entry.alpha * (1 - entry.age) * 0.4;
        if (fadeA < 0.004) continue;

        const r = style.radius * (1 - entry.age * 0.6);
        ctx.beginPath();
        ctx.arc(entry.x, entry.y, r, 0, Math.PI * 2);
        ctx.fillStyle = toCss(theme.base.r, theme.base.g, theme.base.b, fadeA);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  private _drawSplinePaths(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const SEGMENTS = 32;

    for (const edge of this.edges) {
      if (edge.points.length < 2) continue;

      const theme = themeForQoS(edge.qos ?? 'DEFAULT');
      const { base } = theme;

      ctx.beginPath();
      const p0 = evalCatmullRom(edge.points, 0);
      ctx.moveTo(p0.x, p0.y);

      for (let s = 1; s <= SEGMENTS; s++) {
        const u  = s / SEGMENTS;
        const pt = evalCatmullRom(edge.points, u);
        ctx.lineTo(pt.x, pt.y);
      }

      ctx.strokeStyle = toCss(base.r, base.g, base.b, this.splineOpacity);
      ctx.lineWidth   = 1;
      ctx.stroke();
    }
  }

  private _drawQoSBadges(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;

    for (const edge of this.edges) {
      if (edge.points.length < 2) continue;

      const metrics = this.metricsMap.get(edge.edgeId);
      if (!metrics) continue;

      // Badge position: spline midpoint
      const mid = evalCatmullRom(edge.points, 0.5);

      // Badge background
      const bw = 28;
      const bh = 10;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.beginPath();
      ctx.roundRect(mid.x - bw / 2, mid.y - bh / 2, bw, bh, 3);
      ctx.fill();

      // Health indicator dot
      ctx.beginPath();
      ctx.arc(mid.x - 7, mid.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = badgeColor(metrics);
      ctx.fill();

      // Utilisation bar
      const barW = 12;
      const barH = 4;
      const barX = mid.x - 1;
      const barY = mid.y - barH / 2;

      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.fillRect(barX, barY, barW, barH);

      ctx.fillStyle = badgeColor(metrics);
      ctx.fillRect(barX, barY, barW * Math.min(metrics.utilisation, 1), barH);
    }
  }

  // ── Private: destroy GPU resources ─────────────────────────────────────────

  private _destroyGPU(): void {
    if (!this.gpuBuilt) return;
    this.uniformBuf?.destroy();
    this.edgeBufGPU?.destroy();
    this.particleBufGPU?.destroy();
    this.readbackBuf?.destroy();
    this.qosStyleBuf?.destroy();
    this.qosColorBuf?.destroy();
    this.tPos?.destroy();
    this.gpuBuilt = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create an EdgeFlowRenderer (CPU) with QoS profiles resolved from a map.
 *
 * @example
 * ```ts
 * const renderer = createEdgeFlowRenderer(ctx, rawEdges, {
 *   'attn→ffn':      'SENSOR_DATA',
 *   'ffn→layernorm': 'PARAMETERS',
 * });
 * ```
 */
export function createEdgeFlowRenderer(
  ctx:       CanvasRenderingContext2D,
  edges:     Omit<FlowEdge, 'qos'>[],
  qosMap:    Record<string, QoSProfileName>,
  config:    Omit<EdgeFlowRendererConfig, 'edges'> = {},
): EdgeFlowRenderer {
  const resolved: FlowEdge[] = edges.map(e => ({
    ...e,
    qos: qosMap[e.edgeId] ?? 'DEFAULT',
  }));
  return EdgeFlowRenderer.createCPU(ctx, { ...config, edges: resolved });
}

/**
 * Create an EdgeFlowRenderer (CPU) wired to SPH world's `addFluid` method.
 */
export function createEdgeFlowForSPH(
  ctx:       CanvasRenderingContext2D,
  edges:     FlowEdge[],
  addFluid:  (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:    Omit<EdgeFlowRendererConfig, 'edges' | 'onArrival'> = {},
): EdgeFlowRenderer {
  const R = 0.05;
  return EdgeFlowRenderer.createCPU(ctx, {
    ...config,
    edges,
    onArrival: (_edgeId, _targetId, x, y) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, 0);
    },
  });
}

/**
 * Create an EdgeFlowRenderer (GPU) wired to SPH world's `addFluid` method.
 */
export function createEdgeFlowForSPHGPU(
  device:    GPUDevice,
  canvas:    HTMLCanvasElement,
  edges:     FlowEdge[],
  addFluid:  (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:    Omit<EdgeFlowRendererConfig, 'edges' | 'onArrival'> = {},
): EdgeFlowRenderer {
  const R = 0.05;
  return EdgeFlowRenderer.createGPU(device, canvas, {
    ...config,
    edges,
    onArrival: (_edgeId, _targetId, x, y) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, 0);
    },
  });
}

// ─── Standalone spline utilities (re-exported) ──────────────────────────────

export { evalCatmullRom, splineTangent };

// ─── Defaults re-export ──────────────────────────────────────────────────────

export const EDGE_FLOW_DEFAULTS = {
  particlesPerUnit:  DEFAULT_PARTICLES_PER_UNIT,
  maxFlowParticles:  MAX_FLOW_PARTICLES,
  crTension:         CR_TENSION,
  qosFlowStyles:     QOS_FLOW_STYLE,
  gpuTexW:           GPU_TEX_W,
  gpuTexH:           GPU_TEX_H,
  gpuWorkgroupSize:  GPU_WG,
} as const;
