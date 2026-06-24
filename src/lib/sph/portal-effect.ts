/**
 * portal-effect.ts — M799: Portal Warp Effect
 * ─────────────────────────────────────────────────────────────────────────────
 * When an Edge connects two distant Cells, the system renders portals at both
 * endpoints — rotating vortex rings, space-warp distortion, gravitational
 * light bending, and particles being sucked in at the source / ejected at the
 * target.  A stencil-buffer pass renders the "other side" view inside each
 * portal aperture, creating a sci-fi wormhole aesthetic.
 *
 * ── Visual anatomy ───────────────────────────────────────────────────────────
 *
 *   ┌─── Source Cell ─────────────────────────────────────────────────────────┐
 *   │                                                                         │
 *   │      ╭─────────╮                                                        │
 *   │    ╭─┤ VORTEX  ├─╮   ← Rotating ring of spiral arms (4–6 arms)        │
 *   │   ╭┤ ╰─────────╯ ├╮  ← Outer accretion disk with particle suck-in    │
 *   │   │╰──── ⊙ ───────╯│ ← Stencil aperture shows Target Cell's view     │
 *   │   ╰──── EVENT ──────╯ ← Space distortion halo (UV warp falloff)       │
 *   │         HORIZON       ← Light-bending ring at Schwarzschild radius    │
 *   │                                                                         │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─── Target Cell ────────────────────────────────────────────────────────┐
 *   │                                                                         │
 *   │      Same portal structure, but particles are EJECTED outward.          │
 *   │      Stencil aperture shows Source Cell's view.                         │
 *   │      Vortex spins in the opposite direction (exit wormhole).            │
 *   │                                                                         │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * ── Layer decomposition ──────────────────────────────────────────────────────
 *
 *   Layer 0 — Space Distortion (full-screen post-process pass)
 *     Radial UV displacement around each portal centre, with amplitude
 *     decaying as 1/r².  Produces a gravitational lensing halo — nearby
 *     fragments are pulled toward the portal, creating the "bending light"
 *     illusion.  Chromatic aberration splits R/G/B offsets along the radial
 *     direction for a prismatic rim.
 *
 *   Layer 1 — Event Horizon Ring
 *     A bright annular glow at the portal's Schwarzschild radius.  Rendered
 *     as a smoothstep ring with additive bloom, pulsing at the portal's
 *     heartbeat frequency.  Colour follows the edge's QoS theme.
 *
 *   Layer 2 — Vortex Arms (rotating spiral)
 *     4–6 logarithmic spiral arms rotating around the portal centre.
 *     Each arm is a particle trail with decreasing alpha from rim to centre.
 *     Source portals spin clockwise (inward); target portals spin counter-
 *     clockwise (outward).
 *
 *   Layer 3 — Accretion Particles
 *     Particles orbiting the portal in decaying spirals:
 *       Source — particles spiral INWARD and vanish at the aperture
 *       Target — particles spawn at the aperture and spiral OUTWARD
 *     Speed and density scale with edge traffic (QoS mps).
 *
 *   Layer 4 — Stencil Portal Interior
 *     A circular stencil mask at the portal aperture.  Inside the mask,
 *     the scene is rendered from the OTHER endpoint's viewpoint, creating
 *     the "window to another place" effect.  When no GPU stencil is
 *     available, falls back to a gradient-filled disc with the remote
 *     endpoint's QoS theme colour and a subtle noise texture.
 *
 * ── Distance threshold ──────────────────────────────────────────────────────
 *
 *   Portals activate only when the Euclidean distance between source and
 *   target cells exceeds `distanceThreshold` (default: 300 domain units).
 *   Below this, the edge renders normally via EdgeFlowRenderer / neural-
 *   pathway-renderer.  A smooth blend-in zone (±50 units) cross-fades
 *   the portal opacity to avoid pop-in.
 *
 * ── QoS → portal style ──────────────────────────────────────────────────────
 *
 *   SENSOR_DATA  — rapid vortex spin, tight aperture, blue accretion disc,
 *                  fast particle suck-in cadence, high chromatic aberration
 *   PARAMETERS   — moderate spin, warm amber glow, steady particle orbit,
 *                  pulsing heartbeat rhythm
 *   TF_STATIC    — slow majestic rotation, wide aperture, jade-green corona,
 *                  sparse long-lived orbital particles
 *   TOPO_CHANGE  — erratic spin with direction reversals, magenta flicker,
 *                  dense chaotic particle ejection, strong distortion
 *   DEFAULT      — balanced parameters, neutral slate colour
 *
 * ── Dual-mode rendering ─────────────────────────────────────────────────────
 *
 *   1. **Canvas2D** — Full software path.  Vortex arms and accretion
 *      particles drawn with globalCompositeOperation: 'lighter'.  Stencil
 *      emulated via ctx.clip() with a circular path.  Space distortion
 *      approximated as a radial gradient overlay.
 *
 *   2. **WebGPU** — Stencil buffer for portal interior.  A compute pass
 *      advances accretion particles.  The distortion pass runs as a
 *      full-screen post-process (shared WGSL with heat-distortion.ts
 *      architecture).
 *
 * ── Integration ──────────────────────────────────────────────────────────────
 *
 *   ```ts
 *   import { PortalEffectSystem } from '$lib/sph/portal-effect';
 *
 *   const portals = PortalEffectSystem.createCPU(ctx2d, {
 *     edges: topology.edges,
 *     distanceThreshold: 300,
 *   });
 *
 *   // render loop:
 *   portals.update(elapsed, dt);
 *   portals.draw();
 *
 *   // GPU mode:
 *   const portals = PortalEffectSystem.createGPU(device, canvas, {
 *     edges: topology.edges,
 *   });
 *   await portals.build();
 *   // per-frame:
 *   portals.update(elapsed, dt);
 *   portals.encodeDistortion(encoder, srcView, dstView, w, h);
 *   portals.encodeStencilInterior(encoder, colorView, depthStencilView);
 *   portals.encodeVortex(encoder, colorView);
 *   ```
 *
 * ── References ───────────────────────────────────────────────────────────────
 *
 *   src/lib/sph/edge-flow-renderer.ts      — edge spline / particle flow
 *   src/lib/sph/collision-shockwave.ts     — expanding ring UV distortion
 *   src/lib/sph/heat-distortion.ts         — full-screen UV warp post-process
 *   src/lib/sph/color-palette.ts           — QoS → colour theme mapping
 *   src/lib/sph/qosSpatial.ts              — QoS profile definitions
 *   src/lib/sph/particle-effect-system.ts  — particle pool pattern
 *   src/lib/sph/neural-pathway-renderer.ts — bio-visual edge rendering
 */

import type { QoSProfile }        from './types';
import type { QoSProfileName }    from './qosSpatial';
import { QOS_PRESETS }             from './qosSpatial';
import { QOS_THEME }               from './color-palette';
import type { ThemePalette, RGB }  from './color-palette';

// ─── Constants ────────────────────────────────────────────────────────────────

// [orphan-precise] /** Default distance (domain units) beyond which portals activate. */
const DEFAULT_DISTANCE_THRESHOLD = 300;

/** Blend zone half-width for smooth portal fade-in/out. */
const BLEND_ZONE = 50;

/** Maximum portals rendered simultaneously. */
const MAX_PORTALS = 32;

/** Number of spiral arms per vortex. */
const VORTEX_ARM_COUNT = 5;

/** Maximum accretion particles per portal. */
const MAX_ACCRETION_PARTICLES = 64;

/** Total particle pool across all portals. */
const TOTAL_PARTICLE_POOL = MAX_PORTALS * MAX_ACCRETION_PARTICLES;

/** Catmull-Rom tension for edge spline evaluation. */
const CR_TENSION = 0.5;

/** GPU workgroup size for compute passes. */
const GPU_WG = 64;

/** Stencil reference value for portal interior mask. */
const STENCIL_REF = 0x01;

// ─── QoS → portal style mapping ─────────────────────────────────────────────

export interface PortalStyle {
  /** Vortex rotation speed (radians / second). */
  spinSpeed:        number;
  /** Vortex spin direction multiplier (1 = CW source, -1 = CCW target). */
  spinDir:          number;
  /** Portal aperture radius as fraction of total portal radius [0.2, 0.7]. */
  apertureFrac:     number;
  /** Event horizon ring width (domain units). */
  horizonWidth:     number;
  /** Event horizon pulse frequency (Hz). */
  horizonPulseHz:   number;
  /** Accretion particle orbit speed (radians / second). */
  orbitSpeed:       number;
  /** Accretion particle radial in-spiral speed (domain units / second). */
  radialSpeed:      number;
  /** Accretion particle count per portal. */
  particleCount:    number;
  /** Space distortion amplitude (UV displacement magnitude at horizon). */
  distortionAmp:    number;
  /** Chromatic aberration spread at distortion rim [0, 0.05]. */
  chromaticSpread:  number;
  /** Vortex arm angular width (radians). */
  armWidth:         number;
  /** Vortex arm brightness [0, 1]. */
  armBrightness:    number;
}

const QOS_PORTAL_STYLE: Record<QoSProfileName, PortalStyle> = {
  SENSOR_DATA: {
    spinSpeed:       3.5,
    spinDir:         1,
    apertureFrac:    0.35,
    horizonWidth:    3.0,
    horizonPulseHz:  2.0,
    orbitSpeed:      4.0,
    radialSpeed:     35.0,
    particleCount:   48,
    distortionAmp:   0.035,
    chromaticSpread: 0.025,
    armWidth:        0.28,
    armBrightness:   0.85,
  },
  PARAMETERS: {
    spinSpeed:       1.8,
    spinDir:         1,
    apertureFrac:    0.45,
    horizonWidth:    4.0,
    horizonPulseHz:  0.8,
    orbitSpeed:      2.2,
    radialSpeed:     22.0,
    particleCount:   36,
    distortionAmp:   0.025,
    chromaticSpread: 0.015,
    armWidth:        0.35,
    armBrightness:   0.70,
  },
  TF_STATIC: {
    spinSpeed:       0.8,
    spinDir:         1,
    apertureFrac:    0.55,
    horizonWidth:    5.0,
    horizonPulseHz:  0.4,
    orbitSpeed:      1.0,
    radialSpeed:     12.0,
    particleCount:   24,
    distortionAmp:   0.018,
    chromaticSpread: 0.010,
    armWidth:        0.42,
    armBrightness:   0.55,
  },
  TOPO_CHANGE: {
    spinSpeed:       5.0,
    spinDir:         1,
    apertureFrac:    0.30,
    horizonWidth:    2.5,
    horizonPulseHz:  4.0,
    orbitSpeed:      6.0,
    radialSpeed:     50.0,
    particleCount:   56,
    distortionAmp:   0.045,
    chromaticSpread: 0.035,
    armWidth:        0.22,
    armBrightness:   0.95,
  },
  DEFAULT: {
    spinSpeed:       2.0,
    spinDir:         1,
    apertureFrac:    0.42,
    horizonWidth:    3.5,
    horizonPulseHz:  1.0,
    orbitSpeed:      2.5,
    radialSpeed:     25.0,
    particleCount:   32,
    distortionAmp:   0.028,
    chromaticSpread: 0.018,
    armWidth:        0.32,
    armBrightness:   0.72,
  },
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D point in domain space. */
export interface PortalPoint {
  x: number;
  y: number;
}

/** Spline control point for edge routing. */
export interface PortalFlowPoint {
  x: number;
  y: number;
}

/** Edge definition for the portal system. */
export interface PortalEdge {
  edgeId:    string;
  sourceId:  string;
  targetId:  string;
  /** Catmull-Rom control points (≥2, domain space). */
  points:    PortalFlowPoint[];
  weight:    number;
  qos?:      QoSProfileName;
}

/** Accretion particle phase. */
const enum AccretionPhase {
  ORBIT  = 0,   // spiralling toward/away from aperture
  FADE   = 1,   // fading after crossing aperture threshold
  DEAD   = 2,   // awaiting respawn
}

/** Per-particle state for accretion disc. */
interface AccretionParticle {
  /** Angle in polar coordinates (radians). */
  angle:    number;
  /** Radial distance from portal centre (domain units). */
  radius:   number;
  /** Angular velocity (radians / second). */
  omega:    number;
  /** Radial velocity (domain units / second, negative = inward). */
  vr:       number;
  /** Current lifecycle phase. */
  phase:    AccretionPhase;
  /** Alpha [0, 1]. */
  alpha:    number;
  /** Particle size (domain units). */
  size:     number;
  /** Random seed for visual variation. */
  seed:     number;
  /** Portal index this particle belongs to. */
  portalIdx: number;
  /** Is this a source-side (true) or target-side (false) particle? */
  isSource: boolean;
}

/** Runtime state for a single portal (one end of an edge). */
interface PortalInstance {
  /** Centre position in domain space. */
  x:           number;
  y:           number;
  /** Total portal visual radius (domain units). */
  radius:      number;
  /** Current vortex rotation angle (radians). */
  vortexAngle: number;
  /** Portal opacity [0, 1] — driven by distance blend zone. */
  opacity:     number;
  /** Is this the source end (true) or target end (false)? */
  isSource:    boolean;
  /** Edge reference. */
  edge:        PortalEdge;
  /** Resolved style. */
  style:       PortalStyle;
  /** QoS theme colours. */
  theme:       ThemePalette;
  /** Position of the OTHER end (for stencil interior rendering). */
  remoteX:     number;
  remoteY:     number;
}

/** Configuration for PortalEffectSystem. */
export interface PortalEffectConfig {
  edges:               PortalEdge[];
  /** Euclidean distance threshold to activate portals (default 300). */
  distanceThreshold?:  number;
  /** Portal visual radius (default 40 domain units). */
  portalRadius?:       number;
  /** Enable space distortion overlay (default true). */
  distortion?:         boolean;
  /** Enable stencil interior or fallback disc (default true). */
  stencilInterior?:    boolean;
  /** Enable accretion particles (default true). */
  accretionParticles?: boolean;
  /** Enable vortex arm rendering (default true). */
  vortexArms?:         boolean;
  /** Enable event horizon ring (default true). */
  eventHorizon?:       boolean;
  /** Global speed multiplier (default 1.0). */
  speedScale?:         number;
}

// ─── Spline evaluation ───────────────────────────────────────────────────────

function clampIdx(i: number, n: number): number {
  return Math.max(0, Math.min(n - 1, i));
}

function evalCatmullRom(points: PortalFlowPoint[], u: number): PortalPoint {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0 };
  if (n === 1) return { x: points[0].x, y: points[0].y };

  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  const t  = sc - i1;

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

// ─── Utility ──────────────────────────────────────────────────────────────────

function dist(ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  return Math.sqrt(dx * dx + dy * dy);
}

function rng(seed: number, salt: number): number {
  return ((Math.sin(seed * 127.1 + salt * 311.7) * 43758.5453) % 1 + 1) % 1;
}

function toCss(r: number, g: number, b: number, a: number): string {
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a.toFixed(3)})`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// ─── Simple 2-D hash noise for portal interior texture ───────────────────────

function hash2(x: number, y: number): number {
  return ((Math.sin(x * 127.1 + y * 311.7) * 43758.5453) % 1 + 1) % 1;
}

function noise2D(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
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

// ─── WebGPU WGSL: Space Distortion Post-Process ─────────────────────────────

const DISTORTION_WGSL = /* wgsl */`
// Portal space distortion — gravitational lensing + chromatic aberration
// Full-screen post-process pass: reads scene colour, writes warped result.

struct DistortionUniforms {
  width       : f32,
  height      : f32,
  portalCount : u32,
  time        : f32,
}

struct PortalData {
  // Per-portal: cx, cy, radius, amp, chromaticSpread, opacity, pulseHz, _pad
  data : array<vec4f, ${MAX_PORTALS * 2}>,  // 2 vec4 per portal = 8 floats
}

@group(0) @binding(0) var<uniform>       uni     : DistortionUniforms;
@group(0) @binding(1) var<storage, read> portals : PortalData;
@group(0) @binding(2) var                smp     : sampler;
@group(0) @binding(3) var                src     : texture_2d<f32>;

fn portalCx(i: u32)  -> f32 { return portals.data[i * 2u].x; }
fn portalCy(i: u32)  -> f32 { return portals.data[i * 2u].y; }
fn portalR(i: u32)   -> f32 { return portals.data[i * 2u].z; }
fn portalAmp(i: u32) -> f32 { return portals.data[i * 2u].w; }
fn portalChrom(i: u32) -> f32 { return portals.data[i * 2u + 1u].x; }
fn portalOpac(i: u32)  -> f32 { return portals.data[i * 2u + 1u].y; }
fn portalPulse(i: u32) -> f32 { return portals.data[i * 2u + 1u].z; }

@fragment fn fs_main(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let uv = vec2f(fragPos.x / uni.width, fragPos.y / uni.height);

  var totalOffset = vec2f(0.0);
  var totalChrom  = 0.0;
  var maxWeight   = 0.0;

  for (var i = 0u; i < uni.portalCount; i = i + 1u) {
    let cx = portalCx(i) / uni.width;
    let cy = portalCy(i) / uni.height;
    let r  = portalR(i)  / min(uni.width, uni.height);
    let amp   = portalAmp(i);
    let chrom = portalChrom(i);
    let opac  = portalOpac(i);
    let pulse = portalPulse(i);

    let delta = uv - vec2f(cx, cy);
    let d     = length(delta);
    let dir   = select(delta / d, vec2f(0.0), d < 1e-6);

    // Gravitational lensing falloff: 1/r² with smooth cutoff at horizon
    let normD  = d / (r * 3.0);  // falloff range = 3× portal radius
    let falloff = 1.0 / (normD * normD + 0.1) - 1.0 / (1.0 + 0.1);
    let clamped = max(falloff, 0.0);

    // Pulse modulation
    let pulseMod = 1.0 + 0.15 * sin(uni.time * pulse * 6.283185);

    let weight = clamped * amp * opac * pulseMod;

    // Radial inward displacement (toward portal centre)
    totalOffset = totalOffset - dir * weight;
    totalChrom  = totalChrom + chrom * weight;
    maxWeight   = max(maxWeight, weight);
  }

  // Clamp total displacement
  let maxDisp = 0.08;
  let dispLen = length(totalOffset);
  if (dispLen > maxDisp) {
    totalOffset = totalOffset * (maxDisp / dispLen);
  }

  // Chromatic aberration: split R/G/B along displacement direction
  let chromDir = select(
    normalize(totalOffset),
    vec2f(0.0),
    length(totalOffset) < 1e-8,
  );
  let chromOff = chromDir * totalChrom;

  let uvR = uv + totalOffset + chromOff;
  let uvG = uv + totalOffset;
  let uvB = uv + totalOffset - chromOff;

  let colR = textureSample(src, smp, clamp(uvR, vec2f(0.0), vec2f(1.0))).r;
  let colG = textureSample(src, smp, clamp(uvG, vec2f(0.0), vec2f(1.0))).g;
  let colB = textureSample(src, smp, clamp(uvB, vec2f(0.0), vec2f(1.0))).b;
  let colA = textureSample(src, smp, clamp(uvG, vec2f(0.0), vec2f(1.0))).a;

  return vec4f(colR, colG, colB, colA);
}

// Vertex shader — full-screen triangle
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

// ─── WebGPU WGSL: Stencil Mask Pass ─────────────────────────────────────────

const STENCIL_MASK_WGSL = /* wgsl */`
// Writes stencil reference value inside circular portal apertures.
// Vertex shader generates a screen-aligned quad per portal; fragment shader
// discards fragments outside the aperture circle.

struct StencilUniforms {
  portalCount : u32,
  width       : f32,
  height      : f32,
  _pad        : f32,
}

struct PortalCircle {
  // cx, cy, apertureRadius, _pad — per portal
  data : array<vec4f, ${MAX_PORTALS}>,
}

@group(0) @binding(0) var<uniform>       uni     : StencilUniforms;
@group(0) @binding(1) var<storage, read> circles : PortalCircle;

struct VertOut {
  @builtin(position) pos : vec4f,
  @location(0) vUv       : vec2f,
  @location(1) @interpolate(flat) vIdx : u32,
}

var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0),  vec2f(-1.0, 1.0),
);

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  let c   = circles.data[ii];
  let cx  = c.x;
  let cy  = c.y;
  let ar  = c.z;  // aperture radius in domain units

  let qv  = QUAD[vi];
  let ndcX = (cx + qv.x * ar) / uni.width  * 2.0 - 1.0;
  let ndcY = (cy + qv.y * ar) / uni.height * 2.0 - 1.0;

  var out: VertOut;
  out.pos  = vec4f(ndcX, -ndcY, 0.0, 1.0);
  out.vUv  = qv;
  out.vIdx = ii;
  return out;
}

@fragment fn fs_main(in: VertOut) -> @location(0) vec4f {
  // Circular mask — discard outside unit circle
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  // Soft edge fade for the portal rim
  let edge = 1.0 - smoothstep(0.85, 1.0, sqrt(r2));

  // Write a translucent colour (the stencil write is the real purpose)
  return vec4f(0.0, 0.0, 0.0, edge * 0.5);
}
`;

// ─── WebGPU WGSL: Portal Interior Fill ──────────────────────────────────────

const INTERIOR_FILL_WGSL = /* wgsl */`
// Renders inside the stencil-masked portal apertures.  Produces a swirling
// nebula texture that represents "looking through" to the other side.

struct InteriorUniforms {
  time        : f32,
  portalCount : u32,
  width       : f32,
  height      : f32,
}

struct PortalInterior {
  // cx, cy, apertureRadius, opacity, baseR, baseG, baseB, highlightR,
  // highlightG, highlightB, spinAngle, _pad — 3 vec4 per portal
  data : array<vec4f, ${MAX_PORTALS * 3}>,
}

@group(0) @binding(0) var<uniform>       uni      : InteriorUniforms;
@group(0) @binding(1) var<storage, read> interior : PortalInterior;

fn hash2f(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn fbm(p: vec2f) -> f32 {
  var v  = 0.0;
  var a  = 0.5;
  var pp = p;
  for (var i = 0; i < 4; i = i + 1) {
    v  = v + a * hash2f(floor(pp) + vec2f(0.5));
    pp = pp * 2.1 + vec2f(1.7, 3.1);
    a  = a * 0.5;
  }
  return v;
}

struct VertOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  var out: VertOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f(x * 0.5 + 0.5, y * 0.5 + 0.5);
  return out;
}

@fragment fn fs_main(in: VertOut) -> @location(0) vec4f {
  let fragPos = vec2f(in.uv.x * uni.width, in.uv.y * uni.height);

  var col = vec3f(0.0);
  var totalA = 0.0;

  for (var i = 0u; i < uni.portalCount; i = i + 1u) {
    let cx  = interior.data[i * 3u].x;
    let cy  = interior.data[i * 3u].y;
    let ar  = interior.data[i * 3u].z;
    let op  = interior.data[i * 3u].w;

    let delta = fragPos - vec2f(cx, cy);
    let d = length(delta);
    if (d > ar) { continue; }

    let baseR = interior.data[i * 3u + 1u].x;
    let baseG = interior.data[i * 3u + 1u].y;
    let baseB = interior.data[i * 3u + 1u].z;
    let hlR   = interior.data[i * 3u + 1u].w;
    let hlG   = interior.data[i * 3u + 2u].x;
    let hlB   = interior.data[i * 3u + 2u].y;
    let spin  = interior.data[i * 3u + 2u].z;

    // Polar coordinates inside portal
    let normR = d / ar;
    let angle = atan2(delta.y, delta.x) + spin;

    // Swirling nebula pattern
    let spiralU = angle * 0.5 + normR * 3.0 + uni.time * 0.5;
    let spiralV = normR * 2.0 - uni.time * 0.3;
    let n = fbm(vec2f(spiralU, spiralV) * 1.5);

    // Depth illusion: darker at centre, brighter near rim
    let depthGrad = 0.3 + 0.7 * normR;

    // Mix base and highlight by noise
    let baseCol = vec3f(baseR, baseG, baseB);
    let hlCol   = vec3f(hlR, hlG, hlB);
    let mixed   = mix(baseCol * 0.4, hlCol, n * depthGrad);

    // Soft circular edge
    let edge = 1.0 - smoothstep(0.8, 1.0, normR);

    let portalA = op * edge;
    col    = col + mixed * portalA;
    totalA = max(totalA, portalA);
  }

  return vec4f(col, totalA);
}
`;

// ─── PortalEffectSystem ──────────────────────────────────────────────────────

/**
 * PortalEffectSystem — dual-mode (Canvas2D / WebGPU) portal warp renderer.
 *
 * Renders rotating-vortex portals at both endpoints of long-distance edges.
 * Portals feature:
 *   - Rotating logarithmic-spiral vortex arms
 *   - Event horizon glow ring
 *   - Accretion-disc particles (sucked in at source, ejected at target)
 *   - Space-distortion halo with chromatic aberration
 *   - Stencil-masked interior showing swirling nebula (portal view-through)
 *
 * @example
 * ```ts
 * // CPU/Canvas2D mode:
 * const portals = PortalEffectSystem.createCPU(ctx2d, {
 *   edges: topology.edges,
 *   distanceThreshold: 300,
 * });
 *
 * function frame(t: number) {
 *   ctx.clearRect(0, 0, canvas.width, canvas.height);
 *   portals.update(t / 1000, 1 / 60);
 *   portals.draw();
 *   requestAnimationFrame(frame);
 * }
 * ```
 */
export class PortalEffectSystem {
  private readonly mode: 'gpu' | 'cpu';

  // ── Configuration ────────────────────────────────────────────────────────
  private edges:              PortalEdge[];
  private distanceThreshold:  number;
  private portalRadius:       number;
  private enableDistortion:   boolean;
  private enableStencil:      boolean;
  private enableAccretion:    boolean;
  private enableVortex:       boolean;
  private enableHorizon:      boolean;
  private speedScale:         number;

  // ── Runtime state ────────────────────────────────────────────────────────
  private portals:            PortalInstance[]      = [];
  private particles:          AccretionParticle[]   = [];
  private elapsed:            number                = 0;

  // ── Canvas2D ─────────────────────────────────────────────────────────────
  private ctx?:               CanvasRenderingContext2D;

  // ── GPU state ────────────────────────────────────────────────────────────
  private device?:            GPUDevice;
  private canvas?:            HTMLCanvasElement;
  private gpuBuilt            = false;

  // GPU buffers
  private distUniformBuf?:    GPUBuffer;
  private distPortalBuf?:     GPUBuffer;
  private distPipeline?:      GPURenderPipeline;
  private distBG?:            GPUBindGroup;
  private distSampler?:       GPUSampler;

  private stencilUniformBuf?: GPUBuffer;
  private stencilCircleBuf?:  GPUBuffer;
  private stencilPipeline?:   GPURenderPipeline;
  private stencilBG?:         GPUBindGroup;

  private intUniformBuf?:     GPUBuffer;
  private intDataBuf?:        GPUBuffer;
  private intPipeline?:       GPURenderPipeline;
  private intBG?:             GPUBindGroup;

  // ── Private constructor ──────────────────────────────────────────────────

  private constructor(mode: 'gpu' | 'cpu', config: PortalEffectConfig) {
    this.mode               = mode;
    this.edges              = config.edges;
    this.distanceThreshold  = config.distanceThreshold ?? DEFAULT_DISTANCE_THRESHOLD;
    this.portalRadius       = config.portalRadius      ?? 40;
    this.enableDistortion   = config.distortion        ?? true;
    this.enableStencil      = config.stencilInterior   ?? true;
    this.enableAccretion    = config.accretionParticles ?? true;
    this.enableVortex       = config.vortexArms        ?? true;
    this.enableHorizon      = config.eventHorizon      ?? true;
    this.speedScale         = config.speedScale        ?? 1.0;
  }

  // ── Static factories ──────────────────────────────────────────────────────

  /** Create a CPU-mode (Canvas2D) portal renderer. */
  static createCPU(
    ctx:    CanvasRenderingContext2D,
    config: PortalEffectConfig,
  ): PortalEffectSystem {
    const sys = new PortalEffectSystem('cpu', config);
    sys.ctx   = ctx;
    sys._rebuildPortals();
    return sys;
  }

  /** Create a GPU-mode (WebGPU) portal renderer.  Call `build()` before use. */
  static createGPU(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    config: PortalEffectConfig,
  ): PortalEffectSystem {
    const sys    = new PortalEffectSystem('gpu', config);
    sys.device   = device;
    sys.canvas   = canvas;
    return sys;
  }

  // ── GPU build ──────────────────────────────────────────────────────────────

  /** Compile GPU pipelines and allocate buffers.  Required before GPU rendering. */
  async build(): Promise<void> {
    if (this.mode !== 'gpu' || !this.device) return;

    const device = this.device;
    const format = navigator.gpu.getPreferredCanvasFormat();

    this._rebuildPortals();

    // ── Distortion post-process pipeline ────────────────────────────────
    const distMod = device.createShaderModule({ code: DISTORTION_WGSL });

    this.distUniformBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.distPortalBuf = device.createBuffer({
      size:  Math.max(MAX_PORTALS * 2 * 16, 32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.distSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    const distBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    this.distPipeline = device.createRenderPipeline({
      layout:  device.createPipelineLayout({ bindGroupLayouts: [distBGL] }),
      vertex:   { module: distMod, entryPoint: 'vs_main' },
      fragment: {
        module: distMod, entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Stencil mask pipeline ───────────────────────────────────────────
    const stencilMod = device.createShaderModule({ code: STENCIL_MASK_WGSL });

    this.stencilUniformBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.stencilCircleBuf = device.createBuffer({
      size:  Math.max(MAX_PORTALS * 16, 32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const stencilBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.stencilPipeline = device.createRenderPipeline({
      layout:  device.createPipelineLayout({ bindGroupLayouts: [stencilBGL] }),
      vertex:   { module: stencilMod, entryPoint: 'vs_main' },
      fragment: {
        module: stencilMod, entryPoint: 'fs_main',
        targets: [{
          format,
          writeMask: 0x0, // colour write off — stencil-only pass
        }],
      },
      primitive:    { topology: 'triangle-list' },
      depthStencil: {
        format:              'depth24plus-stencil8',
        depthWriteEnabled:   false,
        depthCompare:        'always',
        stencilFront: {
          compare:    'always',
          passOp:     'replace',
          failOp:     'keep',
          depthFailOp:'keep',
        },
        stencilBack: {
          compare:    'always',
          passOp:     'replace',
          failOp:     'keep',
          depthFailOp:'keep',
        },
        stencilReadMask:  0xFF,
        stencilWriteMask: 0xFF,
      },
    });

    // ── Interior fill pipeline (renders inside stencil) ─────────────────
    const intMod = device.createShaderModule({ code: INTERIOR_FILL_WGSL });

    this.intUniformBuf = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.intDataBuf = device.createBuffer({
      size:  Math.max(MAX_PORTALS * 3 * 16, 48),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const intBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.intPipeline = device.createRenderPipeline({
      layout:  device.createPipelineLayout({ bindGroupLayouts: [intBGL] }),
      vertex:   { module: intMod, entryPoint: 'vs_main' },
      fragment: {
        module: intMod, entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format:            'depth24plus-stencil8',
        depthWriteEnabled: false,
        depthCompare:      'always',
        stencilFront: {
          compare:    'equal',
          passOp:     'keep',
          failOp:     'keep',
          depthFailOp:'keep',
        },
        stencilBack: {
          compare:    'equal',
          passOp:     'keep',
          failOp:     'keep',
          depthFailOp:'keep',
        },
        stencilReadMask:  0xFF,
        stencilWriteMask: 0x00,
      },
    });

    this._buildBindGroups();
    this.gpuBuilt = true;

    console.log(
      `[PortalEffectSystem GPU] built: ${this.portals.length} portals, ` +
      `${this.particles.length} accretion particles`,
    );
  }

  // ── Update (advance animation) ─────────────────────────────────────────────

  /**
   * Advance portal animation state.
   * @param elapsed  — total elapsed time in seconds
   * @param dt       — frame delta in seconds
   */
  update(elapsed: number, dt: number): void {
    this.elapsed = elapsed;
    const scaledDt = dt * this.speedScale;

    // Update portal vortex rotation
    for (const portal of this.portals) {
      const spinDir = portal.isSource ? portal.style.spinDir : -portal.style.spinDir;
      portal.vortexAngle += portal.style.spinSpeed * spinDir * scaledDt;
    }

    // Update accretion particles
    if (this.enableAccretion) {
      this._tickAccretionParticles(scaledDt, elapsed);
    }
  }

  // ── Canvas2D draw ──────────────────────────────────────────────────────────

  /** Draw all portal effects (CPU/Canvas2D mode). */
  draw(): void {
    if (!this.ctx) return;
    const ctx = this.ctx;

    for (const portal of this.portals) {
      if (portal.opacity < 0.004) continue;

      // Layer 0: Space distortion (Canvas2D approximation: radial gradient overlay)
      if (this.enableDistortion) {
        this._drawDistortionHalo(ctx, portal);
      }

      // Layer 1: Event horizon ring
      if (this.enableHorizon) {
        this._drawEventHorizon(ctx, portal);
      }

      // Layer 4: Stencil interior (Canvas2D: clip + nebula fill)
      if (this.enableStencil) {
        this._drawPortalInterior(ctx, portal);
      }

      // Layer 2: Vortex arms
      if (this.enableVortex) {
        this._drawVortexArms(ctx, portal);
      }

      // Layer 3: Accretion particles
      if (this.enableAccretion) {
        this._drawAccretionParticles(ctx, portal);
      }
    }
  }

  // ── GPU encode passes ──────────────────────────────────────────────────────

  /**
   * Encode the space-distortion post-process pass (GPU mode).
   * Call after the main scene render, before final blit.
   */
  encodeDistortion(
    encoder:  GPUCommandEncoder,
    srcView:  GPUTextureView,
    dstView:  GPUTextureView,
    width:    number,
    height:   number,
  ): void {
    if (!this.gpuBuilt || !this.enableDistortion || this.portals.length === 0) return;
    if (!this.device || !this.distPipeline || !this.distUniformBuf || !this.distPortalBuf) return;

    // Upload uniforms
    const uniData = new Float32Array(4);
    uniData[0] = width;
    uniData[1] = height;
    new Uint32Array(uniData.buffer)[2] = this.portals.length;
    uniData[3] = this.elapsed;
    this.device.queue.writeBuffer(this.distUniformBuf, 0, uniData);

    // Upload portal data
    const portalData = new Float32Array(this.portals.length * 8);
    for (let i = 0; i < this.portals.length; i++) {
      const p = this.portals[i];
      const b = i * 8;
      portalData[b + 0] = p.x;
      portalData[b + 1] = p.y;
      portalData[b + 2] = p.radius;
      portalData[b + 3] = p.style.distortionAmp;
      portalData[b + 4] = p.style.chromaticSpread;
      portalData[b + 5] = p.opacity;
      portalData[b + 6] = p.style.horizonPulseHz;
      portalData[b + 7] = 0; // padding
    }
    this.device.queue.writeBuffer(this.distPortalBuf, 0, portalData);

    // Rebuild bind group with current src texture
    this.distBG = this.device.createBindGroup({
      layout: this.distPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.distUniformBuf } },
        { binding: 1, resource: { buffer: this.distPortalBuf } },
        { binding: 2, resource: this.distSampler! },
        { binding: 3, resource: srcView },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:    dstView,
        loadOp:  'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.distPipeline);
    pass.setBindGroup(0, this.distBG);
    pass.draw(3);
    pass.end();
  }

  /**
   * Encode the stencil mask + interior fill passes (GPU mode).
   * Requires a depth-stencil attachment with format 'depth24plus-stencil8'.
   */
  encodeStencilInterior(
    encoder:         GPUCommandEncoder,
    colorView:       GPUTextureView,
    depthStencilView: GPUTextureView,
  ): void {
    if (!this.gpuBuilt || !this.enableStencil || this.portals.length === 0) return;
    if (!this.device) return;

    const width  = this.canvas?.width  ?? 1;
    const height = this.canvas?.height ?? 1;

    // ── Pass A: Write stencil mask ──────────────────────────────────────
    if (this.stencilPipeline && this.stencilUniformBuf && this.stencilCircleBuf) {
      // Upload uniforms
      const sUni = new Float32Array(4);
      new Uint32Array(sUni.buffer)[0] = this.portals.length;
      sUni[1] = width;
      sUni[2] = height;
      this.device.queue.writeBuffer(this.stencilUniformBuf, 0, sUni);

      // Upload circles
      const circles = new Float32Array(this.portals.length * 4);
      for (let i = 0; i < this.portals.length; i++) {
        const p = this.portals[i];
        circles[i * 4 + 0] = p.x;
        circles[i * 4 + 1] = p.y;
        circles[i * 4 + 2] = p.radius * p.style.apertureFrac;
        circles[i * 4 + 3] = 0;
      }
      this.device.queue.writeBuffer(this.stencilCircleBuf, 0, circles);

      this.stencilBG = this.device.createBindGroup({
        layout: this.stencilPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.stencilUniformBuf } },
          { binding: 1, resource: { buffer: this.stencilCircleBuf } },
        ],
      });

      const stencilPass = encoder.beginRenderPass({
        colorAttachments: [{
          view:    colorView,
          loadOp:  'load',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view:             depthStencilView,
          depthLoadOp:      'load',
          depthStoreOp:     'store',
          stencilLoadOp:    'clear',
          stencilStoreOp:   'store',
          stencilClearValue: 0,
        },
      });
      stencilPass.setPipeline(this.stencilPipeline);
      stencilPass.setBindGroup(0, this.stencilBG);
      stencilPass.setStencilReference(STENCIL_REF);
      stencilPass.draw(6, this.portals.length);
      stencilPass.end();
    }

    // ── Pass B: Fill interior (stencil test = equal) ────────────────────
    if (this.intPipeline && this.intUniformBuf && this.intDataBuf) {
      const iUni = new Float32Array(4);
      iUni[0] = this.elapsed;
      new Uint32Array(iUni.buffer)[1] = this.portals.length;
      iUni[2] = width;
      iUni[3] = height;
      this.device.queue.writeBuffer(this.intUniformBuf, 0, iUni);

      // Upload interior data (3 vec4 per portal)
      const iData = new Float32Array(this.portals.length * 12);
      for (let i = 0; i < this.portals.length; i++) {
        const p = this.portals[i];
        const b = i * 12;
        iData[b + 0]  = p.x;
        iData[b + 1]  = p.y;
        iData[b + 2]  = p.radius * p.style.apertureFrac;
        iData[b + 3]  = p.opacity;
        iData[b + 4]  = p.theme.base.r;
        iData[b + 5]  = p.theme.base.g;
        iData[b + 6]  = p.theme.base.b;
        iData[b + 7]  = p.theme.highlight.r;
        iData[b + 8]  = p.theme.highlight.g;
        iData[b + 9]  = p.theme.highlight.b;
        iData[b + 10] = p.vortexAngle;
        iData[b + 11] = 0;
      }
      this.device.queue.writeBuffer(this.intDataBuf, 0, iData);

      this.intBG = this.device.createBindGroup({
        layout: this.intPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.intUniformBuf } },
          { binding: 1, resource: { buffer: this.intDataBuf } },
        ],
      });

      const intPass = encoder.beginRenderPass({
        colorAttachments: [{
          view:    colorView,
          loadOp:  'load',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view:             depthStencilView,
          depthLoadOp:      'load',
          depthStoreOp:     'store',
          stencilLoadOp:    'load',
          stencilStoreOp:   'store',
        },
      });
      intPass.setPipeline(this.intPipeline);
      intPass.setBindGroup(0, this.intBG);
      intPass.setStencilReference(STENCIL_REF);
      intPass.draw(3);
      intPass.end();
    }
  }

  /**
   * Encode vortex arm + accretion particle overlay (GPU mode).
   * Additive-blended over the current colour attachment.
   *
   * Note: In the current implementation, vortex arms and accretion particles
   * are rendered CPU-side even in GPU mode (using a 2D overlay canvas).
   * A future iteration could move these to a GPU instanced draw pass.
   */
  encodeVortex(
    _encoder: GPUCommandEncoder,
    _colorView: GPUTextureView,
  ): void {
    // GPU vortex/particle rendering is a stub for future migration.
    // Currently, call draw() on a 2D overlay canvas for these layers.
  }

  // ── Edge management ────────────────────────────────────────────────────────

  /** Replace edges at runtime. */
  setEdges(edges: PortalEdge[]): void {
    this.edges = edges;
    this._rebuildPortals();
  }

  /** Update distance threshold. */
  setDistanceThreshold(v: number): void {
    this.distanceThreshold = v;
    this._rebuildPortals();
  }

  /** Set global speed multiplier. */
  setSpeedScale(v: number): void { this.speedScale = v; }

  /** Toggle space distortion. */
  setDistortion(v: boolean): void { this.enableDistortion = v; }

  /** Toggle stencil interior. */
  setStencilInterior(v: boolean): void { this.enableStencil = v; }

  /** Toggle accretion particles. */
  setAccretionParticles(v: boolean): void { this.enableAccretion = v; }

  /** Toggle vortex arms. */
  setVortexArms(v: boolean): void { this.enableVortex = v; }

  /** Toggle event horizon. */
  setEventHorizon(v: boolean): void { this.enableHorizon = v; }

  /** Read-only portal instances (for external overlay renderers). */
  get activePortals(): readonly PortalInstance[] { return this.portals; }

  /** Number of active portals. */
  get portalCount(): number { return this.portals.length; }

  /** Number of active accretion particles. */
  get activeParticleCount(): number {
    return this.particles.filter(p => p.phase !== AccretionPhase.DEAD).length;
  }

  /** Destroy GPU resources. */
  destroy(): void {
    if (this.mode === 'gpu') this._destroyGPU();
  }

  // ── Private: rebuild portal instances from edges ───────────────────────────

  private _rebuildPortals(): void {
    this.portals  = [];
    this.particles = [];

    for (const edge of this.edges) {
      if (edge.points.length < 2) continue;

      const srcPt = evalCatmullRom(edge.points, 0);
      const tgtPt = evalCatmullRom(edge.points, 1);
      const d     = dist(srcPt.x, srcPt.y, tgtPt.x, tgtPt.y);

      if (d < this.distanceThreshold - BLEND_ZONE) continue;

      // Opacity based on distance blend zone
      const opacity = smoothstep(
        this.distanceThreshold - BLEND_ZONE,
        this.distanceThreshold + BLEND_ZONE,
        d,
      );

      const qosName = edge.qos ?? 'DEFAULT';
      const style   = { ...QOS_PORTAL_STYLE[qosName] };
      const theme   = QOS_THEME[qosName] ?? QOS_THEME.DEFAULT;

      // Source portal
      if (this.portals.length < MAX_PORTALS) {
        this.portals.push({
          x:           srcPt.x,
          y:           srcPt.y,
          radius:      this.portalRadius,
          vortexAngle: 0,
          opacity,
          isSource:    true,
          edge,
          style:       { ...style, spinDir: 1 },
          theme,
          remoteX:     tgtPt.x,
          remoteY:     tgtPt.y,
        });
      }

      // Target portal
      if (this.portals.length < MAX_PORTALS) {
        this.portals.push({
          x:           tgtPt.x,
          y:           tgtPt.y,
          radius:      this.portalRadius,
          vortexAngle: Math.PI, // offset start angle
          opacity,
          isSource:    false,
          edge,
          style:       { ...style, spinDir: -1 },
          theme,
          remoteX:     srcPt.x,
          remoteY:     srcPt.y,
        });
      }
    }

    // Initialise accretion particles
    if (this.enableAccretion) {
      this._initAccretionParticles();
    }
  }

  // ── Private: accretion particle lifecycle ──────────────────────────────────

  private _initAccretionParticles(): void {
    this.particles = [];
    let globalSlot = 0;

    for (let pi = 0; pi < this.portals.length && globalSlot < TOTAL_PARTICLE_POOL; pi++) {
      const portal  = this.portals[pi];
      const count   = Math.min(portal.style.particleCount, TOTAL_PARTICLE_POOL - globalSlot);

      for (let p = 0; p < count; p++, globalSlot++) {
        const seed  = globalSlot * 1.618034;
        const angle = rng(seed, 0) * Math.PI * 2;
        const rFrac = rng(seed, 1);
        const r     = portal.radius * (portal.style.apertureFrac + rFrac * (1.0 - portal.style.apertureFrac));

        this.particles.push({
          angle,
          radius:    r,
          omega:     portal.style.orbitSpeed * (0.7 + rng(seed, 2) * 0.6),
          vr:        portal.isSource ? -portal.style.radialSpeed : portal.style.radialSpeed,
          phase:     AccretionPhase.ORBIT,
          alpha:     0.6 + rng(seed, 3) * 0.4,
          size:      1.5 + rng(seed, 4) * 2.0,
          seed,
          portalIdx: pi,
          isSource:  portal.isSource,
        });
      }
    }
  }

  private _tickAccretionParticles(dt: number, elapsed: number): void {
    for (let i = 0; i < this.particles.length; i++) {
      const p      = this.particles[i];
      const portal = this.portals[p.portalIdx];
      if (!portal) continue;

      const apertureR = portal.radius * portal.style.apertureFrac;

      switch (p.phase) {
        case AccretionPhase.DEAD:
          this._respawnAccretionParticle(p, portal, elapsed);
          break;

        case AccretionPhase.ORBIT: {
          // Angular rotation (faster at smaller radius — Keplerian-ish)
          const keplerFactor = Math.max(0.3, portal.radius / (p.radius + 1));
          p.angle += p.omega * keplerFactor * dt;

          // Radial drift
          p.radius += p.vr * dt;

          // Source: spiral inward — when reaching aperture, begin fade
          if (p.isSource && p.radius <= apertureR) {
            p.phase = AccretionPhase.FADE;
          }
          // Target: spiral outward — when reaching outer rim, begin fade
          if (!p.isSource && p.radius >= portal.radius) {
            p.phase = AccretionPhase.FADE;
          }
          break;
        }

        case AccretionPhase.FADE:
          p.alpha -= 2.0 * dt;
          if (p.alpha <= 0) {
            p.alpha = 0;
            p.phase = AccretionPhase.DEAD;
          }
          break;
      }
    }
  }

  private _respawnAccretionParticle(
    p:      AccretionParticle,
    portal: PortalInstance,
    t:      number,
  ): void {
    const seed = p.seed + t;
    const apertureR = portal.radius * portal.style.apertureFrac;

    if (portal.isSource) {
      // Source: spawn at outer rim, spiral inward
      p.radius = portal.radius * (0.8 + rng(seed, 0) * 0.2);
      p.vr     = -portal.style.radialSpeed * (0.6 + rng(seed, 5) * 0.8);
    } else {
      // Target: spawn at aperture, spiral outward
      p.radius = apertureR * (0.8 + rng(seed, 0) * 0.4);
      p.vr     = portal.style.radialSpeed * (0.6 + rng(seed, 5) * 0.8);
    }

    p.angle = rng(seed, 1) * Math.PI * 2;
    p.omega = portal.style.orbitSpeed * (0.7 + rng(seed, 2) * 0.6);
    p.alpha = 0.5 + rng(seed, 3) * 0.5;
    p.size  = 1.5 + rng(seed, 4) * 2.0;
    p.phase = AccretionPhase.ORBIT;
    p.seed  = seed;
  }

  // ── Private: Canvas2D drawing ──────────────────────────────────────────────

  private _drawDistortionHalo(ctx: CanvasRenderingContext2D, portal: PortalInstance): void {
    const { x, y, radius, opacity, theme, style } = portal;
    const outerR = radius * 3.0;

    // Radial gradient: visible warp halo (Canvas2D can't do UV distortion,
    // so we approximate with a radial gradient overlay)
    const grad = ctx.createRadialGradient(x, y, radius * 0.5, x, y, outerR);
    const { base } = theme;

    grad.addColorStop(0,   toCss(base.r, base.g, base.b, 0));
    grad.addColorStop(0.3, toCss(base.r, base.g, base.b, 0.08 * opacity));
    grad.addColorStop(0.6, toCss(base.r, base.g, base.b, 0.04 * opacity));
    grad.addColorStop(1,   toCss(base.r, base.g, base.b, 0));

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, outerR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  private _drawEventHorizon(ctx: CanvasRenderingContext2D, portal: PortalInstance): void {
    const { x, y, radius, opacity, theme, style } = portal;
    const { highlight, base } = theme;

    // Pulsing brightness
    const pulse = 0.7 + 0.3 * Math.sin(this.elapsed * style.horizonPulseHz * Math.PI * 2);
    const alpha = opacity * pulse;

    const ringR = radius * 0.85;
    const hw    = style.horizonWidth;

    // Outer glow
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    ctx.beginPath();
    ctx.arc(x, y, ringR + hw * 2, 0, Math.PI * 2);
    ctx.lineWidth   = hw * 3;
    ctx.strokeStyle = toCss(base.r, base.g, base.b, alpha * 0.15);
    ctx.stroke();

    // Core ring
    ctx.beginPath();
    ctx.arc(x, y, ringR, 0, Math.PI * 2);
    ctx.lineWidth   = hw;
    ctx.strokeStyle = toCss(highlight.r, highlight.g, highlight.b, alpha * 0.6);
    ctx.stroke();

    // Inner bright ring
    ctx.beginPath();
    ctx.arc(x, y, ringR - hw * 0.3, 0, Math.PI * 2);
    ctx.lineWidth   = hw * 0.4;
    ctx.strokeStyle = toCss(1, 1, 1, alpha * 0.3);
    ctx.stroke();

    ctx.restore();
  }

  private _drawPortalInterior(ctx: CanvasRenderingContext2D, portal: PortalInstance): void {
    const { x, y, radius, opacity, theme, style, vortexAngle } = portal;
    const apertureR = radius * style.apertureFrac;
    const { base, highlight, shadow } = theme;

    ctx.save();

    // Clip to circular aperture
    ctx.beginPath();
    ctx.arc(x, y, apertureR, 0, Math.PI * 2);
    ctx.clip();

    // Dark background
    ctx.fillStyle = toCss(shadow.r * 0.3, shadow.g * 0.3, shadow.b * 0.3, opacity * 0.85);
    ctx.fillRect(x - apertureR, y - apertureR, apertureR * 2, apertureR * 2);

    // Swirling nebula pattern (procedural noise)
    ctx.globalCompositeOperation = 'lighter';
    const steps = 24;
    for (let i = 0; i < steps; i++) {
      const angle = (i / steps) * Math.PI * 2 + vortexAngle * 0.5;
      const nFrac = i / steps;
      const r     = apertureR * (0.2 + nFrac * 0.7);
      const nx    = x + Math.cos(angle) * r;
      const ny    = y + Math.sin(angle) * r;

      const n = noise2D(
        nx * 0.02 + this.elapsed * 0.3,
        ny * 0.02 - this.elapsed * 0.2,
      );

      const blobR  = apertureR * (0.15 + n * 0.25);
      const blobA  = opacity * n * 0.35;

      // Mix base and highlight colours by noise
      const cr = lerp(base.r * 0.5, highlight.r, n);
      const cg = lerp(base.g * 0.5, highlight.g, n);
      const cb = lerp(base.b * 0.5, highlight.b, n);

      const grad = ctx.createRadialGradient(nx, ny, 0, nx, ny, blobR);
      grad.addColorStop(0, toCss(cr, cg, cb, blobA));
      grad.addColorStop(1, toCss(cr, cg, cb, 0));

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(nx, ny, blobR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Centre bright spot (wormhole throat)
    const coreGrad = ctx.createRadialGradient(x, y, 0, x, y, apertureR * 0.35);
    coreGrad.addColorStop(0, toCss(1, 1, 1, opacity * 0.25));
    coreGrad.addColorStop(0.5, toCss(highlight.r, highlight.g, highlight.b, opacity * 0.12));
    coreGrad.addColorStop(1, toCss(highlight.r, highlight.g, highlight.b, 0));
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(x, y, apertureR * 0.35, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  private _drawVortexArms(ctx: CanvasRenderingContext2D, portal: PortalInstance): void {
    const { x, y, radius, opacity, theme, style, vortexAngle } = portal;
    const { highlight, base } = theme;
    const armCount = VORTEX_ARM_COUNT;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    const apertureR = radius * style.apertureFrac;

    for (let arm = 0; arm < armCount; arm++) {
      const armBaseAngle = vortexAngle + (arm / armCount) * Math.PI * 2;

      // Draw spiral arm as a series of points
      const segments = 32;
      for (let s = 0; s < segments; s++) {
        const t     = s / segments;
        // Logarithmic spiral: r = a * e^(b*θ)
        const spiralR = apertureR + (radius - apertureR) * t;
        const spiralAngle = armBaseAngle + t * Math.PI * 1.2; // ~1.2 radians of spiral wrap

        const px = x + Math.cos(spiralAngle) * spiralR;
        const py = y + Math.sin(spiralAngle) * spiralR;

        // Brightness: peaks at mid-arm, fades at tips
        const bell  = Math.sin(Math.PI * t);
        const alpha = opacity * style.armBrightness * bell * 0.4;
        if (alpha < 0.004) continue;

        // Width narrows toward centre
        const w = style.armWidth * radius * (0.2 + 0.8 * t);

        // Colour: blend base to highlight along arm
        const cr = lerp(base.r, highlight.r, t * 0.7);
        const cg = lerp(base.g, highlight.g, t * 0.7);
        const cb = lerp(base.b, highlight.b, t * 0.7);

        // Gradient blob
        const grad = ctx.createRadialGradient(px, py, 0, px, py, w);
        grad.addColorStop(0, toCss(cr, cg, cb, alpha));
        grad.addColorStop(1, toCss(cr, cg, cb, 0));

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, w, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  private _drawAccretionParticles(ctx: CanvasRenderingContext2D, portal: PortalInstance): void {
    const { x, y, opacity, theme } = portal;
    const { base, highlight } = theme;
    const portalIdx = this.portals.indexOf(portal);

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const p of this.particles) {
      if (p.portalIdx !== portalIdx) continue;
      if (p.phase === AccretionPhase.DEAD) continue;
      if (p.alpha < 0.004) continue;

      const px = x + Math.cos(p.angle) * p.radius;
      const py = y + Math.sin(p.angle) * p.radius;

      // Colour: mix by radial position
      const rFrac = p.radius / portal.radius;
      const cr = lerp(highlight.r, base.r, rFrac);
      const cg = lerp(highlight.g, base.g, rFrac);
      const cb = lerp(highlight.b, base.b, rFrac);

      const a = p.alpha * opacity;

      // Core particle
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fillStyle = toCss(cr, cg, cb, a);
      ctx.fill();

      // Glow halo
      ctx.beginPath();
      ctx.arc(px, py, p.size * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = toCss(cr, cg, cb, a * 0.15);
      ctx.fill();
    }

    ctx.restore();
  }

  // ── Private: GPU bind groups ───────────────────────────────────────────────

  private _buildBindGroups(): void {
    // Bind groups are rebuilt per-frame in the encode methods because
    // they reference textures that may change.  This method is a
    // placeholder for static bind group creation if needed.
  }

  // ── Private: destroy GPU resources ─────────────────────────────────────────

  private _destroyGPU(): void {
    if (!this.gpuBuilt) return;
    this.distUniformBuf?.destroy();
    this.distPortalBuf?.destroy();
    this.stencilUniformBuf?.destroy();
    this.stencilCircleBuf?.destroy();
    this.intUniformBuf?.destroy();
    this.intDataBuf?.destroy();
    this.gpuBuilt = false;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create a CPU-mode PortalEffectSystem with QoS profiles resolved from a map.
 */
export function createPortalEffectRenderer(
  ctx:     CanvasRenderingContext2D,
  edges:   Omit<PortalEdge, 'qos'>[],
  qosMap:  Record<string, QoSProfileName>,
  config:  Omit<PortalEffectConfig, 'edges'> = {},
): PortalEffectSystem {
  const resolved: PortalEdge[] = edges.map(e => ({
    ...e,
    qos: qosMap[e.edgeId] ?? 'DEFAULT',
  }));
  return PortalEffectSystem.createCPU(ctx, { ...config, edges: resolved });
}

/**
 * Check whether an edge qualifies for portal rendering based on distance.
 */
export function shouldRenderPortal(
  edge:      PortalEdge,
  threshold: number = DEFAULT_DISTANCE_THRESHOLD,
): boolean {
  if (edge.points.length < 2) return false;
  const src = evalCatmullRom(edge.points, 0);
  const tgt = evalCatmullRom(edge.points, 1);
  return dist(src.x, src.y, tgt.x, tgt.y) >= threshold - BLEND_ZONE;
}

// ─── Defaults re-export ──────────────────────────────────────────────────────

export const PORTAL_DEFAULTS = {
  distanceThreshold: DEFAULT_DISTANCE_THRESHOLD,
  blendZone:         BLEND_ZONE,
  maxPortals:        MAX_PORTALS,
  vortexArmCount:    VORTEX_ARM_COUNT,
  maxAccretionParts: MAX_ACCRETION_PARTICLES,
  totalParticlePool: TOTAL_PARTICLE_POOL,
  stencilRef:        STENCIL_REF,
  qosPortalStyles:   QOS_PORTAL_STYLE,
  gpuWorkgroupSize:  GPU_WG,
} as const;

export { QOS_PORTAL_STYLE };
