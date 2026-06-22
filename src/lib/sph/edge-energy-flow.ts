/**
 * edge-energy-flow.ts — M777
 *
 * Edge Energy Flow Visualisation — data traffic → particle fluid properties.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Maps live data-flow metrics (messages/sec, burst events, bandwidth) onto a
 * GPU particle fluid layer that coats each topology edge.  Unlike the discrete
 * pulse-based EdgeDataFlowViz (M766) or the lifecycle-oriented EdgeFlowRenderer
 * (M742), this module treats every edge as a continuous energy stream where:
 *
 *   • **Particle density** scales with data throughput — idle edges carry a
 *     sparse trickle; saturated edges become dense streams.
 *   • **Particle speed** maps from message rate — higher mps = faster flow.
 *   • **Colour brightness/saturation** intensifies with traffic volume,
 *     desaturating toward a dim base tone during silence.
 *
 * QoS profiles determine the fluid's visual character:
 *
 *   SENSOR_DATA  — 蓝色快速流: cool-blue rapid stream, high turbulence,
 *                  narrow particles, short trails.  Feels like a fast data
 *                  firehose — many small droplets rushing along the edge.
 *
 *   PARAMETERS   — 金色脉冲: warm-gold pulsating flow, moderate speed,
 *                  rhythmic density waves.  Parameters arrive in deliberate
 *                  bursts — the flow swells and contracts like a heartbeat.
 *
 *   TF_STATIC    — 绿色恒定: jade-green steady laminar flow, slow and wide,
 *                  high persistence.  Transform data is constant — the stream
 *                  is serene, wide, and unfading.
 *
 *   TOPO_CHANGE  — 红色爆发: crimson-red explosive bursts, very fast, high
 *                  particle count spikes that decay rapidly.  Topology events
 *                  are rare but dramatic — a brief torrent of red energy.
 *
 *   DEFAULT      — neutral slate, balanced flow.
 *
 * Idle behaviour:
 *   When no data flows (traffic = 0), the edge does not go fully dark.
 *   Instead, a subtle "breathing" pulsation animates a minimal particle
 *   population with a slow sinusoidal alpha/speed modulation (period ~3s).
 *   This keeps every edge visually alive and spatially legible.
 *
 * Architecture:
 *
 *   EdgeEnergyFlow manages a flat pool of EnergyParticle slots distributed
 *   across all registered edges.  Each frame:
 *     1. Traffic metrics update → per-edge target density/speed/brightness
 *     2. Particle tick: advance travel, apply QoS speed + curl perturbation,
 *        respawn dead slots with density-scaled probability
 *     3. Draw: Canvas2D additive-blend circles with QoS colour × brightness
 *
 *   The CPU-only approach avoids a WebGPU dependency while maintaining the AT
 *   aesthetic.  For GPU acceleration, the EdgeFlowRenderer (M742) or
 *   ATSplineParticleLife (M713) systems handle the heavy-lifting; this module
 *   layers on top as a lightweight energy-density overlay.
 *
 * Integration:
 *   ```ts
 *   import { EdgeEnergyFlow } from '$lib/sph/edge-energy-flow';
 *
 *   const flow = new EdgeEnergyFlow(ctx2d, {
 *     edges: topology.edges.map(e => ({
 *       edgeId: e.id, sourceId: e.src, targetId: e.tgt,
 *       points: e.route, qos: e.qosProfile,
 *     })),
 *   });
 *
 *   // Push live traffic metrics:
 *   flow.setTraffic('edge-001', { mps: 85, burstFactor: 0.0 });
 *   flow.setTraffic('edge-002', { mps: 0,  burstFactor: 0.0 });  // idle → breathing
 *
 *   // Burst event (TOPO_CHANGE):
 *   flow.triggerBurst('edge-003', 1.0);  // full-intensity red burst
 *
 *   // Render loop:
 *   flow.update(dt, elapsed);
 *   flow.draw();
 *
 *   // Cleanup:
 *   flow.destroy();
 *   ```
 *
 * References:
 *   src/lib/sph/at-spline-particle.ts   — GPU particle lifecycle (AT port)
 *   src/lib/sph/at-shader-utils.ts      — easing + range + blend utilities
 *   src/lib/sph/edge-data-flow-viz.ts   — pulse-based edge glow (M766)
 *   src/lib/sph/edge-flow-renderer.ts   — GPU/CPU edge flow renderer (M742)
 *   src/lib/sph/color-palette.ts        — QoS → ThemePalette mapping
 *   src/lib/sph/qosSpatial.ts           — QoS profile definitions
 */

import type { QoSProfileName }    from './qosSpatial';
import { QOS_THEME }               from './color-palette';
import type { ThemePalette, RGB }  from './color-palette';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Maximum particles per edge. */
const MAX_PARTICLES_PER_EDGE = 128;

/** Global particle pool cap. */
const MAX_TOTAL_PARTICLES = 4096;

/** Catmull-Rom spline subdivision count for sampling. */
const SPLINE_SEGMENTS = 48;

/** Idle breathing period in seconds. */
const BREATH_PERIOD = 3.0;

/** Minimum alpha for idle breathing baseline. */
const BREATH_ALPHA_MIN = 0.04;

/** Maximum alpha for idle breathing peak. */
const BREATH_ALPHA_MAX = 0.15;

/** Minimum particle count ratio during idle (fraction of allocated slots). */
const IDLE_DENSITY_RATIO = 0.08;

/** Speed of idle breathing particles (normalised spline units / second). */
const IDLE_SPEED = 0.04;

/** Curl-noise amplitude for lateral perturbation (domain units). */
const CURL_AMPLITUDE = 3.0;

/** Curl-noise spatial frequency. */
const CURL_FREQUENCY = 0.008;

/** Smoothing rate for traffic metric interpolation (per second). */
const METRIC_SMOOTHING = 4.0;

/** Burst decay rate — how fast a triggerBurst() event fades (per second). */
const BURST_DECAY = 2.5;

// ─── QoS → fluid profile mapping ─────────────────────────────────────────────

/**
 * FluidProfile — per-QoS visual characteristics that shape the energy stream.
 *
 * These parameters map QoS semantics to fluid aesthetics:
 *   - SENSOR_DATA → fast, dense, blue, turbulent (data firehose)
 *   - PARAMETERS  → moderate, pulsating, gold (heartbeat rhythm)
 *   - TF_STATIC   → slow, wide, green, laminar (constant stream)
 *   - TOPO_CHANGE → explosive, red, brief (shock wave)
 */
export interface FluidProfile {
  /** Base travel speed (normalised spline units / second) at mps=0. */
  speedBase:          number;
  /** Speed added per normalised mps unit (0→1). */
  speedPerMps:        number;
  /** Maximum travel speed cap. */
  speedMax:           number;
  /** Base particle density ratio [0, 1] at mps=0 (fraction of max slots). */
  densityBase:        number;
  /** Density added per normalised mps. */
  densityPerMps:      number;
  /** Maximum density ratio. */
  densityMax:         number;
  /** Base alpha at mps=0 (excluding breathing). */
  alphaBase:          number;
  /** Alpha boost at full traffic. */
  alphaFull:          number;
  /** Particle radius in domain units (base). */
  radiusBase:         number;
  /** Radius multiplier at peak traffic. */
  radiusTraffic:      number;
  /** Curl-noise strength multiplier (turbulence). */
  curlScale:          number;
  /** Pulse modulation: amplitude of rhythmic density wave [0, 1]. */
  pulseAmplitude:     number;
  /** Pulse modulation: frequency in Hz. */
  pulseFrequency:     number;
  /** Trail length (how much the previous position bleeds, 0=none). */
  trailAlpha:         number;
  /** Colour saturation boost at full traffic [0, 1]. */
  saturationBoost:    number;
}

const FLUID_PROFILES: Record<QoSProfileName, FluidProfile> = {
  // ── SENSOR_DATA — 蓝色快速流 ──────────────────────────────────────────────
  SENSOR_DATA: {
    speedBase:       0.15,
    speedPerMps:     1.20,
    speedMax:        1.50,
    densityBase:     0.12,
    densityPerMps:   0.80,
    densityMax:      0.95,
    alphaBase:       0.20,
    alphaFull:       0.85,
    radiusBase:      2.0,
    radiusTraffic:   1.3,
    curlScale:       1.8,
    pulseAmplitude:  0.05,
    pulseFrequency:  0.0,
    trailAlpha:      0.15,
    saturationBoost: 0.30,
  },

  // ── PARAMETERS — 金色脉冲 ─────────────────────────────────────────────────
  PARAMETERS: {
    speedBase:       0.08,
    speedPerMps:     0.50,
    speedMax:        0.70,
    densityBase:     0.15,
    densityPerMps:   0.55,
    densityMax:      0.75,
    alphaBase:       0.25,
    alphaFull:       0.80,
    radiusBase:      3.0,
    radiusTraffic:   1.6,
    curlScale:       0.8,
    pulseAmplitude:  0.40,
    pulseFrequency:  1.2,
    trailAlpha:      0.25,
    saturationBoost: 0.20,
  },

  // ── TF_STATIC — 绿色恒定 ──────────────────────────────────────────────────
  TF_STATIC: {
    speedBase:       0.05,
    speedPerMps:     0.20,
    speedMax:        0.30,
    densityBase:     0.20,
    densityPerMps:   0.40,
    densityMax:      0.65,
    alphaBase:       0.30,
    alphaFull:       0.70,
    radiusBase:      4.0,
    radiusTraffic:   1.2,
    curlScale:       0.3,
    pulseAmplitude:  0.0,
    pulseFrequency:  0.0,
    trailAlpha:      0.40,
    saturationBoost: 0.10,
  },

  // ── TOPO_CHANGE — 红色爆发 ────────────────────────────────────────────────
  TOPO_CHANGE: {
    speedBase:       0.10,
    speedPerMps:     1.80,
    speedMax:        2.20,
    densityBase:     0.05,
    densityPerMps:   0.90,
    densityMax:      1.00,
    alphaBase:       0.15,
    alphaFull:       1.00,
    radiusBase:      2.5,
    radiusTraffic:   2.0,
    curlScale:       2.5,
    pulseAmplitude:  0.0,
    pulseFrequency:  0.0,
    trailAlpha:      0.08,
    saturationBoost: 0.45,
  },

  // ── DEFAULT — neutral slate ───────────────────────────────────────────────
  DEFAULT: {
    speedBase:       0.10,
    speedPerMps:     0.60,
    speedMax:        0.80,
    densityBase:     0.10,
    densityPerMps:   0.60,
    densityMax:      0.80,
    alphaBase:       0.22,
    alphaFull:       0.75,
    radiusBase:      2.5,
    radiusTraffic:   1.4,
    curlScale:       1.0,
    pulseAmplitude:  0.10,
    pulseFrequency:  0.5,
    trailAlpha:      0.20,
    saturationBoost: 0.15,
  },
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D control point. */
export interface EnergyPoint {
  x: number;
  y: number;
}

/** One topology edge for the energy flow visualiser. */
export interface EnergyEdge {
  edgeId:    string;
  sourceId:  string;
  targetId:  string;
  /** Catmull-Rom control points (≥ 2, domain space). */
  points:    EnergyPoint[];
  /** QoS profile name; defaults to 'DEFAULT'. */
  qos?:      QoSProfileName;
  /** Static weight (affects base particle allocation). Default 1. */
  weight?:   number;
}

/** Live traffic metric pushed per-edge. */
export interface TrafficMetric {
  /** Messages per second (0 = idle). */
  mps:          number;
  /** Burst intensity [0, 1] — spikes density/brightness momentarily. */
  burstFactor?: number;
}

/** Configuration for EdgeEnergyFlow. */
export interface EdgeEnergyFlowConfig {
  /** Edge definitions. */
  edges:               EnergyEdge[];
  /** Global brightness multiplier [0, 1]. Default 1.0. */
  brightness?:         number;
  /** Enable idle breathing pulsation. Default true. */
  enableBreathing?:    boolean;
  /** Max mps value for normalisation (mps / maxMps → [0, 1]). Default 100. */
  maxMps?:             number;
  /** Global particle size multiplier. Default 1.0. */
  particleSizeScale?:  number;
  /** Maximum particles per edge override. Default MAX_PARTICLES_PER_EDGE. */
  maxParticlesPerEdge?: number;
  /** Canvas composite operation. Default 'lighter' (additive). */
  compositeOp?:        GlobalCompositeOperation;
}

// ─── Internal types ───────────────────────────────────────────────────────────

/** A single energy particle slot. */
interface EnergyParticle {
  /** Normalised travel [0, 1] along the spline. */
  travel:     number;
  /** Current speed (normalised units / second). */
  speed:      number;
  /** Lateral offset from spline centre (domain units). */
  offset:     number;
  /** Opacity [0, 1]. */
  alpha:      number;
  /** Random seed for noise. */
  seed:       number;
  /** Is this slot alive? */
  alive:      boolean;
  /** Current world X. */
  x:          number;
  /** Current world Y. */
  y:          number;
  /** Previous world X (for trail). */
  prevX:      number;
  /** Previous world Y (for trail). */
  prevY:      number;
}

/** Per-edge runtime state. */
interface EdgeState {
  edge:           EnergyEdge;
  profile:        FluidProfile;
  theme:          ThemePalette;
  /** Allocated particle slots for this edge. */
  particles:      EnergyParticle[];
  /** Pre-computed spline sample points. */
  samples:        EnergyPoint[];
  /** Pre-computed tangent normals at each sample. */
  normals:        EnergyPoint[];
  /** Total arc length (domain units). */
  arcLength:      number;
  /** Cumulative arc length LUT (length = SPLINE_SEGMENTS + 1). */
  arcLUT:         Float32Array;
  /** Smoothed current traffic metric. */
  smoothMps:      number;
  /** Smoothed burst factor. */
  smoothBurst:    number;
  /** Raw target mps (set via setTraffic). */
  targetMps:      number;
  /** Raw burst factor (set via triggerBurst, decays over time). */
  burstEnergy:    number;
  /** Breathing phase accumulator. */
  breathPhase:    number;
}

// ─── Catmull-Rom helpers ──────────────────────────────────────────────────────

function catmullRom(
  p0: EnergyPoint, p1: EnergyPoint,
  p2: EnergyPoint, p3: EnergyPoint,
  t: number,
): EnergyPoint {
  const t2 = t * t, t3 = t2 * t;
  const f1 = -0.5 * t3 + t2        - 0.5 * t;
  const f2 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const f4 =  0.5 * t3 - 0.5 * t2;
  return {
    x: f1 * p0.x + f2 * p1.x + f3 * p2.x + f4 * p3.x,
    y: f1 * p0.y + f2 * p1.y + f3 * p2.y + f4 * p3.y,
  };
}

function evalSpline(points: EnergyPoint[], u: number): EnergyPoint {
  const n = points.length;
  if (n === 0) return { x: 0, y: 0 };
  if (n === 1) return { x: points[0].x, y: points[0].y };
  const clamp = (i: number) => Math.max(0, Math.min(n - 1, i));
  const sc = Math.min(u, 0.9999) * (n - 1);
  const i1 = Math.floor(sc);
  return catmullRom(
    points[clamp(i1 - 1)], points[clamp(i1)],
    points[clamp(i1 + 1)], points[clamp(i1 + 2)],
    sc - i1,
  );
}

/** Build arc-length LUT and sample/normal arrays for a spline. */
function buildSplineData(points: EnergyPoint[]): {
  samples:   EnergyPoint[];
  normals:   EnergyPoint[];
  arcLUT:    Float32Array;
  arcLength: number;
} {
  const N       = SPLINE_SEGMENTS;
  const samples: EnergyPoint[] = new Array(N + 1);
  const normals: EnergyPoint[] = new Array(N + 1);
  const arcLUT  = new Float32Array(N + 1);

  samples[0] = evalSpline(points, 0);
  arcLUT[0]  = 0;

  let cumLen = 0;
  for (let i = 1; i <= N; i++) {
    const u = i / N;
    const p = evalSpline(points, u);
    samples[i] = p;
    const dx = p.x - samples[i - 1].x;
    const dy = p.y - samples[i - 1].y;
    cumLen += Math.sqrt(dx * dx + dy * dy);
    arcLUT[i] = cumLen;
  }

  // Compute perpendicular normals from tangent direction
  for (let i = 0; i <= N; i++) {
    const prev = samples[Math.max(0, i - 1)];
    const next = samples[Math.min(N, i + 1)];
    const tx   = next.x - prev.x;
    const ty   = next.y - prev.y;
    const len  = Math.sqrt(tx * tx + ty * ty) + 1e-10;
    // Perpendicular: rotate tangent 90° CCW
    normals[i] = { x: -ty / len, y: tx / len };
  }

  return { samples, normals, arcLUT, arcLength: cumLen };
}

/** Inverse arc-length parameterisation: uniform distance → spline u. */
function arcLengthToU(arcLUT: Float32Array, totalLen: number, dist: number): number {
  const target = dist * totalLen;
  const N      = arcLUT.length - 1;

  // Binary search for the segment containing target
  let lo = 0, hi = N;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (arcLUT[mid] <= target) lo = mid; else hi = mid;
  }

  const segLen = arcLUT[hi] - arcLUT[lo];
  const frac   = segLen > 1e-8 ? (target - arcLUT[lo]) / segLen : 0;
  return (lo + frac) / N;
}

// ─── Simple 2-D curl noise (CPU) ─────────────────────────────────────────────

function hash(x: number, y: number): number {
  let n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

function noise2d(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a  = hash(ix,     iy);
  const b  = hash(ix + 1, iy);
  const c  = hash(ix,     iy + 1);
  const d  = hash(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function curlNoise(x: number, y: number, t: number): { dx: number; dy: number } {
  const eps = 0.5;
  const n   = noise2d(x, y + eps + t * 0.1);
  const s   = noise2d(x, y - eps + t * 0.1);
  const e   = noise2d(x + eps, y + t * 0.1);
  const w   = noise2d(x - eps, y + t * 0.1);
  // ∂Ψ/∂y for curl x, −∂Ψ/∂x for curl y (divergence-free)
  return {
    dx:  (n - s) / (2 * eps),
    dy: -(e - w) / (2 * eps),
  };
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

function lerpRGB(a: RGB, b: RGB, t: number): RGB {
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

function saturateRGB(c: RGB, boost: number): RGB {
  const lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  return {
    r: Math.min(1, lum + (c.r - lum) * (1 + boost)),
    g: Math.min(1, lum + (c.g - lum) * (1 + boost)),
    b: Math.min(1, lum + (c.b - lum) * (1 + boost)),
  };
}

function rgbToCssRGBA(c: RGB, a: number): string {
  const r = Math.round(Math.min(1, Math.max(0, c.r)) * 255);
  const g = Math.round(Math.min(1, Math.max(0, c.g)) * 255);
  const b = Math.round(Math.min(1, Math.max(0, c.b)) * 255);
  return `rgba(${r},${g},${b},${Math.min(1, Math.max(0, a)).toFixed(3)})`;
}

// ─── EdgeEnergyFlow ───────────────────────────────────────────────────────────

/**
 * EdgeEnergyFlow
 *
 * CPU-driven energy stream visualiser that maps data traffic metrics onto a
 * particle fluid layer along topology edge splines.
 *
 * Each edge maintains a pool of EnergyParticle slots.  Traffic intensity
 * (mps, bursts) drives how many particles are alive, how fast they move,
 * and how bright they glow.  The QoS profile selects a FluidProfile that
 * tailors the visual character — turbulent blue torrents for SENSOR_DATA,
 * pulsating gold waves for PARAMETERS, serene green streams for TF_STATIC,
 * explosive red bursts for TOPO_CHANGE.
 *
 * When traffic drops to zero, particles don't vanish — they enter a gentle
 * breathing mode with slow sinusoidal alpha/speed modulation, maintaining
 * spatial awareness of the network topology.
 */
export class EdgeEnergyFlow {
  private readonly ctx:        CanvasRenderingContext2D;
  private readonly cfg:        Required<Omit<EdgeEnergyFlowConfig, 'edges'>>;
  private readonly edgeMap:    Map<string, EdgeState> = new Map();
  private readonly edgeStates: EdgeState[] = [];
  private elapsed = 0;
  private destroyed = false;

  constructor(
    ctx:    CanvasRenderingContext2D,
    config: EdgeEnergyFlowConfig,
  ) {
    this.ctx = ctx;
    this.cfg = {
      brightness:          config.brightness          ?? 1.0,
      enableBreathing:     config.enableBreathing     ?? true,
      maxMps:              config.maxMps              ?? 100,
      particleSizeScale:   config.particleSizeScale   ?? 1.0,
      maxParticlesPerEdge: config.maxParticlesPerEdge ?? MAX_PARTICLES_PER_EDGE,
      compositeOp:         config.compositeOp         ?? 'lighter',
    };

    this._initEdges(config.edges);
  }

  // ── Edge initialisation ────────────────────────────────────────────────────

  private _initEdges(edges: EnergyEdge[]): void {
    let totalSlots = 0;

    for (const edge of edges) {
      if (edge.points.length < 2) continue;

      const qos     = edge.qos ?? 'DEFAULT';
      const profile = FLUID_PROFILES[qos] ?? FLUID_PROFILES.DEFAULT;
      const theme   = QOS_THEME[qos]      ?? QOS_THEME.DEFAULT;
      const weight  = edge.weight ?? 1;

      // Allocate particle slots proportional to weight, capped
      const slotCount = Math.min(
        this.cfg.maxParticlesPerEdge,
        Math.max(8, Math.round(weight * 32)),
      );

      // Stop if global pool would overflow
      if (totalSlots + slotCount > MAX_TOTAL_PARTICLES) break;
      totalSlots += slotCount;

      const { samples, normals, arcLUT, arcLength } = buildSplineData(edge.points);

      const particles: EnergyParticle[] = [];
      for (let i = 0; i < slotCount; i++) {
        particles.push(this._makeParticle(i, slotCount));
      }

      const state: EdgeState = {
        edge,
        profile,
        theme,
        particles,
        samples,
        normals,
        arcLength,
        arcLUT,
        smoothMps:    0,
        smoothBurst:  0,
        targetMps:    0,
        burstEnergy:  0,
        breathPhase:  Math.random() * Math.PI * 2,  // stagger edges
      };

      this.edgeMap.set(edge.edgeId, state);
      this.edgeStates.push(state);
    }
  }

  private _makeParticle(index: number, total: number): EnergyParticle {
    // Stagger initial positions along the spline for visual diversity
    const travel = (index / Math.max(1, total)) + (Math.random() * 0.05);
    return {
      travel:  travel % 1,
      speed:   IDLE_SPEED,
      offset:  0,
      alpha:   0,
      seed:    Math.random() * 1000,
      alive:   false,
      x:       0,
      y:       0,
      prevX:   0,
      prevY:   0,
    };
  }

  // ── Public API — traffic updates ───────────────────────────────────────────

  /**
   * Push a live traffic metric for an edge.
   * mps and burstFactor are smoothed internally to avoid visual pops.
   */
  setTraffic(edgeId: string, metric: TrafficMetric): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;
    state.targetMps = Math.max(0, metric.mps);
    if (metric.burstFactor !== undefined && metric.burstFactor > state.burstEnergy) {
      state.burstEnergy = Math.min(1, metric.burstFactor);
    }
  }

  /**
   * Fire an immediate burst event on an edge.
   * Spikes density and brightness for a brief flash, then decays.
   *
   * @param edgeId    — topology edge identifier
   * @param intensity — burst intensity [0, 1]; 1 = maximum
   */
  triggerBurst(edgeId: string, intensity = 1.0): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;
    state.burstEnergy = Math.min(1, Math.max(state.burstEnergy, intensity));
  }

  /**
   * Bulk-update all edge traffic metrics at once.
   *
   * @param metrics — Map of edgeId → TrafficMetric
   */
  setAllTraffic(metrics: Map<string, TrafficMetric> | Record<string, TrafficMetric>): void {
    const entries = metrics instanceof Map ? metrics.entries() : Object.entries(metrics);
    for (const [id, metric] of entries) {
      this.setTraffic(id, metric);
    }
  }

  // ── Per-frame update ───────────────────────────────────────────────────────

  /**
   * Advance all particle states by dt seconds.
   *
   * @param dt      — frame delta time in seconds
   * @param elapsed — total elapsed seconds (for noise / breathing phase)
   */
  update(dt: number, elapsed: number): void {
    if (this.destroyed) return;
    this.elapsed = elapsed;
    const clampDt = Math.min(dt, 1 / 15);  // cap at ~15fps minimum

    for (const state of this.edgeStates) {
      this._updateEdge(state, clampDt, elapsed);
    }
  }

  private _updateEdge(state: EdgeState, dt: number, elapsed: number): void {
    const { profile } = state;
    const maxMps = this.cfg.maxMps;

    // ── Smooth traffic metrics ─────────────────────────────────────────────
    const smoothRate = 1 - Math.exp(-METRIC_SMOOTHING * dt);
    state.smoothMps   += (state.targetMps - state.smoothMps) * smoothRate;
    state.smoothBurst += (state.burstEnergy - state.smoothBurst) * smoothRate;

    // Decay burst energy
    state.burstEnergy = Math.max(0, state.burstEnergy - BURST_DECAY * dt);

    // Normalised traffic intensity [0, 1]
    const mpsNorm   = Math.min(1, state.smoothMps / maxMps);
    const burstNorm = state.smoothBurst;

    // ── Derive flow parameters from traffic + QoS profile ──────────────────
    const trafficIntensity = Math.min(1, mpsNorm + burstNorm * 0.5);
    const isIdle           = mpsNorm < 0.01 && burstNorm < 0.01;

    // Breathing modulation for idle state
    state.breathPhase += (2 * Math.PI / BREATH_PERIOD) * dt;
    const breathWave = 0.5 + 0.5 * Math.sin(state.breathPhase);

    // Target speed
    let targetSpeed = profile.speedBase + profile.speedPerMps * mpsNorm;
    targetSpeed     = Math.min(profile.speedMax, targetSpeed);
    if (burstNorm > 0.1) {
      targetSpeed = Math.min(profile.speedMax, targetSpeed + burstNorm * profile.speedMax * 0.5);
    }
    if (isIdle && this.cfg.enableBreathing) {
      targetSpeed = IDLE_SPEED + breathWave * IDLE_SPEED * 0.5;
    }

    // Target density (fraction of allocated slots that should be alive)
    let targetDensity = profile.densityBase + profile.densityPerMps * mpsNorm;
    targetDensity     = Math.min(profile.densityMax, targetDensity);
    if (burstNorm > 0.1) {
      targetDensity = Math.min(1, targetDensity + burstNorm * (1 - targetDensity) * 0.8);
    }
    if (isIdle && this.cfg.enableBreathing) {
      targetDensity = IDLE_DENSITY_RATIO + breathWave * IDLE_DENSITY_RATIO * 0.3;
    }

    // Pulse modulation (PARAMETERS heartbeat rhythm)
    let pulseMod = 1.0;
    if (profile.pulseAmplitude > 0 && profile.pulseFrequency > 0 && !isIdle) {
      const pulsePhase = elapsed * profile.pulseFrequency * 2 * Math.PI;
      pulseMod = 1 + profile.pulseAmplitude * Math.sin(pulsePhase) * mpsNorm;
    }

    // Target alpha
    let targetAlpha = profile.alphaBase + (profile.alphaFull - profile.alphaBase) * trafficIntensity;
    targetAlpha    *= pulseMod;
    if (isIdle && this.cfg.enableBreathing) {
      targetAlpha = BREATH_ALPHA_MIN + (BREATH_ALPHA_MAX - BREATH_ALPHA_MIN) * breathWave;
    }
    targetAlpha *= this.cfg.brightness;

    // ── Tick particles ──────────────────────────────────────────────────────
    const { particles, samples, normals, arcLUT, arcLength, edge } = state;
    const totalSlots     = particles.length;
    const targetAlive    = Math.max(1, Math.round(totalSlots * targetDensity));
    let   currentAlive   = 0;

    for (let i = 0; i < totalSlots; i++) {
      const p = particles[i];

      if (p.alive) {
        currentAlive++;

        // Save previous position for trail
        p.prevX = p.x;
        p.prevY = p.y;

        // Advance travel
        const speedSmooth = p.speed + (targetSpeed - p.speed) * Math.min(1, 3 * dt);
        p.speed  = speedSmooth;
        p.travel += speedSmooth * dt;

        // Alpha approach
        p.alpha += (targetAlpha - p.alpha) * Math.min(1, 5 * dt);

        // Curl-noise lateral perturbation
        const curl = curlNoise(
          p.x * CURL_FREQUENCY,
          p.y * CURL_FREQUENCY,
          elapsed + p.seed,
        );
        p.offset += curl.dx * CURL_AMPLITUDE * profile.curlScale * dt;
        p.offset *= 0.95;  // damping

        // Evaluate spline position
        if (p.travel >= 1.0) {
          // Respawn at start
          p.travel = p.travel % 1;
          p.offset = 0;
          p.prevX  = p.x;
          p.prevY  = p.y;
        }

        const u   = arcLengthToU(arcLUT, arcLength, p.travel);
        const seg = Math.min(SPLINE_SEGMENTS, Math.max(0, Math.round(u * SPLINE_SEGMENTS)));
        const sp  = evalSpline(edge.points, u);
        const nrm = normals[seg] ?? normals[0];

        p.x = sp.x + nrm.x * p.offset;
        p.y = sp.y + nrm.y * p.offset;

        // Kill excess particles (if density dropped)
        if (currentAlive > targetAlive + 4) {
          p.alpha -= 2 * dt;
          if (p.alpha <= 0) {
            p.alive = false;
            p.alpha = 0;
            currentAlive--;
          }
        }
      } else {
        // Spawn particles to meet target density
        if (currentAlive < targetAlive) {
          p.alive  = true;
          p.travel = Math.random();
          p.speed  = targetSpeed * (0.8 + Math.random() * 0.4);
          p.alpha  = 0;  // fade in
          p.offset = 0;
          p.seed   = Math.random() * 1000;

          const u0  = arcLengthToU(arcLUT, arcLength, p.travel);
          const sp0 = evalSpline(edge.points, u0);
          p.x     = sp0.x;
          p.y     = sp0.y;
          p.prevX = sp0.x;
          p.prevY = sp0.y;
          currentAlive++;
        }
      }
    }
  }

  // ── Draw ───────────────────────────────────────────────────────────────────

  /**
   * Render all energy flow particles onto the Canvas2D context.
   * Call after update().
   */
  draw(): void {
    if (this.destroyed) return;
    const { ctx, cfg } = this;

    const prevOp = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = cfg.compositeOp;
    ctx.save();

    for (const state of this.edgeStates) {
      this._drawEdge(state);
    }

    ctx.restore();
    ctx.globalCompositeOperation = prevOp;
  }

  private _drawEdge(state: EdgeState): void {
    const { ctx, cfg, elapsed } = this;
    const { profile, theme, particles } = state;

    // Traffic-dependent colour computation
    const mpsNorm        = Math.min(1, state.smoothMps / cfg.maxMps);
    const burstNorm      = state.smoothBurst;
    const trafficIntensity = Math.min(1, mpsNorm + burstNorm * 0.5);

    // Blend base → highlight based on traffic
    const baseCol = lerpRGB(theme.base, theme.highlight, trafficIntensity * 0.6);
    // Apply saturation boost at high traffic
    const saturated = saturateRGB(baseCol, profile.saturationBoost * trafficIntensity);

    // Particle radius
    const radius = profile.radiusBase *
                   (1 + (profile.radiusTraffic - 1) * trafficIntensity) *
                   cfg.particleSizeScale;

    for (const p of particles) {
      if (!p.alive || p.alpha < 0.005) continue;

      // ── Trail segment ──────────────────────────────────────────────────
      if (profile.trailAlpha > 0) {
        const dx = p.x - p.prevX;
        const dy = p.y - p.prevY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0.5) {
          const trailA = p.alpha * profile.trailAlpha * 0.5;
          ctx.beginPath();
          ctx.moveTo(p.prevX, p.prevY);
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = rgbToCssRGBA(saturated, trailA);
          ctx.lineWidth   = radius * 0.6;
          ctx.lineCap     = 'round';
          ctx.stroke();
        }
      }

      // ── Core particle glow ─────────────────────────────────────────────
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
      grad.addColorStop(0,   rgbToCssRGBA(theme.highlight, p.alpha * 0.9));
      grad.addColorStop(0.4, rgbToCssRGBA(saturated,       p.alpha * 0.6));
      grad.addColorStop(1,   rgbToCssRGBA(saturated,       0));

      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  // ── Edge management ────────────────────────────────────────────────────────

  /**
   * Add a new edge at runtime (e.g. after topology change).
   */
  addEdge(edge: EnergyEdge): void {
    if (this.edgeMap.has(edge.edgeId) || edge.points.length < 2) return;

    const qos     = edge.qos ?? 'DEFAULT';
    const profile = FLUID_PROFILES[qos] ?? FLUID_PROFILES.DEFAULT;
    const theme   = QOS_THEME[qos]      ?? QOS_THEME.DEFAULT;
    const weight  = edge.weight ?? 1;
    const slotCount = Math.min(
      this.cfg.maxParticlesPerEdge,
      Math.max(8, Math.round(weight * 32)),
    );

    const { samples, normals, arcLUT, arcLength } = buildSplineData(edge.points);
    const particles: EnergyParticle[] = [];
    for (let i = 0; i < slotCount; i++) {
      particles.push(this._makeParticle(i, slotCount));
    }

    const state: EdgeState = {
      edge, profile, theme, particles, samples, normals,
      arcLength, arcLUT,
      smoothMps: 0, smoothBurst: 0,
      targetMps: 0, burstEnergy: 0,
      breathPhase: Math.random() * Math.PI * 2,
    };

    this.edgeMap.set(edge.edgeId, state);
    this.edgeStates.push(state);
  }

  /**
   * Remove an edge at runtime.
   */
  removeEdge(edgeId: string): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;
    this.edgeMap.delete(edgeId);
    const idx = this.edgeStates.indexOf(state);
    if (idx >= 0) this.edgeStates.splice(idx, 1);
  }

  /**
   * Update the QoS profile for an existing edge (e.g. dynamic re-profiling).
   */
  setEdgeQoS(edgeId: string, qos: QoSProfileName): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;
    state.profile = FLUID_PROFILES[qos] ?? FLUID_PROFILES.DEFAULT;
    state.theme   = QOS_THEME[qos]      ?? QOS_THEME.DEFAULT;
    state.edge.qos = qos;
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  /** Number of registered edges. */
  get edgeCount(): number { return this.edgeStates.length; }

  /** Total alive particles across all edges. */
  get aliveParticleCount(): number {
    let count = 0;
    for (const s of this.edgeStates) {
      for (const p of s.particles) {
        if (p.alive) count++;
      }
    }
    return count;
  }

  /** Total allocated particle slots. */
  get totalSlots(): number {
    let count = 0;
    for (const s of this.edgeStates) count += s.particles.length;
    return count;
  }

  /** Get current smoothed traffic for an edge (for debug / overlay). */
  getEdgeTraffic(edgeId: string): { mps: number; burst: number; density: number } | null {
    const state = this.edgeMap.get(edgeId);
    if (!state) return null;
    const density = state.particles.filter(p => p.alive).length / state.particles.length;
    return {
      mps:     state.smoothMps,
      burst:   state.smoothBurst,
      density,
    };
  }

  /** Snapshot all fluid profiles (read-only). */
  get fluidProfiles(): Readonly<Record<QoSProfileName, FluidProfile>> {
    return FLUID_PROFILES;
  }

  // ── Global parameter tweaks ────────────────────────────────────────────────

  setBrightness(v: number): void       { this.cfg.brightness = Math.max(0, Math.min(1, v)); }
  setParticleSizeScale(v: number): void { this.cfg.particleSizeScale = Math.max(0.1, v); }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy(): void {
    this.destroyed = true;
    this.edgeMap.clear();
    this.edgeStates.length = 0;
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create an EdgeEnergyFlow pre-wired with a pub/sub event bus.
 *
 * Subscribes to `edge:message` events and converts them to traffic pulses.
 * Subscribes to `edge:burst` events and fires burst triggers.
 *
 * @example
 * ```ts
 * const flow = createEdgeEnergyFlowWithBus(ctx, edges, eventBus);
 * // eventBus.emit('edge:message', { edgeId: 'e1', mps: 50 });
 * // eventBus.emit('edge:burst',   { edgeId: 'e2', intensity: 0.8 });
 * // render loop: flow.update(dt, elapsed); flow.draw();
 * ```
 */
export function createEdgeEnergyFlowWithBus(
  ctx:   CanvasRenderingContext2D,
  edges: EnergyEdge[],
  bus:   {
    on: (event: string, handler: (data: any) => void) => void;
    off?: (event: string, handler: (data: any) => void) => void;
  },
  config?: Omit<EdgeEnergyFlowConfig, 'edges'>,
): EdgeEnergyFlow {
  const flow = new EdgeEnergyFlow(ctx, { ...config, edges });

  const onMessage = (data: { edgeId: string; mps: number }) => {
    flow.setTraffic(data.edgeId, { mps: data.mps });
  };
  const onBurst = (data: { edgeId: string; intensity?: number }) => {
    flow.triggerBurst(data.edgeId, data.intensity ?? 1.0);
  };

  bus.on('edge:message', onMessage);
  bus.on('edge:burst', onBurst);

  // Patch destroy to unsubscribe
  const origDestroy = flow.destroy.bind(flow);
  flow.destroy = () => {
    bus.off?.('edge:message', onMessage);
    bus.off?.('edge:burst', onBurst);
    origDestroy();
  };

  return flow;
}

// ─── Exported constants ───────────────────────────────────────────────────────

export { FLUID_PROFILES };

export const EDGE_ENERGY_FLOW_DEFAULTS = {
  maxParticlesPerEdge: MAX_PARTICLES_PER_EDGE,
  maxTotalParticles:   MAX_TOTAL_PARTICLES,
  breathPeriod:        BREATH_PERIOD,
  idleDensityRatio:    IDLE_DENSITY_RATIO,
  idleSpeed:           IDLE_SPEED,
  curlAmplitude:       CURL_AMPLITUDE,
  curlFrequency:       CURL_FREQUENCY,
  metricSmoothing:     METRIC_SMOOTHING,
  burstDecay:          BURST_DECAY,
} as const;
