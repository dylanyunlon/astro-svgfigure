/**
 * neural-pathway-renderer.ts — M763
 *
 * Neural Pathway Edge Renderer: edges as biological neural synapses.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Visual metaphor:
 *
 *   ╔══════════════════════════════════════════════════════════════════════╗
 *   ║  [Source Cell] ──axon──╢myelin╟──╢myelin╟──▸ synapse ──▸ [Target] ║
 *   ║                  ⚡pulse  ⊙vesicle   ⊙vesicle    ⚡pulse           ║
 *   ╚══════════════════════════════════════════════════════════════════════╝
 *
 *   1. **Axon tube** — semi-transparent tubular corridor drawn along the
 *      Catmull-Rom spline with inner/outer stroke gradients and bioluminescent
 *      glow. Width modulated by edge weight (heavier connections → thicker axon).
 *
 *   2. **Myelin sheath** — periodic capsule segments along the axon with subtle
 *      pearlescent sheen. Gaps between segments are Nodes of Ranvier where
 *      electric pulses jump (saltatory conduction visualised).
 *
 *   3. **Electric signal pulses** — bright, fast particles that leap node-to-node
 *      with a bright flash at each Ranvier gap. Colour maps to QoS profile.
 *      Action-potential waveform modulates brightness (depolarisation spike →
 *      repolarisation dip → refractory fade).
 *
 *   4. **Synaptic vesicles** — slower, larger translucent spheroids that drift
 *      along the axon interior carrying "neurotransmitter" payload. On arrival
 *      at the synaptic cleft (spline end), they burst into a radial scatter of
 *      micro-particles absorbed by the target cell — modelling exocytosis.
 *
 *   5. **Spline flow particles** — fine-grained particles flowing inside the
 *      axon tube with curl-noise perturbation (reuses AT SplineParticleLife
 *      conventions). These represent background ion channel activity and give
 *      the axon a living, pulsing interior texture.
 *
 * QoS → neural style mapping:
 *
 *   SENSOR_DATA  — rapid myelinated axon, frequent action potentials (blue)
 *   PARAMETERS   — thick unmyelinated axon, slow vesicle drift (amber)
 *   TF_STATIC    — heavily myelinated, rare pulses, dense vesicles (green)
 *   TOPO_CHANGE  — demyelinating axon (gaps flicker), burst discharges (magenta)
 *   DEFAULT      — balanced mixed-mode axon (slate)
 *
 * Dual-mode rendering:
 *
 *   - **Canvas2D** — full software path with globalCompositeOperation layering
 *   - **GPU hint** — optional GPUDevice for future compute-driven particle pass
 *     (currently the Canvas2D path handles all rendering; GPU path is stubbed
 *     for M764+ WebGPU migration)
 *
 * Integration:
 *   ```ts
 *   import { NeuralPathwayRenderer } from '$lib/sph/neural-pathway-renderer';
 *
 *   const neural = NeuralPathwayRenderer.create(ctx2d, {
 *     edges: topology.edges,
 *     onVesicleArrival: (edgeId, targetId, x, y) => { … },
 *   });
 *
 *   // render loop:
 *   neural.update(elapsed, dt);
 *   neural.draw();
 *   ```
 *
 * References:
 *   src/lib/sph/edge-flow-renderer.ts   — base edge-flow system
 *   src/lib/sph/spline-particle-life.ts — spline particle lifecycle
 *   src/lib/sph/color-palette.ts        — QoS colour themes
 *   src/lib/sph/qosSpatial.ts           — QoS profile definitions
 */




// ─── Constants ────────────────────────────────────────────────────────────────

/** Max electric pulse particles across all edges. */



import type { QoSProfile }           from './types';
import type { QoSProfileName }       from './qosSpatial';
import { QOS_PRESETS }                from './qosSpatial';
import { QOS_THEME }                  from './color-palette';
import type { ThemePalette, RGB }     from './color-palette';

const MAX_PULSES           = 2048;
/** Max synaptic vesicle particles across all edges. */
const MAX_VESICLES         = 512;
/** Max background flow particles (fine-grained interior). */
const MAX_FLOW_PARTICLES   = 4096;
/** Max burst micro-particles from vesicle exocytosis. */
const MAX_BURST_PARTICLES  = 1024;

/** Catmull-Rom tension. */
const CR_TENSION           = 0.5;
/** Curl-noise evaluation epsilon. */
const CURL_EPS             = 0.01;
/** Myelin sheath segment length (domain units). */
const MYELIN_SEGMENT_LEN   = 0.08;
/** Ranvier node gap ratio (fraction of segment). */
const RANVIER_GAP_RATIO    = 0.18;
/** Action-potential waveform duration (seconds). */
const AP_DURATION          = 0.12;
/** Vesicle burst scatter radius. */
const BURST_SCATTER_RADIUS = 12;
/** Burst particle lifetime. */
const BURST_LIFETIME       = 0.4;

// ─── Lifecycle phases ─────────────────────────────────────────────────────────

export const enum PulsePhase {
  SPAWN    = 0,
  CONDUCT  = 1, // saltatory conduction — leaps node to node
  DECAY    = 2,
  DEAD     = 3,
}

export const enum VesiclePhase {
  SPAWN    = 0,
  TRANSIT  = 1, // drifting along axon interior
  BURST    = 2, // exocytosis at synaptic cleft
  DEAD     = 3,
}

export const enum FlowPhase {
  SPAWN    = 0,
  FLOW     = 1,
  DECAY    = 2,
  DEAD     = 3,
}

// ─── QoS → Neural Style ──────────────────────────────────────────────────────

export interface NeuralStyle {
  /** Axon tube base radius (half-width) in px. */
  axonRadius:         number;
  /** Myelin coverage ratio [0, 1] — 0 = unmyelinated, 1 = fully wrapped. */
  myelinCoverage:     number;
  /** Myelin segment count per spline unit length. */
  myelinDensity:      number;
  /** Myelin opacity. */
  myelinOpacity:      number;
  /** Electric pulse speed (travel units / s). */
  pulseSpeed:         [number, number];
  /** Pulse spawn interval (seconds). */
  pulseInterval:      number;
  /** Pulse glow radius. */
  pulseRadius:        number;
  /** Pulse peak brightness. */
  pulseBrightness:    number;
  /** Vesicle drift speed. */
  vesicleSpeed:       [number, number];
  /** Vesicle spawn interval (seconds). */
  vesicleInterval:    number;
  /** Vesicle radius. */
  vesicleRadius:      number;
  /** Background flow particle count per edge. */
  flowDensity:        number;
  /** Flow speed. */
  flowSpeed:          [number, number];
  /** Flow curl noise strength. */
  curlStrength:       number;
  /** Ranvier flash intensity. */
  ranvierFlash:       number;
}

const NEURAL_STYLES: Record<QoSProfileName, NeuralStyle> = {
  SENSOR_DATA: {
    axonRadius:      3.0,
    myelinCoverage:  0.85,
    myelinDensity:   14,
    myelinOpacity:   0.45,
    pulseSpeed:      [0.6, 0.9],
    pulseInterval:   0.15,
    pulseRadius:     4.5,
    pulseBrightness: 1.0,
    vesicleSpeed:    [0.08, 0.15],
    vesicleInterval: 1.2,
    vesicleRadius:   5.0,
    flowDensity:     16,
    flowSpeed:       [0.25, 0.45],
    curlStrength:    0.012,
    ranvierFlash:    1.0,
  },
  PARAMETERS: {
    axonRadius:      4.5,
    myelinCoverage:  0.3,
    myelinDensity:   6,
    myelinOpacity:   0.25,
    pulseSpeed:      [0.2, 0.35],
    pulseInterval:   0.6,
    pulseRadius:     3.5,
    pulseBrightness: 0.7,
    vesicleSpeed:    [0.05, 0.10],
    vesicleInterval: 0.6,
    vesicleRadius:   7.0,
    flowDensity:     10,
    flowSpeed:       [0.12, 0.25],
    curlStrength:    0.035,
    ranvierFlash:    0.5,
  },
  TF_STATIC: {
    axonRadius:      3.5,
    myelinCoverage:  0.92,
    myelinDensity:   18,
    myelinOpacity:   0.55,
    pulseSpeed:      [0.15, 0.25],
    pulseInterval:   1.5,
    pulseRadius:     3.0,
    pulseBrightness: 0.5,
    vesicleSpeed:    [0.03, 0.07],
    vesicleInterval: 0.4,
    vesicleRadius:   8.0,
    flowDensity:     8,
    flowSpeed:       [0.06, 0.12],
    curlStrength:    0.06,
    ranvierFlash:    0.3,
  },
  TOPO_CHANGE: {
    axonRadius:      2.5,
    myelinCoverage:  0.4,
    myelinDensity:   10,
    myelinOpacity:   0.2,
    pulseSpeed:      [0.7, 1.0],
    pulseInterval:   0.08,
    pulseRadius:     5.5,
    pulseBrightness: 1.2,
    vesicleSpeed:    [0.12, 0.20],
    vesicleInterval: 0.8,
    vesicleRadius:   4.5,
    flowDensity:     20,
    flowSpeed:       [0.35, 0.55],
    curlStrength:    0.02,
    ranvierFlash:    1.4,
  },
  DEFAULT: {
    axonRadius:      3.5,
    myelinCoverage:  0.6,
    myelinDensity:   10,
    myelinOpacity:   0.35,
    pulseSpeed:      [0.35, 0.55],
    pulseInterval:   0.35,
    pulseRadius:     4.0,
    pulseBrightness: 0.8,
    vesicleSpeed:    [0.06, 0.12],
    vesicleInterval: 0.8,
    vesicleRadius:   6.0,
    flowDensity:     12,
    flowSpeed:       [0.18, 0.32],
    curlStrength:    0.03,
    ranvierFlash:    0.7,
  },
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D point on the spline. */
export interface NeuralPoint {
  x: number;
  y: number;
}

/** One edge definition for the neural renderer. */
export interface NeuralEdge {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Catmull-Rom control points (≥2, in canvas/domain space). */
  points:   NeuralPoint[];
  /** Connectivity weight — controls axon thickness & particle density. */
  weight:   number;
  /** QoS profile name; defaults to 'DEFAULT'. */
  qos?:     QoSProfileName;
}

/** Callback when a vesicle arrives & bursts at the synaptic cleft. */
export type OnVesicleArrivalFn = (
  edgeId: string, targetId: string, x: number, y: number
) => void;

/** Callback when an electric pulse reaches the terminal. */
export type OnPulseArrivalFn = (
  edgeId: string, targetId: string, x: number, y: number
) => void;

/** Renderer configuration. */
export interface NeuralPathwayConfig {
  edges:               NeuralEdge[];
  onVesicleArrival?:   OnVesicleArrivalFn;
  onPulseArrival?:     OnPulseArrivalFn;
  /** Global speed scale for all particles. */
  speedScale?:         number;
  /** Whether to draw axon tube background. Default: true. */
  drawAxon?:           boolean;
  /** Whether to draw myelin sheath segments. Default: true. */
  drawMyelin?:         boolean;
  /** Axon interior glow opacity. Default: 0.15. */
  axonGlowOpacity?:    number;
  /** Whether demyelination flicker is active for TOPO_CHANGE. Default: true. */
  demyelinationFlicker?: boolean;
}

// ─── Per-particle state structures ────────────────────────────────────────────

interface PulseParticle {
  edgeIndex:    number;
  travel:       number;  // [0, 1] along spline
  speed:        number;
  phase:        PulsePhase;
  alpha:        number;
  /** Action-potential waveform phase within AP_DURATION. */
  apTime:       number;
  /** Which Ranvier gap this pulse last flashed at. */
  lastRanvier:  number;
  x:            number;
  y:            number;
}

interface VesicleParticle {
  edgeIndex:    number;
  travel:       number;
  speed:        number;
  phase:        VesiclePhase;
  alpha:        number;
  /** Wobble phase for organic drift. */
  wobblePhase:  number;
  /** Lateral offset from spline centreline. */
  lateralOff:   number;
  x:            number;
  y:            number;
}

interface FlowParticleState {
  edgeIndex:    number;
  travel:       number;
  speed:        number;
  phase:        FlowPhase;
  alpha:        number;
  delay:        number;
  seed:         number;
  curlOff:      number;
  x:            number;
  y:            number;
}

interface BurstParticle {
  x:            number;
  y:            number;
  vx:           number;
  vy:           number;
  life:         number;
  maxLife:      number;
  radius:       number;
  edgeIndex:    number;
}

// ─── Precomputed spline data per edge ─────────────────────────────────────────

interface EdgeSplineCache {
  /** Arc lengths at N sample points. */
  arcLengths:  number[];
  /** Total arc length in domain units. */
  totalLength: number;
  /** Myelin segment boundaries as travel-fraction pairs [start, end]. */
  myelinSegs:  Array<[number, number]>;
  /** Ranvier gap centres (travel-fraction). */
  ranvierGaps: number[];
}

// ─── Spline math ──────────────────────────────────────────────────────────────

function evalCatmullRom(pts: NeuralPoint[], t: number): NeuralPoint {
  const n = pts.length;
  if (n < 2) return { x: pts[0]?.x ?? 0, y: pts[0]?.y ?? 0 };

  const tt  = Math.max(0, Math.min(1, t)) * (n - 1);
  const i   = Math.min(Math.floor(tt), n - 2);
  const f   = tt - i;

  const p0 = pts[Math.max(0, i - 1)];
  const p1 = pts[i];
  const p2 = pts[Math.min(i + 1, n - 1)];
  const p3 = pts[Math.min(i + 2, n - 1)];

  const tau = CR_TENSION;
  const f2  = f * f;
  const f3  = f2 * f;

  const h1 = -tau * f + 2 * tau * f2 - tau * f3;
  const h2 = 1 + (tau - 3) * f2 + (2 - tau) * f3;
  const h3 = tau * f + (3 - 2 * tau) * f2 + (tau - 2) * f3;
  const h4 = -tau * f2 + tau * f3;

  return {
    x: h1 * p0.x + h2 * p1.x + h3 * p2.x + h4 * p3.x,
    y: h1 * p0.y + h2 * p1.y + h3 * p2.y + h4 * p3.y,
  };
}

function splineTangent(pts: NeuralPoint[], t: number): NeuralPoint {
  const eps = 0.001;
  const a = evalCatmullRom(pts, Math.max(0, t - eps));
  const b = evalCatmullRom(pts, Math.min(1, t + eps));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  return { x: dx / len, y: dy / len };
}

function splineNormal(pts: NeuralPoint[], t: number): NeuralPoint {
  const tan = splineTangent(pts, t);
  return { x: -tan.y, y: tan.x };
}

// ─── Curl noise (AT simplenoise port — 2D) ───────────────────────────────────

function _hash(x: number, y: number, z: number): number {
  let h = (x * 374761393 + y * 668265263 + z * 1274126177) | 0;
  h = ((h ^ (h >>> 13)) * 1274126177) | 0;
  return (h ^ (h >>> 16)) / 2147483648;
}

function _smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

function _noise3(x: number, y: number, z: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const sx = _smooth(fx), sy = _smooth(fy), sz = _smooth(fz);

  const n000 = _hash(ix, iy, iz);
  const n100 = _hash(ix + 1, iy, iz);
  const n010 = _hash(ix, iy + 1, iz);
  const n110 = _hash(ix + 1, iy + 1, iz);
  const n001 = _hash(ix, iy, iz + 1);
  const n101 = _hash(ix + 1, iy, iz + 1);
  const n011 = _hash(ix, iy + 1, iz + 1);
  const n111 = _hash(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + sx * (n100 - n000);
  const nx10 = n010 + sx * (n110 - n010);
  const nx01 = n001 + sx * (n101 - n001);
  const nx11 = n011 + sx * (n111 - n011);

  const nxy0 = nx00 + sy * (nx10 - nx00);
  const nxy1 = nx01 + sy * (nx11 - nx01);

  return nxy0 + sz * (nxy1 - nxy0);
}

function curlNoise2D(x: number, y: number, time: number): NeuralPoint {
  const eps = CURL_EPS;
  const dndx = (_noise3(x + eps, y, time) - _noise3(x - eps, y, time)) / (2 * eps);
  const dndy = (_noise3(x, y + eps, time) - _noise3(x, y - eps, time)) / (2 * eps);
  return { x: dndy, y: -dndx };
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

function themeFor(qos: QoSProfileName): ThemePalette {
  return QOS_THEME[qos] ?? QOS_THEME.DEFAULT;
}

function styleFor(qos: QoSProfileName): NeuralStyle {
  return NEURAL_STYLES[qos] ?? NEURAL_STYLES.DEFAULT;
}

function rgba(c: RGB, a: number): string {
  return `rgba(${(c.r * 255) | 0},${(c.g * 255) | 0},${(c.b * 255) | 0},${a})`;
}

function rgbaLerp(a: RGB, b: RGB, t: number): RGB {
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

function rgbBrighten(c: RGB, factor: number): RGB {
  return {
    r: Math.min(1, c.r + (1 - c.r) * factor),
    g: Math.min(1, c.g + (1 - c.g) * factor),
    b: Math.min(1, c.b + (1 - c.b) * factor),
  };
}

/** Random in [lo, hi). */
function rng(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

// ─── Action-potential waveform ────────────────────────────────────────────────
//
// Models a simplified Hodgkin-Huxley action potential:
//   t ∈ [0, 0.3] → depolarisation rise  (resting → peak)
//   t ∈ [0.3, 0.6] → repolarisation     (peak → undershoot)
//   t ∈ [0.6, 1.0] → refractory recovery (undershoot → resting)

function actionPotentialWave(phase01: number): number {
  if (phase01 < 0.3) {
    // depolarisation — sharp rise
    const t = phase01 / 0.3;
    return t * t * (3 - 2 * t); // smoothstep 0→1
  } else if (phase01 < 0.6) {
    // repolarisation — drop past resting
    const t = (phase01 - 0.3) / 0.3;
    return 1.0 - 1.3 * t * t * (3 - 2 * t); // overshoot to -0.3
  } else {
    // refractory — recovery toward resting
    const t = (phase01 - 0.6) / 0.4;
    return -0.3 + 0.3 * t * t * (3 - 2 * t); // back to ~0
  }
}

// ─── Spline cache builder ─────────────────────────────────────────────────────

function buildSplineCache(
  pts: NeuralPoint[],
  style: NeuralStyle,
): EdgeSplineCache {
  const SAMPLES = 128;
  const arcLengths: number[] = [0];
  let prev = evalCatmullRom(pts, 0);

  for (let i = 1; i <= SAMPLES; i++) {
    const t = i / SAMPLES;
    const cur = evalCatmullRom(pts, t);
    const dx = cur.x - prev.x;
    const dy = cur.y - prev.y;
    arcLengths.push(arcLengths[i - 1] + Math.sqrt(dx * dx + dy * dy));
    prev = cur;
  }

  const totalLength = arcLengths[SAMPLES];

  // Build myelin segments along the spline
  const myelinSegs: Array<[number, number]> = [];
  const ranvierGaps: number[] = [];

  if (style.myelinCoverage > 0 && totalLength > 0) {
    const segLen = MYELIN_SEGMENT_LEN;
    const gapLen = segLen * RANVIER_GAP_RATIO;
    const stride = segLen + gapLen;
    const numSegs = Math.max(1, Math.floor(totalLength / stride));

    // Centre the segments along the spline
    const usedLen = numSegs * stride - gapLen;
    const startOffset = (totalLength - usedLen) / (2 * totalLength);

    for (let s = 0; s < numSegs; s++) {
      const segStart = startOffset + (s * stride) / totalLength;
      const segEnd   = startOffset + (s * stride + segLen) / totalLength;
      if (segEnd <= 1.0 && segStart >= 0.0) {
        myelinSegs.push([segStart, Math.min(segEnd, 1.0)]);
        // Ranvier gap is the centre between this segment end and next start
        if (s < numSegs - 1) {
          const gapCentre = startOffset + (s * stride + segLen + gapLen * 0.5) / totalLength;
          ranvierGaps.push(gapCentre);
        }
      }
    }
  }

  return { arcLengths, totalLength, myelinSegs, ranvierGaps };
}

// ═══════════════════════════════════════════════════════════════════════════════
// ██  NeuralPathwayRenderer  ██
// ═══════════════════════════════════════════════════════════════════════════════

export class NeuralPathwayRenderer {

  // ── Config ────────────────────────────────────────────────────────────────
  readonly edges:          NeuralEdge[];
  private onVesicleArrival?: OnVesicleArrivalFn;
  private onPulseArrival?:   OnPulseArrivalFn;
  private speedScale:        number;
  private drawAxonFlag:      boolean;
  private drawMyelinFlag:    boolean;
  private axonGlowOpacity:   number;
  private demyelinFlicker:   boolean;

  // ── Canvas context ────────────────────────────────────────────────────────
  private ctx:               CanvasRenderingContext2D;

  // ── Particle pools ────────────────────────────────────────────────────────
  private pulses:            PulseParticle[]     = [];
  private vesicles:          VesicleParticle[]   = [];
  private flowParticles:     FlowParticleState[] = [];
  private burstParticles:    BurstParticle[]     = [];

  // ── Timers (per-edge spawn clocks) ────────────────────────────────────────
  private pulseTimers:       number[]            = [];
  private vesicleTimers:     number[]            = [];

  // ── Spline caches ─────────────────────────────────────────────────────────
  private splineCaches:      EdgeSplineCache[]   = [];

  // ── Elapsed tracking ──────────────────────────────────────────────────────
  private elapsed:           number              = 0;

  // ────────────────────────────────────────────────────────────────────────────
  // Construction
  // ────────────────────────────────────────────────────────────────────────────

  private constructor(
    ctx:    CanvasRenderingContext2D,
    config: NeuralPathwayConfig,
  ) {
    this.ctx              = ctx;
    this.edges            = config.edges;
    this.onVesicleArrival = config.onVesicleArrival;
    this.onPulseArrival   = config.onPulseArrival;
    this.speedScale       = config.speedScale ?? 1.0;
    this.drawAxonFlag     = config.drawAxon ?? true;
    this.drawMyelinFlag   = config.drawMyelin ?? true;
    this.axonGlowOpacity  = config.axonGlowOpacity ?? 0.15;
    this.demyelinFlicker  = config.demyelinationFlicker ?? true;

    // Build per-edge caches & timers
    for (let i = 0; i < this.edges.length; i++) {
      const edge  = this.edges[i];
      const style = styleFor(edge.qos ?? 'DEFAULT');
      this.splineCaches.push(buildSplineCache(edge.points, style));
      this.pulseTimers.push(rng(0, style.pulseInterval));
      this.vesicleTimers.push(rng(0, style.vesicleInterval));
    }

    // Seed initial flow particles
    this._seedFlowParticles();
  }

  /**
   * Factory: create a Canvas2D neural pathway renderer.
   */
  static create(
    ctx:    CanvasRenderingContext2D,
    config: NeuralPathwayConfig,
  ): NeuralPathwayRenderer {
    return new NeuralPathwayRenderer(ctx, config);
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Update (tick all subsystems)
  // ────────────────────────────────────────────────────────────────────────────

  update(elapsed: number, dt: number): void {
    this.elapsed = elapsed;
    const sdt = dt * this.speedScale;

    this._spawnPulses(sdt);
    this._tickPulses(sdt);

    this._spawnVesicles(sdt);
    this._tickVesicles(sdt);

    this._tickFlowParticles(sdt, elapsed);

    this._tickBurstParticles(sdt);
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Draw (Canvas2D composite rendering)
  // ────────────────────────────────────────────────────────────────────────────

  draw(): void {
    const ctx = this.ctx;
    ctx.save();

    // Layer 1: Axon tubes (bottom)
    if (this.drawAxonFlag)  this._drawAxonTubes();

    // Layer 2: Myelin sheath segments
    if (this.drawMyelinFlag) this._drawMyelinSheath();

    // Layer 3: Background flow particles (additive)
    ctx.globalCompositeOperation = 'lighter';
    this._drawFlowParticles();

    // Layer 4: Synaptic vesicles (normal blend, above flow)
    ctx.globalCompositeOperation = 'source-over';
    this._drawVesicles();

    // Layer 5: Electric pulses (additive glow)
    ctx.globalCompositeOperation = 'lighter';
    this._drawPulses();

    // Layer 6: Burst particles (additive)
    this._drawBurstParticles();

    ctx.restore();
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────────────────────────────────

  /** Replace the edge topology at runtime. */
  setEdges(edges: NeuralEdge[]): void {
    (this as { edges: NeuralEdge[] }).edges = edges;
    this.splineCaches.length  = 0;
    this.pulseTimers.length   = 0;
    this.vesicleTimers.length = 0;

    for (let i = 0; i < edges.length; i++) {
      const style = styleFor(edges[i].qos ?? 'DEFAULT');
      this.splineCaches.push(buildSplineCache(edges[i].points, style));
      this.pulseTimers.push(rng(0, style.pulseInterval));
      this.vesicleTimers.push(rng(0, style.vesicleInterval));
    }

    // Reset all particles
    this.pulses.length        = 0;
    this.vesicles.length      = 0;
    this.flowParticles.length = 0;
    this.burstParticles.length = 0;
    this._seedFlowParticles();
  }

  /** Adjust global speed multiplier. */
  setSpeedScale(scale: number): void {
    this.speedScale = scale;
  }

  /** Dispose all particle state. */
  dispose(): void {
    this.pulses.length        = 0;
    this.vesicles.length      = 0;
    this.flowParticles.length = 0;
    this.burstParticles.length = 0;
    this.splineCaches.length  = 0;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Electric pulse subsystem
  // ════════════════════════════════════════════════════════════════════════════

  private _spawnPulses(dt: number): void {
    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge  = this.edges[ei];
      const style = styleFor(edge.qos ?? 'DEFAULT');

      this.pulseTimers[ei] -= dt;
      if (this.pulseTimers[ei] <= 0 && this.pulses.length < MAX_PULSES) {
        this.pulseTimers[ei] = style.pulseInterval * rng(0.7, 1.3);

        const pt = evalCatmullRom(edge.points, 0);
        this.pulses.push({
          edgeIndex:   ei,
          travel:      0,
          speed:       rng(style.pulseSpeed[0], style.pulseSpeed[1]),
          phase:       PulsePhase.CONDUCT,
          alpha:       1.0,
          apTime:      0,
          lastRanvier: -1,
          x:           pt.x,
          y:           pt.y,
        });
      }
    }
  }

  private _tickPulses(dt: number): void {
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const p    = this.pulses[i];
      const edge = this.edges[p.edgeIndex];
      if (!edge) { this.pulses.splice(i, 1); continue; }

      const style = styleFor(edge.qos ?? 'DEFAULT');
      const cache = this.splineCaches[p.edgeIndex];

      if (p.phase === PulsePhase.CONDUCT) {
        p.travel += p.speed * dt;
        p.apTime += dt;

        // Check Ranvier gap crossings for saltatory flash
        if (cache) {
          for (let g = 0; g < cache.ranvierGaps.length; g++) {
            if (g !== p.lastRanvier && Math.abs(p.travel - cache.ranvierGaps[g]) < 0.02) {
              p.lastRanvier = g;
              p.apTime = 0; // reset AP waveform at each node
            }
          }
        }

        if (p.travel >= 1.0) {
          p.phase = PulsePhase.DECAY;
          p.travel = 1.0;
          const pt = evalCatmullRom(edge.points, 1.0);
          p.x = pt.x;
          p.y = pt.y;
          this.onPulseArrival?.(edge.edgeId, edge.targetId, p.x, p.y);
        } else {
          const pt = evalCatmullRom(edge.points, p.travel);
          p.x = pt.x;
          p.y = pt.y;
        }
      }

      if (p.phase === PulsePhase.DECAY) {
        p.alpha -= dt * 3.0;
        if (p.alpha <= 0) {
          p.phase = PulsePhase.DEAD;
        }
      }

      if (p.phase === PulsePhase.DEAD) {
        this.pulses.splice(i, 1);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Synaptic vesicle subsystem
  // ════════════════════════════════════════════════════════════════════════════

  private _spawnVesicles(dt: number): void {
    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge  = this.edges[ei];
      const style = styleFor(edge.qos ?? 'DEFAULT');

      this.vesicleTimers[ei] -= dt;
      if (this.vesicleTimers[ei] <= 0 && this.vesicles.length < MAX_VESICLES) {
        this.vesicleTimers[ei] = style.vesicleInterval * rng(0.8, 1.2);

        const pt = evalCatmullRom(edge.points, 0);
        this.vesicles.push({
          edgeIndex:   ei,
          travel:      0,
          speed:       rng(style.vesicleSpeed[0], style.vesicleSpeed[1]),
          phase:       VesiclePhase.TRANSIT,
          alpha:       0.7,
          wobblePhase: rng(0, Math.PI * 2),
          lateralOff:  rng(-0.4, 0.4),
          x:           pt.x,
          y:           pt.y,
        });
      }
    }
  }

  private _tickVesicles(dt: number): void {
    for (let i = this.vesicles.length - 1; i >= 0; i--) {
      const v    = this.vesicles[i];
      const edge = this.edges[v.edgeIndex];
      if (!edge) { this.vesicles.splice(i, 1); continue; }

      const style = styleFor(edge.qos ?? 'DEFAULT');

      if (v.phase === VesiclePhase.TRANSIT) {
        v.travel += v.speed * dt;
        v.wobblePhase += dt * 3.5;

        if (v.travel >= 1.0) {
          v.phase = VesiclePhase.BURST;
          v.travel = 1.0;
          const pt = evalCatmullRom(edge.points, 1.0);
          v.x = pt.x;
          v.y = pt.y;

          // Exocytosis: spawn burst micro-particles
          this._spawnBurst(v.x, v.y, v.edgeIndex);

          this.onVesicleArrival?.(edge.edgeId, edge.targetId, v.x, v.y);
        } else {
          // Organic wobble — lateral displacement from centreline
          const normal = splineNormal(edge.points, v.travel);
          const wobble = Math.sin(v.wobblePhase) * v.lateralOff * style.axonRadius;
          const pt = evalCatmullRom(edge.points, v.travel);
          v.x = pt.x + normal.x * wobble;
          v.y = pt.y + normal.y * wobble;
        }
      }

      if (v.phase === VesiclePhase.BURST) {
        v.alpha -= dt * 2.5;
        if (v.alpha <= 0) {
          v.phase = VesiclePhase.DEAD;
        }
      }

      if (v.phase === VesiclePhase.DEAD) {
        this.vesicles.splice(i, 1);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Burst (exocytosis) micro-particles
  // ════════════════════════════════════════════════════════════════════════════

  private _spawnBurst(cx: number, cy: number, edgeIndex: number): void {
    const count = Math.min(12, MAX_BURST_PARTICLES - this.burstParticles.length);
    for (let i = 0; i < count; i++) {
      const angle = rng(0, Math.PI * 2);
      const speed = rng(15, BURST_SCATTER_RADIUS * 2.5);
      this.burstParticles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: BURST_LIFETIME,
        maxLife: BURST_LIFETIME,
        radius: rng(1.0, 2.5),
        edgeIndex,
      });
    }
  }

  private _tickBurstParticles(dt: number): void {
    for (let i = this.burstParticles.length - 1; i >= 0; i--) {
      const bp = this.burstParticles[i];
      bp.x += bp.vx * dt;
      bp.y += bp.vy * dt;
      // Drag deceleration
      bp.vx *= 0.96;
      bp.vy *= 0.96;
      bp.life -= dt;
      if (bp.life <= 0) {
        this.burstParticles.splice(i, 1);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Background flow particles
  // ════════════════════════════════════════════════════════════════════════════

  private _seedFlowParticles(): void {
    this.flowParticles.length = 0;
    let total = 0;

    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge  = this.edges[ei];
      const style = styleFor(edge.qos ?? 'DEFAULT');
      const count = Math.min(
        style.flowDensity,
        Math.floor((MAX_FLOW_PARTICLES - total) / Math.max(1, this.edges.length - ei)),
      );

      for (let p = 0; p < count; p++) {
        const travel = rng(0, 1);
        const pt = evalCatmullRom(edge.points, travel);
        this.flowParticles.push({
          edgeIndex: ei,
          travel,
          speed:    rng(style.flowSpeed[0], style.flowSpeed[1]),
          phase:    FlowPhase.FLOW,
          alpha:    rng(0.3, 0.8),
          delay:    0,
          seed:     rng(0, 1000),
          curlOff:  0,
          x:        pt.x,
          y:        pt.y,
        });
        total++;
      }
    }
  }

  private _tickFlowParticles(dt: number, elapsed: number): void {
    for (let i = 0; i < this.flowParticles.length; i++) {
      const fp   = this.flowParticles[i];
      const edge = this.edges[fp.edgeIndex];
      if (!edge) continue;

      const style = styleFor(edge.qos ?? 'DEFAULT');

      if (fp.phase === FlowPhase.FLOW) {
        fp.travel += fp.speed * dt;

        if (fp.travel >= 1.0) {
          // Loop back to start with random offset
          fp.travel -= 1.0;
          fp.speed = rng(style.flowSpeed[0], style.flowSpeed[1]);
          fp.alpha = rng(0.3, 0.8);
        }

        // Curl-noise lateral perturbation
        const noiseScale = 2.0;
        const curl = curlNoise2D(
          fp.travel * noiseScale + fp.seed,
          fp.edgeIndex * 7.3,
          elapsed * 1.5,
        );
        const normal = splineNormal(edge.points, fp.travel);
        fp.curlOff += (curl.x * style.curlStrength - fp.curlOff) * 0.1;

        const pt = evalCatmullRom(edge.points, fp.travel);
        fp.x = pt.x + normal.x * fp.curlOff * style.axonRadius * 8;
        fp.y = pt.y + normal.y * fp.curlOff * style.axonRadius * 8;
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Axon tube
  // ════════════════════════════════════════════════════════════════════════════

  private _drawAxonTubes(): void {
    const ctx = this.ctx;
    const SEGMENTS = 48;

    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge  = this.edges[ei];
      const style = styleFor(edge.qos ?? 'DEFAULT');
      const theme = themeFor(edge.qos ?? 'DEFAULT');
      if (edge.points.length < 2) continue;

      const radius = style.axonRadius * Math.sqrt(edge.weight);

      // Outer glow pass
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.beginPath();
      const p0 = evalCatmullRom(edge.points, 0);
      ctx.moveTo(p0.x, p0.y);
      for (let s = 1; s <= SEGMENTS; s++) {
        const pt = evalCatmullRom(edge.points, s / SEGMENTS);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = rgba(theme.base, this.axonGlowOpacity * 0.5);
      ctx.lineWidth   = radius * 4;
      ctx.lineCap     = 'round';
      ctx.stroke();
      ctx.restore();

      // Inner tube (semi-transparent corridor)
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let s = 1; s <= SEGMENTS; s++) {
        const pt = evalCatmullRom(edge.points, s / SEGMENTS);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = rgba(theme.shadow, this.axonGlowOpacity);
      ctx.lineWidth   = radius * 2;
      ctx.lineCap     = 'round';
      ctx.stroke();

      // Centreline highlight (axoplasm luminescence)
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let s = 1; s <= SEGMENTS; s++) {
        const pt = evalCatmullRom(edge.points, s / SEGMENTS);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = rgba(theme.base, this.axonGlowOpacity * 0.3);
      ctx.lineWidth   = radius * 0.6;
      ctx.lineCap     = 'round';
      ctx.stroke();
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Myelin sheath
  // ════════════════════════════════════════════════════════════════════════════

  private _drawMyelinSheath(): void {
    const ctx = this.ctx;
    const SUB_SEGS = 8;

    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge  = this.edges[ei];
      const style = styleFor(edge.qos ?? 'DEFAULT');
      const theme = themeFor(edge.qos ?? 'DEFAULT');
      const cache = this.splineCaches[ei];
      if (!cache || edge.points.length < 2) continue;

      const radius = style.axonRadius * Math.sqrt(edge.weight);

      // Demyelination flicker for TOPO_CHANGE
      let myelinAlpha = style.myelinOpacity;
      if (this.demyelinFlicker && (edge.qos ?? 'DEFAULT') === 'TOPO_CHANGE') {
        myelinAlpha *= 0.5 + 0.5 * Math.sin(this.elapsed * 12);
      }

      // Pearlescent sheen colour — brighter than base
      const sheenColor = rgbBrighten(theme.base, 0.35);

      for (const [segStart, segEnd] of cache.myelinSegs) {
        // Draw each myelin segment as a thick rounded stroke segment
        ctx.beginPath();
        const s0 = evalCatmullRom(edge.points, segStart);
        ctx.moveTo(s0.x, s0.y);

        for (let s = 1; s <= SUB_SEGS; s++) {
          const t  = segStart + (segEnd - segStart) * (s / SUB_SEGS);
          const pt = evalCatmullRom(edge.points, t);
          ctx.lineTo(pt.x, pt.y);
        }

        ctx.strokeStyle = rgba(sheenColor, myelinAlpha);
        ctx.lineWidth   = radius * 2.8;
        ctx.lineCap     = 'round';
        ctx.stroke();

        // Inner lighter sheen line
        ctx.beginPath();
        ctx.moveTo(s0.x, s0.y);
        for (let s = 1; s <= SUB_SEGS; s++) {
          const t  = segStart + (segEnd - segStart) * (s / SUB_SEGS);
          const pt = evalCatmullRom(edge.points, t);
          ctx.lineTo(pt.x, pt.y);
        }
        ctx.strokeStyle = rgba({ r: 1, g: 1, b: 1 }, myelinAlpha * 0.15);
        ctx.lineWidth   = radius * 1.2;
        ctx.lineCap     = 'round';
        ctx.stroke();
      }

      // Draw Ranvier node markers (small bright gaps)
      for (const gap of cache.ranvierGaps) {
        const pt = evalCatmullRom(edge.points, gap);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, radius * 0.8, 0, Math.PI * 2);
        ctx.fillStyle = rgba(theme.highlight, 0.12);
        ctx.fill();
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Electric pulses
  // ════════════════════════════════════════════════════════════════════════════

  private _drawPulses(): void {
    const ctx = this.ctx;

    for (const p of this.pulses) {
      if (p.phase === PulsePhase.DEAD) continue;

      const edge  = this.edges[p.edgeIndex];
      if (!edge) continue;
      const style = styleFor(edge.qos ?? 'DEFAULT');
      const theme = themeFor(edge.qos ?? 'DEFAULT');

      // Action-potential waveform → brightness modulation
      const apPhase = Math.min(1, p.apTime / AP_DURATION);
      const apWave  = actionPotentialWave(apPhase);
      const brightness = Math.max(0, apWave) * style.pulseBrightness;

      const pulseColor = rgbBrighten(theme.highlight, brightness * 0.6);
      const glowAlpha  = p.alpha * (0.3 + brightness * 0.7);

      // Outer glow
      const gr = style.pulseRadius * 3;
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr);
      grad.addColorStop(0, rgba(pulseColor, glowAlpha));
      grad.addColorStop(0.4, rgba(theme.highlight, glowAlpha * 0.4));
      grad.addColorStop(1, rgba(theme.highlight, 0));

      ctx.beginPath();
      ctx.arc(p.x, p.y, gr, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Core bright dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, style.pulseRadius * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = rgba({ r: 1, g: 1, b: 1 }, p.alpha * brightness);
      ctx.fill();

      // Ranvier flash — bright ring at recent gap crossing
      if (apPhase < 0.15) {
        const cache = this.splineCaches[p.edgeIndex];
        if (cache && p.lastRanvier >= 0 && p.lastRanvier < cache.ranvierGaps.length) {
          const gapT = cache.ranvierGaps[p.lastRanvier];
          const gpt = evalCatmullRom(edge.points, gapT);
          const flashAlpha = (1 - apPhase / 0.15) * style.ranvierFlash;
          ctx.beginPath();
          ctx.arc(gpt.x, gpt.y, style.pulseRadius * 2.5, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(theme.highlight, flashAlpha * 0.6);
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Synaptic vesicles
  // ════════════════════════════════════════════════════════════════════════════

  private _drawVesicles(): void {
    const ctx = this.ctx;

    for (const v of this.vesicles) {
      if (v.phase === VesiclePhase.DEAD) continue;

      const edge  = this.edges[v.edgeIndex];
      if (!edge) continue;
      const style = styleFor(edge.qos ?? 'DEFAULT');
      const theme = themeFor(edge.qos ?? 'DEFAULT');

      const r = style.vesicleRadius;
      const a = v.alpha;

      // Translucent vesicle body — radial gradient for spheroid illusion
      const grad = ctx.createRadialGradient(
        v.x - r * 0.2, v.y - r * 0.25, r * 0.1,
        v.x, v.y, r,
      );
      grad.addColorStop(0, rgba(rgbBrighten(theme.base, 0.4), a * 0.8));
      grad.addColorStop(0.5, rgba(theme.base, a * 0.45));
      grad.addColorStop(1, rgba(theme.shadow, a * 0.1));

      ctx.beginPath();
      ctx.arc(v.x, v.y, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Membrane ring
      ctx.beginPath();
      ctx.arc(v.x, v.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = rgba(theme.base, a * 0.3);
      ctx.lineWidth = 0.8;
      ctx.stroke();

      // Inner "neurotransmitter" dots (2-3 tiny circles)
      const dotCount = 2 + ((v.edgeIndex * 3 + Math.floor(v.wobblePhase)) % 2);
      for (let d = 0; d < dotCount; d++) {
        const angle = v.wobblePhase * 0.4 + (d / dotCount) * Math.PI * 2;
        const dist  = r * 0.35;
        const dx = v.x + Math.cos(angle) * dist;
        const dy = v.y + Math.sin(angle) * dist;
        ctx.beginPath();
        ctx.arc(dx, dy, 1.2, 0, Math.PI * 2);
        ctx.fillStyle = rgba(theme.highlight, a * 0.6);
        ctx.fill();
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Flow particles
  // ════════════════════════════════════════════════════════════════════════════

  private _drawFlowParticles(): void {
    const ctx = this.ctx;

    for (const fp of this.flowParticles) {
      if (fp.phase === FlowPhase.DEAD) continue;

      const edge = this.edges[fp.edgeIndex];
      if (!edge) continue;
      const theme = themeFor(edge.qos ?? 'DEFAULT');

      const r = 1.2;
      ctx.beginPath();
      ctx.arc(fp.x, fp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(theme.base, fp.alpha * 0.35);
      ctx.fill();
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Private: Drawing — Burst (exocytosis) micro-particles
  // ════════════════════════════════════════════════════════════════════════════

  private _drawBurstParticles(): void {
    const ctx = this.ctx;

    for (const bp of this.burstParticles) {
      const edge = this.edges[bp.edgeIndex];
      if (!edge) continue;
      const theme = themeFor(edge.qos ?? 'DEFAULT');

      const lifeRatio = Math.max(0, bp.life / bp.maxLife);
      const a = lifeRatio * 0.8;
      const r = bp.radius * (1.0 + (1.0 - lifeRatio) * 0.5);

      ctx.beginPath();
      ctx.arc(bp.x, bp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(theme.highlight, a);
      ctx.fill();
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Factory helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a NeuralPathwayRenderer with QoS profiles resolved from a map.
 *
 * @example
 * ```ts
 * const neural = createNeuralPathwayRenderer(ctx, rawEdges, {
 *   'attn→ffn':      'SENSOR_DATA',
 *   'ffn→layernorm': 'PARAMETERS',
 * });
 * ```
 */
export function createNeuralPathwayRenderer(
  ctx:    CanvasRenderingContext2D,
  edges:  Omit<NeuralEdge, 'qos'>[],
  qosMap: Record<string, QoSProfileName>,
  config: Omit<NeuralPathwayConfig, 'edges'> = {},
): NeuralPathwayRenderer {
  const resolved: NeuralEdge[] = edges.map(e => ({
    ...e,
    qos: qosMap[e.edgeId] ?? 'DEFAULT',
  }));
  return NeuralPathwayRenderer.create(ctx, { ...config, edges: resolved });
}

/**
 * Create a NeuralPathwayRenderer wired to SPH world's `addFluid` for
 * vesicle exocytosis → SPH injection coupling.
 */
export function createNeuralPathwayForSPH(
  ctx:      CanvasRenderingContext2D,
  edges:    NeuralEdge[],
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<NeuralPathwayConfig, 'edges' | 'onVesicleArrival'> = {},
): NeuralPathwayRenderer {
  const R = 0.05;
  return NeuralPathwayRenderer.create(ctx, {
    ...config,
    edges,
    onVesicleArrival: (_edgeId, _targetId, x, y) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, 0);
    },
  });
}

// ─── Re-exports ──────────────────────────────────────────────────────────────

export { evalCatmullRom, splineTangent, splineNormal };

// ─── Defaults re-export ──────────────────────────────────────────────────────

export const NEURAL_PATHWAY_DEFAULTS = {
  maxPulses:         MAX_PULSES,
  maxVesicles:       MAX_VESICLES,
  maxFlowParticles:  MAX_FLOW_PARTICLES,
  maxBurstParticles: MAX_BURST_PARTICLES,
  crTension:         CR_TENSION,
  myelinSegmentLen:  MYELIN_SEGMENT_LEN,
  ranvierGapRatio:   RANVIER_GAP_RATIO,
  apDuration:        AP_DURATION,
  burstScatterRadius: BURST_SCATTER_RADIUS,
  burstLifetime:     BURST_LIFETIME,
  neuralStyles:      NEURAL_STYLES,
} as const;
