/**
 * edge-data-flow-viz.ts — M766
 *
 * Edge Glow-Pulse Visualiser: animated luminous pulses that travel along
 * topology edge splines, conveying data-flow intensity and direction.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Unlike the particle-based EdgeFlowRenderer (M742), this module renders the
 * edges themselves as glowing ribbons with scrolling pulse wavefronts.  Each
 * pulse is a Gaussian-shaped brightness peak that travels source → target,
 * leaving a soft afterglow trail.  The combination produces a "fibre-optic"
 * or "neural signal" aesthetic that maps directly to the cell pub/sub data
 * flow — every message burst triggers a visible pulse.
 *
 * Visual language:
 *
 *   SENSOR_DATA  — rapid narrow pulses, cool-blue glow, high cadence
 *   PARAMETERS   — broad warm-amber sweeps, moderate cadence
 *   TF_STATIC    — slow jade-green waves, wide and persistent
 *   TOPO_CHANGE  — sharp magenta flashes, staccato rhythm
 *   DEFAULT      — balanced slate pulses
 *
 * Pulse anatomy (per-edge):
 *   ┌─ base glow: faint static luminance along full spline (ambient)
 *   ├─ pulse head: Gaussian peak (σ = pulseWidth) scrolling at pulseSpeed
 *   ├─ afterglow trail: exponential decay behind the head
 *   └─ arrival bloom: brief radial flash at target endpoint on arrival
 *
 * Rendering modes:
 *
 *   1. **Canvas2D** — composites with `globalCompositeOperation: 'lighter'`
 *      for additive blending.  Draws each edge as a series of line segments
 *      with per-segment alpha modulated by the pulse envelope.
 *
 *   2. **WebGL2 overlay** — emits per-edge uniform data (pulse travel, glow
 *      colour, width) consumed by the existing edge-spline.frag shader's
 *      `u_flowSpeed` / `u_time` uniforms.  The fragment shader's built-in
 *      `pulseIntensity` smoothstep already supports this — we drive it with
 *      richer multi-pulse data.
 *
 * Pub/Sub integration:
 *   Call `firePulse(edgeId)` whenever a message is published on that edge's
 *   topic.  The visualiser queues a new pulse wavefront that begins at t=0
 *   (source) and scrolls to t=1 (target) over `travelDuration` seconds.
 *   Multiple pulses can coexist on one edge — they composite additively.
 *
 * Integration:
 *   ```ts
 *   import { EdgeDataFlowViz } from '$lib/sph/edge-data-flow-viz';
 *
 *   const viz = new EdgeDataFlowViz(ctx2d, topology.edges);
 *   // pub/sub hook:
 *   eventBus.on('edge:message', (edgeId) => viz.firePulse(edgeId));
 *   // render loop:
 *   viz.update(dt, elapsed);
 *   viz.draw();
 *   // cleanup:
 *   viz.destroy();
 *   ```
 *
 * References:
 *   src/lib/sph/edge-flow-renderer.ts   — particle-based edge flow (M742)
 *   src/lib/sph/color-palette.ts        — QoS → colour theme mapping
 *   src/lib/sph/qosSpatial.ts           — QoS profile definitions
 *   src/lib/shaders/edge-spline.frag    — GPU edge shader with flow pulse
 *   src/lib/EdgeRenderer.ts             — PixiJS Bézier edge rendering
 *   channels/physics/edge_routes.json   — topology edge route data
 */




// ─── Constants ────────────────────────────────────────────────────────────────

/** Maximum concurrent pulses per edge (ring buffer). */



import type { QoSProfileName }  from './qosSpatial';
import { QOS_THEME }             from './color-palette';
import type { ThemePalette, RGB } from './color-palette';

const MAX_PULSES_PER_EDGE = 8;

/** Spline evaluation subdivision count for Canvas2D drawing. */
const SPLINE_SEGMENTS = 48;

/** Base glow alpha (ambient luminance along the full edge). */
const BASE_GLOW_ALPHA = 0.06;

/** Base glow line width in domain units. */
const BASE_GLOW_WIDTH = 4.0;

/** Pulse peak line width multiplier over base. */
const PULSE_WIDTH_BOOST = 2.5;

/** Arrival bloom radius (domain units). */
const BLOOM_RADIUS = 12.0;

/** Arrival bloom duration (seconds). */
const BLOOM_DURATION = 0.25;

/** Catmull-Rom tension (0.5 = centripetal). */
const CR_TENSION = 0.5;

// ─── Per-QoS pulse style ──────────────────────────────────────────────────────

export interface PulseStyle {
  /** Pulse travel speed (0→1 per second along spline normalised length). */
  pulseSpeed:      number;
  /** Gaussian width σ of the pulse head (normalised spline units). */
  pulseWidth:      number;
  /** Peak brightness multiplier at pulse head centre [0, 1]. */
  peakBrightness:  number;
  /** Afterglow trail length behind the pulse head (normalised units). */
  trailLength:     number;
  /** Afterglow exponential decay rate (higher = faster fade). */
  trailDecay:      number;
  /** Auto-fire cadence: seconds between automatic ambient pulses (0 = off). */
  autoCadence:     number;
  /** Line width multiplier for this QoS style. */
  lineWidthScale:  number;
}

const PULSE_STYLES: Record<QoSProfileName, PulseStyle> = {
  SENSOR_DATA: {
    pulseSpeed:     1.2,
    pulseWidth:     0.06,
    peakBrightness: 0.95,
    trailLength:    0.10,
    trailDecay:     8.0,
    autoCadence:    0.15,
    lineWidthScale: 0.9,
  },
  PARAMETERS: {
    pulseSpeed:     0.55,
    pulseWidth:     0.12,
    peakBrightness: 0.80,
    trailLength:    0.18,
    trailDecay:     4.0,
    autoCadence:    0.6,
    lineWidthScale: 1.1,
  },
  TF_STATIC: {
    pulseSpeed:     0.25,
    pulseWidth:     0.20,
    peakBrightness: 0.65,
    trailLength:    0.30,
    trailDecay:     2.5,
    autoCadence:    1.5,
    lineWidthScale: 1.3,
  },
  TOPO_CHANGE: {
    pulseSpeed:     1.6,
    pulseWidth:     0.04,
    peakBrightness: 1.00,
    trailLength:    0.06,
    trailDecay:     12.0,
    autoCadence:    0.0,
    lineWidthScale: 0.8,
  },
  DEFAULT: {
    pulseSpeed:     0.65,
    pulseWidth:     0.10,
    peakBrightness: 0.75,
    trailLength:    0.15,
    trailDecay:     5.0,
    autoCadence:    0.8,
    lineWidthScale: 1.0,
  },
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D control point on an edge spline. */
export interface VizPoint {
  x: number;
  y: number;
}

/** One topology edge for the glow-pulse visualiser. */
export interface VizEdge {
  edgeId:    string;
  sourceId:  string;
  targetId:  string;
  /** Catmull-Rom control points (≥ 2, domain space). */
  points:    VizPoint[];
  /** QoS profile name; defaults to 'DEFAULT'. */
  qos?:      QoSProfileName;
  /** Is this a skip-connection? Affects glow intensity. */
  isSkip?:   boolean;
}

/** Configuration for EdgeDataFlowViz. */
export interface EdgeDataFlowVizConfig {
  /** Edge definitions. */
  edges:             VizEdge[];
  /** Global brightness multiplier [0, 1]. Default 1.0. */
  brightness?:       number;
  /** Enable ambient auto-pulse cadence. Default true. */
  enableAutoPulse?:  boolean;
  /** Enable arrival bloom flash. Default true. */
  enableBloom?:      boolean;
  /** Callback fired when a pulse arrives at the target endpoint. */
  onPulseArrival?:   (edgeId: string, targetId: string, x: number, y: number) => void;
}

// ─── Internal types ───────────────────────────────────────────────────────────

/** A single pulse wavefront travelling along an edge. */
interface Pulse {
  /** Current normalised travel position [0, 1+]. */
  travel:     number;
  /** Is this pulse alive? Dead pulses are recycled. */
  alive:      boolean;
  /** Birth time (elapsed seconds). */
  birthTime:  number;
  /** Per-pulse brightness jitter [0.8, 1.2]. */
  intensity:  number;
}

/** An arrival bloom flash. */
interface Bloom {
  /** Domain-space position (target endpoint). */
  x:         number;
  y:         number;
  /** Remaining lifetime [0, BLOOM_DURATION]. */
  life:      number;
  /** Bloom colour (from QoS theme). */
  color:     RGB;
}

/** Internal edge state. */
interface EdgeState {
  edge:         VizEdge;
  style:        PulseStyle;
  theme:        ThemePalette;
  /** Ring buffer of active pulses. */
  pulses:       Pulse[];
  /** Next write index into the ring buffer. */
  ringHead:     number;
  /** Arc-length LUT for uniform parameterisation (cumulative). */
  arcLUT:       Float32Array;
  /** Total arc length in domain units. */
  arcLength:    number;
  /** Time accumulator for auto-cadence. */
  autoTimer:    number;
  /** Pre-computed spline sample points for drawing. */
  samples:      VizPoint[];
  /** Pre-computed tangent directions at each sample. */
  tangents:     VizPoint[];
}

// ─── Catmull-Rom evaluation ───────────────────────────────────────────────────

/**
 * Evaluate a Catmull-Rom spline at normalised parameter u ∈ [0, 1].
 * Endpoint-clamped: first and last control points are duplicated.
 */
function evalCatmullRom(pts: VizPoint[], u: number): VizPoint {
  const n = pts.length;
  if (n < 2) return pts[0] ?? { x: 0, y: 0 };
  if (n === 2) {
    return { x: pts[0].x + (pts[1].x - pts[0].x) * u,
             y: pts[0].y + (pts[1].y - pts[0].y) * u };
  }

  const segCount = n - 1;
  const scaled   = u * segCount;
  const seg      = Math.min(Math.floor(scaled), segCount - 1);
  const t        = scaled - seg;

  // Clamp-extended indices
  const i0 = Math.max(seg - 1, 0);
  const i1 = seg;
  const i2 = Math.min(seg + 1, n - 1);
  const i3 = Math.min(seg + 2, n - 1);

  const p0 = pts[i0], p1 = pts[i1], p2 = pts[i2], p3 = pts[i3];

  const tt  = t * t;
  const ttt = tt * t;
  const tau = CR_TENSION;

  const h00 =  2 * ttt - 3 * tt + 1;
  const h10 =      ttt - 2 * tt + t;
  const h01 = -2 * ttt + 3 * tt;
  const h11 =      ttt -     tt;

  const tx0 = tau * (p2.x - p0.x);
  const ty0 = tau * (p2.y - p0.y);
  const tx1 = tau * (p3.x - p1.x);
  const ty1 = tau * (p3.y - p1.y);

  return {
    x: h00 * p1.x + h10 * tx0 + h01 * p2.x + h11 * tx1,
    y: h00 * p1.y + h10 * ty0 + h01 * p2.y + h11 * ty1,
  };
}

/** Evaluate spline tangent (unnormalised) at parameter u. */
function splineTangent(pts: VizPoint[], u: number): VizPoint {
  const EPS = 0.001;
  const a = evalCatmullRom(pts, Math.max(u - EPS, 0));
  const b = evalCatmullRom(pts, Math.min(u + EPS, 1));
  return { x: b.x - a.x, y: b.y - a.y };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** CSS rgba string from linear RGB + alpha. */
function toCss(r: number, g: number, b: number, a: number): string {
  return `rgba(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0},${a.toFixed(3)})`;
}

/** Build arc-length LUT for uniform parameterisation. */
function buildArcLUT(pts: VizPoint[], segments: number): { lut: Float32Array; total: number } {
  const lut = new Float32Array(segments + 1);
  let total = 0;
  let prev  = evalCatmullRom(pts, 0);

  for (let i = 1; i <= segments; i++) {
    const u   = i / segments;
    const cur = evalCatmullRom(pts, u);
    const dx  = cur.x - prev.x;
    const dy  = cur.y - prev.y;
    total    += Math.sqrt(dx * dx + dy * dy);
    lut[i]    = total;
    prev      = cur;
  }

  return { lut, total };
}

/** Map arc-length fraction [0,1] to spline parameter u using the LUT. */
function arcToParam(lut: Float32Array, total: number, frac: number): number {
  const target = frac * total;
  const n      = lut.length - 1;

  // Binary search
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (lut[mid] < target) lo = mid + 1;
    else                   hi = mid;
  }

  if (lo === 0) return 0;
  const segLen = lut[lo] - lut[lo - 1];
  const t = segLen > 0 ? (target - lut[lo - 1]) / segLen : 0;
  return (lo - 1 + t) / n;
}

/** Gaussian function centred at `center` with standard deviation `sigma`. */
function gaussian(x: number, center: number, sigma: number): number {
  const d = x - center;
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

/** Compute the glow intensity at normalised position `s` along the edge,
 *  given all active pulses for that edge. */
function computeGlowAt(s: number, pulses: Pulse[], style: PulseStyle): number {
  let intensity = 0;

  for (let i = 0; i < pulses.length; i++) {
    const p = pulses[i];
    if (!p.alive) continue;

    const head = p.travel;

    // Pulse head: Gaussian peak
    const headGlow = gaussian(s, head, style.pulseWidth) * style.peakBrightness * p.intensity;

    // Afterglow trail: exponential decay behind the head
    let trailGlow = 0;
    if (s < head && (head - s) <= style.trailLength) {
      const trailDist = head - s;
      trailGlow = Math.exp(-trailDist * style.trailDecay)
                * style.peakBrightness * 0.4 * p.intensity;
    }

    intensity += headGlow + trailGlow;
  }

  return Math.min(intensity, 1.0);
}

// ─── Pre-compute sample cache ─────────────────────────────────────────────────

function precomputeSamples(pts: VizPoint[], segments: number): { samples: VizPoint[]; tangents: VizPoint[] } {
  const samples:  VizPoint[] = new Array(segments + 1);
  const tangents: VizPoint[] = new Array(segments + 1);

  for (let i = 0; i <= segments; i++) {
    const u   = i / segments;
    samples[i]  = evalCatmullRom(pts, u);
    const tan   = splineTangent(pts, u);
    const len   = Math.sqrt(tan.x * tan.x + tan.y * tan.y) || 1;
    tangents[i] = { x: tan.x / len, y: tan.y / len };
  }

  return { samples, tangents };
}

// ─── EdgeDataFlowViz ──────────────────────────────────────────────────────────

export class EdgeDataFlowViz {
  private ctx:        CanvasRenderingContext2D;
  private edgeStates: EdgeState[] = [];
  private edgeMap:    Map<string, EdgeState> = new Map();
  private blooms:     Bloom[] = [];
  private brightness: number;
  private autoPulse:  boolean;
  private enableBloom: boolean;
  private onArrival?: (edgeId: string, targetId: string, x: number, y: number) => void;

  // ── Constructor ───────────────────────────────────────────────────────────

  constructor(ctx: CanvasRenderingContext2D, config: EdgeDataFlowVizConfig) {
    this.ctx         = ctx;
    this.brightness  = config.brightness ?? 1.0;
    this.autoPulse   = config.enableAutoPulse ?? true;
    this.enableBloom = config.enableBloom ?? true;
    this.onArrival   = config.onPulseArrival;

    for (const edge of config.edges) {
      if (edge.points.length < 2) continue;

      const qosName = edge.qos ?? 'DEFAULT';
      const style   = PULSE_STYLES[qosName];
      const theme   = QOS_THEME[qosName];

      const { lut, total }       = buildArcLUT(edge.points, SPLINE_SEGMENTS);
      const { samples, tangents } = precomputeSamples(edge.points, SPLINE_SEGMENTS);

      // Initialise pulse ring buffer
      const pulses: Pulse[] = [];
      for (let i = 0; i < MAX_PULSES_PER_EDGE; i++) {
        pulses.push({ travel: 0, alive: false, birthTime: 0, intensity: 1 });
      }

      const state: EdgeState = {
        edge,
        style,
        theme,
        pulses,
        ringHead:  0,
        arcLUT:    lut,
        arcLength: total,
        autoTimer: Math.random() * (style.autoCadence || 1),
        samples,
        tangents,
      };

      this.edgeStates.push(state);
      this.edgeMap.set(edge.edgeId, state);
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Fire a pulse on the specified edge.  Call this when a pub/sub message
   * is sent along the edge.  Multiple pulses on the same edge composite
   * additively.
   */
  firePulse(edgeId: string, intensityOverride?: number): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;

    const slot = state.pulses[state.ringHead];
    slot.travel    = 0;
    slot.alive     = true;
    slot.birthTime = 0; // will be set on next update
    slot.intensity = intensityOverride ?? (0.85 + Math.random() * 0.30);

    state.ringHead = (state.ringHead + 1) % MAX_PULSES_PER_EDGE;
  }

  /**
   * Fire pulses on all edges simultaneously (e.g. epoch boundary flash).
   */
  fireAll(intensityOverride?: number): void {
    for (const state of this.edgeStates) {
      this.firePulse(state.edge.edgeId, intensityOverride);
    }
  }

  /**
   * Update pulse positions and lifecycle.  Call once per frame.
   *
   * @param dt      - Delta time in seconds.
   * @param elapsed - Total elapsed time in seconds (used for auto-cadence phase).
   */
  update(dt: number, elapsed: number): void {
    // Clamp dt to avoid spiral-of-death on tab-switch
    const cdt = Math.min(dt, 0.1);

    for (const state of this.edgeStates) {
      const { style, edge } = state;

      // ── Auto-pulse cadence ──────────────────────────────────────────────
      if (this.autoPulse && style.autoCadence > 0) {
        state.autoTimer -= cdt;
        if (state.autoTimer <= 0) {
          state.autoTimer += style.autoCadence;
          this.firePulse(edge.edgeId, 0.5 + Math.random() * 0.3);
        }
      }

      // ── Advance pulses ──────────────────────────────────────────────────
      for (const pulse of state.pulses) {
        if (!pulse.alive) continue;

        pulse.travel += style.pulseSpeed * cdt;

        // Check arrival: pulse head passed the end of the spline
        if (pulse.travel >= 1.0 + style.trailLength + style.pulseWidth * 3) {
          pulse.alive = false;

          // Trigger arrival bloom
          if (this.enableBloom) {
            const endPt = edge.points[edge.points.length - 1];
            this.blooms.push({
              x:     endPt.x,
              y:     endPt.y,
              life:  BLOOM_DURATION,
              color: state.theme.base,
            });
          }

          // Callback
          if (this.onArrival) {
            const endPt = edge.points[edge.points.length - 1];
            this.onArrival(edge.edgeId, edge.targetId, endPt.x, endPt.y);
          }
        }
      }
    }

    // ── Update blooms ───────────────────────────────────────────────────────
    for (let i = this.blooms.length - 1; i >= 0; i--) {
      this.blooms[i].life -= cdt;
      if (this.blooms[i].life <= 0) {
        this.blooms.splice(i, 1);
      }
    }
  }

  /**
   * Draw all edge glow-pulses onto the Canvas2D context.
   * Should be called after `update()` each frame.
   */
  draw(): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // ── Draw edge glow-pulses ─────────────────────────────────────────────
    for (const state of this.edgeStates) {
      this._drawEdgeGlow(state);
    }

    // ── Draw arrival blooms ───────────────────────────────────────────────
    for (const bloom of this.blooms) {
      this._drawBloom(bloom);
    }

    ctx.restore();
  }

  /**
   * Set global brightness multiplier.
   */
  setBrightness(value: number): void {
    this.brightness = Math.max(0, Math.min(1, value));
  }

  /**
   * Toggle auto-pulse cadence on/off.
   */
  setAutoPulse(enabled: boolean): void {
    this.autoPulse = enabled;
  }

  /**
   * Update QoS profile for a specific edge (e.g. after topology change).
   */
  setEdgeQoS(edgeId: string, qos: QoSProfileName): void {
    const state = this.edgeMap.get(edgeId);
    if (!state) return;

    state.style = PULSE_STYLES[qos];
    state.theme = QOS_THEME[qos];
    state.edge.qos = qos;
  }

  /**
   * Get the current pulse state for an edge (read-only diagnostic).
   */
  getPulseState(edgeId: string): { activePulses: number; arcLength: number } | null {
    const state = this.edgeMap.get(edgeId);
    if (!state) return null;

    let active = 0;
    for (const p of state.pulses) {
      if (p.alive) active++;
    }
    return { activePulses: active, arcLength: state.arcLength };
  }

  /**
   * Provides per-edge glow data for the WebGL2 edge-spline.frag shader.
   * Returns a map of edgeId → { flowSpeed, glowAlpha, glowColor }.
   * The caller writes these into the shader uniforms each frame.
   */
  getShaderUniforms(): Map<string, { flowSpeed: number; glowAlpha: number; glowColor: [number, number, number] }> {
    const out = new Map<string, { flowSpeed: number; glowAlpha: number; glowColor: [number, number, number] }>();

    for (const state of this.edgeStates) {
      // Compute aggregate pulse intensity for the shader
      let maxIntensity = 0;
      let leadTravel   = 0;
      for (const p of state.pulses) {
        if (p.alive && p.intensity > maxIntensity) {
          maxIntensity = p.intensity;
          leadTravel   = p.travel;
        }
      }

      const { base } = state.theme;
      out.set(state.edge.edgeId, {
        flowSpeed: state.style.pulseSpeed,
        glowAlpha: maxIntensity * state.style.peakBrightness * this.brightness,
        glowColor: [base.r, base.g, base.b],
      });
    }

    return out;
  }

  /**
   * Dispose all internal state.
   */
  destroy(): void {
    this.edgeStates.length = 0;
    this.edgeMap.clear();
    this.blooms.length = 0;
  }

  // ── Private: draw a single edge's glow-pulse ──────────────────────────────

  private _drawEdgeGlow(state: EdgeState): void {
    const ctx           = this.ctx;
    const { style, theme, edge, samples, tangents, arcLUT, arcLength } = state;
    const baseColor     = theme.base;
    const highlightColor = theme.highlight;
    const isSkip        = edge.isSkip ?? false;
    const lineW         = BASE_GLOW_WIDTH * style.lineWidthScale * (isSkip ? 1.2 : 1.0);

    for (let i = 0; i < SPLINE_SEGMENTS; i++) {
      const s0 = samples[i];
      const s1 = samples[i + 1];

      // Normalised arc-length fraction at segment midpoint
      const midArcFrac = (arcLUT[i] + arcLUT[i + 1]) / (2 * arcLength);

      // Compute glow intensity from all active pulses
      const pulseI = computeGlowAt(midArcFrac, state.pulses, style);

      // Total alpha: base ambient + pulse contribution
      const totalAlpha = (BASE_GLOW_ALPHA + pulseI * 0.85) * this.brightness;

      if (totalAlpha < 0.003) continue;

      // Colour: blend base → highlight at high pulse intensity
      const blendT = pulseI * pulseI; // non-linear: bright highlight only at peaks
      const cr = baseColor.r + (highlightColor.r - baseColor.r) * blendT;
      const cg = baseColor.g + (highlightColor.g - baseColor.g) * blendT;
      const cb = baseColor.b + (highlightColor.b - baseColor.b) * blendT;

      // Line width: fatten at pulse peaks
      const segWidth = lineW * (1.0 + pulseI * PULSE_WIDTH_BOOST);

      // Draw segment
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.strokeStyle = toCss(cr, cg, cb, totalAlpha);
      ctx.lineWidth   = segWidth;
      ctx.lineCap     = 'round';
      ctx.stroke();

      // ── Outer glow halo (wider, dimmer) ──────────────────────────────────
      if (pulseI > 0.15) {
        const haloAlpha = pulseI * 0.25 * this.brightness;
        const haloWidth = segWidth * 2.5;

        ctx.beginPath();
        ctx.moveTo(s0.x, s0.y);
        ctx.lineTo(s1.x, s1.y);
        ctx.strokeStyle = toCss(cr, cg, cb, haloAlpha);
        ctx.lineWidth   = haloWidth;
        ctx.lineCap     = 'round';
        ctx.stroke();
      }
    }
  }

  // ── Private: draw arrival bloom ────────────────────────────────────────────

  private _drawBloom(bloom: Bloom): void {
    const ctx = this.ctx;
    const t   = 1.0 - (bloom.life / BLOOM_DURATION); // 0 → 1 over lifetime

    // Expand radius, fade alpha
    const radius = BLOOM_RADIUS * (0.3 + t * 0.7);
    const alpha  = (1.0 - t * t) * 0.6 * this.brightness;

    if (alpha < 0.003) return;

    const { r, g, b } = bloom.color;

    // Core flash
    const gradient = ctx.createRadialGradient(
      bloom.x, bloom.y, 0,
      bloom.x, bloom.y, radius,
    );
    gradient.addColorStop(0, toCss(r, g, b, alpha));
    gradient.addColorStop(0.4, toCss(r, g, b, alpha * 0.5));
    gradient.addColorStop(1, toCss(r, g, b, 0));

    ctx.beginPath();
    ctx.arc(bloom.x, bloom.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();
  }
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create an EdgeDataFlowViz from raw edge routes (e.g. from edge_routes.json).
 *
 * @example
 * ```ts
 * import routes from '../../channels/physics/edge_routes.json';
 * const viz = createEdgeDataFlowViz(ctx, routes, {
 *   'e1': 'SENSOR_DATA',
 *   'skip1': 'TOPO_CHANGE',
 * });
 * ```
 */
export function createEdgeDataFlowViz(
  ctx:     CanvasRenderingContext2D,
  routes:  Record<string, {
    edge_id: string;
    sources: string[];
    targets: string[];
    is_skip: boolean;
    points:  VizPoint[];
  }>,
  qosMap:  Record<string, QoSProfileName> = {},
  config:  Omit<EdgeDataFlowVizConfig, 'edges'> = {},
): EdgeDataFlowViz {
  const edges: VizEdge[] = Object.values(routes).map(r => ({
    edgeId:   r.edge_id,
    sourceId: r.sources[0] ?? '',
    targetId: r.targets[0] ?? '',
    points:   r.points,
    qos:      qosMap[r.edge_id] ?? 'DEFAULT',
    isSkip:   r.is_skip,
  }));

  return new EdgeDataFlowViz(ctx, { ...config, edges });
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

export { evalCatmullRom, splineTangent };

export const EDGE_DATA_FLOW_DEFAULTS = {
  maxPulsesPerEdge:  MAX_PULSES_PER_EDGE,
  splineSegments:    SPLINE_SEGMENTS,
  baseGlowAlpha:     BASE_GLOW_ALPHA,
  baseGlowWidth:     BASE_GLOW_WIDTH,
  pulseWidthBoost:   PULSE_WIDTH_BOOST,
  bloomRadius:       BLOOM_RADIUS,
  bloomDuration:     BLOOM_DURATION,
  pulseStyles:       PULSE_STYLES,
} as const;
