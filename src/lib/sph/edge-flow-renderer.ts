/**
 * edge-flow-renderer.ts — M742
 *
 * Edge-Flow Renderer: particles flowing along Catmull-Rom splines with
 * QoS-driven speed and colour.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * This module provides a lightweight **CPU-side** edge-flow particle system
 * designed for Canvas2D rendering.  It complements the heavy WebGPU pipeline
 * in `at-spline-particle.ts` by offering a zero-GPU fallback suitable for
 * overlays, debug views, and low-power devices.
 *
 * Core idea:
 *   Each topology edge owns a set of **flow particles** that travel from
 *   source → target along the edge's Catmull-Rom spline.  The particle's
 *   **speed** and **colour** are determined by the edge's QoS profile:
 *
 *     SENSOR_DATA  — fast flow, cool-blue trail    (high-frequency bursts)
 *     PARAMETERS   — moderate flow, warm-amber      (stable param streams)
 *     TF_STATIC    — slow flow, jade-green          (persistent transforms)
 *     TOPO_CHANGE  — fast pulse, magenta            (disruptive events)
 *     DEFAULT      — balanced flow, neutral slate
 *
 * Particle lifecycle mirrors AT's SplineParticleLife (see at-spline-particle.ts):
 *   SPAWN  → wait out random delay
 *   FLOW   → advance along spline (Catmull-Rom interpolation)
 *   DECAY  → alpha fades after reaching spline end
 *   DEAD   → slot recycled (respawn)
 *
 * Integration:
 *   ```ts
 *   import { EdgeFlowRenderer } from '$lib/sph/edge-flow-renderer';
 *
 *   const renderer = new EdgeFlowRenderer(ctx, {
 *     edges: topology.edges,
 *     onArrival: (edgeId, targetId, x, y) => {
 *       sphWorld.addFluid(x - 0.05, y - 0.05, x + 0.05, y + 0.05, 0.04, 0);
 *     },
 *   });
 *
 *   // render loop:
 *   renderer.update(dt, elapsed);
 *   renderer.draw();
 *   ```
 *
 * References:
 *   src/lib/sph/at-spline-particle.ts   — WebGPU particle pipeline (GPU twin)
 *   src/lib/sph/spline-particle-life.ts — CPU SplineParticleLife reference
 *   src/lib/sph/color-palette.ts        — QoS → colour theme mapping
 *   src/lib/sph/qosSpatial.ts           — QoS profile definitions
 */

import type { QoSProfile }      from './types';
import type { QoSProfileName }  from './qosSpatial';
import { QOS_PRESETS }           from './qosSpatial';
import { QOS_THEME }             from './color-palette';
import type { ThemePalette }     from './color-palette';

// ─── Constants ────────────────────────────────────────────────────────────────

/** Default particle pool per edge. */
const DEFAULT_PARTICLES_PER_EDGE = 12;

/** Maximum particles across all edges. */
const MAX_FLOW_PARTICLES = 4096;

/** Catmull-Rom tension (standard: 0.5). */
const CR_TENSION = 0.5;

// ─── QoS → flow-speed mapping ────────────────────────────────────────────────
//
// Speed is normalised travel-per-second along the [0,1] spline arc.
// Higher mps (messages-per-second) or BEST_EFFORT → faster visual flow.

interface QoSFlowStyle {
  /** Base speed (travel units / second). */
  speed:      [number, number];
  /** Particle trail length in travel units (head-to-tail alpha gradient). */
  trailLen:   number;
  /** Particle radius in domain units. */
  radius:     number;
  /** Decay rate (alpha per second after arrival). */
  decayRate:  number;
  /** Max random spawn delay in seconds. */
  maxDelay:   number;
}

const QOS_FLOW_STYLE: Record<QoSProfileName, QoSFlowStyle> = {
  SENSOR_DATA: {
    speed:    [0.35, 0.55],
    trailLen: 0.08,
    radius:   2.5,
    decayRate: 1.2,
    maxDelay:  0.15,
  },
  PARAMETERS: {
    speed:    [0.18, 0.30],
    trailLen: 0.12,
    radius:   3.0,
    decayRate: 0.6,
    maxDelay:  0.5,
  },
  TF_STATIC: {
    speed:    [0.08, 0.15],
    trailLen: 0.18,
    radius:   3.5,
    decayRate: 0.35,
    maxDelay:  1.0,
  },
  TOPO_CHANGE: {
    speed:    [0.45, 0.70],
    trailLen: 0.06,
    radius:   2.0,
    decayRate: 1.5,
    maxDelay:  0.1,
  },
  DEFAULT: {
    speed:    [0.20, 0.35],
    trailLen: 0.10,
    radius:   2.8,
    decayRate: 0.8,
    maxDelay:  0.3,
  },
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A 2-D control point for an edge spline. */
export interface FlowPoint {
  x: number;
  y: number;
}

/** One topology edge definition for the flow renderer. */
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

/** Per-particle state (CPU). */
export interface FlowParticle {
  /** Edge this particle belongs to. */
  edgeIndex:  number;
  /** Arc-length fraction [0, 1] along the spline. */
  travel:     number;
  /** Per-particle speed (travel / second). */
  speed:      number;
  /** Remaining spawn delay in seconds. */
  delay:      number;
  /** Lifecycle phase. */
  phase:      FlowPhase;
  /** Current opacity [0, 1]. */
  alpha:      number;
  /** Current world position (computed from spline). */
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
  /** Topology edges. */
  edges:              FlowEdge[];
  /** Called when a particle arrives at the target cell. */
  onArrival?:         OnArrivalFn;
  /** Particles per unit of edge weight (default 12). */
  particlesPerUnit?:  number;
  /** Global max particles (default 4096). */
  maxParticles?:      number;
  /** Global speed multiplier (default 1.0). */
  speedScale?:        number;
  /** Draw spline paths underneath particles (default false). */
  drawSplines?:       boolean;
  /** Spline path opacity when drawSplines is true (default 0.15). */
  splineOpacity?:     number;
  /** Enable additive-style glow (default true). */
  glow?:              boolean;
}

// ─── Catmull-Rom spline evaluation ────────────────────────────────────────────

function clampIdx(i: number, n: number): number {
  return Math.max(0, Math.min(n - 1, i));
}

/**
 * Evaluate a Catmull-Rom spline at normalised arc-length fraction u ∈ [0, 1].
 * Tension is the standard 0.5 (centripetal parameterisation).
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

  // Catmull-Rom basis (tension = 0.5)
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
 * Resolve particle colour from QoS theme and travel progress.
 *
 * The colour pipeline is a simplified version of color-palette.ts:
 *   base colour → screen-blend highlight by travel (midpoint brightest)
 *   → alpha = sin(π·travel) bell curve × particle alpha
 */
function resolveFlowColor(
  theme:  ThemePalette,
  travel: number,
  alpha:  number,
): { css: string; r: number; g: number; b: number; a: number } {
  // sin bell: bright at mid-travel, dim at endpoints
  const bell   = Math.sin(Math.PI * Math.min(Math.max(travel, 0), 1));
  const screen = bell * 0.6;

  // Screen-blend base with highlight
  const r = 1 - (1 - theme.base.r) * (1 - theme.highlight.r * screen);
  const g = 1 - (1 - theme.base.g) * (1 - theme.highlight.g * screen);
  const b = 1 - (1 - theme.base.b) * (1 - theme.highlight.b * screen);

  const a = alpha * bell;

  return { css: toCss(r, g, b, a), r, g, b, a };
}

// ─── RNG ──────────────────────────────────────────────────────────────────────

function rng(seed: number, salt: number): number {
  return ((Math.sin(seed * 127.1 + salt * 311.7) * 43758.5453) % 1 + 1) % 1;
}

// ─── EdgeFlowRenderer ─────────────────────────────────────────────────────────

/**
 * EdgeFlowRenderer
 *
 * CPU particle system that renders edge-flow particles onto a Canvas2D context.
 * Particles travel along Catmull-Rom splines defined by topology edges; their
 * speed and colour are derived from each edge's QoS profile.
 *
 * @example
 * ```ts
 * const renderer = new EdgeFlowRenderer(ctx, {
 *   edges: [
 *     {
 *       edgeId: 'attn→ffn', sourceId: 'attn', targetId: 'ffn',
 *       points: [{ x: 100, y: 200 }, { x: 200, y: 150 }, { x: 300, y: 200 }],
 *       weight: 1.0,
 *       qos: 'SENSOR_DATA',
 *     },
 *   ],
 * });
 *
 * function frame(t: number) {
 *   ctx.clearRect(0, 0, canvas.width, canvas.height);
 *   renderer.update(1 / 60, t / 1000);
 *   renderer.draw();
 *   requestAnimationFrame(frame);
 * }
 * requestAnimationFrame(frame);
 * ```
 */
export class EdgeFlowRenderer {
  private readonly ctx:         CanvasRenderingContext2D;
  private readonly onArrival?:  OnArrivalFn;

  private edges:          FlowEdge[];
  private particles:      FlowParticle[];
  private particlesPerUnit: number;
  private maxParticles:   number;
  private speedScale:     number;
  private drawSplines:    boolean;
  private splineOpacity:  number;
  private glow:           boolean;

  private elapsed = 0;
  private seed    = 0;

  constructor(
    ctx:    CanvasRenderingContext2D,
    config: EdgeFlowRendererConfig,
  ) {
    this.ctx              = ctx;
    this.edges            = config.edges;
    this.onArrival        = config.onArrival;
    this.particlesPerUnit = config.particlesPerUnit ?? DEFAULT_PARTICLES_PER_EDGE;
    this.maxParticles     = config.maxParticles     ?? MAX_FLOW_PARTICLES;
    this.speedScale       = config.speedScale       ?? 1.0;
    this.drawSplines      = config.drawSplines      ?? false;
    this.splineOpacity    = config.splineOpacity    ?? 0.15;
    this.glow             = config.glow             ?? true;

    this.particles = this._initParticles();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Advance all particles by `dt` seconds.
   *
   * @param dt      — frame delta in seconds (e.g. 1/60)
   * @param elapsed — total elapsed time in seconds (used for spawn staggering)
   */
  update(dt: number, elapsed: number): void {
    this.elapsed = elapsed;

    for (let i = 0; i < this.particles.length; i++) {
      const p    = this.particles[i];
      const edge = this.edges[p.edgeIndex];
      if (!edge) continue;

      const style = styleForQoS(edge.qos ?? 'DEFAULT');

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
          const spd = p.speed * this.speedScale;
          p.travel += spd * dt;

          const pos = evalCatmullRom(edge.points, Math.min(p.travel, 0.9999));
          p.x = pos.x;
          p.y = pos.y;

          if (p.travel >= 1.0) {
            p.phase = FlowPhase.DECAY;

            // Fire arrival callback
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

  /**
   * Draw all living particles (and optionally spline paths) to the canvas.
   * Call after `update()`.
   */
  draw(): void {
    const { ctx } = this;

    // ── Optional: draw spline paths ──────────────────────────────────────────
    if (this.drawSplines) {
      this._drawSplinePaths();
    }

    // ── Draw particles ───────────────────────────────────────────────────────
    if (this.glow) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
    }

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      if (p.phase === FlowPhase.DEAD || p.phase === FlowPhase.SPAWN) continue;
      if (p.alpha < 0.004) continue;

      const edge  = this.edges[p.edgeIndex];
      if (!edge) continue;

      const qos   = edge.qos ?? 'DEFAULT';
      const theme = themeForQoS(qos);
      const style = styleForQoS(qos);
      const color = resolveFlowColor(theme, p.travel, p.alpha);

      if (color.a < 0.004) continue;

      // ── Draw glow halo ─────────────────────────────────────────────────
      const r = style.radius;

      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color.css;
      ctx.fill();

      // Outer glow (larger, more transparent)
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
  }

  // ── Edge management ─────────────────────────────────────────────────────────

  /**
   * Replace edges at runtime (e.g. after topology update).
   * Particles are re-initialised.
   */
  setEdges(edges: FlowEdge[]): void {
    this.edges     = edges;
    this.particles = this._initParticles();
  }

  /**
   * Update the QoS profile for a specific edge by ID.
   * Takes effect immediately — existing particles on that edge will adopt the
   * new speed / colour on the next update().
   */
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

  /** Read-only snapshot of all particles (for debug/introspection). */
  get snapshot(): readonly FlowParticle[] { return this.particles; }

  /** Number of active (non-DEAD) particles. */
  get activeCount(): number {
    return this.particles.filter(p => p.phase !== FlowPhase.DEAD).length;
  }

  /** Total particle slots. */
  get totalSlots(): number { return this.particles.length; }

  // ── Private: initialisation ─────────────────────────────────────────────────

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
          x:         startPos.x,
          y:         startPos.y,
        });
      }
    }

    return particles;
  }

  private _respawn(p: FlowParticle, index: number, t: number): void {
    const edge  = this.edges[p.edgeIndex];
    if (!edge) return;

    const style = styleForQoS(edge.qos ?? 'DEFAULT');
    const s     = index * 1.618034 + t;
    const speed = style.speed[0] + rng(s, 0) * (style.speed[1] - style.speed[0]);
    const delay = rng(s, 1) * style.maxDelay;

    const startPos = evalCatmullRom(edge.points, 0);

    p.travel = 0;
    p.speed  = speed;
    p.delay  = delay;
    p.phase  = delay > 0.001 ? FlowPhase.SPAWN : FlowPhase.FLOW;
    p.alpha  = delay > 0.001 ? 0 : 1;
    p.x      = startPos.x;
    p.y      = startPos.y;
  }

  // ── Private: spline path drawing ────────────────────────────────────────────

  private _drawSplinePaths(): void {
    const { ctx } = this;
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
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

/**
 * Create an EdgeFlowRenderer with QoS profiles resolved from a
 * Record<edgeId, QoSProfileName>.
 *
 * @example
 * ```ts
 * const renderer = createEdgeFlowRenderer(ctx, rawEdges, {
 *   'attn→ffn':    'SENSOR_DATA',
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
  return new EdgeFlowRenderer(ctx, { ...config, edges: resolved });
}

/**
 * Create an EdgeFlowRenderer wired to an SPH world's `addFluid` method,
 * automatically injecting fluid at particle arrival positions.
 *
 * @param ctx       Canvas2D rendering context
 * @param edges     Topology edges
 * @param addFluid  SPHWorld.addFluid (or equivalent) — (x0, y0, x1, y1, spacing, species) => void
 * @param config    Additional configuration overrides
 */
export function createEdgeFlowForSPH(
  ctx:       CanvasRenderingContext2D,
  edges:     FlowEdge[],
  addFluid:  (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:    Omit<EdgeFlowRendererConfig, 'edges' | 'onArrival'> = {},
): EdgeFlowRenderer {
  const R = 0.05;
  return new EdgeFlowRenderer(ctx, {
    ...config,
    edges,
    onArrival: (_edgeId, _targetId, x, y) => {
      addFluid(x - R, y - R, x + R, y + R, R * 0.8, 0);
    },
  });
}

// ─── Standalone spline utilities (re-exported for external use) ──────────────

export { evalCatmullRom, splineTangent };

// ─── Defaults re-export ──────────────────────────────────────────────────────

export const EDGE_FLOW_DEFAULTS = {
  particlesPerEdge:  DEFAULT_PARTICLES_PER_EDGE,
  maxFlowParticles:  MAX_FLOW_PARTICLES,
  crTension:         CR_TENSION,
  qosFlowStyles:     QOS_FLOW_STYLE,
} as const;
