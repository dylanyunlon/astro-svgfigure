/**
 * flower-edge-renderer.ts — AT FlowerParticleShader → cell-connection particle flow
 *
 * Adapts Active Theory's FlowerParticleShader (Element_6_Work, GPGPU position
 * texture + spiral motion) to drive a particle stream that flows along each
 * cell → cell edge route (cubic Bézier from edge_routes.json).
 *
 * AT source references:
 *   Element_6_Work / FlowerParticleShader  — GPGPU position ping-pong,
 *       spiral motion formula, vScale size-attenuation, species palette uniforms
 *   spline-particle-life.ts  — vScale = speed * uTimeMultiplier * 0.01
 *   antimatter-compute.ts    — AntimatterFBO ping-pong, GPGPU attribute layout
 *   cell-color-palette.ts    — getSpeciesColors() → species palette
 *   EdgeRenderer.ts          — Bézier math, estimateArcLength, control points
 *
 * Architecture:
 *   FlowerEdgeRenderer  (public class)
 *     ├─ Per-edge FlowerEdgeEmitter  (one per route entry)
 *     │     ├─ FlowerParticle[]  (CPU GPGPU surrogate — avoids WebGL2 dep here)
 *     │     └─ arc-length LUT  (uniform parameterisation over the Bézier)
 *     └─ PixiJS Graphics  (one per edge — GPU-composited point splats)
 *
 * Particle motion (AT FlowerParticleShader spiral formula):
 *   Each particle carries a per-particle angle θ₀ (random seed) plus a radial
 *   amplitude A.  During FLOW the position is:
 *
 *     splinePos(travel) + tangentPerp · A · sin(θ₀ + time · uSpiralSpeed)
 *
 *   This is exactly the AT "spiral motion" that makes the flower-petal clusters:
 *   particles oscillate transversally around the spline axis with sine waves
 *   whose phases are staggered by θ₀, producing a rotating-petal silhouette.
 *
 * vScale (AT size attenuation):
 *   AT FlowerParticleShader derives point size from:
 *     vScale = uSize * (1.0 − travel²)   // fades as particle nears target
 *   We replicate this: pointRadius = BASE_RADIUS * (1.0 − travel * travel)
 *   clamped to [MIN_RADIUS, BASE_RADIUS].
 *
 * Color:
 *   Source-cell species → getSpeciesColors(species).fill → RGBA particle tint.
 *   Alpha fades like AT: alpha = sin(π · travel) so particles are brightest at
 *   the midpoint and invisible at both ends (matches FlowerParticleShader fade).
 *
 * Usage:
 *   const fer = new FlowerEdgeRenderer(stage, cellMap);
 *   // in animation loop:
 *   fer.update(dt, elapsed);
 *   // dispose:
 *   fer.destroy();
 *
 * Pub-sub integration (cell-pubsub-loop branch):
 *   FlowerEdgeRenderer subscribes to CellEventSource to react when a cell
 *   changes species — the edge colour and spiral amplitude are live-updated
 *   without re-allocating particles.
 *
 *   const fer = new FlowerEdgeRenderer(stage, cellMap, eventSource);
 */

import { Container, Graphics } from 'pixi.js';
import type { CellEventSource } from '../CellEventSource';
import { getSpeciesColors } from './cell-color-palette';

// ── JSON asset types ─────────────────────────────────────────────────────────

interface RoutePoint { x: number; y: number; }

interface RouteEntry {
  edge_id: string;
  sources: string[];
  targets: string[];
  is_skip: boolean;
  advanced: { semanticType?: string; routing?: string; curvature?: number; };
  points: RoutePoint[];
}

// ── AT spiral + vScale constants ─────────────────────────────────────────────

/**
 * uTimeMultiplier — global time scale (AT FlowerParticleShader default: 0.17).
 * Drives travel-speed and spiral rotation simultaneously so the two stay in sync.
 */
const U_TIME_MULTIPLIER = 0.17;

/**
 * uSpiralSpeed — angular velocity of the spiral oscillation (rad/s).
 * AT FlowerParticleShader uses a uniform in the range [1.5, 4.0]; we pick 2.4
 * as a visually rich default that avoids aliasing at 60 fps.
 */
const U_SPIRAL_SPEED = 2.4;

/**
 * uSpiralAmplitude — max lateral displacement in canvas pixels.
 * AT: ~8 px for a 1920-wide canvas; we scale relative to edge length.
 * The emitter overrides this per-edge as amplitude = edgeLength * 0.018.
 */
const U_SPIRAL_AMPLITUDE_RATIO = 0.018;

/**
 * BASE_RADIUS — maximum point-splat radius (pixels).
 * AT vScale formula drives size from BASE_RADIUS down to MIN_RADIUS near target.
 */
const BASE_RADIUS = 5.0;
const MIN_RADIUS  = 0.8;

/** Particles per pixel of edge arc-length (determines pool density). */
const PARTICLES_PER_PX = 0.12;

/** [min, max] per-particle travel speed as fraction-of-edge / second. */
const SPEED_RANGE: [number, number] = [0.82, 1.21];

/** Max random spawn delay (seconds).  0 = all start immediately (AT default). */
const MAX_SPAWN_DELAY = 0.4;

/** Opacity decay rate per second after reaching travel = 1. */
const DECAY_RATE = 1.8;

/** Arc-length LUT divisions per Bézier segment. */
const ARC_DIVISIONS = 64;

// ── Particle state ────────────────────────────────────────────────────────────

type ParticlePhase = 'spawn' | 'flow' | 'decay' | 'dead';

interface FlowerParticle {
  /** Arc-length travel fraction [0, 1] */
  travel:  number;
  /** Per-particle speed (fraction/sec, pre-scaled by uTimeMultiplier) */
  speed:   number;
  /** Remaining spawn delay (seconds) */
  delay:   number;
  /** Lifecycle phase */
  phase:   ParticlePhase;
  /** Current opacity [0, 1] */
  alpha:   number;
  /** AT spiral: per-particle phase seed θ₀ (radians) */
  theta0:  number;
  /** AT spiral: lateral amplitude (pixels) — set from edge length at spawn */
  amp:     number;
  /** Current world position X */
  x:       number;
  /** Current world position Y */
  y:       number;
}

// ── Bézier + arc-length helpers ──────────────────────────────────────────────

/** Sample a cubic Bézier at parameter t ∈ [0, 1]. */
function bezierPoint(
  p0: RoutePoint, c0: RoutePoint, c1: RoutePoint, p1: RoutePoint,
  t: number,
): RoutePoint {
  const mt = 1 - t;
  return {
    x: mt * mt * mt * p0.x + 3 * mt * mt * t * c0.x + 3 * mt * t * t * c1.x + t * t * t * p1.x,
    y: mt * mt * mt * p0.y + 3 * mt * mt * t * c0.y + 3 * mt * t * t * c1.y + t * t * t * p1.y,
  };
}

/** Finite-difference tangent of a cubic Bézier at t, normalised. */
function bezierTangent(
  p0: RoutePoint, c0: RoutePoint, c1: RoutePoint, p1: RoutePoint,
  t: number,
): RoutePoint {
  const eps = 0.001;
  const a = bezierPoint(p0, c0, c1, p1, Math.max(0, t - eps));
  const b = bezierPoint(p0, c0, c1, p1, Math.min(1, t + eps));
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.sqrt(dx * dx + dy * dy) + 1e-10;
  return { x: dx / len, y: dy / len };
}

/**
 * Build an arc-length LUT for a cubic Bézier.
 * Returns { lut: Float32Array of cumulative normalised lengths, totalLength }.
 */
function buildBezierArcLUT(
  p0: RoutePoint, c0: RoutePoint, c1: RoutePoint, p1: RoutePoint,
  divisions = ARC_DIVISIONS,
): { lut: Float32Array; totalLength: number } {
  const lut = new Float32Array(divisions + 1);
  lut[0] = 0;
  let prev = p0;
  let cum  = 0;
  for (let i = 1; i <= divisions; i++) {
    const t   = i / divisions;
    const cur = bezierPoint(p0, c0, c1, p1, t);
    const dx  = cur.x - prev.x;
    const dy  = cur.y - prev.y;
    cum += Math.sqrt(dx * dx + dy * dy);
    lut[i] = cum;
    prev = cur;
  }
  const total = cum;
  if (total > 0) for (let i = 1; i <= divisions; i++) lut[i] /= total;
  return { lut, totalLength: total };
}

/** Map uniform arc-length fraction u ∈ [0,1] to Bézier parameter t ∈ [0,1]. */
function arcToParam(u: number, lut: Float32Array): number {
  const n = lut.length;
  if (u <= 0) return 0;
  if (u >= 1) return 1;
  let lo = 0, hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (lut[mid] < u) lo = mid; else hi = mid;
  }
  const segLen = lut[hi] - lut[lo];
  const frac   = segLen > 0 ? (u - lut[lo]) / segLen : 0;
  return (lo + frac) / (n - 1);
}

/** Derive Bézier control points from the route's point list. */
function deriveControlPoints(
  p0: RoutePoint, p1: RoutePoint,
  midPt: RoutePoint | null,
  curvature: number,
): { c0: RoutePoint; c1: RoutePoint } {
  if (midPt) {
    return {
      c0: { x: p0.x + (midPt.x - p0.x) * curvature, y: p0.y + (midPt.y - p0.y) * curvature },
      c1: { x: p1.x + (midPt.x - p1.x) * curvature, y: p1.y + (midPt.y - p1.y) * curvature },
    };
  }
  return {
    c0: { x: p0.x + (p1.x - p0.x) / 3, y: p0.y + (p1.y - p0.y) / 3 },
    c1: { x: p0.x + (p1.x - p0.x) * 2 / 3, y: p0.y + (p1.y - p0.y) * 2 / 3 },
  };
}

// ── FlowerEdgeEmitter ─────────────────────────────────────────────────────────

/**
 * One emitter per edge route.  Owns the FlowerParticle pool and evaluates
 * the AT spiral motion formula each frame.
 */
class FlowerEdgeEmitter {
  readonly edgeId:       string;
  readonly sourceId:     string;
  readonly targetId:     string;
  readonly particles:    FlowerParticle[];

  /** Bézier geometry */
  readonly p0: RoutePoint;
  readonly p1: RoutePoint;
  readonly c0: RoutePoint;
  readonly c1: RoutePoint;

  /** Arc-length LUT for uniform parameterisation */
  private readonly lut: Float32Array;

  /** Total Bézier arc length in pixels */
  readonly arcLength: number;

  /** Species-derived palette colour (packed 0xRRGGBB) */
  paletteHex: number;

  /** Particle RGBA opacity multiplier (tweakable at runtime) */
  opacityScale: number = 1.0;

  /** Spiral amplitude in pixels (= arcLength * U_SPIRAL_AMPLITUDE_RATIO) */
  private amp: number;

  constructor(
    route: RouteEntry,
    sourceSpecies: string,
  ) {
    this.edgeId   = route.edge_id;
    this.sourceId = route.sources[0] ?? '';
    this.targetId = route.targets[0] ?? '';

    const pts     = route.points;
    this.p0       = pts[0];
    this.p1       = pts[pts.length - 1];
    const midPt   = pts.length >= 3 ? pts[Math.floor(pts.length / 2)] : null;
    const isSkip  = route.is_skip || route.advanced?.routing === 'SPLINES';
    const curv    = route.advanced?.curvature ?? (isSkip ? 0.6 : 1.0);

    const { c0, c1 } = deriveControlPoints(this.p0, this.p1, midPt, curv);
    this.c0 = c0;
    this.c1 = c1;

    const { lut, totalLength } = buildBezierArcLUT(this.p0, this.c0, this.c1, this.p1);
    this.lut       = lut;
    this.arcLength = totalLength;
    this.amp       = totalLength * U_SPIRAL_AMPLITUDE_RATIO;

    // Palette from species
    this.paletteHex = 0x90A4AE; // fallback grey
    try {
      const col = getSpeciesColors(sourceSpecies);
      this.paletteHex = col.fill.toNumber() & 0xFFFFFF;
    } catch { /* species not in palette — keep fallback */ }

    // Allocate particle pool
    const count = Math.max(4, Math.round(totalLength * PARTICLES_PER_PX));
    this.particles = [];
    this._allocParticles(count);
  }

  // ── Allocation ──────────────────────────────────────────────────────────────

  private _allocParticles(count: number): void {
    this.particles.length = 0;
    for (let i = 0; i < count; i++) {
      const travel = i / count; // stagger uniformly across spline
      const delay  = Math.random() * MAX_SPAWN_DELAY;
      const speed  = SPEED_RANGE[0] + Math.random() * (SPEED_RANGE[1] - SPEED_RANGE[0]);
      const theta0 = Math.random() * Math.PI * 2;
      const { x, y } = this._evalPos(travel, theta0, 0);

      this.particles.push({
        travel, speed, delay,
        phase:  delay > 0 ? 'spawn' : 'flow',
        alpha:  delay > 0 ? 0 : 1,
        theta0,
        amp:    this.amp,
        x, y,
      });
    }
  }

  // ── Per-frame update ────────────────────────────────────────────────────────

  /**
   * Advance all particles one step.
   *
   * @param dt      — wall-clock delta capped to 50 ms
   * @param elapsed — total elapsed seconds (drives spiral phase)
   */
  update(dt: number, elapsed: number): void {
    const scaledDt = dt * U_TIME_MULTIPLIER;

    for (const p of this.particles) {
      switch (p.phase) {
        case 'dead':
          this._respawn(p);
          break;

        case 'spawn':
          p.delay -= dt;
          if (p.delay <= 0) { p.delay = 0; p.phase = 'flow'; p.alpha = 1; }
          break;

        case 'flow': {
          // AT travel advance: travel += speed * timeMultiplier * dt * flowRange
          p.travel += p.speed * scaledDt;

          // AT spiral motion:
          //   lateralOffset = amp · sin(θ₀ + elapsed · uSpiralSpeed)
          // amp decays with travel² to concentrate spiral energy at mid-spline
          // (mirrors the vScale fade in FlowerParticleShader)
          const spiralPhase = p.theta0 + elapsed * U_SPIRAL_SPEED;
          const travelDecay = 1.0 - p.travel * p.travel; // 1 at source, 0 at target
          const lateralOff  = p.amp * travelDecay * Math.sin(spiralPhase);

          const tClamped = Math.min(p.travel, 1.0);
          const { x, y } = this._evalPos(tClamped, p.theta0, lateralOff);
          p.x = x; p.y = y;

          // AT alpha: sin(π·travel) → bright at midpoint, fade at ends
          p.alpha = Math.sin(Math.PI * Math.min(tClamped, 1.0));

          if (p.travel >= 1.0) {
            p.travel = 1.0;
            p.phase  = 'decay';
          }
          break;
        }

        case 'decay':
          p.alpha -= DECAY_RATE * dt;
          if (p.alpha <= 0) { p.alpha = 0; p.phase = 'dead'; }
          break;
      }
    }
  }

  // ── Position evaluation ─────────────────────────────────────────────────────

  /**
   * Compute world position from arc-length travel + spiral lateral offset.
   *
   * @param travel     — normalised arc-length fraction [0, 1]
   * @param _theta0    — unused here; caller has already computed lateralOff
   * @param lateralOff — signed pixel offset perpendicular to tangent
   */
  private _evalPos(travel: number, _theta0: number, lateralOff: number): RoutePoint {
    const t   = arcToParam(Math.min(travel, 1), this.lut);
    const pos = bezierPoint(this.p0, this.c0, this.c1, this.p1, t);

    if (lateralOff === 0) return pos;

    // Perpendicular to tangent (rotate tangent 90°)
    const tan  = bezierTangent(this.p0, this.c0, this.c1, this.p1, t);
    const perpX = -tan.y;
    const perpY =  tan.x;

    return {
      x: pos.x + perpX * lateralOff,
      y: pos.y + perpY * lateralOff,
    };
  }

  // ── Respawn ─────────────────────────────────────────────────────────────────

  private _respawn(p: FlowerParticle): void {
    p.travel = 0;
    p.speed  = SPEED_RANGE[0] + Math.random() * (SPEED_RANGE[1] - SPEED_RANGE[0]);
    p.delay  = Math.random() * MAX_SPAWN_DELAY;
    p.phase  = p.delay > 0 ? 'spawn' : 'flow';
    p.alpha  = p.delay > 0 ? 0 : 1;
    p.theta0 = Math.random() * Math.PI * 2;
    p.amp    = this.amp;
    const { x, y } = this._evalPos(0, p.theta0, 0);
    p.x = x; p.y = y;
  }

  // ── Live palette update ─────────────────────────────────────────────────────

  /** Called when the source cell changes species (pub-sub). */
  updateSpecies(newSpecies: string): void {
    try {
      const col = getSpeciesColors(newSpecies);
      this.paletteHex = col.fill.toNumber() & 0xFFFFFF;
    } catch { /* keep existing colour */ }
  }
}

// ── FlowerEdgeRenderer ────────────────────────────────────────────────────────

/**
 * FlowerEdgeRenderer — renders spiral particle flow along all cell-connection
 * edges (from edge_routes.json) using AT's FlowerParticleShader motion model.
 *
 * Each particle is drawn as a circular PixiJS Graphics splat whose radius
 * follows AT's vScale attenuation:
 *   pointRadius = BASE_RADIUS · (1 − travel²)  clamped to MIN_RADIUS
 *
 * The renderer keeps one Graphics object per edge (double-buffered via clear())
 * and batches all particle splats for that edge in a single draw cycle.
 *
 * @example
 * ```ts
 * import routesJson from '../../physics/edge_routes.json';
 * const cellMap = new Map([['self_attn', { species: 'cil-eye' }], ...]);
 * const fer = new FlowerEdgeRenderer(stage, cellMap);
 *
 * ticker.add(({ deltaTime, deltaMS }) => {
 *   fer.update(deltaMS / 1000, performance.now() / 1000);
 * });
 * ```
 */
export class FlowerEdgeRenderer {
  private readonly emitters: FlowerEdgeEmitter[] = [];
  /** Graphics objects indexed parallel to emitters */
  private readonly gfxList:  Graphics[] = [];
  private readonly stage:    Container;

  /**
   * Unsubscribe handle from CellEventSource (if provided).
   * Stored so we can detach on destroy().
   */
  private readonly _unsubscribe: (() => void) | null = null;

  /**
   * @param stage      — PixiJS Container to add particle graphics to
   * @param cellMap    — maps cellId → { species } for palette lookup
   * @param eventSrc   — optional CellEventSource for live species updates
   */
  constructor(
    stage: Container,
    cellMap: Map<string, { species: string }>,
    eventSrc?: CellEventSource,
  ) {
    this.stage = stage;

    // Lazy import routes at construction time so this module is treeshakeable
    // when imported server-side (SSR/Astro build).
    let routes: Record<string, RouteEntry>;
    try {
      // Dynamic require — bundler replaces with static import in browser build
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      routes = require('../../../physics/edge_routes.json') as Record<string, RouteEntry>;
    } catch {
      console.warn('[FlowerEdgeRenderer] edge_routes.json not found — no particles will render');
      routes = {};
    }

    for (const key of Object.keys(routes)) {
      const route = routes[key];
      if (!route.points || route.points.length < 2) continue;

      const srcId   = route.sources[0] ?? '';
      const species = cellMap.get(srcId)?.species ?? '';

      const emitter = new FlowerEdgeEmitter(route, species);
      const gfx     = new Graphics();

      stage.addChild(gfx);
      this.emitters.push(emitter);
      this.gfxList.push(gfx);
    }

    // Subscribe to live cell-species changes if pub-sub source provided
    if (eventSrc) {
      const listener = (ev: { cell_id: string }) => {
        // Find all emitters whose source cell matches
        for (const em of this.emitters) {
          if (em.sourceId === ev.cell_id) {
            const newSpecies = cellMap.get(ev.cell_id)?.species ?? '';
            em.updateSpecies(newSpecies);
          }
        }
      };
      eventSrc.addListener(listener as Parameters<typeof eventSrc.addListener>[0]);
      this._unsubscribe = () =>
        eventSrc.removeListener(listener as Parameters<typeof eventSrc.removeListener>[0]);
    }
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  /**
   * Advance simulation and redraw all particle splats.
   *
   * @param dt       — frame delta in seconds (capped to 0.05 internally)
   * @param elapsed  — total elapsed seconds since scene start (for spiral phase)
   */
  update(dt: number, elapsed: number): void {
    const safeDt = Math.min(dt, 0.05);

    for (let i = 0; i < this.emitters.length; i++) {
      const em  = this.emitters[i];
      const gfx = this.gfxList[i];

      em.update(safeDt, elapsed);

      gfx.clear();
      this._drawEmitter(gfx, em);
    }
  }

  // ── Drawing ───────────────────────────────────────────────────────────────

  /**
   * Draw all live particles for one emitter as coloured circle splats.
   *
   * AT vScale size attenuation (FlowerParticleShader):
   *   pointRadius = BASE_RADIUS · (1 − travel²)
   * clamped to MIN_RADIUS so particles are never invisible when alive.
   */
  private _drawEmitter(gfx: Graphics, em: FlowerEdgeEmitter): void {
    const color = em.paletteHex;

    for (const p of em.particles) {
      if (p.phase === 'dead' || p.phase === 'spawn') continue;
      if (p.alpha <= 0.01) continue;

      // AT vScale: size decays quadratically to zero at travel = 1
      const travelDecay = 1.0 - p.travel * p.travel;
      const radius = Math.max(MIN_RADIUS, BASE_RADIUS * travelDecay);

      const alpha = Math.min(1, p.alpha * em.opacityScale);

      gfx.circle(p.x, p.y, radius);
      gfx.fill({ color, alpha });
    }
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  /** Total particle count across all emitters. */
  get particleCount(): number {
    return this.emitters.reduce((n, em) => n + em.particles.length, 0);
  }

  /** Active (flow + decay) particle count across all emitters. */
  get activeParticleCount(): number {
    return this.emitters.reduce((n, em) => {
      return n + em.particles.filter(p => p.phase === 'flow' || p.phase === 'decay').length;
    }, 0);
  }

  /** Number of edge emitters. */
  get edgeCount(): number {
    return this.emitters.length;
  }

  /**
   * Scale global particle opacity (0 = invisible, 1 = full).
   * Useful for fade-in / epoch transitions.
   */
  setOpacity(v: number): void {
    const clamped = Math.max(0, Math.min(1, v));
    for (const em of this.emitters) em.opacityScale = clamped;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Remove all graphics from stage and detach event listeners. */
  destroy(): void {
    this._unsubscribe?.();
    for (const gfx of this.gfxList) {
      this.stage.removeChild(gfx);
      gfx.destroy();
    }
    this.emitters.length = 0;
    this.gfxList.length  = 0;
  }
}

// ── Standalone factory ────────────────────────────────────────────────────────

/**
 * Convenience factory: create a FlowerEdgeRenderer and wire it to
 * a CellEventSource for live species-palette updates.
 *
 * @example
 * ```ts
 * import { createFlowerEdgeRenderer } from '$lib/renderers/flower-edge-renderer';
 * import { getCellEventSource } from '$lib/CellEventSource';
 *
 * const fer = createFlowerEdgeRenderer(stage, cellMap, getCellEventSource());
 * app.ticker.add(({ deltaMS }) => {
 *   fer.update(deltaMS / 1000, app.ticker.lastTime / 1000);
 * });
 * ```
 */
export function createFlowerEdgeRenderer(
  stage:    Container,
  cellMap:  Map<string, { species: string }>,
  eventSrc?: CellEventSource,
): FlowerEdgeRenderer {
  return new FlowerEdgeRenderer(stage, cellMap, eventSrc);
}

// ── Re-export motion constants for external tuning / tests ────────────────────

export const FLOWER_EDGE_DEFAULTS = {
  uTimeMultiplier:        U_TIME_MULTIPLIER,
  uSpiralSpeed:           U_SPIRAL_SPEED,
  uSpiralAmplitudeRatio:  U_SPIRAL_AMPLITUDE_RATIO,
  baseRadius:             BASE_RADIUS,
  minRadius:              MIN_RADIUS,
  particlesPerPx:         PARTICLES_PER_PX,
  speedRange:             SPEED_RANGE,
  maxSpawnDelay:          MAX_SPAWN_DELAY,
  decayRate:              DECAY_RATE,
} as const;
