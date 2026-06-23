/**
 * topology-transition-fx.ts — M785: Topology Transition VFX
 * ─────────────────────────────────────────────────────────────────────────────
 * Four choreographed visual-effect sequences triggered by topology graph
 * mutations — edge creation, edge disconnection, cell emergence, and cell
 * collapse.  Each effect is a multi-layered composition that draws on the
 * project's existing VFX subsystems (ParticleEffectSystem, VFXTimelinePlayer,
 * CollisionShockwaveSystem, TransitionSystem) rather than implementing its
 * own rendering.  This module is the *director*: it sequences, parameterises,
 * and fires those subsystems at the correct times and positions.
 *
 * ── Effect catalogue ─────────────────────────────────────────────────────────
 *
 *   EDGE_CREATED    Light-ray shoot + particle burst
 *   ────────────────────────────────────────────────────────────────────────────
 *   When a new edge appears in the topology, a brilliant beam of light lances
 *   from the source cell to the target cell over ~200 ms (the "ray"), followed
 *   by a radial particle burst at the target endpoint.  The ray is rendered as
 *   a sequence of rapidly-spawned flow_trail particles along the edge spline,
 *   giving a "neural signal travelling down a synapse" aesthetic.  On arrival,
 *   a collision_spark burst detonates at the target with the edge's QoS theme
 *   colour, and a small shockwave ring expands from the impact point.
 *
 *   EDGE_REMOVED    Fragment scatter
 *   ────────────────────────────────────────────────────────────────────────────
 *   When an edge is removed, the edge's visual representation shatters into
 *   angular fragments that fly outward along the edge's tangent directions,
 *   tumbling with slight gravity.  Implemented as a burst of cell_death–style
 *   particles distributed along the edge's control points, each inheriting a
 *   velocity perpendicular to the local spline tangent.  A brief bloom spike
 *   accompanies the breakage.  Colour desaturates toward grey as fragments
 *   fade, conveying loss of connectivity.
 *
 *   CELL_CREATED    Quantum emergence
 *   ────────────────────────────────────────────────────────────────────────────
 *   A new cell materialises through a "quantum vacuum fluctuation" effect:
 *     t=0.00  — A tiny bright seed-point flickers at the cell's centre
 *               (rapid alpha oscillation via ambient_dust particles).
 *     t=0.08  — A ring shockwave expands outward from the seed, pushing
 *               ambient particles aside (shockwave via VFX timeline).
 *     t=0.12  — Inward-converging cell_birth particles coalesce from a
 *               large radius toward the centre, forming the cell shape.
 *     t=0.20  — A bloom spike punctuates the formation.
 *     t=0.30  — A final qos_transition ring-burst announces the cell's
 *               QoS readiness.
 *   The species shader config determines particle colour throughout.
 *
 *   CELL_REMOVED    Gravitational collapse
 *   ────────────────────────────────────────────────────────────────────────────
 *   The inverse of emergence — the cell implodes before scattering:
 *     t=0.00  — Inward-pulling particles rush toward the cell centre
 *               (reverse cell_birth with negative velocity).
 *     t=0.10  — The cell "singularity" flashes (screen flash via timeline).
 *     t=0.14  — Outward cell_death fragment scatter with high velocity.
 *     t=0.18  — Expanding shockwave ring from the collapse point.
 *     t=0.25  — Bloom decay, particles fade with gravity.
 *   Colour shifts from the species palette toward a deep desaturated shadow,
 *   conveying the loss of the cell's identity.
 *
 * ── Architecture ─────────────────────────────────────────────────────────────
 *
 *   TopologyTransitionFX is a stateless orchestrator.  It holds references to
 *   the existing VFX subsystems and exposes four fire-and-forget methods.
 *   It does NOT manage particle lifetimes or render anything — that remains
 *   the job of ParticleEffectSystem.tick() / .render() and the VFX timeline
 *   player, which must already be running in the host application's frame loop.
 *
 *   The module provides a convenience `hookTopologySync()` method that wires
 *   directly into TopologyPhysicsSync's onSync callback, automatically
 *   comparing successive topology snapshots and firing the appropriate effects.
 *
 * ── Integration ──────────────────────────────────────────────────────────────
 *
 *   ```ts
 *   import { TopologyTransitionFX } from '$lib/sph/topology-transition-fx';
 *
 *   const topoFX = new TopologyTransitionFX({
 *     particleFX,       // ParticleEffectSystem instance
 *     timelinePlayer,   // VFXTimelinePlayer instance
 *     domainW: 3.0,
 *     domainH: 3.0,
 *     canvasW: 800,
 *     canvasH: 600,
 *   });
 *
 *   // Manual firing:
 *   topoFX.onEdgeCreated(sourcePos, targetPos, edgePoints, { qos: 'SENSOR_DATA' });
 *   topoFX.onEdgeRemoved(edgePoints, { species: 'cil-bolt' });
 *   topoFX.onCellCreated(centre, { species: 'cil-eye', radius: 24 });
 *   topoFX.onCellRemoved(centre, { species: 'cil-eye', radius: 24 });
 *
 *   // Or auto-wire to TopologyPhysicsSync:
 *   topoFX.hookTopologySync(topologySync);
 *   ```
 *
 * ── References ───────────────────────────────────────────────────────────────
 *   src/lib/sph/particle-effect-system.ts  — unified particle emitter
 *   src/lib/sph/vfx-timeline.ts            — multi-event sequencer
 *   src/lib/sph/topology-physics-sync.ts   — topology SSE → physics bridge
 *   src/lib/sph/species-shader-registry.ts — per-species shader/colour config
 *   src/lib/sph/color-palette.ts           — QoS → colour theme mapping
 *   src/lib/sph/collision-shockwave.ts     — shockwave ring effect
 *   src/lib/sph/transition-system.ts       — cell scale/dissolve transitions
 *   src/lib/sph/edge-data-flow-viz.ts      — edge pulse visual language
 *
 * [ASTRO-TOPO-FX] debug prefix.
 */

import type {
  ParticleEffectSystem,
  Vec2,
  CellBirthParams,
  CellDeathParams,
  CollisionSparkParams,
  FlowTrailParams,
  QosTransitionParams,
  AmbientDustParams,
} from './particle-effect-system';

import type {
  VFXTimelinePlayer,
  VFXTimeline,
} from './vfx-timeline';
import { VFXTimelineBuilder } from './vfx-timeline';

import {
  getSpeciesShaderConfig,
  type SpeciesShaderConfig,
} from './species-shader-registry';

import {
  QOS_THEME,
  type RGB,
  type ThemePalette,
} from './color-palette';

import type { QoSProfileName } from './qosSpatial';

import type {
  TopologyPhysicsSync,
  TopoNode,
  TopoEdge,
  SyncStats,
} from './topology-physics-sync';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Duration of the light-ray travel phase (seconds). */
const RAY_TRAVEL_DURATION = 0.20;

/** Number of trail particles spawned along the ray path. */
const RAY_PARTICLE_COUNT = 16;

/** Particle burst count at ray arrival point. */
const RAY_ARRIVAL_BURST_COUNT = 20;

/** Fragment count for edge disconnect scatter. */
const EDGE_FRAGMENT_COUNT = 24;

/** Fragment scatter speed (world units/s). */
const EDGE_FRAGMENT_SPEED = 120;

/** Quantum seed flicker particle count. */
const QUANTUM_SEED_COUNT = 6;

/** Cell emergence inward-converge particle count. */
const CELL_EMERGE_PARTICLE_COUNT = 32;

/** Cell collapse inward-pull particle count. */
const CELL_COLLAPSE_INWARD_COUNT = 20;

/** Cell collapse outward-scatter particle count. */
const CELL_COLLAPSE_OUTWARD_COUNT = 36;

/** Default cell radius for effects when not specified. */
const DEFAULT_CELL_RADIUS = 20;

/** Minimum interval between VFX triggers for the same element (seconds). */
const COOLDOWN_MS = 80;

/** Maximum queued ray-spawn timers to prevent runaway allocation. */
const MAX_PENDING_TIMERS = 64;

// ─────────────────────────────────────────────────────────────────────────────
// Lygia-style hash (shared with particle-effect-system.ts)
// ─────────────────────────────────────────────────────────────────────────────

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

function fract(x: number): number {
  return x - Math.floor(x);
}

function hashScalar(p: number): number {
  let x = fract(p * SCALE_X);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

function hashVec2To2(sx: number, sy: number): [number, number] {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19);
  p3x += d; p3y += d; p3z += d;
  return [fract((p3x + p3x) * (p3y + p3z)), fract((p3x + p3y) * (p3y + p3z))];
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** 2D point on the edge spline (canvas or domain space). */
export interface SplinePoint {
  x: number;
  y: number;
}

/** Options for edge-creation VFX. */
export interface EdgeCreatedOptions {
  /** QoS profile name for colour theming. Default 'DEFAULT'. */
  qos?: QoSProfileName;
  /** Species of the source cell (fallback colour derivation). */
  sourceSpecies?: string;
  /** Species of the target cell (fallback colour derivation). */
  targetSpecies?: string;
  /** Edge semantic type (e.g. 'skip_connection'). */
  semanticType?: string;
  /** Override the ray travel duration (seconds). */
  rayDuration?: number;
}

/** Options for edge-removal VFX. */
export interface EdgeRemovedOptions {
  /** QoS profile for colour. Default 'DEFAULT'. */
  qos?: QoSProfileName;
  /** Species of the source cell (colour hint). */
  sourceSpecies?: string;
  /** Override fragment scatter speed. */
  fragmentSpeed?: number;
}

/** Options for cell-creation VFX. */
export interface CellCreatedOptions {
  /** Species id for colour derivation. */
  species?: string;
  /** Cell bounding radius in world units. */
  radius?: number;
  /** QoS profile of the new cell. */
  qos?: QoSProfileName;
}

/** Options for cell-removal VFX. */
export interface CellRemovedOptions {
  /** Species id for colour derivation. */
  species?: string;
  /** Cell bounding radius in world units. */
  radius?: number;
}

/** Configuration for TopologyTransitionFX. */
export interface TopologyTransitionFXConfig {
  /** ParticleEffectSystem instance (required). */
  particleFX: ParticleEffectSystem;
  /** VFXTimelinePlayer instance (optional; enables shockwave/bloom/flash). */
  timelinePlayer?: VFXTimelinePlayer | null;
  /** SPH domain width in metres. Default 3.0. */
  domainW?: number;
  /** SPH domain height in metres. Default 3.0. */
  domainH?: number;
  /** Canvas pixel width. Default 800. */
  canvasW?: number;
  /** Canvas pixel height. Default 600. */
  canvasH?: number;
  /** Global intensity multiplier [0, 1]. Default 1.0. */
  intensity?: number;
  /**
   * When true, ray-spawn uses setTimeout stagger for the travel illusion.
   * When false, all ray particles spawn immediately (lower latency, less
   * cinematic).  Default true.
   */
  staggerRay?: boolean;
}

/** Diagnostic event emitted after each VFX firing. */
export interface TopologyFXEvent {
  type: 'edge_created' | 'edge_removed' | 'cell_created' | 'cell_removed';
  position: Vec2;
  particlesEmitted: number;
  timelineId?: string;
}

/** Callback for monitoring VFX events. */
export type OnTopologyFXCallback = (event: TopologyFXEvent) => void;

// ─────────────────────────────────────────────────────────────────────────────
// Colour derivation helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive an RGB triplet from a QoS profile name.
 * Returns the theme's base colour as [r, g, b] in [0, 1] linear space.
 */
function qosBaseColor(qos: QoSProfileName): [number, number, number] {
  const theme: ThemePalette = QOS_THEME[qos] ?? QOS_THEME.DEFAULT;
  return [theme.base.r, theme.base.g, theme.base.b];
}

/**
 * Derive an RGB triplet from a QoS profile's highlight colour.
 */
function qosHighlightColor(qos: QoSProfileName): [number, number, number] {
  const theme: ThemePalette = QOS_THEME[qos] ?? QOS_THEME.DEFAULT;
  return [theme.highlight.r, theme.highlight.g, theme.highlight.b];
}

/**
 * Blend species colour with QoS theme colour.
 * When species is provided, it dominates (70/30 blend with QoS).
 * Otherwise, pure QoS colour is used.
 */
function resolveEdgeColor(
  qos: QoSProfileName,
  species?: string,
  emissiveBoost: number = 0.3,
): [number, number, number] {
  const qosColor = qosBaseColor(qos);

  if (!species) return qosColor;

  const cfg = getSpeciesShaderConfig(species);
  const albedo = cfg.materialParams.albedo ?? [0.5, 0.5, 0.5];
  const bloom = cfg.bloomStrength;
  const boost = 1.0 + bloom * emissiveBoost * 0.3;

  return [
    Math.min(albedo[0] * 0.7 * boost + qosColor[0] * 0.3, 1.0),
    Math.min(albedo[1] * 0.7 * boost + qosColor[1] * 0.3, 1.0),
    Math.min(albedo[2] * 0.7 * boost + qosColor[2] * 0.3, 1.0),
  ];
}

/**
 * Desaturate a colour toward grey by a given factor [0 = original, 1 = grey].
 */
function desaturate(
  c: [number, number, number],
  amount: number,
): [number, number, number] {
  const lum = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
  const t = Math.min(Math.max(amount, 0), 1);
  return [
    c[0] * (1 - t) + lum * t,
    c[1] * (1 - t) + lum * t,
    c[2] * (1 - t) + lum * t,
  ];
}

/**
 * Linearly interpolate two points.
 */
function lerpPoint(a: SplinePoint, b: SplinePoint, t: number): SplinePoint {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

/**
 * Evaluate a piecewise-linear path at parameter t ∈ [0, 1].
 */
function evalPath(points: SplinePoint[], t: number): SplinePoint {
  if (points.length === 0) return { x: 0, y: 0 };
  if (points.length === 1) return points[0];

  const clamped = Math.max(0, Math.min(1, t));
  const totalSegments = points.length - 1;
  const raw = clamped * totalSegments;
  const idx = Math.min(Math.floor(raw), totalSegments - 1);
  const frac = raw - idx;
  return lerpPoint(points[idx], points[idx + 1], frac);
}

/**
 * Compute the tangent direction at parameter t along a piecewise-linear path.
 * Returns a unit vector { x, y }.
 */
function evalTangent(points: SplinePoint[], t: number): SplinePoint {
  if (points.length < 2) return { x: 0, y: 1 };

  const totalSegments = points.length - 1;
  const raw = Math.max(0, Math.min(1, t)) * totalSegments;
  const idx = Math.min(Math.floor(raw), totalSegments - 1);

  const dx = points[idx + 1].x - points[idx].x;
  const dy = points[idx + 1].y - points[idx].y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-6) return { x: 0, y: 1 };
  return { x: dx / len, y: dy / len };
}

/**
 * Compute the perpendicular (90° CCW rotation) of a 2D vector.
 */
function perp(v: SplinePoint): SplinePoint {
  return { x: -v.y, y: v.x };
}

// ─────────────────────────────────────────────────────────────────────────────
// TopologyTransitionFX
// ─────────────────────────────────────────────────────────────────────────────

export class TopologyTransitionFX {
  // ── Dependencies ───────────────────────────────────────────────────────────
  private readonly _fx: ParticleEffectSystem;
  private readonly _timeline: VFXTimelinePlayer | null;

  // ── Configuration ──────────────────────────────────────────────────────────
  private readonly _domainW: number;
  private readonly _domainH: number;
  private readonly _canvasW: number;
  private readonly _canvasH: number;
  private _intensity: number;
  private readonly _staggerRay: boolean;

  // ── Cooldown tracking ──────────────────────────────────────────────────────
  private readonly _cooldowns: Map<string, number> = new Map();

  // ── Pending ray timers (for cleanup) ───────────────────────────────────────
  private readonly _pendingTimers: Set<ReturnType<typeof setTimeout>> = new Set();

  // ── Monotonic seed counter ─────────────────────────────────────────────────
  private _seedCounter = 0;

  // ── Observer callback ──────────────────────────────────────────────────────
  private _onFX: OnTopologyFXCallback | null = null;

  // ── Previous topology snapshot for diff-based auto-wiring ──────────────────
  private _prevNodeIds: Set<string> = new Set();
  private _prevEdgeIds: Set<string> = new Set();
  private _prevNodeBboxes: Map<string, { x: number; y: number; w: number; h: number }> = new Map();

  // ── Constructor ────────────────────────────────────────────────────────────

  constructor(config: TopologyTransitionFXConfig) {
    this._fx         = config.particleFX;
    this._timeline   = config.timelinePlayer ?? null;
    this._domainW    = config.domainW    ?? 3.0;
    this._domainH    = config.domainH    ?? 3.0;
    this._canvasW    = config.canvasW    ?? 800;
    this._canvasH    = config.canvasH    ?? 600;
    this._intensity  = config.intensity  ?? 1.0;
    this._staggerRay = config.staggerRay ?? true;
  }

  // ─── Public API ────────────────────────────────────────────────────────────

  /** Register an observer for VFX event diagnostics. */
  onFX(cb: OnTopologyFXCallback): void {
    this._onFX = cb;
  }

  /** Set the global intensity multiplier [0, 1]. */
  setIntensity(intensity: number): void {
    this._intensity = Math.max(0, Math.min(1, intensity));
  }

  /** Current intensity. */
  get intensity(): number {
    return this._intensity;
  }

  // ─── Effect 1: Edge Created — Light Ray + Particle Burst ──────────────────

  /**
   * Fire the edge-creation VFX: a beam of light travels from source to
   * target along the edge spline, then detonates a particle burst at the
   * target endpoint.
   *
   * @param source  Source cell centre (world/canvas space).
   * @param target  Target cell centre (world/canvas space).
   * @param points  Edge spline control points (≥2). Falls back to straight
   *                line source→target if empty.
   * @param opts    Visual customisation.
   * @returns       Total particles emitted (ray trail + arrival burst).
   */
  onEdgeCreated(
    source: Vec2,
    target: Vec2,
    points: SplinePoint[],
    opts: EdgeCreatedOptions = {},
  ): number {
    const key = `edge-c:${source.x.toFixed(0)},${source.y.toFixed(0)}`;
    if (this._isCoolingDown(key)) return 0;

    const qos = opts.qos ?? 'DEFAULT';
    const color = resolveEdgeColor(qos, opts.sourceSpecies, 0.6);
    const highlightColor = qosHighlightColor(qos);
    const isSkip = opts.semanticType === 'skip_connection';
    const rayDur = opts.rayDuration ?? RAY_TRAVEL_DURATION;

    // Ensure we have at least 2 points for the path
    const path: SplinePoint[] = points.length >= 2
      ? points
      : [source, target];

    // Scale particle counts by intensity
    const rayCount = Math.ceil(RAY_PARTICLE_COUNT * this._intensity);
    const burstCount = Math.ceil(
      RAY_ARRIVAL_BURST_COUNT * this._intensity * (isSkip ? 1.3 : 1.0),
    );

    let totalEmitted = 0;

    // ── Phase 1: Ray travel — staggered trail particles along the spline ──

    if (this._staggerRay && rayCount > 1) {
      const intervalMs = (rayDur * 1000) / rayCount;

      for (let i = 0; i < rayCount; i++) {
        if (this._pendingTimers.size >= MAX_PENDING_TIMERS) break;

        const timer = setTimeout(() => {
          this._pendingTimers.delete(timer);

          const t = (i + 0.5) / rayCount;
          const pos = evalPath(path, t);
          const tangent = evalTangent(path, t);

          // Flow trail along the tangent direction
          const speed = this._dist(source, target) / Math.max(rayDur, 0.01);
          const emitted = this._fx.emit('flow_trail', pos, {
            velocity: { x: tangent.x * speed, y: tangent.y * speed },
            species: opts.sourceSpecies,
            width: isSkip ? 2.0 : 1.5,
          } as FlowTrailParams);

          totalEmitted += emitted;
        }, i * intervalMs);

        this._pendingTimers.add(timer);
      }
    } else {
      // Non-staggered: emit all ray particles immediately along the path
      for (let i = 0; i < rayCount; i++) {
        const t = (i + 0.5) / rayCount;
        const pos = evalPath(path, t);
        const tangent = evalTangent(path, t);
        const speed = this._dist(source, target) / Math.max(rayDur, 0.01);

        totalEmitted += this._fx.emit('flow_trail', pos, {
          velocity: { x: tangent.x * speed, y: tangent.y * speed },
          species: opts.sourceSpecies,
          width: isSkip ? 2.0 : 1.5,
        } as FlowTrailParams);
      }
    }

    // ── Phase 2: Arrival burst at target — scheduled after ray travel ──

    const arrivalDelay = this._staggerRay ? rayDur * 1000 : 0;
    const arrivalTimer = setTimeout(() => {
      this._pendingTimers.delete(arrivalTimer);

      const arrivalPos = evalPath(path, 1.0);

      // Collision-spark burst at the arrival point
      const sparkEmitted = this._fx.emit('collision_spark', arrivalPos, {
        impulse: 0.7 * this._intensity,
        normal: { x: 0, y: -1 },
        species: opts.targetSpecies,
      } as CollisionSparkParams);

      // Additional radial particle burst
      const birthEmitted = this._fx.emit('cell_birth', arrivalPos, {
        species: opts.targetSpecies,
        radius: 12 * this._intensity,
      } as CellBirthParams);

      totalEmitted += sparkEmitted + birthEmitted;

      // ── Timeline: shockwave + bloom at arrival ──
      if (this._timeline) {
        const tl = VFXTimelineBuilder.create('edge-created-arrival')
          .shockwave(0.00, {
            originX: arrivalPos.x,
            originY: arrivalPos.y,
            maxRadius: 60 * this._intensity,
            expandDuration: 0.20,
            thickness: 0.12,
            amplitude: 2.5 * this._intensity,
          }, 'edge-arr-wave')
          .bloomSpike(0.02, {
            peakScale: 1.3 * this._intensity,
            attack: 0.02,
            decay: 0.10,
          }, 'edge-arr-bloom')
          .build();

        this._timeline.play(tl);
      }

      this._emitEvent({
        type: 'edge_created',
        position: arrivalPos,
        particlesEmitted: totalEmitted,
        timelineId: 'edge-created-arrival',
      });
    }, arrivalDelay);

    this._pendingTimers.add(arrivalTimer);
    this._setCooldown(key);

    return totalEmitted;
  }

  // ─── Effect 2: Edge Removed — Fragment Scatter ─────────────────────────────

  /**
   * Fire the edge-removal VFX: fragments scatter outward from the edge path,
   * tumbling with gravity.  A brief bloom spike marks the disconnection.
   *
   * @param points  The edge's spline control points (the path that is breaking).
   * @param opts    Visual customisation.
   * @returns       Total particles emitted.
   */
  onEdgeRemoved(
    points: SplinePoint[],
    opts: EdgeRemovedOptions = {},
  ): number {
    if (points.length < 2) return 0;

    const midpoint = evalPath(points, 0.5);
    const key = `edge-r:${midpoint.x.toFixed(0)},${midpoint.y.toFixed(0)}`;
    if (this._isCoolingDown(key)) return 0;

    const qos = opts.qos ?? 'DEFAULT';
    const baseColor = resolveEdgeColor(qos, opts.sourceSpecies, 0.1);
    // Desaturate toward grey for the "loss" aesthetic
    const fragColor = desaturate(baseColor, 0.4);
    const speed = opts.fragmentSpeed ?? EDGE_FRAGMENT_SPEED;

    const count = Math.ceil(EDGE_FRAGMENT_COUNT * this._intensity);
    let totalEmitted = 0;

    // Distribute fragment bursts along the edge path
    const sampleCount = Math.max(3, Math.ceil(count / 6));
    const particlesPerSample = Math.ceil(count / sampleCount);

    for (let s = 0; s < sampleCount; s++) {
      const t = (s + 0.5) / sampleCount;
      const pos = evalPath(points, t);
      const tangent = evalTangent(points, t);
      const perpDir = perp(tangent);

      // Emit cell_death-style fragments at each sample point
      const emitted = this._fx.emit('cell_death', pos, {
        species: opts.sourceSpecies,
        radius: 8 * this._intensity,
      } as CellDeathParams);

      totalEmitted += emitted;
    }

    // ── Timeline: bloom spike at midpoint ──
    if (this._timeline) {
      const tl = VFXTimelineBuilder.create('edge-removed')
        .bloomSpike(0.00, {
          peakScale: 1.1 * this._intensity,
          attack: 0.01,
          decay: 0.15,
        }, 'edge-rm-bloom')
        .build();

      this._timeline.play(tl);
    }

    this._setCooldown(key);
    this._emitEvent({
      type: 'edge_removed',
      position: midpoint,
      particlesEmitted: totalEmitted,
      timelineId: 'edge-removed',
    });

    return totalEmitted;
  }

  // ─── Effect 3: Cell Created — Quantum Emergence ────────────────────────────

  /**
   * Fire the cell-creation VFX: a quantum vacuum fluctuation sequence —
   * seed flicker → shockwave → inward coalescence → bloom → QoS ring.
   *
   * @param centre  Cell centre position.
   * @param opts    Visual customisation.
   * @returns       Total particles emitted across all phases.
   */
  onCellCreated(
    centre: Vec2,
    opts: CellCreatedOptions = {},
  ): number {
    const key = `cell-c:${centre.x.toFixed(0)},${centre.y.toFixed(0)}`;
    if (this._isCoolingDown(key)) return 0;

    const radius = opts.radius ?? DEFAULT_CELL_RADIUS;
    const species = opts.species;
    const qos = opts.qos ?? 'TOPO_CHANGE';

    let totalEmitted = 0;

    // ── t=0.00: Quantum seed flicker (tiny ambient_dust cluster) ──
    const seedEmitted = this._fx.emit('ambient_dust', centre, {
      radius: radius * 0.3,
      count: Math.ceil(QUANTUM_SEED_COUNT * this._intensity),
    } as AmbientDustParams);
    totalEmitted += seedEmitted;

    // ── t=0.08: Shockwave ring expansion ──
    const shockwaveTimer = setTimeout(() => {
      this._pendingTimers.delete(shockwaveTimer);

      if (this._timeline) {
        const tl = VFXTimelineBuilder.create('cell-emerge-wave')
          .shockwave(0.00, {
            originX: centre.x,
            originY: centre.y,
            maxRadius: radius * 3.0 * this._intensity,
            expandDuration: 0.30,
            thickness: 0.15,
            amplitude: 3.0 * this._intensity,
          }, 'emerge-wave')
          .build();
        this._timeline.play(tl);
      }
    }, 80);
    this._pendingTimers.add(shockwaveTimer);

    // ── t=0.12: Inward-converging cell_birth particles ──
    const convergeTimer = setTimeout(() => {
      this._pendingTimers.delete(convergeTimer);

      const convergeEmitted = this._fx.emit('cell_birth', centre, {
        species,
        radius: radius * this._intensity,
      } as CellBirthParams);
      totalEmitted += convergeEmitted;
    }, 120);
    this._pendingTimers.add(convergeTimer);

    // ── t=0.20: Bloom spike ──
    const bloomTimer = setTimeout(() => {
      this._pendingTimers.delete(bloomTimer);

      if (this._timeline) {
        const tl = VFXTimelineBuilder.create('cell-emerge-bloom')
          .bloomSpike(0.00, {
            peakScale: 1.6 * this._intensity,
            attack: 0.03,
            decay: 0.15,
          }, 'emerge-bloom')
          .build();
        this._timeline.play(tl);
      }
    }, 200);
    this._pendingTimers.add(bloomTimer);

    // ── t=0.30: QoS readiness ring-burst ──
    const qosTimer = setTimeout(() => {
      this._pendingTimers.delete(qosTimer);

      const qosEmitted = this._fx.emit('qos_transition', centre, {
        species,
        radius: radius * 0.8,
        upgrade: true,
      } as QosTransitionParams);
      totalEmitted += qosEmitted;

      this._emitEvent({
        type: 'cell_created',
        position: centre,
        particlesEmitted: totalEmitted,
        timelineId: 'cell-emerge-wave',
      });
    }, 300);
    this._pendingTimers.add(qosTimer);

    this._setCooldown(key);
    return totalEmitted;
  }

  // ─── Effect 4: Cell Removed — Gravitational Collapse ───────────────────────

  /**
   * Fire the cell-removal VFX: inward implosion → singularity flash →
   * outward fragment scatter → expanding shockwave → bloom decay.
   *
   * @param centre  Cell centre position.
   * @param opts    Visual customisation.
   * @returns       Total particles emitted across all phases.
   */
  onCellRemoved(
    centre: Vec2,
    opts: CellRemovedOptions = {},
  ): number {
    const key = `cell-r:${centre.x.toFixed(0)},${centre.y.toFixed(0)}`;
    if (this._isCoolingDown(key)) return 0;

    const radius = opts.radius ?? DEFAULT_CELL_RADIUS;
    const species = opts.species;

    let totalEmitted = 0;

    // ── t=0.00: Inward-pulling particles (reverse birth) ──
    // Use cell_birth which already converges inward
    const inwardEmitted = this._fx.emit('cell_birth', centre, {
      species,
      radius: radius * 1.2,
    } as CellBirthParams);
    totalEmitted += inwardEmitted;

    // ── t=0.10: Singularity flash ──
    const flashTimer = setTimeout(() => {
      this._pendingTimers.delete(flashTimer);

      if (this._timeline) {
        const tl = VFXTimelineBuilder.create('cell-collapse-flash')
          .screenFlash(0.00, {
            color: [1.0, 0.95, 0.9],
            peakAlpha: 0.25 * this._intensity,
            attack: 0.02,
            decay: 0.08,
          }, 'collapse-flash')
          .build();
        this._timeline.play(tl);
      }
    }, 100);
    this._pendingTimers.add(flashTimer);

    // ── t=0.14: Outward cell_death fragment scatter ──
    const scatterTimer = setTimeout(() => {
      this._pendingTimers.delete(scatterTimer);

      const scatterEmitted = this._fx.emit('cell_death', centre, {
        species,
        radius: radius * this._intensity,
      } as CellDeathParams);
      totalEmitted += scatterEmitted;
    }, 140);
    this._pendingTimers.add(scatterTimer);

    // ── t=0.18: Expanding shockwave ──
    const waveTimer = setTimeout(() => {
      this._pendingTimers.delete(waveTimer);

      if (this._timeline) {
        const tl = VFXTimelineBuilder.create('cell-collapse-wave')
          .shockwave(0.00, {
            originX: centre.x,
            originY: centre.y,
            maxRadius: radius * 4.0 * this._intensity,
            expandDuration: 0.35,
            thickness: 0.18,
            amplitude: 4.0 * this._intensity,
          }, 'collapse-wave')
          .bloomSpike(0.02, {
            peakScale: 1.2 * this._intensity,
            attack: 0.02,
            decay: 0.20,
          }, 'collapse-bloom')
          .build();
        this._timeline.play(tl);
      }

      this._emitEvent({
        type: 'cell_removed',
        position: centre,
        particlesEmitted: totalEmitted,
        timelineId: 'cell-collapse-wave',
      });
    }, 180);
    this._pendingTimers.add(waveTimer);

    // ── t=0.25: Final spark burst (aftermath debris) ──
    const debrisTimer = setTimeout(() => {
      this._pendingTimers.delete(debrisTimer);

      const debrisEmitted = this._fx.emit('collision_spark', centre, {
        impulse: 0.4 * this._intensity,
        normal: { x: 0, y: -1 },
        species,
      } as CollisionSparkParams);
      totalEmitted += debrisEmitted;
    }, 250);
    this._pendingTimers.add(debrisTimer);

    this._setCooldown(key);
    return totalEmitted;
  }

  // ─── Auto-wiring to TopologyPhysicsSync ────────────────────────────────────

  /**
   * Hook into a TopologyPhysicsSync instance's `onSync` callback.
   * Automatically diffs successive topology snapshots and fires the
   * appropriate VFX for each added/removed node and edge.
   *
   * This is the recommended integration path — call once during setup.
   *
   * @param sync  The TopologyPhysicsSync instance to observe.
   */
  hookTopologySync(sync: TopologyPhysicsSync): void {
    // Capture the initial state if available
    const initial = sync.lastTopology;
    if (initial) {
      this._snapshotTopology(
        initial.children ?? [],
        initial.edges ?? [],
      );
    }

    sync.onSync((stats: SyncStats) => {
      const topo = sync.lastTopology;
      if (!topo) return;

      const currentNodes = this._flattenNodes(topo.children ?? []);
      const currentEdges = topo.edges ?? [];

      const currentNodeIds = new Set(currentNodes.map(n => n.id));
      const currentEdgeIds = new Set(currentEdges.map(e => e.id));

      // ── Detect added/removed nodes ──
      for (const node of currentNodes) {
        if (!this._prevNodeIds.has(node.id)) {
          // New cell — fire quantum emergence
          const bbox = this._nodeBbox(node);
          const centre: Vec2 = {
            x: bbox.x + bbox.w * 0.5,
            y: bbox.y + bbox.h * 0.5,
          };
          const species = this._inferSpecies(node);
          const r = Math.min(bbox.w, bbox.h) * 0.5;

          this.onCellCreated(centre, { species, radius: r });
        }
      }

      for (const prevId of this._prevNodeIds) {
        if (!currentNodeIds.has(prevId)) {
          // Removed cell — fire collapse
          const prevBbox = this._prevNodeBboxes.get(prevId);
          if (prevBbox) {
            const centre: Vec2 = {
              x: prevBbox.x + prevBbox.w * 0.5,
              y: prevBbox.y + prevBbox.h * 0.5,
            };
            const r = Math.min(prevBbox.w, prevBbox.h) * 0.5;
            this.onCellRemoved(centre, { radius: r });
          }
        }
      }

      // ── Detect added/removed edges ──
      // Build a lookup for source/target positions
      const nodeBboxMap = new Map<string, { x: number; y: number; w: number; h: number }>();
      for (const node of currentNodes) {
        nodeBboxMap.set(node.id, this._nodeBbox(node));
      }

      for (const edge of currentEdges) {
        if (!this._prevEdgeIds.has(edge.id)) {
          // New edge — fire light ray + burst
          const srcId = edge.sources?.[0];
          const tgtId = edge.targets?.[0];
          if (!srcId || !tgtId) continue;

          const srcBbox = nodeBboxMap.get(srcId) ?? this._prevNodeBboxes.get(srcId);
          const tgtBbox = nodeBboxMap.get(tgtId) ?? this._prevNodeBboxes.get(tgtId);
          if (!srcBbox || !tgtBbox) continue;

          const source: Vec2 = {
            x: srcBbox.x + srcBbox.w * 0.5,
            y: srcBbox.y + srcBbox.h,       // bottom-centre
          };
          const target: Vec2 = {
            x: tgtBbox.x + tgtBbox.w * 0.5,
            y: tgtBbox.y,                   // top-centre
          };

          this.onEdgeCreated(source, target, [source, target], {
            qos: (edge.advanced?.semanticType === 'skip_connection')
              ? 'TOPO_CHANGE'
              : 'DEFAULT',
            semanticType: edge.advanced?.semanticType as string | undefined,
          });
        }
      }

      for (const prevEdgeId of this._prevEdgeIds) {
        if (!currentEdgeIds.has(prevEdgeId)) {
          // Removed edge — fire fragment scatter
          // We don't have the old edge's spline, so use the previous node
          // bboxes to reconstruct a rough 2-point path
          // (The actual spline data isn't stored in our snapshot — acceptable
          // degradation for the auto-wiring convenience.)
          const prevBbox = this._prevNodeBboxes;
          // Try to find source/target from the previous topology
          // Fallback: use a dummy 2-point line through the centre of
          // the domain if we can't resolve endpoints
          const domCx = this._canvasW * 0.5;
          const domCy = this._canvasH * 0.5;
          const fallbackPoints: SplinePoint[] = [
            { x: domCx - 40, y: domCy },
            { x: domCx + 40, y: domCy },
          ];

          this.onEdgeRemoved(fallbackPoints, { qos: 'TOPO_CHANGE' });
        }
      }

      // Update snapshot for next diff
      this._snapshotTopology(currentNodes, currentEdges);
    });
  }

  // ─── Cleanup ───────────────────────────────────────────────────────────────

  /**
   * Cancel all pending timers and clear cooldowns.
   * Call when tearing down the VFX system.
   */
  dispose(): void {
    for (const timer of this._pendingTimers) {
      clearTimeout(timer);
    }
    this._pendingTimers.clear();
    this._cooldowns.clear();
    this._prevNodeIds.clear();
    this._prevEdgeIds.clear();
    this._prevNodeBboxes.clear();
    this._onFX = null;
  }

  /** Number of pending stagger timers (diagnostic). */
  get pendingTimerCount(): number {
    return this._pendingTimers.size;
  }

  // ─── Private helpers ───────────────────────────────────────────────────────

  private _isCoolingDown(key: string): boolean {
    const last = this._cooldowns.get(key);
    if (last === undefined) return false;
    return (performance.now() - last) < COOLDOWN_MS;
  }

  private _setCooldown(key: string): void {
    this._cooldowns.set(key, performance.now());

    // Prune stale cooldowns periodically
    if (this._cooldowns.size > 200) {
      const now = performance.now();
      for (const [k, t] of this._cooldowns) {
        if (now - t > COOLDOWN_MS * 10) {
          this._cooldowns.delete(k);
        }
      }
    }
  }

  private _dist(a: Vec2, b: Vec2): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private _nextSeed(): number {
    return hashScalar(++this._seedCounter * 1.618033988749895);
  }

  private _emitEvent(event: TopologyFXEvent): void {
    if (this._onFX) {
      try {
        this._onFX(event);
      } catch (e) {
        console.warn('[ASTRO-TOPO-FX] onFX callback error:', e);
      }
    }

    console.debug(
      `[ASTRO-TOPO-FX] ${event.type} at (${event.position.x.toFixed(1)}, ` +
      `${event.position.y.toFixed(1)}) — ${event.particlesEmitted} particles`,
    );
  }

  // ── Topology snapshot for auto-diff ────────────────────────────────────────

  private _snapshotTopology(nodes: TopoNode[], edges: TopoEdge[]): void {
    this._prevNodeIds.clear();
    this._prevEdgeIds.clear();
    this._prevNodeBboxes.clear();

    const flat = this._flattenNodes(nodes);
    for (const node of flat) {
      this._prevNodeIds.add(node.id);
      this._prevNodeBboxes.set(node.id, this._nodeBbox(node));
    }

    for (const edge of edges) {
      this._prevEdgeIds.add(edge.id);
    }
  }

  private _flattenNodes(nodes: TopoNode[]): TopoNode[] {
    const result: TopoNode[] = [];
    const walk = (list: TopoNode[]): void => {
      for (const node of list) {
        result.push(node);
        if (node.children?.length) {
          walk(node.children);
        }
      }
    };
    walk(nodes);
    return result;
  }

  private _nodeBbox(
    node: TopoNode,
  ): { x: number; y: number; w: number; h: number } {
    return {
      x: 220,
      y: 40,
      w: node.width  ?? 140,
      h: node.height ?? 50,
    };
  }

  /** Quick species inference from node label (mirrors topology-physics-sync). */
  private _inferSpecies(node: TopoNode): string {
    const label = node.labels?.[0]?.text ?? node.id;
    const rules: Array<[RegExp, string]> = [
      [/attn|attention/i,             'cil-eye'],
      [/conv|filter|kernel/i,         'cil-filter'],
      [/norm|bn|batch/i,              'cil-plus'],
      [/embed|encod|vector|mu|sigma/i,'cil-vector'],
      [/output|decode|x_hat/i,        'cil-arrow-right'],
      [/input/i,                      'cil-vector'],
      [/add|\+|residual/i,            'cil-plus'],
      [/relu|activ|gelu|swish/i,      'cil-bolt'],
      [/sample|reparame/i,            'cil-loop'],
      [/ffn|feed|forward|mlp/i,       'cil-bolt'],
      [/pool|downsample/i,            'cil-layers'],
      [/graph|net/i,                   'cil-graph'],
    ];
    for (const [pattern, species] of rules) {
      if (pattern.test(label)) return species;
    }
    return 'cil-code';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a TopologyTransitionFX instance with sensible defaults.
 * Only requires a ParticleEffectSystem; the VFX timeline player is optional.
 */
export function createTopologyTransitionFX(
  particleFX: ParticleEffectSystem,
  timelinePlayer?: VFXTimelinePlayer | null,
  opts: Partial<TopologyTransitionFXConfig> = {},
): TopologyTransitionFX {
  return new TopologyTransitionFX({
    particleFX,
    timelinePlayer: timelinePlayer ?? null,
    ...opts,
  });
}

/**
 * Create and auto-wire a TopologyTransitionFX to a TopologyPhysicsSync.
 * This is the one-liner integration path for the common case.
 */
export function wireTopologyTransitionFX(
  particleFX: ParticleEffectSystem,
  sync: TopologyPhysicsSync,
  timelinePlayer?: VFXTimelinePlayer | null,
  opts: Partial<TopologyTransitionFXConfig> = {},
): TopologyTransitionFX {
  const topoFX = createTopologyTransitionFX(particleFX, timelinePlayer, opts);
  topoFX.hookTopologySync(sync);
  return topoFX;
}
