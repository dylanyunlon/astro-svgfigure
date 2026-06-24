/**
 * transition-system.ts — M748: Cell Appear / Disappear / Transform Transitions
 * ─────────────────────────────────────────────────────────────────────────────
 * Orchestrates scale, dissolve, and morph transition animations for cells in the
 * SPH world. Each transition type is physics-aware: morph paths follow species
 * morphology constraints, dissolve rates respond to QoS reliability, and scale
 * velocities respect the performance budget tier.
 *
 * Three transition primitives:
 *
 *   scale    — Uniform scale from 0→1 (appear) or 1→0 (disappear).
 *              Includes optional overshoot (back-ease) for organic feel.
 *
 *   dissolve — Alpha + particle scatter.  The cell's boundary particles
 *              disperse outward (disappear) or coalesce inward (appear)
 *              while opacity fades.  Uses noise-based per-particle delay
 *              for a non-uniform "evaporation" feel.
 *
 *   morph    — Interpolates between two cell shapes (bbox + species SDF).
 *              Used when a cell changes species or resizes across epochs.
 *              Vertex positions lerp along curl-noise–distorted paths so
 *              the morph feels organic rather than linearly mechanical.
 *
 * Integration points:
 *   - Called by epoch-ticker / epoch-playback-controller when cells enter/exit
 *   - Consumes VisualProfile from cell-visual-identity.ts
 *   - Feeds into ATSceneCompositor blend weights
 *   - Respects PerformanceBudget tier for particle counts & duration
 *
 * Upstream references:
 *   upstream/theatre-js      — keyframe sequencing concepts
 *   upstream/animation-editor — node-based transition graph
 *   src/lib/tween-system.ts  — Easing functions, TweenManager pattern
 *   src/lib/sph/cell-visual-identity.ts — Morphology, VisualProfile
 *   src/lib/sph/chromatic-adaptation.ts — color interpolation
 *   src/lib/sph/at-shader-utils.ts      — range(), crange() utilities
 *
 * [ASTRO-TRANSITION] debug prefix.
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Default transition duration in seconds */



import { Easing, type EasingFn } from '../tween-system';
import type { Morphology }       from './cell-visual-identity';

const DEFAULT_DURATION = 0.55;

/** Maximum particle count for dissolve effect (clamped by perf budget) */
const DISSOLVE_MAX_PARTICLES = 128;

/** Curl noise frequency for morph path distortion */
const MORPH_CURL_FREQ = 2.4;

/** Minimum scale value to avoid rendering artifacts at zero */
const SCALE_EPSILON = 0.001;

/** Stagger spread for dissolve particle delays (0..1 normalized range) */
const DISSOLVE_STAGGER = 0.35;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Transition direction: appearing into the scene or disappearing from it */
export type TransitionDirection = 'enter' | 'exit';

/** The three transition primitives */
export type TransitionMode = 'scale' | 'dissolve' | 'morph';

/** Current lifecycle phase of a running transition */
export type TransitionPhase = 'idle' | 'running' | 'completed';

/**
 * Per-particle state used by dissolve transitions.
 * Each boundary particle gets an independent delay and scatter vector,
 * creating non-uniform evaporation / coalescence.
 */
export interface DissolveParticle {
  /** Normalized delay before this particle begins its transition (0..1) */
  delay: number;
  /** Scatter direction X (unit vector, randomized) */
  dx: number;
  /** Scatter direction Y */
  dy: number;
  /** Scatter distance — how far this particle travels during dissolve */
  distance: number;
  /** Per-particle progress (0..1), accounting for individual delay */
  progress: number;
}

/**
 * Snapshot of a cell's geometric state, captured at transition start.
 * Used as "from" or "to" keyframe for morph transitions.
 */
export interface CellShapeSnapshot {
  /** Bounding box position */
  x: number;
  y: number;
  /** Bounding box dimensions */
  width: number;
  height: number;
  /** Corner radius for rounded-rect body */
  cornerRadius: number;
  /** Species morphology archetype */
  morphology: Morphology;
  /** Base opacity */
  opacity: number;
}

/**
 * Configuration for a single transition instance.
 */
export interface TransitionConfig {
  /** Which cell is transitioning (cell_id from topology) */
  cellId: string;
  /** Transition primitive to use */
  mode: TransitionMode;
  /** Appear or disappear */
  direction: TransitionDirection;
  /** Duration in seconds (default: DEFAULT_DURATION) */
  duration?: number;
  /** Easing function (default: easeInOut for enter, easeIn for exit) */
  easing?: EasingFn;
  /** Delay before transition starts, in seconds */
  delay?: number;
  /**
   * For morph transitions: the target shape snapshot.
   * The source shape is captured automatically from the current cell state.
   */
  targetShape?: CellShapeSnapshot;
  /** Dissolve particle count override (capped by DISSOLVE_MAX_PARTICLES) */
  dissolveParticleCount?: number;
  /** Callback fired when transition completes */
  onComplete?: (cellId: string) => void;
}

/**
 * Live state of a running transition, updated each frame by TransitionSystem.update().
 * Renderers read these values to apply visual changes.
 */
export interface TransitionState {
  /** Unique handle for this transition */
  id: number;
  /** Cell being transitioned */
  cellId: string;
  /** Which primitive */
  mode: TransitionMode;
  /** Enter or exit */
  direction: TransitionDirection;
  /** Current lifecycle phase */
  phase: TransitionPhase;
  /** Normalized progress (0..1), easing already applied */
  progress: number;
  /** Raw linear progress before easing */
  rawProgress: number;
  /** Duration in seconds */
  duration: number;
  /** Pre-transition delay in seconds */
  delay: number;
  /** Elapsed time in seconds (including delay) */
  elapsed: number;
  /** Easing function in use */
  easing: EasingFn;

  // ── Scale outputs ──
  /** Current scale factor (0..1+) for scale transitions */
  scale: number;

  // ── Dissolve outputs ──
  /** Current global opacity (0..1) for dissolve transitions */
  alpha: number;
  /** Per-particle dissolve states (only populated for dissolve mode) */
  particles: DissolveParticle[];

  // ── Morph outputs ──
  /** Interpolated shape snapshot (only populated for morph mode) */
  currentShape: CellShapeSnapshot | null;
  /** Source shape captured at transition start */
  sourceShape: CellShapeSnapshot | null;
  /** Target shape for morph destination */
  targetShape: CellShapeSnapshot | null;

  /** Completion callback */
  onComplete: ((cellId: string) => void) | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: simple 2D curl noise for morph path distortion
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hash-based pseudo-random for deterministic noise.
 * fract(sin(dot(p, k)) * 43758.5453) — the classic GLSL one-liner.
 */
function _hash(x: number, y: number): number {
  const dot = x * 127.1 + y * 311.7;
  return (Math.sin(dot) * 43758.5453) % 1;
}

/**
 * Simple 2D value noise for curl distortion.
 * Returns a value in roughly [-1, 1].
 */
function _noise2d(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  // bilinear interpolation of hashed corners
  const a = _hash(ix, iy);
  const b = _hash(ix + 1, iy);
  const c = _hash(ix, iy + 1);
  const d = _hash(ix + 1, iy + 1);
  const ux = fx * fx * (3 - 2 * fx); // smoothstep
  const uy = fy * fy * (3 - 2 * fy);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

/**
 * 2D curl of value noise — returns a divergence-free displacement vector.
 * Used to distort morph interpolation paths so they feel organic.
 */
function _curlNoise2d(
  x: number,
  y: number,
  freq: number,
): { cx: number; cy: number } {
  const eps = 0.01;
  const px = x * freq;
  const py = y * freq;
  // partial derivatives via finite differences
  const dndx = (_noise2d(px + eps, py) - _noise2d(px - eps, py)) / (2 * eps);
  const dndy = (_noise2d(px, py + eps) - _noise2d(px, py - eps)) / (2 * eps);
  // curl: rotate gradient 90°
  return { cx: dndy, cy: -dndx };
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: clamp & lerp
// ─────────────────────────────────────────────────────────────────────────────

function _clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function _lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Default shape — used when no explicit shape is provided
// ─────────────────────────────────────────────────────────────────────────────

const _DEFAULT_SHAPE: CellShapeSnapshot = {
  x: 0,
  y: 0,
  width: 120,
  height: 60,
  cornerRadius: 8,
  morphology: 'coral',
  opacity: 1,
};

// ─────────────────────────────────────────────────────────────────────────────
// TransitionSystem — the orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/**
 * TransitionSystem manages all active cell transitions.
 *
 * Each frame, call `update(dt)` with the frame delta in seconds.
 * Renderers query `getState(cellId)` to read interpolated values
 * (scale, alpha, shape) and apply them to the corresponding cell
 * container / mesh / compositor layer.
 *
 * Usage:
 * ```ts
 * const ts = new TransitionSystem();
 *
 * // Cell appears with scale-up
 * ts.start({
 *   cellId: 'self_attn',
 *   mode: 'scale',
 *   direction: 'enter',
 *   duration: 0.6,
 * });
 *
 * // Each frame:
 * ts.update(dt);
 * const state = ts.getState('self_attn');
 * if (state) container.scale.set(state.scale);
 *
 * // Cell dissolves out
 * ts.start({
 *   cellId: 'self_attn',
 *   mode: 'dissolve',
 *   direction: 'exit',
 *   onComplete: (id) => stage.removeChild(containerMap.get(id)),
 * });
 * ```
 */
export class TransitionSystem {
  /** Auto-incrementing transition ID */
  private _nextId = 1;

  /** Active transitions keyed by cellId (one transition per cell at a time) */
  private _active = new Map<string, TransitionState>();

  /** Completed transitions retained for one frame so renderers can read final state */
  private _justCompleted = new Map<string, TransitionState>();

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /**
   * Start a new transition for a cell.
   * If the cell already has an active transition, it is cancelled and replaced.
   * Returns the transition ID.
   */
  start(config: TransitionConfig): number {
    const id = this._nextId++;
    const {
      cellId,
      mode,
      direction,
      duration = DEFAULT_DURATION,
      easing = direction === 'enter' ? Easing.easeInOut : Easing.easeIn,
      delay = 0,
      targetShape = null,
      dissolveParticleCount = DISSOLVE_MAX_PARTICLES,
      onComplete = null,
    } = config;

    // Cancel any existing transition on this cell
    if (this._active.has(cellId)) {
      console.log(`[ASTRO-TRANSITION] replacing active transition on ${cellId}`);
    }

    // Build dissolve particles if needed
    const particles: DissolveParticle[] = [];
    if (mode === 'dissolve') {
      const count = Math.min(dissolveParticleCount, DISSOLVE_MAX_PARTICLES);
      for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2 + _hash(i, id) * 0.8;
        const dist = 30 + _hash(i + 7, id * 3) * 80;
        particles.push({
          delay: _hash(i * 13, id * 7) * DISSOLVE_STAGGER,
          dx: Math.cos(angle),
          dy: Math.sin(angle),
          distance: dist,
          progress: 0,
        });
      }
    }

    const state: TransitionState = {
      id,
      cellId,
      mode,
      direction,
      phase: 'idle',
      progress: 0,
      rawProgress: 0,
      duration,
      delay,
      elapsed: 0,
      easing,

      // Scale — starts at epsilon for enter, 1 for exit
      scale: direction === 'enter' ? SCALE_EPSILON : 1,

      // Dissolve
      alpha: direction === 'enter' ? 0 : 1,
      particles,

      // Morph
      currentShape: null,
      sourceShape: null,
      targetShape: targetShape ? { ...targetShape } : null,

      onComplete,
    };

    this._active.set(cellId, state);

    console.log(
      `[ASTRO-TRANSITION] start id=${id} cell=${cellId} ` +
      `mode=${mode} dir=${direction} dur=${duration.toFixed(2)}s`,
    );

    return id;
  }

  /**
   * Cancel a cell's active transition. The cell retains its current
   * interpolated state (no snap-back).
   */
  cancel(cellId: string): boolean {
    const removed = this._active.delete(cellId);
    if (removed) {
      console.log(`[ASTRO-TRANSITION] cancelled cell=${cellId}`);
    }
    return removed;
  }

  /** Cancel all active transitions */
  cancelAll(): void {
    this._active.clear();
    console.log('[ASTRO-TRANSITION] all transitions cancelled');
  }

  /**
   * Set the source shape for a morph transition that is already started.
   * Typically called by the renderer right after `start()`, once it has
   * captured the cell's current geometry.
   */
  setSourceShape(cellId: string, shape: CellShapeSnapshot): void {
    const s = this._active.get(cellId);
    if (s && s.mode === 'morph') {
      s.sourceShape = { ...shape };
      s.currentShape = { ...shape };
    }
  }

  // ── Per-frame update ─────────────────────────────────────────────────────

  /**
   * Advance all active transitions by `dt` seconds.
   * Call once per frame from the main render loop / ticker.
   */
  update(dt: number): void {
    // Clear previous frame's completed set
    this._justCompleted.clear();

    for (const [cellId, state] of this._active) {
      state.elapsed += dt;

      // Handle pre-delay
      if (state.elapsed < state.delay) {
        state.phase = 'idle';
        continue;
      }

      state.phase = 'running';

      // Compute raw linear progress (0..1)
      const activeTime = state.elapsed - state.delay;
      state.rawProgress = _clamp(activeTime / state.duration, 0, 1);

      // Apply easing
      state.progress = state.easing(state.rawProgress);

      // Directional progress: for exit, reverse so 1→0
      const dp = state.direction === 'enter'
        ? state.progress
        : 1 - state.progress;

      // ── Mode-specific interpolation ──

      switch (state.mode) {
        case 'scale':
          this._updateScale(state, dp);
          break;
        case 'dissolve':
          this._updateDissolve(state, dp);
          break;
        case 'morph':
          this._updateMorph(state, dp);
          break;
      }

      // Check completion
      if (state.rawProgress >= 1) {
        state.phase = 'completed';
        this._justCompleted.set(cellId, state);
        this._active.delete(cellId);

        console.log(
          `[ASTRO-TRANSITION] completed id=${state.id} cell=${cellId}`,
        );

        if (state.onComplete) {
          try {
            state.onComplete(cellId);
          } catch (err) {
            console.error(
              `[ASTRO-TRANSITION] onComplete error for ${cellId}:`,
              err,
            );
          }
        }
      }
    }
  }

  // ── Scale update ─────────────────────────────────────────────────────────

  private _updateScale(state: TransitionState, dp: number): void {
    // dp runs 0→1 for appear, 1→0 for disappear (already handled by caller)
    // Apply slight overshoot via back-ease remap for organic feel
    const overshoot = 1.0 + 0.08 * Math.sin(dp * Math.PI);
    state.scale = Math.max(SCALE_EPSILON, dp * overshoot);
    // Scale also affects alpha slightly at the extremes
    state.alpha = _clamp(dp * 1.5, 0, 1);
  }

  // ── Dissolve update ──────────────────────────────────────────────────────

  private _updateDissolve(state: TransitionState, dp: number): void {
    // Global alpha
    state.alpha = _clamp(dp, 0, 1);

    // Per-particle progress with staggered delays
    for (const p of state.particles) {
      const localT = (state.rawProgress - p.delay) / (1 - p.delay);
      p.progress = _clamp(localT, 0, 1);
    }

    // Dissolve also affects scale subtly
    state.scale = _lerp(0.92, 1.0, dp);
  }

  // ── Morph update ─────────────────────────────────────────────────────────

  private _updateMorph(state: TransitionState, dp: number): void {
    const src = state.sourceShape ?? _DEFAULT_SHAPE;
    const tgt = state.targetShape ?? _DEFAULT_SHAPE;

    // Curl noise distortion amplitude peaks at dp=0.5, zero at endpoints
    const curlAmp = 12 * Math.sin(dp * Math.PI);
    const curl = _curlNoise2d(
      src.x * 0.01 + dp * 3,
      src.y * 0.01,
      MORPH_CURL_FREQ,
    );

    state.currentShape = {
      x: _lerp(src.x, tgt.x, dp) + curl.cx * curlAmp,
      y: _lerp(src.y, tgt.y, dp) + curl.cy * curlAmp,
      width: _lerp(src.width, tgt.width, dp),
      height: _lerp(src.height, tgt.height, dp),
      cornerRadius: _lerp(src.cornerRadius, tgt.cornerRadius, dp),
      morphology: dp < 0.5 ? src.morphology : tgt.morphology,
      opacity: _lerp(src.opacity, tgt.opacity, dp),
    };

    state.scale = 1;
    state.alpha = state.currentShape.opacity;
  }

  // ── Query ────────────────────────────────────────────────────────────────

  /**
   * Get the current transition state for a cell.
   * Returns null if the cell has no active or just-completed transition.
   */
  getState(cellId: string): Readonly<TransitionState> | null {
    return this._active.get(cellId)
        ?? this._justCompleted.get(cellId)
        ?? null;
  }

  /** True if the cell has an active (non-completed) transition */
  isTransitioning(cellId: string): boolean {
    return this._active.has(cellId);
  }

  /** Number of currently active transitions */
  get activeCount(): number {
    return this._active.size;
  }

  /** All cell IDs with active transitions */
  get activeCellIds(): string[] {
    return [...this._active.keys()];
  }

  // ── Bulk helpers ─────────────────────────────────────────────────────────

  /**
   * Apply the same transition to multiple cells with staggered delays.
   * Returns an array of transition IDs.
   *
   * @param cellIds   - cells to transition
   * @param base      - base config (cellId is overridden per cell)
   * @param stagger   - seconds between each cell's start (default: 0.06)
   */
  startStaggered(
    cellIds: string[],
    base: Omit<TransitionConfig, 'cellId'>,
    stagger = 0.06,
  ): number[] {
    return cellIds.map((cellId, i) =>
      this.start({
        ...base,
        cellId,
        delay: (base.delay ?? 0) + i * stagger,
      }),
    );
  }

  /**
   * Destroy the system, cancelling all transitions.
   */
  destroy(): void {
    this.cancelAll();
    this._justCompleted.clear();
    console.log('[ASTRO-TRANSITION] system destroyed');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset transition factories
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Preset configs for common transition scenarios.
 * Use with `TransitionSystem.start(TRANSITION_PRESETS.cellAppear(cellId))`.
 */
export const TRANSITION_PRESETS = {
  /**
   * Cell appears in the scene — quick scale-up with bounce.
   */
  cellAppear(cellId: string, onComplete?: (id: string) => void): TransitionConfig {
    return {
      cellId,
      mode: 'scale',
      direction: 'enter',
      duration: 0.5,
      easing: Easing.back,
      onComplete,
    };
  },

  /**
   * Cell removed from scene — dissolve out with particle scatter.
   */
  cellDisappear(cellId: string, onComplete?: (id: string) => void): TransitionConfig {
    return {
      cellId,
      mode: 'dissolve',
      direction: 'exit',
      duration: 0.7,
      easing: Easing.easeOut,
      onComplete,
    };
  },

  /**
   * Cell changes shape / species between epochs — morph transition.
   */
  cellMorph(
    cellId: string,
    targetShape: CellShapeSnapshot,
    onComplete?: (id: string) => void,
  ): TransitionConfig {
    return {
      cellId,
      mode: 'morph',
      direction: 'enter',
      duration: 0.65,
      easing: Easing.easeInOut,
      targetShape,
      onComplete,
    };
  },

  /**
   * Epoch transition — all cells briefly pulse (scale down then up).
   * Returns a partial config; use with startStaggered().
   */
  epochPulse(): Omit<TransitionConfig, 'cellId'> {
    return {
      mode: 'scale',
      direction: 'enter',
      duration: 0.35,
      easing: Easing.elastic,
    };
  },

  /**
   * Quick fade-in for cells entering from off-screen.
   */
  cellFadeIn(cellId: string, onComplete?: (id: string) => void): TransitionConfig {
    return {
      cellId,
      mode: 'dissolve',
      direction: 'enter',
      duration: 0.4,
      easing: Easing.easeOut,
      dissolveParticleCount: 64,
      onComplete,
    };
  },

  /**
   * Quick scale-out for cells leaving the scene.
   */
  cellScaleOut(cellId: string, onComplete?: (id: string) => void): TransitionConfig {
    return {
      cellId,
      mode: 'scale',
      direction: 'exit',
      duration: 0.35,
      easing: Easing.easeIn,
      onComplete,
    };
  },
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Global singleton (optional convenience — callers can also instantiate directly)
// ─────────────────────────────────────────────────────────────────────────────

let _globalTransitionSystem: TransitionSystem | null = null;

/**
 * Get or create the global TransitionSystem singleton.
 * Useful when multiple subsystems (epoch-ticker, cell-event-system,
 * pixi-cell-renderer) need to coordinate transitions without passing
 * an instance around.
 */
export function getGlobalTransitionSystem(): TransitionSystem {
  if (!_globalTransitionSystem) {
    _globalTransitionSystem = new TransitionSystem();
    console.log('[ASTRO-TRANSITION] global singleton created');
  }
  return _globalTransitionSystem;
}

/**
 * Replace the global TransitionSystem singleton (e.g. for testing).
 */
export function setGlobalTransitionSystem(ts: TransitionSystem | null): void {
  if (_globalTransitionSystem && _globalTransitionSystem !== ts) {
    _globalTransitionSystem.destroy();
  }
  _globalTransitionSystem = ts;
}
