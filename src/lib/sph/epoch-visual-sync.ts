/**
 * epoch-visual-sync.ts — M757: Epoch ↔ Visual Pipeline Synchroniser
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Bridges the backend cell-pubsub-loop epoch convergence engine to the
 * frontend AT visual render pipeline.  While `sph-epoch-bridge.ts` (M514)
 * synchronises epoch state to *physics* rigid bodies, this module synchronises
 * epoch state to the *visual* layer: species transition effects, bbox morph
 * animations, convergence-gated render quality, and epoch timeline playback.
 *
 * ── Architecture ──────────────────────────────────────────────────────────────
 *
 *   Backend epoch loop           epoch-visual-sync            Visual pipeline
 *   ──────────────────           ─────────────────            ───────────────
 *                                                             
 *   SSE epoch_completed ────────► EpochVisualSync ──────────► TransitionSystem
 *                                    │  ├─ species diff       (scale/dissolve/morph)
 *   SSE cell_params_updated ───►     │  ├─ bbox morph        
 *                                    │  ├─ convergence gate  ► PerformanceBudget
 *   SSE topology_updated ──────►     │  └─ timeline tick     ► RenderCompositor
 *                                    │                         (quality knob)
 *   /api/cells ◄───────── fetch ─────┘
 *   /api/epoch ◄───────── fetch ─────┘
 *
 * ── Five synchronisation channels ─────────────────────────────────────────────
 *
 *   1. **Species Transition**
 *      When a cell's species changes between epochs (detected by diffing the
 *      previous snapshot against the current one), fire a species transition
 *      via the TransitionSystem.  The transition blends color (Oklab lerp),
 *      SDF shape (smooth-min morph), and decoration pattern (cross-fade)
 *      over the configured duration.
 *
 *   2. **Bbox Morph**
 *      When a cell's bbox changes significantly (delta > MORPH_THRESHOLD_PX),
 *      schedule a smooth position/size interpolation rather than a hard snap.
 *      The morph uses exponential easing to match the backend's GROW_RATE
 *      damping, keeping visual motion in sync with the convergence physics.
 *
 *   3. **Convergence Quality Gate**
 *      The backend epoch loop reports convergence state (converged: bool,
 *      max_delta: number) each epoch.  While converging (max_delta > threshold)
 *      the visual pipeline runs in "draft" quality: reduced bloom passes,
 *      lower particle caps, simplified SDF evaluation.  On convergence the
 *      pipeline transitions to "final" quality over QUALITY_RAMP_DURATION
 *      seconds, smoothly enabling full post-processing.
 *
 *   4. **Epoch Timeline**
 *      Maintains a linear epoch timeline normalised to [0, 1] where 0 is
 *      epoch 0 and 1 is the final converged epoch.  Visual elements can
 *      bind to this progress value for gradual reveal animations (e.g.
 *      edge particles fade in as topology solidifies, cell labels gain
 *      full opacity only near convergence).
 *
 *   5. **Snapshot Rollback**
 *      When the backend triggers a divergence rollback (snapshot_manager.py
 *      M174), the visual pipeline must undo its latest transitions and
 *      restore the visual state to the rollback target epoch.  This channel
 *      replays the target epoch's snapshot through all four channels above,
 *      with a brief "rewind" dissolve effect to signal the rollback to
 *      the viewer.
 *
 * ── Coordinate space ──────────────────────────────────────────────────────────
 *
 *   Backend bboxes are in canvas pixels (origin top-left, Y-down).
 *   Visual positions use the same canvas-pixel space — no domain-unit
 *   conversion is needed (unlike SPHEpochBridge which maps to physics
 *   domain units).  Z-layers are forwarded as-is to the render compositor's
 *   layer sorting.
 *
 * ── SSE event contract ────────────────────────────────────────────────────────
 *
 *   event: epoch_completed
 *   data: {
 *     "epoch": 3,
 *     "converged": false,
 *     "max_delta": 12.4,
 *     "convergence_threshold": 2.0,
 *     "cells_moved": ["self_attn", "ffn"],
 *     "rollback": null
 *   }
 *
 *   event: cell_params_updated
 *   data: {
 *     "cell_id": "self_attn",
 *     "params": {
 *       "bbox": { "x": 120, "y": 200, "w": 180, "h": 65 },
 *       "species": "cil-eye",
 *       "opacity": 0.95,
 *       "species_params": { ... }
 *     }
 *   }
 *
 *   event: topology_updated
 *   data: { "topology": { ... } }
 *
 *   event: epoch_rollback
 *   data: {
 *     "target_epoch": 2,
 *     "reason": "param_divergence",
 *     "divergence_factor": 4.2
 *   }
 *
 * ── Usage ─────────────────────────────────────────────────────────────────────
 *
 *   import { EpochVisualSync } from '$lib/sph/epoch-visual-sync';
 *
 *   const sync = new EpochVisualSync({
 *     canvasW: canvas.width,
 *     canvasH: canvas.height,
 *     maxEpochs: 10,
 *     onSpeciesTransition: (cellId, from, to) => { ... },
 *     onBboxMorph:         (cellId, target)   => { ... },
 *     onQualityChange:     (quality)          => { ... },
 *     onTimelineProgress:  (t)                => { ... },
 *     onRollback:          (targetEpoch)      => { ... },
 *     onEpochSnapshot:     (snapshot)         => { ... },
 *   });
 *   sync.start();
 *   // per-frame from rAF:
 *   sync.tick(dt);
 *   // teardown:
 *   sync.destroy();
 *
 * ── References ────────────────────────────────────────────────────────────────
 *   src/lib/sph/sph-epoch-bridge.ts        — M514: epoch → physics bodies
 *   src/lib/sph/topology-physics-sync.ts   — M756: topology → physics bodies
 *   src/lib/species-transition-effect.ts   — M758: species color/SDF/pattern morph
 *   src/lib/sph/transition-system.ts       — M748: scale/dissolve/morph animations
 *   src/lib/sph/render-compositor.ts       — M745: 13-pass AT render pipeline
 *   src/lib/sph/performance-budget.ts      — adaptive quality tier system
 *   src/lib/sph/species-visual-dna.ts      — M733: per-cell visual configuration
 *   src/lib/CellEventSource.ts             — SSE event stream client
 *   channels/epoch_controller.py           — backend epoch growth + convergence
 *   channels/snapshot_manager.py           — M174: divergence rollback snapshots
 *
 * [ASTRO-EPOCH-VIS] debug prefix.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────


import type { TransitionDirection, TransitionMode } from './transition-system';

<<<<<<< HEAD
// [orphan-precise] /** SSE endpoint for cell events. */
=======
/** SSE endpoint for cell events. */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const SSE_URL       = '/api/cell-events';

/** REST endpoint for bulk cell data. */
const CELLS_API_URL = '/api/cells';

/** REST endpoint for current epoch metadata. */
const EPOCH_API_URL = '/api/epoch';

/** Bbox change threshold (px) below which morph animation is skipped. */
const MORPH_THRESHOLD_PX = 1.5;

/**
 * Exponential morph factor per tick.  Chosen to visually match the backend's
 * _GROW_RATE = 0.10 at ~60 fps:  1 - (1 - 0.10)^(1/6) ≈ 0.017.
 * Each rAF tick closes 1.7 % of the remaining gap.
 */
const MORPH_EXP_FACTOR = 0.017;

/** Morph snapping threshold — when remaining delta < this, snap to target. */
const MORPH_SNAP_PX = 0.3;

/**
 * Duration (seconds) of the quality ramp from draft → final when convergence
 * is achieved.  Matches the transition-system DEFAULT_DURATION so the quality
 * boost and any final species transition complete simultaneously.
 */
const QUALITY_RAMP_DURATION = 0.55;

/** Draft quality level (0 → worst, 1 → best).  Used while still converging. */
const QUALITY_DRAFT = 0.35;

/** Final quality level (full post-processing enabled). */
const QUALITY_FINAL = 1.0;

/**
 * Rewind dissolve duration (seconds) for rollback visual effect.
 * Slightly faster than a normal transition to feel "snappy".
 */
const ROLLBACK_DISSOLVE_DURATION = 0.35;

// ── SSE reconnect (mirrors CellEventSource / SPHEpochBridge) ─────────────

const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS     = 30_000;
const ERROR_THRESHOLD    = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Bbox in canvas pixel space (origin top-left, Y-down). */
export interface VisualBbox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Per-cell visual snapshot at a given epoch. */
export interface CellVisualSnapshot {
  cellId:         string;
  bbox:           VisualBbox;
  species:        string;
  opacity:        number;
  z:              number;
  speciesParams:  Record<string, number>;
}

/** Full epoch snapshot — all cells at a point in time. */
export interface EpochSnapshot {
  epoch:       number;
  converged:   boolean;
  maxDelta:    number;
  cells:       Map<string, CellVisualSnapshot>;
  timestamp:   number;
}

/** Convergence quality level. */
export interface QualityState {
  /** Current blended quality in [0, 1]. */
  level:       number;
  /** True when the backend has declared convergence. */
  converged:   boolean;
  /** Raw max_delta from the last epoch. */
  maxDelta:    number;
  /** Normalised convergence progress (1 - maxDelta / threshold), clamped. */
  progress:    number;
}

/** Morph target for a single cell's bbox animation. */
export interface BboxMorphTarget {
  cellId:  string;
  /** Current interpolated bbox (updated each tick). */
  current: VisualBbox;
  /** Target bbox from the latest epoch. */
  target:  VisualBbox;
  /** True once current has snapped to target (morph complete). */
  settled: boolean;
}

/** Epoch timeline state. */
export interface TimelineState {
  /** Current epoch index (integer). */
  epoch:      number;
  /** Maximum expected epochs from backend config. */
  maxEpochs:  number;
  /** Normalised progress [0, 1]. */
  progress:   number;
  /** True after convergence is declared. */
  complete:   boolean;
}

/** SSE epoch_completed payload. */
export interface EpochCompletedPayload {
  epoch:                   number;
  converged:               boolean;
  max_delta:               number;
  convergence_threshold?:  number;
  cells_moved?:            string[];
  rollback?:               { target_epoch: number; reason: string } | null;
}

/** SSE cell_params_updated payload. */
export interface CellParamsPayload {
  cell_id: string;
  params: {
    bbox?:           VisualBbox;
    species?:        string;
    opacity?:        number;
    z?:              number;
    species_params?: Record<string, number>;
    [key: string]:   unknown;
  };
}

/** SSE epoch_rollback payload. */
export interface RollbackPayload {
  target_epoch:      number;
  reason:            string;
  divergence_factor?: number;
}

/** Bulk /api/cells response descriptor. */
export interface CellApiDescriptor {
  cell_id?:  string;
  id?:       string;
  bbox?:     VisualBbox;
  species?:  string;
  opacity?:  number;
  z?:        number;
  species_params?: Record<string, number>;
  [key: string]: unknown;
}

// ── Callback signatures ─────────────────────────────────────────────────────

export type OnSpeciesTransition = (
  cellId: string, fromSpecies: string, toSpecies: string, epoch: number,
) => void;

export type OnBboxMorph = (cellId: string, target: VisualBbox) => void;

export type OnQualityChange = (quality: QualityState) => void;

export type OnTimelineProgress = (timeline: TimelineState) => void;

export type OnRollback = (targetEpoch: number, reason: string) => void;

export type OnEpochSnapshot = (snapshot: EpochSnapshot) => void;

export type OnCellEnter = (cellId: string, snapshot: CellVisualSnapshot) => void;

export type OnCellExit = (cellId: string) => void;

// ── Options ─────────────────────────────────────────────────────────────────

export interface EpochVisualSyncOptions {
  /** Canvas width in pixels for coordinate reference (default: 800). */
  canvasW?:              number;
  /** Canvas height in pixels for coordinate reference (default: 600). */
  canvasH?:              number;
  /** Maximum expected epochs (default: 10, from FAstroRendererConfig). */
  maxEpochs?:            number;
  /** Convergence threshold in pixels (default: 2.0, from _CONVERGENCE_PX). */
  convergenceThreshold?: number;
  /** Bbox morph exponential factor override (default: MORPH_EXP_FACTOR). */
  morphFactor?:          number;
  /** Quality ramp duration override in seconds (default: 0.55). */
  qualityRampDuration?:  number;
  /** Species transition duration in seconds (default: 0.65). */
  speciesTransitionDuration?: number;
  /** Pre-existing EventSource to reuse (bridge will NOT close it). */
  eventSource?:          EventSource;

  // ── Callbacks ───────────────────────────────────────────────────────────
  onSpeciesTransition?:  OnSpeciesTransition;
  onBboxMorph?:          OnBboxMorph;
  onQualityChange?:      OnQualityChange;
  onTimelineProgress?:   OnTimelineProgress;
  onRollback?:           OnRollback;
  onEpochSnapshot?:      OnEpochSnapshot;
  onCellEnter?:          OnCellEnter;
  onCellExit?:           OnCellExit;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal tracked cell state
// ─────────────────────────────────────────────────────────────────────────────

interface TrackedCell {
  /** Latest known species (for diffing). */
  species:        string;
  /** Authoritative bbox from the backend (morph target). */
  targetBbox:     VisualBbox;
  /** Smoothly interpolated bbox (updated each tick). */
  currentBbox:    VisualBbox;
  /** Current opacity. */
  opacity:        number;
  /** Z-layer index. */
  z:              number;
  /** Species-specific visual params. */
  speciesParams:  Record<string, number>;
  /** True once currentBbox has converged to targetBbox. */
  bboxSettled:    boolean;
  /** Epoch at which this cell was last updated. */
  lastEpoch:      number;
}

// ─────────────────────────────────────────────────────────────────────────────
// EpochVisualSync
// ─────────────────────────────────────────────────────────────────────────────

export class EpochVisualSync {

  // ── Configuration ──────────────────────────────────────────────────────

  private readonly _canvasW:              number;
  private readonly _canvasH:              number;
  private readonly _maxEpochs:            number;
  private readonly _convergenceThreshold: number;
  private readonly _morphFactor:          number;
  private readonly _qualityRampDuration:  number;
  private readonly _speciesTransDuration: number;

  // ── Callbacks ──────────────────────────────────────────────────────────

  private readonly _onSpeciesTransition:  OnSpeciesTransition | null;
  private readonly _onBboxMorph:          OnBboxMorph | null;
  private readonly _onQualityChange:      OnQualityChange | null;
  private readonly _onTimelineProgress:   OnTimelineProgress | null;
  private readonly _onRollback:           OnRollback | null;
  private readonly _onEpochSnapshot:      OnEpochSnapshot | null;
  private readonly _onCellEnter:          OnCellEnter | null;
  private readonly _onCellExit:           OnCellExit | null;

  // ── Cell registry ──────────────────────────────────────────────────────

  private _tracked: Map<string, TrackedCell> = new Map();

  // ── Epoch timeline ─────────────────────────────────────────────────────

  private _currentEpoch  = 0;
  private _converged     = false;
  private _maxDelta      = Infinity;

  // ── Quality ramp ───────────────────────────────────────────────────────

  /**
   * Current visual quality in [0, 1].  While converging, stays at
   * QUALITY_DRAFT.  After convergence, ramps to QUALITY_FINAL over
   * _qualityRampDuration seconds.
   */
  private _quality       = QUALITY_DRAFT;
  private _qualityTarget = QUALITY_DRAFT;
  private _qualityVelocity = 0;

  // ── Snapshot history (ring buffer for rollback) ────────────────────────

  private _snapshots: EpochSnapshot[] = [];
  private readonly _maxSnapshotHistory = 16;

  // ── SSE lifecycle ──────────────────────────────────────────────────────

  private _es:               EventSource | null = null;
  private _ownsEventSource = true;
  private _started         = false;
  private _destroyed       = false;

  // ── Reconnect / backoff ────────────────────────────────────────────────

  private _errorCount      = 0;
  private _backoffMs       = BACKOFF_INITIAL_MS;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // ── In-flight fetch guard ──────────────────────────────────────────────

  private _fetchingCells   = false;

  // ── Constructor ────────────────────────────────────────────────────────

  constructor(opts: EpochVisualSyncOptions = {}) {
    this._canvasW              = opts.canvasW              ?? 800;
    this._canvasH              = opts.canvasH              ?? 600;
    this._maxEpochs            = opts.maxEpochs            ?? 10;
    this._convergenceThreshold = opts.convergenceThreshold ?? 2.0;
    this._morphFactor          = opts.morphFactor          ?? MORPH_EXP_FACTOR;
    this._qualityRampDuration  = opts.qualityRampDuration  ?? QUALITY_RAMP_DURATION;
    this._speciesTransDuration = opts.speciesTransitionDuration ?? 0.65;

    this._onSpeciesTransition = opts.onSpeciesTransition ?? null;
    this._onBboxMorph         = opts.onBboxMorph         ?? null;
    this._onQualityChange     = opts.onQualityChange     ?? null;
    this._onTimelineProgress  = opts.onTimelineProgress  ?? null;
    this._onRollback          = opts.onRollback          ?? null;
    this._onEpochSnapshot     = opts.onEpochSnapshot     ?? null;
    this._onCellEnter         = opts.onCellEnter         ?? null;
    this._onCellExit          = opts.onCellExit          ?? null;

    if (opts.eventSource) {
      this._es              = opts.eventSource;
      this._ownsEventSource = false;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Public API
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Open the SSE connection and begin listening for epoch events.
   * Performs an initial `/api/cells` fetch to bootstrap the cell registry.
   */
  start(): void {
    if (this._started || this._destroyed) return;
    this._started = true;

    console.info('[ASTRO-EPOCH-VIS] starting visual sync');

    // Bootstrap: fetch initial cell state
    this._fetchAndApplyCells('bootstrap');

    // Fetch initial epoch metadata
    this._fetchEpochMeta();

    // Open SSE
    this._connectSSE();
  }

  /**
   * Advance bbox morph interpolation and quality ramp by dt seconds.
   * Call this from the main rAF loop.
   *
   * @param dt  Delta time in seconds since last tick.
   * @returns   The current quality state for the render pipeline.
   */
  tick(dt: number): QualityState {
    if (this._destroyed) {
      return this._buildQualityState();
    }

    // ── Bbox morph interpolation ──────────────────────────────────────
    this._tickBboxMorphs(dt);

    // ── Quality ramp ──────────────────────────────────────────────────
    this._tickQualityRamp(dt);

    return this._buildQualityState();
  }

  /**
   * Get the current epoch timeline state.
   */
  getTimeline(): TimelineState {
    return {
      epoch:     this._currentEpoch,
      maxEpochs: this._maxEpochs,
      progress:  Math.min(1.0, this._currentEpoch / Math.max(1, this._maxEpochs)),
      complete:  this._converged,
    };
  }

  /**
   * Get the current visual bbox for a cell (smoothly interpolated).
   * Returns null if the cell is not tracked.
   */
  getCellBbox(cellId: string): VisualBbox | null {
    const tracked = this._tracked.get(cellId);
    return tracked ? { ...tracked.currentBbox } : null;
  }

  /**
   * Get the full visual snapshot for a cell.
   */
  getCellSnapshot(cellId: string): CellVisualSnapshot | null {
    const t = this._tracked.get(cellId);
    if (!t) return null;
    return {
      cellId,
      bbox:          { ...t.currentBbox },
      species:       t.species,
      opacity:       t.opacity,
      z:             t.z,
      speciesParams: { ...t.speciesParams },
    };
  }

  /**
   * Get all currently tracked cell IDs.
   */
  getTrackedCellIds(): string[] {
    return Array.from(this._tracked.keys());
  }

  /**
   * Get a snapshot for a specific historical epoch (if retained in the
   * ring buffer).  Returns null if the epoch has been evicted.
   */
  getHistoricalSnapshot(epoch: number): EpochSnapshot | null {
    return this._snapshots.find(s => s.epoch === epoch) ?? null;
  }

  /**
   * Force an immediate resync by fetching /api/cells and reprocessing.
   * Useful after reconnection or when the caller detects state drift.
   */
  forceResync(): void {
    if (this._destroyed) return;
    console.info('[ASTRO-EPOCH-VIS] forced resync');
    this._fetchAndApplyCells('forced-resync');
  }

  /**
   * Close the SSE connection and release resources.
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._started   = false;

    // Close SSE
    if (this._es && this._ownsEventSource) {
      this._es.close();
    }
    this._es = null;

    // Cancel pending reconnect
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }

    // Clear state
    this._tracked.clear();
    this._snapshots.length = 0;

    console.info('[ASTRO-EPOCH-VIS] destroyed');
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SSE connection management
  // ═══════════════════════════════════════════════════════════════════════

  private _connectSSE(): void {
    if (this._destroyed) return;

    if (!this._es) {
      this._es = new EventSource(SSE_URL);
      this._ownsEventSource = true;
    }

    const es = this._es!;

    // ── epoch_completed ──────────────────────────────────────────────
    es.addEventListener('epoch_completed', (ev: MessageEvent) => {
      this._onSSEEpochCompleted(ev);
    });

    // ── cell_params_updated ──────────────────────────────────────────
    es.addEventListener('cell_params_updated', (ev: MessageEvent) => {
      this._onSSECellParamsUpdated(ev);
    });

    // ── topology_updated ─────────────────────────────────────────────
    es.addEventListener('topology_updated', (ev: MessageEvent) => {
      this._onSSETopologyUpdated(ev);
    });

    // ── epoch_rollback ───────────────────────────────────────────────
    es.addEventListener('epoch_rollback', (ev: MessageEvent) => {
      this._onSSERollback(ev);
    });

    // ── Generic message (fallback for non-typed events) ──────────────
    es.onmessage = (ev: MessageEvent) => {
      // Some backends send untyped messages; try to detect epoch data
      try {
        const data = JSON.parse(ev.data);
        if (typeof data.epoch === 'number' && data.converged !== undefined) {
          this._processEpochCompleted(data as EpochCompletedPayload);
        }
      } catch {
        // Not JSON or not epoch data — ignore
      }
    };

    // ── Error handling + exponential backoff reconnect ────────────────
    es.onerror = () => {
      this._errorCount++;

      if (this._errorCount >= ERROR_THRESHOLD) {
        console.warn(
          `[ASTRO-EPOCH-VIS] SSE error streak (${this._errorCount}), ` +
          `backing off ${this._backoffMs}ms`,
        );

        // Close current connection and schedule reconnect
        if (this._ownsEventSource && this._es) {
          this._es.close();
          this._es = null;
        }

        this._reconnectTimer = setTimeout(() => {
          this._reconnectTimer = null;
          if (!this._destroyed) {
            this._connectSSE();
          }
        }, this._backoffMs);

        // Exponential backoff with cap
        this._backoffMs = Math.min(this._backoffMs * 2, BACKOFF_MAX_MS);
      }
    };

    // ── Successful open resets backoff ────────────────────────────────
    es.onopen = () => {
      console.info('[ASTRO-EPOCH-VIS] SSE connected');
      this._errorCount = 0;
      this._backoffMs  = BACKOFF_INITIAL_MS;

      // Re-fetch cells on reconnect to catch any events we missed
      this._fetchAndApplyCells('sse-reconnect');
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SSE event handlers
  // ═══════════════════════════════════════════════════════════════════════

  private _onSSEEpochCompleted(ev: MessageEvent): void {
    let payload: EpochCompletedPayload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      console.error('[ASTRO-EPOCH-VIS] Failed to parse epoch_completed:', ev.data);
      return;
    }
    this._processEpochCompleted(payload);
  }

  private _processEpochCompleted(payload: EpochCompletedPayload): void {
    const prevEpoch    = this._currentEpoch;
    const prevConverged = this._converged;

    this._currentEpoch = payload.epoch;
    this._converged    = payload.converged;
    this._maxDelta     = payload.max_delta;

    console.debug(
      `[ASTRO-EPOCH-VIS] epoch_completed e=${payload.epoch} ` +
      `converged=${payload.converged} max_delta=${payload.max_delta.toFixed(2)}`,
    );

    // ── Check for inline rollback signal ──────────────────────────────
    if (payload.rollback) {
      this._processRollback({
        target_epoch:      payload.rollback.target_epoch,
        reason:            payload.rollback.reason,
        divergence_factor: undefined,
      });
      return; // Rollback handling includes its own cell fetch
    }

    // ── Update convergence quality gate ───────────────────────────────
    if (payload.converged && !prevConverged) {
      // Convergence just achieved — begin quality ramp to final
      this._qualityTarget = QUALITY_FINAL;
      console.info('[ASTRO-EPOCH-VIS] convergence achieved — ramping to final quality');
    } else if (!payload.converged) {
      this._qualityTarget = QUALITY_DRAFT;
    }

    // ── Fetch full cell state for this epoch ──────────────────────────
    this._fetchAndApplyCells(`epoch-${payload.epoch}`);

    // ── Emit timeline progress ────────────────────────────────────────
    if (this._onTimelineProgress) {
      this._onTimelineProgress(this.getTimeline());
    }
  }

  private _onSSECellParamsUpdated(ev: MessageEvent): void {
    let payload: CellParamsPayload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      console.error('[ASTRO-EPOCH-VIS] Failed to parse cell_params_updated:', ev.data);
      return;
    }

    const { cell_id, params } = payload;
    if (!cell_id || !params) return;

    this._applySingleCellUpdate(cell_id, params);
  }

  private _onSSETopologyUpdated(_ev: MessageEvent): void {
    console.info('[ASTRO-EPOCH-VIS] topology_updated — full resync');
    // Topology may have added/removed cells — do a full reconcile
    this._fetchAndApplyCells('topology-updated');
  }

  private _onSSERollback(ev: MessageEvent): void {
    let payload: RollbackPayload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      console.error('[ASTRO-EPOCH-VIS] Failed to parse epoch_rollback:', ev.data);
      return;
    }
    this._processRollback(payload);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Channel 1: Species Transition Detection
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Detects species changes by comparing the tracked cell's current species
   * against the incoming species.  Fires the species transition callback
   * when a change is detected.
   */
  private _detectSpeciesChange(
    cellId: string, tracked: TrackedCell, newSpecies: string,
  ): void {
    if (tracked.species === newSpecies) return;

    const fromSpecies = tracked.species;
    tracked.species   = newSpecies;

    console.info(
      `[ASTRO-EPOCH-VIS] species change "${cellId}": ` +
      `${fromSpecies} → ${newSpecies} (epoch ${this._currentEpoch})`,
    );

    if (this._onSpeciesTransition) {
      this._onSpeciesTransition(cellId, fromSpecies, newSpecies, this._currentEpoch);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Channel 2: Bbox Morph
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Update a tracked cell's morph target.  If the delta exceeds the
   * threshold, the morph runs; otherwise the bbox snaps immediately.
   */
  private _updateBboxTarget(tracked: TrackedCell, newBbox: VisualBbox): void {
    const dx = Math.abs(newBbox.x - tracked.targetBbox.x);
    const dy = Math.abs(newBbox.y - tracked.targetBbox.y);
    const dw = Math.abs(newBbox.w - tracked.targetBbox.w);
    const dh = Math.abs(newBbox.h - tracked.targetBbox.h);
    const maxDelta = Math.max(dx, dy, dw, dh);

    // Always update the target
    tracked.targetBbox = { ...newBbox };

    if (maxDelta < MORPH_THRESHOLD_PX) {
      // Trivial change — snap immediately
      tracked.currentBbox = { ...newBbox };
      tracked.bboxSettled = true;
    } else {
      // Begin morphing
      tracked.bboxSettled = false;
    }
  }

  /**
   * Advance all in-flight bbox morphs by dt seconds.
   * Uses exponential interpolation: each tick closes (morphFactor) of the
   * remaining gap, producing smooth deceleration into the target.
   */
  private _tickBboxMorphs(_dt: number): void {
    for (const [cellId, tracked] of this._tracked) {
      if (tracked.bboxSettled) continue;

      const cur = tracked.currentBbox;
      const tgt = tracked.targetBbox;

      // Exponential ease: new = cur + (tgt - cur) * factor
      // We use a fixed factor per tick (frame-rate independent to first order
      // since the backend growth rate is also per-epoch, not per-second).
      const f = this._morphFactor;

      const nx = cur.x + (tgt.x - cur.x) * f;
      const ny = cur.y + (tgt.y - cur.y) * f;
      const nw = cur.w + (tgt.w - cur.w) * f;
      const nh = cur.h + (tgt.h - cur.h) * f;

      // Check if we're close enough to snap
      const remaining = Math.max(
        Math.abs(tgt.x - nx),
        Math.abs(tgt.y - ny),
        Math.abs(tgt.w - nw),
        Math.abs(tgt.h - nh),
      );

      if (remaining < MORPH_SNAP_PX) {
        tracked.currentBbox = { ...tgt };
        tracked.bboxSettled = true;
      } else {
        tracked.currentBbox = { x: nx, y: ny, w: nw, h: nh };
      }

      if (this._onBboxMorph) {
        this._onBboxMorph(cellId, tracked.currentBbox);
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Channel 3: Convergence Quality Gate
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Smoothly ramp _quality toward _qualityTarget.
   * Uses critically-damped spring dynamics for natural feel.
   */
  private _tickQualityRamp(dt: number): void {
    if (Math.abs(this._quality - this._qualityTarget) < 0.001) {
      this._quality = this._qualityTarget;
      this._qualityVelocity = 0;
      return;
    }

    // Critically-damped spring: ω = 2π / rampDuration, ζ = 1
    const omega = (2 * Math.PI) / this._qualityRampDuration;
    const dampedOmega = omega; // ζ = 1 → critically damped

    const diff  = this._qualityTarget - this._quality;
    const accel = dampedOmega * dampedOmega * diff
                - 2 * dampedOmega * this._qualityVelocity;

    this._qualityVelocity += accel * dt;
    this._quality         += this._qualityVelocity * dt;

    // Clamp to [0, 1]
    this._quality = Math.max(0, Math.min(1, this._quality));

    if (this._onQualityChange) {
      this._onQualityChange(this._buildQualityState());
    }
  }

  private _buildQualityState(): QualityState {
    const threshold = this._convergenceThreshold;
    const progress  = threshold > 0
      ? Math.max(0, Math.min(1, 1 - this._maxDelta / threshold))
      : 0;

    return {
      level:     this._quality,
      converged: this._converged,
      maxDelta:  this._maxDelta,
      progress,
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Channel 5: Snapshot Rollback
  // ═══════════════════════════════════════════════════════════════════════

  private _processRollback(payload: RollbackPayload): void {
    const targetEpoch = payload.target_epoch;
    console.warn(
      `[ASTRO-EPOCH-VIS] rollback to epoch ${targetEpoch} ` +
      `(reason: ${payload.reason})`,
    );

    this._currentEpoch = targetEpoch;
    this._converged    = false;
    this._qualityTarget = QUALITY_DRAFT;

    // Try to restore from snapshot history
    const snapshot = this.getHistoricalSnapshot(targetEpoch);
    if (snapshot) {
      this._restoreFromSnapshot(snapshot);
    } else {
      // Snapshot evicted — fall back to fetching fresh data
      console.warn(
        `[ASTRO-EPOCH-VIS] snapshot for epoch ${targetEpoch} evicted, fetching fresh`,
      );
      this._fetchAndApplyCells(`rollback-epoch-${targetEpoch}`);
    }

    if (this._onRollback) {
      this._onRollback(targetEpoch, payload.reason);
    }

    if (this._onTimelineProgress) {
      this._onTimelineProgress(this.getTimeline());
    }
  }

  /**
   * Restore all tracked cells to a historical snapshot.
   * Cells present in the snapshot but not currently tracked are entered;
   * cells tracked but absent from the snapshot are exited.
   */
  private _restoreFromSnapshot(snapshot: EpochSnapshot): void {
    const snapshotIds = new Set(snapshot.cells.keys());
    const currentIds  = new Set(this._tracked.keys());

    // Exit cells not in the snapshot
    for (const cellId of currentIds) {
      if (!snapshotIds.has(cellId)) {
        this._tracked.delete(cellId);
        if (this._onCellExit) {
          this._onCellExit(cellId);
        }
      }
    }

    // Enter/update cells from the snapshot
    for (const [cellId, snap] of snapshot.cells) {
      const existing = this._tracked.get(cellId);

      if (existing) {
        // Update in place with immediate snap (rollback, not morph)
        existing.species       = snap.species;
        existing.targetBbox    = { ...snap.bbox };
        existing.currentBbox   = { ...snap.bbox };
        existing.opacity       = snap.opacity;
        existing.z             = snap.z;
        existing.speciesParams = { ...snap.speciesParams };
        existing.bboxSettled   = true;
        existing.lastEpoch     = snapshot.epoch;
      } else {
        // New cell (was in snapshot but not tracked)
        this._tracked.set(cellId, {
          species:       snap.species,
          targetBbox:    { ...snap.bbox },
          currentBbox:   { ...snap.bbox },
          opacity:       snap.opacity,
          z:             snap.z,
          speciesParams: { ...snap.speciesParams },
          bboxSettled:   true,
          lastEpoch:     snapshot.epoch,
        });
        if (this._onCellEnter) {
          this._onCellEnter(cellId, snap);
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Cell data fetching & reconciliation
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Fetch bulk /api/cells and reconcile tracked state.
   * Detects additions, removals, species changes, and bbox updates.
   */
  private _fetchAndApplyCells(reason: string): void {
    if (this._fetchingCells || this._destroyed) return;
    this._fetchingCells = true;

    console.debug(`[ASTRO-EPOCH-VIS] fetching /api/cells (${reason})`);

    fetch(CELLS_API_URL)
      .then(res => {
        if (!res.ok) throw new Error(`/api/cells returned ${res.status}`);
        return res.json() as Promise<CellApiDescriptor[]>;
      })
      .then(descriptors => {
        this._reconcileCells(descriptors);
      })
      .catch(err => {
        console.error('[ASTRO-EPOCH-VIS] Failed to fetch /api/cells:', err);
      })
      .finally(() => {
        this._fetchingCells = false;
      });
  }

  /**
   * Fetch epoch metadata from /api/epoch (non-critical, best-effort).
   */
  private _fetchEpochMeta(): void {
    fetch(EPOCH_API_URL)
      .then(res => {
        if (!res.ok) return null;
        return res.json();
      })
      .then(data => {
        if (!data) return;
        if (typeof data.current === 'number') {
          this._currentEpoch = data.current;
        }
        if (typeof data.max === 'number' && data.max > 0) {
          // Note: _maxEpochs is readonly; we use it only for logging
          console.debug(
            `[ASTRO-EPOCH-VIS] epoch meta: current=${data.current} max=${data.max} ` +
            `status=${data.status}`,
          );
        }
        if (typeof data.converged === 'boolean') {
          this._converged = data.converged;
        }
      })
      .catch(() => {
        // Non-critical — continue without metadata
      });
  }

  /**
   * Reconcile the full set of cell descriptors against tracked state.
   *
   * Three-way diff:
   *   1. Cells in descriptors but not tracked → enter (new cell appeared)
   *   2. Cells tracked but not in descriptors → exit (cell removed)
   *   3. Cells in both → update (check species change, bbox morph)
   */
  private _reconcileCells(descriptors: CellApiDescriptor[]): void {
    const incomingIds = new Set<string>();
    const snapshotCells = new Map<string, CellVisualSnapshot>();

    for (const desc of descriptors) {
      const cellId = desc.cell_id ?? desc.id ?? '';
      if (!cellId) continue;

      incomingIds.add(cellId);

      const bbox: VisualBbox = (desc.bbox as VisualBbox) ?? { x: 220, y: 40, w: 140, h: 50 };
      const species = desc.species ?? 'cil-code';
      const opacity = typeof desc.opacity === 'number' ? desc.opacity : 1.0;
      const z       = typeof desc.z === 'number' ? desc.z : 3;
      const speciesParams = (desc.species_params ?? {}) as Record<string, number>;

      const existing = this._tracked.get(cellId);

      if (existing) {
        // ── Update existing cell ─────────────────────────────────────
        // Channel 1: species change detection
        this._detectSpeciesChange(cellId, existing, species);

        // Channel 2: bbox morph target
        this._updateBboxTarget(existing, bbox);

        // Direct state updates (no interpolation needed)
        existing.opacity       = opacity;
        existing.z             = z;
        existing.speciesParams = speciesParams;
        existing.lastEpoch     = this._currentEpoch;
      } else {
        // ── New cell — enter ─────────────────────────────────────────
        const newCell: TrackedCell = {
          species,
          targetBbox:    { ...bbox },
          currentBbox:   { ...bbox },
          opacity,
          z,
          speciesParams,
          bboxSettled:   true,  // No morph on first appear
          lastEpoch:     this._currentEpoch,
        };
        this._tracked.set(cellId, newCell);

        console.debug(
          `[ASTRO-EPOCH-VIS] cell enter "${cellId}" species=${species} ` +
          `bbox=(${bbox.x},${bbox.y},${bbox.w},${bbox.h})`,
        );

        if (this._onCellEnter) {
          this._onCellEnter(cellId, {
            cellId, bbox: { ...bbox }, species, opacity, z, speciesParams,
          });
        }
      }

      // Snapshot for this cell
      snapshotCells.set(cellId, {
        cellId, bbox: { ...bbox }, species, opacity, z, speciesParams,
      });
    }

    // ── Detect removed cells ─────────────────────────────────────────
    for (const cellId of this._tracked.keys()) {
      if (!incomingIds.has(cellId)) {
        this._tracked.delete(cellId);
        console.debug(`[ASTRO-EPOCH-VIS] cell exit "${cellId}"`);
        if (this._onCellExit) {
          this._onCellExit(cellId);
        }
      }
    }

    // ── Save epoch snapshot ──────────────────────────────────────────
    const snapshot: EpochSnapshot = {
      epoch:     this._currentEpoch,
      converged: this._converged,
      maxDelta:  this._maxDelta,
      cells:     snapshotCells,
      timestamp: Date.now(),
    };

    this._pushSnapshot(snapshot);

    if (this._onEpochSnapshot) {
      this._onEpochSnapshot(snapshot);
    }
  }

  /**
   * Apply a single-cell fast-path update (from cell_params_updated SSE).
   * This avoids a full /api/cells fetch when only one cell changed.
   */
  private _applySingleCellUpdate(
    cellId: string,
    params: CellParamsPayload['params'],
  ): void {
    const existing = this._tracked.get(cellId);

    if (existing) {
      // Species change
      if (params.species) {
        this._detectSpeciesChange(cellId, existing, params.species);
      }

      // Bbox morph
      if (params.bbox) {
        this._updateBboxTarget(existing, params.bbox);
      }

      // Direct updates
      if (typeof params.opacity === 'number') {
        existing.opacity = params.opacity;
      }
      if (typeof params.z === 'number') {
        existing.z = params.z;
      }
      if (params.species_params) {
        existing.speciesParams = params.species_params;
      }

      existing.lastEpoch = this._currentEpoch;

      console.debug(
        `[ASTRO-EPOCH-VIS] cell_params_updated "${cellId}" ` +
        `species=${existing.species} settled=${existing.bboxSettled}`,
      );
    } else {
      // Cell not tracked yet — create a new entry
      const bbox = params.bbox ?? { x: 220, y: 40, w: 140, h: 50 };
      const species = params.species ?? 'cil-code';
      const opacity = typeof params.opacity === 'number' ? params.opacity : 1.0;
      const z       = typeof params.z === 'number' ? params.z : 3;
      const speciesParams = (params.species_params ?? {}) as Record<string, number>;

      this._tracked.set(cellId, {
        species,
        targetBbox:  { ...bbox },
        currentBbox: { ...bbox },
        opacity,
        z,
        speciesParams,
        bboxSettled: true,
        lastEpoch:   this._currentEpoch,
      });

      console.debug(
        `[ASTRO-EPOCH-VIS] cell_params_updated (new) "${cellId}" species=${species}`,
      );

      if (this._onCellEnter) {
        this._onCellEnter(cellId, {
          cellId, bbox: { ...bbox }, species, opacity, z, speciesParams,
        });
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Snapshot ring buffer
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Push a snapshot into the ring buffer, evicting the oldest if full.
   * Matches _MAX_HISTORY = 16 from snapshot_manager.py (M174).
   */
  private _pushSnapshot(snapshot: EpochSnapshot): void {
    // Avoid duplicating the same epoch
    const last = this._snapshots[this._snapshots.length - 1];
    if (last && last.epoch === snapshot.epoch) {
      // Replace in place (same epoch, newer data)
      this._snapshots[this._snapshots.length - 1] = snapshot;
      return;
    }

    this._snapshots.push(snapshot);

    // Evict oldest when exceeding ring buffer capacity
    while (this._snapshots.length > this._maxSnapshotHistory) {
      this._snapshots.shift();
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: compute visual diff between two epoch snapshots
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the visual diff between two epoch snapshots.
 * Returns lists of cells that entered, exited, changed species, or moved.
 * Useful for debugging and analytics.
 */
export function diffEpochSnapshots(
  before: EpochSnapshot,
  after:  EpochSnapshot,
): EpochVisualDiff {
  const entered:        string[] = [];
  const exited:         string[] = [];
  const speciesChanged: Array<{ cellId: string; from: string; to: string }> = [];
  const bboxMoved:      Array<{ cellId: string; delta: number }> = [];

  const beforeIds = new Set(before.cells.keys());
  const afterIds  = new Set(after.cells.keys());

  // Entered: in after but not before
  for (const id of afterIds) {
    if (!beforeIds.has(id)) {
      entered.push(id);
    }
  }

  // Exited: in before but not after
  for (const id of beforeIds) {
    if (!afterIds.has(id)) {
      exited.push(id);
    }
  }

  // Changed: in both
  for (const id of afterIds) {
    if (!beforeIds.has(id)) continue;

    const bCell = before.cells.get(id)!;
    const aCell = after.cells.get(id)!;

    // Species change
    if (bCell.species !== aCell.species) {
      speciesChanged.push({ cellId: id, from: bCell.species, to: aCell.species });
    }

    // Bbox movement
    const dx = Math.abs(aCell.bbox.x - bCell.bbox.x);
    const dy = Math.abs(aCell.bbox.y - bCell.bbox.y);
    const dw = Math.abs(aCell.bbox.w - bCell.bbox.w);
    const dh = Math.abs(aCell.bbox.h - bCell.bbox.h);
    const delta = Math.max(dx, dy, dw, dh);
    if (delta > MORPH_THRESHOLD_PX) {
      bboxMoved.push({ cellId: id, delta });
    }
  }

  return {
    epochBefore: before.epoch,
    epochAfter:  after.epoch,
    entered,
    exited,
    speciesChanged,
    bboxMoved,
  };
}

/** Result of comparing two epoch snapshots. */
export interface EpochVisualDiff {
  epochBefore:     number;
  epochAfter:      number;
  entered:         string[];
  exited:          string[];
  speciesChanged:  Array<{ cellId: string; from: string; to: string }>;
  bboxMoved:       Array<{ cellId: string; delta: number }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: build a quality config from QualityState for the render pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Map the abstract [0, 1] quality level to concrete render pipeline knobs.
 *
 * This produces a configuration object that can be spread into the
 * RenderCompositor or ATRenderPipeline options to scale post-processing
 * intensity with convergence progress.
 *
 * Mirrors the performance-budget tier concept but driven by convergence
 * rather than frame rate.
 */
export function qualityToRenderConfig(quality: QualityState): RenderQualityConfig {
  const t = quality.level;

  return {
    /** Bloom strength: 0.15 (draft) → 0.85 (final). */
    bloomStrength:     lerp(0.15, 0.85, t),
    /** Bloom iterations: 2 (draft) → 6 (final). */
    bloomIterations:   Math.round(lerp(2, 6, t)),
    /** Particle cap multiplier: 0.4 (draft) → 1.0 (final). */
    particleCapScale:  lerp(0.4, 1.0, t),
    /** Volumetric light intensity: 0.0 (draft) → 1.0 (final). */
    volumetricLight:   lerp(0.0, 1.0, smoothstep(0.5, 1.0, t)),
    /** LUT grade intensity: 0.3 (draft) → 1.0 (final). */
    lutGradeIntensity: lerp(0.3, 1.0, t),
    /** Water surface enabled (only at > 60% quality). */
    waterSurface:      t > 0.6,
    /** Atmosphere enabled (only at > 70% quality). */
    atmosphere:        t > 0.7,
    /** SDF evaluation precision: 'low' → 'high'. */
    sdfPrecision:      t < 0.5 ? 'low' : (t < 0.8 ? 'medium' : 'high'),
    /** Edge particle density multiplier. */
    edgeParticleScale: lerp(0.3, 1.0, t),
    /** Label opacity — text fades in as convergence approaches. */
    labelOpacity:      lerp(0.4, 1.0, smoothstep(0.3, 0.9, t)),
  };
}

/** Concrete render knobs derived from quality state. */
export interface RenderQualityConfig {
  bloomStrength:     number;
  bloomIterations:   number;
  particleCapScale:  number;
  volumetricLight:   number;
  lutGradeIntensity: number;
  waterSurface:      boolean;
  atmosphere:        boolean;
  sdfPrecision:      'low' | 'medium' | 'high';
  edgeParticleScale: number;
  labelOpacity:      number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
