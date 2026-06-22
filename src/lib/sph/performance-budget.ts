/**
 * performance-budget.ts
 * 4-tier adaptive performance budget: ULTRA / HIGH / MEDIUM / LOW.
 *
 * Rules
 * ─────
 *  • FPS < 45  → downgrade one tier (after hysteresis frames)
 *  • FPS > 58  → upgrade one tier   (after hysteresis frames)
 *  • Tier order: ULTRA > HIGH > MEDIUM > LOW
 *
 * Usage
 * ─────
 *   const budget = new PerformanceBudget();
 *   // inside rAF loop:
 *   budget.tick(deltaMs);
 *   const cfg = budget.config;   // use cfg.maxParticles, cfg.substeps, …
 */

// ---------------------------------------------------------------------------
// Tier definitions
// ---------------------------------------------------------------------------

export type Tier = 'ULTRA' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface TierConfig {
  tier: Tier;
  /** Maximum active particles in the simulation. */
  maxParticles: number;
  /** Physics substeps per rendered frame. */
  substeps: number;
  /** Neighbour-search radius multiplier (1 = nominal). */
  neighborRadiusScale: number;
  /** Enable surface-tension micro-forces. */
  surfaceTension: boolean;
  /** Enable high-quality PBR shading. */
  hqShading: boolean;
  /** Particle render size multiplier. */
  particleScale: number;
  /** Post-process passes enabled (bloom, caustics, …). */
  postProcessPasses: number;
  /** Shadow / ambient-occlusion quality (0 = off, 1 = low, 2 = high). */
  shadowQuality: 0 | 1 | 2;
}

const TIER_CONFIGS: Record<Tier, TierConfig> = {
  ULTRA: {
    tier:               'ULTRA',
    maxParticles:       50_000,
    substeps:           4,
    neighborRadiusScale:1.0,
    surfaceTension:     true,
    hqShading:          true,
    particleScale:      1.0,
    postProcessPasses:  3,
    shadowQuality:      2,
  },
  HIGH: {
    tier:               'HIGH',
    maxParticles:       30_000,
    substeps:           3,
    neighborRadiusScale:1.0,
    surfaceTension:     true,
    hqShading:          true,
    particleScale:      1.0,
    postProcessPasses:  2,
    shadowQuality:      1,
  },
  MEDIUM: {
    tier:               'MEDIUM',
    maxParticles:       15_000,
    substeps:           2,
    neighborRadiusScale:0.9,
    surfaceTension:     false,
    hqShading:          false,
    particleScale:      1.1,
    postProcessPasses:  1,
    shadowQuality:      0,
  },
  LOW: {
    tier:               'LOW',
    maxParticles:       6_000,
    substeps:           1,
    neighborRadiusScale:0.8,
    surfaceTension:     false,
    hqShading:          false,
    particleScale:      1.3,
    postProcessPasses:  0,
    shadowQuality:      0,
  },
};

const TIER_ORDER: Tier[] = ['LOW', 'MEDIUM', 'HIGH', 'ULTRA'];

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

/** Drop a tier if rolling FPS stays below this. */
const FPS_DROP_THRESHOLD  = 45;
/** Raise a tier if rolling FPS stays above this. */
const FPS_RAISE_THRESHOLD = 58;

/**
 * How many consecutive out-of-band samples before we act.
 * Prevents thrashing on momentary spikes / dips.
 */
const HYSTERESIS_FRAMES = 90;   // ~1.5 s at 60 fps

/** Number of frames used for the rolling average. */
const ROLLING_WINDOW = 30;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export type TierChangeHandler = (next: Tier, prev: Tier, budget: PerformanceBudget) => void;

// ---------------------------------------------------------------------------
// PerformanceBudget
// ---------------------------------------------------------------------------

export class PerformanceBudget {
  // ── public state ──────────────────────────────────────────────────────────

  /** Current active tier. */
  get tier(): Tier { return this._tier; }

  /** Frozen config object for the current tier. */
  get config(): Readonly<TierConfig> { return TIER_CONFIGS[this._tier]; }

  /** Instantaneous rolling-average FPS. */
  get fps(): number { return this._fps; }

  /** Total frames ticked since construction. */
  get frameCount(): number { return this._frameCount; }

  // ── private state ─────────────────────────────────────────────────────────

  private _tier: Tier;
  private _fps: number = 60;
  private _frameCount: number = 0;

  /** Circular buffer of recent frame durations (ms). */
  private readonly _frameTimes: Float32Array;
  private _ftHead: number = 0;
  private _ftFilled: number = 0;

  /** Frames continuously below / above threshold. */
  private _dropStreak:  number = 0;
  private _raiseStreak: number = 0;

  private readonly _listeners: Set<TierChangeHandler> = new Set();

  // ── constructor ───────────────────────────────────────────────────────────

  /**
   * @param initialTier  Starting tier (default HIGH).
   */
  constructor(initialTier: Tier = 'HIGH') {
    this._tier = initialTier;
    this._frameTimes = new Float32Array(ROLLING_WINDOW);
  }

  // ── public API ────────────────────────────────────────────────────────────

  /**
   * Call once per animation frame with the elapsed time since last frame.
   * @param deltaMs  Frame duration in milliseconds.
   */
  tick(deltaMs: number): void {
    // Clamp to sane range to avoid outlier pollution (tab switch, etc.)
    const clamped = Math.min(Math.max(deltaMs, 1), 200);

    // Update rolling buffer
    this._frameTimes[this._ftHead] = clamped;
    this._ftHead = (this._ftHead + 1) % ROLLING_WINDOW;
    if (this._ftFilled < ROLLING_WINDOW) this._ftFilled++;

    // Compute rolling average FPS
    let sum = 0;
    for (let i = 0; i < this._ftFilled; i++) sum += this._frameTimes[i];
    const avgDelta = sum / this._ftFilled;
    this._fps = 1000 / avgDelta;

    this._frameCount++;

    // ── Hysteresis logic ───────────────────────────────────────────────────
    if (this._fps < FPS_DROP_THRESHOLD) {
      this._dropStreak++;
      this._raiseStreak = 0;
    } else if (this._fps > FPS_RAISE_THRESHOLD) {
      this._raiseStreak++;
      this._dropStreak = 0;
    } else {
      // In the comfortable band — reset both streaks
      this._dropStreak  = 0;
      this._raiseStreak = 0;
    }

    if (this._dropStreak >= HYSTERESIS_FRAMES) {
      this._dropStreak = 0;
      this._shiftTier(-1);
    } else if (this._raiseStreak >= HYSTERESIS_FRAMES) {
      this._raiseStreak = 0;
      this._shiftTier(+1);
    }
  }

  /**
   * Force an immediate tier change (e.g. on user preference or device hint).
   */
  setTier(tier: Tier): void {
    if (tier === this._tier) return;
    const prev = this._tier;
    this._tier = tier;
    this._resetStreaks();
    this._emit(tier, prev);
  }

  /**
   * Register a callback fired whenever the tier changes.
   * Returns an unsubscribe function.
   */
  onTierChange(handler: TierChangeHandler): () => void {
    this._listeners.add(handler);
    return () => this._listeners.delete(handler);
  }

  /**
   * Return a plain snapshot for debugging / serialisation.
   */
  snapshot(): {
    tier: Tier;
    fps: number;
    frameCount: number;
    dropStreak: number;
    raiseStreak: number;
    config: TierConfig;
  } {
    return {
      tier:        this._tier,
      fps:         parseFloat(this._fps.toFixed(2)),
      frameCount:  this._frameCount,
      dropStreak:  this._dropStreak,
      raiseStreak: this._raiseStreak,
      config:      TIER_CONFIGS[this._tier],
    };
  }

  // ── static helpers ────────────────────────────────────────────────────────

  /** Return the config for a given tier without instantiating a budget. */
  static configFor(tier: Tier): Readonly<TierConfig> {
    return TIER_CONFIGS[tier];
  }

  /** Ordered list of tiers from lowest to highest. */
  static get tiers(): readonly Tier[] {
    return TIER_ORDER;
  }

  // ── private helpers ───────────────────────────────────────────────────────

  private _shiftTier(delta: -1 | 1): void {
    const idx     = TIER_ORDER.indexOf(this._tier);
    const nextIdx = Math.min(Math.max(idx + delta, 0), TIER_ORDER.length - 1);
    if (nextIdx === idx) return; // already at boundary
    const prev = this._tier;
    this._tier = TIER_ORDER[nextIdx];
    this._emit(this._tier, prev);
  }

  private _resetStreaks(): void {
    this._dropStreak  = 0;
    this._raiseStreak = 0;
  }

  private _emit(next: Tier, prev: Tier): void {
    for (const fn of this._listeners) {
      try { fn(next, prev, this); } catch { /* ignore listener errors */ }
    }
  }
}

// ---------------------------------------------------------------------------
// Convenience singleton (optional — import when a shared global is handy)
// ---------------------------------------------------------------------------

let _globalBudget: PerformanceBudget | null = null;

/** Lazily create / return the process-wide performance budget. */
export function getGlobalBudget(): PerformanceBudget {
  if (!_globalBudget) _globalBudget = new PerformanceBudget();
  return _globalBudget;
}

/** Replace the global budget (useful in tests). */
export function setGlobalBudget(b: PerformanceBudget): void {
  _globalBudget = b;
}
