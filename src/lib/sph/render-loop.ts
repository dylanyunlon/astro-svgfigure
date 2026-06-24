/**
 * render-loop.ts — M782: Render Loop
 *
 * Priority-based ticker + pubsub-driven frame lifecycle for the Cell world.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * This module provides a self-contained render loop scheduler that mirrors
 * PixiJS v8's `Ticker` priority model while integrating the cell-pubsub-loop
 * event system.  Every frame follows a deterministic dispatch order:
 *
 *   ┌──────────────────────────────────────────────────────────────────────┐
 *   │  requestAnimationFrame                                              │
 *   │        │                                                            │
 *   │    [RenderLoop._tick()]                                             │
 *   │        │                                                            │
 *   │    ├─ Compute elapsed time (minFPS / maxFPS clamp)                  │
 *   │    ├─ Flush pubsub inbox (coalesce cell state changes)              │
 *   │    └─ Call listeners in priority order                              │
 *   │        ├─ INTERACTION (50)  — pointer / drag events                 │
 *   │        ├─ HIGH        (25)  — physics accumulator, SPH sub-steps    │
 *   │        ├─ NORMAL       (0)  — gameplay: cell animations, emitters   │
 *   │        ├─ LOW        (-25)  — render pass (scene traverse + draw)   │
 *   │        └─ UTILITY    (-50)  — post-render: pixel readback, stats    │
 *   └──────────────────────────────────────────────────────────────────────┘
 *
 * Design goals:
 *
 *   1. **Deterministic ordering** — Listeners at the same priority run in
 *      insertion order.  Physics always precedes rendering; rendering always
 *      precedes post-render utilities.
 *
 *   2. **Fixed-timestep physics** — An internal accumulator feeds sub-steps
 *      to HIGH-priority listeners at a configurable fixed dt (default 120 Hz).
 *      Remaining listeners receive the real (clamped) frame delta.
 *
 *   3. **Pubsub coalescing** — Cell state mutations (position, species,
 *      QoS profile changes) arrive asynchronously via `postMessage`.
 *      The loop batches them into a per-frame inbox that is flushed once
 *      before the priority dispatch.  This prevents mid-frame tearing.
 *
 *   4. **Performance-adaptive** — A sliding-window FPS tracker drives
 *      automatic quality degradation (particle cap, iteration count) when
 *      frames drop below a configurable threshold, and gradual recovery
 *      when headroom returns.
 *
 *   5. **Pluggable** — External subsystems register via `add()` with a
 *      priority enum, receive a `TickerState` argument every frame, and
 *      can be removed or paused individually.
 *
 * Usage:
 *
 *   ```ts
 *   import { RenderLoop, UPDATE_PRIORITY } from '$lib/sph/render-loop';
 *
 *   const loop = new RenderLoop({ maxFPS: 60, fixedDt: 1 / 120 });
 *
 *   // Physics at HIGH priority (receives fixed sub-steps)
 *   loop.add((tick) => stepPhysics(tick.fixedDt), UPDATE_PRIORITY.HIGH);
 *
 *   // Animation at NORMAL priority
 *   loop.add((tick) => updateAnimations(tick.deltaTime));
 *
 *   // Render at LOW (default render slot)
 *   loop.add((tick) => renderer.render(stage), UPDATE_PRIORITY.LOW);
 *
 *   // Post-render readback
 *   loop.add((tick) => exportStats(tick), UPDATE_PRIORITY.UTILITY);
 *
 *   // Subscribe to cell pubsub events
 *   loop.onCellEvent('position', (cellId, data) => {
 *     cellManager.updatePosition(cellId, data.x, data.y);
 *   });
 *
 *   loop.start();
 *   // ...
 *   loop.stop();
 *   loop.destroy();
 *   ```
 *
 * Coordinate convention:
 *   World space: (0, 0) at top-left, X right, Y down (matching SPH domain).
 *
 * Research: xiaodi #M782 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Priority enum (mirrors PixiJS v8 UPDATE_PRIORITY)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Listener execution priority.  Higher numeric value = earlier execution.
 * Matches PixiJS v8 `UPDATE_PRIORITY` for interop with existing code.
 */
export const UPDATE_PRIORITY = {
  /** Pointer & drag events — first to run. */
  INTERACTION: 50,
  /** Physics accumulator, SPH sub-steps. */
  HIGH: 25,
  /** Gameplay: cell animations, emitters. Default for `add()`. */
  NORMAL: 0,
  /** Scene graph traverse + GPU draw. */
  LOW: -25,
  /** Post-render: pixel readback, DOM sync, stats export. */
  UTILITY: -50,
} as const;

export type UpdatePriority = (typeof UPDATE_PRIORITY)[keyof typeof UPDATE_PRIORITY];

// ─────────────────────────────────────────────────────────────────────────────
// Ticker state — passed to every listener each frame
// ─────────────────────────────────────────────────────────────────────────────

/** Per-frame state delivered to every ticker callback. */
export interface TickerState {
  /**
   * Dimensionless frame-rate multiplier, ~1.0 at 60 fps.
   * Scaled by `speed` and clamped by `minFPS`.
   * Use for simple per-frame multipliers: `sprite.rotation += 0.01 * tick.deltaTime`.
   */
  deltaTime: number;

  /**
   * Elapsed milliseconds since the previous frame.
   * Scaled by `speed` and clamped by `minFPS`.
   * Use for time-based calculations: `px += velocity * tick.deltaMS / 1000`.
   */
  deltaMS: number;

  /**
   * Raw elapsed milliseconds — not scaled by `speed`, not clamped.
   * Use only for profiling or wall-clock measurements.
   */
  elapsedMS: number;

  /**
   * Fixed-timestep delta in seconds.  Only meaningful for HIGH-priority
   * listeners during physics sub-step dispatch.  Other priorities receive
   * the same value but should prefer `deltaMS` or `deltaTime`.
   */
  fixedDt: number;

  /**
   * Number of fixed-timestep sub-steps executed this frame.
   * 0 if the accumulator hasn't banked enough time for a step.
   */
  subStepCount: number;

  /** Current speed multiplier (default 1). */
  speed: number;

  /** Frame index since loop start (monotonically increasing). */
  frame: number;

  /** Smoothed FPS over the sliding window. */
  fps: number;

  /** Monotonic timestamp (ms) of this frame (performance.now). */
  now: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Listener descriptor
// ─────────────────────────────────────────────────────────────────────────────

/** Callback signature for ticker listeners. */
export type TickerCallback = (tick: TickerState) => void;

interface ListenerEntry {
  fn: TickerCallback;
  priority: UpdatePriority;
  /** Unique id for removal. */
  id: number;
  /** When true the listener is skipped but not removed. */
  paused: boolean;
  /** Optional label for debug/profiling. */
  label?: string;
}

/** Handle returned by `add()` for controlling or removing a listener. */
export interface ListenerHandle {
  /** Unique listener id. */
  readonly id: number;
  /** Pause this listener (skipped each frame, but not removed). */
  pause(): void;
  /** Resume a paused listener. */
  resume(): void;
  /** Remove the listener entirely. */
  remove(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell pubsub event types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Known cell event kinds that can be published through the render loop's
 * pubsub inbox.  Each event is coalesced per-cell per-frame: only the latest
 * value survives to the flush phase.
 */
export type CellEventKind =
  | 'position'
  | 'velocity'
  | 'species'
  | 'qos'
  | 'charge'
  | 'radius'
  | 'destroy'
  | 'spawn';

/** Payload for a single cell pubsub event. */
export interface CellEvent<K extends CellEventKind = CellEventKind> {
  kind: K;
  cellId: string;
  /** Arbitrary payload — shape depends on `kind`. */
  data: Record<string, unknown>;
  /** Timestamp when the event was posted (performance.now). */
  postedAt: number;
}

/** Handler registered via `onCellEvent`. */
export type CellEventHandler<K extends CellEventKind = CellEventKind> = (
  cellId: string,
  data: Record<string, unknown>,
  tick: TickerState,
) => void;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** RenderLoop configuration. */
export interface RenderLoopConfig {
  /**
   * Maximum frames per second.  The loop will skip frames to stay at or
   * below this ceiling.  0 = uncapped (vsync-limited).
   * @default 0
   */
  maxFPS: number;

  /**
   * Minimum assumed FPS for delta clamping.  When real FPS drops below
   * this, `deltaTime` and `deltaMS` are capped as if the frame rate were
   * `minFPS`.  Prevents enormous deltas that break physics.
   * @default 10
   */
  minFPS: number;

  /**
   * Fixed physics timestep in seconds.  HIGH-priority listeners receive
   * this as `tick.fixedDt`.  The accumulator may run 0–N sub-steps per
   * frame depending on real frame time.
   * @default 1/120  (120 Hz)
   */
  fixedDt: number;

  /**
   * Maximum real-time seconds the physics accumulator can bank before
   * clamping.  Prevents the "spiral of death" when frames stall.
   * @default fixedDt * 8
   */
  maxAccumulator: number;

  /**
   * Global speed multiplier applied to `deltaTime` and `deltaMS`.
   * Does not affect `elapsedMS`.
   * @default 1
   */
  speed: number;

  /**
   * FPS sliding-window length (number of frames).
   * @default 60
   */
  fpsWindowSize: number;

  /**
   * Below this smoothed FPS, the loop emits 'fps-low' events and
   * listeners can degrade quality.
   * @default 30
   */
  fpsLowThreshold: number;

  /**
   * Above this smoothed FPS, the loop emits 'fps-recovered' events.
   * @default 55
   */
  fpsHighThreshold: number;

  /**
   * When true, the loop auto-pauses on `document.visibilitychange`
   * (hidden) and resumes when visible again.
   * @default true
   */
  autoPauseOnBlur: boolean;
}

const DEFAULT_CONFIG: RenderLoopConfig = {
  maxFPS: 0,
  minFPS: 10,
  fixedDt: 1 / 120,
  maxAccumulator: (1 / 120) * 8,
  speed: 1,
  fpsWindowSize: 60,
  fpsLowThreshold: 30,
  fpsHighThreshold: 55,
  autoPauseOnBlur: true,
};

// ─────────────────────────────────────────────────────────────────────────────
// Performance monitor — sliding-window FPS tracker
// ─────────────────────────────────────────────────────────────────────────────

class FPSMonitor {
  private _samples: Float64Array;
  private _head = 0;
  private _count = 0;
  private _sum = 0;

  constructor(windowSize: number) {
    this._samples = new Float64Array(windowSize);
  }

  push(dtMS: number): void {
    const idx = this._head % this._samples.length;
    if (this._count >= this._samples.length) {
      this._sum -= this._samples[idx];
    } else {
      this._count++;
    }
    this._samples[idx] = dtMS;
    this._sum += dtMS;
    this._head++;
  }

  get fps(): number {
    if (this._count === 0) return 60;
    const avgMS = this._sum / this._count;
    return avgMS > 0 ? 1000 / avgMS : 60;
  }

  reset(): void {
    this._head = 0;
    this._count = 0;
    this._sum = 0;
    this._samples.fill(0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pubsub inbox — per-frame coalescing buffer
// ─────────────────────────────────────────────────────────────────────────────

class CellPubSubInbox {
  /**
   * Pending events indexed by `${kind}:${cellId}`.  Only the latest event
   * per key survives — this is the coalescing mechanism that prevents
   * mid-frame tearing when multiple updates arrive between frames.
   */
  private _pending = new Map<string, CellEvent>();

  /** Handlers keyed by event kind. */
  private _handlers = new Map<CellEventKind, CellEventHandler[]>();

  /** Post an event into the inbox.  Overwrites any prior event with the same kind+cellId. */
  post(event: CellEvent): void {
    const key = `${event.kind}:${event.cellId}`;
    this._pending.set(key, event);
  }

  /** Register a handler for a specific event kind. Returns an unsubscribe function. */
  on<K extends CellEventKind>(kind: K, handler: CellEventHandler<K>): () => void {
    let list = this._handlers.get(kind);
    if (!list) {
      list = [];
      this._handlers.set(kind, list);
    }
    list.push(handler as CellEventHandler);

    return () => {
      const arr = this._handlers.get(kind);
      if (!arr) return;
      const idx = arr.indexOf(handler as CellEventHandler);
      if (idx >= 0) arr.splice(idx, 1);
    };
  }

  /**
   * Flush all pending events — invoke handlers for each coalesced event.
   * Called once per frame, before priority dispatch.
   */
  flush(tick: TickerState): void {
    if (this._pending.size === 0) return;

    for (const event of this._pending.values()) {
      const handlers = this._handlers.get(event.kind);
      if (!handlers || handlers.length === 0) continue;
      for (let i = 0; i < handlers.length; i++) {
        handlers[i](event.cellId, event.data, tick);
      }
    }

    this._pending.clear();
  }

  /** Number of pending (unflushed) events. */
  get pendingCount(): number {
    return this._pending.size;
  }

  /** Clear all pending events without invoking handlers. */
  clear(): void {
    this._pending.clear();
  }

  /** Remove all handlers. */
  removeAllHandlers(): void {
    this._handlers.clear();
  }

  destroy(): void {
    this.clear();
    this.removeAllHandlers();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderLoop
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Priority-based ticker with pubsub-driven cell event coalescing.
 *
 * Mirrors PixiJS v8 `Ticker` priority model:
 *   INTERACTION (50) → HIGH (25) → NORMAL (0) → LOW (-25) → UTILITY (-50)
 *
 * Adds:
 *   • Fixed-timestep physics accumulator for HIGH-priority listeners
 *   • Cell pubsub inbox with per-frame coalescing
 *   • Sliding-window FPS monitoring with adaptive quality hooks
 *   • Auto-pause on page visibility change
 */
export class RenderLoop {
  // ── Configuration ─────────────────────────────────────────────────────────
  private _config: RenderLoopConfig;

  // ── Listeners ─────────────────────────────────────────────────────────────
  private _listeners: ListenerEntry[] = [];
  private _listenersDirty = false;
  private _nextId = 1;

  // ── Loop state ────────────────────────────────────────────────────────────
  private _running = false;
  private _rafId = 0;
  private _lastTime = -1;
  private _frame = 0;
  private _accumulator = 0;

  // ── FPS monitoring ────────────────────────────────────────────────────────
  private _fpsMonitor: FPSMonitor;
  private _fpsState: 'normal' | 'low' = 'normal';

  // ── Pubsub ────────────────────────────────────────────────────────────────
  private _inbox: CellPubSubInbox;

  // ── FPS-adaptive callbacks ────────────────────────────────────────────────
  private _onFPSLow: Array<(fps: number) => void> = [];
  private _onFPSRecovered: Array<(fps: number) => void> = [];

  // ── Visibility change ─────────────────────────────────────────────────────
  private _visibilityHandler: (() => void) | null = null;
  private _wasRunningBeforeBlur = false;

  // ── Current tick state (reused object to avoid GC pressure) ───────────────
  private _tickState: TickerState = {
    deltaTime: 0,
    deltaMS: 0,
    elapsedMS: 0,
    fixedDt: 0,
    subStepCount: 0,
    speed: 1,
    frame: 0,
    fps: 60,
    now: 0,
  };

  // ────────────────────────────────────────────────────────────────────────────

  constructor(config: Partial<RenderLoopConfig> = {}) {
    this._config = { ...DEFAULT_CONFIG, ...config };
    if (this._config.maxAccumulator <= 0) {
      this._config.maxAccumulator = this._config.fixedDt * 8;
    }
    this._fpsMonitor = new FPSMonitor(this._config.fpsWindowSize);
    this._inbox = new CellPubSubInbox();

    // Bind tick so it can be passed directly to rAF
    this._tick = this._tick.bind(this);
  }

  // ── Public API: listener management ───────────────────────────────────────

  /**
   * Register a callback at the given priority.
   * Returns a handle for pausing, resuming, or removing the listener.
   *
   * @param fn       Callback invoked each frame with a `TickerState` argument.
   * @param priority Execution priority (default `UPDATE_PRIORITY.NORMAL`).
   * @param label    Optional label for debug/profiling.
   */
  add(
    fn: TickerCallback,
    priority: UpdatePriority = UPDATE_PRIORITY.NORMAL,
    label?: string,
  ): ListenerHandle {
    const id = this._nextId++;
    const entry: ListenerEntry = { fn, priority, id, paused: false, label };
    this._listeners.push(entry);
    this._listenersDirty = true;

    return {
      id,
      pause: () => { entry.paused = true; },
      resume: () => { entry.paused = false; },
      remove: () => this._removeById(id),
    };
  }

  /** Remove a listener by its handle id. */
  private _removeById(id: number): void {
    const idx = this._listeners.findIndex((e) => e.id === id);
    if (idx >= 0) {
      this._listeners.splice(idx, 1);
    }
  }

  /** Remove all listeners at a specific priority, or all if no priority given. */
  removeAll(priority?: UpdatePriority): void {
    if (priority === undefined) {
      this._listeners.length = 0;
    } else {
      this._listeners = this._listeners.filter((e) => e.priority !== priority);
    }
  }

  // ── Public API: cell pubsub ───────────────────────────────────────────────

  /**
   * Post a cell event into the per-frame inbox.
   * The event will be coalesced (last-write-wins per kind+cellId) and
   * delivered to handlers during the next frame's flush phase.
   */
  postCellEvent(kind: CellEventKind, cellId: string, data: Record<string, unknown> = {}): void {
    this._inbox.post({
      kind,
      cellId,
      data,
      postedAt: typeof performance !== 'undefined' ? performance.now() : Date.now(),
    });
  }

  /**
   * Register a handler for a specific cell event kind.
   * Returns an unsubscribe function.
   *
   * Handlers are invoked once per frame during the inbox flush phase,
   * before the priority dispatch.  Each handler receives the coalesced
   * (latest) data for that cell+kind combination.
   */
  onCellEvent<K extends CellEventKind>(kind: K, handler: CellEventHandler<K>): () => void {
    return this._inbox.on(kind, handler);
  }

  /** Number of pending (unflushed) cell events. */
  get pendingCellEvents(): number {
    return this._inbox.pendingCount;
  }

  // ── Public API: FPS-adaptive hooks ────────────────────────────────────────

  /**
   * Register a callback invoked when smoothed FPS drops below `fpsLowThreshold`.
   * Use this to degrade particle count, reduce iteration counts, etc.
   */
  onFPSLow(fn: (fps: number) => void): () => void {
    this._onFPSLow.push(fn);
    return () => {
      const idx = this._onFPSLow.indexOf(fn);
      if (idx >= 0) this._onFPSLow.splice(idx, 1);
    };
  }

  /**
   * Register a callback invoked when smoothed FPS recovers above `fpsHighThreshold`.
   * Use this to restore quality toward nominal settings.
   */
  onFPSRecovered(fn: (fps: number) => void): () => void {
    this._onFPSRecovered.push(fn);
    return () => {
      const idx = this._onFPSRecovered.indexOf(fn);
      if (idx >= 0) this._onFPSRecovered.splice(idx, 1);
    };
  }

  // ── Public API: lifecycle ─────────────────────────────────────────────────

  /** Start the render loop.  Idempotent. */
  start(): void {
    if (this._running) return;
    this._running = true;
    this._lastTime = -1;
    this._accumulator = 0;
    this._fpsMonitor.reset();
    this._fpsState = 'normal';

    if (this._config.autoPauseOnBlur && typeof document !== 'undefined') {
      this._visibilityHandler = () => {
        if (document.hidden) {
          this._wasRunningBeforeBlur = this._running;
          if (this._running) this._pause();
        } else if (this._wasRunningBeforeBlur) {
          this._resume();
        }
      };
      document.addEventListener('visibilitychange', this._visibilityHandler);
    }

    this._rafId = requestAnimationFrame(this._tick);
  }

  /** Stop the render loop.  Idempotent. */
  stop(): void {
    this._pause();
    this._removeVisibilityListener();
  }

  /** Whether the loop is currently running. */
  get running(): boolean {
    return this._running;
  }

  /** Current smoothed FPS. */
  get fps(): number {
    return this._fpsMonitor.fps;
  }

  /** Current frame index. */
  get frame(): number {
    return this._frame;
  }

  /** Mutable speed multiplier. */
  get speed(): number {
    return this._config.speed;
  }
  set speed(v: number) {
    this._config.speed = Math.max(0, v);
  }

  /** Mutable maxFPS ceiling (0 = uncapped). */
  get maxFPS(): number {
    return this._config.maxFPS;
  }
  set maxFPS(v: number) {
    this._config.maxFPS = Math.max(0, v);
  }

  /** Mutable minFPS clamp floor. */
  get minFPS(): number {
    return this._config.minFPS;
  }
  set minFPS(v: number) {
    this._config.minFPS = Math.max(1, v);
  }

  /** Full teardown — stops the loop, removes all listeners and handlers. */
  destroy(): void {
    this.stop();
    this.removeAll();
    this._inbox.destroy();
    this._onFPSLow.length = 0;
    this._onFPSRecovered.length = 0;
  }

  // ── Internal: pause / resume (used by visibility handler) ─────────────────

  private _pause(): void {
    this._running = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = 0;
    }
  }

  private _resume(): void {
    if (this._running) return;
    this._running = true;
    this._lastTime = -1; // reset so next tick doesn't produce a huge delta
    this._rafId = requestAnimationFrame(this._tick);
  }

  private _removeVisibilityListener(): void {
    if (this._visibilityHandler && typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._visibilityHandler);
      this._visibilityHandler = null;
    }
  }

  // ── Internal: sort listeners by priority (descending) ─────────────────────

  private _sortListeners(): void {
    if (!this._listenersDirty) return;
    // Sort descending by priority (higher runs first).
    // Stable sort preserves insertion order within the same priority.
    this._listeners.sort((a, b) => b.priority - a.priority);
    this._listenersDirty = false;
  }

  // ── Internal: the core frame tick ─────────────────────────────────────────

  private _tick(now: number): void {
    if (!this._running) return;

    // ── Time delta computation ──────────────────────────────────────────────

    if (this._lastTime < 0) {
      this._lastTime = now;
      this._rafId = requestAnimationFrame(this._tick);
      return;
    }

    const rawDtMS = now - this._lastTime;

    // maxFPS gating: skip this frame if we're ahead of the budget
    if (this._config.maxFPS > 0) {
      const minFrameMS = 1000 / this._config.maxFPS;
      if (rawDtMS < minFrameMS) {
        this._rafId = requestAnimationFrame(this._tick);
        return;
      }
    }

    this._lastTime = now;

    // minFPS clamping: cap delta so physics don't explode on frame drops
    const maxDtMS = 1000 / this._config.minFPS;
    const clampedDtMS = Math.min(rawDtMS, maxDtMS);

    // Scaled deltas
    const speed = this._config.speed;
    const scaledDtMS = clampedDtMS * speed;
    const deltaTime = (scaledDtMS / 1000) * 60; // dimensionless, ~1.0 at 60fps

    // FPS monitoring
    this._fpsMonitor.push(rawDtMS);
    const currentFPS = this._fpsMonitor.fps;

    // FPS-adaptive state transitions
    this._checkFPSThresholds(currentFPS);

    // ── Fixed-timestep accumulator for physics ──────────────────────────────

    const fixedDt = this._config.fixedDt;
    this._accumulator += scaledDtMS / 1000;
    if (this._accumulator > this._config.maxAccumulator) {
      this._accumulator = this._config.maxAccumulator;
    }

    let subStepCount = 0;
    while (this._accumulator >= fixedDt) {
      this._accumulator -= fixedDt;
      subStepCount++;
    }

    // ── Build tick state (reuse object to avoid GC) ─────────────────────────

    const tick = this._tickState;
    tick.deltaTime = deltaTime;
    tick.deltaMS = scaledDtMS;
    tick.elapsedMS = rawDtMS;
    tick.fixedDt = fixedDt;
    tick.subStepCount = subStepCount;
    tick.speed = speed;
    tick.frame = this._frame;
    tick.fps = currentFPS;
    tick.now = now;

    // ── Phase 1: Flush pubsub inbox ─────────────────────────────────────────

    this._inbox.flush(tick);

    // ── Phase 2: Dispatch listeners in priority order ───────────────────────

    this._sortListeners();

    const listeners = this._listeners;
    for (let i = 0; i < listeners.length; i++) {
      const entry = listeners[i];
      if (entry.paused) continue;

      if (entry.priority === UPDATE_PRIORITY.HIGH && subStepCount > 0) {
        // HIGH-priority listeners receive one call per physics sub-step
        for (let s = 0; s < subStepCount; s++) {
          entry.fn(tick);
        }
      } else {
        entry.fn(tick);
      }
    }

    // ── Advance frame counter ───────────────────────────────────────────────

    this._frame++;

    // ── Schedule next frame ─────────────────────────────────────────────────

    this._rafId = requestAnimationFrame(this._tick);
  }

  // ── Internal: FPS threshold checks ────────────────────────────────────────

  private _checkFPSThresholds(fps: number): void {
    if (this._fpsState === 'normal' && fps < this._config.fpsLowThreshold) {
      this._fpsState = 'low';
      for (let i = 0; i < this._onFPSLow.length; i++) {
        this._onFPSLow[i](fps);
      }
    } else if (this._fpsState === 'low' && fps > this._config.fpsHighThreshold) {
      this._fpsState = 'normal';
      for (let i = 0; i < this._onFPSRecovered.length; i++) {
        this._onFPSRecovered[i](fps);
      }
    }
  }

  // ── Debug / introspection ─────────────────────────────────────────────────

  /** Snapshot of all registered listeners (for dev tools). */
  listenerSnapshot(): Array<{
    id: number;
    priority: UpdatePriority;
    paused: boolean;
    label?: string;
  }> {
    this._sortListeners();
    return this._listeners.map((e) => ({
      id: e.id,
      priority: e.priority,
      paused: e.paused,
      label: e.label,
    }));
  }

  /** Current configuration (read-only copy). */
  get config(): Readonly<RenderLoopConfig> {
    return { ...this._config };
  }

  /**
   * Factory — create and optionally auto-start a RenderLoop.
   */
  static create(config?: Partial<RenderLoopConfig>, autoStart = false): RenderLoop {
    const loop = new RenderLoop(config);
    if (autoStart) loop.start();
    return loop;
  }
}

// ─── Barrel exports ─────────────────────────────────────────────────────────

export {
  FPSMonitor,
  CellPubSubInbox,
  DEFAULT_CONFIG as RENDER_LOOP_DEFAULTS,
};


