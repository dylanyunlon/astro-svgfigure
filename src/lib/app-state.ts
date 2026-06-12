/**
 * app-state.ts — AppState FSM + World singleton
 *
 * AT AppState (142 refs) + World (global state container) port.
 * Implements the playground page's lifecycle:
 *   loading → idle → playing ↔ editing
 *
 * Architecture:
 *   AppState   finite-state machine with enter/exit/update hooks per state
 *              transition(from, to) drives an eased cross-fade animation
 *   World      singleton: current epoch / cells / viewport / zoom
 *              acts as the shared bus between AppState and FXScene
 *
 * Upstream references:
 *   upstream/theatre-js/studio/src/uiComponents/chordial/hoverActor.ts
 *   upstream/theatre-js/studio/src/uiComponents/chordial/popoverActor.ts
 *   upstream/thing-editor/thing-editor/src/engine/lib/assets/src/basic/scene.c.ts
 *   upstream/pixijs-engine/src/scene/layers/RenderLayer.ts
 */

// ── State names ─────────────────────────────────────────────────────────────

export type StateName = 'loading' | 'idle' | 'playing' | 'editing';

// ── State hook interface ─────────────────────────────────────────────────────

/**
 * Hooks attached to each FSM state.  All are optional — omit any you don't need.
 * AT's Stage calls enter/exit for page transitions and update every rAF tick
 * while the state is active.
 */
export interface StateHooks {
  /** Called once when entering this state (after the previous state's exit). */
  enter?: (from: StateName | null) => void | Promise<void>;
  /** Called once when leaving this state (before the next state's enter). */
  exit?: (to: StateName) => void | Promise<void>;
  /**
   * Called every animation frame while this state is active.
   * @param dt  delta-time in milliseconds since last frame
   */
  update?: (dt: number) => void;
}

// ── Transition animation easing ──────────────────────────────────────────────

/** Ease-in-out cubic, matching AT's default page-transition curve. */
function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Run a timed ease animation.
 * @param durationMs   total duration in milliseconds
 * @param onProgress   called each rAF tick with an eased 0→1 value
 * @returns            a cancel handle
 */
function runTransitionAnim(
  durationMs: number,
  onProgress: (easedT: number) => void,
): () => void {
  let start: number | null = null;
  let rafHandle: number | null = null;
  let cancelled = false;

  const tick = (now: number) => {
    if (cancelled) return;
    if (start === null) start = now;
    const raw = Math.min((now - start) / durationMs, 1);
    onProgress(easeInOutCubic(raw));
    if (raw < 1) {
      rafHandle = requestAnimationFrame(tick);
    } else {
      rafHandle = null;
    }
  };

  rafHandle = requestAnimationFrame(tick);

  return () => {
    cancelled = true;
    if (rafHandle !== null) {
      cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }
  };
}

// ── Valid transition table ───────────────────────────────────────────────────

/**
 * Allowed (from → to) pairs.  Attempting an invalid transition throws in
 * development and is silently dropped in production.
 */
const VALID_TRANSITIONS: ReadonlySet<string> = new Set<string>([
  'loading→idle',
  'idle→playing',
  'idle→editing',
  'playing→idle',
  'playing→editing',
  'editing→idle',
  'editing→playing',
]);

// ── AppState class ───────────────────────────────────────────────────────────

/**
 * AppState — finite-state machine for the playground page lifecycle.
 *
 * Usage:
 * ```ts
 * const fsm = new AppState()
 *
 * fsm.registerHooks('loading', {
 *   enter: async () => { await fetchEpochData() },
 *   update: (dt) => { progressBar.tick(dt) },
 * })
 *
 * fsm.registerHooks('idle', {
 *   enter: () => { showCanvas() },
 * })
 *
 * fsm.start('loading')
 * // … later:
 * await fsm.transition('loading', 'idle')
 * ```
 */
export class AppState {
  private _current: StateName | null = null;
  private _hooks: Map<StateName, StateHooks> = new Map();
  private _transitionInProgress = false;
  /** Duration of the cross-fade animation in ms (default: 300). */
  transitionDuration = 300;

  // rAF handle for the update loop
  private _rafHandle: number | null = null;
  private _lastTs: number | null = null;
  private _cancelAnim: (() => void) | null = null;

  // Listener registry for state-change events
  private _listeners: Array<(from: StateName | null, to: StateName) => void> = [];

  /** Register hooks for a state.  Can be called before or after start(). */
  registerHooks(state: StateName, hooks: StateHooks): void {
    this._hooks.set(state, hooks);
  }

  /** Current FSM state name, or null before start(). */
  get current(): StateName | null {
    return this._current;
  }

  /** True while a transition animation is running. */
  get transitioning(): boolean {
    return this._transitionInProgress;
  }

  /**
   * Start the FSM in the given initial state.
   * Calls enter() for the state and begins the update loop.
   */
  async start(initial: StateName): Promise<void> {
    if (this._current !== null) {
      console.warn(`[AppState] already started in state "${this._current}"`);
      return;
    }
    this._current = initial;
    const hooks = this._hooks.get(initial);
    if (hooks?.enter) await hooks.enter(null);
    this._startUpdateLoop();
  }

  /**
   * Transition from `from` to `to` with a cross-fade animation.
   *
   * The sequence is:
   *   1. Validate the (from → to) pair against VALID_TRANSITIONS.
   *   2. Run the transition animation (opacity 0→1 over transitionDuration ms).
   *      Progress is emitted via the `onTransitionProgress` callback if set.
   *   3. Call exit(to) on the current state's hooks.
   *   4. Swap _current to `to`.
   *   5. Call enter(from) on the new state's hooks.
   *   6. Notify all registered listeners.
   *
   * Awaiting the returned Promise gives you a point-in-time after step 5.
   */
  async transition(from: StateName, to: StateName): Promise<void> {
    const key = `${from}→${to}`;

    if (!VALID_TRANSITIONS.has(key)) {
      const msg = `[AppState] invalid transition "${key}"`;
      if (import.meta.env?.DEV) throw new Error(msg);
      console.warn(msg);
      return;
    }

    if (this._current !== from) {
      console.warn(
        `[AppState] transition("${from}", "${to}") called but current state is "${this._current}"`,
      );
      return;
    }

    if (this._transitionInProgress) {
      console.warn(`[AppState] transition already in progress, dropping "${key}"`);
      return;
    }

    this._transitionInProgress = true;

    // Cancel any leftover animation from a previous transition
    if (this._cancelAnim) {
      this._cancelAnim();
      this._cancelAnim = null;
    }

    // Run the ease animation
    await new Promise<void>((resolve) => {
      this._cancelAnim = runTransitionAnim(this.transitionDuration, (easedT) => {
        this.onTransitionProgress?.(easedT, from, to);
        if (easedT >= 1) resolve();
      });
    });

    // Hooks: exit old state
    const fromHooks = this._hooks.get(from);
    if (fromHooks?.exit) await fromHooks.exit(to);

    // Swap state
    this._current = to;

    // Hooks: enter new state
    const toHooks = this._hooks.get(to);
    if (toHooks?.enter) await toHooks.enter(from);

    this._transitionInProgress = false;

    // Notify listeners
    for (const cb of this._listeners) {
      cb(from, to);
    }
  }

  /**
   * Optional callback that receives the eased progress value (0→1) during
   * a transition animation, plus the from/to state names.  Use this to drive
   * e.g. a loading overlay's opacity or a progress bar.
   */
  onTransitionProgress?: (easedT: number, from: StateName, to: StateName) => void;

  /** Register a listener for completed state changes.  Returns an unsubscribe fn. */
  onStateChange(cb: (from: StateName | null, to: StateName) => void): () => void {
    this._listeners.push(cb);
    return () => {
      this._listeners = this._listeners.filter((l) => l !== cb);
    };
  }

  /** Stop the FSM, cancel animations and the update loop.  Calls exit() on the current state. */
  async dispose(): Promise<void> {
    this._stopUpdateLoop();
    if (this._cancelAnim) {
      this._cancelAnim();
      this._cancelAnim = null;
    }
    if (this._current !== null) {
      const hooks = this._hooks.get(this._current);
      // Fake "to" for the final exit — use current as sentinel
      if (hooks?.exit) await hooks.exit(this._current);
    }
    this._current = null;
    this._listeners = [];
  }

  // ── Update loop ────────────────────────────────────────────────────────────

  private _startUpdateLoop(): void {
    if (this._rafHandle !== null) return;
    const tick = (now: number) => {
      const dt = this._lastTs === null ? 0 : now - this._lastTs;
      this._lastTs = now;

      if (this._current !== null) {
        const hooks = this._hooks.get(this._current);
        hooks?.update?.(dt);
      }

      this._rafHandle = requestAnimationFrame(tick);
    };
    this._rafHandle = requestAnimationFrame(tick);
  }

  private _stopUpdateLoop(): void {
    if (this._rafHandle !== null) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = null;
    }
    this._lastTs = null;
  }
}

// ── Viewport descriptor ──────────────────────────────────────────────────────

export interface Viewport {
  /** Left edge in world-space units. */
  x: number;
  /** Top edge in world-space units. */
  y: number;
  /** Visible width in world-space units (= canvas.width / zoom). */
  width: number;
  /** Visible height in world-space units (= canvas.height / zoom). */
  height: number;
}

// ── Cell descriptor (World-level snapshot) ───────────────────────────────────

export interface WorldCell {
  cell_id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  opacity: number;
  species: string;
}

// ── World singleton ──────────────────────────────────────────────────────────

/**
 * World — the global state container, shared by AppState and FXScene.
 *
 * AT's World object held the "single source of truth" for what was visible
 * on screen.  Here it stores:
 *   - current epoch index
 *   - live cell descriptors (post-layout, pre-render)
 *   - viewport (world-space rect of what is currently on screen)
 *   - zoom level
 *
 * Singleton pattern: import `World` and use it directly — do not instantiate.
 *
 * ```ts
 * import { World } from './app-state'
 *
 * World.epoch = 3
 * World.cells = updatedCells
 * World.zoom  = 1.5
 * ```
 */
class WorldSingleton {
  /** Current epoch index (0-based). */
  epoch: number = 0;

  /** All cells in the current epoch, keyed order by z then cell_id. */
  cells: WorldCell[] = [];

  /**
   * Current viewport: the world-space rectangle visible on screen.
   * Updated by the renderer / camera controller on every pan/zoom.
   */
  viewport: Viewport = { x: 0, y: 0, width: 1920, height: 1080 };

  /** Camera zoom level (1 = 1 world-unit → 1 CSS pixel). */
  zoom: number = 1.0;

  // Change listeners ─────────────────────────────────────────────────────────

  private _epochListeners: Array<(epoch: number) => void> = [];
  private _cellListeners: Array<(cells: WorldCell[]) => void> = [];
  private _viewportListeners: Array<(vp: Viewport, zoom: number) => void> = [];

  /**
   * Atomically update the viewport and zoom, then notify listeners once.
   * Prefer this over setting .viewport and .zoom separately.
   */
  setViewport(vp: Viewport, zoom: number): void {
    this.viewport = vp;
    this.zoom = zoom;
    for (const cb of this._viewportListeners) cb(vp, zoom);
  }

  /** Update the epoch and notify listeners. */
  setEpoch(epoch: number): void {
    this.epoch = epoch;
    for (const cb of this._epochListeners) cb(epoch);
  }

  /** Replace the cell list and notify listeners. */
  setCells(cells: WorldCell[]): void {
    this.cells = cells;
    for (const cb of this._cellListeners) cb(cells);
  }

  /** Subscribe to epoch changes.  Returns unsubscribe fn. */
  onEpochChange(cb: (epoch: number) => void): () => void {
    this._epochListeners.push(cb);
    return () => { this._epochListeners = this._epochListeners.filter((l) => l !== cb); };
  }

  /** Subscribe to cell list changes.  Returns unsubscribe fn. */
  onCellsChange(cb: (cells: WorldCell[]) => void): () => void {
    this._cellListeners.push(cb);
    return () => { this._cellListeners = this._cellListeners.filter((l) => l !== cb); };
  }

  /** Subscribe to viewport / zoom changes.  Returns unsubscribe fn. */
  onViewportChange(cb: (vp: Viewport, zoom: number) => void): () => void {
    this._viewportListeners.push(cb);
    return () => { this._viewportListeners = this._viewportListeners.filter((l) => l !== cb); };
  }

  /**
   * Returns true when a world-space rect (ax, ay, aw, ah) is at least
   * partially visible inside the current viewport.
   */
  isVisible(ax: number, ay: number, aw: number, ah: number): boolean {
    const { x, y, width, height } = this.viewport;
    return ax < x + width && ax + aw > x && ay < y + height && ay + ah > y;
  }

  /** Reset to construction defaults (useful in tests). */
  reset(): void {
    this.epoch = 0;
    this.cells = [];
    this.viewport = { x: 0, y: 0, width: 1920, height: 1080 };
    this.zoom = 1.0;
    this._epochListeners = [];
    this._cellListeners = [];
    this._viewportListeners = [];
  }
}

/** Global World singleton.  Import and use directly — never `new`. */
export const World = new WorldSingleton();

// ── Factory helper ───────────────────────────────────────────────────────────

/**
 * createPlaygroundFSM — wires up a pre-configured AppState for the playground
 * page with the canonical four states: loading → idle → playing ↔ editing.
 *
 * Callers override individual hooks after construction:
 * ```ts
 * const fsm = createPlaygroundFSM()
 * fsm.registerHooks('loading', { enter: async () => loadData() })
 * await fsm.start('loading')
 * ```
 */
export function createPlaygroundFSM(): AppState {
  const fsm = new AppState();

  const DEFAULT_HOOKS: Record<StateName, StateHooks> = {
    loading: {
      enter: () => { /* caller should override */ },
      exit: () => { /* caller should override */ },
      update: (_dt) => { /* tick progress bar or spinner */ },
    },
    idle: {
      enter: () => { /* canvas ready, show UI controls */ },
      exit: () => {},
      update: (_dt) => {},
    },
    playing: {
      enter: () => { /* start epoch timeline */ },
      exit: () => { /* pause timeline */ },
      update: (_dt) => { /* advance timeline position */ },
    },
    editing: {
      enter: () => { /* enable edit handles / drag targets */ },
      exit: () => { /* commit any pending edits */ },
      update: (_dt) => {},
    },
  };

  for (const [state, hooks] of Object.entries(DEFAULT_HOOKS) as [StateName, StateHooks][]) {
    fsm.registerHooks(state, hooks);
  }

  return fsm;
}
