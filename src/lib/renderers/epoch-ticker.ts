/**
 * epoch-ticker.ts — PixiJS Ticker 驱动 epoch 动画帧循环  (M016)
 *
 * 融合 upstream/pixijs-engine/src/ticker/ 的 Ticker + UPDATE_PRIORITY 到
 * epoch 动画循环中，替代原先 theatre-epoch-timeline.ts 使用的手工 onChange
 * 位置推进方式，将每帧回调的调度权完全交给 PixiJS Ticker。
 *
 * ─── 设计决策 ──────────────────────────────────────────────────────────────
 *
 *  Ticker 层次  (UPDATE_PRIORITY, 高→低先执行)
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  INTERACTION (50)  — (保留给 EventSystem, 不注册)               │
 *  │  HIGH       (25)   — physics tick (epochTicker HIGH 钩子)        │
 *  │  NORMAL      (0)   — 帧状态更新 + Theatre.js 位置推进            │
 *  │  LOW        (-25)  — PixiJS renderer.render()  (TickerPlugin)    │
 *  │  UTILITY    (-50)  — 后处理 / 度量收集                           │
 *  └──────────────────────────────────────────────────────────────────┘
 *
 *  Epoch 位置计算
 *  ──────────────
 *  每帧收到 ticker.deltaMS（毫秒，已由 ticker.speed 缩放）:
 *    position += (deltaMS / msPerEpoch) * playbackSpeed
 *  msPerEpoch 默认 = 1000ms（1 秒/epoch），可通过 setMsPerEpoch() 调整。
 *  到达最后一个 epoch 时根据 loopMode 决定：loop / bounce / stop。
 *
 *  与 EpochTimeline 的协作
 *  ──────────────────────
 *  EpochTimeline 在 onChange(sequence.position, …) 回调中已能驱动 PixiJS。
 *  EpochTicker 在 NORMAL 优先级更新 sequence.position，Theatre.js 本身
 *  的 bezier 插值立即生效，EpochTimeline._emitFrame() 在同一帧内被触发。
 *  这样实现了 Ticker ← Theatre.js ← PixiJS 三层解耦。
 *
 *  独立模式（无 Theatre.js）
 *  ─────────────────────────
 *  当不传入 sequenceRef 时，EpochTicker 自行做 epoch 之间的线性 lerp，
 *  直接从 CellState[][] snapshots 中插值，然后发射 EpochTickFrame。
 *
 * 上游参考：
 *   upstream/pixijs-engine/src/ticker/Ticker.ts
 *   upstream/pixijs-engine/src/ticker/TickerListener.ts
 *   upstream/pixijs-engine/src/ticker/const.ts
 *   upstream/pixijs-engine/src/app/TickerPlugin.ts
 *   skills/pixijs/pixijs-ticker/SKILL.md
 */

import { Ticker }           from '../../../upstream/pixijs-engine/src/ticker/Ticker'
import { UPDATE_PRIORITY }  from '../../../upstream/pixijs-engine/src/ticker/const'

// ─── Re-export so consumers don't need a direct upstream import ───────────────
export { Ticker, UPDATE_PRIORITY }
export type { TickerCallback } from '../../../upstream/pixijs-engine/src/ticker/Ticker'

// ─── Public types ─────────────────────────────────────────────────────────────

/**
 * One cell's visual state at a given instant.
 * Mirrors theatre-epoch-timeline.ts CellState so they're interchangeable.
 */
export interface EpochCellState {
  cell_id:  string
  x:        number
  y:        number
  w:        number
  h:        number
  opacity:  number
  z:        number
  r:        number
  g:        number
  b:        number
  color:    string
}

/**
 * One epoch's snapshot — the complete cell states at that instant.
 */
export interface EpochSnapshot {
  epoch: number
  cells: EpochCellState[]
}

/**
 * The interpolated frame payload emitted on every Ticker tick.
 */
export interface EpochTickFrame {
  /** Fractional epoch position (0 … epochCount-1). */
  position:    number
  /** Integer index of the "from" epoch. */
  epochIndex:  number
  /** 0–1 blend factor toward the next epoch (0 = fully at epochIndex). */
  blend:       number
  /** deltaTime from the Ticker (dimensionless scalar ~1.0 at 60fps). */
  deltaTime:   number
  /** deltaMS from the Ticker (real milliseconds, speed-scaled). */
  deltaMS:     number
  /** elapsedMS from the Ticker (raw unscaled milliseconds). */
  elapsedMS:   number
  /** Current measured FPS. */
  fps:         number
  /** Interpolated cell states ready for PixiJS renderer. */
  cells:       EpochCellState[]
}

export type EpochTickCallback = (frame: EpochTickFrame) => void

/** What happens when the epoch sequence reaches the end. */
export type EpochLoopMode = 'loop' | 'bounce' | 'stop'

/**
 * Optional handle to a Theatre.js ISequence-like object.
 * When supplied, EpochTicker writes `position` into it so Theatre.js
 * drives the bezier interpolation instead of our built-in lerp.
 */
export interface SequenceRef {
  /** Read/write the fractional position (0 … length). */
  position: number
  /** Total length of the sequence in position units. */
  length: number
}

export interface EpochTickerOptions {
  /**
   * Milliseconds per epoch unit (wall-clock, before speed scaling).
   * Default: 1000 (1 second per epoch).
   */
  msPerEpoch?: number

  /**
   * Initial playback speed multiplier applied on top of ticker.speed.
   * Default: 1.
   */
  playbackSpeed?: number

  /**
   * Behaviour at the end of the sequence.
   * Default: 'loop'
   */
  loopMode?: EpochLoopMode

  /**
   * If true, the Ticker starts automatically when the first callback is added.
   * Default: true  (mirrors Ticker.shared behaviour)
   */
  autoStart?: boolean

  /**
   * Optional Theatre.js sequence reference.  When set, EpochTicker advances
   * `sequenceRef.position` on every frame instead of interpolating internally.
   */
  sequenceRef?: SequenceRef

  /**
   * Use the global Ticker.shared singleton instead of creating a new instance.
   * Useful when the PixiJS Application was also created with sharedTicker:true.
   * Default: false
   */
  useSharedTicker?: boolean

  /**
   * Maximum FPS cap.  0 = uncapped (default).
   * Forwarded to ticker.maxFPS.
   */
  maxFPS?: number

  /**
   * Minimum FPS floor for deltaTime capping.
   * Forwarded to ticker.minFPS.  Default: 10.
   */
  minFPS?: number
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

function _hexToRgb(hex: string): { r: number; g: number; b: number } {
  const clean = hex.replace('#', '')
  if (clean.length === 3) {
    return {
      r: parseInt(clean[0] + clean[0], 16),
      g: parseInt(clean[1] + clean[1], 16),
      b: parseInt(clean[2] + clean[2], 16),
    }
  }
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  }
}

function _rgbToHex(r: number, g: number, b: number): string {
  const h = (n: number) =>
    Math.round(Math.max(0, Math.min(255, n))).toString(16).padStart(2, '0')
  return `#${h(r)}${h(g)}${h(b)}`
}

/** Linear interpolation between two EpochCellState objects (same cell_id). */
function _lerpCell(a: EpochCellState, b: EpochCellState, t: number): EpochCellState {
  const lerp = (x: number, y: number) => x + (y - x) * t
  const r = Math.round(lerp(a.r, b.r))
  const g = Math.round(lerp(a.g, b.g))
  const bv = Math.round(lerp(a.b, b.b))
  return {
    cell_id: a.cell_id,
    x:       lerp(a.x, b.x),
    y:       lerp(a.y, b.y),
    w:       lerp(a.w, b.w),
    h:       lerp(a.h, b.h),
    opacity: lerp(a.opacity, b.opacity),
    z:       lerp(a.z, b.z),
    r, g, b: bv,
    color:   _rgbToHex(r, g, bv),
  }
}

/**
 * Normalise a raw cell object (from JSON) to a full EpochCellState.
 * Accepts flat {x,y,w,h} or bbox.min/max triple format.
 */
export function normaliseEpochCell(raw: Partial<EpochCellState> & { cell_id: string; fill_color?: string; bbox?: { min: [number,number,number]; max: [number,number,number] } }): EpochCellState {
  let x = 0, y = 0, w = 100, h = 50

  if (typeof raw.x === 'number') {
    x = raw.x; y = raw.y ?? 0
    w = raw.w ?? 100; h = raw.h ?? 50
  } else if (raw.bbox) {
    const [x0, y0] = raw.bbox.min
    const [x1, y1] = raw.bbox.max
    x = x0; y = y0; w = x1 - x0; h = y1 - y0
  }

  const fillColor = (raw as any).fill_color ?? raw.color ?? '#808080'
  const { r, g, b } = _hexToRgb(fillColor)

  return {
    cell_id: raw.cell_id,
    x, y, w, h,
    opacity: raw.opacity ?? 1,
    z:       raw.z       ?? 3,
    r, g, b,
    color:   fillColor,
  }
}

// ─── EpochTicker class ────────────────────────────────────────────────────────

/**
 * EpochTicker — wraps upstream PixiJS Ticker to drive the epoch animation loop.
 *
 * @example
 * ```ts
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 * import { EpochTicker } from './epoch-ticker'
 *
 * const ticker = new EpochTicker(epochData.snapshots)
 *
 * const unsub = ticker.onFrame(({ cells, position, fps }) => {
 *   for (const cell of cells) {
 *     pixiContainers.get(cell.cell_id)?.set({ x: cell.x, y: cell.y, alpha: cell.opacity })
 *   }
 * })
 *
 * ticker.play()       // start loop
 * ticker.speed = 0.5  // slow motion
 * ticker.pause()      // pause
 * ticker.seek(2.5)    // jump to epoch 2.5
 * ticker.destroy()    // cleanup
 * ```
 */
export class EpochTicker {

  // ── Upstream Ticker instance ───────────────────────────────────────────────
  private readonly _ticker: Ticker

  // ── Epoch data ─────────────────────────────────────────────────────────────
  /**
   * Normalised snapshots indexed by epoch index.
   * Each entry is a Map<cell_id, EpochCellState> for O(1) lookup.
   */
  private readonly _snapshots: Map<string, EpochCellState>[]

  /** Total number of epochs. */
  private readonly _epochCount: number

  // ── Playback state ─────────────────────────────────────────────────────────

  /** Current fractional position in [0, epochCount-1]. */
  private _position: number = 0

  /** Playback direction for 'bounce' mode: +1 = forward, -1 = reverse. */
  private _bounceDir: 1 | -1 = 1

  /** Whether playback is currently paused (position still accessible). */
  private _paused: boolean = false

  /** Playback speed multiplier (on top of ticker.speed). */
  private _playbackSpeed: number

  /** Milliseconds per epoch unit (before speed scaling). */
  private _msPerEpoch: number

  /** Loop behaviour at sequence end. */
  private _loopMode: EpochLoopMode

  /** Optional Theatre.js / ISequence reference for external position sync. */
  private _sequenceRef: SequenceRef | null

  // ── Callbacks ──────────────────────────────────────────────────────────────

  /** Frame callbacks registered via onFrame(). */
  private readonly _frameCallbacks: Set<EpochTickCallback> = new Set()

  /** Callbacks registered at HIGH priority (physics, collision etc.). */
  private readonly _highCallbacks: Set<EpochTickCallback> = new Set()

  /** Callbacks registered at LOW priority (post-processing, metrics). */
  private readonly _lowCallbacks: Set<EpochTickCallback> = new Set()

  // ── Ticker listener handles (for removal on destroy) ──────────────────────
  private readonly _tickHigh:   (t: Ticker) => void
  private readonly _tickNormal: (t: Ticker) => void
  private readonly _tickLow:    (t: Ticker) => void

  // ─────────────────────────────────────────────────────────────────────────
  constructor(
    snapshots: EpochSnapshot[] | Array<{ epoch: number; cells: unknown[] }>,
    opts: EpochTickerOptions = {},
  ) {
    // ── Normalise epoch data ───────────────────────────────────────────────
    const sorted = [...snapshots].sort((a, b) => a.epoch - b.epoch)
    this._snapshots = sorted.map(snap =>
      new Map(
        snap.cells.map(raw => {
          const norm = normaliseEpochCell(raw as any)
          return [norm.cell_id, norm] as const
        }),
      ),
    )
    this._epochCount = this._snapshots.length
    if (this._epochCount === 0) {
      throw new Error('[EpochTicker] snapshots array must contain at least one epoch')
    }

    // ── Options ────────────────────────────────────────────────────────────
    this._msPerEpoch     = opts.msPerEpoch     ?? 1000
    this._playbackSpeed  = opts.playbackSpeed  ?? 1
    this._loopMode       = opts.loopMode       ?? 'loop'
    this._sequenceRef    = opts.sequenceRef    ?? null

    // ── Ticker setup ───────────────────────────────────────────────────────
    this._ticker = opts.useSharedTicker ? Ticker.shared : new Ticker()

    if (opts.maxFPS !== undefined) this._ticker.maxFPS = opts.maxFPS
    if (opts.minFPS !== undefined) this._ticker.minFPS = opts.minFPS

    // autoStart: default true to match Ticker.shared behaviour.
    this._ticker.autoStart = opts.autoStart ?? true

    // ── Build bound tick handlers ──────────────────────────────────────────
    // HIGH priority — fire physics/collision hooks BEFORE position update
    this._tickHigh = (t: Ticker) => {
      if (this._highCallbacks.size === 0) return
      const frame = this._buildFrame(t)
      for (const cb of this._highCallbacks) cb(frame)
    }

    // NORMAL priority — advance position + emit to NORMAL callbacks
    this._tickNormal = (t: Ticker) => {
      if (!this._paused) this._advance(t.deltaMS)
      const frame = this._buildFrame(t)
      for (const cb of this._frameCallbacks) cb(frame)
    }

    // LOW priority — after render, fire post-processing / metrics hooks
    this._tickLow = (t: Ticker) => {
      if (this._lowCallbacks.size === 0) return
      const frame = this._buildFrame(t)
      for (const cb of this._lowCallbacks) cb(frame)
    }

    // Register with the Ticker at the three priority levels.
    // The HIGH handler is only attached when _highCallbacks is non-empty
    // (see onHighFrame / removeHighFrame).  LOW likewise.
    this._ticker.add(this._tickNormal, undefined, UPDATE_PRIORITY.NORMAL)
  }

  // ── Public accessors ────────────────────────────────────────────────────────

  /** Total number of epochs in the sequence. */
  get epochCount(): number { return this._epochCount }

  /** Current fractional epoch position (0 … epochCount-1). */
  get position(): number { return this._position }

  /** True if currently paused. */
  get paused(): boolean { return this._paused }

  /** Whether the underlying Ticker has been started. */
  get started(): boolean { return this._ticker.started }

  /** Milliseconds per epoch unit (before speed scaling). */
  get msPerEpoch(): number { return this._msPerEpoch }
  set msPerEpoch(ms: number) {
    if (ms <= 0) throw new RangeError('[EpochTicker] msPerEpoch must be > 0')
    this._msPerEpoch = ms
  }

  /**
   * Playback speed multiplier (applied on top of ticker.speed).
   * 0.5 = half speed; 2.0 = double speed.
   */
  get playbackSpeed(): number { return this._playbackSpeed }
  set playbackSpeed(s: number) {
    this._playbackSpeed = Math.max(0, s)
  }

  /**
   * Direct access to the underlying PixiJS Ticker's speed property.
   * Stacks multiplicatively with playbackSpeed.
   */
  get tickerSpeed(): number { return this._ticker.speed }
  set tickerSpeed(s: number) { this._ticker.speed = s }

  /** Loop behaviour: 'loop' | 'bounce' | 'stop'. */
  get loopMode(): EpochLoopMode { return this._loopMode }
  set loopMode(m: EpochLoopMode) { this._loopMode = m }

  /** The raw PixiJS Ticker instance (for direct access if needed). */
  get ticker(): Ticker { return this._ticker }

  /** Current FPS measured by the Ticker. */
  get FPS(): number { return this._ticker.FPS }

  // ── Playback control ────────────────────────────────────────────────────────

  /**
   * Start the epoch animation loop.
   * Mirrors Theatre.js sequence.play() in intent; uses Ticker.start() under the hood.
   */
  play(speed?: number): this {
    if (speed !== undefined) this._playbackSpeed = speed
    this._paused = false
    this._bounceDir = 1
    if (!this._ticker.started) this._ticker.start()
    return this
  }

  /**
   * Pause the epoch animation loop.
   * Position is preserved; call play() or seek() to resume.
   */
  pause(): this {
    this._paused = true
    return this
  }

  /**
   * Seek to an arbitrary fractional epoch position.
   * Clamped to [0, epochCount-1].
   */
  seek(pos: number): this {
    this._position = Math.max(0, Math.min(pos, this._epochCount - 1))
    this._syncSequenceRef()
    return this
  }

  /**
   * Reset to position 0 and pause.
   * Equivalent to Theatre.js sequence.pause() + position = 0.
   */
  stop(): this {
    this._paused = true
    this._position = 0
    this._bounceDir = 1
    this._syncSequenceRef()
    return this
  }

  // ── Callback registration ───────────────────────────────────────────────────

  /**
   * Register a callback invoked on every frame at NORMAL priority.
   * The callback receives an EpochTickFrame with interpolated CellState[].
   * Returns an unsubscribe function.
   *
   * @example
   * ```ts
   * const unsub = ticker.onFrame(({ cells, position }) => {
   *   for (const c of cells) container.get(c.cell_id)?.set(c)
   * })
   * // later:
   * unsub()
   * ```
   */
  onFrame(cb: EpochTickCallback): () => void {
    this._frameCallbacks.add(cb)
    return () => this._frameCallbacks.delete(cb)
  }

  /**
   * Register a high-priority callback — fires BEFORE position update.
   * Use for physics collision checks or force-field reads that must
   * see the previous frame's position before it advances.
   *
   * Mirrored priority: UPDATE_PRIORITY.HIGH (25).
   */
  onHighFrame(cb: EpochTickCallback): () => void {
    const wasEmpty = this._highCallbacks.size === 0
    this._highCallbacks.add(cb)
    if (wasEmpty) {
      // Attach the HIGH-priority handler to the Ticker on first registration.
      this._ticker.add(this._tickHigh, undefined, UPDATE_PRIORITY.HIGH)
    }
    return () => {
      this._highCallbacks.delete(cb)
      if (this._highCallbacks.size === 0) {
        this._ticker.remove(this._tickHigh)
      }
    }
  }

  /**
   * Register a low-priority callback — fires AFTER the PixiJS render pass.
   * Use for post-processing metrics, export snapshots, or HUD updates.
   *
   * Mirrored priority: UPDATE_PRIORITY.LOW (-25).
   * Note: app.render() is registered by TickerPlugin at LOW; if you use
   * this in an Application, place your metrics reads at UTILITY (-50)
   * to guarantee they run after render.
   */
  onLowFrame(cb: EpochTickCallback): () => void {
    const wasEmpty = this._lowCallbacks.size === 0
    this._lowCallbacks.add(cb)
    if (wasEmpty) {
      this._ticker.add(this._tickLow, undefined, UPDATE_PRIORITY.LOW)
    }
    return () => {
      this._lowCallbacks.delete(cb)
      if (this._lowCallbacks.size === 0) {
        this._ticker.remove(this._tickLow)
      }
    }
  }

  /**
   * Run fn exactly once on the next Ticker frame, then auto-remove.
   * Uses Ticker.addOnce() internally.
   */
  addOnce(fn: EpochTickCallback): void {
    this._ticker.addOnce((t: Ticker) => {
      fn(this._buildFrame(t))
    }, undefined, UPDATE_PRIORITY.NORMAL)
  }

  // ── Snapshot access ─────────────────────────────────────────────────────────

  /**
   * Return the static CellState[] snapshot for a given integer epoch index.
   * No interpolation — the raw normalised values from the JSON.
   */
  readSnapshot(epochIndex: number): EpochCellState[] {
    const snap = this._snapshots[Math.round(epochIndex)]
    if (!snap) {
      console.warn(`[EpochTicker] epoch ${epochIndex} out of range`)
      return []
    }
    return [...snap.values()]
  }

  /**
   * Return the current interpolated frame without waiting for the next tick.
   * Uses the Ticker's most recent timing values.
   */
  getCurrentFrame(): EpochTickFrame {
    return this._buildFrame(this._ticker)
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────────

  /**
   * Destroy the EpochTicker and release all resources.
   * If the Ticker was created by EpochTicker (not shared), it is destroyed.
   * Shared Tickers are not destroyed (they are _protected).
   */
  destroy(): void {
    this._ticker.remove(this._tickHigh)
    this._ticker.remove(this._tickNormal)
    this._ticker.remove(this._tickLow)
    this._frameCallbacks.clear()
    this._highCallbacks.clear()
    this._lowCallbacks.clear()

    // Only destroy the Ticker if we own it (Ticker.shared is _protected).
    // Ticker.destroy() is a no-op on _protected instances.
    this._ticker.destroy()
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  /**
   * Advance the epoch position by the given deltaMS (already speed-scaled by
   * the Ticker).  Applies _playbackSpeed on top and handles loop/bounce/stop.
   */
  private _advance(deltaMS: number): void {
    const epochDelta = (deltaMS / this._msPerEpoch) * this._playbackSpeed * this._bounceDir
    let next = this._position + epochDelta

    const maxPos = this._epochCount - 1

    switch (this._loopMode) {
      case 'loop':
        // Wrap around: 0 → N-1 → 0 → …
        if (next >= this._epochCount) {
          next = next % this._epochCount
        } else if (next < 0) {
          next = (next % this._epochCount + this._epochCount) % this._epochCount
        }
        break

      case 'bounce':
        // Reverse direction at each end.
        if (next > maxPos) {
          next = maxPos - (next - maxPos)
          this._bounceDir = -1
        } else if (next < 0) {
          next = -next
          this._bounceDir = 1
        }
        next = Math.max(0, Math.min(next, maxPos))
        break

      case 'stop':
      default:
        next = Math.max(0, Math.min(next, maxPos))
        if (next >= maxPos || next <= 0) {
          this._paused = true
        }
        break
    }

    this._position = next
    this._syncSequenceRef()
  }

  /**
   * Push the current position into the Theatre.js sequence (if wired).
   * Theatre.js's onChange listener then fires _emitFrame() on EpochTimeline.
   */
  private _syncSequenceRef(): void {
    if (!this._sequenceRef) return
    // Theatre.js sequence length = epochCount - 1 (position 0 … N-1).
    this._sequenceRef.position = this._position
  }

  /**
   * Build an EpochTickFrame from the current _position and Ticker timing.
   * When _sequenceRef is set, Theatre.js has already interpolated the cells
   * and the caller's onFrame callback reads them from EpochTimeline instead;
   * we still emit a lightweight frame with timing data for HUD / metrics.
   * In standalone mode we lerp between the two nearest epoch snapshots.
   */
  private _buildFrame(t: Ticker): EpochTickFrame {
    const pos        = this._position
    const epochIndex = Math.min(Math.floor(pos), this._epochCount - 2)
    const safeIndex  = Math.max(0, epochIndex)
    const blend      = pos - Math.floor(pos)

    const cells = this._interpolateCells(safeIndex, blend)

    return {
      position:   pos,
      epochIndex: safeIndex,
      blend,
      deltaTime:  t.deltaTime,
      deltaMS:    t.deltaMS,
      elapsedMS:  t.elapsedMS,
      fps:        t.FPS,
      cells,
    }
  }

  /**
   * Interpolate cell states between epoch `idx` and `idx+1` by factor `t`.
   *
   * Cells present in only one epoch get opacity lerped from/to 0 so they
   * fade in/out gracefully — mirrors EpochTimeline's Theatre.js keyframe
   * missing-value strategy (opacity=0 for absent cells).
   */
  private _interpolateCells(idx: number, t: number): EpochCellState[] {
    const snapA = this._snapshots[idx]
    const snapB = this._snapshots[Math.min(idx + 1, this._epochCount - 1)]

    // Union of all cell IDs from both epochs.
    const allIds = new Set<string>([...snapA.keys(), ...snapB.keys()])
    const result: EpochCellState[] = []

    for (const id of allIds) {
      const a = snapA.get(id)
      const b = snapB.get(id)

      if (a && b) {
        // Both epochs have the cell — straight lerp.
        result.push(_lerpCell(a, b, t))
      } else if (a) {
        // Cell disappears in next epoch — fade out.
        const faded: EpochCellState = { ...a, opacity: a.opacity * (1 - t) }
        result.push(faded)
      } else if (b) {
        // Cell appears in next epoch — fade in.
        const faded: EpochCellState = { ...b, opacity: b.opacity * t }
        result.push(faded)
      }
    }

    return result
  }
}

// ─── Factory function ─────────────────────────────────────────────────────────

/**
 * Create an EpochTicker from epoch snapshot data.
 *
 * @param snapshots  Either EpochSnapshot[] or the full wrapper object
 *                   { snapshots: EpochSnapshot[] } (as written by channels/physics/).
 * @param opts       Optional EpochTickerOptions.
 *
 * @example
 * ```ts
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 * import { createEpochTicker } from './epoch-ticker'
 *
 * const ticker = createEpochTicker(epochData, {
 *   msPerEpoch:    800,
 *   loopMode:      'bounce',
 *   playbackSpeed: 1.5,
 *   maxFPS:        60,
 * })
 *
 * ticker.onFrame(({ cells, position, fps }) => {
 *   console.log(`[EpochTicker] pos=${position.toFixed(2)} fps=${fps.toFixed(1)}`)
 *   for (const cell of cells) {
 *     pixiContainers.get(cell.cell_id)?.set({ x: cell.x, y: cell.y, alpha: cell.opacity })
 *   }
 * })
 *
 * ticker.play()
 * ```
 */
export function createEpochTicker(
  snapshots: EpochSnapshot[] | { snapshots: EpochSnapshot[] } | Array<{ epoch: number; cells: unknown[] }>,
  opts: EpochTickerOptions = {},
): EpochTicker {
  const raw: Array<{ epoch: number; cells: unknown[] }> = Array.isArray(snapshots)
    ? snapshots
    : (snapshots as { snapshots: EpochSnapshot[] }).snapshots

  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error('[createEpochTicker] snapshots must be a non-empty array')
  }

  return new EpochTicker(raw, opts)
}

// ─── Theatre.js bridge ────────────────────────────────────────────────────────

/**
 * wireEpochTickerToTimeline — bridge between EpochTicker and an EpochTimeline
 * (from theatre-epoch-timeline.ts).
 *
 * Wires the Ticker's position advances into the Theatre.js sequence so that
 * Theatre.js drives bezier interpolation, and the EpochTimeline.onFrame()
 * callbacks continue to fire as usual.
 *
 * Usage:
 * ```ts
 * import { createEpochTimeline } from './theatre-epoch-timeline'
 * import { createEpochTicker, wireEpochTickerToTimeline } from './epoch-ticker'
 *
 * const tl = createEpochTimeline(epochData, { easing: 'spring' })
 * const et = createEpochTicker(epochData, { msPerEpoch: 1000 })
 *
 * // Wire: Ticker advances Theatre.js position.
 * const unwire = wireEpochTickerToTimeline(et, tl)
 *
 * // Theatre.js callbacks still fire via tl.onFrame(…)
 * tl.onFrame(({ cells }) => pixiRenderer.update(cells))
 *
 * // Control playback via the Ticker.
 * et.play(0.5)   // 0.5 epochs/sec
 * et.seek(2)
 * ```
 *
 * @param epochTicker  The EpochTicker driving position advances.
 * @param timeline     Any object exposing { seek(pos: number): void, epochCount: number }.
 * @returns            Unwire function that unregisters the frame callback.
 */
export function wireEpochTickerToTimeline(
  epochTicker: EpochTicker,
  timeline: { seek(pos: number): void; epochCount: number },
): () => void {
  return epochTicker.onFrame(({ position }) => {
    timeline.seek(position)
  })
}

// ─── Subsystem handle interfaces (duck-typed for decoupling) ─────────────────

/**
 * Duck-typed handle for EpochPlaybackController.
 * Accepts an `update(dt)` call each frame to sync playback position.
 */
export interface PlaybackControllerHandle {
  update(dt: number): void
}

/**
 * Duck-typed handle for a Theatre.js sequence.
 * `tick(dt)` advances the Theatre.js sequence position by `dt` milliseconds,
 * triggering its internal bezier interpolation and onChange listeners.
 */
export interface TheatreSequenceHandle {
  tick(dt: number): void
}

/**
 * Duck-typed handle for EdgeParticleSystem.
 * `update(dt)` runs one GPU transform-feedback pass and draws particles.
 */
export interface EdgeParticleHandle {
  update(dt: number): void
}

/**
 * Duck-typed handle for CloudFogBackground.
 * `update(dt)` advances u_time and re-renders the fog layer.
 */
export interface CloudFogHandle {
  update(dt: number): void
}

/**
 * Duck-typed handle for CellInteraction.
 * `check()` runs hit-testing and hover/click state reconciliation.
 */
export interface CellInteractionHandle {
  check(): void
}

/**
 * Duck-typed handle for a PixiJS-style renderer + stage.
 * `render(stage)` submits a draw call for the scene graph.
 */
export interface RendererHandle {
  render(stage: unknown): void
}

/**
 * Full frame-loop subsystem handles.
 * Each is optional — when omitted, that step is a no-op.
 */
export interface FrameLoopSubsystems {
  /** 1) EpochPlaybackController — advances playback state */
  playbackController?: PlaybackControllerHandle
  /** 2) Theatre.js sequence — drives bezier interpolation */
  theatreSequence?:    TheatreSequenceHandle
  /** 3) EdgeParticleSystem — GPU particle simulation + draw */
  edgeParticles?:      EdgeParticleHandle
  /** 4) CloudFogBackground — volumetric fog layer update */
  cloudFog?:           CloudFogHandle
  /** 5) CellInteraction — hover/click hit-test reconciliation */
  cellInteraction?:    CellInteractionHandle
  /** 6) Renderer + stage — final PixiJS scene render */
  renderer?:           RendererHandle
  stage?:              unknown
}

// ─── Convenience: attach to a PixiJS Application ticker ──────────────────────

/**
 * attachEpochTickerToApp — attach an EpochTicker's frame loop to a running
 * PixiJS Application by replacing its default animation loop.
 *
 * Instead of creating a second Ticker, this registers the EpochTicker's
 * position-advance logic as a listener on the Application's existing ticker,
 * then fires the supplied frame callback via app.ticker.
 *
 * This is the preferred pattern when the PixiJS Application already owns
 * a Ticker (the default case); it avoids double-ticking.
 *
 * The tick(dt) function executes the full frame loop in strict order:
 *   1) EpochPlaybackController.update(dt)  — advance playback state
 *   2) Theatre sequence tick                — drive bezier interpolation
 *   3) EdgeParticleSystem.update(dt)        — GPU particle sim + draw
 *   4) CloudFogBackground.update(dt)        — fog layer update
 *   5) CellInteraction.check()              — hover/click reconciliation
 *   6) app.renderer.render(stage)           — final scene render
 *
 * @param app           A PixiJS Application (must have app.ticker).
 * @param snapshots     Epoch snapshot array.
 * @param onFrame       Frame callback; receives EpochTickFrame each tick.
 * @param opts          EpochTickerOptions (useSharedTicker is ignored — app.ticker is used).
 * @param subsystems    Optional subsystem handles for the full frame loop.
 * @returns             stop() to remove the listener and clean up.
 *
 * @example
 * ```ts
 * const { stop } = attachEpochTickerToApp(app, epochData.snapshots, ({ cells }) => {
 *   for (const c of cells) stage.getChildByName(c.cell_id)?.set(c)
 * }, {}, {
 *   playbackController: playback,
 *   theatreSequence:    sequence,
 *   edgeParticles:      eps,
 *   cloudFog:           fog,
 *   cellInteraction:    interaction,
 *   renderer:           app.renderer,
 *   stage:              app.stage,
 * })
 * // later:
 * stop()
 * ```
 */
export function attachEpochTickerToApp(
  app: { ticker: Ticker },
  snapshots: EpochSnapshot[],
  onFrame: EpochTickCallback,
  opts: Omit<EpochTickerOptions, 'useSharedTicker'> = {},
  subsystems: FrameLoopSubsystems = {},
): { stop: () => void } {

  const msPerEpoch    = opts.msPerEpoch    ?? 1000
  const playbackSpeed = opts.playbackSpeed ?? 1
  const loopMode      = opts.loopMode      ?? 'loop'
  const epochCount    = snapshots.length

  // Build snapshot maps for O(1) lookup.
  const sorted = [...snapshots].sort((a, b) => a.epoch - b.epoch)
  const maps: Map<string, EpochCellState>[] = sorted.map(snap =>
    new Map(snap.cells.map(raw => {
      const norm = normaliseEpochCell(raw as any)
      return [norm.cell_id, norm] as const
    })),
  )

  let position  = 0
  let bounceDir: 1 | -1 = 1
  let paused    = false

  function tick(t: Ticker): void {
    const dt = t.deltaMS

    // ── Step 1: EpochPlaybackController.update(dt) ──────────────────────
    // Advance playback state (rate, position tracking, state callbacks).
    subsystems.playbackController?.update(dt)

    // ── Step 2: Theatre sequence tick ───────────────────────────────────
    // Drive Theatre.js bezier interpolation by advancing the sequence
    // position. When no Theatre.js handle is provided, we advance the
    // built-in linear position tracker instead.
    subsystems.theatreSequence?.tick(dt)

    if (!paused) {
      const delta = (dt / msPerEpoch) * playbackSpeed * bounceDir
      let next = position + delta
      const maxPos = epochCount - 1

      if (loopMode === 'loop') {
        if (next >= epochCount) next = next % epochCount
        else if (next < 0)     next = (next % epochCount + epochCount) % epochCount
      } else if (loopMode === 'bounce') {
        if (next > maxPos) { next = maxPos - (next - maxPos); bounceDir = -1 }
        else if (next < 0) { next = -next; bounceDir = 1 }
        next = Math.max(0, Math.min(next, maxPos))
      } else {
        next = Math.max(0, Math.min(next, maxPos))
        if (next >= maxPos || next <= 0) paused = true
      }

      position = next
    }

    // Build frame
    const pos        = position
    const epochIndex = Math.max(0, Math.min(Math.floor(pos), epochCount - 2))
    const blend      = pos - Math.floor(pos)

    // Interpolate cells
    const snapA = maps[epochIndex]
    const snapB = maps[Math.min(epochIndex + 1, epochCount - 1)]
    const allIds = new Set<string>([...snapA.keys(), ...snapB.keys()])
    const cells: EpochCellState[] = []

    for (const id of allIds) {
      const a = snapA.get(id)
      const b = snapB.get(id)
      if (a && b)  cells.push(_lerpCell(a, b, blend))
      else if (a)  cells.push({ ...a, opacity: a.opacity * (1 - blend) })
      else if (b)  cells.push({ ...b, opacity: b.opacity * blend })
    }

    onFrame({ position: pos, epochIndex, blend, deltaTime: t.deltaTime, deltaMS: dt, elapsedMS: t.elapsedMS, fps: t.FPS, cells })

    // ── Step 3: EdgeParticleSystem.update(dt) ───────────────────────────
    // Run one GPU transform-feedback pass and draw edge particles.
    subsystems.edgeParticles?.update(dt)

    // ── Step 4: CloudFogBackground.update(dt) ───────────────────────────
    // Advance u_time and re-render the volumetric fog layer.
    subsystems.cloudFog?.update(dt)

    // ── Step 5: CellInteraction.check() ─────────────────────────────────
    // Reconcile hover/click state against current cell positions.
    subsystems.cellInteraction?.check()

    // ── Step 6: app.renderer.render(stage) ──────────────────────────────
    // Final PixiJS scene render — submit the composited frame.
    if (subsystems.renderer && subsystems.stage !== undefined) {
      subsystems.renderer.render(subsystems.stage)
    }
  }

  app.ticker.add(tick, undefined, UPDATE_PRIORITY.NORMAL)

  return {
    stop: () => app.ticker.remove(tick),
  }
}
