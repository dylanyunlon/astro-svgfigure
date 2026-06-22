/**
 * theatre-bridge.ts — Theatre.js 时间轴动画桥接层  (M762)
 *
 * 统一 Theatre.js core (Project / Sheet / Sequence / SheetObject) 与
 * cell-pubsub-loop epoch 系统之间的桥接，提供：
 *
 *   TheatreTimeline      — 单一 Theatre.js Sheet + Sequence 封装，管理 N 个
 *                          SheetObject 的关键帧注册、bezier 插值、播放控制
 *   TheatreObjectHandle  — 单个 SheetObject 的 reactive 封装，支持
 *                          onValuesChange / snapshot / prop pointer 访问
 *   TheatreRafBridge     — 将 Theatre.js tick 接入外部帧循环
 *                          (PixiJS Ticker / rAF / EpochTicker)
 *   KeyframeBuilder      — 声明式关键帧构建器，从 epoch JSON 批量注入 keyframes
 *
 * ─── 架构 ────────────────────────────────────────────────────────────────────
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │  TheatreTimeline  (本模块)                                         │
 *   │    project: IProject  "astro-svgfigure"                            │
 *   │    sheet:   ISheet    "{sheetName}"                                 │
 *   │    sequence: ISequence  0 … duration                               │
 *   │                                                                     │
 *   │    objects: Map<objectKey, TheatreObjectHandle>                     │
 *   │      ├─ SheetObject "cell_A"  → props { x, y, w, h, opacity, … }  │
 *   │      ├─ SheetObject "cell_B"  → props { … }                       │
 *   │      └─ …                                                          │
 *   │                                                                     │
 *   │    ┌─────────────────────────────────────────────────────────┐      │
 *   │    │  TheatreRafBridge                                       │      │
 *   │    │    createRafDriver({ name, start, stop })               │      │
 *   │    │    tick(performance.now()) — 外部帧循环调用              │      │
 *   │    │    Theatre.js onChange listeners 在 tick 内触发           │      │
 *   │    └─────────────────────────────────────────────────────────┘      │
 *   │                                                                     │
 *   │    ┌─────────────────────────────────────────────────────────┐      │
 *   │    │  KeyframeBuilder                                        │      │
 *   │    │    .at(position, propsMap)     — 在 t=position 插入帧    │      │
 *   │    │    .fromEpochParams(epochJSON) — 从 epoch JSON 批量注入  │      │
 *   │    │    .build() → OnDiskState                               │      │
 *   │    └─────────────────────────────────────────────────────────┘      │
 *   └────────────────────────────────────────────────────────────────────┘
 *            │                                       ▲
 *            │ play / pause / seek                    │ epoch events
 *            ▼                                       │
 *   ┌─────────────────┐                    ┌─────────────────────┐
 *   │  PixiJS Ticker   │                    │  cell-pubsub-loop   │
 *   │  or EpochTicker   │                    │  epoch_controller   │
 *   │  or custom rAF    │                    │  advanceEpoch()     │
 *   └─────────────────┘                    └─────────────────────┘
 *
 * ─── Theatre.js 概念映射 ─────────────────────────────────────────────────────
 *
 *   Theatre.js             cell-pubsub-loop
 *   ──────────             ────────────────
 *   Project                astro-svgfigure 项目
 *   Sheet                  一组 cell 的动画场景（可有多个 instance）
 *   Sequence               epoch 时间线 (position 0…N)
 *   SheetObject            单个 cell 的可动画属性
 *   Keyframe               epoch 快照中的某一帧属性值
 *   RafDriver              外部帧循环驱动器
 *   Atom (dataverse)       reactive 状态容器
 *
 * ─── 与现有模块关系 ─────────────────────────────────────────────────────────
 *
 *   theatre-epoch-timeline.ts     — M152: 2-epoch 特化实现，直接构建 OnDiskState
 *   theatre-epoch-cell-bridge.ts  — M058: PixiJS Container 写入桥接
 *   epoch-playback-controller.ts  — M069: play/pause/seek UI 控制层
 *   epoch-ticker.ts               — M016: PixiJS Ticker 帧循环 + position 推进
 *   tween-system.ts               — TweenManager 手动补间（无 Theatre.js）
 *
 *   本模块 (theatre-bridge.ts) 是通用底层，上述模块可在此基础上构建。
 *   区别：theatre-epoch-timeline 自行拼接 OnDiskState 并管理 epoch 0/1；
 *   theatre-bridge 提供通用 N-keyframe 管理 + reactive handle + raf bridge。
 *
 * Upstream 引用:
 *   upstream/theatre-js/core/src/coreExports.ts      — getProject, onChange, val, types, createRafDriver
 *   upstream/theatre-js/core/src/types/public.ts      — IProject, ISheet, ISequence, ISheetObject
 *   upstream/theatre-js/core/src/sequences/Sequence.ts— position, play(), pause()
 *   upstream/theatre-js/dataverse/src/Atom.ts         — Atom reactive state
 *   upstream/theatre-js/dataverse/src/Ticker.ts       — Ticker (dataverse-level)
 *   upstream/theatre-js/core/src/rafDrivers.ts        — createRafDriver
 */

// ─── Vendored Theatre.js core ───────────────────────────────────────────────

import {
  getProject,
  onChange,
  val,
  types,
  createRafDriver,
} from '../../upstream/theatre-js/core/src/coreExports'

import type {
  IProject,
  ISheet,
  ISheetObject,
  ISequence,
  IRafDriver,
  IPlaybackDirection,
  IPlaybackRange,
  BasicKeyframe,
} from '../../upstream/theatre-js/core/src/types/public'

// ─── Dataverse reactive primitives ──────────────────────────────────────────

import Atom from '../../upstream/theatre-js/dataverse/src/Atom'

// ─── Types ──────────────────────────────────────────────────────────────────

/**
 * Prop schema for a single animatable cell.
 *
 * This is the Theatre.js type descriptor (using `types.number`, etc.).
 * The *value* type (what you read at runtime) is `CellPropValues`.
 */
export interface CellPropSchema {
  x:             ReturnType<typeof types.number>
  y:             ReturnType<typeof types.number>
  w:             ReturnType<typeof types.number>
  h:             ReturnType<typeof types.number>
  opacity:       ReturnType<typeof types.number>
  z:             ReturnType<typeof types.number>
  r:             ReturnType<typeof types.number>
  g:             ReturnType<typeof types.number>
  b:             ReturnType<typeof types.number>
  glowIntensity: ReturnType<typeof types.number>
}

/** Runtime prop values — the resolved numeric form of CellPropSchema. */
export interface CellPropValues {
  x:             number
  y:             number
  w:             number
  h:             number
  opacity:       number
  z:             number
  r:             number
  g:             number
  b:             number
  glowIntensity: number
}

/** A single keyframe entry for the KeyframeBuilder. */
export interface KeyframeEntry {
  /** Position on the sequence timeline (in seconds / epochs). */
  position: number
  /** Property values at this position. Partial — missing props keep previous value. */
  values:   Partial<CellPropValues>
}

/** Epoch-param JSON shape (mirrors theatre-epoch-timeline.ts EpochParamJSON). */
export interface EpochParamJSON {
  cell_id:        string
  species:        string
  bbox:           { x: number; y: number; w: number; h: number; z?: number }
  z?:             number
  opacity:        number
  fill_color:     string
  stroke_color?:  string
  label?:         string
  font_size?:     number
  species_params?: Record<string, unknown>
  epoch?:         number
}

/** Bezier handle presets for keyframe interpolation. */
export type EasingPreset = 'linear' | 'ease' | 'spring' | 'sharp'

/** A snapshot of all objects' current prop values. */
export interface TimelineSnapshot {
  position: number
  objects:  Map<string, CellPropValues>
}

/** Frame event emitted on every tick while the sequence is playing. */
export interface TheatreFrame {
  /** Current sequence position (fractional). */
  position:  number
  /** True if the sequence is actively playing. */
  playing:   boolean
  /** Delta time in ms since the last tick. */
  deltaMs:   number
  /** Interpolated prop values for all registered objects. */
  objects:   Map<string, CellPropValues>
}

export type TheatreFrameCallback = (frame: TheatreFrame) => void

/** Options for TheatreTimeline construction. */
export interface TheatreTimelineOptions {
  /** Theatre.js project ID.  Default: 'astro-svgfigure'. */
  projectId?:    string
  /** Sheet name within the project.  Default: 'main'. */
  sheetName?:    string
  /** Optional sheet instance ID (for multiple instances of the same sheet). */
  instanceId?:   string
  /** Sequence duration in position units.  Default: 10. */
  duration?:     number
  /** Default playback rate (position units per second).  Default: 1. */
  defaultRate?:  number
  /** Default easing for keyframes.  Default: 'ease'. */
  easing?:       EasingPreset
  /**
   * Pre-built Theatre.js project state (OnDiskState JSON).
   * When provided, the project is initialised with these keyframes.
   * When omitted, objects start at their schema defaults.
   */
  state?:        Record<string, unknown>
  /**
   * Custom rafDriver.  When omitted, TheatreTimeline creates its own
   * via createRafDriver() so it can be manually ticked from an external
   * frame loop (PixiJS Ticker, EpochTicker, etc.).
   */
  rafDriver?:    IRafDriver
}

// ─── Constants ──────────────────────────────────────────────────────────────

/** Default cell prop schema — Theatre.js type descriptors with ranges. */
const DEFAULT_CELL_PROPS = {
  x:             types.number(0,   { range: [-4000, 4000] }),
  y:             types.number(0,   { range: [-4000, 4000] }),
  w:             types.number(100, { range: [0, 2000] }),
  h:             types.number(50,  { range: [0, 2000] }),
  opacity:       types.number(1,   { range: [0, 1] }),
  z:             types.number(3,   { range: [0, 20] }),
  r:             types.number(128, { range: [0, 255] }),
  g:             types.number(128, { range: [0, 255] }),
  b:             types.number(128, { range: [0, 255] }),
  glowIntensity: types.number(1,   { range: [0, 5] }),
} as const

/** Bezier handle values for easing presets. [leftX, leftY, rightX, rightY] */
const EASING_HANDLES: Record<EasingPreset, [number, number, number, number]> = {
  linear: [0, 0, 1, 1],
  ease:   [0.25, 0.1, 0.25, 1],
  spring: [0.34, 1.56, 0.64, 1],
  sharp:  [0.4, 0, 0.2, 1],
}

/** Default CellPropValues used as fallback. */
const DEFAULT_VALUES: CellPropValues = {
  x: 0, y: 0, w: 100, h: 50,
  opacity: 1, z: 3,
  r: 128, g: 128, b: 128,
  glowIntensity: 1,
}

/** All prop keys in CellPropValues, used for iteration. */
const PROP_KEYS: (keyof CellPropValues)[] = [
  'x', 'y', 'w', 'h', 'opacity', 'z', 'r', 'g', 'b', 'glowIntensity',
]

// ─── Colour helpers ─────────────────────────────────────────────────────────

function hexToRgb(hex: string): { r: number; g: number; b: number } {
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

// ─── TheatreObjectHandle ────────────────────────────────────────────────────

/**
 * TheatreObjectHandle — reactive wrapper around a single ISheetObject.
 *
 * Provides a simplified interface for:
 *   - Reading current prop values: `handle.value`
 *   - Subscribing to value changes: `handle.onValuesChange(cb)`
 *   - Snapshotting: `handle.snapshot()`
 *   - Accessing the raw Theatre.js object for advanced use
 *
 * The handle uses a dataverse Atom to maintain a local mirror of the
 * SheetObject's props, updated on every Theatre.js tick.  This avoids
 * calling val(obj.props) on every read — the Atom is the source of truth
 * between ticks.
 */
export class TheatreObjectHandle {

  /** The underlying Theatre.js SheetObject. */
  readonly sheetObject: ISheetObject<typeof DEFAULT_CELL_PROPS>

  /** Unique key identifying this object within the sheet. */
  readonly key: string

  /** Reactive state mirror — updated on every Theatre.js onChange tick. */
  private readonly _atom: InstanceType<typeof Atom<CellPropValues>>

  /** Unsubscribe handle for the onValuesChange listener. */
  private _unsub: (() => void) | null = null

  /** External value-change subscribers. */
  private readonly _subscribers: Set<(values: CellPropValues) => void> = new Set()

  constructor(
    key: string,
    sheetObject: ISheetObject<typeof DEFAULT_CELL_PROPS>,
    rafDriver?: IRafDriver,
  ) {
    this.key = key
    this.sheetObject = sheetObject

    // Initialise the Atom with current Theatre.js values.
    const initial = val(sheetObject.props) as CellPropValues
    this._atom = new Atom<CellPropValues>({ ...DEFAULT_VALUES, ...initial })

    // Subscribe to Theatre.js value changes → push into Atom + notify subscribers.
    this._unsub = sheetObject.onValuesChange((values) => {
      const typed = values as CellPropValues
      this._atom.set(typed)
      for (const cb of this._subscribers) cb(typed)
    }, rafDriver)
  }

  /** Current prop values (synchronous read from Atom mirror). */
  get value(): CellPropValues {
    return this._atom.get()
  }

  /**
   * Subscribe to prop value changes.
   * The callback fires on every Theatre.js tick where values have changed.
   *
   * @returns Unsubscribe function.
   */
  onValuesChange(cb: (values: CellPropValues) => void): () => void {
    this._subscribers.add(cb)
    // Immediately emit current value so the subscriber has initial state.
    cb(this.value)
    return () => { this._subscribers.delete(cb) }
  }

  /**
   * Take a snapshot of the current prop values.
   * Returns a plain object copy (not a reference to the Atom).
   */
  snapshot(): CellPropValues {
    return { ...this._atom.get() }
  }

  /**
   * Set the initial value of the SheetObject.
   * This overrides schema defaults but is itself overridden by
   * keyframed or studio-edited values.
   */
  setInitialValue(values: Partial<CellPropValues>): void {
    this.sheetObject.initialValue = values as any
  }

  /**
   * Dispose this handle — unsubscribe from Theatre.js.
   */
  dispose(): void {
    if (this._unsub) {
      this._unsub()
      this._unsub = null
    }
    this._subscribers.clear()
  }
}

// ─── KeyframeBuilder ────────────────────────────────────────────────────────

/**
 * KeyframeBuilder — declarative builder for Theatre.js OnDiskState.
 *
 * Collects keyframes for multiple objects at multiple positions, then
 * serialises them into the Theatre.js OnDiskState JSON structure that
 * can be passed to `getProject(id, { state })`.
 *
 * This replaces the manual OnDiskState construction in theatre-epoch-timeline.ts
 * with a fluent API that supports arbitrary numbers of epochs/keyframes.
 *
 * Usage:
 *   const state = new KeyframeBuilder('epoch', 'ease')
 *     .at(0, { cell_A: { x: 10, y: 20, ... }, cell_B: { x: 50, y: 60, ... } })
 *     .at(1, { cell_A: { x: 110, y: 120, ... }, cell_B: { x: 150, y: 160, ... } })
 *     .build()
 *
 *   const project = getProject('astro-svgfigure', { state })
 */
export class KeyframeBuilder {

  /** Sheet name to embed keyframes under. */
  private readonly _sheetId: string

  /** Easing preset for all bezier handles. */
  private readonly _easing: EasingPreset

  /**
   * Accumulated keyframe data.
   * Structure: objectKey → propName → Array<{ position, value }>
   */
  private readonly _data: Map<string, Map<keyof CellPropValues, Array<{ position: number; value: number }>>> = new Map()

  constructor(sheetId: string = 'main', easing: EasingPreset = 'ease') {
    this._sheetId = sheetId
    this._easing = easing
  }

  /**
   * Register keyframe values for one or more objects at a given timeline position.
   *
   * @param position  The position on the sequence timeline (e.g. 0, 1, 2.5).
   * @param values    Map of objectKey → partial prop values at this position.
   * @returns         `this` for chaining.
   */
  at(position: number, values: Record<string, Partial<CellPropValues>>): this {
    for (const [objectKey, props] of Object.entries(values)) {
      if (!this._data.has(objectKey)) {
        this._data.set(objectKey, new Map())
      }
      const objectMap = this._data.get(objectKey)!

      for (const prop of PROP_KEYS) {
        if (prop in props) {
          if (!objectMap.has(prop)) {
            objectMap.set(prop, [])
          }
          objectMap.get(prop)!.push({
            position,
            value: (props as CellPropValues)[prop],
          })
        }
      }
    }
    return this
  }

  /**
   * Inject keyframes from an epoch-params JSON array.
   *
   * Each EpochParamJSON is placed at the given position.
   * Extracts x, y, w, h, opacity, z, r, g, b from bbox + fill_color.
   *
   * @param position    Sequence position for this epoch.
   * @param params      Array of EpochParamJSON objects (one per cell).
   * @returns           `this` for chaining.
   */
  fromEpochParams(position: number, params: EpochParamJSON[]): this {
    const values: Record<string, Partial<CellPropValues>> = {}

    for (const p of params) {
      const { r, g, b } = hexToRgb(p.fill_color)
      values[p.cell_id] = {
        x:       p.bbox.x,
        y:       p.bbox.y,
        w:       p.bbox.w,
        h:       p.bbox.h,
        opacity: p.opacity,
        z:       p.bbox.z ?? p.z ?? 3,
        r, g, b,
        glowIntensity: 1,
      }
    }

    return this.at(position, values)
  }

  /**
   * Build the Theatre.js OnDiskState JSON.
   *
   * The returned object can be passed to `getProject(id, { state })`.
   *
   * State format: upstream/theatre-js/core/src/types/private/core.ts
   *   definitionVersion = '0.4.0'
   */
  build(): Record<string, unknown> {
    const handles = EASING_HANDLES[this._easing]

    type KFRecord = Record<string, {
      id:             string
      position:       number
      value:          number
      handles:        [number, number, number, number]
      connectedRight: boolean
      type:           'bezier'
    }>

    const tracksByObject: Record<string, {
      trackIdByPropPath: Record<string, string>
      trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }>
    }> = {}

    for (const [objectKey, propsMap] of this._data) {
      const trackIdByPropPath: Record<string, string> = {}
      const trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }> = {}

      for (const [propName, frames] of propsMap) {
        const trackId = `${objectKey}__${propName}`
        const propPath = JSON.stringify([propName])
        trackIdByPropPath[propPath] = trackId

        const keyframes: KFRecord = {}

        // Sort frames by position for consistent ordering.
        const sorted = [...frames].sort((a, b) => a.position - b.position)

        for (let i = 0; i < sorted.length; i++) {
          const frame = sorted[i]
          const kfId = `${objectKey}_${propName}_p${i}`
          keyframes[kfId] = {
            id:             kfId,
            position:       frame.position,
            value:          frame.value,
            handles,
            connectedRight: i < sorted.length - 1,
            type:           'bezier',
          }
        }

        trackData[trackId] = {
          type:      'BasicKeyframedTrack',
          keyframes,
        }
      }

      tracksByObject[objectKey] = { trackIdByPropPath, trackData }
    }

    return {
      shepimaversion: 1,
      definitionVersion: '0.4.0',
      revisionHistory: ['initial'],
      sheetsById: {
        [this._sheetId]: {
          staticOverrides: { byObject: {} },
          sequence: {
            subUnitsPerUnit: 30,
            length: this._computeMaxPosition(),
            tracksByObject,
          },
        },
      },
    }
  }

  /** Compute the maximum position across all keyframes. */
  private _computeMaxPosition(): number {
    let max = 0
    for (const propsMap of this._data.values()) {
      for (const frames of propsMap.values()) {
        for (const f of frames) {
          if (f.position > max) max = f.position
        }
      }
    }
    return Math.max(max, 1) // at least 1 unit duration
  }

  /** Get all unique object keys that have been registered. */
  get objectKeys(): string[] {
    return [...this._data.keys()]
  }

  /** Get the number of keyframes registered for a given object + prop. */
  keyframeCount(objectKey: string, prop: keyof CellPropValues): number {
    return this._data.get(objectKey)?.get(prop)?.length ?? 0
  }

  /** Clear all accumulated keyframe data. */
  clear(): this {
    this._data.clear()
    return this
  }
}

// ─── TheatreRafBridge ───────────────────────────────────────────────────────

/**
 * TheatreRafBridge — manual tick driver for Theatre.js.
 *
 * By default, Theatre.js creates its own requestAnimationFrame loop.
 * When integrating with an external frame loop (PixiJS Ticker, EpochTicker,
 * or a custom rAF), you want Theatre.js to only advance when YOU tell it to.
 *
 * TheatreRafBridge wraps `createRafDriver()` and exposes a `tick(timeMs)`
 * method that the external loop calls each frame.  Theatre.js onChange
 * listeners fire synchronously within the tick call.
 *
 * Usage:
 *   const raf = new TheatreRafBridge('pixi-ticker-driver')
 *   // Pass raf.driver to TheatreTimeline options or sequence.play({ rafDriver })
 *   // In your frame loop:
 *   ticker.add(() => raf.tick(performance.now()))
 */
export class TheatreRafBridge {

  /** The Theatre.js IRafDriver managed by this bridge. */
  readonly driver: IRafDriver

  /** True when the driver has been started (Theatre.js has active work). */
  private _running = false

  /** Timestamp of the last tick, for deltaMs computation. */
  private _lastTime = 0

  /** Accumulated frame count. */
  private _frameCount = 0

  constructor(name: string = 'TheatreBridge_RafDriver') {
    this.driver = createRafDriver({
      name,
      start: () => { this._running = true },
      stop:  () => { this._running = false },
    })
  }

  /**
   * Advance Theatre.js by one tick.
   *
   * Call this from your external frame loop with the current timestamp
   * (typically `performance.now()`).  Theatre.js onChange listeners and
   * sequence position updates fire synchronously within this call.
   *
   * @param timeMs  Current time in milliseconds (e.g. `performance.now()`).
   * @returns       Delta time in ms since the last tick.
   */
  tick(timeMs: number): number {
    const delta = this._lastTime > 0 ? timeMs - this._lastTime : 0
    this._lastTime = timeMs
    this._frameCount++
    this.driver.tick(timeMs)
    return delta
  }

  /** True if Theatre.js has active computations (sequences playing, etc.). */
  get running(): boolean { return this._running }

  /** Total number of ticks since creation. */
  get frameCount(): number { return this._frameCount }

  /** Reset the last-time tracker (e.g. after a long pause). */
  resetTimer(): void { this._lastTime = 0 }
}

// ─── TheatreTimeline ────────────────────────────────────────────────────────

/**
 * TheatreTimeline — unified Theatre.js Sheet + Sequence lifecycle manager.
 *
 * Responsibilities:
 *   1. Creates / retrieves the Theatre.js Project, Sheet, and Sequence.
 *   2. Registers SheetObjects for each cell with the standard prop schema.
 *   3. Wraps each SheetObject in a TheatreObjectHandle for reactive access.
 *   4. Provides play / pause / seek / setRate APIs proxied to the Sequence.
 *   5. Emits per-frame callbacks with all interpolated prop values.
 *   6. Manages a TheatreRafBridge for external frame-loop integration.
 *   7. Supports dynamic object registration (new cells arriving mid-epoch).
 *
 * @example
 * ```ts
 * // Build keyframes from epoch data
 * const builder = new KeyframeBuilder('epoch', 'ease')
 *   .fromEpochParams(0, epoch0Cells)
 *   .fromEpochParams(1, epoch1Cells)
 *   .fromEpochParams(2, epoch2Cells)
 *
 * // Create timeline
 * const timeline = new TheatreTimeline({
 *   sheetName: 'epoch',
 *   state: builder.build(),
 *   duration: 2,
 * })
 *
 * // Register objects for each cell
 * for (const cellId of builder.objectKeys) {
 *   timeline.registerObject(cellId)
 * }
 *
 * // Listen to frames
 * timeline.onFrame(({ position, objects }) => {
 *   for (const [cellId, values] of objects) {
 *     updatePixiContainer(cellId, values)
 *   }
 * })
 *
 * // Play the timeline
 * await timeline.play()
 * ```
 */
export class TheatreTimeline {

  /** Theatre.js Project instance. */
  readonly project: IProject

  /** Theatre.js Sheet instance. */
  readonly sheet: ISheet

  /** Theatre.js Sequence — drives playback. */
  readonly sequence: ISequence

  /** Custom raf driver for external frame-loop integration. */
  readonly rafBridge: TheatreRafBridge

  /** Object handles by key. */
  private readonly _objects: Map<string, TheatreObjectHandle> = new Map()

  /** Frame callbacks. */
  private readonly _frameCallbacks: Set<TheatreFrameCallback> = new Set()

  /** Sequence onChange unsubscribe handles. */
  private readonly _unsubs: Array<() => void> = []

  /** Resolved options. */
  private readonly _opts: Required<TheatreTimelineOptions>

  /** Cached sequence position from the last tick. */
  private _position = 0

  /** Last tick timestamp for delta computation. */
  private _lastTickMs = 0

  /** True once destroy() has been called. */
  private _destroyed = false

  constructor(options: TheatreTimelineOptions = {}) {
    this._opts = {
      projectId:   options.projectId   ?? 'astro-svgfigure',
      sheetName:   options.sheetName   ?? 'main',
      instanceId:  options.instanceId  ?? '',
      duration:    options.duration    ?? 10,
      defaultRate: options.defaultRate ?? 1,
      easing:      options.easing      ?? 'ease',
      state:       options.state       ?? {},
      rafDriver:   options.rafDriver   ?? (null as any),
    }

    // Create or reuse the Theatre.js project.
    const projectConfig: Record<string, unknown> = {}
    if (options.state && Object.keys(options.state).length > 0) {
      projectConfig.state = options.state
    }
    this.project = getProject(this._opts.projectId, projectConfig as any)

    // Create the sheet (with optional instance ID).
    this.sheet = this._opts.instanceId
      ? this.project.sheet(this._opts.sheetName, this._opts.instanceId)
      : this.project.sheet(this._opts.sheetName)

    this.sequence = this.sheet.sequence

    // Set up the raf bridge.
    if (options.rafDriver) {
      // External driver provided — wrap it.
      this.rafBridge = new TheatreRafBridge(
        `${this._opts.projectId}_${this._opts.sheetName}_external`,
      )
      // The caller's rafDriver is used directly for sequence.play().
      // Our rafBridge is still available for manual ticking if needed.
    } else {
      this.rafBridge = new TheatreRafBridge(
        `${this._opts.projectId}_${this._opts.sheetName}`,
      )
    }

    // Subscribe to sequence position changes → emit frames.
    const unsubPosition = onChange(this.sequence.pointer.position, (pos) => {
      this._position = pos
      this._emitFrame()
    }, this._getDriver())

    this._unsubs.push(unsubPosition)
  }

  // ── Object registration ─────────────────────────────────────────────────

  /**
   * Register a SheetObject for a cell.
   *
   * If the object already exists (e.g. from a prior `registerObject` call
   * with the same key), the existing handle is returned.
   *
   * @param key            Unique object key (typically cell_id).
   * @param initialValues  Optional initial prop values to apply.
   * @returns              TheatreObjectHandle for the registered object.
   */
  registerObject(
    key: string,
    initialValues?: Partial<CellPropValues>,
  ): TheatreObjectHandle {
    if (this._objects.has(key)) return this._objects.get(key)!

    const sheetObj = this.sheet.object(key, DEFAULT_CELL_PROPS)

    if (initialValues) {
      sheetObj.initialValue = initialValues as any
    }

    const handle = new TheatreObjectHandle(key, sheetObj, this._getDriver())
    this._objects.set(key, handle)

    return handle
  }

  /**
   * Unregister an object and detach it from the sheet.
   *
   * @param key  The object key to remove.
   */
  unregisterObject(key: string): void {
    const handle = this._objects.get(key)
    if (handle) {
      handle.dispose()
      this.sheet.detachObject(key)
      this._objects.delete(key)
    }
  }

  /**
   * Get the TheatreObjectHandle for a given key.
   * Returns undefined if the key hasn't been registered.
   */
  getObject(key: string): TheatreObjectHandle | undefined {
    return this._objects.get(key)
  }

  /** All registered object keys. */
  get objectKeys(): string[] {
    return [...this._objects.keys()]
  }

  /** Number of registered objects. */
  get objectCount(): number {
    return this._objects.size
  }

  // ── Playback control ────────────────────────────────────────────────────

  /**
   * Play the sequence.
   *
   * @param opts  Playback options (range, rate, direction, iterationCount).
   * @returns     Promise that resolves to `true` when playback completes,
   *              or `false` if interrupted by pause().
   */
  async play(opts: {
    range?:          IPlaybackRange
    rate?:           number
    direction?:      IPlaybackDirection
    iterationCount?: number
  } = {}): Promise<boolean> {
    if (this._destroyed) return false

    return this.sequence.play({
      range:          opts.range          ?? [0, this._opts.duration],
      rate:           opts.rate           ?? this._opts.defaultRate,
      direction:      opts.direction      ?? 'normal',
      iterationCount: opts.iterationCount ?? 1,
      rafDriver:      this._getDriver(),
    })
  }

  /** Pause the sequence. */
  pause(): void {
    this.sequence.pause()
  }

  /**
   * Seek to a specific position on the timeline.
   * Pauses any active playback.
   *
   * @param position  Position in sequence units (0 … duration).
   */
  seek(position: number): void {
    this.sequence.position = Math.max(0, Math.min(this._opts.duration, position))
  }

  /** Current sequence position. */
  get position(): number {
    return this._position
  }

  /** True if the sequence is currently playing. */
  get playing(): boolean {
    return val(this.sequence.pointer.playing)
  }

  /** Sequence duration. */
  get duration(): number {
    return this._opts.duration
  }

  // ── Frame callbacks ─────────────────────────────────────────────────────

  /**
   * Register a callback fired on every animation frame.
   *
   * The callback receives a TheatreFrame with all registered objects'
   * interpolated prop values at the current sequence position.
   *
   * @returns Unsubscribe function.
   */
  onFrame(cb: TheatreFrameCallback): () => void {
    this._frameCallbacks.add(cb)
    return () => { this._frameCallbacks.delete(cb) }
  }

  // ── Manual tick (for external frame loops) ──────────────────────────────

  /**
   * Manually advance Theatre.js by one tick.
   *
   * Call this from a PixiJS Ticker, EpochTicker, or custom rAF loop.
   * Theatre.js onChange listeners fire synchronously within this call,
   * which in turn triggers onFrame callbacks.
   *
   * @param timeMs  Current time in ms (`performance.now()`).
   * @returns       Delta time in ms since last tick.
   */
  tick(timeMs: number): number {
    return this.rafBridge.tick(timeMs)
  }

  // ── Snapshot ────────────────────────────────────────────────────────────

  /**
   * Take a snapshot of all objects' current prop values.
   */
  snapshot(): TimelineSnapshot {
    const objects = new Map<string, CellPropValues>()
    for (const [key, handle] of this._objects) {
      objects.set(key, handle.snapshot())
    }
    return { position: this._position, objects }
  }

  // ── Epoch helpers ─────────────────────────────────────────────────────

  /**
   * Batch-register objects from an array of EpochParamJSON.
   *
   * For each param, registers a SheetObject (if not already registered)
   * and sets its initial values from the param data.
   *
   * @param params  Array of EpochParamJSON objects.
   */
  registerFromEpochParams(params: EpochParamJSON[]): void {
    for (const p of params) {
      const { r, g, b } = hexToRgb(p.fill_color)
      this.registerObject(p.cell_id, {
        x: p.bbox.x,
        y: p.bbox.y,
        w: p.bbox.w,
        h: p.bbox.h,
        opacity: p.opacity,
        z: p.bbox.z ?? p.z ?? 3,
        r, g, b,
        glowIntensity: 1,
      })
    }
  }

  /**
   * Play a range corresponding to a single epoch transition.
   *
   * Convenience method: plays from `epochIndex` to `epochIndex + 1`.
   *
   * @param epochIndex  Starting epoch (integer).
   * @param rate        Playback rate (default: configured defaultRate).
   */
  async playEpochTransition(epochIndex: number, rate?: number): Promise<boolean> {
    return this.play({
      range: [epochIndex, epochIndex + 1],
      rate:  rate ?? this._opts.defaultRate,
    })
  }

  /**
   * Play through all epochs sequentially.
   *
   * @param rate        Playback rate.
   * @param direction   Playback direction.
   */
  async playAll(rate?: number, direction?: IPlaybackDirection): Promise<boolean> {
    return this.play({
      range:     [0, this._opts.duration],
      rate:      rate ?? this._opts.defaultRate,
      direction: direction ?? 'normal',
    })
  }

  /**
   * Loop playback continuously.
   *
   * @param rate  Playback rate.
   */
  async loop(rate?: number): Promise<boolean> {
    return this.play({
      range:          [0, this._opts.duration],
      rate:           rate ?? this._opts.defaultRate,
      iterationCount: Infinity,
    })
  }

  /**
   * Play with ping-pong (alternate) direction.
   *
   * @param rate  Playback rate.
   */
  async pingPong(rate?: number): Promise<boolean> {
    return this.play({
      range:          [0, this._opts.duration],
      rate:           rate ?? this._opts.defaultRate,
      direction:      'alternate',
      iterationCount: Infinity,
    })
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  /**
   * Destroy the timeline — pause sequence, dispose all handles, remove listeners.
   */
  destroy(): void {
    if (this._destroyed) return
    this._destroyed = true

    this.sequence.pause()

    for (const unsub of this._unsubs) unsub()
    this._unsubs.length = 0

    for (const handle of this._objects.values()) {
      handle.dispose()
    }
    this._objects.clear()

    this._frameCallbacks.clear()
  }

  // ── Private ─────────────────────────────────────────────────────────────

  /** Get the IRafDriver to use for Theatre.js calls. */
  private _getDriver(): IRafDriver {
    return this._opts.rafDriver ?? this.rafBridge.driver
  }

  /** Emit a TheatreFrame to all registered callbacks. */
  private _emitFrame(): void {
    if (this._frameCallbacks.size === 0) return

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now()
    const deltaMs = this._lastTickMs > 0 ? now - this._lastTickMs : 0
    this._lastTickMs = now

    const objects = new Map<string, CellPropValues>()
    for (const [key, handle] of this._objects) {
      objects.set(key, handle.value)
    }

    const frame: TheatreFrame = {
      position: this._position,
      playing:  this.playing,
      deltaMs,
      objects,
    }

    for (const cb of this._frameCallbacks) {
      try {
        cb(frame)
      } catch (err) {
        console.warn(`[TheatreTimeline] onFrame callback error:`, err)
      }
    }
  }
}

// ─── Factory functions ──────────────────────────────────────────────────────

/**
 * createTheatreTimeline — create a TheatreTimeline with optional keyframe data.
 *
 * Convenience factory that optionally accepts a KeyframeBuilder or
 * pre-built OnDiskState, constructs the timeline, and registers all
 * objects found in the keyframe data.
 *
 * @param options     TheatreTimeline options.
 * @param keyframes   Optional KeyframeBuilder or pre-built state.
 * @returns           Configured TheatreTimeline with objects registered.
 *
 * @example
 * ```ts
 * const builder = new KeyframeBuilder('epoch')
 *   .fromEpochParams(0, epoch0)
 *   .fromEpochParams(1, epoch1)
 *
 * const timeline = createTheatreTimeline(
 *   { sheetName: 'epoch', duration: 1 },
 *   builder,
 * )
 *
 * timeline.onFrame(({ objects }) => { ... })
 * timeline.play()
 * ```
 */
export function createTheatreTimeline(
  options:   TheatreTimelineOptions = {},
  keyframes?: KeyframeBuilder | Record<string, unknown>,
): TheatreTimeline {

  let state: Record<string, unknown> | undefined
  let objectKeysToRegister: string[] = []

  if (keyframes instanceof KeyframeBuilder) {
    state = keyframes.build()
    objectKeysToRegister = keyframes.objectKeys
  } else if (keyframes && typeof keyframes === 'object') {
    state = keyframes
  }

  const timeline = new TheatreTimeline({
    ...options,
    state: state ?? options.state,
  })

  // Auto-register objects found in the keyframe data.
  for (const key of objectKeysToRegister) {
    timeline.registerObject(key)
  }

  return timeline
}

/**
 * createEpochTheatreTimeline — shortcut for epoch-based timelines.
 *
 * Accepts epoch snapshots as `Map<epochIndex, EpochParamJSON[]>`, builds
 * keyframes, and creates a TheatreTimeline ready for playback.
 *
 * @param epochs   Map of epochIndex → array of cell params for that epoch.
 * @param options  TheatreTimeline options override.
 * @returns        TheatreTimeline configured for epoch playback.
 *
 * @example
 * ```ts
 * const timeline = createEpochTheatreTimeline(
 *   new Map([
 *     [0, epoch0Params],
 *     [1, epoch1Params],
 *     [2, epoch2Params],
 *   ]),
 *   { easing: 'spring' },
 * )
 *
 * await timeline.playAll()
 * ```
 */
export function createEpochTheatreTimeline(
  epochs:  Map<number, EpochParamJSON[]>,
  options: Omit<TheatreTimelineOptions, 'state' | 'duration'> = {},
): TheatreTimeline {

  const sheetName = options.sheetName ?? 'epoch'
  const easing    = options.easing    ?? 'ease'

  const builder = new KeyframeBuilder(sheetName, easing)

  let maxEpoch = 0
  for (const [epochIndex, params] of epochs) {
    builder.fromEpochParams(epochIndex, params)
    if (epochIndex > maxEpoch) maxEpoch = epochIndex
  }

  return createTheatreTimeline(
    {
      ...options,
      sheetName,
      duration: maxEpoch,
    },
    builder,
  )
}

// ─── Re-exports ─────────────────────────────────────────────────────────────

/** Re-export Theatre.js core functions for consumers. */
export { getProject, onChange, val, types, createRafDriver }

/** Re-export Theatre.js types for consumers. */
export type {
  IProject,
  ISheet,
  ISheetObject,
  ISequence,
  IRafDriver,
  IPlaybackDirection,
  IPlaybackRange,
  BasicKeyframe,
}

/** Re-export Atom from dataverse for advanced reactive use. */
export { Atom }

/** Re-export easing handles for external keyframe construction. */
export { EASING_HANDLES, DEFAULT_CELL_PROPS, DEFAULT_VALUES, PROP_KEYS }
