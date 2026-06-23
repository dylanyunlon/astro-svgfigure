/**
 * theatre-epoch-cell-bridge.ts — Theatre.js epoch 时间线 ↔ PixiJS Ticker 桥接  (M058)
 *
 * 功能：epoch N → epoch N+1 切换时，Theatre.js sequence.play() 驱动所有 cell
 * 的平滑过渡动画，每帧从 SheetObject.props 读取当前插值值更新 PixiJS Container。
 *
 * ─── 架构 ──────────────────────────────────────────────────────────────────────
 *
 *   EpochTimeline  (theatre-epoch-timeline.ts)
 *   │  ├─ IProject / ISheet "EpochMaster"
 *   │  ├─ ISequence  0 … N  (1 position unit = 1 epoch)
 *   │  └─ ISheetObject per cell  props: { x, y, w, h, opacity, bloomStrength, z, r, g, b }
 *   │       Theatre.js 自身 bezier 引擎在 keyframe 之间插值，我们只需 val(obj.props)
 *   │
 *   EpochCellBridge  (本文件)
 *   │  ├─ 持有 EpochTimeline 引用
 *   │  ├─ 持有 Map<cell_id, Container> — PixiJS Container 注册表
 *   │  ├─ 注册 EpochTimeline.onFrame() 回调
 *   │  │   → 每帧读取插值后的 CellState[] → 更新 Container.position / .alpha / tint
 *   │  ├─ advanceEpoch(rawCells) — 代理到 EpochTimeline.advanceEpoch()
 *   │  │   Theatre.js sequence.play(range=[N-1,N]) 驱动 0→1 过渡
 *   │  └─ PixiJS Ticker 对接选项：可挂到 app.ticker 上替代内部 rAF
 *
 * ─── Cell props 插值 ────────────────────────────────────────────────────────────
 *
 *   每个 cell 是 EpochMaster sheet 上的一个 ISheetObject，props schema：
 *     { x, y, w, h, opacity, bloomStrength, z, r, g, b }
 *
 *   Theatre.js bezier 引擎在 sequence.play() 期间对每帧每个 prop 做插值。
 *   本模块在 onFrame 回调（Theatre.js onChange 触发）中读取当前值并写入 PixiJS：
 *     container.position.set(v.x, v.y)
 *     container.alpha = v.opacity
 *     container.tint  = rgbToPixiTint(v.r, v.g, v.b)   // 仅当 colorInterpolation=true
 *     container.scale.set(v.w / origW, v.h / origH)     // bbox lerp via scale
 *
 * ─── bloomStrength 对接 ──────────────────────────────────────────────────────────
 *
 *   每个 container 的第一个 child 是 glow Graphics（见 pixi-cell-renderer.ts）。
 *   glow.__bloomFilter  是 AdvancedBloomFilter 实例。
 *   我们将 SheetObject bloomStrength prop 写入 __bloomFilter.bloomScale 覆盖
 *   pixi-cell-renderer 的脉冲动画基准值（__bloomFilterBaseScale）。
 *
 * ─── epoch 切换流程 ──────────────────────────────────────────────────────────────
 *
 *   1. advanceEpoch(rawCells) 被调用（新 epoch 完成）
 *   2. 代理到 EpochTimeline.advanceEpoch(rawCells, rate)
 *      → 内部：注册新 SheetObject（若有新 cell_id），追加 snapshot，
 *               调用 sequence.play({ range: [N-1, N], rate })
 *   3. Theatre.js 开始在 range 内插值所有 SheetObject props
 *   4. onChange(sequence.pointer.position) 每帧触发 EpochTimeline._emitFrame()
 *   5. EpochTimeline.onFrame 回调收到 EpochFrame{ cells: CellState[] }
 *   6. EpochCellBridge._onFrame() 遍历 cells：
 *        container.position.set(x, y)
 *        container.alpha = opacity
 *        tint / scale / bloomScale 按选项写入
 *   7. 下一帧 PixiJS Ticker 渲染更新后的 Container tree
 *
 * Upstream 参考:
 *   upstream/theatre-js/core/src/coreExports.ts    — val(), onChange()
 *   upstream/pixijs-engine/src/ticker/Ticker.ts    — Ticker
 *   src/lib/renderers/theatre-epoch-timeline.ts     — EpochTimeline / CellState
 *   src/lib/renderers/pixi-cell-renderer.ts         — buildCellContainer / __bloomFilter
 */

import type { Application } from '../../../upstream/pixijs-engine/src/app/Application'
import type { Container }   from '../../../upstream/pixijs-engine/src/scene/container/Container'
import { Ticker }           from '../../../upstream/pixijs-engine/src/ticker/Ticker'

import {
  createEpochTimeline,
  type EpochTimeline,
  type EpochFrame,
  type CellState,
  type CellSheetProps,
  type EpochSheet,
  type EpochSnapshotsJSON,
  type RawCellState,
  type EpochTimelineOptions,
} from './theatre-epoch-timeline'

import { GlowFilter } from '../../../upstream/pixijs-filters-v2/src/glow'

// ─── Re-export for consumers ───────────────────────────────────────────────────
export type {
  EpochTimeline,
  EpochFrame,
  CellState,
  EpochSheet,
  EpochSnapshotsJSON,
  RawCellState,
  EpochTimelineOptions,
}

// ─── Public types ──────────────────────────────────────────────────────────────

/**
 * Options for EpochCellBridge construction.
 */
export interface EpochCellBridgeOptions {
  /**
   * Whether to interpolate cell color (tint) from Theatre.js r/g/b props.
   * Default: true
   * Set false to keep the static species palette colours from pixi-cell-renderer.
   */
  colorInterpolation?: boolean

  /**
   * Whether to scale container to match Theatre.js interpolated w/h props.
   * Default: false — w/h changes are decorative metadata; most cells keep fixed geometry.
   * When true: container.scale.x = v.w / origW, container.scale.y = v.h / origH
   */
  bboxScale?: boolean

  /**
   * Whether to forward Theatre.js bloomStrength prop to the AdvancedBloomFilter
   * mounted on the glow child (pixi-cell-renderer glow sprite).
   * Default: true
   */
  bloomBridge?: boolean

  /**
   * Options forwarded to createEpochTimeline().
   * Controls easing, projectId, defaultRate.
   */
  timelineOptions?: EpochTimelineOptions

  /**
   * Playback rate (epochs per second) used for advanceEpoch().
   * Default: 1
   */
  defaultRate?: number
}

/**
 * Per-container original dimensions — saved at registration time so
 * bboxScale mode can compute the scale multiplier accurately.
 */
interface ContainerMeta {
  container: Container
  /** Original width at registration (from bbox or container bounds). */
  origW: number
  /** Original height at registration (from bbox or container bounds). */
  origH: number
}

// ─── Colour helper ─────────────────────────────────────────────────────────────

/**
 * Pack 0–255 r/g/b channels into a PixiJS-compatible 0xRRGGBB integer.
 * Used for container.tint assignment.
 */
function rgbToPixiTint(r: number, g: number, b: number): number {
  return (
    ((Math.round(Math.max(0, Math.min(255, r))) & 0xff) << 16) |
    ((Math.round(Math.max(0, Math.min(255, g))) & 0xff) << 8)  |
    (Math.round(Math.max(0, Math.min(255, b))) & 0xff)
  )
}

// ─── EpochCellBridge ──────────────────────────────────────────────────────────

/**
 * EpochCellBridge — bridges Theatre.js EpochTimeline to PixiJS Container tree.
 *
 * Responsibilities:
 *   1. Owns the EpochTimeline instance (or accepts an externally created one).
 *   2. Maintains a registry of cell_id → { container, origW, origH }.
 *   3. On each Theatre.js frame callback, writes interpolated prop values to
 *      the corresponding PixiJS Container (position, alpha, tint, scale, bloom).
 *   4. advanceEpoch() proxies to EpochTimeline.advanceEpoch() so callers don't
 *      need to import theatre-epoch-timeline directly.
 *   5. Optionally hooks into a PixiJS app.ticker so Theatre.js updates are
 *      flushed every Ticker frame, not just on rAF ticks.
 *
 * @example
 * ```ts
 * import epochSnapshots from '../../../channels/physics/epoch_snapshots.json'
 *
 * const bridge = new EpochCellBridge(epochSnapshots, { easing: 'spring' })
 *
 * // Register PixiJS containers — call after buildCellContainer()
 * bridge.registerContainer('cell_sa_q', container, { w: 120, h: 60 })
 *
 * // Wire to PixiJS Application ticker
 * bridge.attachToApp(app)
 *
 * // Play from epoch 0 → N
 * bridge.play(0.5)   // 0.5 epochs/second
 *
 * // When a new epoch arrives from the pubsub loop:
 * bridge.advanceEpoch(newRawCells)
 * ```
 */
export class EpochCellBridge {

  /** The underlying Theatre.js epoch timeline. */
  readonly timeline: EpochTimeline

  /** Container registry: cell_id → { container, origW, origH }. */
  private readonly _registry = new Map<string, ContainerMeta>()

  /** Resolved options (with defaults applied). */
  private readonly _opts: Required<EpochCellBridgeOptions>

  /** EpochTimeline.onFrame() unsubscribe handle. */
  private _unsubFrame: (() => void) | null = null

  /** Per-cell SheetObject.onValuesChange() unsubscribe handles (M160). */
  private readonly _unsubValues: Array<() => void> = []

  /** True once attachToApp() has been called. */
  private _attached = false

  /** Bound Ticker handler (for detach). */
  private _tickerHandler: ((ticker: Ticker) => void) | null = null

  /** PixiJS Application (if attached). */
  private _app: Application | null = null

  // ── Constructor ──────────────────────────────────────────────────────────────

  constructor(
    data:    EpochSnapshotsJSON | EpochSheet[] | EpochTimeline,
    opts:    EpochCellBridgeOptions = {},
  ) {
    // Accept either a pre-built EpochTimeline or raw snapshot data
    if (data && typeof (data as EpochTimeline).onFrame === 'function') {
      this.timeline = data as EpochTimeline
    } else {
      this.timeline = createEpochTimeline(
        data as EpochSnapshotsJSON | EpochSheet[],
        opts.timelineOptions ?? {},
      )
    }

    this._opts = {
      colorInterpolation: opts.colorInterpolation ?? true,
      bboxScale:          opts.bboxScale          ?? false,
      bloomBridge:        opts.bloomBridge         ?? true,
      timelineOptions:    opts.timelineOptions     ?? {},
      defaultRate:        opts.defaultRate         ?? 1,
    }

    // Subscribe to Theatre.js frame callbacks immediately.
    // Containers registered later will receive updates from the next frame.
    this._unsubFrame = this.timeline.onFrame((frame) => this._onFrame(frame))
  }

  // ── Container registry ───────────────────────────────────────────────────────

  /**
   * Register a PixiJS Container to be driven by Theatre.js for a given cell_id.
   *
   * @param cellId     Cell identifier — must match the EpochTimeline SheetObject name.
   * @param container  The PixiJS Container built by pixi-cell-renderer.buildCellContainer().
   * @param origSize   Original bbox dimensions { w, h }.  Defaults to 100×50 if omitted.
   *
   * @example
   * ```ts
   * const container = buildCellContainer(desc)
   * bridge.registerContainer(desc.cell_id, container, desc.bbox)
   * ```
   */
  registerContainer(
    cellId: string,
    container: Container,
    origSize?: { w: number; h: number },
  ): void {
    this._registry.set(cellId, {
      container,
      origW: origSize?.w ?? 100,
      origH: origSize?.h ?? 50,
    })

    // ── M160: per-cell SheetObject.onValuesChange → PixiJS Container ──────
    // Subscribe directly to the Theatre.js SheetObject so that every prop
    // change (driven by sequence.play() bezier interpolation) is forwarded
    // to the PixiJS Container without waiting for the batched onFrame cycle.
    this._subscribeSheetObject(cellId, container)
  }

  /**
   * Unregister a container (e.g. when the cell is removed from the stage).
   * Does NOT destroy the container — caller is responsible.
   */
  unregisterContainer(cellId: string): void {
    this._registry.delete(cellId)
  }

  /** Register multiple containers from a Map. */
  registerAll(
    entries: Map<string, { container: Container; w: number; h: number }>,
  ): void {
    for (const [cellId, { container, w, h }] of entries) {
      this.registerContainer(cellId, container, { w, h })
    }
  }

  // ── Playback control ─────────────────────────────────────────────────────────

  /**
   * Start playback of the master sequence from position 0 → epochCount-1.
   * @param rate  epochs per second (default: this._opts.defaultRate)
   */
  play(rate?: number): Promise<boolean> {
    return this.timeline.play(rate ?? this._opts.defaultRate)
  }

  /** Pause the master sequence. */
  pause(): void {
    this.timeline.pause()
  }

  /** Stop playback and reset to position 0. */
  stop(): void {
    this.timeline.stop()
  }

  /** Seek to an arbitrary fractional epoch position. */
  seek(pos: number): void {
    this.timeline.seek(pos)
  }

  /**
   * Advance to a new epoch — the core M058 method.
   *
   * Behaviour:
   *   1. Forwards rawCells to EpochTimeline.advanceEpoch(rawCells, rate).
   *   2. Theatre.js sequence.play({ range: [N-1, N], rate }) begins.
   *   3. Every frame, Theatre.js interpolates all SheetObject props.
   *   4. EpochTimeline.onFrame fires → EpochCellBridge._onFrame() reads
   *      interpolated values and writes them to registered PixiJS Containers.
   *
   * @param rawCells  Raw cell states from the new epoch (same format as EpochSheet.cells).
   * @param rate      Playback rate in epochs/second for this transition. Default: defaultRate.
   */
  advanceEpoch(rawCells: RawCellState[], rate?: number): Promise<boolean> {
    return this.timeline.advanceEpoch(rawCells, rate ?? this._opts.defaultRate)
  }

  // ── PixiJS App integration ───────────────────────────────────────────────────

  /**
   * Attach to a PixiJS Application's Ticker.
   *
   * This is optional — Theatre.js drives updates via its own rAF loop through
   * the onChange(sequence.position) subscription.  However, attaching to the
   * PixiJS Ticker ensures Theatre.js value reads are synchronised with the
   * PixiJS render frame, eliminating any sub-frame lag.
   *
   * The Ticker handler calls `this.timeline.getCurrentFrame()` and applies
   * the latest Theatre.js interpolated values to all registered containers,
   * ensuring PixiJS always renders the most up-to-date state before its own
   * draw call (which occurs in the LOW-priority TickerPlugin slot).
   *
   * @param app  Running PixiJS Application instance.
   */
  attachToApp(app: Application): void {
    if (this._attached) return
    this._attached = true
    this._app = app

    // Build a Ticker handler that reads Theatre.js current frame and
    // applies prop values to containers on every PixiJS tick.
    this._tickerHandler = (_ticker: Ticker) => {
      // Read the current interpolated frame from Theatre.js.
      // Theatre.js onChange already fires via its own scheduler; we call
      // getCurrentFrame() here to guarantee synchronous reads aligned with
      // the PixiJS render cycle, especially at high playback speeds.
      const frame = this.timeline.getCurrentFrame()
      this._applyFrame(frame.cells)
    }

    app.ticker.add(this._tickerHandler)
  }

  /**
   * Detach from the PixiJS Application's Ticker.
   * Theatre.js onFrame callbacks continue to fire; only the Ticker integration is removed.
   */
  detachFromApp(): void {
    if (!this._attached || !this._app || !this._tickerHandler) return
    this._app.ticker.remove(this._tickerHandler)
    this._attached      = false
    this._tickerHandler = null
    this._app           = null
  }

  // ── Current state accessors ──────────────────────────────────────────────────

  /** Current fractional epoch position (0 … epochCount-1). */
  get position(): number { return this.timeline.position }

  /** Total number of epochs in the timeline. */
  get epochCount(): number { return this.timeline.epochCount }

  /** True once Theatre.js project state has been hydrated. */
  get ready(): boolean { return this.timeline.ready }

  // ── Snapshot helpers ─────────────────────────────────────────────────────────

  /**
   * Immediately jump all registered containers to the static state of a given epoch.
   * No animation — useful for resetting to epoch 0 or previewing a specific epoch.
   */
  jumpToEpoch(epochIndex: number): void {
    const cells = this.timeline.readEpochSnapshot(epochIndex)
    this._applyFrame(cells)
    this.timeline.seek(epochIndex)
  }

  // ── Destruction ──────────────────────────────────────────────────────────────

  /**
   * Destroy the bridge — unsubscribes all callbacks.
   * Does NOT destroy the EpochTimeline or PixiJS containers (caller's responsibility).
   */
  destroy(): void {
    this._unsubFrame?.()
    this._unsubFrame = null
    // M160: unsubscribe all per-cell SheetObject.onValuesChange listeners
    for (const unsub of this._unsubValues) unsub()
    this._unsubValues.length = 0
    this.detachFromApp()
    this._registry.clear()
  }

  // ── Private: frame application ───────────────────────────────────────────────

  /**
   * Called by EpochTimeline.onFrame() every time Theatre.js updates its sequence position.
   * Simply delegates to _applyFrame() with the interpolated cell states.
   */
  private _onFrame(frame: EpochFrame): void {
    this._applyFrame(frame.cells)
  }

  /**
   * Core update loop — apply Theatre.js interpolated CellState[] to PixiJS Containers.
   *
   * For each cell in the frame:
   *   1. Bbox lerp  → container.position.set(v.x, v.y)
   *      (optional scale) → container.scale.set(v.w / origW, v.h / origH)
   *   2. Opacity fade → container.alpha = v.opacity
   *   3. Color interp → container.tint = rgbToPixiTint(v.r, v.g, v.b)   [if enabled]
   *   4. Bloom bridge → glow.__bloomFilter.bloomScale = v.bloomStrength  [if enabled]
   */
  private _applyFrame(cells: CellState[]): void {
    const { colorInterpolation, bboxScale, bloomBridge } = this._opts

    for (const cell of cells) {
      const meta = this._registry.get(cell.cell_id)
      if (!meta) continue

      const { container, origW, origH } = meta

      // ── 1. Bbox lerp: position ────────────────────────────────────────────
      container.position.set(cell.x, cell.y)

      // Optional: scale container to match interpolated w/h
      if (bboxScale && origW > 0 && origH > 0) {
        container.scale.set(
          cell.w / origW,
          cell.h / origH,
        )
      }

      // ── 2. Opacity fade ──────────────────────────────────────────────────
      // Theatre.js drives opacity via the SheetObject `opacity` prop.
      // Clamp to [0, 1] to guard against floating-point overshoot.
      container.alpha = Math.max(0, Math.min(1, cell.opacity))

      // ── 3. Color interpolation ───────────────────────────────────────────
      // Theatre.js interpolates r/g/b channels (0–255) independently.
      // We pack them into a PixiJS 0xRRGGBB tint and assign it.
      // Setting tint to 0xFFFFFF is a no-op in PixiJS (white = identity tint).
      if (colorInterpolation) {
        ;(container as any).tint = rgbToPixiTint(cell.r, cell.g, cell.b)
      }

      // ── 4. bloomStrength bridge ──────────────────────────────────────────
      // The glow Graphics is always children[0] of a pixi-cell-renderer container.
      // It holds __bloomFilter (AdvancedBloomFilter) and __bloomFilterBaseScale.
      // We write the Theatre.js bloomStrength prop to bloomScale, overriding the
      // static base value.  The Ticker-driven pulse animation in pixi-cell-renderer
      // computes: bloomScale = base * (1 + amp * sin(...))
      // We write our Theatre.js value as the new effective base scale:
      //   __bloomFilterBaseScale = cell.bloomStrength
      // so the pulse amplitude modulates around the Theatre.js keyframed value.
      if (bloomBridge) {
        const glowChild = (container as any).children?.[0] as any
        if (glowChild) {
          const bloomFilter = glowChild.__bloomFilter
          if (bloomFilter) {
            // Update the base scale so the pulse animation respects the Theatre.js value
            glowChild.__bloomFilterBaseScale = cell.bloomStrength
            // Also write directly to bloomScale so the current frame is correct
            // before the Ticker pulse loop runs for this frame.
            bloomFilter.bloomScale = cell.bloomStrength
          }
        }
      }
    }
  }

  // ── Private: per-cell SheetObject subscription (M160) ─────────────────────

  /**
   * Subscribe to a single cell's SheetObject.onValuesChange() and forward
   * interpolated prop values directly to the PixiJS Container.
   *
   * This is the M160 "reactive bridge" — each cell gets its own listener so
   * Theatre.js → PixiJS updates are immediate and per-object, rather than
   * waiting for the batched EpochFrame cycle.
   *
   * Props forwarded:
   *   x, y           → container.position.set(x, y)
   *   opacity        → container.alpha
   *   glowIntensity  → find GlowFilter in container.filters, set outerStrength
   */
  private _subscribeSheetObject(cellId: string, container: Container): void {
    const sheetObj = this.timeline.getObject(cellId)
    if (!sheetObj) return

    const unsub = sheetObj.onValuesChange((values: CellSheetProps) => {
      // ── Position ──────────────────────────────────────────────────────
      container.position.set(values.x, values.y)

      // ── Opacity ───────────────────────────────────────────────────────
      container.alpha = Math.max(0, Math.min(1, values.opacity))

      // ── Glow intensity → GlowFilter.outerStrength ─────────────────────
      // Walk container.filters to find any GlowFilter instance and update
      // its outerStrength to match the Theatre.js glowIntensity prop.
      // Covers both __baselineGlowFilter and any other GlowFilter that may
      // have been added by pixi-cell-renderer or pixi-filters-registry.
      const filters = (container.filters as unknown[] | null) ?? []
      for (const f of filters) {
        if (f instanceof GlowFilter) {
          f.outerStrength = values.glowIntensity
        }
      }
    })

    this._unsubValues.push(unsub)
  }
}

// ─── Factory function ──────────────────────────────────────────────────────────

/**
 * createEpochCellBridge — create an EpochCellBridge from epoch snapshot data.
 *
 * Convenience wrapper over `new EpochCellBridge(data, opts)`.
 *
 * @param data     Either the full JSON wrapper `{ snapshots: EpochSheet[] }` or
 *                 a plain `EpochSheet[]` array, or a pre-built EpochTimeline.
 * @param opts     Bridge options (colorInterpolation, bboxScale, bloomBridge, …).
 *
 * @example
 * ```ts
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 * import { createEpochCellBridge } from './theatre-epoch-cell-bridge'
 *
 * const bridge = createEpochCellBridge(epochData, {
 *   colorInterpolation: true,
 *   bboxScale:          false,
 *   bloomBridge:        true,
 *   timelineOptions:    { easing: 'spring', defaultRate: 0.5 },
 * })
 *
 * // After building cells with pixi-cell-renderer:
 * for (const desc of cellDescs) {
 *   const container = buildCellContainer(desc)
 *   app.stage.addChild(container)
 *   bridge.registerContainer(desc.cell_id, container, desc.bbox)
 * }
 *
 * bridge.attachToApp(app)
 * bridge.play()
 *
 * // When a new epoch arrives from the pubsub channel:
 * pubsub.on('epoch', ({ cells }) => bridge.advanceEpoch(cells))
 * ```
 */
export function createEpochCellBridge(
  data: EpochSnapshotsJSON | EpochSheet[] | EpochTimeline,
  opts: EpochCellBridgeOptions = {},
): EpochCellBridge {
  return new EpochCellBridge(data, opts)
}

// ─── renderCellGraphWithEpochBridge ──────────────────────────────────────────────

/**
 * renderCellGraphWithEpochBridge — high-level convenience function.
 *
 * Initialises a PixiJS Application and an EpochCellBridge together, wiring
 * Theatre.js epoch transitions to the PixiJS stage in a single call.
 *
 * This is the primary M058 integration point for page-level code.
 *
 * @param canvas       HTMLCanvasElement to render into.
 * @param cells        Initial CellDescriptor[] (epoch 0).  Each cell must have
 *                     a matching entry in `epochData`.
 * @param epochData    Epoch snapshot data for Theatre.js timeline.
 * @param buildCell    Function to build a PixiJS Container from a descriptor.
 *                     Typically `buildCellContainer` from pixi-cell-renderer.
 * @param opts         Bridge options.
 *
 * @returns  { bridge, stop }
 *   bridge — EpochCellBridge for playback control and container registration.
 *   stop   — cleanup function.
 *
 * @example
 * ```ts
 * import { renderCellGraphWithEpochBridge } from './theatre-epoch-cell-bridge'
 * import { buildCellContainer } from './pixi-cell-renderer'
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 *
 * const { bridge, stop } = await renderCellGraphWithEpochBridge(
 *   canvas, cells, epochData, buildCellContainer,
 *   { colorInterpolation: true, timelineOptions: { easing: 'spring' } }
 * )
 *
 * bridge.play(0.5)
 *
 * // In the cell-pubsub loop:
 * onEpoch((rawCells) => bridge.advanceEpoch(rawCells))
 *
 * // Cleanup:
 * stop()
 * ```
 */
export async function renderCellGraphWithEpochBridge<TDesc extends {
  cell_id: string;
  bbox: { x: number; y: number; w: number; h: number };
}>(
  canvas:     HTMLCanvasElement,
  cells:      TDesc[],
  epochData:  EpochSnapshotsJSON | EpochSheet[] | EpochTimeline,
  buildCell:  (desc: TDesc) => Container,
  opts:       EpochCellBridgeOptions = {},
): Promise<{
  bridge: EpochCellBridge
  stop:   () => void
}> {
  // Dynamic import to avoid circular deps — pixi-cell-renderer imports us indirectly
  const { Application } = await import('../../upstream/pixijs-engine/src/app/Application')

  const app = new Application()
  await app.init({
    canvas,
    width:           canvas.width,
    height:          canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias:       true,
    resolution:      typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1,
    autoDensity:     true,
  })

  app.stage.sortableChildren = true

  // ── Build and register all initial cells ───────────────────────────────
  const bridge = createEpochCellBridge(epochData, opts)

  for (const desc of cells) {
    const container = buildCell(desc)
    // Set initial alpha from Theatre.js position 0
    const snap = bridge.timeline.readEpochSnapshot(0)
    const snapCell = snap.find(c => c.cell_id === desc.cell_id)
    container.alpha = snapCell?.opacity ?? 1
    app.stage.addChild(container)
    bridge.registerContainer(desc.cell_id, container, desc.bbox)
  }

  // ── Attach bridge to PixiJS Ticker ────────────────────────────────────
  bridge.attachToApp(app)

  // ── Jump to epoch 0 (ensure initial state is applied) ─────────────────
  bridge.jumpToEpoch(0)

  return {
    bridge,
    stop: () => {
      bridge.destroy()
      app.destroy(false, { children: true })
    },
  }
}

// ─── Utility: build a CellDescriptor → container registration map ─────────────

/**
 * buildContainerRegistry — register all cells from a CellDescriptor[] into
 * an EpochCellBridge at once, given a Map of pre-built containers.
 *
 * Convenience helper for callers who already have a container map from
 * renderCellGraph() or their own build loop.
 *
 * @param bridge      EpochCellBridge to register into.
 * @param containers  Map of cell_id → PixiJS Container.
 * @param bboxes      Map of cell_id → { w, h } original dimensions.
 *
 * @example
 * ```ts
 * const containerMap = new Map<string, Container>()
 * for (const desc of cells) {
 *   const c = buildCellContainer(desc)
 *   containerMap.set(desc.cell_id, c)
 *   app.stage.addChild(c)
 * }
 *
 * const bboxMap = new Map(cells.map(d => [d.cell_id, d.bbox]))
 * buildContainerRegistry(bridge, containerMap, bboxMap)
 * ```
 */
export function buildContainerRegistry(
  bridge:     EpochCellBridge,
  containers: Map<string, Container>,
  bboxes:     Map<string, { w: number; h: number }>,
): void {
  for (const [cellId, container] of containers) {
    const bbox = bboxes.get(cellId)
    bridge.registerContainer(cellId, container, bbox)
  }
}

// ─── Epoch pubsub integration ──────────────────────────────────────────────────

/**
 * EpochPubSubBridge — wraps EpochCellBridge to auto-subscribe to a cell-pubsub
 * epoch event emitter.
 *
 * The pubsub loop (channels/epoch_controller.py → WS → client) delivers new
 * epoch snapshots.  This class listens on `emitter.on('epoch', handler)` and
 * calls bridge.advanceEpoch() for each new epoch.
 *
 * @example
 * ```ts
 * const pubsubBridge = new EpochPubSubBridge(bridge, epochEmitter)
 * // From now on, every 'epoch' event auto-advances the Theatre.js timeline.
 * pubsubBridge.destroy() // stop listening
 * ```
 */
export class EpochPubSubBridge {

  private readonly _bridge: EpochCellBridge
  private readonly _emitter: { on: Function; off: Function }
  private readonly _handler: (data: { cells: RawCellState[]; rate?: number }) => void

  constructor(
    bridge:  EpochCellBridge,
    emitter: { on: Function; off: Function },
    eventName: string = 'epoch',
  ) {
    this._bridge  = bridge
    this._emitter = emitter

    this._handler = (data: { cells: RawCellState[]; rate?: number }) => {
      if (Array.isArray(data?.cells)) {
        bridge.advanceEpoch(data.cells, data.rate)
          .catch((err: unknown) => {
            console.warn('[EpochPubSubBridge] advanceEpoch error:', err)
          })
      }
    }

    emitter.on(eventName, this._handler)
  }

  /** Stop listening for epoch events. */
  destroy(eventName: string = 'epoch'): void {
    this._emitter.off(eventName, this._handler)
  }
}
