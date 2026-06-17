/**
 * theatre-epoch-timeline.ts — Theatre.js per-cell SheetObject + epoch_params keyframes (M152)
 *
 * M152: Rewritten to use per-cell SheetObjects on a single "epoch" sheet
 * within the "astro-svgfigure" project, with keyframes loaded directly
 * from channels/convergence/epoch_params/{epochIndex}/{cell}.json.
 *
 * Architecture:
 *
 *   Project "astro-svgfigure"
 *   └── Sheet "epoch"
 *       ├── SheetObject "input_embed"   props: { x, y, width, height, opacity, glowIntensity }
 *       ├── SheetObject "self_attn"     props: { x, y, width, height, opacity, glowIntensity }
 *       ├── SheetObject "add_norm1"     ...
 *       ├── SheetObject "add_norm2"     ...
 *       ├── SheetObject "ffn"           ...
 *       ├── SheetObject "pos_encode"    ...
 *       └── SheetObject "output"        ...
 *
 * Each SheetObject's props are keyframed at position 0 (epoch 0) and
 * position 1 (epoch 1).  Theatre.js's built-in bezier interpolation engine
 * drives the transition via sheet.sequence.play({ range: [0, 1] }).
 *
 * Epoch param files:
 *   channels/convergence/epoch_params/0/{cell_id}.json  — epoch 0 values
 *   channels/convergence/epoch_params/1/{cell_id}.json  — epoch 1 values
 *
 * Each JSON has shape:
 *   { cell_id, species, bbox: { x, y, w, h, z }, opacity, fill_color, ... }
 *
 * Mapping to SheetObject props:
 *   x             ← bbox.x
 *   y             ← bbox.y
 *   width         ← bbox.w
 *   height        ← bbox.h
 *   opacity       ← opacity
 *   glowIntensity ← 1 (default; no source field in epoch_params)
 *
 * Usage:
 *   const tl = await createEpochTimeline()
 *   tl.play()                          // animate epoch 0 → epoch 1
 *   tl.seek(0.5)                       // halfway between epochs
 *   tl.onFrame(({ cells }) => …)       // interpolated CellState[] each tick
 */

// ─── Vendored Theatre.js core ─────────────────────────────────────────────────

import {
  getProject,
  onChange,
  val,
  types,
} from '../../../upstream/theatre-js/core/src/coreExports'

import type {
  IProject,
  ISheet,
  ISheetObject,
  ISequence,
} from '../../../upstream/theatre-js/core/src/types/public'

// ─── Epoch params path ────────────────────────────────────────────────────────

/**
 * Base path to epoch_params directory, relative to project root.
 * At build time this resolves via Astro / Vite import.meta or Node fs.
 */
const EPOCH_PARAMS_BASE = 'channels/convergence/epoch_params'

// ─── Public types ─────────────────────────────────────────────────────────────

/**
 * Raw epoch param JSON as stored in epoch_params/{epoch}/{cell}.json.
 */
export interface EpochParamJSON {
  cell_id:      string
  species:      string
  bbox:         { x: number; y: number; w: number; h: number; z?: number }
  z?:           number
  opacity:      number
  fill_color:   string
  stroke_color: string
  label:        string
  font_size:    number
  species_params?: Record<string, unknown>
  epoch?:       number
  shadow?:      { dx: number; dy: number; blur: number; opacity: number }
}

/**
 * SheetObject prop values — the Theatre.js-managed per-cell props.
 * M152 spec: { x, y, width, height, opacity, glowIntensity }
 */
export interface CellSheetProps {
  x:             number
  y:             number
  width:         number
  height:        number
  opacity:       number
  glowIntensity: number
}

/**
 * One cell's visual state at a given instant.
 * Includes the interpolated SheetObject props plus metadata.
 */
export interface CellState {
  cell_id:       string
  x:             number
  y:             number
  width:         number
  height:        number
  opacity:       number
  glowIntensity: number
}

/** Live interpolated frame emitted on each animation tick. */
export interface EpochFrame {
  /** Fractional sequence position (0 … 1). */
  position:   number
  /** 0–1 blend factor (0 = epoch 0, 1 = epoch 1). */
  blend:      number
  /** Interpolated cell states — ready to hand to renderer. */
  cells:      CellState[]
}

export type FrameCallback = (frame: EpochFrame) => void

/** Easing presets for keyframe handles. */
export type EasingPreset = 'linear' | 'ease' | 'spring'

export interface EpochTimelineOptions {
  /**
   * Easing between epoch keyframes.  Default: 'ease'.
   * 'linear'  — [0,0,1,1]
   * 'ease'    — [0.25,0.1,0.25,1] (CSS ease)
   * 'spring'  — [0.34,1.56,0.64,1] (overshoot spring-ish)
   */
  easing?: EasingPreset
  /**
   * Playback rate in epochs-per-second.  Default: 1.
   */
  defaultRate?: number
}

// ─── Theatre prop schema ──────────────────────────────────────────────────────

/**
 * M152 spec: per-cell SheetObject props.
 *   x, y, width, height  — spatial
 *   opacity               — number(1, { range: [0, 1] })
 *   glowIntensity         — number(1, { range: [0, 3] })
 */
const CELL_PROPS = {
  x:             types.number(0,   { range: [-4000, 4000] }),
  y:             types.number(0,   { range: [-4000, 4000] }),
  width:         types.number(100, { range: [0, 2000] }),
  height:        types.number(50,  { range: [0, 2000] }),
  opacity:       types.number(1,   { range: [0, 1] }),
  glowIntensity: types.number(1,   { range: [0, 3] }),
} as const

// ─── Easing handle presets ────────────────────────────────────────────────────

const EASING_HANDLES: Record<EasingPreset, [number, number, number, number]> = {
  linear: [0, 0, 1, 1],
  ease:   [0.25, 0.1, 0.25, 1],
  spring: [0.34, 1.56, 0.64, 1],
}

// ─── Helpers: extract SheetObject values from epoch param JSON ───────────────

/**
 * Extract the SheetObject prop values from a raw epoch param JSON.
 */
function epochParamToProps(param: EpochParamJSON): CellSheetProps {
  return {
    x:             param.bbox.x,
    y:             param.bbox.y,
    width:         param.bbox.w,
    height:        param.bbox.h,
    opacity:       param.opacity,
    glowIntensity: 1,   // default; epoch_params don't carry glow data
  }
}

// ─── Theatre.js OnDiskState builder ──────────────────────────────────────────

/**
 * Build the Theatre.js `OnDiskState` for the "astro-svgfigure" project.
 *
 * The "epoch" sheet receives one keyframe per prop per cell at position 0
 * (epoch 0 values) and position 1 (epoch 1 values).  Theatre.js's own
 * bezier interpolation engine drives the cross-epoch blending.
 *
 * State format reference: upstream/theatre-js/core/src/types/private/core.ts
 *   definitionVersion = '0.4.0'
 */
function buildTheatreState(
  epoch0Params: Map<string, CellSheetProps>,
  epoch1Params: Map<string, CellSheetProps>,
  easing:       EasingPreset,
): Record<string, unknown> {

  const handles = EASING_HANDLES[easing]

  // Collect all unique cell IDs across both epochs.
  const allCellIds = new Set<string>([
    ...epoch0Params.keys(),
    ...epoch1Params.keys(),
  ])

  type KFRecord = Record<string, {
    id: string
    position: number
    value: number
    handles: [number, number, number, number]
    connectedRight: boolean
    type: 'bezier'
  }>

  const tracksByObject: Record<string, {
    trackIdByPropPath: Record<string, string>
    trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }>
  }> = {}

  const PROPS: (keyof CellSheetProps)[] = ['x', 'y', 'width', 'height', 'opacity', 'glowIntensity']

  for (const cellId of allCellIds) {
    const trackIdByPropPath: Record<string, string> = {}
    const trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }> = {}

    // Get values for both epochs; fall back to defaults if cell absent.
    const v0 = epoch0Params.get(cellId) ?? { x: 0, y: 0, width: 100, height: 50, opacity: 0, glowIntensity: 1 }
    const v1 = epoch1Params.get(cellId) ?? { x: 0, y: 0, width: 100, height: 50, opacity: 0, glowIntensity: 1 }

    for (const prop of PROPS) {
      const trackId  = `${cellId}__${prop}`
      const propPath = JSON.stringify([prop])    // e.g. '["x"]', '["opacity"]'
      trackIdByPropPath[propPath] = trackId

      const keyframes: KFRecord = {}

      // Keyframe at position 0 — epoch 0 values
      keyframes[`${cellId}_${prop}_e0`] = {
        id:             `${cellId}_${prop}_e0`,
        position:       0,
        value:          v0[prop],
        handles,
        connectedRight: true,
        type:           'bezier',
      }

      // Keyframe at position 1 — epoch 1 values
      keyframes[`${cellId}_${prop}_e1`] = {
        id:             `${cellId}_${prop}_e1`,
        position:       1,
        value:          v1[prop],
        handles,
        connectedRight: false,
        type:           'bezier',
      }

      trackData[trackId] = { type: 'BasicKeyframedTrack', keyframes }
    }

    tracksByObject[cellId] = { trackIdByPropPath, trackData }
  }

  const epochSheetState = {
    staticOverrides: { byObject: {} },
    sequence: {
      type:            'PositionalSequence',
      length:          1,               // positions 0 → 1  (epoch 0 → epoch 1)
      subUnitsPerUnit: 30,              // 30 fps grid for Studio scrubbing
      tracksByObject,
    },
  }

  return {
    definitionVersion: '0.4.0',
    revisionHistory:   ['astro-svgfigure-m152-epoch-params'],
    sheetsById: {
      epoch: epochSheetState,
    },
  }
}

// ─── Core class ───────────────────────────────────────────────────────────────

class EpochTimeline {

  /** Theatre.js project — "astro-svgfigure". */
  private readonly _project: IProject

  /** The "epoch" sheet — holds keyframed sequence for all cells. */
  private readonly _epochSheet: ISheet

  /** Master sequence (= _epochSheet.sequence). Length = 1 (position 0 → 1). */
  private readonly _sequence: ISequence

  /** ISheetObjects on the epoch sheet, keyed by cell_id. */
  private readonly _objects: Map<string, ISheetObject<typeof CELL_PROPS>> = new Map()

  /** Epoch 0 param values per cell. */
  private readonly _epoch0: Map<string, CellSheetProps>

  /** Epoch 1 param values per cell. */
  private readonly _epoch1: Map<string, CellSheetProps>

  /** Ordered unique list of all cell IDs. */
  private readonly _allCellIds: string[]

  /** Registered frame callbacks. */
  private readonly _callbacks: Set<FrameCallback> = new Set()

  /** onChange unsubscribe handles. */
  private readonly _unsubs: Array<() => void> = []

  /** True once the Theatre.js project is ready. */
  private _ready: boolean = false

  /** Current playback position on sequence (0 … 1). */
  private _position: number = 0

  constructor(
    epoch0Params: Map<string, CellSheetProps>,
    epoch1Params: Map<string, CellSheetProps>,
    opts: Required<Pick<EpochTimelineOptions, 'easing' | 'defaultRate'>>,
  ) {
    this._epoch0 = epoch0Params
    this._epoch1 = epoch1Params

    this._allCellIds = [...new Set([
      ...epoch0Params.keys(),
      ...epoch1Params.keys(),
    ])]

    // Build Theatre.js OnDiskState with two keyframes per prop per cell.
    const theatreState = buildTheatreState(epoch0Params, epoch1Params, opts.easing)

    // 1) getProject("astro-svgfigure")
    this._project = getProject('astro-svgfigure', { state: theatreState })

    // 2) .sheet("epoch")
    this._epochSheet = this._project.sheet('epoch')
    this._sequence   = this._epochSheet.sequence

    // 3) Create SheetObject per cell with the M152 prop schema.
    for (const cellId of this._allCellIds) {
      const obj = this._epochSheet.object(cellId, CELL_PROPS)
      this._objects.set(cellId, obj)
    }

    // Listen to sequence position changes to stay in sync and emit frames.
    const posUnsub = onChange(
      (this._sequence as unknown as { pointer: { position: unknown } }).pointer.position,
      (pos: number) => {
        this._position = pos
        this._emitFrame()
      },
    )
    this._unsubs.push(posUnsub)

    // Mark ready once the project resolves.
    this._project.ready.then(() => { this._ready = true })
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /** True once the Theatre.js project state has been hydrated. */
  get ready(): boolean { return this._ready }

  /** All cell IDs managed by this timeline. */
  get cellIds(): readonly string[] { return this._allCellIds }

  /** Current fractional sequence position (0 … 1). */
  get position(): number { return this._position }

  /**
   * Seek the sequence to an arbitrary position between 0 and 1.
   * Theatre.js handles interpolation; callbacks receive the new frame.
   */
  seek(pos: number): void {
    const clamped = Math.max(0, Math.min(pos, 1))
    ;(this._sequence as unknown as { position: number }).position = clamped
  }

  /**
   * Play the sequence from position 0 → 1  (epoch 0 → epoch 1).
   * Theatre.js interpolates all SheetObject props via its bezier engine.
   * @param rate  playback rate (default: 1 = one second for full transition)
   */
  play(rate: number = 1): Promise<boolean> {
    return this._sequence.play({
      range:     [0, 1],
      rate,
      direction: 'normal',
    })
  }

  /** Pause the sequence. */
  pause(): void {
    this._sequence.pause()
  }

  /** Reset to position 0 and pause. */
  stop(): void {
    this._sequence.pause()
    ;(this._sequence as unknown as { position: number }).position = 0
  }

  /**
   * Register a callback to receive interpolated frames on every Theatre.js tick.
   * Returns an unsubscribe function.
   */
  onFrame(cb: FrameCallback): () => void {
    this._callbacks.add(cb)
    // Emit the current frame immediately so the caller can initialise.
    cb(this._buildFrame())
    return () => this._callbacks.delete(cb)
  }

  /**
   * Subscribe to changes on a specific cell's props via Theatre.js onChange().
   * Returns an unsubscribe function.
   */
  onCellChange(
    cellId: string,
    cb: (values: CellSheetProps) => void,
  ): () => void {
    const obj = this._objects.get(cellId)
    if (!obj) {
      console.warn(`[EpochTimeline] unknown cell "${cellId}"`)
      return () => {}
    }
    return onChange(obj.props, cb as (v: unknown) => void)
  }

  /**
   * Read the static CellState[] for epoch 0.
   */
  readEpoch0(): CellState[] {
    return this._allCellIds.map(id => {
      const v = this._epoch0.get(id) ?? { x: 0, y: 0, width: 100, height: 50, opacity: 0, glowIntensity: 1 }
      return { cell_id: id, ...v }
    })
  }

  /**
   * Read the static CellState[] for epoch 1.
   */
  readEpoch1(): CellState[] {
    return this._allCellIds.map(id => {
      const v = this._epoch1.get(id) ?? { x: 0, y: 0, width: 100, height: 50, opacity: 0, glowIntensity: 1 }
      return { cell_id: id, ...v }
    })
  }

  /**
   * Return the current interpolated CellState[] without waiting for a callback.
   */
  getCurrentFrame(): EpochFrame {
    return this._buildFrame()
  }

  /**
   * Get the SheetObject for a specific cell (for external Theatre.js Studio use).
   */
  getObject(cellId: string): ISheetObject<typeof CELL_PROPS> | undefined {
    return this._objects.get(cellId)
  }

  /**
   * Destroy the timeline — pause sequence, unsubscribe all onChange listeners,
   * and clear all registered frame callbacks.
   */
  destroy(): void {
    this._sequence.pause()
    for (const unsub of this._unsubs) unsub()
    this._callbacks.clear()
    this._unsubs.length = 0
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  private _emitFrame(): void {
    if (this._callbacks.size === 0) return
    const frame = this._buildFrame()
    for (const cb of this._callbacks) cb(frame)
  }

  /**
   * Build an EpochFrame from Theatre.js current interpolated values.
   *
   * Theatre.js has already done the bezier interpolation between keyframes
   * (driven by _sequence.position), so we just read val(obj.props) for each
   * cell and pack the result.
   */
  private _buildFrame(): EpochFrame {
    const pos   = this._position
    const blend = pos    // position 0…1 is exactly the blend factor

    const cells: CellState[] = []

    for (const [cellId, obj] of this._objects) {
      const v = val(obj.props) as CellSheetProps

      cells.push({
        cell_id:       cellId,
        x:             v.x,
        y:             v.y,
        width:         v.width,
        height:        v.height,
        opacity:       v.opacity,
        glowIntensity: v.glowIntensity,
      })
    }

    return { position: pos, blend, cells }
  }
}

// ─── Epoch params loader ─────────────────────────────────────────────────────

/**
 * Discover all cell JSON files in an epoch_params directory and parse them.
 *
 * Works in both Node.js (SSR / build time) and browser (via fetch) contexts.
 * Returns a Map<cell_id, CellSheetProps>.
 */
async function loadEpochParams(epochIndex: number): Promise<Map<string, CellSheetProps>> {
  const result = new Map<string, CellSheetProps>()

  // Dynamically import all JSON files from the epoch directory.
  // Use Vite's import.meta.glob for build-time bundling, with eager loading.
  // At build time Vite resolves the glob; at runtime the modules are already bundled.
  const epoch0Modules = import.meta.glob<EpochParamJSON>(
    '../../../channels/convergence/epoch_params/0/*.json',
    { eager: true, import: 'default' },
  )
  const epoch1Modules = import.meta.glob<EpochParamJSON>(
    '../../../channels/convergence/epoch_params/1/*.json',
    { eager: true, import: 'default' },
  )

  const modules = epochIndex === 0 ? epoch0Modules : epoch1Modules

  for (const [_path, param] of Object.entries(modules)) {
    if (param && param.cell_id) {
      result.set(param.cell_id, epochParamToProps(param))
    }
  }

  return result
}

/**
 * Load all cell JSON files from an epoch_params directory using Node.js fs.
 * Fallback for SSR / test / non-Vite contexts.
 */
async function loadEpochParamsNode(epochIndex: number): Promise<Map<string, CellSheetProps>> {
  const result = new Map<string, CellSheetProps>()

  try {
    const fs = await import('fs')
    const path = await import('path')

    // Resolve relative to project root.
    const dir = path.resolve(EPOCH_PARAMS_BASE, String(epochIndex))
    if (!fs.existsSync(dir)) {
      console.warn(`[EpochTimeline] epoch_params dir not found: ${dir}`)
      return result
    }

    const files = fs.readdirSync(dir).filter((f: string) => f.endsWith('.json'))

    for (const file of files) {
      const raw = fs.readFileSync(path.join(dir, file), 'utf-8')
      const param: EpochParamJSON = JSON.parse(raw)
      if (param.cell_id) {
        result.set(param.cell_id, epochParamToProps(param))
      }
    }
  } catch (err) {
    console.warn(`[EpochTimeline] Failed to load epoch_params/${epochIndex}:`, err)
  }

  return result
}

// ─── Factory function ─────────────────────────────────────────────────────────

/**
 * Create an EpochTimeline that:
 *   1) Calls getProject("astro-svgfigure").sheet("epoch")
 *   2) Creates a SheetObject per cell with props { x, y, width, height, opacity, glowIntensity }
 *   3) Reads epoch 0 values from channels/convergence/epoch_params/0/{cell}.json
 *   4) Reads epoch 1 values from channels/convergence/epoch_params/1/{cell}.json
 *   5) Uses sheet.sequence to interpolate between the two keyframes
 *
 * @param options  Optional easing and playback rate configuration.
 *
 * @example
 * ```ts
 * const tl = await createEpochTimeline({ easing: 'ease' })
 *
 * tl.onFrame(({ cells, blend }) => {
 *   for (const cell of cells) {
 *     pixiContainers.get(cell.cell_id)?.set({
 *       x: cell.x, y: cell.y,
 *       width: cell.width, height: cell.height,
 *       alpha: cell.opacity,
 *     })
 *   }
 * })
 *
 * tl.play()   // interpolate epoch 0 → epoch 1 over 1 second
 * ```
 */
export async function createEpochTimeline(
  options: EpochTimelineOptions = {},
): Promise<EpochTimeline> {

  const opts = {
    easing:      options.easing      ?? 'ease' as EasingPreset,
    defaultRate: options.defaultRate ?? 1,
  }

  // Load epoch params from JSON files.
  let epoch0Params: Map<string, CellSheetProps>
  let epoch1Params: Map<string, CellSheetProps>

  try {
    // Try Vite import.meta.glob path first (browser / Astro build).
    epoch0Params = await loadEpochParams(0)
    epoch1Params = await loadEpochParams(1)
  } catch {
    // Fallback to Node.js fs for SSR / test contexts.
    epoch0Params = await loadEpochParamsNode(0)
    epoch1Params = await loadEpochParamsNode(1)
  }

  if (epoch0Params.size === 0 && epoch1Params.size === 0) {
    throw new Error(
      '[createEpochTimeline] No cell params found in epoch_params/0 or epoch_params/1',
    )
  }

  return new EpochTimeline(epoch0Params, epoch1Params, opts)
}

/**
 * Synchronous factory — provide pre-loaded epoch params directly.
 * Useful when the caller has already fetched / imported the JSON data.
 *
 * @param epoch0  Map of cell_id → epoch 0 param JSON objects.
 * @param epoch1  Map of cell_id → epoch 1 param JSON objects.
 * @param options Optional easing and playback rate.
 */
export function createEpochTimelineSync(
  epoch0: Map<string, EpochParamJSON>,
  epoch1: Map<string, EpochParamJSON>,
  options: EpochTimelineOptions = {},
): EpochTimeline {

  const opts = {
    easing:      options.easing      ?? 'ease' as EasingPreset,
    defaultRate: options.defaultRate ?? 1,
  }

  const epoch0Props = new Map<string, CellSheetProps>()
  const epoch1Props = new Map<string, CellSheetProps>()

  for (const [id, param] of epoch0) epoch0Props.set(id, epochParamToProps(param))
  for (const [id, param] of epoch1) epoch1Props.set(id, epochParamToProps(param))

  return new EpochTimeline(epoch0Props, epoch1Props, opts)
}

// ─── Re-exports ──────────────────────────────────────────────────────────────

/** Extract CellSheetProps from a raw EpochParamJSON. */
export { epochParamToProps }

export type { EpochTimeline }
