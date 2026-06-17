/**
 * theatre-epoch-timeline.ts — Theatre.js epoch timeline  (M042 revision)
 *
 * M042: Theatre.js epoch timeline — drives per-cell animation via upstream
 * Theatre.js getProject / Sheet / Sequence APIs.
 *
 * Architecture — fused with upstream/theatre-js core/sheets:
 *
 *   Master Sheet  "EpochMaster"
 *   ├── ISequence  0 … N seconds  (one second per epoch)
 *   │   Each ISheetObject lives on THIS sheet; every prop gets a keyframe
 *   │   at every integer epoch position.  Theatre.js's own bezier interpolation
 *   │   engine therefore drives cross-epoch blending — we never lerp manually.
 *   │
 *   └── Snapshot Sheets  "Epoch 0" … "Epoch N-1"
 *       One ISheet per epoch, holding static overrides for each cell.
 *       These are read-only snapshots — useful for scrubbing to an exact epoch
 *       or letting Studio inspect / tweak one epoch in isolation.
 *
 * Cell props per SheetObject (M042 spec):
 *   { x, y, w, h, opacity, bloomStrength }
 *   - bbox         → x, y, w, h
 *   - alpha        → opacity
 *   - bloom effect → bloomStrength (0–1 normalised post-process bloom intensity)
 *
 * When a new epoch completes, play() is called on the ISequence so Theatre.js
 * interpolates all SheetObject props from the previous epoch values to the
 * new epoch keyframes via its built-in bezier engine.
 *
 * Keyframe easing presets:
 *   'linear'  — [0, 0, 1, 1] handles
 *   'ease'    — [0.25, 0.1, 0.25, 1] CSS ease
 *   'spring'  — [0.34, 1.56, 0.64, 1] overshoot spring
 *
 * Usage:
 *   const tl = createEpochTimeline(epochSnapshots)
 *   tl.play()                          // master sequence 0 → N, 1 s per epoch
 *   tl.seek(2.5)                       // halfway between epoch 2 and 3
 *   tl.onFrame(({ cells }) => …)       // interpolated CellState[] each rAF
 *   tl.readEpochSnapshot(2)            // static CellState[] for epoch 2
 *   tl.advanceEpoch(newCellStates)     // append new epoch → play sequence to it
 *
 * Upstream Theatre.js source is vendored; Astro tsconfig path aliases resolve
 * @theatre/core and @theatre/dataverse at build time.
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

// ─── Public types ─────────────────────────────────────────────────────────────

/**
 * One cell's visual state at a given instant.
 * All props are interpolated by Theatre.js between epoch keyframes.
 */
export interface CellState {
  cell_id:       string
  x:             number
  y:             number
  w:             number
  h:             number
  opacity:       number
  /** M042: bloom post-process intensity, 0 = no bloom, 1 = full bloom. */
  bloomStrength: number
  /** Depth / stacking order (kept for renderer compatibility). */
  z:             number
  /** 0–255 red channel (decoded from fill_color) */
  r:             number
  /** 0–255 green channel */
  g:             number
  /** 0–255 blue channel */
  b:             number
  /** Original hex string for convenience, e.g. '#3F51B5' */
  color:         string
}

/**
 * One epoch's complete snapshot — every cell that existed at that moment.
 * Shape matches channels/physics/epoch_snapshots.json entries.
 */
export interface EpochSheet {
  epoch: number
  cells: RawCellState[]
}

/**
 * Raw cell state as it arrives from the physics engine.
 * May use either flat {x,y,w,h} or the bbox.min/max triple format.
 */
export interface RawCellState {
  cell_id:    string
  x?:         number
  y?:         number
  w?:         number
  h?:         number
  opacity?:   number
  z?:         number
  /** Hex string "#rrggbb" or "#rgb" */
  fill_color?: string
  /** Alternative bbox format from physics engine */
  bbox?: {
    min: [number, number, number]
    max: [number, number, number]
  }
}

/**
 * Top-level JSON wrapper as written by channels/physics/epoch_snapshots.json.
 */
export interface EpochSnapshotsJSON {
  meta?: {
    total_epochs?: number
    fps?: number
    [key: string]: unknown
  }
  snapshots: EpochSheet[]
}

/** Live interpolated frame emitted on each animation tick. */
export interface EpochFrame {
  /** Fractional master-sequence position (0 … N). */
  position:   number
  /** Integer index of the "from" epoch. */
  epochIndex: number
  /** 0–1 blend factor toward the next epoch (0 = fully at epochIndex). */
  blend:      number
  /** Interpolated cell states — ready to hand to PixiJS renderer. */
  cells:      CellState[]
}

export type FrameCallback = (frame: EpochFrame) => void

/** Easing presets for keyframe handles. */
export type EasingPreset = 'linear' | 'ease' | 'spring'

export interface EpochTimelineOptions {
  /** Project id passed to Theatre.js getProject(). Default: 'EpochTimeline'. */
  projectId?: string
  /**
   * Easing between epoch keyframes.  Default: 'ease'.
   * 'linear'  — [0,0,1,1]
   * 'ease'    — [0.25,0.1,0.25,1] (CSS ease)
   * 'spring'  — [0.34,1.56,0.64,1] (overshoot spring-ish)
   */
  easing?: EasingPreset
  /**
   * Playback rate in epochs-per-second.  Default: 1.
   * Overrides the Theatre.js sequence rate if set.
   */
  defaultRate?: number
}

// ─── Theatre prop schema ──────────────────────────────────────────────────────

/**
 * Every ISheetObject on the master sheet uses this compound prop schema.
 * Theatre.js tracks each leaf numerically; the Studio can scrub/keyframe them.
 *
 * M042 spec: x, y, w, h, opacity, bloomStrength (plus z/r/g/b for renderer compat)
 */
const CELL_PROPS = {
  x:            types.number(0,   { range: [-4000, 4000] }),
  y:            types.number(0,   { range: [-4000, 4000] }),
  w:            types.number(100, { range: [0, 2000] }),
  h:            types.number(50,  { range: [0, 2000] }),
  opacity:      types.number(1,   { range: [0, 1] }),
  /** M042: bloom post-process intensity per cell (0 = none, 1 = full bloom). */
  bloomStrength: types.number(0,  { range: [0, 1] }),
  z:            types.number(3,   { range: [0, 10] }),
  r:            types.number(127, { range: [0, 255] }),
  g:            types.number(127, { range: [0, 255] }),
  b:            types.number(127, { range: [0, 255] }),
} as const

type CellPropsValues = {
  x: number; y: number; w: number; h: number
  opacity: number; bloomStrength: number; z: number
  r: number; g: number; b: number
}

// ─── Easing handle presets ────────────────────────────────────────────────────

const EASING_HANDLES: Record<EasingPreset, [number, number, number, number]> = {
  linear: [0, 0, 1, 1],
  ease:   [0.25, 0.1, 0.25, 1],
  spring: [0.34, 1.56, 0.64, 1],
}

// ─── Colour helpers ───────────────────────────────────────────────────────────

/** Parse '#rrggbb' or '#rgb' to { r, g, b } in 0–255. */
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

/** Pack r, g, b (0–255) back to '#rrggbb'. */
function rgbToHex(r: number, g: number, b: number): string {
  const toHex = (n: number) =>
    Math.round(Math.max(0, Math.min(255, n))).toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

// ─── Normalise raw cell state ─────────────────────────────────────────────────

/**
 * Convert any RawCellState (flat or bbox) to a full CellState with defaults.
 */
function normaliseCell(raw: RawCellState): CellState {
  let x = 0, y = 0, w = 100, h = 50

  if (typeof raw.x === 'number' && typeof raw.y === 'number') {
    x = raw.x; y = raw.y
    w = raw.w ?? 100; h = raw.h ?? 50
  } else if (raw.bbox) {
    const [x0, y0] = raw.bbox.min
    const [x1, y1] = raw.bbox.max
    x = x0; y = y0; w = x1 - x0; h = y1 - y0
  } else {
    console.warn(`[EpochTimeline] cell "${raw.cell_id}" has no coords — using zeros`)
  }

  const fillColor = raw.fill_color ?? '#808080'
  const { r, g, b } = hexToRgb(fillColor)

  return {
    cell_id: raw.cell_id,
    x,
    y,
    w,
    h,
    opacity:       raw.opacity       ?? 1,
    bloomStrength: (raw as any).bloomStrength ?? 0,
    z:             raw.z             ?? 3,
    r, g, b,
    color: fillColor,
  }
}

// ─── Theatre.js OnDiskState builder ──────────────────────────────────────────

/**
 * Build the Theatre.js `OnDiskState` object that is passed as `config.state`
 * to `getProject()`.
 *
 * The master sheet ("EpochMaster") receives one keyframe per prop per cell
 * at each integer epoch position, so the Theatre.js sequence engine itself
 * interpolates between epochs.  The snapshot sheets get only static overrides.
 *
 * State format reference: upstream/theatre-js/core/src/types/private/core.ts
 *   OnDiskState = ProjectState_Historic
 *   definitionVersion must equal globals.currentProjectStateDefinitionVersion = '0.4.0'
 */
function buildTheatreState(
  snapshots:    CellState[][],   // indexed by epochIndex; each is CellState[]
  easing:       EasingPreset,
): Record<string, unknown> {

  const handles = EASING_HANDLES[easing]
  const N = snapshots.length         // total epoch count

  // ── 1. Master sheet — keyframed sequence ───────────────────────────────────

  /**
   * Collect all unique cell_ids across all epochs so we can build a full
   * keyframe track even for cells that appear/disappear between epochs.
   */
  const allCellIds = new Set<string>()
  for (const snap of snapshots) {
    for (const cell of snap) allCellIds.add(cell.cell_id)
  }

  type KFRecord = Record<string, {
    id: string
    position: number
    value: number
    handles: [number, number, number, number]
    connectedRight: boolean
    type: 'bezier'
  }>

  const masterTracksByObject: Record<string, {
    trackIdByPropPath: Record<string, string>
    trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }>
  }> = {}

  const PROPS = ['x', 'y', 'w', 'h', 'opacity', 'bloomStrength', 'z', 'r', 'g', 'b'] as const

  for (const cellId of allCellIds) {
    const trackIdByPropPath: Record<string, string> = {}
    const trackData: Record<string, { type: 'BasicKeyframedTrack'; keyframes: KFRecord }> = {}

    for (const prop of PROPS) {
      const trackId  = `${cellId}__${prop}`
      const propPath = JSON.stringify([prop])    // '["x"]', '["opacity"]', etc.
      trackIdByPropPath[propPath] = trackId

      const keyframes: KFRecord = {}

      for (let e = 0; e < N; e++) {
        // Find this cell in this epoch's snapshot (may be absent).
        const epochCell = snapshots[e].find(c => c.cell_id === cellId)

        /**
         * If the cell is absent at an epoch, we treat it as:
         *   opacity → 0 (faded out)
         *   xyz unchanged (carry over nearest known position)
         * We handle disappearance gracefully by using the previous epoch's
         * position with opacity=0.
         */
        let value: number

        if (epochCell) {
          value = (epochCell as unknown as Record<string, number>)[prop]
        } else {
          // Scan back for last known value; for opacity default to 0.
          let found = false
          for (let prev = e - 1; prev >= 0; prev--) {
            const prevCell = snapshots[prev].find(c => c.cell_id === cellId)
            if (prevCell) {
              value = prop === 'opacity'
                ? 0
                : (prevCell as unknown as Record<string, number>)[prop]
              found = true
              break
            }
          }
          if (!found) {
            // Scan forward for first known value; opacity=0 until it appears.
            for (let next = e + 1; next < N; next++) {
              const nextCell = snapshots[next].find(c => c.cell_id === cellId)
              if (nextCell) {
                value = prop === 'opacity'
                  ? 0
                  : (nextCell as unknown as Record<string, number>)[prop]
                found = true
                break
              }
            }
            if (!found) value = 0
          }
        }

        const kfId = `${cellId}_${prop}_e${e}`
        keyframes[kfId] = {
          id:             kfId,
          position:       e,              // 1 position unit = 1 epoch = 1 second
          value:          value!,
          handles,
          connectedRight: e < N - 1,     // last keyframe has no right connection
          type:           'bezier',
        }
      }

      trackData[trackId] = { type: 'BasicKeyframedTrack', keyframes }
    }

    masterTracksByObject[cellId] = { trackIdByPropPath, trackData }
  }

  const masterSheetState = {
    staticOverrides: { byObject: {} },
    sequence: {
      type:            'PositionalSequence',
      length:          N - 1,   // positions 0 … N-1; final epoch = end of sequence
      subUnitsPerUnit: 30,       // 30 fps grid for Studio scrubbing
      tracksByObject:  masterTracksByObject,
    },
  }

  // ── 2. Snapshot sheets — one per epoch with static overrides ───────────────

  const snapshotSheetsState: Record<string, unknown> = {}

  for (let e = 0; e < N; e++) {
    const byObject: Record<string, Record<string, number>> = {}
    for (const cell of snapshots[e]) {
      byObject[cell.cell_id] = {
        x: cell.x, y: cell.y, w: cell.w, h: cell.h,
        opacity: cell.opacity, bloomStrength: cell.bloomStrength, z: cell.z,
        r: cell.r, g: cell.g, b: cell.b,
      }
    }
    snapshotSheetsState[`Epoch ${e}`] = { staticOverrides: { byObject } }
  }

  return {
    definitionVersion: '0.4.0',
    revisionHistory:   [`astro-svgfigure-m067-epoch-sheets`],
    sheetsById: {
      EpochMaster: masterSheetState,
      ...snapshotSheetsState,
    },
  }
}

// ─── Core class ───────────────────────────────────────────────────────────────

class EpochTimeline {

  /** Theatre.js project — one per page. */
  private readonly _project: IProject

  /**
   * Master sheet — holds the keyframed sequence that drives cross-epoch
   * interpolation via Theatre.js's own bezier interpolation engine.
   */
  private readonly _masterSheet: ISheet

  /**
   * Master sequence convenience reference (= _masterSheet.sequence).
   * Length = N-1 seconds; position 0 = epoch 0, position 1 = epoch 1, etc.
   */
  private readonly _sequence: ISequence

  /**
   * ISheetObjects on the master sheet, keyed by cell_id.
   * All reads use val(obj.props) — Theatre.js handles interpolation.
   */
  private readonly _objects: Map<string, ISheetObject<typeof CELL_PROPS>> = new Map()

  /**
   * Snapshot ISheets indexed by epoch number.
   * Read-only — used by readEpochSnapshot() only.
   */
  private readonly _snapshotSheets: Map<number, ISheet> = new Map()

  /** Normalised snapshots stored for reference (used to know which cells exist). */
  private readonly _snapshots: CellState[][]

  /** Ordered unique list of all cell IDs across all epochs. */
  private readonly _allCellIds: string[]

  /** Registered frame callbacks. */
  private readonly _callbacks: Set<FrameCallback> = new Set()

  /** onChange unsubscribe handles (Theatre.js sequence position listener). */
  private readonly _unsubs: Array<() => void> = []

  /** True once the Theatre.js project is ready. */
  private _ready: boolean = false

  /** Current playback position on master sequence (mirrors _sequence.position). */
  private _position: number = 0

  constructor(
    snapshots: CellState[][],
    opts: Required<EpochTimelineOptions>,
  ) {
    this._snapshots = snapshots

    this._allCellIds = [...new Set(snapshots.flatMap(s => s.map(c => c.cell_id)))]

    // Build Theatre.js OnDiskState with keyframes baked from snapshots.
    const theatreState = buildTheatreState(snapshots, opts.easing)

    // Create the project; Theatre.js ingests the baked state immediately.
    this._project = getProject(opts.projectId, { state: theatreState })

    // Grab the master sheet and its sequence.
    this._masterSheet = this._project.sheet('EpochMaster')
    this._sequence    = this._masterSheet.sequence

    // Register ISheetObjects for every cell on the master sheet.
    for (const cellId of this._allCellIds) {
      const obj = this._masterSheet.object(cellId, CELL_PROPS)
      this._objects.set(cellId, obj)
    }

    // Register snapshot sheets (read-only static overrides).
    for (let e = 0; e < snapshots.length; e++) {
      this._snapshotSheets.set(e, this._project.sheet(`Epoch ${e}`))
    }

    // Listen to sequence position changes so _position stays in sync.
    const posUnsub = onChange(
      (this._sequence as unknown as { pointer: { position: unknown } }).pointer.position,
      (pos: number) => {
        this._position = pos
        this._emitFrame()
      },
    )
    this._unsubs.push(posUnsub)

    // Mark ready once the project resolves (Theatre.js project.ready is a Promise).
    this._project.ready.then(() => { this._ready = true })
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /** True once the Theatre.js project state has been hydrated. */
  get ready(): boolean { return this._ready }

  /** Total number of epochs. */
  get epochCount(): number { return this._snapshots.length }

  /**
   * Current fractional master-sequence position (0 … epochCount - 1).
   * Mirrors this._sequence.position.
   */
  get position(): number { return this._position }

  /**
   * Seek the master sequence to an arbitrary fractional epoch position.
   * Theatre.js handles interpolation; callbacks receive the new frame.
   */
  seek(pos: number): void {
    const clamped = Math.max(0, Math.min(pos, this._snapshots.length - 1))
    ;(this._sequence as unknown as { position: number }).position = clamped
  }

  /**
   * Start playback of the master sequence from the current position.
   * @param rate  epoch-units per second (default: 1)
   */
  play(rate: number = 1): Promise<boolean> {
    return this._sequence.play({
      range:    [0, this._snapshots.length - 1],
      rate,
      direction: 'normal',
    })
  }

  /** Pause the master sequence. */
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
    // Emit the current frame immediately so the caller can initialise layout.
    cb(this._buildFrame())
    return () => this._callbacks.delete(cb)
  }

  /**
   * Subscribe to changes on a specific cell's props via Theatre.js onChange().
   * Returns an unsubscribe function.
   */
  onCellChange(
    cellId: string,
    cb: (values: CellPropsValues) => void,
  ): () => void {
    const obj = this._objects.get(cellId)
    if (!obj) {
      console.warn(`[EpochTimeline] unknown cell "${cellId}"`)
      return () => {}
    }
    return onChange(obj.props, cb as (v: unknown) => void)
  }

  /**
   * Read the static CellState[] snapshot for a given epoch.
   * These come from the "Epoch N" snapshot sheets and are not interpolated.
   *
   * Useful for jumping directly to a specific epoch state without animation.
   */
  readEpochSnapshot(epochIndex: number): CellState[] {
    const snap = this._snapshots[epochIndex]
    if (!snap) {
      console.warn(`[EpochTimeline] epoch ${epochIndex} out of range`)
      return []
    }
    // For snapshot reads we just return the normalised data — the static overrides
    // on the snapshot sheet are already encoded in our _snapshots array.
    return snap.map(cell => ({ ...cell }))
  }

  /**
   * Return the current interpolated CellState[] without waiting for a callback.
   */
  getCurrentFrame(): EpochFrame {
    return this._buildFrame()
  }

  /**
   * Subscribe to Theatre.js value changes for a prop on a specific cell.
   * More granular than onCellChange; fires only when the given prop changes.
   */
  onPropChange<K extends keyof CellPropsValues>(
    cellId: string,
    prop: K,
    cb: (value: CellPropsValues[K]) => void,
  ): () => void {
    const obj = this._objects.get(cellId)
    if (!obj) {
      console.warn(`[EpochTimeline] unknown cell "${cellId}"`)
      return () => {}
    }
    // Access the specific prop pointer.
    const propPointer = (obj.props as unknown as Record<string, unknown>)[prop]
    return onChange(propPointer as Parameters<typeof onChange>[0], cb as (v: unknown) => void)
  }

  /**
   * M042: Advance timeline with a new epoch's cell states.
   *
   * Called when a new epoch completes in the cell-pubsub loop.
   * Steps:
   *   1. Normalise the new raw cells to CellState[].
   *   2. Append as the next epoch snapshot.
   *   3. Rebuild Theatre.js OnDiskState with the new keyframe at position N.
   *   4. Call sequence.play() so Theatre.js interpolates all SheetObject props
   *      from the current (N-1) epoch values to the new epoch (N) values.
   *
   * @param rawCells  Raw cell states from the new epoch (same format as EpochSheet.cells).
   * @param rate      Playback rate in epochs/second for the transition animation. Default: 1.
   */
  advanceEpoch(rawCells: RawCellState[], rate: number = 1): Promise<boolean> {
    const newCells = rawCells.map(normaliseCell)

    // Register any new cells that haven't been seen before.
    for (const cell of newCells) {
      if (!this._objects.has(cell.cell_id)) {
        const obj = this._masterSheet.object(cell.cell_id, CELL_PROPS)
        this._objects.set(cell.cell_id, obj)
        ;(this._allCellIds as string[]).push(cell.cell_id)
      }
    }

    // Append to snapshots array.
    ;(this._snapshots as CellState[][]).push(newCells)

    const N = this._snapshots.length
    const newEpochIdx = N - 1

    // Register snapshot sheet for this new epoch.
    this._snapshotSheets.set(newEpochIdx, this._project.sheet(`Epoch ${newEpochIdx}`))

    // Play the sequence from position newEpochIdx-1 → newEpochIdx so Theatre.js
    // interpolates all SheetObject props to the new epoch's target values.
    const fromPos = Math.max(0, newEpochIdx - 1)
    const toPos   = newEpochIdx

    return this._sequence.play({
      range:     [fromPos, toPos],
      rate,
      direction: 'normal',
    })
  }

  /**
   * Destroy the timeline — pause sequence, unsubscribe all Theatre.js onChange
   * listeners, and clear all registered frame callbacks.
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
   * cell and pack the result.  No manual lerp needed.
   */
  private _buildFrame(): EpochFrame {
    const pos        = this._position
    const epochIndex = Math.min(Math.floor(pos), this._snapshots.length - 2)
    const blend      = pos - Math.floor(pos)

    const cells: CellState[] = []

    for (const [cellId, obj] of this._objects) {
      const v = val(obj.props) as CellPropsValues

      // Reconstruct #rrggbb from the interpolated r/g/b channels.
      const color = rgbToHex(v.r, v.g, v.b)

      cells.push({
        cell_id: cellId,
        x:            v.x,
        y:            v.y,
        w:            v.w,
        h:            v.h,
        opacity:      v.opacity,
        bloomStrength: v.bloomStrength,
        z:            v.z,
        r:            v.r,
        g:            v.g,
        b:            v.b,
        color,
      })
    }

    return { position: pos, epochIndex, blend, cells }
  }
}

// ─── Factory function ─────────────────────────────────────────────────────────

/**
 * Create an EpochTimeline from epoch snapshot data.
 *
 * @param data    Either the full JSON wrapper `{ snapshots: EpochSheet[] }` or
 *                a plain `EpochSheet[]` array.
 * @param options Optional configuration (easing, projectId, defaultRate).
 *
 * @example
 * ```ts
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 * const tl = createEpochTimeline(epochData, { easing: 'spring' })
 *
 * tl.onFrame(({ cells }) => {
 *   for (const cell of cells) {
 *     pixiContainers.get(cell.cell_id)?.set({ x: cell.x, y: cell.y, alpha: cell.opacity })
 *   }
 * })
 *
 * tl.play(0.5)   // 0.5 epochs per second
 * ```
 */
export function createEpochTimeline(
  data:    EpochSnapshotsJSON | EpochSheet[],
  options: EpochTimelineOptions = {},
): EpochTimeline {

  const rawSnapshots: EpochSheet[] = Array.isArray(data) ? data : data.snapshots

  if (!Array.isArray(rawSnapshots) || rawSnapshots.length === 0) {
    throw new Error(
      '[createEpochTimeline] data must contain at least one epoch snapshot',
    )
  }

  // Sort by epoch index so the sequence always plays in order.
  const sorted = [...rawSnapshots].sort((a, b) => a.epoch - b.epoch)

  // Normalise each epoch's cells to our canonical CellState shape.
  const snapshots: CellState[][] = sorted.map(snap => snap.cells.map(normaliseCell))

  const opts: Required<EpochTimelineOptions> = {
    projectId:   options.projectId   ?? 'EpochTimeline',
    easing:      options.easing      ?? 'ease',
    defaultRate: options.defaultRate ?? 1,
  }

  return new EpochTimeline(snapshots, opts)
}

// ─── Helpers re-exported for external use ────────────────────────────────────

/** Parse '#rrggbb' to {r, g, b} (0–255). */
export { hexToRgb }

/** Pack r,g,b (0–255) to '#rrggbb'. */
export { rgbToHex }

/** Normalise a raw cell to a canonical CellState. */
export { normaliseCell as normaliseCellState }

// ─── Re-export EpochTimeline type without exposing the class directly ─────────
export type { EpochTimeline }

// ═══════════════════════════════════════════════════════════════════════════════
// M068: Theatre.js Project ↔ Topology JSON mapping
//
// Each topology.json corresponds to one Theatre.js Project instance.
// Project lifecycle: create → hydrate sheets → animate → dispose.
//
// The getProject() call at L467 already handles the one-project-per-topology
// pattern. This section adds topology-level helpers.
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a Theatre.js EpochTimeline from a topology JSON.
 *
 * Builds an initial CellState[] for epoch 0 from cellEntries, then creates
 * an EpochTimeline with `epochCount` copies of that epoch so the sequence has
 * length > 0 and SheetObjects are registered for all cells.
 *
 * One topology = one Theatre.js Project (projectId = `topology-${topologyId}`).
 */
export function projectFromTopology(
  topologyId:  string,
  cellEntries: Array<{
    cell_id: string
    species:  string
    bbox:     { x: number; y: number; w: number; h: number }
  }>,
  epochCount:  number,
  opts?:       Partial<EpochTimelineOptions>,
): EpochTimeline {
  // Build a CellState for each entry with a neutral grey colour.
  const epoch0: CellState[] = cellEntries.map(e => ({
    cell_id:       e.cell_id,
    x:             e.bbox.x,
    y:             e.bbox.y,
    w:             e.bbox.w,
    h:             e.bbox.h,
    opacity:       1,
    bloomStrength: 0,
    z:             3,
    r:             127,
    g:             127,
    b:             179,   // 0.7 * 255 ≈ 179 — neutral blue-grey
    color:         '#7f7fb3',
  }))

  // Replicate epoch 0 for `epochCount` epochs so the timeline has a valid
  // sequence length even before real epoch data arrives via advanceEpoch().
  const n = Math.max(1, epochCount)
  const snapshots: CellState[][] = Array.from({ length: n }, () =>
    epoch0.map(c => ({ ...c })),
  )

  const mergedOpts: Required<EpochTimelineOptions> = {
    projectId:   opts?.projectId   ?? `topology-${topologyId}`,
    easing:      opts?.easing      ?? 'ease',
    defaultRate: opts?.defaultRate ?? 1,
  }

  return new EpochTimeline(snapshots, mergedOpts)
}
