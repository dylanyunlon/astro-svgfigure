/**
 * theatre-epoch-timeline.ts — Theatre.js epoch timeline
 *
 * Architecture:
 *   • One ISheet per epoch  ("Epoch 0", "Epoch 1", …)
 *   • Each sheet holds one ISheetObject per cell, keyed by cell_id
 *   • Props per object: { x, y, w, h, opacity, z }
 *   • EpochSequence drives a single 0→N position value whose fractional
 *     part selects the active sheet pair and lerps cell props accordingly
 *
 * Upstream imports go directly to the vendored source so no npm resolution
 * is needed — the Astro tsconfig path aliases (@theatre/core → …) handle
 * the rest at build time.
 *
 * Usage:
 *   const tl = createEpochTimeline(epochSnapshots)
 *   tl.play()                      // 0 → N, 1 s per epoch
 *   tl.seek(2.5)                   // halfway between epoch 2 and 3
 *   tl.onFrame(({ cells }) => …)   // receive interpolated CellState[]
 */

// ─── Upstream Theatre.js (vendored, no npm) ─────────────────────────────────
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

// ─── Public types ────────────────────────────────────────────────────────────

/** One cell's visual state at a specific epoch. */
export interface CellState {
  cell_id: string
  x: number
  y: number
  w: number
  h: number
  opacity: number
  z: number
}

/**
 * One epoch's complete snapshot — every cell that existed at that moment.
 * This is the shape of a single entry inside epoch_snapshots.json.
 */
export interface EpochSheet {
  epoch: number
  cells: CellState[]
}

/**
 * Full input blob expected from channels/physics/epoch_snapshots.json
 * (or any compatible source).
 */
export interface EpochSnapshotsJSON {
  /** Optional top-level metadata the physics engine may include. */
  meta?: {
    total_epochs?: number
    fps?: number
    [key: string]: unknown
  }
  snapshots: EpochSheet[]
}

/** Live interpolated frame emitted on every animation tick. */
export interface EpochFrame {
  /** Current fractional position (0 … N). */
  position: number
  /** Integer index of the "from" epoch. */
  epochIndex: number
  /** 0–1 blend factor toward the next epoch (0 = fully at epochIndex). */
  blend: number
  /** Interpolated cell states for this frame. */
  cells: CellState[]
}

export type FrameCallback = (frame: EpochFrame) => void

// ─── Theatre prop schema ─────────────────────────────────────────────────────

/** The compound prop type used for every cell SheetObject. */
const CELL_PROPS = {
  x:       types.number(0,       { range: [-4000, 4000] }),
  y:       types.number(0,       { range: [-4000, 4000] }),
  w:       types.number(100,     { range: [0, 2000] }),
  h:       types.number(50,      { range: [0, 2000] }),
  opacity: types.number(1,       { range: [0, 1] }),
  z:       types.number(3,       { range: [0, 10] }),
} as const

type CellPropsValues = {
  x: number
  y: number
  w: number
  h: number
  opacity: number
  z: number
}

// ─── Lerp helpers ────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function lerpCellState(a: CellState, b: CellState, t: number): CellState {
  return {
    cell_id: a.cell_id,
    x:       lerp(a.x,       b.x,       t),
    y:       lerp(a.y,       b.y,       t),
    w:       lerp(a.w,       b.w,       t),
    h:       lerp(a.h,       b.h,       t),
    opacity: lerp(a.opacity, b.opacity, t),
    z:       lerp(a.z,       b.z,       t),
  }
}

// ─── Core class ──────────────────────────────────────────────────────────────

class EpochTimeline {
  /** Theatre.js project — one per page, keyed by id. */
  private readonly _project: IProject

  /**
   * One ISheet per epoch, indexed by epoch number.
   * Sheets are named "Epoch 0", "Epoch 1", …
   */
  private readonly _sheets: Map<number, ISheet> = new Map()

  /**
   * Nested map: epochIndex → cell_id → ISheetObject so we can read
   * `.value` without subscribing to change events on every object.
   */
  private readonly _objects: Map<number, Map<string, ISheetObject<typeof CELL_PROPS>>> = new Map()

  /** Canonical snapshots passed in at construction time. */
  private readonly _snapshots: EpochSheet[]

  /** Registered frame callbacks. */
  private readonly _callbacks: Set<FrameCallback> = new Set()

  /** rAF handle. */
  private _rafHandle: number | null = null

  /** Current playback position (0 … epochs-1). */
  private _position: number = 0

  /** Playback rate in epoch-units per second (default: 1 epoch / s). */
  private _rate: number = 1

  /** Whether the timeline is currently playing. */
  private _playing: boolean = false

  /** Timestamp of the last rAF callback (ms). */
  private _lastTs: number | null = null

  constructor(snapshots: EpochSheet[]) {
    if (snapshots.length === 0) {
      throw new Error('[EpochTimeline] snapshots array must not be empty')
    }

    this._snapshots = snapshots

    // Create one Theatre.js project for the whole timeline.
    this._project = getProject('EpochTimeline', {
      // Provide the epoch states as Theatre.js initial state so the Studio
      // can inspect/edit them without a state file.
      state: buildTheatreState(snapshots),
    })

    // Register one sheet + objects per epoch.
    for (const snap of snapshots) {
      const sheetId = `Epoch ${snap.epoch}`
      const sheet   = this._project.sheet(sheetId)
      this._sheets.set(snap.epoch, sheet)

      const objMap = new Map<string, ISheetObject<typeof CELL_PROPS>>()
      for (const cell of snap.cells) {
        const obj = sheet.object(cell.cell_id, CELL_PROPS)
        objMap.set(cell.cell_id, obj)
      }
      this._objects.set(snap.epoch, objMap)
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /** Total number of epochs. */
  get epochCount(): number {
    return this._snapshots.length
  }

  /** Current fractional position (0 … epochCount - 1). */
  get position(): number {
    return this._position
  }

  /**
   * Seek to an arbitrary fractional position.
   * Does not start/stop playback.
   */
  seek(pos: number): void {
    this._position = Math.max(0, Math.min(pos, this._snapshots.length - 1))
    this._emitFrame()
  }

  /**
   * Start playback from the current position.
   * @param rate - epoch-units per second (default: 1)
   */
  play(rate?: number): void {
    if (rate !== undefined) this._rate = rate
    this._playing  = true
    this._lastTs   = null
    this._scheduleRaf()
  }

  /** Pause playback, keeping the current position. */
  pause(): void {
    this._playing = false
    this._cancelRaf()
  }

  /** Stop and reset to position 0. */
  stop(): void {
    this._playing  = false
    this._position = 0
    this._cancelRaf()
    this._emitFrame()
  }

  /** Register a callback to receive interpolated frames. */
  onFrame(cb: FrameCallback): () => void {
    this._callbacks.add(cb)
    // Emit the current frame immediately so the caller can initialise layout.
    cb(this._buildFrame())
    return () => this._callbacks.delete(cb)
  }

  /** Subscribe to changes on a specific cell object in a specific epoch. */
  onCellChange(
    epochIndex: number,
    cellId: string,
    cb: (values: CellPropsValues) => void,
  ): () => void {
    const obj = this._objects.get(epochIndex)?.get(cellId)
    if (!obj) {
      console.warn(`[EpochTimeline] unknown cell "${cellId}" in epoch ${epochIndex}`)
      return () => {}
    }
    return onChange(obj.props, cb)
  }

  /** Release all Theatre.js subscriptions and rAF handles. */
  dispose(): void {
    this._cancelRaf()
    this._callbacks.clear()
  }

  // ── Private ─────────────────────────────────────────────────────────────

  private _scheduleRaf(): void {
    if (this._rafHandle !== null) return
    const tick = (ts: number) => {
      this._rafHandle = null
      if (!this._playing) return

      if (this._lastTs !== null) {
        const dtMs  = ts - this._lastTs
        const dtEpoch = (dtMs / 1000) * this._rate
        this._position = Math.min(
          this._position + dtEpoch,
          this._snapshots.length - 1,
        )
      }
      this._lastTs = ts
      this._emitFrame()

      if (this._position < this._snapshots.length - 1) {
        this._scheduleRaf()
      } else {
        // Reached the end.
        this._playing = false
      }
    }
    this._rafHandle = requestAnimationFrame(tick)
  }

  private _cancelRaf(): void {
    if (this._rafHandle !== null) {
      cancelAnimationFrame(this._rafHandle)
      this._rafHandle = null
    }
    this._lastTs = null
  }

  private _emitFrame(): void {
    if (this._callbacks.size === 0) return
    const frame = this._buildFrame()
    for (const cb of this._callbacks) {
      cb(frame)
    }
  }

  private _buildFrame(): EpochFrame {
    const pos        = this._position
    const epochIndex = Math.min(
      Math.floor(pos),
      this._snapshots.length - 2,  // clamp so we always have a "next"
    )
    const blend      = pos - epochIndex

    const snapA = this._snapshots[epochIndex]
    const snapB = this._snapshots[Math.min(epochIndex + 1, this._snapshots.length - 1)]

    // Read current Theatre.js values for epoch A.
    const valuesA = this._readEpochValues(snapA)
    // Read current Theatre.js values for epoch B.
    const valuesB = this._readEpochValues(snapB)

    // Build interpolated cell list.
    const cellIds = new Set([
      ...valuesA.keys(),
      ...valuesB.keys(),
    ])

    const cells: CellState[] = []
    for (const id of cellIds) {
      const a = valuesA.get(id)
      const b = valuesB.get(id)

      if (a && b) {
        // Both epochs have this cell — lerp.
        cells.push(lerpCellState(a, b, blend))
      } else if (a) {
        // Cell disappears after epoch A — fade out.
        cells.push({ ...a, opacity: lerp(a.opacity, 0, blend) })
      } else if (b) {
        // Cell appears in epoch B — fade in.
        cells.push({ ...b, opacity: lerp(0, b.opacity, blend) })
      }
    }

    return { position: pos, epochIndex, blend, cells }
  }

  /** Read current Theatre.js prop values for every cell in a snapshot. */
  private _readEpochValues(snap: EpochSheet): Map<string, CellState> {
    const map  = new Map<string, CellState>()
    const objs = this._objects.get(snap.epoch)
    if (!objs) return map

    for (const cell of snap.cells) {
      const obj = objs.get(cell.cell_id)
      if (!obj) continue
      const v = val(obj.props) as CellPropsValues
      map.set(cell.cell_id, {
        cell_id: cell.cell_id,
        x:       v.x,
        y:       v.y,
        w:       v.w,
        h:       v.h,
        opacity: v.opacity,
        z:       v.z,
      })
    }
    return map
  }
}

// ─── Theatre.js state builder ────────────────────────────────────────────────

/**
 * Build a Theatre.js on-disk-state-compatible JS object from epoch snapshots.
 * Injecting initial state this way means the Studio can tweak values without
 * needing a state JSON file on disk.
 *
 * Structure mirrors what Theatre writes to state files:
 *  state.sheets[sheetId].staticOverrides.byObject[cellId][propKey] = value
 */
function buildTheatreState(snapshots: EpochSheet[]): Record<string, unknown> {
  const sheetsState: Record<string, unknown> = {}

  for (const snap of snapshots) {
    const sheetId = `Epoch ${snap.epoch}`
    const byObject: Record<string, Record<string, number>> = {}

    for (const cell of snap.cells) {
      byObject[cell.cell_id] = {
        x:       cell.x,
        y:       cell.y,
        w:       cell.w,
        h:       cell.h,
        opacity: cell.opacity,
        z:       cell.z,
      }
    }

    sheetsState[sheetId] = {
      staticOverrides: { byObject },
    }
  }

  return { sheets: sheetsState }
}

// ─── Public factory ──────────────────────────────────────────────────────────

/**
 * Create an EpochTimeline from the contents of
 * `channels/physics/epoch_snapshots.json`.
 *
 * @example
 * ```ts
 * import epochData from '../../../channels/physics/epoch_snapshots.json'
 * const tl = createEpochTimeline(epochData)
 * tl.onFrame(({ cells }) => renderCells(cells))
 * tl.play(0.5) // 0.5 epochs per second
 * ```
 */
export function createEpochTimeline(
  data: EpochSnapshotsJSON | EpochSheet[],
): EpochTimeline {
  // Accept both the raw snapshot array and the full JSON wrapper.
  const snapshots: EpochSheet[] = Array.isArray(data) ? data : data.snapshots

  if (!Array.isArray(snapshots) || snapshots.length === 0) {
    throw new Error(
      '[createEpochTimeline] data must contain at least one epoch snapshot',
    )
  }

  // Sort by epoch index so the timeline always plays in order.
  const sorted = [...snapshots].sort((a, b) => a.epoch - b.epoch)

  // Normalise each cell's bbox → {x, y, w, h} if the upstream JSON uses
  // the channels/physics/cell_registry bbox format ({min:[x,y,z], max:[x,y,z]}).
  const normalised = sorted.map((snap) => ({
    epoch: snap.epoch,
    cells: snap.cells.map(normaliseCellState),
  }))

  return new EpochTimeline(normalised)
}

// ─── Bbox normalisation ──────────────────────────────────────────────────────

/**
 * The physics engine stores cell positions as `bbox.min` / `bbox.max` triples.
 * If the CellState already has flat {x,y,w,h} props those are used directly.
 */
function normaliseCellState(cell: CellState & {
  bbox?: { min: [number, number, number]; max: [number, number, number] }
}): CellState {
  if (
    typeof cell.x === 'number' &&
    typeof cell.y === 'number' &&
    typeof cell.w === 'number' &&
    typeof cell.h === 'number'
  ) {
    // Already normalised.
    return {
      cell_id: cell.cell_id,
      x:       cell.x,
      y:       cell.y,
      w:       cell.w,
      h:       cell.h,
      opacity: cell.opacity ?? 1,
      z:       cell.z ?? 3,
    }
  }

  if (cell.bbox) {
    const [x0, y0] = cell.bbox.min
    const [x1, y1] = cell.bbox.max
    return {
      cell_id: cell.cell_id,
      x:       x0,
      y:       y0,
      w:       x1 - x0,
      h:       y1 - y0,
      opacity: cell.opacity ?? 1,
      z:       cell.z ?? 3,
    }
  }

  console.warn(`[EpochTimeline] cell "${cell.cell_id}" has neither flat coords nor bbox; using zeros`)
  return { cell_id: cell.cell_id, x: 0, y: 0, w: 100, h: 50, opacity: 1, z: 3 }
}

// ─── Re-export EpochTimeline type without exposing the class directly ─────────
export type { EpochTimeline }
