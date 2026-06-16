/**
 * EpochTimeline.tsx — Theatre.js epoch animation timeline for astro-svgfigure
 *
 * Maps epoch 0→4 onto a Theatre.js sequence 0s→4s.
 * Each cell gets a Theatre.js SheetObject with props: x, y, opacity.
 * Keyframes are baked from channels/cell/ * /params.json values,
 * interpolating from dormant state (opacity=0, y offset) at epoch 0
 * to final params.json values at epoch 4.
 *
 * Theatre.js import: @theatre/core CDN ESM (zero-config, browser-only island).
 * Swap to local alias once tsconfig paths for @theatre/core are wired.
 *
 * PixiJS integration:
 *   onChange callbacks write to window.__theatreState:
 *     Map<cell_id, { x, y, opacity }>
 *   CellRenderer / CompositeRenderer poll this map each RAF tick.
 */

import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  type FC,
} from 'react'

// ── Cell params shape (mirrors channels/cell/ * /params.json) ─────────────────

interface BBox {
  x: number
  y: number
  w: number
  h: number
  z: number
}

export interface CellParamsRaw {
  cell_id: string
  species: string
  label: string
  bbox: BBox
  opacity: number
  epoch: number
  fill_color: string
  stroke_color: string
}

// ── Theatre.js object live-prop shape ─────────────────────────────────────────

export interface CellTheatreProps {
  x: number
  y: number
  opacity: number
}

// ── Static cell params (inlined from channels/cell/ * /params.json) ───────────
// Update by running: cat channels/cell/ * /params.json | jq -s '.'

const CELL_PARAMS: CellParamsRaw[] = [
  {
    cell_id: 'input_embed',
    species: 'cil-vector',
    label: 'Input Embedding',
    bbox: { x: 220, y: 40,  w: 160, h: 50, z: 5 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#2E7D32',
    stroke_color: '#2E7D32',
  },
  {
    cell_id: 'pos_encode',
    species: 'cil-vector',
    label: 'Positional Encoding',
    bbox: { x: 220, y: 150, w: 160, h: 50, z: 5 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#2E7D32',
    stroke_color: '#2E7D32',
  },
  {
    cell_id: 'self_attn',
    species: 'cil-eye',
    label: 'Multi-Head Attention',
    bbox: { x: 220, y: 260, w: 160, h: 50, z: 3 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#3F51B5',
    stroke_color: '#3F51B5',
  },
  {
    cell_id: 'add_norm1',
    species: 'cil-plus',
    label: 'Add and Norm',
    bbox: { x: 220, y: 370, w: 140, h: 40, z: 5 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#1E88E5',
    stroke_color: '#1E88E5',
  },
  {
    cell_id: 'ffn',
    species: 'cil-bolt',
    label: 'Feed Forward',
    bbox: { x: 220, y: 470, w: 140, h: 45, z: 3 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#FF6F00',
    stroke_color: '#FF6F00',
  },
  {
    cell_id: 'add_norm2',
    species: 'cil-plus',
    label: 'Add and Norm',
    bbox: { x: 220, y: 575, w: 140, h: 40, z: 5 },
    opacity: 0.05,
    epoch: 9,
    fill_color: '#1E88E5',
    stroke_color: '#1E88E5',
  },
  {
    cell_id: 'output',
    species: 'cil-arrow-right',
    label: 'Output',
    bbox: { x: 220, y: 675, w: 160, h: 50, z: 3 },
    opacity: 0.0937,
    epoch: 9,
    fill_color: '#455A64',
    stroke_color: '#455A64',
  },
]

// ── Epoch count (sequence length in seconds = epoch count) ────────────────────
const TOTAL_EPOCHS = 4

// ── Keyframe helpers ──────────────────────────────────────────────────────────
//
// Each prop uses a simple linear ramp: at epoch 0 cells are dormant
// (opacity=0, y shifted up by 30px), and by epoch TOTAL_EPOCHS they
// reach the full params.json values.
//
// For Theatre.js OnDiskState keyframes the handles use the standard
// bezier defaults: [0, 0, 1, 1] (linear).

type KF = {
  id: string
  position: number
  value: number
  handles: [number, number, number, number]
  connectedRight: boolean
}

function makeKFs(
  prop: 'x' | 'y' | 'opacity',
  cell: CellParamsRaw,
): KF[] {
  const target = prop === 'x' ? cell.bbox.x
               : prop === 'y' ? cell.bbox.y
               : cell.opacity

  const initial = prop === 'x' ? cell.bbox.x
                : prop === 'y' ? cell.bbox.y - 30   // slide in from above
                : 0                                  // fade in from 0

  const kfs: KF[] = []
  for (let e = 0; e <= TOTAL_EPOCHS; e++) {
    const t = e / TOTAL_EPOCHS
    kfs.push({
      id: `${cell.cell_id}_${prop}_e${e}`,
      position: e,                              // 1 epoch = 1 second
      value: initial + (target - initial) * t,
      handles: [0, 0, 1, 1],
      connectedRight: e < TOTAL_EPOCHS,
    })
  }
  return kfs
}

// ── Build Theatre.js OnDiskState ──────────────────────────────────────────────
//
// Structure mirrors @theatre/core/src/types/private/core.ts → OnDiskState.
// trackIdByPropPath uses JSON-encoded prop paths e.g. '["x"]'.

function buildTheatreState() {
  type TrackEntry = {
    type: 'BasicKeyframedTrack'
    keyframes: Record<string, KF>
  }
  type ObjectEntry = {
    trackIdByPropPath: Record<string, string>
    trackData: Record<string, TrackEntry>
  }

  const tracksByObject: Record<string, ObjectEntry> = {}

  for (const cell of CELL_PARAMS) {
    const trackIdByPropPath: Record<string, string> = {}
    const trackData: Record<string, TrackEntry> = {}

    for (const prop of ['x', 'y', 'opacity'] as const) {
      const trackId = `${cell.cell_id}__${prop}`
      const propPath = JSON.stringify([prop])         // '["x"]'

      trackIdByPropPath[propPath] = trackId

      const kfs = makeKFs(prop, cell)
      const kfMap: Record<string, KF> = {}
      for (const kf of kfs) kfMap[kf.id] = kf

      trackData[trackId] = {
        type: 'BasicKeyframedTrack',
        keyframes: kfMap,
      }
    }

    tracksByObject[cell.cell_id] = { trackIdByPropPath, trackData }
  }

  return {
    sheetsById: {
      'cell-evolution': {
        staticOverrides: { byObject: {} },
        sequence: {
          type: 'PositionalSequence' as const,
          length: TOTAL_EPOCHS,
          subUnitsPerUnit: 30,
          tracksByObject,
        },
      },
    },
    definitionVersion: 'Theatre_Core_State_v4' as const,
    revisionHistory: ['astro-svgfigure-m165'],
  }
}

const THEATRE_STATE = buildTheatreState()

// ── Theatre.js module ref (dynamically imported CDN ESM) ─────────────────────

type TheatreCore = {
  getProject: (id: string, config?: { state?: unknown }) => unknown
  onChange: (pointer: unknown, cb: (v: number) => void) => () => void
  val: (pointer: unknown) => number
  types: {
    number: (defaultVal: number, opts?: { range?: [number, number] }) => unknown
  }
}

// ── EpochTimeline props ───────────────────────────────────────────────────────

export interface EpochTimelineProps {
  /** Controlled epoch (0–TOTAL_EPOCHS). Scrubs sequence to that epoch. */
  epoch?: number
  /** Called when the timeline scrubs; parent can sync its own epoch state. */
  onEpochChange?: (epoch: number) => void
  /** Show debug cell-state table (default: true) */
  showDebug?: boolean
}

interface CellLiveState {
  cell_id: string
  label: string
  fill_color: string
  props: CellTheatreProps
}

// ── Global window bridge for PixiJS renderer ──────────────────────────────────
//
// CellRenderer polls window.__theatreState each frame to nudge containers.
// Type: Map<string, CellTheatreProps>

declare global {
  interface Window {
    __theatreState?: Map<string, CellTheatreProps>
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

const EpochTimeline: FC<EpochTimelineProps> = ({
  epoch,
  onEpochChange,
  showDebug = true,
}) => {
  const [liveStates, setLiveStates] = useState<CellLiveState[]>(() =>
    CELL_PARAMS.map(c => ({
      cell_id:    c.cell_id,
      label:      c.label,
      fill_color: c.fill_color,
      props:      { x: c.bbox.x, y: c.bbox.y - 30, opacity: 0 },
    })),
  )
  const [position, setPosition] = useState(0)   // 0 – TOTAL_EPOCHS seconds
  const [playing, setPlaying]   = useState(false)
  const [ready, setReady]       = useState(false)
  const isDragging              = useRef(false)

  // Theatre.js refs (populated after CDN import)
  const sheetRef    = useRef<unknown>(null)
  const sequenceRef = useRef<unknown>(null)
  const coreRef     = useRef<TheatreCore | null>(null)

  // ── Bootstrap Theatre.js from CDN ESM ──────────────────────────────────────
  useEffect(() => {
    // Initialise the global window bridge so PixiJS can read live props
    if (typeof window !== 'undefined') {
      window.__theatreState = new Map<string, CellTheatreProps>()
      for (const c of CELL_PARAMS) {
        window.__theatreState.set(c.cell_id, { x: c.bbox.x, y: c.bbox.y - 30, opacity: 0 })
      }
    }

    let cancelled = false
    const unsubs: Array<() => void> = []

    async function boot() {
      // Dynamic import of Theatre.js core from CDN
      // (swap to local import once @theatre/core tsconfig alias is wired)
      const core = (await import(
        /* @vite-ignore */
        'https://cdn.theatrejs.com/core/0.7.0/theatre.esm.js'
      )) as TheatreCore

      if (cancelled) return
      coreRef.current = core

      const { getProject, onChange, types } = core

      // Create project with pre-baked keyframe state
      const project = getProject('astro-svgfigure', { state: THEATRE_STATE }) as {
        sheet: (name: string) => {
          object: (key: string, props: Record<string, unknown>) => {
            props: Record<string, unknown>
          }
          sequence: {
            position: number
            play: (opts?: { range?: [number, number]; rate?: number; iterationCount?: number }) => Promise<boolean>
            pause: () => void
          }
        }
        ready: Promise<void>
      }

      await project.ready
      if (cancelled) return

      const sheet    = project.sheet('cell-evolution')
      const sequence = sheet.sequence
      sheetRef.current    = sheet
      sequenceRef.current = sequence

      // Register SheetObjects — one per cell, props: x, y, opacity
      for (const cell of CELL_PARAMS) {
        const obj = sheet.object(cell.cell_id, {
          x:       types.number(cell.bbox.x,    { range: [0, 900] }),
          y:       types.number(cell.bbox.y,    { range: [-100, 900] }),
          opacity: types.number(cell.opacity,   { range: [0, 1] }),
        })

        // Subscribe: update React state + PixiJS bridge on every prop change
        const unsub = onChange(
          (obj.props as Record<string, unknown>)['x'],
          (_x: number) => {
            // Read all three props together for the update
            const x       = _x
            const y       = (core.val as (p: unknown) => number)((obj.props as Record<string, unknown>)['y'])
            const opacity = (core.val as (p: unknown) => number)((obj.props as Record<string, unknown>)['opacity'])

            // Push to PixiJS bridge
            if (typeof window !== 'undefined' && window.__theatreState) {
              window.__theatreState.set(cell.cell_id, { x, y, opacity })
            }

            // Update React debug table
            setLiveStates(prev =>
              prev.map(s =>
                s.cell_id === cell.cell_id
                  ? { ...s, props: { x, y, opacity } }
                  : s,
              ),
            )
          },
        )
        unsubs.push(unsub)
      }

      // Also track sequence position to keep scrubber in sync
      const posUnsub = onChange(
        (sequence as unknown as { pointer: { position: unknown } }).pointer.position,
        (pos: number) => {
          setPosition(pos)
          onEpochChange?.(Math.round(Math.min(TOTAL_EPOCHS, Math.max(0, pos))))
        },
      )
      unsubs.push(posUnsub)

      setReady(true)
    }

    boot().catch(console.error)

    return () => {
      cancelled = true
      unsubs.forEach(u => u())
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Sync external epoch prop → sequence position ──────────────────────────
  useEffect(() => {
    if (epoch !== undefined && !isDragging.current && sequenceRef.current) {
      ;(sequenceRef.current as { position: number }).position = epoch
      setPosition(epoch)
    }
  }, [epoch])

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const pos = parseFloat(e.target.value)
      if (sequenceRef.current) {
        ;(sequenceRef.current as { position: number }).position = pos
      }
      setPosition(pos)
      onEpochChange?.(Math.round(pos))
    },
    [onEpochChange],
  )

  const handlePlay = useCallback(async () => {
    if (!sequenceRef.current) return
    const seq = sequenceRef.current as {
      play: (opts: { range: [number, number]; rate: number }) => Promise<boolean>
    }
    setPlaying(true)
    await seq.play({ range: [0, TOTAL_EPOCHS], rate: 1 })
    setPlaying(false)
  }, [])

  const handlePause = useCallback(() => {
    if (!sequenceRef.current) return
    ;(sequenceRef.current as { pause: () => void }).pause()
    setPlaying(false)
  }, [])

  const jumpToEpoch = useCallback(
    (e: number) => {
      if (sequenceRef.current) {
        ;(sequenceRef.current as { position: number }).position = e
      }
      setPosition(e)
      onEpochChange?.(e)
    },
    [onEpochChange],
  )

  const currentEpoch = Math.min(TOTAL_EPOCHS, Math.round(position))

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      style={{
        fontFamily: 'monospace',
        background: '#0d1117',
        color: '#c9d1d9',
        padding: '16px',
        borderRadius: '8px',
        border: '1px solid #21262d',
        maxWidth: 540,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span
          style={{
            fontSize: 11,
            color: ready ? '#3fb950' : '#f0883e',
            textTransform: 'uppercase',
            letterSpacing: 1,
          }}
        >
          Theatre.js · astro-svgfigure / cell-evolution
          {ready ? ' ✓' : ' (loading…)'}
        </span>
      </div>

      {/* Epoch label */}
      <div style={{ fontSize: 28, fontWeight: 700, color: '#58a6ff', marginBottom: 8 }}>
        Epoch {currentEpoch}
        <span style={{ fontSize: 13, color: '#6e7681', marginLeft: 10, fontWeight: 400 }}>
          {position.toFixed(2)}s / {TOTAL_EPOCHS}.00s
        </span>
      </div>

      {/* Sequence scrub bar */}
      <input
        type="range"
        min={0}
        max={TOTAL_EPOCHS}
        step={0.01}
        value={position}
        disabled={!ready}
        onMouseDown={() => { isDragging.current = true }}
        onMouseUp={()   => { isDragging.current = false }}
        onChange={handleScrub}
        style={{ width: '100%', accentColor: '#58a6ff', marginBottom: 12 }}
      />

      {/* Epoch jump buttons */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: '#6e7681',
          marginBottom: 14,
        }}
      >
        {Array.from({ length: TOTAL_EPOCHS + 1 }, (_, e) => (
          <button
            key={e}
            disabled={!ready}
            onClick={() => jumpToEpoch(e)}
            style={{
              background: e === currentEpoch ? '#1f6feb' : 'transparent',
              border: '1px solid #30363d',
              borderRadius: 4,
              color: '#c9d1d9',
              padding: '2px 8px',
              cursor: ready ? 'pointer' : 'default',
              fontSize: 11,
            }}
          >
            E{e}
          </button>
        ))}
      </div>

      {/* Play / Pause / Reset */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          disabled={!ready}
          onClick={playing ? handlePause : handlePlay}
          style={{
            background: !ready ? '#2d333b' : playing ? '#b91c1c' : '#238636',
            border: 'none',
            borderRadius: 6,
            color: '#fff',
            padding: '6px 18px',
            cursor: ready ? 'pointer' : 'default',
            fontSize: 13,
            fontFamily: 'monospace',
          }}
        >
          {playing ? '⏸ Pause' : `▶ Play 0→${TOTAL_EPOCHS}s`}
        </button>
        <button
          disabled={!ready}
          onClick={() => jumpToEpoch(0)}
          style={{
            background: 'transparent',
            border: '1px solid #30363d',
            borderRadius: 6,
            color: '#c9d1d9',
            padding: '6px 12px',
            cursor: ready ? 'pointer' : 'default',
            fontSize: 13,
            fontFamily: 'monospace',
          }}
        >
          ↩ Reset
        </button>
      </div>

      {/* Cell SheetObject state table */}
      {showDebug && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr
              style={{
                color: '#6e7681',
                textAlign: 'left',
                borderBottom: '1px solid #21262d',
              }}
            >
              <th style={{ paddingBottom: 4, paddingRight: 8 }}>cell</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>x</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>y</th>
              <th style={{ paddingBottom: 4 }}>opacity</th>
            </tr>
          </thead>
          <tbody>
            {liveStates.map(s => (
              <tr key={s.cell_id} style={{ borderBottom: '1px solid #161b22' }}>
                <td
                  style={{
                    padding: '3px 8px 3px 0',
                    color: s.fill_color,
                    fontWeight: 600,
                  }}
                >
                  {s.cell_id}
                </td>
                <td style={{ paddingRight: 6 }}>{s.props.x.toFixed(1)}</td>
                <td style={{ paddingRight: 6 }}>{s.props.y.toFixed(1)}</td>
                <td>
                  <span
                    style={{
                      display: 'inline-block',
                      width: 40,
                      textAlign: 'right',
                      color: s.props.opacity > 0.005 ? '#3fb950' : '#6e7681',
                    }}
                  >
                    {s.props.opacity.toFixed(4)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ marginTop: 10, fontSize: 10, color: '#484f58' }}>
        Theatre.js @theatre/core 0.7.0 · Sequence position = epoch number ·
        window.__theatreState → PixiJS
      </div>
    </div>
  )
}

export default EpochTimeline

// ── Named exports for pipeline page integration ───────────────────────────────

/** Epoch count (sequence length in seconds) */
export const EPOCH_COUNT = TOTAL_EPOCHS

/** Raw cell params inlined from channels/cell/ * /params.json */
export { CELL_PARAMS }

/** Pre-baked Theatre.js OnDiskState with keyframes for all cells */
export { THEATRE_STATE }

/** Epoch → seconds helper */
export const epochToSeconds = (epoch: number): number =>
  Math.max(0, Math.min(TOTAL_EPOCHS, epoch))

/** Seconds → epoch (rounded) */
export const secondsToEpoch = (s: number): number =>
  Math.round(Math.max(0, Math.min(TOTAL_EPOCHS, s)))
