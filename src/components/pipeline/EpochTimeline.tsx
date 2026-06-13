/**
 * EpochTimeline.tsx — Theatre.js epoch controller for astro-svgfigure
 *
 * Maps epoch 0→4 onto a Theatre.js timeline 0s→4s.
 * Each cell gets a Theatre.js ISheetObject with props: x, y, w, h, opacity.
 *
 * ── Import strategy ─────────────────────────────────────────────────────────
 * @theatre/core is in upstream/theatre-js/core but has internal deps
 * (@theatre/dataverse, @theatre/utils) not wired via tsconfig paths yet.
 *
 * Option A — add to tsconfig.json paths + astro.config.ts aliases:
 *   "@theatre/core":      ["./upstream/theatre-js/core/src/index.ts"]
 *   "@theatre/dataverse": ["./upstream/theatre-js/dataverse/src/index.ts"]
 *
 * Option B — install published packages (runtime, no Studio needed):
 *   bun add @theatre/core@0.7.0
 *
 * Option C (used below) — CDN ESM shim for zero-config usage in browser:
 *   import { getProject, types } from 'https://cdn.theatrejs.com/core/0.7.0/theatre.esm.js'
 *   (works in Astro client:only islands / vanilla <script type="module">)
 *
 * Switch the import line when the monorepo aliases are configured.
 * ────────────────────────────────────────────────────────────────────────────
 */

// ── Theatre.js import (swap to local path once aliases are wired) ─────────────
// import { getProject, types } from '@theatre/core'                    // Option A/B
// import { getProject, types } from 'https://cdn.theatrejs.com/core/0.7.0/theatre.esm.js' // Option C
//
// Until the alias is set up we ship a lightweight facade that mirrors the
// Theatre.js public API surface used here, so the component is fully typed
// and functional without the upstream dep being importable.
// Replace the three lines below with a real Theatre.js import to enable
// the animation editor / scrubbing UI.

import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  type FC,
} from 'react'

// ── Cell params shape (mirrors channels/cell/*/params.json) ──────────────────

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

// ── Theatre.js object prop shape ─────────────────────────────────────────────

interface CellTheatreProps {
  x: number
  y: number
  w: number
  h: number
  opacity: number
}

// ── Minimal Theatre.js facade (replace with real import) ─────────────────────
//
// This recreates only the ISheetObject.onValuesChange + ISequence.position
// contract that EpochTimeline needs.  When @theatre/core is importable,
// delete this entire block and use getProject / types from there.

interface IFacadeObject {
  props: CellTheatreProps
  onValuesChange(cb: (vals: CellTheatreProps) => void): () => void
}

interface IFacadeSequence {
  position: number  // seconds, 0–4
  play(opts?: { range?: [number, number]; rate?: number }): Promise<void>
  pause(): void
}

interface IFacadeSheet {
  object(key: string, props: CellTheatreProps): IFacadeObject
  sequence: IFacadeSequence
}

interface IFacadeProject {
  sheet(label: string): IFacadeSheet
}

/** Keyframe lerp for a single prop across epoch steps 0-4 */
function lerpEpoch(
  from: number,
  to: number,
  t: number,               // 0–1 within the epoch step
  epochIdx: number,        // which step we're in
  totalEpochs: number,
): number {
  // linear: each epoch occupies 1/totalEpochs of the [0,1] range
  const globalT = (epochIdx + t) / totalEpochs
  return from + (to - from) * globalT
}

function createFacadeProject(_name: string): IFacadeProject {
  return {
    sheet(_label: string): IFacadeSheet {
      const listeners = new Map<string, Set<(v: CellTheatreProps) => void>>()
      const objects = new Map<string, IFacadeObject>()
      let _position = 0 // seconds
      let _raf: number | null = null

      const sequence: IFacadeSequence = {
        get position() { return _position },
        set position(v: number) {
          _position = Math.max(0, Math.min(4, v))
          // notify all objects
          objects.forEach((obj, key) => {
            const cbs = listeners.get(key)
            if (cbs) cbs.forEach(cb => cb(obj.props))
          })
        },
        async play({ range = [0, 4], rate = 1 } = {}) {
          const [start, end] = range
          _position = start
          return new Promise<void>(resolve => {
            let last = performance.now()
            function tick(now: number) {
              const dt = (now - last) / 1000
              last = now
              sequence.position = _position + dt * rate
              if (_position >= end) {
                sequence.position = end
                resolve()
                return
              }
              _raf = requestAnimationFrame(tick)
            }
            _raf = requestAnimationFrame(tick)
          })
        },
        pause() {
          if (_raf !== null) {
            cancelAnimationFrame(_raf)
            _raf = null
          }
        },
      }

      return {
        sequence,
        object(key: string, defaultProps: CellTheatreProps): IFacadeObject {
          // Derive keyframed props from position (0–4 s = epoch 0–4)
          // Simple approach: opacity fades in linearly from 0.05 to defaultProps.opacity
          // x/y/w/h animate from slightly offset starting positions
          function computeProps(pos: number): CellTheatreProps {
            const t = Math.min(pos / 4, 1) // normalised 0–1
            return {
              x: defaultProps.x,
              y: defaultProps.y - (1 - t) * 20,        // slide up slightly on enter
              w: defaultProps.w,
              h: defaultProps.h,
              opacity: 0.0 + t * defaultProps.opacity,  // fade in to target opacity
            }
          }

          let _current = computeProps(sequence.position)
          if (!listeners.has(key)) listeners.set(key, new Set())

          const obj: IFacadeObject = {
            get props() {
              _current = computeProps(_position)
              return _current
            },
            onValuesChange(cb) {
              listeners.get(key)!.add(cb)
              cb(_current)
              return () => listeners.get(key)!.delete(cb)
            },
          }
          objects.set(key, obj)
          return obj
        },
      }
    },
  }
}

// ── Static cell params (loaded from channels/cell/*/params.json) ─────────────
//
// In a server-rendered Astro page these would be imported via:
//   import addNorm1 from '/channels/cell/add_norm1/params.json'
// But as a client React island we inline the values here.
// Update by re-running: cat channels/cell/*/params.json | jq -s '.'

const CELL_PARAMS: CellParamsRaw[] = [
  {
    cell_id: 'input_embed',
    species: 'cil-vector',
    label: 'Input Embedding',
    bbox: { x: 220, y: 40,  w: 160, h: 50, z: 3 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#2E7D32',
    stroke_color: '#2E7D32',
  },
  {
    cell_id: 'pos_encode',
    species: 'cil-vector',
    label: 'Positional Encoding',
    bbox: { x: 220, y: 150, w: 160, h: 50, z: 5 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#2E7D32',
    stroke_color: '#2E7D32',
  },
  {
    cell_id: 'self_attn',
    species: 'cil-eye',
    label: 'Multi-Head Attention',
    bbox: { x: 220, y: 260, w: 160, h: 50, z: 3 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#3F51B5',
    stroke_color: '#3F51B5',
  },
  {
    cell_id: 'add_norm1',
    species: 'cil-plus',
    label: 'Add and Norm',
    bbox: { x: 230, y: 370, w: 140, h: 40, z: 5 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#1E88E5',
    stroke_color: '#1E88E5',
  },
  {
    cell_id: 'ffn',
    species: 'cil-bolt',
    label: 'Feed Forward',
    bbox: { x: 230, y: 470, w: 140, h: 45, z: 3 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#FF6F00',
    stroke_color: '#FF6F00',
  },
  {
    cell_id: 'add_norm2',
    species: 'cil-plus',
    label: 'Add and Norm',
    bbox: { x: 230, y: 575, w: 140, h: 40, z: 5 },
    opacity: 0.05,
    epoch: 4,
    fill_color: '#1E88E5',
    stroke_color: '#1E88E5',
  },
  {
    cell_id: 'output',
    species: 'cil-arrow-right',
    label: 'Output',
    bbox: { x: 220, y: 675, w: 160, h: 50, z: 3 },
    opacity: 0.0937,
    epoch: 4,
    fill_color: '#455A64',
    stroke_color: '#455A64',
  },
]

// ── Theatre.js project + sheet singletons ────────────────────────────────────

// When @theatre/core is importable, replace createFacadeProject with:
//   const project = getProject('astro-svgfigure')
//   const sheet   = project.sheet('cell-evolution')

const theatreProject: IFacadeProject = createFacadeProject('astro-svgfigure')
const sheet = theatreProject.sheet('cell-evolution')

// Register one Theatre.js object per cell
const cellObjects = new Map<string, IFacadeObject>()
for (const cell of CELL_PARAMS) {
  // When using real @theatre/core, use types.compound / types.number:
  //
  //   sheet.object(cell.cell_id, {
  //     x:       types.number(cell.bbox.x, { range: [0, 800] }),
  //     y:       types.number(cell.bbox.y, { range: [0, 800] }),
  //     w:       types.number(cell.bbox.w, { range: [0, 400] }),
  //     h:       types.number(cell.bbox.h, { range: [0, 200] }),
  //     opacity: types.number(cell.opacity, { range: [0, 1] }),
  //   })
  //
  cellObjects.set(
    cell.cell_id,
    sheet.object(cell.cell_id, {
      x:       cell.bbox.x,
      y:       cell.bbox.y,
      w:       cell.bbox.w,
      h:       cell.bbox.h,
      opacity: cell.opacity,
    }),
  )
}

// ── EpochTimeline component ──────────────────────────────────────────────────

export interface EpochTimelineProps {
  /** Controlled epoch (0–4).  When provided the timeline scrubs to that epoch. */
  epoch?: number
  /** Called when the timeline scrubs; parent can sync its own epoch state. */
  onEpochChange?: (epoch: number) => void
  /** Show the debug overlay panel (default: true in dev) */
  showDebug?: boolean
}

interface CellLiveState {
  cell_id: string
  label: string
  fill_color: string
  props: CellTheatreProps
}

/**
 * EpochTimeline
 *
 * Drop into any Astro page / React island:
 *
 *   import EpochTimeline from '@/components/pipeline/EpochTimeline'
 *   <EpochTimeline epoch={currentEpoch} onEpochChange={setEpoch} />
 */
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
      props:      { x: c.bbox.x, y: c.bbox.y, w: c.bbox.w, h: c.bbox.h, opacity: 0 },
    })),
  )
  const [position, setPosition] = useState(0) // 0–4 s
  const [playing, setPlaying]   = useState(false)
  const isDragging               = useRef(false)

  // Subscribe to all cell objects' value changes
  useEffect(() => {
    const unsubs: Array<() => void> = []
    for (const cell of CELL_PARAMS) {
      const obj = cellObjects.get(cell.cell_id)!
      unsubs.push(
        obj.onValuesChange(vals => {
          setLiveStates(prev =>
            prev.map(s =>
              s.cell_id === cell.cell_id ? { ...s, props: { ...vals } } : s,
            ),
          )
        }),
      )
    }
    return () => unsubs.forEach(u => u())
  }, [])

  // Sync external epoch prop → sequence position (epoch 0–4 = 0s–4s)
  useEffect(() => {
    if (epoch !== undefined && !isDragging.current) {
      sheet.sequence.position = epoch
      setPosition(epoch)
    }
  }, [epoch])

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const pos = parseFloat(e.target.value)
      sheet.sequence.position = pos
      setPosition(pos)
      onEpochChange?.(Math.round(pos))
    },
    [onEpochChange],
  )

  const handlePlay = useCallback(async () => {
    setPlaying(true)
    await sheet.sequence.play({ range: [0, 4], rate: 1 })
    setPlaying(false)
  }, [])

  const handlePause = useCallback(() => {
    sheet.sequence.pause()
    setPlaying(false)
  }, [])

  // Current epoch index derived from position
  const currentEpoch = Math.min(4, Math.round(position))

  return (
    <div
      style={{
        fontFamily: 'monospace',
        background: '#0d1117',
        color: '#c9d1d9',
        padding: '16px',
        borderRadius: '8px',
        border: '1px solid #21262d',
        maxWidth: 520,
      }}
    >
      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 11, color: '#6e7681', textTransform: 'uppercase', letterSpacing: 1 }}>
          Theatre.js · astro-svgfigure / cell-evolution
        </span>
      </div>

      {/* ── Epoch label ── */}
      <div style={{ fontSize: 28, fontWeight: 700, color: '#58a6ff', marginBottom: 8 }}>
        Epoch {currentEpoch}
        <span style={{ fontSize: 13, color: '#6e7681', marginLeft: 10, fontWeight: 400 }}>
          {position.toFixed(2)}s / 4.00s
        </span>
      </div>

      {/* ── Scrub bar (Theatre.js sequence.position) ── */}
      <input
        type="range"
        min={0}
        max={4}
        step={0.01}
        value={position}
        onMouseDown={() => { isDragging.current = true }}
        onMouseUp={()   => { isDragging.current = false }}
        onChange={handleScrub}
        style={{ width: '100%', accentColor: '#58a6ff', marginBottom: 12 }}
      />

      {/* ── Epoch markers ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#6e7681', marginBottom: 14 }}>
        {[0, 1, 2, 3, 4].map(e => (
          <button
            key={e}
            onClick={() => {
              sheet.sequence.position = e
              setPosition(e)
              onEpochChange?.(e)
            }}
            style={{
              background: e === currentEpoch ? '#1f6feb' : 'transparent',
              border: '1px solid #30363d',
              borderRadius: 4,
              color: '#c9d1d9',
              padding: '2px 8px',
              cursor: 'pointer',
              fontSize: 11,
            }}
          >
            E{e}
          </button>
        ))}
      </div>

      {/* ── Play / Pause ── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          onClick={playing ? handlePause : handlePlay}
          style={{
            background: playing ? '#b91c1c' : '#238636',
            border: 'none',
            borderRadius: 6,
            color: '#fff',
            padding: '6px 18px',
            cursor: 'pointer',
            fontSize: 13,
            fontFamily: 'monospace',
          }}
        >
          {playing ? '⏸ Pause' : '▶ Play 0→4s'}
        </button>
        <button
          onClick={() => {
            sheet.sequence.position = 0
            setPosition(0)
            onEpochChange?.(0)
          }}
          style={{
            background: 'transparent',
            border: '1px solid #30363d',
            borderRadius: 6,
            color: '#c9d1d9',
            padding: '6px 12px',
            cursor: 'pointer',
            fontSize: 13,
            fontFamily: 'monospace',
          }}
        >
          ↩ Reset
        </button>
      </div>

      {/* ── Cell object state table ── */}
      {showDebug && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ color: '#6e7681', textAlign: 'left', borderBottom: '1px solid #21262d' }}>
              <th style={{ paddingBottom: 4, paddingRight: 8 }}>cell</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>x</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>y</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>w</th>
              <th style={{ paddingBottom: 4, paddingRight: 6 }}>h</th>
              <th style={{ paddingBottom: 4 }}>opacity</th>
            </tr>
          </thead>
          <tbody>
            {liveStates.map(s => (
              <tr key={s.cell_id} style={{ borderBottom: '1px solid #161b22' }}>
                <td style={{ padding: '3px 8px 3px 0', color: s.fill_color, fontWeight: 600 }}>
                  {s.cell_id}
                </td>
                <td style={{ paddingRight: 6 }}>{s.props.x.toFixed(1)}</td>
                <td style={{ paddingRight: 6 }}>{s.props.y.toFixed(1)}</td>
                <td style={{ paddingRight: 6 }}>{s.props.w.toFixed(1)}</td>
                <td style={{ paddingRight: 6 }}>{s.props.h.toFixed(1)}</td>
                <td>
                  <span
                    style={{
                      display: 'inline-block',
                      width: 36,
                      textAlign: 'right',
                      color: s.props.opacity > 0.04 ? '#3fb950' : '#6e7681',
                    }}
                  >
                    {s.props.opacity.toFixed(3)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ marginTop: 10, fontSize: 10, color: '#484f58' }}>
        Theatre.js facade active — swap import to @theatre/core for Studio UI
      </div>
    </div>
  )
}

export default EpochTimeline

// ── Named exports for pipeline page integration ───────────────────────────────

/** Raw access to the Theatre.js sheet — attach Studio in dev with:
 *    import '@theatre/studio'
 *    studio.initialize()
 *    studio.extend(sheet)  // or pass to getEditorSnapshot
 */
export { sheet as cellEvolutionSheet }

/** Live cell object map keyed by cell_id */
export { cellObjects }

/** Epoch → seconds helper (Theatre.js timeline mapping) */
export const epochToSeconds = (epoch: number): number =>
  Math.max(0, Math.min(4, epoch))

/** Seconds → epoch (rounded) */
export const secondsToEpoch = (s: number): number =>
  Math.round(Math.max(0, Math.min(4, s)))
