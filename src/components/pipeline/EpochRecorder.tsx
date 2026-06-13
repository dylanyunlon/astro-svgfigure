/**
 * EpochRecorder.tsx — Epoch state recorder & keyframe animator
 *
 * Fetches epoch history from /api/epochs, plays back cell bbox changes
 * with linear interpolation between keyframes, and renders per-cell
 * x/y-over-epoch sparkline charts.
 *
 * Zero external deps — pure React + useState/useEffect/useRef.
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FC,
} from 'react'

// ── Data structures ──────────────────────────────────────────────────────────

export interface CellSnapshot {
  cell_id: string
  x: number
  y: number
  w: number
  h: number
  opacity: number
  fill_color: string
}

export interface EpochFrame {
  epoch: number
  cells: CellSnapshot[]
}

// ── Fallback seed data (used when /api/epochs is unavailable) ────────────────

const SEED_FRAMES: EpochFrame[] = (() => {
  const cells = [
    { cell_id: 'input_embed', fill_color: '#2E7D32' },
    { cell_id: 'pos_encode',  fill_color: '#2E7D32' },
    { cell_id: 'self_attn',   fill_color: '#3F51B5' },
    { cell_id: 'add_norm1',   fill_color: '#1E88E5' },
    { cell_id: 'ffn',         fill_color: '#FF6F00' },
    { cell_id: 'add_norm2',   fill_color: '#1E88E5' },
    { cell_id: 'output',      fill_color: '#455A64' },
  ]

  // Base bboxes at epoch 0
  const bases: Record<string, { x: number; y: number; w: number; h: number }> = {
    input_embed: { x: 220, y: 40,  w: 160, h: 50 },
    pos_encode:  { x: 220, y: 150, w: 160, h: 50 },
    self_attn:   { x: 220, y: 260, w: 160, h: 50 },
    add_norm1:   { x: 230, y: 370, w: 140, h: 40 },
    ffn:         { x: 230, y: 470, w: 140, h: 45 },
    add_norm2:   { x: 230, y: 575, w: 140, h: 40 },
    output:      { x: 220, y: 675, w: 160, h: 50 },
  }

  return Array.from({ length: 5 }, (_, epoch) => ({
    epoch,
    cells: cells.map(({ cell_id, fill_color }) => {
      const b = bases[cell_id]
      // Each epoch: cells drift slightly and grow in opacity
      const t = epoch / 4
      return {
        cell_id,
        fill_color,
        x: b.x + Math.sin(epoch * 0.8 + cell_id.length) * 8 * t,
        y: b.y - epoch * 3,
        w: b.w + epoch * 2,
        h: b.h + epoch,
        opacity: 0.05 + t * 0.8,
      }
    }),
  }))
})()

// ── Linear interpolation helpers ─────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Given a fractional epoch position (e.g. 1.37), find the surrounding
 * keyframes and linearly interpolate all cell props.
 */
function interpolateFrame(frames: EpochFrame[], pos: number): CellSnapshot[] {
  if (frames.length === 0) return []

  const maxEpoch = frames[frames.length - 1].epoch
  const clampedPos = Math.max(0, Math.min(maxEpoch, pos))

  // Find surrounding frame indices
  let loIdx = 0
  for (let i = 0; i < frames.length - 1; i++) {
    if (frames[i].epoch <= clampedPos && clampedPos <= frames[i + 1].epoch) {
      loIdx = i
      break
    }
    if (frames[i].epoch > clampedPos) break
    loIdx = i
  }
  const hiIdx = Math.min(loIdx + 1, frames.length - 1)

  const lo = frames[loIdx]
  const hi = frames[hiIdx]

  if (lo === hi) return lo.cells.map(c => ({ ...c }))

  const span = hi.epoch - lo.epoch
  const t = span === 0 ? 0 : (clampedPos - lo.epoch) / span

  // Build a map for hi cells for O(1) lookup
  const hiMap = new Map(hi.cells.map(c => [c.cell_id, c]))

  return lo.cells.map(lc => {
    const hc = hiMap.get(lc.cell_id)
    if (!hc) return { ...lc }
    return {
      cell_id:    lc.cell_id,
      fill_color: lc.fill_color,
      x:          lerp(lc.x, hc.x, t),
      y:          lerp(lc.y, hc.y, t),
      w:          lerp(lc.w, hc.w, t),
      h:          lerp(lc.h, hc.h, t),
      opacity:    lerp(lc.opacity, hc.opacity, t),
    }
  })
}

// ── Inline SVG sparkline chart ────────────────────────────────────────────────

interface SparklineProps {
  frames: EpochFrame[]
  cellId: string
  field: 'x' | 'y'
  color: string
  width?: number
  height?: number
  currentPos: number
}

const Sparkline: FC<SparklineProps> = ({
  frames, cellId, field, color,
  width = 200, height = 48, currentPos,
}) => {
  if (frames.length < 2) return <span style={{ color: '#484f58', fontSize: 10 }}>no data</span>

  const values = frames.map(f => {
    const c = f.cells.find(c => c.cell_id === cellId)
    return c ? c[field] : 0
  })
  const epochs = frames.map(f => f.epoch)

  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const rangeV = maxV - minV || 1

  const minE = epochs[0]
  const maxE = epochs[epochs.length - 1]
  const rangeE = maxE - minE || 1

  const PAD = 4
  const W = width - PAD * 2
  const H = height - PAD * 2

  const toX = (e: number) => PAD + ((e - minE) / rangeE) * W
  const toY = (v: number) => PAD + (1 - (v - minV) / rangeV) * H

  const pts = frames.map((f, i) => `${toX(f.epoch)},${toY(values[i])}`).join(' ')

  // Cursor line at currentPos
  const cursorX = toX(Math.max(minE, Math.min(maxE, currentPos)))

  // Interpolated value at cursor
  const interpolated = interpolateFrame(frames, currentPos)
  const curCell = interpolated.find(c => c.cell_id === cellId)
  const curVal = curCell ? curCell[field] : null

  return (
    <svg
      width={width}
      height={height}
      style={{ display: 'block', overflow: 'visible' }}
    >
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map(t => (
        <line
          key={t}
          x1={PAD} x2={PAD + W}
          y1={PAD + t * H} y2={PAD + t * H}
          stroke="#21262d" strokeWidth={0.5}
        />
      ))}

      {/* Epoch markers */}
      {frames.map(f => (
        <line
          key={f.epoch}
          x1={toX(f.epoch)} x2={toX(f.epoch)}
          y1={PAD} y2={PAD + H}
          stroke="#30363d" strokeWidth={0.5} strokeDasharray="2,2"
        />
      ))}

      {/* Polyline */}
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.85}
      />

      {/* Data points */}
      {frames.map((f, i) => (
        <circle
          key={f.epoch}
          cx={toX(f.epoch)}
          cy={toY(values[i])}
          r={2.5}
          fill={color}
          opacity={0.9}
        />
      ))}

      {/* Cursor */}
      <line
        x1={cursorX} x2={cursorX}
        y1={PAD} y2={PAD + H}
        stroke="#f0f6fc" strokeWidth={1} opacity={0.6}
      />
      {curVal !== null && (
        <circle
          cx={cursorX}
          cy={toY(curVal)}
          r={3.5}
          fill="#f0f6fc"
          stroke={color}
          strokeWidth={1.5}
        />
      )}

      {/* Value label */}
      {curVal !== null && (
        <text
          x={cursorX + 4}
          y={PAD + 10}
          fontSize={9}
          fill="#f0f6fc"
          fontFamily="monospace"
        >
          {curVal.toFixed(1)}
        </text>
      )}
    </svg>
  )
}

// ── Mini cell preview rectangle ───────────────────────────────────────────────

interface CellPreviewProps {
  snapshot: CellSnapshot
  scale?: number
}

const CellPreview: FC<CellPreviewProps> = ({ snapshot, scale = 0.35 }) => {
  const svgW = 600 * scale
  const svgH = 780 * scale
  return (
    <svg
      width={svgW}
      height={svgH}
      viewBox="0 0 600 780"
      style={{ display: 'block', background: '#0d1117', borderRadius: 4 }}
    >
      <rect
        x={snapshot.x}
        y={snapshot.y}
        width={snapshot.w}
        height={snapshot.h}
        fill={snapshot.fill_color}
        opacity={snapshot.opacity}
        rx={4}
      />
      <text
        x={snapshot.x + snapshot.w / 2}
        y={snapshot.y + snapshot.h / 2 + 4}
        textAnchor="middle"
        fontSize={10}
        fill="#c9d1d9"
        fontFamily="monospace"
        opacity={Math.min(1, snapshot.opacity * 8)}
      >
        {snapshot.cell_id}
      </text>
    </svg>
  )
}

// ── EpochRecorder ─────────────────────────────────────────────────────────────

export interface EpochRecorderProps {
  /** Override API endpoint (default: /api/epochs) */
  apiUrl?: string
  /** FPS for playback animation loop (default: 30) */
  fps?: number
}

const EpochRecorder: FC<EpochRecorderProps> = ({
  apiUrl = '/api/epochs',
  fps = 30,
}) => {
  // ── State ──────────────────────────────────────────────────────────────────
  const [frames, setFrames]           = useState<EpochFrame[]>(SEED_FRAMES)
  const [loading, setLoading]         = useState(true)
  const [apiError, setApiError]       = useState<string | null>(null)
  const [pos, setPos]                 = useState(0)           // fractional epoch position
  const [playing, setPlaying]         = useState(false)
  const [speed, setSpeed]             = useState(1)           // playback speed multiplier
  const [selectedCell, setSelectedCell] = useState<string | null>(null)
  const [chartField, setChartField]   = useState<'x' | 'y'>('y')

  const rafRef    = useRef<number | null>(null)
  const lastRef   = useRef<number>(0)
  const posRef    = useRef(pos)        // keep ref in sync so RAF closure sees latest pos
  posRef.current  = pos

  // ── Fetch from /api/epochs ─────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setApiError(null)

    fetch(apiUrl)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<EpochFrame[]>
      })
      .then(data => {
        if (cancelled) return
        // Validate minimal shape
        if (
          Array.isArray(data) &&
          data.length > 0 &&
          typeof data[0].epoch === 'number' &&
          Array.isArray(data[0].cells)
        ) {
          // Sort by epoch ascending
          data.sort((a, b) => a.epoch - b.epoch)
          setFrames(data)
        } else {
          throw new Error('Unexpected shape from /api/epochs')
        }
      })
      .catch(err => {
        if (cancelled) return
        setApiError(`${err.message} — showing seed data`)
        // Keep SEED_FRAMES already in state
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [apiUrl])

  // ── Playback RAF loop ──────────────────────────────────────────────────────
  const maxEpoch = frames.length > 0 ? frames[frames.length - 1].epoch : 4

  const stopPlayback = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return

    lastRef.current = performance.now()

    function tick(now: number) {
      const dt = (now - lastRef.current) / 1000   // seconds elapsed
      lastRef.current = now

      const next = posRef.current + dt * speed
      if (next >= maxEpoch) {
        setPos(maxEpoch)
        stopPlayback()
        return
      }
      setPos(next)
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, speed, maxEpoch, stopPlayback])

  const handlePlay = useCallback(() => {
    if (pos >= maxEpoch) setPos(0)   // restart from beginning
    setPlaying(true)
  }, [pos, maxEpoch])

  const handlePause = useCallback(() => stopPlayback(), [stopPlayback])

  const handleReset = useCallback(() => {
    stopPlayback()
    setPos(0)
  }, [stopPlayback])

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    stopPlayback()
    setPos(parseFloat(e.target.value))
  }, [stopPlayback])

  // ── Interpolated snapshot at current position ──────────────────────────────
  const snapshot = interpolateFrame(frames, pos)

  // ── Epoch jump buttons ─────────────────────────────────────────────────────
  const epochIntegers = frames.map(f => f.epoch)
  const currentEpochInt = Math.round(pos)

  // ── Styles ─────────────────────────────────────────────────────────────────
  const panelStyle: React.CSSProperties = {
    fontFamily: 'monospace',
    background: '#0d1117',
    color: '#c9d1d9',
    padding: '16px',
    borderRadius: '8px',
    border: '1px solid #21262d',
    maxWidth: 780,
  }

  const btnStyle = (active?: boolean, danger?: boolean): React.CSSProperties => ({
    background: danger ? '#b91c1c' : active ? '#1f6feb' : '#21262d',
    border: '1px solid #30363d',
    borderRadius: 6,
    color: '#f0f6fc',
    padding: '5px 14px',
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: 'monospace',
  })

  const labelStyle: React.CSSProperties = {
    fontSize: 10,
    color: '#6e7681',
    textTransform: 'uppercase',
    letterSpacing: 1,
    display: 'block',
    marginBottom: 4,
  }

  return (
    <div style={panelStyle}>
      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <div>
          <span style={{ fontSize: 10, color: '#6e7681', textTransform: 'uppercase', letterSpacing: 1 }}>
            EpochRecorder
          </span>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#58a6ff', lineHeight: 1.2 }}>
            Epoch <span style={{ color: '#f0f6fc' }}>{currentEpochInt}</span>
            <span style={{ fontSize: 13, color: '#6e7681', fontWeight: 400, marginLeft: 10 }}>
              pos {pos.toFixed(3)} / {maxEpoch}
            </span>
          </div>
        </div>

        {loading && (
          <span style={{ fontSize: 11, color: '#e3b341', marginLeft: 'auto' }}>
            ⟳ fetching /api/epochs…
          </span>
        )}
        {apiError && (
          <span style={{ fontSize: 11, color: '#f85149', marginLeft: 'auto', maxWidth: 240 }}>
            ⚠ {apiError}
          </span>
        )}
      </div>

      {/* ── Scrub bar ── */}
      <div style={{ marginBottom: 10 }}>
        <label style={labelStyle}>scrub</label>
        <input
          type="range"
          min={0}
          max={maxEpoch}
          step={0.01}
          value={pos}
          onChange={handleScrub}
          style={{ width: '100%', accentColor: '#58a6ff' }}
        />
        {/* Epoch tick labels */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
          {epochIntegers.map(e => (
            <span
              key={e}
              style={{
                fontSize: 9,
                color: e === currentEpochInt ? '#58a6ff' : '#484f58',
                cursor: 'pointer',
                userSelect: 'none',
              }}
              onClick={() => { stopPlayback(); setPos(e) }}
            >
              E{e}
            </span>
          ))}
        </div>
      </div>

      {/* ── Transport controls ── */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
        <button style={btnStyle(!playing)} onClick={playing ? handlePause : handlePlay}>
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button style={btnStyle()} onClick={handleReset}>↩ Reset</button>

        {/* Speed selector */}
        <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginLeft: 8 }}>
          <span style={{ fontSize: 11, color: '#6e7681' }}>speed</span>
          {[0.25, 0.5, 1, 2].map(s => (
            <button
              key={s}
              style={{
                ...btnStyle(speed === s),
                padding: '4px 8px',
                fontSize: 11,
              }}
              onClick={() => setSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Jump to epoch buttons */}
        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
          {epochIntegers.map(e => (
            <button
              key={e}
              style={{
                ...btnStyle(e === currentEpochInt),
                padding: '4px 10px',
                fontSize: 11,
              }}
              onClick={() => { stopPlayback(); setPos(e) }}
            >
              E{e}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main content: preview + table ── */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        {/* Cell list with sparklines */}
        <div style={{ flex: 1, minWidth: 320 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
            <label style={{ ...labelStyle, marginBottom: 0 }}>cells</label>
            <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
              {(['x', 'y'] as const).map(f => (
                <button
                  key={f}
                  style={{ ...btnStyle(chartField === f), padding: '2px 8px', fontSize: 10 }}
                  onClick={() => setChartField(f)}
                >
                  {f}-axis
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              border: '1px solid #21262d',
              borderRadius: 6,
              overflow: 'hidden',
            }}
          >
            {snapshot.map(cell => {
              const isSelected = selectedCell === cell.cell_id
              return (
                <div
                  key={cell.cell_id}
                  onClick={() => setSelectedCell(isSelected ? null : cell.cell_id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '7px 10px',
                    background: isSelected ? '#161b22' : 'transparent',
                    borderBottom: '1px solid #21262d',
                    cursor: 'pointer',
                    transition: 'background 0.1s',
                  }}
                >
                  {/* Color dot */}
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: cell.fill_color, flexShrink: 0,
                  }} />

                  {/* Cell ID */}
                  <span style={{
                    color: cell.fill_color, fontWeight: 600, fontSize: 11,
                    width: 90, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {cell.cell_id}
                  </span>

                  {/* Sparkline */}
                  <div style={{ flex: 1 }}>
                    <Sparkline
                      frames={frames}
                      cellId={cell.cell_id}
                      field={chartField}
                      color={cell.fill_color}
                      width={160}
                      height={36}
                      currentPos={pos}
                    />
                  </div>

                  {/* Live values */}
                  <div style={{ textAlign: 'right', fontSize: 10, color: '#6e7681', flexShrink: 0 }}>
                    <div>x {cell.x.toFixed(1)}</div>
                    <div>y {cell.y.toFixed(1)}</div>
                    <div style={{ color: cell.opacity > 0.1 ? '#3fb950' : '#484f58' }}>
                      α {cell.opacity.toFixed(3)}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Selected cell detail panel */}
        {selectedCell && (() => {
          const cell = snapshot.find(c => c.cell_id === selectedCell)
          if (!cell) return null
          return (
            <div style={{
              width: 220,
              flexShrink: 0,
              background: '#161b22',
              border: '1px solid #30363d',
              borderRadius: 6,
              padding: 12,
            }}>
              <div style={{ ...labelStyle, color: cell.fill_color }}>{cell.cell_id}</div>

              <CellPreview snapshot={cell} scale={0.34} />

              <table style={{ width: '100%', fontSize: 11, marginTop: 10, borderCollapse: 'collapse' }}>
                <tbody>
                  {(['x', 'y', 'w', 'h', 'opacity'] as const).map(f => (
                    <tr key={f} style={{ borderBottom: '1px solid #21262d' }}>
                      <td style={{ color: '#6e7681', padding: '3px 0' }}>{f}</td>
                      <td style={{ textAlign: 'right', fontWeight: 600 }}>
                        {typeof cell[f] === 'number' ? (cell[f] as number).toFixed(f === 'opacity' ? 4 : 1) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Both axis sparklines for selected cell */}
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 10, color: '#6e7681', marginBottom: 4 }}>x over epochs</div>
                <Sparkline frames={frames} cellId={cell.cell_id} field="x"
                  color={cell.fill_color} width={196} height={44} currentPos={pos} />
                <div style={{ fontSize: 10, color: '#6e7681', marginTop: 8, marginBottom: 4 }}>y over epochs</div>
                <Sparkline frames={frames} cellId={cell.cell_id} field="y"
                  color={cell.fill_color} width={196} height={44} currentPos={pos} />
              </div>
            </div>
          )
        })()}
      </div>

      {/* ── Frame index table ── */}
      <div>
        <label style={labelStyle}>keyframes ({frames.length})</label>
        <div style={{
          display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4,
        }}>
          {frames.map(f => {
            const dist = Math.abs(pos - f.epoch)
            const isActive = dist < 0.5
            return (
              <div
                key={f.epoch}
                onClick={() => { stopPlayback(); setPos(f.epoch) }}
                style={{
                  flexShrink: 0,
                  padding: '6px 10px',
                  borderRadius: 5,
                  border: `1px solid ${isActive ? '#58a6ff' : '#30363d'}`,
                  background: isActive ? '#0c2040' : '#161b22',
                  cursor: 'pointer',
                  fontSize: 11,
                  color: isActive ? '#58a6ff' : '#8b949e',
                  minWidth: 56,
                  textAlign: 'center',
                }}
              >
                <div style={{ fontWeight: 700 }}>E{f.epoch}</div>
                <div style={{ fontSize: 9, color: '#484f58' }}>{f.cells.length} cells</div>
              </div>
            )
          })}
        </div>
      </div>

      <div style={{ marginTop: 10, fontSize: 10, color: '#484f58' }}>
        Linear interpolation · {frames.length} keyframes · {snapshot.length} cells
        {apiError ? ' · seed data' : ' · live /api/epochs'}
      </div>
    </div>
  )
}

export default EpochRecorder
