/**
 * CellInteraction.ts — AT Hydra cell interaction layer
 *
 * Handles:
 *  1. Mouse position → cell hitTest (bbox containment)
 *  2. Hover: bloom intensity boost + floating tooltip + neighbour repulsion
 *  3. Click: detail panel expansion + sendPrompt + pulse animation
 *  4. Mouse move → particle trail (Mira experience)
 *
 * Designed to be mounted on top of the PixiJS canvas used by CellRenderer.
 * All visual feedback that lives outside PixiJS (tooltip, panel) is done via
 * lightweight DOM overlays so we don't bloat the WebGL context.
 */

import type { CellParams } from '../components/pipeline/CellRenderer'

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellInteractionOptions {
  /** The canvas element that CellRenderer rendered into */
  canvas: HTMLCanvasElement
  /** Same cells array passed to mountCellRenderer */
  cells: CellParams[]
  /** Coordinate transform: canvas-space → cell-space (from CellRenderer layout) */
  transform: { scale: number; offX: number; offY: number }
  /** Called when a cell is clicked with the cell's label */
  sendPrompt?: (prompt: string) => void
  /** Called when a cell is hovered (undefined = unhover) */
  onHover?: (cell: CellParams | null) => void
  /** Called when a cell is clicked */
  onClick?: (cell: CellParams) => void
}

export interface CellInteractionHandle {
  destroy: () => void
}

// ── Constants ────────────────────────────────────────────────────────────────

const HOVER_BLOOM_CLASS    = 'cell-bloom-active'
const NEIGHBOUR_RADIUS     = 160   // px in cell-space; cells within this radius get nudged
const NEIGHBOUR_PUSH_PX    = 8     // max push distance in canvas-space
const PULSE_DURATION_MS    = 420
const PARTICLE_COUNT       = 18
const PARTICLE_LIFETIME_MS = 800

// ── Hit testing ──────────────────────────────────────────────────────────────

/**
 * hitTest — returns the topmost cell (highest z) whose bbox contains (mx, my)
 * where mx/my are in cell-coordinate space.
 */
function hitTest(cells: CellParams[], mx: number, my: number): CellParams | null {
  let best: CellParams | null = null
  for (const cell of cells) {
    const { x, y, w, h } = cell.bbox
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
      if (!best || cell.z > best.z) best = cell
    }
  }
  return best
}

/**
 * toCell — converts canvas-relative pixel coords to cell-space coords.
 */
function toCell(
  canvasX: number,
  canvasY: number,
  t: CellInteractionOptions['transform'],
): { x: number; y: number } {
  return {
    x: (canvasX - t.offX) / t.scale,
    y: (canvasY - t.offY) / t.scale,
  }
}

// ── Tooltip ──────────────────────────────────────────────────────────────────

function createTooltip(): HTMLDivElement {
  const el = document.createElement('div')
  el.id = 'cell-tooltip'
  Object.assign(el.style, {
    position:      'absolute',
    pointerEvents: 'none',
    background:    'rgba(13,17,23,0.85)',
    border:        '1px solid rgba(255,255,255,0.18)',
    color:         '#e6edf3',
    fontSize:      '11px',
    fontFamily:    'monospace',
    padding:       '4px 8px',
    borderRadius:  '4px',
    whiteSpace:    'nowrap',
    opacity:       '0',
    transition:    'opacity 0.15s ease',
    zIndex:        '9999',
    backdropFilter:'blur(6px)',
  })
  document.body.appendChild(el)
  return el
}

function showTooltip(el: HTMLDivElement, label: string, x: number, y: number): void {
  el.textContent = label
  el.style.left    = `${x + 14}px`
  el.style.top     = `${y - 8}px`
  el.style.opacity = '1'
}

function hideTooltip(el: HTMLDivElement): void {
  el.style.opacity = '0'
}

// ── Neighbour repulsion overlay ───────────────────────────────────────────────

/**
 * Returns neighbour cells within NEIGHBOUR_RADIUS and their push vectors
 * (canvas-space pixels). The hoveredCell itself is excluded.
 */
function computeNeighbourPush(
  cells: CellParams[],
  hovered: CellParams,
  t: CellInteractionOptions['transform'],
): Array<{ cell: CellParams; dx: number; dy: number }> {
  const hcx = (hovered.bbox.x + hovered.bbox.w / 2) * t.scale + t.offX
  const hcy = (hovered.bbox.y + hovered.bbox.h / 2) * t.scale + t.offY

  const result: Array<{ cell: CellParams; dx: number; dy: number }> = []
  for (const cell of cells) {
    if (cell.cell_id === hovered.cell_id) continue
    const cx = (cell.bbox.x + cell.bbox.w / 2) * t.scale + t.offX
    const cy = (cell.bbox.y + cell.bbox.h / 2) * t.scale + t.offY
    const dist = Math.hypot(cx - hcx, cy - hcy)
    if (dist > 0 && dist < NEIGHBOUR_RADIUS) {
      const factor = (1 - dist / NEIGHBOUR_RADIUS) * NEIGHBOUR_PUSH_PX
      result.push({
        cell,
        dx: ((cx - hcx) / dist) * factor,
        dy: ((cy - hcy) / dist) * factor,
      })
    }
  }
  return result
}

// ── Pulse animation ───────────────────────────────────────────────────────────

/**
 * Draws a pulsing ring on a 2-D overlay canvas centred on the clicked cell.
 */
function triggerPulse(
  overlayCtx: CanvasRenderingContext2D,
  cell: CellParams,
  t: CellInteractionOptions['transform'],
  fillColor: string,
): void {
  const cx = (cell.bbox.x + cell.bbox.w / 2) * t.scale + t.offX
  const cy = (cell.bbox.y + cell.bbox.h / 2) * t.scale + t.offY
  const baseR = (Math.max(cell.bbox.w, cell.bbox.h) / 2) * t.scale

  const startTime = performance.now()

  function frame(now: number) {
    const elapsed  = now - startTime
    const progress = Math.min(elapsed / PULSE_DURATION_MS, 1)
    const radius   = baseR * (1 + progress * 0.35)
    const alpha    = (1 - progress) * 0.7

    // Draw ring on overlay
    overlayCtx.save()
    overlayCtx.beginPath()
    overlayCtx.arc(cx, cy, radius, 0, Math.PI * 2)
    overlayCtx.strokeStyle = fillColor
    overlayCtx.globalAlpha = alpha
    overlayCtx.lineWidth   = 2.5
    overlayCtx.stroke()
    overlayCtx.restore()

    if (progress < 1) requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

// ── Particle trail ────────────────────────────────────────────────────────────

interface Particle {
  x: number; y: number
  vx: number; vy: number
  life: number       // 0–1
  radius: number
  color: string
  born: number       // timestamp
}

const PARTICLE_COLORS = ['#58a6ff', '#79c0ff', '#a5d6ff', '#3fb950', '#d2a8ff']

function spawnParticle(x: number, y: number): Particle {
  const angle  = Math.random() * Math.PI * 2
  const speed  = 0.3 + Math.random() * 0.7
  return {
    x, y,
    vx:     Math.cos(angle) * speed,
    vy:     Math.sin(angle) * speed,
    life:   1,
    radius: 1.5 + Math.random() * 2,
    color:  PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
    born:   performance.now(),
  }
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function createDetailPanel(): HTMLDivElement {
  const el = document.createElement('div')
  el.id = 'cell-detail-panel'
  Object.assign(el.style, {
    position:      'absolute',
    right:         '16px',
    top:           '16px',
    width:         '220px',
    background:    'rgba(13,17,23,0.92)',
    border:        '1px solid rgba(255,255,255,0.15)',
    color:         '#e6edf3',
    fontFamily:    'monospace',
    fontSize:      '12px',
    padding:       '12px',
    borderRadius:  '8px',
    backdropFilter:'blur(10px)',
    opacity:       '0',
    transform:     'translateY(-6px)',
    transition:    'opacity 0.2s ease, transform 0.2s ease',
    zIndex:        '9998',
    pointerEvents: 'none',
  })
  document.body.appendChild(el)
  return el
}

function showDetailPanel(el: HTMLDivElement, cell: CellParams): void {
  el.innerHTML = `
    <div style="font-size:10px;opacity:0.5;margin-bottom:4px;">${cell.species}</div>
    <div style="font-size:14px;font-weight:600;margin-bottom:8px;">${cell.label}</div>
    <div style="opacity:0.6;font-size:10px;line-height:1.6;">
      <div>cell_id: ${cell.cell_id}</div>
      <div>epoch: ${cell.epoch}</div>
      <div>z: ${cell.z}</div>
      <div>opacity: ${cell.opacity.toFixed(2)}</div>
      <div>size: ${cell.bbox.w}×${cell.bbox.h}</div>
    </div>
    <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.1);font-size:10px;opacity:0.45;">
      click to ask Mira →
    </div>
  `
  el.style.opacity   = '1'
  el.style.transform = 'translateY(0)'
}

function hideDetailPanel(el: HTMLDivElement): void {
  el.style.opacity   = '0'
  el.style.transform = 'translateY(-6px)'
}

// ── Main mount ────────────────────────────────────────────────────────────────

/**
 * mountCellInteraction
 *
 * Attaches mouse interaction to the CellRenderer canvas.
 * Returns a destroy() handle for cleanup.
 */
export function mountCellInteraction(opts: CellInteractionOptions): CellInteractionHandle {
  const { canvas, cells, transform: t, sendPrompt, onHover, onClick } = opts

  // ── Overlay canvas (for pulse rings + particles) ──────────────────────────
  const overlay = document.createElement('canvas')
  overlay.width  = canvas.width
  overlay.height = canvas.height
  Object.assign(overlay.style, {
    position:      'absolute',
    top:           canvas.offsetTop + 'px',
    left:          canvas.offsetLeft + 'px',
    width:         canvas.style.width  || canvas.width  + 'px',
    height:        canvas.style.height || canvas.height + 'px',
    pointerEvents: 'none',
    zIndex:        '10',
  })
  canvas.parentElement?.appendChild(overlay)
  const ctx = overlay.getContext('2d')!

  // ── DOM overlays ──────────────────────────────────────────────────────────
  const tooltip     = createTooltip()
  const detailPanel = createDetailPanel()

  // ── Neighbour nudge containers (map cell_id → original transform) ─────────
  // We store the PixiJS Container for each cell so we can nudge it.
  // Since we don't have direct access here, we record push state and apply
  // it via CSS `translate` on a per-cell hit region div.  The actual PixiJS
  // containers are nudged via a shared Map exported below.
  const nudgeState = new Map<string, { dx: number; dy: number }>()

  // ── State ─────────────────────────────────────────────────────────────────
  let hoveredCell: CellParams | null = null
  let particles: Particle[]          = []
  let animFrameId: number            = -1
  let lastMouseX = 0
  let lastMouseY = 0

  // ── Animation loop ────────────────────────────────────────────────────────
  function animate(now: number) {
    ctx.clearRect(0, 0, overlay.width, overlay.height)

    // Update + draw particles
    const alive: Particle[] = []
    for (const p of particles) {
      const age  = (now - p.born) / PARTICLE_LIFETIME_MS
      if (age >= 1) continue
      p.x   += p.vx
      p.y   += p.vy
      p.vy  += 0.015 // gravity drift
      p.life = 1 - age

      ctx.save()
      ctx.globalAlpha = p.life * 0.75
      ctx.fillStyle   = p.color
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.radius * p.life, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
      alive.push(p)
    }
    particles = alive

    // Draw hover bloom ring
    if (hoveredCell) {
      const cx = (hoveredCell.bbox.x + hoveredCell.bbox.w / 2) * t.scale + t.offX
      const cy = (hoveredCell.bbox.y + hoveredCell.bbox.h / 2) * t.scale + t.offY
      const r  = (Math.max(hoveredCell.bbox.w, hoveredCell.bbox.h) / 2) * t.scale + 6
      const pulse = 0.5 + 0.5 * Math.sin(now / 400)

      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.strokeStyle = hoveredCell.stroke_color
      ctx.globalAlpha = 0.25 + pulse * 0.25
      ctx.lineWidth   = 1.5 + pulse * 1.5
      ctx.shadowColor = hoveredCell.stroke_color
      ctx.shadowBlur  = 12
      ctx.stroke()
      ctx.restore()
    }

    animFrameId = requestAnimationFrame(animate)
  }
  animFrameId = requestAnimationFrame(animate)

  // ── Neighbour nudge (CSS transforms on a nudge-div layer) ─────────────────
  // We communicate nudge state via a globally accessible Map so CellRenderer
  // can read it on its own tick if integrated.
  ;(window as unknown as Record<string, unknown>)['__cellNudgeState'] = nudgeState

  function applyNudge(hovered: CellParams | null) {
    nudgeState.clear()
    if (!hovered) return
    const pushes = computeNeighbourPush(cells, hovered, t)
    for (const { cell, dx, dy } of pushes) {
      nudgeState.set(cell.cell_id, { dx, dy })
    }
  }

  // ── Mouse move ────────────────────────────────────────────────────────────
  function onMouseMove(e: MouseEvent) {
    const rect  = canvas.getBoundingClientRect()
    const cx    = (e.clientX - rect.left) * (canvas.width  / rect.width)
    const cy    = (e.clientY - rect.top)  * (canvas.height / rect.height)
    lastMouseX  = e.clientX
    lastMouseY  = e.clientY

    // Spawn particles
    if (Math.random() < 0.35) {
      particles.push(spawnParticle(cx, cy))
      if (particles.length > PARTICLE_COUNT) particles.splice(0, particles.length - PARTICLE_COUNT)
    }

    // Hit test
    const cellCoord = toCell(cx, cy, t)
    const hit       = hitTest(cells, cellCoord.x, cellCoord.y)

    if (hit?.cell_id !== hoveredCell?.cell_id) {
      hoveredCell = hit
      onHover?.(hit)
      applyNudge(hit)

      if (hit) {
        showTooltip(tooltip, hit.label, e.clientX, e.clientY)
        showDetailPanel(detailPanel, hit)
        canvas.style.cursor = 'pointer'
      } else {
        hideTooltip(tooltip)
        hideDetailPanel(detailPanel)
        canvas.style.cursor = 'default'
      }
    } else if (hit) {
      // Keep tooltip following mouse
      showTooltip(tooltip, hit.label, e.clientX, e.clientY)
    }
  }

  // ── Mouse click ───────────────────────────────────────────────────────────
  function onMouseClick(e: MouseEvent) {
    const rect      = canvas.getBoundingClientRect()
    const cx        = (e.clientX - rect.left) * (canvas.width  / rect.width)
    const cy        = (e.clientY - rect.top)  * (canvas.height / rect.height)
    const cellCoord = toCell(cx, cy, t)
    const hit       = hitTest(cells, cellCoord.x, cellCoord.y)
    if (!hit) return

    onClick?.(hit)

    // Send prompt to Mira
    sendPrompt?.(`Tell me about ${hit.label}`)

    // Pulse animation
    triggerPulse(ctx, hit, t, hit.stroke_color)
  }

  // ── Mouse leave ───────────────────────────────────────────────────────────
  function onMouseLeave() {
    hoveredCell = null
    onHover?.(null)
    applyNudge(null)
    hideTooltip(tooltip)
    hideDetailPanel(detailPanel)
    canvas.style.cursor = 'default'
  }

  canvas.addEventListener('mousemove',  onMouseMove)
  canvas.addEventListener('click',      onMouseClick)
  canvas.addEventListener('mouseleave', onMouseLeave)

  // ── Destroy ───────────────────────────────────────────────────────────────
  return {
    destroy() {
      canvas.removeEventListener('mousemove',  onMouseMove)
      canvas.removeEventListener('click',      onMouseClick)
      canvas.removeEventListener('mouseleave', onMouseLeave)
      cancelAnimationFrame(animFrameId)
      overlay.remove()
      tooltip.remove()
      detailPanel.remove()
      nudgeState.clear()
      delete (window as unknown as Record<string, unknown>)['__cellNudgeState']
    },
  }
}

// ── Re-export nudge state accessor ───────────────────────────────────────────

/**
 * getCellNudge — returns the push vector for a given cell_id (if any).
 * CellRenderer can call this each frame to offset container positions.
 */
export function getCellNudge(cellId: string): { dx: number; dy: number } | null {
  const map = (window as unknown as Record<string, unknown>)['__cellNudgeState'] as
    Map<string, { dx: number; dy: number }> | undefined
  return map?.get(cellId) ?? null
}
