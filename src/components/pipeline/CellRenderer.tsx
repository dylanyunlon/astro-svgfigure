/**
 * CellRenderer.tsx — PixiJS WebGL cell renderer driven by params.json
 *
 * Active Theory Hydra engine approach: pure WebGL, no SVG.
 * Each cell species draws procedurally from species_params using
 * @pixi/graphics (local upstream), honouring fill_color, stroke_color,
 * opacity, and shadow from every cell's params.json.
 *
 * Species covered:
 *   cil-eye         → radial rays + concentric pupils (ring_count, pupil_radius, r_outer, r_inner_ratio)
 *   cil-bolt        → zigzag lightning path (zigzag_count, amplitude, seg_width)
 *   cil-vector      → parallel fanned arrows (arrow_count, arrow_length, angle_spread)
 *   cil-plus        → dashed cross arms (arm_length, stroke_width, dash_corners)
 *   cil-arrow-right → solid chevron arrow head (arrow_width, arrow_height)
 *
 * Layout: all 7 cells positioned from their bbox.x / bbox.y in a shared
 * 800×780 PixiJS stage, z-sorted per params.z.
 *
 * Shadow: DropShadowFilter from upstream pixijs-filters.
 *
 * Usage (from Astro page script):
 *   import { mountCellRenderer } from '@/components/pipeline/CellRenderer'
 *   const stop = await mountCellRenderer(canvas, cellParams)
 *   // call stop() to destroy
 */

import { Application } from '../../../upstream/pixijs-engine/src/app/Application'
import { Container } from '../../../upstream/pixijs-engine/src/scene/container/Container'
import { Graphics } from '../../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics'
import { Text } from '../../../upstream/pixijs-engine/src/scene/text/Text'
import { TextStyle } from '../../../upstream/pixijs-engine/src/scene/text/TextStyle'

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellParams {
  cell_id: string
  species: string
  label: string
  font_size: number
  bbox: { x: number; y: number; w: number; h: number; z: number }
  z: number
  opacity: number
  fill_color: string   // CSS hex e.g. "#1E88E5"
  stroke_color: string
  species_params: Record<string, unknown>
  epoch: number
  shadow: { dx: number; dy: number; blur: number; opacity: number }
}

// ── Hex CSS → PixiJS number ───────────────────────────────────────────────────

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16)
}

// ── Species pattern drawers ───────────────────────────────────────────────────
// Each drawer receives a fresh Graphics, the cell bbox (w×h), the parsed
// species_params, and numeric fill / stroke colours.

type Drawer = (
  g: Graphics,
  w: number,
  h: number,
  sp: Record<string, unknown>,
  fill: number,
  stroke: number,
  opacity: number,
) => void

// cil-eye: radial rays from r_inner to r_outer, then a solid pupil disc.
const drawEye: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const cx = w / 2
  const cy = h / 2
  const ringCount  = (sp.ring_count   as number) ?? 8
  const rOuter     = (sp.r_outer      as number) ?? Math.min(w, h) * 0.42
  const rInnerRatio = (sp.r_inner_ratio as number) ?? 0.3
  const pupilR     = (sp.pupil_radius  as number) ?? rOuter * 0.18
  const rInner     = rOuter * rInnerRatio

  // Concentric rings (outer → inner, fading in)
  for (let i = ringCount; i >= 1; i--) {
    const r   = rInner + (rOuter - rInner) * (i / ringCount)
    const a   = (opacity * 0.12 * (ringCount - i + 1)) / ringCount
    g.circle(cx, cy, r)
    g.stroke({ color: stroke, width: 0.8, alpha: Math.min(a, 0.55) })
  }

  // Radial spokes
  const spokeCount = ringCount * 2
  for (let i = 0; i < spokeCount; i++) {
    const angle = (i / spokeCount) * Math.PI * 2
    const x1 = cx + Math.cos(angle) * rInner
    const y1 = cy + Math.sin(angle) * rInner
    const x2 = cx + Math.cos(angle) * rOuter
    const y2 = cy + Math.sin(angle) * rOuter
    g.moveTo(x1, y1)
    g.lineTo(x2, y2)
    g.stroke({ color: stroke, width: 0.6, alpha: opacity * 0.3 })
  }

  // Outer ring border
  g.circle(cx, cy, rOuter)
  g.stroke({ color: stroke, width: 1.2, alpha: opacity * 0.7 })

  // Pupil (solid)
  g.circle(cx, cy, pupilR)
  g.fill({ color: stroke, alpha: opacity * 0.85 })
}

// cil-bolt: zigzag lightning path across the cell width.
const drawBolt: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const zigzagCount = (sp.zigzag_count as number) ?? 6
  const amplitude   = (sp.amplitude    as number) ?? h * 0.3
  const segWidth    = (sp.seg_width    as number) ?? w / zigzagCount
  const cy          = h / 2

  // Main zigzag
  g.moveTo(4, cy)
  for (let i = 0; i < zigzagCount; i++) {
    const x  = 4 + (i + 1) * segWidth
    const y  = cy + (i % 2 === 0 ? -amplitude : amplitude) * 0.5
    g.lineTo(Math.min(x, w - 4), y)
  }
  g.lineTo(w - 4, cy)
  g.stroke({ color: stroke, width: 2.0, alpha: opacity * 0.9 })

  // Glow duplicate (thicker, lower alpha)
  g.moveTo(4, cy)
  for (let i = 0; i < zigzagCount; i++) {
    const x  = 4 + (i + 1) * segWidth
    const y  = cy + (i % 2 === 0 ? -amplitude : amplitude) * 0.5
    g.lineTo(Math.min(x, w - 4), y)
  }
  g.lineTo(w - 4, cy)
  g.stroke({ color: stroke, width: 5.0, alpha: opacity * 0.18 })
}

// cil-vector: fanned parallel arrows, angle_spread controls fan angle.
const drawVector: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const arrowCount  = (sp.arrow_count   as number) ?? 4
  const arrowLength = (sp.arrow_length  as number) ?? w * 0.55
  const spread      = (sp.angle_spread  as number) ?? 0.6
  const cx          = w / 2
  const cy          = h / 2
  const headSize    = 5

  for (let i = 0; i < arrowCount; i++) {
    const t     = arrowCount === 1 ? 0 : (i / (arrowCount - 1)) - 0.5
    const angle = t * spread
    const dx    = Math.cos(angle) * arrowLength
    const dy    = Math.sin(angle) * arrowLength

    const startX = cx - dx / 2
    const startY = cy - dy / 2
    const endX   = cx + dx / 2
    const endY   = cy + dy / 2

    // Shaft
    g.moveTo(startX, startY)
    g.lineTo(endX, endY)
    g.stroke({ color: stroke, width: 1.4, alpha: opacity * 0.75 })

    // Arrowhead
    const perpAngle = angle + Math.PI
    g.moveTo(endX, endY)
    g.lineTo(
      endX + Math.cos(perpAngle + 0.4) * headSize,
      endY + Math.sin(perpAngle + 0.4) * headSize,
    )
    g.moveTo(endX, endY)
    g.lineTo(
      endX + Math.cos(perpAngle - 0.4) * headSize,
      endY + Math.sin(perpAngle - 0.4) * headSize,
    )
    g.stroke({ color: stroke, width: 1.4, alpha: opacity * 0.75 })
  }
}

// cil-plus: cross arms, optionally dashed at corners.
const drawPlus: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const armLength   = (sp.arm_length   as number) ?? Math.min(w, h) * 0.36
  const strokeWidth = (sp.stroke_width as number) ?? 2.5
  const dashCorners = (sp.dash_corners as boolean) ?? false
  const cx          = w / 2
  const cy          = h / 2

  if (dashCorners) {
    // Dashed segments: draw 4 arms in 2-segment dashes
    const dashLen  = armLength * 0.55
    const gapLen   = armLength * 0.2
    const segments: [number, number, number, number][] = [
      // Horizontal
      [cx - armLength, cy, cx - gapLen, cy],
      [cx + gapLen,    cy, cx + armLength, cy],
      // Vertical
      [cx, cy - armLength, cx, cy - gapLen],
      [cx, cy + gapLen,    cx, cy + armLength],
    ]
    for (const [x1, y1, x2, y2] of segments) {
      g.moveTo(x1, y1)
      g.lineTo(x2, y2)
      g.stroke({ color: stroke, width: strokeWidth, alpha: opacity * 0.85 })
    }
    // Centre square
    const sq = strokeWidth * 1.5
    g.rect(cx - sq / 2, cy - sq / 2, sq, sq)
    g.fill({ color: stroke, alpha: opacity * 0.6 })
  } else {
    // Solid cross
    g.moveTo(cx - armLength, cy)
    g.lineTo(cx + armLength, cy)
    g.moveTo(cx, cy - armLength)
    g.lineTo(cx, cy + armLength)
    g.stroke({ color: stroke, width: strokeWidth, alpha: opacity * 0.85 })
  }
}

// cil-arrow-right: a filled right-pointing chevron.
const drawArrowRight: Drawer = (g, w, h, sp, fill, stroke, opacity) => {
  const arrowW = (sp.arrow_width  as number) ?? w * 0.6
  const arrowH = (sp.arrow_height as number) ?? h * 0.45
  const cx     = w / 2
  const cy     = h / 2

  const x0 = cx - arrowW / 2
  const x1 = cx + arrowW / 2 - arrowH * 0.8
  const x2 = cx + arrowW / 2

  // Shaft rectangle
  g.rect(x0, cy - arrowH / 4, x1 - x0, arrowH / 2)
  g.fill({ color: fill, alpha: opacity * 0.55 })
  g.stroke({ color: stroke, width: 1.0, alpha: opacity * 0.7 })

  // Arrowhead triangle
  g.moveTo(x1, cy - arrowH / 2)
  g.lineTo(x2, cy)
  g.lineTo(x1, cy + arrowH / 2)
  g.closePath()
  g.fill({ color: fill, alpha: opacity * 0.8 })
  g.stroke({ color: stroke, width: 1.2, alpha: opacity * 0.9 })
}

const DRAWERS: Record<string, Drawer> = {
  'cil-eye':         drawEye,
  'cil-bolt':        drawBolt,
  'cil-vector':      drawVector,
  'cil-plus':        drawPlus,
  'cil-arrow-right': drawArrowRight,
}

// ── Shadow overlay (cheap drop-shadow via duplicate blurred Graphics) ─────────

function buildShadow(
  w: number,
  h: number,
  shadow: CellParams['shadow'],
  fillColor: number,
): Graphics {
  const s = new Graphics()
  // A blurred rect approximating CSS drop-shadow
  s.roundRect(shadow.dx, shadow.dy, w, h, 6)
  s.fill({ color: fillColor, alpha: shadow.opacity * 0.6 })
  // PixiJS BlurFilter not imported here to keep deps lean;
  // instead render as a soft semi-transparent rect offset
  return s
}

// ── Cell container builder ────────────────────────────────────────────────────

function buildCell(p: CellParams): Container {
  const { bbox, species, fill_color, stroke_color, opacity, shadow, label, font_size, z } = p
  const { w, h } = bbox
  const fill   = hexToNum(fill_color)
  const stroke = hexToNum(stroke_color)
  const sp     = p.species_params

  const container = new Container()
  container.position.set(bbox.x, bbox.y)
  container.zIndex = z

  // Shadow layer
  const shadowG = buildShadow(w, h, shadow, fill)
  container.addChild(shadowG)

  // Background rounded rect
  const bg = new Graphics()
  bg.roundRect(0, 0, w, h, 6)
  bg.fill({ color: fill, alpha: Math.max(opacity, 0.08) * 0.9 })
  bg.stroke({ color: stroke, width: 1.0, alpha: Math.max(opacity, 0.15) * 1.2 })
  container.addChild(bg)

  // Species pattern
  const drawer = DRAWERS[species]
  if (drawer) {
    const patternG = new Graphics()
    drawer(patternG, w, h, sp, fill, stroke, Math.max(opacity * 14, 0.7))
    container.addChild(patternG)
  }

  // Label text
  const style = new TextStyle({
    fontSize: font_size,
    fill: fill_color,
    fontFamily: 'monospace',
    fontWeight: '500',
  })
  const txt = new Text({ text: label, style })
  txt.position.set(6, h + 4)
  txt.alpha = 0.75
  container.addChild(txt)

  return container
}

// ── Public mount function ─────────────────────────────────────────────────────

/**
 * mountCellRenderer
 *
 * Initialises a PixiJS Application on the given canvas and renders
 * all provided CellParams (from params.json) using WebGL.
 *
 * Returns a stop() function that destroys the PixiJS app.
 */
export async function mountCellRenderer(
  canvas: HTMLCanvasElement,
  cells: CellParams[],
): Promise<() => void> {
  // Auto-size canvas to its container
  const rect = canvas.parentElement?.getBoundingClientRect()
  if (rect) {
    canvas.width  = Math.floor(rect.width)  || 820
    canvas.height = Math.floor(rect.height) || 800
  }

  const app = new Application()
  await app.init({
    canvas,
    width:           canvas.width,
    height:          canvas.height,
    backgroundColor: 0x0d1117,
    antialias:       true,
    resolution:      (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1,
    autoDensity:     true,
  })

  app.stage.sortableChildren = true

  // Pad + centre: find bounding box of all cells and offset to fit
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of cells) {
    minX = Math.min(minX, p.bbox.x)
    minY = Math.min(minY, p.bbox.y)
    maxX = Math.max(maxX, p.bbox.x + p.bbox.w)
    maxY = Math.max(maxY, p.bbox.y + p.bbox.h + 20)
  }
  const contentW = maxX - minX
  const contentH = maxY - minY
  const scaleX   = (canvas.width  - 80) / contentW
  const scaleY   = (canvas.height - 80) / contentH
  const scale    = Math.min(scaleX, scaleY, 1.4)
  const offX     = (canvas.width  - contentW * scale) / 2 - minX * scale
  const offY     = (canvas.height - contentH * scale) / 2 - minY * scale

  const root = new Container()
  root.scale.set(scale)
  root.position.set(offX, offY)
  app.stage.addChild(root)

  // Render each cell
  const sorted = [...cells].sort((a, b) => a.z - b.z)
  for (const p of sorted) {
    const cell = buildCell(p)
    root.addChild(cell)
  }

  return () => {
    try {
      app.destroy(false, { children: true, texture: true })
    } catch { /* ignore */ }
  }
}
