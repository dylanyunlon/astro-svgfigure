/**
 * CompositeRenderer.tsx — Composite multi-pass WebGL render pipeline
 *
 * Active Theory FXScene-style approach:
 *   Pass 1  →  render all cells to an offscreen PIXI.RenderTexture
 *   Pass 2  →  apply AdvancedBloomFilter (Kawase blur = AT HydraBloom algorithm)
 *              to the full composited frame
 *   Pass 3  →  blit the bloom-composited sprite to the visible screen canvas
 *
 * AdvancedBloomFilter is sourced from upstream/pixijs-filters (same repo),
 * which uses a Kawase-blur multi-pass kernel — identical to the algorithm
 * AT's HydraBloom uses in production FX pipelines.
 *
 * Usage (from Astro page script):
 *   import { mountCompositeRenderer } from '@/components/pipeline/CompositeRenderer'
 *   const stop = await mountCompositeRenderer(canvas, cellParams, bloomOptions)
 *   // call stop() to destroy
 */

import { Application } from '../../../upstream/pixijs-engine/src/app/Application'
import { Container }   from '../../../upstream/pixijs-engine/src/scene/container/Container'
import { Graphics }    from '../../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics'
import { Sprite }      from '../../../upstream/pixijs-engine/src/scene/sprite/Sprite'
import { Text }        from '../../../upstream/pixijs-engine/src/scene/text/Text'
import { TextStyle }   from '../../../upstream/pixijs-engine/src/scene/text/TextStyle'
import { RenderTexture } from '../../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/RenderTexture'
import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom/AdvancedBloomFilter'

// Re-export CellParams so consumers only need one import point
export type { CellParams } from './CellRenderer'
import type { CellParams } from './CellRenderer'

// ── Bloom configuration ───────────────────────────────────────────────────────

export interface BloomOptions {
  /** Brightness threshold for bloom extraction (0–1). Default: 0.5 */
  threshold?: number
  /** Bloom intensity multiplier. Default: 1.4 */
  bloomScale?: number
  /** Output brightness. Default: 1.0 */
  brightness?: number
  /** Kawase kernel blur strength. Default: 4 */
  blur?: number
  /** Kawase kernel quality steps. Default: 4 */
  quality?: number
}

const DEFAULT_BLOOM: Required<BloomOptions> = {
  threshold:  0.5,
  bloomScale: 1.4,
  brightness: 1.0,
  blur:       4,
  quality:    4,
}

// ── Colour helper ─────────────────────────────────────────────────────────────

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16)
}

// ── Species drawers (same procedural logic as CellRenderer) ──────────────────

type Drawer = (
  g: Graphics,
  w: number,
  h: number,
  sp: Record<string, unknown>,
  fill: number,
  stroke: number,
  opacity: number,
) => void

const drawEye: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const cx = w / 2, cy = h / 2
  const ringCount   = (sp.ring_count    as number) ?? 8
  const rOuter      = (sp.r_outer       as number) ?? Math.min(w, h) * 0.42
  const rInnerRatio = (sp.r_inner_ratio as number) ?? 0.3
  const pupilR      = (sp.pupil_radius  as number) ?? rOuter * 0.18
  const rInner      = rOuter * rInnerRatio

  for (let i = ringCount; i >= 1; i--) {
    const r = rInner + (rOuter - rInner) * (i / ringCount)
    const a = (opacity * 0.12 * (ringCount - i + 1)) / ringCount
    g.circle(cx, cy, r)
    g.stroke({ color: stroke, width: 0.8, alpha: Math.min(a, 0.55) })
  }
  const spokeCount = ringCount * 2
  for (let i = 0; i < spokeCount; i++) {
    const angle = (i / spokeCount) * Math.PI * 2
    g.moveTo(cx + Math.cos(angle) * rInner, cy + Math.sin(angle) * rInner)
    g.lineTo(cx + Math.cos(angle) * rOuter, cy + Math.sin(angle) * rOuter)
    g.stroke({ color: stroke, width: 0.6, alpha: opacity * 0.3 })
  }
  g.circle(cx, cy, rOuter)
  g.stroke({ color: stroke, width: 1.2, alpha: opacity * 0.7 })
  g.circle(cx, cy, pupilR)
  g.fill({ color: stroke, alpha: opacity * 0.85 })
}

const drawBolt: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const zigzagCount = (sp.zigzag_count as number) ?? 6
  const amplitude   = (sp.amplitude    as number) ?? h * 0.3
  const segWidth    = (sp.seg_width    as number) ?? w / zigzagCount
  const cy          = h / 2
  const drawPath = () => {
    g.moveTo(4, cy)
    for (let i = 0; i < zigzagCount; i++) {
      const x = 4 + (i + 1) * segWidth
      const y = cy + (i % 2 === 0 ? -amplitude : amplitude) * 0.5
      g.lineTo(Math.min(x, w - 4), y)
    }
    g.lineTo(w - 4, cy)
  }
  drawPath(); g.stroke({ color: stroke, width: 2.0, alpha: opacity * 0.9 })
  drawPath(); g.stroke({ color: stroke, width: 5.0, alpha: opacity * 0.18 })
}

const drawVector: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const arrowCount  = (sp.arrow_count  as number) ?? 4
  const arrowLength = (sp.arrow_length as number) ?? w * 0.55
  const spread      = (sp.angle_spread as number) ?? 0.6
  const cx = w / 2, cy = h / 2
  const headSize = 5
  for (let i = 0; i < arrowCount; i++) {
    const t = arrowCount === 1 ? 0 : (i / (arrowCount - 1)) - 0.5
    const angle = t * spread
    const dx = Math.cos(angle) * arrowLength
    const dy = Math.sin(angle) * arrowLength
    const sx = cx - dx / 2, sy = cy - dy / 2
    const ex = cx + dx / 2, ey = cy + dy / 2
    g.moveTo(sx, sy); g.lineTo(ex, ey)
    g.stroke({ color: stroke, width: 1.4, alpha: opacity * 0.75 })
    const pa = angle + Math.PI
    g.moveTo(ex, ey); g.lineTo(ex + Math.cos(pa + 0.4) * headSize, ey + Math.sin(pa + 0.4) * headSize)
    g.moveTo(ex, ey); g.lineTo(ex + Math.cos(pa - 0.4) * headSize, ey + Math.sin(pa - 0.4) * headSize)
    g.stroke({ color: stroke, width: 1.4, alpha: opacity * 0.75 })
  }
}

const drawPlus: Drawer = (g, w, h, sp, _fill, stroke, opacity) => {
  const armLength   = (sp.arm_length   as number) ?? Math.min(w, h) * 0.36
  const strokeWidth = (sp.stroke_width as number) ?? 2.5
  const dashCorners = (sp.dash_corners as boolean) ?? false
  const cx = w / 2, cy = h / 2
  if (dashCorners) {
    const gapLen = armLength * 0.2
    const segs: [number, number, number, number][] = [
      [cx - armLength, cy, cx - gapLen, cy],
      [cx + gapLen, cy, cx + armLength, cy],
      [cx, cy - armLength, cx, cy - gapLen],
      [cx, cy + gapLen, cx, cy + armLength],
    ]
    for (const [x1, y1, x2, y2] of segs) {
      g.moveTo(x1, y1); g.lineTo(x2, y2)
      g.stroke({ color: stroke, width: strokeWidth, alpha: opacity * 0.85 })
    }
    const sq = strokeWidth * 1.5
    g.rect(cx - sq / 2, cy - sq / 2, sq, sq)
    g.fill({ color: stroke, alpha: opacity * 0.6 })
  } else {
    g.moveTo(cx - armLength, cy); g.lineTo(cx + armLength, cy)
    g.moveTo(cx, cy - armLength); g.lineTo(cx, cy + armLength)
    g.stroke({ color: stroke, width: strokeWidth, alpha: opacity * 0.85 })
  }
}

const drawArrowRight: Drawer = (g, w, h, sp, fill, stroke, opacity) => {
  const arrowW = (sp.arrow_width  as number) ?? w * 0.6
  const arrowH = (sp.arrow_height as number) ?? h * 0.45
  const cx = w / 2, cy = h / 2
  const x0 = cx - arrowW / 2
  const x1 = cx + arrowW / 2 - arrowH * 0.8
  const x2 = cx + arrowW / 2
  g.rect(x0, cy - arrowH / 4, x1 - x0, arrowH / 2)
  g.fill({ color: fill, alpha: opacity * 0.55 })
  g.stroke({ color: stroke, width: 1.0, alpha: opacity * 0.7 })
  g.moveTo(x1, cy - arrowH / 2); g.lineTo(x2, cy); g.lineTo(x1, cy + arrowH / 2); g.closePath()
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

// ── Cell scene builder ────────────────────────────────────────────────────────

function buildCellScene(cells: CellParams[]): Container {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of cells) {
    minX = Math.min(minX, p.bbox.x)
    minY = Math.min(minY, p.bbox.y)
    maxX = Math.max(maxX, p.bbox.x + p.bbox.w)
    maxY = Math.max(maxY, p.bbox.y + p.bbox.h + 20)
  }

  const root = new Container()
  const sorted = [...cells].sort((a, b) => a.z - b.z)

  for (const p of sorted) {
    const { bbox, species, fill_color, stroke_color, opacity, shadow, label, font_size, z } = p
    const { w, h } = bbox
    const fill   = hexToNum(fill_color)
    const stroke = hexToNum(stroke_color)

    const cell = new Container()
    cell.position.set(bbox.x - minX, bbox.y - minY)
    cell.zIndex = z

    // Shadow (soft offset rect)
    const shadowG = new Graphics()
    shadowG.roundRect(shadow.dx, shadow.dy, w, h, 6)
    shadowG.fill({ color: fill, alpha: shadow.opacity * 0.6 })
    cell.addChild(shadowG)

    // Background
    const bg = new Graphics()
    bg.roundRect(0, 0, w, h, 6)
    bg.fill({ color: fill, alpha: Math.max(opacity, 0.08) * 0.9 })
    bg.stroke({ color: stroke, width: 1.0, alpha: Math.max(opacity, 0.15) * 1.2 })
    cell.addChild(bg)

    // Species pattern
    const drawer = DRAWERS[species]
    if (drawer) {
      const patternG = new Graphics()
      drawer(patternG, w, h, p.species_params, fill, stroke, Math.max(opacity * 14, 0.7))
      cell.addChild(patternG)
    }

    // Label
    const style = new TextStyle({ fontSize: font_size, fill: fill_color, fontFamily: 'monospace', fontWeight: '500' })
    const txt = new Text({ text: label, style })
    txt.position.set(6, h + 4)
    txt.alpha = 0.75
    cell.addChild(txt)

    root.addChild(cell)
  }

  root.sortableChildren = true
  return root
}

// ── Public mount function ─────────────────────────────────────────────────────

/**
 * mountCompositeRenderer
 *
 * Three-pass FXScene pipeline:
 *   1. Render cells → offscreen RenderTexture
 *   2. AdvancedBloomFilter (Kawase blur) on the composited frame
 *   3. Blit bloom-composited Sprite → screen
 *
 * @param canvas      Target HTMLCanvasElement
 * @param cells       CellParams array from params.json (same schema as CellRenderer)
 * @param bloomOpts   Optional bloom tuning; falls back to DEFAULT_BLOOM
 * @returns           Async stop() function that destroys the PixiJS app
 */
export async function mountCompositeRenderer(
  canvas: HTMLCanvasElement,
  cells: CellParams[],
  bloomOpts: BloomOptions = {},
): Promise<() => void> {
  // ── Canvas sizing ───────────────────────────────────────────────────────────
  const rect = canvas.parentElement?.getBoundingClientRect()
  const W = Math.floor(rect?.width  ?? 820) || 820
  const H = Math.floor(rect?.height ?? 800) || 800
  canvas.width  = W
  canvas.height = H

  const dpr = (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1

  // ── PixiJS Application ──────────────────────────────────────────────────────
  const app = new Application()
  await app.init({
    canvas,
    width:           W,
    height:          H,
    backgroundColor: 0x0d1117,
    antialias:       true,
    resolution:      dpr,
    autoDensity:     true,
  })

  // ── Pass 1: Build offscreen scene & RenderTexture ───────────────────────────
  // Calculate content bounding box to size the RT exactly
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of cells) {
    minX = Math.min(minX, p.bbox.x)
    minY = Math.min(minY, p.bbox.y)
    maxX = Math.max(maxX, p.bbox.x + p.bbox.w)
    maxY = Math.max(maxY, p.bbox.y + p.bbox.h + 20)
  }
  const contentW = maxX - minX
  const contentH = maxY - minY

  // Scale to fit canvas with padding
  const scaleX = (W - 80) / contentW
  const scaleY = (H - 80) / contentH
  const scale  = Math.min(scaleX, scaleY, 1.4)

  // Offscreen RenderTexture at content resolution × scale
  const rtW = Math.ceil(contentW * scale)
  const rtH = Math.ceil(contentH * scale)

  const renderTexture = RenderTexture.create({ width: rtW, height: rtH })

  // Offscreen scene container (scaled to RT dimensions)
  const offscreenScene = buildCellScene(cells)
  offscreenScene.scale.set(scale)

  // Render cells → RenderTexture (pass 1)
  app.renderer.render({ container: offscreenScene, target: renderTexture })

  // ── Pass 2: AdvancedBloomFilter on composited Sprite ───────────────────────
  const bloom = new AdvancedBloomFilter({
    threshold:  bloomOpts.threshold  ?? DEFAULT_BLOOM.threshold,
    bloomScale: bloomOpts.bloomScale ?? DEFAULT_BLOOM.bloomScale,
    brightness: bloomOpts.brightness ?? DEFAULT_BLOOM.brightness,
    blur:       bloomOpts.blur       ?? DEFAULT_BLOOM.blur,
    quality:    bloomOpts.quality    ?? DEFAULT_BLOOM.quality,
  })

  // Sprite wraps the RenderTexture — AdvancedBloomFilter applied here
  const compositeSprite = new Sprite(renderTexture)
  compositeSprite.filters = [bloom]

  // ── Pass 3: Position composited sprite centred on screen ───────────────────
  const offX = (W - rtW) / 2
  const offY = (H - rtH) / 2
  compositeSprite.position.set(offX, offY)

  app.stage.addChild(compositeSprite)

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  return () => {
    try {
      bloom.destroy()
      renderTexture.destroy(true)
      offscreenScene.destroy({ children: true })
      app.destroy(false, { children: true, texture: true })
    } catch { /* ignore */ }
  }
}
