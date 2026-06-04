/**
 * to-svg.ts — ELK Layouted JSON → Skeleton SVG
 *
 * Architecture refactor modeled on NVIDIA CCCL f984c90:
 *   Extract fused dispatch into dedicated kernels.
 *
 * Before: renderNode() was a 130-line if-chain dispatching 5 paths
 * with rendering logic inlined.  renderSprite() duplicated the
 * classify→resolve→render chain.
 *
 * After:
 *   §1  PRNG primitives (_fnv1a, _sfc32, _seededRand)
 *   §2  Geometry kernels — each is a dedicated pure function:
 *       renderRoundedRectUnion()   — organic blob (cloud/petal edges)
 *       renderFeatureMapStack()    — 3D parallelogram slab stack
 *       renderMathOperator()       — SVG circle + vector glyph
 *       renderScatteredTexture()   — decorative fill for regular nodes
 *   §3  Node frame utilities — extracted common wrapping logic:
 *       nodeLabel()               — truncated text with font sizing
 *       nodeCaptionBelow()        — italic caption below a visual
 *   §4  Node renderers — each path is its own function (no branching):
 *       renderGroupNode()         — blob background + recurse children
 *       renderOperatorNode()      — circle glyph, no box
 *       renderSpriteNode()        — resolve visual + caption below
 *       renderLabelNode()         — naked text, no background
 *       renderLeafNode()          — white box + texture + label
 *   §5  resolveNodeVisual()  — the DoubleBuffer equivalent:
 *       AI image → feature-map stack → organic blob (3-level fallback)
 *   §6  renderNode()  — pure dispatch, zero rendering logic
 *   §7  renderEdge()  — unchanged
 *   §8  elkToSvg()    — top-level entry point
 *
 * GitHub: EmilStenstrom/elkjs-svg, kieler/elkjs
 */

// ═══════════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════════

interface AdvancedEdge {
  routing?: string; lineStyle?: string; strokeDasharray?: string
  strokeWidth?: number; strokeColor?: string
  sourceArrow?: string; targetArrow?: string; directionality?: string
  semanticType?: string
  edgeLabels?: { text: string; position?: number; fontSize?: number; backgroundColor?: string }[]
  curvature?: number; crossesGroupBoundary?: boolean
}

interface ElkNode {
  id: string; x?: number; y?: number; width?: number; height?: number
  labels?: { text: string }[]; children?: ElkNode[]
  renderMode?: 'text' | 'icon' | 'sprite'; isOperator?: boolean; familyId?: string
  spriteRef?: { format: 'png' | 'svg' | 'stack'; url?: string; svg?: string; bbox?: [number, number, number, number]; fit?: 'contain'; stackCount?: number }
}

interface ElkEdge {
  id: string; sources?: string[]; targets?: string[]
  sections?: { startPoint: { x: number; y: number }; endPoint: { x: number; y: number }; bendPoints?: { x: number; y: number }[] }[]
  advanced?: AdvancedEdge; labels?: { text: string }[]
}

interface ElkGraph {
  id: string; x?: number; y?: number; width?: number; height?: number
  children?: ElkNode[]; edges?: ElkEdge[]
}

// ═══════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════

const NODE_FILL = '#FFFFFF'
const STROKE_COLOR = '#4A4A4A'
const TEXT_COLOR = '#1A1A1A'
const DEFAULT_EDGE_COLOR = '#4A4A4A'
const ARROW_SIZE = 8
const PADDING = 20

const SEMANTIC_STYLES: Record<string, Partial<AdvancedEdge>> = {
  data_flow: { strokeColor: '#4A4A4A', strokeWidth: 1.2 },
  gradient_flow: { strokeColor: '#555555', strokeDasharray: '8,4', strokeWidth: 1.2 },
  skip_connection: { strokeColor: '#4A4A4A', strokeWidth: 1.5, curvature: 0.7 },
  optional_path: { strokeColor: '#777777', strokeDasharray: '4,4', strokeWidth: 1 },
  inference_only: { strokeColor: '#555555', strokeDasharray: '10,3,3,3', strokeWidth: 1.2 },
  fan_out: { strokeColor: '#4A4A4A', strokeWidth: 1.2 },
  fan_in: { strokeColor: '#4A4A4A', strokeWidth: 1.2 },
  feedback: { strokeColor: '#555555', strokeDasharray: '6,3', strokeWidth: 1.2 },
  attention: { strokeColor: '#4A4A4A', strokeDasharray: '2,4', strokeWidth: 1.5 },
  concatenation: { strokeColor: '#4A4A4A', strokeWidth: 1.5 },
  residual: { strokeColor: '#4A4A4A', strokeWidth: 1.5, curvature: 0.6 },
  cross_boundary: { strokeColor: '#4A4A4A', strokeWidth: 1.5 },
}

const SKELETON_FILLS = [
  { bg: '#EBF5EB', stroke: '#A5D6A7' },
  { bg: '#FFF3E0', stroke: '#FFCC80' },
  { bg: '#FFF8E1', stroke: '#FFE082' },
  { bg: '#E3F2FD', stroke: '#90CAF9' },
  { bg: '#F3E5F5', stroke: '#CE93D8' },
]

const FAMILY_PALETTES: string[][] = [
  ['#B3D4FC', '#9EC5F0', '#89B6E4', '#74A7D8'],
  ['#F8BBD0', '#F48FB1', '#F06292', '#EC407A'],
  ['#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A'],
  ['#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],
  ['#D1C4E9', '#B39DDB', '#9575CD', '#7E57C2'],
  ['#B2EBF2', '#80DEEA', '#4DD0E1', '#26C6DA'],
]

const BLOB_PALETTE = [
  '#E8D5F5', '#D5E8F5', '#F5E8D5', '#D5F5E0', '#F5D5E0', '#F5F0D5',
]

export interface ToSvgOptions { clean?: boolean }

// ═══════════════════════════════════════════════════════════════════════════
//  §1  PRNG primitives — FNV-1a + sfc32
// ═══════════════════════════════════════════════════════════════════════════
//
// Deterministic seeded PRNG for reproducible geometry.  Same node ID
// always produces the same blob shape, texture pattern, rotation angles.
// Ported from g-harel/blobs internal/rand.ts.

function _fnv1a(str: string): () => number {
  let h = 2166136261 >>> 0
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 16777619)
  }
  return () => {
    h += h << 13; h ^= h >>> 7; h += h << 3; h ^= h >>> 17
    return (h += h << 5) >>> 0
  }
}

function _sfc32(a: number, b: number, c: number, d: number): () => number {
  return () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0
    const t = (a + b) | 0
    a = b ^ (b >>> 9)
    b = (c + (c << 3)) | 0
    c = (c << 21) | (c >>> 11)
    d = (d + 1) | 0
    const r = (t + d) | 0
    c = (c + r) | 0
    return (r >>> 0) / 4294967296
  }
}

function _seededRand(seed: string): () => number {
  const sg = _fnv1a(seed)
  return _sfc32(sg(), sg(), sg(), sg())
}

function _familyPalette(familyId: string): string[] {
  if (!familyId) return FAMILY_PALETTES[0]
  let h = 0
  for (let i = 0; i < familyId.length; i++) h = Math.imul(h ^ familyId.charCodeAt(i), 16777619)
  return FAMILY_PALETTES[((h >>> 0) % FAMILY_PALETTES.length)]
}

// ═══════════════════════════════════════════════════════════════════════════
//  §2  Geometry kernels — dedicated pure functions, zero dispatch logic
// ═══════════════════════════════════════════════════════════════════════════
//
// Each kernel takes (seed/node, x, y, w, h) and returns SVG markup.
// No classification, no fallback chains, no side effects.
// Analogy: CCCL's invoke_histogram_only() vs filter_and_histogram() —
// each is a dedicated kernel that does exactly one thing.

/**
 * Rounded-rectangle union blob — "你以为这是手绘的？实际上这只是一群圆角矩形"
 *
 * 8-12 opaque same-color rounded rectangles, each slightly rotated
 * (±3°–12°) and offset from center.  Visual union creates organic
 * silhouette with concave dips and convex bumps.
 */
function renderRoundedRectUnion(
  seed: string, x: number, y: number, w: number, h: number,
  color: string, count?: number,
): string {
  const rand = _seededRand(seed)
  const rectCount = count || Math.max(8, Math.min(14, 8 + Math.floor(rand() * 5)))
  const cx = x + w / 2, cy = y + h / 2
  let svg = ''

  for (let i = 0; i < rectCount; i++) {
    const rw = w * (0.35 + rand() * 0.45)
    const rh = h * (0.30 + rand() * 0.40)
    const rrx = Math.min(rw, rh) * (0.25 + rand() * 0.20)
    const ox = (rand() - 0.5) * w * 0.35
    const oy = (rand() - 0.5) * h * 0.35
    const rx = cx - rw / 2 + ox, ry = cy - rh / 2 + oy
    const rot = (rand() - 0.5) * 2 * (3 + rand() * 9)
    const rcx = rx + rw / 2, rcy = ry + rh / 2
    svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" rx="${rrx.toFixed(1)}" fill="${color}" opacity="1" transform="rotate(${rot.toFixed(1)} ${rcx.toFixed(1)} ${rcy.toFixed(1)})" />`
  }
  return svg
}

/**
 * Group blob — same technique, tuned for large containers.
 * Fewer rects (6-8), larger relative size, subtler rotation.
 */
function renderGroupBlob(
  seed: string, x: number, y: number, w: number, h: number, color: string,
): string {
  const rand = _seededRand(seed)
  const rectCount = 6 + Math.floor(rand() * 3)
  const cx = x + w / 2, cy = y + h / 2
  let svg = ''

  for (let i = 0; i < rectCount; i++) {
    const rw = w * (0.50 + rand() * 0.40)
    const rh = h * (0.45 + rand() * 0.40)
    const rrx = Math.min(rw, rh) * (0.20 + rand() * 0.15)
    const ox = (rand() - 0.5) * w * 0.25
    const oy = (rand() - 0.5) * h * 0.25
    const rx = cx - rw / 2 + ox, ry = cy - rh / 2 + oy
    const rot = (rand() - 0.5) * 2 * (2 + rand() * 4)
    const rcx = rx + rw / 2, rcy = ry + rh / 2
    svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" rx="${rrx.toFixed(1)}" fill="${color}" opacity="1" transform="rotate(${rot.toFixed(1)} ${rcx.toFixed(1)} ${rcy.toFixed(1)})" />`
  }
  return svg
}

/**
 * Feature-map stack — 3D parallelogram slab (FreqSelect reference).
 * Front face with procedural texture, N back faces offset diagonally.
 */
function renderFeatureMapStack(
  node: ElkNode, x: number, y: number, w: number, h: number,
): string {
  const rand = _seededRand(node.id)
  const familyId = (node as any).familyId || ''
  const palette = _familyPalette(familyId)
  const color = (node as any).fillColor || palette[0]
  const ref = (node as any).spriteRef
  const stackCount = Math.max(1, (ref && ref.stackCount) || 3)

  const dx = Math.min(4, w * 0.05), dy = Math.min(4, h * 0.05)
  const totalDx = dx * (stackCount - 1), totalDy = dy * (stackCount - 1)
  const fw = w - totalDx, fh = h - totalDy
  const fx = x, fy = y + totalDy
  let svg = ''

  // Back layers (back-to-front)
  for (let i = stackCount - 1; i >= 1; i--) {
    const lx = fx + dx * i, ly = fy - dy * i
    const backOpacity = 0.25 + (i / stackCount) * 0.30
    svg += `<rect x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" width="${fw.toFixed(1)}" height="${fh.toFixed(1)}" fill="${color}" opacity="${backOpacity.toFixed(2)}" stroke="#666" stroke-width="0.4" />`
  }

  // Front face
  svg += `<rect x="${fx.toFixed(1)}" y="${fy.toFixed(1)}" width="${fw.toFixed(1)}" height="${fh.toFixed(1)}" fill="${color}" opacity="0.55" stroke="#555" stroke-width="0.6" />`

  // Procedural texture — seeded noise for micro-diff within families
  let memberHash = 0
  for (let i = 0; i < familyId.length; i++) memberHash = Math.imul(memberHash ^ familyId.charCodeAt(i), 0x5bd1e995)
  const memberIndex = (memberHash >>> 0) % 6
  const textureCount = 6 + Math.floor(rand() * 4)
  for (let t = 0; t < textureCount; t++) {
    const tw = fw * (0.12 + rand() * 0.35), th = fh * (0.10 + rand() * 0.30)
    const tx = fx + rand() * Math.max(0, fw - tw), ty = fy + rand() * Math.max(0, fh - th)
    const texOpacity = 0.06 + rand() * 0.16
    const texColor = (memberIndex + t) % 2 === 0 ? '#ffffff' : '#000000'
    svg += `<rect x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" width="${tw.toFixed(1)}" height="${th.toFixed(1)}" rx="1" fill="${texColor}" opacity="${texOpacity.toFixed(2)}" />`
  }

  // 3D perspective edges
  for (let i = 0; i < stackCount - 1; i++) {
    const x1 = fx + dx * i + fw, y1 = fy - dy * i
    const x2 = fx + dx * (i + 1) + fw, y2 = fy - dy * (i + 1)
    svg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#888" stroke-width="0.3" />`
  }
  return svg
}

/**
 * Kernel grid — NxN colored cell grid (AdaKern reference).
 * Renders weighted cells where opacity/shade encodes weight magnitude.
 * Deterministic: same node ID always produces the same grid pattern.
 */
function renderKernelGrid(
  node: ElkNode, x: number, y: number, w: number, h: number,
): string {
  const rand = _seededRand(node.id)
  const familyId = (node as any).familyId || ''
  const palette = _familyPalette(familyId)
  const color = (node as any).fillColor || palette[0]
  const gridSize = 3
  const pad = 4
  const cellW = Math.floor((w - pad * 2 - (gridSize - 1)) / gridSize)
  const cellH = Math.floor((h - pad * 2 - 18 - (gridSize - 1)) / gridSize)
  let svg = ''

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const cx = x + pad + col * (cellW + 1)
      const cy = y + pad + row * (cellH + 1)
      const weight = 0.25 + rand() * 0.75
      svg += `<rect x="${cx}" y="${cy}" width="${cellW}" height="${cellH}" rx="1" fill="${color}" opacity="${weight.toFixed(2)}" stroke="${STROKE_COLOR}" stroke-width="0.3" />`
    }
  }
  return svg
}

/**
 * Math operator — SVG circle + inscribed vector glyph (⊗ ⊕ ⊙ ⊛ ⊖).
 * Pure vector, no AI, no text rendering, crisp at any zoom.
 */
function renderMathOperator(symbol: string, cx: number, cy: number, r: number): string {
  r = Math.max(6, r)
  const s = (symbol || '').trim()
  const circle = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${NODE_FILL}" stroke="${STROKE_COLOR}" stroke-width="1.6" />`
  const sw = 1.6
  const g = r * 0.62

  if (s === '⊗' || /^(x|×|⨂)$/i.test(s)) {
    const d = g * Math.SQRT1_2
    return circle
      + `<line x1="${cx-d}" y1="${cy-d}" x2="${cx+d}" y2="${cy+d}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
      + `<line x1="${cx-d}" y1="${cy+d}" x2="${cx+d}" y2="${cy-d}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
  }
  if (s === '⊕' || /^(\+|⨁)$/.test(s)) {
    return circle
      + `<line x1="${cx-g}" y1="${cy}" x2="${cx+g}" y2="${cy}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
      + `<line x1="${cx}" y1="${cy-g}" x2="${cx}" y2="${cy+g}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
  }
  if (s === '⊖' || s === '−' || s === '–') {
    return circle
      + `<line x1="${cx-g}" y1="${cy}" x2="${cx+g}" y2="${cy}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
  }
  if (s === '⊛' || s === '✱' || /^(\*|⊛|conv)$/i.test(s)) {
    const d2 = g * 0.7
    return circle
      + `<line x1="${cx-g}" y1="${cy}" x2="${cx+g}" y2="${cy}" stroke="${STROKE_COLOR}" stroke-width="${sw*0.8}" stroke-linecap="round" />`
      + `<line x1="${cx}" y1="${cy-g}" x2="${cx}" y2="${cy+g}" stroke="${STROKE_COLOR}" stroke-width="${sw*0.8}" stroke-linecap="round" />`
      + `<line x1="${cx-d2}" y1="${cy-d2}" x2="${cx+d2}" y2="${cy+d2}" stroke="${STROKE_COLOR}" stroke-width="${sw*0.6}" stroke-linecap="round" />`
      + `<line x1="${cx-d2}" y1="${cy+d2}" x2="${cx+d2}" y2="${cy-d2}" stroke="${STROKE_COLOR}" stroke-width="${sw*0.6}" stroke-linecap="round" />`
  }
  if (s === '⊙' || s === '·' || s === '∘') {
    return circle + `<circle cx="${cx}" cy="${cy}" r="${Math.max(1.5, r*0.18)}" fill="${STROKE_COLOR}" />`
  }
  if (s === '○' || s === '◯') return circle
  return circle
    + `<text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${Math.max(9, r*0.9)}" fill="${TEXT_COLOR}">${escapeXml(s)}</text>`
}

/**
 * Scattered decorative texture — small rotated rounded rects inside a node.
 * The "skeleton sketch" look for regular leaf nodes.
 */
function renderScatteredTexture(
  seed: string, x: number, y: number, w: number, h: number,
): string {
  const rand = _seededRand(seed)
  const numScatter = 3 + Math.floor(rand() * 4)
  const pad = 4
  let svg = ''
  for (let si = 0; si < numScatter; si++) {
    const sw2 = w * (0.15 + rand() * 0.35)
    const sh2 = h * (0.12 + rand() * 0.25)
    const sx = x + pad + rand() * Math.max(0, w - sw2 - pad * 2)
    const sy = y + pad + rand() * Math.max(0, h - sh2 - pad * 2)
    const srx = 3 + rand() * 8
    const sOpacity = 0.04 + rand() * 0.08
    const sRotate = (rand() - 0.5) * 6
    svg += `<rect x="${sx.toFixed(1)}" y="${sy.toFixed(1)}" width="${sw2.toFixed(1)}" height="${sh2.toFixed(1)}" rx="${srx.toFixed(1)}" fill="${STROKE_COLOR}" opacity="${sOpacity.toFixed(3)}" transform="rotate(${sRotate.toFixed(1)} ${(sx+sw2/2).toFixed(1)} ${(sy+sh2/2).toFixed(1)})" />`
  }
  return svg
}

// ═══════════════════════════════════════════════════════════════════════════
//  §3  Node frame utilities — extracted common wrapping logic
// ═══════════════════════════════════════════════════════════════════════════
//
// CCCL f984c90 extracted the last-block coordination (threadfence →
// atomicInc → prefix_sum → choose_bucket → reset) into finalize_pass()
// with a caller-supplied counter_update_fn lambda.
//
// We do the same: the repeated <g> opening, label truncation, and
// caption rendering are extracted here.  Each node renderer calls
// these instead of re-implementing string formatting.

function truncateLabel(label: string, width: number): string {
  const maxChars = Math.max(6, Math.floor(width / 8))
  return label.length > maxChars ? label.slice(0, maxChars - 2) + '\u2026' : label
}

function nodeCaptionBelow(label: string, x: number, y: number, w: number, h: number): string {
  const dl = truncateLabel(label, w)
  const labelY = y + h - 4
  const fontSize = Math.max(7, Math.min(10, w / dl.length * 1.2))
  return `<text x="${x + w / 2}" y="${labelY}" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize.toFixed(1)}" fill="${TEXT_COLOR}" font-weight="500" font-style="italic">${escapeXml(dl)}</text>`
}

function openNodeGroup(node: ElkNode, x: number, y: number, w: number, h: number, isGroup: boolean, depth: number): string {
  const nodeType = isGroup ? 'group' : 'leaf'
  return `  <g data-node-id="${escapeXml(node.id)}" data-node-type="${nodeType}" data-depth="${depth}" data-bbox="${x},${y},${w},${h}" data-render-mode="${node.renderMode || 'text'}">`
}

// ═══════════════════════════════════════════════════════════════════════════
//  §4  Node renderers — each path extracted into its own function
// ═══════════════════════════════════════════════════════════════════════════
//
// Before: renderNode() had 5 if-return branches with rendering logic
// inlined, like CCCL's old filter_and_histogram<IsFirstPass> that
// fused histogram-only and filter+histogram into one template function.
//
// After: each path is a dedicated function.  renderNode() (§6) is
// pure dispatch — it classifies the node and calls the right renderer.

function renderGroupNode(
  node: ElkNode, index: number, depth: number,
  x: number, y: number, w: number, h: number,
  label: string, cleanMode: boolean,
): string {
  const dl = truncateLabel(label, w)
  let svg = openNodeGroup(node, x, y, w, h, true, depth)

  if (depth <= 1) {
    const palette = SKELETON_FILLS[index % SKELETON_FILLS.length]
    svg += renderGroupBlob(node.id + '_group', x, y, w, h, palette.bg)
    svg += `<text x="${x + 12}" y="${y + 18}" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${palette.stroke}" font-weight="700" font-style="italic">${escapeXml(dl)}</text>`
  } else {
    svg += `<text x="${x + 8}" y="${y + 14}" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888" font-weight="600" font-style="italic">${escapeXml(dl)}</text>`
  }

  if (node.children) {
    node.children.forEach((c, i) => {
      svg += renderNode(c, i, depth + 1, x - PADDING, y - PADDING, cleanMode)
    })
  }
  svg += `</g>`
  return svg
}

function renderOperatorNode(
  node: ElkNode, x: number, y: number, w: number, h: number, label: string, depth: number,
): string {
  let svg = openNodeGroup(node, x, y, w, h, false, depth)
  svg += renderMathOperator(truncateLabel(label, w), x + w / 2, y + h / 2, Math.min(w, h) * 0.4)
  svg += `</g>`
  return svg
}

function renderSpriteNode(
  node: ElkNode, x: number, y: number, w: number, h: number, label: string, depth: number,
): string {
  let svg = openNodeGroup(node, x, y, w, h, false, depth)
  // Sprite nodes need adequate visual area.  ELK may allocate only 50px
  // height (text-box size), but a feature-map illustration needs at least
  // square aspect ratio.  Expand the visual area while keeping the label
  // below.  The node's layout position stays the same; the visual just
  // extends downward (academic figures routinely have tall feature-map blocks).
  const minVisualH = Math.max(w * 0.75, 80)  // at least 75% of width or 80px
  const visualH = Math.max(h - 18, minVisualH)
  svg += resolveNodeVisual(node, x, y, w, visualH)
  const labelY = y + visualH + 14
  const dl = truncateLabel(label, w)
  const fontSize = Math.max(7, Math.min(10, w / dl.length * 1.2))
  svg += `<text x="${x + w / 2}" y="${labelY}" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize.toFixed(1)}" fill="${TEXT_COLOR}" font-weight="500" font-style="italic">${escapeXml(dl)}</text>`
  svg += `</g>`
  return svg
}

function renderLabelNode(
  node: ElkNode, x: number, y: number, w: number, h: number, label: string, depth: number,
): string {
  const dl = truncateLabel(label, w)
  let svg = openNodeGroup(node, x, y, w, h, false, depth)
  const fontSize = h > 30 ? 13 : 11
  svg += `<text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="600">${escapeXml(dl)}</text>`
  svg += `</g>`
  return svg
}

function renderLeafNode(
  node: ElkNode, x: number, y: number, w: number, h: number,
  label: string, depth: number, cleanMode: boolean,
): string {
  const dl = truncateLabel(label, w)
  let svg = openNodeGroup(node, x, y, w, h, false, depth)
  const mainRx = Math.min(12, Math.min(w, h) * 0.15)
  svg += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${NODE_FILL}" stroke="${STROKE_COLOR}" stroke-width="0.8" rx="${mainRx}" />`

  if (!cleanMode) {
    svg += renderScatteredTexture(node.id, x, y, w, h)
  }

  const hasIcon = !!(node as any).iconHint
  const fontSize = (hasIcon && h >= 50) ? 9 : 12
  svg += `<text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="500">${escapeXml(dl)}</text>`
  svg += `</g>`
  return svg
}

// ═══════════════════════════════════════════════════════════════════════════
//  §5  resolveNodeVisual() — the DoubleBuffer equivalent
// ═══════════════════════════════════════════════════════════════════════════
//
// CCCL f984c90 uses DoubleBuffer<key_in_t> with selector ^= 1 to swap
// between input/output buffers across passes.  Our equivalent resolves
// the visual for a sprite node across 3 levels:
//
//   Level 0: AI-generated image available (url or inline svg) → <image>
//   Level 1: format 'stack' → 3D feature-map parallelogram
//   Level 2: no payload → organic blob placeholder
//
// Like DoubleBuffer::Current(), the caller gets the best available
// visual without knowing which level resolved.

function resolveNodeVisual(
  node: ElkNode, x: number, y: number, w: number, h: number,
): string {
  const ref = node.spriteRef

  // Level 0: AI-generated sprite — <image> or inline <g>
  // Use the FULL node area, preserving aspect ratio via xMidYMid meet.
  // The image was generated at 256x256; let the browser scale it properly.
  if (ref && ref.url) {
    return `<image href="${ref.url}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" preserveAspectRatio="xMidYMid meet" data-sprite="png" />`
  }
  if (ref && ref.format === 'svg' && ref.svg) {
    const natW = ref.bbox?.[2] || w, natH = ref.bbox?.[3] || h
    const sx = w / natW, sy = h / natH
    return `<g transform="translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${sx.toFixed(4)} ${sy.toFixed(4)})" data-sprite="svg">${ref.svg}</g>`
  }

  // Level 1: 3D feature-map stack
  if (ref && ref.format === 'stack') {
    return renderFeatureMapStack(node, x, y, w, h)
  }

  // Level 2: organic blob placeholder
  const familyId = (node as any).familyId || ''
  const palette = _familyPalette(familyId)
  const color = (node as any).fillColor || palette[0]
  return renderRoundedRectUnion(node.id, x, y, w, h, color)
}

// ═══════════════════════════════════════════════════════════════════════════
//  §6  renderNode() — pure dispatch, zero rendering logic
// ═══════════════════════════════════════════════════════════════════════════
//
// After the extraction, renderNode() is what dispatch() became in CCCL
// f984c90: pure orchestration.  It classifies the node into exactly one
// of the 5 renderers and calls it.  No rendering logic here at all.
//
//   Group     → renderGroupNode()      (blob bg + recurse)
//   Operator  → renderOperatorNode()   (circle glyph)
//   Sprite    → renderSpriteNode()     (resolveVisual + caption)
//   LabelOnly → renderLabelNode()      (naked text)
//   Leaf      → renderLeafNode()       (white box + texture)

function renderNode(node: ElkNode, index: number, depth: number = 0, offsetX: number = 0, offsetY: number = 0, cleanMode: boolean = false): string {
  const x = (node.x || 0) + PADDING + offsetX
  const y = (node.y || 0) + PADDING + offsetY
  const w = node.width || 160
  const h = node.height || 60
  const label = node.labels?.[0]?.text || node.id
  const isGroup = Array.isArray(node.children) && node.children.length > 0

  if (isGroup) return renderGroupNode(node, index, depth, x, y, w, h, label, cleanMode)
  if ((node as any).isOperator) return renderOperatorNode(node, x, y, w, h, label, depth)
  if (node.renderMode === 'sprite') return renderSpriteNode(node, x, y, w, h, label, depth)
  if ((node as any).renderMode === 'kernel') {
    let svg = openNodeGroup(node, x, y, w, h, false, depth)
    svg += renderKernelGrid(node, x, y, w, h)
    svg += nodeCaptionBelow(label, x, y, w, h)
    svg += `</g>`
    return svg
  }
  const isLabelOnly = !!(node as any).labelOnly || (h <= 30 && !(node as any).iconHint)
  if (isLabelOnly) return renderLabelNode(node, x, y, w, h, label, depth)

  // ═══ FIX: nodes with iconHint → sprite path (no white box) ═══
  // Even BEFORE classify_nodes() runs, nodes with iconHint should render
  // as sprites (blob placeholder + caption below), NOT as white boxes with
  // centered text. This ensures the skeleton SVG that Gemini receives
  // has NO white rounded-rect boxes — which Gemini would copy into output.
  if ((node as any).iconHint) return renderSpriteNode(node, x, y, w, h, label, depth)

  return renderLeafNode(node, x, y, w, h, label, depth, cleanMode)
}

// ═══════════════════════════════════════════════════════════════════════════
//  §7  renderEdge() — unchanged from original
// ═══════════════════════════════════════════════════════════════════════════

function renderEdge(edge: ElkEdge, offsetX: number = 0, offsetY: number = 0): string {
  if (!edge?.sections?.length) return ''

  const adv = edge.advanced || {}
  const sem = adv.semanticType ? SEMANTIC_STYLES[adv.semanticType] || {} : {}
  const color = adv.strokeColor || sem.strokeColor || DEFAULT_EDGE_COLOR
  const sw = adv.strokeWidth || sem.strokeWidth || 1.5
  const da = adv.strokeDasharray || sem.strokeDasharray || ''
  const curv = adv.curvature || sem.curvature || 0
  const isBidir = adv.directionality === 'bidirectional'
  const mid = `ah-${color.replace('#', '')}`
  const hasSrcArrow = isBidir || adv.sourceArrow === 'arrow'
  const noTgtArrow = adv.targetArrow === 'none'

  let svg = ''

  for (const section of edge.sections) {
    if (!section?.startPoint || !section?.endPoint) continue

    const pts: {x:number;y:number}[] = []
    pts.push({ x: (section.startPoint.x??0)+PADDING+offsetX, y: (section.startPoint.y??0)+PADDING+offsetY })
    if (Array.isArray(section.bendPoints)) {
      for (const bp of section.bendPoints) {
        if (bp && typeof bp.x==='number' && typeof bp.y==='number')
          pts.push({ x: bp.x+PADDING+offsetX, y: bp.y+PADDING+offsetY })
      }
    }
    pts.push({ x: (section.endPoint.x??0)+PADDING+offsetX, y: (section.endPoint.y??0)+PADDING+offsetY })

    let d: string
    if (curv > 0 && pts.length === 2) {
      const mx = (pts[0].x+pts[1].x)/2, my = (pts[0].y+pts[1].y)/2
      const pdx = pts[1].x-pts[0].x, pdy = pts[1].y-pts[0].y
      const pcx = mx - pdy*curv*0.5, pcy = my + pdx*curv*0.5
      d = `M${pts[0].x},${pts[0].y} Q${pcx},${pcy} ${pts[1].x},${pts[1].y}`
    } else {
      d = pts.map((p,i) => (i===0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ')
    }

    let attrs = `d="${d}" fill="none" stroke="${color}" stroke-width="${sw}"`
    if (da) attrs += ` stroke-dasharray="${da}"`
    if (!noTgtArrow) attrs += ` marker-end="url(#${mid})"`
    if (hasSrcArrow) attrs += ` marker-start="url(#${mid}-rev)"`
    svg += `  <path ${attrs} />\n`
  }

  // Edge labels
  const labels = adv.edgeLabels || []
  if (edge.labels?.length && !labels.length) {
    labels.push(...edge.labels.map(l => ({ text: l.text, position: 0.5, fontSize: 10 })))
  }

  for (const lbl of labels) {
    if (!lbl.text || !edge.sections?.[0]) continue
    const sec = edge.sections[0]
    const allPts = [sec.startPoint, ...(sec.bendPoints||[]), sec.endPoint]
    const pos = lbl.position ?? 0.5
    const idx = Math.min(Math.floor(pos*(allPts.length-1)), allPts.length-2)
    const t = (pos*(allPts.length-1)) - idx
    const px = (allPts[idx].x*(1-t) + allPts[idx+1].x*t) + PADDING + offsetX
    const py = (allPts[idx].y*(1-t) + allPts[idx+1].y*t) + PADDING + offsetY
    const fs = lbl.fontSize || 10
    const bg = lbl.backgroundColor || '#FFFFFF'

    svg += `  <rect x="${px-lbl.text.length*fs*0.3}" y="${py-fs*0.7}" width="${lbl.text.length*fs*0.6}" height="${fs*1.4}" fill="${bg}" rx="2" opacity="0.9" />\n`
    svg += `  <text x="${px}" y="${py+fs*0.15}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="${fs}" fill="${color}">${escapeXml(lbl.text)}</text>\n`
  }

  return svg
}

// ═══════════════════════════════════════════════════════════════════════════
//  §8  elkToSvg() — top-level entry point
// ═══════════════════════════════════════════════════════════════════════════

export function elkToSvg(graph: ElkGraph, opts?: ToSvgOptions): string {
  const cleanMode = !!(opts && opts.clean)
  if (!graph) {
    console.warn('[elkToSvg] graph is null/undefined')
    return _fallbackSvg('No graph data')
  }

  const hasChildren = Array.isArray(graph.children) && graph.children.length > 0
  const hasEdges = Array.isArray(graph.edges) && graph.edges.length > 0
  if (!hasChildren && !hasEdges) {
    console.warn('[elkToSvg] graph has no children and no edges')
    return _fallbackSvg('Empty graph (no nodes)')
  }

  let gw = graph.width || 0, gh = graph.height || 0
  if (gw === 0 || gh === 0) {
    function computeBounds(nodes: ElkNode[], ox: number, oy: number) {
      for (const child of nodes) {
        const cx = (child.x || 0) + ox, cy = (child.y || 0) + oy
        const right = cx + (child.width || 160), bottom = cy + (child.height || 60)
        if (right > gw) gw = right
        if (bottom > gh) gh = bottom
        if (Array.isArray(child.children) && child.children.length > 0) {
          computeBounds(child.children, cx, cy)
        }
      }
    }
    computeBounds(graph.children || [], 0, 0)
    gw = Math.max(gw, 200); gh = Math.max(gh, 150)
  }

  const width = gw + PADDING * 2, height = gh + PADDING * 2
  const parts: string[] = []

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="max-width:100%;height:auto;">`)
  parts.push(`  <defs>`)
  parts.push(`    <marker id="ah-default" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE/3}" orient="auto" markerUnits="strokeWidth">`)
  parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE/3}, 0 ${ARROW_SIZE/1.5}" fill="${DEFAULT_EDGE_COLOR}" />`)
  parts.push(`    </marker>`)

  // Generate colored markers for each edge color
  const markerColors = new Set<string>()
  function collectEdgeColors(edgeList: ElkEdge[] | undefined) {
    if (!Array.isArray(edgeList)) return
    for (const e of edgeList) {
      if (e?.advanced?.strokeColor) markerColors.add(e.advanced.strokeColor)
      if (e?.advanced?.semanticType && SEMANTIC_STYLES[e.advanced.semanticType]?.strokeColor)
        markerColors.add(SEMANTIC_STYLES[e.advanced.semanticType].strokeColor!)
    }
  }
  function collectAllEdgeColors(nodes: ElkNode[]) {
    for (const node of nodes) {
      collectEdgeColors((node as any).edges)
      if (node.children) collectAllEdgeColors(node.children)
    }
  }
  collectEdgeColors(graph.edges)
  if (Array.isArray(graph.children)) collectAllEdgeColors(graph.children)
  for (const color of markerColors) {
    const mid = `ah-${color.replace('#', '')}`
    parts.push(`    <marker id="${mid}" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE/3}" orient="auto" markerUnits="strokeWidth">`)
    parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE/3}, 0 ${ARROW_SIZE/1.5}" fill="${color}" />`)
    parts.push(`    </marker>`)
    parts.push(`    <marker id="${mid}-rev" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="0" refY="${ARROW_SIZE/3}" orient="auto-start-reverse" markerUnits="strokeWidth">`)
    parts.push(`      <polygon points="${ARROW_SIZE} 0, 0 ${ARROW_SIZE/3}, ${ARROW_SIZE} ${ARROW_SIZE/1.5}" fill="${color}" />`)
    parts.push(`    </marker>`)
  }
  parts.push(`  </defs>`)
  parts.push(`  <rect width="${width}" height="${height}" fill="#FFFFFF" />`)

  // Root-level edges
  if (Array.isArray(graph.edges)) {
    for (const edge of graph.edges) {
      if (edge) { const r = renderEdge(edge); if (r) parts.push(r) }
    }
  }

  // Nested edges (coordinates relative to parent compound node)
  function collectNestedEdges(nodes: ElkNode[], ox: number, oy: number) {
    for (const node of nodes) {
      const nodeAbsX = (node.x || 0) + ox, nodeAbsY = (node.y || 0) + oy
      if (node && (node as any).edges && Array.isArray((node as any).edges)) {
        for (const edge of (node as any).edges as ElkEdge[]) {
          if (edge) { const r = renderEdge(edge, nodeAbsX, nodeAbsY); if (r) parts.push(r) }
        }
      }
      if (node && node.children && Array.isArray(node.children)) {
        collectNestedEdges(node.children, nodeAbsX, nodeAbsY)
      }
    }
  }
  if (Array.isArray(graph.children)) {
    collectNestedEdges(graph.children, 0, 0)
  }

  // Nodes
  if (Array.isArray(graph.children)) {
    graph.children.slice(0, 100).forEach((node, i) => {
      if (node) parts.push(renderNode(node, i, 0, 0, 0, cleanMode))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════════════════════

function escapeXml(str: string): string {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')
}

function _fallbackSvg(message: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" width="400" height="200" style="max-width:100%;height:auto;">
  <rect width="400" height="200" fill="#FAFAFA" rx="8" stroke="#E0E0E0" stroke-width="1"/>
  <text x="200" y="90" text-anchor="middle" font-family="system-ui, sans-serif" font-size="14" fill="#78909C">⚠ Skeleton Generation Issue</text>
  <text x="200" y="115" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#B0BEC5">${escapeXml(message)}</text>
  <text x="200" y="140" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" fill="#B0BEC5">Try re-generating or check topology JSON</text>
</svg>`
}

export default elkToSvg