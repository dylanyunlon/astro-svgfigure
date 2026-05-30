/**
 * to-svg.ts -- ELK Layouted JSON -> Skeleton SVG
 * Enhanced with advanced edge routing: dashed, bidirectional, curved, labeled edges
 * GitHub: EmilStenstrom/elkjs-svg, kieler/elkjs
 */

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

// Academic paper style: white nodes, dark borders, monochrome edges
const NODE_FILL = '#FFFFFF'
const NODE_FILL_GROUP = 'none'
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

// Render options. `clean` suppresses the decorative scattered rounded-rects
// (skeleton sketch texture) — used for final/export output where the grey
// blobs read as noise. Default false keeps the existing skeleton look so no
// caller breaks.
//
// cleanMode is threaded as a parameter through renderNode — not a module-level
// mutable — to avoid SSR race conditions.  In Astro SSR, module-level vars
// persist across requests; if user A sets clean=true and user B starts a render
// before A finishes, B inherits A's clean state.
export interface ToSvgOptions { clean?: boolean }

export function elkToSvg(graph: ElkGraph, opts?: ToSvgOptions): string {
  const cleanMode = !!(opts && opts.clean)
  if (!graph) {
    console.warn('[elkToSvg] graph is null/undefined')
    return _fallbackSvg('No graph data')
  }
  // Ensure we have at least children or edges to render
  const hasChildren = Array.isArray(graph.children) && graph.children.length > 0
  const hasEdges = Array.isArray(graph.edges) && graph.edges.length > 0
  if (!hasChildren && !hasEdges) {
    console.warn('[elkToSvg] graph has no children and no edges')
    return _fallbackSvg('Empty graph (no nodes)')
  }

  // Calculate bounds from actual node positions if graph width/height are 0
  // Recursively check all nested children for accurate bounds
  let gw = graph.width || 0
  let gh = graph.height || 0
  if (gw === 0 || gh === 0) {
    function computeBounds(nodes: ElkNode[], ox: number, oy: number) {
      for (const child of nodes) {
        const cx = (child.x || 0) + ox
        const cy = (child.y || 0) + oy
        const right = cx + (child.width || 160)
        const bottom = cy + (child.height || 60)
        if (right > gw) gw = right
        if (bottom > gh) gh = bottom
        // Recurse into nested children with accumulated offset
        if (Array.isArray(child.children) && child.children.length > 0) {
          computeBounds(child.children, cx, cy)
        }
      }
    }
    computeBounds(graph.children || [], 0, 0)
    gw = Math.max(gw, 200)
    gh = Math.max(gh, 150)
  }

  const width = gw + PADDING * 2
  const height = gh + PADDING * 2
  const parts: string[] = []

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="max-width:100%;height:auto;">`)
  parts.push(`  <defs>`)
  parts.push(`    <marker id="ah-default" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE/3}" orient="auto" markerUnits="strokeWidth">`)
  parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE/3}, 0 ${ARROW_SIZE/1.5}" fill="${DEFAULT_EDGE_COLOR}" />`)
  parts.push(`    </marker>`)

  // Generate colored markers for each edge color (including nested edges)
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

  if (Array.isArray(graph.edges)) {
    for (const edge of graph.edges) {
      if (edge) { const r = renderEdge(edge); if (r) parts.push(r) }
    }
  }

  // Also render nested edges inside compound nodes (recursive)
  // Nested edge coordinates are relative to their parent compound node
  function collectNestedEdges(nodes: ElkNode[], ox: number, oy: number) {
    for (const node of nodes) {
      const nodeAbsX = (node.x || 0) + ox
      const nodeAbsY = (node.y || 0) + oy
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

  if (Array.isArray(graph.children)) {
    graph.children.slice(0, 100).forEach((node, i) => {
      if (node) parts.push(renderNode(node, i, 0, 0, 0, cleanMode))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

// ═══════════════════════════════════════════════════════════════════════════
//  §4  Skeleton group colors — the 2-3 largest parent groups get colored
//      background panels, like FreqSelect's green / orange / yellow regions
// ═══════════════════════════════════════════════════════════════════════════

const SKELETON_FILLS = [
  { bg: '#EBF5EB', stroke: '#A5D6A7' },  // green  (FreqSelect)
  { bg: '#FFF3E0', stroke: '#FFCC80' },  // orange (AdaDR)
  { bg: '#FFF8E1', stroke: '#FFE082' },  // yellow (AdaKern)
  { bg: '#E3F2FD', stroke: '#90CAF9' },  // blue
  { bg: '#F3E5F5', stroke: '#CE93D8' },  // purple
]

// Per-family palette: nodes in the same family share the first color, each
// member gets a slightly different tint for micro-diff.
const FAMILY_PALETTES: string[][] = [
  ['#B3D4FC', '#9EC5F0', '#89B6E4', '#74A7D8'],  // blues  (feature maps)
  ['#F8BBD0', '#F48FB1', '#F06292', '#EC407A'],  // pinks  (attention / selection)
  ['#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A'],  // greens (decomposed)
  ['#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],  // oranges
  ['#D1C4E9', '#B39DDB', '#9575CD', '#7E57C2'],  // purples
  ['#B2EBF2', '#80DEEA', '#4DD0E1', '#26C6DA'],  // cyans
]

function _familyPalette(familyId: string): string[] {
  if (!familyId) return FAMILY_PALETTES[0]
  let h = 0
  for (let i = 0; i < familyId.length; i++) h = Math.imul(h ^ familyId.charCodeAt(i), 16777619)
  return FAMILY_PALETTES[((h >>> 0) % FAMILY_PALETTES.length)]
}

function renderNode(node: ElkNode, index: number, depth: number = 0, offsetX: number = 0, offsetY: number = 0, cleanMode: boolean = false): string {
  const x = (node.x || 0) + PADDING + offsetX, y = (node.y || 0) + PADDING + offsetY
  const w = node.width || 160, h = node.height || 60
  const isGroup = Array.isArray(node.children) && node.children.length > 0
  const label = node.labels?.[0]?.text || node.id
  const nodeType = isGroup ? 'group' : 'leaf'

  // Smart label truncation based on node width
  const maxChars = Math.max(6, Math.floor(w / 8))
  const dl = label.length > maxChars ? label.slice(0, maxChars - 2) + '\u2026' : label

  let svg = `  <g data-node-id="${escapeXml(node.id)}" data-node-type="${nodeType}" data-depth="${depth}" data-bbox="${x},${y},${w},${h}" data-render-mode="${node.renderMode || 'text'}">`

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  Path A: Skeleton group — parent groups get organic blob backgrounds
  //  using the rounded-rect union technique (cloud/petal edges).
  //  No rectangular frames — the blob IS the group container.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (isGroup) {
    if (depth <= 1) {
      // Top-level / second-level: rounded-rect blob background
      const palette = SKELETON_FILLS[index % SKELETON_FILLS.length]
      svg += _renderGroupBlob(node.id + '_group', x, y, w, h, palette.bg)
      // Group label: top-left, italic, small — like academic figure captions
      svg += `<text x="${x + 12}" y="${y + 18}" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${palette.stroke}" font-weight="700" font-style="italic">${escapeXml(dl)}</text>`
    } else {
      // Deeper group: borderless transparent container, just a label
      svg += `<text x="${x + 8}" y="${y + 14}" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888" font-weight="600" font-style="italic">${escapeXml(dl)}</text>`
    }

    if (node.children) node.children.forEach((c, i) => { svg += renderNode(c, i, depth + 1, x - PADDING, y - PADDING, cleanMode) })
    svg += `</g>`
    return svg
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  Path B: Operator nodes — pure SVG circle + glyph (⊗ ⊕ ⊙ ...)
  //  Zero-cost vector path, no AI, no box. Priority over everything else.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if ((node as any).isOperator) {
    svg += renderMathOperator(dl, x + w / 2, y + h / 2, Math.min(w, h) * 0.4)
    svg += `</g>`
    return svg
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  Path C: Sprite nodes — the blob IS the node body, NO white box
  //  underneath.  Label goes BELOW the blob, small font, like in real
  //  academic figures where "C×H×W" is a tiny caption under a drawn
  //  feature map block.
  //
  //  Family-aware: nodes in the same family share a color palette,
  //  differing only in blob seed (micro-diff on one axis).
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  if (node.renderMode === 'sprite') {
    // If a real AI sprite is available, use it
    if (node.spriteRef && (node.spriteRef.url || node.spriteRef.svg)) {
      svg += renderSprite(node, x, y, w, h)
    } else if (node.spriteRef && node.spriteRef.format === 'stack') {
      // 3D feature-map stack — the academic-figure visual
      svg += renderFeatureMapStack(node, x, y, w, Math.max(h - 18, h * 0.7))
    } else {
      // Organic blob — the sprite placeholder / default visual.
      // Use family palette for color consistency within families.
      const familyId = (node as any).familyId || ''
      const palette = _familyPalette(familyId)

      // Temporarily set fillColor from family palette so renderOrganicBlob picks it up
      const origColor = (node as any).fillColor
      ;(node as any).fillColor = palette[0]
      svg += renderOrganicBlob(node, x, y, w, Math.max(h - 18, h * 0.7))
      ;(node as any).fillColor = origColor
    }

    // Small label BELOW the blob — like academic figure captions
    const labelY = y + h - 4
    const fontSize = Math.max(7, Math.min(10, w / dl.length * 1.2))
    svg += `<text x="${x + w / 2}" y="${labelY}" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize.toFixed(1)}" fill="${TEXT_COLOR}" font-weight="500" font-style="italic">${escapeXml(dl)}</text>`

    svg += `</g>`
    return svg
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  Path D: Label-only nodes (height ≤ 30, no icon) — naked text
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  const isLabelOnly = !!(node as any).labelOnly || (h <= 30 && !(node as any).iconHint)
  if (isLabelOnly) {
    const fontSize = h > 30 ? 13 : 11
    svg += `<text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="600">${escapeXml(dl)}</text>`
    svg += `</g>`
    return svg
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  Path E: Regular leaf nodes — white box + scattered rects + centered label
  //  This is the default for icon/text nodes that aren't sprites or operators.
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  const nodeRand = _seededRand(node.id)
  const mainRx = Math.min(12, Math.min(w, h) * 0.15)
  svg += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${NODE_FILL}" stroke="${STROKE_COLOR}" stroke-width="0.8" rx="${mainRx}" />`

  // Scatter 3-6 small decorative rounded rects (skeleton sketch texture).
  if (!cleanMode) {
    const numScatter = 3 + Math.floor(nodeRand() * 4)
    const pad = 4
    for (let si = 0; si < numScatter; si++) {
      const sw2 = w * (0.15 + nodeRand() * 0.35)
      const sh2 = h * (0.12 + nodeRand() * 0.25)
      const sx = x + pad + nodeRand() * Math.max(0, w - sw2 - pad * 2)
      const sy = y + pad + nodeRand() * Math.max(0, h - sh2 - pad * 2)
      const srx = 3 + nodeRand() * 8
      const sOpacity = 0.04 + nodeRand() * 0.08
      const sRotate = (nodeRand() - 0.5) * 6
      svg += `<rect x="${sx.toFixed(1)}" y="${sy.toFixed(1)}" width="${sw2.toFixed(1)}" height="${sh2.toFixed(1)}" rx="${srx.toFixed(1)}" fill="${STROKE_COLOR}" opacity="${sOpacity.toFixed(3)}" transform="rotate(${sRotate.toFixed(1)} ${(sx+sw2/2).toFixed(1)} ${(sy+sh2/2).toFixed(1)})" />`
    }
  }

  // Label: centered, normal size
  const hasIcon = !!(node as any).iconHint
  const fontSize = (hasIcon && h >= 50) ? 9 : 12
  svg += `<text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="500">${escapeXml(dl)}</text>`

  svg += `</g>`
  return svg
}

function renderMathOperator(symbol: string, cx: number, cy: number, r: number): string {
  // Draw a math operator as a circle plus a vector glyph, centered at (cx,cy).
  // Radius r is min(w,h)*0.4 so it scales with the node box. Pure SVG, no AI,
  // no text — crisp at any zoom. Falls back to a centered glyph for unknown
  // symbols (still vector text, never the scattered-rect decoration).
  r = Math.max(6, r)
  const s = (symbol || '').trim()
  const circle = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${NODE_FILL}" stroke="${STROKE_COLOR}" stroke-width="1.6" />`
  const sw = 1.6
  // inscribe glyph at ~0.62r so it sits inside the circle
  const g = r * 0.62
  if (s === '⊗' || /^(x|×|⨂)$/i.test(s)) {
    // diagonal cross (multiply)
    const d = g * Math.SQRT1_2
    return circle
      + `<line x1="${cx-d}" y1="${cy-d}" x2="${cx+d}" y2="${cy+d}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
      + `<line x1="${cx-d}" y1="${cy+d}" x2="${cx+d}" y2="${cy-d}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
  }
  if (s === '⊕' || /^(\+|⨁)$/.test(s)) {
    // upright plus (add)
    return circle
      + `<line x1="${cx-g}" y1="${cy}" x2="${cx+g}" y2="${cy}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
      + `<line x1="${cx}" y1="${cy-g}" x2="${cx}" y2="${cy+g}" stroke="${STROKE_COLOR}" stroke-width="${sw}" stroke-linecap="round" />`
  }
  if (s === '⊙' || s === '·' || s === '∘') {
    // center dot (Hadamard / composition)
    return circle + `<circle cx="${cx}" cy="${cy}" r="${Math.max(1.5, r*0.18)}" fill="${STROKE_COLOR}" />`
  }
  if (s === '○' || s === '◯') {
    return circle  // hollow circle
  }
  // Unknown operator-like symbol: circle + the glyph as small centered text.
  return circle
    + `<text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${Math.max(9, r*0.9)}" fill="${TEXT_COLOR}">${escapeXml(s)}</text>`
}

// ═══════════════════════════════════════════════════════════════════════════
//  §1  Seeded PRNG — FNV-1a + sfc32
// ═══════════════════════════════════════════════════════════════════════════
//
// Port of g-harel/blobs internal/rand.ts (L5-37).
// Replaces the Java String.hashCode PRNG that had known collisions
// ("Aa" == "BB").  FNV-1a produces no collisions for short ASCII strings;
// sfc32 has 2^128 period and passes TestU01.
//
// CCCL analogy: this is the `init_histograms()` shared primitive that both
// the histogram-only kernel and the fused filter+histogram kernel call.
// The callers don't know or care how bins are zeroed — they just call it.

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


// ═══════════════════════════════════════════════════════════════════════════
//  §2  Organic blob — rounded-rectangle union
// ═══════════════════════════════════════════════════════════════════════════
//
// "你以为这是手绘的？实际上这只是一群圆角矩形"
//
// Technique: 8-12 opaque same-color rounded rectangles, each slightly
// rotated (±3° to ±12°) and offset from center.  Because they share one
// fill and opacity=1, overlapping regions merge visually into a single
// organic silhouette with concave dips and convex bumps — cloud/petal
// edges, not smooth convex ellipses.  No path boolean union, no
// <filter>, no Bézier — just <rect> stacking.
//
// This replaces the previous g-harel/blobs Bézier algorithm (c0e6191)
// to match the hand-drawn look of the reference figures.
//
// Parameters tuned from the reference image analysis:
//   rectCount:  8-12 (family stackCount nudges for micro-diff)
//   sizeRange:  each rect is 35-80% of bbox in each axis
//   rxRange:    corner radius 25-45% of min(rw,rh) — very rounded
//   rotRange:   ±3° to ±12° — subtle but enough for edge texture
//   offsetRange: center ±17.5% of bbox — keeps cluster cohesive

const BLOB_PALETTE = [
  '#E8D5F5',  // lavender
  '#D5E8F5',  // sky
  '#F5E8D5',  // peach
  '#D5F5E0',  // mint
  '#F5D5E0',  // rose
  '#F5F0D5',  // cream
]

// The dedicated kernel: pure geometry, zero side effects.
// Input: (node, x, y, w, h) → Output: SVG markup string.
//
// Each rounded rect is positioned around the blob center with random
// offset, size, corner radius, and rotation.  The deterministic PRNG
// ensures identical output for the same node ID across renders.
function renderOrganicBlob(
  node: ElkNode,
  x: number, y: number, w: number, h: number,
): string {
  const rand = _seededRand(node.id)
  const ref = (node as any).spriteRef

  // 8-12 rounded rects, nudged by family stackCount for variant diversity.
  const rectCount = Math.max(8, Math.min(14,
    (ref && ref.stackCount) ? 8 + (ref.stackCount % 5) : 8 + Math.floor(rand() * 5)))

  const cx = x + w / 2
  const cy = y + h / 2
  const color = (node as any).fillColor || BLOB_PALETTE[0]
  let svg = ''

  for (let i = 0; i < rectCount; i++) {
    // Rect dimensions: 35-80% of bounding box
    const rw = w * (0.35 + rand() * 0.45)
    const rh = h * (0.30 + rand() * 0.40)

    // Corner radius: 25-45% of shorter edge — very rounded, pill-ish
    const rrx = Math.min(rw, rh) * (0.25 + rand() * 0.20)

    // Position: offset from center by up to ±17.5% of bbox
    const ox = (rand() - 0.5) * w * 0.35
    const oy = (rand() - 0.5) * h * 0.35
    const rx = cx - rw / 2 + ox
    const ry = cy - rh / 2 + oy

    // Rotation: ±3° to ±12° around rect center
    const rot = (rand() - 0.5) * 2 * (3 + rand() * 9)
    const rcx = rx + rw / 2
    const rcy = ry + rh / 2

    svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" `
         + `width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" `
         + `rx="${rrx.toFixed(1)}" `
         + `fill="${color}" opacity="1" `
         + `transform="rotate(${rot.toFixed(1)} ${rcx.toFixed(1)} ${rcy.toFixed(1)})" />`
  }

  return svg
}

// ═══════════════════════════════════════════════════════════════════════════
//  §2b  Group blob — same rounded-rect union, tuned for large containers
// ═══════════════════════════════════════════════════════════════════════════
//
// Like renderOrganicBlob but for parent group backgrounds:
//   - Fewer rects (6-8) so the edge texture is gentler at large sizes
//   - Larger relative size per rect (50-90% of bbox) for better coverage
//   - Smaller rotation range (±2° to ±6°) — subtler at scale
//   - Opacity 1.0 — opaque fill, children render on top
function _renderGroupBlob(
  seed: string,
  x: number, y: number, w: number, h: number,
  color: string,
): string {
  const rand = _seededRand(seed)
  const rectCount = 6 + Math.floor(rand() * 3) // 6-8
  const cx = x + w / 2
  const cy = y + h / 2
  let svg = ''

  for (let i = 0; i < rectCount; i++) {
    // Rect dimensions: 50-90% of bounding box — larger for better coverage
    const rw = w * (0.50 + rand() * 0.40)
    const rh = h * (0.45 + rand() * 0.40)

    // Corner radius: 20-35% of shorter edge
    const rrx = Math.min(rw, rh) * (0.20 + rand() * 0.15)

    // Position: offset from center by up to ±12.5% of bbox — tighter cluster
    const ox = (rand() - 0.5) * w * 0.25
    const oy = (rand() - 0.5) * h * 0.25
    const rx = cx - rw / 2 + ox
    const ry = cy - rh / 2 + oy

    // Rotation: ±2° to ±6° — subtler at large scale
    const rot = (rand() - 0.5) * 2 * (2 + rand() * 4)
    const rcx = rx + rw / 2
    const rcy = ry + rh / 2

    svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" `
         + `width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" `
         + `rx="${rrx.toFixed(1)}" `
         + `fill="${color}" opacity="1" `
         + `transform="rotate(${rot.toFixed(1)} ${rcx.toFixed(1)} ${rcy.toFixed(1)})" />`
  }

  return svg
}


// ═══════════════════════════════════════════════════════════════════════════
//  §2c  Feature-map stack — 3D parallelogram stack (FreqSelect reference)
// ═══════════════════════════════════════════════════════════════════════════
//
// Draws the academic-figure visual for feature maps / tensor volumes:
//   - Front face: filled rectangle with procedural texture (seeded noise)
//   - N back faces: offset diagonally (up-right), progressive opacity
//   - stackCount: how many layers in the 3D stack (1 = single slab)
//   - memberIndex: within a family, picks texture seed for micro-diff
//
// This is the visual that replaces text boxes for sprite nodes — the blob
// IS the node body, no white box underneath.  Caption goes below.
//
// Reference: FreqSelect figure — "Input feature C×H×W", "Decomposed feats",
// "Selection map 1×H×W" are all drawn as these stacked parallelogram slabs.
function renderFeatureMapStack(
  node: ElkNode,
  x: number, y: number, w: number, h: number,
): string {
  const rand = _seededRand(node.id)
  const ref = (node as any).spriteRef
  const familyId = (node as any).familyId || ''
  const palette = _familyPalette(familyId)
  const color = (node as any).fillColor || palette[0]

  // Stack depth: from spriteRef.stackCount or default 3
  const stackCount = Math.max(1, (ref && ref.stackCount) || 3)

  // 3D offset per layer
  const dx = Math.min(4, w * 0.05)
  const dy = Math.min(4, h * 0.05)

  // Total offset consumed by stacking
  const totalDx = dx * (stackCount - 1)
  const totalDy = dy * (stackCount - 1)

  // Front face dimensions (shrink to fit stack within bbox)
  const fw = w - totalDx
  const fh = h - totalDy
  const fx = x
  const fy = y + totalDy

  let svg = ''

  // Back layers (drawn first, back-to-front)
  for (let i = stackCount - 1; i >= 1; i--) {
    const lx = fx + dx * i
    const ly = fy - dy * i
    const backOpacity = 0.25 + (i / stackCount) * 0.30
    svg += `<rect x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" width="${fw.toFixed(1)}" height="${fh.toFixed(1)}" `
         + `fill="${color}" opacity="${backOpacity.toFixed(2)}" stroke="#666" stroke-width="0.4" />`
  }

  // Front face
  svg += `<rect x="${fx.toFixed(1)}" y="${fy.toFixed(1)}" width="${fw.toFixed(1)}" height="${fh.toFixed(1)}" `
       + `fill="${color}" opacity="0.55" stroke="#555" stroke-width="0.6" />`

  // Procedural texture on front face — seeded noise rectangles that
  // give the "image content" look.  memberIndex (from familyId hash)
  // shifts texture pattern so family members share the same color
  // but have different internal detail (micro-diff on one axis).
  let memberHash = 0
  for (let i = 0; i < familyId.length; i++) memberHash = Math.imul(memberHash ^ familyId.charCodeAt(i), 0x5bd1e995)
  const memberIndex = (memberHash >>> 0) % 6

  const textureCount = 6 + Math.floor(rand() * 4)
  for (let t = 0; t < textureCount; t++) {
    const tw = fw * (0.12 + rand() * 0.35)
    const th = fh * (0.10 + rand() * 0.30)
    const tx = fx + rand() * Math.max(0, fw - tw)
    const ty = fy + rand() * Math.max(0, fh - th)
    const texOpacity = 0.06 + rand() * 0.16
    const texColor = (memberIndex + t) % 2 === 0 ? '#ffffff' : '#000000'
    svg += `<rect x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" width="${tw.toFixed(1)}" height="${th.toFixed(1)}" `
         + `rx="1" fill="${texColor}" opacity="${texOpacity.toFixed(2)}" />`
  }

  // Diagonal connecting lines (3D perspective edges)
  for (let i = 0; i < stackCount - 1; i++) {
    const x1 = fx + dx * i + fw
    const y1 = fy - dy * i
    const x2 = fx + dx * (i + 1) + fw
    const y2 = fy - dy * (i + 1)
    svg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#888" stroke-width="0.3" />`
  }

  return svg
}


// ═══════════════════════════════════════════════════════════════════════════
//  §3  renderSprite — the dispatch() function
// ═══════════════════════════════════════════════════════════════════════════
//
// CCCL f984c90's dispatch() is pure orchestration:
//
//   // Pass 0: dedicated histogram-only kernel
//   { launcher.doit(histogram_kernel, ...); }
//
//   // Passes 1..N: fused filter+histogram with DoubleBuffer
//   for (; pass < num_passes; pass++) {
//       launcher.doit(topk_kernel, key_bufs.Current(), ...);
//       key_bufs.selector ^= 1;
//   }
//
//   // Final: invoke_last_filter on key_bufs.Current()
//
// Our renderSprite is the same: pure routing, zero rendering logic.
//   format 'stack'              → renderFeatureMapStack  (3D parallelogram stack)
//   no payload (no url, no svg) → renderOrganicBlob      (rounded-rect union blob)
//   format 'svg' with inline SVG → scale + translate      (fused kernel)
//   format 'png' with URL        → <image>                (fused kernel, different op)
//   fallback                     → dashed box + label      (invoke_last_filter)

function renderSprite(node: ElkNode, x: number, y: number, w: number, h: number): string {
  const ref = node.spriteRef
  if (!ref) return ''

  // ── Pass 0a: 3D feature-map stack (explicit 'stack' format)
  if (ref.format === 'stack') {
    return renderFeatureMapStack(node, x, y, w, h)
  }

  // ── Pass 0b: organic blob fallback (no AI image yet)
  if (!ref.url && !ref.svg) {
    return renderOrganicBlob(node, x, y, w, h)
  }

  // ── Passes 1..N: AI-generated sprite — fill ~80% of node center
  const FILL = 0.8
  const drawW = Math.max(1, w * FILL)
  const drawH = Math.max(1, h * FILL)
  const dx = x + (w - drawW) / 2
  const dy = y + (h - drawH) / 2

  if (ref.format === 'svg' && ref.svg) {
    const natW = ref.bbox?.[2] || drawW
    const natH = ref.bbox?.[3] || drawH
    const sx = drawW / natW
    const sy = drawH / natH
    return `<g transform="translate(${dx.toFixed(2)} ${dy.toFixed(2)}) scale(${sx.toFixed(4)} ${sy.toFixed(4)})" data-sprite="svg">${ref.svg}</g>`
  }

  if (ref.url) {
    return `<image href="${ref.url}" x="${dx.toFixed(2)}" y="${dy.toFixed(2)}" width="${drawW.toFixed(2)}" height="${drawH.toFixed(2)}" preserveAspectRatio="none" data-sprite="png" />`
  }

  // ── invoke_last_filter: fallback dashed box — never silently empty.
  // The old code returned '' here, leaving arrows pointing at empty space.
  const fallbackLabel = node.labels?.[0]?.text || node.id
  const maxC = Math.max(6, Math.floor(w / 8))
  const fbl = fallbackLabel.length > maxC ? fallbackLabel.slice(0, maxC - 2) + '\u2026' : fallbackLabel
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#FFFFFF" stroke="#999" stroke-width="0.8" rx="6" stroke-dasharray="4,3" />`
    + `<text x="${x + w / 2}" y="${y + h / 2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#999" font-style="italic">${escapeXml(fbl)}</text>`
}

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
      const dx = pts[1].x-pts[0].x, dy = pts[1].y-pts[0].y
      const cx = mx - dy*curv*0.5, cy = my + dx*curv*0.5
      d = `M${pts[0].x},${pts[0].y} Q${cx},${cy} ${pts[1].x},${pts[1].y}`
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

function escapeXml(str: string): string {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')
}

/** Fallback SVG shown when graph cannot be rendered */
function _fallbackSvg(message: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" width="400" height="200" style="max-width:100%;height:auto;">
  <rect width="400" height="200" fill="#FAFAFA" rx="8" stroke="#E0E0E0" stroke-width="1"/>
  <text x="200" y="90" text-anchor="middle" font-family="system-ui, sans-serif" font-size="14" fill="#78909C">⚠ Skeleton Generation Issue</text>
  <text x="200" y="115" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#B0BEC5">${escapeXml(message)}</text>
  <text x="200" y="140" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" fill="#B0BEC5">Try re-generating or check topology JSON</text>
</svg>`
}

export default elkToSvg