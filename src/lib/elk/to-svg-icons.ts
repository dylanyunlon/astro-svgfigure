/**
 * to-svg-icons.ts — ELK Layouted JSON → Academic SVG with Embedded Icons
 *
 * Enhanced skeleton renderer that consumes `iconHint` fields on nodes,
 * resolving them to Iconify CDN SVG icons embedded directly in the output.
 * This closes the gap between the plain `to-svg.ts` skeleton (box+text only)
 * and the full NanoBanana beautify pipeline (requires Python backend + LLM).
 *
 * Architecture:
 *   node.iconHint ("database", "neural activations", …)
 *     → shared/icon-aliases.json (single source of truth, also used by Python)
 *     → Iconify CDN <image href="https://api.iconify.design/{prefix}/{name}.svg">
 *     → embedded in SVG node alongside label text
 *
 * Falls back gracefully: if no iconHint or no alias match, node renders
 * as a clean colored box with text (same as to-svg.ts).
 *
 * Icon alias data:
 *   - shared/icon-aliases.json is the SINGLE SOURCE OF TRUTH
 *   - Python backend/pipeline/svg_icon_fetcher.py reads the same file
 *   - Adding a new alias in the JSON automatically works in both runtimes
 *
 * Other ported logic:
 *   - backend/pipeline/scaffold_builder.py  (color palette, group handling)
 *   - src/lib/elk/to-svg.ts                 (edge routing, arrow markers)
 *
 * GitHub: kieler/elkjs, Iconify/api, EmilStenstrom/elkjs-svg
 */

// ── Icon Aliases (loaded from shared single source of truth) ────────
// shared/icon-aliases.json is also consumed by Python svg_icon_fetcher.py
// Format: { "default_collection": "tabler", "aliases": { "hint": "icon-name" } }
import iconAliasData from '../../../shared/icon-aliases.json'

const ICON_ALIASES: Record<string, string> = iconAliasData.aliases || {}
const DEFAULT_COLLECTION: string = iconAliasData.default_collection || 'tabler'

// Pre-sort alias keys by length (longest first) for substring matching.
// Computed once at module load, not per-call.
const SORTED_ALIAS_KEYS: string[] = Object.keys(ICON_ALIASES).sort((a, b) => b.length - a.length)

// ── Iconify CDN URL builder ─────────────────────────────────────────

function resolveIconUrl(hint: string): string | null {
  if (!hint) return null
  const low = hint.trim().toLowerCase()

  // 1) Exact match against shared aliases
  if (ICON_ALIASES[low]) return _iconifyUrl(ICON_ALIASES[low])

  // 2) Partial / substring match (longest alias first to avoid false positives)
  for (const term of SORTED_ALIAS_KEYS) {
    if (low.includes(term)) return _iconifyUrl(ICON_ALIASES[term])
  }

  // 3) Try matching individual words from the hint
  const words = low.replace(/[^a-z0-9\s-]/g, '').split(/\s+/)
  for (const w of words) {
    if (ICON_ALIASES[w]) return _iconifyUrl(ICON_ALIASES[w])
  }

  return null
}

function _iconifyUrl(iconName: string): string {
  // iconName is just the name part (e.g. "database", "brain")
  // Prefix with default collection to form full Iconify CDN URL
  return `https://api.iconify.design/${DEFAULT_COLLECTION}/${iconName}.svg?width=28&height=28`
}

// ── Color Palette (from scaffold_builder.py) ────────────────────────

const PALETTE = {
  fills: [
    '#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5',
    '#E0F7FA', '#FBE9E7', '#F1F8E9', '#EDE7F6',
    '#FCE4EC', '#E0F2F1',
  ],
  strokes: [
    '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC',
    '#26C6DA', '#FF7043', '#9CCC65', '#7E57C2',
    '#EC407A', '#26A69A',
  ],
  groupFill: '#FAFBFC',
  groupStroke: '#CBD5E1',
  text: '#1E293B',
  textSecondary: '#475569',
}

// ── Semantic Edge Styles (from to-svg.ts) ───────────────────────────

const SEMANTIC_STYLES: Record<string, { color: string; dash: string; width: number }> = {
  data_flow:        { color: '#78909C', dash: '',      width: 1.5 },
  gradient_flow:    { color: '#E57373', dash: '8,4',   width: 1.5 },
  skip_connection:  { color: '#4CAF50', dash: '',      width: 2 },
  optional_path:    { color: '#9E9E9E', dash: '5,5',   width: 1 },
  inference_only:   { color: '#7986CB', dash: '10,3,3,3', width: 1.5 },
  fan_out:          { color: '#FF9800', dash: '',      width: 1.5 },
  fan_in:           { color: '#2196F3', dash: '',      width: 1.5 },
  feedback:         { color: '#AB47BC', dash: '6,3',   width: 1.5 },
  attention:        { color: '#F44336', dash: '2,4',   width: 2 },
  concatenation:    { color: '#009688', dash: '',      width: 2 },
  residual:         { color: '#4CAF50', dash: '',      width: 2 },
  cross_boundary:   { color: '#607D8B', dash: '',      width: 1.5 },
}

const DEFAULT_EDGE_COLOR = '#94A3B8'
const ARROW_SIZE = 8
const PADDING = 20

// ── Types ───────────────────────────────────────────────────────────

interface ElkNode {
  id: string; x?: number; y?: number; width?: number; height?: number
  labels?: { text: string }[]; children?: ElkNode[]; edges?: ElkEdge[]
  iconHint?: string; group?: boolean; borderless?: boolean
  layoutOptions?: Record<string, string>
}

interface AdvancedEdge {
  routing?: string; lineStyle?: string; strokeDasharray?: string
  strokeWidth?: number; strokeColor?: string
  sourceArrow?: string; targetArrow?: string; directionality?: string
  semanticType?: string; curvature?: number
  edgeLabels?: { text: string; position?: number; fontSize?: number; backgroundColor?: string }[]
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

// ── Depth-based color assignment ────────────────────────────────────

let _colorIndex = 0
const _nodeColorMap = new Map<string, number>()

function getNodeColor(nodeId: string, depth: number): { fill: string; stroke: string } {
  if (!_nodeColorMap.has(nodeId)) {
    _nodeColorMap.set(nodeId, _colorIndex++)
  }
  const idx = _nodeColorMap.get(nodeId)! % PALETTE.fills.length
  return { fill: PALETTE.fills[idx], stroke: PALETTE.strokes[idx] }
}

function resetColors() {
  _colorIndex = 0
  _nodeColorMap.clear()
}

// ── Main Export ─────────────────────────────────────────────────────

export function elkToSvgIcons(graph: ElkGraph): string {
  if (!graph) return _fallbackSvg('No graph data')

  const hasChildren = Array.isArray(graph.children) && graph.children.length > 0
  const hasEdges = Array.isArray(graph.edges) && graph.edges.length > 0
  if (!hasChildren && !hasEdges) return _fallbackSvg('Empty graph (no nodes)')

  resetColors()

  // Calculate bounds
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

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="max-width:100%;height:auto;">`)

  // Defs: arrow markers + drop shadow filter
  parts.push(`  <defs>`)
  parts.push(`    <filter id="shadow" x="-4%" y="-4%" width="108%" height="108%"><feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.08"/></filter>`)
  parts.push(`    <marker id="ah-default" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE / 1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE / 3}" orient="auto" markerUnits="strokeWidth">`)
  parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE / 3}, 0 ${ARROW_SIZE / 1.5}" fill="${DEFAULT_EDGE_COLOR}" />`)
  parts.push(`    </marker>`)

  // Collect edge colors for colored markers
  const markerColors = new Set<string>()
  function collectEdgeColors(edgeList: ElkEdge[] | undefined) {
    if (!Array.isArray(edgeList)) return
    for (const e of edgeList) {
      if (e?.advanced?.strokeColor) markerColors.add(e.advanced.strokeColor)
      if (e?.advanced?.semanticType && SEMANTIC_STYLES[e.advanced.semanticType]?.color)
        markerColors.add(SEMANTIC_STYLES[e.advanced.semanticType].color)
    }
  }
  function collectAllEdgeColors(nodes: ElkNode[]) {
    for (const node of nodes) {
      collectEdgeColors(node.edges)
      if (node.children) collectAllEdgeColors(node.children)
    }
  }
  collectEdgeColors(graph.edges)
  if (Array.isArray(graph.children)) collectAllEdgeColors(graph.children)
  for (const color of markerColors) {
    const mid = `ah-${color.replace('#', '')}`
    parts.push(`    <marker id="${mid}" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE / 1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE / 3}" orient="auto" markerUnits="strokeWidth">`)
    parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE / 3}, 0 ${ARROW_SIZE / 1.5}" fill="${color}" />`)
    parts.push(`    </marker>`)
    parts.push(`    <marker id="${mid}-rev" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE / 1.5}" refX="0" refY="${ARROW_SIZE / 3}" orient="auto-start-reverse" markerUnits="strokeWidth">`)
    parts.push(`      <polygon points="${ARROW_SIZE} 0, 0 ${ARROW_SIZE / 3}, ${ARROW_SIZE} ${ARROW_SIZE / 1.5}" fill="${color}" />`)
    parts.push(`    </marker>`)
  }
  parts.push(`  </defs>`)

  // Background
  parts.push(`  <rect width="${width}" height="${height}" fill="#FAFAFA" rx="6" />`)

  // Render edges (bottom layer) — root level
  if (Array.isArray(graph.edges)) {
    for (const edge of graph.edges) {
      if (edge) { const r = renderEdge(edge); if (r) parts.push(r) }
    }
  }

  // Nested edges
  function collectNestedEdges(nodes: ElkNode[], ox: number, oy: number) {
    for (const node of nodes) {
      const nodeAbsX = (node.x || 0) + ox
      const nodeAbsY = (node.y || 0) + oy
      if (node?.edges && Array.isArray(node.edges)) {
        for (const edge of node.edges) {
          if (edge) { const r = renderEdge(edge, nodeAbsX, nodeAbsY); if (r) parts.push(r) }
        }
      }
      if (node?.children && Array.isArray(node.children)) {
        collectNestedEdges(node.children, nodeAbsX, nodeAbsY)
      }
    }
  }
  if (Array.isArray(graph.children)) {
    collectNestedEdges(graph.children, 0, 0)
  }

  // Render nodes (top layer)
  if (Array.isArray(graph.children)) {
    graph.children.slice(0, 200).forEach((node, i) => {
      if (node) parts.push(renderNode(node, i, 0))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

// ── Node Renderer ───────────────────────────────────────────────────

function renderNode(
  node: ElkNode,
  index: number,
  depth: number,
  offsetX: number = 0,
  offsetY: number = 0,
): string {
  const x = (node.x || 0) + PADDING + offsetX
  const y = (node.y || 0) + PADDING + offsetY
  const w = node.width || 160
  const h = node.height || 60
  const isGroup = (Array.isArray(node.children) && node.children.length > 0) || node.group
  const label = node.labels?.[0]?.text || node.id
  const nodeType = isGroup ? 'group' : 'leaf'

  // Wrap each node in a <g> with semantic attributes for SVG-native layer separation.
  // svg-layer-separator.ts reads data-node-id, data-node-type, and data-bbox
  // to extract layers without rasterization.
  let svg = `  <g data-node-id="${escapeXml(node.id)}" data-node-type="${nodeType}" data-depth="${depth}" data-bbox="${x},${y},${w},${h}">\n`

  if (isGroup) {
    // Group container: dashed border, subtle fill, label at top
    const isBorderless = node.borderless
    const groupStroke = isBorderless ? 'none' : PALETTE.groupStroke
    const groupDash = isBorderless ? '' : ' stroke-dasharray="6,3"'
    const groupFill = isBorderless ? 'transparent' : `${PALETTE.groupFill}`

    svg += `  <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${groupFill}" stroke="${groupStroke}" stroke-width="1.5" rx="12"${groupDash} />\n`
    svg += `  <text x="${x + 12}" y="${y + 18}" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="${PALETTE.textSecondary}" font-weight="600" letter-spacing="0.3">${escapeXml(label)}</text>\n`

    // Render children with offset
    if (node.children) {
      node.children.forEach((c, i) => {
        svg += renderNode(c, index * 10 + i, depth + 1, x - PADDING, y - PADDING)
      })
    }
  } else {
    // ── Leaf node ──────────────────────────────────────────────────
    // Two rendering modes:
    //   1. labelOnly=true  → bare text, no rect/border/fill (academic annotations)
    //   2. normal (default) → colored box with optional icon + label

    const isLabelOnly = !!(node as any).labelOnly

    if (isLabelOnly) {
      // ── Label-only node: naked text, no box ─────────────────────
      // Like "Join Pattern", "Selectivity", "Code" in academic figures.
      // Just text floating at the node position, maybe with a subtle
      // font weight to distinguish it from edge labels.
      const fontSize = h > 30 ? 13 : 11
      const fontWeight = '600'
      const textColor = PALETTE.text

      // Smart label: no truncation needed since there's no box constraint
      svg += `  <text x="${x + w / 2}" y="${y + h / 2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${textColor}" font-weight="${fontWeight}">${escapeXml(label)}</text>\n`

    } else {
      // ── Normal boxed node ───────────────────────────────────────
      const { fill, stroke } = getNodeColor(node.id, depth)
      const iconUrl = resolveIconUrl(node.iconHint || '')
      const hasIcon = !!iconUrl

      const nodeRx = 8
      const shadowFilter = depth < 2 ? ' filter="url(#shadow)"' : ''

      svg += `  <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" stroke-width="1.5" rx="${nodeRx}"${shadowFilter} />\n`

      // Smart label truncation
      const maxChars = Math.max(6, Math.floor(w / 8))
      const dl = label.length > maxChars ? label.slice(0, maxChars - 2) + '\u2026' : label

      if (hasIcon) {
        // Layout: icon (28×28) centered above text
        const iconSize = 28
        const iconX = x + (w - iconSize) / 2
        const totalContent = iconSize + 4 + 14 // icon + gap + text height
        const iconY = y + (h - totalContent) / 2
        const textY = iconY + iconSize + 4 + 10

        svg += `  <image href="${iconUrl}" x="${iconX}" y="${iconY}" width="${iconSize}" height="${iconSize}" opacity="0.85" />\n`
        svg += `  <text x="${x + w / 2}" y="${textY}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="${PALETTE.text}" font-weight="500">${escapeXml(dl)}</text>\n`
      } else {
        // No icon: center text vertically
        svg += `  <text x="${x + w / 2}" y="${y + h / 2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${PALETTE.text}" font-weight="500">${escapeXml(dl)}</text>\n`
      }
    }
  }

  svg += `  </g>\n`
  return svg
}

// ── Edge Renderer (from to-svg.ts, enhanced) ────────────────────────

function renderEdge(edge: ElkEdge, offsetX: number = 0, offsetY: number = 0): string {
  if (!edge?.sections?.length) return ''

  const adv = edge.advanced || {}
  const sem = adv.semanticType ? SEMANTIC_STYLES[adv.semanticType] || {} : {} as any
  const color = adv.strokeColor || sem.color || DEFAULT_EDGE_COLOR
  const sw = adv.strokeWidth || sem.width || 1.5
  const da = adv.strokeDasharray || (adv.lineStyle === 'dashed' ? '8,4' : adv.lineStyle === 'dotted' ? '2,4' : sem.dash || '')
  const curv = adv.curvature || sem.curvature || 0
  const isBidir = adv.directionality === 'bidirectional'
  const mid = `ah-${color.replace('#', '')}`
  const hasSrcArrow = isBidir || adv.sourceArrow === 'arrow'
  const noTgtArrow = adv.targetArrow === 'none'

  let svg = ''

  for (const section of edge.sections) {
    if (!section?.startPoint || !section?.endPoint) continue

    const pts: { x: number; y: number }[] = []
    pts.push({ x: (section.startPoint.x ?? 0) + PADDING + offsetX, y: (section.startPoint.y ?? 0) + PADDING + offsetY })
    if (Array.isArray(section.bendPoints)) {
      for (const bp of section.bendPoints) {
        if (bp && typeof bp.x === 'number' && typeof bp.y === 'number')
          pts.push({ x: bp.x + PADDING + offsetX, y: bp.y + PADDING + offsetY })
      }
    }
    pts.push({ x: (section.endPoint.x ?? 0) + PADDING + offsetX, y: (section.endPoint.y ?? 0) + PADDING + offsetY })

    let d: string
    if (curv > 0 && pts.length === 2) {
      const mx = (pts[0].x + pts[1].x) / 2, my = (pts[0].y + pts[1].y) / 2
      const dx = pts[1].x - pts[0].x, dy = pts[1].y - pts[0].y
      const cx = mx - dy * curv * 0.5, cy = my + dx * curv * 0.5
      d = `M${pts[0].x},${pts[0].y} Q${cx},${cy} ${pts[1].x},${pts[1].y}`
    } else if (pts.length > 2) {
      // Smooth orthogonal corners (rounded bends)
      d = buildSmoothPath(pts)
    } else {
      d = pts.map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ')
    }

    let attrs = `d="${d}" fill="none" stroke="${color}" stroke-width="${sw}"`
    if (da) attrs += ` stroke-dasharray="${da}"`
    attrs += ` stroke-linecap="round" stroke-linejoin="round"`
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
    const allPts = [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint]
    const pos = lbl.position ?? 0.5
    const idx = Math.min(Math.floor(pos * (allPts.length - 1)), allPts.length - 2)
    const t = (pos * (allPts.length - 1)) - idx
    const px = (allPts[idx].x * (1 - t) + allPts[idx + 1].x * t) + PADDING + offsetX
    const py = (allPts[idx].y * (1 - t) + allPts[idx + 1].y * t) + PADDING + offsetY
    const fs = lbl.fontSize || 10
    const bg = lbl.backgroundColor || '#FFFFFF'

    svg += `  <rect x="${px - lbl.text.length * fs * 0.3}" y="${py - fs * 0.7}" width="${lbl.text.length * fs * 0.6}" height="${fs * 1.4}" fill="${bg}" rx="3" opacity="0.92" />\n`
    svg += `  <text x="${px}" y="${py + fs * 0.15}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="${fs}" fill="${color}" font-weight="500">${escapeXml(lbl.text)}</text>\n`
  }

  return svg
}

// ── Smooth path builder (rounded corners at bends) ──────────────────

function buildSmoothPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return ''
  if (points.length === 2) return `M${points[0].x},${points[0].y} L${points[1].x},${points[1].y}`

  const radius = 6
  let d = `M${points[0].x},${points[0].y}`

  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1], curr = points[i], next = points[i + 1]
    const d1 = Math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)
    const d2 = Math.sqrt((next.x - curr.x) ** 2 + (next.y - curr.y) ** 2)
    const r = Math.min(radius, d1 / 2, d2 / 2)

    if (r < 1) { d += ` L${curr.x},${curr.y}`; continue }

    const t1x = curr.x - (r * (curr.x - prev.x)) / d1
    const t1y = curr.y - (r * (curr.y - prev.y)) / d1
    const t2x = curr.x + (r * (next.x - curr.x)) / d2
    const t2y = curr.y + (r * (next.y - curr.y)) / d2

    d += ` L${t1x},${t1y} Q${curr.x},${curr.y} ${t2x},${t2y}`
  }

  d += ` L${points[points.length - 1].x},${points[points.length - 1].y}`
  return d
}

// ── Utilities ───────────────────────────────────────────────────────

function escapeXml(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

function _fallbackSvg(message: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" width="400" height="200" style="max-width:100%;height:auto;">
  <rect width="400" height="200" fill="#FAFAFA" rx="8" stroke="#E0E0E0" stroke-width="1"/>
  <text x="200" y="90" text-anchor="middle" font-family="system-ui, sans-serif" font-size="14" fill="#78909C">\u26A0 SVG Generation Issue</text>
  <text x="200" y="115" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#B0BEC5">${escapeXml(message)}</text>
</svg>`
}

export default elkToSvgIcons
