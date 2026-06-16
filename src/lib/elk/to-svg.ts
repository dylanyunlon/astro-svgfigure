/**
 * to-svg.ts -- ELK Layouted JSON -> Structured Layout JSON (PixiJS) + Skeleton SVG
 * Enhanced with advanced edge routing: dashed, bidirectional, curved, labeled edges
 * GitHub: EmilStenstrom/elkjs-svg, kieler/elkjs
 *
 * M160: SVG string template output replaced by structured JSON for PixiJS consumption.
 * elkToSvg() retained for backward compatibility; prefer elkToLayoutJson() for new consumers.
 */

// ── ELK input types ──────────────────────────────────────────────────────────

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
  ports?: ElkPort[]
}

interface ElkPort {
  id: string; x?: number; y?: number; width?: number; height?: number
  properties?: { side?: string; type?: string }
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

// ── Structured JSON output types (consumed by pixi-cell-renderer.ts) ─────────

/** A single node flattened to absolute canvas coordinates, ready for PixiJS. */
export interface LayoutNode {
  /** Matches ElkNode.id */
  id: string
  /** Display label (first label text, or node id if absent) */
  label: string
  /** Absolute canvas position and dimensions */
  x: number
  y: number
  width: number
  height: number
  /** True when this node contains children (compound/group node) */
  isGroup: boolean
  /** Nesting depth (root children = 0) */
  depth: number
  /** Parent node id, or null for root children */
  parentId: string | null
  /** Port descriptors attached to this node */
  ports: LayoutPort[]
  /** Species hint inferred from label or id, for PixiJS species palette */
  species: string
  /** z-order hint (depth-based, higher depth = higher z) */
  z: number
}

/** A port descriptor in absolute canvas coordinates. */
export interface LayoutPort {
  id: string
  x: number
  y: number
  width: number
  height: number
  side: string
  type: string
}

/** A routed edge with absolute canvas waypoints, ready for PixiJS bezier drawing. */
export interface LayoutEdge {
  id: string
  source: string
  target: string
  /** Absolute canvas waypoints: [start, ...bendPoints, end] per section */
  sections: Array<{
    points: Array<{ x: number; y: number }>
  }>
  /** Visual style hints for PixiJS renderer */
  style: LayoutEdgeStyle
  /** Edge label descriptors */
  labels: Array<{ text: string; x: number; y: number; fontSize: number }>
}

export interface LayoutEdgeStyle {
  strokeColor: string
  strokeWidth: number
  strokeDasharray: string
  curvature: number
  directionality: 'unidirectional' | 'bidirectional'
  sourceArrow: boolean
  targetArrow: boolean
  semanticType: string
}

/** Canvas-level metadata for the full layout. */
export interface LayoutMeta {
  width: number
  height: number
  padding: number
  nodeCount: number
  edgeCount: number
}

/** The complete structured layout JSON output from elkToLayoutJson(). */
export interface ElkLayoutJson {
  meta: LayoutMeta
  nodes: LayoutNode[]
  edges: LayoutEdge[]
}

// ── Visual constants (kept in sync with legacy SVG renderer) ─────────────────

const DEFAULT_EDGE_COLOR = '#94A3B8'
const PADDING = 20

const SEMANTIC_STYLES: Record<string, Partial<AdvancedEdge>> = {
  data_flow:       { strokeColor: '#78909C', strokeWidth: 1.5 },
  gradient_flow:   { strokeColor: '#E57373', strokeDasharray: '8,4', strokeWidth: 1.5 },
  skip_connection: { strokeColor: '#4CAF50', strokeWidth: 2, curvature: 0.7 },
  optional_path:   { strokeColor: '#9E9E9E', strokeDasharray: '5,5', strokeWidth: 1 },
  inference_only:  { strokeColor: '#7986CB', strokeDasharray: '10,3,3,3', strokeWidth: 1.5 },
  fan_out:         { strokeColor: '#FF9800', strokeWidth: 1.5 },
  fan_in:          { strokeColor: '#2196F3', strokeWidth: 1.5 },
  feedback:        { strokeColor: '#AB47BC', strokeDasharray: '6,3', strokeWidth: 1.5 },
  attention:       { strokeColor: '#F44336', strokeDasharray: '2,4', strokeWidth: 2 },
  concatenation:   { strokeColor: '#009688', strokeWidth: 2 },
  residual:        { strokeColor: '#4CAF50', strokeWidth: 2, curvature: 0.6 },
  cross_boundary:  { strokeColor: '#607D8B', strokeWidth: 1.5 },
}

// Simple species inference from node id/label text
const SPECIES_KEYWORDS: Array<[RegExp, string]> = [
  [/conv|filter/i,        'cil-filter'],
  [/pool/i,               'cil-layers'],
  [/attention|attn/i,     'cil-eye'],
  [/activ|relu|gelu/i,    'cil-bolt'],
  [/add|sum|merge/i,      'cil-plus'],
  [/norm|bn|layer.?norm/i,'cil-code'],
  [/embed|input/i,        'cil-vector'],
  [/loop|recur|lstm|gru/i,'cil-loop'],
  [/graph|gnn/i,          'cil-graph'],
  [/output|head/i,        'cil-arrow-right'],
]

function inferSpecies(label: string, id: string): string {
  const text = `${label} ${id}`
  for (const [re, species] of SPECIES_KEYWORDS) {
    if (re.test(text)) return species
  }
  return 'cil-code'
}

// ── Graph bounds helper ───────────────────────────────────────────────────────

function computeGraphBounds(graph: ElkGraph): { gw: number; gh: number } {
  let gw = graph.width || 0
  let gh = graph.height || 0
  if (gw === 0 || gh === 0) {
    function walk(nodes: ElkNode[], ox: number, oy: number) {
      for (const child of nodes) {
        const right  = (child.x || 0) + ox + (child.width  || 160)
        const bottom = (child.y || 0) + oy + (child.height || 60)
        if (right  > gw) gw = right
        if (bottom > gh) gh = bottom
        if (Array.isArray(child.children) && child.children.length > 0)
          walk(child.children, (child.x || 0) + ox, (child.y || 0) + oy)
      }
    }
    walk(graph.children || [], 0, 0)
    gw = Math.max(gw, 200)
    gh = Math.max(gh, 150)
  }
  return { gw, gh }
}

// ── Node flattening ───────────────────────────────────────────────────────────

function flattenNodes(
  nodes: ElkNode[],
  parentAbsX: number,
  parentAbsY: number,
  depth: number,
  parentId: string | null,
  out: LayoutNode[],
): void {
  for (const node of nodes) {
    const absX = (node.x || 0) + parentAbsX + PADDING
    const absY = (node.y || 0) + parentAbsY + PADDING
    const w = node.width  || 160
    const h = node.height || 60
    const isGroup = Array.isArray(node.children) && node.children.length > 0
    const label = node.labels?.[0]?.text || node.id

    const ports: LayoutPort[] = (node.ports || []).map(p => ({
      id:     p.id,
      x:      absX + (p.x || 0),
      y:      absY + (p.y || 0),
      width:  p.width  || 6,
      height: p.height || 6,
      side:   p.properties?.side || 'EAST',
      type:   p.properties?.type || 'source',
    }))

    out.push({
      id:       node.id,
      label,
      x:        absX,
      y:        absY,
      width:    w,
      height:   h,
      isGroup,
      depth,
      parentId,
      ports,
      species:  inferSpecies(label, node.id),
      z:        depth,
    })

    if (isGroup) {
      // Children coords are relative to this node; subtract PADDING because
      // flattenNodes will add it again for children.
      flattenNodes(
        node.children!,
        absX - PADDING,
        absY - PADDING,
        depth + 1,
        node.id,
        out,
      )
    }
  }
}

// ── Edge flattening ───────────────────────────────────────────────────────────

function flattenEdge(
  edge: ElkEdge,
  offsetX: number,
  offsetY: number,
): LayoutEdge | null {
  if (!edge?.sections?.length) return null

  const adv = edge.advanced || {}
  const sem = adv.semanticType ? SEMANTIC_STYLES[adv.semanticType] || {} : {}

  const style: LayoutEdgeStyle = {
    strokeColor:    adv.strokeColor    || sem.strokeColor    || DEFAULT_EDGE_COLOR,
    strokeWidth:    adv.strokeWidth    || sem.strokeWidth    || 1.5,
    strokeDasharray:adv.strokeDasharray|| sem.strokeDasharray|| '',
    curvature:      adv.curvature      || (sem as any).curvature || 0,
    directionality: adv.directionality === 'bidirectional' ? 'bidirectional' : 'unidirectional',
    sourceArrow:    adv.directionality === 'bidirectional' || adv.sourceArrow === 'arrow',
    targetArrow:    adv.targetArrow !== 'none',
    semanticType:   adv.semanticType || '',
  }

  const sections = edge.sections.map(sec => {
    const pts: Array<{ x: number; y: number }> = []
    pts.push({ x: (sec.startPoint.x ?? 0) + PADDING + offsetX, y: (sec.startPoint.y ?? 0) + PADDING + offsetY })
    for (const bp of sec.bendPoints || []) {
      if (typeof bp.x === 'number' && typeof bp.y === 'number')
        pts.push({ x: bp.x + PADDING + offsetX, y: bp.y + PADDING + offsetY })
    }
    pts.push({ x: (sec.endPoint.x ?? 0) + PADDING + offsetX, y: (sec.endPoint.y ?? 0) + PADDING + offsetY })
    return { points: pts }
  })

  // Label positions: interpolate along first section midpoint
  const labels: LayoutEdge['labels'] = []
  const rawLabels = [...(adv.edgeLabels || []), ...(edge.labels || []).map(l => ({ text: l.text, position: 0.5, fontSize: 10 }))]
  const sec0 = edge.sections[0]
  if (sec0) {
    const allPts = [sec0.startPoint, ...(sec0.bendPoints || []), sec0.endPoint]
    for (const lbl of rawLabels) {
      if (!lbl.text) continue
      const pos = (lbl as any).position ?? 0.5
      const idx = Math.min(Math.floor(pos * (allPts.length - 1)), allPts.length - 2)
      const t   = pos * (allPts.length - 1) - idx
      labels.push({
        text:     lbl.text,
        x:        allPts[idx].x * (1 - t) + allPts[idx + 1].x * t + PADDING + offsetX,
        y:        allPts[idx].y * (1 - t) + allPts[idx + 1].y * t + PADDING + offsetY,
        fontSize: (lbl as any).fontSize || 10,
      })
    }
  }

  return {
    id:      edge.id,
    source:  edge.sources?.[0] || '',
    target:  edge.targets?.[0] || '',
    sections,
    style,
    labels,
  }
}

function flattenEdges(
  edgeList: ElkEdge[] | undefined,
  offsetX: number,
  offsetY: number,
  out: LayoutEdge[],
): void {
  if (!Array.isArray(edgeList)) return
  for (const e of edgeList) {
    const le = flattenEdge(e, offsetX, offsetY)
    if (le) out.push(le)
  }
}

function collectNestedEdges(
  nodes: ElkNode[],
  parentAbsX: number,
  parentAbsY: number,
  out: LayoutEdge[],
): void {
  for (const node of nodes) {
    const absX = (node.x || 0) + parentAbsX
    const absY = (node.y || 0) + parentAbsY
    flattenEdges((node as any).edges, absX, absY, out)
    if (Array.isArray(node.children))
      collectNestedEdges(node.children, absX, absY, out)
  }
}

// ── Public: elkToLayoutJson ───────────────────────────────────────────────────

/**
 * elkToLayoutJson — convert an ELK-layouted graph to structured JSON.
 *
 * Output is consumed by pixi-cell-renderer.ts (and any future PixiJS renderer).
 * All coordinates are absolute canvas coordinates with PADDING applied.
 *
 * @param graph  ELK layouted graph (output of elkjs layout pass)
 * @returns      ElkLayoutJson — { meta, nodes, edges }
 */
export function elkToLayoutJson(graph: ElkGraph): ElkLayoutJson {
  const { gw, gh } = computeGraphBounds(graph)
  const canvasW = gw + PADDING * 2
  const canvasH = gh + PADDING * 2

  const nodes: LayoutNode[] = []
  const edges: LayoutEdge[] = []

  if (Array.isArray(graph.children)) {
    flattenNodes(graph.children, 0, 0, 0, null, nodes)
  }

  // Root-level edges
  flattenEdges(graph.edges, 0, 0, edges)

  // Nested edges inside compound nodes (coordinates relative to parent)
  if (Array.isArray(graph.children)) {
    collectNestedEdges(graph.children, 0, 0, edges)
  }

  return {
    meta: {
      width:     canvasW,
      height:    canvasH,
      padding:   PADDING,
      nodeCount: nodes.length,
      edgeCount: edges.length,
    },
    nodes,
    edges,
  }
}

// ── Legacy: elkToSvg (retained for backward compatibility) ───────────────────

// T4: Clean neutral colors — no rainbow, ReactFlow-inspired
const NODE_FILL       = '#F8FAFC'
const NODE_FILL_GROUP = '#F8FAFC80'
const STROKE_COLOR    = '#E2E8F0'
const TEXT_COLOR      = '#1E293B'
const ARROW_SIZE      = 8

export function elkToSvg(graph: ElkGraph): string {
  if (!graph) {
    console.warn('[elkToSvg] graph is null/undefined')
    return _fallbackSvg('No graph data')
  }
  const hasChildren = Array.isArray(graph.children) && graph.children.length > 0
  const hasEdges    = Array.isArray(graph.edges)    && graph.edges.length > 0
  if (!hasChildren && !hasEdges) {
    console.warn('[elkToSvg] graph has no children and no edges')
    return _fallbackSvg('Empty graph (no nodes)')
  }

  const { gw, gh } = computeGraphBounds(graph)
  const width  = gw + PADDING * 2
  const height = gh + PADDING * 2
  const parts: string[] = []

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="max-width:100%;height:auto;">`)
  parts.push(`  <defs>`)
  parts.push(`    <marker id="ah-default" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE/3}" orient="auto" markerUnits="strokeWidth">`)
  parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE/3}, 0 ${ARROW_SIZE/1.5}" fill="${DEFAULT_EDGE_COLOR}" />`)
  parts.push(`    </marker>`)

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
  parts.push(`  <rect width="${width}" height="${height}" fill="#FAFAFA" rx="4" />`)

  if (Array.isArray(graph.edges)) {
    for (const edge of graph.edges) {
      if (edge) { const r = renderEdge(edge); if (r) parts.push(r) }
    }
  }

  function collectNestedEdgesSvg(nodes: ElkNode[], ox: number, oy: number) {
    for (const node of nodes) {
      const nodeAbsX = (node.x || 0) + ox
      const nodeAbsY = (node.y || 0) + oy
      if (node && (node as any).edges && Array.isArray((node as any).edges)) {
        for (const edge of (node as any).edges as ElkEdge[]) {
          if (edge) { const r = renderEdge(edge, nodeAbsX, nodeAbsY); if (r) parts.push(r) }
        }
      }
      if (node && node.children && Array.isArray(node.children))
        collectNestedEdgesSvg(node.children, nodeAbsX, nodeAbsY)
    }
  }
  if (Array.isArray(graph.children)) collectNestedEdgesSvg(graph.children, 0, 0)

  if (Array.isArray(graph.children)) {
    graph.children.slice(0, 100).forEach((node, i) => {
      if (node) parts.push(renderNode(node, i, 0))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

function renderNode(node: ElkNode, index: number, depth: number = 0, offsetX: number = 0, offsetY: number = 0): string {
  const x = (node.x || 0) + PADDING + offsetX, y = (node.y || 0) + PADDING + offsetY
  const w = node.width || 160, h = node.height || 60
  const isGroup = Array.isArray(node.children) && node.children.length > 0
  const fill = isGroup ? NODE_FILL_GROUP : NODE_FILL
  const strokeW = isGroup ? 2 : 1.5
  const strokeDash = isGroup ? ' stroke-dasharray="6,3"' : ''
  const label = node.labels?.[0]?.text || node.id
  const maxChars = Math.max(6, Math.floor(w / 8))
  const dl = label.length > maxChars ? label.slice(0, maxChars - 2) + '\u2026' : label

  let svg = `  <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${STROKE_COLOR}" stroke-width="${strokeW}" rx="8"${strokeDash} />`
  const labelY = isGroup ? y + 16 : y + h / 2
  const fontSize = isGroup ? 11 : 12
  const fontWeight = isGroup ? '600' : '500'
  svg += `  <text x="${x+w/2}" y="${labelY}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="${fontWeight}">${escapeXml(dl)}</text>`

  if (Array.isArray(node.ports) && node.ports.length > 0) {
    for (const port of node.ports) {
      const px = x + (port.x || 0)
      const py = y + (port.y || 0)
      const portType = port.properties?.type || 'source'
      const portSide = port.properties?.side || 'EAST'
      const portFill = portType === 'source' ? '#4F46E5' : DEFAULT_EDGE_COLOR
      svg += `  <circle class="port-indicator" data-port-id="${escapeXml(port.id)}" data-port-type="${portType}" data-port-side="${portSide}" cx="${px}" cy="${py}" r="4" fill="${portFill}" stroke="white" stroke-width="1.5" />`
    }
  }

  if (node.children) node.children.forEach((c, i) => { svg += renderNode(c, index*10+i, depth+1, x - PADDING, y - PADDING) })
  return svg
}

function renderEdge(edge: ElkEdge, offsetX: number = 0, offsetY: number = 0): string {
  if (!edge?.sections?.length) return ''

  const adv = edge.advanced || {}
  const sem = adv.semanticType ? SEMANTIC_STYLES[adv.semanticType] || {} : {}
  const color = adv.strokeColor || sem.strokeColor || DEFAULT_EDGE_COLOR
  const sw    = adv.strokeWidth    || sem.strokeWidth    || 1.5
  const da    = adv.strokeDasharray|| sem.strokeDasharray|| ''
  const curv  = adv.curvature      || (sem as any).curvature || 0
  const isBidir   = adv.directionality === 'bidirectional'
  const mid       = `ah-${color.replace('#', '')}`
  const hasSrcArrow = isBidir || adv.sourceArrow === 'arrow'
  const noTgtArrow  = adv.targetArrow === 'none'

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
      const dx = pts[1].x-pts[0].x,     dy = pts[1].y-pts[0].y
      const cx = mx - dy*curv*0.5,      cy = my + dx*curv*0.5
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

  const labels = adv.edgeLabels || []
  if (edge.labels?.length && !labels.length)
    labels.push(...edge.labels.map(l => ({ text: l.text, position: 0.5, fontSize: 10 })))

  for (const lbl of labels) {
    if (!lbl.text || !edge.sections?.[0]) continue
    const sec    = edge.sections[0]
    const allPts = [sec.startPoint, ...(sec.bendPoints||[]), sec.endPoint]
    const pos    = (lbl as any).position ?? 0.5
    const idx    = Math.min(Math.floor(pos*(allPts.length-1)), allPts.length-2)
    const t      = (pos*(allPts.length-1)) - idx
    const px     = (allPts[idx].x*(1-t) + allPts[idx+1].x*t) + PADDING + offsetX
    const py     = (allPts[idx].y*(1-t) + allPts[idx+1].y*t) + PADDING + offsetY
    const fs     = (lbl as any).fontSize || 10
    const bg     = (lbl as any).backgroundColor || '#FFFFFF'
    svg += `  <rect x="${px-lbl.text.length*fs*0.3}" y="${py-fs*0.7}" width="${lbl.text.length*fs*0.6}" height="${fs*1.4}" fill="${bg}" rx="2" opacity="0.9" />\n`
    svg += `  <text x="${px}" y="${py+fs*0.15}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="${fs}" fill="${color}">${escapeXml(lbl.text)}</text>\n`
  }

  return svg
}

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

export default elkToLayoutJson
