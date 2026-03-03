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

const NODE_COLORS = ['#E3F2FD','#F3E5F5','#E8F5E9','#FFF3E0','#FCE4EC','#E0F7FA','#FFF8E1','#F1F8E9']
const STROKE_COLOR = '#37474F'
const TEXT_COLOR = '#263238'
const DEFAULT_EDGE_COLOR = '#78909C'
const ARROW_SIZE = 8
const PADDING = 20

const SEMANTIC_STYLES: Record<string, Partial<AdvancedEdge>> = {
  data_flow: { strokeColor: '#78909C', strokeWidth: 1.5 },
  gradient_flow: { strokeColor: '#E57373', strokeDasharray: '8,4', strokeWidth: 1.5 },
  skip_connection: { strokeColor: '#4CAF50', strokeWidth: 2, curvature: 0.7 },
  optional_path: { strokeColor: '#9E9E9E', strokeDasharray: '5,5', strokeWidth: 1 },
  inference_only: { strokeColor: '#7986CB', strokeDasharray: '10,3,3,3', strokeWidth: 1.5 },
  fan_out: { strokeColor: '#FF9800', strokeWidth: 1.5 },
  fan_in: { strokeColor: '#2196F3', strokeWidth: 1.5 },
  feedback: { strokeColor: '#AB47BC', strokeDasharray: '6,3', strokeWidth: 1.5 },
  attention: { strokeColor: '#F44336', strokeDasharray: '2,4', strokeWidth: 2 },
  concatenation: { strokeColor: '#009688', strokeWidth: 2 },
  residual: { strokeColor: '#4CAF50', strokeWidth: 2, curvature: 0.6 },
  cross_boundary: { strokeColor: '#607D8B', strokeWidth: 1.5 },
}

export function elkToSvg(graph: ElkGraph): string {
  if (!graph) return ''
  const width = (graph.width || 800) + PADDING * 2
  const height = (graph.height || 600) + PADDING * 2
  const parts: string[] = []

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">`)
  parts.push(`  <defs>`)
  parts.push(`    <marker id="ah-default" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE/1.5}" refX="${ARROW_SIZE}" refY="${ARROW_SIZE/3}" orient="auto" markerUnits="strokeWidth">`)
  parts.push(`      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE/3}, 0 ${ARROW_SIZE/1.5}" fill="${DEFAULT_EDGE_COLOR}" />`)
  parts.push(`    </marker>`)

  // Generate colored markers for each edge color
  const markerColors = new Set<string>()
  if (Array.isArray(graph.edges)) {
    for (const e of graph.edges) {
      if (e?.advanced?.strokeColor) markerColors.add(e.advanced.strokeColor)
      if (e?.advanced?.semanticType && SEMANTIC_STYLES[e.advanced.semanticType]?.strokeColor)
        markerColors.add(SEMANTIC_STYLES[e.advanced.semanticType].strokeColor!)
    }
  }
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

  if (Array.isArray(graph.children)) {
    graph.children.slice(0, 100).forEach((node, i) => {
      if (node) parts.push(renderNode(node, i))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

function renderNode(node: ElkNode, index: number): string {
  const x = (node.x || 0) + PADDING, y = (node.y || 0) + PADDING
  const w = node.width || 160, h = node.height || 60
  const fill = NODE_COLORS[index % NODE_COLORS.length]
  const label = node.labels?.[0]?.text || node.id
  const dl = label.length > 20 ? label.slice(0, 18) + '\u2026' : label

  let svg = `  <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${STROKE_COLOR}" stroke-width="1.5" rx="8" />`
  svg += `  <text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${TEXT_COLOR}" font-weight="500">${escapeXml(dl)}</text>`

  if (node.children) node.children.forEach((c, i) => { svg += renderNode(c, index*10+i) })
  return svg
}

function renderEdge(edge: ElkEdge): string {
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
    pts.push({ x: (section.startPoint.x??0)+PADDING, y: (section.startPoint.y??0)+PADDING })
    if (Array.isArray(section.bendPoints)) {
      for (const bp of section.bendPoints) {
        if (bp && typeof bp.x==='number' && typeof bp.y==='number')
          pts.push({ x: bp.x+PADDING, y: bp.y+PADDING })
      }
    }
    pts.push({ x: (section.endPoint.x??0)+PADDING, y: (section.endPoint.y??0)+PADDING })

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
    const px = (allPts[idx].x*(1-t) + allPts[idx+1].x*t) + PADDING
    const py = (allPts[idx].y*(1-t) + allPts[idx+1].y*t) + PADDING
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

export default elkToSvg
