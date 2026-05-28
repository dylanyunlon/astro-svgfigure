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
  spriteRef?: { format: 'png' | 'svg'; url?: string; svg?: string; bbox?: [number, number, number, number]; fit?: 'contain' }
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

export function elkToSvg(graph: ElkGraph): string {
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
  const nodeType = isGroup ? 'group' : 'leaf'

  // Smart label truncation based on node width
  const maxChars = Math.max(6, Math.floor(w / 8))
  const dl = label.length > maxChars ? label.slice(0, maxChars - 2) + '\u2026' : label

  // Wrap each node in a <g> with semantic attributes for SVG-native layer separation.
  // svg-layer-separator.ts reads data-node-id, data-node-type, and data-bbox
  // to extract layers without rasterization.
  let svg = `  <g data-node-id="${escapeXml(node.id)}" data-node-type="${nodeType}" data-depth="${depth}" data-bbox="${x},${y},${w},${h}" data-render-mode="${node.renderMode || 'text'}">`

  // ── Sprite mode (M214): an AI-generated micro-illustration injected by the
  // backend. Render only when a valid spriteRef is present; otherwise fall
  // through to the normal text/box path (graceful fallback — never an empty
  // box). Layout is fixed by ELK; the sprite is contain-fit + centered into
  // the node's bbox, so arrows (snapped to the box edge by snapEndpointToBox,
  // M204) stay aligned regardless of the sprite's own alpha bounds.
  const spriteSvg = (!isGroup && node.renderMode === 'sprite' && node.spriteRef)
    ? renderSprite(node, x, y, w, h)
    : ''
  if (spriteSvg) {
    svg += spriteSvg
    if (node.children) node.children.forEach((c, i) => { svg += renderNode(c, index*10+i, depth+1, x - PADDING, y - PADDING) })
    svg += `</g>`
    return svg
  }

  const isLabelOnly = !!(node as any).labelOnly
    || (h <= 30 && !(node as any).iconHint)

  if (isLabelOnly) {
    // Label-only node: naked text, no rect/box/fill.
    const fontSize = h > 30 ? 13 : 11
    svg += `<text x="${x+w/2}" y="${y+h/2}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="600">${escapeXml(dl)}</text>`
  } else {
    // ── Organic scattered rounded-rect style ──
    // Seeded PRNG from node id for deterministic randomness
    let seed = 0
    for (let i = 0; i < node.id.length; i++) seed = ((seed << 5) - seed + node.id.charCodeAt(i)) | 0
    const seededRand = () => { seed = (seed * 16807 + 0) % 2147483647; return (seed & 0x7fffffff) / 2147483647 }

    const mainRx = Math.min(12, Math.min(w, h) * 0.15)
    svg += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${STROKE_COLOR}" stroke-width="${isGroup ? 1.5 : 0.8}" rx="${mainRx}"${strokeDash} />`

    // Scatter 3-6 small decorative rounded rects
    if (!isGroup) {
      const numScatter = 3 + Math.floor(seededRand() * 4)
      const pad = 4
      for (let si = 0; si < numScatter; si++) {
        const sw2 = w * (0.15 + seededRand() * 0.35)
        const sh2 = h * (0.12 + seededRand() * 0.25)
        const sx = x + pad + seededRand() * Math.max(0, w - sw2 - pad * 2)
        const sy = y + pad + seededRand() * Math.max(0, h - sh2 - pad * 2)
        const srx = 3 + seededRand() * 8
        const sOpacity = 0.04 + seededRand() * 0.08
        const sRotate = (seededRand() - 0.5) * 6
        svg += `<rect x="${sx.toFixed(1)}" y="${sy.toFixed(1)}" width="${sw2.toFixed(1)}" height="${sh2.toFixed(1)}" rx="${srx.toFixed(1)}" fill="${STROKE_COLOR}" opacity="${sOpacity.toFixed(3)}" transform="rotate(${sRotate.toFixed(1)} ${(sx+sw2/2).toFixed(1)} ${(sy+sh2/2).toFixed(1)})" />`
      }
    }

    // For group nodes, put label at top — larger, bolder
    const hasIcon = !!(node as any).iconHint
    const labelY = isGroup ? y + 20 : y + h / 2
    // Font size: smaller when node has icon (icon takes visual priority)
    const fontSize = isGroup ? 14 : (hasIcon && h >= 50) ? 9 : 12
    const fontWeight = isGroup ? '600' : '500'
    svg += `<text x="${x+w/2}" y="${labelY}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="${fontSize}" fill="${TEXT_COLOR}" font-weight="${fontWeight}">${escapeXml(dl)}</text>`
  }

  // Nested children: pass parent's absolute position as offset (minus PADDING since children add it again)
  if (node.children) node.children.forEach((c, i) => { svg += renderNode(c, index*10+i, depth+1, x - PADDING, y - PADDING) })

  svg += `</g>`
  return svg
}

function renderSprite(node: ElkNode, x: number, y: number, w: number, h: number): string {
  const ref = node.spriteRef
  if (!ref) return ''
  // Inner padding so the sprite never touches the node's logical edge — keeps
  // a small margin like a real figure element inside its slot.
  const pad = Math.min(6, Math.min(w, h) * 0.08)
  const boxW = Math.max(1, w - pad * 2)
  const boxH = Math.max(1, h - pad * 2)

  // Native sprite dimensions from its alpha bbox (fallback to the slot box).
  const natW = ref.bbox?.[2] || boxW
  const natH = ref.bbox?.[3] || boxH
  // contain-fit: scale to fit inside boxW×boxH preserving aspect ratio.
  const scale = Math.min(boxW / natW, boxH / natH)
  const drawW = natW * scale
  const drawH = natH * scale
  // center within the slot
  const dx = x + pad + (boxW - drawW) / 2
  const dy = y + pad + (boxH - drawH) / 2

  if (ref.format === 'svg' && ref.svg) {
    // Vectorized sprite (M217): inline markup, translated + scaled into place.
    return `<g transform="translate(${dx.toFixed(2)} ${dy.toFixed(2)}) scale(${scale.toFixed(4)})" data-sprite="svg">${ref.svg}</g>`
  }
  if (ref.url) {
    // Raster sprite: <image> via data URI. preserveAspectRatio is redundant
    // with our explicit fit but set defensively.
    return `<image href="${ref.url}" x="${dx.toFixed(2)}" y="${dy.toFixed(2)}" width="${drawW.toFixed(2)}" height="${drawH.toFixed(2)}" preserveAspectRatio="xMidYMid meet" data-sprite="png" />`
  }
  return ''
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