/**
 * to-svg.ts — ELK Layouted JSON → Skeleton SVG
 *
 * Renders ELK layout result as a simple skeleton SVG with
 * rectangles for nodes, arrows for edges, and text labels.
 * Used for Step 2 visualization before NanoBanana beautification.
 *
 * GitHub 背书: EmilStenstrom/elkjs-svg, kieler/elkjs
 */

interface ElkNode {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: { text: string }[]
  children?: ElkNode[]
}

interface ElkEdge {
  id: string
  sources?: string[]
  targets?: string[]
  sections?: {
    startPoint: { x: number; y: number }
    endPoint: { x: number; y: number }
    bendPoints?: { x: number; y: number }[]
  }[]
}

interface ElkGraph {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  children?: ElkNode[]
  edges?: ElkEdge[]
}

// Color palette for nodes (academic style)
const NODE_COLORS = [
  '#E3F2FD', // light blue
  '#F3E5F5', // light purple
  '#E8F5E9', // light green
  '#FFF3E0', // light orange
  '#FCE4EC', // light pink
  '#E0F7FA', // light cyan
  '#FFF8E1', // light amber
  '#F1F8E9', // light lime
]

const STROKE_COLOR = '#37474F'
const TEXT_COLOR = '#263238'
const EDGE_COLOR = '#78909C'
const ARROW_SIZE = 8
const PADDING = 20

/**
 * Convert an ELK layouted graph to a skeleton SVG string.
 */
export function elkToSvg(graph: ElkGraph): string {
  const width = (graph.width || 800) + PADDING * 2
  const height = (graph.height || 600) + PADDING * 2

  const parts: string[] = []

  // SVG header
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">`
  )

  // Defs for arrow markers
  parts.push(`
  <defs>
    <marker id="arrowhead" markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE / 1.5}" 
            refX="${ARROW_SIZE}" refY="${ARROW_SIZE / 3}" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, ${ARROW_SIZE} ${ARROW_SIZE / 3}, 0 ${ARROW_SIZE / 1.5}" fill="${EDGE_COLOR}" />
    </marker>
  </defs>`)

  // Background
  parts.push(`  <rect width="${width}" height="${height}" fill="#FAFAFA" rx="4" />`)

  // Render edges first (behind nodes)
  if (graph.edges) {
    for (const edge of graph.edges) {
      parts.push(renderEdge(edge))
    }
  }

  // Render nodes
  if (graph.children) {
    graph.children.forEach((node, i) => {
      parts.push(renderNode(node, i))
    })
  }

  parts.push('</svg>')
  return parts.join('\n')
}

function renderNode(node: ElkNode, index: number): string {
  const x = (node.x || 0) + PADDING
  const y = (node.y || 0) + PADDING
  const w = node.width || 160
  const h = node.height || 60
  const fill = NODE_COLORS[index % NODE_COLORS.length]
  const label = node.labels?.[0]?.text || node.id
  const rx = 8

  let svg = ''

  // Node rectangle with rounded corners
  svg += `  <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${STROKE_COLOR}" stroke-width="1.5" rx="${rx}" />`

  // Label text (centered)
  const textX = x + w / 2
  const textY = y + h / 2
  // Truncate long labels
  const displayLabel = label.length > 20 ? label.slice(0, 18) + '…' : label
  svg += `  <text x="${textX}" y="${textY}" text-anchor="middle" dominant-baseline="central" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${TEXT_COLOR}" font-weight="500">${escapeXml(displayLabel)}</text>`

  // Render nested children recursively
  if (node.children) {
    node.children.forEach((child, i) => {
      svg += renderNode(child, index * 10 + i)
    })
  }

  return svg
}

function renderEdge(edge: ElkEdge): string {
  if (!edge.sections || edge.sections.length === 0) {
    return ''
  }

  let svg = ''
  for (const section of edge.sections) {
    const points: { x: number; y: number }[] = []
    points.push({
      x: section.startPoint.x + PADDING,
      y: section.startPoint.y + PADDING,
    })
    if (section.bendPoints) {
      for (const bp of section.bendPoints) {
        points.push({ x: bp.x + PADDING, y: bp.y + PADDING })
      }
    }
    points.push({
      x: section.endPoint.x + PADDING,
      y: section.endPoint.y + PADDING,
    })

    if (points.length === 2) {
      // Simple line
      svg += `  <line x1="${points[0].x}" y1="${points[0].y}" x2="${points[1].x}" y2="${points[1].y}" stroke="${EDGE_COLOR}" stroke-width="1.5" marker-end="url(#arrowhead)" />`
    } else {
      // Polyline for bent edges
      const d = points.map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ')
      svg += `  <path d="${d}" fill="none" stroke="${EDGE_COLOR}" stroke-width="1.5" marker-end="url(#arrowhead)" />`
    }
  }

  return svg
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

export default elkToSvg
