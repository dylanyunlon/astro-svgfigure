/**
 * elk-to-interactive.ts — Convert ELK layouted JSON ↔ InteractiveGraph
 *
 * Bidirectional: ELK output → interactive editor input, and back
 * GitHub: kieler/elkjs, xyflow/xyflow
 */

import type { InteractiveGraph, InteractiveNode, InteractiveEdge } from './interactive-svg'

const NODE_COLORS = ['#E3F2FD', '#F3E5F5', '#E8F5E9', '#FFF3E0', '#FCE4EC', '#E0F7FA', '#FFF8E1', '#F1F8E9']
const DEFAULT_EDGE_COLOR = '#78909C'

interface ElkLayouted {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  children?: ElkLayoutedNode[]
  edges?: ElkLayoutedEdge[]
}

interface ElkLayoutedNode {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: { text: string }[]
  children?: ElkLayoutedNode[]
}

interface ElkLayoutedEdge {
  id: string
  sources?: string[]
  targets?: string[]
  sections?: {
    startPoint: { x: number; y: number }
    endPoint: { x: number; y: number }
    bendPoints?: { x: number; y: number }[]
  }[]
  labels?: { text: string }[]
  advanced?: {
    strokeColor?: string
    strokeWidth?: number
    strokeDasharray?: string
    semanticType?: string
  }
}

/**
 * Convert ELK layouted JSON → InteractiveGraph
 */
export function elkToInteractive(layouted: ElkLayouted): InteractiveGraph {
  const nodes: InteractiveNode[] = []
  const edges: InteractiveEdge[] = []

  // Flatten nodes (including nested children)
  if (Array.isArray(layouted.children)) {
    layouted.children.forEach((child, i) => {
      flattenNode(child, i, nodes)
    })
  }

  // Convert edges
  if (Array.isArray(layouted.edges)) {
    layouted.edges.forEach((edge, i) => {
      const ie = convertEdge(edge, i)
      if (ie) edges.push(ie)
    })
  }

  // Calculate bounds
  let maxX = 0, maxY = 0
  for (const n of nodes) {
    const r = n.x + n.width
    const b = n.y + n.height
    if (r > maxX) maxX = r
    if (b > maxY) maxY = b
  }

  return {
    nodes,
    edges,
    width: Math.max(maxX + 40, layouted.width || 800),
    height: Math.max(maxY + 40, layouted.height || 600),
  }
}

function flattenNode(node: ElkLayoutedNode, index: number, out: InteractiveNode[]) {
  const hasChildren = Array.isArray(node.children) && node.children.length > 0

  out.push({
    id: node.id,
    x: node.x || 0,
    y: node.y || 0,
    width: node.width || 160,
    height: node.height || 60,
    label: node.labels?.[0]?.text || node.id,
    fill: NODE_COLORS[index % NODE_COLORS.length],
    isGroup: hasChildren,
  })

  if (hasChildren) {
    node.children!.forEach((child, ci) => {
      // Offset child coords by parent position
      const adjustedChild = {
        ...child,
        x: (child.x || 0) + (node.x || 0),
        y: (child.y || 0) + (node.y || 0),
      }
      flattenNode(adjustedChild, index * 10 + ci, out)
    })
  }
}

function convertEdge(edge: ElkLayoutedEdge, index: number): InteractiveEdge | null {
  const sourceId = edge.sources?.[0]
  const targetId = edge.targets?.[0]
  if (!sourceId || !targetId) return null

  const points: { x: number; y: number }[] = []

  if (edge.sections && edge.sections.length > 0) {
    for (const sec of edge.sections) {
      if (sec.startPoint) points.push({ x: sec.startPoint.x, y: sec.startPoint.y })
      if (sec.bendPoints) {
        for (const bp of sec.bendPoints) {
          points.push({ x: bp.x, y: bp.y })
        }
      }
      if (sec.endPoint) points.push({ x: sec.endPoint.x, y: sec.endPoint.y })
    }
  }

  // Fallback: if no sections, create simple straight line
  if (points.length < 2) {
    points.push({ x: 0, y: 0 })
    points.push({ x: 100, y: 100 })
  }

  return {
    id: edge.id || `edge_${index}`,
    sourceId,
    targetId,
    points,
    label: edge.labels?.[0]?.text,
    color: edge.advanced?.strokeColor || DEFAULT_EDGE_COLOR,
    dashArray: edge.advanced?.strokeDasharray,
    strokeWidth: edge.advanced?.strokeWidth || 1.5,
  }
}

/**
 * Convert InteractiveGraph → ELK input JSON (for re-layout)
 */
export function interactiveToElk(graph: InteractiveGraph): any {
  return {
    id: 'root',
    children: graph.nodes
      .filter((n) => !n.isGroup) // Only top-level non-group nodes for re-layout
      .map((n) => ({
        id: n.id,
        width: n.width,
        height: n.height,
        labels: [{ text: n.label }],
      })),
    edges: graph.edges.map((e) => ({
      id: e.id,
      sources: [e.sourceId],
      targets: [e.targetId],
      ...(e.label ? { labels: [{ text: e.label }] } : {}),
    })),
  }
}

export default { elkToInteractive, interactiveToElk }
