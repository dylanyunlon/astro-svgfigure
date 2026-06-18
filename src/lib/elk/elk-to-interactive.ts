/**
 * elk-to-interactive.ts — Convert ELK layouted JSON ↔ InteractiveGraph
 *
 * Bidirectional: ELK output → interactive editor input, and back
 * GitHub: kieler/elkjs, xyflow/xyflow
 */

import type { InteractiveGraph, InteractiveNode, InteractiveEdge } from './interactive-svg'

const NODE_FILL_NEUTRAL = '#F8FAFC' // Will be overridden by interactive-svg theme
const DEFAULT_EDGE_COLOR = '#94A3B8'

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
 * T5: Defensive handling of zero coords, empty children, null sections
 */
export function elkToInteractive(layouted: ElkLayouted): InteractiveGraph {
  const nodes: InteractiveNode[] = []
  const edges: InteractiveEdge[] = []

  if (!layouted) {
    console.warn('[elkToInteractive] layouted is null/undefined')
    return { nodes: [], edges: [], width: 800, height: 600 }
  }

  // Flatten nodes (including nested children)
  if (Array.isArray(layouted.children) && layouted.children.length > 0) {
    layouted.children.forEach((child, i) => {
      if (child) flattenNode(child, i, nodes)
    })
  }

  // Convert edges — need node lookup map for center-to-center fallback
  const nodeMap = new Map<string, InteractiveNode>()
  for (const n of nodes) nodeMap.set(n.id, n)

  if (Array.isArray(layouted.edges)) {
    layouted.edges.forEach((edge, i) => {
      if (edge) {
        const ie = convertEdge(edge, i, nodeMap)
        if (ie) edges.push(ie)
      }
    })
  }

  // Also collect nested edges inside compound nodes (recursive)
  // Nested edge coordinates from ELK are relative to the parent compound node
  function collectNestedEdges(children: any[], ox: number = 0, oy: number = 0) {
    for (const child of children) {
      const childAbsX = (child.x || 0) + ox
      const childAbsY = (child.y || 0) + oy
      if (child && Array.isArray(child.edges)) {
        child.edges.forEach((edge: any, i: number) => {
          if (edge) {
            const ie = convertEdge(edge, edges.length + i, nodeMap, childAbsX, childAbsY)
            if (ie) edges.push(ie)
          }
        })
      }
      if (child && Array.isArray(child.children)) {
        collectNestedEdges(child.children, childAbsX, childAbsY)
      }
    }
  }
  if (Array.isArray(layouted.children)) {
    collectNestedEdges(layouted.children)
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
    fill: NODE_FILL_NEUTRAL,
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

function convertEdge(edge: ElkLayoutedEdge, index: number, nodeMap: Map<string, InteractiveNode>, offsetX: number = 0, offsetY: number = 0): InteractiveEdge | null {
  const sourceId = edge.sources?.[0]
  const targetId = edge.targets?.[0]
  if (!sourceId || !targetId) return null

  const points: { x: number; y: number }[] = []

  if (edge.sections && edge.sections.length > 0) {
    for (const sec of edge.sections) {
      if (sec.startPoint) points.push({ x: sec.startPoint.x + offsetX, y: sec.startPoint.y + offsetY })
      if (sec.bendPoints) {
        for (const bp of sec.bendPoints) {
          points.push({ x: bp.x + offsetX, y: bp.y + offsetY })
        }
      }
      if (sec.endPoint) points.push({ x: sec.endPoint.x + offsetX, y: sec.endPoint.y + offsetY })
    }
  }

  // T5: If no sections/points, compute center-to-center from node positions
  if (points.length < 2) {
    const src = nodeMap.get(sourceId)
    const tgt = nodeMap.get(targetId)
    if (src && tgt) {
      // Source: bottom center, Target: top center (standard top-to-bottom flow)
      points.length = 0
      points.push({ x: src.x + src.width / 2, y: src.y + src.height })
      points.push({ x: tgt.x + tgt.width / 2, y: tgt.y })
    } else {
      // Absolute fallback
      points.length = 0
      points.push({ x: 0, y: 0 })
      points.push({ x: 100, y: 100 })
    }
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
 *
 * Reconstructs the nested compound-node hierarchy from the flat
 * InteractiveGraph node list.  Group (compound) nodes become parents
 * whose `children` array holds the nodes that were spatially inside
 * them, mirroring the ELK `children` nesting that enables
 * `elk.hierarchyHandling: INCLUDE_CHILDREN`.
 */
export function interactiveToElk(graph: InteractiveGraph): any {
  // Build a parentId→children map.  InteractiveNodes carry no explicit
  // parentId, so we infer containment from spatial overlap: a non-group
  // node whose center falls inside a group node is treated as its child.
  // When multiple group nodes overlap, the smallest enclosing group wins.
  const groups = graph.nodes.filter((n) => n.isGroup)
  const nonGroups = graph.nodes.filter((n) => !n.isGroup)

  // Map every nodeId → its direct parent groupId (or null for root)
  const parentOf = new Map<string, string | null>()

  for (const node of nonGroups) {
    const cx = node.x + node.width / 2
    const cy = node.y + node.height / 2
    let bestGroup: InteractiveNode | null = null
    let bestArea = Infinity
    for (const g of groups) {
      if (
        cx >= g.x && cx <= g.x + g.width &&
        cy >= g.y && cy <= g.y + g.height
      ) {
        const area = g.width * g.height
        if (area < bestArea) {
          bestArea = area
          bestGroup = g
        }
      }
    }
    parentOf.set(node.id, bestGroup ? bestGroup.id : null)
  }

  // Groups can also be nested inside other groups
  for (const g of groups) {
    const cx = g.x + g.width / 2
    const cy = g.y + g.height / 2
    let bestParent: InteractiveNode | null = null
    let bestArea = Infinity
    for (const pg of groups) {
      if (pg.id === g.id) continue
      if (
        cx >= pg.x && cx <= pg.x + pg.width &&
        cy >= pg.y && cy <= pg.y + pg.height &&
        pg.width * pg.height > g.width * g.height // parent must be bigger
      ) {
        const area = pg.width * pg.height
        if (area < bestArea) {
          bestArea = area
          bestParent = pg
        }
      }
    }
    parentOf.set(g.id, bestParent ? bestParent.id : null)
  }

  // Collect children per parent
  const childrenOf = new Map<string | null, string[]>()
  for (const [nodeId, pid] of parentOf) {
    if (!childrenOf.has(pid)) childrenOf.set(pid, [])
    childrenOf.get(pid)!.push(nodeId)
  }

  const nodeById = new Map<string, InteractiveNode>()
  for (const n of graph.nodes) nodeById.set(n.id, n)

  function buildElkNode(nodeId: string): any {
    const n = nodeById.get(nodeId)!
    const elkNode: any = {
      id: n.id,
      width: n.width,
      height: n.height,
      labels: [{ text: n.label }],
    }

    const kids = childrenOf.get(nodeId)
    if (kids && kids.length > 0) {
      elkNode.children = kids.map(buildElkNode)
      elkNode.layoutOptions = {
        'elk.padding': '[top=30,left=10,bottom=10,right=10]',
        'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      }
    }
    return elkNode
  }

  // Root-level nodes are those with parentOf === null
  const rootIds = childrenOf.get(null) || []
  // Also include any nodes that didn't end up in parentOf (safety net)
  const allMapped = new Set(parentOf.keys())
  for (const n of graph.nodes) {
    if (!allMapped.has(n.id)) rootIds.push(n.id)
  }

  // Classify edges: if both endpoints share a common compound ancestor,
  // attach the edge to that ancestor; otherwise keep at root.
  function findAncestors(nodeId: string): string[] {
    const path: string[] = []
    let cur: string | null | undefined = parentOf.get(nodeId)
    while (cur) {
      path.push(cur)
      cur = parentOf.get(cur)
    }
    return path
  }

  const rootEdges: any[] = []
  const compoundEdges = new Map<string, any[]>()

  for (const e of graph.edges) {
    const srcAncestors = findAncestors(e.sourceId)
    const tgtAncestors = findAncestors(e.targetId)
    const tgtSet = new Set(tgtAncestors)

    // Find lowest common compound ancestor
    let lca: string | null = null
    // Check if source is directly a child of target's ancestor chain (or vice versa)
    if (tgtSet.has(parentOf.get(e.sourceId)!) && parentOf.get(e.sourceId) === parentOf.get(e.targetId)) {
      lca = parentOf.get(e.sourceId)!
    } else {
      for (const a of srcAncestors) {
        if (tgtSet.has(a)) { lca = a; break }
      }
    }

    const elkEdge: any = {
      id: e.id,
      sources: [e.sourceId],
      targets: [e.targetId],
      ...(e.label ? { labels: [{ text: e.label }] } : {}),
    }

    if (lca) {
      if (!compoundEdges.has(lca)) compoundEdges.set(lca, [])
      compoundEdges.get(lca)!.push(elkEdge)
    } else {
      rootEdges.push(elkEdge)
    }
  }

  // Rebuild tree, attaching compound edges
  function buildElkNodeWithEdges(nodeId: string): any {
    const base = buildElkNode(nodeId)
    const innerEdges = compoundEdges.get(nodeId)
    if (innerEdges && innerEdges.length > 0) {
      base.edges = innerEdges
    }
    return base
  }

  return {
    id: 'root',
    layoutOptions: {
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
    },
    children: rootIds.map(buildElkNodeWithEdges),
    edges: rootEdges,
  }
}

export default { elkToInteractive, interactiveToElk }