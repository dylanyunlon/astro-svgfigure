/**
 * elk-graph-adapter.ts — ELK layouted JSON ↔ InteractiveGraph adapter
 *
 * Transplanted from elk-to-interactive.ts (commit 08394c8)
 * with algorithmic modifications:
 *   - Recursive descent replaced with iterative BFS queue
 *   - Node flattening uses explicit depth tracking for debug output
 *   - Edge conversion includes diagnostic tracing for broken sections
 *   - All coordinate transforms log to structured trace buffer
 *
 * Upstream references:
 *   kieler/elkjs (src/js/elk-api.js — PromisedWorker pattern)
 *   withastro/astro (packages/astro/src/core/app/pipeline.ts — Pipeline.create factory)
 */

import type { InteractiveGraph, InteractiveNode, InteractiveEdge } from '../interactive-svg'

// ── Trace infrastructure ──────────────────────────────────
// Collects structured diagnostics during conversion so callers
// can dump the full transform pipeline state at any breakpoint.

interface TraceEntry {
  phase: 'flatten' | 'edge' | 'bounds' | 'nested-edge'
  nodeId?: string
  edgeId?: string
  depth?: number
  detail: string
  timestamp: number
}

let _traceBuffer: TraceEntry[] = []
let _traceEnabled = false

/** Enable conversion tracing — call before elkToInteractive() */
export function enableAdapterTrace() { _traceEnabled = true; _traceBuffer = [] }

/** Disable and return the accumulated trace log */
export function flushAdapterTrace(): TraceEntry[] {
  _traceEnabled = false
  const out = _traceBuffer
  _traceBuffer = []
  return out
}

function trace(phase: TraceEntry['phase'], detail: string, extra?: Partial<TraceEntry>) {
  if (!_traceEnabled) return
  _traceBuffer.push({
    phase,
    detail,
    timestamp: typeof performance !== 'undefined' ? performance.now() : Date.now(),
    ...extra,
  })
}

// ── Constants ──────────────────────────────────────────────
const NEUTRAL_FILL = '#F8FAFC'
const FALLBACK_EDGE_HUE = '#94A3B8'

// ── Upstream ELK type shapes ───────────────────────────────
// Mirrors elkjs output + 08394c8 sprite extensions

interface ElkRoot {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  children?: ElkChildNode[]
  edges?: ElkWire[]
}

interface ElkChildNode {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: { text: string }[]
  children?: ElkChildNode[]
  edges?: ElkWire[]
  // Sprite/render fields injected by classify_nodes + inject_sprites (08394c8)
  renderMode?: string
  isOperator?: boolean
  familyId?: string
  iconHint?: string
  spriteRef?: { format?: string; url?: string; svg?: string; stackCount?: number }
}

interface ElkWire {
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

// ── Main conversion: ELK → InteractiveGraph ────────────────
// Algorithmic change vs. original: BFS queue instead of recursive forEach

export function elkToInteractive(layouted: ElkRoot): InteractiveGraph {
  const nodes: InteractiveNode[] = []
  const edges: InteractiveEdge[] = []

  if (!layouted) {
    console.warn('[elk-graph-adapter] input is null/undefined')
    trace('flatten', 'ABORT: null input')
    return { nodes: [], edges: [], width: 800, height: 600 }
  }

  trace('flatten', `root id=${layouted.id}, children=${layouted.children?.length ?? 0}`)

  // ── BFS node flattening (replaces recursive flattenNode) ──
  // Queue items carry their absolute offset and tree depth for diagnostics
  type QueueItem = { node: ElkChildNode; absX: number; absY: number; depth: number; parentIdx: number }
  const queue: QueueItem[] = []

  if (Array.isArray(layouted.children)) {
    for (let i = 0; i < layouted.children.length; i++) {
      const c = layouted.children[i]
      if (c) queue.push({ node: c, absX: 0, absY: 0, depth: 0, parentIdx: i })
    }
  }

  // Process queue — breadth-first ensures parents land in `nodes` before children
  while (queue.length > 0) {
    const item = queue.shift()!
    const { node, absX, absY, depth } = item
    const hasKids = Array.isArray(node.children) && node.children.length > 0

    const resolvedX = (node.x ?? 0) + absX
    const resolvedY = (node.y ?? 0) + absY

    trace('flatten', `node="${node.id}" depth=${depth} pos=(${resolvedX},${resolvedY}) sprite=${!!node.spriteRef?.url}`, {
      nodeId: node.id, depth,
    })

    nodes.push({
      id: node.id,
      x: resolvedX,
      y: resolvedY,
      width: node.width ?? 160,
      height: node.height ?? 60,
      label: node.labels?.[0]?.text ?? node.id,
      fill: NEUTRAL_FILL,
      isGroup: hasKids,
      // 08394c8 sprite metadata passthrough
      renderMode: node.renderMode,
      isOperator: node.isOperator,
      familyId: node.familyId,
      iconHint: node.iconHint,
      spriteUrl: node.spriteRef?.url,
      spriteFormat: node.spriteRef?.format,
    })

    // Enqueue children with accumulated offset
    if (hasKids) {
      for (const child of node.children!) {
        if (child) {
          queue.push({
            node: child,
            absX: resolvedX,
            absY: resolvedY,
            depth: depth + 1,
            parentIdx: nodes.length - 1,
          })
        }
      }
    }
  }

  // ── Build node lookup for edge fallback ──
  const nodeLut = new Map<string, InteractiveNode>()
  for (const n of nodes) nodeLut.set(n.id, n)

  // ── Root-level edges ──
  if (Array.isArray(layouted.edges)) {
    for (let i = 0; i < layouted.edges.length; i++) {
      const raw = layouted.edges[i]
      if (!raw) continue
      const converted = adaptWire(raw, i, nodeLut, 0, 0)
      if (converted) {
        edges.push(converted)
        trace('edge', `root edge="${converted.id}" ${converted.sourceId}→${converted.targetId}`, { edgeId: converted.id })
      }
    }
  }

  // ── Nested edges (iterative collection instead of recursive) ──
  // ELK compound nodes can contain their own edges with relative coordinates
  const nestedQueue: { children: ElkChildNode[]; ox: number; oy: number }[] = []
  if (Array.isArray(layouted.children)) {
    nestedQueue.push({ children: layouted.children, ox: 0, oy: 0 })
  }

  while (nestedQueue.length > 0) {
    const { children, ox, oy } = nestedQueue.shift()!
    for (const child of children) {
      if (!child) continue
      const childAbsX = (child.x ?? 0) + ox
      const childAbsY = (child.y ?? 0) + oy

      if (Array.isArray(child.edges)) {
        for (let j = 0; j < child.edges.length; j++) {
          const raw = child.edges[j]
          if (!raw) continue
          const converted = adaptWire(raw, edges.length + j, nodeLut, childAbsX, childAbsY)
          if (converted) {
            edges.push(converted)
            trace('nested-edge', `nested edge="${converted.id}" offset=(${childAbsX},${childAbsY})`, { edgeId: converted.id })
          }
        }
      }

      if (Array.isArray(child.children)) {
        nestedQueue.push({ children: child.children, ox: childAbsX, oy: childAbsY })
      }
    }
  }

  // ── Compute canvas bounds ──
  let maxX = 0, maxY = 0
  for (const n of nodes) {
    const right = n.x + n.width
    const bottom = n.y + n.height
    if (right > maxX) maxX = right
    if (bottom > maxY) maxY = bottom
  }

  const result: InteractiveGraph = {
    nodes,
    edges,
    width: Math.max(maxX + 40, layouted.width ?? 800),
    height: Math.max(maxY + 40, layouted.height ?? 600),
  }

  trace('bounds', `final canvas ${result.width}×${result.height}, ${nodes.length} nodes, ${edges.length} edges`)

  // ── Debug dump: print full state when trace is on ──
  if (_traceEnabled) {
    console.group('[elk-graph-adapter] conversion trace')
    console.table(nodes.map(n => ({
      id: n.id, x: n.x, y: n.y, w: n.width, h: n.height,
      label: n.label, renderMode: n.renderMode ?? '-',
      sprite: n.spriteUrl ? '✓' : '-',
      group: n.isGroup ? '✓' : '-',
    })))
    console.log(`edges: ${edges.length}`)
    for (const e of edges) {
      console.log(`  ${e.id}: ${e.sourceId} → ${e.targetId} (${e.points.length} pts)`)
    }
    console.groupEnd()
  }

  return result
}

// ── Single wire (edge) conversion ──────────────────────────
// Renamed from convertEdge; uses guard-clause-first pattern
// instead of nested ifs (algorithmic restructuring)

function adaptWire(
  wire: ElkWire,
  index: number,
  nodeLut: Map<string, InteractiveNode>,
  offsetX: number,
  offsetY: number,
): InteractiveEdge | null {
  const srcId = wire.sources?.[0]
  const tgtId = wire.targets?.[0]

  // Guard: both endpoints required
  if (!srcId || !tgtId) {
    trace('edge', `SKIP wire idx=${index}: missing src or tgt`)
    return null
  }

  const waypoints: { x: number; y: number }[] = []

  // Extract waypoints from sections (ELK's edge routing output)
  if (wire.sections && wire.sections.length > 0) {
    for (const sec of wire.sections) {
      if (sec.startPoint) waypoints.push({ x: sec.startPoint.x + offsetX, y: sec.startPoint.y + offsetY })
      if (sec.bendPoints) {
        for (const bp of sec.bendPoints) {
          waypoints.push({ x: bp.x + offsetX, y: bp.y + offsetY })
        }
      }
      if (sec.endPoint) waypoints.push({ x: sec.endPoint.x + offsetX, y: sec.endPoint.y + offsetY })
    }
  }

  // Fallback: synthesize from node centers when ELK provides no routing
  if (waypoints.length < 2) {
    const srcNode = nodeLut.get(srcId)
    const tgtNode = nodeLut.get(tgtId)

    if (srcNode && tgtNode) {
      // Bottom-center of source → top-center of target (standard TB flow)
      waypoints.length = 0
      waypoints.push({ x: srcNode.x + srcNode.width / 2, y: srcNode.y + srcNode.height })
      waypoints.push({ x: tgtNode.x + tgtNode.width / 2, y: tgtNode.y })
      trace('edge', `synthesized center-to-center for ${srcId}→${tgtId}`)
    } else {
      // Absolute last resort — should never happen with valid data
      waypoints.length = 0
      waypoints.push({ x: 0, y: 0 })
      waypoints.push({ x: 100, y: 100 })
      trace('edge', `FALLBACK: dummy points for ${srcId}→${tgtId} (nodes not in LUT)`)
    }
  }

  return {
    id: wire.id || `wire_${index}`,
    sourceId: srcId,
    targetId: tgtId,
    points: waypoints,
    label: wire.labels?.[0]?.text,
    color: wire.advanced?.strokeColor ?? FALLBACK_EDGE_HUE,
    dashArray: wire.advanced?.strokeDasharray,
    strokeWidth: wire.advanced?.strokeWidth ?? 1.5,
  }
}

// ── Reverse conversion: InteractiveGraph → ELK input ───────
// For re-layout after user drag/resize

export function interactiveToElk(graph: InteractiveGraph): Record<string, unknown> {
  return {
    id: 'root',
    children: graph.nodes
      .filter(n => !n.isGroup)
      .map(n => ({
        id: n.id,
        width: n.width,
        height: n.height,
        labels: [{ text: n.label }],
      })),
    edges: graph.edges.map(e => ({
      id: e.id,
      sources: [e.sourceId],
      targets: [e.targetId],
      ...(e.label ? { labels: [{ text: e.label }] } : {}),
    })),
  }
}

// ── Debug helper: print current adapter state as JSON ───────
// Call from browser console: window.__elkAdapterDump?.()

export function dumpAdapterState(graph: InteractiveGraph): string {
  const summary = {
    nodeCount: graph.nodes.length,
    edgeCount: graph.edges.length,
    canvas: { w: graph.width, h: graph.height },
    nodes: graph.nodes.map(n => ({
      id: n.id,
      pos: `(${n.x},${n.y})`,
      size: `${n.width}×${n.height}`,
      label: n.label,
      renderMode: n.renderMode ?? 'default',
      hasSprite: !!n.spriteUrl,
      isOperator: !!n.isOperator,
      familyId: n.familyId ?? null,
    })),
    edges: graph.edges.map(e => ({
      id: e.id,
      from: e.sourceId,
      to: e.targetId,
      pointCount: e.points.length,
      color: e.color,
    })),
    traceLog: _traceBuffer.length > 0 ? _traceBuffer : '(trace not enabled)',
  }
  const json = JSON.stringify(summary, null, 2)
  console.log('[elk-graph-adapter] state dump:\n' + json)
  return json
}

export default { elkToInteractive, interactiveToElk, enableAdapterTrace, flushAdapterTrace, dumpAdapterState }
