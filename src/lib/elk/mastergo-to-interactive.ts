/**
 * mastergo-to-interactive.ts — Mastergo Layout → InteractiveGraph
 *
 * Converts pipeline detection output [{id, name, bbox:{x,y,w,h}}]
 * to the InteractiveGraph format consumed by InteractiveSvgEditor.
 * Enables drag-to-adjust on detected/ELK-derived layout.
 *
 * Also converts back: InteractiveGraph → mastergo format for export.
 */

import type { InteractiveGraph, InteractiveNode, InteractiveEdge } from './interactive-svg'

// ── Mastergo types (mirrors Python omniparser_bridge output) ─────────

export interface MastergoElement {
  id: string
  name: string
  bbox: { x: number; y: number; width: number; height: number }
  _elk?: { iconHint?: string; group?: boolean; borderless?: boolean }
  _shared?: boolean
  _state_index?: number
  _refined?: { delta: { dx: number; dy: number; dw: number; dh: number } }
  _snapped?: { delta: { dx: number; dy: number; dw: number; dh: number } }
}

export interface MastergoEdge {
  id: string
  source: string
  target: string
  label: string
  type: string
  sections?: any[]
  style?: { strokeColor?: string; lineStyle?: string; strokeWidth?: number }
}

// ── Color palette for detected elements ──────────────────────────────

const PALETTE = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
  '#EC4899', '#06B6D4', '#F97316', '#14B8A6', '#6366F1',
]

function colorForIndex(i: number): string {
  return PALETTE[i % PALETTE.length]
}

// ── Mastergo → InteractiveGraph ──────────────────────────────────────

export function mastergoToInteractive(
  elements: MastergoElement[],
  edges?: MastergoEdge[],
): InteractiveGraph {
  // Compute canvas bounds from elements
  let maxX = 0, maxY = 0
  for (const el of elements) {
    const r = el.bbox.x + el.bbox.width
    const b = el.bbox.y + el.bbox.height
    if (r > maxX) maxX = r
    if (b > maxY) maxY = b
  }

  const nodes: InteractiveNode[] = elements.map((el, i) => ({
    id: el.id,
    x: el.bbox.x,
    y: el.bbox.y,
    width: el.bbox.width,
    height: el.bbox.height,
    label: el.name,
    fill: el._shared ? '#DBEAFE' : colorForIndex(i),
    isGroup: el._elk?.group ?? false,
  }))

  const edgeList: InteractiveEdge[] = (edges ?? []).map((e, i) => {
    const src = elements.find(el => el.id === e.source)
    const tgt = elements.find(el => el.id === e.target)

    // If sections have geometry, use it; otherwise connect centers
    let points: { x: number; y: number }[] = []
    if (e.sections?.[0]?.startPoint && e.sections?.[0]?.endPoint) {
      const s = e.sections[0]
      points.push(s.startPoint)
      if (s.bendPoints) points.push(...s.bendPoints)
      points.push(s.endPoint)
    } else if (src && tgt) {
      points = [
        { x: src.bbox.x + src.bbox.width / 2, y: src.bbox.y + src.bbox.height / 2 },
        { x: tgt.bbox.x + tgt.bbox.width / 2, y: tgt.bbox.y + tgt.bbox.height / 2 },
      ]
    }

    const style = e.style ?? {}
    return {
      id: e.id || `edge_${i}`,
      sourceId: e.source,
      targetId: e.target,
      points,
      label: e.label || undefined,
      color: style.strokeColor || '#94A3B8',
      dashArray: style.lineStyle === 'dashed' ? '6,4' : undefined,
      strokeWidth: style.strokeWidth || 1.5,
    }
  })

  return {
    nodes,
    edges: edgeList,
    width: Math.max(maxX + 40, 800),
    height: Math.max(maxY + 40, 600),
  }
}

// ── InteractiveGraph → Mastergo (export after user adjustments) ──────

export function interactiveToMastergo(graph: InteractiveGraph): MastergoElement[] {
  return graph.nodes.map(node => ({
    id: node.id,
    name: node.label,
    bbox: {
      x: Math.round(node.x),
      y: Math.round(node.y),
      width: Math.round(node.width),
      height: Math.round(node.height),
    },
  }))
}
