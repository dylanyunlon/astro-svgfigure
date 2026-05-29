/**
 * elkToTldraw / tldrawToElk — bidirectional conversion (M303 + M304).
 *
 * Reference: tldraw/apps/examples/.../CustomShapeMermaids.tsx L76-84
 *   createMermaidDiagram(editor, text, { blueprintRender: { ... } })
 *   iterates mermaid nodes → editor.createShape() per vertex,
 *   then tldraw's arrow system handles edges via bindings.
 *
 * Diff from reference:
 *   - Input: ELK layouted JSON (with x,y already computed) vs mermaid text
 *   - No mermaid parsing needed — ELK already gives us coordinates
 *   - Reverse direction (tldrawToElk) reads editor state back to ELK JSON
 *   - Groups use tldraw parent-child relationship
 */
import type { Editor, TLShapeId, TLShapePartial } from 'tldraw'
import { createShapeId } from 'tldraw'
import { ELK_NODE_TYPE, type ElkNodeShape } from './ElkNodeShapeUtil'

// ═══════════════════════════════════════════════════════════════════════════
//  §1  ELK types (minimal — matches structured_data.txt structure)
// ═══════════════════════════════════════════════════════════════════════════

interface ElkLabel { text: string }
interface ElkBendPoint { x: number; y: number }
interface ElkSection {
  startPoint: { x: number; y: number }
  endPoint: { x: number; y: number }
  bendPoints?: ElkBendPoint[]
}
interface ElkEdge {
  id: string
  sources?: string[]
  targets?: string[]
  sections?: ElkSection[]
  advanced?: { strokeColor?: string; lineStyle?: string; semanticType?: string }
}
interface ElkNode {
  id: string
  x?: number; y?: number; width?: number; height?: number
  labels?: ElkLabel[]
  children?: ElkNode[]
  edges?: ElkEdge[]
  group?: boolean; borderless?: boolean
  renderMode?: string; isOperator?: boolean; familyId?: string; iconHint?: string
  spriteRef?: { format: string; url?: string; svg?: string; stackCount?: number }
}
interface ElkGraph extends ElkNode {
  edges?: ElkEdge[]
}

// ═══════════════════════════════════════════════════════════════════════════
//  §2  elkToTldraw — import ELK graph into tldraw editor
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert an ELK layouted JSON graph into tldraw shapes on the canvas.
 *
 * Follows the same pattern as tldraw's createMermaidDiagram:
 *   1. Clear existing shapes
 *   2. Walk the ELK tree, create an ElkNodeShape per node
 *   3. Groups use tldraw's parent-child: child shape.parentId = group shape id
 *   4. Edges become tldraw arrow shapes with terminal bindings
 *
 * Returns a map of elkId → tldraw shapeId for downstream use.
 */
export function elkToTldraw(
  editor: Editor,
  graph: ElkGraph,
  opts?: { clearFirst?: boolean; offsetX?: number; offsetY?: number },
): Map<string, TLShapeId> {
  const clearFirst = opts?.clearFirst ?? true
  const ox = opts?.offsetX ?? 0
  const oy = opts?.offsetY ?? 0
  const idMap = new Map<string, TLShapeId>()

  if (clearFirst) {
    editor.selectAll()
    editor.deleteShapes(editor.getSelectedShapes())
  }

  // Pass 1: create all node shapes (depth-first)
  const shapePartials: TLShapePartial[] = []
  const parentMap = new Map<string, TLShapeId>()  // elkId → parent tldraw id

  function walkNodes(
    nodes: ElkNode[],
    parentElkId: string | null,
    parentX: number,
    parentY: number,
    depth: number,
  ) {
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      const shapeId = createShapeId(`elk-${node.id}`)
      idMap.set(node.id, shapeId)

      const isGroup = Array.isArray(node.children) && node.children.length > 0
      const label = node.labels?.[0]?.text || node.id
      const w = node.width || 160
      const h = node.height || 50

      // Absolute position = parent offset + node local position
      const absX = parentX + (node.x || 0) + ox
      const absY = parentY + (node.y || 0) + oy

      // Sprite info
      const spriteRef = node.spriteRef
      const spriteUrl = spriteRef?.url || ''
      const spriteFormat = spriteRef?.format || ''

      const partial: TLShapePartial = {
        id: shapeId,
        type: ELK_NODE_TYPE,
        x: absX,
        y: absY,
        props: {
          w,
          h,
          elkId: node.id,
          label,
          renderMode: node.renderMode || 'text',
          familyId: node.familyId || '',
          spriteUrl,
          spriteFormat,
          iconHint: node.iconHint || '',
          isOperator: !!node.isOperator,
          fillColor: '',
          depth,
          isGroup,
        },
      }

      // Parent-child relationship for groups
      if (parentElkId && parentMap.has(parentElkId)) {
        ;(partial as any).parentId = parentMap.get(parentElkId)
      }

      shapePartials.push(partial)

      if (isGroup) {
        parentMap.set(node.id, shapeId)
        walkNodes(node.children!, node.id, absX, absY, depth + 1)
      }
    }
  }

  if (graph.children) {
    walkNodes(graph.children, null, 0, 0, 0)
  }

  // Batch create all shapes
  editor.createShapes(shapePartials)

  // Pass 2: create arrow shapes for edges
  const allEdges: { edge: ElkEdge; offsetX: number; offsetY: number }[] = []

  // Root-level edges
  if (graph.edges) {
    for (const edge of graph.edges) {
      allEdges.push({ edge, offsetX: ox, offsetY: oy })
    }
  }

  // Nested edges (inside compound nodes)
  function collectEdges(nodes: ElkNode[], px: number, py: number) {
    for (const node of nodes) {
      const nx = px + (node.x || 0)
      const ny = py + (node.y || 0)
      if (node.edges) {
        for (const edge of node.edges) {
          allEdges.push({ edge, offsetX: nx + ox, offsetY: ny + oy })
        }
      }
      if (node.children) {
        collectEdges(node.children, nx, ny)
      }
    }
  }
  if (graph.children) {
    collectEdges(graph.children, 0, 0)
  }

  // Create arrow shapes from edge sections
  for (const { edge, offsetX: ex, offsetY: ey } of allEdges) {
    const srcId = edge.sources?.[0]
    const tgtId = edge.targets?.[0]
    if (!srcId || !tgtId) continue

    const srcShapeId = idMap.get(srcId)
    const tgtShapeId = idMap.get(tgtId)

    // Use first section's start/end points for arrow positioning
    const section = edge.sections?.[0]
    if (!section) continue

    const startX = section.startPoint.x + ex
    const startY = section.startPoint.y + ey
    const endX = section.endPoint.x + ex
    const endY = section.endPoint.y + ey

    const arrowId = createShapeId(`elk-edge-${edge.id}`)

    // Create a tldraw arrow shape
    // The arrow will auto-bind to source/target if they overlap
    editor.createShape({
      id: arrowId,
      type: 'arrow',
      x: startX,
      y: startY,
      props: {
        start: { x: 0, y: 0 },
        end: { x: endX - startX, y: endY - startY },
        color: edge.advanced?.strokeColor ? 'grey' : 'black',
        dash: edge.advanced?.lineStyle === 'dashed' ? 'dashed' : 'draw',
        size: 's',
        arrowheadEnd: 'arrow',
        arrowheadStart: 'none',
      },
    })

    // Bind arrow terminals to source/target shapes
    if (srcShapeId) {
      editor.createBinding({
        type: 'arrow',
        fromId: arrowId,
        toId: srcShapeId,
        props: {
          terminal: 'start',
          normalizedAnchor: { x: 0.5, y: 1 },
          isExact: false,
          isPrecise: false,
        },
      })
    }
    if (tgtShapeId) {
      editor.createBinding({
        type: 'arrow',
        fromId: arrowId,
        toId: tgtShapeId,
        props: {
          terminal: 'end',
          normalizedAnchor: { x: 0.5, y: 0 },
          isExact: false,
          isPrecise: false,
        },
      })
    }
  }

  return idMap
}

// ═══════════════════════════════════════════════════════════════════════════
//  §3  tldrawToElk — export editor state back to ELK JSON
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Read the current tldraw editor state and reconstruct an ELK-compatible
 * JSON graph with updated positions (from user dragging).
 *
 * This is the reverse of elkToTldraw. Only ElkNodeShape positions are read;
 * edges are reconstructed from arrow bindings.
 */
export function tldrawToElk(editor: Editor): ElkGraph {
  const shapes = editor.getCurrentPageShapes()
  const elkNodes: ElkNode[] = []
  const nodeById = new Map<string, ElkNode>()

  // Collect all elk-node shapes
  for (const shape of shapes) {
    if (shape.type !== ELK_NODE_TYPE) continue
    const props = (shape as ElkNodeShape).props

    const elkNode: ElkNode = {
      id: props.elkId || shape.id,
      x: shape.x,
      y: shape.y,
      width: props.w,
      height: props.h,
      labels: [{ text: props.label }],
      renderMode: props.renderMode || 'text',
      isOperator: props.isOperator,
      familyId: props.familyId || undefined,
      iconHint: props.iconHint || undefined,
      group: props.isGroup,
    }

    // Preserve spriteRef if present
    if (props.spriteUrl) {
      elkNode.spriteRef = {
        format: (props.spriteFormat || 'png') as any,
        url: props.spriteUrl,
      }
    }

    nodeById.set(elkNode.id, elkNode)

    // Top-level shapes (no parent) go directly into children
    if (!shape.parentId || shape.parentId === editor.getCurrentPageId()) {
      elkNodes.push(elkNode)
    }
  }

  // Reconstruct parent-child from tldraw parentId
  for (const shape of shapes) {
    if (shape.type !== ELK_NODE_TYPE) continue
    const props = (shape as ElkNodeShape).props
    const elkId = props.elkId || shape.id

    if (shape.parentId && shape.parentId !== editor.getCurrentPageId()) {
      // Find parent's elkId
      const parentShape = editor.getShape(shape.parentId)
      if (parentShape && parentShape.type === ELK_NODE_TYPE) {
        const parentElkId = (parentShape as ElkNodeShape).props.elkId
        const parentNode = nodeById.get(parentElkId)
        if (parentNode) {
          if (!parentNode.children) parentNode.children = []
          const childNode = nodeById.get(elkId)
          if (childNode) {
            // Convert to parent-relative coordinates
            childNode.x = (childNode.x || 0) - (parentNode.x || 0)
            childNode.y = (childNode.y || 0) - (parentNode.y || 0)
            parentNode.children.push(childNode)
          }
        }
      }
    }
  }

  // Collect edges from arrow shapes + bindings
  const edges: ElkEdge[] = []
  for (const shape of shapes) {
    if (shape.type !== 'arrow') continue
    const bindings = editor.getBindingsFromShape(shape.id, 'arrow')
    let srcElkId = ''
    let tgtElkId = ''

    for (const binding of bindings) {
      const targetShape = editor.getShape(binding.toId)
      if (!targetShape || targetShape.type !== ELK_NODE_TYPE) continue
      const elkId = (targetShape as ElkNodeShape).props.elkId
      if (binding.props.terminal === 'start') srcElkId = elkId
      else if (binding.props.terminal === 'end') tgtElkId = elkId
    }

    if (srcElkId && tgtElkId) {
      edges.push({
        id: shape.id,
        sources: [srcElkId],
        targets: [tgtElkId],
      })
    }
  }

  return {
    id: 'root',
    children: elkNodes,
    edges,
  }
}
