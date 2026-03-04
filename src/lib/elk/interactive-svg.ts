/**
 * interactive-svg.ts — Interactive SVG Editor Engine
 *
 * Renders ELK layouted JSON as interactive SVG with:
 * - Drag & drop nodes
 * - Double-click to edit labels
 * - Node resize via corner handles
 * - Edges auto-follow node movement
 * - Zoom & pan (mousewheel + middle-click drag)
 * - Snap to grid (borrowing from likec4 #1447)
 * - Add / remove nodes
 * - Export to static SVG / modified JSON
 *
 * Architecture inspired by ReactFlow + ELK.js integration pattern
 * (reactflow.dev/examples/layout/elkjs)
 *
 * GitHub: kieler/elkjs, xyflow/xyflow
 */

// ──── Types ─────────────────────────────────────────────
export interface InteractiveNode {
  id: string
  x: number
  y: number
  width: number
  height: number
  label: string
  fill: string
  children?: InteractiveNode[]
  isGroup?: boolean
}

export interface InteractiveEdge {
  id: string
  sourceId: string
  targetId: string
  points: { x: number; y: number }[]
  label?: string
  color: string
  dashArray?: string
  strokeWidth: number
}

export interface InteractiveGraph {
  nodes: InteractiveNode[]
  edges: InteractiveEdge[]
  width: number
  height: number
}

export interface InteractiveSvgOptions {
  gridSize?: number
  snapToGrid?: boolean
  minNodeWidth?: number
  minNodeHeight?: number
  padding?: number
  onNodeMove?: (nodeId: string, x: number, y: number) => void
  onNodeResize?: (nodeId: string, w: number, h: number) => void
  onLabelEdit?: (nodeId: string, newLabel: string) => void
  onGraphChange?: (graph: InteractiveGraph) => void
}

const DEFAULTS: Required<InteractiveSvgOptions> = {
  gridSize: 10,
  snapToGrid: true,
  minNodeWidth: 80,
  minNodeHeight: 40,
  padding: 20,
  onNodeMove: () => {},
  onNodeResize: () => {},
  onLabelEdit: () => {},
  onGraphChange: () => {},
}

const NODE_COLORS = ['#E3F2FD', '#F3E5F5', '#E8F5E9', '#FFF3E0', '#FCE4EC', '#E0F7FA', '#FFF8E1', '#F1F8E9']
const STROKE_COLOR = '#37474F'
const TEXT_COLOR = '#263238'
const EDGE_COLOR = '#78909C'
const SELECTION_COLOR = '#4F46E5'
const HANDLE_SIZE = 8
const ARROW_SIZE = 8

// ──── Main Class ────────────────────────────────────────
export class InteractiveSvgEditor {
  private container: HTMLElement
  private svg!: SVGSVGElement
  private graph: InteractiveGraph
  private opts: Required<InteractiveSvgOptions>

  // State
  private selectedNodeId: string | null = null
  private dragging: { nodeId: string; offsetX: number; offsetY: number } | null = null
  private resizing: { nodeId: string; corner: string; startX: number; startY: number; startW: number; startH: number; startNX: number; startNY: number } | null = null
  private panning: { startX: number; startY: number; startVX: number; startVY: number } | null = null
  private viewBox = { x: 0, y: 0, w: 0, h: 0 }
  private zoom = 1
  private editingLabelId: string | null = null

  constructor(container: HTMLElement, graph: InteractiveGraph, options?: InteractiveSvgOptions) {
    this.container = container
    this.graph = JSON.parse(JSON.stringify(graph)) // deep clone
    this.opts = { ...DEFAULTS, ...options }
    this.init()
  }

  // ──── Initialization ──────────────────────────────────
  private init() {
    this.container.innerHTML = ''
    const p = this.opts.padding

    // Set viewBox to encompass all nodes
    this.viewBox = {
      x: -p,
      y: -p,
      w: (this.graph.width || 800) + p * 2,
      h: (this.graph.height || 600) + p * 2,
    }

    // Create SVG element
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    this.svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    this.svg.style.width = '100%'
    this.svg.style.height = '100%'
    this.svg.style.minHeight = '320px'
    this.svg.style.cursor = 'default'
    this.svg.style.userSelect = 'none'
    this.updateViewBox()

    // Defs: arrow markers
    const defs = this.createDefs()
    this.svg.appendChild(defs)

    // Background grid (snap-to-grid visual)
    if (this.opts.snapToGrid) {
      this.svg.appendChild(this.createGrid())
    }

    // Render edges first (behind nodes)
    for (const edge of this.graph.edges) {
      this.svg.appendChild(this.createEdgeGroup(edge))
    }

    // Render nodes
    this.graph.nodes.forEach((node, i) => {
      this.svg.appendChild(this.createNodeGroup(node, i))
    })

    // Attach to container
    this.container.appendChild(this.svg)

    // Global mouse events for drag/resize/pan
    this.svg.addEventListener('mousemove', this.onMouseMove.bind(this))
    this.svg.addEventListener('mouseup', this.onMouseUp.bind(this))
    this.svg.addEventListener('mouseleave', this.onMouseUp.bind(this))
    this.svg.addEventListener('wheel', this.onWheel.bind(this), { passive: false })
    this.svg.addEventListener('mousedown', this.onSvgMouseDown.bind(this))

    // Click on empty space to deselect
    this.svg.addEventListener('click', (e) => {
      if ((e.target as Element) === this.svg || (e.target as Element).classList.contains('svg-grid')) {
        this.deselectAll()
      }
    })
  }

  // ──── SVG Element Creators ────────────────────────────

  private createDefs(): SVGDefsElement {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')

    // Default arrow marker
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker')
    marker.setAttribute('id', 'arrow-default')
    marker.setAttribute('markerWidth', String(ARROW_SIZE))
    marker.setAttribute('markerHeight', String(ARROW_SIZE / 1.5))
    marker.setAttribute('refX', String(ARROW_SIZE))
    marker.setAttribute('refY', String(ARROW_SIZE / 3))
    marker.setAttribute('orient', 'auto')
    marker.setAttribute('markerUnits', 'strokeWidth')

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
    polygon.setAttribute('points', `0 0, ${ARROW_SIZE} ${ARROW_SIZE / 3}, 0 ${ARROW_SIZE / 1.5}`)
    polygon.setAttribute('fill', EDGE_COLOR)
    marker.appendChild(polygon)
    defs.appendChild(marker)

    // Grid pattern
    if (this.opts.snapToGrid) {
      const gs = this.opts.gridSize
      const pattern = document.createElementNS('http://www.w3.org/2000/svg', 'pattern')
      pattern.setAttribute('id', 'grid-pattern')
      pattern.setAttribute('width', String(gs))
      pattern.setAttribute('height', String(gs))
      pattern.setAttribute('patternUnits', 'userSpaceOnUse')

      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
      dot.setAttribute('cx', String(gs))
      dot.setAttribute('cy', String(gs))
      dot.setAttribute('r', '0.5')
      dot.setAttribute('fill', '#CBD5E1')
      pattern.appendChild(dot)
      defs.appendChild(pattern)
    }

    return defs
  }

  private createGrid(): SVGRectElement {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.classList.add('svg-grid')
    rect.setAttribute('x', String(this.viewBox.x))
    rect.setAttribute('y', String(this.viewBox.y))
    rect.setAttribute('width', String(this.viewBox.w))
    rect.setAttribute('height', String(this.viewBox.h))
    rect.setAttribute('fill', 'url(#grid-pattern)')
    return rect
  }

  private createNodeGroup(node: InteractiveNode, index: number): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-node-id', node.id)
    g.classList.add('interactive-node')
    g.style.cursor = 'grab'

    // Main rect
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.setAttribute('x', String(node.x))
    rect.setAttribute('y', String(node.y))
    rect.setAttribute('width', String(node.width))
    rect.setAttribute('height', String(node.height))
    rect.setAttribute('fill', node.fill || NODE_COLORS[index % NODE_COLORS.length])
    rect.setAttribute('stroke', STROKE_COLOR)
    rect.setAttribute('stroke-width', node.isGroup ? '2' : '1.5')
    rect.setAttribute('rx', '8')
    if (node.isGroup) rect.setAttribute('stroke-dasharray', '6,3')
    g.appendChild(rect)

    // Label text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text')
    const maxChars = Math.max(6, Math.floor(node.width / 8))
    const dl = node.label.length > maxChars ? node.label.slice(0, maxChars - 2) + '…' : node.label
    text.setAttribute('x', String(node.x + node.width / 2))
    text.setAttribute('y', String(node.isGroup ? node.y + 16 : node.y + node.height / 2))
    text.setAttribute('text-anchor', 'middle')
    text.setAttribute('dominant-baseline', 'central')
    text.setAttribute('font-family', 'system-ui, -apple-system, sans-serif')
    text.setAttribute('font-size', node.isGroup ? '11' : '12')
    text.setAttribute('font-weight', node.isGroup ? '600' : '500')
    text.setAttribute('fill', TEXT_COLOR)
    text.setAttribute('pointer-events', 'none')
    text.textContent = dl
    g.appendChild(text)

    // Resize handles (hidden until selected)
    const corners = ['nw', 'ne', 'sw', 'se']
    for (const corner of corners) {
      const handle = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      handle.classList.add('resize-handle', `resize-${corner}`)
      handle.setAttribute('width', String(HANDLE_SIZE))
      handle.setAttribute('height', String(HANDLE_SIZE))
      handle.setAttribute('fill', SELECTION_COLOR)
      handle.setAttribute('stroke', 'white')
      handle.setAttribute('stroke-width', '1')
      handle.setAttribute('rx', '2')
      handle.style.display = 'none'
      handle.style.cursor = corner === 'nw' || corner === 'se' ? 'nwse-resize' : 'nesw-resize'
      this.positionHandle(handle, node, corner)

      handle.addEventListener('mousedown', (e) => {
        e.stopPropagation()
        this.startResize(node.id, corner, e)
      })

      g.appendChild(handle)
    }

    // Drag events
    g.addEventListener('mousedown', (e) => {
      if ((e.target as Element).classList.contains('resize-handle')) return
      e.stopPropagation()
      this.startDrag(node.id, e)
    })

    // Double-click to edit label
    g.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      this.startLabelEdit(node.id)
    })

    // Click to select
    g.addEventListener('click', (e) => {
      e.stopPropagation()
      this.selectNode(node.id)
    })

    return g
  }

  private createEdgeGroup(edge: InteractiveEdge): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-edge-id', edge.id)
    g.classList.add('interactive-edge')

    if (edge.points.length < 2) return g

    // Build path
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    const d = edge.points.map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ')
    path.setAttribute('d', d)
    path.setAttribute('fill', 'none')
    path.setAttribute('stroke', edge.color || EDGE_COLOR)
    path.setAttribute('stroke-width', String(edge.strokeWidth || 1.5))
    if (edge.dashArray) path.setAttribute('stroke-dasharray', edge.dashArray)
    path.setAttribute('marker-end', 'url(#arrow-default)')
    g.appendChild(path)

    // Edge label
    if (edge.label) {
      const mid = Math.floor(edge.points.length / 2)
      const mp = edge.points[mid]
      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      bg.setAttribute('x', String(mp.x - edge.label.length * 3.5))
      bg.setAttribute('y', String(mp.y - 8))
      bg.setAttribute('width', String(edge.label.length * 7))
      bg.setAttribute('height', '16')
      bg.setAttribute('fill', 'white')
      bg.setAttribute('rx', '3')
      bg.setAttribute('opacity', '0.9')
      g.appendChild(bg)

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      text.setAttribute('x', String(mp.x))
      text.setAttribute('y', String(mp.y + 4))
      text.setAttribute('text-anchor', 'middle')
      text.setAttribute('font-family', 'system-ui, sans-serif')
      text.setAttribute('font-size', '10')
      text.setAttribute('fill', edge.color || EDGE_COLOR)
      text.textContent = edge.label
      g.appendChild(text)
    }

    return g
  }

  // ──── Interaction: Drag ───────────────────────────────

  private startDrag(nodeId: string, e: MouseEvent) {
    const node = this.findNode(nodeId)
    if (!node) return
    const pt = this.clientToSvg(e)
    this.dragging = { nodeId, offsetX: pt.x - node.x, offsetY: pt.y - node.y }
    this.selectNode(nodeId)
    this.svg.style.cursor = 'grabbing'
  }

  private onMouseMove(e: MouseEvent) {
    if (this.dragging) {
      const pt = this.clientToSvg(e)
      const node = this.findNode(this.dragging.nodeId)
      if (!node) return
      let newX = pt.x - this.dragging.offsetX
      let newY = pt.y - this.dragging.offsetY
      if (this.opts.snapToGrid) {
        const gs = this.opts.gridSize
        newX = Math.round(newX / gs) * gs
        newY = Math.round(newY / gs) * gs
      }
      node.x = newX
      node.y = newY
      this.updateNodeVisual(node)
      this.updateEdgesForNode(node.id)
      this.opts.onNodeMove(node.id, newX, newY)
    } else if (this.resizing) {
      this.handleResize(e)
    } else if (this.panning) {
      const dx = (e.clientX - this.panning.startX) / this.zoom
      const dy = (e.clientY - this.panning.startY) / this.zoom
      this.viewBox.x = this.panning.startVX - dx
      this.viewBox.y = this.panning.startVY - dy
      this.updateViewBox()
    }
  }

  private onMouseUp(_e: MouseEvent) {
    if (this.dragging || this.resizing) {
      this.opts.onGraphChange(this.graph)
    }
    this.dragging = null
    this.resizing = null
    this.panning = null
    this.svg.style.cursor = 'default'
  }

  // ──── Interaction: Resize ─────────────────────────────

  private startResize(nodeId: string, corner: string, e: MouseEvent) {
    const node = this.findNode(nodeId)
    if (!node) return
    const pt = this.clientToSvg(e)
    this.resizing = {
      nodeId, corner,
      startX: pt.x, startY: pt.y,
      startW: node.width, startH: node.height,
      startNX: node.x, startNY: node.y,
    }
  }

  private handleResize(e: MouseEvent) {
    if (!this.resizing) return
    const node = this.findNode(this.resizing.nodeId)
    if (!node) return
    const pt = this.clientToSvg(e)
    const dx = pt.x - this.resizing.startX
    const dy = pt.y - this.resizing.startY
    const minW = this.opts.minNodeWidth
    const minH = this.opts.minNodeHeight

    switch (this.resizing.corner) {
      case 'se':
        node.width = Math.max(minW, this.resizing.startW + dx)
        node.height = Math.max(minH, this.resizing.startH + dy)
        break
      case 'sw':
        node.width = Math.max(minW, this.resizing.startW - dx)
        node.height = Math.max(minH, this.resizing.startH + dy)
        node.x = this.resizing.startNX + this.resizing.startW - node.width
        break
      case 'ne':
        node.width = Math.max(minW, this.resizing.startW + dx)
        node.height = Math.max(minH, this.resizing.startH - dy)
        node.y = this.resizing.startNY + this.resizing.startH - node.height
        break
      case 'nw':
        node.width = Math.max(minW, this.resizing.startW - dx)
        node.height = Math.max(minH, this.resizing.startH - dy)
        node.x = this.resizing.startNX + this.resizing.startW - node.width
        node.y = this.resizing.startNY + this.resizing.startH - node.height
        break
    }

    if (this.opts.snapToGrid) {
      const gs = this.opts.gridSize
      node.x = Math.round(node.x / gs) * gs
      node.y = Math.round(node.y / gs) * gs
      node.width = Math.round(node.width / gs) * gs
      node.height = Math.round(node.height / gs) * gs
    }

    this.updateNodeVisual(node)
    this.updateEdgesForNode(node.id)
    this.opts.onNodeResize(node.id, node.width, node.height)
  }

  // ──── Interaction: Pan & Zoom ─────────────────────────

  private onSvgMouseDown(e: MouseEvent) {
    // Middle mouse button or Alt+Left click for panning
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault()
      this.panning = {
        startX: e.clientX, startY: e.clientY,
        startVX: this.viewBox.x, startVY: this.viewBox.y,
      }
      this.svg.style.cursor = 'move'
    }
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault()
    const scaleFactor = e.deltaY > 0 ? 1.1 : 0.9
    const pt = this.clientToSvg(e)

    this.viewBox.x = pt.x - (pt.x - this.viewBox.x) * scaleFactor
    this.viewBox.y = pt.y - (pt.y - this.viewBox.y) * scaleFactor
    this.viewBox.w *= scaleFactor
    this.viewBox.h *= scaleFactor

    this.zoom = (this.graph.width + this.opts.padding * 2) / this.viewBox.w
    this.updateViewBox()
  }

  // ──── Interaction: Label Edit ─────────────────────────

  private startLabelEdit(nodeId: string) {
    const node = this.findNode(nodeId)
    if (!node) return
    this.editingLabelId = nodeId

    // Create foreignObject with input
    const g = this.svg.querySelector(`[data-node-id="${nodeId}"]`)
    if (!g) return

    const fo = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject')
    fo.setAttribute('x', String(node.x + 4))
    fo.setAttribute('y', String(node.y + (node.isGroup ? 2 : node.height / 2 - 14)))
    fo.setAttribute('width', String(node.width - 8))
    fo.setAttribute('height', '28')
    fo.classList.add('label-editor-fo')

    const input = document.createElement('input')
    input.type = 'text'
    input.value = node.label
    input.style.cssText = 'width:100%;height:100%;border:2px solid #4F46E5;border-radius:4px;padding:2px 6px;font-size:12px;font-family:system-ui,sans-serif;text-align:center;background:white;outline:none;box-sizing:border-box;'

    input.addEventListener('blur', () => {
      node.label = input.value || node.label
      this.editingLabelId = null
      fo.remove()
      this.updateNodeVisual(node)
      this.opts.onLabelEdit(node.id, node.label)
      this.opts.onGraphChange(this.graph)
    })

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') input.blur()
      if (e.key === 'Escape') {
        input.value = node.label
        input.blur()
      }
    })

    fo.appendChild(input)
    g.appendChild(fo)

    // Focus after a tick (needed for foreignObject rendering)
    requestAnimationFrame(() => {
      input.focus()
      input.select()
    })
  }

  // ──── Node Selection ──────────────────────────────────

  private selectNode(nodeId: string) {
    this.deselectAll()
    this.selectedNodeId = nodeId
    const g = this.svg.querySelector(`[data-node-id="${nodeId}"]`)
    if (!g) return

    // Highlight border
    const rect = g.querySelector('rect:not(.resize-handle)')
    if (rect) {
      rect.setAttribute('stroke', SELECTION_COLOR)
      rect.setAttribute('stroke-width', '2.5')
    }

    // Show resize handles
    g.querySelectorAll('.resize-handle').forEach((h) => {
      ;(h as SVGElement).style.display = 'block'
    })
  }

  private deselectAll() {
    if (!this.selectedNodeId) return
    const g = this.svg.querySelector(`[data-node-id="${this.selectedNodeId}"]`)
    if (g) {
      const rect = g.querySelector('rect:not(.resize-handle)')
      if (rect) {
        rect.setAttribute('stroke', STROKE_COLOR)
        const node = this.findNode(this.selectedNodeId)
        rect.setAttribute('stroke-width', node?.isGroup ? '2' : '1.5')
      }
      g.querySelectorAll('.resize-handle').forEach((h) => {
        ;(h as SVGElement).style.display = 'none'
      })
    }
    this.selectedNodeId = null
  }

  // ──── Visual Updates ──────────────────────────────────

  private updateNodeVisual(node: InteractiveNode) {
    const g = this.svg.querySelector(`[data-node-id="${node.id}"]`)
    if (!g) return

    const rect = g.querySelector('rect:not(.resize-handle)')
    if (rect) {
      rect.setAttribute('x', String(node.x))
      rect.setAttribute('y', String(node.y))
      rect.setAttribute('width', String(node.width))
      rect.setAttribute('height', String(node.height))
    }

    const text = g.querySelector('text')
    if (text) {
      const maxChars = Math.max(6, Math.floor(node.width / 8))
      const dl = node.label.length > maxChars ? node.label.slice(0, maxChars - 2) + '…' : node.label
      text.setAttribute('x', String(node.x + node.width / 2))
      text.setAttribute('y', String(node.isGroup ? node.y + 16 : node.y + node.height / 2))
      text.textContent = dl
    }

    // Update resize handles
    const corners = ['nw', 'ne', 'sw', 'se']
    for (const corner of corners) {
      const handle = g.querySelector(`.resize-${corner}`) as SVGRectElement | null
      if (handle) this.positionHandle(handle, node, corner)
    }
  }

  private positionHandle(handle: SVGRectElement, node: InteractiveNode, corner: string) {
    const hs = HANDLE_SIZE
    switch (corner) {
      case 'nw':
        handle.setAttribute('x', String(node.x - hs / 2))
        handle.setAttribute('y', String(node.y - hs / 2))
        break
      case 'ne':
        handle.setAttribute('x', String(node.x + node.width - hs / 2))
        handle.setAttribute('y', String(node.y - hs / 2))
        break
      case 'sw':
        handle.setAttribute('x', String(node.x - hs / 2))
        handle.setAttribute('y', String(node.y + node.height - hs / 2))
        break
      case 'se':
        handle.setAttribute('x', String(node.x + node.width - hs / 2))
        handle.setAttribute('y', String(node.y + node.height - hs / 2))
        break
    }
  }

  private updateEdgesForNode(nodeId: string) {
    const node = this.findNode(nodeId)
    if (!node) return

    for (const edge of this.graph.edges) {
      if (edge.sourceId === nodeId || edge.targetId === nodeId) {
        this.recalcEdgePoints(edge)
        this.updateEdgeVisual(edge)
      }
    }
  }

  private recalcEdgePoints(edge: InteractiveEdge) {
    const src = this.findNode(edge.sourceId)
    const tgt = this.findNode(edge.targetId)
    if (!src || !tgt) return

    // Simple center-to-center with port detection
    const sx = src.x + src.width / 2
    const sy = src.y + src.height
    const tx = tgt.x + tgt.width / 2
    const ty = tgt.y

    // For orthogonal routing, add a midpoint
    const midY = (sy + ty) / 2
    if (Math.abs(sy - ty) > 20) {
      edge.points = [
        { x: sx, y: sy },
        { x: sx, y: midY },
        { x: tx, y: midY },
        { x: tx, y: ty },
      ]
    } else {
      edge.points = [
        { x: sx, y: sy },
        { x: tx, y: ty },
      ]
    }
  }

  private updateEdgeVisual(edge: InteractiveEdge) {
    const g = this.svg.querySelector(`[data-edge-id="${edge.id}"]`)
    if (!g) return
    const path = g.querySelector('path')
    if (path && edge.points.length >= 2) {
      const d = edge.points.map((p, i) => (i === 0 ? `M${p.x},${p.y}` : `L${p.x},${p.y}`)).join(' ')
      path.setAttribute('d', d)
    }
  }

  // ──── Coordinate Conversion ───────────────────────────

  private clientToSvg(e: MouseEvent): { x: number; y: number } {
    const rect = this.svg.getBoundingClientRect()
    const scaleX = this.viewBox.w / rect.width
    const scaleY = this.viewBox.h / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX + this.viewBox.x,
      y: (e.clientY - rect.top) * scaleY + this.viewBox.y,
    }
  }

  private updateViewBox() {
    this.svg.setAttribute('viewBox', `${this.viewBox.x} ${this.viewBox.y} ${this.viewBox.w} ${this.viewBox.h}`)
  }

  // ──── Node Lookup ─────────────────────────────────────

  private findNode(id: string): InteractiveNode | undefined {
    return this.graph.nodes.find((n) => n.id === id)
  }

  // ──── Public API ──────────────────────────────────────

  /** Get the current graph state (with user modifications) */
  getGraph(): InteractiveGraph {
    return JSON.parse(JSON.stringify(this.graph))
  }

  /** Get modified graph as ELK-compatible JSON for re-layout */
  getElkJson(): any {
    return {
      id: 'root',
      children: this.graph.nodes.map((n) => ({
        id: n.id,
        width: n.width,
        height: n.height,
        labels: [{ text: n.label }],
      })),
      edges: this.graph.edges.map((e) => ({
        id: e.id,
        sources: [e.sourceId],
        targets: [e.targetId],
        ...(e.label ? { labels: [{ text: e.label }] } : {}),
      })),
    }
  }

  /** Export to static SVG string */
  toStaticSvg(): string {
    // Clone SVG without interactive elements
    const clone = this.svg.cloneNode(true) as SVGSVGElement
    // Remove resize handles and editors
    clone.querySelectorAll('.resize-handle, .label-editor-fo').forEach((el) => el.remove())
    // Reset selection styles
    clone.querySelectorAll('.interactive-node rect:not(.resize-handle)').forEach((rect) => {
      rect.setAttribute('stroke', STROKE_COLOR)
    })
    return new XMLSerializer().serializeToString(clone)
  }

  /** Add a new node */
  addNode(label: string): InteractiveNode {
    const id = `node_${Date.now()}`
    const maxX = Math.max(...this.graph.nodes.map((n) => n.x + n.width), 100)
    const node: InteractiveNode = {
      id,
      x: maxX + 30,
      y: 50,
      width: 160,
      height: 60,
      label,
      fill: NODE_COLORS[this.graph.nodes.length % NODE_COLORS.length],
    }
    this.graph.nodes.push(node)
    this.svg.appendChild(this.createNodeGroup(node, this.graph.nodes.length - 1))
    this.opts.onGraphChange(this.graph)
    return node
  }

  /** Remove selected node */
  removeSelectedNode(): boolean {
    if (!this.selectedNodeId) return false
    const id = this.selectedNodeId
    // Remove node from graph
    this.graph.nodes = this.graph.nodes.filter((n) => n.id !== id)
    // Remove edges connected to this node
    const edgesToRemove = this.graph.edges.filter((e) => e.sourceId === id || e.targetId === id)
    this.graph.edges = this.graph.edges.filter((e) => e.sourceId !== id && e.targetId !== id)
    // Remove from SVG
    this.svg.querySelector(`[data-node-id="${id}"]`)?.remove()
    edgesToRemove.forEach((e) => {
      this.svg.querySelector(`[data-edge-id="${e.id}"]`)?.remove()
    })
    this.selectedNodeId = null
    this.opts.onGraphChange(this.graph)
    return true
  }

  /** Fit view to show all content */
  fitView() {
    const p = this.opts.padding
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const n of this.graph.nodes) {
      if (n.x < minX) minX = n.x
      if (n.y < minY) minY = n.y
      if (n.x + n.width > maxX) maxX = n.x + n.width
      if (n.y + n.height > maxY) maxY = n.y + n.height
    }
    if (minX === Infinity) return
    this.viewBox = {
      x: minX - p,
      y: minY - p,
      w: (maxX - minX) + p * 2,
      h: (maxY - minY) + p * 2,
    }
    this.updateViewBox()
  }

  /** Set zoom level (1 = 100%) */
  setZoom(level: number) {
    const cx = this.viewBox.x + this.viewBox.w / 2
    const cy = this.viewBox.y + this.viewBox.h / 2
    const baseW = (this.graph.width || 800) + this.opts.padding * 2
    const baseH = (this.graph.height || 600) + this.opts.padding * 2
    this.viewBox.w = baseW / level
    this.viewBox.h = baseH / level
    this.viewBox.x = cx - this.viewBox.w / 2
    this.viewBox.y = cy - this.viewBox.h / 2
    this.zoom = level
    this.updateViewBox()
  }

  /** Update from new ELK layout data */
  updateFromLayout(graph: InteractiveGraph) {
    this.graph = JSON.parse(JSON.stringify(graph))
    this.init()
  }

  /** Destroy the editor */
  destroy() {
    this.container.innerHTML = ''
  }
}

export default InteractiveSvgEditor
