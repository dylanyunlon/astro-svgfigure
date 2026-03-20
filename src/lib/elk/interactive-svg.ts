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
export type HandlePosition = 'left' | 'right' | 'top' | 'bottom'
export type HandleType = 'source' | 'target'

export interface NodeHandle {
  id: string
  type: HandleType
  position: HandlePosition
}

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
  handles?: NodeHandle[]
}

export interface InteractiveEdge {
  id: string
  sourceId: string
  targetId: string
  sourceHandleId?: string
  targetHandleId?: string
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

// ──── Theme-aware Colors (astro-pure compatible) ────────
// Use CSS custom properties at runtime; fallback to neutral tones
// References: reactflow.dev/examples/layout/elkjs (clean, no random colors)
function getCSSVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return val || fallback
}

function getThemeColors() {
  return {
    nodeFill: getCSSVar('--color-muted', '#F8FAFC'),
    nodeStroke: getCSSVar('--color-border', '#E2E8F0'),
    textColor: getCSSVar('--color-foreground', '#1E293B'),
    edgeColor: getCSSVar('--color-muted-foreground', '#94A3B8'),
    selectionColor: getCSSVar('--color-primary', '#4F46E5'),
    background: getCSSVar('--color-background', '#FFFFFF'),
  }
}

// Static fallbacks used during SSR or when CSS vars unavailable
const STROKE_COLOR_FALLBACK = '#E2E8F0'
const TEXT_COLOR_FALLBACK = '#1E293B'
const EDGE_COLOR_FALLBACK = '#94A3B8'
const SELECTION_COLOR_FALLBACK = '#4F46E5'
const HANDLE_SIZE = 8
const ARROW_SIZE = 8
const MIN_ZOOM = 0.1
const MAX_ZOOM = 5
const ZOOM_SENSITIVITY = 0.08  // Smoother than 10% per step

// ──── Main Class ────────────────────────────────────────
export class InteractiveSvgEditor {
  private container: HTMLElement
  private svg!: SVGSVGElement
  private graph: InteractiveGraph
  private opts: Required<InteractiveSvgOptions>

  // Theme colors resolved at runtime
  private theme = {
    nodeFill: '#F8FAFC',
    nodeStroke: '#E2E8F0',
    textColor: '#1E293B',
    edgeColor: '#94A3B8',
    selectionColor: '#4F46E5',
    background: '#FFFFFF',
  }

  // State
  private selectedNodeId: string | null = null
  private dragging: { nodeId: string; offsetX: number; offsetY: number } | null = null
  private resizing: { nodeId: string; corner: string; startX: number; startY: number; startW: number; startH: number; startNX: number; startNY: number } | null = null
  private panning: { startX: number; startY: number; startVX: number; startVY: number } | null = null
  private panMoved = false  // Track if mouse moved during pan to prevent deselect on pan-end
  private touchState: { lastDist: number; lastMidX: number; lastMidY: number; startVX: number; startVY: number } | null = null
  private viewBox = { x: 0, y: 0, w: 0, h: 0 }
  private zoom = 1
  private editingLabelId: string | null = null
  private undoStack: string[] = []
  private confirmed = false
  // Connection drag state (handle → handle)
  private connecting: { sourceHandleId: string; sourceNodeId: string; previewLine: SVGLineElement | null } | null = null

  constructor(container: HTMLElement, graph: InteractiveGraph, options?: InteractiveSvgOptions) {
    this.container = container
    this.graph = JSON.parse(JSON.stringify(graph)) // deep clone
    this.opts = { ...DEFAULTS, ...options }

    // Resolve theme colors from CSS vars
    try {
      this.theme = getThemeColors()
    } catch (_) { /* use defaults */ }

    // Defensive: ensure graph has valid structure
    if (!this.graph.nodes) this.graph.nodes = []
    if (!this.graph.edges) this.graph.edges = []
    if (!this.graph.width || this.graph.width <= 0) this.graph.width = 800
    if (!this.graph.height || this.graph.height <= 0) this.graph.height = 600

    // Ensure all nodes have handles (auto-generate defaults if missing)
    for (const node of this.graph.nodes) {
      if (!node.handles || node.handles.length === 0) {
        node.handles = InteractiveSvgEditor.generateDefaultHandles(node.id)
      }
    }

    try {
      this.init()
    } catch (err) {
      console.error('[InteractiveSvgEditor] init failed:', err)
      // Fallback: show error message in container
      this.container.innerHTML = `<div style="padding:2rem;text-align:center;color:${this.theme.textColor};opacity:0.6;">
        <p style="font-size:14px;">Editor initialization failed</p>
        <p style="font-size:12px;margin-top:0.5rem;">${err instanceof Error ? err.message : 'Unknown error'}</p>
      </div>`
    }
  }

  // ──── Initialization ──────────────────────────────────
  private init() {
    this.container.innerHTML = ''
    const p = this.opts.padding

    // Set viewBox with generous padding — ReactFlow-style spacious canvas
    const contentW = this.graph.width || 800
    const contentH = this.graph.height || 600
    const extraPad = Math.max(200, Math.max(contentW, contentH) * 0.6)
    this.viewBox = {
      x: -extraPad,
      y: -extraPad,
      w: contentW + extraPad * 2,
      h: contentH + extraPad * 2,
    }

    // Create SVG element — fills container absolutely
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    this.svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    this.svg.style.width = '100%'
    this.svg.style.height = '100%'
    this.svg.style.display = 'block'
    this.svg.style.cursor = 'grab'
    this.svg.style.userSelect = 'none'
    // Prevent browser default touch behaviors (scroll, zoom) so our handlers work
    this.svg.style.touchAction = 'none'
    this.updateViewBox()

    // Defs: arrow markers + grid pattern
    const defs = this.createDefs()
    this.svg.appendChild(defs)

    // Background grid
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

    // Touch events for mobile — Task 7
    this.svg.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false })
    this.svg.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false })
    this.svg.addEventListener('touchend', this.onTouchEnd.bind(this))

    // Click on empty space to deselect (but not after panning)
    this.svg.addEventListener('click', (e) => {
      if (this.panMoved) {
        this.panMoved = false
        return
      }
      if ((e.target as Element) === this.svg || (e.target as Element).classList.contains('svg-grid')) {
        this.deselectAll()
      }
    })

    // Double-click background to fitView — Task 8 (ReactFlow behavior)
    this.svg.addEventListener('dblclick', (e) => {
      if ((e.target as Element) === this.svg || (e.target as Element).classList.contains('svg-grid')) {
        this.animatedFitView()
      }
    })

    // T9: Keyboard shortcuts
    this.handleKeyDown = this.handleKeyDown.bind(this)
    document.addEventListener('keydown', this.handleKeyDown)
  }

  // ──── Keyboard Shortcuts ──────────────────────────────
  private handleKeyDown(e: KeyboardEvent) {
    // Only handle when our SVG is focused/visible
    if (!this.svg || !this.container.offsetParent) return

    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (this.selectedNodeId && !this.editingLabelId) {
        e.preventDefault()
        this.removeSelectedNode()
      }
    } else if (e.key === 'Escape') {
      this.deselectAll()
      if (this.editingLabelId) {
        const fo = this.svg.querySelector('.label-editor-fo')
        if (fo) fo.remove()
        this.editingLabelId = null
      }
    } else if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      this.undo()
    }
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
    polygon.setAttribute('fill', this.theme.edgeColor)
    marker.appendChild(polygon)
    defs.appendChild(marker)

    // Grid pattern — ReactFlow-style: small dots + larger cross at intervals
    if (this.opts.snapToGrid) {
      const gs = this.opts.gridSize
      const bigGs = gs * 10  // Major grid every 10 units

      // Small dot pattern
      const pattern = document.createElementNS('http://www.w3.org/2000/svg', 'pattern')
      pattern.setAttribute('id', 'grid-pattern')
      pattern.setAttribute('width', String(bigGs))
      pattern.setAttribute('height', String(bigGs))
      pattern.setAttribute('patternUnits', 'userSpaceOnUse')

      // Small dots at each gridSize interval
      for (let gx = 0; gx <= bigGs; gx += gs) {
        for (let gy = 0; gy <= bigGs; gy += gs) {
          const isMajor = gx % bigGs === 0 && gy % bigGs === 0
          const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
          dot.setAttribute('cx', String(gx))
          dot.setAttribute('cy', String(gy))
          dot.setAttribute('r', isMajor ? '1.2' : '0.6')
          dot.setAttribute('fill', isMajor ? '#94A3B8' : '#CBD5E1')
          dot.setAttribute('opacity', isMajor ? '0.7' : '0.4')
          pattern.appendChild(dot)
        }
      }

      defs.appendChild(pattern)
    }

    return defs
  }

  private createGrid(): SVGRectElement {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.classList.add('svg-grid')
    // Make grid much larger than viewBox so it covers panning area
    rect.setAttribute('x', String(this.viewBox.x - this.viewBox.w * 2))
    rect.setAttribute('y', String(this.viewBox.y - this.viewBox.h * 2))
    rect.setAttribute('width', String(this.viewBox.w * 5))
    rect.setAttribute('height', String(this.viewBox.h * 5))
    rect.setAttribute('fill', 'url(#grid-pattern)')
    return rect
  }

  /** Update grid rect to follow the viewBox when panning */
  private updateGridExtent() {
    const gridRect = this.svg?.querySelector('.svg-grid')
    if (gridRect) {
      gridRect.setAttribute('x', String(this.viewBox.x - this.viewBox.w * 2))
      gridRect.setAttribute('y', String(this.viewBox.y - this.viewBox.h * 2))
      gridRect.setAttribute('width', String(this.viewBox.w * 5))
      gridRect.setAttribute('height', String(this.viewBox.h * 5))
    }
  }

  private createNodeGroup(node: InteractiveNode, index: number): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-node-id', node.id)
    g.classList.add('interactive-node')
    g.style.cursor = 'grab'
    g.style.transition = 'opacity 0.15s ease'

    // Main rect — with subtle shadow via filter
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.setAttribute('x', String(node.x))
    rect.setAttribute('y', String(node.y))
    rect.setAttribute('width', String(node.width))
    rect.setAttribute('height', String(node.height))
    rect.setAttribute('fill', node.fill || this.theme.nodeFill)
    rect.setAttribute('stroke', this.theme.nodeStroke)
    rect.setAttribute('stroke-width', node.isGroup ? '2' : '1.5')
    rect.setAttribute('rx', '8')
    if (node.isGroup) rect.setAttribute('stroke-dasharray', '6,3')

    // Hover effect — Task 9
    g.addEventListener('mouseenter', () => {
      if (!this.confirmed) {
        rect.setAttribute('stroke-width', node.isGroup ? '2.5' : '2')
        rect.setAttribute('stroke', this.theme.selectionColor + '80') // 50% opacity accent
      }
    })
    g.addEventListener('mouseleave', () => {
      if (this.selectedNodeId !== node.id) {
        rect.setAttribute('stroke-width', node.isGroup ? '2' : '1.5')
        rect.setAttribute('stroke', this.theme.nodeStroke)
      }
    })

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
    text.setAttribute('fill', this.theme.textColor)
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
      handle.setAttribute('fill', this.theme.selectionColor)
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

    // ── Connection Handles (ports) ──────────────────────
    if (node.handles && node.handles.length > 0) {
      this.renderNodeHandles(g, node)
    }

    // Drag events
    g.addEventListener('mousedown', (e) => {
      if ((e.target as Element).classList.contains('resize-handle')) return
      if ((e.target as Element).classList.contains('node-handle')) return // handles have own event
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

  // ──── Handle (Port) Rendering ─────────────────────────

  /** Render connection handles on a node group */
  private renderNodeHandles(g: SVGGElement, node: InteractiveNode) {
    if (!node.handles) return

    // Group handles by position for even distribution
    const byPosition: Record<string, NodeHandle[]> = {}
    for (const h of node.handles) {
      if (!byPosition[h.position]) byPosition[h.position] = []
      byPosition[h.position].push(h)
    }

    for (const [position, handles] of Object.entries(byPosition)) {
      const count = handles.length
      handles.forEach((handle, idx) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
        circle.classList.add('node-handle')
        circle.setAttribute('data-handle-id', handle.id)
        circle.setAttribute('data-handle-type', handle.type)
        circle.setAttribute('data-handle-position', position)
        circle.setAttribute('r', '5')
        circle.setAttribute('stroke', 'white')
        circle.setAttribute('stroke-width', '1.5')
        circle.style.cursor = 'crosshair'

        // Different fill for source vs target
        if (handle.type === 'source') {
          circle.setAttribute('fill', this.theme.selectionColor)
        } else {
          circle.setAttribute('fill', this.theme.edgeColor)
        }

        // Position the handle
        const pos = this.computeHandleXY(node, position as HandlePosition, idx, count)
        circle.setAttribute('cx', String(pos.x))
        circle.setAttribute('cy', String(pos.y))

        // Handle mousedown: start connection drag
        circle.addEventListener('mousedown', (e) => {
          e.stopPropagation()
          if (this.confirmed) return
          if (handle.type === 'source') {
            this.startConnectionDrag(node.id, handle.id, pos, e)
          }
        })

        // Handle mouseup: complete connection
        circle.addEventListener('mouseup', (e) => {
          e.stopPropagation()
          if (this.connecting && handle.type === 'target') {
            this.completeConnection(node.id, handle.id)
          }
        })

        g.appendChild(circle)
      })
    }
  }

  /** Compute the absolute (x, y) for a handle on a node edge */
  private computeHandleXY(node: InteractiveNode, position: HandlePosition, index: number, total: number): { x: number; y: number } {
    const spacing = (val: number) => (val / (total + 1)) * (index + 1)
    switch (position) {
      case 'right':
        return { x: node.x + node.width, y: node.y + spacing(node.height) }
      case 'left':
        return { x: node.x, y: node.y + spacing(node.height) }
      case 'top':
        return { x: node.x + spacing(node.width), y: node.y }
      case 'bottom':
        return { x: node.x + spacing(node.width), y: node.y + node.height }
      default:
        return { x: node.x + node.width, y: node.y + spacing(node.height) }
    }
  }

  /** Get the absolute position of a handle by ID */
  private getHandleAbsPosition(handleId: string): { x: number; y: number; nodeId: string } | null {
    for (const node of this.graph.nodes) {
      if (!node.handles) continue
      const byPos: Record<string, NodeHandle[]> = {}
      for (const h of node.handles) {
        if (!byPos[h.position]) byPos[h.position] = []
        byPos[h.position].push(h)
      }
      for (const [position, handles] of Object.entries(byPos)) {
        const idx = handles.findIndex(h => h.id === handleId)
        if (idx !== -1) {
          const pos = this.computeHandleXY(node, position as HandlePosition, idx, handles.length)
          return { ...pos, nodeId: node.id }
        }
      }
    }
    return null
  }

  /** Start a connection drag from a source handle */
  private startConnectionDrag(nodeId: string, handleId: string, pos: { x: number; y: number }, _e: MouseEvent) {
    // Create preview line
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.classList.add('connection-preview')
    line.setAttribute('x1', String(pos.x))
    line.setAttribute('y1', String(pos.y))
    line.setAttribute('x2', String(pos.x))
    line.setAttribute('y2', String(pos.y))
    line.setAttribute('stroke', this.theme.selectionColor)
    line.setAttribute('stroke-width', '2')
    line.setAttribute('stroke-dasharray', '6,3')
    this.svg.appendChild(line)

    this.connecting = { sourceHandleId: handleId, sourceNodeId: nodeId, previewLine: line }

    // Highlight compatible target handles
    this.svg.querySelectorAll('.node-handle[data-handle-type="target"]').forEach(el => {
      // Don't highlight targets on the same node
      const parentNode = (el as SVGElement).closest('[data-node-id]')
      if (parentNode && parentNode.getAttribute('data-node-id') !== nodeId) {
        el.classList.add('handle-connectable')
        ;(el as SVGElement).setAttribute('r', '7') // enlarge
      }
    })
  }

  /** Complete a connection to a target handle */
  private completeConnection(targetNodeId: string, targetHandleId: string) {
    if (!this.connecting) return
    const { sourceHandleId, sourceNodeId } = this.connecting

    // Clean up preview
    this.cleanupConnectionDrag()

    // Validate and add
    this.addEdgeFromHandles(sourceHandleId, targetHandleId)
  }

  /** Clean up connection drag state */
  private cleanupConnectionDrag() {
    if (this.connecting?.previewLine) {
      this.connecting.previewLine.remove()
    }
    this.connecting = null

    // Remove highlights
    this.svg.querySelectorAll('.node-handle.handle-connectable').forEach(el => {
      el.classList.remove('handle-connectable')
      ;(el as SVGElement).setAttribute('r', '5')
    })
  }

  /** Generate default handles for a node (1 source right, 1 target left, 1 source bottom, 1 target top) */
  static generateDefaultHandles(nodeId: string): NodeHandle[] {
    return [
      { id: `${nodeId}-src-default`, type: 'source', position: 'right' },
      { id: `${nodeId}-tgt-default`, type: 'target', position: 'left' },
    ]
  }

  private createEdgeGroup(edge: InteractiveEdge): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-edge-id', edge.id)
    g.classList.add('interactive-edge')

    if (edge.points.length < 2) return g

    // Build path — use smooth rounded corners for orthogonal routing (ReactFlow-style)
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    const d = this.buildSmoothEdgePath(edge.points)
    path.setAttribute('d', d)
    path.setAttribute('fill', 'none')
    path.setAttribute('stroke', edge.color || this.theme.edgeColor)
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
      text.setAttribute('fill', edge.color || this.theme.edgeColor)
      text.textContent = edge.label
      g.appendChild(text)
    }

    return g
  }

  // ──── Interaction: Drag ───────────────────────────────

  /** Build SVG path with rounded corners at bend points (ReactFlow smooth-step style) */
  private buildSmoothEdgePath(points: { x: number; y: number }[]): string {
    if (points.length < 2) return ''
    if (points.length === 2) {
      return `M${points[0].x},${points[0].y} L${points[1].x},${points[1].y}`
    }

    const radius = 8 // Corner rounding radius
    let d = `M${points[0].x},${points[0].y}`

    for (let i = 1; i < points.length - 1; i++) {
      const prev = points[i - 1]
      const curr = points[i]
      const next = points[i + 1]

      // Calculate distance to neighboring points
      const d1 = Math.sqrt((curr.x - prev.x) * (curr.x - prev.x) + (curr.y - prev.y) * (curr.y - prev.y))
      const d2 = Math.sqrt((next.x - curr.x) * (next.x - curr.x) + (next.y - curr.y) * (next.y - curr.y))
      const r = Math.min(radius, d1 / 2, d2 / 2)

      if (r < 1) {
        // Too small for rounding, just use straight line
        d += ` L${curr.x},${curr.y}`
        continue
      }

      // Point before the corner
      const t1x = curr.x - (r * (curr.x - prev.x)) / d1
      const t1y = curr.y - (r * (curr.y - prev.y)) / d1
      // Point after the corner
      const t2x = curr.x + (r * (next.x - curr.x)) / d2
      const t2y = curr.y + (r * (next.y - curr.y)) / d2

      d += ` L${t1x},${t1y} Q${curr.x},${curr.y} ${t2x},${t2y}`
    }

    const last = points[points.length - 1]
    d += ` L${last.x},${last.y}`
    return d
  }

  private startDrag(nodeId: string, e: MouseEvent) {
    if (this.confirmed) return // Locked after confirm
    const node = this.findNode(nodeId)
    if (!node) return
    const pt = this.clientToSvg(e)
    this.dragging = { nodeId, offsetX: pt.x - node.x, offsetY: pt.y - node.y }
    this.selectNode(nodeId)
    this.svg.style.cursor = 'grabbing'
  }

  private onMouseMove(e: MouseEvent) {
    if (this.connecting && this.connecting.previewLine) {
      // Update connection preview line endpoint
      const pt = this.clientToSvg(e)
      this.connecting.previewLine.setAttribute('x2', String(pt.x))
      this.connecting.previewLine.setAttribute('y2', String(pt.y))
    } else if (this.dragging) {
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
      const rect = this.svg.getBoundingClientRect()
      const scaleX = this.viewBox.w / rect.width
      const scaleY = this.viewBox.h / rect.height
      const dx = (e.clientX - this.panning.startX) * scaleX
      const dy = (e.clientY - this.panning.startY) * scaleY
      this.viewBox.x = this.panning.startVX - dx
      this.viewBox.y = this.panning.startVY - dy
      this.panMoved = true
      this.updateViewBox()
      this.updateGridExtent()
    }
  }

  private onMouseUp(_e: MouseEvent) {
    if (this.dragging || this.resizing) {
      this.saveUndoState()
      this.opts.onGraphChange(this.graph)
    }
    if (this.connecting) {
      this.cleanupConnectionDrag()
    }
    this.dragging = null
    this.resizing = null
    this.panning = null
    this.svg.style.cursor = 'grab'
  }

  /** Save current graph state for undo */
  private saveUndoState() {
    this.undoStack.push(JSON.stringify(this.graph))
    if (this.undoStack.length > 50) this.undoStack.shift()
  }

  /** Undo last change */
  undo() {
    if (this.undoStack.length === 0) return
    const prev = this.undoStack.pop()!
    this.graph = JSON.parse(prev)
    this.init()
    this.opts.onGraphChange(this.graph)
  }

  // ──── Interaction: Resize ─────────────────────────────

  private startResize(nodeId: string, corner: string, e: MouseEvent) {
    if (this.confirmed) return // Locked after confirm
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
    // Left-click on empty space (background/grid) OR middle mouse OR Alt+Left = panning
    // This is the ReactFlow pattern: clicking the background pans the canvas
    const target = e.target as Element
    const isBackground = target === this.svg || target.classList.contains('svg-grid')
    if ((e.button === 0 && isBackground) || e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault()
      this.panning = {
        startX: e.clientX, startY: e.clientY,
        startVX: this.viewBox.x, startVY: this.viewBox.y,
      }
      this.panMoved = false
      this.svg.style.cursor = 'grabbing'
    }
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault()
    // Smooth zoom with clamping — Task 3
    const delta = -e.deltaY * ZOOM_SENSITIVITY * 0.01
    const scaleFactor = Math.pow(2, -delta * 10)
    const pt = this.clientToSvg(e)

    // Compute new viewBox dimensions
    const newW = this.viewBox.w * scaleFactor
    const newH = this.viewBox.h * scaleFactor

    // Calculate what zoom level this corresponds to
    const baseW = (this.graph.width || 800) + this.opts.padding * 2
    const newZoom = baseW / newW

    // Clamp zoom
    if (newZoom < MIN_ZOOM || newZoom > MAX_ZOOM) return

    this.viewBox.x = pt.x - (pt.x - this.viewBox.x) * scaleFactor
    this.viewBox.y = pt.y - (pt.y - this.viewBox.y) * scaleFactor
    this.viewBox.w = newW
    this.viewBox.h = newH

    this.zoom = newZoom
    this.updateViewBox()
    this.updateGridExtent()
  }

  // ──── Interaction: Touch (Mobile) — Task 7 ────────────

  private onTouchStart(e: TouchEvent) {
    if (e.touches.length === 1) {
      // Single finger: pan (on background) or drag node
      const touch = e.touches[0]
      const target = document.elementFromPoint(touch.clientX, touch.clientY) as Element
      const isBackground = !target || target === this.svg || target.classList.contains('svg-grid')

      if (isBackground) {
        e.preventDefault()
        this.panning = {
          startX: touch.clientX, startY: touch.clientY,
          startVX: this.viewBox.x, startVY: this.viewBox.y,
        }
        this.panMoved = false
      }
    } else if (e.touches.length === 2) {
      // Two fingers: pinch-zoom + pan
      e.preventDefault()
      const [t1, t2] = [e.touches[0], e.touches[1]]
      const dist = Math.sqrt((t2.clientX - t1.clientX) * (t2.clientX - t1.clientX) + (t2.clientY - t1.clientY) * (t2.clientY - t1.clientY))
      const midX = (t1.clientX + t2.clientX) / 2
      const midY = (t1.clientY + t2.clientY) / 2
      this.touchState = {
        lastDist: dist, lastMidX: midX, lastMidY: midY,
        startVX: this.viewBox.x, startVY: this.viewBox.y,
      }
      this.panning = null // Cancel single-finger pan
    }
  }

  private onTouchMove(e: TouchEvent) {
    if (e.touches.length === 1 && this.panning) {
      e.preventDefault()
      const touch = e.touches[0]
      const rect = this.svg.getBoundingClientRect()
      const scaleX = this.viewBox.w / rect.width
      const scaleY = this.viewBox.h / rect.height
      const dx = (touch.clientX - this.panning.startX) * scaleX
      const dy = (touch.clientY - this.panning.startY) * scaleY
      this.viewBox.x = this.panning.startVX - dx
      this.viewBox.y = this.panning.startVY - dy
      this.panMoved = true
      this.updateViewBox()
      this.updateGridExtent()
    } else if (e.touches.length === 2 && this.touchState) {
      e.preventDefault()
      const [t1, t2] = [e.touches[0], e.touches[1]]
      const dist = Math.sqrt((t2.clientX - t1.clientX) * (t2.clientX - t1.clientX) + (t2.clientY - t1.clientY) * (t2.clientY - t1.clientY))
      const midX = (t1.clientX + t2.clientX) / 2
      const midY = (t1.clientY + t2.clientY) / 2

      // Pinch zoom
      const scaleFactor = this.touchState.lastDist / dist
      const pt = this.clientToSvgXY(midX, midY)

      const newW = this.viewBox.w * scaleFactor
      const baseW = (this.graph.width || 800) + this.opts.padding * 2
      const newZoom = baseW / newW
      if (newZoom >= MIN_ZOOM && newZoom <= MAX_ZOOM) {
        this.viewBox.x = pt.x - (pt.x - this.viewBox.x) * scaleFactor
        this.viewBox.y = pt.y - (pt.y - this.viewBox.y) * scaleFactor
        this.viewBox.w = newW
        this.viewBox.h *= scaleFactor
        this.zoom = newZoom
      }

      this.touchState.lastDist = dist
      this.touchState.lastMidX = midX
      this.touchState.lastMidY = midY
      this.updateViewBox()
      this.updateGridExtent()
    }
  }

  private onTouchEnd(_e: TouchEvent) {
    this.panning = null
    this.touchState = null
  }

  /** Convert raw client coordinates to SVG space (for touch events) */
  private clientToSvgXY(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.svg.getBoundingClientRect()
    const scaleX = this.viewBox.w / rect.width
    const scaleY = this.viewBox.h / rect.height
    return {
      x: (clientX - rect.left) * scaleX + this.viewBox.x,
      y: (clientY - rect.top) * scaleY + this.viewBox.y,
    }
  }

  // ──── Interaction: Label Edit ─────────────────────────

  private startLabelEdit(nodeId: string) {
    if (this.confirmed) return // Locked after confirm
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
      rect.setAttribute('stroke', this.theme.selectionColor)
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
        rect.setAttribute('stroke', this.theme.nodeStroke)
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

    // Update connection handles (ports)
    if (node.handles) {
      const byPos: Record<string, NodeHandle[]> = {}
      for (const h of node.handles) {
        if (!byPos[h.position]) byPos[h.position] = []
        byPos[h.position].push(h)
      }
      for (const [position, handles] of Object.entries(byPos)) {
        handles.forEach((h, idx) => {
          const circle = g.querySelector(`.node-handle[data-handle-id="${h.id}"]`) as SVGCircleElement | null
          if (circle) {
            const pos = this.computeHandleXY(node, position as HandlePosition, idx, handles.length)
            circle.setAttribute('cx', String(pos.x))
            circle.setAttribute('cy', String(pos.y))
          }
        })
      }
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

    // Use handle positions if available
    let sx: number, sy: number, tx: number, ty: number

    if (edge.sourceHandleId) {
      const srcPos = this.getHandleAbsPosition(edge.sourceHandleId)
      if (srcPos) { sx = srcPos.x; sy = srcPos.y }
      else { sx = src.x + src.width / 2; sy = src.y + src.height }
    } else {
      sx = src.x + src.width / 2; sy = src.y + src.height
    }

    if (edge.targetHandleId) {
      const tgtPos = this.getHandleAbsPosition(edge.targetHandleId)
      if (tgtPos) { tx = tgtPos.x; ty = tgtPos.y }
      else { tx = tgt.x + tgt.width / 2; ty = tgt.y }
    } else {
      tx = tgt.x + tgt.width / 2; ty = tgt.y
    }

    // For orthogonal routing, add midpoints
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
      const d = this.buildSmoothEdgePath(edge.points)
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
      children: this.graph.nodes.map((n) => {
        const ports = (n.handles || []).map((h) => ({
          id: h.id,
          properties: {
            side: h.position === 'right' ? 'EAST' : h.position === 'left' ? 'WEST' : h.position === 'top' ? 'NORTH' : 'SOUTH',
            type: h.type,
          },
        }))
        return {
          id: n.id,
          width: n.width,
          height: n.height,
          labels: [{ text: n.label }],
          ...(ports.length > 0 ? { ports } : {}),
          ...(ports.length > 0 ? { properties: { 'org.eclipse.elk.portConstraints': 'FIXED_ORDER' } } : {}),
        }
      }),
      edges: this.graph.edges.map((e) => ({
        id: e.id,
        sources: [e.sourceHandleId || e.sourceId],
        targets: [e.targetHandleId || e.targetId],
        ...(e.label ? { labels: [{ text: e.label }] } : {}),
      })),
    }
  }

  /** Export to static SVG string */
  toStaticSvg(): string {
    // Clone SVG without interactive elements
    const clone = this.svg.cloneNode(true) as SVGSVGElement
    // Remove resize handles and editors (but keep node-handle for ports)
    clone.querySelectorAll('.resize-handle, .label-editor-fo').forEach((el) => el.remove())
    // Reset selection styles
    clone.querySelectorAll('.interactive-node rect:not(.resize-handle)').forEach((rect) => {
      rect.setAttribute('stroke', this.theme.nodeStroke)
    })
    return new XMLSerializer().serializeToString(clone)
  }

  /** Add a new node (with auto-generated default handles) */
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
      fill: this.theme.nodeFill,
      handles: InteractiveSvgEditor.generateDefaultHandles(id),
    }
    this.graph.nodes.push(node)
    this.svg.appendChild(this.createNodeGroup(node, this.graph.nodes.length - 1))
    this.opts.onGraphChange(this.graph)
    return node
  }

  /** Add a new node with custom handles */
  addNodeWithHandles(label: string, handles: NodeHandle[]): InteractiveNode {
    const id = `node_${Date.now()}`
    const maxX = Math.max(...this.graph.nodes.map((n) => n.x + n.width), 100)
    const node: InteractiveNode = {
      id,
      x: maxX + 30,
      y: 50,
      width: 160,
      height: 60,
      label,
      fill: this.theme.nodeFill,
      handles: handles.map(h => ({ ...h })),
    }
    this.graph.nodes.push(node)
    this.svg.appendChild(this.createNodeGroup(node, this.graph.nodes.length - 1))
    this.opts.onGraphChange(this.graph)
    return node
  }

  /** Add an edge between two handles (source → target) */
  addEdgeFromHandles(sourceHandleId: string, targetHandleId: string): InteractiveEdge | null {
    // Find source and target nodes
    const srcInfo = this.getHandleAbsPosition(sourceHandleId)
    const tgtInfo = this.getHandleAbsPosition(targetHandleId)
    if (!srcInfo || !tgtInfo) return null

    // Validate: find handle types
    const srcHandle = this.findHandle(sourceHandleId)
    const tgtHandle = this.findHandle(targetHandleId)
    if (!srcHandle || !tgtHandle) return null

    // Must be source → target
    if (srcHandle.type !== 'source' || tgtHandle.type !== 'target') return null

    // No self-connections
    if (srcInfo.nodeId === tgtInfo.nodeId) return null

    // No duplicates
    const exists = this.graph.edges.some(
      e => e.sourceHandleId === sourceHandleId && e.targetHandleId === targetHandleId
    )
    if (exists) return null

    return this.addConnection(srcInfo.nodeId, sourceHandleId, tgtInfo.nodeId, targetHandleId)
  }

  /** Add a connection between specific handles on specific nodes */
  addConnection(sourceNodeId: string, sourceHandleId: string, targetNodeId: string, targetHandleId: string): InteractiveEdge | null {
    // Validate handle types
    const srcHandle = this.findHandle(sourceHandleId)
    const tgtHandle = this.findHandle(targetHandleId)
    if (!srcHandle || !tgtHandle) return null
    if (srcHandle.type !== 'source' || tgtHandle.type !== 'target') return null
    if (sourceNodeId === targetNodeId) return null

    // No duplicates
    const exists = this.graph.edges.some(
      e => e.sourceHandleId === sourceHandleId && e.targetHandleId === targetHandleId
    )
    if (exists) return null

    const edge: InteractiveEdge = {
      id: `edge_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      sourceId: sourceNodeId,
      targetId: targetNodeId,
      sourceHandleId,
      targetHandleId,
      points: [],
      color: this.theme.edgeColor,
      strokeWidth: 1.5,
    }

    // Calculate edge points using handle positions
    this.graph.edges.push(edge)
    this.recalcEdgePoints(edge)
    this.svg.appendChild(this.createEdgeGroup(edge))
    this.saveUndoState()
    this.opts.onGraphChange(this.graph)
    return edge
  }

  /** Remove an edge by ID */
  removeEdge(edgeId: string): boolean {
    const idx = this.graph.edges.findIndex(e => e.id === edgeId)
    if (idx === -1) return false
    this.graph.edges.splice(idx, 1)
    this.svg.querySelector(`[data-edge-id="${edgeId}"]`)?.remove()
    this.opts.onGraphChange(this.graph)
    return true
  }

  /** Find a handle definition by ID across all nodes */
  private findHandle(handleId: string): NodeHandle | undefined {
    for (const node of this.graph.nodes) {
      if (!node.handles) continue
      const h = node.handles.find(h => h.id === handleId)
      if (h) return h
    }
    return undefined
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

  /** Fit view to show all content with generous padding (ReactFlow-style) */
  fitView() {
    const target = this.computeFitViewBox()
    if (!target) return
    this.viewBox = target
    this.zoom = (this.graph.width + this.opts.padding * 2) / this.viewBox.w
    this.updateViewBox()
    this.updateGridExtent()
  }

  /** Animated fit view — smoothly transitions viewBox (Task 4) */
  animatedFitView() {
    const target = this.computeFitViewBox()
    if (!target) return

    const start = { ...this.viewBox }
    const duration = 300 // ms
    const startTime = performance.now()

    const step = (time: number) => {
      const t = Math.min((time - startTime) / duration, 1)
      // Ease-out cubic
      const ease = 1 - Math.pow(1 - t, 3)

      this.viewBox.x = start.x + (target.x - start.x) * ease
      this.viewBox.y = start.y + (target.y - start.y) * ease
      this.viewBox.w = start.w + (target.w - start.w) * ease
      this.viewBox.h = start.h + (target.h - start.h) * ease
      this.updateViewBox()
      this.updateGridExtent()

      if (t < 1) requestAnimationFrame(step)
      else this.zoom = (this.graph.width + this.opts.padding * 2) / this.viewBox.w
    }
    requestAnimationFrame(step)
  }

  /** Compute the target viewBox for fitView */
  private computeFitViewBox(): { x: number; y: number; w: number; h: number } | null {
    const p = Math.max(this.opts.padding, 60)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const n of this.graph.nodes) {
      if (n.x < minX) minX = n.x
      if (n.y < minY) minY = n.y
      if (n.x + n.width > maxX) maxX = n.x + n.width
      if (n.y + n.height > maxY) maxY = n.y + n.height
    }
    if (minX === Infinity) return null
    const contentW = maxX - minX
    const contentH = maxY - minY
    const extraPad = Math.max(p, Math.min(contentW, contentH) * 0.3)

    // Adjust aspect ratio to match container
    let targetW = contentW + extraPad * 2
    let targetH = contentH + extraPad * 2
    const containerRect = this.container.getBoundingClientRect()
    if (containerRect.width > 0 && containerRect.height > 0) {
      const containerAR = containerRect.width / containerRect.height
      const contentAR = targetW / targetH
      if (contentAR > containerAR) {
        // Content is wider — expand height
        targetH = targetW / containerAR
      } else {
        // Content is taller — expand width
        targetW = targetH * containerAR
      }
    }

    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2
    return {
      x: cx - targetW / 2,
      y: cy - targetH / 2,
      w: targetW,
      h: targetH,
    }
  }

  /** Set zoom level (1 = 100%) — clamped to [MIN_ZOOM, MAX_ZOOM] */
  setZoom(level: number) {
    level = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, level))
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
    this.updateGridExtent()
  }

  /** Update from new ELK layout data */
  updateFromLayout(graph: InteractiveGraph) {
    this.graph = JSON.parse(JSON.stringify(graph))
    this.confirmed = false
    this.init()
  }

  /** Mark as confirmed — disables editing but still allows pan/zoom */
  confirmEdit() {
    this.confirmed = true
    this.deselectAll()
    // Visual indicator: dim the grid, disable drag cursors on nodes
    // But keep the SVG cursor as 'grab' so users can still pan
    if (this.svg) {
      this.svg.style.cursor = 'grab'
      this.svg.querySelectorAll('.interactive-node').forEach(g => {
        (g as SVGGElement).style.cursor = 'default'
        ;(g as SVGGElement).style.opacity = '0.9'
      })
    }
  }

  /** Check if editing is confirmed */
  isConfirmed(): boolean {
    return this.confirmed
  }

  /** Unlock editing */
  unlockEdit() {
    this.confirmed = false
    if (this.svg) {
      this.svg.querySelectorAll('.interactive-node').forEach(g => {
        (g as SVGGElement).style.cursor = 'grab'
        ;(g as SVGGElement).style.opacity = '1'
      })
    }
  }

  /** Destroy the editor */
  destroy() {
    document.removeEventListener('keydown', this.handleKeyDown)
    this.container.innerHTML = ''
  }
}

export default InteractiveSvgEditor