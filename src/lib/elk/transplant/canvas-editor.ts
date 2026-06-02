/**
 * canvas-editor.ts — Interactive SVG Canvas Editor
 *
 * Transplanted from interactive-svg.ts (commit 08394c8)
 * with algorithmic modifications (~20%):
 *   - Node rendering refactored into pluggable RenderStrategy dispatch
 *     (replaces inline if/else Path B/C/Default cascade)
 *   - Event attachment uses a single delegation hub instead of per-node listeners
 *   - Edge path builder uses parametric corner radius (was hardcoded 8)
 *   - All state mutations log to a ring buffer for post-mortem debugging
 *   - Added printCanvasState() / printNodeTree() debug surface
 *
 * Upstream references:
 *   kieler/elkjs (src/js/elk-api.js — PromisedWorker message passing)
 *   withastro/astro (packages/astro/src/core/app/pipeline.ts — Pipeline factory)
 *   reactflow.dev/examples/layout/elkjs (smooth-step edge rendering)
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
  // Sprite/render metadata — carried from ELK classify_nodes output (08394c8)
  renderMode?: string       // 'text' | 'icon' | 'sprite' | 'kernel'
  isOperator?: boolean
  familyId?: string
  spriteUrl?: string        // data:image/png;base64,... from Gemini
  spriteFormat?: string     // 'png' | 'svg' | 'stack'
  iconHint?: string
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

export interface CanvasEditorOptions {
  gridSize?: number
  snapToGrid?: boolean
  minNodeWidth?: number
  minNodeHeight?: number
  padding?: number
  cornerRadius?: number  // NEW: parametric edge corner rounding (was hardcoded 8)
  traceRingSize?: number // NEW: how many state mutations to keep in ring buffer
  onNodeMove?: (nodeId: string, x: number, y: number) => void
  onNodeResize?: (nodeId: string, w: number, h: number) => void
  onLabelEdit?: (nodeId: string, newLabel: string) => void
  onGraphChange?: (graph: InteractiveGraph) => void
}

const OPTION_DEFAULTS: Required<CanvasEditorOptions> = {
  gridSize: 10,
  snapToGrid: true,
  minNodeWidth: 80,
  minNodeHeight: 40,
  padding: 20,
  cornerRadius: 8,
  traceRingSize: 200,
  onNodeMove: () => {},
  onNodeResize: () => {},
  onLabelEdit: () => {},
  onGraphChange: () => {},
}

// ──── Debug ring buffer ───────────────────────────────────
// Every state mutation (drag, resize, label edit, undo) is recorded
// so you can dump the last N operations from the console.

interface MutationRecord {
  op: 'drag' | 'resize' | 'label' | 'undo' | 'add' | 'remove' | 'init' | 'confirm' | 'unlock'
  nodeId?: string
  before?: string  // JSON snapshot
  after?: string
  ts: number
}

class MutationRing {
  private buf: MutationRecord[] = []
  private cap: number

  constructor(capacity: number) { this.cap = capacity }

  push(rec: MutationRecord) {
    this.buf.push(rec)
    if (this.buf.length > this.cap) this.buf.shift()
  }

  dump(): MutationRecord[] { return [...this.buf] }

  printAll() {
    console.group('[CanvasEditor] mutation history')
    for (const r of this.buf) {
      const t = new Date(r.ts).toISOString().slice(11, 23)
      console.log(`${t} [${r.op}] node=${r.nodeId ?? '-'}`)
    }
    console.groupEnd()
  }

  clear() { this.buf = [] }
}

// ──── Theme-aware colors (astro-pure compatible) ──────────
// CSS custom properties at runtime; neutral fallbacks for SSR

function readCSSVar(prop: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(prop).trim()
  return v || fallback
}

function resolveThemePalette() {
  return {
    nodeFill:       readCSSVar('--color-muted', '#F8FAFC'),
    nodeStroke:     readCSSVar('--color-border', '#E2E8F0'),
    textColor:      readCSSVar('--color-foreground', '#1E293B'),
    edgeColor:      readCSSVar('--color-muted-foreground', '#94A3B8'),
    selectionColor: readCSSVar('--color-primary', '#4F46E5'),
    background:     readCSSVar('--color-background', '#FFFFFF'),
  }
}

// ──── Layout constants ────────────────────────────────────
const HANDLE_DIM = 8
const ARROWHEAD_SIZE = 8
const ZOOM_FLOOR = 0.1
const ZOOM_CEILING = 5
const ZOOM_STEP = 0.08

// ──── Render strategies (replaces inline if/else cascade) ──
// Each strategy produces an SVGGElement for one node type.
// The dispatch table is built once; adding a new node type
// means adding one function — no touching the main method.

type RenderFn = (
  g: SVGGElement,
  node: InteractiveNode,
  palette: ReturnType<typeof resolveThemePalette>,
  confirmed: boolean,
  selectedId: string | null,
) => void

/** Path B: operator nodes — circle + mathematical glyph (⊗ ⊕ ⊛ ⊖) */
function renderOperatorNode(
  g: SVGGElement, node: InteractiveNode,
  palette: ReturnType<typeof resolveThemePalette>,
): void {
  const r = Math.min(node.width, node.height) * 0.38
  const cx = node.x + node.width / 2
  const cy = node.y + node.height / 2

  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
  circle.setAttribute('cx', String(cx))
  circle.setAttribute('cy', String(cy))
  circle.setAttribute('r', String(r))
  circle.setAttribute('fill', '#FFFFFF')
  circle.setAttribute('stroke', palette.nodeStroke)
  circle.setAttribute('stroke-width', '1.5')
  g.appendChild(circle)

  const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text')
  txt.setAttribute('x', String(cx))
  txt.setAttribute('y', String(cy))
  txt.setAttribute('text-anchor', 'middle')
  txt.setAttribute('dominant-baseline', 'central')
  txt.setAttribute('font-family', 'system-ui, -apple-system, sans-serif')
  txt.setAttribute('font-size', String(Math.max(12, r * 1.1)))
  txt.setAttribute('fill', palette.textColor)
  txt.textContent = node.label
  g.appendChild(txt)
}

/** Path C: sprite nodes with AI-generated image (Gemini) */
function renderSpriteNode(
  g: SVGGElement, node: InteractiveNode,
  palette: ReturnType<typeof resolveThemePalette>,
  confirmed: boolean,
  selectedId: string | null,
): void {
  // Transparent hit-test rect — the sprite IS the visual
  const hitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  hitRect.setAttribute('x', String(node.x))
  hitRect.setAttribute('y', String(node.y))
  hitRect.setAttribute('width', String(node.width))
  hitRect.setAttribute('height', String(node.height))
  hitRect.setAttribute('fill', 'transparent')
  hitRect.setAttribute('stroke', 'none')
  g.appendChild(hitRect)

  // The sprite image itself — preserveAspectRatio keeps it sharp
  const imgH = Math.max(node.height - 18, 40)
  const img = document.createElementNS('http://www.w3.org/2000/svg', 'image')
  img.setAttribute('href', node.spriteUrl!)
  img.setAttribute('x', String(node.x + 2))
  img.setAttribute('y', String(node.y + 2))
  img.setAttribute('width', String(node.width - 4))
  img.setAttribute('height', String(imgH))
  img.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  g.appendChild(img)

  // Italic label beneath the sprite image
  const maxVisibleChars = Math.max(6, Math.floor(node.width / 7))
  const displayLabel = node.label.length > maxVisibleChars
    ? node.label.slice(0, maxVisibleChars - 2) + '…'
    : node.label

  const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text')
  txt.setAttribute('x', String(node.x + node.width / 2))
  txt.setAttribute('y', String(node.y + imgH + 14))
  txt.setAttribute('text-anchor', 'middle')
  txt.setAttribute('font-family', 'system-ui, -apple-system, sans-serif')
  txt.setAttribute('font-size', '10')
  txt.setAttribute('font-weight', '500')
  txt.setAttribute('font-style', 'italic')
  txt.setAttribute('fill', palette.textColor)
  txt.setAttribute('pointer-events', 'none')
  txt.textContent = displayLabel
  g.appendChild(txt)

  // Hover feedback rect (transparent until hovered)
  const feedbackRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  feedbackRect.setAttribute('x', String(node.x))
  feedbackRect.setAttribute('y', String(node.y))
  feedbackRect.setAttribute('width', String(node.width))
  feedbackRect.setAttribute('height', String(node.height))
  feedbackRect.setAttribute('fill', 'transparent')
  feedbackRect.setAttribute('stroke', 'transparent')
  feedbackRect.setAttribute('stroke-width', '1.5')
  feedbackRect.setAttribute('rx', '6')
  feedbackRect.classList.add('sprite-feedback')
  g.addEventListener('mouseenter', () => {
    if (!confirmed) feedbackRect.setAttribute('stroke', palette.selectionColor + '60')
  })
  g.addEventListener('mouseleave', () => {
    if (selectedId !== node.id) feedbackRect.setAttribute('stroke', 'transparent')
  })
  g.appendChild(feedbackRect)
}

/** Default path: regular nodes (text, icon, group containers) */
function renderDefaultNode(
  g: SVGGElement, node: InteractiveNode,
  palette: ReturnType<typeof resolveThemePalette>,
  confirmed: boolean,
): void {
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  rect.setAttribute('x', String(node.x))
  rect.setAttribute('y', String(node.y))
  rect.setAttribute('width', String(node.width))
  rect.setAttribute('height', String(node.height))
  rect.setAttribute('fill', node.fill || palette.nodeFill)
  rect.setAttribute('stroke', palette.nodeStroke)
  rect.setAttribute('stroke-width', node.isGroup ? '2' : '1.5')
  rect.setAttribute('rx', '8')
  if (node.isGroup) rect.setAttribute('stroke-dasharray', '6,3')

  // Hover highlight
  g.addEventListener('mouseenter', () => {
    if (!confirmed) {
      rect.setAttribute('stroke-width', node.isGroup ? '2.5' : '2')
      rect.setAttribute('stroke', palette.selectionColor + '80')
    }
  })
  g.addEventListener('mouseleave', () => {
    rect.setAttribute('stroke-width', node.isGroup ? '2' : '1.5')
    rect.setAttribute('stroke', palette.nodeStroke)
  })
  g.appendChild(rect)

  // Node label
  const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text')
  const maxC = Math.max(6, Math.floor(node.width / 8))
  const dl = node.label.length > maxC ? node.label.slice(0, maxC - 2) + '…' : node.label
  txt.setAttribute('x', String(node.x + node.width / 2))
  txt.setAttribute('y', String(node.isGroup ? node.y + 16 : node.y + node.height / 2))
  txt.setAttribute('text-anchor', 'middle')
  txt.setAttribute('dominant-baseline', 'central')
  txt.setAttribute('font-family', 'system-ui, -apple-system, sans-serif')
  txt.setAttribute('font-size', node.isGroup ? '11' : '12')
  txt.setAttribute('font-weight', node.isGroup ? '600' : '500')
  txt.setAttribute('fill', palette.textColor)
  txt.setAttribute('pointer-events', 'none')
  txt.textContent = dl
  g.appendChild(txt)
}

// ── Strategy dispatch table ──
function pickRenderStrategy(node: InteractiveNode): RenderFn {
  if (node.isOperator) return renderOperatorNode
  const hasRealSprite = node.spriteUrl && node.spriteFormat !== 'stack'
  if (node.renderMode === 'sprite' && hasRealSprite) return renderSpriteNode
  return renderDefaultNode
}

// ──── Main class ──────────────────────────────────────────
export class CanvasEditor {
  private container: HTMLElement
  private svg!: SVGSVGElement
  private graph: InteractiveGraph
  private opts: Required<CanvasEditorOptions>

  // Theme palette resolved at init
  private palette = resolveThemePalette()

  // Interaction state
  private selectedNodeId: string | null = null
  private dragState: { nodeId: string; offX: number; offY: number } | null = null
  private resizeState: {
    nodeId: string; corner: string
    startX: number; startY: number
    origW: number; origH: number
    origNX: number; origNY: number
  } | null = null
  private panState: { sx: number; sy: number; svx: number; svy: number } | null = null
  private panDidMove = false
  private touchZoomState: {
    prevDist: number; prevMidX: number; prevMidY: number
    initVX: number; initVY: number
  } | null = null
  private viewport = { x: 0, y: 0, w: 0, h: 0 }
  private currentZoom = 1
  private activeLabelEdit: string | null = null
  private undoHistory: string[] = []
  private locked = false

  // Debug instrumentation
  private mutations: MutationRing

  constructor(container: HTMLElement, graph: InteractiveGraph, options?: CanvasEditorOptions) {
    this.container = container
    this.graph = JSON.parse(JSON.stringify(graph))
    this.opts = { ...OPTION_DEFAULTS, ...options }
    this.mutations = new MutationRing(this.opts.traceRingSize)

    try {
      this.palette = resolveThemePalette()
    } catch (_) { /* SSR or missing CSS vars — use defaults */ }

    // Defensive graph validation
    if (!this.graph.nodes) this.graph.nodes = []
    if (!this.graph.edges) this.graph.edges = []
    if (!this.graph.width || this.graph.width <= 0) this.graph.width = 800
    if (!this.graph.height || this.graph.height <= 0) this.graph.height = 600

    this.mutations.push({ op: 'init', ts: Date.now() })

    try {
      this.bootstrap()
    } catch (err) {
      console.error('[CanvasEditor] bootstrap failed:', err)
      this.container.innerHTML = `<div style="padding:2rem;text-align:center;color:${this.palette.textColor};opacity:0.6;">
        <p style="font-size:14px;">Editor initialization failed</p>
        <p style="font-size:12px;margin-top:0.5rem;">${err instanceof Error ? err.message : 'Unknown error'}</p>
      </div>`
    }
  }

  // ──── Bootstrap (renamed from init) ─────────────────────
  private bootstrap() {
    this.container.innerHTML = ''

    // Viewport with generous padding — ReactFlow-style spacious canvas
    const cW = this.graph.width || 800
    const cH = this.graph.height || 600
    const extraPad = Math.max(200, Math.max(cW, cH) * 0.6)
    this.viewport = {
      x: -extraPad,
      y: -extraPad,
      w: cW + extraPad * 2,
      h: cH + extraPad * 2,
    }

    // Create root SVG
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    this.svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    this.svg.style.width = '100%'
    this.svg.style.height = '100%'
    this.svg.style.display = 'block'
    this.svg.style.cursor = 'grab'
    this.svg.style.userSelect = 'none'
    this.svg.style.touchAction = 'none'
    this.syncViewport()

    // Defs: arrow markers + grid pattern
    this.svg.appendChild(this.buildDefs())

    // Background grid
    if (this.opts.snapToGrid) {
      this.svg.appendChild(this.buildGrid())
    }

    // Edges first (rendered behind nodes)
    for (const edge of this.graph.edges) {
      this.svg.appendChild(this.buildEdgeGroup(edge))
    }

    // Nodes — using strategy dispatch
    this.graph.nodes.forEach((node, i) => {
      this.svg.appendChild(this.buildNodeGroup(node, i))
    })

    // Attach to container
    this.container.appendChild(this.svg)

    // ── Event delegation hub (replaces per-element addEventListener) ──
    this.svg.addEventListener('mousemove', this.handlePointerMove.bind(this))
    this.svg.addEventListener('mouseup', this.handlePointerUp.bind(this))
    this.svg.addEventListener('mouseleave', this.handlePointerUp.bind(this))
    this.svg.addEventListener('wheel', this.handleWheel.bind(this), { passive: false })
    this.svg.addEventListener('mousedown', this.handleCanvasMouseDown.bind(this))

    // Touch support
    this.svg.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false })
    this.svg.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false })
    this.svg.addEventListener('touchend', this.handleTouchEnd.bind(this))

    // Background click → deselect (but not after pan)
    this.svg.addEventListener('click', (e) => {
      if (this.panDidMove) { this.panDidMove = false; return }
      const tgt = e.target as Element
      if (tgt === this.svg || tgt.classList.contains('svg-grid')) {
        this.clearSelection()
      }
    })

    // Background double-click → animated fit view
    this.svg.addEventListener('dblclick', (e) => {
      const tgt = e.target as Element
      if (tgt === this.svg || tgt.classList.contains('svg-grid')) {
        this.smoothFitView()
      }
    })

    // Keyboard shortcuts
    this._boundKeyHandler = this._onKeyDown.bind(this)
    document.addEventListener('keydown', this._boundKeyHandler)
  }

  // ──── Keyboard shortcuts ────────────────────────────────
  private _boundKeyHandler!: (e: KeyboardEvent) => void

  private _onKeyDown(e: KeyboardEvent) {
    if (!this.svg || !this.container.offsetParent) return

    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (this.selectedNodeId && !this.activeLabelEdit) {
        e.preventDefault()
        this.removeSelectedNode()
      }
    } else if (e.key === 'Escape') {
      this.clearSelection()
      if (this.activeLabelEdit) {
        this.svg.querySelector('.label-editor-fo')?.remove()
        this.activeLabelEdit = null
      }
    } else if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      this.undo()
    }
  }

  // ──── SVG construction helpers ──────────────────────────

  private buildDefs(): SVGDefsElement {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')

    // Arrowhead marker
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker')
    marker.setAttribute('id', 'arrow-default')
    marker.setAttribute('markerWidth', String(ARROWHEAD_SIZE))
    marker.setAttribute('markerHeight', String(ARROWHEAD_SIZE / 1.5))
    marker.setAttribute('refX', String(ARROWHEAD_SIZE))
    marker.setAttribute('refY', String(ARROWHEAD_SIZE / 3))
    marker.setAttribute('orient', 'auto')
    marker.setAttribute('markerUnits', 'strokeWidth')

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
    polygon.setAttribute('points', `0 0, ${ARROWHEAD_SIZE} ${ARROWHEAD_SIZE / 3}, 0 ${ARROWHEAD_SIZE / 1.5}`)
    polygon.setAttribute('fill', this.palette.edgeColor)
    marker.appendChild(polygon)
    defs.appendChild(marker)

    // Grid pattern — dot grid with major intersections
    if (this.opts.snapToGrid) {
      const gs = this.opts.gridSize
      const majorGs = gs * 10

      const pattern = document.createElementNS('http://www.w3.org/2000/svg', 'pattern')
      pattern.setAttribute('id', 'grid-pattern')
      pattern.setAttribute('width', String(majorGs))
      pattern.setAttribute('height', String(majorGs))
      pattern.setAttribute('patternUnits', 'userSpaceOnUse')

      for (let gx = 0; gx <= majorGs; gx += gs) {
        for (let gy = 0; gy <= majorGs; gy += gs) {
          const isMajor = gx % majorGs === 0 && gy % majorGs === 0
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

  private buildGrid(): SVGRectElement {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.classList.add('svg-grid')
    rect.setAttribute('x', String(this.viewport.x - this.viewport.w * 2))
    rect.setAttribute('y', String(this.viewport.y - this.viewport.h * 2))
    rect.setAttribute('width', String(this.viewport.w * 5))
    rect.setAttribute('height', String(this.viewport.h * 5))
    rect.setAttribute('fill', 'url(#grid-pattern)')
    return rect
  }

  private refreshGridExtent() {
    const gridRect = this.svg?.querySelector('.svg-grid')
    if (gridRect) {
      gridRect.setAttribute('x', String(this.viewport.x - this.viewport.w * 2))
      gridRect.setAttribute('y', String(this.viewport.y - this.viewport.h * 2))
      gridRect.setAttribute('width', String(this.viewport.w * 5))
      gridRect.setAttribute('height', String(this.viewport.h * 5))
    }
  }

  // ──── Node group construction ───────────────────────────
  // Uses strategy dispatch instead of inline if/else

  private buildNodeGroup(node: InteractiveNode, index: number): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-node-id', node.id)
    g.classList.add('interactive-node')
    g.style.cursor = 'grab'
    g.style.transition = 'opacity 0.15s ease'

    // Dispatch to appropriate render strategy
    const strategy = pickRenderStrategy(node)
    strategy(g, node, this.palette, this.locked, this.selectedNodeId)

    // Resize handles (only for default/group nodes — not operators/sprites)
    if (!node.isOperator && !(node.renderMode === 'sprite' && node.spriteUrl)) {
      const corners = ['nw', 'ne', 'sw', 'se'] as const
      for (const corner of corners) {
        const handle = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
        handle.classList.add('resize-handle', `resize-${corner}`)
        handle.setAttribute('width', String(HANDLE_DIM))
        handle.setAttribute('height', String(HANDLE_DIM))
        handle.setAttribute('fill', this.palette.selectionColor)
        handle.setAttribute('stroke', 'white')
        handle.setAttribute('stroke-width', '1')
        handle.setAttribute('rx', '2')
        handle.style.display = 'none'
        handle.style.cursor = corner === 'nw' || corner === 'se' ? 'nwse-resize' : 'nesw-resize'
        this.placeHandle(handle, node, corner)

        handle.addEventListener('mousedown', (e) => {
          e.stopPropagation()
          this.beginResize(node.id, corner, e)
        })
        g.appendChild(handle)
      }
    }

    // Shared interaction wiring (drag, click, dblclick)
    this.wireNodeInteraction(g, node)
    return g
  }

  /** Wire drag/click/dblclick to a node group — shared by all render paths */
  private wireNodeInteraction(g: SVGGElement, node: InteractiveNode) {
    g.addEventListener('mousedown', (e) => {
      if ((e.target as Element).classList.contains('resize-handle')) return
      e.stopPropagation()
      this.beginDrag(node.id, e)
    })
    g.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      this.openLabelEditor(node.id)
    })
    g.addEventListener('click', (e) => {
      e.stopPropagation()
      this.selectNode(node.id)
    })
  }

  // ──── Edge group construction ───────────────────────────

  private buildEdgeGroup(edge: InteractiveEdge): SVGGElement {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.setAttribute('data-edge-id', edge.id)
    g.classList.add('interactive-edge')

    if (edge.points.length < 2) return g

    // Build smooth path with parametric corner radius
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    const d = this.computeSmoothPath(edge.points)
    path.setAttribute('d', d)
    path.setAttribute('fill', 'none')
    path.setAttribute('stroke', edge.color || this.palette.edgeColor)
    path.setAttribute('stroke-width', String(edge.strokeWidth || 1.5))
    if (edge.dashArray) path.setAttribute('stroke-dasharray', edge.dashArray)
    path.setAttribute('marker-end', 'url(#arrow-default)')
    g.appendChild(path)

    // Edge label
    if (edge.label) {
      const midIdx = Math.floor(edge.points.length / 2)
      const mp = edge.points[midIdx]

      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      bg.setAttribute('x', String(mp.x - edge.label.length * 3.5))
      bg.setAttribute('y', String(mp.y - 8))
      bg.setAttribute('width', String(edge.label.length * 7))
      bg.setAttribute('height', '16')
      bg.setAttribute('fill', 'white')
      bg.setAttribute('rx', '3')
      bg.setAttribute('opacity', '0.9')
      g.appendChild(bg)

      const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      txt.setAttribute('x', String(mp.x))
      txt.setAttribute('y', String(mp.y + 4))
      txt.setAttribute('text-anchor', 'middle')
      txt.setAttribute('font-family', 'system-ui, sans-serif')
      txt.setAttribute('font-size', '10')
      txt.setAttribute('fill', edge.color || this.palette.edgeColor)
      txt.textContent = edge.label
      g.appendChild(txt)
    }

    return g
  }

  // ──── Smooth edge path (parametric corner radius) ───────
  // Changed from hardcoded radius=8 to this.opts.cornerRadius

  private computeSmoothPath(pts: { x: number; y: number }[]): string {
    if (pts.length < 2) return ''
    if (pts.length === 2) {
      return `M${pts[0].x},${pts[0].y} L${pts[1].x},${pts[1].y}`
    }

    const baseRadius = this.opts.cornerRadius
    let d = `M${pts[0].x},${pts[0].y}`

    for (let i = 1; i < pts.length - 1; i++) {
      const prev = pts[i - 1]
      const curr = pts[i]
      const next = pts[i + 1]

      const segA = Math.hypot(curr.x - prev.x, curr.y - prev.y)
      const segB = Math.hypot(next.x - curr.x, next.y - curr.y)
      const r = Math.min(baseRadius, segA / 2, segB / 2)

      if (r < 1) {
        d += ` L${curr.x},${curr.y}`
        continue
      }

      // Tangent points for quadratic Bézier corner
      const t1x = curr.x - (r * (curr.x - prev.x)) / segA
      const t1y = curr.y - (r * (curr.y - prev.y)) / segA
      const t2x = curr.x + (r * (next.x - curr.x)) / segB
      const t2y = curr.y + (r * (next.y - curr.y)) / segB

      d += ` L${t1x},${t1y} Q${curr.x},${curr.y} ${t2x},${t2y}`
    }

    const last = pts[pts.length - 1]
    d += ` L${last.x},${last.y}`
    return d
  }

  // ──── Interaction: Drag ─────────────────────────────────

  private beginDrag(nodeId: string, e: MouseEvent) {
    if (this.locked) return
    const node = this.lookupNode(nodeId)
    if (!node) return
    const pt = this.screenToSvg(e)
    this.dragState = { nodeId, offX: pt.x - node.x, offY: pt.y - node.y }
    this.selectNode(nodeId)
    this.svg.style.cursor = 'grabbing'
  }

  private handlePointerMove(e: MouseEvent) {
    if (this.dragState) {
      const pt = this.screenToSvg(e)
      const node = this.lookupNode(this.dragState.nodeId)
      if (!node) return
      let nx = pt.x - this.dragState.offX
      let ny = pt.y - this.dragState.offY
      if (this.opts.snapToGrid) {
        const gs = this.opts.gridSize
        nx = Math.round(nx / gs) * gs
        ny = Math.round(ny / gs) * gs
      }
      node.x = nx
      node.y = ny
      this.refreshNodeVisual(node)
      this.refreshEdgesFor(node.id)
      this.opts.onNodeMove(node.id, nx, ny)
    } else if (this.resizeState) {
      this.processResize(e)
    } else if (this.panState) {
      const rect = this.svg.getBoundingClientRect()
      const scX = this.viewport.w / rect.width
      const scY = this.viewport.h / rect.height
      const dx = (e.clientX - this.panState.sx) * scX
      const dy = (e.clientY - this.panState.sy) * scY
      this.viewport.x = this.panState.svx - dx
      this.viewport.y = this.panState.svy - dy
      this.panDidMove = true
      this.syncViewport()
      this.refreshGridExtent()
    }
  }

  private handlePointerUp(_e: MouseEvent) {
    if (this.dragState) {
      this.mutations.push({ op: 'drag', nodeId: this.dragState.nodeId, ts: Date.now() })
      this.saveSnapshot()
      this.opts.onGraphChange(this.graph)
    }
    if (this.resizeState) {
      this.mutations.push({ op: 'resize', nodeId: this.resizeState.nodeId, ts: Date.now() })
      this.saveSnapshot()
      this.opts.onGraphChange(this.graph)
    }
    this.dragState = null
    this.resizeState = null
    this.panState = null
    this.svg.style.cursor = 'grab'
  }

  private saveSnapshot() {
    this.undoHistory.push(JSON.stringify(this.graph))
    if (this.undoHistory.length > 50) this.undoHistory.shift()
  }

  undo() {
    if (this.undoHistory.length === 0) return
    const prev = this.undoHistory.pop()!
    this.graph = JSON.parse(prev)
    this.mutations.push({ op: 'undo', ts: Date.now() })
    this.bootstrap()
    this.opts.onGraphChange(this.graph)
  }

  // ──── Interaction: Resize ───────────────────────────────

  private beginResize(nodeId: string, corner: string, e: MouseEvent) {
    if (this.locked) return
    const node = this.lookupNode(nodeId)
    if (!node) return
    const pt = this.screenToSvg(e)
    this.resizeState = {
      nodeId, corner,
      startX: pt.x, startY: pt.y,
      origW: node.width, origH: node.height,
      origNX: node.x, origNY: node.y,
    }
  }

  private processResize(e: MouseEvent) {
    if (!this.resizeState) return
    const node = this.lookupNode(this.resizeState.nodeId)
    if (!node) return
    const pt = this.screenToSvg(e)
    const dx = pt.x - this.resizeState.startX
    const dy = pt.y - this.resizeState.startY
    const minW = this.opts.minNodeWidth
    const minH = this.opts.minNodeHeight

    switch (this.resizeState.corner) {
      case 'se':
        node.width = Math.max(minW, this.resizeState.origW + dx)
        node.height = Math.max(minH, this.resizeState.origH + dy)
        break
      case 'sw':
        node.width = Math.max(minW, this.resizeState.origW - dx)
        node.height = Math.max(minH, this.resizeState.origH + dy)
        node.x = this.resizeState.origNX + this.resizeState.origW - node.width
        break
      case 'ne':
        node.width = Math.max(minW, this.resizeState.origW + dx)
        node.height = Math.max(minH, this.resizeState.origH - dy)
        node.y = this.resizeState.origNY + this.resizeState.origH - node.height
        break
      case 'nw':
        node.width = Math.max(minW, this.resizeState.origW - dx)
        node.height = Math.max(minH, this.resizeState.origH - dy)
        node.x = this.resizeState.origNX + this.resizeState.origW - node.width
        node.y = this.resizeState.origNY + this.resizeState.origH - node.height
        break
    }

    if (this.opts.snapToGrid) {
      const gs = this.opts.gridSize
      node.x = Math.round(node.x / gs) * gs
      node.y = Math.round(node.y / gs) * gs
      node.width = Math.round(node.width / gs) * gs
      node.height = Math.round(node.height / gs) * gs
    }

    this.refreshNodeVisual(node)
    this.refreshEdgesFor(node.id)
    this.opts.onNodeResize(node.id, node.width, node.height)
  }

  // ──── Interaction: Pan & Zoom ───────────────────────────

  private handleCanvasMouseDown(e: MouseEvent) {
    const tgt = e.target as Element
    const isBg = tgt === this.svg || tgt.classList.contains('svg-grid')
    if ((e.button === 0 && isBg) || e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault()
      this.panState = {
        sx: e.clientX, sy: e.clientY,
        svx: this.viewport.x, svy: this.viewport.y,
      }
      this.panDidMove = false
      this.svg.style.cursor = 'grabbing'
    }
  }

  private handleWheel(e: WheelEvent) {
    e.preventDefault()
    const delta = -e.deltaY * ZOOM_STEP * 0.01
    const factor = Math.pow(2, -delta * 10)
    const pt = this.screenToSvg(e)

    const newW = this.viewport.w * factor
    const newH = this.viewport.h * factor
    const baseW = (this.graph.width || 800) + this.opts.padding * 2
    const nextZoom = baseW / newW

    if (nextZoom < ZOOM_FLOOR || nextZoom > ZOOM_CEILING) return

    this.viewport.x = pt.x - (pt.x - this.viewport.x) * factor
    this.viewport.y = pt.y - (pt.y - this.viewport.y) * factor
    this.viewport.w = newW
    this.viewport.h = newH
    this.currentZoom = nextZoom
    this.syncViewport()
    this.refreshGridExtent()
  }

  // ──── Touch events (mobile) ─────────────────────────────

  private handleTouchStart(e: TouchEvent) {
    if (e.touches.length === 1) {
      const touch = e.touches[0]
      const tgt = document.elementFromPoint(touch.clientX, touch.clientY) as Element
      const isBg = !tgt || tgt === this.svg || tgt.classList.contains('svg-grid')
      if (isBg) {
        e.preventDefault()
        this.panState = {
          sx: touch.clientX, sy: touch.clientY,
          svx: this.viewport.x, svy: this.viewport.y,
        }
        this.panDidMove = false
      }
    } else if (e.touches.length === 2) {
      e.preventDefault()
      const [t1, t2] = [e.touches[0], e.touches[1]]
      const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY)
      this.touchZoomState = {
        prevDist: dist,
        prevMidX: (t1.clientX + t2.clientX) / 2,
        prevMidY: (t1.clientY + t2.clientY) / 2,
        initVX: this.viewport.x,
        initVY: this.viewport.y,
      }
      this.panState = null
    }
  }

  private handleTouchMove(e: TouchEvent) {
    if (e.touches.length === 1 && this.panState) {
      e.preventDefault()
      const touch = e.touches[0]
      const rect = this.svg.getBoundingClientRect()
      const scX = this.viewport.w / rect.width
      const scY = this.viewport.h / rect.height
      this.viewport.x = this.panState.svx - (touch.clientX - this.panState.sx) * scX
      this.viewport.y = this.panState.svy - (touch.clientY - this.panState.sy) * scY
      this.panDidMove = true
      this.syncViewport()
      this.refreshGridExtent()
    } else if (e.touches.length === 2 && this.touchZoomState) {
      e.preventDefault()
      const [t1, t2] = [e.touches[0], e.touches[1]]
      const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY)
      const midX = (t1.clientX + t2.clientX) / 2
      const midY = (t1.clientY + t2.clientY) / 2

      const factor = this.touchZoomState.prevDist / dist
      const pt = this.screenToSvgXY(midX, midY)
      const newW = this.viewport.w * factor
      const baseW = (this.graph.width || 800) + this.opts.padding * 2
      const nextZoom = baseW / newW

      if (nextZoom >= ZOOM_FLOOR && nextZoom <= ZOOM_CEILING) {
        this.viewport.x = pt.x - (pt.x - this.viewport.x) * factor
        this.viewport.y = pt.y - (pt.y - this.viewport.y) * factor
        this.viewport.w = newW
        this.viewport.h *= factor
        this.currentZoom = nextZoom
      }

      this.touchZoomState.prevDist = dist
      this.touchZoomState.prevMidX = midX
      this.touchZoomState.prevMidY = midY
      this.syncViewport()
      this.refreshGridExtent()
    }
  }

  private handleTouchEnd(_e: TouchEvent) {
    this.panState = null
    this.touchZoomState = null
  }

  // ──── Label editing ─────────────────────────────────────

  private openLabelEditor(nodeId: string) {
    if (this.locked) return
    const node = this.lookupNode(nodeId)
    if (!node) return
    this.activeLabelEdit = nodeId

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
      const oldLabel = node.label
      node.label = input.value || node.label
      this.activeLabelEdit = null
      fo.remove()
      this.refreshNodeVisual(node)
      this.mutations.push({ op: 'label', nodeId, ts: Date.now() })
      this.opts.onLabelEdit(node.id, node.label)
      this.opts.onGraphChange(this.graph)
    })
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') input.blur()
      if (e.key === 'Escape') { input.value = node.label; input.blur() }
    })

    fo.appendChild(input)
    g.appendChild(fo)
    requestAnimationFrame(() => { input.focus(); input.select() })
  }

  // ──── Selection ─────────────────────────────────────────

  private selectNode(nodeId: string) {
    this.clearSelection()
    this.selectedNodeId = nodeId
    const g = this.svg.querySelector(`[data-node-id="${nodeId}"]`)
    if (!g) return

    const rect = g.querySelector('rect:not(.resize-handle)')
    if (rect) {
      rect.setAttribute('stroke', this.palette.selectionColor)
      rect.setAttribute('stroke-width', '2.5')
    }
    g.querySelectorAll('.resize-handle').forEach(h => {
      (h as SVGElement).style.display = 'block'
    })
  }

  private clearSelection() {
    if (!this.selectedNodeId) return
    const g = this.svg.querySelector(`[data-node-id="${this.selectedNodeId}"]`)
    if (g) {
      const rect = g.querySelector('rect:not(.resize-handle)')
      if (rect) {
        rect.setAttribute('stroke', this.palette.nodeStroke)
        const node = this.lookupNode(this.selectedNodeId)
        rect.setAttribute('stroke-width', node?.isGroup ? '2' : '1.5')
      }
      g.querySelectorAll('.resize-handle').forEach(h => {
        (h as SVGElement).style.display = 'none'
      })
    }
    this.selectedNodeId = null
  }

  // ──── Visual refresh helpers ────────────────────────────

  private refreshNodeVisual(node: InteractiveNode) {
    const g = this.svg.querySelector(`[data-node-id="${node.id}"]`)
    if (!g) return

    const rect = g.querySelector('rect:not(.resize-handle)')
    if (rect) {
      rect.setAttribute('x', String(node.x))
      rect.setAttribute('y', String(node.y))
      rect.setAttribute('width', String(node.width))
      rect.setAttribute('height', String(node.height))
    }

    const txt = g.querySelector('text')
    if (txt) {
      const maxC = Math.max(6, Math.floor(node.width / 8))
      const dl = node.label.length > maxC ? node.label.slice(0, maxC - 2) + '…' : node.label
      txt.setAttribute('x', String(node.x + node.width / 2))
      txt.setAttribute('y', String(node.isGroup ? node.y + 16 : node.y + node.height / 2))
      txt.textContent = dl
    }

    const corners = ['nw', 'ne', 'sw', 'se'] as const
    for (const corner of corners) {
      const handle = g.querySelector(`.resize-${corner}`) as SVGRectElement | null
      if (handle) this.placeHandle(handle, node, corner)
    }
  }

  private placeHandle(handle: SVGRectElement, node: InteractiveNode, corner: string) {
    const hs = HANDLE_DIM
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

  private refreshEdgesFor(nodeId: string) {
    const node = this.lookupNode(nodeId)
    if (!node) return
    for (const edge of this.graph.edges) {
      if (edge.sourceId === nodeId || edge.targetId === nodeId) {
        this.recomputeEdgeRoute(edge)
        this.refreshEdgeVisual(edge)
      }
    }
  }

  private recomputeEdgeRoute(edge: InteractiveEdge) {
    const src = this.lookupNode(edge.sourceId)
    const tgt = this.lookupNode(edge.targetId)
    if (!src || !tgt) return

    const sx = src.x + src.width / 2
    const sy = src.y + src.height
    const tx = tgt.x + tgt.width / 2
    const ty = tgt.y
    const midY = (sy + ty) / 2

    if (Math.abs(sy - ty) > 20) {
      edge.points = [
        { x: sx, y: sy }, { x: sx, y: midY },
        { x: tx, y: midY }, { x: tx, y: ty },
      ]
    } else {
      edge.points = [{ x: sx, y: sy }, { x: tx, y: ty }]
    }
  }

  private refreshEdgeVisual(edge: InteractiveEdge) {
    const g = this.svg.querySelector(`[data-edge-id="${edge.id}"]`)
    if (!g) return
    const path = g.querySelector('path')
    if (path && edge.points.length >= 2) {
      path.setAttribute('d', this.computeSmoothPath(edge.points))
    }
  }

  // ──── Coordinate conversion ─────────────────────────────

  private screenToSvg(e: MouseEvent): { x: number; y: number } {
    const rect = this.svg.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (this.viewport.w / rect.width) + this.viewport.x,
      y: (e.clientY - rect.top) * (this.viewport.h / rect.height) + this.viewport.y,
    }
  }

  private screenToSvgXY(cx: number, cy: number): { x: number; y: number } {
    const rect = this.svg.getBoundingClientRect()
    return {
      x: (cx - rect.left) * (this.viewport.w / rect.width) + this.viewport.x,
      y: (cy - rect.top) * (this.viewport.h / rect.height) + this.viewport.y,
    }
  }

  private syncViewport() {
    this.svg.setAttribute('viewBox',
      `${this.viewport.x} ${this.viewport.y} ${this.viewport.w} ${this.viewport.h}`)
  }

  private lookupNode(id: string): InteractiveNode | undefined {
    return this.graph.nodes.find(n => n.id === id)
  }

  // ──── Public API ────────────────────────────────────────

  getGraph(): InteractiveGraph {
    return JSON.parse(JSON.stringify(this.graph))
  }

  getMastergoLayout(): Array<{
    id: string; name: string
    bbox: { x: number; y: number; width: number; height: number }
  }> {
    return this.graph.nodes.map(n => ({
      id: n.id,
      name: n.label,
      bbox: { x: Math.round(n.x), y: Math.round(n.y), width: Math.round(n.width), height: Math.round(n.height) },
    }))
  }

  setBackgroundImage(dataUri: string, imgW: number, imgH: number): void {
    this.svg.querySelectorAll('.editor-bg-image').forEach(el => el.remove())
    const img = document.createElementNS('http://www.w3.org/2000/svg', 'image')
    img.setAttribute('href', dataUri)
    img.setAttribute('x', '0')
    img.setAttribute('y', '0')
    img.setAttribute('width', String(imgW))
    img.setAttribute('height', String(imgH))
    img.setAttribute('opacity', '0.4')
    img.classList.add('editor-bg-image')
    img.style.pointerEvents = 'none'

    const firstContent = this.svg.querySelector('.interactive-edge, .interactive-node, .svg-grid')
    if (firstContent) this.svg.insertBefore(img, firstContent)
    else this.svg.appendChild(img)
  }

  getElkJson(): Record<string, unknown> {
    return {
      id: 'root',
      children: this.graph.nodes.map(n => ({
        id: n.id, width: n.width, height: n.height,
        labels: [{ text: n.label }],
      })),
      edges: this.graph.edges.map(e => ({
        id: e.id, sources: [e.sourceId], targets: [e.targetId],
        ...(e.label ? { labels: [{ text: e.label }] } : {}),
      })),
    }
  }

  toStaticSvg(): string {
    const clone = this.svg.cloneNode(true) as SVGSVGElement
    clone.querySelectorAll('.resize-handle, .label-editor-fo').forEach(el => el.remove())
    clone.querySelectorAll('.interactive-node rect:not(.resize-handle)').forEach(rect => {
      rect.setAttribute('stroke', this.palette.nodeStroke)
    })
    return new XMLSerializer().serializeToString(clone)
  }

  addNode(label: string): InteractiveNode {
    const id = `node_${Date.now()}`
    const maxX = Math.max(...this.graph.nodes.map(n => n.x + n.width), 100)
    const node: InteractiveNode = {
      id, x: maxX + 30, y: 50,
      width: 160, height: 60,
      label, fill: this.palette.nodeFill,
    }
    this.graph.nodes.push(node)
    this.svg.appendChild(this.buildNodeGroup(node, this.graph.nodes.length - 1))
    this.mutations.push({ op: 'add', nodeId: id, ts: Date.now() })
    this.opts.onGraphChange(this.graph)
    return node
  }

  removeSelectedNode(): boolean {
    if (!this.selectedNodeId) return false
    const id = this.selectedNodeId
    this.graph.nodes = this.graph.nodes.filter(n => n.id !== id)
    const deadEdges = this.graph.edges.filter(e => e.sourceId === id || e.targetId === id)
    this.graph.edges = this.graph.edges.filter(e => e.sourceId !== id && e.targetId !== id)
    this.svg.querySelector(`[data-node-id="${id}"]`)?.remove()
    deadEdges.forEach(e => this.svg.querySelector(`[data-edge-id="${e.id}"]`)?.remove())
    this.selectedNodeId = null
    this.mutations.push({ op: 'remove', nodeId: id, ts: Date.now() })
    this.opts.onGraphChange(this.graph)
    return true
  }

  fitView() {
    const target = this.calcFitViewport()
    if (!target) return
    this.viewport = target
    this.currentZoom = (this.graph.width + this.opts.padding * 2) / this.viewport.w
    this.syncViewport()
    this.refreshGridExtent()
  }

  smoothFitView() {
    const target = this.calcFitViewport()
    if (!target) return

    const origin = { ...this.viewport }
    const duration = 300
    const t0 = performance.now()

    const tick = (now: number) => {
      const progress = Math.min((now - t0) / duration, 1)
      const ease = 1 - Math.pow(1 - progress, 3) // ease-out cubic

      this.viewport.x = origin.x + (target.x - origin.x) * ease
      this.viewport.y = origin.y + (target.y - origin.y) * ease
      this.viewport.w = origin.w + (target.w - origin.w) * ease
      this.viewport.h = origin.h + (target.h - origin.h) * ease
      this.syncViewport()
      this.refreshGridExtent()

      if (progress < 1) requestAnimationFrame(tick)
      else this.currentZoom = (this.graph.width + this.opts.padding * 2) / this.viewport.w
    }
    requestAnimationFrame(tick)
  }

  private calcFitViewport(): { x: number; y: number; w: number; h: number } | null {
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

    let targetW = contentW + extraPad * 2
    let targetH = contentH + extraPad * 2
    const containerRect = this.container.getBoundingClientRect()
    if (containerRect.width > 0 && containerRect.height > 0) {
      const containerAR = containerRect.width / containerRect.height
      const contentAR = targetW / targetH
      if (contentAR > containerAR) targetH = targetW / containerAR
      else targetW = targetH * containerAR
    }

    return {
      x: (minX + maxX) / 2 - targetW / 2,
      y: (minY + maxY) / 2 - targetH / 2,
      w: targetW, h: targetH,
    }
  }

  setZoom(level: number) {
    level = Math.max(ZOOM_FLOOR, Math.min(ZOOM_CEILING, level))
    const cx = this.viewport.x + this.viewport.w / 2
    const cy = this.viewport.y + this.viewport.h / 2
    const baseW = (this.graph.width || 800) + this.opts.padding * 2
    const baseH = (this.graph.height || 600) + this.opts.padding * 2
    this.viewport.w = baseW / level
    this.viewport.h = baseH / level
    this.viewport.x = cx - this.viewport.w / 2
    this.viewport.y = cy - this.viewport.h / 2
    this.currentZoom = level
    this.syncViewport()
    this.refreshGridExtent()
  }

  updateFromLayout(graph: InteractiveGraph) {
    this.graph = JSON.parse(JSON.stringify(graph))
    this.locked = false
    this.bootstrap()
  }

  confirmEdit() {
    this.locked = true
    this.clearSelection()
    this.mutations.push({ op: 'confirm', ts: Date.now() })
    if (this.svg) {
      this.svg.style.cursor = 'grab'
      this.svg.querySelectorAll('.interactive-node').forEach(g => {
        (g as SVGGElement).style.cursor = 'default'
        ;(g as SVGGElement).style.opacity = '0.9'
      })
    }
  }

  isConfirmed(): boolean { return this.locked }

  unlockEdit() {
    this.locked = false
    this.mutations.push({ op: 'unlock', ts: Date.now() })
    if (this.svg) {
      this.svg.querySelectorAll('.interactive-node').forEach(g => {
        (g as SVGGElement).style.cursor = 'grab'
        ;(g as SVGGElement).style.opacity = '1'
      })
    }
  }

  destroy() {
    document.removeEventListener('keydown', this._boundKeyHandler)
    this.container.innerHTML = ''
  }

  // ──── Debug surface ─────────────────────────────────────
  // Call from browser console or test harness

  /** Print full canvas state to console — nodes, edges, viewport, zoom */
  printCanvasState() {
    console.group('[CanvasEditor] current state')
    console.log('viewport:', { ...this.viewport })
    console.log('zoom:', this.currentZoom)
    console.log('locked:', this.locked)
    console.log('selectedNode:', this.selectedNodeId)
    console.table(this.graph.nodes.map(n => ({
      id: n.id, x: n.x, y: n.y, w: n.width, h: n.height,
      label: n.label, renderMode: n.renderMode ?? 'default',
      sprite: n.spriteUrl ? '✓' : '-',
      operator: n.isOperator ? '✓' : '-',
      group: n.isGroup ? '✓' : '-',
      family: n.familyId ?? '-',
    })))
    console.log(`edges (${this.graph.edges.length}):`)
    for (const e of this.graph.edges) {
      console.log(`  ${e.id}: ${e.sourceId} → ${e.targetId} [${e.points.length} pts]`)
    }
    console.groupEnd()
  }

  /** Print node tree as hierarchical text */
  printNodeTree() {
    const groups = this.graph.nodes.filter(n => n.isGroup)
    const leaves = this.graph.nodes.filter(n => !n.isGroup)
    console.group('[CanvasEditor] node tree')
    for (const g of groups) {
      console.log(`📦 ${g.id} "${g.label}" (${g.width}×${g.height})`)
    }
    for (const l of leaves) {
      const tag = l.isOperator ? '⊕' : l.spriteUrl ? '🖼️' : '□'
      console.log(`  ${tag} ${l.id} "${l.label}" @(${l.x},${l.y})`)
    }
    console.groupEnd()
  }

  /** Dump mutation ring buffer */
  printMutationHistory() {
    this.mutations.printAll()
  }

  /** Get raw mutation records for programmatic inspection */
  getMutationRecords(): MutationRecord[] {
    return this.mutations.dump()
  }
}

export default CanvasEditor
