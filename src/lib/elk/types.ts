/**
 * ELK.js TypeScript Type Definitions
 * Enhanced with advanced edge routing for neural-network-level diagrams
 * GitHub: kieler/elkjs, EmilStenstrom/elkjs-svg
 */

// ============================================================
// ELK Graph Core Types (elkjs compatible)
// ============================================================

export type ElkLayoutOptions = Record<string, string>

export interface ElkLabel {
  id?: string
  text: string
  x?: number
  y?: number
  width?: number
  height?: number
  layoutOptions?: ElkLayoutOptions
}

export interface ElkPort {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: ElkLabel[]
  layoutOptions?: ElkLayoutOptions
}

export interface ElkEdgeSection {
  id: string
  startPoint: ElkPoint
  endPoint: ElkPoint
  bendPoints?: ElkPoint[]
  incomingShape?: string
  outgoingShape?: string
  incomingSections?: string[]
  outgoingSections?: string[]
}

export interface ElkPoint { x: number; y: number }

export interface ElkNode {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: ElkLabel[]
  ports?: ElkPort[]
  children?: ElkNode[]
  edges?: ElkEdge[]
  layoutOptions?: ElkLayoutOptions
  properties?: Record<string, unknown>
}

// ============================================================
// Advanced Edge Routing Types -- Neural-Network Level Arrow System
// ============================================================

/** Edge routing mode */
export type EdgeRoutingMode = 'ORTHOGONAL' | 'SPLINES' | 'POLYLINE' | 'UNDEFINED'

/** Edge visual line style */
export type EdgeLineStyle = 'solid' | 'dashed' | 'dotted' | 'double'

/** Edge arrow type */
export type EdgeArrowType = 'arrow' | 'none' | 'diamond' | 'circle' | 'open'

/** Edge directionality */
export type EdgeDirectionality = 'directed' | 'bidirectional' | 'undirected'

/** Edge semantic type -- common arrow semantics in academic figures */
export type EdgeSemanticType =
  | 'data_flow'
  | 'gradient_flow'
  | 'skip_connection'
  | 'optional_path'
  | 'inference_only'
  | 'fan_out'
  | 'fan_in'
  | 'feedback'
  | 'attention'
  | 'concatenation'
  | 'residual'
  | 'cross_boundary'

/** Edge label configuration */
export interface EdgeLabelConfig {
  text: string
  /** Label position on edge: 0.0 (source) ~ 1.0 (target), default 0.5 */
  position?: number
  /** Label rotation angle (degrees), 'auto' = follow edge direction */
  rotation?: number | 'auto'
  fontSize?: number
  backgroundColor?: string
}

/** Advanced edge properties -- neural-network level rendering control */
export interface AdvancedEdgeProperties {
  routing?: EdgeRoutingMode
  lineStyle?: EdgeLineStyle
  strokeDasharray?: string
  strokeWidth?: number
  strokeColor?: string
  sourceArrow?: EdgeArrowType
  targetArrow?: EdgeArrowType
  directionality?: EdgeDirectionality
  semanticType?: EdgeSemanticType
  edgeLabels?: EdgeLabelConfig[]
  crossesGroupBoundary?: boolean
  curvature?: number
  priority?: number
}

/** ELK edge -- with advanced routing properties */
export interface ElkEdge {
  id: string
  sources: string[]
  targets: string[]
  labels?: ElkLabel[]
  sections?: ElkEdgeSection[]
  layoutOptions?: ElkLayoutOptions
  properties?: Record<string, unknown>
  /** Advanced edge properties -- neural-network level rendering control */
  advanced?: AdvancedEdgeProperties
}

/** ELK root graph -- top-level object passed to elk.layout() */
export interface ElkGraph extends ElkNode {
  id: string
  children: ElkNode[]
  edges: ElkEdge[]
}

// ============================================================
// Layout Result Types
// ============================================================

export interface LayoutedNode extends ElkNode {
  x: number
  y: number
  width: number
  height: number
}

export interface LayoutedEdge extends ElkEdge {
  sections: ElkEdgeSection[]
}

export interface LayoutedGraph extends ElkGraph {
  children: LayoutedNode[]
  edges: LayoutedEdge[]
}

export interface LayoutResult {
  graph: LayoutedGraph
  duration: number
  algorithm: ElkAlgorithm
  bounds: { x: number; y: number; width: number; height: number }
}

// ============================================================
// Algorithm & Preset Types
// ============================================================

export type ElkAlgorithm = 'layered' | 'stress' | 'mrtree' | 'radial' | 'force' | 'disco' | 'box' | 'fixed' | 'random'
export type ElkDirection = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT'

export interface ElkPreset {
  name: string
  description: string
  layoutOptions: ElkLayoutOptions
  defaultNodeSize: { width: number; height: number }
  tags: string[]
}

// ============================================================
// NanoBanana Scaffold Types
// ============================================================

export interface ScaffoldElement {
  id: string
  type: 'box' | 'circle' | 'diamond' | 'parallelogram' | 'cylinder' | 'group'
  label: string
  x: number; y: number; width: number; height: number
  style?: string; fill?: string; stroke?: string; fontSize?: number
  children?: ScaffoldElement[]
}

export interface ScaffoldConnection {
  from: string; to: string
  style: 'arrow' | 'dashed' | 'dotted' | 'thick' | 'bidirectional'
  label?: string; points?: ElkPoint[]; color?: string
  /** Advanced routing properties carried from topology */
  advanced?: AdvancedEdgeProperties
}

export interface NanoBananaScaffold {
  figure_type: string
  canvas: { width: number; height: number; padding?: number }
  elements: ScaffoldElement[]
  connections: ScaffoldConnection[]
  request?: string
  colorScheme?: 'academic' | 'tech' | 'minimal' | 'vibrant' | 'custom'
  customColors?: Record<string, string>
}

// ============================================================
// Pipeline Flow Types
// ============================================================

export type StepStatus = 'idle' | 'running' | 'success' | 'error'

export interface PipelineState {
  currentStep: 0 | 1 | 2 | 3 | 4
  steps: {
    topology: { status: StepStatus; data?: ElkGraph; error?: string }
    layout: { status: StepStatus; data?: LayoutedGraph; error?: string }
    beautify: { status: StepStatus; data?: string; error?: string }
    display: { status: StepStatus; error?: string }
  }
}

export interface GenerateOptions {
  text: string
  model?: string
  algorithm?: ElkAlgorithm
  direction?: ElkDirection
  preset?: string
  colorScheme?: string
  skeletonOnly?: boolean
  edgeRouting?: EdgeRoutingMode
}

// ============================================================
// Semantic Edge Style Defaults
// ============================================================

export const EDGE_SEMANTIC_DEFAULTS: Record<EdgeSemanticType, Partial<AdvancedEdgeProperties>> = {
  data_flow: { lineStyle: 'solid', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#78909C' },
  gradient_flow: { lineStyle: 'dashed', strokeDasharray: '8,4', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#E57373' },
  skip_connection: { lineStyle: 'solid', routing: 'SPLINES', targetArrow: 'arrow', strokeWidth: 2, strokeColor: '#4CAF50', curvature: 0.8 },
  optional_path: { lineStyle: 'dashed', strokeDasharray: '5,5', targetArrow: 'arrow', strokeWidth: 1, strokeColor: '#9E9E9E' },
  inference_only: { lineStyle: 'dashed', strokeDasharray: '10,3,3,3', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#7986CB' },
  fan_out: { lineStyle: 'solid', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#FF9800' },
  fan_in: { lineStyle: 'solid', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#2196F3' },
  feedback: { lineStyle: 'dashed', strokeDasharray: '6,3', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#AB47BC', routing: 'SPLINES' },
  attention: { lineStyle: 'dotted', strokeDasharray: '2,4', targetArrow: 'arrow', strokeWidth: 2, strokeColor: '#F44336' },
  concatenation: { lineStyle: 'solid', targetArrow: 'arrow', strokeWidth: 2, strokeColor: '#009688' },
  residual: { lineStyle: 'solid', routing: 'SPLINES', targetArrow: 'arrow', strokeWidth: 2, strokeColor: '#4CAF50', curvature: 0.6 },
  cross_boundary: { lineStyle: 'solid', targetArrow: 'arrow', strokeWidth: 1.5, strokeColor: '#607D8B', crossesGroupBoundary: true },
}
