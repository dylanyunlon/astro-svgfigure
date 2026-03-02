/**
 * ELK.js TypeScript 类型定义
 *
 * 基于 kieler/elkjs (https://github.com/kieler/elkjs) 的 typings/elk-api.d.ts
 * 以及 Eclipse ELK 图结构规范
 *
 * 这些类型用于:
 * 1. LLM 输出的零坐标拓扑 JSON (topology.json)
 * 2. ELK.js 布局后的精确坐标 JSON (layouted.json)
 * 3. NanoBanana 脚手架 JSON (scaffold.json)
 */

// ============================================================
// ELK Graph 核心类型 (与 elkjs 兼容)
// ============================================================

/** ELK 布局选项 — 键值对形式 */
export type ElkLayoutOptions = Record<string, string>

/** ELK 标签 */
export interface ElkLabel {
  id?: string
  text: string
  x?: number
  y?: number
  width?: number
  height?: number
  layoutOptions?: ElkLayoutOptions
}

/** ELK 端口 — 节点边界上的连接点 */
export interface ElkPort {
  id: string
  x?: number
  y?: number
  width?: number
  height?: number
  labels?: ElkLabel[]
  layoutOptions?: ElkLayoutOptions
}

/** 边的路由段 */
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

/** 坐标点 */
export interface ElkPoint {
  x: number
  y: number
}

/** ELK 节点 — 可包含子节点 (compound graph) */
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
  /** 自定义属性: 节点类型 (用于渲染和脚手架) */
  properties?: Record<string, unknown>
}

/** ELK 边 */
export interface ElkEdge {
  id: string
  sources: string[]
  targets: string[]
  labels?: ElkLabel[]
  sections?: ElkEdgeSection[]
  layoutOptions?: ElkLayoutOptions
  /** 自定义属性: 边的样式 */
  properties?: Record<string, unknown>
}

/** ELK 根图 — 传给 elk.layout() 的顶层对象 */
export interface ElkGraph extends ElkNode {
  id: string
  children: ElkNode[]
  edges: ElkEdge[]
}

// ============================================================
// 布局结果类型
// ============================================================

/** 布局后的节点 — 保证有 x, y 坐标 */
export interface LayoutedNode extends ElkNode {
  x: number
  y: number
  width: number
  height: number
}

/** 布局后的边 — 保证有 sections 路由信息 */
export interface LayoutedEdge extends ElkEdge {
  sections: ElkEdgeSection[]
}

/** 布局后的完整图 */
export interface LayoutedGraph extends ElkGraph {
  children: LayoutedNode[]
  edges: LayoutedEdge[]
}

/** 布局结果封装 */
export interface LayoutResult {
  graph: LayoutedGraph
  /** 布局耗时 (ms) */
  duration: number
  /** 使用的算法 */
  algorithm: ElkAlgorithm
  /** 画布尺寸 (所有节点的包围盒) */
  bounds: {
    x: number
    y: number
    width: number
    height: number
  }
}

// ============================================================
// 算法与预设类型
// ============================================================

/** ELK 支持的布局算法 */
export type ElkAlgorithm = 'layered' | 'stress' | 'mrtree' | 'radial' | 'force' | 'disco' | 'box' | 'fixed' | 'random'

/** 布局方向 */
export type ElkDirection = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT'

/** 预设配置 */
export interface ElkPreset {
  name: string
  description: string
  layoutOptions: ElkLayoutOptions
  /** 默认节点尺寸 */
  defaultNodeSize: { width: number; height: number }
  /** 推荐用途 */
  tags: string[]
}

// ============================================================
// NanoBanana 脚手架类型
// ============================================================

/** 脚手架中的元素 */
export interface ScaffoldElement {
  id: string
  type: 'box' | 'circle' | 'diamond' | 'parallelogram' | 'cylinder' | 'group'
  label: string
  x: number
  y: number
  width: number
  height: number
  style?: string
  fill?: string
  stroke?: string
  fontSize?: number
  /** 子元素 (用于 group 类型) */
  children?: ScaffoldElement[]
}

/** 脚手架中的连接 */
export interface ScaffoldConnection {
  from: string
  to: string
  style: 'arrow' | 'dashed' | 'dotted' | 'thick' | 'bidirectional'
  label?: string
  points?: ElkPoint[]
  color?: string
}

/** NanoBanana JSON 脚手架 — 传给 Gemini 生成 SVG */
export interface NanoBananaScaffold {
  figure_type: string
  canvas: {
    width: number
    height: number
    padding?: number
  }
  elements: ScaffoldElement[]
  connections: ScaffoldConnection[]
  /** 发给 Gemini 的附加指令 */
  request?: string
  /** 配色方案 */
  colorScheme?: 'academic' | 'tech' | 'minimal' | 'vibrant' | 'custom'
  /** 自定义配色 */
  customColors?: Record<string, string>
}

// ============================================================
// Pipeline 流程类型
// ============================================================

/** Pipeline 步骤状态 */
export type StepStatus = 'idle' | 'running' | 'success' | 'error'

/** Pipeline 整体状态 */
export interface PipelineState {
  currentStep: 0 | 1 | 2 | 3 | 4
  steps: {
    topology: { status: StepStatus; data?: ElkGraph; error?: string }
    layout: { status: StepStatus; data?: LayoutedGraph; error?: string }
    beautify: { status: StepStatus; data?: string; error?: string }  // SVG string
    display: { status: StepStatus; error?: string }
  }
}

/** 生成选项 */
export interface GenerateOptions {
  /** 输入的 method 文本 */
  text: string
  /** AI 模型 */
  model?: string
  /** ELK 布局算法 */
  algorithm?: ElkAlgorithm
  /** ELK 布局方向 */
  direction?: ElkDirection
  /** ELK 预设名称 */
  preset?: string
  /** NanoBanana 配色 */
  colorScheme?: string
  /** 是否跳过 NanoBanana 美化 (只输出骨架 SVG) */
  skeletonOnly?: boolean
}
