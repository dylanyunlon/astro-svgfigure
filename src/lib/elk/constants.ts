/**
 * ELK.js 常量配置
 *
 * 算法参考:
 * - https://www.eclipse.org/elk/reference/algorithms.html
 * - https://www.eclipse.org/elk/reference/options.html
 * - https://github.com/kieler/elkjs
 *
 * ReactFlow ELK 集成参考:
 * - https://reactflow.dev/examples/layout/elkjs
 */

import type { ElkAlgorithm, ElkDirection, ElkLayoutOptions } from './types'

// ============================================================
// 算法 ID (elkjs 内置支持)
// ============================================================

/**
 * ELK 支持的算法列表
 * 参考: https://github.com/kieler/elkjs#layout-options
 *
 * - layered: Sugiyama 分层算法, 适合有向图/流程图 (ELK 旗舰算法)
 * - stress: 应力模型, 适合一般图
 * - mrtree: 树形布局, 适合层级结构
 * - radial: 径向布局, 适合以中心节点展开
 * - force: 力导向布局, 适合社交网络类图
 * - disco: 断连组件打包
 * - box/fixed/random: 简单布局 (始终可用)
 */
export const ELK_ALGORITHMS: Record<ElkAlgorithm, { name: string; description: string }> = {
  layered: {
    name: 'Layered (Sugiyama)',
    description: '分层布局, 适合有向流程图、Pipeline、神经网络架构'
  },
  stress: {
    name: 'Stress',
    description: '应力模型布局, 适合一般无向图'
  },
  mrtree: {
    name: 'Mr. Tree',
    description: '树形布局, 适合决策树、组织架构'
  },
  radial: {
    name: 'Radial',
    description: '径向布局, 适合知识图谱、以中心展开的结构'
  },
  force: {
    name: 'Force',
    description: '力导向布局, 适合社交网络、关系图'
  },
  disco: {
    name: 'Disco (Component Packing)',
    description: '打包断连组件'
  },
  box: {
    name: 'Box',
    description: '简单盒装布局'
  },
  fixed: {
    name: 'Fixed',
    description: '固定位置 (使用输入坐标)'
  },
  random: {
    name: 'Random',
    description: '随机放置 (调试用)'
  }
} as const

// ============================================================
// 布局方向
// ============================================================

export const ELK_DIRECTIONS: Record<ElkDirection, string> = {
  DOWN: '从上到下 ↓',
  RIGHT: '从左到右 →',
  UP: '从下到上 ↑',
  LEFT: '从右到左 ←'
} as const

// ============================================================
// 默认布局选项
// 参考: https://reactflow.dev/examples/layout/elkjs (elkOptions)
// ============================================================

/** 默认布局选项 (适合学术论文架构图) */
export const DEFAULT_LAYOUT_OPTIONS: ElkLayoutOptions = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  // 层间距离 (同一层的相邻节点之间)
  'elk.spacing.nodeNode': '40',
  // 层与层之间的距离
  'elk.layered.spacing.nodeNodeBetweenLayers': '80',
  // 边与节点的间距
  'elk.spacing.edgeNode': '30',
  // 边与边的间距
  'elk.spacing.edgeEdge': '20',
  // 端口约束 (固定端口在边框上的位置)
  'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
  // 边路由策略
  'elk.edgeRouting': 'ORTHOGONAL',
  // 内边距
  'elk.padding': '[top=20,left=20,bottom=20,right=20]'
} as const

// ============================================================
// 默认节点尺寸
// ============================================================

/** 默认节点宽高 */
export const DEFAULT_NODE_SIZE = {
  width: 150,
  height: 50
} as const

/** 不同类型的节点尺寸 */
export const NODE_SIZE_BY_TYPE: Record<string, { width: number; height: number }> = {
  default: { width: 150, height: 50 },
  small: { width: 100, height: 40 },
  large: { width: 200, height: 70 },
  wide: { width: 250, height: 50 },
  tall: { width: 120, height: 80 },
  square: { width: 80, height: 80 },
  circle: { width: 60, height: 60 },
  group: { width: 300, height: 200 }
} as const

// ============================================================
// SVG 骨架渲染常量
// ============================================================

export const SKELETON_SVG = {
  /** 节点填充色 */
  nodeFill: '#E3F2FD',
  /** 节点边框色 */
  nodeStroke: '#1565C0',
  /** 节点边框宽度 */
  nodeStrokeWidth: 2,
  /** 节点圆角 */
  nodeRx: 8,
  /** 边的颜色 */
  edgeStroke: '#616161',
  /** 边的宽度 */
  edgeStrokeWidth: 1.5,
  /** 箭头尺寸 */
  arrowSize: 8,
  /** 标签字体大小 */
  labelFontSize: 12,
  /** 标签字体 */
  labelFontFamily: 'Arial, Helvetica, sans-serif',
  /** 标签颜色 */
  labelColor: '#212121',
  /** 画布内边距 */
  canvasPadding: 30,
  /** 背景色 */
  backgroundColor: '#FFFFFF'
} as const

// ============================================================
// NanoBanana 脚手架常量
// ============================================================

export const SCAFFOLD_DEFAULTS = {
  /** 默认图形类型 */
  figureType: 'academic_architecture',
  /** 画布内边距 */
  canvasPadding: 40,
  /** 默认配色方案 */
  colorScheme: 'academic' as const,
  /** 学术配色 */
  colors: {
    academic: {
      primary: '#1565C0',
      secondary: '#2E7D32',
      accent: '#E65100',
      background: '#FAFAFA',
      nodeColors: ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5', '#E0F7FA', '#FBE9E7']
    },
    tech: {
      primary: '#0D47A1',
      secondary: '#00695C',
      accent: '#FF6F00',
      background: '#FAFAFA',
      nodeColors: ['#E1F5FE', '#E0F2F1', '#FFF8E1', '#EDE7F6', '#E0F7FA', '#FCE4EC']
    },
    minimal: {
      primary: '#424242',
      secondary: '#757575',
      accent: '#212121',
      background: '#FFFFFF',
      nodeColors: ['#F5F5F5', '#EEEEEE', '#E0E0E0', '#FAFAFA', '#F5F5F5', '#EEEEEE']
    }
  }
} as const


// ============================================================
// Advanced Edge Routing Constants
// ============================================================

export const EDGE_ROUTING_MODES = {
  ORTHOGONAL: 'ORTHOGONAL',
  SPLINES: 'SPLINES',
  POLYLINE: 'POLYLINE',
  UNDEFINED: 'UNDEFINED',
} as const

export const EDGE_SEMANTIC_COLORS: Record<string, string> = {
  data_flow: '#78909C',
  gradient_flow: '#E57373',
  skip_connection: '#4CAF50',
  optional_path: '#9E9E9E',
  inference_only: '#7986CB',
  fan_out: '#FF9800',
  fan_in: '#2196F3',
  feedback: '#AB47BC',
  attention: '#F44336',
  concatenation: '#009688',
  residual: '#4CAF50',
  cross_boundary: '#607D8B',
} as const

export const EDGE_DASH_PATTERNS: Record<string, string> = {
  solid: '',
  dashed: '8,4',
  dotted: '2,4',
  double: '10,3,3,3',
  gradient_flow: '8,4',
  inference_only: '10,3,3,3',
  feedback: '6,3',
  attention: '2,4',
} as const

export const HIERARCHY_LAYOUT_OPTIONS: Record<string, string> = {
  'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
  'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
  'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
  'elk.edgeRouting': 'ORTHOGONAL',
} as const
