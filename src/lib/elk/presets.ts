/**
 * ELK 布局预设配置
 *
 * 每个预设针对一类学术图形优化了 ELK 参数。
 * 用户在前端可通过 ElkOptions 组件选择预设。
 *
 * 参考:
 * - https://www.eclipse.org/elk/reference/options.html
 * - https://www.eclipse.org/elk/reference/algorithms/org-eclipse-elk-layered.html
 */

import type { ElkPreset } from './types'

export const PRESETS: Record<string, ElkPreset> = {
  'academic-paper': {
    name: '学术论文架构图',
    description: '适合 Method 部分的 Pipeline / Architecture 图, 自上而下分层, 较大间距',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '50',
      'elk.layered.spacing.nodeNodeBetweenLayers': '100',
      'elk.spacing.edgeNode': '30',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.padding': '[top=30,left=30,bottom=30,right=30]'
    },
    defaultNodeSize: { width: 160, height: 55 },
    tags: ['paper', 'method', 'architecture', 'pipeline']
  },

  flowchart: {
    name: '流程图',
    description: '标准流程图布局, 从左到右, 紧凑间距',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': '30',
      'elk.layered.spacing.nodeNodeBetweenLayers': '70',
      'elk.spacing.edgeNode': '20',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.padding': '[top=20,left=20,bottom=20,right=20]'
    },
    defaultNodeSize: { width: 140, height: 50 },
    tags: ['flowchart', 'process', 'workflow']
  },

  'neural-net': {
    name: '神经网络架构',
    description: '适合 CNN/Transformer/GAN 等 NN 架构图, 从左到右, 宽间距',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': '40',
      'elk.layered.spacing.nodeNodeBetweenLayers': '120',
      'elk.spacing.edgeNode': '25',
      'elk.edgeRouting': 'SPLINES',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.padding': '[top=25,left=25,bottom=25,right=25]'
    },
    defaultNodeSize: { width: 120, height: 70 },
    tags: ['neural-network', 'CNN', 'transformer', 'GAN', 'deep-learning']
  },

  architecture: {
    name: '系统架构图',
    description: '适合软件/系统架构图, 分层, 支持子图',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '45',
      'elk.layered.spacing.nodeNodeBetweenLayers': '90',
      'elk.spacing.edgeNode': '30',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      'elk.padding': '[top=25,left=25,bottom=25,right=25]'
    },
    defaultNodeSize: { width: 170, height: 60 },
    tags: ['architecture', 'system', 'microservice', 'infrastructure']
  },

  tree: {
    name: '树形结构',
    description: '适合决策树、组织架构、分类法等',
    layoutOptions: {
      'elk.algorithm': 'mrtree',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '30',
      'elk.mrtree.spacing.nodeNodeBetweenLayers': '60',
      'elk.padding': '[top=20,left=20,bottom=20,right=20]'
    },
    defaultNodeSize: { width: 130, height: 45 },
    tags: ['tree', 'hierarchy', 'taxonomy', 'decision-tree']
  },

  radial: {
    name: '径向图',
    description: '以中心节点向外展开, 适合知识图谱、概念关联',
    layoutOptions: {
      'elk.algorithm': 'radial',
      'elk.spacing.nodeNode': '40',
      'elk.radial.radius': '200',
      'elk.padding': '[top=30,left=30,bottom=30,right=30]'
    },
    defaultNodeSize: { width: 100, height: 45 },
    tags: ['radial', 'knowledge-graph', 'concept-map']
  },

  compact: {
    name: '紧凑布局',
    description: '最小化空间占用, 适合空间受限的场景',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '15',
      'elk.layered.spacing.nodeNodeBetweenLayers': '40',
      'elk.spacing.edgeNode': '10',
      'elk.edgeRouting': 'POLYLINE',
      'elk.padding': '[top=10,left=10,bottom=10,right=10]'
    },
    defaultNodeSize: { width: 100, height: 35 },
    tags: ['compact', 'dense', 'space-saving']
  }
}

/** 所有预设名称列表 */
export const PRESET_NAMES = Object.keys(PRESETS)

/** 根据标签查找匹配的预设 */
export function findPresetByTag(tag: string): string | undefined {
  const lowerTag = tag.toLowerCase()
  for (const [name, preset] of Object.entries(PRESETS)) {
    if (preset.tags.some((t) => t.toLowerCase().includes(lowerTag))) {
      return name
    }
  }
  return undefined
}
