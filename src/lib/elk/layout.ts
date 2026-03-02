/**
 * ELK.js 布局引擎封装
 *
 * 核心文件: 将 LLM 生成的零坐标拓扑 JSON 通过 elkjs 约束求解器
 * 计算出精确的像素位置。
 *
 * 参考实现:
 * - kieler/elkjs: https://github.com/kieler/elkjs
 * - ReactFlow ELK 示例: https://reactflow.dev/examples/layout/elkjs
 * - xyflow/xyflow: https://github.com/xyflow/xyflow
 * - Eclipse ELK 算法: https://www.eclipse.org/elk/reference/algorithms.html
 *
 * 使用方式:
 * ```ts
 * import { layoutGraph, layoutWithPreset } from '@elk/layout'
 *
 * const result = await layoutGraph(topologyJson)
 * // result.graph.children[i] => { id, x, y, width, height, ... }
 * // result.graph.edges[i].sections => 边的路由点
 *
 * const result2 = await layoutWithPreset(topologyJson, 'academic-paper')
 * ```
 */

import ELK from 'elkjs/lib/elk.bundled.js'

import { DEFAULT_LAYOUT_OPTIONS, DEFAULT_NODE_SIZE } from './constants'
import { PRESETS } from './presets'
import type {
  ElkAlgorithm,
  ElkDirection,
  ElkGraph,
  ElkLayoutOptions,
  ElkNode,
  LayoutedGraph,
  LayoutResult
} from './types'

// ============================================================
// ELK 实例 (单例)
// ============================================================

let elkInstance: InstanceType<typeof ELK> | null = null

function getElk(): InstanceType<typeof ELK> {
  if (!elkInstance) {
    elkInstance = new ELK()
  }
  return elkInstance
}

// ============================================================
// 核心布局函数
// ============================================================

/**
 * 对拓扑 JSON 执行 ELK 布局
 *
 * @param graph - LLM 生成的零坐标拓扑 JSON (ElkGraph 格式)
 * @param options - 可选的 ELK 布局选项 (覆盖默认值)
 * @returns 布局结果, 包含精确坐标的图和元数据
 *
 * @example
 * ```ts
 * const topology = {
 *   id: "root",
 *   children: [
 *     { id: "input", width: 150, height: 50, labels: [{ text: "Input" }] },
 *     { id: "encoder", width: 150, height: 50, labels: [{ text: "Encoder" }] }
 *   ],
 *   edges: [
 *     { id: "e1", sources: ["input"], targets: ["encoder"] }
 *   ]
 * }
 * const result = await layoutGraph(topology)
 * console.log(result.graph.children[0].x, result.graph.children[0].y)
 * ```
 */
export async function layoutGraph(
  graph: ElkGraph,
  options?: ElkLayoutOptions
): Promise<LayoutResult> {
  const elk = getElk()
  const startTime = performance.now()

  // 合并布局选项: 默认 → 图自带 → 参数覆盖
  const mergedOptions: ElkLayoutOptions = {
    ...DEFAULT_LAYOUT_OPTIONS,
    ...(graph.layoutOptions || {}),
    ...(options || {})
  }

  // 预处理: 确保所有节点有宽高
  const preparedGraph = prepareGraph(graph, mergedOptions)

  // 执行 ELK 布局
  const layoutedGraph = (await elk.layout(preparedGraph)) as LayoutedGraph

  const duration = performance.now() - startTime
  const bounds = computeBounds(layoutedGraph)
  const algorithm = (mergedOptions['elk.algorithm'] || 'layered') as ElkAlgorithm

  return {
    graph: layoutedGraph,
    duration,
    algorithm,
    bounds
  }
}

/**
 * 使用预设配置进行布局
 *
 * @param graph - 拓扑 JSON
 * @param presetName - 预设名称 (academic-paper / flowchart / neural-net / architecture)
 * @param overrides - 额外覆盖选项
 */
export async function layoutWithPreset(
  graph: ElkGraph,
  presetName: string,
  overrides?: ElkLayoutOptions
): Promise<LayoutResult> {
  const preset = PRESETS[presetName]
  if (!preset) {
    console.warn(`Unknown preset "${presetName}", falling back to defaults`)
    return layoutGraph(graph, overrides)
  }

  const options: ElkLayoutOptions = {
    ...preset.layoutOptions,
    ...(overrides || {})
  }

  // 应用预设的默认节点尺寸 (仅对缺少尺寸的节点)
  const preparedGraph = {
    ...graph,
    children: (graph.children || []).map((node) => ({
      ...node,
      width: node.width || preset.defaultNodeSize.width,
      height: node.height || preset.defaultNodeSize.height
    }))
  }

  return layoutGraph(preparedGraph, options)
}

/**
 * 快速布局: 指定算法和方向
 */
export async function quickLayout(
  graph: ElkGraph,
  algorithm: ElkAlgorithm = 'layered',
  direction: ElkDirection = 'DOWN'
): Promise<LayoutResult> {
  return layoutGraph(graph, {
    'elk.algorithm': algorithm,
    'elk.direction': direction
  })
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 预处理图: 确保所有节点都有宽高, 递归处理子图
 */
function prepareGraph(graph: ElkGraph, options: ElkLayoutOptions): ElkGraph {
  return {
    ...graph,
    layoutOptions: options,
    children: (graph.children || []).map((node) => prepareNode(node)),
    edges: graph.edges || []
  }
}

/**
 * 预处理单个节点
 */
function prepareNode(node: ElkNode): ElkNode {
  const prepared: ElkNode = {
    ...node,
    width: node.width || DEFAULT_NODE_SIZE.width,
    height: node.height || DEFAULT_NODE_SIZE.height
  }

  // 递归处理子节点 (compound graph)
  if (node.children && node.children.length > 0) {
    prepared.children = node.children.map((child) => prepareNode(child))
  }

  return prepared
}

/**
 * 计算布局后图的包围盒
 */
function computeBounds(graph: LayoutedGraph): {
  x: number
  y: number
  width: number
  height: number
} {
  if (!graph.children || graph.children.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const node of graph.children) {
    const nx = node.x ?? 0
    const ny = node.y ?? 0
    const nw = node.width ?? DEFAULT_NODE_SIZE.width
    const nh = node.height ?? DEFAULT_NODE_SIZE.height

    minX = Math.min(minX, nx)
    minY = Math.min(minY, ny)
    maxX = Math.max(maxX, nx + nw)
    maxY = Math.max(maxY, ny + nh)
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  }
}

/**
 * 销毁 ELK 实例 (清理 Web Worker 如果有)
 */
export function destroyElk(): void {
  if (elkInstance) {
    // elkjs 没有显式 destroy, 但置空引用让 GC 回收
    elkInstance = null
  }
}
