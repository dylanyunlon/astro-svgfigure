/**
 * index.ts — Transplant barrel export + global debug harness
 *
 * Exposes all transplanted modules from commit 08394c8 and provides
 * a global debug surface for browser console inspection:
 *
 *   window.__transplantDebug.printAll()
 *   window.__transplantDebug.adapterTrace()
 *   window.__transplantDebug.routeTrace()
 *   window.__transplantDebug.canvasState()
 *   window.__transplantDebug.mutationHistory()
 *
 * This file also wires up the modules so they work together
 * as a cohesive system (adapter → editor → route handler).
 */

// ── Re-exports ─────────────────────────────────────────────
export {
  elkToInteractive,
  interactiveToElk,
  enableAdapterTrace,
  flushAdapterTrace,
  dumpAdapterState,
} from './elk-graph-adapter'

export {
  CanvasEditor,
  type InteractiveNode,
  type InteractiveEdge,
  type InteractiveGraph,
  type CanvasEditorOptions,
} from './canvas-editor'

export {
  routeSpriteResult,
  handleSpriteResponse,
  getSpriteRouteTrace,
  clearSpriteRouteTrace,
  printSpriteRouteTrace,
  type SpriteRouteOptions,
} from './sprite-route-handler'

// ── Integration test helper ────────────────────────────────
// Creates a mock ELK graph and runs it through the full pipeline
// so you can verify the transplant works end-to-end.

import { elkToInteractive, enableAdapterTrace, flushAdapterTrace, dumpAdapterState } from './elk-graph-adapter'
import { CanvasEditor } from './canvas-editor'
import { printSpriteRouteTrace, clearSpriteRouteTrace } from './sprite-route-handler'

/**
 * Run a self-test of the transplanted pipeline.
 * Call from browser console: window.__transplantDebug.selfTest(containerEl)
 */
export function selfTest(container?: HTMLElement): {
  graph: ReturnType<typeof elkToInteractive>
  traceEntries: number
  nodeCount: number
  edgeCount: number
} {
  console.group('[transplant] self-test')

  // 1. Build a mock ELK graph with operator + sprite + regular nodes
  const mockElk = {
    id: 'root',
    width: 600,
    height: 400,
    children: [
      {
        id: 'input_layer',
        x: 50, y: 50, width: 160, height: 60,
        labels: [{ text: 'Input Layer' }],
        renderMode: 'text',
      },
      {
        id: 'conv_op',
        x: 280, y: 50, width: 50, height: 50,
        labels: [{ text: '⊗' }],
        isOperator: true,
        renderMode: 'kernel',
      },
      {
        id: 'feature_map',
        x: 400, y: 30, width: 140, height: 100,
        labels: [{ text: 'Feature Map' }],
        renderMode: 'sprite',
        spriteRef: {
          format: 'png',
          url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==', // tiny placeholder
        },
      },
      {
        id: 'output_group',
        x: 50, y: 200, width: 500, height: 150,
        labels: [{ text: 'Output Group' }],
        children: [
          {
            id: 'softmax',
            x: 30, y: 40, width: 120, height: 50,
            labels: [{ text: 'Softmax' }],
          },
          {
            id: 'argmax_op',
            x: 200, y: 40, width: 50, height: 50,
            labels: [{ text: '⊕' }],
            isOperator: true,
          },
        ],
        edges: [
          {
            id: 'e_softmax_argmax',
            sources: ['softmax'],
            targets: ['argmax_op'],
            sections: [{
              startPoint: { x: 150, y: 65 },
              endPoint: { x: 200, y: 65 },
            }],
          },
        ],
      },
    ],
    edges: [
      {
        id: 'e_input_conv',
        sources: ['input_layer'],
        targets: ['conv_op'],
        sections: [{
          startPoint: { x: 210, y: 80 },
          endPoint: { x: 280, y: 75 },
          bendPoints: [{ x: 245, y: 80 }, { x: 245, y: 75 }],
        }],
      },
      {
        id: 'e_conv_feature',
        sources: ['conv_op'],
        targets: ['feature_map'],
        sections: [{
          startPoint: { x: 330, y: 75 },
          endPoint: { x: 400, y: 80 },
        }],
      },
      {
        id: 'e_feature_output',
        sources: ['feature_map'],
        targets: ['output_group'],
        sections: [{
          startPoint: { x: 470, y: 130 },
          endPoint: { x: 300, y: 200 },
          bendPoints: [{ x: 470, y: 165 }, { x: 300, y: 165 }],
        }],
      },
    ],
  }

  // 2. Run adapter with tracing
  enableAdapterTrace()
  const interactiveGraph = elkToInteractive(mockElk as any)
  const traceEntries = flushAdapterTrace()

  console.log('adapter trace:', traceEntries.length, 'entries')
  console.log('graph:', interactiveGraph.nodes.length, 'nodes,', interactiveGraph.edges.length, 'edges')

  // 3. Dump adapter state
  dumpAdapterState(interactiveGraph)

  // 4. If container provided, mount the editor
  if (container) {
    const editor = new CanvasEditor(container, interactiveGraph, {
      traceRingSize: 100,
    })
    console.log('editor mounted — drag nodes, double-click to edit labels')
    console.log('try: editor.printCanvasState(), editor.printNodeTree()')

    // Expose editor on window for console access
    ;(window as any).__transplantEditor = editor
  } else {
    console.log('no container provided — skipping editor mount')
    console.log('pass a DOM element to mount: selfTest(document.getElementById("my-container"))')
  }

  console.groupEnd()

  return {
    graph: interactiveGraph,
    traceEntries: traceEntries.length,
    nodeCount: interactiveGraph.nodes.length,
    edgeCount: interactiveGraph.edges.length,
  }
}

// ── Global debug surface ───────────────────────────────────
// Attaches to window so you can inspect from the browser console.

if (typeof window !== 'undefined') {
  const debug = {
    selfTest,
    adapterTrace: flushAdapterTrace,
    routeTrace: printSpriteRouteTrace,
    clearRouteTrace: clearSpriteRouteTrace,

    /** Print editor state (requires editor to be mounted) */
    canvasState: () => {
      const editor = (window as any).__transplantEditor as CanvasEditor | undefined
      if (editor) editor.printCanvasState()
      else console.warn('no editor mounted — run selfTest(container) first')
    },

    /** Print editor node tree */
    nodeTree: () => {
      const editor = (window as any).__transplantEditor as CanvasEditor | undefined
      if (editor) editor.printNodeTree()
      else console.warn('no editor mounted')
    },

    /** Print mutation history */
    mutationHistory: () => {
      const editor = (window as any).__transplantEditor as CanvasEditor | undefined
      if (editor) editor.printMutationHistory()
      else console.warn('no editor mounted')
    },

    /** Print everything */
    printAll: () => {
      console.group('[transplant] full debug dump')
      debug.canvasState()
      debug.nodeTree()
      debug.mutationHistory()
      debug.routeTrace()
      console.groupEnd()
    },
  }

  ;(window as any).__transplantDebug = debug
  console.log('[transplant] debug surface ready — try window.__transplantDebug.printAll()')
}
