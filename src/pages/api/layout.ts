/**
 * POST /api/layout — Topology JSON → ELK Layouted JSON
 *
 * Runs ELK.js constraint solver on the server side (Astro SSR)
 * Uses the project's src/lib/elk/layout.ts module
 *
 * GitHub 背书: kieler/elkjs, withastro/astro (API Routes)
 */
import type { APIRoute } from 'astro'

export const prerender = false

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { graph, preset, options } = body

    if (!graph || typeof graph !== 'object') {
      return new Response(
        JSON.stringify({ error: 'graph field is required (ELK JSON)' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // ── Validate & sanitize graph before passing to ELK ──────────────
    const children = Array.isArray(graph.children) ? graph.children : []
    const edges = Array.isArray(graph.edges) ? graph.edges : []

    if (children.length === 0) {
      return new Response(
        JSON.stringify({
          error: 'graph.children is empty — need at least one node',
          debug: { receivedKeys: Object.keys(graph) },
        }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Dynamic import to avoid bundling issues
    const ELK = (await import('elkjs/lib/elk.bundled.js')).default
    const elk = new ELK()

    // Merge layout options
    const defaultOptions: Record<string, string> = {
      'elk.algorithm': options?.algorithm || 'layered',
      'elk.direction': options?.direction || 'DOWN',
      'elk.spacing.nodeNode': String(options?.nodeSpacing || 50),
      'elk.layered.spacing.nodeNodeBetweenLayers': String(options?.layerSpacing || 80),
      'elk.spacing.edgeNode': String(options?.edgeNodeSpacing || 30),
      'elk.edgeRouting': options?.edgeRouting || 'ORTHOGONAL',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    }

    // Apply preset overrides
    const presetOverrides: Record<string, Record<string, string>> = {
      'academic-paper': {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
        'elk.spacing.nodeNode': '60',
        'elk.layered.spacing.nodeNodeBetweenLayers': '100',
      },
      'neural-network': {
        'elk.algorithm': 'layered',
        'elk.direction': 'RIGHT',
        'elk.spacing.nodeNode': '40',
        'elk.layered.spacing.nodeNodeBetweenLayers': '120',
      },
      'flowchart': {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
        'elk.spacing.nodeNode': '50',
        'elk.layered.spacing.nodeNodeBetweenLayers': '80',
      },
    }

    const layoutOptions = {
      ...defaultOptions,
      // Also merge any layoutOptions the LLM put on the graph itself
      ...(graph.layoutOptions || {}),
      ...(preset && presetOverrides[preset] ? presetOverrides[preset] : {}),
    }

    // Collect valid node IDs for edge validation
    const nodeIds = new Set<string>()

    // Sanitize nodes: ensure every child has id, width, height
    const sanitizedChildren = children.map((node: any, i: number) => {
      const id = node.id || `node_${i}`
      nodeIds.add(id)
      return {
        id,
        width: Number(node.width) || 160,
        height: Number(node.height) || 60,
        labels: Array.isArray(node.labels) && node.labels.length > 0
          ? node.labels
          : [{ text: id }],
        // Preserve any nested children
        ...(Array.isArray(node.children) && node.children.length > 0
          ? { children: node.children }
          : {}),
      }
    })

    // Sanitize edges: ensure sources/targets are valid arrays referencing existing nodes
    const seenEdgeIds = new Set<string>()
    const sanitizedEdges = edges
      .filter((edge: any) => {
        const sources = Array.isArray(edge.sources) ? edge.sources : []
        const targets = Array.isArray(edge.targets) ? edge.targets : []
        // Skip edges with no valid endpoints
        if (sources.length === 0 || targets.length === 0) return false
        // Skip edges referencing non-existent nodes
        const allValid = sources.every((s: string) => nodeIds.has(s)) &&
                         targets.every((t: string) => nodeIds.has(t))
        return allValid
      })
      .map((edge: any, i: number) => {
        let id = edge.id || `e_${i}`
        // Deduplicate edge IDs
        if (seenEdgeIds.has(id)) {
          id = `${id}_dup_${i}`
        }
        seenEdgeIds.add(id)
        return {
          id,
          sources: Array.isArray(edge.sources) ? edge.sources : [],
          targets: Array.isArray(edge.targets) ? edge.targets : [],
          ...(edge.advanced ? { advanced: edge.advanced } : {}),
          ...(edge.labels ? { labels: edge.labels } : {}),
        }
      })

    const processedGraph = {
      id: graph.id || 'root',
      layoutOptions,
      children: sanitizedChildren,
      edges: sanitizedEdges,
    }

    const layouted = await elk.layout(processedGraph)

    // Generate skeleton SVG from layouted graph (non-fatal if it fails)
    let skeletonSvg = ''
    try {
      // Use relative import to avoid Vite alias resolution issues in SSR
      const { elkToSvg } = await import('../../lib/elk/to-svg')
      if (layouted && layouted.children && layouted.children.length > 0) {
        skeletonSvg = elkToSvg(layouted)
        if (!skeletonSvg || skeletonSvg.length < 50) {
          console.warn('elkToSvg returned empty/tiny result, layouted:', JSON.stringify(layouted).slice(0, 200))
        }
      } else {
        console.warn('No children in layouted graph — cannot generate skeleton SVG')
      }
    } catch (svgErr: any) {
      console.error('Skeleton SVG generation failed:', svgErr.message, svgErr.stack?.slice(0, 300))
    }

    return new Response(
      JSON.stringify({
        success: true,
        layouted,
        skeletonSvg,
        options: layoutOptions,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (err: any) {
    console.error('ELK layout error:', err)
    return new Response(
      JSON.stringify({
        error: 'ELK layout failed',
        details: err.message,
        hint: 'The topology JSON may have invalid node/edge format. Check that all edge sources/targets reference valid node IDs.',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}