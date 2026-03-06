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

    // Collect ALL valid node IDs recursively (including nested compound children)
    const nodeIds = new Set<string>()

    /**
     * Recursively sanitize a list of nodes, preserving compound node properties
     * (layoutOptions, group, borderless, nested edges, iconHint) that ELK needs
     * for proper hierarchical layout.
     */
    function sanitizeNodeList(nodes: any[], depth: number = 0): any[] {
      const seenIds = new Set<string>()
      return nodes.map((node: any, i: number) => {
        let id = node.id || `node_d${depth}_${i}`
        // Deduplicate IDs at this level
        if (seenIds.has(id) || nodeIds.has(id)) {
          id = `${id}_d${depth}_${i}`
        }
        seenIds.add(id)
        nodeIds.add(id)

        const hasChildren = Array.isArray(node.children) && node.children.length > 0
        const isGroup = hasChildren || node.group

        const sanitized: Record<string, any> = {
          id,
          width: Number(node.width) || (isGroup ? 250 : 160),
          height: Number(node.height) || (isGroup ? 200 : 60),
          labels: Array.isArray(node.labels) && node.labels.length > 0
            ? node.labels
            : [{ text: id.replace(/_/g, ' ') }],
        }

        // Preserve compound node properties essential for ELK layout
        if (node.layoutOptions) sanitized.layoutOptions = node.layoutOptions
        if (node.group) sanitized.group = true
        if (node.borderless) sanitized.borderless = true
        if (node.iconHint) sanitized.iconHint = node.iconHint

        // Recursively sanitize nested children
        if (hasChildren) {
          sanitized.children = sanitizeNodeList(node.children, depth + 1)

          // Ensure compound nodes have padding layoutOptions for proper ELK sizing
          if (!sanitized.layoutOptions) {
            sanitized.layoutOptions = {
              'elk.padding': '[top=30,left=10,bottom=10,right=10]',
            }
          }

          // Sanitize nested edges within compound nodes
          if (Array.isArray(node.edges) && node.edges.length > 0) {
            sanitized.edges = sanitizeEdgeList(
              node.edges, `inner_${id}_e`, depth
            )
          }
        }

        return sanitized
      })
    }

    /**
     * Sanitize a list of edges, validating source/target references.
     * Uses the global nodeIds set which is populated during node sanitization.
     */
    function sanitizeEdgeList(
      edgeList: any[],
      idPrefix: string = 'e',
      _depth: number = 0,
    ): any[] {
      const seenEdgeIds = new Set<string>()
      return edgeList
        .filter((edge: any) => {
          const sources = Array.isArray(edge.sources) ? edge.sources : []
          const targets = Array.isArray(edge.targets) ? edge.targets : []
          if (sources.length === 0 || targets.length === 0) return false
          // Validate all endpoints reference existing nodes
          const allValid = sources.every((s: string) => nodeIds.has(s)) &&
                           targets.every((t: string) => nodeIds.has(t))
          return allValid
        })
        .map((edge: any, i: number) => {
          let id = edge.id || `${idPrefix}_${i}`
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
    }

    // Sanitize nodes first (populates nodeIds with ALL IDs including nested)
    const sanitizedChildren = sanitizeNodeList(children)

    // Then sanitize top-level edges (can now reference nested node IDs too)
    const sanitizedEdges = sanitizeEdgeList(edges, 'e')

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