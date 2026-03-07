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
     * Sanitize edges and SPLIT HYPEREDGES into 1-to-1 simple edges.
     * ELK's layered algorithm throws "Hyperedges are not supported" when an
     * edge has multiple sources OR multiple targets.
     */
    function sanitizeEdgeList(
      edgeList: any[],
      idPrefix: string = 'e',
      _depth: number = 0,
    ): any[] {
      const seenEdgeIds = new Set<string>()
      const result: any[] = []

      for (let i = 0; i < edgeList.length; i++) {
        const edge = edgeList[i]
        if (!edge) continue

        const sources = (Array.isArray(edge.sources) ? edge.sources : [])
          .filter((s: string) => typeof s === 'string' && nodeIds.has(s))
        const targets = (Array.isArray(edge.targets) ? edge.targets : [])
          .filter((t: string) => typeof t === 'string' && nodeIds.has(t))

        if (sources.length === 0 || targets.length === 0) continue

        // Decompose hyperedges into simple 1-to-1 edges
        for (let si = 0; si < sources.length; si++) {
          for (let ti = 0; ti < targets.length; ti++) {
            const baseId = edge.id || `${idPrefix}_${i}`
            const needsSuffix = sources.length > 1 || targets.length > 1
            let id = needsSuffix ? `${baseId}_s${si}_t${ti}` : baseId
            if (seenEdgeIds.has(id)) {
              id = `${id}_dup_${i}`
            }
            seenEdgeIds.add(id)

            const simpleEdge: any = {
              id,
              sources: [sources[si]],
              targets: [targets[ti]],
            }
            if (si === 0 && ti === 0) {
              if (edge.advanced) simpleEdge.advanced = edge.advanced
              if (edge.labels) simpleEdge.labels = edge.labels
            }
            result.push(simpleEdge)
          }
        }
      }

      return result
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