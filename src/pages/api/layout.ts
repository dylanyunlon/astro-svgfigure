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

    // Collect ALL valid node IDs recursively (including nested children)
    const nodeIds = new Set<string>()

    function collectNodeIds(nodes: any[]) {
      for (const node of nodes) {
        if (!node) continue
        const id = node.id || ''
        if (id) nodeIds.add(id)
        if (Array.isArray(node.children) && node.children.length > 0) {
          collectNodeIds(node.children)
        }
      }
    }
    collectNodeIds(children)

    // ── AUTO-CREATE missing nodes referenced by edges ──────────────────
    // The LLM often creates group containers (e.g., "input_group") as empty boxes
    // but then references sub-nodes (e.g., "source_context") in edges that don't
    // exist anywhere. We need to:
    // 1. Find all orphan node IDs referenced in edges
    // 2. Try to intelligently place them inside the most likely parent group
    // 3. Fall back to creating them as top-level nodes
    const orphanIds = new Set<string>()

    function collectOrphanIds(edgeList: any[]) {
      for (const edge of edgeList) {
        if (!edge) continue
        const sources = Array.isArray(edge.sources) ? edge.sources : []
        const targets = Array.isArray(edge.targets) ? edge.targets : []
        for (const s of sources) {
          if (typeof s === 'string' && s && !nodeIds.has(s)) orphanIds.add(s)
        }
        for (const t of targets) {
          if (typeof t === 'string' && t && !nodeIds.has(t)) orphanIds.add(t)
        }
      }
    }
    collectOrphanIds(edges)
    // Also check nested edges inside compound nodes
    function collectOrphanIdsFromNodes(nodes: any[]) {
      for (const node of nodes) {
        if (!node) continue
        if (Array.isArray(node.edges)) collectOrphanIds(node.edges)
        if (Array.isArray(node.children)) collectOrphanIdsFromNodes(node.children)
      }
    }
    collectOrphanIdsFromNodes(children)

    if (orphanIds.size > 0) {
      console.warn(
        `[layout] Auto-creating ${orphanIds.size} orphan nodes referenced in edges but missing from children:`,
        Array.from(orphanIds)
      )

      // Try to match orphan nodes to parent groups by name similarity
      // e.g., "source_context" might belong inside "input_group"
      // For now, we place them as top-level nodes so edges can route properly
      for (const orphanId of orphanIds) {
        // Check if any existing group node (with layoutOptions/padding) might be the parent
        // by checking if the orphan is semantically related
        let placed = false

        for (const node of children) {
          if (!node || !node.id) continue
          // Only place inside nodes that look like groups (have layoutOptions with padding)
          const hasGroupLayout = node.layoutOptions &&
            typeof node.layoutOptions === 'object' &&
            JSON.stringify(node.layoutOptions).includes('elk.padding')

          if (hasGroupLayout) {
            // Heuristic: check if orphan ID contains words from the group label or vice versa
            const groupLabel = (node.labels?.[0]?.text || node.id || '').toLowerCase()
            const orphanLower = orphanId.toLowerCase()

            // Simple keyword matching
            const groupWords = groupLabel.replace(/[^a-z0-9]/g, ' ').split(/\s+/).filter((w: string) => w.length > 2)
            const orphanWords = orphanLower.replace(/_/g, ' ').split(/\s+/).filter((w: string) => w.length > 2)

            // Check if group and orphan share semantic context
            // (This is a best-effort heuristic)
            const isRelated = groupWords.some((gw: string) =>
              orphanWords.some((ow: string) => gw.includes(ow) || ow.includes(gw))
            )

            if (isRelated) {
              if (!Array.isArray(node.children)) node.children = []
              node.children.push({
                id: orphanId,
                width: 140,
                height: 45,
                labels: [{ text: orphanId.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()) }],
              })
              nodeIds.add(orphanId)
              placed = true
              console.log(`[layout] Placed orphan "${orphanId}" inside group "${node.id}"`)
              break
            }
          }
        }

        if (!placed) {
          // Create as top-level node
          children.push({
            id: orphanId,
            width: 160,
            height: 50,
            labels: [{ text: orphanId.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()) }],
          })
          nodeIds.add(orphanId)
          console.log(`[layout] Created orphan "${orphanId}" as top-level node`)
        }
      }
    }

    // Recursively sanitize nodes: ensure every node has id, width, height,
    // and preserve compound node properties (layoutOptions, edges, group, borderless)
    function sanitizeNode(node: any, i: number): any {
      const id = node.id || `node_${i}`
      const result: any = {
        id,
        width: Number(node.width) || 160,
        height: Number(node.height) || 60,
        labels: Array.isArray(node.labels) && node.labels.length > 0
          ? node.labels
          : [{ text: id }],
      }

      // Preserve compound node layout options (critical for ELK padding/hierarchy)
      if (node.layoutOptions && typeof node.layoutOptions === 'object') {
        result.layoutOptions = node.layoutOptions
      }

      // Preserve group/borderless flags
      if (node.group) result.group = true
      if (node.borderless) result.borderless = true
      if (node.iconHint) result.iconHint = node.iconHint

      // Recursively sanitize nested children
      if (Array.isArray(node.children) && node.children.length > 0) {
        result.children = node.children.map((child: any, ci: number) => sanitizeNode(child, ci))

        // Sanitize nested edges inside compound nodes
        if (Array.isArray(node.edges) && node.edges.length > 0) {
          result.edges = sanitizeEdgeList(node.edges)
        }
      }

      return result
    }

    /**
     * Sanitize edges and SPLIT HYPEREDGES into 1-to-1 simple edges.
     *
     * ELK's layered algorithm throws "Hyperedges are not supported" when an
     * edge has multiple sources OR multiple targets. LLMs sometimes generate
     * edges like {sources: ["a","b"], targets: ["c"]} — these must be
     * decomposed into individual edges: a→c and b→c.
     */
    function sanitizeEdgeList(edgeList: any[]): any[] {
      const seenIds = new Set<string>()
      const result: any[] = []

      for (let i = 0; i < edgeList.length; i++) {
        const edge = edgeList[i]
        if (!edge) continue

        const sources = (Array.isArray(edge.sources) ? edge.sources : [])
          .filter((s: string) => typeof s === 'string' && nodeIds.has(s))
        const targets = (Array.isArray(edge.targets) ? edge.targets : [])
          .filter((t: string) => typeof t === 'string' && nodeIds.has(t))

        if (sources.length === 0 || targets.length === 0) continue

        // Decompose hyperedges (M sources × N targets) into M*N simple edges
        // Each simple edge has exactly 1 source and 1 target
        for (let si = 0; si < sources.length; si++) {
          for (let ti = 0; ti < targets.length; ti++) {
            const baseId = edge.id || `e_${i}`
            // Only append suffix if this is a decomposed hyperedge
            const needsSuffix = sources.length > 1 || targets.length > 1
            let id = needsSuffix ? `${baseId}_s${si}_t${ti}` : baseId
            if (seenIds.has(id)) {
              id = `${id}_dup_${i}`
            }
            seenIds.add(id)

            const simpleEdge: any = {
              id,
              sources: [sources[si]],
              targets: [targets[ti]],
            }
            // Preserve labels only on the first decomposed edge
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

    const sanitizedChildren = children.map((node: any, i: number) => sanitizeNode(node, i))

    // Sanitize top-level edges (validate against ALL node IDs including nested)
    const sanitizedEdges = sanitizeEdgeList(edges)

    const processedGraph = {
      id: graph.id || 'root',
      layoutOptions,
      children: sanitizedChildren,
      edges: sanitizedEdges,
    }

    // ── Diagnostic: log edge statistics ──────────────────────────────
    const totalEdges = sanitizedEdges.length
    const hyperEdges = sanitizedEdges.filter((e: any) =>
      (e.sources?.length || 0) > 1 || (e.targets?.length || 0) > 1
    ).length
    if (hyperEdges > 0) {
      // This should never happen after our fix, but log if it does
      console.error(`[layout] BUG: ${hyperEdges}/${totalEdges} hyperedges survived sanitization!`)
    }
    console.log(`[layout] Passing to ELK: ${sanitizedChildren.length} top-level nodes, ${totalEdges} edges`)

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