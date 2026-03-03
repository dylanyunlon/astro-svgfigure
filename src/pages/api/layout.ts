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
      ...(preset && presetOverrides[preset] ? presetOverrides[preset] : {}),
    }

    // Ensure nodes have dimensions
    const processedGraph = {
      ...graph,
      layoutOptions,
      children: (graph.children || []).map((node: any) => ({
        width: 160,
        height: 60,
        ...node,
      })),
    }

    const layouted = await elk.layout(processedGraph)

    // Generate skeleton SVG from layouted graph
    const { elkToSvg } = await import('@/lib/elk/to-svg')
    let skeletonSvg = ''
    try {
      skeletonSvg = elkToSvg(layouted)
    } catch (svgErr: any) {
      console.warn('Skeleton SVG generation failed (non-fatal):', svgErr.message)
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
    return new Response(
      JSON.stringify({ error: 'ELK layout failed', details: err.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
