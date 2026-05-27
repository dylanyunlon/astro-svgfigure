/**
 * POST /api/topology-layered — Text → Layered Multi-Region Topology
 *
 * Proxies to Python backend's layered pipeline:
 *   intent_parse → region_plan → per_region_gen → compose
 *
 * Returns: { elk, layers, regions, cross_region_edges, diagnostics }
 *
 * This is the multi-pass pipeline that replaces the monolithic
 * /api/topology endpoint.  Each region is generated independently
 * and composed into a unified canvas with layer metadata.
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL ||
  import.meta.env.BACKEND_URL ||
  'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { text, model, canvas_width, canvas_height, max_regions, output_format } = body

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'text field is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 240000) // 4 min

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/topology-layered`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          model: model || '',
          canvas_width: canvas_width || null,
          canvas_height: canvas_height || null,
          max_regions: max_regions || 8,
          output_format: output_format || 'elk',
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        let detail: string
        try {
          detail = JSON.parse(errorText).error || errorText
        } catch {
          detail = errorText
        }
        return new Response(
          JSON.stringify({ error: `Backend error: ${backendRes.status}`, details: detail }),
          { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
        )
      }

      const data = await backendRes.json()
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    } catch (fetchErr: any) {
      clearTimeout(timeout)
      if (fetchErr.name === 'AbortError') {
        return new Response(
          JSON.stringify({ error: 'Request timed out (240s)' }),
          { status: 504, headers: { 'Content-Type': 'application/json' } }
        )
      }
      throw fetchErr
    }
  } catch (err: any) {
    return new Response(
      JSON.stringify({
        error: 'Failed to connect to backend',
        details: err.message,
        hint: 'Make sure Python backend is running: python server.py',
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
