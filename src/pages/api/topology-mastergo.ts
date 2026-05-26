/**
 * POST /api/topology-mastergo — Text + Screenshot → Mastergo-quality Topology
 *
 * Proxies to Python FastAPI backend (localhost:8000).
 * 8-step pipeline producing 50+ elements with pixel-precise bboxes.
 *
 * Request:
 *   - text: string (required) — paper method description
 *   - image_b64: string (optional) — screenshot for vision-guided layout
 *   - model: string (optional) — LLM model override
 *   - output_format: "elk" | "mastergo" (optional, default "elk")
 *
 * Response:
 *   - success: boolean
 *   - elk: ELK graph with dense nodes
 *   - mastergo?: flat element list with bboxes (if output_format="mastergo")
 *   - diagnostics: per-step stats
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { text, image_b64, model, output_format } = body

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'text field is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Mastergo pipeline needs more time: dense extraction + vision constraint
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 300000) // 5 min timeout

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/topology-mastergo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          image_b64: image_b64 || '',
          model: model || '',
          output_format: output_format || 'elk',
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        let errorDetail: string
        try {
          const errorJson = JSON.parse(errorText)
          errorDetail = errorJson.error || errorJson.detail || errorText
        } catch {
          errorDetail = errorText
        }
        return new Response(
          JSON.stringify({
            error: `Backend error: ${backendRes.status}`,
            details: errorDetail,
          }),
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
          JSON.stringify({
            error: 'Request timed out (300s)',
            hint: 'Mastergo pipeline processes many sub-elements. Try a shorter description.',
          }),
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
