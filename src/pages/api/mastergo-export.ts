/**
 * POST /api/mastergo-export — Convert existing topology → MasterGo format (M14)
 *
 * Proxies to Python FastAPI backend (localhost:8000).
 * Takes a previous pipeline result (ELK + regions) and converts it
 * to MasterGo Import API format without re-running the LLM pipeline.
 *
 * Request:
 *   - elk: object (required) — ELK graph from previous pipeline run
 *   - regions: array (optional) — Region plans for layer-aware export
 *   - canvas_width: number (optional, default 900)
 *   - canvas_height: number (optional, default 500)
 *   - format: "import" | "layered" | "flat" (optional, default "import")
 *
 * Response:
 *   - success: boolean
 *   - format: string — the format used
 *   - data: object — MasterGo export data
 *   - stats: object — element/layer statistics
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { elk, regions, canvas_width, canvas_height, format } = body

    if (!elk || typeof elk !== 'object') {
      return new Response(
        JSON.stringify({ error: 'elk field is required (ELK graph object)' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 30000) // 30s — no LLM, just conversion

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/mastergo-export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          elk,
          regions: regions || [],
          canvas_width: canvas_width || 900,
          canvas_height: canvas_height || 500,
          format: format || 'import',
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
          JSON.stringify({ error: `Backend error: ${backendRes.status}`, details: errorDetail }),
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
          JSON.stringify({ error: 'Export timed out (30s)' }),
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
