/**
 * POST /api/topology-constrained — WHITE BOX topology generation
 *
 * LLM decides WHAT (entities, relationships, groups).
 * Constraint system decides WHERE and HOW BIG — deterministically.
 *
 * Request:
 *   - text: string (required)
 *   - model: string (optional)
 *   - output_format: "elk" | "mastergo" (optional)
 *   - canvas_width: number (optional, default 900)
 *   - canvas_height: number (optional, default 500)
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { text, model, output_format, canvas_width, canvas_height } = body

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'text field is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 180000)

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/topology-constrained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          model: model || '',
          output_format: output_format || 'elk',
          canvas_width: canvas_width || 900,
          canvas_height: canvas_height || 500,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        return new Response(
          JSON.stringify({ error: `Backend error: ${backendRes.status}`, details: errorText }),
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
          JSON.stringify({ error: 'Request timed out (180s)' }),
          { status: 504, headers: { 'Content-Type': 'application/json' } }
        )
      }
      throw fetchErr
    }
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: 'Failed to connect to backend', details: err.message }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
