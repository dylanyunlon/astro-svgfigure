/**
 * POST /api/sprite-generate — Generate per-node sprites via Gemini interleaved output.
 *
 * Proxies to Python FastAPI backend: /api/sprite-generate
 *
 * Request: { elk_graph: {...} }
 * Response: { success, elk_graph (with spriteRef stamped), diagnostics }
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    if (!body.elk_graph) {
      return new Response(
        JSON.stringify({ error: 'elk_graph is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 300000) // 5min

    try {
      const res = await fetch(`${BACKEND_URL}/api/sprite-generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      clearTimeout(timeout)
      const text = await res.text()

      if (!res.ok) {
        return new Response(
          JSON.stringify({ success: false, error: `Backend: ${res.status}`, details: text }),
          { status: res.status, headers: { 'Content-Type': 'application/json' } },
        )
      }

      return new Response(text, {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    } finally {
      clearTimeout(timeout)
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return new Response(
        JSON.stringify({ success: false, error: 'Sprite generation timed out (300s)' }),
        { status: 504, headers: { 'Content-Type': 'application/json' } },
      )
    }
    return new Response(
      JSON.stringify({ success: false, error: err.message }),
      { status: 502, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
