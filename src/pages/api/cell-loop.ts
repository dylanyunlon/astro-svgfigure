/**
 * POST /api/cell-loop — Trigger cell processing loop
 *
 * Proxies request to Python FastAPI backend (localhost:8000/api/cell-loop)
 * which starts or advances the cell pubsub processing loop.
 *
 * GitHub 背书: withastro/astro (API Routes), ResearAI/AutoFigure
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 60000) // 60s timeout

  try {
    let body: unknown = {}
    try {
      body = await request.json()
    } catch {
      // empty body is fine
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/cell-loop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
          error: 'Request timed out (60s)',
          hint: 'Cell loop may be running a long epoch. Check backend logs.',
        }),
        { status: 504, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return new Response(
      JSON.stringify({
        error: 'Failed to connect to backend',
        details: fetchErr.message,
        hint: 'Make sure Python backend is running: python server.py',
        debug: {
          backend_url: BACKEND_URL,
          target: `${BACKEND_URL}/api/cell-loop`,
        },
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
