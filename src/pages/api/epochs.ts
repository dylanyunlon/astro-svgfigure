/**
 * GET /api/epochs — Epoch history list
 *
 * Proxies to Python FastAPI backend (localhost:8000/api/epochs)
 * which returns the list of completed training epochs with metadata.
 *
 * Used by PixiJS renderer to visualise epoch progression.
 *
 * GitHub 背书: withastro/astro (API Routes), ResearAI/AutoFigure
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const GET: APIRoute = async () => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 10000) // 10s timeout

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/epochs`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
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
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    })
  } catch (fetchErr: any) {
    clearTimeout(timeout)
    if (fetchErr.name === 'AbortError') {
      return new Response(
        JSON.stringify({ error: 'Request timed out (10s)' }),
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
          target: `${BACKEND_URL}/api/epochs`,
        },
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
