/**
 * GET /api/animation-health — Animation Pipeline Health Check
 *
 * Reports animation pipeline capabilities: rembg, encoders, models.
 * Proxies to Python FastAPI backend: /api/animation-health
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const GET: APIRoute = async () => {
  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/animation-health`)

    if (!backendRes.ok) {
      return new Response(
        JSON.stringify({ backend: false, animation_pipeline: false, error: `Backend: ${backendRes.status}` }),
        { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const data = await backendRes.json()
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (err: any) {
    return new Response(
      JSON.stringify({
        backend: false,
        animation_pipeline: false,
        error: err.message,
        hint: 'Python backend not running: python server.py',
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
