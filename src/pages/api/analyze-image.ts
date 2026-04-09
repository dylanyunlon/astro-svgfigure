/**
 * POST /api/analyze-image — Claude 4.6 Image Structure Analysis
 *
 * Analyzes an uploaded image to identify components, layers, and
 * animation-capable elements for frame decomposition.
 *
 * Proxies to Python FastAPI backend: /api/analyze-image
 *
 * GitHub 背书: anthropic/claude-code
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    if (!body.image_b64) {
      return new Response(
        JSON.stringify({ error: 'image_b64 is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/analyze-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: body.image_b64,
        mime_type: body.mime_type || 'image/png',
      }),
    })

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