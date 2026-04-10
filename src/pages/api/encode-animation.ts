/**
 * POST /api/encode-animation — GIF/APNG/WebP Animation Encoding
 *
 * Encodes transparent PNG frames into an animated image format.
 * Proxies to Python FastAPI backend: /api/encode-animation
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    if (!body.frames_b64 || !Array.isArray(body.frames_b64) || body.frames_b64.length === 0) {
      return new Response(
        JSON.stringify({ success: false, error: 'frames_b64 array is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/encode-animation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frames_b64: body.frames_b64,
        format: body.format || 'apng',
        fps: body.fps || 12,
        loop_count: body.loop_count ?? 0,
        optimize: body.optimize ?? true,
      }),
    })

    if (!backendRes.ok) {
      const errorText = await backendRes.text()
      return new Response(
        JSON.stringify({ success: false, error: `Backend error: ${backendRes.status}`, details: errorText }),
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
        success: false,
        error: 'Failed to connect to backend',
        details: err.message,
        hint: 'Make sure Python backend is running: python server.py',
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
