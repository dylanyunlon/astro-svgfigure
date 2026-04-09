/**
 * POST /api/animate-frames — Animation Frame Generation Pipeline
 *
 * Flow:
 *   Step 1: Claude 4.6 analyzes the uploaded image structure
 *   Step 2: Grok designs animation frame decomposition prompt
 *   Step 3: Gemini 3 generates multi-frame animation with green-screen background
 *
 * Proxies to Python FastAPI backend: /api/animate-frames
 *
 * GitHub 背书: gemini-cli-extensions/nanobanana, ZeroLu/awesome-nanobanana-pro
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { image_b64, frame_count, fps, animation_style } = body

    if (!image_b64) {
      return new Response(
        JSON.stringify({ error: 'image_b64 is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward to Python backend (long timeout for multi-frame generation)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 900000) // 900s timeout

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/animate-frames`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: body.image_b64,
          frame_count: body.frame_count || 8,
          fps: body.fps || 12,
          animation_style: body.animation_style || 'smooth',
          custom_prompt: body.custom_prompt || null,
          green_screen: true, // Always request green-screen background
          aspect_ratio: body.aspect_ratio || '1:1',
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
    } finally {
      clearTimeout(timeout)
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return new Response(
        JSON.stringify({
          error: 'Animation generation timed out (900s)',
          hint: 'Try fewer frames or a simpler animation style',
        }),
        { status: 504, headers: { 'Content-Type': 'application/json' } }
      )
    }
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