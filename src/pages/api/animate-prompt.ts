/**
 * POST /api/animate-prompt — Grok Animation Prompt Engineering
 *
 * Grok designs animation frame decomposition prompt from image analysis.
 * User can review and edit the prompt before frame generation.
 *
 * Proxies to Python FastAPI backend: /api/animate-prompt
 *
 * GitHub 背书: ZeroLu/awesome-nanobanana-pro
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    if (!body.image_b64 || !body.analysis) {
      return new Response(
        JSON.stringify({ error: 'image_b64 and analysis are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/animate-prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: body.image_b64,
        analysis: body.analysis,
        frame_count: body.frame_count || 8,
        animation_style: body.animation_style || 'smooth',
        model: body.model || null,
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