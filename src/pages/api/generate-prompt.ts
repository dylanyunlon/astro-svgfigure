/**
 * POST /api/generate-prompt — Prompt engineering (Claude/Grok)
 *
 * Generates a detailed image generation prompt from SVG + method text.
 * User can review and edit the prompt before sending to Gemini 3 Image.
 *
 * GitHub 背书: ZeroLu/awesome-nanobanana-pro
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  // Create AbortController for timeout management
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 300000) // 5 min timeout

  try {
    const body = await request.json()

    if (!body.method_text || !body.svg_content) {
      clearTimeout(timeoutId)
      return new Response(
        JSON.stringify({ error: 'method_text and svg_content are required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const backendRes = await fetch(`${BACKEND_URL}/api/generate-prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method_text: body.method_text,
        svg_content: body.svg_content,
        model: body.model || null,
        reference_image_b64: body.reference_image_b64 || null,
        elk_graph: body.elk_graph || null,
      }),
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

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
    clearTimeout(timeoutId)
    
    if (err.name === 'AbortError') {
      return new Response(
        JSON.stringify({
          error: 'Prompt generation timed out (5 min)',
          hint: 'The AI model is taking too long. Try simplifying your input.',
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