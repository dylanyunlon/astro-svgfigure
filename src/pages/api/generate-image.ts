/**
 * POST /api/generate-image — Step 5: SVG → Gemini 3 Pro Image
 *
 * Pipeline:
 *   a) Grok 4 reverse-engineers a professional prompt from SVG + method text
 *   b) Gemini 3 Pro Image generates publication-quality scientific figure
 *
 * Proxies to Python FastAPI backend: /api/generate-image
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
    const { svg_content, method_text } = body

    if (!svg_content || !method_text) {
      return new Response(
        JSON.stringify({
          error: 'svg_content and method_text are required',
        }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward to Python backend (long timeout for image generation)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 150000) // 150s timeout

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/generate-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          svg_content: body.svg_content,
          method_text: body.method_text,
          reference_image_b64: body.reference_image_b64 || null,
          prompt_model: body.prompt_model || null,
          image_model: body.image_model || 'gemini-3-pro-image-preview',
          aspect_ratio: body.aspect_ratio || '16:9',
          image_size: body.image_size || '4K',
          custom_prompt: body.custom_prompt || null,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        return new Response(
          JSON.stringify({
            error: `Backend error: ${backendRes.status}`,
            details: errorText,
          }),
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
          error: 'Image generation timed out (150s)',
          hint: 'Try a simpler description or a faster model',
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
