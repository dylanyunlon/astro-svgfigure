/**
 * POST /api/generate-image-enriched — Sprite-enriched figure generation (proxy)
 *
 * Proxies to Python backend: /api/generate-image-enriched
 *
 * The frontend sends a LIGHTWEIGHT payload (~30 KB):
 *   - svg_content: stripped (no base64 images)
 *   - elk_graph: spriteRef.url replaced with '__cached__'
 *   - sprite_cache_key: key from /api/sprite-generate
 *   - method_text, custom_prompt, aspect_ratio, etc.
 *
 * The Python backend restores sprite base64 from its server-side cache
 * and passes them to Gemini as separate image inputs.
 *
 * This avoids the nginx 413 error — sprites never transit through nginx.
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { method_text } = body

    if (!method_text) {
      return new Response(
        JSON.stringify({ error: 'method_text is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward to Python backend (long timeout for image generation)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 600000)

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/generate-image-enriched`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          svg_content: body.svg_content || '',
          method_text: body.method_text,
          custom_prompt: body.custom_prompt || null,
          aspect_ratio: body.aspect_ratio || '16:9',
          image_size: body.image_size || '4K',
          elk_graph: body.elk_graph || null,
          sprite_cache_key: body.sprite_cache_key || '',
          skeleton_media_resolution: body.skeleton_media_resolution || null,
          sprite_media_resolution: body.sprite_media_resolution || null,
          image_model: body.image_model || 'gemini-3-pro-image-preview',
          prompt_model: body.prompt_model || null,
          reference_image_b64: body.reference_image_b64 || null,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      const responseText = await backendRes.text()

      if (!backendRes.ok) {
        return new Response(
          JSON.stringify({
            success: false,
            error: `Backend error: ${backendRes.status}`,
            details: responseText.slice(0, 500),
          }),
          { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
        )
      }

      if (!responseText || responseText.trim() === '') {
        return new Response(
          JSON.stringify({
            success: false,
            error: 'Backend returned empty response',
          }),
          { status: 500, headers: { 'Content-Type': 'application/json' } }
        )
      }

      let data: unknown
      try {
        data = JSON.parse(responseText)
      } catch (parseErr: any) {
        return new Response(
          JSON.stringify({
            success: false,
            error: 'Backend returned invalid JSON',
            details: parseErr.message,
            rawResponse: responseText.slice(0, 500),
          }),
          { status: 500, headers: { 'Content-Type': 'application/json' } }
        )
      }

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
          success: false,
          error: 'Enriched image generation timed out (600s)',
        }),
        { status: 504, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return new Response(
      JSON.stringify({
        success: false,
        error: 'Network error connecting to backend',
        details: err.message,
        hint: 'Make sure Python backend is running: python server.py',
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
