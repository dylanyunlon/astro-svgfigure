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
    const timeout = setTimeout(() => controller.abort(), 600000) // 600s timeout

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
          elk_graph: body.elk_graph || null,
          // Per-node sprite images for multi-image Gemini input (max 13 + 1 skeleton = 14)
          sprite_images: body.sprite_images || null,
          // media_resolution hints: HIGH for skeleton, LOW for sprites
          skeleton_media_resolution: body.skeleton_media_resolution || null,
          sprite_media_resolution: body.sprite_media_resolution || null,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      // Read response body as text first to handle empty/malformed responses
      const responseText = await backendRes.text()

      if (!backendRes.ok) {
        return new Response(
          JSON.stringify({
            success: false,
            error: `Backend error: ${backendRes.status}`,
            details: responseText,
          }),
          { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
        )
      }

      // Validate response body is non-empty before parsing
      if (!responseText || responseText.trim() === '') {
        return new Response(
          JSON.stringify({
            success: false,
            error: 'Backend returned empty response',
            hint: 'The AI model may have returned no content. Check backend logs.',
          }),
          { status: 500, headers: { 'Content-Type': 'application/json' } }
        )
      }

      // Parse JSON with explicit error handling
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
          error: 'Image generation timed out (600s)',
          hint: 'Try a simpler description or a faster model',
        }),
        { status: 504, headers: { 'Content-Type': 'application/json' } }
      )
    }
    // Network-level errors (connection refused, DNS failure, etc.)
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