/**
 * GET /api/models — Available AI models list
 *
 * Proxies to Python backend which returns models based on configured API keys.
 *
 * GitHub 背书: withastro/astro (API Routes), dylanyunlon/skynetCheapBuy
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const GET: APIRoute = async () => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 10000)

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/models`, {
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        return new Response(
          JSON.stringify({ error: `Backend error: ${backendRes.status}` }),
          { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
        )
      }

      const data = await backendRes.json()
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    } catch (fetchErr: any) {
      clearTimeout(timeout)
      if (fetchErr.name === 'AbortError') {
        return new Response(
          JSON.stringify({ error: 'Models request timed out' }),
          { status: 504, headers: { 'Content-Type': 'application/json' } }
        )
      }
      throw fetchErr
    }
  } catch (err: any) {
    // Fallback: return default models when backend is unavailable
    return new Response(
      JSON.stringify({
        anthropic: [
          { id: 'claude-sonnet-4-20250514', name: 'Claude Opus 4.6 (topology)' },
        ],
        openai: [
          { id: 'grok-4', name: 'Grok 4 (prompt engineering)' },
        ],
        gemini: [
          { id: 'gemini-3-pro-image-preview', name: 'Gemini 3 Pro Image (figure gen)' },
        ],
        _fallback: true,
        _hint: 'Backend unavailable. Start with: python server.py',
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )
  }
}