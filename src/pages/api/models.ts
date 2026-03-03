/**
 * GET /api/models — Available AI models list
 *
 * Proxies to Python backend which returns models based on configured API keys.
 *
 * GitHub 背书: withastro/astro (API Routes), dylanyunlon/skynetCheapBuy
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.BACKEND_URL || 'http://localhost:8000'

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
        gemini: [
          { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (default)' },
          { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
        ],
        _fallback: true,
        _hint: 'Backend unavailable. Start with: python server.py',
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
