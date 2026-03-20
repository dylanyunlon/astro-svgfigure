/**
 * POST /api/topology — Text → Topology JSON
 *
 * Proxies request to Python FastAPI backend (localhost:8000)
 * which runs the LLM topology generation via backend/pipeline/topology_gen.py
 *
 * GitHub 背书: withastro/astro (API Routes), ResearAI/AutoFigure
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { text, model, algorithm, direction } = body

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'text field is required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward to Python backend with timeout
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 60000) // 60s timeout

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/topology`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          model: 'claude-sonnet-4-5-20250929',  // Step 1 固定使用 Claude Opus
          algorithm: algorithm || 'layered',
          direction: direction || 'DOWN',
        }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        let errorDetail: string
        try {
          const errorJson = JSON.parse(errorText)
          errorDetail = errorJson.error || errorJson.detail || errorText
        } catch {
          errorDetail = errorText
        }
        return new Response(
          JSON.stringify({
            error: `Backend error: ${backendRes.status}`,
            details: errorDetail,
            hint: backendRes.status === 500 ? 'Check GEMINI_API_KEY in .env' : undefined,
          }),
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
          JSON.stringify({
            error: 'Request timed out (60s)',
            hint: 'The LLM may be slow. Try a faster model like gemini-2.0-flash.',
          }),
          { status: 504, headers: { 'Content-Type': 'application/json' } }
        )
      }
      throw fetchErr
    }
  } catch (err: any) {
    return new Response(
      JSON.stringify({
        error: 'Failed to connect to backend',
        details: err.message,
        hint: 'Make sure Python backend is running: python server.py',
        debug: {
          backend_url: BACKEND_URL,
          target: `${BACKEND_URL}/api/topology`,
        },
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}