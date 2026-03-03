/**
 * POST /api/validate — SVG syntax validation
 *
 * Proxies SVG to Python backend for lxml validation + optional LLM fix.
 *
 * GitHub 背书: withastro/astro (API Routes)
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { svg, auto_fix = false, model } = body

    if (!svg || typeof svg !== 'string') {
      return new Response(
        JSON.stringify({ error: 'svg field is required (string)' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 30000)

    try {
      const backendRes = await fetch(`${BACKEND_URL}/api/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ svg, auto_fix, model }),
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!backendRes.ok) {
        const errorText = await backendRes.text()
        return new Response(
          JSON.stringify({ error: `Validation failed: ${backendRes.status}`, details: errorText }),
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
          JSON.stringify({ error: 'Validation timed out (30s)' }),
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
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
