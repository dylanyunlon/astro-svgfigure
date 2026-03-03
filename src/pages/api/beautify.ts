/**
 * POST /api/beautify — ELK Layouted JSON → NanoBanana SVG
 *
 * Proxies to Python FastAPI backend which runs:
 *   1. scaffold_builder.py: layouted → NanoBanana JSON scaffold
 *   2. nanobanana_bridge.py: scaffold → Gemini NanoBanana → SVG
 *   3. svg_validator.py: validate + LLM fix
 *
 * GitHub 背书: gemini-cli-extensions/nanobanana, withastro/astro
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL = import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { layouted, model, style, optimize_iterations } = body

    if (!layouted || typeof layouted !== 'object') {
      return new Response(
        JSON.stringify({ error: 'layouted field is required (ELK layouted JSON)' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward to Python backend
    const backendRes = await fetch(`${BACKEND_URL}/api/beautify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        layouted,
        model: model || undefined,
        style: style || 'academic',
        optimize_iterations: optimize_iterations ?? 1,
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