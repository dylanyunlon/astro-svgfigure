/**
 * POST /api/removebg — remove-bg.io Cloud Background Removal
 *
 * Proxies to Python backend which calls the remove-bg.io API.
 * Falls back to local chroma-key/rembg if API is unavailable.
 *
 * remove-bg.io: Free HD, no watermark, no quota, HMAC-signed API
 * (NOT remove.bg which is Canva's paid service)
 *
 * Request body:
 *   {
 *     frames_b64: string[]   — Base64-encoded frames
 *     api_key?: string       — remove-bg.io signing key (or use env var)
 *   }
 *
 * Response:
 *   {
 *     success: boolean,
 *     results: Array<{ success, image_b64, method_used, quality_score, ... }>,
 *     error?: string,
 *   }
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

const BACKEND_TIMEOUT_MS = 120_000
const MAX_FRAMES = 16 // remove-bg.io: 3 concurrent per token, batch in sequence

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    if (!body.frames_b64 || !Array.isArray(body.frames_b64) || body.frames_b64.length === 0) {
      return jsonResponse({ success: false, error: 'frames_b64 array is required' }, 400)
    }

    if (body.frames_b64.length > MAX_FRAMES) {
      return jsonResponse(
        { success: false, error: `Maximum ${MAX_FRAMES} frames per remove-bg.io request` },
        400,
      )
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS)

    let backendRes: Response
    try {
      backendRes = await fetch(`${BACKEND_URL}/api/removebg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames_b64: body.frames_b64,
          api_key: body.api_key || '',
        }),
        signal: controller.signal,
      })
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return jsonResponse(
          { success: false, error: 'remove-bg.io processing timed out' },
          504,
        )
      }
      return jsonResponse(
        { success: false, error: `Backend unavailable: ${err.message}` },
        502,
      )
    } finally {
      clearTimeout(timeout)
    }

    if (!backendRes.ok) {
      const errorText = await backendRes.text().catch(() => 'Unknown error')
      return jsonResponse(
        { success: false, error: `Backend error: ${backendRes.status}`, details: errorText },
        backendRes.status,
      )
    }

    const data = await backendRes.json()
    return jsonResponse(data, 200)
  } catch (err: any) {
    return jsonResponse(
      { success: false, error: `Server error: ${err.message || 'Unknown'}` },
      500,
    )
  }
}

// GET for availability check (remove-bg.io is free, no credit system)
export const GET: APIRoute = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/removebg/status`, {
      timeout: 10_000,
    } as any)
    if (!res.ok) {
      return jsonResponse({ error: 'Could not check remove-bg.io status' }, res.status)
    }
    return jsonResponse(await res.json(), 200)
  } catch {
    return jsonResponse({ error: 'Backend unavailable' }, 502)
  }
}

function jsonResponse(body: Record<string, unknown>, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
