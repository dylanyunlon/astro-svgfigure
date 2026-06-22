/**
 * POST /api/physics/qos — Set physics simulation Quality-of-Service level
 *
 * Body (JSON):
 *   { "level": "low" | "mid" | "high" }
 *
 * Proxies the request to the Python FastAPI backend.  On success the backend
 * returns the active QoS profile that was applied:
 *   { "ok": true, "level": "high", "profile": { particleCount, substeps, … } }
 *
 * Fallback behaviour:
 *   If the Python backend is offline the route echoes back the requested
 *   level with `{ "ok": true, "offline": true }` so the browser-side UI can
 *   still update its state without blocking on a 502.
 */

import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL ||
  import.meta.env.BACKEND_URL ||
  'http://127.0.0.1:8000'

const VALID_LEVELS = new Set(['low', 'mid', 'high'])
const TIMEOUT_MS = 5_000

// ── QoS profiles (mirrors world/index.astro QOS_PROFILES) ───────────────────
const QOS_PROFILES: Record<string, object> = {
  low:  { particleCount: 4_096,  substeps: 1, smoothingRadius: 0.08, renderScale: 0.5  },
  mid:  { particleCount: 16_384, substeps: 2, smoothingRadius: 0.05, renderScale: 0.75 },
  high: { particleCount: 65_536, substeps: 4, smoothingRadius: 0.03, renderScale: 1.0  },
}

export const POST: APIRoute = async ({ request }) => {
  // ── Parse & validate body ────────────────────────────────────────────────
  let level: string
  try {
    const body = await request.json() as { level?: unknown }
    level = String(body?.level ?? '').toLowerCase()
  } catch {
    return new Response(
      JSON.stringify({ error: 'Invalid JSON body', hint: '{ "level": "low" | "mid" | "high" }' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
  }

  if (!VALID_LEVELS.has(level)) {
    return new Response(
      JSON.stringify({ error: `Unknown QoS level: "${level}"`, valid: [...VALID_LEVELS] }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
  }

  // ── Forward to Python backend ─────────────────────────────────────────────
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/physics/qos`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level }),
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!backendRes.ok) {
      const text = await backendRes.text()
      let detail: string
      try {
        detail = JSON.parse(text)?.detail ?? text
      } catch {
        detail = text
      }
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendRes.status}`, details: detail }),
        { status: backendRes.status, headers: { 'Content-Type': 'application/json' } },
      )
    }

    const data = await backendRes.json()
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (err: any) {
    clearTimeout(timeout)

    // ── Offline fallback: acknowledge client-side QoS switch ──────────────
    // The browser world page can apply the profile locally even when the
    // Python physics backend is not running.
    return new Response(
      JSON.stringify({
        ok: true,
        level,
        profile: QOS_PROFILES[level],
        offline: true,
        hint: 'Python backend unreachable — QoS applied client-side only.',
        error: err?.name === 'AbortError'
          ? `Backend timed out after ${TIMEOUT_MS}ms`
          : err?.message ?? String(err),
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'X-Backend-Status': 'offline',
        },
      },
    )
  }
}
