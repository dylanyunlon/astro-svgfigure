/**
 * POST /api/world/qos — Switch world-level Quality-of-Service
 *
 * Body (JSON):
 *   { "level": "low" | "mid" | "high" }
 *
 * This is the world-scoped QoS endpoint that controls the AT render pipeline
 * pass configuration (geometry, particles, water, volumetric light, bloom, LUT)
 * and render scale.  It differs from /api/physics/qos which targets the SPH
 * physics simulation parameters (particle count, substeps, smoothing radius).
 *
 * Flow:
 *   1. Validate the requested level
 *   2. Forward to Python backend (BACKEND_URL/api/world/qos)
 *   3. On success → return the backend's response (active profile)
 *   4. Backend offline → return the requested profile with `offline: true`
 *      so the browser can apply it client-side without blocking on a 502
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
  low: {
    renderScale: 0.5,
    lutStrength: 0.4,
    passes: {
      geometry: true, flowerParticle: true, splineParticle: false,
      waterSurface: false, volumetricLight: false, bloom: false, lut: true,
    },
  },
  mid: {
    renderScale: 0.75,
    lutStrength: 0.7,
    passes: {
      geometry: true, flowerParticle: true, splineParticle: true,
      waterSurface: true, volumetricLight: false, bloom: true, lut: true,
    },
  },
  high: {
    renderScale: 1.0,
    lutStrength: 0.85,
    passes: {
      geometry: true, flowerParticle: true, splineParticle: true,
      waterSurface: true, volumetricLight: true, bloom: true, lut: true,
    },
  },
}

export const POST: APIRoute = async ({ request }) => {
  // ── Parse & validate body ──────────────────────────────────────────────────
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

  // ── Forward to Python backend ──────────────────────────────────────────────
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/world/qos`, {
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

    // ── Offline fallback: acknowledge client-side QoS switch ─────────────
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
