/**
 * GET /api/health — Health check endpoint
 *
 * Checks:
 *   1. Astro SSR is running (always true if this responds)
 *   2. Python backend connectivity (fetch BACKEND_URL/api/models)
 *
 * Used by HealthCheck.astro component to show connection status.
 *
 * GitHub 背书: withastro/astro (API Routes)
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const GET: APIRoute = async () => {
  const health: Record<string, any> = {
    astro: true,
    backend: false,
    backendUrl: BACKEND_URL,
    models: null,
    timestamp: new Date().toISOString(),
  }

  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)

    const res = await fetch(`${BACKEND_URL}/api/models`, {
      signal: controller.signal,
    })

    clearTimeout(timeout)

    if (res.ok) {
      health.backend = true
      health.models = await res.json()
    } else {
      health.backendError = `HTTP ${res.status}`
    }

    // Check API key status from backend config endpoint
    try {
      const configRes = await fetch(`${BACKEND_URL}/api/config`, {
        signal: AbortSignal.timeout(3000),
      })
      if (configRes.ok) {
        health.config = await configRes.json()
      }
    } catch {
      // Config endpoint optional
    }
  } catch (err: any) {
    health.backendError =
      err.name === 'AbortError' ? 'Timeout (5s)' : err.message
  }

  // Determine pipeline readiness
  health.pipelineReady = health.backend === true

  return new Response(JSON.stringify(health), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}
