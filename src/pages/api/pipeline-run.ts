/**
 * POST /api/pipeline-run — Full End-to-End Post-Generation Pipeline
 *
 * Runs the complete pipeline in a single API call:
 *   removebg → layer-separate → edge-refine → outline → export
 *
 * Proxies to Python FastAPI backend: /api/pipeline-run
 *
 * Pipeline Position: Entry point for Steps 4-8
 *   Step 3: Gemini frame generation (green BG)
 * → Step 4-8: THIS ENDPOINT (full pipeline)
 *
 * Request body:
 *   {
 *     frames_b64: string[]          — Base64-encoded frames (green BG)
 *     // Background removal options
 *     method?: string               — "removebgio" | "rembg" | "chroma" | null (auto)
 *     api_key?: string              — remove-bg.io API key (optional)
 *     tolerance?: number            — Chroma-key tolerance (10-150, default: 60)
 *     edge_blur?: number            — Edge feathering (0-10, default: 1.0)
 *     despill?: boolean             — Green-spill correction (default: true)
 *     // Layer separation options
 *     connectivity?: 4 | 8          — Pixel connectivity (default: 4)
 *     alpha_threshold?: number      — Alpha cutoff (default: 30)
 *     min_component_area?: number   — Min pixels per component (default: 100)
 *     max_components?: number       — Max components (default: 50)
 *     // Edge refinement options
 *     anti_alias?: boolean          — Anti-aliasing (default: true)
 *     edge_smoothing?: number       — Smoothing radius (default: 1.0)
 *     // Outline options
 *     stroke_width?: number         — Outline width (default: 2.0)
 *     stroke_color?: string         — Outline color (default: "#000000")
 *     // Export options
 *     export_format?: string        — "individual" | "zip" | "svg"
 *     // Pipeline control
 *     skip_steps?: string           — Comma-separated steps to skip
 *   }
 *
 * Response:
 *   {
 *     success: boolean,
 *     stages: Array<{ stage, success, processing_time_ms, ... }>,
 *     total_time_ms: number,
 *     frames_input: number,
 *     layers_output: number,
 *     export_data: { format, layers?, zip_b64?, svg_document? },
 *     error?: string,
 *   }
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

/** Pipeline can take a while — 3 minute timeout */
const BACKEND_TIMEOUT_MS = 180_000

/** Max frames per pipeline run */
const MAX_FRAMES = 64

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()

    // ── Validation ───────────────────────────────────────────────────
    if (!body.frames_b64 || !Array.isArray(body.frames_b64) || body.frames_b64.length === 0) {
      return jsonResponse(
        { success: false, error: 'frames_b64 array is required and must be non-empty' },
        400,
      )
    }

    if (body.frames_b64.length > MAX_FRAMES) {
      return jsonResponse(
        { success: false, error: `Maximum ${MAX_FRAMES} frames per pipeline run` },
        400,
      )
    }

    // ── Proxy to Python backend ──────────────────────────────────────
    // Backend (server_animation_routes.py) expects frames_b64 directly
    const backendBody = { ...body }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS)

    let backendRes: Response
    try {
      backendRes = await fetch(`${BACKEND_URL}/api/pipeline-run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendBody),
        signal: controller.signal,
      })
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return jsonResponse(
          { success: false, error: 'Pipeline processing timed out (3 min limit)' },
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
    console.error('[pipeline-run] Unhandled error:', err)
    return jsonResponse(
      { success: false, error: `Server error: ${err.message || 'Unknown'}` },
      500,
    )
  }
}

function jsonResponse(body: Record<string, unknown>, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}