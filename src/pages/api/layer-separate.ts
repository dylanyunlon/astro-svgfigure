/**
 * POST /api/layer-separate — Layer Separation API
 *
 * Separates transparent-background frames into individual component layers.
 * Proxies to Python FastAPI backend: /api/layer-separate
 *
 * Pipeline Position: Step 5 (post background-removal)
 *   Step 4: Green-screen removal → transparent PNG
 * → Step 5: THIS ENDPOINT (layer separation → individual components)
 *   Step 6: Edge refinement / outlining per component
 *
 * Request body:
 *   {
 *     frames_b64: string[]          — Base64-encoded RGBA PNGs (from rembg step)
 *     connectivity?: 4 | 8          — Pixel connectivity mode (default: 4)
 *     alpha_threshold?: number      — Alpha cutoff for "opaque" (default: 30)
 *     min_component_area?: number   — Minimum pixels per component (default: 100)
 *     max_components?: number       — Max components to extract (default: 50)
 *     merge_distance?: number       — Merge nearby components within N px (default: 0)
 *     padding?: number              — Transparent padding around each layer (default: 10)
 *     maintain_position?: boolean   — Keep at original coords (default: true)
 *     sort_by?: string              — "area" | "x" | "y" | "top-left" (default: "area")
 *   }
 *
 * Response:
 *   {
 *     success: boolean,
 *     frame_results: Array<{
 *       frame_index: number,
 *       success: boolean,
 *       num_layers: number,
 *       layers: Array<{
 *         layer_id: number,
 *         image_b64: string,
 *         bbox: [x, y, w, h],
 *         area: number,
 *         centroid: [x, y],
 *       }>,
 *       error?: string,
 *     }>,
 *     stats: {
 *       total_frames: number,
 *       successful_frames: number,
 *       total_layers_extracted: number,
 *       avg_layers_per_frame: number,
 *       layer_count_consistent: boolean,
 *       total_processing_time_ms: number,
 *     },
 *     error?: string,
 *   }
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

/** Maximum frames per request to prevent abuse. */
const MAX_FRAMES = 64

/** Request timeout for the backend call (ms). */
const BACKEND_TIMEOUT_MS = 120_000

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
        { success: false, error: `Maximum ${MAX_FRAMES} frames per request` },
        400,
      )
    }

    // Validate optional numeric params
    const connectivity = body.connectivity ?? 4
    if (![4, 8].includes(connectivity)) {
      return jsonResponse(
        { success: false, error: 'connectivity must be 4 or 8' },
        400,
      )
    }

    const alphaThreshold = clampInt(body.alpha_threshold ?? 30, 0, 255)
    const minArea = clampInt(body.min_component_area ?? 100, 1, 100_000)
    const maxComponents = clampInt(body.max_components ?? 50, 1, 500)
    const mergeDistance = clampInt(body.merge_distance ?? 0, 0, 200)
    const padding = clampInt(body.padding ?? 10, 0, 100)
    const maintainPosition = body.maintain_position !== false
    const sortBy = ['area', 'x', 'y', 'top-left'].includes(body.sort_by)
      ? body.sort_by
      : 'area'

    // ── Proxy to Python backend ──────────────────────────────────────
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS)

    let backendRes: Response
    try {
      backendRes = await fetch(`${BACKEND_URL}/api/layer-separate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames_b64: body.frames_b64,
          connectivity,
          alpha_threshold: alphaThreshold,
          min_component_area: minArea,
          max_components: maxComponents,
          merge_distance: mergeDistance,
          padding,
          maintain_position: maintainPosition,
          sort_by: sortBy,
        }),
        signal: controller.signal,
      })
    } catch (fetchErr: any) {
      if (fetchErr.name === 'AbortError') {
        return jsonResponse(
          { success: false, error: 'Backend timeout — try fewer frames or lower max_components' },
          504,
        )
      }
      return jsonResponse(
        { success: false, error: `Backend connection failed: ${fetchErr.message}` },
        502,
      )
    } finally {
      clearTimeout(timeout)
    }

    if (!backendRes.ok) {
      const errorText = await backendRes.text().catch(() => 'Unknown error')
      return jsonResponse(
        {
          success: false,
          error: `Backend error: ${backendRes.status}`,
          details: errorText,
        },
        backendRes.status,
      )
    }

    const data = await backendRes.json()
    return jsonResponse(data, 200)
  } catch (err: any) {
    console.error('[layer-separate] Unhandled error:', err)
    return jsonResponse(
      { success: false, error: `Server error: ${err.message || 'Unknown'}` },
      500,
    )
  }
}

// ── Helpers ─────────────────────────────────────────────────────────

function jsonResponse(body: Record<string, unknown>, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function clampInt(value: unknown, min: number, max: number): number {
  const n = typeof value === 'number' ? Math.round(value) : parseInt(String(value), 10)
  if (Number.isNaN(n)) return min
  return Math.max(min, Math.min(max, n))
}
