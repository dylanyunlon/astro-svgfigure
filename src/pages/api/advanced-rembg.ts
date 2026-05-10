/**
 * POST /api/advanced-rembg — Advanced Background Removal API
 *
 * Orchestrates multiple background removal strategies and selects
 * the best result per-frame using quality scoring.
 *
 * Strategies:
 *   (A) HSV chroma-key  — fast, deterministic, green-screen specific
 *   (B) rembg / U2-Net  — ML-based, general purpose
 *   (C) Hybrid          — chroma-key + rembg cascade for best quality
 *
 * Proxies to Python FastAPI backend: /api/advanced-rembg
 *
 * Pipeline Position: Step 4 (alternative to basic rembg-frames)
 *   Step 3: Frame generation (Gemini) → green-screen frames
 * → Step 4: THIS ENDPOINT (intelligent background removal)
 *   Step 5: Layer separation
 *
 * Request body:
 *   {
 *     frames_b64: string[]              — Base64-encoded frames with green background
 *     strategy?: "auto"|"chroma"|"rembg"|"hybrid"  — Removal strategy (default: "auto")
 *
 *     // Chroma-key options
 *     hue_center?: number               — Green hue center in degrees (default: auto-detect)
 *     hue_range?: number                — Hue tolerance (default: 35)
 *     saturation_min?: number           — Min saturation for green (default: 40)
 *     value_min?: number                — Min brightness for green (default: 40)
 *     feather_radius?: number           — Edge feathering in pixels (default: 3)
 *     spill_correction?: "none"|"average"|"max_rb"  — Spill removal (default: "average")
 *
 *     // rembg options
 *     rembg_model?: string              — Model name (default: "u2net")
 *     alpha_matting?: boolean           — Use alpha matting (default: false)
 *
 *     // Quality options
 *     quality_threshold?: number        — Min quality to accept (0-1, default: 0.7)
 *     validate?: boolean                — Run transparency validation (default: true)
 *   }
 *
 * Response:
 *   {
 *     success: boolean,
 *     frame_results: Array<{
 *       frame_index: number,
 *       success: boolean,
 *       image_b64: string,           — Transparent PNG result
 *       method_used: string,         — Which strategy won
 *       quality_score: number,       — 0.0 to 1.0
 *       methods_tried: string[],
 *       processing_time_ms: number,
 *       error?: string,
 *     }>,
 *     stats: {
 *       total_frames: number,
 *       successful: number,
 *       avg_quality: number,
 *       method_distribution: Record<string, number>,
 *       total_time_ms: number,
 *     },
 *     validation?: object,           — Transparency validation report (if validate=true)
 *     error?: string,
 *   }
 */
import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

const MAX_FRAMES = 64
const BACKEND_TIMEOUT_MS = 180_000  // 3 minutes — ML models can be slow

const VALID_STRATEGIES = ['auto', 'chroma', 'rembg', 'hybrid'] as const
const VALID_SPILL_MODES = ['none', 'average', 'max_rb'] as const
const VALID_REMBG_MODELS = [
  'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg',
  'silueta', 'isnet-general-use', 'isnet-anime',
  'birefnet-general', 'birefnet-massive',
] as const

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

    // Strategy
    const strategy = VALID_STRATEGIES.includes(body.strategy)
      ? body.strategy
      : 'auto'

    // Chroma-key params
    const hueCenter = body.hue_center != null
      ? clampFloat(body.hue_center, 0, 360)
      : null  // null = auto-detect
    const hueRange = clampFloat(body.hue_range ?? 35, 5, 90)
    const satMin = clampInt(body.saturation_min ?? 40, 0, 255)
    const valMin = clampInt(body.value_min ?? 40, 0, 255)
    const featherRadius = clampInt(body.feather_radius ?? 3, 0, 20)
    const spillCorrection = VALID_SPILL_MODES.includes(body.spill_correction)
      ? body.spill_correction
      : 'average'

    // rembg params
    const rembgModel = VALID_REMBG_MODELS.includes(body.rembg_model)
      ? body.rembg_model
      : 'u2net'
    const alphaMatting = body.alpha_matting === true

    // Quality params
    const qualityThreshold = clampFloat(body.quality_threshold ?? 0.7, 0, 1)
    const validate = body.validate !== false

    // ── Proxy to Python backend ──────────────────────────────────────
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS)

    let backendRes: Response
    try {
      backendRes = await fetch(`${BACKEND_URL}/api/advanced-rembg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames_b64: body.frames_b64,
          strategy,
          hue_center: hueCenter,
          hue_range: hueRange,
          saturation_min: satMin,
          value_min: valMin,
          feather_radius: featherRadius,
          spill_correction: spillCorrection,
          rembg_model: rembgModel,
          alpha_matting: alphaMatting,
          quality_threshold: qualityThreshold,
          validate,
        }),
        signal: controller.signal,
      })
    } catch (fetchErr: any) {
      if (fetchErr.name === 'AbortError') {
        return jsonResponse(
          {
            success: false,
            error: 'Backend timeout — ML-based removal can take time. Try "chroma" strategy for faster results.',
          },
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

      // Special handling for missing rembg dependency
      if (backendRes.status === 500 && errorText.includes('rembg')) {
        return jsonResponse(
          {
            success: false,
            error: 'rembg is not installed on the backend. Use strategy="chroma" for green-screen removal, or install rembg: pip install rembg[gpu]',
            details: errorText,
          },
          500,
        )
      }

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
    console.error('[advanced-rembg] Unhandled error:', err)
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

function clampFloat(value: unknown, min: number, max: number): number {
  const n = typeof value === 'number' ? value : parseFloat(String(value))
  if (Number.isNaN(n)) return min
  return Math.max(min, Math.min(max, n))
}
