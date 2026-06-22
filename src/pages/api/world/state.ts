/**
 * GET /api/world/state — World state snapshot
 *
 * Returns the current world state: cells, edges, QoS level, render mode,
 * epoch counter, and SSE connectivity status.
 *
 * Strategy (dual mode):
 *   1. Proxy to Python FastAPI backend (BACKEND_URL/api/world/state)
 *   2. Backend offline → assemble a fallback snapshot from local data:
 *      - cells + edges from channels/composite_params.json
 *      - QoS defaults to "high", render mode to "simple"
 *
 * The /world page can poll this endpoint on initial load to hydrate its HUD
 * and verify backend connectivity before opening the SSE stream.
 */

import type { APIRoute } from 'astro'
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL ||
  import.meta.env.BACKEND_URL ||
  'http://127.0.0.1:8000'

const TIMEOUT_MS = 8_000

// ── QoS profiles (mirrors world/index.astro & physics/qos.ts) ──────────────
const QOS_PROFILES: Record<string, object> = {
  low:  { renderScale: 0.5,  lutStrength: 0.4,  passes: { geometry: true, flowerParticle: true, splineParticle: false, waterSurface: false, volumetricLight: false, bloom: false, lut: true } },
  mid:  { renderScale: 0.75, lutStrength: 0.7,  passes: { geometry: true, flowerParticle: true, splineParticle: true,  waterSurface: true,  volumetricLight: false, bloom: true,  lut: true } },
  high: { renderScale: 1.0,  lutStrength: 0.85, passes: { geometry: true, flowerParticle: true, splineParticle: true,  waterSurface: true,  volumetricLight: true,  bloom: true,  lut: true } },
}

// ── Filesystem fallback: read composite_params.json ─────────────────────────

interface CompositeCell {
  cell_id: string
  label?: string
  species?: string
  bbox?: { x: number; y: number; w: number; h: number }
  [key: string]: unknown
}

interface CompositeEdge {
  edge_id?: string
  id?: string
  source?: string
  sources?: string[]
  target?: string
  targets?: string[]
  [key: string]: unknown
}

function readCompositeSnapshot(): { cells: CompositeCell[]; edges: CompositeEdge[] } | null {
  const compositeFile = join(process.cwd(), 'channels', 'composite_params.json')
  if (!existsSync(compositeFile)) return null

  try {
    const raw = JSON.parse(readFileSync(compositeFile, 'utf-8'))
    const cells: CompositeCell[] = raw.cells ?? []
    const edges: CompositeEdge[] = raw.edges ?? []
    if (cells.length === 0) return null
    return { cells, edges }
  } catch {
    return null
  }
}

export const GET: APIRoute = async () => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/world/state`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
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
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    })
  } catch (err: any) {
    clearTimeout(timeout)

    // ── Offline fallback: assemble snapshot from local data ────────────────
    const composite = readCompositeSnapshot()
    const cellCount = composite?.cells.length ?? 0
    const edgeCount = composite?.edges.length ?? 0

    const snapshot = {
      ok: true,
      offline: true,
      timestamp: new Date().toISOString(),
      qos: {
        level: 'high',
        profile: QOS_PROFILES['high'],
      },
      renderMode: 'simple',
      cells: composite?.cells ?? [],
      edges: composite?.edges ?? [],
      counts: {
        cells: cellCount,
        edges: edgeCount,
      },
      sse: {
        connected: false,
        hint: 'SSE stream available at /api/cell-events when backend is online.',
      },
      hint: 'Python backend unreachable — world state assembled from local data.',
      error: err?.name === 'AbortError'
        ? `Backend timed out after ${TIMEOUT_MS}ms`
        : err?.message ?? String(err),
    }

    return new Response(JSON.stringify(snapshot), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
        'X-Backend-Status': 'offline',
        'X-Source': composite ? 'composite-params-fallback' : 'defaults-only',
      },
    })
  }
}
