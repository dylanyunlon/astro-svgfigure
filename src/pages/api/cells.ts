/**
 * GET /api/cells — Cell descriptor list
 *
 * 策略 (双模式):
 *   1. 尝试代理到 Python FastAPI 后端 (BACKEND_URL/api/cells)
 *   2. 后端离线时回退: 读取 channels/cell/*/params.json → 组装 CellDescriptor[]
 *      拓扑 (topology.outgoing_edges) 由固定 Transformer 顺序推断
 *
 * GitHub 背书: withastro/astro (API Routes), ResearAI/AutoFigure, xiaodi #17
 */
import type { APIRoute } from 'astro'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL || import.meta.env.BACKEND_URL || 'http://127.0.0.1:8000'

// Default Transformer topology: linear chain with one skip connection
const TOPOLOGY_MAP: Record<string, string[]> = {
  input_embed:  ['pos_encode'],
  pos_encode:   ['self_attn'],
  self_attn:    ['add_norm1'],
  add_norm1:    ['ffn'],
  ffn:          ['add_norm2'],
  add_norm2:    ['output'],
  output:       [],
}

/** Read channels/cell/*/params.json and build CellDescriptor[] */
function readCellsFromFs(): unknown[] {
  const channelsDir = join(process.cwd(), 'channels', 'cell')
  if (!existsSync(channelsDir)) return []

  const cells: unknown[] = []
  for (const cellId of readdirSync(channelsDir)) {
    const paramsPath = join(channelsDir, cellId, 'params.json')
    if (!existsSync(paramsPath)) continue
    try {
      const raw = JSON.parse(readFileSync(paramsPath, 'utf-8'))
      // Inject topology if missing
      if (!raw.topology) {
        raw.topology = {
          incoming_edges: Object.entries(TOPOLOGY_MAP)
            .filter(([, outs]) => outs.includes(cellId))
            .map(([src]) => src),
          outgoing_edges: TOPOLOGY_MAP[cellId] ?? [],
        }
      }
      cells.push(raw)
    } catch { /* skip malformed */ }
  }
  // Sort by bbox.y (top→bottom visual order)
  cells.sort((a: any, b: any) => (a.bbox?.y ?? 0) - (b.bbox?.y ?? 0))
  return cells
}

export const GET: APIRoute = async () => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 8000)

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/cells`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
    })

    clearTimeout(timeout)

    if (!backendRes.ok) {
      const errorText = await backendRes.text()
      let errorDetail: string
      try {
        const errorJson = JSON.parse(errorText)
        errorDetail = errorJson.error || errorJson.detail || errorText
      } catch {
        errorDetail = errorText
      }
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendRes.status}`, details: errorDetail }),
        { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const data = await backendRes.json()
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    })
  } catch (fetchErr: any) {
    clearTimeout(timeout)

    // ── Fallback: read from filesystem ──────────────────────────────────────
    const fsCells = readCellsFromFs()
    if (fsCells.length > 0) {
      return new Response(JSON.stringify(fsCells), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
          'X-Source': 'filesystem-fallback',
        },
      })
    }

    // Backend offline and no FS data
    const isTimeout = fetchErr.name === 'AbortError'
    return new Response(
      JSON.stringify({
        error: isTimeout ? 'Request timed out (8s)' : 'Failed to connect to backend',
        details: fetchErr.message,
        hint: 'Start backend: python server.py',
        debug: { backend_url: BACKEND_URL, target: `${BACKEND_URL}/api/cells` },
      }),
      { status: isTimeout ? 504 : 502, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
