/**
 * GET /api/cells --- Cell descriptor list
 *
 * ---- (------):
 *   1. ---------- Python FastAPI ---- (BACKEND_URL/api/cells)
 //   2. ---------------: -- channels/cell/{id}/params.json --- ---- CellDescriptor[]
//      ---- (topology.outgoing_edges) ------ Transformer ----
 //
// GitHub ----: withastro/astro (API Routes), ResearAI/AutoFigure, xiaodi #17
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

interface CompositeCell {
  cell_id: string;
  label?: string;
  species?: string;
  bbox?: { x: number; y: number; w: number; h: number; z?: number };
  z?: number;
  [key: string]: unknown;
}

interface CompositeEdge {
  edge_id?: string;
  id?: string;
  sources?: string[];
  source?: string;
  targets?: string[];
  target?: string;
  is_skip?: boolean;
  type?: string;
  [key: string]: unknown;
}

interface CompositeParams {
  cells?: CompositeCell[];
  edges?: CompositeEdge[];
  [key: string]: unknown;
}

/**
 * Read channels/composite_params.json (output of assemble_final_svg()) and
 * build CellDescriptor[] + EdgeDescriptor[] for the frontend.
 * Returns null if the file does not exist or has no cells[].
 */
function readFromCompositeParams(): { cells: unknown[]; edges: unknown[] } | null {
  const compositePath = join(process.cwd(), 'channels', 'composite_params.json')
  if (!existsSync(compositePath)) return null

  let composite: CompositeParams
  try {
    composite = JSON.parse(readFileSync(compositePath, 'utf-8'))
  } catch {
    return null
  }

  // composite.cells may be an array or a dict {cell_id: {...}}.
  // Normalize to array.
  let cellsArr: CompositeCell[]
  if (Array.isArray(composite.cells)) {
    cellsArr = composite.cells
  } else if (composite.cells && typeof composite.cells === 'object') {
    cellsArr = Object.entries(composite.cells as Record<string, any>).map(
      ([id, v]) => ({ cell_id: id, ...v } as CompositeCell)
    )
  } else {
    return null
  }
  if (cellsArr.length === 0) return null

  // Stash back as array for downstream code
  composite.cells = cellsArr as any

  // Build a lookup: cell_id --- list of incoming source IDs (derived from edges)
  const incomingMap: Record<string, string[]> = {}
  const outgoingMap: Record<string, string[]> = {}
  for (const cell of cellsArr) {
    incomingMap[cell.cell_id] = []
    outgoingMap[cell.cell_id] = []
  }
  for (const edge of (composite.edges ?? [])) {
    const srcs = edge.sources ?? (edge.source ? [edge.source] : [])
    const tgts = edge.targets ?? (edge.target ? [edge.target] : [])
    for (const src of srcs) {
      for (const tgt of tgts) {
        if (!outgoingMap[src]) outgoingMap[src] = []
        if (!incomingMap[tgt]) incomingMap[tgt] = []
        outgoingMap[src].push(tgt)
        incomingMap[tgt].push(src)
      }
    }
  }

  // Map composite cells --- CellDescriptor
  // bbox may live at c.bbox OR c.agent_params.bbox
  // species may live at c.species OR derived from c.agent_params.species_params
  const cells: unknown[] = composite.cells.map((c) => {
    const ap = (c as any).agent_params ?? {}
    const rawBbox = c.bbox ?? ap.bbox
    const sp = c.species_params ?? ap.species_params ?? {}
    return {
      cell_id: c.cell_id,
      label:   c.label ?? (c as any).name ?? c.cell_id,
      species: c.species ?? (sp as any).species ?? 'cil-eye',
      bbox: rawBbox
        ? { x: rawBbox.x, y: rawBbox.y, w: rawBbox.w, h: rawBbox.h }
        : { x: 0, y: 0, w: 120, h: 40 },
      z: c.z ?? rawBbox?.z ?? 1,
      topology: {
        incoming_edges: incomingMap[c.cell_id] ?? [],
        outgoing_edges: outgoingMap[c.cell_id] ?? [],
      },
      // Pass through extra fields
      ...(c.fill_color   !== undefined && { fill_color:   c.fill_color }),
      ...(c.stroke_color !== undefined && { stroke_color: c.stroke_color }),
      ...(c.opacity      !== undefined ? { opacity: c.opacity } : ap.opacity !== undefined ? { opacity: ap.opacity } : {}),
      ...(c.shadow       !== undefined && { shadow:       c.shadow }),
      species_params: sp,
      ...(c.epoch        !== undefined && { epoch:        c.epoch }),
      ...(c.render_order !== undefined && { render_order: c.render_order }),
      ...(c.z_layer      !== undefined && { z_layer:      c.z_layer }),
      ...(c.is_translucent !== undefined && { is_translucent: c.is_translucent }),
    }
  })

  // Sort by bbox.y (top---bottom visual order)
  cells.sort((a: any, b: any) => (a.bbox?.y ?? 0) - (b.bbox?.y ?? 0))

  // Map composite edges --- EdgeDescriptor
  const edges: unknown[] = (composite.edges ?? []).map((e) => {
    const edgeId  = e.edge_id ?? e.id ?? 'edge'
    const srcs    = e.sources ?? (e.source ? [e.source] : [])
    const tgts    = e.targets ?? (e.target ? [e.target] : [])
    const isSkip  = e.is_skip ?? e.type === 'skip_connection'
    return {
      id:     edgeId,
      source: srcs[0] ?? '',
      target: tgts[0] ?? '',
      type:   isSkip ? 'skip_connection' : 'normal',
    }
  })

  return { cells, edges }
}

/** Read channels/cell/{id}/params.json and build CellDescriptor[] */
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
  // Sort by bbox.y (top---bottom visual order)
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

    // ------ Fallback priority 1: channels/composite_params.json (assemble_final_svg output) ------
    const composite = readFromCompositeParams()
    if (composite) {
      return new Response(JSON.stringify({ cells: composite.cells, edges: composite.edges }), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
          'X-Source': 'composite-params-fallback',
        },
      })
    }

    // ------ Fallback priority 2: channels/cell/{id}/params.json (legacy per-cell files) ------
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
