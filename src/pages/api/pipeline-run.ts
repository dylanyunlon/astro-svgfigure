/**
 * POST /api/pipeline-run — Execute Python pipeline server-side → return all cell params
 *
 * 集成策略 (xiaodi #49):
 *   在 Node.js 子进程中执行 channels/loop_orchestrator.run_loop()
 *   收集所有 channels/cell/*\/params.json → 组装 CellDescriptor[]
 *   注入 topology (TOPOLOGY_MAP) 并按 bbox.y 排序后返回
 *
 * 请求体 (可选):
 *   { max_epochs?: number }   默认 5
 *
 * 响应:
 *   200 { cells: CellDescriptor[], svg_bytes: number, epochs_run: number }
 *   500 { error: string, details: string }
 *
 * GitHub 背书: withastro/astro (API Routes), nodejs/node (child_process)
 */
import type { APIRoute } from 'astro'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'

export const prerender = false

const execFileAsync = promisify(execFile)

// Default Transformer topology (mirrors cells.ts)
const TOPOLOGY_MAP: Record<string, string[]> = {
  input_embed:  ['pos_encode'],
  pos_encode:   ['self_attn'],
  self_attn:    ['add_norm1'],
  add_norm1:    ['ffn'],
  ffn:          ['add_norm2'],
  add_norm2:    ['output'],
  output:       [],
}

/** Read channels/cell/{cellId}/params.json → CellDescriptor[] with topology injected */
function readCellParams(repoRoot: string): unknown[] {
  const channelsDir = join(repoRoot, 'channels', 'cell')
  if (!existsSync(channelsDir)) return []

  const cells: unknown[] = []
  for (const cellId of readdirSync(channelsDir)) {
    const paramsPath = join(channelsDir, cellId, 'params.json')
    if (!existsSync(paramsPath)) continue
    try {
      const raw = JSON.parse(readFileSync(paramsPath, 'utf-8'))
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
  cells.sort((a: any, b: any) => (a.bbox?.y ?? 0) - (b.bbox?.y ?? 0))
  return cells
}

export const POST: APIRoute = async ({ request }) => {
  let maxEpochs = 5
  try {
    const body = await request.json()
    if (typeof body?.max_epochs === 'number' && body.max_epochs > 0) {
      maxEpochs = Math.min(body.max_epochs, 20) // cap at 20 for safety
    }
  } catch { /* empty body is fine */ }

  const repoRoot = process.cwd()

  // Python one-liner: run pipeline and print svg byte count to stdout
  const pyScript = [
    'import sys',
    `sys.path.insert(0, '${repoRoot}')`,
    `sys.path.insert(0, '${join(repoRoot, 'channels')}')`,
    'from channels.loop_orchestrator import run_loop',
    `svg = run_loop(max_epochs=${maxEpochs})`,
    'print(len(svg))',
  ].join('; ')

  try {
    const { stdout, stderr } = await execFileAsync(
      'python3',
      ['-c', pyScript],
      {
        timeout: 120_000, // 2 min max
        env: {
          ...process.env,
          PYTHONPATH: `${repoRoot}:${join(repoRoot, 'channels')}`,
        },
        cwd: repoRoot,
      }
    )

    const svgBytes = parseInt(stdout.trim().split('\n').pop() ?? '0', 10)
    const cells = readCellParams(repoRoot)

    return new Response(
      JSON.stringify({
        cells,
        svg_bytes: isNaN(svgBytes) ? 0 : svgBytes,
        epochs_run: maxEpochs,
        source: 'python-pipeline',
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
          'X-Pipeline-Source': 'python-loop-orchestrator',
        },
      }
    )
  } catch (err: any) {
    const isTimeout = err.killed || err.code === 'ETIMEDOUT'
    return new Response(
      JSON.stringify({
        error: isTimeout ? 'Pipeline timed out (120s)' : 'Pipeline execution failed',
        details: err.stderr ?? err.message ?? String(err),
      }),
      {
        status: isTimeout ? 504 : 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
