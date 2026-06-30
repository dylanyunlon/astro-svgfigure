/**
 * GET /api/cell-events — Server-Sent Events stream
 *
 * Proxies the Python FastAPI SSE stream (localhost:8000/api/cell-events)
 * to the browser.  Emits two named event types:
 *
 *   event: cell_update          — cell param changes (Apollo DataNotifier)
 *   data: { cell_id, params }
 *
 *   event: physics_step         — per-tick SPH physics snapshot
 *   data: { step, particle_count, kinetic_energy, sim_time_ms, qos }
 *
 * M1305 — additionally watches channels/cell/{id}/geometry.json directly
 * from this Node process (independent of the Python backend) and emits:
 *
 *   event: geometry_updated     — fired whenever tick-runner.py overwrites
 *   data: CellGeometry          a cell's geometry.json (see GEOMETRY_FORMAT.md)
 *
 * tick-runner.py runs out-of-process from server.py and fully overwrites
 * geometry.json every tick (no DataNotifier hookup), so the only reliable
 * way to detect changes is mtime polling — mirrors the
 * channels/channel_runtime.py `watch_channel()` poll pattern, just driven
 * from this Astro route instead of Python.  This is consumed by
 * src/lib/sph/cell-geometry-channel.ts (CellGeometryChannel._connectSSE).
 *
 * When the Python backend is offline the route falls back to a synthetic
 * heartbeat stream so the browser EventSource does not enter an error-
 * reconnect storm.  Geometry polling still runs in fallback mode.
 *
 * Reconnect:
 *   The browser EventSource reconnects automatically.  We set `retry: 2000`
 *   in the stream preamble so the browser waits 2 s between retries.
 */

import type { APIRoute } from 'astro'
import { readdirSync, statSync, readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL ||
  import.meta.env.BACKEND_URL ||
  'http://127.0.0.1:8000'

const CONNECT_TIMEOUT_MS = 5_000
const HEARTBEAT_INTERVAL_MS = 15_000
const GEOMETRY_POLL_INTERVAL_MS = 1_000

const CELL_DIR = join(process.cwd(), 'channels', 'cell')

// ── Helpers ──────────────────────────────────────────────────────────────────

function sseComment(text: string): string {
  return `: ${text}\n\n`
}

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

// ── Geometry watcher (M1305) ─────────────────────────────────────────────────
// Polls channels/cell/{id}/geometry.json every GEOMETRY_POLL_INTERVAL_MS and
// emits `geometry_updated` for any file whose mtime advanced since the last
// poll.  Started once per connected client and torn down when the client
// disconnects (mirrors the lifecycle of the backend-proxy / fallback timers
// below).

function discoverCellIds(): string[] {
  if (!existsSync(CELL_DIR)) return []
  try {
    return readdirSync(CELL_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .filter((cellId) => existsSync(join(CELL_DIR, cellId, 'geometry.json')))
  } catch {
    return []
  }
}

function startGeometryWatcher(enqueue: (chunk: Uint8Array) => void): () => void {
  const enc = new TextEncoder()
  const mtimeCache = new Map<string, number>()
  let cellIds = discoverCellIds()
  let pollCount = 0

  const poll = () => {
    // Re-scan channels/cell/ every ~30 polls in case cells were added or
    // removed (division / apoptosis) without restarting this route.
    pollCount += 1
    if (pollCount % 30 === 0) {
      cellIds = discoverCellIds()
    }

    for (const cellId of cellIds) {
      const geometryPath = join(CELL_DIR, cellId, 'geometry.json')
      let mtimeMs: number
      try {
        mtimeMs = statSync(geometryPath).mtimeMs
      } catch {
        continue // cell removed mid-poll
      }

      const prevMtime = mtimeCache.get(cellId)
      if (prevMtime !== undefined && mtimeMs <= prevMtime) continue

      try {
        const geometry = JSON.parse(readFileSync(geometryPath, 'utf-8'))
        // Payload is the raw geometry object (cell_id already lives inside
        // it), matching CellGeometryChannel's `JSON.parse(e.data) as CellGeometry`.
        enqueue(enc.encode(sseEvent('geometry_updated', geometry)))
        mtimeCache.set(cellId, mtimeMs)
      } catch {
        // Mid-write (tick-runner uses tmp + os.replace so this should be
        // rare) — leave mtimeCache unset so the next poll retries.
      }
    }
  }

  // Prime the cache so a freshly connected client doesn't replay every
  // cell's current geometry — only changes from this point on are sent.
  for (const cellId of cellIds) {
    try {
      mtimeCache.set(cellId, statSync(join(CELL_DIR, cellId, 'geometry.json')).mtimeMs)
    } catch { /* no geometry.json yet */ }
  }

  const timer = setInterval(poll, GEOMETRY_POLL_INTERVAL_MS)
  return () => clearInterval(timer)
}

// ── Fallback heartbeat stream ─────────────────────────────────────────────────
// Returned when the Python backend cannot be reached so the browser receives a
// valid text/event-stream and does not loop through rapid reconnect attempts.

function makeFallbackStream(reason: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let stopGeometryWatcher: (() => void) | null = null

  return new ReadableStream<Uint8Array>({
    start(ctrl) {
      // Preamble
      ctrl.enqueue(enc.encode(`retry: 2000\n\n`))
      ctrl.enqueue(enc.encode(sseComment(`backend-offline: ${reason}`)))
      ctrl.enqueue(
        enc.encode(
          sseEvent('physics_step', {
            step: 0,
            particle_count: 0,
            kinetic_energy: 0,
            sim_time_ms: 0,
            qos: 'unknown',
            offline: true,
            reason,
          }),
        ),
      )

      // M1305: geometry.json polling runs independently of backend
      // availability — tick-runner.py writes straight to disk, so the
      // browser still gets live cell shapes even with the Python API down.
      stopGeometryWatcher = startGeometryWatcher((chunk) => {
        try {
          ctrl.enqueue(chunk)
        } catch {
          // Controller closed — geometry watcher cleanup happens in cancel().
        }
      })

      // Periodic heartbeat keeps the connection alive so the browser does not
      // immediately close and attempt a reconnect flood.
      heartbeatTimer = setInterval(() => {
        try {
          ctrl.enqueue(enc.encode(sseComment('heartbeat')))
        } catch {
          // Controller closed (client disconnected) — stop.
          if (heartbeatTimer) clearInterval(heartbeatTimer)
        }
      }, HEARTBEAT_INTERVAL_MS)
    },
    cancel() {
      if (heartbeatTimer) clearInterval(heartbeatTimer)
      stopGeometryWatcher?.()
    },
  })
}

// ── Route handler ─────────────────────────────────────────────────────────────

export const GET: APIRoute = async ({ request }) => {
  const controller = new AbortController()
  const connectTimeout = setTimeout(() => controller.abort(), CONNECT_TIMEOUT_MS)

  // Abort the backend fetch when the client disconnects.
  request.signal?.addEventListener('abort', () => controller.abort())

  const headers = {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no', // Disable Nginx / Vercel edge buffering
  }

  let backendRes: Response
  try {
    backendRes = await fetch(`${BACKEND_URL}/api/cell-events`, {
      method: 'GET',
      headers: { Accept: 'text/event-stream' },
      signal: controller.signal,
      // @ts-ignore — Node 18+ fetch supports this; keeps the TCP connection open
      duplex: 'half',
    })
    clearTimeout(connectTimeout)
  } catch (err: any) {
    clearTimeout(connectTimeout)
    const reason =
      err?.name === 'AbortError' ? `connect timeout (${CONNECT_TIMEOUT_MS}ms)` : err?.message ?? String(err)
    return new Response(makeFallbackStream(reason), { status: 200, headers })
  }

  if (!backendRes.ok || !backendRes.body) {
    const reason = `backend returned HTTP ${backendRes.status}`
    return new Response(makeFallbackStream(reason), { status: 200, headers })
  }

  // ── Pass-through: pipe the backend SSE stream directly to the client ────────
  // The Python server already formats the events correctly.  We just relay them.
  // A TransformStream lets us inject a retry preamble (and M1305 geometry
  // events) without buffering the full backend body.

  const enc = new TextEncoder()
  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>()
  const writer = writable.getWriter()

  // Serialize writes — the backend relay loop and the geometry watcher both
  // push into `writer`, and WritableStreamDefaultWriter.write() calls must
  // not overlap or SSE frames can interleave.
  let writeQueue: Promise<void> = Promise.resolve()
  let writerClosed = false
  const safeWrite = (chunk: Uint8Array) => {
    if (writerClosed) return
    writeQueue = writeQueue.then(() =>
      writer.write(chunk).catch(() => {
        // Writer already closed/errored — geometry watcher will be stopped
        // by the relay loop's `finally` block below.
      }),
    )
  }

  // M1305: start geometry.json polling alongside the backend relay so the
  // browser gets `geometry_updated` even when topology/physics events come
  // from the Python proxy.
  const stopGeometryWatcher = startGeometryWatcher(safeWrite)

  // Write preamble (retry hint) then stream backend body asynchronously.
  ;(async () => {
    try {
      safeWrite(enc.encode('retry: 2000\n\n'))

      const reader = backendRes.body!.getReader()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        safeWrite(value)
      }
    } catch {
      // Client disconnected or backend closed — close gracefully.
    } finally {
      stopGeometryWatcher()
      writerClosed = true
      try {
        await writeQueue
        await writer.close()
      } catch { /* ignore */ }
    }
  })()

  return new Response(readable, { status: 200, headers })
}
