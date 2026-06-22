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
 * When the Python backend is offline the route falls back to a synthetic
 * heartbeat stream so the browser EventSource does not enter an error-
 * reconnect storm.
 *
 * Reconnect:
 *   The browser EventSource reconnects automatically.  We set `retry: 2000`
 *   in the stream preamble so the browser waits 2 s between retries.
 */

import type { APIRoute } from 'astro'

export const prerender = false

const BACKEND_URL =
  import.meta.env.PYTHON_BACKEND_URL ||
  import.meta.env.BACKEND_URL ||
  'http://127.0.0.1:8000'

const CONNECT_TIMEOUT_MS = 5_000
const HEARTBEAT_INTERVAL_MS = 15_000

// ── Helpers ──────────────────────────────────────────────────────────────────

function sseComment(text: string): string {
  return `: ${text}\n\n`
}

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

// ── Fallback heartbeat stream ─────────────────────────────────────────────────
// Returned when the Python backend cannot be reached so the browser receives a
// valid text/event-stream and does not loop through rapid reconnect attempts.

function makeFallbackStream(reason: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  let timer: ReturnType<typeof setInterval> | null = null

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

      // Periodic heartbeat keeps the connection alive so the browser does not
      // immediately close and attempt a reconnect flood.
      timer = setInterval(() => {
        try {
          ctrl.enqueue(enc.encode(sseComment('heartbeat')))
        } catch {
          // Controller closed (client disconnected) — stop.
          if (timer) clearInterval(timer)
        }
      }, HEARTBEAT_INTERVAL_MS)
    },
    cancel() {
      if (timer) clearInterval(timer)
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
  // A TransformStream lets us inject a retry preamble without buffering the full
  // body.

  const enc = new TextEncoder()
  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>()
  const writer = writable.getWriter()

  // Write preamble (retry hint) then stream backend body asynchronously.
  ;(async () => {
    try {
      await writer.write(enc.encode('retry: 2000\n\n'))

      const reader = backendRes.body!.getReader()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        await writer.write(value)
      }
    } catch {
      // Client disconnected or backend closed — close gracefully.
    } finally {
      try { await writer.close() } catch { /* ignore */ }
    }
  })()

  return new Response(readable, { status: 200, headers })
}
