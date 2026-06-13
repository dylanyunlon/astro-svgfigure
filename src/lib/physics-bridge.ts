/**
 * physics-bridge.ts — Main-thread bridge to the physics WebWorker.
 *
 * Wraps the request/response messaging of physics-worker.ts into a clean
 * async/await API so callers never touch postMessage / onmessage directly.
 *
 * Architecture (mirrors AT WorkerManager pattern from
 * upstream/pixijs-engine/src/assets/loader/workers/WorkerManager.ts):
 *
 *   PhysicsBridge          → WorkerManager singleton
 *   computeForces(cells)   → postTask('physics', payload) → Promise<ForceField>
 *   Worker instance        → lazily created, re-used across calls
 *   Pending queue          → resolves / rejects isolated per-call via msgId
 *
 * Usage:
 *   import { computeForces } from './physics-bridge';
 *   const forces = await computeForces(cells);
 *   // forces['attention1'] → { dx: -12.5, dy: 0, dz: 0 }
 */

import type { CellBBox, ForceField } from './physics-worker';

export type { CellBBox, ForceField };

// ─── Worker URL ────────────────────────────────────────────────────────────────
//
// Vite / Astro recognise the `?worker` suffix and bundle physics-worker.ts as a
// separate chunk, returning a constructor.  In non-bundler environments the URL
// import falls back gracefully (the bridge will construct it via new URL + import.meta).
//
// We use the module Worker variant (type: 'module') so physics-worker.ts can use
// ES import/export syntax — same approach as AT's loadImageBitmap.worker.ts.

let _worker: Worker | null = null;

/** Monotonically increasing message ID — used to match responses to requests. */
let _nextId = 0;

/** Pending promise callbacks keyed by message ID. */
const _pending = new Map<
  number,
  { resolve: (ff: ForceField) => void; reject: (err: Error) => void }
>();

// ─── Worker lifecycle ──────────────────────────────────────────────────────────

function _getWorker(): Worker {
  if (_worker) return _worker;

  // new URL(..., import.meta.url) is the standard Vite / Webpack 5 pattern for
  // worker asset resolution.  The `type: 'module'` option mirrors AT's
  // WorkerManager.createWorker({ type: 'module' }) call.
  _worker = new Worker(
    new URL('./physics-worker.ts', import.meta.url),
    { type: 'module' }
  );

  _worker.onmessage = (ev: MessageEvent<{ id: number; force_field?: ForceField; error?: string }>) => {
    const { id, force_field, error } = ev.data;
    const pending = _pending.get(id);
    if (!pending) return; // stale or unexpected message
    _pending.delete(id);

    if (error !== undefined) {
      pending.reject(new Error(`physics-worker error: ${error}`));
    } else if (force_field !== undefined) {
      pending.resolve(force_field);
    } else {
      pending.reject(new Error('physics-worker: unexpected response shape'));
    }
  };

  _worker.onerror = (ev: ErrorEvent) => {
    // Reject all pending requests on worker crash.
    const err = new Error(`physics-worker crashed: ${ev.message}`);
    for (const { reject } of _pending.values()) reject(err);
    _pending.clear();
    _worker = null; // allow lazy re-creation on next call
  };

  return _worker;
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * computeForces — send `cells` to the physics WebWorker and receive the
 * resulting force_field as a resolved Promise.
 *
 * Each call is independent; concurrent calls are correctly multiplexed via
 * per-message `id` tags (same correlation pattern as AT WorkerManager's
 * resolve/reject map keyed by request UUID).
 *
 * @param cells  Array of CellBBox descriptors (id, x, y, w, h, z?).
 * @returns      Promise<ForceField>  — resolves with { [cellId]: {dx, dy, dz} }.
 *               Rejects if the worker throws or crashes.
 *
 * @example
 *   const forces = await computeForces([
 *     { id: 'embed', x: 10, y: 10, w: 120, h: 40, z: 2 },
 *     { id: 'attn',  x: 10, y: 60, w: 120, h: 40, z: 2 },
 *   ]);
 *   console.log(forces['embed']); // { dx: 0, dy: -12.5, dz: 0 }
 */
export function computeForces(cells: CellBBox[]): Promise<ForceField> {
  return new Promise<ForceField>((resolve, reject) => {
    const id = _nextId++;
    _pending.set(id, { resolve, reject });

    try {
      const worker = _getWorker();
      // Tag the outbound message with `id` so the response handler above can
      // route the reply to the correct pending entry.
      //
      // Note: physics-worker.ts currently ignores `id` on the inbound side and
      // echoes it back in the response.  The worker was written without id-tagging
      // to keep it minimal; the bridge wraps that with its own id in the payload.
      worker.postMessage({ id, cells });
    } catch (err) {
      _pending.delete(id);
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/**
 * terminatePhysicsWorker — tear down the worker instance.
 *
 * Call during page unload or hot-module reload to avoid zombie workers.
 * Mirrors WorkerManager.destroy() from AT's WorkerManager.ts.
 */
export function terminatePhysicsWorker(): void {
  if (_worker) {
    _worker.terminate();
    _worker = null;
  }
  for (const { reject } of _pending.values()) {
    reject(new Error('physics-worker: terminated'));
  }
  _pending.clear();
}
