/**
 * physics-bridge.ts — Main-thread bridge to the physics WebWorker.
 *
 * Uses comlink.wrap() to replace manual postMessage/onmessage serialization
 * with a transparent async proxy.  The main thread simply awaits proxy methods
 * as if calling local async functions.
 *
 * Architecture (mirrors AT WorkerManager pattern from
 * upstream/pixijs-engine/src/assets/loader/workers/WorkerManager.ts):
 *
 *   PhysicsBridge          → WorkerManager singleton
 *   computeForces(cells)   → proxy.computeForceField(cells) → Promise<ForceField>
 *   Worker instance        → lazily created, re-used across calls
 *
 * Usage:
 *   import { computeForces, detectCollisions, checkConvergence } from './physics-bridge';
 *   const forces = await computeForces(cells);
 *   // forces['attention1'] → { dx: -12.5, dy: 0, dz: 0 }
 */

import { wrap } from '../../upstream/comlink/src/comlink';
import type { CellBBox, ForceField, OverlapPair, PhysicsWorkerAPI } from './physics-worker';

export type { CellBBox, ForceField, OverlapPair };

// ─── Worker proxy ──────────────────────────────────────────────────────────────
//
// wrap<PhysicsWorkerAPI>(worker) returns a proxy whose methods return Promises,
// backed by comlink's message routing — no manual id tagging or _pending map needed.

let _worker: Worker | null = null;
let _proxy: ReturnType<typeof wrap<PhysicsWorkerAPI>> | null = null;

function _getProxy(): ReturnType<typeof wrap<PhysicsWorkerAPI>> {
  if (_proxy) return _proxy;

  // new URL(..., import.meta.url) is the standard Vite / Webpack 5 pattern for
  // worker asset resolution.  The `type: 'module'` option mirrors AT's
  // WorkerManager.createWorker({ type: 'module' }) call.
  _worker = new Worker(
    new URL('./physics-worker.ts', import.meta.url),
    { type: 'module' }
  );

  // comlink.wrap() replaces the manual onmessage + _pending resolve/reject map.
  // All message serialization, id correlation, and error propagation are handled
  // by comlink internally.
  _proxy = wrap<PhysicsWorkerAPI>(_worker);

  return _proxy;
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * computeForces — send `cells` to the physics WebWorker and receive the
 * resulting force_field as a resolved Promise.
 *
 * Backed by proxy.computeForceField() via comlink; replaces the previous
 * manual postMessage({ id, cells }) + _pending map pattern.
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
  return _getProxy().computeForceField(cells);
}

/**
 * detectCollisions — run sweep-line overlap detection in the worker.
 * Returns all overlapping bbox pairs for the given cells.
 */
export function detectCollisions(cells: CellBBox[]): Promise<OverlapPair[]> {
  return _getProxy().detectCollisions(cells);
}

/**
 * checkConvergence — returns true when all force magnitudes are below `eps`.
 * Useful for terminating the physics iteration loop on the main thread.
 */
export function checkConvergence(forceField: ForceField, eps = 1.0): Promise<boolean> {
  return _getProxy().checkConvergence(forceField, eps);
}

/**
 * terminatePhysicsWorker — tear down the worker instance.
 *
 * Call during page unload or hot-module reload to avoid zombie workers.
 * Mirrors WorkerManager.destroy() from AT's WorkerManager.ts.
 */
export function terminatePhysicsWorker(): void {
  if (_proxy) {
    // Release comlink proxy (sends releaseProxy message to worker).
    // @ts-ignore — releaseProxy is a symbol key on the proxy object.
    _proxy[Symbol.for('Comlink.releaseProxy')]?.();
    _proxy = null;
  }
  if (_worker) {
    _worker.terminate();
    _worker = null;
  }
}
