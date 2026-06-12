/**
 * thread-pool.ts — Off-main-thread cell computation via Web Worker pool.
 *
 * Ported from AT Thread/Worker/GeomThread architecture (89 references):
 *   upstream/pixijs-engine/src/assets/loader/workers/WorkerManager.ts
 *   upstream/pixijs-engine/src/assets/loader/workers/loadImageBitmap.worker.ts
 *
 * AT uses workers for GeomThread (geometry), DracoThread (decompression),
 * MatrixWasm (matrix ops). Here we apply the same pattern to cell-loop
 * computation tasks currently running in the Python backend:
 *
 *   physics    — spring-force/repulsion between cells (loop_orchestrator.py
 *                _tiled_constraint_solve + _sweep_line_overlaps)
 *   layout     — ELK layered-graph layout (elkjs web-worker mode)
 *   visibility — NaniteVisibility frustum culling (cell_component.py
 *                ASTRO-VISIBILITY occlusion pass)
 *
 * Architecture:
 *   ThreadPool manages N = (navigator.hardwareConcurrency - 1) workers.
 *   postTask(type, data) → Promise<result>: promise-based dispatch with
 *   automatic load-balancing to the least-busy worker (idleQueue drain).
 *   Worker code is inlined as a Blob URL so no separate worker bundle is
 *   needed — same pattern as AT WorkerManager with loadImageBitmap.worker.ts.
 *
 * Usage (reduces API polling latency by running physics client-side):
 *   const pool = new ThreadPool();
 *   const forces = await pool.postTask('physics', { bboxes, forceField });
 *   const routed = await pool.postTask('layout', { graph });
 *   const visible = await pool.postTask('visibility', { cells });
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type TaskType = 'physics' | 'layout' | 'visibility';

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  species?: string;
  epoch?: number;
}

export interface ForceVector {
  dx: number;
  dy: number;
  dz: number;
}

/** Input / output contracts for each task type. */

export interface PhysicsInput {
  bboxes: Record<string, BBox>;
  forceField: Record<string, ForceVector>;
  canvasW?: number;
  canvasH?: number;
}

export interface PhysicsOutput {
  forceField: Record<string, ForceVector>;
  collisions: Array<{ a: string; b: string; overlap: number; z: number }>;
}

export interface LayoutNode {
  id: string;
  width: number;
  height: number;
  layoutOptions?: Record<string, string>;
}

export interface LayoutEdge {
  id: string;
  sources: string[];
  targets: string[];
}

export interface ELKGraph {
  id: string;
  layoutOptions?: Record<string, string>;
  children: LayoutNode[];
  edges: LayoutEdge[];
}

export interface LayoutInput {
  graph: ELKGraph;
}

export interface LayoutOutput {
  graph: ELKGraph;
  /** Map from node id → computed {x, y, width, height} */
  positions: Record<string, { x: number; y: number; width: number; height: number }>;
}

export interface VisibilityCell {
  cell_id: string;
  bbox: BBox;
  species?: string;
}

export interface VisibilityInput {
  cells: VisibilityCell[];
  /** Canvas viewport: [minX, minY, maxX, maxY] */
  viewport: [number, number, number, number];
}

export interface VisibilityOutput {
  /** cell_ids that survive frustum culling */
  visible: string[];
  /** cell_ids culled as fully occluded (subset of cells with same z-layer) */
  culled: string[];
}

export type TaskInput = PhysicsInput | LayoutInput | VisibilityInput;
export type TaskOutput = PhysicsOutput | LayoutOutput | VisibilityOutput;

// ── Internal message protocol (mirrors AT WorkerManager uuid scheme) ─────────

interface WorkerRequest {
  uuid: number;
  type: TaskType;
  data: TaskInput;
}

interface WorkerResponse {
  uuid: number;
  result?: TaskOutput;
  error?: string;
}

// ── Inline worker source ──────────────────────────────────────────────────────
//
// All task implementations live here as a self-contained script string that is
// converted to a Blob URL.  This is the exact same approach as AT's
// loadImageBitmap.worker.ts: one self-contained worker handles multiple IDs
// dispatched by the "id" field in the message payload.
//
// Physics algorithm: direct TypeScript port of loop_orchestrator.py
//   _tiled_constraint_solve (commit 7c82b90 FAstroTiledConstraintSolver) and
//   _sweep_line_overlaps (commit 0b4b199 FAstroCellSweepLine).
//
// Visibility algorithm: direct port of assemble_final_svg ASTRO-VISIBILITY
//   occlusion pass (SceneVisibility.cpp 6d345e7).

const WORKER_SOURCE = /* javascript */ `
'use strict';

// ─── Physics: FAstroTiledConstraintSolver ────────────────────────────────────
// Ported from channels/loop_orchestrator.py _tiled_constraint_solve +
// _sweep_line_overlaps (upstream/unreal-renderer commits 7c82b90, 0b4b199).
// Each cell bbox is assigned to a 16×16 canvas tile; constraints are resolved
// only within each tile's 3×3 neighbourhood — O(N·K) vs O(N²).

const TILE_SIZE = 16;

function sweepLineOverlaps(rectsByZ) {
  const OPEN = 1, CLOSE = 0;
  const pairs = [];

  for (const [zLayer, rects] of Object.entries(rectsByZ)) {
    if (rects.length < 2) continue;

    // Build events: [x, -kind, idx] — OPEN at min_x, CLOSE at max_x.
    // Sort: ascending x; at equal x, OPEN (-1) before CLOSE (0).
    const events = [];
    for (let idx = 0; idx < rects.length; idx++) {
      const [, minX, minY, maxX, maxY] = rects[idx];
      if (minX > maxX || minY > maxY) continue;
      events.push([minX, -OPEN,  idx]);
      events.push([maxX, -CLOSE, idx]);
    }
    events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    const active = [];
    for (const [, negKind, idx] of events) {
      const isOpen = (-negKind === OPEN);
      const [cellId, minX, minY, maxX, maxY] = rects[idx];

      if (isOpen) {
        for (const aidx of active) {
          const [aCellId, , aMinY, , aMaxY] = rects[aidx];
          if (aMinY <= maxY && aMaxY >= minY) {
            const [, aMinX, , aMaxX] = rects[aidx];
            const ovX = Math.min(maxX, aMaxX) - Math.max(minX, aMinX);
            const ovY = Math.min(maxY, aMaxY) - Math.max(minY, aMinY);
            if (ovX > 0 && ovY > 0) {
              pairs.push([cellId, aCellId, ovX, ovY, Number(zLayer)]);
            }
          }
        }
        active.push(idx);
      } else {
        const i = active.indexOf(idx);
        if (i !== -1) active.splice(i, 1);
      }
    }
  }
  return pairs;
}

function tiledConstraintSolve(bboxes, forceField, canvasW, canvasH) {
  const cellIds = Object.keys(bboxes);
  if (cellIds.length === 0) return [];

  if (canvasW == null || canvasH == null) {
    let maxX = 0, maxY = 0;
    for (const b of Object.values(bboxes)) {
      maxX = Math.max(maxX, b.x + b.w);
      maxY = Math.max(maxY, b.y + b.h);
    }
    canvasW = canvasW ?? Math.max(Math.ceil(maxX), TILE_SIZE);
    canvasH = canvasH ?? Math.max(Math.ceil(maxY), TILE_SIZE);
  }

  const numTilesX = Math.ceil(canvasW / TILE_SIZE);
  const numTilesY = Math.ceil(canvasH / TILE_SIZE);

  // Phase 1 — assign cells to tiles by bbox centre (BuildTileBatches).
  // tile_cells[ty * numTilesX + tx] → cell_id[]
  const tileCells = Array.from({ length: numTilesX * numTilesY }, () => []);
  for (const cellId of cellIds) {
    const b = bboxes[cellId];
    const cx = b.x + b.w * 0.5;
    const cy = b.y + b.h * 0.5;
    const tx = Math.max(0, Math.min(Math.floor(cx / TILE_SIZE), numTilesX - 1));
    const ty = Math.max(0, Math.min(Math.floor(cy / TILE_SIZE), numTilesY - 1));
    tileCells[ty * numTilesX + tx].push(cellId);
  }

  // Phase 2 — per-tile constraint resolution (ResolveAll, 3×3 neighbourhood).
  const collisions = [];
  const resolvedPairs = new Set();

  for (let ty = 0; ty < numTilesY; ty++) {
    for (let tx = 0; tx < numTilesX; tx++) {
      // Collect neighbourhood cells (self tile + 8 adjacent = 3×3 kernel).
      const neighbourhood = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ntx = tx + dx, nty = ty + dy;
          if (ntx >= 0 && ntx < numTilesX && nty >= 0 && nty < numTilesY) {
            neighbourhood.push(...tileCells[nty * numTilesX + ntx]);
          }
        }
      }
      if (neighbourhood.length < 2) continue;

      // O(K²) pair test within the neighbourhood.
      for (let i = 0; i < neighbourhood.length; i++) {
        for (let j = i + 1; j < neighbourhood.length; j++) {
          const cellA = neighbourhood[i];
          const cellB = neighbourhood[j];
          const pairKey = cellA < cellB ? cellA + '\x00' + cellB : cellB + '\x00' + cellA;
          if (resolvedPairs.has(pairKey)) continue;

          const ba = bboxes[cellA];
          const bb = bboxes[cellB];
          if ((ba.z ?? 3) !== (bb.z ?? 3)) continue;

          const aMinX = ba.x, aMaxX = ba.x + ba.w;
          const aMinY = ba.y, aMaxY = ba.y + ba.h;
          const bMinX = bb.x, bMaxX = bb.x + bb.w;
          const bMinY = bb.y, bMaxY = bb.y + bb.h;

          const ovX = Math.min(aMaxX, bMaxX) - Math.max(aMinX, bMinX);
          const ovY = Math.min(aMaxY, bMaxY) - Math.max(aMinY, bMinY);
          if (ovX <= 0 || ovY <= 0) continue;

          resolvedPairs.add(pairKey);
          collisions.push({ a: cellA, b: cellB, overlap: ovX * ovY, z: ba.z ?? 3 });

          // Repulsion force — axis of least overlap (same formula as Python).
          if (ovX < ovY) {
            const push = ovX / 2 + 5;
            if (ba.x < bb.x) {
              forceField[cellA].dx -= push;
              forceField[cellB].dx += push;
            } else {
              forceField[cellA].dx += push;
              forceField[cellB].dx -= push;
            }
          } else {
            const push = ovY / 2 + 5;
            if (ba.y < bb.y) {
              forceField[cellA].dy -= push;
              forceField[cellB].dy += push;
            } else {
              forceField[cellA].dy += push;
              forceField[cellB].dy -= push;
            }
          }
        }
      }
    }
  }
  return collisions;
}

function runPhysics(input) {
  const { bboxes, canvasW, canvasH } = input;
  const cellIds = Object.keys(bboxes);

  // Zero-initialise force field for this epoch (same as Python physics_engine).
  const forceField = {};
  for (const id of cellIds) {
    forceField[id] = { dx: 0, dy: 0, dz: 0 };
  }

  const collisions = tiledConstraintSolve(bboxes, forceField, canvasW, canvasH);
  return { forceField, collisions };
}

// ─── Layout: ELK web-worker bridge ───────────────────────────────────────────
// elkjs supports a built-in web-worker mode; we trigger it via the bundled UMD.
// If ELK is not available (no importScripts in this context), we return the
// input graph unchanged so the caller can fall back to the main-thread path.

async function runLayout(input) {
  const { graph } = input;
  try {
    // Dynamic import of the ELK bundled build (UMD self-registers as ELK global).
    // This mirrors how src/lib/elk/layout.ts uses elkjs/lib/elk.bundled.js.
    importScripts('https://unpkg.com/elkjs@0.9.3/lib/elk.bundled.js');
    // eslint-disable-next-line no-undef
    const elk = new ELK();
    const result = await elk.layout(graph);

    const positions = {};
    for (const node of (result.children ?? [])) {
      positions[node.id] = { x: node.x ?? 0, y: node.y ?? 0, width: node.width ?? 0, height: node.height ?? 0 };
    }
    return { graph: result, positions };
  } catch {
    // ELK not available or layout failed — return identity so caller can retry
    const positions = {};
    for (const node of (graph.children ?? [])) {
      positions[node.id] = { x: node.x ?? 0, y: node.y ?? 0, width: node.width ?? 0, height: node.height ?? 0 };
    }
    return { graph, positions };
  }
}

// ─── Visibility: NaniteVisibility frustum culling ────────────────────────────
// Ported from channels/cell_component.py [ASTRO-VISIBILITY] pass and
// loop_orchestrator.py assemble_final_svg occlusion block
// (upstream/unreal-renderer/SceneVisibility.cpp commit 6d345e7).
//
// Two-pass algorithm (mirrors SceneVisibilityState::ComputeVisibility):
//   Pass 1 — Viewport frustum test: cull cells whose AABB lies entirely
//     outside the [minX, minY, maxX, maxY] viewport rectangle.
//   Pass 2 — Same-z occlusion cull: cell A occludes cell B when A fully
//     contains B's AABB and both share the same z-layer (the Python version
//     calls this "ASTRO-VISIBILITY" block in assemble_final_svg).

function runVisibility(input) {
  const { cells, viewport } = input;
  const [vpMinX, vpMinY, vpMaxX, vpMaxY] = viewport;

  const visible = [];
  const culled = [];

  // Pass 1 — frustum cull against viewport.
  const afterFrustum = [];
  for (const cell of cells) {
    const { x, y, w, h } = cell.bbox;
    // Cell is outside the viewport if it does not intersect at all.
    if (x + w < vpMinX || x > vpMaxX || y + h < vpMinY || y > vpMaxY) {
      culled.push(cell.cell_id);
    } else {
      afterFrustum.push(cell);
    }
  }

  // Pass 2 — same-z occlusion cull (A fully contains B → B culled).
  // Mirrors the nested loop in assemble_final_svg with [ASTRO-VISIBILITY] tag.
  const occluded = new Set();
  for (let i = 0; i < afterFrustum.length; i++) {
    if (occluded.has(afterFrustum[i].cell_id)) continue;
    const sa = afterFrustum[i];
    const ba = sa.bbox;
    for (let j = 0; j < afterFrustum.length; j++) {
      if (i === j || occluded.has(afterFrustum[j].cell_id)) continue;
      const sb = afterFrustum[j];
      if (sa.bbox.z !== sb.bbox.z) continue;
      const bb = sb.bbox;
      // A fully contains B: all four edges of B are inside A.
      if (ba.x <= bb.x && ba.y <= bb.y && ba.x + ba.w >= bb.x + bb.w && ba.y + ba.h >= bb.y + bb.h) {
        occluded.add(sb.cell_id);
      }
    }
  }

  for (const cell of afterFrustum) {
    if (occluded.has(cell.cell_id)) {
      culled.push(cell.cell_id);
    } else {
      visible.push(cell.cell_id);
    }
  }

  return { visible, culled };
}

// ─── Worker message dispatcher ────────────────────────────────────────────────
// Mirrors AT loadImageBitmap.worker.ts self.onmessage handler:
// receive {uuid, type, data}, dispatch to handler, postMessage result back.

self.onmessage = async function(event) {
  const { uuid, type, data } = event.data;
  try {
    let result;
    if (type === 'physics') {
      result = runPhysics(data);
    } else if (type === 'layout') {
      result = await runLayout(data);
    } else if (type === 'visibility') {
      result = runVisibility(data);
    } else {
      throw new Error('[ThreadPool] Unknown task type: ' + type);
    }
    self.postMessage({ uuid, result });
  } catch (e) {
    self.postMessage({ uuid, error: String(e) });
  }
};
`;

// ── ThreadPool class ──────────────────────────────────────────────────────────
//
// Manages N = max(1, navigator.hardwareConcurrency - 1) workers.
// postTask() is promise-based; load balancing via an idle-queue: tasks are
// dispatched immediately when a worker is free, otherwise queued and drained
// when the next worker becomes available.
//
// Design mirrors AT WorkerManagerClass (WorkerManager.ts):
//   _workerPool  → idleWorkers (available workers ready to accept tasks)
//   _queue       → pendingQueue (tasks waiting for a free worker)
//   _resolveHash → inflightMap  (promise callbacks keyed by UUID)
//   _next()      → _drain()     (process next queue item on worker free)
//   _complete()  → _settle()    (resolve/reject promise from worker response)

let _poolUUID = 0;

interface PendingTask {
  type: TaskType;
  data: TaskInput;
  resolve: (value: TaskOutput) => void;
  reject: (reason: unknown) => void;
}

interface InflightEntry {
  resolve: (value: TaskOutput) => void;
  reject: (reason: unknown) => void;
}

/**
 * ThreadPool — Web Worker pool for off-main-thread cell computation.
 *
 * Ported from AT WorkerManagerClass (upstream/pixijs-engine/src/assets/loader/workers/WorkerManager.ts).
 * Workers run the WORKER_SOURCE blob: physics, layout, and visibility tasks.
 *
 * @example
 * ```ts
 * const pool = new ThreadPool();
 * const { forceField, collisions } = await pool.postTask('physics', { bboxes, forceField });
 * await pool.dispose();
 * ```
 */
export class ThreadPool {
  /** Workers waiting for a task (idle pool). */
  private readonly idleWorkers: Worker[] = [];
  /** Tasks waiting for a free worker. */
  private readonly pendingQueue: PendingTask[] = [];
  /** In-flight tasks keyed by UUID. */
  private readonly inflightMap = new Map<number, InflightEntry>();
  /** Total workers created (hard cap = workerCount). */
  private createdWorkers = 0;
  /** Maximum concurrent workers: hardwareConcurrency - 1, min 1. */
  readonly workerCount: number;
  /** Blob URL for the inline worker script. */
  private readonly workerUrl: string;
  private disposed = false;

  constructor() {
    // N = hardwareConcurrency - 1 (reserve one core for main thread).
    // Mirrors AT MAX_WORKERS = navigator.hardwareConcurrency || 4.
    const concurrency =
      typeof navigator !== 'undefined' && navigator.hardwareConcurrency
        ? navigator.hardwareConcurrency
        : 4;
    this.workerCount = Math.max(1, concurrency - 1);

    // Inline blob URL — no separate worker bundle required.
    // Same pattern as AT CheckImageBitmapWorker / LoadImageBitmapWorker
    // which use the 'worker:' import prefix to create Blob URLs at build time.
    const blob = new Blob([WORKER_SOURCE], { type: 'application/javascript' });
    this.workerUrl = URL.createObjectURL(blob);
  }

  /**
   * postTask — dispatch a task to the next available worker.
   *
   * Mirrors AT WorkerManagerClass._run():
   *   1. Push task onto pendingQueue.
   *   2. Call _drain() to attempt immediate dispatch.
   *   3. Return Promise that resolves when the worker responds.
   */
  postTask(type: 'physics', data: PhysicsInput): Promise<PhysicsOutput>;
  postTask(type: 'layout', data: LayoutInput): Promise<LayoutOutput>;
  postTask(type: 'visibility', data: VisibilityInput): Promise<VisibilityOutput>;
  postTask(type: TaskType, data: TaskInput): Promise<TaskOutput> {
    if (this.disposed) {
      return Promise.reject(new Error('[ThreadPool] Pool has been disposed'));
    }
    return new Promise<TaskOutput>((resolve, reject) => {
      this.pendingQueue.push({ type, data, resolve, reject });
      this._drain();
    });
  }

  /**
   * _drain — dispatch next pending task to an available worker.
   * Mirrors AT WorkerManagerClass._next(): dequeue one item, pop one worker.
   */
  private _drain(): void {
    if (this.pendingQueue.length === 0) return;

    const worker = this._acquireWorker();
    if (!worker) return;  // all workers busy; will re-drain on task completion

    const task = this.pendingQueue.shift()!;
    const uuid = _poolUUID++;

    this.inflightMap.set(uuid, { resolve: task.resolve, reject: task.reject });

    const request: WorkerRequest = { uuid, type: task.type, data: task.data };
    worker.postMessage(request);
  }

  /**
   * _acquireWorker — get or create a worker (up to workerCount cap).
   * Mirrors AT WorkerManagerClass._getWorker():
   *   pop from idle pool; if empty and under cap, spawn new worker.
   */
  private _acquireWorker(): Worker | undefined {
    const idle = this.idleWorkers.pop();
    if (idle) return idle;

    if (this.createdWorkers >= this.workerCount) return undefined;
    this.createdWorkers++;

    const worker = new Worker(this.workerUrl);

    worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      this._settle(event.data);
      // Return worker to idle pool before draining so _drain can reuse it.
      this.idleWorkers.push(worker);
      this._drain();
    });

    worker.addEventListener('error', (event: ErrorEvent) => {
      // Propagate to all in-flight tasks on this worker (worker is broken).
      // In practice AT WorkerManager does not handle this; we add it as a
      // safety measure — reject all pending inflightMap entries.
      const msg = `[ThreadPool] Worker error: ${event.message}`;
      for (const [uuid, entry] of this.inflightMap) {
        entry.reject(new Error(msg));
        this.inflightMap.delete(uuid);
      }
      // Worker is broken — reduce cap so _acquireWorker can spawn a replacement.
      this.createdWorkers--;
      this._drain();
    });

    return worker;
  }

  /**
   * _settle — resolve or reject the promise for a completed task.
   * Mirrors AT WorkerManagerClass._complete().
   */
  private _settle(response: WorkerResponse): void {
    const entry = this.inflightMap.get(response.uuid);
    if (!entry) return;  // stale response (pool was reset); ignore

    this.inflightMap.delete(response.uuid);

    if (response.error !== undefined) {
      entry.reject(new Error(response.error));
    } else {
      entry.resolve(response.result!);
    }
  }

  /**
   * dispose — terminate all workers and reject pending promises.
   * Mirrors AT WorkerManagerClass.reset().
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    // Terminate all idle workers.
    for (const w of this.idleWorkers) w.terminate();
    this.idleWorkers.length = 0;

    // Reject all pending queue entries (workers never picked them up).
    const err = new Error('[ThreadPool] Pool disposed before task completed');
    for (const task of this.pendingQueue) task.reject(err);
    this.pendingQueue.length = 0;

    // Reject all in-flight tasks (worker may have been killed mid-run).
    for (const entry of this.inflightMap.values()) entry.reject(err);
    this.inflightMap.clear();

    // Revoke the Blob URL (mirrors AT CheckImageBitmapWorker.revokeObjectURL()).
    URL.revokeObjectURL(this.workerUrl);
  }

  /** Convenience: number of tasks currently queued but not yet dispatched. */
  get queueDepth(): number {
    return this.pendingQueue.length;
  }

  /** Convenience: number of tasks currently running on workers. */
  get inflightCount(): number {
    return this.inflightMap.size;
  }
}

// ── Module-level singleton ────────────────────────────────────────────────────
//
// Mirrors AT's `const WorkerManager = new WorkerManagerClass()` singleton export.
// Import the singleton for shared use across the app, or construct a private
// ThreadPool instance for isolated lifecycle management.

let _sharedPool: ThreadPool | null = null;

/**
 * getThreadPool — return the shared ThreadPool singleton.
 *
 * Lazily initialised on first call (same deferred init as AT WorkerManager).
 * Call pool.dispose() only if you intend to shut down the entire app.
 */
export function getThreadPool(): ThreadPool {
  if (!_sharedPool || (_sharedPool as ThreadPool).disposed) {
    _sharedPool = new ThreadPool();
  }
  return _sharedPool;
}
