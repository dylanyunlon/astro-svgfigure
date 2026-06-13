/**
 * Thread.ts — AT-compatible Worker manager with shared/generate/cluster API.
 *
 * Mirrors the AT Thread architecture (upstream/pixijs-engine workers pattern):
 *   Thread.shared()   → singleton worker by name (WorkerManager pattern)
 *   Thread.generate() → inline Blob worker from code string (AT worker blob pattern)
 *   Thread.cluster()  → named worker pool (N instances, load-balanced)
 *
 * send(method, data) → Promise<any>: promise-based RPC with auto UUID dispatch,
 * matching the AT WorkerManager uuid-keyed inflightMap pattern from thread-pool.ts.
 */

// ── Internal message protocol ─────────────────────────────────────────────────

interface ThreadRequest {
  uuid: number;
  method: string;
  data: unknown;
}

interface ThreadResponse {
  uuid: number;
  result?: unknown;
  error?: string;
}

// ── Module state ──────────────────────────────────────────────────────────────

let _uuid = 0;

/** Singleton workers keyed by name (Thread.shared cache). */
const _sharedRegistry = new Map<string, Thread>();

/** Cluster pools keyed by `name:count` (Thread.cluster cache). */
const _clusterRegistry = new Map<string, Thread[]>();

// ── Thread class ──────────────────────────────────────────────────────────────

/**
 * Thread — thin wrapper around a single Web Worker with promise-based RPC.
 *
 * Each Thread instance owns one Worker. For pooled / load-balanced execution
 * use Thread.cluster(), which returns an array of Thread instances and exposes
 * the same send() API on each one.
 *
 * @example
 * ```ts
 * // Named worker file in /workers/ (e.g. public/workers/geometry-worker.js)
 * const t = Thread.shared('geometry-worker');
 * const buffers = await t.send('computeVertices', { cells });
 *
 * // Inline Blob worker
 * const t2 = Thread.generate(`
 *   self.onmessage = e => {
 *     const { uuid, method, data } = e.data;
 *     self.postMessage({ uuid, result: data.x * 2 });
 *   };
 * `);
 * const doubled = await t2.send('double', { x: 21 }); // → 42
 *
 * // Worker pool
 * const pool = Thread.cluster('geometry-worker', 4);
 * const results = await Promise.all(pool.map(t => t.send('computeVertices', { cells })));
 * ```
 */
export class Thread {
  /**
   * Base path for named workers resolved by Thread.shared() and Thread.cluster().
   * Override at app startup: `Thread.PATH = '/dist/workers/';`
   */
  static PATH = '/workers/';

  /** Underlying Web Worker. */
  private worker: Worker;

  /** Pending RPC callbacks keyed by UUID (mirrors AT inflightMap). */
  private callbacks: Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void }> =
    new Map();

  /** True after dispose() — sends will reject immediately. */
  private _disposed = false;

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * Create a Thread from an existing Worker or a URL string.
   * Prefer the static factory methods (shared / generate / cluster).
   */
  constructor(workerOrUrl: Worker | string) {
    this.worker =
      workerOrUrl instanceof Worker
        ? workerOrUrl
        : new Worker(workerOrUrl, { type: 'module' });

    this.worker.addEventListener('message', (ev: MessageEvent<ThreadResponse>) => {
      this._settle(ev.data);
    });

    this.worker.addEventListener('error', (ev: ErrorEvent) => {
      const err = new Error(`[Thread] Worker error: ${ev.message}`);
      for (const cb of this.callbacks.values()) cb.reject(err);
      this.callbacks.clear();
    });
  }

  // ── Static factories ────────────────────────────────────────────────────────

  /**
   * shared — return (or create) a singleton Thread for a named worker script.
   *
   * Worker is loaded from `Thread.PATH + name + '.js'`.
   * Subsequent calls with the same name return the cached instance.
   * Mirrors AT WorkerManager singleton pattern.
   *
   * @param name  Worker filename without extension (e.g. 'geometry-worker').
   */
  static shared(name: string): Thread {
    if (_sharedRegistry.has(name)) {
      const existing = _sharedRegistry.get(name)!;
      if (!existing._disposed) return existing;
      _sharedRegistry.delete(name);
    }
    const thread = new Thread(`${Thread.PATH}${name}.js`);
    _sharedRegistry.set(name, thread);
    return thread;
  }

  /**
   * generate — create a Thread from an inline code string via Blob URL.
   *
   * The code string must be a complete self-contained worker script.
   * Mirrors AT loadImageBitmap.worker.ts Blob URL pattern and the
   * WORKER_SOURCE approach in thread-pool.ts.
   *
   * @param code  Worker script source (plain JS / no module syntax).
   */
  static generate(code: string): Thread {
    const blob = new Blob([code], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    // Use classic (non-module) worker for Blob URLs — same as AT WorkerManager.
    const worker = new Worker(url);
    const thread = new Thread(worker);
    // Revoke Blob URL after worker is spawned (AT CheckImageBitmapWorker pattern).
    URL.revokeObjectURL(url);
    return thread;
  }

  /**
   * cluster — create a pool of `count` Thread instances for the same worker.
   *
   * Returns an array; callers choose their own dispatch strategy (round-robin,
   * Promise.all, etc.). Results are cached by `name:count`.
   *
   * @param name   Worker filename without extension.
   * @param count  Number of worker instances to spawn.
   */
  static cluster(name: string, count: number): Thread[] {
    const key = `${name}:${count}`;
    const existing = _clusterRegistry.get(key);
    if (existing && existing.every(t => !t._disposed)) return existing;

    const pool = Array.from({ length: count }, () => new Thread(`${Thread.PATH}${name}.js`));
    _clusterRegistry.set(key, pool);
    return pool;
  }

  // ── Instance API ────────────────────────────────────────────────────────────

  /**
   * send — dispatch a method call to the worker and return a Promise.
   *
   * The worker receives `{ uuid, method, data }` and must reply with
   * `{ uuid, result? }` or `{ uuid, error? }`.
   *
   * Transferable objects (ArrayBuffer, ImageBitmap, …) in `data.transfer`
   * are forwarded to postMessage as the transfer list.
   *
   * @param method  Logical method name (worker dispatches on this string).
   * @param data    Arbitrary payload.
   */
  send(method: string, data: unknown): Promise<unknown> {
    if (this._disposed) {
      return Promise.reject(new Error('[Thread] Worker has been disposed'));
    }

    return new Promise((resolve, reject) => {
      const uuid = _uuid++;
      this.callbacks.set(uuid, { resolve, reject });

      const request: ThreadRequest = { uuid, method, data };

      // Extract transferables if caller embedded them as data.transfer[].
      const transfer = (data as { transfer?: Transferable[] })?.transfer ?? [];
      this.worker.postMessage(request, transfer);
    });
  }

  /**
   * dispose — terminate the worker and reject all pending calls.
   * Idempotent; safe to call multiple times.
   */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;

    this.worker.terminate();

    const err = new Error('[Thread] Worker disposed before response');
    for (const cb of this.callbacks.values()) cb.reject(err);
    this.callbacks.clear();
  }

  // ── Internal ────────────────────────────────────────────────────────────────

  private _settle(response: ThreadResponse): void {
    const cb = this.callbacks.get(response.uuid);
    if (!cb) return; // stale / already settled

    this.callbacks.delete(response.uuid);

    if (response.error !== undefined) {
      cb.reject(new Error(response.error));
    } else {
      cb.resolve(response.result);
    }
  }
}
