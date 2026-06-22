/**
 * sph-epoch-bridge.ts — Backend epoch loop ↔ frontend SPHWorld bridge  (M514)
 *
 * Listens to the server-sent event stream at /api/cell-events and keeps the
 * SPHWorld's CollisionWorld in sync with the backend cell convergence loop:
 *
 *   SSE event              Action
 *   ─────────────────────────────────────────────────────────────────────────
 *   epoch_completed        Fetch GET /api/cells, upsert kinematic rigid bodies
 *                          for every cell using the cell bbox centre as (x, y).
 *   cell_params_updated    Fast-path: move the single named body without a full
 *                          /api/cells round-trip when only one cell changed.
 *   topology_updated       Full teardown + rebuild of all rigid bodies from the
 *                          new topology payload (cell set may have changed).
 *
 * Coordinate mapping
 * ──────────────────
 * Cell params store bboxes in canvas pixels (origin top-left, Y-down).
 * SPHWorld uses domain units (origin bottom-left, Y-up, typically 0–3 m).
 * The bridge normalises pixel coords to domain units using the canvas
 * dimensions provided at construction time (domainW, domainH, canvasW, canvasH).
 *
 *   domain_x = bbox.cx / canvasW  * domainW
 *   domain_y = (1 - bbox.cy / canvasH) * domainH   // flip Y
 *
 * Rigid body lifecycle
 * ─────────────────────
 *   • Bodies are created as 'kinematic' so SPH fluid is repelled but the
 *     backend (not physics sim) controls their position.
 *   • Cell radius is approximated as min(w, h) / 2, normalised to domain units.
 *   • A Map<cellId, bodyId> is kept so updates move existing bodies rather than
 *     creating duplicates.
 *   • On topology_updated all tracked bodies are removed before re-adding.
 *
 * Reconnect strategy
 * ───────────────────
 * Mirrors CellEventSource.ts: exponential backoff (1 s → 30 s, cap) is layered
 * on top of the browser's native EventSource reconnect so a server restart
 * during a long training run never requires a manual page refresh.
 *
 * Usage
 * ─────
 *   import { SPHEpochBridge } from '$lib/sph/sph-epoch-bridge';
 *
 *   const bridge = new SPHEpochBridge(sphWorld, {
 *     domainW: 3.0,
 *     domainH: 3.0,
 *     canvasW: canvas.width,
 *     canvasH: canvas.height,
 *   });
 *   bridge.start();
 *   // …later, on teardown:
 *   bridge.stop();
 *
 * Alternatively, pass a pre-constructed EventSource to reuse an existing
 * SSE connection:
 *
 *   const bridge = new SPHEpochBridge(sphWorld, opts, existingEventSource);
 *   bridge.start();
 */

import { SPHWorld }         from './SPHWorld';
import { CollisionWorld, createCircleBody, createBoxBody } from './collision/CollisionWorld';
import type { RigidBody, ConvexShape } from './collision/CollisionWorld';

// ── SSE endpoint ─────────────────────────────────────────────────────────────

const SSE_URL       = '/api/cell-events';
const CELLS_API_URL = '/api/cells';

// ── Backoff constants (mirrors CellEventSource.ts) ───────────────────────────

const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS     = 30_000;
const ERROR_THRESHOLD    = 3;

// ── Public types ─────────────────────────────────────────────────────────────

/** Slim bbox extracted from /api/cells response (subset of CellParamsJson). */
export interface CellDescriptor {
  cell_id:    string;
  bbox:       { x: number; y: number; w: number; h: number };
  /** Optional z-layer — used only for logging, not mapped to domain. */
  z?:         number;
  /** Arbitrary extra fields ignored by the bridge. */
  [key: string]: unknown;
}

/** Topology payload from `topology_updated` SSE event. */
export interface TopologyPayload {
  topology: {
    children?: Array<{ id: string; [k: string]: unknown }>;
    edges?:    Array<{ sources?: string[]; targets?: string[]; [k: string]: unknown }>;
    [k: string]: unknown;
  };
}

/** Epoch payload from `epoch_completed` SSE event. */
export interface EpochPayload {
  epoch:   number;
  [key: string]: unknown;
}

/** cell_params_updated payload. */
export interface CellParamsUpdatedPayload {
  cell_id: string;
  params:  CellDescriptor['bbox'] extends infer B
    ? { bbox?: { x: number; y: number; w: number; h: number }; [k: string]: unknown }
    : never;
}

/** Options for domain ↔ canvas coordinate conversion. */
export interface SPHEpochBridgeOptions {
  /** SPH domain width in metres (default: 3.0). */
  domainW?: number;
  /** SPH domain height in metres (default: 3.0). */
  domainH?: number;
  /** Canvas pixel width used for coordinate normalisation (default: 800). */
  canvasW?: number;
  /** Canvas pixel height used for coordinate normalisation (default: 600). */
  canvasH?: number;
  /**
   * Radius scale applied after coordinate normalisation.
   * Increase to make cells repel fluid more strongly (default: 1.0).
   */
  radiusScale?: number;
}

// ── SPHEpochBridge ────────────────────────────────────────────────────────────

export class SPHEpochBridge {
  // ── SPH world reference ──────────────────────────────────────────────────
  private readonly _world: SPHWorld;

  /**
   * CollisionWorld extracted from the SPHWorld via getSceneQuery().
   *
   * SPHWorld exposes `getSceneQuery()` publicly but not `collisionWorld` itself.
   * We therefore maintain our own parallel CollisionWorld reference by casting
   * through the private accessor.  If SPHWorld gains a `getCollisionWorld()`
   * method in a future refactor, replace the cast below.
   */
  private readonly _collision: CollisionWorld;

  // ── Coordinate mapping ────────────────────────────────────────────────────
  private readonly _domainW: number;
  private readonly _domainH: number;
  private readonly _canvasW: number;
  private readonly _canvasH: number;
  private readonly _radiusScale: number;

  // ── Cell ↔ body registry ──────────────────────────────────────────────────
  /**
   * Maps cell_id → rigid body id so we can move or remove bodies without
   * scanning the full CollisionWorld.
   */
  private _bodyIds: Map<string, number> = new Map();

  // ── SSE lifecycle ──────────────────────────────────────────────────────────
  private _es:             EventSource | null = null;
  private _ownsEventSource = false;
  private _externalEs:     EventSource | null = null;
  private _started    = false;
  private _stopped    = false;

  // ── Reconnect / backoff ──────────────────────────────────────────────────
  private _errorCount     = 0;
  private _backoffMs      = BACKOFF_INITIAL_MS;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // ── In-flight fetch guard (avoids parallel /api/cells requests) ──────────
  private _fetchingCells = false;

  // ── Constructor ───────────────────────────────────────────────────────────

  /**
   * @param world         The SPHWorld whose CollisionWorld will be updated.
   * @param opts          Optional coordinate mapping overrides.
   * @param eventSource   Optional pre-existing EventSource to attach to.
   *                      When provided the bridge will NOT close it on stop().
   */
  constructor(
    world:        SPHWorld,
    opts:         SPHEpochBridgeOptions = {},
    eventSource?: EventSource,
  ) {
    this._world       = world;
    this._domainW     = opts.domainW    ?? 3.0;
    this._domainH     = opts.domainH    ?? 3.0;
    this._canvasW     = opts.canvasW    ?? 800;
    this._canvasH     = opts.canvasH    ?? 600;
    this._radiusScale = opts.radiusScale ?? 1.0;

    // Extract the CollisionWorld via the private field cast.
    // SPHWorld's collisionWorld field is private; we use a type-cast to access it
    // here because the bridge needs to manipulate bodies directly.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this._collision = (world as any)._collisionWorld as CollisionWorld
      ?? (world as any).collisionWorld as CollisionWorld;

    if (!this._collision) {
      throw new Error(
        '[SPHEpochBridge] Could not access CollisionWorld from SPHWorld. ' +
        'Ensure SPHWorld.init() has been called before constructing the bridge.',
      );
    }

    if (eventSource) {
      this._externalEs    = eventSource;
      this._ownsEventSource = false;
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Begin listening to SSE events.
   * Safe to call once — subsequent calls are no-ops until stop() is called.
   */
  start(): void {
    if (this._stopped) throw new Error('[SPHEpochBridge] Cannot restart a stopped bridge');
    if (this._started) return;
    this._started = true;
    this._openEventSource();
  }

  /**
   * Stop listening to SSE events and clean up all tracked rigid bodies.
   * The bridge cannot be restarted after stop() is called.
   */
  stop(): void {
    if (!this._started || this._stopped) return;
    this._stopped = true;
    this._clearReconnect();
    this._closeEventSource();
    this._removeAllBodies();
  }

  /** Number of rigid bodies currently tracked by the bridge. */
  get trackedBodyCount(): number {
    return this._bodyIds.size;
  }

  /** Read-only snapshot of the cell_id → body_id registry. */
  get bodyRegistry(): ReadonlyMap<string, number> {
    return this._bodyIds;
  }

  // ── SSE lifecycle ─────────────────────────────────────────────────────────

  private _openEventSource(): void {
    if (typeof EventSource === 'undefined') {
      console.warn('[SPHEpochBridge] EventSource not available — SSE disabled');
      return;
    }

    // Use external EventSource when provided, otherwise open our own.
    if (this._externalEs) {
      this._attachHandlers(this._externalEs);
      return;
    }

    const es = new EventSource(SSE_URL);
    this._es             = es;
    this._ownsEventSource = true;
    this._attachHandlers(es);
  }

  private _attachHandlers(es: EventSource): void {
    // ── epoch_completed ────────────────────────────────────────────────────
    es.addEventListener('epoch_completed', (ev: MessageEvent) => {
      this._onEpochCompleted(ev);
    });

    // ── cell_params_updated ────────────────────────────────────────────────
    es.addEventListener('cell_params_updated', (ev: MessageEvent) => {
      this._onCellParamsUpdated(ev);
    });

    // ── topology_updated ───────────────────────────────────────────────────
    es.addEventListener('topology_updated', (ev: MessageEvent) => {
      this._onTopologyUpdated(ev);
    });

    // ── open / error (only for owned EventSources) ─────────────────────────
    if (this._ownsEventSource) {
      es.onopen = () => {
        this._errorCount = 0;
        this._backoffMs  = BACKOFF_INITIAL_MS;
        console.info('[SPHEpochBridge] SSE connected →', SSE_URL);
      };

      es.onerror = () => {
        this._errorCount++;
        if (this._stopped) return;

        if (this._errorCount >= ERROR_THRESHOLD) {
          this._closeEventSource();
          const delay = Math.min(this._backoffMs, BACKOFF_MAX_MS);
          console.warn(
            `[SPHEpochBridge] SSE error #${this._errorCount} — reconnecting in ${delay}ms`,
          );
          this._reconnectTimer = setTimeout(() => {
            if (!this._stopped) {
              this._backoffMs = Math.min(this._backoffMs * 2, BACKOFF_MAX_MS);
              this._openEventSource();
            }
          }, delay);
        }
      };
    }
  }

  private _closeEventSource(): void {
    if (this._es) {
      this._es.close();
      this._es = null;
    }
    // Never close an externally-owned EventSource.
  }

  private _clearReconnect(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  // ── SSE event handlers ────────────────────────────────────────────────────

  /**
   * epoch_completed  →  fetch /api/cells and upsert rigid body positions.
   *
   * We deliberately throttle with a simple in-flight guard: if a previous
   * /api/cells fetch is still pending (e.g. slow server during a burst of
   * epoch completions) we skip the current event rather than pile up requests.
   */
  private _onEpochCompleted(ev: MessageEvent): void {
    let payload: EpochPayload;
    try {
      payload = JSON.parse(ev.data) as EpochPayload;
    } catch {
      console.error('[SPHEpochBridge] Failed to parse epoch_completed payload:', ev.data);
      return;
    }

    const epoch = payload.epoch ?? '?';
    console.debug(`[SPHEpochBridge] epoch_completed epoch=${epoch} — syncing rigid bodies`);

    if (this._fetchingCells) {
      console.debug('[SPHEpochBridge] /api/cells fetch already in flight — skipping');
      return;
    }

    this._fetchingCells = true;
    fetch(CELLS_API_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`/api/cells returned ${res.status}`);
        return res.json() as Promise<CellDescriptor[]>;
      })
      .then((descriptors) => {
        this._upsertBodies(descriptors);
      })
      .catch((err) => {
        console.error('[SPHEpochBridge] Failed to fetch /api/cells:', err);
      })
      .finally(() => {
        this._fetchingCells = false;
      });
  }

  /**
   * cell_params_updated  →  fast-path single-body position update.
   *
   * Avoids a full /api/cells round-trip when only one cell's params changed.
   * Falls back to creating a new body if the cell is not yet tracked.
   */
  private _onCellParamsUpdated(ev: MessageEvent): void {
    let payload: { cell_id: string; params: { bbox?: { x: number; y: number; w: number; h: number } } };
    try {
      payload = JSON.parse(ev.data);
    } catch {
      console.error('[SPHEpochBridge] Failed to parse cell_params_updated payload:', ev.data);
      return;
    }

    const { cell_id, params } = payload;
    if (!params?.bbox) return;

    const bbox = params.bbox;
    const { domX, domY, domR } = this._bboxToDomain(bbox);

    const existingBodyId = this._bodyIds.get(cell_id);

    if (existingBodyId !== undefined) {
      // Move existing kinematic body to new domain position.
      try {
        const body = this._collision.getBody(existingBodyId);
        body.position.x = domX;
        body.position.y = domY;
        console.debug(
          `[SPHEpochBridge] cell_params_updated: moved "${cell_id}" → (${domX.toFixed(3)}, ${domY.toFixed(3)})`,
        );
      } catch (e) {
        // Body may have been removed by a concurrent topology_updated — re-add it.
        console.warn(`[SPHEpochBridge] getBody(${existingBodyId}) failed, re-adding "${cell_id}":`, e);
        this._bodyIds.delete(cell_id);
        this._addBody(cell_id, domX, domY, domR);
      }
    } else {
      // First time we hear about this cell — create a body for it.
      this._addBody(cell_id, domX, domY, domR);
    }
  }

  /**
   * topology_updated  →  full teardown + rebuild.
   *
   * The topology may describe a completely different cell set (cells added or
   * removed), so we wipe all tracked bodies and re-add them from the payload's
   * children list.  Because the topology payload does not carry bbox data we
   * schedule a /api/cells fetch immediately after clearing.
   */
  private _onTopologyUpdated(ev: MessageEvent): void {
    let payload: TopologyPayload;
    try {
      payload = JSON.parse(ev.data) as TopologyPayload;
    } catch {
      console.error('[SPHEpochBridge] Failed to parse topology_updated payload:', ev.data);
      return;
    }

    const childCount = payload.topology?.children?.length ?? 0;
    console.info(
      `[SPHEpochBridge] topology_updated — rebuilding ${childCount} rigid bodies`,
    );

    // ── 1. Remove all existing tracked bodies ─────────────────────────────
    this._removeAllBodies();

    // ── 2. Re-fetch /api/cells to get fresh bbox data for the new topology ─
    if (this._fetchingCells) return;
    this._fetchingCells = true;
    fetch(CELLS_API_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`/api/cells returned ${res.status}`);
        return res.json() as Promise<CellDescriptor[]>;
      })
      .then((descriptors) => {
        this._upsertBodies(descriptors);
        console.info(
          `[SPHEpochBridge] topology rebuild complete — ${this._bodyIds.size} bodies registered`,
        );
      })
      .catch((err) => {
        console.error('[SPHEpochBridge] Failed to fetch /api/cells after topology_updated:', err);
      })
      .finally(() => {
        this._fetchingCells = false;
      });
  }

  // ── Rigid body management ─────────────────────────────────────────────────

  /**
   * Upsert (create or reposition) a rigid body for each cell descriptor.
   *
   * Existing bodies are repositioned in-place; new cells get a fresh kinematic
   * circle body registered with the CollisionWorld.
   */
  private _upsertBodies(descriptors: CellDescriptor[]): void {
    for (const desc of descriptors) {
      const { cell_id, bbox } = desc;
      if (!bbox) continue;

      const { domX, domY, domR } = this._bboxToDomain(bbox);
      const existingBodyId = this._bodyIds.get(cell_id);

      if (existingBodyId !== undefined) {
        // ── Update existing body position ──────────────────────────────────
        try {
          const body = this._collision.getBody(existingBodyId);
          body.position.x = domX;
          body.position.y = domY;
        } catch {
          // Body disappeared (e.g. after removeAllBodies race) — recreate it.
          this._bodyIds.delete(cell_id);
          this._addBody(cell_id, domX, domY, domR);
        }
      } else {
        // ── Create new body ────────────────────────────────────────────────
        this._addBody(cell_id, domX, domY, domR);
      }
    }
  }

  /**
   * Create a kinematic circle body at (domX, domY) with radius domR and
   * register it in the CollisionWorld + local registry.
   *
   * Kinematic bodies are moved by the bridge (not by the physics solver) so the
   * SPH fluid is repelled but cell positions remain authoritative from the backend.
   */
  private _addBody(cell_id: string, domX: number, domY: number, domR: number): void {
    const { body, shape } = createCircleBody(domX, domY, domR, 'kinematic', {
      restitution: 0.2,
      friction:    0.4,
    });
    const id = this._collision.addBody(body, shape);
    this._bodyIds.set(cell_id, id);
    console.debug(
      `[SPHEpochBridge] added body id=${id} for "${cell_id}" ` +
      `@ (${domX.toFixed(3)}, ${domY.toFixed(3)}) r=${domR.toFixed(4)}`,
    );
  }

  /**
   * Remove all bodies currently tracked by this bridge from the CollisionWorld
   * and clear the local registry.
   */
  private _removeAllBodies(): void {
    for (const [cell_id, bodyId] of this._bodyIds) {
      try {
        this._collision.removeBody(bodyId);
      } catch (e) {
        console.warn(`[SPHEpochBridge] removeBody(${bodyId}) for "${cell_id}" failed:`, e);
      }
    }
    this._bodyIds.clear();
    console.debug('[SPHEpochBridge] all tracked bodies removed');
  }

  // ── Coordinate helpers ────────────────────────────────────────────────────

  /**
   * Convert a canvas-space bbox to domain-space centre (domX, domY) and
   * approximate circle radius (domR).
   *
   * Canvas: origin top-left, Y-down, pixels.
   * Domain: origin bottom-left, Y-up, metres (domainW × domainH).
   */
  private _bboxToDomain(
    bbox: { x: number; y: number; w: number; h: number },
  ): { domX: number; domY: number; domR: number } {
    const { _domainW, _domainH, _canvasW, _canvasH, _radiusScale } = this;

    // Centre of the bbox in canvas pixels (Y-down).
    const pxCx = bbox.x + bbox.w * 0.5;
    const pxCy = bbox.y + bbox.h * 0.5;

    // Normalise to [0, 1] then scale to domain.
    const domX = (pxCx / _canvasW) * _domainW;
    // Flip Y: canvas Y=0 is top, domain Y=0 is bottom.
    const domY = (1.0 - pxCy / _canvasH) * _domainH;

    // Radius: use the smaller of the two half-extents, then normalise + scale.
    const pxR  = Math.min(bbox.w, bbox.h) * 0.5;
    const domR = (pxR / Math.min(_canvasW, _canvasH)) * Math.min(_domainW, _domainH) * _radiusScale;

    return { domX, domY, domR: Math.max(domR, 0.01) }; // floor at 1 cm
  }
}
