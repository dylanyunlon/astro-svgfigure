/**
 * topology-physics-sync.ts — M756
 *
 * Bridges SSE `topology_updated` events to the live physics world by
 * performing incremental rigid body add/remove/update and full edge
 * emitter rebuild.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * When the backend writes a new `channels/skeleton/topology.json` the server
 * fires a `topology_updated` SSE event carrying the full ELK topology graph
 * (nodes as `children`, edges as `edges`).  This module receives that event
 * and reconciles the physics world in three steps:
 *
 *   1. **Diff nodes** — compare incoming topology children against the
 *      currently tracked rigid bodies.  Nodes that disappeared are removed;
 *      new nodes are created; surviving nodes have their bbox and species
 *      updated in-place (no destroy+recreate for position-only changes).
 *
 *   2. **Rigid body CRUD** — delegates to `cell-body-bridge.cellsToBodies()`
 *      for species→physics property resolution, then calls `createRigidBody()`
 *      / `sphBridge.removeBody()` / position update on the SPH bridge.
 *
 *   3. **Edge emitter rebuild** — tears down all existing edge emitters and
 *      rebuilds them from the new topology's edge list.  Each edge becomes
 *      an `EmitterConfig` positioned at the source node's bbox exit point
 *      with direction toward the target node.  Skip-connection edges use a
 *      higher base rate and the `TOPO_CHANGE` QoS profile.
 *
 * The module also processes the per-edge `route.json` control-point data
 * (fetched from `/api/edges/{edgeId}/route`) to rebuild `FlowEdge` configs
 * for the `EdgeFlowRenderer` when available.
 *
 * ── Integration ──────────────────────────────────────────────────────────────
 *
 *   import { TopologyPhysicsSync } from '$lib/sph/topology-physics-sync';
 *
 *   const sync = new TopologyPhysicsSync(sphWorld, edgeFlowRenderer, {
 *     domainW: 3.0,
 *     domainH: 3.0,
 *     canvasW: canvas.width,
 *     canvasH: canvas.height,
 *   });
 *   sync.start();          // opens SSE connection
 *   // ...later:
 *   sync.stop();           // closes SSE, no-op afterward
 *
 * ── SSE event schema ─────────────────────────────────────────────────────────
 *
 *   event: topology_updated
 *   data: {
 *     "topology": {
 *       "id": "root",
 *       "layoutOptions": { ... },
 *       "children": [
 *         { "id": "input_embed", "width": 160, "height": 50,
 *           "labels": [{"text": "Input Embedding"}] },
 *         ...
 *       ],
 *       "edges": [
 *         { "id": "e1", "sources": ["input_embed"], "targets": ["pos_encode"] },
 *         { "id": "skip1", "sources": ["pos_encode"], "targets": ["add_norm1"],
 *           "advanced": { "semanticType": "skip_connection", ... } },
 *         ...
 *       ]
 *     }
 *   }
 *
 * ── References ───────────────────────────────────────────────────────────────
 *   src/lib/sph/sph-epoch-bridge.ts    — predecessor: epoch/cell SSE → bodies
 *   src/lib/sph/cell-body-bridge.ts    — species → physics property resolver
 *   src/lib/sph/edge-flow-renderer.ts  — FlowEdge / EdgeFlowRenderer types
 *   src/lib/sph/emitter-strategy.ts    — EmitterConfig / EmissionPattern types
 *   src/lib/sph/rigid-body.ts          — RigidBody / createRigidBody factory
 *   channels/skeleton/topology.json    — canonical ELK topology schema
 *   channels/physics/edge_routes.json  — resolved edge spline control points
 */


import {
} from './cell-body-bridge';
import type { FlowEdge }            from './edge-flow-renderer';
import type { EmitterConfig }        from './emitter-strategy';
import { ContinuousPattern }         from './emitter-strategy';
import {
} from './rigid-body';

  type CellPhysicsConfig,
  cellsToBodies,
  speciesToIndex,

  createRigidBody,
  type RigidBody,
  type RigidBodyOptions,

// ─── Constants ────────────────────────────────────────────────────────────────

const SSE_URL       = '/api/cell-events';
const CELLS_API_URL = '/api/cells';
const EDGE_ROUTES_API_URL = '/api/edges';

/** Backoff for SSE reconnect (mirrors sph-epoch-bridge). */
const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS     = 30_000;
const ERROR_THRESHOLD    = 3;

/** Default emitter rate for normal edges (particles/s). */
const EDGE_EMITTER_RATE_NORMAL = 8;
/** Elevated emitter rate for skip-connection / special edges. */
const EDGE_EMITTER_RATE_SKIP   = 14;
/** Default edge flow weight (controls particle density). */
const EDGE_FLOW_WEIGHT_NORMAL  = 1.0;
const EDGE_FLOW_WEIGHT_SKIP    = 1.6;

// ─── Public types ─────────────────────────────────────────────────────────────

/** Topology node as received in the SSE payload. */
export interface TopoNode {
  id:        string;
  width?:    number;
  height?:   number;
  labels?:   Array<{ text: string }>;
  children?: TopoNode[];
  group?:    boolean;
  [k: string]: unknown;
}

/** Topology edge as received in the SSE payload. */
export interface TopoEdge {
  id:       string;
  sources?: string[];
  targets?: string[];
  advanced?: {
    semanticType?: string;
    routing?:      string;
    curvature?:    number;
    [k: string]: unknown;
  };
  [k: string]: unknown;
}

/** Full topology payload from the `topology_updated` SSE event. */
export interface TopologyPayload {
  topology: {
    id?:            string;
    layoutOptions?: Record<string, string>;
    children?:      TopoNode[];
    edges?:         TopoEdge[];
    [k: string]: unknown;
  };
}

/** Edge route control points (from physics/edge_routes.json or API). */
export interface EdgeRoute {
  edge_id:   string;
  sources:   string[];
  targets:   string[];
  is_skip:   boolean;
  advanced:  Record<string, unknown>;
  points:    Array<{ x: number; y: number }>;
  z?:        number;
}

/** Configuration for TopologyPhysicsSync. */
export interface TopologyPhysicsSyncOptions {
  /** SPH domain width in metres (default: 3.0). */
  domainW?: number;
  /** SPH domain height in metres (default: 3.0). */
  domainH?: number;
  /** Canvas pixel width for coordinate mapping (default: 800). */
  canvasW?: number;
  /** Canvas pixel height for coordinate mapping (default: 600). */
  canvasH?: number;
  /** Radius scale factor for rigid body sizing (default: 1.0). */
  radiusScale?: number;
  /**
   * When true, fetch `/api/edges/{id}/route` for spline points instead of
   * using straight-line fallback (default: true).
   */
  fetchRoutes?: boolean;
}

/**
 * Delegate interface for the SPH world / collision world.
 *
 * This keeps TopologyPhysicsSync decoupled from a specific SPH implementation.
 * The consuming code must pass an object that satisfies this contract.
 */
export interface PhysicsWorldDelegate {
  /** Create a rigid body at the given config. Returns a numeric body handle. */
  addBody(
    id: string, x: number, y: number, w: number, h: number,
    species: number, pinned: boolean,
  ): number;
  /** Remove a rigid body by its numeric handle. */
  removeBody(handle: number): void;
  /** Move an existing body to a new position. */
  moveBody(handle: number, x: number, y: number): void;
  /** Resize an existing body (half-extents). */
  resizeBody?(handle: number, w: number, h: number): void;
}

/**
 * Delegate interface for the EdgeFlowRenderer (or equivalent).
 *
 * Allows the sync module to rebuild edge emitters without importing the
 * full renderer.
 */
export interface EdgeFlowDelegate {
  /** Replace all edge definitions and rebuild internal particle pools. */
  setEdges(edges: FlowEdge[]): void;
}

/** Callback fired after a successful topology sync. */
export type OnSyncCallback = (stats: SyncStats) => void;

/** Diagnostic stats emitted after each sync cycle. */
export interface SyncStats {
  /** Number of bodies added in this sync. */
  bodiesAdded:   number;
  /** Number of bodies removed in this sync. */
  bodiesRemoved: number;
  /** Number of bodies repositioned/resized in this sync. */
  bodiesUpdated: number;
  /** Number of edge emitters rebuilt. */
  edgesRebuilt:  number;
  /** Wall-clock duration of the sync in milliseconds. */
  durationMs:    number;
}

// ─── TopologyPhysicsSync ──────────────────────────────────────────────────────

export class TopologyPhysicsSync {
  // ── Dependencies ────────────────────────────────────────────────────────
  private readonly _physics: PhysicsWorldDelegate;
  private readonly _edgeFlow: EdgeFlowDelegate | null;

  // ── Coordinate mapping ──────────────────────────────────────────────────
  private readonly _domainW:     number;
  private readonly _domainH:     number;
  private readonly _canvasW:     number;
  private readonly _canvasH:     number;
  private readonly _radiusScale: number;
  private readonly _fetchRoutes: boolean;

  // ── State tracking ──────────────────────────────────────────────────────

  /**
   * Maps cell_id → { bodyHandle, bbox, species } for all currently
   * tracked rigid bodies.  This is the single source of truth for what
   * the physics world currently contains.
   */
  private _tracked: Map<string, TrackedBody> = new Map();

  /** Last-known topology snapshot (for external queries / debug). */
  private _lastTopology: TopologyPayload['topology'] | null = null;

  /** Last-known edge list (for rebuild diffing). */
  private _lastEdges: TopoEdge[] = [];

  // ── SSE lifecycle ───────────────────────────────────────────────────────
  private _es:                EventSource | null = null;
  private _externalEs:        EventSource | null = null;
  private _ownsEventSource    = false;
  private _started            = false;
  private _stopped            = false;

  // ── Reconnect / backoff ─────────────────────────────────────────────────
  private _errorCount          = 0;
  private _backoffMs           = BACKOFF_INITIAL_MS;
  private _reconnectTimer:     ReturnType<typeof setTimeout> | null = null;

  // ── Fetch guard ─────────────────────────────────────────────────────────
  private _syncing = false;

  // ── Callbacks ───────────────────────────────────────────────────────────
  private _onSync: OnSyncCallback | null = null;

  // ── Constructor ─────────────────────────────────────────────────────────

  constructor(
    physics:      PhysicsWorldDelegate,
    edgeFlow:     EdgeFlowDelegate | null,
    opts:         TopologyPhysicsSyncOptions = {},
    eventSource?: EventSource,
  ) {
    this._physics     = physics;
    this._edgeFlow    = edgeFlow;
    this._domainW     = opts.domainW     ?? 3.0;
    this._domainH     = opts.domainH     ?? 3.0;
    this._canvasW     = opts.canvasW     ?? 800;
    this._canvasH     = opts.canvasH     ?? 600;
    this._radiusScale = opts.radiusScale ?? 1.0;
    this._fetchRoutes = opts.fetchRoutes ?? true;

    if (eventSource) {
      this._externalEs     = eventSource;
      this._ownsEventSource = false;
    }
  }

  // ─── Public API ─────────────────────────────────────────────────────────

  /** Register a callback that fires after each successful sync. */
  onSync(cb: OnSyncCallback): void {
    this._onSync = cb;
  }

  /** Start listening for SSE topology_updated events. */
  start(): void {
    if (this._started || this._stopped) return;
    this._started = true;
    this._connect();
  }

  /** Stop listening and clean up. */
  stop(): void {
    if (this._stopped) return;
    this._stopped = true;
    this._started = false;

    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }

    if (this._es && this._ownsEventSource) {
      this._es.close();
    }
    this._es = null;
  }

  /** Number of currently tracked rigid bodies. */
  get bodyCount(): number {
    return this._tracked.size;
  }

  /** Read-only snapshot of tracked cell IDs. */
  get trackedCellIds(): string[] {
    return Array.from(this._tracked.keys());
  }

  /** Last-known topology (may be null before first event). */
  get lastTopology(): TopologyPayload['topology'] | null {
    return this._lastTopology;
  }

  /**
   * Manually trigger a sync from an already-parsed topology payload.
   * Useful for initial bootstrapping or testing without SSE.
   */
  async syncFromPayload(payload: TopologyPayload): Promise<SyncStats> {
    return this._handleTopologyUpdate(payload);
  }

  // ─── SSE connection management ──────────────────────────────────────────

  private _connect(): void {
    if (this._stopped) return;

    const es = this._externalEs ?? new EventSource(SSE_URL);
    this._es = es;
    this._ownsEventSource = !this._externalEs;

    es.addEventListener('topology_updated', this._onTopologyUpdated);

    es.addEventListener('open', () => {
      this._errorCount = 0;
      this._backoffMs  = BACKOFF_INITIAL_MS;
      console.debug('[TopologyPhysicsSync] SSE connected');
    });

    es.addEventListener('error', () => {
      this._errorCount++;
      if (this._errorCount >= ERROR_THRESHOLD && this._ownsEventSource) {
        es.close();
        this._scheduleReconnect();
      }
    });
  }

  private _scheduleReconnect(): void {
    if (this._stopped || this._reconnectTimer !== null) return;

    console.debug(
      `[TopologyPhysicsSync] reconnecting in ${this._backoffMs}ms`,
    );

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._backoffMs = Math.min(this._backoffMs * 2, BACKOFF_MAX_MS);
      this._connect();
    }, this._backoffMs);
  }

  // ─── SSE event handler ──────────────────────────────────────────────────

  private _onTopologyUpdated = (ev: MessageEvent): void => {
    let payload: TopologyPayload;
    try {
      payload = JSON.parse(ev.data) as TopologyPayload;
    } catch {
      console.error(
        '[TopologyPhysicsSync] Failed to parse topology_updated payload:',
        ev.data,
      );
      return;
    }

    // Fire-and-forget — errors are logged internally.
    this._handleTopologyUpdate(payload).catch((err) => {
      console.error('[TopologyPhysicsSync] sync failed:', err);
    });
  };

  // ─── Core sync logic ───────────────────────────────────────────────────

  private async _handleTopologyUpdate(
    payload: TopologyPayload,
  ): Promise<SyncStats> {
    if (this._syncing) {
      console.debug('[TopologyPhysicsSync] sync already in progress — skipping');
      return { bodiesAdded: 0, bodiesRemoved: 0, bodiesUpdated: 0, edgesRebuilt: 0, durationMs: 0 };
    }

    this._syncing = true;
    const t0 = performance.now();

    try {
      const topology = payload.topology;
      this._lastTopology = topology;

      const incomingNodes = this._flattenNodes(topology.children ?? []);
      const incomingEdges = topology.edges ?? [];
      this._lastEdges = incomingEdges;

      // ── 1. Diff rigid bodies ─────────────────────────────────────────────

      const incomingIds  = new Set(incomingNodes.map((n) => n.id));
      const currentIds   = new Set(this._tracked.keys());

      // Nodes to remove: in current but not in incoming
      const toRemove: string[] = [];
      for (const id of currentIds) {
        if (!incomingIds.has(id)) toRemove.push(id);
      }

      // Nodes to add: in incoming but not in current
      const toAdd: TopoNode[] = [];
      // Nodes to update: in both
      const toUpdate: TopoNode[] = [];

      for (const node of incomingNodes) {
        if (currentIds.has(node.id)) {
          toUpdate.push(node);
        } else {
          toAdd.push(node);
        }
      }

      // ── 2. Remove stale bodies ───────────────────────────────────────────

      for (const cellId of toRemove) {
        const entry = this._tracked.get(cellId);
        if (entry) {
          try {
            this._physics.removeBody(entry.bodyHandle);
          } catch (e) {
            console.warn(
              `[TopologyPhysicsSync] removeBody(${entry.bodyHandle}) ` +
              `for "${cellId}" failed:`, e,
            );
          }
          this._tracked.delete(cellId);
        }
      }

      // ── 3. Fetch fresh bbox data from /api/cells ─────────────────────────
      //    The topology payload carries node dimensions but not absolute positions
      //    resolved by the ELK layouter.  We fetch the authoritative cell_registry
      //    data from the backend.

      let cellDescriptors: CellDescriptor[] = [];
      try {
        const res = await fetch(CELLS_API_URL);
        if (res.ok) {
          cellDescriptors = await res.json();
        } else {
          console.warn(
            `[TopologyPhysicsSync] /api/cells returned ${res.status}, ` +
            'falling back to topology dimensions',
          );
        }
      } catch (err) {
        console.warn(
          '[TopologyPhysicsSync] /api/cells fetch failed, falling back:',
          err,
        );
      }

      // Build a lookup from cell_id → descriptor for O(1) access
      const descMap = new Map<string, CellDescriptor>();
      for (const desc of cellDescriptors) {
        descMap.set(desc.cell_id ?? desc.id, desc);
      }

      // ── 4. Add new bodies ────────────────────────────────────────────────

      let bodiesAdded = 0;
      for (const node of toAdd) {
        const desc  = descMap.get(node.id);
        const bbox  = this._resolveBbox(node, desc);
        const species = this._resolveSpecies(node, desc);
        const domain  = this._bboxToDomain(bbox);
        const pinned  = species === 'cil-plus';

        const handle = this._physics.addBody(
          node.id,
          domain.domX,
          domain.domY,
          domain.domW,
          domain.domH,
          speciesToIndex(species),
          pinned,
        );

        this._tracked.set(node.id, {
          bodyHandle: handle,
          bbox,
          species,
        });

        bodiesAdded++;
        console.debug(
          `[TopologyPhysicsSync] + body "${node.id}" handle=${handle} ` +
          `species=${species} @ (${domain.domX.toFixed(3)}, ${domain.domY.toFixed(3)})`,
        );
      }

      // ── 5. Update surviving bodies ───────────────────────────────────────

      let bodiesUpdated = 0;
      for (const node of toUpdate) {
        const entry   = this._tracked.get(node.id)!;
        const desc    = descMap.get(node.id);
        const newBbox = this._resolveBbox(node, desc);
        const newSpecies = this._resolveSpecies(node, desc);
        const domain  = this._bboxToDomain(newBbox);

        const bboxChanged =
          newBbox.x !== entry.bbox.x ||
          newBbox.y !== entry.bbox.y ||
          newBbox.w !== entry.bbox.w ||
          newBbox.h !== entry.bbox.h;

        const speciesChanged = newSpecies !== entry.species;

        if (bboxChanged || speciesChanged) {
          // Position update
          this._physics.moveBody(entry.bodyHandle, domain.domX, domain.domY);

          // Size update (if delegate supports it)
          if (bboxChanged && this._physics.resizeBody) {
            this._physics.resizeBody(entry.bodyHandle, domain.domW, domain.domH);
          }

          // If species changed we need to remove + re-add (species affects mass, friction, etc.)
          if (speciesChanged) {
            try {
              this._physics.removeBody(entry.bodyHandle);
            } catch { /* swallow */ }

            const pinned  = newSpecies === 'cil-plus';
            const handle  = this._physics.addBody(
              node.id,
              domain.domX,
              domain.domY,
              domain.domW,
              domain.domH,
              speciesToIndex(newSpecies),
              pinned,
            );
            entry.bodyHandle = handle;
          }

          entry.bbox    = newBbox;
          entry.species = newSpecies;
          bodiesUpdated++;
        }
      }

      // ── 6. Rebuild edge emitters ─────────────────────────────────────────

      let edgesRebuilt = 0;
      if (this._edgeFlow) {
        const flowEdges = await this._buildFlowEdges(incomingEdges, descMap);
        this._edgeFlow.setEdges(flowEdges);
        edgesRebuilt = flowEdges.length;
      }

      // ── 7. Emit stats ────────────────────────────────────────────────────

      const durationMs = performance.now() - t0;
      const stats: SyncStats = {
        bodiesAdded,
        bodiesRemoved: toRemove.length,
        bodiesUpdated,
        edgesRebuilt,
        durationMs,
      };

      console.info(
        `[TopologyPhysicsSync] sync complete: ` +
        `+${bodiesAdded} −${toRemove.length} ~${bodiesUpdated} bodies, ` +
        `${edgesRebuilt} edges — ${durationMs.toFixed(1)}ms`,
      );

      this._onSync?.(stats);
      return stats;
    } finally {
      this._syncing = false;
    }
  }

  // ─── Node utilities ─────────────────────────────────────────────────────

  /**
   * Recursively flatten nested compound nodes into a flat array.
   * Compound containers themselves are included (they may be group bodies).
   */
  private _flattenNodes(nodes: TopoNode[]): TopoNode[] {
    const result: TopoNode[] = [];
    const walk = (list: TopoNode[]): void => {
      for (const node of list) {
        result.push(node);
        if (node.children?.length) {
          walk(node.children);
        }
      }
    };
    walk(nodes);
    return result;
  }

  /**
   * Resolve the authoritative bbox for a node, preferring the /api/cells
   * descriptor (which has ELK-layouted absolute coordinates) over the
   * raw topology dimensions.
   */
  private _resolveBbox(
    node: TopoNode,
    desc?: CellDescriptor,
  ): { x: number; y: number; w: number; h: number } {
    if (desc?.bbox) {
      // Backend bbox may be in min/max or x/y/w/h format
      const b = desc.bbox as Record<string, unknown>;
      if (Array.isArray(b.min) && Array.isArray(b.max)) {
        const min = b.min as number[];
        const max = b.max as number[];
        return {
          x: min[0],
          y: min[1],
          w: Math.abs(max[0] - min[0]),
          h: Math.abs(max[1] - min[1]),
        };
      }
      if (typeof b.x === 'number') {
        return {
          x: b.x as number,
          y: b.y as number,
          w: b.w as number,
          h: b.h as number,
        };
      }
    }

    // Fallback: use topology node dimensions with synthetic Y offset
    return {
      x: 220,
      y: 40,
      w: node.width  ?? 140,
      h: node.height ?? 50,
    };
  }

  /** Resolve species for a node, preferring backend data. */
  private _resolveSpecies(
    node: TopoNode,
    desc?: CellDescriptor,
  ): string {
    if (desc?.species && typeof desc.species === 'string') {
      return desc.species;
    }
    // Infer from label using the same heuristic as topology_to_skeleton.py
    const label = node.labels?.[0]?.text ?? node.id;
    return inferSpecies(label);
  }

  // ─── Coordinate mapping ─────────────────────────────────────────────────

  /**
   * Convert a canvas-pixel bbox to domain-space centre + half-extents.
   *
   * Canvas: origin top-left, Y-down, pixels.
   * Domain: origin bottom-left, Y-up, metres.
   */
  private _bboxToDomain(
    bbox: { x: number; y: number; w: number; h: number },
  ): { domX: number; domY: number; domW: number; domH: number; domR: number } {
    const { _domainW, _domainH, _canvasW, _canvasH, _radiusScale } = this;

    const pxCx = bbox.x + bbox.w * 0.5;
    const pxCy = bbox.y + bbox.h * 0.5;

    const domX = (pxCx / _canvasW) * _domainW;
    const domY = (1.0 - pxCy / _canvasH) * _domainH;

    const domW = (bbox.w / _canvasW) * _domainW * 0.5 * _radiusScale;
    const domH = (bbox.h / _canvasH) * _domainH * 0.5 * _radiusScale;

    const pxR  = Math.min(bbox.w, bbox.h) * 0.5;
    const domR = (pxR / Math.min(_canvasW, _canvasH)) *
                 Math.min(_domainW, _domainH) * _radiusScale;

    return { domX, domY, domW, domH, domR: Math.max(domR, 0.01) };
  }

  // ─── Edge emitter building ──────────────────────────────────────────────

  /**
   * Build FlowEdge configs from topology edges, resolving control points
   * from edge route data when available.
   */
  private async _buildFlowEdges(
    edges: TopoEdge[],
    cellLookup: Map<string, CellDescriptor>,
  ): Promise<FlowEdge[]> {
    // Optionally fetch route data for spline points
    let routeMap = new Map<string, EdgeRoute>();
    if (this._fetchRoutes) {
      routeMap = await this._fetchEdgeRoutes(edges);
    }

    const flowEdges: FlowEdge[] = [];

    for (const edge of edges) {
      const sourceId = edge.sources?.[0];
      const targetId = edge.targets?.[0];
      if (!sourceId || !targetId) continue;

      const isSkip = edge.advanced?.semanticType === 'skip_connection';

      // Resolve control points
      const route = routeMap.get(edge.id);
      let points: Array<{ x: number; y: number }>;

      if (route?.points?.length && route.points.length >= 2) {
        // Use pre-computed spline control points from edge_routes
        points = route.points;
      } else {
        // Fallback: straight line from source bbox bottom-centre to
        // target bbox top-centre
        points = this._straightLinePoints(sourceId, targetId, cellLookup);
      }

      flowEdges.push({
        edgeId:   edge.id,
        sourceId,
        targetId,
        points,
        weight: isSkip ? EDGE_FLOW_WEIGHT_SKIP : EDGE_FLOW_WEIGHT_NORMAL,
        qos:    isSkip ? 'TOPO_CHANGE' : 'DEFAULT',
      });
    }

    return flowEdges;
  }

  /**
   * Fetch edge route data from the backend for spline control points.
   * Falls back gracefully if the endpoint is unavailable.
   */
  private async _fetchEdgeRoutes(
    edges: TopoEdge[],
  ): Promise<Map<string, EdgeRoute>> {
    const result = new Map<string, EdgeRoute>();

    try {
      // Try bulk endpoint first (channels/physics/edge_routes.json)
      const res = await fetch('/api/edge-routes');
      if (res.ok) {
        const data = await res.json() as Record<string, EdgeRoute>;
        for (const [id, route] of Object.entries(data)) {
          result.set(id, route);
        }
        return result;
      }
    } catch {
      // Bulk endpoint unavailable — try individual fetches
    }

    // Individual fetch fallback (parallel, with timeout)
    const TIMEOUT_MS = 2000;
    const fetches = edges.map(async (edge) => {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
        const res = await fetch(
          `${EDGE_ROUTES_API_URL}/${edge.id}/route`,
          { signal: controller.signal },
        );
        clearTimeout(timer);
        if (res.ok) {
          const route = await res.json() as EdgeRoute;
          result.set(edge.id, route);
        }
      } catch {
        // Individual route not available — will use straight-line fallback
      }
    });

    await Promise.allSettled(fetches);
    return result;
  }

  /**
   * Generate a straight-line 2-point path from the source node's bottom
   * centre to the target node's top centre.
   */
  private _straightLinePoints(
    sourceId: string,
    targetId: string,
    cellLookup: Map<string, CellDescriptor>,
  ): Array<{ x: number; y: number }> {
    const srcDesc = cellLookup.get(sourceId);
    const tgtDesc = cellLookup.get(targetId);

    const srcEntry = this._tracked.get(sourceId);
    const tgtEntry = this._tracked.get(targetId);

    // Use tracked bbox if available, else descriptor, else default
    const srcBbox = srcEntry?.bbox ?? this._descToBbox(srcDesc);
    const tgtBbox = tgtEntry?.bbox ?? this._descToBbox(tgtDesc);

    return [
      { x: srcBbox.x + srcBbox.w * 0.5, y: srcBbox.y + srcBbox.h },
      { x: tgtBbox.x + tgtBbox.w * 0.5, y: tgtBbox.y },
    ];
  }

  /** Extract bbox from a CellDescriptor, handling both formats. */
  private _descToBbox(
    desc?: CellDescriptor,
  ): { x: number; y: number; w: number; h: number } {
    if (!desc?.bbox) return { x: 220, y: 40, w: 140, h: 50 };

    const b = desc.bbox as Record<string, unknown>;
    if (Array.isArray(b.min) && Array.isArray(b.max)) {
      const min = b.min as number[];
      const max = b.max as number[];
      return {
        x: min[0],
        y: min[1],
        w: Math.abs(max[0] - min[0]),
        h: Math.abs(max[1] - min[1]),
      };
    }
    return desc.bbox as { x: number; y: number; w: number; h: number };
  }
}

// ─── Tracked body entry ───────────────────────────────────────────────────────

interface TrackedBody {
  bodyHandle: number;
  bbox:       { x: number; y: number; w: number; h: number };
  species:    string;
}

// ─── CellDescriptor (matches /api/cells response) ────────────────────────────

interface CellDescriptor {
  cell_id?: string;
  id?:      string;
  bbox:     Record<string, unknown>;
  species?: string;
  z?:       number;
  [key: string]: unknown;
}

// ─── Species inference (mirrors topology_to_skeleton.py) ──────────────────────

const SPECIES_RULES: Array<[RegExp, string]> = [
  [/attn|attention/i,             'cil-eye'],
  [/conv|filter|kernel/i,         'cil-filter'],
  [/norm|bn|batch/i,              'cil-plus'],
  [/embed|encod|vector|mu|sigma/i,'cil-vector'],
  [/output|decode|x_hat/i,        'cil-arrow-right'],
  [/input/i,                      'cil-vector'],
  [/add|\+|residual/i,            'cil-plus'],
  [/relu|activ|gelu|swish/i,      'cil-bolt'],
  [/sample|reparame/i,            'cil-loop'],
  [/ffn|feed|forward|mlp/i,       'cil-bolt'],
  [/pool|downsample/i,            'cil-layers'],
  [/graph|net/i,                   'cil-graph'],
];

function inferSpecies(label: string): string {
  for (const [pattern, species] of SPECIES_RULES) {
    if (pattern.test(label)) return species;
  }
  return 'cil-code';
}

// ─── Emitter config builder (for external consumers) ──────────────────────────

/**
 * Build EmitterConfig array from topology edges.  Useful when the consuming
 * code needs raw emitter configs instead of FlowEdge objects (e.g. for the
 * SPH world-stepper particle emitter system).
 *
 * Each edge produces one emitter positioned at the source node's bbox exit
 * point, aimed toward the target node.
 */
export function buildEdgeEmitters(
  edges: TopoEdge[],
  tracked: Map<string, { bbox: { x: number; y: number; w: number; h: number } }>,
): EmitterConfig[] {
  const emitters: EmitterConfig[] = [];

  for (const edge of edges) {
    const sourceId = edge.sources?.[0];
    const targetId = edge.targets?.[0];
    if (!sourceId || !targetId) continue;

    const srcEntry = tracked.get(sourceId);
    const tgtEntry = tracked.get(targetId);
    if (!srcEntry || !tgtEntry) continue;

    const srcBbox = srcEntry.bbox;
    const tgtBbox = tgtEntry.bbox;

    // Emitter position: bottom-centre of source bbox
    const ex = srcBbox.x + srcBbox.w * 0.5;
    const ey = srcBbox.y + srcBbox.h;

    // Direction: toward target top-centre
    const tx = tgtBbox.x + tgtBbox.w * 0.5;
    const ty = tgtBbox.y;
    const dx = tx - ex;
    const dy = ty - ey;
    const len = Math.sqrt(dx * dx + dy * dy);
    const dirX = len > 1e-6 ? dx / len : 0;
    const dirY = len > 1e-6 ? dy / len : 1;

    const isSkip = edge.advanced?.semanticType === 'skip_connection';

    emitters.push({
      x:       ex,
      y:       ey,
      dirX,
      dirY,
      rate:    isSkip ? EDGE_EMITTER_RATE_SKIP : EDGE_EMITTER_RATE_NORMAL,
      species: 0, // fluid species for edge particles
      label:   `edge:${edge.id}`,
      pattern: new ContinuousPattern(),
    });
  }

  return emitters;
}
