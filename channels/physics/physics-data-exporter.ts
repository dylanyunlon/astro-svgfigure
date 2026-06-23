/**
 * physics-data-exporter.ts — M788: Aggregated physics channel data exporter
 *
 * Reads all JSON data files in channels/physics/ and produces a single typed
 * PhysicsSnapshot that downstream consumers (rendering pipeline, debug HUD,
 * analytics dashboards, pixi-export) can import without manually wiring up
 * individual channel reads.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design notes
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * • **Channel-file convention** — each JSON file in channels/physics/ is a
 *   "channel" in the Apollo CyberRT sense (see channel_runtime.py).  This
 *   module treats them as read-only data sources: it never writes back.
 *
 * • **Typed interfaces** — every channel file has a corresponding TypeScript
 *   interface.  The top-level `PhysicsSnapshot` aggregates all of them into a
 *   single bag that can be serialized, diffed, or forwarded to a WebWorker.
 *
 * • **Fetch vs fs** — the module exposes two loading strategies:
 *   1. `fetchPhysicsSnapshot()` — browser-side, uses fetch() against the
 *      Astro API routes that serve the channel files.
 *   2. `loadPhysicsSnapshotFromDisk()` — Node/CLI-side, reads the files
 *      directly from the filesystem (used by tests and offline tools).
 *
 * • **Partial loading** — callers can request a subset of channels via the
 *   `PhysicsExportOptions.channels` filter.  Channels not listed are omitted
 *   from the snapshot (their fields are `null`).
 *
 * • **Staleness detection** — each loaded channel records its mtime (disk) or
 *   Last-Modified header (fetch).  `isStale()` compares a snapshot against
 *   the current mtimes without re-reading the data.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   channels/channel_runtime.py        — subscribe() / publish() / on_message()
 *   channels/physics/qos_spatial.py    — QoS → physics parameter mapping
 *   src/lib/physics-bridge.ts          — WebWorker bridge (CellBBox, ForceField)
 *   src/lib/sph/epoch-physics-recorder.ts — epoch snapshot recorder (M768)
 *   src/lib/sph/physics-uniform-bridge.ts — per-body uniform sampler
 *   src/lib/renderers/pixi-export.ts   — CompositeParams / export pipeline
 *
 * Usage (browser):
 *   import { fetchPhysicsSnapshot } from 'channels/physics/physics-data-exporter';
 *   const snap = await fetchPhysicsSnapshot();
 *   console.log(snap.forceField);            // { add_norm2: { dx, dy, dz }, ... }
 *   console.log(snap.cellRegistry.cells);    // { self_attn: { bbox, species, ... }, ... }
 *
 * Usage (Node / CLI):
 *   import { loadPhysicsSnapshotFromDisk } from 'channels/physics/physics-data-exporter';
 *   const snap = await loadPhysicsSnapshotFromDisk('/path/to/channels/physics');
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Channel data interfaces
// ═══════════════════════════════════════════════════════════════════════════════

// ─── species_physics.json ─────────────────────────────────────────────────────

/** Per-species physical material properties. */
export interface SpeciesPhysicsEntry {
  mass: number;
  friction: number;
  restitution: number;
  buoyancy: number;
}

/** species_physics.json — keyed by species ID (e.g. "cil-eye"). */
export type SpeciesPhysicsMap = Record<string, SpeciesPhysicsEntry>;

// ─── force_field.json ─────────────────────────────────────────────────────────

/** Per-cell force vector from the physics engine. */
export interface ForceVector {
  dx: number;
  dy: number;
  dz: number;
  push_from?: string[];
  push_mag?: number;
}

/** force_field.json — keyed by cell ID (e.g. "self_attn"). */
export type ForceFieldMap = Record<string, ForceVector>;

// ─── cell_registry.json ───────────────────────────────────────────────────────

/** Bounding box in [x, y, z] min/max format. */
export interface CellBBox {
  min: [number, number, number];
  max: [number, number, number];
}

/** Single cell entry in the registry. */
export interface CellRegistryEntry {
  bbox: CellBBox;
  species: string;
  z: number;
  constraint_mask: number;
  epoch: number;
}

/** cell_registry.json top-level structure. */
export interface CellRegistryData {
  cells: Record<string, CellRegistryEntry>;
  z_layers: Record<string, string[]>;
}

// ─── edge_routes.json ─────────────────────────────────────────────────────────

/** Routing metadata from the M169 crossing minimiser. */
export interface EdgeM169 {
  crossings_before: number;
  crossings_after: number;
}

/** Edge route advanced properties (skip connections, spline routing). */
export interface EdgeAdvanced {
  semanticType?: string;
  routing?: string;
  curvature?: number;
  [key: string]: unknown;
}

/** Single edge route entry. */
export interface EdgeRouteEntry {
  edge_id: string;
  sources: string[];
  targets: string[];
  is_skip: boolean;
  advanced: EdgeAdvanced;
  points: Array<{ x: number; y: number }>;
  blocked_by: string[];
  m169: EdgeM169;
}

/** edge_routes.json — keyed by edge ID (e.g. "e1", "skip1"). */
export type EdgeRoutesMap = Record<string, EdgeRouteEntry>;

// ─── cell_groups.json ─────────────────────────────────────────────────────────

/** Community group from the Louvain + betweenness algorithm. */
export interface CellGroup {
  community_id: number;
  cells: string[];
  centroid: { x: number; y: number };
  focal_cell: string;
  is_focal_group: boolean;
  betweenness_scores: Record<string, number>;
}

/** cell_groups.json top-level structure. */
export interface CellGroupsData {
  algorithm: string;
  n_communities: number;
  groups: Record<string, CellGroup>;
  betweenness: Record<string, number>;
  crossings_before: number;
  crossings_after: number;
}

// ─── collision.json ───────────────────────────────────────────────────────────

/** Collision detection result (currently sparse). */
export interface CollisionData {
  collisions: unknown[];
  count: number;
}

// ─── converged.json ───────────────────────────────────────────────────────────

/** Convergence state of the physics loop. */
export interface ConvergenceData {
  converged: boolean;
  epoch: number;
  conflicts: number;
}

// ─── z_layers.json ────────────────────────────────────────────────────────────

/** Z-layer voxel entry with occupancy statistics. */
export interface ZLayerEntry {
  count: number;
  weight: number;
  density: number;
}

/** z_layers.json — keyed by "col,row,z" voxel coordinates. */
export type ZLayerMap = Record<string, ZLayerEntry>;

// ─── species_assignment.json ──────────────────────────────────────────────────

/** Gene traits for a cell's species assignment. */
export interface GeneTraits {
  primary_shape: string;
  [key: string]: unknown;
}

/** Single cell species assignment entry. */
export interface SpeciesAssignmentEntry {
  species: string;
  gene_traits: GeneTraits;
  [key: string]: unknown;
}

/** species_assignment.json — keyed by cell ID. */
export type SpeciesAssignmentMap = Record<string, SpeciesAssignmentEntry>;

// ─── species_visual_traits.json ───────────────────────────────────────────────

/** Visual traits per species for rendering. */
export interface SpeciesVisualTraitsEntry {
  _role?: string;
  [key: string]: unknown;
}

/** species_visual_traits.json — keyed by species ID. */
export type SpeciesVisualTraitsMap = Record<string, SpeciesVisualTraitsEntry>;

// ═══════════════════════════════════════════════════════════════════════════════
// Aggregated snapshot
// ═══════════════════════════════════════════════════════════════════════════════

/** Names of all exportable physics channels. */
export type PhysicsChannelName =
  | 'speciesPhysics'
  | 'forceField'
  | 'cellRegistry'
  | 'edgeRoutes'
  | 'cellGroups'
  | 'collision'
  | 'converged'
  | 'zLayers'
  | 'speciesAssignment'
  | 'speciesVisualTraits';

/** Per-channel metadata attached to each loaded channel. */
export interface ChannelMeta {
  /** File path (relative to channels/physics/) or API URL. */
  source: string;
  /** Modification timestamp in ms (mtime on disk, Last-Modified via fetch). */
  mtime: number;
  /** Size in bytes of the raw JSON source (0 if unknown). */
  byteLength: number;
}

/**
 * Complete aggregated physics snapshot.
 *
 * Each channel field is `null` when the channel was not loaded (filtered out
 * by `PhysicsExportOptions.channels` or missing from disk / API).
 */
export interface PhysicsSnapshot {
  /** species_physics.json — per-species mass, friction, restitution, buoyancy. */
  speciesPhysics: SpeciesPhysicsMap | null;
  /** force_field.json — per-cell force vectors from the physics engine. */
  forceField: ForceFieldMap | null;
  /** cell_registry.json — cell bboxes, species, z-layer assignments. */
  cellRegistry: CellRegistryData | null;
  /** edge_routes.json — resolved edge spline control points + M169 metadata. */
  edgeRoutes: EdgeRoutesMap | null;
  /** cell_groups.json — Louvain community detection + betweenness centrality. */
  cellGroups: CellGroupsData | null;
  /** collision.json — collision detection results. */
  collision: CollisionData | null;
  /** converged.json — convergence state of the epoch loop. */
  converged: ConvergenceData | null;
  /** z_layers.json — voxelised z-layer occupancy grid. */
  zLayers: ZLayerMap | null;
  /** species_assignment.json — per-cell species + gene traits. */
  speciesAssignment: SpeciesAssignmentMap | null;
  /** species_visual_traits.json — per-species visual rendering traits. */
  speciesVisualTraits: SpeciesVisualTraitsMap | null;

  /** Per-channel load metadata. */
  meta: Partial<Record<PhysicsChannelName, ChannelMeta>>;
  /** Wall-clock timestamp (ms) when the snapshot was assembled. */
  timestamp: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Channel → filename mapping
// ═══════════════════════════════════════════════════════════════════════════════

/** Maps each channel name to its JSON filename in channels/physics/. */
const CHANNEL_FILES: Record<PhysicsChannelName, string> = {
  speciesPhysics:      'species_physics.json',
  forceField:          'force_field.json',
  cellRegistry:        'cell_registry.json',
  edgeRoutes:          'edge_routes.json',
  cellGroups:          'cell_groups.json',
  collision:           'collision.json',
  converged:           'converged.json',
  zLayers:             'z_layers.json',
  speciesAssignment:   'species_assignment.json',
  speciesVisualTraits: 'species_visual_traits.json',
};

/** All channel names in load order. */
const ALL_CHANNELS: PhysicsChannelName[] = Object.keys(
  CHANNEL_FILES,
) as PhysicsChannelName[];

// ═══════════════════════════════════════════════════════════════════════════════
// Export options
// ═══════════════════════════════════════════════════════════════════════════════

/** Options for controlling which channels are loaded and how. */
export interface PhysicsExportOptions {
  /**
   * Subset of channels to load.  When omitted or empty, all channels are
   * loaded.  Channels not listed will have their snapshot fields set to `null`.
   */
  channels?: PhysicsChannelName[];
  /**
   * Base URL for API fetch (browser mode).  Defaults to '/api/physics'.
   * Each channel is fetched from `${baseUrl}/${filename}`.
   */
  baseUrl?: string;
  /**
   * When true, channel-level fetch/read errors are silently swallowed and
   * the corresponding snapshot field is set to `null`.  When false (default),
   * the first error rejects the returned Promise.
   */
  lenient?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Browser-side loader (fetch)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Load a physics snapshot by fetching channel JSON files from the Astro API.
 *
 * Designed for browser / client-side use.  Each channel file is fetched in
 * parallel from `${baseUrl}/${filename}`.
 *
 * @param opts  Export options controlling channel selection and error handling.
 * @returns     A fully populated `PhysicsSnapshot`.
 *
 * @example
 *   const snap = await fetchPhysicsSnapshot();
 *   if (snap.converged?.converged) {
 *     console.log(`Layout converged at epoch ${snap.converged.epoch}`);
 *   }
 */
export async function fetchPhysicsSnapshot(
  opts: PhysicsExportOptions = {},
): Promise<PhysicsSnapshot> {
  const baseUrl  = opts.baseUrl ?? '/api/physics';
  const lenient  = opts.lenient ?? false;
  const channels = opts.channels?.length ? opts.channels : ALL_CHANNELS;

  const snapshot = _emptySnapshot();

  const fetches = channels.map(async (ch) => {
    const filename = CHANNEL_FILES[ch];
    const url = `${baseUrl}/${filename}`;

    try {
      const res = await fetch(url);
      if (!res.ok) {
        if (lenient) return;
        throw new Error(
          `[physics-data-exporter] fetch ${url} returned ${res.status}`,
        );
      }

      const text = await res.text();
      const data = JSON.parse(text);

      // Parse Last-Modified header for staleness tracking
      const lastMod = res.headers.get('Last-Modified');
      const mtime = lastMod ? new Date(lastMod).getTime() : Date.now();

      _assignChannel(snapshot, ch, data);
      snapshot.meta[ch] = {
        source: url,
        mtime,
        byteLength: new TextEncoder().encode(text).byteLength,
      };
    } catch (err) {
      if (lenient) {
        console.warn(
          `[physics-data-exporter] failed to load channel "${ch}":`,
          err,
        );
        return;
      }
      throw err;
    }
  });

  await Promise.all(fetches);
  snapshot.timestamp = Date.now();
  return snapshot;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Node / CLI-side loader (filesystem)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Load a physics snapshot by reading channel JSON files directly from disk.
 *
 * Designed for Node.js / CLI use (tests, offline analytics, snapshot_manager).
 * Uses dynamic `import('node:fs/promises')` so the module remains importable
 * in browser bundles (the fs import is tree-shaken when unused).
 *
 * @param physicsDir  Absolute path to `channels/physics/` directory.
 * @param opts        Export options controlling channel selection and error handling.
 * @returns           A fully populated `PhysicsSnapshot`.
 *
 * @example
 *   const snap = await loadPhysicsSnapshotFromDisk('/repo/channels/physics');
 *   for (const [cellId, entry] of Object.entries(snap.cellRegistry?.cells ?? {})) {
 *     console.log(cellId, entry.species, entry.bbox);
 *   }
 */
export async function loadPhysicsSnapshotFromDisk(
  physicsDir: string,
  opts: PhysicsExportOptions = {},
): Promise<PhysicsSnapshot> {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');

  const lenient  = opts.lenient ?? false;
  const channels = opts.channels?.length ? opts.channels : ALL_CHANNELS;

  const snapshot = _emptySnapshot();

  const reads = channels.map(async (ch) => {
    const filename = CHANNEL_FILES[ch];
    const filePath = path.join(physicsDir, filename);

    try {
      const stat = await fs.stat(filePath);
      const text = await fs.readFile(filePath, 'utf-8');
      const data = JSON.parse(text);

      _assignChannel(snapshot, ch, data);
      snapshot.meta[ch] = {
        source: filePath,
        mtime: stat.mtimeMs,
        byteLength: stat.size,
      };
    } catch (err) {
      if (lenient) {
        console.warn(
          `[physics-data-exporter] failed to read channel "${ch}" ` +
          `from ${filePath}:`,
          err,
        );
        return;
      }
      throw err;
    }
  });

  await Promise.all(reads);
  snapshot.timestamp = Date.now();
  return snapshot;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Staleness detection
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Check whether any channel in a snapshot is stale relative to the current
 * file mtimes on disk.
 *
 * @param snapshot    A previously loaded `PhysicsSnapshot`.
 * @param physicsDir  Absolute path to `channels/physics/` directory.
 * @returns           Array of channel names whose files have been modified
 *                    since the snapshot was taken.  Empty array = fresh.
 */
export async function detectStaleChannels(
  snapshot: PhysicsSnapshot,
  physicsDir: string,
): Promise<PhysicsChannelName[]> {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');

  const stale: PhysicsChannelName[] = [];

  for (const ch of ALL_CHANNELS) {
    const meta = snapshot.meta[ch];
    if (!meta) continue;

    const filePath = path.join(physicsDir, CHANNEL_FILES[ch]);
    try {
      const stat = await fs.stat(filePath);
      if (stat.mtimeMs > meta.mtime) {
        stale.push(ch);
      }
    } catch {
      // File disappeared — treat as stale
      stale.push(ch);
    }
  }

  return stale;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Derived accessors — convenience helpers over the raw snapshot
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Extract a flat array of cell IDs from the snapshot.
 * Reads from cell_registry first, falls back to force_field keys.
 */
export function cellIdsFromSnapshot(snapshot: PhysicsSnapshot): string[] {
  if (snapshot.cellRegistry?.cells) {
    return Object.keys(snapshot.cellRegistry.cells);
  }
  if (snapshot.forceField) {
    return Object.keys(snapshot.forceField);
  }
  return [];
}

/**
 * Look up the resolved physics properties for a specific cell, combining
 * cell_registry (species) → species_physics (material) → force_field (forces).
 *
 * Returns `null` if the cell is not found in the registry.
 */
export function resolvedCellPhysics(
  snapshot: PhysicsSnapshot,
  cellId: string,
): {
  cellId: string;
  species: string;
  bbox: CellBBox;
  material: SpeciesPhysicsEntry | null;
  force: ForceVector | null;
  community: number | null;
  betweenness: number | null;
} | null {
  const entry = snapshot.cellRegistry?.cells?.[cellId];
  if (!entry) return null;

  const material =
    snapshot.speciesPhysics?.[entry.species] ?? null;
  const force =
    snapshot.forceField?.[cellId] ?? null;

  // Find community membership
  let community: number | null = null;
  let betweenness: number | null = null;
  if (snapshot.cellGroups) {
    betweenness = snapshot.cellGroups.betweenness?.[cellId] ?? null;
    for (const group of Object.values(snapshot.cellGroups.groups)) {
      if (group.cells.includes(cellId)) {
        community = group.community_id;
        break;
      }
    }
  }

  return {
    cellId,
    species: entry.species,
    bbox: entry.bbox,
    material,
    force,
    community,
    betweenness,
  };
}

/**
 * Compute aggregate statistics across all cells in the snapshot.
 * Useful for debug HUDs and analytics dashboards.
 */
export function snapshotStats(snapshot: PhysicsSnapshot): {
  cellCount: number;
  edgeCount: number;
  skipEdgeCount: number;
  communityCount: number;
  converged: boolean;
  epoch: number;
  totalForceMagnitude: number;
  avgMass: number;
  channelsLoaded: number;
  totalBytes: number;
} {
  const cells = snapshot.cellRegistry?.cells ?? {};
  const cellCount = Object.keys(cells).length;

  const edges = snapshot.edgeRoutes ?? {};
  const edgeEntries = Object.values(edges);
  const edgeCount = edgeEntries.length;
  const skipEdgeCount = edgeEntries.filter((e) => e.is_skip).length;

  const communityCount = snapshot.cellGroups?.n_communities ?? 0;
  const converged = snapshot.converged?.converged ?? false;
  const epoch = snapshot.converged?.epoch ?? 0;

  // Total force magnitude across all cells
  let totalForceMagnitude = 0;
  if (snapshot.forceField) {
    for (const fv of Object.values(snapshot.forceField)) {
      totalForceMagnitude += Math.sqrt(
        fv.dx * fv.dx + fv.dy * fv.dy + fv.dz * fv.dz,
      );
    }
  }

  // Average mass across species present in the registry
  let avgMass = 0;
  if (snapshot.speciesPhysics && cellCount > 0) {
    const speciesSeen = new Set<string>();
    for (const c of Object.values(cells)) {
      speciesSeen.add(c.species);
    }
    let massSum = 0;
    let count = 0;
    for (const sp of speciesSeen) {
      const entry = snapshot.speciesPhysics[sp];
      if (entry) {
        massSum += entry.mass;
        count++;
      }
    }
    avgMass = count > 0 ? massSum / count : 0;
  }

  const channelsLoaded = Object.keys(snapshot.meta).length;
  let totalBytes = 0;
  for (const m of Object.values(snapshot.meta)) {
    if (m) totalBytes += m.byteLength;
  }

  return {
    cellCount,
    edgeCount,
    skipEdgeCount,
    communityCount,
    converged,
    epoch,
    totalForceMagnitude,
    avgMass,
    channelsLoaded,
    totalBytes,
  };
}

/**
 * Serialize a snapshot to a JSON string.
 * Strips ChannelMeta.source paths to avoid leaking local filesystem paths
 * in exported data.
 */
export function serializeSnapshot(snapshot: PhysicsSnapshot): string {
  const sanitised = {
    ...snapshot,
    meta: Object.fromEntries(
      Object.entries(snapshot.meta).map(([k, v]) => [
        k,
        v ? { mtime: v.mtime, byteLength: v.byteLength } : null,
      ]),
    ),
  };
  return JSON.stringify(sanitised, null, 2);
}

/**
 * Deserialize a snapshot from a JSON string produced by `serializeSnapshot`.
 */
export function deserializeSnapshot(json: string): PhysicsSnapshot {
  return JSON.parse(json) as PhysicsSnapshot;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/** Create an empty snapshot with all channels null. */
function _emptySnapshot(): PhysicsSnapshot {
  return {
    speciesPhysics:      null,
    forceField:          null,
    cellRegistry:        null,
    edgeRoutes:          null,
    cellGroups:          null,
    collision:           null,
    converged:           null,
    zLayers:             null,
    speciesAssignment:   null,
    speciesVisualTraits: null,
    meta:                {},
    timestamp:           0,
  };
}

/** Assign parsed JSON data to the correct snapshot field. */
function _assignChannel(
  snapshot: PhysicsSnapshot,
  channel: PhysicsChannelName,
  data: unknown,
): void {
  switch (channel) {
    case 'speciesPhysics':
      snapshot.speciesPhysics = data as SpeciesPhysicsMap;
      break;
    case 'forceField':
      snapshot.forceField = data as ForceFieldMap;
      break;
    case 'cellRegistry':
      snapshot.cellRegistry = data as CellRegistryData;
      break;
    case 'edgeRoutes':
      snapshot.edgeRoutes = data as EdgeRoutesMap;
      break;
    case 'cellGroups':
      snapshot.cellGroups = data as CellGroupsData;
      break;
    case 'collision':
      snapshot.collision = data as CollisionData;
      break;
    case 'converged':
      snapshot.converged = data as ConvergenceData;
      break;
    case 'zLayers':
      snapshot.zLayers = data as ZLayerMap;
      break;
    case 'speciesAssignment':
      snapshot.speciesAssignment = data as SpeciesAssignmentMap;
      break;
    case 'speciesVisualTraits':
      snapshot.speciesVisualTraits = data as SpeciesVisualTraitsMap;
      break;
  }
}
