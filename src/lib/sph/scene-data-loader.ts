/**
 * scene-data-loader.ts
 * M964: composite_params.json → GPU CellData[] / EdgeData[]
 *
 * SceneDataLoader reads channels/composite_params.json (and sibling files)
 * and produces the typed arrays that the GPU render loop expects.
 *
 * Data sources (all relative to the project / public root):
 *   channels/composite_params.json          — cells + edge_routes + species_assignment
 *   channels/physics/edge_routes.json       — authoritative Bézier control-points
 *   channels/cell/<id>/bbox.json            — per-cell epoch-9 positions & species
 *   channels/cell/<id>/params.json          — per-cell human label
 *
 * I/O strategy:
 *   • Browser  → fetch() (Vite dev-server serves channels/ from project root)
 *   • Node.js  → fs.readFileSync()  (used in REPL validation scripts)
 *   Both paths are covered in loadScene() via runtime environment detection.
 */

// ─── GPU types — aligned with gpu-render-loop.ts ─────────────────────────────

export interface CellData {
  /** Unique cell identifier matching composite_params.cells key */
  cell_id: string;
  /** Species tag: cil-eye | cil-bolt | cil-vector | cil-plus | cil-arrow-right … */
  species: string;
  /** Bounding-box (scene coords, pixels) */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Z-layer depth (render order; higher = front) */
  z: number;
  /** PBR metallic factor   [0, 1] */
  metallic: number;
  /** PBR roughness factor  [0, 1] */
  roughness: number;
  /** PBR albedo RGB        [0, 1] each channel */
  albedo: [number, number, number];
  /** Human-readable label from params.json (falls back to cell_id) */
  label: string;
}

export interface EdgeData {
  /** Unique edge identifier: e1 … e6, skip1, skip2 */
  edge_id: string;
  /** Source cell id */
  source: string;
  /** Target cell id */
  target: string;
  /** Cubic Bézier / polyline control-points [[x,y], …] */
  controlPoints: [number, number][];
  /** Wire color RGB [0,1] derived from source-cell species */
  color: [number, number, number];
}

// ─── Species → PBR material & wire-color ─────────────────────────────────────
//
// Derived from species_params.primary_color values in composite_params.json
// cross-referenced with the AT options table:
//
//   cil-eye          #1E88E5  → blue   (self-attention)
//   cil-bolt         #F57C00  → orange (feed-forward / activation)
//   cil-vector       #546E7A  → steel-blue (embedding)
//   cil-plus         #E53935  → red    (add-norm / residual)
//   cil-arrow-right  #2E7D32  → green  (output)
//   cil-sine         #7E57C2  → purple (positional encoding)
//   (fallback)                → mid-grey

interface SpeciesMaterial {
  metallic: number;
  roughness: number;
  albedo: [number, number, number];
  wireColor: [number, number, number];
}

const SPECIES_MATERIAL: Record<string, SpeciesMaterial> = {
  // attention — vibrant blue, dielectric (glassy)
  'cil-eye': {
    metallic:  0.04,
    roughness: 0.60,
    albedo:    [0.118, 0.533, 0.898],
    wireColor: [0.118, 0.533, 0.898],
  },
  // feed-forward bolt — vivid orange, high metallic (energetic)
  'cil-bolt': {
    metallic:  0.80,
    roughness: 0.30,
    albedo:    [1.000, 0.435, 0.000],
    wireColor: [1.000, 0.435, 0.000],
  },
  // embedding vector — steel-blue, moderate metallic
  'cil-vector': {
    metallic:  0.15,
    roughness: 0.50,
    albedo:    [0.329, 0.431, 0.478],
    wireColor: [0.329, 0.431, 0.478],
  },
  // add-norm plus — muted red, low metallic (rocky residual)
  'cil-plus': {
    metallic:  0.10,
    roughness: 0.55,
    albedo:    [0.776, 0.157, 0.157],
    wireColor: [0.776, 0.157, 0.157],
  },
  // output arrow — forest green, semi-metallic
  'cil-arrow-right': {
    metallic:  0.30,
    roughness: 0.45,
    albedo:    [0.180, 0.490, 0.196],
    wireColor: [0.180, 0.490, 0.196],
  },
  // positional-sine (some epoch files tag it as cil-sine)
  'cil-sine': {
    metallic:  0.12,
    roughness: 0.52,
    albedo:    [0.494, 0.341, 0.761],
    wireColor: [0.494, 0.341, 0.761],
  },
};

const FALLBACK_MATERIAL: SpeciesMaterial = {
  metallic:  0.20,
  roughness: 0.60,
  albedo:    [0.500, 0.500, 0.500],
  wireColor: [0.500, 0.500, 0.500],
};

function speciesMaterial(species: string): SpeciesMaterial {
  return SPECIES_MATERIAL[species] ?? FALLBACK_MATERIAL;
}

// ─── Raw JSON shapes (subset we actually use) ────────────────────────────────

interface RawBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  species?: string;
  epoch?: number;
}

interface RawCellEntry {
  agent_params?: {
    bbox?: RawBbox;
    opacity?: number;
    species_params?: { primary_color?: string };
  };
  at_params?: {
    cell_id?: string;
    role?: string;
  };
}

interface RawEdgeRoute {
  source: string;
  target: string;
  type?: string;
  control_points: [number, number][];
}

interface RawCompositeParams {
  cells?: Record<string, RawCellEntry>;
  edges?: Record<string, unknown>;
  edge_routes?: Record<string, RawEdgeRoute>;
  species_assignment?: Record<string, { species: string }>;
}

interface RawBboxFile {
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  species?: string;
  epoch?: number;
}

interface RawParamsFile {
  cell_id?: string;
  label?: string;
  species?: string;
}

// ─── I/O helpers — browser (fetch) vs Node.js (readFileSync) ─────────────────

/** Detect if running under Node.js (REPL / test) rather than a browser. */
function isNode(): boolean {
  return typeof process !== 'undefined'
    && typeof process.versions !== 'undefined'
    && typeof process.versions.node !== 'undefined';
}

/**
 * Load JSON from a path/URL.
 * - Browser: uses fetch()
 * - Node.js: uses fs.readFileSync()
 *
 * Returns null (does not throw) on 404 / missing file so callers can
 * treat absent optional files as graceful fallbacks.
 */
async function loadJson<T>(pathOrUrl: string): Promise<T | null> {
  if (isNode()) {
    // Node path: synchronous read wrapped in async for uniform interface
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const fs = require('fs') as typeof import('fs');
      // Resolve relative to cwd (matches Node REPL invoked from project root)
      const raw = fs.readFileSync(pathOrUrl, 'utf8');
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  } else {
    // Browser path: fetch()
    try {
      const res = await fetch(pathOrUrl);
      if (!res.ok) return null;
      return (await res.json()) as T;
    } catch {
      return null;
    }
  }
}

// ─── SceneDataLoader ─────────────────────────────────────────────────────────

export class SceneDataLoader {

  // ── public API ──────────────────────────────────────────────────────────────

  /**
   * High-level convenience entry point.
   *
   * Fetches all required JSON files and returns CellData[] + EdgeData[]
   * ready to be passed to GPURenderLoop.setScene().
   *
   * @param baseDir  Path/URL prefix for channel assets.
   *                 Browser default: '' (Vite serves channels/ from root)
   *                 Node REPL default: '' (relative to cwd, i.e. project root)
   */
  async loadScene(baseDir = ''): Promise<{ cells: CellData[]; edges: EdgeData[] }> {
    const sep = baseDir.endsWith('/') ? '' : (baseDir ? '/' : '');
    const base = baseDir + sep;

    // ── 1. Load composite_params.json (required) ──────────────────────────
    const composite = await loadJson<RawCompositeParams>(
      `${base}channels/composite_params.json`,
    );
    if (!composite) {
      throw new Error('[SceneDataLoader] Failed to load channels/composite_params.json');
    }

    // ── 2. Load physics/edge_routes.json (authoritative, optional) ────────
    const physicsRoutes = await loadJson<Record<string, RawEdgeRoute>>(
      `${base}channels/physics/edge_routes.json`,
    );

    // ── 3. Load per-cell bbox.json + params.json ──────────────────────────
    const cellIds = Object.keys(composite.cells ?? {});
    const [bboxOverrides, labelOverrides] = await this._fetchCellFiles(cellIds, base);

    // ── 4. Build scene data ───────────────────────────────────────────────
    const cells = this._parseCells(composite, bboxOverrides, labelOverrides);
    const edges = this._parseEdges(composite, physicsRoutes, cells);

    return { cells, edges };
  }

  /**
   * Parse a pre-loaded composite_params JSON object.
   *
   * Useful when you already have the JSON in memory (e.g. unit tests,
   * or when the caller controls the fetch lifecycle).
   *
   * Priority order for bbox:
   *   1. bboxOverrides[id]              (channels/cell/<id>/bbox.json — epoch-9)
   *   2. composite.cells[id].agent_params.bbox  (epoch-2 fallback)
   *   3. Default {x:0, y:0, w:100, h:50, z:1}
   *
   * Priority order for species:
   *   1. bboxOverrides[id].species
   *   2. composite.species_assignment[id].species
   *   3. 'unknown'
   *
   * Priority order for edge control-points:
   *   1. externalRoutes (channels/physics/edge_routes.json)
   *   2. composite.edge_routes
   */
  loadFromCompositeParams(
    json: RawCompositeParams,
    bboxOverrides: Record<string, RawBboxFile> = {},
    labelOverrides: Record<string, string> = {},
    externalRoutes: Record<string, RawEdgeRoute> | null = null,
  ): { cells: CellData[]; edges: EdgeData[] } {
    const cells = this._parseCells(json, bboxOverrides, labelOverrides);
    const edges = this._parseEdges(json, externalRoutes, cells);
    return { cells, edges };
  }

  // ── private helpers ─────────────────────────────────────────────────────────

  private _parseCells(
    json: RawCompositeParams,
    bboxOverrides: Record<string, RawBboxFile>,
    labelOverrides: Record<string, string>,
  ): CellData[] {
    const rawCells         = json.cells ?? {};
    const speciesAssignment = json.species_assignment ?? {};
    const cells: CellData[] = [];

    for (const [cellId, entry] of Object.entries(rawCells)) {

      // ── Resolve bbox ──────────────────────────────────────────────────
      const bboxFile  = bboxOverrides[cellId];
      const agentBbox = entry.agent_params?.bbox;

      const resolvedBbox = bboxFile
        ? { x: bboxFile.x, y: bboxFile.y, w: bboxFile.w, h: bboxFile.h, z: bboxFile.z }
        : agentBbox
        ? { x: agentBbox.x, y: agentBbox.y, w: agentBbox.w, h: agentBbox.h, z: agentBbox.z }
        : { x: 0, y: 0, w: 100, h: 50, z: 1 };

      // ── Resolve species ───────────────────────────────────────────────
      const species: string =
        bboxFile?.species
        ?? speciesAssignment[cellId]?.species
        ?? 'unknown';

      // ── Resolve label ─────────────────────────────────────────────────
      const label: string = labelOverrides[cellId] ?? cellId;

      // ── PBR material ──────────────────────────────────────────────────
      const mat = speciesMaterial(species);

      cells.push({
        cell_id:   cellId,
        species,
        x:         resolvedBbox.x,
        y:         resolvedBbox.y,
        w:         resolvedBbox.w,
        h:         resolvedBbox.h,
        z:         resolvedBbox.z,
        metallic:  mat.metallic,
        roughness: mat.roughness,
        albedo:    mat.albedo,
        label,
      });
    }

    return cells;
  }

  private _parseEdges(
    json: RawCompositeParams,
    externalRoutes: Record<string, RawEdgeRoute> | null,
    cells: CellData[],
  ): EdgeData[] {
    // Authoritative routes: physics/edge_routes.json beats composite edge_routes
    const routes: Record<string, RawEdgeRoute> =
      externalRoutes ?? json.edge_routes ?? {};

    // Quick species lookup by cell_id
    const speciesByCell = new Map<string, string>(
      cells.map(c => [c.cell_id, c.species]),
    );

    const edges: EdgeData[] = [];

    for (const [edgeId, route] of Object.entries(routes)) {
      const srcSpecies = speciesByCell.get(route.source) ?? 'unknown';
      const mat        = speciesMaterial(srcSpecies);

      edges.push({
        edge_id:       edgeId,
        source:        route.source,
        target:        route.target,
        controlPoints: route.control_points,
        color:         mat.wireColor,
      });
    }

    return edges;
  }

  /**
   * Load channels/cell/<id>/bbox.json and channels/cell/<id>/params.json
   * for every cell id in parallel. Missing files are silently skipped.
   *
   * Returns [bboxMap, labelMap] where labelMap maps cellId → human label.
   */
  private async _fetchCellFiles(
    cellIds: string[],
    base: string,
  ): Promise<[Record<string, RawBboxFile>, Record<string, string>]> {
    const results = await Promise.all(
      cellIds.map(async (id) => {
        const [bbox, params] = await Promise.all([
          loadJson<RawBboxFile>(`${base}channels/cell/${id}/bbox.json`),
          loadJson<RawParamsFile>(`${base}channels/cell/${id}/params.json`),
        ]);
        return { id, bbox, label: params?.label ?? null };
      }),
    );

    const bboxMap: Record<string, RawBboxFile>  = {};
    const labelMap: Record<string, string>       = {};

    for (const { id, bbox, label } of results) {
      if (bbox  !== null) bboxMap[id]  = bbox;
      if (label !== null) labelMap[id] = label;
    }

    return [bboxMap, labelMap];
  }
}

// ─── Convenience singleton ───────────────────────────────────────────────────

export const sceneDataLoader = new SceneDataLoader();

/**
 * Top-level loadScene() shortcut.
 *
 * Browser usage:
 *   import { loadScene } from './scene-data-loader';
 *   const { cells, edges } = await loadScene();
 *
 * Node REPL validation (run from project root):
 *   const { loadScene } = require('./src/lib/sph/scene-data-loader');
 *   loadScene().then(({ cells, edges }) => { console.log(cells, edges); });
 */
export async function loadScene(
  baseDir = '',
): Promise<{ cells: CellData[]; edges: EdgeData[] }> {
  return sceneDataLoader.loadScene(baseDir);
}
