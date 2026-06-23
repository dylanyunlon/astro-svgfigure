/**
 * scene-data-loader.ts
 * M886: composite_params.json → GPU CellData[] / EdgeData[]
 *
 * SceneDataLoader reads channels/composite_params.json (and two sibling
 * JSON files) and produces the typed arrays that the GPU render loop
 * expects.  All I/O is async-JSON-fetch so it works in both Node and the
 * browser (Vite will inline the channel assets as static files).
 *
 * Data sources (all relative to the project root / public/ root):
 *   channels/composite_params.json          — cells + edge_routes
 *   channels/physics/edge_routes.json       — authoritative control-points
 *   channels/cell/<id>/bbox.json            — per-cell epoch-9 positions
 */

// ─── Re-export shared GPU types ──────────────────────────────────────────────

export interface CellData {
  /** Unique cell identifier matching composite_params.cells key */
  id: string;
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
  /** Human-readable label (same as id, kept for debug overlays) */
  label: string;
}

export interface EdgeData {
  /** Unique edge identifier: e1 … e6, skip1, skip2 */
  id: string;
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
// Colors are derived from the species_params.primary_color values baked into
// composite_params.json and cross-referenced with the AT options table:
//
//   cil-eye          #1E88E5  → blue  (self-attention)
//   cil-bolt         #F57C00  → orange (feed-forward / activation)
//   cil-vector       #546E7A  → steel-blue (embedding)
//   cil-plus         #E53935  → red   (add-norm / residual)
//   cil-arrow-right  #2E7D32  → green (output)
//   cil-sine         #7E57C2  → purple (positional encoding, alias of cil-vector)
//   (fallback)                → mid-grey

interface SpeciesMaterial {
  metallic: number;
  roughness: number;
  albedo: [number, number, number];
  wireColor: [number, number, number];
}

const SPECIES_MATERIAL: Record<string, SpeciesMaterial> = {
  // attention — vibrant blue, high metallic (looks polished/glassy)
  'cil-eye': {
    metallic:  0.04,
    roughness: 0.60,
    albedo:    [0.118, 0.533, 0.898],
    wireColor: [0.118, 0.533, 0.898],
  },
  // feed-forward bolt — vivid orange, high metallic (energy-like)
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
}

// ─── SceneDataLoader ─────────────────────────────────────────────────────────

export class SceneDataLoader {

  // ── public API ──────────────────────────────────────────────────────────────

  /**
   * Parse a fully-loaded composite_params.json object.
   *
   * The loader merges three data sources in priority order:
   *   1. channels/cell/<id>/bbox.json       (epoch-9, most accurate positions)
   *   2. composite_params.cells[id].agent_params.bbox  (epoch-2 fallback)
   *   3. Default bbox {x:0, y:0, w:100, h:50, z:1}   (last resort)
   *
   * Edge control-points come from channels/physics/edge_routes.json and fall
   * back to the edge_routes block embedded in composite_params.json.
   *
   * Call loadFromFiles() if you want automatic file fetching.
   */
  loadFromCompositeParams(
    json: RawCompositeParams,
    bboxOverrides: Record<string, RawBboxFile> = {},
    edgeRoutesOverride: Record<string, RawEdgeRoute> | null = null,
  ): { cells: CellData[]; edges: EdgeData[] } {

    const cells = this._parseCells(json, bboxOverrides);
    const edges = this._parseEdges(json, edgeRoutesOverride, cells);

    return { cells, edges };
  }

  /**
   * Fetch all three JSON sources and return parsed scene data.
   *
   * @param baseUrl  URL prefix for the channel assets (default: '').
   *                 In Vite dev-server, '' works because assets live under
   *                 the project root (imported directly).  For production,
   *                 set to '/channels' if serving from a static host.
   */
  async loadFromFiles(baseUrl = ''): Promise<{ cells: CellData[]; edges: EdgeData[] }> {
    const compositeUrl    = `${baseUrl}/channels/composite_params.json`;
    const edgeRoutesUrl   = `${baseUrl}/channels/physics/edge_routes.json`;

    const [compositeJson, edgeRoutesJson] = await Promise.all([
      fetchJson<RawCompositeParams>(compositeUrl),
      fetchJson<Record<string, RawEdgeRoute>>(edgeRoutesUrl).catch(() => null),
    ]);

    // Fetch per-cell bbox.json files for all cells that appear in composite_params
    const cellIds = Object.keys(compositeJson.cells ?? {});
    const bboxOverrides = await this._fetchBboxFiles(cellIds, baseUrl);

    return this.loadFromCompositeParams(compositeJson, bboxOverrides, edgeRoutesJson);
  }

  // ── private helpers ─────────────────────────────────────────────────────────

  private _parseCells(
    json: RawCompositeParams,
    bboxOverrides: Record<string, RawBboxFile>,
  ): CellData[] {
    const rawCells = json.cells ?? {};
    const speciesAssignment = json.species_assignment ?? {};
    const cells: CellData[] = [];

    for (const [cellId, entry] of Object.entries(rawCells)) {

      // ── Resolve bbox (priority: bbox.json > agent_params.bbox > default) ──
      const bboxFile   = bboxOverrides[cellId];
      const agentBbox  = entry.agent_params?.bbox;

      const bbox: RawBbox = bboxFile
        ? { x: bboxFile.x, y: bboxFile.y, w: bboxFile.w, h: bboxFile.h, z: bboxFile.z }
        : agentBbox
        ? { x: agentBbox.x, y: agentBbox.y, w: agentBbox.w, h: agentBbox.h, z: agentBbox.z }
        : { x: 0, y: 0, w: 100, h: 50, z: 1 };

      // ── Resolve species (priority: bbox.json.species > species_assignment > 'unknown') ──
      const species: string =
        bboxFile?.species
        ?? speciesAssignment[cellId]?.species
        ?? 'unknown';

      // ── PBR material from species ──
      const mat = speciesMaterial(species);

      cells.push({
        id:        cellId,
        species,
        x:         bbox.x,
        y:         bbox.y,
        w:         bbox.w,
        h:         bbox.h,
        z:         bbox.z,
        metallic:  mat.metallic,
        roughness: mat.roughness,
        albedo:    mat.albedo,
        label:     cellId,
      });
    }

    return cells;
  }

  private _parseEdges(
    json: RawCompositeParams,
    externalRoutes: Record<string, RawEdgeRoute> | null,
    cells: CellData[],
  ): EdgeData[] {

    // Authoritative routes: external file beats embedded edge_routes
    const routes: Record<string, RawEdgeRoute> =
      externalRoutes ?? json.edge_routes ?? {};

    // Build a quick species lookup by cell id
    const speciesByCell = new Map<string, string>(
      cells.map(c => [c.id, c.species])
    );

    const edges: EdgeData[] = [];

    for (const [edgeId, route] of Object.entries(routes)) {
      const srcSpecies = speciesByCell.get(route.source) ?? 'unknown';
      const mat        = speciesMaterial(srcSpecies);

      edges.push({
        id:            edgeId,
        source:        route.source,
        target:        route.target,
        controlPoints: route.control_points,
        color:         mat.wireColor,
      });
    }

    return edges;
  }

  /** Fetch channels/cell/<id>/bbox.json for every cell id, silently skip 404s */
  private async _fetchBboxFiles(
    cellIds: string[],
    baseUrl: string,
  ): Promise<Record<string, RawBboxFile>> {
    const entries = await Promise.all(
      cellIds.map(async (id) => {
        const url = `${baseUrl}/channels/cell/${id}/bbox.json`;
        const data = await fetchJson<RawBboxFile>(url).catch(() => null);
        return [id, data] as const;
      })
    );
    const result: Record<string, RawBboxFile> = {};
    for (const [id, data] of entries) {
      if (data !== null) result[id] = data;
    }
    return result;
  }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetchJson: HTTP ${res.status} for ${url}`);
  return res.json() as Promise<T>;
}

// ─── Convenience singleton ───────────────────────────────────────────────────

export const sceneDataLoader = new SceneDataLoader();
