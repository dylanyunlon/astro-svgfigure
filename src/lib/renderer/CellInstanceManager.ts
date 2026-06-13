/**
 * CellInstanceManager.ts — per-species instanced cell rendering
 *
 * Species distribution (7 cells, 5 species) from channels/cell/*/params.json:
 *
 *   species          cells                   color
 *   ─────────────────────────────────────────────────
 *   cil-plus         add_norm1, add_norm2    #1E88E5
 *   cil-vector       input_embed, pos_encode #2E7D32
 *   cil-bolt         ffn                     #FF6F00
 *   cil-eye          self_attn               #3F51B5
 *   cil-arrow-right  output                  #455A64
 *
 * Each species gets one InstancedMesh → one draw call per frame regardless of
 * how many cells share that species.  Matching AT's instanced rendering pattern
 * where position/color/opacity are the only per-instance varying values.
 *
 * Usage:
 *   const mgr = new CellInstanceManager(gl);
 *   await mgr.loadFromParamsDir('/channels/cell');   // fetch all params.json
 *   // or:
 *   mgr.loadFromDescriptors(descriptors);            // supply pre-parsed data
 *   mgr.draw(viewMat, projMat);
 *   mgr.dispose();
 */

import { InstancedMesh, INSTANCED_VERT, INSTANCED_FRAG } from './InstancedMesh';
import type { InstanceData } from './InstancedMesh';

// ── Types ────────────────────────────────────────────────────────────────────

export interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
  z?: number;
}

export interface CellParamsJson {
  cell_id: string;
  species: string;
  bbox: CellBBox;
  z?: number;
  opacity: number;
  fill_color: string;  // hex e.g. "#1E88E5"
  stroke_color: string;
  label?: string;
  species_params?: Record<string, unknown>;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Parse "#RRGGBB" or "#RGB" → [r, g, b] in 0-1 range */
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  if (h.length === 3) {
    return [
      parseInt(h[0] + h[0], 16) / 255,
      parseInt(h[1] + h[1], 16) / 255,
      parseInt(h[2] + h[2], 16) / 255,
    ];
  }
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

/**
 * Build a column-major mat4 that translates to (cx, cy, z) and scales to (w, h).
 * The unit quad in InstancedMesh is 1×1 centred at origin, so we scale by w/h
 * and translate to the cell centre.
 *
 * Canvas coordinate system: origin top-left, Y down.
 * We convert to NDC in the vertex shader via u_view / u_projection, so here
 * we just produce a world-space model matrix.
 */
function bboxToModelMatrix(bbox: CellBBox): Float32Array {
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;
  const w  = bbox.w;
  const h  = bbox.h;

  // column-major:
  // [ w  0  0  cx ]
  // [ 0  h  0  cy ]
  // [ 0  0  1  0  ]
  // [ 0  0  0  1  ]
  return new Float32Array([
    w,  0,  0,  0,
    0,  h,  0,  0,
    0,  0,  1,  0,
    cx, cy, 0,  1,
  ]);
}

// ── CellInstanceManager ──────────────────────────────────────────────────────

export interface SpeciesGroup {
  species: string;
  mesh: InstancedMesh;
  cells: CellParamsJson[];
}

export class CellInstanceManager {
  private gl: WebGL2RenderingContext;
  /** species → SpeciesGroup */
  private groups = new Map<string, SpeciesGroup>();

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  // ── Data loading ───────────────────────────────────────────────────────────

  /**
   * Fetch all params.json files from a directory tree structured as:
   *   {baseDir}/{cell_id}/params.json
   *
   * `cellIds` must be provided since we cannot list directories in a browser.
   * Default list matches the 7 cells in channels/cell.
   */
  async loadFromParamsDir(
    baseDir: string,
    cellIds = ['add_norm1', 'add_norm2', 'ffn', 'input_embed', 'output', 'pos_encode', 'self_attn'],
  ): Promise<void> {
    const fetches = cellIds.map(async (id) => {
      const url = `${baseDir}/${id}/params.json`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`CellInstanceManager: failed to fetch ${url} (${res.status})`);
      return res.json() as Promise<CellParamsJson>;
    });

    const descriptors = await Promise.all(fetches);
    this.loadFromDescriptors(descriptors);
  }

  /**
   * Build species groups from pre-parsed CellParamsJson descriptors.
   * Idempotent — disposes existing meshes before rebuilding.
   */
  loadFromDescriptors(descriptors: CellParamsJson[]): void {
    this.dispose();

    // Group by species
    const bySpecies = new Map<string, CellParamsJson[]>();
    for (const d of descriptors) {
      const arr = bySpecies.get(d.species) ?? [];
      arr.push(d);
      bySpecies.set(d.species, arr);
    }

    // Create one InstancedMesh per species
    for (const [species, cells] of bySpecies) {
      const mesh = new InstancedMesh(
        this.gl,
        INSTANCED_VERT,
        INSTANCED_FRAG,
        Math.max(cells.length, 16), // room to grow
      );
      mesh.setInstanceCount(cells.length);

      for (let i = 0; i < cells.length; i++) {
        this._writeCellInstance(mesh, i, cells[i]);
      }
      mesh.upload();

      this.groups.set(species, { species, mesh, cells });
    }
  }

  /**
   * Update a single cell's instance data without full rebuild.
   * Useful for animation: opacity fade, position lerp, etc.
   */
  updateCell(cellId: string, patch: Partial<Pick<CellParamsJson, 'bbox' | 'opacity' | 'fill_color'>>): void {
    for (const group of this.groups.values()) {
      const idx = group.cells.findIndex(c => c.cell_id === cellId);
      if (idx < 0) continue;

      const cell = { ...group.cells[idx], ...patch };
      group.cells[idx] = cell;
      this._writeCellInstance(group.mesh, idx, cell);
      group.mesh.upload();
      return;
    }
    console.warn(`CellInstanceManager.updateCell: cell_id "${cellId}" not found`);
  }

  // ── Rendering ──────────────────────────────────────────────────────────────

  /**
   * Draw all species groups — one gl.drawElementsInstanced per species.
   * 5 species → 5 draw calls for all 7 cells (vs 7 without instancing).
   * At scale (N epochs × 7 cells) the saving is N×7 → N×5 draw calls.
   */
  draw(
    view?: Float32Array,
    projection?: Float32Array,
  ): void {
    for (const group of this.groups.values()) {
      group.mesh.draw(view, projection);
    }
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Returns a summary suitable for debug overlays */
  stats(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [species, group] of this.groups) {
      out[species] = group.mesh.instanceCount;
    }
    return out;
  }

  /** List of species currently managed */
  get speciesList(): string[] {
    return [...this.groups.keys()];
  }

  /** Total number of cells across all species */
  get totalCells(): number {
    let n = 0;
    for (const g of this.groups.values()) n += g.mesh.instanceCount;
    return n;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  dispose(): void {
    for (const group of this.groups.values()) {
      group.mesh.dispose();
    }
    this.groups.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _writeCellInstance(mesh: InstancedMesh, i: number, cell: CellParamsJson): void {
    const [r, g, b] = hexToRgb(cell.fill_color);
    const data: InstanceData = {
      modelMatrix: bboxToModelMatrix(cell.bbox),
      color: [r, g, b, 1.0],
      opacity: cell.opacity,
    };
    mesh.setInstanceAttribute(i, data);
  }
}
