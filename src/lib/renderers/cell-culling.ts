/**
 * cell-culling.ts — M823: Frustum + Occlusion Culling for Cells
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Determines which cells are visible in the current viewport and skips
 * rendering for off-screen or fully-occluded cells.
 *
 * Two-pass culling:
 *   Pass 1: Frustum — reject cells whose bbox is entirely outside viewport
 *   Pass 2: Occlusion — reject cells fully hidden behind higher-z cells
 *
 * References:
 *   - channels/rendering/occlusion/  (Python-side occlusion logic)
 *   - upstream/unreal-renderer/SceneOcclusion.cpp
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface CellBBox {
  cell_id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  species: string;
  opacity: number;
}

export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
  zoom: number;
}

export interface CullingResult {
  visible: CellBBox[];
  culled_frustum: string[];
  culled_occluded: string[];
  stats: {
    total: number;
    visible: number;
    frustum_culled: number;
    occlusion_culled: number;
    cull_ratio: number;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Frustum Culling — viewport AABB intersection test
// ─────────────────────────────────────────────────────────────────────────────

function isInFrustum(cell: CellBBox, vp: Viewport, margin: number = 50): boolean {
  // Expand viewport by margin to avoid popping at edges
  const vpLeft = vp.x - margin;
  const vpTop = vp.y - margin;
  const vpRight = vp.x + vp.width / vp.zoom + margin;
  const vpBottom = vp.y + vp.height / vp.zoom + margin;

  const cellRight = cell.x + cell.w;
  const cellBottom = cell.y + cell.h;

  // AABB overlap test
  return !(cell.x > vpRight || cellRight < vpLeft ||
           cell.y > vpBottom || cellBottom < vpTop);
}

// ─────────────────────────────────────────────────────────────────────────────
// Occlusion Culling — simple z-based overlap test
// ─────────────────────────────────────────────────────────────────────────────

function isOccluded(cell: CellBBox, occluders: CellBBox[]): boolean {
  // A cell is occluded if a higher-z, fully-opaque cell completely covers it
  for (const occ of occluders) {
    if (occ.z <= cell.z) continue;          // must be in front
    if (occ.opacity < 0.95) continue;       // must be opaque enough to occlude
    // Check if occluder fully covers this cell
    if (occ.x <= cell.x &&
        occ.y <= cell.y &&
        occ.x + occ.w >= cell.x + cell.w &&
        occ.y + occ.h >= cell.y + cell.h) {
      return true;
    }
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main cull pass
// ─────────────────────────────────────────────────────────────────────────────

export function cullCells(cells: CellBBox[], viewport: Viewport): CullingResult {
  const culled_frustum: string[] = [];
  const culled_occluded: string[] = [];
  const frustumPassed: CellBBox[] = [];

  // Pass 1: Frustum culling
  for (const cell of cells) {
    if (isInFrustum(cell, viewport)) {
      frustumPassed.push(cell);
    } else {
      culled_frustum.push(cell.cell_id);
    }
  }

  // Sort by z descending for occlusion (front cells are potential occluders)
  const sorted = [...frustumPassed].sort((a, b) => b.z - a.z);

  // Pass 2: Occlusion culling
  const visible: CellBBox[] = [];
  const confirmedOccluders: CellBBox[] = [];

  for (const cell of sorted) {
    if (isOccluded(cell, confirmedOccluders)) {
      culled_occluded.push(cell.cell_id);
    } else {
      visible.push(cell);
      // This cell is visible, so it becomes a potential occluder for cells behind it
      if (cell.opacity >= 0.95) {
        confirmedOccluders.push(cell);
      }
    }
  }

  // Re-sort visible by z ascending (back-to-front for rendering)
  visible.sort((a, b) => a.z - b.z);

  const total = cells.length;
  return {
    visible,
    culled_frustum,
    culled_occluded,
    stats: {
      total,
      visible: visible.length,
      frustum_culled: culled_frustum.length,
      occlusion_culled: culled_occluded.length,
      cull_ratio: total > 0 ? 1 - visible.length / total : 0,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial hash for fast neighbor queries (used by occlusion)
// ─────────────────────────────────────────────────────────────────────────────

export class SpatialHashGrid {
  private grid: Map<string, CellBBox[]> = new Map();
  private cellSize: number;

  constructor(cellSize: number = 200) {
    this.cellSize = cellSize;
  }

  private key(gx: number, gy: number): string {
    return `${gx},${gy}`;
  }

  clear(): void {
    this.grid.clear();
  }

  insert(cell: CellBBox): void {
    const gx0 = Math.floor(cell.x / this.cellSize);
    const gy0 = Math.floor(cell.y / this.cellSize);
    const gx1 = Math.floor((cell.x + cell.w) / this.cellSize);
    const gy1 = Math.floor((cell.y + cell.h) / this.cellSize);

    for (let gx = gx0; gx <= gx1; gx++) {
      for (let gy = gy0; gy <= gy1; gy++) {
        const k = this.key(gx, gy);
        if (!this.grid.has(k)) this.grid.set(k, []);
        this.grid.get(k)!.push(cell);
      }
    }
  }

  query(x: number, y: number, w: number, h: number): CellBBox[] {
    const seen = new Set<string>();
    const result: CellBBox[] = [];
    const gx0 = Math.floor(x / this.cellSize);
    const gy0 = Math.floor(y / this.cellSize);
    const gx1 = Math.floor((x + w) / this.cellSize);
    const gy1 = Math.floor((y + h) / this.cellSize);

    for (let gx = gx0; gx <= gx1; gx++) {
      for (let gy = gy0; gy <= gy1; gy++) {
        const bucket = this.grid.get(this.key(gx, gy));
        if (!bucket) continue;
        for (const cell of bucket) {
          if (!seen.has(cell.cell_id)) {
            seen.add(cell.cell_id);
            result.push(cell);
          }
        }
      }
    }
    return result;
  }
}
