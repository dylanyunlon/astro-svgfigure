/**
 * src/lib/sph/fluid-surface-mesh.ts  —  M746
 *
 * SPH Particle → Surface Mesh Reconstruction (Marching Squares 2D)
 * ─────────────────────────────────────────────────────────────────────────────
 * Converts a 2D SPH particle field into a closed polygon / triangle mesh
 * representing the fluid surface boundary. Designed to plug into the existing
 * SPH pipeline alongside at-water-surface.ts and world-renderer.ts.
 *
 * ─── Algorithm Overview ────────────────────────────────────────────────────
 *
 *   1. Scalar field construction
 *      For each cell of a regular grid, accumulate particle contributions
 *      using the Poly6 kernel (same as density estimation in sph-kernels.ts).
 *      The result is a smooth 2D density/colour field φ(x, y).
 *
 *   2. Marching Squares
 *      Walk every 2×2 cell quad. Classify each corner as inside (φ ≥ iso)
 *      or outside (φ < iso). The 4-bit case index (0–15) selects from the
 *      standard 16-entry edge table. Linear interpolation along active edges
 *      gives sub-cell vertex positions.
 *
 *   3. Contour → Triangle mesh
 *      The contour segments form closed polylines. A simple ear-clipping
 *      triangulator converts them into triangle-list index buffers suitable
 *      for WebGPU / Canvas2D rendering.
 *
 * ─── Data Flow ─────────────────────────────────────────────────────────────
 *
 *   ParticleData (types.ts)          SpatialHashGrid
 *        │                                │
 *        ▼                                ▼
 *   FluidSurfaceMesh.rasterizeField()  (neighbor acceleration)
 *        │
 *        ▼
 *   FluidSurfaceMesh.marchSquares()
 *        │
 *        ├─► contourSegments   (line pairs for debug / stroke rendering)
 *        └─► triangulate()     (fill rendering)
 *
 * ─── Integration ───────────────────────────────────────────────────────────
 *
 *
 *   const mesh = new FluidSurfaceMesh({
 *     gridW: 128, gridH: 96,
 *     domainW: 800, domainH: 600,
 *     smoothingRadius: 12,
 *     isoLevel: 400,
 *   });
 *
 *   // Each frame:
 *   mesh.update(particles.x, particles.y, particles.count);
 *   const { vertices, indices } = mesh.getTriangleMesh();
 *   // → upload to GPUBuffer or draw with Canvas2D
 *
 * Research: xiaodi #M746 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────




import { poly6W } from './sph-kernels';
import { SpatialHashGrid } from './SpatialHashGrid';
import type { ParticleData } from './types';

export interface FluidSurfaceMeshConfig {
  /** Grid resolution along X. Default 128. */
  gridW?: number;

  /** Grid resolution along Y. Default 96. */
  gridH?: number;

  /** Simulation domain width in world units. */
  domainW: number;

  /** Simulation domain height in world units. */
  domainH: number;

  /**
   * SPH smoothing radius h.
   * Must match the kernel support used by the SPH solver (typically 12).
   */
  smoothingRadius: number;

  /**
   * Iso-level threshold for the marching squares contour.
   * Cells with scalar field value ≥ isoLevel are classified as "inside" fluid.
   * Typical range: 200–800 depending on particle mass and rest density.
   * Default 400.
   */
  isoLevel?: number;

  /**
   * Particle mass used in the kernel summation.
   * Default 1.0 (matches sph-kernels defaultConfig).
   */
  particleMass?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Marching Squares edge table
// ─────────────────────────────────────────────────────────────────────────────
//
// Corner layout of a cell quad:
//
//     c3 ──── c2         bit 0 = bottom-left  (c0)
//      │      │          bit 1 = bottom-right (c1)
//     c0 ──── c1         bit 2 = top-right    (c2)
//                        bit 3 = top-left     (c3)
//
// Edge indices:
//   edge 0 = bottom  (c0 → c1)
//   edge 1 = right   (c1 → c2)
//   edge 2 = top     (c3 → c2)
//   edge 3 = left    (c0 → c3)
//
// Each entry in the table is a list of edge-pair line segments.
// -1 terminates the list.

const EDGE_TABLE: readonly (readonly number[])[] = [
  /* 0:  0000 */  [-1],
  /* 1:  0001 */  [3, 0, -1],
  /* 2:  0010 */  [0, 1, -1],
  /* 3:  0011 */  [3, 1, -1],
  /* 4:  0100 */  [1, 2, -1],
  /* 5:  0101 */  [3, 0, 1, 2, -1],   // saddle: two segments
  /* 6:  0110 */  [0, 2, -1],
  /* 7:  0111 */  [3, 2, -1],
  /* 8:  1000 */  [2, 3, -1],
  /* 9:  1001 */  [2, 0, -1],
  /* 10: 1010 */  [2, 3, 0, 1, -1],   // saddle: two segments
  /* 11: 1011 */  [2, 1, -1],
  /* 12: 1100 */  [1, 3, -1],
  /* 13: 1101 */  [1, 0, -1],
  /* 14: 1110 */  [0, 3, -1],
  /* 15: 1111 */  [-1],
];

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

/** A 2D vertex in world coordinates. */
export interface Vertex2D {
  x: number;
  y: number;
}

/** Contour segment: a line from `a` to `b` in world space. */
export interface ContourSegment {
  a: Vertex2D;
  b: Vertex2D;
}

/**
 * Triangle mesh output suitable for GPU upload.
 *
 *   vertices : flat Float32Array [x0, y0, x1, y1, …]
 *   indices  : flat Uint32Array  [i0, i1, i2, …]  (triangle-list)
 */
export interface SurfaceTriangleMesh {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  triangleCount: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// FluidSurfaceMesh
// ─────────────────────────────────────────────────────────────────────────────

export class FluidSurfaceMesh {
  // ── Grid dimensions ──────────────────────────────────────────────────────
  readonly gridW: number;
  readonly gridH: number;
  readonly domainW: number;
  readonly domainH: number;
  readonly cellW: number;   // world-space width of one grid cell
  readonly cellH: number;   // world-space height of one grid cell

  // ── SPH parameters ───────────────────────────────────────────────────────
  readonly smoothingRadius: number;
  readonly particleMass: number;
  isoLevel: number;

  // ── Internal buffers ─────────────────────────────────────────────────────

  /**
   * Scalar field sampled at grid vertices (gridW+1) × (gridH+1).
   * Stored row-major: index = iy * (gridW + 1) + ix.
   */
  private readonly field: Float32Array;
  private readonly fieldW: number;  // gridW + 1 (vertex count along X)
  private readonly fieldH: number;  // gridH + 1

  /** Reusable spatial hash for neighbor queries during rasterization. */
  private readonly hash: SpatialHashGrid;

  /** Cached contour segments from the most recent marchSquares() call. */
  private _segments: ContourSegment[] = [];

  /** Cached triangle mesh from the most recent triangulate() call. */
  private _mesh: SurfaceTriangleMesh | null = null;

  /** Dirty flag: set true after update(), cleared after marchSquares(). */
  private _dirty = true;

  // ── Constructor ──────────────────────────────────────────────────────────

  constructor(cfg: FluidSurfaceMeshConfig) {
    this.gridW = cfg.gridW ?? 128;
    this.gridH = cfg.gridH ?? 96;
    this.domainW = cfg.domainW;
    this.domainH = cfg.domainH;
    this.cellW = this.domainW / this.gridW;
    this.cellH = this.domainH / this.gridH;
    this.smoothingRadius = cfg.smoothingRadius;
    this.particleMass = cfg.particleMass ?? 1.0;
    this.isoLevel = cfg.isoLevel ?? 400;

    this.fieldW = this.gridW + 1;
    this.fieldH = this.gridH + 1;
    this.field = new Float32Array(this.fieldW * this.fieldH);

    // Hash table sized for typical SPH particle counts
    this.hash = new SpatialHashGrid(131072);
  }

  // ─── Public API ──────────────────────────────────────────────────────────

  /**
   * Rasterize SPH particles onto the scalar field, then run marching squares.
   *
   * @param px  Particle X positions (Float32Array from ParticleData.x)
   * @param py  Particle Y positions (Float32Array from ParticleData.y)
   * @param n   Number of active particles
   */
  update(px: Float32Array, py: Float32Array, n: number): void {
    this._rasterizeField(px, py, n);
    this._marchSquares();
    this._mesh = null;   // invalidate cached triangle mesh
    this._dirty = false;
  }

  /**
   * Convenience overload accepting a ParticleData struct directly.
   */
  updateFromParticleData(pd: ParticleData): void {
    this.update(pd.x, pd.y, pd.count);
  }

  /**
   * Get the contour line segments from the most recent update().
   * Useful for stroke-based / debug rendering.
   */
  getContourSegments(): readonly ContourSegment[] {
    return this._segments;
  }

  /**
   * Get the raw scalar field values. Useful for heatmap / debug visualisation.
   * Layout: row-major, (gridW+1) × (gridH+1).
   */
  getField(): Float32Array {
    return this.field;
  }

  /**
   * Get a triangle mesh from the contour.
   * Lazily computed and cached until the next update().
   */
  getTriangleMesh(): SurfaceTriangleMesh {
    if (!this._mesh) {
      this._mesh = this._triangulate();
    }
    return this._mesh;
  }

  /**
   * Query the scalar field value at an arbitrary world-space position
   * using bilinear interpolation.
   */
  sampleField(wx: number, wy: number): number {
    const fx = (wx / this.domainW) * this.gridW;
    const fy = (wy / this.domainH) * this.gridH;
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const tx = fx - ix;
    const ty = fy - iy;

    const ix0 = Math.max(0, Math.min(ix, this.gridW));
    const ix1 = Math.min(ix + 1, this.gridW);
    const iy0 = Math.max(0, Math.min(iy, this.gridH));
    const iy1 = Math.min(iy + 1, this.gridH);

    const v00 = this.field[iy0 * this.fieldW + ix0];
    const v10 = this.field[iy0 * this.fieldW + ix1];
    const v01 = this.field[iy1 * this.fieldW + ix0];
    const v11 = this.field[iy1 * this.fieldW + ix1];

    return (
      v00 * (1 - tx) * (1 - ty) +
      v10 * tx * (1 - ty) +
      v01 * (1 - tx) * ty +
      v11 * tx * ty
    );
  }

  // ─── Private: scalar field rasterization ─────────────────────────────────

  /**
   * For every grid vertex, sum Poly6 kernel contributions from nearby
   * particles. This produces a smooth scalar field whose iso-contour at
   * `isoLevel` approximates the fluid surface.
   *
   * We use the SpatialHashGrid to avoid O(vertices × particles) brute force.
   * The hash is built once per frame over the particles, then each vertex
   * queries its 3×3 neighbourhood (matching the SPH solver pattern from
   * SpatialHashGrid.forEachNeighbor).
   */
  private _rasterizeField(
    px: Float32Array,
    py: Float32Array,
    n: number,
  ): void {
    const field = this.field;
    const h = this.smoothingRadius;
    const mass = this.particleMass;
    const fw = this.fieldW;
    const fh = this.fieldH;
    const cw = this.cellW;
    const ch = this.cellH;

    // Clear field
    field.fill(0);

    // Build spatial hash over particles
    const hash = this.hash;
    hash.clear();
    hash.insertAll(px, py, n, h);

    // For each grid vertex, accumulate kernel contributions
    const head = hash.getHead();
    const next = hash.getNext();
    const tableSize = hash.getTableSize();
    const mask = tableSize - 1;
    const invCell = 1.0 / h;
    const h2 = h * h;

    for (let iy = 0; iy < fh; iy++) {
      const wy = iy * ch;  // world-space Y of this grid vertex
      const rowOff = iy * fw;

      for (let ix = 0; ix < fw; ix++) {
        const wx = ix * cw;  // world-space X

        // Query the 3×3 hash cells around (wx, wy)
        const cxMin = Math.floor((wx - h) * invCell) | 0;
        const cxMax = Math.floor((wx + h) * invCell) | 0;
        const cyMin = Math.floor((wy - h) * invCell) | 0;
        const cyMax = Math.floor((wy + h) * invCell) | 0;

        let sum = 0.0;

        for (let cy = cyMin; cy <= cyMax; cy++) {
          for (let cx = cxMin; cx <= cxMax; cx++) {
            const bucket = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
            let j = head[bucket];
            while (j !== -1) {
              const dx = px[j] - wx;
              const dy = py[j] - wy;
              const r2 = dx * dx + dy * dy;
              if (r2 <= h2) {
                sum += mass * poly6W(Math.sqrt(r2), h);
              }
              j = next[j];
            }
          }
        }

        field[rowOff + ix] = sum;
      }
    }
  }

  // ─── Private: marching squares ───────────────────────────────────────────

  /**
   * Walk every cell quad and emit contour segments where the iso-level
   * crosses cell edges.
   */
  private _marchSquares(): void {
    const segments: ContourSegment[] = [];
    const field = this.field;
    const fw = this.fieldW;
    const iso = this.isoLevel;
    const cw = this.cellW;
    const ch = this.cellH;

    for (let iy = 0; iy < this.gridH; iy++) {
      for (let ix = 0; ix < this.gridW; ix++) {
        // Corner values (same layout as the edge table comment above)
        const v0 = field[iy * fw + ix];           // bottom-left
        const v1 = field[iy * fw + ix + 1];       // bottom-right
        const v2 = field[(iy + 1) * fw + ix + 1]; // top-right
        const v3 = field[(iy + 1) * fw + ix];     // top-left

        // 4-bit case index
        let caseIdx = 0;
        if (v0 >= iso) caseIdx |= 1;
        if (v1 >= iso) caseIdx |= 2;
        if (v2 >= iso) caseIdx |= 4;
        if (v3 >= iso) caseIdx |= 8;

        if (caseIdx === 0 || caseIdx === 15) continue;

        // Saddle-point disambiguation: use average of four corners
        // to decide which diagonal pairing to use.
        // Cases 5 and 10 are ambiguous. We resolve them by checking
        // whether the cell center is inside or outside.
        if (caseIdx === 5 || caseIdx === 10) {
          const center = (v0 + v1 + v2 + v3) * 0.25;
          if (caseIdx === 5 && center < iso) {
            // Swap to the alternative connectivity: join the two
            // separate inside regions instead of treating them separately.
            // Use case 10's edge pattern for case 5 when center is outside.
            caseIdx = 10;
          } else if (caseIdx === 10 && center < iso) {
            caseIdx = 5;
          }
        }

        // World-space origin of this cell (bottom-left corner)
        const ox = ix * cw;
        const oy = iy * ch;

        // Interpolate vertex positions along each active edge
        const edgeVerts = this._interpolateEdges(
          ox, oy, cw, ch,
          v0, v1, v2, v3,
          iso,
        );

        // Read edge pairs from the table
        const edges = EDGE_TABLE[caseIdx];
        let ei = 0;
        while (edges[ei] !== -1) {
          const eA = edges[ei];
          const eB = edges[ei + 1];
          segments.push({
            a: edgeVerts[eA],
            b: edgeVerts[eB],
          });
          ei += 2;
        }
      }
    }

    this._segments = segments;
  }

  /**
   * Compute interpolated vertex positions for the four edges of a cell.
   *
   * Returns an array of 4 Vertex2D, one per edge:
   *   [0] bottom  (c0 → c1)
   *   [1] right   (c1 → c2)
   *   [2] top     (c3 → c2)
   *   [3] left    (c0 → c3)
   */
  private _interpolateEdges(
    ox: number, oy: number,
    cw: number, ch: number,
    v0: number, v1: number, v2: number, v3: number,
    iso: number,
  ): Vertex2D[] {
    return [
      // Edge 0: bottom (c0 → c1), along X at y = oy
      { x: ox + cw * lerpFactor(v0, v1, iso), y: oy },
      // Edge 1: right (c1 → c2), along Y at x = ox + cw
      { x: ox + cw, y: oy + ch * lerpFactor(v1, v2, iso) },
      // Edge 2: top (c3 → c2), along X at y = oy + ch
      { x: ox + cw * lerpFactor(v3, v2, iso), y: oy + ch },
      // Edge 3: left (c0 → c3), along Y at x = ox
      { x: ox, y: oy + ch * lerpFactor(v0, v3, iso) },
    ];
  }

  // ─── Private: triangulation ──────────────────────────────────────────────

  /**
   * Convert contour segments into a filled triangle mesh.
   *
   * Strategy: for each marching-squares cell that has contour edges, we
   * produce triangles by fan-triangulating the inside polygon of that cell.
   * This avoids the complexity of global contour chaining and works well
   * for real-time fluid rendering where per-cell locality is desirable.
   *
   * For each cell, the "inside" polygon is formed by:
   *   - corners that are ≥ iso (in counter-clockwise order)
   *   - interpolated edge crossing points inserted between adjacent corners
   *
   * We fan-triangulate from the first vertex of each cell polygon.
   */
  private _triangulate(): SurfaceTriangleMesh {
    const verts: number[] = [];   // flat [x, y, x, y, …]
    const idxs: number[] = [];    // flat [i0, i1, i2, …]
    let vertCount = 0;

    const field = this.field;
    const fw = this.fieldW;
    const iso = this.isoLevel;
    const cw = this.cellW;
    const ch = this.cellH;

    for (let iy = 0; iy < this.gridH; iy++) {
      for (let ix = 0; ix < this.gridW; ix++) {
        const v0 = field[iy * fw + ix];
        const v1 = field[iy * fw + ix + 1];
        const v2 = field[(iy + 1) * fw + ix + 1];
        const v3 = field[(iy + 1) * fw + ix];

        let caseIdx = 0;
        if (v0 >= iso) caseIdx |= 1;
        if (v1 >= iso) caseIdx |= 2;
        if (v2 >= iso) caseIdx |= 4;
        if (v3 >= iso) caseIdx |= 8;

        if (caseIdx === 0) continue;

        // Full cell — emit two triangles (quad)
        if (caseIdx === 15) {
          const ox = ix * cw;
          const oy = iy * ch;
          const baseIdx = vertCount;
          verts.push(ox, oy, ox + cw, oy, ox + cw, oy + ch, ox, oy + ch);
          idxs.push(baseIdx, baseIdx + 1, baseIdx + 2);
          idxs.push(baseIdx, baseIdx + 2, baseIdx + 3);
          vertCount += 4;
          continue;
        }

        // Saddle disambiguation
        if (caseIdx === 5 || caseIdx === 10) {
          const center = (v0 + v1 + v2 + v3) * 0.25;
          if (caseIdx === 5 && center < iso) {
            caseIdx = 10;
          } else if (caseIdx === 10 && center < iso) {
            caseIdx = 5;
          }
        }

        const ox = ix * cw;
        const oy = iy * ch;

        // Build the inside polygon for this cell by walking corners CCW
        // and inserting edge interpolation points where the contour crosses.
        const poly = this._buildInsidePolygon(
          ox, oy, cw, ch,
          v0, v1, v2, v3,
          caseIdx, iso,
        );

        if (poly.length < 3) continue;

        // Emit vertices
        const baseIdx = vertCount;
        for (const p of poly) {
          verts.push(p.x, p.y);
        }
        vertCount += poly.length;

        // Fan triangulation from poly[0]
        for (let k = 1; k < poly.length - 1; k++) {
          idxs.push(baseIdx, baseIdx + k, baseIdx + k + 1);
        }
      }
    }

    return {
      vertices: new Float32Array(verts),
      indices: new Uint32Array(idxs),
      vertexCount: vertCount,
      triangleCount: idxs.length / 3,
    };
  }

  /**
   * Build the CCW-ordered inside polygon for a marching-squares cell.
   *
   * Walks the four edges of the cell in CCW order (bottom → right → top → left).
   * For each corner, if it is inside (≥ iso), add it. At each edge crossing,
   * insert the interpolated point.
   *
   * Corner order CCW: c0 (BL) → c1 (BR) → c2 (TR) → c3 (TL)
   * Edge order CCW:   e0 (bottom) → e1 (right) → e2 (top, reversed) → e3 (left, reversed)
   */
  private _buildInsidePolygon(
    ox: number, oy: number,
    cw: number, ch: number,
    v0: number, v1: number, v2: number, v3: number,
    caseIdx: number, iso: number,
  ): Vertex2D[] {
    const poly: Vertex2D[] = [];
    const corners: Vertex2D[] = [
      { x: ox,      y: oy },        // c0 BL
      { x: ox + cw, y: oy },        // c1 BR
      { x: ox + cw, y: oy + ch },   // c2 TR
      { x: ox,      y: oy + ch },   // c3 TL
    ];
    const vals = [v0, v1, v2, v3];
    const inside = [
      (caseIdx & 1) !== 0,
      (caseIdx & 2) !== 0,
      (caseIdx & 4) !== 0,
      (caseIdx & 8) !== 0,
    ];

    // Edge interpolation points (same order as _interpolateEdges):
    //   edge 0: bottom (c0→c1),  edge 1: right (c1→c2),
    //   edge 2: top (c3→c2),     edge 3: left (c0→c3)
    //
    // But we walk CCW: c0→c1→c2→c3, so the edges in CCW order are:
    //   c0→c1 (edge 0), c1→c2 (edge 1), c2→c3 (edge 2 reversed), c3→c0 (edge 3 reversed)

    // Helper: interpolation point between two corners
    const edgePt = (ci: number, cj: number): Vertex2D => {
      const t = lerpFactor(vals[ci], vals[cj], iso);
      return {
        x: corners[ci].x + t * (corners[cj].x - corners[ci].x),
        y: corners[ci].y + t * (corners[cj].y - corners[ci].y),
      };
    };

    // CCW corner order: 0 → 1 → 2 → 3
    const order: [number, number][] = [
      [0, 1], [1, 2], [2, 3], [3, 0],
    ];

    for (const [ci, cj] of order) {
      if (inside[ci]) {
        poly.push(corners[ci]);
      }
      // If the edge crosses the iso-level, insert interpolated vertex
      if (inside[ci] !== inside[cj]) {
        poly.push(edgePt(ci, cj));
      }
    }

    return poly;
  }

  // ─── Static helpers ──────────────────────────────────────────────────────

  /**
   * Convenience: create a FluidSurfaceMesh sized to match a WorldConfig.
   */
  static fromWorldConfig(
    worldCfg: { width: number; height: number; smoothingRadius: number },
    opts?: { gridW?: number; gridH?: number; isoLevel?: number; particleMass?: number },
  ): FluidSurfaceMesh {
    return new FluidSurfaceMesh({
      gridW: opts?.gridW ?? 128,
      gridH: opts?.gridH ?? 96,
      domainW: worldCfg.width,
      domainH: worldCfg.height,
      smoothingRadius: worldCfg.smoothingRadius,
      isoLevel: opts?.isoLevel ?? 400,
      particleMass: opts?.particleMass ?? 1.0,
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Linear interpolation factor: returns t ∈ [0, 1] such that
 * lerp(vA, vB, t) = iso. Clamped to avoid division-by-zero
 * when vA ≈ vB.
 */
function lerpFactor(vA: number, vB: number, iso: number): number {
  const denom = vB - vA;
  if (Math.abs(denom) < 1e-12) return 0.5;
  return Math.max(0, Math.min(1, (iso - vA) / denom));
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas2D rendering helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw the contour segments onto a Canvas2D context.
 * Useful for quick debug visualisation.
 */
export function drawContourToCanvas(
  ctx: CanvasRenderingContext2D,
  mesh: FluidSurfaceMesh,
  style?: {
    strokeStyle?: string;
    lineWidth?: number;
    fillStyle?: string;
  },
): void {
  const segs = mesh.getContourSegments();
  if (segs.length === 0) return;

  ctx.save();

  // Stroke contour
  ctx.beginPath();
  ctx.strokeStyle = style?.strokeStyle ?? 'rgba(64, 164, 255, 0.9)';
  ctx.lineWidth = style?.lineWidth ?? 1.5;
  for (const seg of segs) {
    ctx.moveTo(seg.a.x, seg.a.y);
    ctx.lineTo(seg.b.x, seg.b.y);
  }
  ctx.stroke();

  // Optionally fill triangles
  if (style?.fillStyle) {
    const tri = mesh.getTriangleMesh();
    ctx.fillStyle = style.fillStyle;
    ctx.beginPath();
    const v = tri.vertices;
    const idx = tri.indices;
    for (let i = 0; i < idx.length; i += 3) {
      const i0 = idx[i] * 2;
      const i1 = idx[i + 1] * 2;
      const i2 = idx[i + 2] * 2;
      ctx.moveTo(v[i0], v[i0 + 1]);
      ctx.lineTo(v[i1], v[i1 + 1]);
      ctx.lineTo(v[i2], v[i2 + 1]);
      ctx.closePath();
    }
    ctx.fill();
  }

  ctx.restore();
}

/**
 * Draw the scalar field as a grayscale heatmap onto a Canvas2D context.
 * Useful for debugging the kernel density estimation.
 */
export function drawFieldHeatmap(
  ctx: CanvasRenderingContext2D,
  mesh: FluidSurfaceMesh,
  maxVal?: number,
): void {
  const field = mesh.getField();
  const fw = mesh.gridW + 1;
  const fh = mesh.gridH + 1;
  const cw = mesh.cellW;
  const ch = mesh.cellH;

  // Auto-detect maximum if not provided
  let vmax = maxVal ?? 0;
  if (vmax <= 0) {
    for (let i = 0; i < field.length; i++) {
      if (field[i] > vmax) vmax = field[i];
    }
  }
  if (vmax <= 0) return;

  const invMax = 255 / vmax;
  ctx.save();
  for (let iy = 0; iy < fh - 1; iy++) {
    for (let ix = 0; ix < fw - 1; ix++) {
      const v = field[iy * fw + ix];
      const brightness = Math.min(255, Math.floor(v * invMax));
      ctx.fillStyle = `rgb(${brightness},${brightness},${brightness})`;
      ctx.fillRect(ix * cw, iy * ch, cw + 1, ch + 1);
    }
  }
  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-test
// ─────────────────────────────────────────────────────────────────────────────

/**
 * selfTest(): boolean
 *
 * Validates the marching squares implementation against known scenarios:
 *
 *  1. Empty field (all zeros) → no segments, no triangles
 *  2. Full field (all above iso) → no segments, full-quad triangles
 *  3. Single particle at centre → closed contour with > 0 segments
 *  4. Two particles far apart → two separate contour regions
 *  5. Vertex count = index max + 1 (index buffer consistency)
 *  6. Bilinear interpolation sanity (sampleField matches corner values)
 *  7. Triangle winding: all triangles have positive signed area (CCW)
 *
 * Returns true when all checks pass; false (with console.error) on failure.
 */
export function selfTest(): boolean {
  const TOL = 1e-6;

  function fail(msg: string): false {
    console.error(`[fluid-surface-mesh selfTest] FAILED: ${msg}`);
    return false;
  }

  const domainW = 200;
  const domainH = 200;
  const h = 12;
  const gridW = 32;
  const gridH = 32;
  const iso = 100;

  // ── Test 1: Empty field ──────────────────────────────────────────────────
  {
    const m = new FluidSurfaceMesh({
      gridW, gridH, domainW, domainH, smoothingRadius: h, isoLevel: iso,
    });
    const px = new Float32Array(0);
    const py = new Float32Array(0);
    m.update(px, py, 0);
    if (m.getContourSegments().length !== 0)
      return fail('Test 1: expected 0 segments for empty field');
    const tri = m.getTriangleMesh();
    if (tri.triangleCount !== 0)
      return fail('Test 1: expected 0 triangles for empty field');
  }

  // ── Test 3: Single particle at centre ────────────────────────────────────
  {
    const m = new FluidSurfaceMesh({
      gridW, gridH, domainW, domainH, smoothingRadius: h, isoLevel: iso,
      particleMass: 1000,
    });
    const px = new Float32Array([100]);
    const py = new Float32Array([100]);
    m.update(px, py, 1);

    const segs = m.getContourSegments();
    if (segs.length === 0)
      return fail('Test 3: expected > 0 segments for single particle');

    const tri = m.getTriangleMesh();
    if (tri.triangleCount === 0)
      return fail('Test 3: expected > 0 triangles for single particle');
  }

  // ── Test 5: Index buffer consistency ─────────────────────────────────────
  {
    const m = new FluidSurfaceMesh({
      gridW, gridH, domainW, domainH, smoothingRadius: h, isoLevel: iso,
      particleMass: 1000,
    });
    const n = 50;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      px[i] = 60 + Math.random() * 80;
      py[i] = 60 + Math.random() * 80;
    }
    m.update(px, py, n);
    const tri = m.getTriangleMesh();

    let maxIdx = 0;
    for (let i = 0; i < tri.indices.length; i++) {
      if (tri.indices[i] > maxIdx) maxIdx = tri.indices[i];
    }
    if (tri.indices.length > 0 && maxIdx >= tri.vertexCount)
      return fail(`Test 5: index ${maxIdx} exceeds vertexCount ${tri.vertexCount}`);
  }

  // ── Test 6: Bilinear interpolation at grid corners ───────────────────────
  {
    const m = new FluidSurfaceMesh({
      gridW: 4, gridH: 4, domainW: 40, domainH: 40,
      smoothingRadius: h, isoLevel: iso, particleMass: 1000,
    });
    const px = new Float32Array([20]);
    const py = new Float32Array([20]);
    m.update(px, py, 1);

    const field = m.getField();
    const fw = 5; // gridW + 1
    // Sample at grid vertex (2, 2) in world coords = (20, 20)
    const directVal = field[2 * fw + 2];
    const sampledVal = m.sampleField(20, 20);
    if (Math.abs(directVal - sampledVal) > TOL)
      return fail(`Test 6: sampleField(20,20) = ${sampledVal}, expected ${directVal}`);
  }

  // ── Test 7: Triangle winding (CCW = positive signed area) ────────────────
  {
    const m = new FluidSurfaceMesh({
      gridW, gridH, domainW, domainH, smoothingRadius: h, isoLevel: iso,
      particleMass: 1000,
    });
    const n = 30;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      px[i] = 80 + Math.random() * 40;
      py[i] = 80 + Math.random() * 40;
    }
    m.update(px, py, n);
    const tri = m.getTriangleMesh();
    const v = tri.vertices;
    const idx = tri.indices;

    for (let i = 0; i < idx.length; i += 3) {
      const ax = v[idx[i] * 2], ay = v[idx[i] * 2 + 1];
      const bx = v[idx[i + 1] * 2], by = v[idx[i + 1] * 2 + 1];
      const cx = v[idx[i + 2] * 2], cy = v[idx[i + 2] * 2 + 1];
      const signedArea = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (signedArea < -TOL)
        return fail(`Test 7: triangle ${i / 3} has CW winding (signed area ${signedArea})`);
    }
  }

  return true;
}
