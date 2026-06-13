/**
 * geometry-worker.ts — Off-main-thread geometry computation for cell rendering.
 *
 * Receives cell parameter JSON via Thread.send(), computes vertex positions and
 * index buffers, then postMessages the typed-array buffers back as transferables.
 *
 * Ported from AT GeomThread architecture:
 *   upstream/unreal-renderer/src/workers/geometry-worker.ts
 *   upstream/pixijs-engine/src/assets/loader/workers/WorkerManager.ts
 *
 * Message protocol (matches Thread.ts ThreadRequest / ThreadResponse):
 *   recv: { uuid, method, data: CellGeomInput }
 *   send: { uuid, result: CellGeomOutput }     — on success
 *         { uuid, error: string }               — on failure
 *
 * Supported methods:
 *   'computeVertices'  — full vertex + index + normal + uv buffer computation
 *   'computeAABB'      — axis-aligned bounding box only (fast path)
 *   'computeOutline'   — closed polygon outline for a cell shape
 */

// ── Types ─────────────────────────────────────────────────────────────────────

/** A single cell's spatial parameters received from the main thread. */
export interface CellParams {
  /** Unique cell identifier. */
  cell_id: string;
  /** Centre X in canvas space. */
  x: number;
  /** Centre Y in canvas space. */
  y: number;
  /** Z layer (integer, 1 = foreground). */
  z?: number;
  /** Width of the cell bounding box. */
  width: number;
  /** Height of the cell bounding box. */
  height: number;
  /**
   * Cell shape primitive.
   *   'rect'     — axis-aligned rectangle (default)
   *   'ellipse'  — ellipse inscribed in bbox
   *   'hex'      — regular hexagon inscribed in bbox
   *   'diamond'  — axis-aligned diamond (rhombus)
   */
  shape?: 'rect' | 'ellipse' | 'hex' | 'diamond';
  /** Number of segments for ellipse / hex tessellation (default 32). */
  segments?: number;
  /** Optional 4-component RGBA colour [r,g,b,a] each 0–255. */
  color?: [number, number, number, number];
}

/** Input payload for the 'computeVertices' method. */
export interface CellGeomInput {
  cells: CellParams[];
}

/** Per-cell geometry buffers returned to the main thread. */
export interface CellGeomResult {
  cell_id: string;
  /** Interleaved [x, y, z] vertex positions — Float32Array. */
  positions: Float32Array;
  /** Triangle indices — Uint16Array (or Uint32Array for large meshes). */
  indices: Uint16Array | Uint32Array;
  /** Per-vertex normals [nx, ny, nz] — Float32Array (all [0,0,1] for 2D). */
  normals: Float32Array;
  /** Per-vertex UV coordinates [u, v] — Float32Array. */
  uvs: Float32Array;
  /** Axis-aligned bounding box {minX, minY, maxX, maxY}. */
  aabb: { minX: number; minY: number; maxX: number; maxY: number };
}

/** Output payload for the 'computeVertices' method. */
export interface CellGeomOutput {
  results: CellGeomResult[];
  /**
   * Transferable list forwarded to Thread._settle so the main thread
   * receives the ArrayBuffers without a copy.
   */
  transfer: ArrayBuffer[];
}

/** Output payload for the 'computeAABB' method. */
export interface CellAABBOutput {
  aabbs: Record<string, { minX: number; minY: number; maxX: number; maxY: number }>;
}

/** Closed polygon outline: flat [x0,y0, x1,y1, …] array. */
export interface CellOutlineOutput {
  outlines: Record<string, Float32Array>;
  transfer: ArrayBuffer[];
}

// ── Geometry primitives ───────────────────────────────────────────────────────

/**
 * buildRect — axis-aligned quad.
 *   4 vertices, 2 triangles (6 indices).
 */
function buildRect(
  cx: number,
  cy: number,
  z: number,
  hw: number,
  hh: number
): Pick<CellGeomResult, 'positions' | 'indices' | 'normals' | 'uvs' | 'aabb'> {
  // Vertex order: TL, TR, BR, BL
  const positions = new Float32Array([
    cx - hw, cy - hh, z,
    cx + hw, cy - hh, z,
    cx + hw, cy + hh, z,
    cx - hw, cy + hh, z,
  ]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  return {
    positions,
    indices,
    normals,
    uvs,
    aabb: { minX: cx - hw, minY: cy - hh, maxX: cx + hw, maxY: cy + hh },
  };
}

/**
 * buildEllipse — tessellated ellipse via fan triangulation.
 *   segments+1 vertices (centre + ring), segments triangles.
 */
function buildEllipse(
  cx: number,
  cy: number,
  z: number,
  hw: number,
  hh: number,
  segments: number
): Pick<CellGeomResult, 'positions' | 'indices' | 'normals' | 'uvs' | 'aabb'> {
  const n = Math.max(3, segments);
  const vertCount = n + 1; // centre + ring
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint16Array(n * 3);

  // Centre vertex (index 0)
  positions[0] = cx; positions[1] = cy; positions[2] = z;
  normals[2] = 1;
  uvs[0] = 0.5; uvs[1] = 0.5;

  const step = (Math.PI * 2) / n;
  for (let i = 0; i < n; i++) {
    const angle = i * step;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const vi = (i + 1) * 3;
    positions[vi]     = cx + cos * hw;
    positions[vi + 1] = cy + sin * hh;
    positions[vi + 2] = z;
    normals[vi + 2] = 1;
    const ui = (i + 1) * 2;
    uvs[ui]     = (cos + 1) * 0.5;
    uvs[ui + 1] = (sin + 1) * 0.5;
    // Fan triangle: centre → i+1 → i+2 (wrapping)
    const ii = i * 3;
    indices[ii]     = 0;
    indices[ii + 1] = i + 1;
    indices[ii + 2] = (i + 1) % n + 1;
  }

  return {
    positions,
    indices,
    normals,
    uvs,
    aabb: { minX: cx - hw, minY: cy - hh, maxX: cx + hw, maxY: cy + hh },
  };
}

/**
 * buildHex — regular hexagon (6 vertices + centre fan).
 */
function buildHex(
  cx: number,
  cy: number,
  z: number,
  hw: number,
  hh: number
): Pick<CellGeomResult, 'positions' | 'indices' | 'normals' | 'uvs' | 'aabb'> {
  return buildEllipse(cx, cy, z, hw, hh, 6);
}

/**
 * buildDiamond — rhombus (4 vertices, 2 triangles).
 *   Vertices: top, right, bottom, left.
 */
function buildDiamond(
  cx: number,
  cy: number,
  z: number,
  hw: number,
  hh: number
): Pick<CellGeomResult, 'positions' | 'indices' | 'normals' | 'uvs' | 'aabb'> {
  const positions = new Float32Array([
    cx,      cy - hh, z, // top
    cx + hw, cy,      z, // right
    cx,      cy + hh, z, // bottom
    cx - hw, cy,      z, // left
  ]);
  const indices = new Uint16Array([0, 1, 3, 1, 2, 3]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvs = new Float32Array([0.5, 0, 1, 0.5, 0.5, 1, 0, 0.5]);
  return {
    positions,
    indices,
    normals,
    uvs,
    aabb: { minX: cx - hw, minY: cy - hh, maxX: cx + hw, maxY: cy + hh },
  };
}

// ── Task handlers ─────────────────────────────────────────────────────────────

function handleComputeVertices(data: CellGeomInput): CellGeomOutput {
  const results: CellGeomResult[] = [];
  const transfer: ArrayBuffer[] = [];

  for (const cell of data.cells) {
    const {
      cell_id,
      x,
      y,
      z = 3,
      width,
      height,
      shape = 'rect',
      segments = 32,
    } = cell;

    const hw = width * 0.5;
    const hh = height * 0.5;

    let geom: Pick<CellGeomResult, 'positions' | 'indices' | 'normals' | 'uvs' | 'aabb'>;

    switch (shape) {
      case 'ellipse':
        geom = buildEllipse(x, y, z, hw, hh, segments);
        break;
      case 'hex':
        geom = buildHex(x, y, z, hw, hh);
        break;
      case 'diamond':
        geom = buildDiamond(x, y, z, hw, hh);
        break;
      case 'rect':
      default:
        geom = buildRect(x, y, z, hw, hh);
        break;
    }

    results.push({ cell_id, ...geom });

    // Collect transferables (zero-copy pass to main thread).
    transfer.push(
      geom.positions.buffer,
      geom.indices.buffer,
      geom.normals.buffer,
      geom.uvs.buffer
    );
  }

  return { results, transfer };
}

function handleComputeAABB(data: CellGeomInput): CellAABBOutput {
  const aabbs: CellAABBOutput['aabbs'] = {};
  for (const cell of data.cells) {
    const hw = cell.width * 0.5;
    const hh = cell.height * 0.5;
    aabbs[cell.cell_id] = {
      minX: cell.x - hw,
      minY: cell.y - hh,
      maxX: cell.x + hw,
      maxY: cell.y + hh,
    };
  }
  return { aabbs };
}

function handleComputeOutline(data: CellGeomInput): CellOutlineOutput {
  const outlines: CellOutlineOutput['outlines'] = {};
  const transfer: ArrayBuffer[] = [];

  for (const cell of data.cells) {
    const { cell_id, x, y, width, height, shape = 'rect', segments = 32 } = cell;
    const hw = width * 0.5;
    const hh = height * 0.5;
    const n = shape === 'ellipse' ? Math.max(3, segments) : shape === 'hex' ? 6 : 4;
    const pts = new Float32Array(n * 2);

    if (shape === 'rect') {
      pts.set([x - hw, y - hh, x + hw, y - hh, x + hw, y + hh, x - hw, y + hh]);
    } else if (shape === 'diamond') {
      pts.set([x, y - hh, x + hw, y, x, y + hh, x - hw, y]);
    } else {
      // Ellipse or hex — parametric ring
      const step = (Math.PI * 2) / n;
      for (let i = 0; i < n; i++) {
        const a = i * step;
        pts[i * 2]     = x + Math.cos(a) * hw;
        pts[i * 2 + 1] = y + Math.sin(a) * hh;
      }
    }

    outlines[cell_id] = pts;
    transfer.push(pts.buffer);
  }

  return { outlines, transfer };
}

// ── Worker message dispatcher ─────────────────────────────────────────────────
// Mirrors AT WorkerManager self.onmessage handler.
// Receives { uuid, method, data } → dispatches → postMessage { uuid, result }.

self.onmessage = function (event: MessageEvent) {
  const { uuid, method, data } = event.data as { uuid: number; method: string; data: CellGeomInput };

  try {
    let result: CellGeomOutput | CellAABBOutput | CellOutlineOutput;
    let transfer: Transferable[] = [];

    if (method === 'computeVertices') {
      const out = handleComputeVertices(data);
      result = out;
      transfer = out.transfer;
    } else if (method === 'computeAABB') {
      result = handleComputeAABB(data);
    } else if (method === 'computeOutline') {
      const out = handleComputeOutline(data);
      result = out;
      transfer = out.transfer;
    } else {
      throw new Error(`[geometry-worker] Unknown method: ${method}`);
    }

    self.postMessage({ uuid, result }, transfer);
  } catch (e) {
    self.postMessage({ uuid, error: String(e) });
  }
};
