/**
 * matrix-wasm.ts — Batch matrix math with JS fallback + WASM acceleration hooks
 *
 * Provides:
 *   MatrixWasm — singleton-style class for batched matrix operations used by
 *                the cell graph renderer pipeline.
 *
 *   multiply4x4(a, b, out)         — 4 × 4 column-major matrix multiply
 *   batchTransformVec2(...)         — SIMD-friendly 2-D point array transform
 *   batchAABBTest(boxes, viewport)  — frustum / viewport AABB cull in batch
 *
 * Architecture
 * ────────────
 * All three operations ship a pure-JS implementation today.
 * Each hot path is wrapped in a thin dispatcher that checks `wasmReady`:
 *
 *   if (this._wasm) {
 *     // ── WASM REPLACE POINT ──────────────────────────────────────────── //
 *     // Call into the compiled WASM module.                               //
 *     // See §WASM Integration Guide below.                                //
 *     // ─────────────────────────────────────────────────────────────────  //
 *   }
 *   // JS fallback follows
 *
 * § WASM Integration Guide
 * ─────────────────────────
 * 1. Compile `src/wasm/matrix_ops.wat` (or Rust/C source) to
 *    `public/matrix_ops.wasm`.
 * 2. Call `await MatrixWasm.loadWasm('/matrix_ops.wasm')` once at startup.
 * 3. The WASM module must export:
 *      multiply4x4(aPtr, bPtr, outPtr: i32): void
 *      batchTransformVec2(posPtr, matPtr, outPtr, n: i32): void
 *      batchAABBTest(boxPtr, vp0x, vp0y, vp1x, vp1y, outPtr, n: i32): void
 *    All pointers address the module's own linear memory (shared Float32Array).
 * 4. MatrixWasm.memory gives direct access to the WebAssembly.Memory for
 *    zero-copy data sharing.
 *
 * Upstream references:
 *   upstream/antimatter-gpu/src/compute/matrix.ts
 *   upstream/wasm-simd-test/src/f32x4_mat.wat
 *
 * AT references: MatrixWasm ×31, multiply4x4 ×18, batchTransformVec2 ×12,
 *                batchAABBTest ×7
 */

// ── Public types ──────────────────────────────────────────────────────────────

/**
 * An axis-aligned bounding box.
 * minX, minY, maxX, maxY — in the same coordinate space as the viewport.
 */
export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * A viewport rectangle used as the culling frustum for batchAABBTest().
 */
export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Result of batchAABBTest — one boolean per box. */
export type AABBVisibilityResult = boolean[];

// ── WASM module shape (kept minimal; extend when real WASM arrives) ───────────

interface WasmExports {
  memory: WebAssembly.Memory;
  multiply4x4(aPtr: number, bPtr: number, outPtr: number): void;
  batchTransformVec2(posPtr: number, matPtr: number, outPtr: number, n: number): void;
  batchAABBTest(
    boxPtr: number,
    vp0x: number, vp0y: number,
    vp1x: number, vp1y: number,
    outPtr: number,
    n: number
  ): void;
}

// ── MatrixWasm ────────────────────────────────────────────────────────────────

/**
 * MatrixWasm
 *
 * Provides batched matrix and vector math optimised for the cell graph
 * animation pipeline.  All methods accept and return Float32Array so callers
 * can share TypedArray views with WebGL buffers without extra allocation.
 *
 * @example
 * ```ts
 * // Load WASM (optional — falls back to JS automatically)
 * await MatrixWasm.loadWasm('/matrix_ops.wasm');
 *
 * // 4x4 multiply
 * const out = new Float32Array(16);
 * MatrixWasm.multiply4x4(matA, matB, out);
 *
 * // Bulk transform 2-D positions
 * const positions = new Float32Array([0, 0, 100, 50, 200, 150]);
 * const transformed = new Float32Array(6);
 * MatrixWasm.batchTransformVec2(positions, modelMatrix, transformed);
 *
 * // Cull invisible cells
 * const boxes: AABB[] = cells.map(c => getCellAABB(c));
 * const visible = MatrixWasm.batchAABBTest(boxes, camera.viewport);
 * ```
 */
export class MatrixWasm {
  // ── Singleton WASM state ────────────────────────────────────────────────────

  private static _wasm: WasmExports | null = null;
  private static _wasmMemory: WebAssembly.Memory | null = null;
  private static _wasmHeap: Float32Array | null = null;

  /** True once loadWasm() has resolved successfully. */
  static get wasmReady(): boolean {
    return MatrixWasm._wasm !== null;
  }

  /** Direct access to the WASM linear memory (null until loadWasm() resolves). */
  static get memory(): WebAssembly.Memory | null {
    return MatrixWasm._wasmMemory;
  }

  /**
   * Load and instantiate the WASM matrix-ops module.
   * Safe to call multiple times — subsequent calls are no-ops.
   *
   * @param url  Path to the `.wasm` binary (e.g. `/matrix_ops.wasm`).
   */
  static async loadWasm(url: string): Promise<void> {
    if (MatrixWasm._wasm) return; // already loaded

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`[MatrixWasm] fetch failed: ${response.status} ${response.statusText}`);
      }
      const buffer = await response.arrayBuffer();

      const { instance } = await WebAssembly.instantiate(buffer, {
        env: {
          // Reserved import slots — extend as required by the WASM module
          abort: () => { throw new Error('[MatrixWasm] WASM abort()'); },
        },
      });

      MatrixWasm._wasm = instance.exports as unknown as WasmExports;
      MatrixWasm._wasmMemory = MatrixWasm._wasm.memory;
      MatrixWasm._wasmHeap = new Float32Array(MatrixWasm._wasm.memory.buffer);

      console.info('[MatrixWasm] WASM module loaded successfully.');
    } catch (err) {
      console.warn('[MatrixWasm] Failed to load WASM — JS fallback active.', err);
      // _wasm stays null; JS fallback will be used
    }
  }

  // ── multiply4x4 ─────────────────────────────────────────────────────────────

  /**
   * Multiply two 4 × 4 column-major matrices.
   *
   *   out = a × b
   *
   * All three arrays must be length 16.  `out` may alias `a` or `b` safely
   * because the JS fallback writes to a temporary first.
   *
   * @param a    Left operand  (Float32Array, length 16)
   * @param b    Right operand (Float32Array, length 16)
   * @param out  Output buffer (Float32Array, length 16)
   */
  static multiply4x4(
    a: Readonly<Float32Array>,
    b: Readonly<Float32Array>,
    out: Float32Array
  ): void {
    // ── WASM REPLACE POINT ──────────────────────────────────────────────────
    // if (MatrixWasm._wasm && MatrixWasm._wasmHeap) {
    //   const heap = MatrixWasm._wasmHeap;
    //   const aPtr = 0, bPtr = 64, outPtr = 128;           // byte offsets / 4
    //   heap.set(a,   aPtr);
    //   heap.set(b,   bPtr);
    //   MatrixWasm._wasm.multiply4x4(aPtr * 4, bPtr * 4, outPtr * 4);
    //   out.set(heap.subarray(outPtr, outPtr + 16));
    //   return;
    // }
    // ── END WASM REPLACE POINT ──────────────────────────────────────────────

    // JS fallback — column-major 4×4 multiply
    const tmp = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          // a[k * 4 + row] * b[col * 4 + k]
          sum += a[k * 4 + row] * b[col * 4 + k];
        }
        tmp[col * 4 + row] = sum;
      }
    }
    out.set(tmp);
  }

  // ── batchTransformVec2 ───────────────────────────────────────────────────────

  /**
   * Apply a 4 × 4 column-major transform matrix to an array of 2-D positions.
   *
   * Positions are stored as interleaved (x, y) pairs:
   *   [x0, y0, x1, y1, … xN-1, yN-1]
   * `positions.length` must be even; `out.length` must equal `positions.length`.
   *
   * The z component is taken as 0 and w as 1 for each input point.
   * Only the x and y components of the output are written back.
   *
   * @param positions  Interleaved 2-D input  (Float32Array, even length)
   * @param matrix     4 × 4 column-major matrix (Float32Array, length 16)
   * @param out        Output buffer, same layout as positions
   */
  static batchTransformVec2(
    positions: Readonly<Float32Array>,
    matrix: Readonly<Float32Array>,
    out: Float32Array
  ): void {
    const n = positions.length >>> 1; // number of vec2 pairs

    // ── WASM REPLACE POINT ──────────────────────────────────────────────────
    // if (MatrixWasm._wasm && MatrixWasm._wasmHeap) {
    //   const heap   = MatrixWasm._wasmHeap;
    //   const posPtr = 0;
    //   const matPtr = posPtr + n * 2;        // float32 indices
    //   const outPtr = matPtr + 16;
    //   heap.set(positions, posPtr);
    //   heap.set(matrix,    matPtr);
    //   MatrixWasm._wasm.batchTransformVec2(
    //     posPtr * 4, matPtr * 4, outPtr * 4, n
    //   );
    //   out.set(heap.subarray(outPtr, outPtr + n * 2));
    //   return;
    // }
    // ── END WASM REPLACE POINT ──────────────────────────────────────────────

    // JS fallback — unroll the z=0, w=1 case for speed
    // Column-major layout:  matrix[col * 4 + row]
    const m00 = matrix[0],  m10 = matrix[4],  m30 = matrix[12];
    const m01 = matrix[1],  m11 = matrix[5],  m31 = matrix[13];

    for (let i = 0; i < n; i++) {
      const x = positions[i * 2];
      const y = positions[i * 2 + 1];
      out[i * 2]     = m00 * x + m10 * y + m30;
      out[i * 2 + 1] = m01 * x + m11 * y + m31;
    }
  }

  // ── batchAABBTest ────────────────────────────────────────────────────────────

  /**
   * Test each box in `boxes` against the given `viewport` rectangle.
   *
   * Returns a boolean array: `result[i]` is `true` iff `boxes[i]` intersects
   * the viewport (i.e. the cell is at least partially visible and should be
   * rendered).
   *
   * Overlap is inclusive on all four edges.
   *
   * @param boxes     Array of AABB objects to test.
   * @param viewport  The camera / canvas viewport rectangle.
   */
  static batchAABBTest(boxes: readonly AABB[], viewport: Viewport): AABBVisibilityResult {
    const vx0 = viewport.x;
    const vy0 = viewport.y;
    const vx1 = viewport.x + viewport.width;
    const vy1 = viewport.y + viewport.height;
    const n   = boxes.length;

    // ── WASM REPLACE POINT ──────────────────────────────────────────────────
    // if (MatrixWasm._wasm && MatrixWasm._wasmHeap) {
    //   const heap   = MatrixWasm._wasmHeap;
    //   const boxPtr = 0; // float32 index; each box = 4 floats
    //   for (let i = 0; i < n; i++) {
    //     const b = boxes[i];
    //     heap[boxPtr + i * 4 + 0] = b.minX;
    //     heap[boxPtr + i * 4 + 1] = b.minY;
    //     heap[boxPtr + i * 4 + 2] = b.maxX;
    //     heap[boxPtr + i * 4 + 3] = b.maxY;
    //   }
    //   const outPtr = boxPtr + n * 4;
    //   MatrixWasm._wasm.batchAABBTest(
    //     boxPtr * 4, vx0, vy0, vx1, vy1, outPtr * 4, n
    //   );
    //   const result: boolean[] = new Array(n);
    //   for (let i = 0; i < n; i++) result[i] = heap[outPtr + i] !== 0;
    //   return result;
    // }
    // ── END WASM REPLACE POINT ──────────────────────────────────────────────

    // JS fallback — classic AABB overlap test
    const result: boolean[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const b = boxes[i];
      result[i] = b.maxX >= vx0 && b.minX <= vx1 &&
                  b.maxY >= vy0 && b.minY <= vy1;
    }
    return result;
  }

  // ── Utility helpers ──────────────────────────────────────────────────────────

  /**
   * Create a 4 × 4 identity matrix (column-major Float32Array).
   *
   * Convenience factory so callers don't need to remember the column-major
   * index layout.
   */
  static identity(): Float32Array {
    return new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  }

  /**
   * Build a 2-D translation matrix (4 × 4, column-major).
   *
   * @param tx  Translation along X.
   * @param ty  Translation along Y.
   */
  static translation2D(tx: number, ty: number): Float32Array {
    return new Float32Array([
      1,  0,  0,  0,
      0,  1,  0,  0,
      0,  0,  1,  0,
      tx, ty, 0,  1,
    ]);
  }

  /**
   * Build a 2-D scale matrix (4 × 4, column-major).
   *
   * @param sx  X scale factor.
   * @param sy  Y scale factor.
   */
  static scale2D(sx: number, sy: number): Float32Array {
    return new Float32Array([
      sx, 0,  0,  0,
      0,  sy, 0,  0,
      0,  0,  1,  0,
      0,  0,  0,  1,
    ]);
  }

  /**
   * Build a 2-D rotation matrix (4 × 4, column-major).
   *
   * @param angleRad  Rotation in radians (counter-clockwise).
   */
  static rotation2D(angleRad: number): Float32Array {
    const c = Math.cos(angleRad);
    const s = Math.sin(angleRad);
    return new Float32Array([
       c, s, 0, 0,
      -s, c, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1,
    ]);
  }

  /**
   * Compose a TRS (translate × rotate × scale) matrix for 2-D cell transforms.
   *
   * @param tx   Translation X.
   * @param ty   Translation Y.
   * @param r    Rotation in radians.
   * @param sx   Scale X.
   * @param sy   Scale Y.
   * @param out  Optional pre-allocated output buffer (Float32Array, length 16).
   */
  static trs2D(
    tx: number,
    ty: number,
    r: number,
    sx: number,
    sy: number,
    out?: Float32Array
  ): Float32Array {
    const c  = Math.cos(r);
    const s  = Math.sin(r);
    const result = out ?? new Float32Array(16);

    // Column 0
    result[0]  = c * sx;
    result[1]  = s * sx;
    result[2]  = 0;
    result[3]  = 0;
    // Column 1
    result[4]  = -s * sy;
    result[5]  =  c * sy;
    result[6]  = 0;
    result[7]  = 0;
    // Column 2
    result[8]  = 0;
    result[9]  = 0;
    result[10] = 1;
    result[11] = 0;
    // Column 3
    result[12] = tx;
    result[13] = ty;
    result[14] = 0;
    result[15] = 1;

    return result;
  }

  /**
   * Compute the AABB of a set of 2-D points stored in an interleaved
   * Float32Array ([x0, y0, x1, y1, …]).
   *
   * Useful for deriving cell AABBs after batchTransformVec2().
   */
  static computeAABB(points: Readonly<Float32Array>): AABB {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    const n = points.length >>> 1;
    for (let i = 0; i < n; i++) {
      const x = points[i * 2];
      const y = points[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return { minX, minY, maxX, maxY };
  }
}
