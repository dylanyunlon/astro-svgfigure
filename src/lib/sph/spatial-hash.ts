/**
 * spatial-hash.ts
 * O(N) spatial hash grid for SPH neighbor search.
 *
 * cellSize = h (smoothing radius), so a 3x3 cell query covers all
 * particles within distance 2h — the full kernel support.
 *
 * Hash: ((floor(x/cs)*73856093) ^ (floor(y/cs)*19349663)) & (tableSize-1)
 * tableSize must be a power of 2 (default 4096).
 *
 * Implementation uses flat Int32Array chains to avoid GC pressure from
 * linked-list or per-bucket Array allocations.
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Next power of two >= n (n must be > 0). */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Spatial hash for a single (cx, cy) cell coordinate.
 * Uses unsigned 32-bit arithmetic via |0 and >>> tricks.
 */
function cellHash(cx: number, cy: number, mask: number): number {
  // Multiply with large primes, XOR, mask to table index.
  // All arithmetic kept in signed 32-bit int range then masked.
  const hx = Math.imul(cx | 0, 73856093);
  const hy = Math.imul(cy | 0, 19349663);
  return (hx ^ hy) & mask;
}

// ---------------------------------------------------------------------------
// SpatialHashGrid
// ---------------------------------------------------------------------------

export class SpatialHashGrid {
  private readonly cellSize: number;
  private readonly tableSize: number;
  private readonly mask: number;

  /**
   * heads[bucket] = index into `chains` of the first entry in that bucket,
   * or -1 if empty. Reset to -1 on every clear().
   */
  private readonly heads: Int32Array;

  /**
   * Flat parallel arrays replacing a linked list:
   *   chains_idx[i]  = particle index stored at chain node i
   *   chains_next[i] = index of next chain node, or -1 for end
   *
   * Both grow dynamically (doubling) but are reused across frames.
   */
  private chains_idx: Int32Array;
  private chains_next: Int32Array;
  private chainLen: number = 0; // next free slot

  // Scratch buffer for query() — avoids allocation per query call.
  private queryScratch: number[] = [];

  constructor(cellSize: number, tableSize: number = 4096) {
    if (cellSize <= 0) throw new RangeError("cellSize must be > 0");
    this.cellSize = cellSize;
    this.tableSize = nextPow2(tableSize);
    this.mask = this.tableSize - 1;

    this.heads = new Int32Array(this.tableSize).fill(-1);

    // Pre-allocate chain storage for 1024 particles; grows as needed.
    const initCap = 1024;
    this.chains_idx = new Int32Array(initCap);
    this.chains_next = new Int32Array(initCap);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Clear all entries. O(tableSize) — fills heads with -1.
   * Chain storage length is reset to 0 (capacity kept for reuse).
   */
  clear(): void {
    this.heads.fill(-1);
    this.chainLen = 0;
  }

  /**
   * Insert particle `idx` at world position (x, y).
   * O(1) amortised (occasional doubling of chain storage).
   */
  insert(idx: number, x: number, y: number): void {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    const bucket = cellHash(cx, cy, this.mask);

    // Grow chain storage if needed.
    if (this.chainLen === this.chains_idx.length) {
      this._growChains();
    }

    const node = this.chainLen++;
    this.chains_idx[node] = idx;
    this.chains_next[node] = this.heads[bucket]; // prepend
    this.heads[bucket] = node;
  }

  /**
   * Return all particle indices whose cell falls within the 3x3 neighbourhood
   * of cell containing (x, y). Covers all particles within distance 2h when
   * cellSize == h.
   *
   * Returns an internal scratch array — copy if you need to retain across
   * multiple query() calls.
   */
  query(x: number, y: number): number[] {
    const result = this.queryScratch;
    result.length = 0;

    const cx0 = Math.floor(x / this.cellSize) - 1;
    const cy0 = Math.floor(y / this.cellSize) - 1;

    for (let dx = 0; dx < 3; dx++) {
      for (let dy = 0; dy < 3; dy++) {
        const bucket = cellHash(cx0 + dx, cy0 + dy, this.mask);
        let node = this.heads[bucket];
        while (node !== -1) {
          result.push(this.chains_idx[node]);
          node = this.chains_next[node];
        }
      }
    }

    return result;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private _growChains(): void {
    const oldCap = this.chains_idx.length;
    const newCap = oldCap * 2;

    const newIdx = new Int32Array(newCap);
    const newNext = new Int32Array(newCap);
    newIdx.set(this.chains_idx);
    newNext.set(this.chains_next);

    this.chains_idx = newIdx;
    this.chains_next = newNext;
  }
}

// ---------------------------------------------------------------------------
// buildNeighborLists — convenience wrapper
// ---------------------------------------------------------------------------

/**
 * Build full neighbour lists for all N particles in one pass.
 *
 * @param positions  Interleaved flat array [x0,y0, x1,y1, ..., x_{N-1},y_{N-1}]
 * @param count      Number of particles N
 * @param h          Smoothing radius (cellSize = h)
 * @returns          neighbors[i] = sorted array of neighbor indices (may include i itself)
 *
 * Complexity: O(N * avgNeighbors)  ≈  O(N) for typical SPH densities.
 */
export function buildNeighborLists(
  positions: Float64Array,
  count: number,
  h: number
): number[][] {
  // Reuse a single grid instance across calls by recreating only when h changes.
  // For simplicity here we construct fresh; callers with hot loops should
  // cache the SpatialHashGrid and call clear() + insert() themselves.
  const tableSize = nextPow2(Math.max(4096, count * 2));
  const grid = new SpatialHashGrid(h, tableSize);

  // --- Phase 1: insert all particles ---
  for (let i = 0; i < count; i++) {
    grid.insert(i, positions[i * 2], positions[i * 2 + 1]);
  }

  const h2 = h * h; // squared smoothing radius for exact distance test
  const neighbors: number[][] = new Array(count);

  // --- Phase 2: query each particle ---
  for (let i = 0; i < count; i++) {
    const xi = positions[i * 2];
    const yi = positions[i * 2 + 1];

    // query() returns internal scratch — copy candidates immediately.
    const candidates = grid.query(xi, yi);
    const list: number[] = [];

    for (let k = 0; k < candidates.length; k++) {
      const j = candidates[k];
      const dx = positions[j * 2] - xi;
      const dy = positions[j * 2 + 1] - yi;
      if (dx * dx + dy * dy <= h2) {
        list.push(j);
      }
    }

    neighbors[i] = list;
  }

  return neighbors;
}
