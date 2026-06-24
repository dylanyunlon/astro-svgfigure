// === SpatialHashGrid.ts ===

// ============================================================
//  SpatialHashGrid.ts --- 2D Spatial Hash for SPH Neighbor Search
//  Strategy: cell-chaining linked list (zero GC per frame)
//  All storage: typed arrays only --- no Map / Set / object
// ============================================================







import { MAX_PARTICLES } from './types';

export class SpatialHashGrid {
  private readonly tableSize: number;   // power-of-2
  private readonly mask: number;        // tableSize - 1
  private readonly head: Int32Array;    // head[hash] = first particle index (-1 = empty)
  private readonly next: Int32Array;    // next[i]    = next particle in same bucket (-1 = end)

  /**
   * @param tableSize  Hash table capacity, rounded up to next power-of-2.
   *                   Recommended: next power-of-2 >= expected particle count.
   *                   Defaults to 131072 (good for up to ~100k particles).
   */
  constructor(tableSize = 131072) {
    let ts = 1;
    while (ts < tableSize) ts <<= 1;
    this.tableSize = ts;
    this.mask      = ts - 1;
    this.head      = new Int32Array(ts).fill(-1);
    this.next      = new Int32Array(MAX_PARTICLES).fill(-1);
  }

  // ----------------------------------------------------------
  //  Spatial hash of integer cell coordinates (cx, cy).
  //  Two large primes reduce axis-aligned clustering artifacts.
  // ----------------------------------------------------------
  private hash(cx: number, cy: number): number {
    return (((cx * 92837111) ^ (cy * 689287499)) & this.mask) >>> 0;
  }

  // ----------------------------------------------------------
  //  clear()
  //  Reset every bucket head to -1.  O(tableSize).
  //  Call once per frame before insertAll().
  // ----------------------------------------------------------
  clear(): void {
    this.head.fill(-1);
  }

  // ----------------------------------------------------------
  //  insertAll(px, py, n, cellSize)
  //  Insert the first n particles into the grid.
  //
  //  cellSize: grid cell edge length.  Set equal to the SPH
  //  smoothing radius h so each query only visits 3--3 = 9 cells.
  // ----------------------------------------------------------
  insertAll(
    px: Float32Array,
    py: Float32Array,
    n: number,
    cellSize: number
  ): void {
    const invCell = 1.0 / cellSize;
    const head    = this.head;
    const next    = this.next;
    const mask    = this.mask;

    for (let i = 0; i < n; i++) {
      const cx = Math.floor(px[i] * invCell) | 0;
      const cy = Math.floor(py[i] * invCell) | 0;
      const h  = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
      next[i]  = head[h];  // push-front: new node -> old head
      head[h]  = i;
    }
  }

  // ----------------------------------------------------------
  //  forEachNeighbor(i, px, py, radius, cellSize, cb)
  //
  //  Walk every grid cell overlapping the circle of given radius
  //  centred on particle i at (px[i], py[i]).  For each candidate
  //  particle j (including i itself) within exact radius^2,
  //  invoke cb(j).
  //
  //  No allocation.  cb is called synchronously; return value
  //  is the total neighbor count (including self).
  // ----------------------------------------------------------
  forEachNeighbor(
    i: number,
    px: Float32Array,
    py: Float32Array,
    radius: number,
    cellSize: number,
    cb: (j: number) => void
  ): number {
    const ox      = px[i];
    const oy      = py[i];
    const r2      = radius * radius;
    const invCell = 1.0 / cellSize;
    const head    = this.head;
    const next    = this.next;
    const mask    = this.mask;

    const cxMin = Math.floor((ox - radius) * invCell) | 0;
    const cxMax = Math.floor((ox + radius) * invCell) | 0;
    const cyMin = Math.floor((oy - radius) * invCell) | 0;
    const cyMax = Math.floor((oy + radius) * invCell) | 0;

    let count = 0;

    for (let cy = cyMin; cy <= cyMax; cy++) {
      for (let cx = cxMin; cx <= cxMax; cx++) {
        const h = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
        let j = head[h];
        while (j !== -1) {
          const dx = px[j] - ox;
          const dy = py[j] - oy;
          if (dx * dx + dy * dy <= r2) {
            cb(j);
            count++;
          }
          j = next[j];
        }
      }
    }

    return count;
  }

  // ----------------------------------------------------------
  //  Accessors --- expose internals for GPU upload / debugging
  // ----------------------------------------------------------
  getHead():      Int32Array { return this.head; }
  getNext():      Int32Array { return this.next; }
  getTableSize(): number     { return this.tableSize; }
}
