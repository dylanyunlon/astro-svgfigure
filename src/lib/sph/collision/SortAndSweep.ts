// === src/lib/sph/collision/SortAndSweep.ts ===

/**
 * Sort and Sweep (SAP) – Broad-Phase Collision Detection
 *
 * Maintains two sorted endpoint arrays (X and Y axes).  Each endpoint is
 * stored as a pair of parallel typed arrays:
 *
 *   values  Float32Array  – coordinate value
 *   meta    Int32Array    – packed (bodyId << 1) | isMin
 *                          isMin = 1 → left/bottom edge
 *                          isMin = 0 → right/top   edge
 *
 * Sorting uses insertion sort, which is O(n + k) when the input is nearly
 * sorted (k = number of swaps).  Between simulation frames bodies move only
 * slightly, so k ≪ n in practice.
 *
 * During re-sort every swap is inspected:
 *   min crosses max  → new 1-D overlap starts  (add to overlap set)
 *   max crosses min  → 1-D overlap ends         (remove from overlap set)
 *
 * getPairs() returns pairs that overlap on *both* axes simultaneously.
 */

import type { AABB } from '../types';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Encode a pair (a, b) as a single integer key.  Order-independent. */
function pairKey(a: number, b: number, stride: number): number {
  return a < b ? a * stride + b : b * stride + a;
}

/**
 * Plain insertion sort (no overlap tracking).
 * Used for the very first sort on an unsorted endpoint array.
 */
function insertionSort(
  values: Float32Array,
  meta: Int32Array,
  n: number,
): void {
  for (let i = 1; i < n; i++) {
    const v = values[i];
    const m = meta[i];
    let j = i - 1;
    while (j >= 0 && values[j] > v) {
      values[j + 1] = values[j];
      meta[j + 1] = meta[j];
      j--;
    }
    values[j + 1] = v;
    meta[j + 1] = m;
  }
}

/**
 * Insertion sort with incremental overlap tracking via swap detection.
 *
 * Each swap represents one endpoint crossing another along the axis:
 *
 *   moving element sweeps LEFT (toward smaller indices) past element at j
 *
 *   case A:  moving.isMin && !passed.isMin
 *            → moving body's left edge just passed passed body's right edge
 *            → the two bodies now overlap on this axis  → ADD pair
 *
 *   case B:  !moving.isMin && passed.isMin
 *            → moving body's right edge just passed passed body's left edge
 *            → the two bodies no longer overlap on this axis → DELETE pair
 *
 * Note: same-body swaps (both endpoints of one body) are silently ignored.
 */
function insertionSortTracked(
  values: Float32Array,
  meta: Int32Array,
  n: number,
  overlapSet: Set<number>,
  stride: number,
): void {
  for (let i = 1; i < n; i++) {
    const v = values[i];
    const m = meta[i];
    const aid = m >> 1;
    const aIsMin = (m & 1) === 1;
    let j = i - 1;

    while (j >= 0 && values[j] > v) {
      const pm = meta[j];
      const bid = pm >> 1;
      const bIsMin = (pm & 1) === 1;

      if (aid !== bid) {
        if (aIsMin && !bIsMin) {
          // a's min crosses b's max: overlap starts
          overlapSet.add(pairKey(aid, bid, stride));
        } else if (!aIsMin && bIsMin) {
          // a's max crosses b's min: overlap ends
          overlapSet.delete(pairKey(aid, bid, stride));
        }
      }

      values[j + 1] = values[j];
      meta[j + 1] = meta[j];
      j--;
    }

    values[j + 1] = v;
    meta[j + 1] = m;
  }
}

/**
 * Scan a fully-sorted endpoint array and build the overlap set from scratch.
 * O(n + k) where k is the number of pairs found.
 */
function buildOverlapFromSorted(
  meta: Int32Array,
  n: number,
  overlapSet: Set<number>,
  stride: number,
): void {
  overlapSet.clear();
  // Active stack: body IDs whose min has been seen but max has not yet
  const active: number[] = [];

  for (let i = 0; i < n; i++) {
    const m = meta[i];
    const id = m >> 1;
    const isMin = (m & 1) === 1;

    if (isMin) {
      // Overlaps with every currently active body
      for (let k = 0; k < active.length; k++) {
        overlapSet.add(pairKey(id, active[k], stride));
      }
      active.push(id);
    } else {
      const idx = active.indexOf(id);
      if (idx !== -1) active.splice(idx, 1);
    }
  }
}

// ---------------------------------------------------------------------------
// SortAndSweep
// ---------------------------------------------------------------------------

export class SortAndSweep {
  /** Maximum number of bodies this instance was constructed for. */
  readonly maxBodies: number;

  // Endpoint arrays — two per body per axis (min + max)
  private readonly xValues: Float32Array;
  private readonly xMeta: Int32Array;
  private readonly yValues: Float32Array;
  private readonly yMeta: Int32Array;

  // Per-axis 1-D overlap sets (encoded pair keys)
  private readonly xOverlap: Set<number> = new Set();
  private readonly yOverlap: Set<number> = new Set();

  /**
   * Stride used for pair encoding.  Must be > maxBodies.
   * Using maxBodies itself is exact – a < b < maxBodies ensures unique keys.
   */
  private readonly stride: number;

  private count = 0;
  private initialised = false;

  constructor(maxBodies: number) {
    this.maxBodies = maxBodies;
    this.stride = maxBodies;
    const maxEp = maxBodies * 2;

    this.xValues = new Float32Array(maxEp);
    this.xMeta = new Int32Array(maxEp);
    this.yValues = new Float32Array(maxEp);
    this.yMeta = new Int32Array(maxEp);
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Update all endpoint values from the given AABB array, then re-sort.
   *
   * On the very first call the arrays are sorted from scratch and the overlap
   * sets are built by a linear scan of the sorted result (O(n log n) initial
   * cost, amortised O(n) for subsequent calls when bodies move little).
   *
   * IMPORTANT: bodies must be an array whose index equals body.id
   * (i.e. bodies[i].id === i).  This is the standard SPH convention.
   *
   * @param bodies  Array of AABBs indexed by their id.
   */
  update(bodies: readonly AABB[]): void {
    if (!this.initialised) {
      this._initialize(bodies);
      return;
    }

    const n2 = this.count * 2;

    // Scatter updated coordinate values back into the position-in-sorted-order
    // endpoint arrays.  The meta arrays still hold the correct bodyId/isMin
    // encoding even though order may have shifted; we just overwrite values.
    for (let i = 0; i < n2; i++) {
      const xm = this.xMeta[i];
      const xid = xm >> 1;
      const xIsMin = (xm & 1) === 1;
      const xb = bodies[xid];
      if (xb !== undefined) {
        this.xValues[i] = xIsMin ? xb.minX : xb.maxX;
      }

      const ym = this.yMeta[i];
      const yid = ym >> 1;
      const yIsMin = (ym & 1) === 1;
      const yb = bodies[yid];
      if (yb !== undefined) {
        this.yValues[i] = yIsMin ? yb.minY : yb.maxY;
      }
    }

    // Incremental insertion sort — O(n + k), very fast when k is small
    insertionSortTracked(
      this.xValues,
      this.xMeta,
      n2,
      this.xOverlap,
      this.stride,
    );
    insertionSortTracked(
      this.yValues,
      this.yMeta,
      n2,
      this.yOverlap,
      this.stride,
    );
  }

  /**
   * Return all body-pairs that overlap on **both** X and Y axes.
   * Each pair is [idA, idB] with idA < idB.
   */
  getPairs(): [number, number][] {
    const pairs: [number, number][] = [];
    const stride = this.stride;

    for (const key of this.xOverlap) {
      if (this.yOverlap.has(key)) {
        const hi = key % stride;
        const lo = (key / stride) | 0;
        pairs.push([lo, hi]);
      }
    }

    return pairs;
  }

  /**
   * Reset to an uninitialised state so the next `update()` call performs
   * a full re-initialisation (useful when the body set changes size).
   */
  reset(): void {
    this.count = 0;
    this.initialised = false;
    this.xOverlap.clear();
    this.yOverlap.clear();
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Full initialisation: scatter all endpoints, sort from scratch, build
   * overlap sets via a single linear scan of each sorted array.
   */
  private _initialize(bodies: readonly AABB[]): void {
    this.count = bodies.length;
    const n2 = this.count * 2;

    for (let i = 0; i < this.count; i++) {
      const b = bodies[i];
      const i2 = i * 2;
      this.xValues[i2] = b.minX;
      this.xMeta[i2] = (b.id << 1) | 1;
      this.xValues[i2 + 1] = b.maxX;
      this.xMeta[i2 + 1] = (b.id << 1) | 0;

      this.yValues[i2] = b.minY;
      this.yMeta[i2] = (b.id << 1) | 1;
      this.yValues[i2 + 1] = b.maxY;
      this.yMeta[i2 + 1] = (b.id << 1) | 0;
    }

    insertionSort(this.xValues, this.xMeta, n2);
    insertionSort(this.yValues, this.yMeta, n2);

    buildOverlapFromSorted(this.xMeta, n2, this.xOverlap, this.stride);
    buildOverlapFromSorted(this.yMeta, n2, this.yOverlap, this.stride);

    this.initialised = true;
  }
}
