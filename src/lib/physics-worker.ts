/**
 * physics-worker.ts — WebWorker thread: sweep-line collision detection + force_field computation.
 *
 * Ported from loop_orchestrator.py:
 *   _sweep_line_overlaps()  → FAstroCellSweepLine (SceneSoftwareOcclusion.cpp 0b4b199)
 *   physics_engine()        → force_field accumulation (TiledDeferredLightRendering.cpp 7c82b90)
 *
 * Message protocol:
 *   IN  { cells: CellBBox[] }
 *   OUT { force_field: Record<string, { dx: number; dy: number; dz: number }> }
 *
 * Runs entirely off the main thread — zero DOM / fetch access.
 */

// ─── Types ─────────────────────────────────────────────────────────────────────

export interface CellBBox {
  /** Unique cell identifier, e.g. "attention1". */
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** Z-layer integer (same-z cells only collide). Default 3. */
  z?: number;
}

export interface ForceVector {
  dx: number;
  dy: number;
  dz: number;
}

export type ForceField = Record<string, ForceVector>;

/** Internal sweep-line event. Mirrors FSweepEvent in SceneSoftwareOcclusion.cpp. */
interface SweepEvent {
  x: number;
  /** true = OPEN (min_x), false = CLOSE (max_x). Open sorts before close at equal x. */
  isOpen: boolean;
  /** Index into the rects array for this z-layer. */
  idx: number;
}

/** Internal rect tuple: [id, min_x, min_y, max_x, max_y]. */
type Rect = [string, number, number, number, number];

/** Overlap pair returned by the sweep-line pass. */
interface OverlapPair {
  a: string;
  b: string;
  overlapX: number;
  overlapY: number;
  z: number;
}

// ─── Sweep-line collision detection ────────────────────────────────────────────
//
// FAstroCellSweepLine — O((N+K) log N) sweep-line 2D overlap detection.
// Ported from upstream/unreal-renderer/SceneSoftwareOcclusion.cpp commit 0b4b199.
//
// Algorithm:
//   1. For every cell bbox emit an OPEN event at min_x and a CLOSE event at max_x.
//   2. Sort all events by x; ties: OPEN before CLOSE so touching edges are caught.
//   3. Scan events left→right, maintaining an active set of currently-open rects.
//   4. On OPEN: insert into active set; test new rect against every active rect for
//      Y-interval overlap → emit collision pair.
//   5. On CLOSE: remove from active set.
//   6. Cells on different z-layers never collide (same rule as the original O(N²)).

function sweepLineOverlaps(
  rectsByZ: Map<number, Rect[]>
): OverlapPair[] {
  const pairs: OverlapPair[] = [];

  for (const [zLayer, rects] of rectsByZ) {
    if (rects.length < 2) continue;

    // Build events — two per rect.
    // Sort key: (x, isOpen ? 0 : 1) — OPEN sorts before CLOSE at equal x.
    // Mirrors FSweepEvent.bOpen sort trick in SceneSoftwareOcclusion.cpp.
    const events: SweepEvent[] = [];
    for (let i = 0; i < rects.length; i++) {
      const [, minX, minY, maxX, maxY] = rects[i];
      if (minX > maxX || minY > maxY) continue; // degenerate guard
      events.push({ x: minX, isOpen: true,  idx: i });
      events.push({ x: maxX, isOpen: false, idx: i });
    }

    // Sort: ascending x; at equal x OPEN (0) before CLOSE (1).
    events.sort((a, b) => a.x !== b.x ? a.x - b.x : (a.isOpen ? -1 : 1));

    // Active set: indices of rects whose x-interval spans current sweep position.
    const active: number[] = [];

    for (const ev of events) {
      const [cellId, minX, minY, maxX, maxY] = rects[ev.idx];

      if (ev.isOpen) {
        // Test new rect against every currently-active rect for Y overlap.
        for (const aidx of active) {
          const [aCellId, aMinX, aMinY, aMaxX, aMaxY] = rects[aidx];

          // Y-interval overlap test (mirrors OR.MaxY >= OccludeeMinY && OR.MinY <= OccludeeMaxY)
          if (aMinY <= maxY && aMaxY >= minY) {
            const ovY = Math.min(maxY, aMaxY) - Math.max(minY, aMinY);
            const ovX = Math.min(maxX, aMaxX) - Math.max(minX, aMinX);
            if (ovX > 0 && ovY > 0) {
              pairs.push({ a: cellId, b: aCellId, overlapX: ovX, overlapY: ovY, z: zLayer });
            }
          }
        }
        active.push(ev.idx);
      } else {
        // CLOSE: remove from active set (mirrors RemoveSingleSwap).
        const pos = active.indexOf(ev.idx);
        if (pos !== -1) active.splice(pos, 1);
      }
    }
  }

  return pairs;
}

// ─── Force field computation ────────────────────────────────────────────────────
//
// Tiled constraint solver — O(N·K) spatial partition.
// Ported from FAstroTiledConstraintSolver (TiledDeferredLightRendering.cpp 7c82b90).
//
// Concept mapping (Unreal → physics-worker):
//   Light                → Cell constraint (bbox overlap pair)
//   Screen tile          → Canvas tile region (TILE_SIZE × TILE_SIZE px)
//   Compute thread group → Constraint solver (single worker thread here)
//   LightData CBs        → ConstraintBatch per tile

const TILE_SIZE = 16;

function computeForceField(
  cells: CellBBox[]
): ForceField {
  // Initialise force_field for every cell.
  const forceField: ForceField = {};
  for (const c of cells) {
    forceField[c.id] = { dx: 0, dy: 0, dz: 0 };
  }

  if (cells.length < 2) return forceField;

  // Build bbox map for quick lookup.
  const bboxMap = new Map<string, CellBBox>();
  for (const c of cells) bboxMap.set(c.id, c);

  // ── Phase 1: bucket cells by z-layer for sweep-line ─────────────────────────
  const rectsByZ = new Map<number, Rect[]>();
  for (const c of cells) {
    const z = c.z ?? 3;
    if (!rectsByZ.has(z)) rectsByZ.set(z, []);
    rectsByZ.get(z)!.push([c.id, c.x, c.y, c.x + c.w, c.y + c.h]);
  }

  // ── Phase 2: sweep-line overlap detection (diagnostic pass) ─────────────────
  // The sweep-line pass enumerates pairs; force accumulation is done by the tiled
  // solver below (same two-pass split as physics_engine() in loop_orchestrator.py).
  sweepLineOverlaps(rectsByZ); // result unused for force — run for diagnostic parity

  // ── Phase 3: tiled constraint solver ─────────────────────────────────────────
  // Auto-detect canvas bounds from bbox extents (mirrors GridWorldBox construction).
  let maxX = 0, maxY = 0;
  for (const c of cells) {
    if (c.x + c.w > maxX) maxX = c.x + c.w;
    if (c.y + c.h > maxY) maxY = c.y + c.h;
  }
  const canvasW = Math.max(Math.ceil(maxX), TILE_SIZE);
  const canvasH = Math.max(Math.ceil(maxY), TILE_SIZE);

  const numTilesX = Math.ceil(canvasW / TILE_SIZE);
  const numTilesY = Math.ceil(canvasH / TILE_SIZE);

  // Assign each cell to a tile by bbox centre (mirrors BuildTileBatches in C++).
  const tileCells: string[][][] = Array.from({ length: numTilesY }, () =>
    Array.from({ length: numTilesX }, () => [])
  );

  for (const c of cells) {
    const cx = c.x + c.w * 0.5;
    const cy = c.y + c.h * 0.5;
    const tx = Math.max(0, Math.min(numTilesX - 1, Math.floor(cx / TILE_SIZE)));
    const ty = Math.max(0, Math.min(numTilesY - 1, Math.floor(cy / TILE_SIZE)));
    tileCells[ty][tx].push(c.id);
  }

  // Resolve constraints per tile (mirrors FAstroTiledConstraintSolver::ResolveAll).
  const resolvedPairs = new Set<string>();

  for (let ty = 0; ty < numTilesY; ty++) {
    for (let tx = 0; tx < numTilesX; tx++) {
      // Gather 3×3 neighbourhood.
      const neighbourhood: string[] = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ntx = tx + dx, nty = ty + dy;
          if (ntx >= 0 && ntx < numTilesX && nty >= 0 && nty < numTilesY) {
            neighbourhood.push(...tileCells[nty][ntx]);
          }
        }
      }

      if (neighbourhood.length < 2) continue;

      // Test all O(K²) pairs within the neighbourhood.
      for (let i = 0; i < neighbourhood.length; i++) {
        for (let j = i + 1; j < neighbourhood.length; j++) {
          const idA = neighbourhood[i];
          const idB = neighbourhood[j];

          // Canonical pair key — smaller id first (dedup across shared tile boundaries).
          const pairKey = idA < idB ? `${idA}|${idB}` : `${idB}|${idA}`;
          if (resolvedPairs.has(pairKey)) continue;

          const ba = bboxMap.get(idA)!;
          const bb = bboxMap.get(idB)!;

          // Same-z layer only.
          if ((ba.z ?? 3) !== (bb.z ?? 3)) continue;

          const aMinX = ba.x, aMaxX = ba.x + ba.w;
          const aMinY = ba.y, aMaxY = ba.y + ba.h;
          const bMinX = bb.x, bMaxX = bb.x + bb.w;
          const bMinY = bb.y, bMaxY = bb.y + bb.h;

          const ovX = Math.min(aMaxX, bMaxX) - Math.max(aMinX, bMinX);
          const ovY = Math.min(aMaxY, bMaxY) - Math.max(aMinY, bMinY);

          if (ovX <= 0 || ovY <= 0) continue;

          resolvedPairs.add(pairKey);

          // Repulsion force — axis of least overlap (same formula as loop_orchestrator.py).
          if (ovX < ovY) {
            const push = ovX / 2 + 5;
            if (ba.x < bb.x) {
              forceField[idA].dx -= push;
              forceField[idB].dx += push;
            } else {
              forceField[idA].dx += push;
              forceField[idB].dx -= push;
            }
          } else {
            const push = ovY / 2 + 5;
            if (ba.y < bb.y) {
              forceField[idA].dy -= push;
              forceField[idB].dy += push;
            } else {
              forceField[idA].dy += push;
              forceField[idB].dy -= push;
            }
          }
        }
      }
    }
  }

  return forceField;
}

// ─── Worker message handler ─────────────────────────────────────────────────────

/**
 * onmessage entry point.
 *
 * Receives  { cells: CellBBox[] }
 * Posts     { force_field: ForceField }
 *
 * Error path: posts { error: string } so the bridge Promise can reject cleanly.
 */
self.onmessage = (ev: MessageEvent<{ id: number; cells: CellBBox[] }>) => {
  const { id, cells } = ev.data;
  try {
    if (!Array.isArray(cells)) {
      throw new TypeError('physics-worker: expected { cells: CellBBox[] }');
    }
    const force_field = computeForceField(cells);
    self.postMessage({ id, force_field });
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};

export {};
