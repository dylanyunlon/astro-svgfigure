/**
 * physics-worker.ts — WebWorker thread: sweep-line collision detection + force_field computation.
 *
 * Ported from loop_orchestrator.py:
 *   _sweep_line_overlaps()  → FAstroCellSweepLine (SceneSoftwareOcclusion.cpp 0b4b199)
 *   physics_engine()        → force_field accumulation (TiledDeferredLightRendering.cpp 7c82b90)
 *
 * Comlink API (exposed via comlink.expose):
 *   computeForceField(cells: CellBBox[])           → ForceField
 *   detectCollisions(cells: CellBBox[])             → OverlapPair[]
 *   checkConvergence(forceField: ForceField, eps?)  → boolean
 *
 * Runs entirely off the main thread — zero DOM / fetch access.
 */

import { expose } from '../../upstream/comlink/src/comlink';

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
export interface OverlapPair {
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

function sweepLineOverlaps(
  rectsByZ: Map<number, Rect[]>
): OverlapPair[] {
  const pairs: OverlapPair[] = [];

  for (const [zLayer, rects] of rectsByZ) {
    if (rects.length < 2) continue;

    const events: SweepEvent[] = [];
    for (let i = 0; i < rects.length; i++) {
      const [, minX, minY, maxX, maxY] = rects[i];
      if (minX > maxX || minY > maxY) continue;
      events.push({ x: minX, isOpen: true,  idx: i });
      events.push({ x: maxX, isOpen: false, idx: i });
    }

    events.sort((a, b) => a.x !== b.x ? a.x - b.x : (a.isOpen ? -1 : 1));

    const active: number[] = [];

    for (const ev of events) {
      const [cellId, minX, minY, maxX, maxY] = rects[ev.idx];

      if (ev.isOpen) {
        for (const aidx of active) {
          const [aCellId, aMinX, aMinY, aMaxX, aMaxY] = rects[aidx];

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

const TILE_SIZE = 16;

function _computeForceField(
  cells: CellBBox[]
): ForceField {
  const forceField: ForceField = {};
  for (const c of cells) {
    forceField[c.id] = { dx: 0, dy: 0, dz: 0 };
  }

  if (cells.length < 2) return forceField;

  const bboxMap = new Map<string, CellBBox>();
  for (const c of cells) bboxMap.set(c.id, c);

  const rectsByZ = new Map<number, Rect[]>();
  for (const c of cells) {
    const z = c.z ?? 3;
    if (!rectsByZ.has(z)) rectsByZ.set(z, []);
    rectsByZ.get(z)!.push([c.id, c.x, c.y, c.x + c.w, c.y + c.h]);
  }

  sweepLineOverlaps(rectsByZ); // diagnostic pass

  let maxX = 0, maxY = 0;
  for (const c of cells) {
    if (c.x + c.w > maxX) maxX = c.x + c.w;
    if (c.y + c.h > maxY) maxY = c.y + c.h;
  }
  const canvasW = Math.max(Math.ceil(maxX), TILE_SIZE);
  const canvasH = Math.max(Math.ceil(maxY), TILE_SIZE);

  const numTilesX = Math.ceil(canvasW / TILE_SIZE);
  const numTilesY = Math.ceil(canvasH / TILE_SIZE);

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

  const resolvedPairs = new Set<string>();

  for (let ty = 0; ty < numTilesY; ty++) {
    for (let tx = 0; tx < numTilesX; tx++) {
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

      for (let i = 0; i < neighbourhood.length; i++) {
        for (let j = i + 1; j < neighbourhood.length; j++) {
          const idA = neighbourhood[i];
          const idB = neighbourhood[j];

          const pairKey = idA < idB ? `${idA}|${idB}` : `${idB}|${idA}`;
          if (resolvedPairs.has(pairKey)) continue;

          const ba = bboxMap.get(idA)!;
          const bb = bboxMap.get(idB)!;

          if ((ba.z ?? 3) !== (bb.z ?? 3)) continue;

          const aMinX = ba.x, aMaxX = ba.x + ba.w;
          const aMinY = ba.y, aMaxY = ba.y + ba.h;
          const bMinX = bb.x, bMaxX = bb.x + bb.w;
          const bMinY = bb.y, bMaxY = bb.y + bb.h;

          const ovX = Math.min(aMaxX, bMaxX) - Math.max(aMinX, bMinX);
          const ovY = Math.min(aMaxY, bMaxY) - Math.max(aMinY, bMinY);

          if (ovX <= 0 || ovY <= 0) continue;

          resolvedPairs.add(pairKey);

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

// ─── Comlink-exposed API ────────────────────────────────────────────────────────
//
// expose() replaces the manual self.onmessage handler.
// Main thread calls: const proxy = wrap<PhysicsWorker>(worker);
//   await proxy.computeForceField(cells)
//   await proxy.detectCollisions(cells)
//   await proxy.checkConvergence(forceField, eps?)

const PhysicsWorkerAPI = {
  /**
   * computeForceField — run tiled constraint solver + sweep-line diagnostic pass.
   * Returns accumulated repulsion force for each cell.
   */
  computeForceField(cells: CellBBox[]): ForceField {
    if (!Array.isArray(cells)) {
      throw new TypeError('physics-worker: expected CellBBox[]');
    }
    return _computeForceField(cells);
  },

  /**
   * detectCollisions — run sweep-line overlap detection only.
   * Returns all overlapping pairs for a given set of cells.
   */
  detectCollisions(cells: CellBBox[]): OverlapPair[] {
    if (!Array.isArray(cells)) {
      throw new TypeError('physics-worker: expected CellBBox[]');
    }
    const rectsByZ = new Map<number, Rect[]>();
    for (const c of cells) {
      const z = c.z ?? 3;
      if (!rectsByZ.has(z)) rectsByZ.set(z, []);
      rectsByZ.get(z)!.push([c.id, c.x, c.y, c.x + c.w, c.y + c.h]);
    }
    return sweepLineOverlaps(rectsByZ);
  },

  /**
   * checkConvergence — returns true when all force magnitudes are below `eps`.
   * Mirrors the convergence check in loop_orchestrator.py physics_engine().
   */
  checkConvergence(forceField: ForceField, eps = 1.0): boolean {
    for (const fv of Object.values(forceField)) {
      if (Math.abs(fv.dx) > eps || Math.abs(fv.dy) > eps || Math.abs(fv.dz) > eps) {
        return false;
      }
    }
    return true;
  },
};

// Replace manual self.onmessage with comlink expose().
expose(PhysicsWorkerAPI);

export type PhysicsWorkerAPI = typeof PhysicsWorkerAPI;
