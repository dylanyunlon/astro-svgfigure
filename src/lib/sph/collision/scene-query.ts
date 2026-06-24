


import { BVHTree } from './bvh-tree';
import { AABB } from './aabb-manager';

export interface RayHit {
  bodyId: number;
  t: number;
  hitX: number;
  hitY: number;
  normalX: number;
  normalY: number;
}

export interface OverlapResult {
  bodyIds: number[];
}

export interface ClosestPointResult {
  bodyId: number;
  pointX: number;
  pointY: number;
  distance: number;
}

interface BodyData {
  cx: number;
  cy: number;
  hw: number;
  hh: number;
  angle: number;
}

// Ray vs AABB slab test — returns [tMin, tMax] or null if no intersection
function rayVsAABB(
  ox: number, oy: number,
  invDirX: number, invDirY: number,
  minX: number, minY: number,
  maxX: number, maxY: number
): [number, number] | null {
  const tx1 = (minX - ox) * invDirX;
  const tx2 = (maxX - ox) * invDirX;
  const ty1 = (minY - oy) * invDirY;
  const ty2 = (maxY - oy) * invDirY;

  const tMinX = Math.min(tx1, tx2);
  const tMaxX = Math.max(tx1, tx2);
  const tMinY = Math.min(ty1, ty2);
  const tMaxY = Math.max(ty1, ty2);

  const tMin = Math.max(tMinX, tMinY);
  const tMax = Math.min(tMaxX, tMaxY);

  if (tMax < 0 || tMin > tMax) return null;
  return [tMin, tMax];
}

// Closest point on AABB to a given point
function closestPointOnAABB(
  px: number, py: number,
  minX: number, minY: number,
  maxX: number, maxY: number
): [number, number] {
  const cx = Math.max(minX, Math.min(px, maxX));
  const cy = Math.max(minY, Math.min(py, maxY));
  return [cx, cy];
}

// Compute surface normal for a hit point on an AABB
function aabbSurfaceNormal(
  hitX: number, hitY: number,
  minX: number, minY: number,
  maxX: number, maxY: number
): [number, number] {
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const hw = (maxX - minX) * 0.5;
  const hh = (maxY - minY) * 0.5;

  const dx = hitX - cx;
  const dy = hitY - cy;

  // Overlap in each axis
  const ox = hw - Math.abs(dx);
  const oy = hh - Math.abs(dy);

  if (ox < oy) {
    return [dx > 0 ? 1 : -1, 0];
  } else {
    return [0, dy > 0 ? 1 : -1];
  }
}

export class SceneQuery {
  private bvh: BVHTree;
  private bodies: Map<number, BodyData>;

  constructor(
    bvh: BVHTree,
    bodies: Map<number, BodyData>
  ) {
    this.bvh = bvh;
    this.bodies = bodies;
  }

  raycast(
    originX: number,
    originY: number,
    dirX: number,
    dirY: number,
    maxDist: number
  ): RayHit | null {
    // Normalize direction
    const len = Math.sqrt(dirX * dirX + dirY * dirY);
    if (len < 1e-10) return null;
    const ndx = dirX / len;
    const ndy = dirY / len;

    const invDirX = Math.abs(ndx) > 1e-10 ? 1 / ndx : Infinity;
    const invDirY = Math.abs(ndy) > 1e-10 ? 1 / ndy : Infinity;

    // Query BVH for candidate bodies along ray AABB
    const rayMinX = Math.min(originX, originX + ndx * maxDist);
    const rayMinY = Math.min(originY, originY + ndy * maxDist);
    const rayMaxX = Math.max(originX, originX + ndx * maxDist);
    const rayMaxY = Math.max(originY, originY + ndy * maxDist);

    const rayAABB: AABB = { minX: rayMinX, minY: rayMinY, maxX: rayMaxX, maxY: rayMaxY };
    const candidates = this.bvh.queryAABB(rayAABB);

    let bestT = maxDist;
    let bestHit: RayHit | null = null;

    for (const bodyId of candidates) {
      const body = this.bodies.get(bodyId);
      if (!body) continue;

      const minX = body.cx - body.hw;
      const minY = body.cy - body.hh;
      const maxX = body.cx + body.hw;
      const maxY = body.cy + body.hh;

      const result = rayVsAABB(originX, originY, invDirX, invDirY, minX, minY, maxX, maxY);
      if (!result) continue;

      const [tMin, tMax] = result;
      const t = tMin >= 0 ? tMin : tMax;
      if (t < 0 || t > bestT) continue;

      bestT = t;
      const hitX = originX + ndx * t;
      const hitY = originY + ndy * t;
      const [normalX, normalY] = aabbSurfaceNormal(hitX, hitY, minX, minY, maxX, maxY);

      bestHit = { bodyId, t, hitX, hitY, normalX, normalY };
    }

    return bestHit;
  }

  overlapAABB(aabb: AABB): OverlapResult {
    const bodyIds = this.bvh.queryAABB(aabb);
    return { bodyIds };
  }

  overlapCircle(cx: number, cy: number, radius: number): OverlapResult {
    // Broad phase: AABB enclosing the circle
    const queryAABB: AABB = {
      minX: cx - radius,
      minY: cy - radius,
      maxX: cx + radius,
      maxY: cy + radius,
    };

    const candidates = this.bvh.queryAABB(queryAABB);
    const r2 = radius * radius;
    const result: number[] = [];

    for (const bodyId of candidates) {
      const body = this.bodies.get(bodyId);
      if (!body) continue;

      const minX = body.cx - body.hw;
      const minY = body.cy - body.hh;
      const maxX = body.cx + body.hw;
      const maxY = body.cy + body.hh;

      // Closest point on AABB to circle center
      const [cpx, cpy] = closestPointOnAABB(cx, cy, minX, minY, maxX, maxY);
      const dx = cpx - cx;
      const dy = cpy - cy;
      if (dx * dx + dy * dy <= r2) {
        result.push(bodyId);
      }
    }

    return { bodyIds: result };
  }

  sweepAABB(
    aabb: AABB,
    dirX: number,
    dirY: number,
    maxDist: number
  ): RayHit | null {
    const len = Math.sqrt(dirX * dirX + dirY * dirY);
    if (len < 1e-10) return null;
    const ndx = dirX / len;
    const ndy = dirY / len;

    const hw = (aabb.maxX - aabb.minX) * 0.5;
    const hh = (aabb.maxY - aabb.minY) * 0.5;
    const ocx = (aabb.minX + aabb.maxX) * 0.5;
    const ocy = (aabb.minY + aabb.maxY) * 0.5;

    const invDirX = Math.abs(ndx) > 1e-10 ? 1 / ndx : Infinity;
    const invDirY = Math.abs(ndy) > 1e-10 ? 1 / ndy : Infinity;

    // Swept volume AABB for broad phase
    const endCx = ocx + ndx * maxDist;
    const endCy = ocy + ndy * maxDist;
    const sweepAABBQuery: AABB = {
      minX: Math.min(aabb.minX, endCx - hw),
      minY: Math.min(aabb.minY, endCy - hh),
      maxX: Math.max(aabb.maxX, endCx + hw),
      maxY: Math.max(aabb.maxY, endCy + hh),
    };

    const candidates = this.bvh.queryAABB(sweepAABBQuery);

    let bestT = maxDist;
    let bestHit: RayHit | null = null;

    for (const bodyId of candidates) {
      const body = this.bodies.get(bodyId);
      if (!body) continue;

      // Minkowski sum: expand body AABB by swept shape half-extents
      const expandedMinX = body.cx - body.hw - hw;
      const expandedMinY = body.cy - body.hh - hh;
      const expandedMaxX = body.cx + body.hw + hw;
      const expandedMaxY = body.cy + body.hh + hh;

      const result = rayVsAABB(ocx, ocy, invDirX, invDirY, expandedMinX, expandedMinY, expandedMaxX, expandedMaxY);
      if (!result) continue;

      const [tMin, tMax] = result;
      const t = tMin >= 0 ? tMin : tMax;
      if (t < 0 || t > bestT) continue;

      bestT = t;
      const hitX = ocx + ndx * t;
      const hitY = ocy + ndy * t;
      const [normalX, normalY] = aabbSurfaceNormal(hitX, hitY, expandedMinX, expandedMinY, expandedMaxX, expandedMaxY);

      bestHit = { bodyId, t, hitX, hitY, normalX, normalY };
    }

    return bestHit;
  }

  closestPoint(px: number, py: number, maxDist: number): ClosestPointResult | null {
    const queryAABB: AABB = {
      minX: px - maxDist,
      minY: py - maxDist,
      maxX: px + maxDist,
      maxY: py + maxDist,
    };

    const candidates = this.bvh.queryAABB(queryAABB);

    let bestDist = maxDist;
    let bestResult: ClosestPointResult | null = null;

    for (const bodyId of candidates) {
      const body = this.bodies.get(bodyId);
      if (!body) continue;

      const minX = body.cx - body.hw;
      const minY = body.cy - body.hh;
      const maxX = body.cx + body.hw;
      const maxY = body.cy + body.hh;

      const [cpx, cpy] = closestPointOnAABB(px, py, minX, minY, maxX, maxY);
      const dx = cpx - px;
      const dy = cpy - py;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < bestDist) {
        bestDist = dist;
        bestResult = {
          bodyId,
          pointX: cpx,
          pointY: cpy,
          distance: dist,
        };
      }
    }

    return bestResult;
  }
}
export function createSceneQuery(...a: any[]): any { return {} as any; }
