/**
 * src/lib/sph/collision/aabb-manager.ts
 *
 * AABB lifecycle utilities for 2D rigid bodies (oriented bounding boxes).
 * No external dependencies.
 */





export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * Compute the tightest axis-aligned bounding box for an oriented bounding box (OBB).
 *
 * The OBB is defined by:
 *   - (cx, cy)  — center position
 *   - hw        — half-width  (local X extent)
 *   - hh        — half-height (local Y extent)
 *   - angle     — rotation in radians (counter-clockwise from +X axis)
 *
 * For a rotated rectangle the four corners are:
 *   ±hw * cos(angle) ∓ hh * sin(angle)   (x projections)
 *   ±hw * sin(angle) ± hh * cos(angle)   (y projections)
 *
 * The tight AABB half-extents are:
 *   extX = |hw * cos(angle)| + |hh * sin(angle)|
 *   extY = |hw * sin(angle)| + |hh * cos(angle)|
 */
export function computeAABB(
  cx: number,
  cy: number,
  hw: number,
  hh: number,
  angle: number
): AABB {
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  const extX = Math.abs(hw * cosA) + Math.abs(hh * sinA);
  const extY = Math.abs(hw * sinA) + Math.abs(hh * cosA);

  return {
    minX: cx - extX,
    minY: cy - extY,
    maxX: cx + extX,
    maxY: cy + extY,
  };
}

/**
 * Expand an AABB uniformly by `margin` on all four sides.
 *
 * Used for temporal coherence: a slightly enlarged AABB stays valid for several
 * frames without needing a full refit every tick, reducing broad-phase overhead.
 *
 * @param aabb   - source AABB (not mutated)
 * @param margin - non-negative expansion distance in world units
 * @returns      new enlarged AABB
 */
export function expandAABB(aabb: AABB, margin: number): AABB {
  return {
    minX: aabb.minX - margin,
    minY: aabb.minY - margin,
    maxX: aabb.maxX + margin,
    maxY: aabb.maxY + margin,
  };
}

/**
 * Compute the union (smallest enclosing AABB) of two AABBs.
 *
 * Commonly used when building / refitting BVH nodes.
 *
 * @param a - first AABB (not mutated)
 * @param b - second AABB (not mutated)
 * @returns new AABB that contains both inputs
 */
export function mergeAABB(a: AABB, b: AABB): AABB {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

/**
 * Surface area of a 2-D AABB — i.e. the perimeter of the rectangle.
 *
 * In 2-D the Surface Area Heuristic (SAH) uses perimeter as the cost proxy:
 *   SA = 2 * (width + height)
 *
 * @param a - the AABB to measure
 * @returns perimeter (>= 0)
 */
export function aabbArea(a: AABB): number {
  const width  = a.maxX - a.minX;
  const height = a.maxY - a.minY;
  return 2.0 * (width + height);
}

/**
 * Broad-phase overlap test between two AABBs.
 *
 * Two AABBs are *separated* if one is entirely to the left, right, above, or
 * below the other — any other configuration implies overlap.
 *
 * Touching edges (e.g. a.maxX === b.minX) are considered overlapping so that
 * adjacent static bodies are not missed by the narrow phase.
 *
 * @param a - first AABB
 * @param b - second AABB
 * @returns true when the AABBs intersect (or touch)
 */
export function testAABB(a: AABB, b: AABB): boolean {
  if (a.maxX < b.minX || b.maxX < a.minX) return false;
  if (a.maxY < b.minY || b.maxY < a.minY) return false;
  return true;
}

/**
 * Point-in-AABB containment test.
 *
 * @param aabb - the bounding box
 * @param x    - world-space X coordinate of the point
 * @param y    - world-space Y coordinate of the point
 * @returns true when (x, y) lies inside or on the boundary of `aabb`
 */
export function containsPoint(aabb: AABB, x: number, y: number): boolean {
  return (
    x >= aabb.minX &&
    x <= aabb.maxX &&
    y >= aabb.minY &&
    y <= aabb.maxY
  );
}

/**
 * Slab-method ray vs AABB intersection test.
 *
 * The ray is parameterised as  P(t) = O + t * D  for t ∈ [0, maxT].
 *
 * Algorithm:
 *   For each axis compute the entry/exit t values of the two parallel slabs.
 *   The ray hits the AABB when the overall entry t (tMin) ≤ overall exit t (tMax)
 *   and the interval overlaps [0, maxT].
 *
 * Handles zero-direction components safely by treating them as infinite slabs
 * (the ray either misses the slab entirely or is always inside it).
 *
 * @param aabb  - the bounding box to test against
 * @param ox    - ray origin X
 * @param oy    - ray origin Y
 * @param dx    - ray direction X (need not be normalised)
 * @param dy    - ray direction Y (need not be normalised)
 * @param maxT  - maximum ray parameter (length of the ray segment)
 * @returns     the entry t value in [0, maxT] on a hit, or -1 on a miss
 */
export function raycastAABB(
  aabb: AABB,
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  maxT: number
): number {
  let tMin = 0.0;
  let tMax = maxT;

  // --- X slab ---
  if (Math.abs(dx) < 1e-12) {
    // Ray is parallel to the X slab; miss if origin is outside.
    if (ox < aabb.minX || ox > aabb.maxX) return -1;
  } else {
    const invDx = 1.0 / dx;
    let t1 = (aabb.minX - ox) * invDx;
    let t2 = (aabb.maxX - ox) * invDx;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return -1;
  }

  // --- Y slab ---
  if (Math.abs(dy) < 1e-12) {
    // Ray is parallel to the Y slab; miss if origin is outside.
    if (oy < aabb.minY || oy > aabb.maxY) return -1;
  } else {
    const invDy = 1.0 / dy;
    let t1 = (aabb.minY - oy) * invDy;
    let t2 = (aabb.maxY - oy) * invDy;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return -1;
  }

  // tMin is the entry point; if it is behind the origin tMax is the hit.
  const t = tMin >= 0.0 ? tMin : tMax;
  return t >= 0.0 && t <= maxT ? t : -1;
}
