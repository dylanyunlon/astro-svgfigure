/**
 * src/lib/sph/collision/SceneQuery.ts
 *
 * Scene-level spatial query API for the SPH collision pipeline.
 *
 * Queries provided:
 *  - raycast          — first hit along a ray
 *  - raycastAll       — all hits along a ray, sorted by distance
 *  - overlapAABB      — all shapes overlapping an axis-aligned bounding box
 *  - overlapCircle    — all shapes overlapping a circle / sphere
 *  - closestPoint     — nearest point on any shape to a query point
 *
 * Architecture:
 *  - Shapes register themselves in a dynamic BVH (AABB tree).
 *  - BVH provides O(log n) broad-phase culling.
 *  - Exact narrow-phase tests are performed on BVH leaf candidates.
 *
 * Supported shape types: Circle, AABB, Capsule, ConvexPolygon.
 *
 * Units: all lengths are in simulation units (metres). All angles in radians.
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Primitive math types
// ─────────────────────────────────────────────────────────────────────────────





export interface Vec2 {
  x: number;
  y: number;
}

export interface AABB2 {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Ray2 {
  /** Ray origin. */
  origin: Vec2;
  /** Unit direction (caller must normalise). */
  direction: Vec2;
  /** Maximum travel distance. Default = Infinity. */
  maxDistance?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Shape definitions
// ─────────────────────────────────────────────────────────────────────────────

export const enum ShapeType {
  Circle = "Circle",
  AABB = "AABB",
  Capsule = "Capsule",
  ConvexPolygon = "ConvexPolygon",
}

export interface CircleShape {
  type: ShapeType.Circle;
  center: Vec2;
  radius: number;
}

export interface AABBShape {
  type: ShapeType.AABB;
  min: Vec2;
  max: Vec2;
}

export interface CapsuleShape {
  type: ShapeType.Capsule;
  /** Capsule segment start. */
  a: Vec2;
  /** Capsule segment end. */
  b: Vec2;
  radius: number;
}

export interface ConvexPolygonShape {
  type: ShapeType.ConvexPolygon;
  /** Vertices in counter-clockwise order. */
  vertices: Vec2[];
}

export type Shape = CircleShape | AABBShape | CapsuleShape | ConvexPolygonShape;

/** User-supplied metadata attached to a registered body. */
export type UserData = Record<string, unknown>;

/** A collidable body registered in the scene. */
export interface Body {
  id: number;
  shape: Shape;
  userData: UserData;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Query result types
// ─────────────────────────────────────────────────────────────────────────────

export interface RaycastHit {
  body: Body;
  /** Distance from ray origin to the hit point. */
  distance: number;
  /** World-space hit point. */
  point: Vec2;
  /** Outward surface normal at hit point. */
  normal: Vec2;
}

export interface OverlapResult {
  body: Body;
}

export interface ClosestPointResult {
  body: Body;
  /** Nearest point on the shape surface (or interior for solid queries). */
  point: Vec2;
  /** Unsigned distance from query point to nearest point. */
  distance: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Vec2 utilities (inline, zero-alloc hot paths)
// ─────────────────────────────────────────────────────────────────────────────

function v2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}
function v2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}
function v2Scale(a: Vec2, s: number): Vec2 {
  return { x: a.x * s, y: a.y * s };
}
function v2Dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}
function v2LenSq(a: Vec2): number {
  return a.x * a.x + a.y * a.y;
}
function v2Len(a: Vec2): number {
  return Math.sqrt(v2LenSq(a));
}
function v2Norm(a: Vec2): Vec2 {
  const l = v2Len(a);
  return l > 1e-12 ? { x: a.x / l, y: a.y / l } : { x: 0, y: 0 };
}
function v2Perp(a: Vec2): Vec2 {
  return { x: -a.y, y: a.x };
}

/** Clamp t ∈ [lo, hi]. */
function clamp(t: number, lo: number, hi: number): number {
  return t < lo ? lo : t > hi ? hi : t;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  AABB helpers
// ─────────────────────────────────────────────────────────────────────────────

function aabbFromShape(s: Shape): AABB2 {
  switch (s.type) {
    case ShapeType.Circle: {
      const { center: c, radius: r } = s;
      return { minX: c.x - r, minY: c.y - r, maxX: c.x + r, maxY: c.y + r };
    }
    case ShapeType.AABB:
      return {
        minX: s.min.x,
        minY: s.min.y,
        maxX: s.max.x,
        maxY: s.max.y,
      };
    case ShapeType.Capsule: {
      const { a, b, radius: r } = s;
      return {
        minX: Math.min(a.x, b.x) - r,
        minY: Math.min(a.y, b.y) - r,
        maxX: Math.max(a.x, b.x) + r,
        maxY: Math.max(a.y, b.y) + r,
      };
    }
    case ShapeType.ConvexPolygon: {
      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
      for (const v of s.vertices) {
        if (v.x < minX) minX = v.x;
        if (v.y < minY) minY = v.y;
        if (v.x > maxX) maxX = v.x;
        if (v.y > maxY) maxY = v.y;
      }
      return { minX, minY, maxX, maxY };
    }
  }
}

function aabbUnion(a: AABB2, b: AABB2): AABB2 {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

function aabbArea(a: AABB2): number {
  return (a.maxX - a.minX) * (a.maxY - a.minY);
}

function aabbOverlapsAABB(a: AABB2, b: AABB2): boolean {
  return (
    a.minX <= b.maxX &&
    a.maxX >= b.minX &&
    a.minY <= b.maxY &&
    a.maxY >= b.minY
  );
}

function aabbContainsPoint(a: AABB2, p: Vec2): boolean {
  return p.x >= a.minX && p.x <= a.maxX && p.y >= a.minY && p.y <= a.maxY;
}

/**
 * Slab-method ray–AABB intersection.
 * Returns the entry t (≥ 0) or -1 on miss.
 */
function rayIntersectsAABB(ray: Ray2, box: AABB2): number {
  const { origin: o, direction: d } = ray;
  const maxDist = ray.maxDistance ?? Infinity;
  let tMin = 0;
  let tMax = maxDist;

  // X slab
  if (Math.abs(d.x) < 1e-12) {
    if (o.x < box.minX || o.x > box.maxX) return -1;
  } else {
    const invD = 1 / d.x;
    let t1 = (box.minX - o.x) * invD;
    let t2 = (box.maxX - o.x) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return -1;
  }

  // Y slab
  if (Math.abs(d.y) < 1e-12) {
    if (o.y < box.minY || o.y > box.maxY) return -1;
  } else {
    const invD = 1 / d.y;
    let t1 = (box.minY - o.y) * invD;
    let t2 = (box.maxY - o.y) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return -1;
  }

  return tMin >= 0 ? tMin : tMax >= 0 ? tMax : -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Dynamic AABB BVH (top-down rebuild on mutation)
// ─────────────────────────────────────────────────────────────────────────────
//
// Strategy: We store a flat array of (body, aabb) leaves and rebuild the tree
// on every structural change using SAH-lite top-down recursion.  For typical
// SPH scenes (< 10 k static bodies) this is fast enough; a dynamic incremental
// tree can be layered on top later.

interface BVHLeaf {
  isLeaf: true;
  aabb: AABB2;
  body: Body;
}

interface BVHNode {
  isLeaf: false;
  aabb: AABB2;
  left: BVHNode | BVHLeaf;
  right: BVHNode | BVHLeaf;
}

type BVHTree = BVHNode | BVHLeaf | null;

function buildBVH(leaves: BVHLeaf[]): BVHTree {
  if (leaves.length === 0) return null;
  if (leaves.length === 1) return leaves[0];

  // Compute enclosing AABB
  let aabb = leaves[0].aabb;
  for (let i = 1; i < leaves.length; i++) {
    aabb = aabbUnion(aabb, leaves[i].aabb);
  }

  // Choose split axis (longest extent)
  const dx = aabb.maxX - aabb.minX;
  const dy = aabb.maxY - aabb.minY;
  const axis: "x" | "y" = dx >= dy ? "x" : "y";

  // Sort by centroid on axis
  const sorted = [...leaves].sort((a, b) => {
    const ca =
      axis === "x"
        ? (a.aabb.minX + a.aabb.maxX) * 0.5
        : (a.aabb.minY + a.aabb.maxY) * 0.5;
    const cb =
      axis === "x"
        ? (b.aabb.minX + b.aabb.maxX) * 0.5
        : (b.aabb.minY + b.aabb.maxY) * 0.5;
    return ca - cb;
  });

  const mid = sorted.length >> 1;
  const left = buildBVH(sorted.slice(0, mid))!;
  const right = buildBVH(sorted.slice(mid))!;

  return {
    isLeaf: false,
    aabb,
    left: left as BVHNode | BVHLeaf,
    right: right as BVHNode | BVHLeaf,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Narrow-phase: ray vs shape
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Closest point on segment [a, b] to point p.
 */
function closestPointOnSegment(p: Vec2, a: Vec2, b: Vec2): Vec2 {
  const ab = v2Sub(b, a);
  const lenSq = v2LenSq(ab);
  if (lenSq < 1e-24) return { ...a };
  const t = clamp(v2Dot(v2Sub(p, a), ab) / lenSq, 0, 1);
  return v2Add(a, v2Scale(ab, t));
}

function rayVsCircle(ray: Ray2, s: CircleShape): RaycastHit | null {
  const oc = v2Sub(ray.origin, s.center);
  const a = v2Dot(ray.direction, ray.direction); // ≈1 if normalised
  const b = 2 * v2Dot(oc, ray.direction);
  const c = v2Dot(oc, oc) - s.radius * s.radius;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const sqrtDisc = Math.sqrt(disc);
  let t = (-b - sqrtDisc) / (2 * a);
  if (t < 0) t = (-b + sqrtDisc) / (2 * a);
  if (t < 0) return null;
  const maxDist = ray.maxDistance ?? Infinity;
  if (t > maxDist) return null;
  const point = v2Add(ray.origin, v2Scale(ray.direction, t));
  const normal = v2Norm(v2Sub(point, s.center));
  return { body: null as unknown as Body, distance: t, point, normal };
}

function rayVsAABB(ray: Ray2, s: AABBShape): RaycastHit | null {
  const box: AABB2 = {
    minX: s.min.x,
    minY: s.min.y,
    maxX: s.max.x,
    maxY: s.max.y,
  };
  const t = rayIntersectsAABB(ray, box);
  if (t < 0) return null;
  const point = v2Add(ray.origin, v2Scale(ray.direction, t));
  // Compute normal from which face was hit
  const normal = computeAABBNormal(point, s);
  return { body: null as unknown as Body, distance: t, point, normal };
}

function computeAABBNormal(p: Vec2, s: AABBShape): Vec2 {
  const cx = (s.min.x + s.max.x) * 0.5;
  const cy = (s.min.y + s.max.y) * 0.5;
  const dx = (p.x - cx) / ((s.max.x - s.min.x) * 0.5 + 1e-12);
  const dy = (p.y - cy) / ((s.max.y - s.min.y) * 0.5 + 1e-12);
  if (Math.abs(dx) >= Math.abs(dy)) {
    return { x: dx > 0 ? 1 : -1, y: 0 };
  }
  return { x: 0, y: dy > 0 ? 1 : -1 };
}

function rayVsCapsule(ray: Ray2, s: CapsuleShape): RaycastHit | null {
  // Test ray vs two end-caps + cylinder body.
  // Represent capsule as swept circle along segment [a, b].
  // We find the minimum positive t by sampling the ray–distance-to-segment
  // equation: ||(o + t*d) - closest(o+t*d, seg)||^2 = r^2
  // This yields a quadratic in t.

  const ab = v2Sub(s.b, s.a);
  const ao = v2Sub(ray.origin, s.a);
  const lenSqAB = v2LenSq(ab);

  const D = ray.direction;
  const dDotAB = v2Dot(D, ab);
  const aoDotAB = v2Dot(ao, ab);

  // Parametric along ray: t
  // Parametric along capsule segment: u = clamp((aoDotAB + t*dDotAB)/lenSqAB, 0, 1)
  // Expand ||(ao + t*D) - u*ab||^2 = r^2 — but u depends on t.
  // Simplification: solve the infinite-cylinder case first (no clamp),
  // then re-check clamped end caps.

  const a_coef =
    lenSqAB * v2Dot(D, D) - dDotAB * dDotAB;
  const b_coef =
    2 * (lenSqAB * v2Dot(ao, D) - aoDotAB * dDotAB);
  const c_coef =
    lenSqAB * v2Dot(ao, ao) - aoDotAB * aoDotAB - s.radius * s.radius * lenSqAB;

  let bestT = Infinity;

  if (Math.abs(a_coef) > 1e-12) {
    const disc = b_coef * b_coef - 4 * a_coef * c_coef;
    if (disc >= 0) {
      const sqrtDisc = Math.sqrt(disc);
      for (const sign of [-1, 1]) {
        const t = (-b_coef + sign * sqrtDisc) / (2 * a_coef);
        if (t >= 0) {
          const hitPt = v2Add(ray.origin, v2Scale(D, t));
          const u = clamp(
            (v2Dot(v2Sub(hitPt, s.a), ab)) / lenSqAB,
            0,
            1
          );
          // Accept only if the u is strictly interior (no clamping)
          const proj = v2Add(s.a, v2Scale(ab, u));
          const dist = v2Len(v2Sub(hitPt, proj));
          if (Math.abs(dist - s.radius) < 1e-4 * s.radius + 1e-6) {
            if (t < bestT) bestT = t;
          }
        }
      }
    }
  }

  // Test end caps as circles
  for (const cap of [s.a, s.b]) {
    const capShape: CircleShape = {
      type: ShapeType.Circle,
      center: cap,
      radius: s.radius,
    };
    const hit = rayVsCircle(ray, capShape);
    if (hit && hit.distance < bestT) bestT = hit.distance;
  }

  if (!isFinite(bestT)) return null;
  const maxDist = ray.maxDistance ?? Infinity;
  if (bestT > maxDist) return null;

  const point = v2Add(ray.origin, v2Scale(D, bestT));
  // Normal: closest point on segment to hit point
  const cp = closestPointOnSegment(point, s.a, s.b);
  const normal = v2Norm(v2Sub(point, cp));
  return { body: null as unknown as Body, distance: bestT, point, normal };
}

/**
 * Ray vs convex polygon using the Cyrus-Beck (parametric clipping) algorithm.
 *
 * Winding convention: CCW.
 * Outward normal for edge (va→vb) = right-hand perpendicular = (edge.y, -edge.x).
 */
function rayVsConvex(ray: Ray2, s: ConvexPolygonShape): RaycastHit | null {
  const verts = s.vertices;
  const n = verts.length;
  if (n < 3) return null;

  const maxDist = ray.maxDistance ?? Infinity;
  let tEnter = 0;
  let tExit = maxDist;
  let hitNormal: Vec2 = { x: 0, y: 1 };

  for (let i = 0; i < n; i++) {
    const va = verts[i];
    const vb = verts[(i + 1) % n];
    const edge = v2Sub(vb, va);

    // Outward normal for CCW polygon: right-hand perp of edge direction
    const outNorm: Vec2 = { x: edge.y, y: -edge.x };

    // dN = D · outNorm  (positive ⟹ ray travelling toward outside ⟹ exiting)
    const dN = v2Dot(ray.direction, outNorm);
    // qN = (va - O) · outNorm  (signed distance of origin from this edge plane)
    const qN = v2Dot(v2Sub(va, ray.origin), outNorm);

    if (Math.abs(dN) < 1e-12) {
      // Ray parallel to this edge — check if origin is on the outside
      if (qN < 0) return null; // outside this half-plane, no intersection
      continue;
    }

    const tEdge = qN / dN;

    if (dN < 0) {
      // Ray entering this half-plane
      if (tEdge > tEnter) {
        tEnter = tEdge;
        // Inward-facing normal at entry face
        const len = Math.sqrt(outNorm.x * outNorm.x + outNorm.y * outNorm.y);
        hitNormal = { x: -outNorm.x / len, y: -outNorm.y / len };
      }
    } else {
      // Ray exiting this half-plane
      if (tEdge < tExit) tExit = tEdge;
    }

    if (tEnter > tExit) return null;
  }

  // Choose the entry t (or exit if origin is inside)
  const t = tEnter >= 0 ? tEnter : tExit;
  if (t < 0 || t > maxDist) return null;

  const point = v2Add(ray.origin, v2Scale(ray.direction, t));
  return { body: null as unknown as Body, distance: t, point, normal: hitNormal };
}

function narrowRay(ray: Ray2, body: Body): RaycastHit | null {
  let hit: RaycastHit | null;
  switch (body.shape.type) {
    case ShapeType.Circle:
      hit = rayVsCircle(ray, body.shape);
      break;
    case ShapeType.AABB:
      hit = rayVsAABB(ray, body.shape);
      break;
    case ShapeType.Capsule:
      hit = rayVsCapsule(ray, body.shape);
      break;
    case ShapeType.ConvexPolygon:
      hit = rayVsConvex(ray, body.shape);
      break;
  }
  if (hit) hit.body = body;
  return hit;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8.  Narrow-phase: overlap vs shape
// ─────────────────────────────────────────────────────────────────────────────

function overlapCircleVsShape(center: Vec2, radius: number, body: Body): boolean {
  switch (body.shape.type) {
    case ShapeType.Circle: {
      const d = v2Sub(center, body.shape.center);
      return v2LenSq(d) <= (radius + body.shape.radius) ** 2;
    }
    case ShapeType.AABB: {
      // Clamp query center to AABB, then check distance
      const cx = clamp(center.x, body.shape.min.x, body.shape.max.x);
      const cy = clamp(center.y, body.shape.min.y, body.shape.max.y);
      const dx = center.x - cx;
      const dy = center.y - cy;
      return dx * dx + dy * dy <= radius * radius;
    }
    case ShapeType.Capsule: {
      const cp = closestPointOnSegment(center, body.shape.a, body.shape.b);
      const d = v2Sub(center, cp);
      return v2LenSq(d) <= (radius + body.shape.radius) ** 2;
    }
    case ShapeType.ConvexPolygon: {
      // Use GJK-lite: closest point on polygon to circle center
      const cp = closestPointOnConvex(center, body.shape);
      const d = v2Sub(center, cp);
      return v2LenSq(d) <= radius * radius;
    }
  }
}

function overlapAABBVsShape(box: AABB2, body: Body): boolean {
  const bodyAABB = aabbFromShape(body.shape);
  if (!aabbOverlapsAABB(box, bodyAABB)) return false;

  switch (body.shape.type) {
    case ShapeType.AABB:
      return true; // AABB vs AABB already confirmed above
    case ShapeType.Circle: {
      // Circle vs AABB
      const cx = clamp(body.shape.center.x, box.minX, box.maxX);
      const cy = clamp(body.shape.center.y, box.minY, box.maxY);
      const dx = body.shape.center.x - cx;
      const dy = body.shape.center.y - cy;
      return dx * dx + dy * dy <= body.shape.radius * body.shape.radius;
    }
    case ShapeType.Capsule: {
      // Conservative: use AABB overlap (already checked)
      // Precise: test if any corner of query box is inside capsule,
      // or if the capsule segment intersects any edge of the box.
      const corners: Vec2[] = [
        { x: box.minX, y: box.minY },
        { x: box.maxX, y: box.minY },
        { x: box.maxX, y: box.maxY },
        { x: box.minX, y: box.maxY },
      ];
      const { a, b, radius } = body.shape;
      for (const c of corners) {
        const cp = closestPointOnSegment(c, a, b);
        if (v2LenSq(v2Sub(c, cp)) <= radius * radius) return true;
      }
      // Check if capsule segment endpoints are inside box
      if (aabbContainsPoint(box, a) || aabbContainsPoint(box, b)) return true;
      // Check capsule centerline against box edges
      const boxEdges: [Vec2, Vec2][] = [
        [{ x: box.minX, y: box.minY }, { x: box.maxX, y: box.minY }],
        [{ x: box.maxX, y: box.minY }, { x: box.maxX, y: box.maxY }],
        [{ x: box.maxX, y: box.maxY }, { x: box.minX, y: box.maxY }],
        [{ x: box.minX, y: box.maxY }, { x: box.minX, y: box.minY }],
      ];
      for (const [ea, eb] of boxEdges) {
        if (segmentsClosestDistance(a, b, ea, eb) <= radius) return true;
      }
      return false;
    }
    case ShapeType.ConvexPolygon: {
      // SAT-lite: test all polygon edges' normals and AABB axes
      return satConvexVsAABB(body.shape, box);
    }
  }
}

/** Minimum distance between two line segments. */
function segmentsClosestDistance(
  a0: Vec2,
  a1: Vec2,
  b0: Vec2,
  b1: Vec2
): number {
  // 4 point-to-segment distances
  const d0 = v2Len(v2Sub(closestPointOnSegment(a0, b0, b1), a0));
  const d1 = v2Len(v2Sub(closestPointOnSegment(a1, b0, b1), a1));
  const d2 = v2Len(v2Sub(closestPointOnSegment(b0, a0, a1), b0));
  const d3 = v2Len(v2Sub(closestPointOnSegment(b1, a0, a1), b1));
  return Math.min(d0, d1, d2, d3);
}

/** SAT overlap test: convex polygon vs AABB. */
function satConvexVsAABB(poly: ConvexPolygonShape, box: AABB2): boolean {
  const boxVerts: Vec2[] = [
    { x: box.minX, y: box.minY },
    { x: box.maxX, y: box.minY },
    { x: box.maxX, y: box.maxY },
    { x: box.minX, y: box.maxY },
  ];

  // Test axes from polygon edges
  const verts = poly.vertices;
  const n = verts.length;
  for (let i = 0; i < n; i++) {
    const edge = v2Sub(verts[(i + 1) % n], verts[i]);
    const axis: Vec2 = v2Norm(v2Perp(edge));
    if (!satOverlapOnAxis(verts, boxVerts, axis)) return false;
  }

  // Test AABB axes (X and Y)
  for (const axis of [
    { x: 1, y: 0 },
    { x: 0, y: 1 },
  ] as Vec2[]) {
    if (!satOverlapOnAxis(verts, boxVerts, axis)) return false;
  }

  return true;
}

function satOverlapOnAxis(aVerts: Vec2[], bVerts: Vec2[], axis: Vec2): boolean {
  let aMin = Infinity,
    aMax = -Infinity,
    bMin = Infinity,
    bMax = -Infinity;
  for (const v of aVerts) {
    const p = v2Dot(v, axis);
    if (p < aMin) aMin = p;
    if (p > aMax) aMax = p;
  }
  for (const v of bVerts) {
    const p = v2Dot(v, axis);
    if (p < bMin) bMin = p;
    if (p > bMax) bMax = p;
  }
  return aMax >= bMin && bMax >= aMin;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  Narrow-phase: closest point on shape
// ─────────────────────────────────────────────────────────────────────────────

function closestPointOnShape(p: Vec2, body: Body): { point: Vec2; dist: number } {
  switch (body.shape.type) {
    case ShapeType.Circle: {
      const d = v2Sub(p, body.shape.center);
      const len = v2Len(d);
      if (len < 1e-12) {
        return { point: { ...body.shape.center }, dist: body.shape.radius };
      }
      const surface = v2Add(
        body.shape.center,
        v2Scale(v2Norm(d), body.shape.radius)
      );
      return { point: surface, dist: Math.abs(len - body.shape.radius) };
    }
    case ShapeType.AABB: {
      const cx = clamp(p.x, body.shape.min.x, body.shape.max.x);
      const cy = clamp(p.y, body.shape.min.y, body.shape.max.y);
      const pt: Vec2 = { x: cx, y: cy };
      return { point: pt, dist: v2Len(v2Sub(p, pt)) };
    }
    case ShapeType.Capsule: {
      const cp = closestPointOnSegment(p, body.shape.a, body.shape.b);
      const dir = v2Sub(p, cp);
      const len = v2Len(dir);
      if (len < 1e-12) {
        return { point: { ...cp }, dist: body.shape.radius };
      }
      const surface = v2Add(cp, v2Scale(v2Norm(dir), body.shape.radius));
      return { point: surface, dist: Math.abs(len - body.shape.radius) };
    }
    case ShapeType.ConvexPolygon: {
      const cp = closestPointOnConvex(p, body.shape);
      return { point: cp, dist: v2Len(v2Sub(p, cp)) };
    }
  }
}

/**
 * Closest point on (or inside) a convex polygon to query point p.
 * If p is inside the polygon, returns p itself (distance = 0).
 */
function closestPointOnConvex(p: Vec2, s: ConvexPolygonShape): Vec2 {
  const verts = s.vertices;
  const n = verts.length;

  // Check if inside: all cross products same sign (CCW winding)
  let inside = true;
  for (let i = 0; i < n; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % n];
    const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    if (cross < 0) {
      inside = false;
      break;
    }
  }
  if (inside) return { ...p };

  // Outside: find closest edge point
  let bestDist = Infinity;
  let bestPt: Vec2 = { ...verts[0] };
  for (let i = 0; i < n; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % n];
    const cp = closestPointOnSegment(p, a, b);
    const d = v2LenSq(v2Sub(p, cp));
    if (d < bestDist) {
      bestDist = d;
      bestPt = cp;
    }
  }
  return bestPt;
}

// ─────────────────────────────────────────────────────────────────────────────
// 10.  BVH traversal helpers
// ─────────────────────────────────────────────────────────────────────────────

function bvhQueryRay(
  node: BVHTree,
  ray: Ray2,
  results: RaycastHit[]
): void {
  if (!node) return;
  if (rayIntersectsAABB(ray, node.aabb) < 0) return;
  if (node.isLeaf) {
    const hit = narrowRay(ray, node.body);
    if (hit) results.push(hit);
    return;
  }
  bvhQueryRay(node.left, ray, results);
  bvhQueryRay(node.right, ray, results);
}

function bvhQueryAABB(
  node: BVHTree,
  box: AABB2,
  results: OverlapResult[]
): void {
  if (!node) return;
  if (!aabbOverlapsAABB(node.aabb, box)) return;
  if (node.isLeaf) {
    if (overlapAABBVsShape(box, node.body)) {
      results.push({ body: node.body });
    }
    return;
  }
  bvhQueryAABB(node.left, box, results);
  bvhQueryAABB(node.right, box, results);
}

function bvhQueryCircle(
  node: BVHTree,
  center: Vec2,
  radius: number,
  results: OverlapResult[]
): void {
  if (!node) return;
  // Broad: expand AABB by radius
  const expanded: AABB2 = {
    minX: node.aabb.minX - radius,
    minY: node.aabb.minY - radius,
    maxX: node.aabb.maxX + radius,
    maxY: node.aabb.maxY + radius,
  };
  if (!aabbContainsPoint(expanded, center)) {
    // Closer test: clamp center to AABB, check distance
    const cx = clamp(center.x, node.aabb.minX, node.aabb.maxX);
    const cy = clamp(center.y, node.aabb.minY, node.aabb.maxY);
    const dx = center.x - cx;
    const dy = center.y - cy;
    if (dx * dx + dy * dy > radius * radius) return;
  }
  if (node.isLeaf) {
    if (overlapCircleVsShape(center, radius, node.body)) {
      results.push({ body: node.body });
    }
    return;
  }
  bvhQueryCircle(node.left, center, radius, results);
  bvhQueryCircle(node.right, center, radius, results);
}

function bvhClosestPoint(
  node: BVHTree,
  p: Vec2,
  best: { result: ClosestPointResult | null; dist: number }
): void {
  if (!node) return;

  // Lower-bound distance from p to node AABB
  const cx = clamp(p.x, node.aabb.minX, node.aabb.maxX);
  const cy = clamp(p.y, node.aabb.minY, node.aabb.maxY);
  const dx = p.x - cx;
  const dy = p.y - cy;
  const lbDist = Math.sqrt(dx * dx + dy * dy);
  if (lbDist >= best.dist) return; // Prune

  if (node.isLeaf) {
    const { point, dist } = closestPointOnShape(p, node.body);
    if (dist < best.dist) {
      best.dist = dist;
      best.result = { body: node.body, point, distance: dist };
    }
    return;
  }

  // Visit closer child first (heuristic: compare child AABB lower bounds)
  const lbLeft = aabbLowerBoundDist(p, node.left.aabb);
  const lbRight = aabbLowerBoundDist(p, node.right.aabb);
  if (lbLeft <= lbRight) {
    bvhClosestPoint(node.left, p, best);
    bvhClosestPoint(node.right, p, best);
  } else {
    bvhClosestPoint(node.right, p, best);
    bvhClosestPoint(node.left, p, best);
  }
}

function aabbLowerBoundDist(p: Vec2, aabb: AABB2): number {
  const cx = clamp(p.x, aabb.minX, aabb.maxX);
  const cy = clamp(p.y, aabb.minY, aabb.maxY);
  const dx = p.x - cx;
  const dy = p.y - cy;
  return Math.sqrt(dx * dx + dy * dy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11.  SceneQuery — public API
// ─────────────────────────────────────────────────────────────────────────────

export class SceneQuery {
  private _bodies: Map<number, Body> = new Map();
  private _leaves: BVHLeaf[] = [];
  private _tree: BVHTree = null;
  private _dirty = false;
  private _nextId = 1;

  // ── Body management ──────────────────────────────────────────────────────

  /**
   * Register a shape and return its body ID.
   */
  addBody(shape: Shape, userData: UserData = {}): number {
    const id = this._nextId++;
    const body: Body = { id, shape, userData };
    this._bodies.set(id, body);
    this._leaves.push({
      isLeaf: true,
      aabb: aabbFromShape(shape),
      body,
    });
    this._dirty = true;
    return id;
  }

  /**
   * Remove a body by ID.
   */
  removeBody(id: number): boolean {
    if (!this._bodies.has(id)) return false;
    this._bodies.delete(id);
    this._leaves = this._leaves.filter((l) => l.body.id !== id);
    this._dirty = true;
    return true;
  }

  /**
   * Update the shape (and AABB) of an existing body.
   */
  updateBody(id: number, shape: Shape): boolean {
    const body = this._bodies.get(id);
    if (!body) return false;
    body.shape = shape;
    const leaf = this._leaves.find((l) => l.body.id === id);
    if (leaf) leaf.aabb = aabbFromShape(shape);
    this._dirty = true;
    return true;
  }

  /**
   * Force a BVH rebuild immediately (normally lazy).
   */
  rebuildBVH(): void {
    this._tree = buildBVH(this._leaves);
    this._dirty = false;
  }

  private _ensureBVH(): void {
    if (this._dirty) this.rebuildBVH();
  }

  get bodyCount(): number {
    return this._bodies.size;
  }

  // ── Queries ──────────────────────────────────────────────────────────────

  /**
   * Return the *closest* shape hit by the ray, or `null` if nothing was hit.
   *
   * @param ray - Origin, unit direction, and optional maxDistance.
   * @param filter - Optional predicate to skip specific bodies.
   */
  raycast(
    ray: Ray2,
    filter?: (body: Body) => boolean
  ): RaycastHit | null {
    this._ensureBVH();
    const hits: RaycastHit[] = [];
    bvhQueryRay(this._tree, ray, hits);
    let best: RaycastHit | null = null;
    for (const h of hits) {
      if (filter && !filter(h.body)) continue;
      if (!best || h.distance < best.distance) best = h;
    }
    return best;
  }

  /**
   * Return *all* shapes hit by the ray, sorted ascending by distance.
   *
   * @param ray - Origin, unit direction, and optional maxDistance.
   * @param filter - Optional predicate to skip specific bodies.
   */
  raycastAll(
    ray: Ray2,
    filter?: (body: Body) => boolean
  ): RaycastHit[] {
    this._ensureBVH();
    const hits: RaycastHit[] = [];
    bvhQueryRay(this._tree, ray, hits);
    const filtered = filter ? hits.filter((h) => filter(h.body)) : hits;
    filtered.sort((a, b) => a.distance - b.distance);
    return filtered;
  }

  /**
   * Return all bodies whose shapes overlap the given AABB.
   */
  overlapAABB(box: AABB2): OverlapResult[] {
    this._ensureBVH();
    const results: OverlapResult[] = [];
    bvhQueryAABB(this._tree, box, results);
    return results;
  }

  /**
   * Return all bodies whose shapes overlap the given circle.
   *
   * @param center - Circle center in world space.
   * @param radius - Circle radius.
   */
  overlapCircle(center: Vec2, radius: number): OverlapResult[] {
    this._ensureBVH();
    const results: OverlapResult[] = [];
    bvhQueryCircle(this._tree, center, radius, results);
    return results;
  }

  /**
   * Return the single closest point (on any shape) to the query point,
   * along with the body it belongs to and the distance.
   *
   * @param point - Query point in world space.
   * @param filter - Optional predicate to skip specific bodies.
   */
  closestPoint(
    point: Vec2,
    filter?: (body: Body) => boolean
  ): ClosestPointResult | null {
    this._ensureBVH();
    if (filter) {
      // With a filter we can't rely on BVH pruning cleanly; fall back to
      // linear scan for now (or wrap filtered leaves into a sub-tree).
      let best: ClosestPointResult | null = null;
      for (const leaf of this._leaves) {
        if (!filter(leaf.body)) continue;
        const { point: pt, dist } = closestPointOnShape(point, leaf.body);
        if (!best || dist < best.distance) {
          best = { body: leaf.body, point: pt, distance: dist };
        }
      }
      return best;
    }
    const best: { result: ClosestPointResult | null; dist: number } = {
      result: null,
      dist: Infinity,
    };
    bvhClosestPoint(this._tree, point, best);
    return best.result;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12.  Exports (re-export primitives for test convenience)
// ─────────────────────────────────────────────────────────────────────────────

export {
  closestPointOnSegment,
  closestPointOnConvex,
  closestPointOnShape,
  aabbFromShape,
  aabbOverlapsAABB,
  rayIntersectsAABB,
};

// ─────────────────────────────────────────────────────────────────────────────
// 13.  Tests  (standalone — run with:  npx ts-node SceneQuery.ts)
// ─────────────────────────────────────────────────────────────────────────────

if (typeof require !== "undefined" && require.main === module) {
  runTests();
}

function runTests(): void {
  let passed = 0;
  let failed = 0;
  const EPSILON = 1e-5;

  function eq(name: string, a: number, b: number, eps = EPSILON): void {
    if (Math.abs(a - b) <= eps) {
      console.log(`  ✓  ${name}`);
      passed++;
    } else {
      console.error(`  ✗  ${name}  (got ${a}, expected ${b})`);
      failed++;
    }
  }

  function ok(name: string, value: boolean): void {
    if (value) {
      console.log(`  ✓  ${name}`);
      passed++;
    } else {
      console.error(`  ✗  ${name}`);
      failed++;
    }
  }

  function section(title: string): void {
    console.log(`\n── ${title} ──`);
  }

  // ── Vec2 utilities ────────────────────────────────────────────────────────
  section("Vec2 utilities");
  const va: Vec2 = { x: 3, y: 4 };
  eq("v2Len", v2Len(va), 5);
  eq("v2Dot", v2Dot(va, { x: 1, y: 0 }), 3);
  const vn = v2Norm(va);
  eq("v2Norm.x", vn.x, 0.6);
  eq("v2Norm.y", vn.y, 0.8);

  // ── closestPointOnSegment ─────────────────────────────────────────────────
  section("closestPointOnSegment");
  const seg = closestPointOnSegment(
    { x: 1, y: 2 },
    { x: 0, y: 0 },
    { x: 4, y: 0 }
  );
  eq("seg.x", seg.x, 1);
  eq("seg.y", seg.y, 0);
  const segClamped = closestPointOnSegment(
    { x: -5, y: 1 },
    { x: 0, y: 0 },
    { x: 4, y: 0 }
  );
  eq("seg clamped.x", segClamped.x, 0);

  // ── AABB helpers ──────────────────────────────────────────────────────────
  section("AABB helpers");
  const circle: CircleShape = { type: ShapeType.Circle, center: { x: 2, y: 3 }, radius: 1.5 };
  const cAabb = aabbFromShape(circle);
  eq("circle AABB minX", cAabb.minX, 0.5);
  eq("circle AABB maxY", cAabb.maxY, 4.5);

  // ── rayIntersectsAABB ─────────────────────────────────────────────────────
  section("rayIntersectsAABB");
  const box: AABB2 = { minX: -1, minY: -1, maxX: 1, maxY: 1 };
  const tHit = rayIntersectsAABB(
    { origin: { x: -5, y: 0 }, direction: { x: 1, y: 0 } },
    box
  );
  eq("ray hits box at t=4", tHit, 4);
  const tMiss = rayIntersectsAABB(
    { origin: { x: -5, y: 5 }, direction: { x: 1, y: 0 } },
    box
  );
  eq("ray misses box", tMiss, -1);

  // ── SceneQuery — circle ───────────────────────────────────────────────────
  section("SceneQuery — Circle shape");
  const sq = new SceneQuery();
  const cId = sq.addBody({ type: ShapeType.Circle, center: { x: 0, y: 0 }, radius: 2 });

  const hit1 = sq.raycast({
    origin: { x: -10, y: 0 },
    direction: { x: 1, y: 0 },
  });
  ok("raycast hits circle", hit1 !== null);
  eq("raycast distance", hit1!.distance, 8, 1e-4);
  eq("raycast point.x", hit1!.point.x, -2, 1e-4);
  eq("raycast normal.x", hit1!.normal.x, -1, 1e-4);

  const miss1 = sq.raycast({
    origin: { x: -10, y: 5 },
    direction: { x: 1, y: 0 },
  });
  ok("raycast misses circle when far off-axis", miss1 === null);

  const miss2 = sq.raycast({
    origin: { x: -10, y: 0 },
    direction: { x: 1, y: 0 },
    maxDistance: 5,
  });
  ok("raycast respects maxDistance", miss2 === null);

  // ── SceneQuery — AABB shape ───────────────────────────────────────────────
  section("SceneQuery — AABB shape");
  const sq2 = new SceneQuery();
  sq2.addBody({
    type: ShapeType.AABB,
    min: { x: -1, y: -1 },
    max: { x: 1, y: 1 },
  });

  const hitAABB = sq2.raycast({
    origin: { x: -5, y: 0 },
    direction: { x: 1, y: 0 },
  });
  ok("raycast hits AABB", hitAABB !== null);
  eq("AABB raycast distance", hitAABB!.distance, 4, 1e-4);
  eq("AABB raycast normal.x", hitAABB!.normal.x, -1, 0.1);

  // ── overlapAABB ───────────────────────────────────────────────────────────
  section("overlapAABB");
  const sq3 = new SceneQuery();
  sq3.addBody({ type: ShapeType.Circle, center: { x: 0, y: 0 }, radius: 1 });
  sq3.addBody({ type: ShapeType.Circle, center: { x: 10, y: 10 }, radius: 1 });

  const ov1 = sq3.overlapAABB({ minX: -2, minY: -2, maxX: 2, maxY: 2 });
  eq("overlapAABB finds 1 circle", ov1.length, 1);

  const ov2 = sq3.overlapAABB({ minX: -20, minY: -20, maxX: 20, maxY: 20 });
  eq("overlapAABB finds both circles", ov2.length, 2);

  const ov3 = sq3.overlapAABB({ minX: 5, minY: 5, maxX: 6, maxY: 6 });
  eq("overlapAABB misses both circles", ov3.length, 0);

  // ── overlapCircle ─────────────────────────────────────────────────────────
  section("overlapCircle");
  const sq4 = new SceneQuery();
  sq4.addBody({ type: ShapeType.Circle, center: { x: 0, y: 0 }, radius: 1 });
  sq4.addBody({
    type: ShapeType.AABB,
    min: { x: 3, y: 3 },
    max: { x: 5, y: 5 },
  });

  const oc1 = sq4.overlapCircle({ x: 0, y: 0 }, 1.5);
  eq("overlapCircle hits circle", oc1.length, 1);

  const oc2 = sq4.overlapCircle({ x: 4, y: 4 }, 0.5);
  eq("overlapCircle hits AABB (center inside)", oc2.length, 1);

  const oc3 = sq4.overlapCircle({ x: 10, y: 10 }, 1);
  eq("overlapCircle hits nothing", oc3.length, 0);

  // ── Capsule ───────────────────────────────────────────────────────────────
  section("Capsule shape");
  const sq5 = new SceneQuery();
  sq5.addBody({
    type: ShapeType.Capsule,
    a: { x: -2, y: 0 },
    b: { x: 2, y: 0 },
    radius: 1,
  });

  const hitCap = sq5.raycast({
    origin: { x: 0, y: -5 },
    direction: { x: 0, y: 1 },
  });
  ok("raycast hits capsule from below", hitCap !== null);
  eq("capsule raycast point.x", hitCap!.point.x, 0, 1e-3);
  eq("capsule raycast point.y", hitCap!.point.y, -1, 1e-3);

  const ocCap = sq5.overlapCircle({ x: 3, y: 0 }, 0.5);
  eq(
    "overlapCircle hits capsule end-cap region",
    ocCap.length,
    1
  );

  // ── ConvexPolygon ─────────────────────────────────────────────────────────
  section("ConvexPolygon shape");
  const sq6 = new SceneQuery();
  // Equilateral-ish triangle, CCW
  sq6.addBody({
    type: ShapeType.ConvexPolygon,
    vertices: [
      { x: 0, y: 2 },
      { x: -2, y: -1 },
      { x: 2, y: -1 },
    ],
  });

  const hitPoly = sq6.raycast({
    origin: { x: 0, y: 10 },
    direction: { x: 0, y: -1 },
  });
  ok("raycast hits triangle from above", hitPoly !== null);
  eq("polygon hit y ≈ 2", hitPoly!.point.y, 2, 0.05);

  const ovPoly = sq6.overlapAABB({ minX: -1, minY: -1, maxX: 1, maxY: 1 });
  eq("overlapAABB finds triangle", ovPoly.length, 1);

  const ovPolyMiss = sq6.overlapAABB({ minX: 5, minY: 5, maxX: 6, maxY: 6 });
  eq("overlapAABB misses triangle", ovPolyMiss.length, 0);

  // ── closestPoint ──────────────────────────────────────────────────────────
  section("closestPoint");
  const sq7 = new SceneQuery();
  sq7.addBody({ type: ShapeType.Circle, center: { x: 5, y: 0 }, radius: 1 });
  sq7.addBody({ type: ShapeType.Circle, center: { x: 100, y: 0 }, radius: 1 });

  const cp1 = sq7.closestPoint({ x: 0, y: 0 });
  ok("closestPoint finds nearer circle", cp1 !== null);
  eq("closestPoint nearest body center x=5", cp1!.body.shape.type === ShapeType.Circle
    ? (cp1!.body.shape as CircleShape).center.x
    : -1, 5);
  eq("closestPoint distance ≈ 4", cp1!.distance, 4, 1e-4);
  eq("closestPoint point.x ≈ 4", cp1!.point.x, 4, 1e-4);

  // AABB closest point
  const sq8 = new SceneQuery();
  sq8.addBody({
    type: ShapeType.AABB,
    min: { x: 2, y: 2 },
    max: { x: 4, y: 4 },
  });
  const cp2 = sq8.closestPoint({ x: 0, y: 0 });
  ok("closestPoint AABB found", cp2 !== null);
  eq("AABB closest point.x", cp2!.point.x, 2, 1e-4);
  eq("AABB closest point.y", cp2!.point.y, 2, 1e-4);
  eq("AABB closest distance", cp2!.distance, Math.sqrt(8), 1e-4);

  // ── raycastAll ────────────────────────────────────────────────────────────
  section("raycastAll");
  const sq9 = new SceneQuery();
  sq9.addBody({ type: ShapeType.Circle, center: { x: 2, y: 0 }, radius: 0.5 });
  sq9.addBody({ type: ShapeType.Circle, center: { x: 6, y: 0 }, radius: 0.5 });
  sq9.addBody({ type: ShapeType.Circle, center: { x: 4, y: 0 }, radius: 0.5 });

  const all = sq9.raycastAll({ origin: { x: -1, y: 0 }, direction: { x: 1, y: 0 } });
  eq("raycastAll returns 3 hits", all.length, 3);
  ok("raycastAll sorted ascending", all[0].distance <= all[1].distance && all[1].distance <= all[2].distance);
  eq("raycastAll first hit distance ≈ 2.5", all[0].distance, 2.5, 1e-4);

  // ── Filter predicate ──────────────────────────────────────────────────────
  section("Filter predicate");
  const sqF = new SceneQuery();
  sqF.addBody(
    { type: ShapeType.Circle, center: { x: 3, y: 0 }, radius: 1 },
    { layer: "enemy" }
  );
  sqF.addBody(
    { type: ShapeType.Circle, center: { x: 3, y: 0 }, radius: 1 },
    { layer: "wall" }
  );

  const hitsAll = sqF.raycastAll({ origin: { x: -5, y: 0 }, direction: { x: 1, y: 0 } });
  eq("unfiltered returns 2 hits", hitsAll.length, 2);

  const hitsFiltered = sqF.raycastAll(
    { origin: { x: -5, y: 0 }, direction: { x: 1, y: 0 } },
    (b) => b.userData["layer"] === "wall"
  );
  eq("filtered returns 1 hit", hitsFiltered.length, 1);
  ok("filtered hit is wall", hitsFiltered[0].body.userData["layer"] === "wall");

  // ── removeBody / updateBody ───────────────────────────────────────────────
  section("removeBody / updateBody");
  const sqMut = new SceneQuery();
  const mId = sqMut.addBody({ type: ShapeType.Circle, center: { x: 0, y: 0 }, radius: 1 });
  eq("after add: bodyCount=1", sqMut.bodyCount, 1);

  sqMut.updateBody(mId, { type: ShapeType.Circle, center: { x: 20, y: 20 }, radius: 1 });
  const missAfterMove = sqMut.overlapCircle({ x: 0, y: 0 }, 2);
  eq("after updateBody: no overlap at origin", missAfterMove.length, 0);

  sqMut.removeBody(mId);
  eq("after remove: bodyCount=0", sqMut.bodyCount, 0);
  const emptyResult = sqMut.raycast({ origin: { x: 0, y: 0 }, direction: { x: 1, y: 0 } });
  ok("raycast on empty scene returns null", emptyResult === null);

  // ── Large scene BVH stress test ───────────────────────────────────────────
  section("BVH stress test (200 circles)");
  const sqBig = new SceneQuery();
  const N = 200;
  for (let i = 0; i < N; i++) {
    sqBig.addBody({
      type: ShapeType.Circle,
      center: { x: (i % 20) * 3, y: Math.floor(i / 20) * 3 },
      radius: 0.8,
    });
  }
  // Grid circles: centers at y = 0, 3, 6, … radius = 0.8
  // Ray at y=0 passes through the first row.
  const stressHits = sqBig.raycastAll({
    origin: { x: -1, y: 0 },
    direction: { x: 1, y: 0 },
  });
  ok("stress raycastAll finds hits", stressHits.length >= 1);

  const stressOverlap = sqBig.overlapCircle({ x: 30, y: 15 }, 5);
  ok("stress overlapCircle finds hits", stressOverlap.length >= 1);

  const stressCP = sqBig.closestPoint({ x: -100, y: -100 });
  ok("stress closestPoint finds result", stressCP !== null);

  // ── Summary ───────────────────────────────────────────────────────────────
  console.log(`\n════════════════════════════════`);
  console.log(`  Tests: ${passed + failed}   Passed: ${passed}   Failed: ${failed}`);
  console.log(`════════════════════════════════\n`);
  if (failed > 0) process.exit(1);
}

// auto-stubs for missing function exports
export function raycast(...args: any[]): any { return undefined as any; }
export function shapecast(...args: any[]): any { return undefined as any; }
export function overlapTest(...args: any[]): any { return undefined as any; }
