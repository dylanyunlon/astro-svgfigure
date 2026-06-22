// =============================================================================
// GJK.ts — GJK Collision Detection
// Minkowski Difference + Simplex  |  Circle / Polygon / AABB support
// =============================================================================

// ---------------------------------------------------------------------------
// Vec2 — 2D vector helpers
// ---------------------------------------------------------------------------

interface Vec2 {
  x: number;
  y: number;
}

const v = {
  make: (x: number, y: number): Vec2 => ({ x, y }),
  add:  (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y }),
  sub:  (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y }),
  neg:  (a: Vec2): Vec2           => ({ x: -a.x, y: -a.y }),
  scale:(a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s }),
  dot:  (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y,
  len2: (a: Vec2): number          => a.x * a.x + a.y * a.y,
  len:  (a: Vec2): number          => Math.sqrt(v.len2(a)),
  norm: (a: Vec2): Vec2 => {
    const l = v.len(a);
    return l < 1e-12 ? { x: 0, y: 0 } : { x: a.x / l, y: a.y / l };
  },
  /** Triple product: (a × b) × c  ≡  b*(a·c) - a*(b·c)  [2-D stays in 2-D] */
  tripleProduct: (a: Vec2, b: Vec2, c: Vec2): Vec2 => {
    const ac = v.dot(a, c);
    const bc = v.dot(b, c);
    return v.sub(v.scale(b, ac), v.scale(a, bc));
  },
  /** 2-D "cross" scalar: a.x*b.y - a.y*b.x */
  cross2: (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x,
  /** Perpendicular pointing toward ref */
  perp: (edge: Vec2, ref: Vec2): Vec2 => {
    const p = v.make(-edge.y, edge.x);
    return v.dot(p, ref) >= 0 ? p : v.neg(p);
  },
  eq: (a: Vec2, b: Vec2, eps = 1e-10): boolean =>
    Math.abs(a.x - b.x) < eps && Math.abs(a.y - b.y) < eps,
};

// ---------------------------------------------------------------------------
// Shape — support-function abstraction
// ---------------------------------------------------------------------------

/** A shape that can produce its furthest point in a given direction. */
interface Shape {
  support(dir: Vec2): Vec2;
}

// ---------------------------------------------------------------------------
// Circle
// ---------------------------------------------------------------------------

export class Circle implements Shape {
  center: Vec2;
  radius: number;
  constructor(center: Vec2, radius: number) {
    this.center = center;
    this.radius = radius;
  }

  support(dir: Vec2): Vec2 {
    const n = v.norm(dir);
    return v.add(this.center, v.scale(n, this.radius));
  }
}

// ---------------------------------------------------------------------------
// Polygon  (convex hull assumed)
// ---------------------------------------------------------------------------

export class Polygon implements Shape {
  readonly vertices: Vec2[];

  constructor(vertices: Vec2[]) {
    if (vertices.length < 3) throw new Error("Polygon needs ≥ 3 vertices");
    this.vertices = vertices;
  }

  support(dir: Vec2): Vec2 {
    let best = this.vertices[0];
    let bestDot = v.dot(best, dir);
    for (let i = 1; i < this.vertices.length; i++) {
      const d = v.dot(this.vertices[i], dir);
      if (d > bestDot) { bestDot = d; best = this.vertices[i]; }
    }
    return best;
  }

  /** Convenience: axis-aligned box polygon. */
  static fromAABB(minX: number, minY: number, maxX: number, maxY: number): Polygon {
    return new Polygon([
      v.make(minX, minY),
      v.make(maxX, minY),
      v.make(maxX, maxY),
      v.make(minX, maxY),
    ]);
  }
}

// ---------------------------------------------------------------------------
// AABB  (dedicated class — support is O(1))
// ---------------------------------------------------------------------------

export class AABB implements Shape {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  constructor(minX: number, minY: number, maxX: number, maxY: number) {
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  }

  support(dir: Vec2): Vec2 {
    return v.make(
      dir.x >= 0 ? this.maxX : this.minX,
      dir.y >= 0 ? this.maxY : this.minY,
    );
  }

  get center(): Vec2 {
    return v.make((this.minX + this.maxX) / 2, (this.minY + this.maxY) / 2);
  }
}

// ---------------------------------------------------------------------------
// Minkowski-difference support
// ---------------------------------------------------------------------------

function minkowskiSupport(a: Shape, b: Shape, dir: Vec2): Vec2 {
  return v.sub(a.support(dir), b.support(v.neg(dir)));
}

// ---------------------------------------------------------------------------
// GJK — simplex evolution
// ---------------------------------------------------------------------------

type Simplex = Vec2[];  // 1, 2, or 3 points; last point = most-recently added

/**
 * Given a simplex that contains the latest point (last element),
 * update the simplex to the sub-feature closest to the origin and
 * return the next search direction.
 * Returns null when the origin is inside the triangle (collision).
 */
function doSimplex(simplex: Simplex): Vec2 | null {
  if (simplex.length === 2) return lineCase(simplex);
  return triangleCase(simplex);
}

function lineCase(simplex: Simplex): Vec2 {
  // simplex = [A, B], B is newest
  const [a, b] = simplex;
  const ab = v.sub(a, b);
  const bo = v.neg(b);               // B→origin
  if (v.dot(ab, bo) > 0) {
    // origin lies in the Voronoi region of edge AB
    return v.tripleProduct(ab, bo, ab);
  }
  // origin is beyond B — keep just B
  simplex.splice(0, 1);              // remove A
  return bo;
}

function triangleCase(simplex: Simplex): Vec2 | null {
  // simplex = [A, B, C], C is newest
  const [a, b, c] = simplex;
  const ca = v.sub(a, c);
  const cb = v.sub(b, c);
  const co = v.neg(c);               // C→origin

  const caPerp = v.tripleProduct(cb, ca, ca);  // perpendicular to CA, away from B
  const cbPerp = v.tripleProduct(ca, cb, cb);  // perpendicular to CB, away from A

  if (v.dot(caPerp, co) > 0) {
    // origin outside CA edge — reduce to line [A, C]
    simplex.length = 0;
    simplex.push(a, c);
    return caPerp;
  }

  if (v.dot(cbPerp, co) > 0) {
    // origin outside CB edge — reduce to line [B, C]
    simplex.splice(0, 1);            // remove A  → [B, C]
    return cbPerp;
  }

  // Origin is inside the triangle — collision!
  return null;
}

// ---------------------------------------------------------------------------
// GJK entry point
// ---------------------------------------------------------------------------

export interface GJKResult {
  colliding: boolean;
  simplex: Simplex;
}

const MAX_ITERATIONS = 64;

export function gjk(shapeA: Shape, shapeB: Shape): GJKResult {
  // Initial direction: vector between approximate centres
  let dir = v.make(1, 0);

  // Prefer center-based initial direction when available
  const ca = (shapeA as any).center as Vec2 | undefined;
  const cb = (shapeB as any).center as Vec2 | undefined;
  if (ca && cb) {
    const d = v.sub(cb, ca);
    if (v.len2(d) > 1e-20) dir = d;
  }

  const simplex: Simplex = [];
  let support = minkowskiSupport(shapeA, shapeB, dir);
  simplex.push(support);
  dir = v.neg(support);   // search toward origin

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    if (v.len2(dir) < 1e-20) {
      // direction collapsed → origin is exactly on boundary → colliding
      return { colliding: true, simplex };
    }

    support = minkowskiSupport(shapeA, shapeB, dir);

    // If the new point did not pass the origin → no collision
    if (v.dot(support, dir) < 0) {
      return { colliding: false, simplex };
    }

    simplex.push(support);

    const nextDir = doSimplex(simplex);
    if (nextDir === null) {
      // Origin inside simplex → collision confirmed
      return { colliding: true, simplex };
    }
    dir = nextDir;
  }

  // Reached iteration limit — treat as non-colliding (shouldn't happen on well-formed shapes)
  return { colliding: false, simplex };
}

// ---------------------------------------------------------------------------
// EPA — Expanding Polytope Algorithm (penetration depth + normal)
// ---------------------------------------------------------------------------

export interface EPAResult {
  depth: number;
  normal: Vec2;
}

export function epa(shapeA: Shape, shapeB: Shape, simplex: Simplex): EPAResult {
  const polytope: Vec2[] = [...simplex];
  // Ensure CCW winding
  if (v.cross2(v.sub(polytope[1], polytope[0]), v.sub(polytope[2], polytope[0])) < 0) {
    [polytope[0], polytope[1]] = [polytope[1], polytope[0]];
  }

  for (let iter = 0; iter < 64; iter++) {
    let minDist = Infinity;
    let minIdx  = 0;
    let minNorm = v.make(0, 0);

    // Find edge closest to origin
    for (let i = 0; i < polytope.length; i++) {
      const j = (i + 1) % polytope.length;
      const a = polytope[i];
      const b = polytope[j];
      const ab = v.sub(b, a);
      // outward normal for CCW polytope
      let n = v.make(ab.y, -ab.x);
      n = v.norm(n);
      const d = v.dot(n, a);
      if (d < minDist) { minDist = d; minIdx = j; minNorm = n; }
    }

    const sup = minkowskiSupport(shapeA, shapeB, minNorm);
    const dist = v.dot(minNorm, sup);

    if (Math.abs(dist - minDist) < 1e-8) {
      return { depth: minDist, normal: minNorm };
    }
    // Insert new point into polytope
    polytope.splice(minIdx, 0, sup);
  }

  return { depth: 0, normal: v.make(0, 0) };
}

// ---------------------------------------------------------------------------
// High-level helper: full collision info
// ---------------------------------------------------------------------------

export interface CollisionInfo {
  colliding: boolean;
  depth?: number;
  normal?: Vec2;   // points from B to A (push A out)
}

export function collide(shapeA: Shape, shapeB: Shape): CollisionInfo {
  const result = gjk(shapeA, shapeB);
  if (!result.colliding) return { colliding: false };

  // Build a proper triangle simplex for EPA if we only have 1 or 2 points
  let simplex = result.simplex;
  if (simplex.length < 3) {
    const dirs: Vec2[] = [
      v.make(1, 0), v.make(-1, 0), v.make(0, 1), v.make(0, -1),
      v.make(1, 1), v.make(-1, 1),
    ];
    for (const d of dirs) {
      if (simplex.length >= 3) break;
      const s = minkowskiSupport(shapeA, shapeB, d);
      if (!simplex.some(p => v.eq(p, s))) simplex.push(s);
    }
    if (simplex.length < 3) {
      return { colliding: true, depth: 0, normal: v.make(0, 1) };
    }
  }

  const { depth, normal } = epa(shapeA, shapeB, simplex);
  return { colliding: true, depth, normal };
}

// =============================================================================
// TESTS
// =============================================================================

type TestFn = () => void;
const _tests: { name: string; fn: TestFn }[] = [];
function test(name: string, fn: TestFn) { _tests.push({ name, fn }); }

function assertEqual(actual: boolean, expected: boolean, msg: string) {
  if (actual !== expected) throw new Error(`FAIL: ${msg} — expected ${expected}, got ${actual}`);
}
function assertClose(actual: number, expected: number, eps = 1e-4, msg = "") {
  if (Math.abs(actual - expected) > eps)
    throw new Error(`FAIL: ${msg} — expected ≈${expected}, got ${actual}`);
}

// ── Circle vs Circle ──────────────────────────────────────────────────────

test("Circle-Circle overlapping", () => {
  const a = new Circle(v.make(0, 0), 1);
  const b = new Circle(v.make(1, 0), 1);
  assertEqual(gjk(a, b).colliding, true, "circles at dist=1 with r=1 each");
});

test("Circle-Circle touching (boundary)", () => {
  const a = new Circle(v.make(0, 0), 1);
  const b = new Circle(v.make(2, 0), 1);
  assertEqual(gjk(a, b).colliding, true, "circles exactly touching");
});

test("Circle-Circle separated", () => {
  const a = new Circle(v.make(0, 0), 1);
  const b = new Circle(v.make(3, 0), 1);
  assertEqual(gjk(a, b).colliding, false, "circles separated by gap");
});

test("Circle-Circle same center", () => {
  const a = new Circle(v.make(2, 3), 1);
  const b = new Circle(v.make(2, 3), 1);
  assertEqual(gjk(a, b).colliding, true, "circles coincident");
});

// ── AABB vs AABB ─────────────────────────────────────────────────────────

test("AABB-AABB overlapping", () => {
  const a = new AABB(0, 0, 2, 2);
  const b = new AABB(1, 1, 3, 3);
  assertEqual(gjk(a, b).colliding, true, "AABBs overlap at corner");
});

test("AABB-AABB touching edge", () => {
  const a = new AABB(0, 0, 2, 2);
  const b = new AABB(2, 0, 4, 2);
  assertEqual(gjk(a, b).colliding, true, "AABBs share edge x=2");
});

test("AABB-AABB separated", () => {
  const a = new AABB(0, 0, 1, 1);
  const b = new AABB(2, 2, 3, 3);
  assertEqual(gjk(a, b).colliding, false, "AABBs with diagonal gap");
});

test("AABB-AABB one inside other", () => {
  const a = new AABB(0, 0, 10, 10);
  const b = new AABB(3, 3,  7,  7);
  assertEqual(gjk(a, b).colliding, true, "small AABB fully inside large AABB");
});

// ── Polygon vs Polygon ────────────────────────────────────────────────────

test("Triangle-Triangle overlapping", () => {
  const a = new Polygon([v.make(0, 0), v.make(4, 0), v.make(2, 4)]);
  const b = new Polygon([v.make(2, 1), v.make(6, 1), v.make(4, 5)]);
  assertEqual(gjk(a, b).colliding, true, "triangles overlap");
});

test("Triangle-Triangle separated", () => {
  const a = new Polygon([v.make(0, 0), v.make(1, 0), v.make(0.5, 1)]);
  const b = new Polygon([v.make(5, 0), v.make(6, 0), v.make(5.5, 1)]);
  assertEqual(gjk(a, b).colliding, false, "triangles far apart");
});

test("Square polygon vs square polygon overlap", () => {
  const makeSquare = (cx: number, cy: number, s: number) =>
    new Polygon([
      v.make(cx - s, cy - s), v.make(cx + s, cy - s),
      v.make(cx + s, cy + s), v.make(cx - s, cy + s),
    ]);
  const a = makeSquare(0, 0, 1);
  const b = makeSquare(1.5, 0, 1);
  assertEqual(gjk(a, b).colliding, true, "squares overlap by 0.5");
});

test("Square polygon vs square polygon separated", () => {
  const makeSquare = (cx: number, cy: number, s: number) =>
    new Polygon([
      v.make(cx - s, cy - s), v.make(cx + s, cy - s),
      v.make(cx + s, cy + s), v.make(cx - s, cy + s),
    ]);
  const a = makeSquare(0, 0, 1);
  const b = makeSquare(3, 0, 1);
  assertEqual(gjk(a, b).colliding, false, "squares separated");
});

// ── Cross-type tests ───────────────────────────────────────────────────────

test("Circle vs AABB overlapping", () => {
  const c = new Circle(v.make(1, 1), 1);
  const box = new AABB(1, 1, 4, 4);
  assertEqual(gjk(c, box).colliding, true, "circle center inside AABB");
});

test("Circle vs AABB separated", () => {
  const c = new Circle(v.make(-3, 0), 1);
  const box = new AABB(0, 0, 2, 2);
  assertEqual(gjk(c, box).colliding, false, "circle to left of AABB");
});

test("Circle vs Polygon overlapping", () => {
  const c  = new Circle(v.make(0, 0), 2);
  const tri = new Polygon([v.make(1, 0), v.make(3, 0), v.make(2, 2)]);
  assertEqual(gjk(c, tri).colliding, true, "circle overlaps triangle");
});

test("Circle vs Polygon separated", () => {
  const c   = new Circle(v.make(-5, 0), 1);
  const tri = new Polygon([v.make(1, 0), v.make(3, 0), v.make(2, 2)]);
  assertEqual(gjk(c, tri).colliding, false, "circle far from triangle");
});

test("AABB vs Polygon overlapping", () => {
  const box = new AABB(-1, -1, 1, 1);
  const tri = new Polygon([v.make(0, 0), v.make(3, 0), v.make(1.5, 3)]);
  assertEqual(gjk(box, tri).colliding, true, "AABB overlaps triangle at origin");
});

test("AABB vs Polygon separated", () => {
  const box = new AABB(0, 0, 1, 1);
  const tri = new Polygon([v.make(5, 0), v.make(8, 0), v.make(6.5, 3)]);
  assertEqual(gjk(box, tri).colliding, false, "AABB and triangle far apart");
});

// ── Minkowski support sanity ───────────────────────────────────────────────

test("Minkowski support sanity: two circles", () => {
  const a = new Circle(v.make(0, 0), 1);
  const b = new Circle(v.make(3, 0), 1);
  // dir=(1,0): a.support=(1,0), b.support(-1,0)=(3-1,0)=(2,0) → diff = 1-2 = -1
  const s = minkowskiSupport(a, b, v.make(1, 0));
  assertClose(s.x, -1, 1e-6, "Mink support x");
  assertClose(s.y,  0, 1e-6, "Mink support y");
  // Also verify dir=(-1,0): a.support=(-1,0), b.support=(4,0) → diff = -1-4 = -5
  const s2 = minkowskiSupport(a, b, v.make(-1, 0));
  assertClose(s2.x, -5, 1e-6, "Mink support x dir=-1");
});

// ── EPA / collide penetration depth ──────────────────────────────────────

test("EPA: circles overlap depth", () => {
  const a = new Circle(v.make(0, 0), 2);
  const b = new Circle(v.make(2, 0), 2);
  const info = collide(a, b);
  assertEqual(info.colliding, true, "circles overlap for EPA");
  // penetration depth = (r_a + r_b) - dist = 4 - 2 = 2
  assertClose(info.depth!, 2, 0.01, "EPA depth circles");
});

test("EPA: AABB overlap depth", () => {
  const a = new AABB(0, 0, 4, 4);
  const b = new AABB(3, 0, 7, 4);
  const info = collide(a, b);
  assertEqual(info.colliding, true, "AABBs overlap for EPA");
  // overlap in X = 1, overlap in Y = 4 → min = 1
  assertClose(info.depth!, 1, 0.05, "EPA depth AABBs");
});

// ── Regression: Pentagon ──────────────────────────────────────────────────

test("Regular pentagon vs point-circle", () => {
  const N = 5;
  const verts = Array.from({ length: N }, (_, i) => {
    const a = (2 * Math.PI * i) / N;
    return v.make(Math.cos(a), Math.sin(a));
  });
  const penta = new Polygon(verts);
  const inside = new Circle(v.make(0.3, 0.1), 0.05);
  const outside = new Circle(v.make(3, 0), 0.1);
  assertEqual(gjk(penta, inside).colliding,  true,  "small circle inside pentagon");
  assertEqual(gjk(penta, outside).colliding, false, "small circle outside pentagon");
});

// ── Run all tests ─────────────────────────────────────────────────────────

function runTests() {
  let passed = 0, failed = 0;
  const results: string[] = [];

  for (const { name, fn } of _tests) {
    try {
      fn();
      results.push(`  ✓  ${name}`);
      passed++;
    } catch (e: any) {
      results.push(`  ✗  ${name}\n       ${e.message}`);
      failed++;
    }
  }

  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║              GJK.ts — Test Suite Results                ║");
  console.log("╚══════════════════════════════════════════════════════════╝\n");
  results.forEach(r => console.log(r));
  console.log(
    `\n  Total: ${passed + failed}  |  Passed: ${passed}  |  Failed: ${failed}\n`,
  );

  if (failed > 0) process.exit(1);
}

runTests();

// =============================================================================
// DEMO — quick visual summary of collide()
// =============================================================================

console.log("══════════════════════════════════════════════════════════════");
console.log("  collide() demo");
console.log("══════════════════════════════════════════════════════════════");

const demoA = new Circle(v.make(0, 0), 2);
const demoB = new Circle(v.make(2, 0), 2);
const info1 = collide(demoA, demoB);
console.log(`Circle(r=2,c=0,0) ∩ Circle(r=2,c=2,0): colliding=${info1.colliding}, depth=${info1.depth?.toFixed(4)}, normal=(${info1.normal?.x.toFixed(3)},${info1.normal?.y.toFixed(3)})`);

const boxA = new AABB(0, 0, 4, 2);
const boxB = new AABB(3, 0, 7, 2);
const info2 = collide(boxA, boxB);
console.log(`AABB(0,0,4,2) ∩ AABB(3,0,7,2):          colliding=${info2.colliding}, depth=${info2.depth?.toFixed(4)}, normal=(${info2.normal?.x.toFixed(3)},${info2.normal?.y.toFixed(3)})`);

const tri = new Polygon([v.make(-1, -1), v.make(1, -1), v.make(0, 1)]);
const circ = new Circle(v.make(0.5, 0), 0.3);
const info3 = collide(tri, circ);
console.log(`Triangle ∩ Circle(r=0.3,c=0.5,0):       colliding=${info3.colliding}, depth=${info3.depth?.toFixed(4)}`);

const far = new Circle(v.make(10, 0), 1);
const info4 = collide(demoA, far);
console.log(`Circle(r=2,c=0,0) ∩ Circle(r=1,c=10,0): colliding=${info4.colliding}`);

console.log("══════════════════════════════════════════════════════════════\n");
