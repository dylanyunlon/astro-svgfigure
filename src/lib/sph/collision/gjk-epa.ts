// gjk-epa.ts — GJK + EPA in 2D





export interface ConvexShape {
  support(dirX: number, dirY: number): [number, number];
}

export interface GJKResult {
  colliding: boolean;
  simplex: [number, number][];
}

export interface EPAResult {
  normal: [number, number];
  depth: number;
}

// ─── Shape factories ────────────────────────────────────────────────────────

export function createBoxShape(
  cx: number, cy: number, hw: number, hh: number
): ConvexShape {
  return {
    support(dirX: number, dirY: number): [number, number] {
      return [
        cx + (dirX >= 0 ? hw : -hw),
        cy + (dirY >= 0 ? hh : -hh),
      ];
    },
  };
}

export function createCircleShape(
  cx: number, cy: number, radius: number
): ConvexShape {
  return {
    support(dirX: number, dirY: number): [number, number] {
      const len = Math.sqrt(dirX * dirX + dirY * dirY);
      if (len < 1e-10) return [cx + radius, cy];
      return [
        cx + (dirX / len) * radius,
        cy + (dirY / len) * radius,
      ];
    },
  };
}

// ─── Math helpers ───────────────────────────────────────────────────────────

function dot(ax: number, ay: number, bx: number, by: number): number {
  return ax * bx + ay * by;
}

function cross2(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}

// 2D triple product: (A × B) × C  →  B(A·C) - A(B·C)
function tripleProduct(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number
): [number, number] {
  const ac = dot(ax, ay, cx, cy);
  const bc = dot(bx, by, cx, cy);
  return [bx * ac - ax * bc, by * ac - ay * bc];
}

function negate(x: number, y: number): [number, number] {
  return [-x, -y];
}

function sub(
  ax: number, ay: number, bx: number, by: number
): [number, number] {
  return [ax - bx, ay - by];
}

function normalize(x: number, y: number): [number, number] {
  const len = Math.sqrt(x * x + y * y);
  if (len < 1e-10) return [0, 1];
  return [x / len, y / len];
}

// ─── Minkowski support ───────────────────────────────────────────────────────

function minkSupport(
  A: ConvexShape, B: ConvexShape,
  dirX: number, dirY: number
): [number, number] {
  const [ax, ay] = A.support(dirX, dirY);
  const [bx, by] = B.support(-dirX, -dirY);
  return [ax - bx, ay - by];
}

// ─── GJK ─────────────────────────────────────────────────────────────────────

export function gjk(A: ConvexShape, B: ConvexShape): GJKResult {
  const noCol: GJKResult = { colliding: false, simplex: [] };

  let [dx, dy] = [1, 0];
  let simplex: [number, number][] = [];

  let [sx, sy] = minkSupport(A, B, dx, dy);
  simplex.push([sx, sy]);
  [dx, dy] = negate(sx, sy);

  const MAX_ITER = 64;

  for (let iter = 0; iter < MAX_ITER; iter++) {
    const lenD = Math.sqrt(dx * dx + dy * dy);
    if (lenD < 1e-10) return noCol;

    [sx, sy] = minkSupport(A, B, dx, dy);

    if (dot(sx, sy, dx, dy) < 0) {
      return noCol;
    }

    simplex.push([sx, sy]);

    const result = evolveSimplex(simplex);
    if (result.done) {
      return { colliding: true, simplex };
    }
    simplex = result.simplex;
    [dx, dy] = result.dir;
  }

  return noCol;
}

interface EvolveResult {
  done: boolean;
  simplex: [number, number][];
  dir: [number, number];
}

function evolveSimplex(simplex: [number, number][]): EvolveResult {
  if (simplex.length === 2) {
    return handleLine(simplex);
  }
  if (simplex.length === 3) {
    return handleTriangle(simplex);
  }
  return { done: false, simplex, dir: [1, 0] };
}

function handleLine(simplex: [number, number][]): EvolveResult {
  const [B, A] = simplex;
  const [ax, ay] = A;
  const [bx, by] = B;
  const [abx, aby] = sub(bx, by, ax, ay);
  const [aox, aoy] = negate(ax, ay);

  const [px, py] = tripleProduct(abx, aby, abx, aby, aox, aoy);
  const lenP = Math.sqrt(px * px + py * py);

  if (lenP < 1e-10) {
    // Origin is on the line segment, perpendicular is zero
    // Pick any perpendicular
    return {
      done: false,
      simplex,
      dir: normalize(-aby, abx),
    };
  }

  return {
    done: false,
    simplex,
    dir: normalize(px, py),
  };
}

function handleTriangle(simplex: [number, number][]): EvolveResult {
  const [C, B, A] = simplex;
  const [ax, ay] = A;
  const [bx, by] = B;
  const [cx, cy] = C;

  const [abx, aby] = sub(bx, by, ax, ay);
  const [acx, acy] = sub(cx, cy, ax, ay);
  const [aox, aoy] = negate(ax, ay);

  // Perpendicular to AB away from C
  const [abPerpX, abPerpY] = tripleProduct(acx, acy, abx, aby, abx, aby);
  // Perpendicular to AC away from B
  const [acPerpX, acPerpY] = tripleProduct(abx, aby, acx, acy, acx, acy);

  if (dot(abPerpX, abPerpY, aox, aoy) > 0) {
    // Origin is in AB region
    return {
      done: false,
      simplex: [B, A],
      dir: normalize(abPerpX, abPerpY),
    };
  }

  if (dot(acPerpX, acPerpY, aox, aoy) > 0) {
    // Origin is in AC region
    return {
      done: false,
      simplex: [C, A],
      dir: normalize(acPerpX, acPerpY),
    };
  }

  // Origin is inside triangle
  return { done: true, simplex, dir: [0, 0] };
}

// ─── EPA ─────────────────────────────────────────────────────────────────────

export function epa(
  A: ConvexShape,
  B: ConvexShape,
  simplex: [number, number][]
): EPAResult {
  const MAX_ITER = 64;
  const EPSILON = 1e-6;

  let poly: [number, number][] = [...simplex];

  for (let iter = 0; iter < MAX_ITER; iter++) {
    const { normal, depth, index } = getClosestEdge(poly);

    const [sx, sy] = minkSupport(A, B, normal[0], normal[1]);
    const d = dot(sx, sy, normal[0], normal[1]);

    if (Math.abs(d - depth) < EPSILON) {
      return { normal, depth };
    }

    poly = [
      ...poly.slice(0, index + 1),
      [sx, sy],
      ...poly.slice(index + 1),
    ] as [number, number][];
  }

  const { normal, depth } = getClosestEdge(poly);
  return { normal, depth };
}

interface ClosestEdge {
  normal: [number, number];
  depth: number;
  index: number;
}

function getClosestEdge(poly: [number, number][]): ClosestEdge {
  let minDepth = Infinity;
  let minNormal: [number, number] = [0, 1];
  let minIndex = 0;

  for (let i = 0; i < poly.length; i++) {
    const j = (i + 1) % poly.length;
    const [ax, ay] = poly[i];
    const [bx, by] = poly[j];

    const [ex, ey] = sub(bx, by, ax, ay);

    // Outward-facing normal (CCW winding assumed): rotate edge 90° left
    let [nx, ny] = normalize(ey, -ex);

    const depth = dot(nx, ny, ax, ay);

    // Ensure normal points away from origin
    if (depth < 0) {
      nx = -nx;
      ny = -ny;
    }

    const posDep = Math.abs(depth);

    if (posDep < minDepth) {
      minDepth = posDep;
      minNormal = [nx, ny];
      minIndex = i;
    }
  }

  return { normal: minNormal, depth: minDepth, index: minIndex };
}
