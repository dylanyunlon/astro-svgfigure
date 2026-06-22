// contact-manifold.ts
// Contact point and manifold types with Sutherland-Hodgman clipping

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ContactPoint {
  x: number;
  y: number;
  nx: number;
  ny: number;
  depth: number;
  normalImpulse: number;
  tangentImpulse: number;
  featureId: number;
}

export interface Body {
  id: number;
  x: number;
  y: number;
  angle: number;
  vertices: Float64Array; // [x0,y0, x1,y1, ...]
  friction: number;
  restitution: number;
}

export interface ContactManifold {
  bodyA: Body;
  bodyB: Body;
  points: ContactPoint[];
  friction: number;
  restitution: number;
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

interface Vec2 { x: number; y: number }

function dot(ax: number, ay: number, bx: number, by: number): number {
  return ax * bx + ay * by;
}

function cross2(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}

function lerp(t: number, ax: number, ay: number, bx: number, by: number): Vec2 {
  return { x: ax + t * (bx - ax), y: ay + t * (by - ay) };
}

// ─── Polygon helpers ──────────────────────────────────────────────────────────

interface Edge {
  x0: number; y0: number;
  x1: number; y1: number;
  nx: number; ny: number; // inward normal
}

function getEdges(verts: Float64Array): Edge[] {
  const n = verts.length / 2;
  const edges: Edge[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const x0 = verts[i * 2],     y0 = verts[i * 2 + 1];
    const x1 = verts[j * 2],     y1 = verts[j * 2 + 1];
    const edgeX = x1 - x0, edgeY = y1 - y0;
    const len = Math.sqrt(edgeX * edgeX + edgeY * edgeY);
    // inward normal (right-hand polygon winding = CCW → inward = left of edge direction)
    edges.push({ x0, y0, x1, y1, nx: -edgeY / len, ny: edgeX / len });
  }
  return edges;
}

function getSupport(verts: Float64Array, nx: number, ny: number): number {
  // returns index of vertex with max projection along (nx,ny)
  let best = -Infinity, bestIdx = 0;
  const n = verts.length / 2;
  for (let i = 0; i < n; i++) {
    const d = dot(verts[i * 2], verts[i * 2 + 1], nx, ny);
    if (d > best) { best = d; bestIdx = i; }
  }
  return bestIdx;
}

// Returns the index of the edge of `verts` most perpendicular to (nx,ny)
function getBestEdge(verts: Float64Array, nx: number, ny: number): { i: number; j: number } {
  const si = getSupport(verts, nx, ny);
  const n = verts.length / 2;
  const prev = (si + n - 1) % n;
  const next = (si + 1) % n;
  const ex0 = verts[si * 2] - verts[prev * 2], ey0 = verts[si * 2 + 1] - verts[prev * 2 + 1];
  const ex1 = verts[next * 2] - verts[si * 2], ey1 = verts[next * 2 + 1] - verts[si * 2 + 1];
  const d0 = Math.abs(dot(ex0, ey0, nx, ny));
  const d1 = Math.abs(dot(ex1, ey1, nx, ny));
  return d0 <= d1 ? { i: prev, j: si } : { i: si, j: next };
}

// ─── Sutherland-Hodgman Clipping ──────────────────────────────────────────────

interface ClipVertex { x: number; y: number; featureId: number }

function clipPolygonAgainstEdge(
  poly: ClipVertex[],
  ex0: number, ey0: number,
  ex1: number, ey1: number,
  inwardNx: number, inwardNy: number
): ClipVertex[] {
  const out: ClipVertex[] = [];
  const n = poly.length;
  if (n === 0) return out;

  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    const da = dot(a.x - ex0, a.y - ey0, inwardNx, inwardNy);
    const db = dot(b.x - ex0, b.y - ey0, inwardNx, inwardNy);

    if (da >= 0) out.push(a);                                 // a inside

    if ((da > 0 && db < 0) || (da < 0 && db > 0)) {         // edge crosses plane
      const t = da / (da - db);
      const p = lerp(t, a.x, a.y, b.x, b.y);
      // featureId encodes the clipping edge index packed with vertex index
      out.push({ x: p.x, y: p.y, featureId: a.featureId | (0x8000) });
    }
  }
  return out;
}

function sutherlandHodgman(
  subject: ClipVertex[],
  clipVerts: Float64Array
): ClipVertex[] {
  const edges = getEdges(clipVerts);
  let poly = subject.slice();
  for (const e of edges) {
    if (poly.length === 0) break;
    poly = clipPolygonAgainstEdge(poly, e.x0, e.y0, e.x1, e.y1, e.nx, e.ny);
  }
  return poly;
}

// ─── SAT overlap on one axis ──────────────────────────────────────────────────

interface SATResult { depth: number; nx: number; ny: number }

function satOverlapOnAxis(
  vertsA: Float64Array,
  vertsB: Float64Array,
  nx: number,
  ny: number
): number {
  let minA = Infinity, maxA = -Infinity;
  let minB = Infinity, maxB = -Infinity;
  const na = vertsA.length / 2, nb = vertsB.length / 2;
  for (let i = 0; i < na; i++) {
    const d = dot(vertsA[i * 2], vertsA[i * 2 + 1], nx, ny);
    if (d < minA) minA = d;
    if (d > maxA) maxA = d;
  }
  for (let i = 0; i < nb; i++) {
    const d = dot(vertsB[i * 2], vertsB[i * 2 + 1], nx, ny);
    if (d < minB) minB = d;
    if (d > maxB) maxB = d;
  }
  return Math.min(maxA - minB, maxB - minA);
}

function findLeastPenetrationAxis(
  vertsA: Float64Array,
  vertsB: Float64Array
): SATResult | null {
  const na = vertsA.length / 2;
  let minDepth = Infinity, bestNx = 0, bestNy = 0;

  for (let i = 0; i < na; i++) {
    const j = (i + 1) % na;
    const ex = vertsA[j * 2] - vertsA[i * 2];
    const ey = vertsA[j * 2 + 1] - vertsA[i * 2 + 1];
    const len = Math.sqrt(ex * ex + ey * ey);
    const nx = -ey / len, ny = ex / len;
    const depth = satOverlapOnAxis(vertsA, vertsB, nx, ny);
    if (depth < 0) return null;                               // separating axis found
    if (depth < minDepth) { minDepth = depth; bestNx = nx; bestNy = ny; }
  }
  return { depth: minDepth, nx: bestNx, ny: bestNy };
}

// ─── Main: generateContacts ───────────────────────────────────────────────────

export function generateContacts(bodyA: Body, bodyB: Body): ContactManifold | null {
  const va = bodyA.vertices;
  const vb = bodyB.vertices;

  // SAT from A's edges
  const resA = findLeastPenetrationAxis(va, vb);
  if (!resA) return null;

  // SAT from B's edges
  const resB = findLeastPenetrationAxis(vb, va);
  if (!resB) return null;

  // Choose reference face (least penetration = most stable normal)
  let nx: number, ny: number, refVerts: Float64Array, incVerts: Float64Array;
  if (resA.depth <= resB.depth) {
    nx = resA.nx; ny = resA.ny;
    refVerts = va; incVerts = vb;
  } else {
    nx = -resB.nx; ny = -resB.ny;
    refVerts = vb; incVerts = va;
    // flip so normal always points from A to B
  }

  // Ensure normal points from A centroid toward B centroid
  let cAx = 0, cAy = 0, cBx = 0, cBy = 0;
  const nva = va.length / 2, nvb = vb.length / 2;
  for (let i = 0; i < nva; i++) { cAx += va[i*2]; cAy += va[i*2+1]; }
  for (let i = 0; i < nvb; i++) { cBx += vb[i*2]; cBy += vb[i*2+1]; }
  cAx /= nva; cAy /= nva; cBx /= nvb; cBy /= nvb;
  if (dot(cBx - cAx, cBy - cAy, nx, ny) < 0) { nx = -nx; ny = -ny; }

  // Reference edge on refVerts
  const refEdge = getBestEdge(refVerts, nx, ny);
  const rx0 = refVerts[refEdge.i * 2], ry0 = refVerts[refEdge.i * 2 + 1];
  const rx1 = refVerts[refEdge.j * 2], ry1 = refVerts[refEdge.j * 2 + 1];
  const refLen = Math.sqrt((rx1-rx0)*(rx1-rx0) + (ry1-ry0)*(ry1-ry0));
  const refTx = (rx1 - rx0) / refLen, refTy = (ry1 - ry0) / refLen;

  // Build incident polygon as ClipVertex list
  const incN = incVerts.length / 2;
  const subject: ClipVertex[] = [];
  for (let i = 0; i < incN; i++) {
    subject.push({ x: incVerts[i*2], y: incVerts[i*2+1], featureId: i });
  }

  // Clip incident polygon against reference polygon
  const clipped = sutherlandHodgman(subject, refVerts);
  if (clipped.length === 0) return null;

  // Keep only points behind the reference face plane (depth >= 0)
  const refD = dot(rx0, ry0, nx, ny);
  const contactPoints: ContactPoint[] = [];

  for (const cv of clipped) {
    const depth = refD - dot(cv.x, cv.y, nx, ny);
    if (depth >= 0) {
      contactPoints.push({
        x: cv.x,
        y: cv.y,
        nx,
        ny,
        depth,
        normalImpulse: 0,
        tangentImpulse: 0,
        featureId: cv.featureId,
      });
    }
  }

  if (contactPoints.length === 0) return null;

  return {
    bodyA,
    bodyB,
    points: contactPoints,
    friction: combineFriction(bodyA.friction, bodyB.friction),
    restitution: combineRestitution(bodyA.restitution, bodyB.restitution),
  };
}

// ─── Warm starting ────────────────────────────────────────────────────────────

export function warmStartManifold(
  current: ContactManifold,
  previous: ContactManifold
): void {
  const prevMap = new Map<number, ContactPoint>();
  for (const p of previous.points) prevMap.set(p.featureId, p);

  for (const cp of current.points) {
    const old = prevMap.get(cp.featureId & 0x7FFF);          // mask clipping bit
    if (old) {
      cp.normalImpulse  = old.normalImpulse;
      cp.tangentImpulse = old.tangentImpulse;
    }
  }
}

// ─── Combination rules ────────────────────────────────────────────────────────

/** Geometric mean — standard for friction */
export function combineFriction(fA: number, fB: number): number {
  return Math.sqrt(fA * fB);
}

/** Maximum — standard for restitution (bouncier surface wins) */
export function combineRestitution(rA: number, rB: number): number {
  return Math.max(rA, rB);
}
