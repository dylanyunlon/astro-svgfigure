// contact-generator.ts
// Narrow-phase contact generation for RigidBody2D pairs.
// Produces ContactConstraint records consumed by ContactSolver / impulse-solver.
//
// Supported shape combinations:
//   Circle  ↔ Circle
//   Circle  ↔ Polygon
//   Polygon ↔ Polygon  (SAT + Sutherland-Hodgman clipping)


import {
} from './math';
import {
} from './rigid-body';

  Vec2,
  vec2,
  vec2Add,
  vec2Sub,
  vec2Scale,
  vec2Dot,
  vec2Cross,
  vec2Negate,
  vec2Length,
  vec2LengthSq,
  vec2Normalize,
  vec2Distance,

  RigidBody2D,
  ShapeKind,
  bodyToWorld,
  bodyDirToWorld,
  getWorldVertices,
  getWorldNormals,

// ─── Contact output ───────────────────────────────────────────────────────────
// Compatible with the ContactConstraint in ../types.ts and ContactSolver.ts

export interface GeneratedContact {
  bodyA: number;   // id into body array
  bodyB: number;
  normal: Vec2;    // world-space, points from A toward B
  point: Vec2;     // world-space contact point
  depth: number;   // penetration depth (positive = overlapping)
  normalImpulse: number;
  tangentImpulse: number;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Generate contacts for a pair of bodies.
 * Returns an array of 0–2 contact points (polygons may have two-point manifolds).
 */
export function generateContacts(
  bodyA: RigidBody2D,
  bodyB: RigidBody2D,
): GeneratedContact[] {
  const kindA = bodyA.shape.kind;
  const kindB = bodyB.shape.kind;

  if (kindA === ShapeKind.Circle && kindB === ShapeKind.Circle) {
    return circleVsCircle(bodyA, bodyB);
  }
  if (kindA === ShapeKind.Circle && kindB === ShapeKind.Polygon) {
    return circleVsPolygon(bodyA, bodyB, false);
  }
  if (kindA === ShapeKind.Polygon && kindB === ShapeKind.Circle) {
    return circleVsPolygon(bodyB, bodyA, true);
  }
  // Polygon vs Polygon
  return polygonVsPolygon(bodyA, bodyB);
}

/**
 * Batch contact generation for an array of body pairs (indices).
 * Typically fed by a broad-phase (sort-and-sweep, BVH, etc.).
 */
export function generateAllContacts(
  bodies: RigidBody2D[],
  pairs: ReadonlyArray<[number, number]>,
): GeneratedContact[] {
  const contacts: GeneratedContact[] = [];
  for (const [i, j] of pairs) {
    const cs = generateContacts(bodies[i], bodies[j]);
    for (const c of cs) contacts.push(c);
  }
  return contacts;
}

// ─── Circle vs Circle ─────────────────────────────────────────────────────────

function circleVsCircle(a: RigidBody2D, b: RigidBody2D): GeneratedContact[] {
  if (a.shape.kind !== ShapeKind.Circle || b.shape.kind !== ShapeKind.Circle) return [];

  const cA = vec2Add(a.position, a.shape.offset);
  const cB = vec2Add(b.position, b.shape.offset);
  const rA = a.shape.radius;
  const rB = b.shape.radius;

  const d = vec2Sub(cB, cA);
  const distSq = vec2LengthSq(d);
  const totalR = rA + rB;

  if (distSq >= totalR * totalR) return [];

  const dist = Math.sqrt(distSq);
  const normal = dist > 1e-10
    ? vec2Scale(d, 1 / dist)
    : vec2(0, 1); // degenerate — pick arbitrary normal

  const depth = totalR - dist;
  // Contact point: midpoint of the overlap region on the line between centres
  const point = vec2Add(cA, vec2Scale(normal, rA - depth * 0.5));

  return [{
    bodyA: a.id,
    bodyB: b.id,
    normal,
    point,
    depth,
    normalImpulse: 0,
    tangentImpulse: 0,
  }];
}

// ─── Circle vs Polygon ───────────────────────────────────────────────────────

function circleVsPolygon(
  circleBody: RigidBody2D,
  polyBody: RigidBody2D,
  swapped: boolean,
): GeneratedContact[] {
  if (circleBody.shape.kind !== ShapeKind.Circle) return [];
  if (polyBody.shape.kind !== ShapeKind.Polygon) return [];

  const center = vec2Add(circleBody.position, circleBody.shape.offset);
  const radius = circleBody.shape.radius;
  const verts = getWorldVertices(polyBody);
  const normals = getWorldNormals(polyBody);
  const n = verts.length;

  // Find the edge with deepest penetration of the circle centre
  let maxSep = -Infinity;
  let bestEdge = 0;

  for (let i = 0; i < n; i++) {
    const sep = vec2Dot(vec2Sub(center, verts[i]), normals[i]);
    if (sep > radius) return []; // separating axis found
    if (sep > maxSep) {
      maxSep = sep;
      bestEdge = i;
    }
  }

  // Determine contact point: vertex region or edge region
  const v0 = verts[bestEdge];
  const v1 = verts[(bestEdge + 1) % n];
  const edge = vec2Sub(v1, v0);
  const edgeLenSq = vec2LengthSq(edge);

  let contactPoint: Vec2;
  let contactNormal: Vec2;
  let depth: number;

  if (edgeLenSq < 1e-12) {
    // Degenerate edge — treat as vertex
    const d = vec2Sub(center, v0);
    const dist = vec2Length(d);
    if (dist > radius) return [];
    contactNormal = dist > 1e-10 ? vec2Scale(d, 1 / dist) : vec2(0, 1);
    depth = radius - dist;
    contactPoint = vec2Add(v0, vec2Scale(contactNormal, depth * 0.5));
  } else {
    const t = vec2Dot(vec2Sub(center, v0), edge) / edgeLenSq;

    if (t <= 0) {
      // Closest to v0
      const d = vec2Sub(center, v0);
      const dist = vec2Length(d);
      if (dist > radius) return [];
      contactNormal = dist > 1e-10 ? vec2Scale(d, 1 / dist) : normals[bestEdge];
      depth = radius - dist;
      contactPoint = v0;
    } else if (t >= 1) {
      // Closest to v1
      const d = vec2Sub(center, v1);
      const dist = vec2Length(d);
      if (dist > radius) return [];
      contactNormal = dist > 1e-10 ? vec2Scale(d, 1 / dist) : normals[bestEdge];
      depth = radius - dist;
      contactPoint = v1;
    } else {
      // Closest to edge interior
      contactNormal = normals[bestEdge];
      depth = radius - maxSep;
      const closest = vec2Add(v0, vec2Scale(edge, t));
      contactPoint = closest;
    }
  }

  // Convention: normal points from A to B
  if (swapped) {
    // polyBody is the original A, circleBody is original B
    return [{
      bodyA: polyBody.id,
      bodyB: circleBody.id,
      normal: vec2Negate(contactNormal),
      point: contactPoint,
      depth,
      normalImpulse: 0,
      tangentImpulse: 0,
    }];
  }

  return [{
    bodyA: circleBody.id,
    bodyB: polyBody.id,
    normal: contactNormal,
    point: contactPoint,
    depth,
    normalImpulse: 0,
    tangentImpulse: 0,
  }];
}

// ─── Polygon vs Polygon (SAT + clipping) ──────────────────────────────────────

interface SATAxisResult {
  separation: number;
  edgeIndex: number;
}

/**
 * Find the edge of `refVerts` with the largest (least-negative) separation
 * from `incVerts`.  If any axis has positive separation → no collision.
 */
function findMaxSeparation(
  refVerts: Vec2[],
  refNormals: Vec2[],
  incVerts: Vec2[],
): SATAxisResult {
  let maxSep = -Infinity;
  let bestIdx = 0;

  for (let i = 0; i < refVerts.length; i++) {
    const n = refNormals[i];
    const v = refVerts[i];

    // Project incident vertices onto the reference edge normal
    let minProj = Infinity;
    for (const iv of incVerts) {
      const proj = vec2Dot(vec2Sub(iv, v), n);
      if (proj < minProj) minProj = proj;
    }

    if (minProj > maxSep) {
      maxSep = minProj;
      bestIdx = i;
    }
  }

  return { separation: maxSep, edgeIndex: bestIdx };
}

/**
 * Clip the incident edge segment against the reference edge's side planes,
 * then keep only points behind the reference face.
 * Returns 0–2 contact points.
 */
function clipEdges(
  refV0: Vec2,
  refV1: Vec2,
  refNormal: Vec2,
  incV0: Vec2,
  incV1: Vec2,
): Array<{ point: Vec2; depth: number }> {
  const refEdge = vec2Sub(refV1, refV0);
  const refLen = vec2Length(refEdge);
  if (refLen < 1e-12) return [];
  const refTangent = vec2Scale(refEdge, 1 / refLen);

  // Clip incident edge against the two side planes of the reference edge
  let points: Vec2[] = [incV0, incV1];

  // Side plane 1: perpendicular at refV0 (inward = refTangent direction)
  const offset0 = vec2Dot(refTangent, refV0);
  points = clipSegmentToLine(points, refTangent, offset0);
  if (points.length < 2) return [];

  // Side plane 2: perpendicular at refV1 (inward = -refTangent direction)
  const offset1 = -vec2Dot(refTangent, refV1);
  points = clipSegmentToLine(points, vec2Negate(refTangent), offset1);
  if (points.length < 2) return [];

  // Keep only points behind the reference face
  const refFaceOffset = vec2Dot(refNormal, refV0);
  const result: Array<{ point: Vec2; depth: number }> = [];

  for (const p of points) {
    const sep = vec2Dot(refNormal, p) - refFaceOffset;
    if (sep <= 0) {
      result.push({ point: p, depth: -sep });
    }
  }

  return result;
}

/**
 * Sutherland-Hodgman clip of a two-point segment against a half-plane.
 * The half-plane is defined as { x : dot(normal, x) ≥ offset }.
 */
function clipSegmentToLine(
  points: Vec2[],
  normal: Vec2,
  offset: number,
): Vec2[] {
  const out: Vec2[] = [];
  const d0 = vec2Dot(normal, points[0]) - offset;
  const d1 = vec2Dot(normal, points[1]) - offset;

  if (d0 >= 0) out.push(points[0]);
  if (d1 >= 0) out.push(points[1]);

  // If they're on opposite sides, compute intersection
  if (d0 * d1 < 0) {
    const t = d0 / (d0 - d1);
    out.push(vec2Add(points[0], vec2Scale(vec2Sub(points[1], points[0]), t)));
  }

  return out;
}

/**
 * Find the incident edge: the edge on incBody whose normal is most anti-parallel
 * to the reference normal.
 */
function findIncidentEdge(
  incVerts: Vec2[],
  incNormals: Vec2[],
  refNormal: Vec2,
): [Vec2, Vec2] {
  let minDot = Infinity;
  let bestIdx = 0;
  for (let i = 0; i < incNormals.length; i++) {
    const d = vec2Dot(incNormals[i], refNormal);
    if (d < minDot) {
      minDot = d;
      bestIdx = i;
    }
  }
  return [incVerts[bestIdx], incVerts[(bestIdx + 1) % incVerts.length]];
}

function polygonVsPolygon(a: RigidBody2D, b: RigidBody2D): GeneratedContact[] {
  const vertsA = getWorldVertices(a);
  const normsA = getWorldNormals(a);
  const vertsB = getWorldVertices(b);
  const normsB = getWorldNormals(b);

  if (vertsA.length < 2 || vertsB.length < 2) return [];

  // SAT: test all axes from both polygons
  const satA = findMaxSeparation(vertsA, normsA, vertsB);
  if (satA.separation > 0) return [];

  const satB = findMaxSeparation(vertsB, normsB, vertsA);
  if (satB.separation > 0) return [];

  // Choose reference face (least penetration = most stable)
  let refVerts: Vec2[], refNormals: Vec2[];
  let incVerts: Vec2[], incNormals: Vec2[];
  let refEdgeIdx: number;
  let flip: boolean;

  if (satA.separation >= satB.separation) {
    // A's edge is the reference face
    refVerts = vertsA;
    refNormals = normsA;
    incVerts = vertsB;
    incNormals = normsB;
    refEdgeIdx = satA.edgeIndex;
    flip = false;
  } else {
    // B's edge is the reference face
    refVerts = vertsB;
    refNormals = normsB;
    incVerts = vertsA;
    incNormals = normsA;
    refEdgeIdx = satB.edgeIndex;
    flip = true;
  }

  const refNormal = refNormals[refEdgeIdx];
  const refV0 = refVerts[refEdgeIdx];
  const refV1 = refVerts[(refEdgeIdx + 1) % refVerts.length];

  // Find incident edge
  const [incV0, incV1] = findIncidentEdge(incVerts, incNormals, refNormal);

  // Clip
  const clipped = clipEdges(refV0, refV1, refNormal, incV0, incV1);
  if (clipped.length === 0) return [];

  // Build contacts: normal always points from A toward B
  const normal = flip ? vec2Negate(refNormal) : refNormal;
  const contacts: GeneratedContact[] = [];

  for (const cp of clipped) {
    contacts.push({
      bodyA: a.id,
      bodyB: b.id,
      normal,
      point: cp.point,
      depth: cp.depth,
      normalImpulse: 0,
      tangentImpulse: 0,
    });
  }

  return contacts;
}
