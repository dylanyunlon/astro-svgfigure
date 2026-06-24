// rigid-body.ts
// Collision-subsystem rigid body: lean representation used by the contact
// generator and impulse solver.  Aligned with the ContactConstraint /
// ContactSolver interfaces in the same package.



// ─── Shape primitives ─────────────────────────────────────────────────────────


import {

  Vec2,
  vec2,
  vec2Add,
  vec2Scale,
  vec2CrossSV,
  mat2Rotation,
  mat2MulVec,
  mat2TransposeMulVec,
  wrapAngle,
  type Mat2x2,
} from './math';

export const enum ShapeKind {
  Circle  = 0,
  Polygon = 1,
}

export interface CircleDef {
  kind: ShapeKind.Circle;
  radius: number;
  /** Local-space offset from the body's centre of mass. */
  offset: Vec2;
}

export interface PolygonDef {
  kind: ShapeKind.Polygon;
  /** Vertices in local space, wound CCW. */
  vertices: Vec2[];
  /** Edge normals (outward), one per edge. Pre-computed at creation. */
  normals: Vec2[];
}

export type ShapeDef = CircleDef | PolygonDef;

// ─── Body type flags ──────────────────────────────────────────────────────────

export const enum BodyType {
  /** Zero inverse mass / inertia — never moves. */
  Static    = 0,
  /** Moved by user code; participates in collision but ignores forces. */
  Kinematic = 1,
  /** Fully simulated. */
  Dynamic   = 2,
}

// ─── RigidBody2D ──────────────────────────────────────────────────────────────

export interface RigidBody2D {
  id: number;
  type: BodyType;

  // Pose
  position: Vec2;
  angle: number;

  // Velocity
  linearVelocity: Vec2;
  angularVelocity: number;

  // Mass properties (inverse — 0 means infinite / static)
  invMass: number;
  invInertia: number;

  // Material
  restitution: number;
  friction: number;

  // Shape
  shape: ShapeDef;

  // Accumulated force & torque (reset each step)
  force: Vec2;
  torque: number;

  // User-defined tag (e.g. species index, layer mask)
  userData: number;
}

// ─── Factory helpers ──────────────────────────────────────────────────────────

export interface RigidBody2DOptions {
  id?: number;
  type?: BodyType;
  angle?: number;
  linearVelocity?: Vec2;
  angularVelocity?: number;
  restitution?: number;
  friction?: number;
  userData?: number;
}

let _nextId = 0;

/** Reset the auto-increment ID counter (useful for tests). */
export function resetBodyIdCounter(value = 0): void {
  _nextId = value;
}

/**
 * Create a dynamic circle body.
 * Mass and moment of inertia are derived from `density * π r²`.
 */
export function createCircleBody(
  x: number,
  y: number,
  radius: number,
  density = 1.0,
  opts: RigidBody2DOptions = {},
): RigidBody2D {
  const isStatic = (opts.type ?? BodyType.Dynamic) === BodyType.Static;
  const mass = isStatic ? 0 : density * Math.PI * radius * radius;
  // Moment of inertia for solid disc: I = ½ m r²
  const inertia = isStatic ? 0 : 0.5 * mass * radius * radius;

  return {
    id: opts.id ?? _nextId++,
    type: opts.type ?? BodyType.Dynamic,
    position: vec2(x, y),
    angle: opts.angle ?? 0,
    linearVelocity: opts.linearVelocity ?? vec2(0, 0),
    angularVelocity: opts.angularVelocity ?? 0,
    invMass: mass > 0 ? 1 / mass : 0,
    invInertia: inertia > 0 ? 1 / inertia : 0,
    restitution: opts.restitution ?? 0.3,
    friction: opts.friction ?? 0.4,
    shape: { kind: ShapeKind.Circle, radius, offset: vec2(0, 0) },
    force: vec2(0, 0),
    torque: 0,
    userData: opts.userData ?? 0,
  };
}

/**
 * Create a dynamic box body (axis-aligned in local space).
 * `w` and `h` are full extents (width, height).
 */
export function createBoxBody(
  x: number,
  y: number,
  w: number,
  h: number,
  density = 1.0,
  opts: RigidBody2DOptions = {},
): RigidBody2D {
  const hw = w * 0.5;
  const hh = h * 0.5;
  const vertices: Vec2[] = [
    vec2(-hw, -hh),
    vec2( hw, -hh),
    vec2( hw,  hh),
    vec2(-hw,  hh),
  ];
  const normals = computeOutwardNormals(vertices);

  const isStatic = (opts.type ?? BodyType.Dynamic) === BodyType.Static;
  const mass = isStatic ? 0 : density * w * h;
  // Moment of inertia for rectangle: I = m(w²+h²)/12
  const inertia = isStatic ? 0 : (mass * (w * w + h * h)) / 12;

  return {
    id: opts.id ?? _nextId++,
    type: opts.type ?? BodyType.Dynamic,
    position: vec2(x, y),
    angle: opts.angle ?? 0,
    linearVelocity: opts.linearVelocity ?? vec2(0, 0),
    angularVelocity: opts.angularVelocity ?? 0,
    invMass: mass > 0 ? 1 / mass : 0,
    invInertia: inertia > 0 ? 1 / inertia : 0,
    restitution: opts.restitution ?? 0.3,
    friction: opts.friction ?? 0.4,
    shape: { kind: ShapeKind.Polygon, vertices, normals },
    force: vec2(0, 0),
    torque: 0,
    userData: opts.userData ?? 0,
  };
}

/**
 * Create a body from an arbitrary convex polygon (vertices in CCW winding).
 * Density-based mass uses the polygon area via the shoelace formula.
 */
export function createPolygonBody(
  x: number,
  y: number,
  vertices: Vec2[],
  density = 1.0,
  opts: RigidBody2DOptions = {},
): RigidBody2D {
  const normals = computeOutwardNormals(vertices);
  const isStatic = (opts.type ?? BodyType.Dynamic) === BodyType.Static;
  const area = polygonArea(vertices);
  const mass = isStatic ? 0 : density * area;
  const inertia = isStatic ? 0 : polygonInertia(vertices, mass);

  return {
    id: opts.id ?? _nextId++,
    type: opts.type ?? BodyType.Dynamic,
    position: vec2(x, y),
    angle: opts.angle ?? 0,
    linearVelocity: opts.linearVelocity ?? vec2(0, 0),
    angularVelocity: opts.angularVelocity ?? 0,
    invMass: mass > 0 ? 1 / mass : 0,
    invInertia: inertia > 0 ? 1 / inertia : 0,
    restitution: opts.restitution ?? 0.3,
    friction: opts.friction ?? 0.4,
    shape: { kind: ShapeKind.Polygon, vertices, normals },
    force: vec2(0, 0),
    torque: 0,
    userData: opts.userData ?? 0,
  };
}

// ─── Geometry utilities ───────────────────────────────────────────────────────

/** Compute outward normals for a CCW-wound convex polygon. */
export function computeOutwardNormals(vertices: Vec2[]): Vec2[] {
  const n = vertices.length;
  const normals: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ex = vertices[j].x - vertices[i].x;
    const ey = vertices[j].y - vertices[i].y;
    const len = Math.sqrt(ex * ex + ey * ey);
    // Outward normal for CCW winding: rotate edge 90° CW → (ey, -ex)
    normals.push(len > 1e-12 ? vec2(ey / len, -ex / len) : vec2(0, 0));
  }
  return normals;
}

/** Signed area of a simple polygon via shoelace formula (positive for CCW). */
export function polygonArea(vertices: Vec2[]): number {
  let area = 0;
  const n = vertices.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
  }
  return Math.abs(area) * 0.5;
}

/**
 * Moment of inertia about the centroid for a convex polygon of uniform density.
 * Uses the triangulation method from the polygon's vertices (assumed centroid at origin).
 */
export function polygonInertia(vertices: Vec2[], mass: number): number {
  const n = vertices.length;
  if (n < 3) return 0;

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const a = vertices[i];
    const b = vertices[j];
    const crossAB = Math.abs(a.x * b.y - a.y * b.x);
    numerator +=
      crossAB *
      (a.x * a.x + a.x * b.x + b.x * b.x + a.y * a.y + a.y * b.y + b.y * b.y);
    denominator += crossAB;
  }

  if (denominator < 1e-12) return 0;
  return (mass * numerator) / (6 * denominator);
}

// ─── Transform helpers ────────────────────────────────────────────────────────

/** Transform a local-space point to world space. */
export function bodyToWorld(body: RigidBody2D, localPoint: Vec2): Vec2 {
  const rot = mat2Rotation(body.angle);
  const rotated = mat2MulVec(rot, localPoint);
  return vec2Add(body.position, rotated);
}

/** Transform a world-space point to local space. */
export function worldToBody(body: RigidBody2D, worldPoint: Vec2): Vec2 {
  const rot = mat2Rotation(body.angle);
  const delta: Vec2 = { x: worldPoint.x - body.position.x, y: worldPoint.y - body.position.y };
  return mat2TransposeMulVec(rot, delta);
}

/** Rotate a local-space direction to world space (no translation). */
export function bodyDirToWorld(body: RigidBody2D, localDir: Vec2): Vec2 {
  const rot = mat2Rotation(body.angle);
  return mat2MulVec(rot, localDir);
}

/** Get the world-space vertices of a polygon shape. */
export function getWorldVertices(body: RigidBody2D): Vec2[] {
  const shape = body.shape;
  if (shape.kind !== ShapeKind.Polygon) return [];
  return shape.vertices.map((v) => bodyToWorld(body, v));
}

/** Get the world-space normals of a polygon shape. */
export function getWorldNormals(body: RigidBody2D): Vec2[] {
  const shape = body.shape;
  if (shape.kind !== ShapeKind.Polygon) return [];
  return shape.normals.map((n) => bodyDirToWorld(body, n));
}

// ─── Velocity helpers ─────────────────────────────────────────────────────────

/** Velocity at a world-space point on the body: v + ω × r. */
export function velocityAtPoint(body: RigidBody2D, worldPoint: Vec2): Vec2 {
  const r: Vec2 = { x: worldPoint.x - body.position.x, y: worldPoint.y - body.position.y };
  const wCrossR = vec2CrossSV(body.angularVelocity, r);
  return vec2Add(body.linearVelocity, wCrossR);
}

// ─── Force / impulse application ──────────────────────────────────────────────

/** Apply a force at a world-space point (accumulates into force/torque). */
export function applyForce(body: RigidBody2D, force: Vec2, worldPoint: Vec2): void {
  body.force.x += force.x;
  body.force.y += force.y;
  const rx = worldPoint.x - body.position.x;
  const ry = worldPoint.y - body.position.y;
  body.torque += rx * force.y - ry * force.x;
}

/** Apply an impulse at a world-space point (instant velocity change). */
export function applyImpulse(body: RigidBody2D, impulse: Vec2, worldPoint: Vec2): void {
  body.linearVelocity.x += body.invMass * impulse.x;
  body.linearVelocity.y += body.invMass * impulse.y;
  const rx = worldPoint.x - body.position.x;
  const ry = worldPoint.y - body.position.y;
  body.angularVelocity += body.invInertia * (rx * impulse.y - ry * impulse.x);
}

/** Apply a central impulse (no torque). */
export function applyCentralImpulse(body: RigidBody2D, impulse: Vec2): void {
  body.linearVelocity.x += body.invMass * impulse.x;
  body.linearVelocity.y += body.invMass * impulse.y;
}

// ─── Integration ──────────────────────────────────────────────────────────────

/**
 * Semi-implicit (symplectic) Euler integration.
 * Updates velocity from forces, then position from the new velocity.
 */
export function integrateBody(body: RigidBody2D, dt: number, gravity: Vec2): void {
  if (body.type !== BodyType.Dynamic) return;

  // v += (F/m + g) * dt
  body.linearVelocity.x += (body.force.x * body.invMass + gravity.x) * dt;
  body.linearVelocity.y += (body.force.y * body.invMass + gravity.y) * dt;
  body.angularVelocity += body.torque * body.invInertia * dt;

  // x += v * dt
  body.position.x += body.linearVelocity.x * dt;
  body.position.y += body.linearVelocity.y * dt;
  body.angle = wrapAngle(body.angle + body.angularVelocity * dt);

  // Clear accumulated forces
  body.force.x = 0;
  body.force.y = 0;
  body.torque = 0;
}
