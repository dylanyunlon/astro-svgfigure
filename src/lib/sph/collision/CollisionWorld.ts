// === src/lib/sph/types.ts ===

export interface Vec2 {
  x: number;
  y: number;
}

export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export type ShapeType = 'circle' | 'polygon' | 'box';

export interface ConvexShape {
  type: ShapeType;
  radius?: number;
  halfExtents?: Vec2;
  vertices?: Vec2[];
}

export type BodyType = 'dynamic' | 'static' | 'kinematic';

export interface RigidBody {
  id: number;
  type: BodyType;
  position: Vec2;
  velocity: Vec2;
  angle: number;
  angularVelocity: number;
  mass: number;
  invMass: number;
  inertia: number;
  invInertia: number;
  restitution: number;
  friction: number;
  gravityScale?: number;
}

export interface BVHNode {
  aabb: AABB;
  left: BVHNode | null;
  right: BVHNode | null;
  bodyId: number | null;
  proxyId: number;
}

export interface BroadPhasePair {
  bodyA: number;
  bodyB: number;
}

export interface ContactInfo {
  normal: Vec2;
  depth: number;
  pointA: Vec2;
  pointB: Vec2;
}

export interface ContactConstraint {
  bodyA: RigidBody;
  bodyB: RigidBody;
  contact: ContactInfo;
  accumulatedImpulse: number;
  accumulatedFriction: number;
}

export interface RaycastHit {
  bodyId: number;
  point: Vec2;
  normal: Vec2;
  t: number;
}

export interface OverlapResult {
  bodyIds: number[];
}

export interface SPHParticle {
  position: Vec2;
  velocity: Vec2;
}

export interface SPHWorld {
  obstacles: Array<{ position: Vec2; shape: ConvexShape }>;
  particles: SPHParticle[];
  boundaryBodies: RigidBody[];
}


// === src/lib/sph/collision/C01_Vec2Math.ts ===

// [esbuild-fix] import type { Vec2 } from '../types';

export const vec2 = {
  create(x = 0, y = 0): Vec2 { return { x, y }; },
  clone(v: Vec2): Vec2 { return { x: v.x, y: v.y }; },
  add(a: Vec2, b: Vec2): Vec2 { return { x: a.x + b.x, y: a.y + b.y }; },
  sub(a: Vec2, b: Vec2): Vec2 { return { x: a.x - b.x, y: a.y - b.y }; },
  scale(v: Vec2, s: number): Vec2 { return { x: v.x * s, y: v.y * s }; },
  dot(a: Vec2, b: Vec2): number { return a.x * b.x + a.y * b.y; },
  cross(a: Vec2, b: Vec2): number { return a.x * b.y - a.y * b.x; },
  lengthSq(v: Vec2): number { return v.x * v.x + v.y * v.y; },
  length(v: Vec2): number { return Math.sqrt(vec2.lengthSq(v)); },
  normalize(v: Vec2): Vec2 {
    const l = vec2.length(v);
    return l > 1e-10 ? { x: v.x / l, y: v.y / l } : { x: 0, y: 0 };
  },
  perp(v: Vec2): Vec2 { return { x: -v.y, y: v.x }; },
  neg(v: Vec2): Vec2 { return { x: -v.x, y: -v.y }; },
};


// === src/lib/sph/collision/C02_AABB.ts ===

// [esbuild-fix] import type { AABB, Vec2, ConvexShape, RigidBody } from '../types';

export function aabbFromCircle(center: Vec2, radius: number): AABB {
  return {
    minX: center.x - radius,
    minY: center.y - radius,
    maxX: center.x + radius,
    maxY: center.y + radius,
  };
}

export function aabbFromBox(center: Vec2, halfW: number, halfH: number, angle: number): AABB {
  const cosA = Math.abs(Math.cos(angle));
  const sinA = Math.abs(Math.sin(angle));
  const ex = halfW * cosA + halfH * sinA;
  const ey = halfW * sinA + halfH * cosA;
  return {
    minX: center.x - ex,
    minY: center.y - ey,
    maxX: center.x + ex,
    maxY: center.y + ey,
  };
}

export function computeAABB(body: RigidBody, shape: ConvexShape): AABB {
  if (shape.type === 'circle') {
    return aabbFromCircle(body.position, shape.radius ?? 1);
  }
  if (shape.type === 'box' && shape.halfExtents) {
    return aabbFromBox(body.position, shape.halfExtents.x, shape.halfExtents.y, body.angle);
  }
  if (shape.vertices && shape.vertices.length > 0) {
    const cos = Math.cos(body.angle);
    const sin = Math.sin(body.angle);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const v of shape.vertices) {
      const wx = body.position.x + v.x * cos - v.y * sin;
      const wy = body.position.y + v.x * sin + v.y * cos;
      if (wx < minX) minX = wx;
      if (wy < minY) minY = wy;
      if (wx > maxX) maxX = wx;
      if (wy > maxY) maxY = wy;
    }
    return { minX, minY, maxX, maxY };
  }
  return { minX: body.position.x, minY: body.position.y, maxX: body.position.x, maxY: body.position.y };
}

export function aabbOverlap(a: AABB, b: AABB): boolean {
  return a.maxX >= b.minX && a.minX <= b.maxX &&
         a.maxY >= b.minY && a.minY <= b.maxY;
}

export function aabbContains(outer: AABB, inner: AABB): boolean {
  return inner.minX >= outer.minX && inner.maxX <= outer.maxX &&
         inner.minY >= outer.minY && inner.maxY <= outer.maxY;
}

export function aabbUnion(a: AABB, b: AABB): AABB {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

export function aabbArea(a: AABB): number {
  return (a.maxX - a.minX) * (a.maxY - a.minY);
}


// === src/lib/sph/collision/C03_BVHTree.ts ===

// [esbuild-fix] import type { AABB, BroadPhasePair } from '../types';
// [esbuild-fix] import { aabbOverlap, aabbUnion } from './AABB';

interface BVHNode {
  aabb: AABB;
  left: number;
  right: number;
  parent: number;
  bodyId: number;
  height: number;
}

export class BVHTree {
  private nodes: BVHNode[] = [];
  private root = -1;
  private freeList: number[] = [];

  insertProxy(bodyId: number, aabb: AABB): number {
    const id = this.allocNode();
    const fat = this.fattenAABB(aabb);
    this.nodes[id] = { aabb: fat, left: -1, right: -1, parent: -1, bodyId, height: 0 };
    this.insertLeaf(id);
    return id;
  }

  updateProxy(proxyId: number, aabb: AABB): boolean {
    const node = this.nodes[proxyId];
    if (!node) return false;
    const fat = this.fattenAABB(aabb);
    node.aabb = fat;
    this.removeLeaf(proxyId);
    this.insertLeaf(proxyId);
    return true;
  }

  removeProxy(proxyId: number): void {
    this.removeLeaf(proxyId);
    this.freeList.push(proxyId);
  }

  queryAllPairs(): BroadPhasePair[] {
    const leaves: number[] = [];
    this.collectLeaves(this.root, leaves);
    const pairs: BroadPhasePair[] = [];
    for (let i = 0; i < leaves.length; i++) {
      for (let j = i + 1; j < leaves.length; j++) {
        const a = this.nodes[leaves[i]];
        const b = this.nodes[leaves[j]];
        if (aabbOverlap(a.aabb, b.aabb)) {
          pairs.push({ bodyA: a.bodyId, bodyB: b.bodyId });
        }
      }
    }
    return pairs;
  }

  query(aabb: AABB): number[] {
    const results: number[] = [];
    const stack: number[] = [this.root];
    while (stack.length > 0) {
      const id = stack.pop()!;
      if (id === -1) continue;
      const node = this.nodes[id];
      if (!node || !aabbOverlap(node.aabb, aabb)) continue;
      if (node.bodyId !== -1) {
        results.push(node.bodyId);
      } else {
        stack.push(node.left, node.right);
      }
    }
    return results;
  }

  private allocNode(): number {
    if (this.freeList.length > 0) return this.freeList.pop()!;
    const id = this.nodes.length;
    this.nodes.push({ aabb: { minX: 0, minY: 0, maxX: 0, maxY: 0 }, left: -1, right: -1, parent: -1, bodyId: -1, height: 0 });
    return id;
  }

  private fattenAABB(aabb: AABB, margin = 0.1): AABB {
    return { minX: aabb.minX - margin, minY: aabb.minY - margin, maxX: aabb.maxX + margin, maxY: aabb.maxY + margin };
  }

  private collectLeaves(id: number, out: number[]): void {
    if (id === -1) return;
    const node = this.nodes[id];
    if (!node) return;
    if (node.bodyId !== -1) { out.push(id); return; }
    this.collectLeaves(node.left, out);
    this.collectLeaves(node.right, out);
  }

  private insertLeaf(leaf: number): void {
    if (this.root === -1) { this.root = leaf; this.nodes[leaf].parent = -1; return; }
    let best = this.root;
    while (this.nodes[best].bodyId === -1) {
      const left = this.nodes[best].left;
      const right = this.nodes[best].right;
      if (left === -1) { best = right; break; }
      if (right === -1) { best = left; break; }
      const aL = aabbUnion(this.nodes[left].aabb, this.nodes[leaf].aabb);
      const aR = aabbUnion(this.nodes[right].aabb, this.nodes[leaf].aabb);
      const sL = (aL.maxX - aL.minX) * (aL.maxY - aL.minY);
      const sR = (aR.maxX - aR.minX) * (aR.maxY - aR.minY);
      best = sL < sR ? left : right;
    }
    const oldParent = this.nodes[best].parent;
    const newParent = this.allocNode();
    this.nodes[newParent] = {
      aabb: aabbUnion(this.nodes[best].aabb, this.nodes[leaf].aabb),
      left: best, right: leaf,
      parent: oldParent,
      bodyId: -1,
      height: this.nodes[best].height + 1,
    };
    this.nodes[best].parent = newParent;
    this.nodes[leaf].parent = newParent;
    if (oldParent === -1) {
      this.root = newParent;
    } else {
      if (this.nodes[oldParent].left === best) this.nodes[oldParent].left = newParent;
      else this.nodes[oldParent].right = newParent;
    }
    this.refitAncestors(newParent);
  }

  private removeLeaf(leaf: number): void {
    if (leaf === this.root) { this.root = -1; return; }
    const parent = this.nodes[leaf].parent;
    const grandparent = this.nodes[parent].parent;
    const sibling = this.nodes[parent].left === leaf ? this.nodes[parent].right : this.nodes[parent].left;
    if (grandparent === -1) {
      this.root = sibling;
      if (sibling !== -1) this.nodes[sibling].parent = -1;
    } else {
      if (this.nodes[grandparent].left === parent) this.nodes[grandparent].left = sibling;
      else this.nodes[grandparent].right = sibling;
      if (sibling !== -1) this.nodes[sibling].parent = grandparent;
      this.refitAncestors(grandparent);
    }
    this.freeList.push(parent);
  }

  private refitAncestors(id: number): void {
    let cur = id;
    while (cur !== -1) {
      const node = this.nodes[cur];
      if (node.left !== -1 && node.right !== -1) {
        node.aabb = aabbUnion(this.nodes[node.left].aabb, this.nodes[node.right].aabb);
        node.height = 1 + Math.max(this.nodes[node.left].height, this.nodes[node.right].height);
      }
      cur = node.parent;
    }
  }
}


// === src/lib/sph/collision/C04_SortAndSweep.ts ===

// [esbuild-fix] import type { AABB, BroadPhasePair } from '../types';

interface SAPEntry {
  bodyId: number;
  aabb: AABB;
}

export class SortAndSweep {
  private entries: Map<number, SAPEntry> = new Map();

  insertProxy(bodyId: number, aabb: AABB): void {
    this.entries.set(bodyId, { bodyId, aabb });
  }

  updateProxy(bodyId: number, aabb: AABB): void {
    const e = this.entries.get(bodyId);
    if (e) e.aabb = aabb;
  }

  removeProxy(bodyId: number): void {
    this.entries.delete(bodyId);
  }

  getPairs(): BroadPhasePair[] {
    const list = Array.from(this.entries.values()).sort((a, b) => a.aabb.minX - b.aabb.minX);
    const pairs: BroadPhasePair[] = [];
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        if (list[j].aabb.minX > list[i].aabb.maxX) break;
        if (list[i].aabb.maxY >= list[j].aabb.minY && list[i].aabb.minY <= list[j].aabb.maxY) {
          pairs.push({ bodyA: list[i].bodyId, bodyB: list[j].bodyId });
        }
      }
    }
    return pairs;
  }
}


// === src/lib/sph/collision/C05_NarrowPhase.ts ===

// [esbuild-fix] import type { RigidBody, ConvexShape, ContactInfo, Vec2 } from '../types';
// vec2 utilities inlined below;

function circleVsCircle(
  posA: Vec2, rA: number,
  posB: Vec2, rB: number,
): ContactInfo | null {
  const d = vec2.sub(posB, posA);
  const distSq = vec2.lengthSq(d);
  const radSum = rA + rB;
  if (distSq >= radSum * radSum) return null;
  const dist = Math.sqrt(distSq);
  const normal = dist > 1e-10 ? vec2.scale(d, 1 / dist) : { x: 0, y: 1 };
  return {
    normal,
    depth: radSum - dist,
    pointA: vec2.add(posA, vec2.scale(normal, rA)),
    pointB: vec2.sub(posB, vec2.scale(normal, rB)),
  };
}

function circleVsBox(
  circlePos: Vec2, r: number,
  boxPos: Vec2, halfW: number, halfH: number, boxAngle: number,
): ContactInfo | null {
  const cos = Math.cos(-boxAngle), sin = Math.sin(-boxAngle);
  const dx = circlePos.x - boxPos.x;
  const dy = circlePos.y - boxPos.y;
  const localX = cos * dx - sin * dy;
  const localY = sin * dx + cos * dy;

  const clampX = Math.max(-halfW, Math.min(halfW, localX));
  const clampY = Math.max(-halfH, Math.min(halfH, localY));

  const diffX = localX - clampX;
  const diffY = localY - clampY;
  const distSq = diffX * diffX + diffY * diffY;

  if (distSq >= r * r) return null;

  const dist = Math.sqrt(distSq);
  const nx = dist > 1e-10 ? diffX / dist : 0;
  const ny = dist > 1e-10 ? diffY / dist : 1;

  const cosW = Math.cos(boxAngle), sinW = Math.sin(boxAngle);
  const nwx = cosW * nx - sinW * ny;
  const nwy = sinW * nx + cosW * ny;

  const worldClampX = boxPos.x + (cosW * clampX - sinW * clampY);
  const worldClampY = boxPos.y + (sinW * clampX + cosW * clampY);

  return {
    normal: { x: nwx, y: nwy },
    depth: r - dist,
    pointA: vec2.add(circlePos, vec2.scale({ x: -nwx, y: -nwy }, r)),
    pointB: { x: worldClampX, y: worldClampY },
  };
}

function getBoxAxes(angle: number): Vec2[] {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return [{ x: cos, y: sin }, { x: -sin, y: cos }];
}

function projectBox(center: Vec2, hW: number, hH: number, angle: number, axis: Vec2): [number, number] {
  const axes = getBoxAxes(angle);
  const r = Math.abs(vec2.dot(axes[0], axis)) * hW + Math.abs(vec2.dot(axes[1], axis)) * hH;
  const c = vec2.dot(center, axis);
  return [c - r, c + r];
}

function boxVsBox(
  posA: Vec2, hWA: number, hHA: number, angleA: number,
  posB: Vec2, hWB: number, hHB: number, angleB: number,
): ContactInfo | null {
  const axes = [...getBoxAxes(angleA), ...getBoxAxes(angleB)];
  let minDepth = Infinity;
  let bestAxis: Vec2 = { x: 0, y: 1 };

  for (const axis of axes) {
    const [minA, maxA] = projectBox(posA, hWA, hHA, angleA, axis);
    const [minB, maxB] = projectBox(posB, hWB, hHB, angleB, axis);
    const overlap = Math.min(maxA, maxB) - Math.max(minA, minB);
    if (overlap <= 0) return null;
    if (overlap < minDepth) { minDepth = overlap; bestAxis = axis; }
  }

  const d = vec2.sub(posB, posA);
  if (vec2.dot(d, bestAxis) < 0) bestAxis = vec2.neg(bestAxis);

  return {
    normal: bestAxis,
    depth: minDepth,
    pointA: vec2.add(posA, vec2.scale(bestAxis, Math.min(hWA, hHA))),
    pointB: vec2.sub(posB, vec2.scale(bestAxis, Math.min(hWB, hHB))),
  };
}

export function computeContactInfo(
  bodyA: RigidBody, shapeA: ConvexShape,
  bodyB: RigidBody, shapeB: ConvexShape,
): ContactInfo | null {
  const tA = shapeA.type, tB = shapeB.type;

  if (tA === 'circle' && tB === 'circle') {
    return circleVsCircle(bodyA.position, shapeA.radius!, bodyB.position, shapeB.radius!);
  }

  if (tA === 'circle' && tB === 'box') {
    const hw = shapeB.halfExtents!.x, hh = shapeB.halfExtents!.y;
    const c = circleVsBox(bodyA.position, shapeA.radius!, bodyB.position, hw, hh, bodyB.angle);
    if (!c) return null;
    return { ...c, normal: vec2.neg(c.normal) };
  }

  if (tA === 'box' && tB === 'circle') {
    const hw = shapeA.halfExtents!.x, hh = shapeA.halfExtents!.y;
    return circleVsBox(bodyB.position, shapeB.radius!, bodyA.position, hw, hh, bodyA.angle);
  }

  if (tA === 'box' && tB === 'box') {
    return boxVsBox(
      bodyA.position, shapeA.halfExtents!.x, shapeA.halfExtents!.y, bodyA.angle,
      bodyB.position, shapeB.halfExtents!.x, shapeB.halfExtents!.y, bodyB.angle,
    );
  }

  return null;
}


// === src/lib/sph/collision/C06_ContactSolver.ts ===

// [esbuild-fix] import type { ContactConstraint } from '../types';
// vec2 utilities inlined below;

export class ContactSolver {
  private iterations: number;

  constructor(iterations = 10) {
    this.iterations = iterations;
  }

  solve(constraints: ContactConstraint[]): void {
    for (let iter = 0; iter < this.iterations; iter++) {
      for (const c of constraints) {
        this.resolveContact(c);
      }
    }
  }

  private resolveContact(c: ContactConstraint): void {
    const { bodyA, bodyB, contact } = c;
    const { normal } = contact;

    const relVel = vec2.sub(bodyB.velocity, bodyA.velocity);
    const vn = vec2.dot(relVel, normal);

    if (vn > 0) return;

    const e = Math.min(bodyA.restitution, bodyB.restitution);
    const invMassSum = bodyA.invMass + bodyB.invMass;
    if (invMassSum === 0) return;

    const j = -(1 + e) * vn / invMassSum;
    const impulse = vec2.scale(normal, j);

    bodyA.velocity = vec2.sub(bodyA.velocity, vec2.scale(impulse, bodyA.invMass));
    bodyB.velocity = vec2.add(bodyB.velocity, vec2.scale(impulse, bodyB.invMass));

    const relVelAfter = vec2.sub(bodyB.velocity, bodyA.velocity);
    const tangent = vec2.normalize(vec2.sub(relVelAfter, vec2.scale(normal, vec2.dot(relVelAfter, normal))));
    const vt = vec2.dot(relVelAfter, tangent);
    const mu = (bodyA.friction + bodyB.friction) * 0.5;
    const jt = Math.max(-mu * Math.abs(j), Math.min(mu * Math.abs(j), -vt / invMassSum));
    const frictionImpulse = vec2.scale(tangent, jt);

    bodyA.velocity = vec2.sub(bodyA.velocity, vec2.scale(frictionImpulse, bodyA.invMass));
    bodyB.velocity = vec2.add(bodyB.velocity, vec2.scale(frictionImpulse, bodyB.invMass));
  }
}


// === src/lib/sph/collision/C07_PositionSolver.ts ===

// [esbuild-fix] import type { ContactConstraint } from '../types';
// vec2 utilities inlined below;

const SLOP = 0.01;
const BAUMGARTE = 0.4;

export class PositionSolver {
  private iterations: number;

  constructor(iterations = 3) {
    this.iterations = iterations;
  }

  solve(constraints: ContactConstraint[]): void {
    for (let iter = 0; iter < this.iterations; iter++) {
      for (const c of constraints) {
        this.correctPosition(c);
      }
    }
  }

  private correctPosition(c: ContactConstraint): void {
    const { bodyA, bodyB, contact } = c;
    const { normal, depth } = contact;

    const invMassSum = bodyA.invMass + bodyB.invMass;
    if (invMassSum === 0) return;

    const correction = Math.max(depth - SLOP, 0) * BAUMGARTE / invMassSum;
    const corrVec = vec2.scale(normal, correction);

    if (bodyA.invMass > 0) {
      bodyA.position = vec2.sub(bodyA.position, vec2.scale(corrVec, bodyA.invMass));
    }
    if (bodyB.invMass > 0) {
      bodyB.position = vec2.add(bodyB.position, vec2.scale(corrVec, bodyB.invMass));
    }
  }
}


// === src/lib/sph/collision/C08_SceneQuery.ts ===

// [esbuild-fix] import type { RigidBody, ConvexShape, RaycastHit, OverlapResult, Vec2, AABB } from '../types';
// vec2 utilities inlined below;
// [esbuild-fix] import { aabbOverlap, aabbFromCircle, aabbExpand, AABB } from './AABB';

export class SceneQuery {
  private bodies: RigidBody[] = [];
  private shapes: ConvexShape[] = [];

  setScene(bodies: RigidBody[], shapes: ConvexShape[]): void {
    this.bodies = bodies;
    this.shapes = shapes;
  }

  raycast(origin: Vec2, direction: Vec2, maxDist: number): RaycastHit | null {
    const dir = vec2.normalize(direction);
    let best: RaycastHit | null = null;

    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      const shape = this.shapes[i];
      const hit = this.raycastShape(origin, dir, maxDist, body, shape);
      if (hit && (!best || hit.t < best.t)) best = hit;
    }
    return best;
  }

  overlapAABB(aabb: AABB): OverlapResult {
    const bodyIds: number[] = [];
    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      const shape = this.shapes[i];
      const shapeAABB = computeAABB(body, shape);
      if (aabbOverlap(aabb, shapeAABB)) bodyIds.push(body.id);
    }
    return { bodyIds };
  }

  overlapCircle(center: Vec2, radius: number): OverlapResult {
    const bodyIds: number[] = [];
    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      const shape = this.shapes[i];
      if (this.circleOverlapsShape(center, radius, body, shape)) {
        bodyIds.push(body.id);
      }
    }
    return { bodyIds };
  }

  private raycastShape(
    origin: Vec2, dir: Vec2, maxDist: number,
    body: RigidBody, shape: ConvexShape,
  ): RaycastHit | null {
    if (shape.type === 'circle') {
      return this.raycastCircle(origin, dir, maxDist, body, shape.radius!);
    }
    if (shape.type === 'box' && shape.halfExtents) {
      return this.raycastAABBShape(origin, dir, maxDist, body, shape);
    }
    return null;
  }

  private raycastCircle(
    origin: Vec2, dir: Vec2, maxDist: number,
    body: RigidBody, r: number,
  ): RaycastHit | null {
    const oc = vec2.sub(origin, body.position);
    const b = 2 * vec2.dot(oc, dir);
    const c = vec2.lengthSq(oc) - r * r;
    const disc = b * b - 4 * c;
    if (disc < 0) return null;
    const t = (-b - Math.sqrt(disc)) / 2;
    if (t < 0 || t > maxDist) return null;
    const point = vec2.add(origin, vec2.scale(dir, t));
    const normal = vec2.normalize(vec2.sub(point, body.position));
    return { bodyId: body.id, point, normal, t };
  }

  private raycastAABBShape(
    origin: Vec2, dir: Vec2, maxDist: number,
    body: RigidBody, shape: ConvexShape,
  ): RaycastHit | null {
    const hw = shape.halfExtents!.x, hh = shape.halfExtents!.y;
    const minX = body.position.x - hw, maxX = body.position.x + hw;
    const minY = body.position.y - hh, maxY = body.position.y + hh;
    let tmin = 0, tmax = maxDist;
    const eps = 1e-10;

    for (let axis = 0; axis < 2; axis++) {
      const o = axis === 0 ? origin.x : origin.y;
      const d = axis === 0 ? dir.x : dir.y;
      const mn = axis === 0 ? minX : minY;
      const mx = axis === 0 ? maxX : maxY;
      if (Math.abs(d) < eps) {
        if (o < mn || o > mx) return null;
      } else {
        let t1 = (mn - o) / d, t2 = (mx - o) / d;
        if (t1 > t2) [t1, t2] = [t2, t1];
        tmin = Math.max(tmin, t1);
        tmax = Math.min(tmax, t2);
        if (tmin > tmax) return null;
      }
    }
    const point = vec2.add(origin, vec2.scale(dir, tmin));
    const dx = point.x - body.position.x, dy = point.y - body.position.y;
    const absDX = Math.abs(dx) / hw, absDY = Math.abs(dy) / hh;
    const normal: Vec2 = absDX > absDY ? { x: Math.sign(dx), y: 0 } : { x: 0, y: Math.sign(dy) };
    return { bodyId: body.id, point, normal, t: tmin };
  }

  private circleOverlapsShape(center: Vec2, radius: number, body: RigidBody, shape: ConvexShape): boolean {
    if (shape.type === 'circle') {
      return vec2.lengthSq(vec2.sub(center, body.position)) < (radius + shape.radius!) ** 2;
    }
    if (shape.type === 'box' && shape.halfExtents) {
      const hw = shape.halfExtents.x, hh = shape.halfExtents.y;
      const dx = Math.abs(center.x - body.position.x);
      const dy = Math.abs(center.y - body.position.y);
      if (dx > hw + radius || dy > hh + radius) return false;
      if (dx <= hw || dy <= hh) return true;
      const cornerDist = (dx - hw) ** 2 + (dy - hh) ** 2;
      return cornerDist <= radius * radius;
    }
    return false;
  }
}


// === src/lib/sph/collision/C09_RigidBodyFactory.ts ===

// [esbuild-fix] import type { RigidBody, ConvexShape, BodyType } from '../types';

let _nextId = 1;

export function createCircleBody(
  x: number, y: number, radius: number,
  type: BodyType = 'dynamic',
  opts: Partial<Pick<RigidBody, 'restitution' | 'friction' | 'mass' | 'gravityScale'>> = {},
): { body: RigidBody; shape: ConvexShape } {
  const mass = opts.mass ?? (type === 'static' ? Infinity : Math.PI * radius * radius * 1.0);
  const invMass = isFinite(mass) ? 1 / mass : 0;
  const inertia = isFinite(mass) ? 0.5 * mass * radius * radius : Infinity;
  const invInertia = isFinite(inertia) ? 1 / inertia : 0;

  const body: RigidBody = {
    id: _nextId++,
    type,
    position: { x, y },
    velocity: { x: 0, y: 0 },
    angle: 0,
    angularVelocity: 0,
    mass,
    invMass,
    inertia,
    invInertia,
    restitution: opts.restitution ?? 0.6,
    friction: opts.friction ?? 0.3,
    gravityScale: opts.gravityScale ?? 1,
  };
  const shape: ConvexShape = { type: 'circle', radius };
  return { body, shape };
}

export function createBoxBody(
  x: number, y: number, halfW: number, halfH: number,
  type: BodyType = 'dynamic',
  opts: Partial<Pick<RigidBody, 'restitution' | 'friction' | 'mass' | 'gravityScale'>> = {},
): { body: RigidBody; shape: ConvexShape } {
  const mass = opts.mass ?? (type === 'static' ? Infinity : (halfW * 2) * (halfH * 2) * 1.0);
  const invMass = isFinite(mass) ? 1 / mass : 0;
  const inertia = isFinite(mass) ? (mass / 12) * ((halfW * 2) ** 2 + (halfH * 2) ** 2) : Infinity;
  const invInertia = isFinite(inertia) ? 1 / inertia : 0;

  const body: RigidBody = {
    id: _nextId++,
    type,
    position: { x, y },
    velocity: { x: 0, y: 0 },
    angle: 0,
    angularVelocity: 0,
    mass,
    invMass,
    inertia,
    invInertia,
    restitution: opts.restitution ?? 0.3,
    friction: opts.friction ?? 0.5,
    gravityScale: opts.gravityScale ?? 1,
  };
  const shape: ConvexShape = { type: 'box', halfExtents: { x: halfW, y: halfH } };
  return { body, shape };
}

export function resetIdCounter(): void { _nextId = 1; }


// === src/lib/sph/collision/CollisionWorld.ts ===

import type {
  RigidBody,
  ConvexShape,
  ContactConstraint,
  RaycastHit,
  OverlapResult,
  Vec2,
  AABB,
  SPHWorld,
  BroadPhasePair,
} from '../types';

// vec2 utilities inlined below;
// [esbuild-fix] import { computeAABB } from './AABB';
// [esbuild-fix] import { BVHTree } from './BVHTree';
// [esbuild-fix] import { SortAndSweep } from './SortAndSweep';
// [esbuild-fix] import { ContactSolver } from './ContactSolver';
// [esbuild-fix] import { PositionSolver } from './PositionSolver';
// [esbuild-fix] import { SceneQuery } from './SceneQuery';
import {
  CollisionEventDispatcher,
  type ActiveContactPair,
  type CollisionCallback,
  type CollisionEvent,
} from './CollisionEvents';
// computeContactInfo, createCircleBody, createBoxBody inlined above;

export interface CollisionWorldConfig {
  /** Which broad-phase algorithm to use (default: 'bvh') */
  broadPhase?: 'bvh' | 'sap';
  /** Velocity solver iterations (default: 10) */
  solverIterations?: number;
  /** Position correction iterations (default: 3) */
  positionIterations?: number;
  /** Gravity vector (default: { x: 0, y: -9.8 }) */
  gravity?: Vec2;
}

export class CollisionWorld {
  // ── Broad-phase ────────────────────────────────────────────────────────
  private bvh: BVHTree;
  private sap: SortAndSweep;
  private broadPhaseMode: 'bvh' | 'sap';

  // ── Solvers ────────────────────────────────────────────────────────────
  private contactSolver: ContactSolver;
  private positionSolver: PositionSolver;

  // ── Scene query ────────────────────────────────────────────────────────
  private sceneQuery: SceneQuery;

  // ── World state ────────────────────────────────────────────────────────
  private bodies: RigidBody[] = [];
  private shapes: ConvexShape[] = [];
  /** body.id → index in bodies[] */
  private bodyIndex: Map<number, number> = new Map();
  /** body.id → BVH/SAP proxy id */
  private proxyIds: Map<number, number> = new Map();
  /** Active contact constraints, rebuilt each step */
  private contacts: ContactConstraint[] = [];

  // ── Physics config ─────────────────────────────────────────────────────
  private gravity: Vec2;

  // ── Collision event system ─────────────────────────────────────────
  /** Dispatches onCollisionEnter / Stay / Exit callbacks each step. */
  readonly events: CollisionEventDispatcher = new CollisionEventDispatcher();
  /** Monotone simulation time accumulator (seconds). */
  private _simTime = 0;

  constructor(config: CollisionWorldConfig = {}) {
    this.broadPhaseMode = config.broadPhase ?? 'bvh';
    this.gravity = config.gravity ?? { x: 0, y: -9.8 };

    this.bvh = new BVHTree();
    this.sap = new SortAndSweep();
    this.contactSolver = new ContactSolver(config.solverIterations ?? 10);
    this.positionSolver = new PositionSolver(config.positionIterations ?? 3);
    this.sceneQuery = new SceneQuery();
  }

  // ════════════════════════════════════════════════════════════════════════
  //  Body management
  // ════════════════════════════════════════════════════════════════════════

  /** Register a body + shape. Returns body.id. */
  addBody(body: RigidBody, shape: ConvexShape): number {
    const idx = this.bodies.length;
    this.bodies.push(body);
    this.shapes.push(shape);
    this.bodyIndex.set(body.id, idx);

    const aabb = computeAABB(body, shape);
    if (this.broadPhaseMode === 'bvh') {
      const proxyId = this.bvh.insertProxy(body.id, aabb);
      this.proxyIds.set(body.id, proxyId);
    } else {
      this.sap.insertProxy(body.id, aabb);
      this.proxyIds.set(body.id, body.id);
    }

    this.sceneQuery.setScene(this.bodies, this.shapes);
    return body.id;
  }

  /** Remove a body by id. */
  removeBody(id: number): void {
    const idx = this.bodyIndex.get(id);
    if (idx === undefined) return;

    const proxyId = this.proxyIds.get(id)!;
    if (this.broadPhaseMode === 'bvh') {
      this.bvh.removeProxy(proxyId);
    } else {
      this.sap.removeProxy(id);
    }
    this.proxyIds.delete(id);

    // Swap-remove O(1)
    const last = this.bodies.length - 1;
    if (idx !== last) {
      const lastBody = this.bodies[last];
      this.bodies[idx] = lastBody;
      this.shapes[idx] = this.shapes[last];
      this.bodyIndex.set(lastBody.id, idx);
    }
    this.bodies.pop();
    this.shapes.pop();
    this.bodyIndex.delete(id);

    this.sceneQuery.setScene(this.bodies, this.shapes);
    // Evict from event cache so removed body does not generate spurious Exit events
    this.events.evictBody(id);
  }

  /** Retrieve a body by id. Throws if not found. */
  getBody(id: number): RigidBody {
    const idx = this.bodyIndex.get(id);
    if (idx === undefined) throw new Error(`CollisionWorld: body ${id} not found`);
    return this.bodies[idx];
  }

  // ════════════════════════════════════════════════════════════════════════
  //  Main simulation step
  // ════════════════════════════════════════════════════════════════════════

  /**
   * Advance the simulation by dt seconds.
   *
   * Pipeline:
   *  1. Update all AABBs  (body.position + shape → AABB)
   *  2. Broad phase       → BroadPhasePair[]
   *  3. Narrow phase      → ContactConstraint[] (SAT / circle math)
   *  4. Contact Solver    → velocity impulse iterations
   *  5. Position Solver   → Baumgarte penetration correction
   *  6. Integration       → v += dt·g;  x += dt·v;  angle += dt·ω
   *  7. Update proxies    → broad-phase re-sync post-integration
   */
  step(dt: number): void {
    const n = this.bodies.length;

    // ── 1. Update AABBs ────────────────────────────────────────────────
    for (let i = 0; i < n; i++) {
      const body = this.bodies[i];
      const aabb = computeAABB(body, this.shapes[i]);
      const proxyId = this.proxyIds.get(body.id)!;
      if (this.broadPhaseMode === 'bvh') {
        this.bvh.updateProxy(proxyId, aabb);
      } else {
        this.sap.updateProxy(body.id, aabb);
      }
    }

    // ── 2. Broad phase ─────────────────────────────────────────────────
    const pairs: BroadPhasePair[] =
      this.broadPhaseMode === 'bvh'
        ? this.bvh.queryAllPairs()
        : this.sap.getPairs();

    // ── 3. Narrow phase ────────────────────────────────────────────────
    this.contacts = [];
    for (const pair of pairs) {
      const idxA = this.bodyIndex.get(pair.bodyA);
      const idxB = this.bodyIndex.get(pair.bodyB);
      if (idxA === undefined || idxB === undefined) continue;

      const bodyA = this.bodies[idxA];
      const bodyB = this.bodies[idxB];

      // Skip static–static
      if (bodyA.invMass === 0 && bodyB.invMass === 0) continue;

      const contact = computeContactInfo(bodyA, this.shapes[idxA], bodyB, this.shapes[idxB]);
      if (!contact || contact.depth <= 0) continue;

      this.contacts.push({
        bodyA,
        bodyB,
        contact,
        accumulatedImpulse: 0,
        accumulatedFriction: 0,
      });
    }

    // ── 4. Collision events (Enter / Stay / Exit) ────────────────────
    this._simTime += dt;
    {
      const activePairs: ActiveContactPair[] = this.contacts.map((c) => ({
        bodyA: c.bodyA.id,
        bodyB: c.bodyB.id,
        contact: {
          normal: { x: c.contact.normal.x, y: c.contact.normal.y },
          depth:  c.contact.depth,
          pointA: { x: c.contact.pointA.x, y: c.contact.pointA.y },
          pointB: { x: c.contact.pointB.x, y: c.contact.pointB.y },
        },
      }));
      this.events.update(activePairs, this._simTime);
    }

        // ── 4. Contact Solver (velocity) ───────────────────────────────────
    this.contactSolver.solve(this.contacts);

    // ── 5. Position Solver (penetration correction) ────────────────────
    this.positionSolver.solve(this.contacts);

    // ── 6. Integration ─────────────────────────────────────────────────
    for (let i = 0; i < n; i++) {
      const body = this.bodies[i];
      if (body.invMass === 0) continue;

      const gs = body.gravityScale ?? 1;

      body.velocity.x += dt * this.gravity.x * gs;
      body.velocity.y += dt * this.gravity.y * gs;

      body.position.x += dt * body.velocity.x;
      body.position.y += dt * body.velocity.y;

      body.angle += dt * body.angularVelocity;
    }

    // ── 7. Update proxies post-integration ────────────────────────────
    for (let i = 0; i < n; i++) {
      const body = this.bodies[i];
      const aabb = computeAABB(body, this.shapes[i]);
      const proxyId = this.proxyIds.get(body.id)!;
      if (this.broadPhaseMode === 'bvh') {
        this.bvh.updateProxy(proxyId, aabb);
      } else {
        this.sap.updateProxy(body.id, aabb);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  //  Scene Query pass-throughs
  // ════════════════════════════════════════════════════════════════════════

  raycast(origin: Vec2, direction: Vec2, maxDist = 1000): RaycastHit | null {
    return this.sceneQuery.raycast(origin, direction, maxDist);
  }

  overlapAABB(aabb: AABB): OverlapResult {
    return this.sceneQuery.overlapAABB(aabb);
  }

  overlapCircle(center: Vec2, radius: number): OverlapResult {
    return this.sceneQuery.overlapCircle(center, radius);
  }

  // ════════════════════════════════════════════════════════════════════════
  //  SPH Bridge  (Akinci 2012 bidirectional coupling)
  // ════════════════════════════════════════════════════════════════════════

  /**
   * Sync SPH obstacles → static bodies.
   * New obstacles are registered; existing ones (matched by position) are skipped.
   */
  syncFromSPH(sphWorld: SPHWorld): void {
    for (const obs of sphWorld.obstacles) {
      const alreadyPresent = this.bodies.some(
        (b) => b.type === 'static' &&
               Math.abs(b.position.x - obs.position.x) < 1e-6 &&
               Math.abs(b.position.y - obs.position.y) < 1e-6,
      );
      if (alreadyPresent) continue;

      let body: RigidBody;
      let shape: ConvexShape;

      if (obs.shape.type === 'circle') {
        ({ body, shape } = createCircleBody(
          obs.position.x, obs.position.y,
          obs.shape.radius ?? 1,
          'static',
        ));
      } else {
        const hw = obs.shape.halfExtents?.x ?? 1;
        const hh = obs.shape.halfExtents?.y ?? 1;
        ({ body, shape } = createBoxBody(obs.position.x, obs.position.y, hw, hh, 'static'));
      }
      this.addBody(body, shape);
    }
  }

  /**
   * Write simulation results back into SPH boundary bodies so the fluid
   * solver sees up-to-date boundary positions, velocities, and orientations.
   *
   * Implements the Rigid → Fluid leg of Akinci 2012 bidirectional coupling:
   * after the rigid-body integration step the updated pose is pushed into the
   * SPH boundary representation so that:
   *  • `position` / `angle` reflect the new body pose for boundary particle
   *    re-sampling (called by `refreshBoundaryState` in fluid-rigid-coupling.ts).
   *  • `velocity` / `angularVelocity` are available to compute the boundary
   *    particle velocity  v_b = v_cm + ω × r_b  (Section 3.1 of Akinci 2012)
   *    inside `computeCouplingForces`, enabling proper viscous drag.
   */
  applyToSPH(sphWorld: SPHWorld): void {
    for (const sphBody of sphWorld.boundaryBodies) {
      const idx = this.bodyIndex.get(sphBody.id);
      if (idx === undefined) continue;
      const simBody = this.bodies[idx];
      // Pose
      sphBody.position.x       = simBody.position.x;
      sphBody.position.y       = simBody.position.y;
      sphBody.angle            = simBody.angle;
      // Kinematics — used by boundary-velocity formula in coupling forces
      sphBody.velocity.x       = simBody.velocity.x;
      sphBody.velocity.y       = simBody.velocity.y;
      sphBody.angularVelocity  = simBody.angularVelocity;
    }
  }

  /**
   * Apply fluid-pressure forces accumulated in `fluidForces` to the matching
   * dynamic rigid bodies registered in this CollisionWorld, implementing the
   * Fluid → Rigid leg of Akinci 2012 bidirectional coupling.
   *
   * `fluidForces` maps `body.id` to  { fx, fy, torque }  as computed by
   * `computeCouplingForces` in fluid-rigid-coupling.ts.  Forces are
   * converted to velocity impulses via  Δv = F·dt/m,  Δω = τ·dt/I.
   *
   * Only dynamic bodies are affected; static and kinematic bodies are skipped.
   *
   * @param fluidForces  Per-body force/torque table from the SPH solver.
   * @param dt           Timestep used to convert force → impulse.
   */
  applyFluidForces(
    fluidForces: Map<number, { fx: number; fy: number; torque: number }>,
    dt: number,
  ): void {
    for (const [bodyId, force] of fluidForces) {
      const idx = this.bodyIndex.get(bodyId);
      if (idx === undefined) continue;
      const body = this.bodies[idx];
      if (body.type !== 'dynamic' || body.invMass === 0) continue;

      // Δv = F * dt / m  (invMass = 1/m)
      body.velocity.x       += force.fx     * dt * body.invMass;
      body.velocity.y       += force.fy     * dt * body.invMass;
      // Δω = τ * dt / I  (invInertia = 1/I)
      body.angularVelocity  += force.torque * dt * body.invInertia;
    }
  }

  /**
   * Convenience method that runs one full Akinci 2012 bidirectional coupling
   * step in the context of an SPHWorld:
   *
   *  1. `syncFromSPH`    — register any new obstacles as static bodies.
   *  2. `step(dt)`       — advance the rigid-body simulation (gravity,
   *                        contacts, integration).
   *  3. `applyToSPH`     — push updated pose / kinematics to SPH boundaries
   *                        so `refreshBoundaryState` sees the new positions.
   *
   * Caller is responsible for:
   *  • calling `stepFluidRigidCoupling` (fluid-rigid-coupling.ts) before this
   *    to accumulate `fluidForces`, and
   *  • passing those forces in so they are applied before integration (step 2).
   *
   * @param sphWorld     The SPH world describing obstacles and boundary bodies.
   * @param dt           Timestep in seconds.
   * @param fluidForces  Optional pre-computed fluid→rigid force table.
   */
  stepWithSPH(
    sphWorld: SPHWorld,
    dt: number,
    fluidForces?: Map<number, { fx: number; fy: number; torque: number }>,
  ): void {
    // 1. Register any new SPH obstacles as static rigid bodies
    this.syncFromSPH(sphWorld);

    // 2a. Apply accumulated fluid pressure/viscosity forces before integration
    if (fluidForces && fluidForces.size > 0) {
      this.applyFluidForces(fluidForces, dt);
    }

    // 2b. Advance rigid-body physics (broad/narrow phase, solver, integration)
    this.step(dt);

    // 3. Push updated pose + kinematics back into SPH boundary body proxies
    this.applyToSPH(sphWorld);
  }

  // ════════════════════════════════════════════════════════════════════════
  //  Debug helpers
  // ════════════════════════════════════════════════════════════════════════

  get bodyCount(): number { return this.bodies.length; }
  get contactCount(): number { return this.contacts.length; }

  getPositions(): Array<{ id: number; x: number; y: number }> {
    return this.bodies.map((b) => ({ id: b.id, x: b.position.x, y: b.position.y }));
  }

  exportCollisions(): {
    collisions: Array<{ bodyA: number; bodyB: number; normal: { x: number; y: number }; depth: number }>;
    count: number;
  } {
    const collisions = this.contacts.map((c) => ({
      bodyA: c.bodyA.id,
      bodyB: c.bodyB.id,
      normal: { x: c.contact.normal.x, y: c.contact.normal.y },
      depth: c.contact.depth,
    }));
    return { collisions, count: collisions.length };
  }
}
