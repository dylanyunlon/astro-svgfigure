/**
 * Box3 — Axis-Aligned Bounding Box (AABB) in 3-D
 * AT gap-fill: math/utility (#75 xiaodi)
 */
import { Vec3 } from './Vec3';

export class Box3 {
  /** min corner (all +Infinity when empty). */
  min: Vec3;
  /** max corner (all -Infinity when empty). */
  max: Vec3;

  constructor(
    min = new Vec3(+Infinity, +Infinity, +Infinity),
    max = new Vec3(-Infinity, -Infinity, -Infinity),
  ) {
    this.min = min.clone();
    this.max = max.clone();
  }

  // ── factory ──────────────────────────────────────────────────────────────
  static empty(): Box3 { return new Box3(); }

  static fromPoints(points: Vec3[]): Box3 {
    const b = Box3.empty();
    for (const p of points) b.expandByPoint(p);
    return b;
  }

  static fromCenterSize(center: Vec3, size: Vec3): Box3 {
    const half = size.mul(0.5);
    return new Box3(center.sub(half), center.add(half));
  }

  // ── state ──────────────────────────────────────────────────────────────────
  isEmpty(): boolean {
    return this.min.x > this.max.x || this.min.y > this.max.y || this.min.z > this.max.z;
  }

  clone(): Box3 { return new Box3(this.min, this.max); }
  copy(b: Box3): this { this.min.copy(b.min); this.max.copy(b.max); return this; }

  makeEmpty(): this {
    this.min.set(+Infinity, +Infinity, +Infinity);
    this.max.set(-Infinity, -Infinity, -Infinity);
    return this;
  }

  // ── geometry ──────────────────────────────────────────────────────────────
  center(): Vec3 { return this.min.add(this.max).mul(0.5); }
  size(): Vec3   { return this.max.sub(this.min); }
  /** Half-extents (size/2). */
  extents(): Vec3 { return this.size().mul(0.5); }

  // ── expand ────────────────────────────────────────────────────────────────
  expandByPoint(p: Vec3): this {
    if (p.x < this.min.x) this.min.x = p.x;
    if (p.y < this.min.y) this.min.y = p.y;
    if (p.z < this.min.z) this.min.z = p.z;
    if (p.x > this.max.x) this.max.x = p.x;
    if (p.y > this.max.y) this.max.y = p.y;
    if (p.z > this.max.z) this.max.z = p.z;
    return this;
  }

  expandByBox(b: Box3): this {
    this.expandByPoint(b.min);
    this.expandByPoint(b.max);
    return this;
  }

  expandByScalar(s: number): this {
    this.min.x -= s; this.min.y -= s; this.min.z -= s;
    this.max.x += s; this.max.y += s; this.max.z += s;
    return this;
  }

  /** Expand so that this becomes the union of this and b (returns new Box3). */
  union(b: Box3): Box3 { return this.clone().expandByBox(b); }

  // ── contains ──────────────────────────────────────────────────────────────
  containsPoint(p: Vec3): boolean {
    return (
      p.x >= this.min.x && p.x <= this.max.x &&
      p.y >= this.min.y && p.y <= this.max.y &&
      p.z >= this.min.z && p.z <= this.max.z
    );
  }

  containsBox(b: Box3): boolean {
    return (
      this.min.x <= b.min.x && b.max.x <= this.max.x &&
      this.min.y <= b.min.y && b.max.y <= this.max.y &&
      this.min.z <= b.min.z && b.max.z <= this.max.z
    );
  }

  // ── intersects ────────────────────────────────────────────────────────────
  intersectsBox(b: Box3): boolean {
    return !(
      b.max.x < this.min.x || b.min.x > this.max.x ||
      b.max.y < this.min.y || b.min.y > this.max.y ||
      b.max.z < this.min.z || b.min.z > this.max.z
    );
  }

  intersectsSphere(center: Vec3, radius: number): boolean {
    return this.distanceSqToPoint(center) <= radius * radius;
  }

  /** Intersection of this and b. Returns empty Box3 if no overlap. */
  intersection(b: Box3): Box3 {
    const minX = Math.max(this.min.x, b.min.x);
    const minY = Math.max(this.min.y, b.min.y);
    const minZ = Math.max(this.min.z, b.min.z);
    const maxX = Math.min(this.max.x, b.max.x);
    const maxY = Math.min(this.max.y, b.max.y);
    const maxZ = Math.min(this.max.z, b.max.z);
    if (minX > maxX || minY > maxY || minZ > maxZ) return Box3.empty();
    return new Box3(
      new Vec3(minX, minY, minZ),
      new Vec3(maxX, maxY, maxZ),
    );
  }

  // ── distance ──────────────────────────────────────────────────────────────
  /** Squared distance from point to the nearest point on/in the box. */
  distanceSqToPoint(p: Vec3): number {
    const dx = Math.max(this.min.x - p.x, 0, p.x - this.max.x);
    const dy = Math.max(this.min.y - p.y, 0, p.y - this.max.y);
    const dz = Math.max(this.min.z - p.z, 0, p.z - this.max.z);
    return dx*dx + dy*dy + dz*dz;
  }
  distanceToPoint(p: Vec3): number { return Math.sqrt(this.distanceSqToPoint(p)); }

  /** Clamp a point to the interior of the box. */
  clampPoint(p: Vec3): Vec3 {
    return new Vec3(
      Math.max(this.min.x, Math.min(this.max.x, p.x)),
      Math.max(this.min.y, Math.min(this.max.y, p.y)),
      Math.max(this.min.z, Math.min(this.max.z, p.z)),
    );
  }

  // ── transform ─────────────────────────────────────────────────────────────
  /** Returns AABB translated by offset. */
  translate(offset: Vec3): Box3 {
    return new Box3(this.min.add(offset), this.max.add(offset));
  }

  /** Returns AABB after applying a Mat4 (full matrix transform of all 8 corners). */
  applyMatrix4(m: { transformPoint(v: Vec3): Vec3 }): Box3 {
    if (this.isEmpty()) return Box3.empty();
    const { min: mn, max: mx } = this;
    const corners: Vec3[] = [
      new Vec3(mn.x, mn.y, mn.z), new Vec3(mx.x, mn.y, mn.z),
      new Vec3(mn.x, mx.y, mn.z), new Vec3(mx.x, mx.y, mn.z),
      new Vec3(mn.x, mn.y, mx.z), new Vec3(mx.x, mn.y, mx.z),
      new Vec3(mn.x, mx.y, mx.z), new Vec3(mx.x, mx.y, mx.z),
    ];
    return Box3.fromPoints(corners.map(c => {
      const t = m.transformPoint(c);
      return new Vec3(t.x, t.y, t.z);
    }));
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(b: Box3, eps = 1e-9): boolean {
    return this.min.equals(b.min, eps) && this.max.equals(b.max, eps);
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toString(): string {
    return `Box3(min=${this.min}, max=${this.max})`;
  }
}
