/**
 * Vec3 — 3D vector
 * AT gap-fill: math/utility (#75 xiaodi)
 */
export class Vec3 {
  constructor(public x = 0, public y = 0, public z = 0) {}

  // ── factory ──────────────────────────────────────────────────────────────
  static zero(): Vec3 { return new Vec3(0, 0, 0); }
  static one(): Vec3  { return new Vec3(1, 1, 1); }
  static up(): Vec3   { return new Vec3(0, 1, 0); }
  static right(): Vec3 { return new Vec3(1, 0, 0); }
  static forward(): Vec3 { return new Vec3(0, 0, -1); }
  static from(v: { x: number; y: number; z: number }): Vec3 {
    return new Vec3(v.x, v.y, v.z);
  }
  static fromArray(a: ArrayLike<number>, offset = 0): Vec3 {
    return new Vec3(a[offset], a[offset + 1], a[offset + 2]);
  }

  // ── setters ───────────────────────────────────────────────────────────────
  set(x: number, y: number, z: number): this { this.x = x; this.y = y; this.z = z; return this; }
  copy(v: Vec3): this { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  clone(): Vec3 { return new Vec3(this.x, this.y, this.z); }

  // ── arithmetic (new) ─────────────────────────────────────────────────────
  add(v: Vec3): Vec3 { return new Vec3(this.x + v.x, this.y + v.y, this.z + v.z); }
  sub(v: Vec3): Vec3 { return new Vec3(this.x - v.x, this.y - v.y, this.z - v.z); }
  mul(v: Vec3 | number): Vec3 {
    return typeof v === 'number'
      ? new Vec3(this.x * v, this.y * v, this.z * v)
      : new Vec3(this.x * v.x, this.y * v.y, this.z * v.z);
  }
  div(v: Vec3 | number): Vec3 {
    return typeof v === 'number'
      ? new Vec3(this.x / v, this.y / v, this.z / v)
      : new Vec3(this.x / v.x, this.y / v.y, this.z / v.z);
  }
  negate(): Vec3 { return new Vec3(-this.x, -this.y, -this.z); }

  // ── in-place ──────────────────────────────────────────────────────────────
  addSelf(v: Vec3): this { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
  subSelf(v: Vec3): this { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
  mulSelf(s: number): this { this.x *= s; this.y *= s; this.z *= s; return this; }
  divSelf(s: number): this { this.x /= s; this.y /= s; this.z /= s; return this; }

  // ── geometry ──────────────────────────────────────────────────────────────
  dot(v: Vec3): number { return this.x * v.x + this.y * v.y + this.z * v.z; }
  cross(v: Vec3): Vec3 {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x,
    );
  }
  crossSelf(v: Vec3): this {
    const x = this.y * v.z - this.z * v.y;
    const y = this.z * v.x - this.x * v.z;
    const z = this.x * v.y - this.y * v.x;
    this.x = x; this.y = y; this.z = z;
    return this;
  }
  lengthSq(): number { return this.x * this.x + this.y * this.y + this.z * this.z; }
  length(): number { return Math.sqrt(this.lengthSq()); }
  normalize(): Vec3 {
    const l = this.length();
    return l > 0 ? this.div(l) : this.clone();
  }
  normalizeSelf(): this {
    const l = this.length();
    if (l > 0) { this.x /= l; this.y /= l; this.z /= l; }
    return this;
  }
  distanceTo(v: Vec3): number { return this.sub(v).length(); }
  distanceSqTo(v: Vec3): number { return this.sub(v).lengthSq(); }
  lerp(v: Vec3, t: number): Vec3 {
    return new Vec3(
      this.x + (v.x - this.x) * t,
      this.y + (v.y - this.y) * t,
      this.z + (v.z - this.z) * t,
    );
  }
  lerpSelf(v: Vec3, t: number): this {
    this.x += (v.x - this.x) * t;
    this.y += (v.y - this.y) * t;
    this.z += (v.z - this.z) * t;
    return this;
  }
  reflect(normal: Vec3): Vec3 {
    // r = d - 2(d·n)n
    return this.sub(normal.mul(2 * this.dot(normal)));
  }
  applyQuat(q: { x: number; y: number; z: number; w: number }): Vec3 {
    // v' = q * (0,v) * q^-1  — efficient formula
    const { x: qx, y: qy, z: qz, w: qw } = q;
    const tx = 2 * (qy * this.z - qz * this.y);
    const ty = 2 * (qz * this.x - qx * this.z);
    const tz = 2 * (qx * this.y - qy * this.x);
    return new Vec3(
      this.x + qw * tx + qy * tz - qz * ty,
      this.y + qw * ty + qz * tx - qx * tz,
      this.z + qw * tz + qx * ty - qy * tx,
    );
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(v: Vec3, eps = 1e-9): boolean {
    return (
      Math.abs(this.x - v.x) <= eps &&
      Math.abs(this.y - v.y) <= eps &&
      Math.abs(this.z - v.z) <= eps
    );
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toArray(): [number, number, number] { return [this.x, this.y, this.z]; }
  toFloat32Array(): Float32Array { return new Float32Array([this.x, this.y, this.z]); }
  toString(): string { return `Vec3(${this.x}, ${this.y}, ${this.z})`; }
}
