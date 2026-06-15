/**
 * Vec4 — 4D vector / homogeneous coordinates
 * AT gap-fill: math/utility (#75 xiaodi)
 */
import type { Vec3 } from './Vec3';

export class Vec4 {
  constructor(public x = 0, public y = 0, public z = 0, public w = 1) {}

  // ── factory ──────────────────────────────────────────────────────────────
  static zero(): Vec4 { return new Vec4(0, 0, 0, 0); }
  static one(): Vec4  { return new Vec4(1, 1, 1, 1); }
  static fromVec3(v: Vec3, w = 1): Vec4 { return new Vec4(v.x, v.y, v.z, w); }
  static fromArray(a: ArrayLike<number>, offset = 0): Vec4 {
    return new Vec4(a[offset], a[offset + 1], a[offset + 2], a[offset + 3]);
  }

  // ── setters ───────────────────────────────────────────────────────────────
  set(x: number, y: number, z: number, w: number): this {
    this.x = x; this.y = y; this.z = z; this.w = w; return this;
  }
  copy(v: Vec4): this { this.x = v.x; this.y = v.y; this.z = v.z; this.w = v.w; return this; }
  clone(): Vec4 { return new Vec4(this.x, this.y, this.z, this.w); }

  // ── arithmetic (new) ─────────────────────────────────────────────────────
  add(v: Vec4): Vec4 {
    return new Vec4(this.x + v.x, this.y + v.y, this.z + v.z, this.w + v.w);
  }
  sub(v: Vec4): Vec4 {
    return new Vec4(this.x - v.x, this.y - v.y, this.z - v.z, this.w - v.w);
  }
  mul(v: Vec4 | number): Vec4 {
    return typeof v === 'number'
      ? new Vec4(this.x * v, this.y * v, this.z * v, this.w * v)
      : new Vec4(this.x * v.x, this.y * v.y, this.z * v.z, this.w * v.w);
  }
  div(v: Vec4 | number): Vec4 {
    return typeof v === 'number'
      ? new Vec4(this.x / v, this.y / v, this.z / v, this.w / v)
      : new Vec4(this.x / v.x, this.y / v.y, this.z / v.z, this.w / v.w);
  }
  negate(): Vec4 { return new Vec4(-this.x, -this.y, -this.z, -this.w); }

  // ── in-place ──────────────────────────────────────────────────────────────
  addSelf(v: Vec4): this {
    this.x += v.x; this.y += v.y; this.z += v.z; this.w += v.w; return this;
  }
  subSelf(v: Vec4): this {
    this.x -= v.x; this.y -= v.y; this.z -= v.z; this.w -= v.w; return this;
  }
  mulSelf(s: number): this {
    this.x *= s; this.y *= s; this.z *= s; this.w *= s; return this;
  }
  divSelf(s: number): this {
    this.x /= s; this.y /= s; this.z /= s; this.w /= s; return this;
  }

  // ── geometry ──────────────────────────────────────────────────────────────
  dot(v: Vec4): number {
    return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
  }
  lengthSq(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
  }
  length(): number { return Math.sqrt(this.lengthSq()); }
  normalize(): Vec4 {
    const l = this.length();
    return l > 0 ? this.div(l) : this.clone();
  }
  normalizeSelf(): this {
    const l = this.length();
    if (l > 0) { this.x /= l; this.y /= l; this.z /= l; this.w /= l; }
    return this;
  }
  lerp(v: Vec4, t: number): Vec4 {
    return new Vec4(
      this.x + (v.x - this.x) * t,
      this.y + (v.y - this.y) * t,
      this.z + (v.z - this.z) * t,
      this.w + (v.w - this.w) * t,
    );
  }
  /** Perspective divide — returns xyz/w as a plain object. */
  perspectiveDivide(): { x: number; y: number; z: number } {
    const inv = 1 / this.w;
    return { x: this.x * inv, y: this.y * inv, z: this.z * inv };
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(v: Vec4, eps = 1e-9): boolean {
    return (
      Math.abs(this.x - v.x) <= eps &&
      Math.abs(this.y - v.y) <= eps &&
      Math.abs(this.z - v.z) <= eps &&
      Math.abs(this.w - v.w) <= eps
    );
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toArray(): [number, number, number, number] { return [this.x, this.y, this.z, this.w]; }
  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y, this.z, this.w]);
  }
  toString(): string { return `Vec4(${this.x}, ${this.y}, ${this.z}, ${this.w})`; }
}
