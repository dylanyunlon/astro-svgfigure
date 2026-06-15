/**
 * Vec2 — 2D vector
 * AT gap-fill: math/utility (#75 xiaodi)
 */
export class Vec2 {
  constructor(public x = 0, public y = 0) {}

  // ── factory ──────────────────────────────────────────────────────────────
  static zero(): Vec2 { return new Vec2(0, 0); }
  static one(): Vec2  { return new Vec2(1, 1); }
  static from(v: { x: number; y: number }): Vec2 { return new Vec2(v.x, v.y); }

  // ── mutating setters ──────────────────────────────────────────────────────
  set(x: number, y: number): this { this.x = x; this.y = y; return this; }
  copy(v: Vec2): this { this.x = v.x; this.y = v.y; return this; }
  clone(): Vec2 { return new Vec2(this.x, this.y); }

  // ── arithmetic (returns new Vec2) ─────────────────────────────────────────
  add(v: Vec2): Vec2 { return new Vec2(this.x + v.x, this.y + v.y); }
  sub(v: Vec2): Vec2 { return new Vec2(this.x - v.x, this.y - v.y); }
  mul(v: Vec2 | number): Vec2 {
    return typeof v === 'number'
      ? new Vec2(this.x * v, this.y * v)
      : new Vec2(this.x * v.x, this.y * v.y);
  }
  div(v: Vec2 | number): Vec2 {
    return typeof v === 'number'
      ? new Vec2(this.x / v, this.y / v)
      : new Vec2(this.x / v.x, this.y / v.y);
  }
  negate(): Vec2 { return new Vec2(-this.x, -this.y); }

  // ── in-place arithmetic ───────────────────────────────────────────────────
  addSelf(v: Vec2): this { this.x += v.x; this.y += v.y; return this; }
  subSelf(v: Vec2): this { this.x -= v.x; this.y -= v.y; return this; }
  mulSelf(s: number): this { this.x *= s; this.y *= s; return this; }
  divSelf(s: number): this { this.x /= s; this.y /= s; return this; }

  // ── geometry ──────────────────────────────────────────────────────────────
  dot(v: Vec2): number { return this.x * v.x + this.y * v.y; }
  /** Returns the scalar z-component of the 3-D cross product. */
  cross(v: Vec2): number { return this.x * v.y - this.y * v.x; }
  lengthSq(): number { return this.x * this.x + this.y * this.y; }
  length(): number { return Math.sqrt(this.lengthSq()); }
  normalize(): Vec2 {
    const l = this.length();
    return l > 0 ? this.div(l) : this.clone();
  }
  normalizeSelf(): this {
    const l = this.length();
    if (l > 0) { this.x /= l; this.y /= l; }
    return this;
  }
  distanceTo(v: Vec2): number { return this.sub(v).length(); }
  distanceSqTo(v: Vec2): number { return this.sub(v).lengthSq(); }
  angle(): number { return Math.atan2(this.y, this.x); }
  rotate(radians: number): Vec2 {
    const c = Math.cos(radians), s = Math.sin(radians);
    return new Vec2(this.x * c - this.y * s, this.x * s + this.y * c);
  }
  lerp(v: Vec2, t: number): Vec2 {
    return new Vec2(this.x + (v.x - this.x) * t, this.y + (v.y - this.y) * t);
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(v: Vec2, eps = 1e-9): boolean {
    return Math.abs(this.x - v.x) <= eps && Math.abs(this.y - v.y) <= eps;
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toArray(): [number, number] { return [this.x, this.y]; }
  toString(): string { return `Vec2(${this.x}, ${this.y})`; }
}
