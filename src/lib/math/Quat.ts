/**
 * Quat — unit quaternion for 3-D rotations
 * AT gap-fill: math/utility (#75 xiaodi)
 *
 * Stored as (x, y, z, w) — same convention as Three.js / WebXR.
 */
import { Mat4 } from './Mat4';

export class Quat {
  constructor(
    public x = 0,
    public y = 0,
    public z = 0,
    public w = 1,
  ) {}

  // ── factory ──────────────────────────────────────────────────────────────
  static identity(): Quat { return new Quat(0, 0, 0, 1); }

  /**
   * From Euler angles (radians) in XYZ order (intrinsic).
   * Pass order string to change: 'XYZ' | 'YXZ' | 'ZXY' | 'ZYX' | 'YZX' | 'XZY'
   */
  static fromEuler(x: number, y: number, z: number, order: string = 'XYZ'): Quat {
    const c1 = Math.cos(x / 2), c2 = Math.cos(y / 2), c3 = Math.cos(z / 2);
    const s1 = Math.sin(x / 2), s2 = Math.sin(y / 2), s3 = Math.sin(z / 2);
    const q = new Quat();
    switch (order.toUpperCase()) {
      case 'XYZ':
        q.x = s1*c2*c3 + c1*s2*s3;
        q.y = c1*s2*c3 - s1*c2*s3;
        q.z = c1*c2*s3 + s1*s2*c3;
        q.w = c1*c2*c3 - s1*s2*s3;
        break;
      case 'YXZ':
        q.x = s1*c2*c3 + c1*s2*s3;
        q.y = c1*s2*c3 - s1*c2*s3;
        q.z = c1*c2*s3 - s1*s2*c3;
        q.w = c1*c2*c3 + s1*s2*s3;
        break;
      case 'ZXY':
        q.x = s1*c2*c3 - c1*s2*s3;
        q.y = c1*s2*c3 + s1*c2*s3;
        q.z = c1*c2*s3 + s1*s2*c3;
        q.w = c1*c2*c3 - s1*s2*s3;
        break;
      case 'ZYX':
        q.x = s1*c2*c3 - c1*s2*s3;
        q.y = c1*s2*c3 + s1*c2*s3;
        q.z = c1*c2*s3 - s1*s2*c3;
        q.w = c1*c2*c3 + s1*s2*s3;
        break;
      case 'YZX':
        q.x = s1*c2*c3 + c1*s2*s3;
        q.y = c1*s2*c3 + s1*c2*s3;
        q.z = c1*c2*s3 - s1*s2*c3;
        q.w = c1*c2*c3 - s1*s2*s3;
        break;
      case 'XZY':
        q.x = s1*c2*c3 - c1*s2*s3;
        q.y = c1*s2*c3 - s1*c2*s3;
        q.z = c1*c2*s3 + s1*s2*c3;
        q.w = c1*c2*c3 + s1*s2*s3;
        break;
      default:
        throw new Error(`Quat.fromEuler: unknown order "${order}"`);
    }
    return q;
  }

  /** From axis (unit vector) + angle (radians). */
  static fromAxisAngle(ax: number, ay: number, az: number, angle: number): Quat {
    const half = angle / 2, s = Math.sin(half);
    return new Quat(ax * s, ay * s, az * s, Math.cos(half));
  }

  /** Extract from upper-left 3×3 of a rotation matrix (column-major Mat4). */
  static fromMatrix(m: Mat4): Quat {
    const e = m.elements;
    const m11 = e[0], m12 = e[4], m13 = e[8];
    const m21 = e[1], m22 = e[5], m23 = e[9];
    const m31 = e[2], m32 = e[6], m33 = e[10];
    const trace = m11 + m22 + m33;
    const q = new Quat();
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1);
      q.w = 0.25 / s;
      q.x = (m32 - m23) * s;
      q.y = (m13 - m31) * s;
      q.z = (m21 - m12) * s;
    } else if (m11 > m22 && m11 > m33) {
      const s = 2 * Math.sqrt(1 + m11 - m22 - m33);
      q.w = (m32 - m23) / s;
      q.x = 0.25 * s;
      q.y = (m12 + m21) / s;
      q.z = (m13 + m31) / s;
    } else if (m22 > m33) {
      const s = 2 * Math.sqrt(1 + m22 - m11 - m33);
      q.w = (m13 - m31) / s;
      q.x = (m12 + m21) / s;
      q.y = 0.25 * s;
      q.z = (m23 + m32) / s;
    } else {
      const s = 2 * Math.sqrt(1 + m33 - m11 - m22);
      q.w = (m21 - m12) / s;
      q.x = (m13 + m31) / s;
      q.y = (m23 + m32) / s;
      q.z = 0.25 * s;
    }
    return q;
  }

  // ── basic ops ─────────────────────────────────────────────────────────────
  set(x: number, y: number, z: number, w: number): this {
    this.x = x; this.y = y; this.z = z; this.w = w; return this;
  }
  copy(q: Quat): this { this.x = q.x; this.y = q.y; this.z = q.z; this.w = q.w; return this; }
  clone(): Quat { return new Quat(this.x, this.y, this.z, this.w); }

  dot(q: Quat): number { return this.x*q.x + this.y*q.y + this.z*q.z + this.w*q.w; }
  lengthSq(): number { return this.x**2 + this.y**2 + this.z**2 + this.w**2; }
  length(): number { return Math.sqrt(this.lengthSq()); }

  normalize(): Quat {
    const l = this.length();
    return l > 0
      ? new Quat(this.x/l, this.y/l, this.z/l, this.w/l)
      : Quat.identity();
  }
  normalizeSelf(): this {
    const l = this.length();
    if (l > 0) { this.x /= l; this.y /= l; this.z /= l; this.w /= l; }
    return this;
  }

  conjugate(): Quat { return new Quat(-this.x, -this.y, -this.z, this.w); }
  invert(): Quat { return this.conjugate().normalizeSelf(); }

  // ── multiply ──────────────────────────────────────────────────────────────
  /** Hamilton product: this * q */
  multiply(q: Quat): Quat {
    return Quat.multiply(this, q);
  }
  static multiply(a: Quat, b: Quat): Quat {
    return new Quat(
      a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
      a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
      a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
      a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
  }

  // ── slerp ─────────────────────────────────────────────────────────────────
  /** Spherical linear interpolation from this → q at t ∈ [0,1]. */
  slerp(q: Quat, t: number): Quat {
    return Quat.slerp(this, q, t);
  }
  static slerp(a: Quat, b: Quat, t: number): Quat {
    let { x: bx, y: by, z: bz, w: bw } = b;
    let cosHalf = a.x*bx + a.y*by + a.z*bz + a.w*bw;

    // Take the shorter arc
    if (cosHalf < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; cosHalf = -cosHalf; }

    let s0: number, s1: number;
    if (cosHalf > 0.9999) {
      // Linear fallback
      s0 = 1 - t; s1 = t;
    } else {
      const angle = Math.acos(cosHalf);
      const sinHalf = Math.sqrt(1 - cosHalf * cosHalf);
      s0 = Math.sin((1 - t) * angle) / sinHalf;
      s1 = Math.sin(t * angle) / sinHalf;
    }
    return new Quat(
      s0*a.x + s1*bx,
      s0*a.y + s1*by,
      s0*a.z + s1*bz,
      s0*a.w + s1*bw,
    );
  }

  // ── conversion ────────────────────────────────────────────────────────────
  /** Convert to a column-major rotation Mat4. */
  toMatrix(): Mat4 {
    const { x, y, z, w } = this;
    const x2 = x+x, y2 = y+y, z2 = z+z;
    const xx = x*x2, xy = x*y2, xz = x*z2;
    const yy = y*y2, yz = y*z2, zz = z*z2;
    const wx = w*x2, wy = w*y2, wz = w*z2;
    const e = new Float32Array(16);
    e[0]  = 1-(yy+zz); e[1]  = xy+wz;     e[2]  = xz-wy;     e[3]  = 0;
    e[4]  = xy-wz;     e[5]  = 1-(xx+zz); e[6]  = yz+wx;     e[7]  = 0;
    e[8]  = xz+wy;     e[9]  = yz-wx;     e[10] = 1-(xx+yy); e[11] = 0;
    e[12] = 0;         e[13] = 0;         e[14] = 0;          e[15] = 1;
    return new Mat4(e);
  }

  /** Decompose to Euler angles (radians) in XYZ order. */
  toEulerXYZ(): { x: number; y: number; z: number } {
    const m = this.toMatrix().elements;
    // From rotation matrix
    const y = Math.asin(Math.max(-1, Math.min(1, m[8])));
    let x: number, z: number;
    if (Math.abs(m[8]) < 0.9999) {
      x = Math.atan2(-m[9], m[10]);
      z = Math.atan2(-m[4], m[0]);
    } else {
      x = Math.atan2(m[6], m[5]);
      z = 0;
    }
    return { x, y, z };
  }

  // ── comparison ────────────────────────────────────────────────────────────
  equals(q: Quat, eps = 1e-9): boolean {
    return (
      Math.abs(this.x - q.x) <= eps &&
      Math.abs(this.y - q.y) <= eps &&
      Math.abs(this.z - q.z) <= eps &&
      Math.abs(this.w - q.w) <= eps
    );
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toArray(): [number, number, number, number] { return [this.x, this.y, this.z, this.w]; }
  toString(): string { return `Quat(${this.x}, ${this.y}, ${this.z}, ${this.w})`; }
}
