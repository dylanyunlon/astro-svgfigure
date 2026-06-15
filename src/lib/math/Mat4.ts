/**
 * Mat4 — column-major 4×4 matrix (same layout as WebGL / Three.js)
 * AT gap-fill: math/utility (#75 xiaodi)
 *
 * Element order in the flat array (column-major):
 *   [ m00 m10 m20 m30  m01 m11 m21 m31  m02 m12 m22 m32  m03 m13 m23 m33 ]
 * i.e. elements[col*4 + row]
 */
import type { Vec3 } from './Vec3';
import type { Quat } from './Quat';

export class Mat4 {
  /** Column-major flat storage, 16 elements. */
  readonly elements: Float32Array;

  constructor(elements?: ArrayLike<number>) {
    this.elements = new Float32Array(16);
    if (elements) {
      this.elements.set(elements);
    } else {
      this.identity();
    }
  }

  // ── factory ──────────────────────────────────────────────────────────────
  static identity(): Mat4 { return new Mat4(); }

  static fromArray(a: ArrayLike<number>, offset = 0): Mat4 {
    const m = new Mat4();
    for (let i = 0; i < 16; i++) m.elements[i] = a[offset + i];
    return m;
  }

  clone(): Mat4 { return new Mat4(this.elements); }
  copy(m: Mat4): this { this.elements.set(m.elements); return this; }

  // ── identity ──────────────────────────────────────────────────────────────
  identity(): this {
    const e = this.elements;
    e.fill(0);
    e[0] = e[5] = e[10] = e[15] = 1;
    return this;
  }

  // ── multiply ──────────────────────────────────────────────────────────────
  /** Returns this * b */
  multiply(b: Mat4): Mat4 {
    return Mat4.multiply(this, b);
  }
  /** Returns a * b */
  static multiply(a: Mat4, b: Mat4): Mat4 {
    const ae = a.elements, be = b.elements;
    const out = new Mat4();
    const oe = out.elements;
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += ae[k * 4 + row] * be[col * 4 + k];
        }
        oe[col * 4 + row] = sum;
      }
    }
    return out;
  }

  // ── transpose ─────────────────────────────────────────────────────────────
  transpose(): Mat4 {
    const e = this.elements;
    const out = new Mat4();
    const oe = out.elements;
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        oe[r * 4 + c] = e[c * 4 + r];
    return out;
  }

  // ── inverse ───────────────────────────────────────────────────────────────
  inverse(): Mat4 {
    const e = this.elements;
    const out = new Mat4();
    const oe = out.elements;

    const n11 = e[0],  n21 = e[1],  n31 = e[2],  n41 = e[3];
    const n12 = e[4],  n22 = e[5],  n32 = e[6],  n42 = e[7];
    const n13 = e[8],  n23 = e[9],  n33 = e[10], n43 = e[11];
    const n14 = e[12], n24 = e[13], n34 = e[14], n44 = e[15];

    const t11 = n23*n34*n42 - n24*n33*n42 + n24*n32*n43 - n22*n34*n43 - n23*n32*n44 + n22*n33*n44;
    const t12 = n14*n33*n42 - n13*n34*n42 - n14*n32*n43 + n12*n34*n43 + n13*n32*n44 - n12*n33*n44;
    const t13 = n13*n24*n42 - n14*n23*n42 + n14*n22*n43 - n12*n24*n43 - n13*n22*n44 + n12*n23*n44;
    const t14 = n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34;

    const det = n11*t11 + n21*t12 + n31*t13 + n41*t14;
    if (Math.abs(det) < 1e-15) {
      console.warn('Mat4.inverse(): matrix is singular');
      return Mat4.identity();
    }
    const inv = 1 / det;

    oe[0]  = t11 * inv;
    oe[1]  = (n24*n33*n41 - n23*n34*n41 - n24*n31*n43 + n21*n34*n43 + n23*n31*n44 - n21*n33*n44) * inv;
    oe[2]  = (n22*n34*n41 - n24*n32*n41 + n24*n31*n42 - n21*n34*n42 - n22*n31*n44 + n21*n32*n44) * inv;
    oe[3]  = (n23*n32*n41 - n22*n33*n41 - n23*n31*n42 + n21*n33*n42 + n22*n31*n43 - n21*n32*n43) * inv;
    oe[4]  = t12 * inv;
    oe[5]  = (n13*n34*n41 - n14*n33*n41 + n14*n31*n43 - n11*n34*n43 - n13*n31*n44 + n11*n33*n44) * inv;
    oe[6]  = (n14*n32*n41 - n12*n34*n41 - n14*n31*n42 + n11*n34*n42 + n12*n31*n44 - n11*n32*n44) * inv;
    oe[7]  = (n12*n33*n41 - n13*n32*n41 + n13*n31*n42 - n11*n33*n42 - n12*n31*n43 + n11*n32*n43) * inv;
    oe[8]  = t13 * inv;
    oe[9]  = (n14*n23*n41 - n13*n24*n41 - n14*n21*n43 + n11*n24*n43 + n13*n21*n44 - n11*n23*n44) * inv;
    oe[10] = (n12*n24*n41 - n14*n22*n41 + n14*n21*n42 - n11*n24*n42 - n12*n21*n44 + n11*n22*n44) * inv;
    oe[11] = (n13*n22*n41 - n12*n23*n41 - n13*n21*n42 + n11*n23*n42 + n12*n21*n43 - n11*n22*n43) * inv;
    oe[12] = t14 * inv;
    oe[13] = (n13*n24*n31 - n14*n23*n31 + n14*n21*n33 - n11*n24*n33 - n13*n21*n34 + n11*n23*n34) * inv;
    oe[14] = (n14*n22*n31 - n12*n24*n31 - n14*n21*n32 + n11*n24*n32 + n12*n21*n34 - n11*n22*n34) * inv;
    oe[15] = (n12*n23*n31 - n13*n22*n31 + n13*n21*n32 - n11*n23*n32 - n12*n21*n33 + n11*n22*n33) * inv;

    return out;
  }

  // ── TRS decompose / compose ───────────────────────────────────────────────
  /** Build a matrix from translation, quaternion rotation and scale. */
  static compose(
    position: Vec3,
    quaternion: { x: number; y: number; z: number; w: number },
    scale: Vec3,
  ): Mat4 {
    const { x, y, z, w } = quaternion;
    const { x: sx, y: sy, z: sz } = scale;

    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    const e = new Float32Array(16);
    e[0]  = (1 - (yy + zz)) * sx;
    e[1]  = (xy + wz) * sx;
    e[2]  = (xz - wy) * sx;
    e[3]  = 0;
    e[4]  = (xy - wz) * sy;
    e[5]  = (1 - (xx + zz)) * sy;
    e[6]  = (yz + wx) * sy;
    e[7]  = 0;
    e[8]  = (xz + wy) * sz;
    e[9]  = (yz - wx) * sz;
    e[10] = (1 - (xx + yy)) * sz;
    e[11] = 0;
    e[12] = position.x;
    e[13] = position.y;
    e[14] = position.z;
    e[15] = 1;
    return new Mat4(e);
  }

  // ── projection ────────────────────────────────────────────────────────────
  /**
   * WebGL/column-major perspective projection.
   * @param fovY  vertical field of view in radians
   * @param aspect  width / height
   * @param near  near clip plane (positive)
   * @param far   far clip plane (positive)
   */
  static perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    const e = new Float32Array(16);
    e[0]  = f / aspect;
    e[5]  = f;
    e[10] = (far + near) * nf;
    e[11] = -1;
    e[14] = 2 * far * near * nf;
    return new Mat4(e);
  }

  /** Orthographic projection. */
  static orthographic(
    left: number, right: number,
    bottom: number, top: number,
    near: number, far: number,
  ): Mat4 {
    const w = 1 / (right - left);
    const h = 1 / (top - bottom);
    const d = 1 / (near - far);
    const e = new Float32Array(16);
    e[0]  = 2 * w;
    e[5]  = 2 * h;
    e[10] = 2 * d;
    e[12] = -(right + left) * w;
    e[13] = -(top + bottom) * h;
    e[14] = (far + near) * d;
    e[15] = 1;
    return new Mat4(e);
  }

  // ── view ──────────────────────────────────────────────────────────────────
  static lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    let zx = eye.x - target.x, zy = eye.y - target.y, zz = eye.z - target.z;
    let len = Math.sqrt(zx*zx + zy*zy + zz*zz);
    if (len > 0) { zx /= len; zy /= len; zz /= len; }

    let xx = up.y * zz - up.z * zy;
    let xy = up.z * zx - up.x * zz;
    let xz = up.x * zy - up.y * zx;
    len = Math.sqrt(xx*xx + xy*xy + xz*xz);
    if (len > 0) { xx /= len; xy /= len; xz /= len; }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const e = new Float32Array(16);
    e[0] = xx; e[1] = yx; e[2] = zx; e[3] = 0;
    e[4] = xy; e[5] = yy; e[6] = zy; e[7] = 0;
    e[8] = xz; e[9] = yz; e[10] = zz; e[11] = 0;
    e[12] = -(xx*eye.x + xy*eye.y + xz*eye.z);
    e[13] = -(yx*eye.x + yy*eye.y + yz*eye.z);
    e[14] = -(zx*eye.x + zy*eye.y + zz*eye.z);
    e[15] = 1;
    return new Mat4(e);
  }

  // ── TRS helpers ───────────────────────────────────────────────────────────
  static translation(v: Vec3): Mat4 {
    const m = Mat4.identity();
    const e = m.elements;
    e[12] = v.x; e[13] = v.y; e[14] = v.z;
    return m;
  }

  static scaling(v: Vec3): Mat4 {
    const m = Mat4.identity();
    const e = m.elements;
    e[0] = v.x; e[5] = v.y; e[10] = v.z;
    return m;
  }

  /** Rotation around an arbitrary unit axis by angle (radians). */
  static rotation(axis: Vec3, angle: number): Mat4 {
    const { x, y, z } = axis;
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const e = new Float32Array(16);
    e[0]  = t*x*x + c;   e[1]  = t*x*y + s*z; e[2]  = t*x*z - s*y; e[3]  = 0;
    e[4]  = t*x*y - s*z; e[5]  = t*y*y + c;   e[6]  = t*y*z + s*x; e[7]  = 0;
    e[8]  = t*x*z + s*y; e[9]  = t*y*z - s*x; e[10] = t*z*z + c;   e[11] = 0;
    e[12] = 0; e[13] = 0; e[14] = 0; e[15] = 1;
    return new Mat4(e);
  }

  /** Build from quaternion (normalised). */
  static fromQuat(q: Quat): Mat4 {
    return q.toMatrix();
  }

  // ── transform helpers ─────────────────────────────────────────────────────
  translate(v: Vec3): Mat4 { return this.multiply(Mat4.translation(v)); }
  scale(v: Vec3): Mat4 { return this.multiply(Mat4.scaling(v)); }
  rotate(axis: Vec3, angle: number): Mat4 { return this.multiply(Mat4.rotation(axis, angle)); }

  /** Transform a position Vec3 (w=1) by this matrix → Vec3 */
  transformPoint(v: Vec3): { x: number; y: number; z: number } {
    const e = this.elements;
    const iw = 1 / (e[3]*v.x + e[7]*v.y + e[11]*v.z + e[15]);
    return {
      x: (e[0]*v.x + e[4]*v.y + e[8] *v.z + e[12]) * iw,
      y: (e[1]*v.x + e[5]*v.y + e[9] *v.z + e[13]) * iw,
      z: (e[2]*v.x + e[6]*v.y + e[10]*v.z + e[14]) * iw,
    };
  }

  /** Transform a direction Vec3 (w=0) by this matrix → Vec3 (no translation). */
  transformDirection(v: Vec3): { x: number; y: number; z: number } {
    const e = this.elements;
    return {
      x: e[0]*v.x + e[4]*v.y + e[8] *v.z,
      y: e[1]*v.x + e[5]*v.y + e[9] *v.z,
      z: e[2]*v.x + e[6]*v.y + e[10]*v.z,
    };
  }

  // ── determinant ───────────────────────────────────────────────────────────
  determinant(): number {
    const e = this.elements;
    const n11=e[0], n21=e[1], n31=e[2], n41=e[3];
    const n12=e[4], n22=e[5], n32=e[6], n42=e[7];
    const n13=e[8], n23=e[9], n33=e[10],n43=e[11];
    const n14=e[12],n24=e[13],n34=e[14],n44=e[15];
    return (
      n41*(n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34) +
      n42*(n11*n23*n34 - n11*n24*n33 + n14*n21*n33 - n13*n21*n34 + n13*n24*n31 - n14*n23*n31) +
      n43*(n11*n24*n32 - n11*n22*n34 - n14*n21*n32 + n12*n21*n34 + n14*n22*n31 - n12*n24*n31) +
      n44*(n13*n22*n31 - n11*n23*n32 + n11*n22*n33 - n12*n21*n33 + n12*n23*n31 - n13*n22*n31)
    );
  }

  // ── serialisation ─────────────────────────────────────────────────────────
  toArray(): number[] { return Array.from(this.elements); }
  toFloat32Array(): Float32Array { return this.elements.slice(); }
  toString(): string {
    const e = this.elements;
    const f = (n: number) => n.toFixed(4).padStart(9);
    return [
      `Mat4 [\n`,
      `  ${f(e[0])} ${f(e[4])} ${f(e[8])}  ${f(e[12])}\n`,
      `  ${f(e[1])} ${f(e[5])} ${f(e[9])}  ${f(e[13])}\n`,
      `  ${f(e[2])} ${f(e[6])} ${f(e[10])} ${f(e[14])}\n`,
      `  ${f(e[3])} ${f(e[7])} ${f(e[11])} ${f(e[15])}\n`,
      `]`,
    ].join('');
  }
}
