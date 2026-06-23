// math.ts
// Shared 2D math primitives for the collision subsystem.
// Provides value-type vector/matrix operations with zero allocations where
// possible (scalar-return helpers) and lightweight object returns elsewhere.

// ─── Vec2 ─────────────────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

export function vec2(x: number, y: number): Vec2 {
  return { x, y };
}

export const Vec2Zero: Readonly<Vec2> = { x: 0, y: 0 };

export function vec2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function vec2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function vec2Scale(v: Vec2, s: number): Vec2 {
  return { x: v.x * s, y: v.y * s };
}

export function vec2Negate(v: Vec2): Vec2 {
  return { x: -v.x, y: -v.y };
}

/** Dot product. */
export function vec2Dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

/** 2D cross product (scalar). Positive when b is CCW from a. */
export function vec2Cross(a: Vec2, b: Vec2): number {
  return a.x * b.y - a.y * b.x;
}

/** Cross product of a scalar and a vector: s × v = (-s*v.y, s*v.x). */
export function vec2CrossSV(s: number, v: Vec2): Vec2 {
  return { x: -s * v.y, y: s * v.x };
}

/** Cross product of a vector and a scalar: v × s = (s*v.y, -s*v.x). */
export function vec2CrossVS(v: Vec2, s: number): Vec2 {
  return { x: s * v.y, y: -s * v.x };
}

export function vec2LengthSq(v: Vec2): number {
  return v.x * v.x + v.y * v.y;
}

export function vec2Length(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function vec2Normalize(v: Vec2): Vec2 {
  const len = vec2Length(v);
  if (len < 1e-12) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function vec2Distance(a: Vec2, b: Vec2): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function vec2DistanceSq(a: Vec2, b: Vec2): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return dx * dx + dy * dy;
}

/** Linear interpolation: a + t*(b - a). */
export function vec2Lerp(a: Vec2, b: Vec2, t: number): Vec2 {
  return { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };
}

/** Perpendicular vector (90° CCW rotation): (-y, x). */
export function vec2Perp(v: Vec2): Vec2 {
  return { x: -v.y, y: v.x };
}

/** Clamp each component independently. */
export function vec2Clamp(v: Vec2, min: Vec2, max: Vec2): Vec2 {
  return {
    x: Math.max(min.x, Math.min(max.x, v.x)),
    y: Math.max(min.y, Math.min(max.y, v.y)),
  };
}

/** Component-wise min. */
export function vec2Min(a: Vec2, b: Vec2): Vec2 {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y) };
}

/** Component-wise max. */
export function vec2Max(a: Vec2, b: Vec2): Vec2 {
  return { x: Math.max(a.x, b.x), y: Math.max(a.y, b.y) };
}

/** Absolute value per component. */
export function vec2Abs(v: Vec2): Vec2 {
  return { x: Math.abs(v.x), y: Math.abs(v.y) };
}

// ─── Mat2x2 ───────────────────────────────────────────────────────────────────
// Column-major 2×2 matrix for rotation and local↔world transforms.
//   | a  b |
//   | c  d |

export interface Mat2x2 {
  a: number; b: number;
  c: number; d: number;
}

/** Build a rotation matrix from an angle (radians). */
export function mat2Rotation(angle: number): Mat2x2 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return { a: cos, b: -sin, c: sin, d: cos };
}

/** Identity matrix. */
export const Mat2x2Identity: Readonly<Mat2x2> = { a: 1, b: 0, c: 0, d: 1 };

/** Multiply matrix by vector: M * v. */
export function mat2MulVec(m: Mat2x2, v: Vec2): Vec2 {
  return { x: m.a * v.x + m.b * v.y, y: m.c * v.x + m.d * v.y };
}

/** Transpose-multiply: M^T * v (inverse rotation for orthonormal M). */
export function mat2TransposeMulVec(m: Mat2x2, v: Vec2): Vec2 {
  return { x: m.a * v.x + m.c * v.y, y: m.b * v.x + m.d * v.y };
}

/** Multiply two 2×2 matrices: A * B. */
export function mat2Mul(A: Mat2x2, B: Mat2x2): Mat2x2 {
  return {
    a: A.a * B.a + A.b * B.c,
    b: A.a * B.b + A.b * B.d,
    c: A.c * B.a + A.d * B.c,
    d: A.c * B.b + A.d * B.d,
  };
}

/** Transpose of a 2×2 matrix. */
export function mat2Transpose(m: Mat2x2): Mat2x2 {
  return { a: m.a, b: m.c, c: m.b, d: m.d };
}

/** Determinant of a 2×2 matrix. */
export function mat2Det(m: Mat2x2): number {
  return m.a * m.d - m.b * m.c;
}

/** Inverse of a 2×2 matrix. Returns null if singular. */
export function mat2Inverse(m: Mat2x2): Mat2x2 | null {
  const det = mat2Det(m);
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  return {
    a:  m.d * invDet,
    b: -m.b * invDet,
    c: -m.c * invDet,
    d:  m.a * invDet,
  };
}

// ─── Scalar helpers ───────────────────────────────────────────────────────────

/** Clamp a scalar to [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Approximately equal within tolerance. */
export function approxEqual(a: number, b: number, epsilon = 1e-6): boolean {
  return Math.abs(a - b) <= epsilon;
}

/** Sign function returning -1, 0, or 1. */
export function sign(x: number): number {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

/** Wrap angle to (-π, π]. */
export function wrapAngle(angle: number): number {
  return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}
