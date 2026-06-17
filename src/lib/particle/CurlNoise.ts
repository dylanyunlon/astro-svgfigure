/**
 * CurlNoise.ts — CPU-side curl noise utility for particle turbulence
 *
 * TypeScript counterpart to CurlNoise.frag. Computes a divergence-free
 * 3-D vector field via the curl of three independent simplex noise
 * potential functions. Useful for CPU-driven particle paths, baking
 * flow-field textures, or debugging GPU results.
 *
 * Simplex 3D noise is implemented with a classic permutation table
 * (Ken Perlin's approach) rather than hash33 — better gradient
 * distribution and no modular-arithmetic bias.
 *
 * Curl definition (same as the frag shader):
 *   curl.x = ∂Fz/∂y − ∂Fy/∂z
 *   curl.y = ∂Fx/∂z − ∂Fz/∂x
 *   curl.z = ∂Fy/∂x − ∂Fx/∂y
 *
 * where Fx, Fy, Fz are three spatially-offset simplex noise fields.
 */

// ── Permutation table ───────────────────────────────────────────────────────

/**
 * Perlin's original 256-element permutation, doubled to avoid wrapping.
 */
const PERM: readonly number[] = (() => {
  const p = [
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
  ];
  const perm = new Array<number>(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  return perm;
})();

// ── Simplex 3D constants ────────────────────────────────────────────────────

/** Skew factor for 3D simplex grid: (√4 − 1) / 3 */
const F3 = 1 / 3;

/** Unskew factor: (1 − 1/√4) / 3 */
const G3 = 1 / 6;

/**
 * 12 gradient directions for 3D simplex noise — edges of a cube.
 * Using a flat array indexed by (hash & 11) * 3 for speed.
 */
const GRAD3: readonly number[] = [
   1, 1, 0,  -1, 1, 0,   1,-1, 0,  -1,-1, 0,
   1, 0, 1,  -1, 0, 1,   1, 0,-1,  -1, 0,-1,
   0, 1, 1,   0,-1, 1,   0, 1,-1,   0,-1,-1,
];

// ── Simplex 3D noise ────────────────────────────────────────────────────────

/**
 * Compute 3D simplex noise at the given coordinates.
 *
 * Uses the permutation table for gradient selection and the standard
 * simplex 3D algorithm (Stefan Gustavson's formulation). Returns a
 * value in approximately [−1, 1].
 */
function simplex3D(x: number, y: number, z: number): number {
  // Skew input space to determine which simplex cell we're in
  const s = (x + y + z) * F3;
  const i = Math.floor(x + s);
  const j = Math.floor(y + s);
  const k = Math.floor(z + s);

  // Unskew cell origin back to (x,y,z) space
  const t = (i + j + k) * G3;
  const X0 = i - t;
  const Y0 = j - t;
  const Z0 = k - t;

  // Distances from cell origin
  const x0 = x - X0;
  const y0 = y - Y0;
  const z0 = z - Z0;

  // Determine which simplex we are in (3D has 6 possible simplices)
  let i1: number, j1: number, k1: number;
  let i2: number, j2: number, k2: number;

  if (x0 >= y0) {
    if (y0 >= z0) {
      // X Y Z order
      i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
    } else if (x0 >= z0) {
      // X Z Y order
      i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
    } else {
      // Z X Y order
      i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
    }
  } else {
    if (y0 < z0) {
      // Z Y X order
      i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
    } else if (x0 < z0) {
      // Y Z X order
      i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
    } else {
      // Y X Z order
      i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
    }
  }

  // Offsets for the remaining corners of the simplex
  const x1 = x0 - i1 + G3;
  const y1 = y0 - j1 + G3;
  const z1 = z0 - k1 + G3;

  const x2 = x0 - i2 + 2.0 * G3;
  const y2 = y0 - j2 + 2.0 * G3;
  const z2 = z0 - k2 + 2.0 * G3;

  const x3 = x0 - 1.0 + 3.0 * G3;
  const y3 = y0 - 1.0 + 3.0 * G3;
  const z3 = z0 - 1.0 + 3.0 * G3;

  // Hash coordinates of the four simplex corners
  const ii = i & 255;
  const jj = j & 255;
  const kk = k & 255;

  // Gradient indices via permutation table
  const gi0 = PERM[ii      + PERM[jj      + PERM[kk     ]]] % 12;
  const gi1 = PERM[ii + i1 + PERM[jj + j1 + PERM[kk + k1]]] % 12;
  const gi2 = PERM[ii + i2 + PERM[jj + j2 + PERM[kk + k2]]] % 12;
  const gi3 = PERM[ii +  1 + PERM[jj +  1 + PERM[kk +  1]]] % 12;

  // Contribution from each corner
  let n0 = 0, n1 = 0, n2 = 0, n3 = 0;

  let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
  if (t0 >= 0) {
    t0 *= t0;
    const g = gi0 * 3;
    n0 = t0 * t0 * (GRAD3[g] * x0 + GRAD3[g + 1] * y0 + GRAD3[g + 2] * z0);
  }

  let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
  if (t1 >= 0) {
    t1 *= t1;
    const g = gi1 * 3;
    n1 = t1 * t1 * (GRAD3[g] * x1 + GRAD3[g + 1] * y1 + GRAD3[g + 2] * z1);
  }

  let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
  if (t2 >= 0) {
    t2 *= t2;
    const g = gi2 * 3;
    n2 = t2 * t2 * (GRAD3[g] * x2 + GRAD3[g + 1] * y2 + GRAD3[g + 2] * z2);
  }

  let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
  if (t3 >= 0) {
    t3 *= t3;
    const g = gi3 * 3;
    n3 = t3 * t3 * (GRAD3[g] * x3 + GRAD3[g + 1] * y3 + GRAD3[g + 2] * z3);
  }

  // Scale to [-1, 1]
  return 32.0 * (n0 + n1 + n2 + n3);
}

// ── Curl noise ──────────────────────────────────────────────────────────────

/**
 * Domain offsets for the three independent potential fields (Fx, Fy, Fz).
 * Matching the frag shader's spatial separation (31.41 and 27.18) to keep
 * CPU and GPU results qualitatively similar.
 */
const OFFSET_FY = 31.41;
const OFFSET_FZ = 27.18;

/**
 * Finite-difference step for partial derivative approximation.
 * Same epsilon as the GLSL version.
 */
const EPS = 0.0001;
const INV_2EPS = 1.0 / (2.0 * EPS);

/**
 * Compute a divergence-free 3D curl noise vector at the given position.
 *
 * Three independent simplex noise fields (Fx, Fy, Fz) form a vector
 * potential. The curl of that potential is computed via central finite
 * differences, yielding a solenoidal (∇·v = 0) velocity field ideal
 * for fluid-like particle motion.
 *
 * @param x - World-space x coordinate
 * @param y - World-space y coordinate
 * @param z - World-space z coordinate
 * @returns [curl_x, curl_y, curl_z] divergence-free vector
 *
 * @example
 * ```ts
 * import { curlNoise3D } from '$lib/particle/CurlNoise';
 *
 * // Sample the curl field and apply as velocity
 * const [vx, vy, vz] = curlNoise3D(
 *   pos.x * scale + time * drift,
 *   pos.y * scale + time * drift,
 *   pos.z * scale + time * drift,
 * );
 * pos.x += vx * speed * delta;
 * pos.y += vy * speed * delta;
 * pos.z += vz * speed * delta;
 * ```
 */
export function curlNoise3D(
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  // ── Partial derivatives of Fx (noise at base domain) ──────────────────
  const Fx_py = simplex3D(x, y + EPS, z);
  const Fx_my = simplex3D(x, y - EPS, z);
  const Fx_pz = simplex3D(x, y, z + EPS);
  const Fx_mz = simplex3D(x, y, z - EPS);

  // ── Partial derivatives of Fy (offset by OFFSET_FY on x axis) ────────
  const Fy_pz = simplex3D(x + OFFSET_FY, y, z + EPS);
  const Fy_mz = simplex3D(x + OFFSET_FY, y, z - EPS);
  const Fy_px = simplex3D(x + OFFSET_FY + EPS, y, z);
  const Fy_mx = simplex3D(x + OFFSET_FY - EPS, y, z);

  // ── Partial derivatives of Fz (offset by OFFSET_FZ on y axis) ────────
  const Fz_px = simplex3D(x + EPS, y + OFFSET_FZ, z);
  const Fz_mx = simplex3D(x - EPS, y + OFFSET_FZ, z);
  const Fz_py = simplex3D(x, y + OFFSET_FZ + EPS, z);
  const Fz_my = simplex3D(x, y + OFFSET_FZ - EPS, z);

  // ── Curl = ∇ × F ─────────────────────────────────────────────────────
  const curlX = (Fz_py - Fz_my - (Fy_pz - Fy_mz)) * INV_2EPS;
  const curlY = (Fx_pz - Fx_mz - (Fz_px - Fz_mx)) * INV_2EPS;
  const curlZ = (Fy_px - Fy_mx - (Fx_py - Fx_my)) * INV_2EPS;

  return [curlX, curlY, curlZ];
}
