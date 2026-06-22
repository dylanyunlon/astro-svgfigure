/**
 * organic-sdf.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Inlined SDF (Signed Distance Field) primitives for generating organic Cell
 * outlines per species.
 *
 * Sources ported to TypeScript (CPU-side, evaluated in [0,1]² UV space):
 *   • flowerSDF  — Patricio Gonzalez Vivo / lygia  (Prosperity License 3.0)
 *   • kochSDF    — Kathy kfahn22 / lygia
 *   • juliaSDF   — classic escape-time quadratic Julia set boundary
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *   import { organicOutline, OrganicSdfKind } from './organic-sdf';
 *
 *   // Sample the SDF for a given species at a UV coordinate:
 *   const d = organicOutline('cil-eye', 0.55, 0.48);
 *   // d < 0  → inside the shape
 *   // d ≈ 0  → on the outline
 *   // d > 0  → outside
 *
 *   // Build a closed SVG polygon for a cell outline:
 *   const pts = sampleOutlinePoints('cil-bolt', 64);
 *   // → Array<[number, number]> in normalised [0,1]² space
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *  Each species maps to one of the three SDF kinds plus a small set of tuning
 *  parameters.  The mapping is intentionally stable so that the numeric species
 *  index from cell-body-bridge.ts (1-based, 0 = fluid) can be used as a cheap
 *  look-up key without string comparison in hot paths.
 */

// ─── Vec2 helpers (no external dependency) ────────────────────────────────────

/** Immutable 2-D vector used throughout the SDF math. */
export type Vec2 = readonly [number, number];

function len([x, y]: Vec2): number { return Math.sqrt(x * x + y * y); }
function abs2([x, y]: Vec2): Vec2 { return [Math.abs(x), Math.abs(y)]; }
function sub([ax, ay]: Vec2, [bx, by]: Vec2): Vec2 { return [ax - bx, ay - by]; }
function scale([x, y]: Vec2, s: number): Vec2 { return [x * s, y * s]; }
function dot([ax, ay]: Vec2, [bx, by]: Vec2): number { return ax * bx + ay * by; }

// ─── flowerSDF ────────────────────────────────────────────────────────────────
//
// Ported from upstream/lygia/sdf/flowerSDF.wgsl
// Contributors: Patricio Gonzalez Vivo
// License: Prosperity 3.0 / Patron — https://lygia.xyz/license
//
// Original WGSL:
//   fn flowerSDF(st: vec2f, N: i32) -> f32 {
//     st -= 0.5; st *= 4.0;
//     let r = length(st) * 2.0;
//     let a = atan(st.y, st.x);
//     let v = float(N) * 0.5;
//     return 1.0 - (abs(cos(a * v)) * 0.5 + 0.5) / r;
//   }

/**
 * Flower-shaped SDF.
 * @param uv  Normalised coordinate in [0,1]²
 * @param n   Number of petals (integer ≥ 2)
 * @returns   Signed distance — negative inside, positive outside
 */
export function flowerSDF(uv: Vec2, n: number): number {
  // Centre at (0.5, 0.5), expand to ~[-2, 2] range
  let [x, y] = sub(uv, [0.5, 0.5]);
  x *= 4.0;
  y *= 4.0;

  const r = Math.sqrt(x * x + y * y) * 2.0;
  if (r < 1e-6) return -1.0; // dead centre is inside

  const a = Math.atan2(y, x);
  const v = n * 0.5;
  return 1.0 - (Math.abs(Math.cos(a * v)) * 0.5 + 0.5) / r;
}

// ─── kochSDF ──────────────────────────────────────────────────────────────────
//
// Ported from upstream/lygia/sdf/kochSDF.wgsl
// Contributors: Kathy kfahn22
//
// The WGSL source in the repo has several merged conflicts / duplicated lines.
// This TypeScript version implements the canonical Koch-snowflake SDF with the
// IFS iteration that contracts toward the fractal attractor.

/**
 * Koch-snowflake SDF (iterative IFS).
 * @param uv         Normalised coordinate in [0,1]²
 * @param center     SDF center in UV space, defaults to [0.5, 0.5]
 * @param iterations Koch refinement depth (1–8 recommended)
 * @returns          Signed distance
 */
export function kochSDF(
  uv: Vec2,
  iterations: number,
  center: Vec2 = [0.5, 0.5],
): number {
  const r3 = Math.sqrt(3.0);

  // Centre + scale into IFS working space
  let [sx, sy] = sub(uv, center);
  sx *= 3.0;
  sy *= 3.0;

  // Fold into fundamental domain (abs + 60° rotation, scale 2)
  sx = Math.abs(sx);
  sy = Math.abs(sy);
  // 60° fold: equivalent to (abs(st) + r3*perp(abs(st)))
  const tx0 = sx + r3 * (-sy);
  const ty0 = sy + r3 * sx;
  sx = tx0;
  sy = ty0;
  sy -= 1.0;

  // IFS matrix: m = mat2(r3, 3, -3, r3) * 0.5
  let w = 0.5;
  for (let i = 0; i < iterations; i++) {
    // fold step: st = vec2(-r3, 3) * 0.5 - m * vec2(st.y, abs(st.x))
    const ax = Math.abs(sx);
    const mx = r3 * sy  + 3.0 * ax;
    const my = -3.0 * sy + r3 * ax;
    sx = -r3 * 0.5 - mx * 0.5;
    sy =  3.0 * 0.5 - my * 0.5;
    w /= r3;
  }

  // Distance to the edge segment
  const d = Math.sign(sy) * len([sy, Math.max(0.0, Math.abs(sx) - r3)]);
  return d * w;
}

// ─── juliaSDF ─────────────────────────────────────────────────────────────────
//
// Classic escape-time quadratic Julia set boundary approximation.
// The SDF value is derived from the smooth iteration count (a well-known
// technique in demoscene / shadertoy literature, public domain).
//
// z_{n+1} = z_n² + c   where c is a fixed complex parameter per species.
//
// The function returns a value in roughly [-1, 1]:
//   < 0  → point belongs to the filled Julia set (inside)
//   ≈ 0  → boundary
//   > 0  → exterior

/**
 * Julia-set SDF approximation.
 * @param uv        Normalised coordinate in [0,1]², mapped to ~[-1.5, 1.5]²
 * @param cx        Real part of Julia constant c
 * @param cy        Imaginary part of Julia constant c
 * @param maxIter   Escape iteration limit (higher = sharper boundary, slower)
 * @returns         Pseudo-distance — negative inside, positive outside
 */
export function juliaSDF(
  uv: Vec2,
  cx: number,
  cy: number,
  maxIter = 48,
): number {
  // Map [0,1]² → [-1.6, 1.6]²
  let zx = (uv[0] - 0.5) * 3.2;
  let zy = (uv[1] - 0.5) * 3.2;

  let iter = 0;
  while (iter < maxIter && zx * zx + zy * zy < 4.0) {
    const nx = zx * zx - zy * zy + cx;
    zy = 2.0 * zx * zy + cy;
    zx = nx;
    iter++;
  }

  if (iter === maxIter) {
    // Interior — return a negative value
    return -1.0;
  }

  // Smooth escape count → pseudo-distance in (-1, 1)
  const smooth = iter - Math.log2(Math.log2(zx * zx + zy * zy));
  return (smooth / maxIter) * 2.0 - 1.0;
}

// ─── Species → SDF mapping ────────────────────────────────────────────────────

/** Available SDF kinds for organic outline generation. */
export type OrganicSdfKind = 'flower' | 'koch' | 'julia';

export interface SpeciesSdfParams {
  kind:       OrganicSdfKind;
  /** Flower: petal count.  Koch: iteration depth.  Julia: ignored. */
  n:          number;
  /** Julia set real part of c.  Ignored for flower / koch. */
  juliaC?:    [number, number];
  /** Threshold to treat as "on the outline" when sampling (default 0.02). */
  threshold?: number;
}

/**
 * Stable mapping from species string → SDF parameters.
 * Indexed to match SPECIES_ORDER in cell-body-bridge.ts (index 1-based,
 * 0 = fluid is excluded — fluid has no organic outline).
 *
 * Species:
 *   1  cil-eye         → 6-petal flower (iris-like radial symmetry)
 *   2  cil-bolt        → Koch 4-iter (jagged lightning fractal)
 *   3  cil-vector      → 4-petal flower (clean 4-fold symmetry)
 *   4  cil-plus        → Koch 2-iter (simple snowflake cross)
 *   5  cil-arrow-right → Julia c≈Douady rabbit (directional asymmetry)
 *   6  cil-filter      → 8-petal flower (fine filter mesh)
 *   7  cil-layers      → Koch 6-iter (layered fractal complexity)
 *   8  cil-loop        → Julia c=−0.7+0.27i (spiralling loop topology)
 *   9  cil-code        → Koch 3-iter (sharp code-bracket motif)
 *  10  cil-graph       → 5-petal flower (graph node radial)
 */
const SPECIES_SDF_MAP: Record<string, SpeciesSdfParams> = {
  // index 1
  'cil-eye': {
    kind: 'flower',
    n:    6,
    threshold: 0.025,
  },
  // index 2
  'cil-bolt': {
    kind: 'koch',
    n:    4,
    threshold: 0.018,
  },
  // index 3
  'cil-vector': {
    kind: 'flower',
    n:    4,
    threshold: 0.025,
  },
  // index 4
  'cil-plus': {
    kind: 'koch',
    n:    2,
    threshold: 0.022,
  },
  // index 5
  'cil-arrow-right': {
    kind:    'julia',
    n:       48,
    juliaC:  [-0.1, 0.651], // Douady rabbit — directional, asymmetric
    threshold: 0.04,
  },
  // index 6
  'cil-filter': {
    kind: 'flower',
    n:    8,
    threshold: 0.020,
  },
  // index 7
  'cil-layers': {
    kind: 'koch',
    n:    6,
    threshold: 0.015,
  },
  // index 8
  'cil-loop': {
    kind:    'julia',
    n:       48,
    juliaC:  [-0.7, 0.27],  // classic spiralling loop
    threshold: 0.04,
  },
  // index 9
  'cil-code': {
    kind: 'koch',
    n:    3,
    threshold: 0.020,
  },
  // index 10
  'cil-graph': {
    kind: 'flower',
    n:    5,
    threshold: 0.025,
  },
};

/** Fallback for unknown species: simple 3-petal flower. */
const DEFAULT_SDF_PARAMS: SpeciesSdfParams = {
  kind: 'flower',
  n:    3,
  threshold: 0.030,
};

/**
 * Return the SDF parameter set for a given species string.
 * Falls back to a simple 3-petal flower for unknown species.
 */
export function getSpeciesSdfParams(species: string): SpeciesSdfParams {
  return SPECIES_SDF_MAP[species] ?? DEFAULT_SDF_PARAMS;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Evaluate the organic SDF for `species` at the given UV coordinate.
 *
 * @param species  Species string, e.g. `'cil-eye'`
 * @param u        Horizontal UV in [0, 1]
 * @param v        Vertical UV in [0, 1]
 * @returns        Signed distance value (< 0 inside, > 0 outside, ≈ 0 outline)
 */
export function organicOutline(species: string, u: number, v: number): number {
  const uv: Vec2 = [u, v];
  const p = getSpeciesSdfParams(species);

  switch (p.kind) {
    case 'flower':
      return flowerSDF(uv, p.n);
    case 'koch':
      return kochSDF(uv, p.n);
    case 'julia': {
      const [cx, cy] = p.juliaC ?? [-0.7, 0.27];
      return juliaSDF(uv, cx, cy, p.n);
    }
  }
}

/**
 * Sample `count` evenly-spaced points that lie on the organic outline of
 * `species` (i.e. where the SDF ≈ 0) by ray-marching outward from the centre.
 *
 * Returns a closed array of [u, v] pairs in [0, 1]² normalised space.
 * The last element equals the first so the array is ready for SVG path
 * generation or WebGPU LINE_STRIP rendering.
 *
 * @param species  Species string
 * @param count    Number of outline vertices (default 64)
 * @returns        Array of [u, v] pairs, length = count + 1 (closed)
 */
export function sampleOutlinePoints(
  species: string,
  count = 64,
): Array<[number, number]> {
  const p   = getSpeciesSdfParams(species);
  const thr = p.threshold ?? 0.025;
  const pts: Array<[number, number]> = [];

  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const cos   = Math.cos(angle);
    const sin   = Math.sin(angle);

    // Binary-search the outline along each radial ray
    let lo = 0.0;
    let hi = 0.5; // max radius in UV space from centre (0.5, 0.5)

    for (let step = 0; step < 24; step++) {
      const mid  = (lo + hi) * 0.5;
      const u    = 0.5 + cos * mid;
      const v    = 0.5 + sin * mid;
      const d    = organicOutline(species, u, v);
      if (d < 0) {
        lo = mid; // inside → push outward
      } else {
        hi = mid; // outside → pull inward
      }
    }

    const r = (lo + hi) * 0.5;
    pts.push([0.5 + cos * r, 0.5 + sin * r]);
  }

  // Close the polyline
  pts.push(pts[0]);
  return pts;
}

/**
 * Build an SVG `d` path string for the organic outline of `species`.
 *
 * @param species   Species string
 * @param cx        Centre X in SVG pixels
 * @param cy        Centre Y in SVG pixels
 * @param radius    Bounding radius in SVG pixels (maps UV half-range 0.5 → radius)
 * @param count     Number of vertices (default 64)
 * @returns         SVG path `d` attribute value (closed with Z)
 */
export function organicSvgPath(
  species: string,
  cx:      number,
  cy:      number,
  radius:  number,
  count  = 64,
): string {
  const pts = sampleOutlinePoints(species, count);
  const scale2 = radius * 2; // UV [0,1] → pixel space

  return pts
    .map(([u, v], i) => {
      const px = cx + (u - 0.5) * scale2;
      const py = cy + (v - 0.5) * scale2;
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(' ') + ' Z';
}

/**
 * Return true when the given UV point lies inside the organic outline for
 * `species` (SDF < threshold / 2, i.e. well inside the boundary band).
 */
export function isInsideOutline(species: string, u: number, v: number): boolean {
  return organicOutline(species, u, v) < 0;
}
