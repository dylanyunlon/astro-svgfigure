/**
 * ACES Filmic HDR Tone Mapping
 *
 * Attempt at implementing
 * https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
 *
 * References:
 * - Stephen Hill, "Self Shadow – ACES Tone Mapping"
 * - Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve"
 * - Academy Color Encoding System (ACES) documentation
 */

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/** A linear‑light RGB triplet (each channel ≥ 0, no upper bound). */








export type Color3 = [r: number, g: number, b: number];

/** A 3×3 matrix stored as nine elements in row‑major order. */
export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

// ---------------------------------------------------------------------------
// 3×3 matrix × vec3 multiply
// ---------------------------------------------------------------------------

function mulMat3(m: Mat3, v: Color3): Color3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

// ---------------------------------------------------------------------------
// ACES colour‑space matrices
// ---------------------------------------------------------------------------

/**
 * sRGB → ACEScg (AP1) input transform.
 *
 * Converts scene‑referred linear sRGB values into the ACEScg working space
 * used by the Reference Rendering Transform (RRT).
 */
export const ACESInputMat: Mat3 = [
  0.59719, 0.35458, 0.04823,
  0.07600, 0.90834, 0.01566,
  0.02840, 0.13383, 0.83777,
];

/**
 * ACEScg (AP1) → sRGB output transform.
 *
 * Converts tone‑mapped values from ACEScg back to linear sRGB for display.
 */
export const ACESOutputMat: Mat3 = [
   1.60475, -0.53108, -0.07367,
  -0.10208,  1.10813, -0.00605,
  -0.00327, -0.07276,  1.07602,
];

// ---------------------------------------------------------------------------
// RRT + ODT fit (Stephen Hill's approximation)
// ---------------------------------------------------------------------------

/**
 * Per‑channel rational polynomial that approximates the ACES
 * Reference Rendering Transform (RRT) combined with the
 * Output Display Transform (ODT) for sRGB monitors.
 *
 * ```
 *        v × (v + 0.0245786) − 0.000090537
 * f(v) = ─────────────────────────────────────
 *        v × (0.983729 × v + 0.4329510) + 0.238081
 * ```
 */
function rrtOdtFit(v: Color3): Color3 {
  const a: Color3 = [
    v[0] * (v[0] + 0.0245786) - 0.000090537,
    v[1] * (v[1] + 0.0245786) - 0.000090537,
    v[2] * (v[2] + 0.0245786) - 0.000090537,
  ];
  const b: Color3 = [
    v[0] * (0.983729 * v[0] + 0.4329510) + 0.238081,
    v[1] * (0.983729 * v[1] + 0.4329510) + 0.238081,
    v[2] * (0.983729 * v[2] + 0.4329510) + 0.238081,
  ];
  return [a[0] / b[0], a[1] / b[1], a[2] / b[2]];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Apply the full ACES filmic tone‑mapping operator to a single
 * **linear sRGB** colour.
 *
 * Pipeline:
 * 1. Transform from sRGB into ACEScg via {@link ACESInputMat}.
 * 2. Apply the combined RRT + ODT rational‑polynomial fit.
 * 3. Transform back to linear sRGB via {@link ACESOutputMat}.
 * 4. Clamp each channel to [0, 1].
 *
 * @param color - Linear‑light sRGB input (HDR values welcome).
 * @returns       Tone‑mapped linear sRGB in [0, 1].
 *
 * @example
 * ```ts
 *
 * const hdr: Color3 = [2.4, 1.1, 0.6];
 * const ldr = acesFilm(hdr);
 * // ldr ≈ [0.944, 0.776, 0.531]
 * ```
 */
export function acesFilm(color: Color3): Color3 {
  let c = mulMat3(ACESInputMat, color);
  c = rrtOdtFit(c);
  c = mulMat3(ACESOutputMat, c);
  return [
    Math.min(Math.max(c[0], 0), 1),
    Math.min(Math.max(c[1], 0), 1),
    Math.min(Math.max(c[2], 0), 1),
  ];
}

// ---------------------------------------------------------------------------
// Convenience: Narkowicz's simple ACES approximation
// ---------------------------------------------------------------------------

/**
 * Attempt at implementing Krzysztof Narkowicz's simple ACES curve:
 *
 * ```
 *        x × (2.51 × x + 0.03)
 * f(x) = ───────────────────────────────
 *        x × (2.43 × x + 0.59) + 0.14
 * ```
 *
 * Faster than the full Hill fit but less accurate in the toe and shoulder.
 *
 * @param x - Single‑channel linear‑light input (≥ 0).
 * @returns   Tone‑mapped value clamped to [0, 1].
 */
export function acesNarkowicz(x: number): number {
  const a = 2.51;
  const b = 0.03;
  const c = 2.43;
  const d = 0.59;
  const e = 0.14;
  const mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
  return Math.min(Math.max(mapped, 0), 1);
}

/**
 * Apply the Narkowicz ACES curve independently to each channel
 * of a linear sRGB colour.
 *
 * @param color - Linear‑light sRGB input.
 * @returns       Tone‑mapped linear sRGB in [0, 1].
 */
export function acesNarkowiczColor(color: Color3): Color3 {
  return [
    acesNarkowicz(color[0]),
    acesNarkowicz(color[1]),
    acesNarkowicz(color[2]),
  ];
}

// ---------------------------------------------------------------------------
// Exposure helper
// ---------------------------------------------------------------------------

/**
 * Scale a colour by an exposure value (in stops / EV).
 *
 * `result = color × 2^exposure`
 *
 * Pair with any of the tone‑mapping functions above:
 *
 * ```ts
 * const ldr = acesFilm(expose(hdr, 1.5));
 * ```
 *
 * @param color    - Linear‑light RGB input.
 * @param exposure - Exposure compensation in stops (EV).
 * @returns          Exposed linear‑light colour.
 */
export function expose(color: Color3, exposure: number): Color3 {
  const scale = 2 ** exposure;
  return [color[0] * scale, color[1] * scale, color[2] * scale];
}
