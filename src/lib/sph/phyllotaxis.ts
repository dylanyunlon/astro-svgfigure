/**
 * Phyllotaxis Spiral – Fibonacci Golden Angle Arrangement
 *
 * Models the natural packing pattern found in sunflowers, pinecones,
 * and other botanical structures.
 *
 * Formula:
 *   golden_angle θ = 137.50776405003785°  (≈ 360° / φ²)
 *   r(n)           = sqrt(n)               (uniform area distribution)
 *   theta(n)       = n * golden_angle      (cumulative angular offset)
 *
 * Cartesian:
 *   x(n) = r(n) * cos(theta(n))
 *   y(n) = r(n) * sin(theta(n))
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Golden angle in degrees: 360° × (2 − φ)  where φ = (1 + √5) / 2 */








export const GOLDEN_ANGLE_DEG = 137.50776405003785;

/** Golden angle in radians */
export const GOLDEN_ANGLE_RAD = GOLDEN_ANGLE_DEG * (Math.PI / 180);

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export interface PhyllotaxisPoint {
  /** 0-based index of the seed / floret */
  n: number;
  /** Radial distance from the centre: sqrt(n) */
  r: number;
  /** Cumulative angle in radians: n × golden_angle */
  theta: number;
  /** Cartesian x (unscaled) */
  x: number;
  /** Cartesian y (unscaled) */
  y: number;
}

export interface PhyllotaxisOptions {
  /**
   * Total number of points to generate.
   * @default 1000
   */
  count?: number;

  /**
   * Scale factor applied to the radial coordinate before converting to
   * Cartesian coordinates.  Useful for mapping the unit spiral into a
   * pixel-space canvas.
   * @default 1
   */
  scale?: number;

  /**
   * Offset added to the index before computing r so that the very first
   * seed (n = 0) is not placed at the origin.
   * @default 0
   */
  indexOffset?: number;
}

// ---------------------------------------------------------------------------
// Core generator
// ---------------------------------------------------------------------------

/**
 * Generate an array of phyllotaxis points using the golden-angle spiral.
 *
 * @param options  Configuration (all fields optional).
 * @returns        Array of {@link PhyllotaxisPoint} in index order.
 *
 * @example
 * ```ts
 * const pts = generatePhyllotaxis({ count: 500, scale: 6 });
 * pts.forEach(p => ctx.fillRect(p.x + cx, p.y + cy, 2, 2));
 * ```
 */
export function generatePhyllotaxis(
  options: PhyllotaxisOptions = {}
): PhyllotaxisPoint[] {
  const {
    count = 1000,
    scale = 1,
    indexOffset = 0,
  } = options;

  const points: PhyllotaxisPoint[] = [];

  for (let n = 0; n < count; n++) {
    const idx = n + indexOffset;
    const r = Math.sqrt(idx) * scale;
    const theta = idx * GOLDEN_ANGLE_RAD;
    points.push({
      n,
      r,
      theta,
      x: r * Math.cos(theta),
      y: r * Math.sin(theta),
    });
  }

  return points;
}

// ---------------------------------------------------------------------------
// Single-point helpers
// ---------------------------------------------------------------------------

/**
 * Compute the polar coordinates for a single seed index.
 *
 * @param n  Seed index (0-based).
 * @returns  `{ r, theta }` in unscaled units.
 */
export function polarAt(n: number): { r: number; theta: number } {
  return {
    r: Math.sqrt(n),
    theta: n * GOLDEN_ANGLE_RAD,
  };
}

/**
 * Compute the Cartesian coordinates for a single seed index.
 *
 * @param n      Seed index (0-based).
 * @param scale  Optional scale factor (default 1).
 * @returns      `{ x, y }`.
 */
export function cartesianAt(
  n: number,
  scale = 1
): { x: number; y: number } {
  const r = Math.sqrt(n) * scale;
  const theta = n * GOLDEN_ANGLE_RAD;
  return {
    x: r * Math.cos(theta),
    y: r * Math.sin(theta),
  };
}

// ---------------------------------------------------------------------------
// SVG utilities
// ---------------------------------------------------------------------------

export interface SvgCircleOptions {
  /** Radius of each dot circle element (px). @default 2 */
  dotRadius?: number;
  /** Fill colour for each dot. @default "currentColor" */
  fill?: string;
  /** SVG width.  The viewBox is centred on (0,0). @default 500 */
  width?: number;
  /** SVG height.  @default 500 */
  height?: number;
}

/**
 * Render a phyllotaxis spiral as an inline SVG string.
 *
 * @param phyllotaxisOptions  Options forwarded to {@link generatePhyllotaxis}.
 * @param svgOptions          Visual options for the SVG output.
 * @returns                   A complete `<svg>…</svg>` string.
 *
 * @example
 * ```ts
 * document.body.innerHTML = toSVG({ count: 800, scale: 8 });
 * ```
 */
export function toSVG(
  phyllotaxisOptions: PhyllotaxisOptions = {},
  svgOptions: SvgCircleOptions = {}
): string {
  const {
    dotRadius = 2,
    fill = "currentColor",
    width = 500,
    height = 500,
  } = svgOptions;

  const pts = generatePhyllotaxis(phyllotaxisOptions);
  const cx = width / 2;
  const cy = height / 2;

  const circles = pts
    .map(
      (p) =>
        `  <circle cx="${(p.x + cx).toFixed(3)}" cy="${(p.y + cy).toFixed(3)}" r="${dotRadius}" fill="${fill}"/>`
    )
    .join("\n");

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    circles,
    `</svg>`,
  ].join("\n");
}
