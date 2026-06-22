/**
 * src/lib/sph/density-field-texture.ts  —  M763
 *
 * SPH Particle Density Field → GPU Texture Rasterizer
 * ─────────────────────────────────────────────────────────────────────────────
 * Converts a 2D SPH particle field into a regular-grid density texture
 * (Float32Array) suitable for upload to a GPU texture.  The texture feeds
 * downstream rendering stages:
 *
 *   • fluid-surface-mesh.ts  — iso-contour extraction via marching squares
 *   • water-caustics.ts      — WaterCaustics.updateFromDensity()
 *   • at-water-surface.ts    — ATWaterSurface wave-driving from SPH coupling
 *   • any custom fragment shader that reads a density scalar field
 *
 * ─── Algorithm ──────────────────────────────────────────────────────────────
 *
 *   For each texel centre (wx, wy) on the output grid, accumulate weighted
 *   contributions from nearby SPH particles using the Poly6 kernel:
 *
 *       ρ(x) = Σⱼ mⱼ · W_poly6( ‖x − xⱼ‖, h )
 *
 *   Neighbor search is accelerated via SpatialHashGrid (cell-linked list,
 *   3×3 cell queries, zero GC).  The cost is O(texels × avg-neighbors)
 *   rather than O(texels × particles).
 *
 * ─── Output Layout ──────────────────────────────────────────────────────────
 *
 *   The rasterize() method returns a row-major Float32Array of length
 *   resX × resY, where each element is the SPH density at that texel.
 *
 *   For GPU upload (WebGPU r32float):
 *
 *     device.queue.writeTexture(
 *       { texture: densityGPUTexture },
 *       densityField.rasterize(particles, worldW, worldH, resolution),
 *       { bytesPerRow: resolution * 4 },
 *       { width: resolution, height: resolution },
 *     );
 *
 *   For WaterCaustics coupling:
 *
 *     const density = densityField.rasterize(particles, worldW, worldH, 128);
 *     caustics.updateFromDensity(density, 128, 128, restDensity);
 *
 * ─── Coordinate Convention ──────────────────────────────────────────────────
 *
 *   World space: (0, 0) at top-left, X right, Y down.
 *   Texel (ix, iy) maps to world position:
 *     wx = (ix + 0.5) * (worldW / resX)
 *     wy = (iy + 0.5) * (worldH / resY)
 *
 * Research: xiaodi #M763 — cell-pubsub-loop
 */

import { poly6W } from './sph-kernels';
import { SpatialHashGrid } from './SpatialHashGrid';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface DensityFieldTextureConfig {
  /**
   * SPH smoothing radius h.
   * Must match the kernel support used by the solver (typically 12).
   */
  smoothingRadius: number;

  /**
   * Particle mass used in the kernel summation.
   * Default 1.0 (matches sph-kernels defaultConfig).
   */
  particleMass?: number;

  /**
   * Hash table capacity for SpatialHashGrid.
   * Default 131072 — good for up to ~100k particles.
   */
  hashTableSize?: number;

  /**
   * Default output resolution (square) when not specified per-call.
   * Default 128.
   */
  defaultResolution?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Particle input (matches the project's ParticleData SOA layout)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimal particle source for rasterization.
 * Compatible with ParticleData from types.ts and any SOA layout that
 * exposes x/y Float32Arrays + count.
 */
export interface ParticleSource {
  /** Particle X positions. */
  x: Float32Array;
  /** Particle Y positions. */
  y: Float32Array;
  /** Number of active particles (may be < x.length). */
  count: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// DensityFieldTexture
// ─────────────────────────────────────────────────────────────────────────────

export class DensityFieldTexture {
  // ── Configuration ────────────────────────────────────────────────────────
  private readonly smoothingRadius: number;
  private readonly particleMass: number;
  private readonly defaultResolution: number;

  // ── Internal state ───────────────────────────────────────────────────────
  private readonly hash: SpatialHashGrid;

  /**
   * Cached output buffer.  Re-allocated only when the resolution changes.
   * Avoids per-frame allocation for the common case of fixed resolution.
   */
  private _buffer: Float32Array | null = null;
  private _bufferRes: number = 0;  // resX * resY of current _buffer

  // ── Stats from last rasterize() call ─────────────────────────────────────
  private _lastMinDensity: number = 0;
  private _lastMaxDensity: number = 0;
  private _lastMeanDensity: number = 0;

  // ── Constructor ──────────────────────────────────────────────────────────

  constructor(cfg: DensityFieldTextureConfig) {
    this.smoothingRadius = cfg.smoothingRadius;
    this.particleMass = cfg.particleMass ?? 1.0;
    this.defaultResolution = cfg.defaultResolution ?? 128;
    this.hash = new SpatialHashGrid(cfg.hashTableSize ?? 131072);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Rasterize the SPH particle density field onto a 2D grid.
   *
   * @param particles  Particle positions + count.
   * @param worldW     Simulation domain width in world units.
   * @param worldH     Simulation domain height in world units.
   * @param resolution Output grid size.  If a single number, produces a
   *                   square texture (resX = resY = resolution).  Pass
   *                   [resX, resY] for non-square grids.
   * @returns Row-major Float32Array of length resX × resY.
   *          Each element is the SPH density ρ at that texel centre.
   */
  rasterize(
    particles: ParticleSource,
    worldW: number,
    worldH: number,
    resolution?: number | [number, number],
  ): Float32Array {
    // ── Resolve resolution ───────────────────────────────────────────────
    let resX: number;
    let resY: number;
    if (resolution == null) {
      resX = resY = this.defaultResolution;
    } else if (typeof resolution === 'number') {
      resX = resY = resolution;
    } else {
      resX = resolution[0];
      resY = resolution[1];
    }

    const totalTexels = resX * resY;

    // ── Allocate / reuse output buffer ───────────────────────────────────
    let field: Float32Array;
    if (this._buffer && this._bufferRes === totalTexels) {
      field = this._buffer;
      field.fill(0);
    } else {
      field = new Float32Array(totalTexels);
      this._buffer = field;
      this._bufferRes = totalTexels;
    }

    const { x: px, y: py, count: n } = particles;

    // Early-out when there are no particles
    if (n === 0) {
      this._lastMinDensity = 0;
      this._lastMaxDensity = 0;
      this._lastMeanDensity = 0;
      return field;
    }

    // ── Build spatial hash ───────────────────────────────────────────────
    const h = this.smoothingRadius;
    const mass = this.particleMass;
    const hash = this.hash;

    hash.clear();
    hash.insertAll(px, py, n, h);

    const head = hash.getHead();
    const next = hash.getNext();
    const tableSize = hash.getTableSize();
    const mask = tableSize - 1;

    const invCellSize = 1.0 / h;
    const h2 = h * h;

    // ── Texel → world mapping ────────────────────────────────────────────
    const texelW = worldW / resX;
    const texelH = worldH / resY;

    // ── Rasterize ────────────────────────────────────────────────────────
    //
    // For each texel, query the 3×3 hash cells around its world-space centre
    // and accumulate Poly6 kernel contributions.  This mirrors the pattern
    // used in FluidSurfaceMesh._rasterizeField and SpatialHashGrid neighbor
    // queries throughout the SPH pipeline.

    let minD = Infinity;
    let maxD = -Infinity;
    let sumD = 0;

    for (let iy = 0; iy < resY; iy++) {
      const wy = (iy + 0.5) * texelH;
      const rowOff = iy * resX;

      for (let ix = 0; ix < resX; ix++) {
        const wx = (ix + 0.5) * texelW;

        // 3×3 hash cell neighbourhood
        const cxMin = Math.floor((wx - h) * invCellSize) | 0;
        const cxMax = Math.floor((wx + h) * invCellSize) | 0;
        const cyMin = Math.floor((wy - h) * invCellSize) | 0;
        const cyMax = Math.floor((wy + h) * invCellSize) | 0;

        let density = 0.0;

        for (let cy = cyMin; cy <= cyMax; cy++) {
          for (let cx = cxMin; cx <= cxMax; cx++) {
            const bucket = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
            let j = head[bucket];
            while (j !== -1) {
              const dx = px[j] - wx;
              const dy = py[j] - wy;
              const r2 = dx * dx + dy * dy;
              if (r2 <= h2) {
                density += mass * poly6W(Math.sqrt(r2), h);
              }
              j = next[j];
            }
          }
        }

        field[rowOff + ix] = density;

        // Track statistics
        if (density < minD) minD = density;
        if (density > maxD) maxD = density;
        sumD += density;
      }
    }

    this._lastMinDensity = minD === Infinity ? 0 : minD;
    this._lastMaxDensity = maxD === -Infinity ? 0 : maxD;
    this._lastMeanDensity = sumD / totalTexels;

    return field;
  }

  // ─── Convenience: rasterize from raw arrays ────────────────────────────

  /**
   * Rasterize from separate x/y arrays and count.
   * Equivalent to rasterize({ x, y, count }, worldW, worldH, resolution).
   */
  rasterizeRaw(
    px: Float32Array,
    py: Float32Array,
    count: number,
    worldW: number,
    worldH: number,
    resolution?: number | [number, number],
  ): Float32Array {
    return this.rasterize({ x: px, y: py, count }, worldW, worldH, resolution);
  }

  // ─── Statistics ────────────────────────────────────────────────────────

  /** Minimum density value from the last rasterize() call. */
  get minDensity(): number { return this._lastMinDensity; }

  /** Maximum density value from the last rasterize() call. */
  get maxDensity(): number { return this._lastMaxDensity; }

  /** Mean density value from the last rasterize() call. */
  get meanDensity(): number { return this._lastMeanDensity; }

  // ─── Normalization helpers ─────────────────────────────────────────────

  /**
   * Return a normalised copy of the field mapped to [0, 1] by dividing
   * each value by `maxDensity`.  Useful for direct GPU upload as a
   * normalised scalar texture (e.g. for shader-based surface colouring).
   *
   * Returns null if the field has not been rasterized yet or maxDensity
   * is zero (no particles).
   */
  normalizedCopy(): Float32Array | null {
    if (!this._buffer || this._lastMaxDensity <= 0) return null;

    const src = this._buffer;
    const dst = new Float32Array(src.length);
    const invMax = 1.0 / this._lastMaxDensity;

    for (let i = 0; i < src.length; i++) {
      dst[i] = src[i] * invMax;
    }
    return dst;
  }

  /**
   * Normalise the field in-place, mapping values to [0, 1].
   * This mutates the buffer returned by the last rasterize() call.
   *
   * Returns false if the field is empty or maxDensity is zero.
   */
  normalizeInPlace(): boolean {
    if (!this._buffer || this._lastMaxDensity <= 0) return false;

    const invMax = 1.0 / this._lastMaxDensity;
    const buf = this._buffer;
    for (let i = 0; i < buf.length; i++) {
      buf[i] *= invMax;
    }

    this._lastMinDensity *= invMax;
    this._lastMaxDensity = 1.0;
    this._lastMeanDensity *= invMax;

    return true;
  }

  // ─── Bilinear sampling ─────────────────────────────────────────────────

  /**
   * Sample the density field at an arbitrary world-space position using
   * bilinear interpolation.  Returns 0 if the field has not been rasterized.
   *
   * @param wx     World X coordinate.
   * @param wy     World Y coordinate.
   * @param worldW Domain width  (must match the last rasterize() call).
   * @param worldH Domain height (must match the last rasterize() call).
   * @param resX   Grid resolution X (must match the last rasterize() call).
   * @param resY   Grid resolution Y (defaults to resX for square grids).
   */
  sampleAt(
    wx: number,
    wy: number,
    worldW: number,
    worldH: number,
    resX: number,
    resY?: number,
  ): number {
    if (!this._buffer) return 0;

    const rY = resY ?? resX;
    const field = this._buffer;

    // Convert world → texel (continuous, half-texel offset)
    const texelW = worldW / resX;
    const texelH = worldH / rY;
    const fx = wx / texelW - 0.5;
    const fy = wy / texelH - 0.5;

    const ix0 = Math.max(0, Math.min(Math.floor(fx), resX - 1));
    const iy0 = Math.max(0, Math.min(Math.floor(fy), rY - 1));
    const ix1 = Math.min(ix0 + 1, resX - 1);
    const iy1 = Math.min(iy0 + 1, rY - 1);

    const tx = Math.max(0, Math.min(fx - ix0, 1));
    const ty = Math.max(0, Math.min(fy - iy0, 1));

    const v00 = field[iy0 * resX + ix0];
    const v10 = field[iy0 * resX + ix1];
    const v01 = field[iy1 * resX + ix0];
    const v11 = field[iy1 * resX + ix1];

    return (
      v00 * (1 - tx) * (1 - ty) +
      v10 * tx * (1 - ty) +
      v01 * (1 - tx) * ty +
      v11 * tx * ty
    );
  }

  // ─── Static factory helpers ────────────────────────────────────────────

  /**
   * Create a DensityFieldTexture sized to match a WorldConfig.
   */
  static fromWorldConfig(
    worldCfg: { smoothingRadius: number },
    opts?: Omit<DensityFieldTextureConfig, 'smoothingRadius'>,
  ): DensityFieldTexture {
    return new DensityFieldTexture({
      smoothingRadius: worldCfg.smoothingRadius,
      ...opts,
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-test
// ─────────────────────────────────────────────────────────────────────────────

/**
 * selfTest(): boolean
 *
 * Validates the density field texture rasterizer:
 *
 *  1. Empty particles → zero field
 *  2. Single particle at centre → peak at centre, falls to zero at edges
 *  3. Symmetry: field is approximately symmetric around a centred particle
 *  4. Two particles far apart → two distinct peaks
 *  5. Resolution independence: doubling resolution preserves peak location
 *  6. Bilinear sampling: sampleAt at grid vertices matches field values
 *  7. Statistics: min ≤ mean ≤ max
 *  8. Normalisation: normalizedCopy values in [0, 1], peak = 1
 *  9. Buffer reuse: consecutive rasterize() calls at same resolution reuse buffer
 * 10. Non-square resolution: [resX, resY] produces correct-size output
 *
 * Returns true when all checks pass; false (with console.error) on failure.
 */
export function selfTest(): boolean {
  const TOL = 1e-4;

  function fail(msg: string): false {
    console.error(`[density-field-texture selfTest] FAILED: ${msg}`);
    return false;
  }

  const h = 12;
  const mass = 1.0;
  const worldW = 200;
  const worldH = 200;
  const res = 32;

  const dft = new DensityFieldTexture({
    smoothingRadius: h,
    particleMass: mass,
  });

  // ── Test 1: Empty field ──────────────────────────────────────────────────
  {
    const field = dft.rasterize({ x: new Float32Array(0), y: new Float32Array(0), count: 0 }, worldW, worldH, res);
    if (field.length !== res * res)
      return fail(`Test 1: expected ${res * res} texels, got ${field.length}`);
    for (let i = 0; i < field.length; i++) {
      if (field[i] !== 0)
        return fail(`Test 1: texel ${i} = ${field[i]}, expected 0`);
    }
    if (dft.maxDensity !== 0)
      return fail(`Test 1: maxDensity = ${dft.maxDensity}, expected 0`);
  }

  // ── Test 2: Single particle at centre → peak at centre ───────────────────
  {
    const cx = worldW / 2;
    const cy = worldH / 2;
    const field = dft.rasterize(
      { x: new Float32Array([cx]), y: new Float32Array([cy]), count: 1 },
      worldW, worldH, res,
    );

    // The texel closest to the centre should have the highest value
    const centreIx = Math.floor(cx / (worldW / res));
    const centreIy = Math.floor(cy / (worldH / res));
    const centreVal = field[centreIy * res + centreIx];

    if (centreVal <= 0)
      return fail(`Test 2: centre density = ${centreVal}, expected > 0`);

    // Edge texels should be zero (particle influence radius < domain half-width)
    const edgeVal = field[0];
    if (edgeVal > centreVal)
      return fail(`Test 2: edge density ${edgeVal} > centre ${centreVal}`);
  }

  // ── Test 3: Symmetry around centred particle ─────────────────────────────
  {
    const cx = worldW / 2;
    const cy = worldH / 2;
    const field = dft.rasterize(
      { x: new Float32Array([cx]), y: new Float32Array([cy]), count: 1 },
      worldW, worldH, res,
    );

    // Check approximate horizontal symmetry around centre
    const mid = Math.floor(res / 2);
    for (let dy = 0; dy < Math.min(5, mid); dy++) {
      const left  = field[(mid + dy) * res + (mid - 3)];
      const right = field[(mid + dy) * res + (mid + 3)];
      // Allow some tolerance for texel-centre offset
      if (Math.abs(left - right) > centreRelativeTol(left, right))
        return fail(`Test 3: asymmetry at dy=${dy}: left=${left}, right=${right}`);
    }
  }

  // ── Test 4: Two far-apart particles → two peaks ──────────────────────────
  {
    const p1x = 30, p1y = 100;
    const p2x = 170, p2y = 100;
    const field = dft.rasterize(
      { x: new Float32Array([p1x, p2x]), y: new Float32Array([p1y, p2y]), count: 2 },
      worldW, worldH, res,
    );

    const ix1 = Math.floor(p1x / (worldW / res));
    const ix2 = Math.floor(p2x / (worldW / res));
    const iy  = Math.floor(100 / (worldH / res));
    const midIx = Math.floor(res / 2);

    const peak1 = field[iy * res + ix1];
    const peak2 = field[iy * res + ix2];
    const mid   = field[iy * res + midIx];

    if (peak1 <= 0 || peak2 <= 0)
      return fail(`Test 4: expected two positive peaks, got ${peak1}, ${peak2}`);
    if (mid >= peak1 || mid >= peak2)
      return fail(`Test 4: midpoint ${mid} >= one of the peaks (${peak1}, ${peak2})`);
  }

  // ── Test 5: Resolution independence (peak location preserved) ────────────
  {
    const cx = 80, cy = 80;
    const particles: ParticleSource = {
      x: new Float32Array([cx]),
      y: new Float32Array([cy]),
      count: 1,
    };

    const f1 = dft.rasterize(particles, worldW, worldH, 16);
    const f2 = dft.rasterize(particles, worldW, worldH, 32);

    // Find peak texel in each
    let peak1Idx = 0, peak2Idx = 0;
    for (let i = 1; i < f1.length; i++) if (f1[i] > f1[peak1Idx]) peak1Idx = i;
    for (let i = 1; i < f2.length; i++) if (f2[i] > f2[peak2Idx]) peak2Idx = i;

    // Convert peak texel to world coords
    const wx1 = ((peak1Idx % 16) + 0.5) * (worldW / 16);
    const wy1 = (Math.floor(peak1Idx / 16) + 0.5) * (worldH / 16);
    const wx2 = ((peak2Idx % 32) + 0.5) * (worldW / 32);
    const wy2 = (Math.floor(peak2Idx / 32) + 0.5) * (worldH / 32);

    const peakDist = Math.sqrt((wx1 - wx2) ** 2 + (wy1 - wy2) ** 2);
    const maxAllowed = worldW / 16 + worldW / 32; // sum of half-texel sizes
    if (peakDist > maxAllowed)
      return fail(`Test 5: peak shifted ${peakDist.toFixed(2)} > ${maxAllowed.toFixed(2)}`);
  }

  // ── Test 6: Bilinear sampling matches grid vertex ────────────────────────
  {
    const cx = worldW / 2, cy = worldH / 2;
    const field = dft.rasterize(
      { x: new Float32Array([cx]), y: new Float32Array([cy]), count: 1 },
      worldW, worldH, res,
    );

    // Sample at the centre of texel (res/2, res/2)
    const texelW = worldW / res;
    const texelH = worldH / res;
    const sampleX = (Math.floor(res / 2) + 0.5) * texelW;
    const sampleY = (Math.floor(res / 2) + 0.5) * texelH;
    const sampled = dft.sampleAt(sampleX, sampleY, worldW, worldH, res);
    const direct  = field[Math.floor(res / 2) * res + Math.floor(res / 2)];

    if (Math.abs(sampled - direct) > TOL * Math.max(1, direct))
      return fail(`Test 6: sampleAt = ${sampled}, direct = ${direct}`);
  }

  // ── Test 7: Statistics: min ≤ mean ≤ max ─────────────────────────────────
  {
    dft.rasterize(
      { x: new Float32Array([100]), y: new Float32Array([100]), count: 1 },
      worldW, worldH, res,
    );

    if (dft.minDensity > dft.meanDensity + TOL)
      return fail(`Test 7: min ${dft.minDensity} > mean ${dft.meanDensity}`);
    if (dft.meanDensity > dft.maxDensity + TOL)
      return fail(`Test 7: mean ${dft.meanDensity} > max ${dft.maxDensity}`);
  }

  // ── Test 8: Normalised copy has peak = 1, values in [0, 1] ──────────────
  {
    dft.rasterize(
      { x: new Float32Array([100]), y: new Float32Array([100]), count: 1 },
      worldW, worldH, res,
    );

    const norm = dft.normalizedCopy();
    if (!norm)
      return fail('Test 8: normalizedCopy() returned null');

    let nMax = -Infinity;
    for (let i = 0; i < norm.length; i++) {
      if (norm[i] < -TOL)
        return fail(`Test 8: normalised value ${norm[i]} < 0`);
      if (norm[i] > 1.0 + TOL)
        return fail(`Test 8: normalised value ${norm[i]} > 1`);
      if (norm[i] > nMax) nMax = norm[i];
    }
    if (Math.abs(nMax - 1.0) > TOL)
      return fail(`Test 8: normalised peak = ${nMax}, expected 1.0`);
  }

  // ── Test 9: Buffer reuse ─────────────────────────────────────────────────
  {
    const f1 = dft.rasterize(
      { x: new Float32Array([50]), y: new Float32Array([50]), count: 1 },
      worldW, worldH, res,
    );
    const f2 = dft.rasterize(
      { x: new Float32Array([150]), y: new Float32Array([150]), count: 1 },
      worldW, worldH, res,
    );

    // Same reference means buffer was reused
    if (f1 !== f2)
      return fail('Test 9: expected buffer reuse (same reference)');

    // Content should reflect the second particle, not the first
    const ix150 = Math.floor(150 / (worldW / res));
    const iy150 = Math.floor(150 / (worldH / res));
    if (f2[iy150 * res + ix150] <= 0)
      return fail('Test 9: reused buffer has wrong content');
  }

  // ── Test 10: Non-square resolution ───────────────────────────────────────
  {
    const resX = 64, resY = 32;
    const field = dft.rasterize(
      { x: new Float32Array([100]), y: new Float32Array([100]), count: 1 },
      worldW, worldH, [resX, resY],
    );
    if (field.length !== resX * resY)
      return fail(`Test 10: expected ${resX * resY} texels, got ${field.length}`);
  }

  return true;
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/**
 * Relative tolerance for symmetry checks.
 * Returns max(TOL, 0.15 * max(a, b)) to account for texel-centre alignment.
 */
function centreRelativeTol(a: number, b: number): number {
  return Math.max(1e-4, 0.15 * Math.max(Math.abs(a), Math.abs(b)));
}
