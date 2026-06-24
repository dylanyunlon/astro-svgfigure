/**
 * src/lib/sph/velocity-field-texture.ts  —  M764
 *
 * SPH Particle Velocity Field → GPU Texture Rasterizer
 * ─────────────────────────────────────────────────────────────────────────────
 * Converts a 2D SPH particle velocity field into a regular-grid RG texture
 * (Float32Array, 2 channels per texel: vx, vy) suitable for GPU upload.
 *
 * Whereas flowmap-bridge.ts uses simple nearest-cell splatting (O(N), no
 * smoothing), this module performs kernel-weighted SPH interpolation using the
 * Poly6 kernel — producing physically continuous, smooth velocity fields that
 * exactly match the reconstruction quality of density-field-texture.ts (M763).
 *
 * The texture feeds downstream rendering stages:
 *
 *   • fluid-surface-mesh.ts  — velocity-driven anisotropic iso-contours
 *   • at-water-surface.ts    — ATWaterSurface flow-driven wave direction
 *   • flowmap-bridge.ts      — drop-in replacement for the simple splat path
 *   • any custom fragment shader reading a vec2 velocity field (e.g.
 *     Turing-pattern UV distortion, particle trail advection, LIC rendering)
 *
 * ─── Algorithm ──────────────────────────────────────────────────────────────
 *
 *   For each texel centre (wx, wy) on the output grid, accumulate weighted
 *   velocity contributions from nearby SPH particles:
 *
 *       v(x) = Σⱼ (mⱼ / ρⱼ) · vⱼ · W_poly6( ‖x − xⱼ‖, h )
 *
 *   When per-particle densities are not available, the simplified form is:
 *
 *       v(x) = Σⱼ mⱼ · vⱼ · W_poly6( ‖x − xⱼ‖, h )  /  Σⱼ mⱼ · W_poly6(…)
 *
 *   This Shepard-normalised variant (weight = kernel / total kernel) ensures
 *   the output is a proper weighted average of velocities, regardless of
 *   local particle count.  It avoids the magnitude blow-up that the raw
 *   summation form would produce in high-density regions.
 *
 *   Neighbor search is accelerated via SpatialHashGrid (cell-linked list,
 *   3×3 cell queries, zero GC).  Cost: O(texels × avg-neighbors) rather
 *   than O(texels × particles).
 *
 * ─── Output Layout ──────────────────────────────────────────────────────────
 *
 *   The rasterize() method returns a row-major Float32Array of length
 *   resX × resY × 2, where pairs (field[i*2], field[i*2+1]) store the
 *   interpolated (vx, vy) at that texel.
 *
 *   For GPU upload (WebGPU rg32float):
 *
 *     device.queue.writeTexture(
 *       { texture: velocityGPUTexture },
 *       velocityField.rasterize(particles, worldW, worldH, resolution),
 *       { bytesPerRow: resolution * 2 * 4 },
 *       { width: resolution, height: resolution },
 *     );
 *
 *   For WebGL (gl.RG, gl.FLOAT):
 *
 *     gl.texSubImage2D(
 *       gl.TEXTURE_2D, 0, 0, 0,
 *       resolution, resolution, gl.RG, gl.FLOAT, velocityTex
 *     );
 *
 * ─── Coordinate Convention ──────────────────────────────────────────────────
 *
 *   World space: (0, 0) at top-left, X right, Y down.
 *   Texel (ix, iy) maps to world position:
 *     wx = (ix + 0.5) * (worldW / resX)
 *     wy = (iy + 0.5) * (worldH / resY)
 *
 * Research: xiaodi #M764 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────




import { poly6W } from './sph-kernels';
import { SpatialHashGrid } from './SpatialHashGrid';

export interface VelocityFieldTextureConfig {
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
 * Minimal particle source for velocity field rasterization.
 * Compatible with ParticleData from types.ts and any SOA layout that
 * exposes x/y/vx/vy Float32Arrays + count.
 */
export interface VelocityParticleSource {
  /** Particle X positions. */
  x: Float32Array;
  /** Particle Y positions. */
  y: Float32Array;
  /** Particle X velocities. */
  vx: Float32Array;
  /** Particle Y velocities. */
  vy: Float32Array;
  /** Number of active particles (may be < x.length). */
  count: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// VelocityFieldTexture
// ─────────────────────────────────────────────────────────────────────────────

export class VelocityFieldTexture {
  // ── Configuration ────────────────────────────────────────────────────────
  private readonly smoothingRadius: number;
  private readonly particleMass: number;
  private readonly defaultResolution: number;

  // ── Internal state ───────────────────────────────────────────────────────
  private readonly hash: SpatialHashGrid;

  /**
   * Cached output buffer.  Re-allocated only when the resolution changes.
   * Layout: [vx₀, vy₀, vx₁, vy₁, …] row-major, 2 floats per texel.
   */
  private _buffer: Float32Array | null = null;
  private _bufferRes: number = 0;  // resX * resY of current _buffer

  // ── Stats from last rasterize() call ─────────────────────────────────────
  private _lastMinSpeed: number = 0;
  private _lastMaxSpeed: number = 0;
  private _lastMeanSpeed: number = 0;

  // ── Constructor ──────────────────────────────────────────────────────────

  constructor(cfg: VelocityFieldTextureConfig) {
    this.smoothingRadius = cfg.smoothingRadius;
    this.particleMass = cfg.particleMass ?? 1.0;
    this.defaultResolution = cfg.defaultResolution ?? 128;
    this.hash = new SpatialHashGrid(cfg.hashTableSize ?? 131072);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Rasterize the SPH particle velocity field onto a 2D grid.
   *
   * Uses Shepard-normalised Poly6 kernel interpolation:
   *
   *   v(x) = Σⱼ mⱼ · vⱼ · W(‖x−xⱼ‖, h) / Σⱼ mⱼ · W(‖x−xⱼ‖, h)
   *
   * This produces a proper weighted average of particle velocities,
   * avoiding magnitude blow-up in high-density regions.
   *
   * @param particles  Particle positions, velocities, and count.
   * @param worldW     Simulation domain width in world units.
   * @param worldH     Simulation domain height in world units.
   * @param resolution Output grid size.  If a single number, produces a
   *                   square texture (resX = resY = resolution).  Pass
   *                   [resX, resY] for non-square grids.
   * @returns Row-major Float32Array of length resX × resY × 2.
   *          Each pair (field[i*2], field[i*2+1]) is the interpolated
   *          (vx, vy) at that texel centre.
   */
  rasterize(
    particles: VelocityParticleSource,
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
      field = new Float32Array(totalTexels * 2);
      this._buffer = field;
      this._bufferRes = totalTexels;
    }

    const { x: px, y: py, vx: pvx, vy: pvy, count: n } = particles;

    // Early-out when there are no particles
    if (n === 0) {
      this._lastMinSpeed = 0;
      this._lastMaxSpeed = 0;
      this._lastMeanSpeed = 0;
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
    // and accumulate Shepard-normalised Poly6 kernel contributions.
    // This mirrors the pattern used in density-field-texture.ts but extends
    // it to the 2-channel velocity vector.

    let minS = Infinity;
    let maxS = -Infinity;
    let sumS = 0;

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

        let sumVx = 0.0;
        let sumVy = 0.0;
        let sumW = 0.0;

        for (let cy = cyMin; cy <= cyMax; cy++) {
          for (let cx = cxMin; cx <= cxMax; cx++) {
            const bucket = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
            let j = head[bucket];
            while (j !== -1) {
              const dx = px[j] - wx;
              const dy = py[j] - wy;
              const r2 = dx * dx + dy * dy;
              if (r2 <= h2) {
                const w = mass * poly6W(Math.sqrt(r2), h);
                sumVx += w * pvx[j];
                sumVy += w * pvy[j];
                sumW += w;
              }
              j = next[j];
            }
          }
        }

        // Shepard normalisation: divide by total kernel weight
        const texIdx = (rowOff + ix) * 2;
        if (sumW > 1e-12) {
          const invW = 1.0 / sumW;
          field[texIdx] = sumVx * invW;
          field[texIdx + 1] = sumVy * invW;
        }
        // else: stays at 0,0 (no particles nearby)

        // Track speed statistics
        const vx = field[texIdx];
        const vy = field[texIdx + 1];
        const speed = Math.sqrt(vx * vx + vy * vy);
        if (speed < minS) minS = speed;
        if (speed > maxS) maxS = speed;
        sumS += speed;
      }
    }

    this._lastMinSpeed = minS === Infinity ? 0 : minS;
    this._lastMaxSpeed = maxS === -Infinity ? 0 : maxS;
    this._lastMeanSpeed = sumS / totalTexels;

    return field;
  }

  // ─── Convenience: rasterize from raw arrays ────────────────────────────

  /**
   * Rasterize from separate x/y/vx/vy arrays and count.
   * Equivalent to rasterize({ x, y, vx, vy, count }, worldW, worldH, resolution).
   */
  rasterizeRaw(
    px: Float32Array,
    py: Float32Array,
    pvx: Float32Array,
    pvy: Float32Array,
    count: number,
    worldW: number,
    worldH: number,
    resolution?: number | [number, number],
  ): Float32Array {
    return this.rasterize({ x: px, y: py, vx: pvx, vy: pvy, count }, worldW, worldH, resolution);
  }

  // ─── Statistics ────────────────────────────────────────────────────────

  /** Minimum speed (|v|) from the last rasterize() call. */
  get minSpeed(): number { return this._lastMinSpeed; }

  /** Maximum speed (|v|) from the last rasterize() call. */
  get maxSpeed(): number { return this._lastMaxSpeed; }

  /** Mean speed (|v|) from the last rasterize() call. */
  get meanSpeed(): number { return this._lastMeanSpeed; }

  // ─── Normalization helpers ─────────────────────────────────────────────

  /**
   * Return a normalised copy of the velocity field where each velocity
   * vector is divided by `maxSpeed`, mapping speeds to [0, 1] range.
   * Direction is preserved; magnitude is normalised.
   *
   * Useful for GPU upload as a normalised vector texture where downstream
   * shaders apply their own scaling factor.
   *
   * Returns null if the field has not been rasterized yet or maxSpeed is zero.
   */
  normalizedCopy(): Float32Array | null {
    if (!this._buffer || this._lastMaxSpeed <= 0) return null;

    const src = this._buffer;
    const dst = new Float32Array(src.length);
    const invMax = 1.0 / this._lastMaxSpeed;

    for (let i = 0; i < src.length; i++) {
      dst[i] = src[i] * invMax;
    }
    return dst;
  }

  /**
   * Normalise the velocity field in-place, dividing all components by
   * `maxSpeed`.  This mutates the buffer returned by the last rasterize() call.
   *
   * Returns false if the field is empty or maxSpeed is zero.
   */
  normalizeInPlace(): boolean {
    if (!this._buffer || this._lastMaxSpeed <= 0) return false;

    const invMax = 1.0 / this._lastMaxSpeed;
    const buf = this._buffer;
    for (let i = 0; i < buf.length; i++) {
      buf[i] *= invMax;
    }

    this._lastMinSpeed *= invMax;
    this._lastMeanSpeed *= invMax;
    this._lastMaxSpeed = 1.0;

    return true;
  }

  /**
   * Clamp and normalise velocity components to [−1, +1] using a given
   * `maxSpeed` reference.  Compatible with the normalisation convention
   * used by flowmap-bridge.ts, allowing drop-in replacement.
   *
   * @param maxSpeed  Reference speed; components are divided by this and
   *                  clamped to [−1, +1].  Default: this.maxSpeed.
   */
  clampNormalize(maxSpeed?: number): void {
    if (!this._buffer) return;

    const ms = maxSpeed ?? this._lastMaxSpeed;
    if (ms <= 0) return;

    const invMax = 1.0 / ms;
    const buf = this._buffer;
    for (let i = 0; i < buf.length; i++) {
      buf[i] = Math.max(-1, Math.min(1, buf[i] * invMax));
    }
  }

  // ─── Bilinear sampling ─────────────────────────────────────────────────

  /**
   * Sample the velocity field at an arbitrary world-space position using
   * bilinear interpolation.  Returns [0, 0] if the field has not been
   * rasterized.
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
  ): [number, number] {
    if (!this._buffer) return [0, 0];

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

    // Four corner indices (each texel has 2 floats)
    const i00 = (iy0 * resX + ix0) * 2;
    const i10 = (iy0 * resX + ix1) * 2;
    const i01 = (iy1 * resX + ix0) * 2;
    const i11 = (iy1 * resX + ix1) * 2;

    // Bilinear interpolation for vx and vy separately
    const vx =
      field[i00]     * (1 - tx) * (1 - ty) +
      field[i10]     * tx       * (1 - ty) +
      field[i01]     * (1 - tx) * ty       +
      field[i11]     * tx       * ty;

    const vy =
      field[i00 + 1] * (1 - tx) * (1 - ty) +
      field[i10 + 1] * tx       * (1 - ty) +
      field[i01 + 1] * (1 - tx) * ty       +
      field[i11 + 1] * tx       * ty;

    return [vx, vy];
  }

  /**
   * Sample the speed (|v|) at an arbitrary world-space position via
   * bilinear interpolation.
   */
  sampleSpeedAt(
    wx: number,
    wy: number,
    worldW: number,
    worldH: number,
    resX: number,
    resY?: number,
  ): number {
    const [vx, vy] = this.sampleAt(wx, wy, worldW, worldH, resX, resY);
    return Math.sqrt(vx * vx + vy * vy);
  }

  /**
   * Sample the flow angle (atan2(vy, vx)) at an arbitrary world-space
   * position via bilinear interpolation.  Returns 0 if speed is negligible.
   */
  sampleAngleAt(
    wx: number,
    wy: number,
    worldW: number,
    worldH: number,
    resX: number,
    resY?: number,
  ): number {
    const [vx, vy] = this.sampleAt(wx, wy, worldW, worldH, resX, resY);
    const speed = Math.sqrt(vx * vx + vy * vy);
    return speed > 1e-6 ? Math.atan2(vy, vx) : 0;
  }

  // ─── Speed field extraction ────────────────────────────────────────────

  /**
   * Extract a scalar speed field (|v| at each texel) from the current
   * velocity buffer.  Useful for driving effects that only care about
   * flow magnitude, not direction.
   *
   * @returns Float32Array of length resX × resY, or null if not rasterized.
   */
  speedField(): Float32Array | null {
    if (!this._buffer) return null;

    const totalTexels = this._bufferRes;
    const out = new Float32Array(totalTexels);
    const buf = this._buffer;

    for (let i = 0; i < totalTexels; i++) {
      const vx = buf[i * 2];
      const vy = buf[i * 2 + 1];
      out[i] = Math.sqrt(vx * vx + vy * vy);
    }
    return out;
  }

  // ─── Static factory helpers ────────────────────────────────────────────

  /**
   * Create a VelocityFieldTexture sized to match a WorldConfig.
   */
  static fromWorldConfig(
    worldCfg: { smoothingRadius: number },
    opts?: Omit<VelocityFieldTextureConfig, 'smoothingRadius'>,
  ): VelocityFieldTexture {
    return new VelocityFieldTexture({
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
 * Validates the velocity field texture rasterizer:
 *
 *  1. Empty particles → zero field
 *  2. Single particle with velocity → non-zero interpolated velocity at centre
 *  3. Velocity direction preserved: uniform rightward flow → all vx > 0, vy ≈ 0
 *  4. Symmetry: particle at centre → symmetric speed falloff
 *  5. Two particles, opposite velocities → near-zero velocity at midpoint
 *  6. Speed statistics: min ≤ mean ≤ max
 *  7. Normalisation: normalizedCopy scales by maxSpeed, peak speed ≈ 1
 *  8. Buffer reuse: consecutive rasterize() calls at same resolution reuse buffer
 *  9. Non-square resolution: [resX, resY] produces correct-size output
 * 10. Bilinear sampling: sampleAt at grid vertex matches field value
 *
 * Returns true when all checks pass; false (with console.error) on failure.
 */
export function selfTest(): boolean {
  const TOL = 1e-4;

  function fail(msg: string): false {
    console.error(`[velocity-field-texture selfTest] FAILED: ${msg}`);
    return false;
  }

  const h = 12;
  const mass = 1.0;
  const worldW = 200;
  const worldH = 200;
  const res = 32;

  const vft = new VelocityFieldTexture({
    smoothingRadius: h,
    particleMass: mass,
  });

  // ── Test 1: Empty field ──────────────────────────────────────────────────
  {
    const field = vft.rasterize(
      { x: new Float32Array(0), y: new Float32Array(0), vx: new Float32Array(0), vy: new Float32Array(0), count: 0 },
      worldW, worldH, res,
    );
    if (field.length !== res * res * 2)
      return fail(`Test 1: expected ${res * res * 2} floats, got ${field.length}`);
    for (let i = 0; i < field.length; i++) {
      if (field[i] !== 0)
        return fail(`Test 1: element ${i} = ${field[i]}, expected 0`);
    }
    if (vft.maxSpeed !== 0)
      return fail(`Test 1: maxSpeed = ${vft.maxSpeed}, expected 0`);
  }

  // ── Test 2: Single particle with velocity → non-zero at centre ──────────
  {
    const cx = worldW / 2;
    const cy = worldH / 2;
    const velX = 5.0;
    const velY = -3.0;
    const field = vft.rasterize(
      {
        x: new Float32Array([cx]),
        y: new Float32Array([cy]),
        vx: new Float32Array([velX]),
        vy: new Float32Array([velY]),
        count: 1,
      },
      worldW, worldH, res,
    );

    // The texel closest to the centre should have a velocity close to the particle's
    const centreIx = Math.floor(cx / (worldW / res));
    const centreIy = Math.floor(cy / (worldH / res));
    const idx = (centreIy * res + centreIx) * 2;
    const centreVx = field[idx];
    const centreVy = field[idx + 1];

    if (Math.abs(centreVx - velX) > Math.abs(velX) * 0.2)
      return fail(`Test 2: centre vx = ${centreVx}, expected ≈ ${velX}`);
    if (Math.abs(centreVy - velY) > Math.abs(velY) * 0.2)
      return fail(`Test 2: centre vy = ${centreVy}, expected ≈ ${velY}`);
  }

  // ── Test 3: Uniform rightward flow → all vx > 0, vy ≈ 0 in covered area
  {
    const nParts = 200;
    const pxArr = new Float32Array(nParts);
    const pyArr = new Float32Array(nParts);
    const vxArr = new Float32Array(nParts);
    const vyArr = new Float32Array(nParts);

    // Fill domain centre with particles moving rightward
    for (let i = 0; i < nParts; i++) {
      pxArr[i] = 60 + Math.random() * 80;  // [60, 140]
      pyArr[i] = 60 + Math.random() * 80;  // [60, 140]
      vxArr[i] = 10.0;
      vyArr[i] = 0.0;
    }

    const field = vft.rasterize(
      { x: pxArr, y: pyArr, vx: vxArr, vy: vyArr, count: nParts },
      worldW, worldH, res,
    );

    // Check texels in the central region: vx should be positive, vy ≈ 0
    const lo = Math.floor(res * 0.35);
    const hi = Math.floor(res * 0.65);
    for (let iy = lo; iy <= hi; iy++) {
      for (let ix = lo; ix <= hi; ix++) {
        const ti = (iy * res + ix) * 2;
        if (field[ti] <= 0)
          return fail(`Test 3: vx at (${ix},${iy}) = ${field[ti]}, expected > 0`);
        if (Math.abs(field[ti + 1]) > 1.0)
          return fail(`Test 3: vy at (${ix},${iy}) = ${field[ti + 1]}, expected ≈ 0`);
      }
    }
  }

  // ── Test 4: Symmetry: single particle → symmetric speed falloff ─────────
  {
    const cx = worldW / 2;
    const cy = worldH / 2;
    const field = vft.rasterize(
      {
        x: new Float32Array([cx]),
        y: new Float32Array([cy]),
        vx: new Float32Array([10]),
        vy: new Float32Array([0]),
        count: 1,
      },
      worldW, worldH, res,
    );

    // Speed should be roughly symmetric vertically around the centre
    const mid = Math.floor(res / 2);
    for (let dy = 1; dy < Math.min(4, mid); dy++) {
      const idxUp = ((mid - dy) * res + mid) * 2;
      const idxDn = ((mid + dy) * res + mid) * 2;
      const speedUp = Math.sqrt(field[idxUp] ** 2 + field[idxUp + 1] ** 2);
      const speedDn = Math.sqrt(field[idxDn] ** 2 + field[idxDn + 1] ** 2);
      const maxVal = Math.max(speedUp, speedDn, 1e-6);
      if (Math.abs(speedUp - speedDn) / maxVal > 0.2)
        return fail(`Test 4: asymmetry at dy=${dy}: up=${speedUp.toFixed(4)}, down=${speedDn.toFixed(4)}`);
    }
  }

  // ── Test 5: Two particles, opposite velocities → cancellation at midpoint
  {
    const p1x = 90, p1y = 100;
    const p2x = 110, p2y = 100;
    const field = vft.rasterize(
      {
        x: new Float32Array([p1x, p2x]),
        y: new Float32Array([p1y, p2y]),
        vx: new Float32Array([10, -10]),
        vy: new Float32Array([0, 0]),
        count: 2,
      },
      worldW, worldH, res,
    );

    // The midpoint texel should have near-zero vx (opposing velocities cancel)
    const midIx = Math.floor(100 / (worldW / res));
    const midIy = Math.floor(100 / (worldH / res));
    const midIdx = (midIy * res + midIx) * 2;
    const midVx = Math.abs(field[midIdx]);
    if (midVx > 2.0)
      return fail(`Test 5: midpoint vx = ${midVx}, expected ≈ 0 (cancellation)`);
  }

  // ── Test 6: Statistics: min ≤ mean ≤ max ─────────────────────────────────
  {
    vft.rasterize(
      {
        x: new Float32Array([100]),
        y: new Float32Array([100]),
        vx: new Float32Array([5]),
        vy: new Float32Array([3]),
        count: 1,
      },
      worldW, worldH, res,
    );

    if (vft.minSpeed > vft.meanSpeed + TOL)
      return fail(`Test 6: min ${vft.minSpeed} > mean ${vft.meanSpeed}`);
    if (vft.meanSpeed > vft.maxSpeed + TOL)
      return fail(`Test 6: mean ${vft.meanSpeed} > max ${vft.maxSpeed}`);
  }

  // ── Test 7: Normalised copy has peak speed ≈ 1, values in reasonable range
  {
    vft.rasterize(
      {
        x: new Float32Array([100]),
        y: new Float32Array([100]),
        vx: new Float32Array([8]),
        vy: new Float32Array([6]),
        count: 1,
      },
      worldW, worldH, res,
    );

    const norm = vft.normalizedCopy();
    if (!norm)
      return fail('Test 7: normalizedCopy() returned null');

    let nMaxSpeed = -Infinity;
    for (let i = 0; i < norm.length / 2; i++) {
      const s = Math.sqrt(norm[i * 2] ** 2 + norm[i * 2 + 1] ** 2);
      if (s > nMaxSpeed) nMaxSpeed = s;
    }
    if (Math.abs(nMaxSpeed - 1.0) > 0.05)
      return fail(`Test 7: normalised peak speed = ${nMaxSpeed}, expected ≈ 1.0`);
  }

  // ── Test 8: Buffer reuse ─────────────────────────────────────────────────
  {
    const f1 = vft.rasterize(
      {
        x: new Float32Array([50]),
        y: new Float32Array([50]),
        vx: new Float32Array([1]),
        vy: new Float32Array([0]),
        count: 1,
      },
      worldW, worldH, res,
    );
    const f2 = vft.rasterize(
      {
        x: new Float32Array([150]),
        y: new Float32Array([150]),
        vx: new Float32Array([0]),
        vy: new Float32Array([1]),
        count: 1,
      },
      worldW, worldH, res,
    );

    // Same reference means buffer was reused
    if (f1 !== f2)
      return fail('Test 8: expected buffer reuse (same reference)');

    // Content should reflect the second particle
    const ix150 = Math.floor(150 / (worldW / res));
    const iy150 = Math.floor(150 / (worldH / res));
    const idx = (iy150 * res + ix150) * 2;
    if (f2[idx + 1] <= 0)
      return fail('Test 8: reused buffer has wrong content');
  }

  // ── Test 9: Non-square resolution ───────────────────────────────────────
  {
    const resX = 64, resY = 32;
    const field = vft.rasterize(
      {
        x: new Float32Array([100]),
        y: new Float32Array([100]),
        vx: new Float32Array([1]),
        vy: new Float32Array([1]),
        count: 1,
      },
      worldW, worldH, [resX, resY],
    );
    if (field.length !== resX * resY * 2)
      return fail(`Test 9: expected ${resX * resY * 2} floats, got ${field.length}`);
  }

  // ── Test 10: Bilinear sampling matches grid vertex ──────────────────────
  {
    const field = vft.rasterize(
      {
        x: new Float32Array([worldW / 2]),
        y: new Float32Array([worldH / 2]),
        vx: new Float32Array([7]),
        vy: new Float32Array([-4]),
        count: 1,
      },
      worldW, worldH, res,
    );

    // Sample at the centre of texel (res/2, res/2)
    const tW = worldW / res;
    const tH = worldH / res;
    const half = Math.floor(res / 2);
    const sampleX = (half + 0.5) * tW;
    const sampleY = (half + 0.5) * tH;
    const [svx, svy] = vft.sampleAt(sampleX, sampleY, worldW, worldH, res);
    const idx = (half * res + half) * 2;
    const directVx = field[idx];
    const directVy = field[idx + 1];

    if (Math.abs(svx - directVx) > TOL * Math.max(1, Math.abs(directVx)))
      return fail(`Test 10: sampleAt vx = ${svx}, direct = ${directVx}`);
    if (Math.abs(svy - directVy) > TOL * Math.max(1, Math.abs(directVy)))
      return fail(`Test 10: sampleAt vy = ${svy}, direct = ${directVy}`);
  }

  return true;
}
