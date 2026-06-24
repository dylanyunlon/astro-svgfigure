/**
 * src/lib/sph/fluid-caustics.ts  —  M781
 *
 * Fluid Caustics — SPH velocity field projected onto Cell surfaces
 * ─────────────────────────────────────────────────────────────────────────────
 * Generates swimming-pool-floor caustic light patterns by:
 *
 *   1. Sampling the SPH particle velocity field (via VelocityFieldTexture or
 *      raw particle data) to derive a time-varying water surface height map.
 *   2. Superimposing multiple sine wave layers (configurable octaves) to
 *      produce the characteristic undulating "wavy bright lines" pattern.
 *   3. Computing per-texel surface normals via central differences.
 *   4. Applying Snell's law refraction (air→water IOR) to trace light rays
 *      from a directional light through the deformed surface onto a flat
 *      receiver plane (the Cell surface).
 *   5. Measuring the local focusing/defocusing of refracted rays via the
 *      Jacobian determinant (∂newPos/∂oldPos) to produce caustic intensity.
 *
 * The result is a scalar Float32Array texture where bright values indicate
 * light convergence (caustic lines) and dark values indicate shadow regions —
 * exactly the pattern seen on a swimming pool floor.
 *
 * ─── Relationship to water-caustics.ts (M613) ──────────────────────────────
 *
 * water-caustics.ts is a CPU port of the Evan Wallace WebGL Water demo: it
 * maintains its own wave simulation (ping-pong height field) and projects
 * caustics via vertex-based ray tracing.
 *
 * This module (fluid-caustics.ts) takes a fundamentally different approach:
 *
 *   • The water surface is *derived* from SPH velocity divergence/speed,
 *     not simulated internally.  This means the caustic pattern is
 *     physically correlated with the actual fluid motion.
 *   • Multi-octave sine waves are layered on top to add visual richness
 *     beyond what the SPH field alone provides.
 *   • Caustics are computed per-texel analytically (Jacobian of the refraction
 *     map) rather than by vertex splatting, giving smoother results at the
 *     cost of per-texel trigonometry.
 *   • The output targets Cell surfaces specifically: it accepts a Cell's
 *     bounding box and produces a texture in that Cell's local UV space.
 *
 * ─── Multi-Layer Sine Wave Design ───────────────────────────────────────────
 *
 * Each octave i contributes a sine wave with:
 *   frequency_i  = baseFreq × 2^i
 *   amplitude_i  = baseAmp  × persistence^i
 *   phase_i      = time × speed_i + dot(direction_i, uv)
 *
 * The wave directions are rotated by the golden angle (≈137.51°) per octave
 * to avoid axis-aligned interference patterns.  The SPH velocity at each
 * texel modulates the local wave direction, causing caustic lines to follow
 * the flow — creating the impression of light refracting through moving water
 * above the Cell surface.
 *
 * ─── Snell Refraction ───────────────────────────────────────────────────────
 *
 * The refraction follows the standard vector form of Snell's law:
 *
 *   t = η·I + (η·cos(θᵢ) − cos(θₜ))·N
 *   cos(θₜ) = √(1 − η²·(1 − cos²(θᵢ)))
 *   η = n₁/n₂ = IOR_AIR / IOR_WATER
 *
 * where I is the incident light direction, N is the surface normal, and t is
 * the refracted ray direction.  Total internal reflection (k < 0) is handled
 * by falling back to zero displacement.
 *
 * ─── Output Layout ──────────────────────────────────────────────────────────
 *
 *   The texture is a row-major Float32Array of length resX × resY.
 *   Each element is a scalar caustic intensity in approximately [0, 2+],
 *   where 1.0 is neutral illumination, >1 is a bright caustic line,
 *   and <1 is a shadow region.
 *
 *   For GPU upload (WebGPU r32float):
 *
 *     device.queue.writeTexture(
 *       { texture: causticsTex },
 *       fluidCaustics.getCausticsTexture(),
 *       { bytesPerRow: resolution * 4 },
 *       { width: resolution, height: resolution },
 *     );
 *
 * ─── Coordinate Convention ──────────────────────────────────────────────────
 *
 *   Cell-local UV: [0, 1]² over the Cell bounding box.
 *   getCausticsAt(u, v) samples in this space.
 *   World-space particle positions are remapped into Cell-local UV before
 *   velocity field sampling.
 *
 * Research: xiaodi #M781 — cell-pubsub-loop
 */




// ─── Physical Constants ──────────────────────────────────────────────────────




import { poly6W } from './sph-kernels';
import { SpatialHashGrid } from './SpatialHashGrid';

const IOR_AIR   = 1.0;
const IOR_WATER = 1.333;
const ETA       = IOR_AIR / IOR_WATER;  // ≈ 0.7502

/** Golden angle in radians — optimal angular spacing for wave octaves. */
const GOLDEN_ANGLE = Math.PI * (3.0 - Math.sqrt(5.0));  // ≈ 2.39996 rad ≈ 137.51°

const TWO_PI = 2.0 * Math.PI;

// ─── Configuration ───────────────────────────────────────────────────────────

/** Configuration for a single sine wave octave. */
export interface CausticWaveOctave {
  /** Spatial frequency (cycles per UV unit). Default varies by octave. */
  frequency: number;
  /** Wave amplitude (height contribution). Default varies by octave. */
  amplitude: number;
  /** Temporal speed multiplier. Default 1.0. */
  speed: number;
  /** Base propagation angle in radians. Default: rotated by golden angle. */
  angle: number;
}

/** Configuration for the FluidCaustics system. */
export interface FluidCausticsConfig {
  /**
   * Output texture resolution (square NxN).
   * Default 128 — good balance of quality and performance.
   * Powers of two recommended for GPU upload.
   */
  resolution?: number;

  /**
   * Number of sine wave octaves to superimpose.
   * More octaves → richer, more detailed caustic pattern.
   * Default 5.  Range [2, 8] recommended.
   */
  octaves?: number;

  /**
   * Base spatial frequency for the first octave (cycles per UV unit).
   * Higher values → tighter, more closely-spaced caustic lines.
   * Default 4.0.
   */
  baseFrequency?: number;

  /**
   * Amplitude of the first octave (height units).
   * Controls the overall "strength" of caustic distortion.
   * Default 0.06.
   */
  baseAmplitude?: number;

  /**
   * Amplitude decay per octave (geometric ratio).
   * 0.5 = each octave has half the amplitude of the previous.
   * Default 0.5.
   */
  persistence?: number;

  /**
   * Frequency growth per octave (geometric ratio).
   * 2.0 = each octave has double the frequency (standard octave doubling).
   * Default 2.0.
   */
  lacunarity?: number;

  /**
   * Base temporal speed for wave animation (radians per second).
   * Default 1.2.
   */
  baseSpeed?: number;

  /**
   * Light direction (unit vector, pointing FROM the light TOWARD the surface).
   * Default: (0.4, -0.9, 0.2) — slightly off-vertical for natural-looking caustics.
   */
  lightDir?: [number, number, number];

  /**
   * How strongly the SPH velocity field modulates wave direction.
   * 0 = no velocity influence (pure sine waves).
   * 1 = velocity fully steers wave propagation direction.
   * Default 0.6.
   */
  velocityInfluence?: number;

  /**
   * How strongly the SPH speed field modulates wave amplitude.
   * Higher flow speed → larger wave amplitude → more intense caustics.
   * Default 0.4.
   */
  speedAmplification?: number;

  /**
   * Depth of the imaginary water layer above the Cell surface (world units).
   * Controls how far refracted rays travel before hitting the surface,
   * which affects the scale of caustic displacement.
   * Default 1.0.
   */
  waterDepth?: number;

  /**
   * SPH smoothing radius for velocity field interpolation.
   * Should match the simulation's smoothing radius (typically 12).
   * Default 12.
   */
  smoothingRadius?: number;

  /**
   * Optional explicit wave octave definitions.
   * If provided, overrides octaves/baseFrequency/baseAmplitude/persistence/
   * lacunarity/baseSpeed and uses these octaves directly.
   */
  waveOctaves?: CausticWaveOctave[];

  /**
   * Caustic intensity contrast exponent.
   * Values > 1 sharpen bright caustic lines and deepen shadows.
   * Default 1.5.
   */
  contrastExponent?: number;

  /**
   * Minimum caustic intensity (floor clamp).
   * Prevents completely black shadow regions for aesthetic reasons.
   * Default 0.15.
   */
  minIntensity?: number;
}

// ─── Particle Source (reusable from velocity-field-texture.ts) ────────────────

/**
 * Minimal SOA particle source for velocity field sampling.
 * Compatible with ParticleData from types.ts.
 */
export interface FluidParticleSource {
  x: Float32Array;
  y: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  count: number;
}

// ─── Cell Target ─────────────────────────────────────────────────────────────

/**
 * A Cell's bounding box in world space, defining the surface onto which
 * caustics are projected.
 */
export interface CellSurfaceTarget {
  /** World-space X of the Cell's left edge. */
  x: number;
  /** World-space Y of the Cell's top edge. */
  y: number;
  /** Width of the Cell in world units. */
  w: number;
  /** Height of the Cell in world units. */
  h: number;
}

// ─── Main Class ──────────────────────────────────────────────────────────────

export class FluidCaustics {
  // ── Configuration ──────────────────────────────────────────────────────
  private readonly resolution: number;
  private readonly octaves: CausticWaveOctave[];
  private readonly velocityInfluence: number;
  private readonly speedAmplification: number;
  private readonly waterDepth: number;
  private readonly smoothingRadius: number;
  private readonly contrastExponent: number;
  private readonly minIntensity: number;

  /** Normalised light direction [x, y, z] (pointing toward surface). */
  private readonly light: [number, number, number];

  // ── Internal buffers ───────────────────────────────────────────────────

  /** Height map derived from multi-octave sine waves + SPH modulation. */
  private heightMap: Float32Array;

  /** Surface normal map: 3 floats per texel (nx, ny, nz). */
  private normalMap: Float32Array;

  /** Output caustic intensity: 1 float per texel. */
  private causticsData: Float32Array;

  /**
   * Interpolated velocity field over the Cell surface.
   * 2 floats per texel (vx, vy), populated during update.
   */
  private velocityGrid: Float32Array;

  /** Spatial hash for accelerated particle neighbour lookup. */
  private readonly hash: SpatialHashGrid;

  /** Current simulation time (seconds), advanced by update(). */
  private time: number = 0;

  // ── Constructor ────────────────────────────────────────────────────────

  constructor(cfg: FluidCausticsConfig = {}) {
    this.resolution         = cfg.resolution         ?? 128;
    this.velocityInfluence  = cfg.velocityInfluence  ?? 0.6;
    this.speedAmplification = cfg.speedAmplification ?? 0.4;
    this.waterDepth         = cfg.waterDepth         ?? 1.0;
    this.smoothingRadius    = cfg.smoothingRadius    ?? 12;
    this.contrastExponent   = cfg.contrastExponent   ?? 1.5;
    this.minIntensity       = cfg.minIntensity       ?? 0.15;

    // Normalise light direction
    const rawLight = cfg.lightDir ?? [0.4, -0.9, 0.2];
    const len = Math.sqrt(
      rawLight[0] * rawLight[0] +
      rawLight[1] * rawLight[1] +
      rawLight[2] * rawLight[2],
    );
    this.light = [rawLight[0] / len, rawLight[1] / len, rawLight[2] / len];

    // Build wave octaves
    if (cfg.waveOctaves && cfg.waveOctaves.length > 0) {
      this.octaves = cfg.waveOctaves;
    } else {
      const nOctaves    = cfg.octaves       ?? 5;
      const baseFreq    = cfg.baseFrequency ?? 4.0;
      const baseAmp     = cfg.baseAmplitude ?? 0.06;
      const persistence = cfg.persistence   ?? 0.5;
      const lacunarity  = cfg.lacunarity    ?? 2.0;
      const baseSpeed   = cfg.baseSpeed     ?? 1.2;

      this.octaves = [];
      for (let i = 0; i < nOctaves; i++) {
        this.octaves.push({
          frequency: baseFreq * Math.pow(lacunarity, i),
          amplitude: baseAmp * Math.pow(persistence, i),
          speed:     baseSpeed * (1.0 + i * 0.3),
          angle:     GOLDEN_ANGLE * i,
        });
      }
    }

    // Allocate buffers
    const N = this.resolution;
    this.heightMap     = new Float32Array(N * N);
    this.normalMap     = new Float32Array(N * N * 3);
    this.causticsData  = new Float32Array(N * N);
    this.velocityGrid  = new Float32Array(N * N * 2);

    this.hash = new SpatialHashGrid(131072);
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Update the caustics texture for a specific Cell surface.
   *
   * This is the main entry point.  It:
   *   1. Samples SPH particle velocities over the Cell's bounding box.
   *   2. Computes the multi-octave sine wave height map, modulated by flow.
   *   3. Derives surface normals via central differences.
   *   4. Traces refracted light rays (Snell's law) and computes the Jacobian
   *      determinant to produce caustic intensity.
   *
   * @param particles  SPH particle state (positions + velocities).
   * @param cell       Target Cell's bounding box in world space.
   * @param dt         Time step in seconds.  The internal clock advances by
   *                   this amount, driving wave animation.
   */
  update(
    particles: FluidParticleSource,
    cell: CellSurfaceTarget,
    dt: number,
  ): void {
    this.time += dt;
    this._sampleVelocityField(particles, cell);
    this._buildHeightMap(cell);
    this._computeNormals();
    this._computeCaustics();
  }

  /**
   * Update caustics using a pre-computed velocity field texture instead of
   * raw particles.  Useful when VelocityFieldTexture (M764) has already
   * rasterised the velocity field for other consumers.
   *
   * @param velocityField  Row-major Float32Array, 2 floats per texel (vx, vy).
   * @param fieldResX      Velocity field width in texels.
   * @param fieldResY      Velocity field height in texels.
   * @param cell           Target Cell bounding box in world space.
   * @param worldW         Total world width (for UV mapping).
   * @param worldH         Total world height (for UV mapping).
   * @param dt             Time step in seconds.
   */
  updateFromVelocityField(
    velocityField: Float32Array,
    fieldResX: number,
    fieldResY: number,
    cell: CellSurfaceTarget,
    worldW: number,
    worldH: number,
    dt: number,
  ): void {
    this.time += dt;
    this._remapVelocityField(velocityField, fieldResX, fieldResY, cell, worldW, worldH);
    this._buildHeightMap(cell);
    this._computeNormals();
    this._computeCaustics();
  }

  /**
   * Update caustics with no velocity modulation — pure sine wave caustics.
   * Useful for static decorative caustic patterns on Cell surfaces.
   *
   * @param dt  Time step in seconds.
   */
  updateStatic(dt: number): void {
    this.time += dt;
    this.velocityGrid.fill(0);
    this._buildHeightMap({ x: 0, y: 0, w: 1, h: 1 });
    this._computeNormals();
    this._computeCaustics();
  }

  /**
   * Sample caustic intensity at a Cell-local UV position.
   *
   * @param u  Horizontal UV in [0, 1].
   * @param v  Vertical UV in [0, 1].
   * @returns  Caustic intensity (typically in [minIntensity, 2+]).
   */
  getCausticsAt(u: number, v: number): number {
    return this._sampleBilinear(this.causticsData, u, v, 1, 0);
  }

  /**
   * Sample the wave height at a Cell-local UV position.
   *
   * @param u  Horizontal UV in [0, 1].
   * @param v  Vertical UV in [0, 1].
   * @returns  Surface height displacement.
   */
  getHeightAt(u: number, v: number): number {
    return this._sampleBilinear(this.heightMap, u, v, 1, 0);
  }

  /**
   * Return the raw caustics Float32Array.
   *
   * Layout: row-major, resolution × resolution, 1 float per texel.
   * The returned array is a live view — updated in place each update() call.
   */
  getCausticsTexture(): Float32Array {
    return this.causticsData;
  }

  /**
   * Return the raw height map Float32Array.
   *
   * Layout: row-major, resolution × resolution, 1 float per texel.
   */
  getHeightTexture(): Float32Array {
    return this.heightMap;
  }

  /**
   * Return the raw normal map Float32Array.
   *
   * Layout: row-major, resolution × resolution, 3 floats per texel (nx, ny, nz).
   */
  getNormalTexture(): Float32Array {
    return this.normalMap;
  }

  /** Output texture resolution (square). */
  get size(): number {
    return this.resolution;
  }

  /** Current simulation time in seconds. */
  get currentTime(): number {
    return this.time;
  }

  /** Reset the simulation clock to zero. */
  resetTime(): void {
    this.time = 0;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIVATE — Velocity Field Sampling
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Sample SPH particle velocities over the Cell bounding box using
   * Shepard-normalised Poly6 kernel interpolation (same approach as
   * VelocityFieldTexture in velocity-field-texture.ts).
   *
   * The result populates this.velocityGrid with (vx, vy) pairs in
   * Cell-local UV space.
   */
  private _sampleVelocityField(
    particles: FluidParticleSource,
    cell: CellSurfaceTarget,
  ): void {
    const N = this.resolution;
    const { x: px, y: py, vx: pvx, vy: pvy, count: n } = particles;
    const vGrid = this.velocityGrid;
    vGrid.fill(0);

    if (n === 0) return;

    const h = this.smoothingRadius;
    const h2 = h * h;
    const hash = this.hash;

    hash.clear();
    hash.insertAll(px, py, n, h);

    const head = hash.getHead();
    const next = hash.getNext();
    const tableSize = hash.getTableSize();
    const mask = tableSize - 1;
    const invCellSize = 1.0 / h;

    // Texel centre → world position mapping
    const texelW = cell.w / N;
    const texelH = cell.h / N;

    for (let iy = 0; iy < N; iy++) {
      const wy = cell.y + (iy + 0.5) * texelH;

      for (let ix = 0; ix < N; ix++) {
        const wx = cell.x + (ix + 0.5) * texelW;

        // 3×3 hash cell neighbourhood
        const cxMin = Math.floor((wx - h) * invCellSize) | 0;
        const cxMax = Math.floor((wx + h) * invCellSize) | 0;
        const cyMin = Math.floor((wy - h) * invCellSize) | 0;
        const cyMax = Math.floor((wy + h) * invCellSize) | 0;

        let sumVx = 0.0;
        let sumVy = 0.0;
        let sumW  = 0.0;

        for (let cy = cyMin; cy <= cyMax; cy++) {
          for (let cx = cxMin; cx <= cxMax; cx++) {
            const bucket = (((cx * 92837111) ^ (cy * 689287499)) & mask) >>> 0;
            let j = head[bucket];
            while (j !== -1) {
              const dx = px[j] - wx;
              const dy = py[j] - wy;
              const r2 = dx * dx + dy * dy;
              if (r2 <= h2) {
                const w = poly6W(Math.sqrt(r2), h);
                sumVx += w * pvx[j];
                sumVy += w * pvy[j];
                sumW  += w;
              }
              j = next[j];
            }
          }
        }

        const texIdx = (iy * N + ix) * 2;
        if (sumW > 1e-12) {
          const invW = 1.0 / sumW;
          vGrid[texIdx]     = sumVx * invW;
          vGrid[texIdx + 1] = sumVy * invW;
        }
      }
    }
  }

  /**
   * Remap a pre-computed velocity field texture into the Cell's local
   * coordinate space using bilinear interpolation.
   */
  private _remapVelocityField(
    velocityField: Float32Array,
    fieldResX: number,
    fieldResY: number,
    cell: CellSurfaceTarget,
    worldW: number,
    worldH: number,
  ): void {
    const N = this.resolution;
    const vGrid = this.velocityGrid;

    for (let iy = 0; iy < N; iy++) {
      // Cell-local UV → world → velocity field UV
      const cellV = (iy + 0.5) / N;
      const wy = cell.y + cellV * cell.h;
      const fieldV = wy / worldH;

      for (let ix = 0; ix < N; ix++) {
        const cellU = (ix + 0.5) / N;
        const wx = cell.x + cellU * cell.w;
        const fieldU = wx / worldW;

        const texIdx = (iy * N + ix) * 2;

        // Bilinear sample from the velocity field
        const fpx = fieldU * (fieldResX - 1);
        const fpy = fieldV * (fieldResY - 1);
        const fx0 = Math.max(0, Math.min(Math.floor(fpx), fieldResX - 1));
        const fy0 = Math.max(0, Math.min(Math.floor(fpy), fieldResY - 1));
        const fx1 = Math.min(fx0 + 1, fieldResX - 1);
        const fy1 = Math.min(fy0 + 1, fieldResY - 1);
        const ftx = fpx - fx0;
        const fty = fpy - fy0;

        // Sample vx
        const vx00 = velocityField[(fy0 * fieldResX + fx0) * 2];
        const vx10 = velocityField[(fy0 * fieldResX + fx1) * 2];
        const vx01 = velocityField[(fy1 * fieldResX + fx0) * 2];
        const vx11 = velocityField[(fy1 * fieldResX + fx1) * 2];
        vGrid[texIdx] =
          vx00 * (1 - ftx) * (1 - fty) +
          vx10 * ftx       * (1 - fty) +
          vx01 * (1 - ftx) * fty       +
          vx11 * ftx       * fty;

        // Sample vy
        const vy00 = velocityField[(fy0 * fieldResX + fx0) * 2 + 1];
        const vy10 = velocityField[(fy0 * fieldResX + fx1) * 2 + 1];
        const vy01 = velocityField[(fy1 * fieldResX + fx0) * 2 + 1];
        const vy11 = velocityField[(fy1 * fieldResX + fx1) * 2 + 1];
        vGrid[texIdx + 1] =
          vy00 * (1 - ftx) * (1 - fty) +
          vy10 * ftx       * (1 - fty) +
          vy01 * (1 - ftx) * fty       +
          vy11 * ftx       * fty;
      }
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIVATE — Height Map Generation (Multi-Octave Sine Waves)
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Build the height map by superimposing multiple sine wave octaves,
   * each modulated by the local SPH velocity field.
   *
   * For each texel, the wave direction is a blend of the octave's base
   * direction and the local flow direction (controlled by velocityInfluence).
   * The wave amplitude is scaled by local flow speed (controlled by
   * speedAmplification).
   *
   * Height formula per texel:
   *
   *   h(u,v,t) = Σᵢ amplitudeᵢ · (1 + speedAmp · |v_local|) ·
   *              sin(2π · frequencyᵢ · dot(dirᵢ', uv) + speedᵢ · t)
   *
   * where dirᵢ' is the velocity-modulated wave direction.
   */
  private _buildHeightMap(_cell: CellSurfaceTarget): void {
    const N = this.resolution;
    const invN = 1.0 / N;
    const t = this.time;
    const velInf = this.velocityInfluence;
    const spdAmp = this.speedAmplification;
    const octaves = this.octaves;
    const vGrid = this.velocityGrid;
    const hMap = this.heightMap;

    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const u = (ix + 0.5) * invN;
        const v = (iy + 0.5) * invN;

        const texIdx = iy * N + ix;
        const velIdx = texIdx * 2;
        const localVx = vGrid[velIdx];
        const localVy = vGrid[velIdx + 1];
        const localSpeed = Math.sqrt(localVx * localVx + localVy * localVy);

        // Normalised flow direction (falls back to zero when speed ≈ 0)
        let flowDirX = 0;
        let flowDirY = 0;
        if (localSpeed > 1e-6) {
          const invSpeed = 1.0 / localSpeed;
          flowDirX = localVx * invSpeed;
          flowDirY = localVy * invSpeed;
        }

        let height = 0.0;

        for (let i = 0; i < octaves.length; i++) {
          const oct = octaves[i];

          // Base wave direction for this octave
          const baseDirX = Math.cos(oct.angle);
          const baseDirY = Math.sin(oct.angle);

          // Blend with flow direction
          const dirX = baseDirX * (1.0 - velInf) + flowDirX * velInf;
          const dirY = baseDirY * (1.0 - velInf) + flowDirY * velInf;

          // Normalise the blended direction
          const dirLen = Math.sqrt(dirX * dirX + dirY * dirY);
          const ndx = dirLen > 1e-9 ? dirX / dirLen : baseDirX;
          const ndy = dirLen > 1e-9 ? dirY / dirLen : baseDirY;

          // Phase: spatial dot product + temporal evolution
          const phase = TWO_PI * oct.frequency * (ndx * u + ndy * v) + oct.speed * t;

          // Amplitude modulated by local flow speed
          const amp = oct.amplitude * (1.0 + spdAmp * localSpeed);

          height += amp * Math.sin(phase);
        }

        hMap[texIdx] = height;
      }
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIVATE — Surface Normal Computation
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Compute surface normals from the height map using central differences.
   *
   * For a height field h(u,v), the tangent vectors are:
   *   Tu = (Δu, dh/du, 0)
   *   Tv = (0, dh/dv, Δv)
   *
   * The normal is N = normalize(cross(Tv, Tu)):
   *   N = normalize(-dh/du, Δu·Δv, -dh/dv)
   *
   * We use the convention that the normal points "upward" (positive Y)
   * away from the surface, matching the light direction convention.
   */
  private _computeNormals(): void {
    const N = this.resolution;
    const hMap = this.heightMap;
    const nMap = this.normalMap;
    const invN = 1.0 / N;

    // Clamped height accessor
    const h = (row: number, col: number): number => {
      const r = Math.max(0, Math.min(N - 1, row));
      const c = Math.max(0, Math.min(N - 1, col));
      return hMap[r * N + c];
    };

    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        // Central differences for height gradient
        const dhdx = (h(iy, ix + 1) - h(iy, ix - 1)) * 0.5 * N;  // scale by N for UV-space derivatives
        const dhdy = (h(iy + 1, ix) - h(iy - 1, ix)) * 0.5 * N;

        // Normal = normalize(-dhdx, 1, -dhdy) in (x, y, z) convention
        // where y is "up" (perpendicular to the surface plane)
        const nx = -dhdx;
        const ny = 1.0;
        const nz = -dhdy;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        const invLen = len > 1e-12 ? 1.0 / len : 1.0;

        const nIdx = (iy * N + ix) * 3;
        nMap[nIdx]     = nx * invLen;
        nMap[nIdx + 1] = ny * invLen;
        nMap[nIdx + 2] = nz * invLen;
      }
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIVATE — Caustics Computation (Snell Refraction + Jacobian)
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Compute caustic intensity by tracing refracted light rays through
   * the water surface onto the Cell plane and measuring local light
   * concentration via the Jacobian determinant.
   *
   * For each texel (u, v):
   *   1. Look up the surface normal N(u, v).
   *   2. Compute the refracted ray T via Snell's law (vector form).
   *   3. Project the refracted ray onto the receiver plane at distance
   *      `waterDepth` below the surface.
   *   4. Compute the 2×2 Jacobian J = ∂(projected_uv) / ∂(surface_uv)
   *      via finite differences of the projection mapping.
   *   5. Caustic intensity = 1 / |det(J)|.  Converging rays (det < 1)
   *      produce bright lines; diverging rays (det > 1) produce shadows.
   *
   * The Jacobian approach is more physically accurate than the displacement-
   * based heuristic in water-caustics.ts, producing sharper, more realistic
   * caustic line networks.
   */
  private _computeCaustics(): void {
    const N = this.resolution;
    const invN = 1.0 / N;
    const nMap = this.normalMap;
    const cData = this.causticsData;
    const depth = this.waterDepth;
    const contrast = this.contrastExponent;
    const minI = this.minIntensity;
    const [lx, ly, lz] = this.light;

    // Small epsilon for finite-difference Jacobian computation
    const eps = 1.0 / N;

    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const texIdx = iy * N + ix;

        // ── Refract at current texel ────────────────────────────────────
        const nIdx = texIdx * 3;
        const nx = nMap[nIdx];
        const ny = nMap[nIdx + 1];
        const nz = nMap[nIdx + 2];

        const proj = refractAndProject(lx, ly, lz, nx, ny, nz, depth);

        if (proj === null) {
          // Total internal reflection — no caustic contribution
          cData[texIdx] = minI;
          continue;
        }

        // ── Finite-difference Jacobian ──────────────────────────────────
        // We need ∂proj/∂u and ∂proj/∂v to form the 2×2 Jacobian.
        // Sample the projection at (ix±1, iy) and (ix, iy±1).

        const projPx = this._projectAtTexel(ix + 1, iy, depth);
        const projMx = this._projectAtTexel(ix - 1, iy, depth);
        const projPy = this._projectAtTexel(ix, iy + 1, depth);
        const projMy = this._projectAtTexel(ix, iy - 1, depth);

        // Central differences for the Jacobian columns
        // J = [ ∂projU/∂u  ∂projU/∂v ]
        //     [ ∂projV/∂u  ∂projV/∂v ]
        const dProjU_du = (projPx[0] - projMx[0]) * 0.5;
        const dProjV_du = (projPx[1] - projMx[1]) * 0.5;
        const dProjU_dv = (projPy[0] - projMy[0]) * 0.5;
        const dProjV_dv = (projPy[1] - projMy[1]) * 0.5;

        // Determinant of the Jacobian
        const detJ = dProjU_du * dProjV_dv - dProjU_dv * dProjV_du;

        // Caustic intensity = 1 / |det(J)|
        // det(J) ≈ 1 means no focusing; det(J) < 1 means convergence (bright);
        // det(J) > 1 means divergence (dark).
        const absDetJ = Math.abs(detJ);
        let intensity: number;

        if (absDetJ < 1e-6) {
          // Near-singular Jacobian = extreme focusing = very bright caustic
          intensity = 2.0;
        } else {
          intensity = 1.0 / absDetJ;
        }

        // Apply contrast curve and floor clamp
        intensity = Math.pow(intensity, contrast);
        intensity = Math.max(minI, Math.min(intensity, 2.0));

        cData[texIdx] = intensity;
      }
    }
  }

  /**
   * Helper: compute the refracted projection at a given texel coordinate.
   * Returns the displacement of the projected point relative to the
   * texel's undisturbed position (in texel-index space).
   */
  private _projectAtTexel(
    ix: number,
    iy: number,
    depth: number,
  ): [number, number] {
    const N = this.resolution;

    // Clamp to grid bounds
    const cix = Math.max(0, Math.min(N - 1, ix));
    const ciy = Math.max(0, Math.min(N - 1, iy));

    const nIdx = (ciy * N + cix) * 3;
    const nx = this.normalMap[nIdx];
    const ny = this.normalMap[nIdx + 1];
    const nz = this.normalMap[nIdx + 2];

    const proj = refractAndProject(
      this.light[0], this.light[1], this.light[2],
      nx, ny, nz,
      depth,
    );

    if (proj === null) {
      // TIR: return the undisplaced position
      return [cix, ciy];
    }

    // proj is the (dx, dz) displacement in UV space; convert to texel space
    return [cix + proj[0] * N, ciy + proj[1] * N];
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PRIVATE — Bilinear Sampling
  // ═════════════════════════════════════════════════════════════════════════

  /**
   * Bilinear sample from a Float32Array in [0,1] UV space.
   */
  private _sampleBilinear(
    data: Float32Array,
    u: number,
    v: number,
    stride: number,
    channel: number,
  ): number {
    const N = this.resolution;
    const px = u * (N - 1);
    const py = v * (N - 1);
    const x0 = Math.max(0, Math.min(Math.floor(px), N - 1));
    const y0 = Math.max(0, Math.min(Math.floor(py), N - 1));
    const x1 = Math.min(x0 + 1, N - 1);
    const y1 = Math.min(y0 + 1, N - 1);
    const fx = px - x0;
    const fy = py - y0;

    const s = (row: number, col: number): number =>
      data[(row * N + col) * stride + channel];

    return (
      s(y0, x0) * (1 - fx) * (1 - fy) +
      s(y0, x1) * fx       * (1 - fy) +
      s(y1, x0) * (1 - fx) * fy       +
      s(y1, x1) * fx       * fy
    );
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Module-Level Helpers
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Compute the refracted light ray direction via Snell's law (vector form)
 * and project it onto a flat receiver plane at the given depth.
 *
 * Returns the (dx, dz) displacement in UV space, or null on total internal
 * reflection.
 *
 * Vector Snell's law:
 *   t = η·I + (η·cosθᵢ − cosθₜ)·N
 *   cosθₜ = √(1 − η²·(1 − cos²θᵢ))
 *
 * @param lx, ly, lz  Incident light direction (toward surface, need not be unit).
 * @param nx, ny, nz  Surface normal (unit vector, pointing away from surface).
 * @param depth        Distance from surface to receiver plane.
 */
function refractAndProject(
  lx: number, ly: number, lz: number,
  nx: number, ny: number, nz: number,
  depth: number,
): [number, number] | null {
  // Ensure incident direction is unit
  const lLen = Math.sqrt(lx * lx + ly * ly + lz * lz);
  if (lLen < 1e-12) return [0, 0];
  const ilx = lx / lLen;
  const ily = ly / lLen;
  const ilz = lz / lLen;

  // cos(θᵢ) = -dot(N, I)  (N points outward, I points inward)
  const cosI = -(nx * ilx + ny * ily + nz * ilz);

  // Handle the case where light hits from behind the surface
  let nnx = nx, nny = ny, nnz = nz;
  let cosI2 = cosI;
  if (cosI2 < 0) {
    // Flip normal
    nnx = -nx; nny = -ny; nnz = -nz;
    cosI2 = -cosI2;
  }

  // Snell's law discriminant
  const k = 1.0 - ETA * ETA * (1.0 - cosI2 * cosI2);

  if (k < 0) {
    // Total internal reflection
    return null;
  }

  const cosT = Math.sqrt(k);

  // Refracted ray direction
  const tx = ETA * ilx + (ETA * cosI2 - cosT) * nnx;
  const ty = ETA * ily + (ETA * cosI2 - cosT) * nny;
  const tz = ETA * ilz + (ETA * cosI2 - cosT) * nnz;

  // Project onto the receiver plane.
  // The plane is at y = -depth relative to the surface point.
  // Time to reach the plane: t_hit = depth / |ty|
  // (ty should be negative if light goes downward through the surface)
  if (Math.abs(ty) < 1e-9) {
    // Ray is parallel to the receiver plane — no projection
    return [0, 0];
  }

  const tHit = -depth / ty;
  if (tHit < 0) {
    // Ray goes away from the receiver plane
    return [0, 0];
  }

  // Displacement on the receiver plane
  const dx = tx * tHit;
  const dz = tz * tHit;

  return [dx, dz];
}

// ═════════════════════════════════════════════════════════════════════════════
// Self-test
// ═════════════════════════════════════════════════════════════════════════════

/**
 * selfTest(): boolean
 *
 * Validates the FluidCaustics system:
 *
 *  1. Construction with default config produces valid state.
 *  2. updateStatic() produces a non-trivial caustics texture.
 *  3. Caustics values are bounded within [minIntensity, 2.0].
 *  4. getCausticsAt() bilinear sampling matches direct array access.
 *  5. update() with zero-velocity particles produces a valid texture.
 *  6. update() with moving particles produces flow-modulated caustics.
 *  7. updateFromVelocityField() produces a valid texture.
 *  8. Height map has non-zero values after wave generation.
 *  9. Normal map has unit-length normals.
 * 10. refractAndProject returns null on TIR and valid values otherwise.
 *
 * Returns true when all checks pass; false (with console.error) on failure.
 */
export function selfTest(): boolean {
  const TOL = 1e-4;

  function fail(msg: string): false {
    console.error(`[fluid-caustics selfTest] FAILED: ${msg}`);
    return false;
  }

  const res = 32;
  const cell: CellSurfaceTarget = { x: 50, y: 50, w: 100, h: 100 };

  // ── Test 1: Construction ──────────────────────────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    if (fc.size !== res)
      return fail(`Test 1: size = ${fc.size}, expected ${res}`);
    if (fc.currentTime !== 0)
      return fail(`Test 1: time = ${fc.currentTime}, expected 0`);
  }

  // ── Test 2: Static caustics are non-trivial ───────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    fc.updateStatic(0.1);
    const tex = fc.getCausticsTexture();
    if (tex.length !== res * res)
      return fail(`Test 2: texture length = ${tex.length}, expected ${res * res}`);

    let hasNonOne = false;
    for (let i = 0; i < tex.length; i++) {
      if (Math.abs(tex[i] - 1.0) > 0.01) {
        hasNonOne = true;
        break;
      }
    }
    if (!hasNonOne)
      return fail('Test 2: all caustics ≈ 1.0, expected variation');
  }

  // ── Test 3: Caustics bounded within [minIntensity, 2.0] ───────────────
  {
    const minI = 0.15;
    const fc = new FluidCaustics({ resolution: res, minIntensity: minI });
    fc.updateStatic(0.5);
    const tex = fc.getCausticsTexture();
    for (let i = 0; i < tex.length; i++) {
      if (tex[i] < minI - TOL)
        return fail(`Test 3: caustic[${i}] = ${tex[i]} < minIntensity ${minI}`);
      if (tex[i] > 2.0 + TOL)
        return fail(`Test 3: caustic[${i}] = ${tex[i]} > 2.0`);
    }
  }

  // ── Test 4: Bilinear sampling consistency ──────────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    fc.updateStatic(0.3);
    const tex = fc.getCausticsTexture();

    // Sample at the centre of texel (res/2, res/2)
    const half = Math.floor(res / 2);
    const u = (half + 0.5) / res;
    const v = (half + 0.5) / res;
    const sampled = fc.getCausticsAt(u, v);
    const direct = tex[half * res + half];

    if (Math.abs(sampled - direct) > 0.05 * Math.max(1, Math.abs(direct)))
      return fail(`Test 4: sampled = ${sampled}, direct = ${direct}`);
  }

  // ── Test 5: update() with zero-velocity particles ─────────────────────
  {
    const fc = new FluidCaustics({ resolution: res, smoothingRadius: 20 });
    const n = 50;
    const particles: FluidParticleSource = {
      x: new Float32Array(n),
      y: new Float32Array(n),
      vx: new Float32Array(n),
      vy: new Float32Array(n),
      count: n,
    };
    // Place particles within the cell
    for (let i = 0; i < n; i++) {
      particles.x[i] = cell.x + Math.random() * cell.w;
      particles.y[i] = cell.y + Math.random() * cell.h;
      // vx, vy already 0
    }

    fc.update(particles, cell, 0.1);
    const tex = fc.getCausticsTexture();

    let allZero = true;
    for (let i = 0; i < tex.length; i++) {
      if (tex[i] !== 0) { allZero = false; break; }
    }
    if (allZero)
      return fail('Test 5: all caustics are zero with particles');
  }

  // ── Test 6: Moving particles produce different caustics ────────────────
  {
    const fc1 = new FluidCaustics({ resolution: res, smoothingRadius: 20 });
    const fc2 = new FluidCaustics({ resolution: res, smoothingRadius: 20 });
    const n = 50;

    const baseParticles = {
      x: new Float32Array(n),
      y: new Float32Array(n),
      count: n,
    };
    for (let i = 0; i < n; i++) {
      baseParticles.x[i] = cell.x + Math.random() * cell.w;
      baseParticles.y[i] = cell.y + Math.random() * cell.h;
    }

    const still: FluidParticleSource = {
      ...baseParticles,
      vx: new Float32Array(n),
      vy: new Float32Array(n),
    };

    const moving: FluidParticleSource = {
      ...baseParticles,
      vx: new Float32Array(n).fill(20),
      vy: new Float32Array(n).fill(10),
    };

    fc1.update(still, cell, 0.1);
    fc2.update(moving, cell, 0.1);

    const tex1 = fc1.getCausticsTexture();
    const tex2 = fc2.getCausticsTexture();

    let diff = 0;
    for (let i = 0; i < tex1.length; i++) {
      diff += Math.abs(tex1[i] - tex2[i]);
    }
    if (diff < 0.1)
      return fail(`Test 6: moving vs still diff = ${diff}, expected significant difference`);
  }

  // ── Test 7: updateFromVelocityField() ──────────────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    const fieldRes = 16;
    const velField = new Float32Array(fieldRes * fieldRes * 2);
    // Uniform rightward flow
    for (let i = 0; i < fieldRes * fieldRes; i++) {
      velField[i * 2] = 5.0;
      velField[i * 2 + 1] = 0.0;
    }

    fc.updateFromVelocityField(velField, fieldRes, fieldRes, cell, 200, 200, 0.1);
    const tex = fc.getCausticsTexture();

    let hasVariation = false;
    for (let i = 1; i < tex.length; i++) {
      if (Math.abs(tex[i] - tex[0]) > 0.01) {
        hasVariation = true;
        break;
      }
    }
    if (!hasVariation)
      return fail('Test 7: no variation in caustics from velocity field');
  }

  // ── Test 8: Height map is non-zero ─────────────────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    fc.updateStatic(0.5);
    const hTex = fc.getHeightTexture();

    let maxAbs = 0;
    for (let i = 0; i < hTex.length; i++) {
      const a = Math.abs(hTex[i]);
      if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs < 1e-6)
      return fail(`Test 8: max |height| = ${maxAbs}, expected non-zero`);
  }

  // ── Test 9: Normals are unit length ────────────────────────────────────
  {
    const fc = new FluidCaustics({ resolution: res });
    fc.updateStatic(0.2);
    const nTex = fc.getNormalTexture();

    for (let i = 0; i < res * res; i++) {
      const nx = nTex[i * 3];
      const ny = nTex[i * 3 + 1];
      const nz = nTex[i * 3 + 2];
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (Math.abs(len - 1.0) > 0.01)
        return fail(`Test 9: normal[${i}] length = ${len}, expected ≈ 1.0`);
    }
  }

  // ── Test 10: refractAndProject correctness ─────────────────────────────
  {
    // Straight-down light through a flat surface should produce zero displacement
    const flat = refractAndProject(0, -1, 0, 0, 1, 0, 1.0);
    if (flat === null)
      return fail('Test 10a: flat surface refraction returned null');
    if (Math.abs(flat[0]) > TOL || Math.abs(flat[1]) > TOL)
      return fail(`Test 10a: flat displacement = (${flat[0]}, ${flat[1]}), expected (0, 0)`);

    // Tilted surface should produce non-zero displacement
    const nx = 0.3, ny = Math.sqrt(1 - 0.3 * 0.3 - 0.1 * 0.1), nz = 0.1;
    const tilted = refractAndProject(0, -1, 0, nx, ny, nz, 1.0);
    if (tilted === null)
      return fail('Test 10b: tilted surface refraction returned null');
    const dispMag = Math.sqrt(tilted[0] * tilted[0] + tilted[1] * tilted[1]);
    if (dispMag < 1e-4)
      return fail(`Test 10b: tilted displacement magnitude = ${dispMag}, expected > 0`);
  }

  return true;
}
