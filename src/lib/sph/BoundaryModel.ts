// === src/lib/sph/BoundaryModel.ts ===

// BoundaryModel.ts --- Akinci 2012 rigid-body boundary particles
// Implements volume-weighted boundary particle sampling with Cubic Spline kernel
// for computing -_b (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for SPH")
//
// M547: extended with configurable boundary shapes (rect / circle / polygon) and
//       automatic resampling via `resample()` / `resampleWorld()`.




// --------- Constants ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/** Spacing factor relative to smoothing length h */



import { ObstacleData } from "./types";
import {

  BoundaryShape,
  createPolygonObstacle,
  createBoxObstacle,
  createCircleObstacle,
  BoundaryParticle as WBParticle,
} from "./world-boundary";

// --------- Constants ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// [orphan-precise] /** Spacing factor relative to smoothing length h */
const BOUNDARY_SPACING_FACTOR = 0.8;

/** Cubic Spline kernel normalisation constant (2-D) */
const CS_ALPHA_2D = 10.0 / (7.0 * Math.PI);

// --------- Types ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

export interface BoundaryParticle {
  x: number;
  y: number;
  /** -_b --- Akinci volume estimate for the boundary particle */
  volume: number;
}

// --------- Shape configuration (M547) ---------------------------------------------------------------------------------------------------------------------------------------------

/** Per-entry in the shape registry. */
interface ShapeEntry {
  shape:   BoundaryShape;
  /** First particle index in `this.particles` owned by this entry. */
  start:   number;
  /** Exclusive end index. */
  end:     number;
}

// --------- Kernel (inlined Cubic Spline, 2-D) ------------------------------------------------------------------------------------------------------------------

/**
 * Cubic Spline kernel value W(r, h).
 * Monaghan 1992 form; normalised for 2-D.
 *
 * @param r  scalar distance between particles
 * @param h  smoothing length
 */
function cubicSplineW(r: number, h: number): number {
  const q = r / h;
  const alpha = CS_ALPHA_2D / (h * h);

  if (q < 1.0) {
    const q2 = q * q;
    const q3 = q2 * q;
    return alpha * (1.0 - 1.5 * q2 + 0.75 * q3);
  } else if (q < 2.0) {
    const t = 2.0 - q;
    return alpha * (0.25 * t * t * t);
  }
  return 0.0;
}

/**
 * Cubic Spline kernel gradient magnitude  ---W/---r  (scalar).
 * Caller is responsible for multiplying by the unit direction vector.
 *
 * @param r  scalar distance
 * @param h  smoothing length
 */
function cubicSplinedW_dr(r: number, h: number): number {
  const q = r / h;
  const alpha = CS_ALPHA_2D / (h * h * h);   // extra 1/h from chain rule

  if (q < 1.0) {
    return alpha * (-3.0 * q + 2.25 * q * q);
  } else if (q < 2.0) {
    const t = 2.0 - q;
    return alpha * (-0.75 * t * t);
  }
  return 0.0;
}

// --------- BoundaryModel ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

export class BoundaryModel {
  private device: GPUDevice;
  private restDensity: number;
  readonly h: number;

  particles: BoundaryParticle[] = [];

  /** GPU storage buffer: [ x, y, volume, pad ]  --  N  (Float32, stride = 16 B) */
  gpuBoundaryBuf!: GPUBuffer;

  readonly domainW: number;
  readonly domainH: number;

  // ------ M547: shape registry ------------------------------------------------------------------------------------------------------------------------------------------------------------
  /** Ordered list of registered shapes; used for automatic resampling. */
  private _shapes: ShapeEntry[] = [];

  /**
   * Active world boundary shape.
   * Set via `configureWorldBoundary()` or `resampleWorld()`.
   * When null the legacy sampleBox(0,0,domainW,domainH) is used.
   */
  private _worldShape: BoundaryShape | null = null;

  constructor(
    device: GPUDevice,
    domainW: number,
    domainH: number,
    restDensity: number,
    h: number
  ) {
    this.device = device;
    this.domainW = domainW;
    this.domainH = domainH;
    this.restDensity = restDensity;
    this.h = h;
  }

  // --------- M547: Shape configuration ------------------------------------------------------------------------------------------------------------------------------------------

  /**
   * Register (or replace) the world boundary shape.
   *
   * Replaces any previously configured world boundary in the particle list and
   * immediately resamples particles for the new shape.  After this call you
   * must call `getBuffers()` / `uploadToGPU()` to push the change to the GPU.
   *
   * @param shape  One of `RectBoundaryShape`, `CircleBoundaryShape`, or `PolygonBoundaryShape`.
   * @param layers Number of wall particle layers (default 3).
   */
  configureWorldBoundary(shape: BoundaryShape, layers = 3): void {
    this._worldShape = shape;
    this._resampleWorldShape(shape, layers);
  }

  /**
   * Full resample: rebuild ALL boundary particles from the registered shape
   * registry (world boundary + any obstacles added via `addShape()`).
   *
   * Call this after changing `h`, domain size, or any shape parameter.
   */
  resample(): void {
    this.particles = [];
    this._shapes = [];

    if (this._worldShape !== null) {
      this._resampleWorldShape(this._worldShape, 3);
    }
  }

  /**
   * Convenience: set a new world boundary shape and immediately resample.
   * Equivalent to `configureWorldBoundary(shape, layers)` but returns `this`
   * for chaining.
   */
  resampleWorld(shape: BoundaryShape, layers = 3): this {
    this.configureWorldBoundary(shape, layers);
    return this;
  }

  /**
   * Add an arbitrary shape as an obstacle boundary.
   * The shape is appended to the registry; calling `resample()` regenerates it.
   *
   * @returns The index of the shape in the registry (for later removal).
   */
  addShape(shape: BoundaryShape, layers = 1): number {
    const start = this.particles.length;
    const pts = this._sampleShape(shape, layers);
    for (const p of pts) this.particles.push({ x: p.x, y: p.y, volume: 0 });
    const end = this.particles.length;

    this._shapes.push({ shape, start, end });
    this._computeVolumes();

    return this._shapes.length - 1;
  }

  // --------- Sampling ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  /**
   * Sample a filled box [xMin, xMax] -- [yMin, yMax] with boundary particles.
   * Particles are placed on the perimeter with spacing `h * BOUNDARY_SPACING_FACTOR`.
   * Volumes are initialised via the Akinci - summation approximation.
   */
  sampleBox(
    xMin: number,
    yMin: number,
    xMax: number,
    yMax: number
  ): void {
    const spacing = this.h * BOUNDARY_SPACING_FACTOR;

    // Bottom and top edges
    for (let x = xMin; x <= xMax + 1e-9; x += spacing) {
      this.particles.push({ x, y: yMin, volume: 0 });
      this.particles.push({ x, y: yMax, volume: 0 });
    }
    // Left and right edges (skip corners already added)
    for (let y = yMin + spacing; y < yMax - 1e-9; y += spacing) {
      this.particles.push({ x: xMin, y, volume: 0 });
      this.particles.push({ x: xMax, y, volume: 0 });
    }

    this._computeVolumes();
  }

  /**
   * Sample a circular obstacle with boundary particles placed on its circumference.
   * `obs.r` is the radius; particles are evenly distributed along the arc.
   * Volumes are initialised via the Akinci - summation approximation.
   */
  sampleCircle(obs: ObstacleData): void {
    const spacing = this.h * BOUNDARY_SPACING_FACTOR;
    const nPts = Math.max(8, Math.ceil((2 * Math.PI * obs.r) / spacing));

    for (let k = 0; k < nPts; k++) {
      const angle = (2 * Math.PI * k) / nPts;
      this.particles.push({
        x: obs.cx + obs.r * Math.cos(angle),
        y: obs.cy + obs.r * Math.sin(angle),
        volume: 0,
      });
    }

    this._computeVolumes();
  }

  // --------- Volume initialisation (Akinci Eq. 4) ---------------------------------------------------------------------------------------------------

  /**
   * Compute -_b for every boundary particle via:
   *
   *   -_b(x_b) = ----- / --_k W(x_b --- x_k, h)
   *
   * Only newly-added particles with volume == 0 are (re)computed;
   * this keeps the cost incremental when obstacles are added at runtime.
   *
   * Complexity: O(N-) over boundary particles --- acceptable because boundary
   * particle counts are typically O(100---1000).
   */
  initVolumes(): void {
    this._computeVolumes();
  }

  private _computeVolumes(): void {
    const n = this.particles.length;

    for (let i = 0; i < n; i++) {
      if (this.particles[i].volume !== 0) continue; // already computed

      let wSum = 0.0;
      const xi = this.particles[i].x;
      const yi = this.particles[i].y;

      for (let j = 0; j < n; j++) {
        const dx = xi - this.particles[j].x;
        const dy = yi - this.particles[j].y;
        const r = Math.sqrt(dx * dx + dy * dy);
        wSum += cubicSplineW(r, this.h);
      }

      // Guard against isolated particles (should not occur with valid sampling)
      this.particles[i].volume =
        wSum > 1e-12 ? this.restDensity / wSum : 0.0;
    }
  }

  // --------- GPU upload ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  /**
   * (Re)build the GPU storage buffer from the current particle list.
   * Layout per entry (stride 16 bytes):
   *   offset  0: x       (f32)
   *   offset  4: y       (f32)
   *   offset  8: volume  (f32)
   *   offset 12: pad     (f32 = 0)
   */
  getBuffers(device?: GPUDevice): GPUBuffer {
    const gpu = device ?? this.device;

    this.gpuBoundaryBuf?.destroy();

    const n = this.particles.length;
    const data = new Float32Array(n * 4);

    for (let i = 0; i < n; i++) {
      data[i * 4 + 0] = this.particles[i].x;
      data[i * 4 + 1] = this.particles[i].y;
      data[i * 4 + 2] = this.particles[i].volume;
      data[i * 4 + 3] = 0; // padding
    }

    const byteLen = Math.max(data.byteLength, 16);

    this.gpuBoundaryBuf = gpu.createBuffer({
      label: "boundaryParticles",
      size: byteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    if (n > 0) {
      gpu.queue.writeBuffer(this.gpuBoundaryBuf, 0, data);
    }

    return this.gpuBoundaryBuf;
  }

  /** Convenience alias kept for back-compat with draft API. */
  uploadToGPU(): void {
    this.getBuffers();
  }

  // --------- Legacy obstacle helper ------------------------------------------------------------------------------------------------------------------------------------------------

  /**
   * @deprecated  Prefer `sampleCircle(obs)` --- identical behaviour, consistent naming.
   */
  addCircle(obs: ObstacleData): void {
    this.sampleCircle(obs);
  }

  // --------- Accessors ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  get count(): number {
    return this.particles.length;
  }

  destroy(): void {
    this.gpuBoundaryBuf?.destroy();
  }

  // --------- Private M547 helpers ------------------------------------------------------------------------------------------------------------------------------------------------------

  /** Sample any BoundaryShape to a flat WBParticle array. */
  private _sampleShape(shape: BoundaryShape, layers: number): WBParticle[] {
    const s = this.h * BOUNDARY_SPACING_FACTOR;

    switch (shape.kind) {
      case 'rect': {
        // Use the box obstacle helper centred at half-extents
        const hw = shape.width  / 2;
        const hh = shape.height / 2;
        return createBoxObstacle(hw, hh, hw, hh, s, layers);
      }

      case 'circle': {
        const cx = shape.cx ?? 0;
        const cy = shape.cy ?? 0;
        return createCircleObstacle(cx, cy, shape.radius, s, layers);
      }

      case 'polygon':
        return createPolygonObstacle(shape.vertices, s, layers);
    }
  }

  /** Resample the world-boundary shape (always at the start of `particles`). */
  private _resampleWorldShape(shape: BoundaryShape, layers: number): void {
    // Remove any existing world-boundary particles (always the first entry).
    const existing = this._shapes.find(e => e === this._shapes[0]);
    if (existing) {
      const removed = existing.end - existing.start;
      this.particles.splice(existing.start, removed);
      this._shapes.shift();
      // Shift all subsequent entries
      for (const entry of this._shapes) {
        entry.start -= removed;
        entry.end   -= removed;
      }
    }

    // Insert fresh world boundary particles at the front.
    const pts = this._sampleShape(shape, layers);
    const newParticles: BoundaryParticle[] = pts.map(p => ({
      x: p.x, y: p.y, volume: 0,
    }));
    this.particles.unshift(...newParticles);

    // Register as first entry.
    this._shapes.unshift({ shape, start: 0, end: newParticles.length });

    // Shift all subsequent shape entries.
    for (let i = 1; i < this._shapes.length; i++) {
      this._shapes[i].start += newParticles.length;
      this._shapes[i].end   += newParticles.length;
    }

    // Recompute volumes for all invalidated (volume=0) particles.
    this._computeVolumes();
  }
}

// --------- Re-export kernel for unit tests / shader parity checks ------------------------------------------------------
export { cubicSplineW, cubicSplinedW_dr };
