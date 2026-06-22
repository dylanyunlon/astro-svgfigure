# === src/lib/sph/BoundaryModel.ts ===

// BoundaryModel.ts — Akinci 2012 rigid-body boundary particles
// Implements volume-weighted boundary particle sampling with Cubic Spline kernel
// for computing Ψ_b (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for SPH")

import { ObstacleData } from "./types";

// ─── Constants ────────────────────────────────────────────────────────────────

/** Spacing factor relative to smoothing length h */
const BOUNDARY_SPACING_FACTOR = 0.8;

/** Cubic Spline kernel normalisation constant (2-D) */
const CS_ALPHA_2D = 10.0 / (7.0 * Math.PI);

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BoundaryParticle {
  x: number;
  y: number;
  /** Ψ_b — Akinci volume estimate for the boundary particle */
  volume: number;
}

// ─── Kernel (inlined Cubic Spline, 2-D) ──────────────────────────────────────

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
 * Cubic Spline kernel gradient magnitude  ∂W/∂r  (scalar).
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

// ─── BoundaryModel ────────────────────────────────────────────────────────────

export class BoundaryModel {
  private device: GPUDevice;
  private restDensity: number;
  readonly h: number;

  particles: BoundaryParticle[] = [];

  /** GPU storage buffer: [ x, y, volume, pad ]  ×  N  (Float32, stride = 16 B) */
  gpuBoundaryBuf!: GPUBuffer;

  readonly domainW: number;
  readonly domainH: number;

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

  // ─── Sampling ──────────────────────────────────────────────────────────────

  /**
   * Sample a filled box [xMin, xMax] × [yMin, yMax] with boundary particles.
   * Particles are placed on the perimeter with spacing `h * BOUNDARY_SPACING_FACTOR`.
   * Volumes are initialised via the Akinci Ψ summation approximation.
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
   * Volumes are initialised via the Akinci Ψ summation approximation.
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

  // ─── Volume initialisation (Akinci Eq. 4) ─────────────────────────────────

  /**
   * Compute Ψ_b for every boundary particle via:
   *
   *   Ψ_b(x_b) = ρ₀ / Σ_k W(x_b − x_k, h)
   *
   * Only newly-added particles with volume == 0 are (re)computed;
   * this keeps the cost incremental when obstacles are added at runtime.
   *
   * Complexity: O(N²) over boundary particles — acceptable because boundary
   * particle counts are typically O(100–1000).
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

  // ─── GPU upload ────────────────────────────────────────────────────────────

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

  // ─── Legacy obstacle helper ────────────────────────────────────────────────

  /**
   * @deprecated  Prefer `sampleCircle(obs)` — identical behaviour, consistent naming.
   */
  addCircle(obs: ObstacleData): void {
    this.sampleCircle(obs);
  }

  // ─── Accessors ─────────────────────────────────────────────────────────────

  get count(): number {
    return this.particles.length;
  }

  destroy(): void {
    this.gpuBoundaryBuf?.destroy();
  }
}

// ─── Re-export kernel for unit tests / shader parity checks ──────────────────
export { cubicSplineW, cubicSplinedW_dr };
