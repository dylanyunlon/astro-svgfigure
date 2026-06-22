/**
 * sph-kernels.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * SPH kernel math library for 2-D Lagrangian simulation.
 *
 * All three kernel families follow Müller et al. 2003 ("Particle-Based Fluid
 * Simulation for Interactive Applications") together with the CubicKernel
 * 2-D normalisation from SPlisHSPlasH (SPHKernels.h).
 *
 * Coordinate convention
 *   r  = |x_i − x_j|          (scalar distance, ≥ 0)
 *   dx = x_i − x_j            (signed component)
 *   q  = r / h                 (normalised distance, ∈ [0, 1] for support)
 *
 * No external dependencies.
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─── Configuration ────────────────────────────────────────────────────────────

export interface SPHConfig {
  /** Smoothing radius h (support = 2 h for cubic, h for others). Typical: 12 */
  smoothingRadius: number;
  /** Rest density ρ₀  [kg/m³ sim-units]. Typical: 1000 */
  restDensity: number;
  /** Gas stiffness constant k (equation-of-state). Typical: 50 */
  gasStiffness: number;
  /** Dynamic viscosity μ. Range 0.001 – 0.05 */
  viscosity: number;
  /** Particle mass m. Typical: 1.0 */
  particleMass: number;
  /** Timestep Δt [s sim-units]. Typical: 0.005 */
  dt: number;
  /** Gravitational acceleration g (positive = downward). Typical: 300 */
  gravity: number;
}

export function defaultConfig(): SPHConfig {
  return {
    smoothingRadius: 12,
    restDensity:     1000,
    gasStiffness:    50,
    viscosity:       0.01,
    particleMass:    1.0,
    dt:              0.005,
    gravity:         300,
  };
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/** Safe inverse: returns 0 when the denominator is (near) zero. */
function safeInv(x: number): number {
  return Math.abs(x) > 1e-12 ? 1.0 / x : 0.0;
}

// ─── Cubic Spline kernel ──────────────────────────────────────────────────────
//
//  2-D normalisation constant:  α₂D = 40 / (7 π h²)
//  (matches SPlisHSPlasH CubicKernel, Monaghan 1992)
//
//       ⎧ α · ( 1 − 6q² + 6q³ )       0  ≤ q ≤ ½
//  W  = ⎨ α · 2(1 − q)³               ½  < q ≤ 1
//       ⎩ 0                            q  > 1
//
//  where q = r / h.
//
//  Gradient:
//       ∂W/∂xᵢ = (dW/dr) · (x̂) / h
//
//       ⎧ α/h · ( −12q + 18q² )        0  ≤ q ≤ ½
//  dW/dr= ⎨ α/h · −6(1 − q)²           ½  < q ≤ 1
//       ⎩ 0                             q  > 1

const _2_PI = 2.0 * Math.PI;

function cubicAlpha(h: number): number {
  return 40.0 / (7.0 * Math.PI * h * h);
}

/**
 * Cubic spline kernel value W(r, h).
 * Support radius = h (q is already r/h; support is q ∈ [0,1]).
 */
export function cubicW(r: number, h: number): number {
  const alpha = cubicAlpha(h);
  const q = r * safeInv(h);

  if (q > 1.0) return 0.0;

  if (q <= 0.5) {
    const q2 = q * q;
    const q3 = q2 * q;
    return alpha * (1.0 - 6.0 * q2 + 6.0 * q3);
  } else {
    const t = 1.0 - q;
    return alpha * 2.0 * t * t * t;
  }
}

/**
 * Cubic spline kernel gradient ∇W(rᵢⱼ, h).
 * Returns [∂W/∂x, ∂W/∂y] where (dx, dy) = xᵢ − xⱼ.
 */
export function cubicGradW(
  dx: number,
  dy: number,
  h: number,
): [number, number] {
  const r = Math.sqrt(dx * dx + dy * dy);
  if (r < 1e-12) return [0.0, 0.0];

  const q = r / h;
  if (q > 1.0) return [0.0, 0.0];

  const alpha = cubicAlpha(h);
  const invR  = 1.0 / r;
  const invH  = 1.0 / h;

  let dWdr: number;
  if (q <= 0.5) {
    dWdr = alpha * invH * (-12.0 * q + 18.0 * q * q);
  } else {
    const t = 1.0 - q;
    dWdr = alpha * invH * (-6.0 * t * t);
  }

  // ∇W = (dW/dr) * r̂  where r̂ = (dx, dy) / r
  const scale = dWdr * invR;
  return [scale * dx, scale * dy];
}

// ─── Spiky kernel (pressure gradient) ────────────────────────────────────────
//
//  Müller 2003, designed to stay non-zero gradient near r = 0 so pressure
//  forces keep particles from clumping.
//
//  2-D normalisation:  α = 10 / (π h⁵)
//
//       W(r) = α (h − r)³          0 ≤ r ≤ h
//
//  Gradient:
//       ∇W   = −3α (h − r)² r̂
//            = −3α (h − r)² (rᵢⱼ / r)

function spikyAlpha(h: number): number {
  return 10.0 / (Math.PI * Math.pow(h, 5));
}

/**
 * Spiky kernel gradient ∇W(rᵢⱼ, h) — used for the pressure force term.
 * Returns [∂W/∂x, ∂W/∂y].
 */
export function spikyGradW(
  dx: number,
  dy: number,
  h: number,
): [number, number] {
  const r = Math.sqrt(dx * dx + dy * dy);
  if (r < 1e-12 || r > h) return [0.0, 0.0];

  const alpha  = spikyAlpha(h);
  const diff   = h - r;
  const scale  = -3.0 * alpha * diff * diff / r;
  return [scale * dx, scale * dy];
}

// ─── Poly6 kernel (density estimation) ───────────────────────────────────────
//
//  Müller 2003.  Smooth, computationally cheap, avoids sqrt in the hot
//  density-summation loop (pass r² if desired; here we accept r for API
//  consistency with the other kernels).
//
//  2-D normalisation:  α = 4 / (π h⁸)
//
//       W(r) = α (h² − r²)³       0 ≤ r ≤ h

function poly6Alpha(h: number): number {
  return 4.0 / (Math.PI * Math.pow(h, 8));
}

/**
 * Poly6 kernel value W(r, h) — used for density summation.
 */
export function poly6W(r: number, h: number): number {
  if (r > h) return 0.0;
  const alpha = poly6Alpha(h);
  const diff  = h * h - r * r;
  return alpha * diff * diff * diff;
}

// ─── Viscosity kernel Laplacian ───────────────────────────────────────────────
//
//  Müller 2003.  Designed so ∇²W ≥ 0 everywhere → physically stable viscosity.
//
//  2-D normalisation:  α = 40 / (π h⁵)
//
//       W(r)    = α [ −r³/(2h³) + r²/h² + h/(2r) − 1 ]
//       ∇²W(r)  = α · 6(h − r) / h³                      (scalar)

function viscAlpha(h: number): number {
  return 40.0 / (Math.PI * Math.pow(h, 5));
}

/**
 * Viscosity kernel Laplacian ∇²W(r, h) — scalar, used for viscosity force.
 */
export function viscLaplacianW(r: number, h: number): number {
  if (r < 1e-12 || r > h) return 0.0;
  const alpha = viscAlpha(h);
  return alpha * 6.0 * (h - r) / (h * h * h);
}
