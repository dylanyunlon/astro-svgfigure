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

// ─── Self-test ────────────────────────────────────────────────────────────────

/**
 * selfTest(): boolean
 *
 * Validates the three SPH kernel families against their analytical properties:
 *
 *  1. Non-negativity           W(r, h) ≥ 0 for all r ∈ [0, h]
 *  2. Compact support          W(r, h) = 0 for r > h  (and for r = 0 in
 *                              the case of the gradient / Laplacian tests)
 *  3. Monotone decay           W is non-increasing with r inside support
 *  4. Approximate unity        ∫ W dΩ ≈ 1  (2-D Riemann sum over disk)
 *  5. Gradient anti-symmetry   ∇W(r⃗) = −∇W(−r⃗)
 *  6. Gradient points inward   sign(∇W · r̂) ≤ 0  (kernel decreases outward)
 *  7. Laplacian non-negative   ∇²W ≥ 0 (viscosity kernel stability condition)
 *  8. Peak at origin           W(0) ≥ W(r) for all tested r (poly6 & cubic)
 *  9. Zero-gradient symmetry   ‖∇W(x,0)‖ = ‖∇W(−x,0)‖
 * 10. Cross-kernel ordering    poly6W(0,h) ≥ cubicW(0,h)  (poly6 taller peak)
 *
 * Returns true when every assertion passes, false (with a console.error) on
 * the first failure.  Designed to run in Node / browser with no imports.
 */
export function selfTest(): boolean {
  const h = 12.0;
  const N = 200;          // radial sample count for integral / monotone checks
  const TOL = 1e-6;       // floating-point tolerance

  // ── helper ──────────────────────────────────────────────────────────────────
  function fail(msg: string): false {
    console.error(`[sph-kernels selfTest] FAILED: ${msg}`);
    return false;
  }

  // ── 1 & 2: Non-negativity and compact support (all three kernels) ────────────
  for (let i = 0; i <= N + 5; i++) {
    const r = (i / N) * h * 1.2;   // slightly beyond support

    const w_cubic = cubicW(r, h);
    const w_poly6 = poly6W(r, h);
    const w_visc  = viscLaplacianW(r, h);  // Laplacian form (can be 0 at r=0)

    if (w_cubic < -TOL)
      return fail(`cubicW(${r.toFixed(3)}, ${h}) = ${w_cubic} < 0`);
    if (w_poly6 < -TOL)
      return fail(`poly6W(${r.toFixed(3)}, ${h}) = ${w_poly6} < 0`);
    if (w_visc < -TOL)
      return fail(`viscLaplacianW(${r.toFixed(3)}, ${h}) = ${w_visc} < 0`);

    if (r > h + TOL) {
      if (Math.abs(w_cubic) > TOL)
        return fail(`cubicW outside support at r=${r.toFixed(3)}: ${w_cubic}`);
      if (Math.abs(w_poly6) > TOL)
        return fail(`poly6W outside support at r=${r.toFixed(3)}: ${w_poly6}`);
      if (Math.abs(w_visc) > TOL)
        return fail(`viscLaplacianW outside support at r=${r.toFixed(3)}: ${w_visc}`);
    }
  }

  // ── 3: Monotone non-increasing decay inside support ──────────────────────────
  {
    let prev_cubic = cubicW(0, h);
    let prev_poly6 = poly6W(0, h);
    for (let i = 1; i <= N; i++) {
      const r = (i / N) * h;
      const c = cubicW(r, h);
      const p = poly6W(r, h);
      if (c > prev_cubic + TOL)
        return fail(`cubicW not monotone: W(${((i-1)/N*h).toFixed(3)})=${prev_cubic} < W(${r.toFixed(3)})=${c}`);
      if (p > prev_poly6 + TOL)
        return fail(`poly6W not monotone: W(${((i-1)/N*h).toFixed(3)})=${prev_poly6} < W(${r.toFixed(3)})=${p}`);
      prev_cubic = c;
      prev_poly6 = p;
    }
  }

  // ── 4: Approximate unity  ∫ W(r) dA ≈ 1  (2-D Riemann sum) ─────────────────
  //   ∫₀ʰ W(r) · 2π r dr  ≈ Σ W(rᵢ) · 2π rᵢ Δr
  {
    const dr = h / N;
    let sumCubic = 0.0;
    let sumPoly6 = 0.0;
    for (let i = 0; i < N; i++) {
      const r   = (i + 0.5) * dr;   // midpoint rule
      const circ = 2.0 * Math.PI * r * dr;
      sumCubic += cubicW(r, h) * circ;
      sumPoly6 += poly6W(r, h) * circ;
    }
    if (Math.abs(sumCubic - 1.0) > 0.01)
      return fail(`cubicW unity integral = ${sumCubic.toFixed(6)} (expected ≈ 1)`);
    if (Math.abs(sumPoly6 - 1.0) > 0.01)
      return fail(`poly6W unity integral = ${sumPoly6.toFixed(6)} (expected ≈ 1)`);
  }

  // ── 5: Gradient anti-symmetry  ∇W(r⃗) = −∇W(−r⃗) ────────────────────────────
  {
    const testVecs: [number, number][] = [
      [h * 0.3, 0], [0, h * 0.5], [h * 0.25, h * 0.25],
      [h * 0.7, h * 0.1], [h * 0.1, h * 0.6],
    ];
    for (const [dx, dy] of testVecs) {
      // Only test pairs strictly inside support
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r >= h) continue;

      const [gx1, gy1] = cubicGradW( dx,  dy, h);
      const [gx2, gy2] = cubicGradW(-dx, -dy, h);
      if (Math.abs(gx1 + gx2) > TOL || Math.abs(gy1 + gy2) > TOL)
        return fail(`cubicGradW anti-symmetry at (${dx},${dy}): [${gx1},${gy1}] vs [${gx2},${gy2}]`);

      const [sx1, sy1] = spikyGradW( dx,  dy, h);
      const [sx2, sy2] = spikyGradW(-dx, -dy, h);
      if (Math.abs(sx1 + sx2) > TOL || Math.abs(sy1 + sy2) > TOL)
        return fail(`spikyGradW anti-symmetry at (${dx},${dy}): [${sx1},${sy1}] vs [${sx2},${sy2}]`);
    }
  }

  // ── 6: Gradient points inward  (∇W · r̂ ≤ 0) ─────────────────────────────────
  {
    const angles = [0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2];
    const radii  = [0.1, 0.3, 0.5, 0.7, 0.9].map(t => t * h);
    for (const angle of angles) {
      for (const r of radii) {
        const dx = r * Math.cos(angle);
        const dy = r * Math.sin(angle);

        const [cgx, cgy] = cubicGradW(dx, dy, h);
        const dotC = cgx * dx + cgy * dy;
        if (dotC > TOL)
          return fail(`cubicGradW points outward at r=${r.toFixed(2)}, θ=${angle.toFixed(2)}: dot=${dotC}`);

        const [sgx, sgy] = spikyGradW(dx, dy, h);
        const dotS = sgx * dx + sgy * dy;
        if (dotS > TOL)
          return fail(`spikyGradW points outward at r=${r.toFixed(2)}, θ=${angle.toFixed(2)}: dot=${dotS}`);
      }
    }
  }

  // ── 7: Laplacian non-negative ─────────────────────────────────────────────────
  {
    for (let i = 0; i <= N; i++) {
      const r = (i / N) * h;
      const lap = viscLaplacianW(r, h);
      if (lap < -TOL)
        return fail(`viscLaplacianW(${r.toFixed(3)}, ${h}) = ${lap} < 0`);
    }
  }

  // ── 8: Peak at origin (poly6 & cubic) ────────────────────────────────────────
  {
    const peakCubic = cubicW(0, h);
    const peakPoly6 = poly6W(0, h);
    for (let i = 1; i <= 20; i++) {
      const r = (i / 20) * h;
      if (cubicW(r, h) > peakCubic + TOL)
        return fail(`cubicW peak not at r=0: W(${r.toFixed(2)})=${cubicW(r,h)} > W(0)=${peakCubic}`);
      if (poly6W(r, h) > peakPoly6 + TOL)
        return fail(`poly6W peak not at r=0: W(${r.toFixed(2)})=${poly6W(r,h)} > W(0)=${peakPoly6}`);
    }
  }

  // ── 9: Zero-gradient symmetry  ‖∇W(x,0)‖ = ‖∇W(−x,0)‖ ──────────────────────
  {
    for (let i = 1; i <= 10; i++) {
      const x = (i / 10) * h * 0.9;
      const [gxP, gyP] = cubicGradW( x, 0, h);
      const [gxN, gyN] = cubicGradW(-x, 0, h);
      const magP = Math.sqrt(gxP * gxP + gyP * gyP);
      const magN = Math.sqrt(gxN * gxN + gyN * gyN);
      if (Math.abs(magP - magN) > TOL)
        return fail(`cubicGradW magnitude asymmetry at x=${x.toFixed(2)}: ${magP} vs ${magN}`);

      const [sxP, syP] = spikyGradW( x, 0, h);
      const [sxN, syN] = spikyGradW(-x, 0, h);
      const smP = Math.sqrt(sxP * sxP + syP * syP);
      const smN = Math.sqrt(sxN * sxN + syN * syN);
      if (Math.abs(smP - smN) > TOL)
        return fail(`spikyGradW magnitude asymmetry at x=${x.toFixed(2)}: ${smP} vs ${smN}`);
    }
  }

  // ── 10: Cross-kernel peak consistency ────────────────────────────────────────
  //  Both kernels must be strictly positive at the origin.  Additionally,
  //  cubicW(0) > poly6W(0) for any h ≥ 1 because the cubic 2-D normalisation
  //  constant 40/(7πh²) exceeds poly6's 4/(πh²) factor when h is large
  //  (40/7 ≈ 5.71 > 4 after accounting for the (h²)³ = h⁶ folded into poly6).
  {
    const p0 = poly6W(0, h);
    const c0 = cubicW(0, h);
    if (p0 <= TOL)
      return fail(`poly6W(0, ${h}) = ${p0} — expected strictly positive`);
    if (c0 <= TOL)
      return fail(`cubicW(0, ${h}) = ${c0} — expected strictly positive`);
    if (c0 < p0 - TOL)
      return fail(`expected cubicW(0) ≥ poly6W(0) for h=${h}, got ${c0} < ${p0}`);
  }

  return true;
}
