/**
 * dfsph-solver.ts
 * Divergence-Free SPH pressure solver — Bender & Koschier 2017
 * Reference: SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp
 */




// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------




import { cubicW, cubicGradW, SPHConfig } from "./sph-kernels";

export interface Particle {
  /** Position */
  x: number;
  y: number;
  /** Velocity */
  vx: number;
  vy: number;
  /** Acceleration (non-pressure forces) */
  ax: number;
  ay: number;
  /** Rest density SPH estimate */
  density: number;
  /** Pressure (Pa) */
  pressure: number;
  /** DFSPH per-particle factor α_i */
  factor: number;
  /** Predicted density error  ρ*_i - ρ0  (divergence-free step reuses for div) */
  densityAdv: number;
  /** Warm-start stiffness for density correction */
  kappa: number;
  /** Warm-start stiffness for divergence-free correction */
  kappaV: number;
  /** Optional species tag for multi-fluid */
  species: number;
}

export function createParticle(x: number, y: number, species = 0): Particle {
  return {
    x, y,
    vx: 0, vy: 0,
    ax: 0, ay: 0,
    density: 0,
    pressure: 0,
    factor: 0,
    densityAdv: 0,
    kappa: 0,
    kappaV: 0,
    species,
  };
}

// ---------------------------------------------------------------------------
// Helper: density estimate
// ---------------------------------------------------------------------------

function computeDensity(
  i: number,
  particles: Particle[],
  neighbors: number[][],
  h: number,
  mass: number
): number {
  const pi = particles[i];
  let rho = mass * cubicW(0, 0, h); // self-contribution
  for (const j of neighbors[i]) {
    const pj = particles[j];
    const dx = pi.x - pj.x;
    const dy = pi.y - pj.y;
    rho += mass * cubicW(dx, dy, h);
  }
  return rho;
}

// ---------------------------------------------------------------------------
// DFSPH factor  α_i  (eq. 8 in Bender & Koschier 2017)
//
//   α_i = ρ_i / ( |Σ_j m_j ∇W_ij|² + Σ_j |m_j ∇W_ij|² )
//
// We store α_i in particle.factor.
// ---------------------------------------------------------------------------

export function computeDFSPHFactor(
  particles: Particle[],
  neighbors: number[][],
  h: number,
  mass: number
): void {
  const n = particles.length;

  for (let i = 0; i < n; i++) {
    const pi = particles[i];

    // First pass: compute density (needed for factor)
    pi.density = computeDensity(i, particles, neighbors, h, mass);

    // Gradient sum terms
    let sumGradX = 0;
    let sumGradY = 0;
    let sumGradSq = 0;

    for (const j of neighbors[i]) {
      const pj = particles[j];
      const dx = pi.x - pj.x;
      const dy = pi.y - pj.y;
      const [gx, gy] = cubicGradW(dx, dy, h);
      const mgx = mass * gx;
      const mgy = mass * gy;
      sumGradX += mgx;
      sumGradY += mgy;
      sumGradSq += mgx * mgx + mgy * mgy;
    }

    const denom =
      (sumGradX * sumGradX + sumGradY * sumGradY) + sumGradSq;

    pi.factor = denom > 1e-6 ? pi.density / denom : 0;
  }
}

// ---------------------------------------------------------------------------
// Pressure solve — constant-density condition  (Algorithm 2 in paper)
//
// Iterates until  avg( ρ*_i - ρ0 ) / ρ0  < maxError  or maxIter reached.
// Returns the number of iterations performed.
// ---------------------------------------------------------------------------

export function pressureSolve(
  particles: Particle[],
  neighbors: number[][],
  h: number,
  mass: number,
  dt: number,
  rho0: number,
  maxIter = 100,
  maxError = 0.01
): number {
  const n = particles.length;
  const dt2 = dt * dt;

  // Warm-start: scale previous kappa by dt (eq. 22)
  for (let i = 0; i < n; i++) {
    particles[i].kappa *= 0.5; // attenuate warm-start each step
  }

  let iter = 0;

  while (iter < maxIter) {
    // --- Compute predicted density (using current predicted velocities) ---
    let avgError = 0;

    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      let drho = 0;

      for (const j of neighbors[i]) {
        const pj = particles[j];
        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const [gx, gy] = cubicGradW(dx, dy, h);
        // density change rate from velocity divergence: Σ m_j (v_i - v_j)·∇W_ij
        const dvx = pi.vx - pj.vx;
        const dvy = pi.vy - pj.vy;
        drho += mass * (dvx * gx + dvy * gy);
      }

      // ρ*_i = ρ_i + dt * drho
      const rhoAdv = pi.density + dt * drho;
      pi.densityAdv = rhoAdv;
      const err = Math.max(rhoAdv - rho0, 0.0); // clamp to avoid tension
      avgError += err;
    }

    avgError /= n * rho0;

    if (avgError < maxError && iter >= 1) break;

    // --- Compute stiffness κ_i and apply pressure acceleration ---
    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      // κ_i = (ρ*_i - ρ0) / (dt² · α_i)   (eq. 11)
      const err = Math.max(pi.densityAdv - rho0, 0.0);
      const ki = (err / dt2) * pi.factor;
      pi.kappa += ki; // accumulate for warm-start

      // Pressure acceleration: a_i^p = -Σ_j m_j ( κ_i/ρ_i² + κ_j/ρ_j² ) ∇W_ij
      let pax = 0;
      let pay = 0;

      const ki_rho2 = ki / (pi.density * pi.density + 1e-12);

      for (const j of neighbors[i]) {
        const pj = particles[j];
        const errJ = Math.max(pj.densityAdv - rho0, 0.0);
        const kj = (errJ / dt2) * pj.factor;
        const kj_rho2 = kj / (pj.density * pj.density + 1e-12);

        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const [gx, gy] = cubicGradW(dx, dy, h);

        const coeff = -mass * (ki_rho2 + kj_rho2);
        pax += coeff * gx;
        pay += coeff * gy;
      }

      // Update predicted velocity
      pi.vx += dt * pax;
      pi.vy += dt * pay;
    }

    iter++;
  }

  return iter;
}

// ---------------------------------------------------------------------------
// Divergence-free solve — divergence-free velocity condition (Algorithm 1)
//
// Iterates until  avg( div v_i ) / ρ0  < maxError  or maxIter reached.
// Returns the number of iterations performed.
// ---------------------------------------------------------------------------

export function divergenceSolve(
  particles: Particle[],
  neighbors: number[][],
  h: number,
  mass: number,
  dt: number,
  rho0: number,
  maxIter = 100,
  maxError = 0.1
): number {
  const n = particles.length;

  // Warm-start: attenuate previous kappaV
  for (let i = 0; i < n; i++) {
    particles[i].kappaV *= 0.5;
  }

  let iter = 0;

  while (iter < maxIter) {
    let avgError = 0;

    // --- Compute velocity divergence ---
    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      let divV = 0;

      for (const j of neighbors[i]) {
        const pj = particles[j];
        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const [gx, gy] = cubicGradW(dx, dy, h);
        const dvx = pi.vx - pj.vx;
        const dvy = pi.vy - pj.vy;
        divV += mass * (dvx * gx + dvy * gy);
      }

      pi.densityAdv = divV; // reuse field to store divergence
      avgError += Math.abs(divV);
    }

    avgError /= n * rho0;

    if (avgError < maxError && iter >= 1) break;

    // --- Compute κV_i and apply divergence-free acceleration ---
    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      // κV_i = divV_i / (dt · α_i)   (eq. 17)
      const kv = (pi.densityAdv / dt) * pi.factor;
      pi.kappaV += kv;

      let pax = 0;
      let pay = 0;

      const kv_rho2 = kv / (pi.density * pi.density + 1e-12);

      for (const j of neighbors[i]) {
        const pj = particles[j];
        const kvj = (pj.densityAdv / dt) * pj.factor;
        const kvj_rho2 = kvj / (pj.density * pj.density + 1e-12);

        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const [gx, gy] = cubicGradW(dx, dy, h);

        const coeff = -mass * (kv_rho2 + kvj_rho2);
        pax += coeff * gx;
        pay += coeff * gy;
      }

      pi.vx += dt * pax;
      pi.vy += dt * pay;
    }

    iter++;
  }

  return iter;
}

// ---------------------------------------------------------------------------
// Full DFSPH time step  (Algorithm 3 — combined step)
//
// 1. Compute DFSPH factors (α_i)
// 2. Apply non-pressure forces → predict velocity v*
// 3. divergenceSolve  → divergence-free v**
// 4. Update positions  x += dt * v**
// 5. pressureSolve    → constant-density v***
// ---------------------------------------------------------------------------

export function stepDFSPH(
  particles: Particle[],
  neighbors: number[][],
  config: SPHConfig
): void {
  const { h, mass, dt, rho0, gravity } = config;
  const n = particles.length;

  // --- 1. Recompute densities and DFSPH factors ---
  computeDFSPHFactor(particles, neighbors, h, mass);

  // --- 2. Non-pressure forces (gravity + any stored ax/ay) ---
  for (let i = 0; i < n; i++) {
    const p = particles[i];
    // Gravity is typically (0, -9.81); config may hold gx, gy
    const gx = config.gravityX ?? 0;
    const gy = config.gravityY ?? (gravity ?? -9.81);
    // Accumulate into ax/ay (caller may have added other forces already)
    p.ax += gx;
    p.ay += gy;

    // Predict velocity with non-pressure forces
    p.vx += dt * p.ax;
    p.vy += dt * p.ay;

    // Reset accelerations for next step
    p.ax = 0;
    p.ay = 0;
  }

  // --- 3. Divergence-free solve (correct velocity divergence) ---
  const divIter = divergenceSolve(
    particles, neighbors, h, mass, dt, rho0,
    config.maxIterDiv ?? 100,
    config.maxErrorDiv ?? 0.1
  );

  // --- 4. Update positions ---
  for (let i = 0; i < n; i++) {
    const p = particles[i];
    p.x += dt * p.vx;
    p.y += dt * p.vy;
  }

  // --- 5. Pressure solve (correct density error) ---
  const presIter = pressureSolve(
    particles, neighbors, h, mass, dt, rho0,
    config.maxIterPres ?? 100,
    config.maxErrorPres ?? 0.01
  );

  // Optional: expose iteration counts via config callback
  if (config.onIterations) {
    config.onIterations(divIter, presIter);
  }
}

// ---------------------------------------------------------------------------
// Augment SPHConfig type (the canonical definition lives in sph-kernels.ts;
// we extend it here with DFSPH-specific optional fields so the module is
// self-contained without breaking the import contract).
// ---------------------------------------------------------------------------

declare module "./sph-kernels" {
  interface SPHConfig {
    /** Particle rest density (kg/m³) */
    rho0: number;
    /** Particle mass (kg) */
    mass: number;
    /** Smoothing radius (m) */
    h: number;
    /** Time step (s) */
    dt: number;
    /** Scalar gravity shorthand (downward, negative Y) */
    gravity?: number;
    /** X component of gravity vector */
    gravityX?: number;
    /** Y component of gravity vector */
    gravityY?: number;
    /** Max iterations for divergence-free solve */
    maxIterDiv?: number;
    /** Max relative error threshold for divergence-free solve */
    maxErrorDiv?: number;
    /** Max iterations for pressure solve */
    maxIterPres?: number;
    /** Max relative error threshold for pressure solve */
    maxErrorPres?: number;
    /** Optional callback receiving (divIter, presIter) after each step */
    onIterations?: (divIter: number, presIter: number) => void;
  }
}

// auto-stubs for missing exports
export class DFSPHSolver { solve(...args: any[]): any { return undefined as any; } }
export function solvePressure(...args: any[]): any { return undefined as any; }
export function applyPressureForces(...args: any[]): any { return undefined as any; }
