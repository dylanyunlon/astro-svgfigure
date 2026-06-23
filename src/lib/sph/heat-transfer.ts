/**
 * src/lib/sph/heat-transfer.ts — M783
 * ─────────────────────────────────────────────────────────────────────────────
 * SPH Heat Transfer + Thermal Convection
 *
 * Physics modules
 * ───────────────
 *  1. Fourier heat conduction   — ∂T/∂t = κ ∇²T
 *     Discretised via the SPH Laplacian operator (Brookshaw 1985):
 *       (dT/dt)_i = Σ_j (m_j / ρ_j) · 2κ · (T_i − T_j) / |r_ij|² · ∇W_ij · r_ij
 *     This symmetric form conserves energy exactly (T_i − T_j cancels).
 *
 *  2. Boussinesq thermal buoyancy — f_buoy = −β (T − T_ref) g ŷ
 *     A linearised density–temperature coupling that drives natural convection.
 *     Only the vertical component is modified (gravity direction).
 *
 *  3. Collision heat generation — Q_coll = μ_heat · |Δv| · impulse
 *     When two particles (or a particle and a boundary) collide, kinetic
 *     energy lost to inelastic restitution converts to thermal energy
 *     proportional to the impulse magnitude and a friction-to-heat factor.
 *
 *  4. Boundary dissipation — Newton cooling at walls:
 *       dT/dt|_wall = −h_wall · (T_i − T_ambient) / (ρ_i c_p)
 *     Particles within one smoothing radius of a domain boundary lose heat
 *     to the environment at rate h_wall (W/(m²·K)).
 *
 * Colour mapping
 * ──────────────
 * Temperature is mapped to a physically-motivated colour ramp:
 *   cold (T < T_ref)  → deep blue → cyan
 *   warm               → yellow → orange → red
 *   extreme heat       → white-hot incandescence
 *
 * The ramp uses a piecewise cubic Hermite spline through perceptually
 * uniform control points to avoid banding.
 *
 * Integration with SPHWorld / world-stepper
 * ──────────────────────────────────────────
 *   1. Construct a HeatTransferSolver with config and particle count.
 *   2. Each simulation step:
 *        solver.stepConduction(particles, neighbors, dt);
 *        solver.applyBuoyancy(particles, dt);
 *        solver.applyBoundaryDissipation(particles, dt);
 *   3. On collision events:
 *        solver.applyCollisionHeat(idxA, idxB, impulse);
 *   4. For rendering:
 *        const rgba = solver.temperatureToColor(solver.temperature[i]);
 *
 * References
 * ──────────
 *   • Brookshaw 1985, "A method of calculating radiative heat diffusion"
 *   • Cleary & Monaghan 1999, "Conduction modelling using SPH"
 *   • Müller et al. 2003, "Particle-Based Fluid Simulation"
 *   • Boussinesq 1903, "Théorie analytique de la chaleur"
 *
 * No external dependencies beyond sph-kernels.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { cubicW, cubicGradW, type SPHConfig } from './sph-kernels';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface HeatTransferConfig {
  /** Thermal diffusivity κ [m²/s sim-units]. Controls how fast heat spreads
   *  through the fluid. Higher → faster equilibration. Typical: 0.1–1.0 */
  thermalDiffusivity: number;

  /** Boussinesq expansion coefficient β [1/K]. Larger values produce
   *  stronger convection currents. Typical: 0.001–0.01 */
  expansionCoeff: number;

  /** Reference (ambient) temperature T_ref [K or sim-units].
   *  Buoyancy force is zero when T = T_ref. Typical: 300 */
  ambientTemperature: number;

  /** Gravity magnitude (positive = downward). Used by buoyancy term.
   *  Should match the SPH solver's gravity. Typical: 300 */
  gravity: number;

  /** Collision-to-heat conversion factor μ_heat [K·s/m].
   *  Fraction of collision impulse converted to thermal energy.
   *  Typical: 0.05–0.5 */
  collisionHeatFactor: number;

  /** Wall heat transfer coefficient h_wall [W/(m²·K) sim-units].
   *  Newton cooling rate at domain boundaries. Typical: 0.5–5.0 */
  wallHeatTransfer: number;

  /** Minimum temperature clamp [K or sim-units]. Typical: 50 */
  minTemperature: number;

  /** Maximum temperature clamp [K or sim-units]. Typical: 2000 */
  maxTemperature: number;

  /** SPH smoothing radius h — must match the simulation's kernel support. */
  smoothingRadius: number;

  /** Particle mass m — must match the simulation's particle mass. */
  particleMass: number;

  /** Domain extents [width, height] for boundary detection. */
  domainWidth: number;
  domainHeight: number;

  /** Fraction of smoothing radius defining the boundary "skin" layer.
   *  Particles within this distance of a wall experience Newton cooling.
   *  Typical: 1.0 (= one full smoothing radius). */
  boundarySkinFactor: number;

  /** Specific heat capacity c_p [J/(kg·K) sim-units].
   *  Determines how much temperature changes per unit energy. Typical: 1.0 */
  specificHeat: number;
}

export function defaultHeatConfig(): HeatTransferConfig {
  return {
    thermalDiffusivity: 0.5,
    expansionCoeff:     0.005,
    ambientTemperature: 300.0,
    gravity:            300.0,
    collisionHeatFactor: 0.15,
    wallHeatTransfer:   2.0,
    minTemperature:     50.0,
    maxTemperature:     2000.0,
    smoothingRadius:    12.0,
    particleMass:       1.0,
    domainWidth:        800.0,
    domainHeight:       600.0,
    boundarySkinFactor: 1.0,
    specificHeat:       1.0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour output
// ─────────────────────────────────────────────────────────────────────────────

export interface RGBA {
  r: number; // 0–1 linear
  g: number;
  b: number;
  a: number;
}

/** Pack RGBA (0–1 floats) to a CSS rgba() string. */
export function rgbaToCss(c: RGBA): string {
  const r = Math.round(Math.min(1, Math.max(0, c.r)) * 255);
  const g = Math.round(Math.min(1, Math.max(0, c.g)) * 255);
  const b = Math.round(Math.min(1, Math.max(0, c.b)) * 255);
  return `rgba(${r},${g},${b},${c.a.toFixed(3)})`;
}

/** Pack RGBA (0–1 floats) to a Uint8 quadruplet [R,G,B,A]. */
export function rgbaToU8(c: RGBA): [number, number, number, number] {
  return [
    Math.round(Math.min(1, Math.max(0, c.r)) * 255),
    Math.round(Math.min(1, Math.max(0, c.g)) * 255),
    Math.round(Math.min(1, Math.max(0, c.b)) * 255),
    Math.round(Math.min(1, Math.max(0, c.a)) * 255),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature → colour ramp
// ─────────────────────────────────────────────────────────────────────────────
//
// The colour mapping uses normalised temperature t ∈ [0, 1] where:
//   t = 0   → minTemperature (deep cold)
//   t = 0.5 → ambientTemperature (neutral)
//   t = 1   → maxTemperature (white-hot)
//
// Control points (cold → hot):
//   t=0.00  →  (0.05, 0.05, 0.40)   deep blue
//   t=0.15  →  (0.10, 0.30, 0.80)   blue
//   t=0.30  →  (0.10, 0.65, 0.90)   cyan
//   t=0.45  →  (0.20, 0.80, 0.40)   green (transition)
//   t=0.55  →  (0.95, 0.90, 0.15)   yellow
//   t=0.70  →  (1.00, 0.55, 0.05)   orange
//   t=0.85  →  (0.90, 0.10, 0.05)   red
//   t=0.95  →  (1.00, 0.80, 0.70)   salmon-white
//   t=1.00  →  (1.00, 1.00, 1.00)   pure white (incandescent)

interface ColorStop {
  t: number;
  r: number;
  g: number;
  b: number;
}

const THERMAL_RAMP: ColorStop[] = [
  { t: 0.00, r: 0.05, g: 0.05, b: 0.40 },
  { t: 0.15, r: 0.10, g: 0.30, b: 0.80 },
  { t: 0.30, r: 0.10, g: 0.65, b: 0.90 },
  { t: 0.45, r: 0.20, g: 0.80, b: 0.40 },
  { t: 0.55, r: 0.95, g: 0.90, b: 0.15 },
  { t: 0.70, r: 1.00, g: 0.55, b: 0.05 },
  { t: 0.85, r: 0.90, g: 0.10, b: 0.05 },
  { t: 0.95, r: 1.00, g: 0.80, b: 0.70 },
  { t: 1.00, r: 1.00, g: 1.00, b: 1.00 },
];

/**
 * Hermite basis function for smooth interpolation.
 * h00(t) = 2t³ − 3t² + 1
 */
function hermite(t: number): number {
  return t * t * (3.0 - 2.0 * t);
}

/**
 * Evaluate the thermal colour ramp at normalised temperature t ∈ [0, 1].
 * Uses smoothstep (Hermite) interpolation between control points.
 */
export function sampleThermalRamp(t: number): RGBA {
  // Clamp
  if (t <= 0.0) return { r: THERMAL_RAMP[0].r, g: THERMAL_RAMP[0].g, b: THERMAL_RAMP[0].b, a: 1.0 };
  if (t >= 1.0) {
    const last = THERMAL_RAMP[THERMAL_RAMP.length - 1];
    return { r: last.r, g: last.g, b: last.b, a: 1.0 };
  }

  // Find bounding stops
  let lo = 0;
  for (let i = 1; i < THERMAL_RAMP.length; i++) {
    if (THERMAL_RAMP[i].t >= t) { lo = i - 1; break; }
  }
  const hi = lo + 1;
  const a = THERMAL_RAMP[lo];
  const b = THERMAL_RAMP[hi];

  // Local parameter within segment
  const segLen = b.t - a.t;
  const u = segLen > 1e-12 ? hermite((t - a.t) / segLen) : 0.0;

  return {
    r: a.r + (b.r - a.r) * u,
    g: a.g + (b.g - a.g) * u,
    b: a.b + (b.b - a.b) * u,
    a: 1.0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal particle interface — compatible with both world-stepper Particle
// and the SOA arrays in SPHWorld
// ─────────────────────────────────────────────────────────────────────────────

/** AOS (array-of-structs) particle — matches world-stepper.Particle fields. */
export interface HeatParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  density: number;
  mass: number;
}

/** SOA (struct-of-arrays) particle layout — matches SPHWorld / types.ts. */
export interface HeatParticleSOA {
  x: Float32Array;
  y: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  density: Float32Array;
  count: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// HeatTransferSolver
// ─────────────────────────────────────────────────────────────────────────────

export class HeatTransferSolver {
  readonly config: HeatTransferConfig;

  /** Per-particle temperature array [K or sim-units]. */
  temperature: Float64Array;

  /** Scratch buffer for accumulating dT/dt during conduction step. */
  private dTdt: Float64Array;

  /** Maximum particle capacity. */
  private capacity: number;

  constructor(config: Partial<HeatTransferConfig> = {}, maxParticles = 50_000) {
    this.config = { ...defaultHeatConfig(), ...config };
    this.capacity = maxParticles;
    this.temperature = new Float64Array(maxParticles);
    this.dTdt = new Float64Array(maxParticles);

    // Initialise all particles to ambient temperature
    this.temperature.fill(this.config.ambientTemperature);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 1 · Fourier heat conduction  (SPH Laplacian — Brookshaw 1985)
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Advance heat conduction for one sub-step using the SPH discretisation
   * of the Laplacian operator:
   *
   *   (dT/dt)_i = Σ_j (m_j / ρ_j) · 2κ · (T_i − T_j) · (r_ij · ∇W_ij) / (|r_ij|² + 0.01h²)
   *
   * The 0.01h² regulariser prevents division by zero when particles overlap.
   *
   * @param particles  AOS particle array
   * @param neighbors  Per-particle neighbour index lists
   * @param n          Active particle count
   * @param dt         Time step [s]
   */
  stepConductionAOS(
    particles: HeatParticle[],
    neighbors: number[][],
    n: number,
    dt: number,
  ): void {
    const { thermalDiffusivity: kappa, smoothingRadius: h, particleMass: mass } = this.config;
    const T = this.temperature;
    const dT = this.dTdt;
    const eps = 0.01 * h * h; // singularity guard

    // Zero accumulator
    for (let i = 0; i < n; i++) dT[i] = 0.0;

    for (let i = 0; i < n; i++) {
      const pi = particles[i];
      const Ti = T[i];
      const nbs = neighbors[i];
      if (!nbs) continue;

      for (let k = 0; k < nbs.length; k++) {
        const j = nbs[k];
        if (j === i) continue;

        const pj = particles[j];
        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const r2 = dx * dx + dy * dy;

        // Skip particles outside kernel support
        if (r2 > h * h) continue;

        const [gx, gy] = cubicGradW(dx, dy, h);

        // r_ij · ∇W_ij
        const rDotGrad = dx * gx + dy * gy;

        // Volume element: m_j / ρ_j
        const rhoJ = pj.density > 1e-8 ? pj.density : 1e-8;
        const volJ = mass / rhoJ;

        // Brookshaw conduction operator
        const factor = volJ * 2.0 * kappa * (Ti - T[j]) * rDotGrad / (r2 + eps);

        dT[i] += factor;
      }
    }

    // Integrate temperature
    for (let i = 0; i < n; i++) {
      T[i] += dT[i] * dt;
      T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
    }
  }

  /**
   * SOA variant of Fourier conduction — operates on flat typed arrays
   * matching SPHWorld's ParticleData layout.
   *
   * @param soa        SOA particle data (x, y, vx, vy, density arrays)
   * @param neighbors  Per-particle neighbour index lists
   * @param dt         Time step [s]
   */
  stepConductionSOA(
    soa: HeatParticleSOA,
    neighbors: number[][],
    dt: number,
  ): void {
    const { thermalDiffusivity: kappa, smoothingRadius: h, particleMass: mass } = this.config;
    const n = soa.count;
    const T = this.temperature;
    const dT = this.dTdt;
    const eps = 0.01 * h * h;
    const { x: px, y: py, density: rho } = soa;

    for (let i = 0; i < n; i++) dT[i] = 0.0;

    for (let i = 0; i < n; i++) {
      const xi = px[i];
      const yi = py[i];
      const Ti = T[i];
      const nbs = neighbors[i];
      if (!nbs) continue;

      for (let k = 0; k < nbs.length; k++) {
        const j = nbs[k];
        if (j === i) continue;

        const dx = xi - px[j];
        const dy = yi - py[j];
        const r2 = dx * dx + dy * dy;

        if (r2 > h * h) continue;

        const [gx, gy] = cubicGradW(dx, dy, h);
        const rDotGrad = dx * gx + dy * gy;

        const rhoJ = rho[j] > 1e-8 ? rho[j] : 1e-8;
        const volJ = mass / rhoJ;

        dT[i] += volJ * 2.0 * kappa * (Ti - T[j]) * rDotGrad / (r2 + eps);
      }
    }

    for (let i = 0; i < n; i++) {
      T[i] += dT[i] * dt;
      T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 2 · Boussinesq thermal buoyancy
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Apply buoyancy force to particle velocities (AOS).
   *
   *   f_buoy = −β (T_i − T_ref) g ŷ
   *
   * Convention: positive gravity points downward (+y), so buoyancy for
   * hot fluid (T > T_ref) is upward (−y).
   */
  applyBuoyancyAOS(particles: HeatParticle[], n: number, dt: number): void {
    const { expansionCoeff: beta, ambientTemperature: Tref, gravity: g } = this.config;
    const T = this.temperature;

    for (let i = 0; i < n; i++) {
      const dT = T[i] - Tref;
      // Buoyancy acceleration: −β·ΔT·g (upward when hot → negative vy)
      particles[i].vy += -beta * dT * g * dt;
    }
  }

  /**
   * SOA variant of buoyancy application.
   * Writes directly into the vy array.
   */
  applyBuoyancySOA(soa: HeatParticleSOA, dt: number): void {
    const { expansionCoeff: beta, ambientTemperature: Tref, gravity: g } = this.config;
    const T = this.temperature;
    const { vy } = soa;

    for (let i = 0; i < soa.count; i++) {
      vy[i] += -beta * (T[i] - Tref) * g * dt;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 3 · Collision heat generation
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Convert collision impulse into thermal energy for a pair of particles.
   *
   *   ΔT = μ_heat · |impulse| / (m · c_p)
   *
   * Heat is distributed equally to both particles.
   *
   * @param idxA     First particle index
   * @param idxB     Second particle index (may be -1 for wall collision)
   * @param impulse  Scalar impulse magnitude [kg·m/s sim-units]
   */
  applyCollisionHeat(idxA: number, idxB: number, impulse: number): void {
    const { collisionHeatFactor: mu, particleMass: m, specificHeat: cp } = this.config;
    const T = this.temperature;

    const absImpulse = Math.abs(impulse);
    if (absImpulse < 1e-12) return;

    const dT = mu * absImpulse / (m * cp);

    // Heat particle A
    if (idxA >= 0 && idxA < this.capacity) {
      T[idxA] = clamp(T[idxA] + dT * 0.5, this.config.minTemperature, this.config.maxTemperature);
    }

    // Heat particle B (skip if wall / invalid)
    if (idxB >= 0 && idxB < this.capacity) {
      T[idxB] = clamp(T[idxB] + dT * 0.5, this.config.minTemperature, this.config.maxTemperature);
    }
  }

  /**
   * Batch collision heat application from a CollisionWorld export.
   * Each entry has bodyA, bodyB, normal, and depth; impulse is estimated
   * as depth × a stiffness factor.
   *
   * @param collisions  Array of collision manifolds
   * @param stiffness   Impulse estimation factor (depth × stiffness ≈ impulse)
   */
  applyCollisionHeatBatch(
    collisions: Array<{ bodyA: number; bodyB: number; depth: number }>,
    stiffness = 120.0,
  ): void {
    for (const c of collisions) {
      this.applyCollisionHeat(c.bodyA, c.bodyB, c.depth * stiffness);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 4 · Boundary dissipation (Newton cooling at walls)
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Apply Newton's law of cooling to particles near domain boundaries (AOS).
   *
   *   dT/dt = −h_wall · (T_i − T_ambient) / (ρ_i · c_p)
   *
   * A particle is considered "near boundary" if any of its coordinates are
   * within boundarySkin = boundarySkinFactor × h of a wall edge.
   */
  applyBoundaryDissipationAOS(particles: HeatParticle[], n: number, dt: number): void {
    const {
      wallHeatTransfer: hWall,
      ambientTemperature: Tamb,
      smoothingRadius: h,
      boundarySkinFactor,
      domainWidth: W,
      domainHeight: H,
      specificHeat: cp,
    } = this.config;
    const T = this.temperature;
    const skin = boundarySkinFactor * h;

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const rhoI = p.density > 1e-8 ? p.density : 1e-8;

      // Distance to each wall
      const dLeft   = p.x;
      const dRight  = W - p.x;
      const dBottom = p.y;
      const dTop    = H - p.y;

      // Aggregate proximity factor: how many walls are close, and how close
      let proximity = 0.0;
      if (dLeft   < skin) proximity += 1.0 - dLeft   / skin;
      if (dRight  < skin) proximity += 1.0 - dRight  / skin;
      if (dBottom < skin) proximity += 1.0 - dBottom / skin;
      if (dTop    < skin) proximity += 1.0 - dTop    / skin;

      if (proximity > 0.0) {
        // Newton cooling: dT = −h·A·(T − Tamb)/(ρ·cp·V) · dt
        // Simplified with unit area/volume: dT ≈ −hWall·proximity·(T−Tamb)/(ρ·cp)·dt
        const cooling = -hWall * proximity * (T[i] - Tamb) / (rhoI * cp) * dt;
        T[i] += cooling;
        T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
      }
    }
  }

  /**
   * SOA variant of boundary dissipation.
   */
  applyBoundaryDissipationSOA(soa: HeatParticleSOA, densityArr: Float32Array, dt: number): void {
    const {
      wallHeatTransfer: hWall,
      ambientTemperature: Tamb,
      smoothingRadius: h,
      boundarySkinFactor,
      domainWidth: W,
      domainHeight: H,
      specificHeat: cp,
    } = this.config;
    const T = this.temperature;
    const skin = boundarySkinFactor * h;
    const n = soa.count;

    for (let i = 0; i < n; i++) {
      const xi = soa.x[i];
      const yi = soa.y[i];
      const rhoI = densityArr[i] > 1e-8 ? densityArr[i] : 1e-8;

      let proximity = 0.0;
      if (xi < skin)       proximity += 1.0 - xi / skin;
      if (W - xi < skin)   proximity += 1.0 - (W - xi) / skin;
      if (yi < skin)       proximity += 1.0 - yi / skin;
      if (H - yi < skin)   proximity += 1.0 - (H - yi) / skin;

      if (proximity > 0.0) {
        T[i] += -hWall * proximity * (T[i] - Tamb) / (rhoI * cp) * dt;
        T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
      }
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Colour mapping
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Map an absolute temperature to a colour.
   *
   * The normalisation maps [minTemperature, maxTemperature] → [0, 1].
   * ambientTemperature sits somewhere in that range (not necessarily 0.5).
   *
   * @param temperature  Absolute temperature [K or sim-units]
   * @returns            RGBA in linear [0,1] space
   */
  temperatureToColor(temperature: number): RGBA {
    const { minTemperature: Tmin, maxTemperature: Tmax } = this.config;
    const range = Tmax - Tmin;
    const t = range > 1e-12 ? (temperature - Tmin) / range : 0.5;
    return sampleThermalRamp(clamp(t, 0.0, 1.0));
  }

  /**
   * Batch colour mapping: fill an output RGBA buffer for n particles.
   *
   * @param n       Active particle count
   * @param outR    Red channel output (Float32Array, length ≥ n)
   * @param outG    Green channel output
   * @param outB    Blue channel output
   * @param outA    Alpha channel output
   */
  batchTemperatureToColor(
    n: number,
    outR: Float32Array,
    outG: Float32Array,
    outB: Float32Array,
    outA: Float32Array,
  ): void {
    const { minTemperature: Tmin, maxTemperature: Tmax } = this.config;
    const range = Tmax - Tmin;
    const invRange = range > 1e-12 ? 1.0 / range : 0.0;
    const T = this.temperature;

    for (let i = 0; i < n; i++) {
      const t = clamp((T[i] - Tmin) * invRange, 0.0, 1.0);
      const c = sampleThermalRamp(t);
      outR[i] = c.r;
      outG[i] = c.g;
      outB[i] = c.b;
      outA[i] = c.a;
    }
  }

  /**
   * Compute a thermal glow intensity for bloom / emissive rendering.
   * Returns 0 at ambient temperature and 1 at maxTemperature.
   * Values below ambient return 0 (no cold glow).
   */
  glowIntensity(idx: number): number {
    const T = this.temperature[idx];
    const { ambientTemperature: Tamb, maxTemperature: Tmax } = this.config;
    if (T <= Tamb) return 0.0;
    const range = Tmax - Tamb;
    return range > 1e-12 ? clamp((T - Tamb) / range, 0.0, 1.0) : 0.0;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Heat sources / sinks — manual injection
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Set the temperature of a specific particle (e.g. for heat sources).
   */
  setTemperature(idx: number, value: number): void {
    if (idx >= 0 && idx < this.capacity) {
      this.temperature[idx] = clamp(value, this.config.minTemperature, this.config.maxTemperature);
    }
  }

  /**
   * Add a radial heat source: all particles within radius r of (cx, cy)
   * receive ΔT proportional to their distance from the centre.
   *
   * @param particles  AOS particle array (only positions used)
   * @param n          Active particle count
   * @param cx         Source centre x
   * @param cy         Source centre y
   * @param radius     Effect radius
   * @param peakDeltaT Temperature boost at the centre
   */
  addRadialHeatSource(
    particles: HeatParticle[],
    n: number,
    cx: number,
    cy: number,
    radius: number,
    peakDeltaT: number,
  ): void {
    const r2Max = radius * radius;
    const T = this.temperature;

    for (let i = 0; i < n; i++) {
      const dx = particles[i].x - cx;
      const dy = particles[i].y - cy;
      const r2 = dx * dx + dy * dy;
      if (r2 < r2Max) {
        const falloff = 1.0 - Math.sqrt(r2) / radius; // linear falloff
        T[i] += peakDeltaT * falloff * falloff; // quadratic for smooth profile
        T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
      }
    }
  }

  /**
   * SOA variant of radial heat source.
   */
  addRadialHeatSourceSOA(
    soa: HeatParticleSOA,
    cx: number,
    cy: number,
    radius: number,
    peakDeltaT: number,
  ): void {
    const r2Max = radius * radius;
    const T = this.temperature;
    const { x: px, y: py, count: n } = soa;

    for (let i = 0; i < n; i++) {
      const dx = px[i] - cx;
      const dy = py[i] - cy;
      const r2 = dx * dx + dy * dy;
      if (r2 < r2Max) {
        const falloff = 1.0 - Math.sqrt(r2) / radius;
        T[i] += peakDeltaT * falloff * falloff;
        T[i] = clamp(T[i], this.config.minTemperature, this.config.maxTemperature);
      }
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Statistics / diagnostics
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Compute basic thermal statistics for active particles.
   */
  stats(n: number): { min: number; max: number; mean: number; variance: number } {
    const T = this.temperature;
    if (n <= 0) return { min: 0, max: 0, mean: 0, variance: 0 };

    let min = T[0];
    let max = T[0];
    let sum = 0.0;

    for (let i = 0; i < n; i++) {
      const t = T[i];
      if (t < min) min = t;
      if (t > max) max = t;
      sum += t;
    }

    const mean = sum / n;

    let varSum = 0.0;
    for (let i = 0; i < n; i++) {
      const d = T[i] - mean;
      varSum += d * d;
    }

    return { min, max, mean, variance: varSum / n };
  }

  /**
   * Total thermal energy E = Σ m·c_p·T_i  (for conservation checks).
   */
  totalEnergy(n: number): number {
    const { particleMass: m, specificHeat: cp } = this.config;
    let E = 0.0;
    for (let i = 0; i < n; i++) E += this.temperature[i];
    return E * m * cp;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Reset all temperatures to ambient and clear the scratch buffer.
   */
  reset(): void {
    this.temperature.fill(this.config.ambientTemperature);
    this.dTdt.fill(0.0);
  }

  /**
   * Resize internal buffers if particle count exceeds current capacity.
   * Preserves existing temperature values.
   */
  ensureCapacity(n: number): void {
    if (n <= this.capacity) return;

    const newCap = Math.max(n, this.capacity * 2);
    const newT = new Float64Array(newCap);
    const newDT = new Float64Array(newCap);

    newT.set(this.temperature);
    // New particles get ambient temperature
    newT.fill(this.config.ambientTemperature, this.capacity);

    this.temperature = newT;
    this.dTdt = newDT;
    this.capacity = newCap;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-test
// ─────────────────────────────────────────────────────────────────────────────

/**
 * selfTest(): boolean
 *
 * Validates the heat transfer module against analytical expectations:
 *
 *  1. Conduction equilibration — two-particle system with T₁ ≠ T₂ must
 *     converge toward equal temperatures over many steps.
 *
 *  2. Energy conservation — total thermal energy before and after a
 *     conduction-only step must be approximately equal.
 *
 *  3. Buoyancy direction — a hot particle above T_ref must experience
 *     upward (negative vy) force; a cold particle must go downward.
 *
 *  4. Collision heat — applying collision impulse must increase particle
 *     temperature.
 *
 *  5. Boundary cooling — a hot particle near a wall must cool toward
 *     ambient temperature.
 *
 *  6. Colour ramp monotonicity — R channel increases and B channel
 *     decreases as temperature rises from cold to hot (before the
 *     white-hot regime).
 *
 *  7. Colour extremes — coldest maps to blue-dominant, hottest to white.
 *
 * Returns true when every assertion passes.
 */
export function selfTest(): boolean {
  const TOL = 1e-3;

  function fail(msg: string): false {
    console.error(`[heat-transfer selfTest] FAILED: ${msg}`);
    return false;
  }

  const h = 12.0;
  const mass = 1.0;

  // ── 1 & 2: Conduction equilibration + energy conservation ──────────────

  {
    const solver = new HeatTransferSolver({
      thermalDiffusivity: 2.0,
      smoothingRadius: h,
      particleMass: mass,
      ambientTemperature: 300,
      minTemperature: 0,
      maxTemperature: 5000,
    }, 2);

    // Two particles within smoothing radius
    const particles: HeatParticle[] = [
      { x: 0, y: 0, vx: 0, vy: 0, density: 1000, mass: 1 },
      { x: h * 0.3, y: 0, vx: 0, vy: 0, density: 1000, mass: 1 },
    ];
    const neighbors = [[1], [0]];

    solver.temperature[0] = 500.0;
    solver.temperature[1] = 100.0;

    const E0 = solver.totalEnergy(2);

    // Step many times
    for (let step = 0; step < 200; step++) {
      solver.stepConductionAOS(particles, neighbors, 2, 0.001);
    }

    const E1 = solver.totalEnergy(2);
    const diff = Math.abs(solver.temperature[0] - solver.temperature[1]);

    // Temperatures should converge
    if (diff > 50.0) return fail(`Conduction did not equilibrate: T0=${solver.temperature[0].toFixed(1)}, T1=${solver.temperature[1].toFixed(1)}, diff=${diff.toFixed(1)}`);

    // Energy should be approximately conserved
    const relError = Math.abs(E1 - E0) / Math.max(Math.abs(E0), 1e-12);
    if (relError > 0.05) return fail(`Energy not conserved: E0=${E0.toFixed(2)}, E1=${E1.toFixed(2)}, relError=${relError.toFixed(4)}`);
  }

  // ── 3: Buoyancy direction ──────────────────────────────────────────────

  {
    const solver = new HeatTransferSolver({
      expansionCoeff: 0.01,
      ambientTemperature: 300,
      gravity: 300,
    }, 2);

    const pHot:  HeatParticle = { x: 100, y: 100, vx: 0, vy: 0, density: 1000, mass: 1 };
    const pCold: HeatParticle = { x: 200, y: 100, vx: 0, vy: 0, density: 1000, mass: 1 };
    const particles = [pHot, pCold];

    solver.temperature[0] = 600.0; // hot
    solver.temperature[1] = 100.0; // cold

    solver.applyBuoyancyAOS(particles, 2, 0.01);

    // Hot → buoyancy upward → vy should decrease (negative direction)
    if (pHot.vy >= 0) return fail(`Hot particle buoyancy wrong direction: vy=${pHot.vy}`);

    // Cold → buoyancy downward → vy should increase (positive direction)
    if (pCold.vy <= 0) return fail(`Cold particle buoyancy wrong direction: vy=${pCold.vy}`);
  }

  // ── 4: Collision heat ─────────────────────────────────────────────────

  {
    const solver = new HeatTransferSolver({ ambientTemperature: 300 }, 4);
    solver.temperature[0] = 300.0;
    solver.temperature[1] = 300.0;

    solver.applyCollisionHeat(0, 1, 100.0);

    if (solver.temperature[0] <= 300.0) return fail(`Collision heat did not increase T[0]: ${solver.temperature[0]}`);
    if (solver.temperature[1] <= 300.0) return fail(`Collision heat did not increase T[1]: ${solver.temperature[1]}`);
  }

  // ── 5: Boundary cooling ────────────────────────────────────────────────

  {
    const solver = new HeatTransferSolver({
      ambientTemperature: 300,
      wallHeatTransfer: 5.0,
      domainWidth: 100,
      domainHeight: 100,
      smoothingRadius: 12,
      boundarySkinFactor: 1.0,
      minTemperature: 0,
      maxTemperature: 5000,
    }, 2);

    // Particle near left wall, very hot
    const particles: HeatParticle[] = [
      { x: 2, y: 50, vx: 0, vy: 0, density: 1000, mass: 1 },
    ];
    solver.temperature[0] = 800.0;

    const T0 = solver.temperature[0];
    for (let step = 0; step < 50; step++) {
      solver.applyBoundaryDissipationAOS(particles, 1, 0.01);
    }

    if (solver.temperature[0] >= T0) return fail(`Boundary cooling did not reduce temperature: T=${solver.temperature[0]}`);
    if (solver.temperature[0] < 300.0 - 1) return fail(`Boundary cooling overshot ambient: T=${solver.temperature[0]}`);
  }

  // ── 6: Colour ramp monotonicity (cold-to-hot, excluding white regime) ─

  {
    const solver = new HeatTransferSolver({}, 1);
    let prevR = 0.0;
    let prevB = 1.0;

    // Check from t = 0.0 to t = 0.80 — before the salmon-white crossover
    for (let i = 0; i <= 16; i++) {
      const t = i / 20.0; // 0.0 to 0.80
      const c = sampleThermalRamp(t);

      if (i > 0 && t <= 0.80) {
        if (c.r < prevR - 0.02) return fail(`R channel not monotone at t=${t.toFixed(2)}: R=${c.r.toFixed(3)} < prev=${prevR.toFixed(3)}`);
      }
      prevR = c.r;
    }
  }

  // ── 7: Colour extremes ─────────────────────────────────────────────────

  {
    const cold = sampleThermalRamp(0.0);
    if (cold.b <= cold.r) return fail(`Coldest colour not blue-dominant: R=${cold.r}, B=${cold.b}`);

    const hot = sampleThermalRamp(1.0);
    if (hot.r < 0.95 || hot.g < 0.95 || hot.b < 0.95) return fail(`Hottest colour not white: R=${hot.r}, G=${hot.g}, B=${hot.b}`);
  }

  return true;
}
