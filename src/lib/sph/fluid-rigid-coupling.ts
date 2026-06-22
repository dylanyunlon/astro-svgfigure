/**
 * fluid-rigid-coupling.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Akinci 2012 – Versatile Rigid-Fluid Coupling for Incompressible SPH
 * Reference: "Versatile Rigid-Fluid Coupling for Incompressible SPH",
 *            Akinci et al. 2012, ACM SIGGRAPH.
 *            Implementation mirrors SPlisHSPlasH/BoundaryModel_Akinci2012.cpp
 *
 * This module implements FULL bidirectional coupling:
 *
 *   (A) Fluid → Rigid:  SPH pressure + viscosity forces applied to the rigid
 *       body as linear force and torque.
 *
 *   (B) Rigid → Fluid:  Boundary particle positions/velocities updated from
 *       the rigid body state after each integration step; fluid density
 *       corrected with adaptive boundary-particle volumes.
 *
 *   (C) Adaptive volumes:  Ψ_b recomputed whenever a rigid body moves so that
 *       the boundary mass representation matches the local packing density.
 *
 * 2-D (x,y) coordinate system throughout.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { Particle } from './dfsph-solver';
import { RigidBody, sampleBoundaryParticles } from './rigid-body';
import { cubicW, cubicGradW } from './sph-kernels';

// ─── public re-export so callers can cache the volume table ──────────────────
export type BoundaryVolumeTable = Map<number, Float64Array>;

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 1 – Adaptive boundary-particle volumes  (Akinci 2012, Eq. 4)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Recompute the Akinci volume estimate Ψ_b for every boundary particle of the
 * given rigid body:
 *
 *   Ψ_b(x_b) = 1 / Σ_k  W(|x_b − x_k|, h)
 *
 * The sum runs over **all** boundary particles of the same body (including the
 * self-contribution at r = 0).  The inverse gives a volume proportional to the
 * local particle spacing, so that sparse regions near sharp corners are not
 * under-represented.
 *
 * Call this:
 *  • once after initial sampling, and
 *  • every simulation step in which the rigid body has moved / rotated
 *    (the world-space positions of boundary particles change with the body).
 *
 * @param rb  The rigid body whose boundary particles have already been placed
 *            by `sampleBoundaryParticles`.
 * @param h   SPH smoothing radius.
 * @returns   Float64Array of length `rb.boundaryCount` with per-particle Ψ_b.
 */
export function computeBoundaryVolumes(
  rb: RigidBody,
  h: number,
): Float64Array {
  const n = rb.boundaryCount;
  const volumes = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const xi = rb.boundaryX[i];
    const yi = rb.boundaryY[i];

    // Self-contribution W(0, h)
    let wSum = cubicW(0.0, h);

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const dx = xi - rb.boundaryX[k];
      const dy = yi - rb.boundaryY[k];
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r < h) {
        wSum += cubicW(r, h);
      }
    }

    volumes[i] = wSum > 1e-14 ? 1.0 / wSum : 0.0;
  }

  return volumes;
}

/**
 * Convenience wrapper that:
 *  1. Resamples the boundary particles for every rigid body (so world-space
 *     positions reflect the current body pose), then
 *  2. Recomputes the Ψ_b volume table for each body.
 *
 * Call once per timestep, before `addBoundaryDensity` and
 * `computeCouplingForces`.
 *
 * @param rigidBodies  All rigid bodies participating in the coupling.
 * @param h            SPH smoothing radius.
 * @param spacing      Boundary particle spacing (typically 0.8 * h).
 * @returns            Updated volume table keyed by `rb.id`.
 */
export function refreshBoundaryState(
  rigidBodies: RigidBody[],
  h: number,
  spacing: number,
): BoundaryVolumeTable {
  const table: BoundaryVolumeTable = new Map();

  for (const rb of rigidBodies) {
    // Re-place boundary particles in world space for current pose
    sampleBoundaryParticles(rb, spacing);
    // Recompute adaptive volumes from updated positions
    table.set(rb.id, computeBoundaryVolumes(rb, h));
  }

  return table;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 2 – Boundary density contribution  (Akinci 2012, Eq. 6)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Augment each fluid particle's density estimate with contributions from nearby
 * boundary particles:
 *
 *   ρ_i  +=  mass · Σ_b  Ψ_b · W(|x_i − x_b|, h)
 *
 * This prevents fluid particles from penetrating rigid boundaries because the
 * SPH pressure solver sees a higher effective density near boundary surfaces and
 * generates the necessary repulsive pressure.
 *
 * Must be called **after** the fluid-fluid density summation loop and **before**
 * the pressure solve.
 *
 * @param fluidParticles  Fluid particle array.
 * @param rigidBodies     Rigid bodies in the scene.
 * @param volumes         Pre-computed Ψ_b table (from `refreshBoundaryState`).
 * @param h               SPH smoothing radius.
 * @param mass            Uniform fluid particle mass.
 */
export function addBoundaryDensity(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  volumes: BoundaryVolumeTable,
  h: number,
  mass: number,
): void {
  const nf = fluidParticles.length;

  for (const rb of rigidBodies) {
    const vols = volumes.get(rb.id);
    if (!vols) continue;
    const nb = rb.boundaryCount;

    for (let i = 0; i < nf; i++) {
      const fi = fluidParticles[i];
      const fx = fi.x;
      const fy = fi.y;

      for (let b = 0; b < nb; b++) {
        const dx = fx - rb.boundaryX[b];
        const dy = fy - rb.boundaryY[b];
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r < h) {
          // ρ_i += mass · Ψ_b · W(r, h)
          fi.density += mass * vols[b] * cubicW(r, h);
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 3 – Boundary velocity at sample point  (Akinci 2012, Section 3.1)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Compute the velocity of a boundary particle at world-space position (bx, by)
 * belonging to rigid body `rb`, accounting for both linear and angular motion:
 *
 *   v_b = v_cm  +  ω × r_b
 *
 * In 2-D, ω is a scalar and the cross product gives:
 *
 *   v_b = (vx − ω·ry,  vy + ω·rx)
 *
 * where (rx, ry) = (bx − rb.x, by − rb.y).
 */
function boundaryVelocity(rb: RigidBody, bx: number, by: number): [number, number] {
  const rx = bx - rb.x;
  const ry = by - rb.y;
  const vbx = rb.vx - rb.angVel * ry;
  const vby = rb.vy + rb.angVel * rx;
  return [vbx, vby];
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 4 – Bidirectional coupling forces  (Akinci 2012, Eq. 10–13)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Compute and accumulate bidirectional fluid–rigid coupling forces.
 *
 * ── Fluid ← Rigid (pressure + viscosity) ──────────────────────────────────
 *
 * Pressure force on fluid particle i from boundary particle b:
 *
 *   f^{press}_{i←b}  =  −mass² · (p_i/ρ_i²  +  p_b/ρ_b²) · Ψ_b · ∇W_{ib}
 *
 * where:
 *   • The "mirrored" boundary pressure p_b = p_i  (no-penetration condition,
 *     common simplification in Akinci 2012 and SPlisHSPlasH).
 *   • ρ_b ≈ restDensity  (boundary particles carry rest density).
 *   • ∇W_{ib} = ∇W(x_i − x_b, h)  points from b toward i.
 *
 * Viscosity force on fluid particle i from boundary particle b (Akinci 2012,
 * Eq. 13 — using the "boundary velocity" formulation):
 *
 *   f^{visc}_{i←b}  =  mass · Ψ_b · ν · (v_b − v_i) · ∇²W(r, h)
 *
 * ── Rigid ← Fluid (Newton's 3rd law) ─────────────────────────────────────
 *
 * The rigid body receives the reaction force at each boundary particle:
 *
 *   f_{b←i}  =  −f_{i←b}
 *
 * Linear force accumulation:
 *
 *   F_rigid  +=  Σ_{i,b}  f_{b←i}
 *
 * Torque accumulation (2-D cross product):
 *
 *   τ_rigid  +=  Σ_{i,b}  (x_b − x_cm) × f_{b←i}
 *            =  Σ_{i,b}  rx · f_{b←i,y} − ry · f_{b←i,x}
 *
 * Forces are **added** to `rb.fx / rb.fy / rb.torque`; callers must
 * `resetForces(rb)` before the first coupling call each step.
 *
 * @param fluidParticles  Fluid particle array.
 * @param rigidBodies     Rigid body array.
 * @param volumes         Pre-computed Ψ_b volume table.
 * @param h               SPH smoothing radius.
 * @param mass            Uniform fluid particle mass.
 * @param restDensity     Fluid rest density ρ₀.
 * @param viscosity       Kinematic viscosity ν (set to 0 to skip viscous term).
 * @param dt              Timestep — accepted for API completeness (impulse-based
 *                        callers may scale forces to impulses externally).
 */
export function computeCouplingForces(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  volumes: BoundaryVolumeTable,
  h: number,
  mass: number,
  restDensity: number,
  viscosity: number,
  dt: number,
): void {
  const nf = fluidParticles.length;

  for (const rb of rigidBodies) {
    const vols = volumes.get(rb.id);
    if (!vols) continue;
    const nb = rb.boundaryCount;

    // Per-body coupling accumulators (avoid repeated property writes)
    let rbFx = 0.0;
    let rbFy = 0.0;
    let rbTorque = 0.0;

    for (let b = 0; b < nb; b++) {
      const bx = rb.boundaryX[b];
      const by = rb.boundaryY[b];
      const Vb = vols[b];

      // Boundary particle velocity (rigid body kinematics)
      const [vbx, vby] = boundaryVelocity(rb, bx, by);

      // Arm vector from rigid body CoM to this boundary particle
      const armX = bx - rb.x;
      const armY = by - rb.y;

      for (let i = 0; i < nf; i++) {
        const fi = fluidParticles[i];
        const dx = fi.x - bx;
        const dy = fi.y - by;
        const r = Math.sqrt(dx * dx + dy * dy);

        if (r >= h || r < 1e-14) continue;

        const rhoi = fi.density;
        if (rhoi < 1e-14) continue;

        // ── Pressure force ──────────────────────────────────────────────────
        //  Symmetric pressure formulation:
        //   term = p_i/ρ_i²  +  p_b/ρ_b²
        //  Mirror: p_b = p_i,  ρ_b = restDensity
        const pi    = fi.pressure;
        const rho0  = restDensity > 0 ? restDensity : rhoi;
        const term  = (pi / (rhoi * rhoi)) + (pi / (rho0 * rho0));

        // ∇W(x_i − x_b, h)  [2-D]
        const [gwx, gwy] = cubicGradW(dx, dy, h);

        // f^{press}_{i←b} = −mass² · term · Ψ_b · ∇W
        const pScale  = -mass * mass * term * Vb;
        const pfx = pScale * gwx;
        const pfy = pScale * gwy;

        // ── Viscosity force ──────────────────────────────────────────────────
        //  Viscosity kernel Laplacian  ∇²W(r, h):
        //  Müller 2003:  ∇²W = (40 / π h⁵) · 6(h − r) / h³
        //  We inline a compact 2-D form here to avoid an extra import.
        //  f^{visc}_{i←b} = mass · Ψ_b · ν · (v_b − v_i) · ∇²W
        let vfx = 0.0;
        let vfy = 0.0;
        if (viscosity > 0.0) {
          const lap2W = (40.0 / (Math.PI * Math.pow(h, 5))) * 6.0 * (h - r) / (h * h * h);
          const vScale = mass * Vb * viscosity * lap2W;
          vfx = vScale * (vbx - fi.vx);
          vfy = vScale * (vby - fi.vy);
        }

        // Total force on fluid particle i from boundary particle b
        const fTotX = pfx + vfx;
        const fTotY = pfy + vfy;

        // Accumulate onto fluid particle
        fi.ax += fTotX / mass; // convert force → acceleration
        fi.ay += fTotY / mass;

        // Reaction on rigid body (Newton's 3rd law)
        const rfx = -fTotX;
        const rfy = -fTotY;

        rbFx     += rfx;
        rbFy     += rfy;
        // 2-D torque: τ += rx * fy − ry * fx
        rbTorque += armX * rfy - armY * rfx;
      }
    }

    // Flush accumulated coupling forces to rigid body
    rb.fx     += rbFx;
    rb.fy     += rbFy;
    rb.torque += rbTorque;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 5 – High-level integration helper
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * All-in-one per-timestep coupling pipeline:
 *
 *  1. Refresh boundary particle world-space positions + recompute volumes.
 *  2. Add boundary density contributions to fluid particles.
 *  3. Compute bidirectional forces (fluid → rigid + rigid → fluid).
 *
 * Returns the updated `BoundaryVolumeTable` so callers can cache it if desired.
 *
 * Typical call site (inside the main SPH step loop):
 *
 * ```ts
 * resetForces(rb);                      // clear force accumulators
 * const vols = stepFluidRigidCoupling(  // coupling pipeline
 *   particles, bodies,
 *   h, spacing, mass, restDensity, viscosity, dt,
 * );
 * integrateRigidBodies(bodies, dt, gravity, domainW, domainH);
 * // … rest of SPH pressure solve …
 * ```
 *
 * @param fluidParticles  Fluid particle array.
 * @param rigidBodies     Rigid body array.
 * @param h               SPH smoothing radius.
 * @param spacing         Boundary particle spacing (typically 0.8 × h).
 * @param mass            Uniform fluid particle mass.
 * @param restDensity     Fluid rest density ρ₀.
 * @param viscosity       Kinematic viscosity ν.
 * @param dt              Simulation timestep.
 * @returns               Updated boundary volume table.
 */
export function stepFluidRigidCoupling(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  h: number,
  spacing: number,
  mass: number,
  restDensity: number,
  viscosity: number,
  dt: number,
): BoundaryVolumeTable {
  // Step 1: adaptive boundary state update (positions + volumes)
  const volumes = refreshBoundaryState(rigidBodies, h, spacing);

  // Step 2: boundary → fluid density (prevents penetration)
  addBoundaryDensity(fluidParticles, rigidBodies, volumes, h, mass);

  // Step 3: bidirectional pressure + viscosity forces
  computeCouplingForces(
    fluidParticles, rigidBodies, volumes,
    h, mass, restDensity, viscosity, dt,
  );

  return volumes;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Section 6 – Neighbor-list helpers  (kept for API symmetry / external tools)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build per-boundary-particle neighbor lists for the given rigid bodies:
 *
 *   result[rb.id][b] = [fluidIdx, ...]   all fluid indices within radius h.
 *
 * O(nb × nf) — suitable for scene sizes where nb and nf are both < 10 000.
 * For larger scenes, replace with a spatial hash pass.
 */
export function buildBoundaryNeighbors(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  h: number,
): Map<number, number[][]> {
  const result = new Map<number, number[][]>();

  for (const rb of rigidBodies) {
    const nb = rb.boundaryCount;
    const lists: number[][] = Array.from({ length: nb }, () => []);

    for (let b = 0; b < nb; b++) {
      const bx = rb.boundaryX[b];
      const by = rb.boundaryY[b];

      for (let i = 0; i < fluidParticles.length; i++) {
        const dx = fluidParticles[i].x - bx;
        const dy = fluidParticles[i].y - by;
        if (dx * dx + dy * dy < h * h) {
          lists[b].push(i);
        }
      }
    }

    result.set(rb.id, lists);
  }

  return result;
}
