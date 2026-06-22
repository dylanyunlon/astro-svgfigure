import { Particle } from './dfsph-solver';
import { RigidBody } from './rigid-body';
import { cubicW, cubicGradW } from './sph-kernels';

// ---------------------------------------------------------------------------
// Akinci 2012 – Versatile Rigid-Fluid Coupling for Incompressible SPH
// Reference implementation following SPlisHSPlasH/BoundaryModel_Akinci2012.cpp
// ---------------------------------------------------------------------------

// Helpers -------------------------------------------------------------------

function dist(a: Float64Array, b: Float64Array): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function sub3(a: Float64Array, b: Float64Array): Float64Array {
  return new Float64Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
}

function add3InPlace(target: Float64Array, src: Float64Array): void {
  target[0] += src[0];
  target[1] += src[1];
  target[2] += src[2];
}

function scale3(v: Float64Array, s: number): Float64Array {
  return new Float64Array([v[0] * s, v[1] * s, v[2] * s]);
}

function cross3(a: Float64Array, b: Float64Array): Float64Array {
  return new Float64Array([
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]);
}

// ---------------------------------------------------------------------------
// computeBoundaryVolumes
// ---------------------------------------------------------------------------
// Akinci 2012, Eq. (4):
//   V_b = 1 / Σ_k W(x_b − x_k, h)
// where the sum runs over all *other* boundary particles of the same rigid body.
// This "psi" value encodes the local surface density so that boundary particles
// can contribute a physically meaningful mass to the SPH sum.
// ---------------------------------------------------------------------------
export function computeBoundaryVolumes(
  rb: RigidBody,
  h: number,
): Float64Array {
  const n = rb.boundaryParticles.length;
  const volumes = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const xi = rb.boundaryParticles[i].position;
    let wSum = 0.0;

    // Self-contribution W(0)
    wSum += cubicW(0.0, h);

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const xk = rb.boundaryParticles[k].position;
      const r = dist(xi, xk);
      if (r < h) {
        wSum += cubicW(r, h);
      }
    }

    volumes[i] = wSum > 1e-12 ? 1.0 / wSum : 0.0;
  }

  return volumes;
}

// ---------------------------------------------------------------------------
// addBoundaryDensity
// ---------------------------------------------------------------------------
// Augments each fluid particle's density with contributions from nearby
// boundary particles (Akinci 2012, Eq. 6):
//
//   ρ_i += ρ₀ · Σ_b  V_b · W(x_i − x_b, h)
//
// where ρ₀ is the rest density represented as  mass / (particle volume).
// We use `mass` as the fluid particle mass and approximate rest volume from
// the typical SPH discretisation so the formula becomes:
//
//   ρ_i += mass · Σ_b  V_b · W(x_i − x_b, h)
//
// This prevents fluid particles from penetrating the boundary.
// ---------------------------------------------------------------------------
export function addBoundaryDensity(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  boundaryVolumes: Map<string, Float64Array>,
  _neighbors: number[][], // fluid-fluid neighbor lists (unused here; kept for API symmetry)
  h: number,
  mass: number,
): void {
  for (let i = 0; i < fluidParticles.length; i++) {
    const fi = fluidParticles[i];
    const xi = fi.position;

    for (const rb of rigidBodies) {
      const vols = boundaryVolumes.get(rb.id);
      if (!vols) continue;

      for (let b = 0; b < rb.boundaryParticles.length; b++) {
        const xb = rb.boundaryParticles[b].position;
        const r = dist(xi, xb);
        if (r < h) {
          // ρ_i += mass · V_b · W(|x_i − x_b|, h)
          fi.density += mass * vols[b] * cubicW(r, h);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// buildBoundaryNeighbors
// ---------------------------------------------------------------------------
// For each rigid body, for each of its boundary particles, collect the indices
// of fluid particles that lie within the smoothing radius h.
// Returns:  Map<rbId, boundaryParticleIdx → fluidIdx[]>
// ---------------------------------------------------------------------------
export function buildBoundaryNeighbors(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  h: number,
): Map<string, number[][]> {
  const result = new Map<string, number[][]>();

  for (const rb of rigidBodies) {
    const nb = rb.boundaryParticles.length;
    const bNeighbors: number[][] = Array.from({ length: nb }, () => []);

    for (let b = 0; b < nb; b++) {
      const xb = rb.boundaryParticles[b].position;

      for (let i = 0; i < fluidParticles.length; i++) {
        const xi = fluidParticles[i].position;
        if (dist(xi, xb) < h) {
          bNeighbors[b].push(i);
        }
      }
    }

    result.set(rb.id, bNeighbors);
  }

  return result;
}

// ---------------------------------------------------------------------------
// computeCouplingForces
// ---------------------------------------------------------------------------
// Two-way coupling via SPH pressure gradient (Akinci 2012, Eq. 10-11).
//
// Force on fluid particle i from boundary particle b:
//
//   f_{i←b} = −mass · m_i · ( p_i / ρ_i² ) · V_b · ∇W(x_i − x_b, h)
//
// By Newton's 3rd law the rigid body receives an equal-and-opposite force:
//
//   f_{b←i} = −f_{i←b}
//
// Torque on rigid body:
//
//   τ += (x_b − x_cm) × f_{b←i}
//
// where x_cm is the rigid body centre of mass.
//
// Notes:
//   • We use the symmetric pressure formulation (p_i/ρ_i² + p_b/ρ_b²) where
//     the boundary 'pressure' p_b is mirrored from the fluid particle.
//   • dt is accepted for interface symmetry (used by callers for impulse-based
//     rigid body integrators); direct force accumulation is performed here.
// ---------------------------------------------------------------------------
export function computeCouplingForces(
  fluidParticles: Particle[],
  rigidBodies: RigidBody[],
  boundaryVolumes: Map<string, Float64Array>,
  h: number,
  mass: number,
  _dt: number,
): void {
  for (const rb of rigidBodies) {
    const vols = boundaryVolumes.get(rb.id);
    if (!vols) continue;

    // Reset rigid body force/torque accumulators for coupling contribution
    const rbForce = new Float64Array(3);
    const rbTorque = new Float64Array(3);

    for (let b = 0; b < rb.boundaryParticles.length; b++) {
      const xb = rb.boundaryParticles[b].position;
      const Vb = vols[b];

      // Arm from rigid body centre of mass to this boundary particle (world space)
      const arm = sub3(xb, rb.centerOfMass);

      for (let i = 0; i < fluidParticles.length; i++) {
        const fi = fluidParticles[i];
        const xi = fi.position;
        const r = dist(xi, xb);
        if (r >= h || r < 1e-12) continue;

        // Pressure of fluid particle
        const pi = fi.pressure;
        const rhoi = fi.density;
        if (rhoi < 1e-12) continue;

        // Mirror boundary pressure = fluid pressure (no-penetration condition)
        // Symmetric SPH pressure gradient term: p_i/ρ_i² + p_b/ρ_b²
        // We approximate ρ_b ≈ ρ_i (common simplification in Akinci 2012 impl.)
        const pressureTerm = (pi / (rhoi * rhoi)) + (pi / (rhoi * rhoi));

        // ∇W(x_i − x_b, h)  — points from b toward i
        const xib = sub3(xi, xb);
        const gradW = cubicGradW(xib, r, h);

        // f_{i←b} = −mass² · pressureTerm · V_b · ∇W
        // (mass² because one mass comes from the SPH sum approximation of the
        //  integral and one from the particle's own mass)
        const scale = -mass * mass * pressureTerm * Vb;
        const fOnFluid = scale3(gradW, scale);

        // Accumulate force on fluid particle
        add3InPlace(fi.force, fOnFluid);

        // Reaction force on rigid body (Newton's 3rd law)
        const fOnRigid = scale3(fOnFluid, -1.0);
        add3InPlace(rbForce, fOnRigid);

        // Torque contribution: τ += r × F
        const torqueContrib = cross3(arm, fOnRigid);
        add3InPlace(rbTorque, torqueContrib);
      }
    }

    // Transfer accumulated force and torque to rigid body
    add3InPlace(rb.force, rbForce);
    add3InPlace(rb.torque, rbTorque);
  }
}
