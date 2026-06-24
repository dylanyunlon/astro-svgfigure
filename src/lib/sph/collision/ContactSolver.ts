// === src/lib/sph/collision/ContactSolver.ts ===



// ─────────────────────────────────────────────────────────────────────────────
// Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

<<<<<<< HEAD
// [orphan-precise] /** Baumgarte positional-correction strength (fraction of penetration per step). */
=======
/** Baumgarte positional-correction strength (fraction of penetration per step). */

import type { RigidBody, ContactConstraint } from '../types';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const BAUMGARTE_BETA = 0.2;

/**
 * Penetration slop (metres).  Depths smaller than this are not Baumgarte-corrected,
 * which avoids jitter from floating-point noise at rest.
 */
const PENETRATION_SLOP = 0.005;

/**
 * Restitution threshold (m/s).
 * When the impact speed along the normal is below this value the contact is
 * treated as perfectly inelastic (e = 0).  This prevents tiny "micro-bounces"
 * that would keep the body jittering rather than coming to rest.
 */
const RESTITUTION_VELOCITY_THRESHOLD = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
// Re-export the public interfaces so callers only need one import path.
// ─────────────────────────────────────────────────────────────────────────────

export type { RigidBody, ContactConstraint };

// ─────────────────────────────────────────────────────────────────────────────
// ContactSolver
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sequential-Impulse (Gauss-Seidel) contact constraint solver.
 *
 * ## Algorithm per call to `solve`
 *
 * 1. **Pre-solve snapshot** – record relative normal velocity before any
 *    impulse is applied (used for the restitution target).
 *
 * 2. **Warm start** – apply the accumulated impulses from the *previous* frame
 *    immediately.  This seeds the Gauss-Seidel iteration close to the solution
 *    and dramatically improves convergence, especially for resting stacks.
 *
 * 3. **Velocity iterations** (×`iterations`) – for each contact:
 *    a. Compute the relative velocity at the contact point (including angular
 *       contributions: v_contact = v_cm ± ω × r).
 *    b. **Normal impulse λn**
 *       - *High-speed impact* (|vRelPre| > threshold): use restitution bias
 *         `e·|vRelPre|` as the target separation velocity; Baumgarte disabled
 *         (the impulse already pushes the bodies apart).
 *       - *Resting / slow contact*: use Baumgarte bias `β·max(depth−slop,0)/dt`
 *         to gradually correct position drift; e = 0.
 *       - Clamp: accumulated λn ≥ 0 (contacts can only push, never pull).
 *    c. **Tangent impulse λt** (Coulomb friction)
 *       - Clamp: |λt| ≤ μ · λn  (cone constraint).
 *    d. Apply both impulses to linear and angular velocities of both bodies.
 *
 * ## Notes
 * - Bodies with `invMass = 0` are treated as static (infinite mass).
 * - The solver modifies `body.vx/vy/omega` and `contact.normalImpulse /
 *   tangentImpulse` in-place.
 */
export class ContactSolver {
  /**
   * Resolve all contacts for one simulation step.
   *
   * @param bodies     Array of all rigid bodies in the scene.
   * @param contacts   Active contact constraints referencing indices into `bodies`.
   * @param dt         Time-step duration (seconds).
   * @param iterations Number of Gauss-Seidel iterations (4–20 is typical).
   */
  solve(
    bodies: RigidBody[],
    contacts: ContactConstraint[],
    dt: number,
    iterations: number,
  ): void {
    // ── 1. Pre-solve relative normal velocity (before warm start) ──────────
    //
    // We need the velocity at the *start* of this time-step, before any
    // impulses, to compute the restitution target.  Store it on the constraint.
    // TypeScript will carry the extra field; the interface is widened here only.
    type ContactExt = ContactConstraint & { _vRelNPre: number };
    const contactsExt = contacts as ContactExt[];

    for (const c of contactsExt) {
      const a = bodies[c.bodyA];
      const b = bodies[c.bodyB];
      const nx = c.normal.x;
      const ny = c.normal.y;
      // Relative velocity of B minus A, projected onto the contact normal.
      // Positive = separating, negative = approaching.
      c._vRelNPre = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
    }

    // ── 2. Warm start ─────────────────────────────────────────────────────
    //
    // Apply the accumulated impulses saved from the previous frame as an
    // initial guess, seeding the solver near the solution.
    for (const c of contactsExt) {
      const a = bodies[c.bodyA];
      const b = bodies[c.bodyB];

      const nx = c.normal.x;
      const ny = c.normal.y;
      const tx = -ny; // tangent perpendicular to normal (2-D)
      const ty = nx;

      // Lever arms from each body's centre of mass to the contact point.
      const rAx = c.point.x - a.x;
      const rAy = c.point.y - a.y;
      const rBx = c.point.x - b.x;
      const rBy = c.point.y - b.y;

      // Total warm-start impulse vector.
      const Px = nx * c.normalImpulse + tx * c.tangentImpulse;
      const Py = ny * c.normalImpulse + ty * c.tangentImpulse;

      // Apply −P to body A, +P to body B.
      a.vx    -= a.invMass    * Px;
      a.vy    -= a.invMass    * Py;
      a.omega -= a.invInertia * (rAx * Py - rAy * Px); // 2-D cross product

      b.vx    += b.invMass    * Px;
      b.vy    += b.invMass    * Py;
      b.omega += b.invInertia * (rBx * Py - rBy * Px);
    }

    // ── 3. Velocity iterations ────────────────────────────────────────────
    for (let iter = 0; iter < iterations; iter++) {
      for (const c of contactsExt) {
        const a = bodies[c.bodyA];
        const b = bodies[c.bodyB];

        const nx = c.normal.x;
        const ny = c.normal.y;
        const tx = -ny;
        const ty = nx;

        const rAx = c.point.x - a.x;
        const rAy = c.point.y - a.y;
        const rBx = c.point.x - b.x;
        const rBy = c.point.y - b.y;

        // Velocity at contact point for each body:
        //   v_contact = v_cm + ω × r   (2-D: ω × r = (−ω·ry, ω·rx))
        const vAx = a.vx - a.omega * rAy;
        const vAy = a.vy + a.omega * rAx;
        const vBx = b.vx - b.omega * rBy;
        const vBy = b.vy + b.omega * rBx;

        // Relative velocity (B − A) at the contact point.
        const dvx = vBx - vAx;
        const dvy = vBy - vAy;

        const vRel_n = dvx * nx + dvy * ny; // along normal
        const vRel_t = dvx * tx + dvy * ty; // along tangent

        // ── Effective mass along normal ──────────────────────────────────
        //
        //   1/m* = 1/mA + 1/mB + (rA×n)²/IA + (rB×n)²/IB
        //
        const rAcrossN = rAx * ny - rAy * nx; // scalar 2-D cross
        const rBcrossN = rBx * ny - rBy * nx;
        const effMassN =
          a.invMass + b.invMass +
          rAcrossN * rAcrossN * a.invInertia +
          rBcrossN * rBcrossN * b.invInertia;

        if (effMassN < 1e-12) continue; // both bodies static – skip

        // ── Restitution / Baumgarte target velocity ──────────────────────
        //
        // High-speed impact: apply restitution, suppress Baumgarte
        // (the bounce impulse is strong enough to separate the bodies).
        //
        // Resting / slow contact: no restitution (e = 0), apply Baumgarte
        // positional correction to prevent gradual drift.
        const e_raw    = Math.min(a.restitution, b.restitution);
        const preVel   = c._vRelNPre;
        const isImpact = preVel < -RESTITUTION_VELOCITY_THRESHOLD;

        // Target relative velocity along normal after the impulse.
        const targetVn: number = isImpact
          ? e_raw * (-preVel)   // restitution: reflect at e*|v_impact|
          : (BAUMGARTE_BETA / dt) * Math.max(c.depth - PENETRATION_SLOP, 0);
        //   ^^ Baumgarte bias: push bodies apart at a fraction of depth/dt

        // ── Normal impulse increment ─────────────────────────────────────
        //
        //   Δλn = (targetVn − vRel_n) / effMassN
        //
        const dLambdaN = (targetVn - vRel_n) / effMassN;

        // Clamp accumulated normal impulse to [0, ∞).
        // Contacts can only *push* — they cannot pull bodies together.
        const lambda0N   = c.normalImpulse;
        c.normalImpulse  = Math.max(lambda0N + dLambdaN, 0);
        const clampedDLN = c.normalImpulse - lambda0N;

        // ── Effective mass along tangent ─────────────────────────────────
        const rAcrossT = rAx * ty - rAy * tx;
        const rBcrossT = rBx * ty - rBy * tx;
        const effMassT =
          a.invMass + b.invMass +
          rAcrossT * rAcrossT * a.invInertia +
          rBcrossT * rBcrossT * b.invInertia;

        // ── Friction impulse increment ───────────────────────────────────
        //
        //   Δλt = −vRel_t / effMassT   (drive tangential slip to zero)
        //
        // Coulomb cone: |λt| ≤ μ · λn
        const mu          = (a.friction + b.friction) * 0.5;
        const maxFriction = mu * c.normalImpulse;
        const dLambdaT    = effMassT > 1e-12 ? -vRel_t / effMassT : 0;

        const lambda0T   = c.tangentImpulse;
        c.tangentImpulse = Math.max(
          -maxFriction,
          Math.min(lambda0T + dLambdaT, maxFriction),
        );
        const clampedDLT = c.tangentImpulse - lambda0T;

        // ── Apply impulses to both bodies ────────────────────────────────
        const Px = nx * clampedDLN + tx * clampedDLT;
        const Py = ny * clampedDLN + ty * clampedDLT;

        a.vx    -= a.invMass    * Px;
        a.vy    -= a.invMass    * Py;
        a.omega -= a.invInertia * (rAx * Py - rAy * Px);

        b.vx    += b.invMass    * Px;
        b.vy    += b.invMass    * Py;
        b.omega += b.invInertia * (rBx * Py - rBy * Px);
      }
    }
  }
}

// auto-stub for missing export
export function warmStart(...args: any[]): any { return undefined as any; }
