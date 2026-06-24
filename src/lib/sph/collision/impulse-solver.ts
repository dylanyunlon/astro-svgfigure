


import { Constraint, NonPenetrationConstraint, FrictionConstraint, createNonPenetrationConstraint, createFrictionConstraint } from './constraints';
import { ContactManifold, warmStartManifold } from './contact-manifold';

export interface SolverBody {
  // Position
  x: number; y: number; angle: number;
  // Velocity
  vx: number; vy: number; angVel: number;
  // Inverse mass properties (0 = infinite mass = static)
  invMass: number;
  invInertia: number;
}

export interface SolverConfig {
  iterations: number;           // default 10
  warmStart: boolean;           // default true
  baumgarte: number;            // position correction factor, default 0.2
  slop: number;                 // allowed penetration before correction, default 0.5
  restitutionThreshold: number; // min approach speed for bounce, default 1.0
}

export function defaultSolverConfig(): SolverConfig {
  return {
    iterations: 10,
    warmStart: true,
    baumgarte: 0.2,
    slop: 0.5,
    restitutionThreshold: 1.0,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Compute the relative velocity at a contact point along a given direction.
 *
 *   v_rel = dot(n, vA + wA x rA - vB - wB x rB)
 *
 * In 2D the cross product of a scalar angular velocity w and a 2D vector r is:
 *   w x r = (-w*ry, w*rx)  — we only need the dot product with n:
 *   dot(n, w x r) = w * (nx * ry - ny * rx)  ... wait, let me be precise:
 *   (w x r) in 2D = w * perp(r) where perp(rx, ry) = (-ry, rx)
 *   dot(n, perp(r)) = nx*(-ry) + ny*(rx) = nx*ry*(-1) + ny*rx  Hmm, let's keep it explicit.
 *
 * Actually in 2D:  w × r  gives the velocity contribution:
 *   v_from_rot = { -w * ry,  w * rx }
 * So: dot(n, v_from_rot) = n.x * (-w * ry) + n.y * (w * rx)
 *                        = w * (n.y * rx - n.x * ry)
 */
function relativeVelocityAlongNormal(
  bodyA: SolverBody, bodyB: SolverBody,
  rAx: number, rAy: number,
  rBx: number, rBy: number,
  nx: number, ny: number
): number {
  const vAx = bodyA.vx + (-bodyA.angVel * rAy);
  const vAy = bodyA.vy + ( bodyA.angVel * rAx);
  const vBx = bodyB.vx + (-bodyB.angVel * rBy);
  const vBy = bodyB.vy + ( bodyB.angVel * rBx);
  return nx * (vAx - vBx) + ny * (vAy - vBy);
}

/**
 * Compute the effective mass (inverse) for a 1-DOF constraint.
 *
 *   K = invMassA + invMassB
 *     + invInertiaA * (rA x n)^2
 *     + invInertiaB * (rB x n)^2
 *
 * In 2D: (r x n) scalar = rx * ny - ry * nx
 */
function effectiveMassInv(
  bodyA: SolverBody, bodyB: SolverBody,
  rAx: number, rAy: number,
  rBx: number, rBy: number,
  nx: number, ny: number
): number {
  const rAxN = rAx * ny - rAy * nx; // scalar cross r_A × n
  const rBxN = rBx * ny - rBy * nx; // scalar cross r_B × n
  return (
    bodyA.invMass + bodyB.invMass +
    bodyA.invInertia * rAxN * rAxN +
    bodyB.invInertia * rBxN * rBxN
  );
}

/**
 * Apply impulse P along direction (nx, ny) at offset (rAx, rAy) on bodyA
 * and equal-and-opposite impulse on bodyB at (rBx, rBy).
 */
function applyImpulse(
  bodyA: SolverBody, bodyB: SolverBody,
  rAx: number, rAy: number,
  rBx: number, rBy: number,
  nx: number, ny: number,
  lambda: number
): void {
  // bodyA gains impulse +lambda * n
  bodyA.vx  +=  lambda * nx * bodyA.invMass;
  bodyA.vy  +=  lambda * ny * bodyA.invMass;
  bodyA.angVel += lambda * (rAx * ny - rAy * nx) * bodyA.invInertia;

  // bodyB gains impulse -lambda * n
  bodyB.vx  -=  lambda * nx * bodyB.invMass;
  bodyB.vy  -=  lambda * ny * bodyB.invMass;
  bodyB.angVel -= lambda * (rBx * ny - rBy * nx) * bodyB.invInertia;
}

// ---------------------------------------------------------------------------
// Per-contact point solver state (built once, iterated N times)
// ---------------------------------------------------------------------------

interface ContactSolverPoint {
  // Contact point offsets from body centres
  rAx: number; rAy: number;
  rBx: number; rBy: number;

  // Normal constraint (non-penetration)
  normalMassInv: number;    // 1 / K_n
  bias: number;             // Baumgarte + restitution bias
  lambdaN: number;          // accumulated normal impulse (clamped >= 0)

  // Friction constraint
  tangentMassInv: number;   // 1 / K_t
  lambdaT: number;          // accumulated tangential impulse (clamped to friction cone)
}

interface ManifoldSolver {
  bodyAIdx: number;
  bodyBIdx: number;
  nx: number; ny: number;    // contact normal (from B toward A)
  tx: number; ty: number;    // contact tangent (perpendicular to normal)
  friction: number;          // combined friction coefficient
  points: ContactSolverPoint[];
  manifold: ContactManifold; // reference kept for warm-start write-back
}

// ---------------------------------------------------------------------------
// Build solver islands from manifolds
// ---------------------------------------------------------------------------

function buildManifoldSolver(
  manifold: ContactManifold,
  bodies: SolverBody[],
  config: SolverConfig,
  dt: number
): ManifoldSolver {
  const bodyA = bodies[manifold.bodyAIndex];
  const bodyB = bodies[manifold.bodyBIndex];

  const nx = manifold.normalX;
  const ny = manifold.normalY;
  // Tangent = 90-degree CCW rotation of normal
  const tx = -ny;
  const ty =  nx;

  const points: ContactSolverPoint[] = [];

  for (let i = 0; i < manifold.contacts.length; i++) {
    const contact = manifold.contacts[i];

    // Offsets from body centres to contact point
    const rAx = contact.x - bodyA.x;
    const rAy = contact.y - bodyA.y;
    const rBx = contact.x - bodyB.x;
    const rBy = contact.y - bodyB.y;

    // Effective masses
    const normalMassInv  = effectiveMassInv(bodyA, bodyB, rAx, rAy, rBx, rBy, nx, ny);
    const tangentMassInv = effectiveMassInv(bodyA, bodyB, rAx, rAy, rBx, rBy, tx, ty);

    // Baumgarte position-error bias — only correct penetration beyond slop
    const penetration = contact.depth; // positive = overlapping
    const positionError = Math.min(0, -(penetration - config.slop));
    // bias drives separation: divide by dt to get a velocity target
    const baumgarteBias = (config.baumgarte / dt) * positionError;

    // Restitution bias (bounce)
    let restitutionBias = 0;
    const vRel = relativeVelocityAlongNormal(bodyA, bodyB, rAx, rAy, rBx, rBy, nx, ny);
    const restitution = manifold.restitution ?? 0;
    if (-vRel > config.restitutionThreshold) {
      // Closing velocity exceeds threshold — add restitution
      restitutionBias = restitution * vRel; // vRel is negative when closing, so this > 0
    }

    const bias = baumgarteBias + restitutionBias;

    points.push({
      rAx, rAy, rBx, rBy,
      normalMassInv:  normalMassInv  > 0 ? 1 / normalMassInv  : 0,
      tangentMassInv: tangentMassInv > 0 ? 1 / tangentMassInv : 0,
      bias,
      lambdaN: manifold.contacts[i].accumulatedNormalImpulse  ?? 0,
      lambdaT: manifold.contacts[i].accumulatedTangentImpulse ?? 0,
    });
  }

  return {
    bodyAIdx: manifold.bodyAIndex,
    bodyBIdx: manifold.bodyBIndex,
    nx, ny, tx, ty,
    friction: manifold.friction ?? 0.3,
    points,
    manifold,
  };
}

// ---------------------------------------------------------------------------
// Warm-start: apply cached impulses to seed velocities
// ---------------------------------------------------------------------------

function warmStart(solver: ManifoldSolver, bodies: SolverBody[]): void {
  const bodyA = bodies[solver.bodyAIdx];
  const bodyB = bodies[solver.bodyBIdx];

  for (const pt of solver.points) {
    // Re-apply stored normal impulse
    applyImpulse(bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy,
      solver.nx, solver.ny, pt.lambdaN);
    // Re-apply stored tangent impulse
    applyImpulse(bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy,
      solver.tx, solver.ty, pt.lambdaT);
  }
}

// ---------------------------------------------------------------------------
// Single Gauss-Seidel iteration over one manifold solver
// (Sequential: each impulse immediately updates body velocities so subsequent
//  constraints in this same iteration already see the corrected state.)
// ---------------------------------------------------------------------------

function iterateManifoldSolver(solver: ManifoldSolver, bodies: SolverBody[]): void {
  const bodyA = bodies[solver.bodyAIdx];
  const bodyB = bodies[solver.bodyBIdx];

  for (const pt of solver.points) {
    // ---- Normal (non-penetration) constraint --------------------------------
    {
      const vRel = relativeVelocityAlongNormal(
        bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy, solver.nx, solver.ny
      );

      // Impulse magnitude needed this sub-step
      // We want:  lambda = -( J*v + bias ) * effectiveMass
      // Since vRel = J * v for this constraint:
      const dLambda = -(vRel + pt.bias) * pt.normalMassInv;

      // Clamp: accumulated normal impulse must be >= 0 (can't pull bodies together)
      const lambdaOld = pt.lambdaN;
      pt.lambdaN = Math.max(0, lambdaOld + dLambda);
      const clampedDelta = pt.lambdaN - lambdaOld;

      applyImpulse(bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy,
        solver.nx, solver.ny, clampedDelta);
    }

    // ---- Tangential (friction) constraint -----------------------------------
    {
      const vRelT = relativeVelocityAlongNormal(
        bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy, solver.tx, solver.ty
      );

      const dLambdaT = -vRelT * pt.tangentMassInv;

      // Coulomb friction cone: |lambdaT| <= mu * lambdaN
      const maxFriction = solver.friction * pt.lambdaN;
      const lambdaTOld = pt.lambdaT;
      pt.lambdaT = Math.max(-maxFriction, Math.min(maxFriction, lambdaTOld + dLambdaT));
      const clampedDeltaT = pt.lambdaT - lambdaTOld;

      applyImpulse(bodyA, bodyB, pt.rAx, pt.rAy, pt.rBx, pt.rBy,
        solver.tx, solver.ty, clampedDeltaT);
    }
  }
}

// ---------------------------------------------------------------------------
// Write accumulated impulses back to manifold for next-frame warm-start
// ---------------------------------------------------------------------------

function writeBackImpulses(solver: ManifoldSolver): void {
  for (let i = 0; i < solver.points.length; i++) {
    const pt = solver.points[i];
    solver.manifold.contacts[i].accumulatedNormalImpulse  = pt.lambdaN;
    solver.manifold.contacts[i].accumulatedTangentImpulse = pt.lambdaT;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Full sequential-impulse solve step.
 *
 * Algorithm (Catto / Box2D style):
 *  1. For each manifold, build a ManifoldSolver (pre-compute effective masses,
 *     bias terms, load cached impulses).
 *  2. Warm-start: apply cached impulses to body velocities.
 *  3. Iterate `config.iterations` times over all manifold solvers.
 *     — SEQUENTIAL: each constraint update immediately modifies body velocities,
 *       so the next constraint in the same iteration sees the corrected state.
 *       This Gauss-Seidel ordering is what makes the solver converge.
 *  4. Write accumulated lambdas back into manifolds for next-frame warm-start.
 *
 * @param bodies   Flat array of solver bodies (mutated in-place).
 * @param manifolds Active contact manifolds for this frame.
 * @param config   Solver tuning parameters.
 * @param dt       Physics time-step (seconds). Defaults to 1/60 if not provided.
 */
export function solveConstraints(
  bodies: SolverBody[],
  manifolds: ContactManifold[],
  config: SolverConfig,
  dt = 1 / 60
): void {
  if (manifolds.length === 0) return;

  // Step 1: Build per-manifold solver state
  const solvers: ManifoldSolver[] = manifolds.map(m =>
    buildManifoldSolver(m, bodies, config, dt)
  );

  // Step 2: Warm-start
  if (config.warmStart) {
    for (const solver of solvers) {
      warmStart(solver, bodies);
    }
  }

  // Step 3: Gauss-Seidel iterations
  // Sequential = each applyImpulse call mutates body velocities immediately.
  // Subsequent constraints in the SAME iteration see the already-corrected
  // velocities — this is the key convergence property vs. Jacobi (parallel).
  for (let iter = 0; iter < config.iterations; iter++) {
    for (const solver of solvers) {
      iterateManifoldSolver(solver, bodies);
    }
  }

  // Step 4: Write accumulated impulses back for warm-starting next frame
  for (const solver of solvers) {
    writeBackImpulses(solver);
  }
}
