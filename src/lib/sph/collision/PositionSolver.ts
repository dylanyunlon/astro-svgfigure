// PositionSolver.ts — Baumgarte position correction
// Resolves residual penetration after velocity-level impulse solving

import { RigidBody, ContactConstraint } from '../types';

export class PositionSolver {
  private beta: number;
  private slop: number;
  private maxCorrection: number;

  constructor(beta = 0.3, slop = 0.005, maxCorrection = 0.2) {
    this.beta = beta;
    this.slop = slop;
    this.maxCorrection = maxCorrection;
  }

  solve(bodies: RigidBody[], contacts: ContactConstraint[], iterations: number = 3): boolean {
    let maxPenetration = 0;

    for (let iter = 0; iter < iterations; iter++) {
      maxPenetration = 0;

      for (const c of contacts) {
        const bodyA = bodies[c.bodyA];
        const bodyB = bodies[c.bodyB];

        // Recompute separation at current positions
        // contact point relative to bodies
        const rAx = c.point.x - bodyA.x;
        const rAy = c.point.y - bodyA.y;
        const rBx = c.point.x - bodyB.x;
        const rBy = c.point.y - bodyB.y;

        // Current separation (negative = penetrating)
        const separation = (
          (bodyB.x + rBx - bodyA.x - rAx) * c.normal.x +
          (bodyB.y + rBy - bodyA.y - rAy) * c.normal.y
        ) - c.depth;

        const penetration = -separation;
        if (penetration > maxPenetration) maxPenetration = penetration;

        // Baumgarte correction
        const correction = Math.min(
          Math.max(this.beta * (penetration - this.slop), 0),
          this.maxCorrection
        );

        if (correction <= 0) continue;

        const totalInvMass = bodyA.invMass + bodyB.invMass;
        if (totalInvMass <= 0) continue;

        const impulse = correction / totalInvMass;

        // Apply position correction along normal
        bodyA.x -= bodyA.invMass * impulse * c.normal.x;
        bodyA.y -= bodyA.invMass * impulse * c.normal.y;
        bodyB.x += bodyB.invMass * impulse * c.normal.x;
        bodyB.y += bodyB.invMass * impulse * c.normal.y;

        // Angular correction
        const rACrossN = rAx * c.normal.y - rAy * c.normal.x;
        const rBCrossN = rBx * c.normal.y - rBy * c.normal.x;
        bodyA.angle -= bodyA.invInertia * rACrossN * impulse;
        bodyB.angle += bodyB.invInertia * rBCrossN * impulse;
      }
    }

    // Converged if max penetration is within tolerance
    return maxPenetration < this.slop * 2;
  }
}
