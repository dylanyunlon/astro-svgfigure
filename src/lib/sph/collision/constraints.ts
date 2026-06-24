

// ─── Types ────────────────────────────────────────────────────────────────────


import { ContactPoint } from "./contact-manifold";

export interface RigidBody {
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  angularVelocity: { x: number; y: number; z: number };
  inverseMass: number;
  inverseInertia: { xx: number; yy: number; zz: number; xy: number; xz: number; yz: number };
  force: { x: number; y: number; z: number };
  torque: { x: number; y: number; z: number };
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

// ─── Math Utilities ───────────────────────────────────────────────────────────

function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function scale(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

function applyInverseInertia(
  I: RigidBody["inverseInertia"],
  v: Vec3
): Vec3 {
  return {
    x: I.xx * v.x + I.xy * v.y + I.xz * v.z,
    y: I.xy * v.x + I.yy * v.y + I.yz * v.z,
    z: I.xz * v.x + I.yz * v.y + I.zz * v.z,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Constraint Interface ─────────────────────────────────────────────────────

export interface Constraint {
  /**
   * Pre-step: compute effective mass, bias, and any cached data.
   * @param bodies  The two rigid bodies involved in this constraint.
   * @param dt      The timestep in seconds.
   */
  prepare(bodies: [RigidBody, RigidBody], dt: number): void;

  /**
   * Velocity-level solve: apply one iteration of impulse projection.
   * @param bodies  The two rigid bodies involved in this constraint.
   * @returns deltaLambda – the change in Lagrange multiplier this iteration.
   */
  solve(bodies: [RigidBody, RigidBody]): number;
}

// ─── NonPenetrationConstraint ─────────────────────────────────────────────────
//
//  C   = dot(pB - pA, n) >= 0
//  Jv  = n · (vB + wB × rB) - n · (vA + wA × rA)
//  Baumgarte bias = -beta / dt * depth   (position correction)

export class NonPenetrationConstraint implements Constraint {
  private contact: ContactPoint;
  private beta: number;

  effectiveMass = 0;
  accumulatedLambda = 0;
  bias = 0;

  private rA: Vec3 = { x: 0, y: 0, z: 0 };
  private rB: Vec3 = { x: 0, y: 0, z: 0 };
  private jA: Vec3 = { x: 0, y: 0, z: 0 }; // angular Jacobian for body A
  private jB: Vec3 = { x: 0, y: 0, z: 0 }; // angular Jacobian for body B

  constructor(contact: ContactPoint, beta = 0.2) {
    this.contact = contact;
    this.beta = beta;
  }

  prepare(bodies: [RigidBody, RigidBody], dt: number): void {
    const [bodyA, bodyB] = bodies;
    const { normal, depth, pointA, pointB } = this.contact;
    const n = normal;

    // Moment arms from body centres of mass to contact points
    this.rA = sub(pointA, bodyA.position);
    this.rB = sub(pointB, bodyB.position);

    // Angular Jacobian components: r × n
    this.jA = cross(this.rA, n); // will negate when applying to A
    this.jB = cross(this.rB, n);

    // Effective mass: 1 / (mA⁻¹ + mB⁻¹ + (rA×n)·IA⁻¹(rA×n) + (rB×n)·IB⁻¹(rB×n))
    const angA = dot(this.jA, applyInverseInertia(bodyA.inverseInertia, this.jA));
    const angB = dot(this.jB, applyInverseInertia(bodyB.inverseInertia, this.jB));
    const denom = bodyA.inverseMass + bodyB.inverseMass + angA + angB;
    this.effectiveMass = denom > 0 ? 1 / denom : 0;

    // Baumgarte stabilisation bias (only if penetrating)
    this.bias = depth < 0 ? (-this.beta / dt) * depth : 0;

    // Warm-starting: re-apply accumulated impulse
    this.accumulatedLambda = Math.max(0, this.accumulatedLambda);
    this._applyImpulse(bodies, this.accumulatedLambda);
  }

  solve(bodies: [RigidBody, RigidBody]): number {
    const [bodyA, bodyB] = bodies;
    const n = this.contact.normal;

    const vA = add(bodyA.velocity, cross(bodyA.angularVelocity, this.rA));
    const vB = add(bodyB.velocity, cross(bodyB.angularVelocity, this.rB));
    const relV = dot(n, sub(vB, vA));

    const lambda = this.effectiveMass * (-relV + this.bias);
    const prevAccumulated = this.accumulatedLambda;
    this.accumulatedLambda = Math.max(0, this.accumulatedLambda + lambda);
    const deltaLambda = this.accumulatedLambda - prevAccumulated;

    this._applyImpulse(bodies, deltaLambda);
    return deltaLambda;
  }

  private _applyImpulse(bodies: [RigidBody, RigidBody], impulse: number): void {
    const [bodyA, bodyB] = bodies;
    const n = this.contact.normal;
    const p = scale(n, impulse);

    bodyA.velocity = sub(bodyA.velocity, scale(p, bodyA.inverseMass));
    bodyB.velocity = add(bodyB.velocity, scale(p, bodyB.inverseMass));

    bodyA.angularVelocity = sub(
      bodyA.angularVelocity,
      applyInverseInertia(bodyA.inverseInertia, scale(this.jA, impulse))
    );
    bodyB.angularVelocity = add(
      bodyB.angularVelocity,
      applyInverseInertia(bodyB.inverseInertia, scale(this.jB, impulse))
    );
  }
}

// ─── FrictionConstraint ───────────────────────────────────────────────────────
//
//  Two tangent directions (t1, t2) per contact point.
//  Impulse is clamped to [-mu * normalLambda, mu * normalLambda].

export class FrictionConstraint implements Constraint {
  private contact: ContactPoint;
  private mu: number;
  private normalConstraint: NonPenetrationConstraint;

  effectiveMass = 0;
  accumulatedLambda = 0;
  bias = 0;

  private tangent: Vec3 = { x: 1, y: 0, z: 0 };
  private rA: Vec3 = { x: 0, y: 0, z: 0 };
  private rB: Vec3 = { x: 0, y: 0, z: 0 };
  private jAngA: Vec3 = { x: 0, y: 0, z: 0 };
  private jAngB: Vec3 = { x: 0, y: 0, z: 0 };

  constructor(
    contact: ContactPoint,
    normalConstraint: NonPenetrationConstraint,
    mu = 0.5
  ) {
    this.contact = contact;
    this.normalConstraint = normalConstraint;
    this.mu = mu;
  }

  prepare(bodies: [RigidBody, RigidBody], _dt: number): void {
    const [bodyA, bodyB] = bodies;
    const n = this.contact.normal;

    // Build a tangent vector perpendicular to n (Gram-Schmidt)
    const arbitrary: Vec3 = Math.abs(n.x) < 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
    const t = cross(n, arbitrary);
    const tLen = Math.sqrt(dot(t, t));
    this.tangent = tLen > 1e-6 ? scale(t, 1 / tLen) : { x: 1, y: 0, z: 0 };

    this.rA = sub(this.contact.pointA, bodyA.position);
    this.rB = sub(this.contact.pointB, bodyB.position);

    this.jAngA = cross(this.rA, this.tangent);
    this.jAngB = cross(this.rB, this.tangent);

    const angA = dot(this.jAngA, applyInverseInertia(bodyA.inverseInertia, this.jAngA));
    const angB = dot(this.jAngB, applyInverseInertia(bodyB.inverseInertia, this.jAngB));
    const denom = bodyA.inverseMass + bodyB.inverseMass + angA + angB;
    this.effectiveMass = denom > 0 ? 1 / denom : 0;

    this.bias = 0; // no positional bias for friction
  }

  solve(bodies: [RigidBody, RigidBody]): number {
    const [bodyA, bodyB] = bodies;

    const vA = add(bodyA.velocity, cross(bodyA.angularVelocity, this.rA));
    const vB = add(bodyB.velocity, cross(bodyB.angularVelocity, this.rB));
    const relV = dot(this.tangent, sub(vB, vA));

    const lambda = this.effectiveMass * (-relV);

    // Coulomb friction cone clamp
    const maxFriction = this.mu * this.normalConstraint.accumulatedLambda;
    const prevAccumulated = this.accumulatedLambda;
    this.accumulatedLambda = clamp(this.accumulatedLambda + lambda, -maxFriction, maxFriction);
    const deltaLambda = this.accumulatedLambda - prevAccumulated;

    // Apply tangential impulse
    const p = scale(this.tangent, deltaLambda);
    bodyA.velocity = sub(bodyA.velocity, scale(p, bodyA.inverseMass));
    bodyB.velocity = add(bodyB.velocity, scale(p, bodyB.inverseMass));
    bodyA.angularVelocity = sub(
      bodyA.angularVelocity,
      applyInverseInertia(bodyA.inverseInertia, scale(this.jAngA, deltaLambda))
    );
    bodyB.angularVelocity = add(
      bodyB.angularVelocity,
      applyInverseInertia(bodyB.inverseInertia, scale(this.jAngB, deltaLambda))
    );

    return deltaLambda;
  }
}

// ─── RestitutionConstraint ────────────────────────────────────────────────────
//
//  Adds a bounce velocity bias when the closing speed exceeds a threshold.
//  bias = restitution * approachSpeed   (only applied once at prepare time)

export class RestitutionConstraint implements Constraint {
  private contact: ContactPoint;
  private restitution: number;
  private velocityThreshold: number;

  effectiveMass = 0;
  accumulatedLambda = 0;
  bias = 0;

  private rA: Vec3 = { x: 0, y: 0, z: 0 };
  private rB: Vec3 = { x: 0, y: 0, z: 0 };
  private jAngA: Vec3 = { x: 0, y: 0, z: 0 };
  private jAngB: Vec3 = { x: 0, y: 0, z: 0 };

  constructor(contact: ContactPoint, restitution = 0.4, velocityThreshold = 1.0) {
    this.contact = contact;
    this.restitution = restitution;
    this.velocityThreshold = velocityThreshold;
  }

  prepare(bodies: [RigidBody, RigidBody], _dt: number): void {
    const [bodyA, bodyB] = bodies;
    const n = this.contact.normal;

    this.rA = sub(this.contact.pointA, bodyA.position);
    this.rB = sub(this.contact.pointB, bodyB.position);

    this.jAngA = cross(this.rA, n);
    this.jAngB = cross(this.rB, n);

    const angA = dot(this.jAngA, applyInverseInertia(bodyA.inverseInertia, this.jAngA));
    const angB = dot(this.jAngB, applyInverseInertia(bodyB.inverseInertia, this.jAngB));
    const denom = bodyA.inverseMass + bodyB.inverseMass + angA + angB;
    this.effectiveMass = denom > 0 ? 1 / denom : 0;

    // Compute relative velocity along normal at prepare time
    const vA = add(bodyA.velocity, cross(bodyA.angularVelocity, this.rA));
    const vB = add(bodyB.velocity, cross(bodyB.angularVelocity, this.rB));
    const approachSpeed = dot(n, sub(vA, vB)); // positive => bodies approaching

    this.bias = approachSpeed > this.velocityThreshold
      ? this.restitution * approachSpeed
      : 0;
  }

  solve(bodies: [RigidBody, RigidBody]): number {
    const [bodyA, bodyB] = bodies;
    const n = this.contact.normal;

    const vA = add(bodyA.velocity, cross(bodyA.angularVelocity, this.rA));
    const vB = add(bodyB.velocity, cross(bodyB.angularVelocity, this.rB));
    const relV = dot(n, sub(vB, vA));

    const lambda = this.effectiveMass * (-relV + this.bias);
    const prevAccumulated = this.accumulatedLambda;
    this.accumulatedLambda = Math.max(0, this.accumulatedLambda + lambda);
    const deltaLambda = this.accumulatedLambda - prevAccumulated;

    const p = scale(n, deltaLambda);
    bodyA.velocity = sub(bodyA.velocity, scale(p, bodyA.inverseMass));
    bodyB.velocity = add(bodyB.velocity, scale(p, bodyB.inverseMass));
    bodyA.angularVelocity = sub(
      bodyA.angularVelocity,
      applyInverseInertia(bodyA.inverseInertia, scale(this.jAngA, deltaLambda))
    );
    bodyB.angularVelocity = add(
      bodyB.angularVelocity,
      applyInverseInertia(bodyB.inverseInertia, scale(this.jAngB, deltaLambda))
    );

    return deltaLambda;
  }
}
