// rigid-body.ts
// 2D rigid body simulation for rectangular SPH transformer cells
// Symplectic Euler integration, AABB collision resolution, boundary sampling









export interface RigidBody {
  id: number;
  // Position
  x: number;
  y: number;
  // Linear velocity
  vx: number;
  vy: number;
  // Rotation
  angle: number;
  angVel: number;
  // Physical properties
  mass: number;
  inertia: number;        // moment of inertia (rectangle: m*(w^2+h^2)/12)
  w: number;              // half-width
  h: number;              // half-height
  restitution: number;    // coefficient of restitution [0,1]
  friction: number;       // surface friction coefficient [0,1]
  pinned: boolean;        // if true, body is static (infinite mass)
  species: number;        // Transformer cell type 0-6
  // Boundary particles (sampled perimeter points in world space)
  boundaryX: Float64Array;
  boundaryY: Float64Array;
  boundaryNx: Float64Array;  // outward normal x
  boundaryNy: Float64Array;  // outward normal y
  boundaryCount: number;
  // Accumulated forces & torque (reset each step)
  fx: number;
  fy: number;
  torque: number;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export interface RigidBodyOptions {
  vx?: number;
  vy?: number;
  angle?: number;
  angVel?: number;
  mass?: number;
  restitution?: number;
  friction?: number;
  pinned?: boolean;
  species?: number;
}

export function createRigidBody(
  id: number,
  x: number,
  y: number,
  w: number,    // full width
  h: number,    // full height
  opts: RigidBodyOptions = {}
): RigidBody {
  const mass = opts.pinned ? Infinity : (opts.mass ?? w * h * 1.0); // density=1
  // Rectangle moment of inertia: I = m*(w^2+h^2)/12
  const inertia = opts.pinned
    ? Infinity
    : (mass * (w * w + h * h)) / 12.0;

  const halfW = w * 0.5;
  const halfH = h * 0.5;

  // Pre-allocate boundary arrays (max perimeter / min spacing = generous bound)
  const maxBoundary = Math.ceil((2 * (w + h)) / 0.5) + 16;

  return {
    id,
    x, y,
    vx: opts.vx ?? 0,
    vy: opts.vy ?? 0,
    angle: opts.angle ?? 0,
    angVel: opts.angVel ?? 0,
    mass,
    inertia,
    w: halfW,   // store as half-extents
    h: halfH,
    restitution: opts.restitution ?? 0.3,
    friction: opts.friction ?? 0.1,
    pinned: opts.pinned ?? false,
    species: opts.species ?? 0,
    boundaryX:  new Float64Array(maxBoundary),
    boundaryY:  new Float64Array(maxBoundary),
    boundaryNx: new Float64Array(maxBoundary),
    boundaryNy: new Float64Array(maxBoundary),
    boundaryCount: 0,
    fx: 0, fy: 0, torque: 0,
  };
}

// ---------------------------------------------------------------------------
// Boundary particle sampling
// ---------------------------------------------------------------------------

/**
 * Walk the rectangle perimeter in local space, then transform to world space.
 * Outward normals are axis-aligned in local space, then rotated with the body.
 * spacing: arc-length between boundary sample points.
 */
export function sampleBoundaryParticles(rb: RigidBody, spacing: number): void {
  const hw = rb.w;   // half-width
  const hh = rb.h;   // half-height

  // Local-space corner walk: bottom-left -> bottom-right -> top-right ->
  //                          top-left -> bottom-left (CCW, y-up convention)
  // Sides: bottom, right, top, left
  // Each side: start corner, direction, outward normal (local)
  const sides: Array<[number, number, number, number, number, number]> = [
    // [startX, startY, dirX, dirY, normalX, normalY]
    [-hw, -hh,  1,  0,  0, -1],  // bottom  (y = -hh, outward normal -y)
    [ hw, -hh,  0,  1,  1,  0],  // right   (x = +hw, outward normal +x)
    [ hw,  hh, -1,  0,  0,  1],  // top     (y = +hh, outward normal +y)
    [-hw,  hh,  0, -1, -1,  0],  // left    (x = -hw, outward normal -x)
  ];

  // Side lengths
  const sideLengths = [2 * hw, 2 * hh, 2 * hw, 2 * hh];

  const cos = Math.cos(rb.angle);
  const sin = Math.sin(rb.angle);

  let count = 0;

  for (let s = 0; s < 4; s++) {
    const [sx, sy, dx, dy, nx, ny] = sides[s];
    const sideLen = sideLengths[s];
    const nSamples = Math.max(1, Math.floor(sideLen / spacing));
    const step = sideLen / nSamples;

    for (let i = 0; i < nSamples; i++) {
      // Local position: start + (i + 0.5) * step * dir
      const t = (i + 0.5) * step;
      const lx = sx + dx * t;
      const ly = sy + dy * t;

      // Rotate to world space
      const wx = rb.x + cos * lx - sin * ly;
      const wy = rb.y + sin * lx + cos * ly;

      // Rotate normal
      const wnx = cos * nx - sin * ny;
      const wny = sin * nx + cos * ny;

      if (count >= rb.boundaryX.length) break;  // safety guard

      rb.boundaryX[count]  = wx;
      rb.boundaryY[count]  = wy;
      rb.boundaryNx[count] = wnx;
      rb.boundaryNy[count] = wny;
      count++;
    }
    if (count >= rb.boundaryX.length) break;
  }

  rb.boundaryCount = count;
}

// ---------------------------------------------------------------------------
// Force accumulation
// ---------------------------------------------------------------------------

/**
 * Apply a force (fx, fy) at world-space point (px, py).
 * Contributes to linear force and torque about the centre of mass.
 */
export function applyForce(
  rb: RigidBody,
  fx: number,
  fy: number,
  px: number,
  py: number
): void {
  rb.fx += fx;
  rb.fy += fy;
  // Torque = r x F  (2D cross product: rx*fy - ry*fx)
  const rx = px - rb.x;
  const ry = py - rb.y;
  rb.torque += rx * fy - ry * fx;
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/**
 * Symplectic Euler integration for all non-pinned rigid bodies.
 * Clamps positions so bodies stay inside [0, domainW] x [0, domainH].
 * Gravity is applied as a body force.
 */
export function integrateRigidBodies(
  bodies: RigidBody[],
  dt: number,
  gravity: number,
  domainW: number,
  domainH: number
): void {
  for (const rb of bodies) {
    if (rb.pinned) continue;

    const invM = 1.0 / rb.mass;
    const invI = 1.0 / rb.inertia;

    // Symplectic Euler: update velocity first, then position
    rb.vx     += (rb.fx) * invM * dt;
    rb.vy     += (rb.fy + rb.mass * gravity) * invM * dt;
    rb.angVel += rb.torque * invI * dt;

    rb.x     += rb.vx * dt;
    rb.y     += rb.vy * dt;
    rb.angle += rb.angVel * dt;

    // Normalise angle to (-PI, PI]
    rb.angle = ((rb.angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

    // Domain boundary clamp with restitution bounce
    const hw = rb.w;
    const hh = rb.h;

    if (rb.x - hw < 0) {
      rb.x = hw;
      if (rb.vx < 0) rb.vx = -rb.vx * rb.restitution;
    } else if (rb.x + hw > domainW) {
      rb.x = domainW - hw;
      if (rb.vx > 0) rb.vx = -rb.vx * rb.restitution;
    }

    if (rb.y - hh < 0) {
      rb.y = hh;
      if (rb.vy < 0) rb.vy = -rb.vy * rb.restitution;
    } else if (rb.y + hh > domainH) {
      rb.y = domainH - hh;
      if (rb.vy > 0) rb.vy = -rb.vy * rb.restitution;
    }
  }
}

// ---------------------------------------------------------------------------
// Rigid-rigid collision resolution (AABB)
// ---------------------------------------------------------------------------

/**
 * Simple AABB overlap detection and separation impulse.
 * Uses axis-aligned bounding boxes (world-space extents of rotated rectangles).
 * For each overlapping pair, applies a separation impulse along the minimum
 * penetration axis and applies friction impulse tangentially.
 */
export function resolveRigidRigidCollisions(bodies: RigidBody[]): void {
  const n = bodies.length;

  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = bodies[i];
      const b = bodies[j];

      // Compute world-space AABB half-extents for rotated rectangles
      const cosA = Math.abs(Math.cos(a.angle));
      const sinA = Math.abs(Math.sin(a.angle));
      const aExtX = a.w * cosA + a.h * sinA;
      const aExtY = a.w * sinA + a.h * cosA;

      const cosB = Math.abs(Math.cos(b.angle));
      const sinB = Math.abs(Math.sin(b.angle));
      const bExtX = b.w * cosB + b.h * sinB;
      const bExtY = b.w * sinB + b.h * cosB;

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const overlapX = (aExtX + bExtX) - Math.abs(dx);
      const overlapY = (aExtY + bExtY) - Math.abs(dy);

      if (overlapX <= 0 || overlapY <= 0) continue;  // no AABB overlap

      // Minimum penetration axis
      let nx: number, ny: number, penetration: number;
      if (overlapX < overlapY) {
        penetration = overlapX;
        nx = dx < 0 ? -1 : 1;
        ny = 0;
      } else {
        penetration = overlapY;
        nx = 0;
        ny = dy < 0 ? -1 : 1;
      }

      // Separate bodies (push apart proportionally to inverse mass)
      const invMA = a.pinned ? 0 : 1.0 / a.mass;
      const invMB = b.pinned ? 0 : 1.0 / b.mass;
      const totalInvM = invMA + invMB;
      if (totalInvM === 0) continue;

      const sep = penetration / totalInvM;
      if (!a.pinned) { a.x -= nx * sep * invMA; a.y -= ny * sep * invMA; }
      if (!b.pinned) { b.x += nx * sep * invMB; b.y += ny * sep * invMB; }

      // Relative velocity at contact
      const rvx = b.vx - a.vx;
      const rvy = b.vy - a.vy;
      const vn  = rvx * nx + rvy * ny;

      if (vn > 0) continue;  // already separating

      // Restitution impulse
      const e = Math.min(a.restitution, b.restitution);
      const jn = -(1 + e) * vn / totalInvM;

      if (!a.pinned) { a.vx -= jn * invMA * nx; a.vy -= jn * invMA * ny; }
      if (!b.pinned) { b.vx += jn * invMB * nx; b.vy += jn * invMB * ny; }

      // Friction impulse (tangential)
      const tx = rvx - vn * nx;
      const ty = rvy - vn * ny;
      const tLen = Math.sqrt(tx * tx + ty * ty);
      if (tLen > 1e-10) {
        const tnx = tx / tLen;
        const tny = ty / tLen;
        const vt  = rvx * tnx + rvy * tny;
        const mu  = (a.friction + b.friction) * 0.5;
        const jt  = Math.max(-mu * Math.abs(jn), Math.min(mu * Math.abs(jn), -vt / totalInvM));
        if (!a.pinned) { a.vx -= jt * invMA * tnx; a.vy -= jt * invMA * tny; }
        if (!b.pinned) { b.vx += jt * invMB * tnx; b.vy += jt * invMB * tny; }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Force reset
// ---------------------------------------------------------------------------

/** Zero accumulated forces and torque before each physics step. */
export function resetForces(rb: RigidBody): void {
  rb.fx     = 0;
  rb.fy     = 0;
  rb.torque = 0;
}
