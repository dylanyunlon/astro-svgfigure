/**
 * physics-uniform-bridge.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Bridges the live SPH world state to per-cell shader uniforms.
 *
 * Reads per-cell physical environment from the running World, converts it into
 * normalised scalar/vector values, and returns a PhysicsUniforms bag that can
 * be fed directly to grayscott-species.frag and sdf-species-library.frag.
 *
 * Design notes
 * ─────────────────────────────────────────────────────────────────────────────
 * • All sampled values are normalised to [0, ∞) or clamped to a sensible
 *   display range so shaders receive consistent inputs regardless of sim scale.
 * • Vorticity is computed via finite-difference curl on the velocity field:
 *     ω = ∂vy/∂x − ∂vx/∂y
 *   approximated over neighbouring particles as a weighted average of the
 *   cross-product of the displacement and velocity vectors.
 * • The spatial hash stored in world._hash is reused when available; if null
 *   (world not yet stepped) we fall back to a brute-force O(N) linear scan.
 * • bodyId is the string form of RigidBody.id (number → string) to match the
 *   convention used by sph-bridge.ts (SPHFrameSnapshot.rigidBodies[].id).
 */

import type { World, Particle } from './world-stepper';
import type { RigidBody } from './rigid-body';

// ─── Public interface ─────────────────────────────────────────────────────────

export interface PhysicsUniforms {
  /** Average density of neighbouring fluid / rest density.  ≥ 0 */
  u_density: number;
  /** Average velocity of neighbouring fluid (world units/s). */
  u_velocity: [number, number];
  /** Average pressure of neighbouring fluid (world units). */
  u_pressure: number;
  /** Number of rigid-body contact points the body is currently involved in. */
  u_contactCount: number;
  /** Number of fluid particles within sampleRadius of the body centre. */
  u_neighborCount: number;
  /** Local kinetic energy = ½ Σ m |v|² over neighbours (normalised by count). */
  u_kineticEnergy: number;
  /**
   * Local vorticity (curl of velocity field):  ω = ∂vy/∂x − ∂vx/∂y
   * Positive = counter-clockwise rotation.  Clamped to [-100, 100].
   */
  u_vorticity: number;
  /** Simulation time (world.time). */
  u_time: number;
}

// ─── Zero / fallback value ────────────────────────────────────────────────────

const ZERO_UNIFORMS: PhysicsUniforms = {
  u_density:      0,
  u_velocity:     [0, 0],
  u_pressure:     0,
  u_contactCount: 0,
  u_neighborCount: 0,
  u_kineticEnergy: 0,
  u_vorticity:    0,
  u_time:         0,
};

// ─── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Collect all fluid Particle objects within `radius` of world-space point
 * (cx, cy).  Uses world._hash (SpatialHashGrid from spatial-hash.ts) when
 * available for O(1) average per query; otherwise performs a linear scan.
 */
function collectNeighbours(
  world: World,
  cx: number,
  cy: number,
  radius: number,
): Particle[] {
  const r2 = radius * radius;
  const result: Particle[] = [];

  // Fast path — reuse the frame's spatial hash built by substep().
  // world._hash is typed as SpatialHash (from spatial-hash.ts) which exposes
  // a query(x, y): number[] returning candidate particle indices.
  if (world._hash) {
    // The SpatialHash from spatial-hash.ts returns indices into the combined
    // [particles, wallParticles] array that was used to build it.  We only
    // care about fluid particles here.
    const all: Particle[] = [...world.particles, ...world.wallParticles];
    // query() returns candidate indices (cells within the hash cell range);
    // we must still distance-filter to the exact radius.
    // @ts-ignore — _hash is typed as SpatialHash (opaque interface); we call
    // the documented query(x, y) method exposed by the spatial-hash.ts impl.
    const candidates: number[] = (world._hash as { query(x: number, y: number): number[] }).query(cx, cy);
    for (const idx of candidates) {
      const p = all[idx];
      if (!p) continue;
      const dx = p.x - cx;
      const dy = p.y - cy;
      if (dx * dx + dy * dy <= r2) {
        result.push(p);
      }
    }
    return result;
  }

  // Slow path — linear scan over all fluid particles.
  for (const p of world.particles) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    if (dx * dx + dy * dy <= r2) {
      result.push(p);
    }
  }
  return result;
}

/**
 * Count how many active contact manifold points involve `bodyId`.
 *
 * CollisionWorld.step() returns the manifolds array; after the step the
 * result is stored in the private `_collisionWorld.manifolds` field.
 * We access it via the return value of the last step call stored in
 * world._collisionWorld.  Since `manifolds` is private we use `getStats()`
 * for the global count and fall back to 0 per-body rather than crashing.
 *
 * When the WorldV2 variant is used the collision world is exposed as
 * `world.collisionWorld`; we check both spellings.
 */
function countContactsForBody(world: World, body: RigidBody): number {
  // Attempt to access the last manifold array.  The CollisionWorld caches
  // the manifolds from the last step() call in a private field; TypeScript
  // prevents direct access, so we walk the prototype chain via a type cast.
  const cw =
    (world as unknown as { collisionWorld?: { manifolds?: unknown[] } }).collisionWorld ??
    (world as unknown as { _collisionWorld?: { manifolds?: unknown[] } })._collisionWorld;

  if (!cw) return 0;

  // If the internal `manifolds` array is accessible via duck-typing, use it.
  const manifolds = (cw as { manifolds?: unknown[] }).manifolds;
  if (!Array.isArray(manifolds)) return 0;

  let count = 0;
  for (const m of manifolds) {
    // ContactManifold has bodyA: Body (with .id) and points: ContactPoint[]
    // OR the collision-world-patched version with handleA/handleB.
    const mf = m as {
      bodyA?: { id?: number };
      bodyB?: { id?: number };
      handleA?: number;
      handleB?: number;
      contacts?: unknown[];
      points?: unknown[];
    };

    const pointCount =
      (Array.isArray(mf.contacts) ? mf.contacts.length : 0) +
      (Array.isArray(mf.points)   ? mf.points.length   : 0);
    if (pointCount === 0) continue;

    const aId = mf.bodyA?.id ?? mf.handleA;
    const bId = mf.bodyB?.id ?? mf.handleB;

    if (aId === body.id || bId === body.id) {
      count += pointCount;
    }
  }
  return count;
}

/**
 * Estimate vorticity (ω = ∂vy/∂x − ∂vx/∂y) at point (cx, cy) from a
 * discrete particle neighbourhood.
 *
 * Each neighbour j contributes to the curl estimate via a
 * SPH-inspired weighted finite difference:
 *
 *   ω ≈ (1 / Σ W_j) · Σ  W_j · [(r̂ × Δv) / |r|]
 *
 * where r̂ = (dx, dy)/|r|, Δv = (vxj − vxi, vyj − vyi), and the
 * "cross product" in 2-D returns the z-component:
 *   r̂ × Δv = dx/|r| · Δvy − dy/|r| · Δvx
 *
 * If there are no neighbours the vorticity is 0.
 */
function estimateVorticity(
  neighbours: Particle[],
  cx: number,
  cy: number,
  avgVx: number,
  avgVy: number,
): number {
  if (neighbours.length < 2) return 0;

  let vorticitySum = 0;
  let weightSum = 0;

  for (const p of neighbours) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const r = Math.sqrt(dx * dx + dy * dy);
    if (r < 1e-6) continue;

    // Weight: simple inverse-distance (linear falloff)
    const w = 1.0 / r;

    // Velocity delta relative to neighbourhood mean
    const dvx = p.vx - avgVx;
    const dvy = p.vy - avgVy;

    // 2-D curl contribution: (dx * dvy - dy * dvx) / r²
    const curl = (dx * dvy - dy * dvx) / (r * r);

    vorticitySum += w * curl;
    weightSum    += w;
  }

  if (weightSum < 1e-9) return 0;

  const omega = vorticitySum / weightSum;
  // Clamp to a display-friendly range
  return Math.max(-100, Math.min(100, omega));
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Sample the physical environment around a single rigid body and return the
 * corresponding shader uniforms.
 *
 * @param world        The live SPH World (must have been stepped at least once
 *                     for useful results; safe to call before first step).
 * @param bodyId       String form of RigidBody.id (e.g. String(rb.id)).
 * @param sampleRadius World-space radius around the body centre to include
 *                     fluid particles.  Typical: 2–4 × config.smoothingRadius.
 */
export function samplePhysicsForBody(
  world: World,
  bodyId: string,
  sampleRadius: number,
): PhysicsUniforms {
  // Locate the target rigid body.
  const body = world.rigidBodies.find((rb) => String(rb.id) === bodyId);
  if (!body) {
    return { ...ZERO_UNIFORMS, u_time: world.time };
  }

  const cx = body.x;
  const cy = body.y;
  const restDensity = world.config.restDensity;

  // Gather neighbouring fluid particles.
  const neighbours = collectNeighbours(world, cx, cy, sampleRadius);
  const n = neighbours.length;

  if (n === 0) {
    return {
      ...ZERO_UNIFORMS,
      u_contactCount: countContactsForBody(world, body),
      u_time:         world.time,
    };
  }

  // ── Accumulate per-particle quantities ──────────────────────────────────────
  let sumDensity  = 0;
  let sumVx       = 0;
  let sumVy       = 0;
  let sumPressure = 0;
  let sumKE       = 0;

  for (const p of neighbours) {
    sumDensity  += p.density;
    sumVx       += p.vx;
    sumVy       += p.vy;
    sumPressure += p.pressure;
    const speed2 = p.vx * p.vx + p.vy * p.vy;
    sumKE       += 0.5 * p.mass * speed2;
  }

  const avgDensity  = sumDensity  / n;
  const avgVx       = sumVx       / n;
  const avgVy       = sumVy       / n;
  const avgPressure = sumPressure / n;
  const avgKE       = sumKE       / n;

  // ── Normalised density ───────────────────────────────────────────────────────
  // Expressed as a ratio relative to rest density so shaders receive a
  // dimensionless value where 1.0 = equilibrium.
  const normDensity = restDensity > 0 ? avgDensity / restDensity : avgDensity;

  // ── Vorticity ────────────────────────────────────────────────────────────────
  const vorticity = estimateVorticity(neighbours, cx, cy, avgVx, avgVy);

  // ── Contact count ────────────────────────────────────────────────────────────
  const contactCount = countContactsForBody(world, body);

  return {
    u_density:       normDensity,
    u_velocity:      [avgVx, avgVy],
    u_pressure:      avgPressure,
    u_contactCount:  contactCount,
    u_neighborCount: n,
    u_kineticEnergy: avgKE,
    u_vorticity:     vorticity,
    u_time:          world.time,
  };
}

/**
 * Convenience wrapper — sample all rigid bodies in the world in one pass.
 *
 * @returns Map keyed by the string form of each RigidBody.id.
 */
export function sampleAllBodies(
  world: World,
  sampleRadius: number,
): Map<string, PhysicsUniforms> {
  const result = new Map<string, PhysicsUniforms>();

  for (const body of world.rigidBodies) {
    const key = String(body.id);
    result.set(key, samplePhysicsForBody(world, key, sampleRadius));
  }

  return result;
}
