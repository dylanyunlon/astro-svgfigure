/**
 * world-stepper.ts
 * Main simulation loop orchestrating all physics modules.
 */


import { SpatialHash, buildSpatialHash, queryNeighbors, findNeighbors } from "./spatial-hash";
import { DFSPHSolver, solvePressure, applyPressureForces, Particle as DfsphParticle, pressureSolve as dfsphPressureSolve, divergenceSolve as dfsphDivergenceSolve } from "./dfsph-solver";
import { RigidBody, integrateRigidBody, applyImpulseToRigidBody, getRigidBodyParticles, integrateRigidBodies } from "./rigid-body";
import { applyBoundaryDensity, clampParticlesToBounds, BoundaryConfig } from "./world-boundary";
import { computeFluidRigidCoupling, transferMomentumToRigid } from "./fluid-rigid-coupling";
import { SpatialPhysics, QoSBridge, syncQoSParticles } from "./qos-spatial-bridge";
import { CollisionWorld, createCollisionWorld } from "./collision/collision-world";
import { SceneQuery, createSceneQuery } from "./collision/scene-query";
import { computeBoundaryDensity } from "./boundary";
import { stepDFSPH } from "./dfsph";
import { clampToDomain } from "./domain";
import { updateTrails } from "./trails";
import { PerformanceBudget } from "./performance-budget";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  ax: number;
  ay: number;
  density: number;
  pressure: number;
  mass: number;
  species: string;
  alpha: number; // trail opacity
}

export interface Emitter {
  x: number;
  y: number;
  dirX: number;
  dirY: number;
  rate: number;         // particles per second
  species: string;
  _accumulator: number; // internal: fractional particle debt
}

export interface WorldConfig {
  width: number;
  height: number;
  gravity: number;
  particleRadius: number;
  smoothingRadius: number;
  restDensity: number;
  viscosity: number;
  dt: number;           // base time step (seconds)
  substeps: number;
  maxParticles: number;
  trailLength: number;
  restitution: number;
}

export interface World {
  particles: Particle[];
  rigidBodies: RigidBody[];
  wallParticles: Particle[];
  config: WorldConfig;
  qos: SpatialPhysics;
  perfBudget: PerformanceBudget;
  frame: number;
  time: number;
  substeps: number;
  emitters: Emitter[];
  trails: Map<number, Array<{ x: number; y: number }>>;
  _hash: SpatialHash | null;
  _solver: DFSPHSolver;
  _collisionWorld: CollisionWorld;
  _nextParticleId: number;
  _nextEmitterId: number;
}

// ---------------------------------------------------------------------------
// Default world configuration
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: Omit<WorldConfig, "width" | "height"> = {
  gravity: 9.81,
  particleRadius: 4,
  smoothingRadius: 12,
  restDensity: 1000,
  viscosity: 0.01,
  dt: 1 / 60,
  substeps: 3,
  maxParticles: 8000,
  trailLength: 20,
  restitution: 0.3,
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Creates a new empty World with sensible defaults.
 */
export function createWorld(
  width: number,
  height: number,
  qos: SpatialPhysics
): World {
  const perfBudget = new PerformanceBudget('HIGH');
  const config: WorldConfig = {
    ...DEFAULT_CONFIG,
    width,
    height,
    substeps: perfBudget.config.substeps,
  };

  const solver: DFSPHSolver = {
    iterations: 50,
    errorThreshold: 0.01,
    omega: 0.5,
  };

  return {
    particles: [],
    rigidBodies: [],
    wallParticles: [],
    config,
    qos,
    perfBudget,
    frame: 0,
    time: 0,
    substeps: config.substeps,
    emitters: [],
    trails: new Map(),
    _hash: null,
    _solver: solver,
    _collisionWorld: new CollisionWorld(),
    _nextParticleId: 0,
    _nextEmitterId: 0,
  };
}

// ---------------------------------------------------------------------------
// Particle helpers
// ---------------------------------------------------------------------------

function makeParticle(
  id: number,
  x: number,
  y: number,
  species: string,
  mass = 1.0
): Particle {
  return {
    id,
    x,
    y,
    vx: 0,
    vy: 0,
    ax: 0,
    ay: 0,
    density: 0,
    pressure: 0,
    mass,
    species,
    alpha: 1,
  };
}

// ---------------------------------------------------------------------------
// Public mutation helpers
// ---------------------------------------------------------------------------

/**
 * Fills a rectangular region with fluid particles on a grid.
 */
export function addFluidBlock(
  world: World,
  x: number,
  y: number,
  w: number,
  h: number,
  spacing: number,
  species: string
): void {
  const cols = Math.floor(w / spacing);
  const rows = Math.floor(h / spacing);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (world.particles.length >= world.config.maxParticles) return;

      const px = x + col * spacing + spacing * 0.5;
      const py = y + row * spacing + spacing * 0.5;
      const id = world._nextParticleId++;
      const p = makeParticle(id, px, py, species);
      world.particles.push(p);
      world.trails.set(id, []);
    }
  }
}

/**
 * Adds a rigid body (axis-aligned rectangle) to the world.
 */
export function addRigidBody(
  world: World,
  id: string,
  x: number,
  y: number,
  w: number,
  h: number,
  opts?: Partial<RigidBody>
): void {
  const body: RigidBody = {
    id,
    x,
    y,
    vx: 0,
    vy: 0,
    angle: 0,
    omega: 0,
    width: w,
    height: h,
    mass: opts?.mass ?? w * h * 0.5,
    inertia: opts?.inertia ?? ((w * h * 0.5) * (w * w + h * h)) / 12,
    isStatic: opts?.isStatic ?? false,
    restitution: opts?.restitution ?? world.config.restitution,
    friction: opts?.friction ?? 0.3,
    ...opts,
  };
  world.rigidBodies.push(body);
}

/**
 * Registers a particle emitter and returns its index.
 */
export function addEmitter(
  world: World,
  x: number,
  y: number,
  dirX: number,
  dirY: number,
  rate: number,
  species: string
): number {
  const emitter: Emitter = {
    x,
    y,
    dirX,
    dirY,
    rate,
    species,
    _accumulator: 0,
  };
  const idx = world._nextEmitterId++;
  world.emitters[idx] = emitter;
  return idx;
}

// ---------------------------------------------------------------------------
// Emitter tick
// ---------------------------------------------------------------------------

function tickEmitters(world: World, dt: number): void {
  const speed = 80; // emit velocity magnitude
  for (const emitter of world.emitters) {
    if (!emitter) continue;
    emitter._accumulator += emitter.rate * dt;
    while (emitter._accumulator >= 1 && world.particles.length < world.config.maxParticles) {
      emitter._accumulator -= 1;
      const id = world._nextParticleId++;
      const p = makeParticle(id, emitter.x, emitter.y, emitter.species);
      const len = Math.hypot(emitter.dirX, emitter.dirY) || 1;
      p.vx = (emitter.dirX / len) * speed;
      p.vy = (emitter.dirY / len) * speed;
      world.particles.push(p);
      world.trails.set(id, []);
    }
  }
}

// ---------------------------------------------------------------------------
// Trail update
// ---------------------------------------------------------------------------

function updateTrails(world: World): void {
  const maxLen = world.config.trailLength;
  for (const p of world.particles) {
    const trail = world.trails.get(p.id);
    if (!trail) continue;
    trail.push({ x: p.x, y: p.y });
    if (trail.length > maxLen) trail.shift();
  }

  // Remove trails for deleted particles
  for (const [id] of world.trails) {
    if (!world.particles.find((p) => p.id === id)) {
      world.trails.delete(id);
    }
  }
}

// ---------------------------------------------------------------------------
// Core substep
// ---------------------------------------------------------------------------

function substep(world: World, dt: number): void {
  const { config, particles, rigidBodies, wallParticles, qos, _solver } = world;
  const all = [...particles, ...wallParticles];

  // 1. Rebuild spatial hash
  world._hash = buildSpatialHash(all, config.smoothingRadius);

  // 2. Query neighbors for each fluid particle
  const neighborsMap = new Map<number, Particle[]>();
  for (const p of particles) {
    const neighbors = queryNeighbors(world._hash, p, config.smoothingRadius) as Particle[];
    neighborsMap.set(p.id, neighbors);
  }

  // 3. Boundary density contribution (wall particles inflate density near edges)
  applyBoundaryDensity(particles, wallParticles, world._hash, config);

  // 4. DFSPH pressure + divergence solve loops
  //
  // DFSPH (Bender & Koschier 2017) splits pressure correction into two nested
  // iterations: a density-error correction loop and a divergence-free loop.
  //
  //  · Pressure correction  — 3 iterations per frame
  //    Each iteration: density pass → force pass (GPU) + CPU pressure solve
  //    Drives  ρ*_i → ρ₀  by correcting predicted velocities.
  //
  //  · Divergence correction — 2 iterations per frame
  //    Each iteration: density pass → force pass (GPU) + CPU divergence solve
  //    Drives  div v_i → 0  (incompressibility in velocity field).
  //
  // The SPHGPUOrchestrator compute passes keep the GPU-side density / pressure
  // buffers in sync with the CPU state so the renderer always sees up-to-date
  // values without an extra readback.

  const PRESSURE_ITERS   = 3;
  const DIVERGENCE_ITERS = 2;

  // ── 4a. Pressure correction loop ──────────────────────────────────────
  for (let iter = 0; iter < PRESSURE_ITERS; iter++) {
    // CPU: run one DFSPH pressure solve step (updates particle.vx/vy).
    // Uses the correctly-imported pressureSolve from dfsph-solver.ts.
    dfsphPressureSolve(
      particles as unknown as DfsphParticle[],
      particles.map((p) => {
        const ns = neighborsMap.get(p.id) ?? [];
        return ns.map((n) => particles.indexOf(n));
      }),
      config.smoothingRadius,
      1.0,           // unit mass
      dt,
      config.restDensity,
      1,             // single iteration per outer loop step
    );

    // GPU: re-compute density + pressure from the corrected velocities so
    // the force pass operates on fresh data; forces are also recalculated.
    if ((world as any).orchestrator) {
      const orch     = (world as any).orchestrator as import("./SPHGPUOrchestrator").SPHGPUOrchestrator;
      const nbLists  = particles.map((p) => {
        const ns = neighborsMap.get(p.id) ?? [];
        return ns.map((n) => particles.indexOf(n));
      });
      orch.uploadNeighborLists(nbLists, particles.length);
      const gpuDevice: GPUDevice | undefined = (world as any)._gpuDevice;
      if (gpuDevice) {
        const enc = gpuDevice.createCommandEncoder({ label: `dfsph-pressure-iter-${iter}` });
        orch.encodeDensityPressure(enc, particles.length);
        orch.encodeForces(enc, particles.length);
        gpuDevice.queue.submit([enc.finish()]);
      }
    }
  }

  // Apply the pressure forces accumulated by the legacy CPU solver
  // (kept for backwards compatibility with the existing substep pipeline).
  applyPressureForces(particles, neighborsMap, config);

  // ── 4b. Divergence-free correction loop ───────────────────────────────
  for (let iter = 0; iter < DIVERGENCE_ITERS; iter++) {
    // CPU: run one DFSPH divergence solve step (corrects velocity divergence).
    dfsphDivergenceSolve(
      particles as unknown as DfsphParticle[],
      particles.map((p) => {
        const ns = neighborsMap.get(p.id) ?? [];
        return ns.map((n) => particles.indexOf(n));
      }),
      config.smoothingRadius,
      1.0,           // unit mass
      dt,
      config.restDensity,
      1,             // single iteration per outer loop step
    );

    // GPU: re-dispatch density + force passes to reflect divergence-corrected
    // velocities before the integrator commits the final positions.
    if ((world as any).orchestrator) {
      const orch    = (world as any).orchestrator as import("./SPHGPUOrchestrator").SPHGPUOrchestrator;
      const nbLists = particles.map((p) => {
        const ns = neighborsMap.get(p.id) ?? [];
        return ns.map((n) => particles.indexOf(n));
      });
      orch.uploadNeighborLists(nbLists, particles.length);
      const gpuDevice: GPUDevice | undefined = (world as any)._gpuDevice;
      if (gpuDevice) {
        const enc = gpuDevice.createCommandEncoder({ label: `dfsph-divergence-iter-${iter}` });
        orch.encodeDensityPressure(enc, particles.length);
        orch.encodeForces(enc, particles.length);
        gpuDevice.queue.submit([enc.finish()]);
      }
    }
  }

  // 5. Fluid–rigid coupling: compute coupling forces, transfer momentum
  const couplingForces = computeFluidRigidCoupling(
    particles,
    rigidBodies,
    neighborsMap,
    config
  );
  transferMomentumToRigid(rigidBodies, couplingForces, dt);

  // 6. Integrate fluid particles (symplectic Euler)
  for (const p of particles) {
    // gravity
    p.ay += config.gravity;

    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    // reset accelerations for next substep
    p.ax = 0;
    p.ay = 0;
  }

  // 7. Collision detection & resolution (before rigid body integration)
  world._collisionWorld.step(rigidBodies, dt);

  // 8. Integrate rigid bodies
  for (const body of rigidBodies) {
    if (!body.isStatic) {
      integrateRigidBody(body, dt, config.gravity);
    }
  }

  // 8. Clamp particles & rigid bodies to world bounds
  const boundary: BoundaryConfig = {
    width: config.width,
    height: config.height,
    restitution: config.restitution,
    particleRadius: config.particleRadius,
  };
  clampParticlesToBounds(particles, boundary);

  // 9. Sync with QoS spatial bridge
  syncQoSParticles(qos, particles);
}

// ---------------------------------------------------------------------------
// Public step entry-point
// ---------------------------------------------------------------------------

/**
 * Advances the world by one frame, running `world.substeps` substeps internally.
 */
export function stepWorld(world: World): void {
  const { config, perfBudget } = world;

  // Sync substeps from the current performance-budget tier
  world.substeps = perfBudget.config.substeps;

  const subDt = config.dt / world.substeps;

  // Emit new particles before physics
  tickEmitters(world, config.dt);

  for (let s = 0; s < world.substeps; s++) {
    substep(world, subDt);
  }

  // Record trail positions once per frame (after all substeps)
  updateTrails(world);

  world.frame++;
  world.time += config.dt;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

export interface WorldStats {
  particleCount: number;
  avgDensity: number;
  maxVelocity: number;
  kineticEnergy: number;
}

/**
 * Returns a snapshot of key simulation statistics for the current frame.
 */
export function getStats(world: World): WorldStats {
  const { particles } = world;
  const n = particles.length;

  if (n === 0) {
    return { particleCount: 0, avgDensity: 0, maxVelocity: 0, kineticEnergy: 0 };
  }

  let totalDensity = 0;
  let maxVelSq = 0;
  let kineticEnergy = 0;

  for (const p of particles) {
    totalDensity += p.density;

    const velSq = p.vx * p.vx + p.vy * p.vy;
    if (velSq > maxVelSq) maxVelSq = velSq;

    kineticEnergy += 0.5 * p.mass * velSq;
  }

  return {
    particleCount: n,
    avgDensity: totalDensity / n,
    maxVelocity: Math.sqrt(maxVelSq),
    kineticEnergy,
  };
}



// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WorldV2 extends World {
  collisionWorld: CollisionWorld;
  _sceneQuery: SceneQuery;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a WorldV2 — a World extended with a fully initialised CollisionWorld
 * and its companion SceneQuery proxy.
 */
export function createWorldV2(
  width: number,
  height: number,
  qos: SpatialPhysics,
): WorldV2 {
  const base = createWorld(width, height, qos);

  const collisionWorld = createCollisionWorld({
    gravity: base.gravity,
    iterations: qos.constraintIterations ?? 8,
    restitution: qos.globalRestitution ?? 0.2,
    friction: qos.globalFriction ?? 0.4,
    allowSleep: qos.allowSleep ?? true,
    sleepThreshold: qos.sleepThreshold ?? 0.01,
  });

  // Register every rigid body that already exists in the base world so the
  // collision world starts in a consistent state.
  for (const rb of base.rigidBodies) {
    collisionWorld.addBody(rb);
  }

  const _sceneQuery = createSceneQuery(collisionWorld);

  return { ...base, collisionWorld, _sceneQuery };
}

// ---------------------------------------------------------------------------
// Main step
// ---------------------------------------------------------------------------

/**
 * Full simulation step with collision pipeline.
 *
 * Execution order
 * ───────────────
 *  1. Spatial hash rebuild + neighbour lists
 *  2. Boundary density  (Akinci mirror particles + rigid boundary samples)
 *  3. DFSPH pressure / divergence solve
 *  4. Fluid→rigid coupling forces  (SPH pressure force on rigid surfaces)
 *  5. Sync rigid-body transforms → CollisionWorld
 *  6. collisionWorld.step()  broad→narrow→constraint solve
 *  7. Read corrected rigid-body velocities back from CollisionWorld
 *  8. Integrate rigid bodies  (with solver-corrected velocities)
 *  9. Domain clamping  (fluid particles + rigid COM)
 * 10. Trail update, frame counter increment
 */
export function stepWorldV2(world: WorldV2): void {
  const { dt, qos, collisionWorld } = world;

  // ── 1. Spatial hash + neighbour lists ─────────────────────────────────────
  buildSpatialHash(world);
  findNeighbors(world);

  // ── 2. Boundary density ───────────────────────────────────────────────────
  // Akinci-style: mirror-particle density contributions + any static boundary
  // particles that were registered with the rigid bodies.
  computeBoundaryDensity(world);

  // ── 3. DFSPH solve ────────────────────────────────────────────────────────
  // Updates fluid particle positions and velocities.  Rigid-body accelerations
  // from the previous coupling step are already baked into the predictor.
  stepDFSPH(world);

  // ── 4. Fluid–rigid coupling forces ────────────────────────────────────────
  // Accumulates SPH pressure and viscosity forces onto rigid body accumulators
  // (rb.force / rb.torque) without yet integrating them.
  computeFluidRigidCoupling(world);

  // ── 5. Push current rigid-body transforms → CollisionWorld ───────────────
  _syncTransformsToCollisionWorld(world);

  // ── 6. Collision pipeline  (broad→narrow→constraint solve) ───────────────
  // The constraint solver internally applies the coupling forces that were
  // accumulated in step 4, resolves penetrations, and corrects velocities.
  collisionWorld.step(dt);

  // ── 7. Read corrected rigid-body velocities back ──────────────────────────
  _syncVelocitiesFromCollisionWorld(world);

  // ── 8. Rigid-body integration ─────────────────────────────────────────────
  // Integrates positions/orientations using the solver-corrected velocities.
  integrateRigidBodies(world);

  // ── 9. Domain clamping ────────────────────────────────────────────────────
  clampToDomain(world);

  // ── 10. Bookkeeping ───────────────────────────────────────────────────────
  updateTrails(world);
  world.frame += 1;
}

// ---------------------------------------------------------------------------
// Scene-query proxy
// ---------------------------------------------------------------------------

/**
 * Return the SceneQuery interface for the world's CollisionWorld.
 * Callers can use this to do ray-casts, overlap tests, etc. without holding a
 * direct reference to the CollisionWorld.
 */
export function getSceneQuery(world: WorldV2): SceneQuery {
  return world._sceneQuery;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Push each rigid body's current kinematic state (position, orientation,
 * linear velocity, angular velocity) into the CollisionWorld so the broad- and
 * narrow-phase work with up-to-date AABBs and shapes.
 *
 * We also forward any accumulated coupling forces so the constraint solver can
 * take them into account when correcting velocities.
 */
function _syncTransformsToCollisionWorld(world: WorldV2): void {
  const { collisionWorld, rigidBodies } = world;

  for (const rb of rigidBodies) {
    collisionWorld.setBodyTransform(rb.id, {
      position: rb.position,
      orientation: rb.orientation,
      linearVelocity: rb.velocity,
      angularVelocity: rb.angularVelocity,
      // Accumulated coupling forces — the solver will integrate these during
      // the constraint phase and zero them out afterwards.
      externalForce: rb.force,
      externalTorque: rb.torque,
      // Pass through mass/inertia in case the body was mutated at runtime
      // (e.g. by adding boundary particles).
      mass: rb.mass,
      inertia: rb.inertia,
    });
  }
}

/**
 * After the CollisionWorld constraint solver has run, read back the corrected
 * linear and angular velocities for every rigid body.  These corrected values
 * are what the integrator in step 8 will use.
 *
 * Also zero the coupling-force accumulators so they don't double-count on the
 * next frame.
 */
function _syncVelocitiesFromCollisionWorld(world: WorldV2): void {
  const { collisionWorld, rigidBodies } = world;

  for (const rb of rigidBodies) {
    const solved = collisionWorld.getBodyState(rb.id);
    if (!solved) continue;

    rb.velocity.x = solved.linearVelocity.x;
    rb.velocity.y = solved.linearVelocity.y;

    if ('z' in rb.velocity && solved.linearVelocity.z !== undefined) {
      (rb.velocity as any).z = solved.linearVelocity.z;
    }

    rb.angularVelocity = solved.angularVelocity;

    // Zero accumulators — integrateRigidBodies will recompute from scratch.
    rb.force.x = 0;
    rb.force.y = 0;
    if ('z' in rb.force) (rb.force as any).z = 0;
    rb.torque = 0;
  }
}

// ---------------------------------------------------------------------------
// Re-exports from original world-stepper
// ---------------------------------------------------------------------------

// Re-exports removed — these functions are already exported where defined above.
