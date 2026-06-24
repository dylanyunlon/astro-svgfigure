/**
 * test-scenes.ts — M789
 *
 * Five verifiable test scenes for the SPH / collision pipeline.
 * Each scene is self-contained and CPU-only: no WebGPU required.
 *
 * TestScene.setup()    — build initial conditions (particles, bodies, etc.)
 * TestScene.validate() — return a list of failure messages (empty = pass)
 */








import type { ParticleData, ObstacleData } from './types';
import { MAX_PARTICLES } from './types';
import {
} from './collision/CollisionWorld';
import { qosToSpatial } from './qosSpatial';
import { QOS_PRESETS } from './qosSpatial';
import type { QoSProfileName } from './qosSpatial';

  CollisionWorld,
  createCircleBody,
  createBoxBody,

// ─── TestScene interface ──────────────────────────────────────────────────────

export interface TestWorld {
  /** CPU-side particle state (fluid). */
  particles: ParticleData;
  /** Registered obstacles (mirrored into the boundary model). */
  obstacles: ObstacleData[];
  /** Rigid-body collision world. */
  collision: CollisionWorld;
  /** Simulation parameters snapshot (viscosity, gravity, dt, etc.). */
  params: {
    gravity: number;
    viscosity: number;
    dt: number;
    domainW: number;
    domainH: number;
  };
  /** Active QoS profile name. */
  qosProfile: QoSProfileName;
}

export interface TestScene {
  name: string;
  setup(world: TestWorld): void;
  validate(world: TestWorld, frame: number): string[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Spawn a rectangular block of fluid particles into the CPU-side arrays. */
function spawnFluidBlock(
  p: ParticleData,
  x0: number, y0: number,
  x1: number, y1: number,
  spacing: number,
  speciesId = 0,
): void {
  let n = p.count;
  for (let py = y0; py < y1; py += spacing) {
    for (let px = x0; px < x1; px += spacing) {
      if (n >= MAX_PARTICLES) return;
      p.x[n]       = px;
      p.y[n]       = py;
      p.vx[n]      = 0;
      p.vy[n]      = 0;
      p.species[n] = speciesId;
      n++;
    }
  }
  p.count = n;
}

/**
 * Minimal CPU-side gravity integration step for particles.
 * Applies gravity and simple floor/wall boundary clamping so we can
 * validate basic motion without the full GPU pipeline.
 */
function cpuStepParticles(world: TestWorld): void {
  const { particles: p, obstacles, params } = world;
  const { gravity, viscosity, dt, domainW, domainH } = params;
  const n = p.count;

  for (let i = 0; i < n; i++) {
    // Gravity (acts on vy; gravity is negative = downward)
    p.vy[i] += gravity * dt;

    // Simple viscous damping
    p.vx[i] *= (1 - viscosity * dt);
    p.vy[i] *= (1 - viscosity * dt);

    // Integrate position
    p.x[i] += p.vx[i] * dt;
    p.y[i] += p.vy[i] * dt;

    // Domain boundaries (reflect)
    if (p.x[i] < 0)       { p.x[i] = 0;       p.vx[i] = Math.abs(p.vx[i]) * 0.5; }
    if (p.x[i] > domainW) { p.x[i] = domainW; p.vx[i] = -Math.abs(p.vx[i]) * 0.5; }
    if (p.y[i] < 0)       { p.y[i] = 0;       p.vy[i] = Math.abs(p.vy[i]) * 0.5; }
    if (p.y[i] > domainH) { p.y[i] = domainH; p.vy[i] = -Math.abs(p.vy[i]) * 0.5; }

    // Obstacle repulsion (circle obstacles)
    for (const obs of obstacles) {
      const dx = p.x[i] - obs.cx;
      const dy = p.y[i] - obs.cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < obs.r && dist > 1e-8) {
        const nx = dx / dist;
        const ny = dy / dist;
        // Push particle out of the obstacle
        p.x[i] = obs.cx + nx * obs.r;
        p.y[i] = obs.cy + ny * obs.r;
        // Reflect velocity
        const vDotN = p.vx[i] * nx + p.vy[i] * ny;
        if (vDotN < 0) {
          p.vx[i] -= 2 * vDotN * nx;
          p.vy[i] -= 2 * vDotN * ny;
          // Dampen bounce
          p.vx[i] *= 0.7;
          p.vy[i] *= 0.7;
        }
      }
    }
  }
}

/** Compute the centre-of-mass Y for all particles. */
function avgY(p: ParticleData): number {
  if (p.count === 0) return 0;
  let sum = 0;
  for (let i = 0; i < p.count; i++) sum += p.y[i];
  return sum / p.count;
}

/** Compute the X-extent (max - min) of all particles. */
function xSpread(p: ParticleData): number {
  if (p.count === 0) return 0;
  let minX = Infinity;
  let maxX = -Infinity;
  for (let i = 0; i < p.count; i++) {
    if (p.x[i] < minX) minX = p.x[i];
    if (p.x[i] > maxX) maxX = p.x[i];
  }
  return maxX - minX;
}

// ─── Helper: create a fresh TestWorld ─────────────────────────────────────────

export function createTestWorld(overrides?: Partial<TestWorld['params']>): TestWorld {
  return {
    particles: {
      x:       new Float32Array(MAX_PARTICLES),
      y:       new Float32Array(MAX_PARTICLES),
      vx:      new Float32Array(MAX_PARTICLES),
      vy:      new Float32Array(MAX_PARTICLES),
      species: new Uint32Array(MAX_PARTICLES),
      count:   0,
    },
    obstacles: [],
    collision: new CollisionWorld(),
    params: {
      gravity:  -9.81,
      viscosity: 0.01,
      dt:        0.016,
      domainW:   4.0,
      domainH:   3.0,
      ...overrides,
    },
    qosProfile: 'DEFAULT',
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Scene 1 — gravity_drop
// ═══════════════════════════════════════════════════════════════════════════════

const gravityDrop: TestScene = {
  name: 'gravity_drop',

  setup(world) {
    // Place a small fluid block near the top of the domain.
    // Domain height is 3.0, so spawn at y ∈ [2.0, 2.5].
    spawnFluidBlock(world.particles, 1.5, 2.0, 2.5, 2.5, 0.1);
  },

  validate(world, frame) {
    const errors: string[] = [];

    if (frame < 2) return errors; // need at least a couple of frames

    // After several frames of gravity the average Y must be lower than the
    // initial spawn midpoint (2.25).  By frame 10+ it should be well below 2.0.
    const ay = avgY(world.particles);

    if (frame >= 5 && ay >= 2.25) {
      errors.push(
        `[gravity_drop] frame ${frame}: avg Y = ${ay.toFixed(3)} — ` +
        `expected < 2.25 (particles should have fallen)`,
      );
    }

    if (frame >= 20 && ay >= 1.5) {
      errors.push(
        `[gravity_drop] frame ${frame}: avg Y = ${ay.toFixed(3)} — ` +
        `expected < 1.5 after 20 frames of free fall`,
      );
    }

    // All particles must remain inside the domain.
    for (let i = 0; i < world.particles.count; i++) {
      if (world.particles.y[i] < -0.01 || world.particles.y[i] > world.params.domainH + 0.01) {
        errors.push(
          `[gravity_drop] frame ${frame}: particle ${i} escaped domain ` +
          `(y = ${world.particles.y[i].toFixed(3)})`,
        );
        break; // one example is enough
      }
    }

    return errors;
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Scene 2 — dam_break
// ═══════════════════════════════════════════════════════════════════════════════

const damBreak: TestScene = {
  name: 'dam_break',

  setup(world) {
    // Fluid column on the left side of the domain.
    // After "breaking" (releasing) it should spread rightward.
    spawnFluidBlock(world.particles, 0.1, 0.1, 0.8, 2.5, 0.1);
  },

  validate(world, frame) {
    const errors: string[] = [];

    if (frame < 3) return errors;

    const spread = xSpread(world.particles);

    // The initial X span is ≈ 0.7 (from 0.1 to 0.8).
    // After gravity pulls particles to the floor, horizontal pressure
    // should push them rightward.  By frame 15 the spread should exceed
    // the initial width noticeably.
    if (frame >= 15 && spread <= 0.8) {
      errors.push(
        `[dam_break] frame ${frame}: x-spread = ${spread.toFixed(3)} — ` +
        `expected > 0.8 (fluid should have spread horizontally)`,
      );
    }

    if (frame >= 40 && spread <= 1.2) {
      errors.push(
        `[dam_break] frame ${frame}: x-spread = ${spread.toFixed(3)} — ` +
        `expected > 1.2 after 40 frames of spreading`,
      );
    }

    return errors;
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Scene 3 — obstacle_bounce
// ═══════════════════════════════════════════════════════════════════════════════

const obstacleBounce: TestScene = {
  name: 'obstacle_bounce',

  setup(world) {
    // Place a circular obstacle at domain centre.
    const obs: ObstacleData = { cx: 2.0, cy: 1.0, r: 0.4, stiffness: 5000 };
    world.obstacles.push(obs);

    // Also mirror into collision world.
    const { body, shape } = createCircleBody(obs.cx, obs.cy, obs.r, 'static');
    world.collision.addBody(body, shape);

    // Drop a small cluster of particles directly above the obstacle.
    spawnFluidBlock(world.particles, 1.8, 2.2, 2.2, 2.6, 0.1);
  },

  validate(world, frame) {
    const errors: string[] = [];
    if (frame < 5) return errors;

    const obs = world.obstacles[0];
    let insideCount = 0;

    for (let i = 0; i < world.particles.count; i++) {
      const dx = world.particles.x[i] - obs.cx;
      const dy = world.particles.y[i] - obs.cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < obs.r * 0.9) {
        insideCount++;
      }
    }

    // After enough frames no particle should be deeply inside the obstacle
    // (the boundary reflection should have pushed them out).
    if (frame >= 10 && insideCount > 0) {
      errors.push(
        `[obstacle_bounce] frame ${frame}: ${insideCount} particle(s) still ` +
        `inside obstacle (r=${obs.r}) — boundary reflection failed`,
      );
    }

    // Particles that were dropped above the obstacle should have been
    // deflected outward.  Check that the x-spread is wider than the
    // initial cluster (0.4) — they should fan out around the obstacle.
    const spread = xSpread(world.particles);
    if (frame >= 20 && spread <= 0.5) {
      errors.push(
        `[obstacle_bounce] frame ${frame}: x-spread = ${spread.toFixed(3)} — ` +
        `expected > 0.5 (particles should deflect around obstacle)`,
      );
    }

    return errors;
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Scene 4 — qos_switch
// ═══════════════════════════════════════════════════════════════════════════════

const qosSwitch: TestScene = {
  name: 'qos_switch',

  setup(world) {
    // Start with DEFAULT profile.
    world.qosProfile = 'DEFAULT';
    const defaultSpatial = qosToSpatial(QOS_PRESETS['DEFAULT']);
    world.params.viscosity = defaultSpatial.viscosity;

    // Spawn particles for later validation.
    spawnFluidBlock(world.particles, 1.0, 1.0, 2.0, 2.0, 0.15);
  },

  validate(world, frame) {
    const errors: string[] = [];

    // At frame 10 we switch to SENSOR_DATA (mps=100 → much lower viscosity).
    if (frame === 10) {
      world.qosProfile = 'SENSOR_DATA';
      const sensorSpatial = qosToSpatial(QOS_PRESETS['SENSOR_DATA']);
      world.params.viscosity = sensorSpatial.viscosity;
    }

    const defaultSpatial = qosToSpatial(QOS_PRESETS['DEFAULT']);
    const sensorSpatial  = qosToSpatial(QOS_PRESETS['SENSOR_DATA']);

    // Before the switch (frames 0-9) the active viscosity must equal DEFAULT.
    if (frame >= 1 && frame < 10) {
      if (Math.abs(world.params.viscosity - defaultSpatial.viscosity) > 1e-6) {
        errors.push(
          `[qos_switch] frame ${frame}: viscosity = ${world.params.viscosity.toFixed(6)} — ` +
          `expected ${defaultSpatial.viscosity.toFixed(6)} (DEFAULT profile)`,
        );
      }
    }

    // After the switch (frames 11+) the viscosity must reflect SENSOR_DATA.
    if (frame > 10) {
      if (Math.abs(world.params.viscosity - sensorSpatial.viscosity) > 1e-6) {
        errors.push(
          `[qos_switch] frame ${frame}: viscosity = ${world.params.viscosity.toFixed(6)} — ` +
          `expected ${sensorSpatial.viscosity.toFixed(6)} (SENSOR_DATA profile)`,
        );
      }

      // The two profiles must actually differ for the test to be meaningful.
      if (Math.abs(defaultSpatial.viscosity - sensorSpatial.viscosity) < 1e-8) {
        errors.push(
          `[qos_switch] DEFAULT and SENSOR_DATA have identical viscosity ` +
          `(${defaultSpatial.viscosity}) — test is degenerate`,
        );
      }
    }

    return errors;
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Scene 5 — collision_detect
// ═══════════════════════════════════════════════════════════════════════════════

const collisionDetect: TestScene = {
  name: 'collision_detect',

  setup(world) {
    // Two dynamic box bodies placed so they will collide.
    // Body A sits at x=1, Body B at x=2, both at y=1.5.
    // Give body B a leftward velocity so it moves toward A.
    const { body: bodyA, shape: shapeA } = createBoxBody(1.0, 1.5, 0.3, 0.3, 'dynamic', {
      mass: 1.0, restitution: 0.5,
    });
    const { body: bodyB, shape: shapeB } = createBoxBody(2.0, 1.5, 0.3, 0.3, 'dynamic', {
      mass: 1.0, restitution: 0.5,
    });
    bodyB.velocity.x = -3.0; // move toward A

    // Place a static floor so bodies don't just fall forever.
    const { body: floor, shape: floorShape } = createBoxBody(2.0, 0.0, 4.0, 0.1, 'static');

    world.collision.addBody(bodyA, shapeA);
    world.collision.addBody(bodyB, shapeB);
    world.collision.addBody(floor, floorShape);
  },

  validate(world, frame) {
    const errors: string[] = [];
    if (frame < 2) return errors;

    // Step the collision world.
    world.collision.step(world.params.dt);

    const result = world.collision.exportCollisions();

    // By frame 5-10 the bodies should have met (gap ≈ 0.4, closing at 3 m/s
    // → ~0.13 s ≈ 8 frames at dt=0.016).  We check across a range.
    if (frame >= 5 && frame <= 30) {
      // Accumulate: if we see at least one contact in this window, good.
      // We stash a flag via a tiny hack: tag a particle species slot.
      if (result.count > 0) {
        // Mark that contact was observed at least once.
        world.particles.species[MAX_PARTICLES - 1] = 1;
      }
    }

    if (frame === 30) {
      const sawContact = world.particles.species[MAX_PARTICLES - 1] === 1;
      if (!sawContact) {
        errors.push(
          `[collision_detect] no contact detected between bodies in ` +
          `frames 5–30 — collision pipeline may be broken`,
        );
      }
    }

    // After many frames the bodies must not have diverged to infinity.
    if (frame >= 40) {
      const positions = world.collision.getPositions();
      for (const pos of positions) {
        if (Math.abs(pos.x) > 100 || Math.abs(pos.y) > 100) {
          errors.push(
            `[collision_detect] frame ${frame}: body ${pos.id} at ` +
            `(${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}) — diverged to infinity`,
          );
          break;
        }
      }
    }

    return errors;
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Registry
// ═══════════════════════════════════════════════════════════════════════════════

export const TEST_SCENES: TestScene[] = [
  gravityDrop,
  damBreak,
  obstacleBounce,
  qosSwitch,
  collisionDetect,
];

// ─── Runner ───────────────────────────────────────────────────────────────────

/**
 * Run all test scenes for the given number of frames and collect results.
 *
 * @param totalFrames  Number of simulation frames per scene (default 60).
 * @returns            Map of scene name → array of failure messages (empty = pass).
 */
export function runAllTestScenes(
  totalFrames = 60,
): Map<string, string[]> {
  const results = new Map<string, string[]>();

  for (const scene of TEST_SCENES) {
    const world = createTestWorld();
    scene.setup(world);

    const allErrors: string[] = [];

    for (let frame = 0; frame < totalFrames; frame++) {
      // Advance fluid particles (CPU integration).
      // The collision_detect scene drives its own collision world step
      // inside validate(), so we only do particle integration here for
      // the fluid-oriented scenes.
      if (scene.name !== 'collision_detect') {
        cpuStepParticles(world);
      }

      const frameErrors = scene.validate(world, frame);
      allErrors.push(...frameErrors);
    }

    results.set(scene.name, allErrors);
  }

  return results;
}
