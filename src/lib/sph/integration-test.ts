import { createWorldV2, stepWorldV2 } from './world-stepper';
import { QoSSpatialBridge } from './qos-spatial-bridge';

// ─── Constants ───────────────────────────────────────────────────────────────
const DOMAIN_MIN = { x: 0, y: 0, z: 0 };
const DOMAIN_MAX = { x: 20, y: 15, z: 10 };
const TARGET_DENSITY = 1000;
const DENSITY_TOLERANCE = 150; // ±15%
const MAX_RIGID_BODY_POSITION = 10000;
const FRAME_COUNT = 100;
const DT = 0.016; // ~60fps timestep

// ─── Helpers ─────────────────────────────────────────────────────────────────
function assertWithMessage(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`ASSERTION FAILED: ${message}`);
  }
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function hrNow(): number {
  // High-resolution time in milliseconds
  if (typeof process !== 'undefined' && process.hrtime) {
    const [sec, ns] = process.hrtime();
    return sec * 1000 + ns / 1e6;
  }
  return Date.now();
}

function hrElapsed(startMs: number): number {
  return hrNow() - startMs;
}

// ─── Dam-break fluid block config ────────────────────────────────────────────
function buildDamBreakFluidBlock() {
  return {
    type: 'fluid' as const,
    position: { x: 0.5, y: 0.5, z: 0.5 },
    dimensions: { width: 5, height: 8, depth: 9 },
    particleSpacing: 0.25,
    density: TARGET_DENSITY,
    viscosity: 0.001,
    pressure: 0,
  };
}

// ─── Rigid body configs ───────────────────────────────────────────────────────
function buildRigidBodies() {
  return [
    {
      id: 'rb_box_0',
      shape: 'box' as const,
      position: { x: 10, y: 2, z: 5 },
      dimensions: { width: 1.0, height: 1.0, depth: 1.0 },
      mass: 2.0,
      velocity: { x: 0, y: 0, z: 0 },
      angularVelocity: { x: 0, y: 0, z: 0 },
      restitution: 0.3,
    },
    {
      id: 'rb_sphere_1',
      shape: 'sphere' as const,
      position: { x: 12, y: 3, z: 5 },
      radius: 0.6,
      mass: 1.5,
      velocity: { x: -0.5, y: 0, z: 0 },
      angularVelocity: { x: 0, y: 0, z: 0 },
      restitution: 0.5,
    },
    {
      id: 'rb_box_2',
      shape: 'box' as const,
      position: { x: 8, y: 1.5, z: 3 },
      dimensions: { width: 0.8, height: 0.8, depth: 0.8 },
      mass: 3.0,
      velocity: { x: 0.2, y: 0, z: 0.1 },
      angularVelocity: { x: 0, y: 0.1, z: 0 },
      restitution: 0.2,
    },
    {
      id: 'rb_capsule_3',
      shape: 'capsule' as const,
      position: { x: 14, y: 4, z: 7 },
      radius: 0.4,
      height: 1.2,
      mass: 1.0,
      velocity: { x: -1.0, y: 0, z: -0.5 },
      angularVelocity: { x: 0, y: 0, z: 0 },
      restitution: 0.4,
    },
    {
      id: 'rb_box_4',
      shape: 'box' as const,
      position: { x: 16, y: 2, z: 4 },
      dimensions: { width: 1.5, height: 0.5, depth: 1.5 },
      mass: 4.0,
      velocity: { x: -0.3, y: 0, z: 0 },
      angularVelocity: { x: 0, y: 0, z: 0.05 },
      restitution: 0.1,
    },
    {
      id: 'rb_sphere_5',
      shape: 'sphere' as const,
      position: { x: 6, y: 5, z: 2 },
      radius: 0.5,
      mass: 0.8,
      velocity: { x: 0.8, y: -0.2, z: 0.3 },
      angularVelocity: { x: 0, y: 0, z: 0 },
      restitution: 0.6,
    },
    {
      id: 'rb_box_6',
      shape: 'box' as const,
      position: { x: 9, y: 6, z: 8 },
      dimensions: { width: 1.2, height: 1.2, depth: 0.6 },
      mass: 2.5,
      velocity: { x: 0.1, y: -0.5, z: -0.2 },
      angularVelocity: { x: 0.05, y: 0, z: 0 },
      restitution: 0.35,
    },
  ];
}

// ─── Main integration test ────────────────────────────────────────────────────
async function runIntegrationTest(): Promise<void> {
  console.log('='.repeat(60));
  console.log('SPH Integration Test — world-stepper + qos-spatial-bridge');
  console.log('='.repeat(60));

  // 1. Create world
  console.log('\n[1/5] Creating SPH world with createWorldV2...');
  const world = await createWorldV2({
    domain: { min: DOMAIN_MIN, max: DOMAIN_MAX },
    gravity: { x: 0, y: -9.81, z: 0 },
    smoothingLength: 0.5,
    timeStep: DT,
    enableQoS: true,
  });

  assertWithMessage(world !== null && world !== undefined, 'createWorldV2 must return a world object');
  console.log('  ✓ World created');

  // 2. Add dam-break fluid block
  console.log('\n[2/5] Adding dam-break fluid block...');
  const fluidBlock = buildDamBreakFluidBlock();
  const fluidResult = await world.addFluidBlock(fluidBlock);

  assertWithMessage(fluidResult.particleCount > 0, 'Fluid block must produce at least 1 particle');
  const particleCount = fluidResult.particleCount;
  console.log(`  ✓ Fluid block added — ${particleCount} particles`);

  // 3. Add 7 rigid bodies
  console.log('\n[3/5] Adding 7 rigid bodies...');
  const rigidBodyDefs = buildRigidBodies();
  assertWithMessage(rigidBodyDefs.length === 7, 'Must define exactly 7 rigid bodies');

  for (const rbDef of rigidBodyDefs) {
    const rbResult = await world.addRigidBody(rbDef);
    assertWithMessage(rbResult.id === rbDef.id, `Rigid body ${rbDef.id} must be registered with correct id`);
    console.log(`  ✓ Added rigid body: ${rbDef.id} (shape=${rbDef.shape})`);
  }

  // 4. Attach QoS spatial bridge
  console.log('\n[4/5] Attaching QoS spatial bridge...');
  const bridge = new QoSSpatialBridge(world);
  await bridge.initialize();
  console.log('  ✓ QoS spatial bridge initialized');

  // 5. Run 100 frames and collect metrics
  console.log(`\n[5/5] Stepping world for ${FRAME_COUNT} frames (dt=${DT}s)...`);

  let totalStepMs = 0;
  const frameTimes: number[] = [];

  for (let frame = 0; frame < FRAME_COUNT; frame++) {
    const t0 = hrNow();
    await stepWorldV2(world, DT);
    const elapsed = hrElapsed(t0);
    frameTimes.push(elapsed);
    totalStepMs += elapsed;

    if ((frame + 1) % 25 === 0) {
      console.log(`  Frame ${String(frame + 1).padStart(3)}: ${elapsed.toFixed(2)} ms`);
    }
  }

  const avgFrameMs = totalStepMs / FRAME_COUNT;
  const particlesPerSec = (particleCount * FRAME_COUNT) / (totalStepMs / 1000);
  const minFrameMs = Math.min(...frameTimes);
  const maxFrameMs = Math.max(...frameTimes);

  // ─── Assertions ─────────────────────────────────────────────────────────────
  console.log('\n' + '─'.repeat(60));
  console.log('Running assertions...');

  // A) All particles within domain
  const state = await world.getState();
  const particles: Array<{ x: number; y: number; z: number; density: number }> = state.particles;

  let outOfDomain = 0;
  for (const p of particles) {
    const inDomain =
      p.x >= DOMAIN_MIN.x && p.x <= DOMAIN_MAX.x &&
      p.y >= DOMAIN_MIN.y && p.y <= DOMAIN_MAX.y &&
      p.z >= DOMAIN_MIN.z && p.z <= DOMAIN_MAX.z;
    if (!inDomain) outOfDomain++;
  }

  assertWithMessage(
    outOfDomain === 0,
    `All particles must be within domain — found ${outOfDomain} outside [${JSON.stringify(DOMAIN_MIN)}, ${JSON.stringify(DOMAIN_MAX)}]`
  );
  console.log(`  ✓ All ${particles.length} particles within domain`);

  // B) Average density near 1000 kg/m³
  const avgDensity = particles.reduce((sum, p) => sum + p.density, 0) / particles.length;
  assertWithMessage(
    Math.abs(avgDensity - TARGET_DENSITY) <= DENSITY_TOLERANCE,
    `Avg density ${avgDensity.toFixed(1)} must be within ±${DENSITY_TOLERANCE} of ${TARGET_DENSITY} kg/m³`
  );
  console.log(`  ✓ Average density: ${avgDensity.toFixed(1)} kg/m³ (target=${TARGET_DENSITY}, tol=±${DENSITY_TOLERANCE})`);

  // C) No rigid body position component > 10000
  const rigidBodies: Array<{ id: string; position: { x: number; y: number; z: number } }> = state.rigidBodies;
  assertWithMessage(rigidBodies.length === 7, `Must have 7 rigid bodies in state, got ${rigidBodies.length}`);

  for (const rb of rigidBodies) {
    const { x, y, z } = rb.position;
    const maxPos = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    assertWithMessage(
      maxPos < MAX_RIGID_BODY_POSITION,
      `Rigid body ${rb.id} position (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) exceeds limit ${MAX_RIGID_BODY_POSITION}`
    );
  }
  console.log(`  ✓ All 7 rigid bodies within position limit (<${MAX_RIGID_BODY_POSITION})`);

  // ─── Timing Report ───────────────────────────────────────────────────────────
  console.log('\n' + '='.repeat(60));
  console.log('TIMING REPORT');
  console.log('='.repeat(60));
  console.log(`  Frames simulated : ${FRAME_COUNT}`);
  console.log(`  Particles        : ${particleCount}`);
  console.log(`  Total wall time  : ${totalStepMs.toFixed(1)} ms`);
  console.log(`  Avg ms/frame     : ${avgFrameMs.toFixed(2)} ms`);
  console.log(`  Min ms/frame     : ${minFrameMs.toFixed(2)} ms`);
  console.log(`  Max ms/frame     : ${maxFrameMs.toFixed(2)} ms`);
  console.log(`  Particles/sec    : ${particlesPerSec.toFixed(0)}`);
  console.log('='.repeat(60));
  console.log('\n✅ All assertions passed. Integration test COMPLETE.\n');
}

// ─── Entry point ─────────────────────────────────────────────────────────────
runIntegrationTest().catch((err) => {
  console.error('\n❌ Integration test FAILED:', err.message ?? err);
  process.exit(1);
});
