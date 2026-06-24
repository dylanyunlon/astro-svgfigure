// sph-worker.ts



// ── Types ────────────────────────────────────────────────────────────────────




import * as Comlink from "comlink";
import { WorldStepperV2, StepSnapshot } from "./world-stepper";
import { QoSSpatialBridge, QoSLevel } from "./qos-spatial-bridge";
import { ATRenderPipeline, type ATRenderPipelineOptions } from "./ATRenderPipeline";
import { PerformanceBudget, type Tier, type TierConfig } from "./performance-budget";
import { serializeWorld, deserializeWorld } from "./world-serializer";

export interface InitOptions {
  gravity: [number, number, number];
  bounds: { min: [number, number, number]; max: [number, number, number] };
  restDensity?: number;
  viscosity?: number;
  surfaceTension?: number;
  timeStep?: number;
  substeps?: number;
}

export interface FluidParams {
  id: string;
  positions: Float32Array;        // interleaved x,y,z
  velocities?: Float32Array;      // interleaved vx,vy,vz
  mass?: number;
  density?: number;
}

export interface BodyParams {
  id: string;
  type: "static" | "kinematic" | "dynamic";
  sdf: Float32Array;              // signed-distance field voxels
  sdfResolution: [number, number, number];
  transform: Float32Array;        // 4×4 column-major
  mass?: number;
  restitution?: number;
  friction?: number;
}

export interface EmitterParams {
  id: string;
  origin: [number, number, number];
  direction: [number, number, number];
  rate: number;                   // particles per second
  speed: number;
  radius: number;
  active?: boolean;
}

export interface WorkerSnapshot {
  positions: Float32Array;        // transferable
  velocities: Float32Array;       // transferable
  densities: Float32Array;        // transferable
  pressures: Float32Array;        // transferable
  particleCount: number;
  simTime: number;
  stepMs: number;
  tier: Tier;
  tierConfig: TierConfig;
}

export interface RaycastHit {
  hit: boolean;
  t: number;
  position: [number, number, number];
  normal: [number, number, number];
  particleIndex: number;
}

// ── Worker state ─────────────────────────────────────────────────────────────

let stepper: WorldStepperV2 | null = null;
let qosBridge: QoSSpatialBridge | null = null;
let perfBudget: PerformanceBudget | null = null;
let initialized = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

function assertInit(): asserts stepper is WorldStepperV2 {
  if (!initialized || !stepper) throw new Error("SPH worker not initialized — call init() first");
}

function cloneF32(src: Float32Array): Float32Array {
  return new Float32Array(src.buffer.slice(0));
}

// ── Exposed API ───────────────────────────────────────────────────────────────

const api = {
  // ── init ────────────────────────────────────────────────────────────────────
  init(opts: InitOptions): void {
    if (initialized) {
      stepper?.dispose();
      qosBridge?.dispose();
      perfBudget = null;
    }

    stepper = new WorldStepperV2({
      gravity:        opts.gravity,
      bounds:         opts.bounds,
      restDensity:    opts.restDensity    ?? 1000,
      viscosity:      opts.viscosity      ?? 0.001,
      surfaceTension: opts.surfaceTension ?? 0.0728,
      timeStep:       opts.timeStep       ?? 1 / 60,
      substeps:       opts.substeps       ?? 4,
    });

    qosBridge = new QoSSpatialBridge(stepper);

    // Adaptive performance budget — auto-adjusts substeps/particle caps per tier
    perfBudget = new PerformanceBudget('HIGH');
    perfBudget.onTierChange((next, _prev) => {
      if (!stepper) return;
      const cfg = PerformanceBudget.configFor(next);
      stepper.setSubsteps(cfg.substeps);
      stepper.setMaxParticles(cfg.maxParticles);
    });

    initialized = true;
  },

  // ── addFluid ─────────────────────────────────────────────────────────────────
  addFluid(params: FluidParams): string {
    assertInit();
    const id = stepper!.addFluidBlock({
      id:         params.id,
      positions:  params.positions,
      velocities: params.velocities ?? new Float32Array(params.positions.length),
      mass:       params.mass    ?? 0.02,
      density:    params.density ?? 1000,
    });
    return id;
  },

  // ── addBody ──────────────────────────────────────────────────────────────────
  addBody(params: BodyParams): string {
    assertInit();
    const id = stepper!.addRigidBody({
      id:             params.id,
      type:           params.type,
      sdf:            params.sdf,
      sdfResolution:  params.sdfResolution,
      transform:      params.transform,
      mass:           params.mass        ?? 1,
      restitution:    params.restitution ?? 0.3,
      friction:       params.friction    ?? 0.5,
    });
    return id;
  },

  // ── addEmitter ───────────────────────────────────────────────────────────────
  addEmitter(params: EmitterParams): string {
    assertInit();
    return stepper!.addEmitter({
      id:        params.id,
      origin:    params.origin,
      direction: params.direction,
      rate:      params.rate,
      speed:     params.speed,
      radius:    params.radius,
      active:    params.active ?? true,
    });
  },

  // ── step → snapshot (transferable) ──────────────────────────────────────────
  step(): Comlink.Transfer<WorkerSnapshot> {
    assertInit();

    const t0 = performance.now();
    const raw: StepSnapshot = stepper!.step();
    const stepMs = performance.now() - t0;

    // Feed frame timing to the adaptive performance budget
    perfBudget!.tick(stepMs);

    // Clone into fresh ArrayBuffers so we can transfer without detaching stepper's memory
    const positions  = cloneF32(raw.positions);
    const velocities = cloneF32(raw.velocities);
    const densities  = cloneF32(raw.densities);
    const pressures  = cloneF32(raw.pressures);

    const snapshot: WorkerSnapshot = {
      positions,
      velocities,
      densities,
      pressures,
      particleCount: raw.particleCount,
      simTime:       raw.simTime,
      stepMs,
      tier:          perfBudget!.tier,
      tierConfig:    perfBudget!.config,
    };

    return Comlink.transfer(snapshot, [
      positions.buffer,
      velocities.buffer,
      densities.buffer,
      pressures.buffer,
    ]);
  },

  // ── setQoS ───────────────────────────────────────────────────────────────────
  setQoS(level: QoSLevel, params?: Record<string, unknown>): void {
    assertInit();
    qosBridge!.setLevel(level, params);
  },

  // ── raycast ──────────────────────────────────────────────────────────────────
  raycast(
    origin: [number, number, number],
    direction: [number, number, number],
    maxDist = 100,
  ): RaycastHit {
    assertInit();
    const result = stepper!.raycast(origin, direction, maxDist);

    if (!result) {
      return { hit: false, t: -1, position: [0, 0, 0], normal: [0, 0, 0], particleIndex: -1 };
    }

    return {
      hit:           true,
      t:             result.t,
      position:      result.position,
      normal:        result.normal,
      particleIndex: result.particleIndex,
    };
  },

  // ── setTier ─────────────────────────────────────────────────────────────────
  setTier(tier: Tier): void {
    assertInit();
    perfBudget!.setTier(tier);
  },

  // ── getTierSnapshot ─────────────────────────────────────────────────────────
  getTierSnapshot(): {
    tier: Tier;
    fps: number;
    frameCount: number;
    config: TierConfig;
  } {
    assertInit();
    const snap = perfBudget!.snapshot();
    return {
      tier:       snap.tier,
      fps:        snap.fps,
      frameCount: snap.frameCount,
      config:     snap.config,
    };
  },

  // ── serialize → binary world snapshot (transferable) ────────────────────────
  serialize(): Comlink.Transfer<ArrayBuffer> {
    assertInit();
    const world = stepper!.getWorld();
    const buf = serializeWorld(world);
    return Comlink.transfer(buf, [buf]);
  },

  // ── deserialize ← restore world from binary snapshot ───────────────────────
  deserialize(buf: ArrayBuffer): void {
    assertInit();
    const world = deserializeWorld(buf);
    stepper!.restoreWorld(world);
  },
};

// ── Comlink expose ────────────────────────────────────────────────────────────

Comlink.expose(api);

export type SPHWorkerAPI = typeof api;
