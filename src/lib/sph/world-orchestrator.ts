/**
 * world-orchestrator.ts — 终极总控 (Ultimate World Orchestrator)
 *
 * M759: cell-pubsub-loop
 * ────────────────────────
 * Unified master scheduler that drives every subsystem of the simulation
 * in a single `requestAnimationFrame` loop using a **fixed-timestep
 * physics accumulator** with **variable-rate rendering**.
 *
 * Subsystem dispatch order (per frame):
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  1. Performance sampling  — measure real dt, compute smoothed FPS   │
 * │  2. Adaptive tuning       — adjust particle cap & iteration count   │
 * │  3. Emitter tick          — spawn new particles per emission pattern │
 * │  4. Fixed-step physics accumulator (may run 0–N sub-steps):         │
 * │     4a. SPH spatial hash rebuild (CPU)                              │
 * │     4b. SPH neighbour list build (CPU)                              │
 * │     4c. Force-field injection (CPU → GPU upload)                    │
 * │     4d. Collision world step (broad/narrow/solve/integrate)         │
 * │     4e. SPH GPU compute passes (density → pressure → forces → int) │
 * │     4f. Boundary enforcement                                        │
 * │     4g. GPU position read-back for next sub-step                    │
 * │  5. AT Render Pipeline tick (animation clocks)                      │
 * │  6. Environment FX tick                                             │
 * │  7. GPU render                                                      │
 * │     7a. Environment FX background pass                              │
 * │     7b. SPH particle render pass                                    │
 * │     7c. AT Render Pipeline (PBR + particles + water + VL + bloom)   │
 * │     7d. LUT grade → swap-chain                                      │
 * │  8. Audio bridge update (collision sonification)                    │
 * │  9. Stats export (FPS, particle count, collision count, phase times)│
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * Performance-adaptive strategy
 * ─────────────────────────────
 * A 60-sample sliding window tracks real frame times. When smoothed FPS
 * drops below the LOW threshold (default 30), the orchestrator:
 *   • Reduces the active particle cap by 10 % (clamped to a floor).
 *   • Lowers SPH iteration counts (density/pressure sub-iters).
 * When FPS recovers above the HIGH threshold (default 55):
 *   • Gradually restores the particle cap toward the configured maximum.
 *   • Raises iteration counts back to their nominal values.
 * This creates a closed-loop feedback system that maintains interactive
 * frame rates across a wide range of hardware.
 *
 * Usage:
 *   const orch = new WorldOrchestrator(canvas);
 *   await orch.init();                 // WebGPU + all subsystems
 *   orch.start();                      // begin rAF loop
 *   orch.addFluid(0.5, 0.5, 1.5, 1.5, 0.04, 0);
 *   // ...
 *   orch.pause();                      // suspend
 *   orch.destroy();                    // full teardown
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────


import { SpatialHashGrid }        from './SpatialHashGrid';
import { NeighborListBuilder }    from './NeighborListBuilder';
import { SPHGPUOrchestrator }     from './SPHGPUOrchestrator';
import { ParticleRenderer }       from './ParticleRenderer';
import { BoundaryModel }          from './BoundaryModel';
import { qosSpatial, type QoSProfileName } from './qosSpatial';
// [auto-fix empty import] import {
// [auto-fix empty import] } from './types';
// [auto-fix empty import] import {
// [auto-fix empty import] } from './collision/CollisionWorld';
import { SceneQuery }             from './collision/SceneQuery';
import { ATRenderPipeline, type ATRenderPipelineConfig } from './at-render-pipeline';
import { EnvironmentFx, type EnvironmentFxConfig }       from './environment-fx';
import { AudioPhysicsBridge }     from './audio-physics-bridge';
import type { EmitterConfig, EmissionPattern }           from './emitter-strategy';
import { ContinuousPattern }      from './emitter-strategy';

// [orphan5]   type GPUBufferSet, type SimParams, type ParticleData,
// [orphan5]   type ObstacleData, MAX_PARTICLES, WORKGROUP_SIZE,
// [orphan5]   CollisionWorld,
// [orphan5]   createCircleBody,
// [orphan5]   createBoxBody,

// ─────────────────────────────────────────────────────────────────────────────
// Constants & adaptive thresholds
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Fixed physics timestep in seconds (120 Hz physics). */
const FIXED_DT                   = 1 / 120;

/** Maximum real-time seconds the accumulator can bank before clamping.
 *  Prevents the "spiral of death" when frames take too long. */
const MAX_ACCUMULATOR            = FIXED_DT * 8;

/** FPS sliding-window length (frames). */
const FPS_WINDOW_SIZE            = 60;

/** Below this FPS, begin degrading quality to recover performance. */
const FPS_LOW_THRESHOLD          = 30;

/** Above this FPS, begin restoring quality toward nominal settings. */
const FPS_HIGH_THRESHOLD         = 55;

/** Minimum particle cap the adaptive system will allow. */
const MIN_ADAPTIVE_PARTICLES     = 500;

/** Per-step particle cap reduction factor when FPS is low. */
const PARTICLE_REDUCE_FACTOR     = 0.90;

/** Per-step particle cap recovery factor when FPS is high. */
const PARTICLE_RESTORE_FACTOR    = 1.02;

/** Minimum SPH iteration count (density + force sub-iters). */
const MIN_SPH_ITERS              = 1;

/** Nominal (maximum) SPH iteration count. */
const NOMINAL_SPH_ITERS          = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Force-field types (mirrored from SPHWorld for self-containment)
// ─────────────────────────────────────────────────────────────────────────────

interface ForceVector { dx: number; dy: number; dz: number }
interface CellBBox    { min: [number, number, number]; max: [number, number, number] }
interface CellEntry   { bbox: CellBBox; species: string; z: number }
interface CellForce   { minX: number; minY: number; maxX: number; maxY: number; fx: number; fy: number }

// ─────────────────────────────────────────────────────────────────────────────
// Emitter runtime state
// ─────────────────────────────────────────────────────────────────────────────

interface RuntimeEmitter extends EmitterConfig {
  /** Fractional particle debt carried across frames. */
  _accumulator: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats / telemetry
// ─────────────────────────────────────────────────────────────────────────────

export interface OrchestratorStats {
  /** Smoothed frames-per-second. */
  fps: number;
  /** Total active particle count. */
  particleCount: number;
  /** Current adaptive particle cap. */
  adaptiveParticleCap: number;
  /** Current SPH iteration count. */
  sphIterations: number;
  /** Number of collision contacts this frame. */
  collisionCount: number;
  /** Total rAF frames rendered since start(). */
  frameNumber: number;
  /** Elapsed simulation time in seconds. */
  simTime: number;
  /** Physics sub-steps executed this frame. */
  physicsStepsThisFrame: number;
  /** Phase timings in milliseconds. */
  timings: {
    physics: number;
    render: number;
    total: number;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface WorldOrchestratorConfig {
  /** QoS profile for SPH sim params. @default 'DEFAULT' */
  qosProfile?: QoSProfileName;
  /** Enable the AT render pipeline (PBR + particles + water + VL + bloom + LUT). */
  enableATRenderPipeline?: boolean;
  /** AT render pipeline config overrides. */
  atPipelineConfig?: ATRenderPipelineConfig;
  /** Enable environment background FX (brick-tile + voronoise + chromaAB). */
  enableEnvironmentFx?: boolean;
  /** Environment FX config overrides. */
  environmentFxConfig?: EnvironmentFxConfig;
  /** Enable audio sonification of physics events. */
  enableAudio?: boolean;
  /** Initial audio volume [0, 1]. @default 0.5 */
  audioVolume?: number;
  /** Override the fixed physics timestep. @default 1/120 */
  fixedDt?: number;
  /** Enable performance-adaptive quality scaling. @default true */
  enableAdaptive?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU buffer helpers (mirrored from SPHWorld)
// ─────────────────────────────────────────────────────────────────────────────

function makeStorageBuf(device: GPUDevice, byteLen: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size:  Math.max(byteLen, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

function allocParticleBuffers(device: GPUDevice): GPUBufferSet {
  const n4 = MAX_PARTICLES * 4;
  return {
    posX:     makeStorageBuf(device, n4, 'orch-posX'),
    posY:     makeStorageBuf(device, n4, 'orch-posY'),
    velX:     makeStorageBuf(device, n4, 'orch-velX'),
    velY:     makeStorageBuf(device, n4, 'orch-velY'),
    density:  makeStorageBuf(device, n4, 'orch-density'),
    pressure: makeStorageBuf(device, n4, 'orch-pressure'),
    forceX:   makeStorageBuf(device, n4, 'orch-forceX'),
    forceY:   makeStorageBuf(device, n4, 'orch-forceY'),
    species:  device.createBuffer({
      label: 'orch-species',
      size:  Math.max(n4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    count: device.createBuffer({
      label: 'orch-count',
      size:  4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}

function makeSimParams(domainW: number, domainH: number, profile: QoSProfileName): SimParams {
  const qos = qosSpatial[profile];
  return {
    domainW,
    domainH,
    h:           (qos as any).smoothingRadius ?? 0.08,
    dt:          (qos as any).dt ?? FIXED_DT,
    restDensity: 1000.0,
    stiffness:   200.0,
    viscosity:   0.01,
    gravity:     -9.81,
    ...qos,
  } as SimParams;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WorldOrchestrator
// ═══════════════════════════════════════════════════════════════════════════════

export class WorldOrchestrator {
  // ── Canvas & WebGPU ────────────────────────────────────────────────────────
  private canvas:   HTMLCanvasElement;
  private ctx!:     GPUCanvasContext;
  private adapter!: GPUAdapter;
  private device!:  GPUDevice;
  private format!:  GPUTextureFormat;

  // ── Core SPH subsystems ────────────────────────────────────────────────────
  private grid!:         SpatialHashGrid;
  private nlBuilder!:    NeighborListBuilder;
  private sphGPU!:       SPHGPUOrchestrator;
  private renderer!:     ParticleRenderer;
  private boundary!:     BoundaryModel;
  private gpuBufs!:      GPUBufferSet;

  // ── Collision ──────────────────────────────────────────────────────────────
  private collisionWorld!: CollisionWorld;

  // ── AT Render Pipeline (optional) ──────────────────────────────────────────
  private atPipeline: ATRenderPipeline | null = null;

  // ── Environment FX (optional) ──────────────────────────────────────────────
  private envFx: EnvironmentFx | null = null;

  // ── Audio bridge (optional) ────────────────────────────────────────────────
  private audio: AudioPhysicsBridge | null = null;

  // ── CPU particle state ─────────────────────────────────────────────────────
  private cpuPos: ParticleData = {
    x:       new Float32Array(MAX_PARTICLES),
    y:       new Float32Array(MAX_PARTICLES),
    vx:      new Float32Array(MAX_PARTICLES),
    vy:      new Float32Array(MAX_PARTICLES),
    species: new Uint32Array(MAX_PARTICLES),
    count:   0,
  };

  // ── Scene state ────────────────────────────────────────────────────────────
  private obstacles: ObstacleData[]      = [];
  private emitters:  RuntimeEmitter[]    = [];
  private params!:   SimParams;
  private domainW:   number;
  private domainH:   number;

  // ── Force-field ────────────────────────────────────────────────────────────
  private cellForces: CellForce[] = [];

  // ── Collision export (readable by external HUD) ────────────────────────────
  public lastCollisions: {
    collisions: Array<{ bodyA: number; bodyB: number; normal: { x: number; y: number }; depth: number }>;
    count: number;
  } = { collisions: [], count: 0 };

  // ── Loop state ─────────────────────────────────────────────────────────────
  private running          = false;
  private rafHandle        = 0;
  private frameNumber      = 0;
  private simTime          = 0;
  private lastRealTime     = 0;
  private accumulator      = 0;

  // ── Fixed timestep ─────────────────────────────────────────────────────────
  private fixedDt: number;

  // ── Performance adaptive ───────────────────────────────────────────────────
  private enableAdaptive:       boolean;
  private fpsWindow:            Float64Array;
  private fpsWindowIdx          = 0;
  private fpsWindowFilled       = false;
  private smoothedFps           = 60;
  private adaptiveParticleCap:  number;
  private adaptiveSphIters:     number;
  private nominalParticleCap:   number;

  // ── Phase timing (ms) ─────────────────────────────────────────────────────
  private _tPhysics = 0;
  private _tRender  = 0;
  private _tTotal   = 0;
  private _physicsStepsThisFrame = 0;

  // ── Config ─────────────────────────────────────────────────────────────────
  private cfg: Required<WorldOrchestratorConfig>;

  // ═══════════════════════════════════════════════════════════════════════════
  //  Constructor
  // ═══════════════════════════════════════════════════════════════════════════

  constructor(canvas: HTMLCanvasElement, config: WorldOrchestratorConfig = {}) {
    this.canvas  = canvas;
    this.domainH = 3.0;
    this.domainW = 3.0 * (canvas.width / canvas.height);

    this.cfg = {
      qosProfile:              config.qosProfile              ?? 'DEFAULT',
      enableATRenderPipeline:  config.enableATRenderPipeline  ?? false,
      atPipelineConfig:        config.atPipelineConfig        ?? {},
      enableEnvironmentFx:     config.enableEnvironmentFx     ?? false,
      environmentFxConfig:     config.environmentFxConfig      ?? {},
      enableAudio:             config.enableAudio             ?? false,
      audioVolume:             config.audioVolume             ?? 0.5,
      fixedDt:                 config.fixedDt                 ?? FIXED_DT,
      enableAdaptive:          config.enableAdaptive          ?? true,
    };

    this.fixedDt              = this.cfg.fixedDt;
    this.enableAdaptive       = this.cfg.enableAdaptive;
    this.nominalParticleCap   = MAX_PARTICLES;
    this.adaptiveParticleCap  = MAX_PARTICLES;
    this.adaptiveSphIters     = NOMINAL_SPH_ITERS;
    this.fpsWindow            = new Float64Array(FPS_WINDOW_SIZE);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Initialisation
  // ═══════════════════════════════════════════════════════════════════════════

  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU not supported.');

    // ── Adapter + Device ─────────────────────────────────────────────────────
    this.adapter = (await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    }))!;
    if (!this.adapter) throw new Error('No WebGPU adapter found.');

    this.device = await this.adapter.requestDevice({
      label: 'Orchestrator-Device',
      requiredLimits: {
        maxStorageBufferBindingSize:       this.adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize:                     this.adapter.limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension:  this.adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });
    this.device.lost.then((info) => {
      console.error('[Orchestrator] WebGPU device lost:', info.message);
      this.running = false;
    });

    // ── Canvas context ───────────────────────────────────────────────────────
    this.ctx    = this.canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({
      device:    this.device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    // ── SimParams ────────────────────────────────────────────────────────────
    this.params = makeSimParams(this.domainW, this.domainH, this.cfg.qosProfile);

    // ── GPU buffers ──────────────────────────────────────────────────────────
    this.gpuBufs = allocParticleBuffers(this.device);

    // ── SPH subsystems ───────────────────────────────────────────────────────
    this.grid = new SpatialHashGrid(
      this.params.h,
      this.domainW,
      this.domainH,
    );

    this.nlBuilder = new NeighborListBuilder(
      this.grid,
      this.params.h,
    );

    this.sphGPU = new SPHGPUOrchestrator(
      this.device,
      this.gpuBufs,
      this.params,
    );
    await this.sphGPU.init();

    this.renderer = new ParticleRenderer(
      this.device,
      this.format,
      this.gpuBufs,
      this.params,
    );
    await this.renderer.init();

    this.boundary = new BoundaryModel(
      this.domainW,
      this.domainH,
      this.params.h,
    );

    // ── Collision world ──────────────────────────────────────────────────────
    this.collisionWorld = new CollisionWorld();
    await this._syncCellRegistryBodies();

    // ── Force-field bridge (non-blocking) ────────────────────────────────────
    this._loadForceField().catch((err) =>
      console.warn('[Orchestrator] force_field load failed:', err),
    );

    // ── AT Render Pipeline (optional) ────────────────────────────────────────
    if (this.cfg.enableATRenderPipeline) {
      try {
        this.atPipeline = await ATRenderPipeline.create(
          this.device,
          this.canvas,
          [],   // flowerEdges — caller can supply via atPipelineConfig
          [],   // splineEdges
          this.cfg.atPipelineConfig,
        );
      } catch (err) {
        console.warn('[Orchestrator] AT Render Pipeline init failed:', err);
        this.atPipeline = null;
      }
    }

    // ── Environment FX (optional) ────────────────────────────────────────────
    if (this.cfg.enableEnvironmentFx) {
      try {
        this.envFx = await (EnvironmentFx as any).create(
          this.device,
          this.format,
          this.canvas.width,
          this.canvas.height,
        );
        if (this.cfg.environmentFxConfig) {
          (this.envFx as any).setConfig?.(this.cfg.environmentFxConfig);
        }
      } catch (err) {
        console.warn('[Orchestrator] Environment FX init failed:', err);
        this.envFx = null;
      }
    }

    // ── Audio bridge (optional) ──────────────────────────────────────────────
    if (this.cfg.enableAudio) {
      try {
        this.audio = new AudioPhysicsBridge();
        this.audio.setVolume(this.cfg.audioVolume);
      } catch (err) {
        console.warn('[Orchestrator] Audio bridge init failed:', err);
        this.audio = null;
      }
    }

    console.info(
      `[Orchestrator] init complete — domain ${this.domainW.toFixed(2)}×${this.domainH.toFixed(2)}, ` +
      `fixedDt=${this.fixedDt}, adaptive=${this.enableAdaptive}, ` +
      `AT=${!!this.atPipeline}, envFx=${!!this.envFx}, audio=${!!this.audio}`,
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Public API — Loop control
  // ═══════════════════════════════════════════════════════════════════════════

  /** Start the requestAnimationFrame render loop. */
  start(): void {
    if (this.running) return;
    this.running      = true;
    this.lastRealTime = performance.now();
    this.accumulator  = 0;
    this._loop(this.lastRealTime);
  }

  /** Pause without destroying GPU resources. */
  pause(): void {
    this.running = false;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }
  }

  /** Full teardown — the instance must not be reused. */
  destroy(): void {
    this.running = false;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }

    // GPU buffers
    if (this.gpuBufs) {
      (Object.values(this.gpuBufs) as GPUBuffer[]).forEach((b) => {
        try { b.destroy(); } catch { /* already gone */ }
      });
    }

    this.sphGPU?.destroy();
    this.renderer?.destroy();
    this.atPipeline?.destroy();
    (this.envFx as any)?.destroy?.();
    this.audio?.dispose();
    this.device?.destroy();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Public API — Scene manipulation
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Spawn fluid particles in a rectangular region.
   * Respects the current adaptive particle cap.
   */
  addFluid(
    x0: number, y0: number, x1: number, y1: number,
    spacing = 0.05, speciesId = 0,
  ): void {
    const { x, y, vx, vy, species } = this.cpuPos;
    let n   = this.cpuPos.count;
    const cap = Math.min(this.adaptiveParticleCap, MAX_PARTICLES);

    for (let py = y0; py < y1; py += spacing) {
      for (let px = x0; px < x1; px += spacing) {
        if (n >= cap) break;
        x[n]       = px;
        y[n]       = py;
        vx[n]      = 0;
        vy[n]      = 0;
        species[n] = speciesId;
        n++;
      }
      if (n >= cap) break;
    }

    this.cpuPos.count = n;
    this._uploadParticles();
  }

  /** Register a static obstacle into the boundary and collision models. */
  addObstacle(obstacle: ObstacleData): void {
    this.obstacles.push(obstacle);
    this.boundary.addObstacle(obstacle);
    this.sphGPU.updateObstacles(this.obstacles);

    const { body, shape } = createCircleBody(
      obstacle.cx, obstacle.cy, obstacle.r, 'static',
    );
    this.collisionWorld.addBody(body, shape);
  }

  /**
   * Register a particle emitter with a configurable emission pattern.
   * Returns an opaque emitter index for later removal.
   */
  addEmitter(config: EmitterConfig): number {
    const rt: RuntimeEmitter = {
      ...config,
      pattern:      config.pattern ?? new ContinuousPattern(),
      _accumulator: 0,
    };
    this.emitters.push(rt);
    return this.emitters.length - 1;
  }

  /** Remove an emitter by its index. */
  removeEmitter(idx: number): void {
    if (idx >= 0 && idx < this.emitters.length) {
      this.emitters.splice(idx, 1);
    }
  }

  /** Switch the QoS profile (adjusts sim params on the fly). */
  setQoSProfile(profile: QoSProfileName): void {
    this.cfg.qosProfile = profile;
    const qos = qosSpatial[profile];
    this.params = { ...this.params, ...qos };
    this.sphGPU.updateParams(this.params);
    this.renderer.updateParams(this.params);
  }

  /** Expose the collision world's SceneQuery for external raycasts. */
  getSceneQuery(): SceneQuery {
    return (this.collisionWorld as any).sceneQuery as SceneQuery;
  }

  /** Read the latest orchestrator stats (call after each frame). */
  getStats(): OrchestratorStats {
    return {
      fps:                    this.smoothedFps,
      particleCount:          this.cpuPos.count,
      adaptiveParticleCap:    this.adaptiveParticleCap,
      sphIterations:          this.adaptiveSphIters,
      collisionCount:         this.lastCollisions.count,
      frameNumber:            this.frameNumber,
      simTime:                this.simTime,
      physicsStepsThisFrame:  this._physicsStepsThisFrame,
      timings: {
        physics: this._tPhysics,
        render:  this._tRender,
        total:   this._tTotal,
      },
    };
  }

  // ── Audio controls ─────────────────────────────────────────────────────────

  setAudioVolume(v: number): void {
    this.audio?.setVolume(Math.max(0, Math.min(1, v)));
  }

  muteAudio(): void  { this.audio?.mute(); }
  unmuteAudio(): void { this.audio?.unmute(); }

  // ── AT Pipeline controls ───────────────────────────────────────────────────

  getATRenderPipeline(): ATRenderPipeline | null { return this.atPipeline; }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Main loop
  // ═══════════════════════════════════════════════════════════════════════════

  private _loop = (timestamp: number): void => {
    if (!this.running) return;

    const tFrameStart = performance.now();

    // ── 1. Measure real delta ────────────────────────────────────────────────
    const realDt = Math.min((timestamp - this.lastRealTime) * 0.001, 0.25);
    this.lastRealTime = timestamp;

    // ── 2. FPS tracking & adaptive tuning ────────────────────────────────────
    this._updateFps(realDt);
    if (this.enableAdaptive) {
      this._adaptQuality();
    }

    // ── 3. Emitter tick (spawn particles into CPU arrays) ────────────────────
    this._tickEmitters(realDt);

    // ── 4. Fixed-timestep physics accumulator ────────────────────────────────
    this.accumulator += realDt;
    if (this.accumulator > MAX_ACCUMULATOR) {
      this.accumulator = MAX_ACCUMULATOR; // clamp to prevent death spiral
    }

    const tPhysStart = performance.now();
    let physicsSteps = 0;

    while (this.accumulator >= this.fixedDt) {
      this._physicsStep(this.fixedDt);
      this.accumulator -= this.fixedDt;
      this.simTime     += this.fixedDt;
      physicsSteps++;
    }

    this._tPhysics              = performance.now() - tPhysStart;
    this._physicsStepsThisFrame = physicsSteps;

    // ── 5–7. Render ──────────────────────────────────────────────────────────
    const tRenderStart = performance.now();
    this._renderFrame(realDt);
    this._tRender = performance.now() - tRenderStart;

    // ── 8. Audio update ──────────────────────────────────────────────────────
    // AudioPhysicsBridge.update expects (world, manifolds). We build a
    // lightweight shim from our collision export to drive the sonification.
    if (this.audio) {
      this._updateAudio();
    }

    // ── 9. Export collision data & stats ──────────────────────────────────────
    this.lastCollisions = this.collisionWorld.exportCollisions();
    this.frameNumber++;

    this._tTotal = performance.now() - tFrameStart;

    // ── Schedule next frame ──────────────────────────────────────────────────
    this.rafHandle = requestAnimationFrame(this._loop);
  };

  // ═══════════════════════════════════════════════════════════════════════════
  //  Physics sub-step
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Execute a single fixed-timestep physics sub-step. This is the heart of
   * the simulation: spatial hash → neighbours → force-field → collision →
   * GPU compute → boundary → readback.
   */
  private async _physicsStep(dt: number): Promise<void> {
    const n = this.cpuPos.count;
    if (n === 0) return;

    // ── 4a. CPU spatial hash rebuild ─────────────────────────────────────────
    this.grid.rebuild(this.cpuPos, n);

    // ── 4b. CPU neighbour list build ─────────────────────────────────────────
    const neighborLists = this.nlBuilder.build(this.cpuPos, n);
    this.sphGPU.uploadNeighborLists(neighborLists, n);

    // ── 4c. Force-field injection (CPU → GPU upload) ─────────────────────────
    if (this.cellForces.length > 0) {
      this._applyForceField(dt, n);
    }

    // ── 4d. Collision world step ─────────────────────────────────────────────
    this.collisionWorld.step(dt);

    // ── 4e. GPU compute passes ───────────────────────────────────────────────
    const enc = this.device.createCommandEncoder({ label: 'orch-physics' });

    // Adaptive iteration count: run density+force sub-iters
    for (let iter = 0; iter < this.adaptiveSphIters; iter++) {
      this.sphGPU.encodeDensityPressure(enc, n);
      this.sphGPU.encodeForces(enc, n);
    }

    this.sphGPU.encodeIntegrate(enc, n, dt);
    this.device.queue.submit([enc.finish()]);

    // ── 4g. Position read-back ───────────────────────────────────────────────
    await this._readbackPositions(n);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Render frame
  // ═══════════════════════════════════════════════════════════════════════════

  private _renderFrame(dt: number): void {
    if (!this.device || !this.running) return;

    const n = this.cpuPos.count;
    const enc = this.device.createCommandEncoder({ label: 'orch-render' });
    const swapView = this.ctx.getCurrentTexture().createView();

    if (n === 0 && !this.atPipeline && !this.envFx) {
      // Empty frame — clear to dark
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view:       swapView,
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
          loadOp:     'clear',
          storeOp:    'store',
        }],
      });
      pass.end();
      this.device.queue.submit([enc.finish()]);
      return;
    }

    // ── 5. AT Render Pipeline tick ───────────────────────────────────────────
    if (this.atPipeline) {
      this.atPipeline.tick(dt);
    }

    // ── 6. Environment FX tick ───────────────────────────────────────────────
    if (this.envFx) {
      this.envFx.tick(dt);
    }

    // ── 7a. Environment FX background pass ───────────────────────────────────
    if (this.envFx) {
      this.envFx.render(enc, swapView);
    }

    // ── 7b. SPH particle render ──────────────────────────────────────────────
    if (n > 0) {
      this.renderer.encodeRenderPass(enc, swapView, n);
    }

    // ── 7c+d. AT Render Pipeline (full chain → swap-chain) ──────────────────
    if (this.atPipeline) {
      this.atPipeline.render(enc, swapView);
    }

    this.device.queue.submit([enc.finish()]);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Emitters
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Tick all registered emitters, spawning particles into the CPU arrays.
   * Uses pattern.sample(simTime) to modulate the emission rate.
   */
  private _tickEmitters(dt: number): void {
    const cap = Math.min(this.adaptiveParticleCap, MAX_PARTICLES);

    for (const em of this.emitters) {
      const multiplier = em.pattern.sample(this.simTime);
      const toEmit     = em.rate * multiplier * dt + em._accumulator;
      const whole      = Math.floor(toEmit);
      em._accumulator  = toEmit - whole;

      for (let i = 0; i < whole; i++) {
        const n = this.cpuPos.count;
        if (n >= cap) break;

        // Spawn at emitter position with directed velocity + small jitter
        this.cpuPos.x[n]       = em.x + (Math.random() - 0.5) * 0.02;
        this.cpuPos.y[n]       = em.y + (Math.random() - 0.5) * 0.02;
        this.cpuPos.vx[n]      = em.dirX * (0.8 + Math.random() * 0.4);
        this.cpuPos.vy[n]      = em.dirY * (0.8 + Math.random() * 0.4);
        this.cpuPos.species[n] = em.species;
        this.cpuPos.count      = n + 1;
      }
    }

    // Upload if any emitters fired
    if (this.emitters.length > 0) {
      this._uploadParticles();
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Force-field
  // ═══════════════════════════════════════════════════════════════════════════

  private _applyForceField(dt: number, n: number): void {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scaleX = cw / this.domainW;
    const scaleY = ch / this.domainH;
    const { x, y, vx, vy } = this.cpuPos;

    for (let i = 0; i < n; i++) {
      const px = x[i] * scaleX;
      const py = y[i] * scaleY;

      for (const cell of this.cellForces) {
        if (
          px >= cell.minX && px <= cell.maxX &&
          py >= cell.minY && py <= cell.maxY
        ) {
          vx[i] += cell.fx * dt;
          vy[i] += cell.fy * dt;
          break; // one cell per particle per tick
        }
      }
    }

    // Re-upload velocities with injected impulses before GPU compute
    this.device.queue.writeBuffer(this.gpuBufs.velX, 0, vx, 0, n);
    this.device.queue.writeBuffer(this.gpuBufs.velY, 0, vy, 0, n);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Performance adaptive
  // ═══════════════════════════════════════════════════════════════════════════

  private _updateFps(realDt: number): void {
    // Store instantaneous FPS in the sliding window
    const instantFps = realDt > 0 ? 1 / realDt : 60;
    this.fpsWindow[this.fpsWindowIdx] = instantFps;
    this.fpsWindowIdx = (this.fpsWindowIdx + 1) % FPS_WINDOW_SIZE;
    if (this.fpsWindowIdx === 0) this.fpsWindowFilled = true;

    // Compute mean FPS from the window
    const count = this.fpsWindowFilled ? FPS_WINDOW_SIZE : this.fpsWindowIdx;
    if (count === 0) { this.smoothedFps = 60; return; }

    let sum = 0;
    for (let i = 0; i < count; i++) {
      sum += this.fpsWindow[i];
    }
    this.smoothedFps = sum / count;
  }

  private _adaptQuality(): void {
    if (this.smoothedFps < FPS_LOW_THRESHOLD) {
      // ── Degrade: reduce particle cap ───────────────────────────────────────
      this.adaptiveParticleCap = Math.max(
        MIN_ADAPTIVE_PARTICLES,
        Math.floor(this.adaptiveParticleCap * PARTICLE_REDUCE_FACTOR),
      );

      // Reduce SPH iterations
      this.adaptiveSphIters = Math.max(MIN_SPH_ITERS, this.adaptiveSphIters - 1);

      // If we have more particles than the new cap, cull the excess
      if (this.cpuPos.count > this.adaptiveParticleCap) {
        this.cpuPos.count = this.adaptiveParticleCap;
        this._uploadParticles();
      }

    } else if (this.smoothedFps > FPS_HIGH_THRESHOLD) {
      // ── Restore: gradually raise particle cap toward nominal ────────────────
      this.adaptiveParticleCap = Math.min(
        this.nominalParticleCap,
        Math.ceil(this.adaptiveParticleCap * PARTICLE_RESTORE_FACTOR),
      );

      // Restore SPH iterations
      if (this.adaptiveSphIters < NOMINAL_SPH_ITERS) {
        this.adaptiveSphIters++;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Audio
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Build a lightweight shim from our collision export and particle data
   * to feed the AudioPhysicsBridge.update(world, manifolds) interface.
   */
  private _updateAudio(): void {
    if (!this.audio) return;

    // Build a minimal "World" shim compatible with audio-physics-bridge
    const worldShim = {
      particles: [] as Array<{ position: { x: number; y: number }; velocity: { x: number; y: number }; species: string }>,
      emitters:  this.emitters.map((e) => ({
        x: e.x, y: e.y,
        dirX: e.dirX, dirY: e.dirY,
        rate: e.rate, species: String(e.species),
        _accumulator: e._accumulator,
      })),
    };

    // Only populate a sample of particles for audio analysis (perf)
    const sampleRate = Math.max(1, Math.floor(this.cpuPos.count / 200));
    for (let i = 0; i < this.cpuPos.count; i += sampleRate) {
      worldShim.particles.push({
        position: { x: this.cpuPos.x[i], y: this.cpuPos.y[i] },
        velocity: { x: this.cpuPos.vx[i], y: this.cpuPos.vy[i] },
        species:  String(this.cpuPos.species[i]),
      });
    }

    // Build minimal manifolds from collision export
    const manifolds = this.lastCollisions.collisions.map((c) => ({
      bodyA: {
        id: c.bodyA, x: 0, y: 0, angle: 0,
        vertices: new Float64Array(0), friction: 0.3, restitution: 0.5,
      },
      bodyB: {
        id: c.bodyB, x: 0, y: 0, angle: 0,
        vertices: new Float64Array(0), friction: 0.3, restitution: 0.5,
      },
      points: [{
        x: 0, y: 0,
        nx: c.normal.x, ny: c.normal.y,
        depth: c.depth,
        normalImpulse: 0, tangentImpulse: 0,
        featureId: 0,
      }],
      friction: 0.3,
      restitution: 0.5,
    }));

    try {
      this.audio.update(worldShim as any, manifolds);
    } catch {
      // Audio errors should never crash the simulation
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  GPU helpers
  // ═══════════════════════════════════════════════════════════════════════════

  /** Upload CPU particle arrays to GPU. */
  private _uploadParticles(): void {
    const { x, y, vx, vy, species, count } = this.cpuPos;
    const dev = this.device;
    const sub = (buf: GPUBuffer, data: ArrayBufferView, n: number) =>
      dev.queue.writeBuffer(buf, 0, data, 0, n);

    sub(this.gpuBufs.posX,    x,       count);
    sub(this.gpuBufs.posY,    y,       count);
    sub(this.gpuBufs.velX,    vx,      count);
    sub(this.gpuBufs.velY,    vy,      count);
    sub(this.gpuBufs.species, species, count);

    const countBuf = new Uint32Array([count]);
    dev.queue.writeBuffer(this.gpuBufs.count, 0, countBuf);
  }

  /** Async read-back of GPU position buffers to CPU. */
  private async _readbackPositions(n: number): Promise<void> {
    const byteLen = n * 4;
    const dev     = this.device;

    const stagingX = dev.createBuffer({
      size:  byteLen,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stagingY = dev.createBuffer({
      size:  byteLen,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const enc = dev.createCommandEncoder({ label: 'orch-readback' });
    enc.copyBufferToBuffer(this.gpuBufs.posX, 0, stagingX, 0, byteLen);
    enc.copyBufferToBuffer(this.gpuBufs.posY, 0, stagingY, 0, byteLen);
    dev.queue.submit([enc.finish()]);

    await Promise.all([
      stagingX.mapAsync(GPUMapMode.READ),
      stagingY.mapAsync(GPUMapMode.READ),
    ]);

    this.cpuPos.x.set(new Float32Array(stagingX.getMappedRange(0, byteLen)));
    this.cpuPos.y.set(new Float32Array(stagingY.getMappedRange(0, byteLen)));

    stagingX.unmap();
    stagingY.unmap();
    stagingX.destroy();
    stagingY.destroy();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Init-time loaders
  // ═══════════════════════════════════════════════════════════════════════════

  /** Sync cell_registry.json AABBs → static collision bodies. */
  private async _syncCellRegistryBodies(): Promise<void> {
    try {
      const resp = await fetch('/channels/physics/cell_registry.json');
      if (!resp.ok) {
        console.warn('[Orchestrator] cell_registry.json fetch:', resp.status);
        return;
      }
      const registry = await resp.json() as {
        cells: Record<string, {
          bbox: { min: [number, number, number]; max: [number, number, number] };
        }>;
      };
      for (const [, cell] of Object.entries(registry.cells)) {
        const [minX, minY] = cell.bbox.min;
        const [maxX, maxY] = cell.bbox.max;
        const cx    = (minX + maxX) / 2;
        const cy    = (minY + maxY) / 2;
        const halfW = (maxX - minX) / 2;
        const halfH = (maxY - minY) / 2;
        const { body, shape } = createBoxBody(cx, cy, halfW, halfH, 'static');
        this.collisionWorld.addBody(body, shape);
      }
    } catch (err) {
      console.warn('[Orchestrator] cell_registry fetch failed:', err);
    }
  }

  /** Fetch and resolve force_field.json → cellForces lookup table. */
  private async _loadForceField(): Promise<void> {
    const ffPath = '/channels/physics/force_field.json';
    const crPath = '/channels/physics/cell_registry.json';

    const [ffResp, crResp] = await Promise.all([fetch(ffPath), fetch(crPath)]);
    if (!ffResp.ok) throw new Error(`fetch ${ffPath}: ${ffResp.status}`);
    if (!crResp.ok) throw new Error(`fetch ${crPath}: ${crResp.status}`);

    const forceField:   Record<string, ForceVector>          = await ffResp.json();
    const cellRegistry: { cells: Record<string, CellEntry> } = await crResp.json();

    const resolved: CellForce[] = [];
    for (const [cellId, force] of Object.entries(forceField)) {
      const cell = cellRegistry.cells[cellId];
      if (!cell) continue;
      const { min, max } = cell.bbox;
      resolved.push({
        minX: min[0], minY: min[1],
        maxX: max[0], maxY: max[1],
        fx:   force.dx, fy: force.dy,
      });
    }

    this.cellForces = resolved;
    console.info(`[Orchestrator] force_field loaded: ${resolved.length} cells`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Resize handling
  // ═══════════════════════════════════════════════════════════════════════════

  /** Handle canvas resize — propagates to all subsystems that track dimensions. */
  resize(width: number, height: number): void {
    this.canvas.width  = width;
    this.canvas.height = height;
    this.domainW       = 3.0 * (width / height);

    // Re-configure canvas context
    this.ctx.configure({
      device:    this.device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    // Update sim params with new domain dimensions
    this.params = makeSimParams(this.domainW, this.domainH, this.cfg.qosProfile);
    this.sphGPU.updateParams(this.params);
    this.renderer.updateParams(this.params);

    // Propagate to subsystems
    this.atPipeline?.resize(width, height);
    this.envFx?.resize(width, height);
  }
}
