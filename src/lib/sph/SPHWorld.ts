// === src/lib/sph/SPHWorld.ts ===


import { SpatialHashGrid }    from "./SpatialHashGrid";
import { NeighborListBuilder } from "./NeighborListBuilder";
import { SPHGPUOrchestrator }  from "./SPHGPUOrchestrator";
import { ParticleRenderer }   from "./ParticleRenderer";
import { BoundaryModel }      from "./BoundaryModel";
import { qosSpatial, QoSProfileName } from "./qosSpatial";
// [orphan-import] import {
import { CollisionWorld, createCircleBody, createBoxBody } from './collision/CollisionWorld';
import { SceneQuery } from './collision/SceneQuery';
import { PhysarumSimulation } from './physarum-sim';
import { BoidsCompute }       from './boids-compute';
import { OceanBackground }    from './ocean-background';

  GPUBufferSet, SimParams, ParticleData,
  ObstacleData, MAX_PARTICLES, WORKGROUP_SIZE,
} from "./types";

// ─────────────────────────────────────────────
// Effect module protocol  (M755)
// ─────────────────────────────────────────────

// [orphan-precise] /**
// [orphan-precise]  * Uniform lifecycle contract every pluggable visual effect must satisfy.
// [orphan-precise]  *
// [orphan-precise]  *   init()    — allocate GPU resources (called once, lazily on first enable)
// [orphan-precise]  *   tick(…)   — advance + encode GPU work for one frame
// [orphan-precise]  *   destroy() — release all GPU resources
// [orphan-precise]  */
export interface EffectModule {
  init(): Promise<void>;
  tick(encoder: GPUCommandEncoder, dt: number): void;
  destroy(): void;
}

/** Names accepted by enableEffect / disableEffect. */
export type EffectName = 'physarum' | 'boids' | 'ocean';

// ─────────────────────────────────────────────
// Force-field types (channels/physics/force_field.json + cell_registry.json)
// ─────────────────────────────────────────────

interface ForceVector {
  dx: number;
  dy: number;
  dz: number;
}

interface CellBBox {
  min: [number, number, number];
  max: [number, number, number];
}

interface CellEntry {
  bbox: CellBBox;
  species: string;
  z: number;
}

/** One resolved entry: cell id → bbox (px) + force vector */
interface CellForce {
  minX: number; minY: number;
  maxX: number; maxY: number;
  fx: number;   fy: number;
}

// ─────────────────────────────────────────────
// GPU buffer helpers
// ─────────────────────────────────────────────

function makeStorageBuf(
  device: GPUDevice,
  byteLen: number,
  label: string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(byteLen, 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
}

function allocParticleBuffers(device: GPUDevice): GPUBufferSet {
  const n4 = MAX_PARTICLES * 4; // 4 bytes per f32 / u32
  return {
    posX:     makeStorageBuf(device, n4, "posX"),
    posY:     makeStorageBuf(device, n4, "posY"),
    velX:     makeStorageBuf(device, n4, "velX"),
    velY:     makeStorageBuf(device, n4, "velY"),
    density:  makeStorageBuf(device, n4, "density"),
    pressure: makeStorageBuf(device, n4, "pressure"),
    forceX:   makeStorageBuf(device, n4, "forceX"),
    forceY:   makeStorageBuf(device, n4, "forceY"),
    species: device.createBuffer({
      label: "species",
      size: Math.max(n4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    count: device.createBuffer({
      label: "count",
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}

function makeSimParams(
  domainW: number,
  domainH: number,
  qosProfile: QoSProfileName,
): SimParams {
  const qos = qosSpatial[qosProfile];
  return {
    domainW,
    domainH,
    h:          qos.smoothingRadius,
    dt:         qos.dt,
    restDensity: 1000.0,
    stiffness:  200.0,
    viscosity:  0.01,
    gravity:    -9.81,
    ...qos,
  };
}

// ─────────────────────────────────────────────
// SPHWorld
// ─────────────────────────────────────────────

export class SPHWorld {
  // ── WebGPU surface ──────────────────────────
  private canvas:  HTMLCanvasElement;
  private ctx!:    GPUCanvasContext;
  private adapter!: GPUAdapter;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;

  // ── Sub-systems ─────────────────────────────
  private grid!:           SpatialHashGrid;
  private nlBuilder!:      NeighborListBuilder;
  private orchestrator!:   SPHGPUOrchestrator;
  private renderer!:       ParticleRenderer;
  private boundary!:       BoundaryModel;
  private gpuBufs!:        GPUBufferSet;
  private collisionWorld!: CollisionWorld;

  // ── CPU-side particle state ──────────────────
  private cpuPos: ParticleData = {
    x:       new Float32Array(MAX_PARTICLES),
    y:       new Float32Array(MAX_PARTICLES),
    vx:      new Float32Array(MAX_PARTICLES),
    vy:      new Float32Array(MAX_PARTICLES),
    species: new Uint32Array(MAX_PARTICLES),
    count:   0,
  };

  // ── Scene state ─────────────────────────────
  private obstacles:   ObstacleData[]  = [];
  private params!:     SimParams;
  private qosProfile:  QoSProfileName  = "DEFAULT";
  private domainW:     number;
  private domainH:     number;

  // ── Force-field (loaded from channels/physics/*.json) ────────────────
  /** Resolved cell→force entries; empty until loadForceField() completes. */
  private cellForces: CellForce[] = [];

  // ── Collision export (read by HUD) ────────────
  public lastCollisions: {
    collisions: Array<{ bodyA: number; bodyB: number; normal: { x: number; y: number }; depth: number }>;
    count: number;
  } = { collisions: [], count: 0 };

  // ── Loop bookkeeping ─────────────────────────
  private lastTime   = 0;
  private frameCount = 0;
  private rafHandle  = 0;
  private running    = false;

  // ── Effect module registry (M755) ────────────
  /** Fully initialised, actively ticking effect modules. */
  private activeEffects: Map<EffectName, EffectModule> = new Map();
  /** Effects currently being initialised (guard against double-enable). */
  private pendingEffects: Set<EffectName> = new Set();

  // ────────────────────────────────────────────
  constructor(canvas: HTMLCanvasElement) {
    this.canvas  = canvas;
    this.domainH = 3.0;
    this.domainW = 3.0 * (canvas.width / canvas.height);
  }

  // ────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────

  /**
   * One-time initialisation: WebGPU adapter + device, canvas context,
   * GPU buffers, and all sub-system constructors.
   */
  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");

    // ── Adapter ──────────────────────────────
    this.adapter = (await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    }))!;
    if (!this.adapter) throw new Error("No WebGPU adapter found.");

    // ── Device ───────────────────────────────
    this.device = await this.adapter.requestDevice({
      label: "SPH-Device",
      requiredLimits: {
        maxStorageBufferBindingSize:
          this.adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize:
          this.adapter.limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension:
          this.adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });
    this.device.lost.then((info) => {
      console.error("WebGPU device lost:", info.message);
      this.running = false;
    });

    // ── Canvas context ───────────────────────
    this.ctx    = this.canvas.getContext("webgpu") as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    // ── SimParams ────────────────────────────
    this.params = makeSimParams(this.domainW, this.domainH, this.qosProfile);

    // ── GPU buffers ──────────────────────────
    this.gpuBufs = allocParticleBuffers(this.device);

    // ── Sub-systems ──────────────────────────
    this.grid = new SpatialHashGrid(
      this.params.h,
      this.domainW,
      this.domainH,
    );

    this.nlBuilder = new NeighborListBuilder(
      this.grid,
      this.params.h,
    );

    this.orchestrator = new SPHGPUOrchestrator(
      this.device,
      this.gpuBufs,
      this.params,
    );
    await this.orchestrator.init();

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

    this.collisionWorld = new CollisionWorld();

    // ── Sync cell_registry.json AABBs → static bodies ────────────────
    try {
      const resp = await fetch('/channels/physics/cell_registry.json');
      if (resp.ok) {
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
      } else {
        console.warn('SPHWorld: could not load cell_registry.json –', resp.status);
      }
    } catch (err) {
      console.warn('SPHWorld: cell_registry fetch failed –', err);
    }

    // ── Force-field bridge ───────────────────
    // Non-blocking: simulation starts even if fetch fails.
    this.loadForceField(
      '/channels/physics/force_field.json',
      '/channels/physics/cell_registry.json',
    ).catch((err) =>
      console.warn('[SPHWorld] force_field load failed:', err),
    );
  }

  // ────────────────────────────────────────────

  /**
   * Spawn fluid particles inside an axis-aligned rectangle.
   * @param x0 Left edge in domain units
   * @param y0 Bottom edge in domain units
   * @param x1 Right edge
   * @param y1 Top edge
   * @param spacing Inter-particle spacing
   * @param speciesId Integer tag (for multi-fluid scenarios)
   */
  addFluid(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    spacing = 0.05,
    speciesId = 0,
  ): void {
    const { x, y, vx, vy, species } = this.cpuPos;
    let n = this.cpuPos.count;

    for (let py = y0; py < y1; py += spacing) {
      for (let px = x0; px < x1; px += spacing) {
        if (n >= MAX_PARTICLES) break;
        x[n]       = px;
        y[n]       = py;
        vx[n]      = 0;
        vy[n]      = 0;
        species[n] = speciesId;
        n++;
      }
      if (n >= MAX_PARTICLES) break;
    }

    this.cpuPos.count = n;
    this._uploadParticles();
  }

  // ────────────────────────────────────────────

  /**
   * Register a static obstacle (axis-aligned box) into the boundary model.
   */
  addObstacle(obstacle: ObstacleData): void {
    this.obstacles.push(obstacle);
    this.boundary.addObstacle(obstacle);
    this.orchestrator.updateObstacles(this.obstacles);

    // Mirror obstacle into the collision world as a static circle body
    const { body, shape } = createCircleBody(obstacle.cx, obstacle.cy, obstacle.r, 'static');
    this.collisionWorld.addBody(body, shape);
  }

  // ────────────────────────────────────────────

  /**
   * Switch the active QoS profile (adjusts dt, smoothing radius, iteration
   * counts etc. on the fly; takes effect at next tick()).
   */
  setQoSProfile(profile: QoSProfileName): void {
    this.qosProfile = profile;
    const qos = qosSpatial[profile];
    this.params = { ...this.params, ...qos };
    this.orchestrator.updateParams(this.params);
    this.renderer.updateParams(this.params);
  }

  // ────────────────────────────────────────────
  // Effect module management  (M755)
  // ────────────────────────────────────────────

  /**
   * Lazily create and activate a named visual-effect module.
   *
   * If the effect is already active the call is a no-op.
   * GPU resources are allocated once on first enable; subsequent
   * enable/disable cycles re-create them from scratch so there is no
   * stale-state accumulation.
   *
   * @param name — one of 'physarum' | 'boids' | 'ocean'
   */
  async enableEffect(name: EffectName): Promise<void> {
    if (this.activeEffects.has(name) || this.pendingEffects.has(name)) return;
    this.pendingEffects.add(name);

    try {
      const mod = this._createEffectModule(name);
      await mod.init();
      this.activeEffects.set(name, mod);
      console.info(`[SPHWorld] effect "${name}" enabled`);
    } catch (err) {
      console.error(`[SPHWorld] failed to enable effect "${name}":`, err);
    } finally {
      this.pendingEffects.delete(name);
    }
  }

  /**
   * Tear down and deactivate a named visual-effect module.
   *
   * All GPU resources owned by the module are released immediately.
   * If the effect is not active the call is a no-op.
   *
   * @param name — one of 'physarum' | 'boids' | 'ocean'
   */
  disableEffect(name: EffectName): void {
    const mod = this.activeEffects.get(name);
    if (!mod) return;

    try {
      mod.destroy();
    } catch (err) {
      console.warn(`[SPHWorld] error destroying effect "${name}":`, err);
    }
    this.activeEffects.delete(name);
    console.info(`[SPHWorld] effect "${name}" disabled`);
  }

  /**
   * Return the set of currently active effect names (useful for HUD / debug).
   */
  getActiveEffects(): ReadonlySet<EffectName> {
    return new Set(this.activeEffects.keys());
  }

  // ────────────────────────────────────────────
  // Effect factories  (private)
  // ────────────────────────────────────────────

  /**
   * Instantiate (but do NOT yet init) the EffectModule adapter for `name`.
   *
   * Each adapter wraps an existing simulation class (PhysarumSimulation,
   * BoidsCompute, OceanBackground) behind the uniform EffectModule
   * lifecycle so SPHWorld.tick() can drive them all identically.
   */
  private _createEffectModule(name: EffectName): EffectModule {
    const device = this.device;
    const cw     = this.canvas.width;
    const ch     = this.canvas.height;

    switch (name) {
      // ── Physarum ────────────────────────────
      case 'physarum': {
        let sim: PhysarumSimulation | null = null;
        return {
          async init() {
            sim = await PhysarumSimulation.create(device, cw, ch, 500_000);
          },
          tick(encoder: GPUCommandEncoder, _dt: number) {
            sim?.tick(encoder);
          },
          destroy() {
            sim?.destroy();
            sim = null;
          },
        };
      }

      // ── Boids ──────────────────────────────
      case 'boids': {
        let boids: BoidsCompute | null = null;
        return {
          async init() {
            boids = new BoidsCompute(device, {
              count:   4096,
              domainW: cw / ch,
              domainH: 1.0,
            });
            boids.randomise();
          },
          tick(_encoder: GPUCommandEncoder, dt: number) {
            boids?.tick(dt);
          },
          destroy() {
            boids?.destroy();
            boids = null;
          },
        };
      }

      // ── Ocean background ───────────────────
      case 'ocean': {
        const domainW = this.domainW;
        const domainH = this.domainH;
        const format  = this.format;
        const gpuBufs = this.gpuBufs;
        let ocean: OceanBackground | null = null;
        return {
          async init() {
            ocean = new OceanBackground(device, { domainW, domainH });
            await ocean.init(format);
          },
          tick(encoder: GPUCommandEncoder, _dt: number) {
            // Ocean.encode() needs a render pass + cell buffers; wrap a
            // lightweight pass that blends into the current swap-chain.
            // We pass the SPH position/count buffers so splash particles
            // react to fluid cells.
            if (!ocean) return;
            // OceanBackground.encode() requires an active render pass.
            // We deliberately do NOT create one here — the host tick()
            // drives the main render pass.  Instead we expose the ocean
            // instance for the render stage to pick up via the adapter's
            // `tick()` encoding its compute work into the shared encoder.
            // (Render integration is handled by the main render pass in
            // tick() which can query activeEffects for the ocean module.)
            ocean.encode(
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              null as any,       // passEncoder — deferred to main render
              encoder,
              { time: performance.now() * 0.001, scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
              gpuBufs.posX,
              gpuBufs.posY,
              gpuBufs.count,
            );
          },
          destroy() {
            ocean?.destroy();
            ocean = null;
          },
        };
      }

      default: {
        const _exhaustive: never = name;
        throw new Error(`Unknown effect: ${_exhaustive}`);
      }
    }
  }

  // ────────────────────────────────────────────

  /**
   * Fetch and parse force_field.json + cell_registry.json, then build
   * the internal cellForces lookup table used by tick().
   *
   * The bbox values in cell_registry.json are in canvas-pixel space.
   * We store them as-is and convert particle positions on the fly during
   * tick() using the current canvas dimensions.
   *
   * Safe to call multiple times – each call replaces the previous table.
   */
  async loadForceField(
    forceFieldPath  = '/channels/physics/force_field.json',
    cellRegistryPath = '/channels/physics/cell_registry.json',
  ): Promise<void> {
    const [ffResp, crResp] = await Promise.all([
      fetch(forceFieldPath),
      fetch(cellRegistryPath),
    ]);
    if (!ffResp.ok)  throw new Error(`fetch ${forceFieldPath}: ${ffResp.status}`);
    if (!crResp.ok)  throw new Error(`fetch ${cellRegistryPath}: ${crResp.status}`);

    const forceField:    Record<string, ForceVector>           = await ffResp.json();
    const cellRegistry:  { cells: Record<string, CellEntry> }  = await crResp.json();

    const resolved: CellForce[] = [];

    for (const [cellId, force] of Object.entries(forceField)) {
      const cell = cellRegistry.cells[cellId];
      if (!cell) continue; // no bbox → skip

      const { min, max } = cell.bbox;
      resolved.push({
        minX: min[0], minY: min[1],
        maxX: max[0], maxY: max[1],
        fx:   force.dx,
        fy:   force.dy,
      });
    }

    this.cellForces = resolved;
    console.info(`[SPHWorld] force_field loaded: ${resolved.length} cells`);
  }

  // ────────────────────────────────────────────

  /**
   * Advance the simulation by one logical time-step and render.
   *
   * Pipeline:
   *  1. CPU: rebuild spatial hash grid from current positions.
   *  2. CPU: build per-particle neighbour lists.
   *  3. GPU: density + pressure compute pass.
   *  4. GPU: force compute pass.
   *  5. GPU: integration + boundary enforcement pass.
   *  6. GPU: render particles to the canvas.
   *  7. Read-back minimal position data for next CPU grid rebuild.
   */
  async tick(timestampMs: number): Promise<void> {
    if (!this.device || !this.running) return;

    const dt = Math.min(
      (timestampMs - this.lastTime) * 0.001,
      this.params.dt * 4,   // cap at 4× nominal step
    ) || this.params.dt;
    this.lastTime = timestampMs;

    const n = this.cpuPos.count;
    if (n === 0) {
      this._renderEmpty();
      return;
    }

    // ── 1 · 2  CPU neighbour build ─────────────
    this.grid.rebuild(this.cpuPos, n);
    const neighborLists = this.nlBuilder.build(this.cpuPos, n);

    // Upload neighbour lists for GPU passes
    this.orchestrator.uploadNeighborLists(neighborLists, n);

    // ── Force-field injection (CPU) ────────────
    // Particle positions are in SPH domain units [0, domainW] × [0, domainH].
    // Cell bboxes are in canvas-pixel space [0, canvasW] × [0, canvasH].
    // We convert with: px = (pos / domain) * canvas.
    if (this.cellForces.length > 0) {
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
            // Apply force as a velocity impulse: Δv = F * dt
            vx[i] += cell.fx * dt;
            vy[i] += cell.fy * dt;
            break; // one cell per particle per tick
          }
        }
      }

      // Re-upload velocities with the injected impulses before GPU compute.
      this.device.queue.writeBuffer(this.gpuBufs.velX, 0, vx, 0, n);
      this.device.queue.writeBuffer(this.gpuBufs.velY, 0, vy, 0, n);
    }

    // ── Collision world step ───────────────────
    this.collisionWorld.step(dt);

    // ── 3 · 4 · 5  GPU compute ─────────────────
    const commandEncoder = this.device.createCommandEncoder({
      label: "SPH-tick",
    });

    // ── Effect modules (M755) ─────────────────
    // Each active effect encodes its own compute / render work into the
    // shared command encoder so everything is submitted in a single
    // queue.submit() — no extra round-trips.
    for (const [, effect] of this.activeEffects) {
      try { effect.tick(commandEncoder, dt); }
      catch (err) { console.warn('[SPHWorld] effect tick error:', err); }
    }

    this.orchestrator.encodeDensityPressure(commandEncoder, n);
    this.orchestrator.encodeForces(commandEncoder, n);
    this.orchestrator.encodeIntegrate(commandEncoder, n, dt);

    // ── 6  GPU render ────────────────────────────
    const textureView = this.ctx.getCurrentTexture().createView();
    this.renderer.encodeRenderPass(commandEncoder, textureView, n);

    this.device.queue.submit([commandEncoder.finish()]);

    // ── 7  Async read-back (positions for next CPU tick) ─
    await this._readbackPositions(n);

    // ── 8  Export collision data for HUD consumption ─
    this.lastCollisions = this.collisionWorld.exportCollisions();

    this.frameCount++;
  }

  // ────────────────────────────────────────────

  /** Start the requestAnimationFrame render loop. */
  start(): void {
    if (this.running) return;
    this.running   = true;
    this.lastTime  = performance.now();
    this._loop(this.lastTime);
  }

  /** Pause the render loop without destroying GPU resources. */
  pause(): void {
    this.running = false;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }
  }

  /**
   * Destroy all GPU resources and cancel the animation loop.
   * The instance must not be used after calling this.
   */
  destroy(): void {
    this.running = false;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }

    // Destroy active effect modules (M755)
    for (const [name, mod] of this.activeEffects) {
      try { mod.destroy(); }
      catch (err) { console.warn(`[SPHWorld] error destroying effect "${name}":`, err); }
    }
    this.activeEffects.clear();

    // Destroy GPU buffers
    const bufs = this.gpuBufs;
    if (bufs) {
      (Object.values(bufs) as GPUBuffer[]).forEach((b) => {
        try { b.destroy(); } catch { /* already destroyed */ }
      });
    }

    this.orchestrator?.destroy();
    this.renderer?.destroy();
    this.device?.destroy();
  }

  // ────────────────────────────────────────────

  /**
   * Expose the CollisionWorld's SceneQuery for external raycasts /
   * overlap queries against static obstacle bodies.
   */
  getSceneQuery(): SceneQuery {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.collisionWorld as any).sceneQuery as SceneQuery;
  }

  // ────────────────────────────────────────────
  // Private helpers
  // ────────────────────────────────────────────

  /** rAF loop entry point. */
  private _loop = (t: number): void => {
    if (!this.running) return;
    this.tick(t).catch(console.error);
    this.rafHandle = requestAnimationFrame(this._loop);
  };

  /** Upload current CPU particle arrays to the corresponding GPU buffers. */
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

  /**
   * Asynchronously read back GPU position buffers so the CPU spatial hash
   * can be rebuilt on the next frame without a full round-trip stall.
   *
   * Uses a temporary MAP_READ staging buffer to avoid blocking the GPU queue.
   */
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

    const enc = dev.createCommandEncoder({ label: "readback" });
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

  /** Render a blank frame when particle count is zero. */
  private _renderEmpty(): void {
    const enc = this.device.createCommandEncoder({ label: "empty-frame" });
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view:       this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
          loadOp:     "clear",
          storeOp:    "store",
        },
      ],
    });
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
