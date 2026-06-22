/**
 * ocean-bridge.ts — WebGPU-Ocean ↔ cell-pubsub-loop bridge  (M603)
 *
 * Wraps the upstream WebGPU-Ocean simulators (SPHSimulator + MLSMPMSimulator)
 * and their shared FluidRenderer into a lifecycle object that fits the
 * cell-pubsub-loop canvas stack.
 *
 * ─── Upstream (matsuoka-601/WebGPU-Ocean) ─────────────────────────────────
 * upstream/webgpu-ocean/sph/sph.ts          SPHSimulator  (GPU SPH, WebGPU)
 * upstream/webgpu-ocean/mls-mpm/mls-mpm.ts  MLSMPMSimulator (MLS-MPM, WebGPU)
 * upstream/webgpu-ocean/render/fluidRender.ts FluidRenderer (bilateral-filter
 *                                             screen-space fluid)
 * upstream/webgpu-ocean/camera.ts            Camera (arcball, projection)
 * upstream/webgpu-ocean/common.ts            renderUniformsViews shared buffer
 *
 * ─── Integration contract ─────────────────────────────────────────────────
 * The bridge owns exactly one canvas layer that renders the 3-D fluid.  The
 * canvas must be inserted BELOW the PixiJS cell canvas so the fluid appears
 * as a backdrop.  The host component (Astro page / Svelte island) is
 * responsible for sizing.
 *
 * Lifecycle:
 *
 *   const ob = new OceanBridge(canvas, opts);
 *   await ob.init();          // allocate GPU resources; idempotent
 *   ob.start();               // begin rAF loop
 *   ob.setSimMode('sph');     // switch simulator at runtime
 *   ob.setParticleCount(1);   // preset 0-3 (10 k / 20 k / 30 k / 40 k SPH)
 *   ob.setBoxRatio(0.8);      // squeeze box [0, 1] → [0.5, 1.0] normalised
 *   ob.stop();                // cancel rAF; GPU objects stay alive
 *   ob.destroy();             // release all GPU buffers / pipelines
 *
 * ─── Cell-pubsub integration ─────────────────────────────────────────────
 * The bridge publishes frame telemetry on the "ocean:frame" topic every
 * N_PUBLISH_INTERVAL frames so downstream subscribers (e.g. UIL debug overlay,
 * epoch-ticker) can react to fluid state without coupling directly to WebGPU.
 *
 * Published payload (OceanFrameEvent):
 *   { simMode, numParticles, frameMs, boxRatio }
 *
 * ─── Coordinate conventions ──────────────────────────────────────────────
 * WebGPU-Ocean uses a right-hand Y-up coordinate system.  The SPH domain is
 * centred at the origin; box half-extents are stored in the SPHSimulator's
 * realBoxSizeBuffer.  The MLSMPMSimulator uses a grid-based domain (default
 * 64³ cells) normalised to ±boxHalfExtents.
 *
 * ─── References ──────────────────────────────────────────────────────────
 * upstream/webgpu-ocean/main.ts              canonical integration example
 * src/lib/sph/ocean-background.ts            Gerstner-wave 2-D ocean (alt)
 * src/lib/sph/sph-epoch-bridge.ts            SSE → SPH rigid-body bridge
 * src/lib/renderers/fluid-fbo.ts             FBO-based fluid compositor
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Which upstream simulator drives the frame loop. */
export type SimMode = 'sph' | 'mlsmpm';

/** Particle-count preset index (0 = smallest, 3 = largest). */
export type PresetIndex = 0 | 1 | 2 | 3;

/** Options accepted by the OceanBridge constructor. */
export interface OceanBridgeOptions {
  /**
   * Initial simulation mode.  Defaults to 'sph'.
   * Can be changed at runtime via setSimMode().
   */
  simMode?: SimMode;

  /**
   * Initial particle-count preset (0–3).  Defaults to 1.
   * SPH presets:    10 k / 20 k / 30 k / 40 k particles
   * MLSMPM presets: 40 k / 70 k / 120 k / 200 k particles
   */
  preset?: PresetIndex;

  /**
   * Device pixel ratio override.  Defaults to the upstream value of 0.7 to
   * keep GPU memory usage low on HiDPI screens.
   */
  devicePixelRatio?: number;

  /**
   * How often (in animation frames) telemetry is published on the
   * "ocean:frame" topic.  Set to 0 to disable publishing.  Defaults to 10.
   */
  publishInterval?: number;

  /**
   * Optional EventTarget on which OceanFrameEvent is dispatched.
   * Defaults to window (if available).
   */
  eventTarget?: EventTarget;

  /**
   * Paths to the 6 cubemap face images, in [+X, −X, +Y, −Y, +Z, −Z] order.
   * Defaults to the upstream paths used in WebGPU-Ocean.
   */
  cubemapPaths?: [string, string, string, string, string, string];
}

/** Frame telemetry emitted via CustomEvent("ocean:frame"). */
export interface OceanFrameEvent {
  simMode: SimMode;
  numParticles: number;
  /** Wall-clock duration of the last JS frame in milliseconds. */
  frameMs: number;
  /** Current box-width ratio in [0.5, 1.0]. */
  boxRatio: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Upstream re-exports (lazy dynamic imports to avoid bundling WebGPU shaders
// at parse time — they are large WGSL strings).  The actual module paths point
// at the git-subtree under upstream/webgpu-ocean.
// ─────────────────────────────────────────────────────────────────────────────

type SPHSimulatorCtor   = typeof import('../../../../upstream/webgpu-ocean/sph/sph').SPHSimulator;
type MLSMPMSimulatorCtor = typeof import('../../../../upstream/webgpu-ocean/mls-mpm/mls-mpm').MLSMPMSimulator;
type FluidRendererCtor  = typeof import('../../../../upstream/webgpu-ocean/render/fluidRender').FluidRenderer;
type CameraCtor         = typeof import('../../../../upstream/webgpu-ocean/camera').Camera;

/** Lazily-loaded upstream module set (populated during init()). */
interface UpstreamModules {
  SPHSimulator:    SPHSimulatorCtor;
  MLSMPMSimulator: MLSMPMSimulatorCtor;
  FluidRenderer:   FluidRendererCtor;
  Camera:          CameraCtor;
  renderUniformsViews: typeof import('../../../../upstream/webgpu-ocean/common').renderUniformsViews;
  renderUniformsValues: ArrayBuffer;
  numParticlesMax: number;
  sphParticleStructSize: number;
  mlsmpmParticleStructSize: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset tables (mirrors main.ts from WebGPU-Ocean)
// ─────────────────────────────────────────────────────────────────────────────

const MLSMPM_NUM_PARTICLES: [number, number, number, number] = [40000, 70000, 120000, 200000];
const MLSMPM_BOX_SIZES:     [[number,number,number],[number,number,number],[number,number,number],[number,number,number]] = [
  [35, 25, 55], [40, 30, 60], [45, 40, 80], [50, 50, 80],
];
const MLSMPM_DISTANCES:  [number, number, number, number] = [60, 70, 90, 100];

const SPH_NUM_PARTICLES: [number, number, number, number] = [10000, 20000, 30000, 40000];
const SPH_BOX_SIZES:    [[number,number,number],[number,number,number],[number,number,number],[number,number,number]] = [
  [0.7, 2.0, 0.7], [1.0, 2.0, 1.0], [1.2, 2.0, 1.2], [1.4, 2.0, 1.4],
];
const SPH_DISTANCES: [number, number, number, number] = [2.6, 3.0, 3.4, 3.8];

const MLSMPM_FOV   = 45 * Math.PI / 180;
const MLSMPM_RADIUS = 0.6;
const MLSMPM_ZOOM   = 1.5;

const SPH_FOV    = 45 * Math.PI / 180;
const SPH_RADIUS = 0.04;
const SPH_ZOOM   = 0.05;

const DEFAULT_CUBEMAP_PATHS: OceanBridgeOptions['cubemapPaths'] = [
  'cubemap/posx.png', 'cubemap/negx.png',
  'cubemap/posy.png', 'cubemap/negy.png',
  'cubemap/posz.png', 'cubemap/negz.png',
];

// ─────────────────────────────────────────────────────────────────────────────
// OceanBridge
// ─────────────────────────────────────────────────────────────────────────────

/**
 * OceanBridge — thin adapter between the upstream WebGPU-Ocean rendering stack
 * and the astro-svgfigure cell-pubsub-loop architecture.
 *
 * Responsibilities:
 *  1. WebGPU device / adapter acquisition
 *  2. Cubemap texture loading
 *  3. Shared particle / posvel / renderUniform buffer allocation
 *  4. Instantiation of both SPHSimulator and MLSMPMSimulator + two
 *     FluidRenderer instances (one per simulator, to preserve per-renderer
 *     filter constants)
 *  5. rAF loop with box-squeeze animation, mode switching, and preset reset
 *  6. Frame-telemetry publishing via CustomEvent on configurable EventTarget
 */
export class OceanBridge {
  // ── host canvas ────────────────────────────────────────────────────────────
  private readonly canvas: HTMLCanvasElement;
  private readonly opts:   Required<OceanBridgeOptions>;

  // ── state ──────────────────────────────────────────────────────────────────
  private simMode:        SimMode     = 'sph';
  private preset:         PresetIndex = 1;
  private boxRatio:       number      = 1.0;        // current [0.5, 1.0]
  private targetRatio:    number      = 1.0;        // from setBoxRatio()
  private sphereMode:     boolean     = false;      // billboard-sphere view
  private running:        boolean     = false;
  private rafHandle:      number      = 0;
  private initDone:       boolean     = false;

  // ── GPU objects ────────────────────────────────────────────────────────────
  private device!:        GPUDevice;
  private context!:       GPUCanvasContext;
  private presentationFormat!: GPUTextureFormat;

  private particleBuffer!:     GPUBuffer;
  private posvelBuffer!:       GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;

  // ── upstream instances ────────────────────────────────────────────────────
  private mod!:            UpstreamModules;
  private sphSim!:         InstanceType<SPHSimulatorCtor>;
  private mlsmpmSim!:      InstanceType<MLSMPMSimulatorCtor>;
  private sphRenderer!:    InstanceType<FluidRendererCtor>;
  private mlsmpmRenderer!: InstanceType<FluidRendererCtor>;
  private camera!:         InstanceType<CameraCtor>;

  // ── live box sizes (mutated per frame) ────────────────────────────────────
  private initBoxSize:    [number, number, number] = [1.0, 2.0, 1.0];
  private realBoxSize:    [number, number, number] = [1.0, 2.0, 1.0];

  // ── telemetry ─────────────────────────────────────────────────────────────
  private frameCount: number = 0;
  private lastFrameMs: number = 0;

  constructor(canvas: HTMLCanvasElement, opts: OceanBridgeOptions = {}) {
    this.canvas = canvas;
    this.opts = {
      simMode:         opts.simMode         ?? 'sph',
      preset:          opts.preset          ?? 1,
      devicePixelRatio: opts.devicePixelRatio ?? 0.7,
      publishInterval: opts.publishInterval  ?? 10,
      eventTarget:     opts.eventTarget     ?? (typeof window !== 'undefined' ? window : new EventTarget()),
      cubemapPaths:    opts.cubemapPaths    ?? DEFAULT_CUBEMAP_PATHS,
    };
    this.simMode = this.opts.simMode;
    this.preset  = this.opts.preset;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Initialise WebGPU and allocate all GPU resources.
   * Safe to call multiple times; subsequent calls are no-ops.
   * Throws if WebGPU is not available on the current browser.
   */
  async init(): Promise<void> {
    if (this.initDone) return;

    if (!navigator.gpu) {
      throw new Error('OceanBridge: WebGPU is not supported in this browser.');
    }

    // ── Dynamic import of upstream modules ───────────────────────────────────
    // We import lazily so the WGSL shader strings (several KB each) are only
    // evaluated when the bridge is actually used, not at module-parse time.
    const [sphMod, mlsMod, renderMod, cameraMod, commonMod] = await Promise.all([
      import('../../../../upstream/webgpu-ocean/sph/sph'),
      import('../../../../upstream/webgpu-ocean/mls-mpm/mls-mpm'),
      import('../../../../upstream/webgpu-ocean/render/fluidRender'),
      import('../../../../upstream/webgpu-ocean/camera'),
      import('../../../../upstream/webgpu-ocean/common'),
    ]);

    this.mod = {
      SPHSimulator:    sphMod.SPHSimulator,
      MLSMPMSimulator: mlsMod.MLSMPMSimulator,
      FluidRenderer:   renderMod.FluidRenderer,
      Camera:          cameraMod.Camera,
      renderUniformsViews:  commonMod.renderUniformsViews,
      renderUniformsValues: commonMod.renderUniformsValues,
      numParticlesMax:  commonMod.numParticlesMax,
      sphParticleStructSize:    sphMod.sphParticleStructSize,
      mlsmpmParticleStructSize: mlsMod.mlsmpmParticleStructSize,
    };

    // ── WebGPU adapter + device ───────────────────────────────────────────────
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('OceanBridge: No suitable WebGPU adapter found.');
    }
    this.device = await adapter.requestDevice();

    // ── Canvas sizing ─────────────────────────────────────────────────────────
    const dpr = this.opts.devicePixelRatio;
    this.canvas.width  = dpr * this.canvas.clientWidth;
    this.canvas.height = dpr * this.canvas.clientHeight;

    // ── GPUCanvasContext ──────────────────────────────────────────────────────
    const ctx = this.canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!ctx) {
      throw new Error('OceanBridge: Failed to get WebGPU canvas context.');
    }
    this.context = ctx;
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    });

    // ── Cubemap texture ───────────────────────────────────────────────────────
    const cubemapTextureView = await this._loadCubemap(this.opts.cubemapPaths);

    // ── Shared GPU buffers ────────────────────────────────────────────────────
    const { sphParticleStructSize, mlsmpmParticleStructSize, numParticlesMax, renderUniformsValues } = this.mod;
    const maxStructSize = Math.max(sphParticleStructSize, mlsmpmParticleStructSize);

    this.particleBuffer = this.device.createBuffer({
      label: 'ocean-bridge:particles',
      size:  maxStructSize * numParticlesMax,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.posvelBuffer = this.device.createBuffer({
      label: 'ocean-bridge:posvel',
      size:  32 * numParticlesMax,   // 2 × vec3f + padding
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.renderUniformBuffer = this.device.createBuffer({
      label: 'ocean-bridge:renderUniforms',
      size:  (renderUniformsValues as ArrayBuffer).byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Seed texel_size into the shared renderUniformsViews struct
    this.mod.renderUniformsViews.texel_size.set([
      1.0 / this.canvas.width,
      1.0 / this.canvas.height,
    ]);

    // ── Simulators ────────────────────────────────────────────────────────────
    const { SPHSimulator, MLSMPMSimulator, FluidRenderer, Camera } = this.mod;

    this.sphSim = new SPHSimulator(
      this.particleBuffer, this.posvelBuffer,
      2 * SPH_RADIUS,
      this.device,
    );
    this.mlsmpmSim = new MLSMPMSimulator(
      this.particleBuffer, this.posvelBuffer,
      2 * MLSMPM_RADIUS,
      this.device,
    );

    // ── Renderers (one per sim mode to preserve filter-constant state) ────────
    this.sphRenderer = new FluidRenderer(
      this.device, this.canvas, this.presentationFormat,
      SPH_RADIUS, SPH_FOV,
      this.posvelBuffer, this.renderUniformBuffer, cubemapTextureView,
    );
    this.mlsmpmRenderer = new FluidRenderer(
      this.device, this.canvas, this.presentationFormat,
      MLSMPM_RADIUS, MLSMPM_FOV,
      this.posvelBuffer, this.renderUniformBuffer, cubemapTextureView,
    );

    // ── Camera ────────────────────────────────────────────────────────────────
    this.camera = new Camera(this.canvas);

    // ── Apply initial preset ──────────────────────────────────────────────────
    this._applyPreset(this.simMode, this.preset);

    this.initDone = true;
  }

  /** Start the animation frame loop.  Calls init() automatically if needed. */
  async start(): Promise<void> {
    if (!this.initDone) await this.init();
    if (this.running) return;
    this.running   = true;
    this.rafHandle = requestAnimationFrame(this._frame);
  }

  /** Pause the animation loop without releasing GPU resources. */
  stop(): void {
    this.running = false;
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = 0;
    }
  }

  /**
   * Release all GPU buffers and pipelines.  After destroy() the bridge
   * instance must not be reused — create a new one instead.
   */
  destroy(): void {
    this.stop();
    this.particleBuffer?.destroy();
    this.posvelBuffer?.destroy();
    this.renderUniformBuffer?.destroy();
    this.context?.unconfigure?.();
    this.initDone = false;
  }

  // ── Runtime controls ───────────────────────────────────────────────────────

  /**
   * Switch simulation mode.  Resets the particle layout to the current
   * preset for the new mode.
   */
  setSimMode(mode: SimMode): void {
    if (mode === this.simMode) return;
    this.simMode = mode;
    this._applyPreset(mode, this.preset);
  }

  /**
   * Select a particle-count preset (0 = smallest → 3 = largest).
   * The upstream simulation is immediately reset with new init positions.
   */
  setParticleCount(preset: PresetIndex): void {
    this.preset = preset;
    this._applyPreset(this.simMode, preset);
  }

  /**
   * Set the target box-width ratio in [0, 1].
   *   0   → narrowest (0.5 × initial width)
   *   0.5 → default
   *   1   → full width
   * The ratio is applied smoothly each frame, matching the upstream
   * "closing speed" animation in main.ts.
   */
  setBoxRatio(ratio: number): void {
    this.targetRatio = Math.max(0, Math.min(1, ratio));
  }

  /** Toggle billboard-sphere rendering mode (true = per-particle spheres). */
  setSphereMode(on: boolean): void {
    this.sphereMode = on;
  }

  /** Resize handler — call this when the host canvas changes size. */
  resize(width: number, height: number): void {
    if (!this.initDone) return;
    const dpr = this.opts.devicePixelRatio;
    this.canvas.width  = dpr * width;
    this.canvas.height = dpr * height;
    // Re-upload texel_size to the shared uniform struct
    this.mod.renderUniformsViews.texel_size.set([
      1.0 / this.canvas.width,
      1.0 / this.canvas.height,
    ]);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Load the cubemap faces and upload them as a GPUTexture.
   * Returns the cube-view used by FluidRenderer.
   */
  private async _loadCubemap(paths: OceanBridgeOptions['cubemapPaths']): Promise<GPUTextureView> {
    const bitmaps = await Promise.all(
      (paths as string[]).map(async (src) => {
        const resp = await fetch(src);
        return createImageBitmap(await resp.blob());
      }),
    );

    const tex = this.device.createTexture({
      dimension: '2d',
      size: [bitmaps[0].width, bitmaps[0].height, 6],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    for (let i = 0; i < bitmaps.length; i++) {
      this.device.queue.copyExternalImageToTexture(
        { source: bitmaps[i] },
        { texture: tex, origin: [0, 0, i] },
        [bitmaps[i].width, bitmaps[i].height],
      );
    }

    return tex.createView({ dimension: 'cube' });
  }

  /**
   * Apply a preset for the given simulation mode:
   *   - reset the appropriate simulator with new particle count + box sizes
   *   - reset the camera to the matching viewpoint
   *   - reset box-ratio animation state
   */
  private _applyPreset(mode: SimMode, idx: PresetIndex): void {
    if (!this.initDone) return;

    if (mode === 'sph') {
      this.initBoxSize = [...SPH_BOX_SIZES[idx]] as [number, number, number];
      this.realBoxSize = [...this.initBoxSize]  as [number, number, number];
      this.sphSim.reset(SPH_NUM_PARTICLES[idx], this.initBoxSize);
      this.camera.reset(
        this.canvas,
        SPH_DISTANCES[idx],
        [0, -(this.initBoxSize[1]) + 0.1, 0],
        SPH_FOV,
        SPH_ZOOM,
      );
    } else {
      this.initBoxSize = [...MLSMPM_BOX_SIZES[idx]] as [number, number, number];
      this.realBoxSize = [...this.initBoxSize]       as [number, number, number];
      this.mlsmpmSim.reset(MLSMPM_NUM_PARTICLES[idx], this.initBoxSize);
      this.camera.reset(
        this.canvas,
        MLSMPM_DISTANCES[idx],
        [this.initBoxSize[0] / 2, this.initBoxSize[1] / 4, this.initBoxSize[2] / 2],
        MLSMPM_FOV,
        MLSMPM_ZOOM,
      );
    }

    // Reset box-ratio animation
    this.boxRatio    = 1.0;
    this.targetRatio = 1.0;
  }

  /**
   * Per-frame animation callback, bound to `this`.
   * Mirrors the frame() closure in upstream/webgpu-ocean/main.ts.
   */
  private readonly _frame = (timestamp: number): void => {
    if (!this.running) return;

    const frameStart = performance.now();

    // ── Box-squeeze animation ─────────────────────────────────────────────────
    // targetRatio lives in [0, 1]; map to upstream [0.5, 1.0] range.
    const mappedTarget = this.targetRatio / 2 + 0.5;
    const minSpeed     = this.simMode === 'sph' ? -0.015 : -0.007;
    const delta        = Math.max(mappedTarget - this.boxRatio, minSpeed);
    this.boxRatio     += delta;

    this.realBoxSize[2] = this.initBoxSize[2] * this.boxRatio;

    if (this.simMode === 'sph') {
      this.sphSim.changeBoxSize(this.realBoxSize);
    } else {
      this.mlsmpmSim.changeBoxSize(this.realBoxSize);
    }

    // ── Upload render uniforms ────────────────────────────────────────────────
    this.device.queue.writeBuffer(
      this.renderUniformBuffer, 0,
      this.mod.renderUniformsValues as ArrayBuffer,
    );

    // ── Encode compute + render passes ────────────────────────────────────────
    const enc = this.device.createCommandEncoder();

    if (this.simMode === 'sph') {
      this.sphSim.execute(enc);
      this.sphRenderer.execute(
        this.context, enc,
        this.sphSim.numParticles,
        this.sphereMode,
      );
    } else {
      this.mlsmpmSim.execute(enc);
      this.mlsmpmRenderer.execute(
        this.context, enc,
        this.mlsmpmSim.numParticles,
        this.sphereMode,
      );
    }

    this.device.queue.submit([enc.finish()]);

    this.lastFrameMs = performance.now() - frameStart;

    // ── Telemetry publishing ──────────────────────────────────────────────────
    const interval = this.opts.publishInterval;
    if (interval > 0 && ++this.frameCount % interval === 0) {
      this._publish();
    }

    this.rafHandle = requestAnimationFrame(this._frame);
  };

  /** Dispatch an OceanFrameEvent on the configured EventTarget. */
  private _publish(): void {
    const numParticles =
      this.simMode === 'sph'
        ? this.sphSim.numParticles
        : this.mlsmpmSim.numParticles;

    const payload: OceanFrameEvent = {
      simMode:      this.simMode,
      numParticles,
      frameMs:      this.lastFrameMs,
      boxRatio:     this.boxRatio,
    };

    try {
      this.opts.eventTarget.dispatchEvent(
        new CustomEvent('ocean:frame', { detail: payload, bubbles: false }),
      );
    } catch {
      // EventTarget may not support CustomEvent in all environments (Node SSR).
      // Silently swallow to avoid breaking SSR builds.
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and initialise an OceanBridge in a single call.
 *
 * @example
 * ```ts
 * import { createOceanBridge } from '$lib/sph/ocean-bridge';
 *
 * const bridge = await createOceanBridge(canvasEl, {
 *   simMode: 'sph',
 *   preset: 1,
 * });
 * bridge.start();
 * ```
 */
export async function createOceanBridge(
  canvas: HTMLCanvasElement,
  opts: OceanBridgeOptions = {},
): Promise<OceanBridge> {
  const bridge = new OceanBridge(canvas, opts);
  await bridge.init();
  return bridge;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone WebGPU-availability probe
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns true if WebGPU is available in the current browser context.
 * Use this before mounting the canvas to provide a graceful fallback.
 *
 * @example
 * ```ts
 * if (!isWebGPUSupported()) {
 *   container.replaceWith(fallbackCanvas);
 * }
 * ```
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}
