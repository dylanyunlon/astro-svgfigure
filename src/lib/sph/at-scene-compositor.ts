/**
 * at-scene-compositor.ts — M730: AT Scene Compositor
 * ─────────────────────────────────────────────────────────────────────────────
 * Wires every AT rendering module into a single, runnable scene compositor.
 *
 * This is the top-level glue that turns the scattered per-effect modules
 * (PBR material, flower particles, spline particles, water surface,
 * Navier-Stokes fluid, volumetric light, bloom post-process, particle
 * compositor) into a coherent frame loop driven by SPH physics data.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Per-frame pipeline  (tick → render)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  ① SPH world readback
 *     Read particle positions + velocities from SPHWorld CPU buffers.
 *     Derive per-cell bounding boxes, densities, and flow vectors for
 *     physics-driven material modulation.
 *
 *  ② Navier-Stokes fluid step
 *     Advance the NS grid one frame.  Inject mouse/touch splats and
 *     cell-centre dye impulses derived from SPH velocities.
 *
 *  ③ Particle update (compute)
 *     Dispatch flower + spline particle lifecycle compute passes.
 *     Both renderers advance their tPos ping-pong textures.
 *
 *  ④ Particle sort + composite (compute + render)
 *     ParticleCompositor depth-sorts all active particles, then draws
 *     the alpha and additive-glow layers.
 *
 *  ⑤ PBR material pass
 *     Per-cell ATPBRMaterial (or ATMatcapFresnel) render pass.
 *     Species-specific material params are resolved through the
 *     CellMaterialSystem registry each time a cell is added.
 *
 *  ⑥ Water surface
 *     ATWaterSurface wave sim + mesh render + water-particle overlay.
 *
 *  ⑦ Volumetric light
 *     Screen-space god rays (occlusion → radial blur → Mie scatter).
 *
 *  ⑧ Bloom post-process
 *     UE5-style bright extract → separable Gaussian blur → composite.
 *
 *  ⑨ Final composite → canvas
 *     The bloom output is the final image, presented to the swap-chain
 *     surface (canvas).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Cell management
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * addCell(cellId, species, bbox) registers a rendering identity for a cell.
 * The compositor looks up the species in two registries:
 *   • species-shader-registry.ts — SDF shape, bloom preset, physics bindings
 *   • cell-material-system.ts   — PBR/matcap params, WGSL patch, modulators
 *
 * Each cell gets its own ATPBRMaterial (or ATMatcapFresnel) instance so
 * species-specific uniforms can be uploaded independently.
 *
 * removeCell(cellId) tears down the GPU material for that cell.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const compositor = new ATSceneCompositor();
 *   await compositor.init(device, canvas);
 *
 *   compositor.addCell('cell-0', 'attention', { x: 0, y: 0, w: 1, h: 1 });
 *   compositor.addCell('cell-1', 'ffn',       { x: 1, y: 0, w: 1, h: 1 });
 *
 *   // render loop:
 *   function frame() {
 *     compositor.tick(dt, sphWorld);
 *     requestAnimationFrame(frame);
 *   }
 *   frame();
 *
 *   // on resize:
 *   compositor.resize(newW, newH);
 *
 * Research: xiaodi #M730 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

import {
  ATPBRMaterial,
  ATMatcapFresnel,
  type PBRParams,
  type MatcapParams,
} from './at-pbr-material.js';

import {
  ATFlowerParticleRenderer,
  type FlowerEdgeSpline,
} from './at-flower-particle.js';

import {
  ATSplineParticleLife,
  type EdgeSpline,
} from './at-spline-particle.js';

import {
  ATWaterSurface,
  type ATWaterSurfaceConfig,
} from './at-water-surface.js';

import {
  ATBloomPostProcess,
  type ATBloomParams,
} from './at-bloom-postprocess.js';

import {
  NavierStokesFluid,
  type NavierStokesSplat,
} from './at-navier-stokes.js';

import {
  ATVolumetricLight,
  type ATVolumetricLightParams,
} from './at-volumetric-light.js';

import {
  ParticleCompositor,
  LayerType,
  type LayerDescriptor,
} from './particle-compositor.js';

import {
  getSpeciesShaderConfig,
  type SpeciesShaderConfig,
} from './species-shader-registry.js';

import {
  getCellMaterial,
  type CellSpecies,
  type SpeciesMaterialDef,
} from './cell-material-system.js';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Axis-aligned bounding box for a cell in SPH domain units. */
export interface CellBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Minimal read-only view of SPH world state consumed by the compositor. */
export interface SPHWorldView {
  /** Per-particle X positions (SPH domain units). */
  readonly x: Float32Array;
  /** Per-particle Y positions (SPH domain units). */
  readonly y: Float32Array;
  /** Per-particle X velocities. */
  readonly vx: Float32Array;
  /** Per-particle Y velocities. */
  readonly vy: Float32Array;
  /** Per-particle species index. */
  readonly species: Uint32Array;
  /** Active particle count. */
  readonly count: number;
}

/**
 * Internal bookkeeping for a registered cell's rendering resources.
 */
interface CellEntry {
  cellId:     string;
  species:    string;
  bbox:       CellBBox;
  shaderCfg:  SpeciesShaderConfig;
  materialDef: SpeciesMaterialDef | null;
  /** PBR material instance (null if species uses matcap). */
  pbrMat:     ATPBRMaterial | null;
  /** Matcap material instance (null if species uses pbr/iridescence). */
  matcapMat:  ATMatcapFresnel | null;
}

/**
 * Configuration for the ATSceneCompositor.
 * All fields are optional — sensible defaults are used.
 */
export interface ATSceneCompositorConfig {
  /** Texture format for swap-chain / render targets. Default 'bgra8unorm'. */
  format?: GPUTextureFormat;

  /** Flower particle edge splines for the initial scene. */
  flowerEdges?: FlowerEdgeSpline[];

  /** Spline particle edge splines for the initial scene. */
  splineEdges?: EdgeSpline[];

  /** ATWaterSurface configuration overrides. */
  waterConfig?: ATWaterSurfaceConfig;

  /** ATBloomPostProcess parameter overrides. */
  bloomParams?: ATBloomParams;

  /** ATVolumetricLight parameter overrides. */
  vlParams?: ATVolumetricLightParams;

  /** Whether to enable each optional rendering pass. */
  passes?: Partial<CompositorPassFlags>;
}

/** Flags controlling which rendering passes are active. */
export interface CompositorPassFlags {
  /** Navier-Stokes fluid simulation. Default true. */
  navierStokes: boolean;
  /** Flower particle system. Default true. */
  flowerParticle: boolean;
  /** Spline particle system. Default true. */
  splineParticle: boolean;
  /** Water surface (wave sim + mesh + particles). Default true. */
  waterSurface: boolean;
  /** Volumetric light god rays. Default true. */
  volumetricLight: boolean;
  /** Bloom post-process. Default true. */
  bloom: boolean;
  /** Per-cell PBR/matcap material rendering. Default true. */
  cellMaterials: boolean;
  /** Particle depth-sort compositor. Default true. */
  particleCompositor: boolean;
}

const DEFAULT_PASS_FLAGS: CompositorPassFlags = {
  navierStokes:       true,
  flowerParticle:     true,
  splineParticle:     true,
  waterSurface:       true,
  volumetricLight:    true,
  bloom:              true,
  cellMaterials:      true,
  particleCompositor: true,
};

// ─────────────────────────────────────────────────────────────────────────────
// FBO helper — off-screen colour + depth render target pair
// ─────────────────────────────────────────────────────────────────────────────

interface FBO {
  color:     GPUTexture;
  colorView: GPUTextureView;
  depth:     GPUTexture;
  depthView: GPUTextureView;
}

function createFBO(
  device: GPUDevice,
  w: number,
  h: number,
  format: GPUTextureFormat,
  label: string,
): FBO {
  const color = device.createTexture({
    label:  `${label}-color`,
    size:   [w, h],
    format,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING   |
      GPUTextureUsage.COPY_SRC          |
      GPUTextureUsage.COPY_DST,
  });
  const depth = device.createTexture({
    label:  `${label}-depth`,
    size:   [w, h],
    format: 'depth24plus',
    usage:  GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return {
    color,
    colorView: color.createView(),
    depth,
    depthView: depth.createView(),
  };
}

function destroyFBO(fbo: FBO): void {
  fbo.color.destroy();
  fbo.depth.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// ATSceneCompositor
// ─────────────────────────────────────────────────────────────────────────────

export class ATSceneCompositor {
  // ── Core WebGPU ────────────────────────────────────────────────────────────
  private device!:  GPUDevice;
  private canvas!:  HTMLCanvasElement;
  private ctx!:     GPUCanvasContext;
  private format:   GPUTextureFormat = 'bgra8unorm';

  // ── Dimensions ─────────────────────────────────────────────────────────────
  private width  = 0;
  private height = 0;

  // ── Sub-systems ────────────────────────────────────────────────────────────
  private flower!:        ATFlowerParticleRenderer;
  private spline!:        ATSplineParticleLife;
  private water!:         ATWaterSurface;
  private nsFluid!:       NavierStokesFluid;
  private vlight!:        ATVolumetricLight;
  private bloom!:         ATBloomPostProcess;
  private particleComp!:  ParticleCompositor;

  // ── Cell registry ──────────────────────────────────────────────────────────
  private cells: Map<string, CellEntry> = new Map();

  // ── Intermediate FBOs ──────────────────────────────────────────────────────
  //
  // Frame pipeline:
  //   sceneFBO (clear → cell materials + particles)
  //     → waterFBO (water surface overlay)
  //       → vlFBO (volumetric light)
  //         → bloomFBO (bloom composite)
  //           → canvas swap-chain
  //
  private sceneFBO!: FBO;
  private waterFBO!: FBO;
  private vlFBO!:    FBO;

  // ── Scene uniform buffer (shared by water surface renderPass) ──────────
  private sceneUniformBuf!: GPUBuffer;

  // ── Param caches (for resize recreation of immutable sub-systems) ──────
  private vlParamsCache: ATVolumetricLightParams = {};
  private bloomParamsCache: ATBloomParams = {};

  // ── Configuration / flags ──────────────────────────────────────────────────
  private passes: CompositorPassFlags = { ...DEFAULT_PASS_FLAGS };
  private elapsed = 0;

  // ── Lifecycle state ────────────────────────────────────────────────────────
  private initialised = false;
  private destroyed   = false;

  // ─────────────────────────────────────────────────────────────────────────
  // init(device, canvas)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Initialise all sub-systems.
   *
   * Must be called exactly once before any other method.
   * The caller provides an already-initialised GPUDevice.
   *
   * @param device — WebGPU device.
   * @param canvas — Target HTMLCanvasElement (must have a configured
   *                 GPUCanvasContext, or this method configures one).
   * @param cfg    — Optional configuration overrides.
   */
  async init(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    cfg: ATSceneCompositorConfig = {},
  ): Promise<void> {
    if (this.initialised) return;

    this.device = device;
    this.canvas = canvas;
    this.format = cfg.format ?? 'bgra8unorm';
    this.width  = canvas.width;
    this.height = canvas.height;

    if (cfg.passes) {
      Object.assign(this.passes, cfg.passes);
    }

    // ── Canvas context ────────────────────────────────────────────────────
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    this.ctx.configure({
      device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    const W = this.width;
    const H = this.height;

    // ── Intermediate FBOs ─────────────────────────────────────────────────
    this.sceneFBO = createFBO(device, W, H, this.format, 'at-comp-scene');
    this.waterFBO = createFBO(device, W, H, this.format, 'at-comp-water');
    this.vlFBO    = createFBO(device, W, H, this.format, 'at-comp-vl');

    // ── Particle systems ──────────────────────────────────────────────────
    const flowerEdges = cfg.flowerEdges ?? [];
    const splineEdges = cfg.splineEdges ?? [];

    this.flower = new ATFlowerParticleRenderer(device, canvas, flowerEdges, {});
    await this.flower.build();

    this.spline = new ATSplineParticleLife(device, canvas, splineEdges, {});
    await this.spline.build();

    // ── Particle compositor (depth-sort + composite) ──────────────────────
    this.particleComp = new ParticleCompositor(device, canvas);
    // Layers are registered lazily when the renderers have valid tPos views.
    // For now, build the compositor shell.
    await this.particleComp.build();

    // ── Navier-Stokes fluid ───────────────────────────────────────────────
    this.nsFluid = new NavierStokesFluid(device);

    // ── Water surface ─────────────────────────────────────────────────────
    this.water = new ATWaterSurface(device, this.format, cfg.waterConfig ?? {});
    await this.water.build();

    // ── Scene uniform buffer (water surface needs view/proj/eye) ─────────
    this.sceneUniformBuf = device.createBuffer({
      label: 'at-comp-scene-uniforms',
      size:  144,   // 36 × f32 = 144 bytes (viewProj + modelMat + eye)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Volumetric light ──────────────────────────────────────────────────
    this.vlight = await ATVolumetricLight.create(device, this.format, W, H);
    if (cfg.vlParams) {
      Object.assign(this.vlParamsCache, cfg.vlParams);
      this.vlight.setParams(cfg.vlParams);
    }

    // ── Bloom post-process ────────────────────────────────────────────────
    this.bloom = await ATBloomPostProcess.create(device, this.format, W, H);
    if (cfg.bloomParams) {
      Object.assign(this.bloomParamsCache, cfg.bloomParams);
      this.bloom.setParams(cfg.bloomParams);
    }

    this.initialised = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // addCell(cellId, species, bbox)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a cell and create its per-species rendering material.
   *
   * The species string is looked up in:
   *   1. species-shader-registry  → SpeciesShaderConfig (SDF, bloom, physics)
   *   2. cell-material-system     → SpeciesMaterialDef (PBR params, WGSL patch)
   *
   * If the species string doesn't match a CellSpecies key in the material
   * registry, we still create a fallback PBR material with the shader
   * registry's material params.
   *
   * @param cellId  — Unique cell identifier (e.g. 'cell-0').
   * @param species — Species string (e.g. 'attention', 'cil-eye', 'ffn').
   * @param bbox    — Axis-aligned bounding box in SPH domain units.
   */
  async addCell(
    cellId:  string,
    species: string,
    bbox:    CellBBox,
  ): Promise<void> {
    if (this.destroyed || !this.initialised) return;
    if (this.cells.has(cellId)) return; // already registered

    // ── Lookup registries ─────────────────────────────────────────────────
    let shaderCfg: SpeciesShaderConfig;
    try {
      shaderCfg = getSpeciesShaderConfig(species);
    } catch {
      // Species not in shader registry — use a safe default config stub
      shaderCfg = getSpeciesShaderConfig('cil-eye');
    }

    let materialDef: SpeciesMaterialDef | null = null;
    try {
      // CellSpecies type: 'attention' | 'ffn' | 'layernorm' | 'embedding' | 'softmax'
      materialDef = getCellMaterial(species as CellSpecies);
    } catch {
      // Not a CellSpecies — that's fine, we use shader registry params instead
    }

    // ── Create GPU material ───────────────────────────────────────────────
    let pbrMat:    ATPBRMaterial  | null = null;
    let matcapMat: ATMatcapFresnel | null = null;

    const matType = materialDef?.materialType ?? shaderCfg.materialType;

    if (matType === 'matcap') {
      matcapMat = await ATMatcapFresnel.create(this.device, this.format);
      if (materialDef?.matcapParams) {
        matcapMat.setParams(materialDef.matcapParams);
      }
    } else {
      // 'pbr' or 'iridescence' — both use ATPBRMaterial
      pbrMat = await ATPBRMaterial.create(this.device, this.format);
      const pbrParams: Partial<PBRParams> =
        materialDef?.pbrParams ??
        shaderCfg.materialParams as Partial<PBRParams>;
      if (pbrParams) {
        pbrMat.setParams(pbrParams);
      }
    }

    const entry: CellEntry = {
      cellId,
      species,
      bbox,
      shaderCfg,
      materialDef,
      pbrMat,
      matcapMat,
    };

    this.cells.set(cellId, entry);
  }

  /**
   * Remove a cell and destroy its GPU material resources.
   */
  removeCell(cellId: string): void {
    const entry = this.cells.get(cellId);
    if (!entry) return;
    entry.pbrMat?.destroy();
    entry.matcapMat?.destroy();
    this.cells.delete(cellId);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick(dt, sphWorld)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance all sub-systems by one frame and present to the canvas.
   *
   * Frame pipeline:
   *   1. Update physics (cell material modulation from SPH)
   *   2. Navier-Stokes step (fluid sim)
   *   3. Particle update (flower + spline compute passes)
   *   4. Clear sceneFBO → render cell materials → particle composite
   *   5. Water surface → waterFBO
   *   6. Volumetric light → vlFBO
   *   7. Bloom post-process → canvas
   *
   * @param dt       — Delta time in seconds (e.g. 1/60).
   * @param sphWorld — Read-only view of the SPH particle state.
   *                   Pass null to skip physics-driven updates.
   */
  tick(dt: number, sphWorld: SPHWorldView | null = null): void {
    if (this.destroyed || !this.initialised) return;

    this.elapsed += dt;

    // ── Acquire swap-chain texture ────────────────────────────────────────
    const swapTex  = this.ctx.getCurrentTexture();
    const dstView  = swapTex.createView();

    const encoder = this.device.createCommandEncoder({
      label: 'at-scene-compositor-frame',
    });

    // ──────────────────────────────────────────────────────────────────────
    // Phase 1: Physics → material modulation
    // ──────────────────────────────────────────────────────────────────────

    if (sphWorld && sphWorld.count > 0) {
      this._updateCellMaterialsFromPhysics(sphWorld, dt);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 2: Navier-Stokes fluid step
    // ──────────────────────────────────────────────────────────────────────

    if (this.passes.navierStokes) {
      // Inject per-cell dye splats derived from SPH velocities
      if (sphWorld && sphWorld.count > 0) {
        this._injectNSSplats(encoder, sphWorld);
      }
      this.nsFluid.step(encoder);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 3: Particle compute passes
    // ──────────────────────────────────────────────────────────────────────

    if (this.passes.flowerParticle && this.flower.isBuilt) {
      this.flower.update(encoder, this.elapsed, dt);
    }

    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.update(encoder, this.elapsed, dt);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 4: Clear sceneFBO → cell material passes → particle composite
    // ──────────────────────────────────────────────────────────────────────

    this._clearFBO(encoder, this.sceneFBO);

    // ── 4a: Per-cell material render ──────────────────────────────────────
    if (this.passes.cellMaterials) {
      for (const entry of this.cells.values()) {
        if (entry.pbrMat) {
          entry.pbrMat.tick(dt);
          entry.pbrMat.render(
            encoder,
            this.sceneFBO.colorView,
            this.sceneFBO.depthView,
          );
        } else if (entry.matcapMat) {
          entry.matcapMat.tick(dt);
          entry.matcapMat.render(encoder, this.sceneFBO.colorView);
        }
      }
    }

    // ── 4b: Particle render (flower + spline) ─────────────────────────────
    if (this.passes.flowerParticle && this.flower.isBuilt) {
      this.flower.render(
        encoder,
        this.sceneFBO.colorView,
        this.sceneFBO.depthView,
      );
    }

    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.render(
        encoder,
        this.sceneFBO.colorView,
        this.sceneFBO.depthView,
      );
    }

    // ── 4c: Particle compositor (depth-sort + glow) ───────────────────────
    if (this.passes.particleCompositor) {
      this.particleComp.sort(encoder);
      this.particleComp.renderAlpha(encoder, this.sceneFBO.colorView);
      this.particleComp.renderGlow(encoder, this.sceneFBO.colorView);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 5: Water surface → waterFBO
    // ──────────────────────────────────────────────────────────────────────

    // Copy sceneFBO → waterFBO as base
    this._copyTexture(encoder, this.sceneFBO.color, this.waterFBO.color);

    if (this.passes.waterSurface) {
      this.water.tick(encoder, this.elapsed);
      this.water.renderPass(
        encoder,
        this.waterFBO.colorView,
        this.waterFBO.depthView,
        this.sceneUniformBuf,
      );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 6: Volumetric light → vlFBO
    // ──────────────────────────────────────────────────────────────────────

    if (this.passes.volumetricLight) {
      this.vlight.tick(dt);
      this.vlight.render(
        encoder,
        this.waterFBO.color,   // input scene texture
        this.vlFBO.colorView,  // composited output
      );
    } else {
      this._copyTexture(encoder, this.waterFBO.color, this.vlFBO.color);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Phase 7: Bloom → final output (canvas swap-chain)
    // ──────────────────────────────────────────────────────────────────────

    if (this.passes.bloom) {
      this.bloom.render(
        encoder,
        this.vlFBO.color,   // input scene texture
        dstView,            // final output to canvas
      );
    } else {
      // Blit vlFBO directly to swap-chain
      this._blitToSwapChain(encoder, this.vlFBO.color, dstView);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Submit
    // ──────────────────────────────────────────────────────────────────────

    this.device.queue.submit([encoder.finish()]);

    // ── Async particle handoff readback (non-blocking) ────────────────────
    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.scheduleHandoffReadback().catch(() => {});
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // resize(w, h)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Handle canvas resize.  Reallocates all intermediate FBOs and notifies
   * sub-systems that need size-dependent resources.
   *
   * @param w — New width in pixels.
   * @param h — New height in pixels.
   */
  resize(w: number, h: number): void {
    if (this.destroyed || !this.initialised) return;
    if (this.width === w && this.height === h) return;

    this.width  = w;
    this.height = h;

    // Reconfigure canvas context
    this.ctx.configure({
      device:    this.device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    // Reallocate FBOs
    destroyFBO(this.sceneFBO);
    destroyFBO(this.waterFBO);
    destroyFBO(this.vlFBO);

    this.sceneFBO = createFBO(this.device, w, h, this.format, 'at-comp-scene');
    this.waterFBO = createFBO(this.device, w, h, this.format, 'at-comp-water');
    this.vlFBO    = createFBO(this.device, w, h, this.format, 'at-comp-vl');

    // Sub-systems that need resize notification:
    // Volumetric light and bloom hold size-dependent textures.
    // They are immutable after creation, so we must recreate them.
    // This is acceptable as resize is infrequent.
    this._recreateVolumetricLight(w, h);
    this._recreateBloom(w, h);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration API
  // ─────────────────────────────────────────────────────────────────────────

  /** Update bloom post-process parameters. */
  setBloomParams(p: ATBloomParams): void {
    Object.assign(this.bloomParamsCache, p);
    this.bloom?.setParams(p);
  }

  /** Update volumetric light parameters. */
  setVolumetricLightParams(p: ATVolumetricLightParams): void {
    Object.assign(this.vlParamsCache, p);
    this.vlight?.setParams(p);
  }

  /** Enable/disable individual rendering passes. */
  setPassFlags(flags: Partial<CompositorPassFlags>): void {
    Object.assign(this.passes, flags);
  }

  /**
   * Inject a Navier-Stokes splat (e.g. from mouse/touch interaction).
   * The splat is queued and applied on the next tick().
   */
  private pendingSplats: NavierStokesSplat[] = [];

  queueSplat(splat: NavierStokesSplat): void {
    this.pendingSplats.push(splat);
  }

  /** Add a drop to the water surface (e.g. cell hitting water). */
  addWaterDrop(x: number, y: number, radius: number, strength: number): void {
    this.water?.addDrop(x, y, radius, strength);
  }

  /**
   * Replace flower particle edge splines at runtime.
   * Triggers a full particle rebuild — existing state is discarded.
   */
  async setFlowerEdges(edges: FlowerEdgeSpline[]): Promise<void> {
    if (!this.initialised) return;
    await this.flower.setEdges(edges);
  }

  /**
   * Replace spline particle edge splines at runtime.
   * Triggers a full particle rebuild — existing state is discarded.
   */
  async setSplineEdges(edges: EdgeSpline[]): Promise<void> {
    if (!this.initialised) return;
    await this.spline.setEdges(edges);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources.  The compositor must not be used after this.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    // Destroy cell materials
    for (const entry of this.cells.values()) {
      entry.pbrMat?.destroy();
      entry.matcapMat?.destroy();
    }
    this.cells.clear();

    // Destroy sub-systems
    this.flower?.destroy();
    this.spline?.destroy();
    this.water?.destroy();
    this.nsFluid?.destroy();
    this.vlight?.destroy();
    this.bloom?.destroy();
    this.particleComp?.destroy();

    // Destroy FBOs
    if (this.sceneFBO) destroyFBO(this.sceneFBO);
    if (this.waterFBO) destroyFBO(this.waterFBO);
    if (this.vlFBO)    destroyFBO(this.vlFBO);

    // Destroy shared buffers
    this.sceneUniformBuf?.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  get isInitialised(): boolean { return this.initialised; }
  get isDestroyed():   boolean { return this.destroyed; }
  get cellCount():     number  { return this.cells.size; }
  get elapsedTime():   number  { return this.elapsed; }

  /** Iterate registered cell IDs. */
  cellIds(): IterableIterator<string> {
    return this.cells.keys();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — FBO operations
  // ─────────────────────────────────────────────────────────────────────────

  /** Clear an FBO to transparent black with a fresh depth buffer. */
  private _clearFBO(encoder: GPUCommandEncoder, fbo: FBO): void {
    const pass = encoder.beginRenderPass({
      label: 'at-comp-clear',
      colorAttachments: [{
        view:       fbo.colorView,
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view:           fbo.depthView,
        depthLoadOp:    'clear',
        depthStoreOp:   'store',
        depthClearValue: 1.0,
      },
    });
    pass.end();
  }

  /**
   * GPU texture-to-texture copy (same size).
   * Used to carry forward an FBO's colour into the next stage's base.
   */
  private _copyTexture(
    encoder: GPUCommandEncoder,
    src: GPUTexture,
    dst: GPUTexture,
  ): void {
    encoder.copyTextureToTexture(
      { texture: src },
      { texture: dst },
      [this.width, this.height],
    );
  }

  /**
   * Direct blit from a texture to the swap-chain view when bloom is disabled.
   * Uses copyTextureToTexture since both are the same format and size.
   */
  private _blitToSwapChain(
    encoder: GPUCommandEncoder,
    src: GPUTexture,
    dstView: GPUTextureView,
  ): void {
    // The swap-chain texture is obtained from dstView's parent texture.
    // Since copyTextureToTexture needs a GPUTexture, we use the
    // current texture reference from the context.
    const swapTex = this.ctx.getCurrentTexture();
    encoder.copyTextureToTexture(
      { texture: src },
      { texture: swapTex },
      [this.width, this.height],
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Physics-driven material updates
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Derive per-cell aggregate physics and modulate material params.
   *
   * For each registered cell, we scan the SPH particle arrays to find
   * particles whose position falls within the cell's bounding box.
   * We compute a simple average velocity magnitude (kinetic energy proxy)
   * and use it to modulate the cell's material params:
   *   - Higher KE → increased iridescence / roughness shift
   *   - Bloom strength pulse tied to density
   */
  private _updateCellMaterialsFromPhysics(
    sph: SPHWorldView,
    _dt: number,
  ): void {
    for (const entry of this.cells.values()) {
      const { bbox, pbrMat, matcapMat } = entry;
      if (!pbrMat && !matcapMat) continue;

      // Aggregate particles within the cell bbox
      let sumVelSq = 0;
      let count    = 0;

      for (let i = 0; i < sph.count; i++) {
        const px = sph.x[i];
        const py = sph.y[i];
        if (
          px >= bbox.x && px <= bbox.x + bbox.w &&
          py >= bbox.y && py <= bbox.y + bbox.h
        ) {
          const vx = sph.vx[i];
          const vy = sph.vy[i];
          sumVelSq += vx * vx + vy * vy;
          count++;
        }
      }

      if (count === 0) continue;

      // Normalised kinetic energy proxy (clamped to [0, 1])
      const avgKE = Math.min(1.0, (sumVelSq / count) * 2.0);
      // Density proxy: particle count relative to bbox area
      const area    = Math.max(bbox.w * bbox.h, 0.001);
      const density = Math.min(1.0, count / (area * 500));

      if (pbrMat) {
        // Modulate roughness and iridescence intensity with KE
        pbrMat.setParams({
          roughness:    0.15 + avgKE * 0.45,    // smoother at rest, rougher at high KE
          iridStrength: 0.3 + avgKE * 0.7,      // more shimmer with velocity
        } as Partial<PBRParams>);
      }

      if (matcapMat) {
        // Modulate Fresnel power with density
        matcapMat.setParams({
          fresnelPower: 2.0 + density * 4.0,
        } as Partial<MatcapParams>);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Navier-Stokes splat injection
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Inject NS splats from:
   *   a) queued user splats (mouse/touch)
   *   b) SPH cell-centre velocity impulses (automated coupling)
   */
  private _injectNSSplats(
    encoder: GPUCommandEncoder,
    sph: SPHWorldView,
  ): void {
    // (a) User-queued splats
    for (const splat of this.pendingSplats) {
      this.nsFluid.splat(encoder, splat);
    }
    this.pendingSplats.length = 0;

    // (b) Automated: inject one dye splat per cell based on average velocity
    for (const entry of this.cells.values()) {
      const { bbox } = entry;
      let sumVx = 0;
      let sumVy = 0;
      let count = 0;

      for (let i = 0; i < sph.count; i++) {
        const px = sph.x[i];
        const py = sph.y[i];
        if (
          px >= bbox.x && px <= bbox.x + bbox.w &&
          py >= bbox.y && py <= bbox.y + bbox.h
        ) {
          sumVx += sph.vx[i];
          sumVy += sph.vy[i];
          count++;
        }
      }

      if (count < 5) continue; // skip near-empty cells

      const avgVx = sumVx / count;
      const avgVy = sumVy / count;
      const speed = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
      if (speed < 0.01) continue; // skip negligible velocity

      // Splat position: cell centre, normalised to [0, 1]
      const cx = (bbox.x + bbox.w * 0.5) / (this.width || 1);
      const cy = (bbox.y + bbox.h * 0.5) / (this.height || 1);

      // Dye colour derived from species shader config
      const cfg = entry.shaderCfg;
      const mp  = cfg.materialParams;
      const dyeR = (mp as any).albedo?.[0] ?? 0.5;
      const dyeG = (mp as any).albedo?.[1] ?? 0.5;
      const dyeB = (mp as any).albedo?.[2] ?? 0.8;

      this.nsFluid.splat(encoder, {
        x:  cx,
        y:  cy,
        vx: avgVx * 0.5,
        vy: avgVy * 0.5,
        color: [dyeR, dyeG, dyeB],
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — Resize sub-system recreation
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Recreate volumetric light pass at new resolution.
   * VL holds half-res intermediate textures that must match viewport size.
   */
  private async _recreateVolumetricLight(w: number, h: number): Promise<void> {
    this.vlight.destroy();
    this.vlight = await ATVolumetricLight.create(this.device, this.format, w, h);
    if (Object.keys(this.vlParamsCache).length > 0) {
      this.vlight.setParams(this.vlParamsCache);
    }
  }

  /**
   * Recreate bloom pass at new resolution.
   * Bloom holds internal textures at the old size.
   */
  private async _recreateBloom(w: number, h: number): Promise<void> {
    this.bloom.destroy();
    this.bloom = await ATBloomPostProcess.create(this.device, this.format, w, h);
    if (Object.keys(this.bloomParamsCache).length > 0) {
      this.bloom.setParams(this.bloomParamsCache);
    }
  }
}
