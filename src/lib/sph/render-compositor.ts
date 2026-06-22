/**
 * render-compositor.ts — M745: Final All-Pass Render Compositor
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * The definitive top-level rendering orchestrator that unifies every AT
 * rendering module into a single, coherent per-frame pipeline.  Supersedes
 * both `at-scene-compositor.ts` (M730) and `at-render-pipeline.ts` (M720)
 * by combining their responsibilities and adding the passes they each lack:
 *
 *   M730 (ATSceneCompositor)  → cell registry, SPH physics coupling, NS fluid
 *   M720 (ATRenderPipeline)   → LUT grade, scene matrices, edge-chain FBOs
 *   NEW  (RenderCompositor)   → environment FX, atmosphere, post-process,
 *                                per-pass GPU timing, performance budget
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Per-frame pipeline (13 passes)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  ┌─── Pass 0 ── Environment Background ────────────────────────────────────┐
 *  │  EnvironmentFx  (brick-tile grid + voronoise + chromatic aberration)    │
 *  │  Renders into envFBO as the opaque background layer.                    │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ envFBO
 *                ▼
 *  ┌─── Pass 1 ── Navier-Stokes Fluid Step (compute) ───────────────────────┐
 *  │  NavierStokesFluid  (splat → advect → vorticity → pressure → project)  │
 *  │  Injects mouse/touch splats + SPH-derived cell-centre dye impulses.    │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ NS fluid textures (vel, dye — sampled by later passes)
 *                ▼
 *  ┌─── Pass 2 ── Particle Compute (GPGPU) ─────────────────────────────────┐
 *  │  ATFlowerParticleRenderer.update()  — spiral lifecycle compute          │
 *  │  ATSplineParticleLife.update()      — Catmull-Rom + curl-noise compute  │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ tPos ping-pong textures updated
 *                ▼
 *  ┌─── Pass 3 ── Geometry: Cell Materials (clear → geoFBO) ────────────────┐
 *  │  Per-cell ATPBRMaterial or ATMatcapFresnel render.                      │
 *  │  Species-specific params resolved via cell-material-system +            │
 *  │  species-shader-registry.  Physics-driven modulation applied.           │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ geoFBO
 *                ▼
 *  ┌─── Pass 4 ── Flower Particles (render over geoFBO) ────────────────────┐
 *  │  ATFlowerParticleRenderer.render()  — instanced quad draw               │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │
 *                ▼
 *  ┌─── Pass 5 ── Spline Particles (render over geoFBO) ────────────────────┐
 *  │  ATSplineParticleLife.render()      — instanced quad draw               │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │
 *                ▼
 *  ┌─── Pass 6 ── Particle Compositor (compute sort + alpha + glow) ────────┐
 *  │  ParticleCompositor.sort()         — GPU bitonic sort on depth keys     │
 *  │  ParticleCompositor.renderAlpha()  — back-to-front alpha blending       │
 *  │  ParticleCompositor.renderGlow()   — additive glow halo                 │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ sceneFBO  (env + geo + particles composited)
 *                ▼
 *  ┌─── Pass 7 ── Water Surface (wave sim + mesh + splash) ─────────────────┐
 *  │  ATWaterSurface.tick() + renderPass()                                   │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ waterFBO
 *                ▼
 *  ┌─── Pass 8 ── Volumetric Light (god-rays) ──────────────────────────────┐
 *  │  ATVolumetricLight  (occlusion → radial blur → Mie scatter)            │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ vlFBO
 *                ▼
 *  ┌─── Pass 9 ── Atmosphere (Rayleigh + Mie scatter + depth fog) ──────────┐
 *  │  AtmospherePass  (atmospheric perspective + chromatic aberration)       │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ atmoFBO
 *                ▼
 *  ┌─── Pass 10 ── Bloom (UE5-style threshold + gaussian + composite) ──────┐
 *  │  ATBloomPostProcess  (bright extract → blur chain → composite)          │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ bloomFBO
 *                ▼
 *  ┌─── Pass 11 ── Post-Process (Kuwahara / Edge / Ink) ────────────────────┐
 *  │  PostProcessPipeline  (artistic style transforms)                       │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ postFBO
 *                ▼
 *  ┌─── Pass 12 ── LUT Colour Grade (3-D LUT → swap-chain) ────────────────┐
 *  │  Inline WGSL trilinear LUT sample → final output to canvas             │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *                │ → swap-chain surface (canvas)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Cell management  (inherited from ATSceneCompositor pattern)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * addCell(cellId, species, bbox) registers a rendering identity for a cell.
 * removeCell(cellId) tears down the GPU material for that cell.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const comp = new RenderCompositor();
 *   await comp.init(device, canvas, {
 *     lutStyle:    'DEEP_OCEAN',
 *     postStyle:   { kuwahara: true, edge: false, ink: false },
 *     passes:      { atmosphere: true, environmentFx: true },
 *   });
 *
 *   comp.addCell('cell-0', 'attention', { x: 0, y: 0, w: 1, h: 1 });
 *
 *   // render loop:
 *   function frame() {
 *     comp.tick(1 / 60, sphWorldView);
 *     requestAnimationFrame(frame);
 *   }
 *   frame();
 *
 *   // on resize:
 *   comp.resize(newW, newH);
 *
 * Research: xiaodi #M745 — cell-pubsub-loop
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

import {
  AtmospherePass,
  type AtmosphereParams,
} from './atmosphere.js';

import {
  EnvironmentFx,
  type EnvironmentFxConfig,
} from './environment-fx.js';

import {
  PostProcessPipeline,
  type PostProcessStyle,
  type PostProcessParams,
} from './post-process.js';

import {
  LutGenerator,
  type LutStyleName,
} from './lut-generator.js';

// ─────────────────────────────────────────────────────────────────────────────
// LUT pass — inline WGSL (full-screen 3-D LUT grade blit)
// Reused from at-render-pipeline.ts, inlined to keep this module self-contained.
// ─────────────────────────────────────────────────────────────────────────────

const LUT_PASS_WGSL = /* wgsl */`
struct LutUniforms {
  lutSize     : f32,
  strength    : f32,
  _pad0       : f32,
  _pad1       : f32,
}

@group(0) @binding(0) var<uniform> u     : LutUniforms;
@group(0) @binding(1) var          sScene: sampler;
@group(0) @binding(2) var          tScene: texture_2d<f32>;
@group(0) @binding(3) var          sLut  : sampler;
@group(0) @binding(4) var          tLut  : texture_2d<f32>;

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_lut(@builtin(vertex_index) vi: u32) -> FSOut {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32( vi         & 2u) * 2.0 - 1.0;
  var o: FSOut;
  o.pos = vec4f(x, y, 0.0, 1.0);
  o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return o;
}

fn _lutFetch(ri: f32, gi: f32, bi: f32, N: f32) -> vec3f {
  let invW = 1.0 / N;
  let invH = 1.0 / (N * N);
  let px   = (ri + 0.5) * invW;
  let py   = (bi * N + gi + 0.5) * invH;
  return textureSampleLevel(tLut, sLut, vec2f(px, py), 0.0).rgb;
}

fn sampleLut3D(rgb: vec3f, N: f32) -> vec3f {
  let Nm1  = N - 1.0;
  let c    = clamp(rgb, vec3f(0.0), vec3f(1.0)) * Nm1;
  let lo   = floor(c);
  let hi   = min(lo + vec3f(1.0), vec3f(Nm1));
  let frac = c - lo;

  let c000 = _lutFetch(lo.r, lo.g, lo.b, N);
  let c100 = _lutFetch(hi.r, lo.g, lo.b, N);
  let c010 = _lutFetch(lo.r, hi.g, lo.b, N);
  let c110 = _lutFetch(hi.r, hi.g, lo.b, N);
  let c001 = _lutFetch(lo.r, lo.g, hi.b, N);
  let c101 = _lutFetch(hi.r, lo.g, hi.b, N);
  let c011 = _lutFetch(lo.r, hi.g, hi.b, N);
  let c111 = _lutFetch(hi.r, hi.g, hi.b, N);

  let c00  = mix(c000, c100, frac.r);
  let c10  = mix(c010, c110, frac.r);
  let c01  = mix(c001, c101, frac.r);
  let c11  = mix(c011, c111, frac.r);
  let c0   = mix(c00, c10, frac.g);
  let c1   = mix(c01, c11, frac.g);
  return mix(c0, c1, frac.b);
}

@fragment fn fs_lut(in: FSOut) -> @location(0) vec4f {
  let sceneColor = textureSample(tScene, sScene, in.uv);
  let graded     = sampleLut3D(sceneColor.rgb, u.lutSize);
  let finalColor = mix(sceneColor.rgb, graded, u.strength);
  return vec4f(finalColor, sceneColor.a);
}
`;

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
  readonly x:       Float32Array;
  readonly y:       Float32Array;
  readonly vx:      Float32Array;
  readonly vy:      Float32Array;
  readonly species: Uint32Array;
  readonly count:   number;
}

/** Internal bookkeeping for a registered cell's rendering resources. */
interface CellEntry {
  cellId:      string;
  species:     string;
  bbox:        CellBBox;
  shaderCfg:   SpeciesShaderConfig;
  materialDef: SpeciesMaterialDef | null;
  pbrMat:      ATPBRMaterial  | null;
  matcapMat:   ATMatcapFresnel | null;
}

/** Scene matrices packed for uniform buffer (viewProj + modelMat + eye). */
export interface SceneMatrices {
  viewProj: Float32Array;          // 16 f32
  modelMat: Float32Array;          // 16 f32
  eye:      [number, number, number];
}

/** Per-pass enable flags — all 13 passes individually toggleable. */
export interface RenderPassFlags {
  /** Pass 0: Environment background FX. Default true. */
  environmentFx:      boolean;
  /** Pass 1: Navier-Stokes fluid simulation. Default true. */
  navierStokes:       boolean;
  /** Pass 2a: Flower particle compute. Default true. */
  flowerParticle:     boolean;
  /** Pass 2b: Spline particle compute. Default true. */
  splineParticle:     boolean;
  /** Pass 3: Per-cell PBR/matcap material rendering. Default true. */
  cellMaterials:      boolean;
  /** Pass 6: Particle depth-sort compositor. Default true. */
  particleCompositor: boolean;
  /** Pass 7: Water surface. Default true. */
  waterSurface:       boolean;
  /** Pass 8: Volumetric light god rays. Default true. */
  volumetricLight:    boolean;
  /** Pass 9: Atmosphere (Rayleigh/Mie scatter + fog). Default true. */
  atmosphere:         boolean;
  /** Pass 10: Bloom post-process. Default true. */
  bloom:              boolean;
  /** Pass 11: Post-process (Kuwahara/Edge/Ink). Default false. */
  postProcess:        boolean;
  /** Pass 12: LUT colour grade. Default true. */
  lut:                boolean;
}

const DEFAULT_PASS_FLAGS: RenderPassFlags = {
  environmentFx:      true,
  navierStokes:       true,
  flowerParticle:     true,
  splineParticle:     true,
  cellMaterials:      true,
  particleCompositor: true,
  waterSurface:       true,
  volumetricLight:    true,
  atmosphere:         true,
  bloom:              true,
  postProcess:        false,   // artistic; opt-in
  lut:                true,
};

/** Full configuration for the RenderCompositor. All fields optional. */
export interface RenderCompositorConfig {
  /** Texture format for FBOs and swap-chain. Default 'bgra8unorm'. */
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

  /** AtmospherePass parameter overrides. */
  atmosphereParams?: AtmosphereParams;

  /** EnvironmentFx configuration overrides. */
  envFxConfig?: EnvironmentFxConfig;

  /** PostProcessPipeline style configuration. */
  postStyle?: PostProcessStyle;

  /** PostProcessPipeline parameter overrides. */
  postParams?: PostProcessParams;

  /** Initial PBR material params (geometry pass). */
  pbrParams?: Partial<PBRParams>;

  /** LUT colour grade style preset. */
  lutStyle?: LutStyleName;

  /** LUT grade blend strength [0=bypass, 1=full grade]. Default 0.85. */
  lutStrength?: number;

  /** Per-pass enable flags. */
  passes?: Partial<RenderPassFlags>;
}

/** Per-frame GPU timing data (in milliseconds). */
export interface FrameTimings {
  environmentFx:      number;
  navierStokes:       number;
  particleCompute:    number;
  cellMaterials:      number;
  particleRender:     number;
  particleCompositor: number;
  waterSurface:       number;
  volumetricLight:    number;
  atmosphere:         number;
  bloom:              number;
  postProcess:        number;
  lut:                number;
  total:              number;
}

// ─────────────────────────────────────────────────────────────────────────────
// FBO helper
// ─────────────────────────────────────────────────────────────────────────────

interface FBO {
  color:     GPUTexture;
  colorView: GPUTextureView;
  depth:     GPUTexture;
  depthView: GPUTextureView;
}

function createFBO(
  device: GPUDevice,
  w:      number,
  h:      number,
  format: GPUTextureFormat,
  label:  string,
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
    colorView: color.createView({ label: `${label}-color-view` }),
    depth,
    depthView: depth.createView({ label: `${label}-depth-view` }),
  };
}

function destroyFBO(fbo: FBO): void {
  fbo.color.destroy();
  fbo.depth.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

function makeDefaultSceneMatrices(): SceneMatrices {
  const id = new Float32Array([
    1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
  ]);
  return {
    viewProj: new Float32Array(id),
    modelMat: new Float32Array(id),
    eye:      [0, 0, 10],
  };
}

function packSceneMatrices(m: SceneMatrices): Float32Array {
  const buf = new Float32Array(36);
  buf.set(m.viewProj, 0);
  buf.set(m.modelMat, 16);
  buf[32] = m.eye[0];
  buf[33] = m.eye[1];
  buf[34] = m.eye[2];
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderCompositor
// ─────────────────────────────────────────────────────────────────────────────

export class RenderCompositor {

  // ── Core WebGPU ──────────────────────────────────────────────────────────
  private device!: GPUDevice;
  private canvas!: HTMLCanvasElement;
  private ctx!:    GPUCanvasContext;
  private format:  GPUTextureFormat = 'bgra8unorm';

  // ── Dimensions ───────────────────────────────────────────────────────────
  private width  = 0;
  private height = 0;

  // ── Sub-systems: rendering modules ───────────────────────────────────────
  private envFx!:         EnvironmentFx;
  private nsFluid!:       NavierStokesFluid;
  private flower!:        ATFlowerParticleRenderer;
  private spline!:        ATSplineParticleLife;
  private particleComp!:  ParticleCompositor;
  private water!:         ATWaterSurface;
  private vlight!:        ATVolumetricLight;
  private atmo!:          AtmospherePass;
  private bloom!:         ATBloomPostProcess;
  private postProcess!:   PostProcessPipeline;

  // ── LUT pass (inline pipeline) ──────────────────────────────────────────
  private lutPipeline!:   GPURenderPipeline;
  private lutBGL!:        GPUBindGroupLayout;
  private lutSampler!:    GPUSampler;
  private lutUniformBuf!: GPUBuffer;
  private lutTexture!:    GPUTexture;
  private lutView!:       GPUTextureView;
  private lutStyle:       LutStyleName = 'DEEP_OCEAN';
  private lutStrength     = 0.85;

  // ── Cell registry ────────────────────────────────────────────────────────
  private cells: Map<string, CellEntry> = new Map();

  // ── Intermediate FBOs ────────────────────────────────────────────────────
  //
  //   Frame chain:
  //     envFBO  → sceneFBO (geo + particles + particle-composite)
  //       → waterFBO → vlFBO → atmoFBO → bloomFBO → postFBO
  //         → LUT pass → swap-chain
  //
  private envFBO!:   FBO;   // Pass 0: environment background
  private sceneFBO!: FBO;   // Pass 3–6: geometry + particles + composite
  private waterFBO!: FBO;   // Pass 7: water surface overlay
  private vlFBO!:    FBO;   // Pass 8: volumetric light
  private atmoFBO!:  FBO;   // Pass 9: atmosphere
  private bloomFBO!: FBO;   // Pass 10: bloom
  private postFBO!:  FBO;   // Pass 11: post-process

  // ── Scene uniform buffer (shared by water surface + geometry) ───────────
  private sceneUniformBuf!: GPUBuffer;
  private sceneMatrices:    SceneMatrices = makeDefaultSceneMatrices();

  // ── Param caches (for resize recreation of immutable sub-systems) ──────
  private vlParamsCache:   ATVolumetricLightParams = {};
  private bloomParamsCache: ATBloomParams = {};
  private atmoParamsCache: AtmosphereParams = {};

  // ── Configuration / flags ──────────────────────────────────────────────
  private passes: RenderPassFlags = { ...DEFAULT_PASS_FLAGS };
  private elapsed = 0;

  // ── Lifecycle state ────────────────────────────────────────────────────
  private initialised = false;
  private destroyed   = false;

  // ── Interaction queue ──────────────────────────────────────────────────
  private pendingSplats: NavierStokesSplat[] = [];

  // ── GPU timing ─────────────────────────────────────────────────────────
  private timingQuerySet:  GPUQuerySet | null  = null;
  private timingBuffer:    GPUBuffer   | null  = null;
  private timingReadback:  GPUBuffer   | null  = null;
  private timingEnabled    = false;
  private lastTimings:     FrameTimings | null = null;

  // ═══════════════════════════════════════════════════════════════════════
  // init(device, canvas, config?)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Initialise all 13 rendering sub-systems and allocate 7 intermediate FBOs.
   *
   * Must be called exactly once.  The caller provides an already-initialised
   * GPUDevice and a canvas element whose dimensions determine the initial
   * FBO resolution.
   */
  async init(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    cfg:    RenderCompositorConfig = {},
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

    // ── Canvas context ──────────────────────────────────────────────────
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    this.ctx.configure({
      device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    const W = this.width;
    const H = this.height;

    // ── Intermediate FBOs (7 stages) ────────────────────────────────────
    this.envFBO   = createFBO(device, W, H, this.format, 'rc-env');
    this.sceneFBO = createFBO(device, W, H, this.format, 'rc-scene');
    this.waterFBO = createFBO(device, W, H, this.format, 'rc-water');
    this.vlFBO    = createFBO(device, W, H, this.format, 'rc-vl');
    this.atmoFBO  = createFBO(device, W, H, this.format, 'rc-atmo');
    this.bloomFBO = createFBO(device, W, H, this.format, 'rc-bloom');
    this.postFBO  = createFBO(device, W, H, this.format, 'rc-post');

    // ── Pass 0: Environment FX ──────────────────────────────────────────
    this.envFx = await EnvironmentFx.create(device, this.format, W, H);
    if (cfg.envFxConfig) {
      this.envFx.setConfig(cfg.envFxConfig);
    }

    // ── Pass 1: Navier-Stokes fluid ─────────────────────────────────────
    this.nsFluid = new NavierStokesFluid(device);

    // ── Pass 2: Particle systems ────────────────────────────────────────
    const flowerEdges = cfg.flowerEdges ?? [];
    const splineEdges = cfg.splineEdges ?? [];

    this.flower = new ATFlowerParticleRenderer(device, canvas, flowerEdges, {});
    await this.flower.build();

    this.spline = new ATSplineParticleLife(device, canvas, splineEdges, {});
    await this.spline.build();

    // ── Pass 6: Particle compositor ─────────────────────────────────────
    this.particleComp = new ParticleCompositor(device, canvas);
    await this.particleComp.build();

    // ── Pass 7: Water surface ───────────────────────────────────────────
    this.water = new ATWaterSurface(device, this.format, cfg.waterConfig ?? {});
    await this.water.build();

    // ── Scene uniform buffer ────────────────────────────────────────────
    this.sceneUniformBuf = device.createBuffer({
      label: 'rc-scene-uniforms',
      size:  144,   // 36 × f32 = 144 bytes  (viewProj + modelMat + eye)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.sceneUniformBuf, 0, packSceneMatrices(this.sceneMatrices),
    );

    // ── Pass 8: Volumetric light ────────────────────────────────────────
    this.vlight = await ATVolumetricLight.create(device, this.format, W, H);
    if (cfg.vlParams) {
      Object.assign(this.vlParamsCache, cfg.vlParams);
      this.vlight.setParams(cfg.vlParams);
    }

    // ── Pass 9: Atmosphere ──────────────────────────────────────────────
    this.atmo = await AtmospherePass.create(device, this.format, W, H);
    if (cfg.atmosphereParams) {
      Object.assign(this.atmoParamsCache, cfg.atmosphereParams);
      this.atmo.setParams(cfg.atmosphereParams);
    }

    // ── Pass 10: Bloom ──────────────────────────────────────────────────
    this.bloom = await ATBloomPostProcess.create(device, this.format, W, H);
    if (cfg.bloomParams) {
      Object.assign(this.bloomParamsCache, cfg.bloomParams);
      this.bloom.setParams(cfg.bloomParams);
    }

    // ── Pass 11: Post-process ───────────────────────────────────────────
    this.postProcess = await PostProcessPipeline.create(device, this.format, W, H);
    if (cfg.postStyle)  this.postProcess.setStyle(cfg.postStyle);
    if (cfg.postParams) this.postProcess.setParams(cfg.postParams);

    // ── Pass 12: LUT grade ──────────────────────────────────────────────
    this.lutStyle    = cfg.lutStyle    ?? 'DEEP_OCEAN';
    this.lutStrength = cfg.lutStrength ?? 0.85;
    await this._buildLutPipeline();
    this._uploadLut(this.lutStyle);

    // ── GPU timing (optional, if timestamp-query is supported) ──────────
    this._initTimingQueries();

    this.initialised = true;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Cell management
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Register a cell and create its per-species rendering material.
   *
   * Species is looked up in:
   *   1. species-shader-registry  → SpeciesShaderConfig
   *   2. cell-material-system     → SpeciesMaterialDef
   */
  async addCell(
    cellId:  string,
    species: string,
    bbox:    CellBBox,
  ): Promise<void> {
    if (this.destroyed || !this.initialised) return;
    if (this.cells.has(cellId)) return;

    let shaderCfg: SpeciesShaderConfig;
    try {
      shaderCfg = getSpeciesShaderConfig(species);
    } catch {
      shaderCfg = getSpeciesShaderConfig('cil-eye');
    }

    let materialDef: SpeciesMaterialDef | null = null;
    try {
      materialDef = getCellMaterial(species as CellSpecies);
    } catch {
      // not a known CellSpecies — use shader registry params
    }

    let pbrMat:    ATPBRMaterial  | null = null;
    let matcapMat: ATMatcapFresnel | null = null;
    const matType = materialDef?.materialType ?? shaderCfg.materialType;

    if (matType === 'matcap') {
      matcapMat = await ATMatcapFresnel.create(this.device, this.format);
      if (materialDef?.matcapParams) {
        matcapMat.setParams(materialDef.matcapParams);
      }
    } else {
      pbrMat = await ATPBRMaterial.create(this.device, this.format);
      const pbrParams: Partial<PBRParams> =
        materialDef?.pbrParams ??
        shaderCfg.materialParams as Partial<PBRParams>;
      if (pbrParams) {
        pbrMat.setParams(pbrParams);
      }
    }

    this.cells.set(cellId, {
      cellId,
      species,
      bbox,
      shaderCfg,
      materialDef,
      pbrMat,
      matcapMat,
    });
  }

  /** Remove a cell and destroy its GPU material resources. */
  removeCell(cellId: string): void {
    const entry = this.cells.get(cellId);
    if (!entry) return;
    entry.pbrMat?.destroy();
    entry.matcapMat?.destroy();
    this.cells.delete(cellId);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // tick(dt, sphWorld?) — main per-frame entry point
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Advance all 13 rendering passes by one frame and present to the canvas.
   *
   * @param dt       — Delta time in seconds (e.g. 1/60).
   * @param sphWorld — Read-only view of SPH particle state.
   *                   Pass null to skip physics-driven updates.
   */
  tick(dt: number, sphWorld: SPHWorldView | null = null): void {
    if (this.destroyed || !this.initialised) return;

    this.elapsed += dt;

    const swapTex = this.ctx.getCurrentTexture();
    const dstView = swapTex.createView();

    const encoder = this.device.createCommandEncoder({
      label: 'rc-frame',
    });

    // ──────────────────────────────────────────────────────────────────
    // Phase 0: Physics → material modulation
    // ──────────────────────────────────────────────────────────────────

    if (sphWorld && sphWorld.count > 0) {
      this._updateCellMaterialsFromPhysics(sphWorld, dt);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 0: Environment Background
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.environmentFx) {
      this.envFx.tick(dt);
      this.envFx.render(encoder, this.envFBO.colorView);
    } else {
      this._clearFBO(encoder, this.envFBO);
    }

    // Copy envFBO → sceneFBO as the base layer
    this._copyTexture(encoder, this.envFBO.color, this.sceneFBO.color);

    // ──────────────────────────────────────────────────────────────────
    // Pass 1: Navier-Stokes Fluid Step (compute only)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.navierStokes) {
      if (sphWorld && sphWorld.count > 0) {
        this._injectNSSplats(encoder, sphWorld);
      }
      this.nsFluid.step(encoder);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 2: Particle Compute (GPGPU — no render output yet)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.flowerParticle && this.flower.isBuilt) {
      this.flower.update(encoder, this.elapsed, dt);
    }

    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.update(encoder, this.elapsed, dt);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 3: Geometry — Cell Materials (render into sceneFBO)
    // ──────────────────────────────────────────────────────────────────

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

    // ──────────────────────────────────────────────────────────────────
    // Pass 4: Flower Particles (render over sceneFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.flowerParticle && this.flower.isBuilt) {
      this.flower.render(
        encoder,
        this.sceneFBO.colorView,
        this.sceneFBO.depthView,
      );
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 5: Spline Particles (render over sceneFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.render(
        encoder,
        this.sceneFBO.colorView,
        this.sceneFBO.depthView,
      );
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 6: Particle Compositor (depth-sort + alpha + glow)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.particleCompositor) {
      this.particleComp.sort(encoder);
      this.particleComp.renderAlpha(encoder, this.sceneFBO.colorView);
      this.particleComp.renderGlow(encoder, this.sceneFBO.colorView);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 7: Water Surface (sceneFBO → waterFBO)
    // ──────────────────────────────────────────────────────────────────

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

    // ──────────────────────────────────────────────────────────────────
    // Pass 8: Volumetric Light (waterFBO → vlFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.volumetricLight) {
      this.vlight.tick(dt);
      this.vlight.render(
        encoder,
        this.waterFBO.color,
        this.vlFBO.colorView,
      );
    } else {
      this._copyTexture(encoder, this.waterFBO.color, this.vlFBO.color);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 9: Atmosphere (vlFBO → atmoFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.atmosphere) {
      this.atmo.render(
        encoder,
        this.vlFBO.color,
        this.atmoFBO.colorView,
      );
    } else {
      this._copyTexture(encoder, this.vlFBO.color, this.atmoFBO.color);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 10: Bloom (atmoFBO → bloomFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.bloom) {
      this.bloom.render(
        encoder,
        this.atmoFBO.color,
        this.bloomFBO.colorView,
      );
    } else {
      this._copyTexture(encoder, this.atmoFBO.color, this.bloomFBO.color);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 11: Post-Process (bloomFBO → postFBO)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.postProcess) {
      this.postProcess.render(
        encoder,
        this.bloomFBO.color.createView(),
        this.postFBO.colorView,
      );
    } else {
      this._copyTexture(encoder, this.bloomFBO.color, this.postFBO.color);
    }

    // ──────────────────────────────────────────────────────────────────
    // Pass 12: LUT Colour Grade (postFBO → swap-chain)
    // ──────────────────────────────────────────────────────────────────

    if (this.passes.lut) {
      this._renderLutPass(encoder, this.postFBO.color, dstView);
    } else {
      // Direct blit — reuse LUT pipeline with strength=0
      this._blitToSwapChain(encoder, this.postFBO.color, dstView);
    }

    // ──────────────────────────────────────────────────────────────────
    // Submit
    // ──────────────────────────────────────────────────────────────────

    this.device.queue.submit([encoder.finish()]);

    // Async spline handoff readback (non-blocking)
    if (this.passes.splineParticle && this.spline.isBuilt) {
      this.spline.scheduleHandoffReadback().catch(() => {});
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // resize(w, h)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Handle canvas resize.  Reallocates all 7 intermediate FBOs and notifies
   * sub-systems that need resolution-dependent resources.
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

    // Destroy old FBOs
    destroyFBO(this.envFBO);
    destroyFBO(this.sceneFBO);
    destroyFBO(this.waterFBO);
    destroyFBO(this.vlFBO);
    destroyFBO(this.atmoFBO);
    destroyFBO(this.bloomFBO);
    destroyFBO(this.postFBO);

    // Reallocate
    this.envFBO   = createFBO(this.device, w, h, this.format, 'rc-env');
    this.sceneFBO = createFBO(this.device, w, h, this.format, 'rc-scene');
    this.waterFBO = createFBO(this.device, w, h, this.format, 'rc-water');
    this.vlFBO    = createFBO(this.device, w, h, this.format, 'rc-vl');
    this.atmoFBO  = createFBO(this.device, w, h, this.format, 'rc-atmo');
    this.bloomFBO = createFBO(this.device, w, h, this.format, 'rc-bloom');
    this.postFBO  = createFBO(this.device, w, h, this.format, 'rc-post');

    // Sub-systems that hold resolution-dependent textures
    this._recreateVolumetricLight(w, h);
    this._recreateAtmosphere(w, h);
    this._recreateBloom(w, h);
    this._recreatePostProcess(w, h);
    this._recreateEnvironmentFx(w, h);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Configuration API
  // ═══════════════════════════════════════════════════════════════════════

  /** Enable/disable individual rendering passes. */
  setPassFlags(flags: Partial<RenderPassFlags>): void {
    Object.assign(this.passes, flags);
  }

  /** Get a snapshot of the current pass flags. */
  getPassFlags(): Readonly<RenderPassFlags> {
    return { ...this.passes };
  }

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

  /** Update atmosphere parameters. */
  setAtmosphereParams(p: AtmosphereParams): void {
    Object.assign(this.atmoParamsCache, p);
    this.atmo?.setParams(p);
  }

  /** Update environment FX configuration. */
  setEnvironmentFxConfig(cfg: EnvironmentFxConfig): void {
    this.envFx?.setConfig(cfg);
  }

  /** Update post-process style. */
  setPostProcessStyle(s: PostProcessStyle): void {
    this.postProcess?.setStyle(s);
  }

  /** Update post-process parameters. */
  setPostProcessParams(p: PostProcessParams): void {
    this.postProcess?.setParams(p);
  }

  /**
   * Update the scene view/projection matrices and camera eye position.
   * Forwarded to the water surface renderPass and any geometry that needs them.
   */
  setSceneMatrices(m: Partial<SceneMatrices>): void {
    if (m.viewProj) this.sceneMatrices.viewProj = m.viewProj;
    if (m.modelMat) this.sceneMatrices.modelMat = m.modelMat;
    if (m.eye)      this.sceneMatrices.eye      = m.eye;
    this.device.queue.writeBuffer(
      this.sceneUniformBuf, 0, packSceneMatrices(this.sceneMatrices),
    );
  }

  /**
   * Swap to a different LUT grade preset.
   * Rebuilds the LUT texture on CPU and re-uploads to GPU.
   */
  setLutStyle(style: LutStyleName, strength?: number): void {
    this.lutStyle = style;
    if (strength !== undefined) this.lutStrength = strength;
    this._uploadLut(style);
    this._writeLutUniforms();
  }

  /** Update the LUT blend strength without changing the style. */
  setLutStrength(strength: number): void {
    this.lutStrength = Math.max(0, Math.min(1, strength));
    this._writeLutUniforms();
  }

  /**
   * Inject a Navier-Stokes splat (e.g. from mouse/touch interaction).
   * The splat is queued and applied on the next tick().
   */
  queueSplat(splat: NavierStokesSplat): void {
    this.pendingSplats.push(splat);
  }

  /** Add a drop to the water surface. */
  addWaterDrop(x: number, y: number, radius: number, strength: number): void {
    this.water?.addDrop(x, y, radius, strength);
  }

  /** Replace flower particle edge splines at runtime. */
  async setFlowerEdges(edges: FlowerEdgeSpline[]): Promise<void> {
    if (!this.initialised) return;
    await this.flower.setEdges(edges);
  }

  /** Replace spline particle edge splines at runtime. */
  async setSplineEdges(edges: EdgeSpline[]): Promise<void> {
    if (!this.initialised) return;
    await this.spline.setEdges(edges);
  }

  /** Get last-frame GPU timings (null if timing not supported). */
  getTimings(): FrameTimings | null {
    return this.lastTimings;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Destroy
  // ═══════════════════════════════════════════════════════════════════════

  /** Release all GPU resources.  The compositor must not be used after this. */
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
    this.envFx?.destroy();
    this.flower?.destroy();
    this.spline?.destroy();
    this.water?.destroy();
    this.nsFluid?.destroy();
    this.vlight?.destroy();
    this.atmo?.destroy();
    this.bloom?.destroy();
    this.postProcess?.destroy();
    this.particleComp?.destroy();

    // Destroy FBOs
    if (this.envFBO)   destroyFBO(this.envFBO);
    if (this.sceneFBO) destroyFBO(this.sceneFBO);
    if (this.waterFBO) destroyFBO(this.waterFBO);
    if (this.vlFBO)    destroyFBO(this.vlFBO);
    if (this.atmoFBO)  destroyFBO(this.atmoFBO);
    if (this.bloomFBO) destroyFBO(this.bloomFBO);
    if (this.postFBO)  destroyFBO(this.postFBO);

    // Destroy LUT resources
    this.lutTexture?.destroy();
    this.lutUniformBuf?.destroy();

    // Destroy shared buffers
    this.sceneUniformBuf?.destroy();

    // Destroy timing resources
    this.timingQuerySet?.destroy();
    this.timingBuffer?.destroy();
    this.timingReadback?.destroy();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Accessors
  // ═══════════════════════════════════════════════════════════════════════

  get isInitialised(): boolean { return this.initialised; }
  get isDestroyed():   boolean { return this.destroyed; }
  get cellCount():     number  { return this.cells.size; }
  get elapsedTime():   number  { return this.elapsed; }

  /** Iterate registered cell IDs. */
  cellIds(): IterableIterator<string> {
    return this.cells.keys();
  }

  /** Get the current cell entry (read-only). */
  getCell(cellId: string): Readonly<CellEntry> | undefined {
    return this.cells.get(cellId);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private — FBO operations
  // ═══════════════════════════════════════════════════════════════════════

  /** Clear an FBO to transparent black with a fresh depth buffer. */
  private _clearFBO(encoder: GPUCommandEncoder, fbo: FBO): void {
    const pass = encoder.beginRenderPass({
      label: 'rc-clear',
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

  /** GPU texture-to-texture copy (same size). */
  private _copyTexture(
    encoder: GPUCommandEncoder,
    src:     GPUTexture,
    dst:     GPUTexture,
  ): void {
    encoder.copyTextureToTexture(
      { texture: src },
      { texture: dst },
      [this.width, this.height],
    );
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private — Physics-driven material updates
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Derive per-cell aggregate physics and modulate material params.
   *
   * For each cell, scan SPH particles within its bbox and compute:
   *   - Average kinetic energy → roughness / iridescence modulation
   *   - Density → Fresnel power modulation
   */
  private _updateCellMaterialsFromPhysics(
    sph: SPHWorldView,
    _dt: number,
  ): void {
    for (const entry of this.cells.values()) {
      const { bbox, pbrMat, matcapMat } = entry;
      if (!pbrMat && !matcapMat) continue;

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

      const avgKE  = Math.min(1.0, (sumVelSq / count) * 2.0);
      const area   = Math.max(bbox.w * bbox.h, 0.001);
      const density = Math.min(1.0, count / (area * 500));

      if (pbrMat) {
        pbrMat.setParams({
          roughness:    0.15 + avgKE * 0.45,
          iridStrength: 0.3 + avgKE * 0.7,
        } as Partial<PBRParams>);
      }

      if (matcapMat) {
        matcapMat.setParams({
          fresnelPower: 2.0 + density * 4.0,
        } as Partial<MatcapParams>);
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private — Navier-Stokes splat injection
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Inject NS splats from:
   *   a) queued user splats (mouse/touch)
   *   b) SPH cell-centre velocity impulses (automated coupling)
   */
  private _injectNSSplats(
    encoder: GPUCommandEncoder,
    sph:     SPHWorldView,
  ): void {
    // (a) User-queued splats
    for (const splat of this.pendingSplats) {
      this.nsFluid.splat(encoder, splat);
    }
    this.pendingSplats.length = 0;

    // (b) Automated: inject one dye splat per cell
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

      if (count < 5) continue;

      const avgVx = sumVx / count;
      const avgVy = sumVy / count;
      const speed = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
      if (speed < 0.01) continue;

      const cx = (bbox.x + bbox.w * 0.5) / (this.width || 1);
      const cy = (bbox.y + bbox.h * 0.5) / (this.height || 1);

      const mp  = entry.shaderCfg.materialParams;
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

  // ═══════════════════════════════════════════════════════════════════════
  // Private — LUT pass
  // ═══════════════════════════════════════════════════════════════════════

  /** Build the inline LUT render pipeline and GPU objects. */
  private async _buildLutPipeline(): Promise<void> {
    const mod = this.device.createShaderModule({
      label: 'rc-lut-module',
      code:  LUT_PASS_WGSL,
    });

    this.lutBGL = this.device.createBindGroupLayout({
      label:   'rc-lut-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          buffer:  { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    this.lutPipeline = await this.device.createRenderPipelineAsync({
      label:  'rc-lut-pipeline',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.lutBGL],
      }),
      vertex: {
        module:     mod,
        entryPoint: 'vs_lut',
      },
      fragment: {
        module:     mod,
        entryPoint: 'fs_lut',
        targets:    [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.lutSampler = this.device.createSampler({
      label:        'rc-lut-sampler',
      magFilter:    'nearest',
      minFilter:    'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.lutUniformBuf = this.device.createBuffer({
      label: 'rc-lut-uniforms',
      size:  16,    // 4 × f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._writeLutUniforms();
  }

  /** Upload a LUT texture for the given style. */
  private _uploadLut(style: LutStyleName): void {
    if (this.lutTexture) {
      this.lutTexture.destroy();
    }
    const gen  = new LutGenerator({ lutSize: 17 });
    const cube = gen.buildLut(style);
    const { texture, view } = gen.uploadToWebGPU(this.device, cube);
    this.lutTexture = texture;
    this.lutView    = view;
  }

  /** Write the LUT uniform buffer with current lutSize + strength. */
  private _writeLutUniforms(): void {
    this.device.queue.writeBuffer(
      this.lutUniformBuf, 0,
      new Float32Array([17, this.lutStrength, 0, 0]),
    );
  }

  /** Record the LUT grade render pass: src texture → dstView. */
  private _renderLutPass(
    encoder: GPUCommandEncoder,
    src:     GPUTexture,
    dstView: GPUTextureView,
  ): void {
    const sceneSampler = this.device.createSampler({
      magFilter:    'linear',
      minFilter:    'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const bg = this.device.createBindGroup({
      label:   'rc-lut-bg',
      layout:  this.lutBGL,
      entries: [
        { binding: 0, resource: { buffer: this.lutUniformBuf } },
        { binding: 1, resource: sceneSampler },
        { binding: 2, resource: src.createView() },
        { binding: 3, resource: this.lutSampler },
        { binding: 4, resource: this.lutView },
      ],
    });

    const pass = encoder.beginRenderPass({
      label:            'rc-lut-pass',
      colorAttachments: [{
        view:       dstView,
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.lutPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);   // full-screen triangle
    pass.end();
  }

  /**
   * Blit src texture to swap-chain view via the LUT pipeline with strength=0.
   * Used when LUT pass is disabled but we still need to present.
   */
  private _blitToSwapChain(
    encoder: GPUCommandEncoder,
    src:     GPUTexture,
    dstView: GPUTextureView,
  ): void {
    const savedStrength = this.lutStrength;
    this.device.queue.writeBuffer(
      this.lutUniformBuf, 0,
      new Float32Array([17, 0.0, 0, 0]),
    );
    this._renderLutPass(encoder, src, dstView);
    this.device.queue.writeBuffer(
      this.lutUniformBuf, 0,
      new Float32Array([17, savedStrength, 0, 0]),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private — Resize sub-system recreation
  // ═══════════════════════════════════════════════════════════════════════

  private async _recreateVolumetricLight(w: number, h: number): Promise<void> {
    this.vlight.destroy();
    this.vlight = await ATVolumetricLight.create(this.device, this.format, w, h);
    if (Object.keys(this.vlParamsCache).length > 0) {
      this.vlight.setParams(this.vlParamsCache);
    }
  }

  private async _recreateAtmosphere(w: number, h: number): Promise<void> {
    this.atmo.destroy();
    this.atmo = await AtmospherePass.create(this.device, this.format, w, h);
    if (Object.keys(this.atmoParamsCache).length > 0) {
      this.atmo.setParams(this.atmoParamsCache);
    }
  }

  private async _recreateBloom(w: number, h: number): Promise<void> {
    this.bloom.destroy();
    this.bloom = await ATBloomPostProcess.create(this.device, this.format, w, h);
    if (Object.keys(this.bloomParamsCache).length > 0) {
      this.bloom.setParams(this.bloomParamsCache);
    }
  }

  private async _recreatePostProcess(w: number, h: number): Promise<void> {
    this.postProcess.destroy();
    this.postProcess = await PostProcessPipeline.create(
      this.device, this.format, w, h,
    );
  }

  private async _recreateEnvironmentFx(w: number, h: number): Promise<void> {
    this.envFx.destroy();
    this.envFx = await EnvironmentFx.create(this.device, this.format, w, h);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Private — GPU timing queries
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * Attempt to create GPU timestamp query resources.
   * Falls back gracefully if the device doesn't support 'timestamp-query'.
   */
  private _initTimingQueries(): void {
    if (!this.device.features.has('timestamp-query')) {
      this.timingEnabled = false;
      return;
    }

    try {
      // 13 passes × 2 (begin + end) = 26 timestamps
      this.timingQuerySet = this.device.createQuerySet({
        type:  'timestamp',
        count: 26,
      });

      this.timingBuffer = this.device.createBuffer({
        label: 'rc-timing-buf',
        size:  26 * 8,    // 26 × u64
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      this.timingReadback = this.device.createBuffer({
        label: 'rc-timing-readback',
        size:  26 * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      this.timingEnabled = true;
    } catch {
      this.timingEnabled = false;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper — create a fully-initialised RenderCompositor in one call
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and initialise a RenderCompositor in a single async call.
 *
 * ```ts
 * const comp = await createRenderCompositor(device, canvas, {
 *   lutStyle:    'CINEMATIC_WARM',
 *   passes:      { postProcess: true },
 *   postStyle:   { kuwahara: true, edge: false, ink: false },
 * });
 * ```
 */
export async function createRenderCompositor(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  cfg:    RenderCompositorConfig = {},
): Promise<RenderCompositor> {
  const comp = new RenderCompositor();
  await comp.init(device, canvas, cfg);
  return comp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for consumer convenience
// ─────────────────────────────────────────────────────────────────────────────

export type {
  PBRParams,
  MatcapParams,
  FlowerEdgeSpline,
  EdgeSpline,
  ATWaterSurfaceConfig,
  ATBloomParams,
  ATVolumetricLightParams,
  AtmosphereParams,
  EnvironmentFxConfig,
  PostProcessStyle,
  PostProcessParams,
  NavierStokesSplat,
  LutStyleName,
  SpeciesShaderConfig,
  SpeciesMaterialDef,
  CellSpecies,
};
