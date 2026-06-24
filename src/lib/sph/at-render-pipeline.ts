/**
 * at-render-pipeline.ts — AT Render Pipeline Orchestrator
 *
 * Chains every AT rendering module in a fixed-function forward pipeline,
 * passing each pass's FBO output as the next pass's input texture:
 *
 *   ┌─── Pass 0 ── Geometry (clear → color+depth FBO) ─────────────────────┐
 *   │     PBR Cell Material (Cook-Torrance BRDF + iridescence + Fresnel)    │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ geoFBO (color: TEXTURE_BINDING)
 *                 ▼
 *   ┌─── Pass 1 ── Particles (additive composite over geoFBO) ─────────────┐
 *   │     ATFlowerParticleRenderer  (spiral GPGPU + matcap shading)         │
 *   │     ATSplineParticleLife      (Catmull-Rom + curl-noise lifecycle)     │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ particleFBO
 *                 ▼
 *   ┌─── Pass 2 ── Water Surface (mesh + particle splash) ─────────────────┐
 *   │     ATWaterSurface  (Navier-Stokes waves + water-particle render)     │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ waterFBO
 *                 ▼
 *   ┌─── Pass 3 ── Volumetric Light (god-rays, screen-space radial) ────────┐
 *   │     ATVolumetricLight  (occlusion mask → radial scatter → composite)  │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ vlFBO
 *                 ▼
 *   ┌─── Pass 4 ── Bloom (UE5-style threshold + separable gaussian) ────────┐
 *   │     ATBloomPostProcess  (bright extract → blur chain → composite)     │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ bloomFBO
 *                 ▼
 *   ┌─── Pass 5 ── LUT Grade (3-D colour look-up table) ───────────────────┐
 *   │     LutGenerator.uploadToWebGPU()  (inline WGSL full-screen blit)     │
 *   └───────────────────────────────────────────────────────────────────────┘
 *                 │ → swap-chain surface (dstView)
 *
 * Usage:
 *   const pipe = await ATRenderPipeline.create(device, canvas, format, edges);
 *   // per-frame:
 *   pipe.tick(dt);
 *   const enc = device.createCommandEncoder();
 *   pipe.render(enc, swapChainView);
 *   device.queue.submit([enc.finish()]);
 *
 * Research: xiaodi #M720 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────







import {
} from './at-pbr-material';

} from './at-flower-particle';
} from './at-spline-particle';
} from './at-water-surface';
} from './at-volumetric-light';
} from './at-bloom-postprocess';
} from './lut-generator';

  ATPBRMaterial,
  type PBRParams,
  DEFAULT_PBR_PARAMS,

  ATFlowerParticleRenderer,
  type FlowerEdgeSpline,
  type ATFlowerConfig,

  ATSplineParticleLife,
  type EdgeSpline,
  type ATSplineParticleConfig,

  ATWaterSurface,
  type ATWaterSurfaceConfig,

  ATVolumetricLight,
  type ATVolumetricLightParams,

  ATBloomPostProcess,
  type ATBloomParams,

  LutGenerator,
  type LutStyleName,
  type LutCube,

// ─────────────────────────────────────────────────────────────────────────────
// LUT pass — inline WGSL (full-screen 3-D LUT grade blit)
// The LUT is stored as a horizontal-strip 2-D texture (size × size²).
// ─────────────────────────────────────────────────────────────────────────────

const LUT_PASS_WGSL = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────
struct LutUniforms {
  lutSize     : f32,    // N — one side of the cube (e.g. 17)
  strength    : f32,    // blend [0,1]: 0 = bypass, 1 = full grade
  _pad0       : f32,
  _pad1       : f32,
}

@group(0) @binding(0) var<uniform> u     : LutUniforms;
@group(0) @binding(1) var          sScene: sampler;
@group(0) @binding(2) var          tScene: texture_2d<f32>;   // scene input
@group(0) @binding(3) var          sLut  : sampler;
@group(0) @binding(4) var          tLut  : texture_2d<f32>;   // N × N² strip

// ── Full-screen vertex ────────────────────────────────────────────────────────
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

// ── 3-D LUT fetch helper (strip layout, top-level — WGSL has no nested fns) ──
// ri/gi/bi are integer indices; N is the cube side length.
fn _lutFetch(ri: f32, gi: f32, bi: f32, N: f32) -> vec3f {
  let invW = 1.0 / N;
  let invH = 1.0 / (N * N);
  let px   = (ri + 0.5) * invW;
  let py   = (bi * N + gi + 0.5) * invH;
  return textureSampleLevel(tLut, sLut, vec2f(px, py), 0.0).rgb;
}

// ── 3-D LUT trilinear sample (horizontal-strip layout) ───────────────────────
// The strip texture has dimensions [N, N*N].
// For a given (r,g,b) ∈ [0,1]³:
//   b slice index  bLo = floor(b*(N-1))
//   pixel column   = r_frac * (N-1)  → strip x
//   pixel row      = bLo*N + g_frac*(N-1) → strip y (for bLo slice)
fn sampleLut3D(rgb: vec3f, N: f32) -> vec3f {
  let Nm1   = N - 1.0;

  // Coordinate within [0, N-1]³
  let c    = clamp(rgb, vec3f(0.0), vec3f(1.0)) * Nm1;
  let lo   = floor(c);
  let hi   = min(lo + vec3f(1.0), vec3f(Nm1));
  let frac = c - lo;

  // Trilinear interpolation across 8 lattice corners
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

// ── Fragment ──────────────────────────────────────────────────────────────────
@fragment fn fs_lut(in: FSOut) -> @location(0) vec4f {
  let sceneColor = textureSample(tScene, sScene, in.uv);
  let graded     = sampleLut3D(sceneColor.rgb, u.lutSize);
  let finalColor = mix(sceneColor.rgb, graded, u.strength);
  return vec4f(finalColor, sceneColor.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Scene uniform buffer layout (shared by water surface + any geometry pass
// that needs view/projection matrices).
//   offset  0:  viewProj   mat4x4f  (64 bytes)
//   offset 64:  modelMat   mat4x4f  (64 bytes)
//   offset 128: eye.xyz    + pad    (16 bytes)
//   total: 144 bytes
// ─────────────────────────────────────────────────────────────────────────────

/** Scene matrices packed as Float32Array(36) aligned to 16 bytes. */
export interface SceneMatrices {
  /** Column-major 4×4 view-projection matrix. */
  viewProj : Float32Array;   // 16 f32
  /** Column-major 4×4 model matrix. */
  modelMat : Float32Array;   // 16 f32
  /** World-space camera eye position. */
  eye      : [number, number, number];
}

/** Make a default identity scene uniform buffer content. */
function makeDefaultSceneMatrices(): SceneMatrices {
  // identity mat4
  const id = new Float32Array([
    1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1,
  ]);
  return {
    viewProj : new Float32Array(id),
    modelMat : new Float32Array(id),
    eye      : [0, 0, 10],
  };
}

function packSceneMatrices(m: SceneMatrices): Float32Array {
  const buf = new Float32Array(36);
  buf.set(m.viewProj, 0);
  buf.set(m.modelMat, 16);
  buf[32] = m.eye[0];
  buf[33] = m.eye[1];
  buf[34] = m.eye[2];
  // buf[35] = pad
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Per-pass enable flags — set to false to skip a pass entirely. */
export interface ATPassFlags {
  geometry      : boolean;   // PBR material clear pass
  flowerParticle: boolean;   // AT FlowerParticleRenderer
  splineParticle: boolean;   // AT SplineParticleLife
  waterSurface  : boolean;   // AT WaterSurface
  volumetricLight: boolean;  // AT VolumetricLight god-rays
  bloom         : boolean;   // AT Bloom post-process
  lut           : boolean;   // 3-D LUT grade
}

/** Top-level configuration for the entire AT render pipeline. */
export interface ATRenderPipelineConfig {
  /** Texture format for all FBOs and the swap-chain surface. */
  format?         : GPUTextureFormat;
  /** LUT style preset. */
  lutStyle?       : LutStyleName;
  /** LUT grade blend strength [0=bypass, 1=full grade]. */
  lutStrength?    : number;
  /** Initial PBR material params (geometry pass). */
  pbrParams?      : Partial<PBRParams>;
  /** Initial ATFlowerParticleRenderer config. */
  flowerConfig?   : ATFlowerConfig;
  /** Initial ATSplineParticleLife config. */
  splineConfig?   : ATSplineParticleConfig;
  /** Initial ATWaterSurface config. */
  waterConfig?    : ATWaterSurfaceConfig;
  /** Initial ATVolumetricLight params. */
  vlParams?       : ATVolumetricLightParams;
  /** Initial ATBloom params. */
  bloomParams?    : ATBloomParams;
  /** Per-pass enable flags. Defaults: all enabled. */
  passes?         : Partial<ATPassFlags>;
}

const DEFAULT_PASS_FLAGS: ATPassFlags = {
  geometry       : true,
  flowerParticle : true,
  splineParticle : true,
  waterSurface   : true,
  volumetricLight: true,
  bloom          : true,
  lut            : true,
};

// ─────────────────────────────────────────────────────────────────────────────
// FBO helpers
// ─────────────────────────────────────────────────────────────────────────────

interface FBO {
  color: GPUTexture;
  depth: GPUTexture;
  colorView: GPUTextureView;
  depthView: GPUTextureView;
}

function createFBO(
  device: GPUDevice,
  width : number,
  height: number,
  format: GPUTextureFormat,
  label : string,
): FBO {
  const color = device.createTexture({
    label : `${label}-color`,
    size  : [width, height],
    format,
    usage : GPUTextureUsage.RENDER_ATTACHMENT
           | GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    label : `${label}-depth`,
    size  : [width, height],
    format: 'depth24plus',
    usage : GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return {
    color,
    depth,
    colorView: color.createView({ label: `${label}-color-view` }),
    depthView: depth.createView({ label: `${label}-depth-view` }),
  };
}

function destroyFBO(fbo: FBO): void {
  fbo.color.destroy();
  fbo.depth.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// ATRenderPipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATRenderPipeline — orchestrates all AT rendering passes in a fixed-function
 * forward pipeline, daisy-chaining FBO outputs through each stage.
 *
 * Lifetime:
 *   1. `ATRenderPipeline.create()` — allocates all GPU resources, builds all
 *      sub-pipelines, uploads the LUT texture.
 *   2. `pipe.tick(dt)` — advance all animation clocks (call once per frame).
 *   3. `pipe.render(encoder, dstView)` — record all render passes into
 *      the encoder; submit the encoder externally.
 *   4. `pipe.resize(w, h)` — reallocate FBOs on canvas resize.
 *   5. `pipe.destroy()` — release all GPU resources.
 */
export class ATRenderPipeline {
  // ── Core WebGPU ────────────────────────────────────────────────────────────
  private readonly device  : GPUDevice;
  private readonly canvas  : HTMLCanvasElement;
  private readonly format  : GPUTextureFormat;

  // ── Sub-passes ─────────────────────────────────────────────────────────────
  private pbr         : ATPBRMaterial;
  private flower      : ATFlowerParticleRenderer;
  private spline      : ATSplineParticleLife;
  private water       : ATWaterSurface;
  private vlight      : ATVolumetricLight;
  private bloom       : ATBloomPostProcess;

  // ── LUT pass (custom inline pipeline) ─────────────────────────────────────
  private lutPipeline : GPURenderPipeline;
  private lutBGL      : GPUBindGroupLayout;
  private lutSampler  : GPUSampler;
  private lutUniformBuf: GPUBuffer;
  private lutTexture  : GPUTexture;
  private lutView     : GPUTextureView;
  private lutStyle    : LutStyleName;
  private lutStrength : number;

  // ── Intermediate FBOs ─────────────────────────────────────────────────────
  //   geoFBO      → geometry (PBR) clear + material pass
  //   particleFBO → additive particle composites
  //   waterFBO    → water surface overlay
  //   vlFBO       → volumetric light scatter composite
  //   bloomFBO    → bloom composite (input for LUT)
  private geoFBO      : FBO;
  private particleFBO : FBO;
  private waterFBO    : FBO;
  private vlFBO       : FBO;
  private bloomFBO    : FBO;

  // ── Scene uniform buffer (view/proj/model/eye — water surface) ─────────────
  private sceneUniformBuf: GPUBuffer;
  private sceneMatrices  : SceneMatrices;

  // ── Pass flags & dims ─────────────────────────────────────────────────────
  private passes : ATPassFlags;
  private width  : number;
  private height : number;

  // ─────────────────────────────────────────────────────────────────────────
  // Private constructor — use ATRenderPipeline.create()
  // ─────────────────────────────────────────────────────────────────────────

  private constructor(
    device        : GPUDevice,
    canvas        : HTMLCanvasElement,
    format        : GPUTextureFormat,
    pbr           : ATPBRMaterial,
    flower        : ATFlowerParticleRenderer,
    spline        : ATSplineParticleLife,
    water         : ATWaterSurface,
    vlight        : ATVolumetricLight,
    bloom         : ATBloomPostProcess,
    lutPipeline   : GPURenderPipeline,
    lutBGL        : GPUBindGroupLayout,
    lutSampler    : GPUSampler,
    lutUniformBuf : GPUBuffer,
    lutTexture    : GPUTexture,
    lutView       : GPUTextureView,
    lutStyle      : LutStyleName,
    lutStrength   : number,
    geoFBO        : FBO,
    particleFBO   : FBO,
    waterFBO      : FBO,
    vlFBO         : FBO,
    bloomFBO      : FBO,
    sceneUniformBuf: GPUBuffer,
    passes        : ATPassFlags,
  ) {
    this.device          = device;
    this.canvas          = canvas;
    this.format          = format;
    this.pbr             = pbr;
    this.flower          = flower;
    this.spline          = spline;
    this.water           = water;
    this.vlight          = vlight;
    this.bloom           = bloom;
    this.lutPipeline     = lutPipeline;
    this.lutBGL          = lutBGL;
    this.lutSampler      = lutSampler;
    this.lutUniformBuf   = lutUniformBuf;
    this.lutTexture      = lutTexture;
    this.lutView         = lutView;
    this.lutStyle        = lutStyle;
    this.lutStrength     = lutStrength;
    this.geoFBO          = geoFBO;
    this.particleFBO     = particleFBO;
    this.waterFBO        = waterFBO;
    this.vlFBO           = vlFBO;
    this.bloomFBO        = bloomFBO;
    this.sceneUniformBuf = sceneUniformBuf;
    this.passes          = passes;
    this.width           = canvas.width;
    this.height          = canvas.height;
    this.sceneMatrices   = makeDefaultSceneMatrices();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Factory
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Asynchronously build the complete AT render pipeline.
   *
   * @param device        WebGPU device.
   * @param canvas        Target HTMLCanvasElement (provides width/height).
   * @param flowerEdges   Spline edges for ATFlowerParticleRenderer.
   * @param splineEdges   Spline edges for ATSplineParticleLife.
   * @param cfg           Optional per-pass configuration overrides.
   */
  static async create(
    device      : GPUDevice,
    canvas      : HTMLCanvasElement,
    flowerEdges : FlowerEdgeSpline[] = [],
    splineEdges : EdgeSpline[]       = [],
    cfg         : ATRenderPipelineConfig = {},
  ): Promise<ATRenderPipeline> {
    const format  : GPUTextureFormat = cfg.format      ?? 'bgra8unorm';
    const lutStyle: LutStyleName     = cfg.lutStyle    ?? 'DEEP_OCEAN';
    const lutStrength                = cfg.lutStrength ?? 0.85;
    const W = canvas.width;
    const H = canvas.height;

    const passes: ATPassFlags = {
      ...DEFAULT_PASS_FLAGS,
      ...(cfg.passes ?? {}),
    };

    // ── 0. Geometry pass: PBR material ──────────────────────────────────────
    const pbr = await ATPBRMaterial.create(device, format);
    if (cfg.pbrParams) pbr.setParams(cfg.pbrParams);

    // ── 1. Particles ─────────────────────────────────────────────────────────
    const flower = new ATFlowerParticleRenderer(
      device, canvas, flowerEdges, cfg.flowerConfig ?? {},
    );
    await flower.build();

    const spline = new ATSplineParticleLife(
      device, canvas, splineEdges, cfg.splineConfig ?? {},
    );
    await spline.build();

    // ── 2. Water surface ─────────────────────────────────────────────────────
    const water = new ATWaterSurface(device, format, cfg.waterConfig ?? {});
    await water.build();

    // ── 3. Volumetric light ───────────────────────────────────────────────────
    const vlight = await ATVolumetricLight.create(device, format, W, H);
    if (cfg.vlParams) vlight.setParams(cfg.vlParams);

    // ── 4. Bloom ──────────────────────────────────────────────────────────────
    const bloom = await ATBloomPostProcess.create(device, format, W, H);
    if (cfg.bloomParams) bloom.setParams(cfg.bloomParams);

    // ── 5. LUT pass ───────────────────────────────────────────────────────────
    const {
      lutPipeline, lutBGL, lutSampler, lutUniformBuf,
    } = await ATRenderPipeline._buildLutPipeline(device, format);

    // Upload initial LUT texture
    const lutGen = new LutGenerator({ lutSize: 17 });
    const lutCube = lutGen.buildLut(lutStyle);
    const { texture: lutTexture, view: lutView } =
      lutGen.uploadToWebGPU(device, lutCube);

    // Write initial LUT uniforms
    ATRenderPipeline._writeLutUniforms(
      device, lutUniformBuf, 17, lutStrength,
    );

    // ── Scene uniform buffer (viewProj / modelMat / eye) ─────────────────────
    const sceneUniformBuf = device.createBuffer({
      label: 'at-pipeline-scene-uniforms',
      size : 144,   // 36 × f32 = 144 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const defaultScene = makeDefaultSceneMatrices();
    device.queue.writeBuffer(
      sceneUniformBuf, 0, packSceneMatrices(defaultScene),
    );

    // ── FBO allocation ───────────────────────────────────────────────────────
    const geoFBO      = createFBO(device, W, H, format, 'geo');
    const particleFBO = createFBO(device, W, H, format, 'particle');
    const waterFBO    = createFBO(device, W, H, format, 'water');
    const vlFBO       = createFBO(device, W, H, format, 'vl');
    const bloomFBO    = createFBO(device, W, H, format, 'bloom');

    return new ATRenderPipeline(
      device, canvas, format,
      pbr, flower, spline, water, vlight, bloom,
      lutPipeline, lutBGL, lutSampler, lutUniformBuf,
      lutTexture, lutView,
      lutStyle, lutStrength,
      geoFBO, particleFBO, waterFBO, vlFBO, bloomFBO,
      sceneUniformBuf,
      passes,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — configuration
  // ─────────────────────────────────────────────────────────────────────────

  /** Update PBR material parameters. */
  setPBRParams(p: Partial<PBRParams>): this {
    this.pbr.setParams(p);
    return this;
  }

  /** Update Volumetric Light parameters. */
  setVolumetricLightParams(p: ATVolumetricLightParams): this {
    this.vlight.setParams(p);
    return this;
  }

  /** Update Bloom parameters. */
  setBloomParams(p: ATBloomParams): this {
    this.bloom.setParams(p);
    return this;
  }

  /**
   * Swap to a different LUT grade preset.
   * Rebuilds the LUT texture on the CPU and re-uploads to the GPU.
   */
  setLutStyle(style: LutStyleName, strength?: number): this {
    this.lutStyle    = style;
    if (strength !== undefined) this.lutStrength = strength;
    const gen  = new LutGenerator({ lutSize: 17 });
    const cube = gen.buildLut(style);
    // Destroy old texture and re-upload
    this.lutTexture.destroy();
    const { texture, view } = gen.uploadToWebGPU(this.device, cube);
    this.lutTexture = texture;
    this.lutView    = view;
    ATRenderPipeline._writeLutUniforms(
      this.device, this.lutUniformBuf, 17, this.lutStrength,
    );
    return this;
  }

  /** Update the LUT blend strength without changing the style. */
  setLutStrength(strength: number): this {
    this.lutStrength = Math.max(0, Math.min(1, strength));
    ATRenderPipeline._writeLutUniforms(
      this.device, this.lutUniformBuf, 17, this.lutStrength,
    );
    return this;
  }

  /**
   * Update the scene view/projection matrices and camera eye position.
   * These are forwarded to the water surface renderPass.
   */
  setSceneMatrices(m: Partial<SceneMatrices>): this {
    if (m.viewProj) this.sceneMatrices.viewProj = m.viewProj;
    if (m.modelMat) this.sceneMatrices.modelMat = m.modelMat;
    if (m.eye)      this.sceneMatrices.eye      = m.eye;
    this.device.queue.writeBuffer(
      this.sceneUniformBuf, 0, packSceneMatrices(this.sceneMatrices),
    );
    return this;
  }

  /** Enable or disable individual passes without rebuilding the pipeline. */
  setPassFlags(flags: Partial<ATPassFlags>): this {
    Object.assign(this.passes, flags);
    return this;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API — per-frame
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance all animation clocks by `dt` seconds.
   * Call exactly once per frame before `render()`.
   */
  tick(dt: number): void {
    this.pbr.tick(dt);
    this.vlight.tick(dt);
  }

  /**
   * Record all enabled render passes into `encoder`, reading each pass's
   * result FBO and writing into the next, finally blitting to `dstView`
   * (the swap-chain surface view).
   *
   * The caller is responsible for submitting the encoder.
   */
  render(encoder: GPUCommandEncoder, dstView: GPUTextureView): void {
    // ── Pass 0: Geometry — PBR material (clear → geoFBO) ──────────────────
    if (this.passes.geometry) {
      this._clearFBO(encoder, this.geoFBO);
      this.pbr.render(encoder, this.geoFBO.colorView, this.geoFBO.depthView);
    } else {
      this._clearFBO(encoder, this.geoFBO);  // always provide a valid base
    }

    // Blit geoFBO → particleFBO base (copy so particles composite over geometry)
    this._copyTexture(encoder, this.geoFBO.color, this.particleFBO.color);

    // ── Pass 1a: Flower particles (additive over particleFBO) ──────────────
    if (this.passes.flowerParticle) {
      this.flower.update(encoder, performance.now() / 1000, 0);
      this.flower.render(
        encoder,
        this.particleFBO.colorView,
        this.particleFBO.depthView,
      );
    }

    // ── Pass 1b: Spline particles (additive over particleFBO) ─────────────
    if (this.passes.splineParticle) {
      this.spline.update(encoder, performance.now() / 1000, 0);
      this.spline.render(
        encoder,
        this.particleFBO.colorView,
        this.particleFBO.depthView,
      );
    }

    // Blit particleFBO → waterFBO base
    this._copyTexture(encoder, this.particleFBO.color, this.waterFBO.color);

    // ── Pass 2: Water surface (over waterFBO) ──────────────────────────────
    if (this.passes.waterSurface) {
      this.water.tick(encoder, performance.now() / 1000);
      this.water.renderPass(
        encoder,
        this.waterFBO.colorView,
        this.waterFBO.depthView,
        this.sceneUniformBuf,
      );
    }

    // ── Pass 3: Volumetric light (waterFBO → vlFBO) ────────────────────────
    if (this.passes.volumetricLight) {
      this.vlight.render(
        encoder,
        this.waterFBO.color,   // reads waterFBO as input texture
        this.vlFBO.colorView,  // writes composited result
      );
    } else {
      this._copyTexture(encoder, this.waterFBO.color, this.vlFBO.color);
    }

    // ── Pass 4: Bloom (vlFBO → bloomFBO) ──────────────────────────────────
    if (this.passes.bloom) {
      this.bloom.render(
        encoder,
        this.vlFBO.color,       // reads vlFBO as input texture
        this.bloomFBO.colorView, // writes composited result
      );
    } else {
      this._copyTexture(encoder, this.vlFBO.color, this.bloomFBO.color);
    }

    // ── Pass 5: LUT grade (bloomFBO → dstView) ────────────────────────────
    if (this.passes.lut) {
      this._renderLutPass(encoder, this.bloomFBO.color, dstView);
    } else {
      // Direct blit to swap-chain view
      this._blitToView(encoder, this.bloomFBO.color, dstView);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Resize
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Reallocate all intermediate FBOs to the new resolution.
   * Call when the canvas is resized before the next `render()`.
   */
  resize(width: number, height: number): void {
    if (this.width === width && this.height === height) return;
    this.width  = width;
    this.height = height;

    // Destroy old FBOs
    destroyFBO(this.geoFBO);
    destroyFBO(this.particleFBO);
    destroyFBO(this.waterFBO);
    destroyFBO(this.vlFBO);
    destroyFBO(this.bloomFBO);

    // Reallocate
    this.geoFBO      = createFBO(this.device, width, height, this.format, 'geo');
    this.particleFBO = createFBO(this.device, width, height, this.format, 'particle');
    this.waterFBO    = createFBO(this.device, width, height, this.format, 'water');
    this.vlFBO       = createFBO(this.device, width, height, this.format, 'vl');
    this.bloomFBO    = createFBO(this.device, width, height, this.format, 'bloom');
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /** Release all GPU resources. Safe to call multiple times. */
  destroy(): void {
    this.pbr.destroy();
    this.bloom.destroy?.();
    destroyFBO(this.geoFBO);
    destroyFBO(this.particleFBO);
    destroyFBO(this.waterFBO);
    destroyFBO(this.vlFBO);
    destroyFBO(this.bloomFBO);
    this.lutTexture.destroy();
    this.lutUniformBuf.destroy();
    this.sceneUniformBuf.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Clear an FBO to transparent black with a fresh depth buffer. */
  private _clearFBO(encoder: GPUCommandEncoder, fbo: FBO): void {
    const pass = encoder.beginRenderPass({
      label: 'at-pipeline-clear',
      colorAttachments: [{
        view      : fbo.colorView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view            : fbo.depthView,
        depthLoadOp     : 'clear',
        depthClearValue : 1.0,
        depthStoreOp    : 'store',
      },
    });
    pass.end();
  }

  /**
   * GPU-side texture copy: src → dst (same size/format assumed).
   * Used to forward one pass's output into the next pass's base color.
   */
  private _copyTexture(
    encoder: GPUCommandEncoder,
    src    : GPUTexture,
    dst    : GPUTexture,
  ): void {
    encoder.copyTextureToTexture(
      { texture: src },
      { texture: dst },
      [this.width, this.height, 1],
    );
  }

  /**
   * Full-screen blit of `src` texture into `dstView` using the LUT pipeline
   * with strength = 0 (bypass) — used when LUT pass is disabled.
   */
  private _blitToView(
    encoder: GPUCommandEncoder,
    src    : GPUTexture,
    dstView: GPUTextureView,
  ): void {
    // Reuse LUT pipeline with strength=0 (identity blit)
    const savedStrength = this.lutStrength;
    ATRenderPipeline._writeLutUniforms(
      this.device, this.lutUniformBuf, 17, 0.0,
    );
    this._renderLutPass(encoder, src, dstView);
    ATRenderPipeline._writeLutUniforms(
      this.device, this.lutUniformBuf, 17, savedStrength,
    );
  }

  /** Record the LUT grade render pass: src texture → dstView. */
  private _renderLutPass(
    encoder: GPUCommandEncoder,
    src    : GPUTexture,
    dstView: GPUTextureView,
  ): void {
    const sceneSampler = this.device.createSampler({
      magFilter   : 'linear',
      minFilter   : 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const bg = this.device.createBindGroup({
      label  : 'at-lut-bg',
      layout : this.lutBGL,
      entries: [
        { binding: 0, resource: { buffer: this.lutUniformBuf } },
        { binding: 1, resource: sceneSampler },
        { binding: 2, resource: src.createView() },
        { binding: 3, resource: this.lutSampler },
        { binding: 4, resource: this.lutView },
      ],
    });

    const pass = encoder.beginRenderPass({
      label           : 'at-lut-pass',
      colorAttachments: [{
        view   : dstView,
        loadOp : 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.lutPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);   // full-screen triangle
    pass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Static helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Build the LUT render pipeline and associated GPU objects. */
  private static async _buildLutPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
  ): Promise<{
    lutPipeline   : GPURenderPipeline;
    lutBGL        : GPUBindGroupLayout;
    lutSampler    : GPUSampler;
    lutUniformBuf : GPUBuffer;
  }> {
    const mod = device.createShaderModule({
      label: 'at-lut-module',
      code : LUT_PASS_WGSL,
    });

    const lutBGL = device.createBindGroupLayout({
      label  : 'at-lut-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          buffer : { type: 'uniform' } },
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

    const lutPipeline = await device.createRenderPipelineAsync({
      label : 'at-lut-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [lutBGL] }),
      vertex: {
        module    : mod,
        entryPoint: 'vs_lut',
      },
      fragment: {
        module    : mod,
        entryPoint: 'fs_lut',
        targets   : [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const lutSampler = device.createSampler({
      label       : 'at-lut-sampler',
      magFilter   : 'nearest',   // LUT uses nearest — trilinear done in shader
      minFilter   : 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const lutUniformBuf = device.createBuffer({
      label: 'at-lut-uniforms',
      size : 16,   // 4 × f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return { lutPipeline, lutBGL, lutSampler, lutUniformBuf };
  }

  /** Write LUT uniform buffer: lutSize + strength + 2× pad. */
  private static _writeLutUniforms(
    device   : GPUDevice,
    buf      : GPUBuffer,
    lutSize  : number,
    strength : number,
  ): void {
    device.queue.writeBuffer(
      buf, 0,
      new Float32Array([lutSize, strength, 0, 0]),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports — expose constituent module types for caller convenience
// ─────────────────────────────────────────────────────────────────────────────

export type {
  PBRParams,
  FlowerEdgeSpline,
  ATFlowerConfig,
  EdgeSpline,
  ATSplineParticleConfig,
  ATWaterSurfaceConfig,
  ATVolumetricLightParams,
  ATBloomParams,
  LutStyleName,
};

export {
  DEFAULT_PBR_PARAMS,
};
