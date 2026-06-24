/**
 * god-rays.ts — M761: God Rays Radial Scatter — WebGPU Compute Post-Process
 *
 * 体积光 God Rays 后处理管线，完全基于 WebGPU compute shader：
 *
 *   核心算法：
 *     1. Radial Blur 从光源散射 — 每像素沿光源方向采样累积，模拟光柱效果
 *     2. Occlusion Mask — 亮度阈值提取遮挡掩码，标识光源区域
 *     3. Intensity Decay — 指数衰减 decay^i × weight，距离光源越远衰减越快
 *     4. 多光源支持 — 最多 MAX_LIGHTS (8) 个独立光源，各自参数独立
 *
 *   参考源码：
 *     at-volumetric-light.ts          — AT VolumetricLight×6 render-pass 实现
 *     upstream/unreal-renderer-ue5    — UE5 LightShaftRendering.cpp
 *     upstream/pixijs-filters/godray  — PixiJS God Ray WGSL
 *     upstream/lygia/lighting         — Lygia volumetricLightScattering.wgsl
 *
 *   与 at-volumetric-light.ts 的区别：
 *     - 本模块使用 compute shader 而非 render pass（更适合后处理管线集成）
 *     - 支持多光源（at-volumetric-light.ts 仅支持单光源）
 *     - 所有 pass 均为 compute dispatch，无需全屏三角形 / 顶点着色器
 *     - 使用 storage texture 直接写入，避免 render attachment 开销
 *
 *   管线结构（每帧 dispatch()）：
 *     ┌─ Pass 1: OCCLUSION MASK (compute) ──────────────────────────────────┐
 *     │  scene → occlusionTex (半分辨率)                                     │
 *     │  亮度 > threshold 的像素 → 保留；其余 → 黑                            │
 *     └─────────────────────────────────────────────────────────────────────┘
 *     ┌─ Pass 2: RADIAL BLUR per light (compute) ──────────────────────────┐
 *     │  occlusionTex → raysTex (半分辨率)                                   │
 *     │  对每个光源：沿 UV → lightPos 方向径向采样                            │
 *     │  每步 weight *= decay (指数衰减)                                     │
 *     │  多光源结果累加（additive）                                           │
 *     └─────────────────────────────────────────────────────────────────────┘
 *     ┌─ Pass 3: COMPOSITE (compute) ───────────────────────────────────────┐
 *     │  scene + raysTex → dst                                               │
 *     │  加性混合: output = scene + rays × globalIntensity                    │
 *     └─────────────────────────────────────────────────────────────────────┘
 *
 * 快速使用：
 *   const gr = await GodRaysCompute.create(device, width, height);
 *   gr.setLights([
 *     { pos: [0.5, 0.1], color: [1, 0.9, 0.7], intensity: 1.0 },
 *     { pos: [0.8, 0.2], color: [0.5, 0.7, 1], intensity: 0.6 },
 *   ]);
 *   gr.setParams({ exposure: 0.86, decay: 0.97, density: 0.22 });
 *   // 每帧:
 *   gr.dispatch(encoder, sceneTexView, dstTexView);
 *
 * Research: xiaodi #M761 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum number of simultaneous light sources for god rays. */








const MAX_LIGHTS = 8;

/** Workgroup size for compute dispatches (16×16 = 256 threads). */
const WG_SIZE = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single god-ray light source.
 *
 * Each light produces its own set of radial blur shafts. Multiple lights
 * accumulate additively, each with independent position, color, intensity,
 * and decay characteristics.
 */
export interface GodRayLight {
  /**
   * UV position of the light source in [0,1]² screen space.
   * [0,0] = top-left; [1,1] = bottom-right.
   */
  pos: [number, number];

  /**
   * Light color in linear RGB [0,1]³.
   * Multiplied with the accumulated shaft radiance.
   * @default [1.0, 1.0, 1.0]
   */
  color?: [number, number, number];

  /**
   * Per-light intensity multiplier.
   * Scales the contribution of this light's rays in the composite.
   * @default 1.0
   */
  intensity?: number;

  /**
   * Per-sample exponential decay for this light.
   * Lower = faster fade-out from the light source.
   * Overrides the global decay if set.
   * @default undefined (uses global decay)
   */
  decay?: number;

  /**
   * Per-light density override.
   * Controls how far each radial step marches toward the light.
   * @default undefined (uses global density)
   */
  density?: number;
}

/**
 * Global god-ray parameters (shared across all lights).
 */
export interface GodRaysParams {
  /**
   * Overall brightness scalar.
   * AT: VolumetricLight_home fExposure = 0.86
   * @default 0.86
   */
  exposure?: number;

  /**
   * Controls how far each radial step marches (0–1, fraction of screen).
   * AT: VolumetricLight_home fDensity = 0.22
   * Can be overridden per-light.
   * @default 0.22
   */
  density?: number;

  /**
   * Per-sample exponential decay.  Lower = faster fade-out.
   * AT: ~0.97 (derived from 6-iteration radial blur tapering)
   * Can be overridden per-light.
   * @default 0.97
   */
  decay?: number;

  /**
   * Base tap weight (multiplied by decay^i per iteration).
   * @default 0.4
   */
  weight?: number;

  /**
   * Luminance threshold below which pixels are excluded from the occlusion mask.
   * @default 0.6
   */
  occlusionThreshold?: number;

  /**
   * Number of radial blur samples per ray march.
   * AT: exactly 6 (VolumetricLight×6 node chain).
   * Higher = smoother shafts; lower = faster GPU.
   * @default 64
   */
  numSamples?: number;

  /**
   * Global intensity multiplier for the composite additive blend.
   * output = scene + rays × globalIntensity
   * @default 1.0
   */
  globalIntensity?: number;

  /**
   * Half-resolution processing toggle.
   * When true, occlusion mask and radial blur run at half resolution.
   * @default true
   */
  halfRes?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: resolved defaults
// ─────────────────────────────────────────────────────────────────────────────

interface ResolvedParams extends Required<GodRaysParams> {}

const DEFAULTS: ResolvedParams = {
  exposure           : 0.86,
  density            : 0.22,
  decay              : 0.97,
  weight             : 0.40,
  occlusionThreshold : 0.60,
  numSamples         : 64,
  globalIntensity    : 1.0,
  halfRes            : true,
};

/**
 * Resolved light with all optional fields filled.
 * Packed into the uniform buffer for GPU consumption.
 */
interface ResolvedLight {
  posX      : number;
  posY      : number;
  colorR    : number;
  colorG    : number;
  colorB    : number;
  intensity : number;
  decay     : number;
  density   : number;
}

function resolveLight(light: GodRayLight, globalDecay: number, globalDensity: number): ResolvedLight {
  return {
    posX      : light.pos[0],
    posY      : light.pos[1],
    colorR    : light.color?.[0] ?? 1.0,
    colorG    : light.color?.[1] ?? 1.0,
    colorB    : light.color?.[2] ?? 1.0,
    intensity : light.intensity ?? 1.0,
    decay     : light.decay    ?? globalDecay,
    density   : light.density  ?? globalDensity,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniform struct (shared across all compute passes)
//
// Layout (std140 / WGSL uniform alignment):
//   offset 0:   GRGlobals    — 8 × f32 = 32 bytes
//   offset 32:  GRLight[8]   — 8 × (8 × f32) = 256 bytes
//   Total: 288 bytes
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GR_UNIFORMS = /* wgsl */`
struct GRGlobals {
  exposure           : f32,  // 0
  density            : f32,  // 4
  decay              : f32,  // 8
  weight             : f32,  // 12
  occlusionThreshold : f32,  // 16
  numSamples         : f32,  // 20
  globalIntensity    : f32,  // 24
  numLights          : f32,  // 28
};

struct GRLight {
  posX      : f32,
  posY      : f32,
  colorR    : f32,
  colorG    : f32,
  colorB    : f32,
  intensity : f32,
  decay     : f32,
  density   : f32,
};

struct GRUniforms {
  globals : GRGlobals,
  lights  : array<GRLight, ${MAX_LIGHTS}>,
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Occlusion Mask (compute)
//
// Reads the scene texture, extracts pixels whose luminance exceeds the
// threshold, writes to the occlusion storage texture. This isolates the
// "light source" regions that will seed the radial blur.
//
// Equivalent to at-volumetric-light.ts WGSL_OCCLUSION but in compute form.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_OCCLUSION_COMPUTE = /* wgsl */`
${WGSL_GR_UNIFORMS}

@group(0) @binding(0) var<uniform> u        : GRUniforms;
@group(0) @binding(1) var          sceneSmp  : sampler;
@group(0) @binding(2) var          sceneTex  : texture_2d<f32>;
@group(0) @binding(3) var          occlusOut : texture_storage_2d<rgba16float, write>;

override wgSize: u32 = ${WG_SIZE}u;

@compute @workgroup_size(wgSize, wgSize, 1)
fn csOcclusion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outDims = textureDimensions(occlusOut);
  if (gid.x >= outDims.x || gid.y >= outDims.y) { return; }

  // Map output texel to scene UV (handles half-res)
  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(outDims);

  // Sample scene at this UV
  let color = textureSampleLevel(sceneTex, sceneSmp, uv, 0.0);

  // Rec.709 luminance
  let lum = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

  // Hard threshold mask — keep only bright regions (light sources)
  let keep = step(u.globals.occlusionThreshold, lum);

  textureStore(occlusOut, gid.xy, vec4<f32>(color.rgb * keep, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: Radial Blur (compute) — multi-light god rays
//
// For each pixel:
//   For each active light:
//     March from pixel UV toward light UV in numSamples steps.
//     Accumulate occlusion texture samples with exponential decay.
//     Weight by per-light intensity and color.
//   Sum all light contributions additively.
//
// This is the core god-ray algorithm — the "radial scatter from light source"
// with intensity decay. Each light has its own decay and density so different
// light sources can have distinct shaft characteristics.
//
// Key differences from at-volumetric-light.ts:
//   - Multiple lights in a single dispatch (vs single light per render pass)
//   - Compute shader with storage texture writes
//   - Higher sample count (64 default vs AT's 6) for smoother shafts
//   - Per-light color tinting
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_RADIAL_BLUR_COMPUTE = /* wgsl */`
${WGSL_GR_UNIFORMS}

@group(0) @binding(0) var<uniform> u        : GRUniforms;
@group(0) @binding(1) var          occlusSmp : sampler;
@group(0) @binding(2) var          occlusTex : texture_2d<f32>;
@group(0) @binding(3) var          raysOut   : texture_storage_2d<rgba16float, write>;

override wgSize: u32 = ${WG_SIZE}u;

@compute @workgroup_size(wgSize, wgSize, 1)
fn csRadialBlur(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outDims = textureDimensions(raysOut);
  if (gid.x >= outDims.x || gid.y >= outDims.y) { return; }

  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(outDims);

  // Accumulate rays from all active lights
  var totalRays = vec3<f32>(0.0);

  let nLights = i32(clamp(u.globals.numLights, 0.0, ${MAX_LIGHTS}.0));
  let nSamples = i32(clamp(u.globals.numSamples, 1.0, 128.0));

  for (var li = 0; li < nLights; li++) {
    let light = u.lights[li];
    let lightPos = vec2<f32>(light.posX, light.posY);

    // Direction from this pixel toward the light source
    let toLight = lightPos - uv;

    // Step size: density controls how far we march per sample
    // Higher density = longer shafts but more spread out
    let stepVec = toLight * (light.density / f32(nSamples));

    var sampleUV = uv;
    var lightAccum = vec3<f32>(0.0);
    var w = u.globals.weight;

    // ── Radial blur march: exponential decay accumulation ──────────────
    // Each step:
    //   sample = occlusionTex(sampleUV)
    //   lightAccum += sample * w
    //   w *= decay              ← intensity decay (exponential falloff)
    //   sampleUV += stepVec     ← march toward light
    //
    // This produces the characteristic "shaft" appearance where rays
    // are brightest near the light and fade with distance.
    for (var si = 0; si < nSamples; si++) {
      sampleUV += stepVec;

      // Clamp UV to prevent sampling outside texture bounds
      let clampedUV = clamp(sampleUV, vec2<f32>(0.0), vec2<f32>(1.0));
      let s = textureSampleLevel(occlusTex, occlusSmp, clampedUV, 0.0).rgb;

      lightAccum += s * w;
      w *= light.decay;
    }

    // Apply exposure and per-light color & intensity
    let lightColor = vec3<f32>(light.colorR, light.colorG, light.colorB);
    lightAccum *= u.globals.exposure * lightColor * light.intensity;

    // Additive accumulation across all lights
    totalRays += lightAccum;
  }

  textureStore(raysOut, gid.xy, vec4<f32>(totalRays, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Composite (compute)
//
// Additive blend: output = scene + rays × globalIntensity
// Reads from both the scene and the rays texture, writes to the destination.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE_COMPUTE = /* wgsl */`
${WGSL_GR_UNIFORMS}

@group(0) @binding(0) var<uniform> u          : GRUniforms;
@group(0) @binding(1) var          sceneSmp   : sampler;
@group(0) @binding(2) var          sceneTex   : texture_2d<f32>;
@group(0) @binding(3) var          raysSmp    : sampler;
@group(0) @binding(4) var          raysTex    : texture_2d<f32>;
@group(0) @binding(5) var          dstOut     : texture_storage_2d<rgba16float, write>;

override wgSize: u32 = ${WG_SIZE}u;

@compute @workgroup_size(wgSize, wgSize, 1)
fn csComposite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outDims = textureDimensions(dstOut);
  if (gid.x >= outDims.x || gid.y >= outDims.y) { return; }

  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(outDims);

  let scene = textureSampleLevel(sceneTex, sceneSmp, uv, 0.0);
  let rays  = textureSampleLevel(raysTex,  raysSmp,  uv, 0.0);

  // Additive composite with global intensity control
  let result = scene.rgb + rays.rgb * u.globals.globalIntensity;

  // Soft clamp to HDR range (avoid hard clip artifacts)
  let clamped = min(result, vec3<f32>(4.0));

  textureStore(dstOut, gid.xy, vec4<f32>(clamped, scene.a));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Intermediate storage texture (rgba16float for HDR). */
function makeStorageTex(
  device : GPUDevice,
  width  : number,
  height : number,
  label  : string,
): GPUTexture {
  return device.createTexture({
    label,
    size   : [width, height, 1],
    format : 'rgba16float',
    usage  : GPUTextureUsage.STORAGE_BINDING
           | GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.COPY_SRC,
  });
}

/** Compute the dispatch dimensions (ceil division by workgroup size). */
function dispatchSize(pixels: number): number {
  return Math.ceil(pixels / WG_SIZE);
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer layout
//
// GRGlobals:  8 × f32 = 32 bytes
// GRLight[8]: 8 × (8 × f32) = 256 bytes
// Total: 288 bytes (aligned to 16-byte boundary)
// ─────────────────────────────────────────────────────────────────────────────

const GLOBALS_F32_COUNT  = 8;
const LIGHT_F32_COUNT    = 8;
const UNIFORM_F32_TOTAL  = GLOBALS_F32_COUNT + MAX_LIGHTS * LIGHT_F32_COUNT;
const UNIFORM_BYTE_SIZE  = UNIFORM_F32_TOTAL * 4;  // 288 bytes

// ─────────────────────────────────────────────────────────────────────────────
// GodRaysCompute — main WebGPU compute class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GodRaysCompute — WebGPU Compute God Rays Post-Process.
 *
 * Implements a 3-pass compute-shader pipeline for god ray generation:
 *   1. Occlusion mask extraction (bright pixel threshold)
 *   2. Multi-light radial blur with exponential intensity decay
 *   3. Additive composite over the scene
 *
 * Supports up to 8 simultaneous light sources, each with independent
 * position, color, intensity, decay, and density parameters.
 */
export class GodRaysCompute {
  private readonly device : GPUDevice;
  private readonly width  : number;
  private readonly height : number;

  // ── GPU resources ─────────────────────────────────────────────────────────
  private uniformBuf   : GPUBuffer;
  private sceneSampler : GPUSampler;
  private raysSampler  : GPUSampler;

  // Intermediate textures
  private occlusionTex : GPUTexture;  // half-res bright-pixel mask
  private raysTex      : GPUTexture;  // half-res radial blur accumulation
  private dstTex       : GPUTexture;  // full-res composite output

  // Half-resolution dimensions
  private readonly halfW : number;
  private readonly halfH : number;

  // ── Pipeline state ────────────────────────────────────────────────────────
  private occlusionPipeline  : GPUComputePipeline;
  private radialBlurPipeline : GPUComputePipeline;
  private compositePipeline  : GPUComputePipeline;

  private occlusionBGL  : GPUBindGroupLayout;
  private radialBlurBGL : GPUBindGroupLayout;
  private compositeBGL  : GPUBindGroupLayout;

  // ── Runtime state ─────────────────────────────────────────────────────────
  private params : ResolvedParams;
  private lights : ResolvedLight[] = [];

  private constructor(
    device : GPUDevice,
    width  : number,
    height : number,
  ) {
    this.device = device;
    this.width  = width;
    this.height = height;
    this.halfW  = Math.max(1, width  >> 1);
    this.halfH  = Math.max(1, height >> 1);
    this.params = { ...DEFAULTS };

    // ── Samplers ────────────────────────────────────────────────────────────
    this.sceneSampler = device.createSampler({
      label        : 'gr-scene-sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    this.raysSampler = device.createSampler({
      label        : 'gr-rays-sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    // ── Intermediate textures ───────────────────────────────────────────────
    this.occlusionTex = makeStorageTex(device, this.halfW, this.halfH, 'gr-occlusion');
    this.raysTex      = makeStorageTex(device, this.halfW, this.halfH, 'gr-rays');
    this.dstTex       = makeStorageTex(device, width, height, 'gr-dst');

    // ── Uniform buffer ──────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      label : 'gr-uniforms',
      size  : UNIFORM_BYTE_SIZE,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Pass 1: Occlusion mask BGL + pipeline ───────────────────────────────
    this.occlusionBGL = device.createBindGroupLayout({
      label: 'gr-occlusion-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    });

    const occlusionModule = device.createShaderModule({
      label: 'gr-occlusion-shader',
      code: WGSL_OCCLUSION_COMPUTE,
    });

    this.occlusionPipeline = device.createComputePipeline({
      label  : 'gr-occlusion-pipeline',
      layout : device.createPipelineLayout({
        label: 'gr-occlusion-layout',
        bindGroupLayouts: [this.occlusionBGL],
      }),
      compute: { module: occlusionModule, entryPoint: 'csOcclusion' },
    });

    // ── Pass 2: Radial blur BGL + pipeline ──────────────────────────────────
    this.radialBlurBGL = device.createBindGroupLayout({
      label: 'gr-radial-blur-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    });

    const radialBlurModule = device.createShaderModule({
      label: 'gr-radial-blur-shader',
      code: WGSL_RADIAL_BLUR_COMPUTE,
    });

    this.radialBlurPipeline = device.createComputePipeline({
      label  : 'gr-radial-blur-pipeline',
      layout : device.createPipelineLayout({
        label: 'gr-radial-blur-layout',
        bindGroupLayouts: [this.radialBlurBGL],
      }),
      compute: { module: radialBlurModule, entryPoint: 'csRadialBlur' },
    });

    // ── Pass 3: Composite BGL + pipeline ────────────────────────────────────
    this.compositeBGL = device.createBindGroupLayout({
      label: 'gr-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    });

    const compositeModule = device.createShaderModule({
      label: 'gr-composite-shader',
      code: WGSL_COMPOSITE_COMPUTE,
    });

    this.compositePipeline = device.createComputePipeline({
      label  : 'gr-composite-pipeline',
      layout : device.createPipelineLayout({
        label: 'gr-composite-layout',
        bindGroupLayouts: [this.compositeBGL],
      }),
      compute: { module: compositeModule, entryPoint: 'csComposite' },
    });

    // Sync initial state to GPU
    this._uploadUniforms();
  }

  // ── Factory ───────────────────────────────────────────────────────────────

  /**
   * Factory — asynchronous constructor.
   *
   * @param device - WebGPU device.
   * @param width  - Viewport width in pixels.
   * @param height - Viewport height in pixels.
   */
  static async create(
    device : GPUDevice,
    width  : number,
    height : number,
  ): Promise<GodRaysCompute> {
    return new GodRaysCompute(device, width, height);
  }

  // ── Parameter API ─────────────────────────────────────────────────────────

  /**
   * Update global god-ray parameters.
   * Writes the new values to the GPU uniform buffer.
   *
   * @param p - Partial parameter overrides.
   */
  setParams(p: GodRaysParams): void {
    if (p.exposure           !== undefined) this.params.exposure           = p.exposure;
    if (p.density            !== undefined) this.params.density            = p.density;
    if (p.decay              !== undefined) this.params.decay              = p.decay;
    if (p.weight             !== undefined) this.params.weight             = p.weight;
    if (p.occlusionThreshold !== undefined) this.params.occlusionThreshold = p.occlusionThreshold;
    if (p.numSamples         !== undefined) this.params.numSamples         = p.numSamples;
    if (p.globalIntensity    !== undefined) this.params.globalIntensity    = p.globalIntensity;
    if (p.halfRes            !== undefined) this.params.halfRes            = p.halfRes;
    this._uploadUniforms();
  }

  /**
   * Set the active light sources for god ray generation.
   *
   * Each light produces independent radial blur shafts. Results accumulate
   * additively. Maximum of 8 lights; excess lights are silently dropped.
   *
   * @param lights - Array of light source descriptors.
   */
  setLights(lights: GodRayLight[]): void {
    this.lights = lights.slice(0, MAX_LIGHTS).map(l =>
      resolveLight(l, this.params.decay, this.params.density)
    );
    this._uploadUniforms();
  }

  /**
   * Add a single light source (up to MAX_LIGHTS).
   * Returns the index of the added light, or -1 if at capacity.
   */
  addLight(light: GodRayLight): number {
    if (this.lights.length >= MAX_LIGHTS) return -1;
    this.lights.push(resolveLight(light, this.params.decay, this.params.density));
    this._uploadUniforms();
    return this.lights.length - 1;
  }

  /**
   * Update a light at a specific index.
   */
  updateLight(index: number, light: GodRayLight): void {
    if (index < 0 || index >= this.lights.length) return;
    this.lights[index] = resolveLight(light, this.params.decay, this.params.density);
    this._uploadUniforms();
  }

  /**
   * Remove a light at a specific index.
   */
  removeLight(index: number): void {
    if (index < 0 || index >= this.lights.length) return;
    this.lights.splice(index, 1);
    this._uploadUniforms();
  }

  /** Remove all lights. */
  clearLights(): void {
    this.lights = [];
    this._uploadUniforms();
  }

  // ── Dispatch ──────────────────────────────────────────────────────────────

  /**
   * Execute the full 3-pass god rays compute pipeline.
   *
   * @param encoder       - Active GPUCommandEncoder.
   * @param sceneTexView  - Source scene texture view (input).
   * @param dstTexView    - Destination texture view (output, must support storage write).
   *                        If undefined, writes to the internal dstTex (retrieve via getDstTexture).
   */
  dispatch(
    encoder      : GPUCommandEncoder,
    sceneTexView : GPUTextureView,
    dstTexView?  : GPUTextureView,
  ): void {
    const occlusionView = this.occlusionTex.createView();
    const raysView      = this.raysTex.createView();
    const finalDstView  = dstTexView ?? this.dstTex.createView();

    // ── Pass 1: Occlusion mask ──────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'gr-occlusion-bg',
        layout : this.occlusionBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sceneSampler },
          { binding: 2, resource: sceneTexView },
          { binding: 3, resource: occlusionView },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'gr-occlusion-pass' });
      pass.setPipeline(this.occlusionPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        dispatchSize(this.halfW),
        dispatchSize(this.halfH),
      );
      pass.end();
    }

    // ── Pass 2: Radial blur (multi-light accumulation) ──────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'gr-radial-blur-bg',
        layout : this.radialBlurBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sceneSampler },
          { binding: 2, resource: occlusionView },
          { binding: 3, resource: raysView },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'gr-radial-blur-pass' });
      pass.setPipeline(this.radialBlurPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        dispatchSize(this.halfW),
        dispatchSize(this.halfH),
      );
      pass.end();
    }

    // ── Pass 3: Composite (scene + rays → dst) ─────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'gr-composite-bg',
        layout : this.compositeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sceneSampler },
          { binding: 2, resource: sceneTexView },
          { binding: 3, resource: this.raysSampler },
          { binding: 4, resource: raysView },
          { binding: 5, resource: finalDstView },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'gr-composite-pass' });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        dispatchSize(this.width),
        dispatchSize(this.height),
      );
      pass.end();
    }
  }

  // ── Resource access ───────────────────────────────────────────────────────

  /**
   * Get the internal destination texture (useful when no external dstTexView
   * is provided to dispatch()).
   */
  getDstTexture(): GPUTexture {
    return this.dstTex;
  }

  /**
   * Get the intermediate rays texture (for debug visualization).
   */
  getRaysTexture(): GPUTexture {
    return this.raysTex;
  }

  /**
   * Get the occlusion mask texture (for debug visualization).
   */
  getOcclusionTexture(): GPUTexture {
    return this.occlusionTex;
  }

  // ── Resize ────────────────────────────────────────────────────────────────

  /**
   * Resize internal textures to match a new viewport size.
   * This destroys and recreates all intermediate textures.
   *
   * @param width  - New viewport width.
   * @param height - New viewport height.
   * @returns A new GodRaysCompute instance with the updated dimensions.
   */
  async resize(width: number, height: number): Promise<GodRaysCompute> {
    this.destroy();
    const resized = await GodRaysCompute.create(this.device, width, height);
    resized.setParams(this.params);
    if (this.lights.length > 0) {
      // Re-inject lights from the raw resolved data
      resized.lights = [...this.lights];
      resized._uploadUniforms();
    }
    return resized;
  }

  // ── Resource management ───────────────────────────────────────────────────

  /**
   * Release all GPU resources.
   * The instance must not be used after calling this method.
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.occlusionTex.destroy();
    this.raysTex.destroy();
    this.dstTex.destroy();
  }

  // ── Diagnostics ───────────────────────────────────────────────────────────

  /** Returns the three WGSL shader sources for inspection / hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      occlusion  : WGSL_OCCLUSION_COMPUTE,
      radialBlur : WGSL_RADIAL_BLUR_COMPUTE,
      composite  : WGSL_COMPOSITE_COMPUTE,
    };
  }

  /** Current resolved parameters (including defaults). */
  get currentParams(): Readonly<ResolvedParams> {
    return { ...this.params };
  }

  /** Current active lights (read-only copy). */
  get currentLights(): Readonly<ResolvedLight[]> {
    return this.lights.map(l => ({ ...l }));
  }

  // ── Private ───────────────────────────────────────────────────────────────

  /**
   * Write current params + lights to the GPU uniform buffer.
   *
   * Layout:
   *   [0..7]   = GRGlobals  (exposure, density, decay, weight, threshold,
   *                           numSamples, globalIntensity, numLights)
   *   [8..71]  = GRLight[0..7] × 8 floats each
   *              (posX, posY, colorR, colorG, colorB, intensity, decay, density)
   */
  private _uploadUniforms(): void {
    const buf = new Float32Array(UNIFORM_F32_TOTAL);

    // Globals
    buf[0] = this.params.exposure;
    buf[1] = this.params.density;
    buf[2] = this.params.decay;
    buf[3] = this.params.weight;
    buf[4] = this.params.occlusionThreshold;
    buf[5] = this.params.numSamples;
    buf[6] = this.params.globalIntensity;
    buf[7] = this.lights.length;

    // Lights (8 floats each)
    for (let i = 0; i < this.lights.length; i++) {
      const offset = GLOBALS_F32_COUNT + i * LIGHT_F32_COUNT;
      const l = this.lights[i];
      buf[offset + 0] = l.posX;
      buf[offset + 1] = l.posY;
      buf[offset + 2] = l.colorR;
      buf[offset + 3] = l.colorG;
      buf[offset + 4] = l.colorB;
      buf[offset + 5] = l.intensity;
      buf[offset + 6] = l.decay;
      buf[offset + 7] = l.density;
    }

    // Zero out unused light slots (defensive)
    for (let i = this.lights.length; i < MAX_LIGHTS; i++) {
      const offset = GLOBALS_F32_COUNT + i * LIGHT_F32_COUNT;
      for (let j = 0; j < LIGHT_F32_COUNT; j++) {
        buf[offset + j] = 0;
      }
    }

    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — species-params integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a GodRaysCompute from species-level visual parameters.
 * Bridges the SPH species parameter schema to the god rays API.
 *
 * @param device      - WebGPU device.
 * @param width       - Viewport width.
 * @param height      - Viewport height.
 * @param speciesOpts - Species-specific overrides.
 */
export async function createGodRaysForSpecies(
  device      : GPUDevice,
  width       : number,
  height      : number,
  speciesOpts?: {
    lights?          : GodRayLight[];
    exposure?        : number;
    density?         : number;
    decay?           : number;
    numSamples?      : number;
    globalIntensity? : number;
  },
): Promise<GodRaysCompute> {
  const gr = await GodRaysCompute.create(device, width, height);
  if (speciesOpts) {
    gr.setParams({
      exposure        : speciesOpts.exposure        ?? DEFAULTS.exposure,
      density         : speciesOpts.density         ?? DEFAULTS.density,
      decay           : speciesOpts.decay           ?? DEFAULTS.decay,
      numSamples      : speciesOpts.numSamples      ?? DEFAULTS.numSamples,
      globalIntensity : speciesOpts.globalIntensity ?? DEFAULTS.globalIntensity,
    });
    if (speciesOpts.lights) {
      gr.setLights(speciesOpts.lights);
    }
  }
  return gr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL source fragments for embedding in other shader modules. */
export const GOD_RAYS_WGSL = {
  /** Shared uniform structs (GRGlobals, GRLight, GRUniforms). */
  uniforms   : WGSL_GR_UNIFORMS,
  /** Occlusion mask compute shader. */
  occlusion  : WGSL_OCCLUSION_COMPUTE,
  /** Radial blur compute shader (multi-light). */
  radialBlur : WGSL_RADIAL_BLUR_COMPUTE,
  /** Composite compute shader. */
  composite  : WGSL_COMPOSITE_COMPUTE,
} as const;

/** Maximum number of simultaneous god-ray lights. */
export { MAX_LIGHTS as GOD_RAYS_MAX_LIGHTS };
