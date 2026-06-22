/**
 * at-bloom-postprocess.ts — M714: AT UnrealBloom Post-Process Pipeline — WebGPU/WGSL Port
 *
 * 移植自 ActiveTheory / Unreal Engine 的 UnrealBloom 后处理三段管线：
 *
 *   Stage 1 — LUMINOSITY THRESHOLD (亮度阈值提取)
 *     来源：upstream/unreal-renderer-ue5/Shaders-Private/PostProcessBloom.usf
 *              BloomSetupCommon() / BloomSetupPS / BloomSetupCS
 *          upstream/unreal-renderer/PostProcess/PostProcessBloomSetup.cpp
 *              [ASTRO-BLOOM] 注释标记 M086-M090
 *     原理：提取超过 BloomThreshold 的高亮度像素。
 *       TotalLuminance = dot(rgb, vec3(0.2126, 0.7152, 0.0722)) * exposureScale
 *       BloomLuminance = TotalLuminance - BloomThreshold
 *       BloomAmount    = saturate(BloomLuminance × 0.5)
 *     移植策略：exposureScale 硬编码为 1.0（WebGPU 无 EyeAdaptation buffer），
 *     保留 smoothstep 形态的双侧 saturate 公式。
 *
 *   Stage 2 — DUAL KAWASE / SEPARABLE GAUSSIAN BLUR (高斯模糊)
 *     来源：upstream/pixijs-filters/src/kawase-blur/kawase-blur.wgsl
 *          upstream/pixijs-filters/src/advanced-bloom/AdvancedBloomFilter.ts
 *          upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/PostProcessBloomSetup.cpp
 *              AddGaussianBloomPasses() — 6-stage downsample pyramid
 *     原理：可分离高斯核的水平+垂直两遍卷积，内核权重由 sigma 动态生成。
 *     UE5 使用 6 质量级别的下采样金字塔（BloomStages[6]）。
 *     本实现使用 7-tap 可分离 Gaussian（支持 1–3 遍迭代），权重硬编码
 *     对应 UE5 Q3 级别 sigma≈3.0（BloomKernelSizePercent × BloomSizeScale）。
 *
 *   Stage 3 — COMPOSITE (合成)
 *     来源：upstream/pixijs-filters/src/advanced-bloom/advanced-bloom.wgsl
 *              mainFragment() — color × brightness + bloomColor × bloomScale
 *          upstream/pixijs-filters/src/advanced-bloom/advanced-bloom.frag
 *     原理：原始场景颜色 × brightness 叠加模糊后的泛光颜色 × bloomScale。
 *     output = clamp(scene × brightness + bloomTex × bloomScale, 0, 1)
 *
 * 管线结构（每帧 render()）:
 *   ┌─ renderPass 1 ─ luminosity threshold ─────────────────────────────────┐
 *   │  src(scene) → brightTex  (rgba16float, 全屏三角)                      │
 *   └───────────────────────────────────────────────────────────────────────┘
 *   ┌─ renderPass 2..N ─ separable gaussian blur ────────────────────────────┐
 *   │  pass 2: brightTex → blurHTex  (水平方向, 7-tap)                       │
 *   │  pass 3: blurHTex  → blurVTex  (垂直方向, 7-tap)                       │
 *   │  （可选追加更多 H/V 遍以加大模糊半径）                                    │
 *   └───────────────────────────────────────────────────────────────────────┘
 *   ┌─ renderPass N+1 ─ composite ──────────────────────────────────────────┐
 *   │  src(scene) + blurVTex → dst (用户指定的 GPUTextureView)               │
 *   └───────────────────────────────────────────────────────────────────────┘
 *
 * 用法：
 *   const bloom = await ATBloomPostProcess.create(device, format, width, height);
 *
 *   bloom.setParams({
 *     threshold  : 0.5,   // UE BloomThreshold
 *     bloomScale : 1.2,   // AT bloomScale uniform
 *     brightness : 1.0,   // AT uBrightness uniform
 *     blurPasses : 2,     // gaussian 迭代次数 (1–4)
 *     blurRadius : 2.0,   // 模糊内核扩散系数
 *   });
 *
 *   // 每帧:
 *   bloom.render(encoder, sceneTexture, dstView);
 *
 * Research: xiaodi #M714 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Tweakable bloom parameters (all optional, defaults shown). */
export interface ATBloomParams {
  /**
   * Luminance threshold below which pixels are excluded from bloom.
   * Maps 1:1 to UE5 BloomThreshold.
   * @default 0.5
   */
  threshold?: number;

  /**
   * Multiplier applied to the blurred bloom layer before composite.
   * Matches AT AdvancedBloomFilter uBloomScale.
   * @default 1.0
   */
  bloomScale?: number;

  /**
   * Brightness multiplier applied to the original scene during composite.
   * Matches AT AdvancedBloomFilter uBrightness.
   * @default 1.0
   */
  brightness?: number;

  /**
   * Number of H+V gaussian blur iteration passes (1 = fast, 4 = wide glow).
   * Loosely maps to UE5 EBloomQuality Q1–Q4.
   * @default 2
   */
  blurPasses?: number;

  /**
   * Scales the blur kernel's effective spread per texel.
   * Higher → wider, softer bloom halo.
   * @default 2.0
   */
  blurRadius?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: shared uniforms block
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_BLOOM_UNIFORMS = /* wgsl */`
// Mirrors ATBloomUniforms (Float32Array layout, std140 aligned)
struct BloomUniforms {
  // Stage 1: luminosity threshold
  threshold      : f32,   // UE BloomThreshold
  exposureScale  : f32,   // fixed 1.0 (no EyeAdaptation in WebGPU)
  // Stage 2: gaussian blur
  blurRadius     : f32,   // texel spread coefficient
  _pad0          : f32,
  // Stage 3: composite
  bloomScale     : f32,   // AT uBloomScale
  brightness     : f32,   // AT uBrightness
  _pad1          : f32,
  _pad2          : f32,
  // Resolution
  invWidth       : f32,
  invHeight      : f32,
  _pad3          : f32,
  _pad4          : f32,
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: luminance helper (Rec. 709, same as UE5 Luminance())
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LUMINANCE = /* wgsl */`
fn luminance(c: vec3f) -> f32 {
  // Rec.709 luma — mirrors UE5 Common.ush Luminance()
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — LUMINOSITY THRESHOLD
// Source: upstream/unreal-renderer-ue5/Shaders-Private/PostProcessBloom.usf
//   BloomSetupCommon() → BloomSetupPS
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LUMINOSITY = /* wgsl */`
${WGSL_BLOOM_UNIFORMS}
${WGSL_LUMINANCE}

@group(0) @binding(0) var<uniform> u   : BloomUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

// Fullscreen triangle — no vertex buffer needed
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  // Triangle that covers [-1,1]² in clip space
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f( 3.0,  1.0),
    vec2f(-1.0,  1.0),
  );
  let p = positions[vi];
  var out: VOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv  = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// ── Luminosity threshold (BloomSetupCommon port) ──────────────────────────────
//
//   float4 BloomSetupCommon(UV, ViewportUV, ExposureScaleMiddleGreyLumValue) {
//     half3 LinearColor = SceneColor.rgb;
//     half TotalLuminance = Luminance(LinearColor) * ExposureScale;
//     half BloomLuminance = TotalLuminance - BloomThreshold;
//     half BloomAmount    = saturate(BloomLuminance * 0.5f);
//     return float4(BloomAmount * LinearColor, 0) * View.PreExposure;
//   }
//
// exposureScale is fixed at 1.0 (WebGPU has no EyeAdaptation buffer).
// View.PreExposure is folded into bloomScale at composite stage.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let sceneColor = textureSample(src, smp, in.uv);
  let linearColor = sceneColor.rgb;

  // TotalLuminance = luminance(rgb) * exposureScale
  let totalLum   = luminance(linearColor) * u.exposureScale;
  // BloomLuminance = TotalLuminance - BloomThreshold
  let bloomLum   = totalLum - u.threshold;
  // BloomAmount = saturate(BloomLuminance * 0.5)  ← UE5 formula
  let bloomAmt   = clamp(bloomLum * 0.5, 0.0, 1.0);

  return vec4f(bloomAmt * linearColor, sceneColor.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — SEPARABLE GAUSSIAN BLUR (horizontal pass)
// Source: upstream/pixijs-filters/src/kawase-blur/kawase-blur.wgsl
//         upstream/unreal-renderer-ue5/…/PostProcessWeightedSampleSum (concept)
//
// 7-tap symmetric Gaussian kernel, weights from σ = blurRadius (UE
// equivalent: KernelSizePercent × BloomSizeScale for a given downsample level).
// Two direction variants share the same entry-point; the step direction
// is encoded in the uniform.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GAUSSIAN_BLUR = /* wgsl */`
${WGSL_BLOOM_UNIFORMS}

@group(0) @binding(0) var<uniform> u      : BloomUniforms;
@group(0) @binding(1) var          smp    : sampler;
@group(0) @binding(2) var          src    : texture_2d<f32>;
// horizontal (dir==0) : step = (invWidth * blurRadius, 0)
// vertical   (dir==1) : step = (0, invHeight * blurRadius)
@group(0) @binding(3) var<uniform> dir    : u32;   // 0 = H, 1 = V

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f( 3.0,  1.0),
    vec2f(-1.0,  1.0),
  );
  let p = positions[vi];
  var out: VOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv  = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// ── Symmetric 7-tap Gaussian weights ─────────────────────────────────────────
// Derived from σ = blurRadius via w_i = exp(-0.5 * (i/σ)²).
// UE5 PostProcessWeightedSampleSum computes weights on CPU then uploads;
// here we derive inline for portability (7 taps, hardcoded σ = u.blurRadius).
//
// tap offsets: [-3, -2, -1, 0, 1, 2, 3]
// unnormalized weights computed per fragment — acceptable for interactive use.
fn gaussWeight(offset: f32, sigma: f32) -> f32 {
  let s2 = sigma * sigma;
  return exp(-0.5 * (offset * offset) / max(s2, 0.0001));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let sigma = max(u.blurRadius, 0.1);
  // step direction
  var step: vec2f;
  if (dir == 0u) {
    step = vec2f(u.invWidth * u.blurRadius, 0.0);
  } else {
    step = vec2f(0.0, u.invHeight * u.blurRadius);
  }

  // 7-tap symmetric Gaussian convolution
  // tap offsets: -3 .. +3
  var acc    = vec4f(0.0);
  var wTotal = 0.0;
  for (var i: i32 = -3; i <= 3; i++) {
    let w   = gaussWeight(f32(i), sigma);
    let uv  = in.uv + step * f32(i);
    acc    += textureSample(src, smp, uv) * w;
    wTotal += w;
  }
  return acc / max(wTotal, 1e-6);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3 — COMPOSITE
// Source: upstream/pixijs-filters/src/advanced-bloom/advanced-bloom.wgsl
//   mainFragment(): color × uBrightness + bloomColor × uBloomScale
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */`
${WGSL_BLOOM_UNIFORMS}

@group(0) @binding(0) var<uniform> u        : BloomUniforms;
@group(0) @binding(1) var          smp      : sampler;
@group(0) @binding(2) var          sceneTex : texture_2d<f32>;
@group(0) @binding(3) var          bloomTex : texture_2d<f32>;

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f( 3.0,  1.0),
    vec2f(-1.0,  1.0),
  );
  let p = positions[vi];
  var out: VOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv  = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

// ── AT AdvancedBloomFilter composite port ─────────────────────────────────────
// WGSL original (advanced-bloom.wgsl):
//   var color      = textureSample(uTexture, uSampler, uv);
//   color          = vec4f(color.rgb * uBrightness, color.a);
//   var bloomColor = vec4f(textureSample(uMapTexture, uSampler, uv).rgb, 0.0);
//   bloomColor     = vec4f(bloomColor.rgb * uBloomScale, bloomColor.a);
//   return color + bloomColor;
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let scene     = textureSample(sceneTex, smp, in.uv);
  let bloom     = textureSample(bloomTex, smp, in.uv);

  let outColor  = vec3f(scene.rgb * u.brightness) + vec3f(bloom.rgb * u.bloomScale);
  return vec4f(clamp(outColor, vec3f(0.0), vec3f(1.0)), scene.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Uniform packing
// ─────────────────────────────────────────────────────────────────────────────

/** 12 × f32 = 48 bytes, std140 aligned (4-float rows). */
const UNIFORM_F32_COUNT = 12;

function packUniforms(p: Required<ATBloomParams>, w: number, h: number): Float32Array {
  const buf = new Float32Array(UNIFORM_F32_COUNT);
  buf[0]  = p.threshold;
  buf[1]  = 1.0;           // exposureScale (fixed)
  buf[2]  = p.blurRadius;
  buf[3]  = 0.0;           // _pad0
  buf[4]  = p.bloomScale;
  buf[5]  = p.brightness;
  buf[6]  = 0.0;           // _pad1
  buf[7]  = 0.0;           // _pad2
  buf[8]  = 1.0 / Math.max(w, 1);
  buf[9]  = 1.0 / Math.max(h, 1);
  buf[10] = 0.0;           // _pad3
  buf[11] = 0.0;           // _pad4
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATBloomPostProcess class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU UnrealBloom post-process pipeline ported from ActiveTheory / UE5.
 *
 * Pipeline: luminosity threshold → separable gaussian blur (N passes) → composite.
 */
export class ATBloomPostProcess {
  private readonly device        : GPUDevice;
  private readonly format        : GPUTextureFormat;
  private width                  : number;
  private height                 : number;

  // GPU resources
  private uniformBuf             : GPUBuffer;
  private dirHBuf                : GPUBuffer;   // direction=0 (horizontal)
  private dirVBuf                : GPUBuffer;   // direction=1 (vertical)
  private sampler                : GPUSampler;

  // Pipelines
  private luminosityPipeline     : GPURenderPipeline;
  private blurPipeline           : GPURenderPipeline;
  private compositePipeline      : GPURenderPipeline;

  // Bind group layouts
  private luminosityBGL          : GPUBindGroupLayout;
  private blurBGL                : GPUBindGroupLayout;
  private compositeBGL           : GPUBindGroupLayout;

  // Intermediate textures (ping-pong for blur)
  private brightTex              : GPUTexture;    // after luminosity pass
  private blurTexA               : GPUTexture;    // blur ping
  private blurTexB               : GPUTexture;    // blur pong

  // Current resolved params
  private params                 : Required<ATBloomParams>;

  // ── private constructor ────────────────────────────────────────────────────

  private constructor(
    device           : GPUDevice,
    format           : GPUTextureFormat,
    width            : number,
    height           : number,
    uniformBuf       : GPUBuffer,
    dirHBuf          : GPUBuffer,
    dirVBuf          : GPUBuffer,
    sampler          : GPUSampler,
    luminosityPipeline: GPURenderPipeline,
    blurPipeline     : GPURenderPipeline,
    compositePipeline: GPURenderPipeline,
    luminosityBGL    : GPUBindGroupLayout,
    blurBGL          : GPUBindGroupLayout,
    compositeBGL     : GPUBindGroupLayout,
    brightTex        : GPUTexture,
    blurTexA         : GPUTexture,
    blurTexB         : GPUTexture,
    params           : Required<ATBloomParams>,
  ) {
    this.device             = device;
    this.format             = format;
    this.width              = width;
    this.height             = height;
    this.uniformBuf         = uniformBuf;
    this.dirHBuf            = dirHBuf;
    this.dirVBuf            = dirVBuf;
    this.sampler            = sampler;
    this.luminosityPipeline = luminosityPipeline;
    this.blurPipeline       = blurPipeline;
    this.compositePipeline  = compositePipeline;
    this.luminosityBGL      = luminosityBGL;
    this.blurBGL            = blurBGL;
    this.compositeBGL       = compositeBGL;
    this.brightTex          = brightTex;
    this.blurTexA           = blurTexA;
    this.blurTexB           = blurTexB;
    this.params             = params;
  }

  // ── static factory ─────────────────────────────────────────────────────────

  /**
   * Async factory — compiles all three WGSL pipelines and allocates
   * intermediate textures.
   */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<ATBloomPostProcess> {

    const defaults: Required<ATBloomParams> = {
      threshold  : 0.5,
      bloomScale : 1.0,
      brightness : 1.0,
      blurPasses : 2,
      blurRadius : 2.0,
    };

    // ── Uniforms buffer (48 bytes) ───────────────────────────────────────────
    const uniformBuf = device.createBuffer({
      label : 'at-bloom-uniforms',
      size  : UNIFORM_F32_COUNT * 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuf, 0, packUniforms(defaults, width, height));

    // ── Direction buffers for blur pass (4 bytes each) ───────────────────────
    const dirHBuf = device.createBuffer({
      label : 'at-bloom-dir-h',
      size  : 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const dirVBuf = device.createBuffer({
      label : 'at-bloom-dir-v',
      size  : 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dirHBuf, 0, new Uint32Array([0]));
    device.queue.writeBuffer(dirVBuf, 0, new Uint32Array([1]));

    // ── Sampler (bilinear clamp, matching UE5 SF_Bilinear + AM_Clamp) ────────
    const sampler = device.createSampler({
      label       : 'at-bloom-sampler',
      magFilter   : 'linear',
      minFilter   : 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // ── Intermediate textures ────────────────────────────────────────────────
    const texDesc: GPUTextureDescriptor = {
      size  : { width, height },
      format: 'rgba16float',   // HDR intermediate — matches UE5 bloom chain
      usage : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    };
    const brightTex = device.createTexture({ ...texDesc, label: 'at-bloom-bright' });
    const blurTexA  = device.createTexture({ ...texDesc, label: 'at-bloom-blur-a' });
    const blurTexB  = device.createTexture({ ...texDesc, label: 'at-bloom-blur-b' });

    // ── Bind group layouts ───────────────────────────────────────────────────
    // Stage 1: uniforms + sampler + srcTex
    const luminosityBGL = device.createBindGroupLayout({
      label  : 'at-bloom-luminosity-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
          buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
      ],
    });

    // Stage 2: uniforms + sampler + srcTex + dir
    const blurBGL = device.createBindGroupLayout({
      label  : 'at-bloom-blur-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
          buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          buffer : { type: 'uniform' } },
      ],
    });

    // Stage 3: uniforms + sampler + sceneTex + bloomTex
    const compositeBGL = device.createBindGroupLayout({
      label  : 'at-bloom-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
          buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
      ],
    });

    // ── Pipeline helper ──────────────────────────────────────────────────────
    function makePipeline(
      label    : string,
      wgsl     : string,
      bgl      : GPUBindGroupLayout,
      outFormat: GPUTextureFormat,
    ): GPURenderPipeline {
      const module = device.createShaderModule({ label, code: wgsl });
      return device.createRenderPipeline({
        label,
        layout    : device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex    : { module, entryPoint: 'vs_main' },
        fragment  : { module, entryPoint: 'fs_main',
          targets : [{ format: outFormat }] },
        primitive : { topology: 'triangle-list' },
      });
    }

    // ── Compile the three pipelines ──────────────────────────────────────────
    const luminosityPipeline = makePipeline(
      'at-bloom-luminosity', WGSL_LUMINOSITY, luminosityBGL, 'rgba16float');

    const blurPipeline = makePipeline(
      'at-bloom-blur', WGSL_GAUSSIAN_BLUR, blurBGL, 'rgba16float');

    const compositePipeline = makePipeline(
      'at-bloom-composite', WGSL_COMPOSITE, compositeBGL, format);

    return new ATBloomPostProcess(
      device, format, width, height,
      uniformBuf, dirHBuf, dirVBuf, sampler,
      luminosityPipeline, blurPipeline, compositePipeline,
      luminosityBGL, blurBGL, compositeBGL,
      brightTex, blurTexA, blurTexB,
      defaults,
    );
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Update bloom parameters and flush uniforms to the GPU.
   * Safe to call every frame (diffing is not done — always writes).
   */
  setParams(partial: ATBloomParams): void {
    Object.assign(this.params, partial);
    this.device.queue.writeBuffer(
      this.uniformBuf, 0,
      packUniforms(this.params, this.width, this.height),
    );
  }

  /**
   * Resize internal textures. Call when the canvas changes size.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;

    // Destroy old textures and recreate
    this.brightTex.destroy();
    this.blurTexA.destroy();
    this.blurTexB.destroy();

    const texDesc: GPUTextureDescriptor = {
      size  : { width, height },
      format: 'rgba16float',
      usage : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    };
    this.brightTex = this.device.createTexture({ ...texDesc, label: 'at-bloom-bright' });
    this.blurTexA  = this.device.createTexture({ ...texDesc, label: 'at-bloom-blur-a' });
    this.blurTexB  = this.device.createTexture({ ...texDesc, label: 'at-bloom-blur-b' });

    // Flush updated resolution into uniforms
    this.device.queue.writeBuffer(
      this.uniformBuf, 0,
      packUniforms(this.params, width, height),
    );
  }

  /**
   * Execute the full UnrealBloom pipeline for one frame.
   *
   * @param encoder  - Active GPUCommandEncoder (caller owns submit).
   * @param sceneTex - The scene's rendered GPUTexture (must be TEXTURE_BINDING).
   * @param dstView  - Destination GPUTextureView for the final composited output.
   */
  render(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    const { device, sampler, params } = this;

    const sceneView  = sceneTex.createView();
    const brightView = this.brightTex.createView();
    const blurAView  = this.blurTexA.createView();
    const blurBView  = this.blurTexB.createView();

    // ── Pass 1: luminosity threshold ─────────────────────────────────────────
    // src: sceneView → dst: brightTex
    // Port of UE5 BloomSetupPS / BloomSetupCS (PostProcessBloom.usf)
    {
      const bg = device.createBindGroup({
        label  : 'at-bloom-lum-bg',
        layout : this.luminosityBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: sceneView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label           : 'at-bloom-luminosity-pass',
        colorAttachments: [{
          view    : brightView,
          loadOp  : 'clear',
          storeOp : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(this.luminosityPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Passes 2..N: separable Gaussian blur ─────────────────────────────────
    // Iterates blurPasses times, each time doing H then V.
    // ping: blurTexA, pong: blurTexB — source alternates each half-pass.
    // First half-pass reads from brightTex.
    //
    // Port of UE5 AddGaussianBloomPasses() → AddGaussianBlurPass() →
    //   PostProcessWeightedSampleSum (H + V two-pass separable).
    // AT AdvancedBloomFilter uses KawaseBlurFilter (similar multi-pass concept).
    {
      const blurPasses = Math.max(1, Math.min(params.blurPasses, 4));

      // State for ping-pong
      let readTex  : GPUTexture = this.brightTex;
      let writeTex : GPUTexture = this.blurTexA;
      let useB     = false;

      for (let p = 0; p < blurPasses; p++) {
        // Horizontal pass
        {
          const bg = device.createBindGroup({
            label  : `at-bloom-blur-h-bg-${p}`,
            layout : this.blurBGL,
            entries: [
              { binding: 0, resource: { buffer: this.uniformBuf } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: readTex.createView() },
              { binding: 3, resource: { buffer: this.dirHBuf } },
            ],
          });
          const writeView = writeTex.createView();
          const pass = encoder.beginRenderPass({
            label           : `at-bloom-blur-h-pass-${p}`,
            colorAttachments: [{
              view    : writeView,
              loadOp  : 'clear',
              storeOp : 'store',
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            }],
          });
          pass.setPipeline(this.blurPipeline);
          pass.setBindGroup(0, bg);
          pass.draw(3);
          pass.end();
        }
        // Swap ping-pong for vertical pass
        readTex  = writeTex;
        writeTex = useB ? this.blurTexA : this.blurTexB;
        useB     = !useB;

        // Vertical pass
        {
          const bg = device.createBindGroup({
            label  : `at-bloom-blur-v-bg-${p}`,
            layout : this.blurBGL,
            entries: [
              { binding: 0, resource: { buffer: this.uniformBuf } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: readTex.createView() },
              { binding: 3, resource: { buffer: this.dirVBuf } },
            ],
          });
          const writeView = writeTex.createView();
          const pass = encoder.beginRenderPass({
            label           : `at-bloom-blur-v-pass-${p}`,
            colorAttachments: [{
              view    : writeView,
              loadOp  : 'clear',
              storeOp : 'store',
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            }],
          });
          pass.setPipeline(this.blurPipeline);
          pass.setBindGroup(0, bg);
          pass.draw(3);
          pass.end();
        }
        // Swap for next iteration's horizontal read
        readTex  = writeTex;
        writeTex = useB ? this.blurTexA : this.blurTexB;
        useB     = !useB;
      }

      // Final blur result is in readTex
      // ── Pass N+1: composite ───────────────────────────────────────────────
      // Port of AT AdvancedBloomFilter.wgsl mainFragment():
      //   color × uBrightness + bloomColor × uBloomScale
      {
        const bg = device.createBindGroup({
          label  : 'at-bloom-composite-bg',
          layout : this.compositeBGL,
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuf } },
            { binding: 1, resource: sampler },
            { binding: 2, resource: sceneView },
            { binding: 3, resource: readTex.createView() },
          ],
        });
        const pass = encoder.beginRenderPass({
          label           : 'at-bloom-composite-pass',
          colorAttachments: [{
            view    : dstView,
            loadOp  : 'clear',
            storeOp : 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        });
        pass.setPipeline(this.compositePipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
    }
  }

  /**
   * Release all GPU resources. The instance must not be used after calling
   * this method.
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.dirHBuf.destroy();
    this.dirVBuf.destroy();
    this.brightTex.destroy();
    this.blurTexA.destroy();
    this.blurTexB.destroy();
  }

  // ── Diagnostics ────────────────────────────────────────────────────────────

  /** Returns the three WGSL shader sources for inspection / hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      luminosity : WGSL_LUMINOSITY,
      gaussianBlur: WGSL_GAUSSIAN_BLUR,
      composite  : WGSL_COMPOSITE,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — integrates with BloomVariants.ts species params
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an ATBloomPostProcess seeded from BloomParams (BloomVariants.ts schema).
 *
 * @param device   - WebGPU device.
 * @param format   - Swapchain texture format.
 * @param width    - Viewport width.
 * @param height   - Viewport height.
 * @param species  - Optional species key for BloomVariants lookup.
 */
export async function createATBloomForSpecies(
  device  : GPUDevice,
  format  : GPUTextureFormat,
  width   : number,
  height  : number,
  bloomParams?: { bloomStrength: number; bloomRadius: number; luminosityThreshold: number },
): Promise<ATBloomPostProcess> {
  const bloom = await ATBloomPostProcess.create(device, format, width, height);
  if (bloomParams) {
    bloom.setParams({
      threshold  : bloomParams.luminosityThreshold,
      bloomScale : bloomParams.bloomStrength,
      blurRadius : bloomParams.bloomRadius * 4.0,  // map [0,1] → [0,4] texels
    });
  }
  return bloom;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

export const AT_BLOOM_WGSL = {
  /** Shared uniform struct (BloomUniforms). */
  uniforms     : WGSL_BLOOM_UNIFORMS,
  /** Rec.709 luminance helper fn. */
  luminance    : WGSL_LUMINANCE,
  /** Full luminosity threshold vertex+fragment shader. */
  luminosity   : WGSL_LUMINOSITY,
  /** Separable Gaussian blur vertex+fragment shader (requires dir uniform). */
  gaussianBlur : WGSL_GAUSSIAN_BLUR,
  /** Scene + bloom composite vertex+fragment shader. */
  composite    : WGSL_COMPOSITE,
} as const;
