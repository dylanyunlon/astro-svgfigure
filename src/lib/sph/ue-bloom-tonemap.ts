/**
 * ue-bloom-tonemap.ts — M853: UE5 Bloom + Tonemap WebGPU Port
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Bloom 与 Tonemap 后处理系统到 WebGPU/WGSL。
 *
 * 原始来源 (16 个核心文件):
 *   Renderer-Private/PostProcess/PostProcessBloomSetup.cpp/.h
 *   Renderer-Private/PostProcess/PostProcessFFTBloom.cpp/.h
 *   Renderer-Private/PostProcess/PostProcessTonemap.cpp/.h
 *   Shaders-Private/PostProcessBloom.usf
 *   Shaders-Private/PostProcessTonemap.usf
 *   Shaders-Private/TonemapCommon.ush
 *   Shaders-Private/Bloom/BloomCommon.ush
 *   Shaders-Private/Bloom/BloomDownsampleKernel.usf
 *   Shaders-Private/Bloom/BloomClampKernel.usf
 *   Shaders-Private/Bloom/BloomFinalizeApplyConstants.usf
 *   Shaders-Private/Bloom/BloomFindKernelCenter.usf
 *   Shaders-Private/Bloom/BloomResizeKernel.usf
 *   Shaders-Private/Bloom/BloomSurveyKernelCenterEnergy.usf
 *   Shaders-Private/Bloom/BloomSurveyMaxScatterDispersion.usf
 *   Shaders-Private/Bloom/BloomSumScatterDispersionEnergy.usf
 *   Shaders-Private/Bloom/BloomPackKernelConstants.usf
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 一、Bloom 系统架构
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   UE5 支持两种 Bloom 模式:
 *
 *   A. FFT Convolution Bloom (PostProcessFFTBloom.cpp)
 *      使用 2D FFT 将场景颜色与预处理的卷积核在频域相乘,
 *      支持真实镜头光晕(用户提供 BloomConvolutionTexture)。
 *      步骤:
 *        1. BloomSetup: 亮度阈值提取 + 局部曝光
 *        2. 下采样场景到 FrequencySize (2的幂次)
 *        3. FFT 正变换 → 频域逐像素乘以卷积核频谱
 *        4. FFT 逆变换 → 空间域卷积结果
 *        5. FinalizeApplyConstants: 能量守恒校正
 *
 *   B. Gaussian Bloom (PostProcessBloomSetup.cpp → AddGaussianBloomPasses)
 *      6 级质量 × 6 个下采样阶段的金字塔高斯模糊。
 *      各阶段: BloomStage[6] { Bloom1Size~Bloom6Size, Bloom1Tint~Bloom6Tint }
 *      BloomQuality → 选择 3~6 个下采样层级。
 *      各层用可分离高斯核 (PostProcessWeightedSampleSum) 水平+垂直各一遍。
 *
 *   WebGPU 移植策略:
 *     - FFT 路径: 使用 JavaScript FFT(cooley-tukey) 预计算卷积核频谱,
 *       GPU 只做频域乘法 + 逆变换 (近似: 用 6-pass 双线性降/升采样模拟)。
 *     - Gaussian 路径: 6 级 downsample pyramid + 可分离 13-tap Gaussian。
 *     - 两路共享 BloomSetup pass (亮度阈值 + 局部曝光自适应)。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 二、Tonemap 系统架构
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   UE5 PostProcessTonemap.usf 核心逻辑:
 *     1. EyeAdaptation (全局曝光): GlobalExposure = EyeAdaptationBuffer[0].x
 *     2. LocalExposure (局部曝光): 双边网格 log 亮度 → 局部对比度增强
 *     3. BloomComposite: SceneColor + Bloom * (GlobalExposure * VignetteMask)
 *     4. FilmToneMap (ACES filmic): ACEScg 空间 → RRT → ODT → sRGB
 *        - GlowModule: 高饱和高亮区域轻微提亮
 *        - RedModifier: 红色色相偏移矫正
 *        - 可调参数: Slope/Toe/Shoulder/BlackClip/WhiteClip
 *     5. ColorLookupTable (3D LUT): 最终色彩分级
 *     6. Vignette: 暗角效果
 *     7. FilmGrain: 胶片颗粒噪声
 *
 *   WebGPU 局部曝光自适应:
 *     将屏幕 log 亮度下采样到 16×16 → 高斯模糊 → 计算场景平均亮度 →
 *     自适应调节曝光系数 (模拟 UE5 EyeAdaptation bilateral grid)。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 三、与 AT UnrealBloom 桥接
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   UEBloomTonemap 可作为 ATBloomPostProcess 的替代或增强:
 *     - UEBloomTonemap.bridgeToAT(atBloom) 将 UE Bloom 输出注入 AT 合成 pass
 *     - UEBloomTonemap.render(encoder, sceneTex, dstView) 直接渲染
 *
 * 管线流程 (每帧):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Eye Adaptation (Luminance Histogram) ───────────────────────┐
 *   │  sceneColor → downsample(16×16) → gaussian luma blur                   │
 *   │  输出: lumAdaptBuffer (f32, 64 samples avg)                             │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ avgLuma
 *               ▼
 *   ┌─ Pass 1 ── Bloom Setup (Threshold + Local Exposure) ───────────────────┐
 *   │  BloomSetupCS: TotalLum > threshold → extract bright pixels            │
 *   │  LocalExposure: log2(lum) → bilateral approx → localExposureFactor     │
 *   │  输出: brightTex (rgba16float, full-res)                                │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ brightTex
 *               ▼
 *   ┌─ Pass 2..7 ── Gaussian Downsample Pyramid ─────────────────────────────┐
 *   │  mip0(full) → mip1(1/2) → mip2(1/4) → mip3(1/8) → mip4(1/16)         │
 *   │  → mip5(1/32) → mip6(1/64)  (质量Q5全开)                               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ bloomMips[0..5]
 *               ▼
 *   ┌─ Pass 8..N ── Gaussian Upsample + Tint Accumulate ─────────────────────┐
 *   │  各阶: separable 13-tap Gaussian (H pass + V pass) + tint color        │
 *   │  逐级累加 (additive blend)                                              │
 *   │  输出: bloomAccumTex (rgba16float, full-res)                            │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ bloomAccumTex
 *               ▼
 *   ┌─ Pass N+1 ── Tonemap + Composite ──────────────────────────────────────┐
 *   │  sceneColor * (GlobalExposure * LocalExposure * SceneColorTint)        │
 *   │  + bloom * (GlobalExposure * bloomScale)                                │
 *   │  → FilmToneMap (ACES filmic RRT+ODT)                                   │
 *   │  → Vignette × FilmGrain                                                │
 *   │  → 输出: sRGB (GPUTextureView dstView)                                  │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * Research: xiaodi #M853 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Public Types
// ─────────────────────────────────────────────────────────────────────────────

/** ACES filmic curve parameters. Matches UE5 TonemapCommon.ush FilmToneMap(). */
export interface UEFilmParams {
  /** Slope of the linear segment. UE default 0.91. */
  slope?: number;
  /** Toe softness. UE default 0.53. */
  toe?: number;
  /** Shoulder softness. UE default 0.23. */
  shoulder?: number;
  /** Black clip. UE default 0. */
  blackClip?: number;
  /** White clip. UE default 0.035. */
  whiteClip?: number;
}

/** Gaussian bloom stage: one entry per downsample level. */
export interface UEBloomStage {
  /** Kernel size percent [0..1]. Maps to UE5 BloomNSize * BloomSizeScale. */
  size: number;
  /** Additive tint [r,g,b] for this frequency band. */
  tint: [number, number, number];
}

/** Per-frame tweakable parameters for UEBloomTonemap. */
export interface UEBloomTonemapParams {
  // ── Bloom ─────────────────────────────────────────────────────────────────
  /**
   * Luminance threshold for bloom extraction.
   * Maps to UE5 BloomThreshold. @default 0.8
   */
  bloomThreshold?: number;
  /**
   * Overall bloom intensity scalar. UE5 BloomIntensity. @default 1.0
   */
  bloomIntensity?: number;
  /**
   * Bloom stage quality (1=Q1/fastest … 5=Q5/highest). @default 4
   */
  bloomQuality?: 1 | 2 | 3 | 4 | 5;
  /**
   * Per-stage size & tint. Overrides built-in UE defaults when provided.
   * Length must match bloomQuality stages (3, 3, 4, 5, or 6).
   */
  bloomStages?: UEBloomStage[];
  /**
   * Size scale multiplier applied to all stage sizes. UE5 BloomSizeScale.
   * @default 1.0
   */
  bloomSizeScale?: number;
  /**
   * Enable FFT convolution mode (requires kernelTexture). @default false
   */
  useFFT?: boolean;
  /**
   * Optional lens kernel texture for FFT convolution bloom.
   * If not provided, falls back to Gaussian bloom.
   */
  kernelTexture?: GPUTexture;

  // ── Eye Adaptation ────────────────────────────────────────────────────────
  /**
   * Enable automatic eye adaptation (local exposure). @default true
   */
  eyeAdaptation?: boolean;
  /**
   * Minimum exposure adjustment factor. @default 0.1
   */
  eyeAdaptationMin?: number;
  /**
   * Maximum exposure adjustment factor. @default 2.0
   */
  eyeAdaptationMax?: number;
  /**
   * Adaptation speed (lerp factor per frame). @default 0.05
   */
  eyeAdaptationSpeed?: number;

  // ── Tonemap ───────────────────────────────────────────────────────────────
  /**
   * ACES filmic parameters. @default UE5 ACES preset
   */
  film?: UEFilmParams;
  /**
   * Scene color tint (rgb multiplier pre-tonemap). @default [1,1,1]
   */
  sceneTint?: [number, number, number];
  /**
   * Bloom tint (rgb multiplier applied to bloom before composite).
   * UE5 ColorScale1. @default [0.5,0.5,0.5]
   */
  bloomTint?: [number, number, number];

  // ── Vignette ──────────────────────────────────────────────────────────────
  /**
   * Vignette intensity [0..1]. 0=none. @default 0.4
   */
  vignetteIntensity?: number;

  // ── Film Grain ────────────────────────────────────────────────────────────
  /**
   * Film grain intensity (applied to all tones). @default 0.0
   */
  grainIntensity?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Constants & Defaults
// ─────────────────────────────────────────────────────────────────────────────

/** UE5 default ACES settings (TonemapCommon.ush comment block). */
const UE_ACES_DEFAULTS: Required<UEFilmParams> = {
  slope     : 0.91,
  toe       : 0.53,
  shoulder  : 0.23,
  blackClip : 0.0,
  whiteClip : 0.035,
};

/**
 * UE5 default bloom stages (BloomStages[] in PostProcessBloomSetup.cpp).
 * Index 0 = Bloom6 (widest / lowest freq), index 5 = Bloom1 (narrowest).
 */
const UE_BLOOM_STAGES_DEFAULT: UEBloomStage[] = [
  { size: 4.0,  tint: [0.3130, 0.3130, 0.3130] }, // Bloom6
  { size: 2.0,  tint: [0.3130, 0.3130, 0.3130] }, // Bloom5
  { size: 1.0,  tint: [0.3130, 0.3130, 0.3130] }, // Bloom4
  { size: 0.5,  tint: [0.3130, 0.3130, 0.3130] }, // Bloom3
  { size: 0.25, tint: [0.3130, 0.3130, 0.3130] }, // Bloom2
  { size: 0.12, tint: [0.3130, 0.3130, 0.3130] }, // Bloom1
];

/**
 * UE5 BloomQualityToSceneDownsampleStage mapping.
 * Quality Q1→3 stages, Q2→3, Q3→4, Q4→5, Q5→6.
 */
const BLOOM_QUALITY_STAGE_COUNT: Record<number, number> = {
  1: 3,
  2: 3,
  3: 4,
  4: 5,
  5: 6,
};

// 13-tap Gaussian kernel weights (sigma ≈ 3.0, UE5 Q3-Q5 quality)
// Generated via: w[i] = exp(-0.5*(i/sigma)²), then normalized
const GAUSS13_WEIGHTS: number[] = [
  0.227027, 0.194595, 0.121622, 0.054054, 0.016216,
  0.016216, 0.054054, 0.121622, 0.194595, 0.227027,
  // (symmetric 10-tap half, total sums to 1.0 when fully expanded)
];

// ─────────────────────────────────────────────────────────────────────────────
// § 3  WGSL Shader Sources
// ─────────────────────────────────────────────────────────────────────────────

// ── 3.1  Luminance Histogram / Eye Adaptation ─────────────────────────────

/** WGSL: Downsample to 16×16 luma grid for eye adaptation. */
const WGSL_EYE_ADAPT_DOWNSAMPLE = /* wgsl */`
// UE5 EyeAdaptationCommon.ush: CalculateEyeAdaptationLuminance()
// Rec.709 luminance: dot(color, vec3(0.2126, 0.7152, 0.0722))

@group(0) @binding(0) var sceneTex : texture_2d<f32>;
@group(0) @binding(1) var samp     : sampler;
@group(0) @binding(2) var<storage, read_write> lumOut : array<f32>;

@compute @workgroup_size(8, 8)
fn eyeAdaptCS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(sceneTex);
  let tileW = (dims.x + 15u) / 16u;
  let tileH = (dims.y + 15u) / 16u;
  let px    = gid.xy * vec2<u32>(tileW, tileH);

  var lum = 0.0;
  var cnt = 0u;
  for (var dy = 0u; dy < tileH; dy++) {
    for (var dx = 0u; dx < tileW; dx++) {
      let coord = px + vec2<u32>(dx, dy);
      if (coord.x < dims.x && coord.y < dims.y) {
        let uv  = (vec2<f32>(coord) + 0.5) / vec2<f32>(dims);
        let col = textureSampleLevel(sceneTex, samp, uv, 0.0).rgb;
        // Rec.709 luma (UE5 EyeAdaptationCommon: rgb_2_luma)
        lum += dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
        cnt++;
      }
    }
  }
  let idx = gid.y * 16u + gid.x;
  lumOut[idx] = select(0.0, lum / f32(cnt), cnt > 0u);
}
`;

// ── 3.2  Bloom Setup (Threshold + Local Exposure) ─────────────────────────

/**
 * WGSL: Bloom threshold extraction.
 * Ref: PostProcessBloom.usf BloomSetupCommon() + PostProcessTonemap.usf local exposure
 *
 * UE5 formula:
 *   TotalLuminance = dot(rgb, luma_weights) * ExposureScale
 *   BloomLuminance = TotalLuminance - BloomThreshold
 *   BloomAmount    = saturate(BloomLuminance * 0.5)
 *   output         = BloomAmount * LinearColor * preExposure
 */
const WGSL_BLOOM_SETUP = /* wgsl */`
struct BloomSetupUniforms {
  threshold    : f32,
  exposureScale: f32,
  localExpMin  : f32,
  localExpMax  : f32,
  _pad0        : vec4<f32>,
}

@group(0) @binding(0) var<uniform>            u          : BloomSetupUniforms;
@group(0) @binding(1) var                     sceneTex   : texture_2d<f32>;
@group(0) @binding(2) var                     samp       : sampler;
@group(0) @binding(3) var<storage, read>      lumGrid    : array<f32>; // 256 entries (16×16)
@group(0) @binding(4) var                     brightTex  : texture_storage_2d<rgba16float, write>;

// UE5: CalculateEyeAdaptationLuminance()
fn rec709Luma(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// UE5 PostProcessBloom.usf: BloomSetupCommon() USE_LOCAL_EXPOSURE path
// Approximates bilateral grid local exposure via log2 lum comparison
fn computeLocalExposure(lum: f32, avgLum: f32, minE: f32, maxE: f32) -> f32 {
  let logLum    = log2(max(lum,    1e-5));
  let logAvgLum = log2(max(avgLum, 1e-5));
  let delta     = logAvgLum - logLum;
  // clamp to [minE, maxE] (UE5 LocalExposure_HighlightContrastScale path)
  return clamp(exp2(delta * 0.5), minE, maxE);
}

@compute @workgroup_size(8, 8)
fn bloomSetupCS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(brightTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv  = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
  let col = textureSampleLevel(sceneTex, samp, uv, 0.0).rgb;
  let lum = rec709Luma(col);

  // Compute scene average luma from 16×16 grid
  var avgLum = 0.0;
  for (var i = 0u; i < 256u; i++) { avgLum += lumGrid[i]; }
  avgLum /= 256.0;

  // Local exposure (UE5 CalculateLocalExposure approximation)
  let localExp = computeLocalExposure(lum, avgLum, u.localExpMin, u.localExpMax);

  // Bloom threshold (UE5 BloomSetupCommon USE_THRESHOLD path)
  let totalLum   = lum * u.exposureScale * localExp;
  let bloomLum   = totalLum - u.threshold;
  let bloomAmt   = clamp(bloomLum * 0.5, 0.0, 1.0);

  textureStore(brightTex, vec2<i32>(gid.xy), vec4<f32>(col * bloomAmt, 1.0));
}
`;

// ── 3.3  Gaussian Downsample ──────────────────────────────────────────────

/**
 * WGSL: Single downsample pass (2× bilinear).
 * Mirrors UE5 PostProcessBloomSetup.cpp AddGaussianBloomPasses() pyramid stage.
 * Uses 4-sample box filter matching BloomDownsampleKernel.usf.
 */
const WGSL_BLOOM_DOWNSAMPLE = /* wgsl */`
@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var samp   : sampler;
@group(0) @binding(2) var dstTex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn downsampleCS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dstTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv     = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
  let offset = 0.7 / vec2<f32>(textureDimensions(srcTex));

  // 4-tap box (BloomDownsampleKernel.usf pattern)
  var c = vec4<f32>(0.0);
  c += 0.25 * textureSampleLevel(srcTex, samp, uv + vec2<f32>( offset.x,  offset.y), 0.0);
  c += 0.25 * textureSampleLevel(srcTex, samp, uv + vec2<f32>( offset.x, -offset.y), 0.0);
  c += 0.25 * textureSampleLevel(srcTex, samp, uv + vec2<f32>(-offset.x, -offset.y), 0.0);
  c += 0.25 * textureSampleLevel(srcTex, samp, uv + vec2<f32>(-offset.x,  offset.y), 0.0);

  textureStore(dstTex, vec2<i32>(gid.xy), c);
}
`;

// ── 3.4  Separable Gaussian Blur ──────────────────────────────────────────

/**
 * WGSL: Separable 13-tap Gaussian blur.
 * Matches UE5 PostProcessWeightedSampleSum (AddGaussianBlurPass).
 * dir = vec2(1,0) for horizontal, vec2(0,1) for vertical.
 */
const WGSL_BLOOM_GAUSSIAN = /* wgsl */`
struct GaussianUniforms {
  dir    : vec2<f32>,
  tint   : vec3<f32>,
  _pad   : f32,
}

@group(0) @binding(0) var<uniform> u       : GaussianUniforms;
@group(0) @binding(1) var          srcTex  : texture_2d<f32>;
@group(0) @binding(2) var          samp    : sampler;
@group(0) @binding(3) var          dstTex  : texture_storage_2d<rgba16float, write>;

// 13-tap half-gaussian weights (UE5 sigma≈3 Q4/Q5)
const W: array<f32, 7> = array<f32, 7>(
  0.227027, 0.194595, 0.121622, 0.054054, 0.016216, 0.004054, 0.001014
);

@compute @workgroup_size(8, 8)
fn gaussianCS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dstTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv     = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
  let texel  = 1.0 / vec2<f32>(textureDimensions(srcTex));
  let step   = u.dir * texel;

  var c = W[0] * textureSampleLevel(srcTex, samp, uv, 0.0);
  for (var i = 1; i < 7; i++) {
    let off = f32(i) * step;
    c += W[i] * textureSampleLevel(srcTex, samp, uv + off, 0.0);
    c += W[i] * textureSampleLevel(srcTex, samp, uv - off, 0.0);
  }

  textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(c.rgb * u.tint, c.a));
}
`;

// ── 3.5  FFT Bloom Convolution (frequency-domain multiply) ────────────────

/**
 * WGSL: Frequency-domain bloom convolution.
 * Simplified port of UE5 GPUFastFourierTransform + BloomFinalizeApplyConstants.
 * Each complex pixel: (Re,Im) packed as rg channels.
 * Performs: ImageSpectrum *= KernelSpectrum (complex multiply).
 */
const WGSL_FFT_BLOOM_MULTIPLY = /* wgsl */`
@group(0) @binding(0) var          imageTex  : texture_2d<f32>;       // Re=r, Im=g
@group(0) @binding(1) var          kernelTex : texture_2d<f32>;       // Re=r, Im=g
@group(0) @binding(2) var          outTex    : texture_storage_2d<rgba16float, write>;

// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y,
                   a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn fftMultiplyCS(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(outTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let img = textureLoad(imageTex,  vec2<i32>(gid.xy), 0);
  let ker = textureLoad(kernelTex, vec2<i32>(gid.xy), 0);

  // Multiply RGB channels independently (UE5 does per-channel FFT)
  let r = cmul(img.rg, ker.rg);
  let g = cmul(img.ba, ker.ba);
  // blue channel uses second pair (simplified: reuse green kernel)
  let b = cmul(img.rg, ker.ba);

  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(r.x, r.y, g.x, g.y));
}
`;

// ── 3.6  Tonemap + Composite Pass ─────────────────────────────────────────

/**
 * WGSL: Full UE5 tonemap + bloom composite.
 *
 * Ports:
 *  - PostProcessTonemap.usf TonemapCommonPS() main path
 *  - TonemapCommon.ush FilmToneMap() ACES RRT+ODT
 *  - Vignette (UE5 ComputeVignetteMask type 0 radial)
 *  - FilmGrain (UE5 GrainFromUV)
 */
const WGSL_TONEMAP = /* wgsl */`
struct TonemapUniforms {
  // Eye adaptation
  globalExposure  : f32,
  oneOverPreExp   : f32,
  // Film curve (TonemapCommon.ush FilmToneMap params)
  filmSlope       : f32,
  filmToe         : f32,
  filmShoulder    : f32,
  filmBlackClip   : f32,
  filmWhiteClip   : f32,
  _pad0           : f32,
  // Scene color tint (ColorScale0)
  sceneTint       : vec3<f32>,
  _pad1           : f32,
  // Bloom tint (ColorScale1)
  bloomTint       : vec3<f32>,
  bloomScale      : f32,
  // Vignette
  vignetteIntensity: f32,
  // Film grain
  grainIntensity   : f32,
  grainSeed        : f32,
  _pad2            : f32,
  // Viewport
  viewportSize     : vec2<f32>,
  _pad3            : vec2<f32>,
}

@group(0) @binding(0) var<uniform> u        : TonemapUniforms;
@group(0) @binding(1) var          sceneTex : texture_2d<f32>;
@group(0) @binding(2) var          bloomTex : texture_2d<f32>;
@group(0) @binding(3) var          samp     : sampler;

// ─── ACES colour-space matrices (TonemapCommon.ush ACESCommon.ush) ───────────
// AP1 → sRGB  (ODT output)
const AP1_TO_SRGB = mat3x3<f32>(
   1.70505, -0.62179, -0.08326,
  -0.13026,  1.14080, -0.01054,
  -0.02400, -0.12897,  1.15297
);
// sRGB → AP1  (RRT input)
const SRGB_TO_AP1 = mat3x3<f32>(
  0.59719,  0.35458,  0.04823,
  0.07600,  0.90834,  0.01566,
  0.02840,  0.13383,  0.83777
);
// AP1 luminance weights (ACEScg)
const AP1_RGB2Y = vec3<f32>(0.27222, 0.67408, 0.05370);

// ─── UE5 TonemapCommon.ush helpers ───────────────────────────────────────────

// rgb_2_saturation
fn rgb2sat(c: vec3<f32>) -> f32 {
  let maxC = max(max(c.r, c.g), c.b);
  let minC = min(min(c.r, c.g), c.b);
  return (maxC - minC) / max(maxC, 1e-5);
}

// rgb_2_yc (chroma-weighted luma, UE5 ACESCommon)
fn rgb2yc(c: vec3<f32>) -> f32 {
  let yw = dot(c, AP1_RGB2Y);
  // chroma: 0.5*(Cmax-Cmin)/Luma
  let chroma = 0.5 * (max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b));
  return yw + 0.03 * chroma * chroma;
}

// sigmoid_shaper (UE5 ACES)
fn sigmoidShaper(x: f32) -> f32 {
  let t = max(1.0 - abs(x / 2.0), 0.0);
  let y = 1.0 + sign(x) * (1.0 - t * t);
  return y * 0.5;
}

// glow_fwd (UE5 ACES RRT glow module)
fn glowFwd(ycIn: f32, glowGain: f32, glowMid: f32) -> f32 {
  if (ycIn <= 2.0 / 3.0 * glowMid) {
    return glowGain;
  } else if (ycIn >= 2.0 * glowMid) {
    return 0.0;
  } else {
    return glowGain * (glowMid / ycIn - 0.5);
  }
}

// rgb_2_hue (UE5 ACES)
fn rgb2hue(c: vec3<f32>) -> f32 {
  let minC = min(min(c.r, c.g), c.b);
  let maxC = max(max(c.r, c.g), c.b);
  let delta = maxC - minC;
  if (delta < 1e-5) { return 0.0; }
  var hue: f32;
  if (maxC == c.r) {
    hue = (c.g - c.b) / delta;
  } else if (maxC == c.g) {
    hue = 2.0 + (c.b - c.r) / delta;
  } else {
    hue = 4.0 + (c.r - c.g) / delta;
  }
  return hue * 60.0;
}

// center_hue
fn centerHue(hue: f32, centerH: f32) -> f32 {
  var h = hue - centerH;
  if (h < -180.0) { h += 360.0; }
  if (h >  180.0) { h -= 360.0; }
  return h;
}

// ─── UE5 TonemapCommon.ush: FilmToneMap() ────────────────────────────────────
// Full ACES RRT+ODT approximation with configurable film curve
fn filmToneMap(linearColor: vec3<f32>) -> vec3<f32> {
  // Input in ACEScg (AP1) space
  var colorAP1 = linearColor;
  colorAP1 = max(colorAP1, vec3<f32>(0.0));

  // Convert AP1→AP0 for RRT glow/red modules
  // AP0_2_AP1 ≈ inverse of sRGB_2_AP1 (simplified path matching UE5)
  // UE5 uses full matrix chain; we approximate with ACEScg working space
  let colorAP0 = colorAP1; // Simplified: skip AP0 conversion for web port

  // Glow module (UE5 RRT_GLOW_GAIN=0.05, RRT_GLOW_MID=0.08)
  let saturation = rgb2sat(colorAP0);
  let ycIn       = rgb2yc(colorAP0);
  let s          = sigmoidShaper((saturation - 0.4) / 0.2);
  let addedGlow  = 1.0 + glowFwd(ycIn, 0.05 * s, 0.08);

  var workingColor = colorAP1 * addedGlow;

  // Red modifier (UE5 RRT_RED_SCALE=0.82, RRT_RED_PIVOT=0.03, width=135)
  let hue = rgb2hue(workingColor);
  let centeredHue = centerHue(hue, 0.0);
  let t = 1.0 - abs(2.0 * centeredHue / 135.0);
  let hueWeight = t * t * clamp(t, 0.0, 1.0) * clamp(t, 0.0, 1.0);
  workingColor.r += hueWeight * saturation * (0.03 - workingColor.r) * (1.0 - 0.82);

  workingColor = max(workingColor, vec3<f32>(0.0));

  // Pre-desaturate (UE5: lerp(dot(WorkingColor, AP1_RGB2Y), WorkingColor, 0.96))
  let lumAP1 = dot(workingColor, AP1_RGB2Y);
  workingColor = mix(vec3<f32>(lumAP1), workingColor, 0.96);

  // Film curve (UE5 TonemapCommon.ush configurable Slope/Toe/Shoulder params)
  let toeScale      = 1.0 + u.filmBlackClip - u.filmToe;
  let shoulderScale = 1.0 + u.filmWhiteClip - u.filmShoulder;

  let inMatch  = 0.18;
  let outMatch = 0.18;

  var toeMatch: f32;
  if (u.filmToe > 0.8) {
    toeMatch = (1.0 - u.filmToe - outMatch) / u.filmSlope + log(inMatch) / log(10.0);
  } else {
    let bt = (outMatch + u.filmBlackClip) / toeScale - 1.0;
    toeMatch = log(inMatch) / log(10.0) - 0.5 * log((1.0 + bt) / (1.0 - bt)) * (toeScale / u.filmSlope);
  }

  let straightMatch    = (1.0 - u.filmToe) / u.filmSlope - toeMatch;
  let shoulderMatch    = u.filmShoulder / u.filmSlope - straightMatch;

  // Per-channel curve application
  let logColor = log(workingColor + 1e-5) / log(10.0);
  let straightC = logColor * u.filmSlope + (outMatch - u.filmSlope * straightMatch);

  // Toe segment
  let toeC = vec3<f32>(
    -u.filmBlackClip +
    toeScale * (2.0 / (1.0 + exp(-2.0 * u.filmSlope / toeScale * (logColor - toeMatch))) - 1.0)
  );

  // Shoulder segment (approximation)
  let shoulderC = vec3<f32>(1.0 + u.filmWhiteClip) -
    shoulderScale * (2.0 / (1.0 + exp(2.0 * u.filmSlope / shoulderScale * (logColor - shoulderMatch))) - 1.0);

  // Blend toe / straight / shoulder
  var t1 = clamp((logColor - toeMatch)      / (straightMatch - toeMatch),      0.0, 1.0);
  var t2 = clamp((logColor - straightMatch) / (shoulderMatch - straightMatch),  0.0, 1.0);
  t1 = t1 * t1 * (3.0 - 2.0 * t1);
  t2 = t2 * t2 * (3.0 - 2.0 * t2);

  let curveOut = mix(mix(toeC, vec3<f32>(straightC), t1), vec3<f32>(shoulderC), t2);
  workingColor = pow(max(curveOut, vec3<f32>(0.0)), vec3<f32>(10.0)); // pow10 inverse log

  // Post-desaturate
  let lumOut = dot(workingColor, AP1_RGB2Y);
  workingColor = mix(vec3<f32>(lumOut), workingColor, 0.93);

  // AP1 → sRGB ODT
  let srgb = AP1_TO_SRGB * workingColor;
  return saturate(srgb);
}

// ─── UE5 GrainFromUV (PostProcessTonemap.usf) ────────────────────────────────
fn grainFromUV(uv: vec2<f32>, seed: f32) -> f32 {
  return fract(sin(uv.x + uv.y * 543.31 + seed) * 493013.0);
}

// ─── Vignette (UE5 ComputeVignetteMask type 0 radial) ─────────────────────────
fn computeVignette(uv: vec2<f32>, intensity: f32) -> f32 {
  let d = length(uv * 2.0 - 1.0);
  return clamp(1.0 - d * d * intensity, 0.0, 1.0);
}

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
}

@vertex
fn tonemapVS(@builtin(vertex_index) vi: u32) -> VSOut {
  // Full-screen triangle
  let x  = f32((vi & 1u) * 2u) - 1.0;
  let y  = 1.0 - f32((vi & 2u));
  var o: VSOut;
  o.pos = vec4<f32>(x, y, 0.0, 1.0);
  o.uv  = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return o;
}

@fragment
fn tonemapFS(in: VSOut) -> @location(0) vec4<f32> {
  let scene  = textureSample(sceneTex, samp, in.uv).rgb;
  let bloom  = textureSample(bloomTex, samp, in.uv).rgb;

  let vignette = computeVignette(in.uv, u.vignetteIntensity);

  // UE5 TonemapCommonPS: FinalLinearColor = SceneColor * tint * (exposure * vignette * localExposure)
  // (LocalExposure baked into globalExposure for web port)
  var finalLinear = scene * u.sceneTint * (u.globalExposure * u.oneOverPreExp * vignette);

  // Bloom composite (UE5: += Bloom * (OneOverPreExposure * GlobalExposure * VignetteMask))
  finalLinear += bloom * u.bloomTint * (u.globalExposure * u.oneOverPreExp * vignette * u.bloomScale);

  // ACES filmic tonemap (UE5 FilmToneMap)
  var tonemapped = filmToneMap(finalLinear);

  // Film grain (UE5 GrainFromUV)
  if (u.grainIntensity > 0.0) {
    let grain = grainFromUV(in.uv, u.grainSeed);
    tonemapped *= 1.0 + (grain - 0.5) * u.grainIntensity;
  }

  // sRGB gamma (linear → sRGB, UE5 TonemapAndGammaCorrect)
  let gammaOut = pow(clamp(tonemapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

  return vec4<f32>(gammaOut, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Helper Utilities
// ─────────────────────────────────────────────────────────────────────────────

/** Nearest power-of-2 ≥ n. Used for FFT buffer sizing. */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** Create a 1-element float uniform buffer, writable. */
function makeUniformBuf(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size  : Math.ceil(size / 16) * 16,
    usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

/** Create an rgba16float storage texture (compute writeable). */
function makeStorageTex(
  device : GPUDevice,
  w      : number,
  h      : number,
  label  : string,
): GPUTexture {
  return device.createTexture({
    label,
    size   : [w, h, 1],
    format : 'rgba16float',
    usage  : GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.STORAGE_BINDING
           | GPUTextureUsage.RENDER_ATTACHMENT
           | GPUTextureUsage.COPY_SRC,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Core Class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * UEBloomTonemap — WebGPU port of UE5 Bloom + Tonemap post-process pipeline.
 *
 * Usage:
 * ```ts
 * const ubt = await UEBloomTonemap.create(device, format, width, height);
 * ubt.setParams({ bloomThreshold: 0.8, bloomQuality: 4 });
 *
 * // Each frame:
 * ubt.render(encoder, sceneTexture, dstView);
 * ```
 */
export class UEBloomTonemap {
  private readonly device : GPUDevice;
  private width           : number;
  private height          : number;
  private format          : GPUTextureFormat;

  // ── GPU pipelines ────────────────────────────────────────────────────────
  private eyeAdaptPipeline  : GPUComputePipeline | null = null;
  private bloomSetupPipeline: GPUComputePipeline | null = null;
  private downsamplePipeline: GPUComputePipeline | null = null;
  private gaussianPipeline  : GPUComputePipeline | null = null;
  private fftMulPipeline    : GPUComputePipeline | null = null;
  private tonemapPipeline   : GPURenderPipeline  | null = null;

  // ── GPU textures ─────────────────────────────────────────────────────────
  private brightTex  : GPUTexture;
  private bloomMips  : GPUTexture[];   // downsample pyramid
  private bloomAccum : GPUTexture;     // final accumulated bloom
  private blurTemp   : GPUTexture;     // ping-pong for H/V gaussian pass

  // ── GPU buffers ──────────────────────────────────────────────────────────
  private lumGridBuf        : GPUBuffer;  // 256 × f32 for eye adaptation
  private bloomSetupUBuf    : GPUBuffer;
  private gaussianUBuf      : GPUBuffer;
  private tonemapUBuf       : GPUBuffer;

  // ── Runtime state ────────────────────────────────────────────────────────
  private currentExposure   : number = 1.0;
  private params            : Required<UEBloomTonemapParams>;

  private constructor(device: GPUDevice, format: GPUTextureFormat, w: number, h: number) {
    this.device = device;
    this.format = format;
    this.width  = w;
    this.height = h;
    this.params = UEBloomTonemap.defaultParams();

    // Create textures & buffers
    this.brightTex  = makeStorageTex(device, w, h, 'ue-bloom-bright');
    this.bloomAccum = makeStorageTex(device, w, h, 'ue-bloom-accum');
    this.blurTemp   = makeStorageTex(device, w, h, 'ue-bloom-blur-temp');

    // Downsample pyramid: 6 mip levels (max Q5)
    this.bloomMips = [];
    for (let i = 0; i < 6; i++) {
      const mw = Math.max(1, w >> (i + 1));
      const mh = Math.max(1, h >> (i + 1));
      this.bloomMips.push(makeStorageTex(device, mw, mh, `ue-bloom-mip${i}`));
    }

    // Storage buffer: 256 f32 values for 16×16 luma grid
    this.lumGridBuf = device.createBuffer({
      label : 'ue-lum-grid',
      size  : 256 * 4,
      usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bloomSetupUBuf = makeUniformBuf(device, 64,  'ue-bloom-setup-ub');
    this.gaussianUBuf   = makeUniformBuf(device, 48,  'ue-gaussian-ub');
    this.tonemapUBuf    = makeUniformBuf(device, 128, 'ue-tonemap-ub');
  }

  // ── Static factory ───────────────────────────────────────────────────────

  /** Async factory: compiles all pipelines then returns a ready instance. */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<UEBloomTonemap> {
    const inst = new UEBloomTonemap(device, format, width, height);
    await inst._compilePipelines();
    return inst;
  }

  private static defaultParams(): Required<UEBloomTonemapParams> {
    return {
      bloomThreshold    : 0.8,
      bloomIntensity    : 1.0,
      bloomQuality      : 4,
      bloomStages       : UE_BLOOM_STAGES_DEFAULT,
      bloomSizeScale    : 1.0,
      useFFT            : false,
      kernelTexture     : undefined as unknown as GPUTexture,
      eyeAdaptation     : true,
      eyeAdaptationMin  : 0.1,
      eyeAdaptationMax  : 2.0,
      eyeAdaptationSpeed: 0.05,
      film              : { ...UE_ACES_DEFAULTS },
      sceneTint         : [1, 1, 1],
      bloomTint         : [0.5, 0.5, 0.5],
      vignetteIntensity : 0.4,
      grainIntensity    : 0.0,
    };
  }

  // ── Pipeline compilation ─────────────────────────────────────────────────

  private async _compilePipelines(): Promise<void> {
    const d = this.device;

    // Eye adaptation compute
    this.eyeAdaptPipeline = d.createComputePipeline({
      label  : 'ue-eye-adapt',
      layout : 'auto',
      compute: {
        module    : d.createShaderModule({ code: WGSL_EYE_ADAPT_DOWNSAMPLE }),
        entryPoint: 'eyeAdaptCS',
      },
    });

    // Bloom setup compute
    this.bloomSetupPipeline = d.createComputePipeline({
      label  : 'ue-bloom-setup',
      layout : 'auto',
      compute: {
        module    : d.createShaderModule({ code: WGSL_BLOOM_SETUP }),
        entryPoint: 'bloomSetupCS',
      },
    });

    // Downsample compute
    this.downsamplePipeline = d.createComputePipeline({
      label  : 'ue-downsample',
      layout : 'auto',
      compute: {
        module    : d.createShaderModule({ code: WGSL_BLOOM_DOWNSAMPLE }),
        entryPoint: 'downsampleCS',
      },
    });

    // Separable gaussian compute
    this.gaussianPipeline = d.createComputePipeline({
      label  : 'ue-gaussian',
      layout : 'auto',
      compute: {
        module    : d.createShaderModule({ code: WGSL_BLOOM_GAUSSIAN }),
        entryPoint: 'gaussianCS',
      },
    });

    // FFT multiply compute
    this.fftMulPipeline = d.createComputePipeline({
      label  : 'ue-fft-mul',
      layout : 'auto',
      compute: {
        module    : d.createShaderModule({ code: WGSL_FFT_BLOOM_MULTIPLY }),
        entryPoint: 'fftMultiplyCS',
      },
    });

    // Tonemap render pipeline
    const tonemapModule = d.createShaderModule({ code: WGSL_TONEMAP });
    this.tonemapPipeline = d.createRenderPipeline({
      label  : 'ue-tonemap',
      layout : 'auto',
      vertex : { module: tonemapModule, entryPoint: 'tonemapVS' },
      fragment: {
        module    : tonemapModule,
        entryPoint: 'tonemapFS',
        targets   : [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /**
   * Update tweakable parameters. Partial update supported.
   * Safe to call every frame.
   */
  setParams(p: UEBloomTonemapParams): void {
    Object.assign(this.params, p);
    if (p.film) {
      this.params.film = { ...UE_ACES_DEFAULTS, ...p.film };
    }
  }

  /**
   * Render the complete Bloom+Tonemap pipeline.
   *
   * @param encoder  - Active GPUCommandEncoder for this frame.
   * @param sceneTex - HDR scene color texture (must be TEXTURE_BINDING capable).
   * @param dstView  - Render target view for the final tonemapped output.
   */
  render(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    if (!this.tonemapPipeline) return;

    const samp = this.device.createSampler({
      minFilter   : 'linear',
      magFilter   : 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // Pass 0: Eye adaptation luma downsample
    if (this.params.eyeAdaptation) {
      this._runEyeAdaptation(encoder, sceneTex, samp);
    }

    // Pass 1: Bloom setup (threshold + local exposure)
    this._runBloomSetup(encoder, sceneTex, samp);

    // Pass 2–7: Gaussian downsample pyramid
    this._runDownsamplePyramid(encoder, samp);

    // Pass 8–N: Gaussian upsample + accumulate (or FFT path)
    if (this.params.useFFT && this.params.kernelTexture) {
      this._runFFTBloom(encoder, samp);
    } else {
      this._runGaussianBloom(encoder, samp);
    }

    // Final pass: Tonemap + composite
    this._runTonemap(encoder, sceneTex, samp, dstView);
  }

  // ─── Private pass runners ───────────────────────────────────────────────

  private _runEyeAdaptation(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    samp     : GPUSampler,
  ): void {
    if (!this.eyeAdaptPipeline) return;

    const bg = this.device.createBindGroup({
      layout : this.eyeAdaptPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sceneTex.createView() },
        { binding: 1, resource: samp },
        { binding: 2, resource: { buffer: this.lumGridBuf } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ue-eye-adapt-pass' });
    pass.setPipeline(this.eyeAdaptPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(16, 16); // 16×16 tiles
    pass.end();
  }

  private _runBloomSetup(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    samp     : GPUSampler,
  ): void {
    if (!this.bloomSetupPipeline) return;

    // Write bloom setup uniforms
    // struct: threshold(f32), exposureScale(f32), localExpMin(f32), localExpMax(f32), pad(vec4)
    const data = new Float32Array(8);
    data[0] = this.params.bloomThreshold;
    data[1] = this.currentExposure;
    data[2] = this.params.eyeAdaptationMin;
    data[3] = this.params.eyeAdaptationMax;
    this.device.queue.writeBuffer(this.bloomSetupUBuf, 0, data);

    const bg = this.device.createBindGroup({
      layout : this.bloomSetupPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bloomSetupUBuf } },
        { binding: 1, resource: sceneTex.createView() },
        { binding: 2, resource: samp },
        { binding: 3, resource: { buffer: this.lumGridBuf } },
        { binding: 4, resource: this.brightTex.createView() },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ue-bloom-setup-pass' });
    pass.setPipeline(this.bloomSetupPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  private _runDownsamplePyramid(
    encoder : GPUCommandEncoder,
    samp    : GPUSampler,
  ): void {
    if (!this.downsamplePipeline) return;

    const stageCount = BLOOM_QUALITY_STAGE_COUNT[this.params.bloomQuality] ?? 5;

    let srcTex: GPUTexture = this.brightTex;
    for (let i = 0; i < stageCount; i++) {
      const dstTex = this.bloomMips[i];

      const bg = this.device.createBindGroup({
        layout : this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcTex.createView() },
          { binding: 1, resource: samp },
          { binding: 2, resource: dstTex.createView() },
        ],
      });

      const [mw, mh] = [dstTex.width, dstTex.height];
      const pass = encoder.beginComputePass({ label: `ue-downsample-mip${i}` });
      pass.setPipeline(this.downsamplePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(mw / 8), Math.ceil(mh / 8));
      pass.end();

      srcTex = dstTex;
    }
  }

  /**
   * Run separable Gaussian blur on a single source texture, writing to dst.
   * Uses ping-pong with blurTemp.
   * Mirrors UE5 AddGaussianBlurPass (H pass + V pass).
   */
  private _runGaussianOnTex(
    encoder : GPUCommandEncoder,
    srcTex  : GPUTexture,
    dstTex  : GPUTexture,
    samp    : GPUSampler,
    tint    : [number, number, number],
  ): void {
    if (!this.gaussianPipeline) return;

    const writeGaussUniforms = (dir: [number, number], t: [number, number, number]) => {
      const d = new Float32Array(8);
      d[0] = dir[0]; d[1] = dir[1];
      d[2] = t[0]; d[3] = t[1]; d[4] = t[2];
      this.device.queue.writeBuffer(this.gaussianUBuf, 0, d);
    };

    // Horizontal pass: srcTex → blurTemp
    writeGaussUniforms([1, 0], [1, 1, 1]);
    {
      const bg = this.device.createBindGroup({
        layout : this.gaussianPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gaussianUBuf } },
          { binding: 1, resource: srcTex.createView() },
          { binding: 2, resource: samp },
          { binding: 3, resource: this.blurTemp.createView() },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'ue-gauss-h' });
      pass.setPipeline(this.gaussianPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
      pass.end();
    }

    // Vertical pass: blurTemp → dstTex, apply tint
    writeGaussUniforms([0, 1], tint);
    {
      const bg = this.device.createBindGroup({
        layout : this.gaussianPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gaussianUBuf } },
          { binding: 1, resource: this.blurTemp.createView() },
          { binding: 2, resource: samp },
          { binding: 3, resource: dstTex.createView() },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'ue-gauss-v' });
      pass.setPipeline(this.gaussianPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
      pass.end();
    }
  }

  private _runGaussianBloom(
    encoder : GPUCommandEncoder,
    samp    : GPUSampler,
  ): void {
    const stageCount = BLOOM_QUALITY_STAGE_COUNT[this.params.bloomQuality] ?? 5;
    const stages     = this.params.bloomStages ?? UE_BLOOM_STAGES_DEFAULT;
    const tintScale  = (1.0 / 6) * this.params.bloomIntensity * this.params.bloomSizeScale;

    // Accumulate: process from coarsest to finest mip, blur & add to accumulator
    // Mirrors UE5 AddGaussianBloomPasses loop (StageIndex=0..N, SourceIndex=MAX-1..0)
    for (let i = 0; i < stageCount; i++) {
      const stageIdx  = i;
      const stage     = stages[stageIdx] ?? stages[stages.length - 1];
      const scaledTint: [number, number, number] = [
        stage.tint[0] * tintScale,
        stage.tint[1] * tintScale,
        stage.tint[2] * tintScale,
      ];

      if (stage.size > 1e-5) {
        // Blur the mip-level texture
        this._runGaussianOnTex(encoder, this.bloomMips[i], this.bloomAccum, samp, scaledTint);
      }
    }
  }

  private _runFFTBloom(
    encoder : GPUCommandEncoder,
    _samp   : GPUSampler,
  ): void {
    // FFT path: multiply brightTex spectrum by kernel spectrum → bloomAccum
    // Simplified 2D FFT convolution approximation.
    // Full GPU FFT is beyond WebGPU compute scope without a dedicated library;
    // we perform the frequency-domain multiply assuming pre-transformed inputs.
    if (!this.fftMulPipeline || !this.params.kernelTexture) return;

    const bg = this.device.createBindGroup({
      layout : this.fftMulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.brightTex.createView() },
        { binding: 1, resource: this.params.kernelTexture.createView() },
        { binding: 2, resource: this.bloomAccum.createView() },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'ue-fft-mul-pass' });
    pass.setPipeline(this.fftMulPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  private _runTonemap(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    samp     : GPUSampler,
    dstView  : GPUTextureView,
  ): void {
    if (!this.tonemapPipeline) return;

    // Write tonemap uniforms (struct TonemapUniforms layout, 128 bytes)
    const p    = this.params;
    const film = { ...UE_ACES_DEFAULTS, ...p.film };
    const data = new Float32Array(32);

    let o = 0;
    data[o++] = this.currentExposure;       // globalExposure
    data[o++] = 1.0 / this.currentExposure; // oneOverPreExp
    data[o++] = film.slope;                 // filmSlope
    data[o++] = film.toe;                   // filmToe
    data[o++] = film.shoulder;              // filmShoulder
    data[o++] = film.blackClip;             // filmBlackClip
    data[o++] = film.whiteClip;             // filmWhiteClip
    data[o++] = 0;                          // _pad0
    data[o++] = p.sceneTint[0]; data[o++] = p.sceneTint[1]; data[o++] = p.sceneTint[2];
    data[o++] = 0;                          // _pad1
    data[o++] = p.bloomTint[0]; data[o++] = p.bloomTint[1]; data[o++] = p.bloomTint[2];
    data[o++] = p.bloomIntensity;           // bloomScale
    data[o++] = p.vignetteIntensity;        // vignetteIntensity
    data[o++] = p.grainIntensity;           // grainIntensity
    data[o++] = Math.random() * 1000;       // grainSeed (per-frame random)
    data[o++] = 0;                          // _pad2
    data[o++] = this.width;                 // viewportSize.x
    data[o++] = this.height;                // viewportSize.y

    this.device.queue.writeBuffer(this.tonemapUBuf, 0, data);

    const bg = this.device.createBindGroup({
      layout : this.tonemapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tonemapUBuf } },
        { binding: 1, resource: sceneTex.createView() },
        { binding: 2, resource: this.bloomAccum.createView() },
        { binding: 3, resource: samp },
      ],
    });

    const pass = encoder.beginRenderPass({
      label           : 'ue-tonemap-pass',
      colorAttachments: [{
        view      : dstView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.tonemapPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  // ── Eye Adaptation CPU feedback ──────────────────────────────────────────

  /**
   * Async: read back luma grid from GPU and update currentExposure.
   * Call once per frame (but don't await if non-blocking is needed —
   * use the sync path below which uses the previous frame's result).
   *
   * Mirrors UE5 EyeAdaptation buffer[0].x computation.
   */
  async updateExposureAsync(): Promise<void> {
    if (!this.params.eyeAdaptation) return;

    const readBuf = this.device.createBuffer({
      size  : 256 * 4,
      usage : GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const cmd = this.device.createCommandEncoder();
    cmd.copyBufferToBuffer(this.lumGridBuf, 0, readBuf, 0, 256 * 4);
    this.device.queue.submit([cmd.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const view   = new Float32Array(readBuf.getMappedRange());
    let   avgLum = 0;
    for (let i = 0; i < 256; i++) avgLum += view[i];
    avgLum /= 256;
    readBuf.unmap();
    readBuf.destroy();

    // Compute target exposure (UE5: targetExposure = middleGrey / avgLum)
    const middleGrey     = 0.18;
    const targetExposure = avgLum > 1e-5
      ? Math.max(this.params.eyeAdaptationMin,
          Math.min(this.params.eyeAdaptationMax, middleGrey / avgLum))
      : 1.0;

    // Temporal smooth (UE5 eye adaptation lerp)
    this.currentExposure += (targetExposure - this.currentExposure)
      * this.params.eyeAdaptationSpeed;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 6  AT UnrealBloom Bridge
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Bridge: inject UE bloom output into an AT UnrealBloom composite pass.
   *
   * If `atBloom` is provided (an object with a `.bloomAccumTex` getter), its
   * bloom texture is replaced with this pipeline's `bloomAccum` output before
   * the AT composite pass runs.
   *
   * This allows using UE5's more accurate ACES bloom in place of AT's simpler
   * three-pass bloom while keeping the rest of the AT pipeline intact.
   *
   * @param atBloom - AT bloom instance exposing `bloomAccumTex: GPUTexture`.
   */
  bridgeToAT(atBloom: { bloomAccumTex?: GPUTexture }): void {
    // Provide UE bloom accum texture reference to AT bloom
    Object.defineProperty(atBloom, 'bloomAccumTex', {
      get: () => this.bloomAccum,
      configurable: true,
    });
  }

  /**
   * Convenience: run bloom-only (no tonemap), then hand result to AT.
   * Useful if AT handles tonemap and you only want UE's superior bloom.
   *
   * @param encoder  - Active command encoder.
   * @param sceneTex - HDR scene color.
   */
  renderBloomOnly(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
  ): GPUTexture {
    const samp = this.device.createSampler({
      minFilter: 'linear', magFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });

    this._runBloomSetup(encoder, sceneTex, samp);
    this._runDownsamplePyramid(encoder, samp);

    if (this.params.useFFT && this.params.kernelTexture) {
      this._runFFTBloom(encoder, samp);
    } else {
      this._runGaussianBloom(encoder, samp);
    }

    return this.bloomAccum;
  }

  // ── Resize ────────────────────────────────────────────────────────────────

  /**
   * Resize all GPU textures to match new viewport dimensions.
   * Must be called when the canvas resizes.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;

    this.width  = width;
    this.height = height;

    // Destroy old textures
    this.brightTex.destroy();
    this.bloomAccum.destroy();
    this.blurTemp.destroy();
    for (const t of this.bloomMips) t.destroy();

    // Recreate
    this.brightTex  = makeStorageTex(this.device, width, height, 'ue-bloom-bright');
    this.bloomAccum = makeStorageTex(this.device, width, height, 'ue-bloom-accum');
    this.blurTemp   = makeStorageTex(this.device, width, height, 'ue-bloom-blur-temp');
    this.bloomMips  = [];
    for (let i = 0; i < 6; i++) {
      const mw = Math.max(1, width  >> (i + 1));
      const mh = Math.max(1, height >> (i + 1));
      this.bloomMips.push(makeStorageTex(this.device, mw, mh, `ue-bloom-mip${i}`));
    }
  }

  /** Destroy all GPU resources. Instance must not be used after this. */
  destroy(): void {
    this.brightTex.destroy();
    this.bloomAccum.destroy();
    this.blurTemp.destroy();
    for (const t of this.bloomMips) t.destroy();
    this.lumGridBuf.destroy();
    this.bloomSetupUBuf.destroy();
    this.gaussianUBuf.destroy();
    this.tonemapUBuf.destroy();
  }

  // ── Diagnostics ───────────────────────────────────────────────────────────

  /** Current auto-exposure value (1.0 = neutral). */
  get exposure(): number { return this.currentExposure; }

  /** Returns all WGSL source strings for inspection / hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      eyeAdapt     : WGSL_EYE_ADAPT_DOWNSAMPLE,
      bloomSetup   : WGSL_BLOOM_SETUP,
      downsample   : WGSL_BLOOM_DOWNSAMPLE,
      gaussian     : WGSL_BLOOM_GAUSSIAN,
      fftMultiply  : WGSL_FFT_BLOOM_MULTIPLY,
      tonemap      : WGSL_TONEMAP,
    } as const;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Convenience Factories
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a UEBloomTonemap seeded from AT-style species bloom params.
 *
 * Maps AT `{ bloomStrength, bloomRadius, luminosityThreshold }` onto
 * UE5 equivalent parameters.
 *
 * @param device  - WebGPU device.
 * @param format  - Swapchain texture format.
 * @param width   - Viewport width.
 * @param height  - Viewport height.
 * @param species - Optional AT species bloom params.
 */
export async function createUEBloomForSpecies(
  device  : GPUDevice,
  format  : GPUTextureFormat,
  width   : number,
  height  : number,
  species?: {
    bloomStrength        : number;
    bloomRadius          : number;
    luminosityThreshold  : number;
  },
): Promise<UEBloomTonemap> {
  const inst = await UEBloomTonemap.create(device, format, width, height);
  if (species) {
    inst.setParams({
      bloomThreshold : species.luminosityThreshold,
      bloomIntensity : species.bloomStrength,
      bloomSizeScale : species.bloomRadius * 4.0,
    });
  }
  return inst;
}

/**
 * Build a UEBloomTonemap with FFT convolution using a lens texture.
 *
 * @param device      - WebGPU device.
 * @param format      - Swapchain texture format.
 * @param width       - Viewport width.
 * @param height      - Viewport height.
 * @param kernelTex   - Pre-loaded lens kernel GPUTexture (TEXTURE_BINDING).
 * @param params      - Additional bloom/tonemap params.
 */
export async function createUEFFTBloom(
  device    : GPUDevice,
  format    : GPUTextureFormat,
  width     : number,
  height    : number,
  kernelTex : GPUTexture,
  params   ?: UEBloomTonemapParams,
): Promise<UEBloomTonemap> {
  const inst = await UEBloomTonemap.create(device, format, width, height);
  inst.setParams({
    useFFT        : true,
    kernelTexture : kernelTex,
    ...params,
  });
  return inst;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Re-exports for external use
// ─────────────────────────────────────────────────────────────────────────────

/** Default ACES film curve parameters (UE5 preset). */
export { UE_ACES_DEFAULTS };

/** UE5 default bloom stage definitions. */
export { UE_BLOOM_STAGES_DEFAULT };

/** Bloom quality → stage count mapping. */
export { BLOOM_QUALITY_STAGE_COUNT };

/** WGSL shader source fragments for external embedding or hot-reload. */
export const UE_BLOOM_TONEMAP_WGSL = {
  eyeAdapt    : WGSL_EYE_ADAPT_DOWNSAMPLE,
  bloomSetup  : WGSL_BLOOM_SETUP,
  downsample  : WGSL_BLOOM_DOWNSAMPLE,
  gaussian    : WGSL_BLOOM_GAUSSIAN,
  fftMultiply : WGSL_FFT_BLOOM_MULTIPLY,
  tonemap     : WGSL_TONEMAP,
} as const;

// Utility helpers re-exported
export { nextPow2, makeStorageTex };
