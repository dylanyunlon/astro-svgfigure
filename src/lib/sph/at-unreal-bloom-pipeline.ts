/**
 * at-unreal-bloom-pipeline.ts — M844: AT Unreal Bloom Pipeline — WebGPU/WGSL Port
 *
 * 完整移植 ActiveTheory 的 UE 风格 UnrealBloom 后处理管线，基于 compiled.vs 着色器：
 *   UnrealBloom.fs / UnrealBloomPass.fs / UnrealBloomComposite.glsl
 *   UnrealBloomGaussian.glsl / UnrealBloomLuminosity.glsl
 *   BloomLuminosityPass.glsl / DownSample.glsl / UpSample.glsl
 *
 * 管线结构（每帧 render()）:
 *   Stage 1 — LUMINOSITY THRESHOLD : scene → brightTex (luma + smoothstep)
 *   Stage 2 — DOWNSAMPLE PYRAMID   : brightTex → 5级 13-tap 加权下采样
 *   Stage 3 — GAUSSIAN BLUR        : 每级 H+V 可分离高斯 (gaussianPdf)
 *   Stage 4 — UPSAMPLE CHAIN       : 9-tap 帐篷滤波 + tint 累积上采样
 *   Stage 5 — COMPOSITE            : lerpBloomFactor + additive blend
 *
 * 用法：
 *   const pipeline = await ATUnrealBloomPipeline.create(device, format, w, h);
 *   pipeline.setParams({ bloomStrength: 1.2, bloomRadius: 0.8, ... });
 *   pipeline.render(encoder, sceneTexture, dstView);
 *
 * Research: xiaodi #M844 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Number of mip levels in the downsample / upsample pyramid. */
const MIP_LEVELS = 5;

/** Maximum supported gaussian kernel radius (compile-time constant in WGSL). */
const MAX_KERNEL_RADIUS = 16;

/** Default sigma for the gaussian blur kernel. */
const DEFAULT_SIGMA = 3.0;

/** Default kernel radius (tap count = radius × 2 + 1). */
const DEFAULT_KERNEL_RADIUS = 5;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configurable bloom parameters — all optional, merged into resolved defaults.
 * Maps directly to the uniform values consumed by each pipeline stage.
 */
export interface ATUnrealBloomParams {
  /**
   * Overall bloom intensity multiplier.
   * Maps to UnrealBloomComposite.glsl `bloomStrength`.
   * @default 1.0
   */
  bloomStrength?: number;

  /**
   * Bloom radius — controls lerpBloomFactor interpolation.
   * Maps to UnrealBloomComposite.glsl `bloomRadius`.
   * 0 = sharp bloom, 1 = wide/diffuse bloom.
   * @default 0.5
   */
  bloomRadius?: number;

  /**
   * RGB tint color applied during final composite.
   * Maps to UnrealBloomComposite.glsl `bloomTintColor`.
   * @default [1, 1, 1]
   */
  bloomTintColor?: [number, number, number];

  /**
   * Luminance threshold — pixels below this luma value are suppressed.
   * Maps to UnrealBloomLuminosity.glsl `luminosityThreshold`.
   * @default 0.0
   */
  luminosityThreshold?: number;

  /**
   * Smooth transition width for the luminosity threshold.
   * Maps to UnrealBloomLuminosity.glsl `smoothWidth`.
   * @default 0.01
   */
  smoothWidth?: number;

  /**
   * Fallback color for sub-threshold pixels (RGB).
   * Maps to UnrealBloomLuminosity.glsl `defaultColor`.
   * @default [0, 0, 0]
   */
  defaultColor?: [number, number, number];

  /**
   * Opacity of the default (sub-threshold) color.
   * Maps to UnrealBloomLuminosity.glsl `defaultOpacity`.
   * @default 0.0
   */
  defaultOpacity?: number;

  /**
   * Gaussian blur sigma — controls kernel spread.
   * Maps to UnrealBloomGaussian.glsl `SIGMA` constant.
   * Higher values = wider blur at each mip level.
   * @default 3.0
   */
  gaussianSigma?: number;

  /**
   * Gaussian blur kernel radius (number of taps = radius × 2 + 1).
   * Maps to UnrealBloomGaussian.glsl `KERNEL_RADIUS` constant.
   * @default 5
   */
  gaussianKernelRadius?: number;

  /**
   * Upsample tent filter radius scale.
   * Maps to UpSample.glsl `uRadius`.
   * @default 1.0
   */
  upsampleRadius?: number;

  /**
   * Upsample accumulation intensity.
   * Maps to UpSample.glsl `uIntensity`.
   * @default 1.0
   */
  upsampleIntensity?: number;

  /**
   * Per-mip tint colors for the upsample chain.
   * Each entry maps to UpSample.glsl `uTint` for that mip level.
   * @default [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
   */
  mipTints?: [number, number, number][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolved params (all fields required, used internally)
// ─────────────────────────────────────────────────────────────────────────────

interface ResolvedParams {
  bloomStrength       : number;
  bloomRadius         : number;
  bloomTintColor      : [number, number, number];
  luminosityThreshold : number;
  smoothWidth         : number;
  defaultColor        : [number, number, number];
  defaultOpacity      : number;
  gaussianSigma       : number;
  gaussianKernelRadius: number;
  upsampleRadius      : number;
  upsampleIntensity   : number;
  mipTints            : [number, number, number][];
}

function resolveDefaults(): ResolvedParams {
  return {
    bloomStrength       : 1.0,
    bloomRadius         : 0.5,
    bloomTintColor      : [1, 1, 1],
    luminosityThreshold : 0.0,
    smoothWidth         : 0.01,
    defaultColor        : [0, 0, 0],
    defaultOpacity      : 0.0,
    gaussianSigma       : DEFAULT_SIGMA,
    gaussianKernelRadius: DEFAULT_KERNEL_RADIUS,
    upsampleRadius      : 1.0,
    upsampleIntensity   : 1.0,
    mipTints            : Array.from({ length: MIP_LEVELS }, () =>
      [1, 1, 1] as [number, number, number]),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: fullscreen triangle vertex shader (shared by all passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FULLSCREEN_TRI_VERTEX = /* wgsl */`
struct VOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

// Fullscreen triangle — covers [-1,1]² clip space, no vertex buffer needed.
// Port of AT compiled.vs fullscreen quad vertex shaders:
//   void main() { vUv = uv; gl_Position = vec4(position, 1.0); }
// Adapted to single-triangle (3 verts) for WebGPU efficiency.
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
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: luma helper — Rec.601 coefficients (matching compiled.vs luma.fs)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LUMA = /* wgsl */`
// Port of compiled.vs :: luma.fs
//   float luma(vec3 color) { return dot(color, vec3(0.299, 0.587, 0.114)); }
fn luma_vec3(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn luma_vec4(color: vec4f) -> f32 {
  return dot(color.rgb, vec3f(0.299, 0.587, 0.114));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Stage 1 — LUMINOSITY THRESHOLD
//
// Source: compiled.vs :: UnrealBloomLuminosity.glsl / BloomLuminosityPass.glsl
//
//   void main() {
//     vec4 texel = texture2D(tDiffuse, vUv);
//     float v = luma(texel.xyz);
//     vec4 outputColor = vec4(defaultColor.rgb, defaultOpacity);
//     float alpha = smoothstep(luminosityThreshold,
//                              luminosityThreshold + smoothWidth, v);
//     gl_FragColor = mix(outputColor, texel, alpha);
//   }
//
// Both UnrealBloomLuminosity.glsl and BloomLuminosityPass.glsl share the
// same fragment logic; the only difference is pass wiring.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LUMINOSITY_THRESHOLD = /* wgsl */`
// ── Luminosity Threshold Uniforms ────────────────────────────────────────────
// Mirrors UnrealBloomLuminosity.glsl / BloomLuminosityPass.glsl uniforms.
struct LuminosityUniforms {
  luminosityThreshold : f32,
  smoothWidth         : f32,
  defaultOpacity      : f32,
  _pad0               : f32,
  defaultColor        : vec3f,
  _pad1               : f32,
}

${WGSL_LUMA}

${WGSL_FULLSCREEN_TRI_VERTEX}

@group(0) @binding(0) var<uniform> u   : LuminosityUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

// ── Fragment: luminosity threshold extraction ────────────────────────────────
// Direct port of UnrealBloomLuminosity.glsl.
// The smoothstep ensures a soft transition at the threshold boundary,
// avoiding hard cutoff artifacts in the bloom halo.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let texel = textureSample(src, smp, in.uv);
  let v = luma_vec3(texel.rgb);
  let outputColor = vec4f(u.defaultColor, u.defaultOpacity);
  let alpha = smoothstep(u.luminosityThreshold,
                         u.luminosityThreshold + u.smoothWidth, v);
  return mix(outputColor, texel, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Stage 2 — DOWNSAMPLE (13-tap weighted filter)
//
// Source: compiled.vs :: DownSample.glsl
//
//   vec3 weights = vec3(0.03125, 0.0625, 0.125);  // 1/32, 1/16, 1/8
//   13 taps in cross + diamond pattern with halfPixel bilinear offsets.
//   Karis-average-inspired energy-preserving filter.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DOWNSAMPLE = /* wgsl */`
// ── Downsample Uniforms ──────────────────────────────────────────────────────
struct DownsampleUniforms {
  invResolution : vec2f,
  radius        : f32,
  _pad0         : f32,
}

${WGSL_FULLSCREEN_TRI_VERTEX}

@group(0) @binding(0) var<uniform> u   : DownsampleUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

// ── Fragment: 13-tap weighted downsample ─────────────────────────────────────
// Direct port of compiled.vs :: DownSample.glsl
// Uses 13 texture fetches with weights [1/32, 1/16, 1/8] in cross+diamond.
// The pattern sums to 1.0, preserving energy across downsample levels.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let pxSize    = u.invResolution;
  let halfPixel = u.invResolution * 0.5;

  let w0 = 0.03125;   // corner  (1/32)
  let w1 = 0.0625;    // edge    (1/16)
  let w2 = 0.125;     // center  (1/8)

  // Bilinear-friendly half-pixel offset positions
  let br = in.uv - halfPixel;
  let bl = in.uv + vec2f(halfPixel.x, -halfPixel.y);
  let tr = in.uv + halfPixel;
  let tl = in.uv + vec2f(-halfPixel.x, halfPixel.y);

  // Row 1: top row (y = -1)
  let A = textureSample(src, smp, in.uv + vec2f(-1.0, -1.0) * pxSize).rgb * w0;
  let B = textureSample(src, smp, in.uv + vec2f( 0.0, -1.0) * pxSize).rgb * w1;
  let C = textureSample(src, smp, in.uv + vec2f( 1.0, -1.0) * pxSize).rgb * w0;

  // Row 2: upper-mid (halfPixel offsets + left edge)
  let D = textureSample(src, smp, br).rgb * w2;
  let E = textureSample(src, smp, bl).rgb * w2;
  let F = textureSample(src, smp, in.uv + vec2f(-1.0, 0.0) * pxSize).rgb * w1;

  // Center
  let G = textureSample(src, smp, in.uv).rgb * w2;

  // Row 3: right edge + lower-mid (halfPixel offsets)
  let H = textureSample(src, smp, in.uv + vec2f(1.0, 0.0) * pxSize).rgb * w1;
  let I = textureSample(src, smp, tl).rgb * w2;
  let J = textureSample(src, smp, tr).rgb * w2;

  // Row 4: bottom row (y = +1)
  let K = textureSample(src, smp, in.uv + vec2f(-1.0, 1.0) * pxSize).rgb * w0;
  let L = textureSample(src, smp, in.uv + vec2f( 0.0, 1.0) * pxSize).rgb * w1;
  let M = textureSample(src, smp, in.uv + vec2f( 1.0, 1.0) * pxSize).rgb * w0;

  let sum = A + B + C + D + E + F + G + H + I + J + K + L + M;

  return vec4f(sum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Stage 3 — GAUSSIAN BLUR (separable, H/V via direction uniform)
//
// Source: compiled.vs :: UnrealBloomGaussian.glsl
//
//   float gaussianPdf(in float x, in float sigma) {
//     return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
//   }
//   Loop from 1 to KERNEL_RADIUS, symmetric sampling, normalize by weightSum.
//   In GLSL SIGMA and KERNEL_RADIUS are #define; in WGSL we use uniforms.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GAUSSIAN_BLUR = /* wgsl */`
// ── Gaussian Blur Uniforms ───────────────────────────────────────────────────
struct GaussianUniforms {
  invTexSize   : vec2f,
  direction    : vec2f,
  sigma        : f32,
  kernelRadius : f32,
  _pad0        : f32,
  _pad1        : f32,
}

${WGSL_FULLSCREEN_TRI_VERTEX}

@group(0) @binding(0) var<uniform> u   : GaussianUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

// ── gaussianPdf ──────────────────────────────────────────────────────────────
// Direct port of UnrealBloomGaussian.glsl:
//   0.39894 ≈ 1/√(2π)
fn gaussianPdf(x: f32, sigma: f32) -> f32 {
  let s2 = sigma * sigma;
  return 0.39894 * exp(-0.5 * x * x / max(s2, 0.0001)) / max(sigma, 0.0001);
}

// ── Fragment: separable gaussian blur ────────────────────────────────────────
// Direction uniform controls H vs V: H=(1,0), V=(0,1).
// Loop uses compile-time max; runtime check exits early for smaller kernels.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let invSize = u.invTexSize;
  let fSigma  = u.sigma;
  let kRadius = i32(u.kernelRadius);

  // Center tap
  var weightSum  = gaussianPdf(0.0, fSigma);
  var diffuseSum = textureSample(src, smp, in.uv).rgb * weightSum;

  // Symmetric taps: i = 1 .. kernelRadius-1
  for (var i: i32 = 1; i < ${MAX_KERNEL_RADIUS}; i++) {
    if (i >= kRadius) { break; }

    let x = f32(i);
    let w = gaussianPdf(x, fSigma);
    let uvOffset = u.direction * invSize * x;

    let sample1 = textureSample(src, smp, in.uv + uvOffset).rgb;
    let sample2 = textureSample(src, smp, in.uv - uvOffset).rgb;

    diffuseSum += (sample1 + sample2) * w;
    weightSum  += 2.0 * w;
  }

  return vec4f(diffuseSum / max(weightSum, 1e-6), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Stage 4 — UPSAMPLE (9-tap tent filter with accumulation)
//
// Source: compiled.vs :: UpSample.glsl
//
//   9-tap tent filter (3×3) with weights [1/16, 1/8, 1/4] summing to 1.0.
//   tNext accumulates bloom chain from coarser mips upward.
//   next += min(vec3(1.0), sum * uIntensity) * uTint;
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_UPSAMPLE = /* wgsl */`
// ── Upsample Uniforms ────────────────────────────────────────────────────────
struct UpsampleUniforms {
  invResolution : vec2f,
  radius        : f32,
  intensity     : f32,
  tint          : vec3f,
  _pad0         : f32,
}

${WGSL_FULLSCREEN_TRI_VERTEX}

@group(0) @binding(0) var<uniform> u    : UpsampleUniforms;
@group(0) @binding(1) var          smp  : sampler;
@group(0) @binding(2) var          tMap : texture_2d<f32>;  // current blurred mip
@group(0) @binding(3) var          tNext: texture_2d<f32>;  // accumulator from lower mip

// ── Fragment: 9-tap tent filter upsample ─────────────────────────────────────
// Direct port of compiled.vs :: UpSample.glsl
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let texelSize = u.invResolution * u.radius;

  // 9-tap tent filter (weights sum to 1.0)
  var sum = vec3f(0.0);

  // Row 1: top
  sum += textureSample(tMap, smp, in.uv + vec2f(-texelSize.x, -texelSize.y)).rgb * 0.0625;
  sum += textureSample(tMap, smp, in.uv + vec2f( 0.0,         -texelSize.y)).rgb * 0.125;
  sum += textureSample(tMap, smp, in.uv + vec2f( texelSize.x, -texelSize.y)).rgb * 0.0625;

  // Row 2: middle
  sum += textureSample(tMap, smp, in.uv + vec2f(-texelSize.x,  0.0)).rgb * 0.125;
  sum += textureSample(tMap, smp, in.uv).rgb * 0.25;
  sum += textureSample(tMap, smp, in.uv + vec2f( texelSize.x,  0.0)).rgb * 0.125;

  // Row 3: bottom
  sum += textureSample(tMap, smp, in.uv + vec2f(-texelSize.x,  texelSize.y)).rgb * 0.0625;
  sum += textureSample(tMap, smp, in.uv + vec2f( 0.0,          texelSize.y)).rgb * 0.125;
  sum += textureSample(tMap, smp, in.uv + vec2f( texelSize.x,  texelSize.y)).rgb * 0.0625;

  // Accumulate from the lower mip chain
  var next = textureSample(tNext, smp, in.uv).rgb;
  next += min(vec3f(1.0), sum * u.intensity) * u.tint;

  return vec4f(next, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Stage 5 — COMPOSITE (lerpBloomFactor + additive blend)
//
// Source: compiled.vs :: UnrealBloomComposite.glsl + UnrealBloomPass.fs
//
// UnrealBloomComposite.glsl:
//   float lerpBloomFactor(const in float factor) {
//     float mirrorFactor = 1.2 - factor;
//     return mix(factor, mirrorFactor, bloomRadius);
//   }
//   gl_FragColor = bloomStrength * lerpBloomFactor(1.0)
//                * vec4(bloomTintColor, 1.0) * texture2D(blurTexture1, vUv);
//
// UnrealBloomPass.fs:
//   color.rgb += getUnrealBloom(vUv);
//
// Combined: finalColor = scene.rgb + bloomStrength × lerpBloomFactor × tint × bloom
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */`
// ── Composite Uniforms ───────────────────────────────────────────────────────
struct CompositeUniforms {
  bloomStrength  : f32,
  bloomRadius    : f32,
  _pad0          : f32,
  _pad1          : f32,
  bloomTintColor : vec3f,
  _pad2          : f32,
}

${WGSL_FULLSCREEN_TRI_VERTEX}

@group(0) @binding(0) var<uniform> u        : CompositeUniforms;
@group(0) @binding(1) var          smp      : sampler;
@group(0) @binding(2) var          sceneTex : texture_2d<f32>;
@group(0) @binding(3) var          bloomTex : texture_2d<f32>;

// ── lerpBloomFactor ──────────────────────────────────────────────────────────
// Direct port of UnrealBloomComposite.glsl:
//   When bloomRadius = 0: returns factor unchanged.
//   When bloomRadius = 1: returns 1.2 - factor (wider bloom levels).
fn lerpBloomFactor(factor: f32) -> f32 {
  let mirrorFactor = 1.2 - factor;
  return mix(factor, mirrorFactor, u.bloomRadius);
}

// ── Fragment: scene + bloom composite ────────────────────────────────────────
// Combines UnrealBloomComposite weighted bloom with UnrealBloomPass additive.
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, smp, in.uv);
  let bloom = textureSample(bloomTex, smp, in.uv);

  let factor = lerpBloomFactor(1.0);
  let compositeBloom = u.bloomStrength * factor
    * vec4f(u.bloomTintColor, 1.0) * bloom;

  let result = vec3f(scene.rgb + compositeBloom.rgb);

  return vec4f(result, scene.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Uniform packing helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Pack luminosity threshold uniforms (32 bytes = 8 × f32). */
function packLuminosityUniforms(p: ResolvedParams): Float32Array {
  const buf = new Float32Array(8);
  buf[0] = p.luminosityThreshold;
  buf[1] = p.smoothWidth;
  buf[2] = p.defaultOpacity;
  buf[3] = 0.0;  // _pad0
  buf[4] = p.defaultColor[0];
  buf[5] = p.defaultColor[1];
  buf[6] = p.defaultColor[2];
  buf[7] = 0.0;  // _pad1
  return buf;
}

/** Pack downsample uniforms (16 bytes = 4 × f32). */
function packDownsampleUniforms(w: number, h: number): Float32Array {
  const buf = new Float32Array(4);
  buf[0] = 1.0 / Math.max(w, 1);
  buf[1] = 1.0 / Math.max(h, 1);
  buf[2] = 1.0;  // radius (reserved)
  buf[3] = 0.0;  // _pad0
  return buf;
}

/** Pack gaussian blur uniforms (32 bytes = 8 × f32). */
function packGaussianUniforms(
  w: number, h: number, dirH: boolean,
  sigma: number, kRadius: number,
): Float32Array {
  const buf = new Float32Array(8);
  buf[0] = 1.0 / Math.max(w, 1);
  buf[1] = 1.0 / Math.max(h, 1);
  buf[2] = dirH ? 1.0 : 0.0;
  buf[3] = dirH ? 0.0 : 1.0;
  buf[4] = sigma;
  buf[5] = Math.min(kRadius, MAX_KERNEL_RADIUS);
  buf[6] = 0.0;
  buf[7] = 0.0;
  return buf;
}

/** Pack upsample uniforms (32 bytes = 8 × f32). */
function packUpsampleUniforms(
  w: number, h: number,
  radius: number, intensity: number,
  tint: [number, number, number],
): Float32Array {
  const buf = new Float32Array(8);
  buf[0] = 1.0 / Math.max(w, 1);
  buf[1] = 1.0 / Math.max(h, 1);
  buf[2] = radius;
  buf[3] = intensity;
  buf[4] = tint[0];
  buf[5] = tint[1];
  buf[6] = tint[2];
  buf[7] = 0.0;
  return buf;
}

/** Pack composite uniforms (32 bytes = 8 × f32). */
function packCompositeUniforms(p: ResolvedParams): Float32Array {
  const buf = new Float32Array(8);
  buf[0] = p.bloomStrength;
  buf[1] = p.bloomRadius;
  buf[2] = 0.0;
  buf[3] = 0.0;
  buf[4] = p.bloomTintColor[0];
  buf[5] = p.bloomTintColor[1];
  buf[6] = p.bloomTintColor[2];
  buf[7] = 0.0;
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mip-level size helper
// ─────────────────────────────────────────────────────────────────────────────

/** Compute the resolution at a given mip level (integer division, min 1). */
function mipSize(base: number, level: number): number {
  return Math.max(1, base >> level);
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: per-mip GPU resource set
// ─────────────────────────────────────────────────────────────────────────────

interface MipResources {
  width   : number;
  height  : number;
  downTex : GPUTexture;
  blurHTex: GPUTexture;
  blurVTex: GPUTexture;
  upTex   : GPUTexture;
  downUniformBuf   : GPUBuffer;
  gaussHUniformBuf : GPUBuffer;
  gaussVUniformBuf : GPUBuffer;
  upUniformBuf     : GPUBuffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: GPU resource factory helpers
// ─────────────────────────────────────────────────────────────────────────────

function createRenderTex(
  device: GPUDevice, label: string, w: number, h: number,
): GPUTexture {
  return device.createTexture({
    label,
    size  : { width: Math.max(1, w), height: Math.max(1, h) },
    format: 'rgba16float',
    usage : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

function createUniformBuf(
  device: GPUDevice, label: string, data: Float32Array,
): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size  : data.byteLength,
    usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: bind group layout factories
// ─────────────────────────────────────────────────────────────────────────────

/** Standard 3-binding BGL: uniform + sampler + texture_2d (shared by stages 1-3). */
function createSingleTexBGL(device: GPUDevice, label: string): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
        buffer : { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT,
        sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' } },
    ],
  });
}

/** 4-binding BGL: uniform + sampler + 2 × texture_2d (stages 4-5). */
function createDualTexBGL(device: GPUDevice, label: string): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: render pass helper — eliminates boilerplate for fullscreen draws
// ─────────────────────────────────────────────────────────────────────────────

/** Execute a fullscreen-triangle draw with a bind group onto a render target. */
function runFullscreenPass(
  encoder  : GPUCommandEncoder,
  pipeline : GPURenderPipeline,
  bg       : GPUBindGroup,
  target   : GPUTextureView,
  label    : string,
): void {
  const pass = encoder.beginRenderPass({
    label,
    colorAttachments: [{
      view      : target,
      loadOp    : 'clear' as GPULoadOp,
      storeOp   : 'store' as GPUStoreOp,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.draw(3);
  pass.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// ATUnrealBloomPipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU UnrealBloom post-process pipeline — full AT / UE-style implementation.
 *
 * Pipeline stages:
 *   1. Luminosity threshold extraction (UnrealBloomLuminosity.glsl)
 *   2. 5-level downsample pyramid — 13-tap weighted filter (DownSample.glsl)
 *   3. Separable gaussian blur H+V per mip (UnrealBloomGaussian.glsl)
 *   4. Upsample chain — 9-tap tent filter + tint accumulation (UpSample.glsl)
 *   5. Composite — lerpBloomFactor + additive blend (UnrealBloomComposite.glsl)
 */
export class ATUnrealBloomPipeline {
  // ── Core GPU state ─────────────────────────────────────────────────────────
  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;
  private width           : number;
  private height          : number;

  // ── Shared resources ───────────────────────────────────────────────────────
  private sampler             : GPUSampler;
  private luminosityUniformBuf: GPUBuffer;
  private compositeUniformBuf : GPUBuffer;

  // ── Textures ───────────────────────────────────────────────────────────────
  private brightTex : GPUTexture;
  private blackTex  : GPUTexture;

  // ── Per-mip resources ──────────────────────────────────────────────────────
  private mips : MipResources[];

  // ── Render pipelines ───────────────────────────────────────────────────────
  private luminosityPipeline : GPURenderPipeline;
  private downsamplePipeline : GPURenderPipeline;
  private gaussianPipeline   : GPURenderPipeline;
  private upsamplePipeline   : GPURenderPipeline;
  private compositePipeline  : GPURenderPipeline;

  // ── Bind group layouts ─────────────────────────────────────────────────────
  private singleTexBGL : GPUBindGroupLayout;   // stages 1-3: uniform+sampler+tex
  private dualTexBGL   : GPUBindGroupLayout;   // stages 4-5: uniform+sampler+tex+tex

  // ── Params ─────────────────────────────────────────────────────────────────
  private params : ResolvedParams;

  // ── Private constructor ────────────────────────────────────────────────────

  private constructor(init: {
    device: GPUDevice; format: GPUTextureFormat;
    width: number; height: number;
    sampler: GPUSampler;
    luminosityUniformBuf: GPUBuffer; compositeUniformBuf: GPUBuffer;
    brightTex: GPUTexture; blackTex: GPUTexture;
    mips: MipResources[];
    luminosityPipeline: GPURenderPipeline; downsamplePipeline: GPURenderPipeline;
    gaussianPipeline: GPURenderPipeline; upsamplePipeline: GPURenderPipeline;
    compositePipeline: GPURenderPipeline;
    singleTexBGL: GPUBindGroupLayout; dualTexBGL: GPUBindGroupLayout;
    params: ResolvedParams;
  }) {
    this.device              = init.device;
    this.format              = init.format;
    this.width               = init.width;
    this.height              = init.height;
    this.sampler             = init.sampler;
    this.luminosityUniformBuf= init.luminosityUniformBuf;
    this.compositeUniformBuf = init.compositeUniformBuf;
    this.brightTex           = init.brightTex;
    this.blackTex            = init.blackTex;
    this.mips                = init.mips;
    this.luminosityPipeline  = init.luminosityPipeline;
    this.downsamplePipeline  = init.downsamplePipeline;
    this.gaussianPipeline    = init.gaussianPipeline;
    this.upsamplePipeline    = init.upsamplePipeline;
    this.compositePipeline   = init.compositePipeline;
    this.singleTexBGL        = init.singleTexBGL;
    this.dualTexBGL          = init.dualTexBGL;
    this.params              = init.params;
  }

  // ── Static factory ─────────────────────────────────────────────────────────

  /**
   * Async factory — compiles all five WGSL pipelines and allocates the
   * full mip pyramid of intermediate textures.
   */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<ATUnrealBloomPipeline> {

    const params = resolveDefaults();

    // ── Sampler (bilinear clamp — matches UE5 SF_Bilinear + AM_Clamp) ────────
    const sampler = device.createSampler({
      label       : 'at-ub-sampler',
      magFilter   : 'linear',
      minFilter   : 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // ── Shared uniform buffers ───────────────────────────────────────────────
    const luminosityUniformBuf = createUniformBuf(
      device, 'at-ub-luminosity-uniforms',
      packLuminosityUniforms(params),
    );

    const compositeUniformBuf = createUniformBuf(
      device, 'at-ub-composite-uniforms',
      packCompositeUniforms(params),
    );

    // ── Textures ─────────────────────────────────────────────────────────────
    const brightTex = createRenderTex(device, 'at-ub-bright', width, height);

    // 1×1 black texture for the first upsample (no accumulator yet)
    const blackTex = device.createTexture({
      label : 'at-ub-black',
      size  : { width: 1, height: 1 },
      format: 'rgba16float',
      usage : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    {
      const enc = device.createCommandEncoder({ label: 'at-ub-clear-black' });
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view      : blackTex.createView(),
          loadOp    : 'clear' as GPULoadOp,
          storeOp   : 'store' as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    // ── Per-mip resources ────────────────────────────────────────────────────
    const mips: MipResources[] = [];
    for (let i = 0; i < MIP_LEVELS; i++) {
      const mw = mipSize(width,  i + 1);
      const mh = mipSize(height, i + 1);

      const downTex  = createRenderTex(device, `at-ub-down-${i}`,  mw, mh);
      const blurHTex = createRenderTex(device, `at-ub-blurH-${i}`, mw, mh);
      const blurVTex = createRenderTex(device, `at-ub-blurV-${i}`, mw, mh);
      const upTex    = createRenderTex(device, `at-ub-up-${i}`,    mw, mh);

      const srcW = (i === 0) ? width  : mipSize(width,  i);
      const srcH = (i === 0) ? height : mipSize(height, i);

      const downUniformBuf = createUniformBuf(
        device, `at-ub-down-u-${i}`, packDownsampleUniforms(srcW, srcH));

      const gaussHUniformBuf = createUniformBuf(
        device, `at-ub-gaussH-u-${i}`,
        packGaussianUniforms(mw, mh, true, params.gaussianSigma, params.gaussianKernelRadius));

      const gaussVUniformBuf = createUniformBuf(
        device, `at-ub-gaussV-u-${i}`,
        packGaussianUniforms(mw, mh, false, params.gaussianSigma, params.gaussianKernelRadius));

      const tint = (params.mipTints[i] || [1, 1, 1]) as [number, number, number];
      const upUniformBuf = createUniformBuf(
        device, `at-ub-up-u-${i}`,
        packUpsampleUniforms(mw, mh, params.upsampleRadius, params.upsampleIntensity, tint));

      mips.push({
        width: mw, height: mh,
        downTex, blurHTex, blurVTex, upTex,
        downUniformBuf, gaussHUniformBuf, gaussVUniformBuf, upUniformBuf,
      });
    }

    // ── Bind group layouts ───────────────────────────────────────────────────
    const singleTexBGL = createSingleTexBGL(device, 'at-ub-single-bgl');
    const dualTexBGL   = createDualTexBGL(device, 'at-ub-dual-bgl');

    // ── Pipeline compilation helper ──────────────────────────────────────────
    function makePipeline(
      label: string, wgsl: string, bgl: GPUBindGroupLayout, outFormat: GPUTextureFormat,
    ): GPURenderPipeline {
      const module = device.createShaderModule({ label, code: wgsl });
      return device.createRenderPipeline({
        label,
        layout   : device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex   : { module, entryPoint: 'vs_main' },
        fragment : { module, entryPoint: 'fs_main', targets: [{ format: outFormat }] },
        primitive: { topology: 'triangle-list' },
      });
    }

    // ── Compile all five pipelines ───────────────────────────────────────────
    const luminosityPipeline = makePipeline(
      'at-ub-luminosity', WGSL_LUMINOSITY_THRESHOLD, singleTexBGL, 'rgba16float');
    const downsamplePipeline = makePipeline(
      'at-ub-downsample', WGSL_DOWNSAMPLE, singleTexBGL, 'rgba16float');
    const gaussianPipeline = makePipeline(
      'at-ub-gaussian', WGSL_GAUSSIAN_BLUR, singleTexBGL, 'rgba16float');
    const upsamplePipeline = makePipeline(
      'at-ub-upsample', WGSL_UPSAMPLE, dualTexBGL, 'rgba16float');
    const compositePipeline = makePipeline(
      'at-ub-composite', WGSL_COMPOSITE, dualTexBGL, format);

    return new ATUnrealBloomPipeline({
      device, format, width, height, sampler,
      luminosityUniformBuf, compositeUniformBuf,
      brightTex, blackTex, mips,
      luminosityPipeline, downsamplePipeline,
      gaussianPipeline, upsamplePipeline, compositePipeline,
      singleTexBGL, dualTexBGL, params,
    });
  }

  // ── Public API: setParams ──────────────────────────────────────────────────

  /**
   * Update bloom parameters. Flushes all affected uniform buffers to the GPU.
   * Safe to call every frame (no diffing — always writes).
   */
  setParams(partial: ATUnrealBloomParams): void {
    const p = this.params;

    if (partial.bloomStrength        !== undefined) p.bloomStrength        = partial.bloomStrength;
    if (partial.bloomRadius          !== undefined) p.bloomRadius          = partial.bloomRadius;
    if (partial.bloomTintColor       !== undefined) p.bloomTintColor       = partial.bloomTintColor;
    if (partial.luminosityThreshold  !== undefined) p.luminosityThreshold  = partial.luminosityThreshold;
    if (partial.smoothWidth          !== undefined) p.smoothWidth          = partial.smoothWidth;
    if (partial.defaultColor         !== undefined) p.defaultColor         = partial.defaultColor;
    if (partial.defaultOpacity       !== undefined) p.defaultOpacity       = partial.defaultOpacity;
    if (partial.gaussianSigma        !== undefined) p.gaussianSigma        = partial.gaussianSigma;
    if (partial.gaussianKernelRadius !== undefined) p.gaussianKernelRadius = partial.gaussianKernelRadius;
    if (partial.upsampleRadius       !== undefined) p.upsampleRadius       = partial.upsampleRadius;
    if (partial.upsampleIntensity    !== undefined) p.upsampleIntensity    = partial.upsampleIntensity;
    if (partial.mipTints             !== undefined) p.mipTints             = partial.mipTints;

    // Flush luminosity uniforms
    this.device.queue.writeBuffer(this.luminosityUniformBuf, 0, packLuminosityUniforms(p));

    // Flush composite uniforms
    this.device.queue.writeBuffer(this.compositeUniformBuf, 0, packCompositeUniforms(p));

    // Flush per-mip gaussian + upsample uniforms
    for (let i = 0; i < this.mips.length; i++) {
      const m = this.mips[i];

      this.device.queue.writeBuffer(m.gaussHUniformBuf, 0,
        packGaussianUniforms(m.width, m.height, true, p.gaussianSigma, p.gaussianKernelRadius));

      this.device.queue.writeBuffer(m.gaussVUniformBuf, 0,
        packGaussianUniforms(m.width, m.height, false, p.gaussianSigma, p.gaussianKernelRadius));

      const tint = (p.mipTints[i] || [1, 1, 1]) as [number, number, number];
      this.device.queue.writeBuffer(m.upUniformBuf, 0,
        packUpsampleUniforms(m.width, m.height, p.upsampleRadius, p.upsampleIntensity, tint));
    }
  }

  // ── Public API: resize ─────────────────────────────────────────────────────

  /**
   * Resize all internal textures. Call when the canvas / viewport size changes.
   * Destroys and recreates all intermediate textures and updates
   * resolution-dependent uniform buffers.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;

    const p = this.params;

    // Recreate full-resolution brightTex
    this.brightTex.destroy();
    this.brightTex = createRenderTex(this.device, 'at-ub-bright', width, height);

    // Recreate per-mip textures and update uniform buffers
    for (let i = 0; i < this.mips.length; i++) {
      const m = this.mips[i];
      const mw = mipSize(width,  i + 1);
      const mh = mipSize(height, i + 1);

      m.downTex.destroy(); m.blurHTex.destroy();
      m.blurVTex.destroy(); m.upTex.destroy();

      m.downTex  = createRenderTex(this.device, `at-ub-down-${i}`,  mw, mh);
      m.blurHTex = createRenderTex(this.device, `at-ub-blurH-${i}`, mw, mh);
      m.blurVTex = createRenderTex(this.device, `at-ub-blurV-${i}`, mw, mh);
      m.upTex    = createRenderTex(this.device, `at-ub-up-${i}`,    mw, mh);

      m.width  = mw;
      m.height = mh;

      const srcW = (i === 0) ? width  : mipSize(width,  i);
      const srcH = (i === 0) ? height : mipSize(height, i);
      this.device.queue.writeBuffer(m.downUniformBuf, 0, packDownsampleUniforms(srcW, srcH));

      this.device.queue.writeBuffer(m.gaussHUniformBuf, 0,
        packGaussianUniforms(mw, mh, true, p.gaussianSigma, p.gaussianKernelRadius));
      this.device.queue.writeBuffer(m.gaussVUniformBuf, 0,
        packGaussianUniforms(mw, mh, false, p.gaussianSigma, p.gaussianKernelRadius));

      const tint = (p.mipTints[i] || [1, 1, 1]) as [number, number, number];
      this.device.queue.writeBuffer(m.upUniformBuf, 0,
        packUpsampleUniforms(mw, mh, p.upsampleRadius, p.upsampleIntensity, tint));
    }
  }

  // ── Private: stages 1–4 (shared between render() and renderBloomOnly()) ───

  /**
   * Execute stages 1–4 of the bloom pipeline, producing the fully
   * accumulated bloom in mips[0].upTex.
   *
   * Stage 1: Luminosity threshold — scene → brightTex
   *   Port of compiled.vs :: UnrealBloomLuminosity.glsl / BloomLuminosityPass.glsl
   *   Extracts bright pixels via luma() + smoothstep.
   *
   * Stage 2: Downsample pyramid — brightTex → mip[0..4].downTex
   *   Port of compiled.vs :: DownSample.glsl
   *   5 levels of 2× downsampling using 13-tap weighted filter.
   *   Weights [1/32, 1/16, 1/8] with halfPixel bilinear offsets.
   *
   * Stage 3: Gaussian blur — mip[i].downTex → blurHTex → blurVTex
   *   Port of compiled.vs :: UnrealBloomGaussian.glsl
   *   Separable H+V blur per mip using gaussianPdf() dynamic weights.
   *
   * Stage 4: Upsample chain — mip[4].blurV → ... → mip[0].upTex
   *   Port of compiled.vs :: UpSample.glsl
   *   9-tap tent filter + tNext accumulation + uTint per-mip coloring.
   */
  private _renderStages1to4(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
  ): void {
    const { device, sampler, mips } = this;
    const sceneView  = sceneTex.createView();
    const brightView = this.brightTex.createView();

    // ── Stage 1: LUMINOSITY THRESHOLD ────────────────────────────────────────
    {
      const bg = device.createBindGroup({
        label  : 'at-ub-luminosity-bg',
        layout : this.singleTexBGL,
        entries: [
          { binding: 0, resource: { buffer: this.luminosityUniformBuf } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: sceneView },
        ],
      });
      runFullscreenPass(encoder, this.luminosityPipeline, bg, brightView,
        'at-ub-luminosity-pass');
    }

    // ── Stage 2: DOWNSAMPLE PYRAMID ──────────────────────────────────────────
    for (let i = 0; i < MIP_LEVELS; i++) {
      const m = mips[i];
      const srcTex = (i === 0) ? this.brightTex : mips[i - 1].downTex;

      const bg = device.createBindGroup({
        label  : `at-ub-down-bg-${i}`,
        layout : this.singleTexBGL,
        entries: [
          { binding: 0, resource: { buffer: m.downUniformBuf } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: srcTex.createView() },
        ],
      });
      runFullscreenPass(encoder, this.downsamplePipeline, bg,
        m.downTex.createView(), `at-ub-down-pass-${i}`);
    }

    // ── Stage 3: GAUSSIAN BLUR (H + V per mip) ──────────────────────────────
    for (let i = 0; i < MIP_LEVELS; i++) {
      const m = mips[i];

      // Horizontal blur: downTex → blurHTex
      {
        const bg = device.createBindGroup({
          label  : `at-ub-gaussH-bg-${i}`,
          layout : this.singleTexBGL,
          entries: [
            { binding: 0, resource: { buffer: m.gaussHUniformBuf } },
            { binding: 1, resource: sampler },
            { binding: 2, resource: m.downTex.createView() },
          ],
        });
        runFullscreenPass(encoder, this.gaussianPipeline, bg,
          m.blurHTex.createView(), `at-ub-gaussH-pass-${i}`);
      }

      // Vertical blur: blurHTex → blurVTex
      {
        const bg = device.createBindGroup({
          label  : `at-ub-gaussV-bg-${i}`,
          layout : this.singleTexBGL,
          entries: [
            { binding: 0, resource: { buffer: m.gaussVUniformBuf } },
            { binding: 1, resource: sampler },
            { binding: 2, resource: m.blurHTex.createView() },
          ],
        });
        runFullscreenPass(encoder, this.gaussianPipeline, bg,
          m.blurVTex.createView(), `at-ub-gaussV-pass-${i}`);
      }
    }

    // ── Stage 4: UPSAMPLE CHAIN ──────────────────────────────────────────────
    for (let i = MIP_LEVELS - 1; i >= 0; i--) {
      const m = mips[i];
      const nextTex = (i === MIP_LEVELS - 1) ? this.blackTex : mips[i + 1].upTex;

      const bg = device.createBindGroup({
        label  : `at-ub-up-bg-${i}`,
        layout : this.dualTexBGL,
        entries: [
          { binding: 0, resource: { buffer: m.upUniformBuf } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: m.blurVTex.createView() },
          { binding: 3, resource: nextTex.createView() },
        ],
      });
      runFullscreenPass(encoder, this.upsamplePipeline, bg,
        m.upTex.createView(), `at-ub-up-pass-${i}`);
    }
  }

  // ── Private: stage 5 composite ─────────────────────────────────────────────

  /**
   * Stage 5: Composite — scene + bloom → output.
   * Port of compiled.vs :: UnrealBloomComposite.glsl + UnrealBloomPass.fs
   *
   * UnrealBloomComposite applies lerpBloomFactor weighting and tint:
   *   lerpBloomFactor(factor) = mix(factor, 1.2 - factor, bloomRadius)
   *   result = bloomStrength × lerpBloomFactor(1.0) × bloomTintColor × bloom
   *
   * UnrealBloomPass adds it to the scene:
   *   color.rgb += getUnrealBloom(vUv)
   *
   * The final bloom texture is mips[0].upTex — the fully accumulated
   * upsample result at the finest mip level.
   */
  private _renderComposite(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    const bloomTex = this.mips[0].upTex;

    const bg = this.device.createBindGroup({
      label  : 'at-ub-composite-bg',
      layout : this.dualTexBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: sceneTex.createView() },
        { binding: 3, resource: bloomTex.createView() },
      ],
    });

    const pass = encoder.beginRenderPass({
      label           : 'at-ub-composite-pass',
      colorAttachments: [{
        view      : dstView,
        loadOp    : 'clear' as GPULoadOp,
        storeOp   : 'store' as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  // ── Public API: render ─────────────────────────────────────────────────────

  /**
   * Execute the full 5-stage UnrealBloom pipeline for one frame.
   *
   * Records all render passes onto the provided command encoder.
   * The caller is responsible for encoder.finish() and device.queue.submit().
   *
   * Pass sequence:
   *   1. Luminosity threshold: scene → brightTex
   *   2. Downsample pyramid: brightTex → mip[0..4].downTex
   *   3. Gaussian blur: mip[i].downTex → mip[i].blurHTex → mip[i].blurVTex
   *   4. Upsample chain: mip[4].blurVTex → ... → mip[0].upTex
   *   5. Composite: scene + mip[0].upTex → dstView
   */
  render(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    this._renderStages1to4(encoder, sceneTex);
    this._renderComposite(encoder, sceneTex, dstView);
  }

  // ── Public API: destroy ────────────────────────────────────────────────────

  /**
   * Release all GPU resources held by this pipeline.
   * After calling destroy(), the instance must not be used.
   */
  destroy(): void {
    this.luminosityUniformBuf.destroy();
    this.compositeUniformBuf.destroy();
    this.brightTex.destroy();
    this.blackTex.destroy();

    for (const m of this.mips) {
      m.downTex.destroy();
      m.blurHTex.destroy();
      m.blurVTex.destroy();
      m.upTex.destroy();
      m.downUniformBuf.destroy();
      m.gaussHUniformBuf.destroy();
      m.gaussVUniformBuf.destroy();
      m.upUniformBuf.destroy();
    }
    this.mips.length = 0;
  }

  // ── Public API: getParams ──────────────────────────────────────────────────

  /** Returns a read-only snapshot of the currently resolved parameters. */
  getParams(): Readonly<ResolvedParams> {
    return { ...this.params };
  }

  // ── Public API: getMipCount ────────────────────────────────────────────────

  /** Returns the number of mip levels in the downsample/upsample pyramid. */
  get mipCount(): number {
    return MIP_LEVELS;
  }

  // ── Public API: getMipResolution ───────────────────────────────────────────

  /** Returns the resolution [width, height] of a given mip level. */
  getMipResolution(level: number): [number, number] {
    const idx = Math.max(0, Math.min(level, this.mips.length - 1));
    const m = this.mips[idx];
    return [m.width, m.height];
  }

  // ── Public API: getBloomTexture ────────────────────────────────────────────

  /**
   * Returns the final upsampled bloom GPUTexture (before composite).
   * Useful for debugging, custom compositing, or feeding into additional
   * post-process passes.
   */
  getBloomTexture(): GPUTexture {
    return this.mips[0].upTex;
  }

  // ── Public API: getMipTextures ─────────────────────────────────────────────

  /**
   * Returns references to all intermediate textures at a given mip level.
   * Useful for debugging the pipeline stages individually.
   */
  getMipTextures(level: number): {
    downTex: GPUTexture; blurHTex: GPUTexture;
    blurVTex: GPUTexture; upTex: GPUTexture;
  } {
    const idx = Math.max(0, Math.min(level, this.mips.length - 1));
    const m = this.mips[idx];
    return { downTex: m.downTex, blurHTex: m.blurHTex, blurVTex: m.blurVTex, upTex: m.upTex };
  }

  // ── Public API: renderBloomOnly ────────────────────────────────────────────

  /**
   * Execute only the bloom extraction + blur stages (1–4) without the final
   * composite. The result is left in the upsampled bloom texture, retrievable
   * via getBloomTexture().
   *
   * Use when compositing bloom yourself in a custom shader, or when chaining
   * multiple post-process effects.
   */
  renderBloomOnly(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
  ): void {
    this._renderStages1to4(encoder, sceneTex);
  }

  // ── Public API: compositeOnly ──────────────────────────────────────────────

  /**
   * Execute only the composite stage (5), reading from the already-rendered
   * bloom texture. Call after renderBloomOnly() when you've done additional
   * processing on the bloom or scene textures.
   */
  compositeOnly(
    encoder  : GPUCommandEncoder,
    sceneTex : GPUTexture,
    dstView  : GPUTextureView,
  ): void {
    this._renderComposite(encoder, sceneTex, dstView);
  }

  // ── Diagnostics ────────────────────────────────────────────────────────────

  /** Returns all five WGSL shader source strings for inspection or hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      luminosityThreshold : WGSL_LUMINOSITY_THRESHOLD,
      downsample          : WGSL_DOWNSAMPLE,
      gaussianBlur        : WGSL_GAUSSIAN_BLUR,
      upsample            : WGSL_UPSAMPLE,
      composite           : WGSL_COMPOSITE,
    };
  }

  /** Returns a human-readable summary of the pipeline configuration. */
  toString(): string {
    const p = this.params;
    const lines = [
      `ATUnrealBloomPipeline [${this.width}×${this.height}]`,
      `  mipLevels          : ${MIP_LEVELS}`,
      `  bloomStrength      : ${p.bloomStrength}`,
      `  bloomRadius        : ${p.bloomRadius}`,
      `  bloomTintColor     : [${p.bloomTintColor.join(', ')}]`,
      `  luminosityThreshold: ${p.luminosityThreshold}`,
      `  smoothWidth        : ${p.smoothWidth}`,
      `  gaussianSigma      : ${p.gaussianSigma}`,
      `  gaussianKernelRadius: ${p.gaussianKernelRadius}`,
      `  upsampleRadius     : ${p.upsampleRadius}`,
      `  upsampleIntensity  : ${p.upsampleIntensity}`,
    ];
    for (let i = 0; i < MIP_LEVELS; i++) {
      const m = this.mips[i];
      const tint = p.mipTints[i] || [1, 1, 1];
      lines.push(`  mip[${i}]: ${m.width}×${m.height}  tint=[${tint.join(', ')}]`);
    }
    return lines.join('\n');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — integrates with BloomVariants.ts species params
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an ATUnrealBloomPipeline seeded from BloomParams (BloomVariants.ts),
 * providing sensible default mappings from AT bloom parameters to the full
 * UE-style pipeline.
 *
 * Parameter mapping:
 *   - bloomStrength → bloomStrength (direct)
 *   - bloomRadius → bloomRadius (direct, controls lerpBloomFactor)
 *   - luminosityThreshold → luminosityThreshold (direct)
 *   - bloomTintColor → bloomTintColor (default: white)
 *   - gaussianSigma → derived from bloomRadius (wider radius → higher sigma)
 */
export async function createATUnrealBloomForSpecies(
  device      : GPUDevice,
  format      : GPUTextureFormat,
  width       : number,
  height      : number,
  bloomParams?: {
    bloomStrength        : number;
    bloomRadius          : number;
    luminosityThreshold  : number;
    bloomTintColor?      : [number, number, number];
  },
): Promise<ATUnrealBloomPipeline> {
  const pipeline = await ATUnrealBloomPipeline.create(device, format, width, height);

  if (bloomParams) {
    // Map AT bloomRadius [0,1] to gaussian sigma [1,6] for visually
    // consistent bloom spread across the mip pyramid.
    const sigmaFromRadius = 1.0 + bloomParams.bloomRadius * 5.0;

    pipeline.setParams({
      bloomStrength       : bloomParams.bloomStrength,
      bloomRadius         : bloomParams.bloomRadius,
      luminosityThreshold : bloomParams.luminosityThreshold,
      bloomTintColor      : bloomParams.bloomTintColor || [1, 1, 1],
      gaussianSigma       : sigmaFromRadius,
    });
  }

  return pipeline;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Individual WGSL shader source strings for each stage of the pipeline.
 * Useful for embedding bloom stages in custom shader pipelines,
 * hot-reload integration, and shader debugging.
 */
export const AT_UNREAL_BLOOM_WGSL = {
  /** Fullscreen triangle vertex shader (shared by all stages). */
  fullscreenVertex     : WGSL_FULLSCREEN_TRI_VERTEX,
  /** Rec.601 luma helper function. */
  luma                 : WGSL_LUMA,
  /** Stage 1: luminosity threshold extraction fragment shader. */
  luminosityThreshold  : WGSL_LUMINOSITY_THRESHOLD,
  /** Stage 2: 13-tap weighted downsample fragment shader. */
  downsample           : WGSL_DOWNSAMPLE,
  /** Stage 3: separable gaussian blur fragment shader. */
  gaussianBlur         : WGSL_GAUSSIAN_BLUR,
  /** Stage 4: 9-tap tent filter upsample fragment shader. */
  upsample             : WGSL_UPSAMPLE,
  /** Stage 5: lerpBloomFactor composite fragment shader. */
  composite            : WGSL_COMPOSITE,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Type re-exports for downstream consumers
// ─────────────────────────────────────────────────────────────────────────────

export type { ResolvedParams as ATUnrealBloomResolvedParams };
