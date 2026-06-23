/**
 * temporal-reprojection.ts — M794: Temporal Anti-Aliasing (TAA)
 * ─────────────────────────────────────────────────────────────────────────────
 * 时间抗锯齿——利用上一帧的颜色 buffer + motion vector 做 temporal accumulation。
 * Jitter camera + reject outliers + tonemap 前 blend。
 *
 * 算法概览
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Motion Vector Generation ────────────────────────────────────┐
 *   │  当前帧 depth + 当前/上一帧 viewProj 矩阵 → per-pixel screen-space     │
 *   │  motion vector (velocity buffer).                                       │
 *   │  若上游已提供 motion vector texture 可直接注入，跳过此 pass。            │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ motionVecTex (rg16float: dx, dy in NDC)
 *                ▼
 *   ┌─ Pass 1 ── Temporal Accumulation & Resolve ────────────────────────────┐
 *   │  1. Jitter camera sub-pixel offset (Halton 2,3 低差异序列)              │
 *   │  2. 用 motion vector 将当前像素反投影到上一帧 history buffer            │
 *   │  3. 对 history 采样做 neighbourhood clamp / clip (YCoCg AABB)          │
 *   │     → 拒绝 ghosting / disocclusion 产生的过时像素                       │
 *   │  4. Tonemap 前的 HDR 权重空间混合 (Karis 2014)                          │
 *   │     current = tonemapped current, history = tonemapped clamped history  │
 *   │     result = lerp(history, current, α) 其中 α ∈ [0.04, 0.20]          │
 *   │  5. Inverse tonemap 写回 history buffer + 输出到 resolve target        │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ resolvedTex → 下游 post-process chain
 *                │ historyTex  ← ping-pong 保存到下一帧
 *
 * Jitter 序列
 * ─────────────────────────────────────────────────────────────────────────────
 * 使用 Halton(2,3) 低差异序列生成 sub-pixel jitter offset，注入到 projection
 * matrix 的 [2][0] 和 [2][1] 分量。序列长度 16 帧后循环。
 *
 * Neighbourhood Clamp (YCoCg Variance Clip)
 * ─────────────────────────────────────────────────────────────────────────────
 * 在 3×3 neighbourhood 内计算当前帧颜色的 min/max AABB (in YCoCg space)，
 * 将 history 颜色 clip 到该 AABB 内。相比简单 clamp，variance clip 对
 * 运动模糊和半透明粒子更友好 (Salvi, SIGGRAPH 2016)。
 *
 * HDR Weight Blend (Karis 2014)
 * ─────────────────────────────────────────────────────────────────────────────
 * 在 blend 前对 current 和 history 分别做 Reinhard tonemap → blend → inverse
 * tonemap，避免 HDR 高光在累积过程中产生 firefly 闪烁。权重函数：
 *   w(c) = 1 / (1 + luminance(c))
 *
 * 设计决策
 * ─────────────────────────────────────────────────────────────────────────────
 * • Motion vector pass 独立，可被外部管线提供的 velocity buffer 替代。
 * • Ping-pong history 双缓冲避免 read-after-write hazard。
 * • YCoCg 色彩空间比 RGB 更紧凑，AABB clip 更精准，减少 color bleeding。
 * • Blend factor α 随 motion magnitude 动态调整：静止 → 0.04（更多累积），
 *   快速运动 → 0.20（偏向当前帧），避免拖影。
 * • Halton 16 帧循环在覆盖率与收敛速度之间取得平衡。
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/post-process.ts               — fullscreen-quad + bind group
 *   src/lib/sph/tone-mapping.ts               — ACES filmic tone mapping
 *   src/lib/sph/dof-bokeh.ts                  — WebGPU multi-pass pipeline
 *   src/lib/sph/screen-space-reflections.ts   — temporal filter pattern
 *   src/lib/sph/velocity-field-texture.ts     — velocity texture conventions
 *   src/lib/sph/render-compositor.ts          — pass orchestration (13-pass)
 *
 * Reference papers & talks:
 *   "High Quality Temporal Supersampling" — Brian Karis, SIGGRAPH 2014 (UE4)
 *   "An Excursion in Temporal Supersampling" — Marco Salvi, GDC 2016
 *   "Temporal Reprojection Anti-Aliasing in INSIDE" — Pedersen, GDC 2016
 *   "A Survey of Temporal Antialiasing Techniques" — Yang et al., 2020
 *   "Filmic SMAA" — Jimenez, SIGGRAPH 2012 (enhanced temporal component)
 *
 * Research: xiaodi #M794 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Tunable TAA parameters. */
export interface TAAParams {
  /**
   * Minimum blend factor (α) for static pixels.
   * Lower → more temporal accumulation → smoother but risks ghosting.
   * @default 0.04
   */
  blendFactorMin?: number;

  /**
   * Maximum blend factor (α) for fast-moving pixels.
   * Higher → favours current frame → less ghosting but noisier.
   * @default 0.20
   */
  blendFactorMax?: number;

  /**
   * Motion vector magnitude (in NDC) at which blend reaches blendFactorMax.
   * Controls the motion-adaptive ramp.
   * @default 0.02
   */
  motionScale?: number;

  /**
   * Halton jitter sequence length (power of 2 recommended, max 64).
   * Longer → more sample coverage before repeat, but slower visual
   * convergence per cycle. 16 is the sweet spot for most scenes.
   * @default 16
   */
  jitterSequenceLength?: number;

  /**
   * Variance clip gamma — multiplier on the stddev box for the
   * YCoCg neighbourhood AABB. 1.0 = tight clip (aggressive rejection),
   * higher = more permissive (allow more history reuse).
   * @default 1.0
   */
  varianceClipGamma?: number;

  /**
   * When true, jitter is applied to the projection matrix.
   * Turn off for debug or when the upstream camera already jitters.
   * @default true
   */
  enableJitter?: boolean;

  /**
   * When true, use the Karis HDR weight function (1/(1+lum)) during
   * blend to suppress firefly flicker in high-dynamic-range scenes.
   * @default true
   */
  enableHDRWeight?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Halton low-discrepancy sequence
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute element `index` of the Halton sequence with the given `base`.
 * Returns a value in [0, 1).
 */
function halton(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

/** Pre-computed jitter offsets table: Halton(2,3), centred around 0. */
function buildJitterTable(length: number): Float32Array {
  const table = new Float32Array(length * 2);
  for (let i = 0; i < length; i++) {
    table[i * 2]     = halton(i + 1, 2) - 0.5;  // x: Halton base-2
    table[i * 2 + 1] = halton(i + 1, 3) - 0.5;  // y: Halton base-3
  }
  return table;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader: Motion Vector Generation
// ─────────────────────────────────────────────────────────────────────────────

const MOTION_VEC_WGSL = /* wgsl */`
// ─── Uniforms ──────────────────────────────────────────────────────────────
struct MotionUniforms {
  // Current frame view-projection matrix (4×4, column-major)
  currViewProj : mat4x4f,
  // Previous frame view-projection matrix (4×4, column-major)
  prevViewProj : mat4x4f,
  // Inverse of current view-projection (for depth → world reconstruction)
  currInvViewProj : mat4x4f,
  // Resolution
  width  : f32,
  height : f32,
  _pad0  : f32,
  _pad1  : f32,
}

@group(0) @binding(0) var<uniform>  u         : MotionUniforms;
@group(0) @binding(1) var           smpNearest : sampler;
@group(0) @binding(2) var           depthTex   : texture_2d<f32>;

// ─── Full-screen quad vertex ──────────────────────────────────────────────
struct Vert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_motion(@builtin(vertex_index) vi: u32) -> Vert {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  var uv = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0),
  );
  var out: Vert;
  out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv  = uv[vi];
  return out;
}

// ─── Fragment: reconstruct world position from depth, reproject ──────────
@fragment fn fs_motion(in: Vert) -> @location(0) vec4f {
  let uv    = in.uv;
  let depth = textureSample(depthTex, smpNearest, uv).r;

  // NDC position of current fragment  (UV → [-1,1], depth stays [0,1])
  let ndcCurr = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);

  // Reconstruct world position via inverse view-projection
  let worldH = u.currInvViewProj * ndcCurr;
  let world  = worldH.xyz / worldH.w;

  // Project into previous frame's clip space
  let prevClip = u.prevViewProj * vec4f(world, 1.0);
  let prevNDC  = prevClip.xy / prevClip.w;

  // Motion vector = current NDC − previous NDC (screen-space velocity)
  let currNDC = vec2f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
  let motion  = currNDC - prevNDC;

  // Store as RG (motion.x, motion.y), BA unused (set to 0,1)
  return vec4f(motion, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader: Temporal Resolve (accumulation + neighbourhood clip)
// ─────────────────────────────────────────────────────────────────────────────

const TAA_RESOLVE_WGSL = /* wgsl */`
// ─── Uniforms ──────────────────────────────────────────────────────────────
struct TAAUniforms {
  width          : f32,
  height         : f32,
  blendFactorMin : f32,
  blendFactorMax : f32,
  motionScale    : f32,
  clipGamma      : f32,
  enableHDR      : f32,
  jitterX        : f32,
  jitterY        : f32,
  _pad0          : f32,
  _pad1          : f32,
  _pad2          : f32,
}

@group(0) @binding(0) var<uniform>  u          : TAAUniforms;
@group(0) @binding(1) var           smpLinear  : sampler;
@group(0) @binding(2) var           smpNearest : sampler;
@group(0) @binding(3) var           currentTex : texture_2d<f32>;  // jittered current frame
@group(0) @binding(4) var           historyTex : texture_2d<f32>;  // previous resolved frame
@group(0) @binding(5) var           motionTex  : texture_2d<f32>;  // motion vectors (rg)

// ─── Colour space conversions ─────────────────────────────────────────────

// RGB → YCoCg  (decorrelated, tighter AABB)
fn rgbToYCoCg(c: vec3f) -> vec3f {
  let y  = ( c.r + 2.0 * c.g + c.b) * 0.25;
  let co = ( c.r                - c.b) * 0.5;
  let cg = (-c.r + 2.0 * c.g - c.b) * 0.25;
  return vec3f(y, co, cg);
}

// YCoCg → RGB
fn ycocgToRGB(c: vec3f) -> vec3f {
  let y  = c.x;
  let co = c.y;
  let cg = c.z;
  return vec3f(y + co - cg, y + cg, y - co - cg);
}

// Luminance (Rec.709)
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Karis HDR weight: reduces contribution of very bright pixels
fn hdrWeight(c: vec3f) -> f32 {
  return 1.0 / (1.0 + luminance(c));
}

// Reinhard tonemap (for HDR-weighted blend)
fn reinhard(c: vec3f) -> vec3f {
  return c / (1.0 + luminance(c));
}

// Inverse Reinhard
fn inverseReinhard(c: vec3f) -> vec3f {
  return c / max(1.0 - luminance(c), 1e-6);
}

// ─── Variance clip in YCoCg ──────────────────────────────────────────────
// Clip the history colour to the AABB defined by the mean ± gamma*stddev
// of the 3×3 neighbourhood of the current frame (Salvi 2016).

fn clipAABB(aabbMin: vec3f, aabbMax: vec3f, histYCoCg: vec3f, avgYCoCg: vec3f) -> vec3f {
  // Ray from average toward history
  let dir = histYCoCg - avgYCoCg;
  let center = (aabbMin + aabbMax) * 0.5;
  let halfExtent = (aabbMax - aabbMin) * 0.5 + vec3f(1e-7);

  // Parametric clip: find t where the ray exits the AABB
  let invDir = 1.0 / (dir + vec3f(1e-7));
  let tMin = (aabbMin - avgYCoCg) * invDir;
  let tMax = (aabbMax - avgYCoCg) * invDir;
  let t0   = min(tMin, tMax);
  let t1   = max(tMin, tMax);
  let tEnter = max(t0.x, max(t0.y, t0.z));
  let tExit  = min(t1.x, min(t1.y, t1.z));

  // If the history is already inside the AABB, keep it
  if (tEnter >= tExit || tExit < 0.0) {
    return clamp(histYCoCg, aabbMin, aabbMax);
  }

  // Clip: walk along the ray to the AABB boundary
  let t = clamp(tExit, 0.0, 1.0);
  return avgYCoCg + dir * t;
}

// ─── Full-screen quad vertex ──────────────────────────────────────────────
struct Vert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_taa(@builtin(vertex_index) vi: u32) -> Vert {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  var uv = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0),
  );
  var out: Vert;
  out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv  = uv[vi];
  return out;
}

// ─── Fragment: temporal resolve ──────────────────────────────────────────
@fragment fn fs_taa(in: Vert) -> @location(0) vec4f {
  let uv    = in.uv;
  let pixel = vec2f(1.0 / u.width, 1.0 / u.height);

  // ── 1. Unjitter current sample: offset UV by negative jitter
  let unjitteredUV = uv - vec2f(u.jitterX, u.jitterY) * pixel;

  // ── 2. Sample motion vector and compute reprojected UV
  let motion     = textureSample(motionTex, smpNearest, uv).rg;
  // Motion is in NDC [-1,1], convert to UV [0,1] delta:
  let motionUV   = motion * vec2f(0.5, -0.5);
  let historyUV  = uv - motionUV;

  // ── 3. Sample current (unjittered) and history
  let currentRGB  = textureSample(currentTex, smpLinear, unjitteredUV).rgb;
  var historyRGB  = textureSample(historyTex, smpLinear, historyUV).rgb;

  // ── 4. Neighbourhood statistics (3×3) in YCoCg for variance clip
  var momM1 = vec3f(0.0);   // first moment  (mean)
  var momM2 = vec3f(0.0);   // second moment (variance)
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sampleUV = unjitteredUV + vec2f(f32(dx), f32(dy)) * pixel;
      let s = textureSample(currentTex, smpLinear, sampleUV).rgb;
      let sy = rgbToYCoCg(s);
      momM1 += sy;
      momM2 += sy * sy;
    }
  }
  momM1 /= 9.0;
  momM2 /= 9.0;
  let stddev = sqrt(max(momM2 - momM1 * momM1, vec3f(0.0)));

  // AABB = mean ± gamma * stddev
  let aabbMin = momM1 - stddev * u.clipGamma;
  let aabbMax = momM1 + stddev * u.clipGamma;

  // Clip history in YCoCg space
  let histYCoCg    = rgbToYCoCg(historyRGB);
  let currentYCoCg = rgbToYCoCg(currentRGB);
  let clippedYCoCg = clipAABB(aabbMin, aabbMax, histYCoCg, momM1);
  historyRGB       = ycocgToRGB(clippedYCoCg);

  // ── 5. Motion-adaptive blend factor
  let motionMag = length(motion);
  let alpha     = mix(u.blendFactorMin, u.blendFactorMax,
                      clamp(motionMag / u.motionScale, 0.0, 1.0));

  // ── 6. Blend in HDR weight space (Karis 2014) or direct
  var result: vec3f;
  if (u.enableHDR > 0.5) {
    // Tonemap before blend to suppress fireflies
    let currTM = reinhard(currentRGB);
    let histTM = reinhard(historyRGB);
    let wCurr  = hdrWeight(currentRGB);
    let wHist  = hdrWeight(historyRGB);
    // Weighted blend
    let blended = mix(histTM * wHist, currTM * wCurr, alpha)
                  / mix(wHist, wCurr, alpha);
    // Inverse tonemap back to HDR
    result = inverseReinhard(blended);
  } else {
    result = mix(historyRGB, currentRGB, alpha);
  }

  // ── 7. Reject history entirely for out-of-screen reprojections
  if (historyUV.x < 0.0 || historyUV.x > 1.0 ||
      historyUV.y < 0.0 || historyUV.y > 1.0) {
    result = currentRGB;
  }

  return vec4f(max(result, vec3f(0.0)), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Number of floats in the motion vector uniform buffer. */
const MOTION_UNIFORM_FLOATS = 4 * 4 * 3 + 4;  // 3 mat4x4 + 4 scalars (width,height,pad,pad)

/** Number of floats in the TAA resolve uniform buffer. */
const TAA_UNIFORM_FLOATS = 12;  // see TAAUniforms struct

/** Default jitter sequence length. */
const DEFAULT_JITTER_LENGTH = 16;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalReprojection class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Full TAA pipeline: motion vector generation → temporal resolve.
 *
 * Manages:
 * - Jitter table (Halton 2,3)
 * - Motion vector render pass (depth + viewProj reprojection)
 * - History ping-pong double buffer
 * - Neighbourhood variance clip in YCoCg
 * - HDR-weighted temporal blend (Karis 2014)
 *
 * @example
 * ```ts
 * const taa = await TemporalReprojection.create(device, format, 1920, 1080);
 *
 * // Each frame:
 * taa.setViewProjection(currVP, prevVP, currInvVP);
 * const jitter = taa.nextJitter(width, height);
 * // Apply jitter to your projection matrix before scene render …
 *
 * taa.renderMotionVectors(encoder, depthView);
 * taa.resolve(encoder, currentSceneView, dstView);
 * ```
 */
export class TemporalReprojection {

  // ── GPU resources ────────────────────────────────────────────────────────
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;

  // Motion vector pass
  private readonly motionPipeline : GPURenderPipeline;
  private readonly motionBGL      : GPUBindGroupLayout;
  private readonly motionUniformBuf: GPUBuffer;
  private motionTex    : GPUTexture;
  private motionView   : GPUTextureView;
  private motionBG     : GPUBindGroup | null = null;
  private cachedDepthSrc: GPUTextureView | null = null;

  // Resolve pass
  private readonly resolvePipeline : GPURenderPipeline;
  private readonly resolveBGL      : GPUBindGroupLayout;
  private readonly resolveUniformBuf: GPUBuffer;

  // History ping-pong
  private historyTexA  : GPUTexture;
  private historyTexB  : GPUTexture;
  private historyViewA : GPUTextureView;
  private historyViewB : GPUTextureView;
  private historyIndex = 0;   // 0 = read A / write B, 1 = read B / write A

  // Samplers
  private readonly samplerLinear  : GPUSampler;
  private readonly samplerNearest : GPUSampler;

  // ── CPU state ────────────────────────────────────────────────────────────
  private width  : number;
  private height : number;
  private params : Required<TAAParams>;
  private frameIndex = 0;
  private jitterTable: Float32Array;

  // Matrices (column-major Float32Array, 16 floats each)
  private currViewProj    = new Float32Array(16);
  private prevViewProj    = new Float32Array(16);
  private currInvViewProj = new Float32Array(16);

  // ── Constructor (private — use static `create`) ──────────────────────────

  private constructor(
    device          : GPUDevice,
    format          : GPUTextureFormat,
    motionPipeline  : GPURenderPipeline,
    motionBGL       : GPUBindGroupLayout,
    motionUniformBuf: GPUBuffer,
    resolvePipeline : GPURenderPipeline,
    resolveBGL      : GPUBindGroupLayout,
    resolveUniformBuf: GPUBuffer,
    samplerLinear   : GPUSampler,
    samplerNearest  : GPUSampler,
    width           : number,
    height          : number,
  ) {
    this.device           = device;
    this.format           = format;
    this.motionPipeline   = motionPipeline;
    this.motionBGL        = motionBGL;
    this.motionUniformBuf = motionUniformBuf;
    this.resolvePipeline  = resolvePipeline;
    this.resolveBGL       = resolveBGL;
    this.resolveUniformBuf = resolveUniformBuf;
    this.samplerLinear    = samplerLinear;
    this.samplerNearest   = samplerNearest;
    this.width            = width;
    this.height           = height;

    this.params = {
      blendFactorMin       : 0.04,
      blendFactorMax       : 0.20,
      motionScale          : 0.02,
      jitterSequenceLength : DEFAULT_JITTER_LENGTH,
      varianceClipGamma    : 1.0,
      enableJitter         : true,
      enableHDRWeight      : true,
    };

    this.jitterTable = buildJitterTable(this.params.jitterSequenceLength);

    // Create textures
    const texDesc = this._texDesc(width, height);
    this.motionTex   = device.createTexture({ ...texDesc, format: 'rg16float',   label: 'TAA motionVec' });
    this.motionView  = this.motionTex.createView();
    this.historyTexA = device.createTexture({ ...texDesc, format: 'rgba16float', label: 'TAA historyA' });
    this.historyTexB = device.createTexture({ ...texDesc, format: 'rgba16float', label: 'TAA historyB' });
    this.historyViewA = this.historyTexA.createView();
    this.historyViewB = this.historyTexB.createView();

    // Identity matrices as initial state
    this.currViewProj[0] = this.currViewProj[5] = this.currViewProj[10] = this.currViewProj[15] = 1;
    this.prevViewProj[0] = this.prevViewProj[5] = this.prevViewProj[10] = this.prevViewProj[15] = 1;
    this.currInvViewProj[0] = this.currInvViewProj[5] = this.currInvViewProj[10] = this.currInvViewProj[15] = 1;
  }

  // ── Factory ──────────────────────────────────────────────────────────────

  /**
   * Create a ready-to-use TAA pipeline.
   *
   * @param device - GPUDevice
   * @param format - Swap-chain / output texture format (e.g. 'bgra8unorm')
   * @param width  - Render target width in pixels
   * @param height - Render target height in pixels
   * @param params - Optional initial TAA parameters
   */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
    params?: TAAParams,
  ): Promise<TemporalReprojection> {

    // ── Motion vector pipeline ──────────────────────────────────────────
    const motionModule = device.createShaderModule({
      code  : MOTION_VEC_WGSL,
      label : 'TAA motionVec shader',
    });

    const motionBGL = device.createBindGroupLayout({
      label: 'TAA motionVec BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'non-filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    const motionPipeline = await device.createRenderPipelineAsync({
      label    : 'TAA motionVec pipeline',
      layout   : device.createPipelineLayout({ bindGroupLayouts: [motionBGL] }),
      vertex   : { module: motionModule, entryPoint: 'vs_motion' },
      fragment : {
        module     : motionModule,
        entryPoint : 'fs_motion',
        targets    : [{ format: 'rg16float' as GPUTextureFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const motionUniformBuf = device.createBuffer({
      label : 'TAA motionVec uniforms',
      size  : MOTION_UNIFORM_FLOATS * 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── TAA resolve pipeline ────────────────────────────────────────────
    const resolveModule = device.createShaderModule({
      code  : TAA_RESOLVE_WGSL,
      label : 'TAA resolve shader',
    });

    const resolveBGL = device.createBindGroupLayout({
      label: 'TAA resolve BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'non-filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },
      ],
    });

    const resolvePipeline = await device.createRenderPipelineAsync({
      label    : 'TAA resolve pipeline',
      layout   : device.createPipelineLayout({ bindGroupLayouts: [resolveBGL] }),
      vertex   : { module: resolveModule, entryPoint: 'vs_taa' },
      fragment : {
        module     : resolveModule,
        entryPoint : 'fs_taa',
        targets    : [{ format: 'rgba16float' as GPUTextureFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const resolveUniformBuf = device.createBuffer({
      label : 'TAA resolve uniforms',
      size  : TAA_UNIFORM_FLOATS * 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Samplers ────────────────────────────────────────────────────────
    const samplerLinear = device.createSampler({
      label        : 'TAA linear sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    const samplerNearest = device.createSampler({
      label        : 'TAA nearest sampler',
      magFilter    : 'nearest',
      minFilter    : 'nearest',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    const instance = new TemporalReprojection(
      device, format,
      motionPipeline, motionBGL, motionUniformBuf,
      resolvePipeline, resolveBGL, resolveUniformBuf,
      samplerLinear, samplerNearest,
      width, height,
    );

    if (params) instance.setParams(params);

    return instance;
  }

  // ── Configuration ────────────────────────────────────────────────────────

  /** Update TAA parameters (partial overrides). */
  setParams(p: TAAParams): this {
    this.params = { ...this.params, ...p };
    // Rebuild jitter table if sequence length changed
    if (p.jitterSequenceLength !== undefined) {
      this.jitterTable = buildJitterTable(this.params.jitterSequenceLength);
      this.frameIndex = 0;
    }
    return this;
  }

  /** Get current params (read-only snapshot). */
  getParams(): Readonly<Required<TAAParams>> {
    return { ...this.params };
  }

  /**
   * Provide current and previous frame view-projection matrices.
   * Must be called every frame before `renderMotionVectors`.
   *
   * All matrices are column-major Float32Array (length 16), matching
   * the WebGPU/wgsl mat4x4f memory layout.
   */
  setViewProjection(
    currViewProj    : Float32Array,
    prevViewProj    : Float32Array,
    currInvViewProj : Float32Array,
  ): this {
    this.currViewProj.set(currViewProj);
    this.prevViewProj.set(prevViewProj);
    this.currInvViewProj.set(currInvViewProj);
    return this;
  }

  // ── Jitter ───────────────────────────────────────────────────────────────

  /**
   * Advance the Halton jitter sequence and return the sub-pixel offset
   * that should be applied to the projection matrix before scene rendering.
   *
   * The returned values are in **pixel units** — divide by viewport size
   * and multiply by 2 (NDC range) to get the projection-matrix offset:
   *
   * ```ts
   * const [jx, jy] = taa.nextJitter(width, height);
   * proj[2][0] += jx * 2 / width;
   * proj[2][1] += jy * 2 / height;
   * ```
   *
   * @returns [jitterX, jitterY] in pixels, centred around 0.
   */
  nextJitter(viewportW: number, viewportH: number): [number, number] {
    if (!this.params.enableJitter) {
      this.frameIndex++;
      return [0, 0];
    }
    const idx = this.frameIndex % this.params.jitterSequenceLength;
    const jx  = this.jitterTable[idx * 2];
    const jy  = this.jitterTable[idx * 2 + 1];
    this.frameIndex++;
    return [jx, jy];
  }

  /**
   * Get the current frame's jitter offset (same as last `nextJitter` result)
   * without advancing the sequence. Useful for passing to the resolve shader.
   */
  currentJitter(): [number, number] {
    const idx = ((this.frameIndex - 1 + this.params.jitterSequenceLength)
                 % this.params.jitterSequenceLength);
    if (!this.params.enableJitter) return [0, 0];
    return [this.jitterTable[idx * 2], this.jitterTable[idx * 2 + 1]];
  }

  // ── Render: Motion Vectors ───────────────────────────────────────────────

  /**
   * Record the motion-vector generation pass.
   *
   * Reads the depth buffer from the current scene render and the view-projection
   * matrices to produce a per-pixel screen-space velocity texture.
   *
   * @param encoder  Active GPUCommandEncoder
   * @param depthView Scene depth buffer texture view (r32float or depth24plus etc.)
   */
  renderMotionVectors(
    encoder  : GPUCommandEncoder,
    depthView: GPUTextureView,
  ): void {
    this._uploadMotionUniforms();
    const bg = this._motionBindGroup(depthView);

    const pass = encoder.beginRenderPass({
      label: 'TAA motionVec pass',
      colorAttachments: [{
        view      : this.motionView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.motionPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(6);
    pass.end();
  }

  /**
   * Inject an externally-generated motion vector texture instead of running
   * the built-in motion vector pass. Useful when the scene pipeline already
   * produces a velocity buffer.
   *
   * @param motionView - A GPUTextureView with RG motion vectors in NDC.
   */
  setExternalMotionVectors(motionView: GPUTextureView): void {
    this.motionView = motionView;
  }

  // ── Render: Temporal Resolve ─────────────────────────────────────────────

  /**
   * Record the temporal resolve pass: read the current jittered frame +
   * history buffer + motion vectors, produce the temporally accumulated
   * result.
   *
   * The result is written to **two** targets simultaneously:
   *   1. `dstView`  — the resolved output for downstream post-processing
   *   2. Internal history buffer (ping-pong) — carried to next frame
   *
   * Since WebGPU doesn't allow the same texture as both input and output,
   * we write to the *inactive* history slice and flip at the end.
   *
   * @param encoder      Active GPUCommandEncoder
   * @param currentView  Current frame's colour (jittered scene render)
   * @param dstView      Output texture view (next pass in post-process chain)
   */
  resolve(
    encoder     : GPUCommandEncoder,
    currentView : GPUTextureView,
    dstView     : GPUTextureView,
  ): void {
    this._uploadResolveUniforms();

    // Read from current history, write to the other
    const [readHistory, writeHistory] = this.historyIndex === 0
      ? [this.historyViewA, this.historyViewB]
      : [this.historyViewB, this.historyViewA];

    const bg = this._resolveBindGroup(currentView, readHistory);

    // Pass 1: render to the writeHistory (internal accumulation)
    const passHistory = encoder.beginRenderPass({
      label: 'TAA resolve → history',
      colorAttachments: [{
        view      : writeHistory,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    passHistory.setPipeline(this.resolvePipeline);
    passHistory.setBindGroup(0, bg);
    passHistory.draw(6);
    passHistory.end();

    // Pass 2: copy the resolved history to the user's dstView
    // We render the same shader again but to dstView. This is cheaper
    // than a texture copy because we avoid a full-res blit barrier.
    const passDst = encoder.beginRenderPass({
      label: 'TAA resolve → output',
      colorAttachments: [{
        view      : dstView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    passDst.setPipeline(this.resolvePipeline);
    passDst.setBindGroup(0, bg);
    passDst.draw(6);
    passDst.end();

    // Flip ping-pong index
    this.historyIndex = 1 - this.historyIndex;
  }

  // ── Resize ───────────────────────────────────────────────────────────────

  /**
   * Recreate internal textures after a viewport resize.
   * Must be called before the next frame's render calls.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;

    // Destroy old textures
    this.motionTex.destroy();
    this.historyTexA.destroy();
    this.historyTexB.destroy();

    // Recreate
    const texDesc = this._texDesc(width, height);
    this.motionTex   = this.device.createTexture({ ...texDesc, format: 'rg16float',   label: 'TAA motionVec' });
    this.motionView  = this.motionTex.createView();
    this.historyTexA = this.device.createTexture({ ...texDesc, format: 'rgba16float', label: 'TAA historyA' });
    this.historyTexB = this.device.createTexture({ ...texDesc, format: 'rgba16float', label: 'TAA historyB' });
    this.historyViewA = this.historyTexA.createView();
    this.historyViewB = this.historyTexB.createView();

    // Invalidate cached bind groups
    this.motionBG     = null;
    this.cachedDepthSrc = null;
    this.historyIndex = 0;
  }

  // ── Cleanup ──────────────────────────────────────────────────────────────

  /** Release all GPU resources owned by this pipeline. */
  destroy(): void {
    this.motionUniformBuf.destroy();
    this.resolveUniformBuf.destroy();
    this.motionTex.destroy();
    this.historyTexA.destroy();
    this.historyTexB.destroy();
  }

  // ── Accessors (for external integration) ─────────────────────────────────

  /** Get the motion vector texture view (for debug visualisation etc.). */
  getMotionVectorView(): GPUTextureView {
    return this.motionView;
  }

  /** Get the current history buffer view (read-only, for debug). */
  getHistoryView(): GPUTextureView {
    return this.historyIndex === 0 ? this.historyViewA : this.historyViewB;
  }

  /** Current frame index (monotonically increasing). */
  getFrameIndex(): number {
    return this.frameIndex;
  }

  // ── Internal helpers ─────────────────────────────────────────────────────

  private _texDesc(w: number, h: number): GPUTextureDescriptor {
    return {
      size   : [w, h],
      format : 'rgba16float',  // overridden per-texture
      usage  : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    };
  }

  private _uploadMotionUniforms(): void {
    const data = new Float32Array(MOTION_UNIFORM_FLOATS);
    // mat4x4 current viewProj  (16 floats, offset 0)
    data.set(this.currViewProj, 0);
    // mat4x4 previous viewProj (16 floats, offset 16)
    data.set(this.prevViewProj, 16);
    // mat4x4 inverse current   (16 floats, offset 32)
    data.set(this.currInvViewProj, 32);
    // resolution + padding     (4 floats, offset 48)
    data[48] = this.width;
    data[49] = this.height;
    data[50] = 0;  // _pad0
    data[51] = 0;  // _pad1
    this.device.queue.writeBuffer(this.motionUniformBuf, 0, data);
  }

  private _uploadResolveUniforms(): void {
    const p = this.params;
    const [jx, jy] = this.currentJitter();

    const data = new Float32Array(TAA_UNIFORM_FLOATS);
    data[0]  = this.width;
    data[1]  = this.height;
    data[2]  = p.blendFactorMin;
    data[3]  = p.blendFactorMax;
    data[4]  = p.motionScale;
    data[5]  = p.varianceClipGamma;
    data[6]  = p.enableHDRWeight ? 1.0 : 0.0;
    data[7]  = jx;
    data[8]  = jy;
    data[9]  = 0;  // _pad0
    data[10] = 0;  // _pad1
    data[11] = 0;  // _pad2
    this.device.queue.writeBuffer(this.resolveUniformBuf, 0, data);
  }

  private _motionBindGroup(depthView: GPUTextureView): GPUBindGroup {
    if (this.motionBG && this.cachedDepthSrc === depthView) {
      return this.motionBG;
    }
    this.motionBG = this.device.createBindGroup({
      layout  : this.motionBGL,
      label   : 'TAA motionVec BG',
      entries : [
        { binding: 0, resource: { buffer: this.motionUniformBuf } },
        { binding: 1, resource: this.samplerNearest },
        { binding: 2, resource: depthView },
      ],
    });
    this.cachedDepthSrc = depthView;
    return this.motionBG;
  }

  private _resolveBindGroup(
    currentView : GPUTextureView,
    historyView : GPUTextureView,
  ): GPUBindGroup {
    // Always recreate — currentView changes every frame
    return this.device.createBindGroup({
      layout  : this.resolveBGL,
      label   : 'TAA resolve BG',
      entries : [
        { binding: 0, resource: { buffer: this.resolveUniformBuf } },
        { binding: 1, resource: this.samplerLinear },
        { binding: 2, resource: this.samplerNearest },
        { binding: 3, resource: currentView },
        { binding: 4, resource: historyView },
        { binding: 5, resource: this.motionView },
      ],
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection matrix jitter helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Apply a sub-pixel jitter offset to a column-major 4×4 projection matrix.
 *
 * Modifies `proj` in-place by adding the jitter to the translation
 * components of the projection (elements [8] and [9] in column-major
 * layout, i.e. row 0/1 of column 2).
 *
 * @param proj     Column-major Float32Array(16) projection matrix
 * @param jitterX  Sub-pixel offset in pixels (from `nextJitter`)
 * @param jitterY  Sub-pixel offset in pixels (from `nextJitter`)
 * @param width    Viewport width in pixels
 * @param height   Viewport height in pixels
 * @returns        The same `proj` array, modified in-place
 *
 * @example
 * ```ts
 * const [jx, jy] = taa.nextJitter(w, h);
 * applyJitterToProjection(projMatrix, jx, jy, w, h);
 * // … render scene with jittered projection …
 * ```
 */
export function applyJitterToProjection(
  proj    : Float32Array,
  jitterX : number,
  jitterY : number,
  width   : number,
  height  : number,
): Float32Array {
  // Column-major: element [8] = row 0 col 2, element [9] = row 1 col 2
  proj[8] += (2.0 * jitterX) / width;
  proj[9] += (2.0 * jitterY) / height;
  return proj;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset configurations
// ─────────────────────────────────────────────────────────────────────────────

/** Pre-configured TAA quality presets. */
export const TAAPresets = {

  /**
   * 高质量：最大 temporal 累积，适合静态或缓慢移动的场景。
   * 提供最平滑的结果但快速运动时可能出现轻微残影。
   */
  high(taa: TemporalReprojection): TemporalReprojection {
    return taa.setParams({
      blendFactorMin       : 0.02,
      blendFactorMax       : 0.12,
      motionScale          : 0.015,
      jitterSequenceLength : 16,
      varianceClipGamma    : 1.25,
      enableHDRWeight      : true,
    });
  },

  /**
   * 平衡模式：适度累积 + 较紧的 variance clip。
   * 适合大多数 cell 场景中粒子和相机都在移动的情况。
   */
  balanced(taa: TemporalReprojection): TemporalReprojection {
    return taa.setParams({
      blendFactorMin       : 0.04,
      blendFactorMax       : 0.20,
      motionScale          : 0.02,
      jitterSequenceLength : 8,
      varianceClipGamma    : 1.0,
      enableHDRWeight      : true,
    });
  },

  /**
   * 低延迟：偏向当前帧，minimal ghosting。
   * 适合快速交互（拖拽 cell、缩放）或 VR 场景。
   */
  responsive(taa: TemporalReprojection): TemporalReprojection {
    return taa.setParams({
      blendFactorMin       : 0.10,
      blendFactorMax       : 0.35,
      motionScale          : 0.03,
      jitterSequenceLength : 4,
      varianceClipGamma    : 0.75,
      enableHDRWeight      : true,
    });
  },

  /**
   * 直通模式：禁用 jitter 和 temporal 累积（debug 用）。
   * blendFactorMin/Max 都设 1.0 → 100% 使用当前帧。
   */
  passthrough(taa: TemporalReprojection): TemporalReprojection {
    return taa.setParams({
      blendFactorMin       : 1.0,
      blendFactorMax       : 1.0,
      motionScale          : 1.0,
      jitterSequenceLength : 1,
      varianceClipGamma    : 1.0,
      enableJitter         : false,
      enableHDRWeight      : false,
    });
  },
} as const;
