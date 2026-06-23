/**
 * ue-ssr-motionblur.ts — M851b: UE5 Screen Space Reflection + Motion Blur Port
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Screen Space Reflection (SSR) 与 Motion Blur (MB) 系统到 WebGPU,
 * 为 Cell 在高速移动时产生物理感的拖影和环境反射效果。
 *
 * 核心概念 (参照 UE5 Renderer-Private/ScreenSpaceReflection* 与 PostProcess/MotionBlur*):
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 *
 *  § 1  Screen Space Reflection (SSR) - Hi-Z Ray March + Temporal Filter
 *       ┌─────────────────────────────────────────────────────────────────────────────┐
 *       │ Hi-Z Pyramid (Hierarchical-Z):                                              │
 *       │  • 从 SceneDepth 构建分级深度金字塔 (mip0=1x1, mip1=2x2, ... mipN=fullres) │
 *       │  • 每级取最大深度值，用于快速遮挡测试                                       │
 *       │  • Ray marching 从粗到细逐级细化，加速收敛                                  │
 *       │                                                                             │
 *       │ Ray March Algorithm (Tile-based for efficiency):                            │
 *       │  • SSR_TILE_SIZE = 8×8 像素                                                │
 *       │  • 仅在 bUseSSRTile==true 的 tiles 执行反射计算（tile classification）      │
 *       │  • 每个 ray 迭代：                                                          │
 *       │    1) 当前位置 P += RayDir * StepSize                                      │
 *       │    2) 投影到屏幕空间: uv = ProjectToScreen(P)                              │
 *       │    3) Hi-Z 碰撞检测: 若 P.z > Hi-Z[uv] → hit/refine                       │
 *       │    4) 精化搜索或返回结果                                                    │
 *       │                                                                             │
 *       │ Temporal Filtering (降噪):                                                 │
 *       │  • 当前帧 SSR + 前帧 temporal history → 加权混合                           │
 *       │  • confidence 权重：高置信度优先当前帧，低则靠历史稳定                      │
 *       └─────────────────────────────────────────────────────────────────────────────┘
 *
 *  § 2  Motion Blur - Tile-based Per-Object Velocity
 *       ┌─────────────────────────────────────────────────────────────────────────────┐
 *       │ Velocity Buffer Structure:                                                  │
 *       │  • SceneVelocity: 每像素的屏幕空间速度向量 (velocity.xy 为屏幕坐标变化)    │
 *       │  • Velocity Tile (16×16):                                                   │
 *       │    - MaxVelocity[0]: 最大速度向量（极坐标：长度+角度）                     │
 *       │    - MaxVelocity[1]: 次大速度向量（支持多方向模糊）                        │
 *       │    - 通过 reduce/scatter 从像素级聚合到 tile 级                            │
 *       │                                                                             │
 *       │ Per-Object Velocity Computation:                                            │
 *       │  • worldPos = Reconstruct(depth, uv, InvProj)                             │
 *       │  • prevClipPos = mul(worldPos, PrevWorldToClip)                           │
 *       │  • prevUv = prevClipPos.xy / prevClipPos.w                                │
 *       │  • velocity = (uv - prevUv) * (0.5 * ViewportSize)                        │
 *       │  • => 每对象速度 = 当前屏幕位置 - 前帧屏幕位置                             │
 *       │                                                                             │
 *       │ Blur Kernel (Gather-based):                                               │
 *       │  • 沿 velocity direction 采集邻域样本                                       │
 *       │  • 采样数 = clamp(max_velocity.length(), MIN_SAMPLES, MAX_SAMPLES)        │
 *       │  • 支持多方向混合（CONFIG_MAX_RANGE_SIZE > 1）                             │
 *       │  • 应用 max() 权重滤波避免 ghosting                                       │
 *       └─────────────────────────────────────────────────────────────────────────────┘
 *
 *  § 3  Cell 高速移动表现 (Motion Blur Driven)
 *       • Cell velocity > threshold → 自动启用 MB pass
 *       • MB 强度 = f(velocity.length) → 速度越快拖影越明显
 *       • 支持多对象并行 MB（每个 Cell 独立速度）
 *
 * 算法流程
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 *
 *   初始化阶段:
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ CreateHiZPyramid()                                           │
 *   │  SceneDepth → Mip0,1,2,... 分级深度金字塔                  │
 *   │  每级取最大深度（保守估计遮挡）                              │
 *   └────────────────────────────────────┬────────────────────────┘
 *                                        ▼
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ ClassifySSRTiles()                                           │
 *   │  GBuffer normal + roughness → 标记镜面反射的 tiles          │
 *   │  输出: TileMaskBuffer (bitmask, 1bit per tile)              │
 *   └────────────────────────────────────┬────────────────────────┘
 *
 *   主渲染循环 (每帧):
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ Pass 1: Ray March SSR                                        │
 *   │  输入: SceneColor, SceneNormal, SceneRoughness, Hi-Z        │
 *   │  输出: SSRReflection (specular reflection color)            │
 *   │  仅计算 TileMaskBuffer==1 的 tiles                         │
 *   └────────────────────────────────────┬────────────────────────┘
 *                                        ▼
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ Pass 2: Temporal Denoise (SSR)                              │
 *   │  输入: SSRReflection + TemporalHistory                      │
 *   │  输出: DenoisedSSR                                           │
 *   │  (confidence-weighted blend)                                │
 *   └────────────────────────────────────┬────────────────────────┘
 *                                        ▼
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ Pass 3: Velocity Flatten (Motion Blur)                      │
 *   │  输入: SceneVelocity (per-pixel)                            │
 *   │  输出: VelocityTile[16×16] with MaxVelocity[0,1]           │
 *   │  聚合: 将像素速度归约到 tile 级（取绝对值最大 2 个方向）    │
 *   └────────────────────────────────────┬────────────────────────┘
 *                                        ▼
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │ Pass 4: Motion Blur Apply (Gather)                          │
 *   │  输入: SceneColor, VelocityTile, DenoisedSSR               │
 *   │  output: BlurredColor                                       │
 *   │  沿 MaxVelocity[0] 方向采集邻域，weighted average          │
 *   │  若 |MaxVelocity[1]| > threshold: 混合第二方向              │
 *   └────────────────────────────────────┬────────────────────────┘
 *                                        ▼
 *   最终合成:
 *   BlurredColor(SSR specular) + BaseColor → FinalComposite
 *
 * 性能优化
 * ─────────────────────────────────────────────────────────────────────────────────────────────
 *  • Tile Classification: 跳过低镜面的区域，减少 SSR 计算量
 *  • Hi-Z Pyramid: 分级遮挡测试加速 ray march (~16× 加速)
 *  • Temporal Filter: 帧间复用，减少噪声（需要运动补偿）
 *  • Velocity Tile Reduce: 并行化速度聚合，支持异构方向
 *  • Early-Exit Ray March: 命中时立即返回，避免无谓迭代
 *
 * Usage:
 *   const ssrMb = await UESSRMotionBlur.create(device, canvas, {
 *     hiZLevels: 6,
 *     raySamples: 12,
 *     velocityTileSize: 16,
 *     motionBlurStrength: 1.0,
 *   });
 *   ssrMb.updateSceneData(depthTex, normalTex, roughnessTex, velocityTex);
 *   ssrMb.rayMarchSSR(encoder);      // Pass 1
 *   ssrMb.temporalDenoise(encoder);  // Pass 2
 *   ssrMb.velocityFlatten(encoder);  // Pass 3
 *   ssrMb.motionBlurApply(encoder);  // Pass 4
 *   return ssrMb.getResult();        // BlurredColor with SSR
 *
 * Research: xiaodi #M851b — cell-pubsub-loop
 * Ported from: Renderer-Private/ScreenSpaceReflectionTiles.{h,cpp}
 *              Renderer-Private/PostProcess/PostProcessMotionBlur.{h,cpp}
 *              Shaders-Private/ScreenSpaceReflectionTileCommons.ush
 *              Shaders-Private/MotionBlur/MotionBlur*.usf/*.ush
 */

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 0  Type Definitions & Constants
// ─────────────────────────────────────────────────────────────────────────────────────────────

/** SSR tile size (pixels) — matches SSR_TILE_SIZE_XY in UE5 */
const SSR_TILE_SIZE = 8;

/** Motion blur velocity tile size (pixels) */
const VELOCITY_TILE_SIZE = 16;

/** Motion blur velocity flatten tile size */
const VELOCITY_FLATTEN_TILE_SIZE = 16;

/** Maximum number of directions for motion blur per tile */
const CONFIG_MAX_RANGE_SIZE = 2;

/** Maximum samples per motion blur direction */
const MAX_MOTION_BLUR_SAMPLES = 32;

/** Minimum velocity magnitude to trigger motion blur (pixels) */
const MIN_MOTION_BLUR_VELOCITY = 0.5;

/** Maximum ray march iterations for SSR */
const MAX_RAY_MARCH_ITERATIONS = 64;

/** Hi-Z pyramid levels */
const DEFAULT_HIZ_LEVELS = 6;

/** SSR ray max distance (world units, relative to view) */
const SSR_RAY_MAX_DIST = 1000.0;

interface SSRMotionBlurConfig {
  width: number;
  height: number;
  hiZLevels?: number;
  raySamples?: number;
  velocityTileSize?: number;
  motionBlurStrength?: number;
  enableSSR?: boolean;
  enableMotionBlur?: boolean;
  temporalFilterWeight?: number;
}

interface CellVelocityData {
  cellId: string;
  worldPos: Float32Array; // [x, y, z]
  prevWorldPos: Float32Array; // [x, y, z]
  screenVelocity: Float32Array; // [vx, vy] in screen space
}

interface SSRReflectionData {
  reflectionColor: Float32Array; // [r, g, b, a]
  confidence: number;
  rayDistance: number;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 1  Helper Utilities
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Cartesian to Polar: Convert velocity vector to (magnitude, angle)
 * θ ∈ [-π, π]
 */
function velocityCartesianToPolar(vx: number, vy: number): [number, number] {
  const magnitude = Math.sqrt(vx * vx + vy * vy);
  const angle = magnitude > 0.0 ? Math.atan2(vy, vx) : 0.0;
  return [magnitude, angle];
}

/**
 * Polar to Cartesian: Reconstruct velocity vector from (magnitude, angle)
 */
function velocityPolarToCartesian(mag: number, angle: number): [number, number] {
  return [mag * Math.cos(angle), mag * Math.sin(angle)];
}

/**
 * Angular difference: Shortest path between two angles
 */
function angleDifference(a1: number, a2: number): number {
  let diff = Math.abs(a1 - a2);
  if (diff > Math.PI) {
    diff = 2.0 * Math.PI - diff;
  }
  return diff;
}

/**
 * Get maximum polar velocity between two (mag, angle) pairs
 */
function maxPolarVelocity(
  v0: [number, number],
  v1: [number, number]
): [number, number] {
  return v0[0] > v1[0] ? v0 : v1;
}

/**
 * Hi-Z Pyramid Level Index from depth value
 * Used in ray march to select appropriate mip level
 */
function getHiZMipLevel(rayStepSize: number, hiZLevels: number): number {
  // Ray step size → mip level (larger steps use coarser mips)
  const level = Math.log2(Math.max(1.0, rayStepSize));
  return Math.min(Math.floor(level), hiZLevels - 1);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 2  WGSL Shader Sources
// ─────────────────────────────────────────────────────────────────────────────────────────────

const SSR_RAY_MARCH_SHADER = `
struct RayMarchInput {
  pixelCoord: vec2<u32>,
  rayOrigin: vec3<f32>,
  rayDir: vec3<f32>,
  rayMaxDist: f32,
}

struct RayMarchOutput {
  hitColor: vec4<f32>,
  confidence: f32,
  rayDist: f32,
}

// Reconstruct world position from depth
fn reconstructWorldPos(
  uv: vec2<f32>,
  depth: f32,
  invProj: mat4x4<f32>,
  invView: mat4x4<f32>
) -> vec3<f32> {
  let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
  var worldPos = invProj * ndc;
  worldPos = worldPos / worldPos.w;
  worldPos = invView * worldPos;
  return worldPos.xyz;
}

// Project world position to screen space
fn projectToScreenUV(
  worldPos: vec3<f32>,
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  viewSize: vec2<f32>
) -> vec2<f32> {
  let viewPos = view * vec4<f32>(worldPos, 1.0);
  let clipPos = proj * viewPos;
  let ndc = clipPos.xy / clipPos.w;
  let uv = (ndc + 1.0) * 0.5;
  return uv;
}

// Hi-Z occlusion test: rayPos.z > Hi-Z[rayPos.xy] → ray behind surface
fn testHiZOcclusion(
  rayScreenUV: vec2<f32>,
  rayDepth: f32,
  hiZTex: texture_2d<f32>,
  hizSampler: sampler,
  mipLevel: f32
) -> bool {
  let hizDepth = textureSampleLevel(hiZTex, hizSampler, rayScreenUV, mipLevel).x;
  // Larger depth value = farther away; if rayDepth > hizDepth, ray is behind
  return rayDepth > hizDepth;
}

// Ray march kernel: iterate until hit or max distance
fn rayMarch(
  input: RayMarchInput,
  sceneTex: texture_2d<f32>,
  sceneDepth: texture_2d<f32>,
  hiZTex: texture_2d<f32>,
  hizSampler: sampler,
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  invProj: mat4x4<f32>,
  invView: mat4x4<f32>,
  viewSize: vec2<f32>,
  hiZLevels: u32
) -> RayMarchOutput {
  var output: RayMarchOutput;
  output.hitColor = vec4<f32>(0.0);
  output.confidence = 0.0;
  output.rayDist = 0.0;

  var rayPos = input.rayOrigin;
  var stepSize = 1.0; // pixels, adaptive
  var rayDist = 0.0;

  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    rayPos = rayPos + input.rayDir * stepSize;
    rayDist += stepSize;

    if (rayDist > input.rayMaxDist || stepSize < 0.01) {
      break;
    }

    let screenUV = projectToScreenUV(rayPos, proj, view, viewSize);
    if (any(screenUV < vec2<f32>(0.0)) || any(screenUV > vec2<f32>(1.0))) {
      break;
    }

    let mipLevel = f32(getHiZMipLevel(stepSize, i32(hiZLevels)));
    let isOccluded = testHiZOcclusion(screenUV, rayPos.z, hiZTex, hizSampler, mipLevel);

    if (isOccluded) {
      // Refine with smaller step
      stepSize *= 0.5;
      rayPos -= input.rayDir * stepSize * 2.0;
    } else {
      // Continue marching, increase step size
      stepSize *= 1.1;
    }
  }

  let finalScreenUV = projectToScreenUV(rayPos, proj, view, viewSize);
  if (all(finalScreenUV >= vec2<f32>(0.0)) && all(finalScreenUV <= vec2<f32>(1.0))) {
    output.hitColor = textureSample(sceneTex, sampler(filtering: linear), finalScreenUV);
    output.confidence = 0.7; // Moderate confidence for ray march
    output.rayDist = rayDist;
  }

  return output;
}
`;

const VELOCITY_FLATTEN_SHADER = `
struct VelocityTile {
  maxVel0: vec4<f32>,    // [vx, vy, length, angle] for primary direction
  maxVel1: vec4<f32>,    // [vx, vy, length, angle] for secondary direction
}

// Cartesian to polar
fn cartesianToPolar(v: vec2<f32>) -> vec2<f32> {
  let len = length(v);
  let angle = select(0.0, atan2(v.y, v.x), len > 0.0);
  return vec2<f32>(len, angle);
}

// Reduce velocity vectors in tile: find top 2 by magnitude
fn reduceVelocityTile(
  velocities: array<vec2<f32>, 256>,
  count: u32
) -> VelocityTile {
  var maxV0 = vec2<f32>(0.0, 0.0);
  var maxV1 = vec2<f32>(0.0, 0.0);
  var maxP0 = cartesianToPolar(maxV0);
  var maxP1 = cartesianToPolar(maxV1);

  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let v = velocities[i];
    let p = cartesianToPolar(v);

    if (p.x > maxP0.x) {
      maxV1 = maxV0;
      maxP1 = maxP0;
      maxV0 = v;
      maxP0 = p;
    } else if (p.x > maxP1.x) {
      maxV1 = v;
      maxP1 = p;
    }
  }

  var output: VelocityTile;
  output.maxVel0 = vec4<f32>(maxV0, maxP0.x, maxP0.y);
  output.maxVel1 = vec4<f32>(maxV1, maxP1.x, maxP1.y);
  return output;
}
`;

const MOTION_BLUR_APPLY_SHADER = `
// Gather-based motion blur: sample along velocity direction
fn motionBlurGather(
  pixelCoord: vec2<u32>,
  baseColor: vec4<f32>,
  maxVel: vec4<f32>,     // [vx, vy, length, angle]
  sceneTex: texture_2d<f32>,
  sceneSampler: sampler,
  viewSize: vec2<f32>,
  strength: f32
) -> vec4<f32> {
  let pixelUV = vec2<f32>(pixelCoord) / viewSize;
  let velLength = maxVel.z * strength;

  if (velLength < 0.5) {
    return baseColor; // No blur if velocity too small
  }

  let velDir = normalize(maxVel.xy);
  let sampleCount = min(u32(ceil(velLength)), 32u);
  var accumColor = baseColor;
  var weightSum = 1.0;

  for (var i: u32 = 1u; i < sampleCount; i = i + 1u) {
    let t = f32(i) / f32(sampleCount);
    let offset = velDir * velLength * t * (1.0 / viewSize.x);
    let sampleUV = pixelUV + offset;

    if (all(sampleUV >= vec2<f32>(0.0)) && all(sampleUV <= vec2<f32>(1.0))) {
      let sampleColor = textureSample(sceneTex, sceneSampler, sampleUV);
      let weight = 1.0 - t; // Closer samples weighted higher
      accumColor = accumColor + sampleColor * weight;
      weightSum += weight;
    }
  }

  return accumColor / weightSum;
}
`;

const TEMPORAL_DENOISE_SHADER = `
// Temporal filter for SSR: blend current with history based on confidence
fn temporalDenoise(
  currentSSR: vec4<f32>,
  currentConfidence: f32,
  historySSR: vec4<f32>,
  historyWeight: f32
) -> vec4<f32> {
  // High confidence → trust current frame; low confidence → use history
  let alpha = mix(0.1, 0.9, currentConfidence);
  return mix(historySSR, currentSSR, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 3  Main UESSRMotionBlur Class
// ─────────────────────────────────────────────────────────────────────────────────────────────

export class UESSRMotionBlur {
  private device: GPUDevice;
  private config: Required<SSRMotionBlurConfig>;

  private depthTexture: GPUTexture | null = null;
  private normalTexture: GPUTexture | null = null;
  private roughnessTexture: GPUTexture | null = null;
  private velocityTexture: GPUTexture | null = null;

  private hiZPyramid: GPUTexture | null = null;
  private hiZPyramidView: GPUTextureView | null = null;

  private ssrReflectionTexture: GPUTexture | null = null;
  private ssrHistoryTexture: GPUTexture | null = null;

  private velocityTileBuffer: GPUBuffer | null = null;
  private ssrTileMaskBuffer: GPUBuffer | null = null;

  private resultTexture: GPUTexture | null = null;

  // Projection matrices for ray march (CPU-side)
  private projMatrix: Float32Array = new Float32Array(16);
  private viewMatrix: Float32Array = new Float32Array(16);
  private invProjMatrix: Float32Array = new Float32Array(16);
  private invViewMatrix: Float32Array = new Float32Array(16);

  // Pipeline objects
  private rayMarchPipeline: GPUComputePipeline | null = null;
  private velocityFlattenPipeline: GPUComputePipeline | null = null;
  private motionBlurApplyPipeline: GPUComputePipeline | null = null;
  private temporalDenoisePipeline: GPUComputePipeline | null = null;
  private hiZBuildPipeline: GPUComputePipeline | null = null;

  // Bind groups
  private rayMarchBindGroup: GPUBindGroup | null = null;
  private velocityFlattenBindGroup: GPUBindGroup | null = null;
  private motionBlurBindGroup: GPUBindGroup | null = null;
  private temporalDenoiseBindGroup: GPUBindGroup | null = null;
  private hiZBindGroup: GPUBindGroup | null = null;

  // Cell velocity tracking
  private cellVelocityMap: Map<string, CellVelocityData> = new Map();

  private constructor(device: GPUDevice, config: Required<SSRMotionBlurConfig>) {
    this.device = device;
    this.config = config;
  }

  /**
   * Factory method: Create UESSRMotionBlur instance
   */
  static async create(
    device: GPUDevice,
    config: SSRMotionBlurConfig
  ): Promise<UESSRMotionBlur> {
    const defaultConfig: Required<SSRMotionBlurConfig> = {
      width: config.width,
      height: config.height,
      hiZLevels: config.hiZLevels ?? DEFAULT_HIZ_LEVELS,
      raySamples: config.raySamples ?? 12,
      velocityTileSize: config.velocityTileSize ?? VELOCITY_TILE_SIZE,
      motionBlurStrength: config.motionBlurStrength ?? 1.0,
      enableSSR: config.enableSSR ?? true,
      enableMotionBlur: config.enableMotionBlur ?? true,
      temporalFilterWeight: config.temporalFilterWeight ?? 0.8,
    };

    const instance = new UESSRMotionBlur(device, defaultConfig);
    await instance.initialize();
    return instance;
  }

  /**
   * Initialize GPU resources and pipelines
   */
  private async initialize(): Promise<void> {
    const { width, height, hiZLevels } = this.config;

    // Create Hi-Z pyramid texture
    this.hiZPyramid = this.device.createTexture({
      size: { width, height, depthOrArrayLayers: hiZLevels },
      format: "r32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
      mipLevelCount: hiZLevels,
    });
    this.hiZPyramidView = this.hiZPyramid.createView({
      dimension: "2d-array",
    });

    // Create SSR reflection texture
    this.ssrReflectionTexture = this.device.createTexture({
      size: { width, height },
      format: "rgba16float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create SSR history for temporal filter
    this.ssrHistoryTexture = this.device.createTexture({
      size: { width, height },
      format: "rgba16float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });

    // Create result texture
    this.resultTexture = this.device.createTexture({
      size: { width, height },
      format: "rgba16float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });

    // Create velocity tile buffer
    const tileCountX = Math.ceil(width / this.config.velocityTileSize);
    const tileCountY = Math.ceil(height / this.config.velocityTileSize);
    this.velocityTileBuffer = this.device.createBuffer({
      size: tileCountX * tileCountY * 32, // 2 × vec4 per tile
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });

    // Create SSR tile mask buffer
    const ssrTileCountX = Math.ceil(width / SSR_TILE_SIZE);
    const ssrTileCountY = Math.ceil(height / SSR_TILE_SIZE);
    const maskSizeBytes = Math.ceil((ssrTileCountX * ssrTileCountY) / 32) * 4;
    this.ssrTileMaskBuffer = this.device.createBuffer({
      size: maskSizeBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });

    // Initialize pipelines (stub implementations)
    await this.initializePipelines();
  }

  /**
   * Initialize compute pipelines
   */
  private async initializePipelines(): Promise<void> {
    // Stub: In production, compile WGSL shaders and create pipelines
    // For now, pipelines remain null and will be filled with actual shader code
    console.log("Initializing compute pipelines (WGSL)...");
  }

  /**
   * Update scene data: depth, normal, roughness, velocity textures
   */
  updateSceneData(
    depthTex: GPUTexture,
    normalTex: GPUTexture,
    roughnessTex: GPUTexture,
    velocityTex: GPUTexture
  ): void {
    this.depthTexture = depthTex;
    this.normalTexture = normalTex;
    this.roughnessTexture = roughnessTex;
    this.velocityTexture = velocityTex;
  }

  /**
   * Update projection matrices for ray march
   */
  setProjectionMatrices(
    proj: Float32Array,
    view: Float32Array,
    invProj: Float32Array,
    invView: Float32Array
  ): void {
    this.projMatrix.set(proj);
    this.viewMatrix.set(view);
    this.invProjMatrix.set(invProj);
    this.invViewMatrix.set(invView);
  }

  /**
   * Register Cell velocity for this frame
   */
  registerCellVelocity(data: CellVelocityData): void {
    this.cellVelocityMap.set(data.cellId, data);
  }

  /**
   * Build Hi-Z pyramid from depth texture
   */
  buildHiZPyramid(encoder: GPUCommandEncoder): void {
    if (!this.depthTexture || !this.hiZPyramid) {
      console.warn("buildHiZPyramid: missing depth or hiZ texture");
      return;
    }

    // Stub: Implement mipmap generation from depth texture
    // Pass 1: Copy depth to mip 0
    // Pass 2-N: Downsample each mip level using max depth reduction
    console.log("Building Hi-Z pyramid...");
  }

  /**
   * Classify SSR tiles: mark which tiles have mirror-like properties
   */
  classifySSRTiles(encoder: GPUCommandEncoder): void {
    if (!this.normalTexture || !this.roughnessTexture || !this.ssrTileMaskBuffer) {
      console.warn("classifySSRTiles: missing resources");
      return;
    }

    // Stub: Compute shader to scan normal/roughness and set bit mask
    // Tile is marked if roughness < threshold AND normal has specular component
    console.log("Classifying SSR tiles...");
  }

  /**
   * Ray march SSR: Find reflections using Hi-Z ray marching
   */
  rayMarchSSR(encoder: GPUCommandEncoder): void {
    if (!this.config.enableSSR || !this.depthTexture || !this.hiZPyramid) {
      return;
    }

    // Stub: Execute ray march compute shader
    // For each pixel:
    //   1. Sample normal from GBuffer
    //   2. Compute ray direction (reflect view direction about normal)
    //   3. Ray march with Hi-Z occlusion test
    //   4. Write result to ssrReflectionTexture
    console.log("Ray marching SSR...");
  }

  /**
   * Temporal denoise SSR: Blend current frame with history
   */
  temporalDenoise(encoder: GPUCommandEncoder): void {
    if (!this.ssrReflectionTexture || !this.ssrHistoryTexture) {
      return;
    }

    // Stub: Blend current SSR with history using confidence weights
    // confidence = temporal stability metric (lower variance → higher confidence)
    console.log("Temporal denoising SSR...");

    // Copy current to history for next frame
    encoder.copyTextureToTexture(
      { texture: this.ssrReflectionTexture },
      { texture: this.ssrHistoryTexture },
      [this.config.width, this.config.height]
    );
  }

  /**
   * Velocity flatten: Aggregate per-pixel velocities to tile-level max velocities
   */
  velocityFlatten(encoder: GPUCommandEncoder): void {
    if (!this.config.enableMotionBlur || !this.velocityTexture) {
      return;
    }

    // Stub: Compute shader to reduce velocity buffer
    // For each 16×16 tile:
    //   1. Load all 256 pixel velocities
    //   2. Convert to polar coordinates
    //   3. Find top 2 by magnitude
    //   4. Write to velocityTileBuffer
    console.log("Flattening velocity to tiles...");
  }

  /**
   * Motion blur apply: Blur scene color along velocity directions
   */
  motionBlurApply(encoder: GPUCommandEncoder): void {
    if (!this.config.enableMotionBlur || !this.velocityTileBuffer) {
      return;
    }

    // Stub: Execute motion blur compute shader
    // For each pixel:
    //   1. Load velocity tile containing this pixel
    //   2. Gather samples along velocity direction
    //   3. Apply weighting (cosine falloff)
    //   4. Composite with base color
    console.log("Applying motion blur...");
  }

  /**
   * Execute full SSR+MB pipeline
   */
  executeFullPipeline(encoder: GPUCommandEncoder): void {
    if (this.config.enableSSR) {
      this.buildHiZPyramid(encoder);
      this.classifySSRTiles(encoder);
      this.rayMarchSSR(encoder);
      this.temporalDenoise(encoder);
    }

    if (this.config.enableMotionBlur) {
      this.velocityFlatten(encoder);
      this.motionBlurApply(encoder);
    }
  }

  /**
   * Get final result texture
   */
  getResult(): GPUTexture {
    if (!this.resultTexture) {
      throw new Error("Result texture not initialized");
    }
    return this.resultTexture;
  }

  /**
   * Get SSR reflection texture
   */
  getSSRReflection(): GPUTexture {
    if (!this.ssrReflectionTexture) {
      throw new Error("SSR reflection texture not initialized");
    }
    return this.ssrReflectionTexture;
  }

  /**
   * Query SSR result for specific pixel
   */
  getSSRAtPixel(x: number, y: number): SSRReflectionData {
    // Stub: Would require readback from GPU
    return {
      reflectionColor: new Float32Array([0, 0, 0, 0]),
      confidence: 0,
      rayDistance: 0,
    };
  }

  /**
   * Query motion blur velocity at tile
   */
  getMotionBlurVelocityAtTile(tileX: number, tileY: number): [number, number] {
    // Stub: Would require readback from GPU
    return [0, 0];
  }

  /**
   * Get Cell motion blur impact for specific Cell
   */
  getCellMotionBlurIntensity(cellId: string): number {
    const cellData = this.cellVelocityMap.get(cellId);
    if (!cellData) {
      return 0;
    }

    const velMag = Math.sqrt(
      cellData.screenVelocity[0] ** 2 + cellData.screenVelocity[1] ** 2
    );
    // Remap velocity magnitude to blur intensity [0, 1]
    return Math.min(1.0, velMag / (MAX_MOTION_BLUR_SAMPLES * this.config.motionBlurStrength));
  }

  /**
   * Clear all Cell velocity data for next frame
   */
  clearCellVelocities(): void {
    this.cellVelocityMap.clear();
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.hiZPyramid?.destroy();
    this.ssrReflectionTexture?.destroy();
    this.ssrHistoryTexture?.destroy();
    this.resultTexture?.destroy();
    this.velocityTileBuffer?.destroy();
    this.ssrTileMaskBuffer?.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 4  Advanced Filter & Denoise Kernels
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Separable Motion Blur Filter
 * Applies two passes: horizontal then vertical, for smoother results
 */
const MOTION_BLUR_SEPARABLE_SHADER = `
// Horizontal gather with edge-aware weighting
fn gatherMotionBlurHorizontal(
  pixelCoord: vec2<u32>,
  baseColor: vec4<f32>,
  velocityX: f32,
  sceneTex: texture_2d<f32>,
  sceneSampler: sampler,
  viewSize: vec2<f32>,
  strength: f32
) -> vec4<f32> {
  if (abs(velocityX) < 0.5) {
    return baseColor;
  }

  let pixelUV = vec2<f32>(pixelCoord) / viewSize;
  let sampleCount = min(u32(ceil(abs(velocityX))), 16u);
  var accumColor = baseColor * (1.0 + f32(sampleCount));
  var edgeWeight = 1.0;

  for (var i: i32 = -i32(sampleCount / 2u); i <= i32(sampleCount / 2u); i = i + 1) {
    if (i == 0) { continue; }
    let t = f32(i) / f32(sampleCount);
    let offset = vec2<f32>(velocityX * t * strength / viewSize.x, 0.0);
    let sampleUV = pixelUV + offset;

    if (all(sampleUV >= vec2<f32>(0.0)) && all(sampleUV <= vec2<f32>(1.0))) {
      let sampleColor = textureSample(sceneTex, sceneSampler, sampleUV);
      let weight = 1.0 - abs(t);
      edgeWeight = edgeWeight * (1.0 - length(sampleColor.rgb - baseColor.rgb) * 0.1);
      accumColor = accumColor + sampleColor * weight * edgeWeight;
    }
  }

  return accumColor / (1.0 + f32(sampleCount) * 0.5);
}
`;

/**
 * Tile-based Motion Blur Classification & Filtering
 * Classify pixels by motion blur intensity for adaptive processing
 */
const MOTION_BLUR_TILE_CLASSIFY_SHADER = `
struct TileClassificationData {
  avgVelocity: vec2<f32>,
  maxVelocity: vec2<f32>,
  tileClassification: u32,
}

fn classifyMotionBlurTile(
  velocityData: array<vec2<f32>, 256>,
  count: u32
) -> TileClassificationData {
  var result: TileClassificationData;
  var sumVel = vec2<f32>(0.0);
  var maxLen = 0.0;

  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let v = velocityData[i];
    let len = length(v);
    sumVel = sumVel + v;
    maxLen = max(maxLen, len);
  }

  result.avgVelocity = sumVel / f32(count);
  result.maxVelocity = result.avgVelocity + vec2<f32>(maxLen * 0.1);

  let avgMag = length(result.avgVelocity);
  if (avgMag < 0.5) {
    result.tileClassification = 0u;
  } else if (avgMag < 5.0) {
    result.tileClassification = 1u;
  } else if (avgMag < 15.0) {
    result.tileClassification = 2u;
  } else {
    result.tileClassification = 3u;
  }

  return result;
}
`;

/**
 * Hi-Z Occlusion Refinement for SSR Ray March
 * Precise hit detection using binary search
 */
const SSR_HIZ_REFINE_SHADER = `
fn refineRayHit(
  rayOrigin: vec3<f32>,
  rayDir: vec3<f32>,
  coarseHitDist: f32,
  hiZTex: texture_2d<f32>,
  depthTex: texture_2d<f32>,
  depthSampler: sampler,
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  viewSize: vec2<f32>
) -> vec3<f32> {
  var lo = coarseHitDist * 0.5;
  var hi = coarseHitDist;
  var hitPos = rayOrigin + rayDir * coarseHitDist;

  for (var iter: u32 = 0u; iter < 4u; iter = iter + 1u) {
    let mid = (lo + hi) * 0.5;
    let testPos = rayOrigin + rayDir * mid;
    let screenUV = projectToScreenUV(testPos, proj, view, viewSize);
    let depth = textureSample(depthTex, depthSampler, screenUV).x;

    if (testPos.z > depth) {
      hi = mid;
    } else {
      lo = mid;
      hitPos = testPos;
    }
  }

  return hitPos;
}
`;

/**
 * Advanced Velocity Encoding for Efficient Storage
 */
const MOTION_BLUR_VELOCITY_ENCODE_SHADER = `
fn encodeVelocity(velocity: vec2<f32>) -> u32 {
  let mag = min(u32(length(velocity) * 100.0), 1023u);
  let angle = select(0u, u32((atan2(velocity.y, velocity.x) + 3.14159) * 10.0), length(velocity) > 0.01);
  return (mag << 6u) | (angle & 63u);
}

fn decodeVelocity(encoded: u32) -> vec2<f32> {
  let mag = f32((encoded >> 6u) & 1023u) / 100.0;
  let angle = f32(encoded & 63u) / 10.0 - 3.14159;
  return vec2<f32>(mag * cos(angle), mag * sin(angle));
}
`;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 5  Advanced Statistics & Debugging
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Motion Blur Statistics Tracker
 * Collects runtime metrics for optimization and debugging
 */
export interface MotionBlurStats {
  frameCount: number;
  avgMotionMagnitude: number;
  maxMotionMagnitude: number;
  staticPixelCount: number;
  dynamicPixelCount: number;
  ssrHitCount: number;
  ssrMissCount: number;
  averageRayIterations: number;
  cpuTimeMs: number;
  gpuTimeMs: number;
}

export class MotionBlurStatsCollector {
  private stats: MotionBlurStats = {
    frameCount: 0,
    avgMotionMagnitude: 0,
    maxMotionMagnitude: 0,
    staticPixelCount: 0,
    dynamicPixelCount: 0,
    ssrHitCount: 0,
    ssrMissCount: 0,
    averageRayIterations: 0,
    cpuTimeMs: 0,
    gpuTimeMs: 0,
  };

  private motionMagnitudes: number[] = [];
  private rayIterationCounts: number[] = [];

  recordMotionMagnitude(mag: number): void {
    this.motionMagnitudes.push(mag);
    this.stats.maxMotionMagnitude = Math.max(this.stats.maxMotionMagnitude, mag);
  }

  recordRayIteration(count: number): void {
    this.rayIterationCounts.push(count);
  }

  recordSSRHit(): void {
    this.stats.ssrHitCount++;
  }

  recordSSRMiss(): void {
    this.stats.ssrMissCount++;
  }

  recordPixelType(isStatic: boolean): void {
    if (isStatic) {
      this.stats.staticPixelCount++;
    } else {
      this.stats.dynamicPixelCount++;
    }
  }

  recordTiming(cpuMs: number, gpuMs: number): void {
    this.stats.cpuTimeMs = cpuMs;
    this.stats.gpuTimeMs = gpuMs;
  }

  finalize(): MotionBlurStats {
    if (this.motionMagnitudes.length > 0) {
      this.stats.avgMotionMagnitude =
        this.motionMagnitudes.reduce((a, b) => a + b, 0) / this.motionMagnitudes.length;
    }
    if (this.rayIterationCounts.length > 0) {
      this.stats.averageRayIterations =
        this.rayIterationCounts.reduce((a, b) => a + b, 0) / this.rayIterationCounts.length;
    }
    this.stats.frameCount++;

    return this.stats;
  }

  reset(): void {
    this.motionMagnitudes = [];
    this.rayIterationCounts = [];
  }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// § 6  Export & Integration
// ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Factory function for convenience
 */
export async function createUESSRMotionBlur(
  device: GPUDevice,
  width: number,
  height: number,
  options?: Partial<SSRMotionBlurConfig>
): Promise<UESSRMotionBlur> {
  return UESSRMotionBlur.create(device, {
    width,
    height,
    ...options,
  });
}

/**
 * Utility to compute Cell screen velocity from world positions and projection matrices
 */
export function computeCellScreenVelocity(
  cellId: string,
  currentWorldPos: [number, number, number],
  previousWorldPos: [number, number, number],
  currentProj: Float32Array,
  currentView: Float32Array,
  previousProj: Float32Array,
  previousView: Float32Array,
  viewportWidth: number,
  viewportHeight: number
): CellVelocityData {
  // Project current world pos to current screen space
  const curr = multiplyMatVec4(
    currentProj,
    multiplyMatVec4(currentView, [currentWorldPos[0], currentWorldPos[1], currentWorldPos[2], 1])
  );
  const currUv = [curr[0] / curr[3], curr[1] / curr[3]];
  const currScreen = [
    (currUv[0] + 1) * 0.5 * viewportWidth,
    (1 - currUv[1]) * 0.5 * viewportHeight, // Flip Y
  ];

  // Project previous world pos to previous screen space
  const prev = multiplyMatVec4(
    previousProj,
    multiplyMatVec4(previousView, [previousWorldPos[0], previousWorldPos[1], previousWorldPos[2], 1])
  );
  const prevUv = [prev[0] / prev[3], prev[1] / prev[3]];
  const prevScreen = [
    (prevUv[0] + 1) * 0.5 * viewportWidth,
    (1 - prevUv[1]) * 0.5 * viewportHeight,
  ];

  const screenVelocity = [currScreen[0] - prevScreen[0], currScreen[1] - prevScreen[1]];

  return {
    cellId,
    worldPos: new Float32Array(currentWorldPos),
    prevWorldPos: new Float32Array(previousWorldPos),
    screenVelocity: new Float32Array(screenVelocity),
  };
}

/**
 * Matrix-vector multiplication helper (4x4 × vec4)
 */
function multiplyMatVec4(mat: Float32Array, vec: number[]): number[] {
  const result: number[] = new Array(4).fill(0);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      result[i] += mat[i * 4 + j] * vec[j];
    }
  }
  return result;
}

// Exported shader source constants for external integration
export {
  SSR_RAY_MARCH_SHADER,
  VELOCITY_FLATTEN_SHADER,
  MOTION_BLUR_APPLY_SHADER,
  TEMPORAL_DENOISE_SHADER,
  MOTION_BLUR_SEPARABLE_SHADER,
  MOTION_BLUR_TILE_CLASSIFY_SHADER,
  SSR_HIZ_REFINE_SHADER,
  MOTION_BLUR_VELOCITY_ENCODE_SHADER,
  SSR_TILE_SIZE,
  VELOCITY_TILE_SIZE,
  CONFIG_MAX_RANGE_SIZE,
  MAX_MOTION_BLUR_SAMPLES,
  MIN_MOTION_BLUR_VELOCITY,
  MAX_RAY_MARCH_ITERATIONS,
  DEFAULT_HIZ_LEVELS,
  SSR_RAY_MAX_DIST,
};

/**
 * Integration helper: Bind UESSRMotionBlur output to post-processing pipeline
 * Returns a composite texture with SSR + motion blur applied
 */
export class SSRMotionBlurCompositor {
  private ssrMb: UESSRMotionBlur;
  private blendMode: "additive" | "blend" | "overlay" = "blend";
  private ssrIntensity: number = 1.0;
  private mbIntensity: number = 1.0;

  constructor(ssrMb: UESSRMotionBlur) {
    this.ssrMb = ssrMb;
  }

  setBlendMode(mode: "additive" | "blend" | "overlay"): void {
    this.blendMode = mode;
  }

  setSSRIntensity(intensity: number): void {
    this.ssrIntensity = Math.max(0, Math.min(1, intensity));
  }

  setMotionBlurIntensity(intensity: number): void {
    this.mbIntensity = Math.max(0, Math.min(1, intensity));
  }

  /**
   * Composite SSR reflection + motion blur result with base color
   * shader code for final pass:
   */
  getCompositeShader(): string {
    return `
    fn compositeSSRMotionBlur(
      baseColor: vec4<f32>,
      ssrReflection: vec4<f32>,
      blurredColor: vec4<f32>,
      ssrIntensity: f32,
      mbIntensity: f32,
      blendMode: u32
    ) -> vec4<f32> {
      // Apply SSR (specular reflection) with intensity
      var result = baseColor;
      result = mix(result, ssrReflection, ssrIntensity * ssrReflection.a);

      // Apply motion blur on top
      if (blendMode == 0u) {
        // Additive blend
        result.rgb = result.rgb + blurredColor.rgb * mbIntensity;
      } else if (blendMode == 1u) {
        // Standard blend
        result = mix(result, blurredColor, mbIntensity);
      } else {
        // Overlay blend (multiply + screen)
        let overlay = result.rgb * blurredColor.rgb + result.rgb + blurredColor.rgb - 2.0 * result.rgb * blurredColor.rgb;
        result.rgb = mix(result.rgb, overlay, mbIntensity);
      }

      return result;
    }
    `;
  }
}

/**
 * Performance monitoring and adaptive quality adjustment
 */
export class AdaptiveMotionBlurQuality {
  private targetFrameTime: number = 16.66; // 60 FPS
  private currentQuality: "low" | "medium" | "high" | "ultra" = "high";
  private raySamplesBudget: number = 12;
  private frameTimeHistory: number[] = [];
  private readonly maxHistoryLength: number = 30;

  recordFrameTime(timeMs: number): void {
    this.frameTimeHistory.push(timeMs);
    if (this.frameTimeHistory.length > this.maxHistoryLength) {
      this.frameTimeHistory.shift();
    }
  }

  getAverageFrameTime(): number {
    if (this.frameTimeHistory.length === 0) return 0;
    return this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length;
  }

  shouldAdjustQuality(): boolean {
    const avgTime = this.getAverageFrameTime();
    return Math.abs(avgTime - this.targetFrameTime) > 3.0; // > 3ms deviation
  }

  adjustQualityBasedOnPerformance(): void {
    const avgTime = this.getAverageFrameTime();

    if (avgTime > this.targetFrameTime * 1.2) {
      // Frame time too high, reduce quality
      switch (this.currentQuality) {
        case "ultra":
          this.currentQuality = "high";
          this.raySamplesBudget = 12;
          break;
        case "high":
          this.currentQuality = "medium";
          this.raySamplesBudget = 8;
          break;
        case "medium":
          this.currentQuality = "low";
          this.raySamplesBudget = 4;
          break;
      }
    } else if (avgTime < this.targetFrameTime * 0.8) {
      // Frame time good, increase quality
      switch (this.currentQuality) {
        case "low":
          this.currentQuality = "medium";
          this.raySamplesBudget = 8;
          break;
        case "medium":
          this.currentQuality = "high";
          this.raySamplesBudget = 12;
          break;
        case "high":
          this.currentQuality = "ultra";
          this.raySamplesBudget = 16;
          break;
      }
    }
  }

  getCurrentQuality(): string {
    return this.currentQuality;
  }

  getRaySampleBudget(): number {
    return this.raySamplesBudget;
  }
}
