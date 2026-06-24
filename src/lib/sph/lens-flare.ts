/**
 * lens-flare.ts — M797: Lens Flare — Ghosts + Halos + Star Bursts
 * ─────────────────────────────────────────────────────────────────────────────
 * 镜头光晕后处理——当摄像机靠近光源时在 screen space 产生完整的光学伪像：
 *   • Ghosts        — 沿光源-中心轴线排列的多层半透明缩放副本
 *   • Halo Ring     — 环绕光源的衍射光环
 *   • Star Burst    — 从光源中心向外辐射的十字/多角星芒
 *   • Anamorphic Streak (可选) — 变形镜头水平拉伸光条纹
 *
 * 全部基于 screen-space 计算（无 3D 光线追踪），通过多层 texture 叠加
 * 在 composite pass 合成到场景中。遮挡测试（Occlusion Test）在 compute pass
 * 中采样 depth buffer，当光源被 Cell 几何体遮挡时光晕自动消失/淡出。
 *
 * 算法概览
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Occlusion Test (compute) ────────────────────────────────────┐
 *   │  对每个光源：在 depth buffer 上采样光源位置周围 N×N 邻域               │
 *   │  计算可见比率 visibility ∈ [0,1]（0 = 完全被 Cell 遮挡，1 = 全可见）    │
 *   │  平滑衰减 (temporal EMA) 避免突兀切换                                   │
 *   │  输出 → occlusionBuf (per-light float visibility)                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ occlusionBuf
 *                ▼
 *   ┌─ Pass 1 ── Ghost Generation (compute) ──────────────────────────────────┐
 *   │  场景 bright-pass 提取 → downsample → 沿光轴翻转并缩放 N 层 ghost       │
 *   │  每层 ghost 独立 scale / tint / distortion，模拟多层镜片内反射            │
 *   │  输出 → ghostTex (rgba16float, half-res)                                │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ ghostTex
 *                ▼
 *   ┌─ Pass 2 ── Halo + StarBurst (compute) ──────────────────────────────────┐
 *   │  Halo: 到光源 UV 距离 → 薄环 SDF → 径向 chromatic aberration            │
 *   │  StarBurst: 基于光源 UV 角度的放射状 streak 图案                         │
 *   │    衍射尖刺数量可配置 (4/6/8)，高光强度径向指数衰减                       │
 *   │  Anamorphic streak (可选): 水平方向 1D Gaussian blur,                    │
 *   │    aspect ratio 可达 20:1，模拟变形宽银幕镜头光条                         │
 *   │  输出 → featuresTex (rgba16float, full-res)                             │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ featuresTex
 *                ▼
 *   ┌─ Pass 3 ── Composite (compute) ─────────────────────────────────────────┐
 *   │  scene + ghostTex + featuresTex × visibility → dst                      │
 *   │  加性混合，应用全局 intensity 和 per-light tint                          │
 *   │  ghostTex 上采样 (bilinear) 到全分辨率后叠加                             │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                │ → dstView (swap-chain surface or next FBO)
 *
 * Occlusion 遮挡测试
 * ─────────────────────────────────────────────────────────────────────────────
 * 对每个光源在 depth buffer 的屏幕投影位置采样 OCCLUSION_KERNEL×OCCLUSION_KERNEL
 * (默认 5×5) 的邻域。将光源深度与 depth buffer 值比较：
 *   • depthSample < lightDepth - bias → 被遮挡 (Cell 在光源前方)
 *   • 否则 → 可见
 * 可见比率 = visibleCount / totalSamples，再通过 temporal EMA (alpha = 0.15)
 * 平滑帧间抖动，使遮挡/解除遮挡过渡柔和自然。
 *
 * 设计决策
 * ─────────────────────────────────────────────────────────────────────────────
 * • Screen-space only: 无需知道 3D 光源几何体，仅需 UV + depth。
 * • Compute pipeline: 与 god-rays.ts / dof-bokeh.ts 一致的架构，
 *   所有 pass 均为 compute dispatch，无需全屏三角形。
 * • Half-res ghost: 降低 ghost 生成开销，bilinear 上采样保持质量。
 * • Temporal smoothing: EMA 避免遮挡状态的二值跳变 (popping)。
 * • Chromatic aberration on halo: 红/绿/蓝三通道微小偏移模拟色散。
 * • Anamorphic streak 独立开关: 非所有场景都需要变形镜头效果。
 *
 * 参考源码
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/god-rays.ts                — WebGPU compute post-process 架构
 *   src/lib/sph/dof-bokeh.ts               — half-res pipeline + temporal
 *   src/lib/sph/at-bloom-postprocess.ts    — bright-pass extraction pattern
 *   src/lib/sph/post-process.ts            — fullscreen compute patterns
 *   upstream/john-chapman/lens-flare       — Ghost + halo + starburst algorithm
 *   upstream/unreal-renderer-ue5           — LensFlareSceneProxy / PostProcessLensFlares.cpp
 *   upstream/lygia/lighting                — Lygia lensFlare.wgsl
 *
 * 快速使用：
 *   const lf = await LensFlareCompute.create(device, width, height);
 *   lf.setLights([
 *     { uvPos: [0.5, 0.3], depth: 0.98, color: [1, 0.95, 0.8], intensity: 1.0 },
 *   ]);
 *   lf.setParams({ ghostCount: 5, haloRadius: 0.45, starBurstRays: 6 });
 *   // 每帧:
 *   lf.dispatch(encoder, sceneTexView, depthTexView, dstTexView);
 *
 * Research: xiaodi #M797 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum number of simultaneous flare light sources. */








const MAX_FLARE_LIGHTS = 4;

/** Number of ghost layers per light (each reflected / scaled independently). */
const MAX_GHOST_LAYERS = 8;

/** Occlusion test kernel size (NxN samples around light UV). */
const OCCLUSION_KERNEL = 5;

/** Workgroup size for compute dispatches (16×16 = 256 threads). */
const WG = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single lens-flare light source definition.
 *
 * Each light source produces its own complete set of flare artifacts (ghosts,
 * halo, star burst). Up to {@link MAX_FLARE_LIGHTS} simultaneous lights.
 */
export interface FlareLightSource {
  /**
   * UV position of the light source in [0,1]² screen space.
   * [0,0] = top-left; [1,1] = bottom-right.
   */
  uvPos: [number, number];

  /**
   * Normalised depth of the light source in [0,1] range (NDC depth).
   * Used by the occlusion test to compare against the depth buffer.
   * 0 = near plane, 1 = far plane.
   */
  depth: number;

  /**
   * Tint color of this light's flare artifacts, RGB in [0,1]³.
   * Applied as a multiplicative tint to ghosts, halo, and star burst.
   */
  color: [number, number, number];

  /**
   * Overall intensity multiplier for this light's flare contribution.
   * @default 1.0
   */
  intensity?: number;
}

/**
 * Single ghost layer configuration for fine-tuning individual reflections.
 */
export interface GhostLayerConfig {
  /**
   * Scale factor for this ghost relative to screen size.
   * Negative values flip the ghost across the center (typical lens behavior).
   * @default computed from layer index
   */
  scale: number;

  /**
   * Tint color multiplier for this ghost layer [r, g, b] in [0,1]³.
   * @default [1, 1, 1]
   */
  tint: [number, number, number];

  /**
   * Opacity of this ghost layer in [0,1].
   * @default 0.15
   */
  opacity: number;
}

/**
 * Tweakable lens-flare parameters (all optional, defaults shown).
 */
export interface LensFlareParams {
  // ── Ghost ───────────────────────────────────────────────────────────────
  /**
   * Number of ghost layers to generate (1–8).
   * More layers = richer flare but higher cost.
   * @default 5
   */
  ghostCount?: number;

  /**
   * Base spacing between ghost layers along the light–center axis.
   * @default 0.25
   */
  ghostSpacing?: number;

  /**
   * Luminance threshold for bright-pass extraction before ghost generation.
   * Only pixels brighter than this produce ghosts.
   * @default 0.7
   */
  ghostThreshold?: number;

  /**
   * Per-layer ghost configurations. If fewer entries than ghostCount,
   * remaining layers use auto-generated defaults.
   */
  ghostLayers?: GhostLayerConfig[];

  /**
   * Chromatic dispersion strength for ghost layers.
   * Higher values create more pronounced RGB fringing.
   * @default 0.01
   */
  ghostChromatic?: number;

  // ── Halo ────────────────────────────────────────────────────────────────
  /**
   * Radius of the halo ring in UV space [0,1].
   * @default 0.45
   */
  haloRadius?: number;

  /**
   * Width (thickness) of the halo ring.
   * @default 0.06
   */
  haloWidth?: number;

  /**
   * Chromatic aberration offset for the halo's RGB channels.
   * Each channel samples at a slightly different radius to simulate dispersion.
   * @default 0.005
   */
  haloChromatic?: number;

  /**
   * Opacity of the halo ring.
   * @default 0.25
   */
  haloOpacity?: number;

  // ── Star Burst ──────────────────────────────────────────────────────────
  /**
   * Number of rays in the star burst pattern.
   * Common values: 4 (cross), 6 (star), 8 (ornate star).
   * @default 6
   */
  starBurstRays?: number;

  /**
   * Angular extent of each ray in radians.
   * Narrower = sharper spikes.
   * @default 0.08
   */
  starBurstWidth?: number;

  /**
   * Length of star burst rays (radius from center in UV space).
   * @default 0.35
   */
  starBurstLength?: number;

  /**
   * Intensity multiplier for star burst contribution.
   * @default 0.3
   */
  starBurstIntensity?: number;

  /**
   * Rotation offset for star burst pattern (radians).
   * Allows rotating the spike pattern to avoid alignment with screen axes.
   * @default 0.0
   */
  starBurstRotation?: number;

  // ── Anamorphic Streak ───────────────────────────────────────────────────
  /**
   * Enable anamorphic (horizontal) streak effect.
   * Simulates the characteristic horizontal light streak of anamorphic lenses.
   * @default false
   */
  anamorphicEnabled?: boolean;

  /**
   * Aspect ratio of the anamorphic streak (width:height).
   * Higher values = wider/longer horizontal streak.
   * @default 10.0
   */
  anamorphicRatio?: number;

  /**
   * Intensity of the anamorphic streak contribution.
   * @default 0.2
   */
  anamorphicIntensity?: number;

  /**
   * Tint color applied to the anamorphic streak [r, g, b].
   * Typical: slightly blue-shifted [0.7, 0.8, 1.0].
   * @default [0.7, 0.85, 1.0]
   */
  anamorphicTint?: [number, number, number];

  // ── Occlusion ───────────────────────────────────────────────────────────
  /**
   * Depth comparison bias for occlusion testing.
   * Small positive bias prevents self-occlusion z-fighting.
   * @default 0.002
   */
  occlusionBias?: number;

  /**
   * Temporal smoothing factor (EMA alpha) for occlusion visibility.
   * Lower = slower fade in/out, higher = faster response.
   * @default 0.15
   */
  occlusionSmoothing?: number;

  // ── Global ──────────────────────────────────────────────────────────────
  /**
   * Master intensity multiplier applied to the final composite.
   * @default 1.0
   */
  globalIntensity?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal resolved types (no optionals)
// ─────────────────────────────────────────────────────────────────────────────

interface ResolvedParams {
  ghostCount        : number;
  ghostSpacing      : number;
  ghostThreshold    : number;
  ghostChromatic    : number;
  haloRadius        : number;
  haloWidth         : number;
  haloChromatic     : number;
  haloOpacity       : number;
  starBurstRays     : number;
  starBurstWidth    : number;
  starBurstLength   : number;
  starBurstIntensity: number;
  starBurstRotation : number;
  anamorphicEnabled : boolean;
  anamorphicRatio   : number;
  anamorphicIntensity: number;
  anamorphicTint    : [number, number, number];
  occlusionBias     : number;
  occlusionSmoothing: number;
  globalIntensity   : number;
}

interface ResolvedLight {
  uvX      : number;
  uvY      : number;
  depth    : number;
  r        : number;
  g        : number;
  b        : number;
  intensity: number;
}

/** Packed ghost-layer data for GPU uniform. */
interface ResolvedGhostLayer {
  scale   : number;
  tintR   : number;
  tintG   : number;
  tintB   : number;
  opacity : number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS: ResolvedParams = {
  ghostCount        : 5,
  ghostSpacing      : 0.25,
  ghostThreshold    : 0.7,
  ghostChromatic    : 0.01,
  haloRadius        : 0.45,
  haloWidth         : 0.06,
  haloChromatic     : 0.005,
  haloOpacity       : 0.25,
  starBurstRays     : 6,
  starBurstWidth    : 0.08,
  starBurstLength   : 0.35,
  starBurstIntensity: 0.3,
  starBurstRotation : 0.0,
  anamorphicEnabled : false,
  anamorphicRatio   : 10.0,
  anamorphicIntensity: 0.2,
  anamorphicTint    : [0.7, 0.85, 1.0],
  occlusionBias     : 0.002,
  occlusionSmoothing: 0.15,
  globalIntensity   : 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Shared uniform structs
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FLARE_UNIFORMS = /* wgsl */`
// ─── Lens Flare Uniform Structures ──────────────────────────────────────────

struct FlareLight {
  uvPos     : vec2f,   // screen UV [0,1]²
  depth     : f32,     // NDC depth for occlusion test
  intensity : f32,     // per-light multiplier
  color     : vec3f,   // tint RGB
  _pad0     : f32,
};

struct GhostLayer {
  scale   : f32,
  tintR   : f32,
  tintG   : f32,
  tintB   : f32,
  opacity : f32,
  _pad0   : f32,
  _pad1   : f32,
  _pad2   : f32,
};

struct FlareUniforms {
  // resolution
  width              : f32,
  height             : f32,
  halfWidth          : f32,
  halfHeight         : f32,

  // ghost params
  ghostCount         : f32,
  ghostSpacing       : f32,
  ghostThreshold     : f32,
  ghostChromatic     : f32,

  // halo params
  haloRadius         : f32,
  haloWidth          : f32,
  haloChromatic      : f32,
  haloOpacity        : f32,

  // star burst params
  starBurstRays      : f32,
  starBurstWidth     : f32,
  starBurstLength    : f32,
  starBurstIntensity : f32,
  starBurstRotation  : f32,

  // anamorphic params
  anamorphicEnabled  : f32,  // 0.0 or 1.0
  anamorphicRatio    : f32,
  anamorphicIntensity: f32,
  anamorphicTintR    : f32,
  anamorphicTintG    : f32,
  anamorphicTintB    : f32,

  // occlusion params
  occlusionBias      : f32,
  occlusionKernel    : f32,  // OCCLUSION_KERNEL (integer stored as float)

  // global
  globalIntensity    : f32,
  numLights          : f32,
  _pad0              : f32,
  _pad1              : f32,
  _pad2              : f32,
  _pad3              : f32,

  // light array
  lights             : array<FlareLight, ${MAX_FLARE_LIGHTS}>,

  // ghost layer array
  ghostLayers        : array<GhostLayer, ${MAX_GHOST_LAYERS}>,
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Pass 0 — Occlusion Test (compute)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_OCCLUSION_COMPUTE = /* wgsl */`
${WGSL_FLARE_UNIFORMS}

@group(0) @binding(0) var<uniform>      u        : FlareUniforms;
@group(0) @binding(1) var               depthSmp : sampler;
@group(0) @binding(2) var               depthTex : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> occlusionBuf : array<f32>;
// occlusionBuf layout: [prevVis0, prevVis1, prevVis2, prevVis3]
//                      followed by [outVis0, outVis1, outVis2, outVis3]

@compute @workgroup_size(${MAX_FLARE_LIGHTS}, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let lightIdx = gid.x;
  if (lightIdx >= u32(u.numLights)) { return; }

  let light = u.lights[lightIdx];
  let kernel = i32(u.occlusionKernel);
  let halfK  = kernel / 2;
  let bias   = u.occlusionBias;
  let lightDepth = light.depth;

  // Texel size for depth texture sampling
  let texelX = 1.0 / u.width;
  let texelY = 1.0 / u.height;

  var visibleCount = 0.0;
  var totalCount   = 0.0;

  for (var dy = -halfK; dy <= halfK; dy++) {
    for (var dx = -halfK; dx <= halfK; dx++) {
      let sampleUV = light.uvPos + vec2f(f32(dx) * texelX * 2.0, f32(dy) * texelY * 2.0);

      // Skip out-of-screen samples
      if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
        continue;
      }

      let depthSample = textureSampleLevel(depthTex, depthSmp, sampleUV, 0.0).r;
      totalCount += 1.0;

      // If depth sample >= light depth - bias, the light is NOT occluded at this texel
      if (depthSample >= lightDepth - bias) {
        visibleCount += 1.0;
      }
    }
  }

  let rawVis = select(0.0, visibleCount / totalCount, totalCount > 0.0);

  // Temporal EMA smoothing: read previous visibility
  let prevVis = occlusionBuf[lightIdx];
  let alpha   = 0.15;  // occlusionSmoothing baked here, overridden by CPU
  let smoothedVis = mix(prevVis, rawVis, alpha);

  // Write smoothed visibility (used by composite pass)
  occlusionBuf[lightIdx] = smoothedVis;
  // Also write to the output slot for CPU readback if needed
  occlusionBuf[${MAX_FLARE_LIGHTS}u + lightIdx] = smoothedVis;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Pass 1 — Ghost Generation (compute)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GHOST_COMPUTE = /* wgsl */`
${WGSL_FLARE_UNIFORMS}

@group(0) @binding(0) var<uniform>      u         : FlareUniforms;
@group(0) @binding(1) var               sceneSmp  : sampler;
@group(0) @binding(2) var               sceneTex  : texture_2d<f32>;
@group(0) @binding(3) var<storage, read> occBuf   : array<f32>;
@group(0) @binding(4) var               ghostOut  : texture_storage_2d<rgba16float, write>;

// Luminance helper
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let halfW = u32(u.halfWidth);
  let halfH = u32(u.halfHeight);
  if (gid.x >= halfW || gid.y >= halfH) { return; }

  let uv = vec2f(
    (f32(gid.x) + 0.5) / u.halfWidth,
    (f32(gid.y) + 0.5) / u.halfHeight,
  );

  let ghostCount   = i32(u.ghostCount);
  let spacing      = u.ghostSpacing;
  let threshold    = u.ghostThreshold;
  let chromatic    = u.ghostChromatic;
  let numLights    = i32(u.numLights);

  var accum = vec4f(0.0);

  for (var li = 0; li < numLights; li++) {
    let light = u.lights[li];
    let vis = occBuf[li];
    if (vis < 0.001) { continue; }

    // Vector from light UV to center (0.5, 0.5)
    let toCenter = vec2f(0.5, 0.5) - light.uvPos;

    for (var gi = 0; gi < ghostCount; gi++) {
      let layer = u.ghostLayers[gi];
      let ghostScale = layer.scale;

      // Ghost sample point: flip UV across center and offset along axis
      let offset = toCenter * spacing * f32(gi + 1);
      let ghostUV = uv + offset * ghostScale;

      // Boundary fade: soft falloff near edges
      let borderDist = min(
        min(ghostUV.x, 1.0 - ghostUV.x),
        min(ghostUV.y, 1.0 - ghostUV.y),
      );
      let borderFade = smoothstep(0.0, 0.1, borderDist);

      if (borderFade < 0.001) { continue; }

      // Sample with chromatic aberration (RGB at slightly different UVs)
      let chrOffset = (ghostUV - vec2f(0.5, 0.5)) * chromatic;
      let sR = textureSampleLevel(sceneTex, sceneSmp, ghostUV - chrOffset, 0.0).r;
      let sG = textureSampleLevel(sceneTex, sceneSmp, ghostUV, 0.0).g;
      let sB = textureSampleLevel(sceneTex, sceneSmp, ghostUV + chrOffset, 0.0).b;
      let sampled = vec3f(sR, sG, sB);

      // Bright-pass filter: only bright pixels contribute
      let lum = luminance(sampled);
      let brightWeight = smoothstep(threshold, threshold + 0.1, lum);
      if (brightWeight < 0.001) { continue; }

      // Apply ghost layer tint and opacity
      let tint = vec3f(layer.tintR, layer.tintG, layer.tintB);
      let contribution = sampled * brightWeight * tint * layer.opacity * borderFade;

      // Apply light color tint and visibility
      accum += vec4f(contribution * light.color * light.intensity * vis, 0.0);
    }
  }

  textureStore(ghostOut, vec2i(gid.xy), accum);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Pass 2 — Halo + StarBurst + Anamorphic (compute)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FEATURES_COMPUTE = /* wgsl */`
${WGSL_FLARE_UNIFORMS}

@group(0) @binding(0) var<uniform>      u          : FlareUniforms;
@group(0) @binding(1) var               sceneSmp   : sampler;
@group(0) @binding(2) var               sceneTex   : texture_2d<f32>;
@group(0) @binding(3) var<storage, read> occBuf    : array<f32>;
@group(0) @binding(4) var               featuresOut: texture_storage_2d<rgba16float, write>;

const PI = 3.14159265359;

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ─── Halo ring SDF with chromatic aberration ────────────────────────────────
fn computeHalo(uv: vec2f, lightUV: vec2f) -> vec3f {
  let toLight = uv - lightUV;
  let dist    = length(toLight);
  let radius  = u.haloRadius;
  let width   = u.haloWidth;
  let chrOff  = u.haloChromatic;

  // SDF: distance from the ring at 'radius' from center
  let ringR = smoothstep(width, 0.0, abs(dist - radius + chrOff));
  let ringG = smoothstep(width, 0.0, abs(dist - radius));
  let ringB = smoothstep(width, 0.0, abs(dist - radius - chrOff));

  return vec3f(ringR, ringG, ringB) * u.haloOpacity;
}

// ─── Star burst — N-ray radial spike pattern ────────────────────────────────
fn computeStarBurst(uv: vec2f, lightUV: vec2f) -> vec3f {
  let toLight   = uv - lightUV;
  let dist      = length(toLight);
  let angle     = atan2(toLight.y, toLight.x) + u.starBurstRotation;
  let numRays   = u.starBurstRays;
  let rayWidth  = u.starBurstWidth;
  let rayLength = u.starBurstLength;
  let intensity = u.starBurstIntensity;

  if (dist > rayLength || dist < 0.001) {
    return vec3f(0.0);
  }

  // Periodic angular function: cos(angle * numRays) creates N-fold symmetry
  let angularFactor = pow(abs(cos(angle * numRays * 0.5)), 1.0 / rayWidth);

  // Radial decay: exponential falloff from center
  let radialDecay = exp(-dist * 8.0 / rayLength);

  // Combine: narrow angular peaks × radial falloff
  let spike = angularFactor * radialDecay * intensity;

  // Slight warmth gradient along rays
  let tint = mix(vec3f(1.0, 0.95, 0.85), vec3f(0.85, 0.9, 1.0), dist / rayLength);
  return tint * spike;
}

// ─── Anamorphic horizontal streak ───────────────────────────────────────────
fn computeAnamorphicStreak(uv: vec2f, lightUV: vec2f) -> vec3f {
  if (u.anamorphicEnabled < 0.5) {
    return vec3f(0.0);
  }

  let ratio     = u.anamorphicRatio;
  let intensity = u.anamorphicIntensity;
  let tint      = vec3f(u.anamorphicTintR, u.anamorphicTintG, u.anamorphicTintB);

  // Vertical distance to light row (very tight)
  let dy = abs(uv.y - lightUV.y);
  let verticalFade = exp(-dy * dy * ratio * ratio * 50.0);

  // Horizontal distance — much wider spread
  let dx = abs(uv.x - lightUV.x);
  let horizontalFade = exp(-dx * dx * ratio * 0.5);

  // Bright hot-spot near center
  let dist = length(uv - lightUV);
  let centerBoost = exp(-dist * dist * 80.0);

  let streak = verticalFade * horizontalFade * intensity + centerBoost * intensity * 0.3;
  return tint * streak;
}

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let w = u32(u.width);
  let h = u32(u.height);
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = vec2f(
    (f32(gid.x) + 0.5) / u.width,
    (f32(gid.y) + 0.5) / u.height,
  );

  let numLights = i32(u.numLights);
  var accum = vec4f(0.0);

  for (var li = 0; li < numLights; li++) {
    let light = u.lights[li];
    let vis = occBuf[li];
    if (vis < 0.001) { continue; }

    let lightColor = light.color * light.intensity * vis;

    // Halo ring
    let halo = computeHalo(uv, light.uvPos);
    accum += vec4f(halo * lightColor, 0.0);

    // Star burst
    let star = computeStarBurst(uv, light.uvPos);
    accum += vec4f(star * lightColor, 0.0);

    // Anamorphic streak
    let streak = computeAnamorphicStreak(uv, light.uvPos);
    accum += vec4f(streak * lightColor, 0.0);
  }

  textureStore(featuresOut, vec2i(gid.xy), accum);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Pass 3 — Composite (compute)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE_COMPUTE = /* wgsl */`
${WGSL_FLARE_UNIFORMS}

@group(0) @binding(0) var<uniform>      u           : FlareUniforms;
@group(0) @binding(1) var               linearSmp   : sampler;
@group(0) @binding(2) var               sceneTex    : texture_2d<f32>;
@group(0) @binding(3) var               ghostTex    : texture_2d<f32>;  // half-res
@group(0) @binding(4) var               featuresTex : texture_2d<f32>;  // full-res
@group(0) @binding(5) var               dstOut      : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let w = u32(u.width);
  let h = u32(u.height);
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = vec2f(
    (f32(gid.x) + 0.5) / u.width,
    (f32(gid.y) + 0.5) / u.height,
  );

  // Scene color pass-through
  let sceneColor = textureSampleLevel(sceneTex, linearSmp, uv, 0.0);

  // Ghost contribution (bilinear upsampled from half-res)
  let ghostColor = textureSampleLevel(ghostTex, linearSmp, uv, 0.0);

  // Features (halo + star burst + anamorphic) at full-res
  let featuresColor = textureSampleLevel(featuresTex, linearSmp, uv, 0.0);

  // Additive composite with global intensity
  let flare = (ghostColor.rgb + featuresColor.rgb) * u.globalIntensity;
  let result = vec4f(sceneColor.rgb + flare, sceneColor.a);

  textureStore(dstOut, vec2i(gid.xy), result);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GPU Uniform layout helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Byte size of the uniform buffer.
 *
 * Layout (std140-like, f32 = 4 bytes):
 *   FlareUniforms scalars : 32 floats = 128 bytes
 *   lights[4] : 4 × 8 floats = 128 bytes
 *   ghostLayers[8] : 8 × 8 floats = 256 bytes
 *   Total: 512 bytes
 */
const UNIFORM_FLOAT_COUNT =
  32                             // scalar params (padded to 32)
  + MAX_FLARE_LIGHTS * 8        // lights (uvPos.xy, depth, intensity, color.rgb, _pad)
  + MAX_GHOST_LAYERS * 8;       // ghost layers (scale, tint.rgb, opacity, pad×3)

const UNIFORM_BYTE_SIZE = UNIFORM_FLOAT_COUNT * 4;

/** Byte size of the occlusion storage buffer: prev[MAX] + out[MAX]. */
const OCCLUSION_BUF_SIZE = MAX_FLARE_LIGHTS * 2 * 4;

// ─────────────────────────────────────────────────────────────────────────────
// Default ghost layer generation
// ─────────────────────────────────────────────────────────────────────────────

function generateDefaultGhostLayers(count: number): ResolvedGhostLayer[] {
  const layers: ResolvedGhostLayer[] = [];
  for (let i = 0; i < count; i++) {
    // Alternating positive/negative scales create the classic reflected-ghost pattern
    const sign = (i % 2 === 0) ? -1 : 1;
    const baseScale = 0.2 + (i / Math.max(1, count - 1)) * 1.2;
    layers.push({
      scale  : sign * baseScale,
      tintR  : 1.0 - i * 0.05,
      tintG  : 1.0 - i * 0.02,
      tintB  : 0.9 + i * 0.015,
      opacity: 0.18 - i * 0.015,
    });
  }
  return layers;
}

// ─────────────────────────────────────────────────────────────────────────────
// LensFlareCompute — Main orchestrator class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GPU-driven lens-flare post-process pipeline.
 *
 * Produces screen-space ghosts, halo ring, star burst, and optional
 * anamorphic streak effects. Occlusion testing against the depth buffer
 * ensures flare artifacts disappear when the light source is hidden behind
 * Cell geometry.
 *
 * All passes are compute dispatches — no render passes or fullscreen
 * triangles required.
 */
export class LensFlareCompute {
  private readonly device : any /*GPUDevice*/;
  private readonly width  : number;
  private readonly height : number;

  // ── GPU resources ─────────────────────────────────────────────────────────
  private uniformBuf    : any /*GPUBuffer*/;
  private occlusionBuf  : any /*GPUBuffer*/;   // storage: temporal visibility per light
  private linearSampler : GPUSampler;

  // Intermediate textures
  private ghostTex    : GPUTexture;    // half-res ghost accumulation
  private featuresTex : GPUTexture;    // full-res halo + starburst + anamorphic
  private dstTex      : GPUTexture;    // full-res composite output

  // Half-resolution dimensions
  private readonly halfW : number;
  private readonly halfH : number;

  // ── Pipeline state ────────────────────────────────────────────────────────
  private occlusionPipeline  : any /*GPUComputePipeline*/;
  private ghostPipeline      : any /*GPUComputePipeline*/;
  private featuresPipeline   : any /*GPUComputePipeline*/;
  private compositePipeline  : any /*GPUComputePipeline*/;

  private occlusionBGL  : GPUBindGroupLayout;
  private ghostBGL      : GPUBindGroupLayout;
  private featuresBGL   : GPUBindGroupLayout;
  private compositeBGL  : GPUBindGroupLayout;

  // ── Runtime state ─────────────────────────────────────────────────────────
  private params : ResolvedParams;
  private lights : ResolvedLight[] = [];
  private ghostLayers : ResolvedGhostLayer[] = [];

  private constructor(
    device : any /*GPUDevice*/,
    width  : number,
    height : number,
  ) {
    this.device = device;
    this.width  = width;
    this.height = height;
    this.halfW  = Math.max(1, width  >> 1);
    this.halfH  = Math.max(1, height >> 1);
    this.params = { ...DEFAULTS };
    this.ghostLayers = generateDefaultGhostLayers(DEFAULTS.ghostCount);

    // ── Sampler ────────────────────────────────────────────────────────────
    this.linearSampler = device.createSampler({
      label        : 'lf-linear-sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    // ── Uniform buffer ─────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      label : 'lf-uniforms',
      size  : UNIFORM_BYTE_SIZE,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Occlusion storage buffer ───────────────────────────────────────────
    this.occlusionBuf = device.createBuffer({
      label : 'lf-occlusion',
      size  : OCCLUSION_BUF_SIZE,
      usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── Intermediate textures ──────────────────────────────────────────────
    this.ghostTex = device.createTexture({
      label  : 'lf-ghost-tex',
      size   : [this.halfW, this.halfH],
      format : 'rgba16float',
      usage  : GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.featuresTex = device.createTexture({
      label  : 'lf-features-tex',
      size   : [width, height],
      format : 'rgba16float',
      usage  : GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.dstTex = device.createTexture({
      label  : 'lf-dst-tex',
      size   : [width, height],
      format : 'rgba16float',
      usage  : GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // ── Pipelines (initialized in create()) ────────────────────────────────
    this.occlusionPipeline  = null!;
    this.ghostPipeline      = null!;
    this.featuresPipeline   = null!;
    this.compositePipeline  = null!;
    this.occlusionBGL       = null!;
    this.ghostBGL           = null!;
    this.featuresBGL        = null!;
    this.compositeBGL       = null!;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Factory
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Create a fully-initialized LensFlareCompute instance.
   *
   * @param device - WebGPU device
   * @param width  - Framebuffer width in pixels
   * @param height - Framebuffer height in pixels
   */
  static async create(
    device : any /*GPUDevice*/,
    width  : number,
    height : number,
  ): Promise<LensFlareCompute> {
    const lf = new LensFlareCompute(device, width, height);
    await lf.initPipelines();
    return lf;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pipeline initialization
  // ─────────────────────────────────────────────────────────────────────────

  private async initPipelines(): Promise<void> {
    const device = this.device;

    // ── Pass 0: Occlusion ──────────────────────────────────────────────────
    this.occlusionBGL = device.createBindGroupLayout({
      label   : 'lf-occlusion-bgl',
      entries : [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const occModule = device.createShaderModule({
      label : 'lf-occlusion-shader',
      code  : WGSL_OCCLUSION_COMPUTE,
    });
    this.occlusionPipeline = await device.createComputePipelineAsync({
      label   : 'lf-occlusion-pipeline',
      layout  : device.createPipelineLayout({ bindGroupLayouts: [this.occlusionBGL] }),
      compute : { module: occModule, entryPoint: 'main' },
    });

    // ── Pass 1: Ghost Generation ───────────────────────────────────────────
    this.ghostBGL = device.createBindGroupLayout({
      label   : 'lf-ghost-bgl',
      entries : [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    const ghostModule = device.createShaderModule({
      label : 'lf-ghost-shader',
      code  : WGSL_GHOST_COMPUTE,
    });
    this.ghostPipeline = await device.createComputePipelineAsync({
      label   : 'lf-ghost-pipeline',
      layout  : device.createPipelineLayout({ bindGroupLayouts: [this.ghostBGL] }),
      compute : { module: ghostModule, entryPoint: 'main' },
    });

    // ── Pass 2: Halo + StarBurst + Anamorphic ──────────────────────────────
    this.featuresBGL = device.createBindGroupLayout({
      label   : 'lf-features-bgl',
      entries : [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    const featuresModule = device.createShaderModule({
      label : 'lf-features-shader',
      code  : WGSL_FEATURES_COMPUTE,
    });
    this.featuresPipeline = await device.createComputePipelineAsync({
      label   : 'lf-features-pipeline',
      layout  : device.createPipelineLayout({ bindGroupLayouts: [this.featuresBGL] }),
      compute : { module: featuresModule, entryPoint: 'main' },
    });

    // ── Pass 3: Composite ──────────────────────────────────────────────────
    this.compositeBGL = device.createBindGroupLayout({
      label   : 'lf-composite-bgl',
      entries : [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    const compModule = device.createShaderModule({
      label : 'lf-composite-shader',
      code  : WGSL_COMPOSITE_COMPUTE,
    });
    this.compositePipeline = await device.createComputePipelineAsync({
      label   : 'lf-composite-pipeline',
      layout  : device.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
      compute : { module: compModule, entryPoint: 'main' },
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: parameter setters
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Update lens-flare parameters. Only provided fields are overwritten.
   */
  setParams(p: LensFlareParams): void {
    if (p.ghostCount         !== undefined) this.params.ghostCount         = Math.max(1, Math.min(MAX_GHOST_LAYERS, p.ghostCount));
    if (p.ghostSpacing       !== undefined) this.params.ghostSpacing       = p.ghostSpacing;
    if (p.ghostThreshold     !== undefined) this.params.ghostThreshold     = p.ghostThreshold;
    if (p.ghostChromatic     !== undefined) this.params.ghostChromatic     = p.ghostChromatic;
    if (p.haloRadius         !== undefined) this.params.haloRadius         = p.haloRadius;
    if (p.haloWidth          !== undefined) this.params.haloWidth          = p.haloWidth;
    if (p.haloChromatic      !== undefined) this.params.haloChromatic      = p.haloChromatic;
    if (p.haloOpacity        !== undefined) this.params.haloOpacity        = p.haloOpacity;
    if (p.starBurstRays      !== undefined) this.params.starBurstRays      = p.starBurstRays;
    if (p.starBurstWidth     !== undefined) this.params.starBurstWidth     = p.starBurstWidth;
    if (p.starBurstLength    !== undefined) this.params.starBurstLength    = p.starBurstLength;
    if (p.starBurstIntensity !== undefined) this.params.starBurstIntensity = p.starBurstIntensity;
    if (p.starBurstRotation  !== undefined) this.params.starBurstRotation  = p.starBurstRotation;
    if (p.anamorphicEnabled  !== undefined) this.params.anamorphicEnabled  = p.anamorphicEnabled;
    if (p.anamorphicRatio    !== undefined) this.params.anamorphicRatio    = p.anamorphicRatio;
    if (p.anamorphicIntensity !== undefined) this.params.anamorphicIntensity = p.anamorphicIntensity;
    if (p.anamorphicTint     !== undefined) this.params.anamorphicTint     = p.anamorphicTint;
    if (p.occlusionBias      !== undefined) this.params.occlusionBias      = p.occlusionBias;
    if (p.occlusionSmoothing !== undefined) this.params.occlusionSmoothing = p.occlusionSmoothing;
    if (p.globalIntensity    !== undefined) this.params.globalIntensity    = p.globalIntensity;

    // Regenerate ghost layers if count changed or custom layers provided
    if (p.ghostLayers) {
      this.ghostLayers = p.ghostLayers.map(l => ({
        scale  : l.scale,
        tintR  : l.tint[0],
        tintG  : l.tint[1],
        tintB  : l.tint[2],
        opacity: l.opacity,
      }));
      // Pad to MAX_GHOST_LAYERS
      while (this.ghostLayers.length < MAX_GHOST_LAYERS) {
        this.ghostLayers.push({ scale: 0, tintR: 0, tintG: 0, tintB: 0, opacity: 0 });
      }
    } else if (p.ghostCount !== undefined) {
      this.ghostLayers = generateDefaultGhostLayers(this.params.ghostCount);
    }
  }

  /**
   * Set the active flare light sources.
   *
   * @param lights - Array of up to {@link MAX_FLARE_LIGHTS} light definitions.
   */
  setLights(lights: FlareLightSource[]): void {
    this.lights = lights.slice(0, MAX_FLARE_LIGHTS).map(l => ({
      uvX      : l.uvPos[0],
      uvY      : l.uvPos[1],
      depth    : l.depth,
      r        : l.color[0],
      g        : l.color[1],
      b        : l.color[2],
      intensity: l.intensity ?? 1.0,
    }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform upload
  // ─────────────────────────────────────────────────────────────────────────

  private uploadUniforms(): void {
    const data = new Float32Array(UNIFORM_FLOAT_COUNT);
    const p = this.params;
    let o = 0;

    // Scalar params (32 floats block)
    data[o++] = this.width;
    data[o++] = this.height;
    data[o++] = this.halfW;
    data[o++] = this.halfH;

    data[o++] = p.ghostCount;
    data[o++] = p.ghostSpacing;
    data[o++] = p.ghostThreshold;
    data[o++] = p.ghostChromatic;

    data[o++] = p.haloRadius;
    data[o++] = p.haloWidth;
    data[o++] = p.haloChromatic;
    data[o++] = p.haloOpacity;

    data[o++] = p.starBurstRays;
    data[o++] = p.starBurstWidth;
    data[o++] = p.starBurstLength;
    data[o++] = p.starBurstIntensity;
    data[o++] = p.starBurstRotation;

    data[o++] = p.anamorphicEnabled ? 1.0 : 0.0;
    data[o++] = p.anamorphicRatio;
    data[o++] = p.anamorphicIntensity;
    data[o++] = p.anamorphicTint[0];
    data[o++] = p.anamorphicTint[1];
    data[o++] = p.anamorphicTint[2];

    data[o++] = p.occlusionBias;
    data[o++] = OCCLUSION_KERNEL;

    data[o++] = p.globalIntensity;
    data[o++] = this.lights.length;
    data[o++] = 0; // _pad0
    data[o++] = 0; // _pad1
    data[o++] = 0; // _pad2
    data[o++] = 0; // _pad3

    // Light array (MAX_FLARE_LIGHTS × 8 floats each)
    for (let i = 0; i < MAX_FLARE_LIGHTS; i++) {
      const base = 32 + i * 8;
      if (i < this.lights.length) {
        const l = this.lights[i];
        data[base + 0] = l.uvX;
        data[base + 1] = l.uvY;
        data[base + 2] = l.depth;
        data[base + 3] = l.intensity;
        data[base + 4] = l.r;
        data[base + 5] = l.g;
        data[base + 6] = l.b;
        data[base + 7] = 0; // _pad
      }
      // else: zeros (inactive light)
    }

    // Ghost layer array (MAX_GHOST_LAYERS × 8 floats each)
    const ghostBase = 32 + MAX_FLARE_LIGHTS * 8;
    for (let i = 0; i < MAX_GHOST_LAYERS; i++) {
      const base = ghostBase + i * 8;
      if (i < this.ghostLayers.length) {
        const gl = this.ghostLayers[i];
        data[base + 0] = gl.scale;
        data[base + 1] = gl.tintR;
        data[base + 2] = gl.tintG;
        data[base + 3] = gl.tintB;
        data[base + 4] = gl.opacity;
        data[base + 5] = 0; // _pad0
        data[base + 6] = 0; // _pad1
        data[base + 7] = 0; // _pad2
      }
    }

    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-frame dispatch
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Dispatch all lens-flare compute passes for the current frame.
   *
   * @param encoder      - Active GPUCommandEncoder
   * @param sceneTexView - Scene color texture view (input)
   * @param depthTexView - Depth buffer texture view (for occlusion testing)
   * @param dstTexView   - Destination texture view (output, additive composite)
   */
  dispatch(
    encoder      : any /*GPUCommandEncoder*/,
    sceneTexView : any /*GPUTextureView*/,
    depthTexView : any /*GPUTextureView*/,
    dstTexView   : any /*GPUTextureView*/,
  ): void {
    if (this.lights.length === 0) return;

    // Upload current parameters
    this.uploadUniforms();

    // ── Pass 0: Occlusion test ─────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label   : 'lf-occlusion-bg',
        layout  : this.occlusionBGL,
        entries : [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: depthTexView },
          { binding: 3, resource: { buffer: this.occlusionBuf } },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'lf-occlusion-pass' });
      pass.setPipeline(this.occlusionPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1); // MAX_FLARE_LIGHTS threads in one workgroup
      pass.end();
    }

    // ── Pass 1: Ghost generation (half-res) ────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label   : 'lf-ghost-bg',
        layout  : this.ghostBGL,
        entries : [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: sceneTexView },
          { binding: 3, resource: { buffer: this.occlusionBuf } },
          { binding: 4, resource: this.ghostTex.createView() },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'lf-ghost-pass' });
      pass.setPipeline(this.ghostPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        Math.ceil(this.halfW / WG),
        Math.ceil(this.halfH / WG),
      );
      pass.end();
    }

    // ── Pass 2: Halo + StarBurst + Anamorphic (full-res) ───────────────────
    {
      const bg = this.device.createBindGroup({
        label   : 'lf-features-bg',
        layout  : this.featuresBGL,
        entries : [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: sceneTexView },
          { binding: 3, resource: { buffer: this.occlusionBuf } },
          { binding: 4, resource: this.featuresTex.createView() },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'lf-features-pass' });
      pass.setPipeline(this.featuresPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        Math.ceil(this.width / WG),
        Math.ceil(this.height / WG),
      );
      pass.end();
    }

    // ── Pass 3: Composite (full-res) ───────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label   : 'lf-composite-bg',
        layout  : this.compositeBGL,
        entries : [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: sceneTexView },
          { binding: 3, resource: this.ghostTex.createView() },
          { binding: 4, resource: this.featuresTex.createView() },
          { binding: 5, resource: dstTexView },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'lf-composite-pass' });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(
        Math.ceil(this.width / WG),
        Math.ceil(this.height / WG),
      );
      pass.end();
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Resize
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Recreate a new LensFlareCompute for a different resolution.
   *
   * Call this when the canvas/framebuffer resizes. The old instance's GPU
   * resources should be destroyed first via {@link destroy}.
   */
  static async resize(
    old    : LensFlareCompute,
    device : any /*GPUDevice*/,
    width  : number,
    height : number,
  ): Promise<LensFlareCompute> {
    const savedParams = { ...old.params };
    const savedLights = [...old.lights];
    const savedLayers = [...old.ghostLayers];
    old.destroy();

    const fresh = await LensFlareCompute.create(device, width, height);
    fresh.params      = savedParams;
    fresh.lights      = savedLights;
    fresh.ghostLayers = savedLayers;
    return fresh;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cleanup
  // ─────────────────────────────────────────────────────────────────────────

  /** Release all GPU resources. */
  destroy(): void {
    this.uniformBuf.destroy();
    this.occlusionBuf.destroy();
    this.ghostTex.destroy();
    this.featuresTex.destroy();
    this.dstTex.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  /** Current resolved parameters (read-only snapshot). */
  getParams(): Readonly<ResolvedParams> {
    return { ...this.params };
  }

  /** Current light sources (read-only snapshot). */
  getLights(): readonly Readonly<ResolvedLight>[] {
    return this.lights.map(l => ({ ...l }));
  }

  /** Intermediate ghost texture view (for debug visualization). */
  getGhostTexView(): any /*GPUTextureView*/ {
    return this.ghostTex.createView();
  }

  /** Intermediate features texture view (for debug visualization). */
  getFeaturesTexView(): any /*GPUTextureView*/ {
    return this.featuresTex.createView();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Species integration helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convenience factory: create a LensFlareCompute tuned for a specific
 * cell-species visual style.
 *
 * Integrates with the species-visual-dna system to apply species-appropriate
 * flare colors, intensities, and star burst patterns.
 *
 * @param device      - WebGPU device
 * @param width       - Framebuffer width
 * @param height      - Framebuffer height
 * @param speciesOpts - Per-species overrides (partial LensFlareParams + lights)
 */
export async function createLensFlareForSpecies(
  device      : any /*GPUDevice*/,
  width       : number,
  height      : number,
  speciesOpts : Partial<LensFlareParams> & { lights?: FlareLightSource[] },
): Promise<LensFlareCompute> {
  const lf = await LensFlareCompute.create(device, width, height);
  lf.setParams({
    ghostCount         : speciesOpts.ghostCount         ?? DEFAULTS.ghostCount,
    ghostSpacing       : speciesOpts.ghostSpacing       ?? DEFAULTS.ghostSpacing,
    ghostThreshold     : speciesOpts.ghostThreshold     ?? DEFAULTS.ghostThreshold,
    ghostChromatic     : speciesOpts.ghostChromatic     ?? DEFAULTS.ghostChromatic,
    haloRadius         : speciesOpts.haloRadius         ?? DEFAULTS.haloRadius,
    haloWidth          : speciesOpts.haloWidth          ?? DEFAULTS.haloWidth,
    haloChromatic      : speciesOpts.haloChromatic      ?? DEFAULTS.haloChromatic,
    haloOpacity        : speciesOpts.haloOpacity        ?? DEFAULTS.haloOpacity,
    starBurstRays      : speciesOpts.starBurstRays      ?? DEFAULTS.starBurstRays,
    starBurstWidth     : speciesOpts.starBurstWidth     ?? DEFAULTS.starBurstWidth,
    starBurstLength    : speciesOpts.starBurstLength    ?? DEFAULTS.starBurstLength,
    starBurstIntensity : speciesOpts.starBurstIntensity ?? DEFAULTS.starBurstIntensity,
    starBurstRotation  : speciesOpts.starBurstRotation  ?? DEFAULTS.starBurstRotation,
    anamorphicEnabled  : speciesOpts.anamorphicEnabled  ?? DEFAULTS.anamorphicEnabled,
    anamorphicRatio    : speciesOpts.anamorphicRatio    ?? DEFAULTS.anamorphicRatio,
    anamorphicIntensity: speciesOpts.anamorphicIntensity ?? DEFAULTS.anamorphicIntensity,
    anamorphicTint     : speciesOpts.anamorphicTint     ?? DEFAULTS.anamorphicTint,
    occlusionBias      : speciesOpts.occlusionBias      ?? DEFAULTS.occlusionBias,
    occlusionSmoothing : speciesOpts.occlusionSmoothing ?? DEFAULTS.occlusionSmoothing,
    globalIntensity    : speciesOpts.globalIntensity    ?? DEFAULTS.globalIntensity,
    ghostLayers        : speciesOpts.ghostLayers,
  });
  if (speciesOpts.lights) {
    lf.setLights(speciesOpts.lights);
  }
  return lf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL source fragments for embedding in other shader modules. */
export const LENS_FLARE_WGSL = {
  /** Shared uniform structs (FlareLight, GhostLayer, FlareUniforms). */
  uniforms   : WGSL_FLARE_UNIFORMS,
  /** Occlusion test compute shader. */
  occlusion  : WGSL_OCCLUSION_COMPUTE,
  /** Ghost generation compute shader. */
  ghost      : WGSL_GHOST_COMPUTE,
  /** Halo + star burst + anamorphic compute shader. */
  features   : WGSL_FEATURES_COMPUTE,
  /** Composite compute shader. */
  composite  : WGSL_COMPOSITE_COMPUTE,
} as const;

/** Maximum number of simultaneous lens-flare light sources. */
export { MAX_FLARE_LIGHTS };

/** Maximum number of ghost layers per light source. */
export { MAX_GHOST_LAYERS };

/** Default lens-flare parameters. */
export { DEFAULTS as LENS_FLARE_DEFAULTS };
