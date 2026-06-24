/**
 * environment-fog.ts — M744: 深度雾 + 体积光 God Rays 合成后处理
 *
 * 将指数深度雾（exp² / exp / linear）与屏幕空间 God Rays 径向模糊合成为
 * 一个统一的多 Pass WebGPU 后处理管线。两者共享同一个光源位置和深度场，
 * 使雾中的体积光散射在物理上保持一致。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 设计来源
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   深度雾：
 *     - src/lib/sph/atmosphere.ts            — exp² 雾 + Rayleigh 散射结构
 *     - upstream/lygia/lighting/fog.glsl     — Lygia 通用雾函数 (linear/exp/exp2)
 *     - GPU Gems 3 Ch.13 "Volumetric Light Scattering as a Post-Process"
 *     - src/lib/CloudFog.ts                  — AT CloudFog 平面云雾 (WebGL2)
 *
 *   体积光 God Rays：
 *     - src/lib/sph/at-volumetric-light.ts   — M716 AT VolumetricLight WGSL Port
 *     - upstream/pixijs-filters/src/godray   — PixiJS God Ray WGSL + Perlin
 *     - upstream/lygia/lighting/volumetricLightScattering.wgsl — Mie 散射
 *     - upstream/unreal-renderer-ue5/Renderer-Private/LightShaftRendering.cpp
 *
 *   合成策略：
 *     - src/lib/sph/at-scene-compositor.ts   — 场景级 pass 编排 (⑦⑧ 步)
 *     - src/lib/sph/particle-compositor.ts   — 深度排序 + alpha 合成
 *     - src/lib/sph/environment-fx.ts        — compute + render 双 pass 结构
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 管线结构（每帧 render()）
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 1: DEPTH FOG ─────────────────────────────────────────────────────┐
 *   │  scene + depthField → fogTex (全分辨率)                                  │
 *   │  从 depthField 采样归一化深度, 按雾模式 (linear/exp/exp²) 计算雾因子,     │
 *   │  混合场景色与雾色。雾色可受光源方向调制 (日出/日落渐变)。                   │
 *   │  输出: 雾化后的场景色 + alpha 通道保留原始深度 (供后续 pass 使用)          │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 2: OCCLUSION EXTRACT ─────────────────────────────────────────────┐
 *   │  scene → occlusionTex (半分辨率)                                         │
 *   │  提取高亮度像素作为光源遮挡掩码。与 at-volumetric-light.ts Pass 1 相同。  │
 *   │  深度加权: 近处高亮获得更高权重, 避免远处雾亮度误触发 god rays。           │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 3: GOD RAY RADIAL BLUR ──────────────────────────────────────────┐
 *   │  occlusionTex → raysTex (半分辨率)                                      │
 *   │  从每像素向光源 UV 径向采样, decay^i 衰减。                              │
 *   │  AT: 6 次迭代, 默认 exposure=0.86, density=0.22。                       │
 *   │  雾密度调制: 在雾浓区域额外增强光散射强度 (雾中光束更明显)。              │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 4: FOG-RAYS COMPOSITE ────────────────────────────────────────────┐
 *   │  fogTex + raysTex → dst (用户指定 GPUTextureView)                       │
 *   │  加性混合: output = fogScene + rays × raysScale                          │
 *   │  Mie 相函数加权: 视角接近光源方向时 god rays 更亮 (前散射)               │
 *   │  深度衰减: 极远处 god rays 融入雾色, 避免深度不连续                       │
 *   │  可选 ACES tone mapping 钳位 HDR 溢出                                   │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 快速使用
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const envFog = await EnvironmentFog.create(device, format, w, h);
 *   envFog.setParams({
 *     fogMode: 'exp2', fogDensity: 0.35, fogColor: [0.02, 0.04, 0.08],
 *     lightPos: [0.5, 0.1], exposure: 0.86, density: 0.22,
 *   });
 *   // 每帧:
 *   envFog.render(encoder, sceneTex, depthFieldTex, dstView);
 *
 * Research: xiaodi #M744 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public parameter types
// ─────────────────────────────────────────────────────────────────────────────

/** Fog attenuation model. */








export type FogMode = 'linear' | 'exp' | 'exp2';

/**
 * Tweakable parameters for EnvironmentFog.
 *
 * 分为三组:
 *   1. 深度雾参数 (fog*)
 *   2. God Rays 参数 (exposure, density, decay, ...)
 *   3. 合成参数 (raysScale, tonemap, ...)
 */
export interface EnvironmentFogParams {
  // ── 深度雾 ──────────────────────────────────────────────────────────────────

  /**
   * 雾衰减模式。
   *   linear: fogFactor = (fogEnd - depth) / (fogEnd - fogStart)
   *   exp:    fogFactor = exp(-fogDensity × depth)
   *   exp2:   fogFactor = exp(-(fogDensity × depth)²)
   * @default 'exp2'
   */
  fogMode?: FogMode;

  /**
   * 雾密度 (exp / exp² 模式使用)。
   * atmosphere.ts: fogDensity 默认 0.18。
   * @default 0.35
   */
  fogDensity?: number;

  /**
   * Linear 雾起始深度 (归一化 [0,1])。
   * @default 0.1
   */
  fogStart?: number;

  /**
   * Linear 雾结束深度 (归一化 [0,1])。
   * @default 0.9
   */
  fogEnd?: number;

  /**
   * 雾色 (线性 sRGB [r, g, b])。
   * 默认: 深海蓝灰, 与 environment-fx.ts bgColor 协调。
   * @default [0.02, 0.04, 0.08]
   */
  fogColor?: [number, number, number];

  /**
   * 雾色受光源方向调制的强度。
   * 0 = 均匀雾色; 1 = 面向光源的雾偏暖 (日出效果)。
   * @default 0.3
   */
  fogLightTint?: number;

  /**
   * 光源暖色调 (用于雾色调制, 线性 sRGB)。
   * @default [0.9, 0.6, 0.3]
   */
  fogLightColor?: [number, number, number];

  /**
   * 深度场反转标志。
   * true = depthField 中 0 = 近, 1 = 远 (标准 Z-buffer)。
   * false = depthField 中高值 = 近 (metaball field 积累, 如 atmosphere.ts)。
   * @default false
   */
  depthInverted?: boolean;

  /**
   * 高度雾衰减指数。
   * >0 时开启高度雾: 低处雾更浓, 高处雾更淡。
   * 0 = 纯深度雾, 不考虑屏幕 Y 坐标。
   * @default 0.0
   */
  heightFalloff?: number;

  // ── God Rays ────────────────────────────────────────────────────────────────

  /**
   * God Rays 总体亮度。
   * AT: VolumetricLight_home fExposure = 0.86。
   * @default 0.86
   */
  exposure?: number;

  /**
   * 径向模糊步进密度 (屏幕空间比例)。
   * AT: VolumetricLight_home fDensity = 0.22。
   * @default 0.22
   */
  density?: number;

  /**
   * 每步指数衰减系数。
   * @default 0.97
   */
  decay?: number;

  /**
   * 基础采样权重。
   * @default 0.4
   */
  weight?: number;

  /**
   * 遮挡掩码亮度阈值。
   * @default 0.6
   */
  occlusionThreshold?: number;

  /**
   * 光源 UV 位置 [0,1]²。
   * [0,0] = 左上; [1,1] = 右下。
   * @default [0.5, 0.05]
   */
  lightPos?: [number, number];

  /**
   * God Rays 径向模糊采样次数。
   * AT: VolumetricLight×6 = 6。
   * @default 6
   */
  numSamples?: number;

  /**
   * 雾密度对 god rays 的增强系数。
   * 在雾浓区域, god rays 散射更强 (物理正确: 雾中光束可见度更高)。
   * @default 0.5
   */
  fogRayBoost?: number;

  // ── 合成 ────────────────────────────────────────────────────────────────────

  /**
   * God rays 层在最终合成时的乘数。
   * @default 1.0
   */
  raysScale?: number;

  /**
   * Mie 散射不对称因子 g ∈ (−1, 1)。
   * Lygia: VOLUMETRICLIGHTSCATTERING_FACTOR = 0.25。
   * @default 0.25
   */
  mieG?: number;

  /**
   * 是否启用 ACES filmic tone mapping 钳位 HDR 溢出。
   * @default true
   */
  tonemap?: boolean;

  /**
   * Tone mapping 前的曝光乘数。
   * @default 1.0
   */
  tonemapExposure?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Default values
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS: Required<EnvironmentFogParams> = {
  fogMode           : 'exp2',
  fogDensity        : 0.35,
  fogStart          : 0.1,
  fogEnd            : 0.9,
  fogColor          : [0.02, 0.04, 0.08],
  fogLightTint      : 0.3,
  fogLightColor     : [0.9, 0.6, 0.3],
  depthInverted     : false,
  heightFalloff     : 0.0,
  exposure          : 0.86,
  density           : 0.22,
  decay             : 0.97,
  weight            : 0.4,
  occlusionThreshold: 0.6,
  lightPos          : [0.5, 0.05],
  numSamples        : 6,
  fogRayBoost       : 0.5,
  raysScale         : 1.0,
  mieG              : 0.25,
  tonemap           : true,
  tonemapExposure   : 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Uniform struct (shared across all passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_UNIFORMS = /* wgsl */`
struct EnvFogUniforms {
  // ── 深度雾 (vec4 aligned rows) ─────────────────────────────────────────────
  fogDensity     : f32,  // exp/exp² 密度
  fogStart       : f32,  // linear 雾起始
  fogEnd         : f32,  // linear 雾结束
  fogMode        : f32,  // 0=linear, 1=exp, 2=exp2

  fogR           : f32,
  fogG           : f32,
  fogB           : f32,
  fogLightTint   : f32,

  fogLightR      : f32,
  fogLightG      : f32,
  fogLightB      : f32,
  depthInverted  : f32,  // 0.0 or 1.0

  heightFalloff  : f32,
  // ── God Rays ───────────────────────────────────────────────────────────────
  exposure       : f32,  // AT fExposure = 0.86
  density        : f32,  // AT fDensity  = 0.22
  decay          : f32,

  weight         : f32,
  occThreshold   : f32,
  lightPosX      : f32,
  lightPosY      : f32,

  numSamples     : f32,
  fogRayBoost    : f32,
  // ── 合成 ───────────────────────────────────────────────────────────────────
  raysScale      : f32,
  mieG           : f32,

  doTonemap      : f32,  // 0.0 or 1.0
  tonemapExposure: f32,
  _pad0          : f32,
  _pad1          : f32,
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Full-screen triangle vertex (reusable across all passes)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FULLSCREEN_VERT = /* wgsl */`
struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  // Full-screen triangle: vid 0→(-1,-1), 1→(3,-1), 2→(-1,3)
  let x = f32(i32(vid & 1u) * 4 - 1);
  let y = f32(i32(vid & 2u) * 2 - 1);
  var out: VsOut;
  out.pos = vec4<f32>(x, y, 0.0, 1.0);
  out.uv  = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Depth Fog
//
// 从 depthField 纹理采样归一化深度, 按雾模式计算雾因子,
// 混合场景色与雾色。
//
// Sources:
//   atmosphere.ts                — exp² 雾结构
//   lygia/lighting/fog.glsl     — linear/exp/exp2 统一接口
//   CloudFog.ts                 — AT 平面雾概念 (alpha = f(depth))
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DEPTH_FOG = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(0) var<uniform> u       : EnvFogUniforms;
@group(0) @binding(1) var          smp     : sampler;
@group(0) @binding(2) var          sceneTex: texture_2d<f32>;
@group(0) @binding(3) var          depthTex: texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

// ── Fog factor functions (lygia/lighting/fog.glsl → WGSL) ────────────────────

fn fogLinear(depth: f32, start: f32, end: f32) -> f32 {
  return clamp((end - depth) / (end - start + 1e-7), 0.0, 1.0);
}

fn fogExp(depth: f32, density: f32) -> f32 {
  return clamp(exp(-density * depth), 0.0, 1.0);
}

fn fogExp2(depth: f32, density: f32) -> f32 {
  let d = density * depth;
  return clamp(exp(-d * d), 0.0, 1.0);
}

@fragment
fn fs_depthFog(in: VsOut) -> @location(0) vec4<f32> {
  let sceneColor = textureSample(sceneTex, smp, in.uv);
  let depthRaw   = textureSample(depthTex, smp, in.uv).r;

  // 深度归一化: 支持标准 Z-buffer (0=近,1=远) 和反转 (高值=近)
  var depth = depthRaw;
  if (u.depthInverted > 0.5) {
    depth = 1.0 - depthRaw;
  }

  // 雾因子: 1.0 = 完全清晰 (无雾), 0.0 = 完全雾化
  var fogFactor: f32;
  let mode = u.fogMode;
  if (mode < 0.5) {
    fogFactor = fogLinear(depth, u.fogStart, u.fogEnd);
  } else if (mode < 1.5) {
    fogFactor = fogExp(depth, u.fogDensity);
  } else {
    fogFactor = fogExp2(depth, u.fogDensity);
  }

  // 可选高度雾: 屏幕 Y 坐标越低 (bottom) 雾越浓
  if (u.heightFalloff > 0.0) {
    // in.uv.y: 0=top, 1=bottom → 低处 (大 Y) 雾更浓
    let heightAtten = exp(-u.heightFalloff * (1.0 - in.uv.y));
    fogFactor = mix(fogFactor, 1.0, heightAtten);
  }

  // 基础雾色
  let baseFogCol = vec3<f32>(u.fogR, u.fogG, u.fogB);

  // 光源方向调制: 面向光源的区域雾色偏暖 (日出/日落效果)
  let lightDir  = normalize(vec2<f32>(u.lightPosX, u.lightPosY) - in.uv);
  let viewDir   = normalize(in.uv - vec2<f32>(0.5, 0.5));
  let sunFacing = dot(lightDir, -viewDir) * 0.5 + 0.5;  // [0,1]
  let warmCol   = vec3<f32>(u.fogLightR, u.fogLightG, u.fogLightB);
  let fogCol    = mix(baseFogCol, warmCol, sunFacing * u.fogLightTint);

  // 混合: sceneColor × fogFactor + fogColor × (1 - fogFactor)
  let fogged = mix(fogCol, sceneColor.rgb, fogFactor);

  // alpha 通道编码原始深度 (供后续 pass 使用)
  return vec4<f32>(fogged, depth);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: Occlusion Extract (半分辨率)
//
// 提取高亮度像素。深度加权: 近处高亮权重更高, 避免远处雾亮度
// 误触发 god rays。
//
// 参考: at-volumetric-light.ts WGSL_OCCLUSION
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_OCCLUSION = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(0) var<uniform> u       : EnvFogUniforms;
@group(0) @binding(1) var          smp     : sampler;
@group(0) @binding(2) var          sceneTex: texture_2d<f32>;
@group(0) @binding(3) var          depthTex: texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

@fragment
fn fs_occlusion(in: VsOut) -> @location(0) vec4<f32> {
  let color    = textureSample(sceneTex, smp, in.uv);
  let depthRaw = textureSample(depthTex, smp, in.uv).r;

  // Rec.709 luminance
  let lum = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

  // 深度加权: 近处物体 (低 depth) 的高亮更可能是光源
  var depth = depthRaw;
  if (u.depthInverted > 0.5) {
    depth = 1.0 - depthRaw;
  }
  let depthWeight = 1.0 - depth * 0.6;  // 近处=1.0, 远处=0.4

  let keep = step(u.occThreshold, lum * depthWeight);
  return vec4<f32>(color.rgb * keep, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: God Ray Radial Blur (半分辨率)
//
// 径向模糊 + 雾密度增强。在雾浓区域, 光散射更明显 (物理正确)。
//
// Sources:
//   at-volumetric-light.ts WGSL_GOD_RAYS — 核心径向模糊
//   GPU Gems 3 Ch.13 — 光散射后处理
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GOD_RAYS = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(0) var<uniform> u          : EnvFogUniforms;
@group(0) @binding(1) var          smp        : sampler;
@group(0) @binding(2) var          occTex     : texture_2d<f32>;
@group(0) @binding(3) var          foggedTex  : texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

@fragment
fn fs_godRays(in: VsOut) -> @location(0) vec4<f32> {
  let lightPos = vec2<f32>(u.lightPosX, u.lightPosY);
  let toLight  = lightPos - in.uv;
  let n        = max(u.numSamples, 1.0);
  let stepVec  = toLight * (u.density / n);

  var sampleUV   = in.uv;
  var accumLight = vec3<f32>(0.0);
  var w          = u.weight;

  // AT VolumetricLight×6 径向模糊展开
  let iMax = i32(clamp(u.numSamples, 1.0, 32.0));
  for (var i = 0; i < iMax; i++) {
    sampleUV   += stepVec;
    let sample  = textureSample(occTex, smp, sampleUV).rgb;
    accumLight += sample * w;
    w          *= u.decay;
  }

  // 曝光缩放 (AT: fExposure = 0.86)
  var rayColor = accumLight * u.exposure;

  // 雾密度增强: 从 foggedTex 的 alpha 通道读取编码深度,
  // 根据深度推算雾浓度, 在雾浓区域增强 god rays 散射
  let encodedDepth = textureSample(foggedTex, smp, in.uv).a;
  var fogAmount: f32;
  let mode = u.fogMode;
  if (mode < 0.5) {
    fogAmount = 1.0 - fogLinear(encodedDepth, u.fogStart, u.fogEnd);
  } else if (mode < 1.5) {
    fogAmount = 1.0 - fogExp(encodedDepth, u.fogDensity);
  } else {
    fogAmount = 1.0 - fogExp2(encodedDepth, u.fogDensity);
  }
  // 雾中光束增强: fogAmount 0=清晰, 1=全雾
  rayColor *= 1.0 + fogAmount * u.fogRayBoost;

  return vec4<f32>(rayColor, 1.0);
}

// 复用深度雾函数 (与 Pass 1 一致)
fn fogLinear(depth: f32, start: f32, end: f32) -> f32 {
  return clamp((end - depth) / (end - start + 1e-7), 0.0, 1.0);
}
fn fogExp(depth: f32, density: f32) -> f32 {
  return clamp(exp(-density * depth), 0.0, 1.0);
}
fn fogExp2(depth: f32, density: f32) -> f32 {
  let d = density * depth;
  return clamp(exp(-d * d), 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 4: Fog-Rays Composite
//
// 将雾化场景与 god rays 加性合成, 施加 Mie 相函数加权和可选
// ACES filmic tone mapping。
//
// Sources:
//   at-volumetric-light.ts WGSL_COMPOSITE — 加性混合
//   at-volumetric-light.ts WGSL_MIE_SCATTER — Henyey-Greenstein 相函数
//   tone-mapping.ts — ACES filmic curve
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */`
${WGSL_UNIFORMS}

@group(0) @binding(0) var<uniform> u        : EnvFogUniforms;
@group(0) @binding(1) var          smp      : sampler;
@group(0) @binding(2) var          fogTex   : texture_2d<f32>;
@group(0) @binding(3) var          raysTex  : texture_2d<f32>;

${WGSL_FULLSCREEN_VERT}

const PI: f32 = 3.14159265358979323846;

// ── Henyey-Greenstein Mie 相函数 (lygia volumetricLightScattering.wgsl) ──────
fn henyeyGreenstein(g: f32, cosTheta: f32) -> f32 {
  let g2  = g * g;
  let num = 1.0 - g2;
  let den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
  return num / (4.0 * PI * den + 1e-7);
}

// ── ACES filmic tone mapping (Krzysztof Narkowicz 简化版) ────────────────────
// src/lib/sph/tone-mapping.ts 中有完整 RRT+ODT, 这里用简化版保证性能
fn acesFilm(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
  let fogScene = textureSample(fogTex,  smp, in.uv);
  let rays     = textureSample(raysTex, smp, in.uv);

  // ── Mie 相函数加权 ────────────────────────────────────────────────────────
  // 视角接近光源方向时 god rays 更亮 (前散射)
  let lightPos  = vec2<f32>(u.lightPosX, u.lightPosY);
  let toLightUV = normalize(lightPos - in.uv);
  let centerDir = normalize(in.uv - vec2<f32>(0.5));
  let cosTheta  = dot(toLightUV, -centerDir) * 0.5 + 0.5;

  let phase     = henyeyGreenstein(u.mieG, cosTheta);
  let phaseNorm = clamp(phase / (1.0 / (4.0 * PI) + 1e-7), 0.0, 3.0);
  let mieWeight = mix(1.0, phaseNorm, 0.35);

  // ── 深度衰减: 极远处 god rays 融入雾色 ────────────────────────────────────
  // fogScene.a 编码深度 (Pass 1 写入)
  let depth        = fogScene.a;
  let depthFade    = mix(1.0, 0.15, smoothstep(0.7, 1.0, depth));

  // ── 加性合成 ──────────────────────────────────────────────────────────────
  var result = fogScene.rgb + rays.rgb * u.raysScale * mieWeight * depthFade;

  // ── 可选 ACES tone mapping ────────────────────────────────────────────────
  if (u.doTonemap > 0.5) {
    result = acesFilm(result * u.tonemapExposure);
  }

  return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Create a GPUTexture for an intermediate render target. */
function makeRT(
  device : GPUDevice,
  width  : number,
  height : number,
  format : GPUTextureFormat,
  label  : string,
): GPUTexture {
  return device.createTexture({
    label,
    size  : [width, height, 1],
    format,
    usage : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
}

/** Build a bind-group-layout: uniform + sampler + N textures. */
function makeBGL(
  device      : GPUDevice,
  label       : string,
  numTextures : number,
): GPUBindGroupLayout {
  const entries: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
      buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT,
      sampler: { type: 'filtering' } },
  ];
  for (let i = 0; i < numTextures; i++) {
    entries.push({
      binding    : 2 + i,
      visibility : GPUShaderStage.FRAGMENT,
      texture    : { sampleType: 'float', viewDimension: '2d' },
    });
  }
  return device.createBindGroupLayout({ label, entries });
}

/** Build a render pipeline with the given WGSL and entry points. */
function makePipeline(
  device  : GPUDevice,
  label   : string,
  wgsl    : string,
  vsEntry : string,
  fsEntry : string,
  bgl     : GPUBindGroupLayout,
  format  : GPUTextureFormat,
): GPURenderPipeline {
  const module = device.createShaderModule({ label, code: wgsl });
  const layout = device.createPipelineLayout({
    label,
    bindGroupLayouts: [bgl],
  });
  return device.createRenderPipeline({
    label,
    layout,
    vertex   : { module, entryPoint: vsEntry },
    fragment : { module, entryPoint: fsEntry, targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}

/** Fog mode string → uniform float. */
function fogModeToFloat(mode: FogMode): number {
  switch (mode) {
    case 'linear': return 0.0;
    case 'exp':    return 1.0;
    case 'exp2':   return 2.0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// EnvironmentFog — main WebGPU class
// ─────────────────────────────────────────────────────────────────────────────

/** Uniform buffer size: 28 f32 = 112 bytes (7 × vec4 aligned). */
const UNIFORM_FLOATS = 28;
const UNIFORM_SIZE   = UNIFORM_FLOATS * 4;

/**
 * EnvironmentFog — 深度雾 + 体积光 God Rays 统一合成后处理。
 *
 * 四 Pass 管线:
 *   1. 深度雾化 (scene + depth → fogTex)
 *   2. 遮挡提取 (scene + depth → occlusionTex, 半分辨率)
 *   3. 径向模糊 god rays (occlusionTex + fogTex → raysTex, 半分辨率)
 *   4. Mie 加权合成 + ACES tone mapping (fogTex + raysTex → dst)
 */
export class EnvironmentFog {
  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;
  private width  : number;
  private height : number;

  // ── GPU resources ──────────────────────────────────────────────────────────
  private uniformBuf : GPUBuffer;
  private sampler    : GPUSampler;

  private fogTex       : GPUTexture;   // 全分辨率: 雾化场景 (alpha=depth)
  private occlusionTex : GPUTexture;   // 半分辨率: 亮度遮挡掩码
  private raysTex      : GPUTexture;   // 半分辨率: god rays 径向模糊

  // ── Pipeline state ─────────────────────────────────────────────────────────
  private fogBGL        : GPUBindGroupLayout;
  private occlusionBGL  : GPUBindGroupLayout;
  private raysBGL       : GPUBindGroupLayout;
  private compositeBGL  : GPUBindGroupLayout;

  private fogPipeline       : GPURenderPipeline;
  private occlusionPipeline : GPURenderPipeline;
  private raysPipeline      : GPURenderPipeline;
  private compositePipeline : GPURenderPipeline;

  // ── Runtime state ──────────────────────────────────────────────────────────
  private params : Required<EnvironmentFogParams>;

  private constructor(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ) {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;
    this.params = { ...DEFAULTS };

    const halfW = Math.max(1, width  >> 1);
    const halfH = Math.max(1, height >> 1);

    // ── Intermediate textures ────────────────────────────────────────────────
    this.fogTex       = makeRT(device, width,  height, format, 'env-fog-fogTex');
    this.occlusionTex = makeRT(device, halfW,  halfH,  format, 'env-fog-occlusionTex');
    this.raysTex      = makeRT(device, halfW,  halfH,  format, 'env-fog-raysTex');

    // ── Uniform buffer ───────────────────────────────────────────────────────
    this.uniformBuf = device.createBuffer({
      label : 'env-fog-uniforms',
      size  : UNIFORM_SIZE,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Bilinear sampler ─────────────────────────────────────────────────────
    this.sampler = device.createSampler({
      label        : 'env-fog-sampler',
      magFilter    : 'linear',
      minFilter    : 'linear',
      addressModeU : 'clamp-to-edge',
      addressModeV : 'clamp-to-edge',
    });

    // ── Bind group layouts ───────────────────────────────────────────────────
    // Pass 1 (depth fog):    uniform + sampler + sceneTex + depthTex
    this.fogBGL       = makeBGL(device, 'env-fog-fog-bgl',       2);
    // Pass 2 (occlusion):    uniform + sampler + sceneTex + depthTex
    this.occlusionBGL = makeBGL(device, 'env-fog-occ-bgl',       2);
    // Pass 3 (god rays):     uniform + sampler + occTex + fogTex
    this.raysBGL      = makeBGL(device, 'env-fog-rays-bgl',      2);
    // Pass 4 (composite):    uniform + sampler + fogTex + raysTex
    this.compositeBGL = makeBGL(device, 'env-fog-composite-bgl', 2);

    // ── Render pipelines ─────────────────────────────────────────────────────
    this.fogPipeline       = makePipeline(device, 'env-fog-fog-pipeline',
      WGSL_DEPTH_FOG, 'vs_main', 'fs_depthFog',  this.fogBGL,       format);

    this.occlusionPipeline = makePipeline(device, 'env-fog-occ-pipeline',
      WGSL_OCCLUSION,  'vs_main', 'fs_occlusion', this.occlusionBGL, format);

    this.raysPipeline      = makePipeline(device, 'env-fog-rays-pipeline',
      WGSL_GOD_RAYS,   'vs_main', 'fs_godRays',   this.raysBGL,      format);

    this.compositePipeline = makePipeline(device, 'env-fog-composite-pipeline',
      WGSL_COMPOSITE,  'vs_main', 'fs_composite',  this.compositeBGL, format);

    // Sync initial params
    this._uploadUniforms();
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Async factory (mirrors ATVolumetricLight.create() convention).
   *
   * @param device - WebGPU device.
   * @param format - Swapchain / render target texture format.
   * @param width  - Viewport width in pixels.
   * @param height - Viewport height in pixels.
   */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<EnvironmentFog> {
    return new EnvironmentFog(device, format, width, height);
  }

  // ── Parameter API ──────────────────────────────────────────────────────────

  /**
   * Update parameters. Partial — unspecified keys keep current values.
   * Writes updated uniforms to GPU immediately.
   */
  setParams(p: EnvironmentFogParams): void {
    Object.assign(this.params, p);
    this._uploadUniforms();
  }

  /** Returns current resolved parameters (all fields guaranteed). */
  get currentParams(): Readonly<Required<EnvironmentFogParams>> {
    return { ...this.params };
  }

  // ── Resize ─────────────────────────────────────────────────────────────────

  /**
   * Recreate intermediate textures on viewport resize.
   * Must be called when the canvas dimensions change.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;

    this.fogTex.destroy();
    this.occlusionTex.destroy();
    this.raysTex.destroy();

    const halfW = Math.max(1, width  >> 1);
    const halfH = Math.max(1, height >> 1);

    this.fogTex       = makeRT(this.device, width, height,  this.format, 'env-fog-fogTex');
    this.occlusionTex = makeRT(this.device, halfW, halfH,   this.format, 'env-fog-occlusionTex');
    this.raysTex      = makeRT(this.device, halfW, halfH,   this.format, 'env-fog-raysTex');

    this.width  = width;
    this.height = height;
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Execute the full 4-pass depth fog + god rays pipeline.
   *
   * @param encoder      - Active GPUCommandEncoder.
   * @param sceneTex     - Source scene texture (before fog).
   * @param depthFieldTex - Depth / field texture (single channel or R channel used).
   *                        For metaball scenes: field accumulation texture from
   *                        ParticleRenderer, where high values = near.
   *                        For standard Z-buffer: 0=near, 1=far.
   *                        Set params.depthInverted accordingly.
   * @param dstView      - Destination GPUTextureView (e.g. swapchain).
   */
  render(
    encoder       : GPUCommandEncoder,
    sceneTex      : GPUTexture,
    depthFieldTex : GPUTexture,
    dstView       : GPUTextureView,
  ): void {
    const sceneView     = sceneTex.createView();
    const depthView     = depthFieldTex.createView();
    const fogView       = this.fogTex.createView();
    const occlusionView = this.occlusionTex.createView();
    const raysView      = this.raysTex.createView();

    // ── Pass 1: Depth Fog ────────────────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'env-fog-fog-bg',
        layout : this.fogBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
          { binding: 3, resource: depthView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'env-fog-fog-pass',
        colorAttachments: [{
          view      : fogView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(this.fogPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 2: Occlusion Extract (半分辨率) ─────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'env-fog-occ-bg',
        layout : this.occlusionBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: sceneView },
          { binding: 3, resource: depthView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'env-fog-occ-pass',
        colorAttachments: [{
          view      : occlusionView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.occlusionPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 3: God Ray Radial Blur (半分辨率) ───────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'env-fog-rays-bg',
        layout : this.raysBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: occlusionView },
          { binding: 3, resource: fogView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'env-fog-rays-pass',
        colorAttachments: [{
          view      : raysView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.raysPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // ── Pass 4: Fog-Rays Composite ───────────────────────────────────────────
    {
      const bg = this.device.createBindGroup({
        label  : 'env-fog-composite-bg',
        layout : this.compositeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: fogView },
          { binding: 3, resource: raysView },
        ],
      });
      const pass = encoder.beginRenderPass({
        label: 'env-fog-composite-pass',
        colorAttachments: [{
          view      : dstView,
          loadOp    : 'clear',
          storeOp   : 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
  }

  // ── Resource management ────────────────────────────────────────────────────

  /** Release all GPU resources. The instance must not be used after this. */
  destroy(): void {
    this.uniformBuf.destroy();
    this.fogTex.destroy();
    this.occlusionTex.destroy();
    this.raysTex.destroy();
  }

  // ── Diagnostics ────────────────────────────────────────────────────────────

  /** Returns the four WGSL shader sources for inspection / hot-reload. */
  get wgslSources(): Readonly<Record<string, string>> {
    return {
      depthFog  : WGSL_DEPTH_FOG,
      occlusion : WGSL_OCCLUSION,
      godRays   : WGSL_GOD_RAYS,
      composite : WGSL_COMPOSITE,
    };
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Write current params to the GPU uniform buffer.
   * Layout matches EnvFogUniforms (28 × f32 = 112 bytes).
   */
  private _uploadUniforms(): void {
    const p   = this.params;
    const buf = new Float32Array(UNIFORM_FLOATS);

    // Row 0: fog basics
    buf[ 0] = p.fogDensity;
    buf[ 1] = p.fogStart;
    buf[ 2] = p.fogEnd;
    buf[ 3] = fogModeToFloat(p.fogMode);

    // Row 1: fog color + tint
    buf[ 4] = p.fogColor[0];
    buf[ 5] = p.fogColor[1];
    buf[ 6] = p.fogColor[2];
    buf[ 7] = p.fogLightTint;

    // Row 2: fog light color + depth inversion
    buf[ 8] = p.fogLightColor[0];
    buf[ 9] = p.fogLightColor[1];
    buf[10] = p.fogLightColor[2];
    buf[11] = p.depthInverted ? 1.0 : 0.0;

    // Row 3: height falloff + god rays basics
    buf[12] = p.heightFalloff;
    buf[13] = p.exposure;
    buf[14] = p.density;
    buf[15] = p.decay;

    // Row 4: god rays continued
    buf[16] = p.weight;
    buf[17] = p.occlusionThreshold;
    buf[18] = p.lightPos[0];
    buf[19] = p.lightPos[1];

    // Row 5: god rays + composite
    buf[20] = p.numSamples;
    buf[21] = p.fogRayBoost;
    buf[22] = p.raysScale;
    buf[23] = p.mieG;

    // Row 6: tonemap + padding
    buf[24] = p.tonemap ? 1.0 : 0.0;
    buf[25] = p.tonemapExposure;
    buf[26] = 0;  // _pad0
    buf[27] = 0;  // _pad1

    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — species-params integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an EnvironmentFog from species-level rendering parameters.
 * Bridges the SPH species parameter schema to the unified fog+rays API.
 *
 * @param device      - WebGPU device.
 * @param format      - Swapchain texture format.
 * @param width       - Viewport width.
 * @param height      - Viewport height.
 * @param speciesOpts - Optional species-specific overrides.
 */
export async function createEnvironmentFogForSpecies(
  device      : GPUDevice,
  format      : GPUTextureFormat,
  width       : number,
  height      : number,
  speciesOpts?: {
    fogDensity?     : number;
    fogColor?       : [number, number, number];
    fogMode?        : FogMode;
    lightExposure?  : number;
    lightDensity?   : number;
    lightPos?       : [number, number];
    raysScale?      : number;
    mieG?           : number;
    heightFalloff?  : number;
    depthInverted?  : boolean;
  },
): Promise<EnvironmentFog> {
  const envFog = await EnvironmentFog.create(device, format, width, height);
  if (speciesOpts) {
    envFog.setParams({
      fogDensity    : speciesOpts.fogDensity     ?? DEFAULTS.fogDensity,
      fogColor      : speciesOpts.fogColor       ?? DEFAULTS.fogColor,
      fogMode       : speciesOpts.fogMode        ?? DEFAULTS.fogMode,
      exposure      : speciesOpts.lightExposure  ?? DEFAULTS.exposure,
      density       : speciesOpts.lightDensity   ?? DEFAULTS.density,
      lightPos      : speciesOpts.lightPos       ?? DEFAULTS.lightPos,
      raysScale     : speciesOpts.raysScale      ?? DEFAULTS.raysScale,
      mieG          : speciesOpts.mieG           ?? DEFAULTS.mieG,
      heightFalloff : speciesOpts.heightFalloff  ?? DEFAULTS.heightFalloff,
      depthInverted : speciesOpts.depthInverted  ?? DEFAULTS.depthInverted,
    });
  }
  return envFog;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset factory helpers — 常用雾 + 光照组合
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ready-made fog+rays presets for common atmosphere registers.
 * Each mutates the passed EnvironmentFog in-place and returns it.
 */
export const EnvironmentFogPresets = {

  /**
   * deepOcean — 深海散射
   * 浓密蓝色 exp² 雾 + 微弱上方 god rays (模拟水面透射光)
   */
  deepOcean(fog: EnvironmentFog): EnvironmentFog {
    fog.setParams({
      fogMode      : 'exp2',
      fogDensity   : 0.55,
      fogColor     : [0.005, 0.02, 0.06],
      fogLightTint : 0.15,
      fogLightColor: [0.2, 0.5, 0.7],
      heightFalloff: 1.2,
      exposure     : 0.4,
      density      : 0.18,
      lightPos     : [0.5, 0.0],
      numSamples   : 8,
      raysScale    : 0.6,
      mieG         : 0.15,
      fogRayBoost  : 0.8,
      tonemap      : true,
    });
    return fog;
  },

  /**
   * bioLabHaze — 生化实验室薄雾
   * 轻柔绿色薄雾 + 明亮 god rays (与 environment-fx.ts bioLabClassic 搭配)
   */
  bioLabHaze(fog: EnvironmentFog): EnvironmentFog {
    fog.setParams({
      fogMode      : 'exp',
      fogDensity   : 0.2,
      fogColor     : [0.01, 0.03, 0.02],
      fogLightTint : 0.25,
      fogLightColor: [0.3, 0.9, 0.5],
      heightFalloff: 0.0,
      exposure     : 0.86,
      density      : 0.22,
      lightPos     : [0.5, 0.05],
      numSamples   : 6,
      raysScale    : 1.0,
      mieG         : 0.25,
      fogRayBoost  : 0.5,
      tonemap      : true,
    });
    return fog;
  },

  /**
   * sunriseDawn — 日出晨雾
   * 温暖橙金色调 + 强烈 god rays (太阳刚升起)
   */
  sunriseDawn(fog: EnvironmentFog): EnvironmentFog {
    fog.setParams({
      fogMode      : 'exp2',
      fogDensity   : 0.3,
      fogColor     : [0.04, 0.025, 0.015],
      fogLightTint : 0.7,
      fogLightColor: [1.0, 0.65, 0.25],
      heightFalloff: 0.8,
      exposure     : 1.1,
      density      : 0.28,
      lightPos     : [0.5, 0.15],
      numSamples   : 10,
      raysScale    : 1.4,
      mieG         : 0.35,
      fogRayBoost  : 0.6,
      tonemap      : true,
      tonemapExposure: 1.2,
    });
    return fog;
  },

  /**
   * cryoVault — 低温冷冻舱
   * 冰蓝色薄雾 + 冷色 god rays (与 environment-fx.ts cryogenics 搭配)
   */
  cryoVault(fog: EnvironmentFog): EnvironmentFog {
    fog.setParams({
      fogMode      : 'exp',
      fogDensity   : 0.25,
      fogColor     : [0.015, 0.025, 0.05],
      fogLightTint : 0.2,
      fogLightColor: [0.5, 0.8, 1.0],
      heightFalloff: 0.3,
      exposure     : 0.7,
      density      : 0.2,
      lightPos     : [0.5, 0.02],
      numSamples   : 6,
      raysScale    : 0.8,
      mieG         : 0.2,
      fogRayBoost  : 0.4,
      tonemap      : true,
    });
    return fog;
  },

  /**
   * voidAbyss — 虚空深渊
   * 极浓黑雾 + 微弱红色 god rays (恐怖/深渊氛围)
   */
  voidAbyss(fog: EnvironmentFog): EnvironmentFog {
    fog.setParams({
      fogMode      : 'exp2',
      fogDensity   : 0.7,
      fogColor     : [0.008, 0.003, 0.003],
      fogLightTint : 0.4,
      fogLightColor: [0.8, 0.15, 0.05],
      heightFalloff: 0.5,
      exposure     : 0.5,
      density      : 0.15,
      lightPos     : [0.5, 0.5],
      numSamples   : 12,
      raysScale    : 0.5,
      mieG         : 0.4,
      fogRayBoost  : 1.0,
      tonemap      : true,
      tonemapExposure: 0.8,
    });
    return fog;
  },

} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Re-export WGSL fragments — other shaders may embed them
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL source fragments for embedding in other shader modules. */
export const ENVIRONMENT_FOG_WGSL = {
  /** Shared uniform struct (EnvFogUniforms). */
  uniforms    : WGSL_UNIFORMS,
  /** Full-screen triangle vertex shader. */
  fullscreenVs: WGSL_FULLSCREEN_VERT,
  /** Depth fog fragment shader. */
  depthFog    : WGSL_DEPTH_FOG,
  /** Occlusion extract fragment shader. */
  occlusion   : WGSL_OCCLUSION,
  /** God ray radial blur fragment shader. */
  godRays     : WGSL_GOD_RAYS,
  /** Fog-rays composite fragment shader (Mie + ACES). */
  composite   : WGSL_COMPOSITE,
} as const;
