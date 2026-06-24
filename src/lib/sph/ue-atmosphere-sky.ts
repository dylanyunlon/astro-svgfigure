/**
 * ue-atmosphere-sky.ts — M849b: UE5 SkyAtmosphere Port
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 SkyAtmosphere 渲染系统到 WebGPU/WGSL。
 * Cell 世界大气层背景渲染，提供物理准确的天空颜色、太阳盘及大气透视效果。
 *
 * 原始来源:
 *   upstream/unreal-renderer-ue5/Shaders-Private/SkyAtmosphere.usf        (1954行)
 *   upstream/unreal-renderer-ue5/Shaders-Private/SkyAtmosphereCommon.ush  (364行)
 *   upstream/unreal-renderer-ue5/Renderer-Private/SkyAtmosphereRendering.h (261行)
 *   upstream/unreal-renderer-ue5/Renderer-Private/SkyAtmosphereRendering.cpp (2407行)
 *
 * 四大核心渲染通道（忠实移植 UE5 结构）:
 *
 *   1. Transmittance LUT (256×64)
 *      沿大气层光路积分 Beer-Lambert 消光，输出 sqrt 编码透射率。
 *      对应 UE5: RenderTransmittanceLutCS
 *      UVMapping: (height↔Xr, zenithCosAngle↔Xmu) Bruneton 2017 参数化
 *
 *   2. Multi-Scattering LUT (32×32)
 *      离线预计算多重散射贡献（基于无限级数收敛 L/(1-MultiScatAs1)）。
 *      对应 UE5: RenderMultiScatteredLuminanceLutCS
 *      两方向近似：±Z 方向各积分一次，均分球面立体角 4π
 *
 *   3. Sky View LUT (192×108)
 *      以相机为中心的低分辨率天空全图（纬度/经度参数化，非线性分布近地平线）。
 *      包含单次散射 + 多重散射近似 + 可选太阳盘。
 *      对应 UE5: RenderSkyViewLutCS + FastSky 分支
 *
 *   4. Camera Aerial Perspective Volume (32×32×16)
 *      相机空间 3D 体积纹理，存储每个 froxel 的内散射亮度+透射率。
 *      深度轴平方分布（非线性，近端密集），用于给不透明物体叠加大气透视。
 *      对应 UE5: RenderCameraAerialPerspectiveVolumeCS
 *
 *   5. Sky Render Pass
 *      全屏天空背景合成。优先从 SkyViewLUT 采样（FastSky 路径），
 *      太阳盘通过 Transmittance LUT 计算（软边渐变，避免 TAA 闪烁）。
 *      对应 UE5: RenderSkyAtmosphereRayMarchingPS
 *
 * 大气物理模型（对应 UE5 FAtmosphereUniformShaderParameters）:
 * ─────────────────────────────────────────────────────────────────────────────
 *   Rayleigh 散射：高斯指数密度分布（scale height 8km），λ⁻⁴ 波长依赖
 *   Mie 散射：指数密度分布（scale height 1.2km），Henyey-Greenstein 相位函数
 *   Ozone 吸收：双层线性剖面（峰值在 25km），吸收无散射
 *   地面反照率：Lambertian，可配置
 *   星球半径：6360km（底部），大气顶：6460km（默认 100km 大气厚度）
 *
 * 算法流程
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Transmittance LUT ──────────────────────────────────────────┐
 *   │  CS 256×64 线程: 沿视线积分 OpticalDepth → exp(-OD) → sqrt 编码        │
 *   │  采样数: 40 步（固定）                                                  │
 *   │  输出: transmittanceLUT (rgba16float, w=256, h=64)                     │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │
 *               ▼
 *   ┌─ Pass 1 ── Multi-Scattering LUT ───────────────────────────────────────┐
 *   │  CS 32×32: 两方向各积分 → MultiScatAs1 + InScatterLum                  │
 *   │  级数求和: L_ms = L1 / (1 - MultiScatAs1)                              │
 *   │  输出: multiScatLUT (rgba16float, 32×32)                               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │
 *               ▼
 *   ┌─ Pass 2 ── Sky View LUT ────────────────────────────────────────────────┐
 *   │  CS 192×108: 纬度/经度非线性参数化                                     │
 *   │  每线程: 射线步进 (变步数 4-14) + 多重散射 + Mie/Rayleigh 相位函数     │
 *   │  输出: skyViewLUT (rgba16float, 192×108) [rgb=亮度, a=transmittance]    │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │
 *               ▼
 *   ┌─ Pass 3 ── Aerial Perspective Volume ──────────────────────────────────┐
 *   │  CS 32×32×16: 3D froxel 积分                                           │
 *   │  Z 轴: 平方分布深度切片 [0, AP_KM_PER_SLICE*16] km                    │
 *   │  输出: apVolume (rgba16float, 32×32×16) [rgb=内散射, a=透射率]         │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │
 *               ▼
 *   ┌─ Pass 4 ── Sky Render ──────────────────────────────────────────────────┐
 *   │  FS: 从 SkyViewLUT 采样背景天空 + 太阳盘（Transmittance×盘亮度×软边）  │
 *   │  深度测试: DeviceZ==far 的像素才渲染天空背景                             │
 *   │  输出: 合并到场景颜色缓冲 (rgba16float)                                 │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * 与 Cell PubSub Loop 的集成
 * ─────────────────────────────────────────────────────────────────────────────
 * • UEAtmosphereSky.setSunDirection(dir) — 每帧更新太阳方向（影响所有LUT）
 * • UEAtmosphereSky.setViewData(pos, dir, invProj) — 注入相机信息
 * • UEAtmosphereSky.render(encoder, sceneColor, depthTex) — 执行完整管线
 * • LUT 惰性重建：仅当大气参数或太阳方向改变时重新计算 Transmittance/MultiScat LUT
 *
 * 上游参考
 * ─────────────────────────────────────────────────────────────────────────────
 *   SkyAtmosphere.usf             — 所有 CS/PS 着色器逻辑
 *   SkyAtmosphereCommon.ush       — LUT UV 参数化、GetAerialPerspective、Medium采样
 *   SkyAtmosphereRendering.h      — FAtmosphereUniformShaderParameters 结构
 *   SkyAtmosphereRendering.cpp    — LUT 尺寸、格式、管线创建
 *   ParticipatingMediaCommon.ush  — Rayleigh/Mie 相位函数（RayleighPhase, HenyeyGreensteins）
 *
 * Research: xiaodi #M849b — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────




// ─────────────────────────────────────────────────────────────────────────────
// Public Types & Configuration
// ─────────────────────────────────────────────────────────────────────────────


import type { CellSpecies }     from './cell-material-system';
import type { PhysicsUniforms } from './physics-uniform-bridge';

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * 大气层物理参数 — 对应 UE5 FAtmosphereUniformShaderParameters
// [orphan-precise]  * 默认值基于地球大气 (Bruneton 2017 / UE5 默认组件设置)
// [orphan-precise]  */
=======
/**
 * 大气层物理参数 — 对应 UE5 FAtmosphereUniformShaderParameters
 * 默认值基于地球大气 (Bruneton 2017 / UE5 默认组件设置)
 */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface AtmosphereParams {
  /** 多重散射系数乘数。@default 1 */
  multiScatteringFactor: number;
  /** 星球底部半径 (km)。@default 6360 */
  bottomRadiusKm: number;
  /** 大气顶部半径 (km)。@default 6460 */
  topRadiusKm: number;

  // ── Rayleigh 散射 ──────────────────────────────────────────────────────────
  /** Rayleigh 密度指数衰减系数 (1/km, 负值)。@default -0.125 (= -1/8km) */
  rayleighDensityExpScale: number;
  /** Rayleigh 散射系数 [R,G,B] (1/km)。@default [0.005802, 0.013558, 0.033100] */
  rayleighScattering: [number, number, number];

  // ── Mie 散射 ──────────────────────────────────────────────────────────────
  /** Mie 密度指数衰减系数 (1/km, 负值)。@default -0.8333 (= -1/1.2km) */
  mieDensityExpScale: number;
  /** Mie 散射系数 (1/km)。@default [0.003996, 0.003996, 0.003996] */
  mieScattering: [number, number, number];
  /** Mie 消光系数 (1/km, 散射+吸收)。@default [0.004440, 0.004440, 0.004440] */
  mieExtinction: [number, number, number];
  /** Mie 吸收系数 (1/km)。@default [0.000444, 0.000444, 0.000444] */
  mieAbsorption: [number, number, number];
  /** Henyey-Greenstein 不对称因子 g ∈ [0,1]。@default 0.8 */
  miePhaseG: number;

  // ── Ozone 吸收（双层线性剖面）──────────────────────────────────────────────
  /** 第0层宽度 (km)。@default 25 */
  absorptionDensity0LayerWidth: number;
  /** 第0层线性系数。@default 1/15 */
  absorptionDensity0LinearTerm: number;
  /** 第0层常数项。@default -2/3 */
  absorptionDensity0ConstantTerm: number;
  /** 第1层线性系数 (负数，向上衰减)。@default -1/15 */
  absorptionDensity1LinearTerm: number;
  /** 第1层常数项。@default 8/3 */
  absorptionDensity1ConstantTerm: number;
  /** Ozone 消光系数 [R,G,B] (1/km)。@default [0.000650, 0.001881, 0.000085] */
  absorptionExtinction: [number, number, number];

  // ── 地面 ─────────────────────────────────────────────────────────────────
  /** 地面反照率 [R,G,B]。@default [0.1, 0.1, 0.1] */
  groundAlbedo: [number, number, number];
}

/** 太阳/方向光参数 */
export interface AtmosphereSunParams {
  /** 太阳方向（世界空间，单位向量，指向太阳）。@default [0.0, 0.0, 1.0] */
  direction: [number, number, number];
  /** 太阳外层空间照度 [R,G,B] (kLux)。@default [1.0, 1.0, 1.0] */
  illuminanceOuterSpace: [number, number, number];
  /** 太阳盘张角的半角余弦（越接近1越小）。@default 0.9999747 (~0.27°) */
  discCosHalfApexAngle: number;
  /** 太阳盘亮度 (nit)。@default 1.6e9 */
  discLuminance: number;
}

/** 相机参数 */
export interface AtmosphereCameraParams {
  /** 相机世界位置 (cm)。UE5 使用翻译世界坐标（translated world space） */
  worldPos: [number, number, number];
  /** 星球中心世界位置 (cm)。 */
  planetCenter: [number, number, number];
  /** 视图方向矩阵（3x3, row-major, 用于 SkyView LUT 参考系） */
  viewMatrix: Float32Array;
  /** 逆投影矩阵（4x4, row-major） */
  invProjMatrix: Float32Array;
  /** 视口宽度 (px) */
  width: number;
  /** 视口高度 (px) */
  height: number;
}

/** UEAtmosphereSky 完整配置 */
export interface UEAtmosphereSkyConfig {
  /** 大气物理参数 */
  atmosphere: AtmosphereParams;
  /** 太阳参数 */
  sun: AtmosphereSunParams;
  /** 天空亮度因子（SkyLuminanceFactor）。@default 1 */
  skyLuminanceFactor: number;
  /** 天空+大气透视亮度因子。@default 1 */
  skyAndAerialPerspectiveLuminanceFactor: number;
  /** 大气透视视距缩放（AerialPerspectiveViewDistanceScale）。@default 1 */
  aerialPerspectiveViewDistanceScale: number;
  /** Aerial Perspective 起始深度 (km)。@default 0.1 */
  aerialPerspectiveStartDepthKm: number;
  /** 是否渲染太阳盘。@default true */
  renderSunDisk: boolean;
  /** 是否启用多重散射。@default true */
  enableMultiScattering: boolean;
  /** Transmittance LUT 采样数（步数）。@default 40 */
  transmittanceSampleCount: number;
  /** MultiScattering LUT 采样数。@default 20 */
  multiScatteringSampleCount: number;
  /** SkyView LUT 最小采样数。@default 4 */
  fastSkySampleCountMin: number;
  /** SkyView LUT 最大采样数。@default 14 */
  fastSkySampleCountMax: number;
  /** Aerial Perspective 每切片采样数缩放。@default 1 */
  cameraAerialPerspectiveSampleCountPerSlice: number;
}

export const DEFAULT_ATMOSPHERE_PARAMS: AtmosphereParams = {
  multiScatteringFactor:           1.0,
  bottomRadiusKm:                  6360.0,
  topRadiusKm:                     6460.0,
  rayleighDensityExpScale:         -0.125,
  rayleighScattering:              [0.005802, 0.013558, 0.033100],
  mieDensityExpScale:              -0.8333333,
  mieScattering:                   [0.003996, 0.003996, 0.003996],
  mieExtinction:                   [0.004440, 0.004440, 0.004440],
  mieAbsorption:                   [0.000444, 0.000444, 0.000444],
  miePhaseG:                       0.8,
  absorptionDensity0LayerWidth:    25.0,
  absorptionDensity0LinearTerm:    1.0 / 15.0,
  absorptionDensity0ConstantTerm:  -2.0 / 3.0,
  absorptionDensity1LinearTerm:    -1.0 / 15.0,
  absorptionDensity1ConstantTerm:  8.0 / 3.0,
  absorptionExtinction:            [0.000650, 0.001881, 0.000085],
  groundAlbedo:                    [0.1, 0.1, 0.1],
};

export const DEFAULT_SUN_PARAMS: AtmosphereSunParams = {
  direction:             [0.0, 0.3, 0.9535],
  illuminanceOuterSpace: [1.0, 1.0, 1.0],
  discCosHalfApexAngle:  0.9999747,
  discLuminance:         1.6e9,
};

export const DEFAULT_ATMOSPHERE_SKY_CONFIG: UEAtmosphereSkyConfig = {
  atmosphere:                              DEFAULT_ATMOSPHERE_PARAMS,
  sun:                                     DEFAULT_SUN_PARAMS,
  skyLuminanceFactor:                      1.0,
  skyAndAerialPerspectiveLuminanceFactor:  1.0,
  aerialPerspectiveViewDistanceScale:      1.0,
  aerialPerspectiveStartDepthKm:           0.1,
  renderSunDisk:                           true,
  enableMultiScattering:                   true,
  transmittanceSampleCount:                40,
  multiScatteringSampleCount:              20,
  fastSkySampleCountMin:                   4,
  fastSkySampleCountMax:                   14,
  cameraAerialPerspectiveSampleCountPerSlice: 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// LUT 尺寸常量（对应 UE5 SkyAtmosphereRendering.cpp）
// ─────────────────────────────────────────────────────────────────────────────

/** Transmittance LUT 宽度 */
const LUT_TRANSMITTANCE_W = 256;
/** Transmittance LUT 高度 */
const LUT_TRANSMITTANCE_H = 64;
/** Multi-Scattering LUT 尺寸 */
const LUT_MULTISCAT_W = 32;
const LUT_MULTISCAT_H = 32;
/** Sky View LUT 尺寸（宽:高 = 16:9 低分辨率）*/
const LUT_SKYVIEW_W = 192;
const LUT_SKYVIEW_H = 108;
/** Aerial Perspective Volume 尺寸 */
const AP_VOL_W    = 32;
const AP_VOL_H    = 32;
const AP_VOL_D    = 16;
/** 每切片深度 (km)，对应 UE5 AP_KM_PER_SLICE = 4.0 */
const AP_KM_PER_SLICE = 4.0;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader Source — 公共大气工具函数
// ─────────────────────────────────────────────────────────────────────────────

const ATMOSPHERE_COMMON_WGSL = /* wgsl */`

// ─── 常量 ──────────────────────────────────────────────────────────────────
const PI              : f32 = 3.14159265358979323846;
const CM_TO_SKY_UNIT  : f32 = 0.00001;      // cm → km
const SKY_UNIT_TO_CM  : f32 = 100000.0;     // km → cm
const PLANET_RADIUS_OFFSET : f32 = 0.001;   // 1m 安全偏移 (km)

// ─── 大气参数 UBO ──────────────────────────────────────────────────────────
struct AtmosphereUniforms {
  multiScatteringFactor          : f32,
  bottomRadiusKm                 : f32,
  topRadiusKm                    : f32,
  rayleighDensityExpScale        : f32,

  rayleighScatteringR            : f32,
  rayleighScatteringG            : f32,
  rayleighScatteringB            : f32,
  mieDensityExpScale             : f32,

  mieScatteringR                 : f32,
  mieScatteringG                 : f32,
  mieScatteringB                 : f32,
  mieExtinctionR                 : f32,

  mieExtinctionG                 : f32,
  mieExtinctionB                 : f32,
  mieAbsorptionR                 : f32,
  mieAbsorptionG                 : f32,

  mieAbsorptionB                 : f32,
  miePhaseG                      : f32,
  absorptionDensity0LayerWidth   : f32,
  absorptionDensity0LinearTerm   : f32,

  absorptionDensity0ConstantTerm : f32,
  absorptionDensity1LinearTerm   : f32,
  absorptionDensity1ConstantTerm : f32,
  absorptionExtinctionR          : f32,

  absorptionExtinctionG          : f32,
  absorptionExtinctionB          : f32,
  groundAlbedoR                  : f32,
  groundAlbedoG                  : f32,

  groundAlbedoB                  : f32,
  _pad0                          : f32,
  _pad1                          : f32,
  _pad2                          : f32,
}

// ─── 射线-球体求交 ─────────────────────────────────────────────────────────
// 返回两个 t (可为负), sol.x <= sol.y
fn ray_intersect_sphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> vec2f {
  let oc = ro - center;
  let b  = dot(oc, rd);
  let c  = dot(oc, oc) - radius * radius;
  let discriminant = b * b - c;
  if (discriminant < 0.0) { return vec2f(-1.0, -1.0); }
  let sqrtD = sqrt(discriminant);
  return vec2f(-b - sqrtD, -b + sqrtD);
}

fn ray_sphere_nearest(ro: vec3f, rd: vec3f, radius: f32) -> f32 {
  let sol = ray_intersect_sphere(ro, rd, vec3f(0.0), radius);
  if (sol.x < 0.0 && sol.y < 0.0) { return -1.0; }
  if (sol.x < 0.0) { return max(0.0, sol.y); }
  if (sol.y < 0.0) { return max(0.0, sol.x); }
  return max(0.0, min(sol.x, sol.y));
}

// ─── Transmittance LUT UV 参数化（Bruneton 2017）─────────────────────────────
// 对应 UE5: getTransmittanceLutUvs / fromTransmittanceLutUVs
fn transmittance_params_to_uv(
  viewHeight      : f32,
  viewZenithCosAngle : f32,
  bottomR         : f32,
  topR            : f32,
) -> vec2f {
  let H    = sqrt(max(0.0, topR * topR - bottomR * bottomR));
  let rho  = sqrt(max(0.0, viewHeight * viewHeight - bottomR * bottomR));
  let disc = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0) + topR * topR;
  let D    = max(0.0, -viewHeight * viewZenithCosAngle + sqrt(disc));
  let dmin = topR - viewHeight;
  let dmax = rho + H;
  let xmu  = (D - dmin) / (dmax - dmin);
  let xr   = rho / H;
  return vec2f(xmu, xr);
}

fn transmittance_uv_to_params(uv: vec2f, bottomR: f32, topR: f32) -> vec2f {
  // returns vec2f(viewHeight, viewZenithCosAngle)
  let H   = sqrt(topR * topR - bottomR * bottomR);
  let rho = H * uv.y;
  let viewHeight = sqrt(rho * rho + bottomR * bottomR);
  let dmin = topR - viewHeight;
  let dmax = rho + H;
  let D    = dmin + uv.x * (dmax - dmin);
  var cosAngle = 1.0;
  if (D != 0.0) {
    cosAngle = clamp((H * H - rho * rho - D * D) / (2.0 * viewHeight * D), -1.0, 1.0);
  }
  return vec2f(viewHeight, cosAngle);
}

// ─── Transmittance LUT 采样（带 sqrt 解码）────────────────────────────────
fn sample_transmittance_lut(
  tex           : texture_2d<f32>,
  smp           : sampler,
  viewHeight    : f32,
  zenithCosAngle: f32,
  bottomR       : f32,
  topR          : f32,
) -> vec3f {
  let uv = transmittance_params_to_uv(viewHeight, zenithCosAngle, bottomR, topR);
  let encoded = textureSampleLevel(tex, smp, uv, 0.0).rgb;
  return encoded * encoded;  // decode: sqrt 编码 → 平方解码
}

// ─── MultiScattering LUT 采样 ─────────────────────────────────────────────
fn sample_multiscat_lut(
  tex            : texture_2d<f32>,
  smp            : sampler,
  worldPos       : vec3f,
  zenithCosAngle : f32,
  bottomR        : f32,
  topR           : f32,
) -> vec3f {
  let uv = vec2f(
    clamp(zenithCosAngle * 0.5 + 0.5, 0.0, 1.0),
    clamp((length(worldPos) - bottomR) / (topR - bottomR), 0.0, 1.0),
  );
  return textureSampleLevel(tex, smp, uv, 0.0).rgb;
}

// ─── 大气介质采样 ─────────────────────────────────────────────────────────
// 对应 UE5: SampleAtmosphereMediumRGB
struct MediumSample {
  scatteringMie : vec3f,
  scatteringRay : vec3f,
  extinction    : vec3f,
  scattering    : vec3f,
  albedo        : vec3f,
}

fn sample_atmosphere_medium(worldPos: vec3f, atm: AtmosphereUniforms) -> MediumSample {
  let sampleHeight = max(0.0, length(worldPos) - atm.bottomRadiusKm);

  let densityMie = exp(atm.mieDensityExpScale * sampleHeight);
  let densityRay = exp(atm.rayleighDensityExpScale * sampleHeight);

  var densityOzo : f32;
  if (sampleHeight < atm.absorptionDensity0LayerWidth) {
    densityOzo = clamp(atm.absorptionDensity0LinearTerm * sampleHeight + atm.absorptionDensity0ConstantTerm, 0.0, 1.0);
  } else {
    densityOzo = clamp(atm.absorptionDensity1LinearTerm * sampleHeight + atm.absorptionDensity1ConstantTerm, 0.0, 1.0);
  }

  var s: MediumSample;
  let mieS    = vec3f(atm.mieScatteringR, atm.mieScatteringG, atm.mieScatteringB);
  let mieExt  = vec3f(atm.mieExtinctionR, atm.mieExtinctionG, atm.mieExtinctionB);
  let rayS    = vec3f(atm.rayleighScatteringR, atm.rayleighScatteringG, atm.rayleighScatteringB);
  let ozoExt  = vec3f(atm.absorptionExtinctionR, atm.absorptionExtinctionG, atm.absorptionExtinctionB);

  s.scatteringMie = densityMie * mieS;
  s.scatteringRay = densityRay * rayS;
  s.scattering    = s.scatteringMie + s.scatteringRay;
  let extinctionMie = densityMie * mieExt;
  let extinctionRay = densityRay * rayS;
  let extinctionOzo = densityOzo * ozoExt;
  s.extinction    = extinctionMie + extinctionRay + extinctionOzo;
  s.albedo        = s.scattering / max(vec3f(0.001), s.extinction);
  return s;
}

// ─── 相位函数 ────────────────────────────────────────────────────────────────
// Rayleigh 相位函数（对应 UE5 RayleighPhase）
fn rayleigh_phase(cosTheta: f32) -> f32 {
  return 3.0 * (1.0 + cosTheta * cosTheta) / (16.0 * PI);
}

// Henyey-Greenstein Mie 相位函数（对应 UE5 HenyeyGreensteinPhase）
fn henyey_greenstein_phase(cosTheta: f32, g: f32) -> f32 {
  let g2    = g * g;
  let denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (4.0 * PI * pow(abs(denom), 1.5) + 1e-7);
}

// ─── 大气顶部移入（确保射线起点在大气内）──────────────────────────────────
// 返回 false 表示射线不与大气相交
fn move_to_top_atmosphere(worldPos: ptr<function, vec3f>, worldDir: vec3f, topR: f32) -> bool {
  let viewHeight = length(*worldPos);
  if (viewHeight > topR) {
    let tTop = ray_sphere_nearest(*worldPos, worldDir, topR);
    if (tTop >= 0.0) {
      let up = (*worldPos) / viewHeight;
      *worldPos = *worldPos + worldDir * tTop + up * (-PLANET_RADIUS_OFFSET);
      return true;
    } else {
      return false;
    }
  }
  return true;
}

// ─── SkyViewLUT 参数化 ─────────────────────────────────────────────────────
// 对应 UE5: SkyViewLutParamsToUv / UvToSkyViewLutParams
fn sky_view_lut_params_to_uv(
  intersectGround    : bool,
  viewZenithCosAngle : f32,
  viewHeight         : f32,
  bottomR            : f32,
  lutW               : f32,
  lutH               : f32,
) -> vec2f {
  let vHorizon    = sqrt(viewHeight * viewHeight - bottomR * bottomR);
  let cosBeta     = vHorizon / viewHeight;
  let beta        = acos(cosBeta);
  let zenithHorizAngle = PI - beta;
  let viewZenithAngle  = acos(clamp(viewZenithCosAngle, -1.0, 1.0));

  var uvY : f32;
  if (!intersectGround) {
    var coord = viewZenithAngle / zenithHorizAngle;
    coord = 1.0 - coord;
    coord = sqrt(coord);
    coord = 1.0 - coord;
    uvY = coord * 0.5;
  } else {
    var coord = (viewZenithAngle - zenithHorizAngle) / beta;
    coord = sqrt(coord);
    uvY = coord * 0.5 + 0.5;
  }
  // Sub-texel mapping
  let invW = 1.0 / lutW;
  let invH = 1.0 / lutH;
  let uvYFinal = (uvY + 0.5 * invH) * (lutH / (lutH + 1.0));
  return vec2f(0.5, uvYFinal); // 方位角在 render pass 填充
}

fn uv_to_sky_view_lut_dir(uv: vec2f, viewHeight: f32, bottomR: f32, lutW: f32, lutH: f32) -> vec3f {
  // 逆映射 sub-texel
  let invW = 1.0 / lutW;
  let invH = 1.0 / lutH;
  let uvAdj = vec2f(
    (uv.x - 0.5 * invW) * (lutW / (lutW - 1.0)),
    (uv.y - 0.5 * invH) * (lutH / (lutH - 1.0)),
  );

  let vHorizon         = sqrt(viewHeight * viewHeight - bottomR * bottomR);
  let cosBeta          = vHorizon / viewHeight;
  let beta             = acos(cosBeta);
  let zenithHorizAngle = PI - beta;

  var viewZenithAngle : f32;
  if (uvAdj.y < 0.5) {
    var coord = 2.0 * uvAdj.y;
    coord = 1.0 - coord;
    coord = coord * coord;
    coord = 1.0 - coord;
    viewZenithAngle = zenithHorizAngle * coord;
  } else {
    var coord = uvAdj.y * 2.0 - 1.0;
    coord = coord * coord;
    viewZenithAngle = zenithHorizAngle + beta * coord;
  }

  let cosVZA = cos(viewZenithAngle);
  let sinVZA = sqrt(max(0.0, 1.0 - cosVZA * cosVZA)) * select(-1.0, 1.0, viewZenithAngle > 0.0);
  let longAngle = uvAdj.x * 2.0 * PI;
  let cosLong   = cos(longAngle);
  let sinLong   = sqrt(max(0.0, 1.0 - cosLong * cosLong)) * select(-1.0, 1.0, longAngle <= PI);
  return vec3f(sinVZA * cosLong, sinVZA * sinLong, cosVZA);
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader: Pass 0 — Transmittance LUT
// 对应 UE5: RenderTransmittanceLutCS
// ─────────────────────────────────────────────────────────────────────────────

const TRANSMITTANCE_LUT_WGSL = ATMOSPHERE_COMMON_WGSL + /* wgsl */`

@group(0) @binding(0) var<uniform> atm : AtmosphereUniforms;
@group(0) @binding(1) var<uniform> lutParams : TransmittanceLutParams;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;

struct TransmittanceLutParams {
  sampleCount : f32,
  lutW        : f32,
  lutH        : f32,
  _pad        : f32,
}

@compute @workgroup_size(8, 8)
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3u) {
  let w = u32(${LUT_TRANSMITTANCE_W});
  let h = u32(${LUT_TRANSMITTANCE_H});
  if (gid.x >= w || gid.y >= h) { return; }

  let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(w), f32(h));
  let params = transmittance_uv_to_params(uv, atm.bottomRadiusKm, atm.topRadiusKm);
  let viewHeight      = params.x;
  let viewZenithCosA  = params.y;

  let worldPos = vec3f(0.0, 0.0, viewHeight);
  let worldDir = vec3f(0.0, sqrt(max(0.0, 1.0 - viewZenithCosA * viewZenithCosA)), viewZenithCosA);

  // 向大气顶部射线步进，积分光学深度
  // Identical to UE5 IntegrateSingleScatteredLuminance with MieRayPhase=false, Ground=false
  let tTop = ray_sphere_nearest(worldPos, worldDir, atm.topRadiusKm);
  if (tTop < 0.0) {
    textureStore(outputTex, gid.xy, vec4f(1.0, 1.0, 1.0, 1.0));
    return;
  }

  let sampleCount = lutParams.sampleCount;
  let dt          = tTop / sampleCount;
  var opticalDepth = vec3f(0.0);

  for (var i = 0.0; i < sampleCount; i += 1.0) {
    let t = tTop * (i + 0.5) / sampleCount;
    let P = worldPos + t * worldDir;
    let med = sample_atmosphere_medium(P, atm);
    opticalDepth += med.extinction * dt;
  }

  let transmittance = exp(-opticalDepth);
  // UE5: EncodeTransmittance = sqrt(t)
  let encoded = sqrt(clamp(transmittance, vec3f(0.0), vec3f(1.0)));
  textureStore(outputTex, gid.xy, vec4f(encoded, 1.0));
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader: Pass 1 — Multi-Scattering LUT
// 对应 UE5: RenderMultiScatteredLuminanceLutCS (廉价近似：±Z 两方向)
// ─────────────────────────────────────────────────────────────────────────────

const MULTISCAT_LUT_WGSL = ATMOSPHERE_COMMON_WGSL + /* wgsl */`

@group(0) @binding(0) var<uniform> atm        : AtmosphereUniforms;
@group(0) @binding(1) var<uniform> msParams   : MultiScatLutParams;
@group(0) @binding(2) var          transLut   : texture_2d<f32>;
@group(0) @binding(3) var          lutSampler : sampler;
@group(0) @binding(4) var          outputTex  : texture_storage_2d<rgba16float, write>;

struct MultiScatLutParams {
  sampleCount : f32,
  lutW        : f32,
  lutH        : f32,
  _pad        : f32,
}

// 积分一条射线的散射/光学深度（只返回 L 和 MultiScatAs1）
struct ScatterResult {
  L            : vec3f,
  multiScatAs1 : vec3f,
}

fn integrate_scattering_for_ms(
  worldPos   : vec3f,
  worldDir   : vec3f,
  lightDir   : vec3f,
  sampleCount: f32,
  atm        : AtmosphereUniforms,
) -> ScatterResult {
  var res: ScatterResult;
  res.L            = vec3f(0.0);
  res.multiScatAs1 = vec3f(0.0);

  let solB = ray_intersect_sphere(worldPos, worldDir, vec3f(0.0), atm.bottomRadiusKm);
  let solT = ray_intersect_sphere(worldPos, worldDir, vec3f(0.0), atm.topRadiusKm);
  if (all(solT < vec2f(0.0))) { return res; }

  var tMax : f32;
  if (all(solB < vec2f(0.0))) {
    tMax = max(solT.x, solT.y);
  } else {
    tMax = max(0.0, min(solB.x, solB.y));
  }

  let dt          = tMax / sampleCount;
  let uniformPhase = 1.0 / (4.0 * PI);
  var throughput  = vec3f(1.0);

  for (var i = 0.0; i < sampleCount; i += 1.0) {
    let t = tMax * (i + 0.3) / sampleCount;
    let P = worldPos + t * worldDir;
    let pHeight = length(P);
    let med = sample_atmosphere_medium(P, atm);

    let sampleOD    = med.extinction * dt;
    let sampleT     = exp(-sampleOD);

    // MultiScatAs1 — isotropic phase, integrate over sphere
    res.multiScatAs1 += throughput * med.scattering * dt;

    // 太阳透射率
    let upVec  = P / pHeight;
    let lightZenithCosA = dot(lightDir, upVec);
    let transToLight = sample_transmittance_lut(
      transLut, lutSampler, pHeight, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm
    );

    // 行星阴影
    let tPlanet = ray_sphere_nearest(P, lightDir, atm.bottomRadiusKm);
    let shadow  = select(0.0, 1.0, tPlanet < 0.0);

    // 单次散射 luminance（白色光源，各向同性相位函数）
    let S    = shadow * transToLight * med.scattering * uniformPhase;
    let sint = (S - S * sampleT) / max(med.extinction, vec3f(1e-9));
    res.L       += throughput * sint;
    throughput  *= sampleT;
  }

  // 地面反弹
  if (all(solB >= vec2f(0.0))) {
    let P       = worldPos + tMax * worldDir;
    let pHeight = length(P);
    let upVec   = P / pHeight;
    let lightZenithCosA = dot(lightDir, upVec);
    let transToLight = sample_transmittance_lut(
      transLut, lutSampler, pHeight, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm
    );
    let nDotL = max(0.0, dot(upVec, lightDir));
    let groundAlbedo = vec3f(atm.groundAlbedoR, atm.groundAlbedoG, atm.groundAlbedoB);
    res.L += transToLight * throughput * nDotL * groundAlbedo / PI;
  }

  return res;
}

@compute @workgroup_size(8, 8)
fn cs_multiscat(@builtin(global_invocation_id) gid: vec3u) {
  let w = u32(${LUT_MULTISCAT_W});
  let h = u32(${LUT_MULTISCAT_H});
  if (gid.x >= w || gid.y >= h) { return; }

  let pixPos = vec2f(gid.xy) + 0.5;
  let cosLightZenithAngle = (pixPos.x / f32(w)) * 2.0 - 1.0;
  let lightDir = vec3f(0.0, sqrt(clamp(1.0 - cosLightZenithAngle * cosLightZenithAngle, 0.0, 1.0)), cosLightZenithAngle);
  let viewHeight = atm.bottomRadiusKm + (pixPos.y / f32(h)) * (atm.topRadiusKm - atm.bottomRadiusKm);
  let worldPos   = vec3f(0.0, 0.0, viewHeight);

  // 两方向近似（±Z）
  let r0 = integrate_scattering_for_ms(worldPos,  vec3f(0, 0,  1), lightDir, msParams.sampleCount, atm);
  let r1 = integrate_scattering_for_ms(worldPos,  vec3f(0, 0, -1), lightDir, msParams.sampleCount, atm);

  let sphereSolidAngle   = 4.0 * PI;
  let isotropicPhase     = 1.0 / sphereSolidAngle;
  let integratedIllum    = (sphereSolidAngle * 0.5) * (r0.L + r1.L);
  let multiScatAs1       = 0.5 * (r0.multiScatAs1 + r1.multiScatAs1);
  let inScatteredLum     = integratedIllum * isotropicPhase;

  // 级数求和: 1 + ms + ms^2 + ms^3 + ms^4 + ms^5 (对应 UE5 MULTI_SCATTERING_POWER_SERIE==0)
  let ms2 = multiScatAs1 * multiScatAs1;
  let L   = inScatteredLum * (1.0 + multiScatAs1 + ms2 + multiScatAs1 * ms2 + ms2 * ms2);
  let Lfinal = L * atm.multiScatteringFactor;

  textureStore(outputTex, gid.xy, vec4f(Lfinal, 0.0));
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader: Pass 2 — Sky View LUT
// 对应 UE5: RenderSkyViewLutCS
// ─────────────────────────────────────────────────────────────────────────────

const SKYVIEW_LUT_WGSL = ATMOSPHERE_COMMON_WGSL + /* wgsl */`

@group(0) @binding(0) var<uniform> atm       : AtmosphereUniforms;
@group(0) @binding(1) var<uniform> svParams  : SkyViewLutParams;
@group(0) @binding(2) var          transLut  : texture_2d<f32>;
@group(0) @binding(3) var          msLut     : texture_2d<f32>;
@group(0) @binding(4) var          lutSampler: sampler;
@group(0) @binding(5) var          outputTex : texture_storage_2d<rgba16float, write>;

struct SkyViewLutParams {
  // 相机高度（距星球中心 km）
  viewHeightKm               : f32,
  // 太阳方向（在 SkyView 本地坐标系，Z轴朝上）
  lightDirX                  : f32,
  lightDirY                  : f32,
  lightDirZ                  : f32,
  // 光照度
  lightIllumR                : f32,
  lightIllumG                : f32,
  lightIllumB                : f32,
  // 采样参数
  sampleCountMin             : f32,
  sampleCountMax             : f32,
  distToSampleCountMaxInvKm  : f32,
  skyLumFactor               : f32,
  enableMultiScat            : f32,
}

@compute @workgroup_size(8, 8)
fn cs_skyview(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(${LUT_SKYVIEW_W});
  let H = u32(${LUT_SKYVIEW_H});
  if (gid.x >= W || gid.y >= H) { return; }

  let fW = f32(W);
  let fH = f32(H);
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(fW, fH);

  let viewHeight = svParams.viewHeightKm;
  let worldPos   = vec3f(0.0, 0.0, viewHeight);
  let lightDir   = normalize(vec3f(svParams.lightDirX, svParams.lightDirY, svParams.lightDirZ));
  let lightIllum = vec3f(svParams.lightIllumR, svParams.lightIllumG, svParams.lightIllumB);

  // 从 UV 还原视线方向（SkyView 本地坐标系）
  let worldDir = uv_to_sky_view_lut_dir(uv, viewHeight, atm.bottomRadiusKm, fW, fH);

  // 判断是否与地面相交
  let intersectGround = ray_sphere_nearest(worldPos, worldDir, atm.bottomRadiusKm) >= 0.0;

  // 移动到大气顶
  var wp = worldPos;
  if (!move_to_top_atmosphere(&wp, worldDir, atm.topRadiusKm)) {
    textureStore(outputTex, gid.xy, vec4f(0.0));
    return;
  }

  // 自适应采样数
  let solB = ray_intersect_sphere(wp, worldDir, vec3f(0.0), atm.bottomRadiusKm);
  let solT = ray_intersect_sphere(wp, worldDir, vec3f(0.0), atm.topRadiusKm);
  var tMax : f32;
  if (all(solB < vec2f(0.0))) {
    tMax = max(solT.x, solT.y);
  } else {
    tMax = max(0.0, min(solB.x, solB.y));
  }

  let sampleCountF = clamp(
    mix(svParams.sampleCountMin, svParams.sampleCountMax, clamp(tMax * svParams.distToSampleCountMaxInvKm, 0.0, 1.0)),
    svParams.sampleCountMin, svParams.sampleCountMax
  );
  let sampleCount     = max(1.0, sampleCountF);
  let sampleCountFloor = floor(sampleCount);
  let tMaxFloor       = tMax * sampleCountFloor / sampleCount;
  let dt              = tMax / sampleCount;

  // 相位函数
  let cosTheta         = dot(lightDir, worldDir);
  let miePhase         = henyey_greenstein_phase(-cosTheta, atm.miePhaseG);
  let rayleighPhaseVal = rayleigh_phase(cosTheta);
  let uniformPhase     = 1.0 / (4.0 * PI);

  var L          = vec3f(0.0);
  var throughput = vec3f(1.0);
  let pixelNoise = 0.3;

  for (var i = 0.0; i < sampleCount; i += 1.0) {
    // 非线性样本分布（平方分布减少近端采样不足）
    var t0 = i / sampleCountFloor;
    var t1 = (i + 1.0) / sampleCountFloor;
    t0 = t0 * t0;
    t1 = t1 * t1;
    t0 = tMaxFloor * t0;
    t1 = select(tMaxFloor * t1, tMax, t1 > 1.0);
    let t  = t0 + (t1 - t0) * pixelNoise;
    let dti = t1 - t0;

    let P       = wp + t * worldDir;
    let pHeight = length(P);
    let med     = sample_atmosphere_medium(P, atm);

    let sampleOD = med.extinction * dti;
    let sampleTr = exp(-sampleOD);

    // 太阳透射 + 行星阴影
    let upVec           = P / pHeight;
    let lightZenithCosA = dot(lightDir, upVec);
    let transToLight    = sample_transmittance_lut(transLut, lutSampler, pHeight, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm);
    let tPlanet         = ray_sphere_nearest(P, lightDir, atm.bottomRadiusKm);
    let shadow          = select(0.0, 1.0, tPlanet < 0.0);

    // 相位 × 散射
    let miePhaseS     = med.scatteringMie * miePhase;
    let rayPhaseS     = med.scatteringRay * rayleighPhaseVal;
    let phaseScatter  = miePhaseS + rayPhaseS;

    // 多重散射
    var multiScatLum = vec3f(0.0);
    if (svParams.enableMultiScat > 0.5) {
      multiScatLum = sample_multiscat_lut(msLut, lutSampler, P, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm);
    }

    let S    = lightIllum * (shadow * transToLight * phaseScatter + multiScatLum * med.scattering);
    let sint = (S - S * sampleTr) / max(med.extinction, vec3f(1e-9));
    L          += throughput * sint;
    throughput *= sampleTr;
  }

  // 地面反弹
  if (intersectGround) {
    let P       = wp + tMax * worldDir;
    let pHeight = length(P);
    let upVec   = P / pHeight;
    let lightZenithCosA = dot(lightDir, upVec);
    let transToLight = sample_transmittance_lut(transLut, lutSampler, pHeight, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm);
    let nDotL = max(0.0, dot(upVec, lightDir));
    let groundAlbedo = vec3f(atm.groundAlbedoR, atm.groundAlbedoG, atm.groundAlbedoB);
    L += lightIllum * transToLight * throughput * nDotL * groundAlbedo / PI;
  }

  L *= svParams.skyLumFactor;

  let transmittance = dot(throughput, vec3f(1.0 / 3.0));
  textureStore(outputTex, gid.xy, vec4f(L, transmittance));
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader: Pass 3 — Aerial Perspective Volume
// 对应 UE5: RenderCameraAerialPerspectiveVolumeCS
// ─────────────────────────────────────────────────────────────────────────────

const AERIAL_PERSPECTIVE_WGSL = ATMOSPHERE_COMMON_WGSL + /* wgsl */`

@group(0) @binding(0) var<uniform> atm      : AtmosphereUniforms;
@group(0) @binding(1) var<uniform> apParams : AerialPerspectiveParams;
@group(0) @binding(2) var          transLut : texture_2d<f32>;
@group(0) @binding(3) var          msLut    : texture_2d<f32>;
@group(0) @binding(4) var          lutSmp   : sampler;
@group(0) @binding(5) var          outputVol: texture_storage_3d<rgba16float, write>;

struct AerialPerspectiveParams {
  // 相机位置 (km, 相对于星球中心)
  camPosX        : f32,
  camPosY        : f32,
  camPosZ        : f32,
  startDepthKm   : f32,
  // 太阳方向
  lightDirX      : f32,
  lightDirY      : f32,
  lightDirZ      : f32,
  // 光照度
  lightIllumR    : f32,
  lightIllumG    : f32,
  lightIllumB    : f32,
  sampleCountPerSlice : f32,
  kmPerSlice          : f32,
  // 逆投影 (简化: fov 参数)
  fovHalfTanX    : f32,
  fovHalfTanY    : f32,
  apViewDistScale: f32,
  enableMultiScat: f32,
}

@compute @workgroup_size(4, 4, 4)
fn cs_aerial_perspective(@builtin(global_invocation_id) gid: vec3u) {
  let volW = u32(${AP_VOL_W});
  let volH = u32(${AP_VOL_H});
  let volD = u32(${AP_VOL_D});
  if (gid.x >= volW || gid.y >= volH || gid.z >= volD) { return; }

  let fW = f32(volW); let fH = f32(volH); let fD = f32(volD);
  let uv  = (vec2f(gid.xy) + 0.5) / vec2f(fW, fH);

  // NDC → 世界方向（简化 frustum 重建）
  let ndcX    = uv.x * 2.0 - 1.0;
  let ndcY    = 1.0 - uv.y * 2.0;
  let worldDir = normalize(vec3f(ndcX * apParams.fovHalfTanX, ndcY * apParams.fovHalfTanY, 1.0));

  let camPos    = vec3f(apParams.camPosX, apParams.camPosY, apParams.camPosZ);
  let lightDir  = normalize(vec3f(apParams.lightDirX, apParams.lightDirY, apParams.lightDirZ));
  let lightIllum= vec3f(apParams.lightIllumR, apParams.lightIllumG, apParams.lightIllumB);

  // 深度切片（平方分布）
  let sliceF    = (f32(gid.z) + 0.5) / fD;
  let sliceSqrt = sliceF * sliceF * fD;
  let tMax      = sliceSqrt * apParams.kmPerSlice;

  var rayStart  = camPos + apParams.startDepthKm * worldDir;
  var startPos  = rayStart;

  if (!move_to_top_atmosphere(&startPos, worldDir, atm.topRadiusKm)) {
    textureStore(outputVol, gid, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let sampleCount = max(1.0, (f32(gid.z) + 1.0) * apParams.sampleCountPerSlice);
  let dt          = tMax / sampleCount;

  let cosTheta     = dot(lightDir, worldDir);
  let miePhaseVal  = henyey_greenstein_phase(-cosTheta, atm.miePhaseG);
  let rayPhaseVal  = rayleigh_phase(cosTheta);

  var L          = vec3f(0.0);
  var throughput = vec3f(1.0);
  let pixelNoise = 0.3;

  for (var i = 0.0; i < sampleCount; i += 1.0) {
    let t = tMax * (i + pixelNoise) / sampleCount;
    let P = startPos + t * worldDir;
    let pHeight = length(P);
    if (pHeight < atm.bottomRadiusKm) { break; }

    let med      = sample_atmosphere_medium(P, atm);
    let sampleOD = med.extinction * dt * apParams.apViewDistScale;
    let sampleTr = exp(-sampleOD);

    let upVec           = P / pHeight;
    let lightZenithCosA = dot(lightDir, upVec);
    let transToLight    = sample_transmittance_lut(transLut, lutSmp, pHeight, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm);
    let tPlanet         = ray_sphere_nearest(P, lightDir, atm.bottomRadiusKm);
    let shadow          = select(0.0, 1.0, tPlanet < 0.0);

    let phaseScatter = med.scatteringMie * miePhaseVal + med.scatteringRay * rayPhaseVal;

    var multiScatLum = vec3f(0.0);
    if (apParams.enableMultiScat > 0.5) {
      multiScatLum = sample_multiscat_lut(msLut, lutSmp, P, lightZenithCosA, atm.bottomRadiusKm, atm.topRadiusKm);
    }

    let S    = lightIllum * (shadow * transToLight * phaseScatter + multiScatLum * med.scattering);
    let sint = (S - S * sampleTr) / max(med.extinction, vec3f(1e-9));
    L          += throughput * sint;
    throughput *= sampleTr;
  }

  let transmittance = dot(throughput, vec3f(1.0 / 3.0));
  textureStore(outputVol, gid, vec4f(L, transmittance));
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Shader: Pass 4 — Sky Render (全屏天空 + 太阳盘)
// 对应 UE5: RenderSkyAtmosphereRayMarchingPS (FastSky 分支)
// ─────────────────────────────────────────────────────────────────────────────

const SKY_RENDER_WGSL = ATMOSPHERE_COMMON_WGSL + /* wgsl */`

@group(0) @binding(0) var<uniform> atm       : AtmosphereUniforms;
@group(0) @binding(1) var<uniform> skyParams : SkyRenderParams;
@group(0) @binding(2) var          transLut  : texture_2d<f32>;
@group(0) @binding(3) var          skyLut    : texture_2d<f32>;
@group(0) @binding(4) var          depthTex  : texture_2d<f32>;
@group(0) @binding(5) var          lutSampler: sampler;
@group(0) @binding(6) var          outputTex : texture_storage_2d<rgba16float, write>;

struct SkyRenderParams {
  // 相机位置（km，相对于星球中心，Z轴向上）
  camPosX       : f32,
  camPosY       : f32,
  camPosZ       : f32,
  skyLumFactor  : f32,
  // 太阳方向（世界空间）
  lightDirX     : f32,
  lightDirY     : f32,
  lightDirZ     : f32,
  // 太阳盘参数
  sunDiscCosHalfApex : f32,
  sunDiscLumR        : f32,
  sunDiscLumG        : f32,
  sunDiscLumB        : f32,
  renderSunDisc      : f32,  // 0 or 1
  // 视图参数
  viewportW      : f32,
  viewportH      : f32,
  // SkyViewLUT local referential row 0-2 (Z-up 3x3 matrix)
  refRow0X : f32, refRow0Y : f32, refRow0Z : f32, _pad0 : f32,
  refRow1X : f32, refRow1Y : f32, refRow1Z : f32, _pad1 : f32,
  refRow2X : f32, refRow2Y : f32, refRow2Z : f32, _pad2 : f32,
  // 逆投影参数 (tan(fov/2))
  fovHalfTanX : f32,
  fovHalfTanY : f32,
  nearClip    : f32,
  _pad3       : f32,
}

struct FSVert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_sky(@builtin(vertex_index) vi: u32) -> FSVert {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32( vi         & 2u) * 2.0 - 1.0;
  var out: FSVert;
  out.pos = vec4f(x, y, 1.0, 1.0);
  out.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return out;
}

@fragment fn fs_sky(in: FSVert) -> @location(0) vec4f {
  let pixPos = in.uv;

  // 检查深度：只在 sky (far) 像素渲染
  let depth = textureSample(depthTex, lutSampler, pixPos).r;
  if (depth < 1.0) {
    discard;
  }

  // 世界空间视线方向（逆投影）
  let ndcX     = pixPos.x * 2.0 - 1.0;
  let ndcY     = 1.0 - pixPos.y * 2.0;
  let viewDir  = normalize(vec3f(ndcX * skyParams.fovHalfTanX, ndcY * skyParams.fovHalfTanY, 1.0));

  let camPos   = vec3f(skyParams.camPosX, skyParams.camPosY, skyParams.camPosZ);
  let viewHeight = length(camPos);

  // SkyView LUT 本地坐标系变换（Z轴对齐垂直方向）
  let refRow0 = vec3f(skyParams.refRow0X, skyParams.refRow0Y, skyParams.refRow0Z);
  let refRow1 = vec3f(skyParams.refRow1X, skyParams.refRow1Y, skyParams.refRow1Z);
  let refRow2 = vec3f(skyParams.refRow2X, skyParams.refRow2Y, skyParams.refRow2Z);
  let localDir = vec3f(dot(refRow0, viewDir), dot(refRow1, viewDir), dot(refRow2, viewDir));

  let upVectorLocal        = vec3f(0.0, 0.0, 1.0);
  let viewZenithCosAngle   = dot(localDir, upVectorLocal);
  let intersectGround      = ray_sphere_nearest(vec3f(0.0, 0.0, viewHeight), localDir, atm.bottomRadiusKm) >= 0.0;

  // SkyView LUT UV 采样
  // UV.y 非线性参数化（对应 UE5 SkyViewLutParamsToUv）
  let vHorizon    = sqrt(max(0.0, viewHeight * viewHeight - atm.bottomRadiusKm * atm.bottomRadiusKm));
  let cosBeta     = vHorizon / viewHeight;
  let beta        = acos(cosBeta);
  let zenithHorizAngle = PI - beta;
  let viewZenithAngle  = acos(clamp(viewZenithCosAngle, -1.0, 1.0));

  var uvY: f32;
  if (!intersectGround) {
    var coord = viewZenithAngle / zenithHorizAngle;
    coord = 1.0 - coord;
    coord = sqrt(coord);
    coord = 1.0 - coord;
    uvY = coord * 0.5;
  } else {
    var coord = (viewZenithAngle - zenithHorizAngle) / beta;
    coord = sqrt(max(0.0, coord));
    uvY = coord * 0.5 + 0.5;
  }

  // Sub-texel 修正
  let lutW = f32(${LUT_SKYVIEW_W});
  let lutH = f32(${LUT_SKYVIEW_H});
  let uvYFinal = (uvY + 0.5 / lutH) * (lutH / (lutH + 1.0));

  // UV.x = 方位角
  let azimuth = (atan2(-localDir.y, -localDir.x) + PI) / (2.0 * PI);
  let uvXFinal = (azimuth + 0.5 / lutW) * (lutW / (lutW + 1.0));

  let skyLutSample = textureSampleLevel(skyLut, lutSampler, vec2f(uvXFinal, uvYFinal), 0.0);
  var skyLuminance = skyLutSample.rgb * skyParams.skyLumFactor;

  // 太阳盘（对应 UE5 GetLightDiskLuminance）
  if (skyParams.renderSunDisc > 0.5) {
    let lightDir       = normalize(vec3f(skyParams.lightDirX, skyParams.lightDirY, skyParams.lightDirZ));
    let viewDotLight   = dot(viewDir, lightDir);
    let cosHalfApex    = skyParams.sunDiscCosHalfApex;
    if (viewDotLight > cosHalfApex) {
      // 行星阴影检测
      let tPlanet = ray_sphere_nearest(camPos, viewDir, atm.bottomRadiusKm);
      if (tPlanet < 0.0) {
        // 软边渐变（避免 TAA 闪烁）
        let softEdge = clamp(2.0 * (viewDotLight - cosHalfApex) / max(1e-7, 1.0 - cosHalfApex), 0.0, 1.0);
        // 从 Transmittance LUT 获取到太阳方向的透射率
        let planetCamLen = length(camPos);
        let upVecCam     = camPos / planetCamLen;
        let lightCosA    = dot(lightDir, upVecCam);
        let transToSun   = sample_transmittance_lut(transLut, lutSampler, planetCamLen, lightCosA, atm.bottomRadiusKm, atm.topRadiusKm);
        let discLum      = vec3f(skyParams.sunDiscLumR, skyParams.sunDiscLumG, skyParams.sunDiscLumB);
        skyLuminance += transToSun * discLum * softEdge;
      }
    }
  }

  return vec4f(skyLuminance, 1.0);
}

`;

// ─────────────────────────────────────────────────────────────────────────────
// UBO 打包工具
// ─────────────────────────────────────────────────────────────────────────────

function packAtmosphereUBO(p: AtmosphereParams): Float32Array {
  return new Float32Array([
    p.multiScatteringFactor,
    p.bottomRadiusKm,
    p.topRadiusKm,
    p.rayleighDensityExpScale,

    p.rayleighScattering[0], p.rayleighScattering[1], p.rayleighScattering[2],
    p.mieDensityExpScale,

    p.mieScattering[0], p.mieScattering[1], p.mieScattering[2],
    p.mieExtinction[0],

    p.mieExtinction[1], p.mieExtinction[2],
    p.mieAbsorption[0], p.mieAbsorption[1],

    p.mieAbsorption[2],
    p.miePhaseG,
    p.absorptionDensity0LayerWidth,
    p.absorptionDensity0LinearTerm,

    p.absorptionDensity0ConstantTerm,
    p.absorptionDensity1LinearTerm,
    p.absorptionDensity1ConstantTerm,
    p.absorptionExtinction[0],

    p.absorptionExtinction[1], p.absorptionExtinction[2],
    p.groundAlbedo[0], p.groundAlbedo[1],

    p.groundAlbedo[2], 0, 0, 0,
  ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Class: UEAtmosphereSky
// ─────────────────────────────────────────────────────────────────────────────

/** GPU 渲染管线资源句柄集合（内部使用） */
interface AtmosphereGPUResources {
  // UBO
  uboAtmosphere    : GPUBuffer;
  uboTransLutParams: GPUBuffer;
  uboMsLutParams   : GPUBuffer;
  uboSkyViewParams : GPUBuffer;
  uboApParams      : GPUBuffer;
  uboSkyRender     : GPUBuffer;

  // Textures
  transmittanceLUT : GPUTexture;
  multiScatLUT     : GPUTexture;
  skyViewLUT       : GPUTexture;
  aerialPerspVol   : GPUTexture;
  outputSkyTex     : GPUTexture;

  // Pipelines
  pipeTransmittance : GPUComputePipeline;
  pipeMultiScat     : GPUComputePipeline;
  pipeSkyView       : GPUComputePipeline;
  pipeAerialPersp   : GPUComputePipeline;
  pipeSkeyRender    : GPURenderPipeline;

  // Samplers
  linearSampler     : GPUSampler;
}

/** UEAtmosphereSky.render() 输入资源 */
export interface AtmosphereSkyRenderInputs {
  /** 场景深度纹理（r32float 或 depth32float，用于天空遮罩） */
  depthTex  : GPUTexture;
  /** 相机参数 */
  camera    : AtmosphereCameraParams;
}

/**
 * UEAtmosphereSky — UE5 SkyAtmosphere WebGPU 移植主类
 *
 * 用法:
 * ```ts
 * const sky = await UEAtmosphereSky.create(device, 'rgba16float', 1920, 1080);
 * sky.setSunDirection([0.0, 0.5, 0.866]);
 * // 每帧:
 * sky.setViewData(camera);
 * const outputTex = sky.render(encoder, { depthTex, camera });
 * ```
 */
export class UEAtmosphereSky {
  private device  : GPUDevice;
  private width   : number;
  private height  : number;
  private format  : GPUTextureFormat;
  private config  : UEAtmosphereSkyConfig;

  private gpu!    : AtmosphereGPUResources;

  // LUT 失效标记
  private lutDirty       = true;  // Transmittance + MultiScat LUT 需要重建
  private skyViewDirty   = true;  // SkyView LUT 每帧更新

  private _initialized   = false;

  private constructor(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
    config : UEAtmosphereSkyConfig,
  ) {
    this.device  = device;
    this.format  = format;
    this.width   = width;
    this.height  = height;
    this.config  = { ...config, atmosphere: { ...config.atmosphere }, sun: { ...config.sun } };
  }

  /**
   * 工厂方法：创建并初始化所有 GPU 资源
   */
  static async create(
    device  : GPUDevice,
    format  : GPUTextureFormat = 'rgba16float',
    width   : number,
    height  : number,
    config  : Partial<UEAtmosphereSkyConfig> = {},
  ): Promise<UEAtmosphereSky> {
    const fullConfig = {
      ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
      ...config,
      atmosphere: { ...DEFAULT_ATMOSPHERE_PARAMS, ...(config.atmosphere ?? {}) },
      sun:        { ...DEFAULT_SUN_PARAMS,        ...(config.sun       ?? {}) },
    };
    const sky = new UEAtmosphereSky(device, format, width, height, fullConfig);
    await sky._init();
    return sky;
  }

  // ─── 公开 API ─────────────────────────────────────────────────────────────

  /**
   * 更新太阳方向（触发 SkyView LUT 重建）
   * @param dir 单位向量，世界空间，指向太阳
   */
  setSunDirection(dir: [number, number, number]): void {
    const n = Math.hypot(dir[0], dir[1], dir[2]);
    this.config.sun.direction = [dir[0] / n, dir[1] / n, dir[2] / n];
    this.skyViewDirty = true;
  }

  /**
   * 更新太阳外层照度（触发 SkyView LUT 重建）
   */
  setSunIlluminance(illum: [number, number, number]): void {
    this.config.sun.illuminanceOuterSpace = [...illum];
    this.skyViewDirty = true;
  }

  /**
   * 更新大气参数（触发所有 LUT 完整重建）
   */
  setAtmosphereParams(params: Partial<AtmosphereParams>): void {
    Object.assign(this.config.atmosphere, params);
    this.lutDirty     = true;
    this.skyViewDirty = true;
    // 更新 GPU UBO
    const data = packAtmosphereUBO(this.config.atmosphere);
    this.device.queue.writeBuffer(this.gpu.uboAtmosphere, 0, data);
  }

  /**
   * 执行完整大气天空渲染管线
   * @param encoder GPUCommandEncoder
   * @param inputs  渲染输入资源
   * @returns 天空颜色输出纹理（与 width/height 一致）
   */
  render(encoder: GPUCommandEncoder, inputs: AtmosphereSkyRenderInputs): GPUTexture {
    if (!this._initialized) {
      console.warn('[UEAtmosphereSky] not initialized');
      return this.gpu.outputSkyTex;
    }

    // Pass 0: Transmittance LUT（仅当大气参数改变时重建）
    if (this.lutDirty) {
      this._encodeTransmittanceLUT(encoder);
    }

    // Pass 1: Multi-Scattering LUT（仅当大气参数改变时重建）
    if (this.lutDirty) {
      this._encodeMultiScatLUT(encoder);
      this.lutDirty = false;
    }

    // Pass 2: Sky View LUT（每帧更新，太阳方向/相机高度变化）
    if (this.skyViewDirty) {
      this._updateSkyViewUBO(inputs.camera);
      this._encodeSkyViewLUT(encoder);
      this.skyViewDirty = false;
    }

    // Pass 3: Aerial Perspective Volume（每帧）
    this._updateAerialPerspUBO(inputs.camera);
    this._encodeAerialPerspective(encoder);

    // Pass 4: Sky Render（每帧）
    this._updateSkyRenderUBO(inputs.camera);
    this._encodeSkyRender(encoder, inputs.depthTex);

    return this.gpu.outputSkyTex;
  }

  /**
   * 直接获取 Aerial Perspective Volume 纹理（供不透明物体大气透视合成使用）
   */
  getAerialPerspectiveVolume(): GPUTexture {
    return this.gpu.aerialPerspVol;
  }

  /**
   * 直接获取 Transmittance LUT（供外部材质系统使用）
   */
  getTransmittanceLUT(): GPUTexture {
    return this.gpu.transmittanceLUT;
  }

  /**
   * 调整输出分辨率
   */
  async resize(newWidth: number, newHeight: number): Promise<void> {
    this.width  = newWidth;
    this.height = newHeight;
    this._destroyTextures();
    this._createTextures();
    this.skyViewDirty = true;
  }

  /** 释放所有 GPU 资源 */
  destroy(): void {
    this._destroyTextures();
    this.gpu.uboAtmosphere?.destroy();
    this.gpu.uboTransLutParams?.destroy();
    this.gpu.uboMsLutParams?.destroy();
    this.gpu.uboSkyViewParams?.destroy();
    this.gpu.uboApParams?.destroy();
    this.gpu.uboSkyRender?.destroy();
  }

  // ─── 内部初始化 ───────────────────────────────────────────────────────────

  private async _init(): Promise<void> {
    const d = this.device;

    // --- UBO ---
    const makeUBO = (size: number, label: string) =>
      d.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });

    const atmData = packAtmosphereUBO(this.config.atmosphere);

    const uboAtmosphere     = makeUBO(32 * 4,  'atmo_ubo_atmosphere');
    const uboTransLutParams = makeUBO(4  * 4,  'atmo_ubo_trans_lut_params');
    const uboMsLutParams    = makeUBO(4  * 4,  'atmo_ubo_ms_lut_params');
    const uboSkyViewParams  = makeUBO(16 * 4,  'atmo_ubo_skyview_params');
    const uboApParams       = makeUBO(16 * 4,  'atmo_ubo_ap_params');
    const uboSkyRender      = makeUBO(32 * 4,  'atmo_ubo_sky_render');

    d.queue.writeBuffer(uboAtmosphere, 0, atmData);
    d.queue.writeBuffer(uboTransLutParams, 0, new Float32Array([
      this.config.transmittanceSampleCount,
      LUT_TRANSMITTANCE_W, LUT_TRANSMITTANCE_H, 0,
    ]));
    d.queue.writeBuffer(uboMsLutParams, 0, new Float32Array([
      this.config.multiScatteringSampleCount,
      LUT_MULTISCAT_W, LUT_MULTISCAT_H, 0,
    ]));

    // --- Sampler ---
    const linearSampler = d.createSampler({
      magFilter   : 'linear',
      minFilter   : 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    // --- Pipelines ---
    const pipeTransmittance = await this._createTransmittancePipeline();
    const pipeMultiScat     = await this._createMultiScatPipeline();
    const pipeSkyView       = await this._createSkyViewPipeline();
    const pipeAerialPersp   = await this._createAerialPerspPipeline();
    const pipeSkeyRender    = await this._createSkyRenderPipeline();

    this.gpu = {
      uboAtmosphere, uboTransLutParams, uboMsLutParams,
      uboSkyViewParams, uboApParams, uboSkyRender,
      linearSampler,
      pipeTransmittance, pipeMultiScat, pipeSkyView, pipeAerialPersp, pipeSkeyRender,
    } as any;

    this._createTextures();
    this._initialized = true;
  }

  private _createTextures(): void {
    const d = this.device;
    const make2d = (w: number, h: number, label: string): GPUTexture =>
      d.createTexture({
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        label,
      });
    const make3d = (w: number, h: number, depth: number, label: string): GPUTexture =>
      d.createTexture({
        size: [w, h, depth],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        label,
      });
    const makeOut = (w: number, h: number, label: string): GPUTexture =>
      d.createTexture({
        size: [w, h],
        format: this.format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        label,
      });

    (this.gpu as any).transmittanceLUT = make2d(LUT_TRANSMITTANCE_W, LUT_TRANSMITTANCE_H, 'atmo_transmittance_lut');
    (this.gpu as any).multiScatLUT     = make2d(LUT_MULTISCAT_W, LUT_MULTISCAT_H, 'atmo_multiscat_lut');
    (this.gpu as any).skyViewLUT       = make2d(LUT_SKYVIEW_W, LUT_SKYVIEW_H, 'atmo_skyview_lut');
    (this.gpu as any).aerialPerspVol   = make3d(AP_VOL_W, AP_VOL_H, AP_VOL_D, 'atmo_aerial_persp_vol');
    (this.gpu as any).outputSkyTex     = makeOut(this.width, this.height, 'atmo_sky_output');
  }

  private _destroyTextures(): void {
    this.gpu.transmittanceLUT?.destroy();
    this.gpu.multiScatLUT?.destroy();
    this.gpu.skyViewLUT?.destroy();
    this.gpu.aerialPerspVol?.destroy();
    this.gpu.outputSkyTex?.destroy();
  }

  // ─── Pipeline 创建 ────────────────────────────────────────────────────────

  private async _createTransmittancePipeline(): Promise<GPUComputePipeline> {
    const mod = this.device.createShaderModule({ code: TRANSMITTANCE_LUT_WGSL, label: 'transmittance_lut' });
    return this.device.createComputePipeline({
      layout : 'auto',
      compute: { module: mod, entryPoint: 'cs_transmittance' },
    });
  }

  private async _createMultiScatPipeline(): Promise<GPUComputePipeline> {
    const mod = this.device.createShaderModule({ code: MULTISCAT_LUT_WGSL, label: 'multiscat_lut' });
    return this.device.createComputePipeline({
      layout : 'auto',
      compute: { module: mod, entryPoint: 'cs_multiscat' },
    });
  }

  private async _createSkyViewPipeline(): Promise<GPUComputePipeline> {
    const mod = this.device.createShaderModule({ code: SKYVIEW_LUT_WGSL, label: 'skyview_lut' });
    return this.device.createComputePipeline({
      layout : 'auto',
      compute: { module: mod, entryPoint: 'cs_skyview' },
    });
  }

  private async _createAerialPerspPipeline(): Promise<GPUComputePipeline> {
    const mod = this.device.createShaderModule({ code: AERIAL_PERSPECTIVE_WGSL, label: 'aerial_persp' });
    return this.device.createComputePipeline({
      layout : 'auto',
      compute: { module: mod, entryPoint: 'cs_aerial_perspective' },
    });
  }

  private async _createSkyRenderPipeline(): Promise<GPURenderPipeline> {
    const mod = this.device.createShaderModule({ code: SKY_RENDER_WGSL, label: 'sky_render' });
    return this.device.createRenderPipeline({
      layout  : 'auto',
      vertex  : { module: mod, entryPoint: 'vs_sky' },
      fragment: {
        module     : mod,
        entryPoint : 'fs_sky',
        targets    : [{ format: this.format, blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        }}],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  // ─── UBO 更新 ─────────────────────────────────────────────────────────────

  private _updateSkyViewUBO(camera: AtmosphereCameraParams): void {
    const { sun } = this.config;
    const camPosCm  = camera.worldPos;
    const planetCm  = camera.planetCenter;
    // 相对于星球中心 (km)
    const camKm = [
      (camPosCm[0] - planetCm[0]) * 0.00001,
      (camPosCm[1] - planetCm[1]) * 0.00001,
      (camPosCm[2] - planetCm[2]) * 0.00001,
    ];
    const viewHeightKm = Math.sqrt(camKm[0] ** 2 + camKm[1] ** 2 + camKm[2] ** 2);

    // SkyView 本地参考系：Z轴 = 相机到星球中心方向（向上）
    const upVec = [camKm[0] / viewHeightKm, camKm[1] / viewHeightKm, camKm[2] / viewHeightKm];
    // 将太阳方向变换到本地参考系（简化：直接使用世界空间，对称情况下等价）
    const lightDir = sun.direction;

    const illum  = sun.illuminanceOuterSpace.map(v => v * this.config.skyAndAerialPerspectiveLuminanceFactor);
    const distMax = this.config.fastSkySampleCountMax * 1000.0; // km

    this.device.queue.writeBuffer(this.gpu.uboSkyViewParams, 0, new Float32Array([
      viewHeightKm,
      lightDir[0], lightDir[1], lightDir[2],
      illum[0], illum[1], illum[2],
      this.config.fastSkySampleCountMin,
      this.config.fastSkySampleCountMax,
      1.0 / distMax,
      this.config.skyLuminanceFactor,
      this.config.enableMultiScattering ? 1.0 : 0.0,
    ]));
  }

  private _updateAerialPerspUBO(camera: AtmosphereCameraParams): void {
    const { sun } = this.config;
    const camPosCm = camera.worldPos;
    const planetCm = camera.planetCenter;
    const camKm = [
      (camPosCm[0] - planetCm[0]) * 0.00001,
      (camPosCm[1] - planetCm[1]) * 0.00001,
      (camPosCm[2] - planetCm[2]) * 0.00001,
    ];
    const illum = sun.illuminanceOuterSpace.map(v => v * this.config.skyAndAerialPerspectiveLuminanceFactor);
    const fovHalfTanX = 1.0; // 默认 90° FOV (tan 45° = 1.0) — 应由 invProj 覆盖
    const fovHalfTanY = camera.height / camera.width;

    this.device.queue.writeBuffer(this.gpu.uboApParams, 0, new Float32Array([
      camKm[0], camKm[1], camKm[2],
      this.config.aerialPerspectiveStartDepthKm,
      sun.direction[0], sun.direction[1], sun.direction[2],
      illum[0], illum[1], illum[2],
      this.config.cameraAerialPerspectiveSampleCountPerSlice,
      AP_KM_PER_SLICE,
      fovHalfTanX,
      fovHalfTanY,
      this.config.aerialPerspectiveViewDistanceScale,
      this.config.enableMultiScattering ? 1.0 : 0.0,
    ]));
  }

  private _updateSkyRenderUBO(camera: AtmosphereCameraParams): void {
    const { sun } = this.config;
    const camPosCm = camera.worldPos;
    const planetCm = camera.planetCenter;
    const camKm = [
      (camPosCm[0] - planetCm[0]) * 0.00001,
      (camPosCm[1] - planetCm[1]) * 0.00001,
      (camPosCm[2] - planetCm[2]) * 0.00001,
    ];

    // 构建 SkyView 本地参考系（3x3，Z轴对齐相机上方向）
    const viewHeightKm = Math.sqrt(camKm[0] ** 2 + camKm[1] ** 2 + camKm[2] ** 2);
    const upZ = viewHeightKm > 0
      ? [camKm[0] / viewHeightKm, camKm[1] / viewHeightKm, camKm[2] / viewHeightKm]
      : [0, 0, 1];
    // 构建与 upZ 正交的参考系
    const tmpX: [number, number, number] = Math.abs(upZ[2]) < 0.999 ? [0, 0, 1] : [1, 0, 0];
    const right = normalizeVec3(crossVec3(tmpX, upZ as [number,number,number]));
    const fwd   = crossVec3(upZ as [number,number,number], right);

    // 逆投影参数
    const fovHalfTanX = camera.width > 0 ? 1.0 : 1.0;
    const fovHalfTanY = camera.width > 0 ? camera.height / camera.width : 1.0;

    const discLum = sun.discLuminance;
    this.device.queue.writeBuffer(this.gpu.uboSkyRender, 0, new Float32Array([
      // row 0: camPos + skyLumFactor
      camKm[0], camKm[1], camKm[2], this.config.skyLuminanceFactor,
      // row 1: lightDir + renderSunDisc
      sun.direction[0], sun.direction[1], sun.direction[2],
      // sun disc
      sun.discCosHalfApexAngle,
      discLum, discLum, discLum,
      this.config.renderSunDisk ? 1.0 : 0.0,
      // viewport
      camera.width, camera.height,
      // ref row 0
      right[0], right[1], right[2], 0,
      // ref row 1
      fwd[0],   fwd[1],   fwd[2],   0,
      // ref row 2 (= up)
      upZ[0],   upZ[1],   upZ[2],   0,
      // fov
      fovHalfTanX, fovHalfTanY, 0.1, 0,
    ]));
  }

  // ─── Encode 通道 ──────────────────────────────────────────────────────────

  private _encodeTransmittanceLUT(encoder: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout : this.gpu.pipeTransmittance.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gpu.uboAtmosphere } },
        { binding: 1, resource: { buffer: this.gpu.uboTransLutParams } },
        { binding: 2, resource: this.gpu.transmittanceLUT.createView() },
      ],
      label: 'bg_transmittance',
    });
    const pass = encoder.beginComputePass({ label: 'transmittance_lut' });
    pass.setPipeline(this.gpu.pipeTransmittance);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(LUT_TRANSMITTANCE_W / 8),
      Math.ceil(LUT_TRANSMITTANCE_H / 8),
    );
    pass.end();
  }

  private _encodeMultiScatLUT(encoder: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout : this.gpu.pipeMultiScat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gpu.uboAtmosphere } },
        { binding: 1, resource: { buffer: this.gpu.uboMsLutParams } },
        { binding: 2, resource: this.gpu.transmittanceLUT.createView() },
        { binding: 3, resource: this.gpu.linearSampler },
        { binding: 4, resource: this.gpu.multiScatLUT.createView() },
      ],
      label: 'bg_multiscat',
    });
    const pass = encoder.beginComputePass({ label: 'multiscat_lut' });
    pass.setPipeline(this.gpu.pipeMultiScat);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(LUT_MULTISCAT_W / 8),
      Math.ceil(LUT_MULTISCAT_H / 8),
    );
    pass.end();
  }

  private _encodeSkyViewLUT(encoder: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout : this.gpu.pipeSkyView.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gpu.uboAtmosphere } },
        { binding: 1, resource: { buffer: this.gpu.uboSkyViewParams } },
        { binding: 2, resource: this.gpu.transmittanceLUT.createView() },
        { binding: 3, resource: this.gpu.multiScatLUT.createView() },
        { binding: 4, resource: this.gpu.linearSampler },
        { binding: 5, resource: this.gpu.skyViewLUT.createView() },
      ],
      label: 'bg_skyview',
    });
    const pass = encoder.beginComputePass({ label: 'skyview_lut' });
    pass.setPipeline(this.gpu.pipeSkyView);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(LUT_SKYVIEW_W / 8),
      Math.ceil(LUT_SKYVIEW_H / 8),
    );
    pass.end();
  }

  private _encodeAerialPerspective(encoder: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout : this.gpu.pipeAerialPersp.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gpu.uboAtmosphere } },
        { binding: 1, resource: { buffer: this.gpu.uboApParams } },
        { binding: 2, resource: this.gpu.transmittanceLUT.createView() },
        { binding: 3, resource: this.gpu.multiScatLUT.createView() },
        { binding: 4, resource: this.gpu.linearSampler },
        { binding: 5, resource: this.gpu.aerialPerspVol.createView() },
      ],
      label: 'bg_aerial_persp',
    });
    const pass = encoder.beginComputePass({ label: 'aerial_persp' });
    pass.setPipeline(this.gpu.pipeAerialPersp);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(AP_VOL_W / 4),
      Math.ceil(AP_VOL_H / 4),
      Math.ceil(AP_VOL_D / 4),
    );
    pass.end();
  }

  private _encodeSkyRender(encoder: GPUCommandEncoder, depthTex: GPUTexture): void {
    // 注意: fs_sky 使用 depthTex 做 far pixel 遮罩
    // 创建一个临时的 depth view（格式需为 float）
    const depthView = depthTex.createView({
      aspect: depthTex.format.includes('depth') ? 'all' : 'all',
    });

    const bg = this.device.createBindGroup({
      layout : this.gpu.pipeSkeyRender.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gpu.uboAtmosphere } },
        { binding: 1, resource: { buffer: this.gpu.uboSkyRender } },
        { binding: 2, resource: this.gpu.transmittanceLUT.createView() },
        { binding: 3, resource: this.gpu.skyViewLUT.createView() },
        { binding: 4, resource: depthView },
        { binding: 5, resource: this.gpu.linearSampler },
        { binding: 6, resource: this.gpu.outputSkyTex.createView() },
      ],
      label: 'bg_sky_render',
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view       : this.gpu.outputSkyTex.createView(),
        loadOp     : 'clear',
        storeOp    : 'store',
        clearValue : { r: 0, g: 0, b: 0, a: 0 },
      }],
      label: 'sky_render',
    });
    pass.setPipeline(this.gpu.pipeSkeyRender);
    pass.setBindGroup(0, bg);
    pass.draw(3); // 全屏三角形
    pass.end();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 向量工具（避免引入外部依赖）
// ─────────────────────────────────────────────────────────────────────────────

function crossVec3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalizeVec3(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell World 集成适配器
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CellAtmosphereBackground — Cell 世界大气层背景管理器
 *
 * 封装 UEAtmosphereSky，提供与 CellSpecies 体系兼容的接口。
 * 通过 pubsub 机制响应时间/光照变化，驱动大气层动态效果。
 *
 * 示例:
 * ```ts
 * const cellAtmo = await CellAtmosphereBackground.create(device, canvas, species);
 * cellAtmo.setTimeOfDay(0.3);  // 0=午夜, 0.25=日出, 0.5=正午, 0.75=日落
 * ```
 */
export class CellAtmosphereBackground {
  private sky       : UEAtmosphereSky;
  private timeOfDay : number = 0.5;

  private constructor(sky: UEAtmosphereSky) {
    this.sky = sky;
  }

  static async create(
    device   : GPUDevice,
    width    : number,
    height   : number,
    species ?: Partial<CellSpecies>,
    config  ?: Partial<UEAtmosphereSkyConfig>,
  ): Promise<CellAtmosphereBackground> {
    // 根据 Cell species 颜色调整大气颜色
    const atmosConfig = CellAtmosphereBackground._buildConfigFromSpecies(species, config);
    const sky = await UEAtmosphereSky.create(device, 'rgba16float', width, height, atmosConfig);
    const inst = new CellAtmosphereBackground(sky);
    inst.setTimeOfDay(0.5);
    return inst;
  }

  /**
   * 设置时间（驱动太阳位置）
   * @param t 0..1, 0.0=午夜, 0.25=日出, 0.5=正午, 0.75=日落
   */
  setTimeOfDay(t: number): void {
    this.timeOfDay = t;
    // 太阳仰角随时间变化（正弦曲线，正午最高）
    const angle   = (t - 0.25) * 2.0 * Math.PI;  // 0.25 → 0 rad（日出）
    const elevation = Math.sin(angle);
    const azimuth   = Math.cos(angle);
    const sunDir: [number, number, number] = [
      azimuth * 0.7071,
      azimuth * 0.7071,
      elevation,
    ];
    const len = Math.sqrt(sunDir[0] ** 2 + sunDir[1] ** 2 + sunDir[2] ** 2) || 1;
    this.sky.setSunDirection([sunDir[0] / len, sunDir[1] / len, sunDir[2] / len]);

    // 动态调整太阳照度（落日/黎明时降低）
    const dayFactor  = Math.max(0.0, elevation);
    const sunIntensity = 0.2 + 0.8 * dayFactor;
    this.sky.setSunIlluminance([sunIntensity, sunIntensity, sunIntensity]);
  }

  /**
   * 每帧渲染大气背景
   */
  render(
    encoder : GPUCommandEncoder,
    inputs  : AtmosphereSkyRenderInputs,
  ): GPUTexture {
    return this.sky.render(encoder, inputs);
  }

  /** 获取 Aerial Perspective 体积纹理（供不透明物体 pass 使用） */
  getAerialPerspectiveVolume(): GPUTexture {
    return this.sky.getAerialPerspectiveVolume();
  }

  /** 获取 Transmittance LUT */
  getTransmittanceLUT(): GPUTexture {
    return this.sky.getTransmittanceLUT();
  }

  async resize(w: number, h: number): Promise<void> {
    await this.sky.resize(w, h);
  }

  destroy(): void {
    this.sky.destroy();
  }

  // 根据 CellSpecies 色相构建大气配置（例：蓝色 Cell 场景→偏蓝大气）
  private static _buildConfigFromSpecies(
    species?: Partial<CellSpecies>,
    override?: Partial<UEAtmosphereSkyConfig>,
  ): Partial<UEAtmosphereSkyConfig> {
    const base: Partial<UEAtmosphereSkyConfig> = {
      ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
      ...override,
    };
    if (!species) return base;

    // 如果 species 有 color 属性，轻微调整大气散射色调
    const color = (species as any).color as [number, number, number] | undefined;
    if (color) {
      const scale = 0.15;  // 最大色调偏移量
      const atm   = { ...DEFAULT_ATMOSPHERE_PARAMS };
      atm.rayleighScattering = [
        DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[0] * (1.0 + color[0] * scale),
        DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[1] * (1.0 + color[1] * scale),
        DEFAULT_ATMOSPHERE_PARAMS.rayleighScattering[2] * (1.0 + color[2] * scale),
      ];
      base.atmosphere = atm;
    }
    return base;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 预设大气配置
// ─────────────────────────────────────────────────────────────────────────────

/** 地球大气（默认，Bruneton 2017 参数） */
export const PRESET_EARTH_ATMOSPHERE: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: DEFAULT_ATMOSPHERE_PARAMS,
};

/** 火星大气（薄红色大气） */
export const PRESET_MARS_ATMOSPHERE: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: {
    ...DEFAULT_ATMOSPHERE_PARAMS,
    topRadiusKm         : 6380.0,   // 火星半径 3390km，大气厚 ~20km
    bottomRadiusKm      : 3390.0,
    rayleighScattering  : [0.019918, 0.011358, 0.004966],  // 偏红
    mieScattering       : [0.002000, 0.001800, 0.001500],
    mieExtinction       : [0.002200, 0.002000, 0.001700],
    miePhaseG           : 0.76,
    absorptionExtinction: [0.000250, 0.000300, 0.000025],
    groundAlbedo        : [0.35, 0.18, 0.08],
  },
  skyLuminanceFactor: 0.08,
};

/** 外星黄昏大气（Cell 世界幻想效果）*/
export const PRESET_ALIEN_TWILIGHT: Partial<UEAtmosphereSkyConfig> = {
  atmosphere: {
    ...DEFAULT_ATMOSPHERE_PARAMS,
    rayleighScattering  : [0.012000, 0.004500, 0.019000],  // 紫/蓝
    mieScattering       : [0.006000, 0.005000, 0.004000],  // 暖色 Mie
    miePhaseG           : 0.85,
    groundAlbedo        : [0.05, 0.03, 0.07],
  },
  skyLuminanceFactor   : 1.2,
  renderSunDisk        : true,
};

/** 深夜模式（极弱光照）*/
export const PRESET_NIGHT_SKY: Partial<UEAtmosphereSkyConfig> = {
  ...DEFAULT_ATMOSPHERE_SKY_CONFIG,
  skyLuminanceFactor: 0.001,
  renderSunDisk: false,
  sun: {
    ...DEFAULT_SUN_PARAMS,
    illuminanceOuterSpace: [0.001, 0.001, 0.003],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// 便捷导出（供外部模块使用）
// ─────────────────────────────────────────────────────────────────────────────

export {
  LUT_TRANSMITTANCE_W, LUT_TRANSMITTANCE_H,
  LUT_MULTISCAT_W, LUT_MULTISCAT_H,
  LUT_SKYVIEW_W, LUT_SKYVIEW_H,
  AP_VOL_W, AP_VOL_H, AP_VOL_D,
  AP_KM_PER_SLICE,
};
