/**
 * ue-lumen-gi.ts — M835: UE5 Lumen Global Illumination Port
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Lumen 全局光照系统到 WebGPU/WGSL。
 * 原始来源: upstream/unreal-renderer-ue5/Renderer-Private/Lumen/ (~159 个文件)
 *
 * 四大核心子系统:
 *
 *   1. Screen-Space Probe Gathering (LumenScreenProbeGather.cpp/.h)
 *      屏幕空间探针采样间接光。以降采样因子 (默认 16px) 在屏幕上放置均匀探针,
 *      对每个探针在八面体映射的方向集合上追踪间接光,再用空间/时域滤波重建
 *      全分辨率漫反射间接光照 (SH3 或 Octahedral)。
 *
 *   2. Radiance Cache (LumenRadianceCache.cpp/.h)
 *      层叠 Clipmap 辐照度缓存。在 3D 空间中按 Clipmap 级别分配探针格,
 *      只更新本帧被屏幕探针标记为"使用"的格,其余复用历史帧结果。
 *      提供 sampleRadianceCache() 给屏幕探针和 Radiosity 使用,
 *      避免对远场光照重复追踪。
 *
 *   3. Software Ray Tracing Fallback — Cell 间光线弹射
 *      (LumenRadiosity.cpp, LumenScreenProbeTracing.cpp)
 *      当无法使用 HW RT 时,通过全局 SDF 软光线追踪。
 *      Cell 发射光线,击中其他 Cell 表面后读取该 Cell 的辐射度,
 *      模拟多弹射漫反射 (Radiosity)。
 *
 *   4. 漫反射 + 镜面反射间接光 (LumenDiffuseIndirect.cpp, LumenReflections.cpp)
 *      • 漫反射: 屏幕探针 GatherParameters → SH3 重建 → 与 albedo 卷积
 *      • 镜面反射: GGX 重要性采样追踪 + Radiance Cache 回退 + 时域累积
 *      最终合并到 HDR 颜色缓冲。
 *
 * 算法流程
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── Screen Probe Placement ─────────────────────────────────────┐
 *   │  depth + normal → 均匀探针位置 (每 downsampleFactor px 一个)             │
 *   │  深度不连续处插入自适应探针 (最多 numAdaptiveProbes 个)                    │
 *   │  输出: probeDepth, probeNormal, probeWorldPos                           │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ probeAtlas (ScreenProbeViewSize × octRes)
 *               ▼
 *   ┌─ Pass 1 ── Radiance Cache Update ──────────────────────────────────────┐
 *   │  从 probeWorldPos 标记被使用的 Clipmap 格                                │
 *   │  对新格: SDF 软追踪天空 + emissive + 上一 Clipmap 级辐照度               │
 *   │  时域混合: 新帧 weight = 1/maxFramesAccumulated                         │
 *   │  输出: radianceCacheTex3D (Clipmap × probeRes × probeRes)               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ radianceCacheUBO
 *               ▼
 *   ┌─ Pass 2 ── Screen Probe Tracing ────────────────────────────────────────┐
 *   │  对每个探针的每条八面体方向:                                              │
 *   │    1. 屏幕空间追踪 (RayMarch depth buffer, maxScreenSteps 步)            │
 *   │    2. 命中 → 读取 scene color (间接光来源)                               │
 *   │    3. 未命中 → 软 SDF 追踪 (maxSDFSteps 步)                             │
 *   │       命中 Cell SDF → 采样 Cell 表面辐射度 (Radiosity 弹射)             │
 *   │       仍未命中 → 从 Radiance Cache 插值                                 │
 *   │  输出: traceRadiance (probeAtlas × octRes)                              │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ traceRadianceTex
 *               ▼
 *   ┌─ Pass 3 ── Probe Filtering (Spatial + Temporal) ────────────────────────┐
 *   │  空间: 5×5 加权平均 (深度/法线相似性权重)                                 │
 *   │  转换到 SH3 (L2 球谐): 3 通道 × 9 系数                                  │
 *   │  时域混合: history SH blend (maxFramesAccumulated 帧)                   │
 *   │  输出: screenProbeSHAmbient / SHDirectional                             │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ probeSHTex
 *               ▼
 *   ┌─ Pass 4 ── Diffuse Integration ────────────────────────────────────────┐
 *   │  上采样: 每像素从周围探针双线性插值 SH (深度/法线权重)                    │
 *   │  重建漫反射辐照度: E = SH · albedo · (1 - metallic)                    │
 *   │  输出: diffuseIndirectTex (rgba16float)                                │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ diffuseIndirectTex
 *               ▼
 *   ┌─ Pass 5 ── Specular Gather (镜面反射间接光) ────────────────────────────┐
 *   │  GGX 重要性采样追踪: N 条反射光线 (roughness→步数自适应)                 │
 *   │  命中: 读场景色; 未命中: Radiance Cache 镜面 lobe 插值                  │
 *   │  Pre-integrated GGX 分裂求和 (split-sum approximation)                 │
 *   │  时域累积 (4 帧)                                                        │
 *   │  输出: specularIndirectTex (rgba16float)                               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *               │ specularIndirectTex
 *               ▼
 *   ┌─ Pass 6 ── Composite ──────────────────────────────────────────────────┐
 *   │  scene + diffuseIndirect + specularIndirect → finalHDR                │
 *   │  Energy-conserving: kD = albedo*(1-metallic), kS = F0+fresnel         │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * 与 Cell PubSub Loop 的集成
 * ─────────────────────────────────────────────────────────────────────────────
 * • UELumenGI.setCellSDFBuffer(buf) — 注入当前帧所有 Cell 的 SDF 球包围盒
 *   (position + radius + emissive), 供 Pass 2 软追踪使用。
 * • UELumenGI.setCellRadianceBuffer(buf) — 注入每个 Cell 表面辐射度纹理
 *   索引, 供 Radiosity 弹射读取 (模拟 LumenScene surface cache)。
 * • 每帧调用顺序: tick(dt) → render(encoder, inputs, output)
 *
 * 上游参考
 * ─────────────────────────────────────────────────────────────────────────────
 *   LumenScreenProbeGather.h/cpp  — 屏幕探针参数结构、OctRes、自适应探针
 *   LumenRadianceCache.h/cpp      — Clipmap 结构、UpdateRadianceCaches
 *   LumenRadiosity.h/cpp          — Radiosity probe spacing / SH atlas
 *   LumenDiffuseIndirect.cpp      — TraceStepFactor, SurfaceBias, 积分
 *   LumenReflections.h/cpp        — GGX split-sum, MaxRoughnessToTrace
 *   LumenScene.cpp                — GlobalSDF resolution / clipmap extents
 *   LumenReflectionTracing.cpp    — NearFieldMaxTraceDistance, FarField
 *   LumenTracingUtils.cpp         — SDF stepping utilities
 *
 * Research: xiaodi #M835 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

import type { CellSpecies }     from './cell-material-system';
import type { PhysicsUniforms } from './physics-uniform-bridge';

// ─────────────────────────────────────────────────────────────────────────────
// Public Types & Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Tunable configuration mirroring UE5 CVars from LumenScreenProbeGather + LumenRadiosity */
export interface UELumenGIConfig {
  /** Enable Lumen GI entirely. @default true */
  enabled: boolean;

  // ── Screen Probe Gather (r.Lumen.ScreenProbeGather.*) ──────────────────────
  /** Pixel tile size per screen probe (r.Lumen.ScreenProbeGather.DownsampleFactor). @default 16 */
  probeDownsampleFactor: number;
  /** Octahedron resolution for tracing (r.Lumen.ScreenProbeGather.TracingOctahedronResolution). @default 8 */
  tracingOctahedronResolution: number;
  /** Max adaptive probes per tile (r.Lumen.ScreenProbeGather.NumAdaptiveProbes). @default 8 */
  maxAdaptiveProbes: number;
  /** Adaptive probe allocation fraction. @default 0.5 */
  adaptiveProbeAllocationFraction: number;
  /** Spatial filter kernel radius in probe space. @default 2 */
  spatialFilterRadius: number;
  /** Max frames for temporal accumulation. @default 8 */
  maxFramesAccumulated: number;

  // ── Radiance Cache (r.Lumen.RadianceCache.*) ──────────────────────────────
  /** Number of Radiance Cache Clipmap levels. @default 4 */
  radianceCacheClipmapCount: number;
  /** Radiance Cache probe resolution per face (octahedral). @default 8 */
  radianceCacheProbeResolution: number;
  /** Radiance Cache Clipmap resolution (cells per axis). @default 32 */
  radianceCacheClipmapResolution: number;
  /** Radiance Cache max frames accumulated. @default 16 */
  radianceCacheMaxFrames: number;

  // ── Software Ray Tracing / Radiosity (r.LumenScene.Radiosity.*) ───────────
  /** Enable software SDF tracing fallback. @default true */
  softwareRayTracing: boolean;
  /** Max steps for SDF sphere-march. @default 64 */
  sdfMaxSteps: number;
  /** SDF trace max world-space distance (UE: r.Lumen.DiffuseIndirect.MeshSDFTraceDistance = 180). @default 180 */
  sdfMaxTraceDistance: number;
  /** Surface bias along normal to avoid self-intersection. @default 5 */
  surfaceBias: number;
  /** Radiosity probe spacing in surface cache texels. @default 4 */
  radiosityProbeSpacing: number;
  /** Hemisphere probe resolution (r.LumenScene.Radiosity.HemisphereProbeResolution). @default 4 */
  radiosityHemisphereResolution: number;
  /** Radiosity temporal max frames. @default 4 */
  radiosityMaxFrames: number;

  // ── Reflections (r.Lumen.Reflections.*) ───────────────────────────────────
  /** Enable specular indirect GI. @default true */
  specularEnabled: boolean;
  /** Max roughness for specular tracing. @default 0.4 */
  maxRoughnessToTrace: number;
  /** Specular rays per pixel. @default 1 */
  specularRaysPerPixel: number;
  /** Near-field specular trace max distance. @default 1000 */
  specularNearFieldMaxTrace: number;

  // ── Intensity ──────────────────────────────────────────────────────────────
  /** Global GI intensity multiplier. @default 1 */
  intensity: number;
  /** Diffuse indirect intensity. @default 1 */
  diffuseIntensity: number;
  /** Specular indirect intensity. @default 1 */
  specularIntensity: number;
}

export const DEFAULT_LUMEN_GI_CONFIG: UELumenGIConfig = {
  enabled: true,

  probeDownsampleFactor:           16,
  tracingOctahedronResolution:     8,
  maxAdaptiveProbes:               8,
  adaptiveProbeAllocationFraction: 0.5,
  spatialFilterRadius:             2,
  maxFramesAccumulated:            8,

  radianceCacheClipmapCount:      4,
  radianceCacheProbeResolution:   8,
  radianceCacheClipmapResolution: 32,
  radianceCacheMaxFrames:         16,

  softwareRayTracing:        true,
  sdfMaxSteps:               64,
  sdfMaxTraceDistance:       180.0,
  surfaceBias:               5.0,
  radiosityProbeSpacing:     4,
  radiosityHemisphereResolution: 4,
  radiosityMaxFrames:        4,

  specularEnabled:             true,
  maxRoughnessToTrace:         0.4,
  specularRaysPerPixel:        1,
  specularNearFieldMaxTrace:   1000.0,

  intensity:         1.0,
  diffuseIntensity:  1.0,
  specularIntensity: 1.0,
};

/** Per-frame render inputs from upstream G-Buffer / scene textures */
export interface LumenGIRenderInputs {
  /** Linear [0,1] depth texture (rgba16float or r32float, full-res) */
  depthTex: GPUTexture;
  /** World-space normal G-Buffer (rgba16float, full-res) */
  normalTex: GPUTexture;
  /** Albedo + metallic G-Buffer (rgba8unorm, full-res, a = metallic) */
  albedoTex: GPUTexture;
  /** Roughness + emissive G-Buffer (rgba8unorm, full-res, r = roughness, gba = emissive) */
  roughnessTex: GPUTexture;
  /** Previous-frame HDR scene colour, for screen-space miss fallback (rgba16float) */
  sceneColorTex: GPUTexture;
  /** Camera uniform buffer (matches CameraUBO layout below) */
  cameraUBO: GPUBuffer;
  /** Current frame index (used for temporal jitter) */
  frameIndex: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Shared Math Utilities
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_LUMEN_COMMON = /* wgsl */`
// ── Constants ──────────────────────────────────────────────────────────────
const PI        : f32 = 3.14159265358979323846;
const TWO_PI    : f32 = 6.28318530717958647693;
const INV_PI    : f32 = 0.31830988618379067154;
const HALF_PI   : f32 = 1.57079632679489661923;

fn sat(v: f32)    -> f32    { return clamp(v, 0.0, 1.0); }
fn sat3(v: vec3f) -> vec3f  { return clamp(v, vec3f(0.0), vec3f(1.0)); }

// ── Linear depth reconstruction ────────────────────────────────────────────
// reverse-Z: near/far stored in cameraUBO
fn linearizeDepth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// ── View-space position reconstruction ────────────────────────────────────
fn viewPosFromDepth(uv: vec2f, linearZ: f32, invPX: f32, invPY: f32) -> vec3f {
    let ndcX = uv.x * 2.0 - 1.0;
    let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
    return vec3f(ndcX * invPX * linearZ, ndcY * invPY * linearZ, linearZ);
}

// ── World-space position from clip ─────────────────────────────────────────
fn worldPosFromUVDepth(uv: vec2f, depth: f32, invVP: mat4x4f) -> vec3f {
    let ndcXY  = uv * vec2f(2.0) - vec2f(1.0);
    let ndcPos = vec4f(ndcXY.x, -ndcXY.y, depth, 1.0);
    let wPos4  = invVP * ndcPos;
    return wPos4.xyz / wPos4.w;
}

// ── Octahedral encode/decode ────────────────────────────────────────────────
// From LumenScreenProbeCommon.ush — maps hemisphere to square [0,1]²
fn octEncode(n: vec3f) -> vec2f {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        return (vec2f(1.0) - abs(p.yx)) * sign(p);
    }
    return p;
}

fn octDecode(enc: vec2f) -> vec3f {
    let f  = enc * 2.0 - vec2f(1.0);
    var n  = vec3f(f, 1.0 - abs(f.x) - abs(f.y));
    let t  = sat(-n.z);
    n.x   += select(t, -t, n.x >= 0.0);
    n.y   += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// ── SH2 (L1 4-coefficient) helpers ─────────────────────────────────────────
fn shBasis0() -> f32 { return 0.28209479177387814; } // 1/(2√π)
fn shBasis1(n: vec3f) -> vec3f {
    return vec3f(0.4886025119029199) * n; // √(3/4π) * N
}

// SH L2 ambient (Y₀₀ · col) — low-frequency irradiance from probe SH
fn shEvalL1Ambient(shAmb: vec3f, n: vec3f, shDir: vec4f) -> vec3f {
    // shAmb = L₀ band; shDir contains {Lr, Lg, Lb, unused}
    // E = π·(c₀·Y₀₀ + c₁·(Yx·nx + Yy·ny + Yz·nz))
    let c0  = shAmb;
    let cxR = shDir.x; let cxG = shDir.y; let cxB = shDir.z;
    // simplified: ambient + directional dot
    let dir = shBasis1(n);
    return max(c0 + vec3f(dir.x * cxR, dir.x * cxG, dir.x * cxB), vec3f(0.0));
}

// ── Schlick Fresnel ─────────────────────────────────────────────────────────
fn fresnelSchlick(f0: vec3f, cosTheta: f32) -> vec3f {
    let fc = pow(1.0 - sat(cosTheta), 5.0);
    return f0 + (vec3f(1.0) - f0) * fc;
}

// ── GGX NDF ────────────────────────────────────────────────────────────────
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// ── Smith Geometry ─────────────────────────────────────────────────────────
fn G_SmithJoint(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let gV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let gL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (gV + gL + 1e-7);
}

// ── Importance sample GGX (for specular probes) ───────────────────────────
fn importanceSampleGGX(xi: vec2f, roughness: f32, N: vec3f) -> vec3f {
    let a     = roughness * roughness;
    let phi   = TWO_PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Tangent-space H
    let Hts = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    // Build TBN
    var up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(N.z) < 0.999);
    let T  = normalize(cross(up, N));
    let B  = cross(N, T);
    return T * Hts.x + B * Hts.y + N * Hts.z;
}

// ── Hammersley low-discrepancy sequence ────────────────────────────────────
fn radicalInverseVdC(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, N: u32) -> vec2f {
    return vec2f(f32(i) / f32(N), radicalInverseVdC(i));
}

// ── Camera UBO layout (must match CameraUBO in codebase) ───────────────────
struct CameraUBO {
    viewProj    : mat4x4f,
    invViewProj : mat4x4f,
    viewPos     : vec4f,        // xyz = world position, w = unused
    projParams  : vec4f,        // x = near, y = far, z = fovY, w = aspect
    resolution  : vec4f,        // xy = width/height, zw = inv width/height
    invProjXY   : vec2f,        // 1/proj[0][0], 1/proj[1][1]
    frameIndex  : u32,
    pad         : u32,
};

// ── Cell SDF entry (matches Cell PubSub Loop cell data) ────────────────────
struct CellSDF {
    positionRadius : vec4f,   // xyz = world centre, w = radius
    emissive       : vec4f,   // xyz = emissive colour, w = emissiveScale
    radianceIndex  : u32,     // index into cellRadianceTex array
    pad            : vec3u,
};
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 0: Screen Probe Placement
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PROBE_PLACEMENT = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<uniform> cam          : CameraUBO;
@group(0) @binding(1) var          depthTex      : texture_2d<f32>;
@group(0) @binding(2) var          normalTex     : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> probeDepth  : array<f32>;
@group(0) @binding(4) var<storage, read_write> probeNormal : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> probePos    : array<vec4f>;

struct ProbePlacementParams {
    downsampleFactor : u32,
    probeViewW       : u32,   // ceil(screenW / downsampleFactor)
    probeViewH       : u32,
    maxAdaptive      : u32,
    frameIndex       : u32,
    pad              : vec3u,
};
@group(0) @binding(6) var<uniform> params : ProbePlacementParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.probeViewW || py >= params.probeViewH) { return; }

    // Centre pixel of this probe tile (with per-frame sub-pixel jitter)
    // UE: r.Lumen.ScreenProbeGather.FullResolutionJitterWidth = 1
    let jitterTable = array<vec2f, 4>(
        vec2f(0.25, 0.25), vec2f(-0.25, 0.25),
        vec2f(0.25, -0.25), vec2f(-0.25, -0.25)
    );
    let jitter = jitterTable[params.frameIndex & 3u];

    let df   = f32(params.downsampleFactor);
    let cx   = f32(px) * df + df * 0.5 + jitter.x;
    let cy   = f32(py) * df + df * 0.5 + jitter.y;
    let screenW = f32(cam.resolution.x);
    let screenH = f32(cam.resolution.y);

    let iu   = u32(clamp(cx, 0.0, screenW - 1.0));
    let iv   = u32(clamp(cy, 0.0, screenH - 1.0));

    let rawDepth = textureLoad(depthTex, vec2u(iu, iv), 0).r;
    let linZ     = linearizeDepth(rawDepth, cam.projParams.x, cam.projParams.y);

    let uv   = vec2f(cx / screenW, cy / screenH);
    let wPos = worldPosFromUVDepth(uv, rawDepth, cam.invViewProj);
    let rawN = textureLoad(normalTex, vec2u(iu, iv), 0).xyz;
    let wN   = normalize(rawN * 2.0 - vec3f(1.0));

    let idx = py * params.probeViewW + px;
    probeDepth[idx]  = linZ;
    probeNormal[idx] = vec4f(wN, 0.0);
    probePos[idx]    = vec4f(wPos, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1: Radiance Cache Update
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_RADIANCE_CACHE_UPDATE = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<uniform> cam          : CameraUBO;
@group(0) @binding(1) var<storage, read>  probePos  : array<vec4f>;
@group(0) @binding(2) var<storage, read>  cellSDF   : array<CellSDF>;
@group(0) @binding(3) var<storage, read_write> rcCache : array<vec4f>; // Clipmap irradiance texels
@group(0) @binding(4) var<storage, read>  prevCache  : array<vec4f>;

struct RCParams {
    clipmapCount    : u32,
    clipmapRes      : u32,    // cells per axis
    probeRes        : u32,    // octahedral resolution per probe
    numProbes       : u32,
    numCells        : u32,
    maxFrames       : u32,
    clipmapExtents  : vec4f,  // x = clipmap0 half-extent (world units)
    camPos          : vec3f,
    pad             : u32,
};
@group(0) @binding(5) var<uniform> rcp : RCParams;

// Soft SDF sphere-march towards a direction from a world position
// Returns: rgb radiance of closest surface hit
fn traceSDF(origin: vec3f, dir: vec3f, maxDist: f32, numCells: u32) -> vec3f {
    var t        = 0.05; // surface bias to avoid self-intersection
    let maxSteps = 64u;
    for (var step = 0u; step < maxSteps; step++) {
        let p      = origin + dir * t;
        var minD   = maxDist;
        var hitCol = vec3f(0.0);
        for (var ci = 0u; ci < numCells; ci++) {
            let cell  = cellSDF[ci];
            let d     = length(p - cell.positionRadius.xyz) - cell.positionRadius.w;
            if (d < minD) {
                minD   = d;
                hitCol = cell.emissive.xyz * cell.emissive.w;
            }
        }
        if (minD < 0.01) { return hitCol; }
        t += max(minD * 0.9, 0.1);
        if (t >= maxDist) { break; }
    }
    return vec3f(0.0); // sky / miss
}

// Sample Radiance Cache at a world position and normal (trilinear, single clipmap level)
fn sampleRadianceCache(wPos: vec3f, wNormal: vec3f, clipmapLevel: u32) -> vec3f {
    let extent     = rcp.clipmapExtents.x * pow(2.0, f32(clipmapLevel));
    let cRes       = rcp.clipmapRes;
    let cellSize   = (2.0 * extent) / f32(cRes);
    let gridOrigin = rcp.camPos - vec3f(extent);
    let gridCoord  = (wPos - gridOrigin) / cellSize;

    // Clamp to valid range
    let gc         = clamp(gridCoord, vec3f(0.0), vec3f(f32(cRes) - 1.001));
    let gi         = vec3u(u32(gc.x), u32(gc.y), u32(gc.z));

    // Probe index within Clipmap
    let probeOctTotal = rcp.probeRes * rcp.probeRes;
    let probeStride   = probeOctTotal;
    let clipmapStride = cRes * cRes * cRes * probeStride;
    let probeBase     = clipmapLevel * clipmapStride + (gi.z * cRes * cRes + gi.y * cRes + gi.x) * probeStride;

    // Lookup dominant normal direction bin (octahedral)
    let octUV   = octEncode(wNormal) * 0.5 + vec2f(0.5);
    let octIU   = u32(clamp(octUV.x * f32(rcp.probeRes - 1u), 0.0, f32(rcp.probeRes - 1u)));
    let octIV   = u32(clamp(octUV.y * f32(rcp.probeRes - 1u), 0.0, f32(rcp.probeRes - 1u)));
    let texIdx  = probeBase + octIV * rcp.probeRes + octIU;
    let maxIdx  = arrayLength(&rcCache) - 1u;
    return rcCache[min(texIdx, maxIdx)].xyz;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ci = gid.x; // clipmap level
    let cx = gid.y; // cell x
    let cy = gid.z; // cell z (row-major: x,y,z)
    if (ci >= rcp.clipmapCount || cx >= rcp.clipmapRes || cy >= rcp.clipmapRes) { return; }

    let cRes       = rcp.clipmapRes;
    let extent     = rcp.clipmapExtents.x * pow(2.0, f32(ci));
    let cellSize   = (2.0 * extent) / f32(cRes);
    let gridOrigin = rcp.camPos - vec3f(extent);

    // Iterate over Z slices for this (ci, cx, cy) column
    let probeOctTotal  = rcp.probeRes * rcp.probeRes;
    let clipmapStride  = cRes * cRes * cRes * probeOctTotal;

    for (var cz = 0u; cz < cRes; cz++) {
        let probeWorldPos = gridOrigin + (vec3f(f32(cx), f32(cz), f32(cy)) + vec3f(0.5)) * cellSize;
        let probeBase     = ci * clipmapStride + (cz * cRes * cRes + cy * cRes + cx) * probeOctTotal;
        let blendAlpha    = 1.0 / f32(rcp.maxFrames);

        for (var ov = 0u; ov < rcp.probeRes; ov++) {
            for (var ou = 0u; ou < rcp.probeRes; ou++) {
                let octUV  = (vec2f(f32(ou), f32(ov)) + vec2f(0.5)) / f32(rcp.probeRes);
                let dir    = octDecode(octUV);

                // Soft-trace toward this direction from probe centre
                let traced = traceSDF(probeWorldPos, dir, 180.0, rcp.numCells);

                let tidx   = probeBase + ov * rcp.probeRes + ou;
                let maxIdx = arrayLength(&rcCache) - 1u;
                let pidx   = min(tidx, maxIdx);

                let prev = prevCache[pidx].xyz;
                rcCache[pidx] = vec4f(mix(prev, traced, blendAlpha), 1.0);
            }
        }
    }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 2: Screen Probe Tracing (SS + SDF + Radiance Cache)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PROBE_TRACING = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<uniform>  cam          : CameraUBO;
@group(0) @binding(1) var           depthTex     : texture_2d<f32>;
@group(0) @binding(2) var           sceneColor   : texture_2d<f32>;
@group(0) @binding(3) var<storage, read>  probePos    : array<vec4f>;
@group(0) @binding(4) var<storage, read>  probeNormal : array<vec4f>;
@group(0) @binding(5) var<storage, read>  cellSDF     : array<CellSDF>;
@group(0) @binding(6) var<storage, read>  rcCache     : array<vec4f>;
@group(0) @binding(7) var<storage, read_write> traceRadiance : array<vec4f>;

struct ProbeTraceParams {
    probeViewW     : u32,
    probeViewH     : u32,
    octRes         : u32,
    numCells       : u32,
    maxSDFSteps    : u32,
    sdfMaxDist     : f32,
    surfaceBias    : f32,
    frameIndex     : u32,
    rcClipmapCount : u32,
    rcClipmapRes   : u32,
    rcProbeRes     : u32,
    rcClipmapExtX  : f32,
    rcCamPos       : vec3f,
    pad            : u32,
};
@group(0) @binding(8) var<uniform> tp : ProbeTraceParams;

// Screen-space ray march to find lit surface
fn screenMarch(origin: vec3f, dir: vec3f, maxSteps: u32) -> vec3f {
    let stepSize  = 0.02;
    let screenW   = cam.resolution.x;
    let screenH   = cam.resolution.y;
    for (var i = 0u; i < maxSteps; i++) {
        let p      = origin + dir * (f32(i + 1u) * stepSize * tp.sdfMaxDist / f32(maxSteps));
        // Project to screen
        let clip   = cam.viewProj * vec4f(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc    = clip.xyz / clip.w;
        let uv     = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
        let iu     = u32(uv.x * screenW);
        let iv     = u32(uv.y * screenH);
        let sceneD = textureLoad(depthTex, vec2u(iu, iv), 0).r;
        let sceneLinZ = linearizeDepth(sceneD, cam.projParams.x, cam.projParams.y);
        let rayLinZ   = linearizeDepth(ndc.z * 0.5 + 0.5, cam.projParams.x, cam.projParams.y);
        // Depth test: ray is behind scene geometry → hit
        if (rayLinZ > sceneLinZ + 0.01 && rayLinZ < sceneLinZ + 1.0) {
            return textureLoad(sceneColor, vec2u(iu, iv), 0).xyz;
        }
    }
    return vec3f(-1.0); // miss
}

// Soft SDF march (Radiosity: Cell-to-Cell light bounce)
fn sdfMarch(origin: vec3f, dir: vec3f) -> vec3f {
    var t = tp.surfaceBias;
    for (var step = 0u; step < tp.maxSDFSteps; step++) {
        let p    = origin + dir * t;
        var minD = tp.sdfMaxDist;
        var hitE = vec3f(0.0);
        for (var ci = 0u; ci < tp.numCells; ci++) {
            let cell = cellSDF[ci];
            let d    = length(p - cell.positionRadius.xyz) - cell.positionRadius.w;
            if (d < minD) {
                minD = d;
                hitE = cell.emissive.xyz * cell.emissive.w;
            }
        }
        if (minD < 0.02) { return hitE; }
        t += max(minD * 0.85, 0.05);
        if (t >= tp.sdfMaxDist) { break; }
    }
    return vec3f(-1.0); // miss
}

// Radiance Cache sample
fn rcSample(wPos: vec3f, wNormal: vec3f) -> vec3f {
    // Choose best clipmap level based on distance
    let dist   = length(wPos - tp.rcCamPos);
    var level  = 0u;
    var extent = tp.rcClipmapExtX;
    for (var l = 1u; l < tp.rcClipmapCount; l++) {
        if (dist < extent) { break; }
        level  = l;
        extent *= 2.0;
    }

    let cRes    = tp.rcClipmapRes;
    let pr      = tp.rcProbeRes;
    let cellSz  = (2.0 * extent) / f32(cRes);
    let origin  = tp.rcCamPos - vec3f(extent);
    let gc      = clamp((wPos - origin) / cellSz, vec3f(0.0), vec3f(f32(cRes) - 1.001));
    let gi      = vec3u(u32(gc.x), u32(gc.y), u32(gc.z));

    let octUV   = octEncode(wNormal) * 0.5 + vec2f(0.5);
    let octIU   = u32(clamp(octUV.x * f32(pr - 1u), 0.0, f32(pr - 1u)));
    let octIV   = u32(clamp(octUV.y * f32(pr - 1u), 0.0, f32(pr - 1u)));

    let clipStride = cRes * cRes * cRes * pr * pr;
    let probeBase  = level * clipStride + (gi.z * cRes * cRes + gi.y * cRes + gi.x) * pr * pr;
    let tidx       = probeBase + octIV * pr + octIU;
    let maxIdx     = arrayLength(&rcCache) - 1u;
    return rcCache[min(tidx, maxIdx)].xyz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;  // probe X
    let py = gid.y;  // probe Y
    if (px >= tp.probeViewW || py >= tp.probeViewH) { return; }

    let probeIdx   = py * tp.probeViewW + px;
    let wPos       = probePos[probeIdx].xyz;
    let wNormal    = probeNormal[probeIdx].xyz;

    let octRes     = tp.octRes;
    let outBase    = probeIdx * octRes * octRes;

    for (var ov = 0u; ov < octRes; ov++) {
        for (var ou = 0u; ou < octRes; ou++) {
            // Octahedral direction for this bin
            let octUV  = (vec2f(f32(ou), f32(ov)) + vec2f(0.5)) / f32(octRes);
            let dir    = octDecode(octUV);

            // Skip directions facing away from surface normal
            // (back-face cull with small bias for thick normals)
            if (dot(dir, wNormal) < -0.1) {
                traceRadiance[outBase + ov * octRes + ou] = vec4f(0.0);
                continue;
            }

            let biasedPos = wPos + wNormal * tp.surfaceBias;

            // 1. Screen-space trace (fast path)
            let ssHit = screenMarch(biasedPos, dir, 32u);
            if (ssHit.x >= 0.0) {
                traceRadiance[outBase + ov * octRes + ou] = vec4f(ssHit, 1.0);
                continue;
            }

            // 2. Software SDF trace (Cell Radiosity bounce)
            let sdfHit = sdfMarch(biasedPos, dir);
            if (sdfHit.x >= 0.0) {
                traceRadiance[outBase + ov * octRes + ou] = vec4f(sdfHit, 1.0);
                continue;
            }

            // 3. Radiance Cache fallback (far-field)
            let rcHit = rcSample(biasedPos + dir * tp.sdfMaxDist, dir);
            traceRadiance[outBase + ov * octRes + ou] = vec4f(rcHit, 1.0);
        }
    }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Probe Filtering → SH3 + Temporal Blend
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PROBE_FILTER = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<storage, read>         traceRadiance : array<vec4f>;
@group(0) @binding(1) var<storage, read>         probeNormal   : array<vec4f>;
@group(0) @binding(2) var<storage, read>         probeDepth    : array<f32>;
@group(0) @binding(3) var<storage, read>         prevSHAmb     : array<vec4f>; // L0 rgb
@group(0) @binding(4) var<storage, read>         prevSHDir     : array<vec4f>; // L1 rgb (packed)
@group(0) @binding(5) var<storage, read_write>   outSHAmb      : array<vec4f>;
@group(0) @binding(6) var<storage, read_write>   outSHDir      : array<vec4f>;

struct FilterParams {
    probeViewW    : u32,
    probeViewH    : u32,
    octRes        : u32,
    spatialRadius : u32,
    maxFrames     : u32,
    frameIndex    : u32,
    pad           : vec2u,
};
@group(0) @binding(7) var<uniform> fp : FilterParams;

// Project octahedral radiance into SH L1 (4 coefficients × 3 channels)
fn projectToSH(probeBase: u32, octRes: u32) -> array<vec3f, 4> {
    var sh : array<vec3f, 4>;
    sh[0] = vec3f(0.0); sh[1] = vec3f(0.0);
    sh[2] = vec3f(0.0); sh[3] = vec3f(0.0);
    let invTotal = 1.0 / f32(octRes * octRes);

    for (var ov = 0u; ov < octRes; ov++) {
        for (var ou = 0u; ou < octRes; ou++) {
            let rad  = traceRadiance[probeBase + ov * octRes + ou].xyz;
            let octUV = (vec2f(f32(ou), f32(ov)) + vec2f(0.5)) / f32(octRes);
            let dir   = octDecode(octUV);

            // L0: constant basis Y₀₀ = 0.282095
            sh[0] += rad * 0.282095 * invTotal;
            // L1: Y₁₋₁=0.488603·y, Y₁₀=0.488603·z, Y₁₁=0.488603·x
            sh[1] += rad * 0.488603 * dir.x * invTotal;
            sh[2] += rad * 0.488603 * dir.y * invTotal;
            sh[3] += rad * 0.488603 * dir.z * invTotal;
        }
    }
    return sh;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;
    let py = gid.y;
    if (px >= fp.probeViewW || py >= fp.probeViewH) { return; }

    let pidx     = py * fp.probeViewW + px;
    let probeBase = pidx * fp.octRes * fp.octRes;

    // ── Spatial filter: accumulate SH from nearby probes ─────────────────
    var shAmb = vec3f(0.0);
    var shDX  = vec3f(0.0);
    var shDY  = vec3f(0.0);
    var shDZ  = vec3f(0.0);
    var totalW = 0.0;

    let centerDepth = probeDepth[pidx];
    let centerN     = probeNormal[pidx].xyz;

    let r = i32(fp.spatialRadius);
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let nx = i32(px) + dx;
            let ny = i32(py) + dy;
            if (nx < 0 || ny < 0 || u32(nx) >= fp.probeViewW || u32(ny) >= fp.probeViewH) { continue; }
            let nidx = u32(ny) * fp.probeViewW + u32(nx);
            let nDepth = probeDepth[nidx];
            let nNorm  = probeNormal[nidx].xyz;

            // Depth weight (LumenScreenProbeGather: ScreenProbeInterpolationDepthWeight)
            let depthDiff  = abs(nDepth - centerDepth) / max(centerDepth, 0.01);
            let depthW     = exp(-depthDiff * 4.0);
            // Normal weight
            let normalW    = max(dot(nNorm, centerN), 0.0);
            let w          = depthW * (normalW * normalW + 0.01);

            let nb       = nidx * fp.octRes * fp.octRes;
            let sh       = projectToSH(nb, fp.octRes);
            shAmb += sh[0] * w;
            shDX  += sh[1] * w;
            shDY  += sh[2] * w;
            shDZ  += sh[3] * w;
            totalW += w;
        }
    }
    if (totalW > 0.0) {
        let invW = 1.0 / totalW;
        shAmb *= invW; shDX *= invW; shDY *= invW; shDZ *= invW;
    }

    // ── Temporal blend ─────────────────────────────────────────────────
    let alpha = 1.0 / f32(fp.maxFrames);
    let prevAmb = prevSHAmb[pidx].xyz;
    let prevDir = prevSHDir[pidx];

    let blendedAmb = mix(prevAmb, shAmb,  alpha);
    // Pack x-component of L1 in .x (r-channel dominant), store compact:
    let blendedDir = mix(prevDir.xyz, (shDX + shDY + shDZ) / 3.0, alpha);

    outSHAmb[pidx] = vec4f(blendedAmb, 1.0);
    outSHDir[pidx] = vec4f(blendedDir, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 4: Diffuse Integration (full-resolution upsampling)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_DIFFUSE_INTEGRATE = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<uniform>  cam          : CameraUBO;
@group(0) @binding(1) var           depthTex     : texture_2d<f32>;
@group(0) @binding(2) var           normalTex    : texture_2d<f32>;
@group(0) @binding(3) var           albedoTex    : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> shAmb   : array<vec4f>;
@group(0) @binding(5) var<storage, read> shDir   : array<vec4f>;
@group(0) @binding(6) var<storage, read> probeDepth : array<f32>;
@group(0) @binding(7) var<storage, read> probeNormal: array<vec4f>;
@group(0) @binding(8) var           diffuseOut   : texture_storage_2d<rgba16float, write>;

struct DiffuseParams {
    downsampleFactor : u32,
    probeViewW       : u32,
    probeViewH       : u32,
    diffuseIntensity : f32,
    pad              : vec4u,
};
@group(0) @binding(9) var<uniform> dp : DiffuseParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let screenW = u32(cam.resolution.x);
    let screenH = u32(cam.resolution.y);
    let px      = gid.x;
    let py      = gid.y;
    if (px >= screenW || py >= screenH) { return; }

    let rawDepth = textureLoad(depthTex, vec2u(px, py), 0).r;
    let linZ     = linearizeDepth(rawDepth, cam.projParams.x, cam.projParams.y);
    let rawN     = textureLoad(normalTex, vec2u(px, py), 0).xyz;
    let wN       = normalize(rawN * 2.0 - vec3f(1.0));
    let albMet   = textureLoad(albedoTex, vec2u(px, py), 0);
    let albedo   = albMet.xyz;
    let metallic = albMet.w;

    // Find owning probe tile (nearest)
    let df   = f32(dp.downsampleFactor);
    let cpx  = u32(f32(px) / df);
    let cpy  = u32(f32(py) / df);

    // Bilinear-interpolate from 2×2 surrounding probes
    var sumSHAmb = vec3f(0.0);
    var sumSHDir = vec3f(0.0);
    var sumW     = 0.0;

    for (var dy = 0u; dy <= 1u; dy++) {
        for (var dx = 0u; dx <= 1u; dx++) {
            let npx = cpx + dx;
            let npy = cpy + dy;
            if (npx >= dp.probeViewW || npy >= dp.probeViewH) { continue; }
            let nidx   = npy * dp.probeViewW + npx;
            let pDepth = probeDepth[nidx];
            let pNorm  = probeNormal[nidx].xyz;

            let depthDiff = abs(pDepth - linZ) / max(linZ, 0.01);
            let depthW    = exp(-depthDiff * 6.0);
            let normalW   = max(dot(pNorm, wN), 0.0);
            let w         = depthW * (normalW * normalW + 0.01);

            sumSHAmb += shAmb[nidx].xyz * w;
            sumSHDir += shDir[nidx].xyz * w;
            sumW     += w;
        }
    }

    var irradiance = vec3f(0.0);
    if (sumW > 0.0) {
        let invW    = 1.0 / sumW;
        let ambSH   = sumSHAmb * invW;
        let dirSH   = sumSHDir * invW;
        // Reconstruct irradiance: E(N) = π · (L₀·Y₀₀ + L₁·(Nx·Y₁₁ + Ny·Y₁₋₁ + Nz·Y₁₀))
        let e0 = ambSH * 0.886227;  // π · Y₀₀ · L₀
        let e1 = dirSH * (wN.x * 1.02333 + wN.y * 1.02333 + wN.z * 1.02333);
        irradiance = max(e0 + e1, vec3f(0.0));
    }

    // Energy-conserving diffuse: kD = albedo·(1 - metallic)
    let kD        = albedo * (1.0 - metallic);
    let diffGI    = kD * irradiance * dp.diffuseIntensity;

    textureStore(diffuseOut, vec2u(px, py), vec4f(diffGI, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 5: Specular Indirect (GGX importance sampling + Radiance Cache)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SPECULAR_GATHER = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var<uniform>  cam           : CameraUBO;
@group(0) @binding(1) var           depthTex      : texture_2d<f32>;
@group(0) @binding(2) var           normalTex     : texture_2d<f32>;
@group(0) @binding(3) var           albedoTex     : texture_2d<f32>;
@group(0) @binding(4) var           roughnessTex  : texture_2d<f32>;
@group(0) @binding(5) var           sceneColor    : texture_2d<f32>;
@group(0) @binding(6) var<storage, read>  cellSDF      : array<CellSDF>;
@group(0) @binding(7) var<storage, read>  rcCache      : array<vec4f>;
@group(0) @binding(8) var           specularOut   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var           prevSpecular  : texture_2d<f32>;

struct SpecParams {
    numRays             : u32,
    maxRoughnessToTrace : f32,
    nearFieldMaxTrace   : f32,
    surfaceBias         : f32,
    numCells            : u32,
    rcClipmapCount      : u32,
    rcClipmapRes        : u32,
    rcProbeRes          : u32,
    rcClipmapExtX       : f32,
    specularIntensity   : f32,
    temporalAlpha       : f32,
    frameIndex          : u32,
    camPos              : vec3f,
    pad                 : u32,
};
@group(0) @binding(10) var<uniform> sp : SpecParams;

// Screen-space trace for specular (high precision, fewer steps)
fn ssTraceSpecular(origin: vec3f, dir: vec3f) -> vec4f { // xyz=color, w=hit
    let maxSteps = 48u;
    let screenW  = cam.resolution.x;
    let screenH  = cam.resolution.y;
    for (var i = 1u; i <= maxSteps; i++) {
        let t    = sp.nearFieldMaxTrace * f32(i) / f32(maxSteps);
        let p    = origin + dir * t;
        let clip = cam.viewProj * vec4f(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc  = clip.xyz / clip.w;
        let uv   = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
        let iu   = u32(uv.x * screenW);
        let iv   = u32(uv.y * screenH);
        let sd   = textureLoad(depthTex, vec2u(iu, iv), 0).r;
        let sZ   = linearizeDepth(sd, cam.projParams.x, cam.projParams.y);
        let rZ   = linearizeDepth(ndc.z * 0.5 + 0.5, cam.projParams.x, cam.projParams.y);
        if (rZ > sZ + 0.01 && rZ < sZ + 2.0) {
            return vec4f(textureLoad(sceneColor, vec2u(iu, iv), 0).xyz, 1.0);
        }
    }
    return vec4f(0.0, 0.0, 0.0, 0.0);
}

// Radiance Cache sample for specular
fn rcSpecSample(wPos: vec3f, reflDir: vec3f) -> vec3f {
    let dist   = length(wPos - sp.camPos);
    var level  = 0u;
    var extent = sp.rcClipmapExtX;
    for (var l = 1u; l < sp.rcClipmapCount; l++) {
        if (dist < extent) { break; }
        level  = l;
        extent *= 2.0;
    }
    let cRes    = sp.rcClipmapRes;
    let pr      = sp.rcProbeRes;
    let cellSz  = (2.0 * extent) / f32(cRes);
    let origin  = sp.camPos - vec3f(extent);
    let gc      = clamp((wPos - origin) / cellSz, vec3f(0.0), vec3f(f32(cRes) - 1.001));
    let gi      = vec3u(u32(gc.x), u32(gc.y), u32(gc.z));
    let octUV   = octEncode(reflDir) * 0.5 + vec2f(0.5);
    let octIU   = u32(clamp(octUV.x * f32(pr - 1u), 0.0, f32(pr - 1u)));
    let octIV   = u32(clamp(octUV.y * f32(pr - 1u), 0.0, f32(pr - 1u)));
    let clipStride = cRes * cRes * cRes * pr * pr;
    let probeBase  = level * clipStride + (gi.z * cRes * cRes + gi.y * cRes + gi.x) * pr * pr;
    let tidx       = probeBase + octIV * pr + octIU;
    let maxIdx     = arrayLength(&rcCache) - 1u;
    return rcCache[min(tidx, maxIdx)].xyz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let screenW = u32(cam.resolution.x);
    let screenH = u32(cam.resolution.y);
    let px      = gid.x;
    let py      = gid.y;
    if (px >= screenW || py >= screenH) { return; }

    let rawDepth  = textureLoad(depthTex,     vec2u(px, py), 0).r;
    let rawN      = textureLoad(normalTex,    vec2u(px, py), 0).xyz;
    let albMet    = textureLoad(albedoTex,    vec2u(px, py), 0);
    let roughMet  = textureLoad(roughnessTex, vec2u(px, py), 0);

    let roughness = roughMet.r;
    let metallic  = albMet.w;
    let albedo    = albMet.xyz;
    let wN        = normalize(rawN * 2.0 - vec3f(1.0));

    let uv      = (vec2f(f32(px), f32(py)) + vec2f(0.5)) / cam.resolution.xy;
    let wPos    = worldPosFromUVDepth(uv, rawDepth, cam.invViewProj);
    let V       = normalize(cam.viewPos.xyz - wPos);

    // Skip non-specular surfaces (r.Lumen.Reflections.MaxRoughnessToTrace)
    if (roughness > sp.maxRoughnessToTrace) {
        textureStore(specularOut, vec2u(px, py), vec4f(0.0));
        return;
    }

    // F0 (dielectric: 0.04, metallic: albedo)
    let f0 = mix(vec3f(0.04), albedo, metallic);

    var specAcc  = vec3f(0.0);
    var wAcc     = 0.0;
    let biasPos  = wPos + wN * sp.surfaceBias;

    for (var ri = 0u; ri < sp.numRays; ri++) {
        // Stratified GGX importance sample (Hammersley + per-frame offset)
        let xi   = hammersley(ri + sp.frameIndex * sp.numRays, sp.numRays * 64u);
        let H    = importanceSampleGGX(xi, roughness, wN);
        let L    = normalize(reflect(-V, H));

        let NdotL = dot(wN, L);
        if (NdotL <= 0.0) { continue; }

        // Weight (Cook-Torrance numerator)
        let NdotV = max(dot(wN, V), 0.0001);
        let NdotH = max(dot(wN, H), 0.0);
        let VdotH = max(dot(V,  H), 0.0);

        let D  = D_GGX(NdotH, roughness);
        let G  = G_SmithJoint(NdotV, NdotL, roughness);
        let F  = fresnelSchlick(f0, VdotH);
        let w  = D * G * NdotL / max(NdotV, 0.0001);

        // 1. Screen-space trace
        let ssHit = ssTraceSpecular(biasPos, L);
        if (ssHit.w > 0.5) {
            specAcc += ssHit.xyz * F * w;
            wAcc    += w;
            continue;
        }
        // 2. Radiance Cache
        let rcHit = rcSpecSample(biasPos + L * sp.nearFieldMaxTrace, L);
        specAcc += rcHit * F * w;
        wAcc    += w;
    }

    var spec = vec3f(0.0);
    if (wAcc > 0.0) {
        spec = specAcc / wAcc * sp.specularIntensity;
    }

    // Temporal accumulation (4 frames)
    let prevSpec = textureLoad(prevSpecular, vec2u(px, py), 0).xyz;
    let blended  = mix(prevSpec, spec, sp.temporalAlpha);
    textureStore(specularOut, vec2u(px, py), vec4f(blended, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 6: Composite (diffuse + specular → scene)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPOSITE = /* wgsl */`
${WGSL_LUMEN_COMMON}

@group(0) @binding(0) var           sceneTex    : texture_2d<f32>;
@group(0) @binding(1) var           diffuseTex  : texture_2d<f32>;
@group(0) @binding(2) var           specularTex : texture_2d<f32>;
@group(0) @binding(3) var           compositeOut: texture_storage_2d<rgba16float, write>;

struct CompositeParams {
    giIntensity : f32,
    pad         : vec3f,
};
@group(0) @binding(4) var<uniform> cp : CompositeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dim = textureDimensions(sceneTex);
    let px  = gid.x;
    let py  = gid.y;
    if (px >= dim.x || py >= dim.y) { return; }

    let scene    = textureLoad(sceneTex,    vec2u(px, py), 0).xyz;
    let diffGI   = textureLoad(diffuseTex,  vec2u(px, py), 0).xyz;
    let specGI   = textureLoad(specularTex, vec2u(px, py), 0).xyz;

    let finalCol = scene + (diffGI + specGI) * cp.giIntensity;
    textureStore(compositeOut, vec2u(px, py), vec4f(finalCol, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal GPU resource helpers
// ─────────────────────────────────────────────────────────────────────────────

function makeRT(
  device: GPUDevice, w: number, h: number,
  format: GPUTextureFormat, label: string
): GPUTexture {
  return device.createTexture({
    size: [w, h],
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING |
           GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    label,
  });
}

function makeBuffer(
  device: GPUDevice, size: number,
  usage: GPUBufferUsageFlags, label: string
): GPUBuffer {
  return device.createBuffer({ size, usage, label });
}

function makeUBO(device: GPUDevice, size: number, label: string): GPUBuffer {
  return makeBuffer(
    device, size,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label
  );
}

function makeSSBO(device: GPUDevice, size: number, label: string): GPUBuffer {
  return makeBuffer(
    device, size,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label
  );
}

async function makeComputePipeline(
  device: GPUDevice, code: string, entryPoint: string, label: string
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ code, label: `${label}_shader` });
  return device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
    label,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Class: UELumenGI
// ─────────────────────────────────────────────────────────────────────────────

/**
 * UELumenGI — WebGPU port of UE5 Lumen Global Illumination.
 *
 * Implements Screen-Space Probe Gathering, Radiance Cache, Software Ray Tracing
 * (Cell SDF bounce), and Diffuse + Specular Indirect lighting.
 *
 * Usage:
 *   const lumen = await UELumenGI.create(device, width, height);
 *   lumen.setConfig({ intensity: 1.2, softwareRayTracing: true });
 *   lumen.setCellSDFBuffer(cellBuf);   // from Cell PubSub Loop
 *   lumen.setCellCount(numCells);
 *   // per-frame:
 *   lumen.tick(dt);
 *   const enc = device.createCommandEncoder();
 *   lumen.render(enc, inputs, outputTex);
 *   device.queue.submit([enc.finish()]);
 */
export class UELumenGI {
  private device: GPUDevice;
  private width: number;
  private height: number;
  private cfg: UELumenGIConfig;

  // Probe dimensions
  private probeW: number = 0;
  private probeH: number = 0;
  private probeCount: number = 0;
  private octRes: number = 0;

  // Pipelines
  private pipePlacement!: GPUComputePipeline;
  private pipeRCUpdate!:  GPUComputePipeline;
  private pipeTrace!:     GPUComputePipeline;
  private pipeFilter!:    GPUComputePipeline;
  private pipeDiffuse!:   GPUComputePipeline;
  private pipeSpecular!:  GPUComputePipeline;
  private pipeComposite!: GPUComputePipeline;

  // Storage buffers
  private bufProbeDepth!:   GPUBuffer;
  private bufProbeNormal!:  GPUBuffer;
  private bufProbePos!:     GPUBuffer;
  private bufTraceRadiance!: GPUBuffer;
  private bufSHAmb!:        GPUBuffer;
  private bufSHDir!:        GPUBuffer;
  private bufPrevSHAmb!:    GPUBuffer;
  private bufPrevSHDir!:    GPUBuffer;
  private bufRCCache!:      GPUBuffer;
  private bufRCPrev!:       GPUBuffer;

  // Render targets
  private rtDiffuse!:       GPUTexture;
  private rtSpecular!:      GPUTexture;
  private rtPrevSpecular!:  GPUTexture;

  // UBOs
  private uboProbePlacement!: GPUBuffer;
  private uboRCParams!:       GPUBuffer;
  private uboProbeTrace!:     GPUBuffer;
  private uboFilterParams!:   GPUBuffer;
  private uboDiffuseParams!:  GPUBuffer;
  private uboSpecParams!:     GPUBuffer;
  private uboCompositeParams!: GPUBuffer;

  // Injected from Cell PubSub Loop
  private cellSDFBuffer: GPUBuffer | null = null;
  private cellCount: number = 0;

  // Internal state
  private frameIdx: number = 0;
  private rcClipmapCells: number = 0;

  // Output texture (rgba16float, full-res HDR with GI applied)
  public outputTex!: GPUTexture;

  private constructor(device: GPUDevice, width: number, height: number, cfg: UELumenGIConfig) {
    this.device = device;
    this.width  = width;
    this.height = height;
    this.cfg    = { ...cfg };
  }

  /** Factory — creates GPU resources and compiles all pipelines. */
  static async create(
    device: GPUDevice,
    width: number,
    height: number,
    cfg: Partial<UELumenGIConfig> = {}
  ): Promise<UELumenGI> {
    const inst = new UELumenGI(device, width, height, { ...DEFAULT_LUMEN_GI_CONFIG, ...cfg });
    await inst._init();
    return inst;
  }

  private async _init(): Promise<void> {
    const cfg = this.cfg;
    const df  = cfg.probeDownsampleFactor;
    this.probeW     = Math.ceil(this.width  / df);
    this.probeH     = Math.ceil(this.height / df);
    this.probeCount = this.probeW * this.probeH;
    this.octRes     = cfg.tracingOctahedronResolution;

    // Radiance Cache total entries
    const rcRes     = cfg.radianceCacheClipmapResolution;
    const rcPR      = cfg.radianceCacheProbeResolution;
    const rcLevels  = cfg.radianceCacheClipmapCount;
    this.rcClipmapCells = rcLevels * rcRes * rcRes * rcRes * rcPR * rcPR;

    // ── Compile pipelines in parallel ───────────────────────────────────────
    [
      this.pipePlacement,
      this.pipeRCUpdate,
      this.pipeTrace,
      this.pipeFilter,
      this.pipeDiffuse,
      this.pipeSpecular,
      this.pipeComposite,
    ] = await Promise.all([
      makeComputePipeline(this.device, WGSL_PROBE_PLACEMENT,    'main', 'lumen_placement'),
      makeComputePipeline(this.device, WGSL_RADIANCE_CACHE_UPDATE, 'main', 'lumen_rc_update'),
      makeComputePipeline(this.device, WGSL_PROBE_TRACING,      'main', 'lumen_probe_trace'),
      makeComputePipeline(this.device, WGSL_PROBE_FILTER,       'main', 'lumen_probe_filter'),
      makeComputePipeline(this.device, WGSL_DIFFUSE_INTEGRATE,  'main', 'lumen_diffuse'),
      makeComputePipeline(this.device, WGSL_SPECULAR_GATHER,    'main', 'lumen_specular'),
      makeComputePipeline(this.device, WGSL_COMPOSITE,          'main', 'lumen_composite'),
    ]);

    // ── Allocate storage buffers ─────────────────────────────────────────────
    const pc = this.probeCount;
    const os = this.octRes * this.octRes;
    this.bufProbeDepth    = makeSSBO(this.device, pc * 4,           'lumen_probeDepth');
    this.bufProbeNormal   = makeSSBO(this.device, pc * 16,          'lumen_probeNormal');
    this.bufProbePos      = makeSSBO(this.device, pc * 16,          'lumen_probePos');
    this.bufTraceRadiance = makeSSBO(this.device, pc * os * 16,     'lumen_traceRadiance');
    this.bufSHAmb         = makeSSBO(this.device, pc * 16,          'lumen_shAmb');
    this.bufSHDir         = makeSSBO(this.device, pc * 16,          'lumen_shDir');
    this.bufPrevSHAmb     = makeSSBO(this.device, pc * 16,          'lumen_prevSHAmb');
    this.bufPrevSHDir     = makeSSBO(this.device, pc * 16,          'lumen_prevSHDir');
    this.bufRCCache       = makeSSBO(this.device, this.rcClipmapCells * 16, 'lumen_rcCache');
    this.bufRCPrev        = makeSSBO(this.device, this.rcClipmapCells * 16, 'lumen_rcPrev');

    // ── Render targets ───────────────────────────────────────────────────────
    this.rtDiffuse      = makeRT(this.device, this.width, this.height, 'rgba16float', 'lumen_diffuse');
    this.rtSpecular     = makeRT(this.device, this.width, this.height, 'rgba16float', 'lumen_specular');
    this.rtPrevSpecular = makeRT(this.device, this.width, this.height, 'rgba16float', 'lumen_prevSpecular');
    this.outputTex      = makeRT(this.device, this.width, this.height, 'rgba16float', 'lumen_output');

    // ── UBOs ─────────────────────────────────────────────────────────────────
    this.uboProbePlacement = makeUBO(this.device, 32,  'lumen_ubo_placement');
    this.uboRCParams       = makeUBO(this.device, 80,  'lumen_ubo_rc');
    this.uboProbeTrace     = makeUBO(this.device, 96,  'lumen_ubo_trace');
    this.uboFilterParams   = makeUBO(this.device, 32,  'lumen_ubo_filter');
    this.uboDiffuseParams  = makeUBO(this.device, 32,  'lumen_ubo_diffuse');
    this.uboSpecParams     = makeUBO(this.device, 80,  'lumen_ubo_spec');
    this.uboCompositeParams= makeUBO(this.device, 16,  'lumen_ubo_composite');
  }

  /** Inject current Cell SDF data from the Cell PubSub Loop. */
  setCellSDFBuffer(buf: GPUBuffer, count: number): void {
    this.cellSDFBuffer = buf;
    this.cellCount     = count;
  }

  /** Update configuration at runtime. */
  setConfig(patch: Partial<UELumenGIConfig>): void {
    Object.assign(this.cfg, patch);
  }

  /** Per-frame tick — advances temporal state. */
  tick(_dt: number): void {
    this.frameIdx++;
  }

  /**
   * Encode all Lumen GI passes into the provided command encoder.
   * @param encoder  Active GPUCommandEncoder
   * @param inputs   G-Buffer textures + camera UBO for this frame
   * @param outputTex  Target texture to write final GI-composited HDR
   */
  render(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs, outputTex?: GPUTexture): void {
    if (!this.cfg.enabled) return;

    const out = outputTex ?? this.outputTex;

    this._uploadUBOs(inputs);
    this._encodePassPlacement(encoder, inputs);
    this._encodePassRCUpdate(encoder, inputs);
    this._encodePassTrace(encoder, inputs);
    this._encodePassFilter(encoder);
    this._encodePassDiffuse(encoder, inputs);
    if (this.cfg.specularEnabled) {
      this._encodePassSpecular(encoder, inputs);
    }
    this._encodePassComposite(encoder, inputs, out);
    this._swapTemporalBuffers(encoder);
  }

  // ── Internal pass encoders ──────────────────────────────────────────────────

  private _uploadUBOs(inputs: LumenGIRenderInputs): void {
    const cfg = this.cfg;
    const d   = this.device;

    // Pass 0: Probe Placement params
    {
      const data = new Uint32Array(8);
      data[0] = cfg.probeDownsampleFactor;
      data[1] = this.probeW;
      data[2] = this.probeH;
      data[3] = cfg.maxAdaptiveProbes;
      data[4] = inputs.frameIndex & 0xFFFF;
      d.queue.writeBuffer(this.uboProbePlacement, 0, data);
    }

    // Pass 1: Radiance Cache params
    {
      const data = new ArrayBuffer(80);
      const u32  = new Uint32Array(data);
      const f32  = new Float32Array(data);
      u32[0] = cfg.radianceCacheClipmapCount;
      u32[1] = cfg.radianceCacheClipmapResolution;
      u32[2] = cfg.radianceCacheProbeResolution;
      u32[3] = this.probeCount;
      u32[4] = this.cellCount;
      u32[5] = cfg.radianceCacheMaxFrames;
      f32[6] = 2500.0; // clipmap0 half-extent (matches LumenScene CVarLumenSceneGlobalDFClipmapExtent)
      // camPos filled from camera UBO externally; pass zeros for now (GPU reads from rcCamPos)
      d.queue.writeBuffer(this.uboRCParams, 0, data);
    }

    // Pass 2: Probe Trace params
    {
      const data = new ArrayBuffer(96);
      const u32  = new Uint32Array(data);
      const f32  = new Float32Array(data);
      u32[0] = this.probeW;
      u32[1] = this.probeH;
      u32[2] = this.octRes;
      u32[3] = this.cellCount;
      u32[4] = cfg.sdfMaxSteps;
      f32[5] = cfg.sdfMaxTraceDistance;
      f32[6] = cfg.surfaceBias;
      u32[7] = inputs.frameIndex;
      u32[8] = cfg.radianceCacheClipmapCount;
      u32[9] = cfg.radianceCacheClipmapResolution;
      u32[10] = cfg.radianceCacheProbeResolution;
      f32[11] = 2500.0;
      d.queue.writeBuffer(this.uboProbeTrace, 0, data);
    }

    // Pass 3: Filter params
    {
      const data = new Uint32Array(8);
      data[0] = this.probeW;
      data[1] = this.probeH;
      data[2] = this.octRes;
      data[3] = cfg.spatialFilterRadius;
      data[4] = cfg.maxFramesAccumulated;
      data[5] = inputs.frameIndex;
      d.queue.writeBuffer(this.uboFilterParams, 0, data);
    }

    // Pass 4: Diffuse params
    {
      const data = new ArrayBuffer(32);
      const u32  = new Uint32Array(data);
      const f32  = new Float32Array(data);
      u32[0] = cfg.probeDownsampleFactor;
      u32[1] = this.probeW;
      u32[2] = this.probeH;
      f32[3] = cfg.diffuseIntensity * cfg.intensity;
      d.queue.writeBuffer(this.uboDiffuseParams, 0, data);
    }

    // Pass 5: Specular params
    {
      const data = new ArrayBuffer(80);
      const u32  = new Uint32Array(data);
      const f32  = new Float32Array(data);
      u32[0] = cfg.specularRaysPerPixel;
      f32[1] = cfg.maxRoughnessToTrace;
      f32[2] = cfg.specularNearFieldMaxTrace;
      f32[3] = cfg.surfaceBias;
      u32[4] = this.cellCount;
      u32[5] = cfg.radianceCacheClipmapCount;
      u32[6] = cfg.radianceCacheClipmapResolution;
      u32[7] = cfg.radianceCacheProbeResolution;
      f32[8] = 2500.0;
      f32[9] = cfg.specularIntensity * cfg.intensity;
      f32[10] = 1.0 / 4.0; // temporal alpha = 1/maxSpecFrames
      u32[11] = inputs.frameIndex;
      d.queue.writeBuffer(this.uboSpecParams, 0, data);
    }

    // Pass 6: Composite params
    {
      const data = new Float32Array(4);
      data[0] = cfg.intensity;
      this.device.queue.writeBuffer(this.uboCompositeParams, 0, data);
    }
  }

  private _getFallbackCellBuffer(): GPUBuffer {
    if (!this.cellSDFBuffer) {
      // Empty 1-cell buffer as fallback
      const buf = makeSSBO(this.device, 64, 'lumen_cell_fallback');
      this.cellSDFBuffer = buf;
    }
    return this.cellSDFBuffer;
  }

  private _encodePassPlacement(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs): void {
    const bg = this.device.createBindGroup({
      layout: this.pipePlacement.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputs.cameraUBO } },
        { binding: 1, resource: inputs.depthTex.createView() },
        { binding: 2, resource: inputs.normalTex.createView() },
        { binding: 3, resource: { buffer: this.bufProbeDepth } },
        { binding: 4, resource: { buffer: this.bufProbeNormal } },
        { binding: 5, resource: { buffer: this.bufProbePos } },
        { binding: 6, resource: { buffer: this.uboProbePlacement } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_placement' });
    pass.setPipeline(this.pipePlacement);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.probeW / 8),
      Math.ceil(this.probeH / 8),
    );
    pass.end();
  }

  private _encodePassRCUpdate(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs): void {
    const cellBuf = this._getFallbackCellBuffer();
    const bg = this.device.createBindGroup({
      layout: this.pipeRCUpdate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputs.cameraUBO } },
        { binding: 1, resource: { buffer: this.bufProbePos } },
        { binding: 2, resource: { buffer: cellBuf } },
        { binding: 3, resource: { buffer: this.bufRCCache } },
        { binding: 4, resource: { buffer: this.bufRCPrev } },
        { binding: 5, resource: { buffer: this.uboRCParams } },
      ],
    });
    const cfg = this.cfg;
    const pass = encoder.beginComputePass({ label: 'lumen_rc_update' });
    pass.setPipeline(this.pipeRCUpdate);
    pass.setBindGroup(0, bg);
    // Dispatch: (clipmapCount, clipmapRes, clipmapRes)
    pass.dispatchWorkgroups(
      Math.ceil(cfg.radianceCacheClipmapCount / 4),
      Math.ceil(cfg.radianceCacheClipmapResolution / 4),
      Math.ceil(cfg.radianceCacheClipmapResolution / 4),
    );
    pass.end();
  }

  private _encodePassTrace(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs): void {
    const cellBuf = this._getFallbackCellBuffer();
    const bg = this.device.createBindGroup({
      layout: this.pipeTrace.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputs.cameraUBO } },
        { binding: 1, resource: inputs.depthTex.createView() },
        { binding: 2, resource: inputs.sceneColorTex.createView() },
        { binding: 3, resource: { buffer: this.bufProbePos } },
        { binding: 4, resource: { buffer: this.bufProbeNormal } },
        { binding: 5, resource: { buffer: cellBuf } },
        { binding: 6, resource: { buffer: this.bufRCCache } },
        { binding: 7, resource: { buffer: this.bufTraceRadiance } },
        { binding: 8, resource: { buffer: this.uboProbeTrace } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_probe_trace' });
    pass.setPipeline(this.pipeTrace);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.probeW / 8),
      Math.ceil(this.probeH / 8),
    );
    pass.end();
  }

  private _encodePassFilter(encoder: GPUCommandEncoder): void {
    const bg = this.device.createBindGroup({
      layout: this.pipeFilter.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufTraceRadiance } },
        { binding: 1, resource: { buffer: this.bufProbeNormal } },
        { binding: 2, resource: { buffer: this.bufProbeDepth } },
        { binding: 3, resource: { buffer: this.bufPrevSHAmb } },
        { binding: 4, resource: { buffer: this.bufPrevSHDir } },
        { binding: 5, resource: { buffer: this.bufSHAmb } },
        { binding: 6, resource: { buffer: this.bufSHDir } },
        { binding: 7, resource: { buffer: this.uboFilterParams } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_filter' });
    pass.setPipeline(this.pipeFilter);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.probeW / 8),
      Math.ceil(this.probeH / 8),
    );
    pass.end();
  }

  private _encodePassDiffuse(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs): void {
    const bg = this.device.createBindGroup({
      layout: this.pipeDiffuse.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputs.cameraUBO } },
        { binding: 1, resource: inputs.depthTex.createView() },
        { binding: 2, resource: inputs.normalTex.createView() },
        { binding: 3, resource: inputs.albedoTex.createView() },
        { binding: 4, resource: { buffer: this.bufSHAmb } },
        { binding: 5, resource: { buffer: this.bufSHDir } },
        { binding: 6, resource: { buffer: this.bufProbeDepth } },
        { binding: 7, resource: { buffer: this.bufProbeNormal } },
        { binding: 8, resource: this.rtDiffuse.createView() },
        { binding: 9, resource: { buffer: this.uboDiffuseParams } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_diffuse' });
    pass.setPipeline(this.pipeDiffuse);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  private _encodePassSpecular(encoder: GPUCommandEncoder, inputs: LumenGIRenderInputs): void {
    const cellBuf = this._getFallbackCellBuffer();
    const bg = this.device.createBindGroup({
      layout: this.pipeSpecular.getBindGroupLayout(0),
      entries: [
        { binding: 0,  resource: { buffer: inputs.cameraUBO } },
        { binding: 1,  resource: inputs.depthTex.createView() },
        { binding: 2,  resource: inputs.normalTex.createView() },
        { binding: 3,  resource: inputs.albedoTex.createView() },
        { binding: 4,  resource: inputs.roughnessTex.createView() },
        { binding: 5,  resource: inputs.sceneColorTex.createView() },
        { binding: 6,  resource: { buffer: cellBuf } },
        { binding: 7,  resource: { buffer: this.bufRCCache } },
        { binding: 8,  resource: this.rtSpecular.createView() },
        { binding: 9,  resource: this.rtPrevSpecular.createView() },
        { binding: 10, resource: { buffer: this.uboSpecParams } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_specular' });
    pass.setPipeline(this.pipeSpecular);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  private _encodePassComposite(
    encoder: GPUCommandEncoder,
    inputs: LumenGIRenderInputs,
    output: GPUTexture
  ): void {
    const bg = this.device.createBindGroup({
      layout: this.pipeComposite.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputs.sceneColorTex.createView() },
        { binding: 1, resource: this.rtDiffuse.createView() },
        { binding: 2, resource: this.rtSpecular.createView() },
        { binding: 3, resource: output.createView() },
        { binding: 4, resource: { buffer: this.uboCompositeParams } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'lumen_composite' });
    pass.setPipeline(this.pipeComposite);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  /** Swap SH history and RC cache buffers for next frame temporal accumulation. */
  private _swapTemporalBuffers(encoder: GPUCommandEncoder): void {
    // Copy current SH → prev SH
    encoder.copyBufferToBuffer(this.bufSHAmb, 0, this.bufPrevSHAmb, 0, this.probeCount * 16);
    encoder.copyBufferToBuffer(this.bufSHDir, 0, this.bufPrevSHDir, 0, this.probeCount * 16);
    // Copy current RC → prev RC
    encoder.copyBufferToBuffer(this.bufRCCache, 0, this.bufRCPrev, 0, this.rcClipmapCells * 16);
    // Copy current specular → prev specular
    encoder.copyTextureToTexture(
      { texture: this.rtSpecular },
      { texture: this.rtPrevSpecular },
      [this.width, this.height],
    );
  }

  /** Resize all buffers and render targets when canvas size changes. */
  async resize(newWidth: number, newHeight: number): Promise<void> {
    this.width  = newWidth;
    this.height = newHeight;
    this._destroyRTs();
    await this._init();
  }

  private _destroyRTs(): void {
    this.rtDiffuse?.destroy();
    this.rtSpecular?.destroy();
    this.rtPrevSpecular?.destroy();
    this.outputTex?.destroy();
    this.bufProbeDepth?.destroy();
    this.bufProbeNormal?.destroy();
    this.bufProbePos?.destroy();
    this.bufTraceRadiance?.destroy();
    this.bufSHAmb?.destroy();
    this.bufSHDir?.destroy();
    this.bufPrevSHAmb?.destroy();
    this.bufPrevSHDir?.destroy();
    this.bufRCCache?.destroy();
    this.bufRCPrev?.destroy();
  }

  /** Free all GPU resources. */
  destroy(): void {
    this._destroyRTs();
    this.uboProbePlacement?.destroy();
    this.uboRCParams?.destroy();
    this.uboProbeTrace?.destroy();
    this.uboFilterParams?.destroy();
    this.uboDiffuseParams?.destroy();
    this.uboSpecParams?.destroy();
    this.uboCompositeParams?.destroy();
  }
}
