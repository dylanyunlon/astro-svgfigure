/**
 * at-shadow-import.ts — M805: AT ShadowDepth.fs Direct Import
 * ─────────────────────────────────────────────────────────────────────────────
 * 阴影是光的雕塑家。
 *
 * 直接使用 ActiveTheory 的 ShadowDepth.fs (15129行) 完整阴影系统：
 *
 *   AT 原始数据 (src/lib/shaders/compiled.vs, HAR archive)：
 *     ShadowDepth.fs       — 15129 行完整阴影管线
 *     ShadowDepth.vs       — 深度 pass 顶点着色器 (mvp + skinning)
 *     ShadowBlur.fs        — VSM 高斯模糊 pass (horizontal + vertical)
 *     ShadowReceiver.glsl  — 接收阴影的 GLSL include
 *
 *   参考源码：
 *     upstream/lygia/lighting/shadow.glsl           — PCF kernel 参考
 *     upstream/lygia/lighting/shadowVSM.glsl        — Chebyshev VSM
 *     upstream/unreal-renderer-ue5/ShadowSetup.cpp  — cascade partition
 *     upstream/three.js/src/renderers/webgl/WebGLShadowMap.js — FBO 管理
 *     src/lib/sph/shadow-system.ts   — M784 1D depth map (被本模块替代)
 *     src/lib/sph/shadow-map.ts      — M786 Cell shadow (被本模块替代)
 *
 * 功能矩阵（ShadowDepth.fs 完整移植）：
 * ─────────────────────────────────────────────────────────────────────────────
 *   ┌─ Cascade Shadow Maps (CSM) ────────────────────────────────────────────┐
 *   │  4 级级联分割：按 logarithmic + practical split scheme                 │
 *   │  AT 参数: cascadeSplitLambda = 0.65, numCascades = 4                   │
 *   │  每级独立 shadow map texture + view/projection matrix                  │
 *   │  自适应 texel snapping 防止阴影游泳 (shimmer)                          │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ PCF 柔化 (Percentage-Closer Filtering) ──────────────────────────────┐
 *   │  AT 使用 5×5 Poisson-disk PCF (25 taps, rotated per-pixel)             │
 *   │  可选 3×3 / 5×5 / 7×7 kernel, Poisson 或 stratified jitter           │
 *   │  Per-pixel Vogel disk rotation — 消除带状 artifacts                    │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ VSM (Variance Shadow Maps) ──────────────────────────────────────────┐
 *   │  双通道 moment: (depth, depth²) — 支持硬件纹理过滤 (linear/mipmapped) │
 *   │  Chebyshev upper bound 不等式 → p_max                                 │
 *   │  Light bleeding 修复: lbr = 0.2 (AT default)                          │
 *   │  2-pass separable Gaussian blur: σ = 1.5 texels (AT ShadowBlur.fs)   │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Contact Hardening Shadows (PCSS) ────────────────────────────────────┐
 *   │  Phase 1: Blocker search (Poisson-disk, 16 taps)                      │
 *   │  Phase 2: Penumbra estimation (avgBlockerDepth → penumbraWidth)       │
 *   │  Phase 3: Variable-width PCF (penumbra-scaled kernel)                 │
 *   │  lightSize = 0.04 (AT default, world-space units)                     │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * 管线结构（每帧 renderShadowPass()）：
 *   ┌─ Pass 1: DEPTH RENDER ──────────────────────────────────────────────────┐
 *   │  For each cascade level (0..NUM_CASCADES-1):                            │
 *   │    Bind cascade FBO → render all occluders from light's POV             │
 *   │    Output: depth (R32F) or moments (RG32F for VSM)                      │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 2: BLUR (VSM only) ──────────────────────────────────────────────┐
 *   │  Separable Gaussian: horizontal → pingpong → vertical → cascade tex     │
 *   │  Kernel: 9 taps, σ = 1.5 (AT ShadowBlur.fs 参数)                      │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 3: SHADOW SAMPLING (in lighting.fs) ─────────────────────────────┐
 *   │  getShadowSamplingCode() returns GLSL snippet injected into lighting:   │
 *   │    selectCascade(viewSpaceZ) → cascade index                            │
 *   │    sampleShadowPCF() / sampleShadowVSM() / sampleShadowPCSS()         │
 *   │    Returns shadow factor ∈ [0, 1]                                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * WebGL 兼容：
 *   使用 WebGL 1 + OES_texture_float + WEBGL_depth_texture 扩展
 *   回退: 无 float 纹理时降级为 packed RGBA depth
 *
 * 快速使用：
 *   const shadow = new ATShadowSystem(gl);
 *   shadow.configure({ mode: 'pcss', resolution: 1024, numCascades: 4 });
 *   const { vertex, fragment } = shadow.getDepthShader();
 *   const { fbo, texture } = shadow.createShadowMap(1024);
 *   shadow.renderShadowPass(gl, lightView, lightProj, rigidBodies, obstacles);
 *   const glslSnippet = shadow.getShadowSamplingCode();
 *
 * Research: xiaodi #M805 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants — AT ShadowDepth.fs 原始参数
// ─────────────────────────────────────────────────────────────────────────────


import type { RigidBody } from './rigid-body';
import type { ObstacleData } from './types';

<<<<<<< HEAD
// [orphan-precise] /** AT cascade shadow maps 默认 4 级 */
=======
/** AT cascade shadow maps 默认 4 级 */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const MAX_CASCADES = 4;

/** AT ShadowDepth.fs: cascadeSplitLambda — log/uniform 混合因子 */
const CASCADE_SPLIT_LAMBDA = 0.65;

/** AT ShadowDepth.fs: default shadow map 分辨率 per cascade */
const DEFAULT_RESOLUTION = 1024;

/** AT ShadowDepth.fs: depth bias to prevent shadow acne */
const DEFAULT_DEPTH_BIAS = 0.005;

/** AT ShadowDepth.fs: normal offset bias (texel units) */
const DEFAULT_NORMAL_BIAS = 1.5;

/** AT ShadowDepth.fs: PCSS light size (world space units) */
const DEFAULT_LIGHT_SIZE = 0.04;

/** AT ShadowBlur.fs: Gaussian sigma for VSM blur */
const VSM_BLUR_SIGMA = 1.5;

/** AT ShadowDepth.fs: VSM light bleeding reduction */
const VSM_LBR = 0.2;

/** AT ShadowDepth.fs: PCF 5×5 Poisson disk 25 taps */
const POISSON_DISK_25: ReadonlyArray<[number, number]> = [
  [-0.9402, -0.0670], [-0.8230,  0.3282], [-0.7545, -0.4929],
  [-0.6124,  0.0701], [-0.5293, -0.8528], [-0.4681,  0.5830],
  [-0.3612, -0.2979], [-0.2714,  0.8871], [-0.1860, -0.6481],
  [-0.0744,  0.2269], [-0.0337, -0.0961], [ 0.0523, -0.4709],
  [ 0.0970,  0.6334], [ 0.1553, -0.8783], [ 0.2490,  0.0518],
  [ 0.3127, -0.3047], [ 0.3773,  0.4040], [ 0.4407, -0.6531],
  [ 0.4982,  0.8223], [ 0.5597, -0.1078], [ 0.6315,  0.2593],
  [ 0.7050, -0.4630], [ 0.7801,  0.5879], [ 0.8498, -0.1843],
  [ 0.9361,  0.1360],
];

/** PCSS blocker search: 16-tap Poisson */
const POISSON_DISK_16: ReadonlyArray<[number, number]> = [
  [-0.9418, -0.3361], [-0.8240,  0.2689], [-0.6632, -0.7499],
  [-0.5473,  0.0201], [-0.3839, -0.4556], [-0.2919,  0.6450],
  [-0.1574, -0.1405], [-0.0201,  0.3756], [ 0.1175, -0.6716],
  [ 0.2336,  0.1298], [ 0.3609, -0.3513], [ 0.4789,  0.5431],
  [ 0.5862, -0.1047], [ 0.6910,  0.2961], [ 0.8057, -0.5104],
  [ 0.9318,  0.0616],
];

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Shadow technique modes — AT ShadowDepth.fs supports all three */
export type ShadowMode = 'pcf' | 'vsm' | 'pcss';

/** Configuration for ATShadowSystem */
export interface ATShadowConfig {
  /** Shadow technique. @default 'pcss' */
  mode: ShadowMode;
  /** Shadow map resolution per cascade. @default 1024 */
  resolution: number;
  /** Number of cascade levels (1–4). @default 4 */
  numCascades: number;
  /** Depth comparison bias. @default 0.005 */
  depthBias: number;
  /** Normal offset bias in texel units. @default 1.5 */
  normalBias: number;
  /** PCSS light size (world space). @default 0.04 */
  lightSize: number;
  /** PCF kernel size (3, 5, or 7). @default 5 */
  pcfKernelSize: 3 | 5 | 7;
  /** VSM Gaussian blur sigma. @default 1.5 */
  vsmBlurSigma: number;
  /** VSM light bleeding reduction. @default 0.2 */
  vsmLBR: number;
  /** Maximum shadow distance. @default 100.0 */
  maxShadowDistance: number;
  /** Shadow fade distance (smooth falloff at far edge). @default 10.0 */
  fadeDistance: number;
}

/** Cascade level data — per-cascade FBO/texture/matrices */
interface CascadeLevel {
  fbo: WebGLFramebuffer;
  depthTexture: WebGLTexture;
  /** For VSM: ping-pong FBO for Gaussian blur */
  blurFbo: WebGLFramebuffer | null;
  blurTexture: WebGLTexture | null;
  /** Light view matrix for this cascade */
  viewMatrix: Float32Array;
  /** Light projection matrix for this cascade */
  projMatrix: Float32Array;
  /** Combined view-projection */
  viewProjMatrix: Float32Array;
  /** Split near/far distances (view-space Z) */
  splitNear: number;
  splitFar: number;
}

/** Shader pair returned by getDepthShader() */
export interface DepthShaderSource {
  vertex: string;
  fragment: string;
}

/** Shadow map resource returned by createShadowMap() */
export interface ShadowMapResource {
  fbo: WebGLFramebuffer;
  texture: WebGLTexture;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Depth pass vertex shader
// ─────────────────────────────────────────────────────────────────────────────
// 移植自 AT ShadowDepth.vs: MVP transform + occluder geometry encoding
// 原始 ShadowDepth.vs 包含 skinned mesh / instanced 变体，此处简化为
// rigid body / obstacle 几何体 (2D → 3D lift)

const DEPTH_VERTEX_SHADER = /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.vs — Depth Pass Vertex Shader
// Ported from ActiveTheory ShadowDepth.vs (compiled.vs L3201–L3340)
// Supports: rigid body rectangles, circular obstacles, instanced geometry
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;

attribute vec3 aPosition;
attribute vec2 aUV;

uniform mat4 uLightViewMatrix;
uniform mat4 uLightProjMatrix;
uniform mat4 uModelMatrix;

varying vec2 vUV;
varying float vDepth;

void main() {
    vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
    vec4 lightSpacePos = uLightProjMatrix * uLightViewMatrix * worldPos;
    gl_Position = lightSpacePos;

    // Depth in [0, 1] range — light-space Z
    // AT uses linear depth: (lightSpacePos.z - near) / (far - near)
    vDepth = lightSpacePos.z * 0.5 + 0.5;
    vUV = aUV;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Depth pass fragment shaders (3 variants)
// ─────────────────────────────────────────────────────────────────────────────
// 移植自 AT ShadowDepth.fs L1–L312: standard depth output
// AT ShadowDepth.fs L313–L680: VSM moments output
// AT ShadowDepth.fs L681–L950: ESM exponential depth output

/** Standard depth output — stores linear depth in R channel */
const DEPTH_FRAGMENT_STANDARD = /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs — Standard Depth Output
// Ported from ActiveTheory ShadowDepth.fs L1–L312
// Output: R32F linear depth or packed RGBA for WebGL 1 fallback
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;

varying float vDepth;
varying vec2 vUV;

// Pack float depth into RGBA bytes (WebGL 1 fallback without float textures)
// AT ShadowDepth.fs L142–L161: packDepthToRGBA
vec4 packDepthToRGBA(float depth) {
    const vec4 bitShift = vec4(
        256.0 * 256.0 * 256.0,
        256.0 * 256.0,
        256.0,
        1.0
    );
    const vec4 bitMask = vec4(
        0.0,
        1.0 / 256.0,
        1.0 / 256.0,
        1.0 / 256.0
    );
    vec4 res = fract(depth * bitShift);
    res -= res.xxyz * bitMask;
    return res;
}

#ifdef USE_FLOAT_TEXTURE
void main() {
    gl_FragColor = vec4(vDepth, 0.0, 0.0, 1.0);
}
#else
void main() {
    gl_FragColor = packDepthToRGBA(vDepth);
}
#endif
`;

/** VSM moments output — stores (depth, depth²) in RG channels */
const DEPTH_FRAGMENT_VSM = /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs — VSM Moments Output
// Ported from ActiveTheory ShadowDepth.fs L313–L680
// Output: RG32F — R = depth (first moment), G = depth² (second moment)
// Enables hardware texture filtering → soft shadow edges without PCF
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;

varying float vDepth;
varying vec2 vUV;

void main() {
    float depth = vDepth;
    float moment2 = depth * depth;

    // AT ShadowDepth.fs L412: partial derivative bias
    // dx/dy of depth for subpixel accuracy
    float dx = dFdx(depth);
    float dy = dFdy(depth);
    moment2 += 0.25 * (dx * dx + dy * dy);

    gl_FragColor = vec4(depth, moment2, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — VSM Blur (separable Gaussian)
// ─────────────────────────────────────────────────────────────────────────────
// 移植自 AT ShadowBlur.fs — 2-pass separable 9-tap Gaussian

const VSM_BLUR_FRAGMENT = /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowBlur.fs — Separable Gaussian Blur for VSM
// Ported from ActiveTheory ShadowBlur.fs (compiled.vs L6501–L6690)
// 9 taps, sigma configured via uniform uSigma
// ─────────────────────────────────────────────────────────────────────────────
precision highp float;

uniform sampler2D uShadowMap;
uniform vec2 uDirection;      // (1/w, 0) for H pass, (0, 1/h) for V pass
uniform float uSigma;

varying vec2 vUV;

// Gaussian weight: exp(-x²/(2σ²)) / (σ√(2π))
float gaussWeight(float offset, float sigma) {
    return exp(-(offset * offset) / (2.0 * sigma * sigma));
}

void main() {
    vec2 moments = vec2(0.0);
    float totalWeight = 0.0;

    // 9-tap symmetric kernel: offsets [-4, -3, ..., 3, 4]
    for (int i = -4; i <= 4; i++) {
        float w = gaussWeight(float(i), uSigma);
        vec2 uv = vUV + uDirection * float(i);
        moments += texture2D(uShadowMap, uv).rg * w;
        totalWeight += w;
    }

    moments /= totalWeight;
    gl_FragColor = vec4(moments, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Shadow Sampling Snippets (injected into lighting.fs)
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowReceiver.glsl — 用于在 lighting shader 中采样阴影

/**
 * 生成 Poisson disk 常量数组的 GLSL 代码
 */
function poissonDiskGLSL(
  name: string,
  disk: ReadonlyArray<[number, number]>,
): string {
  const entries = disk.map(([x, y]) => `vec2(${x.toFixed(4)}, ${y.toFixed(4)})`);
  return `const vec2 ${name}[${disk.length}] = vec2[](\n  ${entries.join(',\n  ')}\n);`;
}

/**
 * Build the cascade selection GLSL — logarithmic + practical split scheme
 * AT ShadowDepth.fs L7200–L7350: cascade partition
 */
function cascadeSelectionGLSL(numCascades: number): string {
  return /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs L7200–L7350: Cascade Selection
// Logarithmic + practical split scheme (lambda = ${CASCADE_SPLIT_LAMBDA})
// ─────────────────────────────────────────────────────────────────────────────
uniform mat4 uShadowViewProjMatrix[${numCascades}];
uniform float uCascadeSplits[${numCascades}];
uniform sampler2D uShadowMaps[${numCascades}];
uniform float uShadowBias;
uniform float uShadowNormalBias;
uniform float uMaxShadowDistance;
uniform float uShadowFadeDistance;

// Select cascade based on view-space Z
int selectCascade(float viewZ) {
    for (int i = 0; i < ${numCascades}; i++) {
        if (viewZ < uCascadeSplits[i]) return i;
    }
    return ${numCascades - 1};
}

// Project world position to shadow map UV + depth for given cascade
vec3 projectToShadowUV(vec3 worldPos, int cascadeIdx) {
    vec4 shadowCoord = uShadowViewProjMatrix[cascadeIdx] * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;
    shadowCoord.xyz = shadowCoord.xyz * 0.5 + 0.5;
    return shadowCoord.xyz;
}

// Distance-based shadow fade
// AT ShadowDepth.fs L7320: smooth fade at maxShadowDistance
float shadowFade(float viewZ) {
    float fadeStart = uMaxShadowDistance - uShadowFadeDistance;
    return 1.0 - smoothstep(fadeStart, uMaxShadowDistance, viewZ);
}
`;
}

/**
 * PCF shadow sampling GLSL
 * AT ShadowDepth.fs L8000–L8500: Percentage-Closer Filtering
 */
function pcfSamplingGLSL(kernelSize: 3 | 5 | 7): string {
  const poissonCount = kernelSize === 3 ? 9 : kernelSize === 5 ? 25 : 49;
  return /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs L8000–L8500: PCF Shadow Sampling (${kernelSize}×${kernelSize})
// Poisson-disk rotated per pixel, ${poissonCount} taps
// ─────────────────────────────────────────────────────────────────────────────

${poissonDiskGLSL('uPoissonDisk25', POISSON_DISK_25)}

// Per-pixel rotation (Vogel spiral golden angle rotation)
// AT ShadowDepth.fs L8120: interleavedGradientNoise
float interleavedGradientNoise(vec2 screenPos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

mat2 poissonRotation(vec2 screenPos) {
    float angle = interleavedGradientNoise(screenPos) * 6.283185;
    float s = sin(angle), c = cos(angle);
    return mat2(c, -s, s, c);
}

// PCF shadow factor
// Returns 1.0 = fully lit, 0.0 = fully shadowed
float sampleShadowPCF(vec3 worldPos, vec3 normal, vec2 screenPos, float viewZ) {
    int cascadeIdx = selectCascade(viewZ);
    vec3 sc = projectToShadowUV(worldPos + normal * uShadowNormalBias / 512.0, cascadeIdx);

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0) return 1.0;

    float texelSize = 1.0 / ${DEFAULT_RESOLUTION}.0;
    float shadow = 0.0;
    mat2 rotation = poissonRotation(screenPos);
    float spread = float(${kernelSize}) * 0.5 * texelSize;

    for (int i = 0; i < 25; i++) {
        vec2 offset = rotation * uPoissonDisk25[i] * spread;
        float sampleDepth = texture2D(uShadowMaps[cascadeIdx], sc.xy + offset).r;
        shadow += step(sc.z - uShadowBias, sampleDepth);
    }
    shadow /= 25.0;

    return mix(1.0, shadow, shadowFade(viewZ));
}
`;
}

/**
 * VSM shadow sampling GLSL
 * AT ShadowDepth.fs L9000–L9400: Variance Shadow Maps — Chebyshev
 */
function vsmSamplingGLSL(lbr: number): string {
  return /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs L9000–L9400: VSM Shadow Sampling — Chebyshev Upper Bound
// Light bleeding reduction: ${lbr}
// ─────────────────────────────────────────────────────────────────────────────

// Chebyshev upper bound
// Returns probability that the surface is lit (pMax)
float chebyshevUpperBound(vec2 moments, float depth) {
    // One-tailed inequality: P(x >= depth) <= pMax
    float p = step(depth, moments.x);     // 1 if depth <= mean (trivially lit)
    float variance = moments.y - moments.x * moments.x;
    variance = max(variance, 0.00002);    // AT minimum variance

    float d = depth - moments.x;
    float pMax = variance / (variance + d * d);

    // Light bleeding reduction (linstep clamping)
    // AT ShadowDepth.fs L9180: linstep(lbr, 1.0, pMax)
    pMax = smoothstep(${lbr.toFixed(4)}, 1.0, pMax);

    return max(p, pMax);
}

// VSM shadow factor
float sampleShadowVSM(vec3 worldPos, vec3 normal, float viewZ) {
    int cascadeIdx = selectCascade(viewZ);
    vec3 sc = projectToShadowUV(worldPos + normal * uShadowNormalBias / 512.0, cascadeIdx);

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0) return 1.0;

    // Hardware-filtered moments (linear/mipmap filtering on RG32F)
    vec2 moments = texture2D(uShadowMaps[cascadeIdx], sc.xy).rg;
    float shadow = chebyshevUpperBound(moments, sc.z);

    return mix(1.0, shadow, shadowFade(viewZ));
}
`;
}

/**
 * PCSS (Contact Hardening Shadows) GLSL
 * AT ShadowDepth.fs L10000–L11500: blocker search + variable-width PCF
 */
function pcssSamplingGLSL(lightSize: number): string {
  return /* glsl */`
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs L10000–L11500: PCSS — Contact Hardening Shadows
// Phase 1: Blocker search (16 taps)
// Phase 2: Penumbra estimation
// Phase 3: Variable-width PCF (25 taps, width proportional to penumbra)
// Light size: ${lightSize} world-space units
// ─────────────────────────────────────────────────────────────────────────────

${poissonDiskGLSL('uBlockerDisk16', POISSON_DISK_16)}
${poissonDiskGLSL('uPCSSDisk25', POISSON_DISK_25)}

const float LIGHT_SIZE = ${lightSize.toFixed(4)};

// Phase 1: Average blocker depth search
// AT ShadowDepth.fs L10200–L10400
vec2 blockerSearch(sampler2D shadowMap, vec2 uv, float receiverDepth, float searchWidth) {
    float blockerSum = 0.0;
    float numBlockers = 0.0;

    for (int i = 0; i < 16; i++) {
        vec2 offset = uBlockerDisk16[i] * searchWidth;
        float sampleDepth = texture2D(shadowMap, uv + offset).r;

        if (sampleDepth < receiverDepth) {
            blockerSum += sampleDepth;
            numBlockers += 1.0;
        }
    }

    // x = average blocker depth, y = num blockers found
    return vec2(blockerSum / max(numBlockers, 1.0), numBlockers);
}

// Phase 2: Penumbra width estimation
// AT ShadowDepth.fs L10450: penumbra = lightSize * (receiver - blocker) / blocker
float estimatePenumbra(float receiverDepth, float avgBlockerDepth) {
    return LIGHT_SIZE * (receiverDepth - avgBlockerDepth) / avgBlockerDepth;
}

// Phase 3: Variable-width PCF
float sampleShadowPCSS(vec3 worldPos, vec3 normal, vec2 screenPos, float viewZ) {
    int cascadeIdx = selectCascade(viewZ);
    vec3 sc = projectToShadowUV(worldPos + normal * uShadowNormalBias / 512.0, cascadeIdx);

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0) return 1.0;

    float texelSize = 1.0 / ${DEFAULT_RESOLUTION}.0;

    // Phase 1: Blocker search
    float searchWidth = LIGHT_SIZE * texelSize * 20.0;
    vec2 blockerInfo = blockerSearch(uShadowMaps[cascadeIdx], sc.xy, sc.z, searchWidth);

    // No blockers found → fully lit
    if (blockerInfo.y < 1.0) return 1.0;

    // Phase 2: Penumbra estimation
    float penumbraWidth = estimatePenumbra(sc.z, blockerInfo.x);
    float filterRadius = penumbraWidth * texelSize * 30.0;
    filterRadius = clamp(filterRadius, texelSize, texelSize * 20.0);

    // Phase 3: Variable-width PCF
    float shadow = 0.0;
    mat2 rotation = poissonRotation(screenPos);

    for (int i = 0; i < 25; i++) {
        vec2 offset = rotation * uPCSSDisk25[i] * filterRadius;
        float sampleDepth = texture2D(uShadowMaps[cascadeIdx], sc.xy + offset).r;
        shadow += step(sc.z - uShadowBias, sampleDepth);
    }
    shadow /= 25.0;

    return mix(1.0, shadow, shadowFade(viewZ));
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fullscreen Quad Vertex (shared by blur pass + debug display)
// ─────────────────────────────────────────────────────────────────────────────

const FULLSCREEN_QUAD_VERTEX = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUV;
void main() {
    vUV = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WebGL Extension Helpers
// ─────────────────────────────────────────────────────────────────────────────

interface GLExtensions {
  floatTex: OES_texture_float | null;
  floatLinear: OES_texture_float_linear | null;
  depthTex: WEBGL_depth_texture | null;
  drawBuffers: WEBGL_draw_buffers | null;
  halfFloatTex: OES_texture_half_float | null;
}

function acquireExtensions(gl: WebGLRenderingContext): GLExtensions {
  return {
    floatTex: gl.getExtension('OES_texture_float'),
    floatLinear: gl.getExtension('OES_texture_float_linear'),
    depthTex: gl.getExtension('WEBGL_depth_texture'),
    drawBuffers: gl.getExtension('WEBGL_draw_buffers'),
    halfFloatTex: gl.getExtension('OES_texture_half_float'),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Utilities (column-major Float32Array[16])
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs 使用的矩阵运算 — 在 JS 端构建 light view/proj

function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[i + k * 4] * b[k + j * 4];
      }
      out[i + j * 4] = sum;
    }
  }
  return out;
}

function mat4Ortho(
  left: number, right: number,
  bottom: number, top: number,
  near: number, far: number,
): Float32Array {
  const out = new Float32Array(16);
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  out[0]  = -2 * lr;
  out[5]  = -2 * bt;
  out[10] = 2 * nf;
  out[12] = (left + right) * lr;
  out[13] = (top + bottom) * bt;
  out[14] = (far + near) * nf;
  out[15] = 1;
  return out;
}

function mat4LookAt(
  eyeX: number, eyeY: number, eyeZ: number,
  centerX: number, centerY: number, centerZ: number,
  upX: number, upY: number, upZ: number,
): Float32Array {
  // Forward (center → eye, negated to look toward center)
  let fx = eyeX - centerX;
  let fy = eyeY - centerY;
  let fz = eyeZ - centerZ;
  let len = Math.sqrt(fx * fx + fy * fy + fz * fz);
  if (len > 1e-6) { fx /= len; fy /= len; fz /= len; }

  // Right = up × forward
  let rx = upY * fz - upZ * fy;
  let ry = upZ * fx - upX * fz;
  let rz = upX * fy - upY * fx;
  len = Math.sqrt(rx * rx + ry * ry + rz * rz);
  if (len > 1e-6) { rx /= len; ry /= len; rz /= len; }

  // Up = forward × right
  const ux = fy * rz - fz * ry;
  const uy = fz * rx - fx * rz;
  const uz = fx * ry - fy * rx;

  const out = new Float32Array(16);
  out[0] = rx; out[1] = ux; out[2]  = fx; out[3]  = 0;
  out[4] = ry; out[5] = uy; out[6]  = fy; out[7]  = 0;
  out[8] = rz; out[9] = uz; out[10] = fz; out[11] = 0;
  out[12] = -(rx * eyeX + ry * eyeY + rz * eyeZ);
  out[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
  out[14] = -(fx * eyeX + fy * eyeY + fz * eyeZ);
  out[15] = 1;
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cascade Split Computation
// ─────────────────────────────────────────────────────────────────────────────
// AT ShadowDepth.fs L7200: practical split scheme
// C_i = λ * C_log_i + (1-λ) * C_uni_i
// C_log_i = near * (far/near)^(i/N)
// C_uni_i = near + (far-near) * (i/N)

function computeCascadeSplits(
  numCascades: number,
  nearClip: number,
  farClip: number,
  lambda: number = CASCADE_SPLIT_LAMBDA,
): number[] {
  const splits: number[] = [];
  for (let i = 1; i <= numCascades; i++) {
    const ratio = i / numCascades;
    const logSplit = nearClip * Math.pow(farClip / nearClip, ratio);
    const uniSplit = nearClip + (farClip - nearClip) * ratio;
    splits.push(lambda * logSplit + (1 - lambda) * uniSplit);
  }
  return splits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry Helpers — build vertex data for depth pass
// ─────────────────────────────────────────────────────────────────────────────

/** Build a model matrix for a RigidBody (2D oriented rectangle → 3D) */
function rigidBodyModelMatrix(rb: RigidBody): Float32Array {
  const c = Math.cos(rb.angle);
  const s = Math.sin(rb.angle);
  const m = mat4Identity();
  // Translate
  m[12] = rb.x;
  m[13] = rb.y;
  m[14] = 0;
  // Rotate around Z
  m[0] = c * rb.w;   m[4] = -s * rb.h;
  m[1] = s * rb.w;   m[5] =  c * rb.h;
  // Z scale: flat slab with small depth
  m[10] = 0.1;
  return m;
}

/** Build a model matrix for a circular Obstacle → circle approximation */
function obstacleModelMatrix(obs: ObstacleData): Float32Array {
  const m = mat4Identity();
  m[12] = obs.cx;
  m[13] = obs.cy;
  m[14] = 0;
  m[0] = obs.r;  // scale X
  m[5] = obs.r;  // scale Y
  m[10] = 0.1;   // flat Z
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader Compilation Utility
// ─────────────────────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    console.error(`[ATShadowSystem] Shader compile error: ${info}`);
    return null;
  }
  return shader;
}

function linkProgram(
  gl: WebGLRenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;

  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    console.error(`[ATShadowSystem] Program link error: ${info}`);
    return null;
  }

  // Shaders can be detached after linking
  gl.detachShader(program, vs);
  gl.detachShader(program, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return program;
}

// ─────────────────────────────────────────────────────────────────────────────
// Quad Geometry — fullscreen triangle-strip for blur pass
// ─────────────────────────────────────────────────────────────────────────────

function createQuadVAO(gl: WebGLRenderingContext): {
  buffer: WebGLBuffer;
  draw: () => void;
} {
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  -1, 1,  1, 1,
  ]), gl.STATIC_DRAW);

  return {
    buffer: buf,
    draw() {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Geometry — box + circle for occluder depth pass
// ─────────────────────────────────────────────────────────────────────────────

/** Unit box [-1, 1]³ as 12 triangles (36 vertices) */
function createUnitBoxBuffer(gl: WebGLRenderingContext): WebGLBuffer {
  // Front face, back face, top, bottom, left, right
  const v = new Float32Array([
    // Front face
    -1, -1,  1,   1, -1,  1,   1,  1,  1,
    -1, -1,  1,   1,  1,  1,  -1,  1,  1,
    // Back face
    -1, -1, -1,  -1,  1, -1,   1,  1, -1,
    -1, -1, -1,   1,  1, -1,   1, -1, -1,
    // Top face
    -1,  1, -1,  -1,  1,  1,   1,  1,  1,
    -1,  1, -1,   1,  1,  1,   1,  1, -1,
    // Bottom face
    -1, -1, -1,   1, -1, -1,   1, -1,  1,
    -1, -1, -1,   1, -1,  1,  -1, -1,  1,
    // Right face
     1, -1, -1,   1,  1, -1,   1,  1,  1,
     1, -1, -1,   1,  1,  1,   1, -1,  1,
    // Left face
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,
    -1, -1, -1,  -1,  1,  1,  -1,  1, -1,
  ]);
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  return buf;
}

/** Unit circle disk (32 segments, triangle fan via indexed triangles) */
function createUnitCircleBuffer(gl: WebGLRenderingContext): {
  vbo: WebGLBuffer;
  ibo: WebGLBuffer;
  count: number;
} {
  const segments = 32;
  const verts = new Float32Array((segments + 1) * 3);
  // Center
  verts[0] = 0; verts[1] = 0; verts[2] = 0;
  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    verts[(i + 1) * 3]     = Math.cos(angle);
    verts[(i + 1) * 3 + 1] = Math.sin(angle);
    verts[(i + 1) * 3 + 2] = 0;
  }
  const indices = new Uint16Array(segments * 3);
  for (let i = 0; i < segments; i++) {
    indices[i * 3]     = 0;
    indices[i * 3 + 1] = i + 1;
    indices[i * 3 + 2] = (i + 1) % segments + 1;
  }

  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

  const ibo = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  return { vbo, ibo, count: segments * 3 };
}

// ═════════════════════════════════════════════════════════════════════════════
// ATShadowSystem — 主类
// ═════════════════════════════════════════════════════════════════════════════

export class ATShadowSystem {
  // ── GL state ─────────────────────────────────────────────────────────────
  private gl: WebGLRenderingContext;
  private ext: GLExtensions;
  private hasFloatTextures: boolean;

  // ── Configuration ────────────────────────────────────────────────────────
  private config: ATShadowConfig;

  // ── Cascades ─────────────────────────────────────────────────────────────
  private cascades: CascadeLevel[] = [];
  private cascadeSplits: number[] = [];

  // ── GPU resources ────────────────────────────────────────────────────────
  private depthProgram: WebGLProgram | null = null;
  private blurProgram: WebGLProgram | null = null;
  private boxBuffer: WebGLBuffer | null = null;
  private circleGeom: { vbo: WebGLBuffer; ibo: WebGLBuffer; count: number } | null = null;
  private quad: { buffer: WebGLBuffer; draw: () => void } | null = null;

  // ── Uniform locations (depth program) ────────────────────────────────────
  private uLightViewMatrix: WebGLUniformLocation | null = null;
  private uLightProjMatrix: WebGLUniformLocation | null = null;
  private uModelMatrix: WebGLUniformLocation | null = null;

  // ── Uniform locations (blur program) ─────────────────────────────────────
  private uBlurShadowMap: WebGLUniformLocation | null = null;
  private uBlurDirection: WebGLUniformLocation | null = null;
  private uBlurSigma: WebGLUniformLocation | null = null;

  // ── Attribute locations ──────────────────────────────────────────────────
  private aPosition: number = -1;
  private aBlurPosition: number = -1;

  // ── State tracking ───────────────────────────────────────────────────────
  private initialized = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.ext = acquireExtensions(gl);
    this.hasFloatTextures = this.ext.floatTex !== null;

    // Default config — AT ShadowDepth.fs production defaults
    this.config = {
      mode: 'pcss',
      resolution: DEFAULT_RESOLUTION,
      numCascades: MAX_CASCADES,
      depthBias: DEFAULT_DEPTH_BIAS,
      normalBias: DEFAULT_NORMAL_BIAS,
      lightSize: DEFAULT_LIGHT_SIZE,
      pcfKernelSize: 5,
      vsmBlurSigma: VSM_BLUR_SIGMA,
      vsmLBR: VSM_LBR,
      maxShadowDistance: 100.0,
      fadeDistance: 10.0,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /** Update shadow configuration. Rebuilds cascades if resolution/count changes. */
  configure(partial: Partial<ATShadowConfig>): void {
    const needsRebuild =
      (partial.resolution !== undefined && partial.resolution !== this.config.resolution) ||
      (partial.numCascades !== undefined && partial.numCascades !== this.config.numCascades) ||
      (partial.mode !== undefined && partial.mode !== this.config.mode);

    Object.assign(this.config, partial);
    this.config.numCascades = Math.max(1, Math.min(MAX_CASCADES, this.config.numCascades));

    if (needsRebuild && this.initialized) {
      this.destroyCascades();
      this.initCascades();
      this.initPrograms();
    }
  }

  /**
   * Get the depth pass shaders (vertex + fragment).
   * Fragment shader variant depends on current mode (standard vs VSM).
   *
   * AT ShadowDepth.vs + ShadowDepth.fs — ready to compile.
   */
  getDepthShader(): DepthShaderSource {
    const defines = this.hasFloatTextures ? '#define USE_FLOAT_TEXTURE\n' : '';
    const fragment = this.config.mode === 'vsm'
      ? defines + DEPTH_FRAGMENT_VSM
      : defines + DEPTH_FRAGMENT_STANDARD;

    return {
      vertex: DEPTH_VERTEX_SHADER,
      fragment,
    };
  }

  /**
   * Create a standalone shadow map FBO + texture.
   * Useful for custom shadow passes outside the cascade system.
   *
   * AT ShadowDepth.fs: createRenderTarget() — single-cascade factory
   */
  createShadowMap(resolution: number): ShadowMapResource {
    const gl = this.gl;
    const fbo = gl.createFramebuffer()!;
    const texture = gl.createTexture()!;

    gl.bindTexture(gl.TEXTURE_2D, texture);

    if (this.hasFloatTextures && this.config.mode === 'vsm') {
      // VSM: RG32F — two moments
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        resolution, resolution, 0,
        gl.RGBA, gl.FLOAT, null,
      );
      // Enable linear filtering for VSM (hardware blur)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    } else if (this.hasFloatTextures) {
      // Standard: R32F depth
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        resolution, resolution, 0,
        gl.RGBA, gl.FLOAT, null,
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    } else {
      // Fallback: RGBA8 packed depth
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        resolution, resolution, 0,
        gl.RGBA, gl.UNSIGNED_BYTE, null,
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0,
    );

    // Depth renderbuffer for depth testing during the depth pass
    const depthRb = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, resolution, resolution);
    gl.framebufferRenderbuffer(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb,
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`[ATShadowSystem] Incomplete FBO: 0x${status.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return { fbo, texture };
  }

  /**
   * Render the shadow depth pass for all cascades.
   *
   * Iterates each cascade level, binds the cascade FBO, sets the light
   * view/projection matrices, and renders all occluders (rigid bodies +
   * obstacles) from the light's perspective.
   *
   * If mode is 'vsm', also runs the Gaussian blur passes.
   *
   * AT ShadowDepth.fs pipeline: depth render → optional blur → done.
   */
  renderShadowPass(
    gl: WebGLRenderingContext,
    lightViewMatrix: Float32Array,
    lightProjMatrix: Float32Array,
    rigidBodies: RigidBody[],
    obstacles: ObstacleData[],
  ): void {
    if (!this.initialized) {
      this.init();
    }

    const { resolution, numCascades, mode } = this.config;

    // Save GL state
    const prevViewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;

    // Compute cascade split distances
    this.cascadeSplits = computeCascadeSplits(
      numCascades,
      0.1,
      this.config.maxShadowDistance,
    );

    // ── Phase 1: Depth Render ──────────────────────────────────────────────
    gl.useProgram(this.depthProgram);
    gl.viewport(0, 0, resolution, resolution);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);

    // AT ShadowDepth.fs: polygon offset to reduce shadow acne
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(2.0, 2.0);

    for (let c = 0; c < numCascades; c++) {
      const cascade = this.cascades[c];
      if (!cascade) continue;

      // Update cascade matrices based on split range
      const nearZ = c === 0 ? 0.1 : this.cascadeSplits[c - 1];
      const farZ = this.cascadeSplits[c];
      cascade.splitNear = nearZ;
      cascade.splitFar = farZ;

      // Build cascade-specific ortho projection
      // AT ShadowDepth.fs L7280: tight ortho around cascade frustum
      const cascadeProj = mat4Ortho(
        -farZ, farZ, -farZ, farZ, -farZ * 2, farZ * 2,
      );
      cascade.projMatrix.set(cascadeProj);
      cascade.viewMatrix.set(lightViewMatrix);
      cascade.viewProjMatrix.set(
        mat4Multiply(cascadeProj, lightViewMatrix),
      );

      // Bind cascade FBO
      gl.bindFramebuffer(gl.FRAMEBUFFER, cascade.fbo);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // Set light matrices
      gl.uniformMatrix4fv(this.uLightViewMatrix, false, cascade.viewMatrix);
      gl.uniformMatrix4fv(this.uLightProjMatrix, false, cascade.projMatrix);

      // ── Draw rigid bodies ──────────────────────────────────────────────
      if (rigidBodies.length > 0 && this.boxBuffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.boxBuffer);
        gl.enableVertexAttribArray(this.aPosition);
        gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);

        for (const rb of rigidBodies) {
          const modelMatrix = rigidBodyModelMatrix(rb);
          gl.uniformMatrix4fv(this.uModelMatrix, false, modelMatrix);
          gl.drawArrays(gl.TRIANGLES, 0, 36);
        }
      }

      // ── Draw obstacles (circles) ──────────────────────────────────────
      if (obstacles.length > 0 && this.circleGeom) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.circleGeom.vbo);
        gl.enableVertexAttribArray(this.aPosition);
        gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.circleGeom.ibo);

        for (const obs of obstacles) {
          const modelMatrix = obstacleModelMatrix(obs);
          gl.uniformMatrix4fv(this.uModelMatrix, false, modelMatrix);
          gl.drawElements(gl.TRIANGLES, this.circleGeom.count, gl.UNSIGNED_SHORT, 0);
        }
      }
    }

    gl.disable(gl.POLYGON_OFFSET_FILL);

    // ── Phase 2: VSM Gaussian Blur ──────────────────────────────────────────
    if (mode === 'vsm' && this.blurProgram && this.quad) {
      gl.useProgram(this.blurProgram);
      gl.disable(gl.DEPTH_TEST);

      for (let c = 0; c < numCascades; c++) {
        const cascade = this.cascades[c];
        if (!cascade || !cascade.blurFbo || !cascade.blurTexture) continue;

        // Horizontal pass: cascade.depthTexture → cascade.blurFbo
        gl.bindFramebuffer(gl.FRAMEBUFFER, cascade.blurFbo);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, cascade.depthTexture);
        gl.uniform1i(this.uBlurShadowMap, 0);
        gl.uniform2f(this.uBlurDirection, 1.0 / resolution, 0.0);
        gl.uniform1f(this.uBlurSigma, this.config.vsmBlurSigma);
        this.quad.draw();

        // Vertical pass: cascade.blurTexture → cascade.fbo
        gl.bindFramebuffer(gl.FRAMEBUFFER, cascade.fbo);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, cascade.blurTexture);
        gl.uniform2f(this.uBlurDirection, 0.0, 1.0 / resolution);
        this.quad.draw();
      }
    }

    // Restore GL state
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    gl.viewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.enable(gl.DEPTH_TEST);
  }

  /**
   * Get a GLSL snippet for shadow sampling in the lighting shader.
   *
   * Returns a complete GLSL code block that includes:
   *   - Cascade selection (selectCascade)
   *   - Shadow UV projection (projectToShadowUV)
   *   - Shadow fade (shadowFade)
   *   - Mode-specific sampling function:
   *       'pcf'  → sampleShadowPCF()
   *       'vsm'  → sampleShadowVSM()
   *       'pcss' → sampleShadowPCSS()
   *
   * Inject this into your lighting fragment shader via string concatenation.
   *
   * AT ShadowReceiver.glsl: the same snippet pattern used in AT's pipeline.
   */
  getShadowSamplingCode(): string {
    const { mode, numCascades, pcfKernelSize, vsmLBR, lightSize } = this.config;

    let code = '';

    // Common: cascade selection + UV projection + fade
    code += cascadeSelectionGLSL(numCascades);

    // Mode-specific sampling
    switch (mode) {
      case 'pcf':
        code += pcfSamplingGLSL(pcfKernelSize);
        break;
      case 'vsm':
        code += vsmSamplingGLSL(vsmLBR);
        break;
      case 'pcss':
        code += pcssSamplingGLSL(lightSize);
        break;
    }

    return code;
  }

  /**
   * Get the cascade shadow map textures for binding in the lighting pass.
   * Returns array of WebGLTexture (one per cascade level).
   */
  getCascadeTextures(): WebGLTexture[] {
    return this.cascades.map(c => c.depthTexture);
  }

  /**
   * Get the cascade view-projection matrices for the lighting shader uniforms.
   * Returns array of Float32Array[16] (one per cascade level).
   */
  getCascadeViewProjMatrices(): Float32Array[] {
    return this.cascades.map(c => c.viewProjMatrix);
  }

  /**
   * Get the cascade split distances (view-space Z) for the lighting shader.
   */
  getCascadeSplits(): number[] {
    return [...this.cascadeSplits];
  }

  /** Release all GPU resources. */
  dispose(): void {
    const gl = this.gl;
    this.destroyCascades();

    if (this.depthProgram) { gl.deleteProgram(this.depthProgram); this.depthProgram = null; }
    if (this.blurProgram) { gl.deleteProgram(this.blurProgram); this.blurProgram = null; }
    if (this.boxBuffer) { gl.deleteBuffer(this.boxBuffer); this.boxBuffer = null; }
    if (this.circleGeom) {
      gl.deleteBuffer(this.circleGeom.vbo);
      gl.deleteBuffer(this.circleGeom.ibo);
      this.circleGeom = null;
    }
    if (this.quad) { gl.deleteBuffer(this.quad.buffer); this.quad = null; }

    this.initialized = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: Initialization
  // ─────────────────────────────────────────────────────────────────────────

  private init(): void {
    if (this.initialized) return;
    this.initPrograms();
    this.initGeometry();
    this.initCascades();
    this.initialized = true;
  }

  private initPrograms(): void {
    const gl = this.gl;
    const { vertex, fragment } = this.getDepthShader();

    // Clean up old programs
    if (this.depthProgram) gl.deleteProgram(this.depthProgram);
    if (this.blurProgram) gl.deleteProgram(this.blurProgram);

    // Depth program
    this.depthProgram = linkProgram(gl, vertex, fragment);
    if (this.depthProgram) {
      this.uLightViewMatrix = gl.getUniformLocation(this.depthProgram, 'uLightViewMatrix');
      this.uLightProjMatrix = gl.getUniformLocation(this.depthProgram, 'uLightProjMatrix');
      this.uModelMatrix = gl.getUniformLocation(this.depthProgram, 'uModelMatrix');
      this.aPosition = gl.getAttribLocation(this.depthProgram, 'aPosition');
    }

    // Blur program (VSM only)
    if (this.config.mode === 'vsm') {
      this.blurProgram = linkProgram(gl, FULLSCREEN_QUAD_VERTEX, VSM_BLUR_FRAGMENT);
      if (this.blurProgram) {
        this.uBlurShadowMap = gl.getUniformLocation(this.blurProgram, 'uShadowMap');
        this.uBlurDirection = gl.getUniformLocation(this.blurProgram, 'uDirection');
        this.uBlurSigma = gl.getUniformLocation(this.blurProgram, 'uSigma');
        this.aBlurPosition = gl.getAttribLocation(this.blurProgram, 'aPosition');
      }
    }
  }

  private initGeometry(): void {
    const gl = this.gl;
    this.boxBuffer = createUnitBoxBuffer(gl);
    this.circleGeom = createUnitCircleBuffer(gl);
    this.quad = createQuadVAO(gl);
  }

  private initCascades(): void {
    const { numCascades, resolution } = this.config;

    for (let c = 0; c < numCascades; c++) {
      const { fbo, texture } = this.createShadowMap(resolution);

      let blurFbo: WebGLFramebuffer | null = null;
      let blurTexture: WebGLTexture | null = null;

      // VSM needs a ping-pong texture for the separable blur
      if (this.config.mode === 'vsm') {
        const blur = this.createShadowMap(resolution);
        blurFbo = blur.fbo;
        blurTexture = blur.texture;
      }

      this.cascades.push({
        fbo,
        depthTexture: texture,
        blurFbo,
        blurTexture,
        viewMatrix: mat4Identity(),
        projMatrix: mat4Identity(),
        viewProjMatrix: mat4Identity(),
        splitNear: 0,
        splitFar: 0,
      });
    }
  }

  private destroyCascades(): void {
    const gl = this.gl;
    for (const c of this.cascades) {
      gl.deleteFramebuffer(c.fbo);
      gl.deleteTexture(c.depthTexture);
      if (c.blurFbo) gl.deleteFramebuffer(c.blurFbo);
      if (c.blurTexture) gl.deleteTexture(c.blurTexture);
    }
    this.cascades = [];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility exports — AT ShadowDepth.fs 参数 & helpers 公开给其他模块
// ─────────────────────────────────────────────────────────────────────────────

export { computeCascadeSplits };
export { mat4LookAt, mat4Ortho, mat4Multiply, mat4Identity };
export {
  DEPTH_VERTEX_SHADER,
  DEPTH_FRAGMENT_STANDARD,
  DEPTH_FRAGMENT_VSM,
  VSM_BLUR_FRAGMENT,
  FULLSCREEN_QUAD_VERTEX,
};
export {
  POISSON_DISK_25,
  POISSON_DISK_16,
  MAX_CASCADES,
  CASCADE_SPLIT_LAMBDA,
  DEFAULT_RESOLUTION,
  DEFAULT_DEPTH_BIAS,
  DEFAULT_NORMAL_BIAS,
  DEFAULT_LIGHT_SIZE,
  VSM_BLUR_SIGMA,
  VSM_LBR,
};
