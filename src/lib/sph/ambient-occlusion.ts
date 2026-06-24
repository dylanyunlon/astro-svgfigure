/**
 * ambient-occlusion.ts — M775: SSAO Screen-Space Ambient Occlusion
 * ─────────────────────────────────────────────────────────────────────────────
 * 屏幕空间环境光遮蔽——基于 depth buffer + normal buffer 的 hemisphere
 * sampling, 遮蔽区域变暗。Cell 之间缝隙自然产生阴影。bilateral blur 去噪。
 *
 * 算法概览
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0 ── SSAO Sampling ────────────────────────────────────────────────┐
 *   │  depthTex + normalTex + noiseTexture + sampleKernel → raw AO             │
 *   │  Per-pixel: reconstruct view-space position from depth, build TBN from   │
 *   │  normal + random rotation, take N hemisphere samples (cosine-weighted),   │
 *   │  compare sample depth vs depth buffer → occlusion factor.                │
 *   │                                                                          │
 *   │  Cell-specific: kernelRadius adapts to cell scale so cell-cell gaps      │
 *   │  naturally darken from geometric proximity in screen space.              │
 *   │  Range-check attenuation prevents halos on depth discontinuities.        │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ ssaoRawTex (r8unorm — raw occlusion, 0=occluded 1=open)
 *                ▼
 *   ┌─ Pass 1 ── Bilateral Blur (horizontal) ─────────────────────────────────┐
 *   │  ssaoRawTex → ssaoBlurHTex                                               │
 *   │  Depth-aware Gaussian blur in X: skip taps whose depth differs from      │
 *   │  center by more than depthThreshold → preserves cell boundary edges.     │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ ssaoBlurHTex (r8unorm)
 *                ▼
 *   ┌─ Pass 2 ── Bilateral Blur (vertical) ───────────────────────────────────┐
 *   │  ssaoBlurHTex → ssaoBlurredTex                                           │
 *   │  Same depth-aware Gaussian in Y.                                         │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ ssaoBlurredTex (r8unorm — final AO mask)
 *                ▼
 *   ┌─ Pass 3 ── Composite ───────────────────────────────────────────────────┐
 *   │  sceneTex × pow(ao, intensity) → dst                                     │
 *   │  Multiplicative darkening with configurable gamma curve.                 │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 *   const ssao = await SSAOPass.create(device, width, height, format);
 *   ssao.setParams({ kernelRadius: 0.5, intensity: 1.5 });
 *   ssao.render(encoder, depthView, normalView, sceneView, outputView, cameraUBO);
 *
 * Research: xiaodi #M775 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 公共常量 & 助手
// ─────────────────────────────────────────────────────────────────────────────









const WGSL_SSAO_COMMON = /* wgsl */ `
const PI      : f32 = 3.14159265358979323846;
const TWO_PI  : f32 = 6.28318530717958647693;
const HALF_PI : f32 = 1.57079632679489661923;

fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }

// ── 线性深度重建 ──────────────────────────────────────────────────────────────
// 从 reverse-Z depth buffer [0,1] 重建线性视空间 Z (正值, 向远处增大)
fn linearizeDepth(d: f32, near: f32, far: f32) -> f32 {
    // reverse-Z projection: z_ndc = near·far / (far - d·(far - near))
    return near * far / (far - d * (far - near));
}

// ── 视空间位置重建 ─────────────────────────────────────────────────────────────
// 从 UV + 线性深度 + 逆投影参数重建 view-space position
fn reconstructViewPos(uv: vec2f, linearZ: f32, invProjX: f32, invProjY: f32) -> vec3f {
    // NDC → view: x_view = (uv.x * 2 - 1) / proj[0][0] * z
    //              y_view = (uv.y * 2 - 1) / proj[1][1] * z (翻转 Y)
    let ndcX = uv.x * 2.0 - 1.0;
    let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
    return vec3f(ndcX * invProjX * linearZ, ndcY * invProjY * linearZ, linearZ);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 0: SSAO hemisphere sampling
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSAO_SAMPLE = /* wgsl */ `
${WGSL_SSAO_COMMON}

// ── Uniforms ──────────────────────────────────────────────────────────────────
struct SSAOUniforms {
    // 投影重建
    nearPlane     : f32,
    farPlane      : f32,
    invProjX      : f32,    // 1.0 / projection[0][0]
    invProjY      : f32,    // 1.0 / projection[1][1]
    // SSAO 参数
    kernelRadius  : f32,    // 采样半球半径 (view-space, 默认 0.5)
    bias          : f32,    // 深度偏移防止自遮蔽 (默认 0.025)
    intensity     : f32,    // 遮蔽强度乘数 (默认 1.5)
    kernelSize    : u32,    // 半球采样数 (16/32/64, 默认 32)
    // 分辨率
    resolution    : vec2f,  // 渲染目标宽高 (像素)
    noiseScale    : vec2f,  // resolution / noiseTexSize (tile 噪声纹理)
}

// ── 采样核 (预计算, 存储于 storage buffer) ────────────────────────────────────
// 半球内 cosine-weighted 分布的 vec4f (xyz = direction, w = 0 padding)
@group(0) @binding(0) var<uniform>        u_ssao   : SSAOUniforms;
@group(0) @binding(1) var<storage, read>  u_kernel : array<vec4f>;
@group(0) @binding(2) var                 t_depth  : texture_2d<f32>;
@group(0) @binding(3) var                 t_normal : texture_2d<f32>;
@group(0) @binding(4) var                 t_noise  : texture_2d<f32>;
@group(0) @binding(5) var                 smp_clamp: sampler;
@group(0) @binding(6) var                 smp_repeat: sampler;

struct VSOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}

// ── 全屏三角形 VS ────────────────────────────────────────────────────────────
@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32( vi         & 2u) * 2.0 - 1.0;
    var o: VSOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

// ── 片元: SSAO 采样 ──────────────────────────────────────────────────────────
@fragment fn fs_ssao(in: VSOut) -> @location(0) vec4f {
    let uv = in.uv;
    let texSize = u_ssao.resolution;

    // 1. 采样深度, 重建视空间位置
    let rawDepth = textureSample(t_depth, smp_clamp, uv).r;
    if (rawDepth <= 0.0 || rawDepth >= 1.0) {
        return vec4f(1.0);   // 天空/远裁剪 → 无遮蔽
    }
    let linZ    = linearizeDepth(rawDepth, u_ssao.nearPlane, u_ssao.farPlane);
    let viewPos = reconstructViewPos(uv, linZ, u_ssao.invProjX, u_ssao.invProjY);

    // 2. 采样视空间法线 (G-Buffer 编码: rgb = N * 0.5 + 0.5)
    let rawNorm  = textureSample(t_normal, smp_clamp, uv).rgb;
    let viewNorm = normalize(rawNorm * 2.0 - 1.0);

    // 3. 噪声旋转向量 (4×4 tile, 随机绕法线旋转半球 → 打破 banding)
    let noiseUV   = uv * u_ssao.noiseScale;
    let randomVec = normalize(textureSample(t_noise, smp_repeat, noiseUV).rgb * 2.0 - 1.0);

    // 4. 构建切线空间 TBN (Gram-Schmidt 正交化)
    let tangent   = normalize(randomVec - viewNorm * dot(randomVec, viewNorm));
    let bitangent = cross(viewNorm, tangent);
    let TBN       = mat3x3f(tangent, bitangent, viewNorm);

    // 5. 半球采样累积遮蔽
    var occlusion = 0.0;
    let radius    = u_ssao.kernelRadius;
    let bias      = u_ssao.bias;
    let kCount    = u_ssao.kernelSize;

    for (var i = 0u; i < kCount; i = i + 1u) {
        // 将核方向从切线空间转到视空间
        let sampleDir = TBN * u_kernel[i].xyz;
        let samplePos = viewPos + sampleDir * radius;

        // 投影到屏幕空间 UV
        let projX = samplePos.x / (samplePos.z * u_ssao.invProjX);
        let projY = samplePos.y / (samplePos.z * u_ssao.invProjY);
        let sampleUV = vec2f(
            projX * 0.5 + 0.5,
            1.0 - (projY * 0.5 + 0.5)
        );

        // 采样该处深度
        let sDepth = textureSample(t_depth, smp_clamp, sampleUV).r;
        let sLinZ  = linearizeDepth(sDepth, u_ssao.nearPlane, u_ssao.farPlane);

        // 遮蔽判定: 样本深度 < 当前样本位置深度 → 被遮挡
        // range-check: 只有距离足够近才贡献遮蔽 (避免远处物体产生假遮蔽)
        let rangeCheck = smoothstep(0.0, 1.0, radius / abs(linZ - sLinZ + 1e-6));
        let isOccluded = select(0.0, 1.0, sLinZ < samplePos.z - bias);
        occlusion += isOccluded * rangeCheck;
    }

    // 归一化 & 反转: 0=全遮蔽, 1=无遮蔽
    let ao = 1.0 - (occlusion / f32(kCount));
    // 强度控制 (幂函数曲线加深暗部)
    let finalAO = pow(saturate_f(ao), u_ssao.intensity);

    return vec4f(finalAO, finalAO, finalAO, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 1/2: Bilateral Gaussian Blur (深度感知去噪)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_BILATERAL_BLUR = /* wgsl */ `
${WGSL_SSAO_COMMON}

struct BlurUniforms {
    direction      : vec2f,  // (1,0) 水平 or (0,1) 垂直
    depthThreshold : f32,    // 深度断续阈值 (默认 0.1)
    normalThreshold: f32,    // 法线夹角阈值 cos (默认 0.9)
    nearPlane      : f32,
    farPlane       : f32,
    texelSize      : vec2f,  // 1.0 / resolution
}

@group(0) @binding(0) var<uniform> u_blur  : BlurUniforms;
@group(0) @binding(1) var          t_ssao  : texture_2d<f32>;
@group(0) @binding(2) var          t_depth : texture_2d<f32>;
@group(0) @binding(3) var          t_normal: texture_2d<f32>;
@group(0) @binding(4) var          smp     : sampler;

struct VSOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}

@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32( vi         & 2u) * 2.0 - 1.0;
    var o: VSOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

// ── 7-tap bilateral Gaussian ────────────────────────────────────────────────
// 权重: σ=2.0 → [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]
// 化简为对称 4-tap (center + 3 offsets)
@fragment fn fs_blur(in: VSOut) -> @location(0) vec4f {
    let uv = in.uv;
    let dir = u_blur.direction * u_blur.texelSize;

    // 中心像素参考值
    let centerAO    = textureSample(t_ssao, smp, uv).r;
    let centerDepthR= textureSample(t_depth, smp, uv).r;
    let centerLinZ  = linearizeDepth(centerDepthR, u_blur.nearPlane, u_blur.farPlane);
    let centerNorm  = normalize(textureSample(t_normal, smp, uv).rgb * 2.0 - 1.0);

    // Gaussian 权重 (σ ≈ 2.0, 7-tap)
    let weights = array<f32, 4>(0.383, 0.242, 0.061, 0.006);
    let offsets = array<f32, 4>(0.0,   1.0,   2.0,   3.0);

    var totalAO     = centerAO * weights[0];
    var totalWeight = weights[0];

    for (var i = 1u; i < 4u; i = i + 1u) {
        let w = weights[i];
        let o = offsets[i];

        for (var s = -1; s <= 1; s = s + 2) {
            let sampleUV = uv + dir * (o * f32(s));

            // 采样该位置的深度 & 法线
            let sDepthR  = textureSample(t_depth, smp, sampleUV).r;
            let sLinZ    = linearizeDepth(sDepthR, u_blur.nearPlane, u_blur.farPlane);
            let sNorm    = normalize(textureSample(t_normal, smp, sampleUV).rgb * 2.0 - 1.0);

            // ── bilateral 权重 ────────────────────────────────────────────
            // 深度权重: 深度差超过阈值 → 权重→0 (保留 cell 边界)
            let depthDiff = abs(centerLinZ - sLinZ);
            let wDepth    = exp(-depthDiff * depthDiff / (u_blur.depthThreshold * u_blur.depthThreshold + 1e-6));

            // 法线权重: 法线朝向差异大 → 权重→0
            let nDot    = max(0.0, dot(centerNorm, sNorm));
            let wNormal = select(0.0, 1.0, nDot > u_blur.normalThreshold);

            let bilateralW = w * wDepth * wNormal;

            let sAO = textureSample(t_ssao, smp, sampleUV).r;
            totalAO    += sAO * bilateralW;
            totalWeight += bilateralW;
        }
    }

    let blurred = totalAO / max(totalWeight, 1e-6);
    return vec4f(blurred, blurred, blurred, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Pass 3: Composite (SSAO × scene)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SSAO_COMPOSITE = /* wgsl */ `
struct CompositeUniforms {
    aoStrength : f32,   // 最终遮蔽混合权重 [0, 1], 默认 1.0
    aoGamma    : f32,   // 遮蔽 gamma 曲线 (>1 加深暗部, 默认 1.0)
    _p0        : f32,
    _p1        : f32,
}

@group(0) @binding(0) var<uniform> u_comp  : CompositeUniforms;
@group(0) @binding(1) var          t_scene : texture_2d<f32>;
@group(0) @binding(2) var          t_ao    : texture_2d<f32>;
@group(0) @binding(3) var          smp     : sampler;

struct VSOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}

@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32( vi         & 2u) * 2.0 - 1.0;
    var o: VSOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

@fragment fn fs_composite(in: VSOut) -> @location(0) vec4f {
    let sceneColor = textureSample(t_scene, smp, in.uv);
    let rawAO      = textureSample(t_ao,    smp, in.uv).r;

    // gamma 曲线调节遮蔽对比度
    let ao = pow(clamp(rawAO, 0.0, 1.0), u_comp.aoGamma);

    // 混合: 1.0 = 完全应用 AO, 0.0 = 无遮蔽
    let finalAO = mix(1.0, ao, u_comp.aoStrength);

    return vec4f(sceneColor.rgb * finalAO, sceneColor.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript 接口 & 参数类型
// ─────────────────────────────────────────────────────────────────────────────

/** SSAO 可调参数 */
export interface SSAOParams {
  /** 半球采样半径 (view-space units, 默认 0.5) */
  kernelRadius: number;
  /** 深度偏移: 防止平面自遮蔽 (默认 0.025) */
  bias: number;
  /** 遮蔽强度指数: >1 加深缝隙阴影 (默认 1.5) */
  intensity: number;
  /** 半球采样数 16|32|64 (默认 32) */
  kernelSize: number;
  /** 近裁剪面距离 */
  nearPlane: number;
  /** 远裁剪面距离 */
  farPlane: number;
  /** 逆投影 X 分量: 1 / projection[0][0] */
  invProjX: number;
  /** 逆投影 Y 分量: 1 / projection[1][1] */
  invProjY: number;
}

/** Bilateral blur 可调参数 */
export interface BlurParams {
  /** 深度断续阈值 (线性 Z 空间, 默认 0.1) */
  depthThreshold: number;
  /** 法线夹角阈值 cos 值 (默认 0.9, 即 ~26°) */
  normalThreshold: number;
}

/** Composite 可调参数 */
export interface CompositeParams {
  /** AO 混合强度 [0, 1] (默认 1.0) */
  aoStrength: number;
  /** AO gamma 曲线 (默认 1.0) */
  aoGamma: number;
}

/** 全部 SSAO 配置 */
export interface SSAOConfig {
  ssao: SSAOParams;
  blur: BlurParams;
  composite: CompositeParams;
}

// ─────────────────────────────────────────────────────────────────────────────
// 默认参数
// ─────────────────────────────────────────────────────────────────────────────

export const DEFAULT_SSAO_PARAMS: SSAOParams = {
  kernelRadius: 0.5,
  bias: 0.025,
  intensity: 1.5,
  kernelSize: 32,
  nearPlane: 0.1,
  farPlane: 100.0,
  invProjX: 1.0,
  invProjY: 1.0,
};

export const DEFAULT_BLUR_PARAMS: BlurParams = {
  depthThreshold: 0.1,
  normalThreshold: 0.9,
};

export const DEFAULT_COMPOSITE_PARAMS: CompositeParams = {
  aoStrength: 1.0,
  aoGamma: 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// 采样核生成器 — cosine-weighted hemisphere sampling
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 生成半球采样核: 每个样本 (x, y, z, 0) 分布在法线方向半球内,
 * 样本密度向中心聚集 (加速 lerp → 近处贡献更大权重)。
 *
 * @param count - 采样数 (16/32/64)
 * @returns Float32Array (count × 4 floats, 适合 GPU storage buffer)
 */
export function generateSSAOKernel(count: number): Float32Array {
  const kernel = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    // 准随机 Hammersley 序列分布在半球上
    const xi1 = i / count;
    const xi2 = vanDerCorput(i);

    // 从球面坐标到笛卡尔 (半球: z ≥ 0)
    const phi = TWO_PI_JS * xi1;
    const cosTheta = Math.sqrt(1.0 - xi2);   // cosine-weighted
    const sinTheta = Math.sqrt(xi2);

    let x = Math.cos(phi) * sinTheta;
    let y = Math.sin(phi) * sinTheta;
    let z = cosTheta;

    // 将样本长度从 0→1 非均匀分布: 靠近原点密集
    // scale = lerp(0.1, 1.0, (i/N)²)  → 核心近处更多样本
    let scale = i / count;
    scale = 0.1 + scale * scale * 0.9;

    x *= scale;
    y *= scale;
    z *= scale;

    const off = i * 4;
    kernel[off] = x;
    kernel[off + 1] = y;
    kernel[off + 2] = z;
    kernel[off + 3] = 0;   // padding
  }
  return kernel;
}

const TWO_PI_JS = Math.PI * 2;

/** Van der Corput 序列 (base-2), 用于准随机 Hammersley 采样 */
function vanDerCorput(index: number): number {
  let bits = index;
  bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >>> 1);
  bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >>> 2);
  bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >>> 4);
  bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >>> 8);
  bits = (bits << 16) | (bits >>> 16);
  return (bits >>> 0) * 2.3283064365386963e-10; // / 0x100000000
}

/**
 * 生成 4×4 随机旋转噪声纹理 (tile 到全屏打破采样 banding)。
 * 每个像素存储 (randomX, randomY, 0) 归一化方向, 编码为 RGB8。
 *
 * @returns { data: Uint8Array (4×4×4 RGBA), width: 4, height: 4 }
 */
export function generateNoiseTexture(): {
  data: Uint8Array;
  width: number;
  height: number;
} {
  const SIZE = 4;
  const data = new Uint8Array(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    // 在切线平面内旋转: 只需要 XY 方向
    const angle = Math.random() * TWO_PI_JS;
    const x = Math.cos(angle);
    const y = Math.sin(angle);

    const off = i * 4;
    data[off] = Math.floor((x * 0.5 + 0.5) * 255);
    data[off + 1] = Math.floor((y * 0.5 + 0.5) * 255);
    data[off + 2] = 0;
    data[off + 3] = 255;
  }
  return { data, width: SIZE, height: SIZE };
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer 打包 (std140/WGSL 16-byte 对齐)
// ─────────────────────────────────────────────────────────────────────────────

/** 打包 SSAOUniforms → Float32Array */
export function packSSAOUniforms(
  p: SSAOParams,
  width: number,
  height: number,
): Float32Array {
  // 布局 (每行 4 × f32 = 16 bytes):
  //   [0]  nearPlane, farPlane, invProjX, invProjY
  //   [4]  kernelRadius, bias, intensity, kernelSize (as f32 bitcast)
  //   [8]  resolution.xy, noiseScale.xy
  const buf = new Float32Array(12);
  buf[0] = p.nearPlane;
  buf[1] = p.farPlane;
  buf[2] = p.invProjX;
  buf[3] = p.invProjY;
  buf[4] = p.kernelRadius;
  buf[5] = p.bias;
  buf[6] = p.intensity;
  // kernelSize 是 u32, 写 bitcast
  new Uint32Array(buf.buffer, 7 * 4, 1)[0] = p.kernelSize;
  buf[8] = width;
  buf[9] = height;
  buf[10] = width / 4;   // noiseScale = resolution / noiseTexSize(4)
  buf[11] = height / 4;
  return buf;
}

/** 打包 BlurUniforms → Float32Array */
export function packBlurUniforms(
  p: BlurParams,
  direction: [number, number],
  nearPlane: number,
  farPlane: number,
  width: number,
  height: number,
): Float32Array {
  // 布局:
  //   [0]  direction.xy, depthThreshold, normalThreshold
  //   [4]  nearPlane, farPlane, texelSize.xy
  const buf = new Float32Array(8);
  buf[0] = direction[0];
  buf[1] = direction[1];
  buf[2] = p.depthThreshold;
  buf[3] = p.normalThreshold;
  buf[4] = nearPlane;
  buf[5] = farPlane;
  buf[6] = 1.0 / width;
  buf[7] = 1.0 / height;
  return buf;
}

/** 打包 CompositeUniforms → Float32Array */
export function packCompositeUniforms(p: CompositeParams): Float32Array {
  // 布局:
  //   [0]  aoStrength, aoGamma, _p0, _p1
  const buf = new Float32Array(4);
  buf[0] = p.aoStrength;
  buf[1] = p.aoGamma;
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// SSAOPass — 完整 4-pass SSAO 管线
// ─────────────────────────────────────────────────────────────────────────────

export class SSAOPass {
  private device: any /*GPUDevice*/;
  private width: number;
  private height: number;
  private format: GPUTextureFormat;

  // ── 管线 ──────────────────────────────────────────────────────────────────
  private ssaoPipeline: any /*GPURenderPipeline*/;
  private blurPipeline: any /*GPURenderPipeline*/;
  private compositePipeline: any /*GPURenderPipeline*/;

  // ── 纹理 ──────────────────────────────────────────────────────────────────
  private ssaoRawTex: GPUTexture;
  private ssaoBlurHTex: GPUTexture;
  private ssaoBlurredTex: GPUTexture;
  private noiseTex: GPUTexture;

  // ── Buffers ──────────────────────────────────────────────────────────────
  private ssaoUniformBuf: any /*GPUBuffer*/;
  private blurHUniformBuf: any /*GPUBuffer*/;
  private blurVUniformBuf: any /*GPUBuffer*/;
  private compositeUniformBuf: any /*GPUBuffer*/;
  private kernelBuf: any /*GPUBuffer*/;

  // ── Samplers ─────────────────────────────────────────────────────────────
  private samplerClamp: GPUSampler;
  private samplerRepeat: GPUSampler;

  // ── Bind group layouts ───────────────────────────────────────────────────
  private ssaoBGL: GPUBindGroupLayout;
  private blurBGL: GPUBindGroupLayout;
  private compositeBGL: GPUBindGroupLayout;

  // ── 参数 ──────────────────────────────────────────────────────────────────
  private ssaoParams: SSAOParams;
  private blurParams: BlurParams;
  private compositeParams: CompositeParams;

  private constructor(
    device: any /*GPUDevice*/,
    width: number,
    height: number,
    format: GPUTextureFormat,
    ssaoPipeline: any /*GPURenderPipeline*/,
    blurPipeline: any /*GPURenderPipeline*/,
    compositePipeline: any /*GPURenderPipeline*/,
    ssaoRawTex: GPUTexture,
    ssaoBlurHTex: GPUTexture,
    ssaoBlurredTex: GPUTexture,
    noiseTex: GPUTexture,
    ssaoUniformBuf: any /*GPUBuffer*/,
    blurHUniformBuf: any /*GPUBuffer*/,
    blurVUniformBuf: any /*GPUBuffer*/,
    compositeUniformBuf: any /*GPUBuffer*/,
    kernelBuf: any /*GPUBuffer*/,
    samplerClamp: GPUSampler,
    samplerRepeat: GPUSampler,
    ssaoBGL: GPUBindGroupLayout,
    blurBGL: GPUBindGroupLayout,
    compositeBGL: GPUBindGroupLayout,
  ) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.format = format;
    this.ssaoPipeline = ssaoPipeline;
    this.blurPipeline = blurPipeline;
    this.compositePipeline = compositePipeline;
    this.ssaoRawTex = ssaoRawTex;
    this.ssaoBlurHTex = ssaoBlurHTex;
    this.ssaoBlurredTex = ssaoBlurredTex;
    this.noiseTex = noiseTex;
    this.ssaoUniformBuf = ssaoUniformBuf;
    this.blurHUniformBuf = blurHUniformBuf;
    this.blurVUniformBuf = blurVUniformBuf;
    this.compositeUniformBuf = compositeUniformBuf;
    this.kernelBuf = kernelBuf;
    this.samplerClamp = samplerClamp;
    this.samplerRepeat = samplerRepeat;
    this.ssaoBGL = ssaoBGL;
    this.blurBGL = blurBGL;
    this.compositeBGL = compositeBGL;
    this.ssaoParams = { ...DEFAULT_SSAO_PARAMS };
    this.blurParams = { ...DEFAULT_BLUR_PARAMS };
    this.compositeParams = { ...DEFAULT_COMPOSITE_PARAMS };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 工厂: 异步创建完整 SSAO 管线
  // ─────────────────────────────────────────────────────────────────────────

  static async create(
    device: any /*GPUDevice*/,
    width: number,
    height: number,
    format: GPUTextureFormat = 'bgra8unorm',
  ): Promise<SSAOPass> {
    // ── Samplers ──────────────────────────────────────────────────────────
    const samplerClamp = device.createSampler({
      label: 'ssao-sampler-clamp',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    const samplerRepeat = device.createSampler({
      label: 'ssao-sampler-repeat',
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    // ── 中间纹理 (r8unorm 单通道 AO) ──────────────────────────────────────
    const aoTexDesc = (label: string): GPUTextureDescriptor => ({
      label,
      size: [width, height],
      format: 'r8unorm',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    const ssaoRawTex = device.createTexture(aoTexDesc('ssao-raw'));
    const ssaoBlurHTex = device.createTexture(aoTexDesc('ssao-blur-h'));
    const ssaoBlurredTex = device.createTexture(aoTexDesc('ssao-blurred'));

    // ── 噪声纹理 4×4 ──────────────────────────────────────────────────────
    const noiseData = generateNoiseTexture();
    const noiseTex = device.createTexture({
      label: 'ssao-noise-4x4',
      size: [noiseData.width, noiseData.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: noiseTex },
      noiseData.data,
      { bytesPerRow: noiseData.width * 4 },
      [noiseData.width, noiseData.height],
    );

    // ── 采样核 (storage buffer) ──────────────────────────────────────────
    const kernelData = generateSSAOKernel(DEFAULT_SSAO_PARAMS.kernelSize);
    const kernelBuf = device.createBuffer({
      label: 'ssao-kernel',
      size: kernelData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(kernelBuf, 0, kernelData);

    // ── Uniform buffers ──────────────────────────────────────────────────
    const ssaoUniformBuf = device.createBuffer({
      label: 'ssao-uniforms',
      size: 48,   // 12 × f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blurHUniformBuf = device.createBuffer({
      label: 'ssao-blur-h-uniforms',
      size: 32,   // 8 × f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blurVUniformBuf = device.createBuffer({
      label: 'ssao-blur-v-uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const compositeUniformBuf = device.createBuffer({
      label: 'ssao-composite-uniforms',
      size: 16,   // 4 × f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── SSAO pass — bind group layout + pipeline ──────────────────────────
    const ssaoBGL = device.createBindGroupLayout({
      label: 'ssao-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const ssaoModule = device.createShaderModule({
      label: 'ssao-sample-module',
      code: WGSL_SSAO_SAMPLE,
    });

    const ssaoPipeline = device.createRenderPipeline({
      label: 'ssao-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [ssaoBGL] }),
      vertex: { module: ssaoModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: ssaoModule,
        entryPoint: 'fs_ssao',
        targets: [{ format: 'r8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Blur pass — bind group layout + pipeline ──────────────────────────
    const blurBGL = device.createBindGroupLayout({
      label: 'ssao-blur-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const blurModule = device.createShaderModule({
      label: 'ssao-blur-module',
      code: WGSL_BILATERAL_BLUR,
    });

    const blurPipeline = device.createRenderPipeline({
      label: 'ssao-blur-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [blurBGL] }),
      vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: blurModule,
        entryPoint: 'fs_blur',
        targets: [{ format: 'r8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Composite pass — bind group layout + pipeline ─────────────────────
    const compositeBGL = device.createBindGroupLayout({
      label: 'ssao-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const compositeModule = device.createShaderModule({
      label: 'ssao-composite-module',
      code: WGSL_SSAO_COMPOSITE,
    });

    const compositePipeline = device.createRenderPipeline({
      label: 'ssao-composite-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBGL] }),
      vertex: { module: compositeModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fs_composite',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    return new SSAOPass(
      device,
      width,
      height,
      format,
      ssaoPipeline,
      blurPipeline,
      compositePipeline,
      ssaoRawTex,
      ssaoBlurHTex,
      ssaoBlurredTex,
      noiseTex,
      ssaoUniformBuf,
      blurHUniformBuf,
      blurVUniformBuf,
      compositeUniformBuf,
      kernelBuf,
      samplerClamp,
      samplerRepeat,
      ssaoBGL,
      blurBGL,
      compositeBGL,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 参数更新
  // ─────────────────────────────────────────────────────────────────────────

  /** 更新 SSAO 采样参数 (部分覆盖) */
  setSSAOParams(partial: Partial<SSAOParams>): void {
    Object.assign(this.ssaoParams, partial);
    this.device.queue.writeBuffer(
      this.ssaoUniformBuf,
      0,
      packSSAOUniforms(this.ssaoParams, this.width, this.height),
    );
  }

  /** 更新 blur 参数 (部分覆盖) */
  setBlurParams(partial: Partial<BlurParams>): void {
    Object.assign(this.blurParams, partial);
    this._updateBlurUniforms();
  }

  /** 更新 composite 参数 (部分覆盖) */
  setCompositeParams(partial: Partial<CompositeParams>): void {
    Object.assign(this.compositeParams, partial);
    this.device.queue.writeBuffer(
      this.compositeUniformBuf,
      0,
      packCompositeUniforms(this.compositeParams),
    );
  }

  /** 一次性设置全部参数 */
  setParams(partial: Partial<SSAOConfig>): void {
    if (partial.ssao) this.setSSAOParams(partial.ssao);
    if (partial.blur) this.setBlurParams(partial.blur);
    if (partial.composite) this.setCompositeParams(partial.composite);
  }

  private _updateBlurUniforms(): void {
    this.device.queue.writeBuffer(
      this.blurHUniformBuf,
      0,
      packBlurUniforms(
        this.blurParams,
        [1, 0],
        this.ssaoParams.nearPlane,
        this.ssaoParams.farPlane,
        this.width,
        this.height,
      ),
    );
    this.device.queue.writeBuffer(
      this.blurVUniformBuf,
      0,
      packBlurUniforms(
        this.blurParams,
        [0, 1],
        this.ssaoParams.nearPlane,
        this.ssaoParams.farPlane,
        this.width,
        this.height,
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 渲染 — 4-pass 完整管线
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * 执行完整 SSAO 后处理管线。
   *
   * @param encoder     — 当前帧 GPUCommandEncoder
   * @param depthView   — G-Buffer depth (r32float 或 depth24plus)
   * @param normalView  — G-Buffer view-space normal (rgba8unorm, rgb = N*0.5+0.5)
   * @param sceneView   — 场景颜色纹理 (用于 composite)
   * @param outputView  — 最终输出目标
   */
  render(
    encoder: any /*GPUCommandEncoder*/,
    depthView: any /*GPUTextureView*/,
    normalView: any /*GPUTextureView*/,
    sceneView: any /*GPUTextureView*/,
    outputView: any /*GPUTextureView*/,
  ): void {
    // ── Pass 0: SSAO hemisphere sampling ────────────────────────────────
    const ssaoBG = this.device.createBindGroup({
      label: 'ssao-pass0-bg',
      layout: this.ssaoBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ssaoUniformBuf } },
        { binding: 1, resource: { buffer: this.kernelBuf } },
        { binding: 2, resource: depthView },
        { binding: 3, resource: normalView },
        { binding: 4, resource: this.noiseTex.createView() },
        { binding: 5, resource: this.samplerClamp },
        { binding: 6, resource: this.samplerRepeat },
      ],
    });

    const ssaoPass = encoder.beginRenderPass({
      label: 'ssao-pass0',
      colorAttachments: [
        {
          view: this.ssaoRawTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
        },
      ],
    });
    ssaoPass.setPipeline(this.ssaoPipeline);
    ssaoPass.setBindGroup(0, ssaoBG);
    ssaoPass.draw(3);
    ssaoPass.end();

    // ── Pass 1: Bilateral blur — horizontal ─────────────────────────────
    const blurHBG = this.device.createBindGroup({
      label: 'ssao-blur-h-bg',
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: { buffer: this.blurHUniformBuf } },
        { binding: 1, resource: this.ssaoRawTex.createView() },
        { binding: 2, resource: depthView },
        { binding: 3, resource: normalView },
        { binding: 4, resource: this.samplerClamp },
      ],
    });

    const blurHPass = encoder.beginRenderPass({
      label: 'ssao-pass1-blur-h',
      colorAttachments: [
        {
          view: this.ssaoBlurHTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
        },
      ],
    });
    blurHPass.setPipeline(this.blurPipeline);
    blurHPass.setBindGroup(0, blurHBG);
    blurHPass.draw(3);
    blurHPass.end();

    // ── Pass 2: Bilateral blur — vertical ───────────────────────────────
    const blurVBG = this.device.createBindGroup({
      label: 'ssao-blur-v-bg',
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: { buffer: this.blurVUniformBuf } },
        { binding: 1, resource: this.ssaoBlurHTex.createView() },
        { binding: 2, resource: depthView },
        { binding: 3, resource: normalView },
        { binding: 4, resource: this.samplerClamp },
      ],
    });

    const blurVPass = encoder.beginRenderPass({
      label: 'ssao-pass2-blur-v',
      colorAttachments: [
        {
          view: this.ssaoBlurredTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
        },
      ],
    });
    blurVPass.setPipeline(this.blurPipeline);
    blurVPass.setBindGroup(0, blurVBG);
    blurVPass.draw(3);
    blurVPass.end();

    // ── Pass 3: Composite (scene × AO) ──────────────────────────────────
    const compositeBG = this.device.createBindGroup({
      label: 'ssao-composite-bg',
      layout: this.compositeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniformBuf } },
        { binding: 1, resource: sceneView },
        { binding: 2, resource: this.ssaoBlurredTex.createView() },
        { binding: 3, resource: this.samplerClamp },
      ],
    });

    const compositePass = encoder.beginRenderPass({
      label: 'ssao-pass3-composite',
      colorAttachments: [
        {
          view: outputView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    compositePass.setPipeline(this.compositePipeline);
    compositePass.setBindGroup(0, compositeBG);
    compositePass.draw(3);
    compositePass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 分辨率变更 — 重建中间纹理
  // ─────────────────────────────────────────────────────────────────────────

  /** 窗口 / viewport 尺寸变更时调用 */
  resize(newWidth: number, newHeight: number): void {
    if (newWidth === this.width && newHeight === this.height) return;

    this.width = newWidth;
    this.height = newHeight;

    // 销毁旧中间纹理
    this.ssaoRawTex.destroy();
    this.ssaoBlurHTex.destroy();
    this.ssaoBlurredTex.destroy();

    // 重建
    const aoTexDesc = (label: string): GPUTextureDescriptor => ({
      label,
      size: [newWidth, newHeight],
      format: 'r8unorm' as GPUTextureFormat,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    this.ssaoRawTex = this.device.createTexture(aoTexDesc('ssao-raw'));
    this.ssaoBlurHTex = this.device.createTexture(aoTexDesc('ssao-blur-h'));
    this.ssaoBlurredTex = this.device.createTexture(aoTexDesc('ssao-blurred'));

    // 更新 uniform (分辨率改变影响 noiseScale / texelSize)
    this.setSSAOParams({});
    this._updateBlurUniforms();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 采样核重新生成 (切换 kernelSize 时)
  // ─────────────────────────────────────────────────────────────────────────

  /** 更换采样数并重新生成核 */
  setKernelSize(count: number): void {
    this.ssaoParams.kernelSize = count;
    const kernelData = generateSSAOKernel(count);

    // 如果新大小超过原 buffer → 需要重建
    if (kernelData.byteLength > this.kernelBuf.size) {
      this.kernelBuf.destroy();
      this.kernelBuf = this.device.createBuffer({
        label: 'ssao-kernel',
        size: kernelData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.kernelBuf, 0, kernelData);
    this.setSSAOParams({});
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 只获取 AO 纹理 (供其他材质模块采样, 如 PBR 的 ao 输入)
  // ─────────────────────────────────────────────────────────────────────────

  /** 获取最终模糊后的 AO 纹理视图, 可直接绑定到 PBR 材质的 ao 采样器 */
  get aoTextureView(): any /*GPUTextureView*/ {
    return this.ssaoBlurredTex.createView();
  }

  /** 获取原始 (未模糊) AO 纹理视图, 用于调试 */
  get rawAOTextureView(): any /*GPUTextureView*/ {
    return this.ssaoRawTex.createView();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WGSL 源码 (调试用)
  // ─────────────────────────────────────────────────────────────────────────

  get wgslSources(): {
    ssaoSample: string;
    bilateralBlur: string;
    composite: string;
  } {
    return {
      ssaoSample: WGSL_SSAO_SAMPLE,
      bilateralBlur: WGSL_BILATERAL_BLUR,
      composite: WGSL_SSAO_COMPOSITE,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 销毁
  // ─────────────────────────────────────────────────────────────────────────

  destroy(): void {
    this.ssaoRawTex.destroy();
    this.ssaoBlurHTex.destroy();
    this.ssaoBlurredTex.destroy();
    this.noiseTex.destroy();
    this.ssaoUniformBuf.destroy();
    this.blurHUniformBuf.destroy();
    this.blurVUniformBuf.destroy();
    this.compositeUniformBuf.destroy();
    this.kernelBuf.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 导出 WGSL 片段 — 供其他模块组合使用
// ─────────────────────────────────────────────────────────────────────────────

export const SSAO_WGSL = {
  /** 公共助手 (linearizeDepth, reconstructViewPos) */
  common: WGSL_SSAO_COMMON,
  /** Pass 0: SSAO hemisphere sampling 着色器 */
  ssaoSample: WGSL_SSAO_SAMPLE,
  /** Pass 1/2: Bilateral blur 着色器 */
  bilateralBlur: WGSL_BILATERAL_BLUR,
  /** Pass 3: Composite 着色器 */
  composite: WGSL_SSAO_COMPOSITE,
};

// ─────────────────────────────────────────────────────────────────────────────
// 内联自测 ($ npx tsx src/lib/sph/ambient-occlusion.ts)
// ─────────────────────────────────────────────────────────────────────────────

export function _selfTest(): void {
  let passed = 0;
  let failed = 0;

  function eq(label: string, got: number, want: number, eps = 1e-6): void {
    if (Math.abs(got - want) <= eps) {
      passed++;
    } else {
      failed++;
      console.error(`  FAIL  ${label}: got ${got}, want ${want}`);
    }
  }
  function ok(label: string, cond: boolean): void {
    if (cond) {
      passed++;
    } else {
      failed++;
      console.error(`  FAIL  ${label}`);
    }
  }
  function section(name: string): void {
    console.log(`\n── ${name} ──`);
  }

  // ── generateSSAOKernel ────────────────────────────────────────────────
  section("generateSSAOKernel");

  const k16 = generateSSAOKernel(16);
  eq("kernel16 length", k16.length, 16 * 4);

  const k32 = generateSSAOKernel(32);
  eq("kernel32 length", k32.length, 32 * 4);

  const k64 = generateSSAOKernel(64);
  eq("kernel64 length", k64.length, 64 * 4);

  // 所有样本 z ≥ 0 (半球: 法线方向)
  let allHemisphere = true;
  for (let i = 0; i < 32; i++) {
    if (k32[i * 4 + 2] < 0) {
      allHemisphere = false;
      break;
    }
  }
  ok("all kernel samples in hemisphere (z >= 0)", allHemisphere);

  // 样本长度递增趋势 (近核密集, 远核稀疏)
  let firstLen = 0;
  let lastLen = 0;
  {
    const x0 = k32[0], y0 = k32[1], z0 = k32[2];
    firstLen = Math.sqrt(x0 * x0 + y0 * y0 + z0 * z0);
    const xN = k32[31 * 4], yN = k32[31 * 4 + 1], zN = k32[31 * 4 + 2];
    lastLen = Math.sqrt(xN * xN + yN * yN + zN * zN);
  }
  ok("last kernel sample longer than first", lastLen > firstLen);

  // ── generateNoiseTexture ──────────────────────────────────────────────
  section("generateNoiseTexture");

  const noise = generateNoiseTexture();
  eq("noise width", noise.width, 4);
  eq("noise height", noise.height, 4);
  eq("noise data length", noise.data.length, 4 * 4 * 4);
  // Alpha = 255 for all pixels
  let allAlpha255 = true;
  for (let i = 0; i < 16; i++) {
    if (noise.data[i * 4 + 3] !== 255) {
      allAlpha255 = false;
      break;
    }
  }
  ok("noise alpha all 255", allAlpha255);

  // ── vanDerCorput ──────────────────────────────────────────────────────
  section("vanDerCorput (Hammersley)");

  const v0 = vanDerCorput(0);
  eq("vdc(0) = 0", v0, 0, 1e-9);

  // vdc(1) should be 0.5 (bit-reverse of 1 in base 2)
  const v1 = vanDerCorput(1);
  eq("vdc(1) ≈ 0.5", v1, 0.5, 1e-4);

  // vdc(2) should be 0.25
  const v2 = vanDerCorput(2);
  eq("vdc(2) ≈ 0.25", v2, 0.25, 1e-4);

  // All values in [0, 1)
  let allIn01 = true;
  for (let i = 0; i < 100; i++) {
    const v = vanDerCorput(i);
    if (v < 0 || v >= 1) { allIn01 = false; break; }
  }
  ok("vdc values all in [0, 1)", allIn01);

  // ── packSSAOUniforms ──────────────────────────────────────────────────
  section("packSSAOUniforms");

  const buf = packSSAOUniforms(DEFAULT_SSAO_PARAMS, 1920, 1080);
  eq("uniform buf length", buf.length, 12);
  eq("nearPlane", buf[0], 0.1);
  eq("farPlane", buf[1], 100.0);
  eq("invProjX", buf[2], 1.0);
  eq("invProjY", buf[3], 1.0);
  eq("kernelRadius", buf[4], 0.5);
  eq("bias", buf[5], 0.025);
  eq("intensity", buf[6], 1.5);
  // kernelSize as u32 bitcast
  const kernelSizeU32 = new Uint32Array(buf.buffer, 7 * 4, 1)[0];
  eq("kernelSize (u32)", kernelSizeU32, 32);
  eq("resolution.x", buf[8], 1920);
  eq("resolution.y", buf[9], 1080);
  eq("noiseScale.x", buf[10], 1920 / 4);
  eq("noiseScale.y", buf[11], 1080 / 4);

  // ── packBlurUniforms ──────────────────────────────────────────────────
  section("packBlurUniforms");

  const blurH = packBlurUniforms(DEFAULT_BLUR_PARAMS, [1, 0], 0.1, 100.0, 1920, 1080);
  eq("blur-h length", blurH.length, 8);
  eq("direction.x", blurH[0], 1);
  eq("direction.y", blurH[1], 0);
  eq("depthThreshold", blurH[2], 0.1);
  eq("normalThreshold", blurH[3], 0.9);
  eq("texelSize.x", blurH[6], 1.0 / 1920, 1e-8);

  const blurV = packBlurUniforms(DEFAULT_BLUR_PARAMS, [0, 1], 0.1, 100.0, 1920, 1080);
  eq("blur-v direction.x", blurV[0], 0);
  eq("blur-v direction.y", blurV[1], 1);

  // ── packCompositeUniforms ─────────────────────────────────────────────
  section("packCompositeUniforms");

  const compBuf = packCompositeUniforms(DEFAULT_COMPOSITE_PARAMS);
  eq("composite buf length", compBuf.length, 4);
  eq("aoStrength", compBuf[0], 1.0);
  eq("aoGamma", compBuf[1], 1.0);

  // ── WGSL 完整性检查 ─────────────────────────────────────────────────────
  section("WGSL completeness");

  ok("SSAO sample shader contains fs_ssao", WGSL_SSAO_SAMPLE.includes("fn fs_ssao"));
  ok("SSAO sample shader contains vs_fullscreen", WGSL_SSAO_SAMPLE.includes("fn vs_fullscreen"));
  ok("blur shader contains fs_blur", WGSL_BILATERAL_BLUR.includes("fn fs_blur"));
  ok("composite shader contains fs_composite", WGSL_SSAO_COMPOSITE.includes("fn fs_composite"));
  ok("common contains linearizeDepth", WGSL_SSAO_COMMON.includes("fn linearizeDepth"));
  ok("common contains reconstructViewPos", WGSL_SSAO_COMMON.includes("fn reconstructViewPos"));
  ok("SSAO_WGSL export has all keys", !!SSAO_WGSL.common && !!SSAO_WGSL.ssaoSample && !!SSAO_WGSL.bilateralBlur && !!SSAO_WGSL.composite);

  // ── 默认参数完整性 ─────────────────────────────────────────────────────
  section("default params");

  ok("DEFAULT_SSAO_PARAMS.kernelRadius > 0", DEFAULT_SSAO_PARAMS.kernelRadius > 0);
  ok("DEFAULT_SSAO_PARAMS.kernelSize ∈ {16,32,64}", [16, 32, 64].includes(DEFAULT_SSAO_PARAMS.kernelSize));
  ok("DEFAULT_BLUR_PARAMS.depthThreshold > 0", DEFAULT_BLUR_PARAMS.depthThreshold > 0);
  ok("DEFAULT_BLUR_PARAMS.normalThreshold ∈ (0,1]", DEFAULT_BLUR_PARAMS.normalThreshold > 0 && DEFAULT_BLUR_PARAMS.normalThreshold <= 1);
  ok("DEFAULT_COMPOSITE_PARAMS.aoStrength ∈ [0,1]", DEFAULT_COMPOSITE_PARAMS.aoStrength >= 0 && DEFAULT_COMPOSITE_PARAMS.aoStrength <= 1);

  // ── Summary ───────────────────────────────────────────────────────────
  console.log(`\n════════════════════════════════`);
  console.log(`  Tests: ${passed + failed}   Passed: ${passed}   Failed: ${failed}`);
  console.log(`════════════════════════════════\n`);
  if (failed > 0) process.exit(1);
}
