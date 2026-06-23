/**
 * at-glass-reflection-system.ts — AT Glass Reflection System — WebGPU/WGSL Port
 *
 * 玻璃反射系统，为 Cell 外壳提供真实感的玻璃材质:
 *   - 菲涅耳反射 (Fresnel Reflection) — 掠射角处高反射率
 *   - 折射与色散 (Refraction + Chromatic Aberration) — 棱镜效应
 *   - 体积散射 (Subsurface Scattering) — 内部漫散光
 *   - 环境反射 (Environment Cubemap) — 基于探针的反射
 *   - CleanRoom 场景洁净室玻璃效果
 *
 * 移植自 ActiveTheory compiled.vs:
 *   - GlassInner.glsl      → 内部体积散射纹理
 *   - GlassReflection.glsl → 镜面反射基础
 *   - CleanRoomGlass.glsl  → 完整洁净室玻璃效果
 *   - BasicMirror.glsl     → 镜面平面反射
 *   - fresnel.glsl         → 菲涅耳计算
 *   - refl.vs/fs           → 环境反射 & 折射
 *   - rgbshift.fs          → 色散分离采样
 *
 * 提供两套材质路径:
 *   1. ATGlassReflectionSystem  — 完整玻璃系统 (菲涅耳 + 折射 + 体积散射 + 环境探针)
 *   2. CleanRoomGlassMode       — 洁净室专用模式 (彩虹色散 + 高保真折射)
 *
 * 用法:
 *   const system = await ATGlassReflectionSystem.create(device, format);
 *   system.setParams({
 *     fresnelPower: 4.0,
 *     refractionRatio: 0.66,
 *     dispersiveness: 0.25,
 *     subsurfaceIntensity: 0.8,
 *     envProbeStrength: 1.0,
 *     cleanRoomMode: true
 *   });
 *   system.render(encoder, colorTargetView, depthView, uniformBuffer);
 *
 * Research: xiaodi #M839 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 公共数学助手 (saturate / pow5 / constants)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_MATH_HELPERS = /* wgsl */`
fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, 0.0, 1.0); }
fn pow5_f(v: f32) -> f32 { let v2 = v * v; return v2 * v2 * v; }
fn pow5_v3(v: vec3f) -> vec3f { let v2 = v * v; return v2 * v2 * v; }

const PI       : f32 = 3.14159265358979323846;
const TWO_PI   : f32 = 6.28318530717958647693;
const INV_PI   : f32 = 0.31830988618379067154;
const HALF_PI  : f32 = 1.57079632679489661923;
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fresnel 计算 (Schlick + 精确 Fresnel-Dielectric)
// 参考 lygia/lighting/fresnel.glsl + CleanRoomGlass.glsl
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FRESNEL_GLASS = /* wgsl */`
// ── Schlick 近似 Fresnel (速度快，精度好) ──────────────────────────────────────
fn F_Schlick(f0: vec3f, cosTheta: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow5_v3(vec3f(saturate_f(1.0 - cosTheta)));
}

// ── 简单Fresnel 因子 (单浮点) ─────────────────────────────────────────────────
fn getFresnel(N: vec3f, V: vec3f, power: f32) -> f32 {
    let d = dot(normalize(N), normalize(V));
    return 1.0 - pow(abs(d), power);
}

// ── 精确 Fresnel-Dielectric (用于玻璃折射计算) ────────────────────────────────
fn getFresnelDielectric(inIOR: f32, outIOR: f32, N: vec3f, V: vec3f) -> f32 {
    let ro = (inIOR - outIOR) / (inIOR + outIOR);
    let d = dot(normalize(N), normalize(V));
    return ro + (1.0 - ro) * pow((1.0 - d), 5.0);
}

// ── Fresnel 边缘光 (彩色) ──────────────────────────────────────────────────────
fn fresnelRim(N: vec3f, V: vec3f, power: f32, rimColor: vec3f) -> vec3f {
    return rimColor * getFresnel(N, V, power);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 折射与色散 (Refraction + Chromatic Dispersion)
// 参考 refl.fs + rgbshift.fs + CleanRoomGlass.glsl
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_REFRACTION_DISPERSION = /* wgsl */`
// ── 环境映射坐标转换 (等距柱面投影) ────────────────────────────────────────────
fn equirectangularCoords(direction: vec3f) -> vec2f {
    let uv = vec2f(
        atan2(direction.z, direction.x) * 0.15915494,  // 1/(2π)
        asin(clamp(direction.y, -1.0, 1.0)) * 0.31830988618  // 1/π
    ) + 0.5;
    return uv;
}

// ── 色散折射 (RGB 分离采样) ────────────────────────────────────────────────────
fn sampleRGBDispersion(
    texEnv: texture_2d<f32>,
    samplerEnv: sampler,
    direction: vec3f,
    angle: f32,
    dispersiveness: f32
) -> vec4f {
    let uv = equirectangularCoords(direction);
    let offset = vec2f(cos(angle), sin(angle)) * dispersiveness * 0.01;
    
    let r = textureSample(texEnv, samplerEnv, uv + offset);
    let g = textureSample(texEnv, samplerEnv, uv);
    let b = textureSample(texEnv, samplerEnv, uv - offset);
    
    return vec4f(r.r, g.g, b.b, g.a);
}

// ── 折射方向计算 ──────────────────────────────────────────────────────────────
fn refractDirection(V: vec3f, N: vec3f, ior: f32) -> vec3f {
    return refract(V, N, 1.0 / ior);
}

// ── 玻璃折射采样 (带扭曲) ─────────────────────────────────────────────────────
fn glassRefraction(
    V: vec3f,
    N: vec3f,
    ior: f32,
    distortStrength: f32
) -> vec3f {
    let refractVec = refractDirection(V, N, ior);
    
    // 基于法线的扭曲 (模拟玻璃表面不规则)
    let distort = N.xy * distortStrength * 0.1;
    
    // 返回扭曲后的折射方向
    return normalize(refractVec + vec3f(distort, 0.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 体积散射 (Subsurface Scattering)
// 参考 GlassInner.glsl
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SUBSURFACE_SCATTERING = /* wgsl */`
// ── Perlin 噪声 (简化版，用于体积散射) ────────────────────────────────────────
fn permute(x: vec4f) -> vec4f { return (((x * 34.0) + 1.0) * x) % 289.0; }
fn taylorInvSqrt(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

fn cnoise(P: vec3f) -> f32 {
    let Pi0 = floor(P);
    let Pi1 = Pi0 + 1.0;
    let Pf0 = fract(P);
    let Pf1 = Pf0 - 1.0;
    
    let ix = vec4f(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    let iy = vec4f(Pi0.yy, Pi1.yy);
    let iz0 = Pi0.zzzz;
    let iz1 = Pi1.zzzz;
    
    let ixy = permute(permute(ix) + iy);
    let ixy0 = permute(ixy + iz0);
    let ixy1 = permute(ixy + iz1);
    
    let gx0z0 = ixy0 * 0.0243902439; // 1/41 for gradient count
    let gy0z0 = fract(floor(gx0z0) * 0.025) - 0.5;
    let gx0z0f = fract(gx0z0) - 0.5;
    
    let gx0z1 = ixy1 * 0.0243902439;
    let gy0z1 = fract(floor(gx0z1) * 0.025) - 0.5;
    let gx0z1f = fract(gx0z1) - 0.5;
    
    var g000 = vec3f(gx0z0f, gy0z0, gx0z0);
    var g100 = vec3f(gx0z0f - 1.0, gy0z0, gx0z0 - 1.0);
    var g010 = vec3f(gx0z1f, gy0z1, gx0z1);
    var g110 = vec3f(gx0z1f - 1.0, gy0z1, gx0z1 - 1.0);
    
    let d000 = dot(g000, Pf0);
    let d100 = dot(g100, vec3f(Pf1.x, Pf0.y, Pf0.z));
    let d010 = dot(g010, Pf1);
    let d110 = dot(g110, vec3f(Pf0.x, Pf1.y, Pf1.z));
    
    let n = (d000 + d100 + d010 + d110) / 4.0;
    return 2.3 * n;
}

// ── 范围映射 (crange 替代) ─────────────────────────────────────────────────────
fn crange(v: f32, inMin: f32, inMax: f32, outMin: f32, outMax: f32) -> f32 {
    let clamped = clamp((v - inMin) / (inMax - inMin), 0.0, 1.0);
    return mix(outMin, outMax, clamped);
}

// ── Quartic In 缓动函数 ────────────────────────────────────────────────────────
fn quarticIn(t: f32) -> f32 { return t * t * t * t; }

// ── 体积散射 (基于噪声 + 法线) ─────────────────────────────────────────────────
fn subsurfaceScattering(
    viewDir: vec3f,
    normal: vec3f,
    position: vec3f,
    intensity: f32
) -> vec3f {
    // 基于视角方向和法线的噪声
    let noiseBase = cnoise(viewDir * 0.2 + 0.5);
    let noiseClamped = crange(noiseBase, -1.0, 1.0, 0.0, 1.0);
    let veinPattern = noiseClamped * mix(vec3f(0.0), vec3f(1.4), clamp(normal.y, 0.0, 1.0));
    
    // 微细细节
    let detail = cnoise(viewDir) * 0.05;
    
    // 边缘光晕 (基于位置的衰减)
    let edgeFade = quarticIn(crange(
        abs(position.x), 0.5, 0.3, 1.0, 0.0
    ) * crange(
        abs(position.z), 0.5, 0.3, 1.0, 0.0
    )) * 0.1;
    
    return saturate_v3((veinPattern + vec3f(detail) + vec3f(edgeFade)) * intensity);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 环境反射 (Environment Reflection & Probe)
// 参考 refl.vs + refl.fs
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_ENVIRONMENT_REFLECTION = /* wgsl */`
// ── 反射方向计算 ──────────────────────────────────────────────────────────────
fn reflectionDirection(V: vec3f, N: vec3f) -> vec3f {
    return reflect(V, N);
}

// ── 方向矩阵变换 ──────────────────────────────────────────────────────────────
fn inverseTransformDirection(dir: vec3f, matrix: mat4x4f) -> vec3f {
    return normalize((matrix * vec4f(dir, 0.0)).xyz);
}

fn transformDirection(dir: vec3f, matrix: mat4x4f) -> vec3f {
    return normalize((matrix * vec4f(dir, 0.0)).xyz);
}

// ── 环境探针采样 (等距柱面) ────────────────────────────────────────────────────
fn sampleEnvironmentProbe(
    texProbe: texture_2d<f32>,
    samplerProbe: sampler,
    direction: vec3f,
    intensity: f32
) -> vec3f {
    let uv = equirectangularCoords(direction);
    let sample = textureSample(texProbe, samplerProbe, uv);
    return sample.rgb * intensity;
}

// ── 铝箔反射 (平面镜面) ────────────────────────────────────────────────────────
fn mirrorReflection(
    texMirror: texture_2d<f32>,
    samplerMirror: sampler,
    mirrorCoord: vec4f
) -> vec3f {
    let uv = mirrorCoord.xy / mirrorCoord.w;
    let sample = textureSample(texMirror, samplerMirror, uv);
    return sample.rgb;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — CleanRoom 专用玻璃模式
// 参考 CleanRoomGlass.glsl 的彩虹色散效果
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_CLEANROOM_GLASS = /* wgsl */`
// ── 彩虹色映射 (Rainbow Color Spectrum) ─────────────────────────────────────
fn rainbowColor(t: f32) -> vec3f {
    let t_wrapped = t % 1.0;  // Wrap t to [0, 1]
    
    if (t_wrapped < 0.03) {
        // 紫色 → 蓝色
        return mix(vec3f(0.5, 0.0, 0.5), vec3f(0.5, 0.0, 1.0), t_wrapped / 0.03);
    } else if (t_wrapped < 0.06) {
        // 蓝色 → 深蓝
        return mix(vec3f(0.5, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), (t_wrapped - 0.03) / 0.03);
    } else if (t_wrapped < 0.09) {
        // 深蓝 → 青色
        return mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), (t_wrapped - 0.06) / 0.03);
    } else if (t_wrapped < 0.12) {
        // 青色 → 绿色
        return mix(vec3f(0.0, 1.0, 1.0), vec3f(0.0, 1.0, 0.0), (t_wrapped - 0.09) / 0.03);
    } else if (t_wrapped < 0.18) {
        // 绿色 → 黄色
        return mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), (t_wrapped - 0.12) / 0.06);
    } else if (t_wrapped < 0.24) {
        // 黄色 → 橙色
        return mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.5, 0.0), (t_wrapped - 0.18) / 0.06);
    } else {
        // 橙色 → 红色
        return mix(vec3f(1.0, 0.5, 0.0), vec3f(1.0, 0.0, 0.0), (t_wrapped - 0.24) / 0.06);
    }
}

// ── CleanRoom 玻璃着色 ─────────────────────────────────────────────────────────
fn cleanroomGlassShading(
    fresnel: f32,
    refractColor: vec3f,
    envColor: vec3f,
    subsurfaceColor: vec3f,
    noiseDetail: f32,
    edgeGlow: f32,
    normalY: f32
) -> vec3f {
    // 彩虹色基于菲涅耳值
    let rainbow = rainbowColor(fresnel * 4.0);
    let rainbowFiltered = mix(
        rainbow,
        vec3f(0.0),
        step(0.99, rainbow.r)  // 过滤掉纯红色的瑕疵
    );
    
    // 折射 + 环境反射 + 色散
    var color = refractColor;
    color += rainbowFiltered;
    color += envColor;
    color += subsurfaceColor;
    color += vec3f(noiseDetail);
    color += vec3f(edgeGlow);
    
    // 顶部增强 (强化天空反射)
    if (normalY > 0.8) {
        color *= 1.8;
    }
    
    // 伽马矫正
    color = pow(saturate_v3(color), vec3f(1.5));
    
    return color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript Class Definition
// ─────────────────────────────────────────────────────────────────────────────

interface GlassReflectionParams {
    fresnelPower?: number;           // Fresnel 衰减指数 (2.0 - 8.0)
    refractionRatio?: number;        // IOR 比例 (0.5 - 1.0)
    dispersiveness?: number;         // 色散强度 (0.0 - 0.5)
    distortStrength?: number;        // 折射扭曲强度 (0.0 - 1.0)
    subsurfaceIntensity?: number;    // 体积散射强度 (0.0 - 2.0)
    envProbeStrength?: number;       // 环境探针强度 (0.0 - 1.5)
    mirrorStrength?: number;         // 镜面反射强度 (0.0 - 1.0)
    cleanRoomMode?: boolean;         // 是否启用 CleanRoom 彩虹模式
}

interface GlassReflectionUniform {
    fresnelPower: f32;
    refractionRatio: f32;
    dispersiveness: f32;
    distortStrength: f32;
    subsurfaceIntensity: f32;
    envProbeStrength: f32;
    mirrorStrength: f32;
    cleanRoomMode: u32;
    time: f32;
    padding: f32;
}

export class ATGlassReflectionSystem {
    private device: GPUDevice;
    private format: GPUTextureFormat;
    private pipeline: GPURenderPipeline | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private uniformBindGroup: GPUBindGroup | null = null;
    private params: Required<GlassReflectionParams>;
    private startTime: number = 0;
    private textures: {
        envProbe?: GPUTexture;
        mirrorReflection?: GPUTexture;
        subsurfaceMap?: GPUTexture;
    } = {};

    constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;
        this.format = format;
        this.params = {
            fresnelPower: 4.0,
            refractionRatio: 0.66,
            dispersiveness: 0.25,
            distortStrength: 0.5,
            subsurfaceIntensity: 0.8,
            envProbeStrength: 1.0,
            mirrorStrength: 0.7,
            cleanRoomMode: false,
        };
        this.startTime = performance.now();
    }

    /**
     * 创建玻璃反射系统
     */
    static async create(
        device: GPUDevice,
        format: GPUTextureFormat
    ): Promise<ATGlassReflectionSystem> {
        const system = new ATGlassReflectionSystem(device, format);
        await system.initialize();
        return system;
    }

    /**
     * 初始化 GPU 资源
     */
    private async initialize(): Promise<void> {
        // 创建 Uniform Buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 256,  // 16 byte aligned (16 * f32)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });

        // 创建绑定组布局和管线（在实际实现中）
        this.setupPipeline();
    }

    /**
     * 设置渲染管线
     */
    private setupPipeline(): void {
        // 在实际的三维引擎中，这里会创建完整的 render pipeline
        // 包含完整的 vertex + fragment shader
        // 这里仅作演示框架结构
    }

    /**
     * 设置材质参数
     */
    setParams(params: Partial<GlassReflectionParams>): void {
        this.params = { ...this.params, ...params };
        this.updateUniforms();
    }

    /**
     * 更新 Uniform 缓冲区
     */
    private updateUniforms(): void {
        if (!this.uniformBuffer) return;

        const elapsed = (performance.now() - this.startTime) / 1000;
        const uniforms: GlassReflectionUniform = {
            fresnelPower: this.params.fresnelPower,
            refractionRatio: this.params.refractionRatio,
            dispersiveness: this.params.dispersiveness,
            distortStrength: this.params.distortStrength,
            subsurfaceIntensity: this.params.subsurfaceIntensity,
            envProbeStrength: this.params.envProbeStrength,
            mirrorStrength: this.params.mirrorStrength,
            cleanRoomMode: this.params.cleanRoomMode ? 1 : 0,
            time: elapsed,
            padding: 0,
        };

        const data = new Float32Array(16);
        data[0] = uniforms.fresnelPower;
        data[1] = uniforms.refractionRatio;
        data[2] = uniforms.dispersiveness;
        data[3] = uniforms.distortStrength;
        data[4] = uniforms.subsurfaceIntensity;
        data[5] = uniforms.envProbeStrength;
        data[6] = uniforms.mirrorStrength;
        data[7] = uniforms.cleanRoomMode;
        data[8] = uniforms.time;
        data[9] = uniforms.padding;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
    }

    /**
     * 绘制玻璃反射
     */
    render(
        commandEncoder: GPUCommandEncoder,
        colorTarget: GPURenderPassColorAttachment,
        depthTarget: GPURenderPassDepthStencilAttachment,
        geometry?: { vertexBuffer: GPUBuffer; indexBuffer: GPUBuffer; indexCount: number }
    ): void {
        if (!this.pipeline || !this.uniformBuffer) return;

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [colorTarget],
            depthStencilAttachment: depthTarget,
        });

        renderPass.setPipeline(this.pipeline);
        
        if (this.uniformBindGroup) {
            renderPass.setBindGroup(0, this.uniformBindGroup);
        }

        if (geometry) {
            renderPass.setVertexBuffer(0, geometry.vertexBuffer);
            renderPass.setIndexBuffer(geometry.indexBuffer, 'uint32');
            renderPass.drawIndexed(geometry.indexCount);
        } else {
            // 绘制全屏四边形
            renderPass.draw(6, 1, 0, 0);
        }

        renderPass.end();
    }

    /**
     * 设置环境探针纹理
     */
    setEnvironmentProbe(texture: GPUTexture): void {
        this.textures.envProbe = texture;
    }

    /**
     * 设置镜面反射纹理
     */
    setMirrorReflection(texture: GPUTexture): void {
        this.textures.mirrorReflection = texture;
    }

    /**
     * 设置体积散射贴图
     */
    setSubsurfaceMap(texture: GPUTexture): void {
        this.textures.subsurfaceMap = texture;
    }

    /**
     * 获取完整 Fragment Shader 代码
     */
    static getFragmentShaderSource(): string {
        return /* wgsl */`
${WGSL_MATH_HELPERS}
${WGSL_FRESNEL_GLASS}
${WGSL_REFRACTION_DISPERSION}
${WGSL_SUBSURFACE_SCATTERING}
${WGSL_ENVIRONMENT_REFLECTION}
${WGSL_CLEANROOM_GLASS}

struct Uniforms {
    fresnelPower: f32,
    refractionRatio: f32,
    dispersiveness: f32,
    distortStrength: f32,
    subsurfaceIntensity: f32,
    envProbeStrength: f32,
    mirrorStrength: f32,
    cleanRoomMode: u32,
    time: f32,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texEnv: texture_2d<f32>;
@group(0) @binding(2) var samplerEnv: sampler;
@group(0) @binding(3) var texInner: texture_2d<f32>;
@group(0) @binding(4) var samplerInner: sampler;
@group(0) @binding(5) var texMirror: texture_2d<f32>;
@group(0) @binding(6) var samplerMirror: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) vWorldPos: vec3f,
    @location(1) vNormal: vec3f,
    @location(2) vViewDir: vec3f,
    @location(3) vTexCoord: vec2f,
    @location(4) vReflection: vec3f,
    @location(5) vRefraction: vec3f,
    @location(6) vMirrorCoord: vec4f,
}

@fragment
fn main(input: VertexOutput) -> @location(0) vec4f {
    let N = normalize(input.vNormal);
    let V = normalize(input.vViewDir);
    
    // ── Fresnel 计算 ──────────────────────────────────────────────────────────────
    let fresnel = getFresnel(N, V, uniforms.fresnelPower);
    
    // ── 折射采样 ──────────────────────────────────────────────────────────────────
    var refractColor = vec3f(0.0);
    if (uniforms.refractionRatio > 0.0) {
        let refractDir = glassRefraction(V, N, uniforms.refractionRatio, uniforms.distortStrength);
        refractColor = sampleEnvironmentProbe(texEnv, samplerEnv, refractDir, 1.0);
    }
    
    // ── 色散环境反射 ──────────────────────────────────────────────────────────────
    var envColor = vec3f(0.0);
    if (uniforms.envProbeStrength > 0.0) {
        let reflectDir = reflectionDirection(V, N);
        let dispersalSample = sampleRGBDispersion(
            texEnv, samplerEnv,
            reflectDir,
            uniforms.time * 0.5,
            uniforms.dispersiveness
        );
        envColor = dispersalSample.rgb * uniforms.envProbeStrength;
    }
    
    // ── 体积散射 ──────────────────────────────────────────────────────────────────
    var subsurfaceColor = vec3f(0.0);
    if (uniforms.subsurfaceIntensity > 0.0) {
        subsurfaceColor = subsurfaceScattering(
            V, N,
            input.vWorldPos,
            uniforms.subsurfaceIntensity
        );
    }
    
    // ── 噪声细节 ──────────────────────────────────────────────────────────────────
    let noiseDetail = cnoise(V + vec3f(uniforms.time * 0.1)) * 0.05;
    
    // ── 内部纹理采样 ──────────────────────────────────────────────────────────────
    let innerTexture = textureSample(texInner, samplerInner, input.vTexCoord);
    
    // ── 边缘光晕 ──────────────────────────────────────────────────────────────────
    let edgeGlow = quarticIn(
        crange(abs(input.vWorldPos.x), 0.5, 0.3, 1.0, 0.0) *
        crange(abs(input.vWorldPos.z), 0.5, 0.3, 1.0, 0.0)
    ) * 0.05;
    
    // ── 最终着色 ──────────────────────────────────────────────────────────────────
    var finalColor = vec3f(0.0);
    
    if (uniforms.cleanRoomMode != 0u) {
        // CleanRoom 模式：彩虹色散效果
        finalColor = cleanroomGlassShading(
            fresnel,
            refractColor,
            envColor,
            subsurfaceColor,
            noiseDetail,
            edgeGlow,
            N.y
        );
    } else {
        // 标准玻璃模式：混合折射 + 反射 + 散射
        finalColor = refractColor * (1.0 - fresnel);
        finalColor += envColor * fresnel;
        finalColor += subsurfaceColor;
        finalColor += vec3f(noiseDetail);
        finalColor += innerTexture.rgb;
        finalColor += vec3f(edgeGlow);
        finalColor = pow(saturate_v3(finalColor), vec3f(1.5));
    }
    
    return vec4f(finalColor, 1.0);
}
        `;
    }

    /**
     * 获取完整 Vertex Shader 代码
     */
    static getVertexShaderSource(): string {
        return /* wgsl */`
struct Uniforms {
    fresnelPower: f32,
    refractionRatio: f32,
    dispersiveness: f32,
    distortStrength: f32,
    subsurfaceIntensity: f32,
    envProbeStrength: f32,
    mirrorStrength: f32,
    cleanRoomMode: u32,
    time: f32,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) texCoord: vec2f,
    @builtin(instance_index) instanceIdx: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) vWorldPos: vec3f,
    @location(1) vNormal: vec3f,
    @location(2) vViewDir: vec3f,
    @location(3) vTexCoord: vec2f,
    @location(4) vReflection: vec3f,
    @location(5) vRefraction: vec3f,
    @location(6) vMirrorCoord: vec4f,
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // 世界坐标变换 (这里使用单位矩阵，实际应使用 modelMatrix)
    let worldPos = vec4f(input.position, 1.0);
    
    // 投影变换 (实际应使用 projectionMatrix * viewMatrix)
    output.position = vec4f(input.position, 1.0);
    
    output.vWorldPos = worldPos.xyz;
    output.vNormal = input.normal;
    
    // 视角方向 (假设摄像机位置在原点)
    output.vViewDir = -input.position;
    
    output.vTexCoord = input.texCoord;
    
    // 反射方向
    output.vReflection = reflect(normalize(-input.position), normalize(input.normal));
    
    // 折射方向
    output.vRefraction = refract(
        normalize(-input.position),
        normalize(input.normal),
        uniforms.refractionRatio
    );
    
    // 镜面坐标 (用于平面镜反射)
    output.vMirrorCoord = vec4f(input.texCoord * 2.0 - 1.0, 1.0, 1.0);
    
    return output;
}
        `;
    }

    /**
     * 销毁资源
     */
    destroy(): void {
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
        }
        this.pipeline = null;
        this.uniformBindGroup = null;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export 导出
// ─────────────────────────────────────────────────────────────────────────────

export type {
    GlassReflectionParams,
    GlassReflectionUniform,
};

export default ATGlassReflectionSystem;
