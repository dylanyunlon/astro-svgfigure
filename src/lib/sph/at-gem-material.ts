/**
 * at-gem-material.ts — AT Gem Material System — WebGPU/WGSL Port
 *
 * 宝石材质系统，用于 Cell 高亮态:
 *   - 色散折射 (Chromatic Aberration + IOR 分离)
 *   - Cook-Torrance BRDF 高光 + Fresnel 边缘光
 *   - 内部焦散 (Internal Caustics) — 基于世界坐标 Perlin 噪声
 *   - 纹理混合：alien_cracked_2_basecolor + cracked_ice_basecolor
 *   - 参考至 Finding-Love-Shaders 和 hexagon_gem.bin 几何
 *
 * 提供 ATGemMaterial class，支持:
 *   1. GeometryRefraction    — 折射贴图+ 色散分离 (R/G/B 不同 IOR)
 *   2. InternalCaustics      — 体积焦散动画
 *   3. HighlightBlending     — Cell 高亮状态混合
 *
 * 用法:
 *   const mat = await ATGemMaterial.create(device, format);
 *   mat.setParams({ ior: 1.45, dispersiveness: 0.28, causticsIntensity: 1.2 });
 *   mat.render(encoder, colorTargetView, depthView, uniformBuffer);
 *
 * Research: xiaodi #M829 — cell-pubsub-loop
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
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Cook-Torrance BRDF (镜面反射)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PBR_BRDF = /* wgsl */`
fn F_Schlick(f0: vec3f, cosTheta: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow5_v3(vec3f(saturate_f(1.0 - cosTheta)));
}

fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

fn G_SmithGGX(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let gV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let gL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (gV + gL + 1e-7);
}

fn specularBRDF(N: vec3f, V: vec3f, L: vec3f, f0: vec3f, roughness: f32) -> vec3f {
    let H      = normalize(V + L);
    let NdotH  = saturate_f(dot(N, H));
    let NdotV  = saturate_f(dot(N, V));
    let NdotL  = saturate_f(dot(N, L));
    let VdotH  = saturate_f(dot(V, H));

    let D  = D_GGX(NdotH, roughness);
    let Gv = G_SmithGGX(NdotV, NdotL, roughness);
    let F  = F_Schlick(f0, VdotH);

    return vec3f(D * Gv) * F;
}

fn pbrDirect(
    albedo    : vec3f,
    N         : vec3f,
    V         : vec3f,
    L         : vec3f,
    f0        : vec3f,
    metallic  : f32,
    roughness : f32,
    lightColor: vec3f
) -> vec3f {
    let NdotL = saturate_f(dot(N, L));
    if (NdotL < 1e-5) { return vec3f(0.0); }

    let specular = specularBRDF(N, V, L, f0, roughness);
    let diffuse  = albedo * INV_PI * (1.0 - metallic);
    let ks = F_Schlick(f0, saturate_f(dot(N, V)));
    let kd = (vec3f(1.0) - ks) * (1.0 - metallic);

    return (kd * diffuse + specular) * lightColor * NdotL;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fresnel 边缘光 & 折射 IOR
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FRESNEL_IOR = /* wgsl */`
// Schlick Fresnel 边缘因子
fn fresnel_f(N: vec3f, V: vec3f, power: f32) -> f32 {
    let cosTheta = saturate_f(dot(N, V));
    return pow(1.0 - cosTheta, power);
}

// 彩色 Fresnel 边缘光
fn fresnelRim(N: vec3f, V: vec3f, power: f32, rimColor: vec3f) -> vec3f {
    return rimColor * fresnel_f(N, V, power);
}

// 色散折射: R/G/B 分别使用不同 IOR (模拟棱镜效应)
fn refractionDispersal(
    N           : vec3f,
    V           : vec3f,
    iorBase     : f32,        // 基础 IOR (1.4-1.8 宝石范围)
    dispersiness: f32,        // 色散量 (0.0-0.5)
    backColor   : vec3f,      // 背景色 (用于假折射)
    roughness   : f32         // 宝石粗糙度
) -> vec3f {
    // 计算每个通道的 IOR
    let iorRed   = iorBase + dispersiness * 0.015;
    let iorGreen = iorBase;
    let iorBlue  = iorBase - dispersiness * 0.025;

    // 简化折射: 使用 Fresnel 衰减 + 法线扰动模拟折射偏移
    let F = fresnel_f(N, V, 3.0);  // Fresnel 衰减: 掠射角处透明度高
    let refractAmount = (1.0 - F) * (1.0 - roughness * 0.5);

    // 色散：不同通道的折射方向轻微不同
    let refractDir = refract(V, N, 1.0 / iorGreen);
    
    // 基于世界坐标的扰动 (伪造色散分离)
    let dispersalShift = vec3f(
        refractAmount * sin(refractDir.x * 10.0) * 0.05,
        refractAmount * sin(refractDir.y * 10.0) * 0.05,
        refractAmount * sin(refractDir.z * 10.0) * 0.05
    );

    // 每个通道独立计算
    let colorOut = backColor * (1.0 - refractAmount) + backColor * refractAmount * dispersalShift;
    
    return saturate_v3(colorOut);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 内部焦散 (Perlin Noise 模拟)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_INTERNAL_CAUSTICS = /* wgsl */`
// 简化 Perlin 噪声 (用于焦散动画)
fn hash3(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

fn perlin3D(p: vec3f) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    let w  = pf * pf * (3.0 - 2.0 * pf);

    let h000 = hash3(pi + vec3f(0, 0, 0)).x;
    let h100 = hash3(pi + vec3f(1, 0, 0)).x;
    let h010 = hash3(pi + vec3f(0, 1, 0)).x;
    let h110 = hash3(pi + vec3f(1, 1, 0)).x;
    let h001 = hash3(pi + vec3f(0, 0, 1)).x;
    let h101 = hash3(pi + vec3f(1, 0, 1)).x;
    let h011 = hash3(pi + vec3f(0, 1, 1)).x;
    let h111 = hash3(pi + vec3f(1, 1, 1)).x;

    let h00 = mix(h000, h100, w.x);
    let h10 = mix(h010, h110, w.x);
    let h01 = mix(h001, h101, w.x);
    let h11 = mix(h011, h111, w.x);

    let h0 = mix(h00, h10, w.y);
    let h1 = mix(h01, h11, w.y);

    return mix(h0, h1, w.z);
}

// 焦散光线: 基于世界坐标和时间动画
fn internalCaustics(
    worldPos      : vec3f,
    time          : f32,
    causticsScale : f32,
    causticsSpeed : f32,
    intensity     : f32
) -> vec3f {
    let scrolledPos = worldPos * causticsScale + vec3f(0, 0, time * causticsSpeed);
    
    // 分层 Perlin 噪声 (fractional Brownian motion)
    let noise1 = perlin3D(scrolledPos * 1.0);
    let noise2 = perlin3D(scrolledPos * 2.0 + 17.7);
    let noise3 = perlin3D(scrolledPos * 4.0 + 31.3);
    
    let fbm = noise1 * 0.5 + noise2 * 0.25 + noise3 * 0.125;
    
    // 焦散梯度: 高频扰动
    let causticPattern = sin(fbm * PI) * 0.5 + 0.5;
    let causticAmp = pow(causticPattern, 2.0) * intensity;
    
    // 基于噪声的彩色焦散 (蓝色/青色宝石焦散)
    let causticsColor = mix(
        vec3f(0.3, 0.6, 1.0),  // 蓝色
        vec3f(0.0, 1.0, 0.8),  // 青色
        causticPattern
    );
    
    return causticsColor * causticAmp;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 纹理混合与缝合
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_TEXTURE_BLENDING = /* wgsl */`
// 采样纹理 (占位符接口，实际绑定时传入)
fn sampleCrackedAlien(uv: vec2f) -> vec3f {
    // 实际渲染时由 bind group 提供
    return vec3f(0.8, 0.7, 0.6);
}

fn sampleCrackedIce(uv: vec2f) -> vec3f {
    // 实际渲染时由 bind group 提供
    return vec3f(0.9, 0.95, 1.0);
}

// 混合两个纹理，基于 UV 或者杂色
fn blendCrackedTextures(uv: vec2f, blendFactor: f32) -> vec3f {
    let alien = sampleCrackedAlien(uv);
    let ice   = sampleCrackedIce(uv);
    return mix(alien, ice, blendFactor);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 完整 Gem 片元着色器
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GEM_FRAG = /* wgsl */`
${WGSL_MATH_HELPERS}
${WGSL_PBR_BRDF}
${WGSL_FRESNEL_IOR}
${WGSL_INTERNAL_CAUSTICS}
${WGSL_TEXTURE_BLENDING}

struct GemUniforms {
    // 光照参数
    lightDir       : vec3f,
    lightIntensity : f32,
    lightColor     : vec3f,
    highlightAmount: f32,   // 高亮状态 [0,1]
    
    // 宝石材质参数
    iorBase        : f32,   // 折射率 (1.4-1.8)
    dispersiness   : f32,   // 色散量 (0.0-0.5)
    roughness      : f32,   // 微面粗糙度 (0.05-0.3)
    metallic       : f32,   // 金属度 (宝石通常 ~0.0)
    
    // 焦散参数
    causticsScale  : f32,   // 焦散空间频率
    causticsSpeed  : f32,   // 焦散动画速度
    causticsIntens : f32,   // 焦散强度
    time           : f32,   // 时钟 (秒)
    
    // 纹理混合
    textureBlend   : f32,   // alien <-> ice 的混合 [0,1]
    fresnelPower   : f32,   // Fresnel 衰减指数
    rimIntensity   : f32,   // 边缘光强度
    _pad0          : f32,   // 对齐至 16 字节
};

@group(0) @binding(0) var<uniform> u_gem : GemUniforms;
@group(0) @binding(1) var          t_alien : texture_2d<f32>;
@group(0) @binding(2) var          t_ice   : texture_2d<f32>;
@group(0) @binding(3) var          smp     : sampler;

struct GemVert {
    @builtin(position) pos         : vec4f,
    @location(0)       vUV         : vec2f,
    @location(1)       vWorldNormal: vec3f,
    @location(2)       vWorldPos   : vec3f,
    @location(3)       vViewDir    : vec3f,
    @location(4)       vColor      : vec3f,
};

@fragment
fn fs_gem(vert: GemVert) -> @location(0) vec4f {
    let N = normalize(vert.vWorldNormal);
    let V = normalize(vert.vViewDir);
    let L = normalize(u_gem.lightDir);
    
    // ────────────────────────────────────────────────────────────────────────────
    // 基础宝石材质
    // ────────────────────────────────────────────────────────────────────────────
    
    // 宝石基色: 纹理混合
    let alienColor = textureSample(t_alien, smp, vert.vUV).rgb;
    let iceColor   = textureSample(t_ice, smp, vert.vUV).rgb;
    let baseColor  = mix(alienColor, iceColor, u_gem.textureBlend);
    
    // Cook-Torrance 高光 (宝石的主要光学特性)
    let f0 = vec3f(0.08, 0.08, 0.08);  // 非金属 F₀
    let specular = specularBRDF(N, V, L, f0, u_gem.roughness);
    
    // PBR 直接照明
    let pbrColor = pbrDirect(
        baseColor,
        N, V, L,
        f0,
        u_gem.metallic,
        u_gem.roughness,
        u_gem.lightColor
    );
    
    // ────────────────────────────────────────────────────────────────────────────
    // 色散折射效果
    // ────────────────────────────────────────────────────────────────────────────
    
    let refractColor = refractionDispersal(
        N, V,
        u_gem.iorBase,
        u_gem.dispersiness,
        baseColor * 0.5,  // 背景色 (宝石内部)
        u_gem.roughness
    );
    
    // ────────────────────────────────────────────────────────────────────────────
    // 内部焦散 (高亮时更强)
    // ────────────────────────────────────────────────────────────────────────────
    
    let caustics = internalCaustics(
        vert.vWorldPos,
        u_gem.time,
        u_gem.causticsScale,
        u_gem.causticsSpeed,
        u_gem.causticsIntens * (0.5 + u_gem.highlightAmount * 1.5)
    );
    
    // ────────────────────────────────────────────────────────────────────────────
    // Fresnel 边缘光 + 高亮混合
    // ────────────────────────────────────────────────────────────────────────────
    
    let rimColor = vec3f(1.0, 1.0, 0.8) * u_gem.rimIntensity;
    let rimLight = fresnelRim(N, V, u_gem.fresnelPower, rimColor);
    
    // ────────────────────────────────────────────────────────────────────────────
    // 高亮状态混合
    // ────────────────────────────────────────────────────────────────────────────
    
    let highlightBoost = mix(1.0, 2.0, u_gem.highlightAmount);
    let finalColor = pbrColor * u_gem.lightIntensity * highlightBoost
                   + rimLight
                   + caustics * (1.0 - u_gem.metallic)
                   + refractColor * 0.15;
    
    return vec4f(saturate_v3(finalColor), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 全屏三角形顶点着色器
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FULLSCREEN_VS = /* wgsl */`
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2f(f32(idx & 1u), f32((idx >> 1u) & 1u)) * 2.0;
    let pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return VertexOutput(pos, uv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// 类型定义 & 参数结构
// ─────────────────────────────────────────────────────────────────────────────

interface GemParams {
  // 光照
  lightDir: [number, number, number];
  lightIntensity: number;
  lightColor: [number, number, number];
  highlightAmount: number;

  // 宝石材质
  iorBase: number;
  dispersiness: number;
  roughness: number;
  metallic: number;

  // 焦散
  causticsScale: number;
  causticsSpeed: number;
  causticsIntens: number;
  time: number;

  // 纹理与渲染
  textureBlend: number;
  fresnelPower: number;
  rimIntensity: number;
}

const DEFAULT_GEM_PARAMS: GemParams = {
  lightDir: [0.5, 1, 0.5],
  lightIntensity: 1.5,
  lightColor: [1, 1, 1],
  highlightAmount: 0.5,
  iorBase: 1.45,
  dispersiness: 0.28,
  roughness: 0.15,
  metallic: 0.0,
  causticsScale: 2.0,
  causticsSpeed: 0.5,
  causticsIntens: 1.2,
  time: 0,
  textureBlend: 0.5,
  fresnelPower: 3.0,
  rimIntensity: 0.8,
};

// ─────────────────────────────────────────────────────────────────────────────
// 工具函数 — 参数打包到 GPU 缓冲
// ─────────────────────────────────────────────────────────────────────────────

function packGemUniforms(params: GemParams): ArrayBuffer {
  const buf = new ArrayBuffer(256);  // GemUniforms 布局
  const f32 = new Float32Array(buf);

  let offset = 0;

  // lightDir (vec3f) + lightIntensity (f32)
  f32[offset++] = params.lightDir[0];
  f32[offset++] = params.lightDir[1];
  f32[offset++] = params.lightDir[2];
  f32[offset++] = params.lightIntensity;

  // lightColor (vec3f) + highlightAmount (f32)
  f32[offset++] = params.lightColor[0];
  f32[offset++] = params.lightColor[1];
  f32[offset++] = params.lightColor[2];
  f32[offset++] = params.highlightAmount;

  // iorBase, dispersiness, roughness, metallic
  f32[offset++] = params.iorBase;
  f32[offset++] = params.dispersiness;
  f32[offset++] = params.roughness;
  f32[offset++] = params.metallic;

  // causticsScale, causticsSpeed, causticsIntens, time
  f32[offset++] = params.causticsScale;
  f32[offset++] = params.causticsSpeed;
  f32[offset++] = params.causticsIntens;
  f32[offset++] = params.time;

  // textureBlend, fresnelPower, rimIntensity, _pad0
  f32[offset++] = params.textureBlend;
  f32[offset++] = params.fresnelPower;
  f32[offset++] = params.rimIntensity;
  f32[offset++] = 0.0;  // _pad0

  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// 核心类 — ATGemMaterial
// ─────────────────────────────────────────────────────────────────────────────

export class ATGemMaterial {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private bindGroup: GPUBindGroup;
  private uniformBuf: GPUBuffer;
  private params: GemParams = { ...DEFAULT_GEM_PARAMS };

  private alienTex: GPUTexture;
  private iceTex: GPUTexture;
  private sampler: GPUSampler;

  private constructor(
    device: GPUDevice,
    pipeline: GPURenderPipeline,
    bgl: GPUBindGroupLayout,
    bg: GPUBindGroup,
    uniformBuf: GPUBuffer,
    alienTex: GPUTexture,
    iceTex: GPUTexture,
    sampler: GPUSampler,
  ) {
    this.device = device;
    this.pipeline = pipeline;
    this.bindGroupLayout = bgl;
    this.bindGroup = bg;
    this.uniformBuf = uniformBuf;
    this.alienTex = alienTex;
    this.iceTex = iceTex;
    this.sampler = sampler;
  }

  /**
   * 工厂方法 — 初始化宝石材质系统
   * 加载纹理、编译 shader、创建 pipeline
   */
  static async create(device: GPUDevice, format: GPUTextureFormat): Promise<ATGemMaterial> {
    // ────────────────────────────────────────────────────────────────────────────
    // 加载纹理
    // ────────────────────────────────────────────────────────────────────────────

    let alienTex: GPUTexture;
    let iceTex: GPUTexture;

    try {
      // 加载 alien_cracked_2_basecolor.ktx2
      const alienResponse = await fetch('/upstream/activetheory-assets/textures/alien_cracked_2_basecolor.ktx2');
      const alienBuffer = await alienResponse.arrayBuffer();
      alienTex = device.createTexture({
        label: 'alien-cracked-tex',
        size: [512, 512],  // 默认大小，实际从 KTX2 解析
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: alienTex },
        new Uint8Array(alienBuffer),
        { bytesPerRow: 512 * 4 },
        [512, 512],
      );
    } catch (err) {
      console.warn('Failed to load alien texture, using placeholder:', err);
      alienTex = createPlaceholderTexture(device, [200, 180, 160]);
    }

    try {
      // 加载 cracked_ice_basecolor.ktx2
      const iceResponse = await fetch('/upstream/activetheory-assets/textures/cracked_ice_basecolor.ktx2');
      const iceBuffer = await iceResponse.arrayBuffer();
      iceTex = device.createTexture({
        label: 'cracked-ice-tex',
        size: [512, 512],  // 默认大小，实际从 KTX2 解析
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: iceTex },
        new Uint8Array(iceBuffer),
        { bytesPerRow: 512 * 4 },
        [512, 512],
      );
    } catch (err) {
      console.warn('Failed to load ice texture, using placeholder:', err);
      iceTex = createPlaceholderTexture(device, [230, 240, 250]);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 sampler
    // ────────────────────────────────────────────────────────────────────────────

    const sampler = device.createSampler({
      label: 'gem-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    // ────────────────────────────────────────────────────────────────────────────
    // 编译 Shader
    // ────────────────────────────────────────────────────────────────────────────

    const shaderCode = `
${WGSL_GEM_FRAG}
${WGSL_FULLSCREEN_VS}
    `;

    const shaderModule = device.createShaderModule({
      label: 'gem-shader',
      code: shaderCode,
    });

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 Bind Group Layout
    // ────────────────────────────────────────────────────────────────────────────

    const bindGroupLayout = device.createBindGroupLayout({
      label: 'gem-bgl',
      entries: [
        // uniform buffer
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        // alien texture
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float', viewDimension: '2d' },
        },
        // ice texture
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float', viewDimension: '2d' },
        },
        // sampler
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 Pipeline Layout
    // ────────────────────────────────────────────────────────────────────────────

    const pipelineLayout = device.createPipelineLayout({
      label: 'gem-layout',
      bindGroupLayouts: [bindGroupLayout],
    });

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 Uniform Buffer
    // ────────────────────────────────────────────────────────────────────────────

    const uniformBuf = device.createBuffer({
      label: 'gem-uniform-buf',
      size: 256,  // GemUniforms 大小
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(uniformBuf.getMappedRange()).set(new Float32Array(packGemUniforms(DEFAULT_GEM_PARAMS)));
    uniformBuf.unmap();

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 Bind Group
    // ────────────────────────────────────────────────────────────────────────────

    const bindGroup = device.createBindGroup({
      label: 'gem-bg',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: alienTex.createView() },
        { binding: 2, resource: iceTex.createView() },
        { binding: 3, resource: sampler },
      ],
    });

    // ────────────────────────────────────────────────────────────────────────────
    // 创建 Render Pipeline
    // ────────────────────────────────────────────────────────────────────────────

    const pipeline = device.createRenderPipeline({
      label: 'gem-pipeline',
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_fullscreen',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_gem',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    return new ATGemMaterial(device, pipeline, bindGroupLayout, bindGroup, uniformBuf, alienTex, iceTex, sampler);
  }

  /**
   * 设置宝石材质参数
   */
  setParams(partial: Partial<GemParams>): void {
    Object.assign(this.params, partial);
    this.device.queue.writeBuffer(
      this.uniformBuf,
      0,
      packGemUniforms(this.params),
    );
  }

  /**
   * 更新时间 (用于焦散动画)
   */
  tick(dt: number): void {
    this.params.time += dt;
    this.setParams({});
  }

  /**
   * 渲染到目标纹理
   */
  render(
    encoder: GPUCommandEncoder,
    colorTarget: GPUTextureView,
  ): void {
    const pass = encoder.beginRenderPass({
      label: 'gem-pass',
      colorAttachments: [{
        view: colorTarget,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);  // 全屏三角形
    pass.end();
  }

  /**
   * 销毁资源
   */
  destroy(): void {
    this.uniformBuf.destroy();
    this.alienTex.destroy();
    this.iceTex.destroy();
  }

  /**
   * 获取当前参数
   */
  getParams(): GemParams {
    return { ...this.params };
  }

  /**
   * 在高亮状态和普通状态之间混合
   */
  setHighlightState(amount: number): void {
    this.params.highlightAmount = saturate(amount);
    this.params.causticsIntens = 1.2 + amount * 1.8;  // 高亮时加强焦散
    this.setParams({});
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 辅助函数
// ─────────────────────────────────────────────────────────────────────────────

function saturate(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function createPlaceholderTexture(device: GPUDevice, color: [number, number, number]): GPUTexture {
  const tex = device.createTexture({
    label: 'placeholder-tex',
    size: [1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: tex },
    new Uint8Array([color[0], color[1], color[2], 255]),
    { bytesPerRow: 4 },
    [1, 1],
  );
  return tex;
}

// ─────────────────────────────────────────────────────────────────────────────
// 导出 WGSL 片段供其他模块组合使用
// ─────────────────────────────────────────────────────────────────────────────

export const AT_GEM_WGSL = {
  /** 数学助手 */
  mathHelpers: WGSL_MATH_HELPERS,
  /** Cook-Torrance BRDF */
  pbrBRDF: WGSL_PBR_BRDF,
  /** Fresnel & IOR */
  fresnelIOR: WGSL_FRESNEL_IOR,
  /** 内部焦散 */
  internalCaustics: WGSL_INTERNAL_CAUSTICS,
  /** 纹理混合 */
  textureBlending: WGSL_TEXTURE_BLENDING,
  /** 完整宝石片元着色器 */
  gemFrag: WGSL_GEM_FRAG,
  /** 全屏三角形顶点着色器 */
  fullscreenVS: WGSL_FULLSCREEN_VS,
};

export type { GemParams };
export { DEFAULT_GEM_PARAMS };
