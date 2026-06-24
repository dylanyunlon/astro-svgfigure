/**
 * atmosphere.ts — 大气散射 + 雾效果后处理
 *
 * 功能：
 *   1. RAYLEIGH 散射  — 基于 lygia/lighting/atmosphere.wgsl 的 Rayleigh 相位函数，
 *                       给远处粒子施加大气透视（近处清晰, 远处偏蓝/发雾）。
 *   2. MIE 散射       — Henyey-Greenstein 相位函数模拟前散射光晕（太阳附近雾霭）。
 *   3. 深度雾          — 指数平方雾（exp²），基于归一化深度字段混合雾色，
 *                       与 metaball field 字段联动: field 越低 → 越远 → 越多雾。
 *   4. 色差偏移        — 大气色差: R 通道轻微向外偏移, B 通道向内，模拟真实镜头。
 *
 * 设计参考：
 *   - upstream/lygia/lighting/atmosphere.wgsl  (Rayleigh + Mie 散射结构)
 *   - upstream/lygia/lighting/common/rayleigh.wgsl (相位函数)
 *   - GPU Gems 2 Ch.16 "Accurate Atmospheric Scattering"
 *
 * 用法：
 *   const atmo = await AtmospherePass.create(device, format, width, height);
 *   // 每帧（在 ParticleRenderer 输出之后调用）：
 *   atmo.setParams({ sunElevation: 0.6, fogDensity: 0.18 });
 *   atmo.render(encoder, particleOutputView, finalTargetView, fieldTexView, sampler);
 */

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader — 大气散射 + 深度雾全屏后处理
// ─────────────────────────────────────────────────────────────────────────────









const ATMOSPHERE_WGSL = /* wgsl */`

// ─── Uniforms ────────────────────────────────────────────────────────────────
struct AtmoUniforms {
  // 分辨率
  width          : f32,
  height         : f32,
  // 太阳方向 (归一化 screen-space XY + elevation)
  sunX           : f32,
  sunY           : f32,
  sunElevation   : f32,   // [0, 1]  0=地平线, 1=正顶
  // Rayleigh 散射系数 (λ⁻⁴, 预乘缩放)
  rayleighR      : f32,   // 默认 5.5e-6
  rayleighG      : f32,   // 默认 13.0e-6
  rayleighB      : f32,   // 默认 22.4e-6
  // Mie 散射系数 (各向同性预乘)
  mieCoeff       : f32,   // 默认 21.0e-6
  mieG           : f32,   // Henyey-Greenstein 不对称因子 [0,1], 默认 0.76
  // 雾参数
  fogDensity     : f32,   // exp² 密度 [0, 1]
  fogNear        : f32,   // 雾起始深度 (field 阈值, 低于此值才生雾)
  // 雾色 (线性 sRGB)
  fogR           : f32,
  fogG           : f32,
  fogB           : f32,
  // 大气透视强度
  atmosphereStrength : f32,  // [0, 1]
  // 色差幅度 (像素)
  aberrationAmt  : f32,
  // 太阳功率 (高光强度)
  sunPower       : f32,
  _pad0          : f32,
}

@group(0) @binding(0) var<uniform> u       : AtmoUniforms;
@group(0) @binding(1) var          smp     : sampler;
@group(0) @binding(2) var          srcTex  : texture_2d<f32>;  // particle 渲染结果
@group(0) @binding(3) var          fieldTex: texture_2d<f32>;  // metaball field 积累

// ─── 全屏三角形顶点 ───────────────────────────────────────────────────────────
struct FSVert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> FSVert {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32( vi         & 2u) * 2.0 - 1.0;
  var out: FSVert;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return out;
}

// ─── 物理常数 ─────────────────────────────────────────────────────────────────
const PI : f32 = 3.14159265358979;

// ─── Rayleigh 相位函数 ────────────────────────────────────────────────────────
// 来源: lygia/lighting/common/rayleigh.wgsl
// μ = cos(scattering angle)
fn rayleighPhase(mu: f32) -> f32 {
  return 3.0 * (1.0 + mu * mu) / (16.0 * PI);
}

// ─── Henyey-Greenstein Mie 相位函数 ──────────────────────────────────────────
// 来源: lygia/lighting/common/henyeyGreenstein.wgsl
fn henyeyGreenstein(mu: f32, g: f32) -> f32 {
  let g2  = g * g;
  let denom = 1.0 + g2 - 2.0 * g * mu;
  return (1.0 - g2) / (4.0 * PI * pow(abs(denom), 1.5) + 1e-6);
}

// ─── 大气散射颜色计算 ─────────────────────────────────────────────────────────
// 根据视线方向 (screen UV → 伪 3D ray) 与太阳方向计算单次散射颜色。
// 简化版: 不做光线步进, 仅使用相位 + 光学深度近似 (适合 2D 粒子场景)。
fn atmosphereScatter(uv: vec2f, sunDir: vec3f) -> vec3f {
  // 将 2D 屏幕坐标映射到半球视线方向
  let ndc   = uv * 2.0 - vec2f(1.0);
  let eyeDir = normalize(vec3f(ndc * 0.8, 1.0));  // 轻微广角投影

  let mu    = dot(eyeDir, sunDir);                 // 散射角余弦

  // Rayleigh 系数 (λ⁻⁴ 预乘)
  let betaR = vec3f(u.rayleighR, u.rayleighG, u.rayleighB);

  // 光学深度近似: 沿视线积分大气密度, 用简单指数衰减代替步进
  // h(elevation) → density scale: 低仰角 (sunElevation→0) 路径更长 → 更多散射
  let optDepth  = max(0.01, sunDir.y + 0.05) ;  // 避免 div by zero
  let depthScale = 1.0 / optDepth;              // 光学深度倍增

  // Rayleigh 单次散射
  let phaseR  = rayleighPhase(mu);
  let scatterR = betaR * phaseR * depthScale * 0.001;

  // Mie 单次散射 (太阳附近光晕)
  let betaM   = vec3f(u.mieCoeff);
  let phaseM  = henyeyGreenstein(mu, u.mieG);
  let scatterM = betaM * phaseM * depthScale * 0.0005;

  // 透射率: 到达相机前被散射/吸收的比例
  let transmittance = exp(-betaR * depthScale * 0.08 - betaM * depthScale * 0.04);

  // 合并 Rayleigh + Mie, 乘以太阳功率
  let scatter = (scatterR + scatterM) * u.sunPower;

  return clamp(scatter, vec3f(0.0), vec3f(1.0));
}

// ─── 指数平方雾 ───────────────────────────────────────────────────────────────
// field 值越低 → 粒子越远/越少 → fogFactor 越大
fn expSquaredFog(field: f32) -> f32 {
  // 将 field 反转为 "距离": 远 = 1-field (归一化到 0–1 内)
  let dist = clamp(1.0 - field / max(u.fogNear, 0.001), 0.0, 1.0);
  return 1.0 - exp(-u.fogDensity * u.fogDensity * dist * dist * 8.0);
}

// ─── 色差偏移 ─────────────────────────────────────────────────────────────────
fn chromaticAberration(uv: vec2f, strength: f32) -> vec3f {
  let px     = vec2f(strength / u.width, strength / u.height);
  let center = vec2f(0.5);
  let dir    = normalize(uv - center + vec2f(1e-4));

  let r = textureSampleLevel(srcTex, smp, uv + dir * px * 1.5,  0.0).r;
  let g = textureSampleLevel(srcTex, smp, uv,                   0.0).g;
  let b = textureSampleLevel(srcTex, smp, uv - dir * px * 1.0,  0.0).b;
  return vec3f(r, g, b);
}

// ─── Fragment main ────────────────────────────────────────────────────────────
@fragment fn fs_main(in: FSVert) -> @location(0) vec4f {
  let uv = in.uv;

  // 1. 读取原始粒子颜色 (带色差偏移)
  let srcAlpha  = textureSampleLevel(srcTex,   smp, uv, 0.0).a;
  var color     = chromaticAberration(uv, u.aberrationAmt);

  // 透明区域 (背景) 直接输出
  if (srcAlpha < 0.01) {
    return vec4f(color, srcAlpha);
  }

  // 2. 采样 field 纹理获取深度代理
  let fieldSample = textureSampleLevel(fieldTex, smp, uv, 0.0).r;

  // 3. 太阳方向 (screen-space 3D)
  let sunDir = normalize(vec3f(u.sunX, u.sunY, max(u.sunElevation, 0.01)));

  // 4. 大气散射颜色
  let atmoColor = atmosphereScatter(uv, sunDir);

  // 5. 深度雾混合
  let fogColor  = vec3f(u.fogR, u.fogG, u.fogB);
  let fogFactor = expSquaredFog(fieldSample) * u.atmosphereStrength;

  // 先混合雾色
  color = mix(color, fogColor, fogFactor * 0.65);

  // 6. 大气透视叠加: 在雾的基础上叠加 Rayleigh 蓝偏
  //    越远 (fogFactor 越大) Rayleigh 贡献越强
  let rayleighTint = mix(vec3f(0.0), atmoColor, fogFactor * u.atmosphereStrength);
  color = color + rayleighTint * 0.4;

  // 7. 近处保持清晰: field 强的地方 (近处稠密粒子) 还原饱和度
  let nearnessMask = smoothstep(0.1, 0.6, fieldSample);
  color = mix(color, textureSampleLevel(srcTex, smp, uv, 0.0).rgb, nearnessMask * 0.3);

  // 8. 色调映射 (Reinhard) 防止过曝
  color = color / (color + vec3f(1.0));

  // 9. 伽马近似 sRGB 输出
  color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

  return vec4f(color, srcAlpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript 公共接口
// ─────────────────────────────────────────────────────────────────────────────

export interface AtmosphereParams {
  /** 太阳在屏幕空间的 X 方向 (NDC, -1~1), 默认 0.3 */
  sunX?: number;
  /** 太阳在屏幕空间的 Y 方向 (NDC, -1~1), 默认 0.5 */
  sunY?: number;
  /** 太阳仰角 [0=地平线, 1=正顶], 控制散射路径长度, 默认 0.4 */
  sunElevation?: number;
  /** Rayleigh 散射系数 R 通道 (λ⁻⁴), 默认 5.5e-6 */
  rayleighR?: number;
  /** Rayleigh 散射系数 G 通道, 默认 13.0e-6 */
  rayleighG?: number;
  /** Rayleigh 散射系数 B 通道, 默认 22.4e-6 */
  rayleighB?: number;
  /** Mie 散射系数, 默认 21e-6 */
  mieCoeff?: number;
  /** Henyey-Greenstein 不对称因子 [0,1], 默认 0.76 */
  mieG?: number;
  /** 指数平方雾密度 [0,1], 默认 0.18 */
  fogDensity?: number;
  /** field 值高于此视为"近处", 默认 0.35 (= metaball threshold) */
  fogNear?: number;
  /** 雾色 RGB [0,1], 默认淡蓝灰 [0.72, 0.80, 0.90] */
  fogColor?: [number, number, number];
  /** 大气透视总强度 [0,1], 默认 0.85 */
  atmosphereStrength?: number;
  /** 色差偏移像素数, 默认 1.5 */
  aberrationAmt?: number;
  /** 太阳高光功率倍数, 默认 20.0 */
  sunPower?: number;
}

// 内部状态 (16 个 f32 = 64 字节, 对齐至 16)
const UNIFORM_F32_COUNT = 20;  // 含 pad, 必须是 4 的倍数
const UNIFORM_BYTE_SIZE = UNIFORM_F32_COUNT * 4;

// ─────────────────────────────────────────────────────────────────────────────
// AtmospherePass
// ─────────────────────────────────────────────────────────────────────────────

export class AtmospherePass {
  private readonly device : any /*GPUDevice*/;
  private readonly format : GPUTextureFormat;

  private pipeline  !: any /*GPURenderPipeline*/;
  private bgl       !: GPUBindGroupLayout;
  private uniBuf    !: any /*GPUBuffer*/;
  private sampler   !: GPUSampler;

  // 当前绑定组 (依赖外部 texture view, 每帧重建或按需重建)
  private bg        : any /*GPUBindGroup*/ | null = null;
  private lastSrcView  : any /*GPUTextureView*/ | null = null;
  private lastFldView  : any /*GPUTextureView*/ | null = null;

  // 渲染分辨率
  private width  : number;
  private height : number;

  // 参数 (含默认值)
  private params: Required<AtmosphereParams> = {
    sunX              : 0.3,
    sunY              : 0.5,
    sunElevation      : 0.4,
    rayleighR         : 5.5e-6,
    rayleighG         : 13.0e-6,
    rayleighB         : 22.4e-6,
    mieCoeff          : 21e-6,
    mieG              : 0.76,
    fogDensity        : 0.18,
    fogNear           : 0.35,
    fogColor          : [0.72, 0.80, 0.90],
    atmosphereStrength: 0.85,
    aberrationAmt     : 1.5,
    sunPower          : 20.0,
  };

  private constructor(device: any /*GPUDevice*/, format: GPUTextureFormat, width: number, height: number) {
    this.device = device;
    this.format = format;
    this.width  = width;
    this.height = height;
  }

  // ── 工厂方法 ────────────────────────────────────────────────────────────────
  static async create(
    device : any /*GPUDevice*/,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<AtmospherePass> {
    const pass = new AtmospherePass(device, format, width, height);
    await pass._buildPipeline();
    return pass;
  }

  // ── 公共 API ─────────────────────────────────────────────────────────────────
  setParams(p: AtmosphereParams): void {
    Object.assign(this.params, p);
  }

  /** 更新分辨率（如 canvas resize）*/
  resize(width: number, height: number): void {
    this.width  = width;
    this.height = height;
  }

  /**
   * 执行大气散射后处理 Pass。
   *
   * @param encoder   当前帧的 GPUCommandEncoder
   * @param srcView   ParticleRenderer 的输出纹理 view（粒子渲染结果）
   * @param dstView   最终输出目标 view（swap-chain 或后续 pass 的输入）
   * @param fieldView metaball field 累积纹理 view（提供深度信息）
   */
  render(
    encoder  : any /*GPUCommandEncoder*/,
    srcView  : any /*GPUTextureView*/,
    dstView  : any /*GPUTextureView*/,
    fieldView: any /*GPUTextureView*/,
  ): void {
    this._uploadUniforms();
    this._ensureBindGroup(srcView, fieldView);

    const pass = encoder.beginRenderPass({
      label           : "atmosphere-pass",
      colorAttachments: [{
        view    : dstView,
        loadOp  : "load",
        storeOp : "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg!);
    pass.draw(3);   // 全屏三角形
    pass.end();
  }

  /** 释放所有 GPU 资源 */
  destroy(): void {
    this.uniBuf?.destroy();
  }

  // ── 私有实现 ─────────────────────────────────────────────────────────────────

  private async _buildPipeline(): Promise<void> {
    const d  = this.device;
    const sm = d.createShaderModule({ label: "atmosphere-shader", code: ATMOSPHERE_WGSL });

    this.bgl = d.createBindGroupLayout({
      label  : "atmosphere-bgl",
      entries: [
        // binding 0: uniform buffer
        {
          binding   : 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer    : { type: "uniform" },
        },
        // binding 1: sampler
        {
          binding   : 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler   : { type: "filtering" },
        },
        // binding 2: source texture (particle render output)
        {
          binding   : 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture   : { sampleType: "float", viewDimension: "2d" },
        },
        // binding 3: field texture (metaball accumulation, depth proxy)
        {
          binding   : 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture   : { sampleType: "float", viewDimension: "2d" },
        },
      ],
    });

    this.pipeline = await d.createRenderPipelineAsync({
      label : "atmosphere-pipeline",
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      vertex  : { module: sm, entryPoint: "vs_main" },
      fragment: {
        module    : sm,
        entryPoint: "fs_main",
        targets   : [{
          format: this.format,
          blend : {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.uniBuf = d.createBuffer({
      label: "atmo-uniform",
      size : UNIFORM_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = d.createSampler({
      label    : "atmo-sampler",
      magFilter: "linear",
      minFilter: "linear",
    });
  }

  private _uploadUniforms(): void {
    const p   = this.params;
    const arr = new Float32Array(UNIFORM_F32_COUNT);

    arr[0]  = this.width;
    arr[1]  = this.height;
    arr[2]  = p.sunX;
    arr[3]  = p.sunY;
    arr[4]  = p.sunElevation;
    arr[5]  = p.rayleighR;
    arr[6]  = p.rayleighG;
    arr[7]  = p.rayleighB;
    arr[8]  = p.mieCoeff;
    arr[9]  = p.mieG;
    arr[10] = p.fogDensity;
    arr[11] = p.fogNear;
    arr[12] = p.fogColor[0];
    arr[13] = p.fogColor[1];
    arr[14] = p.fogColor[2];
    arr[15] = p.atmosphereStrength;
    arr[16] = p.aberrationAmt;
    arr[17] = p.sunPower;
    arr[18] = 0.0;  // _pad0
    arr[19] = 0.0;  // _pad1

    this.device.queue.writeBuffer(this.uniBuf, 0, arr);
  }

  private _ensureBindGroup(srcView: any /*GPUTextureView*/, fieldView: any /*GPUTextureView*/): void {
    // 如果 texture views 没变，复用上一个绑定组
    if (this.bg && this.lastSrcView === srcView && this.lastFldView === fieldView) return;

    this.bg = this.device.createBindGroup({
      label  : "atmosphere-bg",
      layout : this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: srcView },
        { binding: 3, resource: fieldView },
      ],
    });
    this.lastSrcView = srcView;
    this.lastFldView = fieldView;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 便捷工厂 + 预设
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 大气效果预设
 *
 * CLEAR_DAY   — 晴天，淡蓝色大气透视，轻微雾
 * DUSK        — 黄昏，橙红色散射，中等雾
 * DEEP_OCEAN  — 深海，强蓝色 Rayleigh，高密度雾
 * HAZE        — 霾天，高雾密度，低大气色差
 */
export const ATMOSPHERE_PRESETS = {
  CLEAR_DAY: {
    sunElevation      : 0.65,
    fogDensity        : 0.12,
    fogColor          : [0.75, 0.83, 0.95] as [number, number, number],
    atmosphereStrength: 0.70,
    rayleighR         : 5.5e-6,
    rayleighG         : 13.0e-6,
    rayleighB         : 22.4e-6,
    mieCoeff          : 12e-6,
    sunPower          : 22.0,
  },
  DUSK: {
    sunElevation      : 0.08,
    fogDensity        : 0.28,
    fogColor          : [0.85, 0.60, 0.40] as [number, number, number],
    atmosphereStrength: 0.95,
    rayleighR         : 18e-6,
    rayleighG         : 9e-6,
    rayleighB         : 4e-6,
    mieCoeff          : 35e-6,
    mieG              : 0.88,
    sunPower          : 28.0,
  },
  DEEP_OCEAN: {
    sunElevation      : 0.35,
    fogDensity        : 0.45,
    fogColor          : [0.05, 0.18, 0.55] as [number, number, number],
    atmosphereStrength: 1.0,
    rayleighR         : 3e-6,
    rayleighG         : 15e-6,
    rayleighB         : 38e-6,
    mieCoeff          : 8e-6,
    sunPower          : 15.0,
  },
  HAZE: {
    sunElevation      : 0.30,
    fogDensity        : 0.55,
    fogColor          : [0.80, 0.80, 0.78] as [number, number, number],
    atmosphereStrength: 0.90,
    rayleighR         : 8e-6,
    rayleighG         : 14e-6,
    rayleighB         : 20e-6,
    mieCoeff          : 60e-6,
    mieG              : 0.60,
    aberrationAmt     : 0.5,
    sunPower          : 12.0,
  },
} satisfies Record<string, AtmosphereParams>;
