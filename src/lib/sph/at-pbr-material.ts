/**
 * at-pbr-material.ts — AT PBR Lighting System — WebGPU/WGSL Port
 *
 * 移植自 ActiveTheory 的完整 PBR 光照系统 (src/lib/shaders/compiled.vs):
 *   - pbr-cell-surface.frag  → Cook-Torrance BRDF (GGX/Smith)
 *   - matcap-fresnel-cell.frag → Matcap + Fresnel rim material
 *   - lygia/lighting/fresnel.glsl → Schlick Fresnel
 *   - lygia/lighting/iridescence.glsl → 薄膜干涉彩虹色
 *
 * 提供两条材质路径:
 *   1. PBRCellMaterial   — 全量 Cook-Torrance BRDF + 薄膜彩虹 + 大气雾
 *   2. MatcapFresnel     — Matcap + Fresnel rim (轻量, ~10× 快于全量 PBR)
 *
 * 用法:
 *   const mat = await ATPBRMaterial.create(device, format);
 *   mat.setParams({ albedo: [0.4, 0.8, 1.0], roughness: 0.35, metallic: 0.1 });
 *   mat.render(encoder, colorTargetView, depthView, uniformBuffer);
 *
 * Research: xiaodi #M712 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 公共数学助手 (内联自 lygia/math/)
// ─────────────────────────────────────────────────────────────────────────────









const WGSL_MATH_HELPERS = /* wgsl */`
// ── saturate ──────────────────────────────────────────────────────────────────
fn saturate_f(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn saturate_v3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }

// ── pow5 (lygia/math/pow5.glsl) ───────────────────────────────────────────────
fn pow5_f(v: f32) -> f32 { let v2 = v * v; return v2 * v2 * v; }
fn pow5_v3(v: vec3f) -> vec3f { let v2 = v * v; return v2 * v2 * v; }

const PI       : f32 = 3.14159265358979323846;
const TWO_PI   : f32 = 6.28318530717958647693;
const INV_PI   : f32 = 0.31830988618379067154;
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Cook-Torrance BRDF (移植自 lygia/lighting/pbr.glsl)
// GGX NDF + Smith 联合遮蔽函数 + Schlick Fresnel
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PBR_BRDF = /* wgsl */`
// ── Schlick Fresnel F₀+(1-F₀)(1-cosθ)⁵ ──────────────────────────────────────
fn F_Schlick(f0: vec3f, cosTheta: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow5_v3(vec3f(saturate_f(1.0 - cosTheta)));
}

// ── GGX / Trowbridge-Reitz NDF ────────────────────────────────────────────────
// 微面元法线分布函数: 控制高光形状
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

// ── Smith 联合遮蔽-阴影函数 (高度相关 GGX) ────────────────────────────────────
fn G_SmithGGX(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let gV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let gL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (gV + gL + 1e-7);
}

// ── Cook-Torrance 镜面 BRDF ───────────────────────────────────────────────────
// 返回 D*G*F; 调用方乘以 NdotL
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

// ── 完整 PBR 直接照明 (点光源) ────────────────────────────────────────────────
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

    // 能量守恒: 镜面反射占用漫反射份额
    let ks = F_Schlick(f0, saturate_f(dot(N, V)));
    let kd = (vec3f(1.0) - ks) * (1.0 - metallic);

    return (kd * diffuse + specular) * lightColor * NdotL;
}

// ── 简化 IBL 环境光 ───────────────────────────────────────────────────────────
// 无 mipmap env texture 时退回到常量半球环境光
fn pbrAmbientSimple(
    albedo   : vec3f,
    N        : vec3f,
    V        : vec3f,
    f0       : vec3f,
    metallic : f32,
    roughness: f32,
    ao       : f32,
    envColor : vec3f   // 场景环境色 (HDR 球谐系数近似)
) -> vec3f {
    let ks = F_Schlick(f0, saturate_f(dot(N, V)));
    let kd = (vec3f(1.0) - ks) * (1.0 - metallic);

    // 金属对环境光有更强的镜面贡献
    let diffuse  = kd * albedo * envColor;
    let specular = ks * envColor * mix(0.04, 1.0, metallic);

    return (diffuse + specular) * ao;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Fresnel 边缘光 (移植自 lygia/lighting/fresnel.glsl)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FRESNEL = /* wgsl */`
// ── Schlick Fresnel 边缘因子 ──────────────────────────────────────────────────
// 返回 0=正对 .. 1=掠射; power 控制过渡锐度 (4-8 典型值)
fn fresnel_f(N: vec3f, V: vec3f, power: f32) -> f32 {
    let cosTheta = saturate_f(dot(N, V));
    return pow(1.0 - cosTheta, power);
}

// ── 彩色 Fresnel 边缘光 ───────────────────────────────────────────────────────
fn fresnelRim(N: vec3f, V: vec3f, power: f32, rimColor: vec3f) -> vec3f {
    return rimColor * fresnel_f(N, V, power);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 薄膜干涉彩虹 (移植自 lygia/lighting/iridescence.glsl)
// Airy 函数近似, 在 450/550/650nm 三波段采样
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_IRIDESCENCE = /* wgsl */`
// ── 标量 Schlick Fresnel (单波长) ─────────────────────────────────────────────
fn _fresnelScalar(cosI: f32, eta: f32) -> f32 {
    let r0 = (1.0 - eta) / (1.0 + eta);
    let r0sq = r0 * r0;
    return r0sq + (1.0 - r0sq) * pow5_f(1.0 - cosI);
}

// ── 薄膜干涉 RGB (Airy 一阶近似) ─────────────────────────────────────────────
//   thickness — 薄膜厚度 nm (100-1000)
//   ior       — 薄膜折射率 n₂ (1.3-1.6 生物薄膜)
//   cosTheta  — 入射角余弦
fn iridescence(thickness: f32, ior: f32, cosTheta: f32) -> vec3f {
    // Snell 定律折射角余弦
    let sinThetaT2 = max(0.0, 1.0 - (1.0 - cosTheta * cosTheta) / (ior * ior));
    let cosThetaT  = sqrt(sinThetaT2);

    // 空气/薄膜界面 F₁, 薄膜/衬底界面 F₂
    let F1 = _fresnelScalar(cosTheta,  ior);
    let F2 = _fresnelScalar(cosThetaT, 1.0 / ior);

    // 光学路程差 (nm): 2 n₂ t cosθ_t
    let OPD = 2.0 * ior * thickness * cosThetaT;

    // 可见光波长 nm: 650 R / 550 G / 450 B
    let lambda = vec3f(650.0, 550.0, 450.0);

    // 相位差 δ = 2π·OPD/λ
    let delta = TWO_PI * OPD / lambda;

    // Airy 级数一阶近似: R = F1² + F2² + 2·F1·F2·cos(δ)
    let F1sq = F1 * F1;
    let F2sq = F2 * F2;
    let R    = vec3f(F1sq + F2sq) + 2.0 * F1 * F2 * cos(delta);

    return saturate_v3(R);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Matcap + Fresnel Cell Material
// 移植自 matcap-fresnel-cell.frag — AT 轻量材质
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_MATCAP_FRESNEL_FRAG = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────
struct MatcapUniforms {
    fresnelPower  : f32,   // rim 衰减指数, 默认 3.0
    fresnelColor  : vec3f, // rim 色调
    noiseScale    : f32,   // 法线扰动空间频率, 默认 1.4
    noiseStrength : f32,   // 法线扰动幅度 [0, 0.5]
    species       : vec3f, // 物种颜料色 RGB
    tintStrength  : f32,   // 物种色调混合权重 [0,1]
    time          : f32,   // 动画时钟 (秒)
    // 填充至 16 字节对齐
    _pad0         : f32,
    _pad1         : f32,
    _pad2         : f32,
}

@group(0) @binding(0) var<uniform> u_mc  : MatcapUniforms;
@group(0) @binding(1) var          t_mc  : texture_2d<f32>;   // 256×256 matcap
@group(0) @binding(2) var          smp   : sampler;

// ── 顶点/片元 IO ──────────────────────────────────────────────────────────────
struct MCVert {
    @builtin(position) pos         : vec4f,
    @location(0)       vUV         : vec2f,
    @location(1)       vWorldNormal: vec3f,
    @location(2)       vViewNormal : vec3f,
    @location(3)       vWorldPos   : vec3f,
}

// ── Simplex 噪声 (内联自 lygia/generative/snoise.glsl) ────────────────────────
fn _sn_mod289_v3(x: vec3f) -> vec3f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn _sn_mod289_v4(x: vec4f) -> vec4f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn _sn_permute(x: vec4f)   -> vec4f { return _sn_mod289_v4((x * 34.0 + 1.0) * x); }
fn _sn_tiSqrt(r: vec4f)    -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

fn snoise3(v: vec3f) -> f32 {
    let C = vec2f(1.0/6.0, 1.0/3.0);
    let D = vec4f(0.0, 0.5, 1.0, 2.0);

    var i  = floor(v + dot(v, vec3f(C.y)));
    let x0 = v - i + dot(i, vec3f(C.x));

    let g  = step(x0.yzx, x0.xyz);
    let l  = vec3f(1.0) - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + vec3f(C.x);
    let x2 = x0 - i2 + vec3f(C.y);
    let x3 = x0 - vec3f(D.y);

    i = _sn_mod289_v3(i);
    let p = _sn_permute(
        _sn_permute(
            _sn_permute(i.z + vec4f(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4f(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4f(0.0, i1.x, i2.x, 1.0));

    let n_  = 0.142857142857;
    let ns  = n_ * D.wyz - D.xzx;

    let j   = p - 49.0 * floor(p * ns.z * ns.z);
    let x_  = floor(j * ns.z);
    let y_  = floor(j - 7.0 * x_);

    let xv  = x_ * ns.x + ns.y;
    let yv  = y_ * ns.x + ns.y;
    let h   = vec4f(1.0) - abs(xv) - abs(yv);

    let b0  = vec4f(xv.xy, yv.xy);
    let b1  = vec4f(xv.zw, yv.zw);
    let s0  = floor(b0) * 2.0 + vec4f(1.0);
    let s1  = floor(b1) * 2.0 + vec4f(1.0);
    let sh  = -step(h, vec4f(0.0));

    let a0  = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1  = b1.xzyw + s1.xzyw * sh.zzww;

    var p0  = vec3f(a0.xy, h.x);
    var p1  = vec3f(a0.zw, h.y);
    var p2  = vec3f(a1.xy, h.z);
    var p3  = vec3f(a1.zw, h.w);

    let norm = _sn_tiSqrt(vec4f(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

    var m = max(0.6 - vec4f(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4f(0.0));
    m = m * m;
    return 42.0 * dot(m*m, vec4f(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// ── 噪声法线扰动 ───────────────────────────────────────────────────────────────
fn perturbNormalNoise(
    baseN   : vec3f,
    worldPos: vec3f,
    scale   : f32,
    strength: f32,
    t       : f32
) -> vec3f {
    let p  = worldPos * scale + vec3f(0.0, 0.0, t * 0.17);
    let nx = snoise3(p + vec3f(17.53, 0.0,   0.0));
    let ny = snoise3(p + vec3f(0.0,  31.71,  0.0));
    let nz = snoise3(p + vec3f(0.0,   0.0,  53.19));
    return normalize(baseN + vec3f(nx, ny, nz) * strength);
}

// ── 片元主函数 ────────────────────────────────────────────────────────────────
@fragment fn fs_matcap(in: MCVert) -> @location(0) vec4f {

    // 1. 噪声扰动世界空间法线
    let worldN = normalize(in.vWorldNormal);
    let pertN  = perturbNormalNoise(worldN, in.vWorldPos,
                                    u_mc.noiseScale, u_mc.noiseStrength, u_mc.time);

    // 2. 转换到视空间 — matcap UV 依赖视空间法线
    //    视空间 Z 轴指向观察者; vViewNormal 已在顶点阶段变换
    let viewN = normalize(in.vViewNormal + (pertN - worldN));  // 增量扰动

    // 3. Matcap UV 映射: u=(Nv.x*0.5+0.5), v=(Nv.y*0.5+0.5)
    var mcUV = clamp(viewN.xy * 0.5 + 0.5, vec2f(0.01), vec2f(0.99));
    let matcapColor = textureSample(t_mc, smp, mcUV).rgb;

    // 4. 物种色调
    let tintedColor = matcapColor * u_mc.species;
    let baseColor   = mix(matcapColor, tintedColor, u_mc.tintStrength);

    // 5. Fresnel 边缘光 (Schlick)
    let NdotV    = saturate_f(viewN.z);               // 视空间 Z ≈ NdotV
    let rimFactor = pow(1.0 - NdotV, u_mc.fresnelPower);
    var rim       = rimFactor * u_mc.fresnelColor;

    // 噪声扰动区域轻微衰减 rim (避免过亮)
    let noiseAtten = 1.0 - saturate_f(u_mc.noiseStrength * 1.5);
    rim *= mix(1.0, noiseAtten, 0.4);

    // 6. 合成: rim 以加法叠加 (同 AT upstream 惯例)
    let litColor = baseColor + rim;

    // 7. 软边 vignette alpha
    let edgeDist = 2.0 * length(in.vUV - vec2f(0.5));
    let alpha    = 1.0 - smoothstep(0.78, 1.0, edgeDist);

    // 8. 预乘 alpha 输出
    return vec4f(litColor * alpha, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 全量 PBR Cell Surface 材质
// 移植自 pbr-cell-surface.frag
// Cook-Torrance + Fresnel rim + 薄膜彩虹 + Reinhard 色调映射
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PBR_FRAG = /* wgsl */`
// ── PBR Uniforms ─────────────────────────────────────────────────────────────
struct PBRUniforms {
    // 材质
    albedo         : vec3f,
    metallic       : f32,
    roughness      : f32,
    ao             : f32,
    // 光源 & 相机
    lightPos       : vec3f,
    _p0            : f32,
    lightColor     : vec3f,
    _p1            : f32,
    cameraPos      : vec3f,
    _p2            : f32,
    // Fresnel 边缘光
    fresnelPower   : f32,
    fresnelColor   : vec3f,
    // 薄膜彩虹
    iridThickness  : f32,   // nm, 默认 500
    iridIOR        : f32,   // 薄膜 n₂, 默认 1.45
    iridStrength   : f32,   // 混合权重 [0,1]
    // 大气雾 (简化版)
    atmoDensity    : f32,
    atmoDepth      : f32,   // 本格深度 [0,1]
    atmoFogColor   : vec3f,
    // 环境光
    envColor       : vec3f,
    // 动画
    time           : f32,
    _p3            : f32,
}

@group(0) @binding(0) var<uniform> u_pbr : PBRUniforms;

// ── 顶点/片元 IO ──────────────────────────────────────────────────────────────
struct PBRVert {
    @builtin(position) pos         : vec4f,
    @location(0)       vUV         : vec2f,
    @location(1)       vWorldPos   : vec3f,
    @location(2)       vWorldNormal: vec3f,
}

// ── 片元主函数 ────────────────────────────────────────────────────────────────
@fragment fn fs_pbr(in: PBRVert) -> @location(0) vec4f {

    // ── 1. 法线 & 方向向量 ─────────────────────────────────────────────────────
    let N = normalize(in.vWorldNormal);
    let V = normalize(u_pbr.cameraPos - in.vWorldPos);
    let L = normalize(u_pbr.lightPos  - in.vWorldPos);

    // ── 2. PBR 材质参数 ────────────────────────────────────────────────────────
    // f0: 法线入射 Fresnel 反射率
    //   绝缘体 → 0.04; 金属 → albedo 着色
    let f0 = mix(vec3f(0.04), u_pbr.albedo, u_pbr.metallic);

    // ── 3. 直接照明 (Cook-Torrance) ────────────────────────────────────────────
    let directLit = pbrDirect(
        u_pbr.albedo, N, V, L, f0,
        u_pbr.metallic, u_pbr.roughness, u_pbr.lightColor
    );

    // ── 4. 环境光 IBL (简化球谐近似) ──────────────────────────────────────────
    let ambientLit = pbrAmbientSimple(
        u_pbr.albedo, N, V, f0,
        u_pbr.metallic, u_pbr.roughness, u_pbr.ao,
        u_pbr.envColor
    );

    var pbrColor = directLit + ambientLit;

    // ── 5. Fresnel 边缘光 ──────────────────────────────────────────────────────
    var rim = fresnelRim(N, V, u_pbr.fresnelPower, u_pbr.fresnelColor);
    // 粗糙表面边缘光散射更宽, 衰减峰值
    rim *= mix(1.0, 0.3, u_pbr.roughness);
    pbrColor += rim;

    // ── 6. 薄膜干涉彩虹 ───────────────────────────────────────────────────────
    let cosIncidence = saturate_f(dot(N, V));

    // 膜厚随时间轻微振荡 (模拟细胞膜形变, ±50 nm)
    let thickAnim = u_pbr.iridThickness
                  + sin(u_pbr.time * 0.4 + in.vUV.x * PI) * 50.0;
    let iridBase  = iridescence(u_pbr.iridThickness, u_pbr.iridIOR, cosIncidence);
    let iridAnim  = iridescence(thickAnim,            u_pbr.iridIOR, cosIncidence);
    let iridColor = mix(iridBase, iridAnim, 0.35);

    pbrColor = mix(pbrColor, pbrColor + iridColor, u_pbr.iridStrength);

    // ── 7. 简化大气雾 (Beer-Lambert 透射率) ──────────────────────────────────
    let transmittance = exp(-u_pbr.atmoDepth * u_pbr.atmoDensity);
    let inscatter     = u_pbr.atmoFogColor * (1.0 - transmittance)
                                           * saturate_f(u_pbr.atmoDepth);
    pbrColor = pbrColor * transmittance + inscatter;

    // ── 8. Reinhard 色调映射 ──────────────────────────────────────────────────
    pbrColor = pbrColor / (pbrColor + vec3f(1.0));

    // ── 9. sRGB gamma 校正 ─────────────────────────────────────────────────────
    pbrColor = pow(pbrColor, vec3f(1.0 / 2.2));

    // ── 10. 软边 vignette alpha ───────────────────────────────────────────────
    let edgeDist = 2.0 * length(in.vUV - vec2f(0.5));
    let alpha    = 1.0 - smoothstep(0.80, 1.0, edgeDist);

    // ── 11. 预乘 alpha 输出 ───────────────────────────────────────────────────
    return vec4f(pbrColor * alpha, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — 通用全屏四边形顶点着色器
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_FULLSCREEN_VS = /* wgsl */`
struct FSOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
}
@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FSOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32( vi         & 2u) * 2.0 - 1.0;
    var o: FSOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript 接口 & 参数类型
// ─────────────────────────────────────────────────────────────────────────────

/** PBR 全量材质参数 */
export interface PBRParams {
  /** 基础色 [R,G,B] 线性 sRGB */
  albedo        : [number, number, number];
  /** 金属度 [0=绝缘, 1=金属] */
  metallic      : number;
  /** 粗糙度 [0=镜面, 1=漫反射] */
  roughness     : number;
  /** 环境光遮蔽 [0,1] */
  ao            : number;
  /** 世界空间点光源位置 */
  lightPos      : [number, number, number];
  /** HDR 光源颜色/强度 */
  lightColor    : [number, number, number];
  /** 世界空间相机位置 */
  cameraPos     : [number, number, number];
  /** Fresnel 边缘光指数 (默认 4.0) */
  fresnelPower  : number;
  /** Fresnel 边缘色调 */
  fresnelColor  : [number, number, number];
  /** 薄膜厚度 nm (默认 500) */
  iridThickness : number;
  /** 薄膜折射率 (默认 1.45) */
  iridIOR       : number;
  /** 彩虹混合权重 [0,1] */
  iridStrength  : number;
  /** 大气雾密度 */
  atmoDensity   : number;
  /** 格深度归一化 [0,1] */
  atmoDepth     : number;
  /** 大气雾色 */
  atmoFogColor  : [number, number, number];
  /** 环境光颜色 */
  envColor      : [number, number, number];
  /** 动画时钟 (秒) */
  time          : number;
}

/** Matcap+Fresnel 轻量材质参数 */
export interface MatcapParams {
  /** Fresnel rim 指数 (默认 3.0) */
  fresnelPower  : number;
  /** Fresnel rim 色调 */
  fresnelColor  : [number, number, number];
  /** 法线噪声空间频率 (默认 1.4) */
  noiseScale    : number;
  /** 法线噪声幅度 [0, 0.5] */
  noiseStrength : number;
  /** 物种颜色 */
  species       : [number, number, number];
  /** 物种色调权重 [0,1] */
  tintStrength  : number;
  /** 动画时钟 */
  time          : number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 默认参数
// ─────────────────────────────────────────────────────────────────────────────

export const DEFAULT_PBR_PARAMS: PBRParams = {
  albedo        : [0.4, 0.75, 1.0],
  metallic      : 0.05,
  roughness     : 0.35,
  ao            : 1.0,
  lightPos      : [5.0, 8.0, 5.0],
  lightColor    : [2.5, 2.3, 2.0],
  cameraPos     : [0.0, 0.0, 10.0],
  fresnelPower  : 4.0,
  fresnelColor  : [0.4, 0.85, 1.0],
  iridThickness : 500.0,
  iridIOR       : 1.45,
  iridStrength  : 0.35,
  atmoDensity   : 0.5,
  atmoDepth     : 0.0,
  atmoFogColor  : [0.06, 0.12, 0.28],
  envColor      : [0.08, 0.10, 0.15],
  time          : 0.0,
};

export const DEFAULT_MATCAP_PARAMS: MatcapParams = {
  fresnelPower  : 3.0,
  fresnelColor  : [0.5, 0.9, 1.0],
  noiseScale    : 1.4,
  noiseStrength : 0.18,
  species       : [0.4, 0.7, 1.0],
  tintStrength  : 0.85,
  time          : 0.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// PBR Uniform buffer 打包 (std140/wgsl 16-byte 对齐)
// ─────────────────────────────────────────────────────────────────────────────

/** 打包 PBRParams → Float32Array (std140 对齐, 每个 vec3 后填 pad) */
export function packPBRUniforms(p: PBRParams): Float32Array {
  // 布局 (每行 4× f32 = 16 bytes):
  //   [0]  albedo.xyz, metallic
  //   [4]  roughness, ao, _p0, _p1
  //   [8]  lightPos.xyz, _p2
  //   [12] lightColor.xyz, _p3
  //   [16] cameraPos.xyz, fresnelPower
  //   [20] fresnelColor.xyz, iridThickness
  //   [24] iridIOR, iridStrength, atmoDensity, atmoDepth
  //   [28] atmoFogColor.xyz, _p4
  //   [32] envColor.xyz, time
  //   [36] _p5, _p6, _p7, _p8          ← pad to 40 f32 (160 bytes)
  const buf = new Float32Array(40);
  buf[0]  = p.albedo[0];       buf[1]  = p.albedo[1];       buf[2]  = p.albedo[2];       buf[3]  = p.metallic;
  buf[4]  = p.roughness;       buf[5]  = p.ao;              buf[6]  = 0;                 buf[7]  = 0;
  buf[8]  = p.lightPos[0];     buf[9]  = p.lightPos[1];     buf[10] = p.lightPos[2];     buf[11] = 0;
  buf[12] = p.lightColor[0];   buf[13] = p.lightColor[1];   buf[14] = p.lightColor[2];   buf[15] = 0;
  buf[16] = p.cameraPos[0];    buf[17] = p.cameraPos[1];    buf[18] = p.cameraPos[2];    buf[19] = p.fresnelPower;
  buf[20] = p.fresnelColor[0]; buf[21] = p.fresnelColor[1]; buf[22] = p.fresnelColor[2]; buf[23] = p.iridThickness;
  buf[24] = p.iridIOR;         buf[25] = p.iridStrength;    buf[26] = p.atmoDensity;     buf[27] = p.atmoDepth;
  buf[28] = p.atmoFogColor[0]; buf[29] = p.atmoFogColor[1]; buf[30] = p.atmoFogColor[2]; buf[31] = 0;
  buf[32] = p.envColor[0];     buf[33] = p.envColor[1];     buf[34] = p.envColor[2];     buf[35] = p.time;
  return buf;
}

/** 打包 MatcapParams → Float32Array */
export function packMatcapUniforms(p: MatcapParams): Float32Array {
  // 布局:
  //   [0]  fresnelPower, fresnelColor.xyz
  //   [4]  noiseScale, noiseStrength, _p0, _p1
  //   [8]  species.xyz, tintStrength
  //   [12] time, _p2, _p3, _p4
  const buf = new Float32Array(16);
  buf[0]  = p.fresnelPower;   buf[1]  = p.fresnelColor[0];  buf[2]  = p.fresnelColor[1];  buf[3]  = p.fresnelColor[2];
  buf[4]  = p.noiseScale;     buf[5]  = p.noiseStrength;    buf[6]  = 0;                   buf[7]  = 0;
  buf[8]  = p.species[0];     buf[9]  = p.species[1];       buf[10] = p.species[2];        buf[11] = p.tintStrength;
  buf[12] = p.time;           buf[13] = 0;                  buf[14] = 0;                   buf[15] = 0;
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL 模块组合器 — 拼接所有公共依赖 + 目标着色器
// ─────────────────────────────────────────────────────────────────────────────

function buildPBRShader(): string {
  return [
    WGSL_MATH_HELPERS,
    WGSL_PBR_BRDF,
    WGSL_FRESNEL,
    WGSL_IRIDESCENCE,
    WGSL_FULLSCREEN_VS,
    WGSL_PBR_FRAG,
  ].join('\n\n');
}

function buildMatcapShader(): string {
  return [
    WGSL_MATH_HELPERS,
    WGSL_FRESNEL,
    WGSL_MATCAP_FRESNEL_FRAG,
  ].join('\n\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// ATPBRMaterial — 全量 PBR 渲染管线
// ─────────────────────────────────────────────────────────────────────────────

export class ATPBRMaterial {
  private device        : GPUDevice;
  private pipeline      : GPURenderPipeline;
  private uniformBuf    : GPUBuffer;
  private bindGroup!    : GPUBindGroup;
  private bindGroupLayout: GPUBindGroupLayout;
  private params        : PBRParams;
  private format        : GPUTextureFormat;

  private constructor(
    device         : GPUDevice,
    pipeline       : GPURenderPipeline,
    bindGroupLayout: GPUBindGroupLayout,
    uniformBuf     : GPUBuffer,
    format         : GPUTextureFormat,
  ) {
    this.device         = device;
    this.pipeline       = pipeline;
    this.bindGroupLayout= bindGroupLayout;
    this.uniformBuf     = uniformBuf;
    this.format         = format;
    this.params         = { ...DEFAULT_PBR_PARAMS };
    this._rebuildBindGroup();
  }

  /** 工厂: 异步创建完整 PBR 管线 */
  static async create(
    device : GPUDevice,
    format : GPUTextureFormat = 'bgra8unorm',
  ): Promise<ATPBRMaterial> {
    const code = buildPBRShader();
    const mod  = device.createShaderModule({ label: 'at-pbr', code });

    const bgl = device.createBindGroupLayout({
      label: 'at-pbr-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createRenderPipeline({
      label : 'at-pbr-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex  : { module: mod, entryPoint: 'vs_fullscreen' },
      fragment: {
        module    : mod,
        entryPoint: 'fs_pbr',
        targets   : [{
          format,
          blend: {
            // 预乘 alpha 混合
            color: { srcFactor: 'one',           dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',           dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const uniformBuf = device.createBuffer({
      label: 'at-pbr-uniforms',
      size : 160,  // 40 f32 × 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new ATPBRMaterial(device, pipeline, bgl, uniformBuf, format);
  }

  /** 更新材质参数 (部分覆盖) */
  setParams(partial: Partial<PBRParams>): void {
    Object.assign(this.params, partial);
    this.device.queue.writeBuffer(
      this.uniformBuf, 0, packPBRUniforms(this.params)
    );
  }

  /** 每帧推进时间 */
  tick(dt: number): void {
    this.params.time += dt;
    this.setParams({});
  }

  /** 渲染到目标视图 */
  render(
    encoder    : GPUCommandEncoder,
    colorTarget: GPUTextureView,
    depthTarget?: GPUTextureView,
  ): void {
    const pass = encoder.beginRenderPass({
      label          : 'at-pbr-pass',
      colorAttachments: [{
        view      : colorTarget,
        loadOp    : 'load',
        storeOp   : 'store',
      }],
      ...(depthTarget && {
        depthStencilAttachment: {
          view            : depthTarget,
          depthLoadOp     : 'load',
          depthStoreOp    : 'store',
          stencilLoadOp   : 'load',
          stencilStoreOp  : 'store',
        },
      }),
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);  // 全屏三角形
    pass.end();
  }

  /** WGSL 源码 (调试用) */
  get wgslSource(): string { return buildPBRShader(); }

  private _rebuildBindGroup(): void {
    this.bindGroup = this.device.createBindGroup({
      label  : 'at-pbr-bg',
      layout : this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });
  }

  destroy(): void {
    this.uniformBuf.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATMatcapFresnel — 轻量 Matcap+Fresnel 材质 (~10× 快于全量 PBR)
// ─────────────────────────────────────────────────────────────────────────────

export class ATMatcapFresnel {
  private device          : GPUDevice;
  private pipeline        : GPURenderPipeline;
  private uniformBuf      : GPUBuffer;
  private bindGroup!      : GPUBindGroup;
  private bindGroupLayout : GPUBindGroupLayout;
  private params          : MatcapParams;

  private constructor(
    device          : GPUDevice,
    pipeline        : GPURenderPipeline,
    bindGroupLayout : GPUBindGroupLayout,
    uniformBuf      : GPUBuffer,
    matcapTex       : GPUTexture,
    sampler         : GPUSampler,
  ) {
    this.device          = device;
    this.pipeline        = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.uniformBuf      = uniformBuf;
    this.params          = { ...DEFAULT_MATCAP_PARAMS };
    this._rebuildBindGroup(matcapTex, sampler);
  }

  /**
   * 工厂: 创建 Matcap+Fresnel 管线
   * @param matcapImageBitmap — 256×256 matcap 球形贴图 (ImageBitmap)
   */
  static async create(
    device          : GPUDevice,
    format          : GPUTextureFormat = 'bgra8unorm',
    matcapImageBitmap?: ImageBitmap,
  ): Promise<ATMatcapFresnel> {
    const code = buildMatcapShader();
    const mod  = device.createShaderModule({ label: 'at-matcap', code });

    const bgl = device.createBindGroupLayout({
      label  : 'at-matcap-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer : { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const pipeline = device.createRenderPipeline({
      label : 'at-matcap-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex  : { module: mod, entryPoint: 'vs_fullscreen' },
      fragment: {
        module    : mod,
        entryPoint: 'fs_matcap',
        targets   : [{
          format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const uniformBuf = device.createBuffer({
      label: 'at-matcap-uniforms',
      size : 64,   // 16 f32 × 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 如果没有提供 matcap 贴图, 创建一个 1×1 占位白贴图
    let matcapTex: GPUTexture;
    if (matcapImageBitmap) {
      matcapTex = device.createTexture({
        label : 'matcap-tex',
        size  : [matcapImageBitmap.width, matcapImageBitmap.height],
        format: 'rgba8unorm',
        usage : GPUTextureUsage.TEXTURE_BINDING
              | GPUTextureUsage.COPY_DST
              | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture(
        { source: matcapImageBitmap },
        { texture: matcapTex },
        [matcapImageBitmap.width, matcapImageBitmap.height],
      );
    } else {
      matcapTex = device.createTexture({
        label : 'matcap-placeholder',
        size  : [1, 1],
        format: 'rgba8unorm',
        usage : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: matcapTex },
        new Uint8Array([200, 210, 230, 255]),
        { bytesPerRow: 4 },
        [1, 1],
      );
    }

    const sampler = device.createSampler({
      label     : 'matcap-sampler',
      magFilter : 'linear',
      minFilter : 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    return new ATMatcapFresnel(device, pipeline, bgl, uniformBuf, matcapTex, sampler);
  }

  setParams(partial: Partial<MatcapParams>): void {
    Object.assign(this.params, partial);
    this.device.queue.writeBuffer(
      this.uniformBuf, 0, packMatcapUniforms(this.params)
    );
  }

  tick(dt: number): void {
    this.params.time += dt;
    this.setParams({});
  }

  render(
    encoder    : GPUCommandEncoder,
    colorTarget: GPUTextureView,
  ): void {
    const pass = encoder.beginRenderPass({
      label           : 'at-matcap-pass',
      colorAttachments: [{
        view   : colorTarget,
        loadOp : 'load',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  get wgslSource(): string { return buildMatcapShader(); }

  private _rebuildBindGroup(tex: GPUTexture, sampler: GPUSampler): void {
    this.bindGroup = this.device.createBindGroup({
      label  : 'at-matcap-bg',
      layout : this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: tex.createView() },
        { binding: 2, resource: sampler },
      ],
    });
  }

  destroy(): void {
    this.uniformBuf.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 导出 WGSL 片段 — 供其他材质模块组合使用
// ─────────────────────────────────────────────────────────────────────────────

export const AT_PBR_WGSL = {
  /** 数学助手 (saturate / pow5 / PI 常数) */
  mathHelpers  : WGSL_MATH_HELPERS,
  /** Cook-Torrance BRDF (F_Schlick, D_GGX, G_SmithGGX, pbrDirect, pbrAmbientSimple) */
  pbrBRDF      : WGSL_PBR_BRDF,
  /** Schlick Fresnel 边缘光 (fresnel_f, fresnelRim) */
  fresnel      : WGSL_FRESNEL,
  /** 薄膜干涉彩虹 (iridescence) */
  iridescence  : WGSL_IRIDESCENCE,
  /** Matcap + Fresnel 片元着色器 (含 Simplex 噪声扰动) */
  matcapFrag   : WGSL_MATCAP_FRESNEL_FRAG,
  /** 全量 PBR 片元着色器 */
  pbrFrag      : WGSL_PBR_FRAG,
  /** 全屏三角形顶点着色器 */
  fullscreenVS : WGSL_FULLSCREEN_VS,
};
