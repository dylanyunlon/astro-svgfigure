/**
 * at-mousefluid-import.ts — M815: AT Mousefluid Interactive Fluid — Direct Import
 * ─────────────────────────────────────────────────────────────────────────────
 * 鼠标是笔，流体是墨。
 *
 * 直接移植 ActiveTheory 的完整 mousefluid 交互流体系统——用户鼠标/触摸驱动
 * 流体扰动。splat + advect + divergence + pressure + curl 全链路。
 *
 *   AT 原始数据 (upstream/activetheory-assets/compiled.vs)：
 *     fluidBase.vs                  — 6664 行  流体 pass 基础顶点着色器
 *     splatShader.fs                — 6720 行  线段 splat (AT 特色: 点→线段距离)
 *     advectionShader.fs            — 6600 行  半拉格朗日对流 (linear filter)
 *     advectionManualFilteringShader.fs — 6579 行  手动双线性对流 (fallback)
 *     curlShader.fs                 — 6627 行  涡度计算 (curl ω)
 *     vorticityShader.fs            — 6759 行  涡度约束 (vorticity confinement)
 *     divergenceShader.fs           — 6646 行  散度计算 (div v)
 *     pressureShader.fs             — 6698 行  Jacobi 压力求解
 *     gradientSubtractShader.fs     — 6678 行  压力梯度减法 (投影步)
 *     clearShader.fs                — 6619 行  压力衰减清除
 *     mousefluid.fs                 — 7888 行  下游消费: getFluidVelocity()
 *
 *   参考源码：
 *     upstream/ogl/examples/post-fluid-distortion.html — Pavel Dobryakov 移植
 *     upstream/lygia/simulate/simpleAndFastFluid.glsl  — LYGIA 流体参考
 *     src/lib/sph/at-navier-stokes.ts    — M715 WebGPU 流体计算 (WGSL)
 *     src/lib/sph/interactive-fluid.ts   — M743 交互控制器
 *     src/lib/sph/ogl-flowmap-bridge.ts  — M614 OGL Flowmap 桥接
 *     src/lib/sph/flowmap-bridge.ts      — M573 SPH→Flowmap 桥接
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * AT mousefluid vs OGL/Pavel Dobryakov 的关键差异
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   1. **线段 Splat** — AT splatShader.fs 使用 point-to-line-segment 距离
 *      (prevPoint → point)，而非 OGL 的简单 Gaussian 点 splat。这让高速鼠标
 *      移动时轨迹连续，不出现断裂。AT 还使用 cubicOut 缓动函数和 screen blend
 *      模式。
 *
 *   2. **Boundary 处理** — AT divergenceShader.fs 使用边界反射
 *      (if vL.x < 0.0, L = -C.x)，而 gradientSubtract 和 pressure 使用
 *      boundary() 函数（当前透传，支持 clamp）。
 *
 *   3. **mousefluid.fs 消费层** — AT 提供 getFluidVelocity() 和
 *      getFluidVelocityMask() 工具函数，通过 tFluid + tFluidMask 纹理
 *      将流体速度场输出到下游着色器（粒子、表面扭曲等）。
 *
 *   4. **Splat screen blend** — AT 的 splatShader 使用
 *      blendScreen(base, splat) 混合模式（1-(1-a)(1-b)），配合 uAdd
 *      在 screen 和 additive 之间插值。OGL 版本只用纯加法。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 管线结构（每帧 step()）
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass 0: SPLAT ─────────────────────────────────────────────────────────┐
 *   │  对每个待处理的 pointer splat:                                           │
 *   │    ① velocity splat: 线段距离 × cubicOut × (dx, dy, 1)                  │
 *   │       → velocity ping-pong write, swap                                  │
 *   │    ② dye splat: 同一线段距离 × cubicOut × hue colour                    │
 *   │       → density ping-pong write, swap                                   │
 *   │  AT splatShader.fs: prevPoint→point 线段, screen blend, canRender 开关  │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 1: CURL ──────────────────────────────────────────────────────────┐
 *   │  从速度场计算涡度 ω = ∂vy/∂x − ∂vx/∂y                                 │
 *   │  AT curlShader.fs: 四邻域有限差分, 0.5 * (R-L-T+B)                      │
 *   │  → curl FBO (r16float, single)                                          │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 2: VORTICITY CONFINEMENT ─────────────────────────────────────────┐
 *   │  从 curl 纹理计算涡度梯度方向，施加涡度约束力                              │
 *   │  AT vorticityShader.fs:                                                 │
 *   │    force = 0.5 * (|T|-|B|, |R|-|L|)                                    │
 *   │    force = normalize(force) * curl_strength * C                         │
 *   │    force.y *= -1.0 (AT Y-flip 约定)                                    │
 *   │    vel += force * dt                                                    │
 *   │  → velocity ping-pong write, swap                                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 3: DIVERGENCE ────────────────────────────────────────────────────┐
 *   │  计算速度场散度 div = 0.5 * (R-L + T-B)                                 │
 *   │  AT divergenceShader.fs: 边界反射 (if vL.x<0, L=-C.x)                  │
 *   │  → divergence FBO (r16float, single)                                    │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 4: PRESSURE CLEAR ────────────────────────────────────────────────┐
 *   │  衰减上一帧压力场: p *= pressureDissipation                              │
 *   │  AT clearShader.fs: gl_FragColor = value * texture2D(uTexture, vUv)     │
 *   │  → pressure ping-pong write, swap                                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 5: PRESSURE SOLVE (Jacobi × N) ──────────────────────────────────┐
 *   │  求解 Poisson 方程 ∇²p = div, N 次 Jacobi 迭代                         │
 *   │  AT pressureShader.fs: p = (L+R+B+T - div) * 0.25                      │
 *   │  带 boundary() 函数 (当前透传, 可启用 clamp)                             │
 *   │  → pressure ping-pong read↔write, swap each iteration                   │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 6: GRADIENT SUBTRACT ─────────────────────────────────────────────┐
 *   │  从速度场减去压力梯度: v -= ∇p → 无散速度场                              │
 *   │  AT gradientSubtractShader.fs: vel -= (R-L, T-B)                        │
 *   │  → velocity ping-pong write, swap                                       │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─ Pass 7: ADVECTION (× 2) ──────────────────────────────────────────────┐
 *   │  7a: 速度自对流 — vel[coord - dt * vel * texelSize] * dissipation       │
 *   │  7b: 染料对流   — dye[coord - dt * vel * texelSize] * dissipation       │
 *   │  AT advectionShader.fs / advectionManualFilteringShader.fs (fallback)    │
 *   │  → velocity write/swap, density write/swap                              │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   输出: density.read (染料 RGB+密度), velocity.read (速度场 XY)
 *   下游消费: mousefluid.fs → getFluidVelocity() / getFluidVelocityMask()
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 纹理布局 (WebGPU)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   velTex[2]   rgba16float   simRes × simRes    XY=velocity, ZW=unused
 *   dyeTex[2]   rgba16float   dyeRes × dyeRes    RGB=dye colour, A=density
 *   divTex      r16float      simRes × simRes    散度标量
 *   preTex[2]   r16float      simRes × simRes    压力 (Jacobi ping-pong)
 *   curlTex     r16float      simRes × simRes    涡度标量
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 用法
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   import { ATMouseFluid } from '$lib/sph/at-mousefluid-import';
 *
 *   const fluid = ATMouseFluid.create(device, canvas, {
 *     simRes:          128,
 *     dyeRes:          512,
 *     pressureIters:   4,
 *     curlStrength:    20,
 *     splatRadius:     0.2,
 *     velocityDissipation: 0.98,
 *     densityDissipation:  0.97,
 *     pressureDissipation: 0.8,
 *   });
 *
 *   // rAF loop:
 *   fluid.step();  // flush pointer queue → sim → advect
 *   const dyeView = fluid.getDyeTextureView();    // tFluid
 *   const velView = fluid.getVelocityTextureView();
 *   // bind to downstream render pass as tFluid / tFluidMask ...
 *
 *   fluid.destroy();
 *
 * Research: xiaodi #M815 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// GLSL Sources — direct from AT compiled.vs (line numbers in header)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT fluidBase.vs (compiled.vs:6664)
 * 基础顶点着色器 — 计算四邻域 UV 偏移供所有 fluid pass 使用。
 */
export const AT_FLUID_BASE_VS = /* glsl */ `
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;

void main () {
    vUv = uv;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(position, 1.0);
}
`;

/**
 * AT splatShader.fs (compiled.vs:6720)
 * AT 特色线段 splat — prevPoint→point 点到线段距离 + cubicOut + screen blend。
 * 比 OGL/Pavel Dobryakov 的 Gaussian 点 splat 更精细。
 */
export const AT_SPLAT_FS = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec3 bgColor;
uniform vec2 point;
uniform vec2 prevPoint;
uniform float radius;
uniform float canRender;
uniform float uAdd;

float blendScreen(float base, float blend) {
    return 1.0-((1.0-base)*(1.0-blend));
}

vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(blendScreen(base.r, blend.r), blendScreen(base.g, blend.g), blendScreen(base.b, blend.b));
}

float l(vec2 uv, vec2 point1, vec2 point2) {
    vec2 pa = uv - point1, ba = point2 - point1;
    pa.x *= aspectRatio;
    ba.x *= aspectRatio;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float cubicOut(float t) {
    float f = t - 1.0;
    return f * f * f + 1.0;
}

void main () {
    vec3 splat = (1.0 - cubicOut(clamp(l(vUv, prevPoint.xy, point.xy) / radius, 0.0, 1.0))) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    base *= canRender;

    vec3 outColor = mix(blendScreen(base, splat), base + splat, uAdd);
    gl_FragColor = vec4(outColor, 1.0);
}
`;

/**
 * AT advectionShader.fs (compiled.vs:6600)
 * 半拉格朗日对流 — 线性纹理过滤版本。
 */
export const AT_ADVECTION_FS = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    gl_FragColor = dissipation * texture2D(uSource, coord);
    gl_FragColor.a = 1.0;
}
`;

/**
 * AT advectionManualFilteringShader.fs (compiled.vs:6579)
 * 手动双线性插值对流 — 不支持 linear filtering 时的 fallback。
 */
export const AT_ADVECTION_MANUAL_FS = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
void main () {
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
    gl_FragColor.a = 1.0;
}
`;

/**
 * AT curlShader.fs (compiled.vs:6627)
 * 涡度计算 — 2D curl ω = ∂vy/∂x − ∂vx/∂y。
 */
export const AT_CURL_FS = /* glsl */ `
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}
`;

/**
 * AT vorticityShader.fs (compiled.vs:6759)
 * 涡度约束 — 从 curl 梯度方向施加反扩散力。
 * 注意 force.y *= -1.0 是 AT 的 Y-flip 约定。
 */
export const AT_VORTICITY_FS = /* glsl */ `
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 vel = texture2D(uVelocity, vUv).xy;
    gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
}
`;

/**
 * AT divergenceShader.fs (compiled.vs:6646)
 * 散度计算 — 带边界反射的有限差分。
 */
export const AT_DIVERGENCE_FS = /* glsl */ `
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
   if (vL.x < 0.0) { L = -C.x; }
   if (vR.x > 1.0) { R = -C.x; }
   if (vT.y > 1.0) { T = -C.y; }
   if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

/**
 * AT pressureShader.fs (compiled.vs:6698)
 * Jacobi 压力求解 — p = (L+R+B+T - div) * 0.25。
 * 带 boundary() 函数预留 clamp 支持。
 */
export const AT_PRESSURE_FS = /* glsl */ `
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
vec2 boundary (vec2 uv) {
    return uv;
    // uncomment if you use wrap or repeat texture mode
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    float C = texture2D(uPressure, vUv).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}
`;

/**
 * AT gradientSubtractShader.fs (compiled.vs:6678)
 * 压力梯度减法 — v -= ∇p → 无散速度场。
 * 带 boundary() 函数预留 clamp 支持。
 */
export const AT_GRADIENT_SUBTRACT_FS = /* glsl */ `
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
vec2 boundary (vec2 uv) {
    return uv;
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}
`;

/**
 * AT clearShader.fs (compiled.vs:6619)
 * 压力衰减清除 — gl_FragColor = value * texture2D(uTexture, vUv)。
 */
export const AT_CLEAR_FS = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}
`;

/**
 * AT mousefluid.fs (compiled.vs:7888)
 * 下游消费层 — 从 tFluid/tFluidMask 提取流体速度供其他着色器使用。
 */
export const AT_MOUSEFLUID_CONSUME_FS = /* glsl */ `
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

vec2 getFluidVelocity() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return texture2D(tFluid, vUv).xy * fluidMask;
}

vec3 getFluidVelocityMask() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return vec3(texture2D(tFluid, vUv).xy * fluidMask, fluidMask);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL Compute Shaders — AT pipeline ported to WebGPU
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 共享 uniforms 结构体 — 所有 compute pass 通过同一 uniform buffer 读参数。
 *
 * 内存布局 (std140-ish, 16-byte aligned):
 *   offset  0: texelSize (vec2) + dt (f32) + dissipation (f32)      = 16 bytes
 *   offset 16: point (vec2) + prevPoint (vec2)                      = 16 bytes
 *   offset 32: color (vec3) + radius (f32)                          = 16 bytes
 *   offset 48: curlStrength (f32) + pressureDissipation (f32)
 *              + aspectRatio (f32) + canRender (f32)                 = 16 bytes
 *   offset 64: uAdd (f32) + simRes (f32) + dyeRes (f32) + pad      = 16 bytes
 *                                                          Total     = 80 bytes
 */
export const FLUID_UNIFORM_SIZE = 80;

/** Byte offsets into the uniform buffer. */
export const FluidUniformOffset = {
  texelSize:            0,   // vec2  (8 bytes, padded to 16 with dt+diss)
  dt:                   8,   // f32
  dissipation:         12,   // f32
  point:               16,   // vec2
  prevPoint:           24,   // vec2
  color:               32,   // vec3 (12 bytes, padded to 16 with radius)
  radius:              44,   // f32
  curlStrength:        48,   // f32
  pressureDissipation: 52,   // f32
  aspectRatio:         56,   // f32
  canRender:           60,   // f32
  uAdd:                64,   // f32
  simRes:              68,   // f32
  dyeRes:              72,   // f32
  _pad:                76,   // f32
} as const;

/** Workgroup size for compute passes — 16×16 = 256 threads. */
const WG = 16;

/**
 * WGSL: AT splatShader.fs port — line-segment splat with cubicOut + screen blend.
 *
 * AT splatShader.fs 的核心特色:
 *   - l(): 点到线段 (prevPoint→point) 的距离，aspect ratio 校正
 *   - cubicOut(): 三次缓出缓动
 *   - blendScreen(): 屏幕混合模式
 *   - uAdd: 在 screen blend (0.0) 和 additive blend (1.0) 之间插值
 *   - canRender: 整体开关
 */
export const WGSL_SPLAT = /* wgsl */ `
struct Params {
  texelSize: vec2f,
  dt: f32,
  dissipation: f32,
  point: vec2f,
  prevPoint: vec2f,
  color: vec3f,
  radius: f32,
  curlStrength: f32,
  pressureDissipation: f32,
  aspectRatio: f32,
  canRender: f32,
  uAdd: f32,
  simRes: f32,
  dyeRes: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var targetIn:  texture_2d<f32>;
@group(0) @binding(2) var targetOut: texture_storage_2d<rgba16float, write>;

// AT l() — point-to-line-segment distance with aspect correction
fn lineSegDist(uv: vec2f, p1: vec2f, p2: vec2f) -> f32 {
    var pa = uv - p1;
    var ba = p2 - p1;
    pa.x *= p.aspectRatio;
    ba.x *= p.aspectRatio;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-10), 0.0, 1.0);
    return length(pa - ba * h);
}

// AT cubicOut()
fn cubicOut(t: f32) -> f32 {
    let f = t - 1.0;
    return f * f * f + 1.0;
}

// AT blendScreen()
fn blendScreen1(base: f32, blend: f32) -> f32 {
    return 1.0 - ((1.0 - base) * (1.0 - blend));
}
fn blendScreen3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(
        blendScreen1(base.x, blend.x),
        blendScreen1(base.y, blend.y),
        blendScreen1(base.z, blend.z),
    );
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(targetIn);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let dist = lineSegDist(uv, p.prevPoint, p.point);
    let splatVal = (1.0 - cubicOut(clamp(dist / p.radius, 0.0, 1.0))) * p.color;
    var base = textureLoad(targetIn, gid.xy, 0).xyz;
    base *= p.canRender;

    let outColor = mix(blendScreen3(base, splatVal), base + splatVal, vec3f(p.uAdd));
    textureStore(targetOut, gid.xy, vec4f(outColor, 1.0));
}
`;

/**
 * WGSL: AT advectionShader.fs port — semi-Lagrangian advection.
 * back-trace: coord = uv - dt * vel(uv) * texelSize, then sample source.
 *
 * Uses bilinear fetch (manual, since compute shaders can't use samplers
 * on storage textures). Matches AT advectionManualFilteringShader.fs.
 */
export const WGSL_ADVECT = /* wgsl */ `
struct Params {
  texelSize: vec2f,
  dt: f32,
  dissipation: f32,
  point: vec2f,
  prevPoint: vec2f,
  color: vec3f,
  radius: f32,
  curlStrength: f32,
  pressureDissipation: f32,
  aspectRatio: f32,
  canRender: f32,
  uAdd: f32,
  simRes: f32,
  dyeRes: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var velIn:    texture_2d<f32>;
@group(0) @binding(2) var srcIn:    texture_2d<f32>;
@group(0) @binding(3) var dst:      texture_storage_2d<rgba16float, write>;

// Manual bilinear interpolation (matches AT advectionManualFilteringShader.fs bilerp)
fn bilerp(tex: texture_2d<f32>, uv: vec2f, texSize: vec2f) -> vec4f {
    let st = uv * texSize - 0.5;
    let iuv = floor(st);
    let fuv = fract(st);
    let dims = vec2i(textureDimensions(tex));
    let c00 = vec2i(clamp(vec2i(iuv), vec2i(0), dims - 1));
    let c10 = vec2i(clamp(vec2i(iuv) + vec2i(1, 0), vec2i(0), dims - 1));
    let c01 = vec2i(clamp(vec2i(iuv) + vec2i(0, 1), vec2i(0), dims - 1));
    let c11 = vec2i(clamp(vec2i(iuv) + vec2i(1, 1), vec2i(0), dims - 1));
    let a = textureLoad(tex, c00, 0);
    let b = textureLoad(tex, c10, 0);
    let c = textureLoad(tex, c01, 0);
    let d = textureLoad(tex, c11, 0);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let velDims = textureDimensions(velIn);
    let srcDims = textureDimensions(srcIn);
    let dstDims = textureDimensions(dst);
    if (gid.x >= dstDims.x || gid.y >= dstDims.y) { return; }

    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dstDims);
    let vel = bilerp(velIn, uv, vec2f(velDims)).xy;
    let coord = uv - p.dt * vel * p.texelSize;
    var result = p.dissipation * bilerp(srcIn, coord, vec2f(srcDims));
    result.w = 1.0;
    textureStore(dst, gid.xy, result);
}
`;

/**
 * WGSL: AT curlShader.fs port — 2D vorticity ω = ∂vy/∂x − ∂vx/∂y.
 * 四邻域有限差分: vorticity = 0.5 * (R - L - T + B)。
 */
export const WGSL_CURL = /* wgsl */ `
@group(0) @binding(0) var velIn: texture_2d<f32>;
@group(0) @binding(1) var curlOut: texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(velIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let L = textureLoad(velIn, clamp(id + vec2i(-1, 0), vec2i(0), dims - 1), 0).y;
    let R = textureLoad(velIn, clamp(id + vec2i( 1, 0), vec2i(0), dims - 1), 0).y;
    let T = textureLoad(velIn, clamp(id + vec2i( 0, 1), vec2i(0), dims - 1), 0).x;
    let B = textureLoad(velIn, clamp(id + vec2i( 0,-1), vec2i(0), dims - 1), 0).x;

    let vorticity = 0.5 * (R - L - T + B);
    textureStore(curlOut, id, vec4f(vorticity, 0.0, 0.0, 1.0));
}
`;

/**
 * WGSL: AT vorticityShader.fs port — vorticity confinement.
 * 从 curl 梯度方向施加反扩散力，force.y *= -1.0 (AT Y-flip)。
 */
export const WGSL_VORTICITY = /* wgsl */ `
struct Params {
  texelSize: vec2f,
  dt: f32,
  dissipation: f32,
  point: vec2f,
  prevPoint: vec2f,
  color: vec3f,
  radius: f32,
  curlStrength: f32,
  pressureDissipation: f32,
  aspectRatio: f32,
  canRender: f32,
  uAdd: f32,
  simRes: f32,
  dyeRes: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var velIn:  texture_2d<f32>;
@group(0) @binding(2) var curlIn: texture_2d<f32>;
@group(0) @binding(3) var velOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(velIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let cL = textureLoad(curlIn, clamp(id + vec2i(-1, 0), vec2i(0), dims - 1), 0).x;
    let cR = textureLoad(curlIn, clamp(id + vec2i( 1, 0), vec2i(0), dims - 1), 0).x;
    let cT = textureLoad(curlIn, clamp(id + vec2i( 0, 1), vec2i(0), dims - 1), 0).x;
    let cB = textureLoad(curlIn, clamp(id + vec2i( 0,-1), vec2i(0), dims - 1), 0).x;
    let cC = textureLoad(curlIn, id, 0).x;

    var force = 0.5 * vec2f(abs(cT) - abs(cB), abs(cR) - abs(cL));
    force /= length(force) + 0.0001;
    force *= p.curlStrength * cC;
    force.y *= -1.0;  // AT Y-flip convention

    let vel = textureLoad(velIn, id, 0).xy;
    textureStore(velOut, id, vec4f(vel + force * p.dt, 0.0, 1.0));
}
`;

/**
 * WGSL: AT divergenceShader.fs port — velocity divergence with boundary reflection.
 * div = 0.5 * (R - L + T - B), 边界时反射: if L out, L = -C.x。
 */
export const WGSL_DIVERGENCE = /* wgsl */ `
@group(0) @binding(0) var velIn: texture_2d<f32>;
@group(0) @binding(1) var divOut: texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(velIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let C = textureLoad(velIn, id, 0).xy;

    // AT boundary reflection: out-of-bounds neighbours reflect velocity
    var fL = textureLoad(velIn, clamp(id + vec2i(-1, 0), vec2i(0), dims - 1), 0).x;
    var fR = textureLoad(velIn, clamp(id + vec2i( 1, 0), vec2i(0), dims - 1), 0).x;
    var fT = textureLoad(velIn, clamp(id + vec2i( 0, 1), vec2i(0), dims - 1), 0).y;
    var fB = textureLoad(velIn, clamp(id + vec2i( 0,-1), vec2i(0), dims - 1), 0).y;

    if (id.x == 0)          { fL = -C.x; }
    if (id.x == dims.x - 1) { fR = -C.x; }
    if (id.y == dims.y - 1) { fT = -C.y; }
    if (id.y == 0)          { fB = -C.y; }

    let div = 0.5 * (fR - fL + fT - fB);
    textureStore(divOut, id, vec4f(div, 0.0, 0.0, 1.0));
}
`;

/**
 * WGSL: AT pressureShader.fs port — Jacobi iteration.
 * p = (L + R + B + T - divergence) * 0.25。
 * boundary() 函数当前透传（与 AT 一致）。
 */
export const WGSL_PRESSURE = /* wgsl */ `
@group(0) @binding(0) var preIn: texture_2d<f32>;
@group(0) @binding(1) var divIn: texture_2d<f32>;
@group(0) @binding(2) var preOut: texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(preIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pL = textureLoad(preIn, clamp(id + vec2i(-1, 0), vec2i(0), dims - 1), 0).x;
    let pR = textureLoad(preIn, clamp(id + vec2i( 1, 0), vec2i(0), dims - 1), 0).x;
    let pT = textureLoad(preIn, clamp(id + vec2i( 0, 1), vec2i(0), dims - 1), 0).x;
    let pB = textureLoad(preIn, clamp(id + vec2i( 0,-1), vec2i(0), dims - 1), 0).x;

    let div = textureLoad(divIn, id, 0).x;
    let pressure = (pL + pR + pB + pT - div) * 0.25;
    textureStore(preOut, id, vec4f(pressure, 0.0, 0.0, 1.0));
}
`;

/**
 * WGSL: AT gradientSubtractShader.fs port — projection step.
 * vel -= (R - L, T - B) 使速度场无散。
 */
export const WGSL_GRADIENT_SUBTRACT = /* wgsl */ `
@group(0) @binding(0) var preIn: texture_2d<f32>;
@group(0) @binding(1) var velIn: texture_2d<f32>;
@group(0) @binding(2) var velOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(preIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pL = textureLoad(preIn, clamp(id + vec2i(-1, 0), vec2i(0), dims - 1), 0).x;
    let pR = textureLoad(preIn, clamp(id + vec2i( 1, 0), vec2i(0), dims - 1), 0).x;
    let pT = textureLoad(preIn, clamp(id + vec2i( 0, 1), vec2i(0), dims - 1), 0).x;
    let pB = textureLoad(preIn, clamp(id + vec2i( 0,-1), vec2i(0), dims - 1), 0).x;

    var vel = textureLoad(velIn, id, 0).xy;
    vel -= vec2f(pR - pL, pT - pB);
    textureStore(velOut, id, vec4f(vel, 0.0, 1.0));
}
`;

/**
 * WGSL: AT clearShader.fs port — pressure dissipation.
 * output = value * input。
 */
export const WGSL_CLEAR = /* wgsl */ `
struct ClearParams {
  value: f32,
};

@group(0) @binding(0) var<uniform> cp: ClearParams;
@group(0) @binding(1) var texIn:  texture_2d<f32>;
@group(0) @binding(2) var texOut: texture_storage_2d<r16float, write>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(texIn));
    let id = vec2i(gid.xy);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let val = textureLoad(texIn, id, 0) * cp.value;
    textureStore(texOut, id, val);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** A pending splat from pointer input. */
export interface PointerSplat {
  /** Current pointer position, normalised UV [0,1]. */
  x: number;
  y: number;
  /** Previous pointer position, normalised UV [0,1]. */
  px: number;
  py: number;
  /** Velocity delta (pixels × scale factor). */
  dx: number;
  dy: number;
  /** Dye colour [R, G, B], 0–1. */
  r: number;
  g: number;
  b: number;
}

/** Configuration for ATMouseFluid. */
export interface ATMouseFluidConfig {
  /** Simulation grid resolution (square). Default 128. */
  simRes?: number;
  /** Dye texture resolution (square). Default 512. */
  dyeRes?: number;
  /** Jacobi pressure iterations per frame. Default 4 (AT uses 3–4). */
  pressureIters?: number;
  /** Vorticity confinement strength. Default 20 (AT curlStrength). */
  curlStrength?: number;
  /** Splat radius in UV space. Default 0.2 (AT splatShader.fs radius). */
  splatRadius?: number;
  /** Velocity dissipation per frame. Default 0.98. */
  velocityDissipation?: number;
  /** Dye density dissipation per frame. Default 0.97. */
  densityDissipation?: number;
  /** Pressure dissipation per frame. Default 0.8. */
  pressureDissipation?: number;
  /** Splat uAdd — 0 = screen blend (AT default), 1 = pure additive. */
  splatBlendMode?: number;
  /** Simulation dt in seconds. Default 1/60. */
  dt?: number;
}

/** Read-only view of current fluid state. */
export interface ATMouseFluidState {
  readonly dyeTextureView: GPUTextureView;
  readonly velocityTextureView: GPUTextureView;
  readonly simRes: number;
  readonly dyeRes: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ping-pong texture helper
// ─────────────────────────────────────────────────────────────────────────────

interface PingPong {
  texA: GPUTexture;
  texB: GPUTexture;
  viewA: GPUTextureView;
  viewB: GPUTextureView;
  /** Index of the current "read" texture (0=A, 1=B). */
  readIdx: 0 | 1;
}

function createPingPong(
  device: GPUDevice,
  width: number,
  height: number,
  format: GPUTextureFormat,
  label: string,
): PingPong {
  const desc: GPUTextureDescriptor = {
    size: { width, height },
    format,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  };
  const texA = device.createTexture({ ...desc, label: `${label}_A` });
  const texB = device.createTexture({ ...desc, label: `${label}_B` });
  return {
    texA,
    texB,
    viewA: texA.createView(),
    viewB: texB.createView(),
    readIdx: 0,
  };
}

function ppRead(pp: PingPong): GPUTexture {
  return pp.readIdx === 0 ? pp.texA : pp.texB;
}
function ppWrite(pp: PingPong): GPUTexture {
  return pp.readIdx === 0 ? pp.texB : pp.texA;
}
function ppReadView(pp: PingPong): GPUTextureView {
  return pp.readIdx === 0 ? pp.viewA : pp.viewB;
}
function ppWriteView(pp: PingPong): GPUTextureView {
  return pp.readIdx === 0 ? pp.viewB : pp.viewA;
}
function ppSwap(pp: PingPong): void {
  pp.readIdx = pp.readIdx === 0 ? 1 : 0;
}
function ppDestroy(pp: PingPong): void {
  pp.texA.destroy();
  pp.texB.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// HSL colour generator (golden-ratio hue sequence for multi-touch splats)
// ─────────────────────────────────────────────────────────────────────────────

const GOLDEN_ANGLE = 0.381966011250105; // 1 / φ²

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  const sector = Math.floor(h * 6) % 6;
  if (sector === 0)      { r = c; g = x; b = 0; }
  else if (sector === 1) { r = x; g = c; b = 0; }
  else if (sector === 2) { r = 0; g = c; b = x; }
  else if (sector === 3) { r = 0; g = x; b = c; }
  else if (sector === 4) { r = x; g = 0; b = c; }
  else                   { r = c; g = 0; b = x; }
  return [r + m, g + m, b + m];
}

function nextSplatColour(index: number): [number, number, number] {
  const hue = (index * GOLDEN_ANGLE) % 1.0;
  return hslToRgb(hue, 0.9, 0.55);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch helper
// ─────────────────────────────────────────────────────────────────────────────

function dispatchSize(total: number): number {
  return Math.ceil(total / WG);
}

// ─────────────────────────────────────────────────────────────────────────────
// ATMouseFluid — main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT mousefluid 交互流体系统 — 完整 WebGPU 移植。
 *
 * 包含完整的 splat → curl → vorticity → divergence → pressure clear →
 * pressure solve → gradient subtract → advect (×2) 管线，加上
 * 鼠标/触摸指针追踪和 splat 队列管理。
 */
export class ATMouseFluid {
  // ── Config ──
  readonly simRes: number;
  readonly dyeRes: number;
  private pressureIters: number;
  private curlStrength: number;
  private splatRadius: number;
  private velocityDissipation: number;
  private densityDissipation: number;
  private pressureDissipation: number;
  private splatBlendMode: number;
  private dt: number;

  // ── GPU resources ──
  private device: GPUDevice;
  private velocity: PingPong;
  private density: PingPong;
  private pressure: PingPong;
  private divergenceTex: GPUTexture;
  private divergenceView: GPUTextureView;
  private curlTex: GPUTexture;
  private curlView: GPUTextureView;

  // ── Uniform buffers ──
  private uniformBuf: GPUBuffer;
  private clearUniformBuf: GPUBuffer;

  // ── Pipelines ──
  private splatPipeline: GPUComputePipeline;
  private advectPipeline: GPUComputePipeline;
  private curlPipeline: GPUComputePipeline;
  private vorticityPipeline: GPUComputePipeline;
  private divergencePipeline: GPUComputePipeline;
  private pressurePipeline: GPUComputePipeline;
  private gradSubPipeline: GPUComputePipeline;
  private clearPipeline: GPUComputePipeline;

  // ── Pointer tracking ──
  private splatQueue: PointerSplat[] = [];
  private pointerMap = new Map<number, { x: number; y: number }>();
  private splatIndex = 0;
  private canvas: HTMLCanvasElement | null = null;
  private boundPointerDown: ((e: PointerEvent) => void) | null = null;
  private boundPointerMove: ((e: PointerEvent) => void) | null = null;
  private boundPointerUp: ((e: PointerEvent) => void) | null = null;

  // ── Lifecycle ──
  private destroyed = false;

  private constructor(
    device: GPUDevice,
    config: Required<ATMouseFluidConfig>,
  ) {
    this.device = device;
    this.simRes = config.simRes;
    this.dyeRes = config.dyeRes;
    this.pressureIters = config.pressureIters;
    this.curlStrength = config.curlStrength;
    this.splatRadius = config.splatRadius / 100; // AT normalisation
    this.velocityDissipation = config.velocityDissipation;
    this.densityDissipation = config.densityDissipation;
    this.pressureDissipation = config.pressureDissipation;
    this.splatBlendMode = config.splatBlendMode;
    this.dt = config.dt;

    // ── Textures ──
    const VEL_FMT: GPUTextureFormat = 'rgba16float';
    const DYE_FMT: GPUTextureFormat = 'rgba16float';
    const SCL_FMT: GPUTextureFormat = 'r16float';

    this.velocity = createPingPong(device, this.simRes, this.simRes, VEL_FMT, 'at_mf_vel');
    this.density = createPingPong(device, this.dyeRes, this.dyeRes, DYE_FMT, 'at_mf_dye');
    this.pressure = createPingPong(device, this.simRes, this.simRes, SCL_FMT, 'at_mf_pre');

    const sclDesc: GPUTextureDescriptor = {
      size: { width: this.simRes, height: this.simRes },
      format: SCL_FMT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    };
    this.divergenceTex = device.createTexture({ ...sclDesc, label: 'at_mf_div' });
    this.divergenceView = this.divergenceTex.createView();
    this.curlTex = device.createTexture({ ...sclDesc, label: 'at_mf_curl' });
    this.curlView = this.curlTex.createView();

    // ── Uniform buffers ──
    this.uniformBuf = device.createBuffer({
      size: FLUID_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'at_mf_params',
    });
    this.clearUniformBuf = device.createBuffer({
      size: 16, // f32 value + 12 padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'at_mf_clear_params',
    });

    // ── Pipelines ──
    this.splatPipeline = this.createPipeline('at_mf_splat', WGSL_SPLAT, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: VEL_FMT } },
    ]);

    this.advectPipeline = this.createPipeline('at_mf_advect', WGSL_ADVECT, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: VEL_FMT } },
    ]);

    this.curlPipeline = this.createPipeline('at_mf_curl', WGSL_CURL, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: SCL_FMT } },
    ]);

    this.vorticityPipeline = this.createPipeline('at_mf_vort', WGSL_VORTICITY, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: VEL_FMT } },
    ]);

    this.divergencePipeline = this.createPipeline('at_mf_div', WGSL_DIVERGENCE, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: SCL_FMT } },
    ]);

    this.pressurePipeline = this.createPipeline('at_mf_pre', WGSL_PRESSURE, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: SCL_FMT } },
    ]);

    this.gradSubPipeline = this.createPipeline('at_mf_gradsub', WGSL_GRADIENT_SUBTRACT, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: VEL_FMT } },
    ]);

    this.clearPipeline = this.createPipeline('at_mf_clear', WGSL_CLEAR, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: SCL_FMT } },
    ]);
  }

  // ── Factory ──

  /**
   * Create and initialise an ATMouseFluid instance.
   * Optionally attaches pointer listeners to a canvas.
   */
  static create(
    device: GPUDevice,
    canvas?: HTMLCanvasElement | null,
    config?: ATMouseFluidConfig,
  ): ATMouseFluid {
    const cfg: Required<ATMouseFluidConfig> = {
      simRes:               config?.simRes               ?? 128,
      dyeRes:               config?.dyeRes               ?? 512,
      pressureIters:        config?.pressureIters         ?? 4,
      curlStrength:         config?.curlStrength          ?? 20,
      splatRadius:          config?.splatRadius           ?? 0.2,
      velocityDissipation:  config?.velocityDissipation   ?? 0.98,
      densityDissipation:   config?.densityDissipation    ?? 0.97,
      pressureDissipation:  config?.pressureDissipation   ?? 0.8,
      splatBlendMode:       config?.splatBlendMode        ?? 0.0,
      dt:                   config?.dt                    ?? 1 / 60,
    };
    const fluid = new ATMouseFluid(device, cfg);
    if (canvas) {
      fluid.attachPointerListeners(canvas);
    }
    return fluid;
  }

  // ── Pointer listeners ──

  /**
   * Attach mouse/touch pointer listeners to a canvas.
   * Converts pointer events into splats in the queue.
   */
  attachPointerListeners(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;

    this.boundPointerDown = (e: PointerEvent) => {
      canvas.setPointerCapture(e.pointerId);
      const x = e.offsetX / canvas.clientWidth;
      const y = 1 - e.offsetY / canvas.clientHeight; // AT Y-flip
      this.pointerMap.set(e.pointerId, { x, y });
    };

    this.boundPointerMove = (e: PointerEvent) => {
      const prev = this.pointerMap.get(e.pointerId);
      const x = e.offsetX / canvas.clientWidth;
      const y = 1 - e.offsetY / canvas.clientHeight;

      if (!prev) {
        // mouse move without down — still track for hover fluid
        this.pointerMap.set(e.pointerId, { x, y });
        return;
      }

      const dx = (x - prev.x) * canvas.clientWidth * 5;
      const dy = (y - prev.y) * canvas.clientHeight * 5;

      if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
        const [r, g, b] = nextSplatColour(this.splatIndex++);
        this.splatQueue.push({
          x, y,
          px: prev.x, py: prev.y,
          dx, dy,
          r, g, b,
        });
      }
      this.pointerMap.set(e.pointerId, { x, y });
    };

    this.boundPointerUp = (e: PointerEvent) => {
      this.pointerMap.delete(e.pointerId);
    };

    canvas.addEventListener('pointerdown', this.boundPointerDown);
    canvas.addEventListener('pointermove', this.boundPointerMove);
    canvas.addEventListener('pointerup', this.boundPointerUp);
    canvas.addEventListener('pointerleave', this.boundPointerUp);
  }

  /**
   * Detach pointer listeners.
   */
  detachPointerListeners(): void {
    if (!this.canvas) return;
    if (this.boundPointerDown) this.canvas.removeEventListener('pointerdown', this.boundPointerDown);
    if (this.boundPointerMove) this.canvas.removeEventListener('pointermove', this.boundPointerMove);
    if (this.boundPointerUp) {
      this.canvas.removeEventListener('pointerup', this.boundPointerUp);
      this.canvas.removeEventListener('pointerleave', this.boundPointerUp);
    }
    this.canvas = null;
    this.boundPointerDown = null;
    this.boundPointerMove = null;
    this.boundPointerUp = null;
  }

  /**
   * Manually enqueue a splat (for programmatic injection, idle ripple, etc.)
   */
  enqueueSplat(splat: PointerSplat): void {
    this.splatQueue.push(splat);
  }

  // ── Per-frame step ──

  /**
   * Run one frame of the full AT mousefluid pipeline.
   *
   * Order (matching AT compiled.vs pipeline):
   *   0. Flush splat queue → velocity + density splats
   *   1. Curl
   *   2. Vorticity confinement
   *   3. Divergence
   *   4. Pressure clear
   *   5. Pressure solve (Jacobi × N)
   *   6. Gradient subtract
   *   7a. Advect velocity (self-advection)
   *   7b. Advect density (dye advection)
   */
  step(): void {
    if (this.destroyed) return;
    const encoder = this.device.createCommandEncoder({ label: 'at_mf_frame' });

    // ── Pass 0: SPLAT ──
    // Flush all pending pointer splats
    const aspect = this.canvas
      ? this.canvas.clientWidth / this.canvas.clientHeight
      : 1.0;

    for (const s of this.splatQueue) {
      // Velocity splat
      this.writeSplatUniforms(s.x, s.y, s.px, s.py, s.dx, s.dy, 1.0, aspect, 1.0);
      this.encodeSplat(encoder, this.velocity, this.simRes);

      // Dye splat
      this.writeSplatUniforms(s.x, s.y, s.px, s.py, s.r, s.g, s.b, aspect, 0.0);
      this.encodeSplat(encoder, this.density, this.dyeRes);
    }
    this.splatQueue.length = 0;

    // ── Pass 1: CURL ──
    {
      const bg = this.createBindGroup(this.curlPipeline, [
        ppReadView(this.velocity),
        this.curlView,
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_curl' });
      pass.setPipeline(this.curlPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
    }

    // ── Pass 2: VORTICITY CONFINEMENT ──
    {
      this.writeVorticityUniforms();
      const bg = this.createBindGroup(this.vorticityPipeline, [
        this.uniformBuf,
        ppReadView(this.velocity),
        this.curlView,
        ppWriteView(this.velocity),
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_vort' });
      pass.setPipeline(this.vorticityPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
      ppSwap(this.velocity);
    }

    // ── Pass 3: DIVERGENCE ──
    {
      const bg = this.createBindGroup(this.divergencePipeline, [
        ppReadView(this.velocity),
        this.divergenceView,
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_div' });
      pass.setPipeline(this.divergencePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
    }

    // ── Pass 4: PRESSURE CLEAR ──
    {
      const clearData = new Float32Array([this.pressureDissipation, 0, 0, 0]);
      this.device.queue.writeBuffer(this.clearUniformBuf, 0, clearData);
      const bg = this.createBindGroup(this.clearPipeline, [
        this.clearUniformBuf,
        ppReadView(this.pressure),
        ppWriteView(this.pressure),
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_clear' });
      pass.setPipeline(this.clearPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
      ppSwap(this.pressure);
    }

    // ── Pass 5: PRESSURE SOLVE (Jacobi × N) ──
    for (let i = 0; i < this.pressureIters; i++) {
      const bg = this.createBindGroup(this.pressurePipeline, [
        ppReadView(this.pressure),
        this.divergenceView,
        ppWriteView(this.pressure),
      ]);
      const pass = encoder.beginComputePass({ label: `at_mf_pre_${i}` });
      pass.setPipeline(this.pressurePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
      ppSwap(this.pressure);
    }

    // ── Pass 6: GRADIENT SUBTRACT ──
    {
      const bg = this.createBindGroup(this.gradSubPipeline, [
        ppReadView(this.pressure),
        ppReadView(this.velocity),
        ppWriteView(this.velocity),
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_gradsub' });
      pass.setPipeline(this.gradSubPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
      ppSwap(this.velocity);
    }

    // ── Pass 7a: ADVECT VELOCITY (self-advection) ──
    {
      this.writeAdvectUniforms(1.0 / this.simRes, this.velocityDissipation);
      const bg = this.createBindGroup(this.advectPipeline, [
        this.uniformBuf,
        ppReadView(this.velocity),
        ppReadView(this.velocity),
        ppWriteView(this.velocity),
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_advect_vel' });
      pass.setPipeline(this.advectPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.simRes), dispatchSize(this.simRes));
      pass.end();
      ppSwap(this.velocity);
    }

    // ── Pass 7b: ADVECT DENSITY (dye advection) ──
    {
      this.writeAdvectUniforms(1.0 / this.dyeRes, this.densityDissipation);
      const bg = this.createBindGroup(this.advectPipeline, [
        this.uniformBuf,
        ppReadView(this.velocity),
        ppReadView(this.density),
        ppWriteView(this.density),
      ]);
      const pass = encoder.beginComputePass({ label: 'at_mf_advect_dye' });
      pass.setPipeline(this.advectPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatchSize(this.dyeRes), dispatchSize(this.dyeRes));
      pass.end();
      ppSwap(this.density);
    }

    this.device.queue.submit([encoder.finish()]);
  }

  // ── Output accessors ──

  /** Get the current dye (density) texture view — tFluid for downstream shaders. */
  getDyeTextureView(): GPUTextureView {
    return ppReadView(this.density);
  }

  /** Get the current dye texture — for direct binding. */
  getDyeTexture(): GPUTexture {
    return ppRead(this.density);
  }

  /** Get the current velocity texture view — for tFluidMask or velocity reads. */
  getVelocityTextureView(): GPUTextureView {
    return ppReadView(this.velocity);
  }

  /** Get the current velocity texture — for direct binding. */
  getVelocityTexture(): GPUTexture {
    return ppRead(this.velocity);
  }

  /** Get read-only state snapshot. */
  getState(): ATMouseFluidState {
    return {
      dyeTextureView: this.getDyeTextureView(),
      velocityTextureView: this.getVelocityTextureView(),
      simRes: this.simRes,
      dyeRes: this.dyeRes,
    };
  }

  // ── Runtime parameter update ──

  /**
   * Update simulation parameters at runtime.
   * Only provided fields are changed; others keep their current values.
   */
  updateParams(params: Partial<ATMouseFluidConfig>): void {
    if (params.pressureIters !== undefined) this.pressureIters = params.pressureIters;
    if (params.curlStrength !== undefined)  this.curlStrength = params.curlStrength;
    if (params.splatRadius !== undefined)   this.splatRadius = params.splatRadius / 100;
    if (params.velocityDissipation !== undefined) this.velocityDissipation = params.velocityDissipation;
    if (params.densityDissipation !== undefined)  this.densityDissipation = params.densityDissipation;
    if (params.pressureDissipation !== undefined) this.pressureDissipation = params.pressureDissipation;
    if (params.splatBlendMode !== undefined) this.splatBlendMode = params.splatBlendMode;
    if (params.dt !== undefined) this.dt = params.dt;
  }

  // ── Destroy ──

  /** Release all GPU resources and detach listeners. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.detachPointerListeners();
    ppDestroy(this.velocity);
    ppDestroy(this.density);
    ppDestroy(this.pressure);
    this.divergenceTex.destroy();
    this.curlTex.destroy();
    this.uniformBuf.destroy();
    this.clearUniformBuf.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private createPipeline(
    label: string,
    code: string,
    entries: GPUBindGroupLayoutEntry[],
  ): GPUComputePipeline {
    const layout = this.device.createBindGroupLayout({ entries, label: `${label}_bgl` });
    const module = this.device.createShaderModule({ code, label: `${label}_sm` });
    return this.device.createComputePipeline({
      label,
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [layout],
        label: `${label}_pl`,
      }),
      compute: { module, entryPoint: 'main' },
    });
  }

  private createBindGroup(
    pipeline: GPUComputePipeline,
    resources: (GPUTextureView | GPUBuffer)[],
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({
        binding: i,
        resource: r instanceof GPUBuffer ? { buffer: r } : r,
      })),
    });
  }

  /**
   * Write splat uniforms — AT splatShader.fs parameters.
   * colR/colG/colB: for velocity splats, (dx, dy, 1); for dye splats, (r, g, b).
   * addMode: 1.0 for velocity (pure additive), 0.0 for dye (screen blend).
   */
  private writeSplatUniforms(
    x: number, y: number,
    px: number, py: number,
    colR: number, colG: number, colB: number,
    aspect: number,
    addMode: number,
  ): void {
    const buf = new ArrayBuffer(FLUID_UNIFORM_SIZE);
    const f = new Float32Array(buf);
    // texelSize
    f[0] = 1.0 / this.simRes;
    f[1] = 1.0 / this.simRes;
    // dt, dissipation (unused by splat, but fill for struct alignment)
    f[2] = this.dt;
    f[3] = 1.0;
    // point
    f[4] = x;
    f[5] = y;
    // prevPoint
    f[6] = px;
    f[7] = py;
    // color
    f[8] = colR;
    f[9] = colG;
    f[10] = colB;
    // radius
    f[11] = this.splatRadius;
    // curlStrength, pressureDissipation (unused by splat)
    f[12] = this.curlStrength;
    f[13] = this.pressureDissipation;
    // aspectRatio
    f[14] = aspect;
    // canRender
    f[15] = 1.0;
    // uAdd
    f[16] = addMode;
    // simRes, dyeRes
    f[17] = this.simRes;
    f[18] = this.dyeRes;
    f[19] = 0;
    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }

  private writeVorticityUniforms(): void {
    const buf = new ArrayBuffer(FLUID_UNIFORM_SIZE);
    const f = new Float32Array(buf);
    f[0] = 1.0 / this.simRes;
    f[1] = 1.0 / this.simRes;
    f[2] = this.dt;
    f[3] = 1.0;
    // point, prevPoint, color, radius — unused
    f[12] = this.curlStrength;
    f[13] = this.pressureDissipation;
    f[14] = 1.0; // aspectRatio
    f[15] = 1.0; // canRender
    f[16] = 0.0; // uAdd
    f[17] = this.simRes;
    f[18] = this.dyeRes;
    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }

  private writeAdvectUniforms(texelInv: number, dissipation: number): void {
    const buf = new ArrayBuffer(FLUID_UNIFORM_SIZE);
    const f = new Float32Array(buf);
    f[0] = 1.0 / this.simRes; // texelSize for velocity lookup
    f[1] = 1.0 / this.simRes;
    f[2] = this.dt;
    f[3] = dissipation;
    // rest zeroed / unused — struct alignment
    f[17] = this.simRes;
    f[18] = this.dyeRes;
    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }

  /**
   * Encode a splat pass — AT splatShader.fs equivalent.
   * Writes into a ping-pong texture pair (velocity or density).
   */
  private encodeSplat(
    encoder: GPUCommandEncoder,
    pp: PingPong,
    res: number,
  ): void {
    const bg = this.createBindGroup(this.splatPipeline, [
      this.uniformBuf,
      ppReadView(pp),
      ppWriteView(pp),
    ]);
    const pass = encoder.beginComputePass({ label: 'at_mf_splat' });
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(dispatchSize(res), dispatchSize(res));
    pass.end();
    ppSwap(pp);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export all shader constants for downstream integration
// ─────────────────────────────────────────────────────────────────────────────

export {
  WG as FLUID_WORKGROUP_SIZE,
  nextSplatColour,
  hslToRgb,
};

// ─────────────────────────────────────────────────────────────────────────────
// Default export
// ─────────────────────────────────────────────────────────────────────────────

export default ATMouseFluid;
