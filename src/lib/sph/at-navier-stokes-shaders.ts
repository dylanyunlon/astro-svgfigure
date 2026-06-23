/**
 * at-navier-stokes-shaders.ts — M857: AT Navier-Stokes Fluid Shader Pipeline
 * ─────────────────────────────────────────────────────────────────────────────
 * 鼠标是笔，流体是墨。
 *
 * 直接从 upstream/activetheory-assets/compiled.vs 提取 AT 的完整 Navier-Stokes
 * 流体 shader 集合，封装为面向 WebGL2 的 fluid simulation pipeline。
 * 这套 shader 驱动 AT 的鼠标交互流体（mousefluid）系统——所有运动、颜色扩散、
 * 涡度约束均由此实现。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 管线 — 11 个 shader pass（每帧执行顺序）
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  Pass 0  backgroundShader.fs   — 棋盘格背景（调试/开发可视化用）
 *  Pass 1  colorShader.fs        — 单色填充（FBO 初始化 / clear colour）
 *  Pass 2  clearShader.fs        — 压力场衰减清除（value × texture）
 *  Pass 3  splatShader.fs        — 线段 splat：鼠标/触摸点 → 流体速度+颜色注入
 *                                   AT 特色：prevPoint→point 线段距离，cubicOut 缓动，
 *                                   screen blend 混合，canRender/uAdd 双模式
 *  Pass 4  curlShader.fs         — 涡度计算 ω = 0.5*(R-L-T+B)，存入 curl FBO
 *  Pass 5  divergenceShader.fs   — 散度计算 div = 0.5*(R-L+T-B)，存入 div FBO
 *                                   AT 边界反射：vL.x<0 → L=-C.x 等
 *  Pass 6  pressureShader.fs     — Jacobi 压力迭代：p=(L+R+B+T-div)*0.25
 *                                   (ping-pong × N 次)
 *  Pass 7  gradientSubtractShader.fs — 压力梯度减法 v -= 0.5*(R-L, T-B)
 *                                      AT boundary() 透传（可选 clamp）
 *  Pass 8  advectionShader.fs    — 半拉格朗日对流（线性 textureLookup）
 *                                   coord = vUv - dt * vel * texelSize
 *  Pass 9  advectionManualFilteringShader.fs — 手动双线性对流（fallback）
 *                                   AT bilerp() 在不支持线性滤波 float 纹理时使用
 *  Pass 10 displayShader.fs      — 流体颜色输出：RGB+透明度 max(R,G,B)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Vertex shader（所有 pass 共用）
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  fluidBase.vs — 计算相邻四邻域 UV（vL, vR, vT, vB）供 fs 使用
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 来源
 * ─────────────────────────────────────────────────────────────────────────────
 *  • upstream/activetheory-assets/compiled.vs  (块分隔符 {@})
 *    — advectionManualFilteringShader.fs  行 6579
 *    — advectionShader.fs                行 6600
 *    — backgroundShader.fs               行 6610
 *    — clearShader.fs                    行 6619
 *    — colorShader.fs                    行 6624
 *    — curlShader.fs                     行 6627
 *    — displayShader.fs                  行 6640
 *    — divergenceShader.fs               行 6646
 *    — gradientSubtractShader.fs         行 6678
 *    — pressureShader.fs                 行 6698
 *    — splatShader.fs                    行 6720
 *    — fluidBase.vs                      行 6756  (shared vertex shader)
 *    — vorticityShader.fs                行 6759  (涡度约束，补充 pass)
 *
 *  • src/lib/sph/at-mousefluid-import.ts  — M815 AT mousefluid pipeline 参考
 *  • src/lib/sph/at-navier-stokes.ts      — M715 WebGPU WGSL 端口
 *  • upstream/ogl/examples/post-fluid-distortion.html — Pavel Dobryakov 参考
 *
 * Research: M857 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// Vertex Shader — fluidBase.vs
// ─────────────────────────────────────────────────────────────────────────────
//
// 共用顶点着色器。计算当前像素的四邻域 UV 坐标（vL, vR, vT, vB），
// 供所有流体 pass 的 fragment shader 直接读取，避免在 fs 内重复做偏移计算。
// texelSize 由 CPU 每帧更新（vec2(1/W, 1/H)）。

export const AT_FLUID_BASE_VS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0 — backgroundShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 棋盘格背景渲染，用于调试可视化。SCALE=25 控制格子粒度，
// 颜色在 0.8–0.9 灰阶间交替（不影响流体模拟本身）。
//
// Uniforms:
//   uTexture     sampler2D  — 当前帧纹理（当前实现未使用，保留接口）
//   aspectRatio  float      — 宽高比（保持格子正方形）

export const AT_BACKGROUND_FS = /* glsl */`
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float aspectRatio;
#define SCALE 25.0
void main () {
    vec2 uv = floor(vUv * SCALE * vec2(aspectRatio, 1.0));
    float v = mod(uv.x + uv.y, 2.0);
    v = v * 0.1 + 0.8;
    gl_FragColor = vec4(vec3(v), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — colorShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 单色填充。用于 FBO 初始化或 clear colour pass。
// 直接将 uniform color（vec4 RGBA）写入 gl_FragColor，无纹理采样。
//
// Uniforms:
//   color  vec4  — RGBA 填充色

export const AT_COLOR_FS = /* glsl */`
uniform vec4 color;
void main () {
    gl_FragColor = color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — clearShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 压力场衰减清除。读取现有纹理后乘以衰减系数 value（< 1.0）实现逐帧压力
// 衰减。标准用法：每帧在 pressure Jacobi 迭代前执行，防止压力累积。
//
// Uniforms:
//   uTexture  sampler2D  — 待清除的纹理（通常为 pressure FBO）
//   value     float      — 衰减系数，0.0 = 完全清除，1.0 = 保持不变

export const AT_CLEAR_FS = /* glsl */`
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — splatShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// AT 线段 Splat——鼠标/触摸交互的核心注入 pass。
//
// AT 独有特性（区别于 OGL/Pavel 的简单 Gaussian 点 splat）：
//   • 使用 prevPoint→point 线段距离 l()，高速移动时轨迹连续无断裂
//   • cubicOut 缓动函数让 splat 边缘柔和
//   • blendScreen 混合模式（1-(1-a)(1-b)），配合 uAdd 在 screen/additive 间插值
//   • canRender float 开关（0.0=禁用，1.0=启用），用于条件渲染
//   • aspectRatio 修正 X 轴拉伸，保证圆形 splat
//
// Uniforms:
//   uTarget     sampler2D  — 读取当前帧 velocity 或 dye 纹理
//   aspectRatio float      — 宽高比
//   color       vec3       — splat 颜色（velocity pass: (dx,dy,1); dye pass: hue RGB）
//   bgColor     vec3       — 背景色（当前未使用，保留 AT 接口）
//   point       vec2       — 当前鼠标位置（归一化 [0,1]）
//   prevPoint   vec2       — 上一帧鼠标位置
//   radius      float      — splat 半径（归一化，典型值 0.02–0.05）
//   canRender   float      — 渲染开关（0.0 或 1.0）
//   uAdd        float      — screen/additive 混合插值（0=screen, 1=additive）

export const AT_SPLAT_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4 — curlShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 涡度（curl / vorticity）计算。
// 从速度场的四邻域采样计算 2D curl：
//   ω = 0.5 * (∂vy/∂x − ∂vx/∂y)
//     = 0.5 * (R.y - L.y - T.x + B.x)
// 结果存入 curl FBO 的 R 通道（范围约 [-1, 1]，映射到 [0, 1] 需 0.5 offset）。
//
// 依赖 fluidBase.vs 提供 vL, vR, vT, vB。
//
// Uniforms:
//   uVelocity  sampler2D  — 速度场纹理（XY = velocity）

export const AT_CURL_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 5 — divergenceShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 散度（divergence）计算。
// 用于压力 Poisson 求解的右端项：
//   div = 0.5 * (∂vx/∂x + ∂vy/∂y)
//       = 0.5 * (R.x - L.x + T.y - B.y)
//
// AT 特色：边界反射条件——
//   if (vL.x < 0.0) { L = -C.x; }   // 左边界：无滑移反射
//   if (vR.x > 1.0) { R = -C.x; }   // 右边界
//   if (vT.y > 1.0) { T = -C.y; }   // 上边界
//   if (vB.y < 0.0) { B = -C.y; }   // 下边界
// 这确保速度在边界法向分量为 0（no-penetration 条件）。
//
// 依赖 fluidBase.vs 提供 vL, vR, vT, vB。
//
// Uniforms:
//   uVelocity  sampler2D  — 速度场纹理

export const AT_DIVERGENCE_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 6 — pressureShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// Jacobi 迭代压力求解。每次迭代：
//   p_new = (L + R + B + T - divergence) * 0.25
// 标准离散 Poisson 方程（dx=1，α=1，β=4）的 Jacobi 松弛步。
// 通常执行 20–50 次 ping-pong 迭代以收敛。
//
// AT boundary() 函数当前透传（可选 clamp 模式，注释在代码中）。
//
// 依赖 fluidBase.vs 提供 vL, vR, vT, vB。
//
// Uniforms:
//   uPressure    sampler2D  — 当前压力场（ping-pong 输入）
//   uDivergence  sampler2D  — 散度场（每帧固定，不 ping-pong）

export const AT_PRESSURE_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 7 — gradientSubtractShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 压力梯度减法（投影步）。从速度场减去压力梯度以强制不可压缩性：
//   v_new = v - 0.5 * (∇p)
//   ∇p = (R - L, T - B)
// 这是 Helmholtz-Hodge 分解的"投影"步，将速度场投影到 divergence-free 子空间。
//
// AT boundary() 透传（与 pressureShader.fs 相同的设计）。
//
// 依赖 fluidBase.vs 提供 vL, vR, vT, vB。
//
// Uniforms:
//   uPressure  sampler2D  — 收敛后的压力场
//   uVelocity  sampler2D  — 待修正的速度场

export const AT_GRADIENT_SUBTRACT_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 8 — advectionShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 半拉格朗日对流（标准版，依赖 GPU 线性滤波）。
// 反向追踪：从当前位置沿速度方向回溯一步，采样上一帧的场值：
//   coord = vUv - dt * vel.xy * texelSize
//   gl_FragColor = dissipation * texture2D(uSource, coord)
//
// 同时用于速度自对流（uSource=velocity）和染料对流（uSource=dye）。
// dissipation < 1.0 实现逐帧衰减（速度衰减 ~0.999，染料衰减 ~0.995）。
// 要求纹理支持线性滤波 float（WebGL2 EXT_color_buffer_float + LINEAR）。
// 不支持时使用 advectionManualFilteringShader.fs (Pass 9)。
//
// Uniforms:
//   uVelocity    sampler2D  — 速度场纹理（提供反向追踪速度）
//   uSource      sampler2D  — 待对流的场（velocity 或 dye）
//   texelSize    vec2       — vec2(1/W, 1/H)
//   dt           float      — 时间步长（秒，典型值 1/60）
//   dissipation  float      — 衰减系数

export const AT_ADVECTION_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 9 — advectionManualFilteringShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 半拉格朗日对流（手动双线性滤波版）。
// 在不支持 float 纹理线性插值的设备上（如部分移动端 WebGL1/ES2）的 fallback。
// 使用 bilerp() 手动计算双线性插值，绕过 GPU 硬件滤波限制。
//
// bilerp() 实现：
//   st = uv / tsize - 0.5            // 转换到纹素坐标
//   四角采样 (iuv+0.5, iuv+1.5) × tsize
//   mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y)
//
// 额外增加 dyeTexelSize uniform，支持 velocity 和 dye 分辨率不同的情况
// （AT 典型配置：velocity 128×128，dye 512×512）。
//
// Uniforms:
//   uVelocity      sampler2D  — 速度场纹理
//   uSource        sampler2D  — 待对流的场
//   texelSize      vec2       — 速度场纹素尺寸 vec2(1/velW, 1/velH)
//   dyeTexelSize   vec2       — 源场纹素尺寸 vec2(1/srcW, 1/srcH)
//   dt             float      — 时间步长
//   dissipation    float      — 衰减系数

export const AT_ADVECTION_MANUAL_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pass 10 — displayShader.fs
// ─────────────────────────────────────────────────────────────────────────────
//
// 流体颜色输出。将 dye/density 纹理渲染到屏幕，透明度由颜色强度决定：
//   a = max(R, G, B)
// 这使得密度为 0 的区域完全透明，允许背景透过（AT 的 alpha-blended overlay 效果）。
//
// Uniforms:
//   uTexture  sampler2D  — dye/density 纹理

export const AT_DISPLAY_FS = /* glsl */`
varying vec2 vUv;
uniform sampler2D uTexture;
void main () {
    vec3 C = texture2D(uTexture, vUv).rgb;
    float a = max(C.r, max(C.g, C.b));
    gl_FragColor = vec4(C, a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// 附加 — vorticityShader.fs（涡度约束，补充 Pass）
// ─────────────────────────────────────────────────────────────────────────────
//
// 涡度约束（vorticity confinement）。从 curlShader.fs 输出的 curl 纹理读取
// 邻域涡度，计算涡度梯度方向，施加约束力以重新激活被数值扩散抑制的小尺度涡旋：
//   force = 0.5 * (|T|-|B|, |R|-|L|)           // 涡度梯度
//   force = normalize(force) * curl_strength * C // 归一化 × 中心涡度
//   force.y *= -1.0                              // AT Y 轴翻转约定
//   vel += force * dt
//
// 注：compiled.vs 中有一行被注释掉的 "force.y += 400.3"，AT 可能曾用于测试。
//
// Uniforms:
//   uVelocity  sampler2D  — 速度场纹理（读取 + 写入）
//   uCurl      sampler2D  — curl FBO 纹理（来自 curlShader.fs）
//   curl       float      — 涡度约束强度（典型值 2.0–30.0）
//   dt         float      — 时间步长

export const AT_VORTICITY_FS = /* glsl */`
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

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline 描述符
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT Navier-Stokes pipeline 中单个 shader pass 的描述。
 */
export interface ATFluidShaderPass {
  /** Pass 名称（对应 compiled.vs 中的块名称）。 */
  name: string;
  /** Fragment shader 源码（GLSL ES 2.0 / WebGL）。 */
  fragmentShader: string;
  /** Vertex shader 源码（所有 pass 共用 AT_FLUID_BASE_VS）。 */
  vertexShader: string;
  /** 该 pass 需要的 uniform 名称列表（文档用途）。 */
  uniforms: readonly string[];
  /** 该 pass 在每帧中的执行顺序索引（0-based）。 */
  passIndex: number;
}

/**
 * AT Navier-Stokes 11 个 shader pass 的完整 pipeline 描述符数组。
 * 按标准 AT mousefluid 执行顺序排列。
 *
 * 典型每帧顺序（splat 在用户交互时插入 Pass 2 之前）：
 *   0  background   — 可选，仅调试
 *   1  color        — 可选，仅 FBO 初始化
 *   2  clear        — 压力衰减（pressureIter 前）
 *   3  splat        — 鼠标注入（每个 pointer event）
 *   4  curl         — 涡度计算
 *   4b vorticity    — 涡度约束（紧接 curl 之后）
 *   5  divergence   — 散度计算
 *   6  pressure     — Jacobi 迭代 × N（ping-pong）
 *   7  gradient     — 梯度减法（投影）
 *   8  advection    — 速度+染料对流
 *   10 display      — 输出到屏幕
 */
export const AT_NAVIER_STOKES_PIPELINE: readonly ATFluidShaderPass[] = [
  {
    name: 'backgroundShader.fs',
    fragmentShader: AT_BACKGROUND_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uTexture', 'aspectRatio'],
    passIndex: 0,
  },
  {
    name: 'colorShader.fs',
    fragmentShader: AT_COLOR_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['color'],
    passIndex: 1,
  },
  {
    name: 'clearShader.fs',
    fragmentShader: AT_CLEAR_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uTexture', 'value'],
    passIndex: 2,
  },
  {
    name: 'splatShader.fs',
    fragmentShader: AT_SPLAT_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uTarget', 'aspectRatio', 'color', 'bgColor', 'point', 'prevPoint', 'radius', 'canRender', 'uAdd'],
    passIndex: 3,
  },
  {
    name: 'curlShader.fs',
    fragmentShader: AT_CURL_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uVelocity'],
    passIndex: 4,
  },
  {
    name: 'divergenceShader.fs',
    fragmentShader: AT_DIVERGENCE_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uVelocity'],
    passIndex: 5,
  },
  {
    name: 'pressureShader.fs',
    fragmentShader: AT_PRESSURE_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uPressure', 'uDivergence'],
    passIndex: 6,
  },
  {
    name: 'gradientSubtractShader.fs',
    fragmentShader: AT_GRADIENT_SUBTRACT_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uPressure', 'uVelocity'],
    passIndex: 7,
  },
  {
    name: 'advectionShader.fs',
    fragmentShader: AT_ADVECTION_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uVelocity', 'uSource', 'texelSize', 'dt', 'dissipation'],
    passIndex: 8,
  },
  {
    name: 'advectionManualFilteringShader.fs',
    fragmentShader: AT_ADVECTION_MANUAL_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uVelocity', 'uSource', 'texelSize', 'dyeTexelSize', 'dt', 'dissipation'],
    passIndex: 9,
  },
  {
    name: 'displayShader.fs',
    fragmentShader: AT_DISPLAY_FS,
    vertexShader: AT_FLUID_BASE_VS,
    uniforms: ['uTexture'],
    passIndex: 10,
  },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Uniform 默认值
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT mousefluid 的默认 uniform 参数。
 * 与 at-mousefluid-import.ts 中的默认值保持一致。
 */
export interface ATFluidUniforms {
  /** 时间步长（秒）。默认 1/60。 */
  dt: number;
  /** 速度场衰减系数（0–1）。默认 0.999。 */
  velocityDissipation: number;
  /** 染料/密度场衰减系数（0–1）。默认 0.995。 */
  densityDissipation: number;
  /** 压力衰减系数（clearShader value）。默认 0.8。 */
  pressureDissipation: number;
  /** Jacobi 压力迭代次数。默认 25。 */
  pressureIterations: number;
  /** 涡度约束强度。默认 30.0。 */
  curl: number;
  /** Splat 半径（归一化）。默认 0.005。 */
  splatRadius: number;
  /** Screen/additive 混合（0=screen，1=additive）。默认 0.0。 */
  splatAdd: number;
  /** 宽高比（canvas.width / canvas.height）。默认 1.0。 */
  aspectRatio: number;
}

/** AT mousefluid 默认参数（与 at-mousefluid-import.ts ATMouseFluidConfig 对齐）。 */
export const AT_FLUID_UNIFORMS_DEFAULTS: Readonly<ATFluidUniforms> = {
  dt: 1 / 60,
  velocityDissipation: 0.999,
  densityDissipation:  0.995,
  pressureDissipation: 0.8,
  pressureIterations:  25,
  curl:                30.0,
  splatRadius:         0.005,
  splatAdd:            0.0,
  aspectRatio:         1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// FBO 配置
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT 流体 FBO 布局描述。
 * AT 典型配置：velocity/pressure/divergence/curl 使用低分辨率，dye 使用高分辨率。
 */
export interface ATFluidFBOConfig {
  /** 速度场分辨率（宽度）。默认 128。 */
  simWidth: number;
  /** 速度场分辨率（高度）。默认 128。 */
  simHeight: number;
  /** 染料场分辨率（宽度）。默认 1024。 */
  dyeWidth: number;
  /** 染料场分辨率（高度）。默认 1024。 */
  dyeHeight: number;
}

/** AT 流体 FBO 默认分辨率配置。 */
export const AT_FLUID_FBO_DEFAULTS: Readonly<ATFluidFBOConfig> = {
  simWidth:  128,
  simHeight: 128,
  dyeWidth:  1024,
  dyeHeight: 1024,
};

// ─────────────────────────────────────────────────────────────────────────────
// 快捷访问 map
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 按名称快速访问 shader 源码的 map。
 * key = compiled.vs 中的块名称（如 'splatShader.fs'）。
 */
export const AT_SHADER_MAP: Readonly<Record<string, string>> = {
  'fluidBase.vs':                       AT_FLUID_BASE_VS,
  'backgroundShader.fs':                AT_BACKGROUND_FS,
  'colorShader.fs':                     AT_COLOR_FS,
  'clearShader.fs':                     AT_CLEAR_FS,
  'splatShader.fs':                     AT_SPLAT_FS,
  'curlShader.fs':                      AT_CURL_FS,
  'vorticityShader.fs':                 AT_VORTICITY_FS,
  'divergenceShader.fs':                AT_DIVERGENCE_FS,
  'pressureShader.fs':                  AT_PRESSURE_FS,
  'gradientSubtractShader.fs':          AT_GRADIENT_SUBTRACT_FS,
  'advectionShader.fs':                 AT_ADVECTION_FS,
  'advectionManualFilteringShader.fs':  AT_ADVECTION_MANUAL_FS,
  'displayShader.fs':                   AT_DISPLAY_FS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports
// ─────────────────────────────────────────────────────────────────────────────

export {
  AT_FLUID_BASE_VS         as fluidBaseVS,
  AT_BACKGROUND_FS         as backgroundShaderFS,
  AT_COLOR_FS              as colorShaderFS,
  AT_CLEAR_FS              as clearShaderFS,
  AT_SPLAT_FS              as splatShaderFS,
  AT_CURL_FS               as curlShaderFS,
  AT_VORTICITY_FS          as vorticityShaderFS,
  AT_DIVERGENCE_FS         as divergenceShaderFS,
  AT_PRESSURE_FS           as pressureShaderFS,
  AT_GRADIENT_SUBTRACT_FS  as gradientSubtractShaderFS,
  AT_ADVECTION_FS          as advectionShaderFS,
  AT_ADVECTION_MANUAL_FS   as advectionManualFilteringShaderFS,
  AT_DISPLAY_FS            as displayShaderFS,
};
