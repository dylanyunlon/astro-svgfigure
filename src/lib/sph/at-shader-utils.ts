/**
 * at-shader-utils.ts — Active Theory Shader Utilities — WGSL Port
 *
 * 移植自 Active Theory compiled.vs 三个核心 GLSL 库：
 *
 *   eases.glsl    — 缓动函数 (quadratic / cubic / quartic / quintic /
 *                   sine / expo / circ / back / elastic / bounce, In/Out/InOut)
 *   range.glsl    — 区间重映射: range() / crange() / rangeMirror()
 *                   float + vec2 + vec3 + vec4 重载
 *   blendmodes.glsl — Photoshop 混合模式 (Add, Multiply, Screen, Overlay,
 *                   SoftLight, HardLight, Darken, Lighten, Difference,
 *                   Exclusion, ColorDodge, ColorBurn, Subtract)
 *                   float + vec3 重载 + opacity 变体
 *
 * 额外新增（非 compiled.vs 直接移植，但 AT 各 shader 中高频使用）:
 *   UV transforms — uvScale/Rotate/Translate/Tile/Polar + center 变体
 *   rotation      — rotate2D, rotationX/Y/Z (mat4x4), rotationAxisAngle,
 *                   rotationEuler (mat3x3), applyRotation/Dir helpers
 *
 * 每个区块均可单独注入 WGSL 字符串，也可使用合并的 AT_SHADER_UTILS_WGSL。
 *
 * WGSL 命名约定（无函数重载，后缀区分类型）：
 *   <fn>        → f32 标量版
 *   <fn>2       → vec2f 版
 *   <fn>3       → vec3f 版
 *   <fn>4       → vec4f 版
 *   <fn>3Opacity→ vec3f + opacity 混合版
 *
 * 来源：
 *   upstream/lygia/math/map.wgsl        → range / crange 基础
 *   upstream/lygia/color/blend/*.wgsl   → blend modes 基础
 *   upstream/lygia/math/cubic.wgsl      → cubic ease kernel
 *   upstream/lygia/math/quintic.wgsl    → quintic ease kernel
 *   src/lib/particle/antimatter.ts      → AT range/crange float 实现参考
 *   src/lib/tween-system.ts             → AT JS easing 实现参考
 *
 * Research: xiaodi #M717 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 1  eases.glsl → WGSL
//
// AT compiled.vs 使用 Penner easing 全套（in/out/inout 三变体）加
// back / elastic / bounce 共 30+ 函数。
// WGSL 无重载，用 easeInQuad / easeOutQuad / easeInOutQuad 等后缀区分。
// ─────────────────────────────────────────────────────────────────────────────









export const WGSL_EASES = /* wgsl */`
// ──────────────────────────────────────────────────────────────────────────────
// AT eases.glsl — Penner easing functions (WGSL port)
// Ref: Robert Penner's Easing Equations, MIT license
// AT usage: compiled.vs WorkDetailParticles, ProtonAntimatter transition masks
// ──────────────────────────────────────────────────────────────────────────────

// ── Linear ────────────────────────────────────────────────────────────────────
fn easeLinear(t: f32) -> f32 { return t; }

// ── Quadratic ─────────────────────────────────────────────────────────────────
fn easeInQuad(t: f32) -> f32 { return t * t; }
fn easeOutQuad(t: f32) -> f32 { return t * (2.0 - t); }
fn easeInOutQuad(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) { return 0.5 * s * s; }
    let u = s - 1.0;
    return -0.5 * (u * (u - 2.0) - 1.0);
}

// ── Cubic ─────────────────────────────────────────────────────────────────────
fn easeInCubic(t: f32) -> f32 { return t * t * t; }
fn easeOutCubic(t: f32) -> f32 {
    let u = t - 1.0;
    return u * u * u + 1.0;
}
fn easeInOutCubic(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) { return 0.5 * s * s * s; }
    let u = s - 2.0;
    return 0.5 * (u * u * u + 2.0);
}

// ── Quartic ───────────────────────────────────────────────────────────────────
fn easeInQuart(t: f32) -> f32 { return t * t * t * t; }
fn easeOutQuart(t: f32) -> f32 {
    let u = t - 1.0;
    return -(u * u * u * u - 1.0);
}
fn easeInOutQuart(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) { return 0.5 * s * s * s * s; }
    let u = s - 2.0;
    return -0.5 * (u * u * u * u - 2.0);
}

// ── Quintic ───────────────────────────────────────────────────────────────────
fn easeInQuint(t: f32) -> f32 { return t * t * t * t * t; }
fn easeOutQuint(t: f32) -> f32 {
    let u = t - 1.0;
    return u * u * u * u * u + 1.0;
}
fn easeInOutQuint(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) { return 0.5 * s * s * s * s * s; }
    let u = s - 2.0;
    return 0.5 * (u * u * u * u * u + 2.0);
}

// ── Sine ──────────────────────────────────────────────────────────────────────
const AT_HALF_PI : f32 = 1.5707963267948966;
const AT_PI      : f32 = 3.141592653589793;

fn easeInSine(t: f32) -> f32 { return 1.0 - cos(t * AT_HALF_PI); }
fn easeOutSine(t: f32) -> f32 { return sin(t * AT_HALF_PI); }
fn easeInOutSine(t: f32) -> f32 { return -0.5 * (cos(AT_PI * t) - 1.0); }

// ── Expo ──────────────────────────────────────────────────────────────────────
fn easeInExpo(t: f32) -> f32 {
    return select(pow(2.0, 10.0 * t - 10.0), 0.0, t == 0.0);
}
fn easeOutExpo(t: f32) -> f32 {
    return select(1.0 - pow(2.0, -10.0 * t), 1.0, t == 1.0);
}
fn easeInOutExpo(t: f32) -> f32 {
    if (t == 0.0) { return 0.0; }
    if (t == 1.0) { return 1.0; }
    let s = t * 2.0;
    if (s < 1.0) { return 0.5 * pow(2.0, 10.0 * s - 10.0); }
    return 0.5 * (2.0 - pow(2.0, -10.0 * (s - 1.0)));
}

// ── Circ ──────────────────────────────────────────────────────────────────────
fn easeInCirc(t: f32) -> f32 { return 1.0 - sqrt(max(1.0 - t * t, 0.0)); }
fn easeOutCirc(t: f32) -> f32 {
    let u = t - 1.0;
    return sqrt(max(1.0 - u * u, 0.0));
}
fn easeInOutCirc(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) { return -0.5 * (sqrt(max(1.0 - s * s, 0.0)) - 1.0); }
    let u = s - 2.0;
    return 0.5 * (sqrt(max(1.0 - u * u, 0.0)) + 1.0);
}

// ── Back (overshoot) ──────────────────────────────────────────────────────────
// AT: c1 = 1.70158 (overshoot constant used in compiled.vs WorkDetailParticles)
const AT_BACK_C1 : f32 = 1.70158;
const AT_BACK_C3 : f32 = 2.70158;   // c1 + 1
const AT_BACK_C2 : f32 = 2.5949095; // c1 * 1.525
const AT_BACK_C4 : f32 = 3.5949095; // c2 + 1

fn easeInBack(t: f32) -> f32 {
    return AT_BACK_C3 * t * t * t - AT_BACK_C1 * t * t;
}
fn easeOutBack(t: f32) -> f32 {
    let u = t - 1.0;
    return 1.0 + AT_BACK_C3 * u * u * u + AT_BACK_C1 * u * u;
}
fn easeInOutBack(t: f32) -> f32 {
    let s = t * 2.0;
    if (s < 1.0) {
        return 0.5 * (s * s * ((AT_BACK_C4 + 1.0) * s - AT_BACK_C4));
    }
    let u = s - 2.0;
    return 0.5 * (u * u * ((AT_BACK_C4 + 1.0) * u + AT_BACK_C4) + 2.0);
}

// ── Elastic (spring) ──────────────────────────────────────────────────────────
const AT_ELASTIC_C4 : f32 = 2.0943951023931953;  // (2*PI)/3
const AT_ELASTIC_C5 : f32 = 1.3962634015954636;  // (2*PI)/4.5

fn easeInElastic(t: f32) -> f32 {
    if (t == 0.0) { return 0.0; }
    if (t == 1.0) { return 1.0; }
    return -pow(2.0, 10.0 * t - 10.0) * sin((t * 10.0 - 10.75) * AT_ELASTIC_C4);
}
fn easeOutElastic(t: f32) -> f32 {
    if (t == 0.0) { return 0.0; }
    if (t == 1.0) { return 1.0; }
    return pow(2.0, -10.0 * t) * sin((t * 10.0 - 0.75) * AT_ELASTIC_C4) + 1.0;
}
fn easeInOutElastic(t: f32) -> f32 {
    if (t == 0.0) { return 0.0; }
    if (t == 1.0) { return 1.0; }
    let s = t * 2.0;
    if (s < 1.0) {
        return -0.5 * pow(2.0, 10.0 * s - 10.0) * sin((s * 10.0 - 11.125) * AT_ELASTIC_C5);
    }
    return pow(2.0, -10.0 * (s - 1.0)) * sin((s * 10.0 - 11.125) * AT_ELASTIC_C5) * 0.5 + 1.0;
}

// ── Bounce ────────────────────────────────────────────────────────────────────
fn easeOutBounce(t: f32) -> f32 {
    let n1 : f32 = 7.5625;
    let d1 : f32 = 2.75;
    var t_ = t;
    if (t_ < 1.0 / d1) {
        return n1 * t_ * t_;
    } else if (t_ < 2.0 / d1) {
        t_ -= 1.5 / d1;
        return n1 * t_ * t_ + 0.75;
    } else if (t_ < 2.5 / d1) {
        t_ -= 2.25 / d1;
        return n1 * t_ * t_ + 0.9375;
    } else {
        t_ -= 2.625 / d1;
        return n1 * t_ * t_ + 0.984375;
    }
}
fn easeInBounce(t: f32) -> f32 { return 1.0 - easeOutBounce(1.0 - t); }
fn easeInOutBounce(t: f32) -> f32 {
    if (t < 0.5) { return (1.0 - easeOutBounce(1.0 - 2.0 * t)) * 0.5; }
    return (1.0 + easeOutBounce(2.0 * t - 1.0)) * 0.5;
}

// ── Smooth-step kernels (lygia cubic / quintic) ───────────────────────────────
// AT compiled.vs uses smoothstep kernel variants in particle life curves.
fn easeSmoothstep(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }          // cubic (C1)
fn easeSmootherstep(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); } // quintic (C2)
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  range.glsl → WGSL
//
// AT range() is used in EVERY shader: Gem.fs, Sky.fs, Terrain.vs, compiled.vs.
// float/vec2/vec3/vec4 + clamped crange variants.
// Exactly matches AT's compiled.vs arithmetic (vec3 sub trick preserved as
// equivalent scalar form for readability; result identical).
// ─────────────────────────────────────────────────────────────────────────────

export const WGSL_RANGE = /* wgsl */`
// ──────────────────────────────────────────────────────────────────────────────
// AT range.glsl — value remapping utilities (WGSL port)
// AT: range(oldValue, oldMin, oldMax, newMin, newMax)
//     crange → clamped variant
//     rangeMirror → ping-pong mirror before remap
// ──────────────────────────────────────────────────────────────────────────────

// ── range — unclamped linear remap ───────────────────────────────────────────
fn range(v: f32, iMin: f32, iMax: f32, oMin: f32, oMax: f32) -> f32 {
    return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}
fn range2(v: vec2f, iMin: vec2f, iMax: vec2f, oMin: vec2f, oMax: vec2f) -> vec2f {
    return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}
fn range3(v: vec3f, iMin: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f {
    return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}
fn range4(v: vec4f, iMin: vec4f, iMax: vec4f, oMin: vec4f, oMax: vec4f) -> vec4f {
    return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}

// ── range scalar shorthand (maps to [0,1] output when oMin=0, oMax=1) ────────
fn rangeNorm(v: f32, iMin: f32, iMax: f32) -> f32 {
    return (v - iMin) / (iMax - iMin);
}

// ── crange — clamped remap ────────────────────────────────────────────────────
fn crange(v: f32, iMin: f32, iMax: f32, oMin: f32, oMax: f32) -> f32 {
    return clamp(range(v, iMin, iMax, oMin, oMax), min(oMin, oMax), max(oMin, oMax));
}
fn crange2(v: vec2f, iMin: vec2f, iMax: vec2f, oMin: vec2f, oMax: vec2f) -> vec2f {
    let r = range2(v, iMin, iMax, oMin, oMax);
    return clamp(r, min(oMin, oMax), max(oMin, oMax));
}
fn crange3(v: vec3f, iMin: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f {
    let r = range3(v, iMin, iMax, oMin, oMax);
    return clamp(r, min(oMin, oMax), max(oMin, oMax));
}
fn crange4(v: vec4f, iMin: vec4f, iMax: vec4f, oMin: vec4f, oMax: vec4f) -> vec4f {
    let r = range4(v, iMin, iMax, oMin, oMax);
    return clamp(r, min(oMin, oMax), max(oMin, oMax));
}

// ── rangeMirror — ping-pong mirror then remap (AT: used in colour cycling) ────
// Mirror the input in [iMin, iMax] (triangle wave), then remap to [oMin, oMax].
fn rangeMirror(v: f32, iMin: f32, iMax: f32, oMin: f32, oMax: f32) -> f32 {
    let span  = iMax - iMin;
    let t     = v - iMin;
    let cycle = abs(t - 2.0 * span * floor((t + span) / (2.0 * span)));
    return oMin + (oMax - oMin) * cycle / span;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 3  UV transforms + Rotation matrices → WGSL
//
// AT compiled.vs commonly applies UV scale/rotate/translate in vertex shaders
// (Gem.vs, Sky.vs, WorkDetailParticles, ProtonAntimatter particle UVs).
// Also: 2D/3D rotation matrices used for particle spin, curl field orientation.
// Provides: uvScale, uvRotate, uvTranslate, uvScaleCenter, uvRotateCenter,
//           rotate2D, rotationX/Y/Z (mat4x4), rotationAxisAngle.
// ─────────────────────────────────────────────────────────────────────────────

export const WGSL_UV_TRANSFORMS = /* wgsl */`
// ──────────────────────────────────────────────────────────────────────────────
// AT UV transforms + rotation matrices (WGSL)
// Used in: Gem.vs UV tiling, Sky.vs parallax, WorkDetailParticles spin,
//          ProtonAntimatter orbit rotation, curl-flow-field orientation.
// Ref: AT compiled.vs UV manipulation patterns + standard rotation matrices
// ──────────────────────────────────────────────────────────────────────────────

// ── UV manipulation ──────────────────────────────────────────────────────────

// Scale UV around origin (0,0)
fn uvScale(uv: vec2f, scale: vec2f) -> vec2f {
    return uv * scale;
}

// Scale UV around center (0.5, 0.5) — AT pattern for texture zoom
fn uvScaleCenter(uv: vec2f, scale: vec2f) -> vec2f {
    return (uv - vec2f(0.5)) * scale + vec2f(0.5);
}

// Translate UV
fn uvTranslate(uv: vec2f, offset: vec2f) -> vec2f {
    return uv + offset;
}

// Rotate UV around origin by angle (radians)
fn uvRotate(uv: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(uv.x * c - uv.y * s,
                 uv.x * s + uv.y * c);
}

// Rotate UV around center (0.5, 0.5) — AT pattern for spinning textures
fn uvRotateCenter(uv: vec2f, angle: f32) -> vec2f {
    let centered = uv - vec2f(0.5);
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(centered.x * c - centered.y * s,
                 centered.x * s + centered.y * c) + vec2f(0.5);
}

// Flip UV axes
fn uvFlipX(uv: vec2f) -> vec2f { return vec2f(1.0 - uv.x, uv.y); }
fn uvFlipY(uv: vec2f) -> vec2f { return vec2f(uv.x, 1.0 - uv.y); }

// Tile UV (repeat) — AT uses for repeating patterns in Sky.fs
fn uvTile(uv: vec2f, tiles: vec2f) -> vec2f {
    return fract(uv * tiles);
}

// Tile with offset per row — AT brick pattern in Terrain.vs
fn uvTileOffset(uv: vec2f, tiles: vec2f, rowOffset: f32) -> vec2f {
    let scaled = uv * tiles;
    let row = floor(scaled.y);
    let offset = row * rowOffset;
    return fract(vec2f(scaled.x + offset, scaled.y));
}

// Aspect-ratio-correct UV — maps UV to preserve aspect in a quad
fn uvAspect(uv: vec2f, aspect: f32) -> vec2f {
    var result = uv - vec2f(0.5);
    if (aspect > 1.0) {
        result.x *= aspect;
    } else {
        result.y /= aspect;
    }
    return result + vec2f(0.5);
}

// Polar UV — cartesian (0..1, 0..1) → (angle/TAU, radius)
fn uvPolar(uv: vec2f) -> vec2f {
    let centered = uv - vec2f(0.5);
    let angle = atan2(centered.y, centered.x);
    let radius = length(centered) * 2.0;
    return vec2f(angle / (2.0 * AT_PI) + 0.5, radius);
}

// ── 2D rotation matrix ───────────────────────────────────────────────────────

fn rotate2D(angle: f32) -> mat2x2f {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2f(c, s, -s, c);
}

// ── 3D rotation matrices ─────────────────────────────────────────────────────
// AT compiled.vs uses these for particle spin / orbit in WorkDetailParticles,
// ProtonAntimatter 3D orbit, curl-aura field orientation.

fn rotationX(angle: f32) -> mat4x4f {
    let c = cos(angle);
    let s = sin(angle);
    return mat4x4f(
        1.0, 0.0, 0.0, 0.0,
        0.0,   c,   s, 0.0,
        0.0,  -s,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

fn rotationY(angle: f32) -> mat4x4f {
    let c = cos(angle);
    let s = sin(angle);
    return mat4x4f(
          c, 0.0,  -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
          s, 0.0,   c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

fn rotationZ(angle: f32) -> mat4x4f {
    let c = cos(angle);
    let s = sin(angle);
    return mat4x4f(
          c,   s, 0.0, 0.0,
         -s,   c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

// Rotation around arbitrary axis (Rodrigues' rotation formula)
// AT: used in ProtonAntimatter orbit axis, curl-flow-field arbitrary spin
fn rotationAxisAngle(axis: vec3f, angle: f32) -> mat4x4f {
    let a = normalize(axis);
    let c = cos(angle);
    let s = sin(angle);
    let t = 1.0 - c;
    return mat4x4f(
        t * a.x * a.x + c,         t * a.x * a.y + s * a.z,   t * a.x * a.z - s * a.y,  0.0,
        t * a.x * a.y - s * a.z,   t * a.y * a.y + c,         t * a.y * a.z + s * a.x,  0.0,
        t * a.x * a.z + s * a.y,   t * a.y * a.z - s * a.x,   t * a.z * a.z + c,        0.0,
        0.0,                        0.0,                        0.0,                       1.0
    );
}

// 3x3 rotation from Euler angles (ZYX order) — for normal/direction transforms
fn rotationEuler(pitch: f32, yaw: f32, roll: f32) -> mat3x3f {
    let cx = cos(pitch); let sx = sin(pitch);
    let cy = cos(yaw);   let sy = sin(yaw);
    let cz = cos(roll);  let sz = sin(roll);
    return mat3x3f(
        cy * cz,                    cy * sz,                   -sy,
        sx * sy * cz - cx * sz,     sx * sy * sz + cx * cz,    sx * cy,
        cx * sy * cz + sx * sz,     cx * sy * sz - sx * cz,    cx * cy
    );
}

// Apply mat4 rotation to vec3 position (ignores translation row)
fn applyRotation(m: mat4x4f, v: vec3f) -> vec3f {
    return (m * vec4f(v, 1.0)).xyz;
}

// Apply mat4 rotation to vec3 direction (w=0, no translation)
fn applyRotationDir(m: mat4x4f, v: vec3f) -> vec3f {
    return (m * vec4f(v, 0.0)).xyz;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 4  blendmodes.glsl → WGSL
//
// Photoshop blend modes — ported from lygia/color/blend/*.wgsl
// + Jamie Owen reference implementations.
// AT compiled.vs uses Add, Multiply, Screen, Overlay, SoftLight in
// WorkItemShader, WorkDetailParticles, WaterCeilingShader.
// ─────────────────────────────────────────────────────────────────────────────

export const WGSL_BLEND_MODES = /* wgsl */`
// ──────────────────────────────────────────────────────────────────────────────
// AT blendmodes.glsl — Photoshop blend modes (WGSL port)
// Naming: blend<Mode>  → scalar f32
//         blend<Mode>3 → vec3f (per-channel)
//         blend<Mode>3Opacity → vec3f with opacity compositing
// Ref: Jamie Owen, https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
//      lygia/color/blend/*.wgsl (Patricio Gonzalez Vivo, MIT)
// ──────────────────────────────────────────────────────────────────────────────

// ── Add ───────────────────────────────────────────────────────────────────────
fn blendAdd(base: f32, blend: f32) -> f32 { return min(base + blend, 1.0); }
fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f { return min(base + blend, vec3f(1.0)); }
fn blendAdd3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendAdd3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Subtract ──────────────────────────────────────────────────────────────────
fn blendSubtract(base: f32, blend: f32) -> f32 { return max(base + blend - 1.0, 0.0); }
fn blendSubtract3(base: vec3f, blend: vec3f) -> vec3f {
    return max(base + blend - vec3f(1.0), vec3f(0.0));
}
fn blendSubtract3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendSubtract3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Multiply ──────────────────────────────────────────────────────────────────
fn blendMultiply(base: f32, blend: f32) -> f32 { return base * blend; }
fn blendMultiply3(base: vec3f, blend: vec3f) -> vec3f { return base * blend; }
fn blendMultiply3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendMultiply3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Screen ────────────────────────────────────────────────────────────────────
fn blendScreen(base: f32, blend: f32) -> f32 {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}
fn blendScreen3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendScreen(base.r, blend.r),
                 blendScreen(base.g, blend.g),
                 blendScreen(base.b, blend.b));
}
fn blendScreen3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendScreen3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Overlay ───────────────────────────────────────────────────────────────────
fn blendOverlay(base: f32, blend: f32) -> f32 {
    if (base < 0.5) { return 2.0 * base * blend; }
    return 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
}
fn blendOverlay3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendOverlay(base.r, blend.r),
                 blendOverlay(base.g, blend.g),
                 blendOverlay(base.b, blend.b));
}
fn blendOverlay3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendOverlay3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Soft Light ────────────────────────────────────────────────────────────────
fn blendSoftLight(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) {
        return 2.0 * base * blend + base * base * (1.0 - 2.0 * blend);
    }
    return sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend);
}
fn blendSoftLight3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendSoftLight(base.r, blend.r),
                 blendSoftLight(base.g, blend.g),
                 blendSoftLight(base.b, blend.b));
}
fn blendSoftLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendSoftLight3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Hard Light ────────────────────────────────────────────────────────────────
// HardLight(B,L) = Overlay(L,B)  (roles swapped)
fn blendHardLight(base: f32, blend: f32) -> f32 { return blendOverlay(blend, base); }
fn blendHardLight3(base: vec3f, blend: vec3f) -> vec3f { return blendOverlay3(blend, base); }
fn blendHardLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendHardLight3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Darken ────────────────────────────────────────────────────────────────────
fn blendDarken(base: f32, blend: f32) -> f32 { return min(base, blend); }
fn blendDarken3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendDarken(base.r, blend.r),
                 blendDarken(base.g, blend.g),
                 blendDarken(base.b, blend.b));
}
fn blendDarken3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendDarken3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Lighten ───────────────────────────────────────────────────────────────────
fn blendLighten(base: f32, blend: f32) -> f32 { return max(base, blend); }
fn blendLighten3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendLighten(base.r, blend.r),
                 blendLighten(base.g, blend.g),
                 blendLighten(base.b, blend.b));
}
fn blendLighten3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendLighten3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Difference ────────────────────────────────────────────────────────────────
fn blendDifference(base: f32, blend: f32) -> f32 { return abs(base - blend); }
fn blendDifference3(base: vec3f, blend: vec3f) -> vec3f { return abs(base - blend); }
fn blendDifference3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendDifference3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Exclusion ─────────────────────────────────────────────────────────────────
fn blendExclusion(base: f32, blend: f32) -> f32 {
    return base + blend - 2.0 * base * blend;
}
fn blendExclusion3(base: vec3f, blend: vec3f) -> vec3f {
    return base + blend - 2.0 * base * blend;
}
fn blendExclusion3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendExclusion3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Color Dodge ───────────────────────────────────────────────────────────────
fn blendColorDodge(base: f32, blend: f32) -> f32 {
    return select(min(base / (1.0 - blend), 1.0), 1.0, blend >= 1.0);
}
fn blendColorDodge3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendColorDodge(base.r, blend.r),
                 blendColorDodge(base.g, blend.g),
                 blendColorDodge(base.b, blend.b));
}
fn blendColorDodge3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendColorDodge3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Color Burn ────────────────────────────────────────────────────────────────
fn blendColorBurn(base: f32, blend: f32) -> f32 {
    return select(max(1.0 - (1.0 - base) / blend, 0.0), 0.0, blend <= 0.0);
}
fn blendColorBurn3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendColorBurn(base.r, blend.r),
                 blendColorBurn(base.g, blend.g),
                 blendColorBurn(base.b, blend.b));
}
fn blendColorBurn3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendColorBurn3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Linear Burn ───────────────────────────────────────────────────────────────
fn blendLinearBurn(base: f32, blend: f32) -> f32 { return max(base + blend - 1.0, 0.0); }
fn blendLinearBurn3(base: vec3f, blend: vec3f) -> vec3f {
    return max(base + blend - vec3f(1.0), vec3f(0.0));
}
fn blendLinearBurn3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendLinearBurn3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Linear Dodge (Add) ────────────────────────────────────────────────────────
fn blendLinearDodge(base: f32, blend: f32) -> f32 { return min(base + blend, 1.0); }
fn blendLinearDodge3(base: vec3f, blend: vec3f) -> vec3f {
    return min(base + blend, vec3f(1.0));
}
fn blendLinearDodge3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendLinearDodge3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Vivid Light ───────────────────────────────────────────────────────────────
fn blendVividLight(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) { return blendColorBurn(base, 2.0 * blend); }
    return blendColorDodge(base, 2.0 * (blend - 0.5));
}
fn blendVividLight3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendVividLight(base.r, blend.r),
                 blendVividLight(base.g, blend.g),
                 blendVividLight(base.b, blend.b));
}
fn blendVividLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendVividLight3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Pin Light ─────────────────────────────────────────────────────────────────
fn blendPinLight(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) { return blendDarken(base, 2.0 * blend); }
    return blendLighten(base, 2.0 * (blend - 0.5));
}
fn blendPinLight3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendPinLight(base.r, blend.r),
                 blendPinLight(base.g, blend.g),
                 blendPinLight(base.b, blend.b));
}
fn blendPinLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendPinLight3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Hard Mix ──────────────────────────────────────────────────────────────────
fn blendHardMix(base: f32, blend: f32) -> f32 {
    return step(1.0 - base, blend);
}
fn blendHardMix3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendHardMix(base.r, blend.r),
                 blendHardMix(base.g, blend.g),
                 blendHardMix(base.b, blend.b));
}
fn blendHardMix3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendHardMix3(base, blend) * opacity + base * (1.0 - opacity);
}

// ── Reflect ───────────────────────────────────────────────────────────────────
fn blendReflect(base: f32, blend: f32) -> f32 {
    return select(min(base * base / (1.0 - blend), 1.0), 1.0, blend >= 1.0);
}
fn blendReflect3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendReflect(base.r, blend.r),
                 blendReflect(base.g, blend.g),
                 blendReflect(base.b, blend.b));
}
fn blendReflect3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendReflect3(base, blend) * opacity + base * (1.0 - opacity);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Merged export — AT_SHADER_UTILS_WGSL
//
// Drop-in for a WGSL shader that needs all three libraries.
// Order: RANGE (no deps) → EASES (no deps) → BLEND_MODES (no deps)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Combined WGSL string: range.glsl + eases.glsl + UV transforms + blendmodes.glsl
 *
 * Usage in a WebGPU compute/render shader:
 *
 *   ```ts
 *   import { AT_SHADER_UTILS_WGSL } from '$lib/sph/at-shader-utils.ts';
 *
 *   const code = /* wgsl *\/ `
 *     ${AT_SHADER_UTILS_WGSL}
 *
 *     @compute @workgroup_size(64)
 *     fn main(@builtin(global_invocation_id) gid: vec3u) {
 *       let t   = f32(gid.x) / 1024.0;
 *       let pos = range(t, 0.0, 1.0, -5.0, 5.0);
 *       let ease = easeOutCubic(t);
 *       let uv  = uvRotateCenter(vec2f(t, 0.5), 0.785);
 *       let rot = rotationY(t * AT_PI);
 *       // ...
 *     }
 *   `;
 *   ```
 *
 * Or inject only what you need:
 *   ```ts
 *   import { WGSL_RANGE, WGSL_EASES, WGSL_UV_TRANSFORMS, WGSL_BLEND_MODES } from '$lib/sph/at-shader-utils.ts';
 *   ```
 */
export const AT_SHADER_UTILS_WGSL: string = [
  WGSL_RANGE,
  WGSL_EASES,
  WGSL_UV_TRANSFORMS,
  WGSL_BLEND_MODES,
].join('\n');

// ─────────────────────────────────────────────────────────────────────────────
// § 6  TypeScript mirrors — same functions available CPU-side
//
// Used by tween-system.ts, proton-controller.ts, physics-animation.ts etc.
// Kept in sync with the WGSL implementations above so CPU preview / unit
// tests always match GPU behaviour.
// ─────────────────────────────────────────────────────────────────────────────

// ── range / crange ────────────────────────────────────────────────────────────

/** Linear remap: map v from [iMin, iMax] → [oMin, oMax] */
export function range(v: number, iMin: number, iMax: number, oMin: number, oMax: number): number {
  return oMin + (oMax - oMin) * (v - iMin) / (iMax - iMin);
}

/** Clamped remap */
export function crange(v: number, iMin: number, iMax: number, oMin: number, oMax: number): number {
  const lo = Math.min(oMin, oMax);
  const hi = Math.max(oMin, oMax);
  return Math.max(lo, Math.min(hi, range(v, iMin, iMax, oMin, oMax)));
}

/** Normalize v from [iMin, iMax] → [0, 1] */
export function rangeNorm(v: number, iMin: number, iMax: number): number {
  return (v - iMin) / (iMax - iMin);
}

/** Ping-pong mirror remap */
export function rangeMirror(v: number, iMin: number, iMax: number, oMin: number, oMax: number): number {
  const span = iMax - iMin;
  const t    = v - iMin;
  const cycle = Math.abs(t - 2 * span * Math.floor((t + span) / (2 * span)));
  return oMin + (oMax - oMin) * cycle / span;
}

// ── Easing functions (CPU mirrors) ────────────────────────────────────────────

export type EasingFn = (t: number) => number;

export const ATEasing = {
  linear:        (t: number) => t,

  // Quadratic
  easeInQuad:    (t: number) => t * t,
  easeOutQuad:   (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

  // Cubic
  easeInCubic:    (t: number) => t * t * t,
  easeOutCubic:   (t: number) => { const u = t - 1; return u * u * u + 1; },
  easeInOutCubic: (t: number) => {
    const s = t * 2;
    if (s < 1) return 0.5 * s * s * s;
    const u = s - 2; return 0.5 * (u * u * u + 2);
  },

  // Quartic
  easeInQuart:    (t: number) => t * t * t * t,
  easeOutQuart:   (t: number) => { const u = t - 1; return -(u * u * u * u - 1); },
  easeInOutQuart: (t: number) => {
    const s = t * 2;
    if (s < 1) return 0.5 * s * s * s * s;
    const u = s - 2; return -0.5 * (u * u * u * u - 2);
  },

  // Quintic
  easeInQuint:    (t: number) => t * t * t * t * t,
  easeOutQuint:   (t: number) => { const u = t - 1; return u * u * u * u * u + 1; },
  easeInOutQuint: (t: number) => {
    const s = t * 2;
    if (s < 1) return 0.5 * s * s * s * s * s;
    const u = s - 2; return 0.5 * (u * u * u * u * u + 2);
  },

  // Sine
  easeInSine:    (t: number) => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine:   (t: number) => Math.sin(t * Math.PI / 2),
  easeInOutSine: (t: number) => -0.5 * (Math.cos(Math.PI * t) - 1),

  // Expo
  easeInExpo:    (t: number) => t === 0 ? 0 : Math.pow(2, 10 * t - 10),
  easeOutExpo:   (t: number) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: (t: number) => {
    if (t === 0) return 0; if (t === 1) return 1;
    const s = t * 2;
    if (s < 1) return 0.5 * Math.pow(2, 10 * s - 10);
    return 0.5 * (2 - Math.pow(2, -10 * (s - 1)));
  },

  // Circ
  easeInCirc:    (t: number) => 1 - Math.sqrt(Math.max(1 - t * t, 0)),
  easeOutCirc:   (t: number) => { const u = t - 1; return Math.sqrt(Math.max(1 - u * u, 0)); },
  easeInOutCirc: (t: number) => {
    const s = t * 2;
    if (s < 1) return -0.5 * (Math.sqrt(Math.max(1 - s * s, 0)) - 1);
    const u = s - 2; return 0.5 * (Math.sqrt(Math.max(1 - u * u, 0)) + 1);
  },

  // Back
  easeInBack:    (t: number) => { const c1 = 1.70158, c3 = c1 + 1; return c3 * t * t * t - c1 * t * t; },
  easeOutBack:   (t: number) => { const c1 = 1.70158, c3 = c1 + 1; const u = t - 1; return 1 + c3 * u * u * u + c1 * u * u; },
  easeInOutBack: (t: number) => {
    const c2 = 2.5949095, c4 = c2 + 1;
    const s = t * 2;
    if (s < 1) return 0.5 * (s * s * ((c4 + 1) * s - c4));
    const u = s - 2; return 0.5 * (u * u * ((c4 + 1) * u + c4) + 2);
  },

  // Elastic
  easeInElastic: (t: number) => {
    if (t === 0 || t === 1) return t;
    const c4 = (2 * Math.PI) / 3;
    return -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
  },
  easeOutElastic: (t: number) => {
    if (t === 0 || t === 1) return t;
    const c4 = (2 * Math.PI) / 3;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
  easeInOutElastic: (t: number) => {
    if (t === 0 || t === 1) return t;
    const c5 = (2 * Math.PI) / 4.5;
    const s = t * 2;
    if (s < 1) return -0.5 * Math.pow(2, 10 * s - 10) * Math.sin((s * 10 - 11.125) * c5);
    return Math.pow(2, -10 * (s - 1)) * Math.sin((s * 10 - 11.125) * c5) * 0.5 + 1;
  },

  // Bounce
  easeOutBounce: (t: number): number => {
    const n1 = 7.5625, d1 = 2.75;
    if (t < 1 / d1)        return n1 * t * t;
    if (t < 2 / d1)        { t -= 1.5 / d1;  return n1 * t * t + 0.75; }
    if (t < 2.5 / d1)      { t -= 2.25 / d1; return n1 * t * t + 0.9375; }
                             t -= 2.625 / d1; return n1 * t * t + 0.984375;
  },
  easeInBounce: (t: number) => {
    const n1 = 7.5625, d1 = 2.75;
    let u = 1 - t;
    if (u < 1 / d1)        return 1 - n1 * u * u;
    if (u < 2 / d1)        { u -= 1.5 / d1;  return 1 - (n1 * u * u + 0.75); }
    if (u < 2.5 / d1)      { u -= 2.25 / d1; return 1 - (n1 * u * u + 0.9375); }
                             u -= 2.625 / d1; return 1 - (n1 * u * u + 0.984375);
  },
  easeInOutBounce: (t: number) => {
    const n1 = 7.5625, d1 = 2.75;
    if (t < 0.5) {
      let u = 1 - 2 * t;
      if (u < 1 / d1)      return (1 - n1 * u * u) * 0.5;
      if (u < 2 / d1)      { u -= 1.5 / d1;  return (1 - n1 * u * u - 0.75) * 0.5; }
      if (u < 2.5 / d1)    { u -= 2.25 / d1; return (1 - n1 * u * u - 0.9375) * 0.5; }
                             u -= 2.625 / d1; return (1 - n1 * u * u - 0.984375) * 0.5;
    }
    let v = 2 * t - 1;
    if (v < 1 / d1)        return (1 + n1 * v * v) * 0.5;
    if (v < 2 / d1)        { v -= 1.5 / d1;  return (1 + n1 * v * v + 0.75) * 0.5; }
    if (v < 2.5 / d1)      { v -= 2.25 / d1; return (1 + n1 * v * v + 0.9375) * 0.5; }
                             v -= 2.625 / d1; return (1 + n1 * v * v + 0.984375) * 0.5;
  },

  // Smooth-step kernels
  smoothstep:   (t: number) => t * t * (3 - 2 * t),
  smootherstep: (t: number) => t * t * t * (t * (t * 6 - 15) + 10),
} as const satisfies Record<string, EasingFn>;

export type ATEasingName = keyof typeof ATEasing;

// ── UV transforms (CPU mirrors) ──────────────────────────────────────────────

export function uvScale(uv: [number, number], scale: [number, number]): [number, number] {
  return [uv[0] * scale[0], uv[1] * scale[1]];
}

export function uvScaleCenter(uv: [number, number], scale: [number, number]): [number, number] {
  return [(uv[0] - 0.5) * scale[0] + 0.5, (uv[1] - 0.5) * scale[1] + 0.5];
}

export function uvTranslate(uv: [number, number], offset: [number, number]): [number, number] {
  return [uv[0] + offset[0], uv[1] + offset[1]];
}

export function uvRotate(uv: [number, number], angle: number): [number, number] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [uv[0] * c - uv[1] * s, uv[0] * s + uv[1] * c];
}

export function uvRotateCenter(uv: [number, number], angle: number): [number, number] {
  const cx = uv[0] - 0.5, cy = uv[1] - 0.5;
  const c = Math.cos(angle), s = Math.sin(angle);
  return [cx * c - cy * s + 0.5, cx * s + cy * c + 0.5];
}

export function uvTile(uv: [number, number], tiles: [number, number]): [number, number] {
  const fract = (v: number) => v - Math.floor(v);
  return [fract(uv[0] * tiles[0]), fract(uv[1] * tiles[1])];
}

/** 2D rotation matrix as flat [a, b, c, d] → [[a,b],[c,d]] column-major */
export function rotate2D(angle: number): [number, number, number, number] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c, s, -s, c];
}

/** 4×4 rotation matrix as flat 16-element array (column-major, WebGPU convention) */
export type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

export function rotationX(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
}

export function rotationY(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];
}

export function rotationZ(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1];
}

export function rotationAxisAngle(axis: [number, number, number], angle: number): Mat4 {
  const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
  const ax = axis[0] / len, ay = axis[1] / len, az = axis[2] / len;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t*ax*ax+c,      t*ax*ay+s*az,  t*ax*az-s*ay, 0,
    t*ax*ay-s*az,   t*ay*ay+c,     t*ay*az+s*ax, 0,
    t*ax*az+s*ay,   t*ay*az-s*ax,  t*az*az+c,    0,
    0,              0,             0,              1,
  ];
}
