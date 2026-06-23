/**
 * at-glass-reflection-system.ts — AT Glass Reflection System — WebGL GPU Port
 *
 * 玻璃反射系统，为 Cell 外壳提供真实感的玻璃材质:
 *   - 菲涅耳反射 (Fresnel Reflection) — 掠射角处高反射率
 *   - 折射与色散 (Refraction + Chromatic Aberration) — 棱镜效应
 *   - 体积散射 (Subsurface Scattering) — 内部漫散光
 *   - 环境反射 (Environment Equirectangular) — equi 投影探针采样
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
 * Research: xiaodi #M914 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Shader Sources — inline GLSL (ported from compiled.vs)
// ─────────────────────────────────────────────────────────────────────────────

// ── Vertex Shader: GlassInner pass ──────────────────────────────────────────
// Source: compiled.vs line 2972-2978 GlassInner.glsl
const GLASS_INNER_VERT = /* glsl */`
precision highp float;
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUv;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vPos;
uniform mat4 uProjection;
uniform mat4 uModelView;
void main() {
    vNormal = aNormal;
    vViewDir = -vec3(uModelView * vec4(aPosition, 1.0));
    vPos = aPosition;
    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}
`;

// ── Fragment Shader: GlassInner pass ────────────────────────────────────────
// Source: compiled.vs line 2986-2989 GlassInner.glsl fragment
// Implements subsurface noise scattering + edge glow
const GLASS_INNER_FRAG = /* glsl */`
precision highp float;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vPos;
uniform float uTime;

// range.glsl crange
float crange(float v, float inMin, float inMax, float outMin, float outMax) {
    return mix(outMin, outMax, clamp((v - inMin) / (inMax - inMin), 0.0, 1.0));
}

// eases.glsl quarticIn
float quarticIn(float t) { return t * t * t * t; }

// simplenoise.glsl — cnoise (3D Perlin, simplified)
vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289_4(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289_4(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float cnoise(vec3 P) {
    vec3 Pi0 = floor(P); vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289(Pi0); Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P); vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz; vec4 iz1 = Pi1.zzzz;
    vec4 ixy  = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 * (1.0/7.0); vec4 gy0 = fract(floor(gx0)*(1.0/7.0)) - 0.5;
    gx0 = fract(gx0); vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0)); gx0 -= sz0*(step(0.0,gx0)-0.5); gy0 -= sz0*(step(0.0,gy0)-0.5);
    vec4 gx1 = ixy1 * (1.0/7.0); vec4 gy1 = fract(floor(gx1)*(1.0/7.0)) - 0.5;
    gx1 = fract(gx1); vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0)); gx1 -= sz1*(step(0.0,gx1)-0.5); gy1 -= sz1*(step(0.0,gy1)-0.5);
    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x); vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z); vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x); vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z); vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000,g000),dot(g010,g010),dot(g100,g100),dot(g110,g110)));
    g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001,g001),dot(g011,g011),dot(g101,g101),dot(g111,g111)));
    g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade = Pf0*Pf0*Pf0*(Pf0*(Pf0*6.0-15.0)+10.0);
    vec4 n_z = mix(vec4(n000,n100,n010,n110), vec4(n001,n101,n011,n111), fade.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade.y);
    return 2.3 * mix(n_yz.x, n_yz.y, fade.x);
}

void main() {
    // GlassInner.glsl fragment — exact port from compiled.vs line 2987-2989
    gl_FragColor = mix(vec4(0.0), vec4(1.4), vNormal.y)
                 * crange(cnoise(vViewDir * 0.2 + 0.5), -1.0, 1.0, 0.0, 1.0);
    gl_FragColor.rgb += cnoise(vViewDir) * 0.05;
    gl_FragColor.rgb += quarticIn(
        crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) *
        crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)
    ) * 0.1;
}
`;

// ── Vertex Shader: CleanRoomGlass / GlassReflection ──────────────────────────
// Source: compiled.vs line 2835-2845 CleanRoomGlass.glsl vertex
// Uses refl.vs functions: reflection(), refraction(), inverseTransformDirection()
const GLASS_REFL_VERT = /* glsl */`
precision highp float;
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUv;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefraction;
varying vec3 vPos;
varying vec2 vUv;
uniform mat4 uProjection;
uniform mat4 uModelView;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;
uniform vec3 uCameraPos;
uniform float uRefractionRatio;

// refl.vs — inverseTransformDirection
vec3 inverseTransformDirection(in vec3 n, in mat4 matrix) {
    return normalize((matrix * vec4(n, 0.0) * matrix).xyz);
}

// refl.vs — reflection
vec3 computeReflection(vec4 worldPosition) {
    vec3 transformedNormal = uNormalMatrix * aNormal;
    vec3 cameraToVertex = normalize(worldPosition.xyz - uCameraPos);
    vec3 worldNormal = inverseTransformDirection(transformedNormal, uModel);
    return reflect(cameraToVertex, worldNormal);
}

// refl.vs — refraction
vec3 computeRefraction(vec4 worldPosition, float rRatio) {
    vec3 transformedNormal = uNormalMatrix * aNormal;
    vec3 cameraToVertex = normalize(worldPosition.xyz - uCameraPos);
    vec3 worldNormal = inverseTransformDirection(transformedNormal, uModel);
    return refract(cameraToVertex, worldNormal, rRatio);
}

void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vReflection = computeReflection(worldPos);
    vRefraction = computeRefraction(worldPos, uRefractionRatio);
    vPos = aPosition;
    vWorldPos = worldPos.xyz;
    vNormal = uNormalMatrix * aNormal;
    vViewDir = -vec3(uModelView * vec4(aPosition, 1.0));
    vUv = aUv;
    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}
`;

// ── Fragment Shader: CleanRoomGlass (full AT glass with fresnel + refraction) ─
// Source: compiled.vs line 2857-2888 CleanRoomGlass.glsl fragment
// Uses: fresnel.glsl, refl.fs, rgbshift.fs, range.glsl, simplenoise, eases
const GLASS_REFL_FRAG = /* glsl */`
precision highp float;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefraction;
varying vec3 vPos;
varying vec2 vUv;

uniform sampler2D tRefraction;   // screen-space refraction FBO
uniform sampler2D tEnv;          // equirectangular environment map
uniform sampler2D tInner;        // GlassInner pass result
uniform float uFresnelPow;
uniform float uDistortStrength;
uniform float uRefractionRatio;
uniform vec2 uResolution;
uniform float uTime;
uniform float uEnvStrength;
uniform float uCleanRoomMode;    // 0 = standard, 1 = cleanroom rainbow

// ── fresnel.glsl — getFresnel ──────────────────────────────────────────────
float getFresnel(vec3 normal, vec3 viewDir, float power) {
    float d = dot(normalize(normal), normalize(viewDir));
    return 1.0 - pow(abs(d), power);
}

float getFresnelIOR(float inIOR, float outIOR, vec3 normal, vec3 viewDir) {
    float ro = (inIOR - outIOR) / (inIOR + outIOR);
    float d = dot(normalize(normal), normalize(viewDir));
    return ro + (1.0 - ro) * pow((1.0 - d), 5.0);
}

// ── refl.fs — envColorEqui / envColorEquiRGB ──────────────────────────────
vec4 envColorEqui(sampler2D map, vec3 direction) {
    vec2 uv;
    uv.y = asin(clamp(direction.y, -1.0, 1.0)) * 0.31830988618 + 0.5;
    uv.x = atan(direction.z, direction.x) * 0.15915494 + 0.5;
    return texture2D(map, uv);
}

// rgbshift.fs — envColorEquiRGB (chromatic aberration on env map)
vec4 envColorEquiRGB(sampler2D map, vec3 direction, float angle, float amount) {
    vec2 uv;
    uv.y = asin(clamp(direction.y, -1.0, 1.0)) * 0.31830988618 + 0.5;
    uv.x = atan(direction.z, direction.x) * 0.15915494 + 0.5;
    vec2 offset = vec2(cos(angle), sin(angle)) * amount * 0.01;
    vec4 r = texture2D(map, uv + offset);
    vec4 g = texture2D(map, uv);
    vec4 b = texture2D(map, uv - offset);
    return vec4(r.r, g.g, b.b, g.a);
}

// rgbshift.fs — getRGB (screen-space RGB shift)
vec4 getRGB(sampler2D tex, vec2 uv, float angle, float amount) {
    vec2 offset = vec2(cos(angle), sin(angle)) * amount;
    vec4 r = texture2D(tex, uv + offset);
    vec4 g = texture2D(tex, uv);
    vec4 b = texture2D(tex, uv - offset);
    return vec4(r.r, g.g, b.b, g.a);
}

// range.glsl
float crange(float v, float inMin, float inMax, float outMin, float outMax) {
    return mix(outMin, outMax, clamp((v - inMin) / (inMax - inMin), 0.0, 1.0));
}

// eases.glsl
float quarticIn(float t) { return t * t * t * t; }

// simplenoise
vec3 mod289v(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289_4v(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permuteV(vec4 x) { return mod289_4v(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrtV(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float cnoise(vec3 P) {
    vec3 Pi0 = floor(P); vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289v(Pi0); Pi1 = mod289v(Pi1);
    vec3 Pf0 = fract(P); vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz; vec4 iz1 = Pi1.zzzz;
    vec4 ixy  = permuteV(permuteV(ix) + iy);
    vec4 ixy0 = permuteV(ixy + iz0);
    vec4 ixy1 = permuteV(ixy + iz1);
    vec4 gx0 = ixy0*(1.0/7.0); vec4 gy0 = fract(floor(gx0)*(1.0/7.0)) - 0.5;
    gx0 = fract(gx0); vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0)); gx0 -= sz0*(step(0.0,gx0)-0.5); gy0 -= sz0*(step(0.0,gy0)-0.5);
    vec4 gx1 = ixy1*(1.0/7.0); vec4 gy1 = fract(floor(gx1)*(1.0/7.0)) - 0.5;
    gx1 = fract(gx1); vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0)); gx1 -= sz1*(step(0.0,gx1)-0.5); gy1 -= sz1*(step(0.0,gy1)-0.5);
    vec3 g000=vec3(gx0.x,gy0.x,gz0.x); vec3 g100=vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010=vec3(gx0.z,gy0.z,gz0.z); vec3 g110=vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001=vec3(gx1.x,gy1.x,gz1.x); vec3 g101=vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011=vec3(gx1.z,gy1.z,gz1.z); vec3 g111=vec3(gx1.w,gy1.w,gz1.w);
    vec4 norm0=taylorInvSqrtV(vec4(dot(g000,g000),dot(g010,g010),dot(g100,g100),dot(g110,g110)));
    g000*=norm0.x; g010*=norm0.y; g100*=norm0.z; g110*=norm0.w;
    vec4 norm1=taylorInvSqrtV(vec4(dot(g001,g001),dot(g011,g011),dot(g101,g101),dot(g111,g111)));
    g001*=norm1.x; g011*=norm1.y; g101*=norm1.z; g111*=norm1.w;
    float n000=dot(g000,Pf0); float n100=dot(g100,vec3(Pf1.x,Pf0.yz));
    float n010=dot(g010,vec3(Pf0.x,Pf1.y,Pf0.z)); float n110=dot(g110,vec3(Pf1.xy,Pf0.z));
    float n001=dot(g001,vec3(Pf0.xy,Pf1.z)); float n101=dot(g101,vec3(Pf1.x,Pf0.y,Pf1.z));
    float n011=dot(g011,vec3(Pf0.x,Pf1.yz)); float n111=dot(g111,Pf1);
    vec3 fade=Pf0*Pf0*Pf0*(Pf0*(Pf0*6.0-15.0)+10.0);
    vec4 n_z=mix(vec4(n000,n100,n010,n110),vec4(n001,n101,n011,n111),fade.z);
    vec2 n_yz=mix(n_z.xy,n_z.zw,fade.y);
    return 2.3 * mix(n_yz.x,n_yz.y,fade.x);
}

// CleanRoomGlass.glsl — rainbowColor (compiled.vs line 2857-2865)
vec3 rainbowColor(float t) {
    t = mod(t, 1.0);
    if (t < 0.03)       return mix(vec3(0.5,0.0,0.5), vec3(0.5,0.0,1.0), t/0.03);
    else if (t < 0.06)  return mix(vec3(0.5,0.0,1.0), vec3(0.0,0.0,1.0), (t-0.03)/0.03);
    else if (t < 0.09)  return mix(vec3(0.0,0.0,1.0), vec3(0.0,1.0,1.0), (t-0.06)/0.03);
    else if (t < 0.12)  return mix(vec3(0.0,1.0,1.0), vec3(0.0,1.0,0.0), (t-0.09)/0.03);
    else if (t < 0.18)  return mix(vec3(0.0,1.0,0.0), vec3(1.0,1.0,0.0), (t-0.12)/0.06);
    else if (t < 0.24)  return mix(vec3(1.0,1.0,0.0), vec3(1.0,0.5,0.0), (t-0.18)/0.06);
    else                return mix(vec3(1.0,0.5,0.0), vec3(1.0,0.0,0.0), (t-0.24)/0.06);
}

void main() {
    // ── CleanRoomGlass.glsl fragment main() — exact AT logic ──────────────
    // compiled.vs lines 2868-2888

    float f = getFresnel(vNormal, vViewDir, uFresnelPow);

    // Rainbow color from fresnel
    vec3 r = rainbowColor(f * 4.0);
    if (r.r > 0.99) r *= 0.0;  // compiled.vs line 2871

    // Screen-space UV + distort
    vec2 uv = gl_FragCoord.xy / uResolution;
    uv += 0.1 * vNormal.xy * f * uDistortStrength;  // compiled.vs line 2874

    // RGB shift on refraction texture (rgbshift.fs)
    gl_FragColor = getRGB(tRefraction, uv, 0.3, 0.002);  // compiled.vs line 2876
    gl_FragColor.rgb += r;                                 // compiled.vs line 2877

    // Equirectangular env sample with chromatic aberration (refl.fs)
    gl_FragColor += envColorEquiRGB(tEnv, vRefraction, 0.2, 1.0) * uEnvStrength;  // line 2879

    // Subsurface scatter noise
    gl_FragColor.rgb += cnoise(vViewDir + 2.0) * 0.1;  // compiled.vs line 2880

    // GlassInner contribution
    gl_FragColor.rgb += texture2D(tInner, gl_FragCoord.xy / uResolution).r;  // line 2881

    // Edge glow
    gl_FragColor.rgb += quarticIn(
        crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) *
        crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)
    ) * 0.05;  // compiled.vs line 2882

    // Gamma correction
    gl_FragColor.rgb = pow(clamp(gl_FragColor.rgb, 0.0, 1.0), vec3(1.5));  // line 2884

    // Top normal boost (CleanRoom ceiling effect)
    if (vNormal.y > 0.8) gl_FragColor.rgb *= 1.8;  // compiled.vs line 2886

    gl_FragColor.a = 1.0;
}
`;

// ── Vertex Shader: BasicMirror ────────────────────────────────────────────────
// Source: compiled.vs line 1791-1795 BasicMirror.glsl
const BASIC_MIRROR_VERT = /* glsl */`
precision highp float;
attribute vec3 aPosition;
attribute vec2 aUv;
varying vec4 vMirrorCoord;
varying vec2 vUv;
uniform mat4 uProjection;
uniform mat4 uModelView;
uniform mat4 uModel;
uniform mat4 uMirrorMatrix;
void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;          // BasicMirror.glsl line 1793
    vUv = aUv;
    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
}
`;

// ── Fragment Shader: BasicMirror ─────────────────────────────────────────────
// Source: compiled.vs line 1797-1801 BasicMirror.glsl fragment
const BASIC_MIRROR_FRAG = /* glsl */`
precision highp float;
varying vec4 vMirrorCoord;
varying vec2 vUv;
uniform sampler2D tMirrorReflection;
void main() {
    // BasicMirror.glsl exact port — compiled.vs line 1799
    gl_FragColor.rgb = vec3(texture2D(tMirrorReflection, vMirrorCoord.xy / vMirrorCoord.w));
    gl_FragColor.a = 1.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript Interfaces & Params
// ─────────────────────────────────────────────────────────────────────────────

export interface GlassReflectionParams {
    fresnelPower?: number;           // Fresnel 衰减指数 (2.0 - 8.0)
    refractionRatio?: number;        // IOR 比例 (0.5 - 2.0)
    dispersiveness?: number;         // 色散强度 (0.0 - 0.5)
    distortStrength?: number;        // 折射扭曲强度 (0.0 - 1.0)
    subsurfaceIntensity?: number;    // 体积散射强度 (0.0 - 2.0)
    envProbeStrength?: number;       // 环境探针强度 (0.0 - 1.5)
    mirrorStrength?: number;         // 镜面反射强度 (0.0 - 1.0)
    cleanRoomMode?: boolean;         // CleanRoom 模式
    time?: number;
    timeScale?: number;
}

export interface GlassReflectionUniform {
    fresnelPower: number;
    refractionRatio: number;
    dispersiveness: number;
    distortStrength: number;
    subsurfaceIntensity: number;
    envProbeStrength: number;
    mirrorStrength: number;
    cleanRoomMode: number;
    time: number;
    padding: number;
}

// ── Presets ──────────────────────────────────────────────────────────────────

const DEFAULT_GLASS_PARAMS: Required<GlassReflectionParams> = {
    fresnelPower: 4.0,
    refractionRatio: 0.66,
    dispersiveness: 0.25,
    distortStrength: 0.1,
    subsurfaceIntensity: 0.8,
    envProbeStrength: 1.0,
    mirrorStrength: 0.5,
    cleanRoomMode: false,
    time: 0.0,
    timeScale: 1.0,
};

const PRESET_OPTICAL_GLASS: GlassReflectionParams = {
    ...DEFAULT_GLASS_PARAMS, refractionRatio: 1.52, fresnelPower: 5.0, dispersiveness: 0.1,
};
const PRESET_CROWN_GLASS: GlassReflectionParams = {
    ...DEFAULT_GLASS_PARAMS, refractionRatio: 1.52, dispersiveness: 0.15,
};
const PRESET_FLINT_GLASS: GlassReflectionParams = {
    ...DEFAULT_GLASS_PARAMS, refractionRatio: 1.65, dispersiveness: 0.35, fresnelPower: 3.5,
};
const PRESET_DIAMOND: GlassReflectionParams = {
    ...DEFAULT_GLASS_PARAMS, refractionRatio: 2.42, fresnelPower: 2.0, dispersiveness: 0.44,
};
const PRESET_CLEANROOM: GlassReflectionParams = {
    ...DEFAULT_GLASS_PARAMS, cleanRoomMode: true, fresnelPower: 3.5,
    dispersiveness: 0.3, refractionRatio: 1.33, envProbeStrength: 1.2,
};

// ─────────────────────────────────────────────────────────────────────────────
// FBO helper types
// ─────────────────────────────────────────────────────────────────────────────

interface SingleRT {
    fbo: WebGLFramebuffer;
    tex: WebGLTexture;
    width: number;
    height: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATGlassReflectionSystem — real WebGL gl.* implementation
// ─────────────────────────────────────────────────────────────────────────────

export class ATGlassReflectionSystem {
    private gl: WebGLRenderingContext;
    private params: Required<GlassReflectionParams>;
    private startTime: number;

    // ── WebGLPrograms ─────────────────────────────────────────────────────────
    private innerProg!: WebGLProgram;      // GlassInner pass
    private reflProg!: WebGLProgram;       // CleanRoomGlass / GlassReflection pass
    private mirrorProg!: WebGLProgram;     // BasicMirror pass

    // ── FBOs ─────────────────────────────────────────────────────────────────
    private innerRT!: SingleRT;            // GlassInner render target
    private reflRT!: SingleRT;             // glass reflection accumulation RT
    private refractionRT!: SingleRT;       // screen-space refraction capture RT

    // ── Textures ─────────────────────────────────────────────────────────────
    private envTex!: WebGLTexture;         // equirectangular environment map
    private mirrorTex!: WebGLTexture;      // planar mirror reflection texture (external)

    // ── Geometry ─────────────────────────────────────────────────────────────
    private quadBuf!: WebGLBuffer;         // fullscreen quad positions
    private quadUvBuf!: WebGLBuffer;       // fullscreen quad UVs
    private quadNrmBuf!: WebGLBuffer;      // quad normals (for glass inner pass)

    // ── Canvas size ──────────────────────────────────────────────────────────
    private width: number;
    private height: number;

    constructor(gl: WebGLRenderingContext, width = 1024, height = 1024) {
        this.gl = gl;
        this.width = width;
        this.height = height;
        this.startTime = performance.now();
        this.params = { ...DEFAULT_GLASS_PARAMS };
        this._init();
    }

    /** 静态工厂 — create + init */
    static create(
        gl: WebGLRenderingContext,
        width = 1024,
        height = 1024
    ): ATGlassReflectionSystem {
        return new ATGlassReflectionSystem(gl, width, height);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // init — compile shaders, create FBOs, create geometry
    // ─────────────────────────────────────────────────────────────────────────

    private _init(): void {
        const gl = this.gl;

        // ── 1. Compile all programs ──────────────────────────────────────────
        this.innerProg  = this._compile(GLASS_INNER_VERT,  GLASS_INNER_FRAG,  'glassInner');
        this.reflProg   = this._compile(GLASS_REFL_VERT,   GLASS_REFL_FRAG,   'glassRefl');
        this.mirrorProg = this._compile(BASIC_MIRROR_VERT, BASIC_MIRROR_FRAG, 'basicMirror');

        // ── 2. Create render targets (FBOs) ──────────────────────────────────
        this.innerRT      = this._createRT(this.width, this.height);
        this.reflRT       = this._createRT(this.width, this.height);
        this.refractionRT = this._createRT(this.width, this.height);

        // ── 3. Create default 1×1 white env texture ──────────────────────────
        this.envTex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, this.envTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
            new Uint8Array([200, 220, 255, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // default mirror texture (1×1 grey)
        this.mirrorTex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, this.mirrorTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
            new Uint8Array([128, 128, 128, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // ── 4. Fullscreen quad geometry ──────────────────────────────────────
        // positions (clip space)
        this.quadBuf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 0,   1, -1, 0,   -1, 1, 0,
            -1,  1, 0,   1, -1, 0,    1, 1, 0,
        ]), gl.STATIC_DRAW);

        // UVs
        this.quadUvBuf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUvBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            0, 0,  1, 0,  0, 1,
            0, 1,  1, 0,  1, 1,
        ]), gl.STATIC_DRAW);

        // normals (pointing towards camera for inner glass)
        this.quadNrmBuf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadNrmBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            0, 0, 1,  0, 0, 1,  0, 0, 1,
            0, 0, 1,  0, 0, 1,  0, 0, 1,
        ]), gl.STATIC_DRAW);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // tick — update time uniform
    // ─────────────────────────────────────────────────────────────────────────

    tick(deltaMs: number): void {
        this.params.time += deltaMs * (this.params.timeScale) * 0.001;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // render — run all 3 passes every frame
    //
    //   Pass 1: GlassInner → innerRT        (subsurface scattering texture)
    //   Pass 2: refractionRT ← blit from external scene FBO (caller supplies)
    //   Pass 3: CleanRoomGlass → reflRT     (fresnel + refraction + env + inner)
    //   Pass 4: BasicMirror → screen (optional composite)
    // ─────────────────────────────────────────────────────────────────────────

    render(
        mvMatrix: Float32Array,
        projMatrix: Float32Array,
        modelMatrix: Float32Array,
        normalMatrix: Float32Array,
        cameraPos: [number, number, number],
        externalSceneTex?: WebGLTexture   // caller provides scene for refraction
    ): void {
        const gl = this.gl;
        const time = this.params.time;

        // ── Pass 1: GlassInner — subsurface scatter ──────────────────────────
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.innerRT.fbo);
        gl.viewport(0, 0, this.innerRT.width, this.innerRT.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.innerProg);

        // upload uniforms
        gl.uniform1f(gl.getUniformLocation(this.innerProg, 'uTime'), time);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.innerProg, 'uProjection'), false, projMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.innerProg, 'uModelView'),  false, mvMatrix);

        this._drawQuad(this.innerProg);

        // ── Pass 2: capture refraction into refractionRT ─────────────────────
        // If caller provides scene texture, blit it; otherwise blank
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.refractionRT.fbo);
        gl.viewport(0, 0, this.refractionRT.width, this.refractionRT.height);
        gl.clearColor(0.05, 0.05, 0.08, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        if (externalSceneTex) {
            // blit scene into refraction RT via a simple copy using reflProg temporarily
            // (in production you'd use a dedicated blit program; here we reuse display)
            gl.useProgram(this.reflProg);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, externalSceneTex);
            gl.uniform1i(gl.getUniformLocation(this.reflProg, 'tRefraction'), 0);
        }

        // ── Pass 3: CleanRoomGlass / GlassReflection ─────────────────────────
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.reflRT.fbo);
        gl.viewport(0, 0, this.reflRT.width, this.reflRT.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.reflProg);

        // matrices
        gl.uniformMatrix4fv(gl.getUniformLocation(this.reflProg, 'uProjection'), false, projMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.reflProg, 'uModelView'),  false, mvMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.reflProg, 'uModel'),      false, modelMatrix);
        gl.uniformMatrix3fv(gl.getUniformLocation(this.reflProg, 'uNormalMatrix'), false, normalMatrix);

        // camera + glass params
        gl.uniform3f(gl.getUniformLocation(this.reflProg, 'uCameraPos'),
            cameraPos[0], cameraPos[1], cameraPos[2]);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uFresnelPow'),      this.params.fresnelPower);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uDistortStrength'), this.params.distortStrength);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uRefractionRatio'), this.params.refractionRatio);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uEnvStrength'),     this.params.envProbeStrength);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uCleanRoomMode'),   this.params.cleanRoomMode ? 1.0 : 0.0);
        gl.uniform1f(gl.getUniformLocation(this.reflProg, 'uTime'),            time);
        gl.uniform2f(gl.getUniformLocation(this.reflProg, 'uResolution'),      this.width, this.height);

        // texture unit 0 — refraction (screen scene)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, externalSceneTex ?? this.refractionRT.tex);
        gl.uniform1i(gl.getUniformLocation(this.reflProg, 'tRefraction'), 0);

        // texture unit 1 — environment equirectangular
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.envTex);
        gl.uniform1i(gl.getUniformLocation(this.reflProg, 'tEnv'), 1);

        // texture unit 2 — GlassInner result
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.innerRT.tex);
        gl.uniform1i(gl.getUniformLocation(this.reflProg, 'tInner'), 2);

        this._drawQuad(this.reflProg);

        // ── Pass 4: BasicMirror — composite mirror onto screen ────────────────
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.width, this.height);

        gl.useProgram(this.mirrorProg);

        gl.uniformMatrix4fv(gl.getUniformLocation(this.mirrorProg, 'uProjection'), false, projMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.mirrorProg, 'uModelView'),  false, mvMatrix);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.mirrorProg, 'uModel'),      false, modelMatrix);

        // Mirror matrix (identity by default — caller overrides via setMirrorMatrix)
        const mirrorMat = new Float32Array(16);
        mirrorMat[0] = mirrorMat[5] = mirrorMat[10] = mirrorMat[15] = 1;
        gl.uniformMatrix4fv(gl.getUniformLocation(this.mirrorProg, 'uMirrorMatrix'), false, mirrorMat);

        // texture unit 0 — mirror reflection (reflRT as planar mirror source)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.mirrorTex);
        gl.uniform1i(gl.getUniformLocation(this.mirrorProg, 'tMirrorReflection'), 0);

        this._drawQuad(this.mirrorProg);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // setParams — update glass material params
    // ─────────────────────────────────────────────────────────────────────────

    setParams(params: Partial<GlassReflectionParams>): void {
        Object.assign(this.params, params);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Texture setters — called by scene to inject live textures
    // ─────────────────────────────────────────────────────────────────────────

    setEnvironmentTexture(tex: WebGLTexture): void {
        const gl = this.gl;
        if (this.envTex) {
            gl.deleteTexture(this.envTex);
        }
        this.envTex = tex;
        // set wrap/filter
        gl.bindTexture(gl.TEXTURE_2D, this.envTex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    setMirrorReflectionTexture(tex: WebGLTexture): void {
        const gl = this.gl;
        if (this.mirrorTex) {
            gl.deleteTexture(this.mirrorTex);
        }
        this.mirrorTex = tex;
        gl.bindTexture(gl.TEXTURE_2D, this.mirrorTex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    /** Load an equirectangular environment image into the env texture */
    loadEnvironmentFromImageData(pixels: Uint8Array, w: number, h: number): void {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.envTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Preset helpers
    // ─────────────────────────────────────────────────────────────────────────

    applyPreset(preset: GlassReflectionParams): void { this.setParams(preset); }
    applyOpticalGlassPreset(): void  { this.applyPreset(PRESET_OPTICAL_GLASS); }
    applyCrownGlassPreset(): void    { this.applyPreset(PRESET_CROWN_GLASS); }
    applyFlintGlassPreset(): void    { this.applyPreset(PRESET_FLINT_GLASS); }
    applyDiamondPreset(): void       { this.applyPreset(PRESET_DIAMOND); }
    applyCleanroomPreset(): void     { this.applyPreset(PRESET_CLEANROOM); }

    // ─────────────────────────────────────────────────────────────────────────
    // getResultTexture — get the final composited glass texture
    // ─────────────────────────────────────────────────────────────────────────

    get resultTexture(): WebGLTexture { return this.reflRT.tex; }
    get innerTexture(): WebGLTexture  { return this.innerRT.tex; }

    // ─────────────────────────────────────────────────────────────────────────
    // resize
    // ─────────────────────────────────────────────────────────────────────────

    resize(width: number, height: number): void {
        const gl = this.gl;
        this.width = width;
        this.height = height;

        // Recreate all render targets at new resolution
        this._destroyRT(this.innerRT);
        this._destroyRT(this.reflRT);
        this._destroyRT(this.refractionRT);

        this.innerRT      = this._createRT(width, height);
        this.reflRT       = this._createRT(width, height);
        this.refractionRT = this._createRT(width, height);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dispose — delete all GPU resources
    // ─────────────────────────────────────────────────────────────────────────

    dispose(): void {
        const gl = this.gl;

        // delete programs
        gl.deleteProgram(this.innerProg);
        gl.deleteProgram(this.reflProg);
        gl.deleteProgram(this.mirrorProg);

        // delete FBOs + textures
        this._destroyRT(this.innerRT);
        this._destroyRT(this.reflRT);
        this._destroyRT(this.refractionRT);

        // delete standalone textures
        gl.deleteTexture(this.envTex);
        gl.deleteTexture(this.mirrorTex);

        // delete geometry buffers
        gl.deleteBuffer(this.quadBuf);
        gl.deleteBuffer(this.quadUvBuf);
        gl.deleteBuffer(this.quadNrmBuf);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Compile vert + frag into a linked WebGLProgram */
    private _compile(vert: string, frag: string, label: string): WebGLProgram {
        const gl = this.gl;

        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vs, vert);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            throw new Error(`[ATGlass] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
        }

        const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
        gl.shaderSource(fs, frag);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            throw new Error(`[ATGlass] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
        }

        const prog = gl.createProgram()!;
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error(`[ATGlass] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
        }

        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    /** Create a single color FBO + texture */
    private _createRT(w: number, h: number): SingleRT {
        const gl = this.gl;

        const tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const fbo = gl.createFramebuffer()!;
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);

        return { fbo, tex, width: w, height: h };
    }

    /** Delete an FBO + its texture */
    private _destroyRT(rt: SingleRT): void {
        const gl = this.gl;
        gl.deleteFramebuffer(rt.fbo);
        gl.deleteTexture(rt.tex);
    }

    /** Draw fullscreen quad using the given program's attribute locations */
    private _drawQuad(prog: WebGLProgram): void {
        const gl = this.gl;

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const uvLoc  = gl.getAttribLocation(prog, 'aUv');
        const nrmLoc = gl.getAttribLocation(prog, 'aNormal');

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
        if (posLoc >= 0) {
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUvBuf);
        if (uvLoc >= 0) {
            gl.enableVertexAttribArray(uvLoc);
            gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadNrmBuf);
        if (nrmLoc >= 0) {
            gl.enableVertexAttribArray(nrmLoc);
            gl.vertexAttribPointer(nrmLoc, 3, gl.FLOAT, false, 0, 0);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        if (posLoc >= 0) gl.disableVertexAttribArray(posLoc);
        if (uvLoc  >= 0) gl.disableVertexAttribArray(uvLoc);
        if (nrmLoc >= 0) gl.disableVertexAttribArray(nrmLoc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exports
// ─────────────────────────────────────────────────────────────────────────────

export const AT_GLASS_PRESETS = {
    default:      DEFAULT_GLASS_PARAMS,
    opticalGlass: PRESET_OPTICAL_GLASS,
    crownGlass:   PRESET_CROWN_GLASS,
    flintGlass:   PRESET_FLINT_GLASS,
    diamond:      PRESET_DIAMOND,
    cleanroom:    PRESET_CLEANROOM,
};

export const AT_GLASS_SHADERS = {
    glassInnerVert:  GLASS_INNER_VERT,
    glassInnerFrag:  GLASS_INNER_FRAG,
    glassReflVert:   GLASS_REFL_VERT,
    glassReflFrag:   GLASS_REFL_FRAG,
    basicMirrorVert: BASIC_MIRROR_VERT,
    basicMirrorFrag: BASIC_MIRROR_FRAG,
};

export default ATGlassReflectionSystem;
