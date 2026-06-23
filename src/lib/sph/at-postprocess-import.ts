/**
 * at-postprocess-import.ts — M806: AT Post-Process Pipeline Direct Import
 *
 * 直接使用 ActiveTheory 原始后处理 shader 链 (GLSL)，不改写逻辑。
 * 所有 `#require(...)` 依赖已内联展开。
 *
 * AT 原始 shader 来源 (upstream/activetheory-assets/shaders/):
 *
 *   1. FXAA.vs + FXAA.fs
 *      反锯齿后处理。5 邻域 luma FXAA + tMask 选择性 AA。
 *      内联依赖: conditionals.glsl (when_lt / when_gt)
 *      uniforms: tDiffuse, tMask, resolution
 *
 *   2. LensFlarePrefilter.fs → LensFlareDown.fs → LensFlareUp.fs (3-pass)
 *      镜头光晕三段管线：
 *        Pass A — Prefilter: luma threshold + rotateUV
 *        Pass B — Down: 6-tap 水平加权模糊
 *        Pass C — Up: 3-tap 水平 + tHigh blend + 垂直 soften
 *      内联依赖: transformUV.glsl, luma.fs
 *
 *   3. LightVolume.vs + LightVolume.fs
 *      体积光 instanced geometry + 旋转/噪声/mask 驱动的 alpha。
 *      内联依赖: instance.vs, rotation.glsl, rgb2hsv.fs, range.glsl,
 *               transformUV.glsl, simplenoise.glsl
 *
 * 所有 shader 源码直接从 AT 文件粘贴，仅做 #require 内联。
 */

// ═══════════════════════════════════════════════════════════════════════════
// §1  GLSL Shader Sources — AT 原始代码 + 内联依赖
// ═══════════════════════════════════════════════════════════════════════════

// ── 全屏三角 / 四边形 vertex shader (lens flare passes 共用) ──
const FULLSCREEN_VERT = `
attribute vec3 position;
attribute vec2 uv;

varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
`;

// ── FXAA.vs — AT 原始 ──
const FXAA_VERT = `
attribute vec3 position;
attribute vec2 uv;

uniform vec2 resolution;

varying vec2 vUv;
varying vec2 v_rgbNW;
varying vec2 v_rgbNE;
varying vec2 v_rgbSW;
varying vec2 v_rgbSE;
varying vec2 v_rgbM;

void main() {
    vUv = uv;

    vec2 fragCoord = uv * resolution;
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);

    gl_Position = vec4(position, 1.0);
}
`;

// ── FXAA.fs — AT 原始 + 内联 conditionals.glsl ──
const FXAA_FRAG = `
precision highp float;

uniform sampler2D tDiffuse;
uniform sampler2D tMask;
uniform vec2 resolution;

varying vec2 vUv;
varying vec2 v_rgbNW;
varying vec2 v_rgbNE;
varying vec2 v_rgbSW;
varying vec2 v_rgbSE;
varying vec2 v_rgbM;

// ── conditionals.glsl (inlined) ──
float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}
// ── end conditionals.glsl ──

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
            vec2 v_rgbNW, vec2 v_rgbNE,
            vec2 v_rgbSW, vec2 v_rgbSE,
            vec2 v_rgbM) {
    vec4 color;
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec3 rgbNW = texture2D(tex, v_rgbNW).xyz;
    vec3 rgbNE = texture2D(tex, v_rgbNE).xyz;
    vec3 rgbSW = texture2D(tex, v_rgbSW).xyz;
    vec3 rgbSE = texture2D(tex, v_rgbSE).xyz;
    vec4 texColor = texture2D(tex, v_rgbM);
    vec3 rgbM  = texColor.xyz;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;

    vec3 rgbA = 0.5 * (
        texture2D(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture2D(tex, fragCoord * inverseVP + dir * -0.5).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * 0.5).xyz);

    float lumaB = dot(rgbB, luma);

    color = vec4(rgbB, texColor.a);
    color = mix(color, vec4(rgbA, texColor.a), when_lt(lumaB, lumaMin));
    color = mix(color, vec4(rgbA, texColor.a), when_gt(lumaB, lumaMax));

    return color;
}

void main() {
    vec2 fragCoord = vUv * resolution;
    float mask = texture2D(tMask, vUv).r;
    if (mask < 0.5) {
        gl_FragColor = fxaa(tDiffuse, fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);
    } else {
        gl_FragColor = texture2D(tDiffuse, vUv);
    }
    gl_FragColor.a = 1.0;
}
`;

// ── LensFlarePrefilter.fs — AT 原始 + 内联 transformUV.glsl, luma.fs ──
const LENS_PREFILTER_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform float uThreshold;
uniform float uRotate;

varying vec2 vUv;

// ── luma.fs (inlined) ──
float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}
// ── end luma.fs ──

// ── transformUV.glsl (inlined, rotateUV only) ──
vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    float c = cos(r);
    float s = sin(r);
    mat2 m = mat2(c, -s,
                  s, c);
    vec2 st = uv - origin;
    st = m * st;
    return st + origin;
}

vec2 rotateUV(vec2 uv, float r) {
    return rotateUV(uv, r, vec2(0.5));
}
// ── end transformUV.glsl ──

void main() {
    vec2 uv = vUv;

    uv = rotateUV(uv, -uRotate);

    vec4 c = texture2D(tMap, vec2(uv.x, uv.y));

    // threshold the brightness
    float brightness = luma(c.rgb);
    if (brightness < uThreshold) {
        c = vec4(0.);
    }

    gl_FragColor = vec4(c.rgb, 1.0);
}
`;

// ── LensFlareDown.fs — AT 原始 ──
const LENS_DOWN_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform float uStretch;
uniform vec2 uResolution;

varying vec2 vUv;

void main() {
    vec2 uv = vUv;

    float dx = 1. / uResolution.x;

    float stretch = uStretch;

    float u0 = uv.x - ((dx * 5.) * stretch);
    float u1 = uv.x - ((dx * 3.) * stretch);
    float u2 = uv.x - ((dx * 1.) * stretch);
    float u3 = uv.x + ((dx * 1.) * stretch);
    float u4 = uv.x + ((dx * 3.) * stretch);
    float u5 = uv.x + ((dx * 5.) * stretch);

    vec3 c0 = texture2D(tMap, vec2(u0, uv.y)).rgb;
    vec3 c1 = texture2D(tMap, vec2(u1, uv.y)).rgb;
    vec3 c2 = texture2D(tMap, vec2(u2, uv.y)).rgb;
    vec3 c3 = texture2D(tMap, vec2(u3, uv.y)).rgb;
    vec3 c4 = texture2D(tMap, vec2(u4, uv.y)).rgb;
    vec3 c5 = texture2D(tMap, vec2(u5, uv.y)).rgb;

    vec3 col =  vec3((c0 + c1 * 2. + c2 * 3. + c3 * 3. + c4 * 2. + c5) / 12.);

    gl_FragColor = vec4( col.rgb, 1.0 );
}
`;

// ── LensFlareUp.fs — AT 原始 ──
const LENS_UP_FRAG = `
precision highp float;

uniform sampler2D tHigh;
uniform sampler2D tScene;
uniform float uStretch;
uniform float uSoftenEdge;
uniform vec2 uResolution;

varying vec2 vUv;

void main() {
    vec2 uv = vUv;

    float dx = 1. / uResolution.x;

    float u0 = uv.x - dx;
    float u1 = uv.x;
    float u2 = uv.x + dx;

    // sample horizontally
    vec3 c0 = texture2D(tScene, vec2(u0, uv.y)).rgb / 4.;
    vec3 c1 = texture2D(tScene, vec2(u1, uv.y)).rgb / 2.;
    vec3 c2 = texture2D(tScene, vec2(u2, uv.y)).rgb / 4.;

    vec3 c3 = texture2D(tHigh, uv).rgb;

    vec3 cStretch = c0 + c1 + c2;

    // sample vertically
    vec3 c4 = texture2D(tScene, vec2(uv.x, uv.y - (dx * 0.75))).rgb / 4.;
    vec3 c5 = texture2D(tScene, vec2(uv.x, uv.y + (dx * 0.75))).rgb / 4.;

    cStretch += (c4 + c5) * uSoftenEdge;

    vec4 col = vec4(cStretch, 1.);

    gl_FragColor = col;
}
`;

// ── Composite fragment — scene + flare additive blend ──
const COMPOSITE_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform sampler2D tFlare;

varying vec2 vUv;

void main() {
    vec4 scene = texture2D(tScene, vUv);
    vec4 flare = texture2D(tFlare, vUv);
    gl_FragColor = vec4(scene.rgb + flare.rgb, 1.0);
}
`;

// ── LightVolume.vs — AT 原始 + 内联 instance.vs, rotation.glsl ──
const LIGHT_VOLUME_VERT = `
attribute vec3 position;
attribute vec2 uv;
attribute vec3 offset;
attribute vec3 attribs;

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform float uSeparation;
uniform vec3 uScale;
uniform float uOffset;

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vAttribs;
varying float vOffset;

// ── instance.vs (inlined, transformPosition only) ──
vec3 transformPosition(vec3 position, vec3 offset, vec3 scale) {
    vec3 pos = position * scale;
    return pos + offset;
}
// ── end instance.vs ──

// ── rotation.glsl (inlined) ──
mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}
// ── end rotation.glsl ──

void main() {
    vec3 pos = transformPosition(position, offset * uSeparation, uScale);
    pos = vec3(vec4(pos, 1.0) * rotationMatrix(vec3(0.0, 0.0, 1.0), radians(360.0 * 0.1 * offset.z * uOffset)));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    vUv = uv;
    vPos = pos;
    vAttribs = attribs;
    vOffset = offset.z * 10.0;
}
`;

// ── LightVolume.fs — AT 原始 + 内联 rgb2hsv.fs, range.glsl, transformUV.glsl, simplenoise.glsl ──
const LIGHT_VOLUME_FRAG = `
precision highp float;

uniform vec3 uColor;
uniform float uHueShift;
uniform float uRotateTexture;
uniform float uMaskScale;
uniform float uNoiseSpeed;
uniform float uNoiseScale;
uniform float uNoiseRange;
uniform float uRotateSpeed;
uniform float uScrollX;
uniform float uScrollY;
uniform float uAlpha;
uniform float time;
uniform sampler2D tMap;
uniform sampler2D tMask;

varying vec2 vUv;
varying vec3 vPos;
varying vec3 vAttribs;
varying float vOffset;

// ── rgb2hsv.fs (inlined) ──
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
// ── end rgb2hsv.fs ──

// ── range.glsl (inlined) ──
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
// ── end range.glsl ──

// ── transformUV.glsl (inlined) ──
vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    float c = cos(r);
    float s = sin(r);
    mat2 m = mat2(c, -s,
                  s, c);
    vec2 st = uv - origin;
    st = m * st;
    return st + origin;
}

vec2 rotateUV(vec2 uv, float r) {
    return rotateUV(uv, r, vec2(0.5));
}

vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec2 st = uv - origin;
    st /= scale;
    return st + origin;
}

vec2 scaleUV(vec2 uv, vec2 scale) {
    return scaleUV(uv, scale, vec2(0.5));
}
// ── end transformUV.glsl ──

// ── simplenoise.glsl (inlined, desktop sinf=sin) ──
#define sinf sin

float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += (sinf(v.x * 0.9 / s + t * 10.0) + sinf(v.x * 2.4 / s + t * 15.0) + sinf(v.x * -3.5 / s + t * 4.0) + sinf(v.x * -2.5 / s + t * 7.1)) * 0.3;
    noise += (sinf(v.y * -0.3 / s + t * 18.0) + sinf(v.y * 1.6 / s + t * 18.0) + sinf(v.y * 2.6 / s + t * 8.0) + sinf(v.y * -2.6 / s + t * 4.5)) * 0.3;
    return noise;
}
// ── end simplenoise.glsl ──

void main() {
    vec3 color = rgb2hsv(uColor);
    color += vOffset * uHueShift * 0.01;
    color = hsv2rgb(color);

    vec2 auv = vUv;
    if (uRotateTexture > 0.0) {
        auv = rotateUV(vUv, time * uRotateTexture * 0.1);
    }

    float alpha = texture2D(tMap, auv).r;

    vec2 uv = scaleUV(vUv, vec2(uMaskScale));

    if (uNoiseSpeed > 0.0) {
        float noise = cnoise(vPos * uNoiseScale + (time * uNoiseSpeed));
        uv += noise * uNoiseRange * 0.1;
        uv = scaleUV(uv, vec2(range(noise, -1.0, 0.0, 0.96, 1.02)));
        uv.x += sin(time * 0.04) * 0.3;
    }

    if (uRotateSpeed > 0.0) {
        uv = rotateUV(uv, uRotateSpeed * time * range(vAttribs.x, 0.0, 1.0, 0.5, 1.5));
        uv.x += time * uScrollX * 0.1 * range(vAttribs.y, 0.0, 1.0, 0.5, 1.5);
        uv.y += time * uScrollY * 0.1 * range(vAttribs.z, 0.0, 1.0, 0.5, 1.5);
    }

    float mask = texture2D(tMask, uv).r;
    alpha *= mask;

    gl_FragColor = vec4(color, alpha * uAlpha);
}
`;

// ═══════════════════════════════════════════════════════════════════════════
// §2  Exported shader source constants
// ═══════════════════════════════════════════════════════════════════════════

export const AT_FXAA_VERT         = FXAA_VERT;
export const AT_FXAA_FRAG         = FXAA_FRAG;
export const AT_LENS_PREFILTER_FRAG = LENS_PREFILTER_FRAG;
export const AT_LENS_DOWN_FRAG    = LENS_DOWN_FRAG;
export const AT_LENS_UP_FRAG      = LENS_UP_FRAG;
export const AT_COMPOSITE_FRAG    = COMPOSITE_FRAG;
export const AT_FULLSCREEN_VERT   = FULLSCREEN_VERT;
export const AT_LIGHT_VOLUME_VERT = LIGHT_VOLUME_VERT;
export const AT_LIGHT_VOLUME_FRAG = LIGHT_VOLUME_FRAG;

// ═══════════════════════════════════════════════════════════════════════════
// §3  Params Interface
// ═══════════════════════════════════════════════════════════════════════════

export interface ATPostProcessParams {
  fxaaEnabled?: boolean;
  lensFlare?: {
    threshold?: number;   // default 0.8
    rotate?: number;      // default 0.0
    stretch?: number;     // default 1.0
    softenEdge?: number;  // default 0.5
  };
  lightVolume?: {
    color?: [number, number, number];
    hueShift?: number;
    rotateTexture?: number;
    maskScale?: number;
    noiseSpeed?: number;
    noiseScale?: number;
    noiseRange?: number;
    rotateSpeed?: number;
    scrollX?: number;
    scrollY?: number;
    alpha?: number;
    separation?: number;
    scale?: [number, number, number];
    offset?: number;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  WebGL Helpers
// ═══════════════════════════════════════════════════════════════════════════

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info}`);
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

interface FBO {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

function createFBO(gl: WebGLRenderingContext, width: number, height: number): FBO {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const framebuffer = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return { framebuffer, texture, width, height };
}

function destroyFBO(gl: WebGLRenderingContext, fbo: FBO): void {
  gl.deleteFramebuffer(fbo.framebuffer);
  gl.deleteTexture(fbo.texture);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  ATPostProcessPipeline — WebGL
// ═══════════════════════════════════════════════════════════════════════════

export class ATPostProcessPipeline {
  private gl: WebGLRenderingContext;
  private width: number;
  private height: number;

  // Programs
  private fxaaProgram: WebGLProgram | null = null;
  private prefilterProgram: WebGLProgram | null = null;
  private downProgram: WebGLProgram | null = null;
  private upProgram: WebGLProgram | null = null;
  private compositeProgram: WebGLProgram | null = null;
  private lightVolumeProgram: WebGLProgram | null = null;

  // FBOs
  private fxaaFBO: FBO | null = null;
  private prefilterFBO: FBO | null = null;
  private downFBO: FBO | null = null;
  private upFBO: FBO | null = null;
  private compositeFBO: FBO | null = null;
  private lightVolumeFBO: FBO | null = null;

  // Fullscreen quad
  private quadVAO: WebGLBuffer | null = null;
  private quadUVBuffer: WebGLBuffer | null = null;

  // White 1×1 texture (default tMask for FXAA when no mask provided)
  private whiteTex: WebGLTexture | null = null;

  // Params
  private params: ATPostProcessParams = {
    fxaaEnabled: true,
    lensFlare: {
      threshold: 0.8,
      rotate: 0.0,
      stretch: 1.0,
      softenEdge: 0.5,
    },
    lightVolume: {
      color: [1.0, 1.0, 1.0],
      hueShift: 0.0,
      rotateTexture: 0.0,
      maskScale: 1.0,
      noiseSpeed: 0.0,
      noiseScale: 1.0,
      noiseRange: 0.1,
      rotateSpeed: 0.0,
      scrollX: 0.0,
      scrollY: 0.0,
      alpha: 1.0,
      separation: 1.0,
      scale: [1.0, 1.0, 1.0],
      offset: 1.0,
    },
  };

  constructor(gl: WebGLRenderingContext, width: number, height: number) {
    this.gl = gl;
    this.width = width;
    this.height = height;
  }

  // ── init(): 编译所有 shader、创建 FBO、创建全屏 quad ──
  init(): void {
    const gl = this.gl;

    // Programs
    this.fxaaProgram = createProgram(gl, FXAA_VERT, FXAA_FRAG);
    this.prefilterProgram = createProgram(gl, FULLSCREEN_VERT, LENS_PREFILTER_FRAG);
    this.downProgram = createProgram(gl, FULLSCREEN_VERT, LENS_DOWN_FRAG);
    this.upProgram = createProgram(gl, FULLSCREEN_VERT, LENS_UP_FRAG);
    this.compositeProgram = createProgram(gl, FULLSCREEN_VERT, COMPOSITE_FRAG);
    this.lightVolumeProgram = createProgram(gl, LIGHT_VOLUME_VERT, LIGHT_VOLUME_FRAG);

    // FBOs
    this.fxaaFBO = createFBO(gl, this.width, this.height);
    this.prefilterFBO = createFBO(gl, this.width, this.height);
    this.downFBO = createFBO(gl, this.width, this.height);
    this.upFBO = createFBO(gl, this.width, this.height);
    this.compositeFBO = createFBO(gl, this.width, this.height);
    this.lightVolumeFBO = createFBO(gl, this.width, this.height);

    // Fullscreen quad: NDC positions + UVs
    // positions: [-1,-1, 1,-1, -1,1, 1,1]  (triangle strip)
    // uvs:       [ 0, 0, 1, 0,  0,1, 1,1]
    const positions = new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]);
    const uvs = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);

    this.quadVAO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVAO);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    this.quadUVBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUVBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

    // White 1×1 texture (FXAA mask default — all 0 → FXAA everywhere)
    this.whiteTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── drawQuad: 绑定全屏 quad 属性并绘制 ──
  private drawQuad(program: WebGLProgram): void {
    const gl = this.gl;

    const posLoc = gl.getAttribLocation(program, 'position');
    const uvLoc = gl.getAttribLocation(program, 'uv');

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVAO!);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    if (uvLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUVBuffer!);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.disableVertexAttribArray(posLoc);
    if (uvLoc >= 0) gl.disableVertexAttribArray(uvLoc);
  }

  // ── setUniform helpers ──
  private setUniform1f(program: WebGLProgram, name: string, value: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc) this.gl.uniform1f(loc, value);
  }

  private setUniform2f(program: WebGLProgram, name: string, x: number, y: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc) this.gl.uniform2f(loc, x, y);
  }

  private setUniform3f(program: WebGLProgram, name: string, x: number, y: number, z: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc) this.gl.uniform3f(loc, x, y, z);
  }

  private setUniform1i(program: WebGLProgram, name: string, value: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc) this.gl.uniform1i(loc, value);
  }

  private bindTexture(unit: number, texture: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  // ── applyFXAA ──
  applyFXAA(input: WebGLTexture, mask?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const program = this.fxaaProgram!;
    const fbo = this.fxaaFBO!;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(program);

    this.bindTexture(0, input);
    this.setUniform1i(program, 'tDiffuse', 0);

    this.bindTexture(1, mask || this.whiteTex!);
    this.setUniform1i(program, 'tMask', 1);

    this.setUniform2f(program, 'resolution', this.width, this.height);

    this.drawQuad(program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo.texture;
  }

  // ── applyLensFlare (3-pass: prefilter → down → up, then composite) ──
  applyLensFlare(input: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const p = this.params.lensFlare || {};
    const threshold = p.threshold ?? 0.8;
    const rotate = p.rotate ?? 0.0;
    const stretch = p.stretch ?? 1.0;
    const softenEdge = p.softenEdge ?? 0.5;

    // ── Pass A: Prefilter ──
    {
      const program = this.prefilterProgram!;
      const fbo = this.prefilterFBO!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tMap', 0);
      this.setUniform1f(program, 'uThreshold', threshold);
      this.setUniform1f(program, 'uRotate', rotate);

      this.drawQuad(program);
    }

    // ── Pass B: Down (6-tap horizontal blur) ──
    {
      const program = this.downProgram!;
      const fbo = this.downFBO!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, this.prefilterFBO!.texture);
      this.setUniform1i(program, 'tMap', 0);
      this.setUniform1f(program, 'uStretch', stretch);
      this.setUniform2f(program, 'uResolution', this.width, this.height);

      this.drawQuad(program);
    }

    // ── Pass C: Up (3-tap horizontal + tHigh blend + soften) ──
    {
      const program = this.upProgram!;
      const fbo = this.upFBO!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, this.prefilterFBO!.texture); // tHigh = prefiltered bright
      this.setUniform1i(program, 'tHigh', 0);

      this.bindTexture(1, this.downFBO!.texture); // tScene = blurred down
      this.setUniform1i(program, 'tScene', 1);

      this.setUniform1f(program, 'uStretch', stretch);
      this.setUniform1f(program, 'uSoftenEdge', softenEdge);
      this.setUniform2f(program, 'uResolution', this.width, this.height);

      this.drawQuad(program);
    }

    // ── Composite: scene + flare ──
    {
      const program = this.compositeProgram!;
      const fbo = this.compositeFBO!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tScene', 0);

      this.bindTexture(1, this.upFBO!.texture);
      this.setUniform1i(program, 'tFlare', 1);

      this.drawQuad(program);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this.compositeFBO!.texture;
  }

  // ── applyLightVolume ──
  // Renders instanced light volume geometry into an FBO and returns the texture.
  // vertexBuffer / uvBuffer / offsetBuffer / attribsBuffer: 预创建的 WebGLBuffers
  // instanceCount: instance 数量
  // projectionMatrix / modelViewMatrix: 4×4 Float32Array
  applyLightVolume(
    input: WebGLTexture,
    lightPos: [number, number],
    vertexBuffer: WebGLBuffer,
    uvBuffer: WebGLBuffer,
    offsetBuffer: WebGLBuffer,
    attribsBuffer: WebGLBuffer,
    vertexCount: number,
    instanceCount: number,
    projectionMatrix: Float32Array,
    modelViewMatrix: Float32Array,
    tMap: WebGLTexture,
    tMask: WebGLTexture,
    elapsedTime: number,
  ): WebGLTexture {
    const gl = this.gl;
    const program = this.lightVolumeProgram!;
    const fbo = this.lightVolumeFBO!;
    const lv = this.params.lightVolume || {};

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Copy scene into FBO first (so we blend on top)
    gl.useProgram(this.compositeProgram!);
    this.bindTexture(0, input);
    this.setUniform1i(this.compositeProgram!, 'tScene', 0);
    // Use a black texture for tFlare to just copy scene
    this.bindTexture(1, this.whiteTex!); // white is actually black (0,0,0,255)
    this.setUniform1i(this.compositeProgram!, 'tFlare', 1);
    this.drawQuad(this.compositeProgram!);

    // Now draw light volumes on top with blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.useProgram(program);

    // Matrices
    const projLoc = gl.getUniformLocation(program, 'projectionMatrix');
    if (projLoc) gl.uniformMatrix4fv(projLoc, false, projectionMatrix);

    const mvLoc = gl.getUniformLocation(program, 'modelViewMatrix');
    if (mvLoc) gl.uniformMatrix4fv(mvLoc, false, modelViewMatrix);

    // Uniforms
    const color = lv.color || [1, 1, 1];
    this.setUniform3f(program, 'uColor', color[0], color[1], color[2]);
    this.setUniform1f(program, 'uHueShift', lv.hueShift ?? 0.0);
    this.setUniform1f(program, 'uRotateTexture', lv.rotateTexture ?? 0.0);
    this.setUniform1f(program, 'uMaskScale', lv.maskScale ?? 1.0);
    this.setUniform1f(program, 'uNoiseSpeed', lv.noiseSpeed ?? 0.0);
    this.setUniform1f(program, 'uNoiseScale', lv.noiseScale ?? 1.0);
    this.setUniform1f(program, 'uNoiseRange', lv.noiseRange ?? 0.1);
    this.setUniform1f(program, 'uRotateSpeed', lv.rotateSpeed ?? 0.0);
    this.setUniform1f(program, 'uScrollX', lv.scrollX ?? 0.0);
    this.setUniform1f(program, 'uScrollY', lv.scrollY ?? 0.0);
    this.setUniform1f(program, 'uAlpha', lv.alpha ?? 1.0);
    this.setUniform1f(program, 'uSeparation', lv.separation ?? 1.0);
    const scale = lv.scale || [1, 1, 1];
    this.setUniform3f(program, 'uScale', scale[0], scale[1], scale[2]);
    this.setUniform1f(program, 'uOffset', lv.offset ?? 1.0);
    this.setUniform1f(program, 'time', elapsedTime);

    // Textures
    this.bindTexture(0, tMap);
    this.setUniform1i(program, 'tMap', 0);
    this.bindTexture(1, tMask);
    this.setUniform1i(program, 'tMask', 1);

    // Attributes
    const posLoc = gl.getAttribLocation(program, 'position');
    const uvLoc = gl.getAttribLocation(program, 'uv');
    const offsetLoc = gl.getAttribLocation(program, 'offset');
    const attribsLoc = gl.getAttribLocation(program, 'attribs');

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    if (uvLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
    }

    if (offsetLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuffer);
      gl.enableVertexAttribArray(offsetLoc);
      gl.vertexAttribPointer(offsetLoc, 3, gl.FLOAT, false, 0, 0);
    }

    if (attribsLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, attribsBuffer);
      gl.enableVertexAttribArray(attribsLoc);
      gl.vertexAttribPointer(attribsLoc, 3, gl.FLOAT, false, 0, 0);
    }

    // Draw — if ANGLE_instanced_arrays is available, use instancing
    const ext = gl.getExtension('ANGLE_instanced_arrays');
    if (ext && instanceCount > 1) {
      // offset and attribs are per-instance
      if (offsetLoc >= 0) ext.vertexAttribDivisorANGLE(offsetLoc, 1);
      if (attribsLoc >= 0) ext.vertexAttribDivisorANGLE(attribsLoc, 1);

      ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, vertexCount, instanceCount);

      // Reset divisors
      if (offsetLoc >= 0) ext.vertexAttribDivisorANGLE(offsetLoc, 0);
      if (attribsLoc >= 0) ext.vertexAttribDivisorANGLE(attribsLoc, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
    }

    // Cleanup
    gl.disableVertexAttribArray(posLoc);
    if (uvLoc >= 0) gl.disableVertexAttribArray(uvLoc);
    if (offsetLoc >= 0) gl.disableVertexAttribArray(offsetLoc);
    if (attribsLoc >= 0) gl.disableVertexAttribArray(attribsLoc);

    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return fbo.texture;
  }

  // ── process(): 完整后处理链 scene → FXAA → LensFlare → output ──
  process(sceneTexture: WebGLTexture): WebGLTexture {
    let current = sceneTexture;

    // FXAA
    if (this.params.fxaaEnabled !== false) {
      current = this.applyFXAA(current);
    }

    // Lens Flare (prefilter → down → up → composite)
    current = this.applyLensFlare(current);

    return current;
  }

  // ── setParams ──
  setParams(params: ATPostProcessParams): void {
    if (params.fxaaEnabled !== undefined) {
      this.params.fxaaEnabled = params.fxaaEnabled;
    }
    if (params.lensFlare) {
      this.params.lensFlare = { ...this.params.lensFlare, ...params.lensFlare };
    }
    if (params.lightVolume) {
      this.params.lightVolume = { ...this.params.lightVolume, ...params.lightVolume };
    }
  }

  // ── resize ──
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;

    const gl = this.gl;

    // Recreate all FBOs at new size
    const fbos: (keyof ATPostProcessPipeline)[] = [
      'fxaaFBO', 'prefilterFBO', 'downFBO', 'upFBO', 'compositeFBO', 'lightVolumeFBO'
    ];
    for (const key of fbos) {
      const fbo = this[key] as FBO | null;
      if (fbo) {
        destroyFBO(gl, fbo);
        (this as any)[key] = createFBO(gl, width, height);
      }
    }
  }

  // ── destroy ──
  destroy(): void {
    const gl = this.gl;

    // Delete programs
    const programs = [
      this.fxaaProgram, this.prefilterProgram, this.downProgram,
      this.upProgram, this.compositeProgram, this.lightVolumeProgram,
    ];
    for (const p of programs) {
      if (p) gl.deleteProgram(p);
    }

    // Delete FBOs
    const fbos = [
      this.fxaaFBO, this.prefilterFBO, this.downFBO,
      this.upFBO, this.compositeFBO, this.lightVolumeFBO,
    ];
    for (const fbo of fbos) {
      if (fbo) destroyFBO(gl, fbo);
    }

    // Delete buffers
    if (this.quadVAO) gl.deleteBuffer(this.quadVAO);
    if (this.quadUVBuffer) gl.deleteBuffer(this.quadUVBuffer);

    // Delete white texture
    if (this.whiteTex) gl.deleteTexture(this.whiteTex);

    this.fxaaProgram = null;
    this.prefilterProgram = null;
    this.downProgram = null;
    this.upProgram = null;
    this.compositeProgram = null;
    this.lightVolumeProgram = null;
    this.fxaaFBO = null;
    this.prefilterFBO = null;
    this.downFBO = null;
    this.upFBO = null;
    this.compositeFBO = null;
    this.lightVolumeFBO = null;
    this.quadVAO = null;
    this.quadUVBuffer = null;
    this.whiteTex = null;
  }
}
