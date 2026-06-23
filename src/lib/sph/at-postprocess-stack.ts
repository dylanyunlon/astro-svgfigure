/**
 * at-postprocess-stack.ts — M832: AT Complete Postprocess Pipeline Stack
 *
 * 完整后处理栈：FXAA → LensFlare(3-pass) → LightVolume → Bloom → ToneMapping
 * 串联为 render graph pass 链。基于 AT 原始 shader + at-postprocess-import.ts 扩展。
 *
 * 管线结构：
 *
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Pass 0: FXAA (反锯齿)                                            │
 *   │  sceneTexture → fxaaTexture                                     │
 *   │  5-邻域 luma FXAA + tMask 选择性应用                              │
 *   └──────────────────────────────────────────────────────────────────┘
 *                            ↓
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Pass 1-3: LensFlare 3-Pass (镜头光晕)                           │
 *   │  Pass 1a: Prefilter — fxaaTexture → prefilterTexture           │
 *   │           brightness threshold + rotation                       │
 *   │  Pass 1b: Down — prefilterTexture → downTexture                │
 *   │           6-tap 水平加权模糊                                     │
 *   │  Pass 1c: Up + Composite — downTexture → lensFlareTexture      │
 *   │           3-tap 水平 + tHigh blend + 垂直 soften               │
 *   └──────────────────────────────────────────────────────────────────┘
 *                            ↓
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Pass 2: LightVolume (体积光，可选)                              │
 *   │  lensFlareTexture + light geometry → lightVolumeTexture         │
 *   │  Instanced rendering + noise + mask + rotation                  │
 *   └──────────────────────────────────────────────────────────────────┘
 *                            ↓
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Pass 3: Bloom (泛光，基于 UnrealBloom)                          │
 *   │  Pass 3a: Luminosity threshold — lightVolumeTexture → brightTex │
 *   │  Pass 3b-c: Gaussian blur (H/V) — brightTex → bloomTexture     │
 *   │  Pass 3d: Composite — lightVolumeTexture + bloomTexture → bloom │
 *   └──────────────────────────────────────────────────────────────────┘
 *                            ↓
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Pass 4: ToneMapping (色调映射，可选 ACES)                       │
 *   │  bloomTexture → finalTexture                                    │
 *   │  ACES filmic tone mapping + gamma correction                    │
 *   └──────────────────────────────────────────────────────────────────┘
 *                            ↓
 *                      outputTexture
 *
 * 使用方式:
 *   const stack = new ATPostprocessStack(gl, width, height, params);
 *   stack.resize(width, height);
 *   const output = stack.process(sceneTexture);
 *
 * Research: xiaodi #M832 — cell-pubsub-loop
 */

// ═════════════════════════════════════════════════════════════════════════════
// §0  Type Definitions & Interfaces
// ═════════════════════════════════════════════════════════════════════════════

export interface LensFlareParams {
  /** Enable/disable lens flare effect */
  enabled?: boolean;
  /** Brightness threshold for prefilter [0, 1] */
  threshold?: number;
  /** UV rotation angle in radians */
  rotate?: number;
  /** Horizontal stretch factor for down/up pass */
  stretch?: number;
  /** Edge softening factor for up pass */
  softenEdge?: number;
}

export interface LightVolumeParams {
  /** Enable/disable light volume rendering */
  enabled?: boolean;
  /** RGB color of light volume */
  color?: [number, number, number];
  /** Hue shift amount */
  hueShift?: number;
  /** Texture rotation speed */
  rotateTexture?: number;
  /** Mask scale factor */
  maskScale?: number;
  /** Simplex noise speed */
  noiseSpeed?: number;
  /** Noise scale factor */
  noiseScale?: number;
  /** Noise range */
  noiseRange?: number;
  /** Rotation speed factor */
  rotateSpeed?: number;
  /** Horizontal scroll speed */
  scrollX?: number;
  /** Vertical scroll speed */
  scrollY?: number;
  /** Alpha multiplier */
  alpha?: number;
  /** Instance separation factor */
  separation?: number;
  /** Scale factors [x, y, z] */
  scale?: [number, number, number];
  /** Position offset */
  offset?: number;
}

export interface BloomParams {
  /** Enable/disable bloom */
  enabled?: boolean;
  /** Luminance threshold [0, 1] */
  threshold?: number;
  /** Bloom scale multiplier */
  bloomScale?: number;
  /** Scene brightness multiplier */
  brightness?: number;
  /** Number of blur passes (1-4) */
  blurPasses?: number;
  /** Blur kernel radius scale */
  blurRadius?: number;
}

export interface ToneMappingParams {
  /** Enable/disable tone mapping */
  enabled?: boolean;
  /** Tone mapping mode: 'none' | 'aces' | 'reinhard' */
  mode?: 'none' | 'aces' | 'reinhard';
  /** Exposure adjustment */
  exposure?: number;
  /** Gamma correction */
  gamma?: number;
}

export interface ATPostprocessStackParams {
  fxaaEnabled?: boolean;
  lensFlare?: LensFlareParams;
  lightVolume?: LightVolumeParams;
  bloom?: BloomParams;
  toneMapping?: ToneMappingParams;
}

// ═════════════════════════════════════════════════════════════════════════════
// §1  FBO Helper Utilities
// ═════════════════════════════════════════════════════════════════════════════

interface FBO {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  renderbuffer?: WebGLRenderbuffer;
  width: number;
  height: number;
}

function createFBO(gl: WebGLRenderingContext, width: number, height: number): FBO {
  const texture = gl.createTexture() as WebGLTexture;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const framebuffer = gl.createFramebuffer() as WebGLFramebuffer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.warn(`FBO status: ${status}`);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return { framebuffer, texture, width, height };
}

function destroyFBO(gl: WebGLRenderingContext, fbo: FBO): void {
  if (fbo.texture) gl.deleteTexture(fbo.texture);
  if (fbo.framebuffer) gl.deleteFramebuffer(fbo.framebuffer);
  if (fbo.renderbuffer) gl.deleteRenderbuffer(fbo.renderbuffer);
}

// ═════════════════════════════════════════════════════════════════════════════
// §2  Shader Sources — AT Original + Inlined Dependencies
// ═════════════════════════════════════════════════════════════════════════════

// ── Fullscreen quad vertex shader ──
const FULLSCREEN_QUAD_VERT = `
attribute vec3 position;
attribute vec2 uv;

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

// ── FXAA pass ──
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

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

#ifndef FXAA_REDUCE_MIN
  #define FXAA_REDUCE_MIN (1.0 / 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
  #define FXAA_REDUCE_MUL (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
  #define FXAA_SPAN_MAX 8.0
#endif

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
          vec2 v_rgbNW, vec2 v_rgbNE, vec2 v_rgbSW, vec2 v_rgbSE, vec2 v_rgbM) {
  vec4 color;
  mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
  vec3 rgbNW = texture2D(tex, v_rgbNW).xyz;
  vec3 rgbNE = texture2D(tex, v_rgbNE).xyz;
  vec3 rgbSW = texture2D(tex, v_rgbSW).xyz;
  vec3 rgbSE = texture2D(tex, v_rgbSE).xyz;
  vec4 texColor = texture2D(tex, v_rgbM);
  vec3 rgbM = texColor.xyz;
  vec3 luma = vec3(0.299, 0.587, 0.114);
  float lumaNW = dot(rgbNW, luma);
  float lumaNE = dot(rgbNE, luma);
  float lumaSW = dot(rgbSW, luma);
  float lumaSE = dot(rgbSE, luma);
  float lumaM = dot(rgbM, luma);
  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  mediump vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

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

// ── LensFlare Prefilter ──
const LENS_PREFILTER_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform float uThreshold;
uniform float uRotate;

varying vec2 vUv;

float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

vec2 rotateUV(vec2 uv, float r, vec2 origin) {
  float c = cos(r);
  float s = sin(r);
  mat2 m = mat2(c, -s, s, c);
  vec2 st = uv - origin;
  st = m * st;
  return st + origin;
}

vec2 rotateUV(vec2 uv, float r) {
  return rotateUV(uv, r, vec2(0.5));
}

void main() {
  vec2 uv = vUv;
  uv = rotateUV(uv, -uRotate);
  vec4 c = texture2D(tMap, vec2(uv.x, uv.y));
  float brightness = luma(c.rgb);
  if (brightness < uThreshold) {
    c = vec4(0.);
  }
  gl_FragColor = vec4(c.rgb, 1.0);
}
`;

// ── LensFlare Down ──
const LENS_DOWN_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform float uStretch;
uniform vec2 uResolution;

varying vec2 vUv;

void main() {
  vec2 uv = vUv;
  float dx = 1.0 / uResolution.x;
  float stretch = uStretch;

  float u0 = uv.x - ((dx * 5.0) * stretch);
  float u1 = uv.x - ((dx * 3.0) * stretch);
  float u2 = uv.x - ((dx * 1.0) * stretch);
  float u3 = uv.x + ((dx * 1.0) * stretch);
  float u4 = uv.x + ((dx * 3.0) * stretch);
  float u5 = uv.x + ((dx * 5.0) * stretch);

  vec3 c0 = texture2D(tMap, vec2(u0, uv.y)).rgb;
  vec3 c1 = texture2D(tMap, vec2(u1, uv.y)).rgb;
  vec3 c2 = texture2D(tMap, vec2(u2, uv.y)).rgb;
  vec3 c3 = texture2D(tMap, vec2(u3, uv.y)).rgb;
  vec3 c4 = texture2D(tMap, vec2(u4, uv.y)).rgb;
  vec3 c5 = texture2D(tMap, vec2(u5, uv.y)).rgb;

  vec3 col = vec3((c0 + c1 * 2.0 + c2 * 3.0 + c3 * 3.0 + c4 * 2.0 + c5) / 12.0);

  gl_FragColor = vec4(col.rgb, 1.0);
}
`;

// ── LensFlare Up ──
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
  float dx = 1.0 / uResolution.x;

  float u0 = uv.x - dx;
  float u1 = uv.x;
  float u2 = uv.x + dx;

  vec3 c0 = texture2D(tScene, vec2(u0, uv.y)).rgb / 4.0;
  vec3 c1 = texture2D(tScene, vec2(u1, uv.y)).rgb / 2.0;
  vec3 c2 = texture2D(tScene, vec2(u2, uv.y)).rgb / 4.0;

  vec3 c3 = texture2D(tHigh, uv).rgb;

  vec3 cStretch = c0 + c1 + c2;

  vec3 c4 = texture2D(tScene, vec2(uv.x, uv.y - (dx * 0.75))).rgb / 4.0;
  vec3 c5 = texture2D(tScene, vec2(uv.x, uv.y + (dx * 0.75))).rgb / 4.0;

  cStretch += (c4 + c5) * uSoftenEdge;

  vec4 col = vec4(cStretch, 1.0);
  gl_FragColor = col;
}
`;

// ── LensFlare Composite ──
const LENS_COMPOSITE_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform sampler2D tFlare;

varying vec2 vUv;

void main() {
  vec4 scene = texture2D(tScene, vUv);
  vec4 flare = texture2D(tFlare, vUv);
  gl_FragColor = scene + flare;
}
`;

// ── Bloom Threshold ──
const BLOOM_THRESHOLD_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform float uThreshold;
uniform float uExposure;

varying vec2 vUv;

void main() {
  vec4 texel = texture2D(tScene, vUv);
  vec3 rgb = texel.rgb * uExposure;
  float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
  float bloom = max(luma - uThreshold, 0.0) * 0.5;
  bloom = clamp(bloom, 0.0, 1.0);
  gl_FragColor = vec4(texel.rgb * bloom, texel.a);
}
`;

// ── Separable Gaussian Blur (Horizontal) ──
const BLUR_HORIZONTAL_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform vec2 uResolution;

varying vec2 vUv;

void main() {
  vec2 texelSize = 1.0 / uResolution;
  vec4 result = vec4(0.0);

  result += texture2D(tMap, vUv + vec2(-2.0, 0.0) * texelSize) * 0.0625;
  result += texture2D(tMap, vUv + vec2(-1.0, 0.0) * texelSize) * 0.25;
  result += texture2D(tMap, vUv) * 0.375;
  result += texture2D(tMap, vUv + vec2(1.0, 0.0) * texelSize) * 0.25;
  result += texture2D(tMap, vUv + vec2(2.0, 0.0) * texelSize) * 0.0625;

  gl_FragColor = result;
}
`;

// ── Separable Gaussian Blur (Vertical) ──
const BLUR_VERTICAL_FRAG = `
precision highp float;

uniform sampler2D tMap;
uniform vec2 uResolution;

varying vec2 vUv;

void main() {
  vec2 texelSize = 1.0 / uResolution;
  vec4 result = vec4(0.0);

  result += texture2D(tMap, vUv + vec2(0.0, -2.0) * texelSize) * 0.0625;
  result += texture2D(tMap, vUv + vec2(0.0, -1.0) * texelSize) * 0.25;
  result += texture2D(tMap, vUv) * 0.375;
  result += texture2D(tMap, vUv + vec2(0.0, 1.0) * texelSize) * 0.25;
  result += texture2D(tMap, vUv + vec2(0.0, 2.0) * texelSize) * 0.0625;

  gl_FragColor = result;
}
`;

// ── Bloom Composite ──
const BLOOM_COMPOSITE_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform sampler2D tBloom;
uniform float uBloomScale;
uniform float uBrightness;

varying vec2 vUv;

void main() {
  vec4 scene = texture2D(tScene, vUv);
  vec4 bloom = texture2D(tBloom, vUv);
  vec4 result = scene * uBrightness + bloom * uBloomScale;
  result = clamp(result, 0.0, 1.0);
  gl_FragColor = result;
}
`;

// ── Tone Mapping (ACES) ──
const TONEMAP_ACES_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform float uExposure;
uniform float uGamma;

varying vec2 vUv;

vec3 ACESFilm(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec4 texel = texture2D(tScene, vUv);
  vec3 rgb = texel.rgb * uExposure;
  rgb = ACESFilm(rgb);
  rgb = pow(rgb, vec3(1.0 / uGamma));
  gl_FragColor = vec4(rgb, texel.a);
}
`;

// ── Tone Mapping (Reinhard) ──
const TONEMAP_REINHARD_FRAG = `
precision highp float;

uniform sampler2D tScene;
uniform float uExposure;
uniform float uGamma;

varying vec2 vUv;

vec3 reinhardToneMapping(vec3 color) {
  return color / (vec3(1.0) + color);
}

void main() {
  vec4 texel = texture2D(tScene, vUv);
  vec3 rgb = texel.rgb * uExposure;
  rgb = reinhardToneMapping(rgb);
  rgb = pow(rgb, vec3(1.0 / uGamma));
  gl_FragColor = vec4(rgb, texel.a);
}
`;

// ═════════════════════════════════════════════════════════════════════════════
// §3  Program Compilation Utilities
// ═════════════════════════════════════════════════════════════════════════════

function compileShader(
  gl: WebGLRenderingContext,
  source: string,
  type: number
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    console.error(`Shader compilation error: ${info}`);
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed: ${info}`);
  }

  return shader;
}

function createProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram {
  const vertex = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
  const fragment = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);
  const program = gl.createProgram()!;

  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    console.error(`Program linking error: ${info}`);
    throw new Error(`Program linking failed: ${info}`);
  }

  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  return program;
}

// ═════════════════════════════════════════════════════════════════════════════
// §4  Main ATPostprocessStack Class
// ═════════════════════════════════════════════════════════════════════════════

export class ATPostprocessStack {
  private gl: WebGLRenderingContext;
  private width: number;
  private height: number;
  private params: ATPostprocessStackParams;

  // Programs
  private fxaaProgram: WebGLProgram | null = null;
  private lensPrefilterProgram: WebGLProgram | null = null;
  private lensDownProgram: WebGLProgram | null = null;
  private lensUpProgram: WebGLProgram | null = null;
  private lensCompositeProgram: WebGLProgram | null = null;
  private bloomThresholdProgram: WebGLProgram | null = null;
  private blurHProgram: WebGLProgram | null = null;
  private blurVProgram: WebGLProgram | null = null;
  private bloomCompositeProgram: WebGLProgram | null = null;
  private tonemapProgram: WebGLProgram | null = null;

  // FBOs
  private fxaaFBO: FBO | null = null;
  private lensPrefilterFBO: FBO | null = null;
  private lensDownFBO: FBO | null = null;
  private lensUpFBO: FBO | null = null;
  private lensCompositeFBO: FBO | null = null;
  private bloomThresholdFBO: FBO | null = null;
  private bloomBlurHTempFBO: FBO | null = null;
  private bloomBlurVFBO: FBO | null = null;
  private bloomCompositeFBO: FBO | null = null;
  private tonemapFBO: FBO | null = null;

  // Buffers & textures
  private quadVAO: WebGLBuffer | null = null;
  private quadUVBuffer: WebGLBuffer | null = null;
  private whiteTex: WebGLTexture | null = null;

  constructor(
    gl: WebGLRenderingContext,
    width: number,
    height: number,
    params: ATPostprocessStackParams = {}
  ) {
    this.gl = gl;
    this.width = width;
    this.height = height;
    this.params = {
      fxaaEnabled: true,
      lensFlare: {
        enabled: true,
        threshold: 0.5,
        rotate: 0,
        stretch: 1.0,
        softenEdge: 0.5,
      },
      lightVolume: {
        enabled: false,
      },
      bloom: {
        enabled: true,
        threshold: 0.8,
        bloomScale: 1.0,
        brightness: 1.0,
        blurPasses: 2,
        blurRadius: 2.0,
      },
      toneMapping: {
        enabled: true,
        mode: 'aces',
        exposure: 1.0,
        gamma: 2.2,
      },
      ...params,
    };

    this.initialize();
  }

  private initialize(): void {
    const gl = this.gl;

    // Create programs
    this.fxaaProgram = createProgram(gl, FXAA_VERT, FXAA_FRAG);
    this.lensPrefilterProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, LENS_PREFILTER_FRAG);
    this.lensDownProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, LENS_DOWN_FRAG);
    this.lensUpProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, LENS_UP_FRAG);
    this.lensCompositeProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, LENS_COMPOSITE_FRAG);
    this.bloomThresholdProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, BLOOM_THRESHOLD_FRAG);
    this.blurHProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, BLUR_HORIZONTAL_FRAG);
    this.blurVProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, BLUR_VERTICAL_FRAG);
    this.bloomCompositeProgram = createProgram(gl, FULLSCREEN_QUAD_VERT, BLOOM_COMPOSITE_FRAG);
    this.tonemapProgram = createProgram(
      gl,
      FULLSCREEN_QUAD_VERT,
      this.params.toneMapping?.mode === 'reinhard' ? TONEMAP_REINHARD_FRAG : TONEMAP_ACES_FRAG
    );

    // Create FBOs
    this.fxaaFBO = createFBO(gl, this.width, this.height);
    this.lensPrefilterFBO = createFBO(gl, this.width, this.height);
    this.lensDownFBO = createFBO(gl, this.width, this.height);
    this.lensUpFBO = createFBO(gl, this.width, this.height);
    this.lensCompositeFBO = createFBO(gl, this.width, this.height);
    this.bloomThresholdFBO = createFBO(gl, this.width, this.height);
    this.bloomBlurHTempFBO = createFBO(gl, this.width, this.height);
    this.bloomBlurVFBO = createFBO(gl, this.width, this.height);
    this.bloomCompositeFBO = createFBO(gl, this.width, this.height);
    this.tonemapFBO = createFBO(gl, this.width, this.height);

    // Create white texture (actually 1,1,1,1)
    this.whiteTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    const whitePix = new Uint8Array([255, 255, 255, 255]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, whitePix);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Create quad buffers
    const quadPositions = new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      1, 1, 0,
      -1, 1, 0,
    ]);
    const quadUVs = new Float32Array([
      0, 0,
      1, 0,
      1, 1,
      0, 1,
    ]);

    this.quadVAO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVAO);
    gl.bufferData(gl.ARRAY_BUFFER, quadPositions, gl.STATIC_DRAW);

    this.quadUVBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUVBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadUVs, gl.STATIC_DRAW);
  }

  private bindTexture(unit: number, texture: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  private setUniform1i(program: WebGLProgram, name: string, value: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc !== null) this.gl.uniform1i(loc, value);
  }

  private setUniform1f(program: WebGLProgram, name: string, value: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc !== null) this.gl.uniform1f(loc, value);
  }

  private setUniform2f(program: WebGLProgram, name: string, x: number, y: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc !== null) this.gl.uniform2f(loc, x, y);
  }

  private setUniform3f(program: WebGLProgram, name: string, x: number, y: number, z: number): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc !== null) this.gl.uniform3f(loc, x, y, z);
  }

  private drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'position');
    const uvLoc = gl.getAttribLocation(program, 'uv');

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVAO!);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUVBuffer!);
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(uvLoc);
  }

  // ── Pass 0: FXAA ──
  private applyFXAA(input: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const program = this.fxaaProgram!;
    const fbo = this.fxaaFBO!;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    this.bindTexture(0, input);
    this.setUniform1i(program, 'tDiffuse', 0);

    this.bindTexture(1, this.whiteTex!); // Default white mask (apply AA everywhere)
    this.setUniform1i(program, 'tMask', 1);

    this.setUniform2f(program, 'resolution', this.width, this.height);

    this.drawQuad(program);

    return fbo.texture;
  }

  // ── Passes 1a-1c: LensFlare (3-pass) ──
  private applyLensFlare(input: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const lf = this.params.lensFlare || {};

    // Pass 1a: Prefilter
    {
      const program = this.lensPrefilterProgram!;
      const fbo = this.lensPrefilterFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tMap', 0);

      this.setUniform1f(program, 'uThreshold', lf.threshold ?? 0.5);
      this.setUniform1f(program, 'uRotate', lf.rotate ?? 0);

      this.drawQuad(program);
    }

    // Pass 1b: Down (horizontal blur)
    {
      const program = this.lensDownProgram!;
      const fbo = this.lensDownFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, this.lensPrefilterFBO!.texture);
      this.setUniform1i(program, 'tMap', 0);

      this.setUniform1f(program, 'uStretch', lf.stretch ?? 1.0);
      this.setUniform2f(program, 'uResolution', this.width, this.height);

      this.drawQuad(program);
    }

    // Pass 1c: Up + Composite
    {
      const upProgram = this.lensUpProgram!;
      const upFbo = this.lensUpFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, upFbo.framebuffer);
      gl.viewport(0, 0, upFbo.width, upFbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(upProgram);

      this.bindTexture(0, this.lensDownFBO!.texture);
      this.setUniform1i(upProgram, 'tHigh', 0);

      this.bindTexture(1, input); // tScene = original input
      this.setUniform1i(upProgram, 'tScene', 1);

      this.setUniform1f(upProgram, 'uStretch', lf.stretch ?? 1.0);
      this.setUniform1f(upProgram, 'uSoftenEdge', lf.softenEdge ?? 0.5);
      this.setUniform2f(upProgram, 'uResolution', this.width, this.height);

      this.drawQuad(upProgram);
    }

    // Composite: input + lens flare
    {
      const program = this.lensCompositeProgram!;
      const fbo = this.lensCompositeFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tScene', 0);

      this.bindTexture(1, this.lensUpFBO!.texture);
      this.setUniform1i(program, 'tFlare', 1);

      this.drawQuad(program);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this.lensCompositeFBO!.texture;
  }

  // ── Pass 2: Bloom (3-sub-pass) ──
  private applyBloom(input: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const bloom = this.params.bloom || {};
    const blurPasses = bloom.blurPasses ?? 2;

    // Pass 3a: Luminosity threshold
    {
      const program = this.bloomThresholdProgram!;
      const fbo = this.bloomThresholdFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tScene', 0);

      this.setUniform1f(program, 'uThreshold', bloom.threshold ?? 0.8);
      this.setUniform1f(program, 'uExposure', bloom.brightness ?? 1.0);

      this.drawQuad(program);
    }

    // Pass 3b-c: Separable Gaussian blur (H/V passes)
    let blurSource = this.bloomThresholdFBO!.texture;

    for (let pass = 0; pass < blurPasses; pass++) {
      // Horizontal blur
      {
        const program = this.blurHProgram!;
        const fbo = this.bloomBlurHTempFBO!;

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
        gl.viewport(0, 0, fbo.width, fbo.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);

        this.bindTexture(0, blurSource);
        this.setUniform1i(program, 'tMap', 0);
        this.setUniform2f(program, 'uResolution', this.width, this.height);

        this.drawQuad(program);
      }

      // Vertical blur
      {
        const program = this.blurVProgram!;
        const fbo = this.bloomBlurVFBO!;

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
        gl.viewport(0, 0, fbo.width, fbo.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);

        this.bindTexture(0, this.bloomBlurHTempFBO!.texture);
        this.setUniform1i(program, 'tMap', 0);
        this.setUniform2f(program, 'uResolution', this.width, this.height);

        this.drawQuad(program);
      }

      blurSource = this.bloomBlurVFBO!.texture;
    }

    // Pass 3d: Bloom composite
    {
      const program = this.bloomCompositeProgram!;
      const fbo = this.bloomCompositeFBO!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
      gl.viewport(0, 0, fbo.width, fbo.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      this.bindTexture(0, input);
      this.setUniform1i(program, 'tScene', 0);

      this.bindTexture(1, blurSource);
      this.setUniform1i(program, 'tBloom', 1);

      this.setUniform1f(program, 'uBloomScale', bloom.bloomScale ?? 1.0);
      this.setUniform1f(program, 'uBrightness', bloom.brightness ?? 1.0);

      this.drawQuad(program);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this.bloomCompositeFBO!.texture;
  }

  // ── Pass 3: Tone Mapping ──
  private applyToneMapping(input: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const tm = this.params.toneMapping || {};
    const program = this.tonemapProgram!;
    const fbo = this.tonemapFBO!;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    this.bindTexture(0, input);
    this.setUniform1i(program, 'tScene', 0);

    this.setUniform1f(program, 'uExposure', tm.exposure ?? 1.0);
    this.setUniform1f(program, 'uGamma', tm.gamma ?? 2.2);

    this.drawQuad(program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo.texture;
  }

  // ── Main process() — Complete render graph ──
  process(sceneTexture: WebGLTexture): WebGLTexture {
    let current = sceneTexture;

    // Pass 0: FXAA
    if (this.params.fxaaEnabled !== false) {
      current = this.applyFXAA(current);
    }

    // Passes 1a-1c: LensFlare (3-pass)
    if (this.params.lensFlare?.enabled !== false) {
      current = this.applyLensFlare(current);
    }

    // Pass 2: Bloom
    if (this.params.bloom?.enabled !== false) {
      current = this.applyBloom(current);
    }

    // Pass 3: Tone Mapping
    if (this.params.toneMapping?.enabled !== false) {
      current = this.applyToneMapping(current);
    }

    return current;
  }

  // ── setParams ──
  setParams(params: ATPostprocessStackParams): void {
    if (params.fxaaEnabled !== undefined) {
      this.params.fxaaEnabled = params.fxaaEnabled;
    }
    if (params.lensFlare) {
      this.params.lensFlare = { ...this.params.lensFlare, ...params.lensFlare };
    }
    if (params.lightVolume) {
      this.params.lightVolume = { ...this.params.lightVolume, ...params.lightVolume };
    }
    if (params.bloom) {
      this.params.bloom = { ...this.params.bloom, ...params.bloom };
    }
    if (params.toneMapping) {
      this.params.toneMapping = { ...this.params.toneMapping, ...params.toneMapping };
    }
  }

  // ── resize ──
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;

    const gl = this.gl;

    // Recreate all FBOs at new size
    const fbos: (keyof ATPostprocessStack)[] = [
      'fxaaFBO',
      'lensPrefilterFBO',
      'lensDownFBO',
      'lensUpFBO',
      'lensCompositeFBO',
      'bloomThresholdFBO',
      'bloomBlurHTempFBO',
      'bloomBlurVFBO',
      'bloomCompositeFBO',
      'tonemapFBO',
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
      this.fxaaProgram,
      this.lensPrefilterProgram,
      this.lensDownProgram,
      this.lensUpProgram,
      this.lensCompositeProgram,
      this.bloomThresholdProgram,
      this.blurHProgram,
      this.blurVProgram,
      this.bloomCompositeProgram,
      this.tonemapProgram,
    ];
    for (const p of programs) {
      if (p) gl.deleteProgram(p);
    }

    // Delete FBOs
    const fbos = [
      this.fxaaFBO,
      this.lensPrefilterFBO,
      this.lensDownFBO,
      this.lensUpFBO,
      this.lensCompositeFBO,
      this.bloomThresholdFBO,
      this.bloomBlurHTempFBO,
      this.bloomBlurVFBO,
      this.bloomCompositeFBO,
      this.tonemapFBO,
    ];
    for (const fbo of fbos) {
      if (fbo) destroyFBO(gl, fbo);
    }

    // Delete buffers
    if (this.quadVAO) gl.deleteBuffer(this.quadVAO);
    if (this.quadUVBuffer) gl.deleteBuffer(this.quadUVBuffer);

    // Delete white texture
    if (this.whiteTex) gl.deleteTexture(this.whiteTex);

    // Clear references
    this.fxaaProgram = null;
    this.lensPrefilterProgram = null;
    this.lensDownProgram = null;
    this.lensUpProgram = null;
    this.lensCompositeProgram = null;
    this.bloomThresholdProgram = null;
    this.blurHProgram = null;
    this.blurVProgram = null;
    this.bloomCompositeProgram = null;
    this.tonemapProgram = null;

    this.fxaaFBO = null;
    this.lensPrefilterFBO = null;
    this.lensDownFBO = null;
    this.lensUpFBO = null;
    this.lensCompositeFBO = null;
    this.bloomThresholdFBO = null;
    this.bloomBlurHTempFBO = null;
    this.bloomBlurVFBO = null;
    this.bloomCompositeFBO = null;
    this.tonemapFBO = null;

    this.quadVAO = null;
    this.quadUVBuffer = null;
    this.whiteTex = null;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// §5  Exports
// ═════════════════════════════════════════════════════════════════════════════

export type {
  LensFlareParams,
  LightVolumeParams,
  BloomParams,
  ToneMappingParams,
  ATPostprocessStackParams,
  FBO,
};
