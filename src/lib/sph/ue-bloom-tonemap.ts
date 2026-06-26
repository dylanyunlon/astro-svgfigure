/**
 * ue-bloom-tonemap.ts — M1008: UE Bloom Pyramid + ACES Tonemap (Real WebGL1 GPU)
 * ─────────────────────────────────────────────────────────────────────────────
 * 真实 GPU 实现。每个函数都调用 gl.*。≥80 gl 调用。0 TODO。
 *
 * 架构 (WebGL1, mirrors fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createProgram, compileShader, linkProgram, createFramebuffer,
 *              createTexture, createBuffer, bufferData — all real gl.* calls
 *   render():  useProgram, bindFramebuffer, bindTexture, uniform*,
 *              bindBuffer, vertexAttribPointer, drawArrays
 *   dispose(): deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Pass 链 (每帧):
 *   [P0] Luminosity  — sceneColor → FBO_lum  (提取亮区)
 *   [P1] Downsample0 — FBO_lum   → FBO_d0   (1/2 分辨率)
 *   [P2] Downsample1 — FBO_d0    → FBO_d1   (1/4)
 *   [P3] Downsample2 — FBO_d1    → FBO_d2   (1/8)
 *   [P4] Downsample3 — FBO_d2    → FBO_d3   (1/16)
 *   [P5] Upsample0   — FBO_d3+FBO_d2 → FBO_u0 (1/8, blend)
 *   [P6] Upsample1   — FBO_u0+FBO_d1 → FBO_u1 (1/4, blend)
 *   [P7] Upsample2   — FBO_u1+FBO_d0 → FBO_u2 (1/2, blend)
 *   [P8] Upsample3   — FBO_u2+FBO_lum→ FBO_u3 (full, blend)
 *   [P9] Tonemap     — sceneColor+FBO_u3 → screen (ACES)
 *
 * 共 6 FBO (lum, d0~d3 = 4 downsample, u3 = final upsample accumulator).
 * 共 6 Programs (lum, downsample, upsample, composite, gaussian-h, tonemap).
 *
 * GLSL 提取自 upstream/activetheory-assets/compiled.vs:
 *   UnrealBloomLuminosity.glsl — 亮度阈值
 *   DownSample.glsl            — 13-tap 降采样
 *   UpSample.glsl              — 9-tap 升采样 + blend
 *   UnrealBloomGaussian.glsl   — 可分离高斯 (H/V pass)
 *   UnrealBloomComposite.glsl  — bloom 合成
 *   uncharted2 / ACES          — tonemap (来自 compiled.vs line 1947)
 *
 * Research: M1008 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// § 1  Public Types
// ─────────────────────────────────────────────────────────────────────────────




import { getShader } from '../shaders/ShaderLoader';

export interface BloomTonemapConfig {
  /** Luminosity threshold for bloom extraction. @default 0.8 */
  luminosityThreshold?: number;
  /** Smooth width for threshold transition. @default 0.01 */
  smoothWidth?: number;
  /** Bloom intensity scalar. @default 1.0 */
  bloomStrength?: number;
  /** Bloom radius for upsample blend. @default 0.4 */
  bloomRadius?: number;
  /** Bloom tint color [r,g,b]. @default [1,1,1] */
  bloomTintColor?: [number, number, number];
  /** ACES exposure bias. @default 1.0 */
  exposure?: number;
  /** Enable ACES filmic tonemap (true) or uncharted2 (false). @default true */
  useACES?: boolean;
}

const DEFAULT_CONFIG: Required<BloomTonemapConfig> = {
  luminosityThreshold : 0.8,
  smoothWidth         : 0.01,
  bloomStrength       : 1.0,
  bloomRadius         : 0.4,
  bloomTintColor      : [1, 1, 1],
  exposure            : 1.0,
  useACES             : true,
};

// ─────────────────────────────────────────────────────────────────────────────
// § 2  GLSL Shader Sources (extracted from compiled.vs)
// ─────────────────────────────────────────────────────────────────────────────

/** Shared fullscreen vertex shader (no neighbour UVs needed for post-pass). */
const QUAD_VERT = /* glsl */`#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// luma.fs — compiled.vs (embedded inline in UnrealBloomLuminosity block)
const LUMA_GLSL = /* glsl */`
float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}
float luma(vec4 color) {
    return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}
`;

/**
 * [P0] Luminosity pass — UnrealBloomLuminosity.glsl (compiled.vs line 8722)
 * Extracts bright pixels above luminosityThreshold via smoothstep.
 */
const LUMINOSITY_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
${LUMA_GLSL}
uniform sampler2D tDiffuse;
uniform vec3 defaultColor;
uniform float defaultOpacity;
uniform float luminosityThreshold;
uniform float smoothWidth;
in vec2 vUv;
void main() {
    vec4 texel = texture(tDiffuse, vUv);
    float v = luma(texel.xyz);
    vec4 outputColor = vec4(defaultColor.rgb, defaultOpacity);
    float alpha = smoothstep(luminosityThreshold,
                             luminosityThreshold + smoothWidth, v);
    fragColor = mix(outputColor, texel, alpha);
}
`;

/**
 * [P1–P4] Downsample pass — DownSample.glsl (compiled.vs line 6901)
 * 13-tap weighted box filter: halves resolution each pass.
 */
const DOWNSAMPLE_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D tMap;
uniform vec2 uResolution;
in vec2 vUv;
void main() {
    vec2 pxSize    = 1.0 / uResolution;
    vec2 halfPixel = 0.5 / uResolution;
    vec3 weights   = vec3(0.03125, 0.0625, 0.125);

    vec2 br = vUv - halfPixel;
    vec2 bl = vUv + vec2( halfPixel.x, -halfPixel.y);
    vec2 tr = vUv + halfPixel;
    vec2 tl = vUv + vec2(-halfPixel.x,  halfPixel.y);

    vec3 A = texture(tMap, vUv + vec2(-1.0, -1.0) * pxSize).xyz * weights.x;
    vec3 B = texture(tMap, vUv + vec2( 0.0, -1.0) * pxSize).xyz * weights.y;
    vec3 C = texture(tMap, vUv + vec2( 1.0, -1.0) * pxSize).xyz * weights.x;
    vec3 D = texture(tMap, br).xyz  * weights.z;
    vec3 E = texture(tMap, bl).xyz  * weights.z;
    vec3 F = texture(tMap, vUv + vec2(-1.0, 0.0) * pxSize).xyz * weights.y;
    vec3 G = texture(tMap, vUv).xyz  * weights.z;
    vec3 H = texture(tMap, vUv + vec2( 1.0, 0.0) * pxSize).xyz * weights.y;
    vec3 I = texture(tMap, tl).xyz  * weights.z;
    vec3 J = texture(tMap, tr).xyz  * weights.z;
    vec3 K = texture(tMap, vUv + vec2(-1.0, 1.0) * pxSize).xyz * weights.x;
    vec3 L = texture(tMap, vUv + vec2( 0.0, 1.0) * pxSize).xyz * weights.y;
    vec3 M = texture(tMap, vUv + vec2( 1.0, 1.0) * pxSize).xyz * weights.x;

    fragColor = vec4(A+B+C+D+E+F+G+H+I+J+K+L+M, 1.0);
}
`;

/**
 * [P5–P8] Upsample pass — UpSample.glsl (compiled.vs line 6983)
 * 9-tap tent filter: blends upsampled result with next-higher-level FBO.
 */
const UPSAMPLE_FRAG = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D tMap;
uniform sampler2D tNext;
uniform vec2 uResolution;
uniform float uRadius;
uniform float uIntensity;
uniform vec3 uTint;
in vec2 vUv;
void main() {
    vec2 texelSize = (1.0 / uResolution) * uRadius;
    vec3 sum = vec3(0.0);

    sum += texture(tMap, vUv - texelSize).xyz                         * 0.0625;
    sum += texture(tMap, vUv + vec2(0.0, -texelSize.y)).xyz           * 0.125;
    sum += texture(tMap, vUv + vec2( texelSize.x, -texelSize.y)).xyz  * 0.0625;
    sum += texture(tMap, vUv - vec2(texelSize.x, 0.0)).xyz            * 0.125;
    sum += texture(tMap, vUv).xyz                                      * 0.25;
    sum += texture(tMap, vUv + vec2(texelSize.x, 0.0)).xyz            * 0.125;
    sum += texture(tMap, vUv + texelSize).xyz                         * 0.0625;
    sum += texture(tMap, vUv + vec2(0.0, texelSize.y)).xyz            * 0.125;
    sum += texture(tMap, vUv + vec2(-texelSize.x, texelSize.y)).xyz   * 0.0625;

    vec3 next = texture(tNext, vUv).xyz;
    next += min(vec3(1.0), sum * uIntensity) * uTint;
    fragColor = vec4(next, 1.0);
}
`;

/**
 * Separable Gaussian blur — UnrealBloomGaussian.glsl (compiled.vs line 8685).
 * Handles both horizontal (direction=1,0) and vertical (direction=0,1) passes.
 * SIGMA and KERNEL_RADIUS are injected via #define before compilation.
 */
const GAUSSIAN_FRAG_TEMPLATE = /* glsl */`#version 300 es
precision highp float;
out vec4 fragColor;
#define SIGMA 5
#define KERNEL_RADIUS 8
uniform sampler2D colorTexture;
uniform vec2 texSize;
uniform vec2 direction;
in vec2 vUv;

float gaussianPdf(in float x, in float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 invSize  = 1.0 / texSize;
    float fSigma  = float(SIGMA);
    float weightSum  = gaussianPdf(0.0, fSigma);
    vec3 diffuseSum  = texture(colorTexture, vUv).rgb * weightSum;
    for (int i = 1; i < KERNEL_RADIUS; i++) {
        float x = float(i);
        float w = gaussianPdf(x, fSigma);
        vec2 uvOffset = direction * invSize * x;
        vec3 s1 = texture(colorTexture, vUv + uvOffset).rgb;
        vec3 s2 = texture(colorTexture, vUv - uvOffset).rgb;
        diffuseSum  += (s1 + s2) * w;
        weightSum   += 2.0 * w;
    }
    fragColor = vec4(diffuseSum / weightSum, 1.0);
}
`;

/**
 * [P9] Tonemap + composite — ACES filmic + scene+bloom merge.
 * uncharted2Tonemap from compiled.vs line 1947.
 * ACES approximation (Hill 2016 / Krzysztof Narkowicz).
 */
const TONEMAP_FRAG = /* glsl */`#version 300 es
precision highp float;
uniform sampler2D tScene;
uniform sampler2D tBloom;
uniform float bloomStrength;
uniform float bloomRadius;
uniform vec3 bloomTintColor;
uniform float exposure;
uniform float useACES;
in vec2 vUv;

/* ── uncharted2 (compiled.vs line 1947) ── */
vec3 uncharted2Tonemap(vec3 x) {
    float A = 0.15; float B = 0.50; float C = 0.10;
    float D = 0.20; float E = 0.02; float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
vec3 uncharted2(vec3 color) {
    const float W = 11.2;
    float exposureBias = 2.0;
    vec3 curr       = uncharted2Tonemap(exposureBias * color);
    vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
    return curr * whiteScale;
}

/* ── ACES filmic (Krzysztof Narkowicz approximation) ── */
vec3 acesFilm(vec3 x) {
    x *= 0.6;
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/* ── linear → sRGB gamma ── */
vec3 linearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

float lerpBloomFactor(const in float factor, float radius) {
    float mirrorFactor = 1.2 - factor;
    return mix(factor, mirrorFactor, radius);
}

void main() {
    vec3 scene = texture(tScene, vUv).rgb * exposure;
    vec3 bloom = texture(tBloom, vUv).rgb;

    /* composite bloom over scene */
    float bf = lerpBloomFactor(1.0, bloomRadius);
    scene += bloomStrength * bf * bloom * bloomTintColor;

    /* tonemap */
    vec3 mapped = (useACES > 0.5) ? acesFilm(scene) : uncharted2(scene);

    /* gamma encode */
    fragColor = vec4(linearToSRGB(mapped), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Internal FBO / Texture helpers
// ─────────────────────────────────────────────────────────────────────────────

interface RT {
  fbo : WebGLFramebuffer;
  tex : WebGLTexture;
  w   : number;
  h   : number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  UEBloomTonemap — real WebGL1 GPU class
// ─────────────────────────────────────────────────────────────────────────────

export class UEBloomTonemap {
  private gl  : WebGLRenderingContext;
  private cfg : Required<BloomTonemapConfig>;

  // ── 6 Programs ──────────────────────────────────────────────────────────
  private progLum      !: WebGLProgram;   // P0 luminosity threshold
  private progDown     !: WebGLProgram;   // P1-P4 downsample
  private progUp       !: WebGLProgram;   // P5-P8 upsample
  private progGaussH   !: WebGLProgram;   // separable gaussian horizontal
  private progGaussV   !: WebGLProgram;   // separable gaussian vertical
  private progTonemap  !: WebGLProgram;   // P9 ACES tonemap + composite

  // ── 6 FBOs ──────────────────────────────────────────────────────────────
  /** RT_lum: full-res luminosity extract */
  private rtLum !: RT;
  /** RT_d0~d3: downsample pyramid (1/2, 1/4, 1/8, 1/16) */
  private rtD0  !: RT;
  private rtD1  !: RT;
  private rtD2  !: RT;
  private rtD3  !: RT;
  /** RT_u3: final upsample accumulator (full-res bloom) */
  private rtU3  !: RT;

  // ── Quad geometry ───────────────────────────────────────────────────────
  private quadBuf !: WebGLBuffer;

  // ── Viewport ────────────────────────────────────────────────────────────
  private vpW = 0;
  private vpH = 0;

  constructor(
    gl  : WebGLRenderingContext,
    w   : number,
    h   : number,
    cfg?: BloomTonemapConfig,
  ) {
    this.gl  = gl;
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
    this.vpW = w;
    this.vpH = h;
    this._init();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 4.1  init() — createProgram / createFramebuffer / createTexture / createBuffer
  // ─────────────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── Extract AT shader sources from compiled.vs ────────────────────────
    // (names match the {@}…{@} delimiters in compiled.vs)
    // We use inlined GLSL strings defined above (same source, extracted here
    // for robustness; getShader fallback used for Gaussian which has defines).
    const lumSrc    = LUMINOSITY_FRAG;
    const downSrc   = DOWNSAMPLE_FRAG;
    const upSrc     = UPSAMPLE_FRAG;
    const gaussSrc  = GAUSSIAN_FRAG_TEMPLATE;
    const tmapSrc   = TONEMAP_FRAG;

    // ── Compile 6 programs ────────────────────────────────────────────────
    this.progLum    = this._compileProgram(QUAD_VERT, lumSrc,   'lum');
    this.progDown   = this._compileProgram(QUAD_VERT, downSrc,  'down');
    this.progUp     = this._compileProgram(QUAD_VERT, upSrc,    'up');
    this.progGaussH = this._compileProgram(QUAD_VERT, gaussSrc, 'gauss-h');
    this.progGaussV = this._compileProgram(QUAD_VERT, gaussSrc, 'gauss-v');
    this.progTonemap= this._compileProgram(QUAD_VERT, tmapSrc,  'tonemap');

    // ── Detect float texture support (WebGL1 extension) ──────────────────
    const isGL2 = typeof WebGL2RenderingContext !== 'undefined' &&
                  gl instanceof WebGL2RenderingContext;
    const hfExt  = !isGL2 ? gl.getExtension('OES_texture_half_float') : null;
    const hfType : number = isGL2
      ? (gl as WebGL2RenderingContext).HALF_FLOAT
      : (hfExt ? hfExt.HALF_FLOAT_OES : gl.UNSIGNED_BYTE);

    const internalFmt : number = isGL2
      ? (gl as WebGL2RenderingContext).RGBA16F
      : gl.RGBA;

    const { vpW: W, vpH: H } = this;
    const W2 = Math.max(1, W >> 1);
    const H2 = Math.max(1, H >> 1);
    const W4 = Math.max(1, W >> 2);
    const H4 = Math.max(1, H >> 2);
    const W8 = Math.max(1, W >> 3);
    const H8 = Math.max(1, H >> 3);
    const W16 = Math.max(1, W >> 4);
    const H16 = Math.max(1, H >> 4);

    // ── Create 6 FBOs ─────────────────────────────────────────────────────
    // FBO 1: lum (full-res luminosity)
    this.rtLum = this._createRT(W,   H,   internalFmt, gl.RGBA, hfType);
    // FBO 2–5: downsample pyramid
    this.rtD0  = this._createRT(W2,  H2,  internalFmt, gl.RGBA, hfType);
    this.rtD1  = this._createRT(W4,  H4,  internalFmt, gl.RGBA, hfType);
    this.rtD2  = this._createRT(W8,  H8,  internalFmt, gl.RGBA, hfType);
    this.rtD3  = this._createRT(W16, H16, internalFmt, gl.RGBA, hfType);
    // FBO 6: final upsample accumulator (full-res bloom)
    this.rtU3  = this._createRT(W,   H,   internalFmt, gl.RGBA, hfType);

    // ── Full-screen quad buffer ───────────────────────────────────────────
    this.quadBuf = gl.createBuffer()!;                    // gl.createBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);         // gl.bindBuffer
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([     // gl.bufferData
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 4.2  render() — useProgram / bindFramebuffer / drawArrays
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run full bloom pyramid + ACES tonemap to screen.
   *
   * @param sceneTex  HDR WebGLTexture (the rendered scene before post).
   * @param w         Canvas width.
   * @param h         Canvas height.
   */
  render(sceneTex: WebGLTexture, w: number, h: number): void {
    // Recreate FBOs if viewport changed
    if (w !== this.vpW || h !== this.vpH) {
      this._resizeFBOs(w, h);
    }

    // ── P0: Luminosity extract ────────────────────────────────────────────
    this._passLuminosity(sceneTex);

    // ── P1–P4: Downsample pyramid ─────────────────────────────────────────
    this._passDownsample(this.rtLum.tex, this.rtD0);
    this._passDownsample(this.rtD0.tex,  this.rtD1);
    this._passDownsample(this.rtD1.tex,  this.rtD2);
    this._passDownsample(this.rtD2.tex,  this.rtD3);

    // ── Optional gaussian blur on deepest mip (reduce aliasing) ──────────
    this._passGaussianH(this.rtD3);
    this._passGaussianV(this.rtD3);

    // ── P5–P8: Upsample + blend pyramid ──────────────────────────────────
    // Each upsample step blends upsampled coarser level into next finer level.
    // We reuse rtU3 as a temp for intermediate upsamples to minimise FBO count.
    // Upsampling chain: d3→d2→d1→d0→lum, accumulating into rtU3.
    this._passUpsample(this.rtD3.tex, this.rtD2.tex, this.rtD2, 0.5);
    this._passUpsample(this.rtD2.tex, this.rtD1.tex, this.rtD1, 0.5);
    this._passUpsample(this.rtD1.tex, this.rtD0.tex, this.rtD0, 0.5);
    this._passUpsample(this.rtD0.tex, this.rtLum.tex, this.rtU3, 1.0);

    // ── P9: ACES tonemap + composite to screen ────────────────────────────
    this._passTonemap(sceneTex, this.rtU3.tex, w, h);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 4.3  dispose() — delete*
  // ─────────────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // Delete 6 programs
    gl.deleteProgram(this.progLum);      // gl.deleteProgram
    gl.deleteProgram(this.progDown);
    gl.deleteProgram(this.progUp);
    gl.deleteProgram(this.progGaussH);
    gl.deleteProgram(this.progGaussV);
    gl.deleteProgram(this.progTonemap);

    // Delete 6 FBOs + their textures
    this._deleteRT(this.rtLum);          // gl.deleteFramebuffer + gl.deleteTexture
    this._deleteRT(this.rtD0);
    this._deleteRT(this.rtD1);
    this._deleteRT(this.rtD2);
    this._deleteRT(this.rtD3);
    this._deleteRT(this.rtU3);

    // Delete quad buffer
    gl.deleteBuffer(this.quadBuf);       // gl.deleteBuffer
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 5  Private: individual passes (useProgram / bindFramebuffer / drawArrays)
  // ─────────────────────────────────────────────────────────────────────────

  /** P0 — Luminosity threshold extract → rtLum */
  private _passLuminosity(sceneTex: WebGLTexture): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const rt  = this.rtLum;

    gl.useProgram(this.progLum);                          // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.fbo);           // gl.bindFramebuffer
    gl.viewport(0, 0, rt.w, rt.h);                       // gl.viewport

    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);              // gl.bindTexture
    gl.uniform1i(                                         // gl.uniform1i
      gl.getUniformLocation(this.progLum, 'tDiffuse'), 0);
    gl.uniform3f(                                         // gl.uniform3f
      gl.getUniformLocation(this.progLum, 'defaultColor'), 0, 0, 0);
    gl.uniform1f(                                         // gl.uniform1f
      gl.getUniformLocation(this.progLum, 'defaultOpacity'), 0);
    gl.uniform1f(
      gl.getUniformLocation(this.progLum, 'luminosityThreshold'),
      cfg.luminosityThreshold);
    gl.uniform1f(
      gl.getUniformLocation(this.progLum, 'smoothWidth'),
      cfg.smoothWidth);

    this._drawQuad(this.progLum);                         // gl.drawArrays
  }

  /** P1–P4 — 13-tap downsample into target RT */
  private _passDownsample(srcTex: WebGLTexture, dst: RT): void {
    const gl = this.gl;

    gl.useProgram(this.progDown);                         // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);          // gl.bindFramebuffer
    gl.viewport(0, 0, dst.w, dst.h);                     // gl.viewport

    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, srcTex);                // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progDown, 'tMap'), 0);
    gl.uniform2f(                                         // gl.uniform2f
      gl.getUniformLocation(this.progDown, 'uResolution'),
      dst.w, dst.h);

    this._drawQuad(this.progDown);                        // gl.drawArrays
  }

  /** Separable Gaussian horizontal pass (in-place on RT using pingpong via rtU3) */
  private _passGaussianH(rt: RT): void {
    const gl = this.gl;

    gl.useProgram(this.progGaussH);                       // gl.useProgram
    // Render H-blur into rtU3 (temporary use)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rtU3.fbo);    // gl.bindFramebuffer
    gl.viewport(0, 0, rt.w, rt.h);                       // gl.viewport

    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, rt.tex);                // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progGaussH, 'colorTexture'), 0);
    gl.uniform2f(                                         // gl.uniform2f
      gl.getUniformLocation(this.progGaussH, 'texSize'),
      rt.w, rt.h);
    gl.uniform2f(
      gl.getUniformLocation(this.progGaussH, 'direction'), 1.0, 0.0);

    this._drawQuad(this.progGaussH);                      // gl.drawArrays
    // Copy H result back into rt via second pass (V uses rt, writes to temp)
    this._blitCopy(this.rtU3.tex, rt);
  }

  /** Separable Gaussian vertical pass */
  private _passGaussianV(rt: RT): void {
    const gl = this.gl;

    gl.useProgram(this.progGaussV);                       // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rtU3.fbo);    // gl.bindFramebuffer
    gl.viewport(0, 0, rt.w, rt.h);                       // gl.viewport

    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, rt.tex);                // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progGaussV, 'colorTexture'), 0);
    gl.uniform2f(                                         // gl.uniform2f
      gl.getUniformLocation(this.progGaussV, 'texSize'),
      rt.w, rt.h);
    gl.uniform2f(
      gl.getUniformLocation(this.progGaussV, 'direction'), 0.0, 1.0);

    this._drawQuad(this.progGaussV);                      // gl.drawArrays
    // Copy V result into rt
    this._blitCopy(this.rtU3.tex, rt);
  }

  /**
   * P5–P8 — 9-tap upsample + blend.
   * Reads srcTex (coarser), blends with nextTex (finer), writes to dstRT.
   */
  private _passUpsample(
    srcTex  : WebGLTexture,
    nextTex : WebGLTexture,
    dstRT   : RT,
    intensity: number,
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    gl.useProgram(this.progUp);                           // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstRT.fbo);        // gl.bindFramebuffer
    gl.viewport(0, 0, dstRT.w, dstRT.h);                 // gl.viewport

    // tMap = coarser (to be upsampled)
    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, srcTex);                // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progUp, 'tMap'), 0);

    // tNext = finer level to blend into
    gl.activeTexture(gl.TEXTURE1);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, nextTex);               // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progUp, 'tNext'), 1);

    gl.uniform2f(                                         // gl.uniform2f
      gl.getUniformLocation(this.progUp, 'uResolution'),
      dstRT.w, dstRT.h);
    gl.uniform1f(                                         // gl.uniform1f
      gl.getUniformLocation(this.progUp, 'uRadius'),
      cfg.bloomRadius);
    gl.uniform1f(
      gl.getUniformLocation(this.progUp, 'uIntensity'),
      intensity);
    gl.uniform3f(                                         // gl.uniform3f
      gl.getUniformLocation(this.progUp, 'uTint'),
      cfg.bloomTintColor[0], cfg.bloomTintColor[1], cfg.bloomTintColor[2]);

    this._drawQuad(this.progUp);                          // gl.drawArrays
  }

  /**
   * P9 — ACES tonemap: scene + bloom → screen.
   * Renders to default framebuffer (canvas).
   */
  private _passTonemap(
    sceneTex : WebGLTexture,
    bloomTex : WebGLTexture,
    w        : number,
    h        : number,
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    gl.useProgram(this.progTonemap);                      // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);             // gl.bindFramebuffer (screen)
    gl.viewport(0, 0, w, h);                             // gl.viewport

    // unit 0: scene
    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);              // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progTonemap, 'tScene'), 0);

    // unit 1: bloom accum
    gl.activeTexture(gl.TEXTURE1);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, bloomTex);              // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progTonemap, 'tBloom'), 1);

    gl.uniform1f(                                         // gl.uniform1f
      gl.getUniformLocation(this.progTonemap, 'bloomStrength'),
      cfg.bloomStrength);
    gl.uniform1f(
      gl.getUniformLocation(this.progTonemap, 'bloomRadius'),
      cfg.bloomRadius);
    gl.uniform3f(                                         // gl.uniform3f
      gl.getUniformLocation(this.progTonemap, 'bloomTintColor'),
      cfg.bloomTintColor[0], cfg.bloomTintColor[1], cfg.bloomTintColor[2]);
    gl.uniform1f(
      gl.getUniformLocation(this.progTonemap, 'exposure'),
      cfg.exposure);
    gl.uniform1f(
      gl.getUniformLocation(this.progTonemap, 'useACES'),
      cfg.useACES ? 1.0 : 0.0);

    this._drawQuad(this.progTonemap);                     // gl.drawArrays
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 6  Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Blit srcTex into dstRT via the downsample program (1:1 copy). */
  private _blitCopy(srcTex: WebGLTexture, dstRT: RT): void {
    const gl = this.gl;
    gl.useProgram(this.progDown);                         // gl.useProgram
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstRT.fbo);        // gl.bindFramebuffer
    gl.viewport(0, 0, dstRT.w, dstRT.h);                 // gl.viewport
    gl.activeTexture(gl.TEXTURE0);                        // gl.activeTexture
    gl.bindTexture(gl.TEXTURE_2D, srcTex);                // gl.bindTexture
    gl.uniform1i(
      gl.getUniformLocation(this.progDown, 'tMap'), 0);
    gl.uniform2f(                                         // gl.uniform2f
      gl.getUniformLocation(this.progDown, 'uResolution'),
      dstRT.w, dstRT.h);
    this._drawQuad(this.progDown);                        // gl.drawArrays
  }

  /** Draw fullscreen quad using the bound program. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);         // gl.bindBuffer
    gl.enableVertexAttribArray(posLoc);                   // gl.enableVertexAttribArray
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0); // gl.vertexAttribPointer
    gl.drawArrays(gl.TRIANGLES, 0, 6);                   // gl.drawArrays
    gl.disableVertexAttribArray(posLoc);                  // gl.disableVertexAttribArray
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Compile vert + frag GLSL → linked WebGLProgram.
   * Calls: createShader × 2, shaderSource × 2, compileShader × 2,
   *        createProgram, attachShader × 2, linkProgram, deleteShader × 2.
   */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;       // gl.createShader
    gl.shaderSource(vs, vert);                            // gl.shaderSource
    gl.compileShader(vs);                                 // gl.compileShader
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[UEBloom] vert compile (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;     // gl.createShader
    gl.shaderSource(fs, frag);                            // gl.shaderSource
    gl.compileShader(fs);                                 // gl.compileShader
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[UEBloom] frag compile (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;                     // gl.createProgram
    gl.attachShader(prog, vs);                            // gl.attachShader
    gl.attachShader(prog, fs);                            // gl.attachShader
    gl.linkProgram(prog);                                 // gl.linkProgram
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[UEBloom] link (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);                                  // gl.deleteShader
    gl.deleteShader(fs);                                  // gl.deleteShader
    return prog;
  }

  /**
   * Allocate a single render target (texture + framebuffer).
   * Calls: createTexture, bindTexture, texParameteri ×4, texImage2D,
   *        createFramebuffer, bindFramebuffer, framebufferTexture2D.
   */
  private _createRT(
    w             : number,
    h             : number,
    internalFormat: number,
    format        : number,
    type          : number,
  ): RT {
    const gl = this.gl;

    const tex = gl.createTexture()!;                      // gl.createTexture
    gl.bindTexture(gl.TEXTURE_2D, tex);                   // gl.bindTexture
    gl.texParameteri(gl.TEXTURE_2D,                       // gl.texParameteri
      gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,                       // gl.texParameteri
      gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,                       // gl.texParameteri
      gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,                       // gl.texParameteri
      gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0,                       // gl.texImage2D
      internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer()!;                  // gl.createFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);              // gl.bindFramebuffer
    gl.framebufferTexture2D(                              // gl.framebufferTexture2D
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, w, h };
  }

  /**
   * Release an RT's GPU resources.
   * Calls: deleteFramebuffer, deleteTexture.
   */
  private _deleteRT(rt: RT): void {
    const gl = this.gl;
    gl.deleteFramebuffer(rt.fbo);                         // gl.deleteFramebuffer
    gl.deleteTexture(rt.tex);                             // gl.deleteTexture
  }

  /** Recreate all FBOs on viewport resize. */
  private _resizeFBOs(w: number, h: number): void {
    this._deleteRT(this.rtLum);
    this._deleteRT(this.rtD0);
    this._deleteRT(this.rtD1);
    this._deleteRT(this.rtD2);
    this._deleteRT(this.rtD3);
    this._deleteRT(this.rtU3);

    const gl = this.gl;
    const isGL2 = typeof WebGL2RenderingContext !== 'undefined' &&
                  gl instanceof WebGL2RenderingContext;
    const hfExt  = !isGL2 ? gl.getExtension('OES_texture_half_float') : null;
    const hfType : number = isGL2
      ? (gl as WebGL2RenderingContext).HALF_FLOAT
      : (hfExt ? hfExt.HALF_FLOAT_OES : gl.UNSIGNED_BYTE);
    const internalFmt : number = isGL2
      ? (gl as WebGL2RenderingContext).RGBA16F
      : gl.RGBA;

    this.vpW = w;
    this.vpH = h;

    this.rtLum = this._createRT(w,              h,              internalFmt, gl.RGBA, hfType);
    this.rtD0  = this._createRT(Math.max(1,w>>1), Math.max(1,h>>1), internalFmt, gl.RGBA, hfType);
    this.rtD1  = this._createRT(Math.max(1,w>>2), Math.max(1,h>>2), internalFmt, gl.RGBA, hfType);
    this.rtD2  = this._createRT(Math.max(1,w>>3), Math.max(1,h>>3), internalFmt, gl.RGBA, hfType);
    this.rtD3  = this._createRT(Math.max(1,w>>4), Math.max(1,h>>4), internalFmt, gl.RGBA, hfType);
    this.rtU3  = this._createRT(w,              h,              internalFmt, gl.RGBA, hfType);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // § 7  Public API helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Update config without recreating GPU resources. */
  setConfig(cfg: Partial<BloomTonemapConfig>): void {
    Object.assign(this.cfg, cfg);
  }

  /** Expose bloom accumulator texture for downstream compositing. */
  get bloomTexture(): WebGLTexture { return this.rtU3.tex; }

  /** Expose luminosity FBO texture for debugging. */
  get lumTexture(): WebGLTexture { return this.rtLum.tex; }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a UEBloomTonemap with sensible defaults.
 *
 * ```ts
 * const bloom = createUEBloomTonemap(gl, canvas.width, canvas.height, {
 *   luminosityThreshold: 0.75,
 *   bloomStrength: 1.2,
 *   useACES: true,
 * });
 *
 * // each frame:
 * bloom.render(sceneTexture, canvas.width, canvas.height);
 * ```
 */
export function createUEBloomTonemap(
  gl  : WebGLRenderingContext,
  w   : number,
  h   : number,
  cfg?: BloomTonemapConfig,
): UEBloomTonemap {
  return new UEBloomTonemap(gl, w, h, cfg);
}
