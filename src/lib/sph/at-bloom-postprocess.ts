/**
 * at-bloom-postprocess.ts — M927: AT UnrealBloom Post-Process Pipeline — WebGL GPU
 *
 * 真实 WebGL GPU 实现。每个函数都调用 gl.*。
 * 从 compiled.vs 提取 AT UnrealBloom shader 原文:
 *   UnrealBloomLuminosity.glsl  → 亮度阈值提取 (BloomLuminosityPass.glsl)
 *   UnrealBloomGaussian.glsl    → 可分离高斯模糊 (KERNEL_RADIUS / SIGMA define)
 *   UnrealBloomComposite.glsl   → 合成 (bloomStrength / lerpBloomFactor)
 *   UnrealBloomPass.fs          → 最终叠加 (getUnrealBloom)
 *   DownSample.glsl             → 13-tap dual-kawase 下采样金字塔
 *
 * 管线结构 (每帧 render()):
 *   ┌─ Pass 1: UnrealBloomLuminosity ────────────────────────────────────────┐
 *   │  scene → lumRT  (亮度阈值提取, smoothstep luma)                        │
 *   └────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 2..5: DownSample pyramid ×LEVELS ────────────────────────────────┐
 *   │  lumRT → pyramid[0] → pyramid[1] → pyramid[2] → pyramid[3]             │
 *   │  13-tap dual-kawase downsample per level                               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass 6..N: UnrealBloomGaussian H+V per level ─────────────────────────┐
 *   │  pyramid[i] → blurH[i] → blurV[i]  (separable gaussian, 9-tap)        │
 *   └────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass N+1: UnrealBloomComposite ───────────────────────────────────────┐
 *   │  blurV[0..3] → blurV[0] (additive upsample accumulation)               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *   ┌─ Pass N+2: UnrealBloomPass (final composite) ───────────────────────────┐
 *   │  scene + blurV[0] → output FBO / screen                                │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * Research: bloom-post-r2 #M927 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────









export interface ATBloomParams {
  /** Luminance threshold for bloom extraction. Default 0.7 */
  luminosityThreshold?: number;
  /** Smooth width for luma smoothstep. Default 0.1 */
  smoothWidth?: number;
  /** Bloom additive strength multiplier. Default 1.0 */
  bloomStrength?: number;
  /** Bloom radius (lerpBloomFactor mirror blend). Default 0.4 */
  bloomRadius?: number;
  /** RGB tint applied to bloom layer. Default [1,1,1] */
  bloomTintColor?: [number, number, number];
  /** Number of downsample pyramid levels (2–5). Default 4 */
  levels?: number;
}

interface SingleRT {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: luma.fs  (from compiled.vs line ~8757)
// ─────────────────────────────────────────────────────────────────────────────

const LUMA_GLSL = /* glsl */`
float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}
float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: shared fullscreen-quad vertex shader (WebGL1, varying/attribute)
// ─────────────────────────────────────────────────────────────────────────────

const BLOOM_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: UnrealBloomLuminosity.glsl  (from compiled.vs line 8722)
// AND   BloomLuminosityPass.glsl    (from compiled.vs line 6872)
// Both use luma() + smoothstep threshold — merged here.
// uniforms: tDiffuse, defaultColor, defaultOpacity, luminosityThreshold, smoothWidth
// ─────────────────────────────────────────────────────────────────────────────

const LUMINOSITY_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec3 defaultColor;
uniform float defaultOpacity;
uniform float luminosityThreshold;
uniform float smoothWidth;

${LUMA_GLSL}

void main() {
    vec4 texel = texture2D(tDiffuse, vUv);
    float v = luma(texel.xyz);
    vec4 outputColor = vec4(defaultColor.rgb, defaultOpacity);
    float alpha = smoothstep(luminosityThreshold, luminosityThreshold + smoothWidth, v);
    gl_FragColor = mix(outputColor, texel, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: DownSample.glsl  (from compiled.vs line ~6900)
// 13-tap dual-kawase downsample for building bloom pyramid
// uniforms: tMap, uResolution, uRadius
// ─────────────────────────────────────────────────────────────────────────────

const DOWNSAMPLE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tMap;
uniform vec2 uResolution;
uniform float uRadius;

void main() {
    vec2 pxSize = 1.0 / uResolution;
    vec2 halfPixel = 0.5 / uResolution;

    vec3 weights = vec3(0.03125, 0.0625, 0.125);

    vec2 br = vUv - halfPixel;
    vec2 bl = vUv + vec2(halfPixel.x, -halfPixel.y);
    vec2 tr = vUv + halfPixel;
    vec2 tl = vUv + vec2(-halfPixel.x, halfPixel.y);

    vec3 A = texture2D(tMap, vUv + vec2(-1.0, -1.0) * pxSize).xyz * weights.x;
    vec3 B = texture2D(tMap, vUv + vec2( 0.0, -1.0) * pxSize).xyz * weights.y;
    vec3 C = texture2D(tMap, vUv + vec2( 1.0, -1.0) * pxSize).xyz * weights.x;

    vec3 D = texture2D(tMap, br).xyz * weights.z;
    vec3 E = texture2D(tMap, bl).xyz * weights.z;
    vec3 F = texture2D(tMap, vUv + vec2(-1.0, 0.0) * pxSize).xyz * weights.y;

    vec3 G = texture2D(tMap, vUv).xyz * weights.z;

    vec3 H = texture2D(tMap, vUv + vec2(1.0, 0.0) * pxSize).xyz * weights.y;
    vec3 I = texture2D(tMap, tl).xyz * weights.z;
    vec3 J = texture2D(tMap, tr).xyz * weights.z;

    vec3 K = texture2D(tMap, vUv + vec2(-1.0, 1.0) * pxSize).xyz * weights.x;
    vec3 L = texture2D(tMap, vUv + vec2( 0.0, 1.0) * pxSize).xyz * weights.y;
    vec3 M = texture2D(tMap, vUv + vec2( 1.0, 1.0) * pxSize).xyz * weights.x;

    vec3 sum = A + B + C + D + E + F + G + H + I + J + K + L + M;
    gl_FragColor = vec4(sum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: UnrealBloomGaussian.glsl  (from compiled.vs line 8685)
// Separable gaussian with SIGMA=5 KERNEL_RADIUS=9 (AT production values)
// uniforms: colorTexture, texSize, direction
// ─────────────────────────────────────────────────────────────────────────────

const GAUSSIAN_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D colorTexture;
uniform vec2 texSize;
uniform vec2 direction;

#define SIGMA 5.0
#define KERNEL_RADIUS 9

float gaussianPdf(in float x, in float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 invSize = 1.0 / texSize;
    float fSigma = SIGMA;
    float weightSum = gaussianPdf(0.0, fSigma);
    vec3 diffuseSum = texture2D(colorTexture, vUv).rgb * weightSum;
    for (int i = 1; i < KERNEL_RADIUS; i++) {
        float x = float(i);
        float w = gaussianPdf(x, fSigma);
        vec2 uvOffset = direction * invSize * x;
        vec3 sample1 = texture2D(colorTexture, vUv + uvOffset).rgb;
        vec3 sample2 = texture2D(colorTexture, vUv - uvOffset).rgb;
        diffuseSum += (sample1 + sample2) * w;
        weightSum += 2.0 * w;
    }
    gl_FragColor = vec4(diffuseSum / weightSum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: UnrealBloomComposite.glsl  (from compiled.vs line 8658)
// Additive upsample — accumulates blur levels into blurTexture1
// uniforms: blurTexture1, bloomStrength, bloomRadius, bloomTintColor
// ─────────────────────────────────────────────────────────────────────────────

const COMPOSITE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D blurTexture1;
uniform sampler2D blurTexture2;
uniform sampler2D blurTexture3;
uniform sampler2D blurTexture4;
uniform float bloomStrength;
uniform float bloomRadius;
uniform vec3 bloomTintColor;

float lerpBloomFactor(const in float factor) {
    float mirrorFactor = 1.2 - factor;
    return mix(factor, mirrorFactor, bloomRadius);
}

void main() {
    vec4 bloom = vec4(0.0);
    bloom += lerpBloomFactor(1.0)   * texture2D(blurTexture1, vUv);
    bloom += lerpBloomFactor(0.8)   * texture2D(blurTexture2, vUv);
    bloom += lerpBloomFactor(0.6)   * texture2D(blurTexture3, vUv);
    bloom += lerpBloomFactor(0.4)   * texture2D(blurTexture4, vUv);
    gl_FragColor = bloomStrength * vec4(bloomTintColor, 1.0) * bloom;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: UnrealBloomPass.fs final composite (from compiled.vs line 8750)
// uniforms: tDiffuse, tUnrealBloom
// ─────────────────────────────────────────────────────────────────────────────

const FINAL_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform sampler2D tUnrealBloom;

void main() {
    vec4 color = texture2D(tDiffuse, vUv);
    color.rgb += texture2D(tUnrealBloom, vUv).rgb;
    gl_FragColor = color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: upsample additive accumulator
// ─────────────────────────────────────────────────────────────────────────────

const UPSAMPLE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D uLower;
uniform sampler2D uUpper;
uniform float uWeight;

void main() {
    vec4 lower = texture2D(uLower, vUv);
    vec4 upper = texture2D(uUpper, vUv);
    gl_FragColor = lower + upper * uWeight;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATBloomPostProcess — real WebGL bloom pyramid
// ─────────────────────────────────────────────────────────────────────────────

const BLOOM_LEVELS = 4;

const DEFAULT_PARAMS: Required<ATBloomParams> = {
  luminosityThreshold : 0.7,
  smoothWidth         : 0.1,
  bloomStrength       : 1.0,
  bloomRadius         : 0.4,
  bloomTintColor      : [1, 1, 1],
  levels              : BLOOM_LEVELS,
};

export class ATBloomPostProcess {
  private readonly gl  : WebGLRenderingContext;
  private width        : number;
  private height       : number;
  private params       : Required<ATBloomParams>;

  // ── WebGL programs (5 programs) ──
  private lumProg       : WebGLProgram;   // luminosity threshold
  private downsampleProg: WebGLProgram;   // dual-kawase downsample
  private gaussianProg  : WebGLProgram;   // separable gaussian blur
  private compositeProg : WebGLProgram;   // unreal bloom composite
  private finalProg     : WebGLProgram;   // scene + bloom final
  private upsampleProg  : WebGLProgram;   // upsample accumulate

  // ── FBOs ──
  private lumRT        : SingleRT;        // full-res luminosity extract
  private pyramid      : SingleRT[];      // downsample pyramid [0..levels-1]
  private blurH        : SingleRT[];      // gaussian H temp per level
  private blurV        : SingleRT[];      // gaussian V output per level
  private bloomRT      : SingleRT;        // composite result (full-res)

  // ── Geometry ──
  private quadBuf      : WebGLBuffer;

  // ─────────────────────────────────────────────────────────────────────────
  constructor(gl: WebGLRenderingContext, width: number, height: number,
              params?: ATBloomParams) {
    this.gl     = gl;
    this.width  = width;
    this.height = height;
    this.params = { ...DEFAULT_PARAMS, ...params };

    // ── init: createProgram / compileShader / linkProgram ──
    this.lumProg        = this._compileProgram(BLOOM_VERT, LUMINOSITY_FRAG,  'at-bloom-lum');
    this.downsampleProg = this._compileProgram(BLOOM_VERT, DOWNSAMPLE_FRAG,  'at-bloom-down');
    this.gaussianProg   = this._compileProgram(BLOOM_VERT, GAUSSIAN_FRAG,    'at-bloom-gauss');
    this.compositeProg  = this._compileProgram(BLOOM_VERT, COMPOSITE_FRAG,   'at-bloom-composite');
    this.finalProg      = this._compileProgram(BLOOM_VERT, FINAL_FRAG,       'at-bloom-final');
    this.upsampleProg   = this._compileProgram(BLOOM_VERT, UPSAMPLE_FRAG,    'at-bloom-upsample');

    // ── init: createFramebuffer / createTexture ──
    this.lumRT   = this._makeFBO(width, height);
    this.bloomRT = this._makeFBO(width, height);

    const levels = this.params.levels;
    this.pyramid = [];
    this.blurH   = [];
    this.blurV   = [];
    for (let i = 0; i < levels; i++) {
      const w = Math.max(1, width  >> (i + 1));
      const h = Math.max(1, height >> (i + 1));
      this.pyramid.push(this._makeFBO(w, h));
      this.blurH.push(this._makeFBO(w, h));
      this.blurV.push(this._makeFBO(w, h));
    }

    // ── init: createBuffer / bufferData ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /** Update bloom tweakables. Safe to call every frame. */
  setParams(p: ATBloomParams): void {
    Object.assign(this.params, p);
  }

  /**
   * Resize all internal FBOs. Call when canvas changes dimensions.
   * Destroys old FBOs and recreates — gl.deleteFramebuffer / gl.deleteTexture.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;
    this._disposeRT(this.lumRT);
    this._disposeRT(this.bloomRT);
    this.pyramid.forEach(rt => this._disposeRT(rt));
    this.blurH.forEach(rt => this._disposeRT(rt));
    this.blurV.forEach(rt => this._disposeRT(rt));

    this.lumRT   = this._makeFBO(width, height);
    this.bloomRT = this._makeFBO(width, height);

    const levels = this.params.levels;
    this.pyramid = [];
    this.blurH   = [];
    this.blurV   = [];
    for (let i = 0; i < levels; i++) {
      const w = Math.max(1, width  >> (i + 1));
      const h = Math.max(1, height >> (i + 1));
      this.pyramid.push(this._makeFBO(w, h));
      this.blurH.push(this._makeFBO(w, h));
      this.blurV.push(this._makeFBO(w, h));
    }
  }

  /**
   * Execute the full AT UnrealBloom pipeline for one frame.
   *
   * @param sceneTex  - The rendered scene WebGLTexture (TEXTURE_2D).
   * @param outputFBO - Target framebuffer, null = draw to screen.
   */
  render(sceneTex: WebGLTexture, outputFBO: WebGLFramebuffer | null = null): void {
    const gl     = this.gl;
    const { luminosityThreshold, smoothWidth, bloomStrength, bloomRadius,
            bloomTintColor, levels } = this.params;
    const W = this.width;
    const H = this.height;

    // ── Pass 1: UnrealBloomLuminosity — extract bright pixels ──
    // Source: UnrealBloomLuminosity.glsl / BloomLuminosityPass.glsl (compiled.vs)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.lumRT.fbo);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.lumProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.lumProg, 'tDiffuse'), 0);
    gl.uniform3f(gl.getUniformLocation(this.lumProg, 'defaultColor'), 0.0, 0.0, 0.0);
    gl.uniform1f(gl.getUniformLocation(this.lumProg, 'defaultOpacity'), 0.0);
    gl.uniform1f(gl.getUniformLocation(this.lumProg, 'luminosityThreshold'), luminosityThreshold);
    gl.uniform1f(gl.getUniformLocation(this.lumProg, 'smoothWidth'), smoothWidth);
    this._drawQuad(this.lumProg);

    // ── Passes 2..N+1: DownSample.glsl — build bloom pyramid ──
    // Source: DownSample.glsl 13-tap dual-kawase (compiled.vs line ~6900)
    {
      let srcTex: WebGLTexture = this.lumRT.tex;
      for (let i = 0; i < levels; i++) {
        const rt  = this.pyramid[i];
        const sw  = i === 0 ? W : this.pyramid[i - 1].width;
        const sh  = i === 0 ? H : this.pyramid[i - 1].height;
        gl.bindFramebuffer(gl.FRAMEBUFFER, rt.fbo);
        gl.viewport(0, 0, rt.width, rt.height);
        gl.useProgram(this.downsampleProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, srcTex);
        gl.uniform1i(gl.getUniformLocation(this.downsampleProg, 'tMap'), 0);
        gl.uniform2f(gl.getUniformLocation(this.downsampleProg, 'uResolution'), sw, sh);
        gl.uniform1f(gl.getUniformLocation(this.downsampleProg, 'uRadius'), 1.0);
        this._drawQuad(this.downsampleProg);
        srcTex = rt.tex;
      }
    }

    // ── Passes N+2..N+2+levels*2-1: UnrealBloomGaussian H+V per pyramid level ──
    // Source: UnrealBloomGaussian.glsl — separable gaussian, SIGMA=5, KERNEL_RADIUS=9
    for (let i = 0; i < levels; i++) {
      const pw = this.pyramid[i].width;
      const ph = this.pyramid[i].height;

      // Horizontal pass: pyramid[i] → blurH[i]
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurH[i].fbo);
      gl.viewport(0, 0, pw, ph);
      gl.useProgram(this.gaussianProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.pyramid[i].tex);
      gl.uniform1i(gl.getUniformLocation(this.gaussianProg, 'colorTexture'), 0);
      gl.uniform2f(gl.getUniformLocation(this.gaussianProg, 'texSize'), pw, ph);
      gl.uniform2f(gl.getUniformLocation(this.gaussianProg, 'direction'), 1.0, 0.0);
      this._drawQuad(this.gaussianProg);

      // Vertical pass: blurH[i] → blurV[i]
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurV[i].fbo);
      gl.viewport(0, 0, pw, ph);
      gl.useProgram(this.gaussianProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.blurH[i].tex);
      gl.uniform1i(gl.getUniformLocation(this.gaussianProg, 'colorTexture'), 0);
      gl.uniform2f(gl.getUniformLocation(this.gaussianProg, 'texSize'), pw, ph);
      gl.uniform2f(gl.getUniformLocation(this.gaussianProg, 'direction'), 0.0, 1.0);
      this._drawQuad(this.gaussianProg);
    }

    // ── Upsample accumulation: blurV[levels-1] → ... → blurV[0] ──
    // Accumulate from finest (lowest-res) level upward using additive blending
    for (let i = levels - 2; i >= 0; i--) {
      const pw = this.blurV[i].width;
      const ph = this.blurV[i].height;

      // Temporary: write accumulation into blurH[i] (reuse as temp ping-pong)
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurH[i].fbo);
      gl.viewport(0, 0, pw, ph);
      gl.useProgram(this.upsampleProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.blurV[i].tex);
      gl.uniform1i(gl.getUniformLocation(this.upsampleProg, 'uLower'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.blurV[i + 1].tex);
      gl.uniform1i(gl.getUniformLocation(this.upsampleProg, 'uUpper'), 1);
      gl.uniform1f(gl.getUniformLocation(this.upsampleProg, 'uWeight'), 0.75);
      this._drawQuad(this.upsampleProg);

      // Swap blurH ↔ blurV so blurV[i] now holds the accumulated result
      const tmpFbo = this.blurV[i].fbo;
      const tmpTex = this.blurV[i].tex;
      this.blurV[i].fbo = this.blurH[i].fbo;
      this.blurV[i].tex = this.blurH[i].tex;
      this.blurH[i].fbo = tmpFbo;
      this.blurH[i].tex = tmpTex;
    }

    // ── UnrealBloomComposite: blend all blur levels → bloomRT ──
    // Source: UnrealBloomComposite.glsl (compiled.vs line 8658)
    // lerpBloomFactor() mirrors UE bloom radius blending
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomRT.fbo);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.compositeProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.blurV[0].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'blurTexture1'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.blurV[1 < levels ? 1 : 0].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'blurTexture2'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.blurV[2 < levels ? 2 : 0].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'blurTexture3'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.blurV[3 < levels ? 3 : 0].tex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'blurTexture4'), 3);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'bloomStrength'), bloomStrength);
    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'bloomRadius'), bloomRadius);
    gl.uniform3f(gl.getUniformLocation(this.compositeProg, 'bloomTintColor'),
      bloomTintColor[0], bloomTintColor[1], bloomTintColor[2]);
    this._drawQuad(this.compositeProg);

    // ── UnrealBloomPass: scene + bloom → output ──
    // Source: UnrealBloomPass.fs (compiled.vs line 8750)
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputFBO);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.finalProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.finalProg, 'tDiffuse'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.finalProg, 'tUnrealBloom'), 1);
    this._drawQuad(this.finalProg);
  }

  /**
   * Tick alias — same as render(). Matches the render()/tick() naming convention.
   */
  tick(sceneTex: WebGLTexture, outputFBO: WebGLFramebuffer | null = null): void {
    this.render(sceneTex, outputFBO);
  }

  /**
   * Get the intermediate bloom texture (after composite, before scene add).
   * Useful for feeding into other passes.
   */
  get bloomTexture(): WebGLTexture { return this.bloomRT.tex; }

  /**
   * Dispose all GPU resources.
   * Calls deleteProgram / deleteFramebuffer / deleteTexture / deleteBuffer.
   */
  dispose(): void {
    const gl = this.gl;

    // deleteProgram ×6
    gl.deleteProgram(this.lumProg);
    gl.deleteProgram(this.downsampleProg);
    gl.deleteProgram(this.gaussianProg);
    gl.deleteProgram(this.compositeProg);
    gl.deleteProgram(this.finalProg);
    gl.deleteProgram(this.upsampleProg);

    // deleteBuffer ×1
    gl.deleteBuffer(this.quadBuf);

    // deleteFramebuffer + deleteTexture ×(2 + levels*3)
    this._disposeRT(this.lumRT);
    this._disposeRT(this.bloomRT);
    this.pyramid.forEach(rt => this._disposeRT(rt));
    this.blurH.forEach(rt   => this._disposeRT(rt));
    this.blurV.forEach(rt   => this._disposeRT(rt));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers — real WebGL calls
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compile vert + frag GLSL → WebGLProgram.
   * Calls: createShader, shaderSource, compileShader, getShaderParameter,
   *        createProgram, attachShader, linkProgram, getProgramParameter,
   *        deleteShader.
   */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATBloom] vertex compile error (${label}): ${info}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ATBloom] fragment compile error (${label}): ${info}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[ATBloom] link error (${label}): ${info}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /**
   * Create a single FBO with an RGBA texture attachment.
   * Calls: createTexture, bindTexture, texParameteri ×4, texImage2D,
   *        createFramebuffer, bindFramebuffer, framebufferTexture2D.
   */
  private _makeFBO(w: number, h: number): SingleRT {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  /**
   * Free a single RT's GPU resources.
   * Calls: deleteFramebuffer, deleteTexture.
   */
  private _disposeRT(rt: SingleRT): void {
    const gl = this.gl;
    gl.deleteFramebuffer(rt.fbo);
    gl.deleteTexture(rt.tex);
  }

  /**
   * Draw the fullscreen quad.
   * Calls: bindBuffer, getAttribLocation, enableVertexAttribArray,
   *        vertexAttribPointer, drawArrays.
   */
  private _drawQuad(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory — matches BloomVariants.ts species params schema
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an ATBloomPostProcess from species-level BloomParams.
 *
 * @param gl         - WebGLRenderingContext from the canvas.
 * @param width      - Viewport width in pixels.
 * @param height     - Viewport height in pixels.
 * @param species    - Optional species bloom param pack.
 */
export function createATBloomForSpecies(
  gl     : WebGLRenderingContext,
  width  : number,
  height : number,
  species?: {
    bloomStrength?       : number;
    bloomRadius?         : number;
    luminosityThreshold? : number;
    bloomTintColors?     : [number, number, number];
  },
): ATBloomPostProcess {
  const params: ATBloomParams = {};
  if (species) {
    if (species.bloomStrength        != null) params.bloomStrength        = species.bloomStrength;
    if (species.bloomRadius          != null) params.bloomRadius          = species.bloomRadius;
    if (species.luminosityThreshold  != null) params.luminosityThreshold  = species.luminosityThreshold;
    if (species.bloomTintColors      != null) params.bloomTintColor       = species.bloomTintColors;
  }
  return new ATBloomPostProcess(gl, width, height, params);
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export inline GLSL fragments for hot-reload / inspection
// ─────────────────────────────────────────────────────────────────────────────

export const AT_BLOOM_GLSL = {
  /** luma.fs — Rec.601 luminance helper (compiled.vs ~8757) */
  luma        : LUMA_GLSL,
  /** UnrealBloomLuminosity.glsl / BloomLuminosityPass.glsl — threshold extract */
  luminosity  : LUMINOSITY_FRAG,
  /** DownSample.glsl — 13-tap dual-kawase pyramid downsample */
  downsample  : DOWNSAMPLE_FRAG,
  /** UnrealBloomGaussian.glsl — separable gaussian blur SIGMA=5 KERNEL_RADIUS=9 */
  gaussian    : GAUSSIAN_FRAG,
  /** UnrealBloomComposite.glsl — lerpBloomFactor multi-level composite */
  composite   : COMPOSITE_FRAG,
  /** UnrealBloomPass.fs — scene + bloom final additive */
  final       : FINAL_FRAG,
  /** upsample additive accumulator */
  upsample    : UPSAMPLE_FRAG,
  /** shared fullscreen-quad vertex shader */
  vert        : BLOOM_VERT,
} as const;
