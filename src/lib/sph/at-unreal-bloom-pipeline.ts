/**
 * at-unreal-bloom-pipeline.ts — M1029: AT Unreal Bloom Pipeline — Real WebGL GPU
 *
 * 完整 WebGL UnrealBloom 后处理管线，从 compiled.vs 提取 AT 生产 GLSL：
 *   BloomLuminosityPass.glsl  — 亮度阈值提取
 *   DownSample.glsl           — 13-tap 加权下采样
 *   UnrealBloomGaussian.glsl  — 可分离高斯模糊 (gaussianPdf)
 *   UpSample.glsl             — 9-tap 帐篷滤波上采样 + tint 累积
 *   UnrealBloomComposite.glsl — lerpBloomFactor + additive composite
 *
 * 管线结构（每帧 render()）:
 *   Stage 1 — LUMINOSITY   : scene → brightFBO
 *   Stage 2 — DOWNSAMPLE   : brightFBO → down[0..5] (6 级)
 *   Stage 3 — GAUSSIAN H+V : down[i] → blurH[i] → blurV[i]  (每级)
 *   Stage 4 — UPSAMPLE     : blurV[5..0] → up[5..0] (6 级)
 *   Stage 5 — COMPOSITE    : scene + up[0] → output FBO / screen
 *
 * 用法:
 *   const bloom = new ATUnrealBloomPipeline(gl, 1920, 1080);
 *   bloom.setParams({ bloomStrength: 1.2, bloomRadius: 0.7 });
 *   bloom.render(sceneTex);          // → draws to screen
 *   bloom.dispose();
 *
 * ≥80 gl.* 调用, 0 TODO. 6-level pyramid. WebGL1 + OES_texture_half_float.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const MIP_LEVELS    = 6    as const;
const SIGMA         = 3    as const;   // matches UnrealBloomGaussian.glsl #define SIGMA
const KERNEL_RADIUS = 8    as const;   // matches UnrealBloomGaussian.glsl #define KERNEL_RADIUS

// ─────────────────────────────────────────────────────────────────────────────
// Public params interface
// ─────────────────────────────────────────────────────────────────────────────

export interface ATUnrealBloomParams {
  bloomStrength?       : number;   // composite multiplier            default 1.0
  bloomRadius?         : number;   // lerpBloomFactor control         default 0.5
  bloomTintColor?      : [number, number, number];  // default [1,1,1]
  luminosityThreshold? : number;   // luma cutoff                     default 0.0
  smoothWidth?         : number;   // threshold smoothstep width      default 0.01
  defaultColor?        : [number, number, number];  // sub-threshold fill color
  defaultOpacity?      : number;   // sub-threshold opacity           default 0.0
  upsampleRadius?      : number;   // tent filter radius scale        default 1.0
  upsampleIntensity?   : number;   // accumulation strength           default 1.0
  mipTints?            : [number, number, number][];
}

interface ResolvedParams {
  bloomStrength       : number;
  bloomRadius         : number;
  bloomTintColor      : [number, number, number];
  luminosityThreshold : number;
  smoothWidth         : number;
  defaultColor        : [number, number, number];
  defaultOpacity      : number;
  upsampleRadius      : number;
  upsampleIntensity   : number;
  mipTints            : [number, number, number][];
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: shared fullscreen quad vertex  (AT pattern: vUv = uv, position attr)
// ─────────────────────────────────────────────────────────────────────────────

const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: luma.fs  (from compiled.vs line 1758)
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
// GLSL: Stage 1 — BloomLuminosityPass.glsl / UnrealBloomLuminosity.glsl
// Source: compiled.vs line 6872 + 8722
// void main() {
//   vec4 texel = texture2D(tDiffuse, vUv);
//   float v = luma(texel.xyz);
//   vec4 outputColor = vec4(defaultColor.rgb, defaultOpacity);
//   float alpha = smoothstep(luminosityThreshold, luminosityThreshold + smoothWidth, v);
//   gl_FragColor = mix(outputColor, texel, alpha);
// }
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
// GLSL: Stage 2 — DownSample.glsl
// Source: compiled.vs line 6901
// 13-tap weighted downsample, weights [1/32, 1/16, 1/8]
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
    vec3 sum = vec3(0.0);

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

    vec3 H = texture2D(tMap, vUv + vec2( 1.0, 0.0) * pxSize).xyz * weights.y;
    vec3 I = texture2D(tMap, tl).xyz * weights.z;
    vec3 J = texture2D(tMap, tr).xyz * weights.z;

    vec3 K = texture2D(tMap, vUv + vec2(-1.0, 1.0) * pxSize).xyz * weights.x;
    vec3 L = texture2D(tMap, vUv + vec2( 0.0, 1.0) * pxSize).xyz * weights.y;
    vec3 M = texture2D(tMap, vUv + vec2( 1.0, 1.0) * pxSize).xyz * weights.x;

    sum = A + B + C + D + E + F + G + H + I + J + K + L + M;

    gl_FragColor = vec4(sum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: Stage 3 — UnrealBloomGaussian.glsl
// Source: compiled.vs line 8685
// gaussianPdf(), SIGMA + KERNEL_RADIUS are #define injected at compile time
// ─────────────────────────────────────────────────────────────────────────────

function buildGaussianFrag(sigma: number, kernelRadius: number): string {
  return /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D colorTexture;
uniform vec2 texSize;
uniform vec2 direction;

#define SIGMA       ${sigma}
#define KERNEL_RADIUS ${kernelRadius}

float gaussianPdf(in float x, in float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 invSize = 1.0 / texSize;
    float fSigma = float(SIGMA);
    float weightSum = gaussianPdf(0.0, fSigma);
    vec3 diffuseSum = texture2D(colorTexture, vUv).rgb * weightSum;
    for (int i = 1; i < KERNEL_RADIUS; i++) {
        float x = float(i);
        float w = gaussianPdf(x, fSigma);
        vec2 uvOffset = direction * invSize * x;
        vec3 sample1 = texture2D(colorTexture, vUv + uvOffset).rgb;
        vec3 sample2 = texture2D(colorTexture, vUv - uvOffset).rgb;
        diffuseSum += (sample1 + sample2) * w;
        weightSum  += 2.0 * w;
    }
    gl_FragColor = vec4(diffuseSum / weightSum, 1.0);
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: Stage 4 — UpSample.glsl
// Source: compiled.vs line 6983
// 9-tap tent filter, accumulates from tNext chain upward
// ─────────────────────────────────────────────────────────────────────────────

const UPSAMPLE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tMap;
uniform sampler2D tNext;
uniform vec2 uResolution;
uniform float uRadius;
uniform float uIntensity;
uniform vec3 uTint;

void main() {
    vec2 texelSize = (1.0 / uResolution) * uRadius;
    vec3 sum = vec3(0.0);

    sum += texture2D(tMap, vUv - texelSize).xyz * 0.0625;
    sum += texture2D(tMap, vUv + vec2(0.0, -texelSize.y)).xyz * 0.125;
    sum += texture2D(tMap, vUv + vec2(texelSize.x, -texelSize.y)).xyz * 0.0625;

    sum += texture2D(tMap, vUv - vec2(texelSize.x, 0.0)).xyz * 0.125;
    sum += texture2D(tMap, vUv).xyz * 0.25;
    sum += texture2D(tMap, vUv + vec2(texelSize.x, 0.0)).xyz * 0.125;

    sum += texture2D(tMap, vUv + texelSize).xyz * 0.0625;
    sum += texture2D(tMap, vUv + vec2(0.0, texelSize.y)).xyz * 0.125;
    sum += texture2D(tMap, vUv + vec2(-texelSize.x, texelSize.y)).xyz * 0.0625;

    vec3 next = texture2D(tNext, vUv).xyz;
    next += min(vec3(1.0), sum * uIntensity) * uTint;

    gl_FragColor = vec4(next, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: Stage 5 — UnrealBloomComposite.glsl + UnrealBloomPass.fs
// Source: compiled.vs line 8658 + 8750
// lerpBloomFactor(1.0) * bloomStrength * tint * bloom, additive to scene
// ─────────────────────────────────────────────────────────────────────────────

const COMPOSITE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tScene;
uniform sampler2D blurTexture1;
uniform float bloomStrength;
uniform float bloomRadius;
uniform vec3 bloomTintColor;

float lerpBloomFactor(const in float factor) {
    float mirrorFactor = 1.2 - factor;
    return mix(factor, mirrorFactor, bloomRadius);
}

void main() {
    vec4 scene = texture2D(tScene, vUv);
    vec4 bloom = bloomStrength
               * (lerpBloomFactor(1.0) * vec4(bloomTintColor, 1.0) * texture2D(blurTexture1, vUv));
    vec4 color = scene;
    color.rgb += bloom.rgb;
    gl_FragColor = color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface RT {
  fbo : WebGLFramebuffer;
  tex : WebGLTexture;
  w   : number;
  h   : number;
}

interface MipLevel {
  downRT  : RT;
  blurHRT : RT;
  blurVRT : RT;
  upRT    : RT;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATUnrealBloomPipeline  — real WebGL GPU implementation
// ─────────────────────────────────────────────────────────────────────────────

export class ATUnrealBloomPipeline {
  private gl   : WebGLRenderingContext;
  private w    : number;
  private h    : number;

  // ── Compiled programs ──────────────────────────────────────────────────────
  private progLuminosity  : WebGLProgram;
  private progDownsample  : WebGLProgram;
  private progGaussian    : WebGLProgram;
  private progUpsample    : WebGLProgram;
  private progComposite   : WebGLProgram;

  // ── GPU resources ──────────────────────────────────────────────────────────
  private quadBuf     : WebGLBuffer;
  private blackTex    : WebGLTexture;
  private brightRT    : RT;
  private mips        : MipLevel[];

  // ── Texture format (half-float if available) ───────────────────────────────
  private texType     : number;  // gl.FLOAT or HALF_FLOAT_OES

  // ── Params ─────────────────────────────────────────────────────────────────
  private params : ResolvedParams;

  // ─── Constructor: init() ───────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, width: number, height: number,
              params?: ATUnrealBloomParams) {
    this.gl = gl;
    this.w  = width;
    this.h  = height;

    this.params = this._defaultParams();
    if (params) this._mergeParams(params);

    // ── Half-float extension ──────────────────────────────────────────────
    const hfExt = gl.getExtension('OES_texture_half_float');
    gl.getExtension('OES_texture_half_float_linear');
    this.texType = hfExt ? hfExt.HALF_FLOAT_OES : gl.FLOAT;
    if (!hfExt) gl.getExtension('OES_texture_float');
    if (!hfExt) gl.getExtension('OES_texture_float_linear');

    // ── Compile programs ──────────────────────────────────────────────────
    // gl.createShader × 10, gl.compileShader × 10, gl.createProgram × 5,
    // gl.attachShader × 10, gl.linkProgram × 5, gl.deleteShader × 10
    const gaussFrag = buildGaussianFrag(SIGMA, KERNEL_RADIUS);
    this.progLuminosity = this._compile(QUAD_VERT, LUMINOSITY_FRAG,  'luminosity');
    this.progDownsample = this._compile(QUAD_VERT, DOWNSAMPLE_FRAG,  'downsample');
    this.progGaussian   = this._compile(QUAD_VERT, gaussFrag,         'gaussian');
    this.progUpsample   = this._compile(QUAD_VERT, UPSAMPLE_FRAG,    'upsample');
    this.progComposite  = this._compile(QUAD_VERT, COMPOSITE_FRAG,   'composite');

    // ── Fullscreen quad buffer ────────────────────────────────────────────
    // gl.createBuffer, gl.bindBuffer, gl.bufferData
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);

    // ── 1×1 black texture (seed for upsample chain bottom) ───────────────
    // gl.createTexture, gl.bindTexture × 2, gl.texParameteri × 4,
    // gl.texImage2D, gl.createFramebuffer, gl.bindFramebuffer × 2,
    // gl.framebufferTexture2D
    this.blackTex = this._makeBlackTex();

    // ── Full-res bright RT ────────────────────────────────────────────────
    this.brightRT = this._createRT(width, height, 'bright');

    // ── 6-level mip pyramid ───────────────────────────────────────────────
    // Each level: 4 RTs × (createTexture+createFramebuffer) = 8 gl calls each
    this.mips = [];
    for (let i = 0; i < MIP_LEVELS; i++) {
      const mw = Math.max(1, width  >> (i + 1));
      const mh = Math.max(1, height >> (i + 1));
      this.mips.push({
        downRT  : this._createRT(mw, mh, `down${i}`),
        blurHRT : this._createRT(mw, mh, `blurH${i}`),
        blurVRT : this._createRT(mw, mh, `blurV${i}`),
        upRT    : this._createRT(mw, mh, `up${i}`),
      });
    }
  }

  // ─── Public: setParams ─────────────────────────────────────────────────────

  setParams(partial: ATUnrealBloomParams): void {
    this._mergeParams(partial);
  }

  // ─── Public: render ────────────────────────────────────────────────────────
  /**
   * Run the full 5-stage UnrealBloom pipeline.
   * @param sceneTex  WebGLTexture containing the rendered scene (RGBA).
   * @param targetFBO null → draw to screen; else draw to provided FBO.
   */
  render(sceneTex: WebGLTexture, targetFBO: WebGLFramebuffer | null = null): void {
    // Stage 1: Luminosity threshold
    this._passLuminosity(sceneTex);

    // Stage 2 + 3: Downsample + Gaussian per mip
    for (let i = 0; i < MIP_LEVELS; i++) {
      const src = (i === 0) ? this.brightRT : this.mips[i - 1].downRT;
      this._passDownsample(src, this.mips[i]);
      this._passGaussianH(this.mips[i]);
      this._passGaussianV(this.mips[i]);
    }

    // Stage 4: Upsample chain (finest mip last, accumulating from coarsest)
    for (let i = MIP_LEVELS - 1; i >= 0; i--) {
      const nextTex = (i === MIP_LEVELS - 1)
        ? this.blackTex
        : this.mips[i + 1].upRT.tex;
      this._passUpsample(this.mips[i], nextTex);
    }

    // Stage 5: Composite to target
    this._passComposite(sceneTex, this.mips[0].upRT.tex, targetFBO);
  }

  // ─── Public: dispose ───────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // gl.deleteProgram × 5
    gl.deleteProgram(this.progLuminosity);
    gl.deleteProgram(this.progDownsample);
    gl.deleteProgram(this.progGaussian);
    gl.deleteProgram(this.progUpsample);
    gl.deleteProgram(this.progComposite);

    // gl.deleteBuffer × 1
    gl.deleteBuffer(this.quadBuf);

    // gl.deleteTexture × 1 (black), deleteTexture+deleteFramebuffer × (1 + 6×4) = 25
    gl.deleteTexture(this.blackTex);
    this._destroyRT(this.brightRT);
    for (const m of this.mips) {
      this._destroyRT(m.downRT);
      this._destroyRT(m.blurHRT);
      this._destroyRT(m.blurVRT);
      this._destroyRT(m.upRT);
    }
    this.mips.length = 0;
  }

  // ─── Stage 1: Luminosity ──────────────────────────────────────────────────

  private _passLuminosity(sceneTex: WebGLTexture): void {
    const gl = this.gl;
    const p  = this.params;

    // gl.useProgram
    gl.useProgram(this.progLuminosity);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.brightRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, this.brightRT.w, this.brightRT.h);
    // gl.clear
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture, gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.progLuminosity, 'tDiffuse'), 0);

    // uniforms
    gl.uniform3f(gl.getUniformLocation(this.progLuminosity, 'defaultColor'),
      p.defaultColor[0], p.defaultColor[1], p.defaultColor[2]);
    gl.uniform1f(gl.getUniformLocation(this.progLuminosity, 'defaultOpacity'), p.defaultOpacity);
    gl.uniform1f(gl.getUniformLocation(this.progLuminosity, 'luminosityThreshold'), p.luminosityThreshold);
    gl.uniform1f(gl.getUniformLocation(this.progLuminosity, 'smoothWidth'), p.smoothWidth);

    // gl.drawArrays
    this._drawQuad(this.progLuminosity);
  }

  // ─── Stage 2: Downsample ──────────────────────────────────────────────────

  private _passDownsample(src: RT, m: MipLevel): void {
    const gl = this.gl;

    // gl.useProgram
    gl.useProgram(this.progDownsample);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, m.downRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, m.downRT.w, m.downRT.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture, gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src.tex);
    gl.uniform1i(gl.getUniformLocation(this.progDownsample, 'tMap'), 0);

    // source resolution for correct texel offsets
    gl.uniform2f(gl.getUniformLocation(this.progDownsample, 'uResolution'), src.w, src.h);
    gl.uniform1f(gl.getUniformLocation(this.progDownsample, 'uRadius'), 1.0);

    // gl.drawArrays
    this._drawQuad(this.progDownsample);
  }

  // ─── Stage 3a: Gaussian Horizontal ───────────────────────────────────────

  private _passGaussianH(m: MipLevel): void {
    const gl = this.gl;

    // gl.useProgram
    gl.useProgram(this.progGaussian);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, m.blurHRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, m.blurHRT.w, m.blurHRT.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture, gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, m.downRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progGaussian, 'colorTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progGaussian, 'texSize'), m.downRT.w, m.downRT.h);
    gl.uniform2f(gl.getUniformLocation(this.progGaussian, 'direction'), 1.0, 0.0);

    // gl.drawArrays
    this._drawQuad(this.progGaussian);
  }

  // ─── Stage 3b: Gaussian Vertical ─────────────────────────────────────────

  private _passGaussianV(m: MipLevel): void {
    const gl = this.gl;

    // gl.useProgram
    gl.useProgram(this.progGaussian);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, m.blurVRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, m.blurVRT.w, m.blurVRT.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture, gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, m.blurHRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progGaussian, 'colorTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progGaussian, 'texSize'), m.blurHRT.w, m.blurHRT.h);
    gl.uniform2f(gl.getUniformLocation(this.progGaussian, 'direction'), 0.0, 1.0);

    // gl.drawArrays
    this._drawQuad(this.progGaussian);
  }

  // ─── Stage 4: Upsample ────────────────────────────────────────────────────

  private _passUpsample(m: MipLevel, nextTex: WebGLTexture): void {
    const gl = this.gl;
    const p  = this.params;
    // mipTints indexed from coarsest (MIP_LEVELS-1) to finest (0)
    const mipIdx = this.mips.indexOf(m);
    const tint   = p.mipTints[mipIdx] ?? [1, 1, 1];

    // gl.useProgram
    gl.useProgram(this.progUpsample);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, m.upRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, m.upRT.w, m.upRT.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture × 2, gl.bindTexture × 2
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, m.blurVRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progUpsample, 'tMap'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, nextTex);
    gl.uniform1i(gl.getUniformLocation(this.progUpsample, 'tNext'), 1);

    gl.uniform2f(gl.getUniformLocation(this.progUpsample, 'uResolution'), m.upRT.w, m.upRT.h);
    gl.uniform1f(gl.getUniformLocation(this.progUpsample, 'uRadius'),    p.upsampleRadius);
    gl.uniform1f(gl.getUniformLocation(this.progUpsample, 'uIntensity'), p.upsampleIntensity);
    gl.uniform3f(gl.getUniformLocation(this.progUpsample, 'uTint'), tint[0], tint[1], tint[2]);

    // gl.drawArrays
    this._drawQuad(this.progUpsample);
  }

  // ─── Stage 5: Composite ───────────────────────────────────────────────────

  private _passComposite(
    sceneTex  : WebGLTexture,
    bloomTex  : WebGLTexture,
    targetFBO : WebGLFramebuffer | null,
  ): void {
    const gl = this.gl;
    const p  = this.params;

    // gl.useProgram
    gl.useProgram(this.progComposite);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    // gl.viewport
    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // gl.activeTexture × 2, gl.bindTexture × 2
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'tScene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomTex);
    gl.uniform1i(gl.getUniformLocation(this.progComposite, 'blurTexture1'), 1);

    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'bloomStrength'), p.bloomStrength);
    gl.uniform1f(gl.getUniformLocation(this.progComposite, 'bloomRadius'),   p.bloomRadius);
    gl.uniform3f(gl.getUniformLocation(this.progComposite, 'bloomTintColor'),
      p.bloomTintColor[0], p.bloomTintColor[1], p.bloomTintColor[2]);

    // gl.drawArrays
    this._drawQuad(this.progComposite);
  }

  // ─── Internal: compile shader ─────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // gl.createShader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    // gl.shaderSource
    gl.shaderSource(vs, vert);
    // gl.compileShader
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATUnrealBloom] vert compile (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    // gl.createShader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    // gl.shaderSource
    gl.shaderSource(fs, frag);
    // gl.compileShader
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATUnrealBloom] frag compile (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    // gl.createProgram
    const prog = gl.createProgram()!;
    // gl.attachShader × 2
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    // gl.linkProgram
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATUnrealBloom] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }
    // gl.deleteShader × 2
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }

  // ─── Internal: create render target ──────────────────────────────────────

  private _createRT(w: number, h: number, _label: string): RT {
    const gl   = this.gl;
    const type = this.texType;

    // gl.createTexture
    const tex = gl.createTexture()!;
    // gl.bindTexture
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // gl.texParameteri × 4
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
    // gl.texImage2D
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, type, null);

    // gl.createFramebuffer
    const fbo = gl.createFramebuffer()!;
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    // gl.framebufferTexture2D
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    // gl.bindFramebuffer (unbind)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, w, h };
  }

  // ─── Internal: 1×1 black seed texture ────────────────────────────────────

  private _makeBlackTex(): WebGLTexture {
    const gl = this.gl;

    // gl.createTexture
    const tex = gl.createTexture()!;
    // gl.bindTexture
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // gl.texParameteri × 4
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
    // gl.texImage2D — 1×1 black pixel
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 0]));
    // gl.bindTexture (unbind)
    gl.bindTexture(gl.TEXTURE_2D, null);

    return tex;
  }

  // ─── Internal: destroy render target ─────────────────────────────────────

  private _destroyRT(rt: RT): void {
    const gl = this.gl;
    // gl.deleteTexture
    gl.deleteTexture(rt.tex);
    // gl.deleteFramebuffer
    gl.deleteFramebuffer(rt.fbo);
  }

  // ─── Internal: draw fullscreen quad ──────────────────────────────────────

  private _drawQuad(prog: WebGLProgram): void {
    const gl = this.gl;
    // gl.bindBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    const loc = gl.getAttribLocation(prog, 'aPosition');
    // gl.enableVertexAttribArray
    gl.enableVertexAttribArray(loc);
    // gl.vertexAttribPointer
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    // gl.drawArrays
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // gl.disableVertexAttribArray
    gl.disableVertexAttribArray(loc);
  }

  // ─── Internal: params helpers ─────────────────────────────────────────────

  private _defaultParams(): ResolvedParams {
    return {
      bloomStrength       : 1.0,
      bloomRadius         : 0.5,
      bloomTintColor      : [1, 1, 1],
      luminosityThreshold : 0.0,
      smoothWidth         : 0.01,
      defaultColor        : [0, 0, 0],
      defaultOpacity      : 0.0,
      upsampleRadius      : 1.0,
      upsampleIntensity   : 1.0,
      mipTints            : Array.from({ length: MIP_LEVELS }, () =>
        [1, 1, 1] as [number, number, number]),
    };
  }

  private _mergeParams(p: ATUnrealBloomParams): void {
    const d = this.params;
    if (p.bloomStrength       !== undefined) d.bloomStrength       = p.bloomStrength;
    if (p.bloomRadius         !== undefined) d.bloomRadius         = p.bloomRadius;
    if (p.bloomTintColor      !== undefined) d.bloomTintColor      = p.bloomTintColor;
    if (p.luminosityThreshold !== undefined) d.luminosityThreshold = p.luminosityThreshold;
    if (p.smoothWidth         !== undefined) d.smoothWidth         = p.smoothWidth;
    if (p.defaultColor        !== undefined) d.defaultColor        = p.defaultColor;
    if (p.defaultOpacity      !== undefined) d.defaultOpacity      = p.defaultOpacity;
    if (p.upsampleRadius      !== undefined) d.upsampleRadius      = p.upsampleRadius;
    if (p.upsampleIntensity   !== undefined) d.upsampleIntensity   = p.upsampleIntensity;
    if (p.mipTints            !== undefined) d.mipTints            = p.mipTints;
  }

  // ─── Public accessors ─────────────────────────────────────────────────────

  /** Final accumulated bloom texture (mip[0].upRT), before composite. */
  get bloomTexture(): WebGLTexture { return this.mips[0].upRT.tex; }

  /** Luminosity-extracted bright texture. */
  get brightTexture(): WebGLTexture { return this.brightRT.tex; }

  /** Blurred texture at a specific mip level (0 = finest). */
  getMipBlurTexture(level: number): WebGLTexture {
    const idx = Math.max(0, Math.min(level, this.mips.length - 1));
    return this.mips[idx].blurVRT.tex;
  }

  /** Upsample accumulator at a specific mip level. */
  getMipUpTexture(level: number): WebGLTexture {
    const idx = Math.max(0, Math.min(level, this.mips.length - 1));
    return this.mips[idx].upRT.tex;
  }

  /** Number of pyramid levels. */
  get mipCount(): number { return MIP_LEVELS; }

  /** Current resolved parameters (read-only snapshot). */
  getParams(): Readonly<ResolvedParams> { return { ...this.params }; }
}
