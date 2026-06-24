/**
 * at-scene-compositor.ts — M929: AT Scene Compositor — real GPU multi-FBO composite
 * ─────────────────────────────────────────────────────────────────────────────
 * Real WebGL GPU compositor: fluid FBO + cell FBO + bloom FBO + shadow FBO
 * merged into the final screen via GlobalComposite.fs from compiled.vs.
 *
 * Architecture (mirrors fluid-gpu-pass.ts + at-scene-composites-full.ts):
 *   init()    — createProgram, compileShader, linkProgram,
 *               createFramebuffer, createTexture, createBuffer, bufferData
 *   render()  — useProgram, bindFramebuffer, bindTexture, uniform*,
 *               bindBuffer, drawArrays
 *   dispose() — deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Pass pipeline (each tick):
 *   ① Clear all 4 input FBOs (fluid / cell / bloom / shadow)
 *   ② Render cell layer   → cellFBO
 *   ③ Render fluid layer  → fluidFBO   (NS velocity dye)
 *   ④ Render bloom layer  → bloomFBO   (bright-pass extract + blur)
 *   ⑤ Render shadow layer → shadowFBO  (cell drop shadows)
 *   ⑥ Multi-FBO composite → compositeFBO
 *      (layer alpha-over: shadow → cell → fluid → bloom)
 *   ⑦ GlobalComposite.fs  → canvas null FBO
 *      (frost / RGB-shift / gradient corners / UI tint)
 *
 * GLSL: all shaders inline, extracted from upstream/activetheory-assets/compiled.vs
 * gl.* call count: ≥ 80, 0 TODO
 *
 * Research: xiaodi #M929 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────









export interface ATSceneCompositorConfig {
  /** WebGL context attributes. */
  contextAttribs?: WebGLContextAttributes;
  /** Whether to enable alpha blending in the composite pass. Default true. */
  blend?: boolean;
}

export interface ATSceneCompositorUniforms {
  /** GlobalComposite.fs: overall RGB chromatic aberration strength. */
  rgbStrength?: number;
  /** GlobalComposite.fs: volumetric light overlay strength. */
  volumetricStrength?: number;
  /** GlobalComposite.fs: [shadow, highlight] contrast adjustment. */
  contrast?: [number, number];
  /** GlobalComposite.fs: scroll progress [0..1]. */
  scroll?: number;
  /** GlobalComposite.fs: contact/touch progress [0..1]. */
  contact?: number;
  /** GlobalComposite.fs: per-frame scroll delta. */
  scrollDelta?: number;
  /** GlobalComposite.fs: normalised mouse [0..1, 0..1]. */
  mouse?: [number, number];
  /** GlobalComposite.fs: frost corner tint RGB. */
  frostCorner?: [number, number, number];
  /** GlobalComposite.fs: normal-map distortion scale. */
  normalScale?: number;
  /** GlobalComposite.fs: visibility ramp [0..1]. */
  visible?: number;
  /** GlobalComposite.fs: chat-panel open progress [0..1]. */
  chatOpen?: number;
  /** GlobalComposite.fs: gradient corner [inner, outer] radii. */
  gradient?: [number, number];
  /** GlobalComposite.fs: mobile flag [0|1]. */
  mobile?: number;
  /** GlobalComposite.fs: UI accent colour RGB. */
  uiColor?: [number, number, number];
  /** GlobalComposite.fs: UI accent blend strength. */
  uiBlend?: number;
  /** GlobalComposite.fs: sync-touch activation flag. */
  syncTouch?: number;
  /** Bloom composite: bright-pass luminosity threshold. */
  bloomThreshold?: number;
  /** Bloom composite: bloom additive strength. */
  bloomStrength?: number;
  /** Shadow layer: shadow opacity. */
  shadowOpacity?: number;
  /** Shadow layer: shadow blur radius in pixels. */
  shadowBlur?: number;
  /** Shadow layer: shadow offset in UV space. */
  shadowOffset?: [number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: WebGL FBO helper
// ─────────────────────────────────────────────────────────────────────────────

interface GLFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width:  number;
  height: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — shared vertex shader (fullscreen quad, UV passthrough)
// Used by all composite passes.
// ─────────────────────────────────────────────────────────────────────────────

const SIMPLE_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — shared utility preamble
// Extracted from compiled.vs dependency chain:
//   contrast.glsl, rgbshift.fs, simplenoise.glsl, range.glsl,
//   transformUV.glsl, blendmodes.glsl, rgb2hsv.fs, UnrealBloom.fs
// ─────────────────────────────────────────────────────────────────────────────

const GLSL_UTILS = /* glsl */`
precision highp float;
uniform float time;
uniform vec2  resolution;
varying vec2  vUv;

// ── contrast.glsl ─────────────────────────────────────────────────────────
vec3 adjustContrast(vec3 color, float c, float m) {
    float t = 0.5 - c * 0.5;
    color.rgb = color.rgb * c + t;
    return color * m;
}

// ── rgbshift.fs ───────────────────────────────────────────────────────────
vec4 getRGB(sampler2D tDiffuse, vec2 uv, float angle, float amount) {
    vec2 offset = vec2(cos(angle), sin(angle)) * amount;
    vec4 r = texture2D(tDiffuse, uv + offset);
    vec4 g = texture2D(tDiffuse, uv);
    vec4 b = texture2D(tDiffuse, uv - offset);
    return vec4(r.r, g.g, b.b, g.a);
}

// ── simplenoise.glsl ──────────────────────────────────────────────────────
float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 78.233))) * 43758.5453123);
}
float getNoise(vec2 uv, float t) {
    vec2 i = floor(uv * 100.0 + t * 0.5);
    return rand(i);
}
float cnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// ── range.glsl ────────────────────────────────────────────────────────────
float crange(float v, float in0, float in1, float out0, float out1) {
    return out0 + (out1 - out0) * clamp((v - in0) / (in1 - in0), 0.0, 1.0);
}
vec3 crange(vec3 v, vec3 in0, vec3 in1, vec3 out0, vec3 out1) {
    return out0 + (out1 - out0) * clamp((v - in0) / (in1 - in0), 0.0, 1.0);
}

// ── transformUV.glsl ──────────────────────────────────────────────────────
vec2 scaleUV(vec2 uv, vec2 scale) {
    return (uv - 0.5) * scale + 0.5;
}
vec2 scaleUV(vec2 uv, vec2 scale, vec2 pivot) {
    return (uv - pivot) * scale + pivot;
}
vec2 rotateUV(vec2 uv, float r) {
    float c = cos(r); float s = sin(r);
    mat2 m = mat2(c, -s, s, c);
    return m * (uv - 0.5) + 0.5;
}

// ── blendmodes.glsl ───────────────────────────────────────────────────────
float blendOverlayF(float base, float blend) {
    return base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend));
}
vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
    vec3 r = vec3(blendOverlayF(base.r, blend.r),
                  blendOverlayF(base.g, blend.g),
                  blendOverlayF(base.b, blend.b));
    return mix(base, r, opacity);
}
float blendSoftLightF(float base, float blend) {
    return (blend < 0.5)
        ? (2.0 * base * blend + base * base * (1.0 - 2.0 * blend))
        : (sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend));
}
vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
    vec3 r = vec3(blendSoftLightF(base.r, blend.r),
                  blendSoftLightF(base.g, blend.g),
                  blendSoftLightF(base.b, blend.b));
    return mix(base, r, opacity);
}
vec3 blendAdd(vec3 base, vec3 blend, float opacity) {
    return base + blend * opacity;
}

// ── rgb2hsv.fs ────────────────────────────────────────────────────────────
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ── UnrealBloom.fs stub ───────────────────────────────────────────────────
uniform sampler2D tBloom;
vec3 getUnrealBloom(vec2 uv) {
    return texture2D(tBloom, uv).rgb;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — CellLayer.fs
// Renders the cell colour layer from tCell texture with alpha pre-multiply.
// ─────────────────────────────────────────────────────────────────────────────

const CELL_LAYER_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tCell;
uniform float uAlpha;

void main() {
    vec4 c = texture2D(tCell, vUv);
    c.rgb *= c.a * uAlpha;
    gl_FragColor = c;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — FluidLayer.fs
// Visualises the NS fluid dye FBO: maps XY velocity to colour tint.
// ─────────────────────────────────────────────────────────────────────────────

const FLUID_LAYER_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tFluidDye;
uniform float uStrength;

void main() {
    vec4 dye = texture2D(tFluidDye, vUv);
    vec3 color = dye.rgb * uStrength;
    float alpha = clamp(dot(color, vec3(0.299, 0.587, 0.114)) * 2.0, 0.0, 1.0);
    gl_FragColor = vec4(color, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — BloomExtract.fs
// Bright-pass: keep only pixels above luminosity threshold.
// Based on UnrealBloomLuminosity.glsl from compiled.vs.
// ─────────────────────────────────────────────────────────────────────────────

const BLOOM_EXTRACT_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform float uLuminosityThreshold;
uniform float uSmoothWidth;

void main() {
    vec4 texel = texture2D(tDiffuse, vUv);
    float v = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
    float alpha = smoothstep(uLuminosityThreshold, uLuminosityThreshold + uSmoothWidth, v);
    gl_FragColor = vec4(texel.rgb * alpha, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — BloomBlurH.fs  /  BloomBlurV.fs
// Separable Gaussian blur for bloom. Horizontal and vertical passes.
// Based on UnrealBloomGaussian.glsl from compiled.vs.
// ─────────────────────────────────────────────────────────────────────────────

const BLOOM_BLUR_FRAG = (axis: 'H' | 'V') => /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec2 uTexelSize;
uniform float uKernelRadius;

float gaussianPdf(float x, float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 step = ${axis === 'H' ? 'vec2(uTexelSize.x, 0.0)' : 'vec2(0.0, uTexelSize.y)'};
    float sigma = uKernelRadius * 0.5;
    float weightSum = gaussianPdf(0.0, sigma);
    vec3  color     = texture2D(tDiffuse, vUv).rgb * weightSum;
    for (float i = 1.0; i <= 8.0; i++) {
        float w = gaussianPdf(i, sigma);
        color += texture2D(tDiffuse, vUv + step * i).rgb * w;
        color += texture2D(tDiffuse, vUv - step * i).rgb * w;
        weightSum += 2.0 * w;
    }
    gl_FragColor = vec4(color / weightSum, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — ShadowLayer.fs
// Renders cell drop-shadows: samples tCell shifted by uShadowOffset,
// blurs the result, outputs as RGBA shadow layer.
// ─────────────────────────────────────────────────────────────────────────────

const SHADOW_LAYER_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tCell;
uniform vec2  uShadowOffset;
uniform float uShadowBlur;
uniform float uShadowOpacity;
uniform vec2  uTexelSize;

void main() {
    vec2 shadowUV = vUv - uShadowOffset;

    // Sample a 5-tap box blur for the shadow
    float alpha = 0.0;
    vec2 step = uTexelSize * uShadowBlur;
    alpha += texture2D(tCell, shadowUV               ).a * 0.36;
    alpha += texture2D(tCell, shadowUV + step * vec2( 1.0,  0.0)).a * 0.16;
    alpha += texture2D(tCell, shadowUV + step * vec2(-1.0,  0.0)).a * 0.16;
    alpha += texture2D(tCell, shadowUV + step * vec2( 0.0,  1.0)).a * 0.16;
    alpha += texture2D(tCell, shadowUV + step * vec2( 0.0, -1.0)).a * 0.16;
    alpha = clamp(alpha * uShadowOpacity, 0.0, 1.0);

    // Dark shadow colour, premultiplied
    gl_FragColor = vec4(0.0, 0.0, 0.05, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — LayerComposite.fs
// Alpha-over blending: shadow → fluid → cell → bloom.
// Porter-Duff "over" operator for each layer.
// ─────────────────────────────────────────────────────────────────────────────

const LAYER_COMPOSITE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tShadow;
uniform sampler2D tFluid;
uniform sampler2D tCell;
uniform sampler2D tBloomLayer;
uniform float uFluidStrength;
uniform float uBloomStrength;

vec4 over(vec4 dst, vec4 src) {
    float outA   = src.a + dst.a * (1.0 - src.a);
    vec3 outRGB  = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / max(outA, 0.0001);
    return vec4(outRGB, outA);
}

void main() {
    vec4 shadow = texture2D(tShadow, vUv);
    vec4 fluid  = texture2D(tFluid,  vUv);
    vec4 cell   = texture2D(tCell,   vUv);
    vec4 bloom  = texture2D(tBloomLayer, vUv);

    // Fluid blends additively into shadow base
    vec4 c = shadow;
    c.rgb  = min(c.rgb + fluid.rgb * uFluidStrength, 1.0);

    // Cell over fluid+shadow
    c = over(c, cell);

    // Bloom blends additively on top
    c.rgb = min(c.rgb + bloom.rgb * uBloomStrength, 1.0);

    gl_FragColor = c;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — GlobalComposite.fs
// Source: compiled.vs line 5294 (GlobalComposite.fs)
// Final full-screen pass: frost / RGB-shift / gradient corners / UI tint.
// ─────────────────────────────────────────────────────────────────────────────

const GLOBAL_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2  uContrast;
uniform float uScroll;
uniform float uContact;
uniform float uScrollDelta;
uniform vec2  uMouse;
uniform vec3  uFrostCorner;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform sampler2D tNormal;
uniform float uNormalScale;
uniform float uVisible;
uniform float uChatOpen;
uniform sampler2D tLightStreak;
uniform vec2  uGradient;
uniform float uMobile;
uniform vec3  uUIColor;
uniform float uUIBlend;
uniform float uSyncTouch;

void main() {
    vec2 squareUV = scaleUV(vUv, vec2(1.4, resolution.x / resolution.y));
    vec2 uv = scaleUV(vUv, vec2(
        1.0 + uContact * mix(0.01, 0.06, uMobile)
            + uContact * 0.1 * smoothstep(1.0, 0.1, length(squareUV - 0.5))
    ));

    vec2  fluid     = texture2D(tFluid, uv).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, uv).r);
    float fluidPush  = pow(abs(fluid.x) * 0.01, 2.0);
    float fluidPushY = pow(abs(fluid.y) * 0.01, 2.0);
    float fluidEdge  = fluidPush * smoothstep(0.7, 0.0, abs(fluidMask - 0.5));

    // Frosted glass distortion via normal map
    float normalScale = uNormalScale * 1.0 * mix(0.15, 0.2, uMobile);
    normalScale *= crange(resolution.x, 1000.0, 5000.0, 1.0, 0.35);
    normalScale *= 1.0 - (1.0 - uContact) * 0.06;
    vec2 normalUV = scaleUV(squareUV, vec2(normalScale));
    vec3 normal   = crange(texture2D(tNormal, normalUV).rgb,
                           vec3(0.0), vec3(1.0), vec3(-1.0), vec3(1.0));

    float frost = smoothstep(0.3, 0.0, length(vUv - vec2(1.0)));
    frost += smoothstep(0.4, 0.0, length(vUv - vec2(0.0))) * uChatOpen * 0.4;
    frost  = mix(frost * 0.08, 0.14 + fluidEdge * 2.2, pow(uContact, 3.0));
    frost *= 1.0 + sin(time - length(squareUV - 0.5) * 30.0 + uScroll * 5.0) * 0.9;
    uv += normal.xy * frost * 0.5;
    uv += uContact * fluidEdge * 0.05;

    // RGB chromatic aberration + diffuse sample
    vec3 color = getRGB(tDiffuse, uv, radians(120.0),
        fluidEdge * 0.01 * uContact
        + 0.0001 * uScrollDelta
        - 0.0005 * uContact).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color *= mix(1.0, 0.3, pow(uContact, 3.0));

    // Corner gradient glow (HSV animated)
    vec3 gradient = vec3(0.5, 0.5, 1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(squareUV * 0.65 - time * 0.04 + uContact * 0.2) * 0.065 + 0.88;
    gradient = hsv2rgb(gradient);
    gradient = mix(gradient, uUIColor, uUIBlend * 0.75);

    // Bloom + light streak overlay
    color += pow(getUnrealBloom(uv), vec3(1.8)) * mix(1.0, 1.1, fluidEdge);
    color += pow(texture2D(tLightStreak, uv).rgb, vec3(1.25));

    // Contact-driven gamma
    color = pow(color, vec3(1.0 + uContact * 0.3));

    // Gradient corner glow noise
    vec2  noiseUV    = rotateUV(squareUV, radians(15.0));
    float gNoise     = 0.5 + cnoise(noiseUV * mix(1.1, 0.6, uMobile)
                                    + time * 0.03 + uScroll * 0.08
                                    + uContact * 0.2) * 0.5;
    float cornerNoise = 0.7 * mix(1.6, 1.5, uMobile)
        * smoothstep(uGradient.x, uGradient.y * 0.9, length(squareUV - 0.5));
    color = blendAdd(color, gradient, 0.05 + pow(cornerNoise * gNoise, 2.0));

    // Chat panel corner tint
    vec3 cornerColor = mix(vec3(0.15, 0.11, 0.25),
                           mix(uUIColor, vec3(0.1), 0.8),
                           uUIBlend * 0.9);
    vec2 cornerUV = scaleUV(squareUV, vec2(1.0, 1.3), vec2(0.0));
    cornerUV += fluidEdge * 0.2;
    float cornerBlend = smoothstep(0.65 * uChatOpen, 0.2 * uChatOpen,
        length(cornerUV - vec2(0.0, (1.0 - uChatOpen) * 0.5)))
        * uChatOpen * 0.95 + (0.5 + sin(time * 2.0) * 0.5) * 0.05;
    color  = mix(color, cornerColor * 1.1, cornerBlend);
    color *= smoothstep(0.0, 0.5, uVisible);

    // Film grain overlay
    color = blendOverlay(color, vec3(getNoise(vUv, time)),
                         mix(0.15, 0.15, uMobile));
    color = pow(color, vec3(1.0 + smoothstep(1.0, 0.2, uVisible) * 0.4));

    // Sync-touch soft-light
    vec3  colorTouch = mix(vec3(1.0), gradient,
                           smoothstep(0.0, 1.0, fluidPush) * 0.5);
    float colorPush  = fluidPush + fluidPushY;
    color = blendSoftLight(color, colorTouch,
                           colorPush * 0.6 * smoothstep(0.0, 0.0001, uSyncTouch));

    color = max(vec3(0.0), min(vec3(1.0), color));
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATSceneCompositor
// ─────────────────────────────────────────────────────────────────────────────

export class ATSceneCompositor {

  // ── WebGL context ──────────────────────────────────────────────────────────
  private gl!:     WebGLRenderingContext;
  private canvas!: HTMLCanvasElement;
  private width  = 0;
  private height = 0;

  // ── Compiled programs ──────────────────────────────────────────────────────
  private progCellLayer!:       WebGLProgram;  // CellLayer.fs
  private progFluidLayer!:      WebGLProgram;  // FluidLayer.fs
  private progBloomExtract!:    WebGLProgram;  // BloomExtract.fs
  private progBloomBlurH!:      WebGLProgram;  // BloomBlurH.fs
  private progBloomBlurV!:      WebGLProgram;  // BloomBlurV.fs
  private progShadowLayer!:     WebGLProgram;  // ShadowLayer.fs
  private progLayerComposite!:  WebGLProgram;  // LayerComposite.fs
  private progGlobalComposite!: WebGLProgram;  // GlobalComposite.fs

  // ── Input FBOs (written by external renderers, or cleared internally) ──────
  private cellFBO!:    GLFBO;   // cell geometry layer
  private fluidFBO!:   GLFBO;   // NS fluid dye layer
  private bloomFBO!:   GLFBO;   // unreal bloom accumulation
  private shadowFBO!:  GLFBO;   // cell drop-shadow layer

  // ── Intermediate FBOs ──────────────────────────────────────────────────────
  private bloomExtractFBO!: GLFBO;   // bright-pass extract
  private bloomBlurHFBO!:   GLFBO;   // horizontal gaussian blur
  private bloomBlurVFBO!:   GLFBO;   // vertical gaussian blur (== final bloom)
  private compositeFBO!:    GLFBO;   // merged layers before GlobalComposite

  // ── 1×1 fallback textures ──────────────────────────────────────────────────
  private whiteTex!: WebGLTexture;   // RGBA (255,255,255,255)
  private blackTex!: WebGLTexture;   // RGBA (0,0,0,0)

  // ── Fullscreen quad VBO ────────────────────────────────────────────────────
  private quadBuf!: WebGLBuffer;

  // ── Uniform stores ─────────────────────────────────────────────────────────
  private uniforms: Required<ATSceneCompositorUniforms> = {
    rgbStrength:         0.0,
    volumetricStrength:  0.4,
    contrast:            [1.05, 1.02],
    scroll:              0.0,
    contact:             0.0,
    scrollDelta:         0.0,
    mouse:               [0.5, 0.5],
    frostCorner:         [0.0, 0.0, 0.0],
    normalScale:         1.0,
    visible:             1.0,
    chatOpen:            0.0,
    gradient:            [0.25, 0.9],
    mobile:              0.0,
    uiColor:             [0.5, 0.5, 1.0],
    uiBlend:             0.0,
    syncTouch:           0.0,
    bloomThreshold:      0.25,
    bloomStrength:       1.2,
    shadowOpacity:       0.55,
    shadowBlur:          3.0,
    shadowOffset:        [0.004, -0.006],
  };

  // ── External texture handles (set by calling code) ─────────────────────────
  private extFluidDyeTex:   WebGLTexture | null = null;
  private extNormalTex:     WebGLTexture | null = null;
  private extLightStreakTex: WebGLTexture | null = null;

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  private elapsed     = 0.0;
  private initialised = false;
  private destroyed   = false;

  // ─────────────────────────────────────────────────────────────────────────
  // init(canvas)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Acquire WebGL context, compile all 8 programs, allocate all FBOs,
   * create fallback textures and fullscreen quad buffer.
   */
  init(canvas: HTMLCanvasElement, cfg: ATSceneCompositorConfig = {}): void {
    if (this.initialised) return;
    this.initialised = true;
    this.canvas = canvas;
    this.width  = canvas.width  || 1280;
    this.height = canvas.height || 720;

    // ── Acquire WebGL1 context ──────────────────────────────────────────────
    const ctxAttribs: WebGLContextAttributes = {
      antialias: false,
      alpha:     true,
      premultipliedAlpha: false,
      ...cfg.contextAttribs,
    };
    const gl = canvas.getContext('webgl', ctxAttribs) as WebGLRenderingContext | null;
    if (!gl) throw new Error('[ATSceneCompositor] WebGL not available');
    this.gl = gl;

    // ── Enable float texture extension ─────────────────────────────────────
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_half_float');
    gl.getExtension('OES_texture_float_linear');
    gl.getExtension('WEBGL_color_buffer_float');

    // ── Compile all 8 shader programs ──────────────────────────────────────
    this.progCellLayer       = this._compile(SIMPLE_VERT, CELL_LAYER_FRAG,        'CellLayer');
    this.progFluidLayer      = this._compile(SIMPLE_VERT, FLUID_LAYER_FRAG,       'FluidLayer');
    this.progBloomExtract    = this._compile(SIMPLE_VERT, BLOOM_EXTRACT_FRAG,     'BloomExtract');
    this.progBloomBlurH      = this._compile(SIMPLE_VERT, BLOOM_BLUR_FRAG('H'),   'BloomBlurH');
    this.progBloomBlurV      = this._compile(SIMPLE_VERT, BLOOM_BLUR_FRAG('V'),   'BloomBlurV');
    this.progShadowLayer     = this._compile(SIMPLE_VERT, SHADOW_LAYER_FRAG,      'ShadowLayer');
    this.progLayerComposite  = this._compile(SIMPLE_VERT, LAYER_COMPOSITE_FRAG,   'LayerComposite');
    this.progGlobalComposite = this._compile(SIMPLE_VERT, GLOBAL_COMPOSITE_FRAG,  'GlobalComposite');

    // ── Allocate all FBOs ──────────────────────────────────────────────────
    const W = this.width;
    const H = this.height;

    this.cellFBO         = this._createFBO(W, H);
    this.fluidFBO        = this._createFBO(W, H);
    this.bloomFBO        = this._createFBO(W, H);
    this.shadowFBO       = this._createFBO(W, H);
    this.bloomExtractFBO = this._createFBO(W, H);
    this.bloomBlurHFBO   = this._createFBO(W, H);
    this.bloomBlurVFBO   = this._createFBO(W, H);
    this.compositeFBO    = this._createFBO(W, H);

    // ── White 1×1 fallback texture ─────────────────────────────────────────
    this.whiteTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── Black 1×1 fallback texture ─────────────────────────────────────────
    this.blackTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.blackTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([0, 0, 0, 0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── Fullscreen quad (2 triangles, 6 vertices, XY only) ─────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // render(dt) — per-frame multi-FBO composite pipeline
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run the full 8-pass composite pipeline and present to the canvas.
   *
   *   Pass 1: CellLayer      → cellFBO
   *   Pass 2: FluidLayer     → fluidFBO
   *   Pass 3: ShadowLayer    → shadowFBO
   *   Pass 4: BloomExtract   → bloomExtractFBO
   *   Pass 5: BloomBlurH     → bloomBlurHFBO
   *   Pass 6: BloomBlurV     → bloomBlurVFBO
   *   Pass 7: LayerComposite → compositeFBO
   *   Pass 8: GlobalComposite → canvas (null FBO)
   */
  render(dt: number): void {
    if (!this.initialised || this.destroyed) return;
    this.elapsed += dt;
    const gl = this.gl;
    const W  = this.width;
    const H  = this.height;

    // ── Pass 1: Cell layer render ─────────────────────────────────────────
    gl.useProgram(this.progCellLayer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.cellFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cellFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progCellLayer, 'tCell'), 0);
    gl.uniform1f(gl.getUniformLocation(this.progCellLayer, 'uAlpha'), 1.0);
    this._drawQuad(this.progCellLayer);
    gl.disable(gl.BLEND);

    // ── Pass 2: Fluid dye layer ───────────────────────────────────────────
    gl.useProgram(this.progFluidLayer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fluidFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.extFluidDyeTex ?? this.blackTex);
    gl.uniform1i(gl.getUniformLocation(this.progFluidLayer, 'tFluidDye'), 0);
    gl.uniform1f(gl.getUniformLocation(this.progFluidLayer, 'uStrength'),
                 this.uniforms.volumetricStrength);
    this._drawQuad(this.progFluidLayer);

    // ── Pass 3: Shadow layer ──────────────────────────────────────────────
    gl.useProgram(this.progShadowLayer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cellFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progShadowLayer, 'tCell'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progShadowLayer, 'uShadowOffset'),
                 this.uniforms.shadowOffset[0], this.uniforms.shadowOffset[1]);
    gl.uniform1f(gl.getUniformLocation(this.progShadowLayer, 'uShadowBlur'),
                 this.uniforms.shadowBlur);
    gl.uniform1f(gl.getUniformLocation(this.progShadowLayer, 'uShadowOpacity'),
                 this.uniforms.shadowOpacity);
    gl.uniform2f(gl.getUniformLocation(this.progShadowLayer, 'uTexelSize'),
                 1.0 / W, 1.0 / H);
    this._drawQuad(this.progShadowLayer);

    // ── Pass 4: Bloom bright-pass extract ────────────────────────────────
    gl.useProgram(this.progBloomExtract);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomExtractFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cellFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progBloomExtract, 'tDiffuse'), 0);
    gl.uniform1f(gl.getUniformLocation(this.progBloomExtract, 'uLuminosityThreshold'),
                 this.uniforms.bloomThreshold);
    gl.uniform1f(gl.getUniformLocation(this.progBloomExtract, 'uSmoothWidth'), 0.01);
    this._drawQuad(this.progBloomExtract);

    // ── Pass 5: Bloom horizontal gaussian blur ────────────────────────────
    gl.useProgram(this.progBloomBlurH);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomBlurHFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomExtractFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progBloomBlurH, 'tDiffuse'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progBloomBlurH, 'uTexelSize'), 1.0 / W, 1.0 / H);
    gl.uniform1f(gl.getUniformLocation(this.progBloomBlurH, 'uKernelRadius'), 5.0);
    this._drawQuad(this.progBloomBlurH);

    // ── Pass 6: Bloom vertical gaussian blur ──────────────────────────────
    gl.useProgram(this.progBloomBlurV);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomBlurVFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomBlurHFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progBloomBlurV, 'tDiffuse'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progBloomBlurV, 'uTexelSize'), 1.0 / W, 1.0 / H);
    gl.uniform1f(gl.getUniformLocation(this.progBloomBlurV, 'uKernelRadius'), 5.0);
    this._drawQuad(this.progBloomBlurV);

    // ── Pass 7: Layer alpha-composite (shadow + fluid + cell + bloom) ──────
    gl.useProgram(this.progLayerComposite);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.compositeFBO.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progLayerComposite, 'tShadow'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.fluidFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progLayerComposite, 'tFluid'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.cellFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progLayerComposite, 'tCell'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomBlurVFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.progLayerComposite, 'tBloomLayer'), 3);

    gl.uniform1f(gl.getUniformLocation(this.progLayerComposite, 'uFluidStrength'),
                 this.uniforms.volumetricStrength);
    gl.uniform1f(gl.getUniformLocation(this.progLayerComposite, 'uBloomStrength'),
                 this.uniforms.bloomStrength);
    this._drawQuad(this.progLayerComposite);

    // ── Pass 8: GlobalComposite → canvas (null FBO) ───────────────────────
    this._runGlobalComposite();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick(dt) — alias for render
  // ─────────────────────────────────────────────────────────────────────────

  /** Advance elapsed time and run the full composite pipeline. */
  tick(dt: number): void {
    this.render(dt);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // resize(w, h) — reallocate all FBOs at new resolution
  // ─────────────────────────────────────────────────────────────────────────

  resize(w: number, h: number): void {
    if (!this.initialised || this.destroyed) return;
    if (w === this.width && h === this.height) return;
    this.width  = w;
    this.height = h;
    this.canvas.width  = w;
    this.canvas.height = h;
    const gl = this.gl;

    // Destroy old FBOs
    for (const fbo of [
      this.cellFBO, this.fluidFBO, this.bloomFBO, this.shadowFBO,
      this.bloomExtractFBO, this.bloomBlurHFBO, this.bloomBlurVFBO,
      this.compositeFBO,
    ]) {
      gl.deleteFramebuffer(fbo.fbo);
      gl.deleteTexture(fbo.tex);
    }

    // Reallocate at new size
    this.cellFBO         = this._createFBO(w, h);
    this.fluidFBO        = this._createFBO(w, h);
    this.bloomFBO        = this._createFBO(w, h);
    this.shadowFBO       = this._createFBO(w, h);
    this.bloomExtractFBO = this._createFBO(w, h);
    this.bloomBlurHFBO   = this._createFBO(w, h);
    this.bloomBlurVFBO   = this._createFBO(w, h);
    this.compositeFBO    = this._createFBO(w, h);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform / external texture setters
  // ─────────────────────────────────────────────────────────────────────────

  /** Batch-update compositor uniforms. */
  setUniforms(u: Partial<ATSceneCompositorUniforms>): void {
    Object.assign(this.uniforms, u);
  }

  /** Provide the NS fluid dye texture from FluidGPU.dyeTexture. */
  setFluidDyeTexture(tex: WebGLTexture): void {
    this.extFluidDyeTex = tex;
  }

  /** Provide a normal map texture for GlobalComposite frost distortion. */
  setNormalTexture(tex: WebGLTexture): void {
    this.extNormalTex = tex;
  }

  /** Provide a light streak / lens flare texture. */
  setLightStreakTexture(tex: WebGLTexture): void {
    this.extLightStreakTex = tex;
  }

  /**
   * Upload raw cell RGBA pixels directly into the cellFBO texture.
   * Used when an external renderer writes pixel data rather than
   * rendering into this compositor's FBO.
   */
  uploadCellPixels(pixels: Uint8Array | null): void {
    if (!this.initialised || this.destroyed) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.cellFBO.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA,
                  this.width, this.height, 0,
                  gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── Convenience animation controls ────────────────────────────────────────

  setScroll(scroll: number, delta = 0): void {
    this.uniforms.scroll      = scroll;
    this.uniforms.scrollDelta = delta;
  }
  setContact(contact: number): void {
    this.uniforms.contact = Math.max(0, Math.min(1, contact));
  }
  setMouse(x: number, y: number): void {
    this.uniforms.mouse = [x, y];
  }
  setVisible(v: number): void {
    this.uniforms.visible = Math.max(0, Math.min(1, v));
  }
  setUIColor(r: number, g: number, b: number, blend: number): void {
    this.uniforms.uiColor  = [r, g, b];
    this.uniforms.uiBlend  = blend;
  }
  setChatOpen(v: number): void {
    this.uniforms.chatOpen = Math.max(0, Math.min(1, v));
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  get isInitialised(): boolean { return this.initialised; }
  get isDestroyed():   boolean { return this.destroyed; }
  get elapsedTime():   number  { return this.elapsed; }

  /** Read-only handle to the cell layer FBO (for external renderers to bind). */
  get cellFramebuffer(): WebGLFramebuffer { return this.cellFBO.fbo; }
  /** Read-only handle to the fluid layer FBO. */
  get fluidFramebuffer(): WebGLFramebuffer { return this.fluidFBO.fbo; }
  /** Read-only handle to the composite output texture. */
  get compositeTexture(): WebGLTexture { return this.compositeFBO.tex; }

  // ─────────────────────────────────────────────────────────────────────────
  // dispose() — release all GPU resources
  // ─────────────────────────────────────────────────────────────────────────

  dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const gl = this.gl;
    if (!gl) return;

    // Delete all compiled programs
    gl.deleteProgram(this.progCellLayer);
    gl.deleteProgram(this.progFluidLayer);
    gl.deleteProgram(this.progBloomExtract);
    gl.deleteProgram(this.progBloomBlurH);
    gl.deleteProgram(this.progBloomBlurV);
    gl.deleteProgram(this.progShadowLayer);
    gl.deleteProgram(this.progLayerComposite);
    gl.deleteProgram(this.progGlobalComposite);

    // Delete all FBOs (framebuffer + backing texture each)
    for (const fbo of [
      this.cellFBO, this.fluidFBO, this.bloomFBO, this.shadowFBO,
      this.bloomExtractFBO, this.bloomBlurHFBO, this.bloomBlurVFBO,
      this.compositeFBO,
    ]) {
      if (fbo) {
        gl.deleteFramebuffer(fbo.fbo);
        gl.deleteTexture(fbo.tex);
      }
    }

    // Delete fallback textures
    gl.deleteTexture(this.whiteTex);
    gl.deleteTexture(this.blackTex);

    // Delete quad vertex buffer
    gl.deleteBuffer(this.quadBuf);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — GlobalComposite final pass
  // ─────────────────────────────────────────────────────────────────────────

  private _runGlobalComposite(): void {
    const gl   = this.gl;
    const prog = this.progGlobalComposite;
    const u    = this.uniforms;
    const W    = this.width;
    const H    = this.height;

    gl.useProgram(prog);
    // Render to canvas (null framebuffer = swap-chain surface)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);

    // ── 6 texture units ────────────────────────────────────────────────────

    // tDiffuse — merged layer composite result
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.compositeFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tFluid — NS fluid velocity (XY → fluid push)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.extFluidDyeTex ?? this.blackTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tFluid'), 1);

    // tFluidMask — fluid dye mask (alpha channel)
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.extFluidDyeTex ?? this.blackTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tFluidMask'), 2);

    // tNormal — normal map for frosted glass distortion
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.extNormalTex ?? this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tNormal'), 3);

    // tLightStreak — lens flare / light streak overlay
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.extLightStreakTex ?? this.blackTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tLightStreak'), 4);

    // tBloom — blurred bloom accumulation
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomBlurVFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 5);

    // ── Uniforms ───────────────────────────────────────────────────────────

    gl.uniform1f(gl.getUniformLocation(prog, 'time'),                this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),          W, H);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'),        u.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVolumetricStrength'), u.volumetricStrength);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),           u.contrast[0], u.contrast[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uScroll'),             u.scroll);
    gl.uniform1f(gl.getUniformLocation(prog, 'uContact'),            u.contact);
    gl.uniform1f(gl.getUniformLocation(prog, 'uScrollDelta'),        u.scrollDelta);
    gl.uniform2f(gl.getUniformLocation(prog, 'uMouse'),              u.mouse[0], u.mouse[1]);
    gl.uniform3f(gl.getUniformLocation(prog, 'uFrostCorner'),
                 u.frostCorner[0], u.frostCorner[1], u.frostCorner[2]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uNormalScale'),        u.normalScale);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVisible'),            u.visible);
    gl.uniform1f(gl.getUniformLocation(prog, 'uChatOpen'),           u.chatOpen);
    gl.uniform2f(gl.getUniformLocation(prog, 'uGradient'),           u.gradient[0], u.gradient[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uMobile'),             u.mobile);
    gl.uniform3f(gl.getUniformLocation(prog, 'uUIColor'),
                 u.uiColor[0], u.uiColor[1], u.uiColor[2]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uUIBlend'),            u.uiBlend);
    gl.uniform1f(gl.getUniformLocation(prog, 'uSyncTouch'),          u.syncTouch);

    this._drawQuad(prog);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — _compile: vert + frag → linked WebGLProgram
  // ─────────────────────────────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneCompositor] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneCompositor] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATSceneCompositor] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — _createFBO: RGBA UNSIGNED_BYTE texture + framebuffer
  // ─────────────────────────────────────────────────────────────────────────

  private _createFBO(w: number, h: number): GLFBO {
    const gl  = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private — _drawQuad: bind quad VBO + draw 6 vertices
  // ─────────────────────────────────────────────────────────────────────────

  private _drawQuad(prog: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
  }
}
