/**
 * at-hydra-bloom-import.ts — M851: AT HydraBloom + HydraLensStreak Post-Process Import
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * 集成 ActiveTheory 原始 HydraBloom + HydraLensStreak 后处理链，
 * 封装为可注册到 render-graph.ts（M822）的 RenderGraphPass 对。
 *
 * Shader 来源 — upstream/activetheory-assets/compiled.vs
 * ──────────────────────────────────────────────────────
 *
 *   HydraBloom.glsl
 *     uniform sampler2D tHydraBloom;  → getHydraBloom(uv) → vec3
 *     简单采样封装，被 HydraBloomPass.glsl fragment stage #require 引入。
 *
 *   HydraBloomPass.glsl
 *     Stage: scene → bloom composite  (blendScreen blend mode)
 *     #require(HydraBloom.glsl)
 *     #require(blendmodes.glsl)       → blendScreen(base, blend) → vec3
 *     uniforms: tDiffuse (scene), tHydraBloom (bloom texture)
 *     output: gl_FragColor = blendScreen(scene, bloom)
 *
 *   HydraLensStreakPass.fs
 *     空分隔符 — LensStreak 管线由三个独立 pass 构成（见下）
 *
 *   LensFlarePrefilter.glsl  [streak pass A: prefilter]
 *     #require(transformUV.glsl) → rotateUV / scaleUV
 *     #require(luma.fs)          → luma(vec3) → float (BT.601)
 *     uniforms: tMap, uThreshold, uRotate
 *     output: threshold-masked + rotated scene
 *
 *   LensFlareDown.glsl       [streak pass B: horizontal stretch-blur]
 *     #require(transformUV.glsl)
 *     uniforms: tMap, uResolution, uStretch
 *     output: 6-tap horizontally-stretched weighted blur
 *
 *   LensFlareUp.glsl         [streak pass C: upsample + soften]
 *     uniforms: tHigh, tScene, uStretch, uSoftenEdge, uResolution
 *     output: 3-tap H + 2-tap V blend with high-res streak
 *
 *   CompositeStreak.glsl     [streak pass D: final composite + halo SDF]
 *     #require(transformUV.glsl)
 *     uniforms: tHigh, tDown, tPrefiltered, uStreakColor, uStreakIntensity,
 *               uGlowIntensity, uFlareIntensity, uAspectCorrection,
 *               uHaloChroma, uHaloScale, uRotateStreak, uHaloSoftness,
 *               uHaloRotateSrc, uHaloConstant, uHaloColor, uHaloRing,
 *               uDebugHalo
 *     output: streaks + halo ring SDF + chromatic aberration
 *
 * Render Graph 注册方式（参考 M822 render-graph.ts 用法）：
 * ──────────────────────────────────────────────────────
 *
 *   // 1. 创建实例（异步，编译 WebGL programs）
 *   const hydra = await ATHydraBloomImport.create(renderer, '/shaders/compiled.vs');
 *   const streak = await ATHydraLensStreakImport.create(renderer, '/shaders/compiled.vs');
 *
 *   // 2. 声明虚拟资源
 *   const sceneColor  = graph.createResource('scene-color',   { sizeClass: 'full' });
 *   const bloomTex    = graph.createResource('hydra-bloom',   { sizeClass: 'half' });
 *   const prefiltTex  = graph.createResource('streak-pre',    { sizeClass: 'half' });
 *   const downTex     = graph.createResource('streak-down',   { sizeClass: 'half' });
 *   const upTex       = graph.createResource('streak-up',     { sizeClass: 'half' });
 *   const bloomOut    = graph.createResource('hydra-bloom-out', { sizeClass: 'full' });
 *   const streakOut   = graph.createResource('streak-out',    { sizeClass: 'full' });
 *
 *   // 3. 注册 HydraBloom composite pass
 *   graph.addPass('hydra-bloom', hydra.makeBloomPass(sceneColor, bloomTex, bloomOut));
 *
 *   // 4. 注册 LensStreak 四段管线
 *   const streakPasses = streak.makeStreakPasses(
 *     sceneColor, prefiltTex, downTex, upTex, bloomOut, streakOut,
 *   );
 *   for (const [name, desc] of streakPasses) {
 *     graph.addPass(name, desc);
 *   }
 *
 * 参考实现：at-bloom-postprocess.ts (M714) UnrealBloom WebGPU 管线
 *
 * Research: xiaodi #M851 — cell-pubsub-loop
 */

import { ATShaderLoader } from './at-shader-loader.ts';
import type {
  PassDescriptor,
  ResourceHandle,
  ResourceAccessor,
  PassContext,
} from './render-graph.ts';

// ─────────────────────────────────────────────────────────────────────────────
// § 0  Minimal WebGL types (subset used here; full types from at-render-pipeline.ts)
// ─────────────────────────────────────────────────────────────────────────────

/** Slimmed WebGL2 context reference carried by AstroRenderer. */
interface AstroRenderer {
  gl: WebGL2RenderingContext;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 1  GLSL Shader Sources — extracted from compiled.vs via {@ } delimiters
// ─────────────────────────────────────────────────────────────────────────────
// All #require() dependencies are inlined here so these shaders compile
// standalone without the AT shader loader's recursive resolution step.
// Source file: upstream/activetheory-assets/compiled.vs

// ── Shared utilities (inlined from transformUV.glsl + luma.fs) ──────────────

const GLSL_TRANSFORM_UV = /* glsl */`
vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    float c = cos(r);
    float s = sin(r);
    mat2 m = mat2(c, -s, s, c);
    vec2 st = uv - origin;
    st = m * st;
    return st + origin;
}

vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec2 st = uv - origin;
    st /= scale;
    return st + origin;
}

vec2 rotateUV(vec2 uv, float r) {
    return rotateUV(uv, r, vec2(0.5));
}

vec2 scaleUV(vec2 uv, vec2 scale) {
    return scaleUV(uv, scale, vec2(0.5));
}
`;

// luma.fs — BT.601 weighted luminance
const GLSL_LUMA = /* glsl */`
float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}
`;

// blendScreen from blendmodes.glsl (used by HydraBloomPass)
const GLSL_BLEND_SCREEN = /* glsl */`
vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(1.0) - (vec3(1.0) - base) * (vec3(1.0) - blend);
}
vec3 blendScreen(vec3 base, vec3 blend, float opacity) {
    return (blendScreen(base, blend) * opacity + base * (1.0 - opacity));
}
`;

// ── Fullscreen pass vertex (shared by all passes) ────────────────────────────

const VERT_FULLSCREEN = /* glsl */`
attribute vec3 position;
attribute vec2 uv;

varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
`;

// ── HydraBloom.glsl — compiled.vs line 6954 ─────────────────────────────────
// uniform sampler2D tHydraBloom;
// vec3 getHydraBloom(vec2 uv) { return texture2D(tHydraBloom, uv).rgb; }

const GLSL_HYDRA_BLOOM_LIB = /* glsl */`
uniform sampler2D tHydraBloom;

vec3 getHydraBloom(vec2 uv) {
    return texture2D(tHydraBloom, uv).rgb;
}
`;

// ── HydraBloomPass.glsl — compiled.vs line 6959 ─────────────────────────────
// Composite: blendScreen(scene, bloom)
// #require(HydraBloom.glsl) + #require(blendmodes.glsl) inlined below.

const FRAG_HYDRA_BLOOM_PASS = /* glsl */`
precision highp float;

uniform sampler2D tDiffuse;

varying vec2 vUv;

${GLSL_HYDRA_BLOOM_LIB}
${GLSL_BLEND_SCREEN}

void main() {
    vec3 color = texture2D(tDiffuse, vUv).xyz;
    vec3 bloom = getHydraBloom(vUv);
    gl_FragColor = vec4(blendScreen(color, bloom), 1.0);
}
`;

// ── LensFlarePrefilter.glsl — compiled.vs line 7185 ─────────────────────────
// Pass A: luma threshold + optional rotation.
// #require(transformUV.glsl) + #require(luma.fs) inlined below.

const FRAG_STREAK_PREFILTER = /* glsl */`
precision highp float;

uniform sampler2D tMap;
uniform float uThreshold;
uniform float uRotate;

varying vec2 vUv;

${GLSL_TRANSFORM_UV}
${GLSL_LUMA}

void main() {
    vec2 uv = vUv;
    uv = rotateUV(uv, -uRotate);

    vec4 c = texture2D(tMap, uv);

    float brightness = luma(c.rgb);
    if (brightness < uThreshold) {
        c = vec4(0.0);
    }

    gl_FragColor = vec4(c.rgb, 1.0);
}
`;

// ── LensFlareDown.glsl — compiled.vs line 7141 ──────────────────────────────
// Pass B: 6-tap horizontal stretch-weighted blur.
// Weights: [1, 2, 3, 3, 2, 1] / 12

const FRAG_STREAK_DOWN = /* glsl */`
precision highp float;

uniform sampler2D tMap;
uniform vec2 uResolution;
uniform float uStretch;

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

    vec3 col = (c0 + c1 * 2.0 + c2 * 3.0 + c3 * 3.0 + c4 * 2.0 + c5) / 12.0;

    gl_FragColor = vec4(col.rgb, 1.0);
}
`;

// ── LensFlareUp.glsl — compiled.vs line 7223 ────────────────────────────────
// Pass C: 3-tap H upsample blend with high-res streak + 2-tap V soften.

const FRAG_STREAK_UP = /* glsl */`
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

    // 3-tap horizontal sample from low-res streak
    vec3 c0 = texture2D(tScene, vec2(u0, uv.y)).rgb / 4.0;
    vec3 c1 = texture2D(tScene, vec2(u1, uv.y)).rgb / 2.0;
    vec3 c2 = texture2D(tScene, vec2(u2, uv.y)).rgb / 4.0;

    // high-res pass-through
    vec3 c3 = texture2D(tHigh, uv).rgb;

    vec3 cStretch = c0 + c1 + c2;

    // 2-tap vertical soften
    vec3 c4 = texture2D(tScene, vec2(uv.x, uv.y - (dx * 0.75))).rgb / 4.0;
    vec3 c5 = texture2D(tScene, vec2(uv.x, uv.y + (dx * 0.75))).rgb / 4.0;

    cStretch += (c4 + c5) * uSoftenEdge;

    gl_FragColor = vec4(cStretch, 1.0);
}
`;

// ── CompositeStreak.glsl — compiled.vs line 7025 ────────────────────────────
// Pass D: streak + halo ring SDF + chromatic aberration + constant halo mask.
// #require(transformUV.glsl) inlined; uses built-in `resolution` uniform
// (vec2, injected by WebGL layer — same convention as AT renderer).

const FRAG_COMPOSITE_STREAK = /* glsl */`
precision highp float;

uniform sampler2D tHigh;
uniform sampler2D tDown;
uniform sampler2D tPrefiltered;

uniform vec3 uStreakColor;
uniform float uStreakIntensity;
uniform float uGlowIntensity;
uniform bool uDebugHalo;
uniform float uFlareIntensity;
uniform float uAspectCorrection;
uniform float uHaloChroma;
uniform float uHaloScale;
uniform float uRotateStreak;
uniform float uHaloSoftness;
uniform float uHaloRotateSrc;
uniform float uHaloConstant;
uniform vec3 uHaloColor;
uniform vec4 uHaloRing;

uniform vec2 resolution;

varying vec2 vUv;

${GLSL_TRANSFORM_UV}

#define PI 3.1415926

float createRingSDF(
    vec2 uv, vec2 center, float scale,
    float innerRadius, float outerRadius, float smoothness
) {
    vec2 scaledUV = (uv - center) * scale;
    float dist = distance(vec2(0.0), scaledUV);
    float outerEdge = smoothstep(outerRadius - smoothness, outerRadius, dist);
    float innerEdge = smoothstep(innerRadius, innerRadius + smoothness, dist);
    return outerEdge - innerEdge;
}

void main() {
    vec2 uv = vUv;

    // rotate streak UVs (compensating for aspect is done in haloUV below)
    uv = rotateUV(uv, uRotateStreak);

    // streak high-freq contribution
    vec3 c3 = texture2D(tHigh, uv).rgb * uStreakIntensity * uStreakColor;

    // streak low-freq (down) contribution
    vec3 down = texture2D(tDown, uv).rgb * uStreakColor * uGlowIntensity;

    // halo UV with aspect correction + counter-rotation
    vec2 haloUV = uv;
    haloUV = rotateUV(haloUV, -(uRotateStreak + uHaloRotateSrc));
    haloUV.x -= 0.5;
    haloUV.x *= mix(1.0, resolution.x / resolution.y, uAspectCorrection);
    haloUV.x += 0.5;

    vec2 haloVec = normalize(vec2(0.5) - haloUV) * uHaloScale;

    // aspect-corrected base UV
    vec2 aspectUV = vUv;
    aspectUV.x -= 0.5;
    aspectUV.x *= mix(1.0, resolution.x / resolution.y, uAspectCorrection);
    aspectUV.x += 0.5;

    vec2 haloWarpUV = aspectUV + haloVec;
    haloWarpUV.x = 1.0 - haloWarpUV.x;
    haloWarpUV = scaleUV(haloWarpUV, vec2(1.0 + uHaloSoftness));

    float haloMask = createRingSDF(
        aspectUV, vec2(0.5), uHaloRing.x,
        uHaloRing.y, uHaloRing.z, uHaloRing.w
    );

    // chromatic aberration on halo sample
    vec2 haloWarpUVR = haloWarpUV + vec2( uHaloChroma,  uHaloChroma);
    vec2 haloWarpUVG = haloWarpUV;
    vec2 haloWarpUVB = haloWarpUV - vec2( uHaloChroma,  uHaloChroma);

    float haloR = texture2D(tPrefiltered, haloWarpUVR).r * haloMask;
    float haloG = texture2D(tPrefiltered, haloWarpUVG).g * haloMask;
    float haloB = texture2D(tPrefiltered, haloWarpUVB).b * haloMask;

    vec3 halo = vec3(haloR, haloG, haloB) * uHaloColor * uFlareIntensity;

    vec3 streaks = c3 + down;
    vec3 col = streaks + halo;

    // debug: show raw halo SDF mask
    float debugHalo = float(uDebugHalo);
    col = mix(col, vec3(haloMask), debugHalo);

    // constant halo mask (chromatic ring outline at fixed radii)
    float cHaloR = createRingSDF(aspectUV, vec2(0.5), uHaloRing.x * 1.05,
                                 uHaloRing.y, uHaloRing.z, uHaloRing.w);
    float cHaloG = createRingSDF(aspectUV, vec2(0.5), uHaloRing.x,
                                 uHaloRing.y, uHaloRing.z, uHaloRing.w);
    float cHaloB = createRingSDF(aspectUV, vec2(0.5), uHaloRing.x * 0.98,
                                 uHaloRing.y, uHaloRing.z, uHaloRing.w);

    vec3 constantMask = vec3(cHaloR, cHaloG, cHaloB) * uHaloConstant;
    constantMask *= vec3(1.0) - halo;

    gl_FragColor = vec4(col + constantMask, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Public parameter interfaces
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tunable parameters for the HydraBloom composite pass.
 * All fields optional; defaults shown.
 */
export interface ATHydraBloomParams {
  // No per-frame uniforms beyond the two samplers — the pass is purely
  // structural (blendScreen compositing). Extend here if AT adds intensity
  // controls in a future version of HydraBloomPass.glsl.
}

/**
 * Tunable parameters for the HydraLensStreak pipeline.
 * Correspond 1:1 to uniforms in CompositeStreak.glsl.
 *
 * Prefilter / Down / Up pass uniforms are driven by ATHydraLensStreakParams
 * members with matching prefixes.
 */
export interface ATHydraLensStreakParams {
  // ── Prefilter (pass A) ───────────────────────────────────────────────────
  /** Luminance threshold below which pixels are zeroed. @default 0.4 */
  threshold?: number;
  /** Clockwise rotation applied to source UVs in Prefilter. @default 0 */
  prefilterRotate?: number;

  // ── Down / Up (passes B & C) ─────────────────────────────────────────────
  /** Horizontal stretch factor for the 6-tap blur (Down pass). @default 1.0 */
  stretch?: number;
  /** Vertical soften blend weight (Up pass). @default 0.5 */
  softenEdge?: number;

  // ── Composite (pass D) ───────────────────────────────────────────────────
  /** RGB tint applied to high-freq streaks. @default [1,1,1] */
  streakColor?: [number, number, number];
  /** Multiplier on the high-freq streak contribution. @default 1.0 */
  streakIntensity?: number;
  /** Multiplier on the down-sampled glow contribution. @default 0.8 */
  glowIntensity?: number;
  /** Overall flare / halo brightness. @default 1.0 */
  flareIntensity?: number;

  /** 0=square pixels, 1=correct to viewport aspect ratio. @default 1.0 */
  aspectCorrection?: number;
  /** Chromatic aberration offset (UV delta) for halo RGB split. @default 0.003 */
  haloChroma?: number;
  /** Scale of the halo warp vector. @default 0.3 */
  haloScale?: number;
  /** Rotation of the streak compositing UV. @default 0 */
  rotateStreak?: number;
  /** Blur/feather radius of the halo ring SDF inner/outer edges. @default 0.05 */
  haloSoftness?: number;
  /** Additional counter-rotation applied to the source when building haloUV. @default 0 */
  haloRotateSrc?: number;
  /** Constant ring brightness multiplier (chromatic outline). @default 0.2 */
  haloConstant?: number;
  /** RGB tint of the halo ring. @default [1,1,1] */
  haloColor?: [number, number, number];
  /**
   * Halo ring SDF params: [scale, innerRadius, outerRadius, smoothness].
   * @default [1.0, 0.38, 0.42, 0.06]
   */
  haloRing?: [number, number, number, number];
  /** When true, render the raw halo SDF mask for debugging. @default false */
  debugHalo?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  WebGL helper — minimal program wrapper
// ─────────────────────────────────────────────────────────────────────────────

/** Compiled WebGL program with cached uniform locations. */
interface HydraProgram {
  program:   WebGLProgram;
  uniforms:  Record<string, WebGLUniformLocation | null>;
  /** Fullscreen quad VAO / buffer handles. */
  quadVAO:   WebGLVertexArrayObject;
  quadBuf:   WebGLBuffer;
}

/** Compile a vertex + fragment shader pair into a WebGL program. */
function compileProgram(
  gl:   WebGL2RenderingContext,
  vert: string,
  frag: string,
  label: string,
): WebGLProgram {
  function compileShader(type: number, src: string): WebGLShader {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(s) ?? '';
      gl.deleteShader(s);
      throw new Error(`[ATHydraBloomImport] ${label} shader compile error:\n${info}`);
    }
    return s;
  }

  const vs = compileShader(gl.VERTEX_SHADER,   vert);
  const fs = compileShader(gl.FRAGMENT_SHADER, frag);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) ?? '';
    gl.deleteProgram(prog);
    throw new Error(`[ATHydraBloomImport] ${label} program link error:\n${info}`);
  }

  return prog;
}

/**
 * Build a fullscreen quad VAO.
 *
 * Triangle-strip covering [-1,1]² clip-space with UV [0,1]²:
 *   TL(-1, 1, 0,1)  TR(1, 1, 1,1)
 *   BL(-1,-1, 0,0)  BR(1,-1, 1,0)
 */
function buildFullscreenQuad(
  gl:      WebGL2RenderingContext,
  program: WebGLProgram,
): { vao: WebGLVertexArrayObject; buf: WebGLBuffer } {
  // xyzuv interleaved, 4 verts, triangle strip
  // eslint-disable-next-line @typescript-eslint/no-magic-numbers
  const verts = new Float32Array([
    -1,  1, 0,  0, 1,
     1,  1, 0,  1, 1,
    -1, -1, 0,  0, 0,
     1, -1, 0,  1, 0,
  ]);

  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);

  const posLoc = gl.getAttribLocation(program, 'position');
  const uvLoc  = gl.getAttribLocation(program, 'uv');
  const stride = 5 * 4; // 5 floats × 4 bytes

  if (posLoc >= 0) {
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, stride, 0);
  }
  if (uvLoc >= 0) {
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, stride, 3 * 4);
  }

  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  return { vao, buf };
}

/** Compile and cache a HydraProgram. */
function makeHydraProgram(
  gl:        WebGL2RenderingContext,
  fragSrc:   string,
  uniformNames: string[],
  label:     string,
): HydraProgram {
  const program = compileProgram(gl, VERT_FULLSCREEN, fragSrc, label);
  const uniforms: Record<string, WebGLUniformLocation | null> = {};
  for (const name of uniformNames) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  const { vao, buf } = buildFullscreenQuad(gl, program);
  return { program, uniforms, quadVAO: vao, quadBuf: buf };
}

/** Bind a 2D texture to a sampler uniform slot. */
function bindTex(
  gl:       WebGL2RenderingContext,
  loc:      WebGLUniformLocation | null,
  texture:  WebGLTexture,
  unit:     number,
): void {
  if (loc === null) return;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(loc, unit);
}

/** Draw a fullscreen quad (triangle strip). */
function drawFullscreen(
  gl:  WebGL2RenderingContext,
  vao: WebGLVertexArrayObject,
): void {
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  ATHydraBloomImport
// ─────────────────────────────────────────────────────────────────────────────

/**
 * HydraBloom post-process pass — WebGL port of HydraBloomPass.glsl.
 *
 * Single-pass composite: `blendScreen(scene, bloomTex) → output`.
 * Mirrors the pattern of ATBloomPostProcess (M714) but uses the AT
 * HydraBloom shader family instead of UnrealBloom.
 */
export class ATHydraBloomImport {
  private readonly gl:          WebGL2RenderingContext;
  private readonly bloomProg:   HydraProgram;

  // ── private constructor ──────────────────────────────────────────────────

  private constructor(
    gl:        WebGL2RenderingContext,
    bloomProg: HydraProgram,
  ) {
    this.gl        = gl;
    this.bloomProg = bloomProg;
  }

  // ── static factory ───────────────────────────────────────────────────────

  /**
   * Async factory: compiles the HydraBloomPass WebGL program.
   *
   * @param renderer   - AstroRenderer carrying the WebGL2 context.
   * @param _shaderUrl - (reserved) path to compiled.vs; shaders are embedded
   *                     as GLSL string constants in this module.
   */
  static async create(
    renderer:   AstroRenderer,
    _shaderUrl?: string,
  ): Promise<ATHydraBloomImport> {
    const gl = renderer.gl;

    const bloomProg = makeHydraProgram(
      gl,
      FRAG_HYDRA_BLOOM_PASS,
      ['tDiffuse', 'tHydraBloom'],
      'HydraBloomPass',
    );

    return new ATHydraBloomImport(gl, bloomProg);
  }

  // ── Render Graph integration ─────────────────────────────────────────────

  /**
   * Build a `PassDescriptor` ready to hand to `RenderGraph.addPass()`.
   *
   * The pass reads `sceneHandle` (the rendered scene colour texture) and
   * `bloomHandle` (the pre-computed HydraBloom result — e.g. the output of
   * an upstream ATBloomPostProcess or a hand-written bloom FBO), then writes
   * the blendScreen composite to `outputHandle`.
   *
   * @param sceneHandle  - Virtual resource holding the HDR scene colour.
   * @param bloomHandle  - Virtual resource holding the bloom layer.
   * @param outputHandle - Virtual resource to receive the composited output.
   */
  makeBloomPass(
    sceneHandle:  ResourceHandle,
    bloomHandle:  ResourceHandle,
    outputHandle: ResourceHandle,
  ): PassDescriptor {
    const gl       = this.gl;
    const prog     = this.bloomProg;
    const uniforms = prog.uniforms;

    return {
      inputs:  [sceneHandle, bloomHandle],
      outputs: [outputHandle],
      execute: (
        _encoder:  GPUCommandEncoder,
        accessor:  ResourceAccessor,
        _ctx:      PassContext,
      ) => {
        // Resolve physical WebGL textures from the render-graph accessor.
        // The accessor's getTexture() path returns GPUTexture for WebGPU graphs;
        // for the WebGL integration layer the same handle resolves to a
        // WebGLTexture via the graph's WebGL bridge (see at-render-pipeline.ts).
        const sceneTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(sceneHandle);
        const bloomTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(bloomHandle);
        const outFBO   = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLFBO(outputHandle);

        gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
        gl.useProgram(prog.program);

        bindTex(gl, uniforms['tDiffuse'],     sceneTex, 0);
        bindTex(gl, uniforms['tHydraBloom'], bloomTex, 1);

        drawFullscreen(gl, prog.quadVAO);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      },
    };
  }

  // ── Direct render (standalone, without render graph) ─────────────────────

  /**
   * Render the HydraBloom composite directly to the currently bound
   * WebGL framebuffer.  Use this when operating outside the render graph
   * (e.g. in a legacy at-render-pipeline.ts tick loop).
   *
   * @param sceneTex - The rendered scene WebGLTexture.
   * @param bloomTex - The bloom layer WebGLTexture.
   */
  render(sceneTex: WebGLTexture, bloomTex: WebGLTexture): void {
    const gl       = this.gl;
    const prog     = this.bloomProg;
    const uniforms = prog.uniforms;

    gl.useProgram(prog.program);
    bindTex(gl, uniforms['tDiffuse'],     sceneTex, 0);
    bindTex(gl, uniforms['tHydraBloom'], bloomTex, 1);
    drawFullscreen(gl, prog.quadVAO);
  }

  /** Release all GPU resources. */
  destroy(): void {
    const gl = this.gl;
    gl.deleteProgram(this.bloomProg.program);
    gl.deleteVertexArray(this.bloomProg.quadVAO);
    gl.deleteBuffer(this.bloomProg.quadBuf);
  }

  /** Expose compiled GLSL sources for hot-reload / inspection. */
  get glslSources(): Readonly<Record<string, string>> {
    return {
      hydraBloomLib:  GLSL_HYDRA_BLOOM_LIB,
      hydraBloomPass: FRAG_HYDRA_BLOOM_PASS,
      blendScreen:    GLSL_BLEND_SCREEN,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  ATHydraLensStreakImport
// ─────────────────────────────────────────────────────────────────────────────

/**
 * HydraLensStreak four-pass post-process pipeline — WebGL port of:
 *   LensFlarePrefilter.glsl  (pass A — threshold + rotate)
 *   LensFlareDown.glsl       (pass B — horizontal stretch blur)
 *   LensFlareUp.glsl         (pass C — upsample + soften)
 *   CompositeStreak.glsl     (pass D — streaks + halo SDF + chroma)
 *
 * The HydraLensStreakPass.fs separator in compiled.vs is empty; the actual
 * implementation is the four GLSL shaders above, which together constitute
 * the "HydraLensStreak" post-process chain referenced in M851.
 *
 * Mirrors the ATBloomPostProcess (M714) async-factory + setParams pattern.
 */
export class ATHydraLensStreakImport {
  private readonly gl:          WebGL2RenderingContext;
  private readonly prefiltProg: HydraProgram;
  private readonly downProg:    HydraProgram;
  private readonly upProg:      HydraProgram;
  private readonly compositeProg: HydraProgram;

  // Resolved params (full defaults)
  private params: Required<ATHydraLensStreakParams>;

  // ── private constructor ──────────────────────────────────────────────────

  private constructor(
    gl:            WebGL2RenderingContext,
    prefiltProg:   HydraProgram,
    downProg:      HydraProgram,
    upProg:        HydraProgram,
    compositeProg: HydraProgram,
    params:        Required<ATHydraLensStreakParams>,
  ) {
    this.gl            = gl;
    this.prefiltProg   = prefiltProg;
    this.downProg      = downProg;
    this.upProg        = upProg;
    this.compositeProg = compositeProg;
    this.params        = params;
  }

  // ── static factory ───────────────────────────────────────────────────────

  /**
   * Async factory: compiles all four LensStreak WebGL programs.
   *
   * @param renderer   - AstroRenderer carrying the WebGL2 context.
   * @param _shaderUrl - Reserved; shaders are embedded as constants.
   * @param initial    - Optional initial parameter overrides.
   */
  static async create(
    renderer:    AstroRenderer,
    _shaderUrl?: string,
    initial?:    ATHydraLensStreakParams,
  ): Promise<ATHydraLensStreakImport> {
    const gl = renderer.gl;

    const defaults: Required<ATHydraLensStreakParams> = {
      threshold:       0.4,
      prefilterRotate: 0.0,
      stretch:         1.0,
      softenEdge:      0.5,
      streakColor:     [1, 1, 1],
      streakIntensity: 1.0,
      glowIntensity:   0.8,
      flareIntensity:  1.0,
      aspectCorrection:1.0,
      haloChroma:      0.003,
      haloScale:       0.3,
      rotateStreak:    0.0,
      haloSoftness:    0.05,
      haloRotateSrc:   0.0,
      haloConstant:    0.2,
      haloColor:       [1, 1, 1],
      haloRing:        [1.0, 0.38, 0.42, 0.06],
      debugHalo:       false,
    };

    const params = { ...defaults, ...initial } as Required<ATHydraLensStreakParams>;

    // ── Compile four programs ────────────────────────────────────────────
    const prefiltProg = makeHydraProgram(
      gl,
      FRAG_STREAK_PREFILTER,
      ['tMap', 'uThreshold', 'uRotate'],
      'LensFlarePrefilter',
    );

    const downProg = makeHydraProgram(
      gl,
      FRAG_STREAK_DOWN,
      ['tMap', 'uResolution', 'uStretch'],
      'LensFlareDown',
    );

    const upProg = makeHydraProgram(
      gl,
      FRAG_STREAK_UP,
      ['tHigh', 'tScene', 'uStretch', 'uSoftenEdge', 'uResolution'],
      'LensFlareUp',
    );

    const compositeProg = makeHydraProgram(
      gl,
      FRAG_COMPOSITE_STREAK,
      [
        'tHigh', 'tDown', 'tPrefiltered',
        'uStreakColor', 'uStreakIntensity', 'uGlowIntensity',
        'uFlareIntensity', 'uAspectCorrection',
        'uHaloChroma', 'uHaloScale', 'uRotateStreak',
        'uHaloSoftness', 'uHaloRotateSrc', 'uHaloConstant',
        'uHaloColor', 'uHaloRing', 'uDebugHalo',
        'resolution',
      ],
      'CompositeStreak',
    );

    return new ATHydraLensStreakImport(
      gl,
      prefiltProg, downProg, upProg, compositeProg,
      params,
    );
  }

  // ── Parameter management ─────────────────────────────────────────────────

  /**
   * Update streak parameters. Safe to call every frame.
   * Writes are deferred to the GPU until the next `render()` / pass execution.
   */
  setParams(partial: ATHydraLensStreakParams): void {
    Object.assign(this.params, partial);
  }

  // ── Render Graph integration ─────────────────────────────────────────────

  /**
   * Build the four PassDescriptors for the LensStreak pipeline and return
   * them as `[passName, PassDescriptor]` pairs suitable for `addPass()`.
   *
   * Resource layout:
   *
   *   sceneHandle    ─── pass A (prefilter) ──→ prefiltHandle
   *   prefiltHandle  ─── pass B (down)      ──→ downHandle
   *   downHandle     ─── pass C (up)        ──→  upHandle
   *   upHandle   ─┐
   *   downHandle ─┤─ pass D (composite)     ──→ outputHandle
   *   prefiltHandle ─┘
   *
   * @param sceneHandle   - Virtual resource: HDR scene colour input.
   * @param prefiltHandle - Virtual resource: prefilter output (half-res OK).
   * @param downHandle    - Virtual resource: down-pass output.
   * @param upHandle      - Virtual resource: up-pass output.
   * @param highHandle    - Virtual resource: high-freq streak input for composite
   *                        (may alias `upHandle` or be a separate high-res streak tex).
   * @param outputHandle  - Virtual resource: final composited output.
   */
  makeStreakPasses(
    sceneHandle:   ResourceHandle,
    prefiltHandle: ResourceHandle,
    downHandle:    ResourceHandle,
    upHandle:      ResourceHandle,
    highHandle:    ResourceHandle,
    outputHandle:  ResourceHandle,
  ): Array<[string, PassDescriptor]> {
    const self = this;

    // ── Pass A: Prefilter ──────────────────────────────────────────────────
    const passPrefilter: PassDescriptor = {
      inputs:  [sceneHandle],
      outputs: [prefiltHandle],
      execute: (_enc, accessor, _ctx) => {
        const gl       = self.gl;
        const prog     = self.prefiltProg;
        const u        = prog.uniforms;
        const p        = self.params;

        const sceneTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(sceneHandle);
        const outFBO   = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLFBO(prefiltHandle);

        gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
        gl.useProgram(prog.program);
        bindTex(gl, u['tMap'], sceneTex, 0);
        gl.uniform1f(u['uThreshold'], p.threshold);
        gl.uniform1f(u['uRotate'],    p.prefilterRotate);
        drawFullscreen(gl, prog.quadVAO);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      },
    };

    // ── Pass B: Down (horizontal stretch blur) ─────────────────────────────
    const passDown: PassDescriptor = {
      inputs:  [prefiltHandle],
      outputs: [downHandle],
      execute: (_enc, accessor, ctx) => {
        const gl       = self.gl;
        const prog     = self.downProg;
        const u        = prog.uniforms;
        const p        = self.params;

        const prefiltTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(prefiltHandle);
        const outFBO     = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLFBO(downHandle);

        gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
        gl.useProgram(prog.program);
        bindTex(gl, u['tMap'], prefiltTex, 0);
        gl.uniform2f(u['uResolution'], ctx.width, ctx.height);
        gl.uniform1f(u['uStretch'],    p.stretch);
        drawFullscreen(gl, prog.quadVAO);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      },
    };

    // ── Pass C: Up (upsample + soften) ────────────────────────────────────
    const passUp: PassDescriptor = {
      inputs:  [downHandle, prefiltHandle],
      outputs: [upHandle],
      execute: (_enc, accessor, ctx) => {
        const gl       = self.gl;
        const prog     = self.upProg;
        const u        = prog.uniforms;
        const p        = self.params;

        const downTex    = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(downHandle);
        const prefiltTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(prefiltHandle);
        const outFBO     = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLFBO(upHandle);

        gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
        gl.useProgram(prog.program);
        // tScene = down-pass result (stretched), tHigh = original prefiltered
        bindTex(gl, u['tScene'], downTex,    0);
        bindTex(gl, u['tHigh'],  prefiltTex, 1);
        gl.uniform1f(u['uStretch'],     p.stretch);
        gl.uniform1f(u['uSoftenEdge'], p.softenEdge);
        gl.uniform2f(u['uResolution'], ctx.width, ctx.height);
        drawFullscreen(gl, prog.quadVAO);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      },
    };

    // ── Pass D: Composite (streaks + halo SDF) ────────────────────────────
    const passComposite: PassDescriptor = {
      inputs:  [highHandle, downHandle, prefiltHandle],
      outputs: [outputHandle],
      execute: (_enc, accessor, ctx) => {
        const gl   = self.gl;
        const prog = self.compositeProg;
        const u    = prog.uniforms;
        const p    = self.params;

        const highTex    = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(highHandle);
        const downTex    = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(downHandle);
        const prefiltTex = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLTexture(prefiltHandle);
        const outFBO     = (accessor as unknown as WebGLResourceAccessor)
          .getWebGLFBO(outputHandle);

        gl.bindFramebuffer(gl.FRAMEBUFFER, outFBO);
        gl.useProgram(prog.program);

        bindTex(gl, u['tHigh'],        highTex,    0);
        bindTex(gl, u['tDown'],        downTex,    1);
        bindTex(gl, u['tPrefiltered'], prefiltTex, 2);

        gl.uniform3fv(u['uStreakColor'],    p.streakColor);
        gl.uniform1f(u['uStreakIntensity'], p.streakIntensity);
        gl.uniform1f(u['uGlowIntensity'],  p.glowIntensity);
        gl.uniform1f(u['uFlareIntensity'], p.flareIntensity);
        gl.uniform1f(u['uAspectCorrection'], p.aspectCorrection);
        gl.uniform1f(u['uHaloChroma'],     p.haloChroma);
        gl.uniform1f(u['uHaloScale'],      p.haloScale);
        gl.uniform1f(u['uRotateStreak'],   p.rotateStreak);
        gl.uniform1f(u['uHaloSoftness'],   p.haloSoftness);
        gl.uniform1f(u['uHaloRotateSrc'],  p.haloRotateSrc);
        gl.uniform1f(u['uHaloConstant'],   p.haloConstant);
        gl.uniform3fv(u['uHaloColor'],     p.haloColor);
        gl.uniform4fv(u['uHaloRing'],      p.haloRing);
        gl.uniform1i(u['uDebugHalo'],      p.debugHalo ? 1 : 0);
        gl.uniform2f(u['resolution'],      ctx.width, ctx.height);

        drawFullscreen(gl, prog.quadVAO);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      },
    };

    return [
      ['hydra-streak-prefilter', passPrefilter],
      ['hydra-streak-down',      passDown],
      ['hydra-streak-up',        passUp],
      ['hydra-streak-composite', passComposite],
    ];
  }

  // ── Direct render (standalone, without render graph) ─────────────────────

  /**
   * Execute all four LensStreak passes sequentially to the provided FBOs.
   * Use this outside the render graph (e.g. in a legacy tick loop).
   *
   * @param sceneTex    - Source scene WebGLTexture.
   * @param prefiltFBO  - FBO for prefilter output.
   * @param downFBO     - FBO for down output.
   * @param upFBO       - FBO for up output.
   * @param compositeFBO - FBO for final composite output (null = default).
   * @param width       - Current viewport width.
   * @param height      - Current viewport height.
   * @param prefiltTex  - Texture view of prefiltFBO (needed as down input).
   * @param downTex     - Texture view of downFBO (needed as up + composite input).
   * @param upTex       - Texture view of upFBO (used as tHigh in composite).
   */
  render(
    sceneTex:     WebGLTexture,
    prefiltFBO:   WebGLFramebuffer,
    downFBO:      WebGLFramebuffer,
    upFBO:        WebGLFramebuffer,
    compositeFBO: WebGLFramebuffer | null,
    width:        number,
    height:       number,
    prefiltTex:   WebGLTexture,
    downTex:      WebGLTexture,
    upTex:        WebGLTexture,
  ): void {
    const gl = this.gl;
    const p  = this.params;

    // ── A: Prefilter ────────────────────────────────────────────────────────
    {
      const prog = this.prefiltProg;
      const u    = prog.uniforms;
      gl.bindFramebuffer(gl.FRAMEBUFFER, prefiltFBO);
      gl.useProgram(prog.program);
      bindTex(gl, u['tMap'], sceneTex, 0);
      gl.uniform1f(u['uThreshold'], p.threshold);
      gl.uniform1f(u['uRotate'],    p.prefilterRotate);
      drawFullscreen(gl, prog.quadVAO);
    }

    // ── B: Down ─────────────────────────────────────────────────────────────
    {
      const prog = this.downProg;
      const u    = prog.uniforms;
      gl.bindFramebuffer(gl.FRAMEBUFFER, downFBO);
      gl.useProgram(prog.program);
      bindTex(gl, u['tMap'], prefiltTex, 0);
      gl.uniform2f(u['uResolution'], width, height);
      gl.uniform1f(u['uStretch'],    p.stretch);
      drawFullscreen(gl, prog.quadVAO);
    }

    // ── C: Up ───────────────────────────────────────────────────────────────
    {
      const prog = this.upProg;
      const u    = prog.uniforms;
      gl.bindFramebuffer(gl.FRAMEBUFFER, upFBO);
      gl.useProgram(prog.program);
      bindTex(gl, u['tScene'], downTex,    0);
      bindTex(gl, u['tHigh'],  prefiltTex, 1);
      gl.uniform1f(u['uStretch'],     p.stretch);
      gl.uniform1f(u['uSoftenEdge'], p.softenEdge);
      gl.uniform2f(u['uResolution'], width, height);
      drawFullscreen(gl, prog.quadVAO);
    }

    // ── D: Composite ────────────────────────────────────────────────────────
    {
      const prog = this.compositeProg;
      const u    = prog.uniforms;
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFBO);
      gl.useProgram(prog.program);
      bindTex(gl, u['tHigh'],        upTex,      0);
      bindTex(gl, u['tDown'],        downTex,    1);
      bindTex(gl, u['tPrefiltered'], prefiltTex, 2);
      gl.uniform3fv(u['uStreakColor'],    p.streakColor);
      gl.uniform1f(u['uStreakIntensity'], p.streakIntensity);
      gl.uniform1f(u['uGlowIntensity'],  p.glowIntensity);
      gl.uniform1f(u['uFlareIntensity'], p.flareIntensity);
      gl.uniform1f(u['uAspectCorrection'], p.aspectCorrection);
      gl.uniform1f(u['uHaloChroma'],     p.haloChroma);
      gl.uniform1f(u['uHaloScale'],      p.haloScale);
      gl.uniform1f(u['uRotateStreak'],   p.rotateStreak);
      gl.uniform1f(u['uHaloSoftness'],   p.haloSoftness);
      gl.uniform1f(u['uHaloRotateSrc'],  p.haloRotateSrc);
      gl.uniform1f(u['uHaloConstant'],   p.haloConstant);
      gl.uniform3fv(u['uHaloColor'],     p.haloColor);
      gl.uniform4fv(u['uHaloRing'],      p.haloRing);
      gl.uniform1i(u['uDebugHalo'],      p.debugHalo ? 1 : 0);
      gl.uniform2f(u['resolution'],      width, height);
      drawFullscreen(gl, prog.quadVAO);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Release all GPU resources. */
  destroy(): void {
    const gl = this.gl;
    for (const prog of [
      this.prefiltProg, this.downProg,
      this.upProg, this.compositeProg,
    ]) {
      gl.deleteProgram(prog.program);
      gl.deleteVertexArray(prog.quadVAO);
      gl.deleteBuffer(prog.quadBuf);
    }
  }

  /** Expose compiled GLSL sources for hot-reload / inspection. */
  get glslSources(): Readonly<Record<string, string>> {
    return {
      prefilter:        FRAG_STREAK_PREFILTER,
      down:             FRAG_STREAK_DOWN,
      up:               FRAG_STREAK_UP,
      compositeStreak:  FRAG_COMPOSITE_STREAK,
      transformUV:      GLSL_TRANSFORM_UV,
      luma:             GLSL_LUMA,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  WebGLResourceAccessor bridge
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Extension interface that the WebGL integration layer must implement on
 * the ResourceAccessor it passes to execute callbacks.
 *
 * The render-graph.ts (M822) ResourceAccessor is WebGPU-native; for WebGL
 * pipelines the same handle-based API is provided by the render-pipeline's
 * own bridge class that implements this interface alongside ResourceAccessor.
 *
 * Usage in pass callbacks:
 *   const tex = (accessor as unknown as WebGLResourceAccessor)
 *                 .getWebGLTexture(handle);
 */
export interface WebGLResourceAccessor {
  /**
   * Resolve a virtual ResourceHandle to its backing WebGLTexture.
   * Throws if the resource was not allocated for this frame.
   */
  getWebGLTexture(handle: ResourceHandle): WebGLTexture;

  /**
   * Resolve a virtual ResourceHandle to its backing WebGLFramebuffer.
   * Returns `null` for the swapchain surface.
   */
  getWebGLFBO(handle: ResourceHandle): WebGLFramebuffer | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Convenience factory — paired HydraBloom + LensStreak chain
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create both ATHydraBloomImport and ATHydraLensStreakImport in a single call.
 *
 * Convenience for callers that always register both post-process chains
 * (the common M851 usage pattern).
 *
 * @example
 * ```ts
 * const { bloom, streak } = await createATHydraChain(renderer);
 *
 * // Declare resources
 * const scene     = graph.createResource('scene',      { sizeClass: 'full' });
 * const bloomTex  = graph.createResource('bloom-tex',  { sizeClass: 'half' });
 * const bloomOut  = graph.createResource('bloom-out',  { sizeClass: 'full' });
 * const prefilt   = graph.createResource('prefilt',    { sizeClass: 'half' });
 * const streakDn  = graph.createResource('streak-dn',  { sizeClass: 'half' });
 * const streakUp  = graph.createResource('streak-up',  { sizeClass: 'half' });
 * const streakOut = graph.createResource('streak-out', { sizeClass: 'full' });
 *
 * // Register HydraBloom pass
 * graph.addPass('hydra-bloom', bloom.makeBloomPass(scene, bloomTex, bloomOut));
 *
 * // Register LensStreak passes
 * for (const [name, desc] of streak.makeStreakPasses(
 *   scene, prefilt, streakDn, streakUp, bloomOut, streakOut,
 * )) {
 *   graph.addPass(name, desc);
 * }
 *
 * graph.compile(width, height);
 * ```
 */
export async function createATHydraChain(
  renderer:      AstroRenderer,
  shaderUrl?:    string,
  streakParams?: ATHydraLensStreakParams,
): Promise<{
  bloom:  ATHydraBloomImport;
  streak: ATHydraLensStreakImport;
}> {
  const [bloom, streak] = await Promise.all([
    ATHydraBloomImport.create(renderer, shaderUrl),
    ATHydraLensStreakImport.create(renderer, shaderUrl, streakParams),
  ]);
  return { bloom, streak };
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Named GLSL exports — for external embedding / hot-reload
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Raw GLSL source strings extracted from compiled.vs, exported for
 * external embedding, shader inspection, or hot-reload workflows.
 *
 * Mirrors the `AT_BLOOM_WGSL` export pattern from at-bloom-postprocess.ts (M714).
 */
export const AT_HYDRA_GLSL = {
  // ── HydraBloom family ───────────────────────────────────────────────────
  /** HydraBloom.glsl — tHydraBloom sampler + getHydraBloom() helper. */
  hydraBloomLib:  GLSL_HYDRA_BLOOM_LIB,
  /** HydraBloomPass.glsl (fragment) — blendScreen(scene, bloom) composite. */
  hydraBloomPass: FRAG_HYDRA_BLOOM_PASS,

  // ── LensStreak family ───────────────────────────────────────────────────
  /** LensFlarePrefilter.glsl (fragment) — luma threshold + rotateUV. */
  streakPrefilter: FRAG_STREAK_PREFILTER,
  /** LensFlareDown.glsl (fragment) — 6-tap horizontal stretch blur. */
  streakDown:      FRAG_STREAK_DOWN,
  /** LensFlareUp.glsl (fragment) — 3-tap H + 2-tap V upsample. */
  streakUp:        FRAG_STREAK_UP,
  /** CompositeStreak.glsl (fragment) — streaks + halo SDF + chroma. */
  compositeStreak: FRAG_COMPOSITE_STREAK,

  // ── Shared utilities ────────────────────────────────────────────────────
  /** transformUV.glsl — rotateUV / scaleUV / skewUV helpers. */
  transformUV: GLSL_TRANSFORM_UV,
  /** luma.fs — BT.601 weighted luminance for vec3/vec4. */
  luma:        GLSL_LUMA,
  /** blendScreen from blendmodes.glsl. */
  blendScreen: GLSL_BLEND_SCREEN,
  /** Shared fullscreen-quad vertex shader. */
  vertFullscreen: VERT_FULLSCREEN,
} as const;
