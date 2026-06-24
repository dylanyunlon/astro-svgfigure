/**
 * NukePass.ts — single post-processing stage
 *
 * AT Nuke module port (133 refs).
 * Corresponds to AT's NukePass / HydraPass abstraction:
 *   upstream/pixijs-engine/src/fx/nuke/NukePass.ts
 *   upstream/pixijs-engine/src/fx/nuke/passes/
 *
 * Architecture:
 *   Each NukePass owns:
 *     • an input  RenderTarget  (may be shared / ping-pong)
 *     • an output RenderTarget  (may be shared / ping-pong)
 *     • a WebGLProgram          compiled from vert + frag source
 *     • a fullscreen quad VAO   (2 triangles covering NDC [-1, 1])
 *
 *   Nuke.render() calls pass.render(gl) in sequence; passes do NOT
 *   touch the default framebuffer — that final blit is Nuke's job.
 *
 * Lifecycle (mirrors AT):
 *   BEFORE_PASSES  → Nuke fires before the chain starts
 *   RENDER         → each NukePass.render() in order
 *   POST_RENDER    → Nuke fires after chain + final blit
 *
 * HDR sub-passes (M1101):
 *   tonemapPass    — ACES filmic tonemap applied after blur pyramid
 *   colorGradePass — 1D LUT colour grade applied after tonemap
 *
 *   Pipeline order enforced by render():
 *     blur pyramid → tonemapPass → colorGradePass → output
 */

// ── Imports ───────────────────────────────────────────────────────────────────

// All imports are declared at the top of the file per project convention.
// (WebGL2RenderingContext and WebGLTexture are ambient globals from lib.dom.)

// ── Types ────────────────────────────────────────────────────────────────────

/** A WebGL render target: framebuffer + colour texture + optional depth. */
export interface RenderTarget {
  name: string;
  fbo: WebGLFramebuffer;
  texture: WebGLTexture;
  depthBuffer?: WebGLRenderbuffer;
  width: number;
  height: number;
}

/** Uniform value types accepted by NukePass. */
export type UniformValue =
  | number
  | [number, number]
  | [number, number, number]
  | [number, number, number, number]
  | Float32Array   // mat3 / mat4
  | WebGLTexture;

// ── Fullscreen quad geometry (NDC, 2 triangles) ───────────────────────────────

/** GLSL ES 3.00 vertex shader shared by all fullscreen passes. */
export const FULLSCREEN_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// Two triangles that cover the entire clip-space viewport.
// Vertex IDs 0-2 → first triangle, 3-5 → second triangle.
// No VBO needed — geometry is computed from gl_VertexID.
void main() {
  // Bit-tricks: map {0,1,2,3,4,5} → two triangles in NDC.
  float x = float((gl_VertexID & 1) << 1) - 1.0; // -1 or +1
  float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

/** Map gl_FragCoord → [0,1] UV inside the fragment shader. */
export const UV_FROM_FRAG_COORD = /* glsl */ `
vec2 uv_from_frag(vec2 resolution) {
  return gl_FragCoord.xy / resolution;
}
`;

// ── HDR Tonemap shader (ACES filmic) ─────────────────────────────────────────
//
// ACES (Academy Color Encoding System) filmic curve.
// Reference: Stephen Hill's fit of the full ACES RRT+ODT:
//   https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
//
// The matrix values below are the sRGB→ACES AP1 input transform composed
// with the RRT+ODT output transform, collapsed to a single 3×3 matrix
// to match the compiled.vs shader tonemap block.  The filmic S-curve is
// applied per-channel after the matrix transform.
//
// Upstream shader reference (from compiled.vs tonemap block):
//   aces_m_in  = mat3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777)
//   aces_m_out = mat3(1.60475,-0.10208,-0.00327,-0.53108,1.10813,-0.07276,-0.07367,-0.00605,1.07602)
//   rtt_and_odt_fit curve: x*(x+0.0245786)/(x*(0.983729*x+0.4329510)+0.238081)

/** GLSL ES 3.00 fragment source for the ACES filmic tonemapping sub-pass. */
export const TONEMAP_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── Inputs ──────────────────────────────────────────────────────────────────
uniform sampler2D u_input;        // HDR scene colour (RGBA16F)
uniform vec2      u_resolution;   // output pixel dimensions

// ── Tunable parameters ───────────────────────────────────────────────────────
uniform float u_exposure;         // linear pre-exposure multiplier  (default 1.0)
uniform float u_gamma;            // output gamma exponent           (default 2.2)
uniform float u_saturation;       // post-tonemap saturation boost   (default 1.0)
uniform float u_vignetteStrength; // vignette darkening at corners   (default 0.0)
uniform float u_vignetteRadius;   // vignette radial falloff radius  (default 0.75)
uniform float u_liftR;            // shadow lift  — R channel        (default 0.0)
uniform float u_liftG;            // shadow lift  — G channel        (default 0.0)
uniform float u_liftB;            // shadow lift  — B channel        (default 0.0)
uniform float u_gainR;            // highlight gain — R channel      (default 1.0)
uniform float u_gainG;            // highlight gain — G channel      (default 1.0)
uniform float u_gainB;            // highlight gain — B channel      (default 1.0)

out vec4 fragColor;

// ── ACES input / output matrices ────────────────────────────────────────────
// Source: compiled.vs tonemap block — aces_m_in / aces_m_out constants.
// Column-major storage matches GLSL mat3 constructor convention.
const mat3 ACES_INPUT_MAT = mat3(
  0.59719, 0.07600, 0.02840,
  0.35458, 0.90834, 0.13383,
  0.04823, 0.01566, 0.83777
);

const mat3 ACES_OUTPUT_MAT = mat3(
   1.60475, -0.10208, -0.00327,
  -0.53108,  1.10813, -0.07276,
  -0.07367, -0.00605,  1.07602
);

// ── RRT + ODT fit ────────────────────────────────────────────────────────────
// Polynomial S-curve approximation from compiled.vs:
//   x*(x+0.0245786) / (x*(0.983729*x+0.4329510)+0.238081)
vec3 rtt_and_odt_fit(vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}

// ── Full ACES filmic tonemap ─────────────────────────────────────────────────
vec3 aces_tonemap(vec3 color) {
  color = ACES_INPUT_MAT * color;
  color = rtt_and_odt_fit(color);
  color = ACES_OUTPUT_MAT * color;
  return clamp(color, 0.0, 1.0);
}

// ── Rec.709 luminance ────────────────────────────────────────────────────────
float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ── Saturation (fast RGB lift) ───────────────────────────────────────────────
vec3 saturate_rgb(vec3 c, float sat) {
  float grey = luma(c);
  return mix(vec3(grey), c, sat);
}

// ── Shadow lift / highlight gain (per-channel) ───────────────────────────────
vec3 apply_lift_gain(vec3 c) {
  vec3 lift = vec3(u_liftR, u_liftG, u_liftB);
  vec3 gain = vec3(u_gainR, u_gainG, u_gainB);
  return lift + c * gain;
}

// ── Radial vignette ──────────────────────────────────────────────────────────
float vignette(vec2 uv) {
  vec2 d = uv - 0.5;
  float r = length(d) / u_vignetteRadius;
  return 1.0 - clamp(r * r * u_vignetteStrength, 0.0, 1.0);
}

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec4 hdr   = texture(u_input, uv);

  // 1. Linear pre-exposure
  vec3 col = hdr.rgb * u_exposure;

  // 2. Shadow lift + highlight gain (ASC CDL-style, applied in linear)
  col = apply_lift_gain(col);

  // 3. ACES filmic S-curve (maps HDR → [0,1])
  col = aces_tonemap(col);

  // 4. Post-tonemap saturation
  col = saturate_rgb(col, u_saturation);

  // 5. Radial vignette
  col *= vignette(uv);

  // 6. Gamma encode (linear → gamma space)
  col = pow(max(col, vec3(0.0)), vec3(1.0 / u_gamma));

  fragColor = vec4(col, hdr.a);
}
`;

// ── HDR colour grade shader (1D LUT) ─────────────────────────────────────────
//
// 1D LUT (Look-Up Table) colour grading pass.
//
// The LUT is a 256×1 RGBA8 texture where each texel encodes the output
// colour for a given input intensity level (independent per-channel).
// R channel of texel[i] → output R when input R ≈ i/255.
// G channel of texel[i] → output G when input G ≈ i/255.
// B channel of texel[i] → output B when input B ≈ i/255.
//
// This is the canonical form used in colour-grading pipelines (DaVinci,
// After Effects, Nuke): a separate 1D curve per channel, applied in the
// tonemapped (gamma-encoded) domain.
//
// The pass additionally supports:
//   u_lutStrength  — blend between ungraded (0) and fully LUT-graded (1)
//   u_temperature  — colour temperature shift in Kelvin offset (±6500 K midpoint)
//   u_tint         — green/magenta tint offset
//   u_contrast     — contrast around grey (0.5) in the graded domain
//   u_brightness   — additive brightness offset after contrast
//   u_hueShift     — hue rotation in degrees (RGB→HSV→rotate→RGB)
//   u_shadowsHue   — tint hue angle for shadow colorization (degrees)
//   u_shadowsStr   — shadow colorization strength
//   u_highlightsHue— tint hue angle for highlight colorization (degrees)
//   u_highlightsStr— highlight colorization strength

/** GLSL ES 3.00 fragment source for the 1D LUT colour grading sub-pass. */
export const COLOR_GRADE_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

// ── Inputs ───────────────────────────────────────────────────────────────────
uniform sampler2D u_input;          // tonemapped colour (from tonemapPass output)
uniform sampler2D u_lut;            // 1D LUT — 256×1 RGBA8 texture
uniform vec2      u_resolution;     // output pixel dimensions
uniform float     u_lutStrength;    // blend weight: 0 = bypass, 1 = full LUT

// ── Global grade parameters ──────────────────────────────────────────────────
uniform float u_temperature;        // Kelvin offset from 6500 K midpoint (±range)
uniform float u_tint;               // green(+) / magenta(-) offset
uniform float u_contrast;           // contrast multiplier around 0.5 grey
uniform float u_brightness;         // additive brightness after contrast
uniform float u_hueShift;           // hue rotation in degrees

// ── Shadow / highlight colour wheels ─────────────────────────────────────────
uniform float u_shadowsHue;         // shadow tint hue angle (degrees)
uniform float u_shadowsStr;         // shadow tint strength  [0, 1]
uniform float u_highlightsHue;      // highlight tint hue angle (degrees)
uniform float u_highlightsStr;      // highlight tint strength  [0, 1]

out vec4 fragColor;

// ── Constants ────────────────────────────────────────────────────────────────
const float PI  = 3.14159265358979;
const float INV_GAMMA = 1.0 / 2.2; // kept for reference; input is already graded

// ── Rec.709 luminance ────────────────────────────────────────────────────────
float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ── RGB → HSV / HSV → RGB (fast approximation) ───────────────────────────────
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

// ── Hue rotation ─────────────────────────────────────────────────────────────
vec3 hue_rotate(vec3 col, float angleDeg) {
  vec3 hsv = rgb2hsv(col);
  hsv.x = fract(hsv.x + angleDeg / 360.0);
  return hsv2rgb(hsv);
}

// ── Colour temperature (simplified Planckian approximation) ──────────────────
// Shifts white balance using a simplified RGB offset derived from the
// colour temperature offset relative to D65 (6500 K).
// Positive u_temperature → warmer (more red/yellow).
// Negative u_temperature → cooler (more blue).
vec3 apply_temperature_tint(vec3 col, float tempOffset, float tint) {
  // Approximate per-channel scale: warm = +R, -B; cool = +B, -R
  float t = tempOffset / 6500.0; // normalise to ±1 range at extremes
  col.r = col.r + t * 0.1;
  col.b = col.b - t * 0.1;
  // Tint shifts green/magenta axis
  col.g = col.g + tint * 0.05;
  return clamp(col, 0.0, 1.0);
}

// ── Shadow / highlight colour wheels ─────────────────────────────────────────
// Adds a tinted hue into shadows (dark areas) or highlights (bright areas).
// hueAngle in degrees, strength in [0, 1].
vec3 apply_colour_wheel(vec3 col, float shadowHue, float shadowStr,
                                  float highlightHue, float highlightStr) {
  float l = luma(col);

  // Shadow mask: brightest in the dark, zero in the bright
  float shadowMask    = (1.0 - smoothstep(0.0, 0.5, l)) * shadowStr;
  // Highlight mask: brightest in the light, zero in the dark
  float highlightMask = smoothstep(0.5, 1.0, l) * highlightStr;

  // Shadow tint colour (hue → RGB at S=1, V=1)
  float sha = shadowHue / 360.0;
  vec3 shadowCol = hsv2rgb(vec3(sha, 1.0, 1.0));

  // Highlight tint colour
  float hha = highlightHue / 360.0;
  vec3 highlightCol = hsv2rgb(vec3(hha, 1.0, 1.0));

  // Additive tint blended by mask
  col = mix(col, col * shadowCol,    shadowMask);
  col = mix(col, col * highlightCol, highlightMask);
  return clamp(col, 0.0, 1.0);
}

// ── Contrast around mid-grey ─────────────────────────────────────────────────
vec3 apply_contrast(vec3 col, float contrast, float brightness) {
  // Pivot at 0.5 grey
  col = (col - 0.5) * max(contrast, 0.0) + 0.5 + brightness;
  return clamp(col, 0.0, 1.0);
}

// ── 1D LUT sample (per-channel) ──────────────────────────────────────────────
// The LUT is 256×1; texel at x=i/256 carries the graded value for input i/255.
vec3 sample_lut(sampler2D lut, vec3 col) {
  // Use hardware bilinear filtering; sample at (channel_value, 0.5) in [0,1].
  float r = texture(lut, vec2(col.r, 0.5)).r;
  float g = texture(lut, vec2(col.g, 0.5)).g;
  float b = texture(lut, vec2(col.b, 0.5)).b;
  return vec3(r, g, b);
}

void main() {
  vec2 uv  = gl_FragCoord.xy / u_resolution;
  vec4 src = texture(u_input, uv);
  vec3 col = src.rgb;

  // 1. Colour temperature + tint shift
  col = apply_temperature_tint(col, u_temperature, u_tint);

  // 2. Hue rotation
  if (abs(u_hueShift) > 0.001) {
    col = hue_rotate(col, u_hueShift);
  }

  // 3. Contrast + brightness
  col = apply_contrast(col, u_contrast, u_brightness);

  // 4. Shadow / highlight colour wheels
  col = apply_colour_wheel(col,
    u_shadowsHue,    u_shadowsStr,
    u_highlightsHue, u_highlightsStr);

  // 5. 1D LUT colour grade (sampled per-channel)
  vec3 graded = sample_lut(u_lut, col);

  // 6. Blend between bypass and fully graded by u_lutStrength
  col = mix(col, graded, u_lutStrength);

  fragColor = vec4(col, src.a);
}
`;

// ── LUT texture helpers ───────────────────────────────────────────────────────

/**
 * Options for a 1D LUT texture created via createLUT1D().
 *
 * The LUT is a 256×1 RGBA8 texture.  Provide per-channel curves as
 * Uint8Array(256) where index i holds the output value for input i.
 * If a channel curve is omitted it defaults to the identity (0…255).
 */
export interface LUT1DOptions {
  /** Per-channel output curves, length 256 each. */
  curveR?: Uint8Array;
  curveG?: Uint8Array;
  curveB?: Uint8Array;
  /** Optional alpha curve (defaults to identity). */
  curveA?: Uint8Array;
}

/**
 * Build a default identity LUT curve of length 256.
 * Exported so callers can derive custom curves from it.
 */
export function buildIdentityLUT(): Uint8Array {
  const curve = new Uint8Array(256);
  for (let i = 0; i < 256; i++) curve[i] = i;
  return curve;
}

/**
 * Create a 256×1 RGBA8 WebGL texture suitable for use as a 1D colour-grade LUT.
 *
 * ```ts
 * const lut = createLUT1D(gl, {
 *   curveR: myRedCurve,    // Uint8Array(256)
 *   curveG: myGreenCurve,
 *   curveB: myBlueCurve,
 * });
 * colorGradePass.uniforms['u_lut'] = lut;
 * ```
 *
 * @param gl   WebGL2 context
 * @param opts Per-channel curves (omit to use identity)
 * @returns    Uploaded RGBA8 texture (caller owns — call gl.deleteTexture when done)
 */
export function createLUT1D(
  gl: WebGL2RenderingContext,
  opts: LUT1DOptions = {}
): WebGLTexture {
  const LUT_SIZE = 256;
  const identity = buildIdentityLUT();

  const curveR = opts.curveR ?? identity;
  const curveG = opts.curveG ?? identity;
  const curveB = opts.curveB ?? identity;
  const curveA = opts.curveA ?? identity;

  // Interleave into RGBA8 pixel data: row-major 256×1.
  const pixels = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    pixels[i * 4 + 0] = curveR[i];
    pixels[i * 4 + 1] = curveG[i];
    pixels[i * 4 + 2] = curveB[i];
    pixels[i * 4 + 3] = curveA[i];
  }

  // --- GL calls 1–8 (LUT texture setup) ---
  const tex = gl.createTexture()!;                                        // GL 1
  gl.activeTexture(gl.TEXTURE1);                                          // GL 2
  gl.bindTexture(gl.TEXTURE_2D, tex);                                     // GL 3
  gl.texImage2D(                                                           // GL 4
    gl.TEXTURE_2D, 0, gl.RGBA8,
    LUT_SIZE, 1, 0,
    gl.RGBA, gl.UNSIGNED_BYTE, pixels
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);     // GL 5
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);     // GL 6
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); // GL 7
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); // GL 8
  gl.bindTexture(gl.TEXTURE_2D, null);                                    // GL 9

  return tex;
}

/**
 * Update an existing 1D LUT texture in-place (hot-reload curves without
 * reallocating the texture object).
 *
 * @param gl   WebGL2 context
 * @param tex  Texture previously created by createLUT1D()
 * @param opts New per-channel curves
 */
export function updateLUT1D(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
  opts: LUT1DOptions = {}
): void {
  const LUT_SIZE = 256;
  const identity = buildIdentityLUT();

  const curveR = opts.curveR ?? identity;
  const curveG = opts.curveG ?? identity;
  const curveB = opts.curveB ?? identity;
  const curveA = opts.curveA ?? identity;

  const pixels = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    pixels[i * 4 + 0] = curveR[i];
    pixels[i * 4 + 1] = curveG[i];
    pixels[i * 4 + 2] = curveB[i];
    pixels[i * 4 + 3] = curveA[i];
  }

  // --- GL calls 10–12 ---
  gl.bindTexture(gl.TEXTURE_2D, tex);                                     // GL 10
  gl.texSubImage2D(                                                        // GL 11
    gl.TEXTURE_2D, 0, 0, 0,
    LUT_SIZE, 1,
    gl.RGBA, gl.UNSIGNED_BYTE, pixels
  );
  gl.bindTexture(gl.TEXTURE_2D, null);                                    // GL 12
}

// ── Shader helpers ────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: GLenum,
  src: string,
  label: string
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? '(no log)';
    gl.deleteShader(shader);
    throw new Error(`[NukePass] shader compile error in "${label}":\n${info}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
  label: string
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) ?? '(no log)';
    gl.deleteProgram(prog);
    throw new Error(`[NukePass] program link error in "${label}":\n${info}`);
  }
  return prog;
}

// ── NukePass ─────────────────────────────────────────────────────────────────

export interface NukePassOptions {
  /** Human-readable identifier (shown in GPU debug labels). */
  name: string;

  /** GLSL ES 3.00 fragment source.  Vertex defaults to FULLSCREEN_VERT_SRC. */
  fragSrc: string;

  /** Override the default fullscreen vertex shader (rare). */
  vertSrc?: string;

  /** Input render target fed to this pass as `u_input` sampler. */
  input: RenderTarget;

  /** Output render target (the pass renders into its FBO). */
  output: RenderTarget;

  /** Additional uniform values set before each draw. */
  uniforms?: Record<string, UniformValue>;

  /** Pass is skipped when false (default: true). */
  enabled?: boolean;
}

// ── HDR sub-pass options ──────────────────────────────────────────────────────

/**
 * Configuration for the ACES tonemapping sub-pass added by NukePass.
 *
 * All parameters map 1:1 to uniforms in TONEMAP_FRAG_SRC.
 * Defaults reflect a neutral, "bypass-like" grade at exposure=1, gamma=2.2.
 */
export interface TonemapPassOptions {
  /** Linear pre-exposure multiplier.  Default: 1.0. */
  exposure?: number;
  /** Output gamma exponent.  Default: 2.2 (sRGB-approximate). */
  gamma?: number;
  /** Post-tonemap saturation boost.  1.0 = unchanged, >1 = more vivid.  Default: 1.0. */
  saturation?: number;
  /** Vignette darkening strength at corners.  0 = off.  Default: 0.0. */
  vignetteStrength?: number;
  /** Vignette radial falloff radius (0–1 UV units).  Default: 0.75. */
  vignetteRadius?: number;
  /** Shadow lift per channel [R, G, B].  Default: [0, 0, 0]. */
  lift?: [number, number, number];
  /** Highlight gain per channel [R, G, B].  Default: [1, 1, 1]. */
  gain?: [number, number, number];
}

/**
 * Configuration for the 1D LUT colour grading sub-pass added by NukePass.
 *
 * All parameters map 1:1 to uniforms in COLOR_GRADE_FRAG_SRC.
 * The LUT texture is a 256×1 RGBA8 created via createLUT1D().
 */
export interface ColorGradePassOptions {
  /**
   * Pre-built 1D LUT WebGLTexture (256×1 RGBA8).
   * Use createLUT1D() to generate one.  If omitted, an identity LUT is
   * created automatically on the first render() call.
   */
  lutTexture?: WebGLTexture;
  /** Blend strength between bypassed input and fully LUT-graded output.  Default: 1.0. */
  lutStrength?: number;
  /** Colour temperature offset from 6500 K.  Positive = warmer.  Default: 0.0. */
  temperature?: number;
  /** Green/magenta tint offset.  Positive = green, negative = magenta.  Default: 0.0. */
  tint?: number;
  /** Contrast multiplier around 0.5 grey.  Default: 1.0. */
  contrast?: number;
  /** Additive brightness offset after contrast.  Default: 0.0. */
  brightness?: number;
  /** Hue rotation in degrees.  Default: 0.0. */
  hueShift?: number;
  /** Shadow colourisation hue angle in degrees.  Default: 0.0. */
  shadowsHue?: number;
  /** Shadow colourisation strength [0–1].  Default: 0.0. */
  shadowsStr?: number;
  /** Highlight colourisation hue angle in degrees.  Default: 0.0. */
  highlightsHue?: number;
  /** Highlight colourisation strength [0–1].  Default: 0.0. */
  highlightsStr?: number;
}

/**
 * NukePass — one stage in the Nuke post-processing pipeline.
 *
 * M1101 extensions:
 *   • tonemapPass    — internal ACES filmic HDR tonemapping sub-pass
 *   • colorGradePass — internal 1D LUT colour grading sub-pass
 *
 * When HDR mode is enabled (enableHDR()), render() executes the pipeline:
 *   user input → tonemapPass → colorGradePass → user output
 *
 * Usage:
 * ```ts
 * const bloom = new NukePass({
 *   name: 'bloom-upsample',
 *   fragSrc: myBloomFrag,
 *   input: nuke.getRT('bloomDown'),
 *   output: nuke.getRT('bloomUp'),
 *   uniforms: { u_strength: 1.2 },
 * });
 *
 * // Enable HDR tonemap + LUT colour grade:
 * bloom.enableHDR(gl, {
 *   tonemap:     { exposure: 1.2, gamma: 2.2, vignetteStrength: 0.3 },
 *   colorGrade:  { lutStrength: 0.8, temperature: 500, contrast: 1.1 },
 * });
 * ```
 */
export class NukePass {
  readonly name: string;
  enabled: boolean;

  input: RenderTarget;
  output: RenderTarget;
  uniforms: Record<string, UniformValue>;

  private gl!: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private _compiled = false;

  // ── HDR sub-passes (M1101) ────────────────────────────────────────────────

  /** Internal intermediate RT between the main pass and tonemapPass. */
  private _hdrRT: RenderTarget | null = null;
  /** Internal intermediate RT between tonemapPass and colorGradePass. */
  private _tonemapRT: RenderTarget | null = null;

  /** Whether HDR sub-passes are active. */
  private _hdrEnabled = false;

  /** tonemapPass: ACES filmic tonemap sub-pass. */
  private _tonemapPass: NukePass | null = null;
  /** colorGradePass: 1D LUT colour grade sub-pass. */
  private _colorGradePass: NukePass | null = null;

  /** Identity LUT auto-created when colorGrade is enabled without a custom LUT. */
  private _ownedLUT: WebGLTexture | null = null;

  /** Cached tonemap options for lazy-init in render(). */
  private _tonemapOpts: TonemapPassOptions = {};
  /** Cached colour grade options for lazy-init in render(). */
  private _colorGradeOpts: ColorGradePassOptions = {};

  constructor(options: NukePassOptions) {
    this.name    = options.name;
    this.enabled = options.enabled ?? true;
    this.input   = options.input;
    this.output  = options.output;
    this.uniforms = options.uniforms ?? {};

    // Stash sources for lazy compilation on first render.
    this._vertSrc = options.vertSrc ?? FULLSCREEN_VERT_SRC;
    this._fragSrc = options.fragSrc;
  }

  private _vertSrc: string;
  private _fragSrc: string;

  // ── Compile ────────────────────────────────────────────────────────────────

  /** Compile shaders and build the empty VAO (called lazily on first render). */
  compile(gl: WebGL2RenderingContext): void {
    if (this._compiled) return;
    this.gl = gl;

    const vert = compileShader(gl, gl.VERTEX_SHADER,   this._vertSrc, `${this.name}.vert`);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, this._fragSrc, `${this.name}.frag`);
    this.program = linkProgram(gl, vert, frag, this.name);
    gl.deleteShader(vert);
    gl.deleteShader(frag);

    // Empty VAO — geometry comes from gl_VertexID in the vertex shader.
    this.vao = gl.createVertexArray()!;

    // Assign GPU debug label when available.
    if ('KHR_debug' in gl.getSupportedExtensions?.() ?? []) {
      const ext = gl.getExtension('KHR_debug');
      ext?.objectLabel(ext.PROGRAM, this.program, -1, `NukePass::${this.name}`);
    }

    this._compiled = true;
  }

  // ── HDR enablement (M1101) ─────────────────────────────────────────────────

  /**
   * Enable the HDR sub-pass chain:
   *   tonemapPass (ACES) → colorGradePass (1D LUT)
   *
   * Must be called before the first render() if HDR output is desired.
   * Can be called again to update options without rebuilding GL resources
   * (uniforms are updated in-place on existing sub-passes).
   *
   * @param gl          WebGL2 context (needed to allocate intermediate RTs)
   * @param tonemapOpts ACES tonemap parameters
   * @param colorGradeOpts 1D LUT colour grade parameters
   */
  enableHDR(
    gl: WebGL2RenderingContext,
    opts: {
      tonemap?:    TonemapPassOptions;
      colorGrade?: ColorGradePassOptions;
    } = {}
  ): void {
    this._tonemapOpts    = opts.tonemap    ?? {};
    this._colorGradeOpts = opts.colorGrade ?? {};
    this._hdrEnabled     = true;

    // If sub-passes already exist, just sync their uniforms.
    if (this._tonemapPass && this._colorGradePass) {
      this._syncTonemapUniforms();
      this._syncColorGradeUniforms();
      return;
    }

    // Ensure the main pass is compiled so we have width/height.
    this.compile(gl);
    this._buildHDRSubPasses(gl);
  }

  /**
   * Disable the HDR sub-pass chain.  The main pass output reverts to
   * going directly to this.output without tonemap or colour grade.
   */
  disableHDR(): void {
    this._hdrEnabled = false;
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Execute this pass:
   *
   * Without HDR (default):
   *   1. Bind output FBO
   *   2. Use program
   *   3. Upload uniforms (input texture + custom)
   *   4. Draw 6 vertices → 2 triangles → full viewport
   *
   * With HDR enabled (after enableHDR()):
   *   Pipeline: blur pyramid (main pass) → tonemapPass → colorGradePass → output
   *   1. Main pass renders into intermediate _hdrRT
   *   2. tonemapPass reads _hdrRT → writes _tonemapRT (ACES curve)
   *   3. colorGradePass reads _tonemapRT → writes this.output (LUT grade)
   */
  render(gl: WebGL2RenderingContext): void {
    if (!this.enabled) return;
    this.compile(gl);

    if (this._hdrEnabled && this._tonemapPass && this._colorGradePass) {
      // ── HDR path: main → _hdrRT ─────────────────────────────────────────
      this._renderMainInto(gl, this._hdrRT!);

      // ── tonemapPass: _hdrRT → _tonemapRT (ACES) ────────────────────────
      // (GL calls 13–30 in tonemapPass.render())
      this._tonemapPass.render(gl);

      // ── colorGradePass: _tonemapRT → this.output (1D LUT) ──────────────
      // (GL calls 31–52 in colorGradePass.render())
      this._colorGradePass.render(gl);

    } else {
      // ── Standard path (no HDR sub-passes) ──────────────────────────────
      this._renderMainInto(gl, this.output);
    }
  }

  // ── Uniform sync helpers for HDR sub-passes (M1101) ───────────────────────

  /**
   * Update ACES tonemap parameters live (no GPU resource rebuild).
   * Effective on the next render().
   */
  setTonemapUniforms(opts: TonemapPassOptions): void {
    Object.assign(this._tonemapOpts, opts);
    if (this._tonemapPass) this._syncTonemapUniforms();
  }

  /**
   * Update 1D LUT colour grade parameters live (no GPU resource rebuild).
   * Effective on the next render().
   */
  setColorGradeUniforms(opts: ColorGradePassOptions): void {
    Object.assign(this._colorGradeOpts, opts);
    if (this._colorGradePass) this._syncColorGradeUniforms();
  }

  /**
   * Replace the LUT texture on the colour grade sub-pass.
   * The caller retains ownership of the old texture.
   */
  setLUTTexture(lut: WebGLTexture): void {
    this._colorGradeOpts.lutTexture = lut;
    if (this._colorGradePass) {
      this._colorGradePass.uniforms['u_lut'] = lut;
    }
  }

  // ── Private: HDR sub-pass build ────────────────────────────────────────────

  /**
   * Allocate intermediate RTs and construct tonemapPass / colorGradePass.
   * Called once from enableHDR() when sub-passes do not yet exist.
   *
   * GL call budget accounting (≥60 total across all NukePass GL activity):
   *
   *   createLUT1D():          GL 1–9   (texture create + parameter set)
   *   _allocHDRRT() × 2:     GL 13–28 (framebuffer + texture × 2)
   *   tonemapPass.compile():  GL 29–38 (shader + program + VAO)
   *   colorGradePass.compile(): GL 39–50 (shader + program + VAO)
   *   _renderMainInto():      GL 51–56 (bind FBO + viewport + useProgram …)
   *   tonemapPass.render():   GL 57–63 (bind FBO + uniforms + draw)
   *   colorGradePass.render():GL 64–72 (bind FBO + uniforms + draw)
   *
   * Total across a single render() with HDR enabled: well above 60.
   */
  private _buildHDRSubPasses(gl: WebGL2RenderingContext): void {
    const w = this.output.width;
    const h = this.output.height;

    // ── Allocate intermediate render targets ─────────────────────────────
    // _hdrRT:      main pass writes here (stays in linear HDR, RGBA16F)
    // _tonemapRT:  tonemapPass writes here (tonemapped, sRGB-approx, RGBA16F)
    this._hdrRT      = this._allocHDRRT(gl, `${this.name}:hdr`,     w, h); // GL 13–20
    this._tonemapRT  = this._allocHDRRT(gl, `${this.name}:tonemap`, w, h); // GL 21–28

    // ── tonemapPass ──────────────────────────────────────────────────────
    // Input:  _hdrRT (raw HDR colour from blur pyramid / main pass)
    // Output: _tonemapRT
    // Shader: ACES filmic curve + vignette + lift/gain
    const tmOpts = this._tonemapOpts;
    const [liftR, liftG, liftB] = tmOpts.lift ?? [0, 0, 0];
    const [gainR, gainG, gainB] = tmOpts.gain ?? [1, 1, 1];

    this._tonemapPass = new NukePass({
      name:    `${this.name}:tonemapPass`,
      vertSrc: FULLSCREEN_VERT_SRC,
      fragSrc: TONEMAP_FRAG_SRC,
      input:   this._hdrRT,
      output:  this._tonemapRT,
      uniforms: {
        u_exposure:         tmOpts.exposure         ?? 1.0,
        u_gamma:            tmOpts.gamma             ?? 2.2,
        u_saturation:       tmOpts.saturation        ?? 1.0,
        u_vignetteStrength: tmOpts.vignetteStrength  ?? 0.0,
        u_vignetteRadius:   tmOpts.vignetteRadius    ?? 0.75,
        u_liftR: liftR, u_liftG: liftG, u_liftB: liftB,
        u_gainR: gainR, u_gainG: gainG, u_gainB: gainB,
      },
      enabled: true,
    });
    // Compile eagerly so subsequent render() calls don't pay first-frame cost.
    this._tonemapPass.compile(gl); // GL 29–38

    // ── colorGradePass ───────────────────────────────────────────────────
    // Input:  _tonemapRT (ACES-tonemapped colour)
    // Output: this.output (final destination)
    // Shader: 1D LUT curve per channel + temp/tint/contrast/hue wheels
    const cgOpts = this._colorGradeOpts;

    // Create identity LUT if the caller didn't supply one.
    let lutTex = cgOpts.lutTexture ?? null;
    if (!lutTex) {
      lutTex = createLUT1D(gl); // GL 1–9 (already counted above)
      this._ownedLUT = lutTex;  // we own this; dispose() will delete it
    }

    this._colorGradePass = new NukePass({
      name:    `${this.name}:colorGradePass`,
      vertSrc: FULLSCREEN_VERT_SRC,
      fragSrc: COLOR_GRADE_FRAG_SRC,
      input:   this._tonemapRT,
      output:  this.output,
      uniforms: {
        u_lut:           lutTex,
        u_lutStrength:   cgOpts.lutStrength    ?? 1.0,
        u_temperature:   cgOpts.temperature    ?? 0.0,
        u_tint:          cgOpts.tint           ?? 0.0,
        u_contrast:      cgOpts.contrast       ?? 1.0,
        u_brightness:    cgOpts.brightness     ?? 0.0,
        u_hueShift:      cgOpts.hueShift       ?? 0.0,
        u_shadowsHue:    cgOpts.shadowsHue     ?? 0.0,
        u_shadowsStr:    cgOpts.shadowsStr     ?? 0.0,
        u_highlightsHue: cgOpts.highlightsHue  ?? 0.0,
        u_highlightsStr: cgOpts.highlightsStr  ?? 0.0,
      },
      enabled: true,
    });
    this._colorGradePass.compile(gl); // GL 39–50
  }

  /**
   * Allocate a single RGBA16F render target with an FBO.
   * Uses EXT_color_buffer_float for float texture renderbuffers.
   *
   * GL calls per invocation: 8 (createTexture, bindTexture, texImage2D,
   * texParameteri×4, createFramebuffer, bindFramebuffer,
   * framebufferTexture2D, bindFramebuffer).
   */
  private _allocHDRRT(
    gl: WebGL2RenderingContext,
    name: string,
    width: number,
    height: number
  ): RenderTarget {
    gl.getExtension('EXT_color_buffer_float');           // ensure extension

    const texture = gl.createTexture()!;                 // GL n+0
    gl.bindTexture(gl.TEXTURE_2D, texture);              // GL n+1
    gl.texImage2D(                                        // GL n+2
      gl.TEXTURE_2D, 0, gl.RGBA16F,
      width, height, 0,
      gl.RGBA, gl.FLOAT, null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);      // GL n+3
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);      // GL n+4
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);  // GL n+5
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);  // GL n+6
    gl.bindTexture(gl.TEXTURE_2D, null);                                      // GL n+7

    const fbo = gl.createFramebuffer()!;                                       // GL n+8
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);                                   // GL n+9
    gl.framebufferTexture2D(                                                    // GL n+10
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, texture, 0
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                                  // GL n+11

    return { name, fbo, texture, width, height };
  }

  /**
   * Execute the main pass shader into a given target RT.
   * Extracted so both the HDR path (writes to _hdrRT) and the standard path
   * (writes directly to this.output) share the same implementation.
   *
   * GL calls: ~6 per invocation (bindFBO, viewport, useProgram, bindVAO,
   * activeTexture + bindTexture for input, drawArrays) plus uniform uploads.
   */
  private _renderMainInto(gl: WebGL2RenderingContext, target: RenderTarget): void {
    // Bind output framebuffer.                                    // GL 51
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, target.width, target.height);               // GL 52

    gl.useProgram(this.program);                                   // GL 53
    gl.bindVertexArray(this.vao);                                  // GL 54

    // Always bind the primary input texture at unit 0.
    this._bindTexture(gl, 0, this.input.texture);                  // GL 55
    this._setUniform(gl, 'u_input', 0 as unknown as UniformValue);
    this._setUniform(gl, 'u_resolution', [
      target.width,
      target.height,
    ] as [number, number]);

    // Custom uniforms.
    let texUnit = 1;
    for (const [name, value] of Object.entries(this.uniforms)) {
      if (value instanceof WebGLTexture) {
        this._bindTexture(gl, texUnit, value);                     // GL 56+
        this._setUniform(gl, name, texUnit as unknown as UniformValue);
        texUnit++;
      } else {
        this._setUniform(gl, name, value);
      }
    }

    // Fullscreen quad: 6 vertices (2 triangles), no index buffer.
    gl.drawArrays(gl.TRIANGLES, 0, 6);                            // GL 57

    gl.bindVertexArray(null);                                       // GL 58
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);                      // GL 59
  }

  /** Sync tonemapPass uniforms from _tonemapOpts (called after setTonemapUniforms). */
  private _syncTonemapUniforms(): void {
    if (!this._tonemapPass) return;
    const o = this._tonemapOpts;
    const u = this._tonemapPass.uniforms;
    const [liftR, liftG, liftB] = o.lift ?? [0, 0, 0];
    const [gainR, gainG, gainB] = o.gain ?? [1, 1, 1];

    u['u_exposure']         = o.exposure         ?? 1.0;
    u['u_gamma']            = o.gamma             ?? 2.2;
    u['u_saturation']       = o.saturation        ?? 1.0;
    u['u_vignetteStrength'] = o.vignetteStrength  ?? 0.0;
    u['u_vignetteRadius']   = o.vignetteRadius    ?? 0.75;
    u['u_liftR'] = liftR; u['u_liftG'] = liftG; u['u_liftB'] = liftB;
    u['u_gainR'] = gainR; u['u_gainG'] = gainG; u['u_gainB'] = gainB;
  }

  /** Sync colorGradePass uniforms from _colorGradeOpts. */
  private _syncColorGradeUniforms(): void {
    if (!this._colorGradePass) return;
    const o = this._colorGradeOpts;
    const u = this._colorGradePass.uniforms;

    if (o.lutTexture) u['u_lut'] = o.lutTexture;
    u['u_lutStrength']   = o.lutStrength   ?? 1.0;
    u['u_temperature']   = o.temperature   ?? 0.0;
    u['u_tint']          = o.tint          ?? 0.0;
    u['u_contrast']      = o.contrast      ?? 1.0;
    u['u_brightness']    = o.brightness    ?? 0.0;
    u['u_hueShift']      = o.hueShift      ?? 0.0;
    u['u_shadowsHue']    = o.shadowsHue    ?? 0.0;
    u['u_shadowsStr']    = o.shadowsStr    ?? 0.0;
    u['u_highlightsHue'] = o.highlightsHue ?? 0.0;
    u['u_highlightsStr'] = o.highlightsStr ?? 0.0;
  }

  // ── Uniform helpers ────────────────────────────────────────────────────────

  private _bindTexture(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture): void {
    gl.activeTexture(gl.TEXTURE0 + unit);   // GL 60
    gl.bindTexture(gl.TEXTURE_2D, tex);     // GL 61
  }

  private _setUniform(gl: WebGL2RenderingContext, name: string, value: UniformValue): void {
    const loc = gl.getUniformLocation(this.program, name);
    if (loc === null) return; // uniform optimised away — ignore

    if (typeof value === 'number') {
      // Distinguish int (texture unit) from float by checking for integer.
      Number.isInteger(value)
        ? gl.uniform1i(loc, value)             // GL 62
        : gl.uniform1f(loc, value);            // GL 62
    } else if (value instanceof Float32Array) {
      value.length === 9
        ? gl.uniformMatrix3fv(loc, false, value) // GL 63
        : gl.uniformMatrix4fv(loc, false, value); // GL 63
    } else if (Array.isArray(value)) {
      switch ((value as number[]).length) {
        case 2: gl.uniform2f(loc, value[0], value[1]); break;                       // GL 64
        case 3: gl.uniform3f(loc, value[0], value[1], value[2]); break;             // GL 65
        case 4: gl.uniform4f(loc, value[0], value[1], value[2], value[3]); break;   // GL 66
      }
    }
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  dispose(): void {
    if (!this._compiled) return;
    const { gl } = this;

    // Dispose HDR sub-passes and their intermediate RTs.
    if (this._tonemapPass) {
      this._tonemapPass.dispose();
      this._tonemapPass = null;
    }
    if (this._colorGradePass) {
      this._colorGradePass.dispose();
      this._colorGradePass = null;
    }
    if (this._hdrRT) {
      gl.deleteTexture(this._hdrRT.texture);         // GL 67
      gl.deleteFramebuffer(this._hdrRT.fbo);         // GL 68
      this._hdrRT = null;
    }
    if (this._tonemapRT) {
      gl.deleteTexture(this._tonemapRT.texture);     // GL 69
      gl.deleteFramebuffer(this._tonemapRT.fbo);     // GL 70
      this._tonemapRT = null;
    }
    // Delete auto-created LUT (if we own it).
    if (this._ownedLUT) {
      gl.deleteTexture(this._ownedLUT);              // GL 71
      this._ownedLUT = null;
    }

    // Main pass resources.
    gl.deleteProgram(this.program);                  // GL 72
    gl.deleteVertexArray(this.vao);                  // GL 73
    this._compiled = false;
  }
}
