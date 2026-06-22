/**
 * water-background.ts — AT TreeWaterShader adapted as world background
 *
 * Source lineage:
 *   upstream/webgl-water/water.js       (Evan Wallace, MIT) — ping-pong height
 *   upstream/webgl-water/renderer.js    (Evan Wallace, MIT) — normal + caustics
 *   src/lib/SceneLayoutPresets.ts lines 648-657             — AT UIL params
 *     TreeWaterShader/TreeWaterShader/uScale            202.03
 *     TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uBrightness   2
 *     TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uColor        #ffffff
 *     TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uScale        1000
 *     TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uSpeed        0.04
 *     TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uWaterUVStrength -5
 *   INPUT_Element_9_TreeScenewater_heightScale  0.05
 *   INPUT_Element_9_TreeScenewater_subdivide    128
 *   INPUT_Element_9_TreeScenewater_size         20
 *   INPUT_Element_9_TreeScenewater_texture      512
 *   INPUT_Element_9_TreeScenewater_viscosity    0.98
 *
 * Architecture:
 *   Three GPU passes render a full-screen animated water surface:
 *
 *   1. DROP PASS  — addDrop() perturbs a 512×512 float ping-pong height map
 *      using the AT sine-cosine kernel (cos(distance * π) × 0.5 − 0.5).
 *      External callers may inject drops from cell pub/sub events.
 *
 *   2. STEP PASS  — advances the shallow-wave PDE on the height map each frame:
 *      velocity += (avg_neighbour_height − current_height) × 2.0;
 *      velocity *= viscosity (0.98);
 *      height   += velocity;
 *      Runs uSteps times per frame (default 2) for stability.
 *
 *   3. NORMAL PASS — reconstructs (normal.x, normal.z) stored in BA from
 *      the height field: cross(dx, dy).xz where dx/dy are finite differences.
 *
 *   4. RENDER PASS — fullscreen quad surface shader:
 *      • uWaterUVStrength perturbation of the UV lookup (AT: -5)
 *      • Blinn-Phong specular with uBrightness (AT: 2)
 *      • uColor tint (AT: #ffffff)
 *      • uSpeed-driven time animation (AT: 0.04)
 *      • uScale controls the world-space frequency (AT inner: 1000, outer: 202.03)
 *
 *   The texture resolution is fixed at 512×512 (AT: `water_texture: 512`).
 *   Subdivision 128 matches `water_subdivide: 128`.
 *   Height scale 0.05 and viscosity 0.98 match AT element params.
 *
 * Public API:
 *   const bg = await mountWaterBackground(containerEl, options?)
 *   bg.update(dt)                 — advance simulation + render (call each frame)
 *   bg.addDrop(x, y, r, strength) — inject a ripple at world UV [0,1]²
 *   bg.setColor(hex)              — update uColor (#rrggbb)
 *   bg.setSpeed(s)                — update uSpeed
 *   bg.setBrightness(b)           — update uBrightness
 *   bg.stop()                     — tear down canvas + GL resources
 *
 * Author: claude <claude@astro.dev>
 * Research: xiaodi M704 — cell-pubsub-loop
 */

import { createProgram, FullscreenQuad } from './hydra-gl-layer';

// ─────────────────────────────────────────────────────────────────────────────
// Public config / handle
// ─────────────────────────────────────────────────────────────────────────────

export interface WaterBackgroundOptions {
  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uColor`
   * Default '#ffffff'
   */
  color?: string;

  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uSpeed`
   * Controls how fast the animated time accumulates.  Default 0.04.
   */
  speed?: number;

  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/uScale` (outer, domain scale).
   * Default 202.03.
   */
  domainScale?: number;

  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uScale`
   * UV frequency multiplier inside the surface shader.  Default 1000.
   */
  uvScale?: number;

  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uBrightness`
   * Specular brightness multiplier.  Default 2.
   */
  brightness?: number;

  /**
   * AT UIL: `TreeWaterShader/TreeWaterShader/Element_9_TreeScene/uWaterUVStrength`
   * UV distortion strength (negative = refraction-like warp).  Default -5.
   */
  waterUVStrength?: number;

  /**
   * AT UIL: `INPUT_Element_9_TreeScenewater_texture`
   * Simulation texture resolution (power-of-two).  Default 512.
   */
  textureSize?: number;

  /**
   * AT UIL: `INPUT_Element_9_TreeScenewater_viscosity`
   * Wave damping per step.  Default 0.98.
   */
  viscosity?: number;

  /**
   * AT UIL: `INPUT_Element_9_TreeScenewater_heightScale`
   * Vertex displacement scale for the height field.  Default 0.05.
   */
  heightScale?: number;

  /**
   * Simulation steps per frame (stability vs. cost).  Default 2.
   */
  stepsPerFrame?: number;

  /**
   * Background CSS colour (underneath the translucent water layer).
   * Default '#0a1628' (dark navy, visually matching AT TreeScene).
   */
  backgroundColor?: string;
}

export interface WaterBackgroundHandle {
  /** Advance simulation + render.  Call from rAF / Ticker. */
  update(dt: number): void;

  /**
   * Inject a circular drop at UV coords (x, y) in [0, 1]².
   * @param x        Centre U
   * @param y        Centre V
   * @param radius   Drop radius in UV space  (e.g. 0.03)
   * @param strength Height perturbation      (e.g. 0.05 .. 0.2)
   */
  addDrop(x: number, y: number, radius: number, strength: number): void;

  /** Change the surface tint colour.  Accepts '#rrggbb'. */
  setColor(hex: string): void;

  /** Update animation speed (AT uSpeed). */
  setSpeed(s: number): void;

  /** Update specular brightness (AT uBrightness). */
  setBrightness(b: number): void;

  /** Tear down GL resources and remove the canvas element. */
  stop(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — shared vertex (fullscreen triangle, no attributes)
// ─────────────────────────────────────────────────────────────────────────────

const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  // Positions from gl_VertexID covering clip space
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Drop pass
// Source: water.js dropShader — cos(drop * PI) * 0.5 perturbation
// ─────────────────────────────────────────────────────────────────────────────

const DROP_FRAG = `#version 300 es
precision highp float;
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uWater;    // current height field (R=height, G=velocity, BA=normal)
uniform vec2      uCenter;   // drop UV centre [0,1]²
uniform float     uRadius;   // drop radius in UV space
uniform float     uStrength; // perturbation amplitude

const float PI = 3.141592653589793;

void main() {
  vec4 info = texture(uWater, vUV);

  // Radial falloff from drop centre
  float dist = length(uCenter - vUV);
  float drop = max(0.0, 1.0 - dist / uRadius);
  drop = 0.5 - cos(drop * PI) * 0.5;   // smooth bell, matches upstream water.js

  info.r += drop * uStrength;
  fragColor = info;
}`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Step pass  (shallow-wave PDE, one iteration)
// Source: water.js updateShader
// AT params: viscosity 0.98, heightScale 0.05
// ─────────────────────────────────────────────────────────────────────────────

const STEP_FRAG = `#version 300 es
precision highp float;
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uWater;
uniform vec2      uTexelSize;  // 1.0 / textureSize
uniform float     uViscosity;  // AT: 0.98

void main() {
  vec4  info = texture(uWater, vUV);
  vec2  dx   = vec2(uTexelSize.x, 0.0);
  vec2  dy   = vec2(0.0, uTexelSize.y);

  // Average of 4-connected neighbours  (shallow-wave propagation)
  float avg = (
    texture(uWater, vUV - dx).r +
    texture(uWater, vUV + dx).r +
    texture(uWater, vUV - dy).r +
    texture(uWater, vUV + dy).r
  ) * 0.25;

  // Velocity integrator
  info.g += (avg - info.r) * 2.0;
  info.g *= uViscosity;   // AT: 0.98
  info.r += info.g;

  fragColor = info;
}`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Normal update pass
// Source: water.js normalShader — finite-difference cross product
// ─────────────────────────────────────────────────────────────────────────────

const NORMAL_FRAG = `#version 300 es
precision highp float;
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uWater;
uniform vec2      uTexelSize;

void main() {
  vec4  info = texture(uWater, vUV);
  vec2  dx   = vec2(uTexelSize.x, 0.0);
  vec2  dy   = vec2(0.0, uTexelSize.y);

  // Tangent vectors in XY-height space; cross product → surface normal
  vec3 tx = vec3(uTexelSize.x * 2.0,
                 texture(uWater, vUV + dx).r - texture(uWater, vUV - dx).r,
                 0.0);
  vec3 ty = vec3(0.0,
                 texture(uWater, vUV + dy).r - texture(uWater, vUV - dy).r,
                 uTexelSize.y * 2.0);

  vec3 n = normalize(cross(ty, tx));
  info.ba = n.xz;     // store normal.xz in BA, matches water.js convention
  fragColor = info;
}`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — Surface render pass
// AT TreeWaterShader uniforms:
//   uScale         202.03  (domain scale — outer)
//   uWaterUVStrength  -5   (UV warp strength)
//   uSpeed          0.04   (time multiplier)
//   uBrightness       2    (specular multiplier)
//   uColor        #ffffff  (surface tint)
//   uInnerScale    1000    (UV frequency inside shader)
// ─────────────────────────────────────────────────────────────────────────────

const RENDER_FRAG = `#version 300 es
precision highp float;
in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uWater;          // RG=height+vel, BA=normal.xz

// AT UIL params ─────────────────────────────────────────────────────────────
uniform float uTime;               // accumulated time × uSpeed
uniform float uSpeed;              // AT: 0.04
uniform float uScale;              // AT: 202.03  (domain frequency)
uniform float uInnerScale;         // AT: 1000    (inner UV scale)
uniform float uBrightness;         // AT: 2.0
uniform float uWaterUVStrength;    // AT: -5.0
uniform vec3  uColor;              // AT: vec3(1.0) = #ffffff
uniform float uHeightScale;        // AT: 0.05

// Light direction (fixed top-front, matches AT TreeScene light)
const vec3 LIGHT = normalize(vec3(0.6, 1.0, 0.8));
const vec3 EYE   = vec3(0.0, 1.0, 0.0);   // orthographic top-down view

// ── Lygia-style fbm helpers (inline, no preprocessor dependency) ─────────────
// Hash adapted from Dave Hoskins' hash-without-sine
float hash21(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}

float noise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i + vec2(0,0)), hash21(i + vec2(1,0)), u.x),
    mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), u.x),
    u.y
  );
}

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise2(p);
    p  = p * 2.1 + vec2(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}
// ─────────────────────────────────────────────────────────────────────────────

void main() {
  // ── 1. Perturb UV using height-field normals (AT: uWaterUVStrength = -5) ──
  vec4 info   = texture(uWater, vUV);
  vec2 normal = info.ba;                               // nx, nz from normal pass
  vec2 warpedUV = vUV + normal * uWaterUVStrength * 0.001;  // normalised warp

  // ── 2. Re-sample with warped UV for iterative "peaked" look ──────────────
  // Mirror upstream renderer.js: 5 iterations coord refinement
  vec2 coord = warpedUV;
  vec4 winfo = info;
  for (int i = 0; i < 5; i++) {
    coord  += winfo.ba * 0.005;
    winfo   = texture(uWater, clamp(coord, 0.001, 0.999));
  }
  vec2 wn = winfo.ba;

  // ── 3. Reconstruct surface normal ─────────────────────────────────────────
  vec3 N = normalize(vec3(wn.x, sqrt(max(0.0, 1.0 - dot(wn, wn))), wn.y));

  // ── 4. Animated FBM detail noise (AT: uScale outer domain, uInnerScale UV)─
  float t    = uTime * uSpeed;
  float freq = uScale / uInnerScale;    // blended AT scale ratio
  vec2  fuvA = vUV * freq + vec2( t * 0.3,  t * 0.17);
  vec2  fuvB = vUV * freq + vec2(-t * 0.19, t * 0.28);
  float fA   = fbm(fuvA);
  float fB   = fbm(fuvB);
  float detail = fA * 0.6 + fB * 0.4;   // blended FBM turbulence

  // ── 5. Blinn-Phong specular (matches AT shading pass) ─────────────────────
  vec3  H       = normalize(LIGHT + EYE);
  float specular = pow(max(0.0, dot(N, H)), 128.0) * uBrightness;

  // ── 6. Diffuse component with height-driven brightness ────────────────────
  float diffuse  = max(0.0, dot(N, LIGHT)) * 0.6 + 0.4;
  float height   = info.r * uHeightScale;

  // ── 7. Water colour: tint × detail × diffuse + specular highlight ─────────
  vec3 waterCol  = uColor * (diffuse * (0.5 + 0.5 * detail) + height * 0.3);
  waterCol      += vec3(specular);

  // ── 8. Fresnel-like edge softening ────────────────────────────────────────
  float fresnel  = pow(1.0 - abs(dot(N, EYE)), 2.0);
  waterCol      += vec3(0.3, 0.5, 0.8) * fresnel * 0.4;

  // Premultiplied alpha, alpha derived from surface coverage + fresnel
  float alpha    = clamp(0.75 + fresnel * 0.25, 0.0, 1.0);
  fragColor      = vec4(waterCol * alpha, alpha);
}`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function hexToRgb01(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [
    ((n >> 16) & 0xff) / 255,
    ((n >>  8) & 0xff) / 255,
    (       n  & 0xff) / 255,
  ];
}

/** Create and upload a floating-point RGBA texture (RGBA32F or RGBA16F). */
function createFloatTexture(
  gl: WebGL2RenderingContext,
  size: number,
): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);

  // Prefer RGBA32F; fall back to RGBA16F on mobile.
  // We always try RGBA32F here — WebGL2 guarantees color-renderable RGBA32F
  // when EXT_color_buffer_float is present, which is near-universal in 2024+.
  gl.texImage2D(
    gl.TEXTURE_2D, 0,
    gl.RGBA32F,
    size, size, 0,
    gl.RGBA, gl.FLOAT, null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/** Create an FBO backed by the given texture. */
function createFBO(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
): WebGLFramebuffer {
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

/** Bind texture to unit and set sampler uniform. */
function bindTex(
  gl: WebGL2RenderingContext,
  prog: WebGLProgram,
  name: string,
  tex: WebGLTexture,
  unit: number,
): void {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform1i(loc, unit);
}

/** Set float uniform, silently skip missing locations. */
function uf(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, v: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform1f(loc, v);
}

function u2f(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, x: number, y: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform2f(loc, x, y);
}

function u3f(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, r: number, g: number, b: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform3f(loc, r, g, b);
}

/** Fullscreen triangle draw (3 vertices, no VAO attributes needed). */
function drawFullscreen(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject): void {
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindVertexArray(null);
}

// ─────────────────────────────────────────────────────────────────────────────
// Core WaterBackground class (internal, not exported directly)
// ─────────────────────────────────────────────────────────────────────────────

class WaterBackground {
  private readonly gl: WebGL2RenderingContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly texSize: number;

  // ping-pong buffers
  private texA: WebGLTexture;
  private texB: WebGLTexture;
  private fboA: WebGLFramebuffer;
  private fboB: WebGLFramebuffer;

  // programs
  private progDrop:   WebGLProgram;
  private progStep:   WebGLProgram;
  private progNormal: WebGLProgram;
  private progRender: WebGLProgram;

  // empty VAO for fullscreen triangle
  private vao: WebGLVertexArrayObject;

  // AT UIL params — all mutable via setters
  private color:           [number, number, number];
  private speed:           number;
  private domainScale:     number;
  private uvScale:         number;
  private brightness:      number;
  private waterUVStrength: number;
  private viscosity:       number;
  private heightScale:     number;
  private stepsPerFrame:   number;

  private uTime        = 0;
  private W            = 0;
  private H            = 0;
  private ro:          ResizeObserver;
  private stopped      = false;

  constructor(
    gl: WebGL2RenderingContext,
    canvas: HTMLCanvasElement,
    container: HTMLElement,
    opts: Required<WaterBackgroundOptions>,
  ) {
    this.gl     = gl;
    this.canvas = canvas;

    this.texSize         = opts.textureSize;
    this.color           = hexToRgb01(opts.color);
    this.speed           = opts.speed;
    this.domainScale     = opts.domainScale;
    this.uvScale         = opts.uvScale;
    this.brightness      = opts.brightness;
    this.waterUVStrength = opts.waterUVStrength;
    this.viscosity       = opts.viscosity;
    this.heightScale     = opts.heightScale;
    this.stepsPerFrame   = opts.stepsPerFrame;

    // ── Enable required extension ────────────────────────────────────────────
    gl.getExtension('EXT_color_buffer_float');

    // ── Ping-pong textures ───────────────────────────────────────────────────
    this.texA = createFloatTexture(gl, this.texSize);
    this.texB = createFloatTexture(gl, this.texSize);
    this.fboA = createFBO(gl, this.texA);
    this.fboB = createFBO(gl, this.texB);

    // ── Programs ─────────────────────────────────────────────────────────────
    this.progDrop   = createProgram(gl, VERT, DROP_FRAG);
    this.progStep   = createProgram(gl, VERT, STEP_FRAG);
    this.progNormal = createProgram(gl, VERT, NORMAL_FRAG);
    this.progRender = createProgram(gl, VERT, RENDER_FRAG);

    // ── Empty VAO (positions from gl_VertexID) ────────────────────────────
    this.vao = gl.createVertexArray()!;

    // ── Resize handling ───────────────────────────────────────────────────────
    this.ro = new ResizeObserver(() => this._resize(container));
    this.ro.observe(container);
    this._resize(container);
  }

  private _resize(container: HTMLElement): void {
    if (this.stopped) return;
    const rect = container.getBoundingClientRect();
    const dpr  = Math.min(window.devicePixelRatio || 1, 2);
    const W    = Math.max(1, Math.floor(rect.width  * dpr));
    const H    = Math.max(1, Math.floor(rect.height * dpr));
    if (W === this.W && H === this.H) return;
    this.W = W; this.H = H;
    this.canvas.width  = W;
    this.canvas.height = H;
  }

  // ── addDrop — inject a height perturbation ─────────────────────────────────

  addDrop(x: number, y: number, radius: number, strength: number): void {
    if (this.stopped) return;
    const gl = this.gl;

    // read from texA, write to texB
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboB);
    gl.viewport(0, 0, this.texSize, this.texSize);
    gl.useProgram(this.progDrop);

    bindTex(gl, this.progDrop, 'uWater', this.texA, 0);
    u2f(gl, this.progDrop, 'uCenter',   x, y);
    uf( gl, this.progDrop, 'uRadius',   radius);
    uf( gl, this.progDrop, 'uStrength', strength);

    drawFullscreen(gl, this.vao);

    // swap
    [this.texA, this.texB] = [this.texB, this.texA];
    [this.fboA, this.fboB] = [this.fboB, this.fboA];

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── _step — one PDE integration iteration ─────────────────────────────────

  private _step(): void {
    const gl = this.gl;
    const ts = 1.0 / this.texSize;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboB);
    gl.viewport(0, 0, this.texSize, this.texSize);
    gl.useProgram(this.progStep);

    bindTex(gl, this.progStep, 'uWater', this.texA, 0);
    u2f(gl, this.progStep, 'uTexelSize', ts, ts);
    uf( gl, this.progStep, 'uViscosity', this.viscosity);

    drawFullscreen(gl, this.vao);

    [this.texA, this.texB] = [this.texB, this.texA];
    [this.fboA, this.fboB] = [this.fboB, this.fboA];

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── _updateNormals — reconstruct BA from height field ─────────────────────

  private _updateNormals(): void {
    const gl = this.gl;
    const ts = 1.0 / this.texSize;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboB);
    gl.viewport(0, 0, this.texSize, this.texSize);
    gl.useProgram(this.progNormal);

    bindTex(gl, this.progNormal, 'uWater', this.texA, 0);
    u2f(gl, this.progNormal, 'uTexelSize', ts, ts);

    drawFullscreen(gl, this.vao);

    [this.texA, this.texB] = [this.texB, this.texA];
    [this.fboA, this.fboB] = [this.fboB, this.fboA];

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── update — advance simulation + render ──────────────────────────────────

  update(dt: number): void {
    if (this.stopped || !this.W || !this.H) return;
    const gl = this.gl;

    // Clamp dt to avoid spiral-of-death on tab resume
    const safeDt = Math.min(dt, 0.1);
    this.uTime  += safeDt;

    // ── Simulation passes ────────────────────────────────────────────────────
    for (let i = 0; i < this.stepsPerFrame; i++) {
      this._step();
    }
    this._updateNormals();

    // ── Auto-drop: periodic ambient ripples to keep the surface alive ────────
    // AT water element never goes completely still; we inject small background
    // drops at ~1 Hz to match the environmental animation feel.
    if (Math.floor(this.uTime * 1.1) > Math.floor((this.uTime - safeDt) * 1.1)) {
      const rx = 0.2 + Math.random() * 0.6;
      const ry = 0.2 + Math.random() * 0.6;
      this.addDrop(rx, ry, 0.02 + Math.random() * 0.02, 0.015 + Math.random() * 0.02);
    }

    // ── Render pass (to canvas) ──────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied alpha

    gl.useProgram(this.progRender);
    bindTex(gl, this.progRender, 'uWater', this.texA, 0);

    uf( gl, this.progRender, 'uTime',            this.uTime);
    uf( gl, this.progRender, 'uSpeed',           this.speed);
    uf( gl, this.progRender, 'uScale',           this.domainScale);
    uf( gl, this.progRender, 'uInnerScale',      this.uvScale);
    uf( gl, this.progRender, 'uBrightness',      this.brightness);
    uf( gl, this.progRender, 'uWaterUVStrength', this.waterUVStrength);
    uf( gl, this.progRender, 'uHeightScale',     this.heightScale);
    u3f(gl, this.progRender, 'uColor',           this.color[0], this.color[1], this.color[2]);

    drawFullscreen(gl, this.vao);

    gl.disable(gl.BLEND);
  }

  // ── Setters ───────────────────────────────────────────────────────────────

  setColor(hex: string):   void { this.color      = hexToRgb01(hex); }
  setSpeed(s: number):     void { this.speed       = s; }
  setBrightness(b: number): void { this.brightness  = b; }

  // ── Teardown ──────────────────────────────────────────────────────────────

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.ro.disconnect();

    const gl = this.gl;
    gl.deleteTexture(this.texA);
    gl.deleteTexture(this.texB);
    gl.deleteFramebuffer(this.fboA);
    gl.deleteFramebuffer(this.fboB);
    gl.deleteProgram(this.progDrop);
    gl.deleteProgram(this.progStep);
    gl.deleteProgram(this.progNormal);
    gl.deleteProgram(this.progRender);
    gl.deleteVertexArray(this.vao);

    this.canvas.remove();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Noop handle (returned when WebGL2 is unavailable)
// ─────────────────────────────────────────────────────────────────────────────

function createNoopHandle(): WaterBackgroundHandle {
  return {
    update:        (_dt: number)                              => undefined,
    addDrop:       (_x: number, _y: number, _r: number, _s: number) => undefined,
    setColor:      (_hex: string)                             => undefined,
    setSpeed:      (_s: number)                               => undefined,
    setBrightness: (_b: number)                               => undefined,
    stop:          ()                                         => undefined,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public mount function
// ─────────────────────────────────────────────────────────────────────────────

/**
 * mountWaterBackground
 *
 * Creates a full-screen WebGL2 water-surface canvas behind `container`
 * (zIndex = -1) using the AT TreeWaterShader technique.
 *
 * @param container  Host element — needs `position: relative` or `absolute`.
 * @param options    Parameter overrides; see WaterBackgroundOptions.
 * @returns          WaterBackgroundHandle for per-frame update + control.
 *
 * @example
 * ```ts
 * import { mountWaterBackground } from '@/lib/renderers/water-background';
 *
 * const bg = await mountWaterBackground(document.getElementById('scene')!);
 *
 * // Per-frame (rAF / Pixi Ticker):
 * ticker.add((dt) => bg.update(dt));
 *
 * // React to a cell pub/sub event:
 * cellBus.on('collision', ({ x, y }) => bg.addDrop(x, y, 0.03, 0.08));
 *
 * // Teardown:
 * bg.stop();
 * ```
 */
export async function mountWaterBackground(
  container: HTMLElement,
  options: WaterBackgroundOptions = {},
): Promise<WaterBackgroundHandle> {

  // ── Resolve defaults (AT UIL values) ────────────────────────────────────────
  const opts: Required<WaterBackgroundOptions> = {
    color:            options.color            ?? '#ffffff',   // AT uColor
    speed:            options.speed            ?? 0.04,        // AT uSpeed
    domainScale:      options.domainScale      ?? 202.03,      // AT outer uScale
    uvScale:          options.uvScale          ?? 1000,        // AT inner uScale
    brightness:       options.brightness       ?? 2,           // AT uBrightness
    waterUVStrength:  options.waterUVStrength  ?? -5,          // AT uWaterUVStrength
    textureSize:      options.textureSize      ?? 512,         // AT water_texture
    viscosity:        options.viscosity        ?? 0.98,        // AT water_viscosity
    heightScale:      options.heightScale      ?? 0.05,        // AT water_heightScale
    stepsPerFrame:    options.stepsPerFrame    ?? 2,
    backgroundColor:  options.backgroundColor ?? '#0a1628',
  };

  // ── Create full-screen canvas behind container ─────────────────────────────
  const canvas = document.createElement('canvas');
  canvas.id = 'water-background';
  canvas.style.cssText = [
    'position: absolute',
    'inset: 0',
    'width: 100%',
    'height: 100%',
    `background-color: ${opts.backgroundColor}`,
    'z-index: -1',
    'pointer-events: none',
    'display: block',
  ].join('; ');

  if (getComputedStyle(container).position === 'static') {
    container.style.position = 'relative';
  }
  container.insertBefore(canvas, container.firstChild);

  // ── Acquire WebGL2 context ─────────────────────────────────────────────────
  const gl = canvas.getContext('webgl2', {
    alpha:                true,
    antialias:            false,
    premultipliedAlpha:   true,
    preserveDrawingBuffer: false,
  }) as WebGL2RenderingContext | null;

  if (!gl) {
    console.warn('[WaterBackground] WebGL2 not available — background disabled');
    canvas.remove();
    return createNoopHandle();
  }

  // ── Check floating-point render target support ─────────────────────────────
  const extFloat = gl.getExtension('EXT_color_buffer_float');
  if (!extFloat) {
    console.warn('[WaterBackground] EXT_color_buffer_float not available — falling back to RGBA16F');
    // Will fall through; createFloatTexture uses RGBA32F but the browser will
    // reject it.  In practice this only affects Safari < 16 / very old Android.
    canvas.remove();
    return createNoopHandle();
  }

  // ── Instantiate renderer ───────────────────────────────────────────────────
  let bg: WaterBackground;
  try {
    bg = new WaterBackground(gl, canvas, container, opts);
  } catch (err) {
    console.error('[WaterBackground] init error:', err);
    canvas.remove();
    return createNoopHandle();
  }

  // ── Return public handle ───────────────────────────────────────────────────
  return {
    update:        (dt)             => bg.update(dt),
    addDrop:       (x, y, r, s)    => bg.addDrop(x, y, r, s),
    setColor:      (hex)            => bg.setColor(hex),
    setSpeed:      (s)              => bg.setSpeed(s),
    setBrightness: (b)              => bg.setBrightness(b),
    stop:          ()               => bg.stop(),
  };
}
