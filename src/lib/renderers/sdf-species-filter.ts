/**
 * sdf-species-filter.ts — M045: cil-bolt SDF shader → PixiJS Filter
 *
 * 将 src/lib/shaders/cil-bolt.frag 的 SDF 闪电 shader 封装为 PixiJS Filter，
 * 供 buildCellContainer 在 species === 'cil-bolt' 时挂载到 Graphics 背景 quad，
 * 替代原先的 SPECIES_PATTERNS['cil-bolt'] Graphics 画法。
 *
 * ## 坐标适配
 * cil-bolt.frag 原用 gl_FragCoord + u_bbox 计算 UV：
 *   uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw
 * PixiJS Filter 已将坐标归一化为 vTextureCoord ∈ [0,1]，直接替代 uv 计算，
 * 无需传递 u_bbox / u_resolution。
 *
 * ## Uniforms
 *   u_time        (f32)        — 动画时间，由 Ticker 每帧驱动（秒）
 *   u_fillColor   (vec3<f32>)  — 闪电颜色 [r, g, b] 0-1
 *   u_opacity     (f32)        — 整体透明度，与 container.alpha 独立
 *   u_zigzagCount (f32)        — 折线段数（默认 6）
 *   u_amplitude   (f32)        — 折线横向幅度（默认 0.35，归一化坐标）
 *
 * ## 使用示例
 *   const filter = new CilBoltSDFFilter({ fillColor: [1, 0.65, 0.15] });
 *   graphics.filters = [filter];
 *   app.ticker.add(() => { filter.time += app.ticker.deltaMS / 1000; });
 *
 * Upstream reference:
 *   src/lib/shaders/cil-bolt.frag      — 原始 SDF 着色器
 *   upstream/pixijs-filters/src/crt/CRTFilter.ts  — time-driven Filter 模式
 *   upstream/pixijs-filters/src/shockwave/ShockwaveFilter.ts — apply() 更新 time
 *   upstream/pixijs-filters/src/defaults/ — vertex shader
 */

import {
  Filter,
  GlProgram,
  GpuProgram,
  type FilterSystem,
  type RenderSurface,
  type Texture,
} from 'pixi.js';
import { vertex, wgslVertex } from '../../upstream/pixijs-filters/src/defaults';

// ── Fragment shader — adapted cil-bolt.frag for PixiJS Filter ───────────────
//
// Key changes from cil-bolt.frag:
//  1. Remove #version / precision (PixiJS prepends them)
//  2. Replace gl_FragCoord + u_bbox UV with vTextureCoord (already 0-1 range)
//  3. Remove u_bbox / u_resolution uniforms (not needed in Filter context)
//  4. Output finalColor (PixiJS v8 naming) instead of gl_FragColor
//  5. Keep all AT bloom constants and SDF logic identical
//
const CIL_BOLT_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

// ── cil-bolt SDF uniforms ───────────────────────────────────────────────────
uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_zigzagCount;
uniform float u_amplitude;
uniform float u_time;

// ---- AT UIL params (from channels/physics/xiaodi_options_table.json / cil-bolt) ----
const float AT_BLOOM_INTENSITY        = 1.0;
const float AT_BLOOM_RADIUS           = 1.0;
const float AT_GLOBAL_BLOOM_STRENGTH  = 0.3;
const float AT_GLOBAL_BLOOM_RADIUS    = 0.2;
const float AT_HOME_BLOOM_STRENGTH    = 0.6;
const float AT_HOME_BLOOM_RADIUS      = 0.8;
const float AT_LIGHT_INTENSITY        = 2.19;
const float AT_WIGGLE_SPEED           = 0.7;
const float AT_LUMINOSITY_THRESHOLD   = 0.0;

// ── lygia/math/saturate.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

// ── lygia/sdf/lineSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Inigo Quiles
#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif

// Stroke mask derived from lineSDF
float strokeMask(vec2 p, vec2 a, vec2 b, float w) {
    float d = lineSDF(p, a, b);
    return smoothstep(w, w * 0.4, d);
}

void main() {
  // vTextureCoord is [0,1]^2 — remap to [-1,1] NDC
  vec2 uv = vTextureCoord;
  vec2 p  = uv * 2.0 - 1.0;   // [-1, 1]

  float strokeW = 0.045;
  float total   = 0.0;

  float steps = u_zigzagCount;
  float dy    = 2.0 / steps;

  // Animated phase offset — speed driven by AT_WIGGLE_SPEED
  float phase = sin(u_time * 2.5 * AT_WIGGLE_SPEED) * 0.15;

  // Core stroke — zigzag segments via lineSDF
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;

    float t0    = -1.0 + i       * dy;
    float t1    = -1.0 + (i+1.0) * dy;
    float side0 = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float side1 = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);

    vec2 a = vec2(side0 * u_amplitude + phase, t0);
    vec2 b = vec2(side1 * u_amplitude + phase, t1);

    total = max(total, strokeMask(p, a, b, strokeW));
  }

  // Global bloom pass
  float glowGlobal  = 0.0;
  float globalGlowW = strokeW * (3.5 * AT_GLOBAL_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowGlobal = max(glowGlobal, strokeMask(p, a, b, globalGlowW) * AT_GLOBAL_BLOOM_STRENGTH);
  }

  // Home bloom pass
  float glowHome  = 0.0;
  float homeGlowW = strokeW * (5.0 * AT_HOME_BLOOM_RADIUS / AT_BLOOM_RADIUS);
  for (float i = 0.0; i < 32.0; i++) {
    if (i >= steps) break;
    float t0   = -1.0 + i * dy;
    float t1   = -1.0 + (i+1.0) * dy;
    float s0   = (mod(i,       2.0) < 1.0 ? 1.0 : -1.0);
    float s1   = (mod(i + 1.0, 2.0) < 1.0 ? 1.0 : -1.0);
    vec2 a     = vec2(s0 * u_amplitude + phase, t0);
    vec2 b     = vec2(s1 * u_amplitude + phase, t1);
    glowHome   = max(glowHome, strokeMask(p, a, b, homeGlowW) * AT_HOME_BLOOM_STRENGTH);
  }

  float lum     = dot(u_fillColor, vec3(0.2126, 0.7152, 0.0722));
  float lumGate = step(AT_LUMINOSITY_THRESHOLD, lum);

  float bloomSum = (glowGlobal + glowHome) * lumGate * AT_BLOOM_INTENSITY * (AT_LIGHT_INTENSITY / 2.19);
  float alpha    = clamp(total + bloomSum, 0.0, 1.0);

  finalColor = vec4(u_fillColor, alpha * u_opacity);
}
`;

// ── WGSL stub — kept minimal; runtime uses GL path ──────────────────────────
// Proper WGSL port would be needed for WebGPU renderer. For now we emit the
// same vertex + a passthrough fragment so the GpuProgram doesn't fail on init.
const CIL_BOLT_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilBoltUniforms {
  u_fillColor:   vec3<f32>,
  u_opacity:     f32,
  u_zigzagCount: f32,
  u_amplitude:   f32,
  u_time:        f32,
}

@group(1) @binding(0) var<uniform> cilBoltUniforms : CilBoltUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  // WGSL stub: passthrough — full WGSL port deferred (WebGL path used in practice)
  return textureSample(uTexture, uSampler, uv) * vec4<f32>(cilBoltUniforms.u_fillColor, cilBoltUniforms.u_opacity);
}
`;

// ── CilBoltSDFFilter options ─────────────────────────────────────────────────

export interface CilBoltSDFFilterOptions {
  /**
   * Fill/stroke colour for the lightning bolt, as [r, g, b] in 0–1 range.
   * Defaults to cil-bolt species colour (0xFFA726 → [1.0, 0.643, 0.149]).
   */
  fillColor?: [number, number, number];
  /**
   * Overall opacity of the bolt (independent of container.alpha).
   * @default 1.0
   */
  opacity?: number;
  /**
   * Number of zigzag segments (lightning bolt teeth).
   * @default 6
   */
  zigzagCount?: number;
  /**
   * Horizontal amplitude of the zigzag in normalised [-1, 1] space.
   * @default 0.35
   */
  amplitude?: number;
  /**
   * Starting animation time (seconds).
   * @default 0
   */
  time?: number;
}

// ── CilBoltSDFFilter ─────────────────────────────────────────────────────────

/**
 * CilBoltSDFFilter — PixiJS Filter wrapping the cil-bolt.frag SDF lightning shader.
 *
 * Designed to be applied to a transparent Graphics quad that covers the cell
 * content area.  The shader draws the animated zigzag lightning bolt using
 * pure SDF math (no raster graphics), with AT-calibrated bloom passes for glow.
 *
 * ## Ticker integration (M045)
 * The caller must drive `filter.time` each frame:
 *
 *   const filter = new CilBoltSDFFilter({ fillColor: [1, 0.64, 0.15] });
 *   container.__boltFilter = filter;
 *   // In the app.ticker.add() loop:
 *   filter.time = elapsed;  // or: filter.time += ticker.deltaMS / 1000
 *
 * @example
 *   const boltFilter = new CilBoltSDFFilter({
 *     fillColor: [1.0, 0.643, 0.149],
 *     zigzagCount: 6,
 *     amplitude: 0.35,
 *   });
 *   patternGraphics.filters = [boltFilter];
 */
export class CilBoltSDFFilter extends Filter {
  /** Default options. */
  public static readonly DEFAULT_OPTIONS: Required<CilBoltSDFFilterOptions> = {
    fillColor:   [1.0, 0.643, 0.149],   // cil-bolt orange (0xFFA726)
    opacity:     1.0,
    zigzagCount: 6,
    amplitude:   0.35,
    time:        0,
  };

  /**
   * Typed uniform accessors (backed by the `cilBoltUniforms` resource).
   * Updated in apply() before each draw call.
   */
  public uniforms: {
    u_fillColor:   Float32Array;   // vec3
    u_opacity:     number;
    u_zigzagCount: number;
    u_amplitude:   number;
    u_time:        number;
  };

  /**
   * Current animation time in seconds.
   * Set this every frame via the PixiJS Ticker to animate the lightning bolt.
   */
  public time: number;

  constructor(options?: CilBoltSDFFilterOptions) {
    const opts = { ...CilBoltSDFFilter.DEFAULT_OPTIONS, ...options };

    const gpuProgram = GpuProgram.from({
      vertex: {
        source: wgslVertex,
        entryPoint: 'mainVertex',
      },
      fragment: {
        source: CIL_BOLT_WGSL,
        entryPoint: 'mainFragment',
      },
    });

    const glProgram = GlProgram.from({
      vertex,
      fragment: CIL_BOLT_FRAGMENT,
      name: 'cil-bolt-sdf-filter',
    });

    super({
      gpuProgram,
      glProgram,
      resources: {
        cilBoltUniforms: {
          u_fillColor:   { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:     { value: opts.opacity,     type: 'f32' },
          u_zigzagCount: { value: opts.zigzagCount, type: 'f32' },
          u_amplitude:   { value: opts.amplitude,   type: 'f32' },
          u_time:        { value: opts.time,         type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilBoltUniforms.uniforms as typeof this.uniforms;
    this.time = opts.time;
  }

  /**
   * apply() — called by PixiJS every frame this filter is active.
   * Syncs `this.time` → `u_time` uniform before rendering.
   */
  public override apply(
    filterManager: FilterSystem,
    input: Texture,
    output: RenderSurface,
    clearMode: boolean,
  ): void {
    // Push current time into the GLSL uniform
    this.uniforms.u_time = this.time;
    filterManager.applyFilter(this, input, output, clearMode);
  }

  // ── Uniform accessors ──────────────────────────────────────────────────────

  /** Lightning bolt fill colour as [r, g, b] 0–1. */
  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  /** Overall bolt opacity (0–1). */
  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  /** Number of zigzag segments. */
  get zigzagCount(): number { return this.uniforms.u_zigzagCount; }
  set zigzagCount(value: number) { this.uniforms.u_zigzagCount = value; }

  /** Zigzag horizontal amplitude in normalised [-1,1] space. */
  get amplitude(): number { return this.uniforms.u_amplitude; }
  set amplitude(value: number) { this.uniforms.u_amplitude = value; }
}

// ══════════════════════════════════════════════════════════════════════════════
// M046: CilVectorSDFFilter — cil-vector.frag → PixiJS Filter
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-vector.frag (arrow grid SDF using polySDF + lineSDF).
// Adapts the shader for PixiJS Filter context:
//   - vTextureCoord replaces gl_FragCoord + u_bbox UV
//   - finalColor replaces gl_FragColor
//   - u_bbox / u_resolution removed (not needed in Filter)
//
// Uniforms:
//   u_fillColor   (vec3) — arrow colour [r,g,b] 0-1
//   u_opacity     (f32)  — overall opacity
//   u_arrowCount  (f32)  — arrows per row/col in the grid
//   u_angleSpread (f32)  — variation in arrow angle (radians)

const CIL_VECTOR_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_arrowCount;
uniform float u_angleSpread;

#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif
#ifndef TAU
#define TAU 6.2831853071795864769252867665590
#endif

#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in vec2 st, in int V) {
    st = st * 2.0 - 1.0;
    float a = atan(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(0.5 + a / v) * v - a) * r;
}
#endif

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float drawArrow(vec2 p, float angle, float scale) {
    float c = cos(angle), s = sin(angle);
    vec2 lp = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / scale;
    float shaft = sdBox(lp - vec2(-0.15, 0.0), vec2(0.22, 0.045));
    vec2 headUV = (lp - vec2(0.13, 0.0)) / 0.32 + 0.5;
    headUV = headUV - 0.5;
    float tmp = headUV.x;
    headUV.x = -headUV.y;
    headUV.y =  tmp;
    headUV = headUV + 0.5;
    float head = polySDF(headUV, 3) * 0.32 - 0.16;
    float d = min(shaft, head);
    return smoothstep(0.01, -0.01, d);
}

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = vTextureCoord;
    float n     = u_arrowCount;
    vec2  cell  = floor(uv * n);
    vec2  local = fract(uv * n) - 0.5;
    float jitter = (rand(cell) * 2.0 - 1.0) * u_angleSpread;
    float angle  = jitter;
    float scale = 0.45;
    float mask  = drawArrow(local, angle, scale);
    finalColor = vec4(u_fillColor, mask * u_opacity);
}
`;

const CIL_VECTOR_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilVectorUniforms {
  u_fillColor:   vec3<f32>,
  u_opacity:     f32,
  u_arrowCount:  f32,
  u_angleSpread: f32,
}

@group(1) @binding(0) var<uniform> cilVectorUniforms : CilVectorUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  return textureSample(uTexture, uSampler, uv) * vec4<f32>(cilVectorUniforms.u_fillColor, cilVectorUniforms.u_opacity);
}
`;

export interface CilVectorSDFFilterOptions {
  /** Arrow fill colour [r,g,b] 0-1. Defaults to cil-vector green (0x66BB6A). */
  fillColor?: [number, number, number];
  /** Overall opacity (0-1). @default 1.0 */
  opacity?: number;
  /** Arrows per row/column in the grid. @default 4 */
  arrowCount?: number;
  /** Max random angle spread in radians. @default 0.4 */
  angleSpread?: number;
}

/**
 * CilVectorSDFFilter — PixiJS Filter wrapping the cil-vector.frag SDF arrow-grid shader.
 *
 * Renders a grid of randomised-angle arrows via polySDF (triangle head) +
 * sdBox (shaft).  Designed for cil-vector (embedding / positional-encoding) cells.
 *
 * @example
 *   const filter = new CilVectorSDFFilter({ fillColor: [0.4, 0.73, 0.42] });
 *   patternGraphics.filters = [filter];
 */
export class CilVectorSDFFilter extends Filter {
  public static readonly DEFAULT_OPTIONS: Required<CilVectorSDFFilterOptions> = {
    fillColor:   [0.4, 0.733, 0.416],  // 0x66BB6A
    opacity:     1.0,
    arrowCount:  4,
    angleSpread: 0.4,
  };

  public uniforms: {
    u_fillColor:   Float32Array;
    u_opacity:     number;
    u_arrowCount:  number;
    u_angleSpread: number;
  };

  constructor(options?: CilVectorSDFFilterOptions) {
    const opts = { ...CilVectorSDFFilter.DEFAULT_OPTIONS, ...options };

    const gpuProgram = GpuProgram.from({
      vertex: { source: wgslVertex, entryPoint: 'mainVertex' },
      fragment: { source: CIL_VECTOR_WGSL, entryPoint: 'mainFragment' },
    });

    const glProgram = GlProgram.from({
      vertex,
      fragment: CIL_VECTOR_FRAGMENT,
      name: 'cil-vector-sdf-filter',
    });

    super({
      gpuProgram,
      glProgram,
      resources: {
        cilVectorUniforms: {
          u_fillColor:   { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:     { value: opts.opacity,     type: 'f32' },
          u_arrowCount:  { value: opts.arrowCount,  type: 'f32' },
          u_angleSpread: { value: opts.angleSpread, type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilVectorUniforms.uniforms as typeof this.uniforms;
  }

  public override apply(
    filterManager: FilterSystem,
    input: Texture,
    output: RenderSurface,
    clearMode: boolean,
  ): void {
    filterManager.applyFilter(this, input, output, clearMode);
  }

  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  get arrowCount(): number { return this.uniforms.u_arrowCount; }
  set arrowCount(value: number) { this.uniforms.u_arrowCount = value; }

  get angleSpread(): number { return this.uniforms.u_angleSpread; }
  set angleSpread(value: number) { this.uniforms.u_angleSpread = value; }
}

// ══════════════════════════════════════════════════════════════════════════════
// M046: CilPlusSDFFilter — cil-plus.frag → PixiJS Filter
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-plus.frag (plus/cross SDF using rectSDF + sdBox2).
// Adapts for PixiJS Filter context (vTextureCoord, finalColor).
//
// Uniforms:
//   u_fillColor   (vec3) — plus colour [r,g,b] 0-1
//   u_opacity     (f32)  — overall opacity
//   u_armLength   (f32)  — half-length of each arm [0..1]
//   u_strokeWidth (f32)  — half-width of stroke [0..1]

const CIL_PLUS_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_armLength;
uniform float u_strokeWidth;

#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in vec2 st, in vec2 s) {
    vec2 p = st * 2.0 - 1.0;
    return max(abs(p.x / s.x), abs(p.y / s.y));
}
float sdBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
#endif

float sdPlus(vec2 p, float armLen, float sw) {
    float h = sdBox2(p, vec2(armLen, sw));
    float v = sdBox2(p, vec2(sw, armLen));
    return min(h, v);
}

void main() {
    vec2 uv = vTextureCoord;
    vec2 p  = uv * 2.0 - 1.0;

    float d    = sdPlus(p, u_armLength, u_strokeWidth);
    float mask = smoothstep(0.015, -0.015, d);
    float glow = smoothstep(0.08, 0.0, d) * 0.25;
    float alpha = clamp(mask + glow, 0.0, 1.0);

    finalColor = vec4(u_fillColor, alpha * u_opacity);
}
`;

const CIL_PLUS_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilPlusUniforms {
  u_fillColor:   vec3<f32>,
  u_opacity:     f32,
  u_armLength:   f32,
  u_strokeWidth: f32,
}

@group(1) @binding(0) var<uniform> cilPlusUniforms : CilPlusUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  return textureSample(uTexture, uSampler, uv) * vec4<f32>(cilPlusUniforms.u_fillColor, cilPlusUniforms.u_opacity);
}
`;

export interface CilPlusSDFFilterOptions {
  /** Plus colour [r,g,b] 0-1. Defaults to cil-plus pink (0xEC407A). */
  fillColor?: [number, number, number];
  /** Overall opacity (0-1). @default 1.0 */
  opacity?: number;
  /** Half-length of each arm in [-1,1] space. @default 0.55 */
  armLength?: number;
  /** Half-width of stroke in [-1,1] space. @default 0.12 */
  strokeWidth?: number;
}

/**
 * CilPlusSDFFilter — PixiJS Filter wrapping the cil-plus.frag SDF plus/cross shader.
 *
 * Renders a crisp plus sign with a soft inner glow using rectSDF / sdBox2.
 * Designed for cil-plus (Add & Norm) cells.
 *
 * @example
 *   const filter = new CilPlusSDFFilter({ fillColor: [0.93, 0.25, 0.48] });
 *   patternGraphics.filters = [filter];
 */
export class CilPlusSDFFilter extends Filter {
  public static readonly DEFAULT_OPTIONS: Required<CilPlusSDFFilterOptions> = {
    fillColor:   [0.925, 0.251, 0.478],  // 0xEC407A
    opacity:     1.0,
    armLength:   0.55,
    strokeWidth: 0.12,
  };

  public uniforms: {
    u_fillColor:   Float32Array;
    u_opacity:     number;
    u_armLength:   number;
    u_strokeWidth: number;
  };

  constructor(options?: CilPlusSDFFilterOptions) {
    const opts = { ...CilPlusSDFFilter.DEFAULT_OPTIONS, ...options };

    const gpuProgram = GpuProgram.from({
      vertex: { source: wgslVertex, entryPoint: 'mainVertex' },
      fragment: { source: CIL_PLUS_WGSL, entryPoint: 'mainFragment' },
    });

    const glProgram = GlProgram.from({
      vertex,
      fragment: CIL_PLUS_FRAGMENT,
      name: 'cil-plus-sdf-filter',
    });

    super({
      gpuProgram,
      glProgram,
      resources: {
        cilPlusUniforms: {
          u_fillColor:   { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:     { value: opts.opacity,     type: 'f32' },
          u_armLength:   { value: opts.armLength,   type: 'f32' },
          u_strokeWidth: { value: opts.strokeWidth, type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilPlusUniforms.uniforms as typeof this.uniforms;
  }

  public override apply(
    filterManager: FilterSystem,
    input: Texture,
    output: RenderSurface,
    clearMode: boolean,
  ): void {
    filterManager.applyFilter(this, input, output, clearMode);
  }

  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  get armLength(): number { return this.uniforms.u_armLength; }
  set armLength(value: number) { this.uniforms.u_armLength = value; }

  get strokeWidth(): number { return this.uniforms.u_strokeWidth; }
  set strokeWidth(value: number) { this.uniforms.u_strokeWidth = value; }
}

// ══════════════════════════════════════════════════════════════════════════════
// M046: CilArrowRightSDFFilter — cil-arrow-right.frag → PixiJS Filter
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-arrow-right.frag (tiled scrolling chevron via lineSDF).
// Adapts for PixiJS Filter context (vTextureCoord, finalColor).
// The u_time uniform should be driven by the Ticker for scroll animation.
//
// Uniforms:
//   u_fillColor   (vec3) — chevron colour [r,g,b] 0-1
//   u_opacity     (f32)  — overall opacity
//   u_arrowWidth  (f32)  — stroke thickness [0..1]
//   u_time        (f32)  — animation time (seconds), drives horizontal scroll

const CIL_ARROW_RIGHT_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_arrowWidth;
uniform float u_time;

#ifndef FNC_SATURATE
#define FNC_SATURATE
#define saturate(V) clamp(V, 0.0, 1.0)
#endif

#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF(in vec2 st, in vec2 a, in vec2 b) {
    vec2 b_to_a = b - a;
    vec2 to_a   = st - a;
    float h = saturate(dot(to_a, b_to_a) / dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a);
}
#endif

float sdArrowRight(vec2 p, float w) {
    vec2 a1 = vec2(-0.45,  0.40);
    vec2 b1 = vec2( 0.45,  0.0 );
    vec2 a2 = vec2(-0.45, -0.40);
    vec2 b2 = vec2( 0.45,  0.0 );
    float d1 = lineSDF(p, a1, b1);
    float d2 = lineSDF(p, a2, b2);
    return min(d1, d2) - w;
}

void main() {
    vec2 uv = vTextureCoord;

    float cols   = 3.0;
    float rows   = 3.0;
    vec2  scroll = vec2(u_time * 0.25, 0.0);

    vec2  tiled  = fract(uv * vec2(cols, rows) + scroll);
    vec2  lp     = tiled * 2.0 - 1.0;

    float d    = sdArrowRight(lp, u_arrowWidth * 0.5);
    float mask = smoothstep(0.02, -0.01, d);

    float fade  = smoothstep(0.0, 0.6, tiled.x);
    float alpha = mask * (0.4 + 0.6 * fade);

    finalColor = vec4(u_fillColor, clamp(alpha, 0.0, 1.0) * u_opacity);
}
`;

const CIL_ARROW_RIGHT_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilArrowRightUniforms {
  u_fillColor:  vec3<f32>,
  u_opacity:    f32,
  u_arrowWidth: f32,
  u_time:       f32,
}

@group(1) @binding(0) var<uniform> cilArrowRightUniforms : CilArrowRightUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  return textureSample(uTexture, uSampler, uv) * vec4<f32>(cilArrowRightUniforms.u_fillColor, cilArrowRightUniforms.u_opacity);
}
`;

export interface CilArrowRightSDFFilterOptions {
  /** Chevron colour [r,g,b] 0-1. Defaults to cil-arrow-right slate (0x78909C). */
  fillColor?: [number, number, number];
  /** Overall opacity (0-1). @default 1.0 */
  opacity?: number;
  /** Stroke thickness in normalised space. @default 0.08 */
  arrowWidth?: number;
  /** Starting animation time (seconds). @default 0 */
  time?: number;
}

/**
 * CilArrowRightSDFFilter — PixiJS Filter wrapping the cil-arrow-right.frag SDF chevron shader.
 *
 * Renders a 3×3 tiled scrolling chevron pattern using lineSDF.
 * The animation time drives horizontal scroll — set `filter.time` each frame via Ticker.
 * Designed for cil-arrow-right (skip connection / routing flow) cells.
 *
 * ## Ticker integration
 *   const filter = new CilArrowRightSDFFilter({ fillColor: [0.47, 0.565, 0.612] });
 *   container.__arrowRightFilter = filter;
 *   // In the app.ticker.add() loop:
 *   filter.time = elapsed;
 *
 * @example
 *   const filter = new CilArrowRightSDFFilter({ arrowWidth: 0.08 });
 *   patternGraphics.filters = [filter];
 */
export class CilArrowRightSDFFilter extends Filter {
  public static readonly DEFAULT_OPTIONS: Required<CilArrowRightSDFFilterOptions> = {
    fillColor:  [0.471, 0.565, 0.612],  // 0x78909C
    opacity:    1.0,
    arrowWidth: 0.08,
    time:       0,
  };

  public uniforms: {
    u_fillColor:  Float32Array;
    u_opacity:    number;
    u_arrowWidth: number;
    u_time:       number;
  };

  /** Current animation time in seconds. Drive with Ticker for scroll animation. */
  public time: number;

  constructor(options?: CilArrowRightSDFFilterOptions) {
    const opts = { ...CilArrowRightSDFFilter.DEFAULT_OPTIONS, ...options };

    const gpuProgram = GpuProgram.from({
      vertex: { source: wgslVertex, entryPoint: 'mainVertex' },
      fragment: { source: CIL_ARROW_RIGHT_WGSL, entryPoint: 'mainFragment' },
    });

    const glProgram = GlProgram.from({
      vertex,
      fragment: CIL_ARROW_RIGHT_FRAGMENT,
      name: 'cil-arrow-right-sdf-filter',
    });

    super({
      gpuProgram,
      glProgram,
      resources: {
        cilArrowRightUniforms: {
          u_fillColor:  { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:    { value: opts.opacity,    type: 'f32' },
          u_arrowWidth: { value: opts.arrowWidth, type: 'f32' },
          u_time:       { value: opts.time,       type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilArrowRightUniforms.uniforms as typeof this.uniforms;
    this.time = opts.time;
  }

  public override apply(
    filterManager: FilterSystem,
    input: Texture,
    output: RenderSurface,
    clearMode: boolean,
  ): void {
    this.uniforms.u_time = this.time;
    filterManager.applyFilter(this, input, output, clearMode);
  }

  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  get arrowWidth(): number { return this.uniforms.u_arrowWidth; }
  set arrowWidth(value: number) { this.uniforms.u_arrowWidth = value; }
}
