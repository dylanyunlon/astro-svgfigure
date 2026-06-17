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
// M062: CilVectorSDFFilter — cil-vector SDF shader → PixiJS Filter (arrow field animation)
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-vector.frag (arrow grid SDF using polySDF + sdBox).
// M062 upgrades the static M046 filter with full Ticker-driven u_time animation:
//   - u_time drives per-arrow curl-noise angle rotation, making the vector field flow
//   - u_arrow_length controls relative shaft+head scale (maps from params.json arrow_length px)
//   - u_field_scale controls the curl-noise spatial frequency (from params.json field_scale)
//
// Shader changes vs M046:
//   + uniform float u_time        — animation clock (seconds), drives angle modulation
//   + uniform float u_arrow_length — normalised arrow scale relative to cell half-size [0.1, 1.0]
//   + uniform float u_field_scale  — curl-noise frequency (higher = tighter swirls) [0.5, 8.0]
//   • angle = jitter + curl_noise(cell_centre + time * flow_speed) * TAU
//     where curl_noise is a simple 2D hash-based pseudo-curl giving coherent flow
//   • scale driven by u_arrow_length (replaces hard-coded 0.45)
//
// Adapts for PixiJS Filter context:
//   - vTextureCoord replaces gl_FragCoord + u_bbox UV
//   - finalColor replaces gl_FragColor
//   - u_bbox / u_resolution removed (not needed in Filter)
//
// Uniforms:
//   u_fillColor    (vec3) — arrow colour [r,g,b] 0-1
//   u_opacity      (f32)  — overall opacity
//   u_arrowCount   (f32)  — arrows per row/col in the grid
//   u_angleSpread  (f32)  — static per-cell angle jitter (radians)
//   u_time         (f32)  — animation clock (seconds); driven by Ticker
//   u_arrow_length (f32)  — normalised arrow length scale [0.1, 1.0]
//   u_field_scale  (f32)  — curl-noise spatial frequency

const CIL_VECTOR_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_arrowCount;
uniform float u_angleSpread;
uniform float u_time;
uniform float u_arrow_length;
uniform float u_field_scale;

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

// ── Pseudo-random hash (lygia/math/rand inlined) ─────────────────────────────
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(127.1, 311.7))) * 43758.5453);
}

// ── Curl-noise-like coherent angle field ──────────────────────────────────────
// Samples a 2D gradient field at (p * field_scale + time_offset) and returns
// an angle in [0, TAU].  Uses two orthogonal hash values to form a smooth
// pseudo-rotation.  The time offset makes the field advect — arrows appear to
// "flow" across the cell as u_time increases.
//
// Implementation: simple bilinear-like interpolation on a hash grid.
// Not true curl noise but gives the same coherent, divergence-free look at
// the scale of a single cell (where we only need O(arrowCount^2) samples).
float curlAngle(vec2 p, float fieldScale, float t) {
    // Advect p gently in time — creates a slowly drifting flow
    vec2 fp = p * fieldScale + vec2(t * 0.18, t * 0.11);

    // Integer cell and fractional parts of the scaled/advected coordinate
    vec2 i = floor(fp);
    vec2 f = fract(fp);

    // Smooth interpolation weights (smoothstep)
    vec2 u = f * f * (3.0 - 2.0 * f);

    // Hash at four corners of the noise cell → angles
    float a00 = rand(i + vec2(0.0, 0.0)) * TAU;
    float a10 = rand(i + vec2(1.0, 0.0)) * TAU;
    float a01 = rand(i + vec2(0.0, 1.0)) * TAU;
    float a11 = rand(i + vec2(1.0, 1.0)) * TAU;

    // Bilinear mix of unit-circle components to avoid angle wraparound artefacts
    vec2 v00 = vec2(cos(a00), sin(a00));
    vec2 v10 = vec2(cos(a10), sin(a10));
    vec2 v01 = vec2(cos(a01), sin(a01));
    vec2 v11 = vec2(cos(a11), sin(a11));

    vec2 vx0 = mix(v00, v10, u.x);
    vec2 vx1 = mix(v01, v11, u.x);
    vec2 vf  = mix(vx0, vx1, u.y);

    return atan(vf.y, vf.x);  // returns angle in (-PI, PI]
}

void main() {
    vec2 uv = vTextureCoord;
    float n     = u_arrowCount;
    vec2  cell  = floor(uv * n);
    vec2  local = fract(uv * n) - 0.5;

    // ── Static per-cell jitter (from u_angleSpread) ───────────────────────────
    float jitter = (rand(cell) * 2.0 - 1.0) * u_angleSpread;

    // ── Curl-noise time-driven base angle ─────────────────────────────────────
    // curlAngle is sampled at the cell's UV centre so all pixels within a cell
    // share the same flowing angle, preserving the grid aesthetic.
    vec2 cellCentre = (cell + 0.5) / n;   // [0,1] UV of cell centre
    float flowAngle = curlAngle(cellCentre, u_field_scale, u_time);

    float angle = flowAngle + jitter;

    // u_arrow_length: [0.1, 1.0] normalised scale; 0.45 was the M046 default.
    float scale = clamp(u_arrow_length * 0.9, 0.1, 0.9);

    float mask  = drawArrow(local, angle, scale);
    finalColor = vec4(u_fillColor, mask * u_opacity);
}
`;

// ── WGSL stub — kept minimal; runtime uses GL path ──────────────────────────
const CIL_VECTOR_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilVectorUniforms {
  u_fillColor:    vec3<f32>,
  u_opacity:      f32,
  u_arrowCount:   f32,
  u_angleSpread:  f32,
  u_time:         f32,
  u_arrow_length: f32,
  u_field_scale:  f32,
}

@group(1) @binding(0) var<uniform> cilVectorUniforms : CilVectorUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  // WGSL stub: passthrough — full WGSL port deferred (WebGL path used in practice)
  return textureSample(uTexture, uSampler, uv)
       * vec4<f32>(cilVectorUniforms.u_fillColor, cilVectorUniforms.u_opacity);
}
`;

// ── CilVectorSDFFilter options ────────────────────────────────────────────────

export interface CilVectorSDFFilterOptions {
  /**
   * Arrow fill colour [r,g,b] 0-1.
   * Defaults to cil-vector green (0x66BB6A → [0.4, 0.733, 0.416]).
   */
  fillColor?: [number, number, number];
  /** Overall opacity (0-1). @default 1.0 */
  opacity?: number;
  /** Arrows per row/column in the grid. @default 4 */
  arrowCount?: number;
  /** Max static random angle spread in radians (applied on top of curl noise). @default 0.4 */
  angleSpread?: number;
  /**
   * Starting animation time (seconds). Driven by Ticker each frame to make
   * the vector field flow.  Equivalent to `filter.time = elapsed` in the Ticker.
   * @default 0
   */
  time?: number;
  /**
   * Normalised arrow length scale relative to cell half-size [0.1, 1.0].
   * Maps from species_params.arrow_length (pixels) via px / (min(w,h)/2).
   * @default 0.5
   */
  arrowLength?: number;
  /**
   * Curl-noise spatial frequency — higher = tighter, more turbulent swirls.
   * Maps from species_params.field_scale.
   * @default 2.5
   */
  fieldScale?: number;
}

// ── CilVectorSDFFilter ────────────────────────────────────────────────────────

/**
 * CilVectorSDFFilter — PixiJS Filter wrapping the cil-vector SDF arrow-grid shader.
 *
 * M062 upgrade: adds Ticker-driven `u_time` uniform so the vector field flows
 * coherently over time using a bilinear-interpolated curl-noise angle field.
 * Each arrow's direction is the sum of:
 *   • a per-cell static jitter (u_angleSpread)
 *   • a time-advected smooth noise angle (u_field_scale, u_time)
 *
 * Designed for cil-vector (embedding / positional-encoding) cells where the
 * flowing arrows visualise information streaming through the embedding space.
 *
 * ## Ticker integration (M062)
 * The caller must drive `filter.time` each frame to animate the vector field:
 *
 *   const filter = new CilVectorSDFFilter({ fillColor: [0.4, 0.73, 0.42] });
 *   container.__vectorFilter = filter;
 *   // In the app.ticker.add() loop:
 *   filter.time = elapsed;  // seconds since start
 *
 * ## Uniform mapping from species_params (M062)
 *   arrow_count  → arrowCount  (grid density)
 *   angle_spread → angleSpread (static jitter, radians)
 *   arrow_length → arrowLength (px → px/(min(w,h)/2), normalised [0.1, 1.0])
 *   field_scale  → fieldScale  (noise frequency, direct)
 *
 * @example
 *   const vectorFilter = new CilVectorSDFFilter({
 *     fillColor:   [0.4, 0.733, 0.416],
 *     arrowCount:  5,
 *     angleSpread: 0.8,
 *     arrowLength: 0.5,
 *     fieldScale:  2.5,
 *   });
 *   patternGraphics.filters = [vectorFilter];
 *   app.ticker.add(() => { vectorFilter.time += app.ticker.deltaMS / 1000; });
 */
export class CilVectorSDFFilter extends Filter {
  /** Default options — calibrated to cil-vector species visual style. */
  public static readonly DEFAULT_OPTIONS: Required<CilVectorSDFFilterOptions> = {
    fillColor:   [0.4, 0.733, 0.416],  // 0x66BB6A — cil-vector green
    opacity:     1.0,
    arrowCount:  4,
    angleSpread: 0.4,
    time:        0,
    arrowLength: 0.5,   // ~midpoint; maps from arrow_length_px / halfMin
    fieldScale:  2.5,   // moderate curl-noise frequency
  };

  /**
   * Typed uniform accessors (backed by the `cilVectorUniforms` resource).
   * Updated in apply() before each draw call.
   */
  public uniforms: {
    u_fillColor:    Float32Array;   // vec3
    u_opacity:      number;
    u_arrowCount:   number;
    u_angleSpread:  number;
    u_time:         number;
    u_arrow_length: number;
    u_field_scale:  number;
  };

  /**
   * Current animation time in seconds.
   * Set this every frame via the PixiJS Ticker to animate the vector field flow.
   */
  public time: number;

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
          u_fillColor:    { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:      { value: opts.opacity,      type: 'f32' },
          u_arrowCount:   { value: opts.arrowCount,   type: 'f32' },
          u_angleSpread:  { value: opts.angleSpread,  type: 'f32' },
          u_time:         { value: opts.time,         type: 'f32' },
          u_arrow_length: { value: opts.arrowLength,  type: 'f32' },
          u_field_scale:  { value: opts.fieldScale,   type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilVectorUniforms.uniforms as typeof this.uniforms;
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
    this.uniforms.u_time = this.time;
    filterManager.applyFilter(this, input, output, clearMode);
  }

  // ── Uniform accessors ──────────────────────────────────────────────────────

  /** Arrow fill colour as [r, g, b] 0–1. */
  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  /** Overall opacity (0–1). */
  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  /** Arrows per row/column. */
  get arrowCount(): number { return this.uniforms.u_arrowCount; }
  set arrowCount(value: number) { this.uniforms.u_arrowCount = value; }

  /** Static per-cell angle jitter in radians. */
  get angleSpread(): number { return this.uniforms.u_angleSpread; }
  set angleSpread(value: number) { this.uniforms.u_angleSpread = value; }

  /** Normalised arrow length scale [0.1, 1.0]. */
  get arrowLength(): number { return this.uniforms.u_arrow_length; }
  set arrowLength(value: number) { this.uniforms.u_arrow_length = value; }

  /** Curl-noise spatial frequency. */
  get fieldScale(): number { return this.uniforms.u_field_scale; }
  set fieldScale(value: number) { this.uniforms.u_field_scale = value; }
}

// ══════════════════════════════════════════════════════════════════════════════
// M063: CilPlusSDFFilter — cil-plus.frag → PixiJS Filter (cross pulse animation)
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-plus.frag (plus/cross SDF using rectSDF + sdBox2).
// Adapts for PixiJS Filter context (vTextureCoord, finalColor).
//
// M063 升级：添加 u_time 脉冲动画
//   - 十字 SDF glow 环随 sin(u_time) 周期性呼吸
//   - u_cross_width  ↔ u_strokeWidth  (half-width of cross stroke)
//   - u_cross_radius ↔ u_armLength    (half-length of each arm)
//   - u_pulse_speed  — 脉冲频率 (rad/s，默认 2.0)
//   - u_pulse_amp    — 脉冲幅度 (0-1 glow intensity modulation，默认 0.3)
//
// Uniforms:
//   u_fillColor    (vec3) — plus colour [r,g,b] 0-1
//   u_opacity      (f32)  — overall opacity
//   u_cross_radius (f32)  — half-length of each arm [0..1]   (≡ armLength / u_armLength)
//   u_cross_width  (f32)  — half-width of stroke [0..1]      (≡ strokeWidth / u_strokeWidth)
//   u_time         (f32)  — animation time (seconds), drives pulse glow
//   u_pulse_speed  (f32)  — glow pulse angular frequency (rad/s)
//   u_pulse_amp    (f32)  — glow pulse amplitude (0-1 modulates glow intensity)

const CIL_PLUS_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

uniform vec3  u_fillColor;
uniform float u_opacity;
uniform float u_cross_radius;   // half-length of each arm in NDC [-1,1]
uniform float u_cross_width;    // half-width of stroke in NDC [-1,1]
uniform float u_time;           // animation time (seconds)
uniform float u_pulse_speed;    // pulse angular frequency (rad/s)
uniform float u_pulse_amp;      // pulse glow amplitude (0–1)

// ── lygia/sdf/rectSDF.glsl (inlined) ────────────────────────────────────────
// contributors: Patricio Gonzalez Vivo
#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in vec2 st, in vec2 s) {
    vec2 p = st * 2.0 - 1.0;
    return max(abs(p.x / s.x), abs(p.y / s.y));
}
// Signed box SDF — used for arm extrusions
float sdBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
#endif

// SDF for an axis-aligned plus centered at origin.
// Two overlapping sdBox2 rectangles — identical to cil-plus.frag sdPlus().
float sdPlus(vec2 p, float armLen, float sw) {
    float h = sdBox2(p, vec2(armLen, sw));
    float v = sdBox2(p, vec2(sw, armLen));
    return min(h, v);
}

void main() {
    vec2 uv = vTextureCoord;
    vec2 p  = uv * 2.0 - 1.0;   // remap [0,1] → [-1,1]

    float d    = sdPlus(p, u_cross_radius, u_cross_width);

    // Core fill mask — sharp anti-aliased edge
    float mask = smoothstep(0.015, -0.015, d);

    // Pulse glow: sin-wave modulated outer glow ring driven by u_time
    // The pulse cycles the glow intensity between (base ± u_pulse_amp).
    float pulse       = 0.5 + 0.5 * sin(u_time * u_pulse_speed);   // [0,1]
    float glowBase    = 0.20;                                         // static inner glow floor
    float glowPulse   = glowBase + u_pulse_amp * pulse;              // modulated peak
    float glowNear    = smoothstep(0.08, 0.0,  d) * glowPulse;      // near-edge bloom
    float glowFar     = smoothstep(0.22, 0.0,  d) * (u_pulse_amp * pulse * 0.35); // wide halo

    float alpha = clamp(mask + glowNear + glowFar, 0.0, 1.0);

    finalColor = vec4(u_fillColor, alpha * u_opacity);
}
`;

// ── WGSL stub — kept minimal; runtime uses GL path ──────────────────────────
const CIL_PLUS_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilPlusUniforms {
  u_fillColor:    vec3<f32>,
  u_opacity:      f32,
  u_cross_radius: f32,
  u_cross_width:  f32,
  u_time:         f32,
  u_pulse_speed:  f32,
  u_pulse_amp:    f32,
}

@group(1) @binding(0) var<uniform> cilPlusUniforms : CilPlusUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  // WGSL stub: passthrough — full WGSL port deferred (WebGL path used in practice)
  return textureSample(uTexture, uSampler, uv)
       * vec4<f32>(cilPlusUniforms.u_fillColor, cilPlusUniforms.u_opacity);
}
`;

// ── CilPlusSDFFilter options ─────────────────────────────────────────────────

export interface CilPlusSDFFilterOptions {
  /**
   * Plus/cross fill colour as [r, g, b] in 0–1 range.
   * Defaults to cil-plus pink (0xEC407A → [0.925, 0.251, 0.478]).
   */
  fillColor?: [number, number, number];
  /** Overall opacity of the cross (independent of container.alpha). @default 1.0 */
  opacity?: number;
  /**
   * Half-length of each arm in NDC [-1, 1] space.
   * Maps to u_cross_radius (equivalent to legacy u_armLength).
   * Derived from species_params.arm_length / (min(w,h)/2) in buildCellContainer.
   * @default 0.55
   */
  armLength?: number;
  /**
   * Half-width of the cross stroke in NDC [-1, 1] space.
   * Maps to u_cross_width (equivalent to legacy u_strokeWidth).
   * Derived from species_params.stroke_width / (min(w,h)/2) in buildCellContainer.
   * @default 0.12
   */
  strokeWidth?: number;
  /**
   * Starting animation time (seconds).
   * Driven each frame by the PixiJS Ticker.
   * @default 0
   */
  time?: number;
  /**
   * Glow pulse angular frequency in radians per second.
   * Higher values = faster pulsing.
   * @default 2.0
   */
  pulseSpeed?: number;
  /**
   * Glow pulse amplitude — how much the outer halo expands/contracts.
   * Range [0, 1]; 0 = no pulse (static glow only).
   * @default 0.3
   */
  pulseAmp?: number;
}

// ── CilPlusSDFFilter ─────────────────────────────────────────────────────────

/**
 * CilPlusSDFFilter — PixiJS Filter wrapping the cil-plus.frag SDF plus/cross shader.
 *
 * M063 upgrade: adds u_time-driven pulse animation to the cross glow.
 * The outer halo cycles via sin(u_time * u_pulse_speed), producing a soft
 * breathing effect consistent with AT HydraBloom animation patterns.
 *
 * Uniform mapping from species_params:
 *   species_params.arm_length   → armLength   → u_cross_radius
 *   species_params.stroke_width → strokeWidth → u_cross_width
 *   species_params.pulse_speed  → pulseSpeed  → u_pulse_speed
 *   species_params.pulse_amp    → pulseAmp    → u_pulse_amp
 *
 * ## Ticker integration (M063)
 * The caller must drive `filter.time` each frame:
 *
 *   const filter = new CilPlusSDFFilter({ fillColor: [0.93, 0.25, 0.48] });
 *   container.__plusFilter = filter;
 *   // In the app.ticker.add() loop:
 *   filter.time = elapsed;   // seconds since start
 *
 * @example
 *   const plusFilter = new CilPlusSDFFilter({
 *     fillColor:   [0.925, 0.251, 0.478],
 *     armLength:   0.55,
 *     strokeWidth: 0.12,
 *     pulseSpeed:  2.0,
 *     pulseAmp:    0.3,
 *   });
 *   patternGraphics.filters = [plusFilter];
 */
export class CilPlusSDFFilter extends Filter {
  /** Default options — calibrated for cil-plus (Add & Norm) cells. */
  public static readonly DEFAULT_OPTIONS: Required<CilPlusSDFFilterOptions> = {
    fillColor:   [0.925, 0.251, 0.478],  // 0xEC407A — cil-plus pink
    opacity:     1.0,
    armLength:   0.55,
    strokeWidth: 0.12,
    time:        0,
    pulseSpeed:  2.0,   // ~0.33 Hz breathing cycle
    pulseAmp:    0.3,
  };

  /**
   * Typed uniform accessors (backed by the `cilPlusUniforms` resource).
   * Updated in apply() before each draw call.
   */
  public uniforms: {
    u_fillColor:    Float32Array;   // vec3
    u_opacity:      number;
    u_cross_radius: number;         // ≡ armLength
    u_cross_width:  number;         // ≡ strokeWidth
    u_time:         number;
    u_pulse_speed:  number;
    u_pulse_amp:    number;
  };

  /**
   * Current animation time in seconds.
   * Set this every frame via the PixiJS Ticker to animate the cross pulse.
   */
  public time: number;

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
          u_fillColor:    { value: new Float32Array(opts.fillColor), type: 'vec3<f32>' },
          u_opacity:      { value: opts.opacity,     type: 'f32' },
          u_cross_radius: { value: opts.armLength,   type: 'f32' },
          u_cross_width:  { value: opts.strokeWidth, type: 'f32' },
          u_time:         { value: opts.time,        type: 'f32' },
          u_pulse_speed:  { value: opts.pulseSpeed,  type: 'f32' },
          u_pulse_amp:    { value: opts.pulseAmp,    type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilPlusUniforms.uniforms as typeof this.uniforms;
    this.time = opts.time;
  }

  /**
   * apply() — called by PixiJS every frame this filter is active.
   * Syncs `this.time` → `u_time` uniform before rendering to drive pulse animation.
   */
  public override apply(
    filterManager: FilterSystem,
    input: Texture,
    output: RenderSurface,
    clearMode: boolean,
  ): void {
    this.uniforms.u_time = this.time;
    filterManager.applyFilter(this, input, output, clearMode);
  }

  // ── Uniform accessors ──────────────────────────────────────────────────────

  /** Cross fill colour as [r, g, b] 0–1. */
  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  /** Overall cross opacity (0–1). */
  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  /** Half-length of each arm in NDC space (u_cross_radius). */
  get armLength(): number { return this.uniforms.u_cross_radius; }
  set armLength(value: number) { this.uniforms.u_cross_radius = value; }

  /** Half-width of the cross stroke in NDC space (u_cross_width). */
  get strokeWidth(): number { return this.uniforms.u_cross_width; }
  set strokeWidth(value: number) { this.uniforms.u_cross_width = value; }

  /** Pulse angular frequency (rad/s). */
  get pulseSpeed(): number { return this.uniforms.u_pulse_speed; }
  set pulseSpeed(value: number) { this.uniforms.u_pulse_speed = value; }

  /** Pulse glow amplitude (0–1). */
  get pulseAmp(): number { return this.uniforms.u_pulse_amp; }
  set pulseAmp(value: number) { this.uniforms.u_pulse_amp = value; }
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

// ══════════════════════════════════════════════════════════════════════════════
// M039: CilEyeSDFFilter — cil-eye.frag → PixiJS Filter
// ══════════════════════════════════════════════════════════════════════════════
//
// Wraps src/lib/shaders/cil-eye.frag (attention/perception cell: pupil + iris
// ring + radial rays + sclera halo, all drawn via circleSDF).
//
// Adapts the shader for PixiJS Filter context:
//   - vTextureCoord replaces gl_FragCoord + u_bbox UV
//   - finalColor replaces gl_FragColor
//   - u_bbox / u_resolution removed (not needed in Filter)
//   - u_time driven by Ticker each frame (radial ray rotation)
//
// AT UIL defaults (from channels/physics/xiaodi_options_table.json / cil-eye):
//   bloomStrength=1.2  bloomRadius=1.0  ambientIntensity=3.44
//   ambientColor=#0bed90  lightExposure=0.86  shadowFar=40  shadowBias=0.001
//
// Uniforms:
//   u_fillColor        (vec3) — base cell colour [r,g,b] 0-1
//   u_opacity          (f32)  — overall opacity
//   u_numRays          (f32)  — radial ray count (default 8)
//   u_pupilRadius      (f32)  — pupil radius in [-1,1] space (default 0.22)
//   u_focalIntensity   (f32)  — ray brightness multiplier (default 1.0)
//   u_time             (f32)  — animation time (seconds), drives ray rotation
//   u_bloomStrength    (f32)  — bloom ring intensity (default 1.2)
//   u_bloomRadius      (f32)  — bloom ring width factor (default 1.0)
//   u_ambientIntensity (f32)  — ambient light intensity (default 3.44)
//   u_ambientColor     (vec3) — ambient light colour (default #0bed90)
//   u_lightExposure    (f32)  — volumetric light exposure (default 0.86)
//   u_shadowFar        (f32)  — shadow far plane (default 40.0)
//   u_shadowBias       (f32)  — shadow bias (default 0.001)

const CIL_EYE_FRAGMENT = /* glsl */`
in vec2 vTextureCoord;
out vec4 finalColor;

uniform sampler2D uTexture;

// ── cil-eye SDF uniforms ─────────────────────────────────────────────────────
uniform vec3  u_fillColor;
uniform float u_opacity;

uniform float u_numRays;
uniform float u_pupilRadius;
uniform float u_focalIntensity;
uniform float u_time;

// AT bloom uniforms (defaults from UIL cil-eye entry)
uniform float u_bloomStrength;   // default 1.2
uniform float u_bloomRadius;     // default 1.0

// AT ambient light uniforms
uniform float u_ambientIntensity; // default 3.44
uniform vec3  u_ambientColor;     // default #0bed90 = (0.047, 0.929, 0.565)
uniform float u_lightExposure;    // default 0.86

// AT shadow uniforms
uniform float u_shadowFar;     // default 40.0
uniform float u_shadowBias;    // default 0.001

// ── lygia/sdf/circleSDF.glsl (inlined) ──────────────────────────────────────
#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF
float circleSDF(in vec2 v) {
    v -= 0.5;
    return length(v) * 2.0;
}
#endif

void main() {
  // vTextureCoord is [0,1]^2 — already the UV we need (replaces u_bbox calculation)
  vec2 uv = vTextureCoord;

  // circleSDF result: 0 at centre, ~1.41 at corner
  float dist = circleSDF(uv);

  // Centred coordinates for angle
  vec2 p = uv * 2.0 - 1.0;
  float angle = atan(p.y, p.x);

  // --- Pupil ---
  float pupilR = u_pupilRadius * 0.5;
  float pupil  = 1.0 - smoothstep(pupilR - 0.01, pupilR + 0.01, dist);

  // --- Iris ring ---
  float irisInner = (u_pupilRadius + 0.02) * 0.5;
  float irisOuter = (u_pupilRadius + 0.08) * 0.5;
  float iris = smoothstep(irisInner, irisOuter, dist)
             * (1.0 - smoothstep(0.425, 0.5, dist));

  // --- Radial rays ---
  float halfStep = 3.14159265 / u_numRays;
  float rayAngle = mod(angle + u_time * 0.3, halfStep * 2.0) - halfStep;
  float rayMask  = smoothstep(0.07, 0.0, abs(rayAngle));
  float rayFade  = smoothstep(0.5, irisInner + 0.06, dist)
                 * smoothstep(pupilR, pupilR + 0.06, dist);
  float rays     = rayMask * rayFade * u_focalIntensity;

  // --- Sclera (outer halo) ---
  float sclera = smoothstep(0.525, 0.44, dist);

  // --- AT ambient lighting ---
  float ambientFalloff = 1.0 - smoothstep(0.0, 0.6, dist);
  vec3  ambientContrib = u_ambientColor * u_ambientIntensity * u_lightExposure * ambientFalloff;

  // --- AT bloom glow ring ---
  float bloomCenter = (u_pupilRadius + 0.15) * 0.5;
  float bloomRing   = exp(-pow((dist - bloomCenter) / max(u_bloomRadius * 0.09, 0.005), 2.0));
  float bloom       = bloomRing * u_bloomStrength * 0.35;

  // --- AT shadow attenuation ---
  float shadowNorm   = clamp(dist / (u_shadowFar * 0.0125), 0.0, 1.0);
  float shadowFactor = 1.0 - shadowNorm * (1.0 - u_shadowBias * 100.0);

  float alpha = clamp(sclera * (iris + rays) + pupil, 0.0, 1.0);

  vec3 fc = u_fillColor + ambientContrib * (iris + bloom) * alpha;
  fc += u_fillColor * bloom;
  fc *= shadowFactor;

  finalColor = vec4(fc, alpha * u_opacity);
}
`;

// ── WGSL stub (WebGPU fallback — GL path used in practice) ──────────────────
const CIL_EYE_WGSL = /* wgsl */`
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler;

struct CilEyeUniforms {
  u_fillColor:        vec3<f32>,
  u_opacity:          f32,
  u_numRays:          f32,
  u_pupilRadius:      f32,
  u_focalIntensity:   f32,
  u_time:             f32,
  u_bloomStrength:    f32,
  u_bloomRadius:      f32,
  u_ambientIntensity: f32,
  u_ambientColor:     vec3<f32>,
  u_lightExposure:    f32,
  u_shadowFar:        f32,
  u_shadowBias:       f32,
}

@group(1) @binding(0) var<uniform> cilEyeUniforms : CilEyeUniforms;

@fragment
fn mainFragment(
  @builtin(position) position: vec4<f32>,
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  // WGSL stub: passthrough — full WGSL port deferred (WebGL path used in practice)
  return textureSample(uTexture, uSampler, uv)
       * vec4<f32>(cilEyeUniforms.u_fillColor, cilEyeUniforms.u_opacity);
}
`;

// ── CilEyeSDFFilter options ──────────────────────────────────────────────────

export interface CilEyeSDFFilterOptions {
  /**
   * Base fill colour for the eye, as [r, g, b] in 0–1 range.
   * Defaults to cil-eye indigo (0x5C6BC0 → [0.361, 0.420, 0.753]).
   */
  fillColor?: [number, number, number];
  /** Overall opacity of the eye (independent of container.alpha). @default 1.0 */
  opacity?: number;
  /** Number of radial rays emanating from the iris ring. @default 8 */
  numRays?: number;
  /** Pupil radius in [-1,1] normalised space. @default 0.22 */
  pupilRadius?: number;
  /** Brightness multiplier for the radial rays. @default 1.0 */
  focalIntensity?: number;
  /** Starting animation time (seconds). @default 0 */
  time?: number;
  /** AT bloom ring strength (UIL homebloom/bloomStrength). @default 1.2 */
  bloomStrength?: number;
  /** AT bloom ring radius factor (UIL homebloom/bloomRadius). @default 1.0 */
  bloomRadius?: number;
  /** AT ambient light intensity (L_Element_11 intensity). @default 3.44 */
  ambientIntensity?: number;
  /** AT ambient light colour [r,g,b] 0-1 (L_Element_11 #0bed90). @default [0.047, 0.929, 0.565] */
  ambientColor?: [number, number, number];
  /** AT volumetric light exposure (VolumetricLight fExposure). @default 0.86 */
  lightExposure?: number;
  /** AT shadow far plane (SHADOW_Element_9 far). @default 40.0 */
  shadowFar?: number;
  /** AT shadow bias (derived from shadow size 1024). @default 0.001 */
  shadowBias?: number;
}

// ── CilEyeSDFFilter ──────────────────────────────────────────────────────────

/**
 * CilEyeSDFFilter — PixiJS Filter wrapping the cil-eye.frag SDF eye shader.
 *
 * Renders the attention/perception cell pattern: pupil (solid disc), iris ring,
 * radial rotating rays, and sclera halo — all from pure SDF (circleSDF), with
 * AT-calibrated ambient lighting, bloom glow ring, and shadow attenuation.
 *
 * ## Ticker integration (M039)
 * The caller must drive `filter.time` each frame to animate ray rotation:
 *
 *   const filter = new CilEyeSDFFilter({ fillColor: [0.361, 0.420, 0.753] });
 *   container.__eyeFilter = filter;
 *   // In the app.ticker.add() loop:
 *   filter.time = elapsed;  // seconds since start
 *
 * @example
 *   const eyeFilter = new CilEyeSDFFilter({
 *     fillColor: [0.361, 0.420, 0.753],
 *     numRays: 8,
 *     pupilRadius: 0.22,
 *   });
 *   patternGraphics.filters = [eyeFilter];
 */
export class CilEyeSDFFilter extends Filter {
  /** Default options — AT UIL calibrated values for cil-eye. */
  public static readonly DEFAULT_OPTIONS: Required<CilEyeSDFFilterOptions> = {
    fillColor:        [0.361, 0.420, 0.753],  // 0x5C6BC0 — cil-eye indigo
    opacity:          1.0,
    numRays:          8,
    pupilRadius:      0.22,
    focalIntensity:   1.0,
    time:             0,
    bloomStrength:    1.2,   // UIL homebloom/bloomStrength
    bloomRadius:      1.0,   // UIL homebloom/bloomRadius
    ambientIntensity: 3.44,  // L_Element_11 intensity
    ambientColor:     [0.047, 0.929, 0.565],  // #0bed90
    lightExposure:    0.86,  // VolumetricLight fExposure
    shadowFar:        40.0,  // SHADOW_Element_9 far
    shadowBias:       0.001, // derived from shadow size 1024
  };

  public uniforms: {
    u_fillColor:        Float32Array;   // vec3
    u_opacity:          number;
    u_numRays:          number;
    u_pupilRadius:      number;
    u_focalIntensity:   number;
    u_time:             number;
    u_bloomStrength:    number;
    u_bloomRadius:      number;
    u_ambientIntensity: number;
    u_ambientColor:     Float32Array;   // vec3
    u_lightExposure:    number;
    u_shadowFar:        number;
    u_shadowBias:       number;
  };

  /**
   * Current animation time in seconds.
   * Set this every frame via the PixiJS Ticker to animate radial ray rotation.
   */
  public time: number;

  constructor(options?: CilEyeSDFFilterOptions) {
    const opts = { ...CilEyeSDFFilter.DEFAULT_OPTIONS, ...options };

    const gpuProgram = GpuProgram.from({
      vertex: {
        source: wgslVertex,
        entryPoint: 'mainVertex',
      },
      fragment: {
        source: CIL_EYE_WGSL,
        entryPoint: 'mainFragment',
      },
    });

    const glProgram = GlProgram.from({
      vertex,
      fragment: CIL_EYE_FRAGMENT,
      name: 'cil-eye-sdf-filter',
    });

    super({
      gpuProgram,
      glProgram,
      resources: {
        cilEyeUniforms: {
          u_fillColor:        { value: new Float32Array(opts.fillColor),        type: 'vec3<f32>' },
          u_opacity:          { value: opts.opacity,          type: 'f32' },
          u_numRays:          { value: opts.numRays,          type: 'f32' },
          u_pupilRadius:      { value: opts.pupilRadius,      type: 'f32' },
          u_focalIntensity:   { value: opts.focalIntensity,   type: 'f32' },
          u_time:             { value: opts.time,             type: 'f32' },
          u_bloomStrength:    { value: opts.bloomStrength,    type: 'f32' },
          u_bloomRadius:      { value: opts.bloomRadius,      type: 'f32' },
          u_ambientIntensity: { value: opts.ambientIntensity, type: 'f32' },
          u_ambientColor:     { value: new Float32Array(opts.ambientColor),     type: 'vec3<f32>' },
          u_lightExposure:    { value: opts.lightExposure,    type: 'f32' },
          u_shadowFar:        { value: opts.shadowFar,        type: 'f32' },
          u_shadowBias:       { value: opts.shadowBias,       type: 'f32' },
        },
      },
    });

    this.uniforms = this.resources.cilEyeUniforms.uniforms as typeof this.uniforms;
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
    this.uniforms.u_time = this.time;
    filterManager.applyFilter(this, input, output, clearMode);
  }

  // ── Uniform accessors ──────────────────────────────────────────────────────

  /** Eye fill colour as [r, g, b] 0–1. */
  get fillColor(): Float32Array { return this.uniforms.u_fillColor; }
  set fillColor(value: [number, number, number] | Float32Array) {
    this.uniforms.u_fillColor[0] = value[0];
    this.uniforms.u_fillColor[1] = value[1];
    this.uniforms.u_fillColor[2] = value[2];
  }

  /** Overall eye opacity (0–1). */
  get opacity(): number { return this.uniforms.u_opacity; }
  set opacity(value: number) { this.uniforms.u_opacity = value; }

  /** Number of radial rays. */
  get numRays(): number { return this.uniforms.u_numRays; }
  set numRays(value: number) { this.uniforms.u_numRays = value; }

  /** Pupil radius in [-1,1] normalised space. */
  get pupilRadius(): number { return this.uniforms.u_pupilRadius; }
  set pupilRadius(value: number) { this.uniforms.u_pupilRadius = value; }

  /** Ray brightness multiplier. */
  get focalIntensity(): number { return this.uniforms.u_focalIntensity; }
  set focalIntensity(value: number) { this.uniforms.u_focalIntensity = value; }

  /** AT bloom ring strength. */
  get bloomStrength(): number { return this.uniforms.u_bloomStrength; }
  set bloomStrength(value: number) { this.uniforms.u_bloomStrength = value; }

  /** AT bloom ring radius factor. */
  get bloomRadius(): number { return this.uniforms.u_bloomRadius; }
  set bloomRadius(value: number) { this.uniforms.u_bloomRadius = value; }
}
