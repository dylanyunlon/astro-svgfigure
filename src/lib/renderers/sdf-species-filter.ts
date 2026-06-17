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
