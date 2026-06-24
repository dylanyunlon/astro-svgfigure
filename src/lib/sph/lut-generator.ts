// src/lib/sph/lut-generator.ts
// M624: 3-D LUT colour-grading generator — QoS driven tones
//
// Generates 3-D LUT textures (17³ cube) for the lut-pipeline.frag post-process
// pass and computes per-frame QoS zone weights for the u_qosWeights vec4
// uniform.
//
// Design
// ──────
// A 3-D colour LUT maps every possible input RGB triplet to an output RGB via
// trilinear interpolation inside a cube of pre-computed colour grades.  This
// module:
//
//   1. Generates one or more LUT cubes from named preset styles (CINEMATIC,
//      SENSOR, TOPO_ALERT, NEUTRAL) in CPU Float32 arrays.
//
//   2. Uploads the chosen LUT to a WebGL2 / WebGPU texture in the 17×(17×17)
//      horizontal-strip layout that lut-pipeline.frag expects.
//
//   3. Derives u_qosWeights (vec4) from a live QoSProfile at runtime, mapping
//      reliability + durability → zone index → weight.  The four zones are:
//        zone 0 = RELIABLE   + TRANSIENT_LOCAL   (warm persistent channels)
//        zone 1 = RELIABLE   + VOLATILE          (neutral workhorse channels)
//        zone 2 = BEST_EFFORT + VOLATILE         (cool sensor streams)
//        zone 3 = BEST_EFFORT + TRANSIENT_LOCAL  (alert / override channels)
//
//   4. Provides a LutPipelineState object that accumulates the four uniforms
//      the shader needs (strength, exposure, saturation, contrast) from QoS
//      semantics plus optional manual overrides.
//
// Usage
// ─────
//   const gen = new LutGenerator({ lutSize: 17 });
//   const lut  = gen.buildLut('CINEMATIC');
//
//   // On WebGL2 context creation:
//   const tex  = gen.uploadToWebGL(gl, lut);
//
//   // Per-frame, given the current channel's QoS profile:
//   const state = gen.deriveState(qosProfile, { strength: 0.85 });
//   // → state.qosWeights, state.exposure, state.saturation, state.contrast
//   //   ready to pass as uniforms
//
// References
// ──────────
//   • lut-pipeline.frag  — consuming shader (src/lib/shaders/lut-pipeline.frag)
//   • color-palette.ts   — QoS per-channel base colours (M566)
//   • qos-spatial-bridge.ts — Apollo QoS profiles & mapping utilities
//   • DaVinci Resolve LUT spec: .cube format, 17³ identity tables
//   • Colour grading math: GPU Gems 2, ch. 24 (NVIDIA 2005)
//
// Research: xiaodi #M624 — cell-pubsub-loop / lut-grading




// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

/** Named LUT style presets. */



import type { QoSProfile } from './types';
import type { QoSProfileName } from './qosSpatial';
import { QOS_PRESETS } from './qosSpatial';

export type LutStyleName =
  | 'NEUTRAL'        // Identity — no grade (debug / bypass)
  | 'CINEMATIC'      // Warm lifted film emulation (RELIABLE channels)
  | 'SENSOR'         // Cool desaturated high-contrast (BEST_EFFORT channels)
  | 'TOPO_ALERT'     // Pushed S-curve, crushed blacks, magenta tint (override)
  | 'DEEP_OCEAN'     // Teal-blue water grade for SPH visualisation
  | 'GOLDEN_HOUR';   // Warm amber sunset grade

/** A generated 3-D LUT cube. */
export interface LutCube {
  /** Edge length (N); typically 17. */
  size:    number;
  /**
   * RGBA data linearised in [R, G, B, A] order, covering all N³ samples.
   * Index: (r_i + g_i*N + b_i*N*N) * 4
   * Each component is in [0,1] as Float32.
   */
  data:    Float32Array;
  /** Human-readable name for debug display. */
  style:   LutStyleName;
}

/** Derived shader uniforms for one QoS profile + optional manual overrides. */
export interface LutPipelineState {
  /** vec4 per-zone intensity modulators → u_qosWeights */
  qosWeights:  [number, number, number, number];
  /** LUT blend strength [0,1] → u_lutStrength */
  strength:    number;
  /** Pre-LUT exposure in stops → u_exposure */
  exposure:    number;
  /** Post-LUT saturation scale [0,2] → u_saturation */
  saturation:  number;
  /** Post-LUT S-curve contrast [0,2] → u_contrast */
  contrast:    number;
}

/** Constructor options for LutGenerator. */
export interface LutGeneratorOptions {
  /** LUT edge size N.  Must be ≥ 2.  Recommended: 17 or 33.  Default: 17. */
  lutSize?: number;
}

/** Manual override values; partial — unset keys fall back to QoS-derived defaults. */
export interface LutStateOverrides {
  strength?:   number;
  exposure?:   number;
  saturation?: number;
  contrast?:   number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Zone classification
// ─────────────────────────────────────────────────────────────────────────────

/** Maps a QoSProfile to one of the four shader zones. */
export function classifyQoSZone(qos: QoSProfile): 0 | 1 | 2 | 3 {
  const reliable    = qos.reliability === 'RELIABLE';
  const persistent  = qos.durability  === 'TRANSIENT_LOCAL';

  if (reliable  && persistent)  return 0;  // RELIABLE   + TRANSIENT_LOCAL
  if (reliable  && !persistent) return 1;  // RELIABLE   + VOLATILE
  if (!reliable && !persistent) return 2;  // BEST_EFFORT + VOLATILE
  return 3;                                 // BEST_EFFORT + TRANSIENT_LOCAL
}

/** Returns the QoS zone index for a named Apollo CyberRT profile. */
export function classifyQoSProfileName(name: QoSProfileName): 0 | 1 | 2 | 3 {
  return classifyQoSZone(QOS_PRESETS[name]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (pure — no DOM / GPU deps)
// ─────────────────────────────────────────────────────────────────────────────

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** sRGB → linear. */
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** linear → sRGB. */
function linearToSrgb(c: number): number {
  c = clamp01(c);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Rec.709 luminance from linear RGB. */
function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Luma-preserving saturation scale. */
function saturateColor(r: number, g: number, b: number, s: number): [number, number, number] {
  const lum = luminance(r, g, b);
  return [
    clamp01(lum + (r - lum) * s),
    clamp01(lum + (g - lum) * s),
    clamp01(lum + (b - lum) * s),
  ];
}

/**
 * Filmic S-curve contrast, pivoted at 0.18 mid-grey.
 * strength 1.0 = neutral; >1 increases contrast.
 */
function sContrast(c: number, strength: number): number {
  const pivot = 0.18;
  const d = c - pivot;
  const s = d * strength;
  return clamp01(pivot + d / Math.sqrt(1 + s * s));
}

// ─────────────────────────────────────────────────────────────────────────────
// Grade functions (work in linear-light sRGB)
// Each takes linear [r,g,b] ∈ [0,1] and returns modified linear [r,g,b].
// These mirror the per-zone GLSL grade functions in lut-pipeline.frag.
// ─────────────────────────────────────────────────────────────────────────────

type RGB3 = [number, number, number];

/** Zone 0 — RELIABLE + TRANSIENT_LOCAL: warm lifted shadows. */
function gradeWarmLift(r: number, g: number, b: number): RGB3 {
  const lum = luminance(r, g, b);
  const shadowMask = 1 - Math.min(lum / 0.4, 1);
  r = clamp01(r + 0.12 * shadowMask * 0.35);
  g = clamp01(g + 0.07 * shadowMask * 0.35);
  b = clamp01(b + 0.02 * shadowMask * 0.35);
  return [clamp01(r * 1.04), clamp01(g), clamp01(b * 0.96)];
}

/** Zone 1 — RELIABLE + VOLATILE: neutral film. */
function gradeNeutralFilm(r: number, g: number, b: number): RGB3 {
  r = clamp01(0.02 + r * 0.97);
  g = clamp01(0.02 + g * 0.97);
  b = clamp01(0.02 + b * 0.97);
  return [clamp01(r * 1.015), clamp01(g * 1.005), clamp01(b * 0.98)];
}

/** Zone 2 — BEST_EFFORT + VOLATILE: cool sensor. */
function gradeCoolSensor(r: number, g: number, b: number): RGB3 {
  [r, g, b] = saturateColor(r, g, b, 0.70);
  b = clamp01(b * 1.12 + 0.04);
  r = clamp01(r * 0.92 - 0.01);
  return [sContrast(r, 1.4), sContrast(g, 1.4), sContrast(b, 1.4)];
}

/** Zone 3 — BEST_EFFORT + TRANSIENT_LOCAL: dramatic alert. */
function gradeDramaticAlert(r: number, g: number, b: number): RGB3 {
  r = clamp01((r - 0.04) / 0.96);
  g = clamp01((g - 0.04) / 0.96);
  b = clamp01((b - 0.04) / 0.96);
  r = sContrast(r, 1.8);
  g = sContrast(g, 1.8);
  b = sContrast(b, 1.8);
  return [clamp01(r * 1.06), clamp01(g * 0.96), clamp01(b * 1.04)];
}

/** Deep-ocean teal grade. */
function gradeDeepOcean(r: number, g: number, b: number): RGB3 {
  [r, g, b] = saturateColor(r, g, b, 0.80);
  b = clamp01(b * 1.18 + 0.06);
  g = clamp01(g * 1.06 + 0.02);
  r = clamp01(r * 0.88 - 0.02);
  return [sContrast(r, 1.2), sContrast(g, 1.2), sContrast(b, 1.1)];
}

/** Golden-hour warm sunset grade. */
function gradeGoldenHour(r: number, g: number, b: number): RGB3 {
  r = clamp01(r * 1.10 + 0.05);
  g = clamp01(g * 1.04 + 0.02);
  b = clamp01(b * 0.82 - 0.04);
  return [sContrast(r, 1.3), sContrast(g, 1.15), sContrast(b, 1.0)];
}

// Grade dispatch table
const GRADE_FNS: Record<LutStyleName, (r: number, g: number, b: number) => RGB3> = {
  NEUTRAL:      (r, g, b) => [r, g, b] as RGB3,
  CINEMATIC:    gradeWarmLift,
  SENSOR:       gradeCoolSensor,
  TOPO_ALERT:   gradeDramaticAlert,
  DEEP_OCEAN:   gradeDeepOcean,
  GOLDEN_HOUR:  gradeGoldenHour,
};

// ─────────────────────────────────────────────────────────────────────────────
// LutGenerator
// ─────────────────────────────────────────────────────────────────────────────

export class LutGenerator {
  readonly lutSize: number;

  constructor({ lutSize = 17 }: LutGeneratorOptions = {}) {
    if (lutSize < 2) throw new RangeError(`lutSize must be ≥ 2; got ${lutSize}`);
    this.lutSize = lutSize;
  }

  // ── LUT construction ───────────────────────────────────────────────────────

  /**
   * Builds a 3-D LUT cube for the given style.
   *
   * The cube is filled by:
   *   for each (r_i, g_i, b_i) index triplet →
   *     normalise to [0,1] → apply grade fn in linear space → store sRGB
   *
   * The shader samples this sRGB-encoded cube (per DaVinci Resolve convention)
   * and converts back to linear internally for further processing.
   */
  buildLut(style: LutStyleName = 'NEUTRAL'): LutCube {
    const N    = this.lutSize;
    const data = new Float32Array(N * N * N * 4);
    const fn   = GRADE_FNS[style];

    for (let bi = 0; bi < N; bi++) {
      for (let gi = 0; gi < N; gi++) {
        for (let ri = 0; ri < N; ri++) {
          // Normalise indices → linear sRGB input [0,1]
          const rLin = srgbToLinear(ri / (N - 1));
          const gLin = srgbToLinear(gi / (N - 1));
          const bLin = srgbToLinear(bi / (N - 1));

          // Apply grade in linear space
          const [outR, outG, outB] = fn(rLin, gLin, bLin);

          // Convert output back to sRGB for storage (DaVinci convention)
          const idx = (ri + gi * N + bi * N * N) * 4;
          data[idx + 0] = clamp01(linearToSrgb(outR));
          data[idx + 1] = clamp01(linearToSrgb(outG));
          data[idx + 2] = clamp01(linearToSrgb(outB));
          data[idx + 3] = 1.0;
        }
      }
    }

    return { size: N, data, style };
  }

  /**
   * Builds a blended LUT by mixing two styles.
   * @param styleA  First style name.
   * @param styleB  Second style name.
   * @param t       Blend factor [0,1]; 0 = full styleA, 1 = full styleB.
   */
  buildBlendedLut(styleA: LutStyleName, styleB: LutStyleName, t: number): LutCube {
    t = clamp01(t);
    const lutA = this.buildLut(styleA);
    const lutB = this.buildLut(styleB);
    const N    = this.lutSize;
    const data = new Float32Array(N * N * N * 4);
    const inv  = 1 - t;

    for (let i = 0; i < data.length; i++) {
      data[i] = inv * lutA.data[i] + t * lutB.data[i];
    }

    return { size: N, data, style: 'NEUTRAL' };
  }

  // ── Texture upload — WebGL2 ────────────────────────────────────────────────

  /**
   * Uploads a LutCube to WebGL2 as a 2-D RGBA32F texture in the horizontal-
   * strip layout that lut-pipeline.frag expects:
   *
   *   width  = N
   *   height = N × N
   *   row v = b_slice * N + g_index  (bottom-left origin in OpenGL)
   *   column u = r_index
   *
   * The caller owns the returned WebGLTexture and must delete it when done.
   */
  uploadToWebGL(gl: WebGL2RenderingContext, lut: LutCube): WebGLTexture {
    const N     = lut.size;
    const strip = new Float32Array(N * N * N * 4);

    // Re-layout: cube index (ri, gi, bi) → strip index (u=ri, v=bi*N+gi)
    for (let bi = 0; bi < N; bi++) {
      for (let gi = 0; gi < N; gi++) {
        for (let ri = 0; ri < N; ri++) {
          const srcIdx  = (ri + gi * N + bi * N * N) * 4;
          const stripV  = bi * N + gi;   // row in strip texture
          const dstIdx  = (ri + stripV * N) * 4;
          strip[dstIdx + 0] = lut.data[srcIdx + 0];
          strip[dstIdx + 1] = lut.data[srcIdx + 1];
          strip[dstIdx + 2] = lut.data[srcIdx + 2];
          strip[dstIdx + 3] = lut.data[srcIdx + 3];
        }
      }
    }

    const tex = gl.createTexture();
    if (!tex) throw new Error('LutGenerator: WebGL2 createTexture failed');

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0,
      gl.RGBA32F,
      N,          // width  = N
      N * N,      // height = N²
      0,
      gl.RGBA, gl.FLOAT,
      strip,
    );
    // LINEAR filtering → GPU handles trilinear interpolation across the strip
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return tex;
  }

  /**
   * Uploads a LutCube to WebGPU as a GPUTexture (rgba32float, 2-D).
   * Returns { texture, view } ready for binding.
   *
   * Requires the device to support the 'float32-filterable' feature for
   * smooth trilinear filtering in the shader.  If unsupported, the caller
   * should fall back to rgba16float or nearest sampling.
   */
  uploadToWebGPU(
    device: GPUDevice,
    lut:    LutCube,
  ): { texture: GPUTexture; view: GPUTextureView } {
    const N     = lut.size;
    const strip = new Float32Array(N * N * N * 4);

    for (let bi = 0; bi < N; bi++) {
      for (let gi = 0; gi < N; gi++) {
        for (let ri = 0; ri < N; ri++) {
          const srcIdx = (ri + gi * N + bi * N * N) * 4;
          const stripV = bi * N + gi;
          const dstIdx = (ri + stripV * N) * 4;
          strip[dstIdx + 0] = lut.data[srcIdx + 0];
          strip[dstIdx + 1] = lut.data[srcIdx + 1];
          strip[dstIdx + 2] = lut.data[srcIdx + 2];
          strip[dstIdx + 3] = lut.data[srcIdx + 3];
        }
      }
    }

    const texture = device.createTexture({
      label:  `lut-${lut.style}`,
      size:   [N, N * N, 1],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    device.queue.writeTexture(
      { texture },
      strip.buffer,
      { bytesPerRow: N * 4 * 4, rowsPerImage: N * N },
      [N, N * N, 1],
    );

    const view = texture.createView({ label: `lut-${lut.style}-view` });
    return { texture, view };
  }

  // ── QoS → shader uniform derivation ───────────────────────────────────────

  /**
   * Derives LutPipelineState from a live QoSProfile.
   *
   * Zone classification rules (mirror lut-pipeline.frag zone comments):
   *   zone 0 — RELIABLE   + TRANSIENT_LOCAL  → strong warm grade, slight +exposure
   *   zone 1 — RELIABLE   + VOLATILE         → neutral grade, balanced
   *   zone 2 — BEST_EFFORT + VOLATILE        → strong cool grade, -saturation
   *   zone 3 — BEST_EFFORT + TRANSIENT_LOCAL → alert grade, -exposure (crushed)
   *
   * MPS (messages per second) and historyDepth modulate strength and saturation
   * within the zone.
   */
  deriveState(
    qos:       QoSProfile,
    overrides: LutStateOverrides = {},
  ): LutPipelineState {
    const zone = classifyQoSZone(qos);

    // Base QoS weight: concentrated on the classified zone
    const qosWeights: [number, number, number, number] = [0, 0, 0, 0];
    qosWeights[zone] = 1.0;

    // MPS influence: high throughput channels blend slightly toward zone 2 (sensor cool)
    const mpsNorm = qos.mps > 0 ? clamp01(qos.mps / 120.0) : 0;
    if (zone !== 2 && mpsNorm > 0.3) {
      const bleed = (mpsNorm - 0.3) * 0.5;  // up to 0.35 bleed toward sensor
      qosWeights[zone]  = clamp01(qosWeights[zone]  - bleed);
      qosWeights[2]     = clamp01(qosWeights[2]     + bleed);
    }

    // History depth influence: deep queues blend toward warm persistent zone 0
    const depthNorm = clamp01(qos.historyDepth / 1000.0);
    if (zone !== 0 && depthNorm > 0.1) {
      const bleed = depthNorm * 0.25;
      qosWeights[zone]  = clamp01(qosWeights[zone]  - bleed);
      qosWeights[0]     = clamp01(qosWeights[0]     + bleed);
    }

    // Zone-driven base uniforms
    let strength   = 0.75;
    let exposure   = 0.0;
    let saturation = 1.0;
    let contrast   = 1.0;

    switch (zone) {
      case 0: // RELIABLE + TRANSIENT_LOCAL — warm, stable
        strength   = 0.85;
        exposure   = +0.15;   // slightly bright (confidence)
        saturation = 1.10;
        contrast   = 1.05;
        break;
      case 1: // RELIABLE + VOLATILE — neutral, workhorse
        strength   = 0.70;
        exposure   = 0.0;
        saturation = 1.00;
        contrast   = 1.00;
        break;
      case 2: // BEST_EFFORT + VOLATILE — cool, urgent sensor
        strength   = 0.80;
        exposure   = -0.10;   // slightly darker (uncertainty)
        saturation = 0.85;
        contrast   = 1.15;
        break;
      case 3: // BEST_EFFORT + TRANSIENT_LOCAL — dramatic alert
        strength   = 0.90;
        exposure   = -0.20;   // crushed (drama)
        saturation = 1.20;
        contrast   = 1.30;
        break;
    }

    // Priority modulates exposure by ±0.1 stop (priority 0-3 → -0.05 … +0.25)
    const priority = (qos as unknown as { priority?: number }).priority ?? 1;
    exposure += (priority - 1) * 0.05;  // priority 1 = 0, priority 3 = +0.1, priority 0 = -0.05

    return {
      qosWeights,
      strength:   overrides.strength   ?? clamp01(strength),
      exposure:   overrides.exposure   ?? exposure,
      saturation: overrides.saturation ?? saturation,
      contrast:   overrides.contrast   ?? contrast,
    };
  }

  /**
   * Convenience wrapper: derives state from a named Apollo CyberRT profile.
   */
  deriveStateFromName(
    name:      QoSProfileName,
    overrides: LutStateOverrides = {},
  ): LutPipelineState {
    return this.deriveState(QOS_PRESETS[name], overrides);
  }

  // ── LUT style recommendation ───────────────────────────────────────────────

  /**
   * Recommends a LUT style based on QoS zone and visual context.
   * Useful when building a multi-zone composited scene where each channel
   * wants its own LUT variant.
   */
  recommendStyle(qos: QoSProfile): LutStyleName {
    const zone = classifyQoSZone(qos);
    const styleMap: Record<0 | 1 | 2 | 3, LutStyleName> = {
      0: 'CINEMATIC',   // warm persistent
      1: 'GOLDEN_HOUR', // neutral-warm balanced
      2: 'SENSOR',      // cool sensor
      3: 'TOPO_ALERT',  // dramatic override
    };
    return styleMap[zone];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton convenience export (default 17³ cube)
// ─────────────────────────────────────────────────────────────────────────────

/** Shared LutGenerator instance with default 17³ resolution. */
export const lutGenerator = new LutGenerator({ lutSize: 17 });

// ─────────────────────────────────────────────────────────────────────────────
// Pre-built LUT cache (lazy)
// ─────────────────────────────────────────────────────────────────────────────

const _lutCache = new Map<LutStyleName, LutCube>();

/**
 * Returns (and lazily builds) a cached LutCube for the given style.
 * Thread-safe for single-threaded JS environments; call from main thread only.
 */
export function getCachedLut(style: LutStyleName): LutCube {
  let cube = _lutCache.get(style);
  if (!cube) {
    cube = lutGenerator.buildLut(style);
    _lutCache.set(style, cube);
  }
  return cube;
}

/**
 * Pre-warms all LUT styles into the cache.
 * Call once during scene initialisation to avoid first-frame stalls.
 */
export function prewarmLutCache(): void {
  const styles: LutStyleName[] = [
    'NEUTRAL', 'CINEMATIC', 'SENSOR', 'TOPO_ALERT', 'DEEP_OCEAN', 'GOLDEN_HOUR',
  ];
  for (const s of styles) getCachedLut(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// .cube file export (DaVinci Resolve / Adobe interchange format)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Serialises a LutCube to the Adobe/DaVinci .cube text format.
 * Useful for exporting grades to NLEs or offline colour tools.
 *
 * @returns Multi-line string; write to a .cube file.
 */
export function exportDotCube(lut: LutCube): string {
  const N    = lut.size;
  const lines: string[] = [
    `# Generated by lut-generator.ts — ${lut.style}`,
    `TITLE "${lut.style}"`,
    `LUT_3D_SIZE ${N}`,
    '',
  ];

  // .cube order: R iterates fastest, then G, then B
  for (let bi = 0; bi < N; bi++) {
    for (let gi = 0; gi < N; gi++) {
      for (let ri = 0; ri < N; ri++) {
        const idx = (ri + gi * N + bi * N * N) * 4;
        const r   = lut.data[idx + 0].toFixed(6);
        const g   = lut.data[idx + 1].toFixed(6);
        const b   = lut.data[idx + 2].toFixed(6);
        lines.push(`${r} ${g} ${b}`);
      }
    }
  }

  return lines.join('\n');
}
