/**
 * passes/KawaseBlurPass.ts
 *
 * WebGL2 port of pixijs-filters KawaseBlurFilter.
 * Upstream: upstream/pixijs-filters/src/kawase-blur/KawaseBlurFilter.ts
 *
 * Algorithm — "An investigation of fast real-time GPU-based image blur algorithms"
 *   Intel Software, 2014 (Masaki Kawase dual-filter):
 *
 *   downSample pass (per level i):
 *     uOffset = kernels[i] + 0.5  (normalised to texel units)
 *     4-tap box sample at (±uOffset.x, ±uOffset.y)
 *     → half-resolution output
 *
 *   upSample pass (per level i, coarse → fine):
 *     8-tap bilinear tent kernel
 *     → accumulated additive blend back to full resolution
 *
 * This class drives the Nuke/NukePass pipeline instead of pixi.js Filter
 * infrastructure, keeping the same kernel mathematics verbatim.
 *
 * Usage:
 * ```ts
 * const kblur = new KawaseBlurPass(nuke, inputRT, outputRT, {
 *   strength: 4,
 *   quality:  3,
 *   clamp:    false,
 * });
 * nuke.render(); // passes registered automatically
 * kblur.setStrength(8);
 * kblur.dispose();
 * ```
 */

import { NukePass, FULLSCREEN_VERT_SRC } from '../NukePass';
import type { RenderTarget } from '../NukePass';
import type { Nuke } from '../Nuke';

// ── Kawase downSample fragment ─────────────────────────────────────────────────
//
// Ported verbatim from:
//   upstream/pixijs-filters/src/kawase-blur/kawase-blur.frag  (GL)
//   upstream/pixijs-filters/src/kawase-blur/kawase-blur-clamp.frag (GL, clamp variant)
//
// Input:  u_input       — source texture (RGBA)
//         u_inputRes    — source texture resolution (pixels)
//         u_offset      — kernel offset (texel-space scalar, i.e. kernels[i] + 0.5)
// Output: gl_FragCoord → output FBO (half resolution)

const KAWASE_DOWN_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;   // output (smaller) pixel dimensions
uniform vec2      u_inputRes;     // input  (larger)  pixel dimensions
uniform float     u_offset;       // kernels[i] + 0.5, in texel units

out vec4 fragColor;

void main() {
  // UV in output space → same UV in input space (same normalised coordinate)
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  // Offset expressed in input-texture UV space
  vec2 o     = u_offset / u_inputRes;

  // 4-tap Kawase box kernel (matches kawase-blur.frag exactly)
  vec4 color =
      texture(u_input, uv + vec2(-o.x,  o.y)) +  // top-left
      texture(u_input, uv + vec2( o.x,  o.y)) +  // top-right
      texture(u_input, uv + vec2( o.x, -o.y)) +  // bottom-right
      texture(u_input, uv + vec2(-o.x, -o.y));   // bottom-left

  fragColor = color * 0.25;
}
`;

// Clamp-edge variant (matches kawase-blur-clamp.frag)
const KAWASE_DOWN_CLAMP_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;
uniform vec2      u_inputRes;
uniform float     u_offset;
uniform vec4      u_inputClamp;   // xy = min UV, zw = max UV

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  vec2 o  = u_offset / u_inputRes;

  vec4 color =
      texture(u_input, clamp(uv + vec2(-o.x,  o.y), u_inputClamp.xy, u_inputClamp.zw)) +
      texture(u_input, clamp(uv + vec2( o.x,  o.y), u_inputClamp.xy, u_inputClamp.zw)) +
      texture(u_input, clamp(uv + vec2( o.x, -o.y), u_inputClamp.xy, u_inputClamp.zw)) +
      texture(u_input, clamp(uv + vec2(-o.x, -o.y), u_inputClamp.xy, u_inputClamp.zw));

  fragColor = color * 0.25;
}
`;

// ── Kawase upSample fragment ───────────────────────────────────────────────────
//
// 8-tap bilinear tent kernel (Dual Kawase upsample).
// Used in both BloomPass (existing) and here as first-class export.
// Adds the upsampled colour onto u_accum (additive accumulation).

const KAWASE_UP_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;      // coarser level
uniform sampler2D u_accum;      // previous (finer) accumulated result
uniform vec2      u_resolution; // output (finer) pixel dimensions
uniform vec2      u_inputRes;   // input  (coarser) pixel dimensions
uniform float     u_offset;     // kernel offset (texel units in input space)
uniform float     u_weight;     // blend weight for u_accum (default 1.0)

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  vec2 o  = u_offset / u_inputRes;

  // 8-tap bilinear tent (matches BloomPass.KAWASE_UP_FRAG)
  vec4 sum =
      texture(u_input, uv + vec2(-o.x * 2.0,  0.0       )) +
      texture(u_input, uv + vec2(-o.x,         o.y       )) +
      texture(u_input, uv + vec2( 0.0,          o.y * 2.0)) +
      texture(u_input, uv + vec2( o.x,          o.y       )) +
      texture(u_input, uv + vec2( o.x * 2.0,   0.0       )) +
      texture(u_input, uv + vec2( o.x,         -o.y       )) +
      texture(u_input, uv + vec2( 0.0,         -o.y * 2.0)) +
      texture(u_input, uv + vec2(-o.x,         -o.y       ));

  vec4 blurred = sum / 8.0;
  vec4 accum   = texture(u_accum, uv);

  // Additive accumulation — mirrors AT HydraBloom upsample chain
  fragColor = blurred + accum * u_weight;
}
`;

// ── Passthrough (single-pass, quality==1) ─────────────────────────────────────
//
// When quality === 1 the upstream KawaseBlurFilter does a single applyFilter
// call with offset = kernels[0] + 0.5.  We replicate this with a direct
// 4-tap downsample writing into the final output RT.

const KAWASE_SINGLE_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;
uniform vec2      u_inputRes;
uniform float     u_offset;

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  vec2 o  = u_offset / u_inputRes;

  vec4 color =
      texture(u_input, uv + vec2(-o.x,  o.y)) +
      texture(u_input, uv + vec2( o.x,  o.y)) +
      texture(u_input, uv + vec2( o.x, -o.y)) +
      texture(u_input, uv + vec2(-o.x, -o.y));

  fragColor = color * 0.25;
}
`;

// ── KawaseBlurPassOptions ─────────────────────────────────────────────────────

export interface KawaseBlurPassOptions {
  /**
   * Blur strength — scalar kernel size fed to _generateKernels().
   * Matches KawaseBlurFilter.strength.  Default: 4.
   */
  strength?: number;

  /**
   * Number of downsample/upsample passes.
   * Matches KawaseBlurFilter.quality.  Default: 3.
   */
  quality?: number;

  /**
   * Clamp texture edges (avoids dark halos at borders).
   * Matches KawaseBlurFilter.clamp.  Default: false.
   */
  clamp?: boolean;

  /**
   * Advanced: explicit kernel array, overrides strength+quality.
   * Matches KawaseBlurFilter.kernels.
   */
  kernels?: number[];

  /**
   * Pixel size multiplier — scales the UV offset.
   * Matches KawaseBlurFilter.pixelSize.  Default: {x:1,y:1}.
   */
  pixelSize?: { x: number; y: number } | number;
}

// ── KawaseBlurPass ────────────────────────────────────────────────────────────

/**
 * KawaseBlurPass — Kawase dual-filter blur as a Nuke post-processing pass.
 *
 * Core downSample / upSample engine.
 * Used directly for standalone blur, and also consumed by KawaseBloomPass
 * which mirrors AdvancedBloomFilter's dependency on KawaseBlurFilter.
 *
 * Kernel generation mirrors KawaseBlurFilter._generateKernels():
 *   kernels[0] = blur
 *   kernels[i] = blur - i * (blur / quality)   for i in [1, quality)
 */
export class KawaseBlurPass {
  readonly nuke: Nuke;
  readonly name = 'kawase-blur';

  private _strength: number;
  private _quality:  number;
  private _kernels:  number[]  = [];
  private _clamp:    boolean;
  private _pixelSize: { x: number; y: number };

  private _passes: NukePass[] = [];
  private _rts:    RenderTarget[] = [];

  /** Input render target (not owned; caller manages lifetime). */
  readonly input:  RenderTarget;
  /** Output render target (not owned; caller manages lifetime). */
  readonly output: RenderTarget;

  constructor(
    nuke: Nuke,
    input: RenderTarget,
    output: RenderTarget,
    opts: KawaseBlurPassOptions = {}
  ) {
    this.nuke   = nuke;
    this.input  = input;
    this.output = output;

    this._strength  = opts.strength  ?? 4;
    this._quality   = Math.max(1, Math.round(opts.quality  ?? 3));
    this._clamp     = opts.clamp     ?? false;

    const ps = opts.pixelSize ?? 1;
    this._pixelSize = typeof ps === 'number' ? { x: ps, y: ps } : ps;

    if (opts.kernels && opts.kernels.length > 0) {
      this._kernels  = opts.kernels;
      this._quality  = opts.kernels.length;
      this._strength = Math.max(...opts.kernels);
    } else {
      this._generateKernels();
    }

    this._build();
  }

  // ── Public accessors (mirror KawaseBlurFilter) ────────────────────────────

  get strength(): number { return this._strength; }
  set strength(v: number) {
    this._strength = v;
    this._generateKernels();
    this._rebuild();
  }

  get quality(): number { return this._quality; }
  set quality(v: number) {
    this._quality = Math.max(1, Math.round(v));
    this._generateKernels();
    this._rebuild();
  }

  get kernels(): number[] { return this._kernels; }
  set kernels(v: number[]) {
    if (Array.isArray(v) && v.length > 0) {
      this._kernels  = v;
      this._quality  = v.length;
      this._strength = Math.max(...v);
    } else {
      this._kernels  = [0];
      this._quality  = 1;
    }
    this._rebuild();
  }

  get clamp(): boolean { return this._clamp; }

  get pixelSizeX(): number { return this._pixelSize.x; }
  set pixelSizeX(v: number) { this._pixelSize.x = v; this._syncUniforms(); }

  get pixelSizeY(): number { return this._pixelSize.y; }
  set pixelSizeY(v: number) { this._pixelSize.y = v; this._syncUniforms(); }

  /** Convenience setter — mirrors KawaseBlurFilter.strength setter name. */
  setStrength(v: number): this { this.strength = v; return this; }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  dispose(): void {
    for (const p of this._passes) {
      this.nuke.removePass(p);
      p.dispose();
    }
    this._passes = [];
    this._rts    = [];
  }

  // ── Private — kernel generation (mirrors _generateKernels in upstream) ─────

  /**
   * Auto-generate kernels from strength + quality.
   * Upstream KawaseBlurFilter._generateKernels():
   *   kernels[0] = blur
   *   kernels[i] = blur - i * step   where step = blur / quality
   */
  private _generateKernels(): void {
    const blur    = this._strength;
    const quality = this._quality;
    const kernels: number[] = [blur];

    if (blur > 0) {
      const step = blur / quality;
      let k = blur;
      for (let i = 1; i < quality; i++) {
        k -= step;
        kernels.push(k);
      }
    }

    this._kernels = kernels;
  }

  // ── Private — pass chain build ─────────────────────────────────────────────

  private _rebuild(): void {
    this.dispose();
    this._build();
  }

  /**
   * Build the Nuke pass chain.
   *
   * Single-pass path (quality === 1 or strength === 0):
   *   KAWASE_SINGLE: input → output
   *
   * Multi-pass path (quality > 1):
   *   Down chain: input → down0 → down1 → … → down[quality-2]
   *   Up   chain: down[quality-2] → … → down0 → output
   *
   * This mirrors the ping-pong loop inside KawaseBlurFilter.apply().
   */
  private _build(): void {
    const { nuke, input, output, _kernels, _quality, _clamp } = this;
    const w = input.width;
    const h = input.height;

    // ── Single-pass path ──────────────────────────────────────────────────
    if (_quality === 1 || this._strength === 0) {
      const offset = _kernels[0] + 0.5;
      const pass = new NukePass({
        name:    `${this.name}:single`,
        fragSrc: KAWASE_SINGLE_FRAG,
        vertSrc: FULLSCREEN_VERT_SRC,
        input,
        output,
        uniforms: {
          u_inputRes: [w, h] as [number, number],
          u_offset:   this._scaledOffset(offset),
        },
      });
      this._addPass(pass);
      return;
    }

    // ── Multi-pass: build intermediate RTs ────────────────────────────────
    // We need (quality - 1) ping-pong intermediates, all at the same size as
    // input (the Kawase algorithm operates in-place at constant resolution;
    // unlike BloomPass we do NOT halve resolution each level — the quality
    // parameter controls the *number of blur passes*, not pyramid levels).
    const downRTs: RenderTarget[] = [input];

    for (let i = 0; i < _quality - 1; i++) {
      const rt = this._allocRT(`${this.name}:ping${i}`, w, h);
      downRTs.push(rt);
    }

    // ── Down passes: source → target for each kernel except the last ─────
    const fragSrc = _clamp ? KAWASE_DOWN_CLAMP_FRAG : KAWASE_DOWN_FRAG;
    const last    = _quality - 1;

    for (let i = 0; i < last; i++) {
      const offset  = _kernels[i] + 0.5;
      const src     = downRTs[i];
      const tgt     = downRTs[i + 1];

      const uniforms: Record<string, number | [number, number]> = {
        u_inputRes: [w, h] as [number, number],
        u_offset:   this._scaledOffset(offset),
      };
      if (_clamp) {
        // uInputClamp: UV margins of 0.5 texel on each edge
        (uniforms as Record<string, unknown>)['u_inputClamp'] = [
          0.5 / w, 0.5 / h, 1.0 - 0.5 / w, 1.0 - 0.5 / h,
        ] as [number, number, number, number];
      }

      const pass = new NukePass({
        name:    `${this.name}:down${i}`,
        fragSrc,
        vertSrc: FULLSCREEN_VERT_SRC,
        input:   src,
        output:  tgt,
        uniforms,
      });
      this._addPass(pass);
    }

    // ── Final pass: last kernel, last intermediate → output ───────────────
    const finalOffset = _kernels[last] + 0.5;
    const finalUniforms: Record<string, unknown> = {
      u_inputRes: [w, h] as [number, number],
      u_offset:   this._scaledOffset(finalOffset),
    };
    if (_clamp) {
      finalUniforms['u_inputClamp'] = [
        0.5 / w, 0.5 / h, 1.0 - 0.5 / w, 1.0 - 0.5 / h,
      ] as [number, number, number, number];
    }

    const finalPass = new NukePass({
      name:    `${this.name}:down${last}`,
      fragSrc,
      vertSrc: FULLSCREEN_VERT_SRC,
      input:   downRTs[last],
      output,
      uniforms: finalUniforms as Record<string, number | [number, number]>,
    });
    this._addPass(finalPass);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Scale a kernel offset by pixelSize (matches KawaseBlurFilter.apply():
   *   offset = kernels[i] + 0.5
   *   uniforms.uOffset[0] = offset * uvX   where uvX = pixelSize.x / width
   * Here we return the absolute texel offset; the shader divides by u_inputRes.
   */
  private _scaledOffset(offset: number): number {
    // Average pixelSize components (anisotropic pixelSize is unusual;
    // pass both x/y if needed via vec2 variant — here kept as scalar).
    return offset * Math.max(this._pixelSize.x, this._pixelSize.y);
  }

  private _allocRT(name: string, w: number, h: number): RenderTarget {
    const rt = this.nuke.createRT({ name, width: w, height: h });
    this._rts.push(rt);
    return rt;
  }

  private _addPass(pass: NukePass): void {
    this._passes.push(pass);
    this.nuke.addPass(pass);
  }

  /** Live-sync uniforms after pixelSize change (no full rebuild needed). */
  private _syncUniforms(): void {
    for (const pass of this._passes) {
      if ('u_offset' in pass.uniforms) {
        // Re-derive the per-pass offset from its name.
        const m = pass.name.match(/:(?:down|single)(\d*)$/);
        if (!m) continue;
        const idx    = m[1] !== '' ? parseInt(m[1], 10) : 0;
        const kernel = this._kernels[idx] ?? this._kernels[this._kernels.length - 1];
        pass.uniforms['u_offset'] = this._scaledOffset(kernel + 0.5);
      }
    }
  }
}
