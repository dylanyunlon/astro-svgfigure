/**
 * passes/KawaseBloomPass.ts
 *
 * WebGL2 port of pixijs-filters AdvancedBloomFilter.
 * Upstream: upstream/pixijs-filters/src/advanced-bloom/AdvancedBloomFilter.ts
 *
 * Architecture mirrors AdvancedBloomFilter.apply() exactly:
 *
 *   Step 1 — ExtractBrightness pass
 *     Extracts pixels above `threshold` luminance into brightRT.
 *     Shader: extract-brightness.frag (ported verbatim as EXTRACT_BRIGHTNESS_FRAG).
 *
 *   Step 2 — KawaseBlur pass (via KawaseBlurPass)
 *     Blurs brightRT → bloomRT.
 *     Core: KawaseBlurPass (downSample/upSample engine from KawaseBlurFilter).
 *
 *   Step 3 — Composite pass
 *     Blends original scene + bloomRT with bloomScale/brightness controls.
 *     Shader: advanced-bloom.frag (ported verbatim as ADVANCED_BLOOM_FRAG).
 *
 * This is the canonical `AdvancedBloomFilter` dependency on `KawaseBlurFilter`
 * translated into the project's Nuke/NukePass pipeline.
 *
 * Usage:
 * ```ts
 * const bloom = new KawaseBloomPass(nuke, sceneRT, outputRT, {
 *   threshold:  0.5,
 *   bloomScale: 1.0,
 *   brightness: 1.0,
 *   blur:       8,
 *   quality:    4,
 * });
 * nuke.render();
 * bloom.setUniforms({ bloomScale: 1.5 });
 * bloom.dispose();
 * ```
 */

import { NukePass, FULLSCREEN_VERT_SRC } from '../NukePass';
import type { RenderTarget } from '../NukePass';
import type { Nuke } from '../Nuke';
import { KawaseBlurPass } from './KawaseBlurPass';

// ── Step 1: ExtractBrightness ─────────────────────────────────────────────────
//
// Ported verbatim from:
//   upstream/pixijs-filters/src/advanced-bloom/extract-brightness.frag
//
// Algorithm: (max + min) * 0.5 brightness — fast approximation of HSL lightness.
// Pixels below threshold are zeroed out.

const EXTRACT_BRIGHTNESS_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;
uniform float     u_threshold;

out vec4 fragColor;

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec4 color = texture(u_input, uv);

  // Fast HSL-lightness brightness (matches extract-brightness.frag exactly)
  float _max = max(max(color.r, color.g), color.b);
  float _min = min(min(color.r, color.g), color.b);
  float brightness = (_max + _min) * 0.5;

  // Zero pixels below threshold; pass bright pixels through unchanged
  fragColor = brightness > u_threshold ? color : vec4(0.0);
}
`;

// ── Step 3: AdvancedBloom composite ──────────────────────────────────────────
//
// Ported verbatim from:
//   upstream/pixijs-filters/src/advanced-bloom/advanced-bloom.frag
//
//   finalColor = (scene * brightness) + (bloomMap * bloomScale)
//
// Matches AdvancedBloomFilter.apply() — uniforms uBloomScale / uBrightness.

const ADVANCED_BLOOM_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;       // original scene (uTexture in upstream)
uniform sampler2D u_bloomMap;    // blurred bright map (uMapTexture in upstream)
uniform vec2      u_resolution;
uniform float     u_bloomScale;  // uBloomScale
uniform float     u_brightness;  // uBrightness

out vec4 fragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;

  vec4 color = texture(u_input, uv);
  // Apply brightness to base colour
  color.rgb *= u_brightness;

  // Bloom map: alpha channel zeroed (matches upstream vec4(bloomColor.rgb, 0.0))
  vec4 bloomColor = vec4(texture(u_bloomMap, uv).rgb, 0.0);
  bloomColor.rgb *= u_bloomScale;

  fragColor = color + bloomColor;
}
`;

// ── KawaseBloomPassOptions ────────────────────────────────────────────────────

export interface KawaseBloomPassOptions {
  /**
   * Luminance threshold — pixels below this are excluded from bloom.
   * Matches AdvancedBloomFilter.threshold.  Default: 0.5.
   */
  threshold?: number;

  /**
   * Bloom layer brightness multiplier.
   * Matches AdvancedBloomFilter.bloomScale.  Default: 1.
   */
  bloomScale?: number;

  /**
   * Base scene brightness multiplier.
   * Matches AdvancedBloomFilter.brightness.  Default: 1.
   */
  brightness?: number;

  /**
   * KawaseBlur strength (scalar kernel size).
   * Matches AdvancedBloomFilter constructor `blur`.  Default: 8.
   */
  blur?: number;

  /**
   * KawaseBlur quality (number of passes).
   * Matches AdvancedBloomFilter constructor `quality`.  Default: 4.
   */
  quality?: number;

  /**
   * Advanced: explicit Kawase kernels (overrides blur + quality).
   * Matches AdvancedBloomFilter constructor `kernels`.
   */
  kernels?: number[];

  /**
   * Pixel size passed down to KawaseBlurPass.
   * Matches AdvancedBloomFilter.pixelSize.  Default: {x:1,y:1}.
   */
  pixelSize?: { x: number; y: number } | number;
}

// ── KawaseBloomPass ───────────────────────────────────────────────────────────

/**
 * KawaseBloomPass — AdvancedBloomFilter ported to the Nuke pipeline.
 *
 * Internally instantiates a KawaseBlurPass (the `_blurFilter` equivalent)
 * and adds an ExtractBrightness NukePass (the `_extractFilter` equivalent)
 * plus a composite NukePass.
 *
 * All AdvancedBloomFilter public properties are mirrored.
 */
export class KawaseBloomPass {
  readonly nuke: Nuke;
  readonly name = 'kawase-bloom';

  // Mirrors AdvancedBloomFilter public fields
  bloomScale: number;
  brightness: number;

  private _threshold:  number;
  private _extractPass: NukePass;
  private _blurPass:    KawaseBlurPass;
  private _compositePass: NukePass;

  // Internal RTs (owned by this class)
  private _brightRT: RenderTarget;
  private _bloomRT:  RenderTarget;

  constructor(
    nuke: Nuke,
    sceneInput: RenderTarget,
    output: RenderTarget,
    opts: KawaseBloomPassOptions = {}
  ) {
    this.nuke       = nuke;
    this.bloomScale = opts.bloomScale ?? 1;
    this.brightness = opts.brightness ?? 1;
    this._threshold = opts.threshold  ?? 0.5;

    const w = sceneInput.width;
    const h = sceneInput.height;

    // ── Step 1: ExtractBrightness (mirrors _extractFilter in AdvancedBloomFilter)
    this._brightRT = nuke.createRT({ name: `${this.name}:bright`, width: w, height: h });

    this._extractPass = new NukePass({
      name:    `${this.name}:extract`,
      fragSrc: EXTRACT_BRIGHTNESS_FRAG,
      vertSrc: FULLSCREEN_VERT_SRC,
      input:   sceneInput,
      output:  this._brightRT,
      uniforms: {
        u_threshold: this._threshold,
      },
    });
    nuke.addPass(this._extractPass);

    // ── Step 2: KawaseBlur (mirrors _blurFilter in AdvancedBloomFilter)
    // AdvancedBloomFilter sets:
    //   strength: options.kernels ?? options.blur
    //   quality:  options.kernels ? undefined : options.quality
    this._bloomRT = nuke.createRT({ name: `${this.name}:bloom`, width: w, height: h });

    this._blurPass = new KawaseBlurPass(nuke, this._brightRT, this._bloomRT, {
      strength:  opts.kernels ? Math.max(...opts.kernels) : (opts.blur ?? 8),
      quality:   opts.kernels ? opts.kernels.length       : (opts.quality ?? 4),
      kernels:   opts.kernels,
      pixelSize: opts.pixelSize,
    });

    // ── Step 3: Composite (mirrors final filterManager.applyFilter in AdvancedBloomFilter)
    this._compositePass = new NukePass({
      name:    `${this.name}:composite`,
      fragSrc: ADVANCED_BLOOM_FRAG,
      vertSrc: FULLSCREEN_VERT_SRC,
      input:   sceneInput,
      output,
      uniforms: {
        u_bloomMap:   this._bloomRT.texture,
        u_bloomScale: this.bloomScale,
        u_brightness: this.brightness,
      },
    });
    nuke.addPass(this._compositePass);
  }

  // ── Public API (mirrors AdvancedBloomFilter) ──────────────────────────────

  /**
   * Luminance threshold for brightness extraction.
   * Mirrors AdvancedBloomFilter.threshold getter/setter.
   */
  get threshold(): number { return this._threshold; }
  set threshold(v: number) {
    this._threshold = v;
    this._extractPass.uniforms['u_threshold'] = v;
  }

  /**
   * KawaseBlur kernels.
   * Mirrors AdvancedBloomFilter.kernels getter/setter.
   */
  get kernels(): number[] { return this._blurPass.kernels; }
  set kernels(v: number[]) { this._blurPass.kernels = v; }

  /**
   * Blur strength (KawaseBlurPass.strength).
   * Mirrors AdvancedBloomFilter.blur getter/setter.
   */
  get blur(): number { return this._blurPass.strength; }
  set blur(v: number) { this._blurPass.strength = v; }

  /**
   * Blur quality (number of Kawase passes).
   * Mirrors AdvancedBloomFilter.quality getter/setter.
   */
  get quality(): number { return this._blurPass.quality; }
  set quality(v: number) { this._blurPass.quality = v; }

  /**
   * Pixel size of the Kawase blur.
   * Mirrors AdvancedBloomFilter.pixelSize getter/setter.
   */
  get pixelSizeX(): number { return this._blurPass.pixelSizeX; }
  set pixelSizeX(v: number) { this._blurPass.pixelSizeX = v; }

  get pixelSizeY(): number { return this._blurPass.pixelSizeY; }
  set pixelSizeY(v: number) { this._blurPass.pixelSizeY = v; }

  /**
   * Batch-update mutable uniforms — no pass chain rebuild.
   * Mirrors BloomPass.setUniforms() pattern.
   */
  setUniforms(opts: {
    bloomScale?: number;
    brightness?: number;
    threshold?:  number;
  }): this {
    if (opts.bloomScale !== undefined) {
      this.bloomScale = opts.bloomScale;
      this._compositePass.uniforms['u_bloomScale'] = opts.bloomScale;
    }
    if (opts.brightness !== undefined) {
      this.brightness = opts.brightness;
      this._compositePass.uniforms['u_brightness'] = opts.brightness;
    }
    if (opts.threshold !== undefined) {
      this.threshold = opts.threshold;
    }
    return this;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  dispose(): void {
    // Composite pass
    this.nuke.removePass(this._compositePass);
    this._compositePass.dispose();

    // KawaseBlurPass (owns its own NukePasses internally)
    this._blurPass.dispose();

    // ExtractBrightness pass
    this.nuke.removePass(this._extractPass);
    this._extractPass.dispose();

    // Internal RTs — note: Nuke does not own these; we created them via createRT
    // so they are tracked in Nuke's RT registry — explicitly destroy via nuke
    // if Nuke exposes destroyRT, otherwise they'll be GC'd with the FBO.
    // (Nuke.ts createRT stores RTs internally; disposal of the Nuke itself
    //  will clean them up.  Individual RT disposal is not exposed in the public
    //  API, matching the pattern in BloomPass.ts.)
  }
}
