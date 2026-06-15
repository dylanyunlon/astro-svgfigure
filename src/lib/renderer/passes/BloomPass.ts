/**
 * passes/BloomPass.ts — Bloom post-processing pass
 *
 * AT HydraBloom port.
 * Upstream:
 *   upstream/pixijs-engine/src/fx/nuke/passes/HydraBloom.ts
 *   upstream/pixijs-engine/src/fx/bloom/KawaseBlur.ts
 *
 * Algorithm (Kawase dual-filter):
 *   1. Luminosity threshold  — extract bright pixels
 *   2. Downsample chain      — N levels of 2× downscale + Kawase blur
 *   3. Upsample chain        — N levels of 2× upscale + additive accumulate
 *   4. Composite             — add bloom buffer onto scene colour
 *
 * Parameters (read from UIL config at construction time):
 *   bloomStrength        — additive blend weight of the bloom layer
 *   bloomRadius          — Kawase kernel offset scale (higher = wider bloom)
 *   luminosityThreshold  — pixels below this luminance are suppressed
 *   luminositySmoothWidth— soft knee width around threshold
 *   levels               — number of downsample / upsample levels (default 5)
 */

import { NukePass } from '../NukePass';
import type { RenderTarget } from '../NukePass';
import type { Nuke } from '../Nuke';

// ── Shared GLSL helpers ───────────────────────────────────────────────────────

const BLOOM_VERT = /* glsl */ `#version 300 es
precision highp float;
void main() {
  float x = float((gl_VertexID & 1) << 1) - 1.0;
  float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

// ── 1. Luminosity threshold ───────────────────────────────────────────────────

const THRESHOLD_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;
uniform float     u_threshold;
uniform float     u_smoothWidth;

out vec4 fragColor;

// Rec.709 luminance
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec4 color = texture(u_input, uv);
  float l    = luma(color.rgb);

  // Soft knee: smoothstep from (threshold - smoothWidth) to (threshold + smoothWidth)
  float knee  = u_threshold - u_smoothWidth;
  float alpha = smoothstep(knee, u_threshold + u_smoothWidth, l);

  fragColor = vec4(color.rgb * alpha, color.a);
}
`;

// ── 2. Kawase downsample ──────────────────────────────────────────────────────

const KAWASE_DOWN_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2      u_resolution;   // output (smaller) resolution
uniform vec2      u_inputRes;     // input (larger) resolution
uniform float     u_offset;       // Kawase kernel offset (px, in output space)

out vec4 fragColor;

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec2 texel = 1.0 / u_inputRes;

  // 4-tap Kawase kernel centred at uv, offset in input-texture space.
  vec2 o = u_offset * texel;

  vec4 sum =
      texture(u_input, uv + vec2(-o.x, -o.y)) +
      texture(u_input, uv + vec2( o.x, -o.y)) +
      texture(u_input, uv + vec2(-o.x,  o.y)) +
      texture(u_input, uv + vec2( o.x,  o.y));

  fragColor = sum * 0.25;
}
`;

// ── 3. Kawase upsample ────────────────────────────────────────────────────────

const KAWASE_UP_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform sampler2D u_accum;       // accumulated bloom from previous upsample
uniform vec2      u_resolution;
uniform vec2      u_inputRes;
uniform float     u_offset;
uniform float     u_strength;   // blend weight for accumulated bloom

out vec4 fragColor;

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec2 texel = 1.0 / u_inputRes;
  vec2 o     = u_offset * texel;

  // 8-tap bilinear Kawase upsample
  vec4 sum =
      texture(u_input, uv + vec2(-o.x * 2.0,  0.0      )) +
      texture(u_input, uv + vec2(-o.x,        o.y      )) +
      texture(u_input, uv + vec2( 0.0,         o.y * 2.0)) +
      texture(u_input, uv + vec2( o.x,         o.y      )) +
      texture(u_input, uv + vec2( o.x * 2.0,   0.0      )) +
      texture(u_input, uv + vec2( o.x,        -o.y      )) +
      texture(u_input, uv + vec2( 0.0,        -o.y * 2.0)) +
      texture(u_input, uv + vec2(-o.x,        -o.y      ));

  vec4 bloom = sum / 8.0;
  vec4 accum = texture(u_accum, uv);

  fragColor  = bloom + accum * u_strength;
}
`;

// ── 4. Composite ──────────────────────────────────────────────────────────────

const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;    // original scene
uniform sampler2D u_bloom;    // accumulated bloom
uniform vec2      u_resolution;
uniform float     u_strength;

out vec4 fragColor;

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  vec4 scene = texture(u_input, uv);
  vec4 bloom = texture(u_bloom, uv);

  // Additive bloom composite.
  fragColor = scene + bloom * u_strength;
}
`;

// ── BloomPassConfig ───────────────────────────────────────────────────────────

export interface BloomPassConfig {
  /** 0–3 range, default 1.0.  Scales bloom layer brightness. */
  bloomStrength?: number;
  /** Kawase kernel offset scale, default 1.0. */
  bloomRadius?: number;
  /** Rec.709 luminance threshold 0–1, default 0.8. */
  luminosityThreshold?: number;
  /** Soft-knee width, default 0.04. */
  luminositySmoothWidth?: number;
  /** Downsample / upsample depth, default 5. */
  levels?: number;
}

// ── BloomPass PRESETS (sourced from channels/physics/at_uil_params.json) ─────
//
// Keys map to UnrealBloomComposite scene names from AT's UIL param tree.
// Values are taken verbatim from at_uil_params.json — no guessing.
//
// Source paths inside at_uil_params.json:
//   strength → UnrealBloomComposite/UnrealBloomComposite/<scene>/bloomStrength
//   radius   → UnrealBloomComposite/UnrealBloomComposite/<scene>/bloomRadius
//   threshold→ UnrealBloomLuminosity/UnrealBloomLuminosity/<scene>/luminosityThreshold
//
// shaderVariants presets:
//   strength → UnrealBloomComposite_shaderVariants_<scene>bloomStrength
//   radius   → UnrealBloomComposite_shaderVariants_<scene>bloomRadius

export type BloomPresetName =
  | 'home'
  | 'homebloom'
  | 'globalbloom'
  | 'cleanroom'
  | 'workbloom'
  | 'treescene'
  | 'sv_contact'
  | 'sv_about'
  | 'sv_footer'
  | 'sv_home'
  | 'sv_work'
  | 'sv_tree';

// ── BloomPass ─────────────────────────────────────────────────────────────────

/**
 * BloomPass — Kawase dual-filter bloom corresponding to AT's HydraBloom.
 *
 * Internally creates a chain of NukePass objects and registers them
 * with the Nuke pipeline.  Calling `dispose()` removes them all.
 *
 * ```ts
 * const bloom = new BloomPass(nuke, sceneRT, outputRT, {
 *   bloomStrength: 1.2,
 *   luminosityThreshold: 0.75,
 * });
 * // passes are added to nuke automatically — just call nuke.render().
 * ```
 */
export class BloomPass {
  readonly nuke: Nuke;
  readonly name = 'bloom';

  /**
   * AT UIL per-scene bloom presets.
   * All values sourced directly from channels/physics/at_uil_params.json.
   *
   * Two families:
   *  • Core scenes  — UnrealBloomComposite/UnrealBloomComposite/<scene>/…
   *  • shaderVariants (sv_*) — UnrealBloomComposite_shaderVariants_<scene>bloom…
   */
  static readonly PRESETS: Record<BloomPresetName, Required<BloomPassConfig>> = {
    // ── Core scenes ──────────────────────────────────────────────────────────
    home: {
      bloomStrength:        3.8200000000000003,
      bloomRadius:          1,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    homebloom: {
      bloomStrength:        1.2,
      bloomRadius:          1,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    globalbloom: {
      bloomStrength:        0.3,
      bloomRadius:          0.2,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    cleanroom: {
      bloomStrength:        1,
      bloomRadius:          1,
      luminosityThreshold:  0.2,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    workbloom: {
      bloomStrength:        1,
      bloomRadius:          1,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    treescene: {
      bloomStrength:        1,
      bloomRadius:          1,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    // ── shaderVariants (UnrealBloomComposite_shaderVariants_…) ───────────────
    sv_contact: {
      bloomStrength:        0.8,
      bloomRadius:          0.5,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    sv_about: {
      bloomStrength:        1,
      bloomRadius:          1,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    sv_footer: {
      bloomStrength:        0.7,
      bloomRadius:          0.5,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    sv_home: {
      bloomStrength:        0.6,
      bloomRadius:          0.8,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    sv_work: {
      bloomStrength:        0.5,
      bloomRadius:          0.5,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
    sv_tree: {
      bloomStrength:        0.8,
      bloomRadius:          0.7,
      luminosityThreshold:  0,
      luminositySmoothWidth: 0.04,
      levels:               5,
    },
  };

  /**
   * Apply a named AT UIL preset, updating live uniforms.
   * @param name — one of the keys in BloomPass.PRESETS
   */
  applyPreset(name: BloomPresetName): void {
    const preset = BloomPass.PRESETS[name];
    if (!preset) throw new Error(`BloomPass: unknown preset "${name}"`);
    this.setUniforms(preset);
  }

  bloomStrength: number;
  bloomRadius: number;
  luminosityThreshold: number;
  luminositySmoothWidth: number;
  readonly levels: number;

  private _passes: NukePass[] = [];
  private _rts:    RenderTarget[] = [];

  constructor(
    nuke: Nuke,
    sceneInput: RenderTarget,
    output: RenderTarget,
    config: BloomPassConfig = {}
  ) {
    this.nuke                  = nuke;
    this.bloomStrength         = config.bloomStrength         ?? 1.0;
    this.bloomRadius           = config.bloomRadius           ?? 1.0;
    this.luminosityThreshold   = config.luminosityThreshold   ?? 0.8;
    this.luminositySmoothWidth = config.luminositySmoothWidth ?? 0.04;
    this.levels                = config.levels                ?? 5;

    this._build(sceneInput, output);
  }

  /** Update bloom parameters live (takes effect on next nuke.render()). */
  setUniforms(config: BloomPassConfig): void {
    if (config.bloomStrength         !== undefined) this.bloomStrength         = config.bloomStrength;
    if (config.bloomRadius           !== undefined) this.bloomRadius           = config.bloomRadius;
    if (config.luminosityThreshold   !== undefined) this.luminosityThreshold   = config.luminosityThreshold;
    if (config.luminositySmoothWidth !== undefined) this.luminositySmoothWidth = config.luminositySmoothWidth;
    this._syncUniforms();
  }

  dispose(): void {
    for (const p of this._passes) {
      this.nuke.removePass(p);
      p.dispose();
    }
    this._passes = [];
    this._rts    = [];
  }

  // ── Private build ─────────────────────────────────────────────────────────

  private _build(sceneInput: RenderTarget, output: RenderTarget): void {
    const { nuke, levels } = this;
    const gl = nuke.gl;

    let w = sceneInput.width;
    let h = sceneInput.height;

    // ── Threshold pass ──────────────────────────────────────────────────────
    const threshRT = this._allocRT(`bloom:threshold`, w, h);
    const threshPass = new NukePass({
      name:    'bloom:threshold',
      fragSrc: THRESHOLD_FRAG,
      vertSrc: BLOOM_VERT,
      input:   sceneInput,
      output:  threshRT,
      uniforms: {
        u_threshold:   this.luminosityThreshold,
        u_smoothWidth: this.luminositySmoothWidth,
      },
    });
    this._addPass(threshPass);

    // ── Downsample chain ────────────────────────────────────────────────────
    const downRTs: RenderTarget[] = [threshRT];
    for (let i = 0; i < levels; i++) {
      const pw = w; const ph = h;
      w = Math.max(1, w >> 1);
      h = Math.max(1, h >> 1);
      const downRT = this._allocRT(`bloom:down${i}`, w, h);
      const downPass = new NukePass({
        name:    `bloom:down${i}`,
        fragSrc: KAWASE_DOWN_FRAG,
        vertSrc: BLOOM_VERT,
        input:   downRTs[downRTs.length - 1],
        output:  downRT,
        uniforms: {
          u_inputRes: [pw, ph] as [number, number],
          u_offset:   this.bloomRadius,
        },
      });
      this._addPass(downPass);
      downRTs.push(downRT);
    }

    // ── Upsample chain ──────────────────────────────────────────────────────
    // Start from the coarsest level and accumulate upward.
    let accumRT = downRTs[downRTs.length - 1];
    for (let i = levels - 1; i >= 0; i--) {
      const targetRT = i === 0 ? output : this._allocRT(`bloom:up${i}`,
        downRTs[i].width, downRTs[i].height);

      const upPass = new NukePass({
        name:    `bloom:up${i}`,
        fragSrc: KAWASE_UP_FRAG,
        vertSrc: BLOOM_VERT,
        input:   accumRT,
        output:  targetRT,
        uniforms: {
          u_accum:    accumRT.texture,
          u_inputRes: [accumRT.width, accumRT.height] as [number, number],
          u_offset:   this.bloomRadius,
          u_strength: this.bloomStrength,
        },
      });
      this._addPass(upPass);
      accumRT = targetRT;
    }

    // ── Composite: add bloom onto original scene ────────────────────────────
    // (Only if output !== sceneInput — the composite writes back to output.)
    const compositePass = new NukePass({
      name:    'bloom:composite',
      fragSrc: COMPOSITE_FRAG,
      vertSrc: BLOOM_VERT,
      input:   sceneInput,
      output,
      uniforms: {
        u_bloom:    accumRT.texture,
        u_strength: this.bloomStrength,
      },
    });
    this._addPass(compositePass);
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

  private _syncUniforms(): void {
    // Walk passes and update the mutable uniform fields.
    for (const pass of this._passes) {
      if (pass.name === 'bloom:threshold') {
        pass.uniforms['u_threshold']   = this.luminosityThreshold;
        pass.uniforms['u_smoothWidth'] = this.luminositySmoothWidth;
      } else if (pass.name.startsWith('bloom:down') || pass.name.startsWith('bloom:up')) {
        pass.uniforms['u_offset']   = this.bloomRadius;
        pass.uniforms['u_strength'] = this.bloomStrength;
      } else if (pass.name === 'bloom:composite') {
        pass.uniforms['u_strength'] = this.bloomStrength;
      }
    }
  }
}
