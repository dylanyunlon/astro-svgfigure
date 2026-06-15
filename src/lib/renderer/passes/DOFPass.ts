/**
 * passes/DOFPass.ts — Depth of Field post-processing pass
 *
 * AT DOF module port.
 * Upstream:
 *   upstream/pixijs-engine/src/fx/nuke/passes/DOFPass.ts
 *   upstream/pixijs-engine/src/fx/dof/BokehBlur.ts
 *
 * Algorithm:
 *   1. CoC (Circle of Confusion) pass
 *      — reads depth texture + focal params → per-pixel blur radius
 *   2. Bokeh horizontal blur   — separable Gaussian weighted by CoC
 *   3. Bokeh vertical blur     — second axis of separable filter
 *   4. Composite               — lerp between sharp scene and blurred by CoC mask
 *
 * Parameters (defaults sourced from channels/physics/at_uil_params.json):
 *   focalZ           — world-space focal distance (depth at focus)
 *                      AT: HomeSceneVFX_home_uDOF[0] = 0.72
 *   nearTransition   — near-field blur fade distance (world units)
 *                      AT: HomeSceneVFX_home_uDOF[1] = 0.8
 *   farTransition    — far-field blur fade distance (world units)
 *                      AT: HomeSceneVFX_home_uDOF[2] = 0.3
 *   contrast         — sharpens the near/far transition curve
 *                      AT: HomeSceneVFX_home_uDOFContrast avg([1.28, 2.41]) ≈ 1.85
 *   maxCoc           — maximum CoC radius in pixels
 *                      AT: HomeSceneVFX_home_uDOF[3] * 10 = 10
 *   blurSteps        — Gaussian sample count per axis (default 8)
 */

import { NukePass } from '../NukePass';
import type { RenderTarget } from '../NukePass';
import type { Nuke } from '../Nuke';

// ── GLSL sources ──────────────────────────────────────────────────────────────

const DOF_VERT = /* glsl */ `#version 300 es
precision highp float;
void main() {
  float x = float((gl_VertexID & 1) << 1) - 1.0;
  float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

// ── 1. CoC pass ───────────────────────────────────────────────────────────────

const COC_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_depth;       // linear depth [0, 1] in red channel
uniform vec2      u_resolution;
uniform float     u_focalZ;         // focal depth (same units as depth texture)
uniform float     u_nearTransition; // world units for near fade
uniform float     u_farTransition;  // world units for far fade
uniform float     u_contrast;       // curve sharpness (1 = linear, >1 = harder)
uniform float     u_maxCoc;         // maximum CoC radius in pixels

out vec4 fragColor; // rg = signed CoC (r=near, g=far), ba = unused

float signedCoc(float depth) {
  float diff = depth - u_focalZ;

  // Near field: diff < 0
  float nearCoc = clamp(-diff / max(u_nearTransition, 0.0001), 0.0, 1.0);
  nearCoc = pow(nearCoc, u_contrast);

  // Far field: diff > 0
  float farCoc  = clamp( diff / max(u_farTransition,  0.0001), 0.0, 1.0);
  farCoc  = pow(farCoc,  u_contrast);

  return (nearCoc - farCoc); // negative near, positive far
}

void main() {
  vec2 uv    = gl_FragCoord.xy / u_resolution;
  float d    = texture(u_depth, uv).r;
  float coc  = signedCoc(d) * u_maxCoc;

  // Pack: r = nearCoc (0–1), g = farCoc (0–1), b = raw coc in pixels.
  float nearMask = max(-coc, 0.0) / u_maxCoc;
  float farMask  = max( coc, 0.0) / u_maxCoc;
  fragColor = vec4(nearMask, farMask, coc / u_maxCoc * 0.5 + 0.5, 1.0);
}
`;

// ── 2+3. Separable Bokeh blur (H then V) ─────────────────────────────────────

const BOKEH_BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_input;      // scene colour (or result of H pass)
uniform sampler2D u_coc;        // CoC map
uniform vec2      u_resolution;
uniform vec2      u_direction;  // (1,0) for H pass, (0,1) for V pass
uniform float     u_maxCoc;
uniform int       u_steps;      // Gaussian sample count (each side)

out vec4 fragColor;

// Gaussian weight
float gauss(float x, float sigma) {
  return exp(-(x * x) / (2.0 * sigma * sigma));
}

void main() {
  vec2  uv      = gl_FragCoord.xy / u_resolution;
  vec2  texel   = 1.0 / u_resolution;
  float cocHere = (texture(u_coc, uv).b * 2.0 - 1.0) * u_maxCoc;
  float radius  = abs(cocHere);

  vec4  colSum  = texture(u_input, uv);
  float wSum    = 1.0;

  float sigma = max(radius / 3.0, 0.5);

  for (int i = 1; i <= u_steps; i++) {
    float fi  = float(i);
    vec2  off = u_direction * texel * (fi / float(u_steps)) * radius;
    float w   = gauss(fi, sigma);

    vec2 uvA = uv + off;
    vec2 uvB = uv - off;

    // Only blur toward pixels that also have CoC (avoids sharp edge bleed).
    float cocA = abs((texture(u_coc, uvA).b * 2.0 - 1.0) * u_maxCoc);
    float cocB = abs((texture(u_coc, uvB).b * 2.0 - 1.0) * u_maxCoc);
    float wA = w * step(fi - 1.0, cocA); // include if cocA covers this tap
    float wB = w * step(fi - 1.0, cocB);

    colSum += texture(u_input, uvA) * wA;
    colSum += texture(u_input, uvB) * wB;
    wSum   += wA + wB;
  }

  fragColor = colSum / wSum;
}
`;

// ── 4. DOF composite ──────────────────────────────────────────────────────────

const DOF_COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;

uniform sampler2D u_sharp;      // original scene
uniform sampler2D u_blurred;    // bokeh-blurred scene
uniform sampler2D u_coc;        // CoC map
uniform vec2      u_resolution;

out vec4 fragColor;

void main() {
  vec2  uv      = gl_FragCoord.xy / u_resolution;
  vec4  sharp   = texture(u_sharp,   uv);
  vec4  blurred = texture(u_blurred, uv);
  vec4  coc     = texture(u_coc,     uv);

  // Blend mask: near + far fields both get blur; in-focus stays sharp.
  float mask = max(coc.r, coc.g);
  fragColor  = mix(sharp, blurred, clamp(mask, 0.0, 1.0));
}
`;

// ── DOFPassConfig ─────────────────────────────────────────────────────────────

export interface DOFPassConfig {
  /** World-space focal depth.  AT: HomeSceneVFX_home_uDOF[0].  Default: 0.72. */
  focalZ?: number;
  /** Near-field transition distance.  AT: HomeSceneVFX_home_uDOF[1].  Default: 0.8. */
  nearTransition?: number;
  /** Far-field transition distance.  AT: HomeSceneVFX_home_uDOF[2].  Default: 0.3. */
  farTransition?: number;
  /** Curve contrast / sharpness.  AT: avg(HomeSceneVFX_home_uDOFContrast).  Default: 1.85. */
  contrast?: number;
  /** Maximum CoC radius in pixels.  AT: HomeSceneVFX_home_uDOF[3] * 10.  Default: 10. */
  maxCoc?: number;
  /** Gaussian sample count per axis.  Default: 8. */
  blurSteps?: number;
}

// ── DOFPass ───────────────────────────────────────────────────────────────────

/**
 * DOFPass — depth of field using per-pixel CoC + separable Bokeh blur.
 *
 * Requires a depth render target that stores linear depth in the R channel.
 *
 * ```ts
 * const dof = new DOFPass(nuke, sceneRT, depthRT, outputRT, {
 *   focalZ: 0.5,
 *   nearTransition: 0.15,
 *   farTransition: 0.3,
 *   contrast: 2.0,
 * });
 * ```
 */
export class DOFPass {
  readonly nuke: Nuke;
  readonly name = 'dof';

  focalZ:         number;
  nearTransition: number;
  farTransition:  number;
  contrast:       number;
  maxCoc:         number;
  blurSteps:      number;

  private _passes: NukePass[] = [];
  private _rts:    RenderTarget[] = [];

  constructor(
    nuke: Nuke,
    sceneInput: RenderTarget,
    depthInput: RenderTarget,
    output: RenderTarget,
    config: DOFPassConfig = {}
  ) {
    this.nuke           = nuke;
    this.focalZ         = config.focalZ         ?? 0.72;
    this.nearTransition = config.nearTransition  ?? 0.8;
    this.farTransition  = config.farTransition   ?? 0.3;
    this.contrast       = config.contrast        ?? 1.85;
    this.maxCoc         = config.maxCoc          ?? 10;
    this.blurSteps      = config.blurSteps       ?? 8;

    this._build(sceneInput, depthInput, output);
  }

  /** Live-update DOF parameters. */
  setUniforms(config: DOFPassConfig): void {
    if (config.focalZ         !== undefined) this.focalZ         = config.focalZ;
    if (config.nearTransition !== undefined) this.nearTransition = config.nearTransition;
    if (config.farTransition  !== undefined) this.farTransition  = config.farTransition;
    if (config.contrast       !== undefined) this.contrast       = config.contrast;
    if (config.maxCoc         !== undefined) this.maxCoc         = config.maxCoc;
    if (config.blurSteps      !== undefined) this.blurSteps      = config.blurSteps;
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

  private _build(
    sceneInput: RenderTarget,
    depthInput: RenderTarget,
    output: RenderTarget
  ): void {
    const { nuke } = this;
    const w = sceneInput.width;
    const h = sceneInput.height;

    // ── CoC pass ────────────────────────────────────────────────────────────
    const cocRT = this._allocRT('dof:coc', w, h);
    const cocPass = new NukePass({
      name:    'dof:coc',
      fragSrc: COC_FRAG,
      vertSrc: DOF_VERT,
      input:   depthInput,
      output:  cocRT,
      uniforms: {
        u_depth:          depthInput.texture,
        u_focalZ:         this.focalZ,
        u_nearTransition: this.nearTransition,
        u_farTransition:  this.farTransition,
        u_contrast:       this.contrast,
        u_maxCoc:         this.maxCoc,
      },
    });
    this._addPass(cocPass);

    // ── Bokeh H blur ────────────────────────────────────────────────────────
    const blurHRT = this._allocRT('dof:blurH', w, h);
    const blurHPass = new NukePass({
      name:    'dof:blurH',
      fragSrc: BOKEH_BLUR_FRAG,
      vertSrc: DOF_VERT,
      input:   sceneInput,
      output:  blurHRT,
      uniforms: {
        u_coc:       cocRT.texture,
        u_direction: [1, 0] as [number, number],
        u_maxCoc:    this.maxCoc,
        u_steps:     this.blurSteps,
      },
    });
    this._addPass(blurHPass);

    // ── Bokeh V blur ────────────────────────────────────────────────────────
    const blurVRT = this._allocRT('dof:blurV', w, h);
    const blurVPass = new NukePass({
      name:    'dof:blurV',
      fragSrc: BOKEH_BLUR_FRAG,
      vertSrc: DOF_VERT,
      input:   blurHRT,
      output:  blurVRT,
      uniforms: {
        u_coc:       cocRT.texture,
        u_direction: [0, 1] as [number, number],
        u_maxCoc:    this.maxCoc,
        u_steps:     this.blurSteps,
      },
    });
    this._addPass(blurVPass);

    // ── Composite ───────────────────────────────────────────────────────────
    const compositePass = new NukePass({
      name:    'dof:composite',
      fragSrc: DOF_COMPOSITE_FRAG,
      vertSrc: DOF_VERT,
      input:   sceneInput,
      output,
      uniforms: {
        u_sharp:   sceneInput.texture,
        u_blurred: blurVRT.texture,
        u_coc:     cocRT.texture,
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
    for (const pass of this._passes) {
      switch (pass.name) {
        case 'dof:coc':
          pass.uniforms['u_focalZ']         = this.focalZ;
          pass.uniforms['u_nearTransition'] = this.nearTransition;
          pass.uniforms['u_farTransition']  = this.farTransition;
          pass.uniforms['u_contrast']       = this.contrast;
          pass.uniforms['u_maxCoc']         = this.maxCoc;
          break;
        case 'dof:blurH':
        case 'dof:blurV':
          pass.uniforms['u_maxCoc']  = this.maxCoc;
          pass.uniforms['u_steps']   = this.blurSteps;
          break;
      }
    }
  }
}
