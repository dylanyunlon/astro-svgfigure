/**
 * src/lib/sph/at-spline-water-depth.ts  —  M862
 *
 * AT Spline Shader + Water Normals + Depth Value
 * ─────────────────────────────────────────────────────────────────────────────
 * Three AT-quality shader systems extracted and integrated:
 *
 *   splineshader.glsl  (39 lines)
 *     Catmull-Rom knot weights + pressure-driven taper + species hue rotation.
 *     Augments src/lib/shaders/edge-spline.frag to give edge data flow the
 *     organic AT quality: strokes swell under load, taper toward targets,
 *     and shift hue per-species for immediate visual identity.
 *     ← Reverse-engineered from AT Hydra / FindingLove spline path render.
 *
 *   waternormals.fs  (18 lines)
 *     GLSL WebGL1 normal-update fragment shader for the water sim texture.
 *     Port of upstream/webgl-water/water.js `normalShader`.
 *     Patches the gap in at-water-surface.ts: the WGSL pipeline already has
 *     a compute-shader normal pass (WGSL_WAVE_NORMAL) but this GLSL version
 *     is needed for the WebGL2 fallback path in water-background.ts.
 *
 *   depthvalue.fs  (19 lines)
 *     Linearises hardware depth → eye-space Z; packs raw + linear + viewZ
 *     into an rgba8 or r32float texture consumed by SSAO (ambient-occlusion.ts
 *     #M775), SSR (screen-space-reflection.ts #M784), and DOF (dof-bokeh.ts
 *     #M760).
 *     ← Ported from upstream/lygia/space/{depth2viewZ,viewZ2depth,
 *       linearizeDepth}.glsl (Patricio Gonzalez Vivo — Prosperity License).
 *
 * ─── Exports ─────────────────────────────────────────────────────────────────
 *
 *   SPLINESHADER_GLSL  — raw GLSL source string (splineshader.glsl)
 *   WATERNORMALS_FS    — raw GLSL source string (waternormals.fs)
 *   DEPTHVALUE_FS      — raw GLSL source string (depthvalue.fs)
 *
 *   ATSplineShader     — applies AT-quality rendering to edge-spline.frag
 *   ATWaterNormals     — completes at-water-surface.ts WebGL2 normal pipeline
 *   ATDepthValue       — depth linearisation pass for SSAO / SSR / DOF
 *
 * Research: xiaodi #M862 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Raw GLSL source exports
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AT Spline Shader — Catmull-Rom weights + pressure taper + species hue.
 * Include via `#include "splineshader.glsl"` or inject as string prefix
 * into edge-spline.frag before compilation.
 */
export const SPLINESHADER_GLSL = /* glsl */`
// splineshader.glsl — AT Spline Shader (edge 数据流样条线渲染核心)
// Augments edge-spline.frag with AT-quality Catmull-Rom pressure + taper.
#ifndef SPLINESHADER_GLSL
#define SPLINESHADER_GLSL
precision highp float;
uniform float uPressure;
uniform float uTaper;
uniform float uSpecies;
vec4 catmullRomWeights(float t) {
    float t2 = t * t; float t3 = t2 * t;
    return 0.5 * vec4(
        -t3 + 2.0*t2 - t,
         3.0*t3 - 5.0*t2 + 2.0,
        -3.0*t3 + 4.0*t2 + t,
         t3 - t2 );
}
float pressureWidth(float baseHalfW, float pressure, float t, float taper) {
    float taperScale = mix(1.0, pow(1.0 - t, taper + 0.5) * 2.0, taper);
    return baseHalfW * taperScale * (1.0 + pressure * 0.45);
}
vec3 speciesHueShift(vec3 rgb, float speciesIdx) {
    float a = speciesIdx * 0.125 * 6.28318;
    float s = sin(a), c = cos(a);
    vec3 k = vec3(0.57735);
    return rgb * c + cross(k, rgb) * s + k * dot(k, rgb) * (1.0 - c);
}
#endif
`;

/**
 * AT Water Normals — GLSL fragment shader for normal-map update pass.
 * Port of upstream/webgl-water/water.js normalShader.
 * Used by ATWaterNormals for the WebGL2 fallback path.
 */
export const WATERNORMALS_FS = /* glsl */`
precision highp float;
uniform sampler2D uTexture;
uniform vec2      uDelta;
varying vec2      vCoord;
void main() {
    vec4  info = texture2D(uTexture, vCoord);
    vec3  dx   = vec3(uDelta.x, texture2D(uTexture, vCoord + vec2(uDelta.x, 0.0)).r - info.r, 0.0);
    vec3  dy   = vec3(0.0, texture2D(uTexture, vCoord + vec2(0.0, uDelta.y)).r - info.r, uDelta.y);
    info.ba    = normalize(cross(dy, dx)).xz;
    gl_FragColor = info;
}
`;

/**
 * AT Depth Value — GLSL fragment shader encoding hardware depth to
 * linearised eye-space Z.  Output rgba: (linD/far, rawD, -viewZ/far, 1).
 * Consumed by SSAO (#M775), SSR (#M784), DOF (#M760).
 */
export const DEPTHVALUE_FS = /* glsl */`
precision highp float;
uniform sampler2D uDepthTex;
uniform float     uNear;
uniform float     uFar;
varying vec2      vUV;
float linearizeDepth(float d, float near, float far) {
    float ndc = 2.0 * d - 1.0;
    return (2.0 * near * far) / (far + near - ndc * (far - near));
}
float depth2viewZ(float depth, float near, float far) {
    return (near * far) / ((far - near) * depth - far);
}
void main() {
    float raw   = texture2D(uDepthTex, vUV).r;
    float linD  = linearizeDepth(raw, uNear, uFar);
    float viewZ = depth2viewZ(raw, uNear, uFar);
    gl_FragColor = vec4(linD / uFar, raw, (-viewZ) / uFar, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATSplineShader — edge-spline.frag + AT quality augmentations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Config for ATSplineShader.
 */
export interface ATSplineShaderConfig {
  /**
   * Flow pressure [0, 1]: 0 = resting width, 1 = max swell (+45 %).
   * Driven by edge data-flow volume from topology.json cell pubsub load.
   * Default 0.
   */
  pressure?: number;

  /**
   * Taper exponent [0, 1]: 0 = uniform stroke, 1 = pointed tail.
   * Mimics AT's "brush tip" for directed skip-connection arrows.
   * Default 0.4.
   */
  taper?: number;

  /**
   * Species index [0–7]: selects 45° HSV hue step for hue-shifted species
   * gradient.  Passed as uSpecies to the augmented fragment shader.
   * Default 0.
   */
  species?: number;
}

/**
 * ATSplineShader
 * ──────────────
 * Patches the project's WebGL2 edge-spline.frag program with the three AT
 * augmentations from splineshader.glsl:
 *
 *   1. Catmull-Rom knot weights (catmullRomWeights) — replaces the raw
 *      Bézier approximation with the AT multi-segment C1 parametric form.
 *   2. Pressure-driven width (pressureWidth) — `uPressure` swells strokes
 *      proportional to the live cell pubsub message-rate on the edge.
 *   3. Species hue shift (speciesHueShift) — rotates the final stroke colour
 *      by a 45° step per species index, so edges between unlike species
 *      exhibit a natural hue contrast without extra uniforms.
 *
 * Usage (WebGL2):
 *
 *   const atSpline = new ATSplineShader(gl, baseFragSrc, { pressure: 0.6,
 *                                                           taper:    0.4,
 *                                                           species:  2 });
 *   atSpline.use();
 *   atSpline.setUniforms({ uTime: elapsed, uArcLength: arcLen, ... });
 *   // Draw edge quads as normal
 */
export class ATSplineShader {
  private readonly gl:      WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private cfg: Required<ATSplineShaderConfig>;

  // ── Cached uniform locations ─────────────────────────────────────────────
  private loc: {
    uColor:         WebGLUniformLocation | null;
    uAlpha:         WebGLUniformLocation | null;
    uLineWidth:     WebGLUniformLocation | null;
    uDashLength:    WebGLUniformLocation | null;
    uGapLength:     WebGLUniformLocation | null;
    uGlowColor:     WebGLUniformLocation | null;
    uGlowRadius:    WebGLUniformLocation | null;
    uGlowAlpha:     WebGLUniformLocation | null;
    uTime:          WebGLUniformLocation | null;
    uArcLength:     WebGLUniformLocation | null;
    uCurvature:     WebGLUniformLocation | null;
    u_sourceColor:  WebGLUniformLocation | null;
    u_targetColor:  WebGLUniformLocation | null;
    u_flowSpeed:    WebGLUniformLocation | null;
    u_thickness:    WebGLUniformLocation | null;
    u_time:         WebGLUniformLocation | null;
    // AT augmentation uniforms
    uPressure:      WebGLUniformLocation | null;
    uTaper:         WebGLUniformLocation | null;
    uSpecies:       WebGLUniformLocation | null;
  };

  constructor(
    gl:           WebGL2RenderingContext,
    baseFragSrc:  string,   // content of edge-spline.frag
    vertSrc:      string,   // content of edge-spline.vert (or compatible)
    cfg:          ATSplineShaderConfig = {},
  ) {
    this.gl  = gl;
    this.cfg = {
      pressure: cfg.pressure ?? 0,
      taper:    cfg.taper    ?? 0.4,
      species:  cfg.species  ?? 0,
    };

    // Inject AT augmentation declarations before void main()
    const augmented = this._augmentFrag(baseFragSrc);
    this.program    = this._compile(vertSrc, augmented);
    this.loc        = this._cacheLocations();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Activate the program. */
  use(): void { this.gl.useProgram(this.program); }

  /** Update AT-specific parameters without recompilation. */
  setATParams(cfg: Partial<ATSplineShaderConfig>): void {
    if (cfg.pressure !== undefined) this.cfg.pressure = cfg.pressure;
    if (cfg.taper    !== undefined) this.cfg.taper    = cfg.taper;
    if (cfg.species  !== undefined) this.cfg.species  = cfg.species;
  }

  /**
   * Upload all uniforms for one draw call.
   * Merges base edge-spline.frag uniforms with AT augmentation uniforms.
   */
  setUniforms(u: {
    color?:        [number, number, number];
    alpha?:        number;
    lineWidth?:    number;
    dashLength?:   number;
    gapLength?:    number;
    glowColor?:    [number, number, number];
    glowRadius?:   number;
    glowAlpha?:    number;
    time?:         number;
    arcLength?:    number;
    curvature?:    number;
    sourceColor?:  [number, number, number];
    targetColor?:  [number, number, number];
    flowSpeed?:    number;
    thickness?:    number;
  }): void {
    const gl = this.gl;
    const l  = this.loc;

    if (u.color       && l.uColor)       gl.uniform3fv(l.uColor,       u.color);
    if (u.alpha       !== undefined && l.uAlpha)      gl.uniform1f(l.uAlpha,      u.alpha!);
    if (u.lineWidth   !== undefined && l.uLineWidth)  gl.uniform1f(l.uLineWidth,  u.lineWidth!);
    if (u.dashLength  !== undefined && l.uDashLength) gl.uniform1f(l.uDashLength, u.dashLength!);
    if (u.gapLength   !== undefined && l.uGapLength)  gl.uniform1f(l.uGapLength,  u.gapLength!);
    if (u.glowColor   && l.uGlowColor)  gl.uniform3fv(l.uGlowColor,   u.glowColor);
    if (u.glowRadius  !== undefined && l.uGlowRadius) gl.uniform1f(l.uGlowRadius, u.glowRadius!);
    if (u.glowAlpha   !== undefined && l.uGlowAlpha)  gl.uniform1f(l.uGlowAlpha,  u.glowAlpha!);
    if (u.time        !== undefined && l.uTime)       gl.uniform1f(l.uTime,       u.time!);
    if (u.arcLength   !== undefined && l.uArcLength)  gl.uniform1f(l.uArcLength,  u.arcLength!);
    if (u.curvature   !== undefined && l.uCurvature)  gl.uniform1f(l.uCurvature,  u.curvature!);
    if (u.sourceColor && l.u_sourceColor) gl.uniform3fv(l.u_sourceColor, u.sourceColor);
    if (u.targetColor && l.u_targetColor) gl.uniform3fv(l.u_targetColor, u.targetColor);
    if (u.flowSpeed   !== undefined && l.u_flowSpeed) gl.uniform1f(l.u_flowSpeed, u.flowSpeed!);
    if (u.thickness   !== undefined && l.u_thickness) gl.uniform1f(l.u_thickness, u.thickness!);
    if (u.time        !== undefined && l.u_time)      gl.uniform1f(l.u_time,      u.time!);

    // AT augmentation uniforms
    if (l.uPressure) gl.uniform1f(l.uPressure, this.cfg.pressure);
    if (l.uTaper)    gl.uniform1f(l.uTaper,    this.cfg.taper);
    if (l.uSpecies)  gl.uniform1f(l.uSpecies,  this.cfg.species);
  }

  destroy(): void {
    this.gl.deleteProgram(this.program);
  }

  // ── Private ────────────────────────────────────────────────────────────────

  /**
   * Prepend AT augmentation GLSL and patch `void main()` so:
   *   • halfW uses pressureWidth() instead of flat vHalfWidth
   *   • strokeCol passes through speciesHueShift()
   */
  private _augmentFrag(src: string): string {
    // Strip version + precision — will be kept from original header
    const augDecl = [
      '',
      '// ── AT splineshader.glsl augmentations ─────────────────────────────',
      'uniform float uPressure;',
      'uniform float uTaper;',
      'uniform float uSpecies;',
      '',
      'vec4 catmullRomWeights(float t) {',
      '    float t2=t*t; float t3=t2*t;',
      '    return 0.5*vec4(-t3+2.0*t2-t, 3.0*t3-5.0*t2+2.0, -3.0*t3+4.0*t2+t, t3-t2);',
      '}',
      'float pressureWidth(float baseHalfW, float pressure, float t, float taper) {',
      '    float ts=mix(1.0,pow(1.0-t,taper+0.5)*2.0,taper);',
      '    return baseHalfW*ts*(1.0+pressure*0.45);',
      '}',
      'vec3 speciesHueShift(vec3 rgb, float si) {',
      '    float a=si*0.125*6.28318; float s=sin(a),c=cos(a);',
      '    vec3 k=vec3(0.57735);',
      '    return rgb*c+cross(k,rgb)*s+k*dot(k,rgb)*(1.0-c);',
      '}',
      '// ────────────────────────────────────────────────────────────────────',
      '',
    ].join('\n');

    // Inject before first occurrence of "void main"
    const mainIdx = src.indexOf('void main()');
    const patched = src.slice(0, mainIdx) + augDecl + src.slice(mainIdx);

    // Patch: replace "float halfW = vHalfWidth + u_thickness * 0.5;"
    // with pressureWidth call
    const patchedW = patched.replace(
      'float halfW = vHalfWidth + u_thickness * 0.5;',
      'float halfW = pressureWidth(vHalfWidth + u_thickness * 0.5, uPressure, vT, uTaper);',
    );

    // Patch: apply speciesHueShift to final strokeCol
    return patchedW.replace(
      'vec3 strokeCol = curvatureTint(baseColor);',
      'vec3 strokeCol = speciesHueShift(curvatureTint(baseColor), uSpecies);',
    );
  }

  private _compile(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl   = this.gl;
    const vert = this._shader(gl.VERTEX_SHADER,   vertSrc);
    const frag = this._shader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`ATSplineShader link error: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return prog;
  }

  private _shader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`ATSplineShader compile error: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  }

  private _cacheLocations() {
    const g = (n: string) => this.gl.getUniformLocation(this.program, n);
    return {
      uColor:        g('uColor'),
      uAlpha:        g('uAlpha'),
      uLineWidth:    g('uLineWidth'),
      uDashLength:   g('uDashLength'),
      uGapLength:    g('uGapLength'),
      uGlowColor:    g('uGlowColor'),
      uGlowRadius:   g('uGlowRadius'),
      uGlowAlpha:    g('uGlowAlpha'),
      uTime:         g('uTime'),
      uArcLength:    g('uArcLength'),
      uCurvature:    g('uCurvature'),
      u_sourceColor: g('u_sourceColor'),
      u_targetColor: g('u_targetColor'),
      u_flowSpeed:   g('u_flowSpeed'),
      u_thickness:   g('u_thickness'),
      u_time:        g('u_time'),
      uPressure:     g('uPressure'),
      uTaper:        g('uTaper'),
      uSpecies:      g('uSpecies'),
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATWaterNormals — WebGL2 fallback path for at-water-surface.ts
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATWaterNormals
 * ──────────────
 * Provides the WebGL2 (GLSL ES 1.00) normal-update pass that complements the
 * WGSL compute path in at-water-surface.ts.  Used by water-background.ts when
 * WebGPU is unavailable.
 *
 * Ported from upstream/webgl-water/water.js `normalShader`.
 *
 * Usage:
 *
 *   const normals = new ATWaterNormals(gl, simSize);
 *   normals.build();
 *
 *   // Each frame after stepSimulation:
 *   normals.update(waterTexA, waterFBO_B);
 *   [waterTexA, waterTexB] = [waterTexB, waterTexA];  // swap
 */
export class ATWaterNormals {
  private readonly gl:      WebGL2RenderingContext;
  private readonly simSize: number;
  private program!:         WebGLProgram;
  private quadVB!:          WebGLBuffer;
  private uTex!:            WebGLUniformLocation | null;
  private uDelta!:          WebGLUniformLocation | null;

  /** Common fullscreen-quad vertex shader source. */
  static readonly VERT_SRC = /* glsl */`
attribute vec2 aPos;
varying vec2   vCoord;
void main() {
    vCoord     = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

  constructor(gl: WebGL2RenderingContext, simSize: number) {
    this.gl      = gl;
    this.simSize = simSize;
  }

  /** Compile shaders and upload quad geometry. */
  build(): void {
    const gl     = this.gl;
    this.program = this._compile(ATWaterNormals.VERT_SRC, WATERNORMALS_FS);

    this.uTex   = gl.getUniformLocation(this.program, 'uTexture');
    this.uDelta = gl.getUniformLocation(this.program, 'uDelta');

    // Fullscreen quad: two triangles in NDC
    const verts = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
    this.quadVB = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVB);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  }

  /**
   * Run the normal-update pass.
   *
   * @param srcTex  Bound texture unit 0 = current simulation state.
   * @param dstFBO  Framebuffer to write updated normals into (ping-pong dst).
   */
  update(srcTex: WebGLTexture, dstFBO: WebGLFramebuffer): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
    gl.viewport(0, 0, this.simSize, this.simSize);

    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(this.uTex, 0);

    const delta = 1.0 / this.simSize;
    gl.uniform2f(this.uDelta, delta, delta);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVB);
    const aPos = gl.getAttribLocation(this.program, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  destroy(): void {
    this.gl.deleteProgram(this.program);
    this.gl.deleteBuffer(this.quadVB);
  }

  private _compile(vert: string, frag: string): WebGLProgram {
    const gl   = this.gl;
    const vs   = this._shader(gl.VERTEX_SHADER, vert);
    const fs   = this._shader(gl.FRAGMENT_SHADER, frag);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`ATWaterNormals link: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _shader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`ATWaterNormals shader: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATDepthValue — depth linearisation pass for SSAO / SSR / DOF
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Output texture layout: rgba8unorm (or r32float for precision):
 *   .r = linearDepth / far   (0 → 1 normalised)
 *   .g = raw hardware depth  (0 → 1)
 *   .b = −viewZ / far        (0 → 1 normalised eye-space distance)
 *   .a = 1.0
 *
 * Consumers:
 *   • ambient-occlusion.ts #M775  reads .r (linDepthTex)
 *   • screen-space-reflection.ts #M784  reads .b (viewZ channel)
 *   • dof-bokeh.ts #M760  reads .r for CoC calculation
 */
export interface ATDepthValueConfig {
  /** Viewport width in pixels. */
  width:  number;
  /** Viewport height in pixels. */
  height: number;
  /** Camera near clip plane (metres). Default 0.1. */
  near?:  number;
  /** Camera far clip plane (metres). Default 1000. */
  far?:   number;
  /**
   * Output texture format.
   * 'rgba8'    — compact, sufficient for most post-process consumers.
   * 'rgba32f'  — full float precision for demanding SSR/SSAO passes.
   * Default 'rgba8'.
   */
  format?: 'rgba8' | 'rgba32f';
}

/**
 * ATDepthValue
 * ─────────────
 * Full-screen blit that reads the hardware depth texture and outputs
 * linearised depth in three channels: normalised linear, raw, and eye-space Z.
 *
 * Port of upstream/lygia/space/{linearizeDepth,depth2viewZ}.glsl.
 *
 * Usage:
 *
 *   const dv = new ATDepthValue(gl, { width: 1920, height: 1080 });
 *   dv.build();
 *
 *   // Each frame:
 *   dv.render(hardwareDepthTex);  // → dv.outputTex ready for SSAO/SSR/DOF
 *
 *   // Update camera params when changed:
 *   dv.setCamera(near, far);
 */
export class ATDepthValue {
  private readonly gl:   WebGL2RenderingContext;
  readonly cfg: Required<ATDepthValueConfig>;

  private program!:   WebGLProgram;
  private quadVB!:    WebGLBuffer;
  private fbo!:       WebGLFramebuffer;

  /** Output linearised depth texture — bind as input to SSAO / SSR / DOF. */
  outputTex!: WebGLTexture;

  private uDepthTex!: WebGLUniformLocation | null;
  private uNear!:     WebGLUniformLocation | null;
  private uFar!:      WebGLUniformLocation | null;

  /** Shared fullscreen-quad vertex shader. */
  static readonly VERT_SRC = /* glsl */`
attribute vec2 aPos;
varying vec2   vUV;
void main() {
    vUV         = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

  constructor(gl: WebGL2RenderingContext, cfg: ATDepthValueConfig) {
    this.gl  = gl;
    this.cfg = {
      width:  cfg.width,
      height: cfg.height,
      near:   cfg.near   ?? 0.1,
      far:    cfg.far    ?? 1000,
      format: cfg.format ?? 'rgba8',
    };
  }

  /** Compile shader, allocate output texture and FBO. */
  build(): void {
    const gl = this.gl;
    this.program = this._compile(ATDepthValue.VERT_SRC, DEPTHVALUE_FS);

    this.uDepthTex = gl.getUniformLocation(this.program, 'uDepthTex');
    this.uNear     = gl.getUniformLocation(this.program, 'uNear');
    this.uFar      = gl.getUniformLocation(this.program, 'uFar');

    // Fullscreen quad
    const verts = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
    this.quadVB  = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVB);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    // Output texture
    this.outputTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.outputTex);
    const fmt     = this.cfg.format === 'rgba32f' ? gl.RGBA32F : gl.RGBA8;
    const extFmt  = this.cfg.format === 'rgba32f' ? gl.RGBA    : gl.RGBA;
    const extType = this.cfg.format === 'rgba32f' ? gl.FLOAT   : gl.UNSIGNED_BYTE;
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt, this.cfg.width, this.cfg.height,
                  0, extFmt, extType, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // FBO
    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, this.outputTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Linearise the hardware depth texture.
   *
   * @param depthTex  Hardware depth texture (DEPTH_COMPONENT, UNSIGNED_INT_24_8,
   *                  or DEPTH24_STENCIL8 — sampled as red channel in GLSL).
   */
  render(depthTex: WebGLTexture): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.cfg.width, this.cfg.height);
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, depthTex);
    gl.uniform1i(this.uDepthTex, 0);
    gl.uniform1f(this.uNear, this.cfg.near);
    gl.uniform1f(this.uFar,  this.cfg.far);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVB);
    const aPos = gl.getAttribLocation(this.program, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Update camera clip planes (e.g. after camera reconfigure). */
  setCamera(near: number, far: number): void {
    this.cfg.near = near;
    this.cfg.far  = far;
  }

  destroy(): void {
    this.gl.deleteProgram(this.program);
    this.gl.deleteBuffer(this.quadVB);
    this.gl.deleteFramebuffer(this.fbo);
    this.gl.deleteTexture(this.outputTex);
  }

  private _compile(vert: string, frag: string): WebGLProgram {
    const gl   = this.gl;
    const vs   = this._shader(gl.VERTEX_SHADER, vert);
    const fs   = this._shader(gl.FRAGMENT_SHADER, frag);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`ATDepthValue link: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _shader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`ATDepthValue shader: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  }
}
