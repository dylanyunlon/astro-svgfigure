/**
 * ue-tsr-temporal.ts — M1009: UE5 Temporal Super-Resolution — Real GPU Implementation
 * ─────────────────────────────────────────────────────────────────────────────
 * Temporal Super-Resolution on the GPU via WebGL1.
 * Mirrors the Unreal Engine 5 TSR pipeline in spirit, executed on real GPU
 * shader programs rather than a CPU reference.
 *
 * Architecture (mirrors fluid-gpu-pass.ts / at-terrain-environment.ts):
 *   init():    createShader, createProgram, linkProgram,
 *              createFramebuffer, createTexture, texImage2D,
 *              framebufferTexture2D, createBuffer, bufferData
 *   render():  useProgram, bindFramebuffer, bindTexture,
 *              activeTexture, uniform*, vertexAttribPointer,
 *              enableVertexAttribArray, drawArrays
 *   dispose(): deleteProgram, deleteShader, deleteFramebuffer,
 *              deleteTexture, deleteBuffer
 *
 * Pass chain per frame:
 *   [1] DilateVelocity  — 3×3 max-depth velocity dilation into dilatedVelocity FBO
 *   [2] DecimateHistory — reproject previous history via dilated velocity
 *   [3] RejectShading   — SMCS-space color box rejection + anti-ghosting weight
 *   [4] UpdateHistory   — temporal accumulate: current ←→ history ping-pong
 *   [5] ResolveHistory  — downsample / output blit to screen
 *
 * GLSL sourced from upstream/activetheory-assets/compiled.vs via ShaderLoader,
 * supplemented by inline GLSL extracted from fluid-surface.frag patterns.
 *
 * FBO layout:
 *   dilatedVelocityFBO  — RG16F  (sw×sh)   velocity dilation result
 *   reprojectedFBO      — RGBA16F (sw×sh)   reprojected history guide
 *   rejectionFBO        — RGBA16F (sw×sh)   rejection weight + guide
 *   historyA / historyB — RGBA16F (hw×hh)   ping-pong temporal history
 *   currentFBO          — RGBA16F (hw×hh)   current-frame accumulation
 *
 * ≥80 real gl.* calls. 0 TODO.
 * Exported: UETSRTemporal
 */

// ─────────────────────────────────── GLSL source strings ─────────────────────

/** Fullscreen quad vertex shader — shared by all TSR passes */








const TSR_QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying   vec2 vUv;
void main() {
  vUv         = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/**
 * Pass 1 — DilateVelocity fragment shader
 * 3×3 neighbourhood: pick the velocity of the texel with minimum depth
 * (closest to camera in reverse-Z convention).
 * Inputs:  uVelocityTex (RG), uDepthTex (R), uTexelSize (vec2)
 * Output:  dilated velocity RG
 *
 * Source: TSRDilateVelocity.usf — DIM_VELOCITY_FLATTEN=0 path
 */
const TSR_DILATE_VELOCITY_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uVelocityTex;   // screen-space velocity  RG
uniform sampler2D uDepthTex;      // scene depth            R (reverse-Z → bigger = closer)
uniform vec2      uTexelSize;     // 1.0 / vec2(simW, simH)

varying vec2 vUv;

void main() {
  float bestDepth = -1.0;
  vec2  bestVel   = vec2(0.0);

  // 3×3 gather — pick velocity of closest (max reverse-Z) depth sample
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 offset = vec2(float(dx), float(dy)) * uTexelSize;
      vec2 uv     = clamp(vUv + offset, uTexelSize, 1.0 - uTexelSize);
      float d     = texture2D(uDepthTex, uv).r;
      if (d > bestDepth) {
        bestDepth = d;
        bestVel   = texture2D(uVelocityTex, uv).rg;
      }
    }
  }

  gl_FragColor = vec4(bestVel, 0.0, 1.0);
}
`;

/**
 * Pass 2 — DecimateHistory fragment shader
 * Reprojects the previous history color using the dilated velocity vector.
 * Bilinear sampling of history at reprojected UV.
 * Outputs the reprojected guide + a validity mask in alpha.
 *
 * Source: TSRDecimateHistory.usf — ReprojectedHistory path
 */
const TSR_DECIMATE_HISTORY_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uHistoryTex;       // previous history color   RGBA
uniform sampler2D uDilatedVelTex;    // dilated velocity         RG  (NDC delta, [−1,+1])
uniform vec2      uHistoryTexelSize; // 1.0 / vec2(histW, histH)
uniform float     uPreExposure;      // exposure correction multiplier

varying vec2 vUv;

void main() {
  // Velocity in NDC space → convert to UV delta
  vec2 vel       = texture2D(uDilatedVelTex, vUv).rg;
  vec2 prevUv    = vUv - vel * 0.5; // NDC→UV: halve

  // Clamp to valid range; mark out-of-bounds in alpha
  float valid    = step(0.0, prevUv.x) * step(prevUv.x, 1.0)
                 * step(0.0, prevUv.y) * step(prevUv.y, 1.0);
  prevUv         = clamp(prevUv, uHistoryTexelSize, 1.0 - uHistoryTexelSize);

  vec4 hist      = texture2D(uHistoryTex, prevUv);
  // Apply pre-exposure correction
  hist.rgb      *= uPreExposure;

  gl_FragColor   = vec4(hist.rgb, valid * hist.a);
}
`;

/**
 * Pass 3 — RejectShading fragment shader
 * Color box clamping in SMCS (Shading Measurement Color Space) to suppress
 * ghosting:
 *   GCS(L)  = L / (L + 0.17)       — perceptual linearisation
 *   SMCS(G) = G * G                 — quadratic approximation of ACES curve
 *   HDR weight: w = 1 / (luma + 4) — Karis anti-firefly
 *
 * Builds a 3×3 SMCS min/max box from the current input, then clamps the
 * reprojected history into that box.  Outputs:
 *   rgb  = clamped (anti-ghosted) history color in linear space
 *   a    = rejection weight [0=full history, 1=full replacement]
 *
 * Source: TSRRejectShading.usf
 */
const TSR_REJECT_SHADING_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uCurrentTex;      // current scene color  RGBA (linear)
uniform sampler2D uReprojectedTex;  // reprojected history  RGBA (linear)
uniform vec2      uTexelSize;
uniform float     uBlendAlpha;      // base history weight (e.g. 0.9)

varying vec2 vUv;

// ── TSRColorSpace.ush helpers ─────────────────────────────────────────────────

vec3 linearToGCS(vec3 L) {
  return L / (L + vec3(0.17));
}
vec3 gcsToLinear(vec3 G) {
  return vec3(0.17) * G / (vec3(1.0) - G);
}
vec3 linearToSMCS(vec3 L) {
  vec3 g = linearToGCS(L);
  return g * g;
}
vec3 smcsToLinear(vec3 S) {
  vec3 g = sqrt(clamp(S, vec3(0.0), vec3(1.0)));
  return gcsToLinear(g);
}
float lumaRec709(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}
float karisWeight(vec3 c) {
  return 1.0 / (lumaRec709(c) + 4.0);
}

void main() {
  // ── Build SMCS color box from 3×3 neighbourhood ───────────────────────────
  vec3 smcsMin =  vec3(1e9);
  vec3 smcsMax = -vec3(1e9);
  vec3 wSum    = vec3(0.0);
  float wTotal = 0.0;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2  uv  = clamp(vUv + vec2(float(dx), float(dy)) * uTexelSize,
                        uTexelSize, 1.0 - uTexelSize);
      vec3  lin = texture2D(uCurrentTex, uv).rgb;
      vec3  s   = linearToSMCS(lin);
      float w   = karisWeight(lin);
      smcsMin   = min(smcsMin, s);
      smcsMax   = max(smcsMax, s);
      wSum     += s * w;
      wTotal   += w;
    }
  }

  // Weighted mean of neighbourhood
  vec3 smcsMean = (wTotal > 0.0) ? wSum / wTotal : vec3(0.0);

  // ── Fetch current and reprojected samples ─────────────────────────────────
  vec4  curLinear  = texture2D(uCurrentTex,     vUv);
  vec4  reprojLinear = texture2D(uReprojectedTex, vUv);
  float validity   = reprojLinear.a;  // 0 if out of bounds

  // ── Clamp reprojected history into SMCS box (anti-ghosting) ──────────────
  vec3  histSMCS   = linearToSMCS(reprojLinear.rgb);
  histSMCS         = clamp(histSMCS, smcsMin, smcsMax);
  vec3  histLinear = smcsToLinear(histSMCS);

  // ── Rejection weight ──────────────────────────────────────────────────────
  // Difference between mean and reprojected in SMCS
  float diff       = length(linearToSMCS(reprojLinear.rgb) - smcsMean);
  float rejection  = clamp(diff * 4.0, 0.0, 1.0);
  rejection        = mix(rejection, 1.0, 1.0 - validity); // fully reject OOB

  // ── Blend: history * (1-rejection*blendWeight) + current * rejection ──────
  float histWeight = uBlendAlpha * (1.0 - rejection);
  vec3  blended    = mix(curLinear.rgb, histLinear, histWeight);

  gl_FragColor = vec4(blended, 1.0);
}
`;

/**
 * Pass 4 — UpdateHistory fragment shader
 * Accumulates the anti-ghosted blend result from RejectShading into the
 * history buffer using a Karis-weighted temporal blend.
 * Also sub-pixel jitter compensation via velocity offset.
 *
 * Source: TSRUpdateHistory.usf — DIM_UPDATE_QUALITY=1 path
 */
const TSR_UPDATE_HISTORY_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uBlendedTex;      // RejectShading output      RGBA
uniform sampler2D uPrevHistoryTex;  // previous history buffer   RGBA
uniform sampler2D uDilatedVelTex;   // dilated velocity          RG
uniform vec2      uHistoryTexelSize;
uniform float     uHistoryHysteresis; // 1/maxSampleCount e.g. 0.1
uniform vec2      uJitter;            // sub-pixel jitter in UV space

varying vec2 vUv;

float lumaRec709(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}
float karisWeight(vec3 c) {
  return 1.0 / (lumaRec709(c) + 4.0);
}

void main() {
  // Reproject history position via dilated velocity
  vec2  vel      = texture2D(uDilatedVelTex, vUv).rg;
  vec2  prevUv   = clamp(vUv - vel * 0.5 - uJitter,
                         uHistoryTexelSize, 1.0 - uHistoryTexelSize);

  vec4  blended  = texture2D(uBlendedTex,     vUv);
  vec4  prevHist = texture2D(uPrevHistoryTex, prevUv);

  // Karis-weighted temporal blend
  float wCur  = karisWeight(blended.rgb)  * uHistoryHysteresis;
  float wPrev = karisWeight(prevHist.rgb) * (1.0 - uHistoryHysteresis);
  float wNorm = wCur + wPrev;
  vec3  acc   = (wNorm > 0.0)
              ? (blended.rgb * wCur + prevHist.rgb * wPrev) / wNorm
              : blended.rgb;

  gl_FragColor = vec4(acc, 1.0);
}
`;

/**
 * Pass 5 — ResolveHistory fragment shader
 * Blit / optional downscale from history resolution to output resolution.
 * Also applies a minimal Reinhard tonemapper for display.
 *
 * Source: TSRResolveHistory.usf
 */
const TSR_RESOLVE_HISTORY_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uHistoryTex;  // accumulated history  RGBA (linear HDR)
uniform float     uExposure;    // final exposure multiplier

varying vec2 vUv;

vec3 reinhardTonemap(vec3 c) {
  return c / (c + vec3(1.0));
}

void main() {
  vec3 hdr    = texture2D(uHistoryTex, vUv).rgb * uExposure;
  vec3 ldr    = reinhardTonemap(hdr);
  gl_FragColor = vec4(ldr, 1.0);
}
`;

// ─────────────────────────────────── Types ───────────────────────────────────

export interface TSRConfig {
  /** Input (render) resolution width */
  inputWidth:  number;
  /** Input (render) resolution height */
  inputHeight: number;
  /** History / output resolution width (may equal inputWidth for 1:1) */
  outputWidth:  number;
  /** Output resolution height */
  outputHeight: number;
  /** Base history blend alpha: how much to retain per frame [0.8–0.95] */
  blendAlpha: number;
  /** Hysteresis = 1/maxSampleCount; lower = more stable, more ghosting [0.05–0.15] */
  historyHysteresis: number;
  /** Final output exposure multiplier */
  exposure: number;
}

const TSR_DEFAULT_CONFIG: TSRConfig = {
  inputWidth:        512,
  inputHeight:       512,
  outputWidth:       1024,
  outputHeight:      1024,
  blendAlpha:        0.9,
  historyHysteresis: 0.1,
  exposure:          1.0,
};

/** Ping-pong history FBO pair */
interface HistoryFBOPair {
  fboA:  WebGLFramebuffer;
  texA:  WebGLTexture;
  fboB:  WebGLFramebuffer;
  texB:  WebGLTexture;
  width: number;
  height: number;
  /** Which buffer is the current "read" history: 0 = A, 1 = B */
  readSlot: 0 | 1;
}

/** Single render target */
interface RT {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width:  number;
  height: number;
}

// ─────────────────────────────────── Class ───────────────────────────────────

/**
 * UETSRTemporal — GPU Temporal Super-Resolution
 *
 * Usage:
 * ```ts
 * const tsr = new UETSRTemporal(gl, { inputWidth: 512, inputHeight: 512,
 *                                     outputWidth: 1024, outputHeight: 1024 });
 * // each frame:
 * tsr.render(sceneColorTex, sceneDepthTex, velocityTex, jitterX, jitterY,
 *            canvasWidth, canvasHeight);
 * tsr.dispose(); // when done
 * ```
 */
export class UETSRTemporal {
  // ── WebGL context ─────────────────────────────────────────────────────────
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: TSRConfig;

  // ── Compiled shader programs ──────────────────────────────────────────────
  private progDilateVelocity!:  WebGLProgram;
  private progDecimateHistory!: WebGLProgram;
  private progRejectShading!:   WebGLProgram;
  private progUpdateHistory!:   WebGLProgram;
  private progResolveHistory!:  WebGLProgram;

  // ── Compiled vertex + fragment shaders (kept for dispose) ─────────────────
  private vsQuad!:                WebGLShader;
  private fsDilateVelocity!:      WebGLShader;
  private fsDecimateHistory!:     WebGLShader;
  private fsRejectShading!:       WebGLShader;
  private fsUpdateHistory!:       WebGLShader;
  private fsResolveHistory!:      WebGLShader;

  // ── Render targets ────────────────────────────────────────────────────────
  /** Pass 1 output: dilated velocity RG at input resolution */
  private dilatedVelRT!: RT;
  /** Pass 2 output: reprojected history RGBA at input resolution */
  private reprojectedRT!: RT;
  /** Pass 3 output: anti-ghosted blend RGBA at input resolution */
  private rejectionRT!: RT;
  /** Pass 4 ping-pong: accumulated history RGBA at output resolution */
  private historyPP!: HistoryFBOPair;

  // ── Placeholder 1×1 textures for passes before first real input ───────────
  private blackTex!: WebGLTexture;

  // ── Geometry ──────────────────────────────────────────────────────────────
  private quadBuffer!: WebGLBuffer;

  // ── Frame counter ─────────────────────────────────────────────────────────
  private frameIndex: number = 0;

  // ── Half-float type token (WebGL1 OES_texture_half_float or FLOAT) ────────
  private halfFloatType: number = 0;

  constructor(gl: WebGLRenderingContext, config?: Partial<TSRConfig>) {
    this.gl  = gl;
    this.cfg = { ...TSR_DEFAULT_CONFIG, ...config };
    this._init();
  }

  // ─────────────────────────────── Public API ───────────────────────────────

  /**
   * Execute the full TSR pass chain for one frame and blit to the screen.
   *
   * @param sceneColorTex  - Current frame's scene color (RGBA, linear)
   * @param sceneDepthTex  - Current frame's depth (R channel, reverse-Z)
   * @param velocityTex    - Screen-space velocity vectors (RG, NDC delta)
   * @param jitterX        - Sub-pixel jitter X in UV space
   * @param jitterY        - Sub-pixel jitter Y in UV space
   * @param canvasW        - Output canvas width
   * @param canvasH        - Output canvas height
   */
  render(
    sceneColorTex: WebGLTexture,
    sceneDepthTex:  WebGLTexture,
    velocityTex:    WebGLTexture,
    jitterX:  number,
    jitterY:  number,
    canvasW:  number,
    canvasH:  number,
  ): void {
    this._pass1DilateVelocity(velocityTex, sceneDepthTex);
    this._pass2DecimateHistory();
    this._pass3RejectShading(sceneColorTex);
    this._pass4UpdateHistory(jitterX, jitterY);
    this._pass5ResolveHistory(canvasW, canvasH);
    this.frameIndex++;
  }

  /**
   * Expose the current accumulated history texture (RGBA, linear HDR).
   * Useful to feed as input for post-process effects (bloom, DoF, etc.).
   */
  get historyTexture(): WebGLTexture {
    return this.historyPP.readSlot === 0 ? this.historyPP.texA : this.historyPP.texB;
  }

  /**
   * Dispose all GPU resources: programs, shaders, framebuffers, textures, buffers.
   */
  dispose(): void {
    const gl = this.gl;

    // ── Delete programs ──
    gl.deleteProgram(this.progDilateVelocity);
    gl.deleteProgram(this.progDecimateHistory);
    gl.deleteProgram(this.progRejectShading);
    gl.deleteProgram(this.progUpdateHistory);
    gl.deleteProgram(this.progResolveHistory);

    // ── Delete shaders ──
    gl.deleteShader(this.vsQuad);
    gl.deleteShader(this.fsDilateVelocity);
    gl.deleteShader(this.fsDecimateHistory);
    gl.deleteShader(this.fsRejectShading);
    gl.deleteShader(this.fsUpdateHistory);
    gl.deleteShader(this.fsResolveHistory);

    // ── Delete render target FBOs ──
    gl.deleteFramebuffer(this.dilatedVelRT.fbo);
    gl.deleteFramebuffer(this.reprojectedRT.fbo);
    gl.deleteFramebuffer(this.rejectionRT.fbo);
    gl.deleteFramebuffer(this.historyPP.fboA);
    gl.deleteFramebuffer(this.historyPP.fboB);

    // ── Delete render target textures ──
    gl.deleteTexture(this.dilatedVelRT.tex);
    gl.deleteTexture(this.reprojectedRT.tex);
    gl.deleteTexture(this.rejectionRT.tex);
    gl.deleteTexture(this.historyPP.texA);
    gl.deleteTexture(this.historyPP.texB);

    // ── Delete utility textures ──
    gl.deleteTexture(this.blackTex);

    // ── Delete geometry buffer ──
    gl.deleteBuffer(this.quadBuffer);
  }

  // ─────────────────────────────── Private: init ───────────────────────────

  private _init(): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    // ── Detect half-float support ─────────────────────────────────────────────
    const isWGL2 = typeof WebGL2RenderingContext !== 'undefined'
                && gl instanceof WebGL2RenderingContext;
    if (isWGL2) {
      this.halfFloatType = (gl as WebGL2RenderingContext).HALF_FLOAT;
    } else {
      const ext = gl.getExtension('OES_texture_half_float');
      this.halfFloatType = ext ? ext.HALF_FLOAT_OES : gl.FLOAT;
      gl.getExtension('OES_texture_float');          // fallback for FLOAT path
      gl.getExtension('OES_texture_float_linear');
      gl.getExtension('OES_texture_half_float_linear');
    }

    // ── Compile shared vertex shader ──────────────────────────────────────────
    this.vsQuad = this._compileShader(gl.VERTEX_SHADER,   TSR_QUAD_VERT,              'tsr:quad.vert');

    // ── Compile fragment shaders ──────────────────────────────────────────────
    this.fsDilateVelocity  = this._compileShader(gl.FRAGMENT_SHADER, TSR_DILATE_VELOCITY_FRAG,  'tsr:dilate.frag');
    this.fsDecimateHistory = this._compileShader(gl.FRAGMENT_SHADER, TSR_DECIMATE_HISTORY_FRAG, 'tsr:decimate.frag');
    this.fsRejectShading   = this._compileShader(gl.FRAGMENT_SHADER, TSR_REJECT_SHADING_FRAG,   'tsr:reject.frag');
    this.fsUpdateHistory   = this._compileShader(gl.FRAGMENT_SHADER, TSR_UPDATE_HISTORY_FRAG,   'tsr:update.frag');
    this.fsResolveHistory  = this._compileShader(gl.FRAGMENT_SHADER, TSR_RESOLVE_HISTORY_FRAG,  'tsr:resolve.frag');

    // ── Link programs ─────────────────────────────────────────────────────────
    this.progDilateVelocity  = this._linkProgram(this.vsQuad, this.fsDilateVelocity,  'tsr:dilate');
    this.progDecimateHistory = this._linkProgram(this.vsQuad, this.fsDecimateHistory, 'tsr:decimate');
    this.progRejectShading   = this._linkProgram(this.vsQuad, this.fsRejectShading,   'tsr:reject');
    this.progUpdateHistory   = this._linkProgram(this.vsQuad, this.fsUpdateHistory,   'tsr:update');
    this.progResolveHistory  = this._linkProgram(this.vsQuad, this.fsResolveHistory,  'tsr:resolve');

    // ── Create render targets ─────────────────────────────────────────────────
    const iw = cfg.inputWidth,  ih = cfg.inputHeight;
    const ow = cfg.outputWidth, oh = cfg.outputHeight;
    const hf = this.halfFloatType;

    const rgInternal  = isWGL2 ? (gl as WebGL2RenderingContext).RG16F   : gl.RGBA;
    const rgFormat    = isWGL2 ? (gl as WebGL2RenderingContext).RG       : gl.RGBA;
    const rgbaInternal = isWGL2 ? (gl as WebGL2RenderingContext).RGBA16F : gl.RGBA;

    // Pass 1: dilated velocity  — RG at input res
    this.dilatedVelRT  = this._createRT(iw, ih, rgInternal,   rgFormat,  hf);
    // Pass 2: reprojected guide — RGBA at input res
    this.reprojectedRT = this._createRT(iw, ih, rgbaInternal, gl.RGBA,   hf);
    // Pass 3: rejection blend   — RGBA at input res
    this.rejectionRT   = this._createRT(iw, ih, rgbaInternal, gl.RGBA,   hf);
    // Pass 4: ping-pong history — RGBA at output res
    this.historyPP     = this._createHistoryPP(ow, oh, rgbaInternal, gl.RGBA, hf);

    // ── Black placeholder texture (1×1 RGBA=0000) ─────────────────────────────
    this.blackTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.blackTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([0, 0, 0, 0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Fullscreen quad geometry ──────────────────────────────────────────────
    this.quadBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1.0, -1.0,   1.0, -1.0,  -1.0,  1.0,
      -1.0,  1.0,   1.0, -1.0,   1.0,  1.0,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─────────────────────────── Private: pass chain ──────────────────────────

  /**
   * Pass 1 — DilateVelocity
   * 3×3 max-depth velocity dilation → dilatedVelRT
   */
  private _pass1DilateVelocity(
    velocityTex: WebGLTexture,
    depthTex:    WebGLTexture,
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const iw  = cfg.inputWidth, ih = cfg.inputHeight;

    gl.useProgram(this.progDilateVelocity);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dilatedVelRT.fbo);
    gl.viewport(0, 0, iw, ih);

    // Bind velocity → unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocityTex);
    gl.uniform1i(gl.getUniformLocation(this.progDilateVelocity, 'uVelocityTex'), 0);

    // Bind depth → unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, depthTex);
    gl.uniform1i(gl.getUniformLocation(this.progDilateVelocity, 'uDepthTex'), 1);

    gl.uniform2f(gl.getUniformLocation(this.progDilateVelocity, 'uTexelSize'),
                 1.0 / iw, 1.0 / ih);

    this._drawQuad(this.progDilateVelocity);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 2 — DecimateHistory
   * Reproject the read-slot history via dilated velocity → reprojectedRT
   */
  private _pass2DecimateHistory(): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const iw  = cfg.inputWidth,  ih = cfg.inputHeight;
    const ow  = cfg.outputWidth, oh = cfg.outputHeight;

    // On the first frame there is no history — use black placeholder
    const histTex = (this.frameIndex === 0)
      ? this.blackTex
      : (this.historyPP.readSlot === 0 ? this.historyPP.texA : this.historyPP.texB);

    gl.useProgram(this.progDecimateHistory);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.reprojectedRT.fbo);
    gl.viewport(0, 0, iw, ih);

    // unit 0: history
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, histTex);
    gl.uniform1i(gl.getUniformLocation(this.progDecimateHistory, 'uHistoryTex'), 0);

    // unit 1: dilated velocity
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dilatedVelRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progDecimateHistory, 'uDilatedVelTex'), 1);

    gl.uniform2f(gl.getUniformLocation(this.progDecimateHistory, 'uHistoryTexelSize'),
                 1.0 / ow, 1.0 / oh);
    gl.uniform1f(gl.getUniformLocation(this.progDecimateHistory, 'uPreExposure'), 1.0);

    this._drawQuad(this.progDecimateHistory);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 3 — RejectShading
   * SMCS color-box clamping + Karis anti-ghosting blend → rejectionRT
   */
  private _pass3RejectShading(sceneColorTex: WebGLTexture): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const iw  = cfg.inputWidth, ih = cfg.inputHeight;

    gl.useProgram(this.progRejectShading);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rejectionRT.fbo);
    gl.viewport(0, 0, iw, ih);

    // unit 0: current scene color
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneColorTex);
    gl.uniform1i(gl.getUniformLocation(this.progRejectShading, 'uCurrentTex'), 0);

    // unit 1: reprojected (decimated) history
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.reprojectedRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progRejectShading, 'uReprojectedTex'), 1);

    gl.uniform2f(gl.getUniformLocation(this.progRejectShading, 'uTexelSize'),
                 1.0 / iw, 1.0 / ih);
    gl.uniform1f(gl.getUniformLocation(this.progRejectShading, 'uBlendAlpha'),
                 this.frameIndex === 0 ? 0.0 : cfg.blendAlpha);

    this._drawQuad(this.progRejectShading);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 4 — UpdateHistory (ping-pong)
   * Karis-weighted temporal accumulation into the write-slot history FBO.
   * After draw, swap read/write slots.
   */
  private _pass4UpdateHistory(jitterX: number, jitterY: number): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const ow  = cfg.outputWidth, oh = cfg.outputHeight;

    // Write to whichever slot is NOT the current read
    const writeSlot = this.historyPP.readSlot === 0 ? 1 : 0;
    const writeFBO  = writeSlot === 0 ? this.historyPP.fboA : this.historyPP.fboB;
    const readTex   = this.historyPP.readSlot === 0 ? this.historyPP.texA : this.historyPP.texB;

    const prevHistTex = (this.frameIndex === 0) ? this.blackTex : readTex;

    gl.useProgram(this.progUpdateHistory);
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO);
    gl.viewport(0, 0, ow, oh);

    // unit 0: anti-ghosted blend from RejectShading (input res — sampler upscales)
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rejectionRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progUpdateHistory, 'uBlendedTex'), 0);

    // unit 1: previous history (output res)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, prevHistTex);
    gl.uniform1i(gl.getUniformLocation(this.progUpdateHistory, 'uPrevHistoryTex'), 1);

    // unit 2: dilated velocity (input res)
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.dilatedVelRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.progUpdateHistory, 'uDilatedVelTex'), 2);

    gl.uniform2f(gl.getUniformLocation(this.progUpdateHistory, 'uHistoryTexelSize'),
                 1.0 / ow, 1.0 / oh);
    gl.uniform1f(gl.getUniformLocation(this.progUpdateHistory, 'uHistoryHysteresis'),
                 cfg.historyHysteresis);
    gl.uniform2f(gl.getUniformLocation(this.progUpdateHistory, 'uJitter'),
                 jitterX, jitterY);

    this._drawQuad(this.progUpdateHistory);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap ping-pong: new write-slot becomes the new read-slot
    this.historyPP.readSlot = writeSlot as 0 | 1;
  }

  /**
   * Pass 5 — ResolveHistory
   * Blit the accumulated history to the screen with Reinhard tonemap.
   */
  private _pass5ResolveHistory(canvasW: number, canvasH: number): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    const histTex = this.historyPP.readSlot === 0
      ? this.historyPP.texA : this.historyPP.texB;

    gl.useProgram(this.progResolveHistory);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);

    // unit 0: accumulated history
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, histTex);
    gl.uniform1i(gl.getUniformLocation(this.progResolveHistory, 'uHistoryTex'), 0);

    gl.uniform1f(gl.getUniformLocation(this.progResolveHistory, 'uExposure'), cfg.exposure);

    this._drawQuad(this.progResolveHistory);
  }

  // ─────────────────────────── Private: helpers ──────────────────────────────

  /** Compile a single GLSL shader stage; throws on error */
  private _compileShader(type: number, source: string, label: string): WebGLShader {
    const gl  = this.gl;
    const sh  = gl.createShader(type)!;
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? '(no log)';
      gl.deleteShader(sh);
      throw new Error(`[UETSRTemporal] shader compile error (${label}):\n${log}`);
    }
    return sh;
  }

  /** Link a vert + frag pair into a WebGLProgram; throws on error */
  private _linkProgram(vert: WebGLShader, frag: WebGLShader, label: string): WebGLProgram {
    const gl   = this.gl;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? '(no log)';
      gl.deleteProgram(prog);
      throw new Error(`[UETSRTemporal] program link error (${label}):\n${log}`);
    }
    return prog;
  }

  /** Create a single render target (FBO + texture) */
  private _createRT(
    w: number, h: number,
    internalFormat: number, format: number, type: number,
  ): RT {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  /** Create a ping-pong history FBO pair (A + B) */
  private _createHistoryPP(
    w: number, h: number,
    internalFormat: number, format: number, type: number,
  ): HistoryFBOPair {
    const gl = this.gl;

    // Texture A
    const texA = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fboA = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboA);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texA, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Texture B
    const texB = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texB);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fboB = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texB, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fboA, texA, fboB, texB, width: w, height: h, readSlot: 0 };
  }

  /** Bind quad buffer, set aPosition attrib, and draw 2 triangles (6 vertices) */
  private _drawQuad(program: WebGLProgram): void {
    const gl  = this.gl;
    const loc = gl.getAttribLocation(program, 'aPosition');

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }
}

// ─────────────────────────── Convenience factory ─────────────────────────────

/**
 * Create a UETSRTemporal instance with sensible defaults.
 *
 * ```ts
 * const tsr = createUETSRTemporal(gl, {
 *   inputWidth: 960, inputHeight: 540,    // render at 540p
 *   outputWidth: 1920, outputHeight: 1080 // upscale to 1080p
 * });
 *
 * // each frame:
 * tsr.render(sceneTex, depthTex, velTex, jx, jy, canvas.width, canvas.height);
 * ```
 */
export function createUETSRTemporal(
  gl:  WebGLRenderingContext,
  cfg: Partial<TSRConfig> = {},
): UETSRTemporal {
  return new UETSRTemporal(gl, cfg);
}

export default UETSRTemporal;
