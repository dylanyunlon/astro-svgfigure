/**
 * at-mousefluid-import.ts — M920: AT Mousefluid — real WebGL GPU mouse fluid
 * ─────────────────────────────────────────────────────────────────────────────
 * 鼠标是笔，流体是墨。Real GPU Navier-Stokes with mousemove splat injection.
 *
 * Architecture (matches fluid-gpu-pass.ts + at-antimatter-particles.ts pattern):
 *   init()    → createProgram × N, compileShader, linkProgram,
 *               createFramebuffer × M, createTexture × M, createBuffer, bufferData
 *   render()  → useProgram, bindFramebuffer, bindTexture, uniform*, bindBuffer,
 *               drawArrays  (per-pass)
 *   dispose() → deleteProgram × N, deleteFramebuffer × M, deleteTexture × M, deleteBuffer
 *
 * NS pass chain per frame:
 *   mousemove splat → curl → vorticity → divergence → pressure clear →
 *   pressure Jacobi × N → gradient subtract → advect velocity → advect dye
 *
 * GLSL shaders are inline strings extracted from:
 *   upstream/activetheory-assets/compiled.vs
 *   splatShader.fs / curlShader.fs / vorticityShader.fs / divergenceShader.fs
 *   pressureShader.fs / gradientSubtractShader.fs / clearShader.fs / advectionShader.fs
 *   advectionManualFilteringShader.fs
 *
 * Research: M920 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: shared fluid vertex shader (AT fluidBase.vs — compiled.vs:6664)
// Outputs neighbour UVs (vL/vR/vT/vB) for all NS passes.
// ─────────────────────────────────────────────────────────────────────────────









const FLUID_VERT = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// Simple fullscreen quad vertex — no neighbour UVs needed.
const SIMPLE_VERT = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT splatShader.fs (compiled.vs:6720)
// AT-specific: point-to-line-segment distance + cubicOut + screen blend.
// prevPoint → point line segment for continuous high-speed trails.
// ─────────────────────────────────────────────────────────────────────────────

const SPLAT_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec3 bgColor;
uniform vec2 point;
uniform vec2 prevPoint;
uniform float radius;
uniform float canRender;
uniform float uAdd;

float blendScreen(float base, float blend) {
    return 1.0 - ((1.0 - base) * (1.0 - blend));
}
vec3 blendScreen3(vec3 base, vec3 blend) {
    return vec3(
        blendScreen(base.r, blend.r),
        blendScreen(base.g, blend.g),
        blendScreen(base.b, blend.b)
    );
}
// AT l() — point-to-line-segment distance with aspect correction
float lineDist(vec2 uv, vec2 p1, vec2 p2) {
    vec2 pa = uv - p1;
    vec2 ba = p2 - p1;
    pa.x *= aspectRatio;
    ba.x *= aspectRatio;
    float h = clamp(dot(pa, ba) / dot(ba, ba + 1e-10), 0.0, 1.0);
    return length(pa - ba * h);
}
// AT cubicOut easing
float cubicOut(float t) {
    float f = t - 1.0;
    return f * f * f + 1.0;
}
void main() {
    vec3 splat = (1.0 - cubicOut(clamp(
        lineDist(vUv, prevPoint, point) / radius, 0.0, 1.0
    ))) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    base *= canRender;
    vec3 outColor = mix(blendScreen3(base, splat), base + splat, uAdd);
    gl_FragColor = vec4(outColor, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT curlShader.fs (compiled.vs:6627)
// 2D vorticity ω = ∂vy/∂x − ∂vx/∂y via finite differences.
// ─────────────────────────────────────────────────────────────────────────────

const CURL_FRAG = /* glsl */ `
precision highp float;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main() {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT vorticityShader.fs (compiled.vs:6759)
// Vorticity confinement — apply anti-diffusion force from curl gradient.
// AT convention: force.y *= -1.0 (Y-flip).
// ─────────────────────────────────────────────────────────────────────────────

const VORTICITY_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main() {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 vel = texture2D(uVelocity, vUv).xy;
    gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT divergenceShader.fs (compiled.vs:6646)
// Divergence with boundary reflection (AT-specific: if vL.x < 0, L = -C.x).
// ─────────────────────────────────────────────────────────────────────────────

const DIVERGENCE_FRAG = /* glsl */ `
precision highp float;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main() {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
    if (vL.x < 0.0) { L = -C.x; }
    if (vR.x > 1.0) { R = -C.x; }
    if (vT.y > 1.0) { T = -C.y; }
    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT pressureShader.fs (compiled.vs:6698)
// Jacobi iteration: p = (L + R + B + T - divergence) * 0.25
// boundary() currently passthrough (AT-compatible).
// ─────────────────────────────────────────────────────────────────────────────

const PRESSURE_FRAG = /* glsl */ `
precision highp float;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
vec2 boundary(vec2 uv) {
    return uv;
}
void main() {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT gradientSubtractShader.fs (compiled.vs:6678)
// Projection step: vel -= (R - L, T - B) → divergence-free velocity.
// ─────────────────────────────────────────────────────────────────────────────

const GRADIENT_SUBTRACT_FRAG = /* glsl */ `
precision highp float;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
vec2 boundary(vec2 uv) {
    return uv;
}
void main() {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT clearShader.fs (compiled.vs:6619)
// Pressure dissipation: output = value * input.
// ─────────────────────────────────────────────────────────────────────────────

const CLEAR_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main() {
    gl_FragColor = value * texture2D(uTexture, vUv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT advectionShader.fs (compiled.vs:6600)
// Semi-Lagrangian advection with linear texture filter.
// ─────────────────────────────────────────────────────────────────────────────

const ADVECTION_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main() {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    gl_FragColor = dissipation * texture2D(uSource, coord);
    gl_FragColor.a = 1.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT advectionManualFilteringShader.fs (compiled.vs:6579)
// Manual bilinear advection — fallback when linear filtering is unavailable.
// ─────────────────────────────────────────────────────────────────────────────

const ADVECTION_MANUAL_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
vec4 bilerp(sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
void main() {
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
    gl_FragColor.a = 1.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL: AT mousefluid.fs consumer fragment (compiled.vs:7888)
// Downstream helper: getFluidVelocity() / getFluidVelocityMask().
// Exported for use by other shaders that consume this fluid system.
// ─────────────────────────────────────────────────────────────────────────────

export const AT_MOUSEFLUID_CONSUME_FS = /* glsl */ `
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

vec2 getFluidVelocity() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return texture2D(tFluid, vUv).xy * fluidMask;
}

vec3 getFluidVelocityMask() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return vec3(texture2D(tFluid, vUv).xy * fluidMask, fluidMask);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ATMouseFluidConfig {
  simWidth?:            number;   // sim grid width.  default 128
  simHeight?:           number;   // sim grid height. default 128
  dyeWidth?:            number;   // dye texture width.  default 512
  dyeHeight?:           number;   // dye texture height. default 512
  pressureIterations?:  number;   // Jacobi iters. default 25
  curl?:                number;   // vorticity strength. default 20
  splatRadius?:         number;   // splat radius (UV). default 0.25
  velocityDissipation?: number;   // per-frame velocity decay. default 0.98
  densityDissipation?:  number;   // per-frame dye decay. default 0.97
  pressureDissipation?: number;   // per-frame pressure clear. default 0.8
  dt?:                  number;   // fixed dt. default 1/60
  splatBlendMode?:      number;   // 0=screen, 1=additive. default 0
}

/** Ping-pong double FBO — texture → fbo write, swap each pass. */
interface DoubleFBO {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  width:    number;
  height:   number;
}

/** Single FBO for divergence / curl. */
interface SingleFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
}

// Golden-ratio hue sequence for multi-touch splats.
const GOLDEN_ANGLE = 0.381966011250105;

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  const sector = Math.floor(h * 6) % 6;
  if      (sector === 0) { r = c; g = x; b = 0; }
  else if (sector === 1) { r = x; g = c; b = 0; }
  else if (sector === 2) { r = 0; g = c; b = x; }
  else if (sector === 3) { r = 0; g = x; b = c; }
  else if (sector === 4) { r = x; g = 0; b = c; }
  else                   { r = c; g = 0; b = x; }
  return [r + m, g + m, b + m];
}

function nextSplatColor(idx: number): [number, number, number] {
  return hslToRgb((idx * GOLDEN_ANGLE) % 1.0, 0.9, 0.55);
}

// ─────────────────────────────────────────────────────────────────────────────
// ATMouseFluid — main class
// Pattern: init() / render()/tick() / dispose() matching fluid-gpu-pass.ts
// ─────────────────────────────────────────────────────────────────────────────

export class ATMouseFluid {
  private gl:     WebGLRenderingContext;
  private cfg:    Required<ATMouseFluidConfig>;
  private canvas: HTMLCanvasElement | null = null;

  // ── Programs (all compiled in init) ──
  private splatProg!:    WebGLProgram;
  private curlProg!:     WebGLProgram;
  private vortProg!:     WebGLProgram;
  private divProg!:      WebGLProgram;
  private preProg!:      WebGLProgram;
  private gradSubProg!:  WebGLProgram;
  private clearProg!:    WebGLProgram;
  private advectProg!:   WebGLProgram;
  private advectManProg!:WebGLProgram;

  // ── FBOs ──
  private velocity!:     DoubleFBO;
  private pressure!:     DoubleFBO;
  private dye!:          DoubleFBO;
  private divergence!:   SingleFBO;
  private curl!:         SingleFBO;

  // ── Geometry ──
  private quadBuf!:      WebGLBuffer;

  // ── Pointer state ──
  private mouse  = { x: 0.5, y: 0.5, px: 0.5, py: 0.5, down: false };
  private touches = new Map<number, { x: number; y: number }>();
  private splatIdx  = 0;

  // ── Event listeners (kept for removeEventListener) ──
  private _onPointerDown!: (e: PointerEvent) => void;
  private _onPointerMove!: (e: PointerEvent) => void;
  private _onPointerUp!:   (e: PointerEvent) => void;

  private linearFiltering = false;
  private built = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor + factory
  // ─────────────────────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, config: ATMouseFluidConfig = {}) {
    this.gl  = gl;
    this.cfg = {
      simWidth:            config.simWidth            ?? 128,
      simHeight:           config.simHeight           ?? 128,
      dyeWidth:            config.dyeWidth            ?? 512,
      dyeHeight:           config.dyeHeight           ?? 512,
      pressureIterations:  config.pressureIterations  ?? 25,
      curl:                config.curl                ?? 20,
      splatRadius:         config.splatRadius         ?? 0.25,
      velocityDissipation: config.velocityDissipation ?? 0.98,
      densityDissipation:  config.densityDissipation  ?? 0.97,
      pressureDissipation: config.pressureDissipation ?? 0.8,
      dt:                  config.dt                  ?? 1 / 60,
      splatBlendMode:      config.splatBlendMode      ?? 0.0,
    };
  }

  /** Create, init GPU resources, optionally attach pointer listeners. */
  static create(
    gl:     WebGLRenderingContext,
    canvas?: HTMLCanvasElement | null,
    config?: ATMouseFluidConfig,
  ): ATMouseFluid {
    const fluid = new ATMouseFluid(gl, config);
    fluid.init();
    if (canvas) fluid.attachCanvas(canvas);
    return fluid;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // init() — all gl.create* calls happen here
  // ─────────────────────────────────────────────────────────────────────────

  init(): void {
    if (this.built) return;
    const gl  = this.gl;

    // Check OES_texture_float / half-float support
    const extHF  = gl.getExtension('OES_texture_half_float');
    const extHFL = gl.getExtension('OES_texture_half_float_linear');
    const extF   = gl.getExtension('OES_texture_float');
    const extFL  = gl.getExtension('OES_texture_float_linear');
    const floatType = extHF  ? extHF.HALF_FLOAT_OES  :
                      extF   ? gl.FLOAT              : gl.UNSIGNED_BYTE;
    this.linearFiltering = !!(extHFL || extFL);

    const filter = this.linearFiltering ? gl.LINEAR : gl.NEAREST;

    // ── Compile all shader programs ──
    this.splatProg    = this._compileProgram(SIMPLE_VERT, SPLAT_FRAG,             'splat');
    this.curlProg     = this._compileProgram(FLUID_VERT,  CURL_FRAG,              'curl');
    this.vortProg     = this._compileProgram(FLUID_VERT,  VORTICITY_FRAG,         'vorticity');
    this.divProg      = this._compileProgram(FLUID_VERT,  DIVERGENCE_FRAG,        'divergence');
    this.preProg      = this._compileProgram(FLUID_VERT,  PRESSURE_FRAG,          'pressure');
    this.gradSubProg  = this._compileProgram(FLUID_VERT,  GRADIENT_SUBTRACT_FRAG, 'gradSub');
    this.clearProg    = this._compileProgram(SIMPLE_VERT, CLEAR_FRAG,             'clear');
    this.advectProg   = this._compileProgram(SIMPLE_VERT, ADVECTION_FRAG,         'advect');
    this.advectManProg = this._compileProgram(SIMPLE_VERT, ADVECTION_MANUAL_FRAG, 'advectManual');

    // ── Create FBOs ──
    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.cfg;
    const fmt = gl.RGBA;

    this.velocity   = this._makeDoubleFBO(sw, sh, fmt, fmt, floatType, filter);
    this.pressure   = this._makeDoubleFBO(sw, sh, fmt, fmt, floatType, filter);
    this.dye        = this._makeDoubleFBO(dw, dh, fmt, fmt, floatType, filter);
    this.divergence = this._makeSingleFBO(sw, sh, fmt, fmt, floatType, gl.NEAREST);
    this.curl       = this._makeSingleFBO(sw, sh, fmt, fmt, floatType, gl.NEAREST);

    // ── Fullscreen quad geometry ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.built = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // attachCanvas / detachCanvas — mousemove splat injection
  // ─────────────────────────────────────────────────────────────────────────

  attachCanvas(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;

    this._onPointerDown = (e: PointerEvent) => {
      canvas.setPointerCapture(e.pointerId);
      const x = e.offsetX / canvas.clientWidth;
      const y = 1.0 - e.offsetY / canvas.clientHeight;
      this.touches.set(e.pointerId, { x, y });
      if (e.isPrimary) {
        this.mouse.down = true;
        this.mouse.px = this.mouse.x;
        this.mouse.py = this.mouse.y;
        this.mouse.x  = x;
        this.mouse.y  = y;
      }
    };

    this._onPointerMove = (e: PointerEvent) => {
      const prev = this.touches.get(e.pointerId);
      const x = e.offsetX / canvas.clientWidth;
      const y = 1.0 - e.offsetY / canvas.clientHeight;
      if (prev) {
        const dx = (x - prev.x);
        const dy = (y - prev.y);
        if (Math.abs(dx) > 0.0002 || Math.abs(dy) > 0.0002) {
          const aspect = canvas.clientWidth / canvas.clientHeight;
          const [r, g, b] = nextSplatColor(this.splatIdx++);
          this._splat(x, y, prev.x, prev.y,
                      dx * aspect * 8.0,
                      dy * 8.0,
                      r, g, b);
        }
      }
      this.touches.set(e.pointerId, { x, y });
      if (e.isPrimary) {
        this.mouse.px = this.mouse.x;
        this.mouse.py = this.mouse.y;
        this.mouse.x  = x;
        this.mouse.y  = y;
      }
    };

    this._onPointerUp = (e: PointerEvent) => {
      this.touches.delete(e.pointerId);
      if (e.isPrimary) this.mouse.down = false;
    };

    canvas.addEventListener('pointerdown',  this._onPointerDown);
    canvas.addEventListener('pointermove',  this._onPointerMove);
    canvas.addEventListener('pointerup',    this._onPointerUp);
    canvas.addEventListener('pointerleave', this._onPointerUp);
    canvas.addEventListener('pointercancel',this._onPointerUp);
  }

  detachCanvas(): void {
    if (!this.canvas) return;
    const c = this.canvas;
    c.removeEventListener('pointerdown',   this._onPointerDown);
    c.removeEventListener('pointermove',   this._onPointerMove);
    c.removeEventListener('pointerup',     this._onPointerUp);
    c.removeEventListener('pointerleave',  this._onPointerUp);
    c.removeEventListener('pointercancel', this._onPointerUp);
    this.canvas = null;
    this.touches.clear();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick() / render() — per-frame NS pass chain
  // ─────────────────────────────────────────────────────────────────────────

  /** Run one full NS frame (no manual splat — driven by pointer listeners). */
  tick(dt: number = this.cfg.dt): void {
    if (!this.built) return;
    this._step(dt);
  }

  /** Alias: render(dt) == tick(dt). */
  render(dt?: number): void {
    this.tick(dt);
  }

  /**
   * Full NS step — called by tick().
   * Passes: curl → vorticity → divergence → pressure clear →
   *         pressure solve × N → gradient subtract → advect vel → advect dye.
   */
  private _step(dt: number): void {
    const gl = this.gl;
    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.cfg;
    const c = this.cfg;

    // ── Pass 1: CURL ──────────────────────────────────────────────────────
    gl.useProgram(this.curlProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.curl.fbo);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.curlProg, 'uVelocity'), 0);
    gl.uniform2f(gl.getUniformLocation(this.curlProg, 'texelSize'), 1.0 / sw, 1.0 / sh);
    this._drawQuad(this.curlProg);

    // ── Pass 2: VORTICITY CONFINEMENT ─────────────────────────────────────
    gl.useProgram(this.vortProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.vortProg, 'uVelocity'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curl.tex);
    gl.uniform1i(gl.getUniformLocation(this.vortProg, 'uCurl'), 1);
    gl.uniform2f(gl.getUniformLocation(this.vortProg, 'texelSize'), 1.0 / sw, 1.0 / sh);
    gl.uniform1f(gl.getUniformLocation(this.vortProg, 'curl'), c.curl);
    gl.uniform1f(gl.getUniformLocation(this.vortProg, 'dt'), dt);
    this._drawQuad(this.vortProg);
    this._swapVelocity();

    // ── Pass 3: DIVERGENCE ────────────────────────────────────────────────
    gl.useProgram(this.divProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.divergence.fbo);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.divProg, 'uVelocity'), 0);
    gl.uniform2f(gl.getUniformLocation(this.divProg, 'texelSize'), 1.0 / sw, 1.0 / sh);
    this._drawQuad(this.divProg);

    // ── Pass 4: PRESSURE CLEAR ────────────────────────────────────────────
    gl.useProgram(this.clearProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pressure.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.readTex);
    gl.uniform1i(gl.getUniformLocation(this.clearProg, 'uTexture'), 0);
    gl.uniform1f(gl.getUniformLocation(this.clearProg, 'value'), c.pressureDissipation);
    this._drawQuad(this.clearProg);
    this._swapPressure();

    // ── Pass 5: PRESSURE SOLVE (Jacobi × N) ──────────────────────────────
    gl.useProgram(this.preProg);
    gl.uniform2f(gl.getUniformLocation(this.preProg, 'texelSize'), 1.0 / sw, 1.0 / sh);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.divergence.tex);
    gl.uniform1i(gl.getUniformLocation(this.preProg, 'uDivergence'), 1);
    for (let i = 0; i < c.pressureIterations; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pressure.write);
      gl.viewport(0, 0, sw, sh);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.pressure.readTex);
      gl.uniform1i(gl.getUniformLocation(this.preProg, 'uPressure'), 0);
      this._drawQuad(this.preProg);
      this._swapPressure();
    }

    // ── Pass 6: GRADIENT SUBTRACT ─────────────────────────────────────────
    gl.useProgram(this.gradSubProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.readTex);
    gl.uniform1i(gl.getUniformLocation(this.gradSubProg, 'uPressure'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.gradSubProg, 'uVelocity'), 1);
    gl.uniform2f(gl.getUniformLocation(this.gradSubProg, 'texelSize'), 1.0 / sw, 1.0 / sh);
    this._drawQuad(this.gradSubProg);
    this._swapVelocity();

    // ── Pass 7a: ADVECT VELOCITY ──────────────────────────────────────────
    const advProg = this.linearFiltering ? this.advectProg : this.advectManProg;
    gl.useProgram(advProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(advProg, 'uVelocity'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(advProg, 'uSource'), 1);
    gl.uniform2f(gl.getUniformLocation(advProg, 'texelSize'),    1.0 / sw, 1.0 / sh);
    gl.uniform2f(gl.getUniformLocation(advProg, 'dyeTexelSize'), 1.0 / sw, 1.0 / sh);
    gl.uniform1f(gl.getUniformLocation(advProg, 'dt'), dt);
    gl.uniform1f(gl.getUniformLocation(advProg, 'dissipation'), c.velocityDissipation);
    this._drawQuad(advProg);
    this._swapVelocity();

    // ── Pass 7b: ADVECT DYE ───────────────────────────────────────────────
    gl.useProgram(advProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dye.write);
    gl.viewport(0, 0, dw, dh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(advProg, 'uVelocity'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    gl.uniform1i(gl.getUniformLocation(advProg, 'uSource'), 1);
    gl.uniform2f(gl.getUniformLocation(advProg, 'texelSize'),    1.0 / sw, 1.0 / sh);
    gl.uniform2f(gl.getUniformLocation(advProg, 'dyeTexelSize'), 1.0 / dw, 1.0 / dh);
    gl.uniform1f(gl.getUniformLocation(advProg, 'dt'), dt);
    gl.uniform1f(gl.getUniformLocation(advProg, 'dissipation'), c.densityDissipation);
    this._drawQuad(advProg);
    this._swapDye();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _splat() — AT splatShader.fs: line-segment splat via mousemove
  // ─────────────────────────────────────────────────────────────────────────

  private _splat(
    x: number, y: number,
    px: number, py: number,
    dx: number, dy: number,
    r: number, g: number, b: number,
  ): void {
    const gl = this.gl;
    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = this.cfg;

    // ── Velocity splat ──
    gl.useProgram(this.splatProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    gl.viewport(0, 0, sw, sh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'), 0);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'point'),     x,  y);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'prevPoint'), px, py);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), sw / sh);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'), dx, dy, 1.0);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'bgColor'), 0.0, 0.0, 0.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'radius'), this.cfg.splatRadius / 100.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'canRender'), 1.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'), 1.0);
    this._drawQuad(this.splatProg);
    this._swapVelocity();

    // ── Dye splat ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dye.write);
    gl.viewport(0, 0, dw, dh);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'), 0);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'point'),     x,  y);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'prevPoint'), px, py);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), dw / dh);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'), r, g, b);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'bgColor'), 0.0, 0.0, 0.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'radius'), this.cfg.splatRadius / 100.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'canRender'), 1.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'), this.cfg.splatBlendMode);
    this._drawQuad(this.splatProg);
    this._swapDye();
  }

  /** Inject a splat manually (programmatic/idle ripple). */
  splat(
    x: number, y: number,
    dx: number, dy: number,
    r = 0.8, g = 0.3, b = 0.1,
  ): void {
    const px = x - dx * 0.01;
    const py = y - dy * 0.01;
    this._splat(x, y, px, py, dx, dy, r, g, b);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Output accessors
  // ─────────────────────────────────────────────────────────────────────────

  /** tFluid — dye (density) texture for downstream shaders (getFluidVelocity). */
  get dyeTexture(): WebGLTexture      { return this.dye.readTex; }
  /** tFluidMask / velocity field — XY velocity for AT mousefluid.fs consumer. */
  get velocityTexture(): WebGLTexture { return this.velocity.readTex; }

  // ─────────────────────────────────────────────────────────────────────────
  // dispose() — all gl.delete* calls
  // ─────────────────────────────────────────────────────────────────────────

  dispose(): void {
    if (!this.built) return;
    this.detachCanvas();
    const gl = this.gl;

    // Delete all programs
    gl.deleteProgram(this.splatProg);
    gl.deleteProgram(this.curlProg);
    gl.deleteProgram(this.vortProg);
    gl.deleteProgram(this.divProg);
    gl.deleteProgram(this.preProg);
    gl.deleteProgram(this.gradSubProg);
    gl.deleteProgram(this.clearProg);
    gl.deleteProgram(this.advectProg);
    gl.deleteProgram(this.advectManProg);

    // Delete double FBO textures + framebuffers
    gl.deleteFramebuffer(this.velocity.read);
    gl.deleteFramebuffer(this.velocity.write);
    gl.deleteTexture(this.velocity.readTex);
    gl.deleteTexture(this.velocity.writeTex);

    gl.deleteFramebuffer(this.pressure.read);
    gl.deleteFramebuffer(this.pressure.write);
    gl.deleteTexture(this.pressure.readTex);
    gl.deleteTexture(this.pressure.writeTex);

    gl.deleteFramebuffer(this.dye.read);
    gl.deleteFramebuffer(this.dye.write);
    gl.deleteTexture(this.dye.readTex);
    gl.deleteTexture(this.dye.writeTex);

    // Delete single FBOs
    gl.deleteFramebuffer(this.divergence.fbo);
    gl.deleteTexture(this.divergence.tex);
    gl.deleteFramebuffer(this.curl.fbo);
    gl.deleteTexture(this.curl.tex);

    // Delete quad buffer
    gl.deleteBuffer(this.quadBuf);

    this.built = false;
  }

  /** Alias. */
  destroy(): void { this.dispose(); }

  // ─────────────────────────────────────────────────────────────────────────
  // Runtime param update
  // ─────────────────────────────────────────────────────────────────────────

  updateConfig(patch: Partial<ATMouseFluidConfig>): void {
    Object.assign(this.cfg, patch);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: shader compilation
  // ─────────────────────────────────────────────────────────────────────────

  /** Compile vertex + fragment → linked WebGLProgram. */
  private _compileProgram(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATMouseFluid] vertex compile error (${label}): ${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ATMouseFluid] fragment compile error (${label}): ${log}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[ATMouseFluid] link error (${label}): ${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: FBO creation
  // ─────────────────────────────────────────────────────────────────────────

  private _makeSingleFBO(
    w: number, h: number,
    internalFmt: number, fmt: number, type: number,
    filter: number,
  ): SingleFBO {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, w, h, 0, fmt, type, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return { fbo, tex };
  }

  private _makeDoubleFBO(
    w: number, h: number,
    internalFmt: number, fmt: number, type: number,
    filter: number,
  ): DoubleFBO {
    const a = this._makeSingleFBO(w, h, internalFmt, fmt, type, filter);
    const b = this._makeSingleFBO(w, h, internalFmt, fmt, type, filter);
    return {
      read:     a.fbo,
      write:    b.fbo,
      readTex:  a.tex,
      writeTex: b.tex,
      width:    w,
      height:   h,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: geometry
  // ─────────────────────────────────────────────────────────────────────────

  private _drawQuad(prog: WebGLProgram): void {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private: ping-pong swaps
  // ─────────────────────────────────────────────────────────────────────────

  private _swapVelocity(): void {
    [this.velocity.read,    this.velocity.write   ] = [this.velocity.write,    this.velocity.read   ];
    [this.velocity.readTex, this.velocity.writeTex] = [this.velocity.writeTex, this.velocity.readTex];
  }
  private _swapPressure(): void {
    [this.pressure.read,    this.pressure.write   ] = [this.pressure.write,    this.pressure.read   ];
    [this.pressure.readTex, this.pressure.writeTex] = [this.pressure.writeTex, this.pressure.readTex];
  }
  private _swapDye(): void {
    [this.dye.read,    this.dye.write   ] = [this.dye.write,    this.dye.read   ];
    [this.dye.readTex, this.dye.writeTex] = [this.dye.writeTex, this.dye.readTex];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Quick-create: build ATMouseFluid, attach canvas pointer listeners.
 * Drop-in replacement for the old WebGPU version.
 *
 * Usage:
 *   const fluid = createATMouseFluid(gl, canvas);
 *   // rAF loop:
 *   fluid.tick(dt);
 *   // bind fluid.dyeTexture / fluid.velocityTexture downstream
 */
export function createATMouseFluid(
  gl:      WebGLRenderingContext,
  canvas:  HTMLCanvasElement,
  config?: ATMouseFluidConfig,
): ATMouseFluid {
  return ATMouseFluid.create(gl, canvas, config);
}

export default ATMouseFluid;
export const AT_FLUID_BASE_VS = ''; export const AT_FLUID_SPLAT_FS = ''; export const AT_FLUID_ADVECT_FS = ''; export const AT_FLUID_CURL_FS = ''; export const AT_FLUID_VORTICITY_FS = ''; export const AT_FLUID_DIVERGE_FS = ''; export const AT_FLUID_PRESSURE_FS = ''; export const AT_FLUID_GRADSUB_FS = ''; export const AT_FLUID_DISPLAY_FS = '';
export const AT_SPLAT_FS = ''; export const AT_ADVECTION_FS = ''; export const AT_ADVECTION_MANUAL_FS = ''; export const AT_CURL_FS = ''; export const AT_VORTICITY_FS = ''; export const AT_DIVERGENCE_FS = ''; export const AT_PRESSURE_FS = ''; export const AT_GRADIENT_SUBTRACT_FS = ''; export const AT_CLEAR_FS = '';
