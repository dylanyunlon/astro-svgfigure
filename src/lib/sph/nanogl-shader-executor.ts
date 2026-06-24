/**
 * nanogl-shader-executor.ts — M906: nanogl shader executor
 * ─────────────────────────────────────────────────────────────────────────────
 * 用 upstream/nanogl 把已提取的 GLSL shader 字符串真正编译执行。
 * 对接 at-mousefluid-import.ts 的 11 个流体 shader，实现完整 WebGL1/2
 * multi-pass FBO 链。
 *
 *   compileShader(gl, vert, frag)   → nanogl Program（编译+链接）
 *   executePasses(gl, passes[])     → 按顺序执行所有 pass，返回最终 FBO
 *
 * 管线顺序（对接 at-mousefluid-import.ts pass 注释）：
 *   Pass 0  SPLAT         — velocity + density splat（每个 pointer 各一次）
 *   Pass 1  CURL          — 2D vorticity ω
 *   Pass 2  VORTICITY     — vorticity confinement force
 *   Pass 3  DIVERGENCE    — ∇·v with boundary reflection
 *   Pass 4  PRESSURE_CLEAR— p *= pressureDissipation
 *   Pass 5  PRESSURE×N    — Jacobi iteration（N 次 ping-pong）
 *   Pass 6  GRADIENT_SUB  — v -= ∇p
 *   Pass 7a ADVECT_VEL    — velocity self-advection
 *   Pass 7b ADVECT_DYE    — dye advection
 *
 * 每个 pass:
 *   createFBO → bind → program.use() → setUniforms → drawFullscreenQuad → unbind
 *
 * 参考：
 *   upstream/nanogl/src/program.ts    — Program（compile / use / uniform accessors）
 *   upstream/nanogl/src/fbo.ts        — Fbo（bind / attachColor / getColorTexture）
 *   upstream/nanogl/src/arraybuffer.ts— ArrayBuffer（attrib / attribPointer / draw）
 *   upstream/nanogl/src/texture-2d.ts — Texture2D（fromData / bind / setFilter / clamp）
 *   src/lib/sph/at-mousefluid-import.ts — 11 个 GLSL shader 源
 *   src/lib/sph/at-render-pipeline.ts   — pass 顺序描述
 *
 * Research: xiaodi #M906 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─────────────────────────────────────────────────────────────────────────────
// Upstream nanogl imports (vendored at upstream/nanogl/src/)
// ─────────────────────────────────────────────────────────────────────────────




// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single fullscreen shader pass.
 * Every field mirrors what the AT fluid pipeline passes per-frame.
 */



import Program       from '../../../upstream/nanogl/src/program';
import Fbo           from '../../../upstream/nanogl/src/fbo';
import NanoGLBuffer  from '../../../upstream/nanogl/src/arraybuffer';
import Texture2D     from '../../../upstream/nanogl/src/texture-2d';
import type { GLContext } from '../../../upstream/nanogl/src/types';
import {

// ─────────────────────────────────────────────────────────────────────────────
// AT fluid shader sources (11 GLSL strings)
// ─────────────────────────────────────────────────────────────────────────────

  AT_FLUID_BASE_VS,
  AT_SPLAT_FS,
  AT_ADVECTION_FS,
  AT_ADVECTION_MANUAL_FS,
  AT_CURL_FS,
  AT_VORTICITY_FS,
  AT_DIVERGENCE_FS,
  AT_PRESSURE_FS,
  AT_GRADIENT_SUBTRACT_FS,
  AT_CLEAR_FS,
  AT_MOUSEFLUID_CONSUME_FS,
} from './at-mousefluid-import';

<<<<<<< HEAD
// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /**
// [orphan-precise]  * A single fullscreen shader pass.
// [orphan-precise]  * Every field mirrors what the AT fluid pipeline passes per-frame.
// [orphan-precise]  */
=======
>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface ShaderPass {
  /** Human-readable label (used for debug / error messages). */
  name: string;
  /** Compiled nanogl Program. */
  program: Program;
  /** Target FBO — null means render to the default framebuffer. */
  fbo: Fbo | null;
  /**
   * Uniform setter called just before draw.
   * The function receives the compiled Program so callers can set uniforms
   * via nanogl's dynamic accessor syntax: `prog.uMyUniform(value)`.
   */
  setUniforms: (prog: Program) => void;
  /** Texture units to bind before draw: [unit, Texture2D][] pairs. */
  textures?: [number, Texture2D][];
}

/**
 * A ping-pong FBO pair used by every fluid simulation field.
 * Callers swap read/write each iteration; the executor itself is stateless
 * with respect to ping-pong — the `ShaderPass` always points at the
 * write FBO and the caller handles swapping.
 */
export interface PingPongFbo {
  read:  Fbo;
  write: Fbo;
}

/**
 * Configuration for NanoGLFluidExecutor.create().
 */
export interface FluidExecutorConfig {
  /** Simulation field resolution (square). Default 128. */
  simRes?: number;
  /** Dye texture resolution (square). Default 512. */
  dyeRes?: number;
  /** Jacobi pressure iterations per frame. Default 4. */
  pressureIters?: number;
  /** Vorticity confinement strength. Default 20. */
  curlStrength?: number;
  /** Splat radius in UV space. Default 0.025 (AT normalised: 0.25/100). */
  splatRadius?: number;
  /** Velocity dissipation per frame. Default 0.98. */
  velocityDissipation?: number;
  /** Dye density dissipation per frame. Default 0.97. */
  densityDissipation?: number;
  /** Pressure dissipation per frame. Default 0.8. */
  pressureDissipation?: number;
  /** Splat uAdd — 0.0 = screen blend (AT default), 1.0 = additive. */
  splatBlendMode?: number;
  /** Simulation dt in seconds. Default 1/60. */
  dt?: number;
  /**
   * Whether to use the manual-filtering advection shader (AT fallback for
   * devices that don't support linear filtering on float textures).
   * Default false (uses the standard linear-filter version).
   */
  useManualAdvection?: boolean;
}

/** A pending pointer splat from user interaction. */
export interface PointerSplat {
  /** Current UV (0–1). */
  x: number;
  y: number;
  /** Previous UV (0–1). */
  px: number;
  py: number;
  /** Velocity delta (screen pixels × scale). */
  dx: number;
  dy: number;
  /** Dye colour RGB (0–1). */
  r: number;
  g: number;
  b: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** WebGL1/2 extension cache per context. */
const _extCache = new WeakMap<WebGLRenderingContext | WebGL2RenderingContext, {
  halfFloat?: OES_texture_half_float | null;
  linearHalfFloat?: OES_texture_half_float_linear | null;
  colorBufferFloat?: EXT_color_buffer_float | null;
}>();

function _getExts(gl: GLContext) {
  const ctx = gl as WebGLRenderingContext | WebGL2RenderingContext;
  if (!_extCache.has(ctx)) {
    _extCache.set(ctx, {
      halfFloat:       ctx.getExtension('OES_texture_half_float'),
      linearHalfFloat: ctx.getExtension('OES_texture_half_float_linear'),
      colorBufferFloat: ctx.getExtension('EXT_color_buffer_float'),
    });
  }
  return _extCache.get(ctx)!;
}

/**
 * Resolve the correct GL enums for an rgba16float texture on this context.
 * Returns { format, type, internal }.
 *
 * WebGL2: internal=RGBA16F, format=RGBA, type=HALF_FLOAT
 * WebGL1 + OES_texture_half_float: internal=RGBA, format=RGBA, type=HALF_FLOAT_OES
 * Fallback: internal=RGBA, format=RGBA, type=FLOAT
 */
function _halfFloatEnums(gl: GLContext): { format: number; type: number; internal: number } {
  const ctx = gl as WebGLRenderingContext & WebGL2RenderingContext;
  const isGL2 = !!(ctx as WebGL2RenderingContext).texStorage2D;

  if (isGL2) {
    return {
      internal: 0x881A, // RGBA16F
      format:   ctx.RGBA,
      type:     0x140B, // HALF_FLOAT
    };
  }

  const ext = _getExts(gl);
  if (ext.halfFloat) {
    const HALF_FLOAT_OES = 0x8D61;
    return { internal: ctx.RGBA, format: ctx.RGBA, type: HALF_FLOAT_OES };
  }

  // Last resort: full float
  return { internal: ctx.RGBA, format: ctx.RGBA, type: ctx.FLOAT };
}

/**
 * Create a Texture2D allocated to `w × h` with half-float enums.
 * Applies linear filtering and clamp-to-edge wrapping.
 */
function _createHalfFloatTex(gl: GLContext, w: number, h: number, linear = true): Texture2D {
  const { format, type, internal } = _halfFloatEnums(gl);
  const tex = new Texture2D(gl, format, type, internal);
  tex.setFilter(linear);
  tex.clamp();
  tex.fromData(w, h, null);
  return tex;
}

/**
 * Create a single-channel (R) FBO-backing texture (for pressure/curl/div).
 * Falls back to RGBA if R-only float isn't supported.
 */
function _createScalarTex(gl: GLContext, w: number, h: number): Texture2D {
  const ctx = gl as WebGLRenderingContext & WebGL2RenderingContext;
  const isGL2 = !!(ctx as WebGL2RenderingContext).texStorage2D;

  let format: number, type: number, internal: number;
  if (isGL2) {
    // WebGL2: R16F
    format   = 0x1903; // RED
    type     = 0x140B; // HALF_FLOAT
    internal = 0x822D; // R16F
  } else {
    // WebGL1: fallback to RGBA half-float (R channel only used)
    const { format: f, type: t, internal: i } = _halfFloatEnums(gl);
    format = f; type = t; internal = i;
  }

  const tex = new Texture2D(gl, format, type, internal);
  tex.setFilter(false); // nearest for pressure/curl/div
  tex.clamp();
  tex.fromData(w, h, null);
  return tex;
}

/**
 * Allocate an FBO and attach a colour texture.
 * The FBO is immediately bound, attached, then unbound.
 */
function _createFboWithTex(gl: GLContext, tex: Texture2D): Fbo {
  const ctx = gl as WebGLRenderingContext;
  const fbo = new Fbo(gl);
  fbo.resize(tex.width, tex.height);
  fbo.bind();
  fbo.attach(ctx.COLOR_ATTACHMENT0, tex);
  ctx.bindFramebuffer(ctx.FRAMEBUFFER, null);
  return fbo;
}

/**
 * Create a ping-pong pair: two FBOs each backed by a half-float RGBA texture.
 */
function _createPingPong(
  gl: GLContext,
  w: number,
  h: number,
  scalar = false,
): PingPongFbo {
  const texA = scalar ? _createScalarTex(gl, w, h) : _createHalfFloatTex(gl, w, h);
  const texB = scalar ? _createScalarTex(gl, w, h) : _createHalfFloatTex(gl, w, h);
  return {
    read:  _createFboWithTex(gl, texA),
    write: _createFboWithTex(gl, texB),
  };
}

/** Swap read/write in a PingPongFbo. */
function _swap(pp: PingPongFbo): void {
  const tmp = pp.read;
  pp.read   = pp.write;
  pp.write  = tmp;
}

/** Get the colour texture from an FBO (attachment 0). */
function _colorTex(fbo: Fbo): Texture2D {
  return fbo.getColorTexture(0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fullscreen quad helper (shared per-context)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A reusable fullscreen quad NanoGLBuffer.
 * Two triangles covering clip-space [-1,1] × [-1,1], with UV [0,1] × [0,1].
 * `position` is a vec3 (z=0), `uv` is a vec2.
 */
class FullscreenQuad {
  private _buf: NanoGLBuffer;

  constructor(gl: GLContext) {
    // prettier-ignore
    const verts = new Float32Array([
      // x     y     z    u    v
      -1.0, -1.0,  0.0,  0.0, 0.0,
       1.0, -1.0,  0.0,  1.0, 0.0,
      -1.0,  1.0,  0.0,  0.0, 1.0,
       1.0,  1.0,  0.0,  1.0, 1.0,
    ]);

    this._buf = new NanoGLBuffer(gl, verts, gl.STATIC_DRAW);
    this._buf
      .attrib('position', 3, gl.FLOAT)
      .attrib('uv',       2, gl.FLOAT);
  }

  /** Bind the buffer, link attributes to program, and draw. */
  draw(program: Program): void {
    this._buf.attribPointer(program);
    this._buf.draw((this._buf.gl as WebGLRenderingContext).TRIANGLE_STRIP, 4, 0);
  }

  dispose(): void {
    (this._buf.gl as WebGLRenderingContext).deleteBuffer(this._buf.buffer);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// compileShader — public primitive
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compile and link a GLSL vertex + fragment shader pair using nanogl Program.
 *
 * Automatically prepends a `precision mediump float;` guard when the fragment
 * shader doesn't already declare a default float precision — required by
 * WebGL1 / some WebGL2 contexts.
 *
 * @param gl       The WebGL rendering context.
 * @param vertSrc  GLSL vertex shader source (without `#version` header).
 * @param fragSrc  GLSL fragment shader source.
 * @param defs     Optional `#define` prefix injected into both shaders.
 * @returns        A compiled, linked nanogl `Program`.
 * @throws         If compilation or linking fails (Program.debug is set true).
 */
export function compileShader(
  gl:      GLContext,
  vertSrc: string,
  fragSrc: string,
  defs?:   string,
): Program {
  Program.debug = true; // always log compile errors

  // Ensure fragment shader has a default float precision declaration.
  const hasPrecision = /^\s*precision\s+\w+\s+float\s*;/m.test(fragSrc);
  const guardedFrag  = hasPrecision ? fragSrc : `precision mediump float;\n${fragSrc}`;

  const prog = new Program(gl, vertSrc, guardedFrag, defs);

  if (!prog.ready) {
    throw new Error(
      `[nanogl-shader-executor] compileShader failed — check console for GLSL errors.`,
    );
  }

  return prog;
}

// ─────────────────────────────────────────────────────────────────────────────
// executePasses — public multi-pass executor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Execute an ordered array of `ShaderPass` descriptors.
 *
 * For each pass:
 *   1. Bind the target FBO (or null → default framebuffer).
 *   2. Set the viewport to the FBO dimensions.
 *   3. Call `prog.use()`.
 *   4. Bind all declared textures to their units.
 *   5. Call `setUniforms(prog)`.
 *   6. Draw the fullscreen quad.
 *   7. Unbind FBO.
 *
 * @param gl     The WebGL rendering context.
 * @param passes Ordered list of pass descriptors.
 * @param quad   Optional shared FullscreenQuad; created internally if omitted.
 * @returns      The last pass's FBO (or null for screen-space output).
 */
export function executePasses(
  gl:     GLContext,
  passes: ShaderPass[],
  quad?:  FullscreenQuad,
): Fbo | null {
  const ctx        = gl as WebGLRenderingContext;
  const ownQuad    = quad === undefined;
  const _quad      = ownQuad ? new FullscreenQuad(gl) : quad!;
  let   lastFbo: Fbo | null = null;

  for (const pass of passes) {
    const { program: prog, fbo, setUniforms, textures } = pass;

    // — bind target —
    if (fbo) {
      fbo.bind();
      fbo.defaultViewport();
    } else {
      ctx.bindFramebuffer(ctx.FRAMEBUFFER, null);
    }

    // — use program —
    prog.use();

    // — bind textures —
    if (textures) {
      for (const [unit, tex] of textures) {
        tex.bind(unit);
      }
    }

    // — uniforms (called after use() so nanogl's dynamic setters are ready) —
    setUniforms(prog);

    // — draw —
    _quad.draw(prog);

    // — unbind —
    ctx.bindFramebuffer(ctx.FRAMEBUFFER, null);

    lastFbo = fbo;
  }

  if (ownQuad) _quad.dispose();

  return lastFbo;
}

// ─────────────────────────────────────────────────────────────────────────────
// NanoGLFluidExecutor — full AT fluid pipeline over WebGL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NanoGLFluidExecutor
 *
 * Compiles all 11 AT fluid GLSL shaders with nanogl and executes the full
 * Navier-Stokes fluid simulation pipeline each frame via WebGL FBOs.
 *
 * Mirrors the WebGPU ATMouseFluid pipeline from at-mousefluid-import.ts
 * exactly, but runs on WebGL1/2 via nanogl:
 *
 *   create(gl, canvas?, config?) → executor
 *   executor.step()              → run one frame
 *   executor.getDyeTex()         → Texture2D  (tFluid for downstream shaders)
 *   executor.getVelocityTex()    → Texture2D
 *   executor.enqueueSplat(s)     → add a pointer splat
 *   executor.destroy()           → release all GL resources
 *
 * Usage:
 *   ```ts
 *   import { NanoGLFluidExecutor } from '$lib/sph/nanogl-shader-executor';
 *
 *   const exec = NanoGLFluidExecutor.create(gl, canvas, { simRes: 128 });
 *
 *   // rAF loop:
 *   exec.step();
 *   const dyeTex = exec.getDyeTex(); // bind to tFluid
 *   ```
 */
export class NanoGLFluidExecutor {
  // ── config ──
  readonly simRes: number;
  readonly dyeRes: number;
  private _pressureIters:       number;
  private _curlStrength:        number;
  private _splatRadius:         number;
  private _velocityDissipation: number;
  private _densityDissipation:  number;
  private _pressureDissipation: number;
  private _splatBlendMode:      number;
  private _dt:                  number;
  private _useManualAdvection:  boolean;

  // ── GL context ──
  private _gl:   GLContext;
  private _quad: FullscreenQuad;

  // ── FBO ping-pong fields ──
  private _velocity:  PingPongFbo;
  private _density:   PingPongFbo;
  private _pressure:  PingPongFbo;
  /** Single-write FBOs (not ping-pong). */
  private _curlFbo:   Fbo;
  private _divFbo:    Fbo;

  // ── Compiled programs ──
  private _pSplat:         Program;
  private _pCurl:          Program;
  private _pVorticity:     Program;
  private _pDivergence:    Program;
  private _pPressure:      Program;
  private _pGradSubtract:  Program;
  private _pClear:         Program;
  private _pAdvect:        Program;

  // ── Pointer / splat state ──
  private _splatQueue: PointerSplat[] = [];
  private _pointerMap = new Map<number, { x: number; y: number }>();
  private _splatIndex = 0;
  private _canvas: HTMLCanvasElement | null = null;
  private _onDown?:  (e: PointerEvent) => void;
  private _onMove?:  (e: PointerEvent) => void;
  private _onUp?:    (e: PointerEvent) => void;

  private _destroyed = false;

  // ── Constructor (private — use create()) ──

  private constructor(gl: GLContext, cfg: Required<FluidExecutorConfig>) {
    this._gl                   = gl;
    this.simRes                = cfg.simRes;
    this.dyeRes                = cfg.dyeRes;
    this._pressureIters        = cfg.pressureIters;
    this._curlStrength         = cfg.curlStrength;
    this._splatRadius          = cfg.splatRadius;
    this._velocityDissipation  = cfg.velocityDissipation;
    this._densityDissipation   = cfg.densityDissipation;
    this._pressureDissipation  = cfg.pressureDissipation;
    this._splatBlendMode       = cfg.splatBlendMode;
    this._dt                   = cfg.dt;
    this._useManualAdvection   = cfg.useManualAdvection;

    // Fullscreen quad — shared across all passes
    this._quad = new FullscreenQuad(gl);

    // ── FBOs ──────────────────────────────────────────────────────────────
    this._velocity = _createPingPong(gl, this.simRes, this.simRes);
    this._density  = _createPingPong(gl, this.dyeRes,  this.dyeRes);
    this._pressure = _createPingPong(gl, this.simRes, this.simRes, true);

    const curlTex = _createScalarTex(gl, this.simRes, this.simRes);
    const divTex  = _createScalarTex(gl, this.simRes, this.simRes);
    this._curlFbo = _createFboWithTex(gl, curlTex);
    this._divFbo  = _createFboWithTex(gl, divTex);

    // ── Compile programs ──────────────────────────────────────────────────
    this._pSplat        = compileShader(gl, AT_FLUID_BASE_VS, AT_SPLAT_FS);
    this._pCurl         = compileShader(gl, AT_FLUID_BASE_VS, AT_CURL_FS);
    this._pVorticity    = compileShader(gl, AT_FLUID_BASE_VS, AT_VORTICITY_FS);
    this._pDivergence   = compileShader(gl, AT_FLUID_BASE_VS, AT_DIVERGENCE_FS);
    this._pPressure     = compileShader(gl, AT_FLUID_BASE_VS, AT_PRESSURE_FS);
    this._pGradSubtract = compileShader(gl, AT_FLUID_BASE_VS, AT_GRADIENT_SUBTRACT_FS);
    this._pClear        = compileShader(gl, AT_FLUID_BASE_VS, AT_CLEAR_FS);
    this._pAdvect       = compileShader(
      gl,
      AT_FLUID_BASE_VS,
      cfg.useManualAdvection ? AT_ADVECTION_MANUAL_FS : AT_ADVECTION_FS,
    );
  }

  // ── Factory ──────────────────────────────────────────────────────────────

  /**
   * Create and initialise a NanoGLFluidExecutor.
   *
   * @param gl      WebGL1 or WebGL2 rendering context.
   * @param canvas  Optional canvas element for pointer listener attachment.
   * @param config  Fluid simulation parameters.
   */
  static create(
    gl:      GLContext,
    canvas?: HTMLCanvasElement | null,
    config?: FluidExecutorConfig,
  ): NanoGLFluidExecutor {
    const cfg: Required<FluidExecutorConfig> = {
      simRes:               config?.simRes               ?? 128,
      dyeRes:               config?.dyeRes               ?? 512,
      pressureIters:        config?.pressureIters         ?? 4,
      curlStrength:         config?.curlStrength          ?? 20,
      splatRadius:          config?.splatRadius           ?? 0.025,
      velocityDissipation:  config?.velocityDissipation   ?? 0.98,
      densityDissipation:   config?.densityDissipation    ?? 0.97,
      pressureDissipation:  config?.pressureDissipation   ?? 0.8,
      splatBlendMode:       config?.splatBlendMode        ?? 0.0,
      dt:                   config?.dt                    ?? 1 / 60,
      useManualAdvection:   config?.useManualAdvection    ?? false,
    };

    const exec = new NanoGLFluidExecutor(gl, cfg);
    if (canvas) exec.attachPointerListeners(canvas);
    return exec;
  }

  // ── Pointer listeners ─────────────────────────────────────────────────────

  /**
   * Attach pointer event listeners to a canvas element.
   * Converts pointer events into PointerSplat entries in the queue.
   */
  attachPointerListeners(canvas: HTMLCanvasElement): void {
    this._canvas = canvas;

    this._onDown = (e: PointerEvent) => {
      canvas.setPointerCapture(e.pointerId);
      const x = e.offsetX / canvas.clientWidth;
      const y = 1 - e.offsetY / canvas.clientHeight; // AT Y-flip
      this._pointerMap.set(e.pointerId, { x, y });
    };

    this._onMove = (e: PointerEvent) => {
      const prev = this._pointerMap.get(e.pointerId);
      const x    = e.offsetX / canvas.clientWidth;
      const y    = 1 - e.offsetY / canvas.clientHeight;

      if (!prev) {
        this._pointerMap.set(e.pointerId, { x, y });
        return;
      }

      const dx = (x - prev.x) * canvas.clientWidth  * 5;
      const dy = (y - prev.y) * canvas.clientHeight * 5;

      if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
        const [r, g, b] = _nextColour(this._splatIndex++);
        this._splatQueue.push({ x, y, px: prev.x, py: prev.y, dx, dy, r, g, b });
      }
      this._pointerMap.set(e.pointerId, { x, y });
    };

    this._onUp = (e: PointerEvent) => {
      this._pointerMap.delete(e.pointerId);
    };

    canvas.addEventListener('pointerdown', this._onDown);
    canvas.addEventListener('pointermove', this._onMove);
    canvas.addEventListener('pointerup',   this._onUp);
    canvas.addEventListener('pointerleave', this._onUp);
  }

  /** Detach all pointer listeners from the canvas. */
  detachPointerListeners(): void {
    if (!this._canvas) return;
    if (this._onDown) this._canvas.removeEventListener('pointerdown', this._onDown);
    if (this._onMove) this._canvas.removeEventListener('pointermove', this._onMove);
    if (this._onUp)   {
      this._canvas.removeEventListener('pointerup',    this._onUp);
      this._canvas.removeEventListener('pointerleave', this._onUp);
    }
    this._canvas = null;
    this._onDown = this._onMove = this._onUp = undefined;
  }

  /**
   * Manually enqueue a splat (for programmatic injection, idle ripple, etc.)
   */
  enqueueSplat(s: PointerSplat): void {
    this._splatQueue.push(s);
  }

  // ── Per-frame step ────────────────────────────────────────────────────────

  /**
   * Execute one complete frame of the AT fluid pipeline via nanogl.
   *
   * Order:
   *   0. Flush splat queue (velocity + density splat for each pointer)
   *   1. Curl
   *   2. Vorticity confinement
   *   3. Divergence
   *   4. Pressure clear
   *   5. Pressure solve × N (Jacobi)
   *   6. Gradient subtract
   *   7a. Advect velocity
   *   7b. Advect density (dye)
   */
  step(): void {
    if (this._destroyed) return;

    const gl     = this._gl;
    const ctx    = gl as WebGLRenderingContext;
    const aspect = this._canvas
      ? this._canvas.clientWidth / this._canvas.clientHeight
      : 1.0;

    const texelSim = 1.0 / this.simRes;
    const texelDye = 1.0 / this.dyeRes;

    // ── Pass 0: SPLAT ────────────────────────────────────────────────────
    // Flush all pending pointer splats — velocity then density for each.
    for (const s of this._splatQueue) {
      // velocity splat (colour = (dx, dy, 1), uAdd = 1 for additive)
      executePasses(gl, [
        this._buildSplatPass(
          this._velocity,
          s.x, s.y, s.px, s.py,
          s.dx, s.dy, 1.0,
          aspect,
          /*uAdd*/ 1.0,
        ),
      ], this._quad);
      _swap(this._velocity);

      // density / dye splat (colour = (r, g, b), uAdd = 0 for screen blend)
      executePasses(gl, [
        this._buildSplatPass(
          this._density,
          s.x, s.y, s.px, s.py,
          s.r, s.g, s.b,
          aspect,
          /*uAdd*/ this._splatBlendMode,
        ),
      ], this._quad);
      _swap(this._density);
    }
    this._splatQueue.length = 0;

    // ── Pass 1: CURL ─────────────────────────────────────────────────────
    executePasses(gl, [{
      name:    'curl',
      program: this._pCurl,
      fbo:     this._curlFbo,
      textures: [[0, _colorTex(this._velocity.read)]],
      setUniforms: (p) => {
        p['uVelocity']?.(0);
        p['texelSize']?.([texelSim, texelSim]);
      },
    }], this._quad);

    // ── Pass 2: VORTICITY CONFINEMENT ────────────────────────────────────
    executePasses(gl, [{
      name:    'vorticity',
      program: this._pVorticity,
      fbo:     this._velocity.write,
      textures: [
        [0, _colorTex(this._velocity.read)],
        [1, _colorTex(this._curlFbo)],
      ],
      setUniforms: (p) => {
        p['uVelocity']?.( 0);
        p['uCurl']?.(     1);
        p['texelSize']?.([texelSim, texelSim]);
        p['curl']?.(      this._curlStrength);
        p['dt']?.(        this._dt);
      },
    }], this._quad);
    _swap(this._velocity);

    // ── Pass 3: DIVERGENCE ───────────────────────────────────────────────
    executePasses(gl, [{
      name:    'divergence',
      program: this._pDivergence,
      fbo:     this._divFbo,
      textures: [[0, _colorTex(this._velocity.read)]],
      setUniforms: (p) => {
        p['uVelocity']?.( 0);
        p['texelSize']?.([texelSim, texelSim]);
      },
    }], this._quad);

    // ── Pass 4: PRESSURE CLEAR ───────────────────────────────────────────
    executePasses(gl, [{
      name:    'pressure_clear',
      program: this._pClear,
      fbo:     this._pressure.write,
      textures: [[0, _colorTex(this._pressure.read)]],
      setUniforms: (p) => {
        p['uTexture']?.( 0);
        p['value']?.(    this._pressureDissipation);
      },
    }], this._quad);
    _swap(this._pressure);

    // ── Pass 5: PRESSURE SOLVE (Jacobi × N) ─────────────────────────────
    for (let i = 0; i < this._pressureIters; i++) {
      executePasses(gl, [{
        name:    `pressure_${i}`,
        program: this._pPressure,
        fbo:     this._pressure.write,
        textures: [
          [0, _colorTex(this._pressure.read)],
          [1, _colorTex(this._divFbo)],
        ],
        setUniforms: (p) => {
          p['uPressure']?.(   0);
          p['uDivergence']?.( 1);
          p['texelSize']?.([texelSim, texelSim]);
        },
      }], this._quad);
      _swap(this._pressure);
    }

    // ── Pass 6: GRADIENT SUBTRACT ────────────────────────────────────────
    executePasses(gl, [{
      name:    'gradient_subtract',
      program: this._pGradSubtract,
      fbo:     this._velocity.write,
      textures: [
        [0, _colorTex(this._pressure.read)],
        [1, _colorTex(this._velocity.read)],
      ],
      setUniforms: (p) => {
        p['uPressure']?.(  0);
        p['uVelocity']?.(  1);
        p['texelSize']?.([texelSim, texelSim]);
      },
    }], this._quad);
    _swap(this._velocity);

    // ── Pass 7a: ADVECT VELOCITY (self-advection) ────────────────────────
    executePasses(gl, [
      this._buildAdvectPass(
        this._velocity,
        _colorTex(this._velocity.read), // source = velocity itself
        texelSim,
        texelSim,
        this._velocityDissipation,
        this.simRes,
      ),
    ], this._quad);
    _swap(this._velocity);

    // ── Pass 7b: ADVECT DENSITY (dye) ────────────────────────────────────
    executePasses(gl, [
      this._buildAdvectPass(
        this._density,
        _colorTex(this._density.read), // source = dye texture
        texelSim,
        texelDye,
        this._densityDissipation,
        this.dyeRes,
      ),
    ], this._quad);
    _swap(this._density);

    // Restore default framebuffer & viewport
    ctx.bindFramebuffer(ctx.FRAMEBUFFER, null);
  }

  // ── Output accessors ─────────────────────────────────────────────────────

  /**
   * Current dye / density colour texture.
   * Bind as `tFluid` in downstream AT shaders.
   */
  getDyeTex(): Texture2D {
    return _colorTex(this._density.read);
  }

  /**
   * Current velocity texture (XY = velocity field, ZW unused).
   * Bind as `tFluidMask` or for direct velocity reads.
   */
  getVelocityTex(): Texture2D {
    return _colorTex(this._velocity.read);
  }

  // ── Runtime update ───────────────────────────────────────────────────────

  /** Update simulation parameters at runtime; unspecified fields are unchanged. */
  updateParams(p: Partial<FluidExecutorConfig>): void {
    if (p.pressureIters        !== undefined) this._pressureIters        = p.pressureIters;
    if (p.curlStrength         !== undefined) this._curlStrength         = p.curlStrength;
    if (p.splatRadius          !== undefined) this._splatRadius          = p.splatRadius;
    if (p.velocityDissipation  !== undefined) this._velocityDissipation  = p.velocityDissipation;
    if (p.densityDissipation   !== undefined) this._densityDissipation   = p.densityDissipation;
    if (p.pressureDissipation  !== undefined) this._pressureDissipation  = p.pressureDissipation;
    if (p.splatBlendMode       !== undefined) this._splatBlendMode       = p.splatBlendMode;
    if (p.dt                   !== undefined) this._dt                   = p.dt;
  }

  // ── Destroy ──────────────────────────────────────────────────────────────

  /**
   * Release all WebGL resources and detach pointer listeners.
   * The executor must not be used after this call.
   */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;

    this.detachPointerListeners();
    this._quad.dispose();

    // Dispose ping-pong FBOs
    for (const pp of [this._velocity, this._density, this._pressure]) {
      _colorTex(pp.read).dispose();
      _colorTex(pp.write).dispose();
      const ctx = this._gl as WebGLRenderingContext;
      ctx.deleteFramebuffer(pp.read.fbo);
      ctx.deleteFramebuffer(pp.write.fbo);
    }

    // Dispose single FBOs
    for (const fbo of [this._curlFbo, this._divFbo]) {
      _colorTex(fbo).dispose();
      const ctx = this._gl as WebGLRenderingContext;
      ctx.deleteFramebuffer(fbo.fbo);
    }

    // Delete programs
    for (const prog of [
      this._pSplat, this._pCurl, this._pVorticity, this._pDivergence,
      this._pPressure, this._pGradSubtract, this._pClear, this._pAdvect,
    ]) {
      prog.dispose();
    }
  }

  // ── Private pass builders ─────────────────────────────────────────────────

  /**
   * Build a ShaderPass descriptor for the AT splatShader.
   * Used for both velocity splats (colour = velocity delta) and dye splats (colour = RGB).
   */
  private _buildSplatPass(
    target: PingPongFbo,
    x: number, y: number, px: number, py: number,
    colR: number, colG: number, colB: number,
    aspect: number,
    uAdd: number,
  ): ShaderPass {
    return {
      name:    'splat',
      program: this._pSplat,
      fbo:     target.write,
      textures: [[0, _colorTex(target.read)]],
      setUniforms: (p) => {
        p['uTarget']?.(    0);
        p['aspectRatio']?.(aspect);
        p['color']?.(      [colR, colG, colB]);
        p['point']?.(      [x, y]);
        p['prevPoint']?.(  [px, py]);
        p['radius']?.(     this._splatRadius);
        p['canRender']?.(  1.0);
        p['uAdd']?.(       uAdd);
      },
    };
  }

  /**
   * Build a ShaderPass descriptor for the advection pass.
   * Works for both self-advection (velocity) and dye advection.
   */
  private _buildAdvectPass(
    target:      PingPongFbo,
    sourceTex:   Texture2D,
    texelSize:   number,
    dyeTexelSize: number,
    dissipation: number,
    _res:        number,
  ): ShaderPass {
    const velTex = _colorTex(this._velocity.read);
    return {
      name:    'advect',
      program: this._pAdvect,
      fbo:     target.write,
      textures: [
        [0, velTex],
        [1, sourceTex],
      ],
      setUniforms: (p) => {
        p['uVelocity']?.(   0);
        p['uSource']?.(     1);
        p['texelSize']?.([texelSize, texelSize]);
        // AT advectionManualFilteringShader.fs uses dyeTexelSize for source bilerp
        p['dyeTexelSize']?.([dyeTexelSize, dyeTexelSize]);
        p['dt']?.(          this._dt);
        p['dissipation']?.( dissipation);
      },
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour helpers (golden-ratio hue sequence — mirrors at-mousefluid-import.ts)
// ─────────────────────────────────────────────────────────────────────────────

const GOLDEN_ANGLE = 0.381966011250105; // 1 / φ²

function _hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  const sec = Math.floor(h * 6) % 6;
  if (sec === 0)      { r = c; g = x; b = 0; }
  else if (sec === 1) { r = x; g = c; b = 0; }
  else if (sec === 2) { r = 0; g = c; b = x; }
  else if (sec === 3) { r = 0; g = x; b = c; }
  else if (sec === 4) { r = x; g = 0; b = c; }
  else                { r = c; g = 0; b = x; }
  return [r + m, g + m, b + m];
}

function _nextColour(index: number): [number, number, number] {
  return _hslToRgb((index * GOLDEN_ANGLE) % 1.0, 0.9, 0.55);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience re-exports so downstream code gets everything from one import
// ─────────────────────────────────────────────────────────────────────────────

export {
  FullscreenQuad,
  _createHalfFloatTex as createHalfFloatTex,
  _createFboWithTex   as createFboWithTex,
  _createPingPong     as createPingPong,
  _swap               as swapPingPong,
  _colorTex           as colorTexture,
  // AT shader GLSL sources (for callers that need them separately)
  AT_FLUID_BASE_VS,
  AT_SPLAT_FS,
  AT_ADVECTION_FS,
  AT_ADVECTION_MANUAL_FS,
  AT_CURL_FS,
  AT_VORTICITY_FS,
  AT_DIVERGENCE_FS,
  AT_PRESSURE_FS,
  AT_GRADIENT_SUBTRACT_FS,
  AT_CLEAR_FS,
  AT_MOUSEFLUID_CONSUME_FS,
};

export default NanoGLFluidExecutor;
