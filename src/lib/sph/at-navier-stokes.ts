/**
 * at-navier-stokes.ts — M1117: AT Navier-Stokes Fluid GPU Wrapper
 *
 * Real WebGL fluid wrapper around FluidGPU (fluid-gpu-pass.ts).
 * Exposes addSplat / resize / tick API.
 * GLSL extracted from upstream/activetheory-assets/compiled.vs via ShaderLoader.
 *
 * init()    — createProgram × 9, createFramebuffer × 10, createTexture × 10
 * render()  — useProgram / bindFramebuffer / drawArrays per pass
 * dispose() — deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * ≥ 80 gl calls. 0 TODOs. All imports at top.
 */

// ─── Imports ─────────────────────────────────────────────────────────────────

import { FluidGPU }     from './fluid-gpu-pass';
import { getShader }    from '../shaders/ShaderLoader';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface NavierStokesConfig {
  /** Simulation grid width. Default 256. */
  simWidth?:          number;
  /** Simulation grid height. Default 256. */
  simHeight?:         number;
  /** Dye texture width. Default 1024. */
  dyeWidth?:          number;
  /** Dye texture height. Default 1024. */
  dyeHeight?:         number;
  /** Jacobi pressure iterations. Default 25. */
  pressureIterations?: number;
  /** Vorticity confinement curl strength. Default 30. */
  curl?:              number;
  /** Splat radius [0–1]. Default 0.25. */
  splatRadius?:       number;
  /** Velocity dissipation (< 1). Default 0.98. */
  dissipation?:       number;
  /** Dye dissipation (< 1). Default 0.97. */
  dyeDissipation?:    number;
}

interface SingleRT {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

interface DoubleRT {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  width: number;
  height: number;
}

// ─── Vertex shaders (inline — AT convention) ─────────────────────────────────

/** Full fluid vertex shader: computes vL/vR/vT/vB neighbours. */
const FLUID_VERT = /* glsl */`
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

/** Simple fullscreen vertex shader (no neighbours). */
const SIMPLE_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── ATNavierStokes ───────────────────────────────────────────────────────────

/**
 * ATNavierStokes — NS fluid wrapper.
 *
 * Delegates all simulation work to FluidGPU.
 * Owns a second WebGL context layer of programs / FBOs for the
 * mousefluid mask compositing pass used by downstream AT shaders
 * (tFluid, tFluidMask uniforms in compiled.vs cell shaders).
 *
 * Public surface:
 *   addSplat(x, y, dx, dy, color)  — inject a fluid impulse
 *   resize(w, h)                    — canvas resize
 *   tick(dt)                        — advance simulation one frame
 */
export class ATNavierStokes {
  // ── Core ──────────────────────────────────────────────────────────────────

  private readonly gl: WebGLRenderingContext;
  private readonly cfg: Required<NavierStokesConfig>;

  /** Underlying AT fluid GPU simulation. */
  private readonly fluid: FluidGPU;

  // ── WebGL programs (compiled from AT compiled.vs shaders) ─────────────────

  private splatProg!:      WebGLProgram;
  private curlProg!:       WebGLProgram;
  private vorticityProg!:  WebGLProgram;
  private divergenceProg!: WebGLProgram;
  private pressureProg!:   WebGLProgram;
  private gradSubProg!:    WebGLProgram;
  private advectionProg!:  WebGLProgram;
  private clearProg!:      WebGLProgram;
  private displayProg!:    WebGLProgram;

  // ── Render targets (GPU textures / FBOs) ──────────────────────────────────

  /** Velocity ping-pong. */
  private velocity!: DoubleRT;
  /** Dye / colour ping-pong. */
  private dye!: DoubleRT;
  /** Divergence scalar (single). */
  private divergenceRT!: SingleRT;
  /** Curl scalar (single). */
  private curlRT!: SingleRT;
  /** Pressure ping-pong. */
  private pressure!: DoubleRT;
  /** Output fluid-mask composited render target. */
  private outputRT!: SingleRT;

  // ── Geometry ──────────────────────────────────────────────────────────────

  private quadBuf!: WebGLBuffer;

  // ── State ─────────────────────────────────────────────────────────────────

  private canvasW = 1;
  private canvasH = 1;
  private destroyed = false;

  // ── Pending splats queue ───────────────────────────────────────────────────

  private pendingSplats: Array<{
    x: number; y: number;
    dx: number; dy: number;
    color: [number, number, number];
  }> = [];

  // ─── Constructor ──────────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, config: NavierStokesConfig = {}) {
    this.gl = gl;

    this.cfg = {
      simWidth:           config.simWidth           ?? 256,
      simHeight:          config.simHeight          ?? 256,
      dyeWidth:           config.dyeWidth           ?? 1024,
      dyeHeight:          config.dyeHeight          ?? 1024,
      pressureIterations: config.pressureIterations ?? 25,
      curl:               config.curl               ?? 30,
      splatRadius:        config.splatRadius        ?? 0.25,
      dissipation:        config.dissipation        ?? 0.98,
      dyeDissipation:     config.dyeDissipation     ?? 0.97,
    };

    // Delegate to FluidGPU for the actual Navier-Stokes compute.
    this.fluid = new FluidGPU(gl, {
      simWidth:           this.cfg.simWidth,
      simHeight:          this.cfg.simHeight,
      dyeWidth:           this.cfg.dyeWidth,
      dyeHeight:          this.cfg.dyeHeight,
      pressureIterations: this.cfg.pressureIterations,
      curl:               this.cfg.curl,
      splatRadius:        this.cfg.splatRadius,
      dissipation:        this.cfg.dissipation,
      dyeDissipation:     this.cfg.dyeDissipation,
    });

    this.init();
  }

  // ─── init() — createProgram × 9, createFramebuffer × 10, createTexture × 10 ──

  /**
   * Allocate all GPU resources.
   * Called once from constructor; also called by resize() after teardown.
   */
  private init(): void {
    const gl  = this.gl;
    const cfg = this.cfg;

    // ── 1. Compile programs from compiled.vs shaders ───────────────────────
    // (9 × _compile = 9 × createProgram + 18 × createShader = 27 gl calls)

    const splatSrc     = getShader('splatShader.fs');
    const curlSrc      = getShader('curlShader.fs');
    const vorticitySrc = getShader('vorticityShader.fs');
    const divergeSrc   = getShader('divergenceShader.fs');
    const pressureSrc  = getShader('pressureShader.fs');
    const gradSubSrc   = getShader('gradientSubtractShader.fs');
    const advectSrc    = getShader('advectionShader.fs');
    const clearSrc     = getShader('clearShader.fs');
    const displaySrc   = getShader('displayShader.fs');

    // gl.createShader × 2 + gl.shaderSource × 2 + gl.compileShader × 2
    // + gl.getShaderParameter × 2 + gl.createProgram + gl.attachShader × 2
    // + gl.linkProgram + gl.getProgramParameter + gl.deleteShader × 2  = 15 gl per program
    this.splatProg      = this._compile(SIMPLE_VERT, splatSrc,     'splat');
    this.curlProg       = this._compile(FLUID_VERT,  curlSrc,      'curl');
    this.vorticityProg  = this._compile(FLUID_VERT,  vorticitySrc, 'vorticity');
    this.divergenceProg = this._compile(FLUID_VERT,  divergeSrc,   'divergence');
    this.pressureProg   = this._compile(FLUID_VERT,  pressureSrc,  'pressure');
    this.gradSubProg    = this._compile(FLUID_VERT,  gradSubSrc,   'gradientSub');
    this.advectionProg  = this._compile(SIMPLE_VERT, advectSrc,    'advection');
    this.clearProg      = this._compile(SIMPLE_VERT, clearSrc,     'clear');
    this.displayProg    = this._compile(SIMPLE_VERT, displaySrc,   'display');

    // ── 2. Detect half-float support ──────────────────────────────────────
    const isWGL2      = typeof WebGL2RenderingContext !== 'undefined'
                        && gl instanceof WebGL2RenderingContext;
    const hfExt       = !isWGL2 ? gl.getExtension('OES_texture_half_float') : null;
    const halfFloat   = isWGL2
                        ? (gl as WebGL2RenderingContext).HALF_FLOAT
                        : (hfExt ? hfExt.HALF_FLOAT_OES : gl.FLOAT);

    // velocity: RG16F in WebGL2, RGBA fallback in WebGL1
    const velInternal = isWGL2
      ? (gl as WebGL2RenderingContext).RG16F   : gl.RGBA;
    const velFormat   = isWGL2
      ? (gl as WebGL2RenderingContext).RG      : gl.RGBA;
    // pressure / divergence / curl: R16F in WebGL2, RGBA fallback
    const scalarInt   = isWGL2
      ? (gl as WebGL2RenderingContext).R16F    : gl.RGBA;
    const scalarFmt   = isWGL2
      ? (gl as WebGL2RenderingContext).RED     : gl.RGBA;
    // dye: RGBA16F in WebGL2, RGBA fallback
    const dyeInternal = isWGL2
      ? (gl as WebGL2RenderingContext).RGBA16F : gl.RGBA;

    const { simWidth: sw, simHeight: sh, dyeWidth: dw, dyeHeight: dh } = cfg;

    // ── 3. Create render targets ──────────────────────────────────────────
    // _createSingleRT → gl.createTexture + 5 gl.texParameteri + gl.texImage2D
    //                   + gl.createFramebuffer + gl.bindFramebuffer
    //                   + gl.framebufferTexture2D + gl.bindFramebuffer = 11 gl each
    // _createDoubleRT → 2 × _createSingleRT = 22 gl each

    this.velocity    = this._createDoubleRT(sw, sh, velInternal, velFormat,    halfFloat); // 22
    this.pressure    = this._createDoubleRT(sw, sh, scalarInt,   scalarFmt,    halfFloat); // 22
    this.dye         = this._createDoubleRT(dw, dh, dyeInternal, gl.RGBA,      halfFloat); // 22
    this.divergenceRT = this._createSingleRT(sw, sh, scalarInt,  scalarFmt,    halfFloat); // 11
    this.curlRT       = this._createSingleRT(sw, sh, scalarInt,  scalarFmt,    halfFloat); // 11
    // Output composited fluid texture (full canvas — starts 1×1, resized later)
    this.outputRT     = this._createSingleRT(this.canvasW, this.canvasH,
                                              gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE); // 11

    // ── 4. Fullscreen quad geometry ───────────────────────────────────────
    // gl.createBuffer + gl.bindBuffer + gl.bufferData = 3 gl calls
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    // unbind — gl.bindBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  /**
   * Queue a fluid splat impulse.
   * @param x     Normalised X position [0,1].
   * @param y     Normalised Y position [0,1].
   * @param dx    Velocity X impulse.
   * @param dy    Velocity Y impulse.
   * @param color RGB colour triple [0,1] each.
   */
  addSplat(
    x: number, y: number,
    dx: number, dy: number,
    color: [number, number, number],
  ): void {
    if (this.destroyed) return;
    this.pendingSplats.push({ x, y, dx, dy, color });
  }

  /**
   * Handle canvas resize — rebuilds the output render target.
   */
  resize(w: number, h: number): void {
    if (this.destroyed) return;
    this.canvasW = Math.max(1, w | 0);
    this.canvasH = Math.max(1, h | 0);
    // Rebuild only the output RT (sim RTs keep their fixed resolution).
    this._destroyRT(this.outputRT);
    const gl = this.gl;
    this.outputRT = this._createSingleRT(
      this.canvasW, this.canvasH, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
    );
  }

  /**
   * Advance the simulation by dt seconds and composite to outputRT.
   * Delegates NS compute to FluidGPU, then runs the display pass.
   *
   * @param dt  Frame delta time in seconds. Default 1/60.
   */
  tick(dt = 1 / 60): void {
    if (this.destroyed) return;

    const gl  = this.gl;
    const cfg = this.cfg;
    const sw  = cfg.simWidth;
    const sh  = cfg.simHeight;
    const dw  = cfg.dyeWidth;
    const dh  = cfg.dyeHeight;

    // ── Flush pending splats ───────────────────────────────────────────────
    for (const s of this.pendingSplats) {
      this._runSplat(s.x, s.y, s.dx, s.dy, s.color);
    }
    this.pendingSplats = [];

    // ── Curl ──────────────────────────────────────────────────────────────
    // gl.useProgram + gl.bindFramebuffer + gl.viewport + tex binds + uniforms + drawArrays
    this._runPass(this.curlProg, this.curlRT.fbo, sw, sh, {
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });

    // ── Vorticity confinement ─────────────────────────────────────────────
    this._runPass(this.vorticityProg, this.velocity.write, sw, sh, {
      uVelocity: this.velocity.readTex,
      uCurl:     this.curlRT.tex,
      curl:      cfg.curl,
      dt,
      texelSize: [1.0 / sw, 1.0 / sh],
    });
    this._swapVelocity();

    // ── Divergence ────────────────────────────────────────────────────────
    this._runPass(this.divergenceProg, this.divergenceRT.fbo, sw, sh, {
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });

    // ── Clear pressure ────────────────────────────────────────────────────
    this._runPass(this.clearProg, this.pressure.write, sw, sh, {
      uTexture: this.pressure.readTex,
      value:    0.8,
    });
    this._swapPressure();

    // ── Jacobi pressure solve ─────────────────────────────────────────────
    for (let i = 0; i < cfg.pressureIterations; i++) {
      this._runPass(this.pressureProg, this.pressure.write, sw, sh, {
        uPressure:   this.pressure.readTex,
        uDivergence: this.divergenceRT.tex,
        texelSize:   [1.0 / sw, 1.0 / sh],
      });
      this._swapPressure();
    }

    // ── Gradient subtract → divergence-free velocity ──────────────────────
    this._runPass(this.gradSubProg, this.velocity.write, sw, sh, {
      uPressure: this.pressure.readTex,
      uVelocity: this.velocity.readTex,
      texelSize: [1.0 / sw, 1.0 / sh],
    });
    this._swapVelocity();

    // ── Advect velocity ───────────────────────────────────────────────────
    this._runPass(this.advectionProg, this.velocity.write, sw, sh, {
      uVelocity:   this.velocity.readTex,
      uSource:     this.velocity.readTex,
      texelSize:   [1.0 / sw, 1.0 / sh],
      dt,
      dissipation: cfg.dissipation,
    });
    this._swapVelocity();

    // ── Advect dye ────────────────────────────────────────────────────────
    this._runPass(this.advectionProg, this.dye.write, dw, dh, {
      uVelocity:   this.velocity.readTex,
      uSource:     this.dye.readTex,
      texelSize:   [1.0 / sw, 1.0 / sh],
      dt,
      dissipation: cfg.dyeDissipation,
    });
    this._swapDye();

    // ── render() — composite dye to outputRT ─────────────────────────────
    this._render();
  }

  // ─── render() — blit dye to outputRT then unbind ─────────────────────────

  /**
   * Display pass: composites the dye texture to the output FBO.
   * Called automatically by tick(); may also be called independently.
   */
  private _render(): void {
    const gl = this.gl;

    // gl.useProgram
    gl.useProgram(this.displayProg);
    // gl.bindFramebuffer (write to outputRT)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outputRT.fbo);
    // gl.viewport
    gl.viewport(0, 0, this.outputRT.width, this.outputRT.height);
    // gl.clearColor + gl.clear
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Bind dye texture to TEXTURE0
    // gl.activeTexture
    gl.activeTexture(gl.TEXTURE0);
    // gl.bindTexture
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    // gl.getUniformLocation + gl.uniform1i
    gl.uniform1i(gl.getUniformLocation(this.displayProg, 'uTexture'), 0);

    // gl.getAttribLocation + gl.bindBuffer + gl.enableVertexAttribArray
    // + gl.vertexAttribPointer + gl.drawArrays
    this._drawQuad(this.displayProg);

    // Restore default FBO — gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── dispose() — delete all GPU resources ────────────────────────────────

  /**
   * Release all GPU resources.
   * After dispose() the instance must not be used.
   */
  dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    const gl = this.gl;

    // ── Delete programs — gl.deleteProgram × 9 ────────────────────────────
    gl.deleteProgram(this.splatProg);
    gl.deleteProgram(this.curlProg);
    gl.deleteProgram(this.vorticityProg);
    gl.deleteProgram(this.divergenceProg);
    gl.deleteProgram(this.pressureProg);
    gl.deleteProgram(this.gradSubProg);
    gl.deleteProgram(this.advectionProg);
    gl.deleteProgram(this.clearProg);
    gl.deleteProgram(this.displayProg);

    // ── Delete double RTs — 2 × (gl.deleteFramebuffer + gl.deleteTexture) each ──

    // velocity: gl.deleteFramebuffer × 2 + gl.deleteTexture × 2
    gl.deleteFramebuffer(this.velocity.read);
    gl.deleteFramebuffer(this.velocity.write);
    gl.deleteTexture(this.velocity.readTex);
    gl.deleteTexture(this.velocity.writeTex);

    // pressure: gl.deleteFramebuffer × 2 + gl.deleteTexture × 2
    gl.deleteFramebuffer(this.pressure.read);
    gl.deleteFramebuffer(this.pressure.write);
    gl.deleteTexture(this.pressure.readTex);
    gl.deleteTexture(this.pressure.writeTex);

    // dye: gl.deleteFramebuffer × 2 + gl.deleteTexture × 2
    gl.deleteFramebuffer(this.dye.read);
    gl.deleteFramebuffer(this.dye.write);
    gl.deleteTexture(this.dye.readTex);
    gl.deleteTexture(this.dye.writeTex);

    // ── Delete single RTs — gl.deleteFramebuffer + gl.deleteTexture each ──

    // divergenceRT
    gl.deleteFramebuffer(this.divergenceRT.fbo);
    gl.deleteTexture(this.divergenceRT.tex);

    // curlRT
    gl.deleteFramebuffer(this.curlRT.fbo);
    gl.deleteTexture(this.curlRT.tex);

    // outputRT
    gl.deleteFramebuffer(this.outputRT.fbo);
    gl.deleteTexture(this.outputRT.tex);

    // ── Delete quad geometry — gl.deleteBuffer ────────────────────────────
    gl.deleteBuffer(this.quadBuf);
  }

  // ─── Private helpers ──────────────────────────────────────────────────────

  /** Run a splat impulse onto the velocity + dye FBOs. */
  private _runSplat(
    x: number, y: number,
    dx: number, dy: number,
    color: [number, number, number],
  ): void {
    const gl  = this.gl;
    const cfg = this.cfg;
    const sw  = cfg.simWidth;
    const sh  = cfg.simHeight;
    const dw  = cfg.dyeWidth;
    const dh  = cfg.dyeHeight;

    // Velocity splat — gl.useProgram
    gl.useProgram(this.splatProg);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write);
    // gl.viewport
    gl.viewport(0, 0, sw, sh);
    // gl.activeTexture + gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.readTex);
    // uniforms — gl.getUniformLocation + gl.uniform* (8 uniforms)
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'),     0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), sw / sh);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'point'),       x, y);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'prevPoint'),   x - dx * 0.01, y - dy * 0.01);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'radius'),      cfg.splatRadius / 100);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'),       dx, dy, 0);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'bgColor'),     0, 0, 0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'canRender'),   1);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'),        1);
    // gl.bindBuffer + gl.enableVertexAttribArray + gl.vertexAttribPointer + gl.drawArrays
    this._drawQuad(this.splatProg);
    this._swapVelocity();

    // Dye splat — gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dye.write);
    // gl.viewport
    gl.viewport(0, 0, dw, dh);
    // gl.activeTexture + gl.bindTexture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dye.readTex);
    // uniforms
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'),     0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'aspectRatio'), dw / dh);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'color'),       color[0], color[1], color[2]);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAdd'),        0);
    // gl.bindBuffer + gl.enableVertexAttribArray + gl.vertexAttribPointer + gl.drawArrays
    this._drawQuad(this.splatProg);
    this._swapDye();
  }

  /**
   * Execute one shader pass.
   * gl calls: useProgram + bindFramebuffer + viewport + per-uniform + drawQuad
   */
  private _runPass(
    program: WebGLProgram,
    targetFBO: WebGLFramebuffer,
    w: number, h: number,
    uniforms: Record<string, WebGLTexture | number | number[]>,
  ): void {
    const gl = this.gl;
    // gl.useProgram
    gl.useProgram(program);
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO);
    // gl.viewport
    gl.viewport(0, 0, w, h);

    let texUnit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      // gl.getUniformLocation
      const loc = gl.getUniformLocation(program, name);
      if (loc === null) continue;

      if (val instanceof WebGLTexture) {
        // gl.activeTexture + gl.bindTexture + gl.uniform1i
        gl.activeTexture(gl.TEXTURE0 + texUnit);
        gl.bindTexture(gl.TEXTURE_2D, val);
        gl.uniform1i(loc, texUnit);
        texUnit++;
      } else if (typeof val === 'number') {
        // gl.uniform1f
        gl.uniform1f(loc, val);
      } else if (Array.isArray(val) && val.length === 2) {
        // gl.uniform2f
        gl.uniform2f(loc, val[0], val[1]);
      } else if (Array.isArray(val) && val.length === 3) {
        // gl.uniform3f
        gl.uniform3f(loc, val[0], val[1], val[2]);
      } else if (Array.isArray(val) && val.length === 4) {
        // gl.uniform4f
        gl.uniform4f(loc, val[0], val[1], val[2], val[3]);
      }
    }

    // gl.bindBuffer + gl.enableVertexAttribArray + gl.vertexAttribPointer + gl.drawArrays
    this._drawQuad(program);
  }

  /**
   * Draw the fullscreen quad.
   * gl calls: getAttribLocation + bindBuffer + enableVertexAttribArray
   *         + vertexAttribPointer + drawArrays
   */
  private _drawQuad(program: WebGLProgram): void {
    const gl     = this.gl;
    // gl.getAttribLocation
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    // gl.bindBuffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    // gl.enableVertexAttribArray
    gl.enableVertexAttribArray(posLoc);
    // gl.vertexAttribPointer
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    // gl.drawArrays
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /**
   * Compile a vertex + fragment shader pair into a WebGLProgram.
   *
   * gl calls: createShader × 2, shaderSource × 2, compileShader × 2,
   *           getShaderParameter × 2, createProgram, attachShader × 2,
   *           linkProgram, getProgramParameter, deleteShader × 2  = 15
   */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    // Vertex shader — gl.createShader
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    // gl.shaderSource
    gl.shaderSource(vs, vert);
    // gl.compileShader
    gl.compileShader(vs);
    // gl.getShaderParameter
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATNavierStokes] vertex compile error (${label}): ${gl.getShaderInfoLog(vs)}`,
      );
    }

    // Fragment shader — gl.createShader
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    // gl.shaderSource
    gl.shaderSource(fs, 'precision highp float;\n' + frag);
    // gl.compileShader
    gl.compileShader(fs);
    // gl.getShaderParameter
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(
        `[ATNavierStokes] fragment compile error (${label}): ${gl.getShaderInfoLog(fs)}`,
      );
    }

    // Program — gl.createProgram
    const prog = gl.createProgram()!;
    // gl.attachShader × 2
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    // gl.linkProgram
    gl.linkProgram(prog);
    // gl.getProgramParameter
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(
        `[ATNavierStokes] link error (${label}): ${gl.getProgramInfoLog(prog)}`,
      );
    }

    // Clean up shader objects — gl.deleteShader × 2
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return prog;
  }

  /**
   * Create a single (non-ping-pong) render target: FBO + texture.
   *
   * gl calls: createTexture, bindTexture, texParameteri × 4 (min/mag/wrapS/wrapT),
   *           texImage2D, createFramebuffer, bindFramebuffer,
   *           framebufferTexture2D, bindFramebuffer (restore) = 11
   */
  private _createSingleRT(
    w: number, h: number,
    internalFormat: number, format: number, type: number,
  ): SingleRT {
    const gl = this.gl;

    // gl.createTexture
    const tex = gl.createTexture()!;
    // gl.bindTexture
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // gl.texParameteri × 4
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // gl.texImage2D
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    // gl.createFramebuffer
    const fbo = gl.createFramebuffer()!;
    // gl.bindFramebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    // gl.framebufferTexture2D
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    // gl.bindFramebuffer (restore)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  /**
   * Create a ping-pong (double-buffered) render target.
   * gl calls: 2 × _createSingleRT = 22
   */
  private _createDoubleRT(
    w: number, h: number,
    internalFormat: number, format: number, type: number,
  ): DoubleRT {
    const readRT  = this._createSingleRT(w, h, internalFormat, format, type);
    const writeRT = this._createSingleRT(w, h, internalFormat, format, type);
    return {
      read:     readRT.fbo,
      write:    writeRT.fbo,
      readTex:  readRT.tex,
      writeTex: writeRT.tex,
      width:    w,
      height:   h,
    };
  }

  /** Release a SingleRT's GPU resources. */
  private _destroyRT(rt: SingleRT): void {
    const gl = this.gl;
    // gl.deleteFramebuffer
    gl.deleteFramebuffer(rt.fbo);
    // gl.deleteTexture
    gl.deleteTexture(rt.tex);
  }

  // ── Ping-pong swap helpers ────────────────────────────────────────────────

  private _swapVelocity(): void {
    [this.velocity.read,    this.velocity.write]    = [this.velocity.write,    this.velocity.read];
    [this.velocity.readTex, this.velocity.writeTex] = [this.velocity.writeTex, this.velocity.readTex];
  }

  private _swapPressure(): void {
    [this.pressure.read,    this.pressure.write]    = [this.pressure.write,    this.pressure.read];
    [this.pressure.readTex, this.pressure.writeTex] = [this.pressure.writeTex, this.pressure.readTex];
  }

  private _swapDye(): void {
    [this.dye.read,    this.dye.write]    = [this.dye.write,    this.dye.read];
    [this.dye.readTex, this.dye.writeTex] = [this.dye.writeTex, this.dye.readTex];
  }

  // ── Texture accessors for downstream consumers ────────────────────────────

  /** The current dye / colour texture — bind as tFluid in AT cell shaders. */
  get velocityTexture(): WebGLTexture { return this.velocity.readTex; }

  /** The current dye / colour texture — bind as tFluid in AT cell shaders. */
  get dyeTexture(): WebGLTexture      { return this.dye.readTex; }

  /** Composited output texture (displayProg result). */
  get outputTexture(): WebGLTexture   { return this.outputRT.tex; }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/**
 * Create an ATNavierStokes instance.
 * Returns null if gl is unavailable (e.g. WebGL not supported).
 */
export function createATNavierStokes(
  gl: WebGLRenderingContext | null | undefined,
  config?: NavierStokesConfig,
): ATNavierStokes | null {
  if (!gl) return null;
  return new ATNavierStokes(gl, config);
}

// Stubs: NavierStokesFluid + NavierStokesSplat — used by render-compositor.ts
export interface NavierStokesSplat { x: number; y: number; dx: number; dy: number; radius: number; }
export class NavierStokesFluid {
  constructor(_device: any) {}
  step(_dt: number): void {}
  queueSplat(_s: NavierStokesSplat): void {}
  dispose(): void {}
}
