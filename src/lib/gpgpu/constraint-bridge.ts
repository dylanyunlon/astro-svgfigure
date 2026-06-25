/**
 * constraint-bridge.ts — GPGPU constraint texture ↔ PixiJS renderer bridge
 *
 * 架构职责：
 *   1. 每帧执行 constraint-shader.frag 的 GPU compute pass（fullscreen quad）
 *   2. 把结果纹理（readTexture）直接传给 cell shader uniform — 不落地 CPU
 *   3. 仅在调用 syncToCPU() 时做一次 readPixels 回读（epoch 结束 / 调试用）
 *   4. 检测到 convergence 后可触发回调 onConverged
 *
 * 正常渲染路径（全 GPU）：
 *   bridge.tick(dt)
 *   cellShader.uniforms.uConstraintTex = bridge.gpuTexture   // 直接绑 texture id
 *
 * CPU 回读路径（仅按需）：
 *   const states = await bridge.syncToCPU()
 *   // → CellConstraintState[] 可写回 channels/physics/
 *
 * 参考架构：
 *   src/lib/renderers/fluid-fbo.ts   (fullscreen quad render pass)
 *   src/lib/renderers/antimatter-compute.ts  (readback 模式)
 *   src/lib/gpgpu/constraint-texture.ts      (ping-pong FBO)
 */

import { ConstraintTexture, createConstraintTexture } from './constraint-texture';
import type { CellConstraintState } from './constraint-texture';

// ── Inline vertex shader (reused across passes) ───────────────────────────────

const VERT_SRC = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  // Fullscreen triangle from gl_VertexID — no VBO needed
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// ── Default uniform values ─────────────────────────────────────────────────────

const DEFAULTS = {
  cellRadius:  40.0,
  springK:     0.15,
  damping:     0.85,
};

// ── ConstraintBridgeOptions ───────────────────────────────────────────────────

export interface ConstraintBridgeOptions {
  gl: WebGL2RenderingContext;
  /** Pre-built ConstraintTexture, or omit and call .init() to fetch from JSON */
  constraintTexture?: ConstraintTexture;
  /** Fragment shader GLSL source — defaults to the bundled constraint-shader.frag */
  fragSrc?: string;
  /** Physical cell radius in world pixels (default 40) */
  cellRadius?: number;
  /** Spring stiffness (default 0.15) */
  springK?: number;
  /** Velocity damping per frame (default 0.85) */
  damping?: number;
  /** Called when ALL cells converge; receives final states for CPU use */
  onConverged?: (states: CellConstraintState[]) => void;
}

// ── ConstraintBridge ──────────────────────────────────────────────────────────

export class ConstraintBridge {
  private gl:      WebGL2RenderingContext;
  private texture: ConstraintTexture | null = null;
  private program: WebGLProgram    | null = null;
  private vao:     WebGLVertexArrayObject | null = null;

  // Uniform locations (populated after program link)
  private uState!:       WebGLUniformLocation;
  private uTexSize!:     WebGLUniformLocation;
  private uCellRadius!:  WebGLUniformLocation;
  private uSpringK!:     WebGLUniformLocation;
  private uDamping!:     WebGLUniformLocation;
  private uDt!:          WebGLUniformLocation;
  private uForceUpdate!: WebGLUniformLocation;

  // Config
  private _cellRadius: number;
  private _springK:    number;
  private _damping:    number;
  private _fragSrc:    string | null;
  private _onConverged?: (states: CellConstraintState[]) => void;

  // State
  private _converged   = false;
  private _frameCount  = 0;
  private _forceUpdate = false;

  constructor(opts: ConstraintBridgeOptions) {
    this.gl           = opts.gl;
    this.texture      = opts.constraintTexture ?? null;
    this._fragSrc     = opts.fragSrc ?? null;
    this._cellRadius  = opts.cellRadius ?? DEFAULTS.cellRadius;
    this._springK     = opts.springK   ?? DEFAULTS.springK;
    this._damping     = opts.damping   ?? DEFAULTS.damping;
    this._onConverged = opts.onConverged;
  }

  // ── Async init (when texture not pre-built) ──────────────────────────────

  /**
   * Fetch channels/physics JSON files, build the ConstraintTexture, and
   * compile the GLSL program. Safe to call multiple times (no-op if done).
   */
  async init(opts: {
    forceFieldUrl?:   string;
    cellRegistryUrl?: string;
    fragShaderUrl?:   string;
  } = {}): Promise<void> {
    const { gl } = this;

    if (!this.texture) {
      this.texture = await createConstraintTexture(gl, {
        forceFieldUrl:   opts.forceFieldUrl,
        cellRegistryUrl: opts.cellRegistryUrl,
      });
    }

    if (!this._fragSrc) {
      if (opts.fragShaderUrl) {
        const res = await fetch(opts.fragShaderUrl);
        if (!res.ok) throw new Error(`[ConstraintBridge] fetch shader failed: ${opts.fragShaderUrl}`);
        this._fragSrc = await res.text();
      } else {
        // Inline stub: real projects import raw via bundler.
        // Replace with `import fragSrc from './constraint-shader.frag?raw'` in Vite.
        throw new Error(
          '[ConstraintBridge] fragSrc not provided. ' +
          'Either pass fragSrc option or opts.fragShaderUrl, ' +
          'or import the .frag file directly via your bundler.'
        );
      }
    }

    this._compileProgram(this._fragSrc);
  }

  /**
   * Synchronous init when you already have the texture AND fragSrc compiled in.
   * Prefer this at call sites that import the .frag file as a raw string.
   */
  initSync(texture: ConstraintTexture, fragSrc: string): void {
    this.texture  = texture;
    this._fragSrc = fragSrc;
    this._compileProgram(fragSrc);
  }

  // ── Per-frame update ─────────────────────────────────────────────────────

  /**
   * Run one constraint compute pass on the GPU.
   * @param dt  Delta time in frames (typically 1.0 at 60 fps, or elapsed ms / 16.67)
   */
  tick(dt = 1.0): void {
    if (!this.texture || !this.program) return;
    if (this._converged && !this._forceUpdate) return;

    const { gl, texture, program } = this;

    // ── Bind write FBO ───────────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, texture.writeFBO);
    gl.viewport(0, 0, texture.texSize, texture.texSize);

    gl.useProgram(program);

    // ── Upload uniforms ──────────────────────────────────────────────────────
    gl.uniform1i(this.uState,       0);
    gl.uniform1f(this.uTexSize,     texture.texSize);
    gl.uniform1f(this.uCellRadius,  this._cellRadius);
    gl.uniform1f(this.uSpringK,     this._springK);
    gl.uniform1f(this.uDamping,     this._damping);
    gl.uniform1f(this.uDt,          dt);
    gl.uniform1f(this.uForceUpdate, this._forceUpdate ? 1.0 : 0.0);

    // ── Bind source texture ──────────────────────────────────────────────────
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture.readTexture);

    // ── Draw fullscreen triangle (no VBO — gl_VertexID trick) ───────────────
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.useProgram(null);

    // Swap ping-pong: next readTexture is what we just wrote
    texture.swap();

    this._forceUpdate = false;
    this._frameCount++;
  }

  // ── GPU texture handle for PixiJS cell shader ────────────────────────────

  /**
   * The WebGLTexture holding the latest constraint state.
   * Assign directly to your cell shader's sampler uniform:
   *
   *   gl.activeTexture(gl.TEXTURE1);
   *   gl.bindTexture(gl.TEXTURE_2D, bridge.gpuTexture);
   *   gl.uniform1i(cellShader.uConstraintTex, 1);
   */
  get gpuTexture(): WebGLTexture | null {
    return this.texture?.readTexture ?? null;
  }

  /** Texture size (width = height = texSize) */
  get texSize(): number {
    return this.texture?.texSize ?? 0;
  }

  /** Ordered cell IDs — use to map cellId → pixel index */
  get cellOrder(): string[] {
    return this.texture?.cellOrder ?? [];
  }

  // ── CPU readback (on demand) ──────────────────────────────────────────────

  /**
   * Read all cell states back to CPU.  Triggers gl.readPixels (slow — ~1ms).
   * Call only at epoch boundaries, not every frame.
   * Result can be serialised back to channels/physics/force_field.json.
   */
  syncToCPU(): CellConstraintState[] {
    if (!this.texture) return [];
    const states = this.texture.readAllCellStates();

    // Check global convergence
    const allConverged = states.every(s => s.converged);
    if (allConverged && !this._converged) {
      this._converged = true;
      this._onConverged?.(states);
    }

    return states;
  }

  /**
   * Read a single cell state back to CPU (cheaper for spot checks).
   */
  syncCellToCPU(cellId: string): CellConstraintState | null {
    if (!this.texture) return null;
    const i = this.texture.cellOrder.indexOf(cellId);
    if (i < 0) return null;
    return this.texture.readCellState(i);
  }

  // ── Control ───────────────────────────────────────────────────────────────

  /** Force-resume computation even if converged flag is set */
  forceUpdate(): void {
    this._converged   = false;
    this._forceUpdate = true;
  }

  /** Upload fresh states (e.g. after a physics epoch reset) */
  resetFromStates(states: CellConstraintState[]): void {
    this.texture?.uploadStates(states);
    this._converged   = false;
    this._forceUpdate = true;
    this._frameCount  = 0;
  }

  /** Mutate physics parameters at runtime */
  setParams(p: { cellRadius?: number; springK?: number; damping?: number }): void {
    if (p.cellRadius !== undefined) this._cellRadius = p.cellRadius;
    if (p.springK    !== undefined) this._springK    = p.springK;
    if (p.damping    !== undefined) this._damping    = p.damping;
  }

  get frameCount(): number { return this._frameCount; }
  get isConverged(): boolean { return this._converged; }

  destroy(): void {
    const { gl } = this;
    this.texture?.destroy();
    if (this.program) gl.deleteProgram(this.program);
    if (this.vao)     gl.deleteVertexArray(this.vao);
    this.texture = null;
    this.program = null;
    this.vao     = null;
  }

  // ── Private: GLSL compilation ─────────────────────────────────────────────

  private _compileProgram(fragSrc: string): void {
    const { gl } = this;

    const vert = this._compileShader(gl.VERTEX_SHADER,   VERT_SRC);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`[ConstraintBridge] Program link failed:\n${info}`);
    }

    gl.deleteShader(vert);
    gl.deleteShader(frag);

    this.program = prog;

    // Cache uniform locations
    this.uState       = gl.getUniformLocation(prog, 'uState')!;
    this.uTexSize     = gl.getUniformLocation(prog, 'uTexSize')!;
    this.uCellRadius  = gl.getUniformLocation(prog, 'uCellRadius')!;
    this.uSpringK     = gl.getUniformLocation(prog, 'uSpringK')!;
    this.uDamping     = gl.getUniformLocation(prog, 'uDamping')!;
    this.uDt          = gl.getUniformLocation(prog, 'uDt')!;
    this.uForceUpdate = gl.getUniformLocation(prog, 'uForceUpdate')!;

    // Empty VAO for gl_VertexID fullscreen triangle
    this.vao = gl.createVertexArray()!;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const { gl } = this;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      const typeName = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
      throw new Error(`[ConstraintBridge] ${typeName} shader compile failed:\n${info}`);
    }
    return shader;
  }
}

// ── Usage example (tree-shaken in production) ─────────────────────────────────
//
// import fragSrc from './constraint-shader.frag?raw';          // Vite raw import
// import { ConstraintBridge } from '$lib/gpgpu/constraint-bridge';
//
// const bridge = new ConstraintBridge({ gl, cellRadius: 40, springK: 0.15 });
// await bridge.init({ fragShaderUrl: '/src/lib/gpgpu/constraint-shader.frag' });
//
// // --- or synchronously if you import raw GLSL ---
// // bridge.initSync(constraintTex, fragSrc);
//
// // Render loop:
// ticker.add((dt) => {
//   bridge.tick(dt);
//
//   // GPU path — pass texture directly to cell shader:
//   gl.activeTexture(gl.TEXTURE1);
//   gl.bindTexture(gl.TEXTURE_2D, bridge.gpuTexture);
//   gl.uniform1i(cellShader.uConstraintTex, 1);
//   gl.uniform1f(cellShader.uTexSize, bridge.texSize);
//
//   // CPU path — only at epoch end:
//   // const states = bridge.syncToCPU();
// });
