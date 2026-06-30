/**
 * gl-state-cache.ts — WebGL state cache (learned from Three.js WebGLState)
 *
 * Caches GL state to avoid redundant gl.enable/disable/blendFunc calls.
 * Each render pass should call resetForPass() at start to ensure clean state.
 *
 * Reference: three.js/src/renderers/webgl-fallback/utils/WebGLState.js
 */

export class GLStateCache {
  private gl: WebGL2RenderingContext;

  // Cached state
  private _blendEnabled = false;
  private _blendSrc: number = 0;
  private _blendDst: number = 0;
  private _depthTestEnabled = false;
  private _depthMask = true;
  private _cullFaceEnabled = false;
  private _scissorEnabled = false;
  private _currentProgram: WebGLProgram | null = null;
  private _currentVAO: WebGLVertexArrayObject | null = null;
  private _currentFBO: WebGLFramebuffer | null = null;
  private _viewportX = 0;
  private _viewportY = 0;
  private _viewportW = 0;
  private _viewportH = 0;
  private _activeTexture = -1;

  // Uniform location cache: program → { name → location }
  private _uniformCache: WeakMap<WebGLProgram, Map<string, WebGLUniformLocation | null>> = new WeakMap();

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  // ── Blend ───────────────────────────────────────────────────────────────

  setBlending(enabled: boolean, src?: number, dst?: number): void {
    const gl = this.gl;
    if (this._blendEnabled !== enabled) {
      this._blendEnabled = enabled;
      if (enabled) gl.enable(gl.BLEND);
      else gl.disable(gl.BLEND);
    }
    if (enabled && src !== undefined && dst !== undefined) {
      if (this._blendSrc !== src || this._blendDst !== dst) {
        this._blendSrc = src;
        this._blendDst = dst;
        gl.blendFunc(src, dst);
      }
    }
  }

  setAlphaBlend(): void {
    const gl = this.gl;
    this.setBlending(true, gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  setAdditiveBlend(): void {
    const gl = this.gl;
    this.setBlending(true, gl.ONE, gl.ONE);
  }

  // ── Depth ───────────────────────────────────────────────────────────────

  setDepthTest(enabled: boolean): void {
    const gl = this.gl;
    if (this._depthTestEnabled !== enabled) {
      this._depthTestEnabled = enabled;
      if (enabled) gl.enable(gl.DEPTH_TEST);
      else gl.disable(gl.DEPTH_TEST);
    }
  }

  setDepthMask(mask: boolean): void {
    if (this._depthMask !== mask) {
      this._depthMask = mask;
      this.gl.depthMask(mask);
    }
  }

  // ── Scissor ─────────────────────────────────────────────────────────────

  setScissorTest(enabled: boolean): void {
    const gl = this.gl;
    if (this._scissorEnabled !== enabled) {
      this._scissorEnabled = enabled;
      if (enabled) gl.enable(gl.SCISSOR_TEST);
      else gl.disable(gl.SCISSOR_TEST);
    }
  }

  // ── Viewport ────────────────────────────────────────────────────────────

  setViewport(x: number, y: number, w: number, h: number): void {
    if (this._viewportX !== x || this._viewportY !== y ||
        this._viewportW !== w || this._viewportH !== h) {
      this._viewportX = x; this._viewportY = y;
      this._viewportW = w; this._viewportH = h;
      this.gl.viewport(x, y, w, h);
    }
  }

  // ── Program ─────────────────────────────────────────────────────────────

  useProgram(program: WebGLProgram): void {
    if (this._currentProgram !== program) {
      this._currentProgram = program;
      this.gl.useProgram(program);
    }
  }

  // ── VAO ─────────────────────────────────────────────────────────────────

  bindVAO(vao: WebGLVertexArrayObject | null): void {
    if (this._currentVAO !== vao) {
      this._currentVAO = vao;
      this.gl.bindVertexArray(vao);
    }
  }

  // ── FBO ─────────────────────────────────────────────────────────────────

  bindFramebuffer(fbo: WebGLFramebuffer | null): void {
    if (this._currentFBO !== fbo) {
      this._currentFBO = fbo;
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
    }
  }

  // ── Texture ─────────────────────────────────────────────────────────────

  activeTexture(slot: number): void {
    if (this._activeTexture !== slot) {
      this._activeTexture = slot;
      this.gl.activeTexture(this.gl.TEXTURE0 + slot);
    }
  }

  // ── Uniform cache ───────────────────────────────────────────────────────

  getUniformLocation(program: WebGLProgram, name: string): WebGLUniformLocation | null {
    let cache = this._uniformCache.get(program);
    if (!cache) {
      cache = new Map();
      this._uniformCache.set(program, cache);
    }
    if (cache.has(name)) return cache.get(name)!;
    const loc = this.gl.getUniformLocation(program, name);
    cache.set(name, loc);
    return loc;
  }

  // ── Pass boundary ───────────────────────────────────────────────────────

  /**
   * Reset GL state to known defaults at the start of each render pass.
   * Call this between passes to prevent state leaks.
   */
  resetForPass(): void {
    const gl = this.gl;
    this.setAlphaBlend();
    this.setDepthTest(false);
    this.setDepthMask(true);
    this.setScissorTest(false);
    this.bindFramebuffer(null);
  }

  /**
   * Compile a shader with Three.js-style error reporting.
   * Returns null on failure and logs annotated source.
   */
  compileShader(type: number, source: string): WebGLShader | null {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? '';
      const typeName = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
      // Annotate source with line numbers (Three.js style)
      const lines = source.split('\n');
      const annotated = lines.map((l, i) => `${(i + 1).toString().padStart(4)}: ${l}`).join('\n');
      console.error(`[GLStateCache] ${typeName} shader compile error:\n${log}\n\nSource:\n${annotated}`);
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  /**
   * Link a program with error reporting.
   */
  linkProgram(program: WebGLProgram): boolean {
    const gl = this.gl;
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? '';
      console.error(`[GLStateCache] Program link error:\n${log}`);
      return false;
    }
    return true;
  }
}
