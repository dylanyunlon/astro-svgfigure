/**
 * AstroProgram.ts — WebGL shader program 封装
 *
 * 对标 AT Renderer 的 shader program 管理:
 *   - compile vertex + fragment (复用 AstroRenderer.createProgram)
 *   - uniform / attribute location 缓存 (避免 AT 分析到的大量 getUniformLocation 开销)
 *   - uniform setters: setMatrix4, setFloat, setInt, setVec2/3/4, setTexture
 *   - Texture unit 自动分配
 *
 * AT bundle: uniformMatrix 调用 10 次, gl.createProgram 3 次 → 我们精确对齐。
 */

import { AstroRenderer, WEBGL2 } from './AstroRenderer.js';
import type { UniformBuffer } from './UniformBuffer.js';

// ── AstroProgram ──────────────────────────────────────────────────────────────

export class AstroProgram {
  readonly program:  WebGLProgram;
  readonly renderer: AstroRenderer;

  private _uniformCache  = new Map<string, WebGLUniformLocation | null>();
  private _attribCache   = new Map<string, number>();
  private _textureUnit   = 0;

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(
    renderer: AstroRenderer,
    vertSrc: string,
    fragSrc: string,
  ) {
    this.renderer = renderer;
    this.program  = renderer.createProgram(vertSrc, fragSrc);
  }

  // ── Use / bind ───────────────────────────────────────────────────────────────

  use(): this {
    this.renderer.context.useProgram(this.program);
    this._textureUnit = 0; // reset texture unit counter on each use
    return this;
  }

  // ── Location lookup (cached) ─────────────────────────────────────────────────

  uniformLocation(name: string): WebGLUniformLocation | null {
    if (!this._uniformCache.has(name)) {
      const loc = this.renderer.context.getUniformLocation(this.program, name);
      this._uniformCache.set(name, loc);
    }
    return this._uniformCache.get(name) ?? null;
  }

  attribLocation(name: string): number {
    if (!this._attribCache.has(name)) {
      const loc = this.renderer.context.getAttribLocation(this.program, name);
      this._attribCache.set(name, loc);
    }
    return this._attribCache.get(name)!;
  }

  // ── Uniform setters ──────────────────────────────────────────────────────────

  /** Column-major 4×4 matrix (Float32Array[16]) — AT: uniformMatrix 10 calls */
  setMatrix4(name: string, mat: Float32Array, transpose = false): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniformMatrix4fv(loc, transpose, mat);
    return this;
  }

  /** Column-major 3×3 matrix (Float32Array[9]) */
  setMatrix3(name: string, mat: Float32Array, transpose = false): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniformMatrix3fv(loc, transpose, mat);
    return this;
  }

  setFloat(name: string, value: number): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniform1f(loc, value);
    return this;
  }

  setInt(name: string, value: number): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniform1i(loc, value);
    return this;
  }

  setVec2(name: string, x: number, y: number): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniform2f(loc, x, y);
    return this;
  }

  setVec3(name: string, x: number, y: number, z: number): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniform3f(loc, x, y, z);
    return this;
  }

  setVec4(name: string, x: number, y: number, z: number, w: number): this {
    const loc = this.uniformLocation(name);
    if (loc !== null) this.renderer.context.uniform4f(loc, x, y, z, w);
    return this;
  }

  /**
   * Bind a texture to the next auto-allocated texture unit.
   * AT: gl.bindTexture called 27 times → unit management is critical.
   */
  setTexture(
    name: string,
    texture: WebGLTexture,
    target: number = this.renderer.context.TEXTURE_2D,
  ): this {
    const gl  = this.renderer.context;
    const loc = this.uniformLocation(name);
    if (loc === null) return this;

    const unit = this._textureUnit++;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, texture);             // gl.bindTexture — mirrors AT's 27 calls
    gl.uniform1i(loc, unit);
    return this;
  }

  /** Bind a texture to an explicit unit (for shared textures across programs). */
  setTextureAt(
    name: string,
    texture: WebGLTexture,
    unit: number,
    target: number = this.renderer.context.TEXTURE_2D,
  ): this {
    const gl  = this.renderer.context;
    const loc = this.uniformLocation(name);
    if (loc === null) return this;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, texture);
    gl.uniform1i(loc, unit);
    return this;
  }

  // ── Attribute helpers ────────────────────────────────────────────────────────

  /** Enable a vertex attrib array by name. */
  enableAttrib(name: string): this {
    const loc = this.attribLocation(name);
    if (loc >= 0) this.renderer.context.enableVertexAttribArray(loc);
    return this;
  }

  /** Bind current buffer to attrib and describe layout. */
  attribPointer(
    name: string,
    size: number,
    type: number,
    normalized = false,
    stride = 0,
    offset = 0,
  ): this {
    const loc = this.attribLocation(name);
    if (loc >= 0) {
      this.renderer.context.vertexAttribPointer(loc, size, type, normalized, stride, offset);
    }
    return this;
  }

  // ── UBO binding (WebGL2) ─────────────────────────────────────────────────────

  /**
   * Bind a UniformBuffer to this program's named uniform block.
   *
   * WebGL2: calls gl.uniformBlockBinding so the block index maps to the UBO's
   * binding point — after this, gl.bindBufferBase on that binding point is all
   * that's needed each frame.
   *
   * WebGL1 fallback: calls ubo.fallbackBind(this) to push each field as an
   * individual uniform call via the existing setter methods.
   *
   * Usage:
   *   program.use();
   *   program.bindUniformBlock(viewUBO);          // blockName from layout
   *   program.bindUniformBlock(cellUBO, 'Cell');  // override blockName
   */
  bindUniformBlock(ubo: UniformBuffer, blockName?: string): this {
    const name = blockName ?? ubo.layout.blockName;

    if (this.renderer.version !== WEBGL2) {
      // WebGL1 降级 — push all fields as regular uniforms
      ubo.fallbackBind(this);
      return this;
    }

    const gl = this.renderer.context as WebGL2RenderingContext;

    // Cache the block index lookup (keyed by blockName, same as uniform locations)
    const cacheKey = `__ubo_${name}`;
    if (!this._uniformCache.has(cacheKey)) {
      const blockIndex = gl.getUniformBlockIndex(this.program, name);
      // Store the block index as a sentinel value (-1 = INVALID_INDEX)
      // We piggy-back on _uniformCache with a prefixed key so no extra Map is needed.
      this._uniformCache.set(cacheKey, blockIndex === gl.INVALID_INDEX ? null : (blockIndex as unknown as WebGLUniformLocation));
    }

    const cached = this._uniformCache.get(cacheKey);
    if (cached === null || cached === undefined) return this; // block not found in shader

    const blockIndex = cached as unknown as number;
    gl.uniformBlockBinding(this.program, blockIndex, ubo.bindingPoint);
    // Ensure the buffer is bound to its binding point
    ubo.bind();

    return this;
  }

  // ── Dispose ──────────────────────────────────────────────────────────────────

  dispose(): void {
    this.renderer.context.deleteProgram(this.program);
    this._uniformCache.clear();
    this._attribCache.clear();
  }
}
