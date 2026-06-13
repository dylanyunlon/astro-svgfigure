/**
 * AstroMesh.ts — WebGL 几何体封装
 *
 * 对标 AT Renderer 的 geometry 管理:
 *   - VAO (WebGL2 native / OES_vertex_array_object extension on WebGL1)
 *   - VBO per attribute + optional index buffer (IBO)
 *   - setAttribute(name, data, size) — uploads typed array, wires attrib pointer
 *   - setIndex(data) — 16-bit or 32-bit index buffer
 *   - draw() — drawElements or drawArrays depending on index presence
 *   - dispose() — cleanup all GPU resources
 */

import { AstroRenderer, WEBGL1 } from './AstroRenderer.js';
import { AstroProgram }           from './AstroProgram.js';

// ── AttributeDescriptor ──────────────────────────────────────────────────────

interface AttributeDescriptor {
  buffer:     WebGLBuffer;
  size:       number;       // components per vertex (1-4)
  type:       number;       // gl.FLOAT, gl.UNSIGNED_BYTE, …
  normalized: boolean;
  stride:     number;
  offset:     number;
  count:      number;       // vertex count derived from data.length / size
}

// ── AstroMesh ────────────────────────────────────────────────────────────────

export type DrawMode =
  | 'TRIANGLES'
  | 'TRIANGLE_STRIP'
  | 'TRIANGLE_FAN'
  | 'LINES'
  | 'LINE_STRIP'
  | 'POINTS';

export class AstroMesh {
  readonly renderer: AstroRenderer;

  private _vao:        WebGLVertexArrayObject | null = null;
  private _attrs      = new Map<string, AttributeDescriptor>();
  private _indexBuf:  WebGLBuffer | null = null;
  private _indexCount = 0;
  private _indexType  = 0; // gl.UNSIGNED_SHORT or gl.UNSIGNED_INT
  private _vertCount  = 0;
  private _drawMode:  DrawMode = 'TRIANGLES';

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(renderer: AstroRenderer, drawMode: DrawMode = 'TRIANGLES') {
    this.renderer  = renderer;
    this._drawMode = drawMode;
    this._vao      = this._createVAO();
  }

  // ── VAO helpers ───────────────────────────────────────────────────────────────

  private _createVAO(): WebGLVertexArrayObject | null {
    const { context: gl, version, extensions } = this.renderer;
    if (version !== WEBGL1) {
      return (gl as WebGL2RenderingContext).createVertexArray() ?? null;
    }
    // WebGL1: OES_vertex_array_object extension
    return extensions.vertexArrays?.createVertexArrayOES() ?? null;
  }

  private _bindVAO(): void {
    const { context: gl, version, extensions } = this.renderer;
    if (version !== WEBGL1) {
      (gl as WebGL2RenderingContext).bindVertexArray(this._vao);
    } else {
      extensions.vertexArrays?.bindVertexArrayOES(this._vao);
    }
  }

  private _unbindVAO(): void {
    const { context: gl, version, extensions } = this.renderer;
    if (version !== WEBGL1) {
      (gl as WebGL2RenderingContext).bindVertexArray(null);
    } else {
      extensions.vertexArrays?.bindVertexArrayOES(null);
    }
  }

  // ── setAttribute ─────────────────────────────────────────────────────────────

  /**
   * Upload a vertex attribute buffer.
   *
   * @param name       GLSL attribute name (e.g. "a_position")
   * @param data       Typed array of vertex data
   * @param size       Components per vertex vertex (1–4)
   * @param type       gl.FLOAT (default) | gl.UNSIGNED_BYTE | …
   * @param normalized Whether to normalize integer data
   * @param stride     Byte stride (0 = tightly packed)
   * @param offset     Byte offset into buffer
   * @param usage      gl.STATIC_DRAW (default) | gl.DYNAMIC_DRAW
   */
  setAttribute(
    name:       string,
    data:       BufferSource,
    size        = 3,
    type:       number = this.renderer.context.FLOAT,
    normalized  = false,
    stride      = 0,
    offset      = 0,
    usage:      number = this.renderer.context.STATIC_DRAW,
  ): this {
    const gl = this.renderer.context;

    // Reuse existing buffer if attribute already registered
    let existing = this._attrs.get(name);
    const buf = existing?.buffer ?? gl.createBuffer()!;

    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);

    // Infer vertex count from data length
    const byteLen    = (data as ArrayBufferView).byteLength
                       ?? (data as ArrayBuffer).byteLength;
    const bytePerEl  = type === gl.FLOAT           ? 4
                     : type === gl.UNSIGNED_SHORT   ? 2
                     : type === gl.UNSIGNED_INT     ? 4
                     : 1;
    const count = Math.floor(byteLen / bytePerEl / size);

    this._attrs.set(name, { buffer: buf, size, type, normalized, stride, offset, count });

    // Update vertex count from position attribute (convention: "a_position")
    if (name === 'a_position' || this._vertCount === 0) {
      this._vertCount = count;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return this;
  }

  // ── setIndex ──────────────────────────────────────────────────────────────────

  /**
   * Upload index buffer.
   * 32-bit indices require OES_element_index_uint (WebGL1) or WebGL2.
   */
  setIndex(
    data:  Uint16Array | Uint32Array,
    usage: number = this.renderer.context.STATIC_DRAW,
  ): this {
    const gl = this.renderer.context;

    if (!this._indexBuf) {
      this._indexBuf = gl.createBuffer()!;
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._indexBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, usage);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this._indexCount = data.length;
    this._indexType  = data instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    return this;
  }

  // ── Bind attributes to program ────────────────────────────────────────────────

  /**
   * Wire all attributes into the VAO for a specific program.
   * Call once after setting all attributes (or after program switch).
   */
  bindToProgram(program: AstroProgram): this {
    const gl = this.renderer.context;

    this._bindVAO();

    for (const [name, desc] of this._attrs) {
      const loc = program.attribLocation(name);
      if (loc < 0) continue;

      gl.bindBuffer(gl.ARRAY_BUFFER, desc.buffer);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(
        loc,
        desc.size,
        desc.type,
        desc.normalized,
        desc.stride,
        desc.offset,
      );
    }

    if (this._indexBuf) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._indexBuf);
    }

    this._unbindVAO();
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return this;
  }

  // ── draw ─────────────────────────────────────────────────────────────────────

  /**
   * Issue draw call. Assumes program.use() has been called.
   * Uses VAO if available; otherwise falls back to manual attrib binding
   * (requires program to be provided in that case).
   */
  draw(programFallback?: AstroProgram): void {
    const gl   = this.renderer.context;
    const mode = this._glMode();

    if (this._vao) {
      this._bindVAO();
    } else if (programFallback) {
      // No VAO support — wire attributes manually each draw
      for (const [name, desc] of this._attrs) {
        const loc = programFallback.attribLocation(name);
        if (loc < 0) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, desc.buffer);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, desc.size, desc.type, desc.normalized, desc.stride, desc.offset);
      }
      if (this._indexBuf) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._indexBuf);
      }
    }

    if (this._indexBuf) {
      this.renderer.drawElements(mode, this._indexCount, this._indexType, 0);
    } else {
      this.renderer.drawArrays(mode, 0, this._vertCount);
    }

    if (this._vao) this._unbindVAO();
  }

  // ── Instanced draw (WebGL2) ───────────────────────────────────────────────────

  drawInstanced(instanceCount: number): void {
    const { context: gl, version } = this.renderer;
    const mode = this._glMode();

    this._bindVAO();
    if (version !== WEBGL1) {
      const gl2 = gl as WebGL2RenderingContext;
      if (this._indexBuf) {
        gl2.drawElementsInstanced(mode, this._indexCount, this._indexType, 0, instanceCount);
      } else {
        gl2.drawArraysInstanced(mode, 0, this._vertCount, instanceCount);
      }
    } else {
      const ext = this.renderer.extensions.instancing;
      if (ext) {
        if (this._indexBuf) {
          ext.drawElementsInstancedANGLE(mode, this._indexCount, this._indexType, 0, instanceCount);
        } else {
          ext.drawArraysInstancedANGLE(mode, 0, this._vertCount, instanceCount);
        }
      }
    }
    this._unbindVAO();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────

  private _glMode(): number {
    const gl = this.renderer.context;
    switch (this._drawMode) {
      case 'TRIANGLES':      return gl.TRIANGLES;
      case 'TRIANGLE_STRIP': return gl.TRIANGLE_STRIP;
      case 'TRIANGLE_FAN':   return gl.TRIANGLE_FAN;
      case 'LINES':          return gl.LINES;
      case 'LINE_STRIP':     return gl.LINE_STRIP;
      case 'POINTS':         return gl.POINTS;
    }
  }

  get vertexCount():  number { return this._vertCount; }
  get indexCount():   number { return this._indexCount; }
  get hasIndex():     boolean { return this._indexBuf !== null; }

  // ── Dispose ───────────────────────────────────────────────────────────────────

  dispose(): void {
    const { context: gl, version, extensions } = this.renderer;

    for (const desc of this._attrs.values()) {
      gl.deleteBuffer(desc.buffer);
    }
    this._attrs.clear();

    if (this._indexBuf) {
      gl.deleteBuffer(this._indexBuf);
      this._indexBuf = null;
    }

    if (this._vao) {
      if (version !== WEBGL1) {
        (gl as WebGL2RenderingContext).deleteVertexArray(this._vao);
      } else {
        extensions.vertexArrays?.deleteVertexArrayOES(this._vao);
      }
      this._vao = null;
    }
  }
}
