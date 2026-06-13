/**
 * InstancedMesh.ts — WebGL2 instanced rendering for cell species
 *
 * Draws N identical-geometry cells in a single draw call using:
 *   gl.drawArraysInstanced / gl.drawElementsInstanced
 *
 * Per-instance attributes (divisor = 1):
 *   mat4  a_modelMatrix  — position / scale / rotation (columns 0-3)
 *   vec4  a_color        — RGBA fill colour
 *   float a_opacity      — independent opacity multiplier
 *
 * Shared (divisor = 0):
 *   vec2  a_position     — unit-quad vertex positions
 *   vec2  a_uv           — UV coords passed to the fragment shader
 *
 * Usage:
 *   const mesh = new InstancedMesh(gl, vertSrc, fragSrc);
 *   mesh.setInstanceCount(7);
 *   mesh.setInstanceAttribute(0, mat4, color, opacity);
 *   mesh.draw();
 *
 * AT reference: 38× instanced references in their particle / tube pipelines.
 * Each species group in CellInstanceManager owns one InstancedMesh.
 */

// ── Vertex shader ────────────────────────────────────────────────────────────
export const INSTANCED_VERT = /* glsl */ `#version 300 es
precision highp float;

// per-vertex (shared geometry — unit quad)
in vec2 a_position;
in vec2 a_uv;

// per-instance (divisor = 1)
in mat4 a_modelMatrix;   // 4 consecutive attribute slots (0-3 of instance block)
in vec4 a_color;
in float a_opacity;

// built-in uniforms
uniform mat4 u_view;
uniform mat4 u_projection;

// to fragment
out vec2 v_uv;
out vec4 v_color;
out float v_opacity;

void main() {
  v_uv      = a_uv;
  v_color   = a_color;
  v_opacity = a_opacity;
  gl_Position = u_projection * u_view * a_modelMatrix * vec4(a_position, 0.0, 1.0);
}
`;

// ── Fragment shader (passthrough — species logic lives in SDF renderer) ──────
export const INSTANCED_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec2   v_uv;
in vec4   v_color;
in float  v_opacity;

out vec4 fragColor;

void main() {
  // Simple SDF rounded-rect for the instanced quad.
  // Replace with full species SDF by swapping fragment source in CellInstanceManager.
  vec2 p  = v_uv * 2.0 - 1.0;          // [-1,1]
  vec2 b  = vec2(0.88, 0.78);
  float r = 0.12;
  vec2 q  = abs(p) - b + r;
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;

  float alpha = (1.0 - smoothstep(-0.012, 0.012, d)) * v_opacity;
  fragColor = vec4(v_color.rgb, v_color.a * alpha);
}
`;

// ── Types ────────────────────────────────────────────────────────────────────

export interface InstanceData {
  /** Column-major mat4 (16 floats) */
  modelMatrix: Float32Array;
  /** RGBA 0-1 */
  color: [number, number, number, number];
  /** 0-1 */
  opacity: number;
}

// ── InstancedMesh ────────────────────────────────────────────────────────────

const FLOATS_PER_MAT4   = 16;
const FLOATS_PER_COLOR  = 4;
const FLOATS_PER_OPACITY = 1;
/** Total floats per instance in the interleaved buffer */
const FLOATS_PER_INSTANCE = FLOATS_PER_MAT4 + FLOATS_PER_COLOR + FLOATS_PER_OPACITY; // 21

export class InstancedMesh {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  // geometry
  private vaoGeom: WebGLVertexArrayObject;
  private vboGeom: WebGLBuffer;
  private ibo: WebGLBuffer | null = null;
  private indexCount = 0;

  // instancing
  private vaoInstanced: WebGLVertexArrayObject;
  private vboInstances: WebGLBuffer;
  private instanceBuffer: Float32Array;
  private _instanceCount = 0;
  private _maxInstances: number;

  // uniforms
  private uView: WebGLUniformLocation | null;
  private uProjection: WebGLUniformLocation | null;

  /** Static identity mat4 used as view/projection default */
  private static IDENTITY = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  // ── constructor ─────────────────────────────────────────────────────────────

  constructor(
    gl: WebGL2RenderingContext,
    vertSrc = INSTANCED_VERT,
    fragSrc = INSTANCED_FRAG,
    maxInstances = 1024,
  ) {
    this.gl = gl;
    this._maxInstances = maxInstances;
    this.instanceBuffer = new Float32Array(maxInstances * FLOATS_PER_INSTANCE);

    this.program = this._compileProgram(vertSrc, fragSrc);

    // Unit quad: 2 triangles
    // prettier-ignore
    const quadVerts = new Float32Array([
      // x     y    u    v
      -0.5, -0.5,  0.0, 0.0,
       0.5, -0.5,  1.0, 0.0,
       0.5,  0.5,  1.0, 1.0,
      -0.5,  0.5,  0.0, 1.0,
    ]);
    const quadIdx = new Uint16Array([0, 1, 2, 2, 3, 0]);
    this.indexCount = quadIdx.length;

    // ── Geometry VAO ──────────────────────────────────────────────────────────
    this.vaoGeom = gl.createVertexArray()!;
    this.vboGeom = gl.createBuffer()!;
    this.ibo     = gl.createBuffer()!;

    gl.bindVertexArray(this.vaoGeom);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboGeom);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quadIdx, gl.STATIC_DRAW);

    const stride = 4 * 4; // 4 floats × 4 bytes
    const aPos = gl.getAttribLocation(this.program, 'a_position');
    const aUV  = gl.getAttribLocation(this.program, 'a_uv');
    if (aPos >= 0) { gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, stride, 0); }
    if (aUV  >= 0) { gl.enableVertexAttribArray(aUV);  gl.vertexAttribPointer(aUV,  2, gl.FLOAT, false, stride, 8); }

    // ── Instance VAO (inherits ibo binding from vaoGeom scope) ───────────────
    this.vaoInstanced = gl.createVertexArray()!;
    this.vboInstances = gl.createBuffer()!;

    gl.bindVertexArray(this.vaoInstanced);
    // re-bind geometry so the same IBO is active
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboGeom);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    if (aPos >= 0) { gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, stride, 0); }
    if (aUV  >= 0) { gl.enableVertexAttribArray(aUV);  gl.vertexAttribPointer(aUV,  2, gl.FLOAT, false, stride, 8); }

    // per-instance buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceBuffer, gl.DYNAMIC_DRAW);
    this._bindInstanceAttribs();

    gl.bindVertexArray(null);

    // uniforms
    this.uView       = gl.getUniformLocation(this.program, 'u_view');
    this.uProjection = gl.getUniformLocation(this.program, 'u_projection');
  }

  // ── public API ──────────────────────────────────────────────────────────────

  /** Set the number of instances to draw (≤ maxInstances) */
  setInstanceCount(n: number): void {
    this._instanceCount = Math.min(n, this._maxInstances);
  }

  get instanceCount(): number { return this._instanceCount; }

  /**
   * Write per-instance data for index `i`.
   * Call upload() after all setInstanceAttribute() calls to push to GPU.
   */
  setInstanceAttribute(i: number, data: InstanceData): void {
    if (i < 0 || i >= this._maxInstances) return;
    const base = i * FLOATS_PER_INSTANCE;

    // mat4 (16 floats)
    this.instanceBuffer.set(data.modelMatrix.subarray(0, 16), base);

    // vec4 color (4 floats)
    const cb = base + FLOATS_PER_MAT4;
    this.instanceBuffer[cb]     = data.color[0];
    this.instanceBuffer[cb + 1] = data.color[1];
    this.instanceBuffer[cb + 2] = data.color[2];
    this.instanceBuffer[cb + 3] = data.color[3];

    // float opacity (1 float)
    this.instanceBuffer[base + FLOATS_PER_MAT4 + FLOATS_PER_COLOR] = data.opacity;
  }

  /** Upload instance buffer to GPU */
  upload(): void {
    const { gl } = this;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboInstances);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      this.instanceBuffer,
      0,
      this._instanceCount * FLOATS_PER_INSTANCE,
    );
  }

  /**
   * Issue a single instanced draw call.
   * @param view       column-major mat4 (defaults to identity)
   * @param projection column-major mat4 (defaults to identity)
   */
  draw(
    view: Float32Array = InstancedMesh.IDENTITY,
    projection: Float32Array = InstancedMesh.IDENTITY,
  ): void {
    if (this._instanceCount === 0) return;
    const { gl } = this;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uView,       false, view);
    gl.uniformMatrix4fv(this.uProjection, false, projection);

    gl.bindVertexArray(this.vaoInstanced);
    gl.drawElementsInstanced(
      gl.TRIANGLES,
      this.indexCount,
      gl.UNSIGNED_SHORT,
      0,
      this._instanceCount,
    );
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const { gl } = this;
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.vboGeom);
    gl.deleteBuffer(this.vboInstances);
    if (this.ibo) gl.deleteBuffer(this.ibo);
    gl.deleteVertexArray(this.vaoGeom);
    gl.deleteVertexArray(this.vaoInstanced);
  }

  // ── private helpers ─────────────────────────────────────────────────────────

  /**
   * Bind per-instance attributes with vertexAttribDivisor(slot, 1).
   * mat4 occupies 4 consecutive vec4 attribute slots.
   * Must be called with vboInstances bound to ARRAY_BUFFER.
   */
  private _bindInstanceAttribs(): void {
    const { gl } = this;
    const byteStride = FLOATS_PER_INSTANCE * 4;

    // mat4 a_modelMatrix → slots aModelMatrix … aModelMatrix+3
    const aModel = gl.getAttribLocation(this.program, 'a_modelMatrix');
    if (aModel >= 0) {
      for (let col = 0; col < 4; col++) {
        const loc = aModel + col;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, byteStride, col * 16);
        gl.vertexAttribDivisor(loc, 1);
      }
    }

    // vec4 a_color
    const aColor = gl.getAttribLocation(this.program, 'a_color');
    if (aColor >= 0) {
      gl.enableVertexAttribArray(aColor);
      gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, byteStride, FLOATS_PER_MAT4 * 4);
      gl.vertexAttribDivisor(aColor, 1);
    }

    // float a_opacity
    const aOpacity = gl.getAttribLocation(this.program, 'a_opacity');
    if (aOpacity >= 0) {
      gl.enableVertexAttribArray(aOpacity);
      gl.vertexAttribPointer(aOpacity, 1, gl.FLOAT, false, byteStride, (FLOATS_PER_MAT4 + FLOATS_PER_COLOR) * 4);
      gl.vertexAttribDivisor(aOpacity, 1);
    }
  }

  private _compileProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const { gl } = this;
    const vert = this._compileShader(gl.VERTEX_SHADER,   vertSrc);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`InstancedMesh link error: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const { gl } = this;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(`InstancedMesh shader error: ${gl.getShaderInfoLog(s)}`);
    }
    return s;
  }
}
