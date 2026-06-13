/**
 * NukePass.ts — single post-processing stage
 *
 * AT Nuke module port (133 refs).
 * Corresponds to AT's NukePass / HydraPass abstraction:
 *   upstream/pixijs-engine/src/fx/nuke/NukePass.ts
 *   upstream/pixijs-engine/src/fx/nuke/passes/
 *
 * Architecture:
 *   Each NukePass owns:
 *     • an input  RenderTarget  (may be shared / ping-pong)
 *     • an output RenderTarget  (may be shared / ping-pong)
 *     • a WebGLProgram          compiled from vert + frag source
 *     • a fullscreen quad VAO   (2 triangles covering NDC [-1, 1])
 *
 *   Nuke.render() calls pass.render(gl) in sequence; passes do NOT
 *   touch the default framebuffer — that final blit is Nuke's job.
 *
 * Lifecycle (mirrors AT):
 *   BEFORE_PASSES  → Nuke fires before the chain starts
 *   RENDER         → each NukePass.render() in order
 *   POST_RENDER    → Nuke fires after chain + final blit
 */

// ── Types ────────────────────────────────────────────────────────────────────

/** A WebGL render target: framebuffer + colour texture + optional depth. */
export interface RenderTarget {
  name: string;
  fbo: WebGLFramebuffer;
  texture: WebGLTexture;
  depthBuffer?: WebGLRenderbuffer;
  width: number;
  height: number;
}

/** Uniform value types accepted by NukePass. */
export type UniformValue =
  | number
  | [number, number]
  | [number, number, number]
  | [number, number, number, number]
  | Float32Array   // mat3 / mat4
  | WebGLTexture;

// ── Fullscreen quad geometry (NDC, 2 triangles) ───────────────────────────────

/** GLSL ES 3.00 vertex shader shared by all fullscreen passes. */
export const FULLSCREEN_VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

// Two triangles that cover the entire clip-space viewport.
// Vertex IDs 0-2 → first triangle, 3-5 → second triangle.
// No VBO needed — geometry is computed from gl_VertexID.
void main() {
  // Bit-tricks: map {0,1,2,3,4,5} → two triangles in NDC.
  float x = float((gl_VertexID & 1) << 1) - 1.0; // -1 or +1
  float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

/** Map gl_FragCoord → [0,1] UV inside the fragment shader. */
export const UV_FROM_FRAG_COORD = /* glsl */ `
vec2 uv_from_frag(vec2 resolution) {
  return gl_FragCoord.xy / resolution;
}
`;

// ── Shader helpers ────────────────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: GLenum,
  src: string,
  label: string
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? '(no log)';
    gl.deleteShader(shader);
    throw new Error(`[NukePass] shader compile error in "${label}":\n${info}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
  label: string
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) ?? '(no log)';
    gl.deleteProgram(prog);
    throw new Error(`[NukePass] program link error in "${label}":\n${info}`);
  }
  return prog;
}

// ── NukePass ─────────────────────────────────────────────────────────────────

export interface NukePassOptions {
  /** Human-readable identifier (shown in GPU debug labels). */
  name: string;

  /** GLSL ES 3.00 fragment source.  Vertex defaults to FULLSCREEN_VERT_SRC. */
  fragSrc: string;

  /** Override the default fullscreen vertex shader (rare). */
  vertSrc?: string;

  /** Input render target fed to this pass as `u_input` sampler. */
  input: RenderTarget;

  /** Output render target (the pass renders into its FBO). */
  output: RenderTarget;

  /** Additional uniform values set before each draw. */
  uniforms?: Record<string, UniformValue>;

  /** Pass is skipped when false (default: true). */
  enabled?: boolean;
}

/**
 * NukePass — one stage in the Nuke post-processing pipeline.
 *
 * Usage:
 * ```ts
 * const bloom = new NukePass({
 *   name: 'bloom-upsample',
 *   fragSrc: myBloomFrag,
 *   input: nuke.getRT('bloomDown'),
 *   output: nuke.getRT('bloomUp'),
 *   uniforms: { u_strength: 1.2 },
 * });
 * ```
 */
export class NukePass {
  readonly name: string;
  enabled: boolean;

  input: RenderTarget;
  output: RenderTarget;
  uniforms: Record<string, UniformValue>;

  private gl!: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private _compiled = false;

  constructor(options: NukePassOptions) {
    this.name    = options.name;
    this.enabled = options.enabled ?? true;
    this.input   = options.input;
    this.output  = options.output;
    this.uniforms = options.uniforms ?? {};

    // Stash sources for lazy compilation on first render.
    this._vertSrc = options.vertSrc ?? FULLSCREEN_VERT_SRC;
    this._fragSrc = options.fragSrc;
  }

  private _vertSrc: string;
  private _fragSrc: string;

  // ── Compile ────────────────────────────────────────────────────────────────

  /** Compile shaders and build the empty VAO (called lazily on first render). */
  compile(gl: WebGL2RenderingContext): void {
    if (this._compiled) return;
    this.gl = gl;

    const vert = compileShader(gl, gl.VERTEX_SHADER,   this._vertSrc, `${this.name}.vert`);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, this._fragSrc, `${this.name}.frag`);
    this.program = linkProgram(gl, vert, frag, this.name);
    gl.deleteShader(vert);
    gl.deleteShader(frag);

    // Empty VAO — geometry comes from gl_VertexID in the vertex shader.
    this.vao = gl.createVertexArray()!;

    // Assign GPU debug label when available.
    if ('KHR_debug' in gl.getSupportedExtensions?.() ?? []) {
      const ext = gl.getExtension('KHR_debug');
      ext?.objectLabel(ext.PROGRAM, this.program, -1, `NukePass::${this.name}`);
    }

    this._compiled = true;
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  /**
   * Execute this pass:
   *   1. Bind output FBO
   *   2. Use program
   *   3. Upload uniforms (input texture + custom)
   *   4. Draw 6 vertices → 2 triangles → full viewport
   */
  render(gl: WebGL2RenderingContext): void {
    if (!this.enabled) return;
    this.compile(gl);

    // Bind output framebuffer.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.output.fbo);
    gl.viewport(0, 0, this.output.width, this.output.height);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    // Always bind the primary input texture at unit 0.
    this._bindTexture(gl, 0, this.input.texture);
    this._setUniform(gl, 'u_input', 0 as unknown as UniformValue);
    this._setUniform(gl, 'u_resolution', [
      this.output.width,
      this.output.height,
    ] as [number, number]);

    // Custom uniforms.
    let texUnit = 1;
    for (const [name, value] of Object.entries(this.uniforms)) {
      if (value instanceof WebGLTexture) {
        this._bindTexture(gl, texUnit, value);
        this._setUniform(gl, name, texUnit as unknown as UniformValue);
        texUnit++;
      } else {
        this._setUniform(gl, name, value);
      }
    }

    // Fullscreen quad: 6 vertices (2 triangles), no index buffer.
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── Uniform helpers ────────────────────────────────────────────────────────

  private _bindTexture(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture): void {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  private _setUniform(gl: WebGL2RenderingContext, name: string, value: UniformValue): void {
    const loc = gl.getUniformLocation(this.program, name);
    if (loc === null) return; // uniform optimised away — ignore

    if (typeof value === 'number') {
      // Distinguish int (texture unit) from float by checking for integer.
      Number.isInteger(value)
        ? gl.uniform1i(loc, value)
        : gl.uniform1f(loc, value);
    } else if (value instanceof Float32Array) {
      value.length === 9
        ? gl.uniformMatrix3fv(loc, false, value)
        : gl.uniformMatrix4fv(loc, false, value);
    } else if (Array.isArray(value)) {
      switch ((value as number[]).length) {
        case 2: gl.uniform2f(loc, value[0], value[1]); break;
        case 3: gl.uniform3f(loc, value[0], value[1], value[2]); break;
        case 4: gl.uniform4f(loc, value[0], value[1], value[2], value[3]); break;
      }
    }
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  dispose(): void {
    if (!this._compiled) return;
    const { gl } = this;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    this._compiled = false;
  }
}
