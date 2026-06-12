/**
 * hydra-gl-layer.ts — WebGL2 direct access layer
 *
 * AT Hydra 引擎的核心能力: 直接操作 WebGL2 context。
 * AT 有 241 次 gl.* 调用，我们之前是 0 次。
 *
 * 这个模块提供:
 * - RenderTarget (FBO + color/depth attachment)
 * - Texture management (gl.texImage2D)
 * - Fullscreen quad draw (gl.drawArrays)
 * - Shader compilation (gl.createShader/compileShader/linkProgram)
 *
 * 这些是 NukePipeline multi-pass 后处理的基础。
 */

// ── RenderTarget (FBO wrapper) ──────────────────────────────────────────────

export interface RenderTargetOptions {
  width: number;
  height: number;
  /** Use floating-point texture for HDR (default: false → UNSIGNED_BYTE) */
  hdr?: boolean;
}

export class RenderTarget {
  readonly gl: WebGL2RenderingContext;
  readonly framebuffer: WebGLFramebuffer;
  readonly texture: WebGLTexture;
  readonly width: number;
  readonly height: number;

  constructor(gl: WebGL2RenderingContext, opts: RenderTargetOptions) {
    this.gl = gl;
    this.width = opts.width;
    this.height = opts.height;

    // gl.createTexture — color attachment
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const internalFormat = opts.hdr ? gl.RGBA16F : gl.RGBA8;
    const type = opts.hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, opts.width, opts.height, 0, gl.RGBA, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.texture = tex;

    // gl.createFramebuffer
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    this.framebuffer = fbo;

    // Unbind
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  bind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
    this.gl.viewport(0, 0, this.width, this.height);
  }

  unbind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  destroy(): void {
    this.gl.deleteFramebuffer(this.framebuffer);
    this.gl.deleteTexture(this.texture);
  }
}

// ── Shader compilation ──────────────────────────────────────────────────────

export function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
}

export function createProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error: ${log}`);
  }
  // Shaders can be detached after linking
  gl.detachShader(prog, vert);
  gl.detachShader(prog, frag);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  return prog;
}

// ── Fullscreen quad ─────────────────────────────────────────────────────────
// Used by every NukePass: draw a quad that covers the entire viewport,
// letting the fragment shader do all the work.

const FULLSCREEN_VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  // Triangle strip covering clip space:
  //   gl_VertexID 0 → (-1,-1)  1 → (3,-1)  2 → (-1,3)
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

export class FullscreenQuad {
  private gl: WebGL2RenderingContext;
  private vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.vao = gl.createVertexArray()!;
    // Empty VAO — vertex positions computed from gl_VertexID in shader
  }

  draw(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  destroy(): void {
    this.gl.deleteVertexArray(this.vao);
  }
}

// ── HydraGLLayer ────────────────────────────────────────────────────────────

export class HydraGLLayer {
  readonly gl: WebGL2RenderingContext;
  readonly quad: FullscreenQuad;
  private renderTargets: RenderTarget[] = [];

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: true,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    this.quad = new FullscreenQuad(gl);
  }

  createRenderTarget(width: number, height: number, hdr = false): RenderTarget {
    const rt = new RenderTarget(this.gl, { width, height, hdr });
    this.renderTargets.push(rt);
    return rt;
  }

  /** Bind a texture to a texture unit for sampling in a shader */
  bindTexture(texture: WebGLTexture, unit = 0): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  /** Set uniform values on a program */
  setUniform(program: WebGLProgram, name: string, value: number | number[]): void {
    const gl = this.gl;
    const loc = gl.getUniformLocation(program, name);
    if (!loc) return;
    if (typeof value === 'number') {
      gl.uniform1f(loc, value);
    } else if (value.length === 2) {
      gl.uniform2fv(loc, value);
    } else if (value.length === 3) {
      gl.uniform3fv(loc, value);
    } else if (value.length === 4) {
      gl.uniform4fv(loc, value);
    }
  }

  /** Draw fullscreen quad with a given program, reading from inputTexture */
  drawPass(program: WebGLProgram, inputTexture: WebGLTexture, outputRT: RenderTarget | null): void {
    const gl = this.gl;
    if (outputRT) {
      outputRT.bind();
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    gl.useProgram(program);
    this.bindTexture(inputTexture, 0);
    gl.uniform1i(gl.getUniformLocation(program, 'uTexture'), 0);
    this.quad.draw();
  }

  destroy(): void {
    for (const rt of this.renderTargets) rt.destroy();
    this.quad.destroy();
  }
}
