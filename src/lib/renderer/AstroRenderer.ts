/**
 * AstroRenderer.ts — 核心 WebGL 渲染器
 *
 * 对标 AT Renderer 类:
 *   - Renderer.WEBGL1 / WEBGL2 自动切换
 *   - Renderer.extensions (WEBGL_draw_buffers, WEBGL_depth_texture, OES_element_index_uint …)
 *   - Renderer.context
 *   - Renderer.SHADOWS_HIGH / MED / LOW
 *   - Renderer.UBO (Uniform Buffer Objects, WebGL2 only)
 *   - Renderer.overrideViewport
 *   - static instance 单例
 *
 * AT bundle 分析: gl.bindTexture(27x), gl.createProgram(3x), uniformMatrix(10x)
 */

// ── Constants (mirror AT enum values) ────────────────────────────────────────

export const WEBGL1 = 1 as const;
export const WEBGL2 = 2 as const;
export type WebGLVersion = typeof WEBGL1 | typeof WEBGL2;

export const SHADOWS_LOW  = 0 as const;
export const SHADOWS_MED  = 1 as const;
export const SHADOWS_HIGH = 2 as const;
export type ShadowQuality = typeof SHADOWS_LOW | typeof SHADOWS_MED | typeof SHADOWS_HIGH;

// ── Extension registry ────────────────────────────────────────────────────────

export interface AstroExtensions {
  drawBuffers:    WEBGL_draw_buffers | null;        // MRT (WebGL1 fallback)
  depthTexture:   WEBGL_depth_texture | null;       // gl.DEPTH_COMPONENT
  elementIndexUint: OES_element_index_uint | null;  // 32-bit index buffers
  floatTextures:  OES_texture_float | null;         // HDR
  halfFloat:      OES_texture_half_float | null;    // mediump HDR
  instancing:     ANGLE_instanced_arrays | null;    // WebGL1 instancing
  vertexArrays:   OES_vertex_array_object | null;   // WebGL1 VAO
  colorBufferFloat: EXT_color_buffer_float | null;  // WebGL2 HDR FBO
}

// ── Viewport ──────────────────────────────────────────────────────────────────

export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ── RendererOptions ───────────────────────────────────────────────────────────

export interface RendererOptions {
  canvas?:        HTMLCanvasElement;
  alpha?:         boolean;
  antialias?:     boolean;
  depth?:         boolean;
  stencil?:       boolean;
  premultipliedAlpha?: boolean;
  preserveDrawingBuffer?: boolean;
  powerPreference?: 'default' | 'high-performance' | 'low-power';
  shadowQuality?: ShadowQuality;
}

// ── AstroRenderer ─────────────────────────────────────────────────────────────

export class AstroRenderer {
  // ── Static singleton (AT pattern) ──────────────────────────────────────────
  static readonly WEBGL1 = WEBGL1;
  static readonly WEBGL2 = WEBGL2;
  static readonly SHADOWS_LOW  = SHADOWS_LOW;
  static readonly SHADOWS_MED  = SHADOWS_MED;
  static readonly SHADOWS_HIGH = SHADOWS_HIGH;

  private static _instance: AstroRenderer | null = null;

  /** Get or create singleton. Pass opts only on first call. */
  static getInstance(opts?: RendererOptions): AstroRenderer {
    if (!AstroRenderer._instance) {
      AstroRenderer._instance = new AstroRenderer(opts);
    }
    return AstroRenderer._instance;
  }

  /** Destroy singleton (e.g. on hot-reload). */
  static destroyInstance(): void {
    AstroRenderer._instance?.dispose();
    AstroRenderer._instance = null;
  }

  // ── Public API surface ──────────────────────────────────────────────────────

  readonly canvas:     HTMLCanvasElement;
  readonly context:    WebGLRenderingContext | WebGL2RenderingContext;
  readonly version:    WebGLVersion;
  readonly extensions: AstroExtensions;
  shadowQuality:       ShadowQuality;

  /** When set, setViewport() is ignored and this rect is used instead (AT overrideViewport) */
  overrideViewport: Viewport | null = null;

  // ── UBO registry (WebGL2 only) ──────────────────────────────────────────────
  private _ubos = new Map<string, WebGLBuffer>();
  private _uboBindingPoints = new Map<string, number>();
  private _nextUBOBinding = 0;

  // ── Internal state ──────────────────────────────────────────────────────────
  private _viewport: Viewport = { x: 0, y: 0, width: 0, height: 0 };
  private _clearColor = { r: 0, g: 0, b: 0, a: 1 };

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(opts: RendererOptions = {}) {
    const canvas = opts.canvas ?? document.createElement('canvas');
    this.canvas = canvas;

    const ctxOpts: WebGLContextAttributes = {
      alpha:                opts.alpha                ?? false,
      antialias:            opts.antialias            ?? false,
      depth:                opts.depth                ?? true,
      stencil:              opts.stencil              ?? false,
      premultipliedAlpha:   opts.premultipliedAlpha   ?? false,
      preserveDrawingBuffer:opts.preserveDrawingBuffer ?? false,
      powerPreference:      opts.powerPreference      ?? 'high-performance',
    };

    // WebGL2 → WebGL1 fallback (AT: Renderer.WEBGL1/WEBGL2)
    let gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
    let version: WebGLVersion = WEBGL2;

    gl = canvas.getContext('webgl2', ctxOpts) as WebGL2RenderingContext | null;
    if (!gl) {
      gl = (
        canvas.getContext('webgl', ctxOpts) ??
        canvas.getContext('experimental-webgl', ctxOpts)
      ) as WebGLRenderingContext | null;
      version = WEBGL1;
    }

    if (!gl) {
      throw new Error('[AstroRenderer] WebGL not supported in this environment.');
    }

    this.context = gl;
    this.version = version;
    this.shadowQuality = opts.shadowQuality ?? SHADOWS_MED;

    // Acquire extensions
    this.extensions = this._acquireExtensions(gl, version);

    // Initial viewport
    this.setViewport(0, 0, canvas.width, canvas.height);

    if (import.meta.env?.DEV) {
      console.info(
        `[AstroRenderer] WebGL${version} context created.`,
        `Extensions:`, Object.entries(this.extensions)
          .filter(([, v]) => v !== null).map(([k]) => k)
      );
    }
  }

  // ── Extensions ──────────────────────────────────────────────────────────────

  private _acquireExtensions(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    version: WebGLVersion,
  ): AstroExtensions {
    const get = <T>(name: string): T | null => gl.getExtension(name) as T | null;

    if (version === WEBGL2) {
      const gl2 = gl as WebGL2RenderingContext;
      // WebGL2 has draw_buffers & depth_texture natively; extension objects may be null but capability exists.
      return {
        drawBuffers:      get<WEBGL_draw_buffers>('WEBGL_draw_buffers'),       // usually null in WGL2 (built-in)
        depthTexture:     get<WEBGL_depth_texture>('WEBGL_depth_texture'),     // usually null in WGL2 (built-in)
        elementIndexUint: get<OES_element_index_uint>('OES_element_index_uint'),
        floatTextures:    get<OES_texture_float>('OES_texture_float'),
        halfFloat:        get<OES_texture_half_float>('OES_texture_half_float'),
        instancing:       get<ANGLE_instanced_arrays>('ANGLE_instanced_arrays'),
        vertexArrays:     get<OES_vertex_array_object>('OES_vertex_array_object'),
        colorBufferFloat: get<EXT_color_buffer_float>('EXT_color_buffer_float'),
      };
    } else {
      return {
        drawBuffers:      get<WEBGL_draw_buffers>('WEBGL_draw_buffers'),
        depthTexture:     get<WEBGL_depth_texture>('WEBGL_depth_texture'),
        elementIndexUint: get<OES_element_index_uint>('OES_element_index_uint'),
        floatTextures:    get<OES_texture_float>('OES_texture_float'),
        halfFloat:        get<OES_texture_half_float>('OES_texture_half_float'),
        instancing:       get<ANGLE_instanced_arrays>('ANGLE_instanced_arrays'),
        vertexArrays:     get<OES_vertex_array_object>('OES_vertex_array_object'),
        colorBufferFloat: null,
      };
    }
  }

  // ── Viewport ─────────────────────────────────────────────────────────────────

  setViewport(x: number, y: number, width: number, height: number): void {
    if (this.overrideViewport) {
      const v = this.overrideViewport;
      this.context.viewport(v.x, v.y, v.width, v.height);
      this._viewport = { ...v };
    } else {
      this.context.viewport(x, y, width, height);
      this._viewport = { x, y, width, height };
    }
  }

  get viewport(): Readonly<Viewport> { return this._viewport; }

  /** Resize canvas + update viewport */
  resize(width: number, height: number): void {
    this.canvas.width  = width;
    this.canvas.height = height;
    this.setViewport(0, 0, width, height);
  }

  // ── Clear ─────────────────────────────────────────────────────────────────────

  setClearColor(r: number, g: number, b: number, a = 1): void {
    this._clearColor = { r, g, b, a };
    const { gl } = this as any;
    this.context.clearColor(r, g, b, a);
  }

  clear(color = true, depth = true, stencil = false): void {
    const gl = this.context;
    gl.clearColor(this._clearColor.r, this._clearColor.g, this._clearColor.b, this._clearColor.a);
    let bits = 0;
    if (color)   bits |= gl.COLOR_BUFFER_BIT;
    if (depth)   bits |= gl.DEPTH_BUFFER_BIT;
    if (stencil) bits |= gl.STENCIL_BUFFER_BIT;
    gl.clear(bits);
  }

  // ── Shader compilation ────────────────────────────────────────────────────────

  createShader(type: number, src: string): WebGLShader {
    const gl = this.context;
    const shader = gl.createShader(type);
    if (!shader) throw new Error('[AstroRenderer] gl.createShader failed');
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? 'unknown';
      gl.deleteShader(shader);
      const typeName = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
      throw new Error(`[AstroRenderer] ${typeName} shader compile error:\n${log}`);
    }
    return shader;
  }

  createProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.context;
    const vert = this.createShader(gl.VERTEX_SHADER, vertSrc);
    const frag = this.createShader(gl.FRAGMENT_SHADER, fragSrc);

    const program = gl.createProgram();
    if (!program) throw new Error('[AstroRenderer] gl.createProgram failed');
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    gl.deleteShader(vert);
    gl.deleteShader(frag);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? 'unknown';
      gl.deleteProgram(program);
      throw new Error(`[AstroRenderer] Program link error:\n${log}`);
    }
    return program;
  }

  // ── Draw calls ────────────────────────────────────────────────────────────────

  drawArrays(mode: number, first: number, count: number): void {
    this.context.drawArrays(mode, first, count);
  }

  drawElements(mode: number, count: number, type: number, offset: number): void {
    this.context.drawElements(mode, count, type, offset);
  }

  // ── UBO (WebGL2 only — AT: Renderer.UBO) ─────────────────────────────────────

  /**
   * Create or update a named UBO.
   * Usage:
   *   renderer.setUBO('PerFrame', new Float32Array([...]))
   *   renderer.bindUBO('PerFrame', program, 'PerFrame')
   */
  setUBO(name: string, data: BufferSource): void {
    if (this.version !== WEBGL2) return;
    const gl = this.context as WebGL2RenderingContext;

    let buf = this._ubos.get(name) ?? null;
    if (!buf) {
      buf = gl.createBuffer()!;
      this._ubos.set(name, buf);
      this._uboBindingPoints.set(name, this._nextUBOBinding++);
    }
    const bindingPoint = this._uboBindingPoints.get(name)!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, buf);
    gl.bufferData(gl.UNIFORM_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, bindingPoint, buf);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  /** Bind named UBO to a program's named block. */
  bindUBO(uboName: string, program: WebGLProgram, blockName: string): void {
    if (this.version !== WEBGL2) return;
    const gl = this.context as WebGL2RenderingContext;
    const bindingPoint = this._uboBindingPoints.get(uboName);
    if (bindingPoint === undefined) return;
    const blockIndex = gl.getUniformBlockIndex(program, blockName);
    if (blockIndex === gl.INVALID_INDEX) return;
    gl.uniformBlockBinding(program, blockIndex, bindingPoint);
  }

  // ── State helpers ─────────────────────────────────────────────────────────────

  enableDepthTest(fn: number = this.context.LEQUAL): void {
    const gl = this.context;
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(fn);
  }

  enableBlend(src = this.context.SRC_ALPHA, dst = this.context.ONE_MINUS_SRC_ALPHA): void {
    const gl = this.context;
    gl.enable(gl.BLEND);
    gl.blendFunc(src, dst);
  }

  disableBlend(): void {
    this.context.disable(this.context.BLEND);
  }

  // ── Shadow quality helpers ────────────────────────────────────────────────────

  /** Returns shadow map resolution matching current shadowQuality */
  get shadowMapSize(): number {
    switch (this.shadowQuality) {
      case SHADOWS_HIGH: return 4096;
      case SHADOWS_MED:  return 2048;
      case SHADOWS_LOW:  return 1024;
    }
  }

  // ── Dispose ──────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.context;
    for (const buf of this._ubos.values()) {
      gl.deleteBuffer(buf);
    }
    this._ubos.clear();
    this._uboBindingPoints.clear();
  }
}
