/**
 * RenderTarget.ts — WebGL2 Framebuffer Object wrapper
 *
 * AT FXScene composite rendering: each FXScene renders into its own RenderTarget
 * (offscreen framebuffer), then Nuke post-process reads the texture.
 *
 * Features:
 *   - Single color attachment (default) or MRT via drawBuffers
 *   - Optional depth renderbuffer
 *   - HDR (RGBA16F) or LDR (RGBA8) color format
 *   - resize() recreates attachments in-place
 *
 * References:
 *   src/lib/renderers/hydra-gl-layer.ts  — original RenderTarget (single attachment)
 *   src/lib/renderers/nuke-pipeline.ts   — RenderTarget consumer pattern
 *   AT FXScene.ts: RenderTarget used for manualRender offscreen output
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface RenderTargetOptions {
  width: number;
  height: number;
  /**
   * Number of color attachments (MRT).  Each attachment gets its own texture.
   * Requires WebGL2 + drawBuffers.  Default: 1.
   */
  attachments?: number;
  /**
   * Use RGBA16F floating-point textures for HDR rendering.
   * Requires EXT_color_buffer_float.  Default: false → RGBA8.
   */
  hdr?: boolean;
  /**
   * Attach a DEPTH_COMPONENT24 renderbuffer.  Required for 3-D scenes that
   * need depth testing inside the FBO.  Default: false.
   */
  depth?: boolean;
}

// ── RenderTarget ─────────────────────────────────────────────────────────────

export class RenderTarget {
  readonly gl: WebGL2RenderingContext;

  framebuffer: WebGLFramebuffer;
  /** Primary (or only) color texture — convenience alias for textures[0]. */
  get texture(): WebGLTexture { return this.textures[0]; }

  /** All color attachment textures (length === attachmentCount). */
  textures: WebGLTexture[] = [];

  depthRenderbuffer: WebGLRenderbuffer | null = null;

  width: number;
  height: number;

  readonly attachmentCount: number;
  readonly hdr: boolean;
  readonly depth: boolean;

  constructor(gl: WebGL2RenderingContext, opts: RenderTargetOptions) {
    this.gl = gl;
    this.width = opts.width;
    this.height = opts.height;
    this.attachmentCount = opts.attachments ?? 1;
    this.hdr = opts.hdr ?? false;
    this.depth = opts.depth ?? false;

    this.framebuffer = this._createFBO();
  }

  // ── Internal creation ────────────────────────────────────────────────────

  private _createFBO(): WebGLFramebuffer {
    const { gl, width, height, hdr, depth, attachmentCount } = this;

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    const internalFormat = hdr ? gl.RGBA16F : gl.RGBA8;
    const type          = hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
    const drawBufs: number[] = [];

    this.textures = [];

    for (let i = 0; i < attachmentCount; i++) {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, internalFormat,
        width, height, 0,
        gl.RGBA, type, null,
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);

      const attachment = gl.COLOR_ATTACHMENT0 + i;
      gl.framebufferTexture2D(gl.FRAMEBUFFER, attachment, gl.TEXTURE_2D, tex, 0);
      drawBufs.push(attachment);
      this.textures.push(tex);
    }

    // MRT: tell WebGL which draw buffers are active
    if (attachmentCount > 1) {
      gl.drawBuffers(drawBufs);
    }

    // Optional depth
    if (depth) {
      const rb = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
      this.depthRenderbuffer = rb;
    }

    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn('[RenderTarget] Framebuffer incomplete — check texture format support.');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /** Bind this FBO as the current render target. */
  bind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffer);
    this.gl.viewport(0, 0, this.width, this.height);
  }

  /** Unbind — render subsequent draw calls to the canvas. */
  unbind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /**
   * Resize all attachments.  Destroys old GPU objects and recreates them.
   * Call when the canvas is resized before the next render frame.
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width  = width;
    this.height = height;
    this._destroyAttachments();
    this.framebuffer = this._createFBO();
  }

  /** Bind texture[index] to a sampler unit. */
  bindTexture(unit = 0, attachmentIndex = 0): void {
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, this.textures[attachmentIndex]);
  }

  /** Free all GPU resources. */
  destroy(): void {
    this._destroyAttachments();
    this.gl.deleteFramebuffer(this.framebuffer);
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private _destroyAttachments(): void {
    const { gl } = this;
    for (const tex of this.textures) gl.deleteTexture(tex);
    this.textures = [];
    if (this.depthRenderbuffer) {
      gl.deleteRenderbuffer(this.depthRenderbuffer);
      this.depthRenderbuffer = null;
    }
  }
}
