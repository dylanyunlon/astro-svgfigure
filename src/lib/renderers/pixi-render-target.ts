/**
 * pixi-render-target.ts — Offscreen buffer management & bloom multi-pass FBO
 *
 * M011: 融合 upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/
 *       + upstream/pixijs-engine/src/rendering/renderers/gl/renderTarget/
 *       into src/lib/renderers/
 *
 * 模块职责:
 *   PixiRenderTarget      offscreen FBO wrapper — 对接 PixiJS v8 FilterSystem / TexturePool
 *   PixiRenderTargetPool  acquire/release 池，供 bloom ping-pong 使用
 *   BloomFBOPass          multi-pass bloom: extract → H-blur → V-blur → composite
 *                         使用 pixi-filters-registry.ts 的 AdvancedBloomFilter / BloomFilter
 *   BloomFBOPipeline      完整 pipeline: 包裹 BloomFBOPass 并通过 pixi-filters-registry createFilter 创建
 *
 * 上游引用:
 *   upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/RenderTarget.ts
 *   upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/RenderTargetSystem.ts
 *   upstream/pixijs-engine/src/rendering/renderers/gl/GlRenderTarget.ts
 *   upstream/pixijs-engine/src/rendering/renderers/gl/renderTarget/GlRenderTargetAdaptor.ts
 *   upstream/pixijs-engine/src/filters/FilterSystem.ts
 *   upstream/pixijs-filters/src/advanced-bloom/AdvancedBloomFilter.ts
 *
 * 对接 pixi-filters-registry.ts:
 *   BloomFBOPipeline 通过 createFilter('advanced-bloom') / createFilter('bloom') 创建过滤器。
 *   BloomFBOPass 暴露 filterName 字段可被 registry 查询。
 */

// ── PixiJS v8 渲染原语 ──────────────────────────────────────────────────────
import type { Renderer }        from '../../upstream/pixijs-engine/src/rendering/renderers/types';
import type { RenderTarget as PixiRTBase }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/RenderTarget';
import type {
  RenderSurface,
} from '../../upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/RenderTargetSystem';
import { Texture } from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture';
import { TextureSource }
  from '../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/sources/TextureSource';

// ── pixi-filters-registry 对接 ─────────────────────────────────────────────
import {
  createFilter,
  AdvancedBloomFilter,
  BloomFilter,
  KawaseBlurFilter,
} from './pixi-filters-registry';
import type {
  FilterName,
  AdvancedBloomFilterOptions,
} from './pixi-filters-registry';

// ── WebGL2 直接 helpers (from hydra-gl-layer) ──────────────────────────────
// FullscreenQuad + createProgram をそのまま流用可能
import { createProgram } from './hydra-gl-layer';

// ─────────────────────────────────────────────────────────────────────────────
// § 1. PixiRenderTarget — Offscreen buffer (对应 upstream RenderTarget)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options for creating a PixiRenderTarget.
 * Mirrors upstream RenderTargetOptions with project-specific additions.
 */
export interface PixiRenderTargetOptions {
  /** Width in pixels */
  width: number;
  /** Height in pixels */
  height: number;
  /** Pixel ratio / device pixel ratio. Default 1. */
  resolution?: number;
  /**
   * Number of color attachments (MRT).
   * Mirrors upstream: colorTextures?: number.  Default 1.
   */
  colorAttachments?: number;
  /**
   * Use RGBA16F floating-point HDR textures.
   * Requires EXT_color_buffer_float. Default false → RGBA8.
   */
  hdr?: boolean;
  /** Attach a depth renderbuffer. Default false. */
  depth?: boolean;
  /** Attach stencil. Implies depth (DEPTH24_STENCIL8). Default false. */
  stencil?: boolean;
  /**
   * MSAA sample count.  0 or 1 = no MSAA.  4 = typical 4x.
   * Mirrors GlRenderTarget.msaa logic.  Default 0.
   */
  samples?: number;
  /**
   * When true this is a root (canvas-backed) render target.
   * Mirrors upstream RenderTarget.isRoot.  Default false.
   */
  isRoot?: boolean;
}

/**
 * PixiRenderTarget — project-level offscreen buffer wrapper.
 *
 * Manages a WebGL2 Framebuffer Object for offscreen rendering.
 * Designed to slot into the bloom multi-pass pipeline and to be
 * acquired / released via PixiRenderTargetPool for ping-pong passes.
 *
 * Mirrors upstream:
 *   upstream/pixijs-engine/src/rendering/renderers/shared/renderTarget/RenderTarget.ts
 *   upstream/pixijs-engine/src/rendering/renderers/gl/GlRenderTarget.ts
 */
export class PixiRenderTarget {
  // ── identification (mirrors upstream uid pattern) ────────────────────────
  readonly uid: number;
  private static _uidCounter = 0;

  // ── GL state ─────────────────────────────────────────────────────────────
  readonly gl: WebGL2RenderingContext;
  framebuffer: WebGLFramebuffer | null = null;
  /**
   * MSAA resolve FBO (separate from framebuffer, matches GlRenderTarget pattern).
   * When samples > 1, framebuffer holds MSAA renderbuffers; resolveFramebuffer
   * holds the resolved texture attachment.
   */
  resolveFramebuffer: WebGLFramebuffer | null = null;

  /** All color attachment textures.  length === opts.colorAttachments. */
  textures: WebGLTexture[] = [];
  /** Primary color texture — shorthand for textures[0]. */
  get texture(): WebGLTexture | null { return this.textures[0] ?? null; }

  /** MSAA renderbuffers per color attachment (index mirrors GlRenderTarget.msaaRenderBuffer). */
  msaaRenderBuffers: WebGLRenderbuffer[] = [];

  /** Depth / stencil renderbuffer (mirrors GlRenderTarget.depthStencilRenderBuffer). */
  depthStencilRenderBuffer: WebGLRenderbuffer | null = null;

  // ── dimensions (mirrors GlRenderTarget) ───────────────────────────────────
  width: number;
  height: number;
  readonly resolution: number;
  readonly colorAttachments: number;
  readonly hdr: boolean;
  readonly depth: boolean;
  readonly stencil: boolean;
  readonly samples: number;
  readonly isRoot: boolean;

  /** Tracks current attached mip level (mirrors GlRenderTarget._attachedMipLevel). */
  _attachedMipLevel = 0;
  /** Tracks current attached array layer (mirrors GlRenderTarget._attachedLayer). */
  _attachedLayer = 0;

  /** Incremented on each resize — upstream dirtyId pattern. */
  dirtyId = 0;

  private _destroyed = false;

  constructor(gl: WebGL2RenderingContext, opts: PixiRenderTargetOptions) {
    this.uid           = ++PixiRenderTarget._uidCounter;
    this.gl            = gl;
    this.width         = opts.width;
    this.height        = opts.height;
    this.resolution    = opts.resolution   ?? 1;
    this.colorAttachments = opts.colorAttachments ?? 1;
    this.hdr           = opts.hdr          ?? false;
    this.depth         = opts.depth        ?? false;
    this.stencil       = opts.stencil      ?? false;
    this.samples       = opts.samples      ?? 0;
    this.isRoot        = opts.isRoot       ?? false;

    this._buildFBOs();
  }

  // ── Construction ─────────────────────────────────────────────────────────

  private _buildFBOs(): void {
    const gl            = this.gl;
    const { width, height, hdr, colorAttachments, samples } = this;
    const internalFmt   = hdr ? gl.RGBA16F      : gl.RGBA8;
    const type          = hdr ? gl.HALF_FLOAT    : gl.UNSIGNED_BYTE;
    const msaa          = samples > 1;

    // ── Resolve FBO (texture-backed) ──
    const resolveFBO = gl.createFramebuffer()!;
    this.resolveFramebuffer = resolveFBO;
    gl.bindFramebuffer(gl.FRAMEBUFFER, resolveFBO);

    const drawBufs: number[] = [];
    this.textures = [];

    for (let i = 0; i < colorAttachments; i++) {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, internalFmt,
        width, height, 0,
        gl.RGBA, type, null,
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);

      gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0,
      );
      drawBufs.push(gl.COLOR_ATTACHMENT0 + i);
      this.textures.push(tex);
    }

    if (colorAttachments > 1) {
      gl.drawBuffers(drawBufs);
    }

    // ── Depth / stencil attachment ──
    if (this.depth || this.stencil) {
      const rb = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      const dsFormat = this.stencil ? gl.DEPTH24_STENCIL8 : gl.DEPTH_COMPONENT24;
      if (msaa) {
        gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, dsFormat, width, height);
      } else {
        gl.renderbufferStorage(gl.RENDERBUFFER, dsFormat, width, height);
      }
      const dsAttachment = this.stencil
        ? gl.DEPTH_STENCIL_ATTACHMENT
        : gl.DEPTH_ATTACHMENT;
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, dsAttachment, gl.RENDERBUFFER, rb);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
      this.depthStencilRenderBuffer = rb;
    }

    this._warnIfIncomplete();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── MSAA FBO (renderbuffer-backed, mirrors GlRenderTargetAdaptor) ──
    if (msaa) {
      const msaaFBO = gl.createFramebuffer()!;
      this.framebuffer = msaaFBO;
      gl.bindFramebuffer(gl.FRAMEBUFFER, msaaFBO);

      this.msaaRenderBuffers = [];
      for (let i = 0; i < colorAttachments; i++) {
        const rb = gl.createRenderbuffer()!;
        gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
        gl.renderbufferStorageMultisample(
          gl.RENDERBUFFER, samples, internalFmt, width, height,
        );
        gl.framebufferRenderbuffer(
          gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.RENDERBUFFER, rb,
        );
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        this.msaaRenderBuffers.push(rb);
      }

      this._warnIfIncomplete();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      // Non-MSAA: both framebuffer and resolveFramebuffer point to same FBO.
      this.framebuffer = resolveFBO;
    }
  }

  private _warnIfIncomplete(): void {
    const gl     = this.gl;
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(
        `[PixiRenderTarget uid=${this.uid}] Framebuffer incomplete — status 0x${status.toString(16)}.`,
        'Check: HDR requires EXT_color_buffer_float; MSAA needs WebGL2.',
      );
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /**
   * Bind this FBO as the active render target.
   * Uses the MSAA framebuffer when samples > 1 (matches GlRenderTargetAdaptor.startRenderPass).
   */
  bind(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.width, this.height);
  }

  /** Unbind — subsequent draw calls go to the canvas default FBO. */
  unbind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /**
   * For MSAA targets: blit from the MSAA FBO into the resolve (texture) FBO.
   * Call before sampling the texture.  No-op for non-MSAA targets.
   * Mirrors GlRenderTargetAdaptor.finishRenderPass resolve logic.
   */
  resolveIfMSAA(): void {
    if (this.samples <= 1) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.framebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.resolveFramebuffer);
    gl.blitFramebuffer(
      0, 0, this.width, this.height,
      0, 0, this.width, this.height,
      gl.COLOR_BUFFER_BIT,
      gl.LINEAR,
    );
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  }

  /**
   * Bind a resolved color texture to a sampler unit for reading in a shader.
   * @param unit            - GL texture unit index (0-based).
   * @param attachmentIndex - Which color attachment to bind (0-based). Default 0.
   */
  bindTexture(unit = 0, attachmentIndex = 0): void {
    const gl = this.gl;
    const tex = this.textures[attachmentIndex];
    if (!tex) return;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  /**
   * Resize all attachments.
   * Destroys old GPU objects and recreates them.
   * Mirrors upstream RenderTarget.resize() + GlRenderTargetAdaptor._resizeColor().
   */
  resize(width: number, height: number, resolution?: number): void {
    if (
      width === this.width &&
      height === this.height &&
      (resolution === undefined || resolution === this.resolution)
    ) return;

    this.width  = width;
    this.height = height;
    // resolution is readonly in the constructor but we allow it here via cast
    if (resolution !== undefined) {
      (this as { resolution: number }).resolution = resolution;
    }

    this.dirtyId++;
    this._attachedMipLevel = 0;
    this._attachedLayer    = 0;

    this._destroyAttachments();
    this._buildFBOs();
  }

  /** Free all GPU resources. */
  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._destroyAttachments();
    if (this.framebuffer && this.framebuffer !== this.resolveFramebuffer) {
      this.gl.deleteFramebuffer(this.framebuffer);
    }
    if (this.resolveFramebuffer) {
      this.gl.deleteFramebuffer(this.resolveFramebuffer);
    }
    this.framebuffer = null;
    this.resolveFramebuffer = null;
  }

  get destroyed(): boolean { return this._destroyed; }

  get pixelWidth(): number  { return Math.round(this.width  * this.resolution); }
  get pixelHeight(): number { return Math.round(this.height * this.resolution); }

  // ── Private helpers ──────────────────────────────────────────────────────

  private _destroyAttachments(): void {
    const gl = this.gl;
    for (const tex of this.textures) gl.deleteTexture(tex);
    this.textures = [];
    for (const rb of this.msaaRenderBuffers) gl.deleteRenderbuffer(rb);
    this.msaaRenderBuffers = [];
    if (this.depthStencilRenderBuffer) {
      gl.deleteRenderbuffer(this.depthStencilRenderBuffer);
      this.depthStencilRenderBuffer = null;
    }
    // framebuffer / resolveFramebuffer deleted in destroy() or rebuild
    if (this.framebuffer && this.framebuffer !== this.resolveFramebuffer) {
      gl.deleteFramebuffer(this.framebuffer);
      this.framebuffer = null;
    }
    if (this.resolveFramebuffer) {
      gl.deleteFramebuffer(this.resolveFramebuffer);
      this.resolveFramebuffer = null;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2. PixiRenderTargetPool — acquire / release for ping-pong bloom passes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Key used to bucket render targets in the pool.
 * Compact string: "WxH:hdr:depth:stencil:samples:attachments".
 */
function rtPoolKey(opts: PixiRenderTargetOptions): string {
  return [
    opts.width,
    opts.height,
    opts.hdr      ? 1 : 0,
    opts.depth    ? 1 : 0,
    opts.stencil  ? 1 : 0,
    opts.samples  ?? 0,
    opts.colorAttachments ?? 1,
  ].join(':');
}

/**
 * Render target pool — reuses GPU objects across bloom ping-pong passes.
 *
 * Mirrors upstream:
 *   rendering-utils.ts RTPool (WebGL2 direct)
 *   upstream TexturePool (PixiJS level)
 *   upstream GlRenderTargetAdaptor per-target lifecycle
 *
 * Usage:
 * ```ts
 * const pool = new PixiRenderTargetPool(gl);
 * const rtA  = pool.acquire({ width, height, hdr: true });
 * const rtB  = pool.acquire({ width, height, hdr: true });
 * // ... ping-pong rendering ...
 * pool.release(rtA);
 * pool.release(rtB);
 * pool.dispose();
 * ```
 */
export class PixiRenderTargetPool {
  private readonly gl: WebGL2RenderingContext;
  private _free   = new Map<string, PixiRenderTarget[]>();
  private _inUse  = new Set<PixiRenderTarget>();

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  /**
   * Acquire a render target.  Returns a reused target if one of the same
   * descriptor is available; otherwise creates a new one.
   */
  acquire(opts: PixiRenderTargetOptions): PixiRenderTarget {
    const key  = rtPoolKey(opts);
    const pool = this._free.get(key);
    if (pool?.length) {
      const rt = pool.pop()!;
      this._inUse.add(rt);
      return rt;
    }
    const rt = new PixiRenderTarget(this.gl, opts);
    this._inUse.add(rt);
    return rt;
  }

  /**
   * Release a render target back to the pool for reuse.
   * Noop if the render target is not in-use.
   */
  release(rt: PixiRenderTarget): void {
    if (!this._inUse.has(rt)) return;
    this._inUse.delete(rt);
    const key = rtPoolKey({
      width: rt.width,
      height: rt.height,
      hdr: rt.hdr,
      depth: rt.depth,
      stencil: rt.stencil,
      samples: rt.samples,
      colorAttachments: rt.colorAttachments,
    });
    let pool = this._free.get(key);
    if (!pool) { pool = []; this._free.set(key, pool); }
    pool.push(rt);
  }

  /** Release all in-use targets and destroy everything. */
  dispose(): void {
    for (const rt of [...this._inUse]) { this.release(rt); }
    for (const pool of this._free.values()) {
      for (const rt of pool) rt.destroy();
    }
    this._free.clear();
    this._inUse.clear();
  }

  get inUseCount(): number { return this._inUse.size; }
  get freeCount(): number {
    let n = 0;
    for (const pool of this._free.values()) n += pool.length;
    return n;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3. BloomFBOPass — multi-pass bloom (WebGL2 direct, mirrors nuke-pipeline)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options for the BloomFBOPass.
 */
export interface BloomFBOPassOptions {
  /** Luminance threshold [0..1] below which pixels are excluded.  Default 0.4. */
  threshold?: number;
  /** Kawase blur kernel — number of passes.  Default 4. */
  blurPasses?: number;
  /** Strength of the additive bloom composite.  Default 1.0. */
  bloomScale?: number;
  /** Global brightness multiplier.  Default 1.0. */
  brightness?: number;
  /**
   * Which filter from pixi-filters-registry to use as the PixiJS-level bloom
   * (used when this pass is coupled with the PixiJS renderer).
   * Default 'advanced-bloom'.
   */
  filterName?: FilterName;
}

// ── Bloom GLSL shaders ──────────────────────────────────────────────────────

const BLOOM_VERT = /* glsl */`#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  // Fullscreen triangle via gl_VertexID (no VBO needed)
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

/** Pass 1: luminance extract — pixels below threshold are zeroed out. */
const BLOOM_EXTRACT_FRAG = /* glsl */`#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float     uThreshold;
void main() {
  vec4 col = texture(uTexture, vUV);
  float lum = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
  fragColor = lum > uThreshold ? col : vec4(0.0, 0.0, 0.0, col.a);
}`;

/** Pass 2 & 3: Kawase blur (single-direction tap).
 *  Each call runs horizontal OR vertical depending on uDirection. */
const BLOOM_KAWASE_FRAG = /* glsl */`#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform vec2      uTexelSize;   // 1/resolution
uniform vec2      uDirection;   // (1,0) or (0,1)
uniform float     uKernel;      // blur radius in texels
void main() {
  vec4 sum = vec4(0.0);
  // 5-tap Kawase weights (σ ≈ kernel/2)
  sum += texture(uTexture, vUV - 2.0 * uKernel * uDirection * uTexelSize) * 0.06136;
  sum += texture(uTexture, vUV - 1.0 * uKernel * uDirection * uTexelSize) * 0.24477;
  sum += texture(uTexture, vUV)                                             * 0.38774;
  sum += texture(uTexture, vUV + 1.0 * uKernel * uDirection * uTexelSize) * 0.24477;
  sum += texture(uTexture, vUV + 2.0 * uKernel * uDirection * uTexelSize) * 0.06136;
  fragColor = sum;
}`;

/** Pass N+1: additive composite — scene + bloom scaled. */
const BLOOM_COMPOSITE_FRAG = /* glsl */`#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float     uBloomScale;
uniform float     uBrightness;
void main() {
  vec4 scene = texture(uScene, vUV);
  vec4 bloom = texture(uBloom, vUV);
  // Additive blend, then brightness scale
  vec3 result = scene.rgb + bloom.rgb * uBloomScale;
  fragColor   = vec4(result * uBrightness, scene.a);
}`;

// ── VAO for fullscreen triangle ─────────────────────────────────────────────

class FullscreenVAO {
  private readonly gl: WebGL2RenderingContext;
  private vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext) {
    this.gl  = gl;
    // Empty VAO — position computed from gl_VertexID in BLOOM_VERT
    this.vao = gl.createVertexArray()!;
  }

  draw(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  destroy(): void { this.gl.deleteVertexArray(this.vao); }
}

/**
 * BloomFBOPass — WebGL2 direct multi-pass bloom.
 *
 * Pass sequence (mirrors nuke-pipeline BloomPass + AdvancedBloomFilter.apply):
 *   1. Extract:   render scene → extractRT  (threshold cut)
 *   2. H-blur:    extractRT   → hBlurRT[0] (Kawase horizontal, N passes)
 *   3. V-blur:    hBlurRT     → vBlurRT    (Kawase vertical, N passes)
 *   4. Composite: scene + vBlurRT → output (additive, bloomScale, brightness)
 *
 * The filterName field links this pass to pixi-filters-registry so that
 * BloomFBOPipeline can create a matching PixiJS-level filter for the PixiJS
 * renderer side when needed.
 */
export class BloomFBOPass {
  /** Which pixi-filters-registry filter this pass corresponds to. */
  readonly filterName: FilterName = 'advanced-bloom';

  readonly name = 'bloom-fbo';
  enabled = true;

  private readonly gl: WebGL2RenderingContext;
  private readonly pool: PixiRenderTargetPool;
  private readonly quad: FullscreenVAO;

  // Programs compiled once
  private pgExtract:   WebGLProgram;
  private pgKawase:    WebGLProgram;
  private pgComposite: WebGLProgram;

  // Options
  threshold:  number;
  blurPasses: number;
  bloomScale: number;
  brightness: number;

  constructor(gl: WebGL2RenderingContext, opts: BloomFBOPassOptions = {}) {
    this.gl         = gl;
    this.pool       = new PixiRenderTargetPool(gl);
    this.quad       = new FullscreenVAO(gl);
    this.threshold  = opts.threshold  ?? 0.4;
    this.blurPasses = opts.blurPasses ?? 4;
    this.bloomScale = opts.bloomScale ?? 1.0;
    this.brightness = opts.brightness ?? 1.0;

    this.pgExtract   = createProgram(gl, BLOOM_VERT, BLOOM_EXTRACT_FRAG);
    this.pgKawase    = createProgram(gl, BLOOM_VERT, BLOOM_KAWASE_FRAG);
    this.pgComposite = createProgram(gl, BLOOM_VERT, BLOOM_COMPOSITE_FRAG);
  }

  /**
   * Execute the bloom pass.
   * @param inputRT  - The PixiRenderTarget holding the scene to bloom.
   * @param outputRT - Destination target (null = canvas).
   */
  render(inputRT: PixiRenderTarget, outputRT: PixiRenderTarget | null = null): void {
    if (!this.enabled) {
      // Passthrough: blit input to output unchanged
      this._blitPassthrough(inputRT, outputRT);
      return;
    }

    const gl = this.gl;
    const { width, height } = inputRT;
    const rtOpts: PixiRenderTargetOptions = { width, height, hdr: inputRT.hdr };

    // ── Pass 1: Luminance extract ──
    const extractRT = this.pool.acquire(rtOpts);
    extractRT.bind();
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.pgExtract);
    this._bindTexture(inputRT.texture, 0, 'uTexture', this.pgExtract);
    this._setUniform1f(this.pgExtract, 'uThreshold', this.threshold);
    this.quad.draw();
    extractRT.unbind();

    // ── Passes 2 & 3: Kawase blur (horizontal → vertical, N iterations) ──
    const texelW = 1.0 / width;
    const texelH = 1.0 / height;

    // Ping-pong between two RTs for blur iterations
    let src = extractRT;
    let dst = this.pool.acquire(rtOpts);

    for (let i = 0; i < this.blurPasses; i++) {
      const kernel = i + 1.0;   // grow kernel each pass

      // Horizontal
      dst.bind();
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.pgKawase);
      this._bindTexture(src.texture, 0, 'uTexture', this.pgKawase);
      this._setUniform2f(this.pgKawase, 'uTexelSize', texelW, texelH);
      this._setUniform2f(this.pgKawase, 'uDirection',  1.0, 0.0);
      this._setUniform1f(this.pgKawase, 'uKernel', kernel);
      this.quad.draw();
      dst.unbind();

      // Swap
      [src, dst] = [dst, src];

      // Vertical
      dst.bind();
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.pgKawase);
      this._bindTexture(src.texture, 0, 'uTexture', this.pgKawase);
      this._setUniform2f(this.pgKawase, 'uTexelSize', texelW, texelH);
      this._setUniform2f(this.pgKawase, 'uDirection',  0.0, 1.0);
      this._setUniform1f(this.pgKawase, 'uKernel', kernel);
      this.quad.draw();
      dst.unbind();

      // Swap
      [src, dst] = [dst, src];
    }
    // src now holds the blurred result

    // Release intermediate ping-pong target
    this.pool.release(dst === extractRT ? src : dst);

    // ── Pass 4: Composite (scene + bloom) ──
    if (outputRT) {
      outputRT.bind();
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.pgComposite);

    // scene → unit 0, bloom → unit 1
    this._bindTexture(inputRT.texture, 0, 'uScene',  this.pgComposite);
    this._bindTexture(src.texture,     1, 'uBloom',  this.pgComposite);
    this._setUniform1f(this.pgComposite, 'uBloomScale', this.bloomScale);
    this._setUniform1f(this.pgComposite, 'uBrightness', this.brightness);
    this.quad.draw();

    if (outputRT) outputRT.unbind();

    // Return all temps to the pool
    this.pool.release(extractRT);
    if (src !== extractRT) this.pool.release(src);
  }

  /** Free GPU resources. */
  destroy(): void {
    const gl = this.gl;
    gl.deleteProgram(this.pgExtract);
    gl.deleteProgram(this.pgKawase);
    gl.deleteProgram(this.pgComposite);
    this.quad.destroy();
    this.pool.dispose();
  }

  // ── Shader helpers ────────────────────────────────────────────────────────

  private _bindTexture(
    tex: WebGLTexture | null,
    unit: number,
    name: string,
    prog: WebGLProgram,
  ): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const loc = gl.getUniformLocation(prog, name);
    if (loc != null) gl.uniform1i(loc, unit);
  }

  private _setUniform1f(prog: WebGLProgram, name: string, v: number): void {
    const loc = this.gl.getUniformLocation(prog, name);
    if (loc != null) this.gl.uniform1f(loc, v);
  }

  private _setUniform2f(prog: WebGLProgram, name: string, x: number, y: number): void {
    const loc = this.gl.getUniformLocation(prog, name);
    if (loc != null) this.gl.uniform2f(loc, x, y);
  }

  /** Simple passthrough blit when bloom is disabled. */
  private _blitPassthrough(src: PixiRenderTarget, dst: PixiRenderTarget | null): void {
    const gl = this.gl;
    if (dst) {
      dst.bind();
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    // Using composite program with bloomScale=0 brightness=1 → pure scene passthrough
    gl.useProgram(this.pgComposite);
    this._bindTexture(src.texture, 0, 'uScene',  this.pgComposite);
    this._bindTexture(null,        1, 'uBloom',  this.pgComposite);
    this._setUniform1f(this.pgComposite, 'uBloomScale', 0.0);
    this._setUniform1f(this.pgComposite, 'uBrightness', 1.0);
    this.quad.draw();
    if (dst) dst.unbind();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4. BloomFBOPipeline — full pipeline bridging FBO pass ↔ pixi-filters-registry
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options for BloomFBOPipeline.
 */
export interface BloomFBOPipelineOptions {
  /** Canvas to get the WebGL2 context from (required for FBO path). */
  canvas?: HTMLCanvasElement;
  /** Provide an existing WebGL2 context (alternative to canvas). */
  gl?: WebGL2RenderingContext;
  /** FBO pass options forwarded to BloomFBOPass. */
  fboOptions?: BloomFBOPassOptions;
  /**
   * AdvancedBloomFilter options forwarded via pixi-filters-registry.
   * Only used when pixiBloomEnabled = true.
   */
  pixiOptions?: AdvancedBloomFilterOptions;
  /**
   * When true (default), also creates a pixi-filters-registry AdvancedBloomFilter
   * instance for use on PixiJS Containers via container.filters = [...].
   */
  pixiBloomEnabled?: boolean;
}

/**
 * BloomFBOPipeline — unified bloom system.
 *
 * Combines:
 *   - BloomFBOPass   (WebGL2 direct, manages PixiRenderTargetPool)
 *   - AdvancedBloomFilter (from pixi-filters-registry, for PixiJS Container-level bloom)
 *
 * The two sides share configuration (threshold, bloomScale, brightness) so
 * that the FBO path (post-process after PixiJS renders to a render texture)
 * and the PixiJS filter path produce consistent visual results.
 *
 * Usage (PixiJS + direct WebGL blend):
 * ```ts
 * const pipeline = new BloomFBOPipeline({
 *   canvas,
 *   fboOptions:  { threshold: 0.3, blurPasses: 4, bloomScale: 1.2 },
 *   pixiOptions: { threshold: 0.3, blur: 8, bloomScale: 1.2 },
 * });
 *
 * // Attach PixiJS bloom to a cell container:
 * cellContainer.filters = [pipeline.pixiBloomFilter];
 *
 * // After PixiJS renders to an offscreen PixiRenderTarget:
 * pipeline.render(offscreenRT, null);   // null → canvas
 *
 * // Cleanup:
 * pipeline.destroy();
 * ```
 */
export class BloomFBOPipeline {
  /** WebGL2 direct bloom pass (handles multi-FBO ping-pong). */
  readonly fboPass: BloomFBOPass;

  /**
   * PixiJS-level bloom filter created via pixi-filters-registry.
   * Attach to PixiJS Container.filters for container-level bloom.
   * Null when pixiBloomEnabled = false.
   */
  readonly pixiBloomFilter: AdvancedBloomFilter | null;

  private readonly _gl: WebGL2RenderingContext | null;

  constructor(opts: BloomFBOPipelineOptions = {}) {
    // ── WebGL2 context ──
    let gl: WebGL2RenderingContext | null = opts.gl ?? null;
    if (!gl && opts.canvas) {
      gl = opts.canvas.getContext('webgl2', {
        antialias: false,
        alpha: true,
        premultipliedAlpha: false,
      });
      if (!gl) console.warn('[BloomFBOPipeline] WebGL2 not available on canvas.');
    }
    this._gl = gl;

    // ── FBO pass ──
    if (gl) {
      this.fboPass = new BloomFBOPass(gl, opts.fboOptions ?? {});
    } else {
      // Headless / SSR: stub fboPass so the pipeline still type-checks.
      // render() is a no-op in this state.
      this.fboPass = null as unknown as BloomFBOPass;
    }

    // ── PixiJS filter (via pixi-filters-registry) ──
    if (opts.pixiBloomEnabled !== false) {
      this.pixiBloomFilter = createFilter('advanced-bloom', opts.pixiOptions ?? {}) as AdvancedBloomFilter;
    } else {
      this.pixiBloomFilter = null;
    }
  }

  /**
   * Synchronise shared parameters between the FBO pass and the PixiJS filter.
   * Call after changing threshold / bloomScale / brightness.
   */
  sync(): void {
    if (!this.fboPass || !this.pixiBloomFilter) return;
    this.pixiBloomFilter.bloomScale = this.fboPass.bloomScale;
    this.pixiBloomFilter.brightness = this.fboPass.brightness;
    this.pixiBloomFilter.threshold  = this.fboPass.threshold;
  }

  /**
   * Execute the FBO bloom pipeline.
   * @param inputRT  - Scene render target.
   * @param outputRT - Destination target (null = canvas).
   */
  render(inputRT: PixiRenderTarget, outputRT: PixiRenderTarget | null = null): void {
    if (!this.fboPass) return;
    this.fboPass.render(inputRT, outputRT);
  }

  /** Free all GPU resources. */
  destroy(): void {
    this.fboPass?.destroy();
    // pixiBloomFilter has no explicit destroy in upstream pixi-filters-registry
    // (PixiJS Filter is destroyed with the container or app).
  }

  // ── Convenience accessors (mirror fboPass properties) ───────────────────

  get threshold():  number { return this.fboPass?.threshold  ?? 0.4; }
  set threshold(v: number) {
    if (this.fboPass) this.fboPass.threshold = v;
    if (this.pixiBloomFilter) this.pixiBloomFilter.threshold = v;
  }

  get bloomScale(): number { return this.fboPass?.bloomScale ?? 1.0; }
  set bloomScale(v: number) {
    if (this.fboPass) this.fboPass.bloomScale = v;
    if (this.pixiBloomFilter) this.pixiBloomFilter.bloomScale = v;
  }

  get brightness(): number { return this.fboPass?.brightness ?? 1.0; }
  set brightness(v: number) {
    if (this.fboPass) this.fboPass.brightness = v;
    if (this.pixiBloomFilter) this.pixiBloomFilter.brightness = v;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5. Convenience factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a single offscreen PixiRenderTarget for use in the bloom pipeline.
 *
 * @example
 * ```ts
 * const offscreen = createOffscreenTarget(gl, 1280, 720, { hdr: true });
 * offscreen.bind();
 * // ... render scene ...
 * offscreen.unbind();
 * pipeline.render(offscreen, null);
 * ```
 */
export function createOffscreenTarget(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  overrides: Partial<PixiRenderTargetOptions> = {},
): PixiRenderTarget {
  return new PixiRenderTarget(gl, { width, height, ...overrides });
}

/**
 * Build a complete BloomFBOPipeline from a canvas element.
 * Convenience wrapper around `new BloomFBOPipeline`.
 *
 * @example
 * ```ts
 * const pipeline = buildBloomPipeline(canvas, { threshold: 0.35, bloomScale: 1.5 });
 * // Attach PixiJS filter:
 * stage.filters = [pipeline.pixiBloomFilter];
 * // Each frame after PixiJS renders to offscreenRT:
 * pipeline.render(offscreenRT, null);
 * ```
 */
export function buildBloomPipeline(
  canvas: HTMLCanvasElement,
  fboOptions: BloomFBOPassOptions = {},
  pixiOptions: AdvancedBloomFilterOptions = {},
): BloomFBOPipeline {
  return new BloomFBOPipeline({ canvas, fboOptions, pixiOptions });
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6. Re-exports (upstream RenderTarget types consumed by other modules)
// ─────────────────────────────────────────────────────────────────────────────

// Re-export the pixi-filters-registry factories so callers can create any
// filter directly without importing pixi-filters-registry separately.
export {
  createFilter,
  AdvancedBloomFilter,
  BloomFilter,
  KawaseBlurFilter,
};
export type { AdvancedBloomFilterOptions, FilterName };
