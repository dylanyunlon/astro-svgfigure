/**
 * pixi-gpu-bridge.ts — M965: PixiJS + raw WebGL GPU pass coexistence
 *
 * 问题根源
 * ────────
 * pixi-cell-renderer.ts 用 PixiJS Application (内部持有 WebGL context)
 * gpu-render-loop.ts    用原生 WebGL (直接 canvas.getContext('webgl'))
 *
 * 两者竞争同一个 WebGLRenderingContext 的绑定状态：
 *   • framebuffer bindings  (FBO)
 *   • texture unit bindings (TEXTURE0–TEXTURE_N)
 *   • blend/depth/stencil state flags
 *   • vertex array objects (VAO, WebGL2)
 *   • active shader program (useProgram)
 *
 * 同帧内任何一方在另一方 draw call 之间修改上述状态都会造成
 * 渲染撕裂 / 黑屏 / 错误纹理输出。
 *
 * 解决策略（双向桥接）
 * ────────────────────
 *
 * 方向 A — PixiJS → GPU pass  (extractPixiTexture)
 *   1. 让 PixiJS 先把整帧渲染到 RenderTexture (FBO-backed)
 *   2. 通过 renderer._gpuData[renderer.uid] 或 renderer.context.gl
 *      读取该纹理的原生 WebGLTexture handle
 *   3. GPU pass 以此纹理作为 composite 的一层（cellLayer）输入
 *
 * 方向 B — GPU pass → PixiJS  (injectGPUTexture)
 *   1. GPU pass 把结果写入自己的 FBO texture
 *   2. 此函数把该 WebGLTexture 包装成 PixiJS Texture / Sprite
 *   3. 把 Sprite 加入 PixiJS stage 即可在 PixiJS render pass 中合成
 *
 * GL 状态保护
 * ──────────
 * 所有函数在入口处用 saveGLState() 快照当前 GL binding，
 * 在出口处用 restoreGLState() 恢复，
 * 确保调用方（PixiJS 或 GPU pass）的状态不被污染。
 *
 * 调用时机
 * ────────
 *   帧序：
 *     1. pixiRenderer.render(stage) → RenderTexture
 *     2. extractPixiTexture(pixiRenderer, renderTexture)   ← 方向 A
 *     3. gpuRenderLoop.frame(dt)                           ← GPU pass 使用 pixiTex
 *     — 或 —
 *     1. gpuRenderLoop.frame(dt)                           ← GPU pass 写 FBO
 *     2. injectGPUTexture(gl, gpuTex, pixiRenderer)        ← 方向 B
 *     3. pixiRenderer.render(stage)                        ← PixiJS 合成 gpuSprite
 *
 * 上游引用
 * ────────
 *   upstream/pixijs-engine/src/rendering/renderers/gl/WebGLRenderer.ts
 *     → renderer.gl : WebGL2RenderingContext
 *   upstream/pixijs-engine/src/rendering/renderers/gl/texture/GlTextureSystem.ts
 *     → renderer.texture.getGlSource(source) → GlTexture { texture: WebGLTexture }
 *   upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture.ts
 *   upstream/pixijs-engine/src/scene/sprite/Sprite.ts
 *   src/lib/sph/gpu-render-loop.ts
 *   src/lib/renderers/pixi-cell-renderer.ts
 */

// ─── PixiJS upstream imports ─────────────────────────────────────────────────




// ─── Type aliases ─────────────────────────────────────────────────────────────


import { Texture }      from '../../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/Texture';
import { TextureSource } from '../../../upstream/pixijs-engine/src/rendering/renderers/shared/texture/sources/TextureSource';
import { Sprite }       from '../../../upstream/pixijs-engine/src/scene/sprite/Sprite';
import type { Container } from '../../../upstream/pixijs-engine/src/scene/container/Container';

<<<<<<< HEAD
// [orphan-precise] /**
// [orphan-precise]  * Minimal interface satisfied by both PixiJS WebGLRenderer and
// [orphan-precise]  * the Application.renderer property.
// [orphan-precise]  *
// [orphan-precise]  * We intentionally use 'any' for the upstream Renderer type here
// [orphan-precise]  * to keep this bridge decoupled from PixiJS internals.
// [orphan-precise]  * All real accesses go through optional chaining with clear error messages.
// [orphan-precise]  */
=======
/**
 * Minimal interface satisfied by both PixiJS WebGLRenderer and
 * the Application.renderer property.
 *
 * We intentionally use 'any' for the upstream Renderer type here
 * to keep this bridge decoupled from PixiJS internals.
 * All real accesses go through optional chaining with clear error messages.
 */




>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export interface PixiRendererLike {
  /** The native WebGL context that PixiJS allocated on the canvas. */
  gl: WebGL2RenderingContext | WebGLRenderingContext;
  /** PixiJS uid used as a key into per-resource GPU data maps. */
  uid: number;
  /** Texture management sub-system (GlTextureSystem). */
  texture: {
    getGlSource(source: any): { texture: WebGLTexture | null } | null;
  };
  /** Render-target system, used to bind/unbind FBOs. */
  renderTarget?: any;
  /** Canvas backing the renderer. */
  canvas: HTMLCanvasElement | OffscreenCanvas;
}

// ─── GL state snapshot ───────────────────────────────────────────────────────

/**
 * Snapshot of the GL binding state that both PixiJS and GPU passes touch.
 * Captured before, and restored after, each bridge operation.
 */
export interface GLStateSnapshot {
  framebuffer: WebGLFramebuffer | null;
  program:     WebGLProgram    | null;
  vao:         WebGLVertexArrayObject | null; // WebGL2 only
  activeTexture: number;
  textureBindings: Array<WebGLTexture | null>; // units 0–7
  viewport:    Int32Array;
  blendEnabled: boolean;
  depthTestEnabled: boolean;
  scissorEnabled: boolean;
}

/**
 * Capture the current GL binding state into a snapshot object.
 * Call before any bridge operation that changes GL state.
 */
export function saveGLState(gl: WebGL2RenderingContext | WebGLRenderingContext): GLStateSnapshot {
  const MAX_UNITS = 8; // we track units 0–7; sufficient for all current passes

  // Texture unit bindings
  const active = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
  const textureBindings: Array<WebGLTexture | null> = [];
  for (let i = 0; i < MAX_UNITS; i++) {
    gl.activeTexture(gl.TEXTURE0 + i);
    textureBindings.push(gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null);
  }
  gl.activeTexture(active); // restore active unit

  const snap: GLStateSnapshot = {
    framebuffer:    gl.getParameter(gl.FRAMEBUFFER_BINDING)    as WebGLFramebuffer | null,
    program:        gl.getParameter(gl.CURRENT_PROGRAM)        as WebGLProgram     | null,
    vao:            (gl as WebGL2RenderingContext).getParameter
                      ? (gl as WebGL2RenderingContext).getParameter(
                          (gl as WebGL2RenderingContext).VERTEX_ARRAY_BINDING,
                        ) as WebGLVertexArrayObject | null
                      : null,
    activeTexture:  active,
    textureBindings,
    viewport:       Int32Array.from(gl.getParameter(gl.VIEWPORT) as Int32Array),
    blendEnabled:   gl.getParameter(gl.BLEND)     as boolean,
    depthTestEnabled: gl.getParameter(gl.DEPTH_TEST) as boolean,
    scissorEnabled: gl.getParameter(gl.SCISSOR_TEST) as boolean,
  };

  return snap;
}

/**
 * Restore a previously captured GL state snapshot.
 * Call after any bridge operation to leave the context clean for the
 * next caller (PixiJS or GPU pass).
 */
export function restoreGLState(
  gl: WebGL2RenderingContext | WebGLRenderingContext,
  snap: GLStateSnapshot,
): void {
  // Framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, snap.framebuffer);

  // Shader program
  gl.useProgram(snap.program);

  // VAO (WebGL2 only)
  if ((gl as WebGL2RenderingContext).bindVertexArray) {
    (gl as WebGL2RenderingContext).bindVertexArray(snap.vao);
  }

  // Texture units
  for (let i = 0; i < snap.textureBindings.length; i++) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, snap.textureBindings[i]);
  }
  gl.activeTexture(snap.activeTexture);

  // Viewport
  gl.viewport(snap.viewport[0], snap.viewport[1], snap.viewport[2], snap.viewport[3]);

  // Boolean flags
  snap.blendEnabled      ? gl.enable(gl.BLEND)       : gl.disable(gl.BLEND);
  snap.depthTestEnabled  ? gl.enable(gl.DEPTH_TEST)  : gl.disable(gl.DEPTH_TEST);
  snap.scissorEnabled    ? gl.enable(gl.SCISSOR_TEST) : gl.disable(gl.SCISSOR_TEST);
}

// ─── Direction A: PixiJS → GPU pass ──────────────────────────────────────────

/**
 * Result returned by extractPixiTexture().
 */
export interface PixiTextureExtract {
  /** The native WebGL texture handle. Remains valid until PixiJS destroys the RenderTexture. */
  glTexture: WebGLTexture;
  /** Width in pixels of the extracted texture. */
  width: number;
  /** Height in pixels. */
  height: number;
  /**
   * The WebGL context that owns this texture.
   * MUST match the context used by GPURenderLoop.
   */
  gl: WebGL2RenderingContext | WebGLRenderingContext;
}

/**
 * extractPixiTexture — Direction A bridge helper.
 *
 * Extracts the native WebGLTexture handle from a PixiJS RenderTexture (or
 * any Texture whose source has already been uploaded to the GPU).
 *
 * Flow:
 *   1. Obtain the TextureSource from the PixiJS Texture
 *   2. Call renderer.texture.getGlSource(source) → GlTexture
 *   3. Return GlTexture.texture (the native WebGLTexture)
 *
 * The returned texture is owned by PixiJS; do NOT delete it.
 * It is valid as long as the originating Texture is alive.
 *
 * @param pixiRenderer  A PixiJS WebGLRenderer (app.renderer).
 * @param pixiTexture   A PixiJS Texture or RenderTexture that has been
 *                      rendered into this frame.
 * @returns             The native handle + metadata, or null on failure.
 *
 * @example
 * ```ts
 * // 1. Render PixiJS cells into an off-screen RenderTexture
 * const rt = RenderTexture.create({ width: 800, height: 600 });
 * pixiRenderer.render({ container: stage, target: rt });
 *
 * // 2. Extract the native texture handle
 * const extracted = extractPixiTexture(pixiRenderer, rt);
 * if (extracted) {
 *   // 3. Pass to GPU composite pass as a layer
 *   composite.render({ ..., cellTexture: extracted.glTexture });
 * }
 * ```
 */
export function extractPixiTexture(
  pixiRenderer: PixiRendererLike,
  pixiTexture: Texture,
): PixiTextureExtract | null {
  if (!pixiRenderer || !pixiTexture) {
    console.warn('[pixi-gpu-bridge] extractPixiTexture: missing renderer or texture');
    return null;
  }

  const source = pixiTexture.source;
  if (!source) {
    console.warn('[pixi-gpu-bridge] extractPixiTexture: texture has no source');
    return null;
  }

  // GlTextureSystem.getGlSource() returns { texture: WebGLTexture, ... }
  // The texture must have been uploaded before this call.
  let glSource: { texture: WebGLTexture | null } | null = null;
  try {
    glSource = pixiRenderer.texture.getGlSource(source);
  } catch (err) {
    console.warn('[pixi-gpu-bridge] extractPixiTexture: getGlSource failed:', err);
    return null;
  }

  if (!glSource?.texture) {
    // Texture hasn't been uploaded to GPU yet — try triggering upload
    // by checking _gpuData directly (PixiJS stores GlTexture per renderer uid)
    const gpuData = (source as any)._gpuData?.[pixiRenderer.uid];
    if (!gpuData?.texture) {
      console.warn(
        '[pixi-gpu-bridge] extractPixiTexture: texture not on GPU yet.',
        'Ensure renderer.render() has been called before extracting.',
      );
      return null;
    }
    glSource = gpuData;
  }

  return {
    glTexture: glSource.texture as WebGLTexture,
    width:     source.width  ?? pixiTexture.width,
    height:    source.height ?? pixiTexture.height,
    gl:        pixiRenderer.gl,
  };
}

/**
 * extractPixiFrameTexture — convenience wrapper for the common case where
 * you want to capture the current PixiJS frame as a texture without
 * creating a RenderTexture upfront.
 *
 * Uses gl.readPixels under the hood (slow — suitable for occasional use,
 * not per-frame tight loops).  For per-frame use, prefer rendering PixiJS
 * into a RenderTexture and calling extractPixiTexture().
 *
 * @param pixiRenderer  The PixiJS WebGLRenderer.
 * @param width         Canvas width  (defaults to canvas.width).
 * @param height        Canvas height (defaults to canvas.height).
 * @returns             A new WebGLTexture owned by the caller (caller must delete).
 */
export function extractPixiFrameTexture(
  pixiRenderer: PixiRendererLike,
  width?: number,
  height?: number,
): WebGLTexture | null {
  const gl  = pixiRenderer.gl;
  const w   = width  ?? (pixiRenderer.canvas as HTMLCanvasElement).width;
  const h   = height ?? (pixiRenderer.canvas as HTMLCanvasElement).height;

  const snap = saveGLState(gl);

  try {
    // Read current front-buffer pixels (canvas backbuffer after render)
    // PixiJS renders to the default framebuffer (null), so we read from there.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Upload into a new WebGLTexture (owned by caller)
    const tex = gl.createTexture();
    if (!tex) {
      console.warn('[pixi-gpu-bridge] extractPixiFrameTexture: gl.createTexture() failed');
      return null;
    }

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return tex;
  } finally {
    restoreGLState(gl, snap);
  }
}

// ─── Direction B: GPU pass → PixiJS ──────────────────────────────────────────

/**
 * Result returned by injectGPUTexture().
 */
export interface GPUTextureInjection {
  /**
   * A PixiJS Sprite backed by the GPU FBO texture.
   * Add to app.stage (or any Container) to composite the GPU output.
   *
   * The Sprite owns a TextureSource that wraps the raw WebGLTexture without
   * copying pixel data — GPU writes to the FBO texture are immediately
   * visible in the next PixiJS render pass.
   */
  sprite: Sprite;
  /**
   * The PixiJS Texture wrapping the GPU output.
   * Can be used directly on other display objects (TilingSprite, etc.).
   */
  texture: Texture;
  /**
   * Call this to release the TextureSource / Sprite when they are no
   * longer needed (does NOT delete the underlying gpuTexture — that
   * remains owned by the GPU pass).
   */
  destroy(): void;
}

/**
 * injectGPUTexture — Direction B bridge helper.
 *
 * Wraps a raw WebGLTexture produced by a GPU pass FBO into a PixiJS Sprite
 * so that it can be composited inside the PixiJS scene graph.
 *
 * Flow:
 *   1. Create a TextureSource that references the raw WebGLTexture via
 *      the `_gpuData` slot that PixiJS uses internally.
 *   2. Create a PixiJS Texture from that source.
 *   3. Wrap in a Sprite sized to (width × height).
 *   4. Register the raw WebGLTexture in the renderer's texture cache so
 *      PixiJS won't try to re-upload it.
 *
 * The Sprite must be added to the stage AFTER this call and BEFORE the
 * next pixiRenderer.render() call.
 *
 * @param gl           The shared WebGLRenderingContext / WebGL2RenderingContext.
 * @param gpuTexture   The WebGLTexture produced by the GPU pass FBO.
 * @param width        Texture width in pixels.
 * @param height       Texture height in pixels.
 * @param pixiRenderer The PixiJS renderer that will consume this texture.
 * @returns            An injection result or null on failure.
 *
 * @example
 * ```ts
 * // GPU pass produces gpuFluidTex at canvas resolution
 * gpuRenderLoop.frame(dt);
 *
 * // Wrap for PixiJS
 * const injection = injectGPUTexture(
 *   gl, gpuRenderLoop.fluidOutputTexture, 800, 600, pixiRenderer,
 * );
 * if (injection) {
 *   injection.sprite.alpha = 0.6;         // blend
 *   injection.sprite.blendMode = 'add';   // additive composite
 *   app.stage.addChild(injection.sprite); // display over PixiJS cells
 * }
 * ```
 */
export function injectGPUTexture(
  gl: WebGL2RenderingContext | WebGLRenderingContext,
  gpuTexture: WebGLTexture,
  width: number,
  height: number,
  pixiRenderer: PixiRendererLike,
): GPUTextureInjection | null {
  if (!gpuTexture) {
    console.warn('[pixi-gpu-bridge] injectGPUTexture: gpuTexture is null/undefined');
    return null;
  }
  if (!pixiRenderer) {
    console.warn('[pixi-gpu-bridge] injectGPUTexture: pixiRenderer is null/undefined');
    return null;
  }

  const snap = saveGLState(gl);

  try {
    // ── Create a PixiJS TextureSource that references the raw WebGLTexture ──
    //
    // TextureSource is the low-level GPU data container in PixiJS v8.
    // By setting _gpuData[renderer.uid] = { texture: gpuTexture, ... } we
    // tell PixiJS that this source is already on the GPU and doesn't need
    // uploading.  This is the same slot that GlTextureSystem.getGlSource()
    // reads from.
    //
    // We set uploadMethodId to a sentinel that causes PixiJS to skip the
    // normal upload path (it checks !source.resource || source._gpuData first).

    const source = new TextureSource({
      width,
      height,
      format: 'rgba8unorm', // matches gl.RGBA / gl.UNSIGNED_BYTE
      antialias: false,
      autoGenerateMipmaps: false,
    });

    // Stamp the pre-existing GL texture into PixiJS's _gpuData slot.
    // Shape matches GlTexture from upstream GlTextureSystem:
    //   { texture, target, format, internalFormat, type, ... }
    const glTextureWrapper = {
      texture:        gpuTexture,
      target:         gl.TEXTURE_2D,
      format:         gl.RGBA,
      internalFormat: gl.RGBA,
      type:           gl.UNSIGNED_BYTE,
      samplerType:    0,           // FLOAT_SAMPLER
      dirtyId:        -1,          // prevents PixiJS from re-uploading
    };

    // PixiJS stores GPU data per renderer uid to support multi-renderer setups
    if (!(source as any)._gpuData) {
      (source as any)._gpuData = {};
    }
    (source as any)._gpuData[pixiRenderer.uid] = glTextureWrapper;

    // Mark as uploaded so PixiJS skips the upload path entirely
    (source as any)._gpuData.dirty = false;

    // Override dimensions to match the actual GPU texture
    (source as any)._width  = width;
    (source as any)._height = height;
    source.width  = width;
    source.height = height;

    // ── Create a PixiJS Texture from the source ───────────────────────────
    const texture = new Texture({ source });

    // ── Create a Sprite sized to the full GPU texture ─────────────────────
    const sprite = new Sprite(texture);
    sprite.width  = width;
    sprite.height = height;

    // ── Cleanup handle ────────────────────────────────────────────────────
    const destroy = () => {
      sprite.destroy({ texture: true, textureSource: true });
      // Note: we do NOT delete gpuTexture — it is owned by the GPU pass.
    };

    return { sprite, texture, destroy };
  } catch (err) {
    console.error('[pixi-gpu-bridge] injectGPUTexture failed:', err);
    return null;
  } finally {
    restoreGLState(gl, snap);
  }
}

// ─── Shared context guard ─────────────────────────────────────────────────────

/**
 * PixiGPUBridge — stateful bridge that manages the shared WebGL context
 * between a PixiJS renderer and a raw GPU render loop.
 *
 * Encapsulates the full coexistence lifecycle:
 *   • Context sharing guard (same canvas, same WebGL context)
 *   • Per-frame state save/restore around each subsystem's draw calls
 *   • Texture handoff in both directions
 *   • Injection Sprite lifecycle management
 *
 * Usage (PixiJS-first, GPU as overlay):
 * ```ts
 * const bridge = new PixiGPUBridge(app.renderer, gpuLoop);
 *
 * // Each frame:
 * bridge.beginPixiPass();           // save GPU state
 * pixiRenderer.render(stage, { target: offscreenRT });
 * bridge.endPixiPass();             // restore GPU state
 *
 * const pixi = bridge.extractPixiOutput(offscreenRT);
 * if (pixi) {
 *   gpuLoop.setExternalCellTexture(pixi.glTexture);
 * }
 *
 * bridge.beginGPUPass();            // save Pixi state
 * gpuLoop.frame(dt);
 * bridge.endGPUPass();              // restore Pixi state
 * ```
 *
 * Usage (GPU-first, PixiJS as overlay):
 * ```ts
 * bridge.beginGPUPass();
 * gpuLoop.frame(dt);
 * bridge.endGPUPass();
 *
 * const injection = bridge.injectGPUOutput(gpuLoop.outputTexture, w, h);
 * app.stage.addChildAt(injection.sprite, 0); // behind PixiJS cells
 *
 * bridge.beginPixiPass();
 * pixiRenderer.render(stage);
 * bridge.endPixiPass();
 * ```
 */
export class PixiGPUBridge {
  private _pixiRenderer: PixiRendererLike;
  private _gl: WebGL2RenderingContext | WebGLRenderingContext;

  /** Snapshot taken in beginGPUPass(), restored in endGPUPass(). */
  private _gpuSnapBefore: GLStateSnapshot | null = null;
  /** Snapshot taken in beginPixiPass(), restored in endPixiPass(). */
  private _pixiSnapBefore: GLStateSnapshot | null = null;

  /** Active injection sprites managed by this bridge. */
  private _injections: GPUTextureInjection[] = [];

  constructor(
    pixiRenderer: PixiRendererLike,
    /** Optional: a different WebGL context (e.g. from GPURenderLoop). */
    sharedGL?: WebGL2RenderingContext | WebGLRenderingContext,
  ) {
    this._pixiRenderer = pixiRenderer;
    this._gl = sharedGL ?? pixiRenderer.gl;

    // Verify context compatibility: both sides must use the same GL context.
    // If PixiJS and the GPU loop were initialised on different canvases they
    // share the context; different canvases = incompatible, warn loudly.
    if (sharedGL && sharedGL !== pixiRenderer.gl) {
      const pixiCanvas = pixiRenderer.canvas as HTMLCanvasElement;
      const gpuCanvas  = (sharedGL as any).canvas as HTMLCanvasElement | undefined;
      if (pixiCanvas && gpuCanvas && pixiCanvas !== gpuCanvas) {
        console.warn(
          '[PixiGPUBridge] PixiJS and GPU loop are using DIFFERENT canvases/contexts!',
          'Texture sharing between them will require explicit pixel readback (slow).',
          'For zero-copy sharing, pass the same canvas to both App.init() and GPURenderLoop.',
        );
      }
    }
  }

  // ── State save/restore helpers ───────────────────────────────────────────

  /**
   * Call immediately before a PixiJS renderer.render() call.
   * Snapshots the current GL state so the GPU pass can be restored afterward.
   */
  beginPixiPass(): void {
    this._pixiSnapBefore = saveGLState(this._gl);
  }

  /**
   * Call immediately after a PixiJS renderer.render() call.
   * Restores GL state to what it was before PixiJS ran.
   */
  endPixiPass(): void {
    if (this._pixiSnapBefore) {
      restoreGLState(this._gl, this._pixiSnapBefore);
      this._pixiSnapBefore = null;
    }
  }

  /**
   * Call immediately before a GPU pass frame() call.
   * Snapshots PixiJS GL state.
   */
  beginGPUPass(): void {
    this._gpuSnapBefore = saveGLState(this._gl);
  }

  /**
   * Call immediately after a GPU pass frame() call.
   * Restores GL state to what it was before the GPU pass ran.
   */
  endGPUPass(): void {
    if (this._gpuSnapBefore) {
      restoreGLState(this._gl, this._gpuSnapBefore);
      this._gpuSnapBefore = null;
    }
  }

  // ── Direction A: PixiJS → GPU ────────────────────────────────────────────

  /**
   * Extract the native WebGLTexture from a PixiJS Texture / RenderTexture
   * for use in the GPU pass.
   *
   * @param pixiTexture  A PixiJS Texture that has been rendered this frame.
   */
  extractPixiOutput(pixiTexture: Texture): PixiTextureExtract | null {
    return extractPixiTexture(this._pixiRenderer, pixiTexture);
  }

  /**
   * Read the current PixiJS canvas backbuffer as a WebGLTexture.
   * SLOW — uses gl.readPixels.  Use only for one-shot operations.
   *
   * @param width  Override canvas width.
   * @param height Override canvas height.
   * @returns      A new WebGLTexture owned by the caller.
   */
  readPixiFrame(width?: number, height?: number): WebGLTexture | null {
    return extractPixiFrameTexture(this._pixiRenderer, width, height);
  }

  // ── Direction B: GPU → PixiJS ────────────────────────────────────────────

  /**
   * Inject a GPU FBO output texture into the PixiJS scene graph.
   *
   * The returned Sprite must be added to app.stage before the next
   * pixiRenderer.render() call.
   *
   * @param gpuTexture  Raw WebGLTexture from a GPU pass FBO.
   * @param width       Texture width.
   * @param height      Texture height.
   */
  injectGPUOutput(
    gpuTexture: WebGLTexture,
    width: number,
    height: number,
  ): GPUTextureInjection | null {
    const injection = injectGPUTexture(
      this._gl,
      gpuTexture,
      width,
      height,
      this._pixiRenderer,
    );
    if (injection) this._injections.push(injection);
    return injection;
  }

  /**
   * Release all injection Sprites that were created by this bridge.
   * Call when switching scenes or tearing down the bridge.
   */
  releaseInjections(): void {
    for (const inj of this._injections) {
      try { inj.destroy(); } catch (_) { /* ignore */ }
    }
    this._injections = [];
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /**
   * Destroy the bridge and release all managed resources.
   * Does NOT destroy the PixiJS renderer or the GPU loop.
   */
  destroy(): void {
    this.releaseInjections();
    this._gpuSnapBefore  = null;
    this._pixiSnapBefore = null;
  }
}

// ─── Utility: verify shared context ──────────────────────────────────────────

/**
 * assertSharedContext — development helper.
 *
 * Verifies that PixiJS and the GPU loop are sharing the same WebGL context.
 * Logs a detailed warning if they are not.
 *
 * @param pixiRenderer  The PixiJS renderer.
 * @param gpuGL         The WebGL context used by the GPU loop.
 */
export function assertSharedContext(
  pixiRenderer: PixiRendererLike,
  gpuGL: WebGL2RenderingContext | WebGLRenderingContext,
): boolean {
  const pixiGL = pixiRenderer.gl;
  if (pixiGL === gpuGL) return true;

  console.error(
    '[pixi-gpu-bridge] Context mismatch detected!\n',
    '  PixiJS gl:', pixiGL, '\n',
    '  GPU loop gl:', gpuGL, '\n',
    'Both must use the SAME canvas.getContext() result.\n',
    'Fix: pass the same canvas to Application.init() and GPURenderLoop constructor,\n',
    '     OR set GPURenderLoop to use pixiRenderer.gl directly.',
  );
  return false;
}

/**
 * getPixiGL — safely extract the WebGL context from a PixiJS Application or renderer.
 *
 * Handles both `app.renderer.gl` (PixiJS v8 WebGLRenderer) and
 * `app.renderer.context.gl` (some older v8 beta versions).
 */
export function getPixiGL(
  pixiRendererOrApp: any,
): WebGL2RenderingContext | WebGLRenderingContext | null {
  // app.renderer.gl (standard PixiJS v8 WebGLRenderer)
  if (pixiRendererOrApp?.gl instanceof Object) {
    return pixiRendererOrApp.gl as WebGL2RenderingContext;
  }
  // app.renderer (Application.renderer) → nested .gl
  if (pixiRendererOrApp?.renderer?.gl instanceof Object) {
    return pixiRendererOrApp.renderer.gl as WebGL2RenderingContext;
  }
  // app.renderer.context.gl (some builds)
  if (pixiRendererOrApp?.context?.gl instanceof Object) {
    return pixiRendererOrApp.context.gl as WebGL2RenderingContext;
  }
  console.warn('[pixi-gpu-bridge] getPixiGL: could not locate WebGL context on renderer');
  return null;
}

// ─── Factory convenience ──────────────────────────────────────────────────────

/**
 * createPixiGPUBridge — convenience factory.
 *
 * Validates context sharing, creates a PixiGPUBridge, and returns it.
 *
 * @param pixiRenderer  The PixiJS WebGLRenderer (app.renderer).
 * @param gpuGL         Optional: the raw WebGL context from the GPU loop.
 *                      Defaults to pixiRenderer.gl.
 *                      Pass it to surface early context-mismatch warnings.
 */
export function createPixiGPUBridge(
  pixiRenderer: PixiRendererLike,
  gpuGL?: WebGL2RenderingContext | WebGLRenderingContext,
): PixiGPUBridge {
  const resolvedGL = gpuGL ?? pixiRenderer.gl;

  if (gpuGL) {
    assertSharedContext(pixiRenderer, gpuGL);
  }

  return new PixiGPUBridge(pixiRenderer, resolvedGL);
}
