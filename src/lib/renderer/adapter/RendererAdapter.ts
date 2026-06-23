/**
 * RendererAdapter.ts — Renderer ↔ Program ↔ Material 统一适配层
 *
 * 解决的架构问题:
 *   AstroRenderer  管 GL context / extensions / viewport / UBO
 *   AstroProgram   管 shader compile / uniform cache / texture unit
 *   Material       管 blending / depth / side / uniform map
 *
 * 三者各自完备，但 pipeline 消费端（AstroPipeline, CellEventSource,
 * FXSceneCompositor）需要跨越三层做同一件事时非常繁琐。
 *
 * RendererAdapter 提供:
 *   1. ProgramHandle — 从 Material 自动编译 AstroProgram + 双向 uniform 同步
 *   2. DrawContext   — 一帧内的 view/projection/time 等全局 uniform 统一注入
 *   3. MaterialSlot  — 材质热切换，切换时自动 rebind program + upload uniforms
 *   4. RenderPassScope — 自动管理 FBO bind/unbind + viewport restore
 *
 * 设计原则:
 *   - 零拷贝: adapter 不复制 renderer/program/material, 持有引用
 *   - 透明降级: WebGL1 路径与 WebGL2 共用同一 API surface
 *   - cell-pubsub-loop 友好: 每次 cell_update 事件只需
 *     adapter.useMaterial(cellMat) → adapter.draw()
 *
 * Usage:
 *   const adapter = new RendererAdapter(renderer);
 *   const handle  = adapter.createProgram(material);
 *   adapter.beginFrame({ view, projection, time });
 *   adapter.useMaterial(material);
 *   adapter.draw(mesh);
 *   adapter.endFrame();
 */

import { AstroRenderer, WEBGL2 } from '../AstroRenderer.js';
import { AstroProgram } from '../AstroProgram.js';
import { Material } from '../material/Material.js';
import type { BlendingMode, SideMode, UniformValue } from '../material/Material.js';
import type { UniformBuffer } from '../UniformBuffer.js';
import type { Viewport } from '../AstroRenderer.js';

// ── DrawContext — per-frame global uniforms ──────────────────────────────────

export interface DrawContext {
  /** Column-major 4×4 view matrix */
  view?: Float32Array;
  /** Column-major 4×4 projection matrix */
  projection?: Float32Array;
  /** Elapsed time in seconds (for animation uniforms) */
  time?: number;
  /** Canvas resolution [width, height] */
  resolution?: [number, number];
  /** Optional UBOs to bind globally for all programs this frame */
  ubos?: UniformBuffer[];
}

// ── ProgramHandle — compiled program + material binding ─────────────────────

/**
 * A compiled AstroProgram bound to a specific Material.
 * Created via RendererAdapter.createProgram().
 *
 * The handle caches the program and tracks whether the material's
 * shader source has changed (for hot-reload / species switch).
 */
export interface ProgramHandle {
  /** The compiled AstroProgram */
  readonly program: AstroProgram;
  /** The material this program was compiled from */
  readonly material: Material;
  /** Fingerprint of the shader sources used to compile — stale check */
  readonly sourceHash: string;
  /** Dispose the program GPU resources */
  dispose(): void;
}

// ── MaterialSlot — active material state ────────────────────────────────────

interface MaterialSlot {
  material: Material;
  handle: ProgramHandle;
  /** Last frame index this slot was bound — avoids redundant state changes */
  lastBoundFrame: number;
}

// ── RenderPassScope ─────────────────────────────────────────────────────────

/**
 * RAII-style render pass scope.
 * Created by adapter.beginPass(), ended by scope.end().
 * Automatically restores viewport and FBO binding.
 */
export class RenderPassScope {
  private _adapter: RendererAdapter;
  private _prevViewport: Viewport;
  private _fbo: WebGLFramebuffer | null;
  private _ended = false;

  /** @internal — use RendererAdapter.beginPass() */
  constructor(
    adapter: RendererAdapter,
    gl: WebGL2RenderingContext,
    fbo: WebGLFramebuffer | null,
    viewport: Viewport,
  ) {
    this._adapter = adapter;
    this._prevViewport = { ...adapter.renderer.viewport };
    this._fbo = fbo;

    // Bind FBO + set viewport
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    adapter.renderer.setViewport(viewport.x, viewport.y, viewport.width, viewport.height);
  }

  /** End the pass — restore previous FBO and viewport. */
  end(): void {
    if (this._ended) return;
    this._ended = true;

    const gl = this._adapter.renderer.context as WebGL2RenderingContext;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const v = this._prevViewport;
    this._adapter.renderer.setViewport(v.x, v.y, v.width, v.height);
  }
}

// ── RendererAdapter ─────────────────────────────────────────────────────────

export class RendererAdapter {
  readonly renderer: AstroRenderer;

  /** program cache: sourceHash → ProgramHandle */
  private _programs = new Map<string, ProgramHandle>();

  /** Currently active material slot */
  private _activeSlot: MaterialSlot | null = null;

  /** Frame counter — incremented by beginFrame() */
  private _frameIndex = 0;

  /** Per-frame draw context (view/projection/time/resolution) */
  private _ctx: DrawContext = {};

  constructor(renderer: AstroRenderer) {
    this.renderer = renderer;
  }

  // ── Program management ────────────────────────────────────────────────────

  /**
   * Compile (or retrieve cached) an AstroProgram for the given Material.
   *
   * The Material must expose vertex + fragment source via its uniforms map
   * under the reserved keys `__vertSrc` and `__fragSrc`, OR you can pass
   * them explicitly.
   *
   * Returns a ProgramHandle that can be disposed independently.
   */
  createProgram(
    material: Material,
    vertSrc?: string,
    fragSrc?: string,
  ): ProgramHandle {
    const vert = vertSrc ?? (material.getUniform('__vertSrc') as string | undefined);
    const frag = fragSrc ?? (material.getUniform('__fragSrc') as string | undefined);

    if (!vert || !frag) {
      throw new Error(
        `[RendererAdapter] createProgram: material "${material.name}" has no shader source. ` +
        `Pass vertSrc/fragSrc or set __vertSrc/__fragSrc uniforms.`,
      );
    }

    const hash = _hashSources(vert, frag);

    // Return cached if sources haven't changed
    const cached = this._programs.get(hash);
    if (cached) return cached;

    const program = new AstroProgram(this.renderer, vert, frag);

    const handle: ProgramHandle = {
      program,
      material,
      sourceHash: hash,
      dispose: () => {
        program.dispose();
        this._programs.delete(hash);
      },
    };

    this._programs.set(hash, handle);
    material.program = program;

    return handle;
  }

  /**
   * Get or create a ProgramHandle for a material, with hot-reload support.
   * If the material's shader sources changed since last call, recompiles.
   */
  ensureProgram(
    material: Material,
    vertSrc: string,
    fragSrc: string,
  ): ProgramHandle {
    const hash = _hashSources(vertSrc, fragSrc);
    const existing = this._programs.get(hash);
    if (existing && existing.material === material) return existing;

    // Sources changed — dispose old program if material had one
    if (material.program) {
      const oldHash = _findHashForProgram(this._programs, material.program);
      if (oldHash) {
        this._programs.get(oldHash)?.dispose();
      }
    }

    return this.createProgram(material, vertSrc, fragSrc);
  }

  // ── Frame lifecycle ────────────────────────────────────────────────────────

  /**
   * Begin a new frame. Sets up the DrawContext (view/projection/time)
   * that will be injected into every material bound this frame.
   */
  beginFrame(ctx: DrawContext = {}): void {
    this._frameIndex++;
    this._ctx = ctx;
    this._activeSlot = null;
  }

  /**
   * End the current frame. Resets active material state.
   */
  endFrame(): void {
    this._activeSlot = null;
    this._ctx = {};
  }

  /** Current frame index (monotonically increasing). */
  get frameIndex(): number { return this._frameIndex; }

  // ── Material binding ──────────────────────────────────────────────────────

  /**
   * Activate a material for subsequent draw calls.
   *
   * This is the core adapter operation:
   *   1. Look up (or create) the ProgramHandle for this material
   *   2. Call material.bind(gl) → sets blending/depth/side + uploads uniforms
   *   3. Inject DrawContext globals (view/projection/time/resolution)
   *   4. Bind any frame-level UBOs
   *
   * Skips redundant state changes if the same material was already active
   * this frame.
   */
  useMaterial(
    material: Material,
    vertSrc?: string,
    fragSrc?: string,
  ): void {
    const gl = this.renderer.context as WebGL2RenderingContext;

    // Ensure program exists
    if (!material.program) {
      if (!vertSrc || !fragSrc) {
        throw new Error(
          `[RendererAdapter] useMaterial: material "${material.name}" has no program. ` +
          `Call createProgram() first or pass vertSrc/fragSrc.`,
        );
      }
      this.createProgram(material, vertSrc, fragSrc);
    }

    // Redundancy check — skip if same material already bound this frame
    if (
      this._activeSlot &&
      this._activeSlot.material === material &&
      this._activeSlot.lastBoundFrame === this._frameIndex
    ) {
      return;
    }

    // Bind material (program.use + blending + depth + side + uniforms)
    material.bind(gl);

    // Inject frame-level globals
    this._injectDrawContext(material.program!);

    // Bind frame-level UBOs
    if (this._ctx.ubos) {
      for (const ubo of this._ctx.ubos) {
        material.program!.bindUniformBlock(ubo);
      }
    }

    // Track active slot
    const handle = this._findOrCreateHandle(material);
    this._activeSlot = {
      material,
      handle,
      lastBoundFrame: this._frameIndex,
    };
  }

  /**
   * Switch active material's species-driven shader (for CellMaterial hot-swap).
   * Recompiles the program if the fragment source changed.
   */
  switchShader(
    material: Material,
    vertSrc: string,
    fragSrc: string,
  ): ProgramHandle {
    const handle = this.ensureProgram(material, vertSrc, fragSrc);
    // Re-bind with new program
    this.useMaterial(material);
    return handle;
  }

  // ── Render pass scope ─────────────────────────────────────────────────────

  /**
   * Begin a scoped render pass — binds an FBO and viewport.
   * Call scope.end() when done to restore previous state.
   *
   * Usage:
   *   const pass = adapter.beginPass(fbo, { x: 0, y: 0, width: w, height: h });
   *   adapter.useMaterial(mat);
   *   adapter.drawArrays(gl.TRIANGLES, 0, 6);
   *   pass.end();
   */
  beginPass(
    fbo: WebGLFramebuffer | null,
    viewport: Viewport,
  ): RenderPassScope {
    return new RenderPassScope(
      this,
      this.renderer.context as WebGL2RenderingContext,
      fbo,
      viewport,
    );
  }

  // ── Draw helpers ──────────────────────────────────────────────────────────

  /**
   * Convenience: drawArrays with the currently bound material's program.
   */
  drawArrays(mode: number, first: number, count: number): void {
    this.renderer.drawArrays(mode, first, count);
  }

  /**
   * Convenience: drawElements with the currently bound material's program.
   */
  drawElements(mode: number, count: number, type: number, offset: number): void {
    this.renderer.drawElements(mode, count, type, offset);
  }

  /**
   * Convenience: instanced draw (WebGL2 only).
   */
  drawArraysInstanced(mode: number, first: number, count: number, instanceCount: number): void {
    const gl = this.renderer.context;
    if (this.renderer.version === WEBGL2) {
      (gl as WebGL2RenderingContext).drawArraysInstanced(mode, first, count, instanceCount);
    } else {
      const ext = this.renderer.extensions.instancing;
      if (ext) {
        ext.drawArraysInstancedANGLE(mode, first, count, instanceCount);
      }
    }
  }

  /**
   * Convenience: instanced indexed draw (WebGL2 only).
   */
  drawElementsInstanced(
    mode: number,
    count: number,
    type: number,
    offset: number,
    instanceCount: number,
  ): void {
    const gl = this.renderer.context;
    if (this.renderer.version === WEBGL2) {
      (gl as WebGL2RenderingContext).drawElementsInstanced(mode, count, type, offset, instanceCount);
    } else {
      const ext = this.renderer.extensions.instancing;
      if (ext) {
        ext.drawElementsInstancedANGLE(mode, count, type, offset, instanceCount);
      }
    }
  }

  // ── State queries ─────────────────────────────────────────────────────────

  /** The currently active material, or null if none. */
  get activeMaterial(): Material | null {
    return this._activeSlot?.material ?? null;
  }

  /** Number of cached compiled programs. */
  get programCount(): number {
    return this._programs.size;
  }

  /** Debug stats. */
  stats(): { programs: number; frameIndex: number; activeMaterial: string | null } {
    return {
      programs: this._programs.size,
      frameIndex: this._frameIndex,
      activeMaterial: this._activeSlot?.material.name ?? null,
    };
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * Dispose all cached programs. Does NOT dispose the renderer itself.
   */
  dispose(): void {
    for (const handle of this._programs.values()) {
      handle.program.dispose();
    }
    this._programs.clear();
    this._activeSlot = null;
    this._ctx = {};
  }

  // ── Private ───────────────────────────────────────────────────────────────

  /**
   * Inject DrawContext global uniforms into a program.
   * Silently skips uniforms not declared in the shader (location === null).
   */
  private _injectDrawContext(program: AstroProgram): void {
    const { view, projection, time, resolution } = this._ctx;

    if (view) {
      program.setMatrix4('uViewMatrix', view);
      program.setMatrix4('u_view', view);       // alternate convention
    }

    if (projection) {
      program.setMatrix4('uProjectionMatrix', projection);
      program.setMatrix4('u_projection', projection);
    }

    if (time !== undefined) {
      program.setFloat('uTime', time);
      program.setFloat('u_time', time);
    }

    if (resolution) {
      program.setVec2('uResolution', resolution[0], resolution[1]);
      program.setVec2('u_resolution', resolution[0], resolution[1]);
    }
  }

  /**
   * Find the existing ProgramHandle for a material, or create a minimal one
   * from the material's current program.
   */
  private _findOrCreateHandle(material: Material): ProgramHandle {
    // Search cache for a handle bound to this material
    for (const handle of this._programs.values()) {
      if (handle.material === material) return handle;
    }

    // Material already has a program (compiled externally) — wrap it
    if (material.program) {
      const hash = `__external_${material.name}_${this._programs.size}`;
      const handle: ProgramHandle = {
        program: material.program,
        material,
        sourceHash: hash,
        dispose: () => {
          material.program?.dispose();
          this._programs.delete(hash);
        },
      };
      this._programs.set(hash, handle);
      return handle;
    }

    throw new Error(
      `[RendererAdapter] _findOrCreateHandle: material "${material.name}" has no program`,
    );
  }
}

// ── Module-private helpers ──────────────────────────────────────────────────

/**
 * Simple string hash for shader source deduplication.
 * djb2 variant — fast and sufficient for cache keys.
 */
function _hashSources(vert: string, frag: string): string {
  let h = 5381;
  const combined = vert + '\x00' + frag;
  for (let i = 0; i < combined.length; i++) {
    h = ((h << 5) + h + combined.charCodeAt(i)) | 0;
  }
  return `pgm_${(h >>> 0).toString(36)}`;
}

/**
 * Find the cache key (source hash) for a given AstroProgram instance.
 */
function _findHashForProgram(
  cache: Map<string, ProgramHandle>,
  program: AstroProgram,
): string | null {
  for (const [hash, handle] of cache) {
    if (handle.program === program) return hash;
  }
  return null;
}
