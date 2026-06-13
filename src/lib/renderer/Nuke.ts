/**
 * Nuke.ts — post-processing pipeline manager
 *
 * AT Nuke module port (133 refs).
 * Upstream references:
 *   upstream/pixijs-engine/src/fx/nuke/Nuke.ts
 *   upstream/pixijs-engine/src/fx/nuke/NukeRT.ts
 *   src/lib/fx-scene.ts (compositing patterns)
 *
 * Lifecycle events (mirrors AT):
 *   BEFORE_PASSES  — fired once before the pass chain runs
 *   RENDER         — fired per pass (within the chain)
 *   POST_RENDER    — fired after chain + final blit to canvas
 *
 * Key APIs:
 *   nuke.getRT(name)          → named RenderTarget
 *   nuke.attachDrawBuffer()   → MRT / multiple draw buffers
 *   nuke.render()             → execute full pass chain
 *   nuke.recyclePingPong()    → swap ping/pong RT pair
 *   nuke.defaultPass          → the final blit-to-canvas pass
 */

import { NukePass } from './NukePass';
import type { RenderTarget, UniformValue } from './NukePass';

// ── Lifecycle event names ─────────────────────────────────────────────────────

export const NukeEvent = {
  BEFORE_PASSES: 'nuke:before_passes',
  RENDER:        'nuke:render',
  POST_RENDER:   'nuke:post_render',
} as const;
export type NukeEventType = typeof NukeEvent[keyof typeof NukeEvent];

// ── Listener type ─────────────────────────────────────────────────────────────

export type NukeListener = (nuke: Nuke) => void;

// ── RenderTarget factory options ──────────────────────────────────────────────

export interface RTOptions {
  name: string;
  width: number;
  height: number;
  /** Add a depth renderbuffer. Default: false. */
  depth?: boolean;
  /** Texture internal format. Default: RGBA16F for HDR. */
  internalFormat?: GLenum;
  /** Texture min/mag filter. Default: LINEAR. */
  filter?: GLenum;
  /** Texture wrap mode. Default: CLAMP_TO_EDGE. */
  wrap?: GLenum;
}

// ── Ping-pong pair ────────────────────────────────────────────────────────────

export interface PingPongPair {
  ping: RenderTarget;
  pong: RenderTarget;
  /** True if ping is currently the "read" target. */
  pingIsRead: boolean;
  /** Current read target. */
  readonly read: RenderTarget;
  /** Current write target. */
  readonly write: RenderTarget;
}

// ── Blit vertex / fragment (copy last pass output to canvas) ──────────────────

const BLIT_VERT = /* glsl */ `#version 300 es
precision highp float;
void main() {
  float x = float((gl_VertexID & 1) << 1) - 1.0;
  float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

const BLIT_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_input;
uniform vec2      u_resolution;
out vec4 fragColor;
void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  fragColor = texture(u_input, uv);
}
`;

// ── Nuke ──────────────────────────────────────────────────────────────────────

/**
 * Nuke — manages an ordered chain of NukePasses and the GL resources they
 * share (render targets, ping-pong buffers, draw-buffer attachments).
 *
 * Usage:
 * ```ts
 * const nuke = new Nuke(gl, canvas.width, canvas.height);
 *
 * const bloom = nuke.createPass({ name: 'bloom', ... });
 * nuke.addPass(bloom);
 *
 * // Game loop:
 * nuke.render();
 * ```
 */
export class Nuke {
  readonly gl: WebGL2RenderingContext;
  readonly width: number;
  readonly height: number;

  /** Ordered list of post-processing passes. */
  passes: NukePass[] = [];

  /** The built-in final blit pass that writes to the canvas. */
  defaultPass: NukePass;

  private _rts = new Map<string, RenderTarget>();
  private _pingPongs = new Map<string, PingPongPair>();
  private _listeners = new Map<NukeEventType, Set<NukeListener>>();

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl     = gl;
    this.width  = width;
    this.height = height;

    // Create a screen-sized "canvas" render target used by defaultPass output.
    // Its FBO is null (canvas default framebuffer) — we handle that as a special
    // case in _buildBlitPass.
    this.defaultPass = this._buildBlitPass();
  }

  // ── RenderTarget management ───────────────────────────────────────────────

  /**
   * Create and register a named RenderTarget.
   * Throws if a target with the same name already exists.
   */
  createRT(opts: RTOptions): RenderTarget {
    if (this._rts.has(opts.name)) {
      throw new Error(`[Nuke] RenderTarget "${opts.name}" already exists`);
    }
    const rt = this._allocRT(opts);
    this._rts.set(opts.name, rt);
    return rt;
  }

  /**
   * Retrieve a named RenderTarget (AT: `Nuke.getRT(name)`).
   * Throws if not found.
   */
  getRT(name: string): RenderTarget {
    const rt = this._rts.get(name);
    if (!rt) throw new Error(`[Nuke] RenderTarget "${name}" not found`);
    return rt;
  }

  /**
   * Attach multiple draw buffers (MRT) to a framebuffer
   * (AT: `Nuke.attachDrawBuffer()`).
   *
   * @param fbo       The framebuffer to configure.
   * @param textures  Colour textures to attach at COLOR_ATTACHMENTi.
   */
  attachDrawBuffer(fbo: WebGLFramebuffer, textures: WebGLTexture[]): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const drawBuffers: GLenum[] = [];
    textures.forEach((tex, i) => {
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0 + i,
        gl.TEXTURE_2D,
        tex,
        0
      );
      drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
    });
    gl.drawBuffers(drawBuffers);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── Ping-pong management ──────────────────────────────────────────────────

  /**
   * Create a named ping-pong RT pair (two identically-sized targets).
   * Use `recyclePingPong(name)` to swap read/write each frame.
   */
  createPingPong(baseName: string, opts: Omit<RTOptions, 'name'>): PingPongPair {
    const ping = this.createRT({ ...opts, name: `${baseName}:ping` });
    const pong = this.createRT({ ...opts, name: `${baseName}:pong` });

    const pair: PingPongPair = {
      ping,
      pong,
      pingIsRead: true,
      get read()  { return this.pingIsRead ? this.ping  : this.pong; },
      get write() { return this.pingIsRead ? this.pong  : this.ping; },
    };
    this._pingPongs.set(baseName, pair);
    return pair;
  }

  /**
   * Swap a ping-pong pair's read/write roles
   * (AT: `Nuke.recyclePingPong()`).
   */
  recyclePingPong(baseName: string): PingPongPair {
    const pair = this._pingPongs.get(baseName);
    if (!pair) throw new Error(`[Nuke] PingPong "${baseName}" not found`);
    pair.pingIsRead = !pair.pingIsRead;
    return pair;
  }

  // ── Pass management ───────────────────────────────────────────────────────

  addPass(pass: NukePass): this {
    this.passes.push(pass);
    return this;
  }

  removePass(pass: NukePass): this {
    const i = this.passes.indexOf(pass);
    if (i !== -1) this.passes.splice(i, 1);
    return this;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  /**
   * Execute the full post-processing chain, then blit to the canvas.
   *
   * Lifecycle:
   *   BEFORE_PASSES → pass[0].render … pass[n].render → POST_RENDER
   */
  render(): void {
    const { gl } = this;

    this._emit(NukeEvent.BEFORE_PASSES);

    for (const pass of this.passes) {
      if (!pass.enabled) continue;
      this._emit(NukeEvent.RENDER);
      pass.render(gl);
    }

    // Final blit: write last pass output to canvas framebuffer.
    this._blitToCanvas();

    this._emit(NukeEvent.POST_RENDER);
  }

  // ── Listeners ─────────────────────────────────────────────────────────────

  on(event: NukeEventType, listener: NukeListener): this {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event)!.add(listener);
    return this;
  }

  off(event: NukeEventType, listener: NukeListener): this {
    this._listeners.get(event)?.delete(listener);
    return this;
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  dispose(): void {
    const { gl } = this;
    for (const pass of this.passes) pass.dispose();
    this.defaultPass.dispose();
    for (const rt of this._rts.values()) this._freeRT(gl, rt);
    this._rts.clear();
    this._pingPongs.clear();
    this._listeners.clear();
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _allocRT(opts: RTOptions): RenderTarget {
    const { gl } = this;
    const {
      name,
      width,
      height,
      depth           = false,
      internalFormat  = gl.RGBA16F,
      filter          = gl.LINEAR,
      wrap            = gl.CLAMP_TO_EDGE,
    } = opts;

    // Ensure float-texture extension is available.
    gl.getExtension('EXT_color_buffer_float');

    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0,
      gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
    gl.bindTexture(gl.TEXTURE_2D, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, texture, 0);

    let depthBuffer: WebGLRenderbuffer | undefined;
    if (depth) {
      depthBuffer = gl.createRenderbuffer()!;
      gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
        gl.RENDERBUFFER, depthBuffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { name, fbo, texture, depthBuffer, width, height };
  }

  private _freeRT(gl: WebGL2RenderingContext, rt: RenderTarget): void {
    gl.deleteTexture(rt.texture);
    gl.deleteFramebuffer(rt.fbo);
    if (rt.depthBuffer) gl.deleteRenderbuffer(rt.depthBuffer);
  }

  /**
   * Build the default blit pass that copies the last pass's output to the
   * canvas (null framebuffer).  We use a synthetic RenderTarget whose FBO
   * is null to represent the canvas.
   */
  private _buildBlitPass(): NukePass {
    // Sentinel "canvas" RT — NukePass checks for null FBO to bind canvas.
    const canvasRT: RenderTarget = {
      name:    '__canvas__',
      fbo:     null as unknown as WebGLFramebuffer,
      texture: null as unknown as WebGLTexture,
      width:   this.width,
      height:  this.height,
    };

    // Placeholder input — will be swapped to last pass's output before blit.
    const blitPass = new NukePass({
      name:    'nuke:default-blit',
      fragSrc: BLIT_FRAG,
      vertSrc: BLIT_VERT,
      input:   canvasRT, // overwritten in _blitToCanvas
      output:  canvasRT,
    });
    return blitPass;
  }

  private _blitToCanvas(): void {
    const { gl, passes } = this;
    if (passes.length === 0) return;

    const lastPass = [...passes].reverse().find(p => p.enabled);
    if (!lastPass) return;

    // Point the blit pass at the last enabled pass's output.
    this.defaultPass.input = lastPass.output;

    // Bind canvas framebuffer directly.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);

    this.defaultPass.compile(gl);
    // Override output binding: we already bound null FBO above, so call
    // render() but then re-bind null (NukePass will bind its FBO —
    // workaround: we use a patched render here).
    this._renderToCanvas(gl);
  }

  private _renderToCanvas(gl: WebGL2RenderingContext): void {
    // Minimal inline blit that skips FBO re-binding inside NukePass.
    const pass = this.defaultPass;
    pass.compile(gl);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    // Delegate to pass.render() which will re-bind null FBO (its output.fbo
    // is null — our sentinel canvasRT).
    pass.render(gl);
  }

  private _emit(event: NukeEventType): void {
    this._listeners.get(event)?.forEach(fn => fn(this));
  }
}
