/**
 * FXScene.ts — offscreen render-to-texture scene
 *
 * AT FXScene (53 refs).  Extends Scene with:
 *   - A RenderTarget (FBO + texture) as the render destination.
 *   - manualRender mode: scene only re-renders when renderFrame() is called
 *     explicitly (AT's FXScene.manualRender pattern — saves GPU for static
 *     or infrequently-changing layers).
 *   - onCreate lifecycle hook — called once after the GL context and
 *     RenderTarget are ready.  Use it to load shaders, upload geometry, etc.
 *   - texture getter — exposes the resolved texture to Nuke post-process passes.
 *
 * Composite pipeline sketch:
 *
 *   FXScene "background"  → RT0.texture ──┐
 *   FXScene "cells"       → RT1.texture ──┤  NukePipeline
 *   FXScene "edges"       → RT2.texture ──┤  (BloomPass, FXAAPass, …)
 *   FXScene "ui"          → RT3.texture ──┘
 *                                           ↓ canvas
 *
 * References:
 *   src/lib/renderer/Scene.ts          — base traversal + camera
 *   src/lib/renderer/RenderTarget.ts   — FBO wrapper
 *   src/lib/renderers/hydra-gl-layer.ts (RenderTarget pioneer)
 *   src/lib/renderers/nuke-pipeline.ts  (NukePass.render(input, output))
 *   src/lib/fx-scene.ts                (Canvas2D FXLayer/compositor analogue)
 */

import { Scene } from './Scene';
import type { SceneOptions } from './Scene';
import { RenderTarget } from './RenderTarget';
import type { RenderTargetOptions } from './RenderTarget';

// ── Lifecycle callback types ─────────────────────────────────────────────────

/**
 * Called once when the FXScene is ready (GL context + RT initialised).
 * Typical use: compile shaders, create VAOs, upload static textures.
 */
export type OnCreateFn = (scene: FXScene) => void | Promise<void>;

/**
 * Called every frame when the scene should re-render.
 * Return false to prevent the default child traversal (full manual control).
 */
export type OnRenderFn = (
  gl: WebGL2RenderingContext,
  scene: FXScene,
) => void | boolean;

// ── FXSceneOptions ───────────────────────────────────────────────────────────

export interface FXSceneOptions extends SceneOptions {
  /**
   * WebGL2 context.  Required — FXScene cannot exist without a GL context.
   */
  gl: WebGL2RenderingContext;

  /**
   * Passed through to RenderTarget constructor.
   * width/height default to SceneOptions.width/height if omitted here.
   */
  renderTargetOptions?: Partial<RenderTargetOptions>;

  /**
   * When true the scene will NOT automatically re-render every frame.
   * Call renderFrame() manually to trigger a render pass.
   * AT FXScene.manualRender pattern.
   * Default: false (auto-render every frame).
   */
  manualRender?: boolean;

  /**
   * Lifecycle: called once after GL + RenderTarget are ready.
   * May be async — the FXScene will await it before marking itself ready.
   */
  onCreate?: OnCreateFn;

  /**
   * Lifecycle: called at the start of each render pass before child traversal.
   * Return `false` to suppress the automatic child traversal.
   */
  onRender?: OnRenderFn;

  /**
   * Clear color [r, g, b, a] applied before each render pass.
   * Default: [0, 0, 0, 0] (transparent black).
   */
  clearColor?: [number, number, number, number];
}

// ── FXScene ──────────────────────────────────────────────────────────────────

export class FXScene extends Scene {
  readonly gl: WebGL2RenderingContext;

  /** The offscreen framebuffer this scene renders into. */
  readonly renderTarget: RenderTarget;

  /**
   * The resolved colour texture — pass to NukePass / composite shader
   * as a sampler2D uniform.
   */
  get texture(): WebGLTexture {
    return this.renderTarget.texture;
  }

  /**
   * AT: FXScene.manualRender.
   * When true, renderFrame() must be called explicitly.
   */
  manualRender: boolean;

  /**
   * Set to true after onCreate() resolves.
   * Render calls before readiness are silently skipped.
   */
  ready = false;

  /** Signals that the scene content has changed and needs a re-render. */
  private _needsRender = true;

  private readonly _clearColor: [number, number, number, number];
  private readonly _onRender?: OnRenderFn;

  constructor(opts: FXSceneOptions) {
    super({ camera: opts.camera, width: opts.width, height: opts.height });

    this.gl           = opts.gl;
    this.manualRender = opts.manualRender ?? false;
    this._clearColor  = opts.clearColor ?? [0, 0, 0, 0];
    this._onRender    = opts.onRender;

    const rtOpts: RenderTargetOptions = {
      width:       opts.width  ?? 1,
      height:      opts.height ?? 1,
      attachments: opts.renderTargetOptions?.attachments,
      hdr:         opts.renderTargetOptions?.hdr,
      depth:       opts.renderTargetOptions?.depth,
    };

    this.renderTarget = new RenderTarget(this.gl, rtOpts);

    // Kick off async onCreate; mark ready when resolved
    const init = opts.onCreate ? opts.onCreate(this) : undefined;
    if (init instanceof Promise) {
      init.then(() => { this.ready = true; });
    } else {
      this.ready = true;
    }
  }

  // ── Manual render signalling ─────────────────────────────────────────────

  /**
   * Mark this scene as needing a re-render on the next frame.
   * Only meaningful when manualRender === true.
   */
  invalidate(): void {
    this._needsRender = true;
  }

  // ── Core render ─────────────────────────────────────────────────────────

  /**
   * Render scene children into the offscreen RenderTarget.
   *
   * Call this once per frame from your render loop (or from a compositor).
   * When manualRender is true, skips the render pass unless invalidate()
   * has been called since the last frame.
   *
   * After this call, `this.texture` contains the up-to-date scene image.
   */
  renderFrame(): void {
    if (!this.ready) return;
    if (this.manualRender && !this._needsRender) return;

    const { gl, renderTarget } = this;

    // Bind offscreen FBO
    renderTarget.bind();

    // Clear
    const [r, g, b, a] = this._clearColor;
    gl.clearColor(r, g, b, a);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // onRender hook — return false to skip default traversal
    const skipTraversal = this._onRender?.(gl, this) === false;

    if (!skipTraversal) {
      // Base class child traversal
      super.render(gl);
    }

    // Unbind FBO — subsequent GL calls go to canvas (or outer FBO)
    renderTarget.unbind();

    this._needsRender = false;
  }

  /**
   * Override Scene.render() so that callers who hold a Scene reference
   * still trigger the FXScene's offscreen pass correctly.
   */
  override render(gl: WebGL2RenderingContext): void {
    // Ignore the passed-in gl — we always use this.gl + our own RT.
    void gl;
    this.renderFrame();
  }

  // ── Resize ───────────────────────────────────────────────────────────────

  /**
   * Resize both the RenderTarget attachments and the base camera.
   * Call whenever the canvas size changes.
   */
  override resize(width: number, height: number): void {
    super.resize(width, height);
    this.renderTarget.resize(width, height);
    this.invalidate();
  }

  // ── Texture binding convenience ──────────────────────────────────────────

  /**
   * Bind the scene's output texture to a sampler unit.
   * Equivalent to renderTarget.bindTexture(unit).
   *
   * @param unit           gl.TEXTURE0 + unit
   * @param attachmentIndex  Which MRT attachment (default: 0)
   */
  bindTexture(unit = 0, attachmentIndex = 0): void {
    this.renderTarget.bindTexture(unit, attachmentIndex);
  }

  // ── Teardown ─────────────────────────────────────────────────────────────

  override destroy(): void {
    super.destroy();           // clears children
    this.renderTarget.destroy();
  }
}

// ── FXSceneCompositor ────────────────────────────────────────────────────────

/**
 * Composites multiple FXScene textures into the final canvas using a
 * fullscreen quad shader.
 *
 * Usage:
 *   const compositor = new FXSceneCompositor(gl, [bgScene, cellsScene, uiScene]);
 *   // each frame:
 *   compositor.composite();
 *
 * AT equivalent: the final blit pass that reads each FXScene.renderTarget.texture
 * and blends them onto the canvas with per-layer opacity + blend mode.
 */

const COMPOSITE_VERT = `#version 300 es
in vec2 aPosition;
out vec2 vUV;
void main() {
  vUV = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const COMPOSITE_FRAG = `#version 300 es
precision mediump float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float uOpacity;
void main() {
  fragColor = texture(uTexture, vUV) * uOpacity;
}`;

export interface CompositeLayer {
  scene: FXScene;
  /** Per-layer opacity 0–1. Default: 1. */
  opacity?: number;
  /** Skip compositing this layer.  Default: false. */
  hidden?: boolean;
}

export class FXSceneCompositor {
  private readonly gl: WebGL2RenderingContext;
  private readonly layers: CompositeLayer[];
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly uTexture: WebGLUniformLocation;
  private readonly uOpacity: WebGLUniformLocation;

  constructor(gl: WebGL2RenderingContext, layers: CompositeLayer[]) {
    this.gl     = gl;
    this.layers = layers;

    // Compile composite shader
    this.program = _compileProgram(gl, COMPOSITE_VERT, COMPOSITE_FRAG);
    this.uTexture = gl.getUniformLocation(this.program, 'uTexture')!;
    this.uOpacity = gl.getUniformLocation(this.program, 'uOpacity')!;

    // Fullscreen quad VAO
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,   -1, 1,
      -1,  1,   1, -1,    1, 1,
    ]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, 'aPosition');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  /**
   * Render all scene textures onto the default framebuffer (canvas).
   * Layers are drawn in array order (index 0 = bottom).
   */
  composite(): void {
    const { gl } = this;

    // Render each FXScene into its own RT first
    for (const layer of this.layers) {
      if (!layer.hidden) layer.scene.renderFrame();
    }

    // Now blit to canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    for (const layer of this.layers) {
      if (layer.hidden) continue;

      layer.scene.bindTexture(0);
      gl.uniform1i(this.uTexture, 0);
      gl.uniform1f(this.uOpacity, layer.opacity ?? 1.0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  destroy(): void {
    this.gl.deleteProgram(this.program);
    this.gl.deleteVertexArray(this.vao);
  }
}

// ── Internal shader helpers ──────────────────────────────────────────────────

function _compileShader(
  gl: WebGL2RenderingContext,
  src: string,
  type: number,
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`[FXScene] Shader compile error:\n${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

function _compileProgram(
  gl: WebGL2RenderingContext,
  vert: string,
  frag: string,
): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, _compileShader(gl, vert, gl.VERTEX_SHADER));
  gl.attachShader(program, _compileShader(gl, frag, gl.FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`[FXScene] Program link error:\n${gl.getProgramInfoLog(program)}`);
  }
  return program;
}
