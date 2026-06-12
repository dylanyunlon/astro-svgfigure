/**
 * fx-scene.ts — composable render scene with layer management
 *
 * AT FXScene (88 refs) + SceneLayout (39 refs) port.
 * Used for the playground page's render pipeline:
 *   background layer → cells layer → edges layer → UI layer → post-fx layer
 *
 * Architecture:
 *   FXLayer              one draw pass inside a scene; has opacity, blend-mode,
 *                        visibility flag, and a draw() callback
 *   FXScene              ordered collection of FXLayers bound to a canvas context
 *   FXSceneCompositor    composites multiple FXLayers onto the final canvas via
 *                        off-screen CanvasRenderingContext2D buffers
 *   FXSceneVisibility    tracks which layers are inside the current World viewport
 *                        and suppresses draw calls for fully culled layers
 *
 * The compositor deliberately targets plain Canvas2D so it works in both the
 * browser and OffscreenCanvas workers.  PixiJS / WebGL layers can plug in by
 * rendering to a canvas and passing it as an image source.
 *
 * Upstream references:
 *   upstream/pixijs-engine/src/scene/layers/RenderLayer.ts
 *   upstream/thing-editor/thing-editor/src/engine/lib/assets/src/basic/scene.c.ts
 *   src/lib/renderers/hydra-gl-layer.ts  (off-screen RT pattern)
 *   src/lib/renderers/nuke-pipeline.ts   (pass-chain pattern)
 */

import { World } from './app-state';
import type { Viewport } from './app-state';

// ── Blend mode ───────────────────────────────────────────────────────────────

/**
 * CSS/Canvas composite operations available on an FXLayer.
 * Maps directly to CanvasRenderingContext2D.globalCompositeOperation.
 */
export type BlendMode =
  | 'source-over'     // standard alpha compositing (default)
  | 'additive'        // maps to "lighter" — emissive glow layers
  | 'multiply'        // darkening overlay
  | 'screen'          // brightening overlay
  | 'overlay'         // contrast boost
  | 'luminosity';     // hue + saturation from base, luminosity from layer

/** Remap our friendly name to the Canvas2D string. */
function resolveBlendMode(bm: BlendMode): GlobalCompositeOperation {
  return bm === 'additive' ? 'lighter' : (bm as GlobalCompositeOperation);
}

// ── FXLayer ──────────────────────────────────────────────────────────────────

export interface FXLayerOptions {
  name: string;
  /** Render order — lower zIndex draws first (behind). Default: 0. */
  zIndex?: number;
  /** Initial opacity 0–1. Default: 1. */
  opacity?: number;
  /** Blend mode. Default: 'source-over'. */
  blendMode?: BlendMode;
  /** Start visible. Default: true. */
  visible?: boolean;
  /**
   * The draw callback.  Receives the layer's own off-screen context.
   * The compositor later stamps the result onto the main canvas with opacity
   * and blend-mode applied.
   * @param ctx    off-screen 2D context for this layer
   * @param width  canvas logical width
   * @param height canvas logical height
   */
  draw?: FXLayerDrawFn;
  /**
   * Optional world-space bounding rect for this layer.
   * If provided, FXSceneVisibility will cull the layer when it is outside
   * the current viewport entirely.
   */
  bounds?: { x: number; y: number; w: number; h: number };
}

export type FXLayerDrawFn = (
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  width: number,
  height: number,
) => void;

/**
 * FXLayer — a single compositing layer inside an FXScene.
 *
 * Each layer owns an OffscreenCanvas (or regular Canvas as fallback) that is
 * the draw target for its draw() function.  The compositor blits these
 * off-screen surfaces onto the final canvas in zIndex order.
 */
export class FXLayer {
  readonly name: string;
  zIndex: number;
  opacity: number;
  blendMode: BlendMode;
  visible: boolean;
  bounds?: { x: number; y: number; w: number; h: number };

  private _draw: FXLayerDrawFn | null;
  private _offscreen: OffscreenCanvas | HTMLCanvasElement | null = null;
  private _ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

  constructor(opts: FXLayerOptions) {
    this.name       = opts.name;
    this.zIndex     = opts.zIndex    ?? 0;
    this.opacity    = opts.opacity   ?? 1;
    this.blendMode  = opts.blendMode ?? 'source-over';
    this.visible    = opts.visible   ?? true;
    this.bounds     = opts.bounds;
    this._draw      = opts.draw      ?? null;
  }

  /** Register or replace the draw callback. */
  setDraw(fn: FXLayerDrawFn): void {
    this._draw = fn;
  }

  /**
   * Allocate the off-screen buffer.  Must be called once before draw().
   * FXScene calls this automatically when the layer is added.
   */
  allocate(width: number, height: number): void {
    // Prefer OffscreenCanvas (available in workers + modern browsers)
    if (typeof OffscreenCanvas !== 'undefined') {
      this._offscreen = new OffscreenCanvas(width, height);
      this._ctx = this._offscreen.getContext('2d') as OffscreenCanvasRenderingContext2D;
    } else {
      const c = document.createElement('canvas');
      c.width = width;
      c.height = height;
      this._offscreen = c;
      this._ctx = c.getContext('2d')!;
    }
  }

  /**
   * Resize the off-screen buffer.  The existing content is lost — the draw()
   * callback is responsible for re-drawing on the next frame.
   */
  resize(width: number, height: number): void {
    if (!this._offscreen) { this.allocate(width, height); return; }
    this._offscreen.width  = width;
    this._offscreen.height = height;
  }

  /**
   * Execute the draw callback into the layer's off-screen context.
   * Returns false if the layer has no draw callback or is not allocated.
   */
  render(width: number, height: number): boolean {
    if (!this._draw || !this._ctx || !this._offscreen) return false;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    this._draw(ctx, width, height);
    ctx.restore();
    return true;
  }

  /**
   * Returns the off-screen surface as an ImageBitmap (async, zero-copy) or
   * the canvas element itself as a synchronous fallback.
   */
  async getImageSource(): Promise<ImageBitmap | HTMLCanvasElement> {
    if (!this._offscreen) throw new Error(`[FXLayer "${this.name}"] not allocated`);
    if (this._offscreen instanceof OffscreenCanvas) {
      return this._offscreen.transferToImageBitmap();
    }
    return this._offscreen as HTMLCanvasElement;
  }

  /** Returns the off-screen canvas for use as a CanvasImageSource directly. */
  getCanvas(): OffscreenCanvas | HTMLCanvasElement | null {
    return this._offscreen;
  }

  destroy(): void {
    this._draw = null;
    this._ctx  = null;
    if (this._offscreen instanceof OffscreenCanvas) {
      // OffscreenCanvas has no explicit destroy; just dereference
    }
    this._offscreen = null;
  }
}

// ── FXScene ──────────────────────────────────────────────────────────────────

export interface FXSceneOptions {
  /** Canvas to composite into. */
  canvas: HTMLCanvasElement;
  /** Label for debugging. */
  name?: string;
  /** Clear the canvas to this CSS colour before compositing.  Default: transparent. */
  background?: string;
}

/**
 * FXScene — an ordered collection of FXLayers bound to a single canvas.
 *
 * AT used FXScene for page-level compositing: multiple independent draw
 * passes (background, content, UI chrome, bloom overlay) are maintained as
 * separate layers and composited each frame.  This lets you toggle individual
 * passes during editing without rebuilding the full render graph.
 *
 * ```ts
 * const scene = new FXScene({ canvas, name: 'playground' })
 * const bg    = scene.addLayer({ name: 'background', zIndex: 0 })
 * const cells = scene.addLayer({ name: 'cells',      zIndex: 10 })
 * const edges = scene.addLayer({ name: 'edges',      zIndex: 20 })
 * const ui    = scene.addLayer({ name: 'ui',         zIndex: 30 })
 * const postfx = scene.addLayer({ name: 'post-fx',   zIndex: 40, blendMode: 'additive', opacity: 0.4 })
 *
 * bg.setDraw((ctx, w, h) => { ctx.fillStyle = '#1A1A2E'; ctx.fillRect(0, 0, w, h) })
 * cells.setDraw(renderCellsIntoCtx)
 *
 * scene.render()   // composites all visible layers onto canvas
 * ```
 */
export class FXScene {
  readonly name: string;
  readonly canvas: HTMLCanvasElement;

  private _ctx: CanvasRenderingContext2D;
  private _layers: FXLayer[] = [];
  private _background: string;
  private _compositor: FXSceneCompositor;
  private _visibility: FXSceneVisibility;

  constructor(opts: FXSceneOptions) {
    this.name       = opts.name ?? 'FXScene';
    this.canvas     = opts.canvas;
    this._background = opts.background ?? 'transparent';

    const ctx = opts.canvas.getContext('2d');
    if (!ctx) throw new Error(`[FXScene "${this.name}"] failed to get 2D context`);
    this._ctx = ctx;

    this._compositor  = new FXSceneCompositor(this._ctx, this._background);
    this._visibility  = new FXSceneVisibility();
  }

  // ── Layer management ────────────────────────────────────────────────────

  /**
   * Add a new layer to the scene.  The layer is allocated and inserted in
   * zIndex order.  Returns the created FXLayer for further configuration.
   */
  addLayer(opts: FXLayerOptions): FXLayer {
    const layer = new FXLayer(opts);
    layer.allocate(this.canvas.width, this.canvas.height);
    this._layers.push(layer);
    this._sortLayers();
    this._visibility.register(layer);
    return layer;
  }

  /**
   * Remove a layer by name.  Destroys its off-screen buffer.
   * Returns true if a layer was found and removed.
   */
  removeLayer(name: string): boolean {
    const idx = this._layers.findIndex((l) => l.name === name);
    if (idx === -1) return false;
    this._visibility.unregister(this._layers[idx]);
    this._layers[idx].destroy();
    this._layers.splice(idx, 1);
    return true;
  }

  /** Get a layer by name.  Returns undefined if not found. */
  getLayer(name: string): FXLayer | undefined {
    return this._layers.find((l) => l.name === name);
  }

  /** All layers in current render order (ascending zIndex). */
  get layers(): readonly FXLayer[] {
    return this._layers;
  }

  // ── Resize ──────────────────────────────────────────────────────────────

  /** Resize all layer buffers when the canvas dimensions change. */
  resize(width: number, height: number): void {
    this.canvas.width  = width;
    this.canvas.height = height;
    for (const layer of this._layers) {
      layer.resize(width, height);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────

  /**
   * Composite all visible, in-viewport layers onto the canvas.
   * Call once per animation frame.
   */
  render(): void {
    const { width, height } = this.canvas;
    this._visibility.update(World.viewport);

    // Collect layers that should be drawn this frame
    const visible = this._layers.filter((l) => l.visible && this._visibility.isVisible(l));

    // Execute per-layer draw into off-screen buffers
    for (const layer of visible) {
      layer.render(width, height);
    }

    // Composite onto main canvas
    this._compositor.composite(visible, width, height);
  }

  /** Release all resources. */
  destroy(): void {
    for (const layer of this._layers) layer.destroy();
    this._layers = [];
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private _sortLayers(): void {
    this._layers.sort((a, b) => a.zIndex - b.zIndex);
  }
}

// ── FXSceneCompositor ────────────────────────────────────────────────────────

/**
 * FXSceneCompositor — stamps FXLayer off-screen buffers onto the target canvas
 * in order, applying each layer's opacity and blend-mode.
 *
 * AT's compositor was a WebGL-based multi-RT blit; we use Canvas2D drawImage
 * so it works without a GPU context.  The pattern is identical: clear, then
 * for each layer set globalAlpha + globalCompositeOperation, drawImage.
 */
export class FXSceneCompositor {
  private _ctx: CanvasRenderingContext2D;
  private _background: string;

  constructor(ctx: CanvasRenderingContext2D, background: string) {
    this._ctx        = ctx;
    this._background = background;
  }

  /** Update the background fill colour. */
  setBackground(bg: string): void {
    this._background = bg;
  }

  /**
   * Composite the supplied layers onto the bound canvas context.
   * Layers must already be rendered into their off-screen buffers.
   */
  composite(layers: readonly FXLayer[], width: number, height: number): void {
    const ctx = this._ctx;

    // Clear to background
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    if (this._background === 'transparent') {
      ctx.clearRect(0, 0, width, height);
    } else {
      ctx.fillStyle = this._background;
      ctx.fillRect(0, 0, width, height);
    }
    ctx.restore();

    // Blit each layer
    for (const layer of layers) {
      const src = layer.getCanvas();
      if (!src) continue;

      ctx.save();
      ctx.globalAlpha               = Math.max(0, Math.min(1, layer.opacity));
      ctx.globalCompositeOperation  = resolveBlendMode(layer.blendMode);
      ctx.drawImage(src as CanvasImageSource, 0, 0);
      ctx.restore();
    }
  }
}

// ── FXSceneVisibility ────────────────────────────────────────────────────────

/**
 * FXSceneVisibility — viewport-based layer culling.
 *
 * AT's SceneLayout (39 refs) managed visibility of sub-scenes based on scroll
 * position and viewport.  Here we track which FXLayers have a `bounds` rect
 * that overlaps the current World viewport.  Layers without bounds are always
 * considered visible (conservative: background, UI, post-fx layers usually
 * cover the full canvas).
 *
 * ```ts
 * const vis = new FXSceneVisibility()
 * vis.register(cellsLayer)
 * vis.update(World.viewport)
 * if (vis.isVisible(cellsLayer)) {  …  }
 * ```
 */
export class FXSceneVisibility {
  private _visible: Set<FXLayer> = new Set();
  private _registered: Set<FXLayer> = new Set();

  /** Track a layer.  Called by FXScene.addLayer(). */
  register(layer: FXLayer): void {
    this._registered.add(layer);
    // Default to visible until the first update() call
    this._visible.add(layer);
  }

  /** Stop tracking a layer.  Called by FXScene.removeLayer(). */
  unregister(layer: FXLayer): void {
    this._registered.delete(layer);
    this._visible.delete(layer);
  }

  /**
   * Recompute visibility for all registered layers against the given viewport.
   * Called once per frame by FXScene.render() before issuing draw calls.
   */
  update(viewport: Viewport): void {
    for (const layer of this._registered) {
      if (!layer.bounds) {
        // No bounds → always visible (covers full canvas or unbounded UI layer)
        this._visible.add(layer);
        continue;
      }

      const { x: bx, y: by, w: bw, h: bh } = layer.bounds;
      const { x: vx, y: vy, width: vw, height: vh } = viewport;
      const overlaps =
        bx < vx + vw &&
        bx + bw > vx &&
        by < vy + vh &&
        by + bh > vy;

      if (overlaps) {
        this._visible.add(layer);
      } else {
        this._visible.delete(layer);
      }
    }
  }

  /**
   * Returns true if the layer is currently visible (within the viewport or
   * has no bounds).  A layer that is also FXLayer.visible=false is handled
   * by FXScene — this method only reflects the viewport cull result.
   */
  isVisible(layer: FXLayer): boolean {
    return this._visible.has(layer);
  }

  /**
   * Returns a snapshot of all currently visible layer names.
   * Useful for debugging / devtools.
   */
  visibleNames(): string[] {
    return [...this._visible].map((l) => l.name);
  }

  /** Reset all visibility state (e.g. after a scene rebuild). */
  clear(): void {
    this._visible.clear();
    this._registered.clear();
  }
}

// ── Pre-built playground scene factory ──────────────────────────────────────

/**
 * Layer names for the standard playground scene, exported so callers can
 * reference them without string literals.
 */
export const PlaygroundLayers = {
  BACKGROUND: 'background',
  CELLS:      'cells',
  EDGES:      'edges',
  UI:         'ui',
  POST_FX:    'post-fx',
} as const;

/**
 * createPlaygroundScene — builds a pre-configured FXScene with the five
 * canonical layers for the playground page:
 *
 *   background  z=0   source-over  opacity=1     full-canvas bg fill
 *   cells       z=10  source-over  opacity=1     PixiJS cell containers
 *   edges       z=20  source-over  opacity=0.85  bezier edge graph
 *   ui          z=30  source-over  opacity=1     controls / labels
 *   post-fx     z=40  additive     opacity=0.35  bloom / glow overlay
 *
 * Each layer's draw callback is left unset — callers call layer.setDraw(fn).
 *
 * ```ts
 * const scene = createPlaygroundScene(canvas)
 * scene.getLayer(PlaygroundLayers.CELLS)!.setDraw(renderCells)
 * scene.getLayer(PlaygroundLayers.EDGES)!.setDraw(renderEdges)
 *
 * // each rAF tick:
 * scene.render()
 * ```
 */
export function createPlaygroundScene(canvas: HTMLCanvasElement): FXScene {
  const scene = new FXScene({ canvas, name: 'playground', background: '#1A1A2E' });

  scene.addLayer({
    name:      PlaygroundLayers.BACKGROUND,
    zIndex:    0,
    opacity:   1,
    blendMode: 'source-over',
  });

  scene.addLayer({
    name:      PlaygroundLayers.CELLS,
    zIndex:    10,
    opacity:   1,
    blendMode: 'source-over',
  });

  scene.addLayer({
    name:      PlaygroundLayers.EDGES,
    zIndex:    20,
    opacity:   0.85,
    blendMode: 'source-over',
  });

  scene.addLayer({
    name:      PlaygroundLayers.UI,
    zIndex:    30,
    opacity:   1,
    blendMode: 'source-over',
  });

  scene.addLayer({
    name:      PlaygroundLayers.POST_FX,
    zIndex:    40,
    opacity:   0.35,
    blendMode: 'additive',
  });

  return scene;
}
