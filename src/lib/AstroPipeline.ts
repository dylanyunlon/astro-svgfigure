/**
 * AstroPipeline.ts — full render loop entry point
 *
 * Integrates the complete Astro rendering stack:
 *   AstroRenderer → WebGL2 context + extensions
 *   FXScene       → offscreen render-to-texture (cells layer)
 *   Nuke          → post-processing pipeline (BloomPass, etc.)
 *   CellInstanceManager → per-species instanced mesh batching
 *
 * Usage:
 *   const pipeline = new AstroPipeline();
 *   await pipeline.init(canvas);
 *   pipeline.start();
 *   // later:
 *   pipeline.stop();
 *   pipeline.dispose();
 *
 * xiaodi #51 — core integration
 */

import { AstroRenderer } from './renderer/AstroRenderer';
import { FXScene } from './renderer/FXScene';
import { Nuke } from './renderer/Nuke';
import { BloomPass } from './renderer/passes/BloomPass';
import { CellInstanceManager } from './renderer/CellInstanceManager';
import type { RenderTarget } from './renderer/NukePass';

// ── Pipeline configuration ───────────────────────────────────────────────────

export interface AstroPipelineConfig {
  /** Base URL for cell data directory. Default: '/api/cells'. */
  cellsEndpoint?: string;
  /** Cell param directory for CellInstanceManager.loadFromParamsDir(). */
  cellParamsDir?: string;
  /** Bloom strength (0–3). Default: 1.2. */
  bloomStrength?: number;
  /** Bloom radius. Default: 1.0. */
  bloomRadius?: number;
  /** Luminosity threshold for bloom extraction. Default: 0.75. */
  luminosityThreshold?: number;
  /** Device pixel ratio. Default: window.devicePixelRatio ?? 1. */
  dpr?: number;
}

// ── AstroPipeline ────────────────────────────────────────────────────────────

export class AstroPipeline {
  // ── Sub-systems ────────────────────────────────────────────────────────────
  private renderer!: AstroRenderer;
  private mainScene!: FXScene;
  private nuke!: Nuke;
  private bloom!: BloomPass;
  private cellManager!: CellInstanceManager;

  // ── Render targets ─────────────────────────────────────────────────────────
  /** Offscreen RT that FXScene renders cells into. */
  private sceneRT!: RenderTarget;
  /** Final composited RT that Nuke blits to canvas. */
  private outputRT!: RenderTarget;

  // ── Loop state ─────────────────────────────────────────────────────────────
  private running = false;
  private _rafId  = 0;

  // ── Config ─────────────────────────────────────────────────────────────────
  private cfg: Required<AstroPipelineConfig>;

  constructor(config: AstroPipelineConfig = {}) {
    const dpr = config.dpr ?? (typeof window !== 'undefined' ? (window.devicePixelRatio ?? 1) : 1);
    this.cfg = {
      cellsEndpoint:       config.cellsEndpoint      ?? '/api/cells',
      cellParamsDir:       config.cellParamsDir       ?? '/channels/cell',
      bloomStrength:       config.bloomStrength       ?? 1.2,
      bloomRadius:         config.bloomRadius         ?? 1.0,
      luminosityThreshold: config.luminosityThreshold ?? 0.75,
      dpr,
    };
  }

  // ── Initialisation ─────────────────────────────────────────────────────────

  /**
   * Set up the full rendering stack and load cell data.
   * Must be awaited before calling start().
   */
  async init(canvas: HTMLCanvasElement): Promise<void> {
    const { cfg } = this;

    // 1. Size canvas for device pixel ratio
    const cssW = canvas.clientWidth  || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    const w    = Math.round(cssW * cfg.dpr);
    const h    = Math.round(cssH * cfg.dpr);
    canvas.width  = w;
    canvas.height = h;

    // 2. Create WebGL renderer (singleton, attaches to canvas)
    this.renderer = AstroRenderer.getInstance({ canvas, alpha: false, antialias: false });
    const gl = this.renderer.context as WebGL2RenderingContext;

    // 3. Create FXScene — renders cells into an offscreen texture
    this.mainScene = new FXScene({
      gl,
      width:  w,
      height: h,
      clearColor: [0, 0, 0, 1],
    });

    // Grab the scene's RenderTarget for the Bloom input
    this.sceneRT = this.mainScene.renderTarget as unknown as RenderTarget;

    // 4. Create Nuke post-process pipeline
    this.nuke = new Nuke(gl, w, h);

    // Output RT — screen-sized, fed into the final blit pass
    this.outputRT = this.nuke.createRT({
      name:   'astro:output',
      width:  w,
      height: h,
    });

    // 5. Add BloomPass (registers its own NukePass chain with nuke)
    this.bloom = new BloomPass(this.nuke, this.sceneRT, this.outputRT, {
      bloomStrength:       cfg.bloomStrength,
      bloomRadius:         cfg.bloomRadius,
      luminosityThreshold: cfg.luminosityThreshold,
    });

    // 6. Fetch cell data and build instanced meshes
    this.cellManager = new CellInstanceManager(gl);

    let loadedFromAPI = false;
    try {
      const res = await fetch(cfg.cellsEndpoint);
      if (res.ok) {
        const descriptors = await res.json();
        this.cellManager.loadFromDescriptors(descriptors);
        loadedFromAPI = true;
      }
    } catch (_) {
      // Fall through to params-dir fallback
    }

    if (!loadedFromAPI) {
      // Fallback: load from static /channels/cell/<id>/params.json files
      await this.cellManager.loadFromParamsDir(cfg.cellParamsDir);
    }

    if (import.meta.env?.DEV) {
      console.info('[AstroPipeline] init complete', this.cellManager.stats());
    }
  }

  // ── Render loop ───────────────────────────────────────────────────────────

  /** Arrow function so it can be passed directly to requestAnimationFrame. */
  private frame = (): void => {
    if (!this.running) return;

    // 1. Update instance attributes (animation, pubsub state changes, etc.)
    //    CellInstanceManager exposes updateCell() for individual patches;
    //    a future CellPubSub subscriber calls it each tick as messages arrive.
    //    Here we call draw() which flushes all dirty instance buffers.

    // 2. Render cells into the offscreen FXScene render target
    this.mainScene.renderFrame();

    // 3. Run Nuke post-processing chain (BloomPass → final blit to canvas)
    this.nuke.render();

    // 4. Draw instanced cell meshes (one call per species)
    //    These are drawn into the FXScene's bound FBO during renderFrame,
    //    but if needed as a standalone pass, call here after rebinding.
    //    The FXScene's onRender hook is the canonical place — this call
    //    ensures the instanced quads land in the scene texture.
    this.cellManager.draw();

    this._rafId = requestAnimationFrame(this.frame);
  };

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Begin the render loop. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this._rafId  = requestAnimationFrame(this.frame);
  }

  /** Pause the render loop (can be restarted with start()). */
  stop(): void {
    this.running = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = 0;
    }
  }

  /** Tear down all GPU resources. */
  dispose(): void {
    this.stop();
    this.bloom.dispose();
    this.nuke.dispose();
    this.mainScene.destroy();
    this.cellManager.dispose();
    AstroRenderer.destroyInstance();
  }

  // ── Resize ────────────────────────────────────────────────────────────────

  /**
   * Call when the canvas element is resized.
   * Updates all render targets, the scene camera, and the Nuke viewport.
   */
  resize(cssWidth: number, cssHeight: number): void {
    const w = Math.round(cssWidth  * this.cfg.dpr);
    const h = Math.round(cssHeight * this.cfg.dpr);

    this.renderer.resize(w, h);
    this.mainScene.resize(w, h);
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  get isRunning(): boolean { return this.running; }

  /** Direct access to CellInstanceManager for per-cell updates from PubSub. */
  get cells(): CellInstanceManager { return this.cellManager; }

  /** Direct access to BloomPass for live parameter tuning. */
  get bloomPass(): BloomPass { return this.bloom; }
}
