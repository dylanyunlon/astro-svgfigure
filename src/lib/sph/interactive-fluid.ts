/**
 * interactive-fluid.ts — M743: Mouse Splat → Advect → Pressure Fluid Interaction
 *
 * High-level controller that wires DOM mouse/touch/pointer events to the
 * NavierStokesFluid compute pipeline (at-navier-stokes.ts).  Manages the
 * full interaction loop:
 *
 *   pointer-down / pointer-move  →  accumulate splats
 *   requestAnimationFrame        →  splat → step (advect → vorticity →
 *                                   divergence → pressure → gradient)
 *   render callback              →  consumer reads dye / velocity textures
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Design
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  1. **Pointer tracker** — tracks mouse / touch position, converts to
 *     normalised UV [0,1], computes velocity from frame-to-frame delta.
 *     Supports multi-touch (each pointer ID gets its own splat).
 *
 *  2. **Splat queue** — each frame, pending pointer deltas are flushed into
 *     NavierStokesFluid.splat() calls before the simulation step.  This
 *     decouples event frequency from simulation frequency (pointer events
 *     fire at >60 Hz on modern browsers).
 *
 *  3. **Colour palette** — each new pointer gets a deterministic hue from
 *     a golden-ratio HSL sequence (matching AT "mousefluid" dye injection).
 *
 *  4. **Auto-splat** — optional ambient splats when the pointer is stationary
 *     (idle ripple / breathing effect, like AT idle-state mouse fluid).
 *
 *  5. **Lifecycle** — attach() / detach() cleanly bind/unbind DOM listeners;
 *     start() / stop() control the rAF loop; destroy() releases everything.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const fluid = createNavierStokesFluid(device);
 *   const ctrl  = new InteractiveFluid(fluid, canvas, {
 *     onRender(fluid) {
 *       // bind fluid.dyeTextureView / fluid.velocityTextureView
 *       // to your render pass …
 *     },
 *   });
 *   ctrl.start();
 *   // …later…
 *   ctrl.destroy();
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Sources / lineage
 * ─────────────────────────────────────────────────────────────────────────────
 *  • at-navier-stokes.ts — NavierStokesFluid compute pipeline (M715)
 *  • SceneLayoutPresets "mousefluid_scale" / "mousefluid_strength"
 *  • src/lib/shaders/compiled.vs :: fluid-surface.frag (M553)
 *  • upstream/lygia/simulate/simpleAndFastFluid.glsl
 *
 * Research: xiaodi #M743 — cell-pubsub-loop
 * ─────────────────────────────────────────────────────────────────────────────
 */




// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────


import {

<<<<<<< HEAD
// [orphan-precise] /** Callback invoked once per frame after the simulation step. */
=======
/** Callback invoked once per frame after the simulation step. */




  NavierStokesFluid,
  NavierStokesParams,
  NavierStokesSplat,
  createNavierStokesFluid,
} from './at-navier-stokes';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export type FluidRenderCallback = (fluid: NavierStokesFluid) => void;

/** Configuration for InteractiveFluid. */
export interface InteractiveFluidOptions {
  /**
   * Navier-Stokes solver parameters (forwarded to NavierStokesFluid).
   * Can also be updated at runtime via `updateParams()`.
   */
  params?: NavierStokesParams;

  /**
   * Splat force multiplier — scales the pointer-velocity → fluid impulse.
   * Higher values produce more energetic splats.  Default 5000.
   * Matches AT "mousefluid_strength" preset range.
   */
  splatForce?: number;

  /**
   * Splat radius override (normalised [0,1] UV).  If omitted, uses the
   * NavierStokesParams.splatRadius default (0.012).
   */
  splatRadius?: number;

  /**
   * Enable ambient auto-splats when the pointer is stationary.
   * Creates subtle idle ripples.  Default false.
   */
  autoSplat?: boolean;

  /**
   * Auto-splat strength (when autoSplat is enabled).  Default 200.
   */
  autoSplatStrength?: number;

  /**
   * Auto-splat interval in milliseconds.  Default 100 (10 Hz).
   */
  autoSplatInterval?: number;

  /**
   * Render callback — called once per rAF frame after the simulation
   * step completes.  Use it to bind fluid textures to your render pass.
   */
  onRender?: FluidRenderCallback;

  /**
   * Optional GPUDevice — if provided and no NavierStokesFluid is given
   * to the constructor, the controller will create one internally.
   */
  device?: GPUDevice;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal pointer state
// ─────────────────────────────────────────────────────────────────────────────

interface PointerState {
  /** Pointer ID (from PointerEvent). */
  id: number;
  /** Current normalised UV position [0,1]. */
  x: number;
  y: number;
  /** Previous frame normalised UV position. */
  prevX: number;
  prevY: number;
  /** Hue assigned to this pointer (degrees, 0–360). */
  hue: number;
  /** Whether the pointer moved since last flush. */
  moved: boolean;
  /** Timestamp (ms) of the last splat for auto-splat throttling. */
  lastSplatTime: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour utilities
// ─────────────────────────────────────────────────────────────────────────────

/** Golden-ratio hue step for deterministic palette generation. */
const GOLDEN_ANGLE = 137.508;

/** Convert HSL → RGB tuple [0,1] each. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r: number, g: number, b: number;
  if      (h < 60)  { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else              { r = c; g = 0; b = x; }
  return [r + m, g + m, b + m];
}

// ─────────────────────────────────────────────────────────────────────────────
// InteractiveFluid
// ─────────────────────────────────────────────────────────────────────────────

export class InteractiveFluid {
  /** The underlying NavierStokesFluid compute pipeline. */
  readonly fluid: NavierStokesFluid;

  private readonly canvas: HTMLCanvasElement;
  private readonly opts: Required<
    Pick<
      InteractiveFluidOptions,
      'splatForce' | 'splatRadius' | 'autoSplat' | 'autoSplatStrength' | 'autoSplatInterval'
    >
  >;
  private onRender: FluidRenderCallback | null;

  // ── Pointer tracking ────────────────────────────────────────────────────

  private pointers = new Map<number, PointerState>();
  private nextHue  = 0;

  // ── Splat queue (buffered between pointer events and rAF) ──────────────

  private splatQueue: NavierStokesSplat[] = [];

  // ── rAF loop ────────────────────────────────────────────────────────────

  private rafId: number | null    = null;
  private running                 = false;
  private lastFrameTime           = 0;

  // ── DOM listener refs (for clean detach) ────────────────────────────────

  private readonly boundPointerDown : (e: PointerEvent) => void;
  private readonly boundPointerMove : (e: PointerEvent) => void;
  private readonly boundPointerUp   : (e: PointerEvent) => void;
  private readonly boundPointerLeave: (e: PointerEvent) => void;

  private attached  = false;
  private destroyed = false;

  /** Whether the controller owns the NavierStokesFluid (and must destroy it). */
  private readonly ownsFluid: boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @param fluidOrDevice — either a pre-built NavierStokesFluid or a GPUDevice
   *   (if a device is given, the controller creates the fluid internally).
   * @param canvas — the HTMLCanvasElement that receives pointer events.
   * @param options — configuration.
   */
  constructor(
    fluidOrDevice: NavierStokesFluid | GPUDevice,
    canvas: HTMLCanvasElement,
    options: InteractiveFluidOptions = {},
  ) {
    this.canvas = canvas;

    // Resolve fluid instance
    if (fluidOrDevice instanceof GPUDevice || ('createBuffer' in fluidOrDevice)) {
      // It's a GPUDevice — create fluid internally
      const f = createNavierStokesFluid(fluidOrDevice as GPUDevice, options.params);
      if (!f) throw new Error('InteractiveFluid: failed to create NavierStokesFluid');
      this.fluid     = f;
      this.ownsFluid = true;
    } else {
      this.fluid     = fluidOrDevice as NavierStokesFluid;
      this.ownsFluid = false;
    }

    this.opts = {
      splatForce        : options.splatForce        ?? 5000,
      splatRadius       : options.splatRadius        ?? this.fluid.params.splatRadius,
      autoSplat         : options.autoSplat          ?? false,
      autoSplatStrength : options.autoSplatStrength   ?? 200,
      autoSplatInterval : options.autoSplatInterval   ?? 100,
    };

    this.onRender = options.onRender ?? null;

    // Bind event handlers once (stable refs for removeEventListener)
    this.boundPointerDown  = this.handlePointerDown.bind(this);
    this.boundPointerMove  = this.handlePointerMove.bind(this);
    this.boundPointerUp    = this.handlePointerUp.bind(this);
    this.boundPointerLeave = this.handlePointerLeave.bind(this);

    // Auto-attach DOM listeners
    this.attach();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DOM attachment
  // ─────────────────────────────────────────────────────────────────────────

  /** Bind pointer-event listeners to the canvas. */
  attach(): void {
    if (this.attached || this.destroyed) return;
    this.attached = true;

    const c = this.canvas;
    c.addEventListener('pointerdown',  this.boundPointerDown);
    c.addEventListener('pointermove',  this.boundPointerMove);
    c.addEventListener('pointerup',    this.boundPointerUp);
    c.addEventListener('pointerleave', this.boundPointerLeave);
    c.addEventListener('pointercancel', this.boundPointerUp);

    // Prevent touch-scroll / zoom while interacting with the fluid
    c.style.touchAction = 'none';
  }

  /** Remove pointer-event listeners from the canvas. */
  detach(): void {
    if (!this.attached) return;
    this.attached = false;

    const c = this.canvas;
    c.removeEventListener('pointerdown',  this.boundPointerDown);
    c.removeEventListener('pointermove',  this.boundPointerMove);
    c.removeEventListener('pointerup',    this.boundPointerUp);
    c.removeEventListener('pointerleave', this.boundPointerLeave);
    c.removeEventListener('pointercancel', this.boundPointerUp);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pointer event handlers
  // ─────────────────────────────────────────────────────────────────────────

  private normalisePointer(e: PointerEvent): { nx: number; ny: number } {
    const rect = this.canvas.getBoundingClientRect();
    return {
      nx: (e.clientX - rect.left) / rect.width,
      ny: (e.clientY - rect.top) / rect.height,
    };
  }

  private handlePointerDown(e: PointerEvent): void {
    if (this.destroyed) return;
    this.canvas.setPointerCapture(e.pointerId);

    const { nx, ny } = this.normalisePointer(e);
    const hue = this.nextHue;
    this.nextHue = (this.nextHue + GOLDEN_ANGLE) % 360;

    this.pointers.set(e.pointerId, {
      id   : e.pointerId,
      x    : nx,
      y    : ny,
      prevX: nx,
      prevY: ny,
      hue,
      moved: false,
      lastSplatTime: performance.now(),
    });

    // Immediate splat at touch-down position (zero-velocity impulse with dye)
    this.enqueueSplat(nx, ny, 0, 0, hue);
  }

  private handlePointerMove(e: PointerEvent): void {
    if (this.destroyed) return;
    const ptr = this.pointers.get(e.pointerId);
    if (!ptr) return;

    const { nx, ny } = this.normalisePointer(e);

    // Store previous position for velocity computation
    ptr.prevX = ptr.x;
    ptr.prevY = ptr.y;
    ptr.x     = nx;
    ptr.y     = ny;
    ptr.moved = true;

    // Compute pointer velocity in normalised UV / second
    // (pointer events arrive at device Hz; we scale by splatForce instead)
    const dx = nx - ptr.prevX;
    const dy = ny - ptr.prevY;

    this.enqueueSplat(nx, ny, dx * this.opts.splatForce, dy * this.opts.splatForce, ptr.hue);
  }

  private handlePointerUp(e: PointerEvent): void {
    this.pointers.delete(e.pointerId);
  }

  private handlePointerLeave(e: PointerEvent): void {
    this.pointers.delete(e.pointerId);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Splat queue
  // ─────────────────────────────────────────────────────────────────────────

  private enqueueSplat(
    x: number, y: number,
    vx: number, vy: number,
    hue: number,
  ): void {
    const color = hslToRgb(hue, 0.75, 0.55);
    this.splatQueue.push({ x, y, vx, vy, color });
  }

  /**
   * Manually inject a splat (for external callers — e.g. cell-pubsub events,
   * procedural animations, physics impulse bridges).
   */
  addSplat(splat: NavierStokesSplat): void {
    if (this.destroyed) return;
    this.splatQueue.push(splat);
  }

  /**
   * Inject a splat using simple parameters (convenience wrapper).
   *
   * @param x   — normalised X [0,1]
   * @param y   — normalised Y [0,1]
   * @param vx  — velocity impulse X
   * @param vy  — velocity impulse Y
   * @param rgb — dye colour [r,g,b] each in [0,1]
   */
  addSplatXY(
    x: number, y: number,
    vx: number, vy: number,
    rgb: [number, number, number] = [1, 1, 1],
  ): void {
    if (this.destroyed) return;
    this.splatQueue.push({ x, y, vx, vy, color: rgb });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Auto-splat (idle ripple)
  // ─────────────────────────────────────────────────────────────────────────

  private processAutoSplats(now: number): void {
    if (!this.opts.autoSplat) return;

    for (const ptr of this.pointers.values()) {
      if (ptr.moved) {
        ptr.moved = false;
        ptr.lastSplatTime = now;
        continue;
      }

      // Pointer is stationary — inject ambient auto-splat at throttled rate
      if (now - ptr.lastSplatTime >= this.opts.autoSplatInterval) {
        ptr.lastSplatTime = now;

        // Gentle circular ripple: rotating velocity vector
        const angle = now * 0.003;
        const str   = this.opts.autoSplatStrength;
        this.enqueueSplat(
          ptr.x, ptr.y,
          Math.cos(angle) * str,
          Math.sin(angle) * str,
          ptr.hue,
        );
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // rAF loop
  // ─────────────────────────────────────────────────────────────────────────

  /** Start the simulation + render loop. */
  start(): void {
    if (this.running || this.destroyed) return;
    this.running       = true;
    this.lastFrameTime = performance.now();
    this.tick(this.lastFrameTime);
  }

  /** Pause the simulation + render loop (state is preserved). */
  stop(): void {
    this.running = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private tick = (now: number): void => {
    if (!this.running || this.destroyed) return;

    // ── Delta time (clamped to avoid spiral-of-death) ──────────────────
    const dt = Math.min((now - this.lastFrameTime) * 0.001, 0.05);
    this.lastFrameTime = now;

    // Update fluid dt if significantly different from default
    if (Math.abs(dt - this.fluid.params.dt) > 0.002) {
      this.fluid.updateParams({ dt });
    }

    // ── Auto-splats for idle pointers ──────────────────────────────────
    this.processAutoSplats(now);

    // ── Flush splat queue into the GPU pipeline ────────────────────────
    this.flushAndStep();

    // ── Render callback ────────────────────────────────────────────────
    if (this.onRender) {
      this.onRender(this.fluid);
    }

    // ── Schedule next frame ────────────────────────────────────────────
    this.rafId = requestAnimationFrame(this.tick);
  };

  /**
   * Flushes all pending splats and runs one simulation step.
   * Can be called manually for external frame-loop integration
   * (when you don't want the built-in rAF loop).
   */
  flushAndStep(): void {
    if (this.destroyed) return;

    const device  = (this.fluid as any).device as GPUDevice;
    const encoder = device.createCommandEncoder({ label: 'interactive-fluid:frame' });

    // ── Encode all queued splats ──────────────────────────────────────
    const splats = this.splatQueue;
    this.splatQueue = [];

    for (const s of splats) {
      this.fluid.splat(encoder, s);
    }

    // ── Full simulation step: advect → vorticity → divergence →
    //    pressure (Jacobi ×N) → gradient subtract ──────────────────────
    this.fluid.step(encoder);

    // ── Submit ────────────────────────────────────────────────────────
    device.queue.submit([encoder.finish()]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Runtime parameter updates
  // ─────────────────────────────────────────────────────────────────────────

  /** Update splat-interaction parameters at runtime. */
  updateInteraction(partial: Partial<Pick<
    InteractiveFluidOptions,
    'splatForce' | 'splatRadius' | 'autoSplat' | 'autoSplatStrength' | 'autoSplatInterval'
  >>): void {
    if (partial.splatForce        !== undefined) this.opts.splatForce        = partial.splatForce;
    if (partial.splatRadius       !== undefined) this.opts.splatRadius       = partial.splatRadius;
    if (partial.autoSplat         !== undefined) this.opts.autoSplat         = partial.autoSplat;
    if (partial.autoSplatStrength !== undefined) this.opts.autoSplatStrength = partial.autoSplatStrength;
    if (partial.autoSplatInterval !== undefined) this.opts.autoSplatInterval = partial.autoSplatInterval;

    // Propagate splatRadius to the fluid params
    if (partial.splatRadius !== undefined) {
      this.fluid.updateParams({ splatRadius: partial.splatRadius });
    }
  }

  /** Update Navier-Stokes solver parameters at runtime. */
  updateParams(partial: Partial<NavierStokesParams>): void {
    this.fluid.updateParams(partial);
  }

  /** Replace the render callback. */
  setRenderCallback(cb: FluidRenderCallback | null): void {
    this.onRender = cb;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Random splat burst (for demos / idle / ambient)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Injects `count` random splats across the grid.
   * Useful for initial visual splash, demos, or screen-saver mode.
   */
  randomSplats(count: number = 10): void {
    if (this.destroyed) return;

    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 200 + Math.random() * 600;
      const hue   = Math.random() * 360;
      const color = hslToRgb(hue, 0.7, 0.5);

      this.splatQueue.push({
        x    : 0.1 + Math.random() * 0.8,
        y    : 0.1 + Math.random() * 0.8,
        vx   : Math.cos(angle) * speed,
        vy   : Math.sin(angle) * speed,
        color,
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Read-only accessors (convenience)
  // ─────────────────────────────────────────────────────────────────────────

  /** Current velocity texture view (XY=vel, W=curl). */
  get velocityTextureView(): GPUTextureView {
    return this.fluid.velocityTextureView;
  }

  /** Current dye / colour texture view (RGB=colour, W=density). */
  get dyeTextureView(): GPUTextureView {
    return this.fluid.dyeTextureView;
  }

  /** Current pressure texture view. */
  get pressureTextureView(): GPUTextureView {
    return this.fluid.pressureTextureView;
  }

  /** Whether the simulation loop is running. */
  get isRunning(): boolean {
    return this.running;
  }

  /** Number of active pointers currently tracked. */
  get activePointers(): number {
    return this.pointers.size;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  /** Stop loop, detach listeners, release GPU resources if owned. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    this.stop();
    this.detach();
    this.pointers.clear();
    this.splatQueue.length = 0;
    this.onRender = null;

    if (this.ownsFluid) {
      this.fluid.destroy();
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper — matches project createXxx() convention
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates an InteractiveFluid controller.
 *
 * @param fluidOrDevice — a pre-built NavierStokesFluid or a GPUDevice.
 * @param canvas        — the canvas element that receives pointer events.
 * @param options       — interaction and simulation options.
 * @returns InteractiveFluid instance, or null if creation fails.
 */
export function createInteractiveFluid(
  fluidOrDevice: NavierStokesFluid | GPUDevice | null | undefined,
  canvas: HTMLCanvasElement | null | undefined,
  options?: InteractiveFluidOptions,
): InteractiveFluid | null {
  if (!fluidOrDevice || !canvas) return null;
  try {
    return new InteractiveFluid(fluidOrDevice, canvas, options);
  } catch {
    return null;
  }
}
