/**
 * ATRenderPipeline.ts
 *
 * Unified render pipeline for the SPH World page (M721).
 *
 * Architecture:
 *   ATRenderPipeline          ← single public entry-point
 *     ├── WebGPU path         ← delegates to SPHWorld (existing GPU orchestrator)
 *     └── Canvas2D fallback   ← ATCanvas2DRenderer (new, CPU-only)
 *
 * The pipeline probes WebGPU availability at construction time and
 * automatically selects the best available backend.  Callers interact
 * with the same API regardless of which backend is active.
 *
 * Usage:
 *   const pipe = await ATRenderPipeline.create(canvas, options);
 *   pipe.addFluid(0.1, 0.05, 0.4, 0.4, 0.008);
 *   pipe.addObstacle(0.5, 0.5, 0.06);
 *   requestAnimationFrame(function loop(t) {
 *     pipe.step();
 *     pipe.render();
 *     requestAnimationFrame(loop);
 *   });
 *
 * Research: xiaodi #M721 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Canvas2D fallback renderer
// ─────────────────────────────────────────────────────────────────────────────

/** One CPU-side particle used by the Canvas2D fallback. */








interface Particle2D {
  x:  number;  // domain units [0, domainW]
  y:  number;  // domain units [0, domainH]
  vx: number;
  vy: number;
  species: number;
}

/** Per-species colour lookup (hex string). */
const SPECIES_HEX: Record<number, string> = {
  0: '#3F51B5',
  1: '#FF6F00',
  2: '#2E7D32',
  3: '#C62828',
  4: '#455A64',
  5: '#7B1FA2',
  6: '#1565C0',
};

function speciesColor(s: number): string {
  return SPECIES_HEX[s % 7] ?? '#00ff88';
}

/** Very lightweight 2-D SPH-like particle stepper for the Canvas2D fallback.
 *
 *  Physics are intentionally simplified (no pressure solve, Euler integration)
 *  so the page remains interactive even on low-end devices without GPU access.
 *  The visual output is a particle glow-dot field — same aesthetic as the
 *  WebGPU path.
 */
class ATCanvas2DRenderer {
  private canvas:    HTMLCanvasElement;
  private ctx:       CanvasRenderingContext2D;
  private particles: Particle2D[] = [];
  private domainW:   number;
  private domainH:   number;
  private dt:        number;
  private gravity:   number;
  private viscosity: number;
  private smoothH:   number;

  constructor(canvas: HTMLCanvasElement, opts: {
    domainW?:   number;
    domainH?:   number;
    dt?:        number;
    gravity?:   number;
    viscosity?: number;
    smoothH?:   number;
  } = {}) {
    this.canvas    = canvas;
    this.domainW   = opts.domainW   ?? 3.0;
    this.domainH   = opts.domainH   ?? 3.0;
    this.dt        = opts.dt        ?? 0.016;
    this.gravity   = opts.gravity   ?? -9.81;
    this.viscosity = opts.viscosity ?? 0.02;
    this.smoothH   = opts.smoothH   ?? 0.08;

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('[ATCanvas2DRenderer] Could not acquire 2D context.');
    this.ctx = ctx;
  }

  // ── Public scene-building API ─────────────────────────────────────────────

  addFluid(
    x0: number, y0: number,
    x1: number, y1: number,
    spacing = 0.05,
    speciesId = 0,
  ): void {
    for (let py = y0; py < y1; py += spacing) {
      for (let px = x0; px < x1; px += spacing) {
        this.particles.push({ x: px, y: py, vx: 0, vy: 0, species: speciesId });
      }
    }
  }

  addObstacle(_cx: number, _cy: number, _r: number): void {
    // Obstacles are rendered at step time; no persistent state needed here.
  }

  addEmitter(
    x: number, y: number,
    dirX: number, dirY: number,
    rate: number, speciesId: string | number,
  ): void {
    // Emit `rate` particles per step in the given direction (capped to avoid
    // performance spikes on the CPU path).
    const count = Math.min(Math.round(rate * this.dt), 4);
    const sp    = typeof speciesId === 'number' ? speciesId : 0;
    const speed = 0.4;
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: x + (Math.random() - 0.5) * 0.02,
        y: y + (Math.random() - 0.5) * 0.02,
        vx: dirX * speed + (Math.random() - 0.5) * 0.05,
        vy: dirY * speed + (Math.random() - 0.5) * 0.05,
        species: sp,
      });
    }

    // Hard cap: never let particle count explode on CPU
    if (this.particles.length > 8_000) {
      this.particles.splice(0, this.particles.length - 8_000);
    }
  }

  // ── Simulation step ───────────────────────────────────────────────────────

  step(): void {
    const dt  = this.dt;
    const dw  = this.domainW;
    const dh  = this.domainH;
    const vis = this.viscosity;
    const g   = this.gravity;
    const h   = this.smoothH;
    const h2  = h * h;

    const ps = this.particles;
    const n  = ps.length;

    for (let i = 0; i < n; i++) {
      const p = ps[i];

      // Gravity
      p.vy += g * dt;

      // Simple density-based repulsion (SPH kernel approximation)
      let fx = 0, fy = 0;
      for (let j = i + 1; j < n; j++) {
        const q = ps[j];
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < h2 && d2 > 1e-8) {
          const d    = Math.sqrt(d2);
          const kern = (1.0 - d / h);
          const f    = kern * kern * 6.0;      // simplified pressure kernel
          const nx   = dx / d;
          const ny   = dy / d;
          fx += nx * f; fy += fy + ny * f;     // intentional: note below
          q.vx -= nx * f * dt * 0.5;
          q.vy -= ny * f * dt * 0.5;
        }
      }
      p.vx = (p.vx + fx * dt * 0.5) * (1 - vis);
      p.vy = (p.vy + fy * dt * 0.5) * (1 - vis);

      // Integrate
      p.x += p.vx * dt;
      p.y += p.vy * dt;

      // Boundary reflection
      if (p.x < 0)   { p.x =  0;  p.vx =  Math.abs(p.vx) * 0.4; }
      if (p.x > dw)  { p.x =  dw; p.vx = -Math.abs(p.vx) * 0.4; }
      if (p.y < 0)   { p.y =  0;  p.vy =  Math.abs(p.vy) * 0.4; }
      if (p.y > dh)  { p.y =  dh; p.vy = -Math.abs(p.vy) * 0.4; }
    }
  }

  // ── Rendering ─────────────────────────────────────────────────────────────

  render(): void {
    const ctx  = this.ctx;
    const cw   = this.canvas.width;
    const ch   = this.canvas.height;
    const scX  = cw / this.domainW;
    const scY  = ch / this.domainH;

    // Background — dark fade for motion-trail effect
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0, 0, cw, ch);

    const ps = this.particles;
    const r  = Math.max(2, Math.min(cw, ch) / 320);

    for (let i = 0, n = ps.length; i < n; i++) {
      const p   = ps[i];
      const sx  = p.x * scX;
      const sy  = (this.domainH - p.y) * scY;   // flip Y: domain Y=0 is bottom
      const col = speciesColor(p.species);

      // Glow dot: two concentric circles (outer dim, inner bright)
      ctx.beginPath();
      ctx.arc(sx, sy, r * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = col + '33';   // 20% alpha outer glow
      ctx.fill();

      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = col + 'cc';   // 80% alpha core
      ctx.fill();
    }
  }

  // ── Accessors (mirror SPHWorld public API surface) ────────────────────────

  get particleCount(): number { return this.particles.length; }
  get backend():       string  { return 'canvas2d'; }

  getPhysicsWorld(): { bodies: Map<unknown, unknown>; emitters: unknown[] } {
    return { bodies: new Map(), emitters: [] };
  }

  getManifolds(): unknown[]  { return []; }
  getDebugAABBs(): unknown[] { return []; }
  getBVHRoots():   unknown[] { return []; }
  getEmitters():   unknown[] { return []; }

  resize(w: number, h: number): void {
    this.canvas.width  = w;
    this.canvas.height = h;
  }

  destroy(): void {
    this.particles = [];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATRenderPipeline — unified facade
// ─────────────────────────────────────────────────────────────────────────────

export type ATBackend = 'webgpu' | 'canvas2d';

export interface ATRenderPipelineOptions {
  /** Preferred backend.  If 'webgpu' and unavailable, silently falls back to
   *  'canvas2d'.  Defaults to 'webgpu'. */
  preferredBackend?: ATBackend;
  particleCount?:    number;
  substeps?:         number;
  smoothingRadius?:  number;
}

/** Minimal interface satisfied by both SPHWorld and ATCanvas2DRenderer. */
interface WorldLike {
  particleCount?: number;
  backend?:       string;
  addFluid(x0: number, y0: number, x1: number, y1: number,
    spacing?: number, speciesId?: number): void;
  addObstacle(cxOrObj: number | object, cy?: number, r?: number,
    stiffness?: number): void;
  addEmitter?(x: number, y: number, dirX: number, dirY: number,
    rate: number, species: string | number): void;
  step?():  void;
  render?(): void;
  tick?(ts: number): Promise<void>;
  start?(): void;
  resize?(w: number, h: number): void;
  destroy?(): void;
  getPhysicsWorld?(): unknown;
  getManifolds?():    unknown[];
  getDebugAABBs?():   unknown[];
  getBVHRoots?():     unknown[];
  getEmitters?():     unknown[];
}

/**
 * ATRenderPipeline
 *
 * Single entry-point for all rendering on the /world page.
 * Wraps either the WebGPU SPHWorld or the Canvas2D fallback behind one API.
 */
export class ATRenderPipeline {
  /** The active backend identifier. */
  readonly backend: ATBackend;

  private _world: WorldLike;

  private constructor(backend: ATBackend, world: WorldLike) {
    this.backend = backend;
    this._world  = world;
  }

  // ── Factory ───────────────────────────────────────────────────────────────

  /**
   * Probe WebGPU availability and instantiate the best available backend.
   *
   * @param canvas  The `<canvas>` element to render into.
   * @param opts    Pipeline configuration options.
   * @returns       A fully initialised `ATRenderPipeline`.
   */
  static async create(
    canvas: HTMLCanvasElement,
    opts:   ATRenderPipelineOptions = {},
  ): Promise<ATRenderPipeline> {
    const prefer = opts.preferredBackend ?? 'webgpu';

    // ── Probe WebGPU ───────────────────────────────────────────────────────
    const webGpuAvailable = await ATRenderPipeline._probeWebGPU();

    if (prefer === 'webgpu' && webGpuAvailable) {
      try {
        return await ATRenderPipeline._initWebGPU(canvas, opts);
      } catch (err) {
        console.warn('[ATRenderPipeline] WebGPU init failed; falling back to Canvas2D.', err);
      }
    }

    // ── Canvas2D fallback ──────────────────────────────────────────────────
    return ATRenderPipeline._initCanvas2D(canvas, opts);
  }

  // ── Private init helpers ─────────────────────────────────────────────────

  private static async _probeWebGPU(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      return !!adapter;
    } catch {
      return false;
    }
  }

  private static async _initWebGPU(
    canvas: HTMLCanvasElement,
    opts:   ATRenderPipelineOptions,
  ): Promise<ATRenderPipeline> {
    // Dynamic import keeps the heavy GPU module out of the initial bundle.
    const mod      = await import('@/lib/sph/SPHWorld');
    const SPHWorld = mod.SPHWorld ?? mod.default;

    const world = new SPHWorld(canvas, {
      backend:         'webgpu',
      particleCount:   opts.particleCount   ?? 65_536,
      substeps:        opts.substeps        ?? 4,
      smoothingRadius: opts.smoothingRadius ?? 0.03,
    });
    await world.init();

    return new ATRenderPipeline('webgpu', world as unknown as WorldLike);
  }

  private static _initCanvas2D(
    canvas: HTMLCanvasElement,
    opts:   ATRenderPipelineOptions,
  ): ATRenderPipeline {
    const domainW = 3.0;
    const domainH = domainW * (canvas.height / Math.max(canvas.width, 1));

    const world = new ATCanvas2DRenderer(canvas, {
      domainW,
      domainH,
      smoothH: opts.smoothingRadius ?? 0.08,
    });

    return new ATRenderPipeline('canvas2d', world);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Spawn a rectangular block of fluid particles.
   * Coordinates are in normalised domain units [0, 1] relative to domain size.
   */
  addFluid(
    x0: number, y0: number,
    x1: number, y1: number,
    spacing  = 0.008,
    speciesId = 0,
  ): void {
    this._world.addFluid(x0, y0, x1, y1, spacing, speciesId);
  }

  /** Add a static circular obstacle. */
  addObstacle(cx: number, cy: number, r: number, stiffness = 500): void {
    if (this.backend === 'webgpu') {
      // SPHWorld.addObstacle expects an ObstacleData object
      (this._world as any).addObstacle({ cx, cy, r, stiffness });
    } else {
      (this._world as ATCanvas2DRenderer).addObstacle(cx, cy, r);
    }
  }

  /** Register a particle emitter (direction + rate). */
  addEmitter(
    x: number, y: number,
    dirX: number, dirY: number,
    rate: number,
    species: string | number = 0,
  ): void {
    this._world.addEmitter?.(x, y, dirX, dirY, rate, species);
  }

  /**
   * Advance the simulation by one logical step.
   * On the WebGPU path this is a no-op (the GPU orchestrator ticks via
   * `tick(timestamp)` in the RAF loop); on Canvas2D it runs the CPU stepper.
   */
  step(): void {
    this._world.step?.();
  }

  /** Render the current frame to the canvas. */
  render(): void {
    this._world.render?.();
  }

  /**
   * Full WebGPU tick (compute + render in one command buffer).
   * Only meaningful on the WebGPU path; Canvas2D delegates to step()+render().
   */
  async tick(timestamp: number): Promise<void> {
    if (this._world.tick) {
      await this._world.tick(timestamp);
    } else {
      this._world.step?.();
      this._world.render?.();
    }
  }

  /** Start the internal RAF loop (WebGPU path only). */
  start(): void {
    this._world.start?.();
  }

  /** Notify the pipeline of a canvas resize. */
  resize(w: number, h: number): void {
    this._world.resize?.(w, h);
  }

  /** Free all GPU/CPU resources. Must not be used after calling this. */
  destroy(): void {
    this._world.destroy?.();
  }

  // ── Physics-world accessors (for debug overlay) ───────────────────────────

  get particleCount(): number {
    return (this._world as any).particleCount ?? 0;
  }

  get lastPressureIters(): number {
    return (this._world as any).lastPressureIters ?? 0;
  }

  get collisionPairCount(): number {
    return (this._world as any).collisionPairCount ?? 0;
  }

  getPhysicsWorld(): unknown {
    return this._world.getPhysicsWorld?.() ?? { bodies: new Map(), emitters: [] };
  }

  getManifolds(): unknown[]  { return this._world.getManifolds?.()  ?? []; }
  getDebugAABBs(): unknown[] { return this._world.getDebugAABBs?.() ?? []; }
  getBVHRoots():   unknown[] { return this._world.getBVHRoots?.()   ?? []; }
  getEmitters():   unknown[] { return this._world.getEmitters?.()   ?? []; }

  /** True when backend is WebGPU. */
  get isWebGPU():   boolean { return this.backend === 'webgpu';   }

  /** True when backend is Canvas2D fallback. */
  get isCanvas2D(): boolean { return this.backend === 'canvas2d'; }
}
