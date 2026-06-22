/**
 * curl-particle-field.ts — M705
 *
 * AT curl.glsl analytic derivatives driving cell decoration particles.
 *
 * ─── Architecture ────────────────────────────────────────────────────────────
 * Each visible cell in the scene is adorned with a halo of floating point
 * sprites whose motion is governed by the same analytic curl-noise potential
 * functions that drive Active Theory's ProtonAntimatter GPGPU system
 * (src/lib/particle/CurlNoise.ts).
 *
 * Unlike the GPGPU ping-pong path (ParticleSystem.ts / CurlNoise.frag), this
 * renderer runs entirely on the CPU using the TypeScript analytic derivatives
 * from CurlNoise.ts — `curlNoise3D(x, y, z)`.  The analytic approach computes
 * 36 cos() calls instead of the 96-call finite-difference version used by the
 * lygia numerical curl in CurlNoise.frag, making it ~3× cheaper per particle
 * when the particle count is low enough that a full GPGPU pass would be wasteful
 * (< ~8 000 particles total across all cells).
 *
 * For heavier scenes the class exposes `CurlParticleField.useGPU = true` to
 * delegate the simulation step to ParticleSystem.ts (TODO: wired externally).
 *
 * ─── Curl noise derivation ────────────────────────────────────────────────────
 * Three sinusoidal potential functions P1/P2/P3 (8 terms each, same constants
 * as curl.glsl compiled.vs Active Theory reference):
 *
 *   curl.x = ∂P3/∂y − ∂P2/∂z
 *   curl.y = ∂P1/∂z − ∂P3/∂x
 *   curl.z = ∂P2/∂x − ∂P1/∂y
 *
 * All six partial derivatives are exact cos() expansions (no finite differences).
 * Reference: src/lib/particle/CurlNoise.ts
 *
 * ─── Cell pub-sub integration ─────────────────────────────────────────────────
 * CurlParticleField subscribes to cell lifecycle events via the DOM CustomEvent
 * bus used throughout the cell-pubsub-loop branch.  When a cell is registered,
 * updated, or removed, the corresponding particle emitter is created, recycled,
 * or destroyed without touching the rest of the scene.
 *
 *   document.dispatchEvent(new CustomEvent('cell:register',   { detail: desc }));
 *   document.dispatchEvent(new CustomEvent('cell:update',     { detail: desc }));
 *   document.dispatchEvent(new CustomEvent('cell:unregister', { detail: { cell_id } }));
 *
 * ─── PixiJS rendering ────────────────────────────────────────────────────────
 * Each decoration particle is a Particle instance inside a per-cell
 * ParticleContainer.  One ParticleContainer per cell = one draw call per cell.
 * Particle tint is derived from the species color palette (SPECIES_TINT).
 *
 * ─── Upstream references ─────────────────────────────────────────────────────
 *   src/lib/particle/CurlNoise.ts          — analytic curlNoise3D()
 *   src/lib/particle/CurlNoise.frag        — GLSL reference for the same field
 *   upstream/lygia/generative/curl.glsl    — LYGIA curl definition (finite diff)
 *   src/lib/renderers/proton-particles.ts  — PixiJS particle rendering pattern
 *   src/lib/renderers/pixi-cell-renderer.ts — CellDescriptor / SPECIES_TINT
 *
 * Research: xiaodi #M705 — cell-pubsub-loop
 */

import { ParticleContainer, Particle, Texture, Graphics } from 'pixi.js';
import type { Application } from 'pixi.js';
import { curlNoise3D } from '../particle/CurlNoise';

// ── Species tint palette (mirrors pixi-cell-renderer.ts / proton-particles.ts) ─

const SPECIES_TINT: Record<string, number> = {
  'cil-eye':         0x7986CB,
  'cil-vector':      0x81C784,
  'cil-bolt':        0xFFCC80,
  'cil-plus':        0xF48FB1,
  'cil-arrow-right': 0xB0BEC5,
  'cil-filter':      0xCE93D8,
  'cil-code':        0x80CBC4,
  'cil-layers':      0x90CAF9,
  'cil-loop':        0xFFE082,
  'cil-graph':       0xB0BEC5,
};

function speciesTint(species: string): number {
  return SPECIES_TINT[species] ?? 0xffffff;
}

// ── Shared dot texture (4×4 soft radial circle, generated once) ──────────────

let _dotTex: Texture | null = null;

function getDotTexture(): Texture {
  if (_dotTex) return _dotTex;
  try {
    const g = new Graphics();
    g.circle(4, 4, 4);
    g.fill({ color: 0xffffff, alpha: 1 });
    // @ts-ignore — generateTexture on PixiJS v8 Graphics
    _dotTex = g.generateTexture();
    g.destroy();
  } catch {
    _dotTex = Texture.WHITE;
  }
  return _dotTex!;
}

// ── Configuration ─────────────────────────────────────────────────────────────

/**
 * Per-field configuration.  All values have sensible defaults derived from
 * AT UIL params: uCurlNoiseSpeed=5, uCurlNoiseScale=2, uCurlTimeScale=1.
 */
export interface CurlParticleFieldConfig {
  /**
   * Number of decoration particles spawned per registered cell.
   * AT WorkDetailParticles analogue.  Default: 64.
   */
  particlesPerCell?: number;

  /**
   * AT: uCurlNoiseScale — spatial frequency multiplier applied to the cell's
   * normalised world position before sampling the curl field.  Default: 2.0.
   */
  curlScale?: number;

  /**
   * AT: uCurlNoiseSpeed — velocity scale applied to the curl direction each
   * tick.  Default: 5.0 (WorkDetailParticles value).
   */
  curlSpeed?: number;

  /**
   * AT: uCurlTimeScale — multiplier on `time` when building the Z-axis drift
   * component of the sample position.  Default: 1.0.
   */
  curlTimeScale?: number;

  /**
   * Radius of the decoration cloud around the cell centre, in pixels.
   * Particles respawn uniformly within this disc.  Default: 60.
   */
  spawnRadius?: number;

  /**
   * Particle life span in seconds.  After expiry the particle respawns at a
   * new random offset from the cell centre.  Default: 2.0.
   */
  particleLife?: number;

  /**
   * Base alpha for decoration particles (0–1).  Actual per-particle alpha is
   * modulated by life [0, 1] so particles fade out before respawn.
   * Default: 0.55.
   */
  baseAlpha?: number;

  /**
   * Base scale (size) for each particle sprite.  Default: 0.35.
   */
  baseScale?: number;

  /**
   * When true, the field is paused and tick() is a no-op.  Default: false.
   */
  paused?: boolean;
}

// ── Cell descriptor (minimal subset from pixi-cell-renderer.ts) ───────────────

export interface CellDecorDesc {
  cell_id: string;
  species: string;
  /** Cell top-left + dimensions in canvas pixels. */
  bbox: { x: number; y: number; w: number; h: number };
}

// ── Internal per-particle state ───────────────────────────────────────────────

interface DecorParticle {
  /** PixiJS Particle — position / alpha / scale are mutated each tick. */
  px:   Particle;
  /** Current world-space position (canvas px). */
  x:    number;
  y:    number;
  /** Life remaining in seconds [0, maxLife]. */
  life: number;
  /** Max life (seconds) — randomised per particle. */
  maxLife: number;
  /** Initial spawn offset from cell centre (px). */
  ox: number;
  oy: number;
}

// ── Per-cell emitter ──────────────────────────────────────────────────────────

interface CellEmitter {
  desc:      CellDecorDesc;
  container: ParticleContainer;
  particles: DecorParticle[];
  /** Centre of the cell in canvas pixels. */
  cx: number;
  cy: number;
}

// ── CurlParticleField ─────────────────────────────────────────────────────────

/**
 * CurlParticleField — attaches a decoration particle cloud to every registered
 * cell; drives particle motion via AT curl.glsl analytic derivatives.
 *
 * Usage:
 * ```ts
 * const field = new CurlParticleField(app, { particlesPerCell: 48, curlSpeed: 5 });
 *
 * // Register cells (also wired to DOM CustomEvents automatically):
 * field.register({ cell_id: 'attn_0', species: 'cil-eye', bbox: { x:120, y:80, w:140, h:60 } });
 *
 * // In the PixiJS Ticker (or requestAnimationFrame):
 * app.ticker.add((ticker) => field.tick(ticker.deltaMS / 1000));
 *
 * // Cleanup:
 * field.dispose();
 * ```
 */
export class CurlParticleField {
  // ── Public control flag ───────────────────────────────────────────────────

  /**
   * When true, delegate particle simulation to ParticleSystem.ts GPGPU path.
   * Currently a forward-compatibility stub — actual GPU delegation is wired
   * externally by the pipeline that owns both this class and ParticleSystem.
   */
  public useGPU = false;

  // ── Private state ─────────────────────────────────────────────────────────

  private readonly app:   Application;
  private readonly cfg:   Required<CurlParticleFieldConfig>;
  private readonly cells: Map<string, CellEmitter> = new Map();
  private time = 0;

  // DOM event handlers (stored for cleanup)
  private readonly _onRegister:   (e: Event) => void;
  private readonly _onUpdate:     (e: Event) => void;
  private readonly _onUnregister: (e: Event) => void;

  // ── Constructor ───────────────────────────────────────────────────────────

  constructor(app: Application, config: CurlParticleFieldConfig = {}) {
    this.app = app;
    this.cfg = {
      particlesPerCell: config.particlesPerCell ?? 64,
      curlScale:        config.curlScale        ?? 2.0,
      curlSpeed:        config.curlSpeed        ?? 5.0,
      curlTimeScale:    config.curlTimeScale    ?? 1.0,
      spawnRadius:      config.spawnRadius      ?? 60,
      particleLife:     config.particleLife     ?? 2.0,
      baseAlpha:        config.baseAlpha        ?? 0.55,
      baseScale:        config.baseScale        ?? 0.35,
      paused:           config.paused           ?? false,
    };

    // ── Wire DOM pub-sub events ─────────────────────────────────────────────
    this._onRegister = (e: Event) => {
      const ce = e as CustomEvent<CellDecorDesc>;
      this.register(ce.detail);
    };
    this._onUpdate = (e: Event) => {
      const ce = e as CustomEvent<CellDecorDesc>;
      this.update(ce.detail);
    };
    this._onUnregister = (e: Event) => {
      const ce = e as CustomEvent<{ cell_id: string }>;
      this.unregister(ce.detail.cell_id);
    };

    document.addEventListener('cell:register',   this._onRegister);
    document.addEventListener('cell:update',     this._onUpdate);
    document.addEventListener('cell:unregister', this._onUnregister);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Register a cell and spawn its decoration particle cloud.
   * Safe to call multiple times — subsequent calls act as `update()`.
   */
  register(desc: CellDecorDesc): void {
    if (this.cells.has(desc.cell_id)) {
      this.update(desc);
      return;
    }

    const cx = desc.bbox.x + desc.bbox.w * 0.5;
    const cy = desc.bbox.y + desc.bbox.h * 0.5;

    const container = new ParticleContainer(this.cfg.particlesPerCell, {
      position:  true,
      alpha:     true,
      scale:     true,
      tint:      true,
    });

    const tint    = speciesTint(desc.species);
    const tex     = getDotTexture();
    const n       = this.cfg.particlesPerCell;
    const r       = this.cfg.spawnRadius;
    const maxLife = this.cfg.particleLife;

    const particles: DecorParticle[] = [];

    for (let i = 0; i < n; i++) {
      // Spawn uniformly in a disc of radius `r` around cell centre.
      const angle = Math.random() * Math.PI * 2;
      const dist  = Math.sqrt(Math.random()) * r;
      const ox    = Math.cos(angle) * dist;
      const oy    = Math.sin(angle) * dist;

      const px = new Particle({
        texture: tex,
        x:       cx + ox,
        y:       cy + oy,
        alpha:   0,
        scaleX:  this.cfg.baseScale,
        scaleY:  this.cfg.baseScale,
        tint,
      });

      // Stagger life so not all particles respawn simultaneously.
      const life = Math.random() * maxLife;

      particles.push({ px, x: cx + ox, y: cy + oy, life, maxLife, ox, oy });
      container.addParticle(px);
    }

    this.app.stage.addChild(container);

    this.cells.set(desc.cell_id, { desc, container, particles, cx, cy });
  }

  /**
   * Update an existing cell's bbox (e.g. after layout reflow) without
   * destroying its live particle cloud.
   */
  update(desc: CellDecorDesc): void {
    const emitter = this.cells.get(desc.cell_id);
    if (!emitter) {
      this.register(desc);
      return;
    }

    emitter.desc = desc;
    emitter.cx   = desc.bbox.x + desc.bbox.w * 0.5;
    emitter.cy   = desc.bbox.y + desc.bbox.h * 0.5;

    // Update particle tint if species changed.
    const tint = speciesTint(desc.species);
    for (const p of emitter.particles) {
      p.px.tint = tint;
    }
  }

  /**
   * Remove a cell and destroy its particle cloud.
   */
  unregister(cellId: string): void {
    const emitter = this.cells.get(cellId);
    if (!emitter) return;

    this.app.stage.removeChild(emitter.container);
    emitter.container.destroy({ children: true });
    this.cells.delete(cellId);
  }

  /**
   * Advance the simulation by `dt` seconds.
   *
   * For each registered cell:
   *   1. Sample the analytic curl field at the particle's current position.
   *   2. Integrate position: pos += curlDir * curlSpeed * dt.
   *   3. Decrement life; if expired, respawn at a fresh random offset from
   *      the cell centre.
   *   4. Write new position / alpha to the PixiJS Particle.
   *
   * @param dt - Frame delta in seconds (use ticker.deltaMS / 1000).
   */
  tick(dt: number): void {
    if (this.cfg.paused) return;

    this.time += dt;

    const { curlScale, curlSpeed, curlTimeScale, spawnRadius, baseAlpha, baseScale } = this.cfg;

    for (const emitter of this.cells.values()) {
      const { cx, cy, particles } = emitter;

      for (const p of particles) {
        // ── 1. Build the curl sample coordinate ──────────────────────────
        //
        // AT curl.glsl analytic derivatives: potential functions P1/P2/P3 are
        // parameterised in 3-D space.  We map the 2-D canvas position to the
        // XY axes and use time (scaled by uCurlTimeScale) as the Z drift axis,
        // matching the CurlNoise.frag approach:
        //
        //   curlSample = pos * uCurlNoiseScale * 0.1 + vec3(uTime * uCurlTimeScale * 0.1)
        //
        // The 0.1 damping keeps the input in a reasonable range for the
        // sinusoidal potentials whose frequencies are in the [1.2, 7.0] range.

        const sx = p.x * curlScale * 0.1 + this.time * curlTimeScale * 0.1;
        const sy = p.y * curlScale * 0.1 + this.time * curlTimeScale * 0.1;
        const sz = this.time * curlTimeScale * 0.1;

        // ── 2. Analytic curl (AT curl.glsl, 36 cos() calls) ──────────────
        const [cvx, cvy] = curlNoise3D(sx, sy, sz);

        // ── 3. Integrate position ─────────────────────────────────────────
        //
        // AT:  pos += curlForce * uCurlNoiseSpeed * 0.01 * uDelta
        // Here uDelta≈1.0 at 60 fps.  We use dt (seconds) * 60 as the
        // HZ-normalised delta to stay consistent with the GLSL convention.

        const hzDelta = dt * 60.0;
        p.x += cvx * curlSpeed * 0.01 * hzDelta;
        p.y += cvy * curlSpeed * 0.01 * hzDelta;

        // ── 4. Life tick ─────────────────────────────────────────────────
        p.life -= dt;

        if (p.life <= 0) {
          // Respawn: new random offset from cell centre within spawnRadius.
          const angle = Math.random() * Math.PI * 2;
          const dist  = Math.sqrt(Math.random()) * spawnRadius;
          p.ox   = Math.cos(angle) * dist;
          p.oy   = Math.sin(angle) * dist;
          p.x    = cx + p.ox;
          p.y    = cy + p.oy;
          p.life = p.maxLife * (0.5 + Math.random() * 0.5);
        }

        // ── 5. Soft tether — prevent particles drifting too far from cell ─
        //
        // Spring restoring force: beyond 1.5× spawnRadius, gently pull back.
        // This keeps the decoration cloud hugging its cell even during rapid
        // temporal evolution of the field.

        const dx = p.x - cx;
        const dy = p.y - cy;
        const d2 = dx * dx + dy * dy;
        const maxD = spawnRadius * 1.5;
        if (d2 > maxD * maxD) {
          const d   = Math.sqrt(d2);
          const k   = 0.06 * hzDelta;           // spring stiffness (per frame)
          p.x -= (dx / d) * (d - maxD) * k;
          p.y -= (dy / d) * (d - maxD) * k;
        }

        // ── 6. Write to PixiJS Particle ───────────────────────────────────
        const lifeRatio = Math.max(0, p.life / p.maxLife);      // [0, 1]
        const fadeMask  = smoothstep(0, 0.1, lifeRatio)         // fade in at birth
                        * smoothstep(0, 0.08, 1 - lifeRatio);   // fade out before death (inverted)

        p.px.x      = p.x;
        p.px.y      = p.y;
        p.px.alpha  = baseAlpha * fadeMask;
        p.px.scaleX = baseScale * (0.7 + 0.3 * lifeRatio);
        p.px.scaleY = baseScale * (0.7 + 0.3 * lifeRatio);
      }
    }
  }

  /**
   * Pause / resume all particle updates.
   */
  setPaused(paused: boolean): void {
    this.cfg.paused = paused;
  }

  /**
   * Return the number of currently registered cell emitters.
   */
  get cellCount(): number {
    return this.cells.size;
  }

  /**
   * Return the total live particle count across all emitters.
   */
  get particleCount(): number {
    let n = 0;
    for (const e of this.cells.values()) n += e.particles.length;
    return n;
  }

  /**
   * Destroy all particle containers, remove DOM event listeners, and release
   * GPU resources.  The CurlParticleField instance should not be used after
   * calling dispose().
   */
  dispose(): void {
    document.removeEventListener('cell:register',   this._onRegister);
    document.removeEventListener('cell:update',     this._onUpdate);
    document.removeEventListener('cell:unregister', this._onUnregister);

    for (const emitter of this.cells.values()) {
      this.app.stage.removeChild(emitter.container);
      emitter.container.destroy({ children: true });
    }

    this.cells.clear();
  }
}

// ── Convenience factory ───────────────────────────────────────────────────────

/**
 * Create a CurlParticleField, attach it to the app's Ticker, and return both
 * the field and a disposal handle.
 *
 * ```ts
 * const { field, destroy } = attachCurlParticleField(app, {
 *   particlesPerCell: 48,
 *   curlSpeed: 5,
 * });
 *
 * // Later:
 * destroy();
 * ```
 */
export function attachCurlParticleField(
  app:    Application,
  config: CurlParticleFieldConfig = {},
): { field: CurlParticleField; destroy: () => void } {
  const field = new CurlParticleField(app, config);

  const tickerCb = (ticker: { deltaMS: number }) => {
    field.tick(ticker.deltaMS / 1000);
  };

  app.ticker.add(tickerCb);

  const destroy = () => {
    app.ticker.remove(tickerCb);
    field.dispose();
  };

  return { field, destroy };
}

// ── Internal math helpers ──────────────────────────────────────────────────────

/** Hermite smoothstep — matches GLSL built-in. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
