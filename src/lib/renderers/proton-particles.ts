/**
 * proton-particles.ts — Proton GPU particle system for cell state visual effects
 *
 * AT Proton module (37 refs) analogue — sub-modules:
 *   ProtonAntimatter  : GPU-driven particle pool (PixiJS ParticleContainer + Particle)
 *   ProtonPhysics     : per-particle force field (velocity, drag, radial gravity)
 *   ProtonPresets     : three named effect presets driven by cell lifecycle events
 *   ProtonCulling     : frustum-AABB cull — skips emitters outside visible viewport
 *
 * Preset effects:
 *   glow-burst  — fired on cell birth: radial burst of bright specks (additive blend)
 *   fade-trail  — fired on cell move: directional smoke trail following velocity vector
 *   pulse       — fired on epoch switch: concentric ring wave across all live cells
 *
 * Rendering:
 *   ParticleContainer (PixiJS scene/particle-container) renders each particle as a
 *   GPU point sprite with per-particle position / alpha / scale / tint updates every
 *   frame.  A single shared 8×8 white-circle texture is used; species tint is applied
 *   per-particle so only one draw call per ParticleContainer is needed.
 *
 * Upstream references:
 *   upstream/pixijs-engine/src/scene/particle-container/shared/Particle.ts
 *   upstream/pixijs-engine/src/scene/particle-container/shared/ParticleContainer.ts
 *   upstream/pixijs-engine/src/scene/particle-container/shared/particleData.ts
 *
 * Author: dylanyunlon <dogechat@163.com>
 */

import { Application } from '../../upstream/pixijs-engine/src/app/Application';
import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

// ── Lightweight PixiJS surface-level imports (resolved via tsconfig paths) ──
// We use the pixi.js barrel rather than deep upstream paths for the types that
// are only available as compiled API (Texture.WHITE, ParticleContainer, Particle).
import {
  ParticleContainer,
  Particle,
  Texture,
  Graphics,
} from 'pixi.js';

// ── Species → tint colour (mirrors pixi-cell-renderer palette) ──────────────

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
  return SPECIES_TINT[species] ?? 0xFFFFFF;
}

// ── Shared point-sprite texture (8×8 radial soft-circle, generated once) ────

let _dotTexture: Texture | null = null;

function getDotTexture(): Texture {
  if (_dotTexture) return _dotTexture;
  // Generate an 8×8 white circle via Graphics, extract as texture.
  // Graphics.generateTexture is available on the PixiJS renderer path;
  // when the renderer is not yet initialised we fall back to Texture.WHITE.
  try {
    const g = new Graphics();
    g.circle(4, 4, 4);
    g.fill({ color: 0xffffff, alpha: 1 });
    // @ts-ignore — generateTexture is present on PixiJS Graphics in v8
    _dotTexture = g.generateTexture();
    g.destroy();
  } catch {
    _dotTexture = Texture.WHITE;
  }
  return _dotTexture!;
}

// ── ProtonPhysics — per-particle state & integration ────────────────────────

interface PhysicsParticle {
  /** backing PixiJS Particle */
  sprite: Particle;
  /** velocity */
  vx: number;
  vy: number;
  /** radial drag coefficient (0 = no drag, 1 = instant stop) */
  drag: number;
  /** remaining life in frames */
  life: number;
  /** max life — used for alpha curve */
  maxLife: number;
  /** initial scale */
  scale0: number;
  /** target scale at end of life (for shrink/grow) */
  scaleEnd: number;
}

// ── ProtonPresets — burst / trail / pulse spawn recipes ─────────────────────

export type ProtonPresetName = 'glow-burst' | 'fade-trail' | 'pulse';

interface EmitRecipe {
  /** total particles to spawn */
  count: number;
  /** life in frames */
  life: number;
  /** initial speed scalar */
  speed: number;
  /** drag per frame (fraction of velocity removed) */
  drag: number;
  /** initial scale of each sprite */
  scale0: number;
  /** final scale (shrink toward 0 = sparkle, grow to 1.5 = smoke) */
  scaleEnd: number;
  /** colour tint */
  tint: number;
  /** spawn direction angle spread in radians (Math.PI*2 = isotropic) */
  angleSpread: number;
  /** base angle for directional effects (radians, 0 = right) */
  baseAngle: number;
  /** radial spawn radius offset from emitter centre */
  spawnRadius: number;
}

const PRESETS: Record<ProtonPresetName, (tint: number, dirAngle?: number) => EmitRecipe> = {
  'glow-burst': (tint) => ({
    count:       28,
    life:        35,
    speed:       4.5,
    drag:        0.10,
    scale0:      0.9,
    scaleEnd:    0.0,
    tint,
    angleSpread: Math.PI * 2,
    baseAngle:   0,
    spawnRadius: 0,
  }),

  'fade-trail': (tint, dirAngle = 0) => ({
    count:       14,
    life:        22,
    speed:       2.2,
    drag:        0.08,
    scale0:      0.6,
    scaleEnd:    0.1,
    tint,
    angleSpread: Math.PI * 0.45,
    baseAngle:   dirAngle + Math.PI,  // trail emits opposite to motion direction
    spawnRadius: 3,
  }),

  'pulse': (tint) => ({
    count:       18,
    life:        48,
    speed:       1.8,
    drag:        0.04,
    scale0:      1.1,
    scaleEnd:    0.0,
    tint,
    angleSpread: Math.PI * 2,
    baseAngle:   0,
    spawnRadius: 12,
  }),
};

// ── ProtonCulling — AABB frustum reject ─────────────────────────────────────
// Emitters outside the visible viewport are skipped to avoid spawning invisible
// particles.  viewport is updated lazily from the app renderer dimensions.

interface Viewport { x0: number; y0: number; x1: number; y1: number }

function insideViewport(cx: number, cy: number, vp: Viewport, margin = 80): boolean {
  return (
    cx >= vp.x0 - margin &&
    cx <= vp.x1 + margin &&
    cy >= vp.y0 - margin &&
    cy <= vp.y1 + margin
  );
}

// ── ProtonEmitter — one emitter that tracks its own particle pool ────────────

export interface ProtonEmitterOptions {
  /** World-space position */
  x: number;
  y: number;
  preset: ProtonPresetName;
  /** Species tint (hex RGB) */
  tint: number;
  /** Optional direction in radians (used by fade-trail) */
  dirAngle?: number;
}

class ProtonEmitter {
  private readonly _pool: PhysicsParticle[] = [];
  private _done = false;

  constructor(
    private readonly _container: ParticleContainer,
    opts: ProtonEmitterOptions,
  ) {
    const recipe = PRESETS[opts.preset](opts.tint, opts.dirAngle);
    const tex = getDotTexture();

    for (let i = 0; i < recipe.count; i++) {
      const angle = opts.dirAngle !== undefined && opts.preset !== 'glow-burst' && opts.preset !== 'pulse'
        ? recipe.baseAngle + (Math.random() - 0.5) * recipe.angleSpread
        : recipe.baseAngle + Math.random() * recipe.angleSpread;

      const speed = recipe.speed * (0.7 + Math.random() * 0.6);
      const r = recipe.spawnRadius * Math.random();

      const sprite = new Particle({
        texture: tex,
        x: opts.x + Math.cos(angle) * r,
        y: opts.y + Math.sin(angle) * r,
        scaleX: recipe.scale0,
        scaleY: recipe.scale0,
        anchorX: 0.5,
        anchorY: 0.5,
        tint: recipe.tint,
        alpha: 1,
      });

      this._container.addParticle(sprite);

      this._pool.push({
        sprite,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        drag: recipe.drag,
        life: recipe.life,
        maxLife: recipe.life,
        scale0: recipe.scale0,
        scaleEnd: recipe.scaleEnd,
      });
    }
  }

  /** Integrate one frame.  Returns true when all particles have expired. */
  tick(): boolean {
    if (this._done) return true;

    let alive = 0;
    for (const p of this._pool) {
      if (p.life <= 0) continue;
      alive++;

      // ProtonPhysics integration
      p.vx *= (1 - p.drag);
      p.vy *= (1 - p.drag);
      p.sprite.x += p.vx;
      p.sprite.y += p.vy;
      p.life--;

      // alpha: linear fade out over life
      const t = p.life / p.maxLife;                  // 1→0 over lifetime
      p.sprite.alpha = t * t;                         // quadratic ease-out

      // scale: lerp scale0 → scaleEnd
      const s = p.scale0 + (p.scaleEnd - p.scale0) * (1 - t);
      p.sprite.scaleX = s;
      p.sprite.scaleY = s;

      if (p.life <= 0) {
        p.sprite.alpha = 0;
      }
    }

    if (alive === 0) {
      this._done = true;
      // Remove expired sprites from the container
      for (const p of this._pool) {
        this._container.removeParticle(p.sprite);
      }
      return true;
    }
    return false;
  }

  get isDone(): boolean { return this._done; }
}

// ── ProtonAntimatter — GPU particle pool manager ─────────────────────────────
// Manages the shared ParticleContainer and the list of live emitters.
// One ProtonAntimatter instance per scene.

class ProtonAntimatter {
  private readonly _container: ParticleContainer;
  private readonly _emitters: ProtonEmitter[] = [];

  constructor(stage: Container) {
    this._container = new ParticleContainer({
      dynamicProperties: {
        position: true,
        rotation: false,
        uvs:      false,
        color:    true,
        vertex:   false,
      },
    });
    // Additive blend — particles glow on dark backgrounds
    this._container.blendMode = 'add';
    this._container.zIndex   = 100;
    stage.addChild(this._container);
  }

  /** Spawn a new emitter at (cx, cy) with the given preset/species. */
  emit(opts: ProtonEmitterOptions, vp: Viewport): void {
    // ProtonCulling: skip emitters that are entirely off-screen
    if (!insideViewport(opts.x, opts.y, vp)) return;

    this._emitters.push(new ProtonEmitter(this._container, opts));
  }

  /** Called every frame from the ProtonSystem ticker. */
  tick(): void {
    // Integrate all emitters; collect indices of finished ones.
    // Iterate in reverse so splice doesn't shift indices.
    for (let i = this._emitters.length - 1; i >= 0; i--) {
      if (this._emitters[i].tick()) {
        this._emitters.splice(i, 1);
      }
    }
  }

  get activeEmitterCount(): number { return this._emitters.length; }

  destroy(): void {
    this._container.destroy({ children: true });
  }
}

// ── ProtonSystem — top-level public API ─────────────────────────────────────

/**
 * ProtonSystem manages the GPU particle sub-systems for cell lifecycle events.
 *
 * Usage:
 *   const proton = new ProtonSystem(app);
 *
 *   // cell born
 *   proton.cellBorn(cell_id, cx, cy, species);
 *
 *   // cell moved (provide previous position to derive direction)
 *   proton.cellMoved(cell_id, prevX, prevY, cx, cy, species);
 *
 *   // epoch switched — pulse all live cell positions
 *   proton.epochSwitch(liveCellPositions);
 *
 *   // teardown
 *   proton.destroy();
 */
export class ProtonSystem {
  private readonly _antimatter: ProtonAntimatter;
  private readonly _vp: Viewport;
  private _stopped = false;

  constructor(private readonly _app: Application) {
    this._antimatter = new ProtonAntimatter(_app.stage as unknown as Container);
    this._vp = this._currentViewport();

    // Hook into PixiJS ticker
    _app.ticker.add(this._tick, this);
  }

  // ── Cell lifecycle events ────────────────────────────────────────────────

  /**
   * Emit a glow-burst at (cx, cy) when a new cell is born.
   */
  cellBorn(cellId: string, cx: number, cy: number, species: string): void {
    if (this._stopped) return;
    this._refreshViewport();
    this._antimatter.emit({
      x: cx, y: cy,
      preset: 'glow-burst',
      tint:   speciesTint(species),
    }, this._vp);
  }

  /**
   * Emit a fade-trail when a cell moves.  The trail direction is computed from
   * the delta between the previous and current position.
   */
  cellMoved(
    _cellId: string,
    prevX: number,
    prevY: number,
    cx: number,
    cy: number,
    species: string,
  ): void {
    if (this._stopped) return;
    this._refreshViewport();

    const dx = cx - prevX;
    const dy = cy - prevY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1) return;   // sub-pixel move — not worth spawning

    const dirAngle = Math.atan2(dy, dx);

    // Spawn trail at midpoint between old and new position
    this._antimatter.emit({
      x: (prevX + cx) / 2,
      y: (prevY + cy) / 2,
      preset: 'fade-trail',
      tint:   speciesTint(species),
      dirAngle,
    }, this._vp);
  }

  /**
   * Emit a pulse ring at every live cell position on epoch switch.
   *
   * @param cells  Array of { cx, cy, species } for all live cells
   */
  epochSwitch(cells: ReadonlyArray<{ cx: number; cy: number; species: string }>): void {
    if (this._stopped) return;
    this._refreshViewport();

    for (const c of cells) {
      this._antimatter.emit({
        x: c.cx, y: c.cy,
        preset: 'pulse',
        tint:   speciesTint(c.species),
      }, this._vp);
    }
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  private _tick(): void {
    if (this._stopped) return;
    this._antimatter.tick();
  }

  private _refreshViewport(): void {
    const vp = this._currentViewport();
    this._vp.x0 = vp.x0;
    this._vp.y0 = vp.y0;
    this._vp.x1 = vp.x1;
    this._vp.y1 = vp.y1;
  }

  private _currentViewport(): Viewport {
    // @ts-ignore — renderer dimensions exist on the PixiJS Application
    const w: number = (this._app.renderer?.width  as number | undefined) ?? 800;
    // @ts-ignore
    const h: number = (this._app.renderer?.height as number | undefined) ?? 600;
    return { x0: 0, y0: 0, x1: w, y1: h };
  }

  /** Number of currently active emitters (useful for perf monitoring). */
  get activeEmitters(): number { return this._antimatter.activeEmitterCount; }

  destroy(): void {
    if (this._stopped) return;
    this._stopped = true;
    this._app.ticker.remove(this._tick, this);
    this._antimatter.destroy();
  }
}

// ── Convenience factory ──────────────────────────────────────────────────────

/**
 * createProtonSystem — attach a ProtonSystem to a running PixiJS Application.
 *
 * @example
 * ```ts
 * const app = new Application();
 * await app.init({ canvas, width: 800, height: 600 });
 *
 * const proton = createProtonSystem(app);
 *
 * // cell born at (200, 150):
 * proton.cellBorn('c1', 200, 150, 'cil-bolt');
 *
 * // cell moved:
 * proton.cellMoved('c1', 200, 150, 260, 150, 'cil-bolt');
 *
 * // epoch switch:
 * proton.epochSwitch([
 *   { cx: 260, cy: 150, species: 'cil-bolt' },
 *   { cx: 100, cy: 300, species: 'cil-eye'  },
 * ]);
 *
 * // teardown:
 * proton.destroy();
 * ```
 */
export function createProtonSystem(app: Application): ProtonSystem {
  return new ProtonSystem(app);
}
