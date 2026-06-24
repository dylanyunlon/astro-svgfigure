/**
 * src/lib/sph/destruction-system.ts — M789
 *
 * Destruction System — Voronoi fracture on strong collision
 * ─────────────────────────────────────────────────────────────────────────────
 * When a rigid body sustains a collision whose impulse exceeds a configurable
 * threshold, the system fractures its bounding rectangle into Voronoi shards.
 * Each shard becomes an independent debris particle with inherited velocity
 * plus an outward scatter impulse, then fades to transparent over its
 * lifetime and is culled.
 *
 * Architecture
 * ────────────
 *   DestructionSystem  (CPU-side, manages fracture + debris lifecycle)
 *     ├─ fracture(body, contact, impulse) — generate shards, mark body dead
 *     ├─ update(dt)                       — integrate debris physics, fade, cull
 *     ├─ draw(ctx)                        — Canvas 2D render of all live debris
 *     └─ getDebris()                      — snapshot for external renderers
 *
 * Voronoi fracture model
 * ──────────────────────
 *   1. Scatter N seed points inside the body's local-space bounding rect
 *      (count scales with impulse: 4‥maxShards).
 *   2. For each seed, compute a convex Voronoi cell clipped to the body rect.
 *      Uses Fortune's-style incremental convex clip against the four walls,
 *      with each cell bounded by perpendicular bisectors to neighboring seeds.
 *   3. Each cell becomes a DebrisShard with:
 *        - centroid position (world-space)
 *        - polygon vertices (local to centroid)
 *        - velocity = body.v + radialScatter × impulseScale
 *        - angular velocity = random spin
 *        - lifetime, alpha, colour inherited from body species
 *
 * Impulse → fracture mapping
 * ──────────────────────────
 *   impulse  →  t = clamp(impulse × impulseScale, 0, 1)
 *   shards   =  ceil(lerp(minShards, maxShards, √t))
 *   scatter  =  baseScatter × (0.5 + 0.5t)
 *   lifetime =  baseFade × (0.6 + 0.4t)
 *
 * Integration with CollisionWorld / CollisionEvents
 * ──────────────────────────────────────────────────
 *   const destruction = new DestructionSystem();
 *
 *   dispatcher.onCollisionEnter((evt) => {
 *     if (!evt.contact) return;
 *     const impulse = evt.contact.depth * 120;
 *     if (impulse > destruction.config.impulseThreshold) {
 *       const body = world.getBody(evt.bodyA);
 *       if (body) destruction.fracture(body, evt.contact, impulse);
 *     }
 *   });
 *
 *   // Each frame:
 *   destruction.update(dt);
 *   destruction.draw(ctx);
 *
 * Design references
 * ─────────────────
 *   src/lib/sph/contact-sparks.ts          — impulseScale, lygia RNG, draw loop
 *   src/lib/sph/collision-fx-system.ts     — impulse→visual mapping curve
 *   src/lib/sph/transition-system.ts       — dissolve particle fade pattern
 *   src/lib/sph/collision/CollisionEvents.ts — CollisionContactInfo types
 *   src/lib/sph/rigid-body.ts              — RigidBody structure
 */




// ─────────────────────────────────────────────────────────────────────────────
// Lygia random port (shared convention with contact-sparks / collision-fx)
// ─────────────────────────────────────────────────────────────────────────────




import type { CollisionContactInfo } from './collision/CollisionEvents';
import type { RigidBody }           from './rigid-body';

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

function fract(x: number): number { return x - Math.floor(x); }

function lygiaRandom(p: number): number {
  let x = fract(p * SCALE_X);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

function lygiaRandom2(sx: number, sy: number): number {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33);
  p3x += d; p3y += d; p3z += d;
  return fract((p3x + p3y) * p3z);
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface DestructionConfig {
  /** Minimum impulse required to trigger fracture. Default 60. */
  impulseThreshold: number;
  /** Impulse normalisation factor (impulse × scale → [0,1]). Default 0.008. */
  impulseScale: number;
  /** Min Voronoi shards per fracture. Default 4. */
  minShards: number;
  /** Max Voronoi shards per fracture. Default 16. */
  maxShards: number;
  /** Base outward scatter speed (px/s). Default 200. */
  baseScatter: number;
  /** Base debris lifetime (seconds). Default 1.2. */
  baseLifetime: number;
  /** Downward gravity for debris (px/s²). Default 280. */
  gravity: number;
  /** Angular velocity range (rad/s). Default 6. */
  maxSpin: number;
  /** Velocity drag per second (0=none, 1=full). Default 0.6. */
  drag: number;
  /** Max total live debris across all fractures. Default 256. */
  poolLimit: number;
}

const DEFAULT_CONFIG: DestructionConfig = {
  impulseThreshold: 60,
  impulseScale:     0.008,
  minShards:        4,
  maxShards:        16,
  baseScatter:      200,
  baseLifetime:     1.2,
  gravity:          280,
  maxSpin:          6,
  drag:             0.6,
  poolLimit:        256,
};

/** A single polygon vertex relative to the shard centroid. */
interface LocalVertex { x: number; y: number; }

/** An individual debris shard produced by Voronoi fracture. */
export interface DebrisShard {
  /** World-space centroid X */
  x: number;
  /** World-space centroid Y */
  y: number;
  /** Linear velocity X (px/s) */
  vx: number;
  /** Linear velocity Y (px/s) */
  vy: number;
  /** Current rotation angle (rad) */
  angle: number;
  /** Angular velocity (rad/s) */
  angVel: number;
  /** Polygon vertices in local (centroid-relative) space */
  vertices: LocalVertex[];
  /** Remaining lifetime (seconds) */
  life: number;
  /** Total lifetime for alpha ramp */
  maxLife: number;
  /** RGBA fill colour [r,g,b] 0‥255 */
  color: [number, number, number];
  /** Body species index (for external renderer matching) */
  species: number;
  /** ID of the source body that was fractured */
  sourceBodyId: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Species colour palette (matches cell-material-system conventions)
// ─────────────────────────────────────────────────────────────────────────────

const SPECIES_COLORS: [number, number, number][] = [
  [100, 180, 255],  // 0 — Attention (blue)
  [255, 140, 100],  // 1 — FFN (coral)
  [140, 220, 160],  // 2 — LayerNorm (green)
  [200, 160, 255],  // 3 — Embedding (lavender)
  [255, 200, 100],  // 4 — Softmax (gold)
  [180, 210, 240],  // 5 — Residual (pale blue)
  [240, 180, 200],  // 6 — Output (pink)
];

// ─────────────────────────────────────────────────────────────────────────────
// Voronoi fracture (bounded, 2D, convex-clip approach)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate Voronoi-like convex polygon shards inside a rectangle.
 *
 * Algorithm:
 *   1. Scatter `n` seed points inside [-hw, hw] × [-hh, hh].
 *   2. For each seed, start with the full rectangle as a convex polygon.
 *   3. For every other seed, clip the polygon by the perpendicular bisector
 *      half-plane (keeping the side closer to the current seed).
 *   4. The result is a convex Voronoi cell clipped to the bounding rect.
 *
 * Returns an array of { centroid, vertices[] } in local body space.
 */
function voronoiFracture(
  hw: number, hh: number, n: number, seedBase: number,
): Array<{ cx: number; cy: number; verts: LocalVertex[] }> {

  // ── scatter seeds ──
  const seeds: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    const rx = lygiaRandom2(seedBase + i * 7.13, seedBase + i * 3.91);
    const ry = lygiaRandom2(seedBase + i * 5.27, seedBase + i * 11.03);
    seeds.push({
      x: (rx * 2 - 1) * hw * 0.9,
      y: (ry * 2 - 1) * hh * 0.9,
    });
  }

  const shards: Array<{ cx: number; cy: number; verts: LocalVertex[] }> = [];

  for (let i = 0; i < n; i++) {
    // Start with full bounding rectangle
    let poly: LocalVertex[] = [
      { x: -hw, y: -hh },
      { x:  hw, y: -hh },
      { x:  hw, y:  hh },
      { x: -hw, y:  hh },
    ];

    const si = seeds[i];

    // Clip against every other seed's bisector
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const sj = seeds[j];

      // Bisector midpoint and normal (pointing toward seed i)
      const mx = (si.x + sj.x) * 0.5;
      const my = (si.y + sj.y) * 0.5;
      const nx = si.x - sj.x;
      const ny = si.y - sj.y;
      // Normal doesn't need to be unit-length for half-plane clipping

      poly = clipPolygonByHalfPlane(poly, mx, my, nx, ny);
      if (poly.length < 3) break;
    }

    if (poly.length < 3) continue;

    // Compute centroid
    let cx = 0, cy = 0;
    for (const v of poly) { cx += v.x; cy += v.y; }
    cx /= poly.length;
    cy /= poly.length;

    // Shift vertices to centroid-relative
    const verts = poly.map(v => ({ x: v.x - cx, y: v.y - cy }));

    shards.push({ cx, cy, verts });
  }

  return shards;
}

/**
 * Sutherland–Hodgman clip of a convex polygon against a half-plane.
 * Keeps the side where dot(p - point, normal) >= 0.
 */
function clipPolygonByHalfPlane(
  poly: LocalVertex[], px: number, py: number, nx: number, ny: number,
): LocalVertex[] {
  const out: LocalVertex[] = [];
  const len = poly.length;
  if (len === 0) return out;

  for (let i = 0; i < len; i++) {
    const curr = poly[i];
    const next = poly[(i + 1) % len];

    const dCurr = (curr.x - px) * nx + (curr.y - py) * ny;
    const dNext = (next.x - px) * nx + (next.y - py) * ny;

    if (dCurr >= 0) {
      out.push(curr);
      if (dNext < 0) {
        // edge exits — find intersection
        const t = dCurr / (dCurr - dNext);
        out.push({
          x: curr.x + (next.x - curr.x) * t,
          y: curr.y + (next.y - curr.y) * t,
        });
      }
    } else if (dNext >= 0) {
      // edge enters — find intersection
      const t = dCurr / (dCurr - dNext);
      out.push({
        x: curr.x + (next.x - curr.x) * t,
        y: curr.y + (next.y - curr.y) * t,
      });
    }
  }

  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// DestructionSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages Voronoi fracture and debris lifecycle for rigid-body destruction.
 *
 * @example
 * ```ts
 * const destruction = new DestructionSystem({ impulseThreshold: 80 });
 *
 * dispatcher.onCollisionEnter((evt) => {
 *   if (!evt.contact) return;
 *   const impulse = evt.contact.depth * 120;
 *   const body = world.getBody(evt.bodyA);
 *   if (body && impulse > destruction.config.impulseThreshold) {
 *     destruction.fracture(body, evt.contact, impulse);
 *   }
 * });
 *
 * // Each animation frame:
 * destruction.update(dt);
 * destruction.draw(ctx);
 * ```
 */
export class DestructionSystem {
  private _debris: DebrisShard[] = [];
  private _cfg: DestructionConfig;
  private _frame = 0;
  /** Set of body IDs that have already been fractured (avoid double-fracture) */
  private _fracturedIds = new Set<number>();
  /** Callback fired when a body is fractured; host can remove it from the world */
  private _onFracture: ((bodyId: number) => void) | null = null;

  constructor(config: Partial<DestructionConfig> = {}) {
    this._cfg = { ...DEFAULT_CONFIG, ...config };
  }

  /** Expose read-only config for threshold checks. */
  get config(): Readonly<DestructionConfig> { return this._cfg; }

  /** Number of live debris shards. */
  get debrisCount(): number { return this._debris.length; }

  /** Register a callback invoked when a body is fractured. */
  onFracture(cb: (bodyId: number) => void): () => void {
    this._onFracture = cb;
    return () => { if (this._onFracture === cb) this._onFracture = null; };
  }

  /** Check whether a body has already been fractured this session. */
  isFractured(bodyId: number): boolean { return this._fracturedIds.has(bodyId); }

  // ── Fracture ─────────────────────────────────────────────────────────────

  /**
   * Fracture a rigid body into Voronoi shards.
   *
   * @param body     The rigid body to destroy.
   * @param contact  Collision contact info (used for scatter bias direction).
   * @param impulse  Raw impulse magnitude.
   */
  fracture(body: RigidBody, contact: CollisionContactInfo, impulse: number): void {
    if (this._fracturedIds.has(body.id)) return; // already fractured
    this._fracturedIds.add(body.id);

    const cfg = this._cfg;
    const t = Math.min(impulse * cfg.impulseScale, 1);

    // Shard count: more shards for stronger impacts
    const shardCount = Math.ceil(
      cfg.minShards + (cfg.maxShards - cfg.minShards) * Math.sqrt(t),
    );

    // Scatter speed and lifetime scale with impulse
    const scatterSpeed = cfg.baseScatter * (0.5 + 0.5 * t);
    const lifetime     = cfg.baseLifetime * (0.6 + 0.4 * t);

    // Voronoi fracture in local body space
    const seedBase = this._frame * 17.7 + body.id * 31.3;
    const shards = voronoiFracture(body.w, body.h, shardCount, seedBase);

    // Body rotation transform
    const cosA = Math.cos(body.angle);
    const sinA = Math.sin(body.angle);

    // Species colour with slight per-shard variation
    const baseColor = SPECIES_COLORS[body.species % SPECIES_COLORS.length];

    for (let k = 0; k < shards.length; k++) {
      if (this._debris.length >= cfg.poolLimit) break;

      const s = shards[k];

      // Transform centroid to world space
      const wx = body.x + cosA * s.cx - sinA * s.cy;
      const wy = body.y + sinA * s.cx + cosA * s.cy;

      // Radial scatter direction from contact point
      let rdx = wx - contact.pointA.x;
      let rdy = wy - contact.pointA.y;
      const rLen = Math.sqrt(rdx * rdx + rdy * rdy) || 1;
      rdx /= rLen;
      rdy /= rLen;

      // Per-shard random jitter
      const jitter = lygiaRandom2(seedBase + k * 13.7, seedBase + k * 9.1);
      const jAngle = (jitter - 0.5) * Math.PI * 0.6;
      const sdx = rdx * Math.cos(jAngle) - rdy * Math.sin(jAngle);
      const sdy = rdx * Math.sin(jAngle) + rdy * Math.cos(jAngle);

      // Colour variation
      const cVar = (lygiaRandom(seedBase + k * 23.1) - 0.5) * 30;

      // Rotate local vertices by body angle
      const rotVerts = s.verts.map(v => ({
        x: cosA * v.x - sinA * v.y,
        y: sinA * v.x + cosA * v.y,
      }));

      this._debris.push({
        x:  wx,
        y:  wy,
        vx: body.vx + sdx * scatterSpeed,
        vy: body.vy + sdy * scatterSpeed,
        angle:  body.angle + (lygiaRandom(seedBase + k) - 0.5) * 0.5,
        angVel: (lygiaRandom(seedBase + k * 2.3) - 0.5) * 2 * cfg.maxSpin,
        vertices: rotVerts,
        life:    lifetime * (0.7 + 0.3 * lygiaRandom(seedBase + k * 5.7)),
        maxLife: lifetime,
        color: [
          Math.min(255, Math.max(0, baseColor[0] + cVar)),
          Math.min(255, Math.max(0, baseColor[1] + cVar)),
          Math.min(255, Math.max(0, baseColor[2] + cVar)),
        ],
        species: body.species,
        sourceBodyId: body.id,
      });
    }

    this._frame++;
    if (this._onFracture) this._onFracture(body.id);
  }

  // ── Update ───────────────────────────────────────────────────────────────

  /**
   * Advance debris physics: integrate positions, apply gravity + drag,
   * decrement lifetime, cull dead shards.
   */
  update(dt: number): void {
    const cfg = this._cfg;
    const dragFactor = Math.pow(1 - cfg.drag, dt);
    let writeIdx = 0;

    for (let i = 0; i < this._debris.length; i++) {
      const d = this._debris[i];

      // Decrement lifetime
      d.life -= dt;
      if (d.life <= 0) continue; // cull

      // Integrate velocity (gravity + drag)
      d.vy += cfg.gravity * dt;
      d.vx *= dragFactor;
      d.vy *= dragFactor;

      // Integrate position
      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.angle += d.angVel * dt;

      // Compact alive debris in-place
      this._debris[writeIdx++] = d;
    }

    this._debris.length = writeIdx;
  }

  // ── Draw (Canvas 2D) ────────────────────────────────────────────────────

  /**
   * Render all live debris shards to a Canvas 2D context.
   * Each shard is drawn as a filled polygon with alpha fade-out.
   */
  draw(ctx: CanvasRenderingContext2D): void {
    if (this._debris.length === 0) return;

    ctx.save();

    for (const d of this._debris) {
      const t = Math.max(0, d.life / d.maxLife); // 1 → fresh, 0 → dead
      const alpha = t * t; // quadratic fade-out (fast initial, slow end)

      if (alpha < 0.005) continue;

      const [r, g, b] = d.color;

      ctx.save();
      ctx.translate(d.x, d.y);
      ctx.rotate(d.angle);
      ctx.globalAlpha = alpha;

      // Fill polygon
      ctx.beginPath();
      const v0 = d.vertices[0];
      ctx.moveTo(v0.x, v0.y);
      for (let i = 1; i < d.vertices.length; i++) {
        ctx.lineTo(d.vertices[i].x, d.vertices[i].y);
      }
      ctx.closePath();

      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fill();

      // Subtle edge highlight for depth
      ctx.strokeStyle = `rgba(255,255,255,${(alpha * 0.3).toFixed(3)})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();

      ctx.restore();
    }

    ctx.restore();
  }

  // ── Accessors ────────────────────────────────────────────────────────────

  /** Read-only snapshot of all live debris (for GPU renderers). */
  getDebris(): readonly DebrisShard[] { return this._debris; }

  /** Reset all state: clear debris pool and fractured-body set. */
  reset(): void {
    this._debris.length = 0;
    this._fracturedIds.clear();
    this._frame = 0;
  }

  /** Allow a body ID to be fractured again (e.g. after respawn). */
  clearFractured(bodyId: number): void {
    this._fracturedIds.delete(bodyId);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convenience impulse estimator matching contact-sparks / collision-fx
 * convention: penetration depth × stiffness constant.
 */
export function estimateDestructionImpulse(contact: CollisionContactInfo): number {
  return contact.depth * 120;
}
