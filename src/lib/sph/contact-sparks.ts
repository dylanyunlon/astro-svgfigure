/**
 * src/lib/sph/contact-sparks.ts
 *
 * Contact Sparks — M587
 * ─────────────────────
 * Collision contact points spawn natural-world-style spark particles on a
 * Canvas 2D context.  Impulse magnitude drives both particle count and
 * luminosity: a gentle tap produces a handful of dim embers while a heavy
 * impact bursts into dozens of bright, fast-moving sparks.
 *
 * Randomness
 * ──────────
 * Particle scatter is derived from the lygia/generative/random.wgsl
 * sin-less hash family, ported to TypeScript:
 *
 *   random(p)   → scalar in [0, 1)   from scalar seed
 *   random2(st) → scalar in [0, 1)   from 2-D seed (vec2 equivalent)
 *   random22(p) → vec2  in [0, 1)²   from 2-D seed
 *
 * These produce chaotic, low-correlation outputs that avoid the banding
 * artefacts of simple LCG or sin-based pseudo-randoms.
 *
 * Physics model (naturalistic)
 * ────────────────────────────
 *   • Initial velocity: biased along the reflected normal + random cone
 *   • Gravity: downward pull (configurable)
 *   • Air drag: exponential velocity decay each frame
 *   • Life: each spark fades from bright white/amber through orange to dark red
 *           then is removed once alpha < threshold
 *   • Trail: each spark draws a short tail whose length tracks current speed
 *
 * Integration with CollisionWorld / CollisionEvents
 * ─────────────────────────────────────────────────
 *   1. Construct one ContactSparkSystem and keep it alive alongside the world.
 *   2. Subscribe to collision-enter events from CollisionEventDispatcher:
 *
 *       dispatcher.onCollisionEnter((evt) => {
 *         if (!evt.contact) return;
 *         const impulse = evt.contact.depth * 80;   // or compute from Δv
 *         sparks.emit(evt.contact.pointA, evt.contact.normal, impulse);
 *       });
 *
 *   3. Each animation frame call:
 *
 *       sparks.update(dt);        // advance physics
 *       sparks.draw(ctx);         // paint onto canvas
 */

import type { CollisionContactInfo } from './collision/CollisionEvents';

// ─────────────────────────────────────────────────────────────────────────────
// Lygia random.wgsl — TypeScript port (sin-less, RANDOM_SINLESS = true)
//
// RANDOM_SCALE = vec4(0.1031, 0.1030, 0.0973, 0.1099)
// ─────────────────────────────────────────────────────────────────────────────

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

/** Scalar → scalar hash, range [0, 1). */
function lygiaRandom(p: number): number {
  let x = fract(p * SCALE_X);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

/** vec2 → scalar hash, range [0, 1). */
function lygiaRandom2(sx: number, sy: number): number {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33);
  p3x += d; p3y += d; p3z += d;
  return fract((p3x + p3y) * p3z);
}

/** vec2 → vec2 hash, each component in [0, 1). */
function lygiaRandom22(sx: number, sy: number): [number, number] {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19);
  p3x += d; p3y += d; p3z += d;
  return [fract((p3x + p3x) * (p3y + p3z)), fract((p3x + p3y) * (p3y + p3z))];
}

function fract(x: number): number {
  return x - Math.floor(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

export interface SparkConfig {
  /**
   * Impulse scale: maps raw impulse magnitude to [0, 1].
   * Default 0.01 — tune to your velocity / depth units.
   */
  impulseScale: number;

  /** Maximum sparks spawned per collision (capped at this value). Default 64. */
  maxSparksPerHit: number;

  /** Base speed (px/s) at impulse=1. Default 180. */
  baseSpeed: number;

  /** Extra random speed multiplier range [0, speedJitter]. Default 120. */
  speedJitter: number;

  /** Half-angle of the emission cone (radians). Default π/3 (60°). */
  coneHalfAngle: number;

  /** Downward gravity acceleration (px/s²). Default 320. */
  gravity: number;

  /** Velocity drag coefficient (per second). 0 = no drag, 1 = full stop. Default 0.8. */
  drag: number;

  /** Spark lifetime at full impulse (seconds). Default 0.9. */
  maxLifetime: number;

  /** Minimum spark lifetime (seconds). Default 0.25. */
  minLifetime: number;

  /** Trail length multiplier. Larger = longer streaks. Default 0.06. */
  trailScale: number;

  /** Global alpha multiplier applied when drawing. Default 1. */
  opacity: number;

  /** Global particle pool ceiling. Default 512. */
  poolLimit: number;
}

const DEFAULT_CONFIG: SparkConfig = {
  impulseScale:      0.01,
  maxSparksPerHit:   64,
  baseSpeed:         180,
  speedJitter:       120,
  coneHalfAngle:     Math.PI / 3,
  gravity:           320,
  drag:              0.8,
  maxLifetime:       0.9,
  minLifetime:       0.25,
  trailScale:        0.06,
  opacity:           1.0,
  poolLimit:         512,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal spark particle
// ─────────────────────────────────────────────────────────────────────────────

interface Spark {
  x:        number;
  y:        number;
  vx:       number;
  vy:       number;
  life:     number;   // remaining lifetime (seconds)
  maxLife:  number;   // total lifetime for alpha ramp
  /** luminance multiplier (0‥1); derived from impulse strength */
  bright:   number;
  /** seed carried for colour noise */
  seed:     number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Spark colour ramp: white core → amber → orange → dark red → transparent.
 *
 * t = 1  → fresh spark (bright white)
 * t = 0  → dying ember (dark red, nearly transparent)
 *
 * `bright` (0‥1) scales the overall luminance so hard collisions produce
 * white-hot sparks while soft ones glow amber.
 */
function sparkColor(t: number, bright: number): string {
  // colour stops: [r, g, b] in 0‥255
  const stops: [number, number, number][] = [
    [200, 30,  10],   // t=0  dark red
    [255, 100,  20],  // t=0.25 orange
    [255, 180,  40],  // t=0.5  amber
    [255, 230, 120],  // t=0.75 yellow-white
    [255, 255, 230],  // t=1.0  white-hot
  ];

  const idx = Math.min(t * (stops.length - 1), stops.length - 1 - 1e-9);
  const lo  = Math.floor(idx);
  const hi  = Math.min(lo + 1, stops.length - 1);
  const f   = idx - lo;

  const r = (stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f) * (0.5 + bright * 0.5);
  const g = (stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f) * (0.5 + bright * 0.5);
  const b = (stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f) * (0.5 + bright * 0.5);

  const alpha = Math.pow(t, 0.4) * (0.3 + bright * 0.7);

  return `rgba(${r | 0},${g | 0},${b | 0},${alpha.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// ContactSparkSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages a pool of spark particles driven by rigid-body collision events.
 *
 * @example
 * ```ts
 * const sparks = new ContactSparkSystem({ impulseScale: 0.012 });
 *
 * world.events.onCollisionEnter((evt) => {
 *   if (!evt.contact) return;
 *   // Estimate impulse from penetration depth × a tuning constant:
 *   const impulse = evt.contact.depth * 120;
 *   sparks.emit(
 *     { x: evt.contact.pointA.x, y: evt.contact.pointA.y },
 *     { x: evt.contact.normal.x, y: evt.contact.normal.y },
 *     impulse,
 *   );
 * });
 *
 * // In your animation loop:
 * sparks.update(dt);
 * sparks.draw(ctx);
 * ```
 */
export class ContactSparkSystem {
  private sparks: Spark[] = [];
  private cfg:    SparkConfig;
  private _frame  = 0;   // monotone frame counter for seeding

  constructor(config: Partial<SparkConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Emit sparks at a collision contact point.
   *
   * @param point   World-space contact position (canvas pixels).
   * @param normal  Unit contact normal (points away from surface into free space).
   * @param impulse Raw impulse magnitude (velocity-change units); drives count & brightness.
   */
  emit(point: Vec2, normal: Vec2, impulse: number): void {
    const cfg = this.cfg;

    // Map impulse → [0, 1], capped at 1
    const t = Math.min(impulse * cfg.impulseScale, 1.0);
    if (t < 0.01) return;

    // Spark count: 1 at t≈0.1, maxSparksPerHit at t=1, with √ curve for
    // naturalistic feel (many small sparks at light touches, burst at impact)
    const count = Math.max(1, Math.round(cfg.maxSparksPerHit * Math.sqrt(t)));

    // Reflect normal provides the primary emission axis
    const nx = normal.x;
    const ny = normal.y;

    for (let i = 0; i < count; i++) {
      // Pool check
      if (this.sparks.length >= cfg.poolLimit) break;

      // Lygia-style 2-D seeds (position-time based for coherent variety)
      const seed = this._frame * 1000 + i;
      const [r0, r1] = lygiaRandom22(point.x * 0.01 + seed, point.y * 0.01 + i);
      const r2 = lygiaRandom2(seed + 0.5, i + 13.7);
      const r3 = lygiaRandom(seed * 0.7 + i * 3.1);

      // Emission direction: reflect normal + random cone scatter
      const angle = Math.atan2(ny, nx) +
                    (r0 * 2 - 1) * cfg.coneHalfAngle +
                    (r1 * 2 - 1) * 0.3;  // extra micro-jitter

      // Speed: base + jitter, scaled by impulse strength
      const speed = (cfg.baseSpeed + r2 * cfg.speedJitter) * (0.4 + t * 0.6);

      // Lifetime: shorter for weak sparks, longer for strong ones
      const life = cfg.minLifetime + r3 * (cfg.maxLifetime - cfg.minLifetime) * t;

      this.sparks.push({
        x:       point.x + (lygiaRandom(seed + 99) - 0.5) * 4,
        y:       point.y + (lygiaRandom(seed + 77) - 0.5) * 4,
        vx:      Math.cos(angle) * speed,
        vy:      Math.sin(angle) * speed,
        life,
        maxLife: life,
        bright:  t,
        seed,
      });
    }

    this._frame++;
  }

  /**
   * Convenience overload: emit from a CollisionContactInfo directly.
   *
   * `impulse` should be pre-computed from the collision solver, e.g.:
   *   `contact.depth * massEstimate * invDt`
   *
   * or simply `contact.depth * tuningConstant` for a visually-driven scale.
   */
  emitFromContact(contact: CollisionContactInfo, impulse: number): void {
    // Use the midpoint between the two contact points as the spark origin
    const mx = (contact.pointA.x + contact.pointB.x) * 0.5;
    const my = (contact.pointA.y + contact.pointB.y) * 0.5;
    this.emit(
      { x: mx, y: my },
      { x: contact.normal.x, y: contact.normal.y },
      impulse,
    );
  }

  /**
   * Advance all spark particles by `dt` seconds.
   * Call once per animation frame before `draw`.
   */
  update(dt: number): void {
    const cfg = this.cfg;
    const dragFactor = Math.exp(-cfg.drag * dt);

    for (let i = this.sparks.length - 1; i >= 0; i--) {
      const s = this.sparks[i];

      // Gravity
      s.vy += cfg.gravity * dt;

      // Air drag (exponential decay, physically grounded)
      s.vx *= dragFactor;
      s.vy *= dragFactor;

      // Position integration
      s.x += s.vx * dt;
      s.y += s.vy * dt;

      // Life
      s.life -= dt;
      if (s.life <= 0) {
        this.sparks.splice(i, 1);
      }
    }
  }

  /**
   * Draw all active sparks onto the supplied Canvas 2D context.
   * Assumes the context transform matches the world coordinate space.
   */
  draw(ctx: CanvasRenderingContext2D): void {
    const cfg = this.cfg;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';  // additive blend for glow
    ctx.lineCap = 'round';

    for (const s of this.sparks) {
      const t     = Math.max(0, s.life / s.maxLife);   // 1=new, 0=dead
      const speed = Math.hypot(s.vx, s.vy);

      // Core radius scales with brightness and residual life
      const radius = (1.0 + s.bright * 1.5) * Math.sqrt(t);

      // Trail length proportional to current speed
      const trailLen = speed * cfg.trailScale;
      const normX    = speed > 1 ? s.vx / speed : 0;
      const normY    = speed > 1 ? s.vy / speed : 0;

      const color = sparkColor(t, s.bright);

      // ── Trail streak ───────────────────────────────────────────────────
      if (trailLen > 1) {
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x - normX * trailLen, s.y - normY * trailLen);
        ctx.strokeStyle = color;
        ctx.lineWidth   = radius * 0.8 * cfg.opacity;
        ctx.stroke();
      }

      // ── Core dot ───────────────────────────────────────────────────────
      // Small radial gradient gives the hot-core appearance
      if (radius > 0.4) {
        const grd = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, radius * 2.5);
        grd.addColorStop(0, color);
        grd.addColorStop(1, 'rgba(0,0,0,0)');

        ctx.beginPath();
        ctx.arc(s.x, s.y, radius * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.globalAlpha = cfg.opacity;
        ctx.fill();
      }
    }

    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Number of currently live spark particles. */
  get count(): number {
    return this.sparks.length;
  }

  /** Remove all active sparks immediately. */
  clear(): void {
    this.sparks.length = 0;
  }

  /** Mutate config values at runtime (e.g. from a debug panel). */
  configure(overrides: Partial<SparkConfig>): void {
    Object.assign(this.cfg, overrides);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: estimate impulse magnitude from a CollisionContactInfo
//
// When the full rigid-body solver impulse j is available prefer that.
// This helper derives a proxy impulse from the contact depth and a caller-
// supplied mass / velocity estimate, covering use cases where only the
// CollisionContactInfo is accessible.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Estimate a proxy impulse magnitude from contact geometry.
 *
 * @param contact   Contact info from the narrow phase or CollisionEvent.
 * @param relSpeed  Relative speed of the two bodies at the contact point (px/s).
 *                  Pass 0 if not known; depth alone will still scale the result.
 * @param massScale Tuning constant; adjust until the visual density looks right.
 * @returns Impulse proxy suitable for passing to `ContactSparkSystem.emit`.
 */
export function estimateImpulse(
  contact: CollisionContactInfo,
  relSpeed = 0,
  massScale = 1,
): number {
  // depth encodes penetration; relSpeed encodes kinetic energy at contact
  return (contact.depth * 60 + relSpeed) * massScale;
}
