/**
 * src/lib/sph/collision-fx-system.ts — M741
 *
 * Collision FX Flower Burst
 * ─────────────────────────────────────────────────────────────────────────────
 * When two rigid bodies collide the system spawns an impulse-proportional
 * burst of AT-style flower petal particles at the contact point.  Heavier
 * impacts produce more petals with higher luminance / larger size; gentle
 * touches yield a subtle scattering of dim petals.
 *
 * Rendering
 * ─────────
 * Each petal is a CPU-side particle whose draw pass uses the same visual
 * language as ATFlowerParticleRenderer — matcap sphere-map shading, spiral
 * trajectory from the contact point, sin(π·travel) alpha fade, and
 * vScale = uSize·(1 − travel²) size attenuation.  The shader-level maths
 * is replicated in Canvas 2D so no GPU pipeline is required (the collision
 * FX layer sits on top of whatever renderer the host app uses).
 *
 * For hosts that DO run a full WebGPU render pass, a `toFlowerEdgeSplines`
 * helper converts active bursts into ephemeral FlowerEdgeSpline[] that can
 * be fed to ATFlowerParticleRenderer for GPU-accelerated rendering.
 *
 * Impulse → visual mapping
 * ────────────────────────
 *   impulse  →  t = clamp(impulse × impulseScale, 0, 1)
 *   count    =  ceil(maxPetalsPerHit × √t)
 *   bright   =  t
 *   size     =  basePetalSize × (0.4 + 0.6t)
 *
 * This follows the same √-curve used by ContactSparkSystem so the visual
 * density scales naturally: light taps produce one or two dim petals while
 * heavy impacts burst into a full flower.
 *
 * Integration with CollisionWorld / CollisionEvents
 * ──────────────────────────────────────────────────
 *   1. Construct a CollisionFXSystem and keep it alongside the world.
 *   2. Subscribe to collision-enter events:
 *
 *       world.events.onCollisionEnter((evt) => {
 *         if (!evt.contact) return;
 *         const impulse = evt.contact.depth * 120;
 *         fx.emit(evt.contact, impulse);
 *       });
 *
 *   3. Each animation frame:
 *       fx.update(dt);
 *       fx.draw(ctx);       // Canvas 2D
 *       // — or —
 *       fx.drawWebGPU(enc, renderPass);  // GPU (optional, future)
 *
 * Colour palette
 * ──────────────
 * Petal colours sample from an AT-inspired warm flower palette:
 *   [ rose-pink, coral, peach, marigold, lavender, soft-white ]
 * A per-petal random offset keeps variation natural.  The `bright` scalar
 * derived from impulse strength controls luminance:
 *   low impulse  → muted, pastel tones
 *   high impulse → saturated, near-white hot-core petals
 */









import type { CollisionEventDispatcher } from './collision/CollisionEvents';
import type { FlowerEdgeSpline, FlowerPoint3 } from './at-flower-particle';

// [orphan-precise]   CollisionContactInfo,
// [orphan-precise]   CollisionEvent,

// ─────────────────────────────────────────────────────────────────────────────
// Lygia random port (shared with contact-sparks.ts)
// ─────────────────────────────────────────────────────────────────────────────

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

function fract(x: number): number {
  return x - Math.floor(x);
}

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

function lygiaRandom22(sx: number, sy: number): [number, number] {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19);
  p3x += d; p3y += d; p3z += d;
  return [fract((p3x + p3x) * (p3y + p3z)), fract((p3x + p3y) * (p3y + p3z))];
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface Vec2 {
  x: number;
  y: number;
}

/** Configuration for the collision FX flower-burst system. */
export interface CollisionFXConfig {
  /**
   * Impulse scale: maps raw impulse magnitude to normalised [0, 1].
   * Default 0.008 — tune to match your velocity / depth units.
   */
  impulseScale: number;

  /** Maximum petals spawned per collision event. Default 48. */
  maxPetalsPerHit: number;

  /**
   * Base petal radius in canvas/domain units (AT: uSize equivalent).
   * Final size = basePetalSize × (0.4 + 0.6·t) × (1 − travel²).
   * Default 6.
   */
  basePetalSize: number;

  /**
   * Spiral amplitude — how far petals drift laterally from the radial axis.
   * Expressed as a fraction of total burst radius.
   * Default 0.35 (AT FlowerParticleShader: SPIRAL_AMPLITUDE_RATIO × 20).
   */
  spiralAmplitude: number;

  /** Spiral angular speed (rad/s). AT: ~2.4. Default 3.6. */
  spiralSpeed: number;

  /** Radial outward speed of petals (units/s). Default 140. */
  burstSpeed: number;

  /** Extra random speed jitter. Default 80. */
  speedJitter: number;

  /** Downward gravity (units/s²). Default 60. */
  gravity: number;

  /** Air drag coefficient (per second). Default 1.2. */
  drag: number;

  /** Maximum petal lifetime (seconds). Default 0.7. */
  maxLifetime: number;

  /** Minimum petal lifetime (seconds). Default 0.2. */
  minLifetime: number;

  /** Global alpha multiplier. Default 1. */
  opacity: number;

  /** Global particle pool ceiling. Default 512. */
  poolLimit: number;

  /**
   * AT-style curl-noise lateral displacement amplitude.
   * Applied as a secondary wiggle on top of the spiral.
   * Default 0.06.
   */
  curlStrength: number;
}

const DEFAULT_CONFIG: CollisionFXConfig = {
  impulseScale:     0.008,
  maxPetalsPerHit:  48,
  basePetalSize:    6,
  spiralAmplitude:  0.35,
  spiralSpeed:      3.6,
  burstSpeed:       140,
  speedJitter:      80,
  gravity:          60,
  drag:             1.2,
  maxLifetime:      0.7,
  minLifetime:      0.2,
  opacity:          1.0,
  poolLimit:        512,
  curlStrength:     0.06,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal petal particle
// ─────────────────────────────────────────────────────────────────────────────

interface Petal {
  /** Current world position. */
  x: number;
  y: number;
  /** Velocity. */
  vx: number;
  vy: number;
  /** Remaining lifetime (seconds). */
  life: number;
  /** Total lifetime for travel fraction computation. */
  maxLife: number;
  /** Impulse-normalised brightness [0, 1]. */
  bright: number;
  /** AT spiral phase seed (θ₀). */
  theta0: number;
  /** Spiral lateral amplitude (units). */
  amplitude: number;
  /** Radial direction from burst centre (unit normal). */
  radialX: number;
  radialY: number;
  /** Perpendicular to radial (for spiral offset). */
  perpX: number;
  perpY: number;
  /** Per-petal random seed. */
  seed: number;
  /** Colour palette index [0, PALETTE.length). */
  colorIdx: number;
  /** Accumulated elapsed time for spiral phase computation. */
  elapsed: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// AT-inspired flower colour palette
// ─────────────────────────────────────────────────────────────────────────────
// Each entry: [r, g, b] in 0..255
// Warm floral tones inspired by AT's matcap3.png sphere-map shading.

const PALETTE: [number, number, number][] = [
  [255, 140, 160],  // rose-pink
  [255, 110, 100],  // coral
  [255, 185, 130],  // peach
  [255, 200,  70],  // marigold
  [200, 160, 255],  // lavender
  [255, 240, 220],  // soft-white
  [255, 165, 200],  // blush
  [255, 120, 180],  // hot-pink
];

/**
 * Compute the petal colour for a given life fraction and brightness.
 *
 * Mimics the AT FlowerParticleShader visual:
 *   - Bright (high impulse) petals have saturated, near-white hot-core tints
 *   - Dim petals are pastel and muted
 *   - Alpha follows sin(π·travel) — bright at midpoint, fading at edges
 *     multiplied by the AT vScale attenuation (1 − travel²)
 *
 * @param travel  Arc-length fraction [0, 1] where 0 = just spawned, 1 = dead.
 * @param bright  Impulse-normalised brightness [0, 1].
 * @param colorIdx  Palette index.
 * @returns CSS rgba() string.
 */
function petalColor(travel: number, bright: number, colorIdx: number): string {
  const [r0, g0, b0] = PALETTE[colorIdx % PALETTE.length];

  // Luminance boost from impulse strength:
  //   bright=0 → muted (×0.55), bright=1 → near-white (×1.0)
  const lum = 0.55 + bright * 0.45;
  const r = Math.min(255, r0 * lum + bright * 60);
  const g = Math.min(255, g0 * lum + bright * 40);
  const b = Math.min(255, b0 * lum + bright * 30);

  // AT alpha: sin(π·travel) — bright at midpoint, zero at start/end
  const sinFade = Math.sin(Math.PI * travel);
  // AT vScale decay: (1 − travel²)
  const travelDecay = Math.max(0, 1 - travel * travel);
  const alpha = sinFade * travelDecay * (0.4 + bright * 0.6);

  return `rgba(${r | 0},${g | 0},${b | 0},${alpha.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// CollisionFXSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages AT-style flower petal bursts at rigid-body collision contact points.
 *
 * Impulse magnitude drives both particle count and visual intensity —
 * heavy impacts explode into a dense, bright flower while light taps
 * scatter a handful of dim petals.
 *
 * Each petal follows a spiral trajectory radiating outward from the
 * contact point, replicating the AT FlowerParticleShader motion formula:
 *   pos = origin + radial·t·speed + perp·amplitude·sin(θ₀ + elapsed·spiralSpeed)
 *
 * @example
 * ```ts
 * const fx = new CollisionFXSystem({ impulseScale: 0.01 });
 *
 * world.events.onCollisionEnter((evt) => {
 *   if (!evt.contact) return;
 *   const impulse = evt.contact.depth * 120;
 *   fx.emit(evt.contact, impulse);
 * });
 *
 * // Animation loop:
 * fx.update(dt);
 * fx.draw(ctx);
 * ```
 */
export class CollisionFXSystem {
  private petals: Petal[] = [];
  private cfg: CollisionFXConfig;
  private _frame = 0;

  constructor(config: Partial<CollisionFXConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Emit a flower petal burst at a collision contact point.
   *
   * @param contact  CollisionContactInfo from the narrow phase / event system.
   * @param impulse  Raw impulse magnitude; drives count & brightness.
   */
  emit(contact: CollisionContactInfo, impulse: number): void {
    const cfg = this.cfg;

    // Map impulse → normalised [0, 1]
    const t = Math.min(impulse * cfg.impulseScale, 1.0);
    if (t < 0.005) return;

    // Petal count: √-curve for naturalistic density
    const count = Math.max(1, Math.ceil(cfg.maxPetalsPerHit * Math.sqrt(t)));

    // Burst origin: midpoint between the two contact surface points
    const ox = (contact.pointA.x + contact.pointB.x) * 0.5;
    const oy = (contact.pointA.y + contact.pointB.y) * 0.5;

    // Primary emission axis: contact normal
    const nx = contact.normal.x;
    const ny = contact.normal.y;
    const baseAngle = Math.atan2(ny, nx);

    for (let i = 0; i < count; i++) {
      if (this.petals.length >= cfg.poolLimit) break;

      // Lygia-seeded randomness
      const seed = this._frame * 1000 + i;
      const [r0, r1] = lygiaRandom22(ox * 0.01 + seed, oy * 0.01 + i);
      const r2 = lygiaRandom2(seed + 0.37, i + 7.9);
      const r3 = lygiaRandom(seed * 0.61 + i * 2.3);
      const r4 = lygiaRandom(seed * 1.17 + i * 5.1);

      // Emission direction: full-circle around contact for flower look,
      // biased toward the normal hemisphere
      const spread = Math.PI * 2;
      const bias = (r0 - 0.5) * spread;
      const normalBias = (r1 > 0.3) ? 0 : Math.PI;  // 70% toward normal side
      const angle = baseAngle + bias + normalBias * 0;
      // Actually use a flower-like radial burst (full circle) with slight
      // normal bias for realism:
      const petalAngle = baseAngle + (i / count) * Math.PI * 2 +
                         (r0 - 0.5) * 0.6;

      const cosA = Math.cos(petalAngle);
      const sinA = Math.sin(petalAngle);

      // Speed: base + jitter, scaled by impulse
      const speed = (cfg.burstSpeed + r2 * cfg.speedJitter) * (0.4 + t * 0.6);

      // Lifetime: longer for stronger impacts
      const life = cfg.minLifetime + r3 * (cfg.maxLifetime - cfg.minLifetime) * (0.5 + t * 0.5);

      // AT spiral parameters
      const theta0 = r4 * Math.PI * 2;
      const amplitude = cfg.spiralAmplitude * cfg.basePetalSize * (0.5 + t * 0.5);

      // Perpendicular to radial direction (for spiral offset)
      const perpX = -sinA;
      const perpY = cosA;

      this.petals.push({
        x: ox + (lygiaRandom(seed + 99) - 0.5) * 3,
        y: oy + (lygiaRandom(seed + 77) - 0.5) * 3,
        vx: cosA * speed,
        vy: sinA * speed,
        life,
        maxLife: life,
        bright: t,
        theta0,
        amplitude,
        radialX: cosA,
        radialY: sinA,
        perpX,
        perpY,
        seed,
        colorIdx: Math.floor(r1 * PALETTE.length),
        elapsed: 0,
      });
    }

    this._frame++;
  }

  /**
   * Convenience: emit from a CollisionEvent directly.
   *
   * @param event   CollisionEvent (only 'enter' phase events with contact are used).
   * @param impulse Raw impulse magnitude.
   */
  emitFromEvent(event: CollisionEvent, impulse: number): void {
    if (!event.contact) return;
    this.emit(event.contact, impulse);
  }

  /**
   * Convenience: emit using a Vec2 point and normal directly (similar to
   * ContactSparkSystem.emit interface).
   *
   * @param point   World-space contact position.
   * @param normal  Unit contact normal.
   * @param impulse Raw impulse magnitude.
   */
  emitAt(point: Vec2, normal: Vec2, impulse: number): void {
    this.emit(
      {
        normal: { x: normal.x, y: normal.y },
        depth: impulse * this.cfg.impulseScale * 10,
        pointA: { x: point.x, y: point.y },
        pointB: { x: point.x, y: point.y },
      },
      impulse,
    );
  }

  /**
   * Advance all petal particles by `dt` seconds.
   *
   * Physics model mirrors AT FlowerParticleShader motion:
   *   - Radial outward drift (decelerating via drag)
   *   - Spiral lateral oscillation: perp × amplitude × sin(θ₀ + elapsed × spiralSpeed)
   *   - Curl-noise secondary wiggle
   *   - Gravity (gentle, for petal-like floating)
   *   - AT alpha: sin(π·travel) × (1 − travel²)
   */
  update(dt: number): void {
    const cfg = this.cfg;
    const dragFactor = Math.exp(-cfg.drag * dt);

    for (let i = this.petals.length - 1; i >= 0; i--) {
      const p = this.petals[i];

      p.elapsed += dt;

      // Gravity (gentle for petals — they float, not fall)
      p.vy += cfg.gravity * dt;

      // Air drag
      p.vx *= dragFactor;
      p.vy *= dragFactor;

      // AT spiral offset: perp × amplitude × sin(θ₀ + elapsed × spiralSpeed)
      const spiralPhase = p.theta0 + p.elapsed * cfg.spiralSpeed;
      const spiralOffset = p.amplitude * Math.sin(spiralPhase);
      // Curl-noise secondary wiggle (simplified CPU version)
      const curlPhase = p.seed * 0.01 + p.elapsed * 2.1;
      const curlOffset = cfg.curlStrength * cfg.basePetalSize *
                         Math.sin(curlPhase * 3.7) * Math.cos(curlPhase * 2.3);

      // Integration: base velocity + spiral lateral drift
      const lateralX = p.perpX * (spiralOffset + curlOffset) * dt * 2;
      const lateralY = p.perpY * (spiralOffset + curlOffset) * dt * 2;
      p.x += p.vx * dt + lateralX;
      p.y += p.vy * dt + lateralY;

      // Life
      p.life -= dt;
      if (p.life <= 0) {
        // Swap-remove for O(1) pool management
        const last = this.petals.length - 1;
        if (i !== last) {
          this.petals[i] = this.petals[last];
        }
        this.petals.pop();
      }
    }
  }

  /**
   * Draw all active petals onto a Canvas 2D context.
   *
   * Renders each petal as a soft elliptical glow with matcap-inspired
   * radial gradient (AT FlowerParticleShader sphere-map shading).
   * Uses additive blending for the characteristic bright-core flower look.
   */
  draw(ctx: CanvasRenderingContext2D): void {
    const cfg = this.cfg;
    if (this.petals.length === 0) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (const p of this.petals) {
      // Travel fraction: 0 at spawn → 1 at death (AT: arc-length fraction)
      const travel = Math.max(0, 1 - p.life / p.maxLife);

      // AT vScale: uSize × (1 − travel²)
      const travelDecay = Math.max(0, 1 - travel * travel);
      const petalSize = cfg.basePetalSize * (0.4 + 0.6 * p.bright) * travelDecay;

      if (petalSize < 0.3) continue;

      const color = petalColor(travel, p.bright, p.colorIdx);

      // ── AT matcap sphere-map glow (Canvas 2D approximation) ────────────
      // Radial gradient: hot core → soft edge
      const outerR = petalSize * 2.0;
      const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, outerR);
      grd.addColorStop(0, color);
      grd.addColorStop(0.4, color);
      grd.addColorStop(1, 'rgba(0,0,0,0)');

      ctx.beginPath();
      // Slight elliptical stretch along radial axis for petal shape
      ctx.save();
      ctx.translate(p.x, p.y);
      const rotAngle = Math.atan2(p.radialY, p.radialX);
      ctx.rotate(rotAngle);
      ctx.scale(1.0, 0.65);  // petal aspect ratio
      ctx.arc(0, 0, outerR, 0, Math.PI * 2);
      ctx.restore();

      ctx.fillStyle = grd;
      ctx.globalAlpha = cfg.opacity;
      ctx.fill();
    }

    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.restore();
  }

  // ── GPU bridge ─────────────────────────────────────────────────────────────

  /**
   * Convert active petal bursts into ephemeral FlowerEdgeSpline[] suitable
   * for rendering with ATFlowerParticleRenderer.
   *
   * Each burst generates a short radial spline from the contact origin to
   * the petal's current position, enabling the GPU pipeline to apply the
   * full FlowerParticleShader with matcap shading and spiral motion.
   *
   * @returns Array of FlowerEdgeSpline describing current burst trajectories.
   */
  toFlowerEdgeSplines(): FlowerEdgeSpline[] {
    const splines: FlowerEdgeSpline[] = [];
    for (let i = 0; i < this.petals.length; i++) {
      const p = this.petals[i];
      const travel = Math.max(0, 1 - p.life / p.maxLife);
      if (travel > 0.95) continue;  // nearly dead, skip

      // Two-point spline from radial start to current position
      const startX = p.x - p.radialX * p.maxLife * this.cfg.burstSpeed * 0.3;
      const startY = p.y - p.radialY * p.maxLife * this.cfg.burstSpeed * 0.3;
      const points: FlowerPoint3[] = [
        { x: startX, y: startY, z: 0 },
        { x: p.x, y: p.y, z: 0 },
      ];

      splines.push({
        edgeId: `cfx-${this._frame}-${i}`,
        sourceId: 'collision',
        targetId: 'burst',
        points,
        weight: p.bright,
        species: p.colorIdx,
      });
    }
    return splines;
  }

  // ── Wiring helper ──────────────────────────────────────────────────────────

  /**
   * Subscribe this system to a CollisionEventDispatcher's enter events.
   *
   * Returns an unsubscribe function.  The impulse is estimated from
   * contact depth × `depthMultiplier`.
   *
   * @param dispatcher  The world's CollisionEventDispatcher instance.
   * @param depthMultiplier  Scales contact.depth to impulse.  Default 120.
   * @returns Unsubscribe function.
   */
  subscribe(dispatcher: CollisionEventDispatcher, depthMultiplier = 120): () => void {
    return dispatcher.onCollisionEnter((evt: CollisionEvent) => {
      if (!evt.contact) return;
      const impulse = evt.contact.depth * depthMultiplier;
      this.emit(evt.contact, impulse);
    });
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Number of currently live petal particles. */
  get count(): number {
    return this.petals.length;
  }

  /** Remove all active petals immediately. */
  clear(): void {
    this.petals.length = 0;
  }

  /** Mutate config at runtime (e.g. from a debug panel). */
  configure(overrides: Partial<CollisionFXConfig>): void {
    Object.assign(this.cfg, overrides);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: impulse estimation (mirrors contact-sparks.ts estimateImpulse)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Estimate a proxy impulse magnitude from collision contact geometry.
 *
 * When the full rigid-body solver impulse `j` is available, use that
 * directly.  This helper covers cases where only the CollisionContactInfo
 * is accessible (e.g. when wiring from events alone).
 *
 * @param contact    Contact info from the narrow phase or CollisionEvent.
 * @param relSpeed   Relative speed of the two bodies at the contact point.
 * @param massScale  Tuning constant for visual density.
 * @returns Impulse proxy suitable for passing to `CollisionFXSystem.emit`.
 */
export function estimateFlowerImpulse(
  contact: CollisionContactInfo,
  relSpeed = 0,
  massScale = 1,
): number {
  return (contact.depth * 80 + relSpeed * 0.5) * massScale;
}
