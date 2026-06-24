/**
 * src/lib/sph/decal-projector.ts — M792
 *
 * Decal Projector System — collision impact marks on Cell surfaces
 * ─────────────────────────────────────────────────────────────────────────────
 * When a collision occurs at a Cell surface the system projects a decal
 * (crack, scorch, or ripple pattern) onto the Cell's UV space.  The decal
 * persists for a configurable duration then fades to transparent and is
 * culled from the pool.
 *
 * Architecture
 * ────────────
 *   DecalProjector  (CPU side — manages decal instances + projection maths)
 *     ├─ stamp(contact, impulse, cellId, cellTransform)
 *     │      — project a decal from the collision contact onto the cell
 *     ├─ update(dt)       — advance fade timers, cull dead decals
 *     ├─ draw(ctx)        — Canvas 2D render (screen-space overlay)
 *     ├─ getActiveDecals()— snapshot for external / GPU renderers
 *     └─ subscribe(dispatcher) — wire to CollisionEventDispatcher
 *
 *   DecalGPUPipeline  (WebGPU multi-decal blending pass — optional)
 *     ├─ create(device, format)   — compile WGSL, create BGL + PSO
 *     ├─ uploadDecals(decals)     — write decal SSBO from CPU snapshot
 *     └─ render(encoder, src, dst, w, h) — blit decals onto scene
 *
 * Screen-space projection → Cell UV
 * ──────────────────────────────────
 *   1. The collision contact provides a world-space hit point `P` and
 *      surface normal `N`.
 *   2. A local tangent frame (T, B) is derived from N via the canonical
 *      cross-product (N × up → T, N × T → B).
 *   3. The decal occupies a square quad aligned to (T, B) centred at P
 *      with half-extents proportional to impulse strength.
 *   4. The quad is mapped to [0,1]² UV by offsetting from P and
 *      dividing by the decal's world-size.  For spherical Cells a
 *      gnomonic correction bends the UV toward the sphere centre
 *      so the decal conforms to curvature.
 *   5. The resulting UV rect is stored with the decal and sampled by the
 *      pattern function (crack / scorch / ripple) during draw.
 *
 * Decal types
 * ───────────
 *   • Crack    — fracture lines radiating from the impact centre; drawn as
 *                thin forking polylines whose branch count scales with impulse.
 *   • Scorch   — radial burn gradient: dark char core fading through amber to
 *                transparent; mimics heat-transfer surface deposit.
 *   • Ripple   — concentric ring indentation, like a pebble drop frozen in
 *                the surface.  Uses sin(r) × falloff for ring spacing.
 *
 *   The decal type is chosen automatically from impulse magnitude:
 *     light  → ripple  (subtle surface impression)
 *     medium → scorch  (moderate thermal/impact mark)
 *     heavy  → crack   (violent fracture pattern)
 *   Callers may also force a specific type via `stamp()` options.
 *
 * Impulse → visual mapping
 * ────────────────────────
 *   impulse  →  t = clamp(impulse × impulseScale, 0, 1)
 *   size     =  baseSize × (0.5 + 0.5·√t)
 *   opacity  =  0.3 + 0.7·t
 *   branches =  ceil(3 + 9·t)                    (cracks only)
 *   rings    =  ceil(2 + 6·t)                     (ripples only)
 *   lifetime =  baseFade × (0.6 + 0.4·t)
 *
 * Fade model
 * ──────────
 *   Each decal carries a remaining-life timer.  During the last 40% of
 *   its lifetime the decal's master alpha linearly ramps to zero.
 *   The fade zone fraction is configurable (`fadeZone`, default 0.4).
 *
 * Integration with CollisionWorld / CollisionEvents
 * ──────────────────────────────────────────────────
 *   const decals = new DecalProjector();
 *
 *   dispatcher.onCollisionEnter((evt) => {
 *     if (!evt.contact) return;
 *     const impulse = evt.contact.depth * 120;
 *     decals.stamp(evt.contact, impulse, evt.bodyA, cellTransform);
 *   });
 *
 *   // Each frame:
 *   decals.update(dt);
 *   decals.draw(ctx);       // Canvas 2D overlay
 *   // — or —
 *   pipeline.uploadDecals(decals.getActiveDecals());
 *   pipeline.render(encoder, sceneView, dstView, w, h);
 *
 * Design references
 * ─────────────────
 *   src/lib/sph/collision-fx-system.ts     — impulse→visual, lygia RNG, subscribe
 *   src/lib/sph/collision-shockwave.ts     — screen-space UV mapping, GPU pipeline
 *   src/lib/sph/contact-sparks.ts          — impulseScale convention, lygia hash
 *   src/lib/sph/destruction-system.ts      — fracture geometry, impulse threshold
 *   src/lib/sph/ripple-effect.ts           — wave compositing, WGSL patterns
 *   src/lib/sph/collision/CollisionEvents.ts — CollisionContactInfo types
 */


import type {
} from './collision/CollisionEvents';
import type { CollisionEventDispatcher } from './collision/CollisionEvents';

// [orphan3]   CollisionContactInfo,
// [orphan3]   CollisionEvent,

// ─────────────────────────────────────────────────────────────────────────────
// Lygia random port (shared convention with contact-sparks / collision-fx)
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

/** Transform describing a Cell's position, radius, and rotation in world space. */
export interface CellTransform {
  /** World-space centre X. */
  cx: number;
  /** World-space centre Y. */
  cy: number;
  /** Cell radius (for gnomonic curvature correction). */
  radius: number;
  /** Rotation angle (radians). */
  angle: number;
}

/** Decal pattern type. */
export type DecalType = 'crack' | 'scorch' | 'ripple';

/** Configuration for the decal projector system. */
export interface DecalProjectorConfig {
  /**
   * Impulse scale: maps raw impulse magnitude to normalised [0, 1].
   * Default 0.008 — tune to match your velocity / depth units.
   */
  impulseScale: number;

  /**
   * Base decal half-extent in world units (before impulse scaling).
   * Final size = baseSize × (0.5 + 0.5·√t).
   * Default 20.
   */
  baseSize: number;

  /**
   * Maximum decal lifetime (seconds) at full impulse.
   * Actual lifetime = baseFade × (0.6 + 0.4·t).
   * Default 3.0.
   */
  baseFade: number;

  /**
   * Fraction of lifetime over which the decal fades to transparent.
   * 0.4 = last 40% of life is a linear alpha ramp-down.
   * Default 0.4.
   */
  fadeZone: number;

  /**
   * Maximum concurrent decals.  Oldest decal is culled when limit is reached.
   * Default 64.
   */
  maxDecals: number;

  /**
   * Minimum impulse threshold (normalised).
   * Collisions weaker than this produce no decal.
   * Default 0.03.
   */
  minThreshold: number;

  /**
   * Impulse threshold (normalised) below which decal type defaults to 'ripple'.
   * Default 0.25.
   */
  rippleThreshold: number;

  /**
   * Impulse threshold (normalised) above which decal type defaults to 'crack'.
   * Below this (but above rippleThreshold) → 'scorch'.
   * Default 0.65.
   */
  crackThreshold: number;

  /**
   * Maximum crack branches for the heaviest impacts.
   * Actual count = ceil(3 + 9·t).
   * Default 12.
   */
  maxCrackBranches: number;

  /**
   * Maximum concentric rings for ripple decals.
   * Actual count = ceil(2 + 6·t).
   * Default 8.
   */
  maxRippleRings: number;

  /**
   * Global master alpha multiplier applied when drawing.
   * Default 1.
   */
  opacity: number;

  /**
   * Gnomonic curvature correction strength for spherical Cells.
   * 0 = flat projection, 1 = full gnomonic correction.
   * Default 0.7.
   */
  curvatureCorrection: number;

  /**
   * Scorch inner char radius as fraction of decal half-extent.
   * Default 0.3.
   */
  scorchCoreRadius: number;

  /**
   * Crack line width in pixels.
   * Default 1.5.
   */
  crackLineWidth: number;
}

const DEFAULT_CONFIG: DecalProjectorConfig = {
  impulseScale:        0.008,
  baseSize:            20,
  baseFade:            3.0,
  fadeZone:            0.4,
  maxDecals:           64,
  minThreshold:        0.03,
  rippleThreshold:     0.25,
  crackThreshold:      0.65,
  maxCrackBranches:    12,
  maxRippleRings:      8,
  opacity:             1.0,
  curvatureCorrection: 0.7,
  scorchCoreRadius:    0.3,
  crackLineWidth:      1.5,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal decal instance
// ─────────────────────────────────────────────────────────────────────────────

/** Per-crack-branch polyline segment. */
interface CrackBranch {
  /** Normalised direction from decal centre in UV-local space. */
  dx: number;
  dy: number;
  /** Length as fraction of decal half-extent. */
  length: number;
  /** Fork point as fraction of parent length (0 = no fork). */
  forkAt: number;
  /** Fork direction offset angle (radians). */
  forkAngle: number;
  /** Fork sub-branch length fraction. */
  forkLength: number;
}

/** A single projected decal on a Cell surface. */
interface Decal {
  /** World-space centre of the decal. */
  wx: number;
  wy: number;

  /** UV-space centre on the Cell surface (projected). */
  u: number;
  v: number;

  /** Half-extent of the decal in world units. */
  halfExtent: number;

  /** Tangent frame basis vectors (T, B) in world space. */
  tx: number;
  ty: number;
  bx: number;
  by: number;

  /** Decal pattern type. */
  type: DecalType;

  /** Remaining lifetime (seconds). */
  life: number;
  /** Total lifetime for fade computation. */
  maxLife: number;

  /** Impulse-normalised intensity [0, 1]. */
  impulseT: number;

  /** Cell body ID this decal is attached to. */
  cellId: number;

  /** Cell transform snapshot at stamp time (for re-projection on move). */
  cellCx: number;
  cellCy: number;
  cellRadius: number;
  cellAngle: number;

  /** Per-decal random seed. */
  seed: number;

  /** Pre-generated crack branches (only for type === 'crack'). */
  branches: CrackBranch[];

  /** Number of concentric rings (only for type === 'ripple'). */
  ringCount: number;
}

/**
 * GPU-uploadable decal snapshot.
 * Packed as 12 floats per decal for the WGSL storage buffer.
 */
export interface DecalGPU {
  /** World-space centre. */
  wx: number;
  wy: number;
  /** UV centre on Cell surface. */
  u: number;
  v: number;
  /** Half-extent (world). */
  halfExtent: number;
  /** Current opacity (fade-adjusted). */
  alpha: number;
  /** Tangent basis T. */
  tx: number;
  ty: number;
  /** Bitangent basis B. */
  bx: number;
  by: number;
  /** Decal type encoded: 0 = crack, 1 = scorch, 2 = ripple. */
  typeIndex: number;
  /** Random seed for shader-side pattern generation. */
  seed: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour palettes for each decal type
// ─────────────────────────────────────────────────────────────────────────────

/** Scorch colour stops: dark char core → amber edge → transparent. */
const SCORCH_STOPS: [number, number, number, number][] = [
  [30,  20,  15,  1.0],   // t=0  dark char
  [60,  35,  20,  0.9],   // t=0.25  deep brown
  [120, 70,  25,  0.7],   // t=0.5   amber
  [180, 120, 50,  0.4],   // t=0.75  light amber
  [200, 160, 100, 0.0],   // t=1.0   transparent edge
];

/** Crack line colour: dark + subtle highlight on wider cracks. */
const CRACK_COLOR: [number, number, number] = [25, 18, 35];
const CRACK_GLOW:  [number, number, number] = [120, 80, 180];

/** Ripple indentation colour (cool blue-grey). */
const RIPPLE_COLOR: [number, number, number] = [80, 100, 140];

// ─────────────────────────────────────────────────────────────────────────────
// Colour interpolation helpers
// ─────────────────────────────────────────────────────────────────────────────

function scorchColorAt(t: number): string {
  const stops = SCORCH_STOPS;
  const idx = Math.min(t * (stops.length - 1), stops.length - 1 - 1e-9);
  const lo  = Math.floor(idx);
  const hi  = Math.min(lo + 1, stops.length - 1);
  const f   = idx - lo;

  const r = stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f;
  const g = stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f;
  const b = stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f;
  const a = stops[lo][3] + (stops[hi][3] - stops[lo][3]) * f;

  return `rgba(${r | 0},${g | 0},${b | 0},${a.toFixed(3)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Crack branch generation
// ─────────────────────────────────────────────────────────────────────────────

function generateCrackBranches(
  count: number,
  seed: number,
  impulseT: number,
): CrackBranch[] {
  const branches: CrackBranch[] = [];
  const angleStep = (Math.PI * 2) / count;

  for (let i = 0; i < count; i++) {
    const baseSeed = seed * 100 + i;
    const r0 = lygiaRandom(baseSeed);
    const r1 = lygiaRandom(baseSeed + 7.3);
    const r2 = lygiaRandom(baseSeed + 13.1);
    const r3 = lygiaRandom(baseSeed + 19.9);
    const r4 = lygiaRandom(baseSeed + 27.7);

    // Base angle with jitter
    const angle = angleStep * i + (r0 - 0.5) * angleStep * 0.6;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);

    // Length: 0.4–1.0 of half-extent, biased by impulse
    const length = 0.4 + 0.6 * r1 * (0.5 + 0.5 * impulseT);

    // Fork: heavier impacts fork more often
    const forkChance = 0.2 + 0.5 * impulseT;
    const hasFork = r2 < forkChance;

    branches.push({
      dx,
      dy,
      length,
      forkAt:     hasFork ? 0.3 + 0.4 * r3 : 0,
      forkAngle:  hasFork ? (r4 - 0.5) * Math.PI * 0.6 : 0,
      forkLength: hasFork ? 0.3 + 0.4 * r3 * (1 - r3) : 0,
    });
  }

  return branches;
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen-space → Cell UV projection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Project a world-space point onto a Cell's local UV space.
 *
 * The projection uses a tangent-plane model for flat Cells and applies
 * gnomonic curvature correction for spherical Cells.  The result is
 * a [u, v] pair in approximately [0, 1]² centred on the Cell.
 *
 * @param px        World-space hit point X.
 * @param py        World-space hit point Y.
 * @param cell      Cell transform (centre, radius, angle).
 * @param curvature Gnomonic correction strength [0, 1].
 * @returns         [u, v] on the Cell surface.
 */
function worldToUV(
  px: number,
  py: number,
  cell: CellTransform,
  curvature: number,
): [number, number] {
  // Offset from cell centre
  let dx = px - cell.cx;
  let dy = py - cell.cy;

  // Un-rotate into cell-local space
  if (cell.angle !== 0) {
    const cosA = Math.cos(-cell.angle);
    const sinA = Math.sin(-cell.angle);
    const rx = dx * cosA - dy * sinA;
    const ry = dx * sinA + dy * cosA;
    dx = rx;
    dy = ry;
  }

  // Normalise by cell radius → [-1, 1] range
  const r = cell.radius > 0 ? cell.radius : 1;
  let u = dx / r;
  let v = dy / r;

  // Gnomonic curvature correction for spherical cells:
  // Maps the flat tangent-plane projection onto the sphere surface.
  // At curvature=1 this is a full gnomonic inverse; at 0 it's flat.
  if (curvature > 0) {
    const dist2 = u * u + v * v;
    if (dist2 > 1e-6) {
      // Gnomonic: project onto unit sphere then back to UV
      // θ = atan(√(u²+v²))  →  scale = tan(θ)/√(u²+v²) ≈ 1 for small dist
      const dist = Math.sqrt(dist2);
      const theta = Math.atan(dist);
      const scale = 1 + curvature * (theta / dist - 1);
      u *= scale;
      v *= scale;
    }
  }

  // Map from [-1, 1] to [0, 1]
  return [u * 0.5 + 0.5, v * 0.5 + 0.5];
}

/**
 * Build a tangent frame (T, B) from the collision normal.
 *
 * In 2D the tangent is simply the 90° rotation of the normal, and the
 * bitangent is the normal itself.  This gives us a local basis for
 * orienting the decal quad on the surface.
 */
function buildTangentFrame(nx: number, ny: number): { tx: number; ty: number; bx: number; by: number } {
  // T = perp(N) = (-ny, nx),  B = N = (nx, ny)
  const len = Math.sqrt(nx * nx + ny * ny) || 1;
  const nnx = nx / len;
  const nny = ny / len;
  return {
    tx: -nny,
    ty:  nnx,
    bx:  nnx,
    by:  nny,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// DecalProjector  (CPU side — decal lifecycle + Canvas 2D rendering)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages projected impact decals (crack / scorch / ripple) on Cell surfaces.
 * Collision events stamp decals whose opacity fades over time; dead decals
 * are automatically culled from the pool.
 *
 * @example
 * ```ts
 * const decals = new DecalProjector({ impulseScale: 0.01 });
 *
 * dispatcher.onCollisionEnter((evt) => {
 *   if (!evt.contact) return;
 *   const impulse = evt.contact.depth * 120;
 *   decals.stamp(evt.contact, impulse, evt.bodyA, {
 *     cx: body.x, cy: body.y, radius: 30, angle: body.angle,
 *   });
 * });
 *
 * // Animation loop:
 * decals.update(dt);
 * decals.draw(ctx);
 * ```
 */
export class DecalProjector {
  private decals: Decal[] = [];
  private cfg: DecalProjectorConfig;
  private _frame = 0;

  constructor(config: Partial<DecalProjectorConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Stamp a decal at a collision contact point on a Cell surface.
   *
   * @param contact       CollisionContactInfo from the narrow phase.
   * @param impulse       Raw impulse magnitude.
   * @param cellId        Body ID of the Cell receiving the decal.
   * @param cellTransform Cell's current world-space transform.
   * @param forceType     Override automatic type selection.
   */
  stamp(
    contact: CollisionContactInfo,
    impulse: number,
    cellId: number,
    cellTransform: CellTransform,
    forceType?: DecalType,
  ): void {
    const cfg = this.cfg;

    // Map impulse → normalised [0, 1]
    const t = Math.min(impulse * cfg.impulseScale, 1.0);
    if (t < cfg.minThreshold) return;

    // Cull oldest decal if at capacity
    if (this.decals.length >= cfg.maxDecals) {
      // Find and remove the oldest (smallest remaining life)
      let minIdx = 0;
      let minLife = this.decals[0].life;
      for (let i = 1; i < this.decals.length; i++) {
        if (this.decals[i].life < minLife) {
          minLife = this.decals[i].life;
          minIdx = i;
        }
      }
      this.decals.splice(minIdx, 1);
    }

    // Contact point: midpoint of pointA and pointB
    const px = (contact.pointA.x + contact.pointB.x) * 0.5;
    const py = (contact.pointA.y + contact.pointB.y) * 0.5;

    // Project to Cell UV
    const [u, v] = worldToUV(px, py, cellTransform, cfg.curvatureCorrection);

    // Build tangent frame from contact normal
    const frame = buildTangentFrame(contact.normal.x, contact.normal.y);

    // Decal size scales with impulse
    const halfExtent = cfg.baseSize * (0.5 + 0.5 * Math.sqrt(t));

    // Lifetime scales with impulse
    const life = cfg.baseFade * (0.6 + 0.4 * t);

    // Auto-select decal type from impulse tier
    let type: DecalType;
    if (forceType) {
      type = forceType;
    } else if (t < cfg.rippleThreshold) {
      type = 'ripple';
    } else if (t < cfg.crackThreshold) {
      type = 'scorch';
    } else {
      type = 'crack';
    }

    // Per-decal random seed
    const seed = this._frame * 1000 + this.decals.length + lygiaRandom2(px * 0.01, py * 0.01) * 999;

    // Pre-generate crack branches (deferred cost — only for crack type)
    const branchCount = type === 'crack'
      ? Math.ceil(3 + 9 * t)
      : 0;
    const branches = type === 'crack'
      ? generateCrackBranches(Math.min(branchCount, cfg.maxCrackBranches), seed, t)
      : [];

    // Ripple ring count
    const ringCount = type === 'ripple'
      ? Math.ceil(2 + 6 * t)
      : 0;

    this.decals.push({
      wx: px,
      wy: py,
      u,
      v,
      halfExtent,
      tx: frame.tx,
      ty: frame.ty,
      bx: frame.bx,
      by: frame.by,
      type,
      life,
      maxLife: life,
      impulseT: t,
      cellId,
      cellCx: cellTransform.cx,
      cellCy: cellTransform.cy,
      cellRadius: cellTransform.radius,
      cellAngle: cellTransform.angle,
      seed,
      branches,
      ringCount,
    });

    this._frame++;
  }

  /**
   * Convenience: stamp from a CollisionEvent directly.
   *
   * @param event         CollisionEvent (only 'enter' phase events with contact are used).
   * @param impulse       Raw impulse magnitude.
   * @param cellTransform Cell transform for the target body.
   * @param forceType     Override automatic type selection.
   */
  stampFromEvent(
    event: CollisionEvent,
    impulse: number,
    cellTransform: CellTransform,
    forceType?: DecalType,
  ): void {
    if (!event.contact) return;
    this.stamp(event.contact, impulse, event.bodyA, cellTransform, forceType);
  }

  /**
   * Convenience: stamp using raw position + normal.
   *
   * @param point         World-space contact position.
   * @param normal        Unit contact normal.
   * @param impulse       Raw impulse magnitude.
   * @param cellId        Body ID of the Cell.
   * @param cellTransform Cell transform.
   * @param forceType     Override automatic type selection.
   */
  stampAt(
    point: Vec2,
    normal: Vec2,
    impulse: number,
    cellId: number,
    cellTransform: CellTransform,
    forceType?: DecalType,
  ): void {
    this.stamp(
      {
        normal: { x: normal.x, y: normal.y },
        depth: impulse * this.cfg.impulseScale * 10,
        pointA: { x: point.x, y: point.y },
        pointB: { x: point.x, y: point.y },
      },
      impulse,
      cellId,
      cellTransform,
      forceType,
    );
  }

  /**
   * Advance all decal fade timers by `dt` seconds.
   * Dead decals (life ≤ 0) are culled via swap-remove.
   */
  update(dt: number): void {
    for (let i = this.decals.length - 1; i >= 0; i--) {
      this.decals[i].life -= dt;
      if (this.decals[i].life <= 0) {
        // Swap-remove for O(1) pool management
        const last = this.decals.length - 1;
        if (i !== last) {
          this.decals[i] = this.decals[last];
        }
        this.decals.pop();
      }
    }
  }

  /**
   * Update the Cell transform for a specific body (call when the Cell moves).
   *
   * Decals attached to the Cell are re-projected so they stick to the
   * surface as the Cell translates and rotates.
   *
   * @param cellId        Body ID of the Cell.
   * @param cellTransform Updated Cell transform.
   */
  updateCellTransform(cellId: number, cellTransform: CellTransform): void {
    for (const d of this.decals) {
      if (d.cellId !== cellId) continue;

      // Compute the decal's local offset relative to the original cell centre
      const odx = d.wx - d.cellCx;
      const ody = d.wy - d.cellCy;

      // Un-rotate from old cell frame
      const cosOld = Math.cos(-d.cellAngle);
      const sinOld = Math.sin(-d.cellAngle);
      const localX = odx * cosOld - ody * sinOld;
      const localY = odx * sinOld + ody * cosOld;

      // Re-rotate into new cell frame
      const cosNew = Math.cos(cellTransform.angle);
      const sinNew = Math.sin(cellTransform.angle);
      const newDx = localX * cosNew - localY * sinNew;
      const newDy = localX * sinNew + localY * cosNew;

      // Update world position
      d.wx = cellTransform.cx + newDx;
      d.wy = cellTransform.cy + newDy;

      // Update stored cell transform
      d.cellCx = cellTransform.cx;
      d.cellCy = cellTransform.cy;
      d.cellRadius = cellTransform.radius;
      d.cellAngle = cellTransform.angle;

      // Re-project UV
      const [u, v] = worldToUV(
        d.wx, d.wy,
        cellTransform,
        this.cfg.curvatureCorrection,
      );
      d.u = u;
      d.v = v;
    }
  }

  /**
   * Draw all active decals onto a Canvas 2D context.
   *
   * Each decal is rendered at its world-space position using the
   * appropriate pattern function (crack / scorch / ripple).
   * The context should be in world-coordinate space (i.e. apply your
   * camera transform before calling this).
   */
  draw(ctx: CanvasRenderingContext2D): void {
    const cfg = this.cfg;
    if (this.decals.length === 0) return;

    ctx.save();

    for (const d of this.decals) {
      // Compute fade alpha
      const alpha = this._computeAlpha(d) * cfg.opacity;
      if (alpha < 0.005) continue;

      ctx.save();

      // Translate + rotate into decal local space
      ctx.translate(d.wx, d.wy);
      const angle = Math.atan2(d.ty, d.tx);
      ctx.rotate(angle);

      ctx.globalAlpha = alpha;

      switch (d.type) {
        case 'crack':
          this._drawCrack(ctx, d);
          break;
        case 'scorch':
          this._drawScorch(ctx, d);
          break;
        case 'ripple':
          this._drawRipple(ctx, d);
          break;
      }

      ctx.restore();
    }

    ctx.restore();
  }

  /**
   * Return a GPU-uploadable snapshot of all active decals.
   * Each decal is packed as a DecalGPU struct (12 floats).
   */
  getActiveDecals(): DecalGPU[] {
    const result: DecalGPU[] = [];
    for (const d of this.decals) {
      const alpha = this._computeAlpha(d) * this.cfg.opacity;
      if (alpha < 0.005) continue;

      result.push({
        wx: d.wx,
        wy: d.wy,
        u: d.u,
        v: d.v,
        halfExtent: d.halfExtent,
        alpha,
        tx: d.tx,
        ty: d.ty,
        bx: d.bx,
        by: d.by,
        typeIndex: d.type === 'crack' ? 0 : d.type === 'scorch' ? 1 : 2,
        seed: d.seed,
      });
    }
    return result;
  }

  // ── Wiring helper ──────────────────────────────────────────────────────────

  /**
   * Subscribe this system to a CollisionEventDispatcher's enter events.
   *
   * The caller must supply a `getCellTransform` function that returns
   * the CellTransform for a given body ID, since the decal projector
   * needs to know the Cell's position/radius to project onto its surface.
   *
   * Returns an unsubscribe function.
   *
   * @param dispatcher       The world's CollisionEventDispatcher instance.
   * @param getCellTransform Lookup function: bodyId → CellTransform | null.
   * @param depthMultiplier  Scales contact.depth to impulse.  Default 120.
   * @returns Unsubscribe function.
   */
  subscribe(
    dispatcher: CollisionEventDispatcher,
    getCellTransform: (bodyId: number) => CellTransform | null,
    depthMultiplier = 120,
  ): () => void {
    return dispatcher.onCollisionEnter((evt: CollisionEvent) => {
      if (!evt.contact) return;
      const impulse = evt.contact.depth * depthMultiplier;

      // Try to stamp on bodyA first, then bodyB
      const transformA = getCellTransform(evt.bodyA);
      if (transformA) {
        this.stamp(evt.contact, impulse, evt.bodyA, transformA);
      }

      const transformB = getCellTransform(evt.bodyB);
      if (transformB) {
        this.stamp(evt.contact, impulse, evt.bodyB, transformB);
      }
    });
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  /** Number of currently live decals. */
  get count(): number {
    return this.decals.length;
  }

  /** Remove all active decals immediately. */
  clear(): void {
    this.decals.length = 0;
  }

  /** Mutate config at runtime (e.g. from a debug panel). */
  configure(overrides: Partial<DecalProjectorConfig>): void {
    Object.assign(this.cfg, overrides);
  }

  /** Get all decals attached to a specific Cell body. */
  getDecalsForCell(cellId: number): DecalGPU[] {
    return this.getActiveDecals().filter(d => {
      const decal = this.decals.find(dd => dd.wx === d.wx && dd.wy === d.wy && dd.cellId === cellId);
      return !!decal;
    });
  }

  /** Remove all decals attached to a specific Cell body. */
  clearCell(cellId: number): void {
    for (let i = this.decals.length - 1; i >= 0; i--) {
      if (this.decals[i].cellId === cellId) {
        const last = this.decals.length - 1;
        if (i !== last) {
          this.decals[i] = this.decals[last];
        }
        this.decals.pop();
      }
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /** Compute current alpha for a decal (fade zone ramp). */
  private _computeAlpha(d: Decal): number {
    const lifeRatio = d.life / d.maxLife;  // 1 = fresh, 0 = dead
    const fadeZone = this.cfg.fadeZone;

    if (lifeRatio > fadeZone) {
      // Full opacity zone
      return 0.3 + 0.7 * d.impulseT;
    }
    // Linear ramp in fade zone
    const fadeT = lifeRatio / fadeZone;
    return (0.3 + 0.7 * d.impulseT) * fadeT;
  }

  /**
   * Draw a crack decal: radiating fracture lines from the centre.
   * Each branch is a thin line that may fork partway along its length.
   */
  private _drawCrack(ctx: CanvasRenderingContext2D, d: Decal): void {
    const cfg = this.cfg;
    const r = d.halfExtent;

    ctx.strokeStyle = `rgb(${CRACK_COLOR[0]},${CRACK_COLOR[1]},${CRACK_COLOR[2]})`;
    ctx.lineWidth = cfg.crackLineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const b of d.branches) {
      const endX = b.dx * b.length * r;
      const endY = b.dy * b.length * r;

      // Main branch
      ctx.beginPath();
      ctx.moveTo(0, 0);

      // Slight curve: add a mid-point with jitter
      const midJitter = lygiaRandom(d.seed + b.dx * 100) * 0.15 - 0.075;
      const midX = endX * 0.5 + b.dy * midJitter * r;
      const midY = endY * 0.5 - b.dx * midJitter * r;
      ctx.quadraticCurveTo(midX, midY, endX, endY);
      ctx.stroke();

      // Fork sub-branch (if present)
      if (b.forkAt > 0) {
        const forkX = b.dx * b.forkAt * b.length * r;
        const forkY = b.dy * b.forkAt * b.length * r;

        const forkDx = Math.cos(Math.atan2(b.dy, b.dx) + b.forkAngle);
        const forkDy = Math.sin(Math.atan2(b.dy, b.dx) + b.forkAngle);
        const forkEndX = forkX + forkDx * b.forkLength * r;
        const forkEndY = forkY + forkDy * b.forkLength * r;

        ctx.beginPath();
        ctx.moveTo(forkX, forkY);
        ctx.lineTo(forkEndX, forkEndY);
        ctx.lineWidth = cfg.crackLineWidth * 0.7;
        ctx.stroke();
        ctx.lineWidth = cfg.crackLineWidth;
      }
    }

    // Subtle glow at the impact centre
    const glowR = r * 0.15 * d.impulseT;
    if (glowR > 0.5) {
      const grd = ctx.createRadialGradient(0, 0, 0, 0, 0, glowR);
      grd.addColorStop(0, `rgba(${CRACK_GLOW[0]},${CRACK_GLOW[1]},${CRACK_GLOW[2]},0.3)`);
      grd.addColorStop(1, `rgba(${CRACK_GLOW[0]},${CRACK_GLOW[1]},${CRACK_GLOW[2]},0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(0, 0, glowR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * Draw a scorch decal: radial burn gradient.
   * Dark char core fading through amber to transparent at the edges.
   */
  private _drawScorch(ctx: CanvasRenderingContext2D, d: Decal): void {
    const r = d.halfExtent;
    const coreR = r * this.cfg.scorchCoreRadius;

    // Build multi-stop radial gradient
    const grd = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    const stopCount = SCORCH_STOPS.length;
    for (let i = 0; i < stopCount; i++) {
      const t = i / (stopCount - 1);
      grd.addColorStop(t, scorchColorAt(t));
    }

    // Main scorch circle
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Inner char ring for texture (slightly darker, irregular)
    const charGrd = ctx.createRadialGradient(0, 0, coreR * 0.3, 0, 0, coreR);
    charGrd.addColorStop(0, `rgba(15,10,8,0.5)`);
    charGrd.addColorStop(0.6, `rgba(25,18,12,0.3)`);
    charGrd.addColorStop(1, `rgba(40,28,20,0)`);

    ctx.beginPath();
    ctx.arc(0, 0, coreR, 0, Math.PI * 2);
    ctx.fillStyle = charGrd;
    ctx.fill();

    // Noise texture overlay — radial speckle using lygia hash
    const speckleCount = Math.ceil(8 + 16 * d.impulseT);
    ctx.globalCompositeOperation = 'multiply';
    for (let i = 0; i < speckleCount; i++) {
      const [sx, sy] = lygiaRandom22(d.seed + i * 3.7, d.seed + i * 7.1);
      const angle = sx * Math.PI * 2;
      const dist = sy * r * 0.8;
      const speckR = 1 + lygiaRandom(d.seed + i * 11.3) * 3;
      const speckAlpha = 0.1 + 0.2 * lygiaRandom(d.seed + i * 17.9);

      ctx.beginPath();
      ctx.arc(
        Math.cos(angle) * dist,
        Math.sin(angle) * dist,
        speckR, 0, Math.PI * 2,
      );
      ctx.fillStyle = `rgba(20,15,10,${speckAlpha.toFixed(3)})`;
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  /**
   * Draw a ripple decal: concentric ring indentation.
   * sin(r) × falloff produces evenly spaced rings that fade outward.
   */
  private _drawRipple(ctx: CanvasRenderingContext2D, d: Decal): void {
    const r = d.halfExtent;
    const rings = d.ringCount;

    ctx.strokeStyle = `rgb(${RIPPLE_COLOR[0]},${RIPPLE_COLOR[1]},${RIPPLE_COLOR[2]})`;
    ctx.lineWidth = 1.0;

    for (let i = 1; i <= rings; i++) {
      const ringFrac = i / (rings + 1);
      const ringR = ringFrac * r;

      // Alpha fades outward with a sin(π·ringFrac) envelope
      const envelope = Math.sin(Math.PI * ringFrac);
      const ringAlpha = envelope * 0.6 * d.impulseT;

      if (ringAlpha < 0.01) continue;

      ctx.globalAlpha = ringAlpha * this.cfg.opacity * this._computeAlpha(d) / (0.3 + 0.7 * d.impulseT);
      ctx.beginPath();
      ctx.arc(0, 0, ringR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Centre dimple: small filled circle
    const dimpleR = r * 0.08;
    if (dimpleR > 0.3) {
      const dimpleGrd = ctx.createRadialGradient(0, 0, 0, 0, 0, dimpleR);
      dimpleGrd.addColorStop(0, `rgba(${RIPPLE_COLOR[0]},${RIPPLE_COLOR[1]},${RIPPLE_COLOR[2]},0.4)`);
      dimpleGrd.addColorStop(1, `rgba(${RIPPLE_COLOR[0]},${RIPPLE_COLOR[1]},${RIPPLE_COLOR[2]},0)`);
      ctx.globalAlpha = this.cfg.opacity * this._computeAlpha(d) / (0.3 + 0.7 * d.impulseT);
      ctx.beginPath();
      ctx.arc(0, 0, dimpleR, 0, Math.PI * 2);
      ctx.fillStyle = dimpleGrd;
      ctx.fill();
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DecalGPUPipeline  (WebGPU full-screen decal blending pass)
// ─────────────────────────────────────────────────────────────────────────────

/** WGSL shader for the full-screen decal compositing pass. */
const DECAL_COMPOSITE_WGSL = /* wgsl */`
// ── Decal composite shader ────────────────────────────────────────────────
// Reads the scene colour texture, blends projected decals on top, and
// writes to the destination render target.

struct DecalData {
  wx:         f32,
  wy:         f32,
  u:          f32,
  v:          f32,
  halfExtent: f32,
  alpha:      f32,
  tx:         f32,
  ty:         f32,
  bx:         f32,
  by:         f32,
  typeIndex:  f32,
  seed:       f32,
};

struct Uniforms {
  resolution: vec2f,
  decalCount: f32,
  _pad:       f32,
};

@group(0) @binding(0) var sceneTex:   texture_2d<f32>;
@group(0) @binding(1) var sceneSamp:  sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> decals: array<DecalData>;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

// Fullscreen triangle
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VertOut {
  var out: VertOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

// ── Lygia sin-less hash (GPU side) ───────────────────────────────────────
fn lygiaHash(p: f32) -> f32 {
  var x = fract(p * 0.1031);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

fn lygiaHash2(sx: f32, sy: f32) -> f32 {
  var p3x = fract(sx * 0.1031);
  var p3y = fract(sy * 0.1030);
  var p3z = fract(sx * 0.0973);
  let d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33);
  p3x += d; p3y += d; p3z += d;
  return fract((p3x + p3y) * p3z);
}

// ── Pattern functions ────────────────────────────────────────────────────

// Crack pattern: radial fracture lines
fn crackPattern(localUV: vec2f, seed: f32) -> f32 {
  let angle = atan2(localUV.y, localUV.x);
  let dist = length(localUV);
  if (dist > 1.0) { return 0.0; }

  var acc = 0.0;
  let branchCount = 8.0;
  for (var i = 0.0; i < branchCount; i += 1.0) {
    let branchAngle = i / branchCount * 6.283185 + lygiaHash(seed + i) * 0.4;
    let angleDiff = abs(angle - branchAngle);
    let wrapped = min(angleDiff, 6.283185 - angleDiff);
    let width = 0.03 + 0.02 * lygiaHash(seed + i * 7.1);
    let line = smoothstep(width, 0.0, wrapped) * (1.0 - dist);
    acc = max(acc, line);
  }
  return acc;
}

// Scorch pattern: radial burn gradient
fn scorchPattern(localUV: vec2f, seed: f32) -> f32 {
  let dist = length(localUV);
  if (dist > 1.0) { return 0.0; }

  // Noise-perturbed distance
  let noise = lygiaHash2(localUV.x * 5.0 + seed, localUV.y * 5.0 + seed * 0.7) * 0.15;
  let d = dist + noise;
  return clamp(1.0 - d, 0.0, 1.0);
}

// Ripple pattern: concentric rings
fn ripplePattern(localUV: vec2f, seed: f32) -> f32 {
  let dist = length(localUV);
  if (dist > 1.0) { return 0.0; }

  let rings = 6.0;
  let wave = sin(dist * rings * 3.14159) * 0.5 + 0.5;
  let falloff = 1.0 - dist;
  return wave * falloff * 0.5;
}

@fragment fn fs(inp: VertOut) -> @location(0) vec4f {
  let sceneColor = textureSampleLevel(sceneTex, sceneSamp, inp.uv, 0.0);
  var result = sceneColor;
  let res = uniforms.resolution;
  let count = i32(uniforms.decalCount);

  for (var i = 0; i < count; i++) {
    let d = decals[i];

    // Fragment world position from UV
    let fragW = vec2f(inp.uv.x * res.x, inp.uv.y * res.y);

    // Offset from decal centre
    let offset = fragW - vec2f(d.wx, d.wy);

    // Project onto tangent frame to get decal-local coordinates
    let localX = dot(offset, vec2f(d.tx, d.ty));
    let localY = dot(offset, vec2f(d.bx, d.by));

    // Normalise by half-extent → [-1, 1]
    let localUV = vec2f(localX, localY) / d.halfExtent;

    // Reject fragments outside the decal quad
    if (abs(localUV.x) > 1.0 || abs(localUV.y) > 1.0) { continue; }

    // Evaluate pattern
    var pattern = 0.0;
    let typeIdx = i32(d.typeIndex);
    if (typeIdx == 0) {
      pattern = crackPattern(localUV, d.seed);
    } else if (typeIdx == 1) {
      pattern = scorchPattern(localUV, d.seed);
    } else {
      pattern = ripplePattern(localUV, d.seed);
    }

    let decalAlpha = pattern * d.alpha;
    if (decalAlpha < 0.005) { continue; }

    // Blend colour: cracks darken, scorch warms, ripples cool
    var decalColor = vec3f(0.0);
    if (typeIdx == 0) {
      // Crack: darken + subtle purple glow
      decalColor = mix(sceneColor.rgb, vec3f(0.1, 0.07, 0.14), decalAlpha);
    } else if (typeIdx == 1) {
      // Scorch: char brown overlay
      decalColor = mix(sceneColor.rgb, vec3f(0.12, 0.08, 0.04), decalAlpha);
    } else {
      // Ripple: cool indentation
      decalColor = mix(sceneColor.rgb, sceneColor.rgb * 0.7, decalAlpha);
    }

    result = vec4f(decalColor, result.a);
  }

  return result;
}
`;

/**
 * WebGPU full-screen pass that composites projected decals over the scene.
 *
 * Reads a storage buffer of DecalData structs and evaluates procedural
 * patterns (crack / scorch / ripple) in the fragment shader, blending
 * them onto the source scene texture.
 *
 * @example
 * ```ts
 * const pipeline = await DecalGPUPipeline.create(device, canvasFormat);
 *
 * // Each frame:
 * pipeline.uploadDecals(projector.getActiveDecals());
 * pipeline.render(encoder, sceneView, dstView, width, height);
 * ```
 */
export class DecalGPUPipeline {
  private device:          GPUDevice;
  private pipeline:        GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler:         GPUSampler;
  private uniformBuffer:   GPUBuffer;
  private decalBuffer:     GPUBuffer;
  private decalCapacity:   number;

  private constructor(
    device:    GPUDevice,
    pipeline:  GPURenderPipeline,
    bgl:       GPUBindGroupLayout,
    sampler:   GPUSampler,
    uBuf:      GPUBuffer,
    dBuf:      GPUBuffer,
    capacity:  number,
  ) {
    this.device          = device;
    this.pipeline        = pipeline;
    this.bindGroupLayout = bgl;
    this.sampler         = sampler;
    this.uniformBuffer   = uBuf;
    this.decalBuffer     = dBuf;
    this.decalCapacity   = capacity;
  }

  /**
   * Factory: compile the WGSL shader module, create pipeline state objects,
   * and allocate GPU buffers.
   *
   * @param device  WebGPU device.
   * @param format  Canvas / render-target texture format.
   * @param maxDecals  Maximum concurrent decals the SSBO can hold.  Default 64.
   */
  static async create(
    device: GPUDevice,
    format: GPUTextureFormat,
    maxDecals = 64,
  ): Promise<DecalGPUPipeline> {
    const module = device.createShaderModule({
      label: 'decal-composite-shader',
      code: DECAL_COMPOSITE_WGSL,
    });

    const bgl = device.createBindGroupLayout({
      label: 'decal-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: 'decal-composite-layout',
      bindGroupLayouts: [bgl],
    });

    const renderPipeline = device.createRenderPipeline({
      label: 'decal-composite-pipeline',
      layout: pipelineLayout,
      vertex:   { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const sampler = device.createSampler({
      label: 'decal-composite-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // Uniform buffer: resolution (vec2f) + decalCount (f32) + pad (f32) = 16 bytes
    const uniformBuffer = device.createBuffer({
      label: 'decal-uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Decal SSBO: 12 floats × 4 bytes × maxDecals
    const DECAL_STRIDE = 12 * 4;
    const decalBuffer = device.createBuffer({
      label: 'decal-ssbo',
      size: Math.max(DECAL_STRIDE * maxDecals, 48),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new DecalGPUPipeline(
      device, renderPipeline, bgl, sampler,
      uniformBuffer, decalBuffer, maxDecals,
    );
  }

  /**
   * Upload decal data to the GPU storage buffer.
   *
   * @param decals Array of DecalGPU snapshots from DecalProjector.getActiveDecals().
   */
  uploadDecals(decals: DecalGPU[]): void {
    const count = Math.min(decals.length, this.decalCapacity);
    const FLOATS_PER_DECAL = 12;
    const data = new Float32Array(count * FLOATS_PER_DECAL);

    for (let i = 0; i < count; i++) {
      const d = decals[i];
      const off = i * FLOATS_PER_DECAL;
      data[off +  0] = d.wx;
      data[off +  1] = d.wy;
      data[off +  2] = d.u;
      data[off +  3] = d.v;
      data[off +  4] = d.halfExtent;
      data[off +  5] = d.alpha;
      data[off +  6] = d.tx;
      data[off +  7] = d.ty;
      data[off +  8] = d.bx;
      data[off +  9] = d.by;
      data[off + 10] = d.typeIndex;
      data[off + 11] = d.seed;
    }

    this.device.queue.writeBuffer(this.decalBuffer, 0, data);
    this._currentCount = count;
  }

  private _currentCount = 0;

  /**
   * Record the decal compositing render pass.
   *
   * Reads from `srcView` (scene colour after tone-mapping) and writes
   * the decal-blended result to `dstView`.  Typically inserted in the
   * post-process chain between tone-mapping and final blit.
   *
   * @param encoder  GPUCommandEncoder for the current frame.
   * @param srcView  Scene colour texture view (input).
   * @param dstView  Destination render target view (output).
   * @param width    Render target width in pixels.
   * @param height   Render target height in pixels.
   */
  render(
    encoder: GPUCommandEncoder,
    srcView: GPUTextureView,
    dstView: GPUTextureView,
    width:   number,
    height:  number,
  ): void {
    if (this._currentCount === 0) return;

    // Update uniforms
    const uData = new Float32Array([width, height, this._currentCount, 0]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uData);

    const bindGroup = this.device.createBindGroup({
      label: 'decal-composite-bg',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
        { binding: 3, resource: { buffer: this.decalBuffer } },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: 'decal-composite-pass',
      colorAttachments: [{
        view: dstView,
        loadOp: 'load',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);  // fullscreen triangle
    pass.end();
  }

  /** Release GPU resources. */
  destroy(): void {
    this.uniformBuffer.destroy();
    this.decalBuffer.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: impulse estimation (mirrors contact-sparks / collision-fx pattern)
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
 * @returns Impulse proxy suitable for passing to `DecalProjector.stamp`.
 */
export function estimateDecalImpulse(
  contact: CollisionContactInfo,
  relSpeed = 0,
  massScale = 1,
): number {
  return (contact.depth * 80 + relSpeed * 0.5) * massScale;
}
