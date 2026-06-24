/**
 * magnetic-field-lines.ts — M782
 *
 * Magnetic Field Line Visualization — inter-Cell force fields rendered as
 * continuous streamlines with Runge-Kutta integration and directional arrows.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Visual metaphor:
 *
 *   ╔═══════════════════════════════════════════════════════════════════════╗
 *   ║  [Cell A]                                            [Cell B]       ║
 *   ║     ╲  ╲   ╲                                ╱   ╱  ╱               ║
 *   ║      ╲──╲───╲──── field lines ─────────────╱───╱──╱                ║
 *   ║       ╲  ╲   ╲    ───▸ ───▸ ───▸         ╱   ╱  ╱                 ║
 *   ║        ╲  ╲   ╲                         ╱   ╱  ╱                   ║
 *   ║         ⊕ repulse                     attract ⊖                    ║
 *   ╚═══════════════════════════════════════════════════════════════════════╝
 *
 *   1. **Force field sampling** — At each point in the 2-D domain, the net
 *      force vector F(x, y) is the superposition of all Cell contributions.
 *      Each Cell emits a radial field — attractive (−) or repulsive (+) —
 *      scaled by its species-defined charge magnitude. The field falls off
 *      as 1/r² (Coulomb-like) with a softening parameter ε to avoid the
 *      singularity at r = 0.
 *
 *   2. **Runge-Kutta 4 streamline integration** — From uniformly seeded
 *      start points, the RK4 integrator traces the normalised force field
 *      direction forward (and optionally backward) to produce smooth
 *      streamline polylines.  Integration halts when:
 *        • the line exits the domain boundary
 *        • the field magnitude drops below a threshold (dead zone)
 *        • the maximum number of integration steps is reached
 *        • the line approaches too close to a source cell (sink)
 *
 *   3. **Line rendering** — Canvas2D polyline strokes with:
 *        • width proportional to local field strength
 *        • alpha/saturation proportional to field intensity
 *        • smooth Catmull-Rom interpolation between RK4 sample points
 *        • per-segment gradient colouring (QoS-aware palette)
 *
 *   4. **Arrow markers** — Triangular arrowheads placed at regular arc-length
 *      intervals along each streamline, oriented tangent to the field
 *      direction.  Size and opacity scale with field strength.
 *
 *   5. **Field-driven density** — Seed point count is proportional to the
 *      integrated field magnitude in each grid cell: stronger regions spawn
 *      denser line bundles; quiet regions remain sparse.  This mimics the
 *      physical convention where field line density encodes strength.
 *
 * Cell charge model:
 *
 *   Each Cell acts as a point charge in a 2-D electrostatic analogy.
 *   The species determines the "charge sign":
 *
 *     Attractive (sink)    — cil-eye, cil-bolt      (attention, residual)
 *     Repulsive (source)   — cil-vector, cil-plus   (output, add)
 *     Neutral (weak dipole) — cil-arrow-right, etc.  (pass-through)
 *
 *   The charge magnitude is derived from the Cell's physical mass (via
 *   cell-body-bridge.ts CellPhysicsConfig).  Heavier cells produce stronger
 *   fields.  QoS traffic data can modulate charge dynamically: a burst on
 *   an edge temporarily intensifies the field between its endpoints.
 *
 * Coordinate convention:
 *
 *   World space: (0, 0) at top-left, X right, Y down (matching SPH domain).
 *   All positions, velocities, and field vectors use this convention.
 *
 * Integration:
 *
 *   ```ts
 *   import { MagneticFieldLines } from '$lib/sph/magnetic-field-lines';
 *
 *   const field = new MagneticFieldLines(ctx2d, {
 *     domainW: 1200, domainH: 800,
 *     cells: cellPhysicsConfigs,
 *   });
 *
 *   // Optionally modulate a cell's charge in real-time:
 *   field.setCharge('self_attn', -2.5);
 *
 *   // Render loop:
 *   field.update(elapsed, dt);
 *   field.draw();
 *
 *   // Cleanup:
 *   field.destroy();
 *   ```
 *
 * References:
 *   src/lib/sph/cell-body-bridge.ts        — CellPhysicsConfig source
 *   src/lib/sph/cell-interaction-physics.ts — force field conventions
 *   src/lib/sph/curl-flow-field.ts         — field sampling architecture
 *   src/lib/sph/edge-flow-renderer.ts      — spline rendering reference
 *   src/lib/sph/color-palette.ts           — QoS → colour theme mapping
 *   src/lib/sph/qosSpatial.ts              — QoS profile definitions
 *
 * Research: xiaodi #M782 — cell-pubsub-loop
 */

import type { QoSProfileName }    from './qosSpatial';
import { QOS_THEME }               from './color-palette';
import type { ThemePalette, RGB }  from './color-palette';

// ─── Constants ────────────────────────────────────────────────────────────────

// [orphan-precise] /** Maximum streamlines across the entire field. */
const MAX_STREAMLINES = 512;

/** Maximum integration steps per streamline (RK4). */
const MAX_RK4_STEPS = 256;

/** Integration step size (world units). */
const DEFAULT_STEP_SIZE = 3.0;

/** Minimum field magnitude below which integration halts (dead zone). */
const FIELD_DEAD_ZONE = 1e-5;

/** Softening parameter ε² to avoid 1/r² singularity at charge centres. */
const SOFTENING_SQ = 64.0; // ε = 8 world units

/** Minimum approach distance to a source cell before halting. */
const SINK_CAPTURE_RADIUS = 16.0;

/** Arrow marker spacing (arc-length units between consecutive arrows). */
const ARROW_SPACING = 40.0;

/** Arrow head half-width (perpendicular to tangent). */
const ARROW_HALF_W = 4.0;

/** Arrow head length (along tangent). */
const ARROW_LENGTH = 8.0;

/** Maximum line width at peak field strength. */
const MAX_LINE_WIDTH = 3.5;

/** Minimum line width in weak-field regions. */
const MIN_LINE_WIDTH = 0.4;

/** Seed grid resolution for adaptive density seeding. */
const SEED_GRID_RES = 24;

/** Maximum seed attempts per grid cell. */
const MAX_SEEDS_PER_CELL = 4;

/** Field strength threshold for spawning a seed (normalised). */
const SEED_STRENGTH_THRESHOLD = 0.05;

/** Smoothing window for Catmull-Rom interpolation subdivisions. */
const CR_SUBDIVS = 4;

/** Catmull-Rom tension parameter (0.5 = centripetal). */
const CR_TENSION = 0.5;

/** Breathing animation period for idle field lines (seconds). */
const BREATHING_PERIOD = 4.0;

/** Minimum alpha for field lines (even in weak-field regions). */
const MIN_ALPHA = 0.08;

/** Maximum alpha for field lines at peak strength. */
const MAX_ALPHA = 0.85;

// ─── Charge sign by species ──────────────────────────────────────────────────

const SPECIES_CHARGE_SIGN: Record<string, number> = {
  'cil-eye':         -1.0,   // attention — attractive sink
  'cil-bolt':        -1.0,   // residual — attractive sink
  'cil-vector':      +1.0,   // output — repulsive source
  'cil-plus':        +1.0,   // add — repulsive source
  'cil-arrow-right': +0.3,   // pass-through — weak source
  'cil-filter':      -0.5,   // filter — moderate sink
  'cil-layers':      -0.3,   // layers — weak sink
  'cil-loop':        +0.5,   // loop — moderate source
  'cil-code':        +0.2,   // code — weak source
  'cil-graph':       -0.4,   // graph — moderate sink
};

/** Default charge sign for unknown species. */
const DEFAULT_CHARGE_SIGN = 0.0;

// ─── QoS → field-line colour mapping ────────────────────────────────────────

interface FieldLineStyle {
  /** Base RGB for streamline stroke. */
  color: RGB;
  /** Glow colour for additive overdraw. */
  glow: RGB;
  /** Intensity multiplier (scales alpha). */
  intensity: number;
}

const QOS_FIELD_STYLES: Record<QoSProfileName, FieldLineStyle> = {
  SENSOR_DATA: {
    color:     { r: 0.20, g: 0.55, b: 0.95 },
    glow:      { r: 0.40, g: 0.75, b: 1.00 },
    intensity: 1.2,
  },
  PARAMETERS: {
    color:     { r: 0.90, g: 0.60, b: 0.15 },
    glow:      { r: 1.00, g: 0.80, b: 0.40 },
    intensity: 1.0,
  },
  TF_STATIC: {
    color:     { r: 0.10, g: 0.70, b: 0.45 },
    glow:      { r: 0.50, g: 1.00, b: 0.70 },
    intensity: 0.8,
  },
  TOPO_CHANGE: {
    color:     { r: 0.85, g: 0.15, b: 0.60 },
    glow:      { r: 1.00, g: 0.50, b: 0.85 },
    intensity: 1.5,
  },
  DEFAULT: {
    color:     { r: 0.50, g: 0.55, b: 0.65 },
    glow:      { r: 0.70, g: 0.75, b: 0.85 },
    intensity: 0.9,
  },
};

// ─── Public types ────────────────────────────────────────────────────────────

/** A single point charge in the 2-D domain (one per Cell). */
export interface FieldCharge {
  /** Cell identifier (matches CellPhysicsConfig.id). */
  id: string;
  /** Centre position X (world units). */
  x: number;
  /** Centre position Y (world units). */
  y: number;
  /** Charge magnitude (sign determines attract/repel). */
  charge: number;
  /** Species string for default charge-sign look-up. */
  species: string;
  /** Physical mass (modulates field strength). */
  mass: number;
  /** Cell half-width (used for capture radius scaling). */
  hw: number;
  /** Cell half-height (used for capture radius scaling). */
  hh: number;
}

/** Minimal cell descriptor accepted by the constructor. */
export interface FieldCellInput {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  species: string;
  mass: number;
}

/** Configuration for the MagneticFieldLines renderer. */
export interface MagneticFieldLinesConfig {
  /** Domain width (world units). */
  domainW: number;
  /** Domain height (world units). */
  domainH: number;
  /** Cell descriptors (position, species, mass). */
  cells: FieldCellInput[];
  /** QoS profile name for colour theming. Default: 'DEFAULT'. */
  qos?: QoSProfileName;
  /** Integration step size override. Default: 3.0. */
  stepSize?: number;
  /** Maximum streamlines. Default: 512. */
  maxLines?: number;
  /** Whether to integrate backward as well as forward. Default: true. */
  bidirectional?: boolean;
  /** Arrow spacing along streamlines (arc-length units). Default: 40. */
  arrowSpacing?: number;
  /** Global field strength multiplier. Default: 1.0. */
  strengthScale?: number;
  /** Enable breathing animation on idle lines. Default: true. */
  breathing?: boolean;
}

// ─── Internal types ─────────────────────────────────────────────────────────

/** A single point on a streamline polyline. */
interface StreamPoint {
  x: number;
  y: number;
  /** Field magnitude at this point. */
  mag: number;
}

/** A fully traced streamline. */
interface Streamline {
  /** Ordered points from backward tail → seed → forward head. */
  points: StreamPoint[];
  /** Total arc length of the streamline. */
  arcLength: number;
  /** Peak field magnitude encountered along this line. */
  peakMag: number;
  /** Seed grid cell index (for density tracking). */
  seedCell: number;
}

/** 2-D vector tuple for internal computations. */
interface Vec2 {
  x: number;
  y: number;
}

// ─── Utility functions ──────────────────────────────────────────────────────

/** Clamp a value to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Smooth-step (Hermite interpolation). */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

/** Convert linear RGB [0,1] to CSS rgba string. */
function rgbaCSS(r: number, g: number, b: number, a: number): string {
  return `rgba(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0},${a.toFixed(3)})`;
}

// ─── Core: Force Field Evaluator ────────────────────────────────────────────

/**
 * Evaluate the net force vector F(x, y) from all charges using a
 * 2-D Coulomb-like model with softening:
 *
 *   F(x) = Σᵢ  qᵢ · (x − xᵢ) / (|x − xᵢ|² + ε²)^(3/2)
 *
 * The result is NOT normalised — its magnitude encodes field strength.
 */
function evaluateField(
  px: number,
  py: number,
  charges: FieldCharge[],
  strengthScale: number,
): Vec2 {
  let fx = 0.0;
  let fy = 0.0;

  for (let i = 0; i < charges.length; i++) {
    const c = charges[i];
    const dx = px - c.x;
    const dy = py - c.y;
    const r2 = dx * dx + dy * dy + SOFTENING_SQ;
    // Coulomb 2-D: F ∝ q / r² , direction = r̂
    // With softening: F = q · r⃗ / (r² + ε²)^(3/2)
    const invR3 = c.charge / (r2 * Math.sqrt(r2));
    fx += dx * invR3;
    fy += dy * invR3;
  }

  fx *= strengthScale;
  fy *= strengthScale;

  return { x: fx, y: fy };
}

/**
 * Evaluate the field magnitude at a point (cheap: no sqrt on sum).
 */
function fieldMagnitude(px: number, py: number, charges: FieldCharge[], strengthScale: number): number {
  const f = evaluateField(px, py, charges, strengthScale);
  return Math.sqrt(f.x * f.x + f.y * f.y);
}

// ─── Core: RK4 Streamline Integrator ────────────────────────────────────────

/**
 * Trace a single streamline from a seed point using classic 4th-order
 * Runge-Kutta integration of the normalised field direction.
 *
 * The integrator advances along d/ds(x(s)) = F̂(x) where F̂ is the unit
 * vector in the force direction.  Step size h is in arc-length (world units).
 *
 * Returns an array of StreamPoints with position and field magnitude at
 * each sample.
 */
function traceStreamline(
  seedX: number,
  seedY: number,
  charges: FieldCharge[],
  stepSize: number,
  maxSteps: number,
  domainW: number,
  domainH: number,
  direction: 1 | -1,
  strengthScale: number,
): StreamPoint[] {
  const points: StreamPoint[] = [];
  let x = seedX;
  let y = seedY;

  for (let step = 0; step < maxSteps; step++) {
    // Evaluate field at current position
    const f0 = evaluateField(x, y, charges, strengthScale);
    const mag0 = Math.sqrt(f0.x * f0.x + f0.y * f0.y);

    // Dead zone check
    if (mag0 < FIELD_DEAD_ZONE) break;

    points.push({ x, y, mag: mag0 });

    // Normalised direction (with sign for forward/backward)
    const invMag0 = direction / mag0;
    const dx0 = f0.x * invMag0;
    const dy0 = f0.y * invMag0;

    // ── RK4 stages ──

    // k1
    const k1x = dx0 * stepSize;
    const k1y = dy0 * stepSize;

    // k2: evaluate at midpoint using k1
    const mx1 = x + k1x * 0.5;
    const my1 = y + k1y * 0.5;
    const f1 = evaluateField(mx1, my1, charges, strengthScale);
    const mag1 = Math.sqrt(f1.x * f1.x + f1.y * f1.y);
    if (mag1 < FIELD_DEAD_ZONE) break;
    const invMag1 = direction / mag1;
    const k2x = f1.x * invMag1 * stepSize;
    const k2y = f1.y * invMag1 * stepSize;

    // k3: evaluate at midpoint using k2
    const mx2 = x + k2x * 0.5;
    const my2 = y + k2y * 0.5;
    const f2 = evaluateField(mx2, my2, charges, strengthScale);
    const mag2 = Math.sqrt(f2.x * f2.x + f2.y * f2.y);
    if (mag2 < FIELD_DEAD_ZONE) break;
    const invMag2 = direction / mag2;
    const k3x = f2.x * invMag2 * stepSize;
    const k3y = f2.y * invMag2 * stepSize;

    // k4: evaluate at endpoint using k3
    const mx3 = x + k3x;
    const my3 = y + k3y;
    const f3 = evaluateField(mx3, my3, charges, strengthScale);
    const mag3 = Math.sqrt(f3.x * f3.x + f3.y * f3.y);
    if (mag3 < FIELD_DEAD_ZONE) break;
    const invMag3 = direction / mag3;
    const k4x = f3.x * invMag3 * stepSize;
    const k4y = f3.y * invMag3 * stepSize;

    // RK4 weighted average
    x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
    y += (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;

    // ── Termination checks ──

    // Domain boundary exit (with small margin)
    if (x < -10 || x > domainW + 10 || y < -10 || y > domainH + 10) {
      points.push({ x, y, mag: mag0 });
      break;
    }

    // Sink capture: halt if too close to any charge centre
    let captured = false;
    for (let i = 0; i < charges.length; i++) {
      const c = charges[i];
      const cdx = x - c.x;
      const cdy = y - c.y;
      // Capture radius scales with cell size
      const cr = Math.max(SINK_CAPTURE_RADIUS, Math.max(c.hw, c.hh) * 0.8);
      if (cdx * cdx + cdy * cdy < cr * cr) {
        captured = true;
        break;
      }
    }
    if (captured) {
      points.push({ x, y, mag: mag0 });
      break;
    }
  }

  return points;
}

// ─── Core: Adaptive Seed Generation ─────────────────────────────────────────

/**
 * Generate streamline seed points with density proportional to the local
 * field strength.  The domain is divided into a uniform grid; each cell
 * samples the field at its centre and spawns 0–MAX_SEEDS_PER_CELL seeds
 * with count ∝ field magnitude.
 *
 * An exclusion zone around each charge centre prevents seeds from spawning
 * too close to singularities.
 */
function generateSeeds(
  charges: FieldCharge[],
  domainW: number,
  domainH: number,
  maxLines: number,
  strengthScale: number,
): Array<{ x: number; y: number; cellIdx: number }> {
  const cellW = domainW / SEED_GRID_RES;
  const cellH = domainH / SEED_GRID_RES;

  // First pass: sample field magnitude at each grid cell centre
  const gridMag = new Float32Array(SEED_GRID_RES * SEED_GRID_RES);
  let maxMag = 0.0;

  for (let gy = 0; gy < SEED_GRID_RES; gy++) {
    for (let gx = 0; gx < SEED_GRID_RES; gx++) {
      const cx = (gx + 0.5) * cellW;
      const cy = (gy + 0.5) * cellH;
      const mag = fieldMagnitude(cx, cy, charges, strengthScale);
      const idx = gy * SEED_GRID_RES + gx;
      gridMag[idx] = mag;
      if (mag > maxMag) maxMag = mag;
    }
  }

  if (maxMag < FIELD_DEAD_ZONE) return [];

  // Second pass: spawn seeds proportional to normalised magnitude
  const invMax = 1.0 / maxMag;
  const seeds: Array<{ x: number; y: number; cellIdx: number }> = [];

  // Deterministic pseudo-random: use a simple LCG seeded per grid cell
  let rngState = 0x9E3779B9;
  function nextRng(): number {
    rngState = (rngState * 1664525 + 1013904223) & 0x7FFFFFFF;
    return rngState / 0x7FFFFFFF;
  }

  for (let gy = 0; gy < SEED_GRID_RES; gy++) {
    for (let gx = 0; gx < SEED_GRID_RES; gx++) {
      const idx = gy * SEED_GRID_RES + gx;
      const normMag = gridMag[idx] * invMax;

      if (normMag < SEED_STRENGTH_THRESHOLD) continue;

      // Number of seeds: proportional to field strength, capped per cell
      const seedCount = Math.min(
        MAX_SEEDS_PER_CELL,
        Math.ceil(normMag * MAX_SEEDS_PER_CELL),
      );

      for (let s = 0; s < seedCount; s++) {
        if (seeds.length >= maxLines) return seeds;

        // Jittered position within grid cell
        const sx = (gx + nextRng()) * cellW;
        const sy = (gy + nextRng()) * cellH;

        // Exclusion zone: skip if too close to any charge
        let excluded = false;
        for (let ci = 0; ci < charges.length; ci++) {
          const c = charges[ci];
          const ddx = sx - c.x;
          const ddy = sy - c.y;
          const exclR = Math.max(c.hw, c.hh) * 1.2;
          if (ddx * ddx + ddy * ddy < exclR * exclR) {
            excluded = true;
            break;
          }
        }
        if (excluded) continue;

        seeds.push({ x: sx, y: sy, cellIdx: idx });
      }
    }
  }

  return seeds;
}

// ─── Core: Catmull-Rom Interpolation ────────────────────────────────────────

/**
 * Evaluate a Catmull-Rom spline segment at parameter t ∈ [0, 1].
 * p0, p1 are the bracketing control points; pm1 and p2 are the outer
 * control points for tangent estimation.
 */
function catmullRom(
  pm1: number, p0: number, p1: number, p2: number,
  t: number, tension: number,
): number {
  const t2 = t * t;
  const t3 = t2 * t;
  const s = (1.0 - tension) * 0.5;

  const a = -s * pm1 + (2.0 - s) * p0 + (s - 2.0) * p1 + s * p2;
  const b = 2.0 * s * pm1 + (s - 3.0) * p0 + (3.0 - 2.0 * s) * p1 - s * p2;
  const c = -s * pm1 + s * p1;
  const d = p0;

  return a * t3 + b * t2 + c * t + d;
}

/**
 * Subdivide a StreamPoint array using Catmull-Rom interpolation,
 * inserting CR_SUBDIVS intermediate points between each pair.
 * Also interpolates the field magnitude for each subdivided point.
 */
function subdivideStreamline(pts: StreamPoint[]): StreamPoint[] {
  if (pts.length < 2) return pts;

  const out: StreamPoint[] = [];
  const n = pts.length;

  for (let i = 0; i < n - 1; i++) {
    // Clamped control point indices for Catmull-Rom
    const im1 = Math.max(0, i - 1);
    const ip2 = Math.min(n - 1, i + 2);

    out.push(pts[i]);

    for (let s = 1; s < CR_SUBDIVS; s++) {
      const t = s / CR_SUBDIVS;
      out.push({
        x:   catmullRom(pts[im1].x,   pts[i].x,   pts[i + 1].x,   pts[ip2].x,   t, CR_TENSION),
        y:   catmullRom(pts[im1].y,   pts[i].y,   pts[i + 1].y,   pts[ip2].y,   t, CR_TENSION),
        mag: catmullRom(pts[im1].mag, pts[i].mag, pts[i + 1].mag, pts[ip2].mag, t, CR_TENSION),
      });
    }
  }
  out.push(pts[n - 1]);

  return out;
}

// ─── Core: Arc-Length Computation ───────────────────────────────────────────

/** Compute cumulative arc-length array for a polyline. */
function computeArcLengths(pts: StreamPoint[]): Float32Array {
  const arcLen = new Float32Array(pts.length);
  arcLen[0] = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dy = pts[i].y - pts[i - 1].y;
    arcLen[i] = arcLen[i - 1] + Math.sqrt(dx * dx + dy * dy);
  }
  return arcLen;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Public API: MagneticFieldLines
// ═════════════════════════════════════════════════════════════════════════════

export class MagneticFieldLines {
  // ── Config ──
  private readonly ctx: CanvasRenderingContext2D;
  private readonly domainW: number;
  private readonly domainH: number;
  private readonly stepSize: number;
  private readonly maxLines: number;
  private readonly bidirectional: boolean;
  private readonly arrowSpacing: number;
  private readonly strengthScale: number;
  private readonly breathingEnabled: boolean;
  private readonly qosProfile: QoSProfileName;
  private readonly style: FieldLineStyle;

  // ── State ──
  private charges: FieldCharge[] = [];
  private streamlines: Streamline[] = [];
  private dirty = true;             // true → recompute streamlines on next update
  private elapsed = 0.0;            // accumulated elapsed time for animation
  private peakFieldMag = 1.0;       // normalisation reference for rendering

  // ── Reusable scratch buffers ──
  private readonly _scratchForward: StreamPoint[] = [];
  private readonly _scratchBackward: StreamPoint[] = [];

  // ─────────────────────────────────────────────────────────────────────────
  //  Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(ctx: CanvasRenderingContext2D, config: MagneticFieldLinesConfig) {
    this.ctx           = ctx;
    this.domainW       = config.domainW;
    this.domainH       = config.domainH;
    this.stepSize      = config.stepSize      ?? DEFAULT_STEP_SIZE;
    this.maxLines      = Math.min(config.maxLines ?? MAX_STREAMLINES, MAX_STREAMLINES);
    this.bidirectional = config.bidirectional  ?? true;
    this.arrowSpacing  = config.arrowSpacing   ?? ARROW_SPACING;
    this.strengthScale = config.strengthScale  ?? 1.0;
    this.breathingEnabled = config.breathing   ?? true;
    this.qosProfile    = config.qos           ?? 'DEFAULT';
    this.style         = QOS_FIELD_STYLES[this.qosProfile] ?? QOS_FIELD_STYLES.DEFAULT;

    // Initialise charges from cell descriptors
    this.setCells(config.cells);
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  Public API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Replace all cells and rebuild the charge array.
   * Marks the streamline cache dirty for recomputation.
   */
  setCells(cells: FieldCellInput[]): void {
    this.charges = cells.map((c) => {
      const sign = SPECIES_CHARGE_SIGN[c.species] ?? DEFAULT_CHARGE_SIGN;
      return {
        id:      c.id,
        x:       c.x,
        y:       c.y,
        charge:  sign * c.mass * 0.01, // Scale mass → charge magnitude
        species: c.species,
        mass:    c.mass,
        hw:      c.w * 0.5,
        hh:      c.h * 0.5,
      };
    });
    this.dirty = true;
  }

  /**
   * Update the position of a single cell (e.g. after drag).
   * Marks dirty.
   */
  setCellPosition(cellId: string, x: number, y: number): void {
    for (let i = 0; i < this.charges.length; i++) {
      if (this.charges[i].id === cellId) {
        this.charges[i].x = x;
        this.charges[i].y = y;
        this.dirty = true;
        return;
      }
    }
  }

  /**
   * Override the charge value for a specific cell.
   * Positive = repulsive (source), negative = attractive (sink).
   */
  setCharge(cellId: string, charge: number): void {
    for (let i = 0; i < this.charges.length; i++) {
      if (this.charges[i].id === cellId) {
        this.charges[i].charge = charge;
        this.dirty = true;
        return;
      }
    }
  }

  /**
   * Sample the force field at an arbitrary point (for external consumers).
   * Returns { x, y } force vector and magnitude.
   */
  sampleField(px: number, py: number): { x: number; y: number; mag: number } {
    const f = evaluateField(px, py, this.charges, this.strengthScale);
    return { x: f.x, y: f.y, mag: Math.sqrt(f.x * f.x + f.y * f.y) };
  }

  /**
   * Force recomputation of all streamlines on the next update().
   */
  invalidate(): void {
    this.dirty = true;
  }

  /**
   * Update the field line state.  Recomputes streamlines when dirty.
   *
   * @param _elapsed  Total elapsed time in seconds (for animation).
   * @param _dt       Frame delta time (currently unused; reserved for
   *                  future animated line advection).
   */
  update(_elapsed: number, _dt: number): void {
    this.elapsed = _elapsed;

    if (!this.dirty) return;
    this.dirty = false;

    this.rebuildStreamlines();
  }

  /**
   * Draw all streamlines onto the Canvas2D context.
   * Call after update() in the render loop.
   */
  draw(): void {
    const ctx = this.ctx;
    const lines = this.streamlines;
    if (lines.length === 0) return;

    // Breathing animation modulates alpha
    const breathAlpha = this.breathingEnabled
      ? 0.85 + 0.15 * Math.sin(this.elapsed * (2.0 * Math.PI / BREATHING_PERIOD))
      : 1.0;

    const invPeak = this.peakFieldMag > FIELD_DEAD_ZONE ? 1.0 / this.peakFieldMag : 1.0;
    const style = this.style;

    ctx.save();
    ctx.lineCap  = 'round';
    ctx.lineJoin = 'round';

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      const pts = line.points;
      if (pts.length < 2) continue;

      // Draw the streamline as a series of segments with varying width/alpha
      this.drawStreamlineSegments(ctx, pts, invPeak, breathAlpha, style);

      // Draw arrow markers along the streamline
      this.drawArrowMarkers(ctx, pts, line.arcLength, invPeak, breathAlpha, style);
    }

    ctx.restore();
  }

  /**
   * Release resources (no-op for Canvas2D, included for API consistency
   * with GPU-backed renderers).
   */
  destroy(): void {
    this.streamlines = [];
    this.charges = [];
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  Internal: Streamline Rebuilding
  // ─────────────────────────────────────────────────────────────────────────

  private rebuildStreamlines(): void {
    this.streamlines = [];

    if (this.charges.length === 0) return;

    // Generate adaptive seed points
    const seeds = generateSeeds(
      this.charges,
      this.domainW,
      this.domainH,
      this.maxLines,
      this.strengthScale,
    );

    // Track peak field magnitude for normalisation
    let peakMag = 0.0;

    for (let si = 0; si < seeds.length; si++) {
      const seed = seeds[si];

      // Forward trace
      const forward = traceStreamline(
        seed.x, seed.y,
        this.charges,
        this.stepSize,
        MAX_RK4_STEPS,
        this.domainW, this.domainH,
        1,
        this.strengthScale,
      );

      // Backward trace (if bidirectional)
      let backward: StreamPoint[] = [];
      if (this.bidirectional) {
        backward = traceStreamline(
          seed.x, seed.y,
          this.charges,
          this.stepSize,
          MAX_RK4_STEPS,
          this.domainW, this.domainH,
          -1,
          this.strengthScale,
        );
      }

      // Merge: backward (reversed) + forward
      let merged: StreamPoint[];
      if (backward.length > 1) {
        // Reverse backward (it goes seed → far end, we want far end → seed)
        backward.reverse();
        // Remove the duplicate seed point
        backward.pop();
        merged = backward.concat(forward);
      } else {
        merged = forward;
      }

      if (merged.length < 3) continue; // Too short to be meaningful

      // Subdivide via Catmull-Rom for smooth rendering
      const smooth = subdivideStreamline(merged);

      // Compute arc length
      const arcLens = computeArcLengths(smooth);
      const totalArc = arcLens[arcLens.length - 1];
      if (totalArc < this.stepSize * 2) continue; // Degenerate

      // Find peak magnitude along this line
      let linePeak = 0.0;
      for (let i = 0; i < smooth.length; i++) {
        if (smooth[i].mag > linePeak) linePeak = smooth[i].mag;
      }
      if (linePeak > peakMag) peakMag = linePeak;

      this.streamlines.push({
        points:    smooth,
        arcLength: totalArc,
        peakMag:   linePeak,
        seedCell:  seed.cellIdx,
      });
    }

    this.peakFieldMag = peakMag > FIELD_DEAD_ZONE ? peakMag : 1.0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  Internal: Segment Rendering
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Draw streamline segments with per-segment width and alpha driven by
   * local field magnitude.  Uses Canvas2D sub-paths for efficiency.
   */
  private drawStreamlineSegments(
    ctx: CanvasRenderingContext2D,
    pts: StreamPoint[],
    invPeak: number,
    breathAlpha: number,
    style: FieldLineStyle,
  ): void {
    // Batch segments into groups of similar width for fewer stroke calls.
    // For simplicity and visual quality, we draw each segment individually
    // with its own width/alpha — the segment count is bounded by
    // MAX_RK4_STEPS * CR_SUBDIVS which is manageable.

    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i];
      const p1 = pts[i + 1];

      // Average magnitude for this segment
      const avgMag = (p0.mag + p1.mag) * 0.5;
      const normMag = clamp(avgMag * invPeak, 0.0, 1.0);

      // Width: lerp from MIN to MAX based on normalised magnitude
      const width = lerp(MIN_LINE_WIDTH, MAX_LINE_WIDTH, normMag);

      // Alpha: smoothstep with intensity scaling
      const alpha = clamp(
        lerp(MIN_ALPHA, MAX_ALPHA, smoothstep(0.0, 0.6, normMag)) *
        style.intensity * breathAlpha,
        0.0, 1.0,
      );

      // Colour: blend between base and glow based on magnitude
      const glowMix = smoothstep(0.3, 0.9, normMag);
      const r = lerp(style.color.r, style.glow.r, glowMix);
      const g = lerp(style.color.g, style.glow.g, glowMix);
      const b = lerp(style.color.b, style.glow.b, glowMix);

      ctx.strokeStyle = rgbaCSS(r, g, b, alpha);
      ctx.lineWidth   = width;

      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }

    // ── Glow pass: additive composite for high-magnitude regions ──
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i];
      const p1 = pts[i + 1];
      const avgMag = (p0.mag + p1.mag) * 0.5;
      const normMag = clamp(avgMag * invPeak, 0.0, 1.0);

      // Only glow on strong segments
      if (normMag < 0.4) continue;

      const glowAlpha = clamp(
        (normMag - 0.4) * 1.2 * style.intensity * breathAlpha * 0.3,
        0.0, 0.35,
      );
      const width = lerp(MIN_LINE_WIDTH, MAX_LINE_WIDTH, normMag) * 2.5;

      ctx.strokeStyle = rgbaCSS(style.glow.r, style.glow.g, style.glow.b, glowAlpha);
      ctx.lineWidth   = width;

      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  Internal: Arrow Marker Rendering
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Draw triangular arrowheads at regular arc-length intervals along a
   * streamline.  Each arrow is oriented tangent to the local field direction
   * and sized proportionally to field strength.
   */
  private drawArrowMarkers(
    ctx: CanvasRenderingContext2D,
    pts: StreamPoint[],
    totalArc: number,
    invPeak: number,
    breathAlpha: number,
    style: FieldLineStyle,
  ): void {
    if (totalArc < this.arrowSpacing * 0.5) return;

    const arcLens = computeArcLengths(pts);
    const spacing = this.arrowSpacing;

    // Place first arrow at half-spacing offset to avoid crowding at endpoints
    let nextArrowAt = spacing * 0.5;
    let ptIdx = 0;

    while (nextArrowAt < totalArc - spacing * 0.25) {
      // Advance ptIdx to the segment containing nextArrowAt
      while (ptIdx < pts.length - 2 && arcLens[ptIdx + 1] < nextArrowAt) {
        ptIdx++;
      }

      if (ptIdx >= pts.length - 1) break;

      // Interpolate position within the segment
      const segStart = arcLens[ptIdx];
      const segEnd   = arcLens[ptIdx + 1];
      const segLen   = segEnd - segStart;
      const t = segLen > 1e-6 ? (nextArrowAt - segStart) / segLen : 0.0;

      const ax = lerp(pts[ptIdx].x, pts[ptIdx + 1].x, t);
      const ay = lerp(pts[ptIdx].y, pts[ptIdx + 1].y, t);
      const aMag = lerp(pts[ptIdx].mag, pts[ptIdx + 1].mag, t);

      // Tangent direction (from segment)
      const tdx = pts[ptIdx + 1].x - pts[ptIdx].x;
      const tdy = pts[ptIdx + 1].y - pts[ptIdx].y;
      const tLen = Math.sqrt(tdx * tdx + tdy * tdy);
      if (tLen < 1e-6) {
        nextArrowAt += spacing;
        continue;
      }
      const tx = tdx / tLen; // tangent unit x
      const ty = tdy / tLen; // tangent unit y

      // Perpendicular (normal) for arrow wings
      const nx = -ty;
      const ny =  tx;

      // Arrow size scales with field magnitude
      const normMag = clamp(aMag * invPeak, 0.0, 1.0);
      const sizeMul = lerp(0.4, 1.0, normMag);
      const halfW = ARROW_HALF_W * sizeMul;
      const len   = ARROW_LENGTH * sizeMul;

      // Arrow alpha
      const alpha = clamp(
        lerp(MIN_ALPHA + 0.1, MAX_ALPHA, smoothstep(0.0, 0.5, normMag)) *
        style.intensity * breathAlpha,
        0.0, 1.0,
      );

      // Arrow colour (slightly brighter than line)
      const glowMix = smoothstep(0.2, 0.7, normMag);
      const r = lerp(style.color.r, style.glow.r, glowMix * 1.2);
      const g = lerp(style.color.g, style.glow.g, glowMix * 1.2);
      const b = lerp(style.color.b, style.glow.b, glowMix * 1.2);

      // Triangle vertices: tip at front, two wings at back
      const tipX  = ax + tx * len * 0.5;
      const tipY  = ay + ty * len * 0.5;
      const backX = ax - tx * len * 0.5;
      const backY = ay - ty * len * 0.5;

      const wingLX = backX + nx * halfW;
      const wingLY = backY + ny * halfW;
      const wingRX = backX - nx * halfW;
      const wingRY = backY - ny * halfW;

      ctx.fillStyle = rgbaCSS(
        clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1), alpha,
      );

      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(wingLX, wingLY);
      ctx.lineTo(wingRX, wingRY);
      ctx.closePath();
      ctx.fill();

      nextArrowAt += spacing;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  Static Factory
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Convenience factory — creates a MagneticFieldLines instance from a
   * Canvas2D context and configuration object.
   */
  static create(
    ctx: CanvasRenderingContext2D,
    config: MagneticFieldLinesConfig,
  ): MagneticFieldLines {
    return new MagneticFieldLines(ctx, config);
  }
}

// ─── Barrel exports ─────────────────────────────────────────────────────────

export {
  evaluateField,
  fieldMagnitude,
  traceStreamline,
  generateSeeds,
  subdivideStreamline,
  computeArcLengths,
  catmullRom,
  SPECIES_CHARGE_SIGN,
  QOS_FIELD_STYLES,
};

export type {
  FieldLineStyle,
  StreamPoint,
  Streamline,
  Vec2 as FieldVec2,
};
